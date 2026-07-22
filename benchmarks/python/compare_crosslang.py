"""Compare cartan against Manifolds.jl, geomstats and geoopt.

Agreement first, speed second. A speed ratio between implementations that
disagree is not a result, so every timing row in the report is gated on the
corresponding value row having matched.

Reads the three `*_geometry.jsonl` files written by the per-language harnesses
and emits a dated Markdown report.

Run:
    .venv/bin/python python/compare_crosslang.py
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import platform
import subprocess

import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
RESULTS = ROOT / "results"

# Below this, two implementations are computing the same thing and differing
# only in floating-point association order. Above it, they disagree about the
# mathematics, or about a convention.
AGREE_TOL = 1e-12

SOURCES = {
    "cartan": RESULTS / "cartan_geometry.jsonl",
    "manifolds.jl": RESULTS / "julia_geometry.jsonl",
    # geomstats and geoopt share one file, split on the `lib` field.
    "_python": RESULTS / "python_geometry.jsonl",
}


def load() -> dict[tuple, dict]:
    """Index every record by (lib, manifold, dim, op)."""
    out: dict[tuple, dict] = {}
    for name, path in SOURCES.items():
        if not path.exists():
            print(f"missing {path}, skipping {name}")
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out[(r["lib"], r["manifold"], r["dim"], r["op"])] = r
    return out


def max_abs_diff(a: list[float], b: list[float]) -> float | None:
    """Largest absolute elementwise difference, or None if shapes differ."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        return None
    return float(np.max(np.abs(x - y)))


def provenance() -> dict:
    """Record what produced these numbers, so the report is not anonymous."""

    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            ).stdout.strip().splitlines()[0]
        except Exception:  # noqa: BLE001
            return "unavailable"

    cpu = "unknown"
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except Exception:  # noqa: BLE001
        pass

    import importlib.metadata as md

    def ver(pkg: str) -> str:
        try:
            return md.version(pkg)
        except Exception:  # noqa: BLE001
            return "absent"

    return {
        "cpu": cpu,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "rustc": run(["rustc", "--version"]),
        "julia": run(["julia", "--version"]),
        "geomstats": ver("geomstats"),
        "geoopt": ver("geoopt"),
        "numpy": ver("numpy"),
        "torch": ver("torch"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=ROOT / "CROSSLANG.md")
    ap.add_argument("--date", default="", help="date stamp for the report header")
    args = ap.parse_args()

    recs = load()
    comparators = ["manifolds.jl", "geomstats", "geoopt"]

    # `transport_projection` is a probe used to classify geoopt's transport,
    # not an operation cartan implements, so it never enters the main tables.
    keys = sorted({(m, d, o) for (lib, m, d, o) in recs if lib == "cartan"})

    # geoopt implements a projection-based vector transport rather than exact
    # parallel transport, which is the right choice for its purpose: an
    # optimiser needs a cheap transport, not an isometric one. That is a
    # convention difference, not a numerical disagreement, so it is reported
    # separately.
    #
    # The classification is verified rather than asserted. If geoopt's
    # transport ever stops equalling its own projection, this stops treating
    # the mismatch as expected and reports it as a real disagreement.
    convention: list[tuple] = []
    verified_projection = True
    for m, d, o in keys:
        if o != "transport":
            continue
        gt = recs.get(("geoopt", m, d, "transport"))
        gp = recs.get(("geoopt", m, d, "transport_projection"))
        if gt is None or gp is None:
            continue
        delta = max_abs_diff(gt["value"], gp["value"])
        if delta is None or delta > 1e-14:
            verified_projection = False
        else:
            base = recs[("cartan", m, d, o)]
            gap = max_abs_diff(base["value"], gt["value"])
            convention.append((m, d, gap, delta))

    agreements: list[tuple] = []
    worst = 0.0
    disagreements: list[tuple] = []

    for m, d, o in keys:
        base = recs[("cartan", m, d, o)]
        for c in comparators:
            other = recs.get((c, m, d, o))
            if other is None:
                continue
            # Skip the geoopt transport rows handled as a convention
            # difference above, so they are not double-counted as failures.
            if c == "geoopt" and o == "transport" and verified_projection:
                continue
            diff = max_abs_diff(base["value"], other["value"])
            if diff is None:
                disagreements.append((m, d, o, c, "shape mismatch"))
                continue
            agreements.append((m, d, o, c, diff))
            worst = max(worst, diff)
            if diff > AGREE_TOL:
                disagreements.append((m, d, o, c, f"{diff:.3e}"))

    # Speed, reported only where the values matched.
    matched = {(m, d, o, c) for (m, d, o, c, v) in agreements
               if isinstance(v, float) and v <= AGREE_TOL}

    speed: dict[str, list[float]] = collections.defaultdict(list)
    rows = []
    for m, d, o in keys:
        base = recs[("cartan", m, d, o)]
        row = {"case": f"{m} {d}", "op": o, "cartan_ns": base["median_ns"]}
        for c in comparators:
            other = recs.get((c, m, d, o))
            if c == "geoopt" and o == "transport" and verified_projection:
                row[c] = None
                continue
            if other is None or (m, d, o, c) not in matched:
                row[c] = None
                continue
            ratio = other["median_ns"] / base["median_ns"]
            row[c] = (other["median_ns"], ratio)
            speed[c].append(ratio)
        rows.append(row)

    prov = provenance()
    date = args.date or "undated"

    lines: list[str] = []
    lines.append("# Cross-language comparison")
    lines.append("")
    lines.append(f"cartan against Manifolds.jl, geomstats and geoopt. Generated {date}.")
    lines.append("")
    lines.append("All four libraries read the same fixture file, so agreement is measured")
    lines.append("on identical inputs rather than on separately-seeded random draws.")
    lines.append("Regenerate with `python/make_fixtures.py` and the three harnesses.")
    lines.append("")

    lines.append("## Agreement")
    lines.append("")
    if disagreements:
        lines.append(f"**{len(disagreements)} comparisons exceed {AGREE_TOL:.0e}.**")
        lines.append("")
        lines.append("| manifold | dim | op | comparator | max abs difference |")
        lines.append("|---|---|---|---|---|")
        for m, d, o, c, v in disagreements:
            lines.append(f"| {m} | {d} | {o} | {c} | {v} |")
    else:
        lines.append(
            f"Every comparison agrees to better than `{AGREE_TOL:.0e}`. "
            f"Largest deviation across all {len(agreements)} comparisons: "
            f"`{worst:.3e}`."
        )
    lines.append("")

    if convention:
        lines.append("### Verified convention difference: geoopt transport")
        lines.append("")
        lines.append("geoopt's `transp` is a projection onto the target tangent space, not")
        lines.append("exact parallel transport. That is the correct choice for its purpose,")
        lines.append("since a Riemannian optimiser needs a cheap transport rather than an")
        lines.append("isometric one, and it is why the values differ by far more than")
        lines.append("floating point would explain.")
        lines.append("")
        lines.append("This is checked, not assumed: each row confirms geoopt's transport")
        lines.append("equals its own `proju` to 1e-14. If geoopt ever switched to exact")
        lines.append("transport, these rows would be reported as real disagreements.")
        lines.append("")
        lines.append("| manifold | dim | gap vs cartan | equals its own projection to |")
        lines.append("|---|---|---|---|")
        for m, d, gap, delta in convention:
            lines.append(f"| {m} | {d} | {gap:.3e} | {delta:.1e} |")
        lines.append("")

    lines.append("## Timing")
    lines.append("")
    lines.append("Median nanoseconds per call. A ratio above 1 means cartan is faster.")
    lines.append("Rows whose values did not agree are omitted rather than reported.")
    lines.append("")
    lines.append("| case | op | cartan (ns) | Manifolds.jl | geomstats | geoopt |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        cells = []
        for c in comparators:
            v = r[c]
            cells.append("-" if v is None else f"{v[0]:.0f} ({v[1]:.1f}x)")
        lines.append(
            f"| {r['case']} | {r['op']} | {r['cartan_ns']:.0f} | " + " | ".join(cells) + " |"
        )
    lines.append("")

    lines.append("### Median speedup")
    lines.append("")
    lines.append("| comparator | median ratio | cases |")
    lines.append("|---|---|---|")
    for c in comparators:
        if speed[c]:
            lines.append(
                f"| {c} | {np.median(speed[c]):.1f}x | {len(speed[c])} |"
            )
        else:
            lines.append(f"| {c} | - | 0 |")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append("- Absolute timings are hardware-specific. Ratios are the meaningful figure.")
    lines.append("- All three harnesses time in batches, not per call. `Instant::now` and")
    lines.append("  `perf_counter_ns` cost tens of nanoseconds, comparable to the fastest")
    lines.append("  operations here, so per-call timing reported a 15 ns `exp` as 40 ns.")
    lines.append("  Batching also needs the inputs black-boxed: with loop-invariant")
    lines.append("  arguments the optimiser hoists the whole call out and reports 0 ns.")
    lines.append("- The Rust harness is cross-checked against criterion, which measures")
    lines.append("  independently: criterion puts sphere exp at dim 3 at 15.8 ns and this")
    lines.append("  harness at 15 ns.")
    lines.append("- The Python harness uses a fixed 300-sample median and is the least")
    lines.append("  precise of the three, though Python call overhead dwarfs that anyway.")
    lines.append("- Julia timings exclude compilation, which BenchmarkTools warms away. A")
    lines.append("  cold Julia process pays that cost once and this table does not show it.")
    lines.append("- geoopt is measured on float64 CPU tensors. It is built for batched GPU")
    lines.append("  autograd, so single-point CPU timings are not what it optimises for.")
    lines.append("- Parallel transport is compared on the sphere only: the SPD convention")
    lines.append("  differs between libraries, so a mismatch would measure the convention.")
    lines.append("")

    lines.append("## Provenance")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    for k, v in prov.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")
    print(f"comparisons: {len(agreements)}, disagreements: {len(disagreements)}, worst: {worst:.3e}")


if __name__ == "__main__":
    main()
