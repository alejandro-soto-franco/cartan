"""Generate the shared input fixtures for the cross-language comparison.

Every language reads this one file, so agreement is measured on identical
inputs rather than on separately-seeded random draws that only look alike.
Writing the points out explicitly also means the comparison does not depend on
three RNGs agreeing, which they never do.

Conventions differ between the libraries and are recorded here so each harness
can map them:

    sphere ambient N   cartan Sphere<N>   Manifolds.jl Sphere(N-1)   geomstats Hypersphere(dim=N-1)
    spd    size N      cartan Spd<N>      Manifolds.jl SPD(N)        geomstats SPDMatrices(N)

The SPD metric is affine-invariant in all three, which is the default for
Manifolds.jl and cartan, and `SPDAffineMetric` for geomstats.

Run:
    .venv/bin/python python/make_fixtures.py
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

SEED = 42
FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures"

# Ambient dimensions for the sphere, and matrix sizes for SPD.
SPHERE_DIMS = [3, 10, 50]
SPD_DIMS = [3, 6, 10]


def sphere_case(rng: np.random.Generator, n: int) -> dict:
    """A point, a tangent at it, and a second point, on the unit sphere in R^n.

    The tangent is projected onto the tangent space and scaled well inside the
    injectivity radius (pi), so `log` is unambiguous and every library must
    return the same branch.
    """
    p = rng.normal(size=n)
    p /= np.linalg.norm(p)

    v = rng.normal(size=n)
    v -= np.dot(v, p) * p          # project onto T_p
    v *= 0.7 / np.linalg.norm(v)   # well inside the cut locus at pi

    q = np.cos(0.7) * p + np.sin(0.7) * (v / np.linalg.norm(v))
    q /= np.linalg.norm(q)

    return {
        "manifold": "sphere",
        "dim": n,
        "p": p.tolist(),
        "v": v.tolist(),
        "q": q.tolist(),
    }


def spd_case(rng: np.random.Generator, n: int) -> dict:
    """A point, a symmetric tangent, and a second point, in the SPD(n) cone.

    Points are built as A A^T + n I so they are comfortably well-conditioned:
    a near-singular point would make the eigen decomposition dominate the
    timing and turn the accuracy comparison into a conditioning test.
    """
    a = rng.normal(size=(n, n))
    p = a @ a.T + n * np.eye(n)

    s = rng.normal(size=(n, n))
    v = 0.5 * (s + s.T)
    v *= 0.3 / np.linalg.norm(v)

    b = rng.normal(size=(n, n))
    q = b @ b.T + n * np.eye(n)

    return {
        "manifold": "spd",
        "dim": n,
        "p": p.tolist(),
        "v": v.tolist(),
        "q": q.tolist(),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    cases = [sphere_case(rng, n) for n in SPHERE_DIMS]
    cases += [spd_case(rng, n) for n in SPD_DIMS]

    FIXTURES.mkdir(parents=True, exist_ok=True)
    out = FIXTURES / "geometry_cases.json"
    out.write_text(json.dumps({"seed": SEED, "cases": cases}, indent=1))

    print(f"wrote {len(cases)} cases to {out}")
    for c in cases:
        print(f"  {c['manifold']:8} dim={c['dim']}")


if __name__ == "__main__":
    main()
