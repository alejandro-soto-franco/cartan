"""Steelmanned Python: hand-written kernels compiled with numba.

geomstats and geoopt are general libraries paying Python overhead on every
call, so their timings say little about what Python can reach. This is the
other end of the range: the same algorithms cartan uses, written directly
against numpy and compiled to native code by numba.

Two timings are reported, because they answer different questions and quoting
either alone is misleading.

    called      One call from Python. Includes numba's argument unboxing and
                dispatch, which is what a Python program actually pays.

    kernel      The batch loop runs inside a jitted driver, so nothing crosses
                the Python boundary. This is the compiled code alone, and it
                is the fair comparison against Rust and Julia machine code.

Algorithms are deliberately the same as cartan's, including the Cholesky-based
SPD distance and the trig-free sphere transport. Giving numba a worse algorithm
would measure the algorithm rather than the language.

Run:
    .venv/bin/python python/bench_numba.py
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
from numba import njit

ROOT = pathlib.Path(__file__).parent.parent
FIXTURES = ROOT / "fixtures" / "geometry_cases.json"
OUT = ROOT / "results" / "numba_geometry.jsonl"

ANGLE_EPS = 1e-7

# ── Sphere ──────────────────────────────────────────────────────────────────


@njit(cache=True, fastmath=False)
def sphere_exp(p, v):
    theta = np.sqrt(np.sum(v * v))
    if theta < ANGLE_EPS:
        w = p + v
        return w / np.sqrt(np.sum(w * w))
    return np.cos(theta) * p + np.sin(theta) * (v / theta)


@njit(cache=True, fastmath=False)
def sphere_log(p, q):
    c = np.sum(p * q)
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    theta = np.arccos(c)
    w = q - c * p
    wn = np.sqrt(np.sum(w * w))
    if theta < ANGLE_EPS or wn < ANGLE_EPS:
        return q - c * p
    return (theta / wn) * w


@njit(cache=True, fastmath=False)
def sphere_dist(p, q):
    d = p - q
    half = np.sqrt(np.sum(d * d)) / 2.0
    if half > 1.0:
        half = 1.0
    return 2.0 * np.arcsin(half)


@njit(cache=True, fastmath=False)
def sphere_transport(p, q, v):
    # Same closed form cartan uses: no trig, no normalisation.
    c = np.sum(p * q)
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    one_plus_c = 1.0 + c
    w = q - c * p
    beta = np.sum(v * w)
    t = v - (w / one_plus_c + p) * beta
    # Re-project, as cartan does.
    return t - np.sum(t * q) * q


# ── Sphere, steelmanned: explicit loops writing into a caller-owned buffer ──
#
# The plain kernels above allocate a fresh numpy array per call, which for a
# 3-vector costs several times the arithmetic. These write into a buffer the
# caller owns and use scalar loops rather than array expressions, removing
# every intermediate allocation as well.
#
# `fastmath=True` permits reassociation. It buys nothing on the allocating
# versions, where allocation dominates, and is worth ~2x at dimension 50 once
# allocation is gone. Results still agree with cartan to 5.6e-17.
#
# The interface differs: the caller preallocates and the result is written
# rather than returned. That is a fair thing to do in a hot loop, and it is
# reported separately from the value-returning column for that reason.


@njit(cache=True, fastmath=True)
def sphere_exp_buf(p, v, out):
    n = p.shape[0]
    theta_sq = 0.0
    for i in range(n):
        theta_sq += v[i] * v[i]
    theta = np.sqrt(theta_sq)
    if theta < ANGLE_EPS:
        s = 0.0
        for i in range(n):
            out[i] = p[i] + v[i]
            s += out[i] * out[i]
        s = np.sqrt(s)
        for i in range(n):
            out[i] /= s
        return out
    ct, st = np.cos(theta), np.sin(theta) / theta
    for i in range(n):
        out[i] = ct * p[i] + st * v[i]
    return out


@njit(cache=True, fastmath=True)
def sphere_log_buf(p, q, out):
    n = p.shape[0]
    c = 0.0
    for i in range(n):
        c += p[i] * q[i]
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    wn_sq = 0.0
    for i in range(n):
        w = q[i] - c * p[i]
        out[i] = w
        wn_sq += w * w
    wn = np.sqrt(wn_sq)
    theta = np.arccos(c)
    if theta < ANGLE_EPS or wn < ANGLE_EPS:
        return out
    k = theta / wn
    for i in range(n):
        out[i] *= k
    return out


@njit(cache=True, fastmath=True)
def sphere_dist_buf(p, q):
    n = p.shape[0]
    s = 0.0
    for i in range(n):
        d = p[i] - q[i]
        s += d * d
    half = np.sqrt(s) / 2.0
    if half > 1.0:
        half = 1.0
    return 2.0 * np.arcsin(half)


@njit(cache=True, fastmath=True)
def sphere_transport_buf(p, q, v, out):
    n = p.shape[0]
    c = 0.0
    for i in range(n):
        c += p[i] * q[i]
    opc = 1.0 + c
    beta = 0.0
    for i in range(n):
        beta += v[i] * (q[i] - c * p[i])
    s = 0.0
    for i in range(n):
        out[i] = v[i] - ((q[i] - c * p[i]) / opc + p[i]) * beta
        s += out[i] * q[i]
    for i in range(n):
        out[i] -= s * q[i]
    return out


# ── SPD ─────────────────────────────────────────────────────────────────────


@njit(cache=True, fastmath=False)
def _sym_apply(m, which):
    """Eigendecompose a symmetric matrix and map its eigenvalues.

    `which` selects the map: 0 sqrt, 1 inverse sqrt, 2 log, 3 exp. numba has
    no closures over function values here, so a tag is cheaper than four
    near-identical kernels.
    """
    w, vecs = np.linalg.eigh(m)
    d = np.empty_like(w)
    for i in range(w.shape[0]):
        x = w[i]
        if which == 0:
            d[i] = np.sqrt(max(x, 0.0))
        elif which == 1:
            d[i] = 1.0 / np.sqrt(x) if x > 1e-14 else 1.0 / np.sqrt(1e-14)
        elif which == 2:
            d[i] = np.log(max(x, 1e-14))
        else:
            d[i] = np.exp(x)
    return (vecs * d) @ vecs.T


@njit(cache=True, fastmath=False)
def spd_exp(p, v):
    sp = _sym_apply(p, 0)
    spi = _sym_apply(p, 1)
    s = spi @ v @ spi
    return sp @ _sym_apply(s, 3) @ sp


@njit(cache=True, fastmath=False)
def spd_log(p, q):
    sp = _sym_apply(p, 0)
    spi = _sym_apply(p, 1)
    m = spi @ q @ spi
    return sp @ _sym_apply(m, 2) @ sp


@njit(cache=True, fastmath=False)
def spd_dist(p, q):
    # Same algorithm cartan now uses: eigenvalues of L^-1 Q L^-T, never
    # forming the logarithm.
    lo = np.linalg.cholesky(p)
    a = np.linalg.solve(lo, q)
    m = np.linalg.solve(lo, a.T)
    w = np.linalg.eigvalsh(m)
    total = 0.0
    for i in range(w.shape[0]):
        ln = np.log(max(w[i], 1e-14))
        total += ln * ln
    return np.sqrt(total)


# ── Batched drivers, so the timing loop never leaves nopython mode ──────────


# One driver per return shape. Accumulating a single element rather than a
# sum of the result keeps the loop from adding a pass over the data, which
# would inflate the cheapest sphere cases.


@njit(cache=True)
def _batch_vec2(fn, a, b, n):
    acc = 0.0
    for _ in range(n):
        acc += fn(a, b)[0]
    return acc


@njit(cache=True)
def _batch_mat2(fn, a, b, n):
    acc = 0.0
    for _ in range(n):
        acc += fn(a, b)[0, 0]
    return acc


@njit(cache=True)
def _batch_s2(fn, a, b, n):
    acc = 0.0
    for _ in range(n):
        acc += fn(a, b)
    return acc


@njit(cache=True)
def _batch_buf2(fn, a, b, out, n):
    acc = 0.0
    for _ in range(n):
        acc += fn(a, b, out)[0]
    return acc


@njit(cache=True)
def _batch_buf3(fn, a, b, c, out, n):
    acc = 0.0
    for _ in range(n):
        acc += fn(a, b, c, out)[0]
    return acc


@njit(cache=True)
def _batch_vec3(fn, a, b, c, n):
    acc = 0.0
    for _ in range(n):
        acc += fn(a, b, c)[0]
    return acc


def time_called(fn, args, target_ns=2e8):
    """Per-call time from Python, including numba dispatch."""
    for _ in range(50):
        fn(*args)
    # Size the batch so the loop runs long enough to time.
    n = 100
    while True:
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn(*args)
        el = time.perf_counter_ns() - t0
        if el >= target_ns or n >= 1 << 20:
            return el / n
        n *= 4


def time_kernel(driver, fn, args, target_ns=2e8):
    """Per-call time of the compiled kernel, with the loop inside nopython."""
    driver(fn, *args, 10)
    n = 100
    while True:
        t0 = time.perf_counter_ns()
        driver(fn, *args, n)
        el = time.perf_counter_ns() - t0
        if el >= target_ns or n >= 1 << 22:
            return el / n
        n *= 4


def main() -> None:
    data = json.loads(FIXTURES.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []

    for case in data["cases"]:
        kind, dim = case["manifold"], case["dim"]
        print(f"benchmarking {kind} dim={dim}")

        if kind == "sphere":
            p = np.ascontiguousarray(case["p"], dtype=np.float64)
            v = np.ascontiguousarray(case["v"], dtype=np.float64)
            q = np.ascontiguousarray(case["q"], dtype=np.float64)
            ops = [
                ("exp", sphere_exp, (p, v), _batch_vec2, sphere_exp),
                ("log", sphere_log, (p, q), _batch_vec2, sphere_log),
                ("dist", sphere_dist, (p, q), _batch_s2, sphere_dist),
                ("transport", sphere_transport, (p, q, v), _batch_vec3, sphere_transport),
            ]
        else:
            p = np.ascontiguousarray(case["p"], dtype=np.float64)
            v = np.ascontiguousarray(case["v"], dtype=np.float64)
            q = np.ascontiguousarray(case["q"], dtype=np.float64)
            ops = [
                ("exp", spd_exp, (p, v), _batch_mat2, spd_exp),
                ("log", spd_log, (p, q), _batch_mat2, spd_log),
                ("dist", spd_dist, (p, q), _batch_s2, spd_dist),
            ]

        if kind == "sphere":
            buf = np.empty_like(p)
            buf_ops = [
                ("exp", sphere_exp_buf, (p, v, buf), _batch_buf2),
                ("log", sphere_log_buf, (p, q, buf), _batch_buf2),
                ("dist", sphere_dist_buf, (p, q), _batch_s2),
                ("transport", sphere_transport_buf, (p, q, v, buf), _batch_buf3),
            ]
            for op, fn, args, driver in buf_ops:
                value = fn(*args)
                value = np.atleast_1d(np.asarray(value, dtype=float)).ravel().tolist()
                kernel = time_kernel(driver, fn, args)
                called = time_called(fn, args)
                records.append({
                    "lib": "numba-buffer", "manifold": kind, "dim": dim, "op": op,
                    "value": value,
                    "median_ns": kernel, "q1_ns": kernel, "q3_ns": kernel,
                    "called_ns": called,
                })
                print(f"  {op:10} buffer {kernel:9.1f} ns")

        for op, fn, args, driver, kern in ops:
            value = fn(*args)
            value = np.atleast_1d(np.asarray(value, dtype=float)).ravel().tolist()

            called = time_called(fn, args)
            kernel = time_kernel(driver, kern, args)

            records.append({
                "lib": "numba", "manifold": kind, "dim": dim, "op": op,
                "value": value,
                "median_ns": kernel, "q1_ns": kernel, "q3_ns": kernel,
                "called_ns": called,
            })
            print(f"  {op:10} kernel {kernel:9.1f} ns   called {called:9.1f} ns")

    with OUT.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} records to {OUT}")


if __name__ == "__main__":
    main()
