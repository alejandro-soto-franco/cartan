# cartan-py

Python bindings for the cartan ecosystem.

[![PyPI](https://img.shields.io/pypi/v/cartan.svg)](https://pypi.org/project/cartan/)

Part of the [cartan](https://crates.io/crates/cartan) workspace. Built
with [PyO3](https://pyo3.rs/) and [maturin](https://www.maturin.rs/).

## What this crate does

`cartan-py` exposes the cartan Rust library to Python. The goal is to
make the same manifolds, optimizers, and homogenisation schemes available
to Python users with numpy interop, without rewriting the math layer.

The crate is its own Cargo build target separate from the parent
workspace so that Python-build-only deps (PyO3, abi3 selection) don't
leak into the pure-Rust crates.

## Build

```bash
pip install maturin
cd cartan-py
maturin develop --release   # install into the current virtualenv
# or
maturin build --release     # produce a wheel
```

The PyPI distribution uses `abi3-py39`, so a single wheel works for
Python 3.9 and newer.

## Surface (selected)

- Manifolds: `Sphere`, `Stiefel`, `Grassmann`, `Spd`, `SO3`, `SE3`.
- Optimisation: `frechet_mean`, `RGD`, `RCG`, `RTR`.
- Homogenisation: `MoriTanaka`, `SelfConsistent`, `Differential`,
  `Maxwell`, `PonteCastanedaWillis`, and the `FullField` voxel path.
- numpy interop: all manifold points and tangent vectors marshal through
  `numpy.ndarray`.

## License

[MIT](../LICENSE-MIT)
