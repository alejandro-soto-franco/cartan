# cartan-cuda

Batched double-precision manifold operations on CUDA, via
[cuda-oxide](https://github.com/NVlabs/cuda-oxide).

Part of the [cartan](https://crates.io/crates/cartan) workspace, but **not
published** and **not a workspace member**. See [Status](#status).

## Why this exists

Precision. `cartan-gpu` targets wgpu, and WGSL has no `f64`, so everything
through it is single precision while the rest of cartan is double. CUDA has
`f64` natively, so these kernels are held to the same tolerance the CPU code
is.

Measured against the CPU implementation on an RTX 5060 Laptop:

| dim | points | `exp` max error | `log` max error |
|---|---|---|---|
| 3 | 4096 | 3.22e-15 | 3.33e-16 |
| 10 | 4096 | 3.33e-16 | 3.89e-16 |
| 50 | 2048 | 1.67e-16 | 2.71e-16 |

And the affine-invariant distance on SPD(3), 4096 pairs, to **9.58e-15**
relative.

That is machine precision throughout. An f32 path could not be held within
1e-6.

## Kernels are Rust

`rustc-codegen-cuda` compiles them to PTX, so there is no separate shader
language and no string-embedded source:

```rust
#[kernel]
pub fn sphere_tangent_norm(v: &[f64], dim: u32, mut out: DisjointSlice<f64>) {
    let idx = thread::index_1d();
    let i = idx.get();
    let mut sum_sq = 0.0f64;
    for k in 0..dim as usize {
        let x = v[i * dim as usize + k];
        sum_sq += x * x;
    }
    if let Some(slot) = out.get_mut(idx) {
        *slot = sum_sq.sqrt();
    }
}
```

## SPD(3) is one kernel, not two

Distance returns a single scalar per pair, so it fits the one-element-per-thread
shape directly. Each thread does the whole job: Cholesky of `P`, the spectrum of
`L^-1 Q L^-T` by cyclic Jacobi, then `sqrt(sum log^2(lambda))`. The matrix
logarithm is never formed, which is the same reasoning the CPU path uses.

Jacobi suits a GPU for the reason it suits small matrices generally: it is
branch-light, needs no pivoting, and works entirely in registers on a 3x3. The
eigenvectors are not accumulated, since only the spectrum reaches the answer.

## Why the sphere operations are two kernels

`DisjointSlice` gives each thread exactly one output element, which is what
makes the writes provably non-overlapping. A batched `exp` produces `dim`
elements per point and does not fit that shape.

Splitting it does. One kernel runs per point and performs the O(dim)
reduction, writing a single scalar; the next runs per output element and
consumes it. The reduction then happens once per point rather than once per
component, which is the reason not to fold them together.

## Running

```bash
cargo oxide run cartan-cuda
```

Needs the pinned nightly in `rust-toolchain.toml`, a CUDA toolkit with
libNVVM, and `cargo-oxide`. Check with `cargo oxide doctor`.

Point it at a specific backend rather than relying on the global cache:

```bash
export CUDA_OXIDE_BACKEND=~/cuda-oxide/crates/rustc-codegen-cuda/target/x86_64-unknown-linux-gnu/debug/librustc_codegen_cuda.so
```

This matters. `cargo-oxide` resolves the backend in the order: `CUDA_OXIDE_BACKEND`,
project `.cargo/cuda-oxide.toml`, local repo build, cached build at
`~/.cargo/cuda-oxide/`, then auto-fetch from git. A build run from outside the
cuda-oxide checkout falls through to that cache, and `cargo oxide setup`
rebuilds the in-repo backend without refreshing it, so the cache can be
arbitrarily old. Ours was two months stale and rejected `acos` with
`FORBIDDEN CRATE IN DEVICE CODE`, while the in-repo examples using the same
call compiled fine.

`rm -rf ~/.cargo/cuda-oxide` is the documented remedy; the env var above avoids
depending on the cache at all.

## Device-code constraints

Only the local crate, `cuda_device` and `core` may appear in device code. `f64`
transcendentals live in `std`, not `core`, so they reach the GPU through an
allowlist that lowers them to libdevice (`acos` becomes `__nv_acos`). A
function outside that allowlist is rejected at compile time rather than
failing at run time.

Host-side code in this crate is compiled by the same backend, so the
dependencies it can carry are narrower than an ordinary crate's.

## Status

Not published, for three independent reasons:

1. cuda-oxide is not on crates.io, and cargo rejects git dependencies at
   publish time.
2. It pins a nightly toolchain with `rustc-dev` and `llvm-tools`, because it
   is a rustc backend. cartan is stable with an MSRV of 1.89.
3. It builds through `cargo oxide`, not `cargo build`.

None of that affects the rest of cartan: this crate has its own workspace and
sits in the root `exclude` list, so the stable build and CI never see it. It
becomes publishable when cuda-oxide does.

## License

[MIT](LICENSE-MIT)
