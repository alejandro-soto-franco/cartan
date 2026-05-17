# cartan-gpu

Portable GPU compute primitives for the cartan ecosystem.

[![crates.io](https://img.shields.io/crates/v/cartan-gpu.svg)](https://crates.io/crates/cartan-gpu)
[![docs.rs](https://docs.rs/cartan-gpu/badge.svg)](https://docs.rs/cartan-gpu)

Part of the [cartan](https://crates.io/crates/cartan) workspace, but its own
detached Cargo workspace because the build depends on C/C++ toolchains
(VkFFT, glslang) and on optional CUDA / NVIDIA driver components.

## What this crate does

`cartan-gpu` exposes a small, opinionated GPU surface to the rest of the
cartan stack:

- **Vulkan path** (always on): `wgpu` 29 device + queue + typed
  `GpuBuffer<T>` storage, with a kernel-loading scaffold.
- **VkFFT path** (`vkfft`, default on): 1D / 2D / 3D forward and inverse
  FFTs through `VkFftBackend`, using a vendored VkFFT 1.3.4 submodule
  for the actual GPU code.
- **CUDA path** (`cuda`): `CudaDevice` over `cudarc` 0.19 (driver API
  only — no FFT yet at this level).
- **cuFFT path** (`cufft`): mirror of the VkFFT path on the CUDA side —
  `CuFftBackend` over `cudarc::cufft`, with cuBLAS-based on-device
  normalisation.
- **Unified FFT trait**: both `VkFftBackend` and `CuFftBackend`
  implement the same `Fft` trait with an associated `Buffer` type, so
  call-site code is identical across backends.
- **Runtime backend selection**: `UniFftBackend` + `UniBuffer` enum
  dispatch let you defer the Vulkan-vs-CUDA choice to session start.
- **Zero-copy interop** (`vkfft + cufft`, Linux): `SharedFftBuffer`
  exports `VkDeviceMemory` via `VK_KHR_external_memory_fd` and imports
  it into CUDA via `cuImportExternalMemory`, so VkFFT and cuFFT can
  operate on the same physical bytes without any host roundtrip or
  staging buffer.

Forward-then-inverse is identity on every path: VkFFT uses
`cfg.normalize = 1`, cuFFT post-scales by `1/N` with `cublasSscal_v2`.

## Quick start (Vulkan FFT)

```rust,no_run
use cartan_gpu::{Device, Fft, FftDirection, GpuBuffer, VkFftBackend};
use num_complex::Complex32;

let dev = Device::new().unwrap();
let mut fft = VkFftBackend::new(&dev).unwrap();

let n = 1024_usize;
let host: Vec<Complex32> = (0..n).map(|i| Complex32::new(i as f32, 0.0)).collect();
let mut buf = GpuBuffer::<Complex32>::from_slice(
    &dev,
    &host,
    wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_SRC
        | wgpu::BufferUsages::COPY_DST,
).unwrap();

fft.fft_1d(&mut buf, n as u32, 1, FftDirection::Forward).unwrap();
fft.fft_1d(&mut buf, n as u32, 1, FftDirection::Inverse).unwrap();
let back = buf.to_vec(&dev).unwrap();
// `back` equals `host` to within numerical precision.
```

The cuFFT path has the same shape:

```rust,no_run
use cartan_gpu::{CuFftBackend, CudaBuffer, CudaDevice, Fft, FftDirection};

let cuda = CudaDevice::new().unwrap();
let mut fft = CuFftBackend::new(&cuda).unwrap();
let mut buf = CudaBuffer::from_slice(&cuda, &host).unwrap();
fft.fft_1d(&mut buf, n as u32, 1, FftDirection::Forward).unwrap();
fft.fft_1d(&mut buf, n as u32, 1, FftDirection::Inverse).unwrap();
let back = buf.to_vec().unwrap();
```

## Runtime backend selection

`UniFftBackend` lets a single call site work against either backend:

```rust,no_run
use cartan_gpu::{Fft, FftDirection, UniBuffer, UniFftBackend};

let engine = if std::env::var("USE_CUDA").is_ok() {
    UniFftBackend::cuda(&cartan_gpu::CudaDevice::new().unwrap()).unwrap()
} else {
    UniFftBackend::vulkan(&cartan_gpu::Device::new().unwrap()).unwrap()
};

let mut buf = UniBuffer::from_slice(&engine, &host).unwrap();
let mut engine = engine; // mut for the trait method
engine.fft_1d(&mut buf, n as u32, 1, FftDirection::Forward).unwrap();
```

## Zero-copy Vulkan ↔ CUDA

On Linux with `vkfft + cufft` enabled, a single GPU allocation is
addressable from both APIs. The Vulkan VkFFT path writes, the CUDA cuFFT
path reads, no host roundtrip:

```rust,no_run
use cartan_gpu::{CuFftBackend, CudaDevice, Device, FftDirection, SharedFftBuffer, VkFftBackend};

let vk = Device::new().unwrap();
let cuda = CudaDevice::new().unwrap();
let mut vk_fft = VkFftBackend::new(&vk).unwrap();
let mut cu_fft = CuFftBackend::new(&cuda).unwrap();

let n = 1024_usize;
let mut buf = SharedFftBuffer::new(&vk, cuda.cuda_context(), n).unwrap();
buf.upload(&host).unwrap();

vk_fft.fft_1d_shared(&mut buf, n as u32, 1, FftDirection::Forward).unwrap();
cu_fft.fft_1d_shared(&mut buf, n as u32, 1, FftDirection::Inverse).unwrap();

let back = buf.download().unwrap();
// `back` ≈ `host`; the FFT data never touched the CPU between Vk and CUDA.
```

Verified on NVIDIA RTX 5060 Laptop (Blackwell SM 12.0, CUDA 13.1,
driver 580.x): Vk Forward → CUDA Inverse round-trip L-inf = 9e-7.
Cross-API memory sharing is gated by a same-GPU UUID match (Vulkan
`VkPhysicalDeviceIDProperties.deviceUUID` vs CUDA `cuDeviceGetUuid`) so
multi-GPU hosts can't silently fall into a broken non-shared mapping.

## Features

| Feature | Default | Pulls in | Notes |
|---|---|---|---|
| `vkfft` | yes | `cartan-gpu-sys`, `ash` 0.38 | Vulkan FFT via vendored VkFFT 1.3.4 |
| `cuda` | no | `cudarc` 0.19 (driver) | `CudaDevice` only |
| `cufft` | no | `cuda` + `cudarc/cufft` + `cudarc/cublas` | `CuFftBackend` |

The `cuda-NNNNN` ABI feature of cudarc is pinned to `cuda-13010` (CUDA
13.1) at the cartan-gpu level; consumers on a different CUDA installation
can patch the dep or switch to `cudarc/cuda-version-from-build-system`.

## Tests + benchmark

```
cargo test --features "vkfft cuda cufft" --tests
cargo bench --features "vkfft cuda cufft" --bench fft_compare
```

On the development machine (RTX 5060), the bench at n=1024 shows
rustfft ≈ 711 ns CPU, VkFFT ≈ 160 µs, cuFFT ≈ 24.5 µs. GPU backends are
launch-overhead-dominated at small N and dominate sharply at larger N.

## Known limits and follow-ups

- The shared-memory path is Linux-only (uses `OPAQUE_FD`); Windows would
  need a parallel `OPAQUE_WIN32` implementation.
- Cross-backend handoff still uses CPU-side `queue_wait_idle` /
  `stream.synchronize`. External-semaphore sync (importing a
  `VkSemaphore` into CUDA as a `cuExternalSemaphore`) is the
  perf-optimisation next step; the FFT compute cost dominates for any
  reasonable problem size, so the CPU wait is correctness-good even if
  not optimal.
- 2D batched FFTs go through `plan_many` on the cuFFT side; the VkFFT
  path supports batched 1D out of the box and would need additional
  shim wiring for batched higher dims.

## License

[MIT](../LICENSE-MIT)
