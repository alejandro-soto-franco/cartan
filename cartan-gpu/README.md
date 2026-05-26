# cartan-gpu

Portable wgpu-based GPU compute primitives for the cartan ecosystem.

[![crates.io](https://img.shields.io/crates/v/cartan-gpu.svg)](https://crates.io/crates/cartan-gpu)
[![docs.rs](https://docs.rs/cartan-gpu/badge.svg)](https://docs.rs/cartan-gpu)

Part of the [cartan](https://crates.io/crates/cartan) workspace, but its own
detached Cargo workspace so its wgpu/GPU build deps stay isolated from the rest
of the stack.

## What this crate does

`cartan-gpu` exposes a small, opinionated GPU surface to the rest of the
cartan stack:

- **Device**: `wgpu` 29 adapter + device + queue, enforcing Vulkan backend.
- **GpuBuffer\<T\>**: typed storage-buffer wrapper with upload (`from_slice`)
  and synchronous readback (`to_vec`).
- **Kernel**: minimal wgpu compute-pipeline scaffold (single storage buffer
  at group 0 binding 0). Sufficient for proof-of-life shaders; richer bind
  groups are wired in downstream crates.

**FFT has moved to `gpufft`.** With the `fft` feature, `gpufft` is
re-exported as `cartan_gpu::gpufft` for convenience. The `vulkan`, `cuda`,
and `shared` features gate the corresponding gpufft backends.

## Quick start (wgpu compute)

```rust,no_run
use cartan_gpu::{Device, GpuBuffer, Kernel};

let dev = Device::new().unwrap();

let n = 512_usize;
let host: Vec<f32> = (0..n).map(|i| i as f32).collect();
let buf = GpuBuffer::<f32>::from_slice(
    &dev,
    &host,
    wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_SRC
        | wgpu::BufferUsages::COPY_DST,
)
.unwrap();

let kernel = Kernel::from_wgsl(&dev, "hello", include_str!("shaders/hello.wgsl"), "main").unwrap();
kernel.dispatch(&dev, &buf, (n as u32).div_ceil(64), 1, 1);

let out = buf.to_vec(&dev).unwrap();
```

## FFT (via gpufft)

FFT was extracted to the standalone [`gpufft`](https://crates.io/crates/gpufft)
crate in v0.6. Enable the `fft` (or `vulkan` / `cuda` / `shared`) feature to
pull it in and access it as `cartan_gpu::gpufft`:

```toml
[dependencies]
cartan-gpu = { version = "0.6", features = ["vulkan"] }
```

```rust,no_run
use cartan_gpu::gpufft::{Direction, PlanDesc, vulkan::VulkanC2cPlan};
use num_complex::Complex32;

// gpufft owns device creation for FFT work; cartan_gpu::Device is for wgpu compute.
let fft_dev = cartan_gpu::gpufft::vulkan::VulkanDevice::new().unwrap();
let plan = VulkanC2cPlan::new(&fft_dev, PlanDesc { len: 1024, batch: 1 }).unwrap();
// ... execute plan ...
```

### Migrating from cartan-gpu 0.5

| v0.5 (cartan-gpu) | v0.6 (gpufft) |
|---|---|
| `cartan_gpu::Fft` | `gpufft::Fft` |
| `cartan_gpu::FftDirection` | `gpufft::Direction` |
| `cartan_gpu::VkFftBackend` | `gpufft::vulkan::VulkanBackend` |
| `cartan_gpu::CuFftBackend` | `gpufft::cuda::CudaBackend` |
| `cartan_gpu::UniBuffer` | `gpufft::UniBuffer` |
| `cartan_gpu::UniFftBackend` | `gpufft::UniFftBackend` |
| `cartan_gpu::SharedMemory` | `gpufft::shared::SharedMemory` |
| `cartan_gpu::SharedFftBuffer` | `gpufft::shared::SharedFftBuffer` |

Replace `device.plan_*` calls with `gpufft::{vulkan,cuda}::*Plan::new(&fft_dev, PlanDesc { ... })`.

## Features

| Feature | Pulls in | Notes |
|---|---|---|
| `fft` | `gpufft` (no backends) | Re-exports gpufft |
| `vulkan` | `fft` + `gpufft/vulkan` | Vulkan FFT backend |
| `cuda` | `fft` + `gpufft/cuda` | CUDA FFT backend |
| `shared` | `vulkan` + `cuda` + `gpufft/shared` | Zero-copy Vk↔CUDA (Linux) |

## Tests

```
cargo test --no-default-features --tests
```

The three remaining integration tests (`device_smoke`, `hello_shader`,
`buffer_roundtrip`) exercise the wgpu compute layer. FFT tests live in
`gpufft/tests/`.

## License

[MIT](../LICENSE-MIT)
