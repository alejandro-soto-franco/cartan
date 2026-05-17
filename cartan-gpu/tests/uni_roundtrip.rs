//! Backend-agnostic round-trip tests through `UniFftBackend`.
//!
//! Each test constructs whatever backend is available at runtime,
//! writes the same generic code against the unified `Fft` trait, and
//! asserts `forward∘inverse = identity`. The point is to prove that
//! call-site code does not care which backend it ran on.

#![cfg(any(feature = "vkfft", feature = "cufft"))]

use cartan_gpu::{Fft, FftDirection, UniBuffer, UniFftBackend};
use num_complex::Complex32;

fn run_1d_roundtrip(mut engine: UniFftBackend) {
    let n = 1024u32;
    let host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32 * 0.1).sin(), (i as f32 * 0.3).cos()))
        .collect();
    let mut buf = UniBuffer::from_slice(&engine, &host).expect("allocate UniBuffer");

    engine
        .fft_1d(&mut buf, n, 1, FftDirection::Forward)
        .expect("forward");
    engine
        .fft_1d(&mut buf, n, 1, FftDirection::Inverse)
        .expect("inverse");

    let back = buf.to_vec(&engine).expect("download");
    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-5, "L-inf = {linf}");
}

#[cfg(feature = "vkfft")]
#[test]
fn uni_1d_via_vulkan() {
    let dev = match cartan_gpu::Device::new() {
        Ok(d) => d,
        Err(cartan_gpu::GpuError::NoAdapter) => return,
        Err(e) => panic!("Vulkan device: {e}"),
    };
    let engine = UniFftBackend::vulkan(&dev).expect("vulkan engine");
    run_1d_roundtrip(engine);
}

#[cfg(feature = "cufft")]
#[test]
fn uni_1d_via_cuda() {
    let dev = match cartan_gpu::CudaDevice::new() {
        Ok(d) => d,
        Err(cartan_gpu::GpuError::NoAdapter) => return,
        Err(cartan_gpu::GpuError::CudaError(msg)) => {
            eprintln!("uni cuda test skipped: {msg}");
            return;
        }
        Err(e) => panic!("CUDA device: {e}"),
    };
    let engine = UniFftBackend::cuda(&dev).expect("cuda engine");
    run_1d_roundtrip(engine);
}

#[cfg(all(feature = "vkfft", feature = "cufft"))]
#[test]
fn uni_buffer_engine_mismatch_is_caught() {
    let vk_dev = match cartan_gpu::Device::new() {
        Ok(d) => d,
        Err(cartan_gpu::GpuError::NoAdapter) => return,
        Err(e) => panic!("Vulkan device: {e}"),
    };
    let cuda_dev = match cartan_gpu::CudaDevice::new() {
        Ok(d) => d,
        Err(_) => return,
    };

    let vk_engine = UniFftBackend::vulkan(&vk_dev).expect("vulkan engine");
    let mut cuda_engine = UniFftBackend::cuda(&cuda_dev).expect("cuda engine");

    // Allocate against the Vulkan engine, try to run on the CUDA engine.
    let host = vec![Complex32::new(1.0, 0.0); 64];
    let mut buf = UniBuffer::from_slice(&vk_engine, &host).expect("vulkan-side alloc");

    let err = cuda_engine
        .fft_1d(&mut buf, 64, 1, FftDirection::Forward)
        .expect_err("cross-backend run should fail");
    matches!(err, cartan_gpu::GpuError::BackendMismatch);
}
