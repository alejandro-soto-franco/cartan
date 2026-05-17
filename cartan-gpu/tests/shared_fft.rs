//! Cross-backend FFT on shared GPU memory.
//!
//! Demonstrates end-to-end zero-copy interop: a single VkDeviceMemory
//! allocation is FFTd by VkFFT, then inverse-FFTd by cuFFT, on the same
//! physical bytes. If `forward∘inverse = identity` holds, the entire
//! unification stack works.

#![cfg(all(feature = "vkfft", feature = "cufft", target_os = "linux"))]

use cartan_gpu::{
    CuFftBackend, CudaDevice, Device, FftDirection, SharedFftBuffer, VkFftBackend,
};
use num_complex::Complex32;

fn open_or_skip() -> Option<(Device, CudaDevice)> {
    let vk = match Device::new() {
        Ok(d) => d,
        Err(cartan_gpu::GpuError::NoAdapter) => return None,
        Err(e) => panic!("Vulkan device: {e}"),
    };
    let cuda = match CudaDevice::new() {
        Ok(d) => d,
        Err(cartan_gpu::GpuError::NoAdapter) => return None,
        Err(cartan_gpu::GpuError::CudaError(msg)) => {
            eprintln!("shared-fft test skipped, CUDA init failed: {msg}");
            return None;
        }
        Err(e) => panic!("CUDA device: {e}"),
    };
    Some((vk, cuda))
}

#[test]
fn shared_fft_vk_forward_then_vk_inverse_is_identity() {
    let Some((vk_dev, cuda_dev)) = open_or_skip() else {
        return;
    };
    let mut vk_fft = VkFftBackend::new(&vk_dev).expect("vkfft backend");

    let n = 1024usize;
    let buf = match SharedFftBuffer::new(&vk_dev, cuda_dev.cuda_context(), n) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("shared-fft allocation failed: {e}");
            return;
        }
    };
    let mut buf = buf;

    let host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32 * 0.1).sin(), (i as f32 * 0.3).cos()))
        .collect();
    buf.upload(&host).expect("upload");

    vk_fft
        .fft_1d_shared(&mut buf, n as u32, 1, FftDirection::Forward)
        .expect("vk forward");
    vk_fft
        .fft_1d_shared(&mut buf, n as u32, 1, FftDirection::Inverse)
        .expect("vk inverse");

    let back = buf.download().expect("download");
    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-5, "Vk-only shared L-inf = {linf}");
}

#[test]
fn shared_fft_cuda_forward_then_cuda_inverse_is_identity() {
    let Some((vk_dev, cuda_dev)) = open_or_skip() else {
        return;
    };
    let mut cu_fft = CuFftBackend::new(&cuda_dev).expect("cufft backend");

    let n = 1024usize;
    let buf = match SharedFftBuffer::new(&vk_dev, cuda_dev.cuda_context(), n) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("shared-fft allocation failed: {e}");
            return;
        }
    };
    let mut buf = buf;

    let host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32 * 0.1).sin(), (i as f32 * 0.3).cos()))
        .collect();
    buf.upload(&host).expect("upload");

    cu_fft
        .fft_1d_shared(&mut buf, n as u32, 1, FftDirection::Forward)
        .expect("cuda forward");
    cu_fft
        .fft_1d_shared(&mut buf, n as u32, 1, FftDirection::Inverse)
        .expect("cuda inverse");

    let back = buf.download().expect("download");
    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-5, "CUDA-only shared L-inf = {linf}");
}

/// The big one: Vulkan computes the forward FFT, CUDA computes the
/// inverse, on the same physical memory. If this round-trips to the
/// original within numeric precision, zero-copy cross-backend FFT is
/// proven end-to-end.
#[test]
fn shared_fft_vk_forward_then_cuda_inverse_is_identity() {
    let Some((vk_dev, cuda_dev)) = open_or_skip() else {
        return;
    };
    let mut vk_fft = VkFftBackend::new(&vk_dev).expect("vkfft backend");
    let mut cu_fft = CuFftBackend::new(&cuda_dev).expect("cufft backend");

    let n = 1024usize;
    let buf = match SharedFftBuffer::new(&vk_dev, cuda_dev.cuda_context(), n) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("shared-fft allocation failed: {e}");
            return;
        }
    };
    let mut buf = buf;

    let host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32 * 0.1).sin(), (i as f32 * 0.3).cos()))
        .collect();
    buf.upload(&host).expect("upload");

    // Forward via Vulkan.
    vk_fft
        .fft_1d_shared(&mut buf, n as u32, 1, FftDirection::Forward)
        .expect("vk forward");

    // Inverse via CUDA — on the same bytes Vulkan just wrote.
    cu_fft
        .fft_1d_shared(&mut buf, n as u32, 1, FftDirection::Inverse)
        .expect("cuda inverse");

    let back = buf.download().expect("download");
    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);

    eprintln!("cross-backend (Vk→CUDA) shared-buffer round-trip L-inf = {linf}");
    assert!(
        linf < 1e-5,
        "cross-backend shared L-inf = {linf} exceeds 1e-5"
    );
}
