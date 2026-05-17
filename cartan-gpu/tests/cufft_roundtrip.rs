#![cfg(feature = "cufft")]

use cartan_gpu::{CuFftBackend, CudaBuffer, CudaDevice, Fft, FftDirection};
use num_complex::Complex32;

fn open_cuda_or_skip() -> Option<CudaDevice> {
    match CudaDevice::new() {
        Ok(d) => Some(d),
        Err(cartan_gpu::GpuError::NoAdapter) => None,
        Err(cartan_gpu::GpuError::CudaError(msg)) => {
            eprintln!("cuFFT test skipped, no CUDA: {msg}");
            None
        }
        Err(e) => panic!("unexpected error opening CUDA device: {e}"),
    }
}

#[test]
fn cufft_1d_forward_then_inverse_is_identity_up_to_1e5() {
    let Some(dev) = open_cuda_or_skip() else { return };
    let mut fft = CuFftBackend::new(&dev).expect("cuFFT backend");

    let n = 1024u32;
    let host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32).sin(), (i as f32 * 0.3).cos()))
        .collect();
    let mut buf = CudaBuffer::from_slice(&dev, &host).expect("upload");

    fft.fft_1d(&mut buf, n, 1, FftDirection::Forward).unwrap();
    fft.fft_1d(&mut buf, n, 1, FftDirection::Inverse).unwrap();

    let back = buf.to_vec().expect("download");
    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-5, "1D L-inf = {linf} exceeds 1e-5");
}

#[test]
fn cufft_1d_batched() {
    let Some(dev) = open_cuda_or_skip() else { return };
    let mut fft = CuFftBackend::new(&dev).expect("cuFFT backend");

    let (n, batch) = (256u32, 4u32);
    let host: Vec<Complex32> = (0..(n * batch))
        .map(|i| Complex32::new((i as f32 * 0.05).sin(), 0.0))
        .collect();
    let mut buf = CudaBuffer::from_slice(&dev, &host).expect("upload");

    fft.fft_1d(&mut buf, n, batch, FftDirection::Forward).unwrap();
    fft.fft_1d(&mut buf, n, batch, FftDirection::Inverse).unwrap();

    let back = buf.to_vec().expect("download");
    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-5, "1D batched L-inf = {linf}");
}

#[test]
fn cufft_2d_forward_then_inverse_is_identity() {
    let Some(dev) = open_cuda_or_skip() else { return };
    let mut fft = CuFftBackend::new(&dev).expect("cuFFT backend");

    let (nx, ny) = (64u32, 64u32);
    let host: Vec<Complex32> = (0..(nx * ny))
        .map(|i| Complex32::new((i as f32).sin(), (i as f32 * 0.7).cos()))
        .collect();
    let mut buf = CudaBuffer::from_slice(&dev, &host).expect("upload");

    fft.fft_2d(&mut buf, nx, ny, 1, FftDirection::Forward).unwrap();
    fft.fft_2d(&mut buf, nx, ny, 1, FftDirection::Inverse).unwrap();

    let back = buf.to_vec().expect("download");
    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-5, "2D L-inf = {linf}");
}

#[test]
fn cufft_2d_batched() {
    let Some(dev) = open_cuda_or_skip() else { return };
    let mut fft = CuFftBackend::new(&dev).expect("cuFFT backend");

    let (nx, ny, batch) = (32u32, 32u32, 3u32);
    let host: Vec<Complex32> = (0..(nx * ny * batch))
        .map(|i| Complex32::new((i as f32 * 0.11).sin(), (i as f32 * 0.21).cos()))
        .collect();
    let mut buf = CudaBuffer::from_slice(&dev, &host).expect("upload");

    fft.fft_2d(&mut buf, nx, ny, batch, FftDirection::Forward).unwrap();
    fft.fft_2d(&mut buf, nx, ny, batch, FftDirection::Inverse).unwrap();

    let back = buf.to_vec().expect("download");
    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-5, "2D batched L-inf = {linf}");
}

/// Non-power-of-2 size. cuFFT handles arbitrary sizes via mixed-radix
/// decomposition; this confirms our wiring passes the size through
/// without forcing a 2^k constraint anywhere.
#[test]
fn cufft_1d_non_power_of_two() {
    let Some(dev) = open_cuda_or_skip() else { return };
    let mut fft = CuFftBackend::new(&dev).expect("cuFFT backend");

    let n = 1000u32; // not a power of 2 and not a prime — mixed radix territory
    let host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32 * 0.013).sin(), (i as f32 * 0.027).cos()))
        .collect();
    let mut buf = CudaBuffer::from_slice(&dev, &host).expect("upload");

    fft.fft_1d(&mut buf, n, 1, FftDirection::Forward).unwrap();
    fft.fft_1d(&mut buf, n, 1, FftDirection::Inverse).unwrap();

    let back = buf.to_vec().expect("download");
    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-5, "1D non-POT (n={n}) L-inf = {linf}");
}

/// Repeatedly allocate, transform, and drop a buffer + plan to surface
/// GPU-memory leaks. Failure mode is an out-of-memory `CudaError` long
/// before the loop finishes.
#[test]
fn cufft_alloc_exec_drop_stress() {
    let Some(dev) = open_cuda_or_skip() else { return };
    let mut fft = CuFftBackend::new(&dev).expect("cuFFT backend");

    for iter in 0..32 {
        let n = 256u32 + (iter * 16); // vary the size so plan cache doesn't trivially absorb
        let host: Vec<Complex32> =
            (0..n).map(|i| Complex32::new(i as f32, 0.0)).collect();
        let mut buf = CudaBuffer::from_slice(&dev, &host).expect("upload");
        fft.fft_1d(&mut buf, n, 1, FftDirection::Forward).unwrap();
        fft.fft_1d(&mut buf, n, 1, FftDirection::Inverse).unwrap();
        // buf and the per-size plan-cache entry live until next iteration's
        // alloc. After 32 iterations we expect no driver-level OOM.
    }
}

#[test]
fn cufft_3d_forward_then_inverse_is_identity() {
    let Some(dev) = open_cuda_or_skip() else { return };
    let mut fft = CuFftBackend::new(&dev).expect("cuFFT backend");

    let (nx, ny, nz) = (32u32, 32u32, 32u32);
    let host: Vec<Complex32> = (0..(nx * ny * nz))
        .map(|i| Complex32::new((i as f32).sin(), (i as f32 * 1.1).cos()))
        .collect();
    let mut buf = CudaBuffer::from_slice(&dev, &host).expect("upload");

    fft.fft_3d(&mut buf, nx, ny, nz, FftDirection::Forward).unwrap();
    fft.fft_3d(&mut buf, nx, ny, nz, FftDirection::Inverse).unwrap();

    let back = buf.to_vec().expect("download");
    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-5, "3D L-inf = {linf}");
}
