#![cfg(feature = "cufft")]

use cartan_gpu::{CuFft, CuFftBackend, CudaBuffer, CudaDevice, FftDirection};
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

fn normalize(host: &mut [Complex32], scale: f32) {
    for c in host.iter_mut() {
        *c = Complex32::new(c.re * scale, c.im * scale);
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

    // cuFFT does not auto-normalize; scale by 1/N on the host before checking.
    let mut back = buf.to_vec().expect("download");
    normalize(&mut back, 1.0 / (n as f32));

    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-5, "1D L-inf = {linf} exceeds 1e-5");
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

    let mut back = buf.to_vec().expect("download");
    normalize(&mut back, 1.0 / ((nx * ny) as f32));

    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-5, "2D L-inf = {linf}");
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

    let mut back = buf.to_vec().expect("download");
    normalize(&mut back, 1.0 / ((nx * ny * nz) as f32));

    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-5, "3D L-inf = {linf}");
}
