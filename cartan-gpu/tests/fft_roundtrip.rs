#![cfg(feature = "vkfft")]

use cartan_gpu::{Device, Fft, FftDirection, GpuBuffer, VkFftBackend};
use num_complex::Complex32;

#[test]
fn fft_1d_forward_then_inverse_is_identity_up_to_1e5() {
    std::io::Write::flush(&mut std::io::stderr()).ok();
    eprintln!(">>> TEST ENTRY");
    std::io::Write::flush(&mut std::io::stderr()).ok();
    let dev = match Device::new() {
        Ok(d) => d,
        Err(cartan_gpu::GpuError::NoAdapter) => return,
        Err(e) => panic!("{e}"),
    };
    let mut fft = VkFftBackend::new(&dev).unwrap();

    let n = 1024u32;
    let host: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new((i as f32).sin(), (i as f32 * 0.3).cos()))
        .collect();
    let buf = GpuBuffer::<Complex32>::from_slice(
        &dev,
        &host,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    )
    .unwrap();

    fft.fft_1d(&dev, &buf, n, 1, FftDirection::Forward).unwrap();
    fft.fft_1d(&dev, &buf, n, 1, FftDirection::Inverse).unwrap();
    let back = buf.to_vec(&dev).unwrap();

    let linf = host
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(0.0f32, f32::max);
    assert!(linf < 1e-5, "L-inf = {linf} exceeds 1e-5");
}

#[test]
fn fft_2d_forward_then_inverse_is_identity() {
    let dev = match Device::new() {
        Ok(d) => d,
        Err(cartan_gpu::GpuError::NoAdapter) => return,
        Err(e) => panic!("{e}"),
    };
    let mut fft = VkFftBackend::new(&dev).unwrap();

    let (nx, ny) = (64u32, 64u32);
    let host: Vec<Complex32> = (0..(nx * ny))
        .map(|i| Complex32::new((i as f32).sin(), (i as f32 * 0.7).cos()))
        .collect();
    let buf = GpuBuffer::<Complex32>::from_slice(
        &dev,
        &host,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    )
    .unwrap();

    fft.fft_2d(&dev, &buf, nx, ny, 1, FftDirection::Forward).unwrap();
    fft.fft_2d(&dev, &buf, nx, ny, 1, FftDirection::Inverse).unwrap();
    let back = buf.to_vec(&dev).unwrap();

    let linf = host.iter().zip(back.iter()).map(|(a, b)| (a - b).norm()).fold(0.0, f32::max);
    assert!(linf < 1e-5, "2D L-inf = {linf}");
}

#[test]
fn fft_3d_forward_then_inverse_is_identity() {
    let dev = match Device::new() {
        Ok(d) => d,
        Err(cartan_gpu::GpuError::NoAdapter) => return,
        Err(e) => panic!("{e}"),
    };
    let mut fft = VkFftBackend::new(&dev).unwrap();

    let (nx, ny, nz) = (32u32, 32u32, 32u32);
    let host: Vec<Complex32> = (0..(nx * ny * nz))
        .map(|i| Complex32::new((i as f32).sin(), (i as f32 * 1.1).cos()))
        .collect();
    let buf = GpuBuffer::<Complex32>::from_slice(
        &dev,
        &host,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    )
    .unwrap();

    fft.fft_3d(&dev, &buf, nx, ny, nz, FftDirection::Forward).unwrap();
    fft.fft_3d(&dev, &buf, nx, ny, nz, FftDirection::Inverse).unwrap();
    let back = buf.to_vec(&dev).unwrap();

    let linf = host.iter().zip(back.iter()).map(|(a, b)| (a - b).norm()).fold(0.0, f32::max);
    assert!(linf < 1e-5, "3D L-inf = {linf}");
}
