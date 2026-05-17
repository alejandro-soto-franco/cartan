//! Cross-backend FFT benchmark. Compares VkFFT, cuFFT, and rustfft (CPU)
//! on identical Complex32 inputs at a handful of sizes.
//!
//! Run with: `cargo bench --features "vkfft cuda cufft" --bench fft_compare`
//!
//! These numbers are descriptive, not normative — they reflect the local
//! hardware (RTX 5060 Laptop GPU under CUDA 13.1 in our development setup)
//! and are intended for "which backend on this machine" decisions, not
//! cross-machine comparisons.

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use num_complex::Complex32;
use rustfft::FftPlanner;

use cartan_gpu::{
    CuFftBackend, CudaBuffer, CudaDevice, Device, Fft, FftDirection, GpuBuffer, VkFftBackend,
};

fn make_input(n: usize) -> Vec<Complex32> {
    (0..n)
        .map(|i| {
            let t = (i as f32) * 0.001;
            Complex32::new(t.sin(), t.cos())
        })
        .collect()
}

fn bench_rustfft(c: &mut Criterion) {
    let mut group = c.benchmark_group("rustfft_cpu_1d");
    group.measurement_time(Duration::from_secs(3));
    for &n in &[1024usize, 4096, 16384, 65536] {
        let mut planner = FftPlanner::<f32>::new();
        let plan = planner.plan_fft_forward(n);
        let input = make_input(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter_with_setup(
                || input.clone(),
                |mut data| {
                    plan.process(&mut data);
                    black_box(data);
                },
            );
        });
    }
    group.finish();
}

fn bench_vkfft(c: &mut Criterion) {
    let dev = match Device::new() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("vkfft bench skipped: no Vulkan device");
            return;
        }
    };
    let mut fft = match VkFftBackend::new(&dev) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("vkfft bench skipped: backend init failed");
            return;
        }
    };

    let mut group = c.benchmark_group("vkfft_1d");
    group.measurement_time(Duration::from_secs(3));
    for &n in &[1024usize, 4096, 16384, 65536] {
        let input = make_input(n);
        let mut buf = GpuBuffer::<Complex32>::from_slice(
            &dev,
            &input,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        )
        .expect("upload");

        // Warmup so first-call shader compilation doesn't poison the sample.
        let _ = fft.fft_1d(&mut buf, n as u32, 1, FftDirection::Forward);

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                fft.fft_1d(&mut buf, n as u32, 1, FftDirection::Forward)
                    .expect("vkfft fwd");
            });
        });
    }
    group.finish();
}

fn bench_cufft(c: &mut Criterion) {
    let dev = match CudaDevice::new() {
        Ok(d) => d,
        Err(_) => {
            eprintln!("cufft bench skipped: no CUDA device");
            return;
        }
    };
    let mut fft = match CuFftBackend::new(&dev) {
        Ok(f) => f,
        Err(_) => {
            eprintln!("cufft bench skipped: backend init failed");
            return;
        }
    };

    let mut group = c.benchmark_group("cufft_1d");
    group.measurement_time(Duration::from_secs(3));
    for &n in &[1024usize, 4096, 16384, 65536] {
        let input = make_input(n);
        let mut buf = CudaBuffer::from_slice(&dev, &input).expect("upload");
        let _ = fft.fft_1d(&mut buf, n as u32, 1, FftDirection::Forward); // warmup

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                fft.fft_1d(&mut buf, n as u32, 1, FftDirection::Forward)
                    .expect("cufft fwd");
            });
        });
    }
    group.finish();
}

// Silence unused-imports warnings when only some features are enabled.
#[allow(dead_code)]
fn _force_arc_used(_: Arc<()>) {}

criterion_group!(benches, bench_rustfft, bench_vkfft, bench_cufft);
criterion_main!(benches);
