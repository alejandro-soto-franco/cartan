#![cfg(feature = "cuda")]
use cartan_gpu::CudaDevice;

#[test]
fn cuda_device_open_and_probe() {
    let dev = match CudaDevice::new() {
        Ok(d) => d,
        // Skip cleanly on machines with no CUDA runtime / no NVIDIA driver
        // loaded. We never want this to fail CI on a non-CUDA host.
        Err(cartan_gpu::GpuError::NoAdapter) => return,
        Err(cartan_gpu::GpuError::CudaError(msg)) => {
            eprintln!("cuda init skipped: {msg}");
            return;
        }
        Err(e) => panic!("unexpected error: {e}"),
    };

    let name = dev.name().expect("device name");
    let (major, minor) = dev.compute_capability().expect("compute capability");
    let total = dev
        .total_memory_bytes()
        .expect("total memory query");

    eprintln!("CUDA device: {name}");
    eprintln!("  compute capability: {major}.{minor}");
    eprintln!("  total memory: {} MiB", total / (1024 * 1024));

    assert!(!name.is_empty());
    assert!(major >= 3, "compute capability suspiciously low: {major}.{minor}");
    assert!(total > 0);
}
