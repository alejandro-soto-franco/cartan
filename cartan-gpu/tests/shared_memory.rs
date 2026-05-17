//! Zero-copy Vulkan↔CUDA memory sharing test.
//!
//! Allocates a single exportable VkDeviceMemory, writes a known f32
//! pattern via Vulkan's `map_memory`, then reads the same bytes back
//! through CUDA's `cuMemcpyDtoH` on the imported `CUdeviceptr`. If the
//! two views agree, the underlying allocation truly is shared.

#![cfg(all(feature = "vkfft", feature = "cufft", target_os = "linux"))]

use cartan_gpu::{CudaDevice, Device, SharedMemory};
use cudarc::driver::DeviceSlice;

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
            eprintln!("shared-memory test skipped, CUDA init failed: {msg}");
            return None;
        }
        Err(e) => panic!("CUDA device: {e}"),
    };
    Some((vk, cuda))
}

#[test]
fn shared_memory_vulkan_writes_cuda_reads_same_bytes() {
    let Some((vk_dev, cuda_dev)) = open_or_skip() else {
        return;
    };

    // 1024 f32s = 4096 bytes.
    let n = 1024usize;
    let size_bytes = (n * std::mem::size_of::<f32>()) as u64;

    let shared = match SharedMemory::new(&vk_dev, cuda_dev.cuda_context(), size_bytes) {
        Ok(s) => s,
        // External-memory paths can fail in setups we don't control
        // (no compatible memory type, integrated GPU without OPAQUE_FD,
        // sandboxing); skip rather than poison CI.
        Err(e) => {
            eprintln!("shared-memory setup failed, test skipped: {e}");
            return;
        }
    };

    // Write pattern from host through Vulkan's view.
    let pattern: Vec<f32> = (0..n).map(|i| (i as f32) + 1.0).collect();
    let bytes_in: &[u8] = bytemuck::cast_slice(&pattern);
    shared
        .write_host_bytes(bytes_in)
        .expect("Vulkan host write into shared memory");

    // Read back through CUDA's view of the same memory.
    let mapped = shared.cuda_view();
    let stream = mapped.stream().clone();
    let host_bytes: Vec<u8> = stream
        .clone_dtoh(mapped)
        .expect("CUDA dtoh on shared memory");
    let host_f32: Vec<f32> = bytemuck::cast_slice(&host_bytes).to_vec();

    assert_eq!(host_f32.len(), pattern.len());
    for (i, (a, b)) in pattern.iter().zip(host_f32.iter()).enumerate() {
        assert_eq!(
            a, b,
            "shared-memory mismatch at element {i}: vulkan wrote {a}, cuda saw {b}"
        );
    }

    eprintln!(
        "shared-memory: Vulkan wrote {n} f32s, CUDA read identical bytes via the same allocation"
    );
}
