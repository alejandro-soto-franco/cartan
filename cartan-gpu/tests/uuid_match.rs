#![cfg(all(feature = "vkfft", feature = "cufft", target_os = "linux"))]

use cartan_gpu::{CudaDevice, Device};

#[test]
fn vulkan_and_cuda_uuids_match_on_same_gpu() {
    let vk = match Device::new() {
        Ok(d) => d,
        Err(cartan_gpu::GpuError::NoAdapter) => return,
        Err(e) => panic!("Vulkan device: {e}"),
    };
    let cuda = match CudaDevice::new() {
        Ok(d) => d,
        Err(_) => return,
    };

    let vk_uuid = vk.vulkan_device_uuid().expect("vulkan UUID");
    let cu_uuid = cuda.uuid().expect("cuda UUID");

    eprintln!("Vulkan device UUID: {vk_uuid:02x?}");
    eprintln!("CUDA   device UUID: {cu_uuid:02x?}");

    // On a single-GPU host (typical CI/dev setup) the two APIs should
    // report the same UUID. On a multi-GPU host where wgpu and CUDA
    // chose different adapters, this would correctly fail — and the
    // SharedFftBuffer/SharedMemory constructors would reject the
    // cross-GPU import via `GpuUuidMismatch`.
    assert_eq!(
        vk_uuid, cu_uuid,
        "single-GPU host should report matching UUIDs"
    );
}
