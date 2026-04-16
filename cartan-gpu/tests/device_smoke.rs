//! Smoke test: Device::new() succeeds on Vulkan backend.
//!
//! Skipped if no Vulkan adapter is available (e.g. CI without GPU).

use cartan_gpu::Device;

#[test]
fn device_new_on_vulkan_or_skip() {
    let dev = match Device::new() {
        Ok(d) => d,
        Err(cartan_gpu::GpuError::NoAdapter) => {
            eprintln!("skipping: no Vulkan adapter available");
            return;
        }
        Err(e) => panic!("unexpected error: {e}"),
    };
    let info = dev.adapter_info();
    assert_eq!(
        info.backend,
        wgpu::Backend::Vulkan,
        "expected Vulkan, got {:?}",
        info.backend
    );
    assert!(!info.name.is_empty(), "adapter has no name");
    println!("device OK on {} ({:?})", info.name, info.device_type);
}
