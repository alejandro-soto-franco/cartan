#![cfg(feature = "vkfft")]

use cartan_gpu::Device;

#[test]
fn extract_raw_vulkan_handles() {
    let dev = match Device::new() {
        Ok(d) => d,
        Err(cartan_gpu::GpuError::NoAdapter) => return,
        Err(e) => panic!("{e}"),
    };
    let h = dev.raw_vulkan().unwrap();
    assert_ne!(h.physical_device, 0, "physical device handle is null");
    assert_ne!(h.device, 0, "device handle is null");
    assert_ne!(h.queue, 0, "queue handle is null");
    assert_ne!(h.instance, 0, "instance handle is null");
    println!("vulkan handles: {h:?}");
}
