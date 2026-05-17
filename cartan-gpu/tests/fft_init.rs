#![cfg(feature = "vkfft")]
use cartan_gpu::{Device, VkFftBackend};

#[test]
fn vkfft_backend_new() {
    eprintln!("test starting");
    let dev = match Device::new() {
        Ok(d) => d,
        Err(cartan_gpu::GpuError::NoAdapter) => return,
        Err(e) => panic!("{e}"),
    };
    eprintln!("device created");
    let _fft = VkFftBackend::new(&dev).unwrap();
    eprintln!("backend created");
}
