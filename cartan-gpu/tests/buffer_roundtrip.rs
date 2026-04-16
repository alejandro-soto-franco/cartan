use cartan_gpu::{Device, GpuBuffer};

#[test]
fn buffer_upload_download_bit_identity() {
    let dev = match Device::new() {
        Ok(d) => d,
        Err(cartan_gpu::GpuError::NoAdapter) => return,
        Err(e) => panic!("{e}"),
    };

    let host: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5).collect();
    let buf = GpuBuffer::<f32>::from_slice(
        &dev,
        &host,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    )
    .unwrap();

    let back: Vec<f32> = buf.to_vec(&dev).unwrap();
    assert_eq!(back, host);
}
