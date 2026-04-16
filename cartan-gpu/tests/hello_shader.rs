use cartan_gpu::{Device, GpuBuffer, Kernel};

/// Runs shaders/hello.wgsl which adds 1.0 to every element of an f32 storage buffer.
#[test]
fn hello_compute_shader_adds_one() {
    let dev = match Device::new() {
        Ok(d) => d,
        Err(cartan_gpu::GpuError::NoAdapter) => return,
        Err(e) => panic!("{e}"),
    };

    let n = 512usize;
    let host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let buf = GpuBuffer::<f32>::from_slice(
        &dev,
        &host,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    )
    .unwrap();

    let kernel = Kernel::from_wgsl(
        &dev,
        "hello",
        include_str!("../shaders/hello.wgsl"),
        "main",
    )
    .unwrap();
    kernel.dispatch(&dev, &buf, (n as u32).div_ceil(64), 1, 1);

    let out = buf.to_vec(&dev).unwrap();
    for i in 0..n {
        assert_eq!(out[i], (i as f32) + 1.0, "mismatch at {i}: got {}", out[i]);
    }
}
