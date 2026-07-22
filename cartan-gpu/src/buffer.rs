//! `GpuBuffer<T>`, a typed storage-buffer wrapper.
//!
//! Holds a `wgpu::Buffer` plus element count. Host to GPU via
//! `from_slice`; GPU to host via `to_vec`, which uses a staging buffer
//! and blocks on the queue for a synchronous API. Callers needing async
//! read-back can reach into `raw()` directly.

use crate::{Device, GpuError};
use bytemuck::Pod;
use std::marker::PhantomData;
use wgpu::util::DeviceExt;

pub struct GpuBuffer<T: Pod> {
    buffer: wgpu::Buffer,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Pod> GpuBuffer<T> {
    /// Allocate an uninitialized buffer of `len` elements.
    pub fn zeroed(dev: &Device, len: usize, usage: wgpu::BufferUsages) -> Result<Self, GpuError> {
        let size_bytes = (len * std::mem::size_of::<T>()) as u64;
        let buffer = dev.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("cartan-gpu::GpuBuffer"),
            size: size_bytes,
            usage,
            mapped_at_creation: false,
        });
        Ok(Self {
            buffer,
            len,
            _marker: PhantomData,
        })
    }

    /// Upload `host` to a new buffer with the given usage flags.
    /// Must include `COPY_SRC` if you intend to read back via `to_vec`.
    pub fn from_slice(
        dev: &Device,
        host: &[T],
        usage: wgpu::BufferUsages,
    ) -> Result<Self, GpuError> {
        let buffer = dev
            .wgpu_device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cartan-gpu::GpuBuffer::from_slice"),
                contents: bytemuck::cast_slice(host),
                usage,
            });
        Ok(Self {
            buffer,
            len: host.len(),
            _marker: PhantomData,
        })
    }

    /// Copy the buffer's contents back to the host via a staging buffer.
    /// Synchronous; blocks on queue submission.
    pub fn to_vec(&self, dev: &Device) -> Result<Vec<T>, GpuError> {
        let size_bytes = (self.len * std::mem::size_of::<T>()) as u64;
        let staging = dev.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("cartan-gpu::GpuBuffer::to_vec::staging"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = dev
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cartan-gpu::GpuBuffer::to_vec"),
            });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, size_bytes);
        dev.wgpu_queue().submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        dev.wgpu_device()
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::ShaderCompilation {
                msg: format!("poll: {e:?}"),
            })?;
        rx.recv().unwrap().map_err(|e| GpuError::ShaderCompilation {
            msg: format!("map: {e:?}"),
        })?;

        let data = slice.get_mapped_range();
        let out: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn raw(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}
