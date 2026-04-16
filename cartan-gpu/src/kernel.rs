//! Kernel - thin wrapper around a wgpu compute pipeline bound to a single
//! read/write storage buffer at group 0 binding 0.
//!
//! The cryo-EM kernels in cartan-em-gpu use richer bind groups; this minimal
//! variant scaffolds the pipeline cache and is exercised by the hello.wgsl
//! proof-of-life test.

use crate::{Device, GpuBuffer, GpuError};
use bytemuck::Pod;

pub struct Kernel {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl Kernel {
    pub fn from_wgsl(
        dev: &Device,
        label: &str,
        source: &str,
        entry_point: &str,
    ) -> Result<Self, GpuError> {
        let module = dev
            .wgpu_device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        let bind_group_layout =
            dev.wgpu_device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(label),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let pipeline_layout = dev
            .wgpu_device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline =
            dev.wgpu_device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Ok(Self {
            pipeline,
            bind_group_layout,
        })
    }

    pub fn dispatch<T: Pod>(&self, dev: &Device, buf: &GpuBuffer<T>, x: u32, y: u32, z: u32) {
        let bind_group = dev
            .wgpu_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cartan-gpu::Kernel::dispatch"),
                layout: &self.bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.raw().as_entire_binding(),
                }],
            });
        let mut encoder = dev
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cartan-gpu::Kernel::dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cartan-gpu::Kernel::dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(x, y, z);
        }
        dev.wgpu_queue().submit(std::iter::once(encoder.finish()));
    }
}
