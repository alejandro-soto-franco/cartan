//! FFT abstraction and VkFFT-backed implementation.
//!
//! The `Fft` trait is the seam that lets us swap VkFFT for a future pure-Rust
//! backend without cartan-em churn. v0.1 ships one backend: `VkFftBackend`.

use crate::{Device, GpuBuffer, GpuError};
use num_complex::Complex32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FftDirection {
    Forward,
    Inverse,
}

pub trait Fft {
    fn fft_1d(
        &mut self,
        dev: &Device,
        buf: &GpuBuffer<Complex32>,
        n: u32,
        batch: u32,
        direction: FftDirection,
    ) -> Result<(), GpuError>;

    fn fft_2d(
        &mut self,
        dev: &Device,
        buf: &GpuBuffer<Complex32>,
        nx: u32,
        ny: u32,
        batch: u32,
        direction: FftDirection,
    ) -> Result<(), GpuError>;

    fn fft_3d(
        &mut self,
        dev: &Device,
        buf: &GpuBuffer<Complex32>,
        nx: u32,
        ny: u32,
        nz: u32,
        direction: FftDirection,
    ) -> Result<(), GpuError>;
}

#[cfg(feature = "vkfft")]
mod vkfft_impl {
    use super::*;
    use crate::hal_vulkan::RawVulkanHandles;
    use cartan_gpu_sys as sys;
    use std::collections::HashMap;

    #[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
    struct PlanKey {
        nx: u32,
        ny: u32,
        nz: u32,
        batch: u32,
        dim: u8,
    }

    struct VkFftPlan {
        app: sys::VkFFTApplication,
        vk_buffer: ash::vk::Buffer,
        vk_memory: ash::vk::DeviceMemory,
        wgpu_buffer: wgpu::Buffer,
        size_bytes: u64,
    }

    impl VkFftPlan {
        fn destroy(&mut self, ash_device: &ash::Device) {
            unsafe {
                sys::cartan_vkfft_delete(&mut self.app as *mut _);
                ash_device.destroy_buffer(self.vk_buffer, None);
                ash_device.free_memory(self.vk_memory, None);
            }
        }
    }

    pub struct VkFftBackend {
        handles: RawVulkanHandles,
        plans: HashMap<PlanKey, Box<VkFftPlan>>,
        command_pool: ash::vk::CommandPool,
        fence: ash::vk::Fence,
        ash_device: ash::Device,
        ash_instance: ash::Instance,
    }

    impl Drop for VkFftBackend {
        fn drop(&mut self) {
            unsafe {
                for (_, mut plan) in self.plans.drain() {
                    plan.destroy(&self.ash_device);
                }
                self.ash_device.destroy_fence(self.fence, None);
                self.ash_device.destroy_command_pool(self.command_pool, None);
            }
        }
    }

    impl VkFftBackend {
        pub fn new(dev: &crate::Device) -> Result<Self, GpuError> {
            use wgpu::hal::api::Vulkan;

            let handles = dev.raw_vulkan()?;

            let (ash_device, ash_instance) = unsafe {
                let hal_device = dev
                    .device
                    .as_hal::<Vulkan>()
                    .ok_or(GpuError::VulkanHandlesUnavailable)?;
                let device = hal_device.raw_device().clone();

                let hal_instance = dev
                    .instance
                    .as_hal::<Vulkan>()
                    .ok_or(GpuError::VulkanHandlesUnavailable)?;
                let instance = hal_instance.shared_instance().raw_instance().clone();

                (device, instance)
            };

            let command_pool = unsafe {
                let ci = ash::vk::CommandPoolCreateInfo::default()
                    .queue_family_index(handles.queue_family_index)
                    .flags(ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
                ash_device
                    .create_command_pool(&ci, None)
                    .map_err(|e| GpuError::ShaderCompilation {
                        msg: format!("create_command_pool: {e:?}"),
                    })?
            };

            let fence = unsafe {
                ash_device
                    .create_fence(&ash::vk::FenceCreateInfo::default(), None)
                    .map_err(|e| GpuError::ShaderCompilation {
                        msg: format!("create_fence: {e:?}"),
                    })?
            };

            Ok(Self {
                handles,
                plans: HashMap::new(),
                command_pool,
                fence,
                ash_device,
                ash_instance,
            })
        }

        fn find_memory_type(
            &self,
            type_filter: u32,
            properties: ash::vk::MemoryPropertyFlags,
        ) -> Result<u32, GpuError> {
            use ash::vk::Handle;
            let phys_dev = ash::vk::PhysicalDevice::from_raw(self.handles.physical_device);

            let mem_props = unsafe {
                self.ash_instance
                    .get_physical_device_memory_properties(phys_dev)
            };
            for i in 0..mem_props.memory_type_count {
                if (type_filter & (1 << i)) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(properties)
                {
                    return Ok(i);
                }
            }
            Err(GpuError::ShaderCompilation {
                msg: "no suitable memory type for FFT buffer".into(),
            })
        }

        fn create_fft_buffer(
            &self,
            dev: &crate::Device,
            size_bytes: u64,
        ) -> Result<(ash::vk::Buffer, ash::vk::DeviceMemory, wgpu::Buffer), GpuError> {
            use ash::vk::Handle;
            use wgpu::hal::api::Vulkan;

            let vk_buffer = unsafe {
                let ci = ash::vk::BufferCreateInfo::default()
                    .size(size_bytes)
                    .usage(
                        ash::vk::BufferUsageFlags::STORAGE_BUFFER
                            | ash::vk::BufferUsageFlags::TRANSFER_SRC
                            | ash::vk::BufferUsageFlags::TRANSFER_DST,
                    )
                    .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);
                self.ash_device.create_buffer(&ci, None).map_err(|e| {
                    GpuError::ShaderCompilation {
                        msg: format!("create_buffer: {e:?}"),
                    }
                })?
            };

            let mem_req = unsafe { self.ash_device.get_buffer_memory_requirements(vk_buffer) };
            let mem_type = self.find_memory_type(
                mem_req.memory_type_bits,
                ash::vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;

            let vk_memory = unsafe {
                let ai = ash::vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_req.size)
                    .memory_type_index(mem_type);
                self.ash_device.allocate_memory(&ai, None).map_err(|e| {
                    GpuError::ShaderCompilation {
                        msg: format!("allocate_memory: {e:?}"),
                    }
                })?
            };

            unsafe {
                self.ash_device
                    .bind_buffer_memory(vk_buffer, vk_memory, 0)
                    .map_err(|e| GpuError::ShaderCompilation {
                        msg: format!("bind_buffer_memory: {e:?}"),
                    })?;
            }

            // Import the VkBuffer into wgpu via create_buffer_from_hal
            let hal_buffer = unsafe {
                wgpu::hal::vulkan::Buffer::from_raw_managed(
                    vk_buffer,
                    vk_memory,
                    0,
                    size_bytes,
                )
            };

            let wgpu_buffer = unsafe {
                dev.wgpu_device().create_buffer_from_hal::<Vulkan>(
                    hal_buffer,
                    &wgpu::BufferDescriptor {
                        label: Some("cartan-gpu::fft::internal"),
                        size: size_bytes,
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                )
            };

            Ok((vk_buffer, vk_memory, wgpu_buffer))
        }

        fn get_or_create_plan(
            &mut self,
            dev: &crate::Device,
            key: PlanKey,
        ) -> Result<&mut VkFftPlan, GpuError> {
            if self.plans.contains_key(&key) {
                return Ok(self.plans.get_mut(&key).unwrap());
            }

            let total_elements = match key.dim {
                1 => key.nx as u64 * key.batch as u64,
                2 => key.nx as u64 * key.ny as u64 * key.batch as u64,
                3 => key.nx as u64 * key.ny as u64 * key.nz as u64,
                _ => unreachable!(),
            };
            let size_bytes = total_elements * std::mem::size_of::<Complex32>() as u64;

            let (vk_buffer, vk_memory, wgpu_buffer) =
                self.create_fft_buffer(dev, size_bytes)?;

            use ash::vk::Handle;
            let mut phys_dev = self.handles.physical_device;
            let mut device = self.handles.device;
            let mut queue = self.handles.queue;
            let mut fence = self.fence.as_raw();
            let mut pool = self.command_pool.as_raw();
            let mut buf_handle = vk_buffer.as_raw();
            let mut buf_size = size_bytes;

            let mut cfg: sys::VkFFTConfiguration = unsafe { std::mem::zeroed() };
            cfg.FFTdim = key.dim as u64;
            cfg.size[0] = key.nx as u64;
            if key.dim >= 2 {
                cfg.size[1] = key.ny as u64;
            }
            if key.dim >= 3 {
                cfg.size[2] = key.nz as u64;
            }
            if key.dim == 1 {
                cfg.numberBatches = key.batch as u64;
            }

            cfg.physicalDevice = &mut phys_dev as *mut u64 as *mut _;
            cfg.device = &mut device as *mut u64 as *mut _;
            cfg.queue = &mut queue as *mut u64 as *mut _;
            cfg.fence = &mut fence as *mut u64 as *mut _;
            cfg.commandPool = &mut pool as *mut u64 as *mut _;
            cfg.isCompilerInitialized = 0;
            cfg.bufferSize = &mut buf_size as *mut u64;
            cfg.buffer = &mut buf_handle as *mut u64 as *mut _;
            cfg.bufferNum = 1;

            let mut plan = Box::new(VkFftPlan {
                app: unsafe { std::mem::zeroed() },
                vk_buffer,
                vk_memory,
                wgpu_buffer,
                size_bytes,
            });
            let result =
                unsafe { sys::cartan_vkfft_init(&mut plan.app as *mut _, cfg) };
            if result != 0 {
                plan.destroy(&self.ash_device);
                return Err(GpuError::VkFftError(result as i32));
            }

            self.plans.insert(key, plan);
            Ok(self.plans.get_mut(&key).unwrap())
        }

        fn launch(
            &mut self,
            dev: &crate::Device,
            key: PlanKey,
            buf: &GpuBuffer<Complex32>,
            direction: FftDirection,
        ) -> Result<(), GpuError> {
            // Ensure plan exists (creates on first call for this key)
            let _ = self.get_or_create_plan(dev, key)?;

            // Extract plan fields we need; avoids holding &mut self across the unsafe block.
            use ash::vk::Handle;
            let plan = self.plans.get(&key).unwrap();
            let plan_size_bytes = plan.size_bytes;
            let plan_vk_buffer_raw = plan.vk_buffer.as_raw();
            let plan_app_ptr = &plan.app as *const sys::VkFFTApplication as *mut sys::VkFFTApplication;
            let plan_wgpu_buffer_ptr = &plan.wgpu_buffer as *const wgpu::Buffer;

            // Copy from user's buffer to internal FFT buffer
            {
                let mut encoder = dev.wgpu_device().create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("fft_copy_in") },
                );
                unsafe {
                    encoder.copy_buffer_to_buffer(buf.raw(), 0, &*plan_wgpu_buffer_ptr, 0, plan_size_bytes);
                }
                dev.wgpu_queue().submit(std::iter::once(encoder.finish()));
                dev.wgpu_device()
                    .poll(wgpu::PollType::wait_indefinitely())
                    .ok();
            }

            eprintln!("[cartan-gpu] FFT launch: key={:?} buf_handle={:#x} size={}", key, plan_vk_buffer_raw, plan_size_bytes);

            // Ensure all wgpu work (copy-in) is complete before VkFFT touches the buffer.
            unsafe {
                let queue = ash::vk::Queue::from_raw(self.handles.queue);
                self.ash_device.queue_wait_idle(queue).ok();
            }

            // VkFFT dispatch: allocate cmd buf, record, submit, wait, free.
            // VkFFTAppend only records commands; we manage the command buffer.
            unsafe {
                use ash::vk::Handle;

                let alloc_info = ash::vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(ash::vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let cmd_bufs = self.ash_device.allocate_command_buffers(&alloc_info)
                    .map_err(|e| GpuError::ShaderCompilation {
                        msg: format!("allocate_command_buffers: {e:?}"),
                    })?;
                let cmd_buf = cmd_bufs[0];

                let begin_info = ash::vk::CommandBufferBeginInfo::default()
                    .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                self.ash_device.begin_command_buffer(cmd_buf, &begin_info)
                    .map_err(|e| GpuError::ShaderCompilation {
                        msg: format!("begin_command_buffer: {e:?}"),
                    })?;

                let mut cmd_buf_raw = cmd_buf.as_raw();
                let mut buf_handle = plan_vk_buffer_raw;
                let mut params: sys::VkFFTLaunchParams = std::mem::zeroed();
                params.commandBuffer = &mut cmd_buf_raw as *mut u64 as *mut _;
                params.buffer = &mut buf_handle as *mut u64 as *mut _;

                let inverse = matches!(direction, FftDirection::Inverse) as i32;
                let code = sys::cartan_vkfft_append(
                    plan_app_ptr,
                    inverse,
                    &mut params as *mut _,
                );
                if code != 0 {
                    return Err(GpuError::VkFftError(code as i32));
                }

                self.ash_device.end_command_buffer(cmd_buf)
                    .map_err(|e| GpuError::ShaderCompilation {
                        msg: format!("end_command_buffer: {e:?}"),
                    })?;

                let submit_info = ash::vk::SubmitInfo::default()
                    .command_buffers(&cmd_bufs);
                let queue = ash::vk::Queue::from_raw(self.handles.queue);
                self.ash_device.reset_fences(&[self.fence]).ok();
                self.ash_device.queue_submit(queue, &[submit_info], self.fence)
                    .map_err(|e| GpuError::ShaderCompilation {
                        msg: format!("queue_submit: {e:?}"),
                    })?;
                self.ash_device.wait_for_fences(&[self.fence], true, u64::MAX)
                    .map_err(|e| GpuError::ShaderCompilation {
                        msg: format!("wait_for_fences: {e:?}"),
                    })?;

                self.ash_device.free_command_buffers(self.command_pool, &cmd_bufs);
            }

            // Copy result back from internal buffer to user's buffer
            {
                let mut encoder = dev.wgpu_device().create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("fft_copy_out") },
                );
                unsafe {
                    encoder.copy_buffer_to_buffer(&*plan_wgpu_buffer_ptr, 0, buf.raw(), 0, plan_size_bytes);
                }
                dev.wgpu_queue().submit(std::iter::once(encoder.finish()));
                dev.wgpu_device()
                    .poll(wgpu::PollType::wait_indefinitely())
                    .ok();
            }

            Ok(())
        }
    }

    impl Fft for VkFftBackend {
        fn fft_1d(
            &mut self,
            dev: &crate::Device,
            buf: &GpuBuffer<Complex32>,
            n: u32,
            batch: u32,
            direction: FftDirection,
        ) -> Result<(), GpuError> {
            assert_eq!(buf.len() as u32, n * batch);
            self.launch(
                dev,
                PlanKey { nx: n, ny: 1, nz: 1, batch, dim: 1 },
                buf,
                direction,
            )
        }

        fn fft_2d(
            &mut self,
            dev: &crate::Device,
            buf: &GpuBuffer<Complex32>,
            nx: u32,
            ny: u32,
            batch: u32,
            direction: FftDirection,
        ) -> Result<(), GpuError> {
            assert_eq!(buf.len() as u32, nx * ny * batch);
            self.launch(
                dev,
                PlanKey { nx, ny, nz: 1, batch, dim: 2 },
                buf,
                direction,
            )
        }

        fn fft_3d(
            &mut self,
            dev: &crate::Device,
            buf: &GpuBuffer<Complex32>,
            nx: u32,
            ny: u32,
            nz: u32,
            direction: FftDirection,
        ) -> Result<(), GpuError> {
            assert_eq!(buf.len() as u32, nx * ny * nz);
            self.launch(
                dev,
                PlanKey { nx, ny, nz, batch: 1, dim: 3 },
                buf,
                direction,
            )
        }
    }
}

#[cfg(feature = "vkfft")]
pub use vkfft_impl::VkFftBackend;
