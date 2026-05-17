//! FFT abstraction and VkFFT-backed implementation.
//!
//! The `Fft` trait is the seam that lets us swap VkFFT for a future pure-Rust
//! backend without cartan-em churn. v0.1 ships one backend: `VkFftBackend`.

use crate::{GpuBuffer, GpuError};
use num_complex::Complex32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FftDirection {
    Forward,
    Inverse,
}

/// Backend-agnostic FFT operations.
///
/// `Buffer` is the backend-specific Complex32 storage type. Each backend
/// constructor takes a device once and captures whatever wgpu/cudarc
/// handles it needs, so call-site code is uniform across backends.
///
/// Semantic guarantee: forward then inverse is identity on every backend.
/// The Vulkan path achieves this via VkFFT's `cfg.normalize = 1`; the CUDA
/// path explicitly post-scales by `1/N` with cuBLAS `cublasSscal_v2`.
pub trait Fft {
    type Buffer;

    fn fft_1d(
        &mut self,
        buf: &mut Self::Buffer,
        n: u32,
        batch: u32,
        direction: FftDirection,
    ) -> Result<(), GpuError>;

    fn fft_2d(
        &mut self,
        buf: &mut Self::Buffer,
        nx: u32,
        ny: u32,
        batch: u32,
        direction: FftDirection,
    ) -> Result<(), GpuError>;

    fn fft_3d(
        &mut self,
        buf: &mut Self::Buffer,
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
        /// Heap-pinned backing for the pointer-to-handle fields that VkFFT
        /// references throughout `app`'s lifetime. Must be dropped after
        /// `cartan_vkfft_delete(&mut app)` so VkFFT never reads freed memory.
        backing: Box<sys::CartanVkFftBacking>,
        vk_buffer: ash::vk::Buffer,
        wgpu_buffer: wgpu::Buffer,
        size_bytes: u64,
    }

    impl VkFftPlan {
        /// Tear down VkFFT-side resources. The VkBuffer + DeviceMemory are
        /// owned by `wgpu_buffer` (via `Buffer::from_raw_managed`) and will
        /// be destroyed when this struct drops — destroying them here would
        /// double-free and segfault at process teardown.
        fn destroy(&mut self, _ash_device: &ash::Device) {
            unsafe {
                sys::cartan_vkfft_delete(&mut self.app as *mut _);
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
        wgpu_device: wgpu::Device,
        wgpu_queue: wgpu::Queue,
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
                wgpu_device: dev.wgpu_device().clone(),
                wgpu_queue: dev.wgpu_queue().clone(),
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
            size_bytes: u64,
        ) -> Result<(ash::vk::Buffer, wgpu::Buffer), GpuError> {
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
                self.wgpu_device.create_buffer_from_hal::<Vulkan>(
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

            // `vk_memory` is now owned by `wgpu_buffer` via Buffer::from_raw_managed
            // and will be freed when the wgpu::Buffer drops. We deliberately do not
            // keep a separate handle on the Rust side to avoid accidental double-free.
            let _ = vk_memory;

            Ok((vk_buffer, wgpu_buffer))
        }

        fn get_or_create_plan(
            &mut self,
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

            let (vk_buffer, wgpu_buffer) = self.create_fft_buffer(size_bytes)?;

            use ash::vk::Handle;
            let mut plan = Box::new(VkFftPlan {
                app: unsafe { std::mem::zeroed() },
                backing: Box::new(unsafe { std::mem::zeroed() }),
                vk_buffer,
                wgpu_buffer,
                size_bytes,
            });

            // The shim writes into `plan.backing` so the pointer-to-handle
            // fields VkFFT stores in `plan.app` remain valid for the full
            // lifetime of the plan, not just this call.
            let result = unsafe {
                sys::cartan_vkfft_plan(
                    &mut plan.app as *mut _,
                    &mut *plan.backing as *mut _,
                    self.handles.physical_device,
                    self.handles.device,
                    self.handles.queue,
                    self.command_pool.as_raw(),
                    self.fence.as_raw(),
                    plan.vk_buffer.as_raw(),
                    size_bytes,
                    key.dim as u32,
                    key.nx as u64,
                    key.ny as u64,
                    key.nz as u64,
                    key.batch as u64,
                )
            };
            if result != 0 {
                plan.destroy(&self.ash_device);
                return Err(GpuError::VkFftError(result as i32));
            }

            self.plans.insert(key, plan);
            Ok(self.plans.get_mut(&key).unwrap())
        }

        fn launch(
            &mut self,
            key: PlanKey,
            buf: &GpuBuffer<Complex32>,
            direction: FftDirection,
        ) -> Result<(), GpuError> {
            let _ = self.get_or_create_plan(key)?;

            use ash::vk::Handle;
            let plan = self.plans.get(&key).unwrap();
            let plan_size_bytes = plan.size_bytes;
            let plan_vk_buffer_raw = plan.vk_buffer.as_raw();
            let plan_app_ptr =
                &plan.app as *const sys::VkFFTApplication as *mut sys::VkFFTApplication;
            let plan_wgpu_buffer_ptr = &plan.wgpu_buffer as *const wgpu::Buffer;

            // wgpu copy-in: user buffer -> internal FFT buffer.
            {
                let mut encoder = self.wgpu_device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("fft_copy_in") },
                );
                unsafe {
                    encoder.copy_buffer_to_buffer(
                        buf.raw(),
                        0,
                        &*plan_wgpu_buffer_ptr,
                        0,
                        plan_size_bytes,
                    );
                }
                self.wgpu_queue.submit(std::iter::once(encoder.finish()));
                self.wgpu_device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .ok();
            }

            // Drain everything previously submitted to the shared queue so VkFFT
            // sees a quiesced buffer when it allocates its descriptor sets.
            unsafe {
                let queue = ash::vk::Queue::from_raw(self.handles.queue);
                self.ash_device.queue_wait_idle(queue).ok();
            }

            // Single FFI call: shim allocates a command buffer from our pool,
            // records VkFFTAppend, submits, waits on our fence, and frees.
            let inverse = matches!(direction, FftDirection::Inverse) as i32;
            let code = unsafe {
                sys::cartan_vkfft_exec(
                    plan_app_ptr,
                    self.handles.device,
                    self.handles.queue,
                    self.command_pool.as_raw(),
                    self.fence.as_raw(),
                    plan_vk_buffer_raw,
                    inverse,
                )
            };
            if code != 0 {
                return Err(GpuError::VkFftError(code as i32));
            }

            // wgpu copy-out: internal FFT buffer -> user buffer.
            {
                let mut encoder = self.wgpu_device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("fft_copy_out") },
                );
                unsafe {
                    encoder.copy_buffer_to_buffer(
                        &*plan_wgpu_buffer_ptr,
                        0,
                        buf.raw(),
                        0,
                        plan_size_bytes,
                    );
                }
                self.wgpu_queue.submit(std::iter::once(encoder.finish()));
                self.wgpu_device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .ok();
            }

            Ok(())
        }
    }

    impl Fft for VkFftBackend {
        type Buffer = GpuBuffer<Complex32>;

        fn fft_1d(
            &mut self,
            buf: &mut Self::Buffer,
            n: u32,
            batch: u32,
            direction: FftDirection,
        ) -> Result<(), GpuError> {
            assert_eq!(buf.len() as u32, n * batch);
            self.launch(
                PlanKey { nx: n, ny: 1, nz: 1, batch, dim: 1 },
                buf,
                direction,
            )
        }

        fn fft_2d(
            &mut self,
            buf: &mut Self::Buffer,
            nx: u32,
            ny: u32,
            batch: u32,
            direction: FftDirection,
        ) -> Result<(), GpuError> {
            assert_eq!(buf.len() as u32, nx * ny * batch);
            self.launch(
                PlanKey { nx, ny, nz: 1, batch, dim: 2 },
                buf,
                direction,
            )
        }

        fn fft_3d(
            &mut self,
            buf: &mut Self::Buffer,
            nx: u32,
            ny: u32,
            nz: u32,
            direction: FftDirection,
        ) -> Result<(), GpuError> {
            assert_eq!(buf.len() as u32, nx * ny * nz);
            self.launch(
                PlanKey { nx, ny, nz, batch: 1, dim: 3 },
                buf,
                direction,
            )
        }
    }
}

#[cfg(feature = "vkfft")]
pub use vkfft_impl::VkFftBackend;
