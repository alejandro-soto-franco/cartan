//! Device-local FFT-ready memory shared between Vulkan and CUDA.
//!
//! Like [`crate::SharedMemory`] but `DEVICE_LOCAL` instead of host-visible,
//! so it is usable as the storage buffer for both VkFFT and cuFFT plans.
//! Host transfers go through CUDA's `cuMemcpyHtoD`/`cuMemcpyDtoH` against
//! the raw `CUdeviceptr` — no Vulkan staging buffer needed.
//!
//! Pair this with [`crate::VkFftBackend::fft_1d_shared`] and
//! [`crate::CuFftBackend::fft_1d_shared`] (and the 2D/3D variants) to
//! run forward and inverse transforms on the exact same physical
//! allocation, on either backend, without copies.

#![cfg(all(feature = "vkfft", feature = "cufft", target_os = "linux"))]

use std::fs::File;
use std::os::fd::FromRawFd;
use std::sync::Arc;

use ash::khr::external_memory_fd;
use ash::vk;
use cudarc::driver::{CudaContext, DevicePtr, MappedBuffer};
use num_complex::Complex32;

use crate::{Device, GpuError};

/// FFT-ready memory addressable from both Vulkan and CUDA.
pub struct SharedFftBuffer {
    vk_buffer: vk::Buffer,
    vk_memory: vk::DeviceMemory,
    ash_device: ash::Device,
    len_complex: usize,
    mapped: MappedBuffer,
    stream: Arc<cudarc::driver::CudaStream>,
}

impl SharedFftBuffer {
    /// Allocate `len_complex` Complex32 elements as DEVICE_LOCAL memory,
    /// export the fd, and import into the CUDA context.
    pub fn new(
        vk_dev: &Device,
        cuda_ctx: &Arc<CudaContext>,
        len_complex: usize,
    ) -> Result<Self, GpuError> {
        use wgpu::hal::api::Vulkan;

        let size_bytes = (len_complex * core::mem::size_of::<Complex32>()) as u64;

        let (ash_device, ash_instance) = unsafe {
            let hal_device = vk_dev
                .device
                .as_hal::<Vulkan>()
                .ok_or(GpuError::VulkanHandlesUnavailable)?;
            let device = hal_device.raw_device().clone();
            let hal_instance = vk_dev
                .instance
                .as_hal::<Vulkan>()
                .ok_or(GpuError::VulkanHandlesUnavailable)?;
            let instance = hal_instance.shared_instance().raw_instance().clone();
            (device, instance)
        };

        // Buffer flagged for OPAQUE_FD export, usable as storage and copy
        // src/dst (VkFFT requires STORAGE_BUFFER).
        let mut external_buf_info = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let buf_info = vk::BufferCreateInfo::default()
            .size(size_bytes)
            .usage(
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external_buf_info);
        let vk_buffer = unsafe {
            ash_device
                .create_buffer(&buf_info, None)
                .map_err(|e| GpuError::ShaderCompilation {
                    msg: format!("create_buffer (shared-fft): {e:?}"),
                })?
        };

        let mem_req = unsafe { ash_device.get_buffer_memory_requirements(vk_buffer) };
        use ash::vk::Handle;
        let phys_dev_handle: u64 = vk_dev.raw_vulkan()?.physical_device;
        let phys_dev = vk::PhysicalDevice::from_raw(phys_dev_handle);
        let mem_props = unsafe { ash_instance.get_physical_device_memory_properties(phys_dev) };

        let mem_type_idx = (0..mem_props.memory_type_count)
            .find(|&i| {
                let supported = (mem_req.memory_type_bits & (1 << i)) != 0;
                let device_local = mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
                supported && device_local
            })
            .ok_or_else(|| GpuError::ShaderCompilation {
                msg: "no device-local memory type supports OPAQUE_FD export".into(),
            })?;

        let mut export_info = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_req.size)
            .memory_type_index(mem_type_idx)
            .push_next(&mut export_info);
        let vk_memory = unsafe {
            ash_device
                .allocate_memory(&alloc_info, None)
                .map_err(|e| GpuError::ShaderCompilation {
                    msg: format!("allocate_memory (shared-fft): {e:?}"),
                })?
        };
        unsafe {
            ash_device
                .bind_buffer_memory(vk_buffer, vk_memory, 0)
                .map_err(|e| GpuError::ShaderCompilation {
                    msg: format!("bind_buffer_memory (shared-fft): {e:?}"),
                })?;
        }

        let ext_mem_fd_loader = external_memory_fd::Device::new(&ash_instance, &ash_device);
        let fd_info = vk::MemoryGetFdInfoKHR::default()
            .memory(vk_memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let raw_fd = unsafe {
            ext_mem_fd_loader
                .get_memory_fd(&fd_info)
                .map_err(|e| GpuError::ShaderCompilation {
                    msg: format!("vkGetMemoryFdKHR: {e:?}"),
                })?
        };

        let file = unsafe { File::from_raw_fd(raw_fd) };
        let ext_mem = unsafe {
            cuda_ctx
                .import_external_memory(file, mem_req.size)
                .map_err(|e| GpuError::CudaError(format!("import_external_memory: {e:?}")))?
        };
        let mapped = ext_mem
            .map_all()
            .map_err(|e| GpuError::CudaError(format!("ExternalMemory::map_all: {e:?}")))?;

        let stream = cuda_ctx.default_stream();

        Ok(Self {
            vk_buffer,
            vk_memory,
            ash_device,
            len_complex,
            mapped,
            stream,
        })
    }

    /// Raw `VkBuffer` handle for the Vulkan FFT path.
    pub fn vk_buffer(&self) -> vk::Buffer {
        self.vk_buffer
    }

    /// Raw `CUdeviceptr` to the start of the buffer.
    ///
    /// Use as `*mut cufftComplex` / `*mut f32` when bypassing cudarc's
    /// safe wrappers (cuFFT exec, cuBLAS scale, raw memcpy).
    pub fn cuda_ptr(&self) -> cudarc::driver::sys::CUdeviceptr {
        let (ptr, _record) = self.mapped.device_ptr(&self.stream);
        ptr
    }

    /// Number of Complex32 elements.
    pub fn len(&self) -> usize {
        self.len_complex
    }

    pub fn is_empty(&self) -> bool {
        self.len_complex == 0
    }

    /// Upload `host` into the shared memory via CUDA's `cuMemcpyHtoD`.
    /// Synchronous (the default stream is the one CUDA imports use).
    pub fn upload(&self, host: &[Complex32]) -> Result<(), GpuError> {
        assert_eq!(host.len(), self.len_complex, "host slice length must match buffer length");
        let host_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(host.as_ptr().cast::<u8>(), std::mem::size_of_val(host))
        };
        unsafe { cudarc::driver::result::memcpy_htod_sync(self.cuda_ptr(), host_bytes) }
            .map_err(|e| GpuError::CudaError(format!("memcpy_htod_sync: {e:?}")))
    }

    /// Download the buffer's contents back to a host `Vec<Complex32>`.
    pub fn download(&self) -> Result<Vec<Complex32>, GpuError> {
        let mut out = vec![Complex32::new(0.0, 0.0); self.len_complex];
        let out_bytes: &mut [u8] = unsafe {
            core::slice::from_raw_parts_mut(
                out.as_mut_ptr().cast::<u8>(),
                std::mem::size_of_val(out.as_slice()),
            )
        };
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(out_bytes, self.cuda_ptr()) }
            .map_err(|e| GpuError::CudaError(format!("memcpy_dtoh_sync: {e:?}")))?;
        Ok(out)
    }
}

impl Drop for SharedFftBuffer {
    fn drop(&mut self) {
        unsafe {
            self.ash_device.destroy_buffer(self.vk_buffer, None);
            self.ash_device.free_memory(self.vk_memory, None);
        }
    }
}
