//! Zero-copy memory shared between the Vulkan and CUDA backends.
//!
//! Allocates exportable `VkDeviceMemory` via Vulkan with
//! `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT`, retrieves a file
//! descriptor via `vkGetMemoryFdKHR`, and imports it into CUDA through
//! [`cudarc::driver::CudaContext::import_external_memory`].
//!
//! `wgpu-hal` 29 enables `VK_KHR_external_memory_fd` when the adapter
//! supports it (see `wgpu-hal/src/vulkan/adapter.rs:1283`), so no fork
//! is required to reach the extension entry points — we load them off
//! the raw ash device handle exposed by hal_vulkan.
//!
//! Same-physical-GPU is assumed: importing memory from a different GPU
//! than CUDA is bound to will either fail outright or silently produce
//! a non-shared mapping. The minimal demo path here doesn't check the
//! adapter↔device UUID; production use should.
//!
//! This module ships the foundation only — a host-visible mirror that
//! both backends can read/write. Promoting it to a full `SharedFftBuffer`
//! that VkFftBackend and CuFftBackend can run FFTs against requires
//! bypassing cudarc's safe `DevicePtrMut<float2>` constraint with raw
//! cuFFT sys calls on the imported `CUdeviceptr`; tracked separately.

#![cfg(all(feature = "vkfft", feature = "cufft", target_os = "linux"))]

use std::fs::File;
use std::os::fd::FromRawFd;
use std::sync::Arc;

use ash::khr::external_memory_fd;
use ash::vk;
use cudarc::driver::{CudaContext, MappedBuffer};

use crate::{Device, GpuError};

/// A block of GPU-resident memory addressable from both Vulkan and CUDA.
///
/// On Linux with NVIDIA's proprietary driver, both APIs end up pointing
/// at the same physical bytes — no host roundtrip, no staging buffer.
pub struct SharedMemory {
    vk_buffer: vk::Buffer,
    vk_memory: vk::DeviceMemory,
    ash_device: ash::Device,
    size_bytes: u64,
    /// CUDA-side view. Owns the underlying `ExternalMemory` internally,
    /// so its `Drop` releases both the mapping and the imported handle
    /// (which in turn drops the fd-owning File on Linux).
    mapped: MappedBuffer,
}

impl SharedMemory {
    /// Allocate `size_bytes` of host-visible, host-coherent memory on the
    /// Vulkan device, expose it as an `OPAQUE_FD` external handle, and
    /// import it into the CUDA context.
    pub fn new(
        vk_dev: &Device,
        cuda_ctx: &Arc<CudaContext>,
        size_bytes: u64,
    ) -> Result<Self, GpuError> {
        use wgpu::hal::api::Vulkan;

        // 1. Raw ash::Device + ash::Instance from wgpu-hal.
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

        // 2. VkBuffer flagged for OPAQUE_FD export.
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
                    msg: format!("create_buffer (shared): {e:?}"),
                })?
        };

        // 3. Memory requirements + host-visible-coherent type.
        let mem_req = unsafe { ash_device.get_buffer_memory_requirements(vk_buffer) };
        use ash::vk::Handle;
        let phys_dev_handle: u64 = vk_dev.raw_vulkan()?.physical_device;
        let phys_dev = vk::PhysicalDevice::from_raw(phys_dev_handle);
        let mem_props = unsafe { ash_instance.get_physical_device_memory_properties(phys_dev) };

        let wanted =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mem_type_idx = (0..mem_props.memory_type_count)
            .find(|&i| {
                let supported = (mem_req.memory_type_bits & (1 << i)) != 0;
                let has_flags = mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(wanted);
                supported && has_flags
            })
            .ok_or_else(|| GpuError::ShaderCompilation {
                msg: "no host-visible coherent memory type supports OPAQUE_FD export".into(),
            })?;

        // 4. Allocate memory with the export-info chained in.
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
                    msg: format!("allocate_memory (shared): {e:?}"),
                })?
        };
        unsafe {
            ash_device
                .bind_buffer_memory(vk_buffer, vk_memory, 0)
                .map_err(|e| GpuError::ShaderCompilation {
                    msg: format!("bind_buffer_memory (shared): {e:?}"),
                })?;
        }

        // 5. Export as a Unix fd via VK_KHR_external_memory_fd.
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

        // 6. Hand the fd to CUDA. cudarc takes ownership of the File on
        //    Linux (per upstream comment in external_memory.rs).
        let file = unsafe { File::from_raw_fd(raw_fd) };
        let ext_mem = unsafe {
            cuda_ctx
                .import_external_memory(file, mem_req.size)
                .map_err(|e| GpuError::CudaError(format!("import_external_memory: {e:?}")))?
        };

        // 7. Map the full range, getting a CUdeviceptr CUDA can read.
        let mapped = ext_mem
            .map_all()
            .map_err(|e| GpuError::CudaError(format!("ExternalMemory::map_all: {e:?}")))?;

        Ok(Self {
            vk_buffer,
            vk_memory,
            ash_device,
            size_bytes,
            mapped,
        })
    }

    /// Map the Vulkan side host-visible and copy `bytes` in.
    ///
    /// Panics if `bytes.len()` exceeds the allocation. Caller is
    /// responsible for ensuring no concurrent reads from CUDA.
    pub fn write_host_bytes(&self, bytes: &[u8]) -> Result<(), GpuError> {
        assert!(bytes.len() as u64 <= self.size_bytes);
        unsafe {
            let ptr = self
                .ash_device
                .map_memory(
                    self.vk_memory,
                    0,
                    self.size_bytes,
                    vk::MemoryMapFlags::empty(),
                )
                .map_err(|e| GpuError::ShaderCompilation {
                    msg: format!("map_memory: {e:?}"),
                })?;
            core::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast::<u8>(), bytes.len());
            self.ash_device.unmap_memory(self.vk_memory);
        }
        Ok(())
    }

    /// Borrow the CUDA-side view of the shared memory.
    pub fn cuda_view(&self) -> &MappedBuffer {
        &self.mapped
    }

    pub fn len(&self) -> u64 {
        self.size_bytes
    }

    pub fn is_empty(&self) -> bool {
        self.size_bytes == 0
    }
}

impl Drop for SharedMemory {
    fn drop(&mut self) {
        // The MappedBuffer + ExternalMemory drop automatically on the CUDA
        // side; here we tear down the Vulkan resources.
        unsafe {
            self.ash_device.destroy_buffer(self.vk_buffer, None);
            self.ash_device.free_memory(self.vk_memory, None);
        }
    }
}
