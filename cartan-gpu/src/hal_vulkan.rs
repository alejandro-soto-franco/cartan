//! Raw Vulkan handle extraction via `wgpu-hal`.
//!
//! VkFFT requires raw `VkDevice`, `VkPhysicalDevice`, `VkQueue`, and the
//! queue family index. wgpu hides these behind its abstract `Device`; we
//! reach in via `wgpu::hal::api::Vulkan` to pull the raw handles.
//!
//! This is the one unsafe crossing between wgpu and VkFFT; everything else
//! goes through typed wrappers.
//!
//! # Safety invariants
//!
//! - The `Device` that produced these handles must outlive the handles.
//! - Handles here are opaque `u64`s; VkFFT's C ABI takes pointer-to-handle,
//!   which consumers must materialize appropriately.
//! - cartan-gpu never destroys these handles; their lifetime is owned by wgpu.

#![cfg(feature = "vkfft")]

use crate::{Device, GpuError};
use ash::vk::Handle;

#[derive(Debug, Clone, Copy)]
pub struct RawVulkanHandles {
    pub physical_device: u64,
    pub device: u64,
    pub queue: u64,
    pub queue_family_index: u32,
    pub instance: u64,
}

impl Device {
    /// Extract raw Vulkan handles. Fails if the adapter is not Vulkan
    /// or handles cannot be obtained.
    pub fn raw_vulkan(&self) -> Result<RawVulkanHandles, GpuError> {
        use wgpu::hal::api::Vulkan;

        // SAFETY: the Device's backend is verified Vulkan in Device::new(),
        // so as_hal returning Some is expected. Handles are copied before the
        // Deref guards go out of scope.
        let phys_dev: u64;
        let instance: u64;
        unsafe {
            let hal_adapter = self
                .adapter
                .as_hal::<Vulkan>()
                .ok_or(GpuError::VulkanHandlesUnavailable)?;
            phys_dev = hal_adapter.raw_physical_device().as_raw();

            // instance is available on the Adapter's shared_instance handle via its impl
            let hal_instance = self
                .instance
                .as_hal::<Vulkan>()
                .ok_or(GpuError::VulkanHandlesUnavailable)?;
            instance = hal_instance.shared_instance().raw_instance().handle().as_raw();
        }

        // SAFETY: same backend invariant. device/queue handles taken via the
        // vulkan::Device view, which keeps both handles coherent.
        let (device, queue, queue_family_index) = unsafe {
            let hal_device = self
                .device
                .as_hal::<Vulkan>()
                .ok_or(GpuError::VulkanHandlesUnavailable)?;
            let d = hal_device.raw_device().handle().as_raw();
            let q = hal_device.raw_queue().as_raw();
            let qf = hal_device.queue_family_index();
            (d, q, qf)
        };

        Ok(RawVulkanHandles {
            physical_device: phys_dev,
            device,
            queue,
            queue_family_index,
            instance,
        })
    }
}
