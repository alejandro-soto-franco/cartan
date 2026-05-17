//! Device - cartan-gpu's entry point. Wraps a `wgpu::Instance / Adapter / Device / Queue`,
//! enforces Vulkan backend (required for VkFFT integration), and exposes handle
//! accessors used internally by the FFT backend.

use crate::GpuError;

/// Owned GPU device. Held for the lifetime of a cartan-gpu session.
///
/// All wgpu fields are reference-counted internally so `Clone` is cheap;
/// useful for stashing a device alongside backend objects that need to
/// allocate buffers later without re-receiving the device by reference.
#[derive(Clone)]
pub struct Device {
    pub(crate) instance: wgpu::Instance,
    pub(crate) adapter: wgpu::Adapter,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
}

impl Device {
    /// Request a Vulkan adapter and device with defaults suitable for compute.
    /// Returns `GpuError::NoAdapter` if no Vulkan adapter is available.
    pub fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::empty(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            backend_options: wgpu::BackendOptions::default(),
            display: None,
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|_| GpuError::NoAdapter)?;

        if adapter.get_info().backend != wgpu::Backend::Vulkan {
            return Err(GpuError::NotVulkan(adapter.get_info().backend));
        }

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("cartan-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            },
        ))?;

        Ok(Self { instance, adapter, device, queue })
    }

    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    pub fn wgpu_device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn wgpu_queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Vulkan `VK_KHR_external_memory_capabilities` device UUID.
    ///
    /// Used to verify that a CUDA context bound for shared-memory
    /// interop is bound to the same physical GPU as this Vulkan device.
    /// Returns 16 bytes from `VkPhysicalDeviceIDProperties.deviceUUID`.
    #[cfg(feature = "vkfft")]
    pub fn vulkan_device_uuid(&self) -> Result<[u8; 16], GpuError> {
        use ash::vk::Handle;
        use wgpu::hal::api::Vulkan;

        let handles = self.raw_vulkan()?;
        let phys_dev = ash::vk::PhysicalDevice::from_raw(handles.physical_device);

        // We need vkGetPhysicalDeviceProperties2 off the raw instance.
        let ash_instance = unsafe {
            let hal_instance = self
                .instance
                .as_hal::<Vulkan>()
                .ok_or(GpuError::VulkanHandlesUnavailable)?;
            hal_instance.shared_instance().raw_instance().clone()
        };

        let mut id_props = ash::vk::PhysicalDeviceIDProperties::default();
        let mut props2 = ash::vk::PhysicalDeviceProperties2::default().push_next(&mut id_props);
        unsafe {
            ash_instance.get_physical_device_properties2(phys_dev, &mut props2);
        }
        Ok(id_props.device_uuid)
    }
}
