//! Device - cartan-gpu's entry point. Wraps a `wgpu::Instance / Adapter / Device / Queue`
//! and enforces Vulkan backend for compute work.

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
}
