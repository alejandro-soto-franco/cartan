# cartan-gpu Changelog

## [0.6.0] - 2026-05-26

### Breaking
- FFT layer moved to the standalone `gpufft` crate. `Fft`, `FftDirection`,
  `VkFftBackend`, `CuFftBackend`, `UniBuffer`, `UniFftBackend`,
  `SharedMemory`, and `SharedFftBuffer` are removed. Migrate to
  `gpufft::{Direction, vulkan::VulkanBackend, cuda::CudaBackend, shared::SharedFftBuffer}`
  and call `device.plan_c2c::<Complex32>(&PlanDesc { ... })`.
- `cartan-gpu-sys` crate removed; gpufft's `gpufft-vulkan-sys` is now the
  canonical VkFFT FFI shim.
- `GpuError` variants `VkFftError`, `VkFftDisabled`, `VulkanHandlesUnavailable`,
  `CudaError`, `BackendMismatch`, and `GpuUuidMismatch` removed.
- `Device::vulkan_device_uuid()` and `Device::raw_vulkan()` removed.
- Default feature set is now empty (was `vkfft`).

### Retained
- `Device`, `GpuBuffer<T>`, `Kernel` (wgpu compute primitives) are unchanged.
- With the `fft` feature, gpufft is re-exported as `cartan_gpu::gpufft`.
- `GpuError` variants `NoAdapter`, `NotVulkan`, `DeviceCreation`,
  `BufferSize`, and `ShaderCompilation` are retained.
