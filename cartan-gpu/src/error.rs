//! Error type for cartan-gpu. All fallible public APIs return `Result<_, GpuError>`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no adapter available for requested backend")]
    NoAdapter,

    #[error("requested Vulkan backend but adapter is {0:?}")]
    NotVulkan(wgpu::Backend),

    #[error("device creation failed: {0}")]
    DeviceCreation(#[from] wgpu::RequestDeviceError),

    #[error("buffer size mismatch: expected {expected} elements, got {got}")]
    BufferSize { expected: usize, got: usize },

    #[error("shader module creation failed: {msg}")]
    ShaderCompilation { msg: String },

    #[error("VkFFT error code {0}")]
    VkFftError(i32),

    #[error("feature `vkfft` not enabled at compile time")]
    VkFftDisabled,

    #[error("operation requires raw Vulkan handles; adapter is not Vulkan")]
    VulkanHandlesUnavailable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_displays() {
        let e = GpuError::BufferSize { expected: 10, got: 8 };
        assert_eq!(
            format!("{e}"),
            "buffer size mismatch: expected 10 elements, got 8"
        );
    }
}
