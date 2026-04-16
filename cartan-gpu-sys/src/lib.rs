//! Raw FFI bindings to the vendored VkFFT C++ library.
//!
//! This crate is internal plumbing for `cartan-gpu`. It exposes only the
//! subset of VkFFT needed to initialise an FFT plan against raw Vulkan
//! handles, append dispatches to a command buffer, and release resources.
//!
//! Safety: everything here is `unsafe` by construction. The `cartan-gpu`
//! crate wraps these calls in a typed API; downstream cartan crates should
//! never depend on `cartan-gpu-sys` directly.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(improper_ctypes)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

/// Pinned VkFFT submodule tag for this build.
pub const VKFFT_VENDOR_TAG: &str = "v1.3.4";

/// Return the VkFFT runtime version as a packed integer.
///
/// Encoding is VkFFT's own: `major * 10000 + minor * 100 + patch`. For
/// v1.3.4 this is 10304.
pub fn vkfft_runtime_version() -> u64 {
    // SAFETY: `cartan_vkfft_version` takes no arguments and returns a plain int.
    unsafe { cartan_vkfft_version() as u64 }
}

#[cfg(test)]
mod sanity {
    use super::*;

    #[test]
    fn vkfft_linked_and_version_readable() {
        let v = vkfft_runtime_version();
        assert!(v >= 10000, "VkFFT version suspiciously low: {v}");
        assert_eq!(VKFFT_VENDOR_TAG, "v1.3.4");
    }
}
