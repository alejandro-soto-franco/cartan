/* wrapper.h - single include file for bindgen.
 *
 * VkFFT is a header-only library. We include only the Vulkan backend here;
 * CUDA / OpenCL / HIP variants are off via VKFFT_BACKEND=0 define in build.rs.
 *
 * We do NOT include Vulkan headers themselves from Rust's side - consumers
 * of cartan-gpu-sys pass raw Vulkan handles as opaque uint64_t, matching
 * the VkFFT config struct's pointer-to-handle pattern.
 */

#define VKFFT_BACKEND 0  /* Vulkan */

#include "vkFFT/vkFFT.h"
