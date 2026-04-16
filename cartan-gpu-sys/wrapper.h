/* wrapper.h - declares extern wrappers around VkFFT's static inline API.
 *
 * VkFFT is header-only with `static inline` functions, which bindgen cannot
 * emit by default. We wrap the handful of entry points we need as regular
 * extern-C functions so bindgen produces proper FFI declarations and `cc`
 * emits real symbols in libvkfft.a.
 *
 * Only the cartan_vkfft_* wrappers are part of the stable bindings surface;
 * the VkFFT struct types (VkFFTConfiguration, VkFFTApplication, etc.) are
 * re-exported as-is because they're concrete structs, not inline functions.
 */

#ifndef CARTAN_VKFFT_WRAPPER_H
#define CARTAN_VKFFT_WRAPPER_H

#define VKFFT_BACKEND 0  /* Vulkan */

#include "vkFFT/vkFFT.h"

#ifdef __cplusplus
extern "C" {
#endif

int cartan_vkfft_version(void);

VkFFTResult cartan_vkfft_init(VkFFTApplication* app, VkFFTConfiguration cfg);

VkFFTResult cartan_vkfft_append(VkFFTApplication* app, int inverse, VkFFTLaunchParams* params);

void cartan_vkfft_delete(VkFFTApplication* app);

#ifdef __cplusplus
}
#endif

#endif /* CARTAN_VKFFT_WRAPPER_H */
