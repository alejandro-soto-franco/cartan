/* vkfft_shim.c - defines the extern-C wrappers declared in wrapper.h.
 *
 * Including the VkFFT header here materializes the `static inline` functions
 * as concrete symbols in this translation unit. The wrapper functions call
 * straight through; they exist only to give the linker something to find.
 */

#include "wrapper.h"

int cartan_vkfft_version(void) {
    return VkFFTGetVersion();
}

VkFFTResult cartan_vkfft_init(VkFFTApplication* app, VkFFTConfiguration cfg) {
    return initializeVkFFT(app, cfg);
}

VkFFTResult cartan_vkfft_append(VkFFTApplication* app, int inverse, VkFFTLaunchParams* params) {
    return VkFFTAppend(app, inverse, params);
}

void cartan_vkfft_delete(VkFFTApplication* app) {
    deleteVkFFT(app);
}
