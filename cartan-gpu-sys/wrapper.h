/* wrapper.h - high-level C wrappers around VkFFT for Rust FFI.
 *
 * VkFFT uses a pointer-to-handle ABI for Vulkan objects and `static inline`
 * functions. Both are hostile to direct bindgen. These wrappers accept plain
 * handle values (uint64_t), do the pointer casting internally, and manage
 * the VkCommandBuffer lifecycle that VkFFT expects from the caller.
 *
 * Rust calls cartan_vkfft_* functions with raw handle integers;
 * no pointer-to-pointer gymnastics leak into the Rust side.
 */

#ifndef CARTAN_VKFFT_WRAPPER_H
#define CARTAN_VKFFT_WRAPPER_H

#define VKFFT_BACKEND 0  /* Vulkan */

#include "vkFFT/vkFFT.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int cartan_vkfft_version(void);

void cartan_vkfft_delete(VkFFTApplication* app);

/* Create a 1D/2D/3D FFT plan. All handles are passed as raw uint64_t values.
 * Returns VKFFT_SUCCESS (0) on success, nonzero VkFFTResult on failure. */
int cartan_vkfft_plan(
    VkFFTApplication* app,         /* out: zeroed VkFFTApplication to initialize */
    uint64_t physical_device,      /* VkPhysicalDevice handle */
    uint64_t device,               /* VkDevice handle */
    uint64_t queue,                /* VkQueue handle */
    uint64_t command_pool,         /* VkCommandPool handle */
    uint64_t fence,                /* VkFence handle */
    uint64_t buffer,               /* VkBuffer handle */
    uint64_t buffer_size_bytes,    /* buffer size in bytes */
    uint32_t dim,                  /* 1, 2, or 3 */
    uint64_t size_x,
    uint64_t size_y,               /* ignored if dim < 2 */
    uint64_t size_z,               /* ignored if dim < 3 */
    uint64_t batch                 /* number of batches (1D only, else 1) */
);

/* Execute an FFT on the given buffer using the initialized plan.
 * Allocates a command buffer, records VkFFTAppend, submits, waits.
 * Returns VKFFT_SUCCESS (0) on success. */
int cartan_vkfft_exec(
    VkFFTApplication* app,
    uint64_t device,
    uint64_t queue,
    uint64_t command_pool,
    uint64_t fence,
    uint64_t buffer,
    int inverse                    /* 0 = forward, 1 = inverse */
);

#ifdef __cplusplus
}
#endif

#endif /* CARTAN_VKFFT_WRAPPER_H */
