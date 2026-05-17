/* vkfft_shim.c - defines the extern-C wrappers declared in wrapper.h.
 *
 * The pointer-to-handle ABI that VkFFT uses for Vulkan objects is handled
 * entirely here. Rust passes raw uint64_t handle values; this file casts
 * them to the appropriate VkXxx types and takes their addresses where
 * VkFFT expects pointer-to-handle.
 */

#include "wrapper.h"

int cartan_vkfft_version(void) {
    return VkFFTGetVersion();
}

void cartan_vkfft_delete(VkFFTApplication* app) {
    deleteVkFFT(app);
}

int cartan_vkfft_plan(
    VkFFTApplication* app,
    CartanVkFftBacking* backing,
    uint64_t physical_device,
    uint64_t device,
    uint64_t queue,
    uint64_t command_pool,
    uint64_t fence,
    uint64_t buffer,
    uint64_t buffer_size_bytes,
    uint32_t dim,
    uint64_t size_x,
    uint64_t size_y,
    uint64_t size_z,
    uint64_t batch)
{
    /* Write handles into caller-owned backing so VkFFT's pointer fields
     * stay valid for the full app lifetime, not just this call. */
    backing->phys        = physical_device;
    backing->device      = device;
    backing->queue       = queue;
    backing->pool        = command_pool;
    backing->fence       = fence;
    backing->buffer      = buffer;
    backing->buffer_size = buffer_size_bytes;

    VkFFTConfiguration cfg = {0};
    cfg.FFTdim = dim;
    cfg.size[0] = size_x;
    if (dim >= 2) cfg.size[1] = size_y;
    if (dim >= 3) cfg.size[2] = size_z;
    if (dim == 1) cfg.numberBatches = batch;

    /* Vulkan dispatchable handles (VkPhysicalDevice, VkDevice, VkQueue) and
     * non-dispatchable handles (VkCommandPool, VkFence, VkBuffer) are all
     * 8 bytes on 64-bit Linux. Casting through a uint64_t backing field is
     * safe under the same alignment. */
    cfg.physicalDevice = (VkPhysicalDevice*)&backing->phys;
    cfg.device         = (VkDevice*)&backing->device;
    cfg.queue          = (VkQueue*)&backing->queue;
    cfg.commandPool    = (VkCommandPool*)&backing->pool;
    cfg.fence          = (VkFence*)&backing->fence;

    cfg.buffer         = (VkBuffer*)&backing->buffer;
    cfg.bufferSize     = &backing->buffer_size;
    cfg.bufferNum      = 1;

    cfg.isCompilerInitialized = 0;

    /* Normalize the inverse so forward then inverse is identity. Without
     * this VkFFT returns N*input from the inverse and round-trip tests
     * fail by exactly the FFT size. */
    cfg.normalize = 1;

    return (int)initializeVkFFT(app, cfg);
}

int cartan_vkfft_exec(
    VkFFTApplication* app,
    uint64_t device,
    uint64_t queue,
    uint64_t command_pool,
    uint64_t fence,
    uint64_t buffer,
    int inverse)
{
    VkDevice      vk_device = (VkDevice)(uintptr_t)device;
    VkQueue       vk_queue  = (VkQueue)(uintptr_t)queue;
    VkCommandPool vk_pool   = (VkCommandPool)(uintptr_t)command_pool;
    VkFence       vk_fence  = (VkFence)(uintptr_t)fence;
    VkBuffer      vk_buf    = (VkBuffer)(uintptr_t)buffer;

    VkCommandBufferAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd_buf;
    VkResult vk_res = vkAllocateCommandBuffers(vk_device, &alloc_info, &cmd_buf);
    if (vk_res != VK_SUCCESS) return (int)VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS;

    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vk_res = vkBeginCommandBuffer(cmd_buf, &begin_info);
    if (vk_res != VK_SUCCESS) return (int)VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER;

    VkFFTLaunchParams params = {0};
    params.commandBuffer = &cmd_buf;
    params.buffer = &vk_buf;

    VkFFTResult fft_res = VkFFTAppend(app, inverse, &params);
    if (fft_res != VKFFT_SUCCESS) {
        vkEndCommandBuffer(cmd_buf);
        vkFreeCommandBuffers(vk_device, vk_pool, 1, &cmd_buf);
        return (int)fft_res;
    }

    vk_res = vkEndCommandBuffer(cmd_buf);
    if (vk_res != VK_SUCCESS) return (int)VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER;

    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd_buf,
    };
    vkResetFences(vk_device, 1, &vk_fence);
    vk_res = vkQueueSubmit(vk_queue, 1, &submit_info, vk_fence);
    if (vk_res != VK_SUCCESS) return (int)VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE;

    vk_res = vkWaitForFences(vk_device, 1, &vk_fence, VK_TRUE, (uint64_t)1e12);
    if (vk_res != VK_SUCCESS) return (int)VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES;

    vkFreeCommandBuffers(vk_device, vk_pool, 1, &cmd_buf);
    return (int)VKFFT_SUCCESS;
}
