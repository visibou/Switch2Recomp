// The Vulkan entry points the backend uses, loaded at run time from vulkan-1.dll (VK_NO_PROTOTYPES): nxrt.exe starts on a PC without Vulkan
// (--gfx null), and device-level calls go straight to the driver instead of through the loader's trampolines.
#pragma once
#define VK_NO_PROTOTYPES
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <vulkan/vulkan.h>

#define VKF_GLOBAL(X) X(vkCreateInstance) X(vkEnumerateInstanceExtensionProperties) X(vkEnumerateInstanceLayerProperties) X(vkEnumerateInstanceVersion)

#define VKF_INSTANCE(X) \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceFeatures2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceMemoryProperties2) X(vkGetPhysicalDeviceFormatProperties) \
    X(vkGetPhysicalDeviceImageFormatProperties) X(vkGetPhysicalDeviceSparseImageFormatProperties) X(vkEnumerateDeviceExtensionProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr)

// optional: debug utils; the surface functions exist only with VK_KHR_surface (not in a headless instance)
#define VKF_INSTANCE_OPT(X) X(vkCreateDebugUtilsMessengerEXT) X(vkDestroyDebugUtilsMessengerEXT) X(vkSetDebugUtilsObjectNameEXT) \
    X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR) X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) X(vkGetPhysicalDeviceSurfaceFormatsKHR) \
    X(vkGetPhysicalDeviceSurfacePresentModesKHR)

#define VKF_DEVICE(X) \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkDeviceWaitIdle) X(vkQueueWaitIdle) X(vkQueueSubmit2) X(vkQueueBindSparse) \
    X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkWaitSemaphores) X(vkGetSemaphoreCounterValue) X(vkSignalSemaphore) \
    X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences) X(vkGetFenceStatus) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) X(vkUnmapMemory) X(vkFlushMappedMemoryRanges) X(vkInvalidateMappedMemoryRanges) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkGetBufferDeviceAddress) \
    X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) X(vkGetImageSparseMemoryRequirements) X(vkBindImageMemory) X(vkCreateImageView) X(vkDestroyImageView) \
    X(vkCreateSampler) X(vkDestroySampler) \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkResetCommandPool) X(vkAllocateCommandBuffers) X(vkFreeCommandBuffers) \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer) \
    X(vkCmdPipelineBarrier2) X(vkCmdCopyBuffer) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdCopyImage) X(vkCmdBlitImage) \
    X(vkCmdFillBuffer) X(vkCmdUpdateBuffer) X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage) X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdBindPipeline) X(vkCmdSetViewport) X(vkCmdSetScissor) \
    X(vkCmdDraw) X(vkCmdPushConstants) X(vkCmdBindDescriptorSets) \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) X(vkCreateGraphicsPipelines) X(vkCreateComputePipelines) \
    X(vkDestroyPipeline) X(vkCreatePipelineCache) X(vkDestroyPipelineCache) X(vkGetPipelineCacheData) \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) \
    X(vkUpdateDescriptorSets)     X(vkCmdBindVertexBuffers2) X(vkCmdBindIndexBuffer) X(vkCmdDrawIndexed) X(vkCmdDrawIndirect) X(vkCmdDrawIndexedIndirect) X(vkCmdDrawIndirectCount)     X(vkCmdDrawIndexedIndirectCount) X(vkCmdDispatch) X(vkCmdDispatchIndirect) X(vkCmdClearAttachments) X(vkCmdResolveImage)     X(vkCmdBeginQuery) X(vkCmdEndQuery) X(vkCreateQueryPool) X(vkDestroyQueryPool) X(vkGetQueryPoolResults) X(vkResetQueryPool) X(vkCmdResetQueryPool)     X(vkCreateBufferView) X(vkDestroyBufferView) X(vkResetDescriptorPool) X(vkCmdSetViewportWithCount) X(vkCmdSetScissorWithCount) X(vkCmdSetLineWidth)     X(vkCmdSetDepthBias) X(vkCmdSetBlendConstants) X(vkCmdSetStencilCompareMask) X(vkCmdSetStencilWriteMask) X(vkCmdSetStencilReference)     X(vkCmdSetCullMode) X(vkCmdSetFrontFace) X(vkCmdSetPrimitiveTopology) X(vkCmdSetDepthTestEnable) X(vkCmdSetDepthWriteEnable) X(vkCmdSetDepthCompareOp)     X(vkCmdSetDepthBoundsTestEnable) X(vkCmdSetStencilTestEnable) X(vkCmdSetStencilOp) X(vkCmdSetRasterizerDiscardEnable) X(vkCmdSetDepthBiasEnable)     X(vkCmdSetPrimitiveRestartEnable) X(vkCmdWriteTimestamp2)

// with VK_KHR_swapchain only (not on a headless device); the replay's extension entry points (NULL when the device lacks the extension)
#define VKF_DEVICE_OPT(X) X(vkCmdDrawMeshTasksEXT) X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR) X(vkQueuePresentKHR)     X(vkCmdSetVertexInputEXT) X(vkCmdSetColorBlendEnableEXT) X(vkCmdSetColorBlendEquationEXT) X(vkCmdSetColorWriteMaskEXT) X(vkCmdSetLogicOpEnableEXT)     X(vkCmdSetLogicOpEXT) X(vkCmdSetAlphaToCoverageEnableEXT) X(vkCmdSetPolygonModeEXT) X(vkCmdSetDepthClampEnableEXT) X(vkCmdSetPatchControlPointsEXT)     X(vkCmdSetCheckpointNV) X(vkGetQueueCheckpointData2NV) X(vkGetDeviceFaultInfoEXT)

#define VKF_DECL(n) extern PFN_##n n;
VKF_GLOBAL(VKF_DECL)
VKF_INSTANCE(VKF_DECL)
VKF_INSTANCE_OPT(VKF_DECL)
VKF_DEVICE(VKF_DECL)
VKF_DEVICE_OPT(VKF_DECL)
extern PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
#undef VKF_DECL

int vkf_load_global(void);                 // loads vulkan-1.dll; 0 on success, -1 when there is no Vulkan loader
int vkf_load_instance(VkInstance inst);    // the instance-level entry points (0, or -1 when a required one is missing)
int vkf_load_device(VkDevice dev);         // the device-level entry points
