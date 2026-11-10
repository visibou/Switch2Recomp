// Run-time loading of the Vulkan entry points (vk_fn.h). Written from the Vulkan specification's "Command Function Pointers" chapter.
#include "vk_fn.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <stdio.h>

#define VKF_DEF(n) PFN_##n n;
VKF_GLOBAL(VKF_DEF)
VKF_INSTANCE(VKF_DEF)
VKF_INSTANCE_OPT(VKF_DEF)
VKF_DEVICE(VKF_DEF)
VKF_DEVICE_OPT(VKF_DEF)
PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
#undef VKF_DEF

static HMODULE g_lib;

int vkf_load_global(void) {
    if (vkGetInstanceProcAddr) return 0;
    if (!g_lib) g_lib = LoadLibraryW(L"vulkan-1.dll");
    if (!g_lib) return -1;
    vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)(void (*)(void))GetProcAddress(g_lib, "vkGetInstanceProcAddr");
    if (!vkGetInstanceProcAddr) return -1;
#define L(n) n = (PFN_##n)vkGetInstanceProcAddr(NULL, #n);
    VKF_GLOBAL(L)
#undef L
    return vkCreateInstance && vkEnumerateInstanceExtensionProperties ? 0 : -1;
}

int vkf_load_instance(VkInstance inst) {
    int missing = 0;
#define L(n) if (!(n = (PFN_##n)vkGetInstanceProcAddr(inst, #n))) { fprintf(stderr, "vulkan: no %s\n", #n); missing++; }
    VKF_INSTANCE(L)
#undef L
#define O(n) n = (PFN_##n)vkGetInstanceProcAddr(inst, #n);
    VKF_INSTANCE_OPT(O)
#undef O
    return missing ? -1 : 0;
}

int vkf_load_device(VkDevice dev) {
    int missing = 0;
#define L(n) if (!(n = (PFN_##n)vkGetDeviceProcAddr(dev, #n))) { fprintf(stderr, "vulkan: no %s\n", #n); missing++; }
    VKF_DEVICE(L)
#undef L
#define O(n) n = (PFN_##n)vkGetDeviceProcAddr(dev, #n);
    VKF_DEVICE_OPT(O)
#undef O
    return missing ? -1 : 0;
}
