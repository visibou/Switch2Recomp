// Vulkan instance and device bring-up, device memory, one-shot command buffers, deferred destruction (docs/m5-vulkan.md §2).
// Written from the Vulkan 1.3 specification. The list of optional features follows what the NVN replay needs; Ryujinx's VulkanInitialization.cs (MIT,
// git.ryujinx.app) was read for which of them a Switch GPU's SPIR-V commonly uses.
#include "vk_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

vk_device g_vk;
static int g_vk_fault_bin;   // the device keeps VK_EXT_device_fault's vendor binary (Aftermath runs only)

const char *vk_vendor_name(uint32_t v) {
    switch (v) { case 0x10de: return "NVIDIA"; case 0x1002: return "AMD"; case 0x8086: return "Intel"; case 0x13b5: return "Arm"; case 0x5143: return "Qualcomm"; case 0x106b: return "Apple"; }
    return "other";
}

static VKAPI_ATTR VkBool32 VKAPI_CALL on_debug(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT type,
                                               const VkDebugUtilsMessengerCallbackDataEXT *d, void *user) {
    (void)type; (void)user;
    if (sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) { uint32_t n = atomic_fetch_add(&g_vk.val_errors, 1); if (n < 200) VKRED("validation: %s", d->pMessage); }
    else if (sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) { uint32_t n = atomic_fetch_add(&g_vk.val_warnings, 1); if (n < 50) VKLOG("validation warning: %s", d->pMessage); }
    return VK_FALSE;
}

void vk_name(uint64_t handle, VkObjectType type, const char *name) {
    if (!g_vk.validation || !vkSetDebugUtilsObjectNameEXT || !handle) return;
    VkDebugUtilsObjectNameInfoEXT n = {VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT, NULL, type, handle, name};
    vkSetDebugUtilsObjectNameEXT(g_vk.dev, &n);
}

static int no_ext_match(const char *list, const char *name) {   // NXRT_VK_NO_EXT: comma-separated parts of extension names
    char buf[512]; snprintf(buf, sizeof buf, "%s", list);
    for (char *t = strtok(buf, ","); t; t = strtok(NULL, ",")) if (*t && strstr(name, t)) return 1;
    return 0;
}
static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1; return 0; }

// ---------- Optional device extensions with a feature structure: queried and enabled together (every feature the device has, minus the costly robustness ones) ----------
typedef struct { const char *ext, *alt; VkStructureType st; size_t size; int *flag; } ext_feat;
typedef struct { VkStructureType sType; void *pNext; VkBool32 b[64]; } feat_any;   // every feature structure is sType, pNext, then VkBool32s

static int count_true(const feat_any *f, size_t size) { int n = 0; size_t k = (size - offsetof(feat_any, b)) / sizeof(VkBool32); for (size_t i = 0; i < k; i++) n += f->b[i] != 0; return n; }

// NXRT_VK_MEMREPORT=1: what the driver itself says it holds in device memory, by object type (VK_EXT_device_memory_report), in the log every 5 s
typedef struct { uint64_t id, size; uint32_t type, heap; } mr_ent;
enum { MR_N = 1 << 20 };
static mr_ent *g_mr; static SRWLOCK g_mr_lk; static uint64_t g_mr_by[64], g_mr_all;
static void VKAPI_PTR mr_cb(const VkDeviceMemoryReportCallbackDataEXT *d, void *u) {
    (void)u; if (!g_mr) return;
    AcquireSRWLockExclusive(&g_mr_lk);
    uint32_t h = (uint32_t)((d->memoryObjectId * 0x9E3779B97F4A7C15ull) >> 44);
    if (d->type == VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATE_EXT || d->type == VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_IMPORT_EXT) {
        for (uint32_t k = 0; k < MR_N; k++, h = (h + 1) & (MR_N - 1)) if (!g_mr[h].id || g_mr[h].id == ~0ull) { g_mr[h] = (mr_ent){d->memoryObjectId, d->size, (uint32_t)d->objectType & 63, d->heapIndex}; if (d->heapIndex == 0) { g_mr_by[(uint32_t)d->objectType & 63] += d->size; g_mr_all += d->size; } break; }
    } else if (d->type == VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_FREE_EXT || d->type == VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_UNIMPORT_EXT) {
        for (uint32_t k = 0; k < MR_N && g_mr[h].id; k++, h = (h + 1) & (MR_N - 1)) if (g_mr[h].id == d->memoryObjectId) { if (g_mr[h].heap == 0) { g_mr_by[g_mr[h].type] -= g_mr[h].size; g_mr_all -= g_mr[h].size; } g_mr[h].id = ~0ull; break; }
    }
    ReleaseSRWLockExclusive(&g_mr_lk);
}
void vk_memreport_log(void) {
    if (!g_mr) return;
    char b[500]; int o = 0; AcquireSRWLockShared(&g_mr_lk);
    for (int i = 0; i < 64; i++) if (g_mr_by[i] >> 20) o += snprintf(b + o, sizeof b - o, " type %d: %llu MB;", i, (unsigned long long)(g_mr_by[i] >> 20));
    uint64_t all = g_mr_all; ReleaseSRWLockShared(&g_mr_lk);
    VKLOG("the driver holds %llu MB in heap 0:%s", (unsigned long long)(all >> 20), b);
}
int vk_dev_open(const vk_dev_opts *o, const char *const *inst_exts, uint32_t n_inst_exts) {
    if (g_vk.up) return 0;
    InitializeSRWLock(&g_vk.qlock);
    vk_am_init();   // NXRT_VK_AFTERMATH=1: before the instance and the device, as the SDK asks
    if (vkf_load_global()) { VKRED("no Vulkan loader (vulkan-1.dll): the Vulkan renderer cannot start"); return -1; }
    uint32_t iv = VK_API_VERSION_1_0; if (vkEnumerateInstanceVersion) vkEnumerateInstanceVersion(&iv);
    if (iv < VK_API_VERSION_1_3) { VKRED("the Vulkan loader is version %u.%u, 1.3 is needed", VK_API_VERSION_MAJOR(iv), VK_API_VERSION_MINOR(iv)); return -1; }

    // Instance: the window system's extensions, debug utils, surface maintenance; validation when asked and installed
    uint32_t nie = 0; vkEnumerateInstanceExtensionProperties(NULL, &nie, NULL);
    VkExtensionProperties *ie = calloc(nie + 1, sizeof *ie); vkEnumerateInstanceExtensionProperties(NULL, &nie, ie);
    const char *exts[32]; uint32_t ne = 0;
    for (uint32_t i = 0; i < n_inst_exts && ne < 24; i++) exts[ne++] = inst_exts[i];
    if (has_ext(ie, nie, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) { exts[ne++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME; g_vk.has.debug_utils = 1; }
    if (!o->headless && has_ext(ie, nie, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME) && has_ext(ie, nie, VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME)) {
        exts[ne++] = VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME; exts[ne++] = VK_EXT_SURFACE_MAINTENANCE_1_EXTENSION_NAME; g_vk.has.surface_maint1 = 1; }
    free(ie);
    const char *layers[1]; uint32_t nl = 0;
    if (o->validation) {
        uint32_t n = 0; vkEnumerateInstanceLayerProperties(&n, NULL);
        VkLayerProperties *lp = calloc(n + 1, sizeof *lp); vkEnumerateInstanceLayerProperties(&n, lp);
        for (uint32_t i = 0; i < n; i++) if (!strcmp(lp[i].layerName, "VK_LAYER_KHRONOS_validation")) { layers[nl++] = "VK_LAYER_KHRONOS_validation"; g_vk.validation = 1; }
        free(lp);
        if (!g_vk.validation) VKLOG("validation asked for, but VK_LAYER_KHRONOS_validation is not installed");
    }
    { void vkc_rdoc_load(void); vkc_rdoc_load(); }
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "pokopia-native", 1, "nxrt", 1, VK_API_VERSION_1_3};
    VkDebugUtilsMessengerCreateInfoEXT dci = {VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT, NULL, 0,
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT, on_debug, NULL};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, g_vk.validation && g_vk.has.debug_utils ? &dci : NULL, 0, &app, nl, layers, ne, exts};
    VkResult r = vkCreateInstance(&ici, NULL, &g_vk.inst);
    if (r != VK_SUCCESS) { VKRED("vkCreateInstance failed (%d)", r); return -1; }
    if (vkf_load_instance(g_vk.inst)) { VKRED("the Vulkan instance lacks core entry points"); return -1; }
    if (g_vk.validation && vkCreateDebugUtilsMessengerEXT) vkCreateDebugUtilsMessengerEXT(g_vk.inst, &dci, NULL, &g_vk.dbg);

    // Physical device: NXRT_VK_DEVICE=<index or part of the name>, else the first discrete GPU with Vulkan 1.3, else the first 1.3 device
    uint32_t npd = 0; vkEnumeratePhysicalDevices(g_vk.inst, &npd, NULL);
    VkPhysicalDevice *pds = calloc(npd + 1, sizeof *pds); vkEnumeratePhysicalDevices(g_vk.inst, &npd, pds);
    int pick = -1, pick_discrete = 0;
    for (uint32_t i = 0; i < npd; i++) {
        VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i], &p);
        VKLOG("GPU %u: %s (%s, Vulkan %u.%u.%u, driver %#x, type %d)", i, p.deviceName, vk_vendor_name(p.vendorID), VK_API_VERSION_MAJOR(p.apiVersion),
              VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion), p.driverVersion, p.deviceType);
        if (p.apiVersion < VK_API_VERSION_1_3) continue;
        if (o->device_hint && *o->device_hint) {
            char *end; long k = strtol(o->device_hint, &end, 10);
            if ((!*end && k == (long)i) || (*end && strstr(p.deviceName, o->device_hint))) { pick = (int)i; pick_discrete = 2; }
            continue;
        }
        int disc = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        if (pick < 0 || (disc && !pick_discrete)) { pick = (int)i; pick_discrete = disc; }
    }
    if (pick < 0) { VKRED("no Vulkan 1.3 GPU%s%s", o->device_hint ? " matching NXRT_VK_DEVICE=" : "", o->device_hint ? o->device_hint : ""); free(pds); return -1; }
    g_vk.pd = pds[pick]; free(pds);
    vkGetPhysicalDeviceProperties(g_vk.pd, &g_vk.props);
    vkGetPhysicalDeviceMemoryProperties(g_vk.pd, &g_vk.memp);
    g_vk.api = g_vk.props.apiVersion;

    // Queue: one family with graphics, compute and transfer (and sparse binding when it has it); presentation is checked when the surface exists
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(g_vk.pd, &nq, NULL);
    VkQueueFamilyProperties *qf = calloc(nq + 1, sizeof *qf); vkGetPhysicalDeviceQueueFamilyProperties(g_vk.pd, &nq, qf);
    int fam = -1;
    for (uint32_t i = 0; i < nq; i++) {
        VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
        if ((qf[i].queueFlags & need) != need) continue;
        if (fam < 0 || ((qf[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) && !(qf[fam].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT))) fam = (int)i;
    }
    if (fam < 0) { VKRED("%s has no graphics + compute queue", g_vk.props.deviceName); free(qf); return -1; }
    g_vk.qfam = (uint32_t)fam;
    int q_sparse = (qf[fam].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0;
    free(qf);

    // Features: core 1.0-1.3, then the optional extensions' structures
    uint32_t nde = 0; vkEnumerateDeviceExtensionProperties(g_vk.pd, NULL, &nde, NULL);
    VkExtensionProperties *de = calloc(nde + 1, sizeof *de); vkEnumerateDeviceExtensionProperties(g_vk.pd, NULL, &nde, de);
    if (!has_ext(de, nde, VK_KHR_SWAPCHAIN_EXTENSION_NAME) && !o->headless) { VKRED("%s has no VK_KHR_swapchain", g_vk.props.deviceName); free(de); return -1; }
    vk_caps *h = &g_vk.has;
    ext_feat ef[] = {
        {VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT, sizeof(VkPhysicalDeviceCustomBorderColorFeaturesEXT), &h->custom_border},
        {VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT, sizeof(VkPhysicalDeviceTransformFeedbackFeaturesEXT), &h->xfb},
        {VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT, sizeof(VkPhysicalDeviceDepthClipControlFeaturesEXT), &h->depth_clip_control},
        {VK_KHR_INDEX_TYPE_UINT8_EXTENSION_NAME, VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES_KHR, sizeof(VkPhysicalDeviceIndexTypeUint8FeaturesKHR), &h->index_u8},
        {VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT, sizeof(VkPhysicalDeviceConditionalRenderingFeaturesEXT), &h->cond_render},
        {VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT, sizeof(VkPhysicalDeviceProvokingVertexFeaturesEXT), &h->provoking_last},
        {VK_EXT_MESH_SHADER_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT, sizeof(VkPhysicalDeviceMeshShaderFeaturesEXT), &h->mesh_shader},
        {VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT, sizeof(VkPhysicalDeviceRobustness2FeaturesEXT), &h->null_descriptor},
        {VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT, sizeof(VkPhysicalDeviceExtendedDynamicState2FeaturesEXT), &h->eds2_logic_op},
        {VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT, sizeof(VkPhysicalDeviceExtendedDynamicState3FeaturesEXT), &h->eds3},
        {VK_KHR_PRESENT_ID_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_FEATURES_KHR, sizeof(VkPhysicalDevicePresentIdFeaturesKHR), &h->present_id},
        {VK_KHR_PRESENT_WAIT_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR, sizeof(VkPhysicalDevicePresentWaitFeaturesKHR), &h->present_wait},
        {VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT, sizeof(VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT), &h->swap_maint1},
        {VK_KHR_LINE_RASTERIZATION_EXTENSION_NAME, VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_KHR, sizeof(VkPhysicalDeviceLineRasterizationFeaturesKHR), &h->line_raster},
        {VK_KHR_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME, VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR, sizeof(VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR), &h->vertex_divisor},
        {VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES_EXT, sizeof(VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT), &h->prim_list_restart},
        {VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT, sizeof(VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT), &h->fs_interlock},
        {VK_EXT_BLEND_OPERATION_ADVANCED_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BLEND_OPERATION_ADVANCED_FEATURES_EXT, sizeof(VkPhysicalDeviceBlendOperationAdvancedFeaturesEXT), &h->blend_adv},
        {VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_FEATURES_EXT, sizeof(VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT), &h->feedback_loop},
        {VK_EXT_COLOR_WRITE_ENABLE_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COLOR_WRITE_ENABLE_FEATURES_EXT, sizeof(VkPhysicalDeviceColorWriteEnableFeaturesEXT), &h->color_write_enable},
        {VK_EXT_DEPTH_CLAMP_ZERO_ONE_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLAMP_ZERO_ONE_FEATURES_EXT, sizeof(VkPhysicalDeviceDepthClampZeroOneFeaturesEXT), &h->depth_clamp_01},
        {VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT, sizeof(VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT), &h->gpl},
        {VK_EXT_4444_FORMATS_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_4444_FORMATS_FEATURES_EXT, sizeof(VkPhysicalDevice4444FormatsFeaturesEXT), &h->fmt_4444},
        {VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT, sizeof(VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT), &h->vertex_input_dynamic},   // the replay (M5-C)
        {VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT, sizeof(VkPhysicalDeviceImageViewMinLodFeaturesEXT), &h->view_min_lod},   // tile levels left out (vk_sparse.inc)
        {VK_EXT_DEVICE_MEMORY_REPORT_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT, sizeof(VkPhysicalDeviceDeviceMemoryReportFeaturesEXT), &h->mem_report},
        {VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR, sizeof(VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR), &h->pipe_exec},   // NXRT_VK_PIPE_STATS only (vk_pipe.c)
        {VK_EXT_DEVICE_FAULT_EXTENSION_NAME, NULL, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT, sizeof(VkPhysicalDeviceFaultFeaturesEXT), &h->device_fault},   // vk_lost's report
    };
    enum { NEF = sizeof ef / sizeof *ef };
    // Extensions without a feature structure
    struct { const char *ext; int *flag; } ex[] = {
        {VK_NV_VIEWPORT_SWIZZLE_EXTENSION_NAME, &h->viewport_swizzle_nv}, {VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, &h->memory_budget},
        {VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME, &h->stencil_export}, {VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME, &h->gs_passthrough_nv},
        {VK_NV_VIEWPORT_ARRAY_2_EXTENSION_NAME, &h->viewport_array2_nv}, {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, &h->push_descriptor},
        {VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME, NULL},   // with the graphics pipeline library only
        {VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME, &h->checkpoints},   // vk_lost's report; NXRT_VK_CHECKPOINTS=0 leaves it off
    };
    const char *ck = getenv("NXRT_VK_CHECKPOINTS");
    feat_any *fs[NEF]; const char *dexts[64]; uint32_t nd = 0;
    if (!o->headless) dexts[nd++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    void *chain = NULL;
    // NXRT_VK_NO_EXT=part,part (test): device extensions whose name contains one of the parts are left out, as on a device without them
    const char *noext = getenv("NXRT_VK_NO_EXT");
    #define NOEXT(n) (noext && *noext && no_ext_match(noext, n))
    for (int i = 0; i < NEF; i++) {
        fs[i] = NULL;
        const char *name = has_ext(de, nde, ef[i].ext) ? ef[i].ext : ef[i].alt && has_ext(de, nde, ef[i].alt) ? ef[i].alt : NULL;
        if (!name) continue;
        if (NOEXT(name)) { VKLOG("left out for a test: %s", name); continue; }
        if (!strcmp(name, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME) && !getenv("NXRT_VK_PIPE_STATS")) continue;
        if (!strcmp(name, VK_EXT_SWAPCHAIN_MAINTENANCE_1_EXTENSION_NAME) && (!g_vk.has.surface_maint1 || o->headless)) continue;
        if ((!strcmp(name, VK_KHR_PRESENT_ID_EXTENSION_NAME) || !strcmp(name, VK_KHR_PRESENT_WAIT_EXTENSION_NAME)) && o->headless) continue;
        fs[i] = calloc(1, sizeof(feat_any)); fs[i]->sType = ef[i].st; fs[i]->pNext = chain; chain = fs[i];
        dexts[nd++] = name;
    }
    f13.pNext = chain;
    vkGetPhysicalDeviceFeatures2(g_vk.pd, &f2);
    for (unsigned i = 0; i < sizeof ex / sizeof *ex; i++) {
        if (!has_ext(de, nde, ex[i].ext)) continue;
        if (!ex[i].flag && !(has_ext(de, nde, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME))) continue;
        if (ex[i].flag == &h->checkpoints && ck && *ck == '0') continue;
        if (NOEXT(ex[i].ext)) { VKLOG("left out for a test: %s", ex[i].ext); continue; }
        dexts[nd++] = ex[i].ext; if (ex[i].flag) *ex[i].flag = 1;
    }
    if (h->checkpoints && ck && !strcmp(ck, "all")) h->checkpoints = 2;
    int am_cfg = vk_am_on() && has_ext(de, nde, VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME);
    if (am_cfg) dexts[nd++] = VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME;
    else if (vk_am_on()) VKRED("Aftermath: no VK_NV_device_diagnostics_config on this device: dumps without shader debug info or shader error reporting");
    free(de);
    // Required: what the replay cannot do without (docs/m5-vulkan.md §2)
    const char *miss = NULL;
    if (!f13.dynamicRendering) miss = "dynamicRendering"; else if (!f13.synchronization2) miss = "synchronization2"; else if (!f12.timelineSemaphore) miss = "timelineSemaphore";
    else if (!f12.descriptorIndexing || !f12.runtimeDescriptorArray || !f12.descriptorBindingPartiallyBound || !f12.shaderSampledImageArrayNonUniformIndexing) miss = "descriptor indexing";
    else if (!f2.features.textureCompressionBC) miss = "textureCompressionBC";
    else if (!f13.maintenance4) miss = "maintenance4";
    if (miss) { VKRED("%s lacks %s, which the renderer needs", g_vk.props.deviceName, miss); return -1; }
    // Enabled: every supported feature except robustness that costs performance (robustBufferAccess stays on NVIDIA, as Ryujinx does) and capture/replay addresses
    VkPhysicalDeviceFeatures en = f2.features;
    en.robustBufferAccess = g_vk.props.vendorID == 0x10de && f2.features.robustBufferAccess;
    en.inheritedQueries = 0;
    // NXRT_VK_ROBUST (test, the GPU freeze): 1 = robust image access too (image loads and stores outside the image are dropped), 2 = robustness2 for images and buffers as well
    const char *rob_e = getenv("NXRT_VK_ROBUST"); int rob = rob_e ? atoi(rob_e) : 0;
    f13.robustImageAccess = rob >= 1 && f13.robustImageAccess; f12.bufferDeviceAddressCaptureReplay = 0; f12.bufferDeviceAddressMultiDevice = 0; f11.protectedMemory = 0;
    if (rob >= 1 && f2.features.robustBufferAccess) en.robustBufferAccess = VK_TRUE;
    // NXRT_VK_MIN_FEATURES=1 (test, the GPU freeze): of the Vulkan 1.1-1.3 features, only those the renderer and the translated shaders use, as Ryujinx asks,
    // instead of everything the driver offers (some change how it compiles shaders). Off: buffer device addresses (2 shaders of 11,561 use them)
    if (getenv("NXRT_VK_MIN_FEATURES") && *getenv("NXRT_VK_MIN_FEATURES") == '1') {
        VkPhysicalDeviceVulkan11Features k11 = {f11.sType, f11.pNext}; VkPhysicalDeviceVulkan12Features k12 = {f12.sType, f12.pNext}; VkPhysicalDeviceVulkan13Features k13 = {f13.sType, f13.pNext};
        k11.shaderDrawParameters = f11.shaderDrawParameters;
        k12.timelineSemaphore = f12.timelineSemaphore; k12.descriptorIndexing = f12.descriptorIndexing; k12.runtimeDescriptorArray = f12.runtimeDescriptorArray;
        k12.descriptorBindingPartiallyBound = f12.descriptorBindingPartiallyBound; k12.shaderSampledImageArrayNonUniformIndexing = f12.shaderSampledImageArrayNonUniformIndexing;
        k12.samplerMirrorClampToEdge = f12.samplerMirrorClampToEdge; k12.samplerFilterMinmax = f12.samplerFilterMinmax; k12.drawIndirectCount = f12.drawIndirectCount;
        k12.hostQueryReset = f12.hostQueryReset; k12.shaderOutputViewportIndex = f12.shaderOutputViewportIndex; k12.shaderOutputLayer = f12.shaderOutputLayer;
        k12.separateDepthStencilLayouts = f12.separateDepthStencilLayouts; k12.uniformBufferStandardLayout = f12.uniformBufferStandardLayout; k12.scalarBlockLayout = f12.scalarBlockLayout;
        k13.dynamicRendering = f13.dynamicRendering; k13.synchronization2 = f13.synchronization2; k13.maintenance4 = f13.maintenance4;
        f11 = k11; f12 = k12; f13 = k13;
        VKLOG("Vulkan 1.1-1.3 features: only those the renderer uses (NXRT_VK_MIN_FEATURES=1)");
    }
    if (rob) VKLOG("robustness for the freeze test: robust image access %d, robustness2 %s", f13.robustImageAccess, rob >= 2 ? "asked" : "off");
    for (int i = 0; i < NEF; i++) {
        if (!fs[i]) continue;
        if (ef[i].st == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT) {   // task and mesh shaders alone: the others need features this device set does not ask for
            VkPhysicalDeviceMeshShaderFeaturesEXT *ms = (void *)fs[i]; ms->multiviewMeshShader = 0; ms->primitiveFragmentShadingRateMeshShader = 0; ms->meshShaderQueries = 0; h->mesh_shader = ms->meshShader && ms->taskShader; continue; }
        if (ef[i].st == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT) {
            VkPhysicalDeviceRobustness2FeaturesEXT *r2 = (void *)fs[i]; if (rob < 2) { r2->robustBufferAccess2 = 0; r2->robustImageAccess2 = 0; } h->null_descriptor = r2->nullDescriptor != 0; continue; }
        if (ef[i].st == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT) {
            VkPhysicalDeviceCustomBorderColorFeaturesEXT *c = (void *)fs[i]; h->custom_border = c->customBorderColors != 0; h->custom_border_noformat = c->customBorderColorWithoutFormat != 0; continue; }
        if (ef[i].st == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT) {
            VkPhysicalDeviceTransformFeedbackFeaturesEXT *x = (void *)fs[i]; h->xfb = x->transformFeedback != 0; h->xfb_streams = x->geometryStreams != 0; continue; }
        if (ef[i].st == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT) {
            VkPhysicalDeviceExtendedDynamicState2FeaturesEXT *x = (void *)fs[i]; h->eds2_logic_op = x->extendedDynamicState2LogicOp != 0; h->eds2_patch_points = x->extendedDynamicState2PatchControlPoints != 0; continue; }
        if (ef[i].st == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT) {
            VkPhysicalDeviceProvokingVertexFeaturesEXT *x = (void *)fs[i]; x->transformFeedbackPreservesProvokingVertex = x->transformFeedbackPreservesProvokingVertex && h->xfb; h->provoking_last = x->provokingVertexLast != 0; continue; }
        if (ef[i].st == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT) {
            VkPhysicalDeviceFaultFeaturesEXT *x = (void *)fs[i]; if (!vk_am_on()) x->deviceFaultVendorBinary = 0;   // with Aftermath: the driver's own dump, which its decoder reads
            g_vk_fault_bin = x->deviceFaultVendorBinary != 0; h->device_fault = x->deviceFault != 0; continue; }
        *ef[i].flag = count_true(fs[i], ef[i].size) > 0;
    }
    const char *sp = getenv("NXRT_VK_SPARSE");   // test switch (#12): 0 backs every memory pool fully at once instead of sparse granules on first use
    h->sparse_buffer = q_sparse && en.sparseBinding && en.sparseResidencyBuffer && !(sp && *sp == '0');
    h->sparse_aliased = h->sparse_buffer && en.sparseResidencyAliased;
    h->sparse_image = q_sparse && en.sparseBinding && en.sparseResidencyImage2D && !(sp && *sp == '0');
    h->bc = en.textureCompressionBC; h->astc_ldr = en.textureCompressionASTC_LDR;
    h->mirror_clamp_edge = f12.samplerMirrorClampToEdge; h->minmax_filter = f12.samplerFilterMinmax;
    g_vk.f10 = en; g_vk.f11 = f11; g_vk.f12 = f12; g_vk.f13 = f13;
    g_vk.f11.pNext = g_vk.f12.pNext = g_vk.f13.pNext = NULL;
    VkPhysicalDeviceFeatures2 enf = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11, en};

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, g_vk.qfam, 1, &prio};
    VkDeviceCreateInfo dci2 = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enf, 0, 1, &qci, 0, NULL, nd, dexts, NULL};
    VkDeviceDeviceMemoryReportCreateInfoEXT mri = {VK_STRUCTURE_TYPE_DEVICE_DEVICE_MEMORY_REPORT_CREATE_INFO_EXT, &enf, 0, mr_cb, NULL};
    if (getenv("NXRT_VK_MEMREPORT")) { if (g_vk.has.mem_report) { InitializeSRWLock(&g_mr_lk); g_mr = calloc(MR_N, sizeof *g_mr); dci2.pNext = &mri; } else VKLOG("no VK_EXT_device_memory_report here"); }
    if (am_cfg) dci2.pNext = vk_am_device_chain((void *)dci2.pNext);
    r = vkCreateDevice(g_vk.pd, &dci2, NULL, &g_vk.dev);
    for (int i = 0; i < NEF; i++) free(fs[i]);
    if (r != VK_SUCCESS) { VKRED("vkCreateDevice failed (%d) on %s", r, g_vk.props.deviceName); return -1; }
    if (vkf_load_device(g_vk.dev)) { VKRED("the Vulkan device lacks core entry points"); return -1; }
    { void vkd_install(void); vkd_install(); }   // NXRT_VK_DEFER=1: the replay's commands recorded on a thread of their own (vk_defer.c)
    vkGetDeviceQueue(g_vk.dev, g_vk.qfam, 0, &g_vk.q);
    VkSemaphoreTypeCreateInfo tci = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, NULL, VK_SEMAPHORE_TYPE_TIMELINE, 0};
    VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &tci, 0};
    if (vkCreateSemaphore(g_vk.dev, &sci, NULL, &g_vk.tl) != VK_SUCCESS) { VKRED("cannot create the GPU timeline semaphore"); return -1; }
    vk_name((uint64_t)g_vk.tl, VK_OBJECT_TYPE_SEMAPHORE, "gpu timeline");

    VkPhysicalDeviceProperties *p = &g_vk.props;
    VKLOG("device: %s, %s driver %u.%u (raw %#x), Vulkan %u.%u.%u, queue family %u%s, %u device extensions enabled, validation %s",
          p->deviceName, vk_vendor_name(p->vendorID), p->vendorID == 0x10de ? (p->driverVersion >> 22) & 0x3ff : VK_API_VERSION_MAJOR(p->driverVersion),
          p->vendorID == 0x10de ? (p->driverVersion >> 14) & 0xff : VK_API_VERSION_MINOR(p->driverVersion), p->driverVersion,
          VK_API_VERSION_MAJOR(p->apiVersion), VK_API_VERSION_MINOR(p->apiVersion), VK_API_VERSION_PATCH(p->apiVersion), g_vk.qfam,
          q_sparse ? " (with sparse binding)" : "", nd, g_vk.validation ? "on" : "off");
    VKLOG("features: geometry %d tessellation %d multiViewport %d dualSrcBlend %d logicOp %d depthClamp %d samplerAnisotropy %d BC %d ASTC %d sparse buffers %d "
          "custom border %d/%d transform feedback %d depth clip control %d uint8 index %d conditional rendering %d provoking last %d null descriptor %d "
          "eds2 logic op %d eds3 %d NV viewport swizzle %d memory budget %d present wait %d line raster %d vertex divisor %d list restart %d stencil export %d "
          "fs interlock %d GPL %d mirror clamp to edge %d min/max filter %d bufferDeviceAddress %d shaderInt64 %d shaderFloat16 %d",
          en.geometryShader, en.tessellationShader, en.multiViewport, en.dualSrcBlend, en.logicOp, en.depthClamp, en.samplerAnisotropy, h->bc, h->astc_ldr, h->sparse_buffer,
          h->custom_border, h->custom_border_noformat, h->xfb, h->depth_clip_control, h->index_u8, h->cond_render, h->provoking_last, h->null_descriptor,
          h->eds2_logic_op, h->eds3, h->viewport_swizzle_nv, h->memory_budget, h->present_wait, h->line_raster, h->vertex_divisor, h->prim_list_restart, h->stencil_export,
          h->fs_interlock, h->gpl, h->mirror_clamp_edge, h->minmax_filter, f12.bufferDeviceAddress, en.shaderInt64, f12.shaderFloat16);
    VKLOG("if the GPU stops responding: checkpoints %s, the driver's fault report %s", h->checkpoints == 2 ? "before every dispatch and draw" : h->checkpoints ? "before every dispatch" : "off",
          h->device_fault ? "on" : "not available");
    for (uint32_t i = 0; i < g_vk.memp.memoryHeapCount; i++)

        VKLOG("memory heap %u: %.0f MB%s", i, g_vk.memp.memoryHeaps[i].size / 1048576.0, g_vk.memp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? " device local" : "");
    for (uint32_t t = 0; t < g_vk.memp.memoryTypeCount; t++) VKLOG("memory type %u: heap %u flags %#x", t, g_vk.memp.memoryTypes[t].heapIndex, g_vk.memp.memoryTypes[t].propertyFlags);
    // What the translated corpus declares (docs/m5-record-shaders.md section 7, track A's 3,967 modules): DrawParameters, GroupNonUniformBallot / Vote / Shuffle,
    // StorageImageRead/WriteWithoutFormat, ImageGatherExtended, Geometry, texel buffers (core), descriptor indexing (required above)
    VkPhysicalDeviceSubgroupProperties sg = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 pp2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sg};
    vkGetPhysicalDeviceProperties2(g_vk.pd, &pp2);
    h->subgroup_stages = sg.supportedStages; h->subgroup_ops = sg.supportedOperations; h->subgroup_size = sg.subgroupSize;
    VkSubgroupFeatureFlags ops = VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
    VkShaderStageFlags st = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    h->corpus_ok = g_vk.f11.shaderDrawParameters && en.shaderStorageImageReadWithoutFormat && en.shaderStorageImageWriteWithoutFormat && en.shaderImageGatherExtended &&
                   en.geometryShader && (sg.supportedOperations & ops) == ops && (sg.supportedStages & st) == st;
    VKLOG("shader corpus needs: draw parameters %d, formatless storage read/write %d/%d, gather extended %d, geometry %d, subgroup ops %#x in stages %#x (size %u): %s",
          g_vk.f11.shaderDrawParameters, en.shaderStorageImageReadWithoutFormat, en.shaderStorageImageWriteWithoutFormat, en.shaderImageGatherExtended, en.geometryShader,
          sg.supportedOperations, sg.supportedStages, sg.subgroupSize, h->corpus_ok ? "all present" : "[RED] something is missing");
    g_vk.up = 1;
    return 0;
}

void vk_dev_close(void) {
    if (!g_vk.dev) return;
    AcquireSRWLockExclusive(&g_vk.qlock); vkDeviceWaitIdle(g_vk.dev); ReleaseSRWLockExclusive(&g_vk.qlock);
    vk_retire_poll();
    void vk_alloc_shutdown(void); vk_alloc_shutdown();
    if (g_vk.tl) vkDestroySemaphore(g_vk.dev, g_vk.tl, NULL);
    vkDestroyDevice(g_vk.dev, NULL);
    if (g_vk.dbg && vkDestroyDebugUtilsMessengerEXT) vkDestroyDebugUtilsMessengerEXT(g_vk.inst, g_vk.dbg, NULL);
    vkDestroyInstance(g_vk.inst, NULL);
    memset(&g_vk, 0, sizeof g_vk);
}

int vk_mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (int pass = 0; pass < 2; pass++)
        for (uint32_t i = 0; i < g_vk.memp.memoryTypeCount; i++) {
            VkMemoryPropertyFlags f = g_vk.memp.memoryTypes[i].propertyFlags;
            if (!(bits & (1u << i)) || (f & want) != want) continue;
            if (!pass && (f & avoid)) continue;
            return (int)i;
        }
    return -1;
}

int vk_budget(uint64_t *budget, uint64_t *usage) {
    *budget = *usage = 0;
    if (!g_vk.has.memory_budget) return 0;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT b = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 m = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &b};
    vkGetPhysicalDeviceMemoryProperties2(g_vk.pd, &m);
    for (uint32_t i = 0; i < m.memoryProperties.memoryHeapCount; i++)
        if (m.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) { *budget += b.heapBudget[i]; *usage += b.heapUsage[i]; }
    return 1;
}

// ---------- Device memory: per memory type, 64 MB blocks with a first-fit free list; requests of 32 MB and more are dedicated ----------
enum { BLOCK = 64u << 20, DEDICATED_MIN = 8u << 20, MAXBLK = 1024, MAXFREE = 512 };   // 8 MB: what an evicted texture frees must go back to the device, not stay in a block of its memory type
typedef struct { VkDeviceSize off, size; } span_t;
typedef struct { VkDeviceMemory mem; uint32_t type; void *map; span_t fr[MAXFREE]; int nfr; VkDeviceSize used; } block_t;
static block_t *g_blk[MAXBLK]; static int g_nblk;
static SRWLOCK g_alk = SRWLOCK_INIT;
static _Atomic uint64_t g_dedicated_bytes;
// What is in device memory by our own count. The heap usage the driver reports rises with what we place in system memory too (5.4 GB on a 3.96 GB heap, with 2.9 GB of ours device-local), so by that number the device was full for good and textures that fit went to system memory
static _Atomic uint64_t g_dev_used;
uint64_t vk_device_used(void) { return atomic_load(&g_dev_used); }

static int block_take(block_t *b, VkDeviceSize size, VkDeviceSize align, VkDeviceSize *off) {
    for (int i = 0; i < b->nfr; i++) {
        VkDeviceSize a = (b->fr[i].off + align - 1) / align * align, end = b->fr[i].off + b->fr[i].size;
        if (a + size > end) continue;
        span_t lead = {b->fr[i].off, a - b->fr[i].off}, tail = {a + size, end - (a + size)};
        if (lead.size && tail.size) {
            if (b->nfr >= MAXFREE) continue;
            memmove(&b->fr[i + 2], &b->fr[i + 1], (size_t)(b->nfr - i - 1) * sizeof(span_t)); b->fr[i] = lead; b->fr[i + 1] = tail; b->nfr++;
        } else if (lead.size) b->fr[i] = lead;
        else if (tail.size) b->fr[i] = tail;
        else { memmove(&b->fr[i], &b->fr[i + 1], (size_t)(b->nfr - i - 1) * sizeof(span_t)); b->nfr--; }
        *off = a; b->used += size;
        return 1;
    }
    return 0;
}
static void block_give(block_t *b, VkDeviceSize off, VkDeviceSize size) {
    int i = 0; while (i < b->nfr && b->fr[i].off < off) i++;
    int left = i > 0 && b->fr[i - 1].off + b->fr[i - 1].size == off, right = i < b->nfr && off + size == b->fr[i].off;
    if (left && right) { b->fr[i - 1].size += size + b->fr[i].size; memmove(&b->fr[i], &b->fr[i + 1], (size_t)(b->nfr - i - 1) * sizeof(span_t)); b->nfr--; }
    else if (left) b->fr[i - 1].size += size;
    else if (right) { b->fr[i].off = off; b->fr[i].size += size; }
    else if (b->nfr < MAXFREE) { memmove(&b->fr[i + 1], &b->fr[i], (size_t)(b->nfr - i) * sizeof(span_t)); b->fr[i] = (span_t){off, size}; b->nfr++; }
    b->used -= size;   // a free list that is full leaks the span until the block is freed: logged by vk_alloc_stats' caller when it matters
}

static int alloc_in(const VkMemoryRequirements *r, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid, vk_alloc *out);
_Atomic uint64_t g_alloc_ns, g_alloc_n;
static _Thread_local int g_no_new_block;
enum { DEDICATED_MIN_ = 8u << 20 };
// Device memory that does not fit goes to system memory, which the GPU reads over PCIe: slower, but this GPU has 4 GB and the game's data is more. Over the budget Windows would page
// device memory out, far worse, so the budget decides, not the failure alone. NXRT_VK_DEVICE_MB overrides the limit
_Thread_local int vk_rt_alloc;
static _Atomic uint64_t g_sys_used; uint64_t vk_sys_used(void) { return atomic_load(&g_sys_used); }
int vk_alloc_mem(const VkMemoryRequirements *r, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid, vk_alloc *out) {
    atomic_fetch_add(&g_alloc_n, 1);
    if (!(want & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) return alloc_in(r, want, avoid, out);
    static uint64_t limit; if (!limit) { const char *e = getenv("NXRT_VK_DEVICE_MB"); limit = e ? strtoull(e, NULL, 0) << 20 : ~0ull; }
    uint64_t budget = 0, usage = 0; int full = 0;
    if (vk_budget(&budget, &usage)) { uint64_t top = budget / 10 * 9; if (top > limit) top = limit; { static int own = -1; if (own < 0) { const char *e2 = getenv("NXRT_VK_OWN_COUNT"); own = !e2 || *e2 != '0'; } if (own) usage = atomic_load(&g_dev_used); } full = usage + r->size > top; }
    // A render target in system memory costs every frame, a texture only its reads: at 3072x3072 a depth pass took 3.9 ms where the same pass in device memory took 0.1. NXRT_VK_RT_LOCAL=0: as any image (a test)
    { static int rl = -1; if (rl < 0) { const char *e = getenv("NXRT_VK_RT_LOCAL"); rl = !e || *e != '0'; } if (rl && vk_rt_alloc) full = 0; }
    g_no_new_block = full;   // full: only what the blocks already hold (freed textures leave room there; blocks are not given back)
    int got = (!full || r->size < DEDICATED_MIN_) && !alloc_in(r, want, avoid, out);
    g_no_new_block = 0;
    if (got) return 0;
    static _Atomic int said; if (!atomic_exchange(&said, 1)) VKLOG("device memory full (usage %llu MB of a budget of %llu MB): from here on what does not fit goes to system memory", (unsigned long long)(usage >> 20), (unsigned long long)(budget >> 20));
    int sys = 0; for (uint32_t i = 0; i < g_vk.memp.memoryTypeCount; i++) if ((r->memoryTypeBits >> i & 1) && !(g_vk.memp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) sys = 1;
    static _Atomic uint64_t nosys, sysb;
    if (sys && !alloc_in(r, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, out)) { out->sys = 1; atomic_fetch_add(&g_sys_used, out->size); uint64_t t = atomic_fetch_add(&sysb, r->size) + r->size; static _Atomic int k; if (atomic_fetch_add(&k, 1) % 512 == 0) VKLOG("system memory taking overflow: %llu MB so far", (unsigned long long)(t >> 20)); return 0; }
    { uint64_t c = atomic_fetch_add(&nosys, 1); if (!(c % 4096)) VKLOG("a %llu byte allocation has no memory type outside device memory (types %#x): over the budget it fails (%llu times)", (unsigned long long)r->size, r->memoryTypeBits, (unsigned long long)c + 1); }
    return -1;
}
static int alloc_in(const VkMemoryRequirements *r, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid, vk_alloc *out) {
    memset(out, 0, sizeof *out); out->block = -1;
    int type = vk_mem_type(r->memoryTypeBits, want, avoid);
    if (type < 0) return -1;
    int host = (g_vk.memp.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    VkDeviceSize align = r->alignment ? r->alignment : 1;
    if (host && g_vk.props.limits.nonCoherentAtomSize > align) align = g_vk.props.limits.nonCoherentAtomSize;
    if (r->size >= DEDICATED_MIN) {
        VkMemoryAllocateFlagsInfo fi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};   // buffers are made with device addresses (validation: memory must say so too)
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, g_vk.f12.bufferDeviceAddress ? &fi : NULL, r->size, (uint32_t)type};
        if (vkAllocateMemory(g_vk.dev, &ai, NULL, &out->mem) != VK_SUCCESS) return -1;
        out->size = r->size;
        if (host && vkMapMemory(g_vk.dev, out->mem, 0, VK_WHOLE_SIZE, 0, &out->map) != VK_SUCCESS) { vkFreeMemory(g_vk.dev, out->mem, NULL); return -1; }
        atomic_fetch_add(&g_dedicated_bytes, r->size);
        if (g_vk.memp.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) { out->dev = 1; atomic_fetch_add(&g_dev_used, r->size); }
        return 0;
    }
    AcquireSRWLockExclusive(&g_alk);
    for (int pass = 0; pass < 2; pass++) {
        for (int i = 0; i < g_nblk; i++) {
            block_t *b = g_blk[i];
            if (!b || b->type != (uint32_t)type) continue;
            VkDeviceSize off;
            if (block_take(b, r->size, align, &off)) {
                out->mem = b->mem; out->off = off; out->size = r->size; out->block = i; out->map = b->map ? (uint8_t *)b->map + off : NULL;
                ReleaseSRWLockExclusive(&g_alk); return 0;
            }
        }
        if (pass || g_nblk >= MAXBLK || (g_no_new_block && (want & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))) break;
        block_t *b = calloc(1, sizeof *b);
        VkMemoryAllocateFlagsInfo fi = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0};
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, g_vk.f12.bufferDeviceAddress ? &fi : NULL, BLOCK, (uint32_t)type};
        if (!b || vkAllocateMemory(g_vk.dev, &ai, NULL, &b->mem) != VK_SUCCESS) { free(b); break; }
        if (host && vkMapMemory(g_vk.dev, b->mem, 0, VK_WHOLE_SIZE, 0, &b->map) != VK_SUCCESS) { vkFreeMemory(g_vk.dev, b->mem, NULL); free(b); break; }
        b->type = (uint32_t)type; b->fr[0] = (span_t){0, BLOCK}; b->nfr = 1;
        if (g_vk.memp.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) atomic_fetch_add(&g_dev_used, BLOCK);
        int slot = -1; for (int i = 0; i < g_nblk; i++) if (!g_blk[i]) { slot = i; break; }
        if (slot < 0) slot = g_nblk++;
        g_blk[slot] = b;
    }
    ReleaseSRWLockExclusive(&g_alk);
    return -1;
}

void vk_free_mem(vk_alloc *a) {
    if (!a->mem) return;
    if (a->sys) atomic_fetch_sub(&g_sys_used, a->size);
    if (a->block < 0) { atomic_fetch_sub(&g_dedicated_bytes, a->size); if (a->dev) atomic_fetch_sub(&g_dev_used, a->size); vkFreeMemory(g_vk.dev, a->mem, NULL); }
    else {
        AcquireSRWLockExclusive(&g_alk);
        block_t *b = g_blk[a->block];
        if (b) { block_give(b, a->off, a->size); if (!b->used && !b->map) { if (g_vk.memp.memoryTypes[b->type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) atomic_fetch_sub(&g_dev_used, BLOCK); vkFreeMemory(g_vk.dev, b->mem, NULL); free(b); g_blk[a->block] = NULL; } }   // an empty block goes back to the device
        ReleaseSRWLockExclusive(&g_alk);
    }
    memset(a, 0, sizeof *a); a->block = -1;
}

void vk_alloc_shutdown(void) {   // the blocks themselves (everything in them is gone by now)
    AcquireSRWLockExclusive(&g_alk);
    for (int i = 0; i < g_nblk; i++) if (g_blk[i]) { vkFreeMemory(g_vk.dev, g_blk[i]->mem, NULL); free(g_blk[i]); g_blk[i] = NULL; }
    g_nblk = 0;
    ReleaseSRWLockExclusive(&g_alk);
}
void vk_alloc_stats(uint64_t *blocks, uint64_t *block_bytes, uint64_t *used, uint64_t *dedicated) {
    AcquireSRWLockShared(&g_alk);
    uint64_t n = 0, u = 0; for (int i = 0; i < g_nblk; i++) if (g_blk[i]) { n++; u += g_blk[i]->used; }
    ReleaseSRWLockShared(&g_alk);
    *blocks = n; *block_bytes = n * BLOCK; *used = u; *dedicated = atomic_load(&g_dedicated_bytes);
}

uint64_t vk_tl_value(void) { uint64_t v = 0; if (g_vk.tl) vkGetSemaphoreCounterValue(g_vk.dev, g_vk.tl, &v); return v; }
int vk_tl_wait(uint64_t value, uint64_t timeout_ns) {
    if (!value) return 1;
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, NULL, 0, 1, &g_vk.tl, &value};
    VkResult r = vkWaitSemaphores(g_vk.dev, &wi, timeout_ns);
    vk_lost("a wait for the GPU", r);
    return r == VK_SUCCESS;
}

// ---------- A lost device (#12, #17): a checkpoint before each dispatch (and draw) names what the GPU had started and finished when it stopped ----------
enum { CK_N = 4096 };
static struct { uint64_t seq; char kind, indirect, key[48]; uint32_t x, y, z; } g_ck[CK_N];   // a draw's key: "pixel (vs vertex)"
static _Atomic uint64_t g_ck_seq;

// The indirect arguments of each checkpointed draw as the GPU read them, copied into host memory after its pass: a draw whose count came out huge keeps the GPU busy for minutes, reported as nothing
static VkBuffer g_cka; static vk_alloc g_cka_mem; static int g_cka_state;
void vk_ck_args(VkCommandBuffer cb, uint64_t seq, VkBuffer b, VkDeviceSize o, uint32_t n) {
    if (!seq || !cb || n > 32) return;
    if (!g_cka_state) { g_cka_state = -1; VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (VkDeviceSize)CK_N * 32, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
        if (vkCreateBuffer(g_vk.dev, &bci, NULL, &g_cka) == VK_SUCCESS) { VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, g_cka, &mr);
            if (!vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, &g_cka_mem) && g_cka_mem.map) { vkBindBufferMemory(g_vk.dev, g_cka, g_cka_mem.mem, g_cka_mem.off); memset(g_cka_mem.map, 0xff, (size_t)CK_N * 32); g_cka_state = 1; } } }
    if (g_cka_state < 0) return;
    VkBufferCopy r = {o, (VkDeviceSize)(seq % CK_N) * 32, n}; vkCmdCopyBuffer(cb, b, g_cka, 1, &r);
}
uint64_t vk_ck_next(void) { return atomic_load(&g_ck_seq) + 1; }
static const uint32_t *ck_args_of(uint64_t s) { return g_cka_state > 0 ? (const uint32_t *)((const uint8_t *)g_cka_mem.map + (s % CK_N) * 32) : NULL; }
uint64_t vk_checkpoint(VkCommandBuffer cb, char kind, const char *key, uint32_t x, uint32_t y, uint32_t z, int indirect) {
    if (!vkCmdSetCheckpointNV || g_vk.has.checkpoints < (kind == 'd' ? 2 : 1)) return 0;
    uint64_t s = atomic_fetch_add(&g_ck_seq, 1) + 1;
    g_ck[s % CK_N].seq = s; g_ck[s % CK_N].kind = kind; g_ck[s % CK_N].indirect = (char)indirect;
    g_ck[s % CK_N].x = x; g_ck[s % CK_N].y = y; g_ck[s % CK_N].z = z;
    snprintf(g_ck[s % CK_N].key, sizeof g_ck[0].key, "%s", key ? key : "?");
    vkCmdSetCheckpointNV(cb, (const void *)(uintptr_t)s);
    return s;
}

static const char *ck_text(uint64_t s, char *out, size_t n) {
    if (!s || g_ck[s % CK_N].seq != s) { snprintf(out, n, "checkpoint #%llu (no longer kept)", (unsigned long long)s); return out; }
    if (g_ck[s % CK_N].kind == 'd') { const uint32_t *a = g_ck[s % CK_N].indirect ? ck_args_of(s) : NULL;
        if (a) snprintf(out, n, "#%llu an indirect draw with shader %s, the GPU's arguments %u %u %u %u %u", (unsigned long long)s, g_ck[s % CK_N].key, a[0], a[1], a[2], a[3], a[4]);
        else snprintf(out, n, "#%llu a draw with shader %s, %u instances", (unsigned long long)s, g_ck[s % CK_N].key, g_ck[s % CK_N].x); }
    else snprintf(out, n, "#%llu a dispatch of shader %s, %ux%ux%u groups%s", (unsigned long long)s, g_ck[s % CK_N].key, g_ck[s % CK_N].x, g_ck[s % CK_N].y,
                  g_ck[s % CK_N].z, g_ck[s % CK_N].indirect ? " (indirect: what guest memory held when it was recorded; the GPU reads its own copy)" : "");
    return out;
}

const char *vk_ck_describe(uint64_t s, char *out, size_t n) { return ck_text(s, out, n); }
// What a checkpointed draw bound (vk_replay.c, full checkpoints only): printed for the draws the GPU had in flight when it stopped
static char g_ck_note[CK_N][256];
void vk_ck_note(uint64_t s, const char *text) { if (s) snprintf(g_ck_note[s % CK_N], sizeof g_ck_note[0], "%s", text ? text : ""); }
static const char *ck_note(uint64_t s) { return s && g_ck[s % CK_N].seq == s ? g_ck_note[s % CK_N] : ""; }

// The driver's own binary fault dump (VK_EXT_device_fault's vendor binary, kept only with Aftermath on): Aftermath's decoder reads it too
static void save_vendor_binary(void) {
    if (!g_vk_fault_bin || !vk_am_dir() || !vkGetDeviceFaultInfoEXT) return;
    VkDeviceFaultCountsEXT c = {VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    if (vkGetDeviceFaultInfoEXT(g_vk.dev, &c, NULL) != VK_SUCCESS || !c.vendorBinarySize) { VKLOG("Aftermath: the driver gave no vendor fault binary"); return; }
    void *bin = malloc((size_t)c.vendorBinarySize); if (!bin) return;
    VkDeviceFaultAddressInfoEXT *ai = calloc(c.addressInfoCount + 1, sizeof *ai); VkDeviceFaultVendorInfoEXT *vi = calloc(c.vendorInfoCount + 1, sizeof *vi);
    VkDeviceFaultInfoEXT fi = {VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT, NULL, {0}, ai, vi, bin};
    VkResult fr = vkGetDeviceFaultInfoEXT(g_vk.dev, &c, &fi);
    if (fr == VK_SUCCESS || fr == VK_INCOMPLETE) {
        char path[1200]; snprintf(path, sizeof path, "%s\\vendor_fault.bin", vk_am_dir());
        FILE *f = fopen(path, "wb"); if (f) { fwrite(bin, 1, (size_t)c.vendorBinarySize, f); fclose(f); }
        VKLOG("Aftermath: the driver's vendor fault binary, %llu bytes: %s", (unsigned long long)c.vendorBinarySize, path);
    }
    free(bin); free(ai); free(vi);
}

void vk_lost(const char *where, VkResult r) {
    if (r != VK_ERROR_DEVICE_LOST || !g_vk.dev || atomic_exchange(&g_vk.lost, 1)) return;
    VKRED("the GPU stopped responding (device lost, first seen at %s); what it was doing:", where);
    char t[256];
    if (g_vk.has.checkpoints && vkGetQueueCheckpointData2NV) {
        VkCheckpointData2NV cd[32]; uint32_t n = 32; uint64_t lo = 0, hi = 0;
        for (uint32_t i = 0; i < n; i++) cd[i] = (VkCheckpointData2NV){VK_STRUCTURE_TYPE_CHECKPOINT_DATA_2_NV};
        AcquireSRWLockExclusive(&g_vk.qlock); vkGetQueueCheckpointData2NV(g_vk.q, &n, cd); ReleaseSRWLockExclusive(&g_vk.qlock);
        for (uint32_t i = 0; i < n; i++) {
            uint64_t s = (uint64_t)(uintptr_t)cd[i].pCheckpointMarker;
            const char *at = cd[i].stage == VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT ? "started" : cd[i].stage == VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT ? "finished everything before" : "reached";
            VKLOG("[RED]   the GPU %s %s (stage %#llx)", at, ck_text(s, t, sizeof t), (unsigned long long)cd[i].stage);   // VKLOG: VKRED stops after 8 lines a call site
            if (*ck_note(s)) VKLOG("[RED]     bound: %s", ck_note(s));
            if (cd[i].stage == VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT) lo = s; else if (s > hi) hi = s;
        }
        if (!n) VKRED("  no checkpoint was reached");
        for (uint64_t s = lo + 1; lo && s <= hi && s <= lo + 400; s++) { VKLOG("[RED]   in flight: %s", ck_text(s, t, sizeof t)); if (*ck_note(s)) VKLOG("[RED]     bound: %s", ck_note(s)); }   // all of them: VKRED printed 8
    } else VKRED("  (no checkpoints on this GPU: they need NVIDIA's VK_NV_device_diagnostic_checkpoints)");
    if (g_vk.has.device_fault && vkGetDeviceFaultInfoEXT) {
        VkDeviceFaultCountsEXT c = {VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
        VkDeviceFaultAddressInfoEXT ai[8]; VkDeviceFaultVendorInfoEXT vi[8];
        if (vkGetDeviceFaultInfoEXT(g_vk.dev, &c, NULL) == VK_SUCCESS) {
            if (c.addressInfoCount > 8) c.addressInfoCount = 8;
            if (c.vendorInfoCount > 8) c.vendorInfoCount = 8;
            c.vendorBinarySize = 0;
            VkDeviceFaultInfoEXT fi = {VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT, NULL, {0}, c.addressInfoCount ? ai : NULL, c.vendorInfoCount ? vi : NULL, NULL};
            VkResult fr = vkGetDeviceFaultInfoEXT(g_vk.dev, &c, &fi);
            static const char *const aty[] = {"none", "invalid read", "invalid write", "invalid execute", "unknown instruction pointer", "invalid instruction pointer",
                                              "instruction pointer of the fault"};
            if (fr == VK_SUCCESS || fr == VK_INCOMPLETE) {
                VKRED("  the driver's fault report: %s", fi.description[0] ? fi.description : "(no description)");
                for (uint32_t i = 0; i < c.addressInfoCount; i++)
                    VKRED("  fault address: %s at %#llx (within %#llx)", ai[i].addressType < 7 ? aty[ai[i].addressType] : "?", (unsigned long long)ai[i].reportedAddress,
                          (unsigned long long)ai[i].addressPrecision);
                for (uint32_t i = 0; i < c.vendorInfoCount; i++)
                    VKRED("  vendor fault: %s (code %#llx, data %#llx)", vi[i].description, (unsigned long long)vi[i].vendorFaultCode, (unsigned long long)vi[i].vendorFaultData);
            } else VKRED("  the driver's fault report could not be read (%d)", fr);
        } else VKRED("  the driver has no fault report for this loss");
    } else VKRED("  (this driver gives no fault report: VK_EXT_device_fault)");
    if (vk_am_on()) {   // the dump may come minutes later on a hang: wait for it here, the game cannot go on anyway
        save_vendor_binary();
        const char *e = getenv("NXRT_VK_AFTERMATH_WAIT_S");
        int got = vk_am_wait(e && *e ? atoi(e) : 300);
        VKRED("Aftermath: %s", got ? "a dump was written (tools/aftermath_read.py reads it)" : "no dump came");
    }
}

void vk_image_barrier(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to,
                      VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, ss, sa, ds, da, from, to, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img,
                               {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 0, NULL, 0, NULL, 1, &b};
    vkCmdPipelineBarrier2(cb, &di);
}
void vk_mem_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
    VkMemoryBarrier2 b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, NULL, ss, sa, ds, da};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 1, &b, 0, NULL, 0, NULL};
    vkCmdPipelineBarrier2(cb, &di);
}

// ---------- Deferred destruction behind the timeline ----------
// An object goes once every submission that may use it has completed: the slots being recorded when it was retired (the NVN submission, and one-shots of other
// threads) get their values only when committed, and one committed first used to take "assigned + 1", freeing memory the slot still being recorded then
// wrote into: images and pool buffers the GPU kept writing after the driver had reused their memory (the GPU freeze, docs/PLAN_NEXT.md step 5b).
// So an entry first waits until every slot taken before it is committed (vks_oldest_recording), then for the value assigned by then.
typedef struct { uint64_t at, presents, seq; int known; void (*fn)(void *); void *arg; } retire_t;
static retire_t *g_ret; static size_t g_nret, g_capret; static SRWLOCK g_retlk = SRWLOCK_INIT;
static int g_retire_old = -1;   // NXRT_VK_RETIRE_OLD=1: the old rule, "assigned + 1" (to compare)
void vk_retire(void (*fn)(void *), void *arg) {
    if (g_retire_old < 0) { const char *e = getenv("NXRT_VK_RETIRE_OLD"); g_retire_old = e && *e == '1'; }
    uint64_t vks_take_seq(void);
    AcquireSRWLockExclusive(&g_retlk);
    if (g_nret == g_capret) { g_capret = g_capret ? g_capret * 2 : 256; g_ret = realloc(g_ret, g_capret * sizeof *g_ret); }
    g_ret[g_nret++] = g_retire_old ? (retire_t){atomic_load(&g_vk.tl_assigned) + 1, atomic_load(&g_vk.presents), 0, 1, fn, arg}
                                   : (retire_t){0, atomic_load(&g_vk.presents), vks_take_seq(), 0, fn, arg};
    ReleaseSRWLockExclusive(&g_retlk);
}
void vk_retire_poll(void) {
    uint64_t done = g_vk.dev ? vk_tl_value() : UINT64_MAX;
    uint64_t vks_oldest_recording(uint64_t *assigned);
    uint64_t asg = 0, oldest = g_vk.dev ? vks_oldest_recording(&asg) : UINT64_MAX;
    if (g_vk.dev && oldest == UINT64_MAX && atomic_load(&g_vk.tl_submitted) == asg && done >= asg) done = UINT64_MAX;   // idle (no slot being recorded either): everything may go
    for (;;) {
        retire_t run = {0};
        AcquireSRWLockExclusive(&g_retlk);
        uint64_t pr = atomic_load(&g_vk.presents); int presenting = atomic_load(&g_vk.presenting);
        for (size_t i = 0; i < g_nret; i++) if (!g_ret[i].known && g_ret[i].seq < oldest) { g_ret[i].at = asg; g_ret[i].known = 1; }   // every slot that may use it is committed
        for (size_t i = 0; i < g_nret; i++) if (g_ret[i].known && g_ret[i].at <= done && (!presenting || pr >= g_ret[i].presents + 3 || done == UINT64_MAX)) { run = g_ret[i]; g_ret[i] = g_ret[--g_nret]; break; }
        ReleaseSRWLockExclusive(&g_retlk);
        if (!run.fn) break;
        run.fn(run.arg);
    }
}
