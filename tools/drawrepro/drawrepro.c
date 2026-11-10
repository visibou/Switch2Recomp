// drawrepro: one draw of a vertex + fragment SPIR-V pair, outside the game, to reproduce a GPU fault in seconds (docs/DEBUG_TOOLS.md, "drawrepro").
//   drawrepro VS.spv FS.spv [--verts 4] [--inst 2] [--topo strip|list] [--rate inst|vert] [--fill F] [--fillbits HEX] [--zero-from BYTES] [--repeat N] [--timeout-ms 5000]
// Everything the shaders bind is read from their SPIR-V: descriptor sets and bindings, image dimension / arrayness / depth, uniform and storage
// buffers, texel buffers, vertex inputs. Each gets a small dummy resource: images 4x4 (3D 4x4x4, cube 6 faces), buffers 64 KB, every word set to
// --fill (default 0.0) or --fillbits. Vertex inputs: one stream per location, 16-byte stride, --rate. Target: 64x64 B10G11R11 colour + D24S8.
// Prints OK, TIMEOUT or DEVICE_LOST (exit code 0, 2, 3). Nothing here knows any game.
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define FN_I(X) X(vkCreateInstance) X(vkEnumerateInstanceVersion)
#define FN_P(X) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceFeatures2) X(vkGetPhysicalDeviceQueueFamilyProperties) \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkCreateDevice) X(vkGetDeviceProcAddr) X(vkGetPhysicalDeviceFormatProperties)
#define FN_D(X) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkBindBufferMemory) X(vkMapMemory) \
    X(vkCreateImage) X(vkGetImageMemoryRequirements) X(vkBindImageMemory) X(vkCreateImageView) X(vkCreateSampler) X(vkCreateBufferView) \
    X(vkCreateShaderModule) X(vkCreateDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkCreateGraphicsPipelines) X(vkCreateDescriptorPool) \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreateCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) \
    X(vkCmdPipelineBarrier2) X(vkCmdBeginRendering) X(vkCmdEndRendering) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdBindVertexBuffers) \
    X(vkCmdDraw) X(vkCreateFence) X(vkQueueSubmit) X(vkWaitForFences) X(vkCmdClearColorImage)
static PFN_vkGetInstanceProcAddr gipa;
#define DECL(f) static PFN_##f f;
FN_I(DECL) FN_P(DECL) FN_D(DECL)

static VkDevice dev; static VkPhysicalDevice pd; static VkPhysicalDeviceMemoryProperties mp;
static void die(const char *m, int r) { printf("ERROR %s (%d)\n", m, r); exit(1); }
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) die(#x, r_); } while (0)

static uint32_t *load(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) die(p, 0); fseek(f, 0, SEEK_END); long l = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *w = malloc((size_t)l); if (fread(w, 1, (size_t)l, f) != (size_t)l) die("read", 0); fclose(f); *n = (size_t)l / 4; return w;
}
static uint32_t memtype(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    die("memory type", 0); return 0;
}
static uint32_t g_fill;   // the word every buffer and image is filled with
static uint32_t g_zero_from = 0xffffffff;   // --zero-from: uniform buffer bytes from this offset stay zero (the runtime's zero word for nxs_spv AbsGuard)
static VkBuffer mkbuf(VkDeviceSize n, VkBufferUsageFlags u) {
    VkBuffer b; VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, u | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE};
    CK(vkCreateBuffer(dev, &ci, NULL, &b)); VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VkDeviceMemory m; CK(vkAllocateMemory(dev, &ai, NULL, &m)); CK(vkBindBufferMemory(dev, b, m, 0));
    uint32_t *p; CK(vkMapMemory(dev, m, 0, n, 0, (void **)&p)); for (VkDeviceSize i = 0; i < n / 4; i++) p[i] = (u & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) && i * 4 >= g_zero_from ? 0 : g_fill;
    return b;
}

// ---------- SPIR-V reflection: what each module binds and reads ----------
enum { MAXID = 1 << 20 };
typedef struct { int kind; uint32_t dim, depth, arrayed, ms, sampled, ctype; } ty_t;   // kind: 1 image, 2 sampled image, 3 struct (block), 4 pointer, 5 float/int, 6 vector, 7 array
static ty_t *T; static uint32_t *ptr_to, *ptr_sc, *var_ty, *var_sc, *dset, *dbind, *dloc, *dbuiltin, *is_block, *is_bufblock;
typedef struct { uint32_t set, binding; VkDescriptorType type; uint32_t img; } bind_t;
static bind_t B[256]; static int nB;
static uint32_t L[32], Lct[32]; static int nL;   // vertex inputs: location, component type (0 float, 1 sint, 2 uint)
static void reflect(const uint32_t *w, size_t n, int vs) {
    memset(T, 0, sizeof(ty_t) * MAXID);
    for (size_t i = 0; i < MAXID; i++) dset[i] = dbind[i] = dloc[i] = dbuiltin[i] = 0xffffffff, is_block[i] = is_bufblock[i] = 0;
    for (size_t i = 5; i < n; ) {
        uint32_t op = w[i] & 0xffff, ln = w[i] >> 16; if (!ln) break;
        const uint32_t *a = w + i + 1;
        switch (op) {
        case 71: if (a[1] == 34) dset[a[0]] = a[2]; else if (a[1] == 33) dbind[a[0]] = a[2]; else if (a[1] == 30) dloc[a[0]] = a[2]; else if (a[1] == 11) dbuiltin[a[0]] = a[2];
                 else if (a[1] == 2) is_block[a[0]] = 1; else if (a[1] == 3) is_bufblock[a[0]] = 1; break;   // OpDecorate
        case 21: T[a[0]].kind = 5; T[a[0]].ctype = a[2] ? 1 : 2; break;            // OpTypeInt (signedness)
        case 22: T[a[0]].kind = 5; T[a[0]].ctype = 0; break;                       // OpTypeFloat
        case 23: T[a[0]].kind = 6; T[a[0]].ctype = T[a[1]].ctype; break;           // OpTypeVector
        case 25: T[a[0]] = (ty_t){1, a[2], a[3], a[4], a[5], a[6], T[a[1]].ctype}; break;   // OpTypeImage
        case 27: T[a[0]] = T[a[1]]; T[a[0]].kind = 2; break;                       // OpTypeSampledImage
        case 28: case 29: T[a[0]].kind = 7; ptr_to[a[0]] = a[1]; break;            // OpTypeArray / RuntimeArray
        case 30: T[a[0]].kind = 3; break;                                          // OpTypeStruct
        case 32: T[a[0]].kind = 4; ptr_sc[a[0]] = a[1]; ptr_to[a[0]] = a[2]; break; // OpTypePointer
        case 59: var_ty[a[1]] = a[0]; var_sc[a[1]] = a[2];                          // OpVariable
            if (a[2] == 1 && vs && dloc[a[1]] != 0xffffffff && dbuiltin[a[1]] == 0xffffffff && nL < 32) { uint32_t t = ptr_to[a[0]]; L[nL] = dloc[a[1]]; Lct[nL++] = T[t].ctype; }
            if (dbind[a[1]] != 0xffffffff && nB < 256) {
                uint32_t t = ptr_to[a[0]]; while (T[t].kind == 7) t = ptr_to[t];
                bind_t b = {dset[a[1]] == 0xffffffff ? 0 : dset[a[1]], dbind[a[1]], VK_DESCRIPTOR_TYPE_MAX_ENUM, t};
                if (T[t].kind == 2) b.type = T[t].dim == 5 ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                else if (T[t].kind == 1) b.type = T[t].dim == 5 ? (T[t].sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
                                                               : T[t].sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                else if (T[t].kind == 3) b.type = a[2] == 12 || is_bufblock[t] ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                int dup = 0; for (int k = 0; k < nB; k++) dup |= B[k].set == b.set && B[k].binding == b.binding;
                if (!dup && b.type != VK_DESCRIPTOR_TYPE_MAX_ENUM) B[nB++] = b;
            }
            break;
        }
        i += ln;
    }
}

static VkImageView mkimage(const ty_t *t, int storage) {
    int cube = t->dim == 3, d3 = t->dim == 2, d1 = t->dim == 0, depth = t->depth == 1 && !storage;
    VkFormat f = depth ? VK_FORMAT_D16_UNORM : t->ctype == 1 ? VK_FORMAT_R32G32B32A32_SINT : t->ctype == 2 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SFLOAT;
    uint32_t layers = cube ? (t->arrayed ? 12 : 6) : t->arrayed ? 2 : 1;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0, d1 ? VK_IMAGE_TYPE_1D : d3 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D, f,
                            {4, d1 ? 1 : 4, d3 ? 4 : 1}, 1, layers, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                            (storage ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT) | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage im; CK(vkCreateImage(dev, &ci, NULL, &im)); VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, im, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VkDeviceMemory m; CK(vkAllocateMemory(dev, &ai, NULL, &m)); CK(vkBindImageMemory(dev, im, m, 0));
    VkImageViewType vt = cube ? (t->arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE) : d3 ? VK_IMAGE_VIEW_TYPE_3D : d1 ? (t->arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D)
                       : t->arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, im, vt, f, {0}, {depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
    VkImageView v; CK(vkCreateImageView(dev, &vi, NULL, &v));
    static VkImage pend[512]; static int npend; (void)pend; (void)npend;
    extern void note_image(VkImage, int, uint32_t, int); note_image(im, depth, layers, storage);
    return v;
}
static struct { VkImage im; int depth; uint32_t layers; int storage; } IMG[512]; static int nIMG;
void note_image(VkImage im, int depth, uint32_t layers, int storage) { if (nIMG < 512) { IMG[nIMG].im = im; IMG[nIMG].depth = depth; IMG[nIMG].layers = layers; IMG[nIMG].storage = storage; nIMG++; } }

int main(int argc, char **argv) {
    if (argc < 3) { printf("usage: drawrepro VS.spv FS.spv [--verts 4] [--inst 2] [--topo strip|list] [--rate inst|vert] [--fill F] [--fillbits HEX] [--repeat N] [--timeout-ms 5000]\n"); return 1; }
    uint32_t verts = 4, inst = 2, repeat = 1, timeout_ms = 5000; int strip = 1, inst_rate = 1; float fill = 0.0f; memcpy(&g_fill, &fill, 4);
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--verts")) verts = (uint32_t)atoi(argv[i + 1]); else if (!strcmp(argv[i], "--inst")) inst = (uint32_t)atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--topo")) strip = !strcmp(argv[i + 1], "strip"); else if (!strcmp(argv[i], "--rate")) inst_rate = !strcmp(argv[i + 1], "inst");
        else if (!strcmp(argv[i], "--fill")) { fill = (float)atof(argv[i + 1]); memcpy(&g_fill, &fill, 4); } else if (!strcmp(argv[i], "--fillbits")) g_fill = (uint32_t)strtoul(argv[i + 1], NULL, 16);
        else if (!strcmp(argv[i], "--zero-from")) g_zero_from = (uint32_t)atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--repeat")) repeat = (uint32_t)atoi(argv[i + 1]); else if (!strcmp(argv[i], "--timeout-ms")) timeout_ms = (uint32_t)atoi(argv[i + 1]);
    }
    HMODULE lib = LoadLibraryW(L"vulkan-1.dll"); if (!lib) die("vulkan-1.dll", 0);
    gipa = (PFN_vkGetInstanceProcAddr)(void *)GetProcAddress(lib, "vkGetInstanceProcAddr");
#define LI(f) f = (PFN_##f)gipa(NULL, #f);
    FN_I(LI)
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "drawrepro", 1, "drawrepro", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance in; CK(vkCreateInstance(&ici, NULL, &in));
#define LP(f) f = (PFN_##f)gipa(in, #f);
    FN_P(LP)
    uint32_t npd = 8; VkPhysicalDevice pds[8]; CK(vkEnumeratePhysicalDevices(in, &npd, pds));
    VkPhysicalDeviceProperties pp; pd = pds[0];
    for (uint32_t i = 0; i < npd; i++) { vkGetPhysicalDeviceProperties(pds[i], &pp); if (pp.vendorID == 0x10de) { pd = pds[i]; break; } }
    vkGetPhysicalDeviceProperties(pd, &pp); vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    printf("device: %s, driver %u.%u\n", pp.deviceName, pp.driverVersion >> 22, (pp.driverVersion >> 14) & 0xff);
    // every supported core feature on (the runtime enables what the device has, robustBufferAccess included on NVIDIA)
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    f12.bufferDeviceAddressCaptureReplay = VK_FALSE;
    uint32_t nq = 8; VkQueueFamilyProperties qp[8]; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qp);
    uint32_t qf = 0; while (qf < nq && !(qp[qf].queueFlags & VK_QUEUE_GRAPHICS_BIT)) qf++;
    float prio = 1; VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qf, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qci};
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
#define LD(f) f = (PFN_##f)vkGetDeviceProcAddr(dev, #f);
    FN_D(LD)
    VkQueue q; vkGetDeviceQueue(dev, qf, 0, &q);

    T = calloc(MAXID, sizeof(ty_t)); ptr_to = calloc(MAXID, 4); ptr_sc = calloc(MAXID, 4); var_ty = calloc(MAXID, 4); var_sc = calloc(MAXID, 4);
    dset = calloc(MAXID, 4); dbind = calloc(MAXID, 4); dloc = calloc(MAXID, 4); dbuiltin = calloc(MAXID, 4); is_block = calloc(MAXID, 4); is_bufblock = calloc(MAXID, 4);
    size_t nvs, nfs; uint32_t *wvs = load(argv[1], &nvs), *wfs = load(argv[2], &nfs);
    reflect(wvs, nvs, 1); int nBvs = nB; ty_t *Tvs = malloc(sizeof(ty_t) * 256);
    for (int i = 0; i < nB; i++) Tvs[i] = T[B[i].img];
    reflect(wfs, nfs, 0);
    ty_t *Tall = malloc(sizeof(ty_t) * 256); for (int i = 0; i < nBvs; i++) Tall[i] = Tvs[i]; for (int i = nBvs; i < nB; i++) Tall[i] = T[B[i].img];
    printf("bindings %d (vertex %d), vertex inputs %d\n", nB, nBvs, nL);

    VkShaderModule mvs, mfs;
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, nvs * 4, wvs}; CK(vkCreateShaderModule(dev, &smi, NULL, &mvs));
    smi.codeSize = nfs * 4; smi.pCode = wfs; CK(vkCreateShaderModule(dev, &smi, NULL, &mfs));

    // descriptor set layouts: one per set number, every binding visible to both stages
    uint32_t nsets = 0; for (int i = 0; i < nB; i++) if (B[i].set + 1 > nsets) nsets = B[i].set + 1;
    VkDescriptorSetLayout dsl[8]; VkDescriptorSet ds[8];
    VkDescriptorPoolSize ps[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 256}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64},
                                 {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 64}, {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 64}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 64}};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 8, 7, ps}; VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
    VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0, VK_FILTER_LINEAR, VK_FILTER_LINEAR, VK_SAMPLER_MIPMAP_MODE_NEAREST,
                               VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0, VK_FALSE, 1, VK_FALSE, 0, 0, 0.25f};
    VkSampler samp; CK(vkCreateSampler(dev, &sci, NULL, &samp));
    for (uint32_t s = 0; s < nsets; s++) {
        VkDescriptorSetLayoutBinding lb[256]; uint32_t n = 0;
        for (int i = 0; i < nB; i++) if (B[i].set == s) lb[n++] = (VkDescriptorSetLayoutBinding){B[i].binding, B[i].type, 1, VK_SHADER_STAGE_ALL_GRAPHICS, NULL};
        VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, n, lb}; CK(vkCreateDescriptorSetLayout(dev, &li, NULL, &dsl[s]));
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl[s]}; CK(vkAllocateDescriptorSets(dev, &ai, &ds[s]));
    }
    for (int i = 0; i < nB; i++) {
        VkWriteDescriptorSet wr = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds[B[i].set], B[i].binding, 0, 1, B[i].type};
        VkDescriptorImageInfo ii; VkDescriptorBufferInfo bi; VkBufferView bv;
        switch (B[i].type) {
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            ii = (VkDescriptorImageInfo){samp, mkimage(&Tall[i], B[i].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE), B[i].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            wr.pImageInfo = &ii; break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: bi = (VkDescriptorBufferInfo){mkbuf(65536, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT), 0, 65536}; wr.pBufferInfo = &bi; break;
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: bi = (VkDescriptorBufferInfo){mkbuf(1 << 20, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT), 0, 1 << 20}; wr.pBufferInfo = &bi; break;
        default: {
            int st = B[i].type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
            VkBuffer b = mkbuf(65536, st ? VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT : VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT);
            VkBufferViewCreateInfo vci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, b, Tall[i].ctype == 1 ? VK_FORMAT_R32G32B32A32_SINT : Tall[i].ctype == 2 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SFLOAT, 0, VK_WHOLE_SIZE};
            CK(vkCreateBufferView(dev, &vci, NULL, &bv)); wr.pTexelBufferView = &bv; }
        }
        vkUpdateDescriptorSets(dev, 1, &wr, 0, NULL);
    }
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, nsets, dsl}; VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));

    // vertex input: one stream per location
    VkVertexInputBindingDescription vb[32]; VkVertexInputAttributeDescription va[32]; VkBuffer vbuf[32]; VkDeviceSize voff[32];
    for (int i = 0; i < nL; i++) {
        vb[i] = (VkVertexInputBindingDescription){(uint32_t)i, 16, inst_rate ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX};
        va[i] = (VkVertexInputAttributeDescription){L[i], (uint32_t)i, Lct[i] == 1 ? VK_FORMAT_R32G32B32A32_SINT : Lct[i] == 2 ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R32G32B32A32_SFLOAT, 0};
        vbuf[i] = mkbuf(4096, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT); voff[i] = 0;
    }
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, mvs, "main"},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, mfs, "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, NULL, 0, (uint32_t)nL, vb, (uint32_t)nL, va};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, strip ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkViewport vp = {0, 0, 64, 64, 0, 1}; VkRect2D sc = {{0, 0}, {64, 64}};
    VkPipelineViewportStateCreateInfo vps = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, &vp, 1, &sc};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, 0, 0, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, 0, 0, 0, 0, 1};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo dss = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba = {0, 0, 0, 0, 0, 0, 0, 0xf};
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, 0, 0, 1, &cba};
    VkFormat cf = VK_FORMAT_B10G11R11_UFLOAT_PACK32, df = VK_FORMAT_D24_UNORM_S8_UINT;
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, 1, &cf, df, df};
    VkGraphicsPipelineCreateInfo gci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri, 0, 2, st, &vi, &ia, NULL, &vps, &rs, &ms, &dss, &cb, NULL, pl};
    VkPipeline pipe; CK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gci, NULL, &pipe));

    // targets
    VkImageView tv[2]; VkImage ti[2];
    for (int k = 0; k < 2; k++) {
        VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, k ? df : cf, {64, 64, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                                k ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, VK_SHARING_MODE_EXCLUSIVE};
        CK(vkCreateImage(dev, &ci, NULL, &ti[k])); VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, ti[k], &mr);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, memtype(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        VkDeviceMemory m; CK(vkAllocateMemory(dev, &ai, NULL, &m)); CK(vkBindImageMemory(dev, ti[k], m, 0));
        VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, ti[k], VK_IMAGE_VIEW_TYPE_2D, k ? df : cf, {0}, {k ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        CK(vkCreateImageView(dev, &vci, NULL, &tv[k]));
    }
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, qf}; VkCommandPool cp; CK(vkCreateCommandPool(dev, &cpi, NULL, &cp));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1}; VkCommandBuffer cmd; CK(vkAllocateCommandBuffers(dev, &cai, &cmd));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cmd, &bi));
    // layouts: sampled images to SHADER_READ_ONLY (cleared to the fill value first), targets to attachment layouts
    VkImageMemoryBarrier2 br[520]; uint32_t nb = 0;
    for (int i = 0; i < nIMG; i++) br[nb++] = (VkImageMemoryBarrier2){VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, IMG[i].storage ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, IMG[i].im,
        {IMG[i].depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, IMG[i].layers}};
    for (int k = 0; k < 2; k++) br[nb++] = (VkImageMemoryBarrier2){VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, k ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, ti[k],
        {k ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 0, NULL, 0, NULL, nb, br}; vkCmdPipelineBarrier2(cmd, &di);
    VkRenderingAttachmentInfo ca = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, tv[0], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 0, NULL, 0, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingAttachmentInfo da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, tv[1], VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 0, NULL, 0, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo rinfo = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {64, 64}}, 1, 0, 1, &ca, &da, &da};
    vkCmdBeginRendering(cmd, &rinfo);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, nsets, ds, 0, NULL);
    if (nL) vkCmdBindVertexBuffers(cmd, 0, (uint32_t)nL, vbuf, voff);
    for (uint32_t r = 0; r < repeat; r++) vkCmdDraw(cmd, verts, inst, 0, 0);
    vkCmdEndRendering(cmd);
    CK(vkEndCommandBuffer(cmd));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fe; CK(vkCreateFence(dev, &fci, NULL, &fe));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmd}; CK(vkQueueSubmit(q, 1, &si, fe));
    VkResult r = vkWaitForFences(dev, 1, &fe, VK_TRUE, (uint64_t)timeout_ms * 1000000ull);
    if (r == VK_SUCCESS) { printf("OK: %u draw(s) of %u vertices x %u instances finished\n", repeat, verts, inst); return 0; }
    if (r == VK_TIMEOUT) { printf("TIMEOUT: the GPU did not finish in %u ms\n", timeout_ms); fflush(stdout); TerminateProcess(GetCurrentProcess(), 2); }
    printf("DEVICE_LOST (%d)\n", r); fflush(stdout); TerminateProcess(GetCurrentProcess(), 3); return 3;
}
