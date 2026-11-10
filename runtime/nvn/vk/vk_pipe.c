// Pipelines for the replay (vk_pipe.h). The NVN → Vulkan rules follow nxnative's Metal replay (nvn_mtl.m pso_desc and its helpers) where Metal and
// Vulkan meet; the enumerant values are the NVN header's and the Vulkan specification's.
#include "vk_pipe.h"
#include "vk_hmap.h"
#include <process.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void vkpc_note_created(int n);
const wchar_t *vkpc_path(void);

vkp_dyn g_vkp_dyn;
static vk_hmap g_layouts = {sizeof(VkDescriptorSetLayout) * 6}, g_gfx = {sizeof(vkp_gkey)}, g_cs = {sizeof(void *)}, g_cs_layouts = {sizeof(VkDescriptorSetLayout)};
static vkp_stats_t g_st;
static int g_failed_mark;
#define PIPE_FAILED ((void *)&g_failed_mark)
// M6 phase B: the maps are shared by the replay and the prewarm threads (below); pipelines are made outside the lock, the first one put stays
static SRWLOCK g_pl = SRWLOCK_INIT;
static __thread int t_prewarm;   // this thread is a prewarm thread: its pipelines count apart and are not listed again
static void manifest_note_gfx(const vkp_gkey *k);
static void manifest_note_cs(const vks_shader *cs);

// ---------- Translation ----------
VkCompareOp vkp_compare(int f) { return f >= 1 && f <= 8 ? (VkCompareOp)(f - 1) : VK_COMPARE_OP_ALWAYS; }
VkStencilOp vkp_stencil_op(int o) { return o >= 1 && o <= 8 ? (VkStencilOp)(o - 1) : VK_STENCIL_OP_KEEP; }
VkBlendFactor vkp_blend_factor(int f) {
    switch (f) {
    case NVN_BLEND_FUNC_ZERO: return VK_BLEND_FACTOR_ZERO; case NVN_BLEND_FUNC_ONE: return VK_BLEND_FACTOR_ONE;
    case NVN_BLEND_FUNC_SRC_COLOR: return VK_BLEND_FACTOR_SRC_COLOR; case NVN_BLEND_FUNC_ONE_MINUS_SRC_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case NVN_BLEND_FUNC_SRC_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA; case NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case NVN_BLEND_FUNC_DST_ALPHA: return VK_BLEND_FACTOR_DST_ALPHA; case NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case NVN_BLEND_FUNC_DST_COLOR: return VK_BLEND_FACTOR_DST_COLOR; case NVN_BLEND_FUNC_ONE_MINUS_DST_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case NVN_BLEND_FUNC_SRC_ALPHA_SATURATE: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case NVN_BLEND_FUNC_SRC1_COLOR: return VK_BLEND_FACTOR_SRC1_COLOR; case NVN_BLEND_FUNC_ONE_MINUS_SRC1_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case NVN_BLEND_FUNC_SRC1_ALPHA: return VK_BLEND_FACTOR_SRC1_ALPHA; case NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case NVN_BLEND_FUNC_CONSTANT_COLOR: return VK_BLEND_FACTOR_CONSTANT_COLOR; case NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case NVN_BLEND_FUNC_CONSTANT_ALPHA: return VK_BLEND_FACTOR_CONSTANT_ALPHA; case NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    }
    return VK_BLEND_FACTOR_ONE;   // nvn_mtl.m bfac's default
}
VkBlendOp vkp_blend_op(int e) { return e == 2 ? VK_BLEND_OP_SUBTRACT : e == 3 ? VK_BLEND_OP_REVERSE_SUBTRACT : e == 4 ? VK_BLEND_OP_MIN : e == 5 ? VK_BLEND_OP_MAX : VK_BLEND_OP_ADD; }
int vkp_blend_is_src1(int f) { return f >= NVN_BLEND_FUNC_SRC1_COLOR && f <= NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA; }
int vkp_blend_zero_src1(int f) {
    switch (f) {
    case NVN_BLEND_FUNC_SRC1_COLOR: case NVN_BLEND_FUNC_SRC1_ALPHA: return NVN_BLEND_FUNC_ZERO;
    case NVN_BLEND_FUNC_ONE_MINUS_SRC1_COLOR: case NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA: return NVN_BLEND_FUNC_ONE;
    default: return f;
    }
}
// nvn_mtl.m adv_blend: the Porter-Duff modes with uncorrelated overlap are sums of the source and destination weighted by 1, 0, As, Ad and their
// complements (premultiplied colours); a source that is not premultiplied is multiplied by As first. Other modes keep the state's functions (logged)
blend_st vkp_adv_blend(blend_st b) {
    if (!b.adv) return b;
    enum { Z = NVN_BLEND_FUNC_ZERO, O = NVN_BLEND_FUNC_ONE, SA = NVN_BLEND_FUNC_SRC_ALPHA, ISA = NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA, DA = NVN_BLEND_FUNC_DST_ALPHA, IDA = NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA };
    static const uint8_t pd[][4] = {
        [NVN_BLEND_ADVANCED_MODE_ZERO] = {Z, Z, Z, Z},        [NVN_BLEND_ADVANCED_MODE_SRC] = {O, Z, O, Z},           [NVN_BLEND_ADVANCED_MODE_DST] = {Z, O, Z, O},
        [NVN_BLEND_ADVANCED_MODE_SRC_OVER] = {O, ISA, O, ISA}, [NVN_BLEND_ADVANCED_MODE_DST_OVER] = {IDA, O, IDA, O},
        [NVN_BLEND_ADVANCED_MODE_SRC_IN] = {DA, Z, DA, Z},     [NVN_BLEND_ADVANCED_MODE_DST_IN] = {Z, SA, Z, SA},
        [NVN_BLEND_ADVANCED_MODE_SRC_OUT] = {IDA, Z, IDA, Z},  [NVN_BLEND_ADVANCED_MODE_DST_OUT] = {Z, ISA, Z, ISA},
        [NVN_BLEND_ADVANCED_MODE_SRC_ATOP] = {DA, ISA, Z, O},  [NVN_BLEND_ADVANCED_MODE_DST_ATOP] = {IDA, SA, O, Z},
        [NVN_BLEND_ADVANCED_MODE_XOR] = {IDA, ISA, IDA, ISA},  [NVN_BLEND_ADVANCED_MODE_PLUS] = {O, O, O, O},
        [NVN_BLEND_ADVANCED_MODE_PLUS_CLAMPED] = {O, O, O, O}, [NVN_BLEND_ADVANCED_MODE_PLUS_CLAMPED_ALPHA] = {O, O, O, O},
    };
    const uint8_t *f = b.adv < sizeof pd / sizeof *pd && pd[b.adv][0] ? pd[b.adv] : NULL;
    if (!f || b.adv_ovl || (b.adv_npm && f[0] != O && f[0] != Z)) {
        static uint64_t seen; if (!(seen >> (b.adv & 63) & 1)) { seen |= 1ull << (b.adv & 63);
            VKRED("advanced blend mode %#x (overlap %d, %spremultiplied source) has no fixed-function equivalent: the state's blend functions are used", (unsigned)b.adv, (int)b.adv_ovl, b.adv_npm ? "not " : ""); }
        return b;
    }
    b.src_rgb = b.adv_npm && f[0] == O ? SA : f[0]; b.dst_rgb = f[1]; b.src_a = f[2]; b.dst_a = f[3];
    b.eq_rgb = b.eq_a = NVN_BLEND_EQUATION_ADD;
    return b;
}
VkPrimitiveTopology vkp_topology(int p, int *expand) {
    *expand = VKP_EXPAND_NONE;
    switch (p) {
    case NVN_DRAW_PRIMITIVE_POINTS: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case NVN_DRAW_PRIMITIVE_LINES: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case NVN_DRAW_PRIMITIVE_LINE_LOOP: *expand = VKP_EXPAND_LINE_LOOP; return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case NVN_DRAW_PRIMITIVE_LINE_STRIP: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case NVN_DRAW_PRIMITIVE_TRIANGLES: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case NVN_DRAW_PRIMITIVE_TRIANGLE_FAN: case NVN_DRAW_PRIMITIVE_POLYGON: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case NVN_DRAW_PRIMITIVE_QUADS: *expand = VKP_EXPAND_QUADS; return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case NVN_DRAW_PRIMITIVE_QUAD_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;   // the same triangles (only flat shading's provoking vertex differs)
    case NVN_DRAW_PRIMITIVE_LINES_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_LINE_STRIP_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_TRIANGLES_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_PATCHES: return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    }
    *expand = VKP_EXPAND_UNSUPPORTED; return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
}
int vkp_topology_class(VkPrimitiveTopology t) {
    switch (t) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: return 0;
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY: case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: return 1;
    case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST: return 3;
    default: return 2;
    }
}
VkCullModeFlags vkp_cull(int f) { return f == NVN_FACE_FRONT ? VK_CULL_MODE_FRONT_BIT : f == NVN_FACE_BACK ? VK_CULL_MODE_BACK_BIT : f == NVN_FACE_FRONT_AND_BACK ? VK_CULL_MODE_FRONT_AND_BACK : VK_CULL_MODE_NONE; }
// nvn_mtl.m's winding rule (draw_): counter-clockwise when the state says CCW, the other way round when the queue's raster origin is lower left. Metal and
// Vulkan both decide facing on the framebuffer after the viewport transform, and the replay's viewport puts the picture where Metal's does (vk_replay.c)
VkFrontFace vkp_front_face(int front, int queue_origin_ll) { return (front == NVN_FRONT_FACE_CCW) != (queue_origin_ll != 0) ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE; }
VkPolygonMode vkp_polygon_mode(int m) { return m == NVN_POLYGON_MODE_POINT ? VK_POLYGON_MODE_POINT : m == NVN_POLYGON_MODE_LINE ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL; }
VkLogicOp vkp_logic_op(int o) { return o >= 0 && o <= 15 ? (VkLogicOp)o : VK_LOGIC_OP_COPY; }   // NVN and Vulkan list the sixteen in the same order

// Vertex formats (nvn_mtl.m vfmt, vfmt_for_ct): NVN 0x01-0x30 are R / RG / RGB / RGBA × {8 unorm, 8 snorm, 8 uint, 8 sint, 16 float, 16 unorm, 16 snorm,
// 16 uint, 16 sint, 32 float, 32 uint, 32 sint}
static const VkFormat VF[4][12] = {
    {VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT, VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT},
    {VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM, VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT},
    {VK_FORMAT_R8G8B8_UNORM, VK_FORMAT_R8G8B8_SNORM, VK_FORMAT_R8G8B8_UINT, VK_FORMAT_R8G8B8_SINT, VK_FORMAT_R16G16B16_SFLOAT, VK_FORMAT_R16G16B16_UNORM, VK_FORMAT_R16G16B16_SNORM, VK_FORMAT_R16G16B16_UINT, VK_FORMAT_R16G16B16_SINT, VK_FORMAT_R32G32B32_SFLOAT, VK_FORMAT_R32G32B32_UINT, VK_FORMAT_R32G32B32_SINT},
    {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT, VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT},
};
static const uint8_t VB[12] = {1, 1, 1, 1, 2, 2, 2, 2, 2, 4, 4, 4};
uint32_t vkp_vertex_format_size(int f) {
    if (f >= 0x01 && f <= 0x30) return VB[(f - 1) % 12] * ((f - 1) / 12 + 1);
    if (f == 0x3D || f == 0x3E || f == 0x3F || f == 0x69 || f == 0x6A || f == 0x9A) return 4;
    if (f >= 0x51 && f <= 0x68) { int i = (f - 0x51) >> 1; return (uint32_t)((i / 3 + 1) * (1 << (i % 3))); }
    return 0;
}
VkFormat vkp_vertex_format(int f, int ct, int *exact) {
    *exact = 1;
    int n = 0, k = -1;   // components - 1, kind within the 12 (0-11)
    if (f >= 0x01 && f <= 0x30) { n = (f - 1) / 12; k = (f - 1) % 12; }
    else if (f >= 0x51 && f <= 0x68) {   // integers the fetch converts to float without normalizing: Vulkan's scaled formats (8 / 16 bit); 32 bit has none
        int i = (f - 0x51) >> 1, sgn = (f - 0x51) & 1, bits = 8 << (i % 3); n = i / 3;
        static const VkFormat S8[2][4] = {{VK_FORMAT_R8_USCALED, VK_FORMAT_R8G8_USCALED, VK_FORMAT_R8G8B8_USCALED, VK_FORMAT_R8G8B8A8_USCALED}, {VK_FORMAT_R8_SSCALED, VK_FORMAT_R8G8_SSCALED, VK_FORMAT_R8G8B8_SSCALED, VK_FORMAT_R8G8B8A8_SSCALED}};
        static const VkFormat S16[2][4] = {{VK_FORMAT_R16_USCALED, VK_FORMAT_R16G16_USCALED, VK_FORMAT_R16G16B16_USCALED, VK_FORMAT_R16G16B16A16_USCALED}, {VK_FORMAT_R16_SSCALED, VK_FORMAT_R16G16_SSCALED, VK_FORMAT_R16G16B16_SSCALED, VK_FORMAT_R16G16B16A16_SSCALED}};
        if (ct != VKS_CT_FLOAT) *exact = 0;
        if (bits == 8) return S8[sgn][n];
        if (bits == 16) return S16[sgn][n];
        *exact = 0; return VF[n][sgn ? 11 : 10];   // 32-bit: the integers themselves (no conversion to float)
    } else {
        switch (f) {
        case 0x3D: if (ct != VKS_CT_FLOAT) *exact = 0; return ct == VKS_CT_FLOAT ? VK_FORMAT_A2B10G10R10_UNORM_PACK32 : VK_FORMAT_A2B10G10R10_UINT_PACK32;
        case 0x3E: if (ct == VKS_CT_FLOAT) *exact = 0; return VK_FORMAT_A2B10G10R10_UINT_PACK32;
        case 0x69: if (ct != VKS_CT_FLOAT) *exact = 0; return ct == VKS_CT_FLOAT ? VK_FORMAT_A2B10G10R10_SNORM_PACK32 : VK_FORMAT_A2B10G10R10_SINT_PACK32;
        case 0x6A: if (ct == VKS_CT_FLOAT) *exact = 0; return VK_FORMAT_A2B10G10R10_SINT_PACK32;
        case 0x3F: if (ct != VKS_CT_FLOAT) { *exact = 0; return VK_FORMAT_R32_UINT; } return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case 0x9A: if (ct != VKS_CT_FLOAT) { *exact = 0; return VK_FORMAT_R8G8B8A8_UINT; } return VK_FORMAT_B8G8R8A8_UNORM;
        }
        return VK_FORMAT_UNDEFINED;
    }
    int is_int = k == 2 || k == 3 || k == 7 || k == 8 || k == 10 || k == 11, is_signed = k == 3 || k == 8 || k == 11;
    if (ct == VKS_CT_FLOAT) {
        if (!is_int) return VF[n][k];
        if (k >= 10) return VF[n][9];   // 32-bit integers into a float input: the same bits as floats
        *exact = 0; return VF[n][k];    // 8 / 16-bit integers into a float input: Maxwell hands over the integer's bits; Vulkan cannot (logged by the caller)
    }
    // an integer input: the format of the same layout with the input's signedness
    int want_signed = ct == VKS_CT_INT;
    if (is_int) {
        if (is_signed == want_signed) return VF[n][k];
        if (k <= 3) { *exact = 0; return VF[n][want_signed ? 3 : 2]; }   // 8 / 16-bit of the other sign: widened the other way
        if (k <= 8) { *exact = 0; return VF[n][want_signed ? 8 : 7]; }
        return VF[n][want_signed ? 11 : 10];                             // 32-bit: the same bits either way
    }
    if (k == 9) return VF[n][want_signed ? 11 : 10];   // 32-bit float into an integer input: the float's bits (exact)
    // normalized or half into an integer input: Maxwell gives the converted float's bits, Vulkan cannot; the raw bytes as integers (nvn_mtl.m's fallback)
    *exact = 0;
    int bits16 = k >= 4;
    return VF[n][bits16 ? (want_signed ? 8 : 7) : (want_signed ? 3 : 2)];
}

VkFormat vkp_vertex_format_natural(int f, int *ct) {
    int exact;
    *ct = VKS_CT_FLOAT;
    if (f >= 0x01 && f <= 0x30) { int k = (f - 1) % 12; *ct = k == 2 || k == 7 || k == 10 ? VKS_CT_UINT : k == 3 || k == 8 || k == 11 ? VKS_CT_INT : VKS_CT_FLOAT; return vkp_vertex_format(f, *ct, &exact); }
    if (f == 0x3E) { *ct = VKS_CT_UINT; return VK_FORMAT_A2B10G10R10_UINT_PACK32; }
    if (f == 0x6A) { *ct = VKS_CT_INT; return VK_FORMAT_A2B10G10R10_SINT_PACK32; }
    return vkp_vertex_format(f, VKS_CT_FLOAT, &exact);   // float formats, the scaled ones, packed float / normalized
}

// ---------- Layouts ----------
VkPipelineLayout vkp_layout(const vks_shader *const sh[5]) {
    VkDescriptorSetLayout s[6];   // the sets, then whether the fragment stage takes the alpha test's push constants (M6 phase B)
    for (int i = 0; i < 5; i++) s[i] = vksh_empty_dsl();
    int nset = 1;
    for (int i = 0; i < 5; i++) if (sh[i]) { int set = vksh_set_of(sh[i]->stage); s[set] = sh[i]->dsl; if (set + 1 > nset) nset = set + 1; }
    int pc = sh[1] && sh[1]->alpha; s[5] = (VkDescriptorSetLayout)(uintptr_t)pc;
    AcquireSRWLockExclusive(&g_pl);
    VkPipelineLayout l = (VkPipelineLayout)vk_hm_get(&g_layouts, s);
    if (l) { ReleaseSRWLockExclusive(&g_pl); return l; }
    VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, VKP_ALPHA_PC_SIZE};
    VkPipelineLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, (uint32_t)nset, s, pc ? 1u : 0u, pc ? &pr : NULL};
    if (vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &l) != VK_SUCCESS) l = VK_NULL_HANDLE;
    else { vk_hm_put(&g_layouts, s, (void *)l); g_st.layouts++; }
    ReleaseSRWLockExclusive(&g_pl);
    return l;
}
VkPipelineLayout vkp_layout_cs(const vks_shader *cs) {
    AcquireSRWLockExclusive(&g_pl);
    VkPipelineLayout l = (VkPipelineLayout)vk_hm_get(&g_cs_layouts, &cs->dsl);
    if (!l) {
        VkPipelineLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &cs->dsl, 0, NULL};
        if (vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &l) != VK_SUCCESS) l = VK_NULL_HANDLE;
        else { vk_hm_put(&g_cs_layouts, &cs->dsl, (void *)l); g_st.layouts++; }
    }
    ReleaseSRWLockExclusive(&g_pl);
    return l;
}

// ---------- Graphics pipelines ----------
static void note_time(uint64_t t0) {
    uint64_t d = now_ns() - t0;
    if (t_prewarm) { g_st.prewarm_ns += d; return; }   // made ahead of its use: no stall where it is drawn
    g_st.create_ns += d; if (d > g_st.max_ns) g_st.max_ns = d;
}
// The first of two threads that made the same pipeline keeps it in the map; the other's is destroyed. Returns the one in the map
static VkPipeline put_first(vk_hmap *m, const void *k, VkPipeline p) {
    AcquireSRWLockExclusive(&g_pl);
    void *there = vk_hm_get(m, k);
    if (there && there != PIPE_FAILED) { ReleaseSRWLockExclusive(&g_pl); if (p) vkDestroyPipeline(g_vk.dev, p, NULL); return (VkPipeline)there; }
    vk_hm_put(m, k, p ? (void *)p : PIPE_FAILED);
    ReleaseSRWLockExclusive(&g_pl);
    return p;
}
// ---------- Graphics pipeline libraries (NXRT_VK_GPL=1, VK_EXT_graphics_pipeline_library) ----------
// A pipeline is four parts: vertex input, pre-rasterization shaders, fragment shader, fragment output. Each part is compiled once, kept by what it depends on,
// and a new combination is only linked (about a millisecond against 50-250 ms for a whole pipeline: the hitch at every new place). Layouts are made with
// independent sets, so a part made with one stage's set serves every pipeline with that stage. The linked pipeline keeps its own layout (g_gpl_layouts):
// the replay binds its sets with it. Any part that cannot be made sends the pipeline down the whole-pipeline path, as before
static vk_hmap g_gpl_layouts = {sizeof(vkp_gkey)}, g_gpl_parts = {32}, g_gpl_ilay = {sizeof(VkDescriptorSetLayout) * 6};
static uint64_t g_st_gpl_linked, g_st_gpl_failed, g_st_gpl_parts, g_st_gpl_part_ns;
static int gpl_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("NXRT_VK_GPL"); v = e && atoi(e) > 0 && g_vk.has.gpl; if (v) VKLOG("pipelines are linked from parts (NXRT_VK_GPL)"); } return v; }
static void g_gpl_layout_put(const vkp_gkey *k, VkPipelineLayout l) { AcquireSRWLockExclusive(&g_pl); vk_hm_put(&g_gpl_layouts, k, (void *)l); ReleaseSRWLockExclusive(&g_pl); }
// A layout with independent sets: the given sets, VK_NULL_HANDLE where a part does not use one; push constants for the alpha test when pc
static VkPipelineLayout gpl_layout(const VkDescriptorSetLayout s5[5], int nset, int pc) {
    VkDescriptorSetLayout key[6]; memcpy(key, s5, 5 * sizeof *s5); key[5] = (VkDescriptorSetLayout)(uintptr_t)(pc | nset << 4);
    AcquireSRWLockExclusive(&g_pl);
    VkPipelineLayout l = (VkPipelineLayout)vk_hm_get(&g_gpl_ilay, key);
    if (!l) {
        VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, VKP_ALPHA_PC_SIZE};
        VkPipelineLayoutCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT, (uint32_t)nset, s5, pc ? 1u : 0u, pc ? &pr : NULL};
        if (vkCreatePipelineLayout(g_vk.dev, &ci, NULL, &l) != VK_SUCCESS) l = VK_NULL_HANDLE; else { vk_hm_put(&g_gpl_ilay, key, (void *)l); g_st.layouts++; }
    }
    ReleaseSRWLockExclusive(&g_pl);
    return l;
}
// One part, made once for its key (a 32-byte digest of everything the part depends on)
static VkPipeline gpl_part(const uint8_t key[32], VkGraphicsPipelineCreateInfo *ci, VkGraphicsPipelineLibraryFlagsEXT what) {
    AcquireSRWLockShared(&g_pl); void *hit = vk_hm_get(&g_gpl_parts, key); ReleaseSRWLockShared(&g_pl);
    if (hit == PIPE_FAILED) return VK_NULL_HANDLE;
    if (hit) return (VkPipeline)hit;
    uint64_t t0 = now_ns();
    VkGraphicsPipelineLibraryCreateInfoEXT li = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, ci->pNext, what};
    const void *saved = ci->pNext; ci->pNext = &li; VkPipelineCreateFlags sf = ci->flags;
    ci->flags |= VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT;
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r = vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, ci, NULL, &p);
    ci->pNext = saved; ci->flags = sf;
    if (r != VK_SUCCESS) { static int n; if (n++ < 8) VKRED("pipeline part %#x could not be made (%d): whole pipelines for it", what, r); p = VK_NULL_HANDLE; }
    AcquireSRWLockExclusive(&g_pl);
    void *there = vk_hm_get(&g_gpl_parts, key);
    if (there && there != PIPE_FAILED) { ReleaseSRWLockExclusive(&g_pl); if (p) vkDestroyPipeline(g_vk.dev, p, NULL); return (VkPipeline)there; }
    vk_hm_put(&g_gpl_parts, key, p ? (void *)p : PIPE_FAILED); g_st_gpl_parts++; g_st_gpl_part_ns += now_ns() - t0;
    ReleaseSRWLockExclusive(&g_pl);
    return p;
}
static void dig(uint8_t out[32], uint32_t tag, const void *a, size_t na) {   // the part's key: tag + FNV-1a over its inputs, twice with different seeds
    uint64_t h1 = 1469598103934665603ull ^ tag, h2 = 0x9E3779B97F4A7C15ull ^ tag; const uint8_t *p = a;
    for (size_t i = 0; i < na; i++) { h1 = (h1 ^ p[i]) * 1099511628211ull; h2 = (h2 ^ p[i]) * 0x100000001B3ull + 0x632BE59BD9B4E019ull; }
    memset(out, 0, 32); memcpy(out, &tag, 4); memcpy(out + 8, &h1, 8); memcpy(out + 16, &h2, 8); memcpy(out + 24, &na, 8);
}
// The four parts of ci (the whole-pipeline description vkp_graphics built) and their link
static VkPipeline gpl_link(const vkp_gkey *k, VkGraphicsPipelineCreateInfo *ci, VkPipelineLayout *out_layout) {
    int mesh = k->sh[4] && k->sh[4]->mesh, pc = k->sh[1] && k->sh[1]->alpha;
    VkDescriptorSetLayout all[5], pre[5], frag[5]; int nset = 1;
    for (int i = 0; i < 5; i++) { all[i] = vksh_empty_dsl(); pre[i] = frag[i] = VK_NULL_HANDLE; }
    for (int i = 0; i < 5; i++) if (k->sh[i]) { int set = vksh_set_of(k->sh[i]->stage); all[set] = k->sh[i]->dsl; if (set + 1 > nset) nset = set + 1; if (i == 1) frag[set] = k->sh[i]->dsl; else pre[set] = k->sh[i]->dsl; }
    // a part's layout has VK_NULL_HANDLE where it uses no set (independent sets): the link's layout has the rest, empty sets where no stage is
    VkPipelineLayout lall = gpl_layout(all, nset, pc), lpre = gpl_layout(pre, nset, pc), lfrag = gpl_layout(frag, nset, pc);
    if (!lall || !lpre || !lfrag) return VK_NULL_HANDLE;
    VkGraphicsPipelineCreateInfo c; VkPipeline parts[4]; uint8_t key[32];
    // vertex input: the input assembly and (unless dynamic) the vertex layout
    { struct { uint8_t mesh, topo, nattr, nbind; uint8_t a[sizeof k->attr + sizeof k->bind]; } in; memset(&in, 0, sizeof in);
      in.mesh = (uint8_t)mesh; in.topo = k->topo_class; in.nattr = k->nattr; in.nbind = k->nbind; memcpy(in.a, k->attr, sizeof k->attr); memcpy(in.a + sizeof k->attr, k->bind, sizeof k->bind);
      dig(key, 1, &in, sizeof in);
      c = (VkGraphicsPipelineCreateInfo){VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, NULL, 0, 0, NULL, ci->pVertexInputState, ci->pInputAssemblyState, NULL, NULL, NULL, NULL, NULL, NULL, ci->pDynamicState, VK_NULL_HANDLE, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
      parts[0] = mesh ? VK_NULL_HANDLE : gpl_part(key, &c, VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT);
      if (!mesh && !parts[0]) return VK_NULL_HANDLE; }
    // pre-rasterization: every stage but the fragment shader, viewport, rasterization, tessellation
    { VkPipelineShaderStageCreateInfo st[5]; uint32_t ns = 0; const void *keys[5] = {0};
      for (uint32_t i = 0; i < ci->stageCount; i++) if (ci->pStages[i].stage != VK_SHADER_STAGE_FRAGMENT_BIT) st[ns++] = ci->pStages[i];
      for (int i = 0; i < 5; i++) if (i != 1) keys[i] = k->sh[i];
      struct { const void *sh[5]; VkPipelineLayout l; uint8_t m1, clamp, poly, patch; } in; memset(&in, 0, sizeof in);
      memcpy(in.sh, keys, sizeof keys); in.l = lpre; in.m1 = k->depth_m1; in.clamp = k->depth_clamp; in.poly = k->poly; in.patch = k->patch_points;
      dig(key, 2, &in, sizeof in);
      c = (VkGraphicsPipelineCreateInfo){VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, ci->pNext, 0, ns, st, NULL, NULL, ci->pTessellationState, ci->pViewportState, ci->pRasterizationState, NULL, NULL, NULL, ci->pDynamicState, lpre, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
      parts[1] = gpl_part(key, &c, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT);
      if (!parts[1]) return VK_NULL_HANDLE; }
    // fragment shader: its stage, depth / stencil, multisampling (and the depth / stencil formats it renders with)
    { VkPipelineShaderStageCreateInfo st[1]; uint32_t ns = 0;
      for (uint32_t i = 0; i < ci->stageCount; i++) if (ci->pStages[i].stage == VK_SHADER_STAGE_FRAGMENT_BIT) st[ns++] = ci->pStages[i];
      const VkPipelineRenderingCreateInfo *ri = ci->pNext; VkPipelineRenderingCreateInfo fri = *ri; fri.colorAttachmentCount = 0; fri.pColorAttachmentFormats = NULL;
      struct { const void *fs; VkPipelineLayout l; uint32_t df; uint8_t hd, hs, samples, a2c; } in; memset(&in, 0, sizeof in);
      in.fs = k->sh[1]; in.l = lfrag; in.df = k->dfmt; in.hd = k->has_depth; in.hs = k->has_stencil; in.samples = k->samples; in.a2c = k->a2c;
      dig(key, 3, &in, sizeof in);
      c = (VkGraphicsPipelineCreateInfo){VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &fri, 0, ns, ns ? st : NULL, NULL, NULL, NULL, NULL, NULL, ci->pMultisampleState, ci->pDepthStencilState, NULL, ci->pDynamicState, lfrag, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
      parts[2] = gpl_part(key, &c, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT);
      if (!parts[2]) return VK_NULL_HANDLE; }
    // fragment output: the targets' formats, blending, multisampling
    { struct { uint32_t cf[8], df; uint8_t n, hd, hs, samples, a2c, logic_en, logic_op; vkp_att att[8]; } in; memset(&in, 0, sizeof in);
      memcpy(in.cf, k->cfmt, sizeof in.cf); in.df = k->dfmt; in.n = k->ncolor; in.hd = k->has_depth; in.hs = k->has_stencil; in.samples = k->samples; in.a2c = k->a2c; in.logic_en = k->logic_en; in.logic_op = k->logic_op; memcpy(in.att, k->att, sizeof in.att);
      dig(key, 4, &in, sizeof in);
      c = (VkGraphicsPipelineCreateInfo){VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, ci->pNext, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, ci->pMultisampleState, NULL, ci->pColorBlendState, ci->pDynamicState, VK_NULL_HANDLE, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
      parts[3] = gpl_part(key, &c, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT);
      if (!parts[3]) return VK_NULL_HANDLE; }
    // the link (no link-time optimization: fast)
    VkPipeline libs[4]; uint32_t nl = 0; for (int i = 0; i < 4; i++) if (parts[i]) libs[nl++] = parts[i];
    VkPipelineLibraryCreateInfoKHR lci = {VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, NULL, nl, libs};
    c = (VkGraphicsPipelineCreateInfo){VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &lci, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, lall, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(g_vk.dev, VK_NULL_HANDLE, 1, &c, NULL, &p) != VK_SUCCESS) { static int n; if (n++ < 8) VKRED("a pipeline could not be linked from its parts: made whole"); return VK_NULL_HANDLE; }
    *out_layout = lall;
    return p;
}
uint64_t vkp_gpl_stats(uint64_t *linked, uint64_t *failed, uint64_t *part_ns) { *linked = g_st_gpl_linked; *failed = g_st_gpl_failed; *part_ns = g_st_gpl_part_ns; return g_st_gpl_parts; }
static __thread struct { vkp_gkey k; VkPipeline p; VkPipelineLayout l; int ok; } t_last;   // M6 phase B: consecutive draws often share their pipeline
static int pipe_stats_wanted(const vkp_gkey *k); static void pipe_stats_report(VkPipeline p, const vkp_gkey *k);   // NXRT_VK_PIPE_STATS, below
VkPipeline vkp_graphics(const vkp_gkey *k, VkPipelineLayout *layout) {
    if (t_last.ok && !memcmp(&t_last.k, k, sizeof *k)) { *layout = t_last.l; return t_last.p; }
    *layout = vkp_layout(k->sh);
    AcquireSRWLockShared(&g_pl); void *hit = vk_hm_get(&g_gfx, k); void *gl = hit ? vk_hm_get(&g_gpl_layouts, k) : NULL; ReleaseSRWLockShared(&g_pl);
    if (hit == PIPE_FAILED) return VK_NULL_HANDLE;
    if (gl) *layout = (VkPipelineLayout)gl;
    if (hit) { t_last.k = *k; t_last.p = (VkPipeline)hit; t_last.l = *layout; t_last.ok = 1; return (VkPipeline)hit; }
    uint64_t t0 = now_ns();
    VkPipelineShaderStageCreateInfo st[5]; uint32_t ns = 0;
    int mesh = k->sh[4] && k->sh[4]->mesh;   // no vertex input, input assembly or tessellation: those states may not be given
    for (int i = 0; i < 5; i++) if (k->sh[i]) st[ns++] = (VkPipelineShaderStageCreateInfo){VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, vksh_stage_bit(k->sh[i]), k->sh[i]->mod, "main", NULL};
    VkVertexInputAttributeDescription va[16]; VkVertexInputBindingDescription vb[17]; VkVertexInputBindingDivisorDescriptionKHR vd[17]; uint32_t nvd = 0;
    for (int i = 0; i < k->nattr; i++) va[i] = (VkVertexInputAttributeDescription){k->attr[i].loc, k->attr[i].bind, (VkFormat)k->attr[i].fmt, k->attr[i].off};
    for (int i = 0; i < k->nbind; i++) {
        vb[i] = (VkVertexInputBindingDescription){i, k->bind[i].stride, (VkVertexInputRate)k->bind[i].rate};
        if (k->bind[i].rate && k->bind[i].divisor != 1) vd[nvd++] = (VkVertexInputBindingDivisorDescriptionKHR){i, k->bind[i].divisor};
    }
    VkPipelineVertexInputDivisorStateCreateInfoKHR vds = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_KHR, NULL, nvd, vd};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO, nvd ? &vds : NULL, 0, k->nbind, vb, k->nattr, va};
    static const VkPrimitiveTopology rep[4] = {VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_PATCH_LIST};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, rep[k->topo_class == VKP_TOPO_ANY ? 2 : k->topo_class & 3], VK_FALSE};
    VkPipelineTessellationStateCreateInfo ts = {VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO, NULL, 0, k->patch_points ? k->patch_points : 3};
    VkPipelineViewportDepthClipControlCreateInfoEXT dcc = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT, NULL, VK_TRUE};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, k->depth_m1 && g_vk.has.depth_clip_control ? &dcc : NULL, 0, 0, NULL, 0, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, k->depth_clamp, VK_FALSE, vkp_polygon_mode(k->poly ? k->poly : NVN_POLYGON_MODE_FILL),
                                                 VK_CULL_MODE_NONE, VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, (VkSampleCountFlagBits)(k->samples ? k->samples : 1), VK_FALSE, 0, NULL, k->a2c, VK_FALSE};
    VkPipelineDepthStencilStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba[8];
    for (int i = 0; i < k->ncolor; i++) {
        const vkp_att *a = &k->att[i];
        cba[i] = (VkPipelineColorBlendAttachmentState){a->blend, (VkBlendFactor)a->src, (VkBlendFactor)a->dst, (VkBlendOp)a->op, (VkBlendFactor)a->srca, (VkBlendFactor)a->dsta, (VkBlendOp)a->opa, a->wmask};
    }
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, k->logic_en, (VkLogicOp)k->logic_op, k->ncolor, cba, {0, 0, 0, 0}};
    VkDynamicState dyn[64]; uint32_t nd = 0;
    static const VkDynamicState always[] = {
        VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT, VK_DYNAMIC_STATE_LINE_WIDTH, VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
        VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP, VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE};
    for (unsigned i = 0; i < sizeof always / sizeof *always; i++) if (!mesh || (always[i] != VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY && always[i] != VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE)) dyn[nd++] = always[i];
    if (g_vkp_dyn.vertex_input && !mesh) dyn[nd++] = VK_DYNAMIC_STATE_VERTEX_INPUT_EXT;
    if (g_vkp_dyn.blend_enable && k->ncolor) dyn[nd++] = VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT;
    if (g_vkp_dyn.blend_eq && k->ncolor) dyn[nd++] = VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT;
    if (g_vkp_dyn.write_mask && k->ncolor) dyn[nd++] = VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT;
    if (g_vkp_dyn.logic_op_enable) dyn[nd++] = VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT;
    if (g_vkp_dyn.logic_op) dyn[nd++] = VK_DYNAMIC_STATE_LOGIC_OP_EXT;
    if (g_vkp_dyn.a2c) dyn[nd++] = VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT;
    if (g_vkp_dyn.polygon_mode) dyn[nd++] = VK_DYNAMIC_STATE_POLYGON_MODE_EXT;
    if (g_vkp_dyn.depth_clamp) dyn[nd++] = VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT;
    if (g_vkp_dyn.patch_points && k->sh[3] && !mesh) dyn[nd++] = VK_DYNAMIC_STATE_PATCH_CONTROL_POINTS_EXT;
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, nd, dyn};
    VkFormat cf[8]; for (int i = 0; i < k->ncolor; i++) cf[i] = (VkFormat)k->cfmt[i];
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, k->ncolor, cf,
                                        k->has_depth ? (VkFormat)k->dfmt : VK_FORMAT_UNDEFINED, k->has_stencil ? (VkFormat)k->dfmt : VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri, 0, ns, st, g_vkp_dyn.vertex_input || mesh ? NULL : &vi, mesh ? NULL : &ia, k->sh[3] && !mesh ? &ts : NULL, &vp, &rs, &ms, &ds, &cb, &dy, *layout, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
    VkPipeline p = VK_NULL_HANDLE;
    int cap = pipe_stats_wanted(k); if (cap) ci.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR | VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
    // NXRT_VK_GPL=1: made from four parts compiled once each and linked (gpl_link, below); the whole pipeline only when that fails
    if (!cap && gpl_on()) {
        VkPipelineLayout gl = VK_NULL_HANDLE;
        p = gpl_link(k, &ci, &gl);
        if (p) { *layout = gl; note_time(t0); g_st_gpl_linked++;
                 VkPipeline q = put_first(&g_gfx, k, p); g_gpl_layout_put(k, gl);
                 if (q == p) { if (t_prewarm) g_st.prewarmed++; else { g_st.graphics++; manifest_note_gfx(k); } }
                 t_last.ok = 0; return q; }
        g_st_gpl_failed++;
    }
    VkResult r = *layout ? vkCreateGraphicsPipelines(g_vk.dev, cap ? VK_NULL_HANDLE : g_vk.pcache, 1, &ci, NULL, &p) : VK_ERROR_INITIALIZATION_FAILED;
    if (cap && r == VK_SUCCESS) pipe_stats_report(p, k);
    note_time(t0);
    if (r != VK_SUCCESS) {
        g_st.failed++;
        VKRED("graphics pipeline (vertex %s fragment %s geometry %s, %u colour targets) could not be created (%d): its draws are skipped",
              k->sh[0] ? k->sh[0]->key : "-", k->sh[1] ? k->sh[1]->key : "-", k->sh[2] ? k->sh[2]->key : "-", k->ncolor, r);
        put_first(&g_gfx, k, VK_NULL_HANDLE); return VK_NULL_HANDLE;
    }
    VkPipeline q = put_first(&g_gfx, k, p);
    if (q == p) { if (t_prewarm) g_st.prewarmed++; else { g_st.graphics++; manifest_note_gfx(k); } vkpc_note_created(1); }
    return q;
}

// NXRT_VK_PIPE_STATS=KEY[,KEY...] (test, the GPU freeze): graphics pipelines using one of these shaders are compiled outside the pipeline cache with the driver's
// statistics and internal representations captured (VK_KHR_pipeline_executable_properties): the statistics go to the log, each text representation to
// <log>.pipe-<vertex key>-<executable>-<n>.txt. Shows what the driver made of a shader (registers, local memory, its own code if it gives it)
static int pipe_stats_wanted(const vkp_gkey *k) {
    static char pk[8][17]; static int npk = -1;
    if (npk < 0) { npk = 0; const char *e = getenv("NXRT_VK_PIPE_STATS"); while (e && *e && npk < 8) { if (strlen(e) >= 16) { memcpy(pk[npk], e, 16); pk[npk][16] = 0; npk++; } e = strchr(e, ','); if (e) e++; } }
    if (!npk || !g_vk.has.pipe_exec) return 0;
    for (int s = 0; s < 5; s++) if (k->sh[s]) for (int i = 0; i < npk; i++) if (!strcmp(k->sh[s]->key, pk[i])) return 1;
    return 0;
}
const char *nxrt_log_path(void);
static void pipe_stats_report(VkPipeline p, const vkp_gkey *k) {
    PFN_vkGetPipelineExecutablePropertiesKHR gp = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(g_vk.dev, "vkGetPipelineExecutablePropertiesKHR");
    PFN_vkGetPipelineExecutableStatisticsKHR gs = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(g_vk.dev, "vkGetPipelineExecutableStatisticsKHR");
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR gr = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)vkGetDeviceProcAddr(g_vk.dev, "vkGetPipelineExecutableInternalRepresentationsKHR");
    if (!gp || !gs || !gr) { VKRED("pipeline stats: the extension's functions are missing"); return; }
    VkPipelineInfoKHR pi = {VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, NULL, p}; uint32_t ne = 0; gp(g_vk.dev, &pi, &ne, NULL); if (ne > 8) ne = 8;
    VkPipelineExecutablePropertiesKHR ep[8]; for (uint32_t i = 0; i < ne; i++) ep[i] = (VkPipelineExecutablePropertiesKHR){VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR};
    gp(g_vk.dev, &pi, &ne, ep);
    for (uint32_t e = 0; e < ne; e++) {
        VkPipelineExecutableInfoKHR xi = {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, NULL, p, e};
        char line[2048]; size_t n = (size_t)snprintf(line, sizeof line, "pipeline stats (vs %s, fs %s) executable %u \"%s\" (stages %#x, subgroup %u):", k->sh[0] ? k->sh[0]->key : "-", k->sh[1] ? k->sh[1]->key : "-", e, ep[e].name, ep[e].stages, ep[e].subgroupSize);
        uint32_t nst = 0; gs(g_vk.dev, &xi, &nst, NULL); if (nst > 32) nst = 32;
        VkPipelineExecutableStatisticKHR st[32]; for (uint32_t i = 0; i < nst; i++) st[i] = (VkPipelineExecutableStatisticKHR){VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR};
        gs(g_vk.dev, &xi, &nst, st);
        for (uint32_t i = 0; i < nst && n < sizeof line - 80; i++) {
            if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR) n += (size_t)snprintf(line + n, sizeof line - n, " %s=%g;", st[i].name, st[i].value.f64);
            else if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR) n += (size_t)snprintf(line + n, sizeof line - n, " %s=%lld;", st[i].name, (long long)st[i].value.i64);
            else if (st[i].format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR) n += (size_t)snprintf(line + n, sizeof line - n, " %s=%s;", st[i].name, st[i].value.b32 ? "yes" : "no");
            else n += (size_t)snprintf(line + n, sizeof line - n, " %s=%llu;", st[i].name, (unsigned long long)st[i].value.u64);
        }
        VKLOG("%s", line);
        uint32_t nr = 0; gr(g_vk.dev, &xi, &nr, NULL); if (nr > 8) nr = 8;
        VkPipelineExecutableInternalRepresentationKHR ir[8]; for (uint32_t i = 0; i < nr; i++) ir[i] = (VkPipelineExecutableInternalRepresentationKHR){VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR};
        gr(g_vk.dev, &xi, &nr, ir);
        for (uint32_t i = 0; i < nr; i++) ir[i].pData = ir[i].dataSize ? malloc(ir[i].dataSize) : NULL;
        if (nr) gr(g_vk.dev, &xi, &nr, ir);
        for (uint32_t i = 0; i < nr; i++) {
            char path[1200]; const char *lp = nxrt_log_path(); snprintf(path, sizeof path, "%s.pipe-%s-%u-%u.%s", lp && *lp ? lp : "nxrt", k->sh[0] ? k->sh[0]->key : "nokey", e, i, ir[i].isText ? "txt" : "bin");
            FILE *f = ir[i].pData ? fopen(path, "wb") : NULL; if (f) { fwrite(ir[i].pData, 1, ir[i].isText && ir[i].dataSize ? strnlen((const char *)ir[i].pData, ir[i].dataSize) : ir[i].dataSize, f); fclose(f); }
            VKLOG("pipeline stats: executable %u representation %u \"%s\" (%s, %zu bytes)%s%s", e, i, ir[i].name, ir[i].description, ir[i].dataSize, f ? " written to " : "", f ? path : "");
            free(ir[i].pData);
        }
    }
}

VkPipeline vkp_compute(const vks_shader *cs, VkPipelineLayout *layout) {
    *layout = vkp_layout_cs(cs);
    AcquireSRWLockShared(&g_pl); void *hit = vk_hm_get(&g_cs, &cs); ReleaseSRWLockShared(&g_pl);
    if (hit == PIPE_FAILED) return VK_NULL_HANDLE;
    if (hit) return (VkPipeline)hit;
    uint64_t t0 = now_ns();
    VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
                                      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, cs->mod, "main", NULL}, *layout, VK_NULL_HANDLE, -1};
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r = *layout ? vkCreateComputePipelines(g_vk.dev, g_vk.pcache, 1, &ci, NULL, &p) : VK_ERROR_INITIALIZATION_FAILED;
    note_time(t0);
    if (r != VK_SUCCESS) { g_st.failed++; VKRED("compute pipeline %s could not be created (%d): its dispatches are skipped", cs->key, r); put_first(&g_cs, &cs, VK_NULL_HANDLE); return VK_NULL_HANDLE; }
    VkPipeline q = put_first(&g_cs, &cs, p);
    if (q == p) { if (t_prewarm) g_st.prewarmed++; else { g_st.compute++; manifest_note_cs(cs); } vkpc_note_created(1); }
    return q;
}

void vkp_stats(vkp_stats_t *s) { *s = g_st; }

// ---------- M6 phase B: the pipeline manifest and the prewarm ----------
// Every graphics and compute pipeline the replay makes is listed in pipelines.tsv beside the pipeline cache (vk_pcache.c's folder), device-independently: its shaders
// by key with the variants that made them, then the rest of its key in hex (trailing zeros left out). At start two threads at below-normal priority make every
// pipeline listed there and in a seed list (NXRT_VK_PIPELINE_SEED: the same format from any device, e.g. one the installer ships) whose shaders the cache has, in the
// order listed, while the game boots and loads: a pipeline made ahead is found where it is first drawn, so an empty pipeline cache no longer stalls that frame (M5-C,
// M6 phase A: single stalls up to ~2 s in the first minute). The replay's own misses are still made on the spot. NXRT_VK_PREWARM=0 turns the prewarm off,
// NXRT_VK_MANIFEST=0 the listing. Lines: G <vs> <fs> <gs> <tcs> <tes> <key hex> / C <cs>, tab-separated; a shader is "-", <key>, <key>/v<mask hex>:<types>, <key>/f<flags>
static SRWLOCK g_mf_lk = SRWLOCK_INIT; static FILE *g_mf; static int g_mf_state;   // 0 not opened yet, 1 open, -1 off
static vk_hmap g_mf_seen = {8};   // the lines listed already (by hash)
static uint64_t line_hash(const char *l) { return vk_hm_hash(l, (uint32_t)strlen(l)) | 1; }
static int manifest_path(wchar_t *out, size_t n) {
    const wchar_t *pc = vkpc_path(); if (!pc || !pc[0]) return -1;
    _snwprintf(out, n, L"%ls", pc); wchar_t *sl = wcsrchr(out, L'\\'); if (!sl) return -1;
    _snwprintf(sl + 1, n - (size_t)(sl + 1 - out), L"pipelines.tsv");
    return 0;
}
static int sh_desc(const vks_shader *s, char *o, size_t n) {
    if (!s) return snprintf(o, n, "-");
    int k = snprintf(o, n, "%s", s->key);
    if (s->vin_mask) { k += snprintf(o + k, n - k, "/v%x:", s->vin_mask); for (int l = 0; l < 32; l++) if (s->vin_mask >> l & 1) k += snprintf(o + k, n - k, "%u", s->vin_ct[l]); }
    if (s->fs_flags) k += snprintf(o + k, n - k, "/f%u", s->fs_flags);
    return k;
}
static void manifest_add(const char *line) {   // appended and flushed at once: a run that is killed keeps what it listed
    uint64_t h = line_hash(line);
    AcquireSRWLockExclusive(&g_mf_lk);
    if (!g_mf_state) {
        const char *e = getenv("NXRT_VK_MANIFEST"); wchar_t p[MAX_PATH];
        g_mf_state = (e && *e && !atoi(e)) || manifest_path(p, MAX_PATH) || !(g_mf = _wfopen(p, L"ab")) ? -1 : 1;
    }
    if (g_mf_state > 0 && !vk_hm_get(&g_mf_seen, &h)) { vk_hm_put(&g_mf_seen, &h, (void *)1); fputs(line, g_mf); fputc('\n', g_mf); fflush(g_mf); }
    ReleaseSRWLockExclusive(&g_mf_lk);
}
static void manifest_note_gfx(const vkp_gkey *k) {
    char l[2048]; int o = snprintf(l, sizeof l, "G");
    for (int i = 0; i < 5; i++) { l[o++] = '\t'; o += sh_desc(k->sh[i], l + o, sizeof l - (size_t)o); }
    const uint8_t *t = (const uint8_t *)k + offsetof(vkp_gkey, cfmt); size_t tn = sizeof *k - offsetof(vkp_gkey, cfmt);
    while (tn && !t[tn - 1]) tn--;
    l[o++] = '\t'; for (size_t i = 0; i < tn && o + 3 < (int)sizeof l; i++) o += snprintf(l + o, sizeof l - (size_t)o, "%02x", t[i]);
    manifest_add(l);
}
static void manifest_note_cs(const vks_shader *cs) { char l[64]; snprintf(l, sizeof l, "C\t%s", cs->key); manifest_add(l); }
static void manifest_close(void) {
    AcquireSRWLockExclusive(&g_mf_lk);
    if (g_mf) fclose(g_mf);
    g_mf = NULL; g_mf_state = 0; vk_hm_free(&g_mf_seen);
    ReleaseSRWLockExclusive(&g_mf_lk);
}

typedef struct { char sh[5][64]; uint8_t tail[sizeof(vkp_gkey)]; uint16_t tn; uint8_t cs; } pw_ent;
static struct { pw_ent *e; int n, cap; _Atomic int next, skipped, threads; _Atomic int stop; HANDLE th[2]; uint64_t t0; } g_pw;
static void pw_read(const wchar_t *path, const char *what) {   // the entries of one list (lines seen before are not listed again)
    FILE *f = _wfopen(path, L"rb"); if (!f) return;
    char l[2048]; int n0 = g_pw.n;
    while (fgets(l, sizeof l, f)) {
        size_t len = strlen(l); while (len && (l[len - 1] == '\n' || l[len - 1] == '\r')) l[--len] = 0;
        if (!len || (l[0] != 'G' && l[0] != 'C') || l[1] != '\t') continue;
        uint64_t h = line_hash(l);
        AcquireSRWLockExclusive(&g_mf_lk); int seen = vk_hm_get(&g_mf_seen, &h) != NULL; if (!seen) vk_hm_put(&g_mf_seen, &h, (void *)1); ReleaseSRWLockExclusive(&g_mf_lk);
        if (seen) continue;
        if (g_pw.n == g_pw.cap) { g_pw.cap = g_pw.cap ? 2 * g_pw.cap : 256; g_pw.e = realloc(g_pw.e, (size_t)g_pw.cap * sizeof *g_pw.e); }
        pw_ent *e = &g_pw.e[g_pw.n]; memset(e, 0, sizeof *e);
        char *f2 = l + 2, *tab; int i = 0;
        e->cs = l[0] == 'C';
        for (; i < (e->cs ? 1 : 6) && f2; i++) {
            tab = strchr(f2, '\t'); if (tab) *tab = 0;
            if (i < 5) snprintf(e->sh[i], sizeof e->sh[i], "%s", f2);
            else for (size_t k = 0; f2[2 * k] && f2[2 * k + 1] && k < sizeof e->tail; k++) { unsigned v; if (sscanf(f2 + 2 * k, "%2x", &v) != 1) break; e->tail[k] = (uint8_t)v; e->tn = (uint16_t)(k + 1); }
            f2 = tab ? tab + 1 : NULL;
        }
        if (strlen(e->sh[0]) < 16 && strlen(e->sh[4]) < 16) continue;   // a mesh pipeline has no vertex shader: left out of the list they were made in the race, 300 to 500 ms each
        g_pw.n++;
    }
    fclose(f);
    VKLOG("pipeline prewarm: %d pipelines listed in %ls (%s)", g_pw.n - n0, path, what);
}
static const vks_shader *pw_shader(const char *d, int *missing) {   // "-" → NULL; a missing shader sets *missing
    if (!strcmp(d, "-") || !d[0]) return NULL;
    char key[17]; memcpy(key, d, 16); key[16] = 0;
    const vks_shader *s = vksh_by_key(key);
    if (!s) { *missing = 1; return NULL; }
    const char *v = strstr(d, "/v");
    if (v) {
        char *end; uint32_t mask = (uint32_t)strtoul(v + 2, &end, 16); uint8_t ct[32] = {0};
        if (*end == ':') { end++; for (int l = 0; l < 32; l++) if (mask >> l & 1) { if (*end < '0' || *end > '2') { *missing = 1; return NULL; } ct[l] = (uint8_t)(*end++ - '0'); } }
        s = vksh_vin_variant(s, mask, ct);
    }
    const char *f = strstr(d, "/f");
    if (f) s = vksh_fs_variant(s, atoi(f + 2));
    return s;
}
static unsigned __stdcall pw_main(void *a) {
    (void)a;
    SetThreadDescription(GetCurrentThread(), L"vk pipeline prewarm");
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    t_prewarm = 1;
    for (;;) {
        int i = atomic_fetch_add(&g_pw.next, 1);
        if (i >= g_pw.n || atomic_load(&g_pw.stop)) break;
        pw_ent *e = &g_pw.e[i]; int missing = 0; VkPipelineLayout l;
        if (e->cs) { const vks_shader *cs = pw_shader(e->sh[0], &missing); if (cs) vkp_compute(cs, &l); else atomic_fetch_add(&g_pw.skipped, 1); continue; }
        vkp_gkey k; memset(&k, 0, sizeof k);
        for (int s = 0; s < 5 && !missing; s++) k.sh[s] = pw_shader(e->sh[s], &missing);
        if (missing || (!k.sh[0] && !k.sh[4])) { atomic_fetch_add(&g_pw.skipped, 1); continue; }
        memcpy((uint8_t *)&k + offsetof(vkp_gkey, cfmt), e->tail, e->tn);
        vkp_graphics(&k, &l);
    }
    if (atomic_fetch_sub(&g_pw.threads, 1) == 1 && !atomic_load(&g_pw.stop))
        VKLOG("pipeline prewarm done: %llu of %d listed made ahead (%.1f s of creation on 2 background threads), %d skipped (their shaders are not in the cache: another game version), "
              "the rest there already; %.1f s after start", (unsigned long long)g_st.prewarmed, g_pw.n, g_st.prewarm_ns / 1e9, (int)g_pw.skipped, (now_ns() - g_pw.t0) / 1e9);
    return 0;
}
void vkpl_prewarm_start(void) {
    const char *e = getenv("NXRT_VK_PREWARM");
    if (e && *e && !atoi(e)) { VKLOG("pipeline prewarm off (NXRT_VK_PREWARM=0)"); return; }
    wchar_t p[MAX_PATH];
    if (!manifest_path(p, MAX_PATH)) pw_read(p, "pipelines made here before");
    if ((e = getenv("NXRT_VK_PIPELINE_SEED")) && *e) { MultiByteToWideChar(CP_UTF8, 0, e, -1, p, MAX_PATH); pw_read(p, "the seed list, NXRT_VK_PIPELINE_SEED"); }
    if (!g_pw.n) return;
    g_pw.t0 = now_ns(); atomic_store(&g_pw.threads, 2);
    for (int i = 0; i < 2; i++) g_pw.th[i] = (HANDLE)_beginthreadex(NULL, 0, pw_main, NULL, 0, NULL);
}
static void prewarm_stop(void) {
    atomic_store(&g_pw.stop, 1);
    for (int i = 0; i < 2; i++) if (g_pw.th[i]) { WaitForSingleObject(g_pw.th[i], INFINITE); CloseHandle(g_pw.th[i]); g_pw.th[i] = NULL; }
    free(g_pw.e); memset(&g_pw, 0, sizeof g_pw);
}

int vkpl_init(void) {
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT e3 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
    VkPhysicalDeviceExtendedDynamicState2FeaturesEXT e2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_2_FEATURES_EXT, &e3};
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vid = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT, &e2};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &vid};
    vkGetPhysicalDeviceFeatures2(g_vk.pd, &f2);
    VkPhysicalDeviceExtendedDynamicState3PropertiesEXT p3 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p3};
    vkGetPhysicalDeviceProperties2(g_vk.pd, &p2);
    memset(&g_vkp_dyn, 0, sizeof g_vkp_dyn);
    // what vk_dev.c enabled (every feature of these extensions the device reports) and whose entry points loaded
    g_vkp_dyn.vertex_input = g_vk.has.vertex_input_dynamic && vid.vertexInputDynamicState && vkCmdSetVertexInputEXT;
    if (g_vk.has.eds3) {
        g_vkp_dyn.blend_enable = e3.extendedDynamicState3ColorBlendEnable && vkCmdSetColorBlendEnableEXT;
        g_vkp_dyn.blend_eq = e3.extendedDynamicState3ColorBlendEquation && vkCmdSetColorBlendEquationEXT;
        g_vkp_dyn.write_mask = e3.extendedDynamicState3ColorWriteMask && vkCmdSetColorWriteMaskEXT;
        g_vkp_dyn.logic_op_enable = e3.extendedDynamicState3LogicOpEnable && vkCmdSetLogicOpEnableEXT && g_vk.f10.logicOp;
        g_vkp_dyn.a2c = e3.extendedDynamicState3AlphaToCoverageEnable && vkCmdSetAlphaToCoverageEnableEXT;
        g_vkp_dyn.polygon_mode = e3.extendedDynamicState3PolygonMode && vkCmdSetPolygonModeEXT;
        g_vkp_dyn.depth_clamp = e3.extendedDynamicState3DepthClampEnable && vkCmdSetDepthClampEnableEXT && g_vk.f10.depthClamp;
        g_vkp_dyn.topo_unrestricted = p3.dynamicPrimitiveTopologyUnrestricted;
    }
    g_vkp_dyn.logic_op = g_vk.has.eds2_logic_op && vkCmdSetLogicOpEXT;
    g_vkp_dyn.patch_points = g_vk.has.eds2_patch_points && vkCmdSetPatchControlPointsEXT;
    VKLOG("replay pipelines: dynamic vertex input %d, blend enable / equation / write mask %d/%d/%d, logic op %d/%d, alpha to coverage %d, polygon mode %d, depth clamp %d, unrestricted topology %d",
          g_vkp_dyn.vertex_input, g_vkp_dyn.blend_enable, g_vkp_dyn.blend_eq, g_vkp_dyn.write_mask, g_vkp_dyn.logic_op_enable, g_vkp_dyn.logic_op, g_vkp_dyn.a2c,
          g_vkp_dyn.polygon_mode, g_vkp_dyn.depth_clamp, g_vkp_dyn.topo_unrestricted);
    return 0;
}
void vkpl_shutdown(void) {
    prewarm_stop();
    t_last.ok = 0;
    for (uint32_t i = 0; i < g_gfx.cap; i++) if (g_gfx.vals[i] && g_gfx.vals[i] != PIPE_FAILED) vkDestroyPipeline(g_vk.dev, (VkPipeline)g_gfx.vals[i], NULL);
    for (uint32_t i = 0; i < g_cs.cap; i++) if (g_cs.vals[i] && g_cs.vals[i] != PIPE_FAILED) vkDestroyPipeline(g_vk.dev, (VkPipeline)g_cs.vals[i], NULL);
    for (uint32_t i = 0; i < g_layouts.cap; i++) if (g_layouts.vals[i]) vkDestroyPipelineLayout(g_vk.dev, (VkPipelineLayout)g_layouts.vals[i], NULL);
    for (uint32_t i = 0; i < g_cs_layouts.cap; i++) if (g_cs_layouts.vals[i]) vkDestroyPipelineLayout(g_vk.dev, (VkPipelineLayout)g_cs_layouts.vals[i], NULL);
    vk_hm_free(&g_gfx); vk_hm_free(&g_cs); vk_hm_free(&g_layouts); vk_hm_free(&g_cs_layouts);
    memset(&g_st, 0, sizeof g_st);
    manifest_close();
}
