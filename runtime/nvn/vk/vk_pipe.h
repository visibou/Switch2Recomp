// Pipelines for the replay (docs/m5-replay.md): NVN state → Vulkan (the translation functions are pure and tested), pipeline layouts (one descriptor
// set per stage), graphics and compute pipelines cached by what the device cannot set dynamically, made through the incremental pipeline cache
// (vk_pcache.c). What is dynamic on this device (extended dynamic state 1-3, dynamic vertex input) is left out of the key and set per draw.
#pragma once
#include "vk_internal.h"
#include "vk_shader.h"
#include "nvn_obj.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------- Translation (nxnative nvn_mtl.m's rules where it has them: bfac, bop, adv_blend, blend_zero_src1, vfmt, vfmt_for_ct, the winding rule) ----------
VkCompareOp vkp_compare(int nvn_func);                 // NVN depth / stencil / alpha function (NEVER = 1 … ALWAYS = 8)
VkStencilOp vkp_stencil_op(int nvn_op);
VkBlendFactor vkp_blend_factor(int nvn_func);
VkBlendOp vkp_blend_op(int nvn_eq);
blend_st vkp_adv_blend(blend_st b);                    // a Porter-Duff advanced mode as fixed-function factors (unchanged when it has none)
int vkp_blend_is_src1(int nvn_func);
int vkp_blend_zero_src1(int nvn_func);                 // SRC1 factors with an unwritten second source: ZERO / ONE
enum { VKP_EXPAND_NONE = 0, VKP_EXPAND_QUADS, VKP_EXPAND_LINE_LOOP, VKP_EXPAND_UNSUPPORTED };
VkPrimitiveTopology vkp_topology(int nvn_prim, int *expand);
int vkp_topology_class(VkPrimitiveTopology t);          // 0 point, 1 line, 2 triangle, 3 patch
VkCullModeFlags vkp_cull(int nvn_face);
VkFrontFace vkp_front_face(int nvn_front, int queue_origin_ll);
VkPolygonMode vkp_polygon_mode(int nvn_mode);
VkLogicOp vkp_logic_op(int nvn_op);
// An NVN vertex attribute format read by a shader input of component type ct: the format Vulkan fetches with. *exact = 0 when the bits differ from
// Maxwell's (an integer input fed a normalized / half format, etc.); VK_FORMAT_UNDEFINED when there is no counterpart
VkFormat vkp_vertex_format(int nvn_fmt, int ct, int *exact);
uint32_t vkp_vertex_format_size(int nvn_fmt);           // bytes of one element (0 = not a vertex format)
VkFormat vkp_vertex_format_natural(int nvn_fmt, int *ct);   // the format with its own numeric class (*ct: VKS_CT_*): what a retyped input reads

// ---------- Pipelines ----------
typedef struct {   // per colour attachment, as the pipeline sees it
    uint8_t blend, src, dst, srca, dsta, op, opa, wmask;
} vkp_att;
typedef struct {
    const vks_shader *sh[5];                 // NVN stages vertex, fragment, geometry, tess control, tess evaluation
    uint32_t cfmt[8], dfmt;                  // attachment formats (VkFormat); dfmt covers depth and / or stencil
    uint8_t ncolor, samples, has_depth, has_stencil;
    uint8_t topo_class;                      // VKP_TOPO_ANY when the topology is set freely per draw
    uint8_t depth_m1, patch_points;          // NVN depth mode -1..1 (depth clip control); tessellation patch size
    // baked only where the device has no dynamic state for them (zero otherwise, so they do not split the cache)
    vkp_att att[8]; uint8_t logic_en, logic_op, a2c, poly, depth_clamp, pad[3];
    uint8_t nattr, nbind, pad2[2];
    struct { uint8_t loc, bind; uint16_t pad; uint32_t fmt, off; } attr[16];
    struct { uint32_t stride, divisor; uint8_t rate, pad[3]; } bind[17];
} vkp_gkey;
enum { VKP_TOPO_ANY = 0xff };

typedef struct {   // what this device sets dynamically (vkp_init)
    int vertex_input, blend_enable, blend_eq, write_mask, logic_op_enable, logic_op, a2c, polygon_mode, depth_clamp, topo_unrestricted, patch_points;
} vkp_dyn;
extern vkp_dyn g_vkp_dyn;

int vkpl_init(void);
void vkpl_shutdown(void);
VkPipelineLayout vkp_layout(const vks_shader *const sh[5]);          // graphics: one set per stage (empty sets where a stage is absent); the alpha test's push constants
enum { VKP_ALPHA_PC_SIZE = 8 };                                       // {uint function; float reference;} at fragment push constant offset 0 (vksh_fs_variant)
VkPipelineLayout vkp_layout_cs(const vks_shader *cs);
VkPipeline vkp_graphics(const vkp_gkey *k, VkPipelineLayout *layout); // VK_NULL_HANDLE when it cannot be made (logged once per key)
VkPipeline vkp_compute(const vks_shader *cs, VkPipelineLayout *layout);
typedef struct { uint64_t graphics, compute, failed, layouts; uint64_t create_ns, max_ns; uint64_t prewarmed, prewarm_ns; } vkp_stats_t;   // prewarmed: made ahead (vkpl_prewarm_start)
void vkp_stats(vkp_stats_t *s);
// M6 phase B: background threads make the pipelines of pipelines.tsv (the ones made here before, listed beside the pipeline cache) and of NXRT_VK_PIPELINE_SEED ahead of
// their first use (vk_pipe.c); the replay's lookups find them there
void vkpl_prewarm_start(void);

#ifdef __cplusplus
}
#endif
