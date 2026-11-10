// Shaders for the replay (docs/m5-replay.md): a program stage's SPIR-V and reflection from nxshader's cache (gfx/nxs_rt.h), turned into a VkShaderModule
// whose descriptors live in the set of its stage (every translated module uses set 0; a pipeline has one set per stage), a descriptor set layout, and
// the binding table the replay fills (what NVN state each binding takes: nxnative's encode_args kinds, as nxs_refl.py names them).
#pragma once
#include "vk_internal.h"
#include "nvn_obj.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { VKS_VS = 0, VKS_FS = 1, VKS_GS = 2, VKS_TCS = 3, VKS_TES = 4, VKS_CS = 5 };   // NVNshaderStage
enum { VKB_SUPPORT = 1, VKB_CBUF, VKB_C1, VKB_DRV, VKB_SBUF, VKB_TEX, VKB_TBUF, VKB_IMG, VKB_UBOP, VKB_SBUF_UBO, VKB_UNKNOWN };
enum { VKB_SAMP_NONE = 0, VKB_SAMP_TEX, VKB_SAMP_SLOT, VKB_SAMP_ID };   // where a combined image sampler's sampler comes from
enum { VKS_CT_FLOAT = 0, VKS_CT_UINT, VKS_CT_INT };

typedef struct {
    uint32_t binding, count; VkDescriptorType type;
    uint8_t kind, samp_kind, rw, bank;
    int16_t reg, samp_reg;     // NVN slot (uniform / storage / texture / image), sampler slot or ID
    uint16_t c0_off;           // storage buffers: the c0 byte of their address (nxs_refl.py c0_offset)
    // the image the shader declares (SPIR-V OpTypeImage): view type, sampled type, depth comparison, multisample
    VkImageViewType vtype; uint8_t ct, depth, ms, dim;
    uint8_t dref, plain;   // the shader samples it with a depth comparison (OpImage*Dref*), and without one
} vks_bind;

typedef struct vks_shader {
    struct { int reg; uint32_t word, value; } first[8]; int nfirst;   // stores the replay makes before a dispatch (nxs_spv.py FirstStore)
    char key[17]; int stage, set, mesh;   // mesh: 1 a task shader, 2 a mesh shader, in the tessellation stages' places
    VkShaderModule mod;
    VkDescriptorSetLayout dsl;
    int nb; vks_bind *b;
    int nin; struct { uint8_t loc, ct, ncomp; } in[32];   // vertex inputs: location = NVN vertex attribute
    uint32_t out_mask; uint8_t out_ct[8];                   // fragment outputs written
    uint32_t local[3];                                      // compute work group size
    uint8_t drv_read[3];                                    // driver segments c0 / c2 read at run time
    int s2cs; uint32_t s2_ssbo_base, s2_tex_base, s2_ntex;  // Switch 2 packed compute bank (s2_compute_layout)
    int c1_missing;                                         // a c1 binding but no c1 bytes in the reflection (read as zeros)
    uint32_t c1_len; VkBuffer c1buf; VkDeviceSize c1off;    // the shader's own constant segment, in the c1 bank
    uint64_t uses;
    uint32_t *words; size_t nwords;                         // the module as created (vertex shaders: the source of their input variants)
    struct vks_shader *base;                                // a variant's shader; NULL for one from the cache
    struct { uint64_t key; struct vks_shader *s; } var[8]; int nvar;
    uint8_t dual, alpha, fs_flags;                          // a fragment variant: output 1 is Location 0 Index 1; the alpha test reads its push constants; what made it
    uint32_t vin_mask; uint8_t vin_ct[32];                  // a vertex input variant: what made it (the pipeline manifest names variants by these)
    uint8_t ds_reuse;                                       // vk_replay.c set_reusable: 0 not looked at yet, 1 its set may serve the next draw, 2 never
} vks_shader;

int vksh_init(void);
void vksh_shutdown(void);
int vksh_set_of(int stage);                          // descriptor set of a stage's resources
VkDescriptorSetLayout vksh_empty_dsl(void);
// Stage i of program p (SetShaders' order): the shader, or NULL while it is missing from the cache (logged, dumped and requested by nxs_rt once)
const vks_shader *vksh_get(const prog_t *p, unsigned i);
const vks_shader *vksh_by_key(const char *key);   // M6 phase B: the same by shader key (the pipeline prewarm); NULL, quietly, when the cache has no such shader
// The same from a module and its reflection (tests; the cache's entries come through here too). NULL with *why set when they cannot be used
vks_shader *vksh_make(const char *key, const uint8_t *spv, size_t spv_size, const char *json, size_t json_size, const char **why);
void vksh_free(vks_shader *s);
VkShaderStageFlagBits vksh_vk_stage(int stage);
VkShaderStageFlagBits vksh_stage_bit(const vks_shader *s);   // the stage as Vulkan knows it, task and mesh included
void vksh_stats(uint64_t *loaded, uint64_t *failed);
// A vertex shader whose inputs at the locations of `mask` are declared with component type ct[location] (VKS_CT_*) instead, each load bit-cast back to the type
// the body uses: a vertex format of another numeric class then hands the shader Maxwell's register bits (an RGBA16I attribute sign-extended into a uint input,
// a normalized one as float bits). The base shader when mask is 0 or the module cannot be retyped (logged)
const vks_shader *vksh_vin_variant(const vks_shader *vs, uint32_t mask, const uint8_t ct[32]);
// M6 phase B: a fragment shader for dual-source blending (VKSH_FS_DUAL: output 1 as target 0's second source) and / or the alpha test (VKSH_FS_ALPHA: colour 0's alpha
// compared with the reference by NVN's function, both in fragment push constants {uint function; float reference;}, before each return; the fragment discarded when it
// fails). The base shader when no variant can be made (logged once)
enum { VKSH_FS_DUAL = 1, VKSH_FS_ALPHA = 2 };
const vks_shader *vksh_fs_variant(const vks_shader *fs, int flags);
const vks_shader *vksh_id_variant(const vks_shader *fs);
uint32_t *vkspv_dual_source(const uint32_t *words, size_t nwords, size_t *out_n);   // the edits themselves (tests): new words, or NULL
uint32_t *vkspv_alpha_test(const uint32_t *words, size_t nwords, size_t *out_n);
// The retyping itself (tests): new words, or NULL
uint32_t *vkspv_retype_inputs(const uint32_t *words, size_t nwords, uint32_t mask, const uint8_t ct[32], size_t *out_n);

// SPIR-V helpers (vk_shader.c): descriptor set decorations moved to set; per binding the declared image type. Exposed for the tests
int vkspv_set_descriptor_set(uint32_t *words, size_t nwords, uint32_t set);   // decorations changed, or -1 when the module is malformed
typedef struct { uint32_t binding; uint8_t found, dim, arrayed, ms, depth, ct, is_image, dref, plain; } vkspv_img;   // dref / plain: sampled with / without a depth comparison
int vkspv_image_of(const uint32_t *words, size_t nwords, uint32_t binding, vkspv_img *out);   // 0 when the binding exists
// gl_PerVertex's CullDistance member removed when the clip and cull arrays exceed limit and nothing reads or writes it: the new word count (*words replaced),
// nwords when nothing needed changing, 0 when it could not be done
size_t vkspv_trim_cull(uint32_t **words, size_t nwords, uint32_t limit);
// A warp-aggregated atomic counter's ballot(true) made the lane's own bit when the add's result is unused (#12: RTX 50 cards hang on that ballot after early
// returns): the new word count (*words replaced) and *rewritten ballots, nwords when nothing changed
size_t vkspv_ballot_per_lane(uint32_t **words, size_t nwords, int *rewritten);

#ifdef __cplusplus
}
#endif
