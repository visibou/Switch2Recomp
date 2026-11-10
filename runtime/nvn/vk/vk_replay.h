// The replay of NVN command streams on Vulkan (M5-C, docs/m5-replay.md): the records hle_nvn.c writes (nvn_obj.h) become render passes (dynamic rendering),
// pipelines, descriptor sets, draws, dispatches, copies, clears and queries in the submission's main command buffer (vk_submit.c). Behaviour follows
// nxnative's Metal replay (dm/gfx/nvn/nvn_mtl.m mtl_exec and what it calls) where Pokopia needs it; TBDR-specific Metal work (pass merging, load / store
// lookahead, deferred clears) has no counterpart here.
#pragma once
#include "vk_internal.h"
#include "nvn_obj.h"

#ifdef __cplusplus
extern "C" {
#endif

int vkr_init(void);
void vkr_shutdown(void);
void vkr_sub_begin(void);                    // gfx_sub_begin, after vks_begin
void vkr_exec(const uint8_t *handle);        // gfx_exec: one command list (JUMPs and CALLs followed)
void vkr_sub_end(void);                      // before the commit: the open pass ends, pending write-backs are recorded, the submission's arenas go with it
void vkr_queue_activate(uint64_t serial);    // the queue whose bindings the next submission continues (NVN state belongs to a queue)
void vkr_queue_forget(uint64_t serial);
void vkr_set_origin_lower_left(int on);
void vkr_set_queue_origin_lower_left(int on);
void vkr_set_depth_zero(int on);
// The in-order GPU operations the recording layer also asks for outside a record walk (gfx_sub_copy / gfx_sub_fill)
void vkr_copy(uint64_t src, uint64_t dst, uint64_t n);
void vkr_fill(uint64_t dst, uint64_t n, uint32_t value);
void vkr_update(uint64_t dst, uint64_t n, const uint8_t *data);
void vkr_fence(sync_t *s);                   // OP_FENCE_SYNC: the sync gets the submission's value at the commit
// The NVN format of a sampled or texel buffer view in the shader's numeric class (VKS_CT_*) with the same bits, or -1 (it matches, or no format keeps the bits)
int vkr_fmt_as_ct(int nvn_format, int ct);

typedef struct {
    uint64_t handles, records, draws, draws_skipped, dispatches, passes, clears, copies, tex_copies, updates, fills, barriers, queries, bad;
    uint64_t skip_no_shader, skip_no_pipeline, skip_no_target, skip_state, inexact_vertex;
    uint64_t desc_sets, writebacks;
    uint64_t plain_nocmp;   // plain samples bound with a sampler without comparison (vk_replay.c, plain_nocmp)
} vkr_stats_t;
void vkr_stats(vkr_stats_t *s);
void vkr_frame(uint64_t frame);              // per present: statistics every 300 frames
void vkr_gputime_next(void);                  // DM_CTL gputime: the next frame timed on the GPU, in the log
void vkr_passdump_next(long n);              // DM_CTL passdump: the next n frames as NXRT_VK_PASSDUMP logs them; any thread
void vkr_rawdump(const uint64_t *ser, int n, long frames);   // gamectl rawdump: these images at the end of each of the next frames
void vkr_texdump_next(void);                 // DM_CTL texdump: the next frame's images as NXRT_VK_TEXDUMP writes them; any thread

#ifdef __cplusplus
}
#endif
