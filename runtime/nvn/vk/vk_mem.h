// GPU memory (docs/m5-vulkan.md §3): every NVN memory pool gets a device-local VkBuffer of its size (GPU address = guest address - pool base + buffer
// offset), backed lazily in 2 MB granules through sparse binding, so only the parts the GPU uses take video memory. Its contents are kept current from
// guest memory through write-protect dirty tracking (nvn_dirty_win.c): a page is copied when it was written since its last copy, through a staging
// ring, in the submission's upload command buffer. No host memory is ever imported into Vulkan (research/01 lesson 10).
#pragma once
#include "vk_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

int vkm_init(void);
void vkm_shutdown(void);
void vkm_pool_add(const void *mem, uint64_t size);
void vkm_pool_del(const void *mem);
int vkm_addr(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off);   // lookup only: 1 when [addr, addr+len) lies in a pool
// The replay's access to pool memory in the submission being recorded: brings the range's written pages up to date (uploads in this submission's upload
// command buffer, before its main one) and returns the buffer and offset. 1 on success
int vkm_use(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off);
uint32_t vkm_pool_gen(void);
uint64_t vkm_up_ns(void);       // time spent copying pool pages so far (vkm_stats' up_ns without its walk over every pool)   // moves whenever a memory pool comes or goes (a buffer and offset found before may be stale)
// GPU writes into pool memory are copied back to guest memory once the submission completes (NVN pools are coherent with the CPU)
void vkm_gpu_wrote(uint64_t addr, uint64_t len);
uint64_t vkm_wb_base(uint64_t addr, uint64_t len);
void vkm_gpu_wrote_base(uint64_t addr, uint64_t len, uint64_t base);
// GPU writes whose bytes the CPU knows (inline uniform updates, fills): guest memory gets the same bytes at completion; data is copied
void vkm_cpu_also(uint64_t addr, uint64_t len, const void *data, int fill, uint32_t value);
// Every pool's written pages, within the frame's upload budget (the policy until the replay calls vkm_use; then a background prefetch). Returns bytes queued
uint64_t vkm_sync_all(void);
void vkm_frame_tick(uint64_t frame);       // at each present: the budget restarts, hot pages age, statistics every 300 frames

// The staging ring, shared with texture uploads: room for n bytes that live until the submission being recorded completes (or, outside a submission,
// until the next one-shot). Returns the mapped pointer, the buffer and offset; NULL only when memory is exhausted
void *vkm_staging(VkDeviceSize n, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off);
uint64_t vkm_budget_left(void);            // bytes of this frame's upload budget not used yet (opportunistic uploads stop at 0)
void vkm_budget_take(uint64_t n);

typedef struct {
    uint64_t pools, pool_bytes, bound_bytes, sparse_pools;
    uint64_t up_bytes, up_pages, up_runs, up_ns, sync_calls;   // cumulative
    uint64_t staging_bytes, staging_overflow, staging_waits;
    uint64_t wb_merged;   // words the CPU changed in such pages, sent on their own
    uint64_t wb_words, wb_kept, wb_stale;   // words a shader changed; words the CPU changed while the shader ran, kept; uses of a page the CPU changed while a write-back was on its way
    uint64_t hot_pages, writeback_bytes, uncommitted_pages;   // uncommitted: pages stepped over (never touched by the game)
    uint64_t faults, fault_ns, preps;
} vkm_stats_t;
void vkm_stats(vkm_stats_t *s);

#ifdef __cplusplus
}
#endif
