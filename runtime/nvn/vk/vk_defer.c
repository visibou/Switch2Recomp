// Deferred recording (NXRT_VK_DEFER=1): the replay's Vulkan commands are recorded into the real command buffers by a thread of their own.
//
// The translation of an NVN submission spends about a quarter of its time inside the driver (vkCmd* calls and descriptor set updates), on the thread that
// also resolves every binding. With this on, that thread writes each command, with copies of everything its pointers point at, into a ring (a few dozen
// bytes a call), and the recorder thread makes the actual calls, in the same order, into the same command buffers. Translation and recording overlap.
//
// What is deferred: every vkCmd* aimed at a command buffer of the NVN submission being recorded (its main and upload buffers, registered by vk_submit.c
// while it records them), and vkUpdateDescriptorSets called by the thread recording that submission. Everything else (one-shot command buffers of other
// threads, the present thread, object creation) goes straight to the driver as before. The submission's vkEndCommandBuffer is deferred too: vk_submit.c
// hands a committed slot to the queue only once the recorder has ended it (vkd_end, vks_slot_recorded), so timeline values still reach the queue in order.
// Objects a deferred command refers to live until the GPU has completed it (views, buffers and sets are retired after completion), so the copies stay valid.
//
// The ring is single producer (the recording thread), single consumer (the recorder). A call that does not fit waits for the recorder to catch up.
#include "vk_internal.h"
#include "vk_fn.h"
#include <process.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

int vkc_int(const char *name, int def);
void vks_slot_recorded(int slot);   // vk_submit.c: the recorder has ended slot's command buffers

enum {
    D_WRAP = 1, D_END, D_UPDATE_SETS,
    D_BARRIER2, D_COPY_BUF, D_COPY_B2I, D_COPY_I2B, D_COPY_IMG, D_BLIT, D_FILL, D_UPDATE_BUF, D_CLEAR_COLOR, D_CLEAR_DS, D_BEGIN_RENDERING, D_END_RENDERING,
    D_BIND_PIPE, D_SET_VP, D_SET_SC, D_DRAW, D_PUSH, D_BIND_SETS, D_BIND_VB2, D_BIND_IB, D_DRAW_IDX, D_DRAW_IND, D_DRAW_IDX_IND, D_DRAW_IND_CNT, D_DRAW_IDX_IND_CNT,
    D_DISPATCH, D_DISPATCH_IND, D_CLEAR_ATT, D_RESOLVE, D_BEGIN_Q, D_END_Q, D_RESET_QP, D_VP_CNT, D_SC_CNT, D_LINE_W, D_DEPTH_BIAS, D_BLEND_CONST,
    D_ST_CMASK, D_ST_WMASK, D_ST_REF, D_CULL, D_FRONT, D_TOPO, D_DT, D_DW, D_DCMP, D_DBT, D_ST_TEST, D_ST_OP, D_RD, D_BIAS_EN, D_RESTART, D_TS2, D_MESH,
    D_VERTEX_INPUT, D_BLEND_EN, D_BLEND_EQ, D_WMASK, D_LOGIC_EN, D_LOGIC, D_A2C, D_POLY, D_DCLAMP, D_PATCH, D_CHECKPOINT,
};

// ---------- The real entry points (the recorder calls these; the globals point at the thunks below while deferral is on) ----------
static struct {
#define R_DEF(n) PFN_##n n;
    R_DEF(vkCmdPipelineBarrier2) R_DEF(vkCmdCopyBuffer) R_DEF(vkCmdCopyBufferToImage) R_DEF(vkCmdCopyImageToBuffer) R_DEF(vkCmdCopyImage) R_DEF(vkCmdBlitImage)
    R_DEF(vkCmdFillBuffer) R_DEF(vkCmdUpdateBuffer) R_DEF(vkCmdClearColorImage) R_DEF(vkCmdClearDepthStencilImage) R_DEF(vkCmdBeginRendering) R_DEF(vkCmdEndRendering)
    R_DEF(vkCmdBindPipeline) R_DEF(vkCmdSetViewport) R_DEF(vkCmdSetScissor) R_DEF(vkCmdDraw) R_DEF(vkCmdPushConstants) R_DEF(vkCmdBindDescriptorSets)
    R_DEF(vkCmdBindVertexBuffers2) R_DEF(vkCmdBindIndexBuffer) R_DEF(vkCmdDrawIndexed) R_DEF(vkCmdDrawIndirect) R_DEF(vkCmdDrawIndexedIndirect)
    R_DEF(vkCmdDrawIndirectCount) R_DEF(vkCmdDrawIndexedIndirectCount) R_DEF(vkCmdDispatch) R_DEF(vkCmdDispatchIndirect) R_DEF(vkCmdClearAttachments)
    R_DEF(vkCmdResolveImage) R_DEF(vkCmdBeginQuery) R_DEF(vkCmdEndQuery) R_DEF(vkCmdResetQueryPool) R_DEF(vkCmdSetViewportWithCount) R_DEF(vkCmdSetScissorWithCount)
    R_DEF(vkCmdSetLineWidth) R_DEF(vkCmdSetDepthBias) R_DEF(vkCmdSetBlendConstants) R_DEF(vkCmdSetStencilCompareMask) R_DEF(vkCmdSetStencilWriteMask)
    R_DEF(vkCmdSetStencilReference) R_DEF(vkCmdSetCullMode) R_DEF(vkCmdSetFrontFace) R_DEF(vkCmdSetPrimitiveTopology) R_DEF(vkCmdSetDepthTestEnable)
    R_DEF(vkCmdSetDepthWriteEnable) R_DEF(vkCmdSetDepthCompareOp) R_DEF(vkCmdSetDepthBoundsTestEnable) R_DEF(vkCmdSetStencilTestEnable) R_DEF(vkCmdSetStencilOp)
    R_DEF(vkCmdSetRasterizerDiscardEnable) R_DEF(vkCmdSetDepthBiasEnable) R_DEF(vkCmdSetPrimitiveRestartEnable) R_DEF(vkCmdWriteTimestamp2) R_DEF(vkCmdDrawMeshTasksEXT)
    R_DEF(vkCmdSetVertexInputEXT) R_DEF(vkCmdSetColorBlendEnableEXT) R_DEF(vkCmdSetColorBlendEquationEXT) R_DEF(vkCmdSetColorWriteMaskEXT) R_DEF(vkCmdSetLogicOpEnableEXT)
    R_DEF(vkCmdSetLogicOpEXT) R_DEF(vkCmdSetAlphaToCoverageEnableEXT) R_DEF(vkCmdSetPolygonModeEXT) R_DEF(vkCmdSetDepthClampEnableEXT) R_DEF(vkCmdSetPatchControlPointsEXT)
    R_DEF(vkCmdSetCheckpointNV) R_DEF(vkUpdateDescriptorSets) R_DEF(vkEndCommandBuffer)
#undef R_DEF
} R;

// ---------- Which command buffers are deferred ----------
static VkCommandBuffer g_dcb[4]; static _Atomic int g_ndcb;   // the recording slot's buffers (written by the recording thread only)
static DWORD g_dtid;                                          // the thread that records them (its vkUpdateDescriptorSets are deferred)
static int g_on;
static _Atomic uint64_t g_w, g_r;   // bytes written (published) and read, ever (the ring, below)
static void drain(void) { while (atomic_load(&g_r) != atomic_load(&g_w)) SwitchToThread(); }
// 1: cb belongs to the submission being recorded (into the ring). A call for any other buffer goes to the driver; if it comes from the recording thread while the ring
// still holds work, that work is recorded first (a one-shot this thread records, e.g. a texture's first upload, may use descriptor sets whose updates are still queued)
static inline int deferred(VkCommandBuffer cb) {
    int n = g_ndcb; for (int i = 0; i < n; i++) if (g_dcb[i] == cb) return 1;
    if (g_on && GetCurrentThreadId() == g_dtid && atomic_load_explicit(&g_r, memory_order_relaxed) != atomic_load_explicit(&g_w, memory_order_relaxed)) drain();
    return 0;
}

// ---------- The ring ----------
enum { RB = 64u << 20 };
static uint8_t *g_rb;
static uint64_t g_wl;               // the producer's position (published to g_w at each call)
static SRWLOCK g_lk = SRWLOCK_INIT; static CONDITION_VARIABLE g_cv_data = CONDITION_VARIABLE_INIT, g_cv_room = CONDITION_VARIABLE_INIT;
static _Atomic int g_rec_sleeps, g_prod_waits;
static _Atomic uint64_t s_calls, s_bytes, s_waits;
typedef struct { uint32_t id, size; } dhdr;   // size: of the whole record, header included, a multiple of 8

static uint8_t *d_alloc(uint32_t id, size_t payload) {
    uint32_t size = (uint32_t)((sizeof(dhdr) + payload + 7) & ~7ull);
    for (;;) {
        uint64_t at = g_wl % RB, need = size;
        if (at + size > RB) need += RB - at;   // and a wrap record before it
        if (g_wl + need - atomic_load_explicit(&g_r, memory_order_acquire) <= RB) {
            if (at + size > RB) { ((dhdr *)(g_rb + at))->id = D_WRAP; ((dhdr *)(g_rb + at))->size = (uint32_t)(RB - at); g_wl += RB - at; at = 0; }
            dhdr *h = (dhdr *)(g_rb + at); h->id = id; h->size = size;
            return (uint8_t *)(h + 1);
        }
        atomic_fetch_add(&s_waits, 1);   // full: wait for the recorder
        AcquireSRWLockExclusive(&g_lk); atomic_store(&g_prod_waits, 1);
        if (g_wl + need - atomic_load(&g_r) > RB) SleepConditionVariableSRW(&g_cv_room, &g_lk, 1, 0);
        atomic_store(&g_prod_waits, 0); ReleaseSRWLockExclusive(&g_lk);
    }
}
static void d_publish(void) {
    uint64_t at = g_wl % RB; g_wl += ((dhdr *)(g_rb + at))->size;
    atomic_store_explicit(&g_w, g_wl, memory_order_release);
    atomic_fetch_add_explicit(&s_calls, 1, memory_order_relaxed);
    if (atomic_load_explicit(&g_rec_sleeps, memory_order_acquire)) { AcquireSRWLockExclusive(&g_lk); WakeConditionVariable(&g_cv_data); ReleaseSRWLockExclusive(&g_lk); }
}
// payload writers
#define PUT(p, v) do { memcpy((p), &(v), sizeof (v)); (p) += (sizeof (v) + 7) & ~(size_t)7; } while (0)
#define PUTN(p, a, n) do { size_t nb_ = (size_t)(n); if (nb_) memcpy((p), (a), nb_); (p) += (nb_ + 7) & ~(size_t)7; } while (0)
#define GET(p, T, v) T v; memcpy(&v, (p), sizeof v); (p) += (sizeof v + 7) & ~(size_t)7
#define GETN(p, T, v, n) const T *v = (const T *)(p); (p) += ((size_t)(n) * sizeof(T) + 7) & ~(size_t)7
#define SZ(T) ((sizeof(T) + 7) & ~(size_t)7)
#define SZN(T, n) (((size_t)(n) * sizeof(T) + 7) & ~(size_t)7)

// ---------- Thunks: deferred buffers into the ring, everything else to the driver ----------
// Scalar commands: up to 6 arguments, each stored in an 8-byte word
#define TH0(name, ID) static VKAPI_ATTR void VKAPI_CALL t_##name(VkCommandBuffer cb) { if (!deferred(cb)) { R.name(cb); return; } \
    uint8_t *p = d_alloc(ID, 8); PUT(p, cb); d_publish(); }
#define TH1(name, ID, T1) static VKAPI_ATTR void VKAPI_CALL t_##name(VkCommandBuffer cb, T1 a) { if (!deferred(cb)) { R.name(cb, a); return; } \
    uint8_t *p = d_alloc(ID, 16); PUT(p, cb); PUT(p, a); d_publish(); }
#define TH2(name, ID, T1, T2) static VKAPI_ATTR void VKAPI_CALL t_##name(VkCommandBuffer cb, T1 a, T2 b) { if (!deferred(cb)) { R.name(cb, a, b); return; } \
    uint8_t *p = d_alloc(ID, 24); PUT(p, cb); PUT(p, a); PUT(p, b); d_publish(); }
#define TH3(name, ID, T1, T2, T3) static VKAPI_ATTR void VKAPI_CALL t_##name(VkCommandBuffer cb, T1 a, T2 b, T3 c) { if (!deferred(cb)) { R.name(cb, a, b, c); return; } \
    uint8_t *p = d_alloc(ID, 32); PUT(p, cb); PUT(p, a); PUT(p, b); PUT(p, c); d_publish(); }
#define TH4(name, ID, T1, T2, T3, T4) static VKAPI_ATTR void VKAPI_CALL t_##name(VkCommandBuffer cb, T1 a, T2 b, T3 c, T4 d) { if (!deferred(cb)) { R.name(cb, a, b, c, d); return; } \
    uint8_t *p = d_alloc(ID, 40); PUT(p, cb); PUT(p, a); PUT(p, b); PUT(p, c); PUT(p, d); d_publish(); }
#define TH5(name, ID, T1, T2, T3, T4, T5) static VKAPI_ATTR void VKAPI_CALL t_##name(VkCommandBuffer cb, T1 a, T2 b, T3 c, T4 d, T5 e) { if (!deferred(cb)) { R.name(cb, a, b, c, d, e); return; } \
    uint8_t *p = d_alloc(ID, 48); PUT(p, cb); PUT(p, a); PUT(p, b); PUT(p, c); PUT(p, d); PUT(p, e); d_publish(); }
#define TH6(name, ID, T1, T2, T3, T4, T5, T6) static VKAPI_ATTR void VKAPI_CALL t_##name(VkCommandBuffer cb, T1 a, T2 b, T3 c, T4 d, T5 e, T6 f) { if (!deferred(cb)) { R.name(cb, a, b, c, d, e, f); return; } \
    uint8_t *p = d_alloc(ID, 56); PUT(p, cb); PUT(p, a); PUT(p, b); PUT(p, c); PUT(p, d); PUT(p, e); PUT(p, f); d_publish(); }
TH0(vkCmdEndRendering, D_END_RENDERING)
TH2(vkCmdBindPipeline, D_BIND_PIPE, VkPipelineBindPoint, VkPipeline)
TH4(vkCmdDraw, D_DRAW, uint32_t, uint32_t, uint32_t, uint32_t)
TH3(vkCmdBindIndexBuffer, D_BIND_IB, VkBuffer, VkDeviceSize, VkIndexType)
TH5(vkCmdDrawIndexed, D_DRAW_IDX, uint32_t, uint32_t, uint32_t, int32_t, uint32_t)
TH4(vkCmdDrawIndirect, D_DRAW_IND, VkBuffer, VkDeviceSize, uint32_t, uint32_t)
TH4(vkCmdDrawIndexedIndirect, D_DRAW_IDX_IND, VkBuffer, VkDeviceSize, uint32_t, uint32_t)
TH6(vkCmdDrawIndirectCount, D_DRAW_IND_CNT, VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, uint32_t, uint32_t)
TH6(vkCmdDrawIndexedIndirectCount, D_DRAW_IDX_IND_CNT, VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, uint32_t, uint32_t)
TH3(vkCmdDispatch, D_DISPATCH, uint32_t, uint32_t, uint32_t)
TH2(vkCmdDispatchIndirect, D_DISPATCH_IND, VkBuffer, VkDeviceSize)
TH4(vkCmdFillBuffer, D_FILL, VkBuffer, VkDeviceSize, VkDeviceSize, uint32_t)
TH3(vkCmdBeginQuery, D_BEGIN_Q, VkQueryPool, uint32_t, VkQueryControlFlags)
TH2(vkCmdEndQuery, D_END_Q, VkQueryPool, uint32_t)
TH3(vkCmdResetQueryPool, D_RESET_QP, VkQueryPool, uint32_t, uint32_t)
TH1(vkCmdSetLineWidth, D_LINE_W, float)
TH3(vkCmdSetDepthBias, D_DEPTH_BIAS, float, float, float)
TH2(vkCmdSetStencilCompareMask, D_ST_CMASK, VkStencilFaceFlags, uint32_t)
TH2(vkCmdSetStencilWriteMask, D_ST_WMASK, VkStencilFaceFlags, uint32_t)
TH2(vkCmdSetStencilReference, D_ST_REF, VkStencilFaceFlags, uint32_t)
TH1(vkCmdSetCullMode, D_CULL, VkCullModeFlags)
TH1(vkCmdSetFrontFace, D_FRONT, VkFrontFace)
TH1(vkCmdSetPrimitiveTopology, D_TOPO, VkPrimitiveTopology)
TH1(vkCmdSetDepthTestEnable, D_DT, VkBool32)
TH1(vkCmdSetDepthWriteEnable, D_DW, VkBool32)
TH1(vkCmdSetDepthCompareOp, D_DCMP, VkCompareOp)
TH1(vkCmdSetDepthBoundsTestEnable, D_DBT, VkBool32)
TH1(vkCmdSetStencilTestEnable, D_ST_TEST, VkBool32)
TH5(vkCmdSetStencilOp, D_ST_OP, VkStencilFaceFlags, VkStencilOp, VkStencilOp, VkStencilOp, VkCompareOp)
TH1(vkCmdSetRasterizerDiscardEnable, D_RD, VkBool32)
TH1(vkCmdSetDepthBiasEnable, D_BIAS_EN, VkBool32)
TH1(vkCmdSetPrimitiveRestartEnable, D_RESTART, VkBool32)
TH3(vkCmdWriteTimestamp2, D_TS2, VkPipelineStageFlags2, VkQueryPool, uint32_t)
TH3(vkCmdDrawMeshTasksEXT, D_MESH, uint32_t, uint32_t, uint32_t)
TH1(vkCmdSetLogicOpEnableEXT, D_LOGIC_EN, VkBool32)
TH1(vkCmdSetLogicOpEXT, D_LOGIC, VkLogicOp)
TH1(vkCmdSetAlphaToCoverageEnableEXT, D_A2C, VkBool32)
TH1(vkCmdSetPolygonModeEXT, D_POLY, VkPolygonMode)
TH1(vkCmdSetDepthClampEnableEXT, D_DCLAMP, VkBool32)
TH1(vkCmdSetPatchControlPointsEXT, D_PATCH, uint32_t)
TH1(vkCmdSetCheckpointNV, D_CHECKPOINT, const void *)   // the marker is an opaque value

// Commands with arrays: the arrays are copied after the scalars
static VKAPI_ATTR void VKAPI_CALL t_vkCmdPipelineBarrier2(VkCommandBuffer cb, const VkDependencyInfo *di) {
    if (!deferred(cb)) { R.vkCmdPipelineBarrier2(cb, di); return; }
    uint8_t *p = d_alloc(D_BARRIER2, SZ(cb) + SZ(*di) + SZN(VkMemoryBarrier2, di->memoryBarrierCount) + SZN(VkBufferMemoryBarrier2, di->bufferMemoryBarrierCount) + SZN(VkImageMemoryBarrier2, di->imageMemoryBarrierCount));
    PUT(p, cb); PUT(p, *di); PUTN(p, di->pMemoryBarriers, sizeof(VkMemoryBarrier2) * di->memoryBarrierCount);
    PUTN(p, di->pBufferMemoryBarriers, sizeof(VkBufferMemoryBarrier2) * di->bufferMemoryBarrierCount); PUTN(p, di->pImageMemoryBarriers, sizeof(VkImageMemoryBarrier2) * di->imageMemoryBarrierCount);
    d_publish();
}
#define TH_REGIONS(name, ID, T, ...)   /* (cb, <scalars>, n, regions) */
static VKAPI_ATTR void VKAPI_CALL t_vkCmdCopyBuffer(VkCommandBuffer cb, VkBuffer s, VkBuffer d, uint32_t n, const VkBufferCopy *r) {
    if (!deferred(cb)) { R.vkCmdCopyBuffer(cb, s, d, n, r); return; }
    uint8_t *p = d_alloc(D_COPY_BUF, 32 + SZN(VkBufferCopy, n)); PUT(p, cb); PUT(p, s); PUT(p, d); PUT(p, n); PUTN(p, r, sizeof *r * n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdCopyBufferToImage(VkCommandBuffer cb, VkBuffer s, VkImage d, VkImageLayout l, uint32_t n, const VkBufferImageCopy *r) {
    if (!deferred(cb)) { R.vkCmdCopyBufferToImage(cb, s, d, l, n, r); return; }
    uint8_t *p = d_alloc(D_COPY_B2I, 40 + SZN(VkBufferImageCopy, n)); PUT(p, cb); PUT(p, s); PUT(p, d); PUT(p, l); PUT(p, n); PUTN(p, r, sizeof *r * n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdCopyImageToBuffer(VkCommandBuffer cb, VkImage s, VkImageLayout l, VkBuffer d, uint32_t n, const VkBufferImageCopy *r) {
    if (!deferred(cb)) { R.vkCmdCopyImageToBuffer(cb, s, l, d, n, r); return; }
    uint8_t *p = d_alloc(D_COPY_I2B, 40 + SZN(VkBufferImageCopy, n)); PUT(p, cb); PUT(p, s); PUT(p, l); PUT(p, d); PUT(p, n); PUTN(p, r, sizeof *r * n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdCopyImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageCopy *r) {
    if (!deferred(cb)) { R.vkCmdCopyImage(cb, s, sl, d, dl, n, r); return; }
    uint8_t *p = d_alloc(D_COPY_IMG, 48 + SZN(VkImageCopy, n)); PUT(p, cb); PUT(p, s); PUT(p, sl); PUT(p, d); PUT(p, dl); PUT(p, n); PUTN(p, r, sizeof *r * n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdBlitImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageBlit *r, VkFilter f) {
    if (!deferred(cb)) { R.vkCmdBlitImage(cb, s, sl, d, dl, n, r, f); return; }
    uint8_t *p = d_alloc(D_BLIT, 56 + SZN(VkImageBlit, n)); PUT(p, cb); PUT(p, s); PUT(p, sl); PUT(p, d); PUT(p, dl); PUT(p, n); PUT(p, f); PUTN(p, r, sizeof *r * n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdResolveImage(VkCommandBuffer cb, VkImage s, VkImageLayout sl, VkImage d, VkImageLayout dl, uint32_t n, const VkImageResolve *r) {
    if (!deferred(cb)) { R.vkCmdResolveImage(cb, s, sl, d, dl, n, r); return; }
    uint8_t *p = d_alloc(D_RESOLVE, 48 + SZN(VkImageResolve, n)); PUT(p, cb); PUT(p, s); PUT(p, sl); PUT(p, d); PUT(p, dl); PUT(p, n); PUTN(p, r, sizeof *r * n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdUpdateBuffer(VkCommandBuffer cb, VkBuffer b, VkDeviceSize o, VkDeviceSize n, const void *data) {
    if (!deferred(cb)) { R.vkCmdUpdateBuffer(cb, b, o, n, data); return; }
    uint8_t *p = d_alloc(D_UPDATE_BUF, 32 + SZN(uint8_t, n)); PUT(p, cb); PUT(p, b); PUT(p, o); PUT(p, n); PUTN(p, data, n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdClearColorImage(VkCommandBuffer cb, VkImage im, VkImageLayout l, const VkClearColorValue *c, uint32_t n, const VkImageSubresourceRange *r) {
    if (!deferred(cb)) { R.vkCmdClearColorImage(cb, im, l, c, n, r); return; }
    uint8_t *p = d_alloc(D_CLEAR_COLOR, 32 + SZ(*c) + SZN(VkImageSubresourceRange, n)); PUT(p, cb); PUT(p, im); PUT(p, l); PUT(p, n); PUT(p, *c); PUTN(p, r, sizeof *r * n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdClearDepthStencilImage(VkCommandBuffer cb, VkImage im, VkImageLayout l, const VkClearDepthStencilValue *c, uint32_t n, const VkImageSubresourceRange *r) {
    if (!deferred(cb)) { R.vkCmdClearDepthStencilImage(cb, im, l, c, n, r); return; }
    uint8_t *p = d_alloc(D_CLEAR_DS, 32 + SZ(*c) + SZN(VkImageSubresourceRange, n)); PUT(p, cb); PUT(p, im); PUT(p, l); PUT(p, n); PUT(p, *c); PUTN(p, r, sizeof *r * n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdClearAttachments(VkCommandBuffer cb, uint32_t n, const VkClearAttachment *a, uint32_t m, const VkClearRect *r) {
    if (!deferred(cb)) { R.vkCmdClearAttachments(cb, n, a, m, r); return; }
    uint8_t *p = d_alloc(D_CLEAR_ATT, 24 + SZN(VkClearAttachment, n) + SZN(VkClearRect, m)); PUT(p, cb); PUT(p, n); PUT(p, m); PUTN(p, a, sizeof *a * n); PUTN(p, r, sizeof *r * m); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdBeginRendering(VkCommandBuffer cb, const VkRenderingInfo *ri) {
    if (!deferred(cb)) { R.vkCmdBeginRendering(cb, ri); return; }
    if (ri->pNext) { static int said; if (!said++) VKRED("deferred recording: a VkRenderingInfo with a pNext chain is recorded without it"); }
    uint8_t *p = d_alloc(D_BEGIN_RENDERING, SZ(cb) + SZ(*ri) + SZN(VkRenderingAttachmentInfo, ri->colorAttachmentCount + 2));
    VkRenderingAttachmentInfo none; memset(&none, 0, sizeof none);
    PUT(p, cb); PUT(p, *ri); PUTN(p, ri->pColorAttachments, sizeof(VkRenderingAttachmentInfo) * ri->colorAttachmentCount);
    const VkRenderingAttachmentInfo *d = ri->pDepthAttachment ? ri->pDepthAttachment : &none, *s = ri->pStencilAttachment ? ri->pStencilAttachment : &none;
    memcpy(p, d, sizeof *d); memcpy(p + sizeof *d, s, sizeof *s);   // inside the size reserved for the color array + 2
    d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdSetViewport(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkViewport *v) {
    if (!deferred(cb)) { R.vkCmdSetViewport(cb, first, n, v); return; }
    uint8_t *p = d_alloc(D_SET_VP, 24 + SZN(VkViewport, n)); PUT(p, cb); PUT(p, first); PUT(p, n); PUTN(p, v, sizeof *v * n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdSetScissor(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkRect2D *v) {
    if (!deferred(cb)) { R.vkCmdSetScissor(cb, first, n, v); return; }
    uint8_t *p = d_alloc(D_SET_SC, 24 + SZN(VkRect2D, n)); PUT(p, cb); PUT(p, first); PUT(p, n); PUTN(p, v, sizeof *v * n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdSetViewportWithCount(VkCommandBuffer cb, uint32_t n, const VkViewport *v) {
    if (!deferred(cb)) { R.vkCmdSetViewportWithCount(cb, n, v); return; }
    uint8_t *p = d_alloc(D_VP_CNT, 16 + SZN(VkViewport, n)); PUT(p, cb); PUT(p, n); PUTN(p, v, sizeof *v * n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdSetScissorWithCount(VkCommandBuffer cb, uint32_t n, const VkRect2D *v) {
    if (!deferred(cb)) { R.vkCmdSetScissorWithCount(cb, n, v); return; }
    uint8_t *p = d_alloc(D_SC_CNT, 16 + SZN(VkRect2D, n)); PUT(p, cb); PUT(p, n); PUTN(p, v, sizeof *v * n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdPushConstants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st, uint32_t off, uint32_t n, const void *v) {
    if (!deferred(cb)) { R.vkCmdPushConstants(cb, l, st, off, n, v); return; }
    uint8_t *p = d_alloc(D_PUSH, 40 + SZN(uint8_t, n)); PUT(p, cb); PUT(p, l); PUT(p, st); PUT(p, off); PUT(p, n); PUTN(p, v, n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdBindDescriptorSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first, uint32_t n, const VkDescriptorSet *s, uint32_t nd, const uint32_t *d) {
    if (!deferred(cb)) { R.vkCmdBindDescriptorSets(cb, bp, l, first, n, s, nd, d); return; }
    uint8_t *p = d_alloc(D_BIND_SETS, 48 + SZN(VkDescriptorSet, n) + SZN(uint32_t, nd)); PUT(p, cb); PUT(p, bp); PUT(p, l); PUT(p, first); PUT(p, n); PUT(p, nd);
    PUTN(p, s, sizeof *s * n); PUTN(p, d, sizeof *d * nd); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdBindVertexBuffers2(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer *b, const VkDeviceSize *o, const VkDeviceSize *sz, const VkDeviceSize *st) {
    if (!deferred(cb)) { R.vkCmdBindVertexBuffers2(cb, first, n, b, o, sz, st); return; }
    uint32_t has = (sz ? 1u : 0u) | (st ? 2u : 0u);
    uint8_t *p = d_alloc(D_BIND_VB2, 32 + SZN(VkBuffer, n) + 3 * SZN(VkDeviceSize, n)); PUT(p, cb); PUT(p, first); PUT(p, n); PUT(p, has);
    PUTN(p, b, sizeof *b * n); PUTN(p, o, sizeof *o * n); if (sz) PUTN(p, sz, sizeof *sz * n); if (st) PUTN(p, st, sizeof *st * n); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdSetBlendConstants(VkCommandBuffer cb, const float c[4]) {
    if (!deferred(cb)) { R.vkCmdSetBlendConstants(cb, c); return; }
    uint8_t *p = d_alloc(D_BLEND_CONST, 8 + 16); PUT(p, cb); PUTN(p, c, 16); d_publish();
}
static VKAPI_ATTR void VKAPI_CALL t_vkCmdSetVertexInputEXT(VkCommandBuffer cb, uint32_t nb, const VkVertexInputBindingDescription2EXT *b, uint32_t na, const VkVertexInputAttributeDescription2EXT *a) {
    if (!deferred(cb)) { R.vkCmdSetVertexInputEXT(cb, nb, b, na, a); return; }
    uint8_t *p = d_alloc(D_VERTEX_INPUT, 24 + SZN(VkVertexInputBindingDescription2EXT, nb) + SZN(VkVertexInputAttributeDescription2EXT, na)); PUT(p, cb); PUT(p, nb); PUT(p, na);
    PUTN(p, b, sizeof *b * nb); PUTN(p, a, sizeof *a * na); d_publish();
}
#define TH_ARR(name, ID, T) static VKAPI_ATTR void VKAPI_CALL t_##name(VkCommandBuffer cb, uint32_t first, uint32_t n, const T *v) { \
    if (!deferred(cb)) { R.name(cb, first, n, v); return; } uint8_t *p = d_alloc(ID, 24 + SZN(T, n)); PUT(p, cb); PUT(p, first); PUT(p, n); PUTN(p, v, sizeof *v * n); d_publish(); }
TH_ARR(vkCmdSetColorBlendEnableEXT, D_BLEND_EN, VkBool32)
TH_ARR(vkCmdSetColorBlendEquationEXT, D_BLEND_EQ, VkColorBlendEquationEXT)
TH_ARR(vkCmdSetColorWriteMaskEXT, D_WMASK, VkColorComponentFlags)

// Descriptor set updates of the recording thread: copied with every image, buffer and texel view they point at (descriptorCount of each write)
static VKAPI_ATTR void VKAPI_CALL t_vkUpdateDescriptorSets(VkDevice dev, uint32_t nw, const VkWriteDescriptorSet *w, uint32_t nc, const VkCopyDescriptorSet *c) {
    if (!g_ndcb || GetCurrentThreadId() != g_dtid || nc) {
        if (g_on && GetCurrentThreadId() == g_dtid) drain();
        R.vkUpdateDescriptorSets(dev, nw, w, nc, c); return; }
    size_t sz = 8 + SZN(VkWriteDescriptorSet, nw);
    for (uint32_t i = 0; i < nw; i++) sz += w[i].pImageInfo ? SZN(VkDescriptorImageInfo, w[i].descriptorCount) : w[i].pBufferInfo ? SZN(VkDescriptorBufferInfo, w[i].descriptorCount) : w[i].pTexelBufferView ? SZN(VkBufferView, w[i].descriptorCount) : 0;
    uint8_t *p = d_alloc(D_UPDATE_SETS, sz); PUT(p, nw);
    VkWriteDescriptorSet *ws = (VkWriteDescriptorSet *)p; PUTN(p, w, sizeof *w * nw);
    for (uint32_t i = 0; i < nw; i++) {
        if (w[i].pNext) { static int said; if (!said++) VKRED("deferred recording: a descriptor write with a pNext chain is made without it"); ws[i].pNext = NULL; }
        // the pointers inside the copy are rebuilt by the recorder (it walks the same order); here only the arrays go after the writes
        if (w[i].pImageInfo) PUTN(p, w[i].pImageInfo, sizeof(VkDescriptorImageInfo) * w[i].descriptorCount);
        else if (w[i].pBufferInfo) PUTN(p, w[i].pBufferInfo, sizeof(VkDescriptorBufferInfo) * w[i].descriptorCount);
        else if (w[i].pTexelBufferView) PUTN(p, w[i].pTexelBufferView, sizeof(VkBufferView) * w[i].descriptorCount);
    }
    d_publish();
}

// ---------- The recorder ----------
static void run(uint32_t id, uint8_t *p) {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    if (id != D_UPDATE_SETS && id != D_END) { memcpy(&cb, p, sizeof cb); p += 8; }
    switch (id) {
    case D_END: { GET(p, int, slot); GET(p, VkCommandBuffer, up); GET(p, VkCommandBuffer, mn); R.vkEndCommandBuffer(up); R.vkEndCommandBuffer(mn); vks_slot_recorded(slot); break; }
    case D_UPDATE_SETS: { GET(p, uint32_t, nw); VkWriteDescriptorSet *w = (VkWriteDescriptorSet *)p; p += SZN(VkWriteDescriptorSet, nw);
        for (uint32_t i = 0; i < nw; i++) {
            if (w[i].pImageInfo) { w[i].pImageInfo = (const VkDescriptorImageInfo *)p; p += SZN(VkDescriptorImageInfo, w[i].descriptorCount); }
            else if (w[i].pBufferInfo) { w[i].pBufferInfo = (const VkDescriptorBufferInfo *)p; p += SZN(VkDescriptorBufferInfo, w[i].descriptorCount); }
            else if (w[i].pTexelBufferView) { w[i].pTexelBufferView = (const VkBufferView *)p; p += SZN(VkBufferView, w[i].descriptorCount); }
        }
        R.vkUpdateDescriptorSets(g_vk.dev, nw, w, 0, NULL); break; }
    case D_BARRIER2: { GET(p, VkDependencyInfo, di);
        di.pMemoryBarriers = (const VkMemoryBarrier2 *)p; p += SZN(VkMemoryBarrier2, di.memoryBarrierCount);
        di.pBufferMemoryBarriers = (const VkBufferMemoryBarrier2 *)p; p += SZN(VkBufferMemoryBarrier2, di.bufferMemoryBarrierCount);
        di.pImageMemoryBarriers = (const VkImageMemoryBarrier2 *)p; R.vkCmdPipelineBarrier2(cb, &di); break; }
    case D_COPY_BUF: { GET(p, VkBuffer, s); GET(p, VkBuffer, d); GET(p, uint32_t, n); R.vkCmdCopyBuffer(cb, s, d, n, (const VkBufferCopy *)p); break; }
    case D_COPY_B2I: { GET(p, VkBuffer, s); GET(p, VkImage, d); GET(p, VkImageLayout, l); GET(p, uint32_t, n); R.vkCmdCopyBufferToImage(cb, s, d, l, n, (const VkBufferImageCopy *)p); break; }
    case D_COPY_I2B: { GET(p, VkImage, s); GET(p, VkImageLayout, l); GET(p, VkBuffer, d); GET(p, uint32_t, n); R.vkCmdCopyImageToBuffer(cb, s, l, d, n, (const VkBufferImageCopy *)p); break; }
    case D_COPY_IMG: { GET(p, VkImage, s); GET(p, VkImageLayout, sl); GET(p, VkImage, d); GET(p, VkImageLayout, dl); GET(p, uint32_t, n); R.vkCmdCopyImage(cb, s, sl, d, dl, n, (const VkImageCopy *)p); break; }
    case D_BLIT: { GET(p, VkImage, s); GET(p, VkImageLayout, sl); GET(p, VkImage, d); GET(p, VkImageLayout, dl); GET(p, uint32_t, n); GET(p, VkFilter, f); R.vkCmdBlitImage(cb, s, sl, d, dl, n, (const VkImageBlit *)p, f); break; }
    case D_RESOLVE: { GET(p, VkImage, s); GET(p, VkImageLayout, sl); GET(p, VkImage, d); GET(p, VkImageLayout, dl); GET(p, uint32_t, n); R.vkCmdResolveImage(cb, s, sl, d, dl, n, (const VkImageResolve *)p); break; }
    case D_UPDATE_BUF: { GET(p, VkBuffer, b); GET(p, VkDeviceSize, o); GET(p, VkDeviceSize, n); R.vkCmdUpdateBuffer(cb, b, o, n, p); break; }
    case D_CLEAR_COLOR: { GET(p, VkImage, im); GET(p, VkImageLayout, l); GET(p, uint32_t, n); GET(p, VkClearColorValue, c); R.vkCmdClearColorImage(cb, im, l, &c, n, (const VkImageSubresourceRange *)p); break; }
    case D_CLEAR_DS: { GET(p, VkImage, im); GET(p, VkImageLayout, l); GET(p, uint32_t, n); GET(p, VkClearDepthStencilValue, c); R.vkCmdClearDepthStencilImage(cb, im, l, &c, n, (const VkImageSubresourceRange *)p); break; }
    case D_CLEAR_ATT: { GET(p, uint32_t, n); GET(p, uint32_t, m); const VkClearAttachment *a = (const VkClearAttachment *)p; p += SZN(VkClearAttachment, n); R.vkCmdClearAttachments(cb, n, a, m, (const VkClearRect *)p); break; }
    case D_BEGIN_RENDERING: { GET(p, VkRenderingInfo, ri); ri.pNext = NULL;
        const VkRenderingAttachmentInfo *ca = (const VkRenderingAttachmentInfo *)p; p += SZN(VkRenderingAttachmentInfo, ri.colorAttachmentCount);
        const VkRenderingAttachmentInfo *d = (const VkRenderingAttachmentInfo *)p, *s = d + 1;
        ri.pColorAttachments = ri.colorAttachmentCount ? ca : NULL; ri.pDepthAttachment = ri.pDepthAttachment ? d : NULL; ri.pStencilAttachment = ri.pStencilAttachment ? s : NULL;
        R.vkCmdBeginRendering(cb, &ri); break; }
    case D_END_RENDERING: R.vkCmdEndRendering(cb); break;
    case D_BIND_PIPE: { GET(p, VkPipelineBindPoint, bp); GET(p, VkPipeline, pl); R.vkCmdBindPipeline(cb, bp, pl); break; }
    case D_SET_VP: { GET(p, uint32_t, f); GET(p, uint32_t, n); R.vkCmdSetViewport(cb, f, n, (const VkViewport *)p); break; }
    case D_SET_SC: { GET(p, uint32_t, f); GET(p, uint32_t, n); R.vkCmdSetScissor(cb, f, n, (const VkRect2D *)p); break; }
    case D_VP_CNT: { GET(p, uint32_t, n); R.vkCmdSetViewportWithCount(cb, n, (const VkViewport *)p); break; }
    case D_SC_CNT: { GET(p, uint32_t, n); R.vkCmdSetScissorWithCount(cb, n, (const VkRect2D *)p); break; }
    case D_DRAW: { GET(p, uint32_t, a); GET(p, uint32_t, b); GET(p, uint32_t, c); GET(p, uint32_t, d); R.vkCmdDraw(cb, a, b, c, d); break; }
    case D_PUSH: { GET(p, VkPipelineLayout, l); GET(p, VkShaderStageFlags, st); GET(p, uint32_t, off); GET(p, uint32_t, n); R.vkCmdPushConstants(cb, l, st, off, n, p); break; }
    case D_BIND_SETS: { GET(p, VkPipelineBindPoint, bp); GET(p, VkPipelineLayout, l); GET(p, uint32_t, first); GET(p, uint32_t, n); GET(p, uint32_t, nd);
        const VkDescriptorSet *s = (const VkDescriptorSet *)p; p += SZN(VkDescriptorSet, n); R.vkCmdBindDescriptorSets(cb, bp, l, first, n, s, nd, nd ? (const uint32_t *)p : NULL); break; }
    case D_BIND_VB2: { GET(p, uint32_t, first); GET(p, uint32_t, n); GET(p, uint32_t, has);
        const VkBuffer *b = (const VkBuffer *)p; p += SZN(VkBuffer, n); const VkDeviceSize *o = (const VkDeviceSize *)p; p += SZN(VkDeviceSize, n);
        const VkDeviceSize *sz = NULL, *st = NULL; if (has & 1) { sz = (const VkDeviceSize *)p; p += SZN(VkDeviceSize, n); } if (has & 2) st = (const VkDeviceSize *)p;
        R.vkCmdBindVertexBuffers2(cb, first, n, b, o, sz, st); break; }
    case D_BIND_IB: { GET(p, VkBuffer, b); GET(p, VkDeviceSize, o); GET(p, VkIndexType, t); R.vkCmdBindIndexBuffer(cb, b, o, t); break; }
    case D_DRAW_IDX: { GET(p, uint32_t, a); GET(p, uint32_t, b); GET(p, uint32_t, c); GET(p, int32_t, d); GET(p, uint32_t, e); R.vkCmdDrawIndexed(cb, a, b, c, d, e); break; }
    case D_DRAW_IND: { GET(p, VkBuffer, b); GET(p, VkDeviceSize, o); GET(p, uint32_t, n); GET(p, uint32_t, s); R.vkCmdDrawIndirect(cb, b, o, n, s); break; }
    case D_DRAW_IDX_IND: { GET(p, VkBuffer, b); GET(p, VkDeviceSize, o); GET(p, uint32_t, n); GET(p, uint32_t, s); R.vkCmdDrawIndexedIndirect(cb, b, o, n, s); break; }
    case D_DRAW_IND_CNT: { GET(p, VkBuffer, b); GET(p, VkDeviceSize, o); GET(p, VkBuffer, cbuf); GET(p, VkDeviceSize, co); GET(p, uint32_t, mx); GET(p, uint32_t, s); R.vkCmdDrawIndirectCount(cb, b, o, cbuf, co, mx, s); break; }
    case D_DRAW_IDX_IND_CNT: { GET(p, VkBuffer, b); GET(p, VkDeviceSize, o); GET(p, VkBuffer, cbuf); GET(p, VkDeviceSize, co); GET(p, uint32_t, mx); GET(p, uint32_t, s); R.vkCmdDrawIndexedIndirectCount(cb, b, o, cbuf, co, mx, s); break; }
    case D_DISPATCH: { GET(p, uint32_t, x); GET(p, uint32_t, y); GET(p, uint32_t, z); R.vkCmdDispatch(cb, x, y, z); break; }
    case D_DISPATCH_IND: { GET(p, VkBuffer, b); GET(p, VkDeviceSize, o); R.vkCmdDispatchIndirect(cb, b, o); break; }
    case D_FILL: { GET(p, VkBuffer, b); GET(p, VkDeviceSize, o); GET(p, VkDeviceSize, n); GET(p, uint32_t, v); R.vkCmdFillBuffer(cb, b, o, n, v); break; }
    case D_BEGIN_Q: { GET(p, VkQueryPool, qp); GET(p, uint32_t, q); GET(p, VkQueryControlFlags, f); R.vkCmdBeginQuery(cb, qp, q, f); break; }
    case D_END_Q: { GET(p, VkQueryPool, qp); GET(p, uint32_t, q); R.vkCmdEndQuery(cb, qp, q); break; }
    case D_RESET_QP: { GET(p, VkQueryPool, qp); GET(p, uint32_t, f); GET(p, uint32_t, n); R.vkCmdResetQueryPool(cb, qp, f, n); break; }
    case D_LINE_W: { GET(p, float, w); R.vkCmdSetLineWidth(cb, w); break; }
    case D_DEPTH_BIAS: { GET(p, float, a); GET(p, float, b); GET(p, float, c); R.vkCmdSetDepthBias(cb, a, b, c); break; }
    case D_BLEND_CONST: R.vkCmdSetBlendConstants(cb, (const float *)p); break;
    case D_ST_CMASK: { GET(p, VkStencilFaceFlags, f); GET(p, uint32_t, v); R.vkCmdSetStencilCompareMask(cb, f, v); break; }
    case D_ST_WMASK: { GET(p, VkStencilFaceFlags, f); GET(p, uint32_t, v); R.vkCmdSetStencilWriteMask(cb, f, v); break; }
    case D_ST_REF: { GET(p, VkStencilFaceFlags, f); GET(p, uint32_t, v); R.vkCmdSetStencilReference(cb, f, v); break; }
    case D_CULL: { GET(p, VkCullModeFlags, v); R.vkCmdSetCullMode(cb, v); break; }
    case D_FRONT: { GET(p, VkFrontFace, v); R.vkCmdSetFrontFace(cb, v); break; }
    case D_TOPO: { GET(p, VkPrimitiveTopology, v); R.vkCmdSetPrimitiveTopology(cb, v); break; }
    case D_DT: { GET(p, VkBool32, v); R.vkCmdSetDepthTestEnable(cb, v); break; }
    case D_DW: { GET(p, VkBool32, v); R.vkCmdSetDepthWriteEnable(cb, v); break; }
    case D_DCMP: { GET(p, VkCompareOp, v); R.vkCmdSetDepthCompareOp(cb, v); break; }
    case D_DBT: { GET(p, VkBool32, v); R.vkCmdSetDepthBoundsTestEnable(cb, v); break; }
    case D_ST_TEST: { GET(p, VkBool32, v); R.vkCmdSetStencilTestEnable(cb, v); break; }
    case D_ST_OP: { GET(p, VkStencilFaceFlags, f); GET(p, VkStencilOp, a); GET(p, VkStencilOp, b); GET(p, VkStencilOp, c); GET(p, VkCompareOp, d); R.vkCmdSetStencilOp(cb, f, a, b, c, d); break; }
    case D_RD: { GET(p, VkBool32, v); R.vkCmdSetRasterizerDiscardEnable(cb, v); break; }
    case D_BIAS_EN: { GET(p, VkBool32, v); R.vkCmdSetDepthBiasEnable(cb, v); break; }
    case D_RESTART: { GET(p, VkBool32, v); R.vkCmdSetPrimitiveRestartEnable(cb, v); break; }
    case D_TS2: { GET(p, VkPipelineStageFlags2, s); GET(p, VkQueryPool, qp); GET(p, uint32_t, q); R.vkCmdWriteTimestamp2(cb, s, qp, q); break; }
    case D_MESH: { GET(p, uint32_t, x); GET(p, uint32_t, y); GET(p, uint32_t, z); R.vkCmdDrawMeshTasksEXT(cb, x, y, z); break; }
    case D_VERTEX_INPUT: { GET(p, uint32_t, nb); GET(p, uint32_t, na); const VkVertexInputBindingDescription2EXT *b = (const VkVertexInputBindingDescription2EXT *)p; p += SZN(VkVertexInputBindingDescription2EXT, nb);
        R.vkCmdSetVertexInputEXT(cb, nb, b, na, (const VkVertexInputAttributeDescription2EXT *)p); break; }
    case D_BLEND_EN: { GET(p, uint32_t, f); GET(p, uint32_t, n); R.vkCmdSetColorBlendEnableEXT(cb, f, n, (const VkBool32 *)p); break; }
    case D_BLEND_EQ: { GET(p, uint32_t, f); GET(p, uint32_t, n); R.vkCmdSetColorBlendEquationEXT(cb, f, n, (const VkColorBlendEquationEXT *)p); break; }
    case D_WMASK: { GET(p, uint32_t, f); GET(p, uint32_t, n); R.vkCmdSetColorWriteMaskEXT(cb, f, n, (const VkColorComponentFlags *)p); break; }
    case D_LOGIC_EN: { GET(p, VkBool32, v); R.vkCmdSetLogicOpEnableEXT(cb, v); break; }
    case D_LOGIC: { GET(p, VkLogicOp, v); R.vkCmdSetLogicOpEXT(cb, v); break; }
    case D_A2C: { GET(p, VkBool32, v); R.vkCmdSetAlphaToCoverageEnableEXT(cb, v); break; }
    case D_POLY: { GET(p, VkPolygonMode, v); R.vkCmdSetPolygonModeEXT(cb, v); break; }
    case D_DCLAMP: { GET(p, VkBool32, v); R.vkCmdSetDepthClampEnableEXT(cb, v); break; }
    case D_PATCH: { GET(p, uint32_t, v); R.vkCmdSetPatchControlPointsEXT(cb, v); break; }
    case D_CHECKPOINT: { GET(p, const void *, m); R.vkCmdSetCheckpointNV(cb, m); break; }
    default: { static int said; if (!said++) VKRED("deferred recording: an unknown record %u in the ring (skipped)", id); break; }
    }
}
static _Atomic uint64_t s_rec_ns, s_rec_cyc;
static unsigned __stdcall recorder_main(void *arg) {
    (void)arg; SetThreadDescription(GetCurrentThread(), L"vk recorder"); SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    for (;;) {
        uint64_t r = atomic_load_explicit(&g_r, memory_order_relaxed), w = atomic_load_explicit(&g_w, memory_order_acquire);
        if (r == w) {   // empty: spin a little (the producer is usually mid-submission), then sleep until data comes
            for (int i = 0; i < 2000 && r == (w = atomic_load_explicit(&g_w, memory_order_acquire)); i++) YieldProcessor();
            if (r == w) {
                AcquireSRWLockExclusive(&g_lk); atomic_store(&g_rec_sleeps, 1);
                if (r == atomic_load(&g_w)) SleepConditionVariableSRW(&g_cv_data, &g_lk, 2, 0);
                atomic_store(&g_rec_sleeps, 0); ReleaseSRWLockExclusive(&g_lk);
                continue;
            }
        }
        static unsigned nb; int meas = !(nb++ & 63); uint64_t t0 = meas ? now_ns() : 0; ULONG64 c0 = 0, c1 = 0; if (meas) QueryThreadCycleTime(GetCurrentThread(), &c0);
        while (r != w) {
            dhdr *h = (dhdr *)(g_rb + r % RB);
            if (h->id != D_WRAP) run(h->id, (uint8_t *)(h + 1));
            r += h->size;
            atomic_store_explicit(&g_r, r, memory_order_release);
        }
        atomic_store_explicit(&g_r, r, memory_order_release);
        if (atomic_load(&g_prod_waits)) { AcquireSRWLockExclusive(&g_lk); WakeConditionVariable(&g_cv_room); ReleaseSRWLockExclusive(&g_lk); }
        if (meas) { QueryThreadCycleTime(GetCurrentThread(), &c1); atomic_fetch_add(&s_rec_ns, 64 * (now_ns() - t0)); atomic_fetch_add(&s_rec_cyc, 64 * (c1 - c0)); }   // sampled: x64
    }
    return 0;
}

// ---------- Interface (vk_submit.c, vk_dev.c) ----------
int vkd_on(void) { return g_on; }
void vkd_install(void) {   // after the device entry points are loaded: NXRT_VK_DEFER=1 turns it on
    const char *e = getenv("NXRT_VK_DEFER"); if (!e || atoi(e) <= 0) return;
    if (!(g_rb = VirtualAlloc(NULL, RB, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE))) { VKRED("deferred recording: no memory for its ring: off"); return; }
#define SWAP(n) R.n = n; if (n) n = t_##n;
    SWAP(vkCmdPipelineBarrier2) SWAP(vkCmdCopyBuffer) SWAP(vkCmdCopyBufferToImage) SWAP(vkCmdCopyImageToBuffer) SWAP(vkCmdCopyImage) SWAP(vkCmdBlitImage)
    SWAP(vkCmdFillBuffer) SWAP(vkCmdUpdateBuffer) SWAP(vkCmdClearColorImage) SWAP(vkCmdClearDepthStencilImage) SWAP(vkCmdBeginRendering) SWAP(vkCmdEndRendering)
    SWAP(vkCmdBindPipeline) SWAP(vkCmdSetViewport) SWAP(vkCmdSetScissor) SWAP(vkCmdDraw) SWAP(vkCmdPushConstants) SWAP(vkCmdBindDescriptorSets)
    SWAP(vkCmdBindVertexBuffers2) SWAP(vkCmdBindIndexBuffer) SWAP(vkCmdDrawIndexed) SWAP(vkCmdDrawIndirect) SWAP(vkCmdDrawIndexedIndirect)
    SWAP(vkCmdDrawIndirectCount) SWAP(vkCmdDrawIndexedIndirectCount) SWAP(vkCmdDispatch) SWAP(vkCmdDispatchIndirect) SWAP(vkCmdClearAttachments)
    SWAP(vkCmdResolveImage) SWAP(vkCmdBeginQuery) SWAP(vkCmdEndQuery) SWAP(vkCmdResetQueryPool) SWAP(vkCmdSetViewportWithCount) SWAP(vkCmdSetScissorWithCount)
    SWAP(vkCmdSetLineWidth) SWAP(vkCmdSetDepthBias) SWAP(vkCmdSetBlendConstants) SWAP(vkCmdSetStencilCompareMask) SWAP(vkCmdSetStencilWriteMask)
    SWAP(vkCmdSetStencilReference) SWAP(vkCmdSetCullMode) SWAP(vkCmdSetFrontFace) SWAP(vkCmdSetPrimitiveTopology) SWAP(vkCmdSetDepthTestEnable)
    SWAP(vkCmdSetDepthWriteEnable) SWAP(vkCmdSetDepthCompareOp) SWAP(vkCmdSetDepthBoundsTestEnable) SWAP(vkCmdSetStencilTestEnable) SWAP(vkCmdSetStencilOp)
    SWAP(vkCmdSetRasterizerDiscardEnable) SWAP(vkCmdSetDepthBiasEnable) SWAP(vkCmdSetPrimitiveRestartEnable) SWAP(vkCmdWriteTimestamp2) SWAP(vkCmdDrawMeshTasksEXT)
    SWAP(vkCmdSetVertexInputEXT) SWAP(vkCmdSetColorBlendEnableEXT) SWAP(vkCmdSetColorBlendEquationEXT) SWAP(vkCmdSetColorWriteMaskEXT) SWAP(vkCmdSetLogicOpEnableEXT)
    SWAP(vkCmdSetLogicOpEXT) SWAP(vkCmdSetAlphaToCoverageEnableEXT) SWAP(vkCmdSetPolygonModeEXT) SWAP(vkCmdSetDepthClampEnableEXT) SWAP(vkCmdSetPatchControlPointsEXT)
    SWAP(vkCmdSetCheckpointNV) SWAP(vkUpdateDescriptorSets)
#undef SWAP
    R.vkEndCommandBuffer = vkEndCommandBuffer;
    uintptr_t th = _beginthreadex(NULL, 0, recorder_main, NULL, 0, NULL);
    if (!th) { VKRED("deferred recording: cannot start the recorder thread"); return; }
    CloseHandle((HANDLE)th);
    g_on = 1;
    VKLOG("deferred recording on: the NVN submissions' Vulkan commands are recorded by a thread of their own (NXRT_VK_DEFER)");
}
// The recording thread starts recording slot's buffers (vks_begin, vks_split): from now on their commands go into the ring
void vkd_open(VkCommandBuffer up, VkCommandBuffer mn) {
    if (!g_on) return;
    g_dtid = GetCurrentThreadId(); g_dcb[0] = up; g_dcb[1] = mn; atomic_store(&g_ndcb, 2);
}
// Committing it: its buffers are ended by the recorder after everything recorded so far, then vks_slot_recorded(slot); no more commands for them come from here
void vkd_end(int slot, VkCommandBuffer up, VkCommandBuffer mn) {
    uint8_t *p = d_alloc(D_END, 24); PUT(p, slot); PUT(p, up); PUT(p, mn); d_publish();
    atomic_store(&g_ndcb, 0);
}
int vkd_deferred(VkCommandBuffer cb) { return g_on && deferred(cb); }
void vkd_stats(uint64_t *calls, uint64_t *bytes_waits, uint64_t *rec_ns, uint64_t *rec_cyc) { *calls = atomic_load(&s_calls); *bytes_waits = atomic_load(&s_waits); *rec_ns = atomic_load(&s_rec_ns); *rec_cyc = atomic_load(&s_rec_cyc); }
