// The replay of NVN command streams on Vulkan (vk_replay.h; docs/m5-replay.md). Ported behaviour from nxnative's dm/gfx/nvn/nvn_mtl.m (MIT): the record
// walk (mtl_exec), the binding state and its defaults (state_op, state_reset, per-queue state), what each shader binding takes (encode_args_: uniform
// slots, the driver segments c0 / c2 and their tables, the support buffer, c1, textures through the registered handles), the viewport, scissor and
// winding rules (draw_), clears with their scissor and channel mask (OP_CLEAR_COLOR / OP_CLEAR_DS), copies (copy_b2t / copy_t2b / copy_t2t) with the
// storage mirror of storage-backed textures, counters (occlusion queries summed from reset to report). The Vulkan side is ours: dynamic rendering, one
// descriptor set per stage, extended dynamic state, the pool mirrors of vk_mem.c and the images of vk_tex.c, GENERAL as the resting layout of GPU images.
#include "game_profile.h"
#include <stdarg.h>
#include "vk_replay.h"
#include "vk_submit.h"
#include "vk_mem.h"
#include "vk_tex.h"
#include "vk_fmt.h"
#include "vk_shader.h"
#include "vk_pipe.h"
#include "vk_hmap.h"
#include "vk_shot.h"
#include "vk_flight.h"
#include "nvn_timestamp.h"
#include "clear_vert.h"
#include "clear_f_frag.h"
#include "clear_u_frag.h"
#include "clear_i_frag.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void dm_guest_write_prepare(void *addr, size_t len);
int nvn_dpool_base(const void *pool, int tex);   // hle_nvn.c: where a descriptor pool's IDs start in the host table
void vk_note_fence_sync(sync_t *s);              // gfx_vk.c: the sync gets the submission's value at the commit
uint64_t guest_ns(void);
void vkpc_note_created(int n);

#define RED_ONCE(id, ...) do { static int once_##id; if (!once_##id++) VKRED(__VA_ARGS__); } while (0)
// SUPPORT_SIZE: the translator's support buffer (1392 bytes) plus the word nxshader's AbsGuard edit reads as a zero the shader compiler cannot predict (offset 1392;
// the buffer is zero-filled once and never written there, see vkr_init)
enum { SUPPORT_SIZE = 1408, ZERO_SIZE = 65536, NTEX = 256, MAXB = 192, NQ = 4096 };

static vkr_stats_t g_st;
static uint64_t g_pd_frame, g_pd_last; static int g_pd;   // NXRT_VK_PASSDUMP=<present>[-<present>]: the frames after those presents, pass by pass and draw by draw, in the log
#define PD(...) do { if (g_pd) nxrt_logf("[pd] " __VA_ARGS__); } while (0)
static vkt_image *g_td[1024]; static int g_ntd, g_td_on; static uint64_t g_td_frame;   // NXRT_VK_TEXDUMP=<present>: the images that frame drew into or wrote, as PNG at the next present
static int g_no_tex, g_raw_on; static void td_note(vkt_image *im) { if (!g_td_on || !im || g_no_tex) return; for (int i = 0; i < g_ntd; i++) if (g_td[i] == im) return; if (g_ntd < 1024) g_td[g_ntd++] = im; }
const char *nxrt_log_path(void);
static int g_origin_ll = 1, g_queue_origin_ll = 1, g_depth_zero = 0;
static VkPipeline g_last_pipe;   // the graphics pipeline bound last in this submission

// ---------- Resources that live with a submission: arena chunks, descriptor pools, buffer views (returned at completion) ----------
typedef struct rchunk { VkBuffer buf; vk_alloc mem; VkDeviceSize size, used; struct rchunk *next; } rchunk;
typedef struct rdpool { VkDescriptorPool pool; struct rdpool *next; } rdpool;
typedef struct { rchunk *chunks; rdpool *pools; VkBufferView *views; int nviews, cviews; int tq; } subres;   // tq: timestamp pair + 1
static subres g_sub;
// GPU time: a timestamp before and after each submission's replay, read at its completion
enum { NTQ = 256 };
static VkQueryPool g_tqp; static uint32_t g_tq_next; static _Atomic uint64_t g_gpu_ns, g_gpu_subs;
static rchunk *g_free_chunks; static rdpool *g_free_pools; static SRWLOCK g_free_lk = SRWLOCK_INIT;

static void subres_done(void *arg) {
    subres *r = arg;
    uint64_t t[2];
    if (r->tq && vkGetQueryPoolResults(g_vk.dev, g_tqp, 2 * (uint32_t)(r->tq - 1), 2, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS && t[1] >= t[0]) {
        atomic_fetch_add(&g_gpu_ns, (uint64_t)((double)(t[1] - t[0]) * g_vk.props.limits.timestampPeriod)); atomic_fetch_add(&g_gpu_subs, 1);
    }
    AcquireSRWLockExclusive(&g_free_lk);
    while (r->chunks) { rchunk *c = r->chunks; r->chunks = c->next; c->next = g_free_chunks; g_free_chunks = c; }
    while (r->pools) { rdpool *p = r->pools; r->pools = p->next; p->next = g_free_pools; g_free_pools = p; }
    ReleaseSRWLockExclusive(&g_free_lk);
    for (int i = 0; i < r->nviews; i++) vkDestroyBufferView(g_vk.dev, r->views[i], NULL);
    free(r->views); free(r);
}
static rchunk *chunk_take(VkDeviceSize need) {
    AcquireSRWLockExclusive(&g_free_lk);
    rchunk **pp = &g_free_chunks; while (*pp && (*pp)->size < need) pp = &(*pp)->next;
    rchunk *c = *pp; if (c) *pp = c->next;
    ReleaseSRWLockExclusive(&g_free_lk);
    if (c) { c->used = 0; return c; }
    c = calloc(1, sizeof *c); c->size = need > (4u << 20) ? need : (4u << 20);
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, c->size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    if (vkCreateBuffer(g_vk.dev, &bci, NULL, &c->buf) != VK_SUCCESS) { free(c); return NULL; }
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, c->buf, &mr);
    if (vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, &c->mem)) { vkDestroyBuffer(g_vk.dev, c->buf, NULL); free(c); return NULL; }
    vkBindBufferMemory(g_vk.dev, c->buf, c->mem.mem, c->mem.off);
    vk_name((uint64_t)c->buf, VK_OBJECT_TYPE_BUFFER, "replay arena");
    return c;
}
// Host-visible memory the GPU reads in this submission (driver segments, expanded indices, copies of unaligned ranges)
static void *arena(VkDeviceSize n, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off) {
    if (!align) align = 16;
    rchunk *c = g_sub.chunks;
    if (!c || (c->used + align - 1) / align * align + n > c->size) {
        if (!(c = chunk_take(n))) { RED_ONCE(1, "replay arena: no host memory"); return NULL; }
        c->next = g_sub.chunks; g_sub.chunks = c;
    }
    VkDeviceSize o = (c->used + align - 1) / align * align;
    c->used = o + n; *buf = c->buf; *off = o;
    return (uint8_t *)c->mem.map + o;
}
static VkDescriptorPool dpool_take(void) {
    AcquireSRWLockExclusive(&g_free_lk);
    rdpool *p = g_free_pools; if (p) g_free_pools = p->next;
    ReleaseSRWLockExclusive(&g_free_lk);
    if (p) { vkResetDescriptorPool(g_vk.dev, p->pool, 0); }
    else {
        p = calloc(1, sizeof *p);
        VkDescriptorPoolSize ps[] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16384}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8192}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048},
                                     {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 512},
                                     {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1024}};
        VkDescriptorPoolCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 2048, sizeof ps / sizeof *ps, ps};
        if (vkCreateDescriptorPool(g_vk.dev, &ci, NULL, &p->pool) != VK_SUCCESS) { free(p); return VK_NULL_HANDLE; }
    }
    p->next = g_sub.pools; g_sub.pools = p;
    return p->pool;
}
static VkDescriptorSet dset_alloc(VkDescriptorSetLayout l) {
    VkDescriptorSet s = VK_NULL_HANDLE;
    for (int attempt = 0; attempt < 2; attempt++) {
        VkDescriptorPool p = g_sub.pools && !attempt ? g_sub.pools->pool : dpool_take();
        if (!p) break;
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, p, 1, &l};
        if (vkAllocateDescriptorSets(g_vk.dev, &ai, &s) == VK_SUCCESS) { g_st.desc_sets++; return s; }
    }
    RED_ONCE(2, "descriptor sets exhausted"); return VK_NULL_HANDLE;
}
static uint64_t g_n_bview, g_n_ds_set, g_n_ds_skip;   // per-frame counts in the "replay calls" line: buffer views made, dynamic state calls made and skipped (g_ds)
static void view_keep(VkBufferView v) {
    g_n_bview++;
    if (g_sub.nviews == g_sub.cviews) { g_sub.cviews = g_sub.cviews ? 2 * g_sub.cviews : 32; g_sub.views = realloc(g_sub.views, (size_t)g_sub.cviews * sizeof *g_sub.views); }
    g_sub.views[g_sub.nviews++] = v;
}

// ---------- Fixed resources ----------
static struct { VkBuffer zero, sink, support, quads, bounce; vk_alloc zero_m, sink_m, support_m, quads_m, bounce_m; VkSampler samp; VkQueryPool qp; } R0;
// A storage buffer at an offset Vulkan cannot bind (minStorageBufferOffsetAlignment; Metal binds at any 4-byte offset): for a dispatch its bytes are copied into
// an aligned place of R0.bounce, bound there and, when the shader writes it, copied back after the dispatch (M6: a 4-byte counter of a gameplay compute shader)
enum { BOUNCE_SIZE = 1 << 20, BOUNCE_MAX = 64 << 10 };
static struct { VkBuffer pool; VkDeviceSize poff, boff, n; } g_bnc[16]; static int g_nbnc; static VkDeviceSize g_bnc_next;
static int mkbuf(VkDeviceSize n, VkBufferUsageFlags u, VkMemoryPropertyFlags want, VkBuffer *b, vk_alloc *m, const char *name) {
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, n, u, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    if (vkCreateBuffer(g_vk.dev, &bci, NULL, b) != VK_SUCCESS) return -1;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, *b, &mr);
    if (vk_alloc_mem(&mr, want, 0, m)) return -1;
    vkBindBufferMemory(g_vk.dev, *b, m->mem, m->off);
    vk_name((uint64_t)*b, VK_OBJECT_TYPE_BUFFER, name);
    return 0;
}

// ---------- Binding state (nvn_mtl.m's S and R: per queue, kept across submissions) ----------
typedef struct rs {
    uint64_t serial;
    const prog_t *bst[5]; const prog_t *cprog;
    blend_st blend[8]; color_st color; chmask_st chmask; ms_st ms; poly_st poly; ds_st ds;
    va_st attr[16]; int nattr; vs_st stream[16]; int nstream; NVNbufferRange vbuf[16];
    p_bind_buf ubo[6][16], ssbo[6][16];
    uint8_t push[2][DMSH_PUSH_SIZE];   // NVN2 push constants of the graphics stages and of compute, c[24]
    uint64_t tex[6][NTEX], samp[6][128], img[6][32];
    uint32_t st_vmask[2], st_wmask[2], st_ref[2];
    p_poly_offset po; float vp[16][4]; float dr[2]; int has_vp, depth_clamp; uint8_t vp_sw[16][4];
    float blend_color[4], alpha_ref; int64_t copy_stride, copy_img_stride;
    int tf_on, rast_discard, tex_base, samp_base, render_skip, vis_on;
    p_prim_restart restart;
    // render targets (the snapshots copied: the record's command memory may be reused after the submission)
    int nrt; tex_t ct[8]; uint8_t has_ct[8]; view_t cview[8]; uint8_t has_cview[8]; tex_t dt; uint8_t has_dt; view_t dview; uint8_t has_dview;
    p_scissor sc; int has_sc;
    struct rs *next;
} rs;
static rs *g_states, *S;
static void rs_reset(rs *s) {
    uint64_t serial = s->serial; struct rs *next = s->next;
    memset(s, 0, sizeof *s); s->serial = serial; s->next = next;
    for (int i = 0; i < 8; i++) s->blend[i] = (blend_st){(uint64_t)i, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD};
    s->color = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}; s->chmask.mask = 0xffffffffu;
    s->poly = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0};
    s->ds = (ds_st){0, 1, NVN_DEPTH_FUNC_LESS, 0, NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS, 1, 1, 1, 1, 1, 1};
    s->st_vmask[0] = s->st_vmask[1] = s->st_wmask[0] = s->st_wmask[1] = 0xff; s->dr[1] = 1;
}
static rs *rs_get(uint64_t serial) {
    for (rs *s = g_states; s; s = s->next) if (s->serial == serial) return s;
    rs *s = calloc(1, sizeof *s); s->serial = serial; s->next = g_states; g_states = s; rs_reset(s);
    return s;
}

static void maybe_split(void);
// ---------- Programs: the shader of each NVN stage, cached by program serial ----------
typedef struct { const vks_shader *sh[6]; int done; } progc;
static vk_hmap g_progc = {8};
static const vks_shader *prog_stage(const prog_t *p, int stage) {
    if (!p || !p->serial) return NULL;
    progc *c = vk_hm_get(&g_progc, &p->serial);
    if (c && (c->sh[stage] || c->done)) return c->sh[stage];   // done: a stage this program does not have stays NULL, with no second look (a formatted lookup for every draw took 8% of this thread)
    if (!c) { c = calloc(1, sizeof *c); vk_hm_put(&g_progc, &p->serial, c); }
    int got = 0; for (unsigned i = 0; i < p->nstage && i < 6; i++) { const vks_shader *s = vksh_get(p, i); if (s && s->stage >= 0 && s->stage < 6) { c->sh[s->stage] = s; got++; } }
    c->done = got == (int)(p->nstage < 6 ? p->nstage : 6);
    return c->sh[stage];
}

// ---------- Barriers ----------
static void barrier_all(VkCommandBuffer cb) {
    vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    g_st.barriers++;
}

// ---------- GPU writes into pools: copied back to guest memory at completion (vkm_gpu_wrote; transfers cannot be recorded inside a pass) ----------
static struct { uint64_t a, n, base; } g_wb[256]; static int g_nwb;
static void wb_flush(void) { for (int i = 0; i < g_nwb; i++) { if (g_wb[i].base) vkm_gpu_wrote_base(g_wb[i].a, g_wb[i].n, g_wb[i].base); else vkm_gpu_wrote(g_wb[i].a, g_wb[i].n); g_st.writebacks++; } g_nwb = 0; }

// ---------- Occlusion queries: samples counted from a reset to each report, summed on the CPU when the GPU is done (in submission order) ----------
static _Atomic uint8_t g_qbusy[NQ]; static uint32_t g_qnext; static uint64_t g_vis_acc;
typedef struct { int kind; uint32_t q; uint64_t addr; } qpost;   // kind 0 add the result of q, 1 reset, 2 report into addr
static void q_post_run(void *arg) {
    qpost *p = arg;
    if (p->kind == 0) {
        uint64_t v = 0;
        if (vkGetQueryPoolResults(g_vk.dev, R0.qp, p->q, 1, sizeof v, &v, sizeof v, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) g_vis_acc += v;
        atomic_store(&g_qbusy[p->q], 0);
    } else if (p->kind == 1) g_vis_acc = 0;
    else {
        uint64_t rec[2] = {g_vis_acc, nvn_timestamp_from_ns(guest_ns())};
        dm_guest_write_prepare((void *)(uintptr_t)p->addr, sizeof rec);
        memcpy((void *)(uintptr_t)p->addr, rec, sizeof rec);
    }
    free(p);
}
static void q_post(int kind, uint32_t q, uint64_t addr) { qpost *p = malloc(sizeof *p); p->kind = kind; p->q = q; p->addr = addr; vks_post_fn(q_post_run, p); }
static int q_alloc(void) {
    for (int round = 0; round < 2; round++) {
        for (uint32_t k = 0; k < NQ; k++) { uint32_t q = (g_qnext + k) % NQ; if (!atomic_load(&g_qbusy[q])) { g_qnext = q + 1; atomic_store(&g_qbusy[q], 1); vkResetQueryPool(g_vk.dev, R0.qp, q, 1); return (int)q; } }
        RED_ONCE(3, "all %d occlusion queries in flight", NQ); return -1;
    }
    return -1;
}

// ---------- Render passes (dynamic rendering) ----------
typedef struct {
    int valid, open, ncolor, any;
    VkImageView cv[8]; vkt_image *ci[8]; VkFormat cf[8]; uint32_t cw[8], ch[8], cnl[8], clv[8], cl0[8];   // clv / cl0: the level and first layer (partial clears)
    VkImageView dv; vkt_image *di; VkFormat df; int hd, hs; uint32_t dw, dh, dnl, dlv, dl0;
    uint32_t w, h; uint8_t samples;
    int query;
} pass_t;
static pass_t P = {.query = -1};
static uint32_t mipd(uint32_t s, uint32_t l) { s >>= l; return s ? s : 1; }

// The dump frame's colour targets as they stand when each pass ends (a target the game uses for a moment is something else by the end of the frame):
// <log>.pass-<frame>-<n>-c<slot>-ser<serial>-<format>-<w>x<h>.raw, read with tools/pass_raw.py. Copied in the frame's own command buffer, so its order and timing stay
typedef struct { VkBuffer b; vk_alloc m; uint64_t ser; int n, slot; uint32_t w, h, bytes; const char *fmt; } td_copy;
static td_copy g_tc[256]; static int g_ntc, g_tc_pass, g_id_on, g_id_mode;
static void td_pass_copies(VkCommandBuffer cb) {
    g_tc_pass++;
    for (int i = 0; i < P.ncolor; i++) {
        vkt_image *im = P.ci[i];
        if (!im || g_ntc >= 256 || (im->fi->flags & VKF_COMPRESSED) || P.cw[i] < (g_id_mode ? 2000u : 64u)) continue;
        uint32_t w = P.cw[i], h = P.ch[i], bytes = w * h * im->fi->vbpb;
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
        td_copy *c = &g_tc[g_ntc];
        if (vkCreateBuffer(g_vk.dev, &bci, NULL, &c->b) != VK_SUCCESS) continue;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, c->b, &mr);
        if (vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0, &c->m) &&
            vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, &c->m)) { vkDestroyBuffer(g_vk.dev, c->b, NULL); continue; }
        vkBindBufferMemory(g_vk.dev, c->b, c->m.mem, c->m.off);
        vk_image_barrier(cb, im->img, im->aspect, im->rest, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy rg = {0, 0, 0, {im->aspect, P.clv[i], P.cl0[i], 1}, {0, 0, 0}, {w, h, 1}};
        vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, c->b, 1, &rg);
        vk_image_barrier(cb, im->img, im->aspect, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, im->rest, VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
        c->ser = im->serial; c->n = g_tc_pass; c->slot = i; c->w = w; c->h = h; c->bytes = bytes; c->fmt = im->fi->name; g_ntc++;
    }
}
static void td_pass_write(uint64_t frame) {
    const char *lp = nxrt_log_path();
    for (int i = 0; i < g_ntc; i++) {
        td_copy *c = &g_tc[i]; char path[1200];
        snprintf(path, sizeof path, "%s.pass-%llu-%04d-c%d-ser%llu-%s-%ux%u.raw", lp && *lp ? lp : "nxrt", (unsigned long long)frame, c->n, c->slot, (unsigned long long)c->ser, c->fmt, c->w, c->h);
        FILE *f = fopen(path, "wb"); if (f) { fwrite(c->m.map, 1, c->bytes, f); fclose(f); }
        vkDestroyBuffer(g_vk.dev, c->b, NULL); vk_free_mem(&c->m);
    }
    g_ntc = 0;
}
static uint64_t g_pv_ps; static uint32_t g_pass_in_frame;
static uint32_t P_index(void) { return g_pass_in_frame; }
// gamectl gputime: one frame timed on the GPU, a timestamp after each pass and dispatch and at each submission's two ends; the log gets totals and the slowest
enum { GT_MAX = 8000 };
typedef struct { char kind; uint32_t w, h; uint64_t draws; char name[40]; } gt_label;
static VkQueryPool g_gtp; static int g_gt_on, g_gt_n; static gt_label g_gtl[GT_MAX]; static _Atomic long g_gt_req; static uint64_t g_gt_at, g_gt_d0; static uint32_t g_gt_w, g_gt_h; static char g_gt_fmt[20], g_gt_fs[18];
void vkr_gputime_next(void) { atomic_store(&g_gt_req, 1); }
static void gt_mark(VkCommandBuffer cb, char kind, uint32_t w, uint32_t h, uint64_t draws, const char *name) {
    if (!g_gt_on || g_gt_n >= GT_MAX || !cb) return;
    gt_label *l = &g_gtl[g_gt_n]; l->kind = kind; l->w = w; l->h = h; l->draws = draws; snprintf(l->name, sizeof l->name, "%s", name ? name : "");
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, g_gtp, (uint32_t)g_gt_n); g_gt_n++;
}
static int gt_cmp(const void *a, const void *b) { double x = ((const double *)a)[0], y = ((const double *)b)[0]; return x < y ? 1 : x > y ? -1 : 0; }
static void gt_report(void) {
    static uint64_t t[GT_MAX]; static double o[GT_MAX][2]; int n = g_gt_n; if (n < 2) return;
    if (vkGetQueryPoolResults(g_vk.dev, g_gtp, 0, (uint32_t)n, (size_t)n * 8, t, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) { VKLOG("gputime: no results"); return; }
    double per = g_vk.props.limits.timestampPeriod / 1e6, kinds[128] = {0}; int cnt[128] = {0};
    for (int i = 1; i < n; i++) { double ms = (double)(t[i] - t[i - 1]) * per; if (t[i] < t[i - 1]) ms = 0; kinds[(int)g_gtl[i].kind & 127] += ms; cnt[(int)g_gtl[i].kind & 127]++; o[i - 1][0] = ms; o[i - 1][1] = i; }
    VKLOG("gputime: %.2f ms from the first mark to the last; passes %.2f (%d), dispatches %.2f (%d), before each submission's first work (uploads and idle) %.2f (%d), the end of submissions %.2f (%d)",
          (double)(t[n - 1] - t[0]) * per, kinds['p'], cnt['p'], kinds['d'], cnt['d'], kinds['s'], cnt['s'], kinds['e'], cnt['e']);
    qsort(o, (size_t)n - 1, sizeof o[0], gt_cmp);
    for (int i = 0; i < n - 1; i++) { const gt_label *l = &g_gtl[(int)o[i][1]]; VKLOG("gputime: %6.2f ms  %c %ux%u %s, %llu draws (mark %d)", o[i][0], l->kind, l->w, l->h, l->name, (unsigned long long)l->draws, (int)o[i][1]); }
}
static void gt_frame(uint64_t frame) {
    if (g_gt_on) { g_gt_on = 0; g_gt_at = frame + 4; }
    if (g_gt_at && frame >= g_gt_at) { g_gt_at = 0; gt_report(); }
    if (atomic_exchange(&g_gt_req, 0)) {
        if (!g_gtp) { VkQueryPoolCreateInfo ci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, GT_MAX, 0}; if (vkCreateQueryPool(g_vk.dev, &ci, NULL, &g_gtp) != VK_SUCCESS) return; }
        vkResetQueryPool(g_vk.dev, g_gtp, 0, GT_MAX); g_gt_n = 0; g_gt_on = 1;
    }
}
static void pass_end(void) {
    if (!P.open) return;
    VkCommandBuffer cb = vks_cb();
    vkq_close(cb);
    if (P.query >= 0) { vkCmdEndQuery(cb, R0.qp, (uint32_t)P.query); q_post(0, (uint32_t)P.query, 0); P.query = -1; }
    vkCmdEndRendering(cb);
    if (g_gt_on) { char nm[40]; snprintf(nm, sizeof nm, "%s fs %s", g_gt_fmt, g_gt_fs); gt_mark(cb, 'p', g_gt_w, g_gt_h, g_st.draws - g_gt_d0, nm); }
    g_pass_in_frame++; vkq_reset_if(cb);
    if ((g_td_on || g_id_on) && getenv("NXRT_VK_TD_PASSES")) td_pass_copies(cb);
    if (vkv_on()) { for (int i = 0; i < P.ncolor; i++) if (P.ci[i]) vkv_image(cb, P.ci[i], P.clv[i], P.cl0[i], P.cw[i], P.ch[i], i, g_pv_ps, 0); if (P.di) vkv_image(cb, P.di, 0, 0, P.di->w, P.di->h, 9, g_pv_ps, 0); }
    P.open = 0;
    wb_flush();
    vkf_args_flush(cb);
    vkw_flush(cb);
}
// An attachment's view: one level, the layers the NVN view names (else all from its first), the view's format through the image's fallback, identity swizzle
static VkImageView att_view(vkt_image *im, const tex_t *t, const view_t *v, int has_v, int depth, uint32_t *w, uint32_t *h, uint32_t *nl, VkFormat *fmt, uint32_t *lv, uint32_t *l0) {
    uint32_t level = has_v && v->set_levels ? v->base_level : 0, b0 = has_v && v->set_layers ? v->min_layer : 0;
    if (level >= im->levels) level = im->levels - 1;
    int is3d = im->vtype == VK_IMAGE_VIEW_TYPE_3D;
    uint32_t all = is3d ? mipd(im->depth, level) : im->layers, n = has_v && v->set_layers && v->nlayers ? v->nlayers : all > b0 ? all - b0 : 1;
    if (b0 >= all) b0 = all - 1; if (b0 + n > all) n = all - b0;
    VkFormat f = depth ? im->fi->vk : vkf_view_format(t->d.format, has_v && v->set_format ? v->format : t->d.format);
    *w = mipd(im->w, level); *h = mipd(im->h, level); *nl = n; *fmt = f; *lv = level; *l0 = b0;
    VkImageViewType vt = n > 1 || is3d || im->layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    return vkt_view_raw_(im, f, vt, level, 1, b0, n, depth ? im->aspect : VK_IMAGE_ASPECT_COLOR_BIT, NULL);
}
static void pass_targets(void) {
    memset(P.cv, 0, sizeof P.cv); memset(P.ci, 0, sizeof P.ci); memset(P.cf, 0, sizeof P.cf);
    P.dv = VK_NULL_HANDLE; P.di = NULL; P.df = VK_FORMAT_UNDEFINED; P.hd = P.hs = 0; P.ncolor = 0; P.any = 0;
    P.w = P.h = 16384; P.samples = 1;
    for (int i = 0; i < S->nrt && i < 8; i++) {
        if (!S->has_ct[i]) continue;
        const tex_t *t = &S->ct[i];
        int dup = 0; for (int j = 0; j < i; j++) dup |= S->has_ct[j] && S->ct[j].serial == t->serial &&
            (S->has_cview[j] && S->cview[j].set_levels ? S->cview[j].base_level : 0) == (S->has_cview[i] && S->cview[i].set_levels ? S->cview[i].base_level : 0) &&
            (S->has_cview[j] && S->cview[j].set_layers ? S->cview[j].min_layer : 0) == (S->has_cview[i] && S->cview[i].set_layers ? S->cview[i].min_layer : 0);
        if (dup) { RED_ONCE(10, "the same texture level is attached at two colour slots: only the first is attached (nvn_mtl.m rt_dup_slot)"); continue; }
        vkt_image *im = vkt_resolve(t);
        if (!im) continue;
        if (!(im->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) {
            static int n; if (n++ < 8) VKRED("render target ser=%llu %ux%u format %#x (%s) cannot be a colour attachment here (%s): draws to it are lost",
                                             (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format, im->fi->name, im->cls == VKT_STORAGE ? "its content comes from its storage" : "the format cannot");
            continue;
        }
        P.cv[i] = att_view(im, t, &S->cview[i], S->has_cview[i], 0, &P.cw[i], &P.ch[i], &P.cnl[i], &P.cf[i], &P.clv[i], &P.cl0[i]);
        if (!P.cv[i]) continue;
        P.ci[i] = im; P.ncolor = i + 1; P.any = 1; td_note(im);
        if (P.cw[i] < P.w) P.w = P.cw[i]; if (P.ch[i] < P.h) P.h = P.ch[i];
        P.samples = (uint8_t)im->samples;
    }
    if (S->has_dt) {
        const tex_t *t = &S->dt; vkt_image *im = vkt_resolve(t);
        if (im && (im->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
            P.dv = att_view(im, t, &S->dview, S->has_dview, 1, &P.dw, &P.dh, &P.dnl, &P.df, &P.dlv, &P.dl0);
            if (P.dv) {
                P.di = im; P.hd = (im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0; P.hs = (im->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) != 0; P.any = 1;
                if (P.dw < P.w) P.w = P.dw; if (P.dh < P.h) P.h = P.dh;
                P.samples = (uint8_t)im->samples;
            }
        } else if (im) RED_ONCE(11, "depth target format %#x cannot be a depth attachment: not attached", t->d.format);
    }
    if (!P.any) { P.w = P.h = 0; }
    P.valid = 1;
}
// Opens the pass over the current targets (loading what they hold). An attachment-less pass covers the viewport (nvn_mtl.m bare_dims)
static int pass_begin(void) {
    if (P.open) return 1;
    if (!P.valid) pass_targets();
    uint32_t w = P.w, h = P.h;
    if (!P.any) {
        if (S->nrt || S->has_dt) return 0;   // targets set but none could be attached
        w = S->has_vp ? (uint32_t)(S->vp[0][0] + S->vp[0][2]) : 0; h = S->has_vp ? (uint32_t)(S->vp[0][1] + S->vp[0][3]) : 0;
        if (!w || !h) return 0;
        RED_ONCE(12, "attachment-less pass at the viewport's size %ux%u (nvn_mtl.m bare_dims)", w, h);
    }
    VkCommandBuffer cb = vks_cb();
    barrier_all(cb);
    VkRenderingAttachmentInfo ca[8], da = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO}, sa;
    for (int i = 0; i < P.ncolor; i++)
        ca[i] = (VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, P.cv[i], VK_IMAGE_LAYOUT_GENERAL, 0, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
                                            VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, {{{0}}}};
    if (P.dv) da = (VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, P.dv, VK_IMAGE_LAYOUT_GENERAL, 0, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
                                               VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, {{{0}}}};
    sa = da;
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, {{0, 0}, {w, h}}, 1, 0, (uint32_t)P.ncolor, ca, P.hd ? &da : NULL, P.hs ? &sa : NULL};
    vkCmdBeginRendering(cb, &ri);
    P.open = 1; g_st.passes++;
    if (g_gt_on) { g_gt_w = w; g_gt_h = h; g_gt_d0 = g_st.draws; snprintf(g_gt_fmt, sizeof g_gt_fmt, "%s", P.ncolor && P.ci[0] ? P.ci[0]->fi->name : P.di ? P.di->fi->name : "-"); }
    if (g_pd) {
        char b[600]; int o = 0;
        for (int i = 0; i < P.ncolor; i++) if (P.cv[i]) o += snprintf(b + o, sizeof b - o, " c%d=ser%llu/%s/%ux%u", i, (unsigned long long)P.ci[i]->serial, P.ci[i]->fi->name, P.cw[i], P.ch[i]);
        if (P.dv) o += snprintf(b + o, sizeof b - o, " d=ser%llu/%s/%ux%u", (unsigned long long)P.di->serial, P.di->fi->name, P.dw, P.dh);
        PD("pass %llu %ux%u%s", (unsigned long long)g_st.passes, w, h, b);
        for (int i = 0; i < S->nrt; i++) if (S->has_ct[i]) PD("  c%d storage pool %p +%#llx mem %p flags %#x size %#llx image %p", i, (const void *)S->ct[i].pool, (unsigned long long)S->ct[i].off, (const void *)S->ct[i].mem, S->ct[i].d.flags, (unsigned long long)S->ct[i].L.size, (void *)P.ci[i]);
    }
    return 1;
}

// ---------- Clears (nvn_mtl.m OP_CLEAR_COLOR / OP_CLEAR_DS): NVN clears the whole target level (every layer of its view) within the scissor ----------
static int fmt_int_kind(VkFormat f) {   // 0 float / normalized, 1 unsigned integer, 2 signed integer
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_S8_UINT: return 1;
    case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_A2B10G10R10_SINT_PACK32: return 2;
    default: return 0;
    }
}
static unsigned fmt_channels(VkFormat f) {   // the colour channels a format has, as an NVN channel mask (nvn_mtl.m mtl_pf_chmask)
    switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_SNORM: case VK_FORMAT_R16_SFLOAT:
    case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT: case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R8_SRGB: return 0x1;
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_SNORM:
    case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: return 0x3;
    case VK_FORMAT_R5G6B5_UNORM_PACK16: case VK_FORMAT_B5G6R5_UNORM_PACK16: case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32: return 0x7;
    default: return 0xf;
    }
}
static VkClearValue clear_color_value(VkFormat f, const float *rgba, const uint32_t *u, int is_uint) {
    VkClearValue v; int k = fmt_int_kind(f);
    for (int i = 0; i < 4; i++) {
        if (is_uint) { if (k == 0) v.color.float32[i] = (float)u[i]; else v.color.uint32[i] = u[i]; }
        else if (k == 1) v.color.uint32[i] = rgba[i] <= 0 ? 0 : (uint32_t)rgba[i];
        else if (k == 2) v.color.int32[i] = (int32_t)rgba[i];
        else v.color.float32[i] = rgba[i];
    }
    return v;
}
// One attachment cleared within rect: a pass of its own whose load operation clears (the view's format converts the value, sRGB included)
static void clear_pass(VkImageView v, VkFormat f, int ds, int cd, int cs, VkRect2D rect, uint32_t layers, VkClearValue val) {
    pass_end();
    VkCommandBuffer cb = vks_cb();
    barrier_all(cb);
    VkRenderingAttachmentInfo a = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, v, VK_IMAGE_LAYOUT_GENERAL, 0, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_STORE_OP_STORE, val};
    VkRenderingAttachmentInfo d = a, s = a;
    d.loadOp = cd ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD; s.loadOp = cs ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    VkImageAspectFlags asp = vkf_aspect(f);
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, rect, layers ? layers : 1, 0, ds ? 0u : 1u, ds ? NULL : &a,
                          ds && (asp & VK_IMAGE_ASPECT_DEPTH_BIT) ? &d : NULL, ds && (asp & VK_IMAGE_ASPECT_STENCIL_BIT) ? &s : NULL};
    vkCmdBeginRendering(cb, &ri); vkCmdEndRendering(cb);
    g_st.clears++;
}
static int scissor_rect(uint32_t w, uint32_t h, VkRect2D *r) {   // the scissor within a w × h target; 0 when empty
    int64_t x0 = 0, y0 = 0, x1 = w, y1 = h;
    if (S->has_sc) { x0 = S->sc.x > 0 ? S->sc.x : 0; y0 = S->sc.y > 0 ? S->sc.y : 0; x1 = (int64_t)S->sc.x + S->sc.w; y1 = (int64_t)S->sc.y + S->sc.h; if (x1 > w) x1 = w; if (y1 > h) y1 = h; }
    if (x1 <= x0 || y1 <= y0) return 0;
    *r = (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
    return 1;
}
// Partial clears (M6 phase B; nvn_mtl.m clear_color_tri / clear_ds_tri): a colour clear that keeps some channels, or a stencil clear that keeps some bits, is a triangle
// over the target level within the scissor: the clear value (converted for integer targets) through the channel mask, or the masked stencil bits replaced, and depth
// written when it is cleared too (the viewport's depth range is the value). One pass per layer of the attachment
typedef struct { uint32_t fmt; uint8_t ds, kind, samples, cmask, dw, sm, pad[2]; } ckey;
static vk_hmap g_cpipe = {sizeof(ckey)};
static VkPipelineLayout g_cpl; static VkShaderModule g_cmod[4];   // vertex; fragment float / uint / int
static VkPipeline clear_pipe(const ckey *k) {
    VkPipeline p = (VkPipeline)vk_hm_get(&g_cpipe, k);
    if (p) return p;
    if (!g_cpl) {
        VkPushConstantRange pr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
        VkPipelineLayoutCreateInfo lci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL, 1, &pr};
        if (vkCreatePipelineLayout(g_vk.dev, &lci, NULL, &g_cpl) != VK_SUCCESS) return VK_NULL_HANDLE;
        const uint32_t *src[4] = {clear_vert_spv, clear_f_frag_spv, clear_u_frag_spv, clear_i_frag_spv};
        size_t sz[4] = {sizeof clear_vert_spv, sizeof clear_f_frag_spv, sizeof clear_u_frag_spv, sizeof clear_i_frag_spv};
        for (int i = 0; i < 4; i++) {
            VkShaderModuleCreateInfo mci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sz[i], src[i]};
            if (vk_module(&mci, &g_cmod[i], "internal-clear", "runtime") != VK_SUCCESS) return VK_NULL_HANDLE;
        }
    }
    VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_VERTEX_BIT, g_cmod[0], "main", NULL},
                                             {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_FRAGMENT_BIT, g_cmod[1 + (k->kind < 3 ? k->kind : 0)], "main", NULL}};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, NULL, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, NULL, 0, 1, NULL, 1, NULL};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, NULL, 0, VK_FALSE, VK_FALSE, VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE,
                                                 VK_FRONT_FACE_COUNTER_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, NULL, 0, (VkSampleCountFlagBits)(k->samples ? k->samples : 1)};
    VkStencilOpState so = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE, VK_COMPARE_OP_ALWAYS, 0xff, k->sm, 0};
    VkPipelineDepthStencilStateCreateInfo dss = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, NULL, 0, k->dw, k->dw, VK_COMPARE_OP_ALWAYS, VK_FALSE, k->sm != 0, so, so, 0, 1};
    VkPipelineColorBlendAttachmentState ba = {VK_FALSE, 0, 0, 0, 0, 0, 0, k->cmask};
    VkPipelineColorBlendStateCreateInfo cbs = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, NULL, 0, VK_FALSE, 0, k->ds ? 0u : 1u, &ba};
    VkDynamicState dyn[3] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_REFERENCE};
    VkPipelineDynamicStateCreateInfo dy = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, NULL, 0, 3, dyn};
    VkFormat cf = (VkFormat)k->fmt; VkImageAspectFlags asp = k->ds ? vkf_aspect(cf) : 0;
    VkPipelineRenderingCreateInfo ri = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, NULL, 0, k->ds ? 0u : 1u, k->ds ? NULL : &cf,
                                        asp & VK_IMAGE_ASPECT_DEPTH_BIT ? cf : VK_FORMAT_UNDEFINED, asp & VK_IMAGE_ASPECT_STENCIL_BIT ? cf : VK_FORMAT_UNDEFINED};
    VkGraphicsPipelineCreateInfo ci = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &ri, 0, k->ds ? 1u : 2u, st, &vi, &ia, NULL, &vp, &rs, &ms, &dss, &cbs, &dy, g_cpl,
                                       VK_NULL_HANDLE, 0, VK_NULL_HANDLE, -1};
    if (vkCreateGraphicsPipelines(g_vk.dev, g_vk.pcache, 1, &ci, NULL, &p) != VK_SUCCESS) { RED_ONCE(23, "the clear-draw pipeline for format %u could not be made: masked clears are lost", k->fmt); return VK_NULL_HANDLE; }
    vkpc_note_created(1); vk_hm_put(&g_cpipe, k, (void *)p);
    return p;
}
static void ds_forget(void);   // the draws' dynamic state cache (g_ds, below)
static void clear_draw(vkt_image *im, VkFormat f, uint32_t level, uint32_t l0, uint32_t nl, int ds, int kind, unsigned cmask, int dw, unsigned sm, VkRect2D rect,
                       const uint32_t v[4], float depth, uint32_t stencil) {
    ckey k; memset(&k, 0, sizeof k); k.fmt = (uint32_t)f; k.ds = (uint8_t)ds; k.kind = (uint8_t)kind; k.samples = (uint8_t)im->samples; k.cmask = (uint8_t)cmask; k.dw = (uint8_t)dw; k.sm = (uint8_t)sm;
    VkPipeline p = clear_pipe(&k); if (!p) return;
    pass_end();
    VkCommandBuffer cb = vks_cb();
    barrier_all(cb);
    uint32_t w = mipd(im->w, level), h = mipd(im->h, level);
    VkImageAspectFlags asp = ds ? vkf_aspect(f) : VK_IMAGE_ASPECT_COLOR_BIT;
    for (uint32_t layer = 0; layer < (nl ? nl : 1); layer++) {
        VkImageView view = vkt_view_raw_(im, f, VK_IMAGE_VIEW_TYPE_2D, level, 1, l0 + layer, 1, ds ? im->aspect : VK_IMAGE_ASPECT_COLOR_BIT, NULL);
        if (!view) continue;
        VkRenderingAttachmentInfo a = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, NULL, view, VK_IMAGE_LAYOUT_GENERAL, 0, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED,
                                       VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, {{{0}}}};
        VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, NULL, 0, rect, 1, 0, ds ? 0u : 1u, ds ? NULL : &a, ds && (asp & VK_IMAGE_ASPECT_DEPTH_BIT) ? &a : NULL,
                              ds && (asp & VK_IMAGE_ASPECT_STENCIL_BIT) ? &a : NULL};
        vkCmdBeginRendering(cb, &ri);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p); g_last_pipe = VK_NULL_HANDLE; ds_forget();   // its static state leaves the draws' dynamic state undefined
        VkViewport vp = {0, 0, (float)w, (float)h, ds ? depth : 0, ds ? depth : 1};
        vkCmdSetViewport(cb, 0, 1, &vp); vkCmdSetScissor(cb, 0, 1, &rect);
        vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, stencil & 0xff);
        uint32_t pc[4] = {0}; if (v) memcpy(pc, v, 16);
        vkCmdPushConstants(cb, g_cpl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, pc);
        vkCmdDraw(cb, 3, 1, 0, 0);
        vkCmdEndRendering(cb);
    }
    g_st.clears++;
}
static void clear_color(const p_clear_color *c) {
    if (S->render_skip) { if (g_pd) PD("colour clear skipped by conditional rendering, slot %d", c->index); return; }
    if (!P.valid) pass_targets();
    int i = c->index;
    if (i < 0 || i >= 8 || !P.cv[i]) { if (i < 0 || i >= S->nrt || !S->has_ct[i]) RED_ONCE(20, "colour clear of target %d, which is not set", i); return; }
    unsigned ch = fmt_channels(P.cf[i]);
    if ((c->mask & ch) == 0) return;
    VkRect2D r; if (!scissor_rect(P.cw[i], P.ch[i], &r)) return;
    if ((c->mask & ch) != ch) {   // some channels kept: a clear draw through the channel mask (nvn_mtl.m clear_color_tri)
        int kind = fmt_int_kind(P.cf[i]); uint32_t v[4];
        for (int k = 0; k < 4; k++) { float f = c->rgba[k]; int32_t si;   // integer targets take the value converted, as a clear does
            if (kind == 1) v[k] = f > 0 ? (f < 4294967040.f ? (uint32_t)f : 0xffffffffu) : 0;
            else if (kind == 2) { si = !(f == f) ? 0 : f >= 2147483520.f ? INT32_MAX : f <= -2147483648.f ? INT32_MIN : (int32_t)f; memcpy(&v[k], &si, 4); }
            else memcpy(&v[k], &f, 4); }
        { static int n; if (!n++) VKLOG("colour clear with channel mask %#x: drawn through the mask (the first of its kind)", c->mask); }
        clear_draw(P.ci[i], P.cf[i], P.clv[i], P.cl0[i], P.cnl[i], 0, kind, c->mask & ch, 0, 0, r, v, 0, 0);
    } else clear_pass(P.cv[i], P.cf[i], 0, 0, 0, r, P.cnl[i], clear_color_value(P.cf[i], c->rgba, NULL, 0));
    PD("clear c%d ser%llu to %g,%g,%g,%g mask %#x rect %d,%d %ux%u", i, (unsigned long long)P.ci[i]->serial, c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3], c->mask, r.offset.x, r.offset.y, r.extent.width, r.extent.height);
}
static void clear_ds(const p_clear_ds *c) {
    if (S->render_skip) { if (g_pd) PD("depth-stencil clear skipped by conditional rendering"); return; }
    if (!P.valid) pass_targets();
    if (!P.dv) return;
    int cd = c->depth_mask && P.hd; unsigned sm = (unsigned)c->stencil_mask & 0xff; int cs = sm && P.hs;
    if (!cd && !cs) return;
    VkRect2D r; if (!scissor_rect(P.dw, P.dh, &r)) return;
    VkClearValue v; v.depthStencil.depth = c->depth < 0 ? 0 : c->depth > 1 ? 1 : c->depth; v.depthStencil.stencil = (uint32_t)c->stencil & 0xff;
    if (cs && sm != 0xff) {   // some stencil bits kept: a clear draw replacing the masked bits, and depth when it is cleared too (nvn_mtl.m clear_ds_tri)
        { static int n; if (!n++) VKLOG("stencil clear with mask %#x: drawn through the mask (the first of its kind)", sm); }
        clear_draw(P.di, P.df, P.dlv, P.dl0, P.dnl, 1, 0, 0, cd, sm, r, NULL, v.depthStencil.depth, v.depthStencil.stencil);
    } else clear_pass(P.dv, P.df, 1, cd, cs, r, P.dnl, v);
    PD("clear depth %d (%g) stencil %d (%u)", cd, c->depth, cs, (unsigned)c->stencil);
}

// ---------- Descriptors (nvn_mtl.m encode_args_) ----------
static inline uint64_t tex_slot_h(int st, unsigned reg) {   // nvn_mtl.m tex_slot_h: combined slots past 31 share their c2 word with an image slot
    return reg >= 32 && reg - 32 < DMSH_C2_IMG_SLOTS && S->img[st][reg - 32] ? S->img[st][reg - 32] : reg < NTEX ? S->tex[st][reg] : 0;
}
static VkSampler samp_get(int sid) {
    samp_par p;
    if (sid <= 0 || !vkt_samp_copy(sid, &p)) return R0.samp;
    VkSampler s = vkt_sampler(&p);
    return s ? s : R0.samp;
}
static int nvn_target_of(VkImageViewType vt, int ms) {
    switch (vt) {
    case VK_IMAGE_VIEW_TYPE_1D: return NVN_TEXTURE_TARGET_1D; case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return NVN_TEXTURE_TARGET_1D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_3D: return NVN_TEXTURE_TARGET_3D; case VK_IMAGE_VIEW_TYPE_CUBE: return NVN_TEXTURE_TARGET_CUBEMAP; case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return NVN_TEXTURE_TARGET_CUBEMAP_ARRAY;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return ms ? NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY : NVN_TEXTURE_TARGET_2D_ARRAY;
    default: return ms ? NVN_TEXTURE_TARGET_2D_MULTISAMPLE : NVN_TEXTURE_TARGET_2D;
    }
}
static VkImageViewType view_type_of_target(int target, VkImageViewType dflt) {
    switch (target) {
    case NVN_TEXTURE_TARGET_1D: return VK_IMAGE_VIEW_TYPE_1D; case NVN_TEXTURE_TARGET_1D_ARRAY: return VK_IMAGE_VIEW_TYPE_1D_ARRAY; case NVN_TEXTURE_TARGET_3D: return VK_IMAGE_VIEW_TYPE_3D;
    case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case NVN_TEXTURE_TARGET_CUBEMAP: return VK_IMAGE_VIEW_TYPE_CUBE; case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    case NVN_TEXTURE_TARGET_2D: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE: case NVN_TEXTURE_TARGET_RECTANGLE: return VK_IMAGE_VIEW_TYPE_2D;
    }
    return dflt;
}
// The sampled view of texture ID tid as binding b declares it (a 2D texture read as a 2D array and so on: the view's type is the shader's, nvn_mtl.m tex_fit)
// A format of the shader's numeric class (VKS_CT_*) with the same bits: 32-bit components between float, uint and int, 8- and 16-bit integers between uint and
// int (Maxwell hands the shader the register bits; Vulkan wants the view's class to match the declared one). -1 when it already matches or no format keeps the bits
int vkr_fmt_as_ct(int f, int ct) {
    static const int base[4] = {NVN_FORMAT_R8UI, NVN_FORMAT_RG8UI, NVN_FORMAT_RGB8UI, NVN_FORMAT_RGBA8UI};   // 8UI 8I 16F 16 16SN 16UI 16I 32F 32UI 32I
    for (int g = 0; g < 4; g++) {
        int i = f - base[g]; if (i < 0 || i > 9) continue;
        int cls = i == 0 || i == 5 || i == 8 ? VKS_CT_UINT : i == 1 || i == 6 || i == 9 ? VKS_CT_INT : VKS_CT_FLOAT;
        if (cls == ct) return -1;
        if (i >= 7) return base[g] + (ct == VKS_CT_FLOAT ? 7 : ct == VKS_CT_UINT ? 8 : 9);
        if (ct != VKS_CT_FLOAT && cls != VKS_CT_FLOAT) return base[g] + (i <= 1 ? 0 : 5) + (ct == VKS_CT_INT);
        return -1;
    }
    return -1;
}
static const vks_shader *g_cur_sh;   // the shader whose descriptor set write_set is filling (for the messages below)
// Depth comparison on a colour texture: the game binds its default 1x1 / 8x8 black and white textures to the shadow samplers of lights without shadows. Vulkan
// leaves that undefined (VUID-vkCmdDrawIndexed-None-06479, in gameplay); the GPU the game was made for compares with the texel's red channel. A 1x1 depth image
// holding the red value of the texture's first texel stands in, one per value (M6), viewed as the shader declares it (M6 phase B: a 2D view under an arrayed or cube
// declaration was VUID-vkCmdDrawIndexed-viewType-07752): six cube-compatible layers, views 2D, 2D array, cube, cube array
typedef struct { float v; VkImage img; vk_alloc mem; VkImageView view[4]; } dcmp_twin;
static dcmp_twin g_dct[8]; static int g_ndct;
static vk_hmap g_dct_of = {8};   // texture serial → its value + 1 (0: not read yet)
static int read_u32(uint64_t a, uint32_t *v);
static VkImageView dcmp_view(vkt_image *im, VkImageViewType vt) {
    int vi = vt == VK_IMAGE_VIEW_TYPE_2D ? 0 : vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY ? 1 : vt == VK_IMAGE_VIEW_TYPE_CUBE ? 2 : vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY && g_vk.f10.imageCubeArray ? 3 : -1;
    if (vi < 0) return VK_NULL_HANDLE;
    uint64_t k = im->serial; uintptr_t got = (uintptr_t)vk_hm_get(&g_dct_of, &k);
    float v = 1.0f;
    if (got) memcpy(&v, &(uint32_t){(uint32_t)(got - 1)}, 4);
    else {
        // the first texel: from storage for a texture read from guest memory (its upload may still be in this submission's command buffers: a readback now
        // would see the image before it), else read back; texel (0, 0) is the first bytes in the block-linear layout too
        uint8_t px[16] = {0}; NVNcopyRegion rr = {0, 0, 0, 1, 1, 1}; uint8_t rgba[4] = {255, 0, 0, 0}; int got = -1;
        if (!(im->fi->flags & VKF_COMPRESSED) && im->fi->vbpb <= 16) {
            if (im->cls == VKT_STORAGE && im->mem) { uint32_t w[4]; got = 0; for (unsigned q = 0; q < (im->fi->vbpb + 3) / 4 && !got; q++) got = !read_u32((uint64_t)(uintptr_t)im->mem + 4 * q, &w[q]); if (!got) memcpy(px, w, im->fi->vbpb); got = -got; }
            else got = vkt_readback_region(im, 0, 0, &rr, px, im->fi->vbpb);
        }
        if (!got && !vks_to_rgba8(im->fi->vk, px, 1, rgba)) v = rgba[0] / 255.0f;
        uint32_t bits; memcpy(&bits, &v, 4); vk_hm_put(&g_dct_of, &k, (void *)(uintptr_t)((uintptr_t)bits + 1));
        VKLOG("depth comparison on a colour texture (ser%llu %s %ux%u; shader %s): compared with its red channel, %.3f", (unsigned long long)im->serial, im->fi->name, im->w, im->h,
              g_cur_sh ? g_cur_sh->key : "?", v);
    }
    for (int i = 0; i < g_ndct; i++) if (g_dct[i].v == v) return g_dct[i].view[vi];
    if (g_ndct == 8) return VK_NULL_HANDLE;
    dcmp_twin *t = &g_dct[g_ndct];
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, VK_IMAGE_TYPE_2D, VK_FORMAT_D32_SFLOAT, {1, 1, 1}, 1, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if (vkCreateImage(g_vk.dev, &ci, NULL, &t->img) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vk.dev, t->img, &mr);
    if (vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &t->mem)) { vkDestroyImage(g_vk.dev, t->img, NULL); return VK_NULL_HANDLE; }
    vkBindImageMemory(g_vk.dev, t->img, t->mem.mem, t->mem.off);
    static const VkImageViewType vts[4] = {VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY};
    for (int i = 0; i < 4; i++) {
        if (i == 3 && !g_vk.f10.imageCubeArray) continue;
        VkImageViewCreateInfo vci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, t->img, vts[i], VK_FORMAT_D32_SFLOAT, {0}, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, i < 2 ? 1u : 6u}};
        if (vkCreateImageView(g_vk.dev, &vci, NULL, &t->view[i]) != VK_SUCCESS) return VK_NULL_HANDLE;
    }
    VkCommandBuffer cb = vk_once_begin(); VkImageSubresourceRange sr = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 6};
    vk_image_barrier(cb, t->img, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    vkCmdClearDepthStencilImage(cb, t->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &(VkClearDepthStencilValue){v, 0}, 1, &sr);
    vk_image_barrier(cb, t->img, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    if (vk_once_end(cb)) return VK_NULL_HANDLE;
    t->v = v; g_ndct++;
    return t->view[vi];
}
static VkImageView tex_view(int tid, const vks_bind *b, VkImageLayout *layout, vkt_image **out_im) {
    vkt_reg r; *out_im = NULL;
    int pool_n = vkt_pool_n();
    if (tid <= 0 || (pool_n && tid >= pool_n) || !vkt_reg_copy(tid, &r) || r.gone) return VK_NULL_HANDLE;
    if (r.t.d.target == NVN_TEXTURE_TARGET_BUFFER) { RED_ONCE(30, "a buffer texture in a slot a shader samples as an image: it reads 0 (nvn_mtl.m: dummy)"); return VK_NULL_HANDLE; }
    vkt_image *im = vkt_resolve(&r.t);
    if (!im) return VK_NULL_HANDLE;
    view_t v = r.has_view ? r.v : (view_t){0}; int use = r.has_view;
    if (!v.set_format && im->d.format != r.t.d.format) { v.set_format = 1; v.format = (uint16_t)r.t.d.format; use = 1; }   // the texture is the image's twin in another format (vk_tex.c key_of)
    VkImageViewType have = v.set_target ? view_type_of_target(v.target, im->vtype) : im->vtype;
    if (have != b->vtype) { v.set_target = 1; v.target = (uint8_t)nvn_target_of(b->vtype, b->ms); use = 1; }
    if ((have == VK_IMAGE_VIEW_TYPE_3D) != (b->vtype == VK_IMAGE_VIEW_TYPE_3D)) {
        RED_ONCE(31, "a 3D texture read as another type, or the reverse (texture ID %d, ser%llu %ux%ux%u view type %d; shader %s binding %u declares view type %d): it reads 0", tid,
                 (unsigned long long)im->serial, im->w, im->h, im->depth, (int)have, g_cur_sh ? g_cur_sh->key : "?", b->binding, (int)b->vtype);
        return VK_NULL_HANDLE; }
    if (b->depth && !(im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) {
        VkImageView dv = dcmp_view(im, b->vtype);
        if (dv) { *layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; return dv; }
    }
    int cf = vkr_fmt_as_ct(v.set_format ? v.format : r.t.d.format, b->ct);
    if (cf >= 0 && !(im->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) { v.set_format = 1; v.format = (uint16_t)cf; use = 1; }
    *layout = im->rest; *out_im = im; td_note(im);
    return vkt_view(im, use ? &v : NULL, r.t.set_swz, r.t.swz);
}
// Per-submission memo of tex_view and samp_get: a slot's registration, its image and view are the same for every draw of a submission
// unless the game registers or releases something meanwhile (g_vkt_reg_gen). About 15,000 texture bindings a race frame went through
// the registration copy, image resolve and view lookup each time. NXRT_VK_TEX_MEMO=0 turns it off
extern volatile LONG g_vkt_reg_gen;
typedef struct { uint64_t key; uint32_t sub; LONG gen; VkImageView view; VkImageLayout layout; vkt_image *im; } tex_memo_e;
static tex_memo_e g_tmemo[8192]; static struct { uint32_t key; uint32_t sub; LONG gen; VkSampler s; } g_smemo[2048]; static uint32_t g_memo_sub = 1;
static int memo_on(void) { static int v = -1; if (v < 0) v = vkc_int("tex_memo", 1); return v; }
static VkImageView tex_view_m(int tid, const vks_bind *b, VkImageLayout *layout, vkt_image **out_im) {
    if (!memo_on() || tid <= 0) return tex_view(tid, b, layout, out_im);
    uint64_t key = (uint64_t)(uint32_t)tid | (uint64_t)(b->vtype & 0xff) << 32 | (uint64_t)(b->ms & 1) << 40 | (uint64_t)(b->depth & 1) << 41 | (uint64_t)(b->ct & 0xff) << 42;
    LONG gen = g_vkt_reg_gen; uint64_t h = key * 0x9E3779B97F4A7C15ull; tex_memo_e *e = &g_tmemo[h >> 51];
    if (e->key == key && e->sub == g_memo_sub && e->gen == gen) { *layout = e->layout; *out_im = e->im; return e->view; }
    VkImageView v = tex_view(tid, b, layout, out_im);
    if (g_vkt_reg_gen == gen) { e->key = key; e->sub = g_memo_sub; e->gen = gen; e->view = v; e->layout = *layout; e->im = *out_im; }
    return v;
}
static VkSampler samp_get_m(int sid) {
    if (!memo_on() || sid <= 0) return samp_get(sid);
    LONG gen = g_vkt_reg_gen; uint32_t i = ((uint32_t)sid * 2654435761u) >> 21;
    if (g_smemo[i].key == (uint32_t)sid && g_smemo[i].sub == g_memo_sub && g_smemo[i].gen == gen) return g_smemo[i].s;
    VkSampler s = samp_get(sid);
    if (g_vkt_reg_gen == gen) { g_smemo[i].key = (uint32_t)sid; g_smemo[i].sub = g_memo_sub; g_smemo[i].gen = gen; g_smemo[i].s = s; }
    return s;
}
static VkImageView img_view(int tid, const vks_bind *b) {   // a storage image: one level, identity swizzle, a format storage allows
    vkt_reg r;
    if (tid <= 0 || !vkt_reg_copy(tid, &r) || r.gone || r.t.d.target == NVN_TEXTURE_TARGET_BUFFER) return VK_NULL_HANDLE;
    { void nvn_pool_gpu_written(const void *pool); nvn_pool_gpu_written(r.t.pool); }
    vkt_image *im = vkt_resolve(&r.t);
    if (!im) return VK_NULL_HANDLE;
    if (!(im->usage & VK_IMAGE_USAGE_STORAGE_BIT)) { RED_ONCE(32, "image binding of texture ID %d (format %#x) whose image has no storage use: writes are lost", tid, r.t.d.format); return VK_NULL_HANDLE; }
    uint32_t level = r.has_view && r.v.set_levels ? r.v.base_level : 0, b0 = r.has_view && r.v.set_layers ? r.v.min_layer : 0;
    VkFormat f = vkf_unorm_of(vkf_view_format(r.t.d.format, r.has_view && r.v.set_format ? r.v.format : r.t.d.format));
    VkImageViewType vt = b->vtype;
    if (vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) vt = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    im->content = 1; td_note(im);
    return vkt_view_raw_(im, f, vt, level, 1, b0, vt == VK_IMAGE_VIEW_TYPE_2D || vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_3D ? 1 : 0, VK_IMAGE_ASPECT_COLOR_BIT, NULL);
}
static VkBufferView tbuf_view(int tid, int ct) {   // a buffer texture: a texel buffer view over the pool mirror, in the shader's numeric class when the bits allow
    vkt_reg r;
    if (tid <= 0 || !vkt_reg_copy(tid, &r) || r.gone || r.t.d.target != NVN_TEXTURE_TARGET_BUFFER || !r.t.mem) return VK_NULL_HANDLE;
    int cf = vkr_fmt_as_ct(r.t.d.format, ct);
    const vkf_info *fi = vkf_get(cf >= 0 ? cf : r.t.d.format);
    if (!fi || !fi->vk || fi->conv != VKF_COPY) { RED_ONCE(33, "buffer texture format %#x has no texel buffer format", r.t.d.format); return VK_NULL_HANDLE; }
    uint64_t n = (uint64_t)r.t.d.width * fi->bpb; if (!n) return VK_NULL_HANDLE;
    if (n / fi->bpb > g_vk.props.limits.maxTexelBufferElements) n = (uint64_t)g_vk.props.limits.maxTexelBufferElements * fi->bpb;
    VkBuffer b; VkDeviceSize o;
    if (!vkm_use((uint64_t)(uintptr_t)r.t.mem, n, &b, &o)) return VK_NULL_HANDLE;
    if (o % g_vk.props.limits.minTexelBufferOffsetAlignment) { RED_ONCE(34, "buffer texture at an offset Vulkan cannot view (%#llx)", (unsigned long long)o); return VK_NULL_HANDLE; }
    VkBufferViewCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, NULL, 0, b, fi->vk, o, n};
    VkBufferView v = VK_NULL_HANDLE;
    if (vkCreateBufferView(g_vk.dev, &ci, NULL, &v) == VK_SUCCESS) view_keep(v);
    return v;
}
static int c0_ssbo_base(int st) { return st == 0 ? DMSH_C0_SSBO_VS : st == 1 ? DMSH_C0_SSBO_PS : st == 2 ? DMSH_C0_SSBO_GS : st == 5 ? DMSH_C0_SSBO_CS : -1; }
static uint32_t g_draw_binst, g_dispatch_dim[3];
static void put_range(uint8_t *d, unsigned off, uint64_t a, uint64_t n) { if (off + 12 <= DMSH_DRV_SIZE) { uint32_t s = (uint32_t)n; memcpy(d + off, &a, 8); memcpy(d + off + 8, &s, 4); } }
// A driver segment as the shader reads it (dmsh_abi.h): c0 the buffer tables and per-draw words, c2 the handle tables
static void drv_fill(uint8_t *d, const vks_shader *sh, int st, int bank) {
    memset(d, 0, DMSH_DRV_SIZE);
    if (bank == 0) {
        if (sh->s2cs) {   // the Switch 2 compute bank, packed per program: [0x10][uniforms 16 B each][storage 16 B each][textures 8 B each][images 8 B each]
            for (int r = 0; r < 16 && 0x10u + 16u * r + 16 <= sh->s2_ssbo_base; r++) put_range(d, 0x10 + 16 * r, S->ubo[5][r].addr, S->ubo[5][r].size);
            for (int r = 0; r < 16 && sh->s2_ssbo_base + 16u * r + 16 <= sh->s2_tex_base; r++) put_range(d, sh->s2_ssbo_base + 16 * r, S->ssbo[5][r].addr, S->ssbo[5][r].size);
            for (uint32_t r = 0; r < sh->s2_ntex && sh->s2_tex_base + 8 * r + 4 <= DMSH_DRV_SIZE; r++) { uint32_t h = (uint32_t)S->tex[5][r]; memcpy(d + sh->s2_tex_base + 8 * r, &h, 4); }
            for (uint32_t r = 0; r < 8 && sh->s2_tex_base + 8 * (sh->s2_ntex + r) + 4 <= DMSH_DRV_SIZE; r++) { uint32_t h = (uint32_t)S->img[5][r]; memcpy(d + sh->s2_tex_base + 8 * (sh->s2_ntex + r), &h, 4); }
            return;
        }
        int base = c0_ssbo_base(st);
        if (base >= 0) for (int r = 0; r < DMSH_C0_SSBO_SLOTS; r++) put_range(d, base + DMSH_C0_ROW * r, S->ssbo[st][r].addr, S->ssbo[st][r].size);
        if (st != 5) {
            const float max_point = 2047.0f; memcpy(d + DMSH_C0_POINT_SIZE, &max_point, 4);   // Maxwell's point size limit (nvn_mtl.m)
            memcpy(d + DMSH_C0_BASE_INSTANCE, &g_draw_binst, 4);                              // the draw's base instance, which GLSLC adds to the instance ID
        }
        if (st == 1) { d[DMSH_C0_SAMPLE_TABLE] = 0x88; }                                     // one sample at the pixel's centre (nvn_mtl.m c0_sample_rows)
        if (st == 5) {
            memcpy(d + DMSH_C0_DISPATCH, g_dispatch_dim, 12); memcpy(d + DMSH_C0_DISPATCH_NVN2, g_dispatch_dim, 12);
            for (int r = 0; r < DMSH_C0_CS_UBO_SLOTS; r++) put_range(d, DMSH_C0_CS_UBO + DMSH_C0_ROW * r, S->ubo[5][r].addr, S->ubo[5][r].size);
        }
    } else {
        const uint64_t *tx = S->tex[st], *im = S->img[st];
        for (int r = 0; r < DMSH_C2_IMG_SLOTS; r++) { uint32_t v = (uint32_t)im[r]; memcpy(d + DMSH_C2_IMG + DMSH_C2_ROW * r, &v, 4); }
        for (int r = 0; r < DMSH_C2_TEX_SLOTS; r++) {
            if (r >= 32 && (!tx[r] || (r - 32 < DMSH_C2_IMG_SLOTS && im[r - 32]))) continue;
            uint32_t v = (uint32_t)tx[r]; memcpy(d + DMSH_C2_TEX + DMSH_C2_ROW * r, &v, 4);
        }
        for (int r = 0; r < DMSH_C2_SEP_SLOTS; r++) { uint32_t v = (uint32_t)tx[NVN_SEPTEX_SLOT0 + r]; memcpy(d + DMSH_C2_SEPTEX + DMSH_C2_ROW * r, &v, 4); }
        for (int r = 0; r < DMSH_C2_SEP_SLOTS; r++) { uint32_t v = (uint32_t)S->samp[st][r]; memcpy(d + DMSH_C2_SAMP + DMSH_C2_ROW * r, &v, 4); }
    }
}
// A uniform range: the pool mirror at its offset (uploaded as needed), else a copy (offsets Vulkan cannot bind), else zeros
static VkDescriptorBufferInfo ubo_info(const p_bind_buf *u) {
    VkDescriptorBufferInfo zero = {R0.zero, 0, ZERO_SIZE};
    if (!u->addr || !u->size) return zero;
    VkDeviceSize n = u->size > g_vk.props.limits.maxUniformBufferRange ? g_vk.props.limits.maxUniformBufferRange : u->size;
    VkBuffer b; VkDeviceSize o;
    if (!vkm_use(u->addr, n, &b, &o)) { RED_ONCE(40, "a uniform buffer at %#llx is in no memory pool: it reads zeros", (unsigned long long)u->addr); return zero; }
    if (o % g_vk.props.limits.minUniformBufferOffsetAlignment) {
        RED_ONCE(41, "a uniform buffer at an address Vulkan cannot bind (%#llx): read from a copy of guest memory", (unsigned long long)u->addr);
        uint8_t *p = arena(n, g_vk.props.limits.minUniformBufferOffsetAlignment, &b, &o);
        if (!p) return zero;
        memcpy(p, (const void *)(uintptr_t)u->addr, (size_t)n);
    }
    return (VkDescriptorBufferInfo){b, o, n};
}
// A stage's descriptor set serves the next draw too when nothing it is made of has changed (gamectl set set_reuse 0, NXRT_VK_SET_REUSE=0: a new set every draw).
// g_bver[st] moves with every bind record that changes stage st's uniform, storage, texture, sampler or image slots or the push constants; the set is also
// made again for another shader, another submission slot (its descriptor pool goes with the slot), another queue's state, a texture (de)registration,
// a memory pool coming or going, or another base instance (c0 holds it). Shaders whose set depends on more never reuse: storage buffers or images the
// shader writes (each use notes a write-back), storage buffers whose address is read from guest memory (VKB_SBUF_UBO), and anything unknown
static uint64_t g_bver[6];
static uint64_t g_n_set_reused;
static struct { const vks_shader *sh; const void *rs; uint64_t ver, seq; LONG regen; uint32_t pgen, binst; VkDescriptorSet ds; } g_lset[6];
static int set_reusable(const vks_shader *sh) {
    if (!sh->ds_reuse) {
        int ok = 1;
        for (int i = 0; i < sh->nb && ok; i++) { const vks_bind *b = &sh->b[i];
            if (b->rw || b->kind == VKB_IMG || b->kind == VKB_SBUF_UBO || b->kind == VKB_UNKNOWN || b->count != 1) ok = 0; }
        ((vks_shader *)sh)->ds_reuse = ok ? 1 : 2;
    }
    return sh->ds_reuse == 1;
}
// Descriptor set of stage st's shader, filled from the current bindings
static int g_dl_on = -1; static uint64_t g_dl_frame;   // NXRT_VK_DRAWLOG (draw_log below): the frame of the last present
static char g_dl[8][17]; static int g_ndl;
static int dl_find(const char *key) {   // which NXRT_VK_DRAWLOG key this is (-1: none); reads the list the first time
    if (g_dl_on < 0) { g_dl_on = 0; const char *e = getenv("NXRT_VK_DRAWLOG");
        while (e && *e && g_ndl < 8) { if (strlen(e) >= 16) { memcpy(g_dl[g_ndl], e, 16); g_dl[g_ndl][16] = 0; g_ndl++; } e = strchr(e, ','); if (e) e++; }
        g_dl_on = g_ndl > 0; }
    if (key) for (int k = 0; k < g_ndl; k++) if (!strcmp(key, g_dl[k])) return k;
    return -1;
}
static char g_vsnote[256]; static size_t g_vsnote_n;   // the vertex shader's bindings of the draw being recorded (full checkpoints): vk_ck_note
static void vsnote(const char *fmt, ...) {
    if (g_vsnote_n >= sizeof g_vsnote - 1) return;
    va_list ap; va_start(ap, fmt); int k = vsnprintf(g_vsnote + g_vsnote_n, sizeof g_vsnote - g_vsnote_n, fmt, ap); va_end(ap);
    if (k > 0) g_vsnote_n += (size_t)k; if (g_vsnote_n > sizeof g_vsnote - 1) g_vsnote_n = sizeof g_vsnote - 1;
}
// NXRT_VK_PHASES=1: where the replay's time goes, per frame, beside the "replay up to frame" line (exec = all records; draws, of them
// descriptor sets; dispatches; the commits in the middle of a submission)
enum { PH_EXEC, PH_DRAW, PH_WSET, PH_DISP, PH_SPLIT, PH_WS_ALLOC, PH_WS_UBO, PH_WS_SBUF, PH_WS_TEX, PH_WS_UPD, PH_N };
static uint64_t g_ph_ns[PH_N]; static int g_ph_on = -1;
static inline int ph_on(void) { if (g_ph_on < 0) g_ph_on = vkc_int("phases", 0); return g_ph_on; }
// in TSC cycles (a few ns a read; QueryPerformanceCounter's ~30 ns over ~100k reads a frame made the phases 3-4 ms slower than the replay they measured), ms at the report
#define PH_T() __builtin_ia32_rdtsc()
#define PH0() uint64_t ph0_ = ph_on() ? PH_T() : 0
#define PH1(i) do { if (ph0_) g_ph_ns[i] += PH_T() - ph0_; } while (0)
uint64_t vkm_use_cycles(void);   // vk_mem.c: cycles inside vkm_use with the phases on (the memory sync of every binding)
static VkDescriptorSet write_set_(const vks_shader *sh, int st, int *swrites);
extern volatile LONG g_vkt_reg_gen;
static VkDescriptorSet write_set(const vks_shader *sh, int st, int *swrites) {
    PH0();
    static int on = -1; if (on < 0) on = vkc_int("set_reuse", 1);
    uint64_t seq = vks_take_seq(); LONG regen = g_vkt_reg_gen; uint32_t pgen = vkm_pool_gen(), binst = sh->drv_read[0] && st != 5 ? g_draw_binst : 0;
    if (on && st >= 0 && st < 6 && g_vk.has.checkpoints < 2 && !g_dl_on && set_reusable(sh)) {
        if (g_lset[st].ds && g_lset[st].sh == sh && g_lset[st].rs == S && g_lset[st].ver == g_bver[st] && g_lset[st].seq == seq && g_lset[st].regen == regen && g_lset[st].pgen == pgen && g_lset[st].binst == binst) {
            g_n_set_reused++; PH1(PH_WSET); return g_lset[st].ds; }
        VkDescriptorSet r = write_set_(sh, st, swrites);
        g_lset[st].sh = sh; g_lset[st].rs = S; g_lset[st].ver = g_bver[st]; g_lset[st].seq = seq; g_lset[st].regen = regen; g_lset[st].pgen = pgen; g_lset[st].binst = binst; g_lset[st].ds = r;
        PH1(PH_WSET); return r;
    }
    VkDescriptorSet r = write_set_(sh, st, swrites); PH1(PH_WSET); return r;
}
static VkDescriptorSet write_set_(const vks_shader *sh, int st, int *swrites) {
    PH0(); VkDescriptorSet ds = dset_alloc(sh->dsl); PH1(PH_WS_ALLOC);
    if (!ds) return VK_NULL_HANDLE;
    int note = st == 0 && (g_vk.has.checkpoints >= 2 || dl_find(sh->key) >= 0);   // the draw log notes only the shaders it watches
    if (note) { g_vsnote_n = 0; g_vsnote[0] = 0; }
    g_cur_sh = sh;
    if (sh->nb > MAXB) { RED_ONCE(42, "shader %s has %d bindings (more than %d)", sh->key, sh->nb, MAXB); return VK_NULL_HANDLE; }
    VkWriteDescriptorSet w[MAXB]; VkDescriptorBufferInfo bi[MAXB]; VkDescriptorImageInfo ii[MAXB]; VkBufferView bv[MAXB];
    int nw = 0;
    for (int i = 0; i < sh->nb; i++) {
        const vks_bind *b = &sh->b[i];
        VkWriteDescriptorSet *x = &w[nw];
        *x = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, b->binding, 0, 1, b->type, NULL, NULL, NULL};
        if (b->count != 1) { RED_ONCE(43, "shader %s: an array binding (%u elements) is not replayed: left unwritten", sh->key, b->count); continue; }
        uint64_t phb_ = ph_on() ? PH_T() : 0; int phk_ = b->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ? PH_WS_UBO : b->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? PH_WS_SBUF : PH_WS_TEX;
        switch (b->type) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: {
            VkDescriptorBufferInfo *o = &bi[nw]; *o = (VkDescriptorBufferInfo){R0.zero, 0, ZERO_SIZE};
            if (b->kind == VKB_SUPPORT) *o = (VkDescriptorBufferInfo){R0.support, 0, SUPPORT_SIZE};
            else if (b->kind == VKB_CBUF && b->reg == DMSH_PUSH_BANK - 3) {
                VkBuffer ab; VkDeviceSize ao; uint8_t *d = arena(DMSH_PUSH_SIZE, g_vk.props.limits.minUniformBufferOffsetAlignment, &ab, &ao);
                if (d) { memcpy(d, S->push[st == 5], DMSH_PUSH_SIZE); *o = (VkDescriptorBufferInfo){ab, ao, DMSH_PUSH_SIZE}; }
            }
            else if (b->kind == VKB_CBUF) { if (b->reg >= 0 && b->reg < 16) { *o = ubo_info(&S->ubo[st][b->reg]);
                if (note) vsnote("c%d %llx+%llx%s ", b->reg, (unsigned long long)S->ubo[st][b->reg].addr, (unsigned long long)S->ubo[st][b->reg].size, o->buffer == R0.zero ? "(zeros)" : ""); } }
            else if (b->kind == VKB_C1) { if (sh->c1buf) *o = (VkDescriptorBufferInfo){sh->c1buf, sh->c1off, (sh->c1_len + 15) & ~15u}; }
            else if (b->kind == VKB_DRV && (b->bank == 0 || b->bank == 2) && sh->drv_read[b->bank]) {
                VkBuffer ab; VkDeviceSize ao; uint8_t *d = arena(DMSH_DRV_SIZE, g_vk.props.limits.minUniformBufferOffsetAlignment, &ab, &ao);
                if (d) { drv_fill(d, sh, st, b->bank); *o = (VkDescriptorBufferInfo){ab, ao, DMSH_DRV_SIZE}; }
            }
            x->pBufferInfo = o; break; }
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
            VkDescriptorBufferInfo *o = &bi[nw]; *o = (VkDescriptorBufferInfo){R0.sink, 0, ZERO_SIZE};
            const p_bind_buf *u = b->kind == VKB_SBUF && b->reg >= 0 && b->reg < 16 ? &S->ssbo[st][b->reg] : NULL;
            if (b->kind == VKB_UBOP && b->reg >= 0 && b->reg < 16) u = &S->ubo[st][b->reg];   // compute: a uniform slot reached through its address in c0 (DMSH_C0_CS_UBO)
            p_bind_buf ptr_;
            if (b->kind == VKB_SBUF_UBO && b->reg >= 0 && b->reg < 16 && S->ubo[st][b->reg].addr && (uint64_t)b->c0_off + 12 <= S->ubo[st][b->reg].size) {   // address and size kept in the game's own uniform buffer
                uint32_t w_[3]; uint64_t at = S->ubo[st][b->reg].addr + b->c0_off;
                if (read_u32(at, &w_[0]) && read_u32(at + 4, &w_[1]) && read_u32(at + 8, &w_[2])) { memset(&ptr_, 0, sizeof ptr_); ptr_.addr = (uint64_t)w_[1] << 32 | w_[0]; ptr_.size = w_[2]; u = &ptr_; }
            }
            VkBuffer bb; VkDeviceSize oo;
            if (u && u->addr && u->size && vkm_use(u->addr, u->size, &bb, &oo)) {
                VkDeviceSize al = g_vk.props.limits.minStorageBufferOffsetAlignment, need = (u->size + 255) & ~(VkDeviceSize)255;
                if (oo % al && st == 5 && R0.bounce && u->size <= BOUNCE_MAX && g_nbnc < 16) {   // a dispatch (no pass open): through the bounce buffer
                    if (g_bnc_next + need > BOUNCE_SIZE) g_bnc_next = 0;
                    VkDeviceSize bo = g_bnc_next; g_bnc_next += need < al ? al : need;
                    VkCommandBuffer cb = vks_cb(); VkBufferCopy c = {oo, bo, u->size}; barrier_all(cb);   // earlier GPU writes of the pool first
                    vkCmdCopyBuffer(cb, bb, R0.bounce, 1, &c);   // dispatch's barrier_all orders it before the shader
                    *o = (VkDescriptorBufferInfo){R0.bounce, bo, u->size};
                    if (b->rw) { g_bnc[g_nbnc].pool = bb; g_bnc[g_nbnc].poff = oo; g_bnc[g_nbnc].boff = bo; g_bnc[g_nbnc].n = u->size; g_nbnc++;
                        if (swrites) { *swrites = 1; int dup = 0; for (int k = 0; k < g_nwb; k++) dup |= g_wb[k].a == u->addr && g_wb[k].n == u->size;
                                       if (!dup && g_nwb < 256) { g_wb[g_nwb].a = u->addr; g_wb[g_nwb].n = u->size; g_wb[g_nwb].base = 0; g_nwb++; } } }
                    RED_ONCE(45, "a storage buffer at an address Vulkan cannot bind (%#llx, %#llx bytes, %s; shader %s binding %u): bound through a copy (later ones are not logged)",
                             (unsigned long long)u->addr, (unsigned long long)u->size, b->rw ? "written" : "read only", sh->key, b->binding);
                }
                else if (oo % al) RED_ONCE(44, "a storage buffer at an address Vulkan cannot bind (%#llx, %#llx bytes, %s; shader %s binding %u): not bound",
                                                                                   (unsigned long long)u->addr, (unsigned long long)u->size, b->rw ? "written" : "read only", sh->key, b->binding);
                else {
                    *o = (VkDescriptorBufferInfo){bb, oo, u->size};
                    if (b->rw && swrites) { *swrites = 1; int dup = 0; for (int k = 0; k < g_nwb; k++) dup |= g_wb[k].a == u->addr && g_wb[k].n == u->size;
                        if (!dup && g_nwb < 256) { g_wb[g_nwb].a = u->addr; g_wb[g_nwb].n = u->size; g_wb[g_nwb].base = 0; g_nwb++; } }
                }
            }
            x->pBufferInfo = o; break; }
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: {
            VkDescriptorImageInfo *o = &ii[nw]; *o = (VkDescriptorImageInfo){R0.samp, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            if (b->kind == VKB_TEX) {
                uint64_t h = tex_slot_h(st, (unsigned)b->reg);
                int tid = (int)(h & 0xfffff), sid = b->samp_kind == VKB_SAMP_SLOT ? (b->samp_reg >= 0 && b->samp_reg < 128 ? (int)((S->samp[st][b->samp_reg] >> 20) & 0xfff) : 0)
                                                  : b->samp_kind == VKB_SAMP_ID ? b->samp_reg : (int)((h >> 20) & 0xfff);
                vkt_image *im;
                o->imageView = tex_view_m(tid, b, &o->imageLayout, &im);
                o->sampler = samp_get_m(sid);
                { static int vsnt = -1; if (vsnt < 0) { const char *e = getenv("NXRT_VK_VS_NOTEX"); vsnt = e && *e == '1'; }   // test switch: vertex shaders sample null textures (zeros)
                  if (vsnt && st == 0) { o->imageView = VK_NULL_HANDLE; im = NULL; } }
                if (note) vsnote("t%u id%d %s%ux%u ", b->binding, tid, im ? "" : "null ", im ? im->w : 0, im ? im->h : 0);
                if (!o->imageView) o->imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                if ((b->dref || b->plain) && o->imageView) {   // the shader's depth comparison against what is bound: Vulkan leaves a mismatch undefined
                    samp_par sp_; int cmp = sid > 0 && vkt_samp_copy(sid, &sp_) && sp_.cmp_mode == NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE;
                    int dview = im ? (im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0 : 1;   // no image: the depth-compare view of dcmp_view
                    const char *why = b->dref && !dview ? "a depth comparison on a colour image" : b->dref && !cmp ? "a depth comparison with a sampler that does not compare" :
                                      b->plain && cmp ? "a sample without comparison through a comparing sampler" : NULL;
                    // The console's sampler compares only when the instruction asks for it (TEX.DC): a plain sample through a comparing sampler reads the depth itself.
                    // In Vulkan a comparing sampler under a plain OpImageSample is undefined; on NVIDIA it is the suspect of the GPU freeze (docs/PLAN_NEXT.md 5b).
                    // Such a binding gets the same sampler without comparison. gamectl set plain_nocmp 0 (NXRT_VK_PLAIN_NOCMP=0): the comparing sampler, as before
                    int nocmp = 0;
                    if (b->plain && !b->dref && cmp && vkc_int("plain_nocmp", 1)) { samp_par q = sp_; q.cmp_mode = NVN_COMPARE_MODE_NONE; VkSampler s2 = vkt_sampler(&q); if (s2) { o->sampler = s2; nocmp = 1; g_st.plain_nocmp++; } }
                    static uint64_t seen[512]; static int nseen; uint64_t sk = 1469598103934665603ull;   // once per shader, binding and kind
                    for (const char *c = sh->key; *c; c++) sk = (sk ^ (uint8_t)*c) * 1099511628211ull;
                    sk = (sk ^ b->binding) * 1099511628211ull; sk = (sk ^ (uintptr_t)why) * 1099511628211ull;
                    int fresh = why != NULL; for (int k = 0; fresh && k < nseen; k++) if (seen[k] == sk) fresh = 0;
                    if (fresh && nseen < 512) seen[nseen++] = sk; else fresh = 0;
                    if (fresh) {
                        VKLOG("descriptor mismatch: shader %s binding %u (%s%s): %s; texture ID %d ser%llu %s %ux%u, sampler ID %d%s", sh->key, b->binding, b->dref ? "compare" : "", b->dref && b->plain ? " and plain" : b->plain ? "plain" : "",
                              why, tid, im ? (unsigned long long)im->serial : 0ull, im ? im->fi->name : "(depth view)", im ? im->w : 0, im ? im->h : 0, sid, nocmp ? " (bound without comparison)" : ""); }
                }
            }
            x->pImageInfo = o; break; }
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
            VkDescriptorImageInfo *o = &ii[nw]; *o = (VkDescriptorImageInfo){VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL};
            if (b->kind == VKB_IMG && b->reg >= 0 && b->reg < 32) { o->imageView = img_view((int)(S->img[st][b->reg] & 0xfffff), b); if (swrites) *swrites = 1; }
            x->pImageInfo = o; break; }
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
            bv[nw] = b->kind == VKB_TBUF ? tbuf_view((int)(tex_slot_h(st, (unsigned)b->reg) & 0xfffff), b->ct) : VK_NULL_HANDLE;
            x->pTexelBufferView = &bv[nw]; break; }
        default: continue;
        }
        if (phb_) g_ph_ns[phk_] += PH_T() - phb_;
        nw++;
    }
    { PH0(); vkUpdateDescriptorSets(g_vk.dev, (uint32_t)nw, w, 0, NULL); PH1(PH_WS_UPD); }
    return ds;
}

// ---------- Draws (nvn_mtl.m draw_) ----------
static VkFormatFeatureFlags fmt_features(VkFormat f, int buffer) {
    static vk_hmap m = {8};
    uint64_t k = (uint64_t)f | (uint64_t)buffer << 32;
    void *v = vk_hm_get(&m, &k);
    if (v) return (VkFormatFeatureFlags)((uintptr_t)v >> 1);
    VkFormatProperties p; vkGetPhysicalDeviceFormatProperties(g_vk.pd, f, &p);
    VkFormatFeatureFlags r = buffer ? p.bufferFeatures : p.optimalTilingFeatures;
    vk_hm_put(&m, &k, (void *)(((uintptr_t)r << 1) | 1));
    return r;
}
// The inputs whose format's numeric class differs from the declared one in a way Vulkan cannot fetch bit-exactly: the vertex shader variant that declares them
// so (vksh_vin_variant). 32-bit components need none (an integer format read into a float input keeps the bits, and the reverse)
// Dynamic state already set in the command buffer being recorded: a draw sets only what differs from the draw before (most draws repeat their
// neighbours' viewport, depth, stencil and blend state; each vkCmdSet* is a driver call, ~25 a draw). Forgotten when the command buffer changes
// (submission, split, slot) and whenever a pipeline other than a draw's is bound (clear_draw: its static state makes earlier dynamic state undefined).
// gamectl set dyn_dedupe 0 (NXRT_VK_DYN_DEDUPE=0): every draw sets everything, as before
enum { DS_VP, DS_SC, DS_CULL, DS_FRONT, DS_TOPO, DS_BIAS_EN, DS_BIAS, DS_BCOL, DS_DT, DS_DW, DS_DCMP, DS_DBT, DS_ST, DS_SOPF, DS_SOPB, DS_SM0, DS_SM1, DS_RD, DS_RESTART,
       DS_BEN, DS_BEQ, DS_WM, DS_LOGIC_EN, DS_LOGIC, DS_A2C, DS_POLY, DS_DCLAMP, DS_LW, DS_VI, DS_VB0, DS_N = DS_VB0 + 17 };
static struct {
    VkCommandBuffer cb; uint64_t seq, valid; int on;
    VkViewport vp; VkRect2D sc; uint32_t cull, front, topo, bias_en, dt, dw, dcmp, st, rd, restart, logic_en, logic, a2c, poly, dclamp;
    float bias[3], bcol[4]; uint32_t sopf[4], sopb[4], sm[2][3];
} g_ds;
static void ds_forget(void) { g_ds.valid = 0; }
static void ds_begin(VkCommandBuffer cb) {   // a draw's state goes into cb: what g_ds holds counts only if cb is still the same recording (a slot's buffer is reused by later recordings)
    uint64_t seq = vks_take_seq();
    if (g_ds.cb != cb || g_ds.seq != seq) { g_ds.cb = cb; g_ds.seq = seq; g_ds.valid = 0; g_ds.on = vkc_int("dyn_dedupe", 1); }
}
static inline int ds_same(VkCommandBuffer cb, int slot, void *have, const void *want, size_t n) {   // 1: already set to this in cb (ds_begin first); else records it as set
    (void)cb;
    if (g_ds.on && (g_ds.valid >> slot & 1) && !memcmp(have, want, n)) { g_n_ds_skip++; return 1; }
    g_n_ds_set++;
    memcpy(have, want, n); g_ds.valid |= 1ull << slot; return 0;
}
static inline int ds_once(VkCommandBuffer cb, int slot) { static char none; return ds_same(cb, slot, &none, &none, 0); }   // a state whose value never changes: set once per cb
#define DS_U(cb, slot, field, v, call) do { uint32_t v_ = (uint32_t)(v); if (!ds_same(cb, slot, &g_ds.field, &v_, sizeof v_)) { call; } } while (0)
static const vks_shader *vin_shader(const vks_shader *vs) {
    uint32_t mask = 0; uint8_t ct[32] = {0};
    for (int i = 0; i < vs->nin; i++) {
        int a = vs->in[i].loc, nct; if (a >= 16 || a >= S->nattr || !S->attr[a].fmt) continue;
        int exact; VkFormat f = vkp_vertex_format(S->attr[a].fmt, vs->in[i].ct, &exact);
        if (f && exact) continue;
        if (vkp_vertex_format_natural(S->attr[a].fmt, &nct) && nct != vs->in[i].ct) { mask |= 1u << a; ct[a] = (uint8_t)nct; }
    }
    return mask ? vksh_vin_variant(vs, mask, ct) : vs;
}
static VkVertexInputBindingDescription2EXT g_lvb[17]; static VkVertexInputAttributeDescription2EXT g_lva[32]; static uint32_t g_lnb, g_lna;   // the last vertex input (NXRT_VK_DRAWLOG)
static void set_vertex_input(VkCommandBuffer cb, const vks_shader *vs, vkp_gkey *k, uint32_t *used, int *use_zero) {
    VkVertexInputBindingDescription2EXT vb[17]; VkVertexInputAttributeDescription2EXT va[32];
    uint32_t na = 0, nb = 0; *used = 0; *use_zero = 0;
    for (int i = 0; i < vs->nin && na < 32; i++) {
        int a = vs->in[i].loc, ct = vs->in[i].ct;
        VkFormat f = VK_FORMAT_UNDEFINED; int exact = 1, st = 0; uint32_t off = 0;
        if (a < 16 && a < S->nattr && S->attr[a].fmt) {
            f = vkp_vertex_format(S->attr[a].fmt, ct, &exact); st = S->attr[a].stream; off = S->attr[a].offset;
            if (st >= 16) f = VK_FORMAT_UNDEFINED;
            if (f && !(fmt_features(f, 1) & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)) { RED_ONCE(50, "vertex format %#x (Vulkan %d) is not a vertex buffer format on this device: the attribute reads zeros", S->attr[a].fmt, f); f = VK_FORMAT_UNDEFINED; }
            if (f && !exact) { g_st.inexact_vertex++; static int n; if (n++ < 8) VKRED("vertex attribute %d: format %#x into a %s input of shader %s cannot keep Maxwell's bits (nvn_mtl.m converts these on the CPU): read as %d",
                                                                                          a, S->attr[a].fmt, ct == VKS_CT_UINT ? "uint" : ct == VKS_CT_INT ? "int" : "float", vs->key, f); }
        }
        if (!f) {   // no attribute: the zero buffer (nvn_mtl.m ZERO_ATTR)
            f = ct == VKS_CT_UINT ? VK_FORMAT_R32G32B32A32_UINT : ct == VKS_CT_INT ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_SFLOAT;
            va[na++] = (VkVertexInputAttributeDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, NULL, (uint32_t)a, 16, f, 0};
            *use_zero = 1; continue;
        }
        va[na++] = (VkVertexInputAttributeDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, NULL, (uint32_t)a, (uint32_t)st, f, off};
        *used |= 1u << st;
    }
    for (int st = 0; st < 16; st++) if (*used >> st & 1) {
        uint32_t div = S->stream[st].divisor;
        vb[nb++] = (VkVertexInputBindingDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, NULL, (uint32_t)st, S->stream[st].stride,
                                                         div ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX, div ? div : 1};
    }
    if (*use_zero) vb[nb++] = (VkVertexInputBindingDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, NULL, 16, 0, VK_VERTEX_INPUT_RATE_VERTEX, 1};
    if (g_dl_on && dl_find(vs->key) >= 0) { memcpy(g_lvb, vb, nb * sizeof *vb); memcpy(g_lva, va, na * sizeof *va); g_lnb = nb; g_lna = na; }   // for the draw log
    if (g_vkp_dyn.vertex_input) {
        static struct { uint32_t nb, na; VkVertexInputBindingDescription2EXT vb[17]; VkVertexInputAttributeDescription2EXT va[32]; } have, want;
        memset(&want, 0, sizeof want); want.nb = nb; want.na = na; memcpy(want.vb, vb, nb * sizeof *vb); memcpy(want.va, va, na * sizeof *va);
        ds_begin(cb);
        if (!ds_same(cb, DS_VI, &have, &want, sizeof want)) vkCmdSetVertexInputEXT(cb, nb, vb, na, va);
    }
    else {   // baked into the pipeline
        k->nattr = (uint8_t)(na > 16 ? 16 : na); k->nbind = 0;
        for (uint32_t i = 0; i < k->nattr; i++) { k->attr[i].loc = (uint8_t)va[i].location; k->attr[i].bind = (uint8_t)va[i].binding; k->attr[i].fmt = va[i].format; k->attr[i].off = va[i].offset; }
        for (uint32_t i = 0; i < nb; i++) { uint32_t bi = vb[i].binding; k->bind[bi].stride = vb[i].stride; k->bind[bi].rate = (uint8_t)vb[i].inputRate; k->bind[bi].divisor = vb[i].divisor; if (bi + 1 > k->nbind) k->nbind = (uint8_t)(bi + 1); }
    }
}
static void bind_vertex_buffers(VkCommandBuffer cb, uint32_t used, int use_zero) {
    for (int st = 0; st < 16; st++) if (used >> st & 1) {
        NVNbufferRange r = S->vbuf[st]; VkBuffer b = R0.zero; VkDeviceSize o = 0, n = ZERO_SIZE;
        if (r.address && r.size && vkm_use(r.address, r.size, &b, &o)) n = r.size;
        else { b = R0.zero; o = 0; n = ZERO_SIZE; if (r.address) RED_ONCE(51, "vertex stream %d at %#llx is in no memory pool: it reads zeros", st, (unsigned long long)r.address); }
        static struct { VkBuffer b; VkDeviceSize o, n; } have[17]; struct { VkBuffer b; VkDeviceSize o, n; } want = {b, o, n};
        ds_begin(cb);
        if (!ds_same(cb, DS_VB0 + st, &have[st], &want, sizeof want)) vkCmdBindVertexBuffers2(cb, (uint32_t)st, 1, &b, &o, &n, NULL);
    }
    if (use_zero) { static struct { VkBuffer b; VkDeviceSize o, n; } have; struct { VkBuffer b; VkDeviceSize o, n; } want = {R0.zero, 0, ZERO_SIZE};
        ds_begin(cb);
        if (!ds_same(cb, DS_VB0 + 16, &have, &want, sizeof want)) { VkDeviceSize o = 0, n = ZERO_SIZE; vkCmdBindVertexBuffers2(cb, 16, 1, &R0.zero, &o, &n, NULL); } }
}
static int read_u32(uint64_t a, uint32_t *v) {   // guest memory, guarded
    NVN_GUARDED(return 0);
    memcpy(v, (const void *)(uintptr_t)a, 4); NVN_GUARD_END(); return 1;
}
// A draw's row in the flight recording: the program it would use and its arguments, whether or not it is made
static vkf_row *fl_draw(const p_draw *dw, const p_draw_idx *di) {
    vkf_row *r = vkf_row_next(); if (!r) return NULL;
    const vks_shader *v = dw && dw->prim == DM_PRIM_MESH_TASKS && S->bst[4] ? prog_stage(S->bst[4], 4) : S->bst[0] ? prog_stage(S->bst[0], 0) : NULL, *f = S->bst[1] ? prog_stage(S->bst[1], 1) : NULL;
    r->vs = vkf_key(v ? v->key : NULL); r->ps = vkf_key(f ? f->key : NULL); r->pass = (uint32_t)g_st.passes;
    uint64_t ind = dw ? dw->ind : di->ind;
    r->flags = (di ? FL_INDEXED : 0) | (ind ? FL_INDIRECT : 0) | ((dw ? dw->mdi_cnt : di->mdi_cnt) ? FL_MDI : 0) | (dw && dw->prim == DM_PRIM_MESH_TASKS ? FL_MESH : 0);
    r->mdi_i = (uint32_t)(dw ? dw->mdi_i : di->mdi_i);
    if (P.ncolor && P.ci[0]) r->rt = (uint32_t)P.ci[0]->serial;
    if (ind) for (int k = 0; k < (dw ? 4 : 5); k++) read_u32(ind + 4 * (uint64_t)k, &r->a[k]);
    else if (dw) { r->a[0] = (uint32_t)dw->count; r->a[1] = (uint32_t)dw->instances; r->a[2] = (uint32_t)dw->first; r->a[3] = (uint32_t)dw->base_instance; }
    else { r->a[0] = (uint32_t)di->count; r->a[1] = (uint32_t)di->instances; r->a[3] = (uint32_t)di->base_vertex; r->a[4] = (uint32_t)di->base_instance; }
    return r;
}
static void draw(const p_draw *dw, const p_draw_idx *di) {
    vkf_row *fr = fl_draw(dw, di);
    if (S->render_skip) { g_st.draws_skipped++; if (fr) fr->flags |= FL_RC_SKIP; return; }
    int prim = dw ? dw->prim : di->prim, expand;
    int mesh = dw && prim == DM_PRIM_MESH_TASKS;
    if (mesh && vkc_int("no_mesh", 0)) return;
    if (mesh) { prim = NVN_DRAW_PRIMITIVE_TRIANGLES; if (!g_vk.has.mesh_shader || !vkCmdDrawMeshTasksEXT) { RED_ONCE(96, "task and mesh draws: the device has no VK_EXT_mesh_shader, they are not drawn"); return; }
        if (dw->first) RED_ONCE(97, "a mesh draw from a first workgroup other than 0: Vulkan has no parameter for it, drawn from 0"); }
    uint64_t ind = dw ? dw->ind : di->ind, mdi = dw ? dw->mdi_cnt : di->mdi_cnt; int mdi_i = dw ? dw->mdi_i : di->mdi_i;
    if (mdi) { uint32_t cnt; int rd = read_u32(mdi, &cnt);
        if (fr) fr->mdi_cnt = cnt;
        if (rd && (uint32_t)mdi_i >= cnt) { if (fr) { fr->flags |= FL_MDI_CUT; vkf_note_args(fr, ind, dw ? 16 : 20, mdi); } return; }
        RED_ONCE(52, "an indirect-count multi-draw: its count is read from guest memory at replay (a count the GPU writes in the same submission is missed)"); }
    VkPrimitiveTopology topo = vkp_topology(prim, &expand);
    if (expand == VKP_EXPAND_UNSUPPORTED || expand == VKP_EXPAND_LINE_LOOP) { static int n; if (n++ < 4) VKRED("primitive type %d is not drawn", prim); g_st.draws_skipped++; return; }
    if (S->poly.cull == NVN_FACE_FRONT_AND_BACK && !S->rast_discard) return;
    const vks_shader *sh[5] = {0};
    for (int s = 0; s < 5; s++) if (S->bst[s]) sh[s] = prog_stage(S->bst[s], s);
    if (fr && (vkc_hidden(fr->vs) || vkc_hidden(fr->ps))) { fr->flags |= FL_HIDDEN; return; }
    if (!fr && vkc_any_hidden() && (vkc_hidden(vkf_key(sh[0] ? sh[0]->key : NULL)) || vkc_hidden(vkf_key(sh[1] ? sh[1]->key : NULL)))) return;
    if (mesh) {   // a mesh program has no vertex or geometry stage, and a task shader only if it is the same program's: what an earlier program left in those places goes
        sh[0] = sh[2] = NULL;
        if (sh[3] && (sh[3]->mesh != 1 || S->bst[3] != S->bst[4])) sh[3] = NULL;
        if (!sh[4] || sh[4]->mesh != 2) { g_st.skip_no_shader++; g_st.draws_skipped++; return; }
    } else if (sh[4] && sh[4]->mesh) sh[3] = sh[4] = NULL;
    if (!mesh && (!sh[0] || (S->bst[1] && !sh[1] && !S->rast_discard))) { g_st.skip_no_shader++; g_st.draws_skipped++; return; }
    if (!mesh) sh[0] = vin_shader(sh[0]);
    if (S->rast_discard) sh[1] = NULL;
    // the render pass, its scissor and viewport (nvn_mtl.m draw_: viewport flipped for the lower-left origin there; Vulkan's y axis runs the other way)
    // With every draw checkpointed: an indirect draw's arguments, as the GPU holds them, copied out first, outside the pass (the pass is ended and begun again). A GPU stuck inside the pass never gets to copies at its end
    if (ind && g_vk.has.checkpoints >= 2) { VkBuffer ab; VkDeviceSize ao; if (vkm_use(ind, dw ? 16 : 20, &ab, &ao)) { pass_end(); vk_ck_args(vks_cb(), vk_ck_next(), ab, ao, dw ? 16 : 20); } }
    if (!P.open && !pass_begin()) { g_st.skip_no_target++; g_st.draws_skipped++; return; }
    uint32_t tw = P.any ? P.w : (uint32_t)(S->vp[0][0] + S->vp[0][2]), th = P.any ? P.h : (uint32_t)(S->vp[0][1] + S->vp[0][3]);
    VkRect2D sc; if (!scissor_rect(tw, th, &sc)) { if (!S->rast_discard) return; sc = (VkRect2D){{0, 0}, {tw, th}}; }
    VkViewport vp = S->has_vp ? (VkViewport){S->vp[0][0], S->vp[0][1], S->vp[0][2], S->vp[0][3], S->dr[0], S->dr[1]} : (VkViewport){0, 0, (float)tw, (float)th, S->dr[0], S->dr[1]};
    if (vp.width <= 0 || vp.height == 0) return;
    int flip = !g_origin_ll;
    { uint32_t sw; memcpy(&sw, S->vp_sw[0], 4);
      if (sw == 0x100) flip = !flip;   // NEGATIVE_Y: clip y negated, the viewport flipped about its centre
      else if (sw) RED_ONCE(53, "viewport swizzle %d,%d,%d,%d not implemented (drawn unswizzled)", S->vp_sw[0][0], S->vp_sw[0][1] ^ 2, S->vp_sw[0][2] ^ 4, S->vp_sw[0][3] ^ 6); }
    if (flip) { vp.y += vp.height; vp.height = -vp.height; }
    vp.minDepth = vp.minDepth < 0 ? 0 : vp.minDepth > 1 ? 1 : vp.minDepth; vp.maxDepth = vp.maxDepth < 0 ? 0 : vp.maxDepth > 1 ? 1 : vp.maxDepth;
    VkCommandBuffer cb = vks_cb();
    // M6 phase B: target 0 blending with output 1 as its second source takes the dual-source variant (nvn_mtl.m pso_reads_src1 / fs_dual), an alpha function other than
    // ALWAYS the alpha test variant (nvn_mtl.m atest_key: no test when colour 0 goes to an integer target), with the function and reference as push constants (at_bind)
    int fsv = 0, atest = 0;
    if (sh[1] && P.ncolor && P.cv[0] && (S->chmask.mask & 0xf) && (S->color.blend_en & 1) && (sh[1]->out_mask & 2) && g_vk.f10.dualSrcBlend) {
        blend_st b0 = vkp_adv_blend(S->blend[0]);
        if (vkp_blend_is_src1(b0.src_rgb) || vkp_blend_is_src1(b0.dst_rgb) || vkp_blend_is_src1(b0.src_a) || vkp_blend_is_src1(b0.dst_a)) fsv |= VKSH_FS_DUAL;
    }
    if (sh[1] && S->color.alpha_func && S->color.alpha_func != NVN_ALPHA_FUNC_ALWAYS && (sh[1]->out_mask & 1) && !(P.ncolor && P.cv[0] && fmt_int_kind(P.cf[0]))) fsv |= VKSH_FS_ALPHA;
    if (fsv) { sh[1] = vksh_fs_variant(sh[1], fsv); atest = sh[1]->alpha; }
    if (g_id_on && sh[1]) sh[1] = vksh_id_variant(sh[1]);
    // pipeline: what is not dynamic here
    vkp_gkey k; memset(&k, 0, sizeof k);
    for (int s = 0; s < 5; s++) k.sh[s] = sh[s];
    k.ncolor = (uint8_t)P.ncolor; for (int i = 0; i < P.ncolor; i++) k.cfmt[i] = P.cv[i] ? (uint32_t)P.cf[i] : 0;
    k.dfmt = P.dv ? (uint32_t)P.df : 0; k.has_depth = (uint8_t)P.hd; k.has_stencil = (uint8_t)P.hs; k.samples = P.samples;
    k.topo_class = g_vkp_dyn.topo_unrestricted ? VKP_TOPO_ANY : (uint8_t)vkp_topology_class(topo);
    k.depth_m1 = !g_depth_zero;
    VkBool32 ben[8]; VkColorBlendEquationEXT beq[8]; VkColorComponentFlags wm[8];
    const vks_shader *fs = sh[1];
    for (int i = 0; i < P.ncolor; i++) {
        uint32_t m = (S->chmask.mask >> (4 * i)) & 0xf;
        if (!P.cv[i] || !fs || !(fs->out_mask >> i & 1)) m = 0;   // an output the shader does not write: Maxwell leaves the target as it was (nvn_mtl.m pso_desc)
        blend_st bl = vkp_adv_blend(S->blend[i]);
        int en = m && (S->color.blend_en >> i & 1) && (fmt_features(P.cf[i], 0) & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT);
        if (en && (vkp_blend_is_src1(bl.src_rgb) || vkp_blend_is_src1(bl.dst_rgb) || vkp_blend_is_src1(bl.src_a) || vkp_blend_is_src1(bl.dst_a))) {
            if (i == 0 && fs && fs->dual) { static int n; if (!n++) VKLOG("dual-source blending: shader %s's output 1 is target 0's second source", fs->key); }
            else {   // an unwritten second source counts as zero, as nvn_mtl.m blend_zero_src1
                RED_ONCE(54, "dual-source blending (shader %s, target %d): the second source counts as zero (%s)", fs ? fs->key : "-", i, !fs || !(fs->out_mask & 2) ? "the shader does not write output 1" :
                         !g_vk.f10.dualSrcBlend ? "the device has no dual-source blending" : i ? "only target 0 blends with a second source" : "no dual-source variant");
                bl.src_rgb = vkp_blend_zero_src1(bl.src_rgb); bl.dst_rgb = vkp_blend_zero_src1(bl.dst_rgb); bl.src_a = vkp_blend_zero_src1(bl.src_a); bl.dst_a = vkp_blend_zero_src1(bl.dst_a);
            }
        }
        if (m && fs && fs->out_ct[i] == VKS_CT_FLOAT && fmt_int_kind(P.cf[i])) RED_ONCE(55, "shader %s writes float colour %d into an integer target (nvn_mtl.m fs_uint): the values are undefined", fs->key, i);
        ben[i] = en; wm[i] = m;
        beq[i] = (VkColorBlendEquationEXT){vkp_blend_factor(bl.src_rgb), vkp_blend_factor(bl.dst_rgb), vkp_blend_op(bl.eq_rgb), vkp_blend_factor(bl.src_a), vkp_blend_factor(bl.dst_a), vkp_blend_op(bl.eq_a)};
        k.att[i] = (vkp_att){(uint8_t)(g_vkp_dyn.blend_enable ? 0 : en), (uint8_t)(g_vkp_dyn.blend_eq ? 0 : beq[i].srcColorBlendFactor), (uint8_t)(g_vkp_dyn.blend_eq ? 0 : beq[i].dstColorBlendFactor),
                             (uint8_t)(g_vkp_dyn.blend_eq ? 0 : beq[i].srcAlphaBlendFactor), (uint8_t)(g_vkp_dyn.blend_eq ? 0 : beq[i].dstAlphaBlendFactor),
                             (uint8_t)(g_vkp_dyn.blend_eq ? 0 : beq[i].colorBlendOp), (uint8_t)(g_vkp_dyn.blend_eq ? 0 : beq[i].alphaBlendOp), (uint8_t)(g_vkp_dyn.write_mask ? 0 : m)};
    }
    int logic = S->color.logic_op != NVN_LOGIC_OP_COPY && g_vk.f10.logicOp;
    if ((fsv & VKSH_FS_ALPHA) && atest) { static int n; if (!n++) VKLOG("alpha test: function %u, reference %g (shader %s's alpha test variant)", S->color.alpha_func, S->alpha_ref, fs->key); }
    if (!g_vkp_dyn.logic_op_enable) k.logic_en = (uint8_t)logic;
    if (!g_vkp_dyn.logic_op) k.logic_op = (uint8_t)(logic ? vkp_logic_op(S->color.logic_op) : VK_LOGIC_OP_COPY);
    if (!g_vkp_dyn.a2c) k.a2c = S->ms.a2c != 0;
    if (!g_vkp_dyn.polygon_mode) k.poly = (uint8_t)S->poly.mode;
    if (!g_vkp_dyn.depth_clamp) k.depth_clamp = S->depth_clamp != 0 && g_vk.f10.depthClamp;
    uint32_t used; int use_zero;
    if (!mesh) set_vertex_input(cb, sh[0], &k, &used, &use_zero);
    VkPipelineLayout layout;
    VkPipeline pipe = vkp_graphics(&k, &layout);
    if (!pipe) { g_st.skip_no_pipeline++; g_st.draws_skipped++; return; }
    if (pipe != g_last_pipe) { vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); g_last_pipe = pipe;
        // a pipeline without some of the draws' dynamic states (vk_pipe.c: no blend state without colour targets, no topology, restart or vertex input in a mesh pipeline)
        // applies its static values: what g_ds holds for those no longer counts
        if (!P.ncolor) g_ds.valid &= ~(1ull << DS_BEN | 1ull << DS_BEQ | 1ull << DS_WM);
        if (mesh) g_ds.valid &= ~(1ull << DS_TOPO | 1ull << DS_RESTART | 1ull << DS_VI); }
    if (atest) { uint32_t pc[2] = {S->color.alpha_func, 0}; memcpy(&pc[1], &S->alpha_ref, 4); vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, VKP_ALPHA_PC_SIZE, pc); }
    // dynamic state (only what differs from what cb already has: g_ds)
    ds_begin(cb);
    if (!ds_same(cb, DS_VP, &g_ds.vp, &vp, sizeof vp)) vkCmdSetViewportWithCount(cb, 1, &vp);
    if (!ds_same(cb, DS_SC, &g_ds.sc, &sc, sizeof sc)) vkCmdSetScissorWithCount(cb, 1, &sc);
    if (!ds_once(cb, DS_LW)) vkCmdSetLineWidth(cb, 1.0f);
    DS_U(cb, DS_CULL, cull, vkp_cull(S->poly.cull), vkCmdSetCullMode(cb, (VkCullModeFlags)v_));
    DS_U(cb, DS_FRONT, front, vkp_front_face(S->poly.front, g_queue_origin_ll), vkCmdSetFrontFace(cb, (VkFrontFace)v_));
    DS_U(cb, DS_TOPO, topo, topo, vkCmdSetPrimitiveTopology(cb, topo));
    int cls = vkp_topology_class(topo), pm = S->poly.mode;
    int bias = (S->poly.offset_en & (cls == 0 || pm == NVN_POLYGON_MODE_POINT ? 1 : cls == 1 || pm == NVN_POLYGON_MODE_LINE ? 2 : 4)) != 0;
    DS_U(cb, DS_BIAS_EN, bias_en, bias, vkCmdSetDepthBiasEnable(cb, (VkBool32)v_));
    { float bv[3] = {S->po.units, g_vk.f10.depthBiasClamp ? S->po.clamp : 0.0f, S->po.factor}; if (!ds_same(cb, DS_BIAS, g_ds.bias, bv, sizeof bv)) vkCmdSetDepthBias(cb, bv[0], bv[1], bv[2]); }
    if (!ds_same(cb, DS_BCOL, g_ds.bcol, S->blend_color, sizeof g_ds.bcol)) vkCmdSetBlendConstants(cb, S->blend_color);
    DS_U(cb, DS_DT, dt, P.hd && S->ds.dt, vkCmdSetDepthTestEnable(cb, (VkBool32)v_));
    DS_U(cb, DS_DW, dw, P.hd && S->ds.dw, vkCmdSetDepthWriteEnable(cb, (VkBool32)v_));
    DS_U(cb, DS_DCMP, dcmp, vkp_compare(S->ds.df), vkCmdSetDepthCompareOp(cb, (VkCompareOp)v_));
    if (!ds_once(cb, DS_DBT)) vkCmdSetDepthBoundsTestEnable(cb, VK_FALSE);
    DS_U(cb, DS_ST, st, P.hs && S->ds.st, vkCmdSetStencilTestEnable(cb, (VkBool32)v_));
    { uint32_t o[4] = {vkp_stencil_op(S->ds.fail_f), vkp_stencil_op(S->ds.zpass_f), vkp_stencil_op(S->ds.zfail_f), vkp_compare(S->ds.sf_f)};
      if (!ds_same(cb, DS_SOPF, g_ds.sopf, o, sizeof o)) vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_BIT, (VkStencilOp)o[0], (VkStencilOp)o[1], (VkStencilOp)o[2], (VkCompareOp)o[3]); }
    { uint32_t o[4] = {vkp_stencil_op(S->ds.fail_b), vkp_stencil_op(S->ds.zpass_b), vkp_stencil_op(S->ds.zfail_b), vkp_compare(S->ds.sf_b)};
      if (!ds_same(cb, DS_SOPB, g_ds.sopb, o, sizeof o)) vkCmdSetStencilOp(cb, VK_STENCIL_FACE_BACK_BIT, (VkStencilOp)o[0], (VkStencilOp)o[1], (VkStencilOp)o[2], (VkCompareOp)o[3]); }
    for (int f = 0; f < 2; f++) {
        VkStencilFaceFlags fl = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
        uint32_t m[3] = {S->st_vmask[f] & 0xff, S->st_wmask[f] & 0xff, S->st_ref[f] & 0xff};
        if (!ds_same(cb, DS_SM0 + f, g_ds.sm[f], m, sizeof m)) { vkCmdSetStencilCompareMask(cb, fl, m[0]); vkCmdSetStencilWriteMask(cb, fl, m[1]); vkCmdSetStencilReference(cb, fl, m[2]); }
    }
    DS_U(cb, DS_RD, rd, S->rast_discard != 0, vkCmdSetRasterizerDiscardEnable(cb, (VkBool32)v_));
    int restart = S->restart.on && di;
    if (restart && ((di->itype == NVN_INDEX_TYPE_UNSIGNED_BYTE && S->restart.index != 0xff) || (di->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT && S->restart.index != 0xffff) || (di->itype == NVN_INDEX_TYPE_UNSIGNED_INT && S->restart.index != 0xffffffffu)))
        RED_ONCE(57, "primitive restart index %#x is not the index type's largest value: Vulkan restarts only there", S->restart.index);
    if (restart && (cls != 1 && cls != 2 ? 1 : topo == VK_PRIMITIVE_TOPOLOGY_LINE_LIST || topo == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || topo == VK_PRIMITIVE_TOPOLOGY_POINT_LIST) && !g_vk.has.prim_list_restart) restart = 0;
    DS_U(cb, DS_RESTART, restart, restart, vkCmdSetPrimitiveRestartEnable(cb, (VkBool32)v_));
    if (P.ncolor) {   // per attachment: the count is part of what is compared (the arrays are zero past it)
        struct { uint32_t n; VkBool32 v[8]; } be = {(uint32_t)P.ncolor, {0}}; struct { uint32_t n; VkColorBlendEquationEXT v[8]; } bq; struct { uint32_t n; VkColorComponentFlags v[8]; } bw = {(uint32_t)P.ncolor, {0}};
        memset(&bq, 0, sizeof bq); bq.n = (uint32_t)P.ncolor;
        for (int i = 0; i < P.ncolor; i++) { be.v[i] = ben[i]; bq.v[i] = beq[i]; bw.v[i] = wm[i]; }
        static struct { uint32_t n; VkBool32 v[8]; } h_be; static struct { uint32_t n; VkColorBlendEquationEXT v[8]; } h_bq; static struct { uint32_t n; VkColorComponentFlags v[8]; } h_bw;
        if (g_vkp_dyn.blend_enable && !ds_same(cb, DS_BEN, &h_be, &be, sizeof be)) vkCmdSetColorBlendEnableEXT(cb, 0, (uint32_t)P.ncolor, ben);
        if (g_vkp_dyn.blend_eq && !ds_same(cb, DS_BEQ, &h_bq, &bq, sizeof bq)) vkCmdSetColorBlendEquationEXT(cb, 0, (uint32_t)P.ncolor, beq);
        if (g_vkp_dyn.write_mask && !ds_same(cb, DS_WM, &h_bw, &bw, sizeof bw)) vkCmdSetColorWriteMaskEXT(cb, 0, (uint32_t)P.ncolor, wm);
    }
    if (g_vkp_dyn.logic_op_enable) DS_U(cb, DS_LOGIC_EN, logic_en, logic, vkCmdSetLogicOpEnableEXT(cb, (VkBool32)v_));
    if (g_vkp_dyn.logic_op) DS_U(cb, DS_LOGIC, logic, logic ? vkp_logic_op(S->color.logic_op) : VK_LOGIC_OP_COPY, vkCmdSetLogicOpEXT(cb, (VkLogicOp)v_));
    if (g_vkp_dyn.a2c) DS_U(cb, DS_A2C, a2c, S->ms.a2c != 0, vkCmdSetAlphaToCoverageEnableEXT(cb, (VkBool32)v_));
    if (g_vkp_dyn.polygon_mode) DS_U(cb, DS_POLY, poly, vkp_polygon_mode(S->poly.mode), vkCmdSetPolygonModeEXT(cb, (VkPolygonMode)v_));
    if (g_vkp_dyn.depth_clamp) DS_U(cb, DS_DCLAMP, dclamp, S->depth_clamp != 0 && g_vk.f10.depthClamp, vkCmdSetDepthClampEnableEXT(cb, (VkBool32)v_));
    bind_vertex_buffers(cb, used, use_zero);
    // the draw's base instance for c0 + 0x644 (an indirect draw's from its record in guest memory)
    g_draw_binst = (uint32_t)(dw ? dw->base_instance : di->base_instance);
    if (ind) { uint32_t bi; if (read_u32(ind + (dw ? 12 : 16), &bi)) g_draw_binst = bi; }
    int swrites = 0;
    for (int s = 0; s < 5; s++) if (sh[s]) {
        VkDescriptorSet ds = write_set(sh[s], s, &swrites);
        if (!ds) { g_st.draws_skipped++; return; }
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, (uint32_t)sh[s]->set, 1, &ds, 0, NULL);
    }
    if (g_gt_on) snprintf(g_gt_fs, sizeof g_gt_fs, "%s", sh[1] ? sh[1]->key : "-");
    if (g_pd) {
        char b[400]; int o = 0;
        for (int i = 0; i < P.ncolor; i++) o += snprintf(b + o, sizeof b - o, " a%d{en%d wm%x %d,%d,%d/%d,%d,%d raw%#llx}", i, (int)ben[i], (unsigned)wm[i], beq[i].srcColorBlendFactor, beq[i].dstColorBlendFactor, beq[i].colorBlendOp,
                                                         beq[i].srcAlphaBlendFactor, beq[i].dstAlphaBlendFactor, beq[i].alphaBlendOp, (unsigned long long)*(const uint64_t *)&S->blend[i]);
        PD("draw vs=%s fs=%s prim=%d %s count=%d inst=%u first=%d base_v=%d ind=%#llx | depth t%d w%d f%d stencil %d cull %d front %d | vp %g,%g %gx%g sc %d,%d %ux%u |%s",
           sh[0] ? sh[0]->key : sh[4]->key, sh[1] ? sh[1]->key : "-", prim, di ? (di->itype == 0 ? "u8" : di->itype == 1 ? "u16" : "u32") : "-", dw ? dw->count : di->count,
           (unsigned)((dw ? dw->instances : di->instances) > 0 ? (dw ? dw->instances : di->instances) : 1), dw ? dw->first : 0, di ? di->base_vertex : 0, (unsigned long long)ind,
           P.hd && S->ds.dt, P.hd && S->ds.dw, (int)S->ds.df, P.hs && S->ds.st, (int)S->poly.cull, (int)S->poly.front, vp.x, vp.y, vp.width, vp.height, sc.offset.x, sc.offset.y, sc.extent.width, sc.extent.height, b);
        for (int s = 0; s < 5; s++) if (sh[s]) for (int i = 0; i < sh[s]->nb; i++) {
            const vks_bind *bb = &sh[s]->b[i];
            if (bb->kind == VKB_TEX || bb->kind == VKB_TBUF || bb->kind == VKB_IMG) { uint64_t h = bb->kind == VKB_IMG ? S->img[s][bb->reg] : tex_slot_h(s, (unsigned)bb->reg); vkt_reg r; int tid = (int)(h & 0xfffff);
                int ok = tid > 0 && vkt_reg_copy(tid, &r); vkt_image *im = ok ? vkt_by_serial(r.t.serial) : NULL;
                samp_par sp = {0}; int sok = vkt_samp_copy((int)((h >> 20) & 0xfff), &sp);
                if (sok) PD("    sampler min%d mag%d wrap%d,%d,%d lod %g..%g bias %g aniso %g cmp %d/%d red %d", sp.minf, sp.magf, sp.ws, sp.wt, sp.wr, sp.lod_min, sp.lod_max, sp.lod_bias, sp.aniso, sp.cmp_mode, sp.cmp_func, sp.reduction);
                PD("  st%d b%u %s slot%d tid=%d sid=%d %s ser%llu fmt%#x %ux%ux%u tgt%d lv%u view%d/%d %s", s, bb->binding, bb->kind == VKB_TEX ? "tex" : bb->kind == VKB_TBUF ? "tbuf" : "img", bb->reg, tid, (int)((h >> 20) & 0xfff),
                   ok ? "" : "(none)", ok ? (unsigned long long)r.t.serial : 0ull, ok ? r.t.d.format : 0, ok ? r.t.d.width : 0, ok ? r.t.d.height : 0, ok ? r.t.d.depth : 0, ok ? r.t.d.target : 0, ok ? r.t.d.levels : 0,
                   (int)bb->vtype, ok && r.has_view ? (int)r.v.target : -1, im ? (im->cls == VKT_STORAGE ? "storage" : "gpu") : "-");
                if (ok) PD("    storage pool %p +%#llx mem %p flags %#x layout %d size %#llx image %p", (const void *)r.t.pool, (unsigned long long)r.t.off, (const void *)r.t.mem, r.t.d.flags, r.t.has_layout, (unsigned long long)r.t.L.size, (void *)im); }
            else if (bb->kind == VKB_CBUF || bb->kind == VKB_SBUF) { const p_bind_buf *u = bb->kind == VKB_CBUF ? &S->ubo[s][bb->reg] : &S->ssbo[s][bb->reg];
                float f[8] = {0}; if (u->addr && u->size >= 32) memcpy(f, (const void *)(uintptr_t)u->addr, 32); else if (u->addr && u->size >= 16) memcpy(f, (const void *)(uintptr_t)u->addr, 16);
                PD("  st%d b%u %s slot%d addr=%#llx size=%llu first %g,%g,%g,%g %g,%g,%g,%g", s, bb->binding, bb->kind == VKB_CBUF ? "cbuf" : "sbuf", bb->reg, (unsigned long long)u->addr, (unsigned long long)u->size, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]); }
        }
        if (di && di->count > 0 && di->count <= 12) { char t[200]; int o = 0; int isz = di->itype == 0 ? 1 : di->itype == 1 ? 2 : 4;
            for (int k = 0; k < di->count; k++) { uint32_t x = 0; memcpy(&x, (const uint8_t *)(uintptr_t)di->ibuf + (size_t)k * isz, (size_t)isz); o += snprintf(t + o, sizeof t - o, " %u", x); }
            PD("  indices at %#llx:%s", (unsigned long long)di->ibuf, t); }
        if (sh[0] && S->nattr && S->attr[0].fmt == 0x22) { int st = S->attr[0].stream; const uint8_t *vb = (const uint8_t *)(uintptr_t)S->vbuf[st].address; char t[300]; int o = 0;
            for (int k = 0; k < 4 && vb && (uint64_t)(k * S->stream[st].stride + 12) <= S->vbuf[st].size; k++) { float f[3]; memcpy(f, vb + k * S->stream[st].stride + S->attr[0].offset, 12); o += snprintf(t + o, sizeof t - o, " (%g,%g,%g)", f[0], f[1], f[2]); }
            PD("  positions:%s", t); }
        if (sh[0]) for (int i = 0; i < sh[0]->nin; i++) { int a = sh[0]->in[i].loc; if (a >= S->nattr) continue;
            int st = S->attr[a].stream & 15; uint32_t v0 = 0; char hx[40] = "-";   // the first vertex's bytes of the attribute
            if (di && di->ibuf && !di->ind && di->count > 0) { int isz = di->itype == 0 ? 1 : di->itype == 1 ? 2 : 4; memcpy(&v0, (const uint8_t *)(uintptr_t)di->ibuf, (size_t)isz); v0 += (uint32_t)di->base_vertex; }
            else if (dw) v0 = (uint32_t)dw->first;
            uint64_t at = (uint64_t)v0 * S->stream[st].stride + S->attr[a].offset;
            if (S->vbuf[st].address && at + 16 <= S->vbuf[st].size) { const uint8_t *p = (const uint8_t *)(uintptr_t)S->vbuf[st].address + at; for (int k = 0; k < 16; k++) snprintf(hx + 2 * k, 3, "%02x", p[k]); }
            PD("  in%d fmt%#x off%u stream%u stride%u div%u vb=%#llx/%llu ct%d v%u %s", a, S->attr[a].fmt, S->attr[a].offset, S->attr[a].stream,
            S->stream[st].stride, S->stream[st].divisor, (unsigned long long)S->vbuf[st].address, (unsigned long long)S->vbuf[st].size, sh[0]->in[i].ct, v0, hx); }
    }
    if (vkq_on()) vkq_close(cb);
    if (S->vis_on && P.query < 0) { int q = q_alloc(); if (q >= 0) { vkCmdBeginQuery(cb, R0.qp, (uint32_t)q, 0); P.query = q; g_st.queries++; } }
    // the draw
    uint32_t inst = (uint32_t)((dw ? dw->instances : di->instances) > 0 ? (dw ? dw->instances : di->instances) : 1);
    if (g_vk.has.checkpoints >= 2) { char ck[48]; const vks_shader *vt = sh[0] ? sh[0] : sh[4];   // the pixel shader, and the vertex (or mesh) shader in brackets
        snprintf(ck, sizeof ck, "%s%s%.16s%s", sh[1] ? sh[1]->key : "", sh[1] && vt ? " (vs " : "", vt ? vt->key : "", sh[1] && vt ? ")" : "");
        uint64_t cks = vk_checkpoint(cb, 'd', ck, inst, 0, 0, ind != 0);
        if (cks && sh[0]) { char nb[300]; snprintf(nb, sizeof nb, "%s; %s %u from %d", g_vsnote, di ? "indexed, base vertex" : "vertices", di ? (unsigned)di->count : dw ? (unsigned)dw->count : 0u, di ? (int)di->base_vertex : dw ? (int)dw->first : 0); vk_ck_note(cks, nb); } }
    if (fr) { fr->flags |= FL_DRAWN; fr->binst = g_draw_binst; if (ind) vkf_note_args(fr, ind, dw ? 16 : 20, mdi); g_pv_ps = fr->ps;
        if (vkw_wants(fr->ps) || vkw_wants(fr->vs)) for (int s = 0; s < 2; s++) if (sh[s]) for (int i = 0; i < sh[s]->nb; i++) { const vks_bind *bb = &sh[s]->b[i];
            if (bb->kind == VKB_CBUF && S->ubo[s][bb->reg].addr) vkw_note(cb, fr->vs, s, bb->reg, S->ubo[s][bb->reg].addr, (uint32_t)S->ubo[s][bb->reg].size);
            if (bb->kind == VKB_SBUF && bb->reg >= 0 && bb->reg < 16 && S->ssbo[s][bb->reg].addr) vkw_note(cb, fr->vs, s, 100 + bb->reg, S->ssbo[s][bb->reg].addr, (uint32_t)S->ssbo[s][bb->reg].size); } }
    if (vkq_on() && fr && !vkq_resetting()) vkq_begin(cb, fr->vs, fr->ps, dw ? (uint32_t)dw->count : (uint32_t)di->count, di ? di->base_vertex : dw->first, di ? di->ibuf : 0, (uint32_t)P_index(), fr->rt, P.query < 0);
    // NXRT_VK_REPEAT=<vertex shader key>:<n> (test, the GPU freeze): that shader's direct draws are recorded n times in a row, same state, so a fault that
    // depends on one draw comes n times sooner (the picture changes where they blend)
    int rep = 1;
    { static char rk[17]; static int rn = -1; if (rn < 0) { const char *e = getenv("NXRT_VK_REPEAT"); rn = 0; if (e && strlen(e) > 17 && e[16] == ':') { memcpy(rk, e, 16); rk[16] = 0; rn = atoi(e + 17); } }
      if (rn > 1 && sh[0] && !strcmp(sh[0]->key, rk)) rep = rn; }
    // NXRT_VK_DRAWLOG=KEY[,KEY...] (test, up to 8 shader keys): the first draw whose vertex or pixel shader has one of these keys is logged with its instances,
    // vertices and vertex-stage bindings, and a screenshot of that frame goes beside the log; then how many draws of it in each 5 s that had any.
    // Finds when in a run a shader starts to be drawn
    { static uint64_t dn[8], dt[8], dfirst[8]; char (*dl)[17] = g_dl;
      int k = dl_find(sh[0] ? sh[0]->key : NULL); if (k < 0) k = dl_find(sh[1] ? sh[1]->key : NULL);
      if (k >= 0) {
          uint64_t now = now_ns();
          if (!dfirst[k]++) { VKLOG("drawlog: first draw of %s at frame %llu (vs %s, ps %s), %u instances, %u %s; bound: %s", dl[k], (unsigned long long)g_dl_frame, sh[0] ? sh[0]->key : "-", sh[1] ? sh[1]->key : "-",
                                     inst, di ? (unsigned)di->count : dw ? (unsigned)dw->count : 0u, di ? "indices" : "vertices", g_vsnote); vks_shot_request(""); dt[k] = now;
              char vi[1600]; size_t n = 0; n += (size_t)snprintf(vi + n, sizeof vi - n, "topology %d, expand %d, first %d, base instance %u; attributes", (int)topo, (int)expand, dw ? (int)dw->first : di ? (int)di->base_vertex : 0, g_draw_binst);
              for (uint32_t i = 0; i < g_lna && n < sizeof vi - 64; i++) n += (size_t)snprintf(vi + n, sizeof vi - n, " a%u:s%u fmt%d +%u", g_lva[i].location, g_lva[i].binding, (int)g_lva[i].format, g_lva[i].offset);
              n += (size_t)snprintf(vi + n, sizeof vi - n, "; streams");
              for (uint32_t i = 0; i < g_lnb && n < sizeof vi - 96; i++) { uint32_t s = g_lvb[i].binding; n += (size_t)snprintf(vi + n, sizeof vi - n, " s%u stride %u %s div %u range %#llx+%#llx", s, g_lvb[i].stride,
                  g_lvb[i].inputRate == VK_VERTEX_INPUT_RATE_INSTANCE ? "inst" : "vert", g_lvb[i].divisor, s < 16 ? (unsigned long long)S->vbuf[s].address : 0ull, s < 16 ? (unsigned long long)S->vbuf[s].size : 0ull); }
              VKLOG("drawlog: %s vertex input: %s", dl[k], vi); }
          dn[k]++; if (now - dt[k] > 5000000000ull) { VKLOG("drawlog: %s drawn %llu times in %.1f s (frame %llu)", dl[k], (unsigned long long)dn[k], (now - dt[k]) / 1e9, (unsigned long long)g_dl_frame); dn[k] = 0; dt[k] = now; }
          } }
    if (mesh) {
        static uint64_t nm; if (!(nm++ & 0xfff)) VKLOG("mesh draws so far: %llu; this one %d workgroups, task shader %s, mesh shader %s, fragment shader %s", (unsigned long long)nm, dw->count, sh[3] ? sh[3]->key : "-", sh[4]->key, sh[1] ? sh[1]->key : "-");
        vkCmdDrawMeshTasksEXT(cb, (uint32_t)dw->count, 1, 1);
    }
    else if (dw) {
        if (ind && expand == VKP_EXPAND_QUADS) {   // no indirect form of the expansion: the arguments as guest memory holds them now (as the indexed count below; M6, gameplay)
            uint32_t a[4] = {0, 1, 0, 0}; for (int k = 0; k < 4; k++) if (!read_u32(ind + 4 * (uint64_t)k, &a[k])) { RED_ONCE(58, "indirect quads whose arguments cannot be read: not drawn"); return; }
            RED_ONCE(57, "indirect quads: drawn with the arguments guest memory holds when they are recorded (%u vertices, %u instances)", a[0], a[1]);
            if (a[1] > 65536) { RED_ONCE(56, "indirect quads with %u instances in guest memory (a count the GPU had not written yet?): not drawn", a[1]); return; }   // #17: a stale count must not keep the GPU busy past Windows' timeout
            if (a[0] >= 4 && a[1]) { vkCmdBindIndexBuffer(cb, R0.quads, 0, VK_INDEX_TYPE_UINT32); vkCmdDrawIndexed(cb, (a[0] / 4 < 65536 ? a[0] / 4 : 65536) * 6, a[1], 0, (int32_t)a[2], a[3]); }
        }
        else if (ind) { VkBuffer b; VkDeviceSize o;
            if (!vkm_use(ind, 16, &b, &o)) { RED_ONCE(59, "indirect draw arguments at %#llx are in no memory pool", (unsigned long long)ind); return; }
            vkCmdDrawIndirect(cb, b, o, 1, 16); }
        else if (expand == VKP_EXPAND_QUADS) { if (dw->count >= 4) { vkCmdBindIndexBuffer(cb, R0.quads, 0, VK_INDEX_TYPE_UINT32); vkCmdDrawIndexed(cb, (uint32_t)(dw->count / 4) * 6, inst, 0, dw->first, (uint32_t)dw->base_instance); } }
        else if (dw->count > 0) for (int r = 0; r < rep; r++) vkCmdDraw(cb, (uint32_t)dw->count, inst, (uint32_t)dw->first, (uint32_t)dw->base_instance);
    } else {
        int isz = di->itype == NVN_INDEX_TYPE_UNSIGNED_BYTE ? 1 : di->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 4;
        VkIndexType it = isz == 1 ? VK_INDEX_TYPE_UINT8_EXT : isz == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
        if (isz == 1 && !g_vk.has.index_u8) { RED_ONCE(60, "8-bit indices without VK_EXT_index_type_uint8: not drawn"); return; }
        VkBuffer b; VkDeviceSize o;
        if (expand == VKP_EXPAND_QUADS) {   // indexed quads: expanded from guest memory into a triangle list (nvn_mtl.m nvn_expand_primitive)
            if (ind || di->count < 4) { if (ind) RED_ONCE(61, "indirect indexed quads are not drawn"); return; }
            uint32_t nq = (uint32_t)di->count / 4; uint32_t *d = arena((VkDeviceSize)nq * 24, 16, &b, &o); if (!d) return;
            const uint8_t *src = (const uint8_t *)(uintptr_t)di->ibuf;
            for (uint32_t q = 0; q < nq; q++) { uint32_t v[4]; for (int j = 0; j < 4; j++) { uint32_t x = 0; memcpy(&x, src + (size_t)(4 * q + j) * isz, (size_t)isz); v[j] = x; }
                d[6 * q] = v[0]; d[6 * q + 1] = v[1]; d[6 * q + 2] = v[2]; d[6 * q + 3] = v[0]; d[6 * q + 4] = v[2]; d[6 * q + 5] = v[3]; }
            vkCmdBindIndexBuffer(cb, b, o, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cb, nq * 6, inst, 0, di->base_vertex, (uint32_t)di->base_instance);
        } else {
            uint32_t icount = (uint32_t)(di->count > 0 ? di->count : 1);
            if (ind && !read_u32(ind, &icount)) icount = 4096;   // an indirect draw's index count, as guest memory holds it now
            uint64_t n = (uint64_t)(icount ? icount : 1) * isz;
            if (!vkm_use(di->ibuf, n, &b, &o) && !vkm_use(di->ibuf, (uint64_t)isz, &b, &o)) { RED_ONCE(62, "index buffer at %#llx is in no memory pool", (unsigned long long)di->ibuf); return; }
            if (o % isz) { RED_ONCE(63, "index buffer at an address not aligned to its index size: not drawn"); return; }
            vkCmdBindIndexBuffer(cb, b, o, it);
            if (ind) { VkBuffer ab; VkDeviceSize ao; if (!vkm_use(ind, 20, &ab, &ao)) { RED_ONCE(59, "indirect draw arguments are in no memory pool"); return; } vkCmdDrawIndexedIndirect(cb, ab, ao, 1, 20); }
            else if (di->count > 0) for (int r = 0; r < rep; r++) vkCmdDrawIndexed(cb, (uint32_t)di->count, inst, 0, di->base_vertex, (uint32_t)di->base_instance);
        }
    }
    g_st.draws++;
    (void)swrites;
    if (g_nwb >= 240) pass_end();   // the write-back list is nearly full: record the copies now
}

// ---------- Compute (nvn_mtl.m dispatch) ----------
static void dispatch(const p_dispatch *dp) {
    const vks_shader *cs = S->cprog ? prog_stage(S->cprog, VKS_CS) : NULL;
    if (!cs) { g_st.skip_no_shader++; return; }
    VkPipelineLayout layout; VkPipeline pipe = vkp_compute(cs, &layout);
    if (!pipe) return;
    pass_end();
    if (dp->ind || (dp->x && dp->y && dp->z)) for (int i = 0; i < cs->nfirst; i++) { const p_bind_buf *u = &S->ssbo[5][cs->first[i].reg]; if (u->addr) vkr_fill(u->addr + 4ull * cs->first[i].word, 4, cs->first[i].value); }   // the shader's first invocation's stores, in order (nxs_spv.py FirstStore)
    VkCommandBuffer cb = vks_cb();
    vkf_args_flush(cb);
    { vkf_row *fr = vkf_row_next(); if (fr) { fr->vs = vkf_key(cs->key); fr->flags = FL_DISPATCH | FL_DRAWN | (dp->ind ? FL_INDIRECT : 0); fr->pass = (uint32_t)g_st.passes; fr->a[0] = (uint32_t)dp->x; fr->a[1] = (uint32_t)dp->y; fr->a[2] = (uint32_t)dp->z; } if (vkc_hidden(vkf_key(cs->key))) { if (fr) fr->flags = (fr->flags | FL_HIDDEN) & ~(uint32_t)FL_DRAWN; return; } }
    g_dispatch_dim[0] = (uint32_t)dp->x; g_dispatch_dim[1] = (uint32_t)dp->y; g_dispatch_dim[2] = (uint32_t)dp->z;
    if (dp->ind) for (int i = 0; i < 3; i++) read_u32(dp->ind + 4 * i, &g_dispatch_dim[i]);   // for c0's dispatch words: the record as guest memory holds it now
    int swrites = 0;
    VkDescriptorSet ds = write_set(cs, 5, &swrites);
    if (!ds) return;
    barrier_all(cb);
    for (int i = 0; i < g_nwb; i++) g_wb[i].base = vkm_wb_base(g_wb[i].a, g_wb[i].n);   // the ranges before the shader: only the words it changes go back to guest memory
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); g_last_pipe = VK_NULL_HANDLE;
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &ds, 0, NULL);
    if (g_pd) {
        PD("dispatch cs=%s %ux%ux%u ind=%#llx local %ux%ux%u", cs->key, g_dispatch_dim[0], g_dispatch_dim[1], g_dispatch_dim[2], (unsigned long long)dp->ind, cs->local[0], cs->local[1], cs->local[2]);
        for (int i = 0; i < cs->nb; i++) { const vks_bind *bb = &cs->b[i];
            if (bb->kind == VKB_SBUF || bb->kind == VKB_CBUF) { const p_bind_buf *u = bb->kind == VKB_CBUF ? &S->ubo[5][bb->reg] : &S->ssbo[5][bb->reg]; PD("  b%u %s slot%d addr=%#llx size=%llu rw%d", bb->binding, bb->kind == VKB_CBUF ? "cbuf" : "sbuf", bb->reg, (unsigned long long)u->addr, (unsigned long long)u->size, bb->rw); }
            else if (bb->kind == VKB_TEX || bb->kind == VKB_IMG || bb->kind == VKB_TBUF) { uint64_t h = bb->kind == VKB_IMG ? S->img[5][bb->reg] : tex_slot_h(5, (unsigned)bb->reg); vkt_reg r; int tid = (int)(h & 0xfffff); int ok = tid > 0 && vkt_reg_copy(tid, &r);
                PD("  b%u %s slot%d tid=%d ser%llu fmt%#x %ux%u tgt%d", bb->binding, bb->kind == VKB_TEX ? "tex" : bb->kind == VKB_IMG ? "img" : "tbuf", bb->reg, tid, ok ? (unsigned long long)r.t.serial : 0ull, ok ? r.t.d.format : 0, ok ? r.t.d.width : 0, ok ? r.t.d.height : 0, ok ? r.t.d.target : 0); }
            else if (bb->kind == VKB_DRV) PD("  b%u drv c%d read %d (s2 %d ssbo %u tex %u n %u)", bb->binding, bb->bank, cs->drv_read[bb->bank], cs->s2cs, cs->s2_ssbo_base, cs->s2_tex_base, cs->s2_ntex);
        }
    }
    vk_checkpoint(cb, 'c', cs->key, g_dispatch_dim[0], g_dispatch_dim[1], g_dispatch_dim[2], dp->ind != 0);
    if (dp->ind) { VkBuffer b; VkDeviceSize o; if (!vkm_use(dp->ind, 12, &b, &o)) { RED_ONCE(70, "indirect dispatch arguments are in no memory pool"); return; } vkCmdDispatchIndirect(cb, b, o); }
    else if (dp->x > 0 && dp->y > 0 && dp->z > 0) vkCmdDispatch(cb, (uint32_t)dp->x, (uint32_t)dp->y, (uint32_t)dp->z);
    if (g_gt_on) gt_mark(cb, 'd', g_dispatch_dim[0], g_dispatch_dim[1], 0, cs->key);
    g_st.dispatches++;
    if (g_nbnc) {   // written bounced storage buffers back into their pools, before the write-back to guest memory below reads them
        barrier_all(cb);
        for (int i = 0; i < g_nbnc; i++) { VkBufferCopy c = {g_bnc[i].boff, g_bnc[i].poff, g_bnc[i].n}; vkCmdCopyBuffer(cb, R0.bounce, g_bnc[i].pool, 1, &c); }
        g_nbnc = 0;
        if (!g_nwb) barrier_all(cb);
    }
    if (g_nwb) { barrier_all(cb); wb_flush(); }
    if (vkv_on()) for (int i = 0; i < cs->nb; i++) { const vks_bind *bb = &cs->b[i]; vkt_reg rg;
        if (bb->kind == VKB_IMG && bb->reg >= 0 && bb->reg < 32 && vkt_reg_copy((int)(S->img[5][bb->reg] & 0xfffff), &rg)) { vkt_image *im = vkt_by_serial(rg.t.serial);
            if (im) { uint32_t lv = rg.has_view && rg.v.set_levels ? rg.v.base_level : 0, ly = rg.has_view && rg.v.set_layers ? rg.v.min_layer : 0; vkv_image(cb, im, lv, ly, mipd(im->w, lv), mipd(im->h, lv), bb->binding, vkf_key(cs->key), 1); } } }
}

// ---------- Buffer operations in GPU order (the null backend's and track B's effects, now inside the replay's ordering) ----------
void vkr_copy(uint64_t src, uint64_t dst, uint64_t n) {
    g_st.copies++;
    VkBuffer sb, db; VkDeviceSize so, dof;
    VkCommandBuffer cb = vks_cb();
    if (!cb || !n || !vkm_use(src, n, &sb, &so) || !vkm_use(dst, n, &db, &dof)) {   // not two pool ranges: on the CPU, as the null backend does it
        if (vkm_addr(src, n, NULL, NULL) && vkm_addr(dst, n, NULL, NULL)) { dm_guest_write_prepare((void *)(uintptr_t)dst, (size_t)n); memmove((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, (size_t)n); }
        return;
    }
    pass_end(); barrier_all(cb);
    VkBufferCopy c = {so, dof, n};
    vkCmdCopyBuffer(cb, sb, db, 1, &c);
    vkm_gpu_wrote(dst, n);
}
void vkr_fill(uint64_t dst, uint64_t n, uint32_t value) {
    g_st.fills++;
    VkBuffer db; VkDeviceSize dof; VkCommandBuffer cb = vks_cb();
    if (!n) return;
    if (!cb || (dst & 3) || (n & 3) || !vkm_use(dst, n, &db, &dof)) {
        if (!vkm_addr(dst, n, NULL, NULL)) return;
        dm_guest_write_prepare((void *)(uintptr_t)dst, (size_t)n);
        uint8_t *p = (uint8_t *)(uintptr_t)dst; for (uint64_t i = 0; i < n; i++) p[i] = (uint8_t)(value >> 8 * (i & 3));
        return;
    }
    pass_end(); barrier_all(cb);
    vkCmdFillBuffer(cb, db, dof, n, value);
    vkm_cpu_also(dst, n, NULL, 1, value);
}
void vkr_update(uint64_t d, uint64_t n, const uint8_t *data) {
    g_st.updates++;
    VkBuffer db; VkDeviceSize dof; VkCommandBuffer cb = vks_cb();
    if (!n) return;
    if (!cb || !vkm_use(d, n, &db, &dof)) {
        if (vkm_addr(d, n, NULL, NULL)) { dm_guest_write_prepare((void *)(uintptr_t)d, (size_t)n); memcpy((void *)(uintptr_t)d, data, (size_t)n); }
        return;
    }
    pass_end(); barrier_all(cb);
    if (!(dof & 3) && !(n & 3) && n <= 65536) vkCmdUpdateBuffer(cb, db, dof, n, data);   // inline in the command buffer
    else { VkBuffer sb; VkDeviceSize so; uint8_t *p = vkm_staging(n, 16, &sb, &so); if (p) { memcpy(p, data, (size_t)n); VkBufferCopy c = {so, dof, n}; vkCmdCopyBuffer(cb, sb, db, 1, &c); } }
    vkm_cpu_also(d, n, data, 0, 0);
}

// ---------- Texture copies and clears (nvn_mtl.m copy_b2t, copy_t2b, copy_t2t, clear_tex) ----------
static const tex_t *snap(const tex_t *t) { return t && t->magic == MAGIC('T', 'E', 'X', 0) && t->serial ? t : NULL; }
// A transfer on an image: GENERAL images stay so; a sampled-only image goes to the transfer layout and back to its resting one
static VkImageLayout xfer_in(VkCommandBuffer cb, vkt_image *im, int dst) {
    if (im->rest == VK_IMAGE_LAYOUT_GENERAL) return VK_IMAGE_LAYOUT_GENERAL;
    VkImageLayout l = dst ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    vk_image_barrier(cb, im->img, im->aspect, im->rest, l, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                     dst ? VK_ACCESS_2_TRANSFER_WRITE_BIT : VK_ACCESS_2_TRANSFER_READ_BIT);
    return l;
}
static void xfer_out(VkCommandBuffer cb, vkt_image *im, VkImageLayout l) {
    if (l == im->rest) return;
    vk_image_barrier(cb, im->img, im->aspect, l, im->rest, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
}
static VkImageAspectFlags copy_aspect(vkt_image *im) { return im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : im->aspect; }
// The buffer side of a copy: SetCopyRowStride / SetCopyImageStride, else tightly packed (nvn_mtl.m copy_b2t)
static int copy_layout(const vkt_image *im, const NVNcopyRegion *r, size_t *row, size_t *img, uint32_t *row_texels, uint32_t *img_rows) {
    const vkf_info *fi = im->fi;
    size_t rows = (size_t)(r->height + fi->bh - 1) / fi->bh;
    *row = S->copy_stride ? (size_t)S->copy_stride : (size_t)((r->width + fi->bw - 1) / fi->bw) * fi->bpb;
    *img = S->copy_img_stride ? (size_t)S->copy_img_stride : *row * rows;
    if (*row % fi->bpb || (*row && *img % *row)) return 0;
    *row_texels = (uint32_t)(*row / fi->bpb * fi->bw); *img_rows = (uint32_t)(*row ? *img / *row * fi->bh : 0);
    return 1;
}
static void copy_b2t(const p_copy_b2t *c) {
    const tex_t *t = snap(c->tex); if (!t) return;
    PD("copy buffer %#llx -> ser%llu fmt%#x region %d,%d,%d %dx%dx%d", (unsigned long long)c->src, (unsigned long long)t->serial, t->d.format, c->r.xoffset, c->r.yoffset, c->r.zoffset, c->r.width, c->r.height, c->r.depth);
    vkt_image *im = vkt_resolve(t); if (!im) return;
    g_st.tex_copies++;
    if (im->fi->conv != VKF_COPY) { RED_ONCE(80, "buffer→texture copy into a format the device stores converted (%s): not copied", im->fi->name); return; }
    unsigned level = c->has_view && c->view.set_levels ? c->view.base_level : 0, layer0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    if (level >= im->levels) return;
    size_t row, img; uint32_t rt, ir;
    if (!copy_layout(im, &c->r, &row, &img, &rt, &ir)) { RED_ONCE(81, "buffer→texture copy with strides Vulkan cannot express"); return; }
    int is3d = t->d.target == NVN_TEXTURE_TARGET_3D, nz = c->r.depth > 0 ? c->r.depth : 1;
    VkBuffer b; VkDeviceSize o;
    if (!vkm_use(c->src, img * nz, &b, &o)) { RED_ONCE(82, "buffer→texture copy source is in no memory pool"); return; }
    // a storage-backed texture gets the bytes in its storage too (hle_nvn.c nvn_tex_mirror_b2t), or the next upload from there would drop them
    if (im->cls == VKT_STORAGE && !is3d && nvn_tex_mirror_b2t(t, level, layer0, &c->r, (const uint8_t *)(uintptr_t)c->src, row, img)) RED_ONCE(83, "a buffer→texture copy into a storage-backed texture could not be written into its storage");
    pass_end();
    VkCommandBuffer cb = vks_cb(); barrier_all(cb);
    VkImageLayout l = xfer_in(cb, im, 1);
    VkBufferImageCopy rg[64]; int n = 0;
    for (int z = 0; z < (is3d ? 1 : nz) && n < 64; z++)
        rg[n++] = (VkBufferImageCopy){o + (VkDeviceSize)z * img, rt, ir, {copy_aspect(im), level, is3d ? 0 : layer0 + c->r.zoffset + z, 1},
                                      {c->r.xoffset, c->r.yoffset, is3d ? c->r.zoffset : 0}, {(uint32_t)c->r.width, (uint32_t)c->r.height, is3d ? (uint32_t)nz : 1}};
    vkCmdCopyBufferToImage(cb, b, im->img, l, (uint32_t)n, rg);
    xfer_out(cb, im, l);
    im->content = 1;
}
static void copy_t2b(const p_copy_b2t *c) {
    const tex_t *t = snap(c->tex); if (!t) return;
    vkt_image *im = vkt_resolve(t); if (!im) return;
    g_st.tex_copies++;
    if (im->fi->conv != VKF_COPY) { RED_ONCE(84, "texture→buffer copy from a format the device stores converted (%s): not copied", im->fi->name); return; }
    unsigned level = c->has_view && c->view.set_levels ? c->view.base_level : 0, layer0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    if (level >= im->levels) return;
    size_t row, img; uint32_t rt, ir;
    if (!copy_layout(im, &c->r, &row, &img, &rt, &ir)) { RED_ONCE(85, "texture→buffer copy with strides Vulkan cannot express"); return; }
    int is3d = t->d.target == NVN_TEXTURE_TARGET_3D, nz = c->r.depth > 0 ? c->r.depth : 1;
    VkBuffer b; VkDeviceSize o;
    if (!vkm_use(c->src, img * nz, &b, &o)) { RED_ONCE(86, "texture→buffer copy destination is in no memory pool"); return; }
    pass_end();
    VkCommandBuffer cb = vks_cb(); barrier_all(cb);
    VkImageLayout l = xfer_in(cb, im, 0);
    VkBufferImageCopy rg[64]; int n = 0;
    for (int z = 0; z < (is3d ? 1 : nz) && n < 64; z++)
        rg[n++] = (VkBufferImageCopy){o + (VkDeviceSize)z * img, rt, ir, {copy_aspect(im), level, is3d ? 0 : layer0 + c->r.zoffset + z, 1},
                                      {c->r.xoffset, c->r.yoffset, is3d ? c->r.zoffset : 0}, {(uint32_t)c->r.width, (uint32_t)c->r.height, is3d ? (uint32_t)nz : 1}};
    vkCmdCopyImageToBuffer(cb, im->img, l, b, (uint32_t)n, rg);
    xfer_out(cb, im, l);
    vkm_gpu_wrote(c->src, img * nz);   // NVN pools are coherent with the CPU: the bytes reach guest memory when the submission completes
}
// M6 phase B: a GPU copy into a texture whose content comes from its storage (VKT_STORAGE), from a source whose content is the GPU's: the destination region is read back
// after the copy into the submission's arena and written into the storage when the submission completes (nvn_mtl.m mirror_readback / mirror_rb_land), so the next upload
// from there keeps the copy. The texture record is copied (the game may finalize it once it has waited)
typedef struct { tex_t t; unsigned level, layer0; NVNcopyRegion r; size_t row, img; const uint8_t *buf; } mirror_rb;
static void mirror_land(void *p) {
    mirror_rb *m = p;
    if (nvn_tex_mirror_b2t(&m->t, m->level, m->layer0, &m->r, m->buf, m->row, m->img)) RED_ONCE(96, "a GPU copy read back into the storage of texture ser=%llu could not be written there", (unsigned long long)m->t.serial);
    else { static int n; if (!n++) VKLOG("a GPU copy into a storage-backed texture (ser=%llu %dx%d): read back into its storage", (unsigned long long)m->t.serial, m->r.width, m->r.height); }
    free(m);
}
static void mirror_readback(VkCommandBuffer cb, vkt_image *im, const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r) {
    const vkf_info *fi = im->fi; int nz = r->depth > 0 ? r->depth : 1;
    if (fi->conv != VKF_COPY || im->samples > 1 || (im->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) || t->d.target == NVN_TEXTURE_TARGET_3D || r->width <= 0 || r->height <= 0) {
        RED_ONCE(95, "a GPU copy into a storage-backed texture (format %s) cannot be read back into its storage: the next upload from there drops it", fi->name); return; }
    size_t row = (size_t)((r->width + fi->bw - 1) / fi->bw) * fi->bpb, img = row * (size_t)((r->height + fi->bh - 1) / fi->bh);
    VkBuffer b; VkDeviceSize o; uint8_t *p = arena(img * nz, 16, &b, &o);
    if (!p) return;
    barrier_all(cb);   // the copy's writes first
    VkImageLayout l = xfer_in(cb, im, 0);
    VkBufferImageCopy rg[64]; int n = 0;
    for (int z = 0; z < nz && n < 64; z++)
        rg[n++] = (VkBufferImageCopy){o + (VkDeviceSize)z * img, 0, 0, {copy_aspect(im), level, layer0 + r->zoffset + z, 1}, {r->xoffset, r->yoffset, 0}, {(uint32_t)r->width, (uint32_t)r->height, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, l, b, (uint32_t)n, rg);
    xfer_out(cb, im, l);
    vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    mirror_rb *m = malloc(sizeof *m); *m = (mirror_rb){*t, level, layer0, *r, row, img, p};
    vks_post_fn(mirror_land, m);   // before the arena chunk is returned (subres_done is posted at the end of the submission)
    g_st.tex_copies++;
}
static void copy_t2t(const p_copy_t2t *c) {
    const tex_t *st = snap(c->src), *dt = snap(c->dst); if (!st || !dt) return;
    PD("copy ser%llu %dx%d -> ser%llu %dx%d flags %#x", (unsigned long long)st->serial, c->sr.width, c->sr.height, (unsigned long long)dt->serial, c->dr.width, c->dr.height, c->flags);
    vkt_image *s = vkt_resolve(st), *d = vkt_resolve(dt); if (!s || !d) return;
    g_st.tex_copies++;
    unsigned sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0;
    unsigned sz0 = c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0, dz0 = c->has_dv && c->dv.set_layers ? c->dv.min_layer : 0;
    if (sl >= s->levels || dl >= d->levels) return;
    int s3 = st->d.target == NVN_TEXTURE_TARGET_3D, d3 = dt->d.target == NVN_TEXTURE_TARGET_3D, nz = c->sr.depth > 0 ? c->sr.depth : 1;
    static int no_mir = -1; if (no_mir < 0) no_mir = getenv("NXRT_VK_NO_SPMIRROR") != NULL;
    if (d->sp && s->cls == VKT_STORAGE && !no_mir) {   // into a sparse texture from one whose bytes are in memory: the bytes go behind its tiles, and the tiles come from there
        int nvn_tex_mirror_sparse(const tex_t *s, unsigned sl, unsigned ss, const NVNcopyRegion *sr, const tex_t *d, unsigned dl, unsigned ds, const NVNcopyRegion *dr);
        void vkt_sparse_touch(vkt_image *im, unsigned level, unsigned layer, unsigned n);
        static int touch = -1; if (touch < 0) touch = getenv("NXRT_VK_SPTOUCH") != NULL;
        if (!nvn_tex_mirror_sparse(st, sl, sz0, &c->sr, dt, dl, dz0, &c->dr)) { if (touch) { vkt_sparse_touch(d, dl, dz0 + (unsigned)c->dr.zoffset, (unsigned)nz); return; } } else
        RED_ONCE(94, "a texture copy into a sparse texture could not be written through its page table: the GPU copy alone stands");
    }
    int mirror = (c->flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT)) != 0, linear = (c->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) != 0;
    int same = c->sr.width == c->dr.width && c->sr.height == c->dr.height && !mirror && s->fi->vk == d->fi->vk, rb = 0;
    // the copy's bytes in the destination's storage when it is storage-backed (hle_nvn.c nvn_tex_mirror_t2t, from the source's storage when that holds its content)
    if (d->cls == VKT_STORAGE && !d3 && same && s->cls == VKT_STORAGE && !s3 && nvn_tex_mirror_t2t(st, sl, sz0, &c->sr, dt, dl, dz0, &c->dr))
        RED_ONCE(90, "a texture copy into a storage-backed texture could not be written into its storage");
    else if (d->cls == VKT_STORAGE) rb = 1;   // the source's content is the GPU's: read back after the copy (nvn_mtl.m mirror_readback)
    pass_end();
    VkCommandBuffer cb = vks_cb(); barrier_all(cb);
    VkImageLayout ls = xfer_in(cb, s, 0), ld = s == d ? ls : xfer_in(cb, d, 1);
    if (s == d && ls != VK_IMAGE_LAYOUT_GENERAL) { RED_ONCE(92, "a texture copied into itself outside the GENERAL layout: not copied"); xfer_out(cb, s, ls); return; }
    VkImageAspectFlags sa = copy_aspect(s), da = copy_aspect(d);
    if ((c->flags & DM_COPY_RESOLVE) && s->samples > 1 && d->samples == 1 && s->fi->vk == d->fi->vk && !(s->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) {
        VkImageResolve r = {{sa, sl, sz0, 1}, {0, 0, 0}, {da, dl, dz0, 1}, {0, 0, 0}, {mipd(s->w, sl), mipd(s->h, sl), 1}};
        vkCmdResolveImage(cb, s->img, ls, d->img, ld, 1, &r);
    } else if (same && s->samples == d->samples) {
        VkImageCopy r[64]; int n = 0;
        if (s3 || d3) r[n++] = (VkImageCopy){{sa, sl, s3 ? 0 : sz0 + c->sr.zoffset, s3 ? 1 : (uint32_t)nz}, {c->sr.xoffset, c->sr.yoffset, s3 ? c->sr.zoffset : 0},
                                             {da, dl, d3 ? 0 : dz0 + c->dr.zoffset, d3 ? 1 : (uint32_t)nz}, {c->dr.xoffset, c->dr.yoffset, d3 ? c->dr.zoffset : 0}, {(uint32_t)c->sr.width, (uint32_t)c->sr.height, s3 && d3 ? (uint32_t)nz : 1}};
        else r[n++] = (VkImageCopy){{sa, sl, sz0 + c->sr.zoffset, (uint32_t)nz}, {c->sr.xoffset, c->sr.yoffset, 0}, {da, dl, dz0 + c->dr.zoffset, (uint32_t)nz}, {c->dr.xoffset, c->dr.yoffset, 0},
                                    {(uint32_t)c->sr.width, (uint32_t)c->sr.height, 1}};
        vkCmdCopyImage(cb, s->img, ls, d->img, ld, (uint32_t)n, r);
    } else if (!(s->fi->flags & VKF_COMPRESSED) && !(d->fi->flags & VKF_COMPRESSED) && s->samples == 1 && d->samples == 1 && !s3 && !d3 && (sa == da)) {
        // scaling, mirroring or a format change: a blit (nvn_mtl.m draws these: copy_t2t_draw)
        int x0 = c->dr.xoffset, y0 = c->dr.yoffset, x1 = c->dr.xoffset + c->dr.width, y1 = c->dr.yoffset + c->dr.height;
        if (c->flags & NVN_COPY_FLAGS_MIRROR_X_BIT) { int t = x0; x0 = x1; x1 = t; }
        if (c->flags & NVN_COPY_FLAGS_MIRROR_Y_BIT) { int t = y0; y0 = y1; y1 = t; }
        int depth = (sa & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0;
        int filt = linear && !depth && (fmt_features(s->fi->vk, 0) & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
        VkImageBlit r[64]; int n = 0;
        for (int z = 0; z < nz && n < 64; z++)
            r[n++] = (VkImageBlit){{sa, sl, sz0 + c->sr.zoffset + z, 1}, {{c->sr.xoffset, c->sr.yoffset, 0}, {c->sr.xoffset + c->sr.width, c->sr.yoffset + c->sr.height, 1}},
                                   {da, dl, dz0 + c->dr.zoffset + z, 1}, {{x0, y0, 0}, {x1, y1, 1}}};
        if (depth && s->fi->vk != d->fi->vk) RED_ONCE(93, "a depth copy between different formats: not copied");
        else vkCmdBlitImage(cb, s->img, ls, d->img, ld, (uint32_t)n, r, filt ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
    } else RED_ONCE(94, "texture→texture copy case not replayed: %dx%d→%dx%d flags %#x %s→%s samples %u→%u 3D %d→%d", c->sr.width, c->sr.height, c->dr.width, c->dr.height, c->flags,
                    s->fi->name, d->fi->name, (unsigned)s->samples, (unsigned)d->samples, s3, d3);
    if (s != d) xfer_out(cb, d, ld);
    xfer_out(cb, s, ls);
    d->content = 1;
    if (rb) mirror_readback(cb, d, dt, dl, dz0, &c->dr);
}
static void clear_tex(const p_clear_tex *c) {
    const tex_t *t = snap(c->tex); if (!t) return;
    vkt_image *im = vkt_resolve(t); if (!im) return;
    if (im->fi->flags & VKF_COMPRESSED) { RED_ONCE(100, "a clear of a compressed texture: not cleared"); return; }
    unsigned lv = c->has_view && c->view.set_levels ? c->view.base_level : 0, l0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    if (lv >= im->levels) return;
    uint32_t w = mipd(im->w, lv), h = mipd(im->h, lv);
    int is3d = im->vtype == VK_IMAGE_VIEW_TYPE_3D, nz = c->r.depth > 0 ? c->r.depth : 1;
    VkFormat vf = vkf_view_format(t->d.format, c->has_view && c->view.set_format ? c->view.format : t->d.format);
    int depth = (im->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0;
    unsigned ch = depth ? 0xf : fmt_channels(vf);
    if ((c->mask & ch) != ch && !depth) RED_ONCE(101, "texture clear with channel mask %#x: cleared whole (a partial mask needs a clear draw)", c->mask);
    int whole = !c->r.xoffset && !c->r.yoffset && (uint32_t)c->r.width >= w && (uint32_t)c->r.height >= h;
    g_st.clears++;
    if (!whole && (im->usage & (depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) && !is3d) {   // a region: a pass with that render area
        VkImageView v = vkt_view_raw_(im, depth ? im->fi->vk : vf, VK_IMAGE_VIEW_TYPE_2D_ARRAY, lv, 1, l0 + c->r.zoffset, (uint32_t)nz, depth ? im->aspect : VK_IMAGE_ASPECT_COLOR_BIT, NULL);
        VkRect2D r = {{c->r.xoffset, c->r.yoffset}, {(uint32_t)c->r.width, (uint32_t)c->r.height}};
        VkClearValue val; if (depth) { val.depthStencil.depth = c->c.f[0]; val.depthStencil.stencil = c->c.u[1]; } else val = clear_color_value(vf, c->c.f, c->c.u, c->is_uint);
        if (v) clear_pass(v, depth ? im->fi->vk : vf, depth, 1, 1, r, (uint32_t)nz, val);
        im->content = 1;
        return;
    }
    if (!whole) RED_ONCE(102, "a region clear of a texture that cannot be an attachment: the whole level is cleared");
    pass_end();
    VkCommandBuffer cb = vks_cb(); barrier_all(cb);
    VkImageLayout l = xfer_in(cb, im, 1);
    VkImageSubresourceRange rr = {im->aspect, lv, 1, is3d ? 0 : l0 + c->r.zoffset, is3d ? 1 : (uint32_t)nz};
    if (depth) { VkClearDepthStencilValue v = {c->c.f[0], c->c.u[1]}; vkCmdClearDepthStencilImage(cb, im->img, l, &v, 1, &rr); }
    else { VkClearValue v = clear_color_value(im->fi->vk, c->c.f, c->c.u, c->is_uint); vkCmdClearColorImage(cb, im->img, l, &v.color, 1, &rr); }
    xfer_out(cb, im, l);
    im->content = 1;
}

// ---------- Counters ----------
// ---------- Conditional rendering (nvn_mtl.m, profile render_cond) ----------
// SetRenderEnableConditional compares the 64-bit counters at rec and rec + 16. agl wraps its clears in this: the CPU sets rec to 1 when a target needs a clear, each clear is
// followed by ReportCounter(TIMESTAMP_TOP, rec), and a timestamp puts 0 in the counter field, so a target already cleared is not cleared again. A report replayed earlier in this
// submission is not in memory yet (it lands when the GPU completes), so it counts here with its known counter 0; a samples-passed result is known only then, and a condition on it
// keeps drawing. Without this the second clear ran and wiped the lit scene between the forward pass and the sky: the title's ground, buildings and kart came out black
static uint64_t *g_rc_pend; static uint32_t g_rc_npend, g_rc_pcap;   // timestamp reports replayed in this submission
static uint64_t *g_rc_occ; static uint32_t g_rc_ocap, g_rc_on;       // addresses ever written by another report (open addressing, a power of two)
static int rc_occ_has(uint64_t a) { if (!g_rc_ocap) return 0; for (uint32_t i = (uint32_t)(a >> 4) & (g_rc_ocap - 1);; i = (i + 1) & (g_rc_ocap - 1)) { if (g_rc_occ[i] == a) return 1; if (!g_rc_occ[i]) return 0; } }
static void rc_occ_add(uint64_t a) {
    if (rc_occ_has(a)) return;
    if ((g_rc_on + 1) * 2 > g_rc_ocap) {
        uint32_t oc = g_rc_ocap, nc = oc ? oc * 2 : 4096; uint64_t *old = g_rc_occ, *t = calloc(nc, 8);
        for (uint32_t k = 0; k < oc; k++) if (old[k]) { uint32_t i = (uint32_t)(old[k] >> 4) & (nc - 1); while (t[i]) i = (i + 1) & (nc - 1); t[i] = old[k]; }
        g_rc_occ = t; g_rc_ocap = nc; free(old);
    }
    uint32_t i = (uint32_t)(a >> 4) & (g_rc_ocap - 1); while (g_rc_occ[i]) i = (i + 1) & (g_rc_ocap - 1);
    g_rc_occ[i] = a; g_rc_on++;
}
static void rc_note(uint64_t a, int timestamp) {
    if (!timestamp) { rc_occ_add(a); return; }
    for (uint32_t i = 0; i < g_rc_npend; i++) if (g_rc_pend[i] == a) return;
    if (g_rc_npend == g_rc_pcap) { g_rc_pcap = g_rc_pcap ? g_rc_pcap * 2 : 256; g_rc_pend = realloc(g_rc_pend, (size_t)g_rc_pcap * 8); }
    g_rc_pend[g_rc_npend++] = a;
}
static int rc_value(uint64_t a, uint64_t *v) {   // 0: not known at replay
    if (rc_occ_has(a)) return 0;
    for (uint32_t i = g_rc_npend; i-- > 0;) if (g_rc_pend[i] == a) { *v = 0; return 1; }
    uint32_t lo = 0, hi = 0;
    if (!read_u32(a, &lo) || !read_u32(a + 4, &hi)) return 0;
    *v = (uint64_t)hi << 32 | lo; return 1;
}
static void report_counter(uint64_t type, uint64_t addr) {
    if (DM_GAME_RENDER_COND) rc_note(addr, type == NVN_COUNTER_TYPE_TIMESTAMP || type == NVN_COUNTER_TYPE_TIMESTAMP_TOP);
    if (type == NVN_COUNTER_TYPE_SAMPLES_PASSED) {
        if (P.query >= 0) { vkCmdEndQuery(vks_cb(), R0.qp, (uint32_t)P.query); q_post(0, (uint32_t)P.query, 0); P.query = -1; }
        q_post(2, 0, addr); return;
    }
    if (type != NVN_COUNTER_TYPE_TIMESTAMP && type != NVN_COUNTER_TYPE_TIMESTAMP_TOP) { static uint32_t seen; if (!(seen >> (type & 31) & 1)) { seen |= 1u << (type & 31); VKRED("counter type %llu is reported as 0 (only timestamps and samples passed are counted)", (unsigned long long)type); } }
    vks_post_counter(addr, (uint32_t)type);
}
static void reset_counter(uint64_t type) {
    if (type != NVN_COUNTER_TYPE_SAMPLES_PASSED) return;
    if (P.query >= 0) { vkCmdEndQuery(vks_cb(), R0.qp, (uint32_t)P.query); q_post(0, (uint32_t)P.query, 0); P.query = -1; }
    q_post(1, 0, 0); S->vis_on = 1;
}

// ---------- State records (nvn_mtl.m state_op) ----------
static void bind_handles_fresh(const p_bind_handles *b) {   // BindTexturesDeferred: the handles at src as they are now (nvn_mtl.m bh_fresh)
    if (b->defer && b->src && b->n > 0 && b->n <= 32) memcpy((void *)b->h, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n);
}
static void state_op(uint32_t op, const void *a) {
    switch (op) {
    case OP_SCISSOR: S->sc = *(const p_scissor *)a; S->has_sc = 1; break;
    case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: { int tex = op == OP_SET_TEX_POOL, b = nvn_dpool_base(((const p_set_pool *)a)->pool, tex); if (tex) S->tex_base = b; else S->samp_base = b; break; }
    case OP_BIND_PROGRAM: { const p_bind_program *bp = a;
        if (bp->stages & 0x20) S->cprog = bp->prog;
        { int bits = (bp->stages & 0x1f) | (bp->stages & 0x40 ? 8 : 0) | (bp->stages & 0x80 ? 16 : 0);   // NVN2's task and mesh bits: the tessellation stages' places
          for (int st = 0; st < 5; st++) if (bits >> st & 1) S->bst[st] = bp->prog; }
        break; }
    case OP_BIND_STATE: {
        const p_bind_state *b = a; blend_st bs;
        switch (b->kind) {
        case 0: memcpy(&bs, b->bytes, sizeof bs); if (bs.target < 8) S->blend[bs.target] = bs; break;
        case 1: memcpy(&S->color, b->bytes, sizeof S->color); break;
        case 2: memcpy(&S->chmask, b->bytes, sizeof S->chmask); break;
        case 3: memcpy(&S->ms, b->bytes, sizeof S->ms); break;
        case 4: memcpy(&S->poly, b->bytes, sizeof S->poly); break;
        case 5: memcpy(&S->ds, b->bytes, sizeof S->ds); break;
        }
        break; }
    case OP_BIND_VATTR: { const p_bind_vattr *b = a; if (b->n >= 0 && b->n <= 16) { memcpy(S->attr, b->st, 4 * (size_t)b->n); S->nattr = b->n; } break; }
    case OP_BIND_VSTREAM: { const p_bind_vstream *b = a; if (b->n >= 0 && b->n <= 16) { memcpy(S->stream, b->st, 8 * (size_t)b->n); S->nstream = b->n; } break; }
    case OP_BIND_VBUF: case OP_BIND_VBUF1: { p_bind_vbuf tmp; const p_bind_vbuf *b = vbuf_rec(op, a, &tmp); if (b->first >= 0 && b->first + b->n <= 16) memcpy(&S->vbuf[b->first], b->r, sizeof *b->r * (size_t)b->n); break; }
    case OP_BIND_UBO: case OP_BIND_UBO_SMALL: { p_bind_buf tmp; const p_bind_buf *b = bind_buf_rec(op, a, &tmp); if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16) {
        if (memcmp(&S->ubo[b->stage][b->index], b, sizeof *b)) { S->ubo[b->stage][b->index] = *b; g_bver[b->stage]++; } } break; }
    case OP_BIND_SSBO: { const p_bind_buf *b = a; if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16) {
        if (memcmp(&S->ssbo[b->stage][b->index], b, sizeof *b)) { S->ssbo[b->stage][b->index] = *b; g_bver[b->stage]++; } } break; }
    case OP_BIND_TEX: case OP_BIND_TEX1: { p_bind_handles t1; const p_bind_handles *b = bind_rec(op, a, &t1);
        if (b->n < 0 || b->n > 32) { RED_ONCE(110, "a bind record holding count %d: overwritten after it was recorded", b->n); break; }
        bind_handles_fresh(b);
        if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= NTEX && memcmp(&S->tex[b->stage][b->first], b->h, 8 * (size_t)b->n)) { memcpy(&S->tex[b->stage][b->first], b->h, 8 * (size_t)b->n); g_bver[b->stage]++; }
        break; }
    case OP_BIND_SAMP: { const p_bind_handles *b = a; bind_handles_fresh(b); if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= 128 && memcmp(&S->samp[b->stage][b->first], b->h, 8 * (size_t)b->n)) { memcpy(&S->samp[b->stage][b->first], b->h, 8 * (size_t)b->n); g_bver[b->stage]++; } break; }
    case OP_BIND_IMG: { const p_bind_handles *b = a; bind_handles_fresh(b); if (b->stage >= 0 && b->stage < 6 && b->first >= 0 && b->first + b->n <= 32 && memcmp(&S->img[b->stage][b->first], b->h, 8 * (size_t)b->n)) { memcpy(&S->img[b->stage][b->first], b->h, 8 * (size_t)b->n); g_bver[b->stage]++; } break; }
    case OP_STENCIL: { const p_stencil *b = a; for (int f = 0; f < 2; f++) if (b->face & (1 << f)) { if (b->which == 0) S->st_vmask[f] = (uint32_t)b->value_mask; else if (b->which == 1) S->st_wmask[f] = (uint32_t)b->mask; else S->st_ref[f] = (uint32_t)b->ref; } break; }
    case OP_POLY_OFFSET: S->po = *(const p_poly_offset *)a; break;
    case OP_VIEWPORTS: { const p_viewports *b = a; for (int i = 0; i < b->n && b->first + i < 16; i++) memcpy(S->vp[b->first + i], b->v[i], 16); S->has_vp = 1; break; }
    case OP_DEPTH_RANGE: { const p_depth_range *b = a; S->dr[0] = b->n; S->dr[1] = b->f; break; }
    case OP_DEPTH_CLAMP: S->depth_clamp = ((const p_depth_clamp *)a)->on; break;
    case OP_VP_SWIZZLE: { const p_vp_swizzle *b = a; if (b->first >= 0 && b->first + b->n <= 16) memcpy(S->vp_sw[b->first], b->sw, 4 * (size_t)b->n); break; }
    case OP_BLEND_COLOR: memcpy(S->blend_color, ((const p_blend_color *)a)->rgba, 16); break;
    case OP_ALPHA_REF: S->alpha_ref = ((const p_alpha_ref *)a)->ref; break;
    case OP_COPY_STRIDE: { const p_copy_stride *cs = a; if (cs->image) S->copy_img_stride = cs->stride; else S->copy_stride = cs->stride; break; }
    case OP_TF_STATE: { const p_tf_state *b = a; S->tf_on = b->what == 0 || b->what == 3; if (S->tf_on) RED_ONCE(111, "transform feedback is not captured yet"); break; }
    case OP_PRIM_RESTART: S->restart = *(const p_prim_restart *)a; break;
    case OP_RAST_DISCARD: S->rast_discard = ((const p_rast_discard *)a)->on; break;
    default: break;
    }
}
// Linear render targets the CPU can read (NVN_TEXTURE_FLAGS_LINEAR / LINEAR_RENDER_TARGET in a pool without CPU_NO_ACCESS): on the console the pool is the
// CPU's memory too, so what the GPU draws there is what the CPU reads back (MKW draws its decoded "Motion" animations into an 8x1218 RGBA16F
// linear target and its CPU animation decoder F_83A8CA0 reads the texels: without them the vendors and crowds read stale memory, zeros or garbage, and
// were invisible or stretched over the screen). Each one drawn in a submission is read back into its storage when the submission completes
// (mirror_readback, as for copies). NXRT_VK_RT_READBACK=0 turns it off
static struct { tex_t t; unsigned level, layer; } g_rtrb[32]; static int g_nrtrb;
static void rt_readback_note(const tex_t *t, const view_t *v, int has_v) {
    if (!(t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) || !t->pool || (t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) ||
        t->d.target == NVN_TEXTURE_TARGET_3D || !t->mem || !vkc_int("rt_readback", 1)) return;
    unsigned level = has_v && v->set_levels ? v->base_level : 0, layer = has_v && v->set_layers ? v->min_layer : 0;
    for (int i = 0; i < g_nrtrb; i++) if (g_rtrb[i].t.serial == t->serial && g_rtrb[i].level == level && g_rtrb[i].layer == layer) return;
    if (g_nrtrb == 32) { RED_ONCE(97, "more than 32 linear render targets in one submission: the later ones are not read back for the CPU"); return; }
    g_rtrb[g_nrtrb].t = *t; g_rtrb[g_nrtrb].level = level; g_rtrb[g_nrtrb].layer = layer; g_nrtrb++;
}
static void rt_readback_flush(void) {
    static uint64_t n_total; static int logged;
    for (int i = 0; i < g_nrtrb; i++) {
        const tex_t *t = &g_rtrb[i].t; vkt_image *im = vkt_resolve(t); if (!im || !im->content) continue;
        unsigned lv = g_rtrb[i].level; NVNcopyRegion r = {0, 0, 0, (int)(t->d.width >> lv ? t->d.width >> lv : 1), (int)(t->d.height >> lv ? t->d.height >> lv : 1), 1};
        mirror_readback(vks_cb(), im, t, lv, g_rtrb[i].layer, &r); n_total++;
        if (logged < 8) { logged++; VKLOG("linear render target ser=%llu %ux%u fmt %#x flags %#x stride %u at %p: read back into its memory for the CPU (as the console's shared memory has it)",
            (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format, t->d.flags, t->d.stride, (void *)t->mem); }
    }
    g_nrtrb = 0;
}
static void set_rt(const p_set_rt *r) {
    pass_end(); P.valid = 0;
    S->nrt = r->n > 8 ? 8 : r->n < 0 ? 0 : r->n;
    memset(S->has_ct, 0, sizeof S->has_ct);
    for (int i = 0; i < S->nrt; i++) {
        const tex_t *t = snap(r->color[i]);
        if (t) { S->ct[i] = *t; S->has_ct[i] = 1; rt_readback_note(t, &r->cview[i], r->has_cview[i]); }
        S->cview[i] = r->cview[i]; S->has_cview[i] = r->has_cview[i];
    }
    const tex_t *d = snap(r->depth);
    S->has_dt = d != NULL; if (d) S->dt = *d;
    S->dview = r->dview; S->has_dview = r->has_dview;
}
static void render_cond(const p_render_cond *rc) {   // nvn_mtl.m OP_RENDER_COND: the counters as guest memory holds them at replay
    if (rc->mode >= 2) { S->render_skip = rc->mode == 3; if (g_pd) PD("render enable %d", rc->mode == 2); return; }
    uint64_t x = 0, y = 0; int known = rc_value(rc->addr, &x) & rc_value(rc->addr + 16, &y);
    S->render_skip = known && (rc->mode ? x == y : x != y);
    if (g_pd) PD("conditional render %s addr=%#llx values %llu / %llu%s%s", rc->mode ? "draw if not equal" : "draw if equal", (unsigned long long)rc->addr, (unsigned long long)x, (unsigned long long)y, known ? "" : " (unknown, drawn)", S->render_skip ? " -> not drawn" : "");
}

// ---------- Events: the GPU waits (nvnCommandBufferWaitEvent) ----------
// On the console the GPU stops at a wait until the event's value meets the condition; the value is written by the CPU (nvnEventSignal) or by GPU work
// (nvnCommandBufferSignalEvent, applied here when its submission completes). Ignored until 2026-11-07, so the GPU ran on into work the game had told it to hold
// back. Here: a GPU signal already recorded will come first in queue order, and a barrier keeps what follows behind what came before; a value only the CPU can
// write is waited for on this thread, after what is recorded so far has gone to the GPU (as the console's GPU would have run it). NXRT_VK_WAIT_EVENT=0: ignored again
static void split_now(void);
static struct { volatile uint32_t *val; uint32_t v; } g_evp[128]; static int g_nevp;   // the value each event will hold once the GPU signals recorded so far land
static uint64_t g_ev_now, g_ev_gpu, g_ev_cpu, g_ev_late, g_ev_ns;
static int ev_meets(uint32_t cur, uint32_t mode, uint32_t v) { return mode == NVN_EVENT_WAIT_MODE_EQUAL ? cur == v : (int32_t)(cur - v) >= 0; }
static void ev_recorded(volatile uint32_t *val, uint32_t mode, uint32_t v) {
    int i = 0; while (i < g_nevp && g_evp[i].val != val) i++;
    if (i == g_nevp) { if (g_nevp < 128) g_nevp++; else i = (int)(g_ev_gpu % 128); g_evp[i].val = val; g_evp[i].v = *val; }
    g_evp[i].v = mode == NVN_EVENT_SIGNAL_MODE_ADD ? g_evp[i].v + v : v;
}
static void wait_event(volatile uint32_t *val, uint32_t mode, uint32_t v) {
    static int max_ms = -1; if (max_ms < 0) { const char *e = getenv("NXRT_VK_WAIT_EVENT_MS"); max_ms = e && atoi(e) > 0 ? atoi(e) : 250; }
    if (!vkc_int("wait_event", 1) || !val) return;   // gamectl set wait_event 0 (or NXRT_VK_WAIT_EVENT=0): ignored, as before
    static uint64_t last; if (now_ns() - last > 5000000000ull) { last = now_ns(); VKLOG("GPU event waits so far: %llu met already, %llu by GPU signals ahead, %llu for the CPU (%.0f ms waited), %llu not met in time", (unsigned long long)g_ev_now, (unsigned long long)g_ev_gpu, (unsigned long long)g_ev_cpu, g_ev_ns / 1e6, (unsigned long long)g_ev_late); }
    pass_end(); barrier_all(vks_cb());   // what follows waits for what came before, as behind the console's wait
    if (ev_meets(*val, mode, v)) { g_ev_now++; return; }
    for (int i = 0; i < g_nevp; i++) if (g_evp[i].val == val && ev_meets(g_evp[i].v, mode, v)) { g_ev_gpu++; return; }   // a GPU signal ahead in the queue
    static volatile uint32_t *g_ev_nowait[64]; static int g_nnowait;   // events a wait timed out on once: signalled by work replayed later (another NVN queue), never waited for again
    for (int i = 0; i < g_nnowait; i++) if (g_ev_nowait[i] == val) { g_ev_late++; return; }
    g_ev_cpu++;   // only the CPU can write it: what is recorded goes to the GPU first, then this thread waits
    split_now();
    uint64_t t0 = now_ns(), lim = (uint64_t)max_ms * 1000000ull;
    while (!ev_meets(*val, mode, v) && now_ns() - t0 < lim) SwitchToThread();
    uint64_t el = now_ns() - t0; g_ev_ns += el;
    if (!ev_meets(*val, mode, v)) { g_ev_late++; if (g_nnowait < 64) g_ev_nowait[g_nnowait++] = val; if (g_ev_late <= 8 || !(g_ev_late % 1000)) VKRED("a GPU wait on an event at %p (mode %u, value %u, holds %u) was not met in %d ms: going on (%llu so far)", (void *)val, mode, v, *val, max_ms, (unsigned long long)g_ev_late); }
}

// ---------- The record walk (nvn_mtl.m mtl_exec; the framing checks of track B's walk) ----------
static void bad_record(const uint8_t *p, op_hdr h, const char *why) {
    if (g_st.bad++ < 8) VKRED("%s at %p (op %u, size %u): the rest of this command list is not replayed", why, (const void *)p, h.op, h.size);
}
static int walk(const uint8_t *p, int depth, uint64_t *budget) {
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        if (!(*budget)--) { bad_record(p, h, "more than 16M records in one list (a JUMP loop?)"); return -1; }
        if (h.op >= OP_COUNT_ || h.size < 8 || (h.size & 7) || (h.op != OP_END && h.size > (64u << 20))) { bad_record(p, h, "a malformed record"); return -1; }
        const uint8_t *a = p + 8;
        g_st.records++;
        switch (h.op) {
        case OP_END: return 0;
        case OP_BEGIN: break;
        case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); if (!t) { bad_record(p, h, "a JUMP to 0"); return -1; } p = t; continue; }
        case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (depth >= 8 || !t) { bad_record(p, h, depth >= 8 ? "CALLs nested deeper than 8" : "a CALL of handle 0"); return -1; }
            if (walk((const uint8_t *)(uintptr_t)t, depth + 1, budget)) return -1; break; }
        case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); if (val) { vks_post_event(val, m == NVN_EVENT_SIGNAL_MODE_ADD ? 1 : 0, v); ev_recorded(val, m, v); } break; }
        case OP_WAIT_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); wait_event(val, m, v); break; }
        case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); report_counter(t, ad); break; }
        case OP_RESET_COUNTER: { uint64_t t; memcpy(&t, a, 8); reset_counter(t); break; }
        case OP_FENCE_SYNC: { sync_t *s; memcpy(&s, a, 8); if (s) vk_note_fence_sync(s); break; }
        case OP_COPY_B2B: { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); vkr_copy(s_, d, n); break; }
        case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8);
            if (24 + n > h.size) { bad_record(p, h, "a uniform update longer than its record"); return -1; }
            vkr_update(d, n, a + 16); break; }
        case OP_CLEAR_BUF: { const p_clear_buf *cb = (const p_clear_buf *)a; vkr_fill(cb->addr, cb->size, cb->value); break; }
        case OP_SET_RT: set_rt((const p_set_rt *)a); maybe_split(); break;
        case OP_CLEAR_COLOR: clear_color((const p_clear_color *)a); break;
        case OP_CLEAR_DS: clear_ds((const p_clear_ds *)a); break;
        case OP_DRAW: { PH0(); p_draw t; draw(draw_rec(a, h.size, &t), NULL); PH1(PH_DRAW); break; }
        case OP_DRAW_IDX: { PH0(); p_draw_idx t; draw(NULL, draw_idx_rec(a, h.size, &t)); PH1(PH_DRAW); break; }
        case OP_DISPATCH: { PH0(); dispatch((const p_dispatch *)a); PH1(PH_DISP); } maybe_split(); break;
        case OP_BARRIER: pass_end(); g_st.barriers++; break;
        case OP_COPY_B2T: copy_b2t((const p_copy_b2t *)a); break;
        case OP_COPY_T2B: copy_t2b((const p_copy_b2t *)a); break;
        case OP_COPY_T2T: copy_t2t((const p_copy_t2t *)a); break;
        case OP_CLEAR_TEX: clear_tex((const p_clear_tex *)a); break;
        case OP_DRAW_TEX: RED_ONCE(120, "DrawTexture is not replayed yet"); break;
        case OP_RENDER_COND: render_cond((const p_render_cond *)a); break;
        case OP_PUSH_CONST: { const p_push_const *r = (const p_push_const *)a; if ((uint32_t)r->offset + r->size <= DMSH_PUSH_SIZE) { memcpy(S->push[r->compute != 0] + r->offset, r->data, r->size); for (int k = 0; k < 6; k++) g_bver[k]++; } break; }
        case OP_TF_BIND: break;
        default: state_op(h.op, a); break;
        }
        p += h.size;
    }
}
void vkr_exec(const uint8_t *handle) {
    { extern volatile DWORD g_host_focus_tid; g_host_focus_tid = GetCurrentThreadId(); }   // SWR_HOST_PROF: this thread's stacks
    g_st.handles++;
    if (!handle) { op_hdr z = {0, 0}; bad_record(handle, z, "a null command handle"); return; }
    if (!S) S = rs_get(0);
    uint64_t budget = 1u << 24;
    PH0(); walk(handle, 0, &budget); PH1(PH_EXEC);
}

// ---------- Submissions, queues ----------
// A long submission goes to the GPU in pieces: recording and GPU work then overlap, where one whole submission puts them one after the other (26 ms + 18 ms a frame in a race). gamectl set split_ms N, 0 = off
static uint64_t g_split_t; static int g_split_ms; static uint64_t g_splits;
// NXRT_VK_BALLAST_MB=N (test, the GPU freeze): N MB of device memory outside our accounting, written a little by every submission so Windows keeps it
// resident: more pressure on video memory, as if the game needed N MB more. If the freeze comes sooner, memory pressure triggers it
static void ballast_touch(VkCommandBuffer cb) {
    static int st = -1; static VkBuffer b[64]; static int nb;
    if (st < 0) { st = 0; const char *e = getenv("NXRT_VK_BALLAST_MB"); int mb = e ? atoi(e) : 0;
        int mt = vk_mem_type(0xffffffffu, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        for (int done = 0; mb > 0 && mt >= 0 && done < mb && nb < 64; done += 128) {
            VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 128u << 20, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
            VkBuffer x; VkDeviceMemory m; if (vkCreateBuffer(g_vk.dev, &bci, NULL, &x) != VK_SUCCESS) break;
            VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, x, &mr);
            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, (uint32_t)mt};
            if (vkAllocateMemory(g_vk.dev, &ai, NULL, &m) != VK_SUCCESS) { vkDestroyBuffer(g_vk.dev, x, NULL); break; }
            vkBindBufferMemory(g_vk.dev, x, m, 0); b[nb++] = x; }
        if (nb) { st = 1; VKLOG("ballast for the freeze test: %d MB of device memory kept resident (NXRT_VK_BALLAST_MB)", nb * 128); } }
    if (st == 1) { static uint32_t v; v++; for (int i = 0; i < nb; i++) vkCmdFillBuffer(cb, b[i], ((uint64_t)(v * 4096u) % (128u << 20)) & ~3ull, 4, v); }
}
static void sub_open_rest(void) {
    ballast_touch(vks_cb());
    gt_mark(vks_cb(), 's', 0, 0, 0, "start");
    if (g_tqp) { uint32_t q = g_tq_next++ % NTQ; vkCmdResetQueryPool(vks_cb(), g_tqp, 2 * q, 2); vkCmdWriteTimestamp2(vks_cb(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, g_tqp, 2 * q); g_sub.tq = (int)q + 1; }
}
static void split_now(void);
static void maybe_split(void) {
    if (!g_split_ms || P.open) return;
    uint64_t now = now_ns(); if (now - g_split_t < (uint64_t)g_split_ms * 1000000) return;
    split_now();
}
static void split_now_(void);
static void split_now(void) { PH0(); split_now_(); PH1(PH_SPLIT); }
static void split_now_(void) {   // what is recorded so far goes to the GPU, recording goes on in a new slot (no pass may be open)
    g_split_t = now_ns(); g_splits++;
    if (g_nwb) { barrier_all(vks_cb()); wb_flush(); }
    gt_mark(vks_cb(), 'e', 0, 0, 0, "split");
    if (g_sub.tq) vkCmdWriteTimestamp2(vks_cb(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, g_tqp, 2 * (uint32_t)(g_sub.tq - 1) + 1);
    subres *r = malloc(sizeof *r); *r = g_sub; memset(&g_sub, 0, sizeof g_sub); vks_post_fn(subres_done, r);
    vks_split();
    P.valid = 0; P.open = 0; P.query = -1; g_last_pipe = VK_NULL_HANDLE; g_nwb = 0; ds_forget();
    sub_open_rest();
}
void vkr_sub_begin(void) {
    g_split_t = now_ns(); g_split_ms = vkc_int("split_ms", 8);
    memset(&g_sub, 0, sizeof g_sub); g_memo_sub++;   // the texture / sampler memo starts again with each submission
    P.valid = 0; P.open = 0; P.query = -1; g_last_pipe = VK_NULL_HANDLE; g_nwb = 0; g_nrtrb = 0; ds_forget();
    if (!S) S = rs_get(0);
    g_rc_npend = 0; S->render_skip = 0;
    // the slot's reset is recorded (VUID-vkResetQueryPool-firstQuery-02741: a host reset may meet the slot's last use still running while loading floods the queue)
    sub_open_rest();
}
void vkr_sub_end(void) {
    pass_end();
    if (g_nrtrb) rt_readback_flush();
    if (g_nwb) { barrier_all(vks_cb()); wb_flush(); }
    gt_mark(vks_cb(), 'e', 0, 0, 0, "end");
    if (g_sub.tq) vkCmdWriteTimestamp2(vks_cb(), VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, g_tqp, 2 * (uint32_t)(g_sub.tq - 1) + 1);
    subres *r = malloc(sizeof *r); *r = g_sub; memset(&g_sub, 0, sizeof g_sub);
    vks_post_fn(subres_done, r);
    P.valid = 0;
}
void vkr_queue_activate(uint64_t serial) { S = rs_get(serial); P.valid = 0; }
void vkr_queue_forget(uint64_t serial) {
    for (rs **pp = &g_states; *pp; pp = &(*pp)->next) if ((*pp)->serial == serial) { rs *s = *pp; *pp = s->next; if (S == s) S = NULL; free(s); break; }
}
void vkr_set_origin_lower_left(int on) { if (g_origin_ll != on) VKLOG("window origin: %s", on ? "lower left (the viewport as given)" : "upper left (the viewport flipped)"); g_origin_ll = on; }
void vkr_set_queue_origin_lower_left(int on) { g_queue_origin_ll = on; }
void vkr_set_depth_zero(int on) { g_depth_zero = on; }

void vkr_stats(vkr_stats_t *s) { *s = g_st; }
static _Atomic long g_pd_req; static _Atomic int g_td_req;
void vkr_passdump_next(long n) { atomic_store(&g_pd_req, n > 0 ? n : 1); }
void vkr_texdump_next(void) { atomic_store(&g_td_req, 1); }
// gamectl rawdump: named images, every layer of level 0, at the end of each of the next frames, beside the log, read with tools/image_raw.py
static uint64_t g_rd_ser[16]; static volatile long g_rd_n, g_rd_frames;
void vkr_rawdump(const uint64_t *ser, int n, long frames) { g_rd_frames = 0; g_rd_n = n > 16 ? 16 : n; for (int i = 0; i < g_rd_n; i++) g_rd_ser[i] = ser[i]; g_rd_frames = frames; }
static void rawdump_frame(uint64_t frame) {
    if (g_rd_frames <= 0) return;
    g_rd_frames--; vks_finish();
    const char *lp = nxrt_log_path();
    for (int i = 0; i < g_rd_n; i++) { vkt_image *im = vkt_by_serial(g_rd_ser[i]);
        if (!im || (im->fi->flags & VKF_COMPRESSED)) continue;
        size_t bytes = (size_t)im->w * im->h * im->fi->vbpb; uint8_t *b = malloc(bytes);
        for (unsigned l = 0; b && l < im->layers && l < 16; l++) { NVNcopyRegion rg = {0, 0, 0, (int)im->w, (int)im->h, 1}; char path[1200];
            if (vkt_readback_region(im, 0, l, &rg, b, bytes)) continue;
            snprintf(path, sizeof path, "%s.img-%llu-ser%llu-%s-%ux%u-l%u.raw", lp && *lp ? lp : "nxrt", (unsigned long long)frame, (unsigned long long)im->serial, im->fi->name, im->w, im->h, l);
            FILE *f = fopen(path, "wb"); if (f) { fwrite(b, 1, bytes, f); fclose(f); } }
        free(b); }
}
// A frame over 120 ms says what took the time: a stop has one cause or a few, and an average hides them
static void slow_frame(uint64_t frame) {
    extern _Atomic uint64_t g_t_bind, g_n_bind, g_t_gpuwait; uint64_t vkt_up_ns(void);
    static uint64_t t0, pb, pn, pw, pu, pp, pd; static int nlog; uint64_t now = now_ns(), b = atomic_load(&g_t_bind), n = atomic_load(&g_n_bind), w = atomic_load(&g_t_gpuwait), u = vkt_up_ns();
    vkp_stats_t ps; vkp_stats(&ps); struct { uint64_t up_ns; } ms = {vkm_up_ns()};   // not vkm_stats: it walks every pool under the lock, every frame
    static int slow_ms = -1; if (slow_ms < 0) slow_ms = vkc_int("slow_ms", 50);   // NXRT_VK_SLOW_MS: frames longer than this get a line (tools/stutter.py reads them)
    if (t0 && now - t0 > (uint64_t)slow_ms * 1000000 && nlog++ < 20000) VKLOG("slow frame %llu: %.0f ms: sparse binds %.0f ms (%llu), GPU waits %.0f ms, texture uploads %.0f ms, pipelines %.0f ms, pool copies %.0f ms", (unsigned long long)frame, (now - t0) / 1e6, (b - pb) / 1e6, (unsigned long long)(n - pn), (w - pw) / 1e6, (u - pu) / 1e6, (ps.create_ns - pp) / 1e6, (ms.up_ns - pd) / 1e6);
    t0 = now; pb = b; pn = n; pw = w; pu = u; pp = ps.create_ns; pd = ms.up_ns;
}
void vkr_frame(uint64_t frame) {
    static vkr_stats_t prev; static uint64_t pf;
    g_dl_frame = frame;
    slow_frame(frame);
    rawdump_frame(frame);
    vkv_frame(frame);
    vkw_frame(frame);
    vkq_frame(frame); g_pass_in_frame = 0;
    if (g_ntc) { vks_finish(); td_pass_write(frame); }
    if (g_raw_on && g_ntd) {   // NXRT_VK_DUMP_RAW: the small images a dumped frame wrote, every layer of level 0, as they are, read with tools/image_raw.py
        vks_finish();
        for (int i = 0; i < g_ntd; i++) { vkt_image *im = g_td[i];
            if (im->w > 512 || im->h > 512 || (im->fi->flags & VKF_COMPRESSED) || (im->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))) continue;
            size_t bytes = (size_t)im->w * im->h * im->fi->vbpb; uint8_t *b = malloc(bytes); const char *lp = nxrt_log_path();
            for (unsigned l = 0; b && l < im->layers && l < 12; l++) { NVNcopyRegion rg = {0, 0, 0, (int)im->w, (int)im->h, 1}; char path[1200];
                if (vkt_readback_region(im, 0, l, &rg, b, bytes)) continue;
                snprintf(path, sizeof path, "%s.img-%llu-ser%llu-%s-%ux%u-l%u.raw", lp && *lp ? lp : "nxrt", (unsigned long long)frame, (unsigned long long)im->serial, im->fi->name, im->w, im->h, l);
                FILE *f = fopen(path, "wb"); if (f) { fwrite(b, 1, bytes, f); fclose(f); } }
            free(b); }
        g_ntd = 0; g_raw_on = 0;
    }
    g_tc_pass = 0;
    gt_frame(frame);
    long pr = atomic_exchange(&g_pd_req, 0); if (pr > 0) { g_pd_frame = frame; g_pd_last = frame + (uint64_t)pr - 1; }   // DM_CTL passdump / texdump: from this frame on
    if (atomic_exchange(&g_td_req, 0)) g_td_frame = frame;
    // NXRT_VK_DUMP_AT_S=<seconds,seconds,...>: at each moment the colour targets of one frame as its passes leave them, then those of the next frame with every pixel shader drawing its
    // number, so a pixel can be traced to its shader (tools/pass_pair.py). By time: a moment's frame number changes from run to run
    { static uint64_t t0, at[16]; static int nat = -1, cur;
      if (nat < 0) { nat = 0; t0 = now_ns(); for (const char *e = getenv("NXRT_VK_DUMP_AT_S"); e && *e && nat < 16; ) { char *q; at[nat++] = strtoull(e, &q, 10); e = *q ? q + 1 : q; } }
      if (cur < nat && now_ns() - t0 >= at[cur] * 1000000000ull) { cur++; g_td_frame = frame; g_id_mode = !getenv("NXRT_VK_DUMP_RAW"); g_no_tex = g_id_mode; VKLOG("dump by time: frame %llu", (unsigned long long)frame); } }
    g_pd = g_pd_frame && frame >= g_pd_frame && frame <= g_pd_last;
    if (g_td_frame && frame == g_td_frame + 1 && g_ntd) {   // the dump frame's images, now that its work is done
        vks_finish();
        for (int i = 0; i < g_ntd; i++) {
            vkt_image *im = g_td[i]; char path[1200]; const char *lp = nxrt_log_path();
            for (unsigned z = 0; z < (im->vtype == VK_IMAGE_VIEW_TYPE_3D ? (im->depth > 4 ? 4 : im->depth) : 1); z++) {
                snprintf(path, sizeof path, "%s.tex-%llu-ser%llu-%s-%ux%u-z%u.png", lp && *lp ? lp : "nxrt", (unsigned long long)g_td_frame, (unsigned long long)im->serial, im->fi->name, im->w, im->h, z);
                VKLOG("texture dump: ser%llu %s %ux%ux%u %s", (unsigned long long)im->serial, im->fi->name, im->w, im->h, im->depth, vks_dump_image(im, z, path) ? "not dumped" : path);
                if (im->w * im->h <= 16 && im->fi->vbpb <= 16 && !(im->fi->flags & VKF_COMPRESSED) && !(im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) {   // tiny ones: the values themselves
                    uint8_t raw[256]; NVNcopyRegion rr = {0, 0, 0, (int)im->w, (int)im->h, 1}; char t[600]; int o = 0;
                    if (!vkt_readback_region(im, 0, 0, &rr, raw, (size_t)im->w * im->h * im->fi->vbpb))
                        for (uint32_t k = 0; k < im->w * im->h * im->fi->vbpb / 4 && o < 560; k++) { float f; uint32_t u; memcpy(&f, raw + 4 * k, 4); memcpy(&u, raw + 4 * k, 4); o += snprintf(t + o, sizeof t - o, " %g(%#x)", f, u); }
                    VKLOG("texture dump: ser%llu values:%s", (unsigned long long)im->serial, t);
                }
            }
        }
        g_ntd = 0;
    }
    if (g_td_on && getenv("NXRT_VK_DUMP_RAW")) { g_raw_on = 1; g_td_frame = 0; }
    g_td_on = g_td_frame && frame == g_td_frame;
    g_id_on = g_id_mode && g_td_frame && frame == g_td_frame + 1;
    if (g_pd) PD("---- the frame after present %llu ----", (unsigned long long)frame);

    if (frame % 60 || frame == pf) return;
    static uint64_t pgn, pgs; uint64_t gn = atomic_load(&g_gpu_ns), gs = atomic_load(&g_gpu_subs);
    if (gs > pgs) VKLOG("GPU time up to frame %llu: %.2f ms per frame (%.2f ms per submission, %.1f submissions per frame; timestamps around each submission's replay)",
                        (unsigned long long)frame, (gn - pgn) / 1e6 / (double)(frame - pf), (gn - pgn) / 1e6 / (double)(gs - pgs), (double)(gs - pgs) / (double)(frame - pf));
    pgn = gn; pgs = gs;
    vkr_stats_t s = g_st; uint64_t df = frame - pf; vkp_stats_t ps; vkp_stats(&ps); uint64_t shl, shf; vksh_stats(&shl, &shf);
    VKLOG("replay up to frame %llu, per frame: %.1f draws (%.1f skipped: no shader %llu, no pipeline %llu, no target %llu in all), %.1f passes, %.1f dispatches, %.1f clears, "
          "%.1f texture copies, %.1f buffer copies / updates / fills, %.1f descriptor sets, %.1f write-backs; %llu graphics + %llu compute pipelines (%.1f ms creating, worst %.1f ms), %llu shaders (%llu unusable)",
          (unsigned long long)frame, (double)(s.draws - prev.draws) / df, (double)(s.draws_skipped - prev.draws_skipped) / df, (unsigned long long)s.skip_no_shader,
          (unsigned long long)s.skip_no_pipeline, (unsigned long long)s.skip_no_target, (double)(s.passes - prev.passes) / df, (double)(s.dispatches - prev.dispatches) / df,
          (double)(s.clears - prev.clears) / df, (double)(s.tex_copies - prev.tex_copies) / df, (double)(s.copies + s.updates + s.fills - prev.copies - prev.updates - prev.fills) / df,
          (double)(s.desc_sets - prev.desc_sets) / df, (double)(s.writebacks - prev.writebacks) / df, (unsigned long long)ps.graphics, (unsigned long long)ps.compute,
          ps.create_ns / 1e6, ps.max_ns / 1e6, (unsigned long long)shl, (unsigned long long)shf);
    { static uint64_t pv, pset, pskip, preu, pna, puc, puh, pus; uint64_t vkm_wb_noaccess_bytes(void); uint64_t na = vkm_wb_noaccess_bytes();
      void vkm_use_counts(uint64_t *, uint64_t *, uint64_t *); uint64_t uc_, uh_, us_; vkm_use_counts(&uc_, &uh_, &us_);
      VKLOG("replay calls per frame: dynamic state groups set %.1f, already set %.1f (dyn_dedupe); buffer views made %.1f; descriptor sets reused %.1f (set_reuse); write-backs into CPU_NO_ACCESS pools %.0f KB; memory uses %.0f (fast %.0f, synced %.0f)",
          (double)(g_n_ds_set - pset) / df, (double)(g_n_ds_skip - pskip) / df, (double)(g_n_bview - pv) / df, (double)(g_n_set_reused - preu) / df, (double)(na - pna) / 1024.0 / df, (double)(uc_ - puc) / df, (double)(uh_ - puh) / df, (double)(us_ - pus) / df); puc = uc_; puh = uh_; pus = us_; pv = g_n_bview; pset = g_n_ds_set; pskip = g_n_ds_skip; preu = g_n_set_reused; pna = na; }
    { uint64_t vkp_gpl_stats(uint64_t *, uint64_t *, uint64_t *); uint64_t gl, gf, gns, gp = vkp_gpl_stats(&gl, &gf, &gns);
      if (gl || gf) VKLOG("pipelines linked from parts so far: %llu (%llu made whole instead); %llu parts made, %.0f ms", (unsigned long long)gl, (unsigned long long)gf, (unsigned long long)gp, gns / 1e6); }
    if (s.plain_nocmp) VKLOG("plain samples given a sampler without comparison so far: %llu (plain_nocmp)", (unsigned long long)s.plain_nocmp);
    if (ps.prewarmed) VKLOG("pipelines made ahead by the prewarm up to frame %llu: %llu (%.1f ms on its threads)", (unsigned long long)frame, (unsigned long long)ps.prewarmed, ps.prewarm_ns / 1e6);
    if (ph_on()) { static uint64_t pp[PH_N], pt0, pn0, pu; uint64_t tn = PH_T(), nn = now_ns(); double tsc = pn0 && nn > pn0 ? (double)(tn - pt0) / (double)(nn - pn0) : 0; pt0 = tn; pn0 = nn;
        if (!tsc) tsc = 3.2;   // cycles a ns until two reports give the rate
        double k = 1e6 * (double)df * tsc; uint64_t uc = vkm_use_cycles();
        VKLOG("replay phases per frame: memory sync (vkm_use) %.2f ms", (uc - pu) / k); pu = uc;
        VKLOG("replay phases per frame: records %.2f ms (draws %.2f, of them descriptor sets %.2f; dispatches %.2f), commits inside submissions %.2f ms; "
              "sets: allocation %.2f, uniform buffers %.2f, storage buffers %.2f, textures and the rest %.2f, vkUpdateDescriptorSets %.2f",
              (g_ph_ns[PH_EXEC] - pp[PH_EXEC]) / k, (g_ph_ns[PH_DRAW] - pp[PH_DRAW]) / k, (g_ph_ns[PH_WSET] - pp[PH_WSET]) / k, (g_ph_ns[PH_DISP] - pp[PH_DISP]) / k, (g_ph_ns[PH_SPLIT] - pp[PH_SPLIT]) / k,
              (g_ph_ns[PH_WS_ALLOC] - pp[PH_WS_ALLOC]) / k, (g_ph_ns[PH_WS_UBO] - pp[PH_WS_UBO]) / k, (g_ph_ns[PH_WS_SBUF] - pp[PH_WS_SBUF]) / k, (g_ph_ns[PH_WS_TEX] - pp[PH_WS_TEX]) / k, (g_ph_ns[PH_WS_UPD] - pp[PH_WS_UPD]) / k);
        memcpy(pp, g_ph_ns, sizeof pp); }
    prev = s; pf = frame;
}

int vkr_init(void) {
    if (vksh_init() || vkpl_init()) return -1;
    if (!g_vk.has.null_descriptor) VKRED("the device has no null descriptors (VK_EXT_robustness2): unbound textures and images are not replayed correctly");
    VkBufferUsageFlags bu = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (mkbuf(ZERO_SIZE, bu, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &R0.zero, &R0.zero_m, "replay zeros") || mkbuf(ZERO_SIZE, bu, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &R0.sink, &R0.sink_m, "replay write sink") ||
        mkbuf(2048, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &R0.support, &R0.support_m, "support buffer") ||
        mkbuf(65536 * 24, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &R0.quads, &R0.quads_m, "quad indices") ||
        mkbuf(BOUNCE_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &R0.bounce, &R0.bounce_m, "storage bounce")) return -1;
    // the support buffer in Ryujinx's layout (nvn_mtl.m support_buf): render scale 1.0 in each of the 73 entries from byte 192, the rest zero
    memset(R0.support_m.map, 0, 2048);
    for (int i = 0; i < 73; i++) { float one = 1.0f; memcpy((uint8_t *)R0.support_m.map + 192 + 16 * i, &one, 4); }
    uint32_t *q = R0.quads_m.map;
    for (uint32_t i = 0; i < 65536; i++) { uint32_t b = 4 * i; q[6 * i] = b; q[6 * i + 1] = b + 1; q[6 * i + 2] = b + 2; q[6 * i + 3] = b; q[6 * i + 4] = b + 2; q[6 * i + 5] = b + 3; }
    VkCommandBuffer cb = vk_once_begin();
    vkCmdFillBuffer(cb, R0.zero, 0, ZERO_SIZE, 0); vkCmdFillBuffer(cb, R0.sink, 0, ZERO_SIZE, 0);
    vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
    if (vk_once_end(cb)) return -1;
    VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0, VK_FILTER_NEAREST, VK_FILTER_NEAREST, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                               VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, 0, VK_FALSE, 1, VK_FALSE, VK_COMPARE_OP_NEVER, 0, VK_LOD_CLAMP_NONE, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, VK_FALSE};
    if (vkCreateSampler(g_vk.dev, &sci, NULL, &R0.samp) != VK_SUCCESS) return -1;
    VkQueryPoolCreateInfo qci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_OCCLUSION, NQ, 0};
    if (vkCreateQueryPool(g_vk.dev, &qci, NULL, &R0.qp) != VK_SUCCESS) return -1;
    vkResetQueryPool(g_vk.dev, R0.qp, 0, NQ);
    VkQueryPoolCreateInfo tci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_TIMESTAMP, 2 * NTQ, 0};
    if (g_vk.props.limits.timestampComputeAndGraphics && vkCreateQueryPool(g_vk.dev, &tci, NULL, &g_tqp) == VK_SUCCESS) vkResetQueryPool(g_vk.dev, g_tqp, 0, 2 * NTQ);
    S = rs_get(0);
    const char *e = getenv("NXRT_VK_PASSDUMP"); if (e) { char *end; g_pd_frame = strtoull(e, &end, 0); g_pd_last = *end == '-' ? strtoull(end + 1, NULL, 0) : g_pd_frame; }
    if ((e = getenv("NXRT_VK_TEXDUMP"))) g_td_frame = strtoull(e, NULL, 0);
    vkpl_prewarm_start();   // M6 phase B: pipelines listed before, made ahead in the background
    return 0;
}
void vkr_shutdown(void) {
    while (g_states) { rs *s = g_states; g_states = s->next; free(s); }
    S = NULL;
    for (uint32_t i = 0; i < g_progc.cap; i++) free(g_progc.vals[i]);
    vk_hm_free(&g_progc);
    while (g_free_chunks) { rchunk *c = g_free_chunks; g_free_chunks = c->next; vkDestroyBuffer(g_vk.dev, c->buf, NULL); vk_free_mem(&c->mem); free(c); }
    while (g_free_pools) { rdpool *p = g_free_pools; g_free_pools = p->next; vkDestroyDescriptorPool(g_vk.dev, p->pool, NULL); free(p); }
    if (R0.qp) vkDestroyQueryPool(g_vk.dev, R0.qp, NULL);
    if (g_gtp) { vkDestroyQueryPool(g_vk.dev, g_gtp, NULL); g_gtp = VK_NULL_HANDLE; }
    if (g_tqp) vkDestroyQueryPool(g_vk.dev, g_tqp, NULL);
    g_tqp = VK_NULL_HANDLE;
    if (R0.samp) vkDestroySampler(g_vk.dev, R0.samp, NULL);
    VkBuffer bs[5] = {R0.zero, R0.sink, R0.support, R0.quads, R0.bounce}; vk_alloc *ms[5] = {&R0.zero_m, &R0.sink_m, &R0.support_m, &R0.quads_m, &R0.bounce_m};
    for (int i = 0; i < 5; i++) if (bs[i]) { vkDestroyBuffer(g_vk.dev, bs[i], NULL); vk_free_mem(ms[i]); }
    memset(&R0, 0, sizeof R0);
    for (int i = 0; i < g_ndct; i++) { for (int j = 0; j < 4; j++) if (g_dct[i].view[j]) vkDestroyImageView(g_vk.dev, g_dct[i].view[j], NULL); vkDestroyImage(g_vk.dev, g_dct[i].img, NULL); vk_free_mem(&g_dct[i].mem); }
    g_ndct = 0; vk_hm_free(&g_dct_of);
    for (uint32_t i = 0; i < g_cpipe.cap; i++) if (g_cpipe.vals[i]) vkDestroyPipeline(g_vk.dev, (VkPipeline)g_cpipe.vals[i], NULL);
    vk_hm_free(&g_cpipe);
    for (int i = 0; i < 4; i++) if (g_cmod[i]) { vkDestroyShaderModule(g_vk.dev, g_cmod[i], NULL); g_cmod[i] = VK_NULL_HANDLE; }
    if (g_cpl) { vkDestroyPipelineLayout(g_vk.dev, g_cpl, NULL); g_cpl = VK_NULL_HANDLE; }
    vkpl_shutdown(); vksh_shutdown();
}
