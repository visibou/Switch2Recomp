// The null graphics backend: the contract (gfx_backend.h, nvn_obj.h) for a GPU that draws nothing and finishes every submission at once. It walks the recorded
// commands as a replay would, checks their framing, and keeps the effects the CPU can see: buffer copies, uniform updates and fills land in guest memory, recorded
// fences signal their syncs with the submission, and event writes and counter reports are written when it completes (at the commit). This is what M4's null device
// did inside hle_nvn.c (nothing was recorded then); textures and draws are only counted. Texel reads come from the storage (nvn_texio), the ground truth when nothing draws.
#define GFX_BE null
#include "gfx_rename.h"
#include "gfx_null.h"
#include "nvn_texio.h"
#include "nvn_timestamp.h"
#include "nxs_rt.h"
#include "dm.h"
#include <stdatomic.h>

// ---------- Memory pools (gfx_addr resolves GPU addresses = CPU addresses) ----------
typedef struct { uint8_t *lo, *hi; } npool;
static npool *g_pools; static int g_npools, g_cpools; static SRWLOCK g_pool_lk = SRWLOCK_INIT;
void gfx_pool_add(const void *mem, uint64_t size) {
    if (!mem || !size) return;
    AcquireSRWLockExclusive(&g_pool_lk);
    if (g_npools == g_cpools && !(g_pools = realloc(g_pools, (size_t)(g_cpools = g_cpools ? 2 * g_cpools : 256) * sizeof *g_pools))) die("null backend: pool table allocation failed");
    int i = g_npools; while (i > 0 && g_pools[i - 1].lo > (uint8_t *)mem) { g_pools[i] = g_pools[i - 1]; i--; }
    g_pools[i] = (npool){(uint8_t *)mem, (uint8_t *)mem + size}; g_npools++;
    ReleaseSRWLockExclusive(&g_pool_lk);
}
void gfx_pool_del(const void *mem) {
    AcquireSRWLockExclusive(&g_pool_lk);
    for (int i = 0; i < g_npools; i++) if (g_pools[i].lo == mem) { memmove(g_pools + i, g_pools + i + 1, (size_t)(g_npools - i - 1) * sizeof *g_pools); g_npools--; break; }
    ReleaseSRWLockExclusive(&g_pool_lk);
}
void gfx_alias_drop_pool(const void *pool) { (void)pool; }
int gfx_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    int ok = 0; uint8_t *a = (uint8_t *)(uintptr_t)addr;
    AcquireSRWLockShared(&g_pool_lk);
    int lo = 0, hi = g_npools;   // the last pool starting at or below a
    while (lo < hi) { int m = (lo + hi) / 2; if (g_pools[m].lo <= a) lo = m + 1; else hi = m; }
    for (int i = lo - 1; i >= 0 && i >= lo - 8; i--)   // pools may overlap (a virtual pool aliasing a physical one): a few back
        if (a >= g_pools[i].lo && a + len <= g_pools[i].hi) { if (buf) *buf = g_pools[i].lo; if (off) *off = (uint64_t)(a - g_pools[i].lo); ok = 1; break; }
    ReleaseSRWLockShared(&g_pool_lk);
    return ok;
}

// ---------- Counts (tests and the resource line) ----------
static int g_depth_zero, g_origin_ll = 1, g_queue_origin_ll = 1, g_tex_pool_n;
static struct {
    _Atomic uint64_t submits, handles, records, bytes, draws, dispatches, copies, copy_bytes, updates, fills, fences, events, counters, tex_ops, presents, bad, calls;
    _Atomic uint64_t op[OP_COUNT_];
    _Atomic int live_tex, reg_tex, reg_samp, reg_img, progs_released, cpu_written;
} g_n;
void gfx_null_stats(gfx_null_counts *c) {
    memset(c, 0, sizeof *c);
    c->submits = g_n.submits; c->handles = g_n.handles; c->records = g_n.records; c->bytes = g_n.bytes; c->draws = g_n.draws; c->dispatches = g_n.dispatches;
    c->copies = g_n.copies; c->copy_bytes = g_n.copy_bytes; c->updates = g_n.updates; c->fills = g_n.fills; c->fences = g_n.fences; c->events = g_n.events;
    c->counters = g_n.counters; c->tex_ops = g_n.tex_ops; c->presents = g_n.presents; c->bad = g_n.bad; c->calls = g_n.calls;
    for (int i = 0; i < OP_COUNT_; i++) c->op[i] = g_n.op[i];
    c->depth_zero = g_depth_zero; c->origin_ll = g_origin_ll; c->queue_origin_ll = g_queue_origin_ll; c->tex_pool_n = g_tex_pool_n;
    c->live_tex = g_n.live_tex; c->reg_tex = g_n.reg_tex; c->reg_samp = g_n.reg_samp; c->reg_img = g_n.reg_img; c->progs_released = g_n.progs_released; c->cpu_written = g_n.cpu_written;
}
static gfx_null_bind_fn g_bind_hook;
void gfx_null_on_bind_program(gfx_null_bind_fn fn) { g_bind_hook = fn; }

// NXRT_SHADERS=1: resolve every program a replay binds through the shader cache (nxs_rt), as the Vulkan backend does, so a run on the null backend shows which
// of the game's shaders the cache has, dumps the ones it lacks and has them translated on demand. Off by default: M4's null device did no such work
static void resolve_program(const prog_t *p, int stages) {
    (void)stages; const nxs_shader *sh;
    for (unsigned i = 0; i < p->nstage && i < 6; i++) nxs_resolve(p, i, &sh);
}
static int g_resolve;
void gfx_init(void) {
    static int once; if (once++) return;
    LOG("[gfx] null backend: commands are recorded and walked, nothing is drawn; fences, events, counters, copies and uniform updates take effect at the submit");
    const char *e = getenv("NXRT_SHADERS");
    if (e && !strcmp(e, "1")) {
        if (nxs_rt_enabled()) { g_resolve = 1; gfx_null_on_bind_program(resolve_program); LOG("[gfx] null backend: programs are resolved through the shader cache (NXRT_SHADERS=1)"); }
        else LOG("[RED] NXRT_SHADERS=1 but there is no shader cache at %s (NXRT_SHADER_CACHE)", nxs_rt_cache_dir());
    }
}
void gfx_vbuf_stats(uint64_t *resolve, uint64_t *reuse, uint64_t *set, uint64_t *skip, uint64_t *copies, uint64_t *copy_bytes) { *resolve = *reuse = *set = *skip = *copies = *copy_bytes = 0; }
void gfx_set_depth_zero(int on) { g_depth_zero = on; }
void gfx_set_origin_lower_left(int on) { g_origin_ll = on; }
void gfx_set_queue_origin_lower_left(int on) { g_queue_origin_ll = on; }
void bf_want_note(const char *key) { (void)key; }
void gfx_res_snapshot(gfx_res_count *o) { memset(o, 0, sizeof *o); o->pools = (uint32_t)g_npools; o->live_tex = (uint32_t)g_n.live_tex; o->view_slots = (uint32_t)g_n.reg_tex; }
void gfx_res_counts(uint32_t tick, const char *why) { LOG("[gfx] null backend resources at tick %u (%s): %d pools, %d live textures, %d texture and %d sampler registrations", tick, why ? why : "", g_npools, (int)g_n.live_tex, (int)g_n.reg_tex, (int)g_n.reg_samp); }

// ---------- Submissions: the effects the CPU sees, completed at the commit ----------
typedef struct { int kind; volatile uint32_t *val; uint32_t mode, v; uint64_t addr; uint32_t type; } post_op;   // kind 1 event write, 2 counter report
static post_op *g_post; static int g_npost, g_cpost;
static uint64_t g_fence, g_done;   // committed / completed (the same on a null GPU, but waits read g_done under the lock)
static SRWLOCK g_fence_lk = SRWLOCK_INIT; static CONDITION_VARIABLE g_fence_cv = CONDITION_VARIABLE_INIT;
static void post(post_op o) {
    if (g_npost == g_cpost && !(g_post = realloc(g_post, (size_t)(g_cpost = g_cpost ? 2 * g_cpost : 64) * sizeof *g_post))) die("null backend: post list allocation failed");
    g_post[g_npost++] = o;
}
void gfx_prepare(void) {}
void gfx_sub_begin(void) { g_npost = 0; atomic_fetch_add(&g_n.submits, 1); }
void gfx_queue_activate(uint64_t serial) { (void)serial; }
void gfx_queue_forget(uint64_t serial) { (void)serial; }
void gfx_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }
void gfx_exec_rest(const uint64_t *handles, int n) { (void)handles; (void)n; }   // no lookahead to feed
static _Atomic int g_red_mem;
static int guest_range(uint64_t a, uint64_t n, const char *what) {   // a buffer range the null GPU may touch: inside a registered pool
    if (!n || gfx_addr(a, n, NULL, NULL)) return 1;
    if (atomic_fetch_add(&g_red_mem, 1) < 8) LOG("[RED] null backend: %s at %#llx+%#llx is in no memory pool: skipped", what, (unsigned long long)a, (unsigned long long)n);
    return 0;
}
void gfx_sub_copy(uint64_t src, uint64_t dst, uint64_t n) {
    atomic_fetch_add(&g_n.copies, 1); atomic_fetch_add(&g_n.copy_bytes, n);
    if (guest_range(src, n, "buffer copy source") && guest_range(dst, n, "buffer copy destination")) {
        dm_guest_write_prepare((void *)(uintptr_t)dst, (size_t)n);
        memmove((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, (size_t)n);
    }
}
void gfx_sub_fill(uint64_t dst, uint64_t n, uint32_t value) {
    atomic_fetch_add(&g_n.fills, 1);
    if (!guest_range(dst, n, "buffer fill")) return;
    dm_guest_write_prepare((void *)(uintptr_t)dst, (size_t)n);
    uint8_t *p = (uint8_t *)(uintptr_t)dst;
    for (uint64_t i = 0; i < n; i++) p[i] = (uint8_t)(value >> 8 * (i & 3));
}
void gfx_sub_event(volatile uint32_t *val, uint32_t mode, uint32_t v) { post((post_op){1, val, mode, v, 0, 0}); }
void gfx_sub_counter(uint64_t addr, uint32_t type) { post((post_op){2, NULL, 0, 0, addr, type}); }
uint64_t gfx_sub_commit(void) {
    // One time for the whole submission (nxnative: the GPU's end of its last segment; here the commit's guest time). A timestamp report (types 0 and 0x10) also carries it in its
    // counter field, as nxnative writes it; every other counter of a GPU that draws nothing is 0
    uint64_t t = nvn_timestamp_from_ns(guest_ns());
    AcquireSRWLockExclusive(&g_fence_lk);
    uint64_t f = ++g_fence;
    for (int i = 0; i < g_npost; i++) {
        post_op *o = &g_post[i];
        if (o->kind == 1) { if (!o->val) continue; if (o->mode == NVN_EVENT_SIGNAL_MODE_ADD) *o->val += o->v; else *o->val = o->v; atomic_fetch_add(&g_n.events, 1); }
        else if (guest_range(o->addr, 16, "counter report")) {
            uint64_t rec[2] = {o->type == 0 || o->type == 0x10 ? t : 0, t};
            dm_guest_write_prepare((void *)(uintptr_t)o->addr, sizeof rec);
            memcpy((void *)(uintptr_t)o->addr, rec, sizeof rec); atomic_fetch_add(&g_n.counters, 1);
        }
    }
    g_npost = 0; g_done = f;
    ReleaseSRWLockExclusive(&g_fence_lk);
    WakeAllConditionVariable(&g_fence_cv);
    return f;
}
uint64_t gfx_last_fence(void) { AcquireSRWLockShared(&g_fence_lk); uint64_t f = g_fence; ReleaseSRWLockShared(&g_fence_lk); return f; }
int gfx_wait(uint64_t fence, uint64_t timeout_ns) {
    AcquireSRWLockExclusive(&g_fence_lk);
    uint64_t t0 = now_ns();
    while (g_done < fence) {   // a fence no submission has reached yet: only another thread's submit can complete it
        uint64_t el = now_ns() - t0; if (el >= timeout_ns) break;
        uint64_t ms = (timeout_ns - el) / 1000000ull; if (ms > 3600000ull) ms = 3600000ull;
        SleepConditionVariableSRW(&g_fence_cv, &g_fence_lk, (DWORD)(ms ? ms : 1), 0);
    }
    int ok = g_done >= fence;
    ReleaseSRWLockExclusive(&g_fence_lk);
    return ok;
}
int gfx_wait_acq(uint64_t fence, uint64_t timeout_ns) { return gfx_wait(fence, timeout_ns); }
void gfx_finish(void) {}
void gfx_drain(void) {}
void gfx_xv_window_tex(uint64_t serial) { (void)serial; }

// ---------- The record walk ----------
static _Atomic int g_red_rec;
static void bad_record(const uint8_t *p, op_hdr h, const char *why) {
    atomic_fetch_add(&g_n.bad, 1);
    if (atomic_fetch_add(&g_red_rec, 1) < 8) LOG("[RED] null backend: %s at %p (op %u, size %u): the rest of this command list is not replayed", why, (const void *)p, h.op, h.size);
}
enum { MAX_RECORDS = 1 << 24 };   // per handle, CALLs included: a JUMP loop is a corrupt list
static int walk(const uint8_t *p, int depth, uint64_t *budget) {
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        if (!(*budget)--) { bad_record(p, h, "more than 16M records in one list (a JUMP loop?)"); return -1; }
        if (h.op >= OP_COUNT_ || h.size < 8 || (h.size & 7) || (h.op != OP_END && h.size > (64u << 20))) { bad_record(p, h, "a malformed record"); return -1; }
        const uint8_t *a = p + 8;
        atomic_fetch_add(&g_n.records, 1); atomic_fetch_add(&g_n.bytes, h.size); atomic_fetch_add(&g_n.op[h.op], 1);
        switch (h.op) {
            case OP_END: return 0;
            case OP_JUMP: { const uint8_t *t; memcpy(&t, a, 8); if (!t) { bad_record(p, h, "a JUMP to 0"); return -1; } p = t; continue; }
            case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (depth >= 8 || !t) { bad_record(p, h, depth >= 8 ? "CALLs nested deeper than 8" : "a CALL of handle 0"); return -1; }
                atomic_fetch_add(&g_n.calls, 1); if (walk((const uint8_t *)(uintptr_t)t, depth + 1, budget)) return -1; break; }
            case OP_SIGNAL_EVENT: { volatile uint32_t *val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, a + 8, 4); memcpy(&v, a + 12, 4); gfx_sub_event(val, m, v); break; }
            case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, a + 8, 8); gfx_sub_counter(ad, (uint32_t)t); break; }
            case OP_FENCE_SYNC: { sync_t *s; memcpy(&s, a, 8); if (s) { s->fence = g_fence + 1; s->fenced = 1; } atomic_fetch_add(&g_n.fences, 1); break; }   // the fence this submission commits with
            case OP_COPY_B2B: { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, a + 8, 8); memcpy(&n, a + 16, 8); gfx_sub_copy(s_, d, n); break; }
            case OP_UPDATE_UB: { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, a + 8, 8);
                if (24 + n > h.size) { bad_record(p, h, "a uniform update longer than its record"); return -1; }
                atomic_fetch_add(&g_n.updates, 1);
                if (guest_range(d, n, "uniform update")) { dm_guest_write_prepare((void *)(uintptr_t)d, (size_t)n); memcpy((void *)(uintptr_t)d, a + 16, (size_t)n); }
                break; }
            case OP_CLEAR_BUF: { const p_clear_buf *cb = (const p_clear_buf *)a; gfx_sub_fill(cb->addr, cb->size, cb->value); break; }
            case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_TEX: atomic_fetch_add(&g_n.draws, 1); break;
            case OP_DISPATCH: atomic_fetch_add(&g_n.dispatches, 1); break;
            case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_TEX: atomic_fetch_add(&g_n.tex_ops, 1); break;
            case OP_BIND_PROGRAM: if (g_bind_hook) { const p_bind_program *bp = (const p_bind_program *)a; if (bp->prog) g_bind_hook(bp->prog, bp->stages); } break;
            default: break;   // state and bindings: nothing to keep on a GPU that draws nothing
        }
        p += h.size;
    }
}
void gfx_exec(const uint8_t *handle) {
    atomic_fetch_add(&g_n.handles, 1);
    if (!handle) { op_hdr z = {0, 0}; bad_record(handle, z, "a null command handle"); return; }
    uint64_t budget = MAX_RECORDS;
    walk(handle, 0, &budget);
}

// ---------- Present: no window. The recording layer's display paces the game (gfx_link_active is 0) ----------
static _Atomic uint64_t g_ticket;
uint64_t gfx_present(const tex_t *t, double period_s, const int crop[4]) {
    (void)t; (void)period_s; (void)crop;
    uint64_t n = atomic_fetch_add(&g_n.presents, 1) + 1;
    if (g_resolve && (n == 600 || n % 1800 == 0)) {
        nxs_counts c; nxs_counts_get(&c);
        LOG("[gfx] shaders up to present %llu: %llu lookups of %llu shaders, %llu keys missing from the cache (%llu dumped, %llu requested, %llu translated since, %llu failed), %llu stale entries",
            (unsigned long long)n, (unsigned long long)c.lookups, (unsigned long long)c.keys, (unsigned long long)c.missing_keys, (unsigned long long)c.dumped, (unsigned long long)c.requested,
            (unsigned long long)c.arrived, (unsigned long long)c.failed, (unsigned long long)c.stale);
    }
    return atomic_fetch_add(&g_ticket, 1) + 1;
}
uint64_t gfx_present_ext(uint64_t serial, double period, void *ev, uint64_t ev_val, const int crop[4]) { (void)serial; (void)period; (void)ev; (void)ev_val; (void)crop; return atomic_fetch_add(&g_ticket, 1) + 1; }
void gfx_present_wait(uint64_t ticket) { (void)ticket; }   // handed over at once
void gfx_present_flush(void) {}
void gfx_present_test_stall(uint64_t ns) { (void)ns; }
void *gfx_device(void) { return NULL; }
void *gfx_device_classic(void) { return NULL; }
uint64_t gfx_ext_tex(void *tex) { (void)tex; return 0; }
int gfx_display_hz(void) { return 0; }
void gfx_set_tex_pool_n(int n) { g_tex_pool_n = n; }
void gfx_set_frame(uint64_t f) { (void)f; }
void gfx_ub_present(double iv_ms) { (void)iv_ms; }
double gfx_gpu_ms(uint64_t f) { (void)f; return 0; }   // a null GPU takes no time
void gfx_ab_gpu(uint64_t f, double gpu_ms, double frame_ms) { (void)f; (void)gpu_ms; (void)frame_ms; }
void xv_test_counts(int *dc, int *viol) { *dc = *viol = 0; }
int gfx_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; *t0 = *t1 = 0; return 0; }
void gfx_stats(uint64_t *passes, uint64_t *draws) { *passes = 0; *draws = g_n.draws; }
void gfx_args_stats(uint64_t *enc, uint64_t *reuse) { *enc = *reuse = 0; }
void gfx_use_stats(uint64_t *call, uint64_t *skip) { *call = *skip = 0; }
uint64_t gfx_ds_omit(void) { return 0; }
uint64_t gfx_drawable_ns_take(void) { return 0; }
void gfx_bw(uint64_t *load, uint64_t *store, uint64_t *full_rw_passes) { *load = *store = *full_rw_passes = 0; }
uint64_t gfx_udc_saved_bytes(void) { return 0; }
int gfx_has_window(void) { return 0; }
int gfx_fps_target(void) { return 60; }
int gfx_warm_wait(int timeout_ms) { (void)timeout_ms; return 1; }
int gfx_link_active(void) { return 0; }
int gfx_link_wait_tick(int vsyncs, uint64_t timeout_ns) { (void)vsyncs; (void)timeout_ns; return 0; }
int gfx_link_wait_ticks(int ticks, uint64_t timeout_ns) { (void)ticks; (void)timeout_ns; return 0; }
int gfx_link_rate(void) { return 0; }
double gfx_presented_time(uint64_t n) { (void)n; return 0; }
int gfx_link_tick(uint64_t n, double *target, double *lead_ms, int *ticks, double *gap_ms) { (void)n; (void)target; (void)lead_ms; (void)ticks; (void)gap_ms; return 0; }
int gfx_link_stalled(void) { return 0; }
int gfx_snap_on(void) { return 0; }
uint64_t gfx_snap_acquire_wait(void) { return 0; }

// ---------- Resources: registrations are counted; a texel read comes from the storage ----------
void gfx_tex_init(const tex_t *t) { (void)t; atomic_fetch_add(&g_n.live_tex, 1); }
void gfx_tex_release(tex_t *t) { (void)t; atomic_fetch_sub(&g_n.live_tex, 1); }
void gfx_pkg_stats(uint64_t *uploads, uint64_t *reuploads) { *uploads = *reuploads = 0; }
unsigned gfx_cpu_tex_count(void) { return (unsigned)g_n.cpu_written; }
uint64_t gfx_pkg_edge_skips(void) { return 0; }
uint64_t gfx_pkg_miss(void) { return 0; }
uint64_t gfx_upload_ns_take(void) { return 0; }
uint64_t gfx_upload_gpu_ns_take(void) { return 0; }
void gfx_create_stats(uint64_t *lib, uint64_t *pso) { *lib = *pso = 0; }
static _Atomic int g_red_texel;
void gfx_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    if (!nvn_texio_read(t, level, layer0, r, dst, row, img)) return;
    if (!atomic_fetch_add(&g_red_texel, 1)) LOG("[RED] null backend: ReadTexels %dx%d of level %u, format %#x, target %d cannot be read from the storage: zeros", r->width, r->height, level, t->d.format, t->d.target);
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (!bpb || !dst || r->width <= 0 || r->height <= 0) return;
    size_t wb = (size_t)((r->width + bw - 1) / bw) * (size_t)bpb, hb = (size_t)((r->height + bh - 1) / bh), nz = r->depth > 0 ? (size_t)r->depth : 1;
    if (!row) row = wb;
    if (!img) img = row * hb;
    memset(dst, 0, (nz - 1) * img + (hb - 1) * row + wb);
}
void gfx_reg_tex(int id, const tex_t *t, const view_t *v) { (void)id; (void)t; (void)v; atomic_fetch_add(&g_n.reg_tex, 1); }
void gfx_reg_tex_gone(int id) { (void)id; }
void gfx_reg_image(int id, const tex_t *t, const view_t *v) { (void)id; (void)t; (void)v; atomic_fetch_add(&g_n.reg_img, 1); }
void gfx_reg_samp(int id, const samp_par *p) { (void)id; (void)p; atomic_fetch_add(&g_n.reg_samp, 1); }
void gfx_tex_cpu_written(const tex_t *t) { (void)t; atomic_fetch_add(&g_n.cpu_written, 1); }
int gfx_tex_gpu_owned(uint64_t serial) { (void)serial; return 0; }
void gfx_prog_release(prog_t *p) { (void)p; atomic_fetch_add(&g_n.progs_released, 1); }

#include "gfx_ops_table.h"
