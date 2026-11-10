// The Vulkan backend's side of the contract (gfx_backend.h, nvn_obj.h), chosen with nxrt --gfx vulkan (gfx/gfx_ops.h). M5 track B: device, window,
// presentation, memory, textures and the pipeline cache; M5-C: the replay of the command streams (vk_replay.c), the screenshot facility (vk_shot.c).
// Fences, events and counters follow completion (the contract's ordering rule, gfx_backend.h).
#define GFX_BE vk
#include "gfx_rename.h"
#include "gfx_ops.h"
#include "vk_internal.h"
#include "vk_window.h"
#include "vk_submit.h"
#include "vk_present.h"
#include "vk_mem.h"
#include "vk_tex.h"
#include "vk_fmt.h"
#include "vk_astc.h"
#include "vk_replay.h"
#include "vk_shot.h"
#include "vk_flight.h"
#include "ctl_gfx.h"
#include <process.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *nxrt_log_path(void);
_Noreturn void nxrt_die(const char *fmt, ...);
int vkpc_init(void); void vkpc_shutdown(void); void vkpc_stats(uint64_t *created, uint64_t *saves, uint64_t *bytes, uint64_t *loaded);

static INIT_ONCE g_once = INIT_ONCE_STATIC_INIT;
static int g_window, g_sync_all, g_cpu_effects;
static int g_origin_ll = 1;
static uint64_t g_frame;
static _Atomic uint64_t s_fences;

static void on_close(void) { VKLOG("window closed: ending the run"); nxrt_logf("window closed by the user"); ExitProcess(0); }
// DM_CTL's renderer commands (host/ctl_gfx.h)
static void ctl_texdump(const char *path) { (void)path; vkr_texdump_next(); VKLOG("DM_CTL texdump: the next frame's images, beside the log (NXRT_VK_TEXDUMP's names)"); }
static void ctl_passdump(long n) { vkr_passdump_next(n); VKLOG("DM_CTL passdump: the next %ld frames, in this log", n); }

static BOOL CALLBACK start(PINIT_ONCE o, PVOID p, PVOID *c) {
    (void)o; (void)p; (void)c;
    const char *e;
#ifdef NDEBUG
    int validation = 0;
#else
    int validation = 1;   // debug builds validate (docs/m5-vulkan.md §2)
#endif
    if ((e = getenv("NXRT_VK_VALIDATION"))) validation = atoi(e) != 0;
    int headless = (e = getenv("NXRT_VK_HEADLESS")) && atoi(e);
    int mode = VKP_FIFO;
    if ((e = getenv("NXRT_VK_PRESENT"))) mode = !strcmp(e, "mailbox") ? VKP_MAILBOX : !strcmp(e, "immediate") ? VKP_IMMEDIATE : VKP_FIFO;
    int w = 1280, h = 720;
    if ((e = getenv("NXRT_VK_WINDOW"))) sscanf(e, "%dx%d", &w, &h);
    if ((e = getenv("NXRT_VK_SYNC"))) g_sync_all = !strcmp(e, "all");   // per use is the default: the replay brings every range it binds up to date
    g_cpu_effects = (e = getenv("NXRT_VK_CPU_EFFECTS")) && atoi(e);
    uint32_t nx = 0; const char *const *ex = NULL;
    if (!headless) {
        if (vkw_open("Pokopia (pokopia-native, Vulkan)", w, h, 0)) nxrt_die("--gfx vulkan: the window could not be opened (see the log)");
        vkw_set_close_handler(on_close);
        if ((e = getenv("NXRT_VK_FULLSCREEN")) && atoi(e)) vkw_request_fullscreen(1);   // scripts/play.ps1 -Fullscreen (F11 / Alt+Enter toggle it)
        ex = vkw_instance_exts(&nx);
        g_window = 1;
    }
    vk_dev_opts dopt = {validation, headless, getenv("NXRT_VK_DEVICE")};
    if (vk_dev_open(&dopt, ex, nx)) nxrt_die("--gfx vulkan: no usable Vulkan 1.3 device (see the log)");
    vkf_init(); vkf_log_table();
    if (vks_init() || vkpc_init() || vkm_init() || vkt_init() || vka_init(NULL) || vkr_init()) nxrt_die("--gfx vulkan: the backend could not start (see the log)");
    if (g_window && vkp_init(mode)) nxrt_die("--gfx vulkan: no swapchain on the window (see the log)");
    vks_shot_init();
    dm_gfx_ctl = (dm_gfx_ctl_ops){vks_shot_request, ctl_texdump, ctl_passdump};
    VKLOG("backend up: %s; pool sync %s; the NVN command streams are replayed",
          g_window ? "window" : "headless", g_sync_all ? "every written page each frame within the budget (NXRT_VK_SYNC=all)" : "per use (each range as the replay binds it)");
    return TRUE;
}
static void up(void) { InitOnceExecuteOnce(&g_once, start, NULL, NULL); }

void gfx_init(void) { up(); }
uint64_t vk_game_frame(void) { return g_frame; }
int vk_origin_lower_left(void) { return g_origin_ll; }   // vk_shot.c: GL's bottom-up rows are turned over in screenshots
void gfx_vbuf_stats(uint64_t *resolve, uint64_t *reuse, uint64_t *set, uint64_t *skip, uint64_t *copies, uint64_t *copy_bytes) { *resolve = *reuse = *set = *skip = *copies = *copy_bytes = 0; }
void gfx_set_depth_zero(int on) { vkr_set_depth_zero(on); }
void gfx_set_origin_lower_left(int on) { g_origin_ll = on; vkr_set_origin_lower_left(on); }
void gfx_set_queue_origin_lower_left(int on) { vkr_set_queue_origin_lower_left(on); }
void bf_want_note(const char *key) { (void)key; }

// ---------- Memory pools ----------
void gfx_pool_add(const void *mem, uint64_t size) { up(); vkm_pool_add(mem, size); }
void gfx_pool_del(const void *mem) { vkm_pool_del(mem); }
void gfx_alias_drop_pool(const void *pool) { vkt_drop_pool(pool); }
int gfx_addr(uint64_t addr, uint64_t len, void **buf, uint64_t *off) {
    VkBuffer b; VkDeviceSize o;
    if (!vkm_addr(addr, len, &b, &o)) return 0;
    if (buf) *buf = (void *)b;
    if (off) *off = o;
    return 1;
}
void gfx_res_snapshot(gfx_res_count *o) {
    memset(o, 0, sizeof *o);
    vkm_stats_t m; vkm_stats(&m); vkt_stats_t t; vkt_stats(&t);
    o->pools = (uint32_t)m.pools; o->pool_adds = m.pools; o->live_tex = (uint32_t)t.images;
}
void gfx_res_counts(uint32_t tick, const char *why) {
    vkm_stats_t m; vkm_stats(&m); vkt_stats_t t; vkt_stats(&t);
    VKLOG("resources at tick %u (%s): %llu pools (%.1f MB, %.1f MB backed), %llu images (%.1f MB), %llu samplers", tick, why ? why : "",
          (unsigned long long)m.pools, m.pool_bytes / 1048576.0, m.bound_bytes / 1048576.0, (unsigned long long)t.images, t.image_bytes / 1048576.0, (unsigned long long)t.samplers);
}

// ---------- Submissions ----------
static sync_t **g_syncs; static int g_nsync, g_csync;   // fences recorded by this submission: they get its value at the commit
void gfx_prepare(void) { up(); }
void gfx_sub_begin(void) { up(); vks_begin(); g_nsync = 0; vkt_in_submission(); vkr_sub_begin(); }
void gfx_queue_activate(uint64_t serial) { vkr_queue_activate(serial); }
void gfx_queue_forget(uint64_t serial) { vkr_queue_forget(serial); }
void gfx_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }
void gfx_exec_rest(const uint64_t *handles, int n) { (void)handles; (void)n; }

void gfx_sub_copy(uint64_t src, uint64_t dst, uint64_t n) {
    if (g_cpu_effects || !vks_cb()) { void dm_guest_write_prepare(void *addr, size_t len);
        if (gfx_addr(src, n, NULL, NULL) && gfx_addr(dst, n, NULL, NULL)) { dm_guest_write_prepare((void *)(uintptr_t)dst, (size_t)n); memmove((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, (size_t)n); }
        return; }
    vkr_copy(src, dst, n);
}
void gfx_sub_fill(uint64_t dst, uint64_t n, uint32_t value) { vkr_fill(dst, n, value); }
void vk_note_fence_sync(sync_t *s) {
    atomic_fetch_add(&s_fences, 1);
    if (g_nsync == g_csync) { g_csync = g_csync ? 2 * g_csync : 32; g_syncs = realloc(g_syncs, (size_t)g_csync * sizeof *g_syncs); }
    g_syncs[g_nsync++] = s;
}
void gfx_sub_event(volatile uint32_t *val, uint32_t mode, uint32_t v) { vks_post_event(val, mode == NVN_EVENT_SIGNAL_MODE_ADD ? 1 : 0, v); }
void gfx_sub_counter(uint64_t addr, uint32_t type) { vks_post_counter(addr, type); }
uint64_t gfx_sub_commit(void) {
    static uint64_t synced_frame = ~0ull;   // once a frame: until the replay asks for ranges as it uses them (vkm_use)
    vkr_sub_end();
    if (g_sync_all && synced_frame != g_frame) { synced_frame = g_frame; vkm_sync_all(); }
    uint64_t v = vks_commit();
    for (int i = 0; i < g_nsync; i++) { g_syncs[i]->fence = v; g_syncs[i]->fenced = 1; }
    g_nsync = 0;
    return v;
}
uint64_t gfx_last_fence(void) { return atomic_load(&g_vk.tl_assigned); }
int gfx_wait(uint64_t fence, uint64_t timeout_ns) { return vks_wait(fence, timeout_ns); }
int gfx_wait_acq(uint64_t fence, uint64_t timeout_ns) { return vks_wait(fence, timeout_ns); }
void gfx_finish(void) { if (g_vk.up) vks_finish(); }
void gfx_drain(void) { if (!g_vk.up) return; vkp_flush(); vks_finish(); }
void gfx_xv_window_tex(uint64_t serial) { (void)serial; }

// ---------- The record walk: the replay (vk_replay.c) ----------
void gfx_exec(const uint8_t *handle) { vkr_exec(handle); }

// ---------- Presentation ----------
static void frame_tick(void) {
    uint64_t f = ++g_frame;
    vkm_frame_tick(f); vkt_frame(f); vkr_frame(f); vkf_frame(f);
    if (f % 3000 == 0) {
        uint64_t subs, qsubs, empty, posted; vks_stats(&subs, &qsubs, &empty, &posted);
        uint64_t pc, ps, pb, pl; vkpc_stats(&pc, &ps, &pb, &pl);
        vkr_stats_t r; vkr_stats(&r);
        VKLOG("frame %llu: %llu submissions in %llu queue submits; replayed %llu records: %llu draws (%llu skipped), %llu dispatches, %llu passes, %llu texture copies, %llu copies, "
              "%llu uniform updates, %llu fills, %llu fences; pipeline cache %llu bytes, %llu saves; validation errors %u warnings %u", (unsigned long long)f, (unsigned long long)subs,
              (unsigned long long)qsubs, (unsigned long long)r.records, (unsigned long long)r.draws, (unsigned long long)r.draws_skipped, (unsigned long long)r.dispatches,
              (unsigned long long)r.passes, (unsigned long long)r.tex_copies, (unsigned long long)r.copies, (unsigned long long)r.updates, (unsigned long long)r.fills,
              (unsigned long long)atomic_load(&s_fences), (unsigned long long)pb, (unsigned long long)ps, atomic_load(&g_vk.val_errors), atomic_load(&g_vk.val_warnings));
    }
}
uint64_t gfx_present(const tex_t *t, double period_s, const int crop[4]) {
    up(); frame_tick();
    if (t && t->serial) vkt_resolve(t);   // a texture presented without being drawn into this run still has an image to show
    vks_shot_present(t, g_frame, crop);
    vkc_start();
    uint64_t tk = vkp_enqueue(t ? t->serial : 0, period_s, atomic_load(&g_vk.tl_assigned), crop, g_origin_ll);
    vkc_gate(g_frame);
    return tk;
}
uint64_t gfx_present_ext(uint64_t serial, double period, void *ev, uint64_t ev_val, const int crop[4]) {
    (void)ev; (void)ev_val;   // external producers (nxnative's Vulkan / GL layers for other games) draw on our queue: its timeline orders them
    up(); frame_tick();
    return vkp_enqueue(serial, period, atomic_load(&g_vk.tl_assigned), crop, 0);
}
void gfx_present_wait(uint64_t ticket) { vkp_wait(ticket); }
void gfx_present_flush(void) { vkp_flush(); }
void gfx_present_test_stall(uint64_t ns) { vkp_set_stall(ns); }
void *gfx_device(void) { up(); return (void *)g_vk.dev; }
void *gfx_device_classic(void) { return gfx_device(); }
uint64_t gfx_ext_tex(void *tex) { (void)tex; static int n; if (!n++) VKLOG("external textures (gfx_ext_tex) are for nxnative's Vulkan / GL layers, which Pokopia does not use"); return 0; }
int gfx_display_hz(void) { return vkw_display_hz(); }
void gfx_set_tex_pool_n(int n) { vkt_set_pool_n(n); }
void gfx_set_frame(uint64_t f) { (void)f; }
void gfx_ub_present(double iv_ms) { (void)iv_ms; }
double gfx_gpu_ms(uint64_t f) { (void)f; return -1; }
void gfx_ab_gpu(uint64_t f, double gpu_ms, double frame_ms) { (void)f; (void)gpu_ms; (void)frame_ms; }
void xv_test_counts(int *dc, int *viol) { *dc = *viol = 0; }
int gfx_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; *t0 = *t1 = 0; return 0; }
void gfx_stats(uint64_t *passes, uint64_t *draws) { vkr_stats_t r; vkr_stats(&r); *passes = r.passes; *draws = r.draws; }
void gfx_args_stats(uint64_t *enc, uint64_t *reuse) { *enc = *reuse = 0; }
void gfx_use_stats(uint64_t *call, uint64_t *skip) { *call = *skip = 0; }
uint64_t gfx_ds_omit(void) { return 0; }
uint64_t gfx_drawable_ns_take(void) { return 0; }
void gfx_bw(uint64_t *load, uint64_t *store, uint64_t *full_rw_passes) { *load = *store = *full_rw_passes = 0; }
uint64_t gfx_udc_saved_bytes(void) { return 0; }
int gfx_has_window(void) { return vkw_is_open(); }
int gfx_fps_target(void) { const char *e = getenv("DM_FPS"); int v = e ? atoi(e) : 0; return v == 30 ? 30 : 60; }
int gfx_warm_wait(int timeout_ms) { (void)timeout_ms; return 1; }
// No display link: the recording layer paces the game on its own clock (as with the null backend) and FIFO presentation follows the display
int gfx_link_active(void) { return 0; }
int gfx_link_wait_tick(int vsyncs, uint64_t timeout_ns) { (void)vsyncs; (void)timeout_ns; return 0; }
int gfx_link_wait_ticks(int ticks, uint64_t timeout_ns) { (void)ticks; (void)timeout_ns; return 0; }
int gfx_link_rate(void) { return 0; }
double gfx_presented_time(uint64_t n) { return vkp_presented_time(n); }
int gfx_link_tick(uint64_t n, double *target, double *lead_ms, int *ticks, double *gap_ms) { (void)n; (void)target; (void)lead_ms; (void)ticks; (void)gap_ms; return 0; }
int gfx_link_stalled(void) { return 0; }
int gfx_snap_on(void) { return 0; }
uint64_t gfx_snap_acquire_wait(void) { return 0; }

// ---------- Textures ----------
void gfx_tex_init(const tex_t *t) { vkt_tex_init(t); }
void gfx_tex_release(tex_t *t) { vkt_tex_release(t); }
void gfx_pkg_stats(uint64_t *uploads, uint64_t *reuploads) { vkt_stats_t s; vkt_stats(&s); *uploads = s.uploads; *reuploads = s.reuploads; }
unsigned gfx_cpu_tex_count(void) { vkt_stats_t s; vkt_stats(&s); return (unsigned)s.cpu_written; }
uint64_t gfx_pkg_edge_skips(void) { return 0; }
uint64_t gfx_pkg_miss(void) { return 0; }
uint64_t gfx_upload_ns_take(void) { static uint64_t last; vkt_stats_t s; vkt_stats(&s); uint64_t d = s.upload_ns - last; last = s.upload_ns; return d; }
uint64_t gfx_upload_gpu_ns_take(void) { return 0; }
void gfx_create_stats(uint64_t *lib, uint64_t *pso) { *lib = 0; uint64_t c, s, b, l; vkpc_stats(&c, &s, &b, &l); *pso = c; }
void gfx_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) { vkt_read_texels(t, level, layer0, r, dst, row, img); }
void gfx_reg_tex(int id, const tex_t *t, const view_t *v) { vkt_reg_tex(id, t, v, 0); }
void gfx_reg_tex_gone(int id) { vkt_reg_tex_gone(id); }
void gfx_reg_image(int id, const tex_t *t, const view_t *v) { vkt_reg_tex(id, t, v, 1); }
void gfx_tex_cpu_written(const tex_t *t) { vkt_tex_cpu_written(t); }
int gfx_tex_gpu_owned(uint64_t serial) { return vkt_tex_gpu_owned(serial); }
void gfx_reg_samp(int id, const samp_par *p) { vkt_reg_samp(id, p); }
void gfx_prog_release(prog_t *p) { (void)p; }

#include "gfx_ops_table.h"
