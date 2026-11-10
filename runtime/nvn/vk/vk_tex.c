// Textures: images, views, uploads through s2tex, sharing, the CPU-written rule, samplers, registrations (vk_tex.h; docs/m5-vulkan.md §4).
// Behaviour follows nxnative's Metal backend (nvn_mtl.m: tex_mtl_, the g_alias sharing table and alias_take_over, mtl_alias_drop_pool, tex_from_storage,
// tex_cpu_content, mtl_tex_cpu_written, the packaged-texture dirty check); the Vulkan side and the IMAGE rule for block-compressed textures are ours.
// Storage addressing is nvn_tex.c's (the recording layer's), the byte order inside GOBs s2tex's (Switch 2), through gfx/nvn_texio.c.
#include "vk_tex.h"
#include "vk_mem.h"
#include "vk_submit.h"
#include "vk_astc.h"
#include "vk_hmap.h"
#include "nvn_dirty.h"
#include "nvn_texio.h"
#include "s2tex.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint64_t pool; int64_t off; int32_t target, format, samples, cls, storage_img, pkg_bh; uint32_t w, h, d, levels, stride, layout_flags; } skey;
volatile LONG g_vkt_reg_gen;   // see vkt_reg_tex
static vk_hmap g_share = {sizeof(skey)}, g_serial = {8}, g_live = {8}, g_cpu = {8}, g_samp = {sizeof(samp_par)};
static SRWLOCK g_lk = SRWLOCK_INIT;   // every table here; the replay and the present thread take it
static vkt_reg *g_reg; static int g_nreg; static samp_par *g_sreg; static uint8_t *g_sreg_ok; static int g_nsreg;
static int g_pool_n;   // the texture pool's capacity (IDs past it read as empty: the replay's descriptor lookups)
int vkt_pool_n(void) { return g_pool_n; }
static uint64_t g_frame;
static _Atomic uint64_t s_images, s_bytes, s_uploads, s_reuploads, s_up_bytes, s_up_ns, s_shared, s_astc_fast, s_astc_high, s_cpuw, s_evicted, s_samplers;
int vkc_int(const char *name, int def);
static _Atomic uint64_t s_up_ns_sf, s_up_ns_sr, s_up_n_sastc;   // upload time of CPU-storage textures: first upload, again after a change; ASTC ones among them
static _Atomic uint64_t s_bc1_images, s_bc1_redo, s_bc1_saved;   // opaque ASTC as BC1: images, made again as BC3, upload bytes saved
static _Atomic uint64_t s_pf_jobs, s_pf_dropped;
static SRWLOCK g_pf_lk = SRWLOCK_INIT;   // the prefetch queue and what it made (below)
// What the prefetch made of a texture, and when: its levels' keys, and the dirty tracker's stamp taken (and the pages protected) before the storage was read.
// The upload takes the levels by key while no page was written since: no second deswizzle and hash of the storage, no round trip through the workers
typedef struct { uint64_t stamp; const void *span; uint32_t n; struct { uint16_t level, layer; uint32_t w, h; vka_key k; } lv[]; } pf_ready;
static vk_hmap g_pf_ready = {8};   // pf_key -> pf_ready (g_pf_lk)
static _Atomic uint64_t s_pf_fast, s_pf_fast_miss;
static uint64_t pf_key(const void *mem, const nvn_texdesc *d);
static int pf_fast_on(void) { static int v = -1; if (v < 0) v = vkc_int("pf_fast", 0); return v; }
static pf_ready *ready_take(const tex_t *t) {
    uint64_t hk = pf_key(t->mem, &t->d); AcquireSRWLockExclusive(&g_pf_lk);
    pf_ready *r = vk_hm_get(&g_pf_ready, &hk); if (r) vk_hm_del(&g_pf_ready, &hk);
    ReleaseSRWLockExclusive(&g_pf_lk); return r;
}
   // texture prefetch: textures done, refused (queue full)
static _Atomic uint64_t s_tm_read, s_tm_conv, s_tm_stage, s_tm_join, s_ev_back, s_ev_back_b;   // where upload() spends its time; textures that came back after eviction
static vk_hmap g_evicted = {8};   // storage address of each texture evicted so far
// ASTC levels / layers still at the fast pass: where their high-quality data goes
typedef struct { vka_key key; vkt_image *im; uint32_t level, layer, w, h, slices; } upgrade_t;
static upgrade_t *g_upg; static int g_nupg, g_cupg;
static vka_key *g_ready_key; static uint8_t **g_ready_data; static size_t *g_ready_size; static int g_nready, g_cready;

static uint32_t mip(uint32_t s, unsigned l) { s >>= l; return s ? s : 1; }

// ---------- Which content an NVN texture has ----------
static int pool_cpu_writable(const pool_t *p) { return p && !(p->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL)); }
static int cpu_written(uint64_t serial) { return serial && vk_hm_get(&g_cpu, &serial) != NULL; }   // caller holds g_lk
// From storage: a packaged layout; textures the CPU wrote (flushes, WriteTexels); and every block-compressed texture of a CPU-writable pool, IMAGE flag or
// not. A BC or ASTC texture cannot be a render target or a storage image, so its storage is the only place its content can be: Pokopia's icons are BC1 /
// BC7 textures flagged IMAGE whose storage the CPU rewrites, and treating IMAGE as "GPU-owned" left them stale (research/01 lesson 17)
static int from_storage(const tex_t *t) {
    if (!t->mem || t->d.target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    if (t->has_layout || cpu_written(t->serial)) return 1;
    // A linear texture in a CPU-writable pool is pool memory the CPU writes in place (nxnative makes it a view of the pool's buffer: zero-copy). Pokopia's 2x1
    // exposure texture is one: the game writes it on the CPU each frame and a draw copies it into a render target (M5-C)
    if ((t->d.flags & NVN_TEXTURE_FLAGS_LINEAR) && !(t->d.flags & NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET) && pool_cpu_writable(t->pool) && t->d.samples <= 1) return 1;
    const vkf_info *fi = vkf_get(t->d.format);
    return fi && (fi->flags & VKF_COMPRESSED) && pool_cpu_writable(t->pool) && t->d.samples <= 1 &&
           !(t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_SPARSE));
}
static void key_of(const tex_t *t, int cls, skey *k) {
    memset(k, 0, sizeof *k);
    k->pool = (uint64_t)(uintptr_t)t->pool; k->off = t->pool ? t->off : (int64_t)(uintptr_t)t->mem;
    k->target = t->d.target; k->format = t->d.format;
    // One memory, one picture: a render target and its sRGB twin over the same storage are one image with two views, as on the console. As two images, what was drawn through
    // one never reached the other: a movie drawn into an RGBA8 target and sampled as RGBA8_SRGB stayed on the first picture. NXRT_VK_FORMAT_SHARE=0: two images (a test)
    { static int on = -1; if (on < 0) { const char *e = getenv("NXRT_VK_FORMAT_SHARE"); on = !e || *e != '0'; }
      const vkf_info *fi = on && cls == VKT_GPU ? vkf_get(t->d.format) : NULL;
      if (fi && fi->conv == VKF_COPY && !(fi->flags & (VKF_COMPRESSED | VKF_DEPTH | VKF_STENCIL))) k->format = 0x10000 + (int)vkf_unorm_of(fi->vk); }
    k->samples = t->d.samples > 1 ? t->d.samples : 1; k->cls = cls;
    k->storage_img = (t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && !vkf_is_compressed(t->d.format);   // IMAGE adds storage use; it changes nothing for BC / ASTC
    k->pkg_bh = t->has_layout ? t->d.pkg_block_h : 0;
    k->w = t->d.width; k->h = t->d.height; k->d = t->d.depth; k->levels = t->d.levels ? t->d.levels : 1; k->stride = t->d.stride;
    k->layout_flags = t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_SPARSE);
}

// ---------- Image creation ----------
static void shape(const nvn_texdesc *d, VkImageType *type, VkImageViewType *vt, uint32_t *w, uint32_t *h, uint32_t *depth, uint32_t *layers, int *cube) {
    *w = d->width ? d->width : 1; *h = d->height ? d->height : 1; *depth = 1; *layers = 1; *cube = 0; *type = VK_IMAGE_TYPE_2D; *vt = VK_IMAGE_VIEW_TYPE_2D;
    switch (d->target) {
    case NVN_TEXTURE_TARGET_1D: *type = VK_IMAGE_TYPE_1D; *vt = VK_IMAGE_VIEW_TYPE_1D; *h = 1; break;
    case NVN_TEXTURE_TARGET_1D_ARRAY: *type = VK_IMAGE_TYPE_1D; *vt = VK_IMAGE_VIEW_TYPE_1D_ARRAY; *layers = *h; *h = 1; break;   // the layer count is in height
    case NVN_TEXTURE_TARGET_3D: *type = VK_IMAGE_TYPE_3D; *vt = VK_IMAGE_VIEW_TYPE_3D; *depth = d->depth ? d->depth : 1; break;
    case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY: *vt = VK_IMAGE_VIEW_TYPE_2D_ARRAY; *layers = d->depth ? d->depth : 1; break;
    case NVN_TEXTURE_TARGET_CUBEMAP: *vt = VK_IMAGE_VIEW_TYPE_CUBE; *layers = 6; *cube = 1; break;
    case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: *vt = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY; *layers = d->depth ? d->depth : 6; *cube = 1; break;   // layer-faces, as in GL
    default: break;   // 2D, rectangle, multisample
    }
}

static VkImageView make_view(vkt_image *im, VkFormat f, VkImageViewType vt, VkComponentMapping swz, uint32_t base_level, uint32_t nlevels, uint32_t base_layer, uint32_t nlayers, VkImageAspectFlags aspect) {
    VkImageViewCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, NULL, 0, im->img, vt, f, swz, {aspect, base_level, nlevels, base_layer, nlayers}};
    // A view in a format the image's storage use does not allow (an sRGB view of an RGBA8 image that compute shaders write: VUID-VkImageViewCreateInfo-usage-02275,
    // seen in gameplay) leaves storage out of the view's usage
    VkImageViewUsageCreateInfo uci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, NULL, im->usage & ~(VkImageUsageFlags)VK_IMAGE_USAGE_STORAGE_BIT};
    if (im->usage & VK_IMAGE_USAGE_STORAGE_BIT) {
        VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_vk.pd, f, &fp);
        if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && uci.usage) ci.pNext = &uci;
    }
    VkImageViewMinLodCreateInfoEXT ml = {VK_STRUCTURE_TYPE_IMAGE_VIEW_MIN_LOD_CREATE_INFO_EXT, NULL, 0};
    if (im->min_lod) { uint32_t top = base_level + nlevels - 1; ml.minLod = (float)(im->min_lod < top ? im->min_lod : top); ml.pNext = ci.pNext; ci.pNext = &ml; }
    VkImageView v = VK_NULL_HANDLE;
    if (vkCreateImageView(g_vk.dev, &ci, NULL, &v) != VK_SUCCESS) return VK_NULL_HANDLE;
    return v;
}

typedef struct vkt_sparse vkt_sparse;
#define MIN_(a, b) ((a) < (b) ? (a) : (b))
extern _Atomic uint64_t g_alloc_ns, g_alloc_n;
static void prefetch_forget(const vkt_image *o);   // an evicted texture may be prefetched again (below)
static size_t stage_bc(struct vkt_image *im, const uint8_t *data, size_t size, uint8_t *dst);   // opaque ASTC as BC1 (below)
#include "vk_sparse.inc"
// one line every 5 s of wall time (nvn_census_tick's caller): where the replay's time goes while textures stream
uint64_t vkt_up_ns(void) { return atomic_load(&s_up_ns); }
void vkt_log_stats(void) {
    static uint64_t pu, pb, pn, pa, pat, pac; uint64_t u = atomic_load(&s_uploads) + atomic_load(&s_reuploads), b = atomic_load(&s_up_bytes), n = atomic_load(&s_up_ns), a = atomic_load(&g_alloc_ns), ac = atomic_load(&g_alloc_n);
    vka_stats_t st; vka_stats(&st);
    uint64_t bud = 0, use = 0; vk_budget(&bud, &use);
    VKLOG("textures: %llu images, %llu MB; last 5 s: %llu uploads, %llu MB, %.0f ms in upload (ASTC %.0f ms, %.1f Mtexels), %llu allocations %.0f ms; device memory %llu of %llu MB, tiles %llu MB", (unsigned long long)atomic_load(&s_images), (unsigned long long)(atomic_load(&s_bytes) >> 20),
          (unsigned long long)(u - pu), (unsigned long long)((b - pb) >> 20), (n - pn) / 1e6, (st.fast_ns - pat) / 1e6, (st.fast_texels - pa) / 1e6, (unsigned long long)(ac - pac), (a - (uint64_t)0) / 1e6 * 0 + (double)0, (unsigned long long)(use >> 20), (unsigned long long)(bud >> 20), (unsigned long long)(g_sp.live * SP_TILE >> 20));
    pu = u; pb = b; pn = n; pat = st.fast_ns; pa = st.fast_texels; pac = ac;
    { void vks_wait_report(void), kernel_svc_report(void), vk_memreport_log(void); vks_wait_report(); kernel_svc_report(); vk_memreport_log(); }
    { uint64_t rt = 0, cmp = 0, oth = 0, sysb = 0, rts = 0, age[5] = {0}, cage[5] = {0}; uint32_t big = 0, nrt = 0, nrts = 0; AcquireSRWLockShared(&g_lk);
      for (uint32_t i = 0; i < g_share.cap; i++) { vkt_image *o = g_share.vals[i]; if (!o || o->sp) continue;
          uint64_t z = o->alloc.size; if (o->alloc.sys) { sysb += z; if (o->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) { rts += z; nrts++; } continue; }
          uint64_t idle = g_frame - o->last_frame; int ab = idle < 60 ? 0 : idle < 600 ? 1 : idle < 3600 ? 2 : idle < 14400 ? 3 : 4;
          if (o->fi->flags & VKF_COMPRESSED) { cmp += z; cage[ab] += z; } else if (o->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) { rt += z; nrt++; age[ab] += z; if (z >= (32u << 20)) big++; } else oth += z; }
      ReleaseSRWLockShared(&g_lk);
      VKLOG("idle for under 60 / 600 / 3600 / 14400 frames / longer: render targets %llu / %llu / %llu / %llu / %llu MB, compressed textures %llu / %llu / %llu / %llu / %llu MB", (unsigned long long)(age[0] >> 20), (unsigned long long)(age[1] >> 20), (unsigned long long)(age[2] >> 20), (unsigned long long)(age[3] >> 20), (unsigned long long)(age[4] >> 20), (unsigned long long)(cage[0] >> 20), (unsigned long long)(cage[1] >> 20), (unsigned long long)(cage[2] >> 20), (unsigned long long)(cage[3] >> 20), (unsigned long long)(cage[4] >> 20));
      uint64_t dbud = 0, duse = 0; vk_budget(&dbud, &duse);   // the driver's own figures: over its budget, Windows moves video memory out behind our back
      VKLOG("device memory by kind, %llu MB by our count: render targets %llu MB (%u, %u of 32 MB or more), compressed textures %llu MB, other textures %llu MB, tiles %llu MB; textures in system memory %llu MB, of which %u render targets %llu MB; the driver: %llu of %llu MB%s", (unsigned long long)(vk_device_used() >> 20), (unsigned long long)(rt >> 20), nrt, big, (unsigned long long)(cmp >> 20), (unsigned long long)(oth >> 20), (unsigned long long)(g_sp.live * SP_TILE >> 20), (unsigned long long)(sysb >> 20), nrts, (unsigned long long)(rts >> 20),
            (unsigned long long)(duse >> 20), (unsigned long long)(dbud >> 20), duse > dbud ? " (OVER)" : ""); }
    { static uint64_t pr, pc, ps, pj, pe, peb, pst; uint64_t r = atomic_load(&s_tm_read), c = atomic_load(&s_tm_conv), s = atomic_load(&s_tm_stage), j = atomic_load(&s_tm_join), e = atomic_load(&s_ev_back), eb = atomic_load(&s_ev_back_b); vkm_stats_t ms; vkm_stats(&ms);
      { static uint64_t pf_, pr_, pa_; uint64_t f_ = atomic_load(&s_up_ns_sf), r_ = atomic_load(&s_up_ns_sr), a_ = atomic_load(&s_up_n_sastc);
        VKLOG("uploads of CPU-storage textures, last 5 s: first %.0f ms, again after a change %.0f ms, %llu of them ASTC", (f_ - pf_) / 1e6, (r_ - pr_) / 1e6, (unsigned long long)(a_ - pa_)); pf_ = f_; pr_ = r_; pa_ = a_; }
      { uint64_t mb[2][4] = {{0}}; uint32_t cnt[2][4] = {{0}};   // [in system memory][0 render target / storage image, 1 CPU-storage compressed, 2 other compressed, 3 other]
        AcquireSRWLockShared(&g_lk);
        for (uint32_t i = 0; i < g_share.cap; i++) { const vkt_image *o = g_share.vals[i]; if (!o || o->sp) continue;
            int c = (o->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) ? 0 : (o->fi->flags & VKF_COMPRESSED) ? (o->cls == VKT_STORAGE ? 1 : 2) : 3;
            mb[o->alloc.sys ? 1 : 0][c] += o->alloc.size; cnt[o->alloc.sys ? 1 : 0][c]++; }
        ReleaseSRWLockShared(&g_lk);
        VKLOG("images held (MB / count), video memory: targets %llu/%u, CPU-storage compressed %llu/%u, other compressed %llu/%u, other %llu/%u; system memory: %llu/%u, %llu/%u, %llu/%u, %llu/%u; all device memory %llu MB, system %llu MB",
              (unsigned long long)(mb[0][0] >> 20), cnt[0][0], (unsigned long long)(mb[0][1] >> 20), cnt[0][1], (unsigned long long)(mb[0][2] >> 20), cnt[0][2], (unsigned long long)(mb[0][3] >> 20), cnt[0][3],
              (unsigned long long)(mb[1][0] >> 20), cnt[1][0], (unsigned long long)(mb[1][1] >> 20), cnt[1][1], (unsigned long long)(mb[1][2] >> 20), cnt[1][2], (unsigned long long)(mb[1][3] >> 20), cnt[1][3],
              (unsigned long long)(vk_device_used() >> 20), (unsigned long long)(vk_sys_used() >> 20)); }
      VKLOG("opaque ASTC as BC1: %llu images so far, %llu made again as BC3, %llu MB saved in uploads", (unsigned long long)atomic_load(&s_bc1_images), (unsigned long long)atomic_load(&s_bc1_redo), (unsigned long long)(atomic_load(&s_bc1_saved) >> 20));
      { uint64_t d, hh, sk, by; vka_prefetch_stats(&d, &hh, &sk, &by); static uint64_t pd, ph, pfst, pfmi; uint64_t fst = atomic_load(&s_pf_fast), fmi = atomic_load(&s_pf_fast_miss);
        VKLOG("texture prefetch, last 5 s: %llu levels ready, %llu taken by uploads; whole textures straight from it %llu (%llu could not); %llu textures done in all, %llu MB held, %llu dropped (queue full)",
            (unsigned long long)(d - pd), (unsigned long long)(hh - ph), (unsigned long long)(fst - pfst), (unsigned long long)(fmi - pfmi), (unsigned long long)atomic_load(&s_pf_jobs), (unsigned long long)(by >> 20), (unsigned long long)atomic_load(&s_pf_dropped));
        pd = d; ph = hh; pfst = fst; pfmi = fmi; }
      VKLOG("uploads, last 5 s: read %.0f ms, convert %.0f ms, staging %.0f ms (%llu waits for the GPU so far), waiting for the ASTC workers %.0f ms; %llu textures back after eviction, %llu MB", (r - pr) / 1e6, (c - pc) / 1e6, (s - ps) / 1e6, (unsigned long long)ms.staging_waits, (j - pj) / 1e6, (unsigned long long)(e - pe), (unsigned long long)((eb - peb) >> 20));
      { static uint64_t ph, pm, pb2, pn2; uint64_t h, m2, b2, n2; vka_pack_stats(&h, &m2, &b2, &n2); VKLOG("ASTC pack, last 5 s: %llu from the pack (%llu MB, %.0f ms reading), %llu transcoded", (unsigned long long)(h - ph), (unsigned long long)((b2 - pb2) >> 20), (n2 - pn2) / 1e6, (unsigned long long)(m2 - pm)); ph = h; pm = m2; pb2 = b2; pn2 = n2; }
      pr = r; pc = c; ps = s; pj = j; pe = e; peb = eb; (void)pst; }
}
static void view_free_cb(void *v) { vkDestroyImageView(g_vk.dev, (VkImageView)v, NULL); }
static void image_free(vkt_image *im) {
    for (int i = 0; i < im->nviews; i++) vkDestroyImageView(g_vk.dev, im->views[i].v, NULL);
    if (im->present_view && im->present_view != im->view) vkDestroyImageView(g_vk.dev, im->present_view, NULL);
    if (im->view) vkDestroyImageView(g_vk.dev, im->view, NULL);
    if (im->img) vkDestroyImage(g_vk.dev, im->img, NULL);
    sparse_free(im); tex_async_free(im);
    vk_free_mem(&im->alloc);
    atomic_fetch_sub(&s_images, 1); atomic_fetch_sub(&s_bytes, im->alloc.size);
    free(im);
}
static void image_free_cb(void *a) { image_free(a); }

// ---------- Opaque ASTC as BC1 (NXRT_VK_BC1_OPAQUE=1: on; off by default, see the bug note at image_create_) ----------
// ASTC becomes BC3 (vk_astc.cpp): 8 bits a texel, 2.25 times the console's 6x6 ASTC. MKW's textures then took ~10 GB, over a 4 GB card's budget, and the driver
// moved memory and the replay evicted and reloaded textures as the game drove on. 80% of them are opaque, and BC3's colour half is a BC1 block: kept as BC1 they
// take half, with the same colours (BC3's colour block always decodes in BC1's four-colour mode; blocks whose endpoints would select BC1's three-colour mode are
// swapped or flattened so they decode the same) and alpha 1, as before. Which textures: those an earlier look (the prefetch, or a BC3 upload) found opaque; a
// level that turns out not to be makes the image BC3 again (vkt_resolve), so alpha is never lost
static SRWLOCK g_op_lk = SRWLOCK_INIT; static vk_hmap g_opaque = {8};   // texture identity (opaque_key) -> 1 opaque, 2 not
static uint64_t opaque_key(const tex_t *t) { return ((uint64_t)(uintptr_t)t->mem ^ ((uint64_t)t->d.format << 48)) * 0x9E3779B97F4A7C15ull ^ ((uint64_t)t->d.width | (uint64_t)t->d.height << 20 | (uint64_t)t->d.levels << 40); }
static void opaque_note(const tex_t *t, int opaque) { uint64_t k = opaque_key(t); AcquireSRWLockExclusive(&g_op_lk); vk_hm_put(&g_opaque, &k, (void *)(uintptr_t)(opaque ? 1 : 2)); ReleaseSRWLockExclusive(&g_op_lk); }
static int opaque_hint(const tex_t *t) { uint64_t k = opaque_key(t); AcquireSRWLockShared(&g_op_lk); int r = (int)(uintptr_t)vk_hm_get(&g_opaque, &k); ReleaseSRWLockShared(&g_op_lk); return r == 1; }
static int bc3_block_opaque(const uint8_t *b) {   // all 16 alpha values 255, for the form encoders write (both endpoints 255: every code but 6 is 255); any other form counts as not
    if (b[0] != 255 || b[1] != 255) return 0;
    uint64_t ix = 0; memcpy(&ix, b + 2, 6);
    uint64_t c6 = ~(ix ^ 0xDB6DB6DB6DB6ull) & 0xFFFFFFFFFFFFull;   // per 3-bit index: its bits equal to 6 (110)
    c6 = c6 & (c6 >> 1) & (c6 >> 2) & 0x249249249249ull;        // a 1 at the low bit of each index that is 6
    return c6 == 0;
}
int vkt_bc3_opaque(const uint8_t *src, size_t size) { for (size_t o = 0; o + 16 <= size; o += 16) if (!bc3_block_opaque(src + o)) return 0; return 1; }
// BC3 blocks to BC1 (dst holds size / 2 bytes); 1 when every alpha was 255
static int bc3_to_bc1(const uint8_t *src, size_t size, uint8_t *dst) {
    int op = 1;
    for (size_t o = 0; o + 16 <= size; o += 16, dst += 8) {
        const uint8_t *b = src + o; if (op && !bc3_block_opaque(b)) op = 0;
        uint16_t c0, c1; uint32_t ix; memcpy(&c0, b + 8, 2); memcpy(&c1, b + 10, 2); memcpy(&ix, b + 12, 4);
        if (c0 < c1) { uint16_t t = c0; c0 = c1; c1 = t; ix ^= 0x55555555u; }   // swapped endpoints: index 0<->1 and 2<->3
        else if (c0 == c1) ix = 0;                                                // one colour: BC1 would read index 3 as black
        memcpy(dst, &c0, 2); memcpy(dst + 2, &c1, 2); memcpy(dst + 4, &ix, 4);
    }
    return op;
}
static VkFormat bc1_of(VkFormat f) { return f == VK_FORMAT_BC3_SRGB_BLOCK ? VK_FORMAT_BC1_RGB_SRGB_BLOCK : f == VK_FORMAT_BC3_UNORM_BLOCK ? VK_FORMAT_BC1_RGB_UNORM_BLOCK : f; }
static const vkf_info *bc1_info(const vkf_info *fi) {   // the format's entry with BC1 in place of BC3 (one per ASTC format)
    static vkf_info tab[64]; static const vkf_info *src[64]; static int n; static SRWLOCK lk = SRWLOCK_INIT;
    AcquireSRWLockExclusive(&lk);
    for (int i = 0; i < n; i++) if (src[i] == fi) { ReleaseSRWLockExclusive(&lk); return &tab[i]; }
    const vkf_info *r = fi;
    if (n < 64) { tab[n] = *fi; tab[n].vk = bc1_of(fi->vk); tab[n].vbpb = 8; src[n] = fi; r = &tab[n++]; }
    ReleaseSRWLockExclusive(&lk); return r;
}
// staging for one level of transcoded BC3 data: as it is, or as BC1 for a BC1 image (noting a level that was not opaque); returns the bytes staged
static size_t stage_bc(vkt_image *im, const uint8_t *data, size_t size, uint8_t *dst) {
    if (!im->bc1) { if (im->opaque && !vkt_bc3_opaque(data, size)) im->opaque = 0; memcpy(dst, data, size); return size; }
    if (!bc3_to_bc1(data, size, dst)) { im->bc1_bad = 1; im->opaque = 0; }
    atomic_fetch_add(&s_bc1_saved, size / 2); return size / 2;
}
static vkt_image *image_create_(const tex_t *t, int cls, int no_bc1);
static vkt_image *image_create(const tex_t *t, int cls) { return image_create_(t, cls, 0); }
static vkt_image *image_create_(const tex_t *t, int cls, int no_bc1) {
    const vkf_info *fi = vkf_get(t->d.format);
    if (!fi) { static int n; if (n++ < 16) VKRED("texture format %#x has no Vulkan counterpart on this device: not created", t->d.format); return NULL; }
    vkt_image *im = calloc(1, sizeof *im);
    im->pool = t->pool; im->off = t->off; im->mem = t->mem; im->d = t->d; im->cls = cls; im->has_layout = t->has_layout; im->storage = t->L.size; im->fi = fi; im->serial = t->serial;
    VkImageType type; int cube;
    shape(&t->d, &type, &im->vtype, &im->w, &im->h, &im->depth, &im->layers, &cube);
    im->levels = t->d.levels ? t->d.levels : 1;
    uint32_t maxl = 1; for (uint32_t s = im->w > im->h ? im->w : im->h; s > 1; s >>= 1) maxl++;
    if (im->levels > maxl) im->levels = maxl;
    // NXRT_VK_TEX_TRIM=N: large ASTC textures lose their top N levels. Off: shaders ask textures their size, and with this on MKW's heap was overwritten with float data at frame 120, every
    // run (compute results written back to guest memory, most likely from a size that had changed). The picture and the memory saving were right, so it stays as a switch for tests
    { static int trim = -1; if (trim < 0) { const char *e = getenv("NXRT_VK_TEX_TRIM"); trim = e ? atoi(e) : 0; }
      if (trim && fi->conv == VKF_ASTC_BC7 && im->levels >= 3 && t->d.samples <= 1 && t->d.target == NVN_TEXTURE_TARGET_2D && !(t->d.flags & NVN_TEXTURE_FLAGS_SPARSE)) {
        uint32_t k = 0; while (k < (uint32_t)trim && k + 2 < im->levels && (im->w >> k) >= 1024 && (im->h >> k) >= 256) k++;
        if (k) { im->trim = k; im->levels -= k; im->w >>= k; im->h >>= k; } } }
    im->samples = t->d.samples > 1 ? (VkSampleCountFlagBits)t->d.samples : VK_SAMPLE_COUNT_1_BIT;
    im->aspect = vkf_aspect(fi->vk);
    VkImageUsageFlags u = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    int depth = (fi->flags & (VKF_DEPTH | VKF_STENCIL)) != 0;
    if (cls == VKT_GPU && !(fi->flags & VKF_COMPRESSED)) {   // NVN has no flag for "render target": every uncompressed GPU texture may be one
        if (depth && (fi->feat & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) u |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (!depth && (fi->feat & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) u |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if ((t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && (fi->feat & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) u |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    VkImageCreateFlags cf = (cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0) | (!depth ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0);
    if (type == VK_IMAGE_TYPE_3D && !(fi->flags & VKF_COMPRESSED)) cf |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    // drop what this format cannot have in this combination (storage on an sRGB format, a sample count)
    for (int pass = 0; pass < 3; pass++) {
        VkImageFormatProperties ip;
        if (vkGetPhysicalDeviceImageFormatProperties(g_vk.pd, fi->vk, type, VK_IMAGE_TILING_OPTIMAL, u, cf, &ip) == VK_SUCCESS && (ip.sampleCounts & im->samples)) break;
        if (pass == 0) u &= ~VK_IMAGE_USAGE_STORAGE_BIT; else if (pass == 1) u &= ~(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }
    im->usage = u;
    im->rest = (u & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT)) ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, cf, type, fi->vk, {im->w, im->h, im->depth}, im->levels, im->layers, im->samples,
                            VK_IMAGE_TILING_OPTIMAL, u, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    vkt_sparse *sp = sparse_want(t, fi, type, u, cf, im->levels, im->layers);
    im->opaque = 1;
    { static int on = -1; if (on < 0) on = vkc_int("bc1_opaque", 0);   // off: with the prefetch on, some streamed textures (MKW's trackside banners) showed noise or black (docs/HANDOFF NEWEST)
      if (on && !no_bc1 && !sp && fi->conv == VKF_ASTC_BC7 && bc1_of(fi->vk) != fi->vk && opaque_hint(t)) {
          fi = bc1_info(fi); im->fi = fi; im->bc1 = 1; ci.format = fi->vk; atomic_fetch_add(&s_bc1_images, 1); } }
    if (sp) { ci.flags |= VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT | (g_vk.has.sparse_aliased ? VK_IMAGE_CREATE_SPARSE_ALIASED_BIT : 0); ci.usage = im->usage = u; sp->gpu = !(fi->flags & VKF_COMPRESSED) && !t->has_layout; im->sp = sp; im->rest = VK_IMAGE_LAYOUT_GENERAL; }
    if (vkCreateImage(g_vk.dev, &ci, NULL, &im->img) != VK_SUCCESS) { VKRED("vkCreateImage %ux%ux%u format %s (%d) levels %u layers %u failed", im->w, im->h, im->depth, fi->name, fi->vk, im->levels, im->layers); free(im); return NULL; }
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vk.dev, im->img, &mr);
    if (sp) { if (sparse_setup(im, sp, &mr)) { static int k; if (k++ < 4) VKRED("sparse texture %ux%u %s could not be set up", im->w, im->h, fi->name); } }
    else if (vk_rt_alloc = (u & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) != 0, vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &im->alloc)) { vk_rt_alloc = 0; VKRED("no device memory for a %ux%u %s texture (%llu bytes)", im->w, im->h, fi->name, (unsigned long long)mr.size); vkDestroyImage(g_vk.dev, im->img, NULL); free(im); return NULL; }
    vk_rt_alloc = 0;
    if (!sp) vkBindImageMemory(g_vk.dev, im->img, im->alloc.mem, im->alloc.off);
    im->view = make_view(im, fi->vk, im->vtype, fi->swz, 0, im->levels, 0, im->layers, im->aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT ? im->aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT : im->aspect);
    VkFormat pf = vkf_unorm_of(fi->vk);
    im->present_view = pf != fi->vk ? make_view(im, pf, VK_IMAGE_VIEW_TYPE_2D, (VkComponentMapping){0}, 0, 1, 0, 1, VK_IMAGE_ASPECT_COLOR_BIT) : VK_NULL_HANDLE;
    if (!im->present_view && im->vtype == VK_IMAGE_VIEW_TYPE_2D && !depth) im->present_view = im->view;
    atomic_fetch_add(&s_images, 1); atomic_fetch_add(&s_bytes, im->alloc.size);
    return im;
}

// ---------- Uploads from storage ----------
static int guarded_deswizzle(const s2tex_surface *s, const uint8_t *bl, size_t bl_size, uint8_t *lin, size_t row, size_t slice) {
    NVN_GUARDED(return -100);
    int r_ = s2tex_deswizzle(s, bl, bl_size, lin, row, slice); NVN_GUARD_END(); return r_;
}
static int guarded_copy(uint8_t *dst, const uint8_t *src, size_t n) {
    NVN_GUARDED(memset(dst, 0, n); return -1);
    memcpy(dst, src, n); NVN_GUARD_END(); return 0;
}
// The "one" of a component type, for the alpha that 3-component formats gain
static void expand_rgb(const uint8_t *src, uint8_t *dst, size_t texels, uint32_t bpb, int nvn_format) {
    uint32_t comp = bpb / 3, flags = vkf_get(nvn_format)->flags;
    VkFormat f = vkf_get(nvn_format)->vk;
    uint8_t one[4] = {0};
    if (comp == 1) one[0] = (flags & VKF_INT) ? 1 : (f == VK_FORMAT_R8G8B8A8_SNORM ? 0x7f : 0xff);
    else if (comp == 2) { uint16_t o = (flags & VKF_INT) ? 1 : f == VK_FORMAT_R16G16B16A16_SFLOAT ? 0x3c00 : f == VK_FORMAT_R16G16B16A16_SNORM ? 0x7fff : 0xffff; memcpy(one, &o, 2); }
    else { uint32_t o = (flags & VKF_INT) ? 1 : 0x3f800000u; memcpy(one, &o, 4); }
    for (size_t i = 0; i < texels; i++) { memcpy(dst + i * comp * 4, src + i * bpb, bpb); memcpy(dst + i * comp * 4 + bpb, one, comp); }
}
static void unpack_1010102s(const uint8_t *src, uint8_t *dst, size_t texels, int snorm) {
    for (size_t i = 0; i < texels; i++) {
        uint32_t v; memcpy(&v, src + i * 4, 4);
        int32_t c[4] = {(int32_t)(v << 22) >> 22, (int32_t)(v << 12) >> 22, (int32_t)(v << 2) >> 22, (int32_t)v >> 30};
        int16_t o[4];
        for (int k = 0; k < 4; k++) {
            if (!snorm) o[k] = (int16_t)c[k];
            else { float m = k < 3 ? 511.0f : 1.0f, x = c[k] / m; if (x < -1) x = -1; o[k] = (int16_t)lrintf(x * 32767.0f); }
        }
        memcpy(dst + i * 8, o, 8);
    }
}

typedef struct { VkBufferImageCopy r[64]; int n; VkBuffer src; } img_batch;
static void ib_flush(img_batch *b, vkt_image *im, VkCommandBuffer cb) { if (b->n) vkCmdCopyBufferToImage(cb, b->src, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, (uint32_t)b->n, b->r); b->n = 0; }
static void ib_add(img_batch *b, vkt_image *im, VkCommandBuffer cb, VkBuffer src, VkBufferImageCopy r) { if (b->n == 64 || (b->n && b->src != src)) ib_flush(b, im, cb); b->src = src; b->r[b->n++] = r; }

static void upgrade_add(vka_key k, vkt_image *im, uint32_t level, uint32_t layer, uint32_t w, uint32_t h, uint32_t slices) {   // caller holds g_lk
    if (g_nupg == g_cupg) { g_cupg = g_cupg ? 2 * g_cupg : 64; g_upg = realloc(g_upg, (size_t)g_cupg * sizeof *g_upg); }
    g_upg[g_nupg++] = (upgrade_t){k, im, level, layer, w, h, slices};
}

// Every level and layer of t's storage into im, recorded into cb (whose submission owns the staging). Returns bytes uploaded, or -1
static int64_t upload(vkt_image *im, const tex_t *t, VkCommandBuffer cb) {
    const vkf_info *fi = im->fi; uint64_t t0 = now_ns(); int64_t total = 0;
    VkImageAspectFlags all = im->aspect;
    vk_image_barrier(cb, im->img, all, im->content ? im->rest : VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    img_batch b; b.n = 0; b.src = VK_NULL_HANDLE;
    int is3d = t->d.target == NVN_TEXTURE_TARGET_3D;
    if (pf_fast_on() && fi->conv == VKF_ASTC_BC7 && !im->trim && !is3d && !im->sp) {   // levels the prefetch made, while the storage is as it read it
        pf_ready *r = ready_take(t); int ok = r != NULL;
        if (ok) ok = r->span && !nvn_dirty_since_span(r->span, t->mem, (size_t)t->L.size, r->stamp) && r->n >= (uint32_t)im->levels * im->layers;
        for (uint32_t i = 0; ok && i < r->n; i++) if (r->lv[i].level < im->levels && r->lv[i].layer < im->layers) ok = vka_has(&r->lv[i].k, vka_bc7_size(r->lv[i].w, r->lv[i].h, 1));
        if (ok) {
            uint8_t *tmp = NULL; size_t tcap = 0;
            for (uint32_t i = 0; ok && i < r->n; i++) {
                uint32_t lw = r->lv[i].w, lh = r->lv[i].h; if (r->lv[i].level >= im->levels || r->lv[i].layer >= im->layers) continue;
                size_t sz = vka_bc7_size(lw, lh, 1);
                if (sz > tcap) { free(tmp); tmp = malloc(sz); tcap = tmp ? sz : 0; }
                VkBuffer sb; VkDeviceSize so; uint8_t *p = tmp ? vkm_staging(im->bc1 ? sz / 2 : sz, 16, &sb, &so) : NULL;
                if (!p || !vka_take(&r->lv[i].k, tmp, sz)) { ok = 0; break; }
                size_t got = stage_bc(im, tmp, sz, p);
                ib_add(&b, im, cb, sb, (VkBufferImageCopy){so, (lw + 3) / 4 * 4, (lh + 3) / 4 * 4, {im->aspect, r->lv[i].level, r->lv[i].layer, 1}, {0, 0, 0}, {lw, lh, 1}});
                total += (int64_t)got;
            }
            free(tmp);
            if (ok) { free(r); atomic_fetch_add(&s_pf_fast, 1); goto staged; }
            total = 0;   // part of it went: the whole texture once more the long way (the copies already recorded are overwritten)
        }
        if (r) atomic_fetch_add(&s_pf_fast_miss, 1);
        free(r);
    }
    for (uint32_t dl = 0; dl < im->levels; dl++)
        for (uint32_t layer = 0; layer < im->layers; layer++) {
            uint32_t level = dl + im->trim;   // the game's level; dl is the image's
            s2tex_surface s; uint64_t base; size_t row = 0;
            int k = nvn_texio_surface(t, level, layer, &s, &base, &row);
            if (k < 0) { if (!level) { static int n; if (n++ < 8) VKRED("texture %ux%u format %s: level %u layer %u is not addressable in its storage", im->w, im->h, fi->name, level, layer); } continue; }
            uint32_t lw = mip(im->w, dl), lh = im->vtype == VK_IMAGE_VIEW_TYPE_1D || im->vtype == VK_IMAGE_VIEW_TYPE_1D_ARRAY ? 1 : mip(im->h, dl), ld = is3d ? mip(im->depth, dl) : 1;
            size_t lrow = (size_t)s.width * s.bytes_per_block, lsize = lrow * s.height * s.depth;
            uint8_t *lin = malloc(lsize ? lsize : 1);
            int rc; uint64_t tm0 = now_ns();
            if (k == 1) {   // linear: rows at the texture's stride
                rc = 0; for (uint32_t y = 0; y < s.height && !rc; y++) rc = guarded_copy(lin + y * lrow, t->mem + base + y * row, lrow);
            } else rc = guarded_deswizzle(&s, t->mem + base, (size_t)(t->L.size - base), lin, lrow, lrow * s.height);
            atomic_fetch_add(&s_tm_read, now_ns() - tm0); tm0 = now_ns();
            if (rc) { static int n; if (n++ < 8) VKRED("texture %ux%u format %s: level %u layer %u storage could not be read (%d)", im->w, im->h, fi->name, level, layer, rc); free(lin); continue; }
            // what goes to the staging buffer, and how Vulkan reads it
            const uint8_t *out = lin; size_t out_size = lsize; uint8_t *conv = NULL; uint32_t row_texels = s.width * fi->bw, img_rows = s.height * fi->bh;
            if (fi->conv == VKF_RGB_TO_RGBA) { conv = malloc((size_t)s.width * s.height * s.depth * fi->vbpb); expand_rgb(lin, conv, (size_t)s.width * s.height * s.depth, fi->bpb, t->d.format); out = conv; out_size = (size_t)s.width * s.height * s.depth * fi->vbpb; }
            else if (fi->conv == VKF_RGB10A2S_RGBA16) { conv = malloc(lsize * 2); unpack_1010102s(lin, conv, lsize / 4, fi->vk == VK_FORMAT_R16G16B16A16_SNORM); out = conv; out_size = lsize * 2; }
            else if (fi->conv == VKF_ASTC_BC7 || fi->conv == VKF_ASTC_RGBA8) {
                int srgb = (fi->flags & VKF_SRGB) != 0;
                static int amin = -1; if (amin < 0) { amin = vkc_int("async_min", 128); if (amin < 4) amin = 4; }   // NXRT_VK_ASYNC_MIN: smallest level edge sent to the workers (smaller: converted here)
                if (fi->conv == VKF_ASTC_BC7 && ld == 1 && (uint64_t)lw * lh >= (uint64_t)amin * amin) {   // on the workers, all levels at once; the upload waits for them
                    tex_job_add(im, lin, lw, lh, fi->bw, fi->bh, srgb, dl, layer); continue;   // (joined below)
                }
                if (fi->conv == VKF_ASTC_BC7) {
                    out_size = vka_bc7_size(lw, lh, ld); conv = malloc(out_size); vka_key key;
                    int q = vka_get_bc7(lin, lw, lh, ld, fi->bw, fi->bh, srgb, conv, &key);
                    if (q < 0) { free(conv); free(lin); continue; }
                    if (q == VKA_FAST) { upgrade_add(key, im, level, layer, lw, lh, ld); im->astc_fast++; atomic_fetch_add(&s_astc_fast, 1); }
                    else atomic_fetch_add(&s_astc_high, 1);
                    row_texels = (lw + 3) / 4 * 4; img_rows = (lh + 3) / 4 * 4;
                } else {
                    out_size = (size_t)lw * lh * ld * 4; conv = malloc(out_size);
                    if (vka_decode_rgba8(lin, lw, lh, ld, fi->bw, fi->bh, srgb, conv)) { free(conv); free(lin); continue; }
                    row_texels = lw; img_rows = lh;
                }
                out = conv;
            }
            else if (fi->conv == VKF_D24_D32F || fi->conv == VKF_D24S8_D32FS8) {   // 24-bit depth in the low bits (unverified on real data), to float; stencil from the top byte
                size_t n = lsize / 4; float *dz = malloc(n * 4); uint8_t *st = malloc(n);
                for (size_t i = 0; i < n; i++) { uint32_t v; memcpy(&v, lin + i * 4, 4); dz[i] = (float)(v & 0xffffff) / 16777215.0f; st[i] = (uint8_t)(v >> 24); }
                VkBuffer sb; VkDeviceSize so; uint8_t *p = vkm_staging(n * 4, 16, &sb, &so);
                if (p) { memcpy(p, dz, n * 4); ib_add(&b, im, cb, sb, (VkBufferImageCopy){so, row_texels, img_rows, {VK_IMAGE_ASPECT_DEPTH_BIT, dl, layer, 1}, {0, 0, 0}, {lw, lh, ld}}); total += (int64_t)n * 4; }
                if (fi->conv == VKF_D24S8_D32FS8 && (p = vkm_staging(n, 16, &sb, &so))) { memcpy(p, st, n); ib_add(&b, im, cb, sb, (VkBufferImageCopy){so, row_texels, img_rows, {VK_IMAGE_ASPECT_STENCIL_BIT, dl, layer, 1}, {0, 0, 0}, {lw, lh, ld}}); total += (int64_t)n; }
                free(dz); free(st); free(lin); continue;
            }
            else if (im->aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {   // exact depth-stencil storage: one aspect at a time
                VkBuffer sb; VkDeviceSize so; size_t n = (size_t)s.width * s.height * s.depth; uint8_t *p;
                if (fi->vk == VK_FORMAT_D24_UNORM_S8_UINT && (p = vkm_staging(n * 5, 16, &sb, &so))) {
                    for (size_t i = 0; i < n; i++) { uint32_t v; memcpy(&v, lin + i * 4, 4); uint32_t d = v & 0xffffff; memcpy(p + i * 4, &d, 4); p[n * 4 + i] = (uint8_t)(v >> 24); }
                    ib_add(&b, im, cb, sb, (VkBufferImageCopy){so, row_texels, img_rows, {VK_IMAGE_ASPECT_DEPTH_BIT, dl, layer, 1}, {0, 0, 0}, {lw, lh, ld}});
                    ib_add(&b, im, cb, sb, (VkBufferImageCopy){so + n * 4, row_texels, img_rows, {VK_IMAGE_ASPECT_STENCIL_BIT, dl, layer, 1}, {0, 0, 0}, {lw, lh, ld}});
                } else if (fi->vk == VK_FORMAT_D32_SFLOAT_S8_UINT && (p = vkm_staging(n * 5, 16, &sb, &so))) {   // NVN: 32-bit float depth, then a 32-bit word with the stencil in its low byte
                    for (size_t i = 0; i < n; i++) { memcpy(p + i * 4, lin + i * 8, 4); p[n * 4 + i] = lin[i * 8 + 4]; }
                    ib_add(&b, im, cb, sb, (VkBufferImageCopy){so, row_texels, img_rows, {VK_IMAGE_ASPECT_DEPTH_BIT, dl, layer, 1}, {0, 0, 0}, {lw, lh, ld}});
                    ib_add(&b, im, cb, sb, (VkBufferImageCopy){so + n * 4, row_texels, img_rows, {VK_IMAGE_ASPECT_STENCIL_BIT, dl, layer, 1}, {0, 0, 0}, {lw, lh, ld}});
                }
                total += (int64_t)n * 5; free(lin); continue;
            }
            VkBuffer sb; VkDeviceSize so;
            atomic_fetch_add(&s_tm_conv, now_ns() - tm0); tm0 = now_ns();
            int bcl = fi->conv == VKF_ASTC_BC7;   // transcoded BC3 (a BC1 image takes half)
            uint8_t *p = vkm_staging(bcl && im->bc1 ? out_size / 2 : out_size, fi->vbpb >= 4 ? fi->vbpb : 4, &sb, &so);
            if (p) {
                if (bcl) out_size = stage_bc(im, out, out_size, p); else memcpy(p, out, out_size);
                VkImageAspectFlags asp = fi->conv == VKF_S8_DS ? VK_IMAGE_ASPECT_STENCIL_BIT : (im->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) && !(im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? VK_IMAGE_ASPECT_STENCIL_BIT : im->aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT ? im->aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT : im->aspect;
                ib_add(&b, im, cb, sb, (VkBufferImageCopy){so, row_texels, img_rows, {asp, dl, is3d ? 0 : layer, 1}, {0, 0, 0}, {lw, lh, ld}});
                total += (int64_t)out_size;
            }
            atomic_fetch_add(&s_tm_stage, now_ns() - tm0);
            free(conv); free(lin);
        }
    // A texture is whole before anything reads it: left to arrive later, a texture a compute pass reads gave results the game then read back, and it crashed on them at frame 120
    if (im->as) {
        tex_async *a = im->as; uint64_t tj = now_ns();
        while (a->pending) if (!sp_help()) SwitchToThread();
        atomic_fetch_add(&s_tm_join, now_ns() - tj);
        AcquireSRWLockExclusive(&a->lk); sp_job *j = a->done; a->done = NULL; ReleaseSRWLockExclusive(&a->lk);
        sp_job *back = NULL;
        for (sp_job *n; j; j = n) {
            n = j->next; int keep = 0;
            if (j->nr) { sp_region *r = &j->r[0]; VkBuffer sb; VkDeviceSize so; uint8_t *p = vkm_staging(im->bc1 ? r->size / 2 : r->size, 16, &sb, &so);
                if (p) { size_t got = stage_bc(im, r->data, r->size, p); ib_add(&b, im, cb, sb, (VkBufferImageCopy){so, r->rt, r->ih, {im->aspect, r->level, j->y, 1}, {0, 0, 0}, {r->cw, r->ch, 1}}); total += (int64_t)got; }
                else keep = 1; }
            if (keep) { j->next = back; back = j; } else sp_job_free(j);
        }
        if (back) { AcquireSRWLockExclusive(&a->lk); sp_job *e = back; while (e->next) e = e->next; e->next = a->done; a->done = back; ReleaseSRWLockExclusive(&a->lk); }
    }
staged:
    ib_flush(&b, im, cb);
    vk_image_barrier(cb, im->img, all, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, im->rest, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
    im->content = 1; im->uploads++;
    atomic_fetch_add(&s_up_bytes, (uint64_t)total); atomic_fetch_add(&s_up_ns, now_ns() - t0);
    return total;
}
static void clear_new(vkt_image *im, VkCommandBuffer cb) {   // a GPU texture's first content: zeros (its storage is not read; nxnative fills it once from a CPU pool, M5-C)
    vk_image_barrier(cb, im->img, im->aspect, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageSubresourceRange rr = {im->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    if (im->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
        VkClearDepthStencilValue z = {1.0f, 0};
        vkCmdClearDepthStencilImage(cb, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &z, 1, &rr);
    } else if (!(im->fi->flags & VKF_COMPRESSED)) { VkClearColorValue z = {{0}}; vkCmdClearColorImage(cb, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &z, 1, &rr); }
    vk_image_barrier(cb, im->img, im->aspect, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, im->rest, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    im->content = 1;
}

// nxnative's tex_init_fill: a GPU texture whose storage already holds what the game put there before its first use takes that as its initial content (the
// uncompressed textures NVN cannot tell from render targets): in a pool the CPU cannot access (filled when the pool was made), an uncached one, or storage the
// CPU flushed. Not depth, multisample or compressible ones. Pokopia fills its 32x32x32 colour-grading LUT this way (an IMAGE-flagged RGBA8 3D texture)
int nvn_flushed(const void *p, uint64_t n);   // hle_nvn.c
static int mem_any(const uint8_t *p, uint64_t n) {
    NVN_GUARDED(return 0);
    uint64_t i = 0, v; int any = 0; for (; i + 8 <= n; i += 8) { memcpy(&v, p + i, 8); if (v) { any = 1; break; } } if (!any) for (; i < n; i++) if (p[i]) { any = 1; break; }
    NVN_GUARD_END(); return any;
}
// A GPU texture's first content (nvn_mtl.m tex_init_fill). Block-compressed ones count: the game reads files into CPU_NO_ACCESS pools, and MKW's sky coverage
// (256x256 BC7) lives in one. A GPU image of a compressed format cannot be cleared, so left out it held whatever the device memory held: a rainbow sky, and from it every reflection
static vk_hmap g_cpu_unc = {8}, g_unc_drop = {8};   // serial -> pool: uncompressed textures taken from their storage; serial -> 1: those that left at their pool's first GPU write
static int init_fill(const tex_t *t, const vkt_image *im) {
    if (!t->mem || !t->pool || (t->pool->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) || t->has_layout || t->d.samples > 1) return 0;
    if (im->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) return 0;
    int no_cpu = (t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) != 0, uncached = (t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED) != 0;
    if (no_cpu && (t->d.flags & NVN_TEXTURE_FLAGS_COMPRESSIBLE)) return 0;   // the game never writes it: the memory may mean nothing to the CPU
    if (!no_cpu && !uncached && !vk_hm_get(&g_unc_drop, &t->serial) && !nvn_flushed(t->mem, t->L.size)) return 0;
    if (!mem_any(t->mem, t->L.size)) return 0;
    int why = no_cpu ? 0 : uncached ? 1 : 2;
    static uint64_t n[3]; if (n[why]++ < 4) VKLOG("initial content: ser=%llu %ux%ux%u format %#x filled from its storage, %s (%llu so far)", (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format,
                                                 (const char *[]){"a CPU_NO_ACCESS pool's", "a CPU_UNCACHED pool's", "flushed by the CPU before its first use"}[why], (unsigned long long)n[why]);
    return 1;
}

// Records into the submission being recorded, or into a one-shot when there is none (submitted and waited for at the end)
static VkCommandBuffer cb_get(int *once) { VkCommandBuffer cb = vks_upload_cb(); *once = 0; if (!cb && vks_owner() < 0) { cb = vk_once_begin(); *once = 1; } return cb; }
static void cb_put(VkCommandBuffer cb, int once) { if (once) vk_once_end(cb); }

// Uploads a storage-backed image when its pages were written since its last upload (nxnative's order: stamp, protect, then read). Caller holds g_lk
static void refresh(vkt_image *im, const tex_t *t) {
    if (!im->span) im->span = nvn_dirty_span(t->mem, (size_t)t->L.size);
    if (im->content && im->span && !nvn_dirty_since_span(im->span, t->mem, (size_t)t->L.size, im->stamp)) return;
    int once; VkCommandBuffer cb = cb_get(&once);
    if (!cb) return;
    uint64_t st = nvn_dirty_now();
    if (im->span) nvn_dirty_protect_span(im->span, t->mem, (size_t)t->L.size);
    int first = !im->content;
    uint64_t t0 = now_ns();
    if (upload(im, t, cb) >= 0) { im->stamp = st; if (first) atomic_fetch_add(&s_uploads, 1); else atomic_fetch_add(&s_reuploads, 1);
        if (im->fi->conv == VKF_ASTC_BC7 && (!im->as || (!im->as->pending && !im->as->done))) opaque_note(t, im->opaque && !im->bc1_bad); }
    atomic_fetch_add(first ? &s_up_ns_sf : &s_up_ns_sr, now_ns() - t0); if (im->fi->conv == VKF_ASTC_BC7) atomic_fetch_add(&s_up_n_sastc, 1);
    cb_put(cb, once);
}

// Over the memory budget, compressed textures not used for a while go: what the game wrote is in guest memory, so a later use uploads them again (Eden's cache worked the same way on this GPU).
// g_lk held. At most twice a second, until usage is back under 80% of the budget; textures used in the last 240 frames stay: at 15, a race threw out and re-encoded 12,700 textures it was still using
static void evict_old(void) {
    static uint64_t last; uint64_t now = now_ns(); if (now - last < 500000000ull) return;
    last = now;
    // Every step of the game's dynamic resolution puts new render targets over the old ones' storage. Those still in use then stayed for good: 130 render targets, 1.3 GB, in one race
    { static int sup = -1; if (sup < 0) { const char *e = getenv("NXRT_VK_SUPERSEDED"); sup = !e || *e != '0'; }
      uint64_t fr = 0; uint32_t cnt = 0;
      if (sup) for (uint32_t i = 0; i < g_share.cap; i++) {
          vkt_image *o = g_share.vals[i];
          if (!o || !o->superseded || o->sp || g_frame - o->last_frame < 16) continue;
          skey ok; memcpy(&ok, g_share.keys + (size_t)i * g_share.klen, sizeof ok);
          vk_hm_del(&g_share, &ok); i--;
          for (uint32_t j = 0; j < g_serial.cap; j++) if (g_serial.vals[j] == o) { uint64_t sk; memcpy(&sk, g_serial.keys + (size_t)j * 8, 8); vk_hm_del(&g_serial, &sk); j--; }
          fr += o->alloc.size; cnt++; vk_retire(image_free_cb, o); atomic_fetch_add(&s_evicted, 1); InterlockedIncrement(&g_vkt_reg_gen);
      }
      static uint64_t tf, tc; tf += fr; tc += cnt; static uint64_t nl; if (cnt && (nl++ % 16) == 0) VKLOG("superseded render targets freed: %llu, %llu MB so far", (unsigned long long)tc, (unsigned long long)(tf >> 20)); }
    static int mode = -1; if (mode < 0) { const char *e = getenv("NXRT_VK_EVICT"); mode = e ? atoi(e) : 1; } if (!mode) return;
    uint64_t budget, usage; if (!vk_budget(&budget, &usage)) return;
    uint64_t want;
    if (mode >= 2) {   // video memory only, against the allocator's own line (vk_alloc_mem: past 90% new images go to system memory). Mode 1's target (video + system under 80%)
        // could not be reached when sparse tiles, pools and targets alone filled the card: every pass evicted every texture idle 8 s, and they streamed back (MKW: ~700 per 5 s)
        usage = vk_device_used(); want = usage > budget / 100 * 90 ? usage - budget / 100 * 85 : 0;
        if (!want && mode == 2) return;
    } else {
        usage = vk_device_used() + vk_sys_used(); if (usage < budget / 100 * 80) return;   // the driver keeps what we call system memory in video memory while it can: it all competes
        want = usage - budget / 100 * 80 + (64ull << 20);
    }
    uint64_t freed = 0, n = 0;
    for (uint32_t age = 3840; age >= 240 && (freed < want || mode == 3); age /= 2)
        for (uint32_t i = 0; i < g_share.cap && (freed < want || mode == 3); i++) {
            vkt_image *o = g_share.vals[i];
            if (!o || o->sp || o->cls != VKT_STORAGE || !(o->fi->flags & VKF_COMPRESSED) || g_frame - o->last_frame < age || !o->alloc.size) continue;   // only what guest memory can give back: MKW's sky frames are GPU copies into a CPU_NO_ACCESS pool, evicted they came back as garbage
            if (mode == 2 && o->alloc.sys) continue;   // freeing system memory makes no room in video memory
            if (mode == 3 && !o->alloc.sys && freed >= want) continue;   // mode 3: video memory only as far as needed; idle textures in system memory always go (slow to sample)
            skey ok; memcpy(&ok, g_share.keys + (size_t)i * g_share.klen, sizeof ok);
            vk_hm_del(&g_share, &ok); i--; InterlockedIncrement(&g_vkt_reg_gen);
            for (uint32_t j = 0; j < g_serial.cap; j++) if (g_serial.vals[j] == o) { uint64_t sk; memcpy(&sk, g_serial.keys + (size_t)j * 8, 8); vk_hm_del(&g_serial, &sk); j--; }
            { uint64_t ek = (uint64_t)(uintptr_t)o->mem; vk_hm_put(&g_evicted, &ek, (void *)1); }
            prefetch_forget(o);
            if (!o->alloc.sys || mode == 1) freed += o->alloc.size; n++; vk_retire(image_free_cb, o); atomic_fetch_add(&s_evicted, 1);
        }
    static uint64_t logged; if (n && logged++ < 20) VKLOG("texture eviction: %llu images, %llu MB, device memory %llu of %llu MB", (unsigned long long)n, (unsigned long long)(freed >> 20), (unsigned long long)(usage >> 20), (unsigned long long)(budget >> 20));
}

// Storage the CPU wrote before the texture's first use (nvn_mtl.m tex_cpu_content): until the GPU first uses a texture it has not written it. A pool the GPU has not written holds only
// what the CPU wrote, so an uncompressed 2D texture there takes its storage too: MKW's terrain patches, 260x260 R16, RG8, RGBA8, R11G11B10F and RGBA16 in a CPU_CACHED pool,
// written with no flush. As GPU textures of zeros the patches came out flat and their lighting black (cacti, roads, tree trunks). g_lk held
extern volatile int nvn_pool_gpuw_new;
static int cpu_content(const tex_t *t) {
    if (!t->mem || !t->pool || t->has_layout || t->d.samples > 1) return 0;
    const vkf_info *fi = vkf_get(t->d.format); if (!fi) return 0;
    int unc = !(fi->flags & VKF_COMPRESSED) && t->d.target != NVN_TEXTURE_TARGET_3D;
    if (unc && (t->pool->gpu_written || (fi->flags & (VKF_DEPTH | VKF_STENCIL)) || (t->d.flags & (NVN_TEXTURE_FLAGS_DISPLAY | NVN_TEXTURE_FLAGS_VIDEO_DECODE)))) return 0;
    if (t->pool->flags & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL)) return 0;
    if ((t->d.flags & (NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_SPARSE)) || ((t->d.flags & NVN_TEXTURE_FLAGS_IMAGE) && !(fi->flags & VKF_COMPRESSED))) return 0;
    if (!mem_any(t->mem, t->L.size)) return 0;
    vk_hm_put(&g_cpu, &t->serial, (void *)t->pool); atomic_fetch_add(&s_cpuw, 1);
    if (unc) vk_hm_put(&g_cpu_unc, &t->serial, (void *)t->pool);
    static uint64_t n[2]; if (n[unc]++ < 4) VKLOG("texture written by the CPU before its first use ser=%llu %ux%ux%u format %#x%s pool flags %#x: uploaded from its storage, again when it changes", (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, unc ? " (uncompressed, in a pool the GPU has not written)" : "", t->pool->flags);
    return 1;
}
// A pool's first GPU write: its storage is no longer the whole truth, so the uncompressed textures taken from it become GPU textures, filled once more from what they hold
static void cpu_unc_drop(void) {
    if (!nvn_pool_gpuw_new) return;
    nvn_pool_gpuw_new = 0;
    for (uint32_t i = 0; i < g_cpu_unc.cap; i++) {
        const pool_t *p = g_cpu_unc.vals[i]; if (!p || !p->gpu_written) continue;
        uint64_t s; memcpy(&s, g_cpu_unc.keys + (size_t)i * 8, 8);
        vk_hm_del(&g_cpu_unc, &s); vk_hm_del(&g_cpu, &s); vk_hm_put(&g_unc_drop, &s, (void *)1); i--; InterlockedIncrement(&g_vkt_reg_gen);   // the deletion shifted entries: look at this slot again
    }
}
vkt_image *vkt_resolve(const tex_t *t) {
    if (!t || !t->serial || !g_vk.up) return NULL;
    if (t->d.target == NVN_TEXTURE_TARGET_BUFFER) return NULL;   // texel buffers: a VkBufferView over the pool mirror (M5-C)
    AcquireSRWLockExclusive(&g_lk);
    evict_old();
    cpu_unc_drop();
    int cls = from_storage(t) || (!vk_hm_get(&g_serial, &t->serial) && cpu_content(t)) ? VKT_STORAGE : VKT_GPU;
    skey k; key_of(t, cls, &k);
    vkt_image *im = vk_hm_get(&g_share, &k);
    if (im && im->bc1 && im->bc1_bad && (!im->as || (!im->as->pending && !im->as->done))) {   // a level applied later was not opaque: as BC3 from here on
        opaque_note(t, 0); vk_hm_del(&g_share, &k);
        for (uint32_t j = 0; j < g_serial.cap; j++) if (g_serial.vals[j] == im) { uint64_t sk; memcpy(&sk, g_serial.keys + (size_t)j * 8, 8); vk_hm_del(&g_serial, &sk); j--; }
        vk_retire(image_free_cb, im); atomic_fetch_add(&s_bc1_redo, 1); InterlockedIncrement(&g_vkt_reg_gen); im = NULL; }
    if (im && im->d.format != t->d.format) { static int n; if (n++ < 12) VKLOG("one image for two formats: ser=%llu format %#x on the image of ser=%llu format %#x, %ux%u", (unsigned long long)t->serial, t->d.format, (unsigned long long)im->serial, im->d.format, t->d.width, t->d.height); }
    if (!im) {
        // a new texture over storage: entries of the same pool overlapping it under another description go, unless used in the last 16 frames (nxnative's alias_take_over)
        if (t->pool) for (uint32_t i = 0; i < g_share.cap; i++) {
            vkt_image *o = g_share.vals[i]; if (!o || o->pool != t->pool) continue;
            int64_t lo = t->off, hi = t->off + (int64_t)t->L.size;
            if (o->off >= hi || o->off + (int64_t)o->storage <= lo) continue;
            if (g_frame - o->last_frame < 16) { o->superseded = 1; continue; }
            skey ok; memcpy(&ok, g_share.keys + (size_t)i * g_share.klen, sizeof ok);
            vk_hm_del(&g_share, &ok); i--; InterlockedIncrement(&g_vkt_reg_gen);   // the deletion shifted entries: look at this slot again
            for (uint32_t j = 0; j < g_serial.cap; j++) if (g_serial.vals[j] == o) { uint64_t sk; memcpy(&sk, g_serial.keys + (size_t)j * 8, 8); vk_hm_del(&g_serial, &sk); j--; }
            vk_retire(image_free_cb, o); atomic_fetch_add(&s_evicted, 1);
        }
        if (!(im = image_create(t, cls))) { ReleaseSRWLockExclusive(&g_lk); return NULL; }
        { uint64_t ek = (uint64_t)(uintptr_t)t->mem; if (vk_hm_get(&g_evicted, &ek)) { atomic_fetch_add(&s_ev_back, 1); atomic_fetch_add(&s_ev_back_b, im->alloc.size); } }
        vk_hm_put(&g_share, &k, im);
        if (im->sp) { int once; VkCommandBuffer cb = cb_get(&once);   // to GENERAL, where it stays; no first upload, the tiles come as the game maps them
            if (cb) { vk_image_barrier(cb, im->img, im->aspect, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT); im->content = 1; cb_put(cb, once); } }
        else if (cls == VKT_GPU) { int once; VkCommandBuffer cb = cb_get(&once); if (cb) { int f = init_fill(t, im), u = f ? upload(im, t, cb) : -9; if (!f || u < 0) clear_new(im, cb);
            int idle = !im->as || (!im->as->pending && !im->as->done);
            if (f && u >= 0 && im->fi->conv == VKF_ASTC_BC7 && idle) opaque_note(t, im->opaque && !im->bc1_bad);
            if (im->bc1_bad && idle) {   // not opaque after all: the same texture as BC3, uploaded again (the BC1 image goes behind the GPU)
                vkt_image *im3 = image_create_(t, cls, 1);
                if (im3) { vk_hm_put(&g_share, &k, im3); vk_retire(image_free_cb, im); atomic_fetch_add(&s_bc1_redo, 1); im = im3;
                           f = init_fill(t, im); u = f ? upload(im, t, cb) : -9; if (!f || u < 0) clear_new(im, cb); } }
            cb_put(cb, once);
            if (getenv("NXRT_VK_LOGNEW") && (im->fi->flags & VKF_COMPRESSED)) { static int nl; uint64_t w0 = 0, w1 = 0; if (t->mem) { memcpy(&w0, t->mem, 8); memcpy(&w1, t->mem + 8, 8); } if (nl++ < 200) VKLOG("new compressed GPU image ser=%llu %ux%ux%u fmt %#x flags %#x pool flags %#x gpuw %d fill %d upload %d any %d first %llx %llx", (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, t->d.flags, t->pool ? t->pool->flags : 0, t->pool ? t->pool->gpu_written : -1, f, u, t->mem ? mem_any(t->mem, t->L.size) : -1, (unsigned long long)w0, (unsigned long long)w1); } } }
    } else if (im->sp) { atomic_fetch_add(&s_shared, 1);
    } else if (cls == VKT_GPU && !im->content) {   // M6 phase B: made where nothing could record its first content (and layout): recorded now, before its first use
        int once; VkCommandBuffer cb = cb_get(&once);
        if (cb) { static int n; if (n++ < 4) VKLOG("texture ser=%llu %ux%ux%u: its initial content recorded at a later use", (unsigned long long)t->serial, im->w, im->h, im->layers);
                  if (!init_fill(t, im) || upload(im, t, cb) < 0) clear_new(im, cb); cb_put(cb, once); }
        atomic_fetch_add(&s_shared, 1);
    } else atomic_fetch_add(&s_shared, 1);
    vk_hm_put(&g_serial, &t->serial, im);
    im->last_frame = g_frame; im->serial = t->serial;
    if (im->sp) { int once; VkCommandBuffer cb = cb_get(&once); if (cb) { sparse_look(im, t, cb); cb_put(cb, once); } }
    else if (im->as && im->as->done) { int once; VkCommandBuffer cb = cb_get(&once); if (cb) { tex_async_apply(im, cb); cb_put(cb, once); } }
    else if (cls == VKT_STORAGE) {
        refresh(im, t);
        if (im->bc1 && im->bc1_bad && (!im->as || (!im->as->pending && !im->as->done))) {   // not opaque after all: the same texture as BC3, uploaded again now
            vkt_image *im3 = image_create_(t, cls, 1);
            if (im3) { vk_hm_put(&g_share, &k, im3); vk_hm_put(&g_serial, &t->serial, im3); im3->last_frame = g_frame; im3->serial = t->serial;
                       vk_retire(image_free_cb, im); atomic_fetch_add(&s_bc1_redo, 1); InterlockedIncrement(&g_vkt_reg_gen); im = im3; refresh(im, t); }
        }
    }
    ReleaseSRWLockExclusive(&g_lk);
    return im;
}

// ---------- Views: the image's fallback applied to every view (research/01 lesson 8) ----------
static VkComponentSwizzle nvn_swz(unsigned s) { static const VkComponentSwizzle m[6] = {VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ONE, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A}; return s < 6 ? m[s] : VK_COMPONENT_SWIZZLE_IDENTITY; }
static VkComponentSwizzle compose(VkComponentSwizzle user, const VkComponentMapping *base) {   // the user's channel read through the format's base mapping
    switch (user) {
    case VK_COMPONENT_SWIZZLE_R: return base->r == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_R : base->r;
    case VK_COMPONENT_SWIZZLE_G: return base->g == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_G : base->g;
    case VK_COMPONENT_SWIZZLE_B: return base->b == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_B : base->b;
    case VK_COMPONENT_SWIZZLE_A: return base->a == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_A : base->a;
    default: return user;
    }
}
VkImageView vkt_view_raw_(vkt_image *im, VkFormat f, VkImageViewType vt, uint32_t bl, uint32_t nl, uint32_t b0, uint32_t nly, VkImageAspectFlags asp, const VkComponentMapping *mp);
VkImageView vkt_view(vkt_image *im, const view_t *v, int set_swz, uint16_t swz) {
    if (!im) return VK_NULL_HANDLE;
    if (!v && !set_swz) return im->view;
    int fmt = v && v->set_format ? v->format : im->d.format;
    uint32_t bl = v && v->set_levels ? v->base_level : 0, nl = v && v->set_levels ? v->nlevels : im->levels;
    uint32_t b0 = v && v->set_layers ? v->min_layer : 0, nly = v && v->set_layers ? v->nlayers : im->layers;
    if (im->trim && v && v->set_levels) { uint32_t cut = bl < im->trim ? im->trim - bl : 0; nl = nl > cut ? nl - cut : 1; bl = bl > im->trim ? bl - im->trim : 0; }
    if (bl >= im->levels) bl = im->levels - 1; if (!nl || bl + nl > im->levels) nl = im->levels - bl;
    if (b0 >= im->layers) b0 = im->layers - 1; if (!nly || b0 + nly > im->layers) nly = im->layers - b0;
    VkFormat f = vkf_view_format(im->d.format, fmt);   // the image's own fallback, whatever the view asks for
    if (im->bc1) { f = bc1_of(f);
        if (f != VK_FORMAT_BC1_RGB_UNORM_BLOCK && f != VK_FORMAT_BC1_RGB_SRGB_BLOCK) { static int n; if (n++ < 20) VKRED("BC1 image ser=%llu %ux%u (format %#x) viewed as VkFormat %d (view format %#x): not a BC1 format", (unsigned long long)im->serial, im->w, im->h, im->d.format, (int)f, fmt); } }
    const vkf_info *vi = vkf_get(fmt); const VkComponentMapping *base = vi && f == vi->vk ? &vi->swz : &im->fi->swz;
    VkComponentMapping m = *base;
    if (set_swz) m = (VkComponentMapping){compose(nvn_swz(swz & 7), base), compose(nvn_swz(swz >> 3 & 7), base), compose(nvn_swz(swz >> 6 & 7), base), compose(nvn_swz(swz >> 9 & 7), base)};
    if (v && v->set_swizzle) m = (VkComponentMapping){compose(nvn_swz(v->swz[0]), &m), compose(nvn_swz(v->swz[1]), &m), compose(nvn_swz(v->swz[2]), &m), compose(nvn_swz(v->swz[3]), &m)};
    VkImageViewType vt = im->vtype;
    if (v && v->set_target) {
        switch (v->target) {
        case NVN_TEXTURE_TARGET_2D: case NVN_TEXTURE_TARGET_RECTANGLE: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE: vt = VK_IMAGE_VIEW_TYPE_2D; nly = 1; break;
        case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY: vt = VK_IMAGE_VIEW_TYPE_2D_ARRAY; break;
        case NVN_TEXTURE_TARGET_CUBEMAP: vt = VK_IMAGE_VIEW_TYPE_CUBE; nly = 6; break;
        case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: vt = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY; break;
        case NVN_TEXTURE_TARGET_1D: vt = VK_IMAGE_VIEW_TYPE_1D; nly = 1; break;
        case NVN_TEXTURE_TARGET_1D_ARRAY: vt = VK_IMAGE_VIEW_TYPE_1D_ARRAY; break;
        case NVN_TEXTURE_TARGET_3D: vt = VK_IMAGE_VIEW_TYPE_3D; break;
        }
    }
    VkImageAspectFlags asp = im->aspect;
    if (asp == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) asp = v && v->set_dsmode && v->dsmode == 1 ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
    if ((vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) && !(im->vtype == VK_IMAGE_VIEW_TYPE_CUBE || im->vtype == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)) {
        static int n; if (n++ < 8) VKRED("a cube view of a %ux%u texture that is not a cube map: viewed as a 2D array", im->w, im->h);
        vt = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    }
    if (vt == VK_IMAGE_VIEW_TYPE_CUBE && nly != 6) nly = b0 + 6 <= im->layers ? 6 : nly;
    return vkt_view_raw_(im, f, vt, bl, nl, b0, nly, asp, &m);
}
// A view of exact parameters (attachments, storage images, views the shader's declaration asks for), cached with the image's views
VkImageView vkt_view_raw_(vkt_image *im, VkFormat f, VkImageViewType vt, uint32_t bl, uint32_t nl, uint32_t b0, uint32_t nly, VkImageAspectFlags asp, const VkComponentMapping *mp) {
    VkComponentMapping m = mp ? *mp : (VkComponentMapping){0};
    if (bl >= im->levels) bl = im->levels - 1; if (!nl || bl + nl > im->levels) nl = im->levels - bl;
    uint32_t all = im->vtype == VK_IMAGE_VIEW_TYPE_3D && vt != VK_IMAGE_VIEW_TYPE_3D ? mip(im->depth, bl) : im->layers;   // a 3D level viewed as a 2D array: its depth slices
    if (b0 >= all) b0 = all - 1; if (!nly || b0 + nly > all) nly = all - b0;
    uint64_t key = (uint64_t)f ^ (uint64_t)vt << 20 ^ (uint64_t)bl << 24 ^ (uint64_t)nl << 28 ^ (uint64_t)b0 << 34 ^ (uint64_t)nly << 46 ^ (uint64_t)asp << 58 ^
                   ((uint64_t)m.r | (uint64_t)m.g << 3 | (uint64_t)m.b << 6 | (uint64_t)m.a << 9) * 0x9E3779B97F4A7C15ull;
    AcquireSRWLockExclusive(&g_lk);
    for (int i = 0; i < im->nviews; i++) if (im->views[i].key == key) { VkImageView r = im->views[i].v; ReleaseSRWLockExclusive(&g_lk); return r; }
    VkImageView nv = make_view(im, f, vt, m, bl, nl, b0, nly, asp);
    if (nv) {   // a full cache drops its oldest view behind the GPU
        if (im->nviews == VKT_NVIEWS) { vk_retire(view_free_cb, (void *)im->views[0].v); memmove(im->views, im->views + 1, (VKT_NVIEWS - 1) * sizeof im->views[0]); im->nviews--; }
        im->views[im->nviews].key = key; im->views[im->nviews].v = nv; im->nviews++;
    }
    ReleaseSRWLockExclusive(&g_lk);
    return nv ? nv : im->view;
}

// ---------- Samplers ----------
static VkSamplerAddressMode wrap(uint8_t w) {
    switch (w) {
    case NVN_WRAP_MODE_REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case NVN_WRAP_MODE_MIRRORED_REPEAT: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case NVN_WRAP_MODE_CLAMP_TO_BORDER: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case NVN_WRAP_MODE_MIRROR_CLAMP: case NVN_WRAP_MODE_MIRROR_CLAMP_TO_EDGE: case NVN_WRAP_MODE_MIRROR_CLAMP_TO_BORDER:   // GL's mirror-clamp modes: the edge variant (border: M5-C)
        return g_vk.has.mirror_clamp_edge ? VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   // CLAMP (GL's half-border clamp) and CLAMP_TO_EDGE
    }
}
VkSampler vkt_sampler(const samp_par *p) {
    AcquireSRWLockExclusive(&g_lk);
    VkSampler s = (VkSampler)vk_hm_get(&g_samp, p);
    if (s) { ReleaseSRWLockExclusive(&g_lk); return s; }
    int mipf = p->minf >= NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST;
    VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, NULL, 0,
        p->magf == NVN_MAG_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        (p->minf == NVN_MIN_FILTER_LINEAR || p->minf == NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST || p->minf == NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        (p->minf == NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR || p->minf == NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        wrap(p->ws), wrap(p->wt), wrap(p->wr), p->lod_bias, p->aniso > 1.0f && g_vk.f10.samplerAnisotropy, p->aniso > 1.0f ? fminf(p->aniso, g_vk.props.limits.maxSamplerAnisotropy) : 1.0f,
        p->cmp_mode == NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE, (VkCompareOp)(p->cmp_func >= 1 && p->cmp_func <= 8 ? p->cmp_func - 1 : 0),
        mipf ? p->lod_min : 0.0f, mipf ? p->lod_max : 0.25f, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, VK_FALSE};
    VkSamplerCustomBorderColorCreateInfoEXT cb = {VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT, NULL, {{p->border[0], p->border[1], p->border[2], p->border[3]}}, VK_FORMAT_UNDEFINED};
    VkSamplerReductionModeCreateInfo rm = {VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO, NULL,
        p->reduction == NVN_SAMPLER_REDUCTION_MIN ? VK_SAMPLER_REDUCTION_MODE_MIN : p->reduction == NVN_SAMPLER_REDUCTION_MAX ? VK_SAMPLER_REDUCTION_MODE_MAX : VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE};
    const void **tail = &ci.pNext;
    int uses_border = ci.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || ci.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER || ci.addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    if (uses_border) {
        const float *b = p->border; int std = b[0] == 0 && b[1] == 0 && b[2] == 0 ? (b[3] == 0 ? 1 : b[3] == 1 ? 2 : 0) : b[0] == 1 && b[1] == 1 && b[2] == 1 && b[3] == 1 ? 3 : 0;
        if (std) ci.borderColor = std == 1 ? VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK : std == 2 ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK : VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        else if (g_vk.has.custom_border && g_vk.has.custom_border_noformat) { ci.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT; *tail = &cb; tail = (const void **)&cb.pNext; }
    }
    if (rm.reductionMode != VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE && g_vk.has.minmax_filter) { *tail = &rm; tail = (const void **)&rm.pNext; }
    if (vkCreateSampler(g_vk.dev, &ci, NULL, &s) == VK_SUCCESS) { vk_hm_put(&g_samp, p, (void *)s); atomic_fetch_add(&s_samplers, 1); }
    else s = VK_NULL_HANDLE;
    ReleaseSRWLockExclusive(&g_lk);
    return s;
}

// ---------- Object lifetimes and registrations ----------
void vkt_tex_init(const tex_t *t) { if (!t->serial) return; AcquireSRWLockExclusive(&g_lk); vk_hm_put(&g_live, &t->serial, (void *)1); ReleaseSRWLockExclusive(&g_lk); }
void vkt_tex_release(const tex_t *t) { InterlockedIncrement(&g_vkt_reg_gen);   // the serial goes; its image stays shared by storage until its pool is finalized or another texture takes the storage
    if (!t->serial) return;
    AcquireSRWLockExclusive(&g_lk); vk_hm_del(&g_live, &t->serial); vk_hm_del(&g_serial, &t->serial); ReleaseSRWLockExclusive(&g_lk);
}
void vkt_tex_cpu_written(const tex_t *t) {
    if (!t->serial) return;
    AcquireSRWLockExclusive(&g_lk);
    if (!vk_hm_get(&g_cpu, &t->serial)) {
        vk_hm_put(&g_cpu, &t->serial, t->pool ? (void *)t->pool : (void *)1); atomic_fetch_add(&s_cpuw, 1);
        static int n; if (n++ < 16) VKLOG("CPU-written texture ser=%llu %ux%ux%u format %#x flags %#x: uploaded from its storage, again when its pages change", (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, t->d.flags);
    }
    ReleaseSRWLockExclusive(&g_lk);
}
int vkt_tex_gpu_owned(uint64_t serial) {
    AcquireSRWLockShared(&g_lk);
    vkt_image *im = vk_hm_get(&g_serial, &serial);
    int r = im && im->cls == VKT_GPU && (im->usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT));
    ReleaseSRWLockShared(&g_lk);
    return r;
}
void vkt_drop_pool(const void *pool) { InterlockedIncrement(&g_vkt_reg_gen);
    if (!pool) return;
    AcquireSRWLockExclusive(&g_lk);
    for (uint32_t i = 0; i < g_share.cap; i++) {
        vkt_image *o = g_share.vals[i]; if (!o || o->pool != pool) continue;
        skey ok; memcpy(&ok, g_share.keys + (size_t)i * g_share.klen, sizeof ok);
        vk_hm_del(&g_share, &ok); i--;
        for (uint32_t j = 0; j < g_serial.cap; j++) if (g_serial.vals[j] == o) { uint64_t sk; memcpy(&sk, g_serial.keys + (size_t)j * 8, 8); vk_hm_del(&g_serial, &sk); j--; }
        for (int u = 0; u < g_nupg; ) if (g_upg[u].im == o) g_upg[u] = g_upg[--g_nupg]; else u++;
        vk_retire(image_free_cb, o); atomic_fetch_add(&s_evicted, 1);
    }
    for (uint32_t i = 0; i < g_cpu.cap; i++) if (g_cpu.vals[i] == pool) { uint64_t sk; memcpy(&sk, g_cpu.keys + (size_t)i * 8, 8); vk_hm_del(&g_cpu, &sk); i--; }
    ReleaseSRWLockExclusive(&g_lk);
}
// Bumped by every change to the texture / sampler registrations or the images behind them: the replay's per-submission memo of
// texture views and samplers (vk_replay.c tex_memo) holds only while it is unchanged
// ---------- Prefetch (NXRT_VK_TEX_PREFETCH=1: on; off by default): an ASTC texture the game registers is read and transcoded on a worker right away, so its first use
// (vkt_resolve -> upload, on the replay thread, under g_lk) finds every level ready in vk_astc.cpp's prefetch memory. A game that streams textures while
// it plays (MKW: ~150 a second in a race) stopped the replay on each new one. Nothing visible changes: the upload looks levels up by their ASTC bytes
typedef struct pf_job { struct pf_job *next; tex_t t; } pf_job;
static pf_job *g_pf_head, *g_pf_tail; static int g_pf_n; static CONDITION_VARIABLE g_pf_cv = CONDITION_VARIABLE_INIT;
static vk_hmap g_pf_seen = {8};   // storage already prefetched: (mem, format, width) -> 1
// NXRT_VK_PF_FAST=1: the upload takes the prefetch's levels by key while the storage is unchanged (off: an 8-run A/B showed no gain, and the
// worker then also skips protecting the storage and keeping the record)
static unsigned __stdcall pf_worker(void *arg) {
    (void)arg; SetThreadDescription(GetCurrentThread(), L"vk tex prefetch"); SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    for (;;) {
        AcquireSRWLockExclusive(&g_pf_lk);
        while (!g_pf_head) SleepConditionVariableSRW(&g_pf_cv, &g_pf_lk, INFINITE, 0);
        pf_job *j = g_pf_head; g_pf_head = j->next; if (!g_pf_head) g_pf_tail = NULL; g_pf_n--;
        ReleaseSRWLockExclusive(&g_pf_lk);
        const tex_t *t = &j->t; const vkf_info *fi = vkf_get(t->d.format);
        VkImageType type; VkImageViewType vt; uint32_t w, h, depth, layers; int cube; shape(&t->d, &type, &vt, &w, &h, &depth, &layers, &cube);
        uint32_t levels = t->d.levels ? t->d.levels : 1; int all = 1, known = 1;
        uint32_t nl = levels * (layers < 64 ? layers : 64), got = 0;
        pf_ready *rd = pf_fast_on() ? calloc(1, sizeof *rd + (size_t)nl * sizeof rd->lv[0]) : NULL;
        const void *span = rd ? nvn_dirty_span(t->mem, (size_t)t->L.size) : NULL;
        if (rd && span) { rd->stamp = nvn_dirty_now(); rd->span = span; nvn_dirty_protect_span(span, t->mem, (size_t)t->L.size); }   // stamp, protect, then read (a write after this is seen)
        for (uint32_t level = 0; fi && level < levels; level++)
            for (uint32_t layer = 0; layer < layers && layer < 64; layer++) {
                s2tex_surface s; uint64_t base; size_t row = 0;
                int k = nvn_texio_surface(t, level, layer, &s, &base, &row); if (k != 0) continue;   // block-linear levels only (ASTC storage is)
                size_t lrow = (size_t)s.width * s.bytes_per_block, lsize = lrow * s.height * s.depth; uint8_t *lin = malloc(lsize ? lsize : 1);
                if (lin && !guarded_deswizzle(&s, t->mem + base, (size_t)(t->L.size - base), lin, lrow, lrow * s.height))
                { vka_key key; int o = vka_prefetch(lin, mip(w, level), mip(h, level), fi->bw, fi->bh, (fi->flags & VKF_SRGB) != 0, &key);
                  if (o < 0) known = 0; else if (!o) all = 0;
                  if (o != -2 && rd && got < nl) { rd->lv[got].level = (uint16_t)level; rd->lv[got].layer = (uint16_t)layer; rd->lv[got].w = mip(w, level); rd->lv[got].h = mip(h, level); rd->lv[got].k = key; got++; } }
                else known = 0;
                free(lin);
            }
        { static int hint = -1; if (hint < 0) hint = vkc_int("pf_hint", 1); if (fi && known && hint) opaque_note(t, all); }   // NXRT_VK_PF_HINT=0: the prefetch does not mark textures opaque (for the BC1 bug hunt)   // the image is made BC1 at its first use when every level is opaque
        if (rd && span && got == nl) {   // every level of every layer: the upload may take them as they are
            rd->n = got; uint64_t hk = pf_key(t->mem, &t->d);
            AcquireSRWLockExclusive(&g_pf_lk); pf_ready *old = vk_hm_get(&g_pf_ready, &hk); vk_hm_put(&g_pf_ready, &hk, rd); ReleaseSRWLockExclusive(&g_pf_lk); free(old);
        } else free(rd);
        free(j); atomic_fetch_add(&s_pf_jobs, 1);
    }
    return 0;
}
static _Atomic uint64_t s_pf_why[6];   // prefetch refusals: no storage / virtual or sparse / layout / target / not ASTC / seen
static uint64_t pf_key(const void *mem, const nvn_texdesc *d) {
    uint64_t key[2] = {(uint64_t)(uintptr_t)mem ^ ((uint64_t)d->format << 48), (uint64_t)d->width | (uint64_t)d->height << 20 | (uint64_t)d->levels << 40};
    return key[0] * 0x9E3779B97F4A7C15ull ^ key[1];
}
static void prefetch_forget(const vkt_image *o) { uint64_t hk = pf_key(o->mem, &o->d); AcquireSRWLockExclusive(&g_pf_lk); vk_hm_del(&g_pf_seen, &hk); ReleaseSRWLockExclusive(&g_pf_lk); }
static void prefetch_add(const tex_t *t) {
    static int on = -1; if (on < 0) on = vkc_int("tex_prefetch", 0);   // off: 8-run A/B 2026-11-09 showed no stutter gain and +1.7 ms translation CPU
    if (!on) return;
    if (!t->mem || !t->pool) { atomic_fetch_add(&s_pf_why[0], 1); return; }
    if ((t->pool->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) || (t->d.flags & NVN_TEXTURE_FLAGS_SPARSE) || t->d.samples > 1) { atomic_fetch_add(&s_pf_why[1], 1); return; }
    if (t->d.target != NVN_TEXTURE_TARGET_2D && t->d.target != NVN_TEXTURE_TARGET_2D_ARRAY && t->d.target != NVN_TEXTURE_TARGET_CUBEMAP && t->d.target != NVN_TEXTURE_TARGET_CUBEMAP_ARRAY) { atomic_fetch_add(&s_pf_why[3], 1); return; }
    const vkf_info *fi = vkf_get(t->d.format); if (!fi || fi->conv != VKF_ASTC_BC7) { atomic_fetch_add(&s_pf_why[4], 1); return; }
    uint64_t hk = pf_key(t->mem, &t->d);
    static int started; AcquireSRWLockExclusive(&g_pf_lk);
    if (vk_hm_get(&g_pf_seen, &hk)) { ReleaseSRWLockExclusive(&g_pf_lk); atomic_fetch_add(&s_pf_why[5], 1); return; }
    if (g_pf_n >= 65536) { ReleaseSRWLockExclusive(&g_pf_lk); atomic_fetch_add(&s_pf_dropped, 1); return; }
    vk_hm_put(&g_pf_seen, &hk, (void *)1);
    pf_job *j = malloc(sizeof *j); if (!j) { ReleaseSRWLockExclusive(&g_pf_lk); return; }
    j->t = *t; j->next = NULL; if (g_pf_tail) g_pf_tail->next = j; else g_pf_head = j; g_pf_tail = j; g_pf_n++;
    if (!started) { started = 1; for (int k = 0; k < 2; k++) CloseHandle((HANDLE)_beginthreadex(NULL, 256 << 10, pf_worker, NULL, 0, NULL)); }
    ReleaseSRWLockExclusive(&g_pf_lk); WakeConditionVariable(&g_pf_cv);
}
void vkt_reg_tex(int id, const tex_t *t, const view_t *v, int image) {
    if (id < 0 || id > (1 << 20)) return;
    InterlockedIncrement(&g_vkt_reg_gen);
    AcquireSRWLockExclusive(&g_lk);
    if (id >= g_nreg) { int n = id + 1024; g_reg = realloc(g_reg, (size_t)n * sizeof *g_reg); memset(g_reg + g_nreg, 0, (size_t)(n - g_nreg) * sizeof *g_reg); g_nreg = n; }
    vkt_reg *r = &g_reg[id]; r->t = *t; r->has_view = v != NULL; if (v) r->v = *v; r->image = image; r->live = 1; r->gone = 0;
    int known = t->serial && vk_hm_get(&g_serial, &t->serial) != NULL;
    ReleaseSRWLockExclusive(&g_lk);
    static _Atomic uint64_t s_reg_known, s_reg_img; if (known) atomic_fetch_add(&s_reg_known, 1); if (image) atomic_fetch_add(&s_reg_img, 1);
    { static uint64_t n; if (!(++n % 20000)) VKLOG("texture registrations so far: %llu (%llu of a texture already in use, %llu as images); prefetch refused: no storage %llu, virtual/sparse %llu, packaged layout %llu, target %llu, not ASTC %llu, seen %llu",
          (unsigned long long)n, (unsigned long long)s_reg_known, (unsigned long long)s_reg_img, (unsigned long long)s_pf_why[0], (unsigned long long)s_pf_why[1], (unsigned long long)s_pf_why[2], (unsigned long long)s_pf_why[3], (unsigned long long)s_pf_why[4], (unsigned long long)s_pf_why[5]); }
    if (!known && !image) prefetch_add(t);
}
void vkt_reg_tex_gone(int id) { InterlockedIncrement(&g_vkt_reg_gen); AcquireSRWLockExclusive(&g_lk); if (id >= 0 && id < g_nreg) g_reg[id].gone = 1; ReleaseSRWLockExclusive(&g_lk); }
const vkt_reg *vkt_reg_get(int id) { return id >= 0 && id < g_nreg && g_reg[id].live ? &g_reg[id] : NULL; }
void vkt_reg_samp(int id, const samp_par *p) {
    if (id < 0 || id > (1 << 20)) return;
    InterlockedIncrement(&g_vkt_reg_gen);
    AcquireSRWLockExclusive(&g_lk);
    if (id >= g_nsreg) { int n = id + 256; g_sreg = realloc(g_sreg, (size_t)n * sizeof *g_sreg); g_sreg_ok = realloc(g_sreg_ok, (size_t)n); memset(g_sreg_ok + g_nsreg, 0, (size_t)(n - g_nsreg)); g_nsreg = n; }
    g_sreg[id] = *p; g_sreg_ok[id] = 1;
    ReleaseSRWLockExclusive(&g_lk);
}
const samp_par *vkt_samp_get(int id) { return id >= 0 && id < g_nsreg && g_sreg_ok[id] ? &g_sreg[id] : NULL; }
void vkt_set_pool_n(int n) { g_pool_n = n; }

void vkt_read_texels(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) {
    // The storage is the ground truth for everything the CPU or a copy wrote; content the GPU rendered or wrote is read back from the image (the caller
    // has waited for the GPU as NVN asks; the replay's submissions are flushed and waited for here as well, as nxnative's mtl_read_texels does)
    AcquireSRWLockShared(&g_lk);
    vkt_image *im = t->serial ? vk_hm_get(&g_serial, &t->serial) : NULL;
    int gpu = im && im->cls == VKT_GPU && im->content && !(im->fi->flags & VKF_COMPRESSED) && im->fi->conv == VKF_COPY;
    ReleaseSRWLockShared(&g_lk);
    if (gpu && g_vk.up) {
        int nz = r->depth ? r->depth : 1, is3d = t->d.target == NVN_TEXTURE_TARGET_3D;
        size_t wb = (size_t)r->width * im->fi->vbpb, hb = (size_t)r->height;
        uint8_t *tmp = malloc(wb * hb * nz ? wb * hb * nz : 1);
        vks_finish();
        int ok = 1;
        for (int z = 0; z < nz && ok; z++) {
            NVNcopyRegion rr = *r; rr.depth = 1; if (is3d) rr.zoffset = r->zoffset + z;
            ok = !vkt_readback_region(im, level, is3d ? 0 : layer0 + r->zoffset + z, &rr, tmp + (size_t)z * wb * hb, wb * hb);
        }
        if (ok) { for (int z = 0; z < nz; z++) for (size_t y = 0; y < hb; y++) memcpy(dst + z * img + y * row, tmp + (size_t)z * wb * hb + y * wb, wb); free(tmp); return; }
        free(tmp);
        static int n; if (n++ < 4) VKRED("ReadTexels of GPU texture ser=%llu: the readback failed, the storage is read instead", (unsigned long long)t->serial);
    }
    if (!nvn_texio_read(t, level, layer0, r, dst, row, img)) return;
    static int n; if (n++ < 4) VKRED("ReadTexels %dx%d of level %u, format %#x: the storage cannot be addressed", r->width, r->height, level, t->d.format);
}

vkt_image *vkt_by_serial(uint64_t serial) { AcquireSRWLockShared(&g_lk); vkt_image *im = vk_hm_get(&g_serial, &serial); ReleaseSRWLockShared(&g_lk); return im; }
int vkt_reg_copy(int id, vkt_reg *out) {
    AcquireSRWLockShared(&g_lk);
    int ok = id > 0 && id < g_nreg && g_reg[id].live;
    if (ok) *out = g_reg[id];
    ReleaseSRWLockShared(&g_lk);
    return ok;
}
int vkt_samp_copy(int id, samp_par *out) {
    AcquireSRWLockShared(&g_lk);
    int ok = id >= 0 && id < g_nsreg && g_sreg_ok[id];
    if (ok) *out = g_sreg[id];
    ReleaseSRWLockShared(&g_lk);
    return ok;
}

int vkt_readback_region(vkt_image *im, unsigned level, unsigned layer, const NVNcopyRegion *r, uint8_t *dst, size_t size) {
    if (!im || level >= im->levels || r->width <= 0 || r->height <= 0) return -1;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkBuffer b; vk_alloc m;
    if (vkCreateBuffer(g_vk.dev, &bci, NULL, &b) != VK_SUCCESS) return -1;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, b, &mr);
    if (vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0, &m) &&
        vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, &m)) { vkDestroyBuffer(g_vk.dev, b, NULL); return -1; }
    vkBindBufferMemory(g_vk.dev, b, m.mem, m.off);
    VkCommandBuffer cb = vk_once_begin();
    VkImageAspectFlags asp = im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : im->aspect;
    vk_image_barrier(cb, im->img, im->aspect, im->rest, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    int is3d = im->vtype == VK_IMAGE_VIEW_TYPE_3D;
    VkBufferImageCopy c = {0, 0, 0, {asp, level, is3d ? 0 : layer, 1}, {r->xoffset, r->yoffset, is3d ? r->zoffset : 0}, {(uint32_t)r->width, (uint32_t)r->height, 1}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b, 1, &c);
    vk_image_barrier(cb, im->img, im->aspect, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, im->rest, VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
    vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    int rc = vk_once_end(cb);
    if (!rc) memcpy(dst, m.map, size);
    vkDestroyBuffer(g_vk.dev, b, NULL); vk_free_mem(&m);
    return rc;
}

int vkt_readback(vkt_image *im, unsigned level, unsigned layer, uint8_t *dst, size_t size) {
    if (!im || level >= im->levels || layer >= im->layers) return -1;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkBuffer b; vk_alloc m;
    if (vkCreateBuffer(g_vk.dev, &bci, NULL, &b) != VK_SUCCESS) return -1;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, b, &mr);
    if (vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0, &m) &&
        vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, &m)) { vkDestroyBuffer(g_vk.dev, b, NULL); return -1; }
    vkBindBufferMemory(g_vk.dev, b, m.mem, m.off);
    VkCommandBuffer cb = vk_once_begin();
    VkImageAspectFlags asp = im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : im->aspect;
    vk_image_barrier(cb, im->img, im->aspect, im->rest, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    uint32_t lw = mip(im->w, level), lh = mip(im->h, level), ld = im->vtype == VK_IMAGE_VIEW_TYPE_3D ? mip(im->depth, level) : 1;
    VkBufferImageCopy c = {0, 0, 0, {asp, level, layer, 1}, {0, 0, 0}, {lw, lh, ld}};
    vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b, 1, &c);
    vk_image_barrier(cb, im->img, im->aspect, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, im->rest, VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT);
    vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    int rc = vk_once_end(cb);
    if (!rc) memcpy(dst, m.map, size);
    vkDestroyBuffer(g_vk.dev, b, NULL); vk_free_mem(&m);
    return rc;
}

// ---------- ASTC high-quality passes, presentation, frames ----------
static void on_ready(vka_key k, const uint8_t *bc7, size_t size, void *user) {
    (void)user;
    if (g_nready == g_cready) { g_cready = g_cready ? 2 * g_cready : 16; g_ready_key = realloc(g_ready_key, (size_t)g_cready * sizeof *g_ready_key); g_ready_data = realloc(g_ready_data, (size_t)g_cready * sizeof *g_ready_data); g_ready_size = realloc(g_ready_size, (size_t)g_cready * sizeof *g_ready_size); }
    g_ready_key[g_nready] = k; g_ready_data[g_nready] = malloc(size); memcpy(g_ready_data[g_nready], bc7, size); g_ready_size[g_nready] = size; g_nready++;
}
void vkt_in_submission(void) {
    vka_poll(on_ready, NULL, 16);
    if (!g_nready) return;
    VkCommandBuffer cb = vks_upload_cb();
    if (!cb) return;
    AcquireSRWLockExclusive(&g_lk);
    for (int r = 0; r < g_nready && vkm_budget_left(); ) {
        int used = 0;
        for (int u = 0; u < g_nupg; ) {
            upgrade_t *g = &g_upg[u];
            if (g->key.lo != g_ready_key[r].lo || g->key.hi != g_ready_key[r].hi || g_ready_size[r] != vka_bc7_size(g->w, g->h, g->slices)) { u++; continue; }
            vkt_image *im = g->im;   // a BC1 image takes the level as BC1 (BC3 bytes copied into it were noise); a level that is not opaque marks it for BC3 again
            VkBuffer sb; VkDeviceSize so; uint8_t *p = vkm_staging(im->bc1 ? g_ready_size[r] / 2 : g_ready_size[r], 16, &sb, &so);
            if (!p) break;
            size_t staged = stage_bc(im, g_ready_data[r], g_ready_size[r], p);
            VkImageSubresourceRange rr = {VK_IMAGE_ASPECT_COLOR_BIT, g->level, 1, im->vtype == VK_IMAGE_VIEW_TYPE_3D ? 0 : g->layer, 1};
            VkImageMemoryBarrier2 b0 = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, NULL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                        im->rest, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, im->img, rr};
            VkDependencyInfo di = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, NULL, 0, 0, NULL, 0, NULL, 1, &b0}; vkCmdPipelineBarrier2(cb, &di);
            VkBufferImageCopy c = {so, (g->w + 3) / 4 * 4, (g->h + 3) / 4 * 4, {VK_IMAGE_ASPECT_COLOR_BIT, g->level, rr.baseArrayLayer, 1}, {0, 0, 0}, {g->w, g->h, g->slices}};
            vkCmdCopyBufferToImage(cb, sb, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
            b0.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT; b0.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT; b0.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT; b0.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
            b0.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b0.newLayout = im->rest; vkCmdPipelineBarrier2(cb, &di);
            vkm_budget_take(staged); im->astc_fast--; atomic_fetch_add(&s_astc_high, 1);
            g_upg[u] = g_upg[--g_nupg]; used = 1;
        }
        (void)used;
        free(g_ready_data[r]); g_ready_key[r] = g_ready_key[g_nready - 1]; g_ready_data[r] = g_ready_data[g_nready - 1]; g_ready_size[r] = g_ready_size[g_nready - 1]; g_nready--;
    }
    ReleaseSRWLockExclusive(&g_lk);
}

int vkt_present_source(uint64_t serial, vkp_source *out) {
    AcquireSRWLockShared(&g_lk);
    vkt_image *im = vk_hm_get(&g_serial, &serial);
    int ok = im && im->content && im->present_view;
    if (ok) { out->view = im->present_view; out->layout = im->rest; out->w = im->w; out->h = im->h; }
    ReleaseSRWLockShared(&g_lk);
    return ok;
}

void vkt_frame(uint64_t frame) {
    g_frame = frame;
    if (frame % 300) return;
    vkt_stats_t s; vkt_stats(&s); vka_stats_t a; vka_stats(&a);
    VKLOG("textures up to frame %llu: %llu images %.1f MB, uploads %llu + %llu again (%.1f MB, %.1f ms), shared %llu, CPU-written %llu, evicted %llu, samplers %llu; "
          "ASTC: %llu fast passes (%.1f Mtexel/s), %llu high-quality done, cache hits %llu/%llu",
          (unsigned long long)frame, (unsigned long long)s.images, s.image_bytes / 1048576.0, (unsigned long long)s.uploads, (unsigned long long)s.reuploads,
          s.upload_bytes / 1048576.0, s.upload_ns / 1e6, (unsigned long long)s.shared_hits, (unsigned long long)s.cpu_written, (unsigned long long)s.evicted,
          (unsigned long long)s.samplers, (unsigned long long)a.misses, a.fast_ns ? a.fast_texels / (a.fast_ns / 1e3) : 0.0, (unsigned long long)a.high_done,
          (unsigned long long)a.hits_high, (unsigned long long)a.hits_fast);
}

void vkt_stats(vkt_stats_t *s) {
    s->images = atomic_load(&s_images); s->image_bytes = atomic_load(&s_bytes); s->uploads = atomic_load(&s_uploads); s->reuploads = atomic_load(&s_reuploads);
    s->upload_bytes = atomic_load(&s_up_bytes); s->upload_ns = atomic_load(&s_up_ns); s->shared_hits = atomic_load(&s_shared); s->astc_fast = atomic_load(&s_astc_fast);
    s->astc_high = atomic_load(&s_astc_high); s->cpu_written = atomic_load(&s_cpuw); s->evicted = atomic_load(&s_evicted); s->samplers = atomic_load(&s_samplers);
}

int vkt_init(void) { vkp_set_resolver(vkt_present_source); return 0; }
void vkt_shutdown(void) {
    AcquireSRWLockExclusive(&g_lk);
    for (uint32_t i = 0; i < g_share.cap; i++) if (g_share.vals[i]) image_free(g_share.vals[i]);
    vk_hm_free(&g_share); vk_hm_free(&g_serial); vk_hm_free(&g_live); vk_hm_free(&g_cpu);
    for (uint32_t i = 0; i < g_samp.cap; i++) if (g_samp.vals[i]) vkDestroySampler(g_vk.dev, (VkSampler)g_samp.vals[i], NULL);
    vk_hm_free(&g_samp);
    free(g_reg); g_reg = NULL; g_nreg = 0; free(g_sreg); free(g_sreg_ok); g_sreg = NULL; g_sreg_ok = NULL; g_nsreg = 0;
    free(g_upg); g_upg = NULL; g_nupg = g_cupg = 0;
    ReleaseSRWLockExclusive(&g_lk);
}

// gamectl texat ADDR: the images whose storage covers a guest address (debugging: what the GPU keeps of a memory range)
int vkt_list_at(uint64_t addr, char *out, int cap) {
    int n = 0; AcquireSRWLockShared(&g_lk);
    for (uint32_t i = 0; i < g_share.cap && n < cap - 300; i++) {
        vkt_image *im = g_share.vals[i]; if (!im || !im->mem) continue;
        uint64_t lo = (uint64_t)(uintptr_t)im->mem, hi = lo + (im->storage ? im->storage : 1);
        if (addr < lo || addr >= hi) continue;
        n += snprintf(out + n, (size_t)(cap - n), "ser=%llu %ux%ux%u layers %u levels %u fmt %#x (%s) flags %#x stride %u, storage %#llx+%#llx, %s, usage%s%s%s, content %d, uploads %llu, last frame %llu\n",
            (unsigned long long)im->serial, im->d.width, im->d.height, im->d.depth, im->layers, im->levels, im->d.format, im->fi ? im->fi->name : "?", im->d.flags, im->d.stride,
            (unsigned long long)lo, (unsigned long long)(hi - lo), im->cls == VKT_STORAGE ? "storage-backed" : "GPU", (im->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ? " render-target" : "",
            (im->usage & VK_IMAGE_USAGE_STORAGE_BIT) ? " storage-image" : "", (im->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? " depth" : "", im->content,
            (unsigned long long)im->uploads, (unsigned long long)im->last_frame);
    }
    ReleaseSRWLockShared(&g_lk);
    if (!n) n = snprintf(out, (size_t)cap, "no image has storage there\n");
    return n;
}
