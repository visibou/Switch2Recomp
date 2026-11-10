// Large sparse textures as Vulkan sparse images (included by vk_tex.c). Logic after nxnative's nvn_mtl.m sparse_make / sp_tile / sparse_sync (MIT, (c) seedds), which does this on Metal:
// MKW keeps 8192x8192x68 terrain arrays (BC6H 5.8 GB, ASTC 8x8 1.4 GB, more than this GPU has) in sparse virtual pools and maps a small part of them, in NVN's 64 KB tiles.
// Such a texture gets no memory until the game maps a tile: all 16 pages mapped -> the tile gets memory from 256 MB heaps and its content is uploaded (decoded from the block-linear tile, ASTC
// transcoded to BC7, so one NVN tile of ASTC 8x8 is four of Vulkan's 256x256). Unmapped -> its memory goes back. Vulkan's mip tail is bound whole when the image is made. A tile keeps its place in
// a queue; worker threads read and decode tiles, and a look binds and copies what they finished, within a budget per frame, so a burst of mappings spreads over frames.
#define SP_TILE 0x10000u
#define SP_HEAP (256ull << 20)
#define SP_MIN (64ull << 20)
#include <process.h>
int nvn_sparse_pool(const void *pool); uint64_t nvn_sparse_gen(void); uint64_t nvn_sparse_copy(const void *pool, void *dst, uint64_t a, uint64_t n);
uint64_t nvn_sparse_log_seq(const void *pool); int nvn_sparse_log_read(const void *pool, uint64_t *seq, uint64_t (*out)[2], int max);
int nvn_sparse_tile_state(const void *pool, uint64_t a, uint64_t n, uint64_t gen);

struct vkt_sparse {
    uint32_t tw, th, ntiled, levels, layers, per_layer, tail_at;     // NVN's tiles: those of the tiled levels from loff_t[l], then the layer's tail (index tail_at)
    uint32_t tiles_x[16], tiles_y[16], loff_t[16]; uint64_t loff[16], tail_off, layer_size;
    uint32_t vw, vh, nv;                                             // Vulkan's sparse block and how many make an NVN tile (1 or 4)
    uint32_t vtail;                                                  // Vulkan's first tail level
    uint32_t skip;                                                   // tiled levels under this one are left out: no memory, no upload, and the views start above them
    uint32_t *slot;                                                  // per tile, nv entries: memory slot + 1, 0 = not bound
    uint8_t *state;                                                  // per tile: 1 uploaded, 2 waiting in the queue
    uint64_t seq, gen, looked; int seen, gpu, astc;
    uint64_t *q; uint64_t qh, qt, qcap;                              // tiles waiting: layer << 32 | tile
    vk_alloc tail_mem;
    tex_t t;                                                         // what the workers read (the game may rewrite its object)
    const vkf_info *fi; uint32_t im_w, im_h, im_levels;
    volatile LONG pending; volatile LONG dead;                       // jobs out; set when the image goes (the last job frees this)
    struct sp_job *done; SRWLOCK done_lk;                            // finished jobs
    uint32_t *extra, nextra, capextra;                               // slots of levels bound for good
};
static struct { VkDeviceMemory mem[64]; VkDeviceSize off[64]; int nheap; uint32_t *free_; uint32_t nfree, capfree, made; uint64_t budget, live, peak; } g_sp;
static SRWLOCK g_sp_lk = SRWLOCK_INIT;

static int sp_slot(const VkMemoryRequirements *mr, uint32_t *out) {
    static uint64_t failed_at;
    if (!g_sp.nfree) {
        if (failed_at && now_ns() - failed_at < 2000000000ull) return -1;   // no retry for a while: 90,000 failed allocations in 5 s cost more than the failure
        if (!g_sp.budget) { const char *e = getenv("NXRT_VK_SPARSE_MB"); g_sp.budget = (e ? strtoull(e, NULL, 0) : 2048) << 20; }
        if (g_sp.nheap == 64 || (uint64_t)(g_sp.nheap + 1) * SP_HEAP > g_sp.budget) return -1;
        VkMemoryRequirements r = *mr; r.size = SP_HEAP; r.alignment = r.alignment > SP_TILE ? r.alignment : SP_TILE;
        vk_alloc a; if (vk_alloc_mem(&r, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &a)) { VKRED("sparse textures: no device memory for heap %d", g_sp.nheap); failed_at = now_ns(); return -1; }
        g_sp.mem[g_sp.nheap] = a.mem; g_sp.off[g_sp.nheap] = a.off;
        uint32_t base = (uint32_t)g_sp.nheap * (uint32_t)(SP_HEAP / SP_TILE), n = (uint32_t)(SP_HEAP / SP_TILE);
        if (g_sp.capfree < g_sp.nfree + n) { g_sp.capfree = g_sp.nfree + n; g_sp.free_ = realloc(g_sp.free_, g_sp.capfree * sizeof *g_sp.free_); }
        for (uint32_t i = n; i-- > 0;) g_sp.free_[g_sp.nfree++] = base + i;
        g_sp.nheap++;
        VKLOG("sparse textures: heap %d (%llu MB of tiles so far)", g_sp.nheap, (unsigned long long)((uint64_t)g_sp.nheap * SP_HEAP >> 20));
    }
    *out = g_sp.free_[--g_sp.nfree]; g_sp.live++; if (g_sp.live > g_sp.peak) g_sp.peak = g_sp.live;
    return 0;
}
static void sp_slot_free(uint32_t s) { g_sp.free_[g_sp.nfree++] = s; g_sp.live--; }
// Shared tiles (nxnative's sp_tile): the game maps one physical page at about three places, of textures of one format, so a tile whose 16 pages are consecutive physical memory is keyed by that
// memory and every place binds the same device memory (VK_IMAGE_CREATE_SPARSE_ALIASED_BIT). In a race 2,959 MB of mapped pages sit on 976 MB of memory
const uint8_t *nvn_sparse_tile_phys(const void *pool, uint64_t a);
typedef struct { const uint8_t *p; uint32_t fmt, s[4], nv, rc; } sp_share;
enum { SP_SHN = 1 << 18 };
static sp_share *g_sh; static uint32_t *g_sp_ent;   // g_sp_ent[slot]: entry + 1
static uint32_t sp_sh_find(const uint8_t *p, uint32_t fmt, int make) {   // → index, ~0u when absent
    if (!g_sh) { g_sh = calloc(SP_SHN, sizeof *g_sh); g_sp_ent = calloc((size_t)64 * (SP_HEAP / SP_TILE), sizeof *g_sp_ent); }
    uint64_t h = ((uintptr_t)p >> 16) * 0x9E3779B97F4A7C15ull ^ fmt;
    for (uint32_t k = (uint32_t)(h >> 40) & (SP_SHN - 1), n = 0; n < SP_SHN; k = (k + 1) & (SP_SHN - 1), n++) {
        sp_share *e = &g_sh[k];
        if (e->p == p && e->fmt == fmt) return k;
        if (!e->p && !e->rc) { if (make) { e->p = p; e->fmt = fmt; return k; } return ~0u; }
    }
    return ~0u;
}
static void sp_slot_mem(uint32_t s, VkDeviceMemory *m, VkDeviceSize *o) { uint32_t per = (uint32_t)(SP_HEAP / SP_TILE); *m = g_sp.mem[s / per]; *o = g_sp.off[s / per] + (VkDeviceSize)(s % per) * SP_TILE; }

// 1 when this texture should be sparse: a 2D texture or array in a sparse pool, made of NVN tiles Vulkan can bind, at least SP_MIN made whole. Fills what image_create needs
static vkt_sparse *sparse_want(const tex_t *t, const vkf_info *fi, VkImageType type, VkImageUsageFlags u, VkImageCreateFlags cf, uint32_t levels, uint32_t layers) {
    if (!t->pool || !nvn_sparse_pool(t->pool) || !t->mem || (t->d.target != NVN_TEXTURE_TARGET_2D && t->d.target != NVN_TEXTURE_TARGET_2D_ARRAY) || t->d.samples > 1 || !g_vk.has.sparse_image) return NULL;
    if (fi->conv != VKF_COPY && fi->conv != VKF_ASTC_BC7) return NULL;
    uint32_t tw, th, td; int n = nvn_tex_sparse_tiles(&t->d, &tw, &th, &td);
    if (n <= 0 || n > 15 || td != 1 || levels > 16) return NULL;
    if ((uint64_t)t->L.layer_size * layers < SP_MIN) return NULL;
    uint32_t np = 8; VkSparseImageFormatProperties pr[8];
    vkGetPhysicalDeviceSparseImageFormatProperties(g_vk.pd, fi->vk, type, VK_SAMPLE_COUNT_1_BIT, u, VK_IMAGE_TILING_OPTIMAL, &np, pr);
    if (!np) { static int k; if (k++ < 4) VKLOG("sparse texture %ux%u %s: the device has no sparse form of this format, made whole", t->d.width, t->d.height, fi->name); return NULL; }
    uint32_t vw = pr[0].imageGranularity.width, vh = pr[0].imageGranularity.height;
    if (tw % vw || th % vh || tw / vw != th / vh || (tw / vw != 1 && tw / vw != 2)) { static int k; if (k++ < 4) VKLOG("sparse texture %ux%u %s: tile %ux%u, Vulkan's %ux%u: made whole", t->d.width, t->d.height, fi->name, tw, th, vw, vh); return NULL; }
    vkt_sparse *s = calloc(1, sizeof *s);
    s->tw = tw; s->th = th; s->vw = vw; s->vh = vh; s->nv = (tw / vw) * (th / vh); s->ntiled = (uint32_t)n; s->levels = levels; s->layers = layers; s->layer_size = t->L.layer_size;
    s->astc = fi->conv == VKF_ASTC_BC7;
    uint32_t k = 0;
    for (uint32_t l = 0; l < s->ntiled; l++) {
        uint32_t w = mip(t->d.width, l), h = mip(t->d.height, l);
        s->tiles_x[l] = (w + tw - 1) / tw; s->tiles_y[l] = (h + th - 1) / th; s->loff_t[l] = k; k += s->tiles_x[l] * s->tiles_y[l];
        s->loff[l] = nvn_tex_level_offset(&t->d, l);
        uint64_t next = l + 1 < levels ? nvn_tex_level_offset(&t->d, l + 1) : s->layer_size;
        if (next - s->loff[l] != (uint64_t)s->tiles_x[l] * s->tiles_y[l] * SP_TILE) { static int nl; if (nl++ < 4) VKLOG("sparse texture %ux%u: level %u is not whole tiles, made whole", t->d.width, t->d.height, l); free(s); return NULL; }
    }
    s->tail_at = k; s->per_layer = k + (s->ntiled < levels);
    s->tail_off = s->ntiled < levels ? nvn_tex_level_offset(&t->d, s->ntiled) : s->layer_size;
    s->slot = calloc((size_t)layers * s->per_layer * s->nv, sizeof *s->slot); s->state = calloc((size_t)layers * s->per_layer, 1);
    s->qcap = 4096; s->q = malloc(s->qcap * sizeof *s->q);
    s->t = *t; s->fi = fi; s->im_levels = levels; InitializeSRWLock(&s->done_lk);
    return s;
}

static int sp_bind(VkSparseImageMemoryBind *b, uint32_t nb, VkSparseMemoryBind *ob, uint32_t nob, VkImage img) {
    VkSparseImageMemoryBindInfo ii = {img, nb, b}; VkSparseImageOpaqueMemoryBindInfo oi = {img, nob, ob};
    VkBindSparseInfo si = {VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, NULL, 0, NULL, 0, NULL, nob ? 1u : 0u, &oi, nb ? 1u : 0u, &ii, 0, NULL};
    extern _Atomic uint64_t g_t_bind, g_n_bind; uint64_t tb = now_ns();
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence f; vkCreateFence(g_vk.dev, &fci, NULL, &f);
    AcquireSRWLockExclusive(&g_vk.qlock);
    VkResult r = vkQueueBindSparse(g_vk.q, 1, &si, f);
    ReleaseSRWLockExclusive(&g_vk.qlock);
    if (r == VK_SUCCESS) vkWaitForFences(g_vk.dev, 1, &f, VK_TRUE, UINT64_MAX); else VKRED("vkQueueBindSparse (image) failed (%d)", r);
    vkDestroyFence(g_vk.dev, f, NULL);
    atomic_fetch_add(&g_t_bind, now_ns() - tb); atomic_fetch_add(&g_n_bind, 1);
    return r == VK_SUCCESS ? 0 : -1;
}

// After vkCreateImage: Vulkan's mip tail bound whole, and the levels of NVN's tail that are not in it
static int sparse_setup(vkt_image *im, vkt_sparse *s, const VkMemoryRequirements *mr) {
    uint32_t n = 4; VkSparseImageMemoryRequirements rq[4];
    vkGetImageSparseMemoryRequirements(g_vk.dev, im->img, &n, rq);
    if (!n) return -1;
    VkSparseImageMemoryRequirements *q = &rq[0];
    s->vtail = q->imageMipTailFirstLod; s->im_w = im->w; s->im_h = im->h;
    int single = (q->formatProperties.flags & VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT) != 0;
    VkDeviceSize tails = q->imageMipTailFirstLod >= im->levels ? 0 : single ? q->imageMipTailSize : q->imageMipTailSize * im->layers;
    if (tails) {
        VkMemoryRequirements r = *mr; r.size = tails;
        if (vk_alloc_mem(&r, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &s->tail_mem)) { VKRED("sparse texture: no memory for its mip tail (%llu bytes)", (unsigned long long)tails); return -1; }
        uint32_t nb = single ? 1 : im->layers; VkSparseMemoryBind *b = calloc(nb, sizeof *b);
        for (uint32_t y = 0; y < nb; y++) b[y] = (VkSparseMemoryBind){q->imageMipTailOffset + (VkDeviceSize)y * q->imageMipTailStride, q->imageMipTailSize, s->tail_mem.mem, s->tail_mem.off + (VkDeviceSize)y * q->imageMipTailSize, 0};
        int rc = sp_bind(NULL, 0, b, nb, im->img); free(b);
        if (rc) return -1;
    }
    // levels of NVN's tail that Vulkan keeps in tiles: bound for good, their content comes with the tail
    for (uint32_t l = s->ntiled; l < s->vtail && l < im->levels; l++) {
        uint32_t lw = mip(im->w, l), lh = mip(im->h, l);
        for (uint32_t y = 0; y < im->layers; y++) for (uint32_t vy = 0; vy < lh; vy += s->vh) for (uint32_t vx = 0; vx < lw; vx += s->vw) {
            uint32_t sl; if (sp_slot(mr, &sl)) return -1;
            if (s->nextra == s->capextra) { s->capextra = s->capextra ? s->capextra * 2 : 256; s->extra = realloc(s->extra, s->capextra * sizeof *s->extra); }
            s->extra[s->nextra++] = sl;
            VkDeviceMemory m; VkDeviceSize o; sp_slot_mem(sl, &m, &o);
            VkSparseImageMemoryBind b = {{im->aspect, l, y}, {(int32_t)vx, (int32_t)vy, 0}, {MIN_(s->vw, lw - vx), MIN_(s->vh, lh - vy), 1}, m, o, 0};
            if (sp_bind(&b, 1, NULL, 0, im->img)) return -1;
        }
    }
    // The texture setting: the top tile levels of compressed textures left out take three quarters of the tile memory on a card that has too little. NXRT_VK_TILE_SKIP=N
    { static int sk = -1; if (sk < 0) { extern int g_opt_tile_skip; const char *e = getenv("NXRT_VK_TILE_SKIP"); sk = e ? atoi(e) : g_opt_tile_skip;
        if (sk < 0) { uint64_t dl = 0; for (uint32_t h = 0; h < g_vk.memp.memoryHeapCount; h++) if ((g_vk.memp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && g_vk.memp.memoryHeaps[h].size > dl) dl = g_vk.memp.memoryHeaps[h].size; sk = dl < (6ull << 30) ? 1 : 0; }
        if (sk > 2) sk = 2; VKLOG("texture quality: %d tile levels left out", sk); }
      if (sk && g_vk.has.view_min_lod && (im->fi->flags & VKF_COMPRESSED) && s->ntiled > (uint32_t)sk && s->vtail > (uint32_t)sk) { s->skip = (uint32_t)sk; im->min_lod = (uint32_t)sk; } }
    VKLOG("sparse texture %ux%ux%u %s, %u levels: NVN tile %ux%u, Vulkan %ux%u, %u tiles a layer, NVN's tail at level %u, Vulkan's at %u (%llu bytes)", im->w, im->h, im->layers, im->fi->name, im->levels, s->tw, s->th, s->vw, s->vh,
          s->per_layer, s->ntiled, s->vtail, (unsigned long long)tails);
    return 0;
}

static void sp_q_push(vkt_sparse *s, uint32_t y, uint32_t i) {
    if (s->skip && i != s->tail_at && i < s->loff_t[s->skip < s->ntiled ? s->skip : s->ntiled - 1]) return;
    uint8_t *st = &s->state[(size_t)y * s->per_layer + i]; if (*st & 2) return;
    *st |= 2;
    if (s->qt - s->qh == s->qcap) { uint64_t nc = s->qcap * 2, *nq = malloc(nc * sizeof *nq); for (uint64_t k = 0; k < s->qcap; k++) nq[k] = s->q[(s->qh + k) % s->qcap]; s->qt -= s->qh; s->qh = 0; free(s->q); s->q = nq; s->qcap = nc; }
    s->q[s->qt++ % s->qcap] = (uint64_t)y << 32 | i;
}
static void sp_tile_addr(const vkt_sparse *s, const tex_t *t, uint32_t y, uint32_t i, uint64_t *a, uint64_t *n) {
    *a = (uint64_t)(uintptr_t)t->mem + y * s->layer_size;
    if (i == s->tail_at) { *a += s->tail_off; *n = s->layer_size - s->tail_off; return; }
    uint32_t l = 0; while (l + 1 < s->ntiled && i >= s->loff_t[l + 1]) l++;
    *a += s->loff[l] + (uint64_t)(i - s->loff_t[l]) * SP_TILE; *n = SP_TILE;
}
// a place lets go of its memory: shared memory goes when its last place does
static void sp_tile_release(uint32_t *sl, uint32_t nv) {
    uint32_t e = 0; for (uint32_t v = 0; v < nv; v++) if (sl[v] && g_sp_ent && g_sp_ent[sl[v] - 1]) e = g_sp_ent[sl[v] - 1];
    if (e) {
        sp_share *h = &g_sh[e - 1];
        if (!--h->rc) { for (uint32_t v = 0; v < h->nv; v++) { g_sp_ent[h->s[v]] = 0; sp_slot_free(h->s[v]); } h->p = (const uint8_t *)1; h->fmt = 0; }   // a tombstone keeps the probe chains
    } else for (uint32_t v = 0; v < nv; v++) if (sl[v]) sp_slot_free(sl[v] - 1);
    memset(sl, 0, nv * sizeof *sl);
}
// the tile's bound memory is given back when the game unmapped it; a tile it mapped goes in the queue
static void sp_look_tile(vkt_image *im, vkt_sparse *s, const tex_t *t, uint32_t y, uint32_t i, VkSparseImageMemoryBind *ub, uint32_t *nub) {
    uint64_t a, n; sp_tile_addr(s, t, y, i, &a, &n);
    int st = nvn_sparse_tile_state(t->pool, a, n, s->gen); if (st < 0) st = 0;
    if (st & 1) { if (!(s->state[(size_t)y * s->per_layer + i] & 1) || (st & 2)) sp_q_push(s, y, i); return; }
    s->state[(size_t)y * s->per_layer + i] &= ~1;
    if (i == s->tail_at) return;
    uint32_t l = 0; while (l + 1 < s->ntiled && i >= s->loff_t[l + 1]) l++;
    if (l >= s->vtail) return;
    uint32_t j = i - s->loff_t[l], tx = j % s->tiles_x[l], ty = j / s->tiles_x[l], lw = mip(im->w, l), lh = mip(im->h, l), side = s->nv == 4 ? 2 : 1;
    uint32_t *sl = &s->slot[((size_t)y * s->per_layer + i) * s->nv];
    for (uint32_t v = 0; v < s->nv; v++) if (sl[v]) {
        uint32_t x0 = (tx * side + v % side) * s->vw, y0 = (ty * side + v / side) * s->vh;
        if (x0 < lw && y0 < lh && *nub < 4096) ub[(*nub)++] = (VkSparseImageMemoryBind){{im->aspect, l, y}, {(int32_t)x0, (int32_t)y0, 0}, {MIN_(s->vw, lw - x0), MIN_(s->vh, lh - y0), 1}, VK_NULL_HANDLE, 0, 0};
    }
    sp_tile_release(sl, s->nv);
}

// The memory behind a level changed under mapped tiles (a GPU copy, written through the page table): they are read again
void vkt_sparse_touch(vkt_image *im, unsigned level, unsigned layer, unsigned n) {
    vkt_sparse *s = im->sp; if (!s) return;
    AcquireSRWLockExclusive(&g_sp_lk);
    for (uint32_t y = layer; y < layer + n && y < s->layers; y++) {
        uint32_t i0 = level < s->ntiled ? s->loff_t[level] : s->tail_at, i1 = level < s->ntiled ? (level + 1 < s->ntiled ? s->loff_t[level + 1] : s->tail_at) : s->tail_at + 1;
        for (uint32_t i = i0; i < i1 && i < s->per_layer; i++) if (s->state[(size_t)y * s->per_layer + i] & 1) sp_q_push(s, y, i);
    }
    ReleaseSRWLockExclusive(&g_sp_lk);
}
// A tile's content, made on a worker: one region for a tile, one per level for a tail
typedef struct { uint32_t level, x0, y0, cw, ch, rt, ih; size_t size; uint8_t *data; } sp_region;
typedef struct tex_async { struct sp_job *done; SRWLOCK lk; volatile LONG pending, dead; } tex_async;
typedef struct sp_job { struct sp_job *next; vkt_sparse *s; uint32_t y, i, nr; sp_region r[8]; tex_async *ta; uint8_t *lin; uint32_t w, h, bw, bh, srgb; } sp_job;
static sp_job *volatile g_sj_head; static sp_job *g_sj_tail; static SRWLOCK g_sj_lk = SRWLOCK_INIT; static CONDITION_VARIABLE g_sj_cv = CONDITION_VARIABLE_INIT;
static volatile LONG g_sj_out; static int g_sj_threads;
static void sp_job_free(sp_job *j) { for (uint32_t k = 0; k < j->nr; k++) free(j->r[k].data); free(j->lin); free(j); }
static void ta_free(tex_async *a) { for (sp_job *j = a->done, *n; j; j = n) { n = j->next; sp_job_free(j); } free(a); }
static void sp_free_s(vkt_sparse *s) {
    for (sp_job *j = s->done, *n; j; j = n) { n = j->next; sp_job_free(j); }
    free(s->slot); free(s->state); free(s->q); free(s->extra); free(s);
}
static int sp_convert(vkt_sparse *s, sp_region *r, const uint8_t *lin, size_t lsize, uint32_t w, uint32_t h, uint32_t wblk, uint32_t hblk) {
    const vkf_info *fi = s->fi;
    if (s->astc) {
        r->size = vka_bc7_size(w, h, 1); r->data = malloc(r->size); if (!r->data) return -1;
        vka_key pk; int srgb = (fi->flags & VKF_SRGB) != 0;
        if (!vka_pack_get(lin, w, h, fi->bw, fi->bh, srgb, r->data, r->size, &pk)) {
            uint8_t *rgba = malloc((size_t)w * h * 4);
            int bad = !rgba || vka_decode_rgba8(lin, w, h, 1, fi->bw, fi->bh, srgb, rgba) || vka_encode_bc7_st(rgba, w, h, VKA_FAST, r->data);
            free(rgba); if (bad) return -1;
            vka_pack_put(&pk, r->data, r->size);
        }
        r->rt = (w + 3) / 4 * 4; r->ih = (h + 3) / 4 * 4;
    } else {
        r->size = lsize; r->data = malloc(lsize ? lsize : 1); if (!r->data) return -1;
        memcpy(r->data, lin, lsize); r->rt = wblk * fi->bw; r->ih = hblk * fi->bh;
    }
    return 0;
}
static void sp_job_run(sp_job *j) {
    vkt_sparse *s = j->s; const tex_t *t = &s->t; uint32_t y = j->y, i = j->i; uint64_t a, n; sp_tile_addr(s, t, y, i, &a, &n);
    if (i == s->tail_at) {
        uint8_t *tail = malloc(n); if (!tail) return;
        nvn_sparse_copy(t->pool, tail, a, n);
        for (uint32_t tl = s->ntiled; tl < s->im_levels && j->nr < 8; tl++) {
            s2tex_surface q; uint64_t base; size_t row = 0;
            if (nvn_texio_surface(t, tl, y, &q, &base, &row) != 0) continue;
            uint64_t in = base - ((uint64_t)y * s->layer_size + s->tail_off);
            size_t lrow = (size_t)q.width * q.bytes_per_block, lsize = lrow * q.height; uint8_t *lin = malloc(lsize ? lsize : 1);
            uint32_t lw = mip(s->im_w, tl), lh = mip(s->im_h, tl); sp_region *r = &j->r[j->nr];
            if (lin && !s2tex_deswizzle(&q, tail + in, (size_t)(n - in), lin, lrow, lrow * q.height) && !sp_convert(s, r, lin, lsize, lw, lh, q.width, q.height)) {
                r->level = tl; r->x0 = r->y0 = 0; r->cw = lw; r->ch = lh; j->nr++; }
            free(lin);
        }
        free(tail);
        return;
    }
    if (s->gpu) return;
    uint32_t l = 0; while (l + 1 < s->ntiled && i >= s->loff_t[l + 1]) l++;
    uint32_t k = i - s->loff_t[l], tx = k % s->tiles_x[l], ty = k / s->tiles_x[l], lw = mip(s->im_w, l), lh = mip(s->im_h, l), x0 = tx * s->tw, y0 = ty * s->th;
    uint8_t *tile = malloc(SP_TILE), *lin = malloc(SP_TILE);
    nvn_sparse_copy(t->pool, tile, a, SP_TILE);
    unsigned gh, gd; nvn_tex_level_tiles(&t->d, l, &gh, &gd);
    uint32_t gobs = 128u >> gh, rows = 8u << gh, bpb = s->fi->bpb, wblk = gobs * 64u / bpb;
    s2tex_surface q; memset(&q, 0, sizeof q); q.width = wblk; q.height = rows; q.depth = 1; q.bytes_per_block = bpb; q.block_height = gh; q.gobs_x = gobs; q.order = nvn_texio_order();
    size_t lrow = (size_t)wblk * bpb; sp_region *r = &j->r[0];
    if (!s2tex_deswizzle(&q, tile, SP_TILE, lin, lrow, lrow * rows) && !sp_convert(s, r, lin, lrow * rows, s->tw, s->th, wblk, rows)) {
        r->level = l; r->x0 = x0; r->y0 = y0; r->cw = MIN_(s->tw, lw - x0); r->ch = MIN_(s->th, lh - y0); j->nr = 1; }
    free(tile); free(lin);
}
static void sp_tex_job_run(sp_job *j) {   // a level of a texture (vk_tex.c upload): ASTC to BC3, or the pack
    tex_async *a = j->ta;
    if (!a->dead) {
        sp_region *r = &j->r[0]; r->size = vka_bc7_size(j->w, j->h, 1); r->data = malloc(r->size); vka_key pk; int ok = 0;
        if (r->data && vka_pack_get(j->lin, j->w, j->h, (int)j->bw, (int)j->bh, (int)j->srgb, r->data, r->size, &pk)) ok = 1;
        else if (r->data) { uint8_t *rgba = malloc((size_t)j->w * j->h * 4);
            if (rgba && !vka_decode_rgba8(j->lin, j->w, j->h, 1, (int)j->bw, (int)j->bh, (int)j->srgb, rgba) && !vka_encode_bc7_st(rgba, j->w, j->h, VKA_FAST, r->data)) { ok = 1; vka_pack_put(&pk, r->data, r->size); }
            free(rgba); }
        if (ok) { r->level = j->i; r->x0 = r->y0 = 0; r->cw = j->w; r->ch = j->h; r->rt = (j->w + 3) / 4 * 4; r->ih = (j->h + 3) / 4 * 4; j->nr = 1; }
        else { free(r->data); r->data = NULL; }
    }
    free(j->lin); j->lin = NULL;
    AcquireSRWLockExclusive(&a->lk); j->next = a->done; a->done = j; ReleaseSRWLockExclusive(&a->lk);
    InterlockedDecrement(&g_sj_out);
    if (!InterlockedDecrement(&a->pending) && a->dead) ta_free(a);
}
// The replay waits for a texture: rather than spin, it takes the next job of a texture from the queue (they go ahead of the tiles). 1 when it ran one
static int sp_help(void) {
    AcquireSRWLockExclusive(&g_sj_lk);
    sp_job *j = g_sj_head && g_sj_head->ta ? g_sj_head : NULL;
    if (j) { g_sj_head = j->next; if (!g_sj_head) g_sj_tail = NULL; }
    ReleaseSRWLockExclusive(&g_sj_lk);
    if (j) sp_tex_job_run(j);
    return j != NULL;
}
static unsigned __stdcall sp_worker(void *arg) {
    (void)arg; SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    for (;;) {
        AcquireSRWLockExclusive(&g_sj_lk);
        while (!g_sj_head) SleepConditionVariableSRW(&g_sj_cv, &g_sj_lk, INFINITE, 0);
        sp_job *j = g_sj_head; g_sj_head = j->next; if (!g_sj_head) g_sj_tail = NULL;
        ReleaseSRWLockExclusive(&g_sj_lk);
        if (j->ta) { sp_tex_job_run(j); continue; }
        vkt_sparse *s = j->s;
        if (!s->dead) sp_job_run(j);
        AcquireSRWLockExclusive(&s->done_lk); j->next = s->done; s->done = j; ReleaseSRWLockExclusive(&s->done_lk);
        InterlockedDecrement(&g_sj_out);
        if (!InterlockedDecrement(&s->pending) && s->dead) sp_free_s(s);
    }
    return 0;
}
static void sp_workers_start(void) { if (!g_sj_threads) { g_sj_threads = 1; int nw = vkc_int("sj_threads", 4); if (nw < 1) nw = 1; if (nw > 32) nw = 32;   // NXRT_VK_SJ_THREADS
    for (int k = 0; k < nw; k++) CloseHandle((HANDLE)_beginthreadex(NULL, 256 << 10, sp_worker, NULL, 0, NULL)); } }
static void tex_job_add(vkt_image *im, uint8_t *lin, uint32_t w, uint32_t h, int bw, int bh, int srgb, uint32_t dl, uint32_t layer) {
    sp_workers_start();
    if (!im->as) { im->as = calloc(1, sizeof *im->as); InitializeSRWLock(&im->as->lk); }
    sp_job *j = calloc(1, sizeof *j); j->ta = im->as; j->lin = lin; j->w = w; j->h = h; j->bw = (uint32_t)bw; j->bh = (uint32_t)bh; j->srgb = (uint32_t)srgb; j->i = dl; j->y = layer;
    InterlockedIncrement(&im->as->pending); InterlockedIncrement(&g_sj_out);
    AcquireSRWLockExclusive(&g_sj_lk);   // ahead of the tiles: the replay waits for these
    j->next = g_sj_head; g_sj_head = j; if (!g_sj_tail) g_sj_tail = j;
    ReleaseSRWLockExclusive(&g_sj_lk); WakeConditionVariable(&g_sj_cv);
}
// the levels the workers finished, within the frame's budget
static void tex_async_apply(vkt_image *im, VkCommandBuffer cb) {
    static uint64_t bframe, bytes; if (bframe != g_frame) { bframe = g_frame; bytes = 0; }
    if (bytes >= (48u << 20)) return;
    tex_async *a = im->as;
    AcquireSRWLockExclusive(&a->lk); sp_job *j = a->done; a->done = NULL; ReleaseSRWLockExclusive(&a->lk);
    vk_image_barrier(cb, im->img, im->aspect, im->rest, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    sp_job *back = NULL;
    while (j) {
        sp_job *n = j->next; int keep = 0;
        if (j->nr) {
            sp_region *r = &j->r[0]; VkBuffer sb; VkDeviceSize so; uint8_t *p = bytes < (48u << 20) ? vkm_staging(im->bc1 ? r->size / 2 : r->size, 16, &sb, &so) : NULL;
            if (p) { bytes += stage_bc(im, r->data, r->size, p);   // a level that was not opaque leaves bc1_bad: vkt_resolve makes the image again at its next use
                VkBufferImageCopy c = {so, r->rt, r->ih, {im->aspect, r->level, j->y, 1}, {0, 0, 0}, {r->cw, r->ch, 1}};
                vkCmdCopyBufferToImage(cb, sb, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c); }
            else keep = 1;
        }
        if (keep) { j->next = back; back = j; } else sp_job_free(j);
        j = n;
    }
    vk_image_barrier(cb, im->img, im->aspect, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, im->rest, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
    if (back) { AcquireSRWLockExclusive(&a->lk); sp_job *e = back; while (e->next) e = e->next; e->next = a->done; a->done = back; ReleaseSRWLockExclusive(&a->lk); }
}
static void tex_async_free(vkt_image *im) {
    tex_async *a = im->as; if (!a) return;
    im->as = NULL; InterlockedIncrement(&a->pending); a->dead = 1;
    if (!InterlockedDecrement(&a->pending)) ta_free(a);
}
static void sp_job_add(vkt_sparse *s, uint32_t y, uint32_t i) {
    sp_workers_start();
    sp_job *j = calloc(1, sizeof *j); j->s = s; j->y = y; j->i = i;
    InterlockedIncrement(&s->pending); InterlockedIncrement(&g_sj_out);
    AcquireSRWLockExclusive(&g_sj_lk);
    if (g_sj_tail) g_sj_tail->next = j; else g_sj_head = j;
    g_sj_tail = j;
    ReleaseSRWLockExclusive(&g_sj_lk); WakeConditionVariable(&g_sj_cv);
}
// A finished job: the tile's memory bound (collected in b), its content into staging and copied. 0 done, 1 no memory or staging now: try again later
static uint64_t g_sp_shared;
static int sp_apply(vkt_image *im, vkt_sparse *s, sp_job *j, VkCommandBuffer cb, const VkMemoryRequirements *mr, VkSparseImageMemoryBind *b, uint32_t *nb) {
    uint32_t y = j->y, i = j->i;
    if (i != s->tail_at) {
        uint32_t l = 0; while (l + 1 < s->ntiled && i >= s->loff_t[l + 1]) l++;
        if (l < s->vtail) {
            uint32_t k = i - s->loff_t[l], tx = k % s->tiles_x[l], ty = k / s->tiles_x[l], lw = mip(im->w, l), lh = mip(im->h, l), x0 = tx * s->tw, y0 = ty * s->th;
            uint32_t side = s->nv == 4 ? 2 : 1, *sl = &s->slot[((size_t)y * s->per_layer + i) * s->nv];
            uint64_t a, n2; sp_tile_addr(s, &s->t, y, i, &a, &n2);
            int whole = x0 + s->tw <= lw && y0 + s->th <= lh;
            const uint8_t *ph = whole && !sl[0] ? nvn_sparse_tile_phys(s->t.pool, a) : NULL;
            uint32_t he = ph ? sp_sh_find(ph, s->t.d.format, 1) : ~0u;
            sp_share *sh = he != ~0u ? &g_sh[he] : NULL;
            if (sh && sh->rc) {   // this memory has a tile already: the place shares it, nothing to upload
                for (uint32_t v = 0; v < s->nv; v++) {
                    VkDeviceMemory m; VkDeviceSize o; sp_slot_mem(sh->s[v], &m, &o); sl[v] = sh->s[v] + 1;
                    b[(*nb)++] = (VkSparseImageMemoryBind){{im->aspect, l, y}, {(int32_t)(x0 + (v % side) * s->vw), (int32_t)(y0 + (v / side) * s->vh), 0}, {s->vw, s->vh, 1}, m, o, 0};
                }
                sh->rc++; g_sp_shared++;
                for (uint32_t k = 0; k < j->nr; k++) { free(j->r[k].data); j->r[k].data = NULL; }
                j->nr = 0;
                s->state[(size_t)y * s->per_layer + i] = (s->state[(size_t)y * s->per_layer + i] & ~2) | 1;
                return 0;
            }
            for (uint32_t v = 0; v < s->nv; v++) {
                uint32_t vx = x0 + (v % side) * s->vw, vy = y0 + (v / side) * s->vh;
                if (vx >= lw || vy >= lh || sl[v]) continue;
                uint32_t sl1; if (sp_slot(mr, &sl1)) { static int k2; if (k2++ < 8) VKRED("sparse textures: out of the tile budget (%llu tiles live)", (unsigned long long)g_sp.live); return 2; }
                VkDeviceMemory m; VkDeviceSize o; sp_slot_mem(sl1, &m, &o); sl[v] = sl1 + 1;
                b[(*nb)++] = (VkSparseImageMemoryBind){{im->aspect, l, y}, {(int32_t)vx, (int32_t)vy, 0}, {MIN_(s->vw, lw - vx), MIN_(s->vh, lh - vy), 1}, m, o, 0};
            }
            if (sh) { sh->rc = 1; sh->nv = s->nv; for (uint32_t v = 0; v < s->nv; v++) { sh->s[v] = sl[v] - 1; g_sp_ent[sl[v] - 1] = he + 1; } }
        }
    }
    for (uint32_t k = 0; k < j->nr; k++) {
        sp_region *r = &j->r[k]; VkBuffer sb; VkDeviceSize so; uint8_t *p = vkm_staging(r->size, 16, &sb, &so);
        if (!p) return 1;
        memcpy(p, r->data, r->size);
        VkBufferImageCopy c = {so, r->rt, r->ih, {im->aspect, r->level, y, 1}, {(int32_t)r->x0, (int32_t)r->y0, 0}, {r->cw, r->ch, 1}};
        vkCmdCopyBufferToImage(cb, sb, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        free(r->data); r->data = NULL;
    }
    j->nr = 0;
    s->state[(size_t)y * s->per_layer + i] = (s->state[(size_t)y * s->per_layer + i] & ~2) | 1;
    return 0;
}

static int sp_cmp(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
// A look when the texture is resolved: the game's mappings since the last one; tiles go to the workers; what they finished is bound and copied, within the frame's budget
static void sparse_look(vkt_image *im, const tex_t *t, VkCommandBuffer cb) {
    vkt_sparse *s = im->sp;
    uint64_t seq = nvn_sparse_log_seq(t->pool);
    if (seq == s->seq && s->seen && ((s->qh == s->qt && !s->pending && !s->done) || s->looked == g_frame)) return;   // one look a frame: it ran at every bind of the texture
    s->looked = g_frame;
    static VkSparseImageMemoryBind *ub; if (!ub) ub = malloc(4096 * sizeof *ub);
    uint32_t nub = 0; uint64_t gen = nvn_sparse_gen();
    AcquireSRWLockExclusive(&g_sp_lk);
    enum { RANGES = 4096 }; static uint64_t (*ranges)[2]; if (!ranges) ranges = malloc(RANGES * sizeof *ranges);
    if (seq != s->seq || !s->seen) {
        int nr = s->seen ? nvn_sparse_log_read(t->pool, &s->seq, ranges, RANGES) : -1;
        if (nr < 0) {
            if (!s->seen) s->seq = seq;
            for (uint32_t y = 0; y < s->layers; y++) for (uint32_t i = 0; i < s->per_layer; i++) { if (nub + 8 > 4096) { sp_bind(ub, nub, NULL, 0, im->img); nub = 0; } sp_look_tile(im, s, t, y, i, ub, &nub); }
        } else if (nr) {
            qsort(ranges, (size_t)nr, sizeof ranges[0], sp_cmp);
            uint64_t base = (uint64_t)(uintptr_t)t->mem, end = base + (uint64_t)s->layers * s->layer_size;
            for (int k = 0; k < nr; k++) {
                uint64_t lo = ranges[k][0] < base ? base : ranges[k][0], hi = ranges[k][1] > end ? end : ranges[k][1];
                for (uint64_t o = lo < hi ? (lo - base) & ~(uint64_t)(SP_TILE - 1) : ~0ull; o != ~0ull && base + o < hi;) {
                    uint32_t y = (uint32_t)(o / s->layer_size); uint64_t in = o % s->layer_size;
                    if (nub + 8 > 4096) { sp_bind(ub, nub, NULL, 0, im->img); nub = 0; }
                    if (in >= s->tail_off) { if (s->ntiled < s->levels) sp_look_tile(im, s, t, y, s->tail_at, ub, &nub); o += s->layer_size - in; continue; }
                    uint32_t l = 0; while (l + 1 < s->ntiled && in >= s->loff[l + 1]) l++;
                    sp_look_tile(im, s, t, y, s->loff_t[l] + (uint32_t)((in - s->loff[l]) / SP_TILE), ub, &nub);
                    o += SP_TILE;
                }
            }
        }
        if (nub) sp_bind(ub, nub, NULL, 0, im->img);
        s->seen = 1; s->gen = gen;
    }
    // to the workers, at most 96 out at a time
    uint64_t sent = 0;
    while (s->qh != s->qt && g_sj_out < 96) {
        uint64_t e = s->q[s->qh % s->qcap]; s->qh++; sent++;
        sp_job_add(s, (uint32_t)(e >> 32), (uint32_t)e);
    }
    // what they finished, up to 32 MB of uploads a frame across all textures
    static uint64_t bframe, bytes; if (bframe != g_frame) { bframe = g_frame; bytes = 0; }
    AcquireSRWLockExclusive(&s->done_lk); sp_job *j = s->done; s->done = NULL; ReleaseSRWLockExclusive(&s->done_lk);
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vk.dev, im->img, &mr);
    uint32_t nb = 0; uint64_t applied = 0; sp_job *back = NULL;
    while (j) {
        sp_job *n = j->next; uint64_t a, l; sp_tile_addr(s, t, j->y, j->i, &a, &l);
        int st = nvn_sparse_tile_state(t->pool, a, l, ~0ull), rc = 1;
        if (bytes < (32u << 20) && nb + 8 <= 4096) {
            for (uint32_t k = 0; k < j->nr; k++) bytes += j->r[k].size;
            rc = st > 0 && (st & 1) ? sp_apply(im, s, j, cb, &mr, ub, &nb) : 0;
            if (rc) for (uint32_t k = 0; k < j->nr; k++) bytes -= j->r[k].size;
        }
        if (rc == 2) { s->state[(size_t)j->y * s->per_layer + j->i] &= ~2; sp_job_free(j); }   // out of tile memory: dropped, the mapping is looked at again when it changes
        else if (rc) { j->next = back; back = j; }
        else { s->state[(size_t)j->y * s->per_layer + j->i] &= ~2; sp_job_free(j); applied++; }
        j = n;
    }
    if (back) { AcquireSRWLockExclusive(&s->done_lk); sp_job *e = back; while (e->next) e = e->next; e->next = s->done; s->done = back; ReleaseSRWLockExclusive(&s->done_lk); }
    if (nb) sp_bind(ub, nb, NULL, 0, im->img);
    ReleaseSRWLockExclusive(&g_sp_lk);
    static uint64_t looks; if (applied || nub) { looks++; if (looks <= 20 || looks % 500 == 0) VKLOG("sparse texture ser=%llu: %llu tiles applied, %llu sent, %llu waiting, %u unmapped; %llu MB of tiles live (peak %llu)", (unsigned long long)t->serial, (unsigned long long)applied,
        (unsigned long long)sent, (unsigned long long)(s->qt - s->qh), nub, (unsigned long long)(g_sp.live * SP_TILE >> 20), (unsigned long long)(g_sp.peak * SP_TILE >> 20)); }
    (void)g_sp_shared;
}
static void sparse_free(vkt_image *im) {
    vkt_sparse *s = im->sp; if (!s) return;
    AcquireSRWLockExclusive(&g_sp_lk);
    for (size_t i = 0, n = (size_t)s->layers * s->per_layer; i < n; i++) sp_tile_release(&s->slot[i * s->nv], s->nv);
    for (uint32_t i = 0; i < s->nextra; i++) sp_slot_free(s->extra[i]);
    ReleaseSRWLockExclusive(&g_sp_lk);
    if (s->tail_mem.mem) vk_free_mem(&s->tail_mem);
    im->sp = NULL;
    InterlockedIncrement(&s->pending); s->dead = 1;
    if (!InterlockedDecrement(&s->pending)) sp_free_s(s);
}
