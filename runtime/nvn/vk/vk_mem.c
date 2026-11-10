// GPU memory: pool mirrors, sparse backing, the staging ring, dirty-page uploads, the per-frame budget (vk_mem.h; docs/m5-vulkan.md §3).
// nxnative's Metal backend wraps each pool zero-copy (newBufferWithBytesNoCopy, unified memory); a discrete GPU needs copies, and importing guest memory
// as host memory is what research/01 lesson 10 rules out. Behaviour of pool lookups follows nvn_mtl.m (mtl_pool_add, pool_find); the rest is new.
#include "vk_mem.h"
#include "vk_submit.h"
#include "vk_flight.h"
#include "nvn_dirty.h"   // gfx/: the dirty-page tracker the recording layer shares (nxnative's, ported by track A)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void dm_guest_write_prepare(void *addr, size_t len);   // host/crash.c: unprotect before the host writes guest memory

enum { PAGE = 4096, GRAN = 2u << 20, HOT_STREAK = 3, HOT_COOL_FRAMES = 240, MAXPOOL = 16384, RUN_MAX = 1024, MAXWIN = 32 };
// A pool larger than the device's largest buffer (AMD: 2 GiB; the game's biggest pool is 3.4 GiB) gets several buffers, windows over it. With sparse
// binding and aliasing the windows overlap and share the same memory, so any range up to the overlap fits in one of them; otherwise they are side by side.
typedef struct {
    uintptr_t lo, hi, base;       // the pool [lo, hi) as given; base = lo rounded down to a page
    uint64_t size, npages;        // bytes the buffer covers (from base, whole pages)
    const void *key;
    VkBuffer buf[MAXWIN]; uint64_t win_lo[MAXWIN], win_n[MAXWIN]; int nwin;   // window k covers [win_lo, win_lo + win_n) of the pool, from base
    int sparse;
    vk_alloc *gran; uint64_t ngran, bound;   // sparse: one allocation per bound granule, bound into every window that covers it
    vk_alloc mem[MAXWIN];         // not sparse: one allocation per window
    const void *span;             // the dirty tracker's range (registered by the recording layer right after gfx_pool_add)
    uint64_t *up;                 // per page: the tracker's stamp when its device copy was taken (0 = never copied)
    uint32_t *last_dirty; uint8_t *streak, *hot;   // hot pages: written in HOT_STREAK consecutive frames, left writable and copied at the first use of each submission
    uint32_t *hot_at;             // per page: the submission that copied it (a 4 MB pool bound by a thousand draws was 16 GB of copies a frame)
    uint8_t *committed;           // per page: seen committed and readable once (pool_sync asks VirtualQuery only for pages not known yet)
    _Atomic uint8_t *gpu_pending;   // per page: GPU writes recorded whose bytes have not reached guest memory yet (the device copy is the newer one)
    uint32_t **shadow;            // per page written by the GPU: guest memory as the device has it. While a write-back is on its way only the words the CPU changed since go to the device
    uint64_t hot_n, cursor;       // vkm_sync_all's position
    uint64_t up_b, up_never, up_hot;   // bytes uploaded since the last report: all, of pages never copied, of hot pages
    uint32_t nvn_flags;           // the game's NVNmemoryPoolFlags (vkm_pool_flags): CPU_NO_ACCESS = the CPU never reads or writes it
} pool_t_;

static pool_t_ *g_pool[MAXPOOL]; static int g_npool;
static _Atomic uint32_t g_pool_gen = 1;   // moves whenever a pool comes or goes (vkm_use's remembered buffers and offsets are then stale)
static SRWLOCK g_lk = SRWLOCK_INIT;
static uint64_t g_frame = 1;
static _Atomic uint64_t s_wb_words, s_wb_kept, s_wb_stale, s_wb_merged;
static VkBufferUsageFlags g_usage;
static VkDeviceSize g_gran = GRAN, g_max_buffer;
static int g_alias;   // overlapping windows: sparse binding with aliasing (NXRT_VK_NO_ALIAS=1 turns them off, for tests)

static struct {
    VkBuffer buf; vk_alloc mem; uint8_t *map; VkDeviceSize size, head, tail; int wrapped;
    struct { uint64_t value; int owner; VkDeviceSize begin, end; VkBuffer ovf; vk_alloc ovf_mem; } q[4096]; unsigned qh, qt;   // allocations in order; value 0 until owner commits
    SRWLOCK lk;
} g_st;

static uint64_t g_budget = 64ull << 20, g_budget_used;
_Atomic uint64_t g_t_bind, g_n_bind;   // time in sparse binds, with vk_sparse.inc: they wait for the queue
static int g_seen_on = 1;   // gamectl set use_seen 0: every use looks again (a same-run test of vkm_use)
static _Atomic uint64_t s_up_bytes, s_up_pages, s_up_runs, s_up_ns, s_sync, s_st_bytes, s_st_ovf, s_st_waits, s_wb_bytes, s_uncommitted;

// ---------- Staging ring ----------
static void st_reclaim(void) {   // caller holds g_st.lk
    uint64_t done = atomic_load(&g_vk.tl_done);
    while (g_st.qt != g_st.qh) {
        unsigned i = g_st.qt % 4096;
        if (!g_st.q[i].value || g_st.q[i].value > done) break;
        if (g_st.q[i].ovf) { vkDestroyBuffer(g_vk.dev, g_st.q[i].ovf, NULL); vk_free_mem(&g_st.q[i].ovf_mem); g_st.q[i].ovf = VK_NULL_HANDLE; }
        else g_st.tail = g_st.q[i].end;
        g_st.qt++;
    }
    if (g_st.qt == g_st.qh) { g_st.head = g_st.tail = 0; g_st.wrapped = 0; }
}
// Called by vk_submit at each commit: that slot's allocations now live until its timeline value
static _Atomic uint32_t g_sub_epoch = 1;   // moves at each commit
int g_vkm_same_epoch;                      // a commit in the middle of an NVN submission (vks_split): the same submission as far as guest memory goes
void vkm_committed(int owner, uint64_t value) {
    if (!g_vkm_same_epoch) atomic_fetch_add(&g_sub_epoch, 1);
    AcquireSRWLockExclusive(&g_st.lk);
    for (unsigned k = g_st.qt; k != g_st.qh; k++) if (!g_st.q[k % 4096].value && g_st.q[k % 4096].owner == owner) g_st.q[k % 4096].value = value;
    ReleaseSRWLockExclusive(&g_st.lk);
}
int vks_owner(void);
void *vkm_staging(VkDeviceSize n, VkDeviceSize align, VkBuffer *buf, VkDeviceSize *off) {
    int owner = vks_owner();
    if (owner < 0) { VKRED("staging asked for outside a submission"); return NULL; }
    if (!align) align = 16;
    n = (n + 15) & ~(VkDeviceSize)15;
    AcquireSRWLockExclusive(&g_st.lk);
    for (int attempt = 0; attempt < 64; attempt++) {
        st_reclaim();
        if (g_st.qh - g_st.qt >= 4095) goto wait;
        VkDeviceSize a = (g_st.head + align - 1) / align * align;
        int ok = 0;
        if (!g_st.wrapped) {   // free space: [head, size) then [0, tail)
            if (a + n <= g_st.size) ok = 1;
            else if (n <= g_st.tail) { a = 0; g_st.wrapped = 1; ok = 1; }
        } else if (a + n <= g_st.tail) ok = 1;
        if (ok) {
            unsigned i = g_st.qh++ % 4096;
            g_st.q[i].value = 0; g_st.q[i].owner = owner; g_st.q[i].begin = a; g_st.q[i].end = a + n; g_st.q[i].ovf = VK_NULL_HANDLE;
            g_st.head = a + n;
            ReleaseSRWLockExclusive(&g_st.lk);
            atomic_fetch_add(&s_st_bytes, n);
            *buf = g_st.buf; *off = a; return g_st.map + a;
        }
        if (n > g_st.size / 2) break;   // too big for the ring: a buffer of its own
    wait:;
        // the ring is full: wait for the oldest committed allocation's submission; when only the recording's own remain, overflow
        uint64_t v = g_st.qt != g_st.qh ? g_st.q[g_st.qt % 4096].value : 0;
        if (!v) break;
        ReleaseSRWLockExclusive(&g_st.lk);
        atomic_fetch_add(&s_st_waits, 1);
        vks_wait(v, UINT64_MAX);
        AcquireSRWLockExclusive(&g_st.lk);
    }
    // Overflow: a host buffer for this submission alone, released with it. One chunk of at least 32 MB serves the rest of the submission: a buffer and an allocation per request
    // was 1,800 allocations a second, and frames of 10 s, while MKW loads
    static struct { VkBuffer b; uint8_t *map; VkDeviceSize used, size; unsigned qi; int owner; } oc;
    if (oc.b && oc.owner == owner && g_st.qt <= oc.qi && oc.qi != g_st.qh && g_st.q[oc.qi % 4096].ovf == oc.b && !g_st.q[oc.qi % 4096].value) {
        VkDeviceSize a2 = (oc.used + align - 1) / align * align;
        if (a2 + n <= oc.size) { oc.used = a2 + n; ReleaseSRWLockExclusive(&g_st.lk); atomic_fetch_add(&s_st_bytes, n); *buf = oc.b; *off = a2; return oc.map + a2; }
    }
    VkDeviceSize chunk = n > (32u << 20) ? n : (32u << 20);
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, chunk, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkBuffer b = VK_NULL_HANDLE; vk_alloc m;
    void *p = NULL;
    if (vkCreateBuffer(g_vk.dev, &bci, NULL, &b) == VK_SUCCESS) {
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, b, &mr);
        if (!vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &m)) {
            vkBindBufferMemory(g_vk.dev, b, m.mem, m.off); p = m.map;
            unsigned i = g_st.qh++ % 4096;
            g_st.q[i].value = 0; g_st.q[i].owner = owner; g_st.q[i].begin = g_st.q[i].end = 0; g_st.q[i].ovf = b; g_st.q[i].ovf_mem = m;
            oc.b = b; oc.map = p; oc.used = n; oc.size = chunk; oc.qi = g_st.qh - 1; oc.owner = owner;
        } else { vkDestroyBuffer(g_vk.dev, b, NULL); b = VK_NULL_HANDLE; }
    }
    ReleaseSRWLockExclusive(&g_st.lk);
    if (!p) { VKRED("staging: no host memory for %llu bytes", (unsigned long long)n); return NULL; }
    atomic_fetch_add(&s_st_ovf, 1); atomic_fetch_add(&s_st_bytes, n);
    *buf = b; *off = 0; return p;
}
uint64_t vkm_budget_left(void) { return g_budget_used >= g_budget ? 0 : g_budget - g_budget_used; }
void vkm_budget_take(uint64_t n) { g_budget_used += n; }

// ---------- Pools ----------
static int pool_cmp(const void *a, const void *b) { uintptr_t x = (*(pool_t_ *const *)a)->lo, y = (*(pool_t_ *const *)b)->lo; return x < y ? -1 : x > y; }
static pool_t_ *pool_find(uintptr_t a, uint64_t len) {   // caller holds g_lk (shared): binary search on start, then a scan for overlapping pools
    int lo = 0, hi = g_npool - 1, hit = -1;
    while (lo <= hi) { int m = (lo + hi) / 2; if (g_pool[m]->lo <= a) { hit = m; lo = m + 1; } else hi = m - 1; }
    if (hit >= 0 && a + len <= g_pool[hit]->hi) return g_pool[hit];
    for (int i = 0; i < g_npool; i++) if (a >= g_pool[i]->lo && a + len <= g_pool[i]->hi) return g_pool[i];
    return NULL;
}

static int win_of(const pool_t_ *p, uint64_t off) { int k = p->nwin - 1; while (k > 0 && p->win_lo[k] > off) k--; return k; }   // the last window starting at or before off
static int win_for(const pool_t_ *p, uint64_t off, uint64_t n) { int k = win_of(p, off); return off + n <= p->win_lo[k] + p->win_n[k] ? k : -1; }   // the window holding all of it, or -1

static void pool_free(pool_t_ *p) {
    for (int k = 0; k < p->nwin; k++) if (p->buf[k]) vkDestroyBuffer(g_vk.dev, p->buf[k], NULL);
    if (p->sparse) { for (uint64_t g = 0; g < p->ngran; g++) vk_free_mem(&p->gran[g]); free(p->gran); }
    else for (int k = 0; k < p->nwin; k++) vk_free_mem(&p->mem[k]);
    free(p->up); free(p->hot_at); free(p->last_dirty); free(p->streak); free(p->hot); free((void *)p->gpu_pending); free(p->committed); free(p);
}
static void pool_free_cb(void *arg) { pool_free(arg); }

void vkm_pool_add(const void *mem, uint64_t size) {
    if (!g_vk.up || !mem || !size) return;
    pool_t_ *p = calloc(1, sizeof *p);
    p->lo = (uintptr_t)mem; p->hi = p->lo + size; p->key = mem;
    p->base = p->lo & ~(uintptr_t)(PAGE - 1);
    p->size = ((p->hi - p->base) + PAGE - 1) & ~(uint64_t)(PAGE - 1); p->npages = p->size / PAGE;
    p->up = calloc(p->npages, sizeof *p->up); p->last_dirty = calloc(p->npages, sizeof *p->last_dirty); p->streak = calloc(p->npages, 1); p->hot = calloc(p->npages, 1); p->hot_at = calloc(p->npages, sizeof *p->hot_at); p->gpu_pending = calloc(p->npages, 1);
    p->committed = calloc(p->npages, 1);
    // MKW has thousands of small pools. Sparse, each first use is a queue bind that waits for the GPU, with a 2 MB granule for a pool of 16 KB: frames of seconds while it loads. A small pool is a plain
    // buffer that gets its memory at first use
    p->sparse = g_vk.has.sparse_buffer && p->size >= (8u << 20);
    VkDeviceSize bsize = p->sparse ? (p->size + g_gran - 1) / g_gran * g_gran : p->size, w = bsize, ov = 0;
    if (g_max_buffer && bsize > g_max_buffer) {   // windows start on granules (or pages), so a granule never straddles two
        w = p->sparse ? g_max_buffer / g_gran * g_gran : g_max_buffer / PAGE * PAGE;
        if (p->sparse && g_alias) { ov = w / 4 < (256ull << 20) ? w / 4 : (256ull << 20); ov = ov / g_gran * g_gran; }
    }
    for (uint64_t at = 0; w > ov && p->nwin < MAXWIN; at += w - ov) {
        p->win_lo[p->nwin] = at; p->win_n[p->nwin] = bsize - at < w ? bsize - at : w; p->nwin++;
        if (at + w >= bsize) break;
    }
    if (!p->nwin || p->win_lo[p->nwin - 1] + p->win_n[p->nwin - 1] < bsize) {
        VKRED("memory pool %p+%#llx does not fit in %d buffers of the device's largest size (%#llx): not mirrored", mem, (unsigned long long)size, MAXWIN, (unsigned long long)g_max_buffer);
        p->nwin = 0; pool_free(p); return;
    }
    VkBufferCreateFlags fl = p->sparse ? VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT | (ov ? VK_BUFFER_CREATE_SPARSE_ALIASED_BIT : 0) : 0;
    int ok = p->up && p->hot && p->committed;
    for (int k = 0; ok && k < p->nwin; k++) {
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, fl, p->win_n[k], g_usage, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
        ok = vkCreateBuffer(g_vk.dev, &bci, NULL, &p->buf[k]) == VK_SUCCESS;
        if (ok) vk_name((uint64_t)p->buf[k], VK_OBJECT_TYPE_BUFFER, "nvn memory pool");
    }
    if (!ok) { VKRED("cannot create the buffer of memory pool %p+%#llx", mem, (unsigned long long)size); pool_free(p); return; }
    if (p->sparse) { p->ngran = bsize / g_gran; p->gran = calloc(p->ngran, sizeof *p->gran); for (uint64_t g = 0; g < p->ngran; g++) p->gran[g].block = -1; }
    else if (p->size >= (8u << 20)) {   // no sparse binding on this device, a large pool: all of it now
        for (int k = 0; k < p->nwin; k++) {
            VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, p->buf[k], &mr);
            if (vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &p->mem[k])) { VKRED("no device memory for memory pool %p+%#llx", mem, (unsigned long long)size); pool_free(p); return; }
            vkBindBufferMemory(g_vk.dev, p->buf[k], p->mem[k].mem, p->mem[k].off);
        }
        p->bound = p->size;
    }
    if (p->nwin > 1) VKLOG("memory pool %p+%#llx: %d buffers of up to %llu MB%s (the device's largest buffer is %llu MB)", mem, (unsigned long long)size, p->nwin,
                           (unsigned long long)(w >> 20), ov ? ", overlapping by the same memory" : ", side by side", (unsigned long long)(g_max_buffer >> 20));
    AcquireSRWLockExclusive(&g_lk);
    if (g_npool < MAXPOOL) { g_pool[g_npool++] = p; qsort(g_pool, (size_t)g_npool, sizeof *g_pool, pool_cmp); p = NULL; atomic_fetch_add(&g_pool_gen, 1); }
    ReleaseSRWLockExclusive(&g_lk);
    if (p) { VKRED("too many memory pools (%d)", MAXPOOL); pool_free(p); return; }
    static int nlog; if (nlog++ < 32) VKLOG("memory pool %p+%#llx: device-local mirror%s", mem, (unsigned long long)size, g_vk.has.sparse_buffer ? ", backed in 2 MB granules on first use" : "");
}

// The NVN layer tells each pool's flags right after adding it (any backend; a pool this one does not mirror is ignored)
void vkm_pool_flags(const void *mem, uint32_t flags) {
    AcquireSRWLockShared(&g_lk);
    for (int i = 0; i < g_npool; i++) if (g_pool[i]->key == mem) { g_pool[i]->nvn_flags = flags; break; }
    ReleaseSRWLockShared(&g_lk);
}
static _Atomic uint64_t s_wb_noacc_b;   // write-back bytes into pools the CPU cannot access (measured first; NXRT_VK_WB_NOACCESS=0 skips them)
static int wb_skip(uint64_t addr, uint64_t len, int count) {   // 1: a GPU write into a CPU_NO_ACCESS pool, which the CPU never reads: no copy back
    AcquireSRWLockShared(&g_lk); pool_t_ *p = pool_find((uintptr_t)addr, len ? len : 1); int noacc = p && (p->nvn_flags & 1); ReleaseSRWLockShared(&g_lk);
    if (!noacc) return 0;
    if (count) atomic_fetch_add(&s_wb_noacc_b, len);
    static int keep = -1; if (keep < 0) keep = vkc_int("wb_noaccess", 1);
    return !keep;
}
uint64_t vkm_wb_noaccess_bytes(void) { return atomic_load(&s_wb_noacc_b); }
void vkm_pool_del(const void *mem) {
    pool_t_ *p = NULL;
    AcquireSRWLockExclusive(&g_lk);
    for (int i = 0; i < g_npool; i++) if (g_pool[i]->key == mem) { p = g_pool[i]; memmove(&g_pool[i], &g_pool[i + 1], (size_t)(g_npool - i - 1) * sizeof *g_pool); g_npool--; atomic_fetch_add(&g_pool_gen, 1); break; }
    ReleaseSRWLockExclusive(&g_lk);
    if (p) vk_retire(pool_free_cb, p);   // submissions recorded so far may still read it
}

// The window of p holding [addr, addr+len): buffer and offset, or 0 when the range spans two windows that do not overlap that far (logged once)
static int pool_window(pool_t_ *p, uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off) {
    uint64_t o = addr - p->base; int k = win_for(p, o, len ? len : 1);
    if (k < 0) {
        static _Atomic int said; if (!atomic_exchange(&said, 1))
            VKRED("a range %#llx+%#llx spans two of memory pool %p's buffers (the device's largest buffer is %llu MB): it reads zeros (later ones are not logged)",
                  (unsigned long long)addr, (unsigned long long)len, p->key, (unsigned long long)(g_max_buffer >> 20));
        return 0;
    }
    if (buf) *buf = p->buf[k];
    if (off) *off = o - p->win_lo[k];
    return 1;
}

int vkm_addr(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off) {
    AcquireSRWLockShared(&g_lk);
    pool_t_ *p = pool_find((uintptr_t)addr, len ? len : 1);
    int ok = p && pool_window(p, addr, len, buf, off);
    ReleaseSRWLockShared(&g_lk);
    return ok;
}

// Sparse backing for [o, o+n) of the buffer: unbound granules get memory and are bound before anything is recorded against them (a bind is a queue
// operation; it is waited for here, which happens once per 2 MB granule)
static int pool_bind(pool_t_ *p, uint64_t o, uint64_t n) {
    if (!p->sparse) {
        if (p->bound) return 0;
        static SRWLOCK lk = SRWLOCK_INIT; int rc = 0;
        AcquireSRWLockExclusive(&lk);
        if (!p->bound) for (int k = 0; k < p->nwin && !rc; k++) {
            VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, p->buf[k], &mr);
            if (vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &p->mem[k])) { VKRED("no device memory for memory pool %p+%#llx", p->key, (unsigned long long)p->size); rc = -1; }
            else vkBindBufferMemory(g_vk.dev, p->buf[k], p->mem[k].mem, p->mem[k].off);
        }
        if (!rc) p->bound = p->size;
        ReleaseSRWLockExclusive(&lk);
        return rc;
    }
    uint64_t g0 = o / g_gran, g1 = (o + n + g_gran - 1) / g_gran;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, p->buf[0], &mr);
    VkMemoryRequirements gr = {g_gran, mr.alignment, mr.memoryTypeBits};
    for (;;) {
        uint64_t fresh[32]; int nf = 0, rc = 0;   // granules backed in this round; each is bound into every window that covers it (two where windows overlap)
        for (uint64_t g = g0; g < g1 && g < p->ngran && nf < 32; g++) {
            if (p->gran[g].mem) continue;
            { static uint64_t failed; extern uint64_t now_ns(void); if (failed && now_ns() - failed < 100000000ull) { rc = -1; break; }   // not 18,000 times a second
              if (vk_alloc_mem(&gr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &p->gran[g])) { failed = now_ns(); VKRED("no device memory to back pool %p at %#llx", p->key, (unsigned long long)(g * g_gran)); rc = -1; break; } }
            fresh[nf++] = g;
            p->bound += g_gran;
        }
        if (!nf) return rc;
        VkSparseMemoryBind binds[64]; VkSparseBufferMemoryBindInfo bis[MAXWIN]; int nb = 0, nbi = 0;
        for (int k = 0; k < p->nwin; k++) {
            int first = nb;
            for (int i = 0; i < nf; i++) {
                uint64_t at = fresh[i] * g_gran;
                if (at >= p->win_lo[k] && at < p->win_lo[k] + p->win_n[k] && nb < 64)
                    binds[nb++] = (VkSparseMemoryBind){at - p->win_lo[k], g_gran, p->gran[fresh[i]].mem, p->gran[fresh[i]].off, 0};
            }
            if (nb > first) bis[nbi++] = (VkSparseBufferMemoryBindInfo){p->buf[k], (uint32_t)(nb - first), binds + first};
        }
        VkBindSparseInfo si = {VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, NULL, 0, NULL, (uint32_t)nbi, bis, 0, NULL, 0, NULL, 0, NULL};
        uint64_t tb = now_ns();
        VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence f; vkCreateFence(g_vk.dev, &fci, NULL, &f);
        AcquireSRWLockExclusive(&g_vk.qlock);
        VkResult r = vkQueueBindSparse(g_vk.q, 1, &si, f);
        ReleaseSRWLockExclusive(&g_vk.qlock);
        if (r == VK_SUCCESS) vkWaitForFences(g_vk.dev, 1, &f, VK_TRUE, UINT64_MAX); else { VKRED("vkQueueBindSparse failed (%d)", r); vk_lost("a sparse bind", r); rc = -1; }
        vkDestroyFence(g_vk.dev, f, NULL);
        atomic_fetch_add(&g_t_bind, now_ns() - tb); atomic_fetch_add(&g_n_bind, 1);
        if (rc) return rc;
    }
}

// Guest memory is read with a guard: a page that is not mapped any more (a pool the game freed without finalizing) reads as zeros instead of crashing
static int guest_copy(void *dst, const void *src, size_t n) {
    NVN_GUARDED(memset(dst, 0, n); return -1);
    memcpy(dst, src, n); NVN_GUARD_END(); return 0;
}

// A page copied before and written again: hot after HOT_STREAK writes in a row, with at most hot_gap frames between them: a double-buffered upload writes each buffer every other frame,
// and at 2560x1440 a movie's two took 1,470 write faults a frame without this
static void page_rewritten(pool_t_ *p, uint64_t pg) {
    if (p->last_dirty[pg] == (uint32_t)g_frame) return;
    static int gap = -1; if (gap < 0) gap = vkc_int("hot_gap", 2);
    p->streak[pg] = p->last_dirty[pg] + 1 + (uint32_t)gap >= (uint32_t)g_frame ? (uint8_t)(p->streak[pg] < 255 ? p->streak[pg] + 1 : 255) : 1;
    p->last_dirty[pg] = (uint32_t)g_frame;
    if (p->streak[pg] >= HOT_STREAK && !p->hot[pg]) { p->hot[pg] = 1; p->hot_n++; }
}
// Whether page pg's device copy is behind guest memory: never copied, hot, or written (a fault stamped it) since its copy; untracked pages always are
static inline int page_needs(pool_t_ *p, uint64_t pg) {
    if (atomic_load(&p->gpu_pending[pg])) return 0;   // a GPU write in flight: the device copy is ahead until its bytes are written back
    if (p->hot[pg]) return p->hot_at[pg] != atomic_load(&g_sub_epoch);
    if (!p->up[pg] || !p->span) return 1;
    return nvn_dirty_since_span(p->span, (const void *)(p->base + pg * PAGE), PAGE, p->up[pg]);
}
// A clean group of 64 pages at once: one generation scan from the group's oldest copy
static int group_clean(pool_t_ *p, uint64_t g0) {
    if (!p->span || g0 + 64 > p->npages) return 0;
    uint64_t lo = UINT64_MAX;
    for (uint64_t k = g0; k < g0 + 64; k++) { if (!p->up[k] || p->hot[k] || atomic_load(&p->gpu_pending[k])) return 0; if (p->up[k] < lo) lo = p->up[k]; }
    return !nvn_dirty_since_span(p->span, (const void *)(p->base + g0 * PAGE), 64 * PAGE, lo);
}

typedef struct { VkBufferCopy r[256]; int n; VkBuffer src, dst; } copy_batch;
static void batch_flush(copy_batch *b, VkCommandBuffer cb) { if (b->n) vkCmdCopyBuffer(cb, b->src, b->dst, (uint32_t)b->n, b->r); b->n = 0; }
// [off, off+n) of the pool from src at so: one copy per window it crosses (windows that overlap share their memory, so one of them is enough)
static void batch_put(copy_batch *b, pool_t_ *p, VkCommandBuffer cb, VkBuffer src, VkDeviceSize so, uint64_t off, uint64_t n) {
    while (n) {
        int k = win_of(p, off); uint64_t end = p->win_lo[k] + p->win_n[k], m = off + n <= end ? n : end - off;
        if (b->n == 256 || (b->n && (b->src != src || b->dst != p->buf[k]))) batch_flush(b, cb);
        b->src = src; b->dst = p->buf[k]; b->r[b->n++] = (VkBufferCopy){so, off - p->win_lo[k], m};
        so += m; off += m; n -= m;
    }
}
static void pool_fill_zero(pool_t_ *p, VkCommandBuffer cb, uint64_t off, uint64_t n) {
    while (n) { int k = win_of(p, off); uint64_t end = p->win_lo[k] + p->win_n[k], m = off + n <= end ? n : end - off; vkCmdFillBuffer(cb, p->buf[k], off - p->win_lo[k], m, 0); off += m; n -= m; }
}

// Copies the pages among [p0, p1) of pool p whose device copy is behind into cb, until budget bytes; *stop_at (optional) is where it stopped.
// Order per run (nxnative's): take the stamp, write-protect, then read guest memory: a write after the protection faults and is seen at the next check
static uint64_t pool_sync(pool_t_ *p, uint64_t p0, uint64_t p1, VkCommandBuffer cb, uint64_t budget, uint64_t *stop_at) {
    if (!p->span) p->span = nvn_dirty_span((const void *)p->lo, p->hi - p->lo);
    if (p1 > p->npages) p1 = p->npages;
    copy_batch b; b.n = 0; b.src = b.dst = VK_NULL_HANDLE;
    uint64_t queued = 0, t0 = now_ns(), pg = p0;
    while (pg < p1 && queued < budget) {
        if (!(pg & 63) && pg + 64 <= p1 && group_clean(p, pg)) { pg += 64; continue; }
        if (!page_needs(p, pg)) {
            if (atomic_load(&p->gpu_pending[pg]) && (p->hot[pg] ? p->hot_at[pg] != atomic_load(&g_sub_epoch) : p->span && p->up[pg] && nvn_dirty_since_span(p->span, (const void *)(p->base + pg * PAGE), PAGE, p->up[pg]))) {
                atomic_fetch_add(&s_wb_stale, 1);
                if (p->shadow && p->shadow[pg] && vkc_int("pending_merge", 1)) {   // the words the CPU changed go to the device, the GPU's stay (Eden's OnCachedWrite, by comparison instead of a trap)
                    uint64_t stamp = nvn_dirty_now();
                    if (!p->hot[pg] && p->span) nvn_dirty_protect_span(p->span, (const void *)(p->base + pg * PAGE), PAGE);
                    uint32_t now[PAGE / 4], *s = p->shadow[pg]; guest_copy(now, (const void *)(p->base + pg * PAGE), PAGE);
                    for (uint32_t i = 0; i < PAGE / 4; ) {
                        if (now[i] == s[i]) { i++; continue; }
                        uint32_t e = i; while (e < PAGE / 4 && now[e] != s[e]) e++;
                        VkBuffer sb; VkDeviceSize so; uint8_t *dst = vkm_staging((uint64_t)(e - i) * 4, 16, &sb, &so);
                        if (!dst) break;
                        memcpy(dst, now + i, (size_t)(e - i) * 4); memcpy(s + i, now + i, (size_t)(e - i) * 4);
                        batch_put(&b, p, cb, sb, so, pg * PAGE + 4ull * i, (uint64_t)(e - i) * 4);
                        atomic_fetch_add(&s_wb_merged, e - i); i = e;
                    }
                    p->up[pg] = stamp; p->hot_at[pg] = atomic_load(&g_sub_epoch);
                }
            }
            pg++; continue; }
        uint64_t run0 = pg;
        while (pg < p1 && pg - run0 < RUN_MAX && page_needs(p, pg)) { if (p->up[pg] && !p->hot[pg]) page_rewritten(p, pg); pg++; }
        // Pages the game never touched are not committed (the heap commits lazily): nothing to read or protect there. A use reads them as zeros and
        // keeps asking (they stay "never copied"); the background sync steps over the whole uncommitted region. VirtualQuery walks the address space's
        // regions, which the tracker's protections split finely: in gameplay it was 70 % of the render thread's time. Pages seen committed are remembered
        // (a pool's memory stays committed while the pool exists) and only a run that starts on a page not known yet is asked about
        uint64_t known = run0; while (known < pg && p->committed[known]) known++;
        MEMORY_BASIC_INFORMATION mi;
        if (known > run0) pg = known;   // the known part now, the rest in the next run
        else if (VirtualQuery((const void *)(p->base + run0 * PAGE), &mi, sizeof mi)) {
            uint64_t rend = ((uintptr_t)mi.BaseAddress + mi.RegionSize - p->base) / PAGE;
            int readable = mi.State == MEM_COMMIT && !(mi.Protect & (PAGE_NOACCESS | PAGE_GUARD));
            if (rend < pg) pg = rend > run0 ? rend : run0 + 1;
            if (!readable) {
                if (budget == UINT64_MAX) { if (pool_bind(p, run0 * PAGE, (pg - run0) * PAGE)) break; batch_flush(&b, cb); pool_fill_zero(p, cb, run0 * PAGE, (pg - run0) * PAGE); }
                else if (rend > pg) pg = rend < p1 ? rend : p1;
                atomic_fetch_add(&s_uncommitted, pg - run0);
                continue;
            }
            memset(p->committed + run0, 1, (size_t)(pg - run0));
        }
        uint64_t off = run0 * PAGE, n = (pg - run0) * PAGE;
        if (pool_bind(p, off, n)) break;
        p->up_b += n; for (uint64_t k = run0; k < pg; k++) { if (!p->up[k]) p->up_never += PAGE; else if (p->hot[k]) p->up_hot += PAGE; }
        uint64_t stamp = nvn_dirty_now();
        for (uint64_t q0 = run0; q0 < pg; ) {   // protect the pages that are not hot
            if (p->hot[q0]) { q0++; continue; }
            uint64_t q1 = q0; while (q1 < pg && !p->hot[q1]) q1++;
            if (p->span) nvn_dirty_protect_span(p->span, (const void *)(p->base + q0 * PAGE), (size_t)((q1 - q0) * PAGE));
            q0 = q1;
        }
        VkBuffer sb; VkDeviceSize so;
        uint8_t *dst = vkm_staging(n, 16, &sb, &so);
        if (!dst) { pg = run0; break; }
        guest_copy(dst, (const void *)(p->base + off), (size_t)n);
        { uint32_t ep = atomic_load(&g_sub_epoch); for (uint64_t k = run0; k < pg; k++) { p->up[k] = stamp; p->hot_at[k] = ep; } }
        batch_put(&b, p, cb, sb, so, off, n);
        queued += n;
        atomic_fetch_add(&s_up_pages, pg - run0); atomic_fetch_add(&s_up_runs, 1);
    }
    batch_flush(&b, cb);
    if (stop_at) *stop_at = pg;
    atomic_fetch_add(&s_up_bytes, queued); atomic_fetch_add(&s_up_ns, now_ns() - t0);
    return queued;
}

// A range synced in this submission, with no page written since, needs no second look: the same few buffers are bound by thousands of draws (5,000 descriptor sets a frame), and looking took 11 ms a frame
// Only the recording thread gets this far (vks_upload_cb is NULL elsewhere), so the table is that thread's alone. A hit also gives the buffer and offset, so a range
// used again (thousands of descriptor sets bind the same few) skips the pool lookup and its lock: gamectl set use_fast 0 (NXRT_VK_USE_FAST=0) looks them up every time
static struct { uint64_t addr, len, gen; uint32_t ep, pgen; VkBuffer buf; VkDeviceSize off; pool_t_ *pool; } g_seen[8192];   // pool: whose pages to look at when guest memory has changed since
static int g_use_fast = -1;
static uint64_t s_use_hit, s_use_calls, s_use_synced;   // the recording thread's vkm_use calls, fast-path hits, and those that ran pool_sync
void vkm_use_counts(uint64_t *calls, uint64_t *hits, uint64_t *synced) { *calls = s_use_calls; *hits = s_use_hit; *synced = s_use_synced; }
uint32_t vkm_pool_gen(void) { return atomic_load(&g_pool_gen); }
uint64_t vkm_up_ns(void) { return atomic_load(&s_up_ns); }
static uint64_t s_use_cyc; static int g_use_ph = -1;   // NXRT_VK_PHASES: cycles inside vkm_use (the replay thread's only)
uint64_t vkm_use_cycles(void) { return s_use_cyc; }
static int vkm_use_(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off);
int vkm_use(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off) {
    if (g_use_ph < 0) g_use_ph = vkc_int("phases", 0);
    if (!g_use_ph) return vkm_use_(addr, len, buf, off);
    uint64_t t0 = __builtin_ia32_rdtsc(); int r = vkm_use_(addr, len, buf, off); if (vks_owner() >= 0) s_use_cyc += __builtin_ia32_rdtsc() - t0; return r;
}
static int vkm_use_(uint64_t addr, uint64_t len, VkBuffer *buf, VkDeviceSize *off) {
    if (g_use_fast < 0) g_use_fast = vkc_int("use_fast", 1);
    s_use_calls++;
    unsigned h = (unsigned)(((addr >> 4) * 0x9E3779B1u ^ len * 0x85EBCA6Bu) >> 7) & 8191;
    if (g_use_fast && g_seen_on && g_seen[h].addr == addr && g_seen[h].len == len && g_seen[h].pgen == atomic_load(&g_pool_gen) && g_seen[h].buf) {
        uint64_t now = nvn_dirty_now(); uint32_t ep = atomic_load(&g_sub_epoch);
        // the same range synced before: still current if nothing was written anywhere since (in this submission), or if none of its own pages needs a copy: page_needs
        // looks at each page's own stamp, and a hot page needs one copy per submission. Game threads write all the time, and there are 6 submissions a frame: the
        // whole-memory test alone missed most uses. NXRT_VK_USE_FAST=2: only within a submission (as at first)
        int ok = g_seen[h].ep == ep && g_seen[h].gen == now;
        if (!ok && g_seen[h].ep != ep && g_use_fast == 2) g_seen[h].pool = NULL;
        if (!ok && g_seen[h].pool) {   // what pool_sync would look at: each page against the stamp of its own copy (pgen above: the pool is still there)
            pool_t_ *p = g_seen[h].pool; uint64_t p0 = (addr - p->base) / PAGE, p1 = (addr + (len ? len : 1) - p->base + PAGE - 1) / PAGE;
            ok = p1 <= p->npages; for (uint64_t pg = p0; pg < p1 && ok; pg++) ok = !page_needs(p, pg);
        }
        if (ok && vks_upload_cb()) {
            s_use_hit++;
            g_seen[h].gen = now; g_seen[h].ep = ep;
            if (buf) *buf = g_seen[h].buf;
            if (off) *off = g_seen[h].off;
            return 1;
        }
    }
    VkBuffer b_; VkDeviceSize o_; if (!buf) buf = &b_; if (!off) off = &o_;
    uint32_t pgen = atomic_load(&g_pool_gen);
    AcquireSRWLockShared(&g_lk);
    pool_t_ *p = pool_find((uintptr_t)addr, len ? len : 1);
    int ok = p && pool_window(p, addr, len, buf, off);
    ReleaseSRWLockShared(&g_lk);
    if (!ok) return 0;
    VkCommandBuffer cb = vks_upload_cb();
    if (!cb) return 1;
    uint32_t ep = atomic_load(&g_sub_epoch); uint64_t gen = nvn_dirty_now();
    if (g_seen_on && g_seen[h].addr == addr && g_seen[h].len == len && g_seen[h].ep == ep && g_seen[h].gen == gen) { g_seen[h].pgen = pgen; g_seen[h].buf = *buf; g_seen[h].off = *off; g_seen[h].pool = p; return 1; }
    atomic_fetch_add(&s_sync, 1); s_use_synced++;
    pool_sync(p, (addr - p->base) / PAGE, (addr + (len ? len : 1) - p->base + PAGE - 1) / PAGE, cb, UINT64_MAX, NULL);
    g_seen[h].addr = addr; g_seen[h].len = len; g_seen[h].ep = ep; g_seen[h].gen = nvn_dirty_now(); g_seen[h].pgen = pgen; g_seen[h].buf = *buf; g_seen[h].off = *off;
    g_seen[h].pool = p;
    return 1;
}

// Every pool's pages behind guest memory, round robin from where the last call stopped, within the frame's upload budget and a scan limit
static uint64_t g_scan_pages = 262144;   // NXRT_VK_SCAN_PAGES: pages looked at per call (1 GB)
static int g_rr;
uint64_t vkm_sync_all(void) {
    VkCommandBuffer cb = vks_upload_cb();
    if (!cb) return 0;
    uint64_t total = 0, scanned = 0;
    AcquireSRWLockShared(&g_lk);
    for (int k = 0; k < g_npool && vkm_budget_left() && scanned < g_scan_pages; k++) {
        int i = (g_rr + k) % g_npool;
        pool_t_ *p = g_pool[i];
        uint64_t from = p->cursor < p->npages ? p->cursor : 0, to = from + (g_scan_pages - scanned), stop = from;
        uint64_t n = pool_sync(p, from, to, cb, vkm_budget_left(), &stop);
        vkm_budget_take(n); total += n; scanned += stop - from;
        p->cursor = stop >= p->npages ? 0 : stop;
        if (p->cursor) { g_rr = i; break; }   // continue in this pool next time
        g_rr = (i + 1) % g_npool;
    }
    ReleaseSRWLockShared(&g_lk);
    atomic_fetch_add(&s_sync, 1);
    return total;
}

// GPU writes in flight over [a, a+len): +1 when recorded, -1 when their bytes have reached guest memory. The pages are copied again from guest memory after
// that (the write back stamps them), which also brings in anything the CPU wrote beside the GPU's bytes
static void gpu_pending(uint64_t a, uint64_t len, int d) {
    AcquireSRWLockShared(&g_lk);
    pool_t_ *p = pool_find((uintptr_t)a, len);
    if (p) for (uint64_t pg = (a - p->base) / PAGE; pg <= (a + len - 1 - p->base) / PAGE && pg < p->npages; pg++) {
        if (d > 0 && !atomic_load(&p->gpu_pending[pg])) {   // the first one: the use that came before it has just made the device current
            if (!p->shadow) p->shadow = calloc(p->npages, sizeof *p->shadow);
            if (p->shadow && !p->shadow[pg]) p->shadow[pg] = malloc(PAGE);
            if (p->shadow && p->shadow[pg]) guest_copy(p->shadow[pg], (const void *)(p->base + pg * PAGE), PAGE);
        }
        atomic_fetch_add(&p->gpu_pending[pg], (uint8_t)d);
    }
    ReleaseSRWLockShared(&g_lk);
}

// ---------- GPU writes back to guest memory ----------
typedef struct { uint64_t addr, len; VkBuffer buf; vk_alloc mem; const uint8_t *cpu; uint32_t fill; int kind; uint64_t base_at, after_at, ring_end; } wb_t;   // kind 0 readback, 1 known bytes, 2 fill, 3 the words a dispatch changed
// A dispatch's written range, before and after, in a ring of cached memory (kind 3). A storage buffer is written in part: the shader's bytes are the words that differ, and only they reach guest
// memory, so what the CPU wrote in the same range while the GPU worked stays (nxnative's unified memory has no such copy; Eden's buffer cache keeps the same two apart by the byte)
enum { WR_SIZE = 256u << 20 };
static VkBuffer g_wr; static vk_alloc g_wr_mem; static uint64_t g_wr_head; static _Atomic uint64_t g_wr_tail; static int g_wr_state;
static int wr_init(void) {
    if (g_wr_state) return g_wr_state > 0;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, WR_SIZE, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    g_wr_state = -1;
    if (vkCreateBuffer(g_vk.dev, &bci, NULL, &g_wr)) return 0;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, g_wr, &mr);
    if (vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0, &g_wr_mem) &&
        vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, &g_wr_mem)) return 0;
    vkBindBufferMemory(g_vk.dev, g_wr, g_wr_mem.mem, g_wr_mem.off);
    g_wr_state = 1; return 1;
}
// Room for n bytes: *at its offset in the ring. Monotonic head and tail; a piece does not wrap
static int wr_take(uint64_t n, uint64_t *at) {
    if (!wr_init() || n > WR_SIZE / 4) return 0;
    uint64_t h = g_wr_head, o = h % WR_SIZE;
    if (o + n > WR_SIZE) h += WR_SIZE - o, o = 0;
    if (h + n - atomic_load(&g_wr_tail) > WR_SIZE) return 0;
    g_wr_head = h + n; *at = o; return 1;
}
static uint64_t g_wr_end(void) { return g_wr_head; }
// The words that differ between base and after reach guest memory, page by page; a page the shader left alone is not touched (a touch sends it to the device again)
static void wr_land(wb_t *w) {
    const uint32_t *b = (const uint32_t *)((const uint8_t *)g_wr_mem.map + w->base_at), *a = (const uint32_t *)((const uint8_t *)g_wr_mem.map + w->after_at);
    uint32_t *g = (uint32_t *)(uintptr_t)w->addr; uint64_t n = w->len / 4, changed = 0, kept = 0;
    for (uint64_t i = 0; i < n; ) {
        uint64_t e = ((((uintptr_t)(g + i)) | (PAGE - 1)) + 1 - (uintptr_t)g) / 4; if (e > n) e = n;   // to the end of the guest page
        if (!memcmp(a + i, b + i, (size_t)(e - i) * 4)) { i = e; continue; }
        dm_guest_write_prepare(g + i, (size_t)(e - i) * 4);
        for (; i < e; i++) if (a[i] != b[i]) { g[i] = a[i]; changed++; } else if (g[i] != a[i]) kept++;
    }
    atomic_fetch_add(&s_wb_words, changed); atomic_fetch_add(&s_wb_kept, kept);
    AcquireSRWLockShared(&g_lk);
    { pool_t_ *p = pool_find((uintptr_t)w->addr, w->len);
      if (p && p->shadow) for (uint64_t i = 0; i < n; i++) { uint64_t off = w->addr + 4 * i - p->base; uint32_t *s = p->shadow[off / PAGE];
          if (s) { if (a[i] != b[i]) s[(off % PAGE) / 4] = a[i]; } else i += (PAGE - off % PAGE) / 4 - 1; } }
    ReleaseSRWLockShared(&g_lk);
    atomic_store(&g_wr_tail, w->ring_end);
    if (w->len & 3) { uint8_t *d = (uint8_t *)(uintptr_t)w->addr; const uint8_t *ab = (const uint8_t *)a, *bb = (const uint8_t *)b; dm_guest_write_prepare(d + (w->len & ~3ull), 3); for (uint64_t i = w->len & ~3ull; i < w->len; i++) if (ab[i] != bb[i]) d[i] = ab[i]; }
}
// NXRT_VK_WBLOG=<bytes>: every write-back of at least that many bytes in the log, when recorded (rec) and when it lands
// (land: bytes of guest memory it changed, only when some changed; rec only with NXRT_VK_WBLOG_REC=1), up to 100000 lines. To see what the GPU's copies back put into guest memory
static int wblog_min(void) { static int v = -1; if (v < 0) v = vkc_int("wblog", 0); return v; }
static void wblog(const char *what, uint64_t addr, uint64_t len, int kind, uint64_t changed) {
    static volatile LONG n; if (!wblog_min() || len < (uint64_t)wblog_min() || (what[0] == 'r' && !vkc_int("wblog_rec", 0)) || (what[0] == 'l' && !changed) || InterlockedIncrement(&n) > 100000) return;
    VKLOG("wblog: %s frame %llu addr %#llx len %#llx kind %d changed %llu", what, (unsigned long long)g_frame, (unsigned long long)addr, (unsigned long long)len, kind, (unsigned long long)changed); }
static void wb_land(void *arg) {
    wb_t *w = arg;
    if (wblog_min() && w->kind != 3) { uint64_t ch = 0; const uint8_t *src = w->kind == 4 ? (const uint8_t *)g_wr_mem.map + w->after_at : w->kind == 0 ? (const uint8_t *)w->mem.map : w->kind == 1 ? (const uint8_t *)w->cpu : NULL;
        if (src) { const uint8_t *g = (const uint8_t *)(uintptr_t)w->addr; for (uint64_t i = 0; i < w->len; i++) ch += src[i] != g[i]; } wblog("land", w->addr, w->len, w->kind, ch); }
    if (w->kind == 3) { wr_land(w); gpu_pending(w->addr, w->len, -1); atomic_fetch_add(&s_wb_bytes, w->len); free(w); return; }
    if (w->kind == 4) {   // the whole range, page by page: a page that came back unchanged is not written, so it is not sent to the device again
        const uint8_t *a = (const uint8_t *)g_wr_mem.map + w->after_at; uint8_t *g = (uint8_t *)(uintptr_t)w->addr;
        for (uint64_t i = 0; i < w->len; ) { uint64_t e = ((((uintptr_t)(g + i)) | (PAGE - 1)) + 1 - (uintptr_t)g); if (e > w->len) e = w->len;
            if (memcmp(a + i, g + i, (size_t)(e - i))) { dm_guest_write_prepare(g + i, (size_t)(e - i)); memcpy(g + i, a + i, (size_t)(e - i)); }
            i = e; }
        atomic_store(&g_wr_tail, w->ring_end); gpu_pending(w->addr, w->len, -1); atomic_fetch_add(&s_wb_bytes, w->len); free(w); return; }
    dm_guest_write_prepare((void *)(uintptr_t)w->addr, (size_t)w->len);
    uint8_t *d = (uint8_t *)(uintptr_t)w->addr;
    if (w->kind == 0) memcpy(d, w->mem.map, (size_t)w->len);
    else if (w->kind == 1) memcpy(d, w->cpu, (size_t)w->len);
    else for (uint64_t i = 0; i < w->len; i++) d[i] = (uint8_t)(w->fill >> 8 * (i & 3));
    gpu_pending(w->addr, w->len, -1);
    atomic_fetch_add(&s_wb_bytes, w->len);
    if (w->buf) { vkDestroyBuffer(g_vk.dev, w->buf, NULL); vk_free_mem(&w->mem); }
    free((void *)w->cpu); free(w);
}
// NXRT_VK_WB_OFF=1 (TEST ONLY, wrong results possible): no GPU write goes back to guest memory. Measures the most that fewer write-backs could gain
static int wb_off(void) { static int v = -1; if (v < 0) v = vkc_int("wb_off", 0); return v; }
void vkm_gpu_wrote(uint64_t addr, uint64_t len) {
    if (wb_off()) return;
    wblog("rec", addr, len, 4, 0);
    if (wb_skip(addr, len, 1)) return;
    VkBuffer pb; VkDeviceSize po;
    VkCommandBuffer cb = vks_cb();
    if (!cb || !len || !vkm_addr(addr, len, &pb, &po)) return;
    { uint64_t at;   // through the ring: a buffer and an allocation for each of 400 write-backs a frame cost more than the copies
      if (vkc_int("wb_ring", 1) && wr_take((len + 15) & ~15ull, &at)) {
          wb_t *r = calloc(1, sizeof *r); r->addr = addr; r->len = len; r->kind = 4; r->after_at = at; r->ring_end = g_wr_end();
          vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
          VkBufferCopy c = {po, at, len}; vkCmdCopyBuffer(cb, pb, g_wr, 1, &c);
          vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
          gpu_pending(addr, len, 1); vks_post_fn(wb_land, r); return; } }
    wb_t *w = calloc(1, sizeof *w); w->addr = addr; w->len = len;
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, len, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    vkCreateBuffer(g_vk.dev, &bci, NULL, &w->buf);
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, w->buf, &mr);
    if (vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0, &w->mem) &&
        vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, &w->mem)) { vkDestroyBuffer(g_vk.dev, w->buf, NULL); free(w); return; }
    vkBindBufferMemory(g_vk.dev, w->buf, w->mem.mem, w->mem.off);
    vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferCopy c = {po, 0, len};
    vkCmdCopyBuffer(cb, pb, w->buf, 1, &c);
    vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    gpu_pending(addr, len, 1);
    vks_post_fn(wb_land, w);
}
// A dispatch's range before the shader runs: returns the ring offset + 1, 0 when it is not kept (vkm_gpu_wrote_base falls back to the whole range)
uint64_t vkm_wb_base(uint64_t addr, uint64_t len) {
    VkBuffer pb; VkDeviceSize po; uint64_t at;
    VkCommandBuffer cb = vks_cb();
    if (wb_off() || wb_skip(addr, len, 0)) return 0;
    if (vkc_int("wb_whole", 0) || !cb || !len || !vkm_addr(addr, len, &pb, &po) || !wr_take((len + 15) & ~15ull, &at)) return 0;
    VkBufferCopy c = {po, at, len}; vkCmdCopyBuffer(cb, pb, g_wr, 1, &c);
    return at + 1;
}
void vkm_gpu_wrote_base(uint64_t addr, uint64_t len, uint64_t base) {
    if (wb_off()) return;
    wblog("rec", addr, len, 3, 0);
    if (wb_skip(addr, len, 1)) return;
    VkBuffer pb; VkDeviceSize po; uint64_t at;
    VkCommandBuffer cb = vks_cb();
    if (!base || !cb || !vkm_addr(addr, len, &pb, &po) || !wr_take((len + 15) & ~15ull, &at)) { vkm_gpu_wrote(addr, len); return; }
    wb_t *w = calloc(1, sizeof *w); w->addr = addr; w->len = len; w->kind = 3; w->base_at = base - 1; w->after_at = at; w->ring_end = g_wr_end();
    vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferCopy c = {po, at, len}; vkCmdCopyBuffer(cb, pb, g_wr, 1, &c);
    vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    gpu_pending(addr, len, 1);
    vks_post_fn(wb_land, w);
}
// Writes whose bytes the CPU knows (inline uniform updates, fills): the GPU writes the mirror in order, guest memory gets the same bytes at completion
void vkm_cpu_also(uint64_t addr, uint64_t len, const void *data, int fill, uint32_t value) {
    wb_t *w = calloc(1, sizeof *w); w->addr = addr; w->len = len; w->kind = fill ? 2 : 1; w->fill = value;
    if (!fill) { uint8_t *c = malloc((size_t)len); memcpy(c, data, (size_t)len); w->cpu = c; }
    gpu_pending(addr, len, 1);
    vks_post_fn(wb_land, w);
}

// ---------- Frames, statistics ----------
void vkm_stats(vkm_stats_t *s) {
    memset(s, 0, sizeof *s);
    AcquireSRWLockShared(&g_lk);
    for (int i = 0; i < g_npool; i++) { s->pools++; s->pool_bytes += g_pool[i]->hi - g_pool[i]->lo; s->bound_bytes += g_pool[i]->bound; s->sparse_pools += g_pool[i]->sparse; s->hot_pages += g_pool[i]->hot_n; }
    ReleaseSRWLockShared(&g_lk);
    s->up_bytes = atomic_load(&s_up_bytes); s->up_pages = atomic_load(&s_up_pages); s->up_runs = atomic_load(&s_up_runs); s->up_ns = atomic_load(&s_up_ns);
    s->sync_calls = atomic_load(&s_sync); s->staging_bytes = atomic_load(&s_st_bytes); s->staging_overflow = atomic_load(&s_st_ovf); s->staging_waits = atomic_load(&s_st_waits);
    s->writeback_bytes = atomic_load(&s_wb_bytes); s->wb_words = atomic_load(&s_wb_words); s->wb_kept = atomic_load(&s_wb_kept); s->wb_stale = atomic_load(&s_wb_stale); s->wb_merged = atomic_load(&s_wb_merged); s->uncommitted_pages = atomic_load(&s_uncommitted);
    nvn_dirty_stats(&s->faults, &s->preps);
    uint64_t mc, mpg; nvn_dirty_mp_stats(&mc, &mpg, &s->fault_ns);   // the tracker's protection changes (faults and protects)
}

void vkm_frame_tick(uint64_t frame) {
    static vkm_stats_t prev; static uint64_t prev_frame, t_prev;
    g_frame = frame + 1; g_budget_used = 0; g_seen_on = vkc_int("use_seen", 1);
    if (frame % HOT_COOL_FRAMES == 0) {   // hot pages cool down: protected again at their next copy; written again, they come back within HOT_STREAK frames
        AcquireSRWLockShared(&g_lk);
        for (int i = 0; i < g_npool; i++) { pool_t_ *p = g_pool[i]; for (uint64_t k = 0; k < p->npages; k++) if (p->hot[k]) { p->hot[k] = 0; p->up[k] = 0; p->streak[k] = 0; } p->hot_n = 0; }   // copied (and protected) again at the next sync
        ReleaseSRWLockShared(&g_lk);
    }
    if (frame % 60 || frame == prev_frame) return;
    vkm_stats_t s; vkm_stats(&s);
    uint64_t df = frame - prev_frame, now = now_ns(); double secs = t_prev ? (now - t_prev) / 1e9 : 0;
    uint64_t budget = 0, usage = 0; vk_budget(&budget, &usage);
    VKLOG("memory up to frame %llu: %llu pools %.1f MB (backed %.1f MB, %llu sparse); per frame: uploaded %.1f KB in %.1f runs (%.3f ms CPU), %.1f write faults (protection changes %.3f ms), "
          "%.1f host pre-writes, written back %.1f KB (%.0f words changed, %.0f kept for the CPU, %.1f uses while one was on its way, %.0f words sent for them), %.0f MB of uncommitted pool stepped over; hot pages %llu; staging overflow %llu waits %llu; video memory %.0f / %.0f MB%s",
          (unsigned long long)frame, (unsigned long long)s.pools, s.pool_bytes / 1048576.0, s.bound_bytes / 1048576.0, (unsigned long long)s.sparse_pools,
          (s.up_bytes - prev.up_bytes) / 1024.0 / df, (double)(s.up_runs - prev.up_runs) / df, (s.up_ns - prev.up_ns) / 1e6 / df,
          (double)(s.faults - prev.faults) / df, (s.fault_ns - prev.fault_ns) / 1e6 / df, (double)(s.preps - prev.preps) / df,
          (s.writeback_bytes - prev.writeback_bytes) / 1024.0 / df, (double)(s.wb_words - prev.wb_words) / df, (double)(s.wb_kept - prev.wb_kept) / df, (double)(s.wb_stale - prev.wb_stale) / df, (double)(s.wb_merged - prev.wb_merged) / df, (s.uncommitted_pages - prev.uncommitted_pages) * 4096.0 / 1048576.0 / df, (unsigned long long)s.hot_pages, (unsigned long long)s.staging_overflow,
          (unsigned long long)s.staging_waits, usage / 1048576.0, budget / 1048576.0, secs > 0 ? "" : "");
    prev = s; prev_frame = frame; t_prev = now;
    { pool_t_ *top[4] = {0}; AcquireSRWLockShared(&g_lk);
      for (int i = 0; i < g_npool; i++) { pool_t_ *q = g_pool[i]; for (int k = 0; k < 4; k++) if (!top[k] || q->up_b > top[k]->up_b) { for (int j = 3; j > k; j--) top[j] = top[j - 1]; top[k] = q; break; } }
      for (int k = 0; k < 4; k++) if (top[k] && top[k]->up_b >> 20) VKLOG("  pool %#llx+%llu MB: %llu MB uploaded since the last line (never copied %llu, hot %llu), span %d, %llu hot pages", (unsigned long long)top[k]->lo, (unsigned long long)(top[k]->size >> 20),
          (unsigned long long)(top[k]->up_b >> 20), (unsigned long long)(top[k]->up_never >> 20), (unsigned long long)(top[k]->up_hot >> 20), top[k]->span != NULL, (unsigned long long)top[k]->hot_n);
      for (int i = 0; i < g_npool; i++) g_pool[i]->up_b = g_pool[i]->up_never = g_pool[i]->up_hot = 0;
      ReleaseSRWLockShared(&g_lk); }
}

int vkm_init(void) {
    InitializeSRWLock(&g_st.lk);
    const char *e = getenv("NXRT_VK_UPLOAD_MB"); if (e && atoi(e) > 0) g_budget = (uint64_t)atoi(e) << 20;
    e = getenv("NXRT_VK_SCAN_PAGES"); if (e && atoi(e) > 0) g_scan_pages = (uint64_t)atoi(e);
    VkDeviceSize ring = g_budget < (16u << 20) ? (16u << 20) : g_budget > (256u << 20) ? (256u << 20) : g_budget;
    e = getenv("NXRT_VK_STAGING_MB"); if (e && atoi(e) > 0) ring = (VkDeviceSize)atoi(e) << 20;
    g_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
              VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
    if (g_vk.f12.bufferDeviceAddress) g_usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    if (g_vk.has.xfb) g_usage |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT | VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT;
    if (g_vk.has.cond_render) g_usage |= VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT;
    VkPhysicalDeviceMaintenance4Properties m4 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m4};
    vkGetPhysicalDeviceProperties2(g_vk.pd, &p2);
    g_max_buffer = m4.maxBufferSize;
    e = getenv("NXRT_VK_MAX_BUFFER_MB");   // test switch: pretend the device's largest buffer is smaller (AMD's 2048, to try its pool windows on another card)
    if (e && atoi(e) > 0 && ((VkDeviceSize)atoi(e) << 20) < g_max_buffer) g_max_buffer = (VkDeviceSize)atoi(e) << 20;
    g_alias = g_vk.has.sparse_aliased && !(getenv("NXRT_VK_NO_ALIAS") && atoi(getenv("NXRT_VK_NO_ALIAS")));
    if (g_vk.has.sparse_buffer) {   // the granule is a whole number of the device's sparse blocks
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT, GRAN, g_usage, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
        VkBuffer b; VkMemoryRequirements mr;
        if (vkCreateBuffer(g_vk.dev, &bci, NULL, &b) == VK_SUCCESS) { vkGetBufferMemoryRequirements(g_vk.dev, b, &mr); vkDestroyBuffer(g_vk.dev, b, NULL);
            if (mr.alignment > g_gran) g_gran = mr.alignment; else g_gran = (GRAN + mr.alignment - 1) / mr.alignment * mr.alignment; }
    }
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, ring, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    if (vkCreateBuffer(g_vk.dev, &bci, NULL, &g_st.buf) != VK_SUCCESS) return -1;
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, g_st.buf, &mr);
    if (vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &g_st.mem)) { VKRED("no host memory for the staging ring"); return -1; }
    vkBindBufferMemory(g_vk.dev, g_st.buf, g_st.mem.mem, g_st.mem.off);
    g_st.map = g_st.mem.map; g_st.size = ring;
    vk_name((uint64_t)g_st.buf, VK_OBJECT_TYPE_BUFFER, "staging ring");
    VKLOG("memory: staging ring %llu MB, upload budget %llu MB a frame, pools %s (granule %llu KB), largest buffer %.0f GB",
          (unsigned long long)(ring >> 20), (unsigned long long)(g_budget >> 20), g_vk.has.sparse_buffer ? "sparse" : "fully backed", (unsigned long long)(g_gran >> 10), g_max_buffer / 1073741824.0);
    return 0;
}

void vkm_shutdown(void) {
    AcquireSRWLockExclusive(&g_lk);
    for (int i = 0; i < g_npool; i++) pool_free(g_pool[i]);
    g_npool = 0;
    ReleaseSRWLockExclusive(&g_lk);
    AcquireSRWLockExclusive(&g_st.lk);
    for (unsigned k = g_st.qt; k != g_st.qh; k++) { unsigned i = k % 4096; if (g_st.q[i].ovf) { vkDestroyBuffer(g_vk.dev, g_st.q[i].ovf, NULL); vk_free_mem(&g_st.q[i].ovf_mem); } }
    g_st.qt = g_st.qh = 0;
    if (g_st.buf) { vkDestroyBuffer(g_vk.dev, g_st.buf, NULL); vk_free_mem(&g_st.mem); g_st.buf = VK_NULL_HANDLE; }
    ReleaseSRWLockExclusive(&g_st.lk);
}
