// The flight recorder (vk_flight.h). Everything goes to rings that cost a blit and a copy per frame on the GPU and a line of memory per draw on the CPU; nothing is read until a recording is asked for.
#include "vk_flight.h"
#include "vk_mem.h"
#include "vk_submit.h"
#include <math.h>
#include <process.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *nxrt_log_path(void);

enum { NIMG = 120, IW = 1280, IH = 720, NROW = 1 << 20, NARG = 1 << 18, NFRM = 512, NPEND = 8192, ARG_B = 24 };
typedef struct { uint64_t ind, cnt; uint32_t i, len; } pend_t;
static struct {
    int state;   // 0 not started, 1 on, -1 off
    VkImage img; vk_alloc img_mem; VkFormat fmt; int fresh; VkBuffer buf; vk_alloc buf_mem; VkBuffer abuf; vk_alloc amem;
    vkf_row *rows; uint64_t row_head, row_open, arg_head;
    pend_t pend[NPEND]; int npend;
    struct { uint64_t frame, first, n, t; } fr[NFRM]; uint64_t fr_head;
    struct { uint64_t frame, t; } im[NIMG]; uint64_t im_head;
    _Atomic int frozen, want; char label[64]; char last[1024]; _Atomic uint64_t serial;
} F;

uint64_t vkf_key(const char *k) { return k ? strtoull(k, NULL, 16) : 0; }
int vkf_wants(void) { return !atomic_load(&F.frozen) && (F.state >= 0) && vkc_int("flight", 1); }

static int start(VkFormat fmt) {
    if (F.state) return F.state > 0;
    F.state = -1;
    if (!vkc_int("flight", 1)) return 0;
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, fmt, {IW, IH, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if (vkCreateImage(g_vk.dev, &ii, NULL, &F.img)) return 0;
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vk.dev, F.img, &mr);
    if (vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &F.img_mem)) return 0;
    vkBindImageMemory(g_vk.dev, F.img, F.img_mem.mem, F.img_mem.off);
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (VkDeviceSize)NIMG * IW * IH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    if (vkCreateBuffer(g_vk.dev, &bci, NULL, &F.buf)) return 0;
    vkGetBufferMemoryRequirements(g_vk.dev, F.buf, &mr);
    VkMemoryPropertyFlags hc = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (vk_alloc_mem(&mr, hc | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0, &F.buf_mem) && vk_alloc_mem(&mr, hc, 0, &F.buf_mem)) return 0;
    vkBindBufferMemory(g_vk.dev, F.buf, F.buf_mem.mem, F.buf_mem.off);
    bci.size = (VkDeviceSize)NARG * ARG_B;
    if (vkCreateBuffer(g_vk.dev, &bci, NULL, &F.abuf)) return 0;
    vkGetBufferMemoryRequirements(g_vk.dev, F.abuf, &mr);
    if (vk_alloc_mem(&mr, hc | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0, &F.amem) && vk_alloc_mem(&mr, hc, 0, &F.amem)) return 0;
    vkBindBufferMemory(g_vk.dev, F.abuf, F.amem.mem, F.amem.off);
    F.rows = malloc(sizeof(vkf_row) * NROW);
    if (!F.rows) return 0;
    F.fmt = fmt; F.state = 1; F.fresh = 1;
    VKLOG("flight recorder on: the last %d frames at %dx%d (F5, or gamectl flight, writes them)", NIMG, IW, IH);
    return 1;
}

vkf_row *vkf_row_next(void) {
    if (F.state != 1 || atomic_load(&F.frozen)) return NULL;
    vkf_row *r = &F.rows[F.row_head++ % NROW];
    memset(r, 0, sizeof *r);
    return r;
}
void vkf_note_args(vkf_row *r, uint64_t ind, uint32_t len, uint64_t cnt) {
    if (!r || F.npend == NPEND) return;
    uint32_t i = (uint32_t)(F.arg_head++ % NARG);
    F.pend[F.npend++] = (pend_t){ind, cnt, i, len}; r->args_i = i + 1;
}
void vkf_args_flush(VkCommandBuffer cb) {
    if (!F.npend) return;
    for (int i = 0; i < F.npend; i++) { const pend_t *p = &F.pend[i]; VkBuffer b; VkDeviceSize o;
        if (p->ind && vkm_addr(p->ind, p->len, &b, &o)) { VkBufferCopy c = {o, (VkDeviceSize)p->i * ARG_B, p->len}; vkCmdCopyBuffer(cb, b, F.abuf, 1, &c); }
        if (p->cnt && vkm_addr(p->cnt, 4, &b, &o)) { VkBufferCopy c = {o, (VkDeviceSize)p->i * ARG_B + 20, 4}; vkCmdCopyBuffer(cb, b, F.abuf, 1, &c); }
    }
    vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    F.npend = 0;
}
void vkf_frame(uint64_t frame) {
    if (F.state != 1 || atomic_load(&F.frozen)) return;
    uint64_t k = F.fr_head++ % NFRM;
    F.fr[k].frame = frame; F.fr[k].first = F.row_open; F.fr[k].n = F.row_head - F.row_open; F.fr[k].t = now_ns();
    F.row_open = F.row_head;
}

int vkf_present(VkCommandBuffer cb, VkImage src, VkExtent2D ext, VkFormat fmt, uint64_t frame) {
    if (!start(fmt) || atomic_load(&F.frozen) || fmt != F.fmt) return 0;
    uint64_t slot = F.im_head % NIMG;
    vk_image_barrier(cb, F.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageBlit bl = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {(int32_t)ext.width, (int32_t)ext.height, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {IW, IH, 1}}};
    vkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, F.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
    vk_image_barrier(cb, F.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy c = {slot * IW * IH * 4, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {IW, IH, 1}};
    vkCmdCopyImageToBuffer(cb, F.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, F.buf, 1, &c);
    vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    F.im[slot].frame = frame; F.im[slot].t = now_ns(); F.im_head++;
    return 1;
}

static unsigned __stdcall writer(void *arg) {
    (void)arg;
    uint64_t t0 = now_ns(); SYSTEMTIME st; GetLocalTime(&st);
    char dir[1024]; const char *lp = nxrt_log_path(); snprintf(dir, sizeof dir, "%s", lp && *lp ? lp : ".\\x");
    char *s = strrchr(dir, '\\'); char *s2 = strrchr(dir, '/'); if (s2 > s) s = s2; if (s) *s = 0; else strcpy(dir, ".");
    size_t n = strlen(dir); snprintf(dir + n, sizeof dir - n, "\\flight"); CreateDirectoryA(dir, NULL);
    n = strlen(dir); snprintf(dir + n, sizeof dir - n, "\\%04d%02d%02d_%02d%02d%02d%s%s", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, F.label[0] ? "_" : "", F.label);
    CreateDirectoryA(dir, NULL);
    char p[1200]; FILE *ix, *f;
    snprintf(p, sizeof p, "%s\\index.txt", dir); ix = fopen(p, "w");
    if (ix) fprintf(ix, "image %d %d format %d row_bytes %d\n", IW, IH, (int)F.fmt, (int)sizeof(vkf_row));
    snprintf(p, sizeof p, "%s\\frames.rgba", dir); f = fopen(p, "wb");
    uint64_t nim = F.im_head < NIMG ? F.im_head : NIMG, oldest = F.im_head - nim; size_t bytes = (size_t)IW * IH * 4;
    for (uint64_t i = 0; i < nim && f; i++) { uint64_t sl = (oldest + i) % NIMG;
        fwrite((const uint8_t *)F.buf_mem.map + sl * bytes, 1, bytes, f);
        if (ix) fprintf(ix, "i %llu %llu %llu\n", (unsigned long long)i, (unsigned long long)F.im[sl].frame, (unsigned long long)F.im[sl].t); }
    if (f) fclose(f);
    snprintf(p, sizeof p, "%s\\rows.bin", dir); f = fopen(p, "wb");
    uint64_t nf = F.fr_head < NFRM ? F.fr_head : NFRM, rows = 0;
    for (uint64_t i = F.fr_head - nf; i < F.fr_head && f; i++) { uint64_t k = i % NFRM; const uint64_t first = F.fr[k].first, cnt = F.fr[k].n;
        if (F.row_head - first > NROW) continue;   // the ring has passed it
        if (ix) fprintf(ix, "f %llu %llu %llu %llu\n", (unsigned long long)F.fr[k].frame, (unsigned long long)rows, (unsigned long long)cnt, (unsigned long long)F.fr[k].t);
        for (uint64_t r = 0; r < cnt; r++) { vkf_row v = F.rows[(first + r) % NROW];
            if (v.args_i) memcpy(v.g, (const uint8_t *)F.amem.map + (size_t)(v.args_i - 1) * ARG_B, ARG_B);
            fwrite(&v, sizeof v, 1, f); }
        rows += cnt; }
    if (f) fclose(f);
    if (ix) fclose(ix);
    snprintf(F.last, sizeof F.last, "%s", dir);
    atomic_fetch_add(&F.serial, 1);
    VKLOG("flight recording: %llu frames as shown and %llu rows written to %s (%.1f s)", (unsigned long long)nim, (unsigned long long)rows, dir, (now_ns() - t0) / 1e9);
    atomic_store(&F.frozen, 0);
    return 0;
}
void vkf_poll(void) {
    if (!atomic_exchange(&F.want, 0)) return;
    if (F.state != 1) { VKLOG("flight recording asked for, but the recorder is off"); atomic_fetch_add(&F.serial, 1); return; }
    if (atomic_load(&F.frozen)) return;
    atomic_store(&F.frozen, 1);
    AcquireSRWLockExclusive(&g_vk.qlock); vkQueueWaitIdle(g_vk.q); ReleaseSRWLockExclusive(&g_vk.qlock);   // every copy in the rings has landed
    HANDLE t = (HANDLE)_beginthreadex(NULL, 0, writer, NULL, 0, NULL);
    if (t) CloseHandle(t); else atomic_store(&F.frozen, 0);
}
void vkf_trigger(const char *label) {
    snprintf(F.label, sizeof F.label, "%s", label ? label : "");
    atomic_store(&F.want, 1);
}
int vkf_flashes(long frames, char *out, size_t cap, int min_blocks) {
    enum { GW = IW / 8, GH = IH / 8 };
    static uint8_t g[3][GH][GW][3]; size_t n = 0; int found = 0; out[0] = 0;
    uint64_t head = F.im_head, have = head < NIMG - 2 ? head : NIMG - 2; if ((uint64_t)frames > have) frames = (long)have;
    for (uint64_t i = head - (uint64_t)frames; i < head; i++) { uint64_t sl = i % NIMG; const uint8_t *px = (const uint8_t *)F.buf_mem.map + sl * IW * IH * 4;
        memcpy(g[0], g[1], sizeof g[0]); memcpy(g[1], g[2], sizeof g[0]);
        for (int y = 0; y < GH; y++) for (int x = 0; x < GW; x++) memcpy(g[2][y][x], px + ((size_t)y * 8 * IW + (size_t)x * 8) * 4, 3);
        if (i < head - (uint64_t)frames + 2) continue;
        // Blocks that changed for this one frame only. At full resolution fine detail shimmers from frame to frame, in single blocks all over: a panel is one piece, so what counts is the largest connected piece
        static uint8_t hit[GH][GW]; static int st[GH * GW]; int c = 0;
        for (int y = 0; y < GH; y++) for (int x = 0; x < GW; x++) { int d01 = 0, d12 = 0, d02 = 0;
            for (int k = 0; k < 3; k++) { int a = g[0][y][x][k], b = g[1][y][x][k], e = g[2][y][x][k], p = abs(b - a), q = abs(b - e), r = abs(a - e); d01 = p > d01 ? p : d01; d12 = q > d12 ? q : d12; d02 = r > d02 ? r : d02; }
            hit[y][x] = d01 > 24 && d12 > 24 && d02 < 10; }
        for (int y = 0; y < GH; y++) for (int x = 0; x < GW; x++) if (hit[y][x]) {
            int sp = 0, sz = 0; st[sp++] = y * GW + x; hit[y][x] = 0;
            while (sp) { int v = st[--sp], vy = v / GW, vx = v % GW; sz++;
                if (vx > 0 && hit[vy][vx - 1]) { hit[vy][vx - 1] = 0; st[sp++] = v - 1; } if (vx + 1 < GW && hit[vy][vx + 1]) { hit[vy][vx + 1] = 0; st[sp++] = v + 1; }
                if (vy > 0 && hit[vy - 1][vx]) { hit[vy - 1][vx] = 0; st[sp++] = v - GW; } if (vy + 1 < GH && hit[vy + 1][vx]) { hit[vy + 1][vx] = 0; st[sp++] = v + GW; } }
            if (sz > c) c = sz; }
        if (c > min_blocks) { found++; n += (size_t)snprintf(out + n, cap - n, "%llu ", (unsigned long long)F.im[(sl + NIMG - 1) % NIMG].frame); if (n + 32 > cap) break; } }
    return found;
}
const char *vkf_last(uint64_t *serial) { if (serial) *serial = atomic_load(&F.serial); return F.last; }

// ---------- The pass viewer ----------
#include "vk_tex.h"
enum { TW = 240, TH = 135, TMAX = 6000 };
#define DMAX (1500ull << 20)
typedef struct { uint64_t frame, ser, key; uint32_t pass, w, h, level, layer; int slot, compute; char fmt[24]; int depth; uint64_t dat; VkFormat vk; } tile_t;
static struct {
    int state; VkImage img; vk_alloc img_mem; VkBuffer buf; vk_alloc mem; tile_t *t; int n; uint32_t pass;
    VkBuffer dbuf; vk_alloc dmem; uint64_t dn;
    volatile long want; long left; uint64_t frame;
} V;
static int v_start(void) {
    if (V.state) return V.state > 0;
    V.state = -1;
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, NULL, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R8G8B8A8_UNORM, {TW, TH, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_TILING_OPTIMAL,
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED};
    if (vkCreateImage(g_vk.dev, &ii, NULL, &V.img)) return 0;
    VkMemoryRequirements mr; vkGetImageMemoryRequirements(g_vk.dev, V.img, &mr);
    if (vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, &V.img_mem)) return 0;
    vkBindImageMemory(g_vk.dev, V.img, V.img_mem.mem, V.img_mem.off);
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, (VkDeviceSize)TMAX * TW * TH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    if (vkCreateBuffer(g_vk.dev, &bci, NULL, &V.buf)) return 0;
    vkGetBufferMemoryRequirements(g_vk.dev, V.buf, &mr);
    VkMemoryPropertyFlags hc = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (vk_alloc_mem(&mr, hc | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0, &V.mem) && vk_alloc_mem(&mr, hc, 0, &V.mem)) return 0;
    vkBindBufferMemory(g_vk.dev, V.buf, V.mem.mem, V.mem.off);
    bci.size = DMAX; if (!vkCreateBuffer(g_vk.dev, &bci, NULL, &V.dbuf)) {
        vkGetBufferMemoryRequirements(g_vk.dev, V.dbuf, &mr);
        if (vk_alloc_mem(&mr, hc | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0, &V.dmem) && vk_alloc_mem(&mr, hc, 0, &V.dmem)) V.dbuf = VK_NULL_HANDLE; else vkBindBufferMemory(g_vk.dev, V.dbuf, V.dmem.mem, V.dmem.off); }
    V.t = malloc(sizeof(tile_t) * TMAX);
    if (!V.t) return 0;
    V.state = 1; return 1;
}
int vkv_on(void) { return V.left > 0; }
void vkv_request(long frames) { V.want = frames; }
void vkv_image(VkCommandBuffer cb, struct vkt_image *imv, uint32_t level, uint32_t layer, uint32_t w, uint32_t h, int slot, uint64_t key, int compute) {
    vkt_image *im = imv;
    if (V.left <= 0 || !im || V.n >= TMAX || (im->fi->flags & VKF_COMPRESSED) || w < 32 || h < 32) return;
    if (im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) {   // depth has no blit to colour: the image as it is, thumbnailed when written
        uint64_t bytes = (uint64_t)w * h * 4;
        if (w < 640 || !v_start() || !V.dbuf || V.dn + bytes > DMAX || im->fi->vk == VK_FORMAT_D16_UNORM) return;
        int n = V.n++; tile_t *t = &V.t[n];
        memset(t, 0, sizeof *t); t->frame = V.frame; t->ser = im->serial; t->key = key; t->pass = V.pass; t->w = w; t->h = h; t->level = level; t->layer = layer; t->slot = slot; t->depth = 1; t->dat = V.dn; t->vk = im->fi->vk;
        snprintf(t->fmt, sizeof t->fmt, "%s", im->fi->name);
        vk_image_barrier(cb, im->img, im->aspect, im->rest, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy cp = {V.dn, 0, 0, {VK_IMAGE_ASPECT_DEPTH_BIT, level, layer, 1}, {0, 0, 0}, {w, h, 1}};
        vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, V.dbuf, 1, &cp);
        vk_image_barrier(cb, im->img, im->aspect, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, im->rest, VK_PIPELINE_STAGE_2_COPY_BIT, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
        V.dn += bytes; V.pass++;
        return;
    }
    if (im->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) return;
    VkFormatProperties fp; vkGetPhysicalDeviceFormatProperties(g_vk.pd, im->fi->vk, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) || !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) || !v_start()) return;
    int n = V.n++; tile_t *t = &V.t[n];
    memset(t, 0, sizeof *t); t->frame = V.frame; t->ser = im->serial; t->key = key; t->pass = V.pass; t->w = w; t->h = h; t->level = level; t->layer = layer; t->slot = slot; t->compute = compute;
    snprintf(t->fmt, sizeof t->fmt, "%s", im->fi->name);
    vk_image_barrier(cb, im->img, im->aspect, im->rest, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    vk_image_barrier(cb, V.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageBlit bl = {{im->aspect, level, layer, 1}, {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {TW, TH, 1}}};
    vkCmdBlitImage(cb, im->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, V.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
    vk_image_barrier(cb, V.img, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_BLIT_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy cp = {(VkDeviceSize)n * TW * TH * 4, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {TW, TH, 1}};
    vkCmdCopyImageToBuffer(cb, V.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, V.buf, 1, &cp);
    vk_image_barrier(cb, im->img, im->aspect, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, im->rest, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    V.pass++;
}
static void v_write(void) {
    vks_finish();
    SYSTEMTIME st; GetLocalTime(&st);
    char dir[1024], p[1200]; const char *lp = nxrt_log_path(); snprintf(dir, sizeof dir, "%s", lp && *lp ? lp : ".\\x");
    char *s = strrchr(dir, '\\'); char *s2 = strrchr(dir, '/'); if (s2 > s) s = s2; if (s) *s = 0; else strcpy(dir, ".");
    size_t n = strlen(dir); snprintf(dir + n, sizeof dir - n, "\\flight"); CreateDirectoryA(dir, NULL);
    n = strlen(dir); snprintf(dir + n, sizeof dir - n, "\\pv_%04d%02d%02d_%02d%02d%02d", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond); CreateDirectoryA(dir, NULL);
    for (int i = 0; i < V.n; i++) { const tile_t *t = &V.t[i]; if (!t->depth) continue;   // grey, as depth ^ 1/8; the numbers in tiles.txt
        const uint8_t *d = (const uint8_t *)V.dmem.map + t->dat; uint8_t *o = (uint8_t *)V.mem.map + (size_t)i * TW * TH * 4; float lo = 1e30f, hi = -1e30f; double sum = 0; uint64_t cnt = 0, zero = 0;
        for (uint32_t y = 0; y < t->h; y += 2) for (uint32_t x = 0; x < t->w; x += 2) { uint32_t w0; memcpy(&w0, d + ((size_t)y * t->w + x) * 4, 4); float v;
            if (t->vk == VK_FORMAT_D32_SFLOAT || t->vk == VK_FORMAT_D32_SFLOAT_S8_UINT) memcpy(&v, &w0, 4); else v = (float)(w0 & 0xffffff) / 16777215.0f;
            lo = v < lo ? v : lo; hi = v > hi ? v : hi; sum += v; cnt++; zero += v == 0; }
        ((tile_t *)t)->level = (uint32_t)(sum / (cnt ? cnt : 1) * 1e6); ((tile_t *)t)->layer = (uint32_t)(zero * 100 / (cnt ? cnt : 1));
        for (uint32_t y = 0; y < TH; y++) for (uint32_t x = 0; x < TW; x++) { uint32_t w0; memcpy(&w0, d + ((size_t)(y * t->h / TH) * t->w + x * t->w / TW) * 4, 4); float v;
            if (t->vk == VK_FORMAT_D32_SFLOAT || t->vk == VK_FORMAT_D32_SFLOAT_S8_UINT) memcpy(&v, &w0, 4); else v = (float)(w0 & 0xffffff) / 16777215.0f;
            float g = v <= 0 ? 0 : sqrtf(sqrtf(sqrtf(v))); uint8_t b = (uint8_t)(g > 1 ? 255 : g * 255); uint8_t *q = o + ((size_t)y * TW + x) * 4; q[0] = q[1] = q[2] = b; q[3] = 255; } }
    snprintf(p, sizeof p, "%s\\tiles.rgba", dir); FILE *f = fopen(p, "wb");
    if (f) { fwrite(V.mem.map, (size_t)TW * TH * 4, (size_t)V.n, f); fclose(f); }
    snprintf(p, sizeof p, "%s\\tiles.txt", dir); f = fopen(p, "w");
    if (f) { fprintf(f, "tile %d %d\n", TW, TH);
        for (int i = 0; i < V.n; i++) { const tile_t *t = &V.t[i]; fprintf(f, "%llu %u %d %d %llu %016llx %ux%u %u %u %s\n", (unsigned long long)t->frame, t->pass, t->compute, t->slot, (unsigned long long)t->ser, (unsigned long long)t->key, t->w, t->h, t->level, t->layer, t->fmt); }
        fclose(f); }
    VKLOG("pass viewer: %d thumbnails written to %s", V.n, dir);
    snprintf(F.last, sizeof F.last, "%s", dir); atomic_fetch_add(&F.serial, 1);
}
void vkv_frame(uint64_t frame) {
    if (V.left > 0 && --V.left == 0) { v_write(); V.n = 0; }
    if (V.want > 0 && V.left <= 0) { V.left = V.want; V.want = 0; V.n = 0; V.dn = 0; }
    V.frame = frame + 1; V.pass = 0;
}

// ---------- Watching the buffers of a shader's draws ----------
// Each buffer once a frame: guest memory when the draw is recorded, the device's copy at the end of the pass, and guest memory when the submission has finished. Device and record differ: an upload was missed.
// Record and finish differ: the CPU wrote after it submitted (the replay reads memory at submit, the hardware when it reaches the draw). Words of 1e6 or more, NaN and infinity are counted: a broken transform
enum { WSIZE = 512u << 20, WPEND = 4096, WONE = 256 * 1024 };
typedef struct { uint64_t frame, key, addr, at; uint32_t size; int stage, reg, sb; uint8_t *guest; } wat_t;
static struct { uint64_t keys[8]; int nkey; volatile long want; long left; VkBuffer buf; vk_alloc mem; int st; uint64_t frame, head, seen[256]; int nseen; wat_t pend[WPEND]; int np;
    _Atomic uint64_t n, ndev, ncpu, nbad; _Atomic int nlog; } W;
int vkw_wants(uint64_t key) { if (W.left <= 0 || !key) return 0; for (int i = 0; i < W.nkey; i++) if (W.keys[i] == key) return 1; return 0; }
void vkw_request(long frames, const uint64_t *keys, int n) { W.nkey = n > 8 ? 8 : n; for (int i = 0; i < W.nkey; i++) W.keys[i] = keys[i]; W.want = frames; }
void vkw_note(VkCommandBuffer cb, uint64_t key, int stage, int reg, uint64_t addr, uint32_t size) {
    (void)cb;
    if (W.st == 0) { W.st = -1; VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, WSIZE, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
        if (!vkCreateBuffer(g_vk.dev, &bci, NULL, &W.buf)) { VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, W.buf, &mr); VkMemoryPropertyFlags hc = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            if (!vk_alloc_mem(&mr, hc | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0, &W.mem) || !vk_alloc_mem(&mr, hc, 0, &W.mem)) { vkBindBufferMemory(g_vk.dev, W.buf, W.mem.mem, W.mem.off); W.st = 1; } } }
    if (W.st != 1 || W.np >= WPEND) return;
    for (int i = 0; i < W.nseen; i++) if (W.seen[i] == addr) return;
    if (W.nseen < 256) W.seen[W.nseen++] = addr;
    uint32_t n = size > WONE ? WONE : size & ~3u; if (!n) return;
    uint64_t at = W.head; if (at % WSIZE + n > WSIZE) at += WSIZE - at % WSIZE;
    W.head = at + n;
    wat_t *w = &W.pend[W.np++]; w->frame = W.frame; w->key = key; w->addr = addr; w->size = n; w->stage = stage; w->reg = reg; w->at = at % WSIZE; w->guest = malloc(n);
    if (w->guest) memcpy(w->guest, (const void *)(uintptr_t)addr, n);
}
static void wcheck(void *arg) {
    wat_t *w = arg; const uint32_t *d = (const uint32_t *)((const uint8_t *)W.mem.map + w->at), *g = (const uint32_t *)w->guest, *c = (const uint32_t *)(uintptr_t)w->addr; uint32_t nw = w->size / 4;
    int nd = 0, nc = 0, nb = 0, fd = -1, fc = -1;
    for (uint32_t i = 0; i < nw; i++) { if (d[i] != g[i]) { if (fd < 0) fd = (int)i; nd++; } if (c[i] != g[i]) { if (fc < 0) fc = (int)i; nc++; }
        uint32_t e = d[i] >> 23 & 255; if (e >= 147 + 0 && e != 0 && (d[i] & 0x7fffffff) != 0) { float v; memcpy(&v, &d[i], 4); if (!(v > -1e6f && v < 1e6f)) nb++; } }
    if (w->reg >= 100 && w->size <= 16384) {   // the storage buffer itself: flight/watch.bin, records of {frame, key, addr, size} and the three copies
        static FILE *wf; if (!wf) { char p[1200]; const char *lp = nxrt_log_path(); snprintf(p, sizeof p, "%s", lp && *lp ? lp : ".\\x"); char *s = strrchr(p, '\\'); char *s2 = strrchr(p, '/'); if (s2 > s) s = s2; if (s) strcpy(s, "\\flight\\watch.bin"); else strcpy(p, "watch.bin"); wf = fopen(p, "wb"); }
        if (wf) { uint64_t h[4] = {w->frame, w->key, w->addr, w->size}; fwrite(h, 8, 4, wf); fwrite(g, 1, w->size, wf); fwrite(d, 1, w->size, wf); fwrite(c, 1, w->size, wf); fflush(wf); } }
    atomic_fetch_add(&W.n, 1); if (nd) atomic_fetch_add(&W.ndev, 1); if (nc) atomic_fetch_add(&W.ncpu, 1); if (nb) atomic_fetch_add(&W.nbad, 1);
    if (w->reg >= 100 && (nd || nc) && atomic_fetch_add(&W.nlog, 1) < 4000) {
        float gv = 0, dv = 0, cv = 0; if (fd >= 0) { memcpy(&gv, &g[fd], 4); memcpy(&dv, &d[fd], 4); } if (fc >= 0) { memcpy(&cv, &c[fc], 4); }
        int gpu_old = 0; for (uint32_t i = 0; i < nw; i++) gpu_old += d[i] != c[i];   // words the GPU read that are not what the CPU ended with
        VKLOG("watch frame %llu shader %016llx sbuf %d at %#llx+%u: device differs from the record in %d words (first %d: guest %g device %g), the CPU changed %d words after submit (first %d now %g), device differs from the final memory in %d words", (unsigned long long)w->frame, (unsigned long long)w->key, w->reg - 100, (unsigned long long)w->addr, w->size, nd, fd, gv, dv, nc, fc, cv, gpu_old);
    }
    free(w->guest); free(w);
}
void vkw_flush(VkCommandBuffer cb) {
    if (!W.np) return;
    for (int i = 0; i < W.np; i++) { const wat_t *w = &W.pend[i]; VkBuffer b; VkDeviceSize o;
        if (vkm_addr(w->addr, w->size, &b, &o)) { VkBufferCopy c = {o, w->at, w->size}; vkCmdCopyBuffer(cb, b, W.buf, 1, &c); wat_t *h = malloc(sizeof *h); *h = *w; vks_post_fn(wcheck, h); } else free(w->guest); }
    vk_mem_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    W.np = 0;
}
void vkw_frame(uint64_t frame) {
    W.nseen = 0;
    if (W.left > 0 && --W.left == 0) {
        snprintf(F.last, sizeof F.last, "watch running totals: %llu buffers, device differs in %llu, CPU wrote after submit in %llu, huge words in %llu", (unsigned long long)atomic_load(&W.n), (unsigned long long)atomic_load(&W.ndev), (unsigned long long)atomic_load(&W.ncpu), (unsigned long long)atomic_load(&W.nbad));
        VKLOG("%s", F.last); atomic_fetch_add(&F.serial, 1);
    }
    if (W.want > 0 && W.left <= 0) { W.left = W.want; W.want = 0; atomic_store(&W.n, 0); atomic_store(&W.ndev, 0); atomic_store(&W.ncpu, 0); atomic_store(&W.nbad, 0); atomic_store(&W.nlog, 0); }
    W.frame = frame + 1;
}

// ---------- Pixels per draw ----------
enum { QMAX = 60000 };
typedef struct { uint64_t frame, vs, ps, ibuf; uint32_t count, pass, rt; int32_t base_v; int occ; } qrow_t;
static struct { VkQueryPool pool, occ; qrow_t *r; int n, open, occ_open, st, reset; volatile long want; long left; uint64_t frame; } Q;
int vkq_on(void) { return Q.left > 0; }
void vkq_request(long frames) { Q.want = frames; }
void vkq_close(VkCommandBuffer cb) { if (Q.open) { vkCmdEndQuery(cb, Q.pool, (uint32_t)Q.n - 1); Q.open = 0; } if (Q.occ_open) { vkCmdEndQuery(cb, Q.occ, (uint32_t)Q.n - 1); Q.occ_open = 0; } }
void vkq_begin(VkCommandBuffer cb, uint64_t vs, uint64_t ps, uint32_t count, int32_t base_v, uint64_t ibuf, uint32_t pass_i, uint32_t rt, int occ) {
    vkq_close(cb);
    if (Q.st != 1 || Q.n >= QMAX) return;
    qrow_t *q = &Q.r[Q.n++]; q->frame = Q.frame; q->vs = vs; q->ps = ps; q->ibuf = ibuf; q->count = count; q->pass = pass_i; q->rt = rt; q->base_v = base_v; q->occ = occ && Q.occ;
    vkCmdBeginQuery(cb, Q.pool, (uint32_t)Q.n - 1, 0); Q.open = 1;
    if (q->occ) { vkCmdBeginQuery(cb, Q.occ, (uint32_t)Q.n - 1, g_vk.f10.occlusionQueryPrecise ? VK_QUERY_CONTROL_PRECISE_BIT : 0); Q.occ_open = 1; }
}
static void q_write(void) {
    vks_finish();
    SYSTEMTIME st; GetLocalTime(&st);
    char p[1200]; const char *lp = nxrt_log_path(); snprintf(p, sizeof p, "%s", lp && *lp ? lp : ".\\x");
    char *s = strrchr(p, '\\'); char *s2 = strrchr(p, '/'); if (s2 > s) s = s2; if (s) *s = 0; else strcpy(p, ".");
    size_t n = strlen(p); snprintf(p + n, sizeof p - n, "\\flight"); CreateDirectoryA(p, NULL);
    n = strlen(p); snprintf(p + n, sizeof p - n, "\\ds_%04d%02d%02d_%02d%02d%02d.txt", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    FILE *f = fopen(p, "w");
    uint64_t *v = malloc(sizeof(uint64_t) * (size_t)Q.n * 2), *vo = calloc((size_t)Q.n * 2 + 2, sizeof(uint64_t));
    if (vo && Q.occ) for (int i = 0; i < Q.n; i++) if (Q.r[i].occ) vkGetQueryPoolResults(g_vk.dev, Q.occ, (uint32_t)i, 1, 16, &vo[2 * (size_t)i], 16, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (f && v && vo && Q.n) {
        VkResult rc = vkGetQueryPoolResults(g_vk.dev, Q.pool, 0, (uint32_t)Q.n, sizeof(uint64_t) * (size_t)Q.n * 2, v, 16, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
        if (rc < 0) VKRED("draw statistics: results not read (%d)", rc);
        for (int i = 0; i < Q.n; i++) { const qrow_t *q = &Q.r[i]; fprintf(f, "%llu %u %016llx %016llx %u %d %llx %u %llu %lld\n", (unsigned long long)q->frame, q->pass, (unsigned long long)q->vs, (unsigned long long)q->ps, q->count, q->base_v, (unsigned long long)q->ibuf, q->rt, v[2 * (size_t)i + 1] ? (unsigned long long)v[2 * (size_t)i] : 0ull, q->occ && vo[2 * (size_t)i + 1] ? (long long)vo[2 * (size_t)i] : -1ll); }
    }
    if (f) fclose(f);
    free(v); free(vo);
    VKLOG("draw statistics: %d draws written to %s", Q.n, p);
    snprintf(F.last, sizeof F.last, "%s", p); atomic_fetch_add(&F.serial, 1);
    Q.n = 0;
}
void vkq_frame(uint64_t frame) {
    if (Q.left > 0 && --Q.left == 0) q_write();
    if (Q.want > 0 && Q.left <= 0) {
        if (!Q.st) { Q.st = -1; VkQueryPoolCreateInfo ci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_PIPELINE_STATISTICS, QMAX, VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT};
            if (!vkCreateQueryPool(g_vk.dev, &ci, NULL, &Q.pool) && (Q.r = malloc(sizeof(qrow_t) * QMAX))) Q.st = 1;
            VkQueryPoolCreateInfo oi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, NULL, 0, VK_QUERY_TYPE_OCCLUSION, QMAX, 0}; if (vkCreateQueryPool(g_vk.dev, &oi, NULL, &Q.occ)) Q.occ = VK_NULL_HANDLE; }
        if (Q.st == 1) Q.reset = 1;
        Q.left = Q.want; Q.want = 0; Q.n = 0;
    }
    Q.frame = frame + 1;
}
void vkq_reset_if(VkCommandBuffer cb) { if (Q.reset) { vkCmdResetQueryPool(cb, Q.pool, 0, QMAX); if (Q.occ) vkCmdResetQueryPool(cb, Q.occ, 0, QMAX); Q.reset = 0; } }
int vkq_resetting(void) { return Q.reset; }
