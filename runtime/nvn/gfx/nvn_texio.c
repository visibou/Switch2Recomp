// CPU texel access through s2tex (nvn_texio.h).
#include "nvn_texio.h"
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

static _Atomic int g_order = -1;
void nvn_texio_set_order(s2tex_order o) { atomic_store(&g_order, (int)o); }
s2tex_order nvn_texio_order(void) {
    int o = atomic_load(&g_order);
    if (o < 0) { const char *e = getenv("NXRT_TEX_ORDER"); o = e && !strcmp(e, "tx1") ? S2TEX_TX1 : S2TEX_SWITCH2; atomic_store(&g_order, o); }
    return (s2tex_order)o;
}
static uint32_t mip(uint32_t s, unsigned l) { s >>= l; return s ? s : 1; }
static uint32_t blk(uint32_t s, uint32_t b) { return (s + b - 1) / b; }

int nvn_texio_surface(const tex_t *t, unsigned level, unsigned layer, s2tex_surface *s, uint64_t *base, size_t *row) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    unsigned levels = t->d.levels ? t->d.levels : 1;
    if (!bpb || level >= levels || t->d.samples > 1 || !t->mem) return -1;
    int tg = t->d.target, is3d = tg == NVN_TEXTURE_TARGET_3D, is1d = tg == NVN_TEXTURE_TARGET_1D || tg == NVN_TEXTURE_TARGET_1D_ARRAY;
    memset(s, 0, sizeof *s);
    s->width = blk(mip(t->d.width, level), (uint32_t)bw);
    s->height = is1d ? 1 : blk(mip(t->d.height, level), (uint32_t)bh);
    s->depth = is3d ? mip(t->d.depth, level) : 1;
    s->bytes_per_block = (uint32_t)bpb; s->order = nvn_texio_order();
    if (tg == NVN_TEXTURE_TARGET_BUFFER || (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET))) {
        if (level) return -1;
        size_t r = t->d.stride ? t->d.stride : (size_t)s->width * (size_t)bpb;
        if (row) *row = r;
        *base = (uint64_t)layer * t->L.layer_size;
        return *base + (uint64_t)r * (s->height - 1) + (uint64_t)s->width * (uint64_t)bpb <= t->L.size ? 1 : -1;
    }
    unsigned th, td; nvn_tex_level_tiles(&t->d, level, &th, &td);
    s->block_height = th; s->block_depth = td; s->gobs_x = nvn_tex_level_gobs(&t->d, level);
    *base = (uint64_t)layer * t->L.layer_size + nvn_tex_level_offset(&t->d, level);
    uint64_t n = s2tex_size(s);
    return n && *base + n <= t->L.size ? 0 : -1;
}

int nvn_texio_level_span(const tex_t *t, unsigned level, unsigned layer, uint64_t *off, uint64_t *len) {
    s2tex_surface s; size_t row = 0; int k = nvn_texio_surface(t, level, layer, &s, off, &row);
    if (k < 0) return -1;
    *len = k == 1 ? (uint64_t)row * s.height : s2tex_size(&s);
    return 0;
}

// One copy, in either direction: rows of whole blocks, in runs that stay inside one 16-byte unit of a GOB (bits 0-3 of a GOB offset are the byte's own in both orders)
static int texio(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *lin, size_t row, size_t img, int write) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (!bpb || r->width <= 0 || r->height <= 0 || r->xoffset < 0 || r->yoffset < 0 || r->zoffset < 0 || r->xoffset % bw || r->yoffset % bh) return -1;
    int is3d = t->d.target == NVN_TEXTURE_TARGET_3D, nz = r->depth > 0 ? r->depth : 1;
    size_t wb = (size_t)blk((uint32_t)r->width, (uint32_t)bw) * (size_t)bpb, hb = blk((uint32_t)r->height, (uint32_t)bh);
    if (!row) row = wb;
    if (!img) img = row * hb;
    uint32_t x0 = (uint32_t)(r->xoffset / bw) * (uint32_t)bpb, y0 = (uint32_t)(r->yoffset / bh);
    for (int z = 0; z < nz; z++) {
        s2tex_surface s; uint64_t base; size_t lrow = 0;
        unsigned layer = is3d ? layer0 : layer0 + (unsigned)(r->zoffset + z), slice = is3d ? (unsigned)(r->zoffset + z) : 0;
        int k = nvn_texio_surface(t, level, layer, &s, &base, &lrow);
        if (k < 0 || x0 + wb > (size_t)s.width * bpb || y0 + hb > s.height || slice >= s.depth) return -1;
        uint8_t *mem = t->mem + base, *l = lin + (size_t)z * img;
        for (size_t y = 0; y < hb; y++)
            for (size_t x = 0; x < wb;) {
                uint32_t bx = x0 + (uint32_t)x; size_t n = 16 - bx % 16; if (n > wb - x) n = wb - x;
                uint64_t o = k == 1 ? (uint64_t)(y0 + y) * lrow + bx : s2tex_offset(&s, bx, y0 + (uint32_t)y, slice);
                if (write) memcpy(mem + o, l + y * row + x, n); else memcpy(l + y * row + x, mem + o, n);
                x += n;
            }
    }
    return 0;
}
int nvn_texio_read(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img) { return texio(t, level, layer0, r, dst, row, img, 0); }
int nvn_texio_write(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, const uint8_t *src, size_t row, size_t img) { return texio(t, level, layer0, r, (uint8_t *)src, row, img, 1); }
