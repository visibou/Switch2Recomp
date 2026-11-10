/* s2tex: block-linear <-> linear (see s2tex.h and docs/s2tex.md). MIT licence, part of pokopia-native. */
#include "s2tex.h"
#include <string.h>

/* Tegra X1 GOB: byte offset bits 8..0 = x5 y2 y1 x4 y0 x3 x2 x1 x0 (x = byte in the GOB's 64-byte row, y = row 0..7).
 * Switch 2: the same with offset bits 5 and 6 exchanged, i.e. x5 y2 x4 y1 y0 x3 x2 x1 x0. */
uint32_t s2tex_gob_offset(uint32_t x_byte, uint32_t y, s2tex_order order) {
    uint32_t x = x_byte & 63, r = y & 7;
    uint32_t o = (x & 15) | (r & 1) << 4 | (x >> 4 & 1) << 5 | (r >> 1 & 3) << 6 | (x >> 5 & 1) << 8;
    if (order == S2TEX_SWITCH2) o = (o & ~0x60u) | (o & 0x20u) << 1 | (o & 0x40u) >> 1;
    return o;
}

typedef struct {
    uint32_t row_bytes, rows, slices, bh, bd, gobs_x;
    uint64_t block, row_of_blocks, layer_of_blocks;
} geom;

static int geom_of(const s2tex_surface *s, geom *g) {
    if (!s || !s->width || !s->height || !s->bytes_per_block || s->bytes_per_block > 16 || s->block_height > 5 || s->block_depth > 5) return S2TEX_EINVAL;
    if (s->order != S2TEX_TX1 && s->order != S2TEX_SWITCH2) return S2TEX_EINVAL;
    uint64_t rb = (uint64_t)s->width * s->bytes_per_block;
    if (rb > UINT32_MAX) return S2TEX_EINVAL;
    g->row_bytes = (uint32_t)rb;
    g->rows = s->height;
    g->slices = s->depth ? s->depth : 1;
    g->bh = s->block_height;
    g->bd = s->block_depth;
    uint32_t need = (uint32_t)((rb + 63) / 64);
    g->gobs_x = s->gobs_x ? s->gobs_x : need;
    if (g->gobs_x < need) return S2TEX_EINVAL;
    g->block = (uint64_t)S2TEX_GOB_BYTES << (g->bh + g->bd);
    g->row_of_blocks = g->gobs_x * g->block;
    uint64_t block_rows = ((uint64_t)g->rows + (8u << g->bh) - 1) >> (3 + g->bh);
    g->layer_of_blocks = block_rows * g->row_of_blocks;
    return S2TEX_OK;
}

static uint64_t geom_size(const geom *g) {
    uint64_t layers = ((uint64_t)g->slices + (1u << g->bd) - 1) >> g->bd;
    return layers * g->layer_of_blocks;
}

uint64_t s2tex_size(const s2tex_surface *s) {
    geom g;
    return geom_of(s, &g) ? 0 : geom_size(&g);
}

/* Start of the GOB row holding (y, z): the block layer, block row and GOB inside the block, plus y's in-GOB term. */
static uint64_t row_base(const geom *g, uint32_t y, uint32_t z, s2tex_order order) {
    return (uint64_t)(z >> g->bd) * g->layer_of_blocks
         + (uint64_t)(y >> (3 + g->bh)) * g->row_of_blocks
         + ((uint64_t)(z & ((1u << g->bd) - 1)) << (9 + g->bh))
         + ((uint64_t)((y >> 3) & ((1u << g->bh) - 1)) << 9)
         + s2tex_gob_offset(0, y, order);
}

uint64_t s2tex_offset(const s2tex_surface *s, uint32_t x_byte, uint32_t y, uint32_t z) {
    geom g;
    if (geom_of(s, &g)) return 0;
    return row_base(&g, y, z, s->order) + (uint64_t)(x_byte >> 6) * g.block + s2tex_gob_offset(x_byte, 0, s->order);
}

/* The in-GOB order is a bit permutation, so a GOB offset is the sum of an x part and a y part; a row is copied in 16-byte
 * segments (bits 0..3 are x0..x3 in both orders), one of four positions per GOB. */
static int move(const s2tex_surface *s, uint8_t *bl, size_t bl_size, uint8_t *lin, size_t row_pitch, size_t slice_pitch, int to_linear) {
    geom g;
    int e = geom_of(s, &g);
    if (e) return e;
    if (!bl || !lin || row_pitch < g.row_bytes) return S2TEX_EINVAL;
    if (!slice_pitch) slice_pitch = (size_t)g.rows * row_pitch;
    if (slice_pitch < (size_t)(g.rows - 1) * row_pitch + g.row_bytes) return S2TEX_EINVAL;
    if (bl_size < geom_size(&g)) return S2TEX_ESIZE;
    uint32_t seg[4];
    for (uint32_t c = 0; c < 4; c++) seg[c] = s2tex_gob_offset(c * 16, 0, s->order);
    uint32_t nfull = g.row_bytes >> 4, tail = g.row_bytes & 15;
    for (uint32_t z = 0; z < g.slices; z++) {
        for (uint32_t y = 0; y < g.rows; y++) {
            uint8_t *row = lin + (size_t)z * slice_pitch + (size_t)y * row_pitch;
            uint8_t *gob_row = bl + row_base(&g, y, z, s->order);
            for (uint32_t j = 0; j < nfull; j++) {
                uint8_t *p = gob_row + (uint64_t)(j >> 2) * g.block + seg[j & 3];
                if (to_linear) memcpy(row + ((size_t)j << 4), p, 16);
                else memcpy(p, row + ((size_t)j << 4), 16);
            }
            if (tail) {
                uint8_t *p = gob_row + (uint64_t)(nfull >> 2) * g.block + seg[nfull & 3];
                if (to_linear) memcpy(row + ((size_t)nfull << 4), p, tail);
                else memcpy(p, row + ((size_t)nfull << 4), tail);
            }
        }
    }
    return S2TEX_OK;
}

int s2tex_deswizzle(const s2tex_surface *s, const void *bl, size_t bl_size, void *linear, size_t row_pitch, size_t slice_pitch) {
    return move(s, (uint8_t *)(uintptr_t)bl, bl_size, (uint8_t *)linear, row_pitch, slice_pitch, 1);
}

int s2tex_swizzle(const s2tex_surface *s, const void *linear, size_t row_pitch, size_t slice_pitch, void *bl, size_t bl_size) {
    return move(s, (uint8_t *)bl, bl_size, (uint8_t *)(uintptr_t)linear, row_pitch, slice_pitch, 0);
}

static uint32_t mip(uint32_t v, unsigned level) { v = level < 32 ? v >> level : 0; return v ? v : 1; }

/* A level keeps the block height while its rows exceed half a block (8 << (n - 1) rows), else halves it; the same for depth with 1-slice GOBs. */
static uint32_t clamp_log2(uint32_t n, uint32_t unit, uint32_t extent) {
    while (n && ((uint64_t)unit << (n - 1)) >= extent) n--;
    return n;
}

static int level_surface(const s2tex_texture *t, unsigned level, s2tex_surface *s) {
    if (!t || !t->width || !t->height || !t->block_w || !t->block_h || !t->bytes_per_block || t->bytes_per_block > 16) return S2TEX_EINVAL;
    if (t->block_height > 5 || t->block_depth > 5 || level >= (t->levels ? t->levels : 1)) return S2TEX_EINVAL;
    uint32_t w = mip(t->width, level), h = mip(t->height, level), d = mip(t->depth ? t->depth : 1, level);
    s->width = (w + t->block_w - 1) / t->block_w;
    s->height = (h + t->block_h - 1) / t->block_h;
    s->depth = d;
    s->bytes_per_block = t->bytes_per_block;
    s->block_height = clamp_log2(t->block_height, 8, s->height);
    s->block_depth = clamp_log2(t->block_depth, 1, d);
    s->gobs_x = 0;
    s->order = t->order;
    return S2TEX_OK;
}

uint64_t s2tex_level(const s2tex_texture *t, unsigned level, s2tex_surface *out) {
    uint64_t off = 0;
    s2tex_surface s;
    for (unsigned i = 0; i <= level; i++) {
        if (level_surface(t, i, &s)) return UINT64_MAX;
        if (i == level) break;
        off += s2tex_size(&s);
    }
    if (out) *out = s;
    return off;
}

uint64_t s2tex_layer_size(const s2tex_texture *t) {
    unsigned n = t && t->levels ? t->levels : 1;
    s2tex_surface s;
    uint64_t off = s2tex_level(t, n - 1, &s);
    return off == UINT64_MAX ? 0 : off + s2tex_size(&s);
}

uint64_t s2tex_layer_stride(const s2tex_texture *t) {
    s2tex_surface s0;
    uint64_t size = s2tex_layer_size(t);
    if (!size || s2tex_level(t, 0, &s0) == UINT64_MAX) return 0;
    uint64_t align = (uint64_t)S2TEX_GOB_BYTES << (s0.block_height + s0.block_depth);
    return (size + align - 1) / align * align;
}
