/* s2tex: NVIDIA block-linear <-> linear texture layout for the Tegra X1 (Switch) and the Switch 2.
 * MIT licence, part of pokopia-native. Written from public descriptions of the block-linear (GOB) layout; see docs/s2tex.md.
 *
 * A GOB is 512 bytes covering 64 bytes x 8 rows. Blocks are 1 GOB wide, 2^block_height GOBs high and 2^block_depth GOBs deep
 * (GOBs ordered y first, then z, inside a block); blocks fill the surface in x, then y, then z order. The two GPUs differ only in
 * the byte order inside a GOB (s2tex_gob_offset). Plain C99, no dependencies. */
#ifndef S2TEX_H
#define S2TEX_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    S2TEX_TX1 = 0,      /* Tegra X1 (Maxwell) GOB order */
    S2TEX_SWITCH2 = 1   /* Switch 2: the Tegra X1 order with address bits 5 and 6 exchanged */
} s2tex_order;

enum { S2TEX_GOB_BYTES = 512, S2TEX_GOB_WIDTH = 64, S2TEX_GOB_ROWS = 8 };
enum { S2TEX_OK = 0, S2TEX_EINVAL = -1, S2TEX_ESIZE = -2 };

/* One block-linear surface: a mip level of a 2D texture, a 2D array layer, or all slices of one 3D level.
 * Sizes count format blocks: texels for uncompressed formats, 4x4 (or ASTC) blocks for compressed ones. */
typedef struct {
    uint32_t width;           /* in blocks */
    uint32_t height;          /* in block rows */
    uint32_t depth;           /* slices (3D); 0 or 1 for 2D */
    uint32_t bytes_per_block; /* 1..16, any value (3, 6 and 12 included) */
    uint32_t block_height;    /* log2 of the GOBs per block, vertically (0..5) */
    uint32_t block_depth;     /* log2 of the GOBs per block, in depth (0..5) */
    uint32_t gobs_x;          /* GOB columns as laid out; 0 = ceil(width * bytes_per_block / 64). Larger for padded rows */
    s2tex_order order;
} s2tex_surface;

/* Byte offset of (x_byte % 64, y % 8) inside its GOB. */
uint32_t s2tex_gob_offset(uint32_t x_byte, uint32_t y, s2tex_order order);

/* Byte offset of byte x_byte of row y, slice z, from the start of the surface. No bounds check. */
uint64_t s2tex_offset(const s2tex_surface *s, uint32_t x_byte, uint32_t y, uint32_t z);

/* Block-linear bytes the surface occupies: whole blocks, slices rounded up to whole block depths. 0 for invalid parameters. */
uint64_t s2tex_size(const s2tex_surface *s);

/* Block-linear -> linear. Row y of slice z lands at linear + z * slice_pitch + y * row_pitch; row_pitch >= width * bytes_per_block,
 * slice_pitch 0 = height * row_pitch. Bytes between rows are not touched. bl_size is checked against s2tex_size. */
int s2tex_deswizzle(const s2tex_surface *s, const void *bl, size_t bl_size, void *linear, size_t row_pitch, size_t slice_pitch);

/* Linear -> block-linear, the reverse. Storage bytes outside the surface's texels (row and block padding) are not touched. */
int s2tex_swizzle(const s2tex_surface *s, const void *linear, size_t row_pitch, size_t slice_pitch, void *bl, size_t bl_size);

/* Mip chains. A texture's levels lie one after another in its storage, each in whole blocks of its own block height and depth;
 * a level keeps the texture's block height / depth until it is no taller (deeper) than half of a block, then halves it. */
typedef struct {
    uint32_t width, height, depth;    /* level 0 in texels (depth 1 for 2D) */
    uint32_t block_w, block_h;        /* texels per format block (1x1 uncompressed, 4x4 BC, ASTC footprints) */
    uint32_t bytes_per_block;
    uint32_t block_height, block_depth; /* level 0's, log2 GOBs (the texture's layout) */
    uint32_t levels;                  /* 0 counts as 1 */
    s2tex_order order;
} s2tex_texture;

/* Fills *out with level `level`'s surface and returns its byte offset from the start of the texture (one layer). Returns UINT64_MAX for an invalid level. */
uint64_t s2tex_level(const s2tex_texture *t, unsigned level, s2tex_surface *out);

/* Bytes of one layer: all levels. An array texture's layers are this far apart once aligned to level 0's block (s2tex_layer_stride). */
uint64_t s2tex_layer_size(const s2tex_texture *t);
uint64_t s2tex_layer_stride(const s2tex_texture *t);

#ifdef __cplusplus
}
#endif
#endif
