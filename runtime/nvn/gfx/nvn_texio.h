// CPU texel access to an NVN texture's storage: regions of a level copied between the storage's block-linear layout and linear rows, through s2tex
// (src/s2tex, the clean-room layout). Replaces nxnative's dm/gfx/tex/nvn_deswizzle.c (not ported: docs/m4-title.md §7). The storage layout (level offsets,
// block heights, GOB columns) is nvn_tex.c's, the in-GOB byte order the Switch 2's unless nvn_texio_set_order says otherwise.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "nvn_obj.h"
#include "s2tex.h"

void nvn_texio_set_order(s2tex_order o);   // default S2TEX_SWITCH2 (NXRT_TEX_ORDER=tx1 at the first use selects the Tegra X1's)
s2tex_order nvn_texio_order(void);
// Level `level` of layer `layer` as an s2tex surface and its byte offset in the storage. 0 = block-linear, 1 = a linear texture (row stride in *row),
// -1 = no such level, an unknown format, a multisampled texture or a level that leaves the storage
int nvn_texio_surface(const tex_t *t, unsigned level, unsigned layer, s2tex_surface *s, uint64_t *base, size_t *row);
// Region r (texels; z = layers, or slices of a 3D level) of level `level` from layer layer0 on, to / from linear rows of `row` bytes and images of `img` bytes
// (0 = tightly packed). 0 on success, -1 when the region does not fit the level or the texture cannot be addressed (nothing is copied then)
int nvn_texio_read(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, uint8_t *dst, size_t row, size_t img);
int nvn_texio_write(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, const uint8_t *src, size_t row, size_t img);
// The byte span of level `level` of one layer in the storage (for the write preparation of the dirty tracker); 0 on success
int nvn_texio_level_span(const tex_t *t, unsigned level, unsigned layer, uint64_t *off, uint64_t *len);
