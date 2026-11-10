// texkey: the converted-texture pack keys the runtime computes for a texture's levels (vk_astc.cpp make_key over the deswizzled ASTC bytes),
// from the texture's description and its storage bytes. Built as a DLL for tools/texprep/texprep.py. Same layout code as the runtime (nvn_tex.c, s2tex.c).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "nvn_tex.h"
#include "s2tex.h"

static const uint64_t VKA_VERSION = 0x5637414bull + 1;   // must match vk_astc.cpp
static inline uint64_t mix(uint64_t a, uint64_t b) { unsigned __int128 m = (unsigned __int128)a * b; return (uint64_t)m ^ (uint64_t)(m >> 64); }
static uint64_t hash64(const uint8_t *p, size_t n, uint64_t seed) {
    uint64_t h = seed ^ (n * 0x9E3779B97F4A7C15ull); size_t i = 0;
    for (; i + 16 <= n; i += 16) { uint64_t a, b; memcpy(&a, p + i, 8); memcpy(&b, p + i + 8, 8); h = mix(a ^ 0xa0761d6478bd642full ^ h, b ^ 0xe7037ed1a0b428dbull); }
    uint64_t t[2] = {0, 0}; memcpy(t, p + i, n - i);
    return mix(h ^ t[0] ^ 0x8ebc6af09c88c6e3ull, t[1] ^ 0x589965cc75374cc3ull ^ n);
}
static uint32_t mip(uint32_t v, unsigned l) { v >>= l; return v ? v : 1; }

// Per level (layer-major: layer 0 levels 0..n-1, layer 1 ...): out[2*i] = key lo, out[2*i+1] = key hi, sizes[i] = BC3 bytes; 0 key when the level could not be read.
// lin_out (optional, may be NULL): receives the linear ASTC bytes of level `want_level` of layer `want_layer`, lin_cap bytes at most; returns its size in *lin_n.
__declspec(dllexport) int tk_keys(const uint8_t *data, uint64_t n, int fmt, uint32_t w, uint32_t h, uint32_t levels, uint32_t layers, int pkg_block_h,
                                  uint64_t *out, uint64_t *sizes) {
    int bw, bh, bpb = nvn_fmt_bpb(fmt, &bw, &bh);
    if (bpb != 16 || bw < 4) return -1;
    nvn_texdesc d; memset(&d, 0, sizeof d);
    d.target = layers > 1 ? 4 : 1; d.format = fmt; d.width = w; d.height = h; d.depth = 1; d.levels = levels ? levels : 1; d.pkg_block_h = pkg_block_h;
    nvn_texlayout L; nvn_tex_layout(&d, &L);
    int srgb = fmt >= 0x87;
    for (uint32_t y = 0; y < (layers ? layers : 1); y++) for (uint32_t l = 0; l < d.levels; l++) {
        uint64_t *o = out + 2 * ((uint64_t)y * d.levels + l); o[0] = o[1] = 0;
        uint32_t lw = mip(w, l), lh = mip(h, l);
        sizes[(uint64_t)y * d.levels + l] = (uint64_t)((lw + 3) / 4) * ((lh + 3) / 4) * 16;
        s2tex_surface s; memset(&s, 0, sizeof s);
        s.width = (lw + bw - 1) / bw; s.height = (lh + bh - 1) / bh; s.depth = 1; s.bytes_per_block = 16; s.order = S2TEX_SWITCH2;
        unsigned th, td; nvn_tex_level_tiles(&d, l, &th, &td); s.block_height = th; s.block_depth = td; s.gobs_x = nvn_tex_level_gobs(&d, l);
        uint64_t base = (uint64_t)y * L.layer_size + nvn_tex_level_offset(&d, l), need = s2tex_size(&s);
        if (!need || base + need > n) continue;
        size_t lrow = (size_t)s.width * 16, lsize = lrow * s.height; uint8_t *lin = malloc(lsize);
        if (!lin) continue;
        if (!s2tex_deswizzle(&s, data + base, (size_t)(n - base), lin, lrow, lsize)) {
            uint64_t shape = (uint64_t)lw | (uint64_t)lh << 20 | (uint64_t)1 << 40 | (uint64_t)bw << 52 | (uint64_t)bh << 56 | (uint64_t)srgb << 60;
            o[0] = hash64(lin, lsize, VKA_VERSION ^ shape); o[1] = hash64(lin, lsize, ~VKA_VERSION ^ (shape * 0x2545F4914F6CDD1Dull));
        }
        free(lin);
    }
    return 0;
}
