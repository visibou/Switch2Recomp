// texprep: converts a texture's ASTC levels into the runtime's converted-texture pack ahead of play, the same way the runtime's worker does
// (vk_sparse.inc sp_tex_job_run: deswizzle, vka_decode_rgba8, vka_encode_bc7_st at VKA_FAST, vka_pack_put), so the game finds them instead of transcoding.
// Built as a DLL with the runtime's own objects (vk_astc.cpp, astcenc, nvn_tex.c, s2tex.c); driven by tools/texprep/texprep.py. Nothing here knows a game.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include "nvn_tex.h"
#include "s2tex.h"
#include "vk_astc.h"

void vka_key_of(const uint8_t *astc, size_t n, uint32_t w, uint32_t h, uint32_t s, int bw, int bh, int srgb, vka_key *out);
int vka_pack_has(const vka_key *key, size_t size);
int vka_pack_ready(void);
uint64_t vka_pack_bytes(void);
void vka_pack_put(const vka_key *key, const uint8_t *data, size_t size);

// What vk_astc.cpp needs from the rest of the runtime
static int bc3_block_opaque(const uint8_t *b) {   // as vk_tex.c: all 16 alpha values 255 in the form encoders write
    if (b[0] != 255 || b[1] != 255) return 0;
    uint64_t ix = 0; memcpy(&ix, b + 2, 6);
    uint64_t c6 = ~(ix ^ 0xDB6DB6DB6DB6ull) & 0xFFFFFFFFFFFFull;
    c6 = c6 & (c6 >> 1) & (c6 >> 2) & 0x249249249249ull;
    return c6 == 0;
}
int vkt_bc3_opaque(const uint8_t *src, size_t size) { for (size_t o = 0; o + 16 <= size; o += 16) if (!bc3_block_opaque(src + o)) return 0; return 1; }

static uint32_t mip(uint32_t v, unsigned l) { v >>= l; return v ? v : 1; }

// Smallest level the runtime sends through the pack (vk_tex.c upload: NXRT_VK_ASYNC_MIN, default 128): smaller ones are converted where they are used
static uint32_t g_min = 128;
__declspec(dllexport) void tp_set_min(uint32_t m) { g_min = m; }
__declspec(dllexport) int tp_ready(void) { return vka_pack_ready(); }
__declspec(dllexport) uint64_t tp_pack_bytes(void) { return vka_pack_bytes(); }

// One texture (all levels and layers of a 2D or 2D array texture): returns the levels written; *have = levels already in the pack, *skip = levels too small or unreadable.
// fmt: NVN format (ASTC 0x79..0x94); pkg_block_h: the packaged layout byte + 1 (0 = automatic).
__declspec(dllexport) int tp_texture(const uint8_t *data, uint64_t n, int fmt, uint32_t w, uint32_t h, uint32_t levels, uint32_t layers, int pkg_block_h,
                                     int *have, int *skip, uint64_t *bytes) {
    int bw, bh, bpb = nvn_fmt_bpb(fmt, &bw, &bh);
    if (bpb != 16 || bw < 4) return -1;
    nvn_texdesc d; memset(&d, 0, sizeof d);
    d.target = layers > 1 ? 4 : 1; d.format = fmt; d.width = w; d.height = h; d.depth = 1; d.levels = levels ? levels : 1; d.pkg_block_h = pkg_block_h;
    nvn_texlayout L; nvn_tex_layout(&d, &L);
    int srgb = fmt >= 0x87, wrote = 0;
    for (uint32_t y = 0; y < (layers ? layers : 1); y++) for (uint32_t l = 0; l < d.levels; l++) {
        uint32_t lw = mip(w, l), lh = mip(h, l);
        if ((uint64_t)lw * lh < (uint64_t)g_min * g_min) { (*skip)++; continue; }
        s2tex_surface s; memset(&s, 0, sizeof s);
        s.width = (lw + bw - 1) / bw; s.height = (lh + bh - 1) / bh; s.depth = 1; s.bytes_per_block = 16; s.order = S2TEX_SWITCH2;
        unsigned th, td; nvn_tex_level_tiles(&d, l, &th, &td); s.block_height = th; s.block_depth = td; s.gobs_x = nvn_tex_level_gobs(&d, l);
        uint64_t base = (uint64_t)y * L.layer_size + nvn_tex_level_offset(&d, l), need = s2tex_size(&s);
        if (!need || base + need > n) { (*skip)++; continue; }
        size_t lrow = (size_t)s.width * 16, lsize = lrow * s.height; uint8_t *lin = malloc(lsize);
        if (!lin) { (*skip)++; continue; }
        if (s2tex_deswizzle(&s, data + base, (size_t)(n - base), lin, lrow, lsize)) { free(lin); (*skip)++; continue; }
        vka_key k; vka_key_of(lin, lsize, lw, lh, 1, bw, bh, srgb, &k);
        size_t size = vka_bc7_size(lw, lh, 1);
        if (vka_pack_has(&k, size)) { (*have)++; free(lin); continue; }
        uint8_t *rgba = malloc((size_t)lw * lh * 4), *bc = malloc(size);
        if (rgba && bc && !vka_decode_rgba8(lin, lw, lh, 1, bw, bh, srgb, rgba) && !vka_encode_bc7_st(rgba, lw, lh, VKA_FAST, bc)) {
            vka_pack_put(&k, bc, size); wrote++; *bytes += size;
        } else (*skip)++;
        free(rgba); free(bc); free(lin);
    }
    return wrote;
}

// Logging and the clock, as the runtime provides them
#include <windows.h>
uint64_t now_ns(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (uint64_t)((double)c.QuadPart * 1e9 / (double)f.QuadPart); }
void nxrt_logf(const char *fmt, ...) { va_list a; va_start(a, fmt); vfprintf(stderr, fmt, a); va_end(a); fputc('\n', stderr); }
// The pack folder, set in this DLL's own C runtime (the caller's environment changes may not reach it)
__declspec(dllexport) void tp_set_pack_dir(const char *dir) { _putenv_s("NXRT_VK_ASTC_PACK_DIR", dir); }
