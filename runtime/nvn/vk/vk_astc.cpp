// ASTC -> BC7 with a persistent cache (vk_astc.h; docs/m5-vulkan.md §4.3). astc-encoder 5.7.0 (Apache-2.0, third_party/astcenc) decodes, bc7enc
// (MIT / Unlicense, third_party/bc7enc) encodes. Written from their public headers.
#include "vk_astc.h"
#include "third_party/astcenc/astcenc.h"
#include "third_party/bc7enc/bc7enc.h"
#include "third_party/bc7enc/bc7decomp.h"
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shlobj.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>

extern "C" void nxrt_logf(const char *fmt, ...);
extern "C" uint64_t now_ns(void);
#define ALOG(...) nxrt_logf("[vk] " __VA_ARGS__)

// The cache key covers the encoder: change it when the BC7 settings change, and every cached file is made again
static const uint64_t VKA_VERSION = 0x5637414bull + 1;   // "KA7V" + revision
static const uint32_t MAGIC = 0x43374342u;               // "BC7C"
struct file_hdr { uint32_t magic, version, w, h, slices, quality; uint64_t size, check; };

static wchar_t g_dir[MAX_PATH];
static std::atomic<uint64_t> s_hits_high, s_hits_fast, s_miss, s_fast_ns, s_fast_tex, s_high_done, s_high_ns, s_high_tex, s_queued, s_bytes;

// ---------- Hashing (64-bit multiply-mix over 8-byte words, two seeds -> 128 bits) ----------
static inline uint64_t mix(uint64_t a, uint64_t b) { unsigned __int128 m = (unsigned __int128)a * b; return (uint64_t)m ^ (uint64_t)(m >> 64); }
static uint64_t hash64(const uint8_t *p, size_t n, uint64_t seed) {
    uint64_t h = seed ^ (n * 0x9E3779B97F4A7C15ull);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) { uint64_t a, b; memcpy(&a, p + i, 8); memcpy(&b, p + i + 8, 8); h = mix(a ^ 0xa0761d6478bd642full ^ h, b ^ 0xe7037ed1a0b428dbull); }
    uint64_t t[2] = {0, 0}; memcpy(t, p + i, n - i);
    return mix(h ^ t[0] ^ 0x8ebc6af09c88c6e3ull, t[1] ^ 0x589965cc75374cc3ull ^ n);
}
static vka_key make_key(const uint8_t *astc, size_t n, uint32_t w, uint32_t h, uint32_t s, int bw, int bh, int srgb) {
    uint64_t shape = (uint64_t)w | (uint64_t)h << 20 | (uint64_t)s << 40 | (uint64_t)bw << 52 | (uint64_t)bh << 56 | (uint64_t)srgb << 60;
    vka_key k; k.lo = hash64(astc, n, VKA_VERSION ^ shape); k.hi = hash64(astc, n, ~VKA_VERSION ^ (shape * 0x2545F4914F6CDD1Dull)); return k;
}

// the pack key of one level's ASTC bytes (tools/texprep computes keys the same way the runtime does)
extern "C" void vka_key_of(const uint8_t *astc, size_t n, uint32_t w, uint32_t h, uint32_t s, int bw, int bh, int srgb, vka_key *out) { *out = make_key(astc, n, w, h, s, bw, bh, srgb); }
static void path_of(const vka_key &k, int q, wchar_t *out) { _snwprintf(out, MAX_PATH, L"%ls\\%016llx%016llx-q%d.bc7", g_dir, (unsigned long long)k.hi, (unsigned long long)k.lo, q); }
static int cache_read(const vka_key &k, int q, uint8_t *out, size_t size, uint32_t w, uint32_t h, uint32_t s) {
    if (!g_dir[0]) return 0;
    wchar_t p[MAX_PATH]; path_of(k, q, p);
    FILE *f = _wfopen(p, L"rb"); if (!f) return 0;
    file_hdr hd; int ok = fread(&hd, sizeof hd, 1, f) == 1 && hd.magic == MAGIC && hd.version == (uint32_t)VKA_VERSION && hd.w == w && hd.h == h && hd.slices == s &&
                          hd.size == size && fread(out, 1, size, f) == size && hash64(out, size, 7) == hd.check;
    fclose(f);
    return ok;
}
static void cache_write(const vka_key &k, int q, const uint8_t *data, size_t size, uint32_t w, uint32_t h, uint32_t s) {
    if (!g_dir[0]) return;
    wchar_t p[MAX_PATH], t[MAX_PATH]; path_of(k, q, p); _snwprintf(t, MAX_PATH, L"%ls.%lu.tmp", p, GetCurrentThreadId());
    file_hdr hd = {MAGIC, (uint32_t)VKA_VERSION, w, h, s, (uint32_t)q, size, hash64(data, size, 7)};
    FILE *f = _wfopen(t, L"wb"); if (!f) return;
    int ok = fwrite(&hd, sizeof hd, 1, f) == 1 && fwrite(data, 1, size, f) == size;
    ok = fclose(f) == 0 && ok;
    if (ok && MoveFileExW(t, p, MOVEFILE_REPLACE_EXISTING)) s_bytes += size; else DeleteFileW(t);
}

// ---------- Decoding: one astc-encoder context per thread and block size ----------
struct ctx_ent { int bw, bh, srgb; astcenc_context *c; };
static thread_local std::vector<ctx_ent> t_ctx;
static astcenc_context *ctx_get(int bw, int bh, int srgb) {
    for (auto &e : t_ctx) if (e.bw == bw && e.bh == bh && e.srgb == srgb) return e.c;
    astcenc_config cfg; astcenc_context *c = nullptr;
    if (astcenc_config_init(srgb ? ASTCENC_PRF_LDR_SRGB : ASTCENC_PRF_LDR, (unsigned)bw, (unsigned)bh, 1, ASTCENC_PRE_FASTEST, ASTCENC_FLG_DECOMPRESS_ONLY, &cfg) != ASTCENC_SUCCESS) return nullptr;
    if (astcenc_context_alloc(&cfg, 1, &c, nullptr) != ASTCENC_SUCCESS) return nullptr;
    t_ctx.push_back({bw, bh, srgb, c});
    return c;
}
extern "C" int vka_decode_rgba8(const uint8_t *astc, uint32_t w, uint32_t h, uint32_t slices, int bw, int bh, int srgb, uint8_t *rgba) {
    astcenc_context *c = ctx_get(bw, bh, srgb);
    if (!c || !w || !h || !slices) return -1;
    size_t blocks = (size_t)((w + bw - 1) / bw) * ((h + bh - 1) / bh), slice_bytes = blocks * 16;
    for (uint32_t z = 0; z < slices; z++) {
        void *plane = rgba + (size_t)z * w * h * 4;
        astcenc_image img = {w, h, 1, ASTCENC_TYPE_U8, &plane};
        astcenc_swizzle sw = {ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
        if (astcenc_decompress_image(c, astc + z * slice_bytes, slice_bytes, &img, &sw, 0) != ASTCENC_SUCCESS) return -1;
        astcenc_decompress_reset(c);
        // an all-zero block is reserved, and decodes as the error colour (magenta); memory that holds nothing yet reads as nothing here
        const uint64_t *b = (const uint64_t *)(astc + z * slice_bytes); uint32_t nx = (w + bw - 1) / bw, ny = (h + bh - 1) / bh;
        for (uint32_t y = 0; y < ny; y++) for (uint32_t x = 0; x < nx; x++, b += 2) if (!b[0] && !b[1])
            for (uint32_t py = y * bh, ey = py + bh < h ? py + bh : h; py < ey; py++) { uint32_t px = x * bw; memset((uint8_t *)plane + ((size_t)py * w + px) * 4, 0, (size_t)((px + bw < w ? (uint32_t)bw : w - px)) * 4); }
    }
    return 0;
}

// ---------- Encoding: bc7enc, fast (mode 6 alone) or high quality (modes 1 and 6, partitions, uber level 1); the fast pass on a few helper threads ----------
static void params(int q, bc7enc_compress_block_params *p) {
    bc7enc_compress_block_params_init(p);
    if (q == VKA_FAST) { p->m_max_partitions = 0; p->m_uber_level = 0; p->m_try_least_squares = false; p->m_mode17_partition_estimation_filterbank = true; bc7enc_compress_block_params_init_linear_weights(p); }
    else { p->m_max_partitions = BC7ENC_MAX_PARTITIONS; p->m_uber_level = 1; p->m_try_least_squares = true; bc7enc_compress_block_params_init_linear_weights(p); }   // RGB error: the reference is the ASTC decode, texel for texel
}
// The fast pass: BC3 with stb_dxt (public domain), about ten times faster than bc7enc's fastest. MKW streams thousands of ASTC textures, and this runs where the texture is first used.
// BC3 has BC7's 16 bytes a block, so the sizes here keep the BC7 names
#define STB_DXT_STATIC
#include "third_party/stb_dxt_impl.h"
static void encode_rows_bc3(const uint8_t *rgba, uint32_t w, uint32_t h, uint32_t by0, uint32_t by1, uint8_t *out) {
    uint32_t bx_n = (w + 3) / 4; uint8_t px[64];
    for (uint32_t by = by0; by < by1; by++)
        for (uint32_t bx = 0; bx < bx_n; bx++) {
            for (uint32_t y = 0; y < 4; y++) for (uint32_t x = 0; x < 4; x++) {
                uint32_t sx = bx * 4 + x, sy = by * 4 + y; if (sx >= w) sx = w - 1; if (sy >= h) sy = h - 1;
                memcpy(px + (y * 4 + x) * 4, rgba + ((size_t)sy * w + sx) * 4, 4);
            }
            stb_compress_bc3_block(out + ((size_t)by * bx_n + bx) * 16, px, STB_DXT_NORMAL);
        }
}
static void encode_rows(const uint8_t *rgba, uint32_t w, uint32_t h, uint32_t by0, uint32_t by1, const bc7enc_compress_block_params *p, uint8_t *bc7) {
    uint32_t bx_n = (w + 3) / 4;
    uint8_t px[64];
    for (uint32_t by = by0; by < by1; by++)
        for (uint32_t bx = 0; bx < bx_n; bx++) {
            for (uint32_t y = 0; y < 4; y++) for (uint32_t x = 0; x < 4; x++) {   // edge texels repeat into the blocks past the image
                uint32_t sx = bx * 4 + x, sy = by * 4 + y; if (sx >= w) sx = w - 1; if (sy >= h) sy = h - 1;
                memcpy(px + (y * 4 + x) * 4, rgba + ((size_t)sy * w + sx) * 4, 4);
            }
            bc7enc_compress_block(bc7 + ((size_t)by * bx_n + bx) * 16, px, p);
        }
}
static std::atomic<int> g_bc7_init;
static unsigned g_helpers;
static void bc7_init() {   // std::call_once throws system_error here (static libstdc++ on mingw, threads started by the runtime)
    int e = 0;
    if (g_bc7_init.compare_exchange_strong(e, 1)) { bc7enc_compress_block_init(); unsigned n = std::thread::hardware_concurrency(); g_helpers = n >= 16 ? 3 : n >= 8 ? 1 : 0; g_bc7_init = 2; }
    else while (g_bc7_init.load() != 2) std::this_thread::yield();
}
// on the calling thread only: the sparse tile workers are threads already
extern "C" int vka_encode_bc7_st(const uint8_t *rgba, uint32_t w, uint32_t h, int quality, uint8_t *bc7) {
    (void)quality; encode_rows_bc3(rgba, w, h, 0, (h + 3) / 4, bc7); return 0;
}
static thread_local int t_one_thread;   // the prefetch workers encode on their own thread (they are workers already)
extern "C" int vka_encode_bc7(const uint8_t *rgba, uint32_t w, uint32_t h, int quality, uint8_t *bc7) {
    bc7_init();
    uint32_t rows = (h + 3) / 4;
    if (g_helpers == 0 || t_one_thread || (uint64_t)w * h < 512 * 512) { encode_rows_bc3(rgba, w, h, 0, rows, bc7); return 0; }
    // The fast pass runs where the texture is first used: split over at most 4 threads (this PC must not see long all-core loads).
    // A helper that cannot be started (no memory for a thread: std::system_error) leaves its rows to this thread instead of ending the game
    unsigned n = g_helpers + 1; std::vector<std::thread> th; std::vector<unsigned> left;
    for (unsigned i = 1; i < n; i++) {
        try { th.emplace_back([=] { encode_rows_bc3(rgba, w, h, rows * i / n, rows * (i + 1) / n, bc7); }); }
        catch (...) { left.push_back(i); }
    }
    encode_rows_bc3(rgba, w, h, 0, rows / n, bc7);
    for (unsigned i : left) encode_rows_bc3(rgba, w, h, rows * i / n, rows * (i + 1) / n, bc7);
    for (auto &t : th) t.join();
    return 0;
}
extern "C" int vka_decode_bc7_block(const uint8_t *block, uint8_t rgba[64]) { return bc7decomp::unpack_bc7(block, (bc7decomp::color_rgba *)rgba) ? 0 : -1; }
extern "C" size_t vka_bc7_size(uint32_t w, uint32_t h, uint32_t slices) { return (size_t)((w + 3) / 4) * ((h + 3) / 4) * 16 * slices; }

// ---------- The pack: every transcoded level and tile in one file, so a texture is transcoded once, ever ----------
// One file per texture stopped the replay (see vka_get_bc7). Here astc_cache\pack.bin takes the data and index.bin 32 bytes per entry, appended after the data, so a killed game loses at most its last entry.
// Keyed by the ASTC bytes, shape and encoder version: nothing here knows the game. NXRT_VK_ASTC_PACK=0 turns it off, NXRT_VK_ASTC_PACK_MB is the most it holds (default 32000),
// NXRT_VK_ASTC_PACK_DIR another folder (default astc_cache beside the exe). Format 2 (pack2.bin / index2.bin, 2026-11-09): a fully opaque level keeps only the colour half of
// each BC3 block (pad bit 0) and is read back with an alpha block of 255s, which decodes to the same texels, so a whole game's textures fit (MKW: ~36 GB as BC3, ~20 GB so).
// The old pack.bin is left alone (an older build reads it). tools/texprep fills it ahead of play from the game's BNTX files.
#include <unordered_map>
struct pk_ent { uint64_t lo, hi, off; uint32_t size, pad; };   // size: the level's BC3 bytes; pad bit 0: stored compact (size / 2 bytes)
static inline uint64_t pk_stored(const pk_ent &x) { return (x.pad & 1) ? x.size / 2 : x.size; }
static std::mutex g_pk_mu; static std::unordered_map<uint64_t, pk_ent> g_pk; static HANDLE g_pk_data = INVALID_HANDLE_VALUE, g_pk_index = INVALID_HANDLE_VALUE; static uint64_t g_pk_end, g_pk_max; static int g_pk_state;
static std::atomic<uint64_t> s_pk_hits, s_pk_miss, s_pk_read_ns, s_pk_hit_b;
static void pk_open() {   // g_pk_mu held
    g_pk_state = -1;
    const char *e = getenv("NXRT_VK_ASTC_PACK"); if (e && *e == '0') return;
    e = getenv("NXRT_VK_ASTC_PACK_MB"); g_pk_max = (uint64_t)(e && atoi(e) > 0 ? atoi(e) : 32000) << 20;
    wchar_t d[MAX_PATH], p[MAX_PATH];
    if ((e = getenv("NXRT_VK_ASTC_PACK_DIR")) && *e) MultiByteToWideChar(CP_UTF8, 0, e, -1, d, MAX_PATH);
    else { GetModuleFileNameW(NULL, d, MAX_PATH); wchar_t *s = wcsrchr(d, L'\\'); if (s) *s = 0; wcsncat(d, L"\\astc_cache", MAX_PATH - wcslen(d) - 1); }
    CreateDirectoryW(d, NULL);
    _snwprintf(p, MAX_PATH, L"%ls\\pack2.bin", d);
    g_pk_data = CreateFileW(p, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    _snwprintf(p, MAX_PATH, L"%ls\\index2.bin", d);
    g_pk_index = CreateFileW(p, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (g_pk_data == INVALID_HANDLE_VALUE || g_pk_index == INVALID_HANDLE_VALUE) { ALOG("ASTC pack: cannot open %ls (another game running?), transcoding as it comes", d); return; }
    LARGE_INTEGER dn, in; GetFileSizeEx(g_pk_data, &dn); GetFileSizeEx(g_pk_index, &in);
    size_t n = (size_t)in.QuadPart / sizeof(pk_ent); std::vector<pk_ent> v(n); DWORD got = 0;
    if (n) ReadFile(g_pk_index, v.data(), (DWORD)(n * sizeof(pk_ent)), &got, NULL);
    n = got / sizeof(pk_ent); g_pk.reserve(n * 2 + 1024); g_pk_end = 0;
    for (size_t i = 0; i < n; i++) { const pk_ent &x = v[i]; if (x.off + pk_stored(x) > (uint64_t)dn.QuadPart) continue; g_pk[x.lo] = x; if (x.off + pk_stored(x) > g_pk_end) g_pk_end = x.off + pk_stored(x); }
    LARGE_INTEGER z; z.QuadPart = (LONGLONG)n * sizeof(pk_ent); SetFilePointerEx(g_pk_index, z, NULL, FILE_BEGIN); SetEndOfFile(g_pk_index);
    g_pk_state = 1;
    ALOG("ASTC pack: %ls holds %llu pieces, %llu MB", d, (unsigned long long)g_pk.size(), (unsigned long long)(g_pk_end >> 20));
}
// ---------- Prefetch: levels transcoded (or read from the pack) on a worker when the game registers the texture, kept in memory until its upload takes them ----------
// The upload then copies instead of reading the pack or transcoding on the replay thread, which holds the texture lock meanwhile (vk_tex.c vkt_resolve). Keyed by
// the ASTC bytes like the pack: a level whose bytes changed after the prefetch simply misses. Bounded (oldest dropped first): 16 GB machines run near their limit.
#include <deque>
struct pf_ent { uint64_t hi; std::vector<uint8_t> data; };
static std::mutex g_pf_mu; static std::unordered_map<uint64_t, pf_ent> g_pf; static std::deque<uint64_t> g_pf_order; static size_t g_pf_bytes;
static std::atomic<uint64_t> s_pf_done, s_pf_hit, s_pf_skip;
static int pk_read(const vka_key &k, uint8_t *out, size_t size);
static int transcode(const uint8_t *astc, uint32_t w, uint32_t h, uint32_t s, int bw, int bh, int srgb, int q, uint8_t *bc7);
static int pf_take(const vka_key &k, uint8_t *out, size_t size) {
    static int on = -1; if (on < 0) { const char *e = getenv("NXRT_VK_PF_TAKE"); on = !e || *e != '0'; } if (!on) return 0;   // NXRT_VK_PF_TAKE=0: uploads never take prefetched levels (for the BC1 bug hunt)
    std::lock_guard<std::mutex> l(g_pf_mu);
    auto it = g_pf.find(k.lo); if (it == g_pf.end() || it->second.hi != k.hi || it->second.data.size() != size) return 0;
    memcpy(out, it->second.data.data(), size); g_pf_bytes -= size; g_pf.erase(it); s_pf_hit++; return 1;
}
extern "C" int vkt_bc3_opaque(const uint8_t *src, size_t size);   // vk_tex.c
// 1: ready and opaque, 0: ready with some alpha, -1: not looked at (already held, or it failed)
extern "C" int vka_take(const vka_key *key, uint8_t *out, size_t size) { return pf_take(*key, out, size); }
extern "C" int vka_has(const vka_key *key, size_t size) {
    std::lock_guard<std::mutex> l(g_pf_mu); auto it = g_pf.find(key->lo); return it != g_pf.end() && it->second.hi == key->hi && it->second.data.size() == size; }
extern "C" int vka_prefetch(const uint8_t *astc, uint32_t w, uint32_t h, int bw, int bh, int srgb, vka_key *kout) {
    size_t n = (size_t)((w + bw - 1) / bw) * ((h + bh - 1) / bh) * 16, size = vka_bc7_size(w, h, 1);
    vka_key k = make_key(astc, n, w, h, 1, bw, bh, srgb);
    if (kout) *kout = k;
    { std::lock_guard<std::mutex> l(g_pf_mu); if (g_pf.count(k.lo)) { s_pf_skip++; return -1; } }
    std::vector<uint8_t> out(size);
    if (!pk_read(k, out.data(), size)) {
        t_one_thread = 1; int bad = transcode(astc, w, h, 1, bw, bh, srgb, VKA_FAST, out.data()); t_one_thread = 0;
        if (bad) return -2;
        vka_pack_put(&k, out.data(), size);
    }
    int opaque = vkt_bc3_opaque(out.data(), size);
    static size_t cap = 0; if (!cap) { const char *e = getenv("NXRT_VK_PREFETCH_MB"); cap = (size_t)(e && atoi(e) > 0 ? atoi(e) : 256) << 20; }
    std::lock_guard<std::mutex> l(g_pf_mu);
    if (g_pf.count(k.lo)) return opaque;
    g_pf_bytes += size; g_pf[k.lo] = pf_ent{k.hi, std::move(out)}; g_pf_order.push_back(k.lo); s_pf_done++;
    while (g_pf_bytes > cap && !g_pf_order.empty()) { auto it = g_pf.find(g_pf_order.front()); g_pf_order.pop_front(); if (it != g_pf.end()) { g_pf_bytes -= it->second.data.size(); g_pf.erase(it); } }
    if (g_pf_order.size() > 4 * g_pf.size() + 1024) { std::deque<uint64_t> o; for (uint64_t x : g_pf_order) if (g_pf.count(x)) o.push_back(x); g_pf_order.swap(o); }   // taken entries leave stale keys behind
    return opaque;
}
extern "C" void vka_prefetch_stats(uint64_t *done, uint64_t *hits, uint64_t *skipped, uint64_t *bytes) {
    *done = s_pf_done; *hits = s_pf_hit; *skipped = s_pf_skip; std::lock_guard<std::mutex> l(g_pf_mu); *bytes = g_pf_bytes; }
extern "C" int vka_pack_get(const uint8_t *astc, uint32_t w, uint32_t h, int bw, int bh, int srgb, uint8_t *out, size_t size, vka_key *key) {
    size_t n = (size_t)((w + bw - 1) / bw) * ((h + bh - 1) / bh) * 16;
    *key = make_key(astc, n, w, h, 1, bw, bh, srgb);
    if (pf_take(*key, out, size)) return 1;
    return pk_read(*key, out, size);
}
static int pk_read(const vka_key &kk, uint8_t *out, size_t size) {
    const vka_key *key = &kk;
    pk_ent x; int found = 0;
    { std::lock_guard<std::mutex> l(g_pk_mu); if (!g_pk_state) pk_open(); if (g_pk_state < 0) return 0;
      auto it = g_pk.find(key->lo); if (it != g_pk.end() && it->second.hi == key->hi && it->second.size == size) { x = it->second; found = 1; } }
    if (!found) { s_pk_miss++; return 0; }
    uint64_t t0 = now_ns(); OVERLAPPED ov; memset(&ov, 0, sizeof ov); ov.Offset = (DWORD)x.off; ov.OffsetHigh = (DWORD)(x.off >> 32); DWORD got = 0;
    DWORD want = (DWORD)pk_stored(x);
    if (!ReadFile(g_pk_data, out, want, &got, &ov) || got != want) return 0;
    if (x.pad & 1) {   // compact: the colour halves are in the first size / 2 bytes; spread them back from the end so nothing is overwritten before it is read
        size_t nb = size / 16;
        for (size_t b = nb; b-- > 0; ) { uint8_t c[8]; memcpy(c, out + b * 8, 8); uint8_t *o = out + b * 16;
            o[0] = 255; o[1] = 255; memset(o + 2, 0, 6); memcpy(o + 8, c, 8); }
    }
    s_pk_hits++; s_pk_hit_b += size; s_pk_read_ns += now_ns() - t0; return 1;
}
// 1 when the pack holds this level (tools/texprep: skip what is already there)
extern "C" int vka_pack_has(const vka_key *key, size_t size) {
    std::lock_guard<std::mutex> l(g_pk_mu); if (!g_pk_state) pk_open(); if (g_pk_state < 0) return 0;
    auto it = g_pk.find(key->lo); return it != g_pk.end() && it->second.hi == key->hi && it->second.size == size;
}
extern "C" int vka_pack_ready(void) { std::lock_guard<std::mutex> l(g_pk_mu); if (!g_pk_state) pk_open(); return g_pk_state > 0; }
extern "C" uint64_t vka_pack_bytes(void) { std::lock_guard<std::mutex> l(g_pk_mu); return g_pk_end; }
extern "C" void vka_pack_put(const vka_key *key, const uint8_t *data, size_t size) {
    // a fully opaque level goes in compact (the colour halves), made before the lock: the encoder writes alpha blocks of 255 the BC1 check reads as opaque
    std::vector<uint8_t> half; int compact = size >= 16 && !(size % 16) && vkt_bc3_opaque(data, size);
    if (compact) { half.resize(size / 2); for (size_t b = 0; b < size / 16; b++) memcpy(half.data() + b * 8, data + b * 16 + 8, 8); }
    const uint8_t *src = compact ? half.data() : data; DWORD n = (DWORD)(compact ? size / 2 : size);
    std::lock_guard<std::mutex> l(g_pk_mu);
    if (!g_pk_state) pk_open();
    if (g_pk_state <= 0 || g_pk_end + n > g_pk_max || g_pk.count(key->lo)) return;
    pk_ent x; x.lo = key->lo; x.hi = key->hi; x.off = g_pk_end; x.size = (uint32_t)size; x.pad = compact ? 1 : 0;
    OVERLAPPED ov; memset(&ov, 0, sizeof ov); ov.Offset = (DWORD)x.off; ov.OffsetHigh = (DWORD)(x.off >> 32); DWORD put = 0;
    if (!WriteFile(g_pk_data, src, n, &put, &ov) || put != n) return;
    DWORD p2 = 0; if (!WriteFile(g_pk_index, &x, sizeof x, &p2, NULL) || p2 != sizeof x) return;
    g_pk[x.lo] = x; g_pk_end += n;
}
extern "C" void vka_pack_stats(uint64_t *hits, uint64_t *misses, uint64_t *hit_bytes, uint64_t *read_ns) { *hits = s_pk_hits; *misses = s_pk_miss; *hit_bytes = s_pk_hit_b; *read_ns = s_pk_read_ns; }

// ---------- The high-quality worker ----------
struct job { vka_key key; std::vector<uint8_t> astc; uint32_t w, h, s; int bw, bh, srgb; };
struct ready { vka_key key; std::vector<uint8_t> bc7; };
static std::mutex g_mu; static std::condition_variable g_cv, g_idle_cv;
static std::deque<job> g_jobs; static std::deque<ready> g_ready; static int g_busy; static bool g_stop; static std::thread g_worker;

static int transcode(const uint8_t *astc, uint32_t w, uint32_t h, uint32_t s, int bw, int bh, int srgb, int q, uint8_t *bc7) {
    std::vector<uint8_t> rgba((size_t)w * h * 4 * s);
    if (vka_decode_rgba8(astc, w, h, s, bw, bh, srgb, rgba.data())) return -1;
    size_t per = vka_bc7_size(w, h, 1);
    for (uint32_t z = 0; z < s; z++) vka_encode_bc7(rgba.data() + (size_t)z * w * h * 4, w, h, q, bc7 + z * per);
    return 0;
}
static void worker_main() {
    SetThreadDescription(GetCurrentThread(), L"vk astc->bc7");
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    std::unique_lock<std::mutex> lk(g_mu);
    for (;;) {
        g_cv.wait(lk, [] { return g_stop || !g_jobs.empty(); });
        if (g_stop) break;
        job j = std::move(g_jobs.front()); g_jobs.pop_front(); g_busy = 1;
        lk.unlock();
        uint64_t t0 = now_ns();
        std::vector<uint8_t> out(vka_bc7_size(j.w, j.h, j.s));
        int ok = !transcode(j.astc.data(), j.w, j.h, j.s, j.bw, j.bh, j.srgb, VKA_HIGH, out.data());
        if (ok) cache_write(j.key, VKA_HIGH, out.data(), out.size(), j.w, j.h, j.s);
        s_high_ns += now_ns() - t0; s_high_done++; s_high_tex += (uint64_t)j.w * j.h * j.s;
        lk.lock();
        if (ok) g_ready.push_back({j.key, std::move(out)});
        g_busy = 0; g_idle_cv.notify_all();
    }
}

extern "C" int vka_get_bc7(const uint8_t *astc, uint32_t w, uint32_t h, uint32_t s, int bw, int bh, int srgb, uint8_t *bc7, vka_key *key_out) {
    size_t n = (size_t)((w + bw - 1) / bw) * ((h + bh - 1) / bh) * 16 * s, out = vka_bc7_size(w, h, s);
    // One cache file per level and layer: MKW streams thousands, and on a hard disk the file operations alone stopped the replay for 25 s a frame.
    // NXRT_VK_ASTC_CACHE=1 brings the cache and the high-quality second pass back. Without it the fast pass is final
    static int use_cache = -1; if (use_cache < 0) { const char *e = getenv("NXRT_VK_ASTC_CACHE"); use_cache = e && *e == '1'; }
    if (!use_cache) {
        if (key_out) *key_out = vka_key{0, 0};
        // A 2D level goes through the pack like the bigger ones (vk_sparse.inc sp_tex_job_run): MKW streams hundreds of small ASTC textures a second, and each was
        // decoded and encoded again on the replay thread every time it came back. The same fast encoder fills the pack, so the bytes are the same. Off by default
        // (NXRT_VK_SMALL_PACK=1 on): measured 2026-11-09, the pack's random reads and writes on the replay thread cost more than encoding these small levels
        static int small_pack = -1; if (small_pack < 0) { const char *e = getenv("NXRT_VK_SMALL_PACK"); small_pack = e && *e == '1'; }
        vka_key pk; if (s == 1 && small_pack && vka_pack_get(astc, w, h, bw, bh, srgb, bc7, out, &pk)) return VKA_HIGH;
        uint64_t t0 = now_ns(); if (transcode(astc, w, h, s, bw, bh, srgb, VKA_FAST, bc7)) return -1; s_fast_ns += now_ns() - t0; s_fast_tex += (uint64_t)w * h * s; s_miss++;
        if (s == 1 && small_pack) vka_pack_put(&pk, bc7, out);
        return VKA_HIGH; }
    vka_key k = make_key(astc, n, w, h, s, bw, bh, srgb);
    if (key_out) *key_out = k;
    if (cache_read(k, VKA_HIGH, bc7, out, w, h, s)) { s_hits_high++; return VKA_HIGH; }
    int fast = cache_read(k, VKA_FAST, bc7, out, w, h, s);
    if (fast) s_hits_fast++;
    else {
        s_miss++;
        uint64_t t0 = now_ns();
        if (transcode(astc, w, h, s, bw, bh, srgb, VKA_FAST, bc7)) return -1;
        s_fast_ns += now_ns() - t0; s_fast_tex += (uint64_t)w * h * s;
        cache_write(k, VKA_FAST, bc7, out, w, h, s);
    }
    {   // the high-quality pass, once per key
        std::lock_guard<std::mutex> lk(g_mu);
        bool queued = false; for (auto &j : g_jobs) if (j.key.lo == k.lo && j.key.hi == k.hi) queued = true;
        if (!queued && g_worker.joinable()) { g_jobs.push_back({k, std::vector<uint8_t>(astc, astc + n), w, h, s, bw, bh, srgb}); s_queued++; g_cv.notify_one(); }
    }
    return VKA_FAST;
}

extern "C" int vka_poll(vka_ready_fn fn, void *user, int max) {
    int done = 0;
    while (done < max) {
        ready r;
        { std::lock_guard<std::mutex> lk(g_mu); if (g_ready.empty()) break; r = std::move(g_ready.front()); g_ready.pop_front(); }
        fn(r.key, r.bc7.data(), r.bc7.size(), user); done++;
    }
    return done;
}
extern "C" void vka_wait_idle(void) { std::unique_lock<std::mutex> lk(g_mu); g_idle_cv.wait(lk, [] { return g_jobs.empty() && !g_busy; }); }

extern "C" int vka_init(const char *dir) {
    if (dir && *dir) MultiByteToWideChar(CP_UTF8, 0, dir, -1, g_dir, MAX_PATH);
    else {
        const char *e = getenv("NXRT_VK_CACHE_DIR"); wchar_t base[MAX_PATH] = L"";
        if (e && *e) MultiByteToWideChar(CP_UTF8, 0, e, -1, base, MAX_PATH);
        else { wchar_t *la = nullptr; if (SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &la) == S_OK) { _snwprintf(base, MAX_PATH, L"%ls\\pokopia-native", la); CreateDirectoryW(base, nullptr); _snwprintf(base, MAX_PATH, L"%ls\\pokopia-native\\vulkan", la); CoTaskMemFree(la); } }
        if (base[0]) { CreateDirectoryW(base, nullptr); _snwprintf(g_dir, MAX_PATH, L"%ls\\astc-bc7", base); }
    }
    if (g_dir[0]) CreateDirectoryW(g_dir, nullptr);
    { std::lock_guard<std::mutex> lk(g_mu); g_stop = false; }
    if (!g_worker.joinable()) g_worker = std::thread(worker_main);
    ALOG("ASTC -> BC7 cache %ls (fast pass at first use, high quality in the background)", g_dir[0] ? g_dir : L"(none)");
    return 0;
}
extern "C" void vka_shutdown(void) {
    { std::lock_guard<std::mutex> lk(g_mu); g_stop = true; g_jobs.clear(); }
    g_cv.notify_all();
    if (g_worker.joinable()) g_worker.join();
    std::lock_guard<std::mutex> lk(g_mu); g_ready.clear();
}
extern "C" void vka_stats(vka_stats_t *s) {
    s->hits_high = s_hits_high; s->hits_fast = s_hits_fast; s->misses = s_miss; s->fast_ns = s_fast_ns; s->fast_texels = s_fast_tex;
    s->high_done = s_high_done; s->high_ns = s_high_ns; s->high_texels = s_high_tex; s->queued = s_queued; s->cache_bytes = s_bytes;
}
