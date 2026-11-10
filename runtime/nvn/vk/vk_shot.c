// Screenshots (vk_shot.h). PNG from the specification (RFC 2083): IHDR, one IDAT holding a zlib stream of stored deflate blocks (RFC 1950 / 1951), IEND.
#include "vk_shot.h"
#include "vk_tex.h"
#include "vk_submit.h"
#include "vk_present.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

const char *nxrt_log_path(void);

static uint64_t g_at[64]; static int g_nat; static uint64_t g_every, g_count;
static char g_dir[1024], g_file[1024]; static int g_window;
static SRWLOCK g_req_lock = SRWLOCK_INIT; static char g_req[8][1024]; static int g_nreq;   // vks_shot_request (DM_CTL shot): taken in order, one per present

// ---------- PNG ----------
static uint32_t crc_tab[256];
static uint32_t crc32_of(uint32_t c, const uint8_t *p, size_t n) {
    if (!crc_tab[1]) for (uint32_t i = 0; i < 256; i++) { uint32_t v = i; for (int k = 0; k < 8; k++) v = v & 1 ? 0xEDB88320u ^ (v >> 1) : v >> 1; crc_tab[i] = v; }
    c = ~c; for (size_t i = 0; i < n; i++) c = crc_tab[(c ^ p[i]) & 0xff] ^ (c >> 8);
    return ~c;
}
static void be32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16); p[2] = (uint8_t)(v >> 8); p[3] = (uint8_t)v; }
static int chunk(FILE *f, const char *type, const uint8_t *data, uint32_t n) {
    uint8_t h[8]; be32(h, n); memcpy(h + 4, type, 4);
    uint32_t c = crc32_of(0, (const uint8_t *)type, 4); c = crc32_of(c, data, n);
    uint8_t t[4]; be32(t, c);
    return fwrite(h, 1, 8, f) == 8 && (!n || fwrite(data, 1, n, f) == n) && fwrite(t, 1, 4, f) == 4 ? 0 : -1;
}
int vks_png_write(const char *path, const uint8_t *rgba, int w, int h) {
    if (w <= 0 || h <= 0) return -1;
    size_t row = (size_t)w * 4 + 1, raw = row * (size_t)h, nblk = (raw + 65534) / 65535, zn = 2 + raw + nblk * 5 + 4;
    uint8_t *z = malloc(zn); if (!z) return -1;
    size_t o = 0; z[o++] = 0x78; z[o++] = 0x01;
    uint32_t a = 1, b = 0; size_t done = 0;
    uint8_t *line = malloc(row);
    // the filtered rows (filter 0) streamed into stored blocks
    size_t blk_left = 0;
    for (int y = 0; y < h; y++) {
        line[0] = 0; memcpy(line + 1, rgba + (size_t)y * w * 4, (size_t)w * 4);
        for (size_t i = 0; i < row; i++) {
            if (!blk_left) {
                size_t n = raw - done < 65535 ? raw - done : 65535; blk_left = n;
                z[o++] = done + n == raw ? 1 : 0; z[o++] = (uint8_t)n; z[o++] = (uint8_t)(n >> 8); z[o++] = (uint8_t)~n; z[o++] = (uint8_t)(~n >> 8);
            }
            z[o++] = line[i]; a = (a + line[i]) % 65521; b = (b + a) % 65521; blk_left--; done++;
        }
    }
    free(line);
    be32(z + o, b << 16 | a); o += 4;
    wchar_t wp[1100]; if (!MultiByteToWideChar(CP_UTF8, 0, path, -1, wp, 1100)) { free(z); return -1; }
    FILE *f = _wfopen(wp, L"wb"); if (!f) { free(z); return -1; }
    static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    uint8_t ih[13]; be32(ih, (uint32_t)w); be32(ih + 4, (uint32_t)h); ih[8] = 8; ih[9] = 6; ih[10] = ih[11] = ih[12] = 0;
    int rc = fwrite(sig, 1, 8, f) == 8 && !chunk(f, "IHDR", ih, 13) && !chunk(f, "IDAT", z, (uint32_t)o) && !chunk(f, "IEND", NULL, 0) ? 0 : -1;
    fclose(f); free(z);
    return rc;
}

// ---------- Formats → RGBA8 (the stored values; sRGB bytes stay sRGB, as the window shows them) ----------
static uint8_t f2u8(float v) { if (!(v > 0)) return 0; if (v >= 1) return 255; return (uint8_t)lrintf(v * 255.0f); }
static float half(uint16_t h) {
    uint32_t s = (uint32_t)(h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023, f;
    if (!e) { if (!m) f = s; else { e = 127 - 15 + 1; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; f = s | e << 23 | m << 13; } }
    else if (e == 31) f = s | 0x7f800000u | m << 13; else f = s | (e + 127 - 15) << 23 | m << 13;
    float r; memcpy(&r, &f, 4); return r;
}
int vks_to_rgba8(VkFormat f, const uint8_t *s, size_t n, uint8_t *d) {
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: memcpy(d, s, n * 4); return 0;
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: for (size_t i = 0; i < n; i++) { d[4 * i] = s[4 * i + 2]; d[4 * i + 1] = s[4 * i + 1]; d[4 * i + 2] = s[4 * i]; d[4 * i + 3] = s[4 * i + 3]; } return 0;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: for (size_t i = 0; i < n; i++) { uint32_t v; memcpy(&v, s + 4 * i, 4);
            d[4 * i] = (uint8_t)((v & 1023) >> 2); d[4 * i + 1] = (uint8_t)((v >> 10 & 1023) >> 2); d[4 * i + 2] = (uint8_t)((v >> 20 & 1023) >> 2); d[4 * i + 3] = (uint8_t)((v >> 30) * 85); } return 0;
    case VK_FORMAT_R16G16B16A16_SFLOAT: for (size_t i = 0; i < n; i++) for (int c = 0; c < 4; c++) { uint16_t h; memcpy(&h, s + 8 * i + 2 * c, 2); d[4 * i + c] = f2u8(half(h)); } return 0;
    case VK_FORMAT_R16G16B16A16_UNORM: for (size_t i = 0; i < n; i++) for (int c = 0; c < 4; c++) { uint16_t h; memcpy(&h, s + 8 * i + 2 * c, 2); d[4 * i + c] = (uint8_t)(h >> 8); } return 0;
    case VK_FORMAT_R32G32B32A32_SFLOAT: for (size_t i = 0; i < n; i++) for (int c = 0; c < 4; c++) { float v; memcpy(&v, s + 16 * i + 4 * c, 4); d[4 * i + c] = f2u8(v); } return 0;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: for (size_t i = 0; i < n; i++) { uint32_t v; memcpy(&v, s + 4 * i, 4);   // 5-bit exponents, 6 / 6 / 5-bit mantissas
            uint32_t c[3] = {v & 0x7ff, v >> 11 & 0x7ff, v >> 22}; int mb[3] = {6, 6, 5};
            for (int k = 0; k < 3; k++) { uint32_t e = c[k] >> mb[k], m = c[k] & ((1u << mb[k]) - 1); float f = e ? ldexpf(1.0f + (float)m / (float)(1u << mb[k]), (int)e - 15) : ldexpf((float)m, -14 - mb[k]); d[4 * i + k] = f2u8(f); }
            d[4 * i + 3] = 255; } return 0;
    case VK_FORMAT_R32_SFLOAT: for (size_t i = 0; i < n; i++) { float v; memcpy(&v, s + 4 * i, 4); d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = f2u8(v); d[4 * i + 3] = 255; } return 0;
    case VK_FORMAT_R16_SFLOAT: for (size_t i = 0; i < n; i++) { uint16_t h; memcpy(&h, s + 2 * i, 2); d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = f2u8(half(h)); d[4 * i + 3] = 255; } return 0;
    case VK_FORMAT_R8_UNORM: for (size_t i = 0; i < n; i++) { d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = s[i]; d[4 * i + 3] = 255; } return 0;
    case VK_FORMAT_R8G8_UNORM: for (size_t i = 0; i < n; i++) { d[4 * i] = s[2 * i]; d[4 * i + 1] = s[2 * i + 1]; d[4 * i + 2] = 0; d[4 * i + 3] = 255; } return 0;
    case VK_FORMAT_R16G16_SFLOAT: for (size_t i = 0; i < n; i++) { uint16_t h; memcpy(&h, s + 4 * i, 2); d[4 * i] = f2u8(half(h)); memcpy(&h, s + 4 * i + 2, 2); d[4 * i + 1] = f2u8(half(h)); d[4 * i + 2] = 0; d[4 * i + 3] = 255; } return 0;
    case VK_FORMAT_R5G6B5_UNORM_PACK16: for (size_t i = 0; i < n; i++) { uint16_t v; memcpy(&v, s + 2 * i, 2); d[4 * i] = (uint8_t)((v >> 11) * 255 / 31); d[4 * i + 1] = (uint8_t)((v >> 5 & 63) * 255 / 63); d[4 * i + 2] = (uint8_t)((v & 31) * 255 / 31); d[4 * i + 3] = 255; } return 0;
    default: return -1;
    }
}

int vks_shot_capture(const tex_t *t, const int crop[4], int flip, uint8_t **rgba, int *w, int *h) {
    *rgba = NULL;
    vkt_image *im = t ? vkt_by_serial(t->serial) : NULL;
    if (!im || !im->content) return -1;
    NVNcopyRegion r = {0, 0, 0, (int)im->w, (int)im->h, 1};
    if (crop && crop[2] > 0 && crop[3] > 0 && crop[0] >= 0 && crop[1] >= 0 && (uint32_t)(crop[0] + crop[2]) <= im->w && (uint32_t)(crop[1] + crop[3]) <= im->h) { r.xoffset = crop[0]; r.yoffset = crop[1]; r.width = crop[2]; r.height = crop[3]; }
    size_t n = (size_t)r.width * r.height, bytes = n * im->fi->vbpb;
    uint8_t *raw = malloc(bytes), *out = malloc(n * 4);
    if (!raw || !out) { free(raw); free(out); return -1; }
    vks_finish();   // the frame's GPU work first
    if (vkt_readback_region(im, 0, 0, &r, raw, bytes) || vks_to_rgba8(im->fi->vk, raw, n, out)) {
        static int nf; if (nf++ < 4) VKRED("screenshot: the presented image (format %s) could not be read back", im->fi->name);
        free(raw); free(out); return -1;
    }
    free(raw);
    if (flip) for (int y = 0; y < r.height / 2; y++) { uint8_t *a = out + (size_t)y * r.width * 4, *b = out + (size_t)(r.height - 1 - y) * r.width * 4; for (int x = 0; x < r.width * 4; x++) { uint8_t c = a[x]; a[x] = b[x]; b[x] = c; } }
    *rgba = out; *w = r.width; *h = r.height;
    return 0;
}

static void shot_path(char *out, size_t n, uint64_t frame, const char *suffix) {
    const char *lp = nxrt_log_path();
    if (*g_dir) snprintf(out, n, "%s\\shot-%06llu%s.png", g_dir, (unsigned long long)frame, suffix);
    else snprintf(out, n, "%s.shot-%06llu%s.png", lp && *lp ? lp : "nxrt", (unsigned long long)frame, suffix);
}
static int origin_ll(void) { extern int vk_origin_lower_left(void); return vk_origin_lower_left(); }

void vks_shot_present(const tex_t *t, uint64_t frame, const int crop[4]) {
    int want = 0; char path[1100] = "";
    for (int i = 0; i < g_nat; i++) want |= g_at[i] == frame;
    if (g_every && frame % g_every == 0) want = 1;
    if (*g_file) {
        wchar_t wf[1100]; MultiByteToWideChar(CP_UTF8, 0, g_file, -1, wf, 1100);
        if (GetFileAttributesW(wf) != INVALID_FILE_ATTRIBUTES) {
            want = 1;
            FILE *f = _wfopen(wf, L"rb");
            if (f) { size_t k = fread(path, 1, sizeof path - 1, f); path[k] = 0; fclose(f); for (char *q = path; *q; q++) if (*q == '\r' || *q == '\n') { *q = 0; break; } }
            DeleteFileW(wf);
        }
    }
    if (!want && g_nreq) {
        AcquireSRWLockExclusive(&g_req_lock);
        if (g_nreq) { want = 1; snprintf(path, sizeof path, "%s", g_req[0]); memmove(g_req[0], g_req[1], (size_t)--g_nreq * sizeof g_req[0]); }
        ReleaseSRWLockExclusive(&g_req_lock);
    }
    if (!want) return;
    if (!*path) shot_path(path, sizeof path, frame, "");
    uint8_t *px; int w, h;
    uint64_t t0 = now_ns();
    if (vks_shot_capture(t, crop, origin_ll(), &px, &w, &h)) { VKLOG("screenshot of present %llu: nothing to read back", (unsigned long long)frame); return; }
    for (size_t i = 0; i < (size_t)w * h; i++) px[4 * i + 3] = 255;   // opaque, as the window shows it: the game leaves alpha 0 under some HUD panels
    int rc = vks_png_write(path, px, w, h);
    free(px);
    g_count++;
    VKLOG("screenshot: present %llu (%dx%d) %s %s (%.1f ms)", (unsigned long long)frame, w, h, rc ? "could not be written to" : "written to", path, (now_ns() - t0) / 1e6);
    if (g_window) {   // the window as shown: the next present's swapchain picture, on a thread of its own (it waits for the present thread)
        uint32_t *wp = NULL; int ww, wh, bgr;
        if (!vkp_capture_next(&wp, &ww, &wh, &bgr, 2000)) {
            uint8_t *o = malloc((size_t)ww * wh * 4);
            for (size_t i = 0; i < (size_t)ww * wh; i++) { uint32_t c = wp[i]; o[4 * i] = (uint8_t)(bgr ? c >> 16 : c); o[4 * i + 1] = (uint8_t)(c >> 8); o[4 * i + 2] = (uint8_t)(bgr ? c : c >> 16); o[4 * i + 3] = 255; }
            char wpth[1200]; snprintf(wpth, sizeof wpth, "%.*s.window.png", (int)(strlen(path) > 4 ? strlen(path) - 4 : strlen(path)), path);
            if (!vks_png_write(wpth, o, ww, wh)) VKLOG("screenshot: the window (%dx%d) written to %s", ww, wh, wpth);
            free(o); free(wp);
        }
    }
}
uint64_t vks_shot_count(void) { return g_count; }
void vks_shot_request(const char *path) {
    AcquireSRWLockExclusive(&g_req_lock);
    int ok = g_nreq < 8; if (ok) snprintf(g_req[g_nreq++], sizeof g_req[0], "%s", path && *path ? path : "");
    ReleaseSRWLockExclusive(&g_req_lock);
    VKLOG("screenshot of the next present %s %s", ok ? "requested:" : "not requested (8 are waiting already):", path && *path ? path : "beside the log");
}
// A texture as it is now (the GPU finished first), level 0, layer z, as PNG: the replay's texture dumps (NXRT_VK_TEXDUMP)
// Compressed images as DDS (DX10 header), the blocks as they are: <path>.dds
static int dump_dds(vkt_image *im, unsigned layer, const char *path) {
    uint32_t dxgi = 0;
    switch (im->fi->vk) {
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: dxgi = 71; break;  case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: dxgi = 72; break;
    case VK_FORMAT_BC2_UNORM_BLOCK: dxgi = 74; break;       case VK_FORMAT_BC2_SRGB_BLOCK: dxgi = 75; break;
    case VK_FORMAT_BC3_UNORM_BLOCK: dxgi = 77; break;       case VK_FORMAT_BC3_SRGB_BLOCK: dxgi = 78; break;
    case VK_FORMAT_BC4_UNORM_BLOCK: dxgi = 80; break;       case VK_FORMAT_BC5_UNORM_BLOCK: dxgi = 83; break;
    case VK_FORMAT_BC6H_UFLOAT_BLOCK: dxgi = 95; break;     case VK_FORMAT_BC7_UNORM_BLOCK: dxgi = 98; break;
    case VK_FORMAT_BC7_SRGB_BLOCK: dxgi = 99; break;        default: return -1;
    }
    uint32_t bw = (im->w + 3) / 4, bh = (im->h + 3) / 4; size_t bytes = (size_t)bw * bh * im->fi->vbpb;
    NVNcopyRegion r = {0, 0, 0, (int)im->w, (int)im->h, 1};   // the whole level: edge blocks may be partial
    uint8_t *raw = malloc(bytes); int rc = -1;
    if (raw && !vkt_readback_region(im, 0, layer, &r, raw, bytes)) {
        char p2[1300]; snprintf(p2, sizeof p2, "%s.dds", path);
        FILE *f = fopen(p2, "wb");
        if (f) {
            uint32_t h[32 + 5] = {0x20534444u, 124, 0x1007 | 0x80000, im->h, im->w, (uint32_t)(bw * im->fi->vbpb), 0, 1};
            h[19] = 32; h[20] = 4; h[21] = 0x30315844u;   // pixel format: FOURCC "DX10"
            h[27] = 0x1000;
            h[32] = dxgi; h[33] = 3; h[34] = 0; h[35] = 1; h[36] = 0;
            rc = fwrite(h, 4, 37, f) == 37 && fwrite(raw, 1, bytes, f) == bytes ? 0 : -1;
            fclose(f);
        }
    }
    free(raw);
    return rc;
}

// Depth images (M6, the shadow maps): the depth as grey (nearest white), and the values as they are in <path>.raw (16-bit for D16, 32-bit float for D32,
// the 24 bits of D24 in 32-bit words), to compare with the Mac host's DM_CTL texraw
static int dump_depth(vkt_image *im, unsigned layer, const char *path) {
    VkFormat f = im->fi->vk; size_t bpt = f == VK_FORMAT_D16_UNORM ? 2 : 4;
    NVNcopyRegion r = {0, 0, 0, (int)im->w, (int)im->h, 1};
    size_t n = (size_t)im->w * im->h; uint8_t *raw = malloc(n * bpt), *out = malloc(n * 4); int rc = -1;
    if (raw && out && !vkt_readback_region(im, 0, layer, &r, raw, n * bpt)) {
        for (size_t i = 0; i < n; i++) {
            double d; if (bpt == 2) { uint16_t v; memcpy(&v, raw + 2 * i, 2); d = v / 65535.0; }
            else if (f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_D32_SFLOAT_S8_UINT) { float v; memcpy(&v, raw + 4 * i, 4); d = v; }
            else { uint32_t v; memcpy(&v, raw + 4 * i, 4); d = (v & 0xffffff) / 16777215.0; }
            uint8_t g = (uint8_t)lrint(fmin(fmax(1.0 - d, 0), 1) * 255); out[4 * i] = out[4 * i + 1] = out[4 * i + 2] = g; out[4 * i + 3] = 255;
        }
        rc = vks_png_write(path, out, (int)im->w, (int)im->h);
        char p2[1300]; snprintf(p2, sizeof p2, "%s.raw", path); FILE *fr = fopen(p2, "wb"); if (fr) { fwrite(raw, 1, n * bpt, fr); fclose(fr); }
    }
    free(raw); free(out);
    return rc;
}
int vks_dump_image(vkt_image *im, unsigned layer, const char *path) {
    if (im && (im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && im->vtype != VK_IMAGE_VIEW_TYPE_3D) return dump_depth(im, layer, path);
    if (im && !(im->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) && (im->fi->flags & VKF_COMPRESSED)) return dump_dds(im, layer, path);
    if (!im || (im->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) || (im->fi->flags & VKF_COMPRESSED)) return -1;
    NVNcopyRegion r = {0, 0, im->vtype == VK_IMAGE_VIEW_TYPE_3D ? (int)layer : 0, (int)im->w, (int)im->h, 1};
    size_t n = (size_t)im->w * im->h, bytes = n * im->fi->vbpb;
    uint8_t *raw = malloc(bytes), *out = malloc(n * 4); int rc = -1;
    if (raw && out && !vkt_readback_region(im, 0, im->vtype == VK_IMAGE_VIEW_TYPE_3D ? 0 : layer, &r, raw, bytes) && !vks_to_rgba8(im->fi->vk, raw, n, out)) rc = vks_png_write(path, out, (int)im->w, (int)im->h);
    free(raw); free(out);
    return rc;
}

void vks_shot_init(void) {
    const char *e = getenv("NXRT_VK_SHOT");
    for (const char *q = e; q && *q && g_nat < 64; ) { char *end; unsigned long long v = strtoull(q, &end, 0); if (end == q) break; g_at[g_nat++] = v; q = *end == ',' ? end + 1 : end; }
    if ((e = getenv("NXRT_VK_SHOT_EVERY"))) g_every = strtoull(e, NULL, 0);
    if ((e = getenv("NXRT_VK_SHOT_FILE"))) snprintf(g_file, sizeof g_file, "%s", e);
    if ((e = getenv("NXRT_VK_SHOT_DIR"))) { snprintf(g_dir, sizeof g_dir, "%s", e); wchar_t w[1100]; MultiByteToWideChar(CP_UTF8, 0, g_dir, -1, w, 1100); CreateDirectoryW(w, NULL); }
    g_window = (e = getenv("NXRT_VK_SHOT_WINDOW")) && atoi(e);
    if (g_nat || g_every || *g_file) VKLOG("screenshots: %d presents named, every %llu, control file %s, to %s", g_nat, (unsigned long long)g_every, *g_file ? g_file : "-", *g_dir ? g_dir : "beside the log");
}
