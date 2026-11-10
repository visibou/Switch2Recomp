// from nxnative dm/gfx/nvn/hle_nvn.c (MIT, © seedds): the NVN interface layer. M5 track A: nxnative's recording layer ported onto the backend contract
// (gfx_backend.h and nvn_obj.h, gfx_* renamed gfx_*) and merged with the M4 null device's frame loop (docs/m5-record-shaders.md).
//   - Objects live in the game's own NVN object memory in nxnative's layouts (nvn_obj.h). Commands are recorded into the game's command memory in nxnative's
//     record format; when a segment runs out the game's memory callback is called (a guest call), else the host lends a segment, as in nxnative.
//   - Submit, present, sync and resource registration call the gfx_* contract, which forwards to the backend chosen at run time (gfx_select.c, --gfx null|vulkan).
//   - From M4 (in place of nxnative's Mac pacing): a 60 Hz display on a vsync grid paces acquires when the backend has no display link of its own; the frame
//     bookkeeping (guest frame tick, frame statistics, NVN calls per frame, NXRT_EXIT_AT_PRESENT) and the first-frame-request lines.
//   - Changed for Windows: names resolve to host-call tokens (nxnative's counting thunks vanish), SHA-256 through BCrypt, the program-key table is a host hash
//     table, virtual pools reserve with VirtualAlloc and a sparse pool keeps its page table only (Windows cannot alias pages of private memory), CPU texel access
//     goes through nvn_texio (s2tex) in place of nxnative's nvn_deswizzle.c, and call sites come from the guest cpu's x30.
//   - Dropped with the Metal replay: the Mac/iOS present pacing and frame step, thermal, power and thread-policy instrumentation, DM_NVN_TSV rows, captures.
// Environment switches kept from nxnative: DM_NVN_LOG=1 logs object creation, DM_NVN_CMDSTAT=1 command memory statistics, DM_NVN_CBLOG=1 the memory callbacks,
// DM_NVN_SHDUMP=1 keeps programs' control sections for the shader dump, DM_NVN_PROGTRACE=<keys> traces where programs are set, DM_NVN_TEXINFO=WxH texture creation.
#include "hle.h"
#include "hle_registry.h"
#include "nxrt/runtime.h"
#include "nvn_protos.h"
#include "nvn_tex.h"
#include "gfx_ops.h"   // the contract (gfx_backend.h, nvn_obj.h) and the backend chosen at run time
#include "nvn_timestamp.h"
#include "nvn_dirty.h"
#include "nxs_rt.h"
// A pool the GPU writes (nvn_mtl.m mtl_pool_gpu_written): marked when the command is recorded, so a texture's first use comes after it
volatile int nvn_pool_gpuw_new;
void nvn_pool_gpu_written(const void *pool) { pool_t *p = (pool_t *)pool; if (p && !p->gpu_written) { p->gpu_written = 1; nvn_pool_gpuw_new = 1; } }
#include <stdatomic.h>
#include <stddef.h>
#include <bcrypt.h>

// The profile keys the graphics layer reads (nxnative's dm/games/game.py names); Pokopia's game.tsv sets none of them, so game.py's defaults
#ifndef DM_GAME_TEX_FLUSH_UPLOAD
#define DM_GAME_TEX_FLUSH_UPLOAD 0
#endif
#ifndef DM_GAME_TEX_DIRTY_UPLOAD
#define DM_GAME_TEX_DIRTY_UPLOAD 0
#endif
#ifndef DM_GAME_SHADER_KEY_SKIP_LEN
#define DM_GAME_SHADER_KEY_SKIP_OFF 0
#define DM_GAME_SHADER_KEY_SKIP_LEN 0
#endif
#ifndef DM_GAME_RENDER_COND
#define DM_GAME_RENDER_COND 0
#endif

// As nxnative's macro: each implementation is checked at compile time against its nvn_protos.h type. gen_hle.py reads the same lines for the registry and the adapters
#define NVN(name, ret, ...) static ret h_##name(__VA_ARGS__); static const NVNT_##name chk_##name __attribute__((unused)) = h_##name; static ret h_##name(__VA_ARGS__)
// Newer interfaces without prototypes: no signature check
#define NVN_NOPROTO(name, ret, ...) static ret h_##name(__VA_ARGS__)

static int g_log;
static pthread_mutex_t g_sub_mu = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t g_serial;
uint64_t nvn_new_serial(void) { return ++g_serial; }
void nvn_serial_floor(uint64_t s) { uint64_t c = atomic_load(&g_serial); while (c < s && !atomic_compare_exchange_weak(&g_serial, &c, s)) {} }
#define OLOG(...) do { if (g_log) LOG("[nvn] " __VA_ARGS__); } while (0)

// main imports nvnBootstrapLoader alone and finds every other entry point by name through it (or nvnDeviceGetProcAddress): each of the 734 names in nxnative's names.txt
// is a host-call token; the ones this file does not implement are the generated stubs (logged once, neutral). An unknown name returns NULL, as nxnative's lookup
static void *nvn_lookup(const char *name) {
    static int inited; if (!inited) { inited = 1; g_log = getenv("DM_NVN_LOG") != NULL; }
    const nxrt_hle_entry *e = name ? nxrt_hle_find(name, NXRT_HLE_NVN) : NULL;
    if (e) return (void *)(uintptr_t)nxrt_token_of(name);
    LOG("[RED] NVN address lookup of an unknown name: %s", name ? name : "(null)");
    return NULL;
}

// nvnBootstrapLoader is imported by main (defined in sdk); after it is forcibly taken over, the NVN driver in sdk is no longer referenced
HLE_FORCE("nvnBootstrapLoader", "the NVN graphics API is replaced wholesale with a host null device (Metal behind it); the SDK's NVN driver does not run",
          void *, h_nvnBootstrapLoader, const char *name) { return nvn_lookup(name); }

// ---------- Boolean parameters: Switch callers do not extend them, only look at the low 8 bits ----------
#define B(x) ((x) & 0xff)

#include "nvn_obj.h"
#include <process.h>
// Asynchronous submission (aq_*, defined with nvnQueueSubmitCommands below)
static int aq_on(void); static void aq_drain(void); static uint64_t aq_fence_now(void); static int aq_resolve(uint64_t *fence, uint64_t timeout); static int aq_busy(void);
// DM_NVN_CAPTURE (nxnative's nvn_capture.inc) is not ported
#define nvn_cap_submit(n, h) ((void)0)
#define nvn_cap_present(t, crop) ((void)0)

// ---------- Device ----------
NVN(nvnDeviceBuilderSetDefaults, void, NVNdeviceBuilder *b) { memset(b, 0, sizeof(dev_builder_t)); }
NVN(nvnDeviceBuilderSetFlags, void, NVNdeviceBuilder *b, int f) { ((dev_builder_t *)b)->flags = f; }
NVN(nvnDeviceInitialize, NVNboolean, NVNdevice *d, const NVNdeviceBuilder *b) {
    dev_t_ *o = (dev_t_ *)d; memset(o, 0, sizeof *o);
    o->magic = MAGIC('D', 'E', 'V', 0); o->flags = ((const dev_builder_t *)b)->flags;
    static int inited; if (!inited) { inited = 1; g_log = getenv("DM_NVN_LOG") != NULL; gfx_init(); }
    LOG("[nvn] DeviceInitialize %p flags %#x at %.3f s (graphics backend %s)", (void *)d, o->flags, (now_ns() - g_t0_ns) / 1e9, nxrt_gfx_name());
    o->origin = NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT; gfx_set_origin_lower_left(1);   // NVN's default; every game so far switched to upper-left right away, Link's Awakening keeps it
    gfx_set_depth_zero(0);   // NVN's default depth mode: near = -w (GL)
    OLOG("DeviceInitialize %p flags=%#x", (void *)d, o->flags);
    return 1;
}
NVN(nvnDeviceFinalize, void, NVNdevice *d) { ((dev_t_ *)d)->magic = 0; }
NVN(nvnDeviceGetProcAddress, PFNNVNGENERICFUNCPTRPROC, const NVNdevice *d, const char *name) {
    (void)d; return (PFNNVNGENERICFUNCPTRPROC)nvn_lookup(name);
}

// Device properties: values measured on Tegra X1 (nnheaders nvn_DeviceConstantsNX.h, obtained by calling nvnDeviceGetInteger item by item on SMO)
static const int g_devinfo[] = {
    // API version: 55.15 here, 53.313 for SDK 10 and older (nvn_api below; the profile's device_api overrides): SMBW (SDK 16) requires major == 55 and minor >= 15 (main 0x5a7b48); Super Mario 3D World + Bowser's Fury (SDK 10) requires major == 53 and minor >= 313 (main+0x711c88),
    // and on a mismatch it creates no device and leaves the device pointer null, which crashes in nnMain right after. DM_NVN_API=major.minor overrides
    [0x00] = 55, [0x01] = 15,
    [0x02] = 14, [0x03] = 65536, [0x04] = 256, [0x05] = 8, [0x06] = 16, [0x07] = 4, [0x08] = 16, [0x09] = 32,
    [0x0A] = 16, [0x0B] = 4, [0x0C] = 4, [0x0D] = 4, [0x0E] = 16, [0x0F] = 32, [0x10] = 32, [0x11] = 256, [0x12] = 256,
    [0x13] = 4, [0x14] = 8, [0x15] = 512, [0x16] = 256, [0x17] = 2, [0x18] = 256, [0x19] = 4096, [0x1A] = 131072,
    [0x1B] = 16, [0x1C] = 1536, [0x1D] = 1024, [0x1E] = 64, [0x1F] = 1536, [0x20] = 65535, [0x21] = 65535, [0x22] = 65535,
    [0x23] = 8, [0x24] = 1048576, [0x25] = 4096, [0x26] = 16, [0x27] = 4096, [0x28] = 1, [0x29] = 1, [0x2A] = 1,
    [0x2B] = 262144, [0x2C] = 15, [0x2D] = 2048, [0x2E] = 1, [0x2F] = 1, [0x30] = 14, [0x31] = 5, [0x32] = 1, [0x33] = 8,
    [0x34] = 8, [0x35] = 4, [0x36] = 32, [0x37] = 128, [0x38] = 32, [0x39] = 128, [0x3A] = 65536, [0x3B] = 1, [0x3C] = 4,
    [0x3D] = 16384, [0x3E] = 134217728, [0x3F] = 2048, [0x40] = 16384, [0x41] = 16384, [0x42] = 1, [0x43] = 1, [0x44] = 1,
    [0x45] = 1, [0x46] = 5, [0x47] = 1, [0x48] = 1, [0x49] = 1, [0x4A] = 1, [0x4B] = 1, [0x4C] = 1, [0x4D] = 16,
    [0x4E] = 1024, [0x4F] = 32, [0x50] = 4096, [0x51] = 65536, [0x52] = 65536, [0x53] = 256, [0x54] = 16384,
    [0x55] = 262144, [0x56] = 4096, [0x57] = 1, [0x58] = 4, [0x59] = 2, [0x5A] = 0, [0x5B] = 4096, [0x5C] = 16384,
    [0x5D] = 4096, [0x5E] = 128, [0x5F] = 4096,
};
// The Switch 2's values: NVN2 compiled into the game's own sdk answers from constants (Donkey Kong Bananza, sdk 20.5.19: the switch at sdk+0x5d9960 and its table lookup
// sdk+0x5c0a20, read statically; the switch's own constants, among them the versions, are read from the game's sdk at run time, dm_sdk_nvn2_item, and this table has the newest seen, SDK 21.4.4's).
// The item numbers are NVN's: API 231.39 at 0 / 1 and GPU code 201.0-201.20 at 0x2E-0x31 (0xc9 = the version of its shaders' control sections).
// Ampere differs from the X1 in: 32 vertex bindings (the host keeps 16, all its tables hold) and attributes, 1,024 compute threads, an unbounded dispatch width, 16 texture
// levels, 16 KB scratch granularity, 2 KB shader code padding, 32 separate sampler bindings, and items up to 0xA9. DEVINFO_RT: items NVN2 reads from the device's own GPU
// description at run time (scratch scale factors, L2 size, layers, texture sizes, linear stride alignment, pool page size, …), not knowable from the binary: these keep the X1 table's
// value below 0x60, and above it answer 0 with a red line as before
enum { DEVINFO_RT = -1 };
static const int g_devinfo2[] = {
    [0x00] = 231, [0x01] = 39, [0x02] = 14, [0x03] = 65536, [0x04] = 256, [0x05] = 8, [0x06] = 16 /* NVN2 32 */, [0x07] = 4, [0x08] = 16, [0x09] = 32, [0x0A] = 16, [0x0B] = 4,
    [0x0C] = 4, [0x0D] = 4, [0x0E] = 32, [0x0F] = 32, [0x10] = 32, [0x11] = 256, [0x12] = 256, [0x13] = 4, [0x14] = 8, [0x15] = 512, [0x16] = 256, [0x17] = DEVINFO_RT,
    [0x18] = DEVINFO_RT, [0x19] = 4096, [0x1A] = 16384, [0x1B] = 16, [0x1C] = 1024, [0x1D] = 1024, [0x1E] = 64, [0x1F] = 1024, [0x20] = 2147483647, [0x21] = 65535, [0x22] = 65535,
    [0x23] = 8, [0x24] = 1048576, [0x25] = 4096, [0x26] = 16, [0x27] = 4096, [0x28] = 1, [0x29] = 1, [0x2A] = 1, [0x2B] = DEVINFO_RT, [0x2C] = 16, [0x2D] = DEVINFO_RT,
    [0x2E] = 201, [0x2F] = 201, [0x30] = 20, [0x31] = 0, [0x32] = 1, [0x33] = 8, [0x34] = 8, [0x35] = 4, [0x36] = 32, [0x37] = DEVINFO_RT, [0x38] = DEVINFO_RT, [0x39] = 128,
    [0x3A] = DEVINFO_RT, [0x3B] = 1, [0x3C] = 4, [0x3D] = DEVINFO_RT, [0x3E] = DEVINFO_RT, [0x3F] = DEVINFO_RT, [0x40] = DEVINFO_RT, [0x41] = DEVINFO_RT, [0x42] = 1, [0x43] = 1,
    [0x44] = 1, [0x45] = 1, [0x46] = 5, [0x47] = 1, [0x48] = 1, [0x49] = 1, [0x4A] = 1, [0x4B] = 1, [0x4C] = 1, [0x4D] = 16, [0x4E] = 2048, [0x4F] = 32, [0x50] = 4096,
    [0x51] = 65536, [0x52] = 65536, [0x53] = 256, [0x54] = 16384, [0x55] = 262144, [0x56] = 4096, [0x57] = 1, [0x58] = 4, [0x59] = 2, [0x5A] = 0, [0x5B] = 4096, [0x5C] = 16384,
    [0x5D] = 4096, [0x5E] = 128, [0x5F] = 32, [0x60] = 65535, [0x61] = 1, [0x62] = DEVINFO_RT, [0x63] = 256, [0x64] = 16, [0x65] = 65535, [0x66] = 65535, [0x67] = 256,
    [0x68] = 1, [0x69] = 16, [0x6A] = 16, [0x6B] = 16, [0x6C] = 1, [0x6D] = 1, [0x6E] = 1, [0x6F] = 256, [0x70] = 0, [0x71] = 16, [0x72] = 8, [0x73] = 16, [0x74] = 32,
    [0x75] = 31, [0x76] = 536870912, [0x77] = 16777216, [0x78] = 16777216, [0x79] = 1073741824, [0x7A] = 32, [0x7B] = 1, [0x7C] = 64, [0x7D] = DEVINFO_RT, [0x7E] = 1,
    [0x7F] = 2147483647, [0x80] = 32, [0x81] = 32, [0x82] = 4194303, [0x83] = 512, [0x84] = 256, [0x85] = 32, [0x86] = 16383, [0x87] = DEVINFO_RT, [0x88] = 8, [0x89] = 0,
    [0x8A] = 0, [0x8B] = 4, [0x8C] = 256, [0x8D] = 16, [0x8E] = 0, [0x8F] = 1, [0x90] = 31, [0x91] = 0, [0x92] = 64, [0x93] = 1, [0x94] = 58368, [0x95] = 24576, [0x96] = 0,
    [0x97] = 256, [0x98] = 262144, [0x99] = DEVINFO_RT, [0x9A] = DEVINFO_RT, [0x9B] = 0, [0x9C] = 201, [0x9D] = 14, [0x9E] = 201, [0x9F] = 20, [0xA0] = 163840, [0xA1] = 8,
    [0xA2] = 4, [0xA3] = 256, [0xA4] = 256, [0xA5] = 4, [0xA6] = 65536, [0xA7] = 201, [0xA8] = 5, [0xA9] = 1,
};
// The NVN API version the game's own driver would report follows its SDK: games check it and create no device on a mismatch. SDK 10 and older are API 53 (Super Mario 3D World 10.4,
// Age of Calamity 9.6, Link's Awakening 8, Odyssey 7.3, Three Houses 6.5); SDK 11 and later take 55.15
int dm_sdk_ounce(void); int dm_sdk_nvn2_item(unsigned item, int *out);
static int dm_glslc_gpu_code(int *major, int *minor) { *major = *minor = 0; return 0; }   // nxnative's hle_glslc.c is not ported
static void nvn_api(int api[2]) {
#ifdef DM_GAME_NVN_API_MAJOR
    api[0] = DM_GAME_NVN_API_MAJOR; api[1] = DM_GAME_NVN_API_MINOR; LOG("[nvn] device API %d.%d (game profile device_api)", api[0], api[1]);
#else
    // The Switch 2's NVN2 is API 231, and its minor moves with the SDK: the game's sdk holds the answer of its own driver (dm_sdk_nvn2_item). Games require it and create their graphics system
    // only then: Donkey Kong Bananza (SDK 20.5.19) major == 231 and minor >= 30 (main+0x5c61a4), Splatoon Raiders (SDK 21.4.4) minor >= 39 (main+0x31b86b0)
    int m = dm_sdk_major();
    if (dm_sdk_ounce()) { api[0] = 231; if (!dm_sdk_nvn2_item(1, &api[1])) api[1] = g_devinfo2[1]; }
    else { api[0] = DM_SDK_OLD(m) ? 53 : g_devinfo[0]; api[1] = DM_SDK_OLD(m) ? 313 : g_devinfo[1]; }
    LOG("[nvn] device API %d.%d (SDK %d%s)", api[0], api[1], m, dm_sdk_ounce() ? ", Switch 2" : "");
#endif
}
NVN(nvnDeviceGetInteger, void, const NVNdevice *d, NVNdeviceInfo what, int *out) {
    (void)d;
    unsigned w = (unsigned)what;
    // 0x5F separate sampler binding count and 0x60/0x61 are not in the table measured on SMO: 0x5F is given 4096 per the sampler pool limit, unverified; 0x60/0x61 report a failure and return 0
    // 0x67 (SDK 19): FINAL FANTASY TACTICS - The Ivalice Chronicles reads it before every program it loads and passes it as the alignment of its copies of the shader's control section and code
    // (main+0x49800; its allocator raises anything below 8 to 8). Not measured on hardware: 256, the alignment of shader code in a memory pool; the host reads shaders by key, so any value works
    if (dm_sdk_ounce() && w < sizeof g_devinfo2 / sizeof *g_devinfo2 && (g_devinfo2[w] != DEVINFO_RT || w >= sizeof g_devinfo / sizeof *g_devinfo)) {
        if (g_devinfo2[w] == DEVINFO_RT) { LOG("[RED] nvnDeviceGetInteger item %#x: NVN2 reads it from the GPU at run time, not known here; returning 0", w); *out = 0; return; }
        *out = w <= 1 ? 0 : g_devinfo2[w];
        if (w > 1) { dm_sdk_nvn2_item(w, out); OLOG("DeviceGetInteger %#x -> %d (NVN2)", w, *out); return; }   // the sdk's own constant where its switch has one: the GPU code versions move with the SDK
    } else if (w == 0x67) { *out = 256; return; }
    if (w >= sizeof g_devinfo / sizeof *g_devinfo) { LOG("[RED] nvnDeviceGetInteger unknown item %#x, returning 0", w); *out = 0; return; }
    *out = g_devinfo[w];
    // 0x2E / 0x30 describe the GPU code the host can translate, not the linked SDK's driver. Precompiled shaders can come from a newer compiler: ASTRAL CHAIN links SDK 6.4 but its
    // hwfillcolor.vso is 1.15; main+0x9e97d0 rejects the old SDK-based answer 1.14 and initialization unwinds into a null read at main+0x9c0d1c. Our chain translates through 1.16
    // (Bayonetta 3), so advertise that range for every SDK. Retain the floor from a game's own run-time GLSLC (Catherine: Full Body), whose outputs the build also translates.
    if (w == 0x2E || w == 0x30) {
        int gv[2], cur[2] = {1, 16};
        if (dm_glslc_gpu_code(&gv[0], &gv[1]) && (gv[0] > cur[0] || (gv[0] == cur[0] && gv[1] > cur[1]))) { cur[0] = gv[0]; cur[1] = gv[1]; }
        *out = w == 0x2E ? cur[0] : cur[1];
    }
    if (w <= 1) { static int api[2] = {-1, -1}; if (api[0] < 0) { nvn_api(api); const char *e = getenv("DM_NVN_API"); if (e) sscanf(e, "%d.%d", &api[0], &api[1]); } *out = api[w]; }
    OLOG("DeviceGetInteger %#x -> %d", w, *out);
}
NVN(nvnDeviceGetImageHandle, NVNimageHandle, const NVNdevice *d, int id) { (void)d; return (NVNimageHandle)id; }
// Separate texture / sampler handles: value = descriptor index (the sampler index goes in bit 20 and above, consistent with the bit layout of combined handles)
NVN(nvnDeviceGetSeparateTextureHandle, NVNseparateTextureHandle, const NVNdevice *d, int id) { (void)d; return (NVNseparateTextureHandle){(uint64_t)id}; }
NVN(nvnDeviceGetSeparateSamplerHandle, NVNseparateSamplerHandle, const NVNdevice *d, int id) { (void)d; return (NVNseparateSamplerHandle){(uint64_t)id << 20}; }
NVN(nvnDeviceInstallDebugCallback, void, NVNdevice *d, const PFNNVNDEBUGCALLBACKPROC cb, void *data, uint32_t enable) {
    dev_t_ *o = (dev_t_ *)d; o->dbg_cb = B(enable) ? (void *)cb : NULL; o->dbg_data = data;
}
NVN(nvnDeviceSetWindowOriginMode, void, NVNdevice *d, NVNwindowOriginMode m) { ((dev_t_ *)d)->origin = m; gfx_set_origin_lower_left(m == NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT); LOG("[nvn] window origin %s", m ? "upper left (as D3D / Metal)" : "lower left (GL)"); }
NVN(nvnDeviceSetDepthMode, void, NVNdevice *d, NVNdepthMode m) { ((dev_t_ *)d)->depth_mode = m; gfx_set_depth_zero(m != 0); LOG("[nvn] depth mode %s", m ? "near=0 (as D3D / Metal)" : "near=-w (GL)"); }
// Pokémon Sword sets neither mode and reads both, keeping NVN's defaults (lower-left origin, near = -w); the two were red stubs
NVN(nvnDeviceGetWindowOriginMode, NVNwindowOriginMode, const NVNdevice *d) { return ((const dev_t_ *)d)->origin; }
NVN(nvnDeviceGetDepthMode, NVNdepthMode, const NVNdevice *d) { return ((const dev_t_ *)d)->depth_mode; }
NVN(nvnDeviceFinalizeCommandHandle, void, NVNdevice *d, NVNcommandHandle h) { (void)d; (void)h; }

// ---------- Memory pools ----------
NVN(nvnMemoryPoolBuilderSetDefaults, void, NVNmemoryPoolBuilder *b) { memset(b, 0, sizeof(pool_builder_t)); }
NVN(nvnMemoryPoolBuilderSetDevice, void, NVNmemoryPoolBuilder *b, NVNdevice *d) { ((pool_builder_t *)b)->dev = d; }
NVN(nvnMemoryPoolBuilderSetStorage, void, NVNmemoryPoolBuilder *b, void *m, size_t n) { ((pool_builder_t *)b)->mem = m; ((pool_builder_t *)b)->size = n; }
NVN(nvnMemoryPoolBuilderSetFlags, void, NVNmemoryPoolBuilder *b, int f) { ((pool_builder_t *)b)->flags = f; }
static uint8_t *vpool_reserve(pool_t *o); static void vpool_release(pool_t *o);   // sparse virtual pools, see MapVirtual
NVN(nvnMemoryPoolInitialize, NVNboolean, NVNmemoryPool *p, const NVNmemoryPoolBuilder *b) {
    const pool_builder_t *s = (const pool_builder_t *)b; pool_t *o = (pool_t *)p;
    o->magic = MAGIC('P', 'O', 'O', 'L'); o->flags = s->flags; o->mem = s->mem; o->size = s->size; o->vp = NULL; o->vmode = 0; o->gpu_written = 0;
    if (!o->mem && (o->flags & 0x200)) { uint8_t *vr = vpool_reserve(o); if (vr) o->mem = vr; }   // a virtual pool's own address range until its first mapping decides (see MapVirtual)
    else if (o->mem) { gfx_pool_add(o->mem, o->size); nvn_dirty_add_range(o->mem, o->size); { void vkm_pool_flags(const void *mem, uint32_t flags); vkm_pool_flags(o->mem, o->flags); } }
    if (!o->mem && !(o->flags & 0x200)) LOG("[RED] MemoryPoolInitialize: non-virtual pool has no storage flags=%#x size=%#llx", o->flags, o->size);
    if (getenv("DM_NVN_VLOG")) { LOG("[nvn] MemoryPool %p mem=%p size=%#llx flags=%#x", (void *)p, (void *)o->mem, (unsigned long long)o->size, o->flags); dm_backtrace(DM_GUEST_FP()); }
    OLOG("MemoryPool %p mem=%p size=%#llx flags=%#x", (void *)p, (void *)o->mem, o->size, o->flags);
    return 1;
}
static void fl_add(const uint8_t *a, uint64_t n); static void fl_drop(const uint8_t *a, uint64_t n);   // flushed ranges, see nvn_flushed
NVN(nvnMemoryPoolFinalize, void, NVNmemoryPool *p) { if (((pool_t *)p)->flags & 0x200 && ((pool_t *)p)->vp) vpool_release((pool_t *)p);
    if (((pool_t *)p)->mem && !(((pool_t *)p)->flags & 0x200)) {   // A virtual pool's storage is borrowed from a physical pool; do not unregister it
 nvn_dirty_remove_range(((pool_t *)p)->mem); gfx_pool_del(((pool_t *)p)->mem); fl_drop(((pool_t *)p)->mem, ((pool_t *)p)->size); } gfx_alias_drop_pool(p); ((pool_t *)p)->magic = 0; }
static void ft_flush(const uint8_t *a, uint64_t n);
NVN(nvnMemoryPoolFlushMappedRange, void, const NVNmemoryPool *p, ptrdiff_t off, size_t n) {   // see ft_add and nvn_flushed
    const pool_t *o = (const pool_t *)p;
    if (!o->mem) return;
    if (!(o->flags & 0x200)) fl_add(o->mem + off, n);
    ft_flush(o->mem + off, n);
}
NVN(nvnMemoryPoolGetBufferAddress, NVNbufferAddress, const NVNmemoryPool *p) { return (NVNbufferAddress)((const pool_t *)p)->mem; }

// ---------- Descriptor pools ----------
// One ID space per pool. In NVN a texture or sampler ID indexes the pool the command buffer has bound (SetTexturePool / SetSamplerPool), and two pools may use the same IDs. The replay keeps
// one table for each kind, so every pool gets a base in it, in creation order: registrations go to base + ID, and binds recorded on a command buffer get the base of the pool that command
// buffer last bound (bind_handles). A game with one pool of each kind is unchanged (base 0).
// Fire Emblem: Three Houses: a 12,544-entry texture pool and a 3,856-entry sampler pool for the scene, and 4,096-entry ones its 2D UI command buffer binds. Both fill IDs from 256 up; the scene's
// default 8×8 white texture overwrote the UI's textures, and the title menu's labels, the name entry and other menus drew as plain bars or not at all. The scene also registers each draw's
// textures into fresh IDs of a ring in its large pool (8,729 and up a few seconds in); with the table's capacity taken from the last pool alone, those all read as empty slots and the screen was black
enum { NDPOOL = 16 };
static struct dpool_base { const void *pool; int base; } g_tpb[NDPOOL], g_spb[NDPOOL];
static int g_ntpb, g_nspb, g_tnext, g_snext;
static pthread_mutex_t g_dpool_mu = PTHREAD_MUTEX_INITIALIZER;
static int dpool_add(struct dpool_base *t, int *nt, int *next, const void *pool, int n, const char *kind) {   // → the pool's base (assigned at its first Initialize)
    pthread_mutex_lock(&g_dpool_mu);
    int i = 0; while (i < *nt && t[i].pool != pool) i++;
    if (i == *nt) {
        if (*nt == NDPOOL) { pthread_mutex_unlock(&g_dpool_mu); LOG("[RED] %s pools exceed %d: later ones share the first pool's numbering", kind, NDPOOL); return 0; }
        t[i] = (struct dpool_base){pool, *next}; (*nt)++; *next += (n + 255) & ~255;
        if (i) LOG("[nvn] pool #%d (%s, %d entries): numbered from %d in the host table, separate from the earlier pools", i + 1, kind, n, t[i].base);
    }
    int b = t[i].base; pthread_mutex_unlock(&g_dpool_mu); return b;
}
static int dpool_base(const struct dpool_base *t, int nt, const void *pool) { for (int i = 0; i < nt; i++) if (t[i].pool == pool) return t[i].base; return 0; }
int nvn_dpool_base(const void *pool, int tex) { return tex ? dpool_base(g_tpb, g_ntpb, pool) : dpool_base(g_spb, g_nspb, pool); }   // the replay's bound pools (nvn_mtl.m state_op)
// The pools each command buffer last bound are kept in its own host state (cmd_t tex_base / samp_base). They used to be a 64-entry list of command buffers that was never cleared:
// Dark Souls Remastered binds one of 12 sampler pools in more than 64 command buffers, and every later one read its samplers through the first pool's numbering
NVN(nvnTexturePoolInitialize, NVNboolean, NVNtexturePool *tp, const NVNmemoryPool *p, ptrdiff_t off, int n) {
    *(dpool_t *)tp = (dpool_t){(const pool_t *)p, off, n}; OLOG("TexturePool n=%d", n);
    dpool_add(g_tpb, &g_ntpb, &g_tnext, tp, n, "texture"); gfx_set_tex_pool_n(g_tnext);
    return 1;
}
NVN(nvnTexturePoolFinalize, void, NVNtexturePool *tp) { (void)tp; }
NVN(nvnTexturePoolRegisterTexture, void, const NVNtexturePool *tp, int id, const NVNtexture *t, const NVNtextureView *v) {
    gfx_reg_tex(dpool_base(g_tpb, g_ntpb, tp) + id, (const tex_t *)t, (const view_t *)v);
}
// Image descriptors live in the same pool index space as texture descriptors, and image handles are pool indices (nvnDeviceGetImageHandle; BindImages below), so an image
// registers into the same table. Hyrule Warriors: Age of Calamity registers the images its compute shaders write in the first battle (earlier games registered only textures)
NVN(nvnTexturePoolRegisterImage, void, const NVNtexturePool *tp, int id, const NVNtexture *t, const NVNtextureView *v) {
    gfx_reg_image(dpool_base(g_tpb, g_ntpb, tp) + id, (const tex_t *)t, (const view_t *)v);
}
NVN(nvnSamplerPoolInitialize, NVNboolean, NVNsamplerPool *sp, const NVNmemoryPool *p, ptrdiff_t off, int n) {
    *(dpool_t *)sp = (dpool_t){(const pool_t *)p, off, n}; OLOG("SamplerPool n=%d", n); dpool_add(g_spb, &g_nspb, &g_snext, sp, n, "sampler"); return 1;
}
NVN(nvnSamplerPoolFinalize, void, NVNsamplerPool *sp) { (void)sp; }
NVN(nvnSamplerPoolRegisterSampler, void, const NVNsamplerPool *sp, int id, const NVNsampler *s) { gfx_reg_samp(dpool_base(g_spb, g_nspb, sp) + id, &((const samp_t *)s)->p); }
// straight from a builder, without a sampler object (FINAL FANTASY TACTICS - The Ivalice Chronicles registers its samplers this way)
NVN(nvnSamplerPoolRegisterSamplerBuilder, void, const NVNsamplerPool *sp, int id, const NVNsamplerBuilder *b) { gfx_reg_samp(dpool_base(g_spb, g_nspb, sp) + id, &((const samp_builder_t *)b)->p); }
// Unreal Engine reads a descriptor pool's size back at device setup (Octopath Traveler 0)
NVN(nvnTexturePoolGetMemoryPool, const NVNmemoryPool *, const NVNtexturePool *tp) { return (const NVNmemoryPool *)((const dpool_t *)tp)->pool; }
NVN(nvnTexturePoolGetMemoryOffset, ptrdiff_t, const NVNtexturePool *tp) { return (ptrdiff_t)((const dpool_t *)tp)->off; }
NVN(nvnTexturePoolGetSize, int, const NVNtexturePool *tp) { return ((const dpool_t *)tp)->n; }
NVN(nvnSamplerPoolGetMemoryPool, const NVNmemoryPool *, const NVNsamplerPool *sp) { return (const NVNmemoryPool *)((const dpool_t *)sp)->pool; }
NVN(nvnSamplerPoolGetMemoryOffset, ptrdiff_t, const NVNsamplerPool *sp) { return (ptrdiff_t)((const dpool_t *)sp)->off; }
NVN(nvnSamplerPoolGetSize, int, const NVNsamplerPool *sp) { return ((const dpool_t *)sp)->n; }

// ---------- Textures ----------
#define TB(b) ((tex_builder_t *)(b))
NVN(nvnTextureBuilderSetDefaults, void, NVNtextureBuilder *b) {
    NVNdevice *dev = TB(b)->dev; memset(b, 0, sizeof(tex_builder_t)); TB(b)->dev = dev;
    TB(b)->d.target = NVN_TEXTURE_TARGET_2D; TB(b)->d.width = TB(b)->d.height = TB(b)->d.depth = TB(b)->d.levels = 1;
}
NVN(nvnTextureBuilderSetDevice, void, NVNtextureBuilder *b, NVNdevice *d) { TB(b)->dev = d; }
NVN(nvnTextureBuilderSetFlags, void, NVNtextureBuilder *b, int f) { TB(b)->d.flags = f; }
NVN(nvnTextureBuilderSetTarget, void, NVNtextureBuilder *b, NVNtextureTarget t) { TB(b)->d.target = t; }
NVN(nvnTextureBuilderSetWidth, void, NVNtextureBuilder *b, int v) { TB(b)->d.width = v; }
NVN(nvnTextureBuilderSetHeight, void, NVNtextureBuilder *b, int v) { TB(b)->d.height = v; }
NVN(nvnTextureBuilderSetDepth, void, NVNtextureBuilder *b, int v) { TB(b)->d.depth = v; }
NVN(nvnTextureBuilderSetSize1D, void, NVNtextureBuilder *b, int w) { TB(b)->d.width = w; TB(b)->d.height = TB(b)->d.depth = 1; }
NVN(nvnTextureBuilderSetLevels, void, NVNtextureBuilder *b, int v) { TB(b)->d.levels = v; }
NVN(nvnTextureBuilderSetFormat, void, NVNtextureBuilder *b, NVNformat f) { TB(b)->d.format = f; }
NVN(nvnTextureBuilderSetSamples, void, NVNtextureBuilder *b, int s) { TB(b)->d.samples = s; }
NVN(nvnTextureBuilderSetStride, void, NVNtextureBuilder *b, ptrdiff_t s) { TB(b)->d.stride = (uint32_t)s; }
NVN(nvnTextureBuilderSetStorage, void, NVNtextureBuilder *b, NVNmemoryPool *p, ptrdiff_t off) { TB(b)->pool = (pool_t *)p; TB(b)->off = off; }
NVN(nvnTextureBuilderSetPackagedTextureLayout, void, NVNtextureBuilder *b, const NVNpackagedTextureLayout *l) {
    TB(b)->has_layout = l != NULL; if (l) memcpy(TB(b)->layout, l->layout, 8);
    TB(b)->d.pkg_block_h = l ? l->layout[0] + 1 : 0;
}
NVN(nvnTextureBuilderGetFlags, NVNtextureFlags, const NVNtextureBuilder *b) { return (NVNtextureFlags)((const tex_builder_t *)b)->d.flags; }
static uint64_t g_layout_logged;
static void log_layout(const tex_builder_t *b, const nvn_texlayout *L) {
    if (b->has_layout && g_layout_logged++ < 16)
        LOG("[nvn] packaged texture %ux%ux%u levels %u format %#x target %d layout bytes %02x%02x%02x%02x%02x%02x%02x%02x → block height log2=%u (from layout byte 0) size %#llx",
            b->d.width, b->d.height, b->d.depth, b->d.levels, b->d.format, b->d.target, b->layout[0], b->layout[1], b->layout[2],
            b->layout[3], b->layout[4], b->layout[5], b->layout[6], b->layout[7], L->tileH, L->size);
}
NVN(nvnTextureBuilderGetStorageSize, size_t, const NVNtextureBuilder *b) {
    nvn_texlayout L; nvn_tex_layout(&((const tex_builder_t *)b)->d, &L);
    if (!L.bpb) LOG("[RED] TextureBuilderGetStorageSize unknown format %#x", ((const tex_builder_t *)b)->d.format);
    log_layout((const tex_builder_t *)b, &L);
    return L.size;
}
NVN(nvnTextureBuilderGetStorageAlignment, size_t, const NVNtextureBuilder *b) {
    nvn_texlayout L; nvn_tex_layout(&((const tex_builder_t *)b)->d, &L); return L.align;
}
// The tiled levels of a sparse texture and its tile size (nvn_tex_sparse_tiles). OCTOPATH TRAVELER (Unreal Engine 4 on SDK 3.5.1) sizes a sparse texture's mip tail as storage(all
// levels) − storage(numTiledLevels levels) and allocates it from its heap at boot (main+0x1f1a688); unimplemented, the count was stack garbage and it asked for the whole 1.75 GB texture twice
static void sparse_layout(const nvn_texdesc *d, NVNtextureSparseTileLayout *o) {
    uint32_t tw, th, td; memset(o, 0, sizeof *o);
    o->numTiledLevels = nvn_tex_sparse_tiles(d, &tw, &th, &td); o->tileWidth = (int)tw; o->tileHeight = (int)th; o->tileDepth = (int)td;
}
NVN(nvnTextureBuilderGetSparseTileLayout, void, const NVNtextureBuilder *b, NVNtextureSparseTileLayout *o) { sparse_layout(&((const tex_builder_t *)b)->d, o); }
#define TX(t) ((tex_t *)(t))
#define CTX(t) ((const tex_t *)(t))
// Textures the CPU fills in place (profile tex_flush_upload): Tears of the Kingdom streams its textures (TexToGo, already in the GPU's block-linear layout) straight into
// CPU-cached memory pools and publishes each with nvnMemoryPoolFlushMappedRange (9,436 calls at boot), without a packaged layout or WriteTexels. They stayed GPU-only
// textures with content 0, and the G-buffer's base colour came out black. Textures that can hold such data (not render targets, not linear, no packaged layout, in a pool
// the CPU can write) are registered by address range; a flush that covers part of one marks it CPU-written (gfx_tex_cpu_written: uploaded from guest memory, re-uploaded
// when its pages change). Off by default: in another game a flushed range could cover an image a shader writes, whose GPU content the upload would replace
typedef struct { const uint8_t *mem; uint64_t size; tex_t t; uint64_t stamp; const void *span; } flush_tex;   // stamp: dirty-page generation at registration (tex_dirty_upload); span: its tracked range (nvn_dirty_span), so a check needs no search over every pool
static flush_tex *g_ft; static size_t g_nft, g_cft; static pthread_mutex_t g_ft_mu = PTHREAD_MUTEX_INITIALIZER;
static void ft_add(const tex_t *t) {
    // Buffer textures are views onto the pool buffer the GPU reads directly (tex_mtl in nvn_mtl.m); an upload would push Age of Calamity's 1,048,576-texel R32F one through tex_desc, past Metal's 2D limit
    if (!DM_GAME_TEX_FLUSH_UPLOAD || !t->mem || !t->pool || t->has_layout || t->d.target == NVN_TEXTURE_TARGET_BUFFER || (t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) ||
        (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_COMPRESSIBLE))) return;
    pthread_mutex_lock(&g_ft_mu);
    if (g_nft == g_cft && !(g_ft = realloc(g_ft, (g_cft = g_cft ? 2 * g_cft : 1024) * sizeof *g_ft))) die("flush_tex allocation failed");
    uint64_t st = nvn_dirty_now(); if (DM_GAME_TEX_DIRTY_UPLOAD) nvn_dirty_protect(t->mem, t->L.size);   // see nvn_ft_written
    g_ft[g_nft++] = (flush_tex){t->mem, t->L.size, *t, st, nvn_dirty_span(t->mem, t->L.size)};
    pthread_mutex_unlock(&g_ft_mu);
}
static void ft_del(uint64_t serial) {
    if (!DM_GAME_TEX_FLUSH_UPLOAD || !serial) return;
    pthread_mutex_lock(&g_ft_mu);
    for (size_t i = 0; i < g_nft; i++) if (g_ft[i].t.serial == serial) { g_ft[i] = g_ft[--g_nft]; break; }
    pthread_mutex_unlock(&g_ft_mu);
}
// Profile tex_dirty_upload: Tears of the Kingdom fills some textures from the CPU after creating them and publishes them with no NVN flush at all (os::FlushDataCache runs
// natively; its 121-layer ASTC terrain material arrays, 42 MB each). They stayed GPU-only textures with content 0, which ASTC decodes as the error colour, and the cave came out magenta.
// ft_add write-protects a registered texture's pages, and on its first resolution (tex_mtl, submit thread) a write since then makes it CPU-written like a flushed one
int nvn_ft_written(const tex_t *t) {
    if (!DM_GAME_TEX_DIRTY_UPLOAD || !t->serial) return 0;
    int r = 0; pthread_mutex_lock(&g_ft_mu);
    for (size_t i = 0; i < g_nft; i++) if (g_ft[i].t.serial == t->serial) {
        if (nvn_dirty_since_span(g_ft[i].span, g_ft[i].mem, g_ft[i].size, g_ft[i].stamp)) { r = 1; g_ft[i] = g_ft[--g_nft]; }
        break; }
    pthread_mutex_unlock(&g_ft_mu);
    if (r) gfx_tex_cpu_written(t);
    return r;
}
int nvn_ft_tracked(uint64_t serial) {   // registered and not resolved yet: its storage holds whatever the CPU has written (nvn_tex_mirror_b2t / nvn_tex_mirror_t2t)
    if (!DM_GAME_TEX_FLUSH_UPLOAD || !serial) return 0;
    int r = 0; pthread_mutex_lock(&g_ft_mu);
    for (size_t i = 0; i < g_nft; i++) if (g_ft[i].t.serial == serial) { r = 1; break; }
    pthread_mutex_unlock(&g_ft_mu);
    return r;
}
// Later writes (tex_dirty_upload): a texture first resolved before the CPU filled it stays a GPU-only texture, and the first-use check above never runs for it again.
// Once per submission, when guest memory has changed since the last look, the registered textures whose pages were written since registration become CPU-written
// (tex_mtl_ in nvn_mtl.m then drops the GPU-only texture and uploads it). Tears of the Kingdom fills its 5120x5120 BC4 roughness atlas from the CPU after first binding it,
// as tiles stream in. A texture the GPU writes (render target, shader-written image) may share pages with data the CPU writes: it leaves the list without an upload, which would
// replace the GPU's content
// Each poll looks at no more than FT_POLL_BYTES of registered textures, going on where the last one stopped; any guest write starts another pass over the whole list.
// Checking every texture at every submission cost TOTK 0.4 ms of submit CPU per frame on the Mac; this way a texture is picked up a few submissions later at most
enum { FT_POLL_BYTES = 128 << 20 };
void nvn_ft_poll(void) {
    if (!DM_GAME_TEX_DIRTY_UPLOAD) return;
    static uint64_t gen; static size_t cur, left;
    static tex_t *hit; static size_t chit; size_t nhit = 0;
    uint64_t now = nvn_dirty_now(), scanned = 0;
    pthread_mutex_lock(&g_ft_mu);
    if (now != gen) { gen = now; left = g_nft; }   // guest memory changed: every entry gets one more look
    while (left && g_nft && scanned < FT_POLL_BYTES) {
        if (cur >= g_nft) cur = 0;
        scanned += g_ft[cur].size; left--;
        if (nvn_dirty_since_span(g_ft[cur].span, g_ft[cur].mem, g_ft[cur].size, g_ft[cur].stamp)) {
            if (nhit == chit && !(hit = realloc(hit, (chit = chit ? 2 * chit : 64) * sizeof *hit))) die("nvn_ft_poll allocation failed");
            hit[nhit++] = g_ft[cur].t; g_ft[cur] = g_ft[--g_nft];
        } else cur++;
    }
    pthread_mutex_unlock(&g_ft_mu);
    for (size_t i = 0; i < nhit; i++) if (!gfx_tex_gpu_owned(hit[i].serial)) gfx_tex_cpu_written(&hit[i]);   // outside g_ft_mu: the ownership check takes the replay's serial table lock
}
static void ft_flush(const uint8_t *a, uint64_t n) {
    if (!DM_GAME_TEX_FLUSH_UPLOAD || !a || !n) return;
    pthread_mutex_lock(&g_ft_mu);
    for (size_t i = 0; i < g_nft;)
        if (g_ft[i].mem < a + n && a < g_ft[i].mem + g_ft[i].size) { gfx_tex_cpu_written(&g_ft[i].t); g_ft[i] = g_ft[--g_nft]; } else i++;
    pthread_mutex_unlock(&g_ft_mu);
}
// Flushed ranges (every game): what the CPU published with nvnMemoryPoolFlushMappedRange, as disjoint [lo, hi) spans sorted by start (touching spans merge), dropped with their pool.
// Until the GPU first uses a texture it has not written it, so storage the game flushed before that first use is the texture's content: tex_init_fill (nvn_mtl.m) fills the new GPU
// texture from it once, as it does for a pool the CPU cannot access. Breath of the Wild copies its precomputed sky scattering tables (System/KSys/sky.skybin: a 256x32x16 RGBA16F 3D
// texture and two 2D ones) into CPU_CACHED pools of their own with memcpy and flushes them (main+0x1257d50); as GPU textures of zeros the sky had no scattered light and came out
// brownish grey instead of blue (GitHub issue #52). Textures the CPU keeps writing after their first use still need the profile's tex_flush_upload / tex_dirty_upload
typedef struct { const uint8_t *lo, *hi; } fl_span;
static fl_span *g_fl; static size_t g_nfl, g_cfl; static pthread_mutex_t g_fl_mu = PTHREAD_MUTEX_INITIALIZER;
static size_t fl_find(const uint8_t *a, int touch) {   // the first span that ends after a (touch: or at a); g_fl_mu held
    size_t l = 0, h = g_nfl;
    while (l < h) { size_t m = (l + h) / 2; if (g_fl[m].hi < a || (!touch && g_fl[m].hi == a)) l = m + 1; else h = m; }
    return l;
}
static void fl_add(const uint8_t *a, uint64_t n) {
    if (!a || !n) return;
    const uint8_t *b = a + n;
    pthread_mutex_lock(&g_fl_mu);
    size_t i = fl_find(a, 1), j = i;
    if (i < g_nfl && g_fl[i].lo <= a && b <= g_fl[i].hi) { pthread_mutex_unlock(&g_fl_mu); return; }   // already covered (a buffer flushed every frame)
    while (j < g_nfl && g_fl[j].lo <= b) { if (g_fl[j].lo < a) a = g_fl[j].lo; if (g_fl[j].hi > b) b = g_fl[j].hi; j++; }   // spans i .. j-1 overlap or touch [a, b): merged into one
    if (i == j) {
        if (g_nfl == g_cfl && !(g_fl = realloc(g_fl, (g_cfl = g_cfl ? 2 * g_cfl : 256) * sizeof *g_fl))) die("flushed range allocation failed");
        memmove(g_fl + i + 1, g_fl + i, (g_nfl - i) * sizeof *g_fl); g_nfl++;
    } else if (j > i + 1) { memmove(g_fl + i + 1, g_fl + j, (g_nfl - j) * sizeof *g_fl); g_nfl -= j - i - 1; }
    g_fl[i] = (fl_span){a, b};
    pthread_mutex_unlock(&g_fl_mu);
}
static void fl_drop(const uint8_t *a, uint64_t n) {   // a pool goes: its part of every span (spans of neighbouring pools may have merged with it)
    const uint8_t *b = a + n;
    pthread_mutex_lock(&g_fl_mu);
    for (size_t i = fl_find(a, 0); i < g_nfl && g_fl[i].lo < b;) {
        fl_span s = g_fl[i];
        if (s.lo < a && s.hi > b) {   // the pool lies inside the span: split it
            if (g_nfl == g_cfl && !(g_fl = realloc(g_fl, (g_cfl = 2 * g_cfl) * sizeof *g_fl))) die("flushed range allocation failed");
            memmove(g_fl + i + 2, g_fl + i + 1, (g_nfl - i - 1) * sizeof *g_fl); g_nfl++;
            g_fl[i].hi = a; g_fl[i + 1] = (fl_span){b, s.hi}; break;
        }
        if (s.lo < a) { g_fl[i].hi = a; i++; }
        else if (s.hi > b) { g_fl[i].lo = b; break; }
        else { memmove(g_fl + i, g_fl + i + 1, (g_nfl - i - 1) * sizeof *g_fl); g_nfl--; }
    }
    pthread_mutex_unlock(&g_fl_mu);
}
int nvn_flushed(const void *p, uint64_t n) {   // whether the CPU flushed any of [p, p + n)
    const uint8_t *a = p;
    pthread_mutex_lock(&g_fl_mu);
    size_t i = fl_find(a, 0); int r = i < g_nfl && g_fl[i].lo < a + n;
    pthread_mutex_unlock(&g_fl_mu);
    return r;
}
NVN(nvnTextureInitialize, NVNboolean, NVNtexture *t, const NVNtextureBuilder *b) {
    const tex_builder_t *s = (const tex_builder_t *)b; tex_t *o = TX(t);
    // Re-initializing a texture object that is still live (without a Finalize first): release the host texture of the old serial first. SMBW re-initializes its transient render targets on the same object every frame and never Finalizes them;
    //   previously every init created a new private MTLTexture and the old one stayed in the serial table forever, so GPU memory grew about 13 MB per frame (killed by the memory watchdog at 24 GB)
    if (o->magic == MAGIC('T', 'E', 'X', 0) && o->serial) { static uint64_t n; if (!n++) LOG("[nvn] TextureInitialize on a texture object that is still live (no Finalize): releasing the old host texture (first case ser=%llu)", (unsigned long long)o->serial); gfx_tex_release(o); }
    memset(o, 0, sizeof *o);
    o->magic = MAGIC('T', 'E', 'X', 0); o->d = s->d; o->pool = s->pool; o->off = s->off;
    o->has_layout = s->has_layout; memcpy(o->layout, s->layout, 8); o->set_swz = s->set_swz; o->stencil = s->stencil; o->swz = s->swz;
    nvn_tex_layout(&o->d, &o->L);
    o->mem = s->pool && s->pool->mem ? s->pool->mem + s->off : NULL;
    o->serial = nvn_new_serial(); gfx_tex_init(o);
    {   // DM_NVN_TEXINFO=WxH also logs every texture of that size as it is initialized (render targets included): which ones share storage, and how their descriptions differ
        static int ti = -1; static uint32_t tw, th;
        if (ti < 0) { const char *e = getenv("DM_NVN_TEXINFO"); ti = e && sscanf(e, "%ux%u", &tw, &th) == 2; }
        if (ti && o->d.width == tw && o->d.height == th) { static uint64_t n; if (n++ < 200) LOG("[nvn] TEXINFO init ser=%llu mem=%p pool=%p(mem %p size %#llx flags %#x) off=%#llx target=%d format=%#x flags=%#x samples=%d %ux%ux%u levels=%u stride=%u size=%#llx",
            (unsigned long long)o->serial, (void *)o->mem, (void *)o->pool, o->pool ? (void *)o->pool->mem : NULL, o->pool ? (unsigned long long)o->pool->size : 0ull, o->pool ? o->pool->flags : 0,
            (unsigned long long)o->off, o->d.target, o->d.format, o->d.flags, o->d.samples, o->d.width, o->d.height, o->d.depth, o->d.levels, o->d.stride, (unsigned long long)o->L.size); }
    }
    // See SetPackagedTextureData. Linear textures are not re-uploaded as packaged: they are created on the pool buffer (tex_linear), whatever the pool's CPU access, and read what was copied in
    if (s->pkg_data && o->mem) {
        memcpy(o->mem, s->pkg_data, o->L.size);
        if (!(o->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET))) gfx_tex_cpu_written(o);
    }
    OLOG("Texture %p %ux%ux%u lv%u fmt=%#x tgt=%d flags=%#x size=%#llx", (void *)t, o->d.width, o->d.height, o->d.depth,
         o->d.levels, o->d.format, o->d.target, o->d.flags, o->L.size);
    ft_add(o);
    return 1;
}
NVN(nvnTextureFinalize, void, NVNtexture *t) { ft_del(TX(t)->serial); gfx_tex_release(TX(t)); TX(t)->magic = 0; }
NVN(nvnTextureGetFlags, NVNtextureFlags, const NVNtexture *t) { return (NVNtextureFlags)CTX(t)->d.flags; }
NVN(nvnTextureGetWidth, int, const NVNtexture *t) { return (int)CTX(t)->d.width; }
NVN(nvnTextureGetHeight, int, const NVNtexture *t) { return (int)CTX(t)->d.height; }
NVN(nvnTextureGetDepth, int, const NVNtexture *t) { return (int)CTX(t)->d.depth; }
NVN(nvnTextureGetStorageSize, int, const NVNtexture *t) { return (int)CTX(t)->L.size; }
NVN(nvnTextureGetStride, ptrdiff_t, const NVNtexture *t) { return (ptrdiff_t)CTX(t)->d.stride; }   // the row stride of a linear texture as built (0 for block-linear ones)
// The queries below are used by Engage (Unity) at startup: it calls MemoryPoolMap on "the pool the texture / buffer lives in", and with a stub returning 0 it dereferenced a null pool and crashed
NVN(nvnTextureGetLevels, int, const NVNtexture *t) { return (int)CTX(t)->d.levels; }
NVN(nvnTextureGetSparseTileLayout, void, const NVNtexture *t, NVNtextureSparseTileLayout *o) { sparse_layout(&CTX(t)->d, o); }
NVN(nvnTextureGetSamples, int, const NVNtexture *t) { return CTX(t)->d.samples; }
NVN(nvnTextureGetMemoryPool, NVNmemoryPool *, const NVNtexture *t) { return (NVNmemoryPool *)CTX(t)->pool; }
NVN(nvnTextureGetMemoryOffset, ptrdiff_t, const NVNtexture *t) { return (ptrdiff_t)CTX(t)->off; }
// The GPU address of the texture's storage (DRAGON QUEST XI S, Unreal Engine 4.18, asks for it at start): pool address + offset, the same address space as nvnMemoryPoolGetBufferAddress
NVN(nvnTextureGetTextureAddress, NVNtextureAddress, const NVNtexture *t) { const tex_t *o = CTX(t); return o->pool ? (NVNtextureAddress)(o->pool->mem + o->off) : 0; }
// The size of a texture's ZCull save area (SaveZCullData / RestoreZCullData). The host does no ZCull and never touches the area, but games allocate it from their own pools, and NVN
// does not report 0. Cuphead (Unity 2017.4) allocates one per depth target (main+0x1948060) from its pool allocator, which gives a zero-size request the offset of a live block; its
// map of allocations then loses that block (the zero-size one's size overwrites it, and freeing the zero-size one deletes it), and the block's own free later pushes a garbage-sized
// range onto the free list (main+0x1945b10). Buffers allocated after that overlap live ones: copies read staging bytes other copies had overwritten, and sprites broke into slivers,
// went missing, or flickered every fourth frame (the upload ring's period). The hardware's formula is not known; this one grows with the texture: 8 bytes per 32x32 tile and layer,
// at least 256
static size_t zcull_size(const nvn_texdesc *d) {
    uint64_t tiles = (uint64_t)((d->width + 31) / 32) * ((d->height + 31) / 32) * (d->depth ? d->depth : 1);
    return tiles ? (size_t)((tiles * 8 + 0xff) & ~0xffull) : 0x100;
}
NVN(nvnTextureGetZCullStorageSize, size_t, const NVNtexture *t) { return zcull_size(&CTX(t)->d); }
static unsigned tex_swz(const tex_t *t) { return t->set_swz ? t->swz : 2u | 3u << 3 | 4u << 6 | 5u << 9; }   // the channel mapping: identity R G B A when TextureBuilderSetSwizzle was not called
// The mapping counts: BotW and Odyssey rebuild layout captures on the same storage with a new mapping, and replace the registered descriptor only if the two compare different (BotW main+0xec33f0)
NVN(nvnTextureCompare, NVNboolean, const NVNtexture *a, const NVNtexture *b) {
    const tex_t *x = CTX(a), *y = CTX(b);
    return !memcmp(&x->d, &y->d, sizeof x->d) && x->mem == y->mem && tex_swz(x) == tex_swz(y);
}
// No prototype in the header; call sites such as main 0x174899c compare the return value with 1 and set some status bit 0x100 if not equal. Meaning unverified; return 1 for now
NVN_NOPROTO(nvnTextureGetRawStorageClass, int, const NVNtexture *t) { (void)t; return 1; }
// The class a texture from this builder would report, so the two agree: EA SPORTS FC 27's Frostbite renderer asks the builder (the class names the page kind a virtual pool maps,
// NVNmappingRequest.storageClass, which the host's virtual pools do not distinguish)
NVN_NOPROTO(nvnTextureBuilderGetRawStorageClass, int, const NVNtextureBuilder *b) { (void)b; return 1; }
#define VW(v) ((view_t *)(v))
NVN(nvnTextureViewSetDefaults, void, NVNtextureView *v) { memset(v, 0, sizeof(view_t)); }
NVN(nvnTextureViewSetLevels, void, NVNtextureView *v, int base, int n) { VW(v)->set_levels = 1; VW(v)->base_level = base; VW(v)->nlevels = n; }
NVN(nvnTextureViewSetLayers, void, NVNtextureView *v, int min, int n) { VW(v)->set_layers = 1; VW(v)->min_layer = min; VW(v)->nlayers = n; }
NVN(nvnTextureViewSetFormat, void, NVNtextureView *v, NVNformat f) { VW(v)->set_format = 1; VW(v)->format = f; }
NVN(nvnTextureViewSetTarget, void, NVNtextureView *v, NVNtextureTarget t) { VW(v)->set_target = 1; VW(v)->target = t; }
NVN(nvnTextureViewGetLayers, NVNboolean, const NVNtextureView *v, int *min, int *n) {
    const view_t *o = (const view_t *)v; *min = o->min_layer; *n = o->nlayers; return o->set_layers;
}
NVN(nvnTextureGetViewOffset, ptrdiff_t, const NVNtexture *t, const NVNtextureView *v) {
    const tex_t *o = CTX(t); const view_t *w = (const view_t *)v;
    if (!w) return 0;
    uint64_t off = w->set_levels ? nvn_tex_level_offset(&o->d, (unsigned)w->base_level) : 0;
    if (w->set_layers) off += (uint64_t)w->min_layer * o->L.layer_size;
    return (ptrdiff_t)off;
}
// CPU texture read (NVN ReadTexels): for packaged textures the pool memory is the ground truth (de-layout and read directly); GPU-written textures are read back from the MTLTexture (nvn_mtl.m)
static void read_texels(const tex_t *o, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, ptrdiff_t row, ptrdiff_t img) {
    int bw, bh, bpb = nvn_fmt_bpb(o->d.format, &bw, &bh);
    uint32_t wb = (uint32_t)((r->width + bw - 1) / bw) * bpb, hb = (uint32_t)(r->height + bh - 1) / bh;
    if (!row) row = wb;
    if (!img) img = row * hb;
    unsigned level = v && v->set_levels ? (unsigned)v->base_level : 0, layer0 = v && v->set_layers ? (unsigned)v->min_layer : 0;
    aq_drain();   // what queued submissions draw into it first
    gfx_read_texels(o, level, layer0, r, dst, (size_t)row, (size_t)img);
}
NVN(nvnTextureReadTexels, void, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, void *dst) {
    read_texels(CTX(t), (const view_t *)v, r, dst, 0, 0);
}
NVN(nvnTextureReadTexelsStrided, void, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, void *dst, ptrdiff_t row, ptrdiff_t img) {
    read_texels(CTX(t), (const view_t *)v, r, dst, row, img);
}
NVN(nvnTextureInvalidateTexels, void, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r) { (void)t; (void)v; (void)r; }

// ---------- Samplers ----------
#define SB(b) (&((samp_builder_t *)(b))->p)
NVN(nvnSamplerBuilderSetDefaults, void, NVNsamplerBuilder *b) {
    NVNdevice *dev = ((samp_builder_t *)b)->dev; memset(b, 0, sizeof(samp_builder_t)); ((samp_builder_t *)b)->dev = dev;
    samp_par *o = SB(b);
    o->minf = NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR; o->magf = NVN_MAG_FILTER_LINEAR;   // GL defaults
    o->ws = o->wt = o->wr = NVN_WRAP_MODE_REPEAT; o->cmp_func = NVN_COMPARE_FUNC_LESS;
    o->lod_min = -1000.f; o->lod_max = 1000.f; o->aniso = 1.f;
}
NVN(nvnSamplerBuilderSetDevice, void, NVNsamplerBuilder *b, NVNdevice *d) { ((samp_builder_t *)b)->dev = d; }
NVN(nvnSamplerBuilderSetMinMagFilter, void, NVNsamplerBuilder *b, NVNminFilter mi, NVNmagFilter ma) { SB(b)->minf = mi; SB(b)->magf = ma; }
NVN(nvnSamplerBuilderSetWrapMode, void, NVNsamplerBuilder *b, NVNwrapMode s, NVNwrapMode t, NVNwrapMode r) { SB(b)->ws = s; SB(b)->wt = t; SB(b)->wr = r; }
NVN(nvnSamplerBuilderSetLodClamp, void, NVNsamplerBuilder *b, float lo, float hi) { SB(b)->lod_min = lo; SB(b)->lod_max = hi; }
NVN(nvnSamplerBuilderSetLodBias, void, NVNsamplerBuilder *b, float v) { SB(b)->lod_bias = v; }
NVN(nvnSamplerBuilderSetCompare, void, NVNsamplerBuilder *b, NVNcompareMode m, NVNcompareFunc f) { SB(b)->cmp_mode = m; SB(b)->cmp_func = f; }
NVN(nvnSamplerBuilderSetBorderColor, void, NVNsamplerBuilder *b, const float *c) { memcpy(SB(b)->border, c, 16); }
// Metal's fixed border colours are selected numerically, also for integer textures. Bit-casting integer 1 to float would select transparent black instead of opaque black/white.
NVN(nvnSamplerBuilderSetBorderColori, void, NVNsamplerBuilder *b, const int *c) { for (int i = 0; i < 4; i++) SB(b)->border[i] = (float)c[i]; }
NVN(nvnSamplerBuilderSetBorderColorui, void, NVNsamplerBuilder *b, const uint32_t *c) { for (int i = 0; i < 4; i++) SB(b)->border[i] = (float)c[i]; }
NVN(nvnSamplerBuilderSetMaxAnisotropy, void, NVNsamplerBuilder *b, float v) { SB(b)->aniso = v; }
NVN(nvnSamplerBuilderSetLodSnap, void, NVNsamplerBuilder *b, float v) { (void)b; (void)v; }   // No Metal equivalent, ignored
NVN(nvnSamplerBuilderGetMinMagFilter, void, const NVNsamplerBuilder *b, NVNminFilter *mi, NVNmagFilter *ma) {
    const samp_par *o = &((const samp_builder_t *)b)->p; *mi = (NVNminFilter)o->minf; *ma = (NVNmagFilter)o->magf;
}
NVN(nvnSamplerBuilderGetWrapMode, void, const NVNsamplerBuilder *b, NVNwrapMode *s, NVNwrapMode *t, NVNwrapMode *r) {
    const samp_par *o = &((const samp_builder_t *)b)->p; *s = (NVNwrapMode)o->ws; *t = (NVNwrapMode)o->wt; *r = (NVNwrapMode)o->wr;
}
NVN(nvnSamplerBuilderGetLodClamp, void, const NVNsamplerBuilder *b, float *lo, float *hi) {   // Bayonetta 3 reads it back while building its samplers
    const samp_par *o = &((const samp_builder_t *)b)->p; *lo = o->lod_min; *hi = o->lod_max;
}
NVN(nvnSamplerInitialize, NVNboolean, NVNsampler *s, const NVNsamplerBuilder *b) {
    samp_t *o = (samp_t *)s; o->magic = MAGIC('S', 'M', 'P', 0); o->p = ((const samp_builder_t *)b)->p; return 1;
}
NVN(nvnSamplerFinalize, void, NVNsampler *s) { ((samp_t *)s)->magic = 0; }

// ---------- Programs ----------
NVN(nvnProgramInitialize, NVNboolean, NVNprogram *p, NVNdevice *d) {
    (void)d; memset(p, 0, sizeof(prog_t)); ((prog_t *)p)->magic = MAGIC('P', 'R', 'O', 'G'); return 1;
}
static void pk_drop(const void *p);
NVN(nvnProgramFinalize, void, NVNprogram *p) { gfx_prog_release((prog_t *)p); ((prog_t *)p)->magic = 0; pk_drop(p); }
// a program's per-stage keys (side table: guest NVNprogram address → host object; bit i of ok = stage i computed). nxnative kept it in a CFDictionary
typedef struct { uint8_t key[6][8]; uint8_t ok; } prog_keys_t;
typedef struct pk_ent { const void *p; prog_keys_t *k; struct pk_ent *next; } pk_ent;
enum { PK_BUCKETS = 4096 };
static pk_ent *g_pk[PK_BUCKETS]; static pthread_mutex_t g_pk_mu = PTHREAD_MUTEX_INITIALIZER;
static pk_ent **pk_slot(const void *p) { pk_ent **e = &g_pk[((uintptr_t)p >> 4) * 0x9E3779B97F4A7C15ull >> 52]; while (*e && (*e)->p != p) e = &(*e)->next; return e; }   // g_pk_mu held
static prog_keys_t *pk_get(const void *p, int make) {
    pthread_mutex_lock(&g_pk_mu);
    pk_ent **e = pk_slot(p); prog_keys_t *k = *e ? (*e)->k : NULL;
    if (!k && make) { pk_ent *n = calloc(1, sizeof *n); k = calloc(1, sizeof *k); if (!n || !k) die("program key table allocation failed"); n->p = p; n->k = k; *e = n; }
    pthread_mutex_unlock(&g_pk_mu);
    return k;
}
static void pk_drop(const void *p) {
    pthread_mutex_lock(&g_pk_mu);
    pk_ent **e = pk_slot(p), *n = *e;
    if (n) { *e = n->next; free(n->k); free(n); }
    pthread_mutex_unlock(&g_pk_mu);
}
// SHA-256 of a + b (nxnative used CommonCrypto)
static void sha256_2(const void *a, size_t na, const void *b, size_t nb, uint8_t out[32]) {
    static BCRYPT_ALG_HANDLE alg; static INIT_ONCE once = INIT_ONCE_STATIC_INIT; BOOL pending;
    if (InitOnceBeginInitialize(&once, 0, &pending, NULL) && pending) {
        if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, NULL, 0)) die("SHA-256: BCryptOpenAlgorithmProvider failed");
        InitOnceComplete(&once, 0, NULL);
    }
    BCRYPT_HASH_HANDLE h = NULL;
    if (BCryptCreateHash(alg, &h, NULL, 0, NULL, 0, 0) || BCryptHashData(h, (PUCHAR)a, (ULONG)na, 0) || (nb && BCryptHashData(h, (PUCHAR)b, (ULONG)nb, 0)) ||
        BCryptFinishHash(h, out, 32, 0)) die("SHA-256 (BCrypt) failed");
    BCryptDestroyHash(h);
}
static int nvn2_ctl(const void *c) { uint32_t magic, ver; memcpy(&magic, c, 4); memcpy(&ver, (const uint8_t *)c + 4, 4); return magic == 0x98761234u && ver == 0xc9; }
// mapped = 1: computed only when the code area is in a registered memory pool (at SetShaders: Mario sets some programs with an unmapped address first and fills it in before the first draw)
static int keys_hash(const NVNshaderData *sd, int mapped, uint8_t out[8]) {
    uint32_t ncode = 0; void *mb; uint64_t mo;
    if (!sd->control) return 0;
    memcpy(&ncode, (const uint8_t *)sd->control + 0x704, 4);
    // SWITCH2 (private bring-up): GLSLC 0x00220015 control blocks are 0xe0 bytes (+4 = 0xc9, +8 = 0x11, +0x18 = the version); the code area is +0x40 (instructions and padding) plus +0x30 (the 0xe0 reflection trailer)
    int s2 = nvn2_ctl(sd->control); if (s2) memcpy(&ncode, (const uint8_t *)sd->control + 0x44, 4);   // the code area's length; NVN2 sections are 0xd0-0xe0 bytes, +0x704 is past them
    int code = sd->data && ncode && ncode < (1u << 24);
    if (mapped && !(code && gfx_addr(sd->data, ncode, &mb, &mo))) return 0;
    // Control +0x20 holds the section's own length: sections of at least 0x880 bytes (all of Mario's, Engage's, Dread's and 13 Sentinels') use the first 0x880; SDK 10's 3D World has 0x850-byte ones followed by other data, so only their own length is used (shader_split.py ctl_len)
    uint32_t nctl = 0x880, own; memcpy(&own, (const uint8_t *)sd->control + 0x20, 4);
    if (own >= 0x760 && own < 0x880) nctl = own;
    if (s2) nctl = 0x48;   // NVN2: the first 0x48 bytes, through the shader's own hash at +0x20 and the code lengths, +0x30 as zero (nxnative hle_nvn.c, shader_split.py key_ctl: its map's keys)
    uint8_t ctl[0x880], md[32]; memcpy(ctl, sd->control, nctl);
    // Profile key shader_key_skip: bytes of the control section hashed as zeros (shader_split.py key_ctl). Batman: Arkham Knight's engine shares one control section among shaders whose
    // sections differ only in GLSLC's compile information (+0x778..+0x791, +0x7d0..+0x7e7): it hands a program the section of an earlier shader with the same rest, so the bytes there name another shader
    if (DM_GAME_SHADER_KEY_SKIP_LEN && DM_GAME_SHADER_KEY_SKIP_OFF + DM_GAME_SHADER_KEY_SKIP_LEN <= nctl) memset(ctl + DM_GAME_SHADER_KEY_SKIP_OFF, 0, DM_GAME_SHADER_KEY_SKIP_LEN);
    else if (s2) memset(ctl + 0x30, 0, 4);
    sha256_2(ctl, nctl, code ? (const void *)(uintptr_t)sd->data : NULL, code ? ncode : 0, md); memcpy(out, md, 8);
    return 1;
}
int prog_key(const prog_t *p, unsigned st, uint8_t key[8]) {
    if (st >= p->nstage) return 0;
    prog_keys_t *k = pk_get(p, 1);
    if (!(k->ok >> st & 1)) { keys_hash(&p->sd[st], 0, k->key[st]); k->ok |= (uint8_t)(1u << st); }
    memcpy(key, k->key[st], 8);
    return 1;
}
NVN(nvnProgramSetShaders, NVNboolean, NVNprogram *p, int n, const NVNshaderData *sd) {
    prog_t *o = (prog_t *)p;
    if (n < 0 || n > 6) { LOG("[RED] ProgramSetShaders stage count %d", n); return 0; }
    gfx_prog_release(o); o->nstage = n; memcpy(o->sd, sd, n * sizeof *sd); o->serial = nvn_new_serial();
    // The keys are computed here (see prog_key in nvn_obj.h): the control section is only guaranteed valid during this call; the code area lives in a memory pool and stays valid while the program is in use
    static int dump = -1; if (dump < 0) dump = getenv("DM_NVN_SHDUMP") != NULL || nxs_rt_enabled();   // nxs_rt (gfx/nxs_rt.h) keys and dumps programs at replay
    static const char *trace; static int trace_init;
    if (!trace_init) { trace = getenv("DM_NVN_PROGTRACE"); trace_init = 1; }
    prog_keys_t *k = pk_get(p, 1);
    k->ok = 0;
    for (int i = 0; i < n; i++) {
        // SWITCH2 (private bring-up): with a Switch 2 control block the code area is not final at SetShaders (4 of 6 keys hashed here differed from the code at the first draw); key it lazily in prog_key
        int s2_ = sd[i].control && nvn2_ctl(sd[i].control);
        if (!s2_ && keys_hash(&sd[i], 1, k->key[i])) k->ok |= (uint8_t)(1u << i);
        // Diagnostic: trace the loader of selected shader keys (comma-separated hex).
        if (trace && (k->ok & (1u << i))) {
            char key[17]; for (int j = 0; j < 8; j++) snprintf(key + j * 2, 3, "%02x", k->key[i][j]);
            if (strstr(trace, key)) {
                LOG("[nvn] ProgramSetShaders key=%s control=%p code=%#llx", key, sd[i].control, (unsigned long long)sd[i].data);
                dm_backtrace(DM_GUEST_FP());
            }
        }
        // DM_NVN_SHDUMP (debugging) or a shader cache: keep a copy of the control section so the replay layer can key and save shaders missing from the map (never freed: the
        // game may copy the program object by value). A Switch 2 block is its 0xe0 bytes, the rest zeros (nxnative copied 0x880 bytes, reading past a block at the end of its memory)
        if (dump && sd[i].control) { uint8_t *c = calloc(1, 0x880); if (c) { memcpy(c, sd[i].control, s2_ ? 0xe0 : 0x880); o->sd[i].control = c; } }
    }
    OLOG("ProgramSetShaders %p n=%d data0=%#llx", (void *)p, n, n ? (unsigned long long)sd[0].data : 0ull);
    return 1;
}

// ---------- Sync / events ----------
NVN(nvnSyncInitialize, NVNboolean, NVNsync *s, NVNdevice *d) { (void)d; *(sync_t *)s = (sync_t){MAGIC('S', 'Y', 'N', 'C'), 0, 0}; return 1; }
NVN(nvnSyncFinalize, void, NVNsync *s) { ((sync_t *)s)->magic = 0; }
NVN(nvnSyncSetDebugLabel, void, NVNsync *s, const char *l) { (void)s; (void)l; }
// Fence = "all commands previously submitted to the queue have completed": record the fence value of the most recent submit (with asynchronous submission, possibly still
// a queued submission's sequence number, resolved when it is waited on)
NVN(nvnQueueFenceSync, void, NVNqueue *q, NVNsync *s, NVNsyncCondition c, int f) { (void)q; (void)c; (void)f; ((sync_t *)s)->fence = aq_fence_now(); ((sync_t *)s)->fenced = 1; }
// A sync object that was never fenced waits until timeout, as on a real device; a fenced one is released only after the GPU has actually completed up to that fence value
// A GPU-side wait on a sync: the host has a single Metal queue that executes submissions in order, so commands before the fence are always done first and nothing needs waiting. A sync not fenced yet (fenced later by another thread) is not waited for here; logged once
NVN(nvnQueueWaitSync, NVNboolean, NVNqueue *q, const NVNsync *s) { (void)q; { static long n; if (!(n++ & 1023)) LOG("[nvn] QueueWaitSync call %ld", n); } static int once; if (!((const sync_t *)s)->fenced && !once++) LOG("[nvn] QueueWaitSync: the sync object has no fence yet, the host does not wait"); return 1; }
// ZCull data: the host does no ZCull, so there is nothing to save / restore
NVN(nvnCommandBufferRestoreZCullData, void, NVNcommandBuffer *c, NVNbufferAddress a, size_t n) { (void)c; (void)a; (void)n; }
NVN(nvnCommandBufferSaveZCullData, void, NVNcommandBuffer *c, NVNbufferAddress a, size_t n) { (void)c; (void)a; (void)n; }
// SWR_TRACE spans of the calls that make up a frame's critical path (runtime/trace.c; off outside the window)
extern volatile int g_swr_trace_on; unsigned long long swr_trace_ts(void); void swr_trace_span(const char *, unsigned long long);
#define TRS0() unsigned long long trs0_ = g_swr_trace_on ? swr_trace_ts() : 0
#define TRS1(name) do { if (trs0_) swr_trace_span(name, trs0_); } while (0)
static _Atomic uint64_t g_sync_ns;   // time game threads spent blocked in nvnSyncWait (waiting for the GPU), in the frame statistics
NVN(nvnSyncWait, NVNsyncWaitResult, const NVNsync *s, uint64_t timeout) {
    sync_t *o = (sync_t *)s;
    if (!o->fenced && timeout && aq_busy()) aq_drain();   // a fence inside a command buffer still queued for translation (OP_FENCE_SYNC): it is set once that is translated
    if (!o->fenced) return NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED;
    int (*w)(uint64_t, uint64_t) = o->fenced == 2 ? gfx_wait_acq : gfx_wait;
    uint64_t fv = o->fence;
    if (!aq_resolve(&fv, timeout)) return NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED;   // a queued submission's fence: its translation first
    if (fv != o->fence) o->fence = fv;
    if (w(fv, 0)) return NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED;
    TRS0(); uint64_t t0 = now_ns(); int ok = w(fv, timeout); atomic_fetch_add(&g_sync_ns, now_ns() - t0); TRS1("nvnSyncWait (blocked)");   // the frame's blocked SyncWait time (@sync_ms)
    return ok ? NVN_SYNC_WAIT_RESULT_CONDITION_SATISFIED : NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED;
}
NVN(nvnEventBuilderSetDefaults, void, NVNeventBuilder *b) { memset(b, 0, sizeof(ev_builder_t)); }
NVN(nvnEventBuilderSetStorage, void, NVNeventBuilder *b, const NVNmemoryPool *p, int64_t off) { *(ev_builder_t *)b = (ev_builder_t){(const pool_t *)p, off}; }
NVN(nvnEventInitialize, NVNboolean, NVNevent *e, const NVNeventBuilder *b) {
    const ev_builder_t *s = (const ev_builder_t *)b;
    if (!s->pool || !s->pool->mem) { LOG("[RED] EventInitialize has no storage"); return 0; }
    *(ev_t *)e = (ev_t){MAGIC('E', 'V', 'N', 'T'), (volatile uint32_t *)(s->pool->mem + s->off)}; return 1;
}
NVN(nvnEventFinalize, void, NVNevent *e) { ((ev_t *)e)->magic = 0; }
NVN(nvnEventGetValue, uint32_t, const NVNevent *e) { return *((const ev_t *)e)->val; }
static void ev_signal(const ev_t *e, int mode, uint32_t v) { if (mode == NVN_EVENT_SIGNAL_MODE_ADD) *e->val += v; else *e->val = v; }
NVN(nvnEventSignal, void, NVNevent *e, NVNeventSignalMode m, uint32_t v) { ev_signal((const ev_t *)e, m, v); }

// ---------- Queue ----------
#define QB(b) ((queue_builder_t *)(b))
NVN(nvnQueueBuilderSetDefaults, void, NVNqueueBuilder *b) {
    NVNdevice *dev = QB(b)->dev; memset(b, 0, sizeof(queue_builder_t)); QB(b)->dev = dev;
    QB(b)->cmd_sz = 65536; QB(b)->comp_sz = 262144; QB(b)->ctl_sz = 16384; QB(b)->flush = 4096;   // Device defaults
}
NVN(nvnQueueBuilderSetDevice, void, NVNqueueBuilder *b, NVNdevice *d) { QB(b)->dev = d; }
NVN(nvnQueueBuilderSetFlags, void, NVNqueueBuilder *b, int f) { QB(b)->flags = f; }
NVN(nvnQueueBuilderSetCommandMemorySize, void, NVNqueueBuilder *b, size_t n) { QB(b)->cmd_sz = n; }
NVN(nvnQueueBuilderSetComputeMemorySize, void, NVNqueueBuilder *b, size_t n) { QB(b)->comp_sz = n; }
NVN(nvnQueueBuilderSetControlMemorySize, void, NVNqueueBuilder *b, size_t n) { QB(b)->ctl_sz = n; }
NVN(nvnQueueBuilderSetCommandFlushThreshold, void, NVNqueueBuilder *b, size_t n) { QB(b)->flush = n; }
// Queue memory: the sum of the command + compute + control blocks rounded to 4 KiB (the host does not use it; the game only needs to allocate this size)
NVN(nvnQueueBuilderGetQueueMemorySize, size_t, const NVNqueueBuilder *b) {
    const queue_builder_t *o = (const queue_builder_t *)b;
    return (o->cmd_sz + o->comp_sz + o->ctl_sz + 4095) & ~(size_t)4095;
}
NVN(nvnQueueBuilderSetQueueMemory, void, NVNqueueBuilder *b, void *m, size_t n) { QB(b)->mem = m; QB(b)->mem_sz = n; }
NVN(nvnQueueInitialize, NVNboolean, NVNqueue *q, const NVNqueueBuilder *b) {
    NVNdevice *d = ((const queue_builder_t *)b)->dev;
    // NVN emits SET_WINDOW_ORIGIN / FLIP_Y into the queue's initial GPU state.
    // Changing the device origin later changes viewport construction, but does
    // not rewrite that queue state (Xenoblade 2 creates its queue first).
    *(queue_t *)q = (queue_t){MAGIC('Q', 'U', 'E', 'U'), d, 0,
                            !d || ((const dev_t_ *)d)->origin == NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT, nvn_new_serial()};
    OLOG("QueueInitialize %p", (void *)q); return 1;
}
NVN(nvnQueueFinalize, void, NVNqueue *q) {
    aq_drain();
    pthread_mutex_lock(&g_sub_mu); gfx_queue_forget(((queue_t *)q)->serial); ((queue_t *)q)->magic = 0; pthread_mutex_unlock(&g_sub_mu);
}
NVN(nvnQueueGetError, NVNqueueGetErrorResult, NVNqueue *q, NVNqueueErrorInfo *info) {
    (void)q; if (info) memset(info, 0, sizeof *info); return NVN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR;
}
NVN(nvnQueueFlush, void, NVNqueue *q) { (void)q; }
NVN(nvnQueueFinish, void, NVNqueue *q) { (void)q; TRS0(); aq_drain(); gfx_finish(); TRS1("nvnQueueFinish"); }

// ---------- Command buffer: record into the game's command memory, perform CPU-visible side effects at submit ----------
#define CB(c) ((cmd_t *)(c))
// Room kept at the segment end after every record but the END: one JUMP. The END needs none (nothing follows it), and every other record is at least 16 bytes, so a JUMP
// only ever stands where a record was to go: linking never writes past the bytes the recording takes in one piece, which is what GetCommandMemoryUsed reports and games size
// command memory by. Tears of the Kingdom (GitHub issue #61) gives each model's prerecorded lists fixed slots of the size a recording of that list measured, slot k at
// base + size × k, and passes AddCommandMemory max(size, 512) (main+0xbb120). With 24 bytes kept for "a JUMP or an END", a 496-byte list's END did not fit in 512,
// went to a host segment, and the 16-byte JUMP at 488 reached 8 bytes into the next slot; recording that slot wrote its first record over the JUMP's target, and the
// replay jumped to 0x1000000003 (that CALL record's header) after 8 to 30 minutes of play
#define OP_RESERVE 16

NVN(nvnCommandBufferInitialize, NVNboolean, NVNcommandBuffer *c, NVNdevice *d) {
    memset(c, 0, sizeof(cmd_t)); CB(c)->magic = MAGIC('C', 'M', 'D', 'B'); CB(c)->dev = d; return 1;
}
NVN(nvnCommandBufferFinalize, void, NVNcommandBuffer *c) { CB(c)->magic = 0; }
NVN(nvnCommandBufferSetMemoryCallback, void, NVNcommandBuffer *c, PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC cb) { CB(c)->cb = cb; }
NVN(nvnCommandBufferSetMemoryCallbackData, void, NVNcommandBuffer *c, void *d) { CB(c)->cb_data = d; }
static uint8_t *align8(uint8_t *p) { return (uint8_t *)(((uintptr_t)p + 7) & ~(uintptr_t)7); }
// Diagnostic DM_NVN_CMDSTAT=1: every 300 frames log one line of command recording usage -- how many bytes / records were recorded per frame, how large a command memory segment the game gives each time, how many out-of-memory callbacks per frame, how many host-supplied segments,
// and the count and bytes of each command type (to find "how much larger our records are than real-device commands"). Counts only; does not change recording / replay
static struct { int on; uint64_t bytes, recs, grants, grant_bytes, grant_min, grant_max, cbs, hostseg, op_n[64], op_b[64], frame0, waste; } g_cs = { .on = -1 };
static pthread_mutex_t g_cs_mu = PTHREAD_MUTEX_INITIALIZER;
uint64_t dm_nvn_frame_count(void);
static int cs_on(void) { if (g_cs.on < 0) g_cs.on = getenv("DM_NVN_CMDSTAT") != NULL; return g_cs.on; }
static void cs_op(uint32_t op, uint32_t size) { if (!cs_on() || op >= 64) return; pthread_mutex_lock(&g_cs_mu); g_cs.op_n[op]++; g_cs.op_b[op] += size; pthread_mutex_unlock(&g_cs_mu); }
static void cs_dump(void) {   // Called once per frame (at present)
    if (!cs_on()) return; uint64_t f = dm_nvn_frame_count(); if (f - g_cs.frame0 < 300) return;
    pthread_mutex_lock(&g_cs_mu); double n = (double)(f - g_cs.frame0); char top[700]; int k = 0; uint8_t used[64] = {0};
    for (int r = 0; r < 8; r++) { int bi = -1; for (int i = 0; i < 64; i++) if (!used[i] && g_cs.op_b[i] && (bi < 0 || g_cs.op_b[i] > g_cs.op_b[bi])) bi = i; if (bi < 0) break; used[bi] = 1;
        k += snprintf(top + k, sizeof top - k, " op %d: %.0f × %.0f bytes = %.1f KB;", bi, g_cs.op_n[bi] / n, (double)g_cs.op_b[bi] / (double)g_cs.op_n[bi], g_cs.op_b[bi] / n / 1024.0); }
    LOG("[nvn] command memory up to frame %llu: recorded per frame %.1f KB / %.0f records; segments from the game per frame %.2f, mean %.1f KB each (min %.1f / max %.1f); out-of-memory callbacks per frame %.2f; host-lent segments per frame %.3f; segment tails dropped on a switch per frame %.1f KB | per frame%s",
        (unsigned long long)f, g_cs.bytes / n / 1024.0, g_cs.recs / n, g_cs.grants / n, g_cs.grants ? g_cs.grant_bytes / (double)g_cs.grants / 1024.0 : 0, g_cs.grant_min / 1024.0, g_cs.grant_max / 1024.0, g_cs.cbs / n, g_cs.hostseg / n, g_cs.waste / n / 1024.0, top);
    int on = g_cs.on; memset(&g_cs, 0, sizeof g_cs); g_cs.on = on; g_cs.frame0 = f; pthread_mutex_unlock(&g_cs_mu);
}
static void put_hdr(uint8_t *p, uint32_t op, uint32_t size) { op_hdr h = {op, size}; memcpy(p, &h, sizeof h); if (op) cs_op(op, size); }
// New segment: if the recording in progress has records, write a JUMP at the end of the old segment to link it. A recording with none yet starts in the new segment
// (its first record is its handle), and the bytes at the old write position are not its own: after an END that took the last bytes there is no room there at all
static void cmd_new_segment(cmd_t *o, uint8_t *mem, size_t n) {
    uint8_t *start = align8(mem), *end = mem + n;
    if (o->recording == 1 && o->cur) { if (cs_on() && o->end > o->cur) { pthread_mutex_lock(&g_cs_mu); g_cs.waste += (uint64_t)(o->end - o->cur); pthread_mutex_unlock(&g_cs_mu); } put_hdr(o->cur, OP_JUMP, 16); memcpy(o->cur + 8, &start, 8);
        o->used += 16; }   // the JUMP also occupies guest memory of this recording: count it in used, so the game packing the next recording right behind by used does not overwrite it
    o->cur = start; o->end = end;
}
// Host-supplied segment when the game's own command memory runs out: when the game's out-of-memory callback cannot allocate from its command memory pool, it calls AddCommandMemory with an empty pool (p = NULL, n = 0)
// (opening the advanced stats page on the level 88 sorceress save: many list items in one frame, and the original implementation segfaulted here reading NULL+8). Our records are larger than real-device NVN commands, so the budget the game reserved for real-device usage is not necessarily enough,
// so this is not an error state of the game but something we must catch: provide a segment of host memory to keep recording. The segment is replayed in place at submit and reclaimed after being kept for 8 frames (submit is in the same frame as recording or the next one).
// (Do not use the whole game chunk as the range: SMBW sub-allocates many recordings out of one big chunk, and a later grant inside that chunk freed segments still in use.)
// Reclaim follows the game's command memory chunks, not a frame count: the game caches recorded command handles and CALLs them every frame until it re-records them itself (SMBW's object setup handles);
// once a handle was recorded into a host segment, the old "reclaim 8 frames after lending" rewrote the segment while the game still CALLed it (with overflow forced on the Mac, the replay read garbage commands: RGBA16F used as a depth format for a pipeline).
// A segment hangs off one recording through the JUMP at the end of its guest part: that recording occupies [BEGIN, end of JUMP) in guest memory, and the game only gives those bytes out again once they are dead
// (when it re-records that material in place; packing the next recording right behind by used never touches them, since used includes the JUMP). So a segment remembers that guest range and is reclaimed when the game gives out memory
// overlapping it; segments overflowing from a host segment keep the same range.
// The segment pool grows on demand (SMBW opening course: many small command buffers each overflow once per frame, and the old fixed 64 segments × 1 MB ran out within 8 frames); abort only beyond HOSTSEG_CAP in total.
// A segment is HOSTSEG_SZ, with two exceptions (hostseg_size): an empty pool with a length of at most HOSTSEG_SMALL outside a recording gets that much rounded up to HOSTSEG_SMALL
// (Super Mario Odyssey hands out many such 512-byte grants before recording; at 256 KB each, and with no recording to reclaim them by, they reached the cap entering the Wooded
// Kingdom), and a record larger than HOSTSEG_SZ gets a segment that holds it. A free segment is taken again only for a request of its own size; the cap counts every segment's bytes.
// An empty-pool grant outside a recording means the game's own allocator for command memory came back empty. On hardware a null pool cannot be written to, so a game only gets here
// when the host's records outgrew what it measured for them (Odyssey's "DisplayList" heap: docs/reference/memory.md, "Model heaps sized by the host's command record sizes").
// Those segments have no guest range to be reclaimed by, so they are not reclaimed; the cap's abort names the state, and the fix belongs in the record sizes
#define HOSTSEG_SZ (256u << 10)
#define HOSTSEG_SMALL (4u << 10)
#define HOSTSEG_CAP (1u << 30)
uint64_t dm_nvn_frame_count(void);
typedef struct { uint8_t *mem; size_t n; uint64_t frame; int busy; uintptr_t org_lo, org_hi; } hostseg_t;   // org: the guest range of the recording it hangs off (0 = unknown, never reclaimed)
static hostseg_t *g_hostseg; static int g_nhostseg, g_caphostseg; static uint64_t g_hostseg_bytes;
static pthread_mutex_t g_hostseg_mu = PTHREAD_MUTEX_INITIALIZER;
static size_t hostseg_size(const cmd_t *o, int empty_pool, size_t want) {
    if (empty_pool) return want && want <= HOSTSEG_SMALL && !o->recording ? HOSTSEG_SMALL : HOSTSEG_SZ;
    return want + 16 > HOSTSEG_SZ ? (want + 16 + 0xfff) & ~(size_t)0xfff : HOSTSEG_SZ;   // 16: room for the next JUMP (want holds OP_RESERVE already, this is slack for the alignment)
}
// Take a segment of n bytes: origin = the current recording's guest bytes [BEGIN, end of JUMP); if the write pointer is already in a host segment (overflowing again), keep that segment's origin
static uint8_t *hostseg_take(const cmd_t *o, size_t n) {
    static unsigned total; uint64_t f = dm_nvn_frame_count(); uint8_t *m = NULL;
    uintptr_t lo = 0, hi = 0, cur = o ? (uintptr_t)o->cur : 0;
    pthread_mutex_lock(&g_hostseg_mu);
    int in_seg = 0;   // by the segment's end too: a small segment can end where another one starts
    for (int i = 0; i < g_nhostseg && cur; i++) if (cur >= (uintptr_t)g_hostseg[i].mem && o->end == g_hostseg[i].mem + g_hostseg[i].n) { lo = g_hostseg[i].org_lo; hi = g_hostseg[i].org_hi; in_seg = 1; break; }
    if (!in_seg && o && o->recording && o->cur) { lo = o->recording == 1 && o->begin && (uintptr_t)o->begin <= cur ? (uintptr_t)o->begin : cur; hi = cur + 16; }
    for (int i = 0; i < g_nhostseg && !m; i++)
        if (!g_hostseg[i].busy && g_hostseg[i].n == n) { g_hostseg[i] = (hostseg_t){ g_hostseg[i].mem, n, f, 1, lo, hi }; m = g_hostseg[i].mem; }
    if (!m && g_hostseg_bytes + n <= HOSTSEG_CAP && (m = malloc(n))) {
        if (g_nhostseg == g_caphostseg) { g_caphostseg = g_caphostseg ? g_caphostseg * 2 : 64; g_hostseg = realloc(g_hostseg, sizeof *g_hostseg * (size_t)g_caphostseg); }
        g_hostseg[g_nhostseg++] = (hostseg_t){ m, n, f, 1, lo, hi }; g_hostseg_bytes += n;
    }
    unsigned t = ++total; int ns = g_nhostseg; uint64_t bytes = g_hostseg_bytes;
    pthread_mutex_unlock(&g_hostseg_mu);
    if (!m) die("NVN command memory: the game's pool is empty and host-lent segments reached %u MB without being reclaimed by the game's blocks", HOSTSEG_CAP >> 20);
    if (t == 1 || t % 1024 == 0) LOG("[nvn] the game's command memory ran short, the host lends a segment of %zu KB (%u times so far, frame %llu, segment pool %d segments, %llu KB)", n >> 10, t, (unsigned long long)f, ns, (unsigned long long)(bytes >> 10));
    return m;
}
// The game gives out [lo, hi): segments whose recording's guest bytes overlap it are dead, reclaim them
static void hostseg_release(uintptr_t lo, uintptr_t hi) {
    pthread_mutex_lock(&g_hostseg_mu);
    for (int i = 0; i < g_nhostseg; i++) if (g_hostseg[i].busy && g_hostseg[i].org_hi > lo && g_hostseg[i].org_lo < hi) g_hostseg[i].busy = 0;
    pthread_mutex_unlock(&g_hostseg_mu);
}
NVN(nvnCommandBufferAddCommandMemory, void, NVNcommandBuffer *c, const NVNmemoryPool *p, ptrdiff_t off, size_t n) {
    const pool_t *pl = (const pool_t *)p;
    if (!pl || n < 64) { if (cs_on()) { pthread_mutex_lock(&g_cs_mu); g_cs.hostseg++; pthread_mutex_unlock(&g_cs_mu); } size_t sz = hostseg_size(CB(c), !pl, n); cmd_new_segment(CB(c), hostseg_take(CB(c), sz), sz); return; }
    if (!pl->mem) { LOG("[RED] AddCommandMemory: pool has no CPU storage"); return; }
    hostseg_release((uintptr_t)(pl->mem + off), (uintptr_t)(pl->mem + off + n));
    if (cs_on()) { pthread_mutex_lock(&g_cs_mu); g_cs.grants++; g_cs.grant_bytes += n; if (!g_cs.grant_min || n < g_cs.grant_min) g_cs.grant_min = n; if (n > g_cs.grant_max) g_cs.grant_max = n; pthread_mutex_unlock(&g_cs_mu); }
    { static int cblog = -1; if (cblog < 0) cblog = getenv("DM_NVN_CBLOG") != NULL; static uint64_t ng; if (cblog && (++ng <= 60 || ng % 500 == 0)) LOG("[nvn] AddCommandMemory #%llu cmd=%p pool %p +%#llx length %#zx (recording %d)", (unsigned long long)ng, (void *)c, (const void *)pl, (long long)off, n, CB(c)->recording); }
    cmd_new_segment(CB(c), pl->mem + off, n);
    CB(c)->used = 0;   // GetCommandMemoryUsed counts from the start of the memory the game last added, across recordings (see there)
}
NVN(nvnCommandBufferAddControlMemory, void, NVNcommandBuffer *c, void *m, size_t n) {
    (void)m;
    CB(c)->ctl_size = n;   // The host's recordings contain no control-memory entries; the entire last block remains free.
}
// Get space for one record and keep `keep` bytes after it (OP_RESERVE, 0 for the END); if not enough, call back into the game per the NVN convention to add command memory
static uint8_t *cmd_alloc_keep(cmd_t *o, uint32_t size, uint32_t keep) {
    size = (size + 7) & ~7u;
    for (int tries = 0; !o->cur || o->cur + size + keep > o->end; tries++) {
        // The game's callback cannot supply enough (newer-SDK games hand out small chunks sized for real-device commands, or disable the callback and manage memory themselves): continue recording in a host segment instead of aborting
        if (!o->cb || tries == 2) {
            static int once; if (!once++) LOG("[nvn] command memory: the game's callback did not add enough (cb=%p, need %u bytes), using a host-lent segment", (void *)o->cb, size + keep);
            size_t sz = hostseg_size(o, 0, size + keep); cmd_new_segment(o, hostseg_take(o, sz), sz); break;
        }
        if (cs_on()) { pthread_mutex_lock(&g_cs_mu); g_cs.cbs++; pthread_mutex_unlock(&g_cs_mu); }
        { static int cblog = -1; if (cblog < 0) cblog = getenv("DM_NVN_CBLOG") != NULL; static uint64_t ncb;
          if (cblog && (++ncb <= 40 || ncb % 1000 == 0)) LOG("[nvn] command memory callback #%llu cmd=%p need %u (used in this segment %lld)", (unsigned long long)ncb, (void *)o, size + keep, o->cur && o->end ? (long long)(o->cur - (o->end - 1)) : -1LL); }
        GUEST_CALL(o->cb, (uint64_t)(uintptr_t)o, (uint64_t)NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY, (uint64_t)(size + keep), (uint64_t)(uintptr_t)o->cb_data);   // a guest function
    }
    if (cs_on()) { pthread_mutex_lock(&g_cs_mu); g_cs.bytes += size; g_cs.recs++; pthread_mutex_unlock(&g_cs_mu); }
    uint8_t *p = o->cur; o->cur += size; o->used += size; put_hdr(p, 0, size);
    if (o->recording == 2) { o->begin = p; o->recording = 1; }   // The recording's first record is its handle
    return p;
}
static uint8_t *cmd_alloc(cmd_t *o, uint32_t size) { return cmd_alloc_keep(o, size, OP_RESERVE); }
// A recording starts with its first command, as in NVN: no BEGIN record, so an empty recording takes only its END. Games size command memory by GetCommandMemoryUsed
// (nvn_obj.h draw_rec): Breath of the Wild sums one scratch recording per mesh and per shape and render pass into each model's prerecorded commands, and with an
// 8-byte BEGIN on each of several hundred recordings a model came out a 4 KB page over its resource size table entry
NVN(nvnCommandBufferBeginRecording, void, NVNcommandBuffer *c) {
    cmd_t *o = CB(c);
    if (o->recording) LOG("[RED] BeginRecording re-entered");
    o->recording = 2;   // 2: recording, no command yet (cmd_alloc sets begin)
}
NVN(nvnCommandBufferEndRecording, NVNcommandHandle, NVNcommandBuffer *c) {
    cmd_t *o = CB(c);
    uint8_t *p = cmd_alloc_keep(o, 8, 0); put_hdr(p, OP_END, 8);   // nothing follows: the END may take the segment's last bytes
    o->recording = 0;
    return (NVNcommandHandle)o->begin;
}
NVN(nvnCommandBufferIsRecording, NVNboolean, const NVNcommandBuffer *c) { return ((const cmd_t *)c)->recording != 0; }
NVN(nvnCommandBufferCopyCommands, void, NVNcommandBuffer *c, int n, const NVNcommandHandle *h) {
    for (int i = 0; i < n; i++) { uint8_t *p = cmd_alloc(CB(c), 16); put_hdr(p, OP_CALL, 16); memcpy(p + 8, &h[i], 8); }
}
NVN(nvnCommandBufferSignalEvent, void, NVNcommandBuffer *c, const NVNevent *e, NVNeventSignalMode m, NVNeventSignalLocation loc, int flags, uint32_t v) {
    { static long n; if (!(n++ & 1023)) LOG("[nvn] CommandBufferSignalEvent call %ld", n); }
    (void)loc; (void)flags;
    uint8_t *p = cmd_alloc(CB(c), 24); put_hdr(p, OP_SIGNAL_EVENT, 24);
    const ev_t *ev = (const ev_t *)e; volatile uint32_t *val = ev->val; uint32_t mm = m;
    memcpy(p + 8, &val, 8); memcpy(p + 16, &mm, 4); memcpy(p + 20, &v, 4);
}
// A fence recorded in a command buffer: the replay fences the sync with the submission that reaches this point (OP_FENCE_SYNC). Unreal Engine's NVN RHI fences its command buffers this way (Octopath Traveler 0); Pokémon Legends: Z-A fences once a frame this way and waits on it
NVN(nvnCommandBufferFenceSync, void, NVNcommandBuffer *c, NVNsync *s, NVNsyncCondition cond, int flags) {
    (void)cond; (void)flags;
    uint8_t *p = cmd_alloc(CB(c), 16); put_hdr(p, OP_FENCE_SYNC, 16); memcpy(p + 8, &s, 8);
}
NVN(nvnCommandBufferResetCounter, void, NVNcommandBuffer *c, NVNcounterType t) {
    uint8_t *p = cmd_alloc(CB(c), 16); put_hdr(p, OP_RESET_COUNTER, 16); uint64_t tt = t; memcpy(p + 8, &tt, 8);
}
NVN(nvnCommandBufferReportCounter, void, NVNcommandBuffer *c, NVNcounterType t, NVNbufferAddress a) {
    uint8_t *p = cmd_alloc(CB(c), 24); put_hdr(p, OP_REPORT_COUNTER, 24); uint64_t tt = t; memcpy(p + 8, &tt, 8); memcpy(p + 16, &a, 8);
}
NVN(nvnCommandBufferCopyBufferToBuffer, void, NVNcommandBuffer *c, NVNbufferAddress src, NVNbufferAddress dst, size_t n, int flags) {
    (void)flags;
    uint8_t *p = cmd_alloc(CB(c), 32); put_hdr(p, OP_COPY_B2B, 32); uint64_t nn = n;
    memcpy(p + 8, &src, 8); memcpy(p + 16, &dst, 8); memcpy(p + 24, &nn, 8);
}
// uniform update: the data is copied into command memory at record time (NVN semantics: the caller may modify the original data after recording) and written into the buffer at submit
NVN(nvnCommandBufferUpdateUniformBuffer, void, NVNcommandBuffer *c, NVNbufferAddress buf, size_t bufsz, ptrdiff_t off, size_t n, const void *data) {
    (void)bufsz;
    uint32_t sz = (uint32_t)(24 + n);
    uint8_t *p = cmd_alloc(CB(c), sz); put_hdr(p, OP_UPDATE_UB, (sz + 7) & ~7u);
    uint64_t dst = buf + off, nn = n; memcpy(p + 8, &dst, 8); memcpy(p + 16, &nn, 8); memcpy(p + 24, data, n);
}

// ---------- Render commands: parameters are recorded into command memory in the payload format of nvn_obj.h; for replay see nvn_mtl.m ----------
static void *rec(NVNcommandBuffer *c, uint32_t op, size_t payload) {
    uint32_t sz = (uint32_t)((8 + payload + 7) & ~(size_t)7);
    uint8_t *p = cmd_alloc(CB(c), sz); put_hdr(p, op, sz);
    memset(p + 8, 0, sz - 8);
    return p + 8;
}
#define REC(c, OP, T) ((T *)rec((c), (OP), sizeof(T)))
// Binding records with arrays are allocated by the actual count: prefix fields + n elements (the replay side reads only the first n). Previously they were recorded at fixed length for the maximum count (bind vertex buffers 272 bytes, bind textures / samplers 280 bytes,
// while there are usually only 1–2 in practice), and the empty tail took 40–50% of the command bytes per frame; the advanced stats page of the level 88 sorceress recorded 12.5 MB per frame, using up the game's own command memory pool (about 105 segments x 128 KB per frame)
#define RECN(c, OP, T, ARR, N) ((T *)rec((c), (OP), offsetof(T, ARR) + sizeof(((T *)0)->ARR[0]) * (size_t)(N)))
// Draws carry the indirect / multi-draw fields (ind onwards) only when they use them: a plain indexed draw record is 0x28 bytes instead of 0x40 (nvn_obj.h draw_rec)
static void rec_draw(NVNcommandBuffer *c, p_draw d) { size_t n = d.ind || d.mdi_cnt ? sizeof d : offsetof(p_draw, ind); memcpy(rec(c, OP_DRAW, n), &d, n); }
static void rec_draw_idx(NVNcommandBuffer *c, p_draw_idx d) { size_t n = d.ind || d.mdi_cnt ? sizeof d : offsetof(p_draw_idx, ind); memcpy(rec(c, OP_DRAW_IDX, n), &d, n); }
static void copy_view(view_t *dst, uint8_t *has, const NVNtextureView *v) { *has = v != NULL; if (v) *dst = *(const view_t *)v; }
// Records with textures snapshot the texture objects (0x80 bytes each, after the record struct) and replay reads the snapshots: an NVN command no longer depends on the API object once recorded, and the game may finalize, free or reuse it before submitting.
// Unity (Engage) does so when it changes scenes; only the object pointer used to be recorded, and replay read other data (a texture 1869367126 high, a Metal assertion abort).
#define SNAP_OFF(T) ((sizeof(T) + 7) & ~(size_t)7)
#define REC_SNAP(c, OP, T, K) ((T *)rec((c), (OP), SNAP_OFF(T) + (size_t)(K) * sizeof(tex_t)))
static const tex_t *tex_snap(void *rec_, size_t off, int k, const NVNtexture *t, uint64_t ra) {
    if (!t) return NULL;
    tex_t *s = (tex_t *)((uint8_t *)rec_ + off) + k;
    memcpy(s, t, sizeof *s);
    // The object is invalid at record time (Unity finalizes and re-initializes the same render target object between recording and submitting): keep the object pointer as before and read it at submit, when it is usually valid again; replay still checks the magic
    if (s->magic != MAGIC('T', 'E', 'X', 0)) {
        static int red;
        if (red++ < 4) {   // the first 4 log the call site and the start of the object, to tell what the game passed
            module_t *m = module_of((void *)ra); const uint64_t *q = (const uint64_t *)t;
            LOG("[RED] texture object invalid while recording (uninitialized or already finalized): %p index %d, call site %s+%#llx, object start %llx %llx %llx %llx %llx %llx; read again at submit", (const void *)t, k,
                m ? m->name : "?", m ? (unsigned long long)(ra - (uint64_t)m->base) : (unsigned long long)ra,
                (unsigned long long)q[0], (unsigned long long)q[1], (unsigned long long)q[2], (unsigned long long)q[3], (unsigned long long)q[4], (unsigned long long)q[5]);
        }
        return (const tex_t *)t;
    }
    return s;
}
NVN(nvnCommandBufferSetRenderTargets, void, NVNcommandBuffer *c, int n, const NVNtexture *const *ct, const NVNtextureView *const *cv,
    const NVNtexture *dt, const NVNtextureView *dv) {
    if (n < 0 || n > 8) { LOG("[RED] SetRenderTargets count %d", n); n = n < 0 ? 0 : 8; }
    p_set_rt *r = REC_SNAP(c, OP_SET_RT, p_set_rt, n + 1); r->n = n;
    for (int i = 0; i < n; i++) { r->color[i] = ct ? tex_snap(r, SNAP_OFF(p_set_rt), i, ct[i], (uint64_t)(uintptr_t)DM_GUEST_RA()) : NULL; copy_view(&r->cview[i], &r->has_cview[i], cv ? cv[i] : NULL); }
    r->depth = tex_snap(r, SNAP_OFF(p_set_rt), n, dt, (uint64_t)(uintptr_t)DM_GUEST_RA()); copy_view(&r->dview, &r->has_dview, dv);
    for (int i = 0; i < n; i++) if (r->color[i]) nvn_pool_gpu_written(r->color[i]->pool);
    if (r->depth) nvn_pool_gpu_written(r->depth->pool);
}
NVN(nvnCommandBufferClearColor, void, NVNcommandBuffer *c, int index, const float *rgba, int mask) {
    p_clear_color *r = REC(c, OP_CLEAR_COLOR, p_clear_color); r->index = index; r->mask = mask; memcpy(r->rgba, rgba, 16);
}
// Clears of integer targets (Paper Mario: The Thousand-Year Door): recorded as floats, which both replay paths convert to the attachment's integer type (the load action's MTLClearColor,
// clear_color_tri); exact up to 2^24 in magnitude, and 0xffffffff rounds to 2^32, which converts back to it
NVN(nvnCommandBufferClearColori, void, NVNcommandBuffer *c, int index, const int *rgba, int mask) {
    p_clear_color *r = REC(c, OP_CLEAR_COLOR, p_clear_color); r->index = index; r->mask = mask; for (int i = 0; i < 4; i++) r->rgba[i] = (float)rgba[i];
}
NVN(nvnCommandBufferClearColorui, void, NVNcommandBuffer *c, int index, const uint32_t *rgba, int mask) {
    p_clear_color *r = REC(c, OP_CLEAR_COLOR, p_clear_color); r->index = index; r->mask = mask; for (int i = 0; i < 4; i++) r->rgba[i] = (float)rgba[i];
}
NVN(nvnCommandBufferClearDepthStencil, void, NVNcommandBuffer *c, float depth, uint32_t dmask, int stencil, int smask) {
    p_clear_ds *r = REC(c, OP_CLEAR_DS, p_clear_ds); *r = (p_clear_ds){depth, (int)B(dmask), stencil, smask};
}
NVN(nvnCommandBufferSetViewports, void, NVNcommandBuffer *c, int first, int n, const float *v) {
    if (n < 0 || first + n > 16) { LOG("[RED] SetViewports first=%d n=%d", first, n); return; }
    p_viewports *r = REC(c, OP_VIEWPORTS, p_viewports); r->first = first; r->n = n;
    for (int i = 0; i < n; i++) memcpy(r->v[i], v + 4 * i, 16);
}
NVN(nvnCommandBufferSetScissor, void, NVNcommandBuffer *c, int x, int y, int w, int h) { *REC(c, OP_SCISSOR, p_scissor) = (p_scissor){x, y, w, h}; }
NVN(nvnCommandBufferSetDepthRange, void, NVNcommandBuffer *c, float n, float f) { *REC(c, OP_DEPTH_RANGE, p_depth_range) = (p_depth_range){n, f}; }
NVN(nvnCommandBufferBindProgram, void, NVNcommandBuffer *c, const NVNprogram *p, int stages) {
    *REC(c, OP_BIND_PROGRAM, p_bind_program) = (p_bind_program){(const prog_t *)p, stages};
}
static void bind_state(NVNcommandBuffer *c, int kind, const void *st, size_t n) { p_bind_state *r = rec(c, OP_BIND_STATE, offsetof(p_bind_state, bytes) + n); r->kind = kind; memcpy(r->bytes, st, n); }   // only the state's own bytes (nvn_obj.h p_bind_state)
NVN(nvnCommandBufferBindBlendState, void, NVNcommandBuffer *c, const NVNblendState *st) { bind_state(c, 0, st, sizeof(blend_st)); }
NVN(nvnCommandBufferBindColorState, void, NVNcommandBuffer *c, const NVNcolorState *st) { bind_state(c, 1, st, sizeof(color_st)); }
NVN(nvnCommandBufferBindChannelMaskState, void, NVNcommandBuffer *c, const NVNchannelMaskState *st) { bind_state(c, 2, st, sizeof(chmask_st)); }
NVN(nvnCommandBufferBindMultisampleState, void, NVNcommandBuffer *c, const NVNmultisampleState *st) { bind_state(c, 3, st, sizeof(ms_st)); }
NVN(nvnCommandBufferBindPolygonState, void, NVNcommandBuffer *c, const NVNpolygonState *st) { bind_state(c, 4, st, sizeof(poly_st)); }
NVN(nvnCommandBufferBindDepthStencilState, void, NVNcommandBuffer *c, const NVNdepthStencilState *st) { bind_state(c, 5, st, sizeof(ds_st)); }
NVN(nvnCommandBufferBindVertexAttribState, void, NVNcommandBuffer *c, int n, const NVNvertexAttribState *st) {
    if (n < 0 || n > 16) { LOG("[RED] BindVertexAttribState n=%d", n); return; }
    p_bind_vattr *r = RECN(c, OP_BIND_VATTR, p_bind_vattr, st, n); r->first = 0; r->n = n; memcpy(r->st, st, 4 * n);
}
NVN(nvnCommandBufferBindVertexStreamState, void, NVNcommandBuffer *c, int n, const NVNvertexStreamState *st) {
    if (n < 0 || n > 16) { LOG("[RED] BindVertexStreamState n=%d", n); return; }
    p_bind_vstream *r = RECN(c, OP_BIND_VSTREAM, p_bind_vstream, st, n); r->first = 0; r->n = n; memcpy(r->st, st, 8 * n);
}
NVN(nvnCommandBufferBindVertexBuffers, void, NVNcommandBuffer *c, int first, int n, const NVNbufferRange *rg) {
    if (n < 0 || first + n > 16) { LOG("[RED] BindVertexBuffers first=%d n=%d", first, n); return; }
    if (n == 1 && (unsigned)first < 16 && rg[0].size <= UINT32_MAX) {
        *REC(c, OP_BIND_VBUF1, p_bind_vbuf1) = (p_bind_vbuf1){(uint32_t)first, (uint32_t)rg[0].size, rg[0].address}; return;
    }
    p_bind_vbuf *r = RECN(c, OP_BIND_VBUF, p_bind_vbuf, r, n); r->first = first; r->n = n; memcpy(r->r, rg, sizeof *rg * n);
}
// NVN2's task and mesh stages (6, 7) keep their bindings in the tessellation stages' places (3, 4): a program has mesh shaders or tessellation, never both (nxnative's st_host)
static inline NVNshaderStage st_host(NVNshaderStage st) { return (unsigned)st == 6 ? 3 : (unsigned)st == 7 ? 4 : st; }
NVN(nvnCommandBufferBindUniformBuffer, void, NVNcommandBuffer *c, NVNshaderStage st, int i, NVNbufferAddress a, size_t n) { st = st_host(st);
    if ((unsigned)st < 6 && (unsigned)i < 16 && n <= UINT32_MAX) {
        *REC(c, OP_BIND_UBO_SMALL, p_bind_ubo_small) = (p_bind_ubo_small){(uint16_t)st, (uint16_t)i, (uint32_t)n, a}; return;
    }
    *REC(c, OP_BIND_UBO, p_bind_buf) = (p_bind_buf){st, i, a, n};
}
NVN(nvnCommandBufferBindUniformBuffers, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, const NVNbufferRange *rg) { st = st_host(st);
    for (int i = 0; i < n; i++) h_nvnCommandBufferBindUniformBuffer(c, st, first + i, rg[i].address, rg[i].size);
}
NVN(nvnCommandBufferBindStorageBuffer, void, NVNcommandBuffer *c, NVNshaderStage st, int i, NVNbufferAddress a, size_t n) { st = st_host(st);
    *REC(c, OP_BIND_SSBO, p_bind_buf) = (p_bind_buf){st, i, a, n};
}
NVN(nvnCommandBufferBindStorageBuffers, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, const NVNbufferRange *rg) { st = st_host(st);
    for (int i = 0; i < n; i++) *REC(c, OP_BIND_SSBO, p_bind_buf) = (p_bind_buf){st, first + i, rg[i].address, rg[i].size};
}
static p_bind_handles *bind_handles(NVNcommandBuffer *c, int op, int st, int first, int n, const uint64_t *h) {
    if (n < 0 || n > 32) { LOG("[RED] bind handle count %d", n); return NULL; }
    p_bind_handles *r = RECN(c, op, p_bind_handles, h, n); r->stage = st; r->first = first; r->n = n; r->defer = 0; r->src = 0;
    // The driver writes each handle into the c2 table as a 32-bit word (texture ID in bits 0-19, sampler ID in bits 20-31), so the GPU never sees the upper half of the 64-bit handle.
    // EA SPORTS FC 27 binds separate textures with bit 32 set (0x1000001d1): read whole, its sampler ID was 4096, beyond any pool, where the GPU reads sampler 0
    for (int i = 0; i < n; i++) r->h[i] = (uint32_t)h[i];
    {   // this command buffer may have bound a pool other than the first: move its IDs to that pool's part of the table (handle: texture ID in bits 0-19, sampler ID from bit 20)
        int tb = CB(c)->tex_base, sb = CB(c)->samp_base;
        for (int i = 0; i < n && (tb || sb); i++) {
            uint64_t v = r->h[i], tid = v & 0xfffff, sid = v >> 20;
            if (op == OP_BIND_SAMP) { if (sid) r->h[i] = (sid + (uint64_t)sb) << 20; }
            else { if (tid) tid += (uint64_t)tb; if (sid && op == OP_BIND_TEX) sid += (uint64_t)sb; r->h[i] = tid | sid << 20; }
        }
    }
    return r;
}
// Image (writable texture): handle = texture pool index (GetImageHandle returns it as-is; RegisterImage writes the same table as RegisterTexture)
NVN(nvnCommandBufferBindImages, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, const NVNimageHandle *h) { st = st_host(st);
    bind_handles(c, OP_BIND_IMG, st, first, n, (const uint64_t *)h);
}
// Separate texture binds: on hardware they fill their own handle table (c2 0x168 + 8 slot), apart from BindTexture's, so they go to slots 128 + n (NVN_SEPTEX_SLOT0)
NVN(nvnCommandBufferBindSeparateTextures, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, const NVNseparateTextureHandle *h) { st = st_host(st);
    bind_handles(c, OP_BIND_TEX, st, NVN_SEPTEX_SLOT0 + first, n, (const uint64_t *)h);
}
NVN(nvnCommandBufferBindSeparateTexture, void, NVNcommandBuffer *c, NVNshaderStage st, int i, NVNseparateTextureHandle h) { st = st_host(st); uint64_t v; memcpy(&v, &h, 8); bind_handles(c, OP_BIND_TEX, st, NVN_SEPTEX_SLOT0 + i, 1, &v); }   // single (Engage): as the plural version
NVN(nvnCommandBufferBindSeparateSamplers, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, const NVNseparateSamplerHandle *h) { st = st_host(st);
    bind_handles(c, OP_BIND_SAMP, st, first, n, (const uint64_t *)h);
}
NVN(nvnCommandBufferBindSeparateSampler, void, NVNcommandBuffer *c, NVNshaderStage st, int i, NVNseparateSamplerHandle h) { st = st_host(st);
    bind_handles(c, OP_BIND_SAMP, st, i, 1, &h.value);
}
NVN(nvnCommandBufferSetStencilValueMask, void, NVNcommandBuffer *c, NVNface f, int m) { *REC(c, OP_STENCIL, p_stencil) = (p_stencil){f, m, 0, 0, 0}; }
NVN(nvnCommandBufferSetStencilMask, void, NVNcommandBuffer *c, NVNface f, int m) { *REC(c, OP_STENCIL, p_stencil) = (p_stencil){f, 0, m, 0, 1}; }
NVN(nvnCommandBufferSetStencilRef, void, NVNcommandBuffer *c, NVNface f, int v) { *REC(c, OP_STENCIL, p_stencil) = (p_stencil){f, 0, 0, v, 2}; }
NVN(nvnCommandBufferSetPolygonOffsetClamp, void, NVNcommandBuffer *c, float fa, float un, float cl) {
    *REC(c, OP_POLY_OFFSET, p_poly_offset) = (p_poly_offset){fa, un, cl};
}
// NVN2 mesh shading: count task workgroups from first, or mesh workgroups in a program with no task shader
NVN_NOPROTO(nvnCommandBufferDrawMeshTasks, void, NVNcommandBuffer *c, int first, int count) { rec_draw(c, (p_draw){DM_PRIM_MESH_TASKS, first, count, 1, 0, 0}); }
NVN(nvnCommandBufferDrawArrays, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, int first, int n) { rec_draw(c, (p_draw){pr, first, n, 1, 0, 0}); }
// Indirect: the {count, instances, first, base instance} record at ind is usually written by the GPU earlier in the frame (Hyrule Warriors: Age of Calamity's first battle), so the replay
// issues a Metal indirect draw from the pool buffer instead of reading it at record or replay time (nvn_mtl.m draw)
NVN(nvnCommandBufferDrawArraysIndirect, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, NVNbufferAddress ind) { rec_draw(c, (p_draw){pr, 0, 0, 1, 0, ind}); }
NVN(nvnCommandBufferDrawElementsInstanced, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, NVNindexType it, int n, NVNbufferAddress ib,
    int bv, int bi, int inst) {
    rec_draw_idx(c, (p_draw_idx){pr, it, n, bv, bi, inst, ib});
}
NVN(nvnCommandBufferDispatchCompute, void, NVNcommandBuffer *c, int x, int y, int z) {
    // The thunk is a tail jump (br), so the return address here is the call site in the game
    *REC(c, OP_DISPATCH, p_dispatch) = (p_dispatch){x, y, z, (uint64_t)(uintptr_t)DM_GUEST_RA()};
}
// Indirect: the three group counts at ind are read by the GPU when the dispatch runs (nvn_mtl.m dispatch)
NVN(nvnCommandBufferDispatchComputeIndirect, void, NVNcommandBuffer *c, NVNbufferAddress ind) {
    *REC(c, OP_DISPATCH, p_dispatch) = (p_dispatch){0, 0, 0, (uint64_t)(uintptr_t)DM_GUEST_RA(), ind};
}
NVN(nvnCommandBufferBarrier, void, NVNcommandBuffer *c, int bits) { *(int32_t *)rec(c, OP_BARRIER, 8) = bits; }
NVN(nvnCommandBufferSetTexturePool, void, NVNcommandBuffer *c, const NVNtexturePool *p) { REC(c, OP_SET_TEX_POOL, p_set_pool)->pool = (const dpool_t *)p; CB(c)->tex_base = dpool_base(g_tpb, g_ntpb, p); }
NVN(nvnCommandBufferSetSamplerPool, void, NVNcommandBuffer *c, const NVNsamplerPool *p) { REC(c, OP_SET_SAMP_POOL, p_set_pool)->pool = (const dpool_t *)p; CB(c)->samp_base = dpool_base(g_spb, g_nspb, p); }

NVN(nvnCommandBufferCopyBufferToTexture, void, NVNcommandBuffer *c, NVNbufferAddress src, const NVNtexture *t, const NVNtextureView *v,
    const NVNcopyRegion *r, int flags) {
    p_copy_b2t *o = REC_SNAP(c, OP_COPY_B2T, p_copy_b2t, 1); o->src = src; o->tex = tex_snap(o, SNAP_OFF(p_copy_b2t), 0, t, (uint64_t)(uintptr_t)DM_GUEST_RA()); copy_view(&o->view, &o->has_view, v); o->r = *r; o->flags = flags;
    if (o->tex) nvn_pool_gpu_written(o->tex->pool);
}
NVN(nvnCommandBufferCopyTextureToBuffer, void, NVNcommandBuffer *c, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r,
    NVNbufferAddress dst, int flags) {
    p_copy_b2t *o = REC_SNAP(c, OP_COPY_T2B, p_copy_b2t, 1); o->src = dst; o->tex = tex_snap(o, SNAP_OFF(p_copy_b2t), 0, t, (uint64_t)(uintptr_t)DM_GUEST_RA()); copy_view(&o->view, &o->has_view, v); o->r = *r; o->flags = flags;
}
// Clear a texture region with the given value (float / unsigned integer versions; mask = channel write mask)
NVN(nvnCommandBufferClearTexture, void, NVNcommandBuffer *c, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, const float *rgba, int mask) {
    p_clear_tex *o = REC_SNAP(c, OP_CLEAR_TEX, p_clear_tex, 1); o->tex = tex_snap(o, SNAP_OFF(p_clear_tex), 0, t, (uint64_t)(uintptr_t)DM_GUEST_RA()); copy_view(&o->view, &o->has_view, v); o->r = *r; o->mask = mask; o->is_uint = 0; memcpy(o->c.f, rgba, 16);
}
NVN(nvnCommandBufferClearTextureui, void, NVNcommandBuffer *c, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, const uint32_t *rgba, int mask) {
    p_clear_tex *o = REC_SNAP(c, OP_CLEAR_TEX, p_clear_tex, 1); o->tex = tex_snap(o, SNAP_OFF(p_clear_tex), 0, t, (uint64_t)(uintptr_t)DM_GUEST_RA()); copy_view(&o->view, &o->has_view, v); o->r = *r; o->mask = mask; o->is_uint = 1; memcpy(o->c.u, rgba, 16);
}
NVN(nvnCommandBufferCopyTextureToTexture, void, NVNcommandBuffer *c, const NVNtexture *s, const NVNtextureView *sv, const NVNcopyRegion *sr,
    const NVNtexture *d, const NVNtextureView *dv, const NVNcopyRegion *dr, int flags) {
    p_copy_t2t *o = REC_SNAP(c, OP_COPY_T2T, p_copy_t2t, 2); o->src = tex_snap(o, SNAP_OFF(p_copy_t2t), 0, s, (uint64_t)(uintptr_t)DM_GUEST_RA()); o->dst = tex_snap(o, SNAP_OFF(p_copy_t2t), 1, d, (uint64_t)(uintptr_t)DM_GUEST_RA());
    copy_view(&o->sv, &o->has_sv, sv); copy_view(&o->dv, &o->has_dv, dv); o->sr = *sr; o->dr = *dr; o->flags = flags;
    if (o->dst) nvn_pool_gpu_written(o->dst->pool);
}
// A textured rectangle into the bound colour targets, replayed as draw_tex (nvn_mtl.m). The handle's IDs are moved to the pools this command buffer bound, as bind_handles does:
// Streets of Rage 4 draws its movie frames into its scene target this way, from a second texture pool. Only the handle's lower 32 bits reach the GPU (bind_handles)
NVN(nvnCommandBufferDrawTexture, void, NVNcommandBuffer *c, NVNtextureHandle h, const NVNdrawTextureRegion *d, const NVNdrawTextureRegion *s) {
    uint64_t tid = (uint32_t)h & 0xfffff, sid = (uint32_t)h >> 20;
    if (tid) tid += (uint64_t)CB(c)->tex_base;
    if (sid) sid += (uint64_t)CB(c)->samp_base;
    *REC(c, OP_DRAW_TEX, p_draw_tex) = (p_draw_tex){tid | sid << 20, *d, *s};
}
NVN(nvnCommandBufferSetCopyRowStride, void, NVNcommandBuffer *c, ptrdiff_t st) { p_copy_stride *o = REC(c, OP_COPY_STRIDE, p_copy_stride); o->stride = st; o->image = 0; }
NVN(nvnCommandBufferSetCopyImageStride, void, NVNcommandBuffer *c, ptrdiff_t st) { p_copy_stride *o = REC(c, OP_COPY_STRIDE, p_copy_stride); o->stride = st; o->image = 1; }   // Link's Awakening
NVN(nvnCommandBufferSetBlendColor, void, NVNcommandBuffer *c, const float *v) { memcpy(REC(c, OP_BLEND_COLOR, p_blend_color)->rgba, v, 16); }

// ---------- State objects: packed in place (nvn_obj.h) ----------
#define ST(T, x) ((T *)(x))
NVN(nvnBlendStateSetDefaults, void, NVNblendState *s) { *ST(blend_st, s) = (blend_st){0, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD}; }
NVN(nvnBlendStateSetBlendTarget, void, NVNblendState *s, int t) { ST(blend_st, s)->target = t; }
NVN(nvnBlendStateSetBlendFunc, void, NVNblendState *s, NVNblendFunc sr, NVNblendFunc dr, NVNblendFunc sa, NVNblendFunc da) {
    blend_st *o = ST(blend_st, s); o->src_rgb = sr; o->dst_rgb = dr; o->src_a = sa; o->dst_a = da;
}
NVN(nvnBlendStateSetBlendEquation, void, NVNblendState *s, NVNblendEquation er, NVNblendEquation ea) { ST(blend_st, s)->eq_rgb = er; ST(blend_st, s)->eq_a = ea; }
// Advanced blend modes (NV_blend_equation_advanced): kept in the state, and while one is set the hardware ignores the blend functions. The replay blends the Porter-Duff modes with
// the fixed-function factors they amount to (adv_blend in nvn_mtl.m); multiply, screen, overlay and the rest have no Metal blend state. Mode 0 (none, which Luigi's Mansion 3 sets
// on every blend state) is the default; Mario vs. Donkey Kong 1.0.1 sets SRC_OVER (8) on one state in its levels
NVN(nvnBlendStateSetAdvancedMode, void, NVNblendState *s, NVNblendAdvancedMode mode) { ST(blend_st, s)->adv = (unsigned)mode <= NVN_BLEND_ADVANCED_MODE_HSL_LUMINOSITY ? mode : 0; }
NVN(nvnBlendStateSetAdvancedPremultipliedSrc, void, NVNblendState *s, uint32_t pm) { ST(blend_st, s)->adv_npm = !B(pm); }
NVN(nvnBlendStateSetAdvancedOverlap, void, NVNblendState *s, NVNblendAdvancedOverlap o) { ST(blend_st, s)->adv_ovl = (unsigned)o & 3; }
NVN(nvnColorStateSetDefaults, void, NVNcolorState *s) { *ST(color_st, s) = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}; }
NVN(nvnColorStateSetBlendEnable, void, NVNcolorState *s, int i, uint32_t en) {
    color_st *o = ST(color_st, s); if (i < 0 || i > 7) return;
    o->blend_en = B(en) ? (o->blend_en | 1u << i) : (o->blend_en & ~(1u << i));
}
NVN(nvnChannelMaskStateSetDefaults, void, NVNchannelMaskState *s) { ST(chmask_st, s)->mask = 0xffffffffu; }
NVN(nvnChannelMaskStateSetChannelMask, void, NVNchannelMaskState *s, int i, uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
    if (i < 0 || i > 7) return;
    uint32_t m = (B(r) ? 1 : 0) | (B(g) ? 2 : 0) | (B(b) ? 4 : 0) | (B(a) ? 8 : 0);
    chmask_st *o = ST(chmask_st, s); o->mask = (o->mask & ~(0xfu << (4 * i))) | m << (4 * i);
}
NVN(nvnMultisampleStateSetDefaults, void, NVNmultisampleState *s) { memset(s, 0, sizeof(ms_st)); ST(ms_st, s)->enable = 1; }
NVN(nvnMultisampleStateSetMultisampleEnable, void, NVNmultisampleState *s, uint32_t e) { ST(ms_st, s)->enable = B(e) != 0; }
NVN(nvnMultisampleStateSetSamples, void, NVNmultisampleState *s, int n) { ST(ms_st, s)->samples = (uint8_t)n; }
NVN(nvnMultisampleStateSetAlphaToCoverageEnable, void, NVNmultisampleState *s, uint32_t e) { ST(ms_st, s)->a2c = B(e) != 0; }
NVN(nvnMultisampleStateSetSampleLocationsEnable, void, NVNmultisampleState *s, uint32_t e) { ST(ms_st, s)->sl_en = B(e) != 0; }
NVN(nvnMultisampleStateSetSampleLocations, void, NVNmultisampleState *s, int start, int n, const float *v) { (void)s; (void)start; (void)n; (void)v; }
NVN(nvnMultisampleStateGetSamples, int, const NVNmultisampleState *s) { return ((const ms_st *)s)->samples; }
NVN(nvnMultisampleStateGetMultisampleEnable, NVNboolean, const NVNmultisampleState *s) { return ((const ms_st *)s)->enable; }
NVN(nvnPolygonStateSetDefaults, void, NVNpolygonState *s) { *ST(poly_st, s) = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0}; }
NVN(nvnPolygonStateSetCullFace, void, NVNpolygonState *s, NVNface f) { ST(poly_st, s)->cull = f; }
NVN(nvnPolygonStateSetFrontFace, void, NVNpolygonState *s, NVNfrontFace f) { ST(poly_st, s)->front = f; }
NVN(nvnPolygonStateSetPolygonMode, void, NVNpolygonState *s, NVNpolygonMode m) { ST(poly_st, s)->mode = m; }
NVN(nvnPolygonStateSetPolygonOffsetEnables, void, NVNpolygonState *s, int e) { ST(poly_st, s)->offset_en = e; }
NVN(nvnDepthStencilStateSetDefaults, void, NVNdepthStencilState *s) {
    *ST(ds_st, s) = (ds_st){0, 1, NVN_DEPTH_FUNC_LESS, 0, NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS,
                            NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP};
}
NVN(nvnDepthStencilStateSetDepthTestEnable, void, NVNdepthStencilState *s, uint32_t e) { ST(ds_st, s)->dt = B(e) != 0; }
NVN(nvnDepthStencilStateSetDepthWriteEnable, void, NVNdepthStencilState *s, uint32_t e) { ST(ds_st, s)->dw = B(e) != 0; }
NVN(nvnDepthStencilStateSetDepthFunc, void, NVNdepthStencilState *s, NVNdepthFunc f) { ST(ds_st, s)->df = f; }
NVN(nvnDepthStencilStateSetStencilTestEnable, void, NVNdepthStencilState *s, uint32_t e) { ST(ds_st, s)->st = B(e) != 0; }
NVN(nvnDepthStencilStateSetStencilFunc, void, NVNdepthStencilState *s, NVNface f, NVNstencilFunc fn) {
    ds_st *o = ST(ds_st, s); if (f & NVN_FACE_FRONT) o->sf_f = fn; if (f & NVN_FACE_BACK) o->sf_b = fn;
}
NVN(nvnDepthStencilStateSetStencilOp, void, NVNdepthStencilState *s, NVNface f, NVNstencilOp fail, NVNstencilOp zfail, NVNstencilOp zpass) {
    ds_st *o = ST(ds_st, s);
    if (f & NVN_FACE_FRONT) { o->fail_f = fail; o->zfail_f = zfail; o->zpass_f = zpass; }
    if (f & NVN_FACE_BACK) { o->fail_b = fail; o->zfail_b = zfail; o->zpass_b = zpass; }
}
NVN(nvnVertexAttribStateSetDefaults, void, NVNvertexAttribState *s) { *ST(va_st, s) = (va_st){0, 0, 0}; }
NVN(nvnVertexAttribStateSetFormat, void, NVNvertexAttribState *s, NVNformat f, ptrdiff_t off) {
    if (off < 0 || off >= (1 << 19)) LOG("[RED] vertex attribute offset %td exceeds the packed range", off);
    ST(va_st, s)->fmt = f; ST(va_st, s)->offset = (uint32_t)off;
}
NVN(nvnVertexAttribStateSetStreamIndex, void, NVNvertexAttribState *s, int i) { ST(va_st, s)->stream = i; }
NVN(nvnVertexStreamStateSetDefaults, void, NVNvertexStreamState *s) { *ST(vs_st, s) = (vs_st){0, 0}; }
NVN(nvnVertexStreamStateSetStride, void, NVNvertexStreamState *s, ptrdiff_t st) { ST(vs_st, s)->stride = (uint32_t)st; }
NVN(nvnVertexStreamStateSetDivisor, void, NVNvertexStreamState *s, int d) { ST(vs_st, s)->divisor = d; }
// Getters: Monster Hunter Rise reads its stream states back when it builds a draw (about 24,000 calls in its character creation)
NVN(nvnVertexStreamStateGetStride, ptrdiff_t, const NVNvertexStreamState *s) { return (ptrdiff_t)((const vs_st *)s)->stride; }
NVN(nvnVertexStreamStateGetDivisor, int, const NVNvertexStreamState *s) { return ((const vs_st *)s)->divisor; }
NVN(nvnVertexAttribStateGetFormat, void, const NVNvertexAttribState *s, NVNformat *f, ptrdiff_t *off) { const va_st *o = (const va_st *)s; if (f) *f = (NVNformat)o->fmt; if (off) *off = (ptrdiff_t)o->offset; }
NVN(nvnVertexAttribStateGetStreamIndex, int, const NVNvertexAttribState *s) { return ((const va_st *)s)->stream; }

static _Atomic uint64_t g_nvn_ns;   // host time spent in submit and present (since the last frame statistics)
// The submitting thread's CPU time inside submit (QueryThreadCycleTime, TSC cycles): submit's wall time minus this is time it waited (locks, slots, GPU, the completion thread)
static _Atomic uint64_t g_nvn_cyc;
static inline uint64_t thread_cyc(void) { ULONG64 c = 0; QueryThreadCycleTime(GetCurrentThread(), &c); return c; }
static double tsc_per_ns(void) {   // the TSC rate, from two points far apart (the first call and now)
    static uint64_t c0, n0; uint64_t c = __builtin_ia32_rdtsc(), n = now_ns();
    if (!n0) { c0 = c; n0 = n; return 0; }
    return n > n0 + 100000000ull ? (double)(c - c0) / (double)(n - n0) : 0;
}

// ---------- Lifecycle: dm_gfx_suspend (nxnative: before iOS goes to the background; for a minimised window later) ----------
// on=1: subsequent submits / presents block until on=0; before returning, wait for in-progress submit / present calls to exit and for all in-flight GPU work to complete
static pthread_mutex_t g_susp_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_susp_cv = PTHREAD_COND_INITIALIZER;
static int g_susp, g_inflight_api;
static void gfx_enter(void) { pthread_mutex_lock(&g_susp_mu); while (g_susp) pthread_cond_wait(&g_susp_cv, &g_susp_mu); g_inflight_api++; pthread_mutex_unlock(&g_susp_mu); dm_host_work++; }
static void gfx_leave(void) { dm_host_work--; pthread_mutex_lock(&g_susp_mu); g_inflight_api--; pthread_cond_broadcast(&g_susp_cv); pthread_mutex_unlock(&g_susp_mu); }
void dm_gfx_enter(void) { gfx_enter(); }
void dm_gfx_leave(void) { gfx_leave(); }
void dm_gfx_suspend(int on) {
    if (on) aq_drain();
    pthread_mutex_lock(&g_susp_mu);
    g_susp = on != 0;
    if (on) while (g_inflight_api) pthread_cond_wait(&g_susp_cv, &g_susp_mu);
    else pthread_cond_broadcast(&g_susp_cv);
    pthread_mutex_unlock(&g_susp_mu);
    if (on) { gfx_drain(); LOG("[nvn] graphics suspended: in-flight GPU work drained"); } else LOG("[nvn] graphics resumed");
}
// One submission at a time: a backend keeps one submission in progress (nxnative's Metal replay: one command buffer and one upload batch). Fire Emblem: Three Houses initializes two queues and
// submits on both from two threads; NVN orders work on different queues only through syncs, so replaying the two queues' submissions in lock order is one valid order.
// The first submission's prepare step (gfx_prepare: the prewarm and the pipeline prebuild) submits no GPU work, so it runs under the lock but outside the graphics gate. The lock comes
// before the gate: a second submission waiting for it must not count as in flight
static void submit_now(uint64_t serial, int origin_ll, int n, const uint64_t *h) {   // the translation of one submission (the calling thread, or the translation thread)
    dm_host_mutex_lock(&g_sub_mu);
    dm_host_work++; gfx_prepare(); dm_host_work--;
    gfx_enter();
    gfx_queue_activate(serial);
    gfx_set_queue_origin_lower_left(origin_ll);
    gfx_sub_begin();
    nvn_ft_poll();
    gfx_sub_index(h, n);
    for (int i = 0; i < n; i++) { gfx_exec_rest(h + i + 1, n - i - 1); gfx_exec((const uint8_t *)h[i]); }
    gfx_exec_rest(NULL, 0);
    gfx_sub_commit();
    pthread_mutex_unlock(&g_sub_mu);
    gfx_leave();
}

// ---------- Asynchronous submission (SWR_ASYNC_SUBMIT=1) ----------
// On the console nvnQueueSubmitCommands hands the commands to the GPU and returns at once; the GPU reads them, and the memory they point at, later. Here the translation
// into Vulkan takes most of a frame (~27 ms of a 34 ms MKW race frame) on the game's own render thread, which then cannot start the next frame. With this on,
// submits and presents go, in order, to one translation thread and return. Everything the game can observe keeps its order:
// - a fence taken meanwhile (QueueFenceSync, an acquire's sync) holds the sequence number of the last queued work (AQ_SEQ) and resolves to the Vulkan timeline value once
//   that is translated; waiting on it waits for the translation, then for the GPU, so a game that waits for a fence before reusing memory sees the console's order;
// - QueueFinish, ReadTexels, a queue's finalize and graphics suspend wait until the thread has done everything queued;
// - a sync object a queued command buffer fences (OP_FENCE_SYNC) is not fenced yet: a wait on it with a timeout waits for the queue first.
// What the replay reads from guest memory it reads later than before, as the console's GPU does. At most AQ_PRESENTS presents wait in the queue (the frames in flight)
enum { AQ_N = 64, AQ_PRESENTS = 2, AQ_FN = 1024 };
#define AQ_SEQ (1ull << 62)
typedef struct { int kind; uint64_t seq, serial; int origin_ll, n; uint64_t *h; win_t *w; int idx; const tex_t *t; int crop[4]; double period; } aop_t;   // kind 1 submit, 2 present
static aop_t g_aq[AQ_N]; static uint64_t g_aq_tail, g_aq_npres;   // g_aq_tail: sequence number of the last queued op (op seq goes to slot seq % AQ_N)
static _Atomic uint64_t g_aq_done, g_aq_head;   // the last op finished; the last op taken
static struct { uint64_t seq, fence; } g_aq_f[AQ_FN];   // the timeline value after op seq (its fence)
static pthread_mutex_t g_aq_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_aq_cv = PTHREAD_COND_INITIALIZER, g_aq_done_cv = PTHREAD_COND_INITIALIZER;
static _Atomic uint64_t g_aq_ns, g_aq_cyc;   // the translation thread's time in its ops (frame statistics)
static int g_aq_on = -1;
static int aq_on(void) { if (g_aq_on < 0) { const char *e = getenv("SWR_ASYNC_SUBMIT"); g_aq_on = e && atoi(e) > 0; if (g_aq_on) LOG("[nvn] asynchronous submission: submits and presents are translated on a thread of their own (SWR_ASYNC_SUBMIT)"); } return g_aq_on; }
static int aq_busy(void) { return aq_on() && atomic_load(&g_aq_done) < g_aq_tail; }
static unsigned __stdcall aq_main(void *arg) {
    (void)arg; SetThreadDescription(GetCurrentThread(), L"nvn translation"); SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    for (;;) {
        pthread_mutex_lock(&g_aq_mu);
        while (atomic_load(&g_aq_head) == g_aq_tail) pthread_cond_wait(&g_aq_cv, &g_aq_mu);
        uint64_t seq = atomic_load(&g_aq_head) + 1; aop_t op = g_aq[seq % AQ_N]; atomic_store(&g_aq_head, seq);
        pthread_mutex_unlock(&g_aq_mu);
        uint64_t t0 = now_ns(), c0 = thread_cyc();
        if (op.kind == 1) { submit_now(op.serial, op.origin_ll, op.n, op.h); free(op.h); }
        else if (op.kind == 2) { gfx_enter(); uint64_t tk = gfx_present(op.t, op.period, op.crop); gfx_leave(); if (op.idx >= 0 && op.idx < 4) op.w->ticket[op.idx] = tk; }
        atomic_fetch_add(&g_aq_ns, now_ns() - t0); atomic_fetch_add(&g_aq_cyc, thread_cyc() - c0);
        pthread_mutex_lock(&g_aq_mu);
        g_aq_f[seq % AQ_FN].fence = gfx_last_fence(); g_aq_f[seq % AQ_FN].seq = seq;
        if (op.kind == 2) g_aq_npres--;
        atomic_store(&g_aq_done, seq);
        pthread_cond_broadcast(&g_aq_done_cv);
        pthread_mutex_unlock(&g_aq_mu);
    }
    return 0;
}
static uint64_t aq_push(const aop_t *op) {   // queues op (waiting while the queue, or its presents, are full); returns its sequence number
    static int started; pthread_mutex_lock(&g_aq_mu);
    if (!started) { started = 1; uintptr_t th = _beginthreadex(NULL, 0, aq_main, NULL, 0, NULL); if (!th) die("cannot start the NVN translation thread"); CloseHandle((HANDLE)th); }
    dm_host_work++;
    while (g_aq_tail - atomic_load(&g_aq_head) >= AQ_N - 1 || (op->kind == 2 && g_aq_npres >= AQ_PRESENTS)) pthread_cond_wait(&g_aq_done_cv, &g_aq_mu);
    dm_host_work--;
    uint64_t seq = ++g_aq_tail; g_aq[seq % AQ_N] = *op; g_aq[seq % AQ_N].seq = seq;
    if (op->kind == 2) g_aq_npres++;
    pthread_cond_signal(&g_aq_cv);
    pthread_mutex_unlock(&g_aq_mu);
    return seq;
}
static int aq_wait_seq(uint64_t seq, uint64_t timeout) {   // 1 once op seq is done; timeout in ns (0 = only look)
    if (atomic_load(&g_aq_done) >= seq) return 1;
    if (!timeout) return 0;
    uint64_t t0 = now_ns(); int ok = 1;
    dm_host_work++; pthread_mutex_lock(&g_aq_mu);
    while (atomic_load(&g_aq_done) < seq) {
        if (timeout != UINT64_MAX && now_ns() - t0 >= timeout) { ok = 0; break; }
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_nsec += 2000000; if (ts.tv_nsec >= 1000000000) { ts.tv_sec++; ts.tv_nsec -= 1000000000; }
        pthread_cond_timedwait(&g_aq_done_cv, &g_aq_mu, &ts);
    }
    pthread_mutex_unlock(&g_aq_mu); dm_host_work--;
    return ok;
}
static void aq_drain(void) { if (aq_on()) aq_wait_seq(g_aq_tail, UINT64_MAX); }
static uint64_t aq_fence_now(void) {   // the fence of everything queued so far
    if (!aq_on()) return gfx_last_fence();
    pthread_mutex_lock(&g_aq_mu); uint64_t t = g_aq_tail, d = atomic_load(&g_aq_done); pthread_mutex_unlock(&g_aq_mu);
    return t > d ? AQ_SEQ | t : gfx_last_fence();
}
static int aq_resolve(uint64_t *fence, uint64_t timeout) {   // a queued fence (AQ_SEQ) becomes its timeline value once translated; 0 = not translated within timeout
    if (!(*fence & AQ_SEQ)) return 1;
    uint64_t seq = *fence & ~AQ_SEQ;
    if (!aq_wait_seq(seq, timeout)) return 0;
    pthread_mutex_lock(&g_aq_mu);
    *fence = g_aq_f[seq % AQ_FN].seq == seq ? g_aq_f[seq % AQ_FN].fence : gfx_last_fence();   // long done (its entry reused): any later value is complete no sooner, so it is safe
    pthread_mutex_unlock(&g_aq_mu);
    return 1;
}
NVN(nvnQueueSubmitCommands, void, NVNqueue *q, int n, const NVNcommandHandle *h) {
    TRS0();
    uint64_t t0 = now_ns(), c0 = thread_cyc();
    ((queue_t *)q)->submits++;
    nvn_cap_submit(n, (const uint64_t *)h);
    if (aq_on()) {
        aop_t op = {1, 0, ((const queue_t *)q)->serial, ((const queue_t *)q)->origin_ll, n, malloc((size_t)(n ? n : 1) * sizeof(uint64_t))};
        if (!op.h) die("asynchronous submission: out of memory");
        memcpy(op.h, h, (size_t)n * sizeof(uint64_t));
        aq_push(&op);
    } else submit_now(((const queue_t *)q)->serial, ((const queue_t *)q)->origin_ll, n, (const uint64_t *)h);
    atomic_fetch_add(&g_nvn_ns, now_ns() - t0); atomic_fetch_add(&g_nvn_cyc, thread_cyc() - c0);
    TRS1("nvnQueueSubmitCommands");
}

// Commands that only affect GPU state the host does not model, or only execute on the GPU (nxnative's nvn_nops list): empty
NVN(nvnCommandBufferSetPatchSize, void, NVNcommandBuffer *c, int n) { (void)c; (void)n; }
NVN(nvnCommandBufferSetDepthBounds, void, NVNcommandBuffer *c, uint32_t en, float lo, float hi) { (void)c; (void)en; (void)lo; (void)hi; }
NVN(nvnCommandBufferWaitSync, void, NVNcommandBuffer *c, const NVNsync *s) { (void)c; (void)s; }   // the host replays submissions in order: earlier work is done first
NVN(nvnCommandBufferSetShaderScratchMemory, void, NVNcommandBuffer *c, const NVNmemoryPool *p, ptrdiff_t off, size_t n) { (void)c; (void)p; (void)off; (void)n; }
NVN(nvnCommandBufferPushDebugGroupDynamic, void, NVNcommandBuffer *c, uint32_t id, const char *s) { (void)c; (void)id; (void)s; }
NVN(nvnCommandBufferPopDebugGroup, void, NVNcommandBuffer *c) { (void)c; }
NVN(nvnCommandBufferInsertDebugMarkerDynamic, void, NVNcommandBuffer *c, uint32_t id, const char *s) { (void)c; (void)id; (void)s; }
NVN(nvnCommandBufferInsertDebugMarker, void, NVNcommandBuffer *c, const char *s) { (void)c; (void)s; }   // 13 Sentinels
NVN(nvnCommandBufferPushDebugGroup, void, NVNcommandBuffer *c, uint32_t id, const char *s) { (void)c; (void)id; (void)s; }   // Super Mario 3D World + Bowser's Fury (NVN 53)
NVN(nvnCommandBufferPushDebugGroupStatic, void, NVNcommandBuffer *c, uint32_t id, const char *s) { (void)c; (void)id; (void)s; }
NVN(nvnCommandBufferPopDebugGroupId, void, NVNcommandBuffer *c, uint32_t id) { (void)c; (void)id; }
NVN(nvnCommandBufferInsertDebugMarkerStatic, void, NVNcommandBuffer *c, uint32_t id, const char *s) { (void)c; (void)id; (void)s; }
// The GPU waits here until the event's value meets the condition, as on the console: the replay keeps the order (vk_replay.c wait_event). It was ignored before 2026-11-07
NVN(nvnCommandBufferWaitEvent, void, NVNcommandBuffer *c, const NVNevent *e, NVNeventWaitMode m, uint32_t v) {
    { static long n; if (!(n++ & 1023)) LOG("[nvn] CommandBufferWaitEvent call %ld", n); }
    if (!e) return;
    uint8_t *p = cmd_alloc(CB(c), 24); put_hdr(p, OP_WAIT_EVENT, 24);
    const ev_t *ev = (const ev_t *)e; volatile uint32_t *val = ev->val; uint32_t mm = m;
    memcpy(p + 8, &val, 8); memcpy(p + 16, &mm, 4); memcpy(p + 20, &v, 4);
}
NVN(nvnWindowFinalize, void, NVNwindow *w) { (void)w; }

// ---------- Window / present ----------
#define WB(b) ((win_builder_t *)(b))
NVN(nvnWindowBuilderSetDefaults, void, NVNwindowBuilder *b) { NVNdevice *dev = WB(b)->dev; memset(b, 0, sizeof(win_builder_t)); WB(b)->dev = dev; WB(b)->interval = 1; }
NVN(nvnWindowBuilderSetDevice, void, NVNwindowBuilder *b, NVNdevice *d) { WB(b)->dev = d; }
NVN(nvnWindowBuilderSetNativeWindow, void, NVNwindowBuilder *b, NVNnativeWindow w) { WB(b)->native = w; }
NVN(nvnWindowBuilderSetTextures, void, NVNwindowBuilder *b, int n, NVNtexture *const *t) {
    if (n < 0 || n > 4) { LOG("[RED] WindowBuilderSetTextures count %d", n); n = n < 0 ? 0 : 4; }
    WB(b)->ntex = n; for (int i = 0; i < n; i++) WB(b)->tex[i] = t[i];
}
NVN(nvnWindowBuilderSetPresentInterval, void, NVNwindowBuilder *b, int v) { WB(b)->interval = v; }
NVN(nvnWindowInitialize, NVNboolean, NVNwindow *w, const NVNwindowBuilder *b) {
    const win_builder_t *s = (const win_builder_t *)b; win_t *o = (win_t *)w;
    memset(o, 0, sizeof *o); o->magic = MAGIC('W', 'I', 'N', 'D'); o->native = s->native; o->ntex = s->ntex; o->interval = s->interval;
    memcpy(o->tex, s->tex, sizeof o->tex);
    for (int i = 0; i < o->ntex; i++) if (o->tex[i]) gfx_xv_window_tex(CTX(o->tex[i])->serial);
    LOG("[nvn] WindowInitialize swap chain %d textures, interval %d, first %dx%d format %#x", o->ntex, o->interval,
        o->ntex && o->tex[0] ? (int)CTX(o->tex[0])->d.width : 0, o->ntex && o->tex[0] ? (int)CTX(o->tex[0])->d.height : 0, o->ntex && o->tex[0] ? CTX(o->tex[0])->d.format : 0);
    return 1;
}
NVN(nvnWindowGetPresentInterval, int, const NVNwindow *w) { return ((const win_t *)w)->interval; }   // Link's Awakening reads it back every frame
volatile int g_nvn_present_interval = 1;   // the game's current present interval (vblanks a frame): the present pacing reads it (svc_ns2.c swr_present_pace)
NVN(nvnWindowSetPresentInterval, void, NVNwindow *w, int v) {
    static long n; if (((win_t *)w)->interval != v && n++ < 64) LOG("[nvn] the game sets present interval %d → %d", ((win_t *)w)->interval, v);   // it flips every few frames in its dynamic mode
    ((win_t *)w)->interval = v; g_nvn_present_interval = v;
}
NVN(nvnWindowSetCrop, void, NVNwindow *w, int x, int y, int ww, int hh) {
    win_t *o = (win_t *)w; int v[4] = {x, y, ww, hh};
    if (memcmp(o->crop, v, sizeof v)) LOG("[nvn] WindowCrop %d,%d %dx%d", x, y, ww, hh);
    memcpy(o->crop, v, sizeof v);
}
NVN(nvnWindowGetCrop, void, const NVNwindow *w, NVNrectangle *r) {   // Bayonetta 3 reads it back every few frames; before any SetCrop it is the whole first texture, as NVN starts it
    const win_t *o = (const win_t *)w;
    if (o->crop[2] > 0 && o->crop[3] > 0) { r->x = o->crop[0]; r->y = o->crop[1]; r->width = o->crop[2]; r->height = o->crop[3]; return; }
    r->x = r->y = 0; r->width = o->ntex && o->tex[0] ? (int)CTX(o->tex[0])->d.width : 0; r->height = o->ntex && o->tex[0] ? (int)CTX(o->tex[0])->d.height : 0;
}
// ---------- Acquire and present (from M4's null device, nxrt; nxnative's Mac pacing is not ported) ----------
static _Atomic uint64_t g_acquires, g_presents;
uint64_t nxrt_nvn_presents(void) { return atomic_load(&g_presents); }
uint64_t dm_nvn_frame_count(void) { return atomic_load(&g_presents); }   // frames presented so far (hle_vi.c, the command memory statistics, host segments)
static void frame_request(const char *what, int idx) {
    char at[160]; NxwGuestCpu *c = nxrt_cur_cpu(); nxrt_symbolize(c ? c->x[30] : 0, at, sizeof at);
    LOG("[nvn] first frame request: %s (texture %d) at %.3f s, called from %s", what, idx, (now_ns() - g_t0_ns) / 1e9, at);
}
// The display: a screen at 60 Hz on a fixed vsync grid. A present returns at once (nxnative's present only enqueues for its display link) and is shown at the first
// vsync at or after it, at least the window's interval after the frame before; that frame's texture stays on screen until then. Acquiring a texture waits while it
// is still shown or queued, so a game that runs ahead of the display is held back by its swap chain, as on hardware, and a game that paces itself (Pokopia does,
// on its own timer) is never blocked. The grid does not drift. A backend with a display of its own (gfx_link_active) paces the acquires instead
static SRWLOCK g_disp_lk = SRWLOCK_INIT;
static uint64_t g_vs_t0, g_vs_k, g_free_at[4], g_vs_skipped; static _Atomic uint64_t g_acq_wait_ns; static int g_shown = -1;
static uint64_t vsync_at(uint64_t k) { return g_vs_t0 + k * 1000000000ull / 60; }
// Schedules the present of texture idx made at `now`: returns the time it is shown; *skipped = vsyncs before it that showed nothing new (tests drive it with their own clock)
uint64_t nxrt_nvn_display(uint64_t now, int interval, int idx, unsigned *skipped) {
    AcquireSRWLockExclusive(&g_disp_lk);
    uint64_t k = 0; *skipped = 0;
    if (!g_vs_t0) g_vs_t0 = now;
    else {
        uint64_t earliest = g_vs_k + (uint64_t)(interval > 0 ? interval : 1);
        k = (now - g_vs_t0) * 60 / 1000000000ull;
        if (vsync_at(k) < now) k++;
        if (k > earliest) *skipped = (unsigned)(k - earliest);
        else k = earliest;
    }
    g_vs_k = k;
    uint64_t at = vsync_at(k);
    if (g_shown >= 0 && g_shown != idx && g_shown < 4) g_free_at[g_shown] = at;   // replaced on screen at `at`
    if (idx >= 0 && idx < 4) g_free_at[idx] = UINT64_MAX;
    g_shown = idx;
    ReleaseSRWLockExclusive(&g_disp_lk);
    return at;
}
// Until when texture idx is busy (shown or queued); 0 when free
uint64_t nxrt_nvn_texture_busy_until(int idx) {
    AcquireSRWLockShared(&g_disp_lk); uint64_t t = idx >= 0 && idx < 4 ? g_free_at[idx] : 0; ReleaseSRWLockShared(&g_disp_lk); return t;
}
extern int vdec_playing(void);
static int present_itv(const win_t *o) { return o->interval > 0 && !vdec_playing() ? o->interval : 1; }   // vsyncs per frame (the game's present interval)
static uint64_t present_period(const win_t *o) { return (uint64_t)present_itv(o) * 1000000000ull / 60; }
NVN(nvnWindowAcquireTexture, NVNwindowAcquireTextureResult, NVNwindow *w, NVNsync *s, int *idx) {
    win_t *o = (win_t *)w; TRS0();
    *idx = o->ntex ? o->next++ % o->ntex : 0;
    uint64_t t0 = now_ns();
    // Vsync blocks here (real-device NVN: when every texture of the swap chain waits for the display, wait for the next one to be released), not in present: the game often presents holding a lock
    if (gfx_link_active()) gfx_link_wait_tick(present_itv(o), 2 * present_period(o));
    else {
        uint64_t until = nxrt_nvn_texture_busy_until(*idx);
        if (until == UINT64_MAX) until = t0 + 1000000000ull / 60;   // the one texture on screen with no frame after it (a swap chain of one): a vsync
        if (until > t0) { uint64_t d = until - t0; plat_sleep_ns(d < 100000000ull ? d : 100000000ull); }
    }
    // The previous present of this texture has not been handed to the GPU yet (the backend's present queue is a whole swap chain behind): wait for it
    if (aq_on() && *idx < 4 && (o->ticket[*idx] & AQ_SEQ)) aq_wait_seq(o->ticket[*idx] & ~AQ_SEQ, UINT64_MAX);   // its present is still queued: the ticket comes once it is translated
    uint64_t tk = *idx < 4 ? o->ticket[*idx] : 0;
    if (tk) gfx_present_wait(tk);
    uint64_t waited = now_ns() - t0; if (waited > 50000) atomic_fetch_add(&g_acq_wait_ns, waited);
    if (s) { ((sync_t *)s)->fence = aq_fence_now(); ((sync_t *)s)->fenced = 2; }   // the texture is free once the submissions before this point have completed (fenced 2: an acquire's sync)
    if (!atomic_fetch_add(&g_acquires, 1)) frame_request("WindowAcquireTexture", *idx);
    TRS1("nvnWindowAcquireTexture");
    return NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS;
}
// The older API (NVN 53, Super Mario 3D World + Bowser's Fury): the queue waits for the texture on the GPU and no sync object is handed out. The host replays in submission order, so later commands already come after earlier presents; the same as WindowAcquireTexture without a sync object
NVN(nvnQueueAcquireTexture, NVNqueueAcquireTextureResult, NVNqueue *q, NVNwindow *w, int *idx) {
    (void)q; return h_nvnWindowAcquireTexture(w, NULL, idx) == NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS ? (NVNqueueAcquireTextureResult)0 : (NVNqueueAcquireTextureResult)1;
}
// ---------- Frame statistics: frame time and work every 300 frames, and the NVN calls of a frame (nxnative's "[nvn] frame N" lines, counted by the registry) ----------
enum { STAT_N = 300 };
static double g_st_int[STAT_N]; static int g_st_n; static uint64_t g_st_t0, g_st_skip0, g_st_wait0, g_st_nvn0, g_st_cyc0, g_st_sync0;
static int dcmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static void frame_stats(uint64_t n, double interval_ms, uint64_t now, double queued_ms) {
    if (!g_st_n) { g_st_t0 = now; g_st_skip0 = g_vs_skipped; g_st_wait0 = atomic_load(&g_acq_wait_ns); g_st_nvn0 = atomic_load(&g_nvn_ns); g_st_cyc0 = atomic_load(&g_nvn_cyc); g_st_sync0 = atomic_load(&g_sync_ns); tsc_per_ns(); }
    g_st_int[g_st_n] = interval_ms;
    if (++g_st_n < STAT_N) return;
    double secs = (now - g_st_t0) / 1e9, *v = g_st_int;
    qsort(v, STAT_N, sizeof *v, dcmp);
    LOG("[nvn] frame pacing up to frame %llu: %.2f presents/s; present interval p50 %.2f / p95 %.2f / p99 %.2f / max %.2f ms; vsyncs that showed no new frame %llu; "
        "shown %.1f ms after the present; acquire waited %.1f ms in all; submit+present %.2f ms a frame", (unsigned long long)n, (STAT_N - 1) / (secs > 0 ? secs : 1), v[STAT_N / 2], v[STAT_N * 95 / 100],
        v[STAT_N * 99 / 100], v[STAT_N - 1], (unsigned long long)(g_vs_skipped - g_st_skip0), queued_ms, (atomic_load(&g_acq_wait_ns) - g_st_wait0) / 1e6,
        (atomic_load(&g_nvn_ns) - g_st_nvn0) / 1e6 / STAT_N);
    { double r = tsc_per_ns(); if (r > 0) LOG("[nvn] submit CPU time up to frame %llu: %.2f ms a frame on the submitting thread (the rest of submit's time waited); game threads blocked in SyncWait %.2f ms a frame",
                                              (unsigned long long)n, (atomic_load(&g_nvn_cyc) - g_st_cyc0) / r / 1e6 / STAT_N, (atomic_load(&g_sync_ns) - g_st_sync0) / 1e6 / STAT_N); }
    if (aq_on()) { static uint64_t pns, pcyc; double r = tsc_per_ns(); uint64_t a = atomic_load(&g_aq_ns), c = atomic_load(&g_aq_cyc);
        if (r > 0) LOG("[nvn] translation thread up to frame %llu: %.2f ms a frame translating (CPU %.2f ms)", (unsigned long long)n, (a - pns) / 1e6 / STAT_N, (c - pcyc) / r / 1e6 / STAT_N);
        pns = a; pcyc = c; }
    g_st_n = 0;
}
// The NVN calls of one frame (present to present): how many functions and calls, by name for frames 2 and 3 and every 3000th
static uint32_t *g_calls_prev;
static void nvn_calls(uint64_t n, int snapshot) {
    if (!g_calls_prev && !(g_calls_prev = calloc(nxrt_hle_count, sizeof *g_calls_prev))) return;
    char line[2048]; size_t o = 0; unsigned nf = 0; uint64_t nc = 0;
    int detail = !snapshot && (n == 2 || n == 3 || n % 3000 == 0);
    for (unsigned i = 0; i < nxrt_hle_count; i++) {
        if (nxrt_hle_table[i].kind != NXRT_HLE_NVN) continue;
        uint32_t c = nxrt_hle_calls(nxrt_hle_table[i].name), d = c - g_calls_prev[i];
        g_calls_prev[i] = c;
        if (snapshot || !d) continue;
        nf++; nc += d;
        if (detail && o < sizeof line - 80) { int k = snprintf(line + o, sizeof line - o, " %s=%u", nxrt_hle_table[i].name + 3, d); if (k > 0) o += (size_t)k; }
    }
    if (!snapshot) LOG("[nvn] frame %llu: %u functions %llu calls%s%s", (unsigned long long)n, nf, (unsigned long long)nc, detail ? " |" : "", detail ? line : "");
}
// The crop a present shows: the window's own (nvnWindowSetCrop), else the vi layer's (vi::SetLayerCrop, hle_vi.c); on the Switch both become the crop of the buffer queued to the layer.
// Link's Awakening crops only its layer: in handheld mode it draws 1280x720 into its 1920x1080 window textures and crops the layer to that. A layer crop that does not fit the
// texture is reported once and the whole texture shown. lc receives the layer crop
void dm_vi_layer_crop(int c[4]);   // hle_vi.c: x, y, w, h the game last cropped its layer to (all 0 = none)
static const int *present_crop(const int win[4], int tw, int th, int lc[4]) {
    if (win[2] || win[3]) return win;
    dm_vi_layer_crop(lc);
    if (!lc[2] || !lc[3]) return win;
    if (lc[0] + lc[2] > tw || lc[1] + lc[3] > th) {
        static int once; if (!once++) LOG("[RED] vi layer crop %d,%d %dx%d does not fit the window texture %dx%d: the present shows the whole texture", lc[0], lc[1], lc[2], lc[3], tw, th);
        return win;
    }
    return lc;
}
// Each present hands the texture to the backend (it returns a ticket at once), ticks the guest frame (hid's scripted input, DM_DETCLOCK) and the display.
// NXRT_EXIT_AT_PRESENT=N ends the run after the Nth present (boot tests)
static uint64_t g_last_present_ns;
NVN(nvnQueuePresentTexture, void, NVNqueue *q, NVNwindow *w, int idx) {
    (void)q; win_t *o = (win_t *)w; o->presents++;
    { void swr_trace_poll(void); swr_trace_poll(); }   // SWR_TRACE's window starts and ends at presents
    uint64_t n = atomic_fetch_add(&g_presents, 1) + 1;
    if (n == 1) frame_request("QueuePresentTexture", idx);
    if (n == 60 || n == 600 || n % 3600 == 0) LOG("[nvn] %llu presents (%.1f s)", (unsigned long long)n, (now_ns() - g_t0_ns) / 1e9);
    if (n == 1 || (n + 1) % STAT_N == 0) nvn_calls(n, 1);   // a delta also takes the next snapshot
    else if (n == 2 || n == 3 || n % STAT_N == 0) nvn_calls(n, 0);
    gfx_enter();
    uint64_t t0 = now_ns(), c0 = thread_cyc();
    if (idx >= 0 && idx < o->ntex && idx < 4 && o->tex[idx]) {   // enqueue only, return immediately
        const tex_t *t = CTX(o->tex[idx]); int lc[4]; const int *crop = present_crop(o->crop, (int)t->d.width, (int)t->d.height, lc);
        nvn_cap_present(t, crop);
        if (aq_on()) {
            aop_t op = {2}; op.w = o; op.idx = idx; op.t = t; memcpy(op.crop, crop, sizeof op.crop); op.period = present_period(o) / 1e9;
            gfx_leave(); o->ticket[idx] = AQ_SEQ | aq_push(&op); gfx_enter();   // queued: the gate is the translation thread's to pass
        } else o->ticket[idx] = gfx_present(t, present_period(o) / 1e9, crop);
    }
    gfx_leave();
    atomic_fetch_add(&g_nvn_ns, now_ns() - t0); atomic_fetch_add(&g_nvn_cyc, thread_cyc() - c0);
    unsigned skipped; uint64_t now = now_ns(), at = nxrt_nvn_display(now, present_itv(o), idx, &skipped);
    g_vs_skipped += skipped;
    if (idx >= 0 && idx < 4) o->avail[idx] = at + present_period(o);
    if (g_last_present_ns) frame_stats(n, (now - g_last_present_ns) / 1e6, now, (at - now) / 1e6);
    g_last_present_ns = now;
    cs_dump();
    gfx_set_frame(n);
    dm_guest_frame_tick();
    const char *e = getenv("NXRT_EXIT_AT_PRESENT");
    if (e && n >= strtoull(e, NULL, 0)) {
        LOG("NXRT_EXIT_AT_PRESENT: %llu presents, ending the run", (unsigned long long)n);
        nxrt_log_summary(); ExitProcess(0);
    }
}

// ---------- State-only interfaces used by newer-SDK games (SMBW): buffer objects, pool / sampler / texture queries, views, color state, device and window ----------
// They only read and write the object itself and record no commands. Where the host has no matching capability (texture channel swizzle, depth / stencil sampling mode, sampler reduction, virtual mapping) the first occurrence logs one line and the call returns normally.
#define ONCE_LOG(...) do { static int once_; if (!once_++) LOG("[nvn] " __VA_ARGS__); } while (0)

// Buffer object = pool + offset + size; the GPU address and the CPU mapping are both the pool's host pointer + offset (same convention as MemoryPoolGetBufferAddress)
typedef struct { NVNdevice *dev; const pool_t *pool; int64_t off; uint64_t size; } buf_builder_t;   FITS(buf_builder_t, NVNbufferBuilder);
// Guest ring allocators write offset/size directly at +0x18/+0x20 (issue #723).
// The intervening driver bytes have no host meaning; keep them reserved.
typedef struct { uint32_t magic; const pool_t *pool; uint8_t reserved[8]; int64_t off; uint64_t size; } buf_t; FITS(buf_t, NVNbuffer);
_Static_assert(offsetof(buf_t, off) == 0x18, "NVNbuffer guest offset");
_Static_assert(offsetof(buf_t, size) == 0x20, "NVNbuffer guest size");
#define BB(b) ((buf_builder_t *)(b))
#define BF(b) ((const buf_t *)(b))
NVN(nvnBufferBuilderSetDefaults, void, NVNbufferBuilder *b) { NVNdevice *d = BB(b)->dev; memset(b, 0, sizeof(buf_builder_t)); BB(b)->dev = d; }
NVN(nvnBufferBuilderSetDevice, void, NVNbufferBuilder *b, NVNdevice *d) { BB(b)->dev = d; }
NVN(nvnBufferBuilderSetStorage, void, NVNbufferBuilder *b, NVNmemoryPool *p, ptrdiff_t off, size_t n) { BB(b)->pool = (const pool_t *)p; BB(b)->off = off; BB(b)->size = n; }
NVN(nvnBufferInitialize, NVNboolean, NVNbuffer *buf, const NVNbufferBuilder *b) {
    const buf_builder_t *bb = (const buf_builder_t *)b;
    if (!bb->pool || !bb->pool->mem) { LOG("[RED] BufferInitialize: pool has no storage (pool %p flags %#x offset %#llx size %#llx)", (const void *)bb->pool, bb->pool ? bb->pool->flags : 0, (unsigned long long)bb->off, (unsigned long long)bb->size); return 0; }
    // The NVN driver does not check a buffer's range against its pool, and the host's own GPU access never goes by the buffer: it takes an address and a length from the command (gfx_addr resolves it
    // in the pool's memory and reports a range that leaves the pool as its own red line). Breath of the Wild creates a 256 KB buffer over a 144 KB pool for a screen readback that feeds a
    // compressing encoder (when the pause menu opens, and near a save commit): the buffer's one use is a texture-to-buffer copy of the picture's own extent, which lies inside the pool. Not an error by itself: one line.
    if ((uint64_t)bb->off + bb->size > bb->pool->size) ONCE_LOG("BufferInitialize: a buffer reaches past the end of its pool (offset %#llx size %#llx pool %#llx); only an access that leaves the pool is an error, and it is reported where it happens", (unsigned long long)bb->off, (unsigned long long)bb->size, (unsigned long long)bb->pool->size);
    *(buf_t *)buf = (buf_t){.magic = MAGIC('B', 'U', 'F', 0), .pool = bb->pool, .off = bb->off, .size = bb->size};
    OLOG("BufferInitialize %p pool=%p off=%#llx size=%#llx", (void *)buf, (const void *)bb->pool, (unsigned long long)bb->off, (unsigned long long)bb->size);
    return 1;
}
NVN(nvnBufferFinalize, void, NVNbuffer *buf) { ((buf_t *)buf)->magic = 0; }
NVN(nvnBufferSetDebugLabel, void, NVNbuffer *buf, const char *s) { (void)buf; (void)s; }
NVN(nvnBufferGetAddress, NVNbufferAddress, const NVNbuffer *buf) { return (NVNbufferAddress)(BF(buf)->pool->mem + BF(buf)->off); }
// A buffer or pool the CPU has no access to (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) maps to NULL, as in NVN, though the host's storage is CPU memory: EA SPORTS FC 27's Frostbite
// renderer takes a buffer's GPU address as GetAddress + (its data pointer − Map), with a null data pointer for such a buffer (OpcodeCopySubresource), so a real pointer made the
// copy's destination address 0 (FATAL buffer copy address is in no memory pool)
NVN(nvnBufferMap, void *, const NVNbuffer *buf) { return BF(buf)->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS ? NULL : BF(buf)->pool->mem + BF(buf)->off; }
NVN(nvnBufferGetSize, size_t, const NVNbuffer *buf) { return (size_t)BF(buf)->size; }
NVN(nvnBufferGetMemoryPool, NVNmemoryPool *, const NVNbuffer *buf) { return (NVNmemoryPool *)BF(buf)->pool; }
NVN(nvnBufferGetMemoryOffset, ptrdiff_t, const NVNbuffer *buf) { return (ptrdiff_t)BF(buf)->off; }
NVN(nvnBufferFlushMappedRange, void, const NVNbuffer *buf, ptrdiff_t off, size_t n) { (void)buf; (void)off; (void)n; }        // unified memory: nothing to flush
NVN(nvnBufferInvalidateMappedRange, void, const NVNbuffer *buf, ptrdiff_t off, size_t n) { (void)buf; (void)off; (void)n; }

// Memory pool queries
NVN(nvnMemoryPoolGetFlags, NVNmemoryPoolFlags, const NVNmemoryPool *p) { return (NVNmemoryPoolFlags)((const pool_t *)p)->flags; }
NVN(nvnMemoryPoolMap, void *, const NVNmemoryPool *p) { return ((const pool_t *)p)->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS ? NULL : ((const pool_t *)p)->mem; }   // NULL without CPU access (nvnBufferMap)
NVN(nvnMemoryPoolGetSize, size_t, const NVNmemoryPool *p) { return (size_t)((const pool_t *)p)->size; }
NVN(nvnMemoryPoolInvalidateMappedRange, void, const NVNmemoryPool *p, ptrdiff_t off, size_t n) { (void)p; (void)off; (void)n; }
NVN(nvnMemoryPoolSetDebugLabel, void, NVNmemoryPool *p, const char *s) { (void)p; (void)s; }
// Debug labels: only for NVN debugging tools, unused by the host
NVN(nvnWindowSetDebugLabel, void, NVNwindow *w, const char *s) { (void)w; (void)s; }
NVN(nvnTexturePoolSetDebugLabel, void, NVNtexturePool *p, const char *s) { (void)p; (void)s; }
NVN(nvnSamplerPoolSetDebugLabel, void, NVNsamplerPool *p, const char *s) { (void)p; (void)s; }
NVN(nvnQueueSetDebugLabel, void, NVNqueue *q, const char *s) { (void)q; (void)s; }
NVN(nvnProgramSetDebugLabel, void, NVNprogram *p, const char *s) { (void)p; (void)s; }
NVN(nvnCommandBufferSetDebugLabel, void, NVNcommandBuffer *c, const char *s) { (void)c; (void)s; }
// Virtual pool (flags 0x200): a GPU address range with no storage; MapVirtual maps pages of physical pools into it. A request is 0x28 bytes: {physical pool, physical offset, virtual offset, size, storage class}.
// Host approach: if every request can be described by one "base" (same physical pool, same physical offset - virtual offset), the virtual pool's storage points straight at that memory of the physical pool -- the same semantics as one set of physical pages on the real device,
//   and buffer addresses / Metal buffer lookups land on the block the physical pool already registered. SMBW has a single 128 MB virtual pool, mapped whole by one request onto one physical pool at offset 0.
//   Inconsistent mappings (true sparse) are not implemented: reported red, returns failure. It used to always fail: the game treated the pool as unusable, a buffer later carved from it got a null pool, and MemoryPoolGetFlags read NULL+4 and crashed
typedef struct { NVNmemoryPool *phys; int64_t poff, voff; uint64_t size; int32_t storage_class, _pad; } map_req;
_Static_assert(sizeof(map_req) == 0x28, "NVNmappingRequest");
// True sparse virtual pools (Monster Hunter Rise: RE Engine's GPU heap is one 64 GB virtual pool, built from separately allocated 4 MB physical pools mapped at offsets of their own; boot maps 45 of them
// over the first 180 MB). A virtual pool larger than the whole application pool cannot be one alias of a physical pool, so at Initialize it gets an address range of its own (reserved, inaccessible),
// and every mapping request vm_remaps the physical pool's pages over its part of that range: the virtual addresses then read and write the same memory as the physical pool, as the GPU page table does
// on hardware, and buffer / texture addresses stay CPU addresses. Each run of mapped pages is registered with the replay as a pool of its own (gfx_pool_add), rebuilt when a request joins or splits runs.
// Unmapping puts inaccessible pages back. The reservation is the pool's size, at most 64 GB on the Mac and 1 GB on iOS.
// NVN maps at 4 KB, the host aliases at its 16 KB page: a request that is not 16 KB aligned (Monster Hunter Rise's ASTC 6x6 textures, whose mip tail starts 0x81000 into the texture),
// or one beyond the reservation, cannot be aliased. So every request is also kept in a page table at 4 KB (the physical memory behind each virtual page, and the mapping generation
// that set it), and the host reads textures of a sparse pool through it (nvn_sparse_copy): such a texture is uploaded from what the table points at, with unmapped pages read as 0,
// and uploaded again only when a mapping in its range changes or the CPU writes a physical page behind it (nvn_sparse_since; the physical pools are dirty-tracked). The game fills
// its sparse textures (flag 0x20, 32 GB into the pool) that way: it writes the tiles on the CPU into 4 MB physical pools, then maps them; no GPU copy writes them
// Every virtual pool gets that range at Initialize, since the game may create its textures and buffers before mapping anything; its first mapping request then decides (MapVirtual):
// one that maps the whole pool from offset 0 keeps the alias above (Mario's pool: the range is given back, and no object of the pool came before it), anything else makes the pool
// sparse. OCTOPATH TRAVELER (Unreal Engine 4) gives each streamed texture a virtual pool of its own (0x20000 to 0x560000 bytes, flags 0x2a1), creates the texture in it, maps its mip
// tail from a 4 MB physical pool and later the higher mips from other pages of it, after writing their texels there on the CPU: aliased, the texture had no storage (created before any
// mapping) and the later mappings were refused (`[RED] MapVirtual: non-contiguous mapping`). A pool larger than the application pool is sparse from the start
enum { NVRUN = 1024, VPT_LEAF = 512 };   // a page-table leaf covers 2 MB
typedef struct { const uint8_t *phys; uint64_t gen; } vpte;
struct vlog_ent { uint64_t off, len; };   // a mapping request: the pool's change log (nvn_sparse_log_read), kept once a sparse texture asks for it
enum { VLOG_N = 4096 };
struct vpool { const pool_t *pool; uint8_t *base; uint64_t len, size, top; int nrun; struct { uint64_t lo, hi; } run[NVRUN]; vpte **pt; struct vlog_ent *log; _Atomic uint64_t nlog; };   // len: reserved; size: the pool's; top: highest mapped end so far (logged per 128 MB)
static struct vpool **g_vp; static int g_nvp, g_cvp;   // the pools with a range of their own, under g_vp_mu
static pthread_mutex_t g_vp_mu = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t g_vp_gen;   // bumped per mapping request
static struct vpool *vpool_of(const pool_t *o) { return o->vp; }
static uint8_t *vpool_reserve(pool_t *o) {   // → the reserved range, or NULL
    if (!(o->flags & 0x200)) return NULL;
    uint64_t len = o->size < (64ull << 30) ? o->size : 64ull << 30;
    len = (len + 0xFFFFull) & ~0xFFFFull;   // the allocation granularity
    void *a = VirtualAlloc(NULL, (SIZE_T)len, MEM_RESERVE, PAGE_NOACCESS);
    if (!a) { LOG("[RED] virtual pool %p: cannot reserve %#llx bytes of address space (error %lu)", (const void *)o, (unsigned long long)len, (unsigned long)GetLastError()); return NULL; }
    struct vpool *v = calloc(1, sizeof *v);
    if (!v || !(v->pt = calloc((o->size + (2u << 20) - 1) >> 21, sizeof *v->pt))) die("sparse pool allocation failed");
    v->pool = o; v->base = a; v->len = len; v->size = o->size;
    pthread_mutex_lock(&g_vp_mu);
    if (g_nvp == g_cvp && !(g_vp = realloc(g_vp, (g_cvp = g_cvp ? 2 * g_cvp : 64) * sizeof *g_vp))) die("sparse pool table allocation failed");
    g_vp[g_nvp++] = v; o->vp = v;
    o->vmode = o->size > DM_GAME_APP_POOL ? 2 : 0;
    pthread_mutex_unlock(&g_vp_mu);
    if (o->vmode == 2) LOG("[nvn] virtual pool %p (%#llx bytes, flags %#x): sparse, own address range %p+%#llx, kept in a page table", (const void *)o, (unsigned long long)o->size, o->flags, a, (unsigned long long)len);
    return a;
}
static void vpool_drop(pool_t *o) {   // caller holds g_vp_mu
    struct vpool *v = o->vp; if (!v) return;
    for (int k = 0; k < v->nrun; k++) gfx_pool_del(v->base + v->run[k].lo);
    VirtualFree(v->base, 0, MEM_RELEASE);
    for (uint64_t l = 0; l < (v->size + (2u << 20) - 1) >> 21; l++) free(v->pt[l]);
    for (int i = 0; i < g_nvp; i++) if (g_vp[i] == v) { g_vp[i] = g_vp[--g_nvp]; break; }
    free(v->pt); free(v->log); free(v); o->vp = NULL;
}
static void vpool_release(pool_t *o) { pthread_mutex_lock(&g_vp_mu); vpool_drop(o); pthread_mutex_unlock(&g_vp_mu); }
// Mapped runs of a sparse pool: [lo, hi) sorted, disjoint, adjacent runs merged. Caller holds g_vp_mu
static void vrun_set(struct vpool *v, uint64_t lo, uint64_t hi, int mapped) {
    uint64_t nl[NVRUN][2]; int nn = 0; uint64_t mlo = lo, mhi = hi;
    for (int k = 0; k < v->nrun; k++) {
        uint64_t a = v->run[k].lo, b = v->run[k].hi;
        if (b < lo || a > hi || (!mapped && (b == lo || a == hi))) { if (nn < NVRUN) { nl[nn][0] = a; nl[nn][1] = b; nn++; } continue; }   // untouched (an unmap leaves adjacent runs alone)
        gfx_pool_del(v->base + a);
        if (mapped) { if (a < mlo) mlo = a; if (b > mhi) mhi = b; continue; }   // joins the new run
        if (a < lo && nn < NVRUN) { nl[nn][0] = a; nl[nn][1] = lo; nn++; gfx_pool_add(v->base + a, lo - a); }   // what the unmap leaves on either side
        if (b > hi && nn < NVRUN) { nl[nn][0] = hi; nl[nn][1] = b; nn++; gfx_pool_add(v->base + hi, b - hi); }
    }
    if (mapped && nn < NVRUN) { nl[nn][0] = mlo; nl[nn][1] = mhi; nn++; gfx_pool_add(v->base + mlo, mhi - mlo); }
    if (nn == NVRUN) LOG("[RED] virtual pool %p: more than %d separate mapped runs", (const void *)v->pool, NVRUN);
    for (int a = 1; a < nn; a++) for (int b = a; b > 0 && nl[b][0] < nl[b - 1][0]; b--) { uint64_t t0 = nl[b][0], t1 = nl[b][1]; nl[b][0] = nl[b - 1][0]; nl[b][1] = nl[b - 1][1]; nl[b - 1][0] = t0; nl[b - 1][1] = t1; }
    v->nrun = nn; for (int k = 0; k < nn; k++) { v->run[k].lo = nl[k][0]; v->run[k].hi = nl[k][1]; }
}
// Page table of a sparse pool (see above). Caller holds g_vp_mu
static const vpte *vpt_get(const struct vpool *v, uint64_t o) { const vpte *l = v->pt[o >> 21]; return l ? &l[(o >> 12) & (VPT_LEAF - 1)] : NULL; }
static void vpt_set(struct vpool *v, uint64_t voff, uint64_t size, const uint8_t *phys) {
    uint64_t gen = atomic_fetch_add(&g_vp_gen, 1) + 1;
    for (uint64_t o = 0; o < size; o += 0x1000) {
        vpte **l = &v->pt[(voff + o) >> 21];
        if (!*l && !phys) continue;   // never mapped: stays unmapped
        if (!*l && !(*l = calloc(VPT_LEAF, sizeof **l))) die("sparse pool page table allocation failed");
        (*l)[((voff + o) >> 12) & (VPT_LEAF - 1)] = (vpte){phys ? phys + o : NULL, gen};
    }
}
static int vpool_map(struct vpool *v, const map_req *r) {   // caller holds g_vp_mu; → 0 on failure
    if (r->voff < 0 || (uint64_t)r->voff + r->size > v->size) { LOG("[RED] virtual pool %p: request at %#llx+%#llx is beyond the pool's %#llx", (const void *)v->pool, (unsigned long long)r->voff, (unsigned long long)r->size, (unsigned long long)v->size); return 0; }
    if (((uint64_t)r->voff | r->size | (uint64_t)r->poff) & 0xFFF) { LOG("[RED] virtual pool %p: request %#llx+%#llx (physical offset %#llx) is not 4 KB aligned", (const void *)v->pool, (unsigned long long)r->voff, (unsigned long long)r->size, (unsigned long long)r->poff); return 0; }
    const pool_t *ph = (const pool_t *)r->phys;
    if (ph && (!ph->mem || r->poff < 0 || (uint64_t)r->poff + r->size > ph->size)) { LOG("[RED] virtual pool %p: physical pool %p (%#llx bytes, storage %p) cannot supply %#llx+%#llx", (const void *)v->pool, (const void *)ph, (unsigned long long)ph->size, (void *)ph->mem, (unsigned long long)r->poff, (unsigned long long)r->size); return 0; }
    vpt_set(v, (uint64_t)r->voff, r->size, ph ? ph->mem + r->poff : NULL);
    if (v->log) { v->log[atomic_load_explicit(&v->nlog, memory_order_relaxed) % VLOG_N] = (struct vlog_ent){(uint64_t)r->voff, r->size}; atomic_fetch_add_explicit(&v->nlog, 1, memory_order_release); }
    // nxnative aliases the physical pages into the pool's range (vm_remap), so the CPU and the GPU address them there too. Windows cannot map pages of private memory
    // (the game's heap) at a second address, so a sparse pool lives in its page table only: textures read it (nvn_sparse_copy), its range stays reserved and inaccessible
    static _Atomic uint64_t n; if (r->phys && !atomic_fetch_add(&n, 1)) LOG("[RED] virtual pool %p: a sparse mapping (%#llx+%#llx) is kept in the page table only on Windows: textures read it, the CPU and the GPU cannot address the pool's range (later cases are not logged)",
                                                                           (const void *)v->pool, (unsigned long long)r->voff, (unsigned long long)r->size);
    uint64_t end = (uint64_t)r->voff + r->size;
    if (r->phys && end >> 27 > v->top >> 27) LOG("[nvn] virtual pool %p: mapped up to %#llx (page table)", (const void *)v->pool, (unsigned long long)end);
    if (r->phys && end > v->top) v->top = end;
    return 1;
}
NVN(nvnMemoryPoolMapVirtual, NVNboolean, NVNmemoryPool *p, int n, const NVNmappingRequest *rq) {
    pool_t *o = (pool_t *)p; const map_req *r = (const map_req *)rq;
    pthread_mutex_lock(&g_vp_mu);
    if (o->vp && o->vmode == 0 && n > 0) {   // the first mapping decides (see the sparse pools above)
        if (r[0].phys && r[0].voff == 0 && r[0].size == o->size) { vpool_drop(o); o->mem = NULL; o->vmode = 1; }
        else o->vmode = 2;
    }
    struct vpool *v = o->vmode == 2 ? vpool_of(o) : NULL; int ok = 1;
    for (int i = 0; v && i < n && ok; i++) ok = vpool_map(v, &r[i]);
    pthread_mutex_unlock(&g_vp_mu);
    if (v) return (NVNboolean)ok;
    for (int i = 0; i < n; i++) {
        const pool_t *ph = (const pool_t *)r[i].phys;
        if (!ph) continue;   // Unmapping (null physical pool): the host keeps the storage as is
        if (!ph->mem) { LOG("[RED] MapVirtual: physical pool %p has no storage", (const void *)ph); return 0; }
        uint8_t *base = ph->mem + r[i].poff - r[i].voff;
        if (r[i].voff < 0 || (uint64_t)r[i].voff + r[i].size > o->size || (uint64_t)r[i].poff + r[i].size > ph->size) { LOG("[RED] MapVirtual: request out of range (virtual %#llx+%#llx / pool %#llx, physical %#llx / pool %#llx)", (unsigned long long)r[i].voff, (unsigned long long)r[i].size, (unsigned long long)o->size, (unsigned long long)r[i].poff, (unsigned long long)ph->size); return 0; }
        if (o->mem && o->mem != base) { LOG("[RED] MapVirtual: non-contiguous mapping (true sparse) not implemented: virtual pool %p already maps to %p, this request wants %p", (void *)o, (void *)o->mem, (void *)base); return 0; }
        if (!o->mem) LOG("[nvn] MapVirtual: the storage of virtual pool %p (%#llx bytes) points at physical pool %p + %#llx", (void *)o, (unsigned long long)o->size, (const void *)ph, (unsigned long long)(r[i].poff - r[i].voff));
        o->mem = base;
    }
    return 1;
}

// Samplers: the getters read the parameters in the object. Reduction (min / max of the texels a filtered sample covers) maps to the Metal 26 sampler reduction mode (nvn_mtl.m samp_by_id)
NVN(nvnSamplerBuilderSetReductionFilter, void, NVNsamplerBuilder *b, NVNsamplerReduction r) { SB(b)->reduction = (uint8_t)r; }
NVN(nvnSamplerSetDebugLabel, void, NVNsampler *s, const char *l) { (void)s; (void)l; }
#define SP(s) (&((const samp_t *)(s))->p)
NVN(nvnSamplerGetMinMagFilter, void, const NVNsampler *s, NVNminFilter *mi, NVNmagFilter *ma) { *mi = (NVNminFilter)SP(s)->minf; *ma = (NVNmagFilter)SP(s)->magf; }
NVN(nvnSamplerGetWrapMode, void, const NVNsampler *s, NVNwrapMode *a, NVNwrapMode *b, NVNwrapMode *c) { *a = (NVNwrapMode)SP(s)->ws; *b = (NVNwrapMode)SP(s)->wt; *c = (NVNwrapMode)SP(s)->wr; }
NVN(nvnSamplerGetLodClamp, void, const NVNsampler *s, float *lo, float *hi) { *lo = SP(s)->lod_min; *hi = SP(s)->lod_max; }
NVN(nvnSamplerGetLodBias, float, const NVNsampler *s) { return SP(s)->lod_bias; }
NVN(nvnSamplerGetCompare, void, const NVNsampler *s, NVNcompareMode *m, NVNcompareFunc *f) { *m = (NVNcompareMode)SP(s)->cmp_mode; *f = (NVNcompareFunc)SP(s)->cmp_func; }
NVN(nvnSamplerGetBorderColor, void, const NVNsampler *s, float *c) { memcpy(c, SP(s)->border, 16); }
NVN(nvnSamplerGetMaxAnisotropy, float, const NVNsampler *s) { return SP(s)->aniso; }
NVN(nvnSamplerGetReductionFilter, NVNsamplerReduction, const NVNsampler *s) { return (NVNsamplerReduction)SP(s)->reduction; }
// whether two samplers have the same state (games use it to deduplicate sampler pool entries)
NVN(nvnSamplerCompare, NVNboolean, const NVNsampler *a, const NVNsampler *b) {
    const samp_par *x = SP(a), *y = SP(b);
    return x->minf == y->minf && x->magf == y->magf && x->ws == y->ws && x->wt == y->wt && x->wr == y->wr && x->cmp_mode == y->cmp_mode && x->cmp_func == y->cmp_func && x->reduction == y->reduction &&
           x->lod_min == y->lod_min && x->lod_max == y->lod_max && x->lod_bias == y->lod_bias && x->aniso == y->aniso && !memcmp(x->border, y->border, sizeof x->border);
}

// Texture builder / texture / view
NVN(nvnTextureBuilderSetSize2D, void, NVNtextureBuilder *b, int w, int h) { TB(b)->d.width = w; TB(b)->d.height = h; TB(b)->d.depth = 1; }
NVN(nvnTextureBuilderSetSize3D, void, NVNtextureBuilder *b, int w, int h, int d) { TB(b)->d.width = w; TB(b)->d.height = h; TB(b)->d.depth = d; }
NVN(nvnTextureBuilderSetSwizzle, void, NVNtextureBuilder *b, NVNtextureSwizzle r, NVNtextureSwizzle g, NVNtextureSwizzle bl, NVNtextureSwizzle a) {
    tex_builder_t *o = (tex_builder_t *)b; o->set_swz = 1; o->swz = (uint16_t)((r & 7) | (g & 7) << 3 | (bl & 7) << 6 | (a & 7) << 9);
    if (r != NVN_TEXTURE_SWIZZLE_R || g != NVN_TEXTURE_SWIZZLE_G || bl != NVN_TEXTURE_SWIZZLE_B || a != NVN_TEXTURE_SWIZZLE_A) ONCE_LOG("TextureBuilderSetSwizzle not identity (%d %d %d %d): remapped with a Metal view when sampling", (int)r, (int)g, (int)bl, (int)a);
}
// STENCIL: the texture reads the stencil of its depth-stencil storage, as a stencil-mode view does (tex_by_id in nvn_mtl.m). Unreal Engine 4.27 makes its stencil shader resources this way: a second texture
// on the depth buffer's storage, built in stencil mode (DRAGON QUEST VII Reimagined's character occlusion mask, edbfc7a088be545e, tests stencil bits 0x1 and 0x84 of two of them; read as depth, the
// bits followed the depth values and the hero was drawn as a dithered silhouette inside a dithered square)
NVN(nvnTextureBuilderSetDepthStencilMode, void, NVNtextureBuilder *b, NVNtextureDepthStencilMode m) { TB(b)->stencil = m == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL; }
NVN(nvnTextureBuilderGetStorageClass, NVNstorageClass, const NVNtextureBuilder *b) { (void)b; return 0; }
// Used by Bowser's Fury: the texture's channel swizzle (set by TextureBuilderSetSwizzle at build time, identity R G B A when unset) and storage class (the host has none, 0 as for the builder)
NVN(nvnTextureGetSwizzle, void, const NVNtexture *t, NVNtextureSwizzle *r, NVNtextureSwizzle *g, NVNtextureSwizzle *b, NVNtextureSwizzle *a) {
    unsigned w = tex_swz(CTX(t));
    if (r) *r = (NVNtextureSwizzle)(w & 7); if (g) *g = (NVNtextureSwizzle)(w >> 3 & 7); if (b) *b = (NVNtextureSwizzle)(w >> 6 & 7); if (a) *a = (NVNtextureSwizzle)(w >> 9 & 7);
}
NVN(nvnTextureGetStorageClass, NVNstorageClass, const NVNtexture *t) { (void)t; return 0; }
NVN(nvnTextureBuilderGetZCullStorageSize, size_t, const NVNtextureBuilder *b) { return zcull_size(&TB(b)->d); }   // as for the texture (zcull_size)
// Packaged texture data in CPU memory (not written into the pool by the game): NVN copies it into the texture storage at TextureInitialize (the CPU cannot write a compressible pool; the driver copies). 13 Sentinels creates all its textures this way
NVN(nvnTextureBuilderSetPackagedTextureData, void, NVNtextureBuilder *b, const void *data) { TB(b)->pkg_data = data; }
NVN(nvnTextureGetFormat, NVNformat, const NVNtexture *t) { return (NVNformat)CTX(t)->d.format; }
NVN(nvnTextureGetTarget, NVNtextureTarget, const NVNtexture *t) { return (NVNtextureTarget)CTX(t)->d.target; }
NVN(nvnTextureSetDebugLabel, void, NVNtexture *t, const char *s) { (void)t; (void)s; }
NVN(nvnTextureViewGetLevels, NVNboolean, const NVNtextureView *v, int *base, int *n) { const view_t *w = (const view_t *)v; *base = w->base_level; *n = w->nlevels; return w->set_levels; }
NVN(nvnTextureViewGetTarget, NVNboolean, const NVNtextureView *v, NVNtextureTarget *t) { const view_t *w = (const view_t *)v; *t = (NVNtextureTarget)w->target; return w->set_target; }
NVN(nvnTextureViewSetDepthStencilMode, void, NVNtextureView *v, NVNtextureDepthStencilMode m) { VW(v)->set_dsmode = 1; VW(v)->dsmode = (uint8_t)m; }   // STENCIL: the replay binds a stencil-only view (tex_by_id)
NVN(nvnTextureViewSetSwizzle, void, NVNtextureView *v, NVNtextureSwizzle r, NVNtextureSwizzle g, NVNtextureSwizzle bl, NVNtextureSwizzle a) {
    view_t *w = VW(v); w->set_swizzle = 1; w->swz[0] = (uint8_t)r; w->swz[1] = (uint8_t)g; w->swz[2] = (uint8_t)bl; w->swz[3] = (uint8_t)a;
    if (r != NVN_TEXTURE_SWIZZLE_R || g != NVN_TEXTURE_SWIZZLE_G || bl != NVN_TEXTURE_SWIZZLE_B || a != NVN_TEXTURE_SWIZZLE_A) ONCE_LOG("TextureViewSetSwizzle not identity (%d %d %d %d): remapped with a Metal view when sampling", (int)r, (int)g, (int)bl, (int)a);
}

// Color state: alpha test function, logic op (the fields already exist in color_st)
NVN(nvnColorStateSetAlphaTest, void, NVNcolorState *s, NVNalphaFunc f) { ST(color_st, s)->alpha_func = (uint32_t)f; }
NVN(nvnColorStateSetLogicOp, void, NVNcolorState *s, NVNlogicOp op) { ST(color_st, s)->logic_op = (uint32_t)op; }
// Getters and the rest of the multisample state: Pokémon Legends: Z-A (Trinity, SDK 19) reads its state objects and builders back when it builds its pipeline states; the stub returned
// whatever was left in x0 (the object's address), so a blend enable or depth test read back as true, a texture builder's width as an address
NVN(nvnColorStateGetBlendEnable, NVNboolean, const NVNcolorState *s, int i) { return i >= 0 && i < 8 && (((const color_st *)s)->blend_en >> i & 1); }
NVN(nvnColorStateGetAlphaTest, NVNalphaFunc, const NVNcolorState *s) { return (NVNalphaFunc)((const color_st *)s)->alpha_func; }
NVN(nvnColorStateGetLogicOp, NVNlogicOp, const NVNcolorState *s) { return (NVNlogicOp)((const color_st *)s)->logic_op; }
NVN(nvnChannelMaskStateGetChannelMask, void, const NVNchannelMaskState *s, int i, NVNboolean *r, NVNboolean *g, NVNboolean *b, NVNboolean *a) {
    uint32_t m = i >= 0 && i < 8 ? ((const chmask_st *)s)->mask >> (4 * i) & 0xf : 0;
    if (r) *r = (m & 1) != 0; if (g) *g = (m & 2) != 0; if (b) *b = (m & 4) != 0; if (a) *a = (m & 8) != 0;
}
NVN(nvnBlendStateGetBlendTarget, int, const NVNblendState *s) { return (int)((const blend_st *)s)->target; }
NVN(nvnBlendStateGetBlendFunc, void, const NVNblendState *s, NVNblendFunc *sr, NVNblendFunc *dr, NVNblendFunc *sa, NVNblendFunc *da) {
    const blend_st *o = (const blend_st *)s;
    if (sr) *sr = (NVNblendFunc)o->src_rgb; if (dr) *dr = (NVNblendFunc)o->dst_rgb; if (sa) *sa = (NVNblendFunc)o->src_a; if (da) *da = (NVNblendFunc)o->dst_a;
}
NVN(nvnBlendStateGetBlendEquation, void, const NVNblendState *s, NVNblendEquation *er, NVNblendEquation *ea) {
    const blend_st *o = (const blend_st *)s; if (er) *er = (NVNblendEquation)o->eq_rgb; if (ea) *ea = (NVNblendEquation)o->eq_a;
}
NVN(nvnBlendStateGetAdvancedMode, NVNblendAdvancedMode, const NVNblendState *s) { return (NVNblendAdvancedMode)((const blend_st *)s)->adv; }
NVN(nvnBlendStateGetAdvancedPremultipliedSrc, NVNboolean, const NVNblendState *s) { return !((const blend_st *)s)->adv_npm; }
NVN(nvnBlendStateGetAdvancedOverlap, NVNblendAdvancedOverlap, const NVNblendState *s) { return (NVNblendAdvancedOverlap)((const blend_st *)s)->adv_ovl; }
NVN(nvnPolygonStateGetCullFace, NVNface, const NVNpolygonState *s) { return (NVNface)((const poly_st *)s)->cull; }
NVN(nvnPolygonStateGetFrontFace, NVNfrontFace, const NVNpolygonState *s) { return (NVNfrontFace)((const poly_st *)s)->front; }
NVN(nvnPolygonStateGetPolygonMode, NVNpolygonMode, const NVNpolygonState *s) { return (NVNpolygonMode)((const poly_st *)s)->mode; }
NVN(nvnPolygonStateGetPolygonOffsetEnables, NVNpolygonOffsetEnable, const NVNpolygonState *s) { return (NVNpolygonOffsetEnable)((const poly_st *)s)->offset_en; }
NVN(nvnDepthStencilStateGetDepthTestEnable, NVNboolean, const NVNdepthStencilState *s) { return ((const ds_st *)s)->dt; }
NVN(nvnDepthStencilStateGetDepthWriteEnable, NVNboolean, const NVNdepthStencilState *s) { return ((const ds_st *)s)->dw; }
NVN(nvnDepthStencilStateGetDepthFunc, NVNdepthFunc, const NVNdepthStencilState *s) { return (NVNdepthFunc)((const ds_st *)s)->df; }
NVN(nvnDepthStencilStateGetStencilTestEnable, NVNboolean, const NVNdepthStencilState *s) { return ((const ds_st *)s)->st; }
NVN(nvnDepthStencilStateGetStencilFunc, NVNstencilFunc, const NVNdepthStencilState *s, NVNface f) { const ds_st *o = (const ds_st *)s; return (NVNstencilFunc)(f == NVN_FACE_BACK ? o->sf_b : o->sf_f); }
NVN(nvnDepthStencilStateGetStencilOp, void, const NVNdepthStencilState *s, NVNface f, NVNstencilOp *fail, NVNstencilOp *zfail, NVNstencilOp *zpass) {
    const ds_st *o = (const ds_st *)s; int b = f == NVN_FACE_BACK;
    if (fail) *fail = (NVNstencilOp)(b ? o->fail_b : o->fail_f); if (zfail) *zfail = (NVNstencilOp)(b ? o->zfail_b : o->zfail_f); if (zpass) *zpass = (NVNstencilOp)(b ? o->zpass_b : o->zpass_f);
}
// Multisample state: recorded only (the replay uses the sample count, multisample enable and alpha to coverage)
NVN(nvnMultisampleStateSetAlphaToCoverageDither, void, NVNmultisampleState *s, uint32_t e) { ST(ms_st, s)->a2c_dither = B(e) != 0; }
NVN(nvnMultisampleStateSetRasterSamples, void, NVNmultisampleState *s, int n) { ST(ms_st, s)->raster_samples = (uint8_t)n; }
NVN(nvnMultisampleStateSetCoverageModulationMode, void, NVNmultisampleState *s, NVNcoverageModulationMode m) { ST(ms_st, s)->cov_mod = (uint8_t)m; }
NVN(nvnMultisampleStateSetCoverageToColorEnable, void, NVNmultisampleState *s, uint32_t e) { ST(ms_st, s)->cov2color_en = B(e) != 0; }
NVN(nvnMultisampleStateSetCoverageToColorOutput, void, NVNmultisampleState *s, int i) { ST(ms_st, s)->cov2color_out = (uint8_t)i; }
NVN(nvnMultisampleStateSetSampleLocationsGridEnable, void, NVNmultisampleState *s, uint32_t e) { ST(ms_st, s)->sl_grid = B(e) != 0; }
NVN(nvnMultisampleStateGetAlphaToCoverageEnable, NVNboolean, const NVNmultisampleState *s) { return ((const ms_st *)s)->a2c; }
NVN(nvnMultisampleStateGetAlphaToCoverageDither, NVNboolean, const NVNmultisampleState *s) { return ((const ms_st *)s)->a2c_dither; }
NVN(nvnMultisampleStateGetRasterSamples, int, NVNmultisampleState *s) { return ((const ms_st *)s)->raster_samples; }
NVN(nvnMultisampleStateGetCoverageModulationMode, NVNcoverageModulationMode, const NVNmultisampleState *s) { return (NVNcoverageModulationMode)((const ms_st *)s)->cov_mod; }
NVN(nvnMultisampleStateGetCoverageToColorEnable, NVNboolean, const NVNmultisampleState *s) { return ((const ms_st *)s)->cov2color_en; }
NVN(nvnMultisampleStateGetCoverageToColorOutput, int, const NVNmultisampleState *s) { return ((const ms_st *)s)->cov2color_out; }
NVN(nvnMultisampleStateGetSampleLocationsEnable, NVNboolean, const NVNmultisampleState *s) { return ((const ms_st *)s)->sl_en; }
NVN(nvnMultisampleStateGetSampleLocationsGridEnable, NVNboolean, const NVNmultisampleState *s) { return ((const ms_st *)s)->sl_grid; }
// Builder and pool queries
NVN(nvnBufferBuilderGetSize, size_t, const NVNbufferBuilder *b) { return (size_t)((const buf_builder_t *)b)->size; }
NVN(nvnBufferBuilderGetMemoryOffset, ptrdiff_t, const NVNbufferBuilder *b) { return (ptrdiff_t)((const buf_builder_t *)b)->off; }
// The storage set with SetStorage (EA SPORTS FC 27's Frostbite renderer reads it back from its texture builders before it initializes the textures). The pool getters return the
// pool's address, as nvnTextureGetMemoryPool does; the header the prototypes come from declares the builders' by value, so they get no signature check
NVN_NOPROTO(nvnBufferBuilderGetMemoryPool, NVNmemoryPool *, const NVNbufferBuilder *b) { return (NVNmemoryPool *)((const buf_builder_t *)b)->pool; }
NVN_NOPROTO(nvnTextureBuilderGetMemoryPool, NVNmemoryPool *, const NVNtextureBuilder *b) { return (NVNmemoryPool *)((const tex_builder_t *)b)->pool; }
NVN(nvnTextureBuilderGetMemoryOffset, ptrdiff_t, const NVNtextureBuilder *b) { return (ptrdiff_t)((const tex_builder_t *)b)->off; }
NVN_NOPROTO(nvnEventBuilderGetMemoryPool, NVNmemoryPool *, const NVNeventBuilder *b) { return (NVNmemoryPool *)((const ev_builder_t *)b)->pool; }
NVN_NOPROTO(nvnEventBuilderGetMemoryOffset, int64_t, const NVNeventBuilder *b) { return ((const ev_builder_t *)b)->off; }
NVN(nvnTextureBuilderGetTarget, NVNtextureTarget, const NVNtextureBuilder *b) { return (NVNtextureTarget)((const tex_builder_t *)b)->d.target; }
NVN(nvnTextureBuilderGetWidth, int, const NVNtextureBuilder *b) { return (int)((const tex_builder_t *)b)->d.width; }
NVN(nvnTextureBuilderGetHeight, int, const NVNtextureBuilder *b) { return (int)((const tex_builder_t *)b)->d.height; }
NVN(nvnTextureBuilderGetDepth, int, const NVNtextureBuilder *b) { return (int)((const tex_builder_t *)b)->d.depth; }
NVN(nvnTextureBuilderGetLevels, int, const NVNtextureBuilder *b) { return (int)((const tex_builder_t *)b)->d.levels; }
NVN(nvnTextureBuilderGetFormat, NVNformat, const NVNtextureBuilder *b) { return (NVNformat)((const tex_builder_t *)b)->d.format; }
NVN(nvnTextureBuilderGetSamples, int, const NVNtextureBuilder *b) { return ((const tex_builder_t *)b)->d.samples; }
NVN(nvnTextureBuilderGetStride, ptrdiff_t, const NVNtextureBuilder *b) { return (ptrdiff_t)((const tex_builder_t *)b)->d.stride; }
NVN(nvnTextureViewGetFormat, NVNboolean, const NVNtextureView *v, NVNformat *f) { const view_t *w = (const view_t *)v; if (f) *f = (NVNformat)w->format; return w->set_format; }
NVN(nvnTextureViewGetDepthStencilMode, NVNboolean, const NVNtextureView *v, NVNtextureDepthStencilMode *m) { const view_t *w = (const view_t *)v; if (m) *m = (NVNtextureDepthStencilMode)w->dsmode; return w->set_dsmode; }
NVN(nvnTextureViewGetSwizzle, NVNboolean, const NVNtextureView *v, NVNtextureSwizzle *r, NVNtextureSwizzle *g, NVNtextureSwizzle *b, NVNtextureSwizzle *a) {
    const view_t *w = (const view_t *)v; NVNtextureSwizzle *o[4] = {r, g, b, a};
    for (int c = 0; c < 4; c++) if (o[c]) *o[c] = (NVNtextureSwizzle)(w->set_swizzle ? w->swz[c] : NVN_TEXTURE_SWIZZLE_R + c);
    return w->set_swizzle;
}
NVN(nvnTextureGetDepthStencilMode, NVNtextureDepthStencilMode, const NVNtexture *t) { return ((const tex_t *)t)->stencil ? NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL : NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH; }
// Queue priority: the host has one Metal queue
NVN_NOPROTO(nvnQueueBuilderSetQueuePriority, void, NVNqueueBuilder *b, int prio) { (void)b; (void)prio; }
// The driver's external graphics debugger switch (Nsight / NVN debugger attach): Super Smash Bros. Ultimate fetches it with nvnDeviceGetProcAddress(NULL, …) before creating its device and calls
// it without a null check (main+0x37fe92c); the host has no debugger to attach
NVN_NOPROTO(nvnSetExternalDebuggerEnabledNVX, void, NVNboolean enable) { (void)enable; }
// Transform feedback (vertex outputs captured into buffers): Pokémon Legends: Z-A skins its characters this way. A vertex-only program writes skinned positions and packed normals / tangents
// under rasterizer discard, and the character draws read those buffers as vertex streams. Recorded here and run at replay by the shader's main0_tf entry point, a vertex function that stores the
// captured outputs into the bound buffers (dmsh_build.py vs_tf, the layout from the program's control section; nvn_mtl.m tf_* for the replay)
NVN(nvnCommandBufferBindTransformFeedbackBuffer, void, NVNcommandBuffer *c, int i, NVNbufferAddress a, size_t n) { *REC(c, OP_TF_BIND, p_tf_bind) = (p_tf_bind){i, 0, a, n}; }
NVN(nvnCommandBufferBindTransformFeedbackBuffers, void, NVNcommandBuffer *c, int first, int n, const NVNbufferRange *r) {
    for (int i = 0; i < n; i++) *REC(c, OP_TF_BIND, p_tf_bind) = (p_tf_bind){first + i, 0, r[i].address, r[i].size};
}
NVN(nvnCommandBufferBeginTransformFeedback, void, NVNcommandBuffer *c, NVNbufferAddress a) { *REC(c, OP_TF_STATE, p_tf_state) = (p_tf_state){0, 0, a}; }
NVN(nvnCommandBufferEndTransformFeedback, void, NVNcommandBuffer *c, NVNbufferAddress a) { *REC(c, OP_TF_STATE, p_tf_state) = (p_tf_state){1, 0, a}; }
NVN(nvnCommandBufferPauseTransformFeedback, void, NVNcommandBuffer *c, NVNbufferAddress a) { *REC(c, OP_TF_STATE, p_tf_state) = (p_tf_state){2, 0, a}; }
NVN(nvnCommandBufferResumeTransformFeedback, void, NVNcommandBuffer *c, NVNbufferAddress a) { *REC(c, OP_TF_STATE, p_tf_state) = (p_tf_state){3, 0, a}; }
NVN(nvnCommandBufferDrawTransformFeedback, void, NVNcommandBuffer *c, NVNdrawPrimitive m, NVNbufferAddress a) {
    *REC(c, OP_DRAW, p_draw) = (p_draw){.prim=m,.instances=1,.ind=a+16};
}

// Device: combined texture handle = texture id | sampler id << 20 (same encoding as the separate handles)
NVN(nvnDeviceGetTextureHandle, NVNtextureHandle, const NVNdevice *d, int tex, int samp) { (void)d; return (NVNtextureHandle)((uint64_t)(uint32_t)tex | (uint64_t)(uint32_t)samp << 20); }
// Handle for texel fetches only (Monster Hunter Rise): a fetch reads the texture descriptor and no sampler, so the handle is the texture ID with sampler ID 0
NVN(nvnDeviceGetTexelFetchHandle, NVNtextureHandle, const NVNdevice *d, int tex) { (void)d; return (NVNtextureHandle)(uint64_t)(uint32_t)tex; }
// Convert the raw 614.4 MHz report to ns on the same guest clock as GetCurrentTimestampInNanoseconds.
NVN(nvnDeviceGetTimestampInNanoseconds, uint64_t, const NVNdevice *d, const NVNcounterData *c) { (void)d; return nvn_timestamp_to_ns(c->timestamp); }
// Current GPU time (ns): the same clock as the counter timestamps -- the guest clock (guest_ns, nvn_mtl.m gpu_ticks)
NVN(nvnDeviceGetCurrentTimestampInNanoseconds, uint64_t, const NVNdevice *d) { (void)d; return guest_ns(); }
NVN(nvnDeviceRegisterFastClearColor, NVNboolean, NVNdevice *d, const float *c, NVNformat f) { (void)d; (void)c; (void)f; return 1; }
// Same as the color one: a fast clear value is only a hint to the Tegra hardware (Tears of the Kingdom registers its depth clear at startup)
NVN(nvnDeviceRegisterFastClearDepth, NVNboolean, NVNdevice *d, float v) { (void)d; (void)v; return 1; }
// The size of the driver's cache for shaders it compiles from intermediate code (The Great Ace Attorney Chronicles sets 10 at startup); the host compiles nothing at run time
NVN(nvnDeviceSetIntermediateShaderCache, void, NVNdevice *d, int n) { (void)d; (void)n; }

// Window: number of active swapchain textures (the host rotates through the count given at creation; logged once)
NVN_NOPROTO(nvnWindowBuilderSetNumActiveTextures, void, NVNwindowBuilder *b, int n) { (void)b; ONCE_LOG("WindowBuilderSetNumActiveTextures %d (the host rotates through the SetTextures count)", n); }
NVN_NOPROTO(nvnWindowSetNumActiveTextures, void, NVNwindow *w, int n) { (void)w; ONCE_LOG("WindowSetNumActiveTextures %d (the host rotates through the SetTextures count)", n); }

// Command buffer memory usage: record bytes written in this recording (accumulated across segments; the host does not use control memory, always 0); the memory callback switch is ignored.
// It used to be cur - begin: once a recording overflowed into a host segment, cur is in host memory (an address below the guest begin) and it returned 0. SMBW asks for this value after recording each material setup handle,
// and on 0 (with control memory also 0) treats the recording as empty and marks the handle invalid (from main+0x4bb6c it clears bit 0 of the material entry +0x50); later draws no longer CALL it and inherit the previous object's vertex buffers / textures:
// on the iPhone a batch of draws on odd frames of the opening course fetched vertices out of bounds, giving huge triangles and NaN (black band, black-edged glow, grey stripes). Without overflow the accumulated value equals cur - begin.
// As in NVN, the count runs from the start of the command memory the game last added (AddCommandMemory), not from BeginRecording: Fire Emblem: Three Houses adds one 16 MB chunk of a ring to its main
// command buffer and records every frame into it; its memory callback (main+0x9ca960) takes chunk start + used as the position the GPU has reached and frees the ring up to there when the frame's fence
// signals. Counted per recording, the position stayed at one frame's size (about 18 KB), the ring never freed past it, and after the first 16 MB (about 15 s) the callback waited for ring space forever
NVN(nvnCommandBufferGetCommandMemoryUsed, size_t, const NVNcommandBuffer *c) { const cmd_t *m = (const cmd_t *)c;
    return m->begin ? (size_t)m->used : 0; }
NVN(nvnCommandBufferGetControlMemoryUsed, size_t, const NVNcommandBuffer *c) { (void)c; return 0; }
NVN(nvnCommandBufferGetControlMemoryFree, size_t, const NVNcommandBuffer *c) { return ((const cmd_t *)c)->ctl_size; }
NVN(nvnCommandBufferGetControlMemorySize, size_t, const NVNcommandBuffer *c) { return ((const cmd_t *)c)->ctl_size; }
// Size and free bytes of the command memory the game last added (DRAGON QUEST XI S, Unreal Engine 4.18): as in NVN, size = used + free; free is what is left in the current segment
NVN(nvnCommandBufferGetCommandMemoryFree, size_t, const NVNcommandBuffer *c) { const cmd_t *m = (const cmd_t *)c; return m->cur && m->end > m->cur ? (size_t)(m->end - m->cur) : 0; }
NVN(nvnCommandBufferGetCommandMemorySize, size_t, const NVNcommandBuffer *c) { return h_nvnCommandBufferGetCommandMemoryUsed(c) + h_nvnCommandBufferGetCommandMemoryFree(c); }
NVN_NOPROTO(nvnCommandBufferSetCommandMemoryCallbackEnabled, void, NVNcommandBuffer *c, uint32_t on) { (void)c; (void)on; }

// nvnProgramSetShadersExt (NVN 55): each stage is 0x18 bytes = NVNshaderData{data, control} + one 8-byte extension field (filled from SMBW main 0x1074ff0).
// The host identifies stages by the control section and does not use the extension field: convert to plain NVNshaderData and hand it to SetShaders.
typedef struct { NVNbufferAddress data; const void *control; uint64_t ext; } shader_data_ext;
NVN_NOPROTO(nvnProgramSetShadersExt, NVNboolean, NVNprogram *p, int n, const shader_data_ext *sd) {
    if (n < 0 || n > 6) { LOG("[RED] ProgramSetShadersExt stage count %d", n); return 0; }
    NVNshaderData plain[6];
    for (int i = 0; i < n; i++) { plain[i].data = sd[i].data; plain[i].control = sd[i].control; if (sd[i].ext) ONCE_LOG("ProgramSetShadersExt extension field nonzero (%#llx): unused by the host", (unsigned long long)sd[i].ext); }
    return h_nvnProgramSetShaders(p, n, plain);
}

// nvnProgramSetSampleShading (NVN 53+): whether the program's fragment shader runs once per sample of a multisampled target. Metroid Prime Remastered sets it on its programs
// (modes 0, 1 and 2, from CGraphicsDeviceContext's sample shading state). Metal has no such switch: a fragment function runs per sample when it reads the sample index ([[sample_id]],
// what PIXLD.MYINDEX translates to) or interpolates at a sample, and once per pixel otherwise, so the mode is not applied; on the single-sample targets met so far the two agree
NVN_NOPROTO(nvnProgramSetSampleShading, NVNboolean, NVNprogram *p, int mode) { (void)p; ONCE_LOG("nvnProgramSetSampleShading mode %d: Metal shades per sample only where the shader reads the sample index", mode); return 1; }

// ---------- Commands used by newer-SDK games (SMBW): most map onto existing record formats, the replay layer is unchanged ----------
// Singular / combined forms → plural / separate forms; state the replay layer does not have yet (conditional render, depth bounds, ...) is logged once and drawing continues (better to over-draw than to drop).
#include "nvn_texio.h"

NVN(nvnCommandBufferCallCommands, void, NVNcommandBuffer *c, int n, const NVNcommandHandle *h) { h_nvnCommandBufferCopyCommands(c, n, h); }   // the replay layer treats CALL and COPY alike
// Combined texture handle (texture id | sampler id << 20) is recorded into the texture slot as is: the replay layer takes the texture from the low 20 bits and the sampler from the high bits (nvn_mtl.m NVN_RB_SAMPT)
// One combined handle: recorded as the hardware lays the command out, the handle 16 bytes in, where Luigi's Mansion 3 patches it after recording (nvn_obj.h p_bind_tex1)
NVN(nvnCommandBufferBindTexture, void, NVNcommandBuffer *c, NVNshaderStage st, int i, NVNtextureHandle h) { st = st_host(st);
    int32_t tb = CB(c)->tex_base, sb = CB(c)->samp_base;
    if (tb || sb) { *REC(c, OP_BIND_TEX1, p_bind_tex1) = (p_bind_tex1){st, i, (uint64_t)h, tb, sb}; return; }
    p_bind_tex1 *r = rec(c, OP_BIND_TEX1, offsetof(p_bind_tex1, tbase)); r->stage = st | BIND_TEX1_NOBASE; r->slot = i; r->h = (uint64_t)h;   // no bases: the record ends after h
}
_Static_assert(offsetof(p_bind_tex1, h) == 8, "a single texture bind keeps its handle 16 bytes into the record (8 of header)");
NVN(nvnCommandBufferBindTextures, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, const NVNtextureHandle *h) { st = st_host(st); bind_handles(c, OP_BIND_TEX, st, first, n, (const uint64_t *)h); }
// "Deferred" binding: the GPU reads the handles h points to when it executes this command, not at recording. The per-texture binding of The Legend of Zelda: Breath of the Wild's engine (main+0xec2e64) always uses it,
//   passing the address of the handle field in the texture sampler object with a count of 1; command blocks are recorded once and submitted repeatedly, and the handle is written / replaced afterwards. The host used to copy the handles at recording (mostly 0),
//   so every texture of the deferred lighting composite became the dummy texture 0 and the whole 3D picture was black. The address is recorded and the replay layer reads it when it reads the record (bh_fresh)
NVN_NOPROTO(nvnCommandBufferBindTexturesDeferred, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, const NVNtextureHandle *h) { st = st_host(st);
    p_bind_handles *r = bind_handles(c, OP_BIND_TEX, st, first, n, (const uint64_t *)h);
    if (r && h) { r->defer = 1; r->src = (uint64_t)(uintptr_t)h; }
    if (CB(c)->tex_base || CB(c)->samp_base) ONCE_LOG("[RED] deferred-bound handles are read only at replay: the numbering base of this command buffer's bound descriptor pools is not added (pools %d / %d)", CB(c)->tex_base, CB(c)->samp_base);
}
NVN(nvnCommandBufferBindImage, void, NVNcommandBuffer *c, NVNshaderStage st, int i, NVNimageHandle h) { st = st_host(st); uint64_t v = (uint64_t)h; bind_handles(c, OP_BIND_IMG, st, i, 1, &v); }
NVN(nvnCommandBufferBindVertexBuffer, void, NVNcommandBuffer *c, int i, NVNbufferAddress a, size_t n) { NVNbufferRange r = {a, n}; h_nvnCommandBufferBindVertexBuffers(c, i, 1, &r); }

// Draws
NVN(nvnCommandBufferDrawElements, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, NVNindexType it, int n, NVNbufferAddress ib) { h_nvnCommandBufferDrawElementsInstanced(c, pr, it, n, ib, 0, 0, 1); }
NVN(nvnCommandBufferDrawElementsBaseVertex, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, NVNindexType it, int n, NVNbufferAddress ib, int bv) { h_nvnCommandBufferDrawElementsInstanced(c, pr, it, n, ib, bv, 0, 1); }
NVN(nvnCommandBufferDrawArraysInstanced, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, int first, int n, int bi, int inst) {
    rec_draw(c, (p_draw){pr, first, n, inst, bi});
}
// Indirect, as DrawArraysIndirect: the record at ind is read by the GPU (nvn_mtl.m draw)
// Indirect-count multi-draws (Monster Hunter Rise's GPU-driven scene: compute culling writes the draw records and a u32 count, then up to max draws are made from them; 1–49 per call,
// stride 20, the records and counts in its sparse virtual pool). Metal has no draw count read by the GPU, so each multi-draw becomes max ordinary indirect draws that carry the count's address
// and their index; the replay masks their records on the GPU, zeros past the count (nvn_mtl.m mdx_run). Both used to be empty, and the whole scene stayed undrawn
enum { MDI_MAX = 1024 };
NVN(nvnCommandBufferMultiDrawElementsIndirectCount, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, NVNindexType it, NVNbufferAddress ib, NVNbufferAddress ind, NVNbufferAddress cnt, int max, ptrdiff_t stride) {
    if (max > MDI_MAX) { ONCE_LOG("[RED] indirect-count multi-draw of up to %d draws: only the first %d are made", max, MDI_MAX); max = MDI_MAX; }
    for (int i = 0; i < max; i++) rec_draw_idx(c, (p_draw_idx){pr, it, 0, 0, 0, 1, ib, ind + (uint64_t)i * (uint64_t)stride, cnt, i});
}
NVN(nvnCommandBufferMultiDrawArraysIndirectCount, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, NVNbufferAddress ind, NVNbufferAddress cnt, int max, ptrdiff_t stride) {
    if (max > MDI_MAX) { ONCE_LOG("[RED] indirect-count multi-draw of up to %d draws: only the first %d are made", max, MDI_MAX); max = MDI_MAX; }
    for (int i = 0; i < max; i++) rec_draw(c, (p_draw){pr, 0, 0, 1, 0, ind + (uint64_t)i * (uint64_t)stride, cnt, i});
}
// Whether the host can read n bytes at a guest address: outside a sparse virtual pool always; inside one only where pages are mapped (nvn_mtl.m reads multi-draw counts there)
int nvn_host_readable(uint64_t a, uint64_t n) {
    int ok = 1; pthread_mutex_lock(&g_vp_mu);
    for (int k = 0; k < g_nvp; k++) { struct vpool *v = g_vp[k]; if (a < (uint64_t)(uintptr_t)v->base || a >= (uint64_t)(uintptr_t)v->base + v->len) continue;
        uint64_t o = a - (uint64_t)(uintptr_t)v->base; ok = 0;
        for (int r = 0; r < v->nrun; r++) if (o >= v->run[r].lo && o + n <= v->run[r].hi) { ok = 1; break; }
        break; }
    pthread_mutex_unlock(&g_vp_mu); return ok;
}
// Sparse virtual addresses alias physical pools, so guest ranges alone cannot
// prove copy independence.
int nvn_sparse_alias(uint64_t lo, uint64_t hi) {
    if (hi <= lo) return 0;
    int r = 0; pthread_mutex_lock(&g_vp_mu);
    for (int k = 0; k < g_nvp && !r; k++) { const struct vpool *v = g_vp[k];
        uint64_t base = (uint64_t)(uintptr_t)v->base;
        r = lo < base + v->len && base < hi;
    }
    pthread_mutex_unlock(&g_vp_mu); return r;
}
// Textures of a sparse pool, read through its page table (see the sparse pools above): whether a pool is one, a copy of a texture's bytes (unmapped pages read as 0; → bytes not mapped),
// whether its content may have changed since an upload (a mapping in its range newer than gen, or a physical page behind it written after the dirty-page stamp), and the write
// protection of the physical pages behind it (taken before an upload, as for any packaged texture)
static struct vpool *vpool_at(const void *pool, uint64_t a) {   // caller holds g_vp_mu
    struct vpool *v = pool ? ((const pool_t *)pool)->vp : NULL;
    return v && a >= (uint64_t)(uintptr_t)v->base && a < (uint64_t)(uintptr_t)v->base + v->size ? v : NULL;
}
// The physical memory behind the 64 KB tile at a: its first page when all 16 are mapped and consecutive, NULL otherwise (keys shared tiles, nxnative hle_nvn.c)
const uint8_t *nvn_sparse_tile_phys(const void *pool, uint64_t a) {
    pthread_mutex_lock(&g_vp_mu); struct vpool *v = vpool_at(pool, a); const uint8_t *p = NULL;
    uint64_t o = v ? a - (uint64_t)(uintptr_t)v->base : 0;
    if (v && !(o & 0xFFFF) && o + 65536 <= v->size) {
        const vpte *e = vpt_get(v, o); p = e ? e->phys : NULL;
        for (uint64_t k = 1; p && k < 16; k++) { const vpte *f = vpt_get(v, o + (k << 12)); if (!f || f->phys != p + (k << 12)) p = NULL; }
    }
    pthread_mutex_unlock(&g_vp_mu); return p;
}
// A sparse pool's mapping requests (nxnative hle_nvn.c): how many it has had (no lock, called at each look of a texture)
uint64_t nvn_sparse_log_seq(const void *pool) {
    struct vpool *v = pool ? ((const pool_t *)pool)->vp : NULL; if (!v) return 0;
    if (!v->log) { pthread_mutex_lock(&g_vp_mu); if (!v->log && !(v->log = calloc(VLOG_N, sizeof *v->log))) die("sparse pool change log allocation failed"); pthread_mutex_unlock(&g_vp_mu); }
    return atomic_load_explicit(&v->nlog, memory_order_acquire);
}
// The requests from number *seq on as guest address ranges (lo, hi pairs, at most max); *seq moves past them. -1 when the log no longer holds them or there are more than max: scan the texture
int nvn_sparse_log_read(const void *pool, uint64_t *seq, uint64_t (*out)[2], int max) {
    pthread_mutex_lock(&g_vp_mu); struct vpool *v = pool ? ((const pool_t *)pool)->vp : NULL; int n = 0;
    uint64_t end = v ? atomic_load_explicit(&v->nlog, memory_order_relaxed) : 0;
    if (!v || !v->log || end - *seq > VLOG_N || end - *seq > (uint64_t)max) n = -1;
    else for (uint64_t k = *seq; k < end; k++, n++) { const struct vlog_ent *e = &v->log[k % VLOG_N]; out[n][0] = (uint64_t)(uintptr_t)v->base + e->off; out[n][1] = out[n][0] + e->len; }
    *seq = end;
    pthread_mutex_unlock(&g_vp_mu); return n;
}
// One tile of a sparse texture: bit 0 every page of [a, a+n) has memory behind it, bit 1 a mapping in it is newer than gen. -1 outside a sparse pool
int nvn_sparse_tile_state(const void *pool, uint64_t a, uint64_t n, uint64_t gen) {
    pthread_mutex_lock(&g_vp_mu); struct vpool *v = vpool_at(pool, a);
    if (!v) { pthread_mutex_unlock(&g_vp_mu); return -1; }
    int all = 1, chg = 0;
    for (uint64_t o = (a - (uint64_t)(uintptr_t)v->base) & ~0xFFFull, end = a - (uint64_t)(uintptr_t)v->base + n; o < end; o += 0x1000) {
        const vpte *e = o < v->size ? vpt_get(v, o) : NULL;
        if (e && e->gen > gen) chg = 1;
        if (!e || !e->phys) all = 0;
    }
    pthread_mutex_unlock(&g_vp_mu); return all | chg << 1;
}
int nvn_sparse_pool(const void *pool) { return pool && ((const pool_t *)pool)->vp; }   // no lock: set at Initialize, cleared at the first mapping (an aliased pool) or Finalize; called per texture lookup
uint64_t nvn_sparse_gen(void) { return atomic_load(&g_vp_gen); }
// What a GPU copy put behind a sparse texture's pages stays here, by physical page, and is not written to guest memory; tile reads see it (NXRT_VK_SP_OVERLAY=0: written to memory, slower and
// more stalls). Known gap: a page the game later rewrites from the CPU reads as this copy until the game copies over it again. g_vp_mu held
typedef struct { const uint8_t *page; uint8_t *data; } sp_ov;
static sp_ov *g_ov; static uint32_t g_ov_cap, g_ov_n;
static uint8_t *ov_find(const uint8_t *page, int make) {
    if (!g_ov_cap) { if (!make) return NULL; g_ov_cap = 1 << 16; g_ov = calloc(g_ov_cap, sizeof *g_ov); }
    if (make && (g_ov_n + 1) * 2 > g_ov_cap) {
        sp_ov *o = g_ov; uint32_t oc = g_ov_cap; g_ov_cap *= 2; g_ov = calloc(g_ov_cap, sizeof *g_ov);
        for (uint32_t i = 0; i < oc; i++) if (o[i].page) { uint32_t k = (uint32_t)((uintptr_t)o[i].page >> 12) & (g_ov_cap - 1); while (g_ov[k].page) k = (k + 1) & (g_ov_cap - 1); g_ov[k] = o[i]; }
        free(o);
    }
    for (uint32_t k = (uint32_t)((uintptr_t)page >> 12) & (g_ov_cap - 1);; k = (k + 1) & (g_ov_cap - 1)) {
        if (g_ov[k].page == page) return g_ov[k].data;
        if (!g_ov[k].page) {
            if (!make) return NULL;
            g_ov[k].page = page; g_ov[k].data = malloc(0x1000); memcpy(g_ov[k].data, page, 0x1000); g_ov_n++; return g_ov[k].data;
        }
    }
}
static int ov_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("NXRT_VK_SP_OVERLAY"); on = !e || atoi(e) != 0; } return on; }
uint64_t nvn_sparse_copy(const void *pool, void *dst, uint64_t a, uint64_t n) {
    pthread_mutex_lock(&g_vp_mu); struct vpool *v = vpool_at(pool, a); uint64_t miss = 0;
    if (!v) { pthread_mutex_unlock(&g_vp_mu); memset(dst, 0, n); return n; }
    for (uint64_t p = 0, o = a - (uint64_t)(uintptr_t)v->base; p < n;) {
        uint64_t at = o + p, k = 0x1000 - (at & 0xFFF); if (k > n - p) k = n - p;
        const vpte *e = at < v->size ? vpt_get(v, at) : NULL;
        const uint8_t *ov = e && e->phys && ov_on() ? ov_find(e->phys, 0) : NULL;
        if (ov) memcpy((uint8_t *)dst + p, ov + (at & 0xFFF), k);
        else if (e && e->phys) memcpy((uint8_t *)dst + p, e->phys + (at & 0xFFF), k); else { memset((uint8_t *)dst + p, 0, k); miss += k; }
        p += k;
    }
    pthread_mutex_unlock(&g_vp_mu); return miss;
}
// The other way: bytes into what the table points at; unmapped pages drop them, as the hardware's write does
void nvn_sparse_poke(const void *pool, const void *src, uint64_t a, uint64_t n) {
    pthread_mutex_lock(&g_vp_mu); struct vpool *v = vpool_at(pool, a);
    if (v) for (uint64_t p = 0, o = a - (uint64_t)(uintptr_t)v->base; p < n;) {
        uint64_t at = o + p, k = 0x1000 - (at & 0xFFF); if (k > n - p) k = n - p;
        const vpte *e = at < v->size ? vpt_get(v, at) : NULL;
        if (e && e->phys && ov_on()) memcpy(ov_find(e->phys, 1) + (at & 0xFFF), (const uint8_t *)src + p, k);
        else if (e && e->phys) { dm_guest_write_prepare((void *)(e->phys + (at & 0xFFF)), k); memcpy((void *)(e->phys + (at & 0xFFF)), (const uint8_t *)src + p, k); }
        p += k;
    }
    pthread_mutex_unlock(&g_vp_mu);
}
// A GPU copy into a sparse texture writes the memory behind its tiles. The host reads tiles from that memory, so the bytes go there too, or the tile upload puts back what it held before
// (MKW streams the low levels of its terrain arrays this way: all-zero ASTC, which decodes as magenta, tinted every road and rock pink). The level is read through the page table,
// written with the copy's texels and put back. -1 when the copy cannot be done here
int nvn_tex_mirror_sparse(const tex_t *s, unsigned sl, unsigned ss, const NVNcopyRegion *sr, const tex_t *d, unsigned dl, unsigned ds, const NVNcopyRegion *dr) {
    int bw, bh, bpb = nvn_fmt_bpb(s->d.format, &bw, &bh), dbw, dbh, dbpb = nvn_fmt_bpb(d->d.format, &dbw, &dbh);
    if (!s->mem || !d->mem || !d->pool || !bpb || bpb != dbpb || bw != dbw || bh != dbh || s->d.target == NVN_TEXTURE_TARGET_3D || d->d.target == NVN_TEXTURE_TARGET_3D) return -1;
    if (sl >= s->d.levels || dl >= d->d.levels || sr->width != dr->width || sr->height != dr->height || sr->xoffset < 0 || sr->yoffset < 0) return -1;
    int nz = sr->depth ? sr->depth : 1;
    size_t row = (size_t)((sr->width + bw - 1) / bw) * (size_t)bpb, img = row * (size_t)((sr->height + bh - 1) / bh);
    uint64_t lo = nvn_tex_level_offset(&d->d, dl), hi = dl + 1 < d->d.levels ? nvn_tex_level_offset(&d->d, dl + 1) : d->L.layer_size;
    if (hi <= lo || hi > d->L.layer_size) return -1;
    uint8_t *lin = malloc(img * (size_t)nz), *lev = malloc(hi - lo), *was = malloc(hi - lo); int ok = lin && lev && was;
    NVNcopyRegion s2 = *sr; s2.depth = nz;
    ok = ok && !nvn_texio_read(s, sl, ss, &s2, lin, row, img);
    for (int z = 0; z < nz && ok; z++) {
        uint64_t lay = (uint64_t)(ds + (unsigned)dr->zoffset + (unsigned)z) * d->L.layer_size, at = (uint64_t)(uintptr_t)d->mem + lay + lo;
        nvn_sparse_copy(d->pool, lev, at, hi - lo); memcpy(was, lev, hi - lo);
        tex_t f = *d; f.mem = lev - lay - lo;
        NVNcopyRegion one = *dr; one.depth = 1; one.zoffset = dr->zoffset + z;
        ok = !nvn_texio_write(&f, dl, ds, &one, lin + (size_t)z * img, row, img);
        // only the pages the copy changed go back: the rest may have been written since the read above, by the CPU or the GPU, and writing them back would undo that
        for (uint64_t o = 0, k; ok && o < hi - lo; o += k) {
            k = 0x1000 - ((at + o) & 0xFFF); if (k > hi - lo - o) k = hi - lo - o;
            if (memcmp(lev + o, was + o, k)) nvn_sparse_poke(d->pool, lev + o, at + o, k);
        }
    }
    free(lin); free(lev); free(was);
    return ok ? 0 : -1;
}
// Runs of consecutive physical pages behind [a, a+n): fn on each (nonzero stops the walk and is returned), or 1 at the first page mapped after gen (gen ~0 = do not look)
static int vpt_walk(const void *pool, uint64_t a, uint64_t n, uint64_t gen, int (*fn)(const uint8_t *, uint64_t, uint64_t), uint64_t arg) {
    pthread_mutex_lock(&g_vp_mu); struct vpool *v = vpool_at(pool, a); int r = 0;
    if (!v) { pthread_mutex_unlock(&g_vp_mu); return 1; }
    const uint8_t *rs = NULL, *re = NULL; uint64_t o = a - (uint64_t)(uintptr_t)v->base;
    for (uint64_t at = o & ~0xFFFull; at < o + n && at < v->size && !r; at += 0x1000) {
        const vpte *e = vpt_get(v, at);
        if (!e) { at |= (2u << 20) - 1; at -= 0xFFF; continue; }   // a leaf never mapped: skip its 2 MB
        if (gen != ~0ull && e->gen > gen) { r = 1; break; }
        if (!e->phys) continue;
        if (rs && e->phys == re) { re += 0x1000; continue; }
        if (rs) r = fn(rs, (uint64_t)(re - rs), arg);
        rs = e->phys; re = rs + 0x1000;
    }
    if (!r && rs) r = fn(rs, (uint64_t)(re - rs), arg);
    pthread_mutex_unlock(&g_vp_mu); return r;
}
static int run_written(const uint8_t *p, uint64_t n, uint64_t stamp) { return nvn_dirty_since(p, n, stamp); }
static int run_protect(const uint8_t *p, uint64_t n, uint64_t unused) { (void)unused; nvn_dirty_protect(p, n); return 0; }
int nvn_sparse_since(const void *pool, uint64_t a, uint64_t n, uint64_t gen, uint64_t stamp) { return vpt_walk(pool, a, n, gen, run_written, stamp); }
int nvn_sparse_mapped(const void *pool, uint64_t a, uint64_t n) {   // every page of [a, a+n) has memory behind it
    pthread_mutex_lock(&g_vp_mu); struct vpool *v = vpool_at(pool, a); int ok = v != NULL;
    for (uint64_t o = v ? (a - (uint64_t)(uintptr_t)v->base) & ~0xFFFull : 0, end = v ? a - (uint64_t)(uintptr_t)v->base + n : 0; ok && o < end; o += 0x1000) { const vpte *e = o < v->size ? vpt_get(v, o) : NULL; ok = e && e->phys; }
    pthread_mutex_unlock(&g_vp_mu); return ok;
}
void nvn_sparse_protect(const void *pool, uint64_t a, uint64_t n) { vpt_walk(pool, a, n, ~0ull, run_protect, 0); }
NVN(nvnCommandBufferDrawElementsIndirect, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, NVNindexType it, NVNbufferAddress ib, NVNbufferAddress ind) {
    rec_draw_idx(c, (p_draw_idx){pr, it, 0, 0, 0, 1, ib, ind});
}

// Viewport / scissor / depth range: single and multiple forms
NVN(nvnCommandBufferSetViewport, void, NVNcommandBuffer *c, int x, int y, int w, int h) { float v[4] = {(float)x, (float)y, (float)w, (float)h}; h_nvnCommandBufferSetViewports(c, 0, 1, v); }
NVN(nvnCommandBufferSetScissors, void, NVNcommandBuffer *c, int first, int n, const int *r) {
    if (first == 0 && n >= 1) h_nvnCommandBufferSetScissor(c, r[0], r[1], r[2], r[3]);
    if (first + n > 1) ONCE_LOG("SetScissors first=%d n=%d: the host uses only index 0", first, n);
}
NVN(nvnCommandBufferSetDepthRanges, void, NVNcommandBuffer *c, int first, int n, const float *r) {
    if (first == 0 && n >= 1) h_nvnCommandBufferSetDepthRange(c, r[0], r[1]);
    if (first + n > 1) ONCE_LOG("SetDepthRanges first=%d n=%d: the host uses only index 0", first, n);
}

// GPU value write: ReportValue = write a 32-bit value to the address when this point is reached (the replay layer's event write, applied on completion)
NVN(nvnCommandBufferReportValue, void, NVNcommandBuffer *c, uint32_t v, NVNbufferAddress a) {
    uint8_t *p = cmd_alloc(CB(c), 24); put_hdr(p, OP_SIGNAL_EVENT, 24);
    volatile uint32_t *val = (volatile uint32_t *)(uintptr_t)a; uint32_t mm = NVN_EVENT_SIGNAL_MODE_WRITE;
    memcpy(p + 8, &val, 8); memcpy(p + 16, &mm, 4); memcpy(p + 20, &v, 4);
}
// Clear a buffer to a 32-bit value: one OP_CLEAR_BUF record, filled in order within the submission at replay (fillBuffer when all bytes match, otherwise copied from a pattern buffer)
NVN(nvnCommandBufferClearBuffer, void, NVNcommandBuffer *c, NVNbufferAddress a, size_t n, uint32_t v) { *REC(c, OP_CLEAR_BUF, p_clear_buf) = (p_clear_buf){a, n, v, 0}; }

// State the replay layer does not have yet: logged once (conditional rendering is treated as "always draw")
// Render enable (profile render_cond, see OP_RENDER_COND in nvn_mtl.m): recorded and evaluated at replay; without the key everything draws, as before
NVN(nvnCommandBufferSetRenderEnable, void, NVNcommandBuffer *c, uint32_t en) {
    if (DM_GAME_RENDER_COND) *REC(c, OP_RENDER_COND, p_render_cond) = (p_render_cond){(en & 0xff) ? 2 : 3, 0, 0};
    else if (!(en & 0xff)) ONCE_LOG("[RED] SetRenderEnable(false): the host draws anyway for now");
}
NVN(nvnCommandBufferSetRenderEnableConditional, void, NVNcommandBuffer *c, NVNconditionalRenderMode m, NVNbufferAddress a) {
    if (DM_GAME_RENDER_COND) *REC(c, OP_RENDER_COND, p_render_cond) = (p_render_cond){(int)m & 1, 0, a};
    else ONCE_LOG("[RED] SetRenderEnableConditional: the host draws anyway for now");
}
// Fire Emblem: Three Houses turns depth clamp on for its scenes; ignored, the ground right under its close-up battle cameras was clipped away and the lighting pass painted the sky's fog colour there
NVN(nvnCommandBufferSetDepthClamp, void, NVNcommandBuffer *c, uint32_t en) { REC(c, OP_DEPTH_CLAMP, p_depth_clamp)->on = (en & 0xff) != 0; }
// Absolum (FNA3D's NVN driver) keeps the lower-left window origin and sets viewport 0's swizzle every frame (NEGATIVE_Y: clip y negated, how a GL-heritage renderer flips its picture).
// Ignored, its whole picture came out upside down. The replay applies it to the viewport (draw in nvn_mtl.m)
NVN(nvnCommandBufferSetViewportSwizzles, void, NVNcommandBuffer *c, int first, int n, const NVNviewportSwizzle *s) {
    if (n < 0 || first < 0 || first + n > 16) { LOG("[RED] SetViewportSwizzles first=%d n=%d", first, n); return; }
    p_vp_swizzle *r = REC(c, OP_VP_SWIZZLE, p_vp_swizzle); r->first = first; r->n = n;
    for (int i = 0; i < n; i++) for (int k = 0; k < 4; k++) r->sw[i][k] = (uint8_t)((s[4 * i + k] ^ 2 * k) & 7);
}
NVN(nvnCommandBufferSetAlphaRef, void, NVNcommandBuffer *c, float r) { REC(c, OP_ALPHA_REF, p_alpha_ref)->ref = r; }
NVN(nvnCommandBufferSetLineWidth, void, NVNcommandBuffer *c, float w) { (void)c; if (w != 1.0f) ONCE_LOG("SetLineWidth %.2f: Metal has only 1-pixel lines", w); }
NVN(nvnCommandBufferSetPointSize, void, NVNcommandBuffer *c, float s) { (void)c; (void)s; ONCE_LOG("SetPointSize: the point size comes from the shader"); }
NVN(nvnCommandBufferSetPrimitiveRestart, void, NVNcommandBuffer *c, uint32_t en, int idx) { *REC(c, OP_PRIM_RESTART, p_prim_restart) = (p_prim_restart){(en & 0xff) != 0, (uint32_t)idx}; }
// Rasterizer discard: draws run their vertex stage only (with transform feedback, the capture); recorded, and at replay such a draw rasterizes nothing (nvn_mtl.m)
NVN(nvnCommandBufferSetRasterizerDiscard, void, NVNcommandBuffer *c, uint32_t en) { REC(c, OP_RAST_DISCARD, p_rast_discard)->on = (en & 0xff) != 0; }
NVN(nvnCommandBufferSetSampleMask, void, NVNcommandBuffer *c, int m) { (void)c; if ((m & 0xffff) != 0xffff) ONCE_LOG("SetSampleMask %#x: ignored by the host for now", m); }
NVN(nvnCommandBufferSetConservativeRasterEnable, void, NVNcommandBuffer *c, uint32_t en) { (void)c; if (en & 0xff) ONCE_LOG("SetConservativeRasterEnable on: ignored by the host for now"); }
NVN(nvnCommandBufferSetTiledCacheAction, void, NVNcommandBuffer *c, NVNtiledCacheAction a) { (void)c; (void)a; }        // Tegra tiled cache hint, no Metal equivalent
// A bound target's contents may be discarded: they are undefined afterwards, so keeping them is correct (The Great Ace Attorney Chronicles discards colour target 0)
NVN(nvnCommandBufferDiscardColor, void, NVNcommandBuffer *c, int i) { (void)c; (void)i; }
NVN(nvnCommandBufferDiscardDepthStencil, void, NVNcommandBuffer *c) { (void)c; }
NVN(nvnCommandBufferSetTiledCacheTileSize, void, NVNcommandBuffer *c, int w, int h) { (void)c; (void)w; (void)h; }
NVN_NOPROTO(nvnCommandBufferSetColorReductionEnable, void, NVNcommandBuffer *c, uint32_t en) { (void)c; (void)en; }   // color compression hints
NVN_NOPROTO(nvnCommandBufferSetColorReductionThresholds, void, NVNcommandBuffer *c, int cls, const int *t) { (void)c; (void)cls; (void)t; }
// Resolve a multisample texture into a single-sample one (whole texture): recorded as a texture copy with DM_COPY_RESOLVE; replay uses Metal's multisample resolve (copy_t2t in nvn_mtl.m)
static void rec_downsample(NVNcommandBuffer *c, const NVNtexture *s, const NVNtextureView *sv, const NVNtexture *d, const NVNtextureView *dv, uint64_t ra) {
    p_copy_t2t *o = REC_SNAP(c, OP_COPY_T2T, p_copy_t2t, 2);
    o->src = tex_snap(o, SNAP_OFF(p_copy_t2t), 0, s, ra); o->dst = tex_snap(o, SNAP_OFF(p_copy_t2t), 1, d, ra);
    copy_view(&o->sv, &o->has_sv, sv); copy_view(&o->dv, &o->has_dv, dv);
    if (o->dst) nvn_pool_gpu_written(o->dst->pool);
    if (o->src) o->sr = (NVNcopyRegion){0, 0, 0, (int)o->src->d.width, (int)o->src->d.height, 1};
    if (o->dst) o->dr = (NVNcopyRegion){0, 0, 0, (int)o->dst->d.width, (int)o->dst->d.height, 1};
    o->flags = DM_COPY_RESOLVE;
}
NVN(nvnCommandBufferDownsample, void, NVNcommandBuffer *c, const NVNtexture *s, const NVNtexture *d) { rec_downsample(c, s, NULL, d, NULL, (uint64_t)(uintptr_t)DM_GUEST_RA()); }
NVN(nvnCommandBufferDownsampleTextureView, void, NVNcommandBuffer *c, const NVNtexture *s, const NVNtextureView *sv, const NVNtexture *d, const NVNtextureView *dv) {
    rec_downsample(c, s, sv, d, dv, (uint64_t)(uintptr_t)DM_GUEST_RA());
}

// CPU texture write (NVN WriteTexels): the region goes straight into the storage's block-linear layout (nvn_texio, through s2tex; nxnative decoded and re-encoded with nvn_deswizzle.c,
// whose Switch 2 order is not ported). The dirty tracker's protection is lifted first and the backend is told the texture is CPU-written, so it uploads it from the storage
// at its next use and again when its pages change
static void write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, size_t row, size_t img) {
    if (!t->mem) { LOG("[RED] WriteTexels: texture has no storage"); return; }
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (!bpb) { ONCE_LOG("[RED] WriteTexels: unknown format %#x", t->d.format); return; }
    unsigned level = v && v->set_levels ? (unsigned)v->base_level : 0, layer0 = v && v->set_layers ? (unsigned)v->min_layer : 0;
    { static int on = -1; static long nl; if (on < 0) on = getenv("SWR_LOG_WRITETEXELS") != NULL;   // debugging: every CPU texel upload, where it lands and in which layout
      if (on && nl++ < 3000) LOG("WriteTexels: ser=%llu %ux%ux%u fmt %#x flags %#x stride %u levels %u layers %u tileH %u, mem %p size %#llx, level %u layer %u region %d,%d,%d %dx%dx%d, src row %zu",
          (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->d.format, t->d.flags, t->d.stride, t->d.levels, t->L.layers, t->L.tileH, (void *)t->mem,
          (unsigned long long)t->L.size, level, layer0, r->xoffset, r->yoffset, r->zoffset, r->width, r->height, r->depth, row); }
    dm_guest_write_prepare(t->mem, t->L.size);
    if (nvn_texio_write(t, level, layer0, r, src, row, img)) {
        ONCE_LOG("[RED] WriteTexels: region %d,%d,%d %dx%dx%d of level %u (layer %u) does not fit texture %ux%ux%u format %#x target %d: not written", r->xoffset, r->yoffset, r->zoffset,
                 r->width, r->height, r->depth, level, layer0, t->d.width, t->d.height, t->d.depth, t->d.format, t->d.target);
        return;
    }
    gfx_tex_cpu_written(t);
}
NVN(nvnTextureWriteTexels, void, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, const void *src) { write_texels(CTX(t), (const view_t *)v, r, src, 0, 0); }
// with the source's row and image strides in bytes (0 = tightly packed, as WriteTexels)
NVN(nvnTextureWriteTexelsStrided, void, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, const void *src, ptrdiff_t row, ptrdiff_t img) {
    write_texels(CTX(t), (const view_t *)v, r, src, (size_t)row, (size_t)img);
}
// GPU copies into a texture the host uploads from its storage (tex_mtl_'s packaged textures: a packaged layout, CPU-written, or registered by tex_flush_upload / tex_dirty_upload and not
// resolved yet): the hardware's copy writes the storage, so the host writes the copied bytes there as well (copy_b2t / copy_t2t in nvn_mtl.m), and an upload from the storage later
// carries them. Yoshi's Crafted World's texture streamer re-creates a texture with more levels, copies the resident levels from the old texture on the GPU (Unreal's
// RHIAsyncReallocateTexture2D: nvnCommandBufferCopyTextureToTexture, a level at a time) and writes the new, finer levels from the CPU; uploaded from the storage, the copied levels held
// what the pool had there before (another texture's data, zeros), and the Yoshis' felt sampled speckles from the moment a level streamed in. −1 when the copy cannot be mirrored (3D,
// a format mismatch, a region out of range): the GPU copy alone then stands until the next upload
static int mirror_prepare(const tex_t *t, unsigned level, unsigned layer) {   // unprotect (and mark written) the storage of one level of one layer
    uint64_t lo = nvn_tex_level_offset(&t->d, level), hi = level + 1 < t->d.levels ? nvn_tex_level_offset(&t->d, level + 1) : t->L.layer_size;
    if (hi <= lo || hi > t->L.layer_size || (uint64_t)layer * t->L.layer_size + hi > t->L.size) return -1;
    dm_guest_write_prepare(t->mem + (uint64_t)layer * t->L.layer_size + lo, hi - lo);
    return 0;
}
int nvn_tex_mirror_b2t(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, const uint8_t *src, size_t row, size_t img) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (!t->mem || !bpb || t->d.target == NVN_TEXTURE_TARGET_3D || level >= t->d.levels) return -1;
    int nz = r->depth ? r->depth : 1;
    for (int z = 0; z < nz; z++) if (mirror_prepare(t, level, layer0 + (unsigned)(r->zoffset + z))) return -1;
    return nvn_texio_write(t, level, layer0, r, src, row, img);
}
int nvn_tex_mirror_t2t(const tex_t *s, unsigned sl, unsigned ss, const NVNcopyRegion *sr, const tex_t *d, unsigned dl, unsigned ds, const NVNcopyRegion *dr) {
    int bw, bh, bpb = nvn_fmt_bpb(s->d.format, &bw, &bh), dbw, dbh, dbpb = nvn_fmt_bpb(d->d.format, &dbw, &dbh);
    if (!s->mem || !d->mem || !bpb || bpb != dbpb || bw != dbw || bh != dbh || s->d.target == NVN_TEXTURE_TARGET_3D || d->d.target == NVN_TEXTURE_TARGET_3D) return -1;
    if (sl >= s->d.levels || dl >= d->d.levels || sr->width != dr->width || sr->height != dr->height || sr->xoffset < 0 || sr->yoffset < 0) return -1;
    int nz = sr->depth ? sr->depth : 1;
    size_t row = (size_t)((sr->width + bw - 1) / bw) * (size_t)bpb, img = row * (size_t)((sr->height + bh - 1) / bh);
    uint8_t *lin = malloc(img * (size_t)nz); if (!lin) die("texture copy mirror allocation failed");
    NVNcopyRegion s2 = *sr, d2 = *dr; s2.depth = d2.depth = nz;
    int ok = !nvn_texio_read(s, sl, ss, &s2, lin, row, img);
    for (int z = 0; z < nz && ok; z++) ok = !mirror_prepare(d, dl, ds + (unsigned)(dr->zoffset + z));
    ok = ok && !nvn_texio_write(d, dl, ds, &d2, lin, row, img);
    free(lin);
    return ok ? 0 : -1;
}
// The movie decoder's native-texture output (DecoderMode 2, hle_movie.c): a whole frame into level 0 of the client texture, as WriteTexels does; returns the texture format, or −1 when the object is not a texture
int nvn_host_write_texture(const void *nvntex, const void *src, int w, int h, size_t row, int *tw, int *th) {
    const tex_t *t = nvntex;
    MEMORY_BASIC_INFORMATION mi;   // the value the client gives is not necessarily a pointer: look before reading
    if (!t || (uintptr_t)t & 7 || !VirtualQuery(t, &mi, sizeof mi) || mi.State != MEM_COMMIT || (mi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) ||
        (uintptr_t)t + sizeof *t > (uintptr_t)mi.BaseAddress + mi.RegionSize) return -1;
    if (t->magic != MAGIC('T', 'E', 'X', 0)) return -1;
    if (tw) *tw = (int)t->d.width;
    if (th) *th = (int)t->d.height;
    if (src) { NVNcopyRegion r = {0, 0, 0, w, h, 1}; write_texels(t, NULL, &r, src, row, 0); }
    return (int)t->d.format;
}
// work the host finished on the CPU: the sync object counts as signalled at once (fence 0 is always complete)
int nvn_host_signal_sync(void *nvnsync) {
    sync_t *o = nvnsync;
    if (!o || o->magic != MAGIC('S', 'Y', 'N', 'C')) return -1;
    o->fence = 0; o->fenced = 1; return 0;
}
NVN(nvnTextureFlushTexels, void, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r) { (void)t; (void)v; (void)r; }   // unified memory + dirty page tracking: nothing to flush

// ---------- NVN2's entries MKW calls (nxnative hle_nvn.c, MIT, (c) seedds) ----------
// Programs from a builder: the stages are {data, control} pairs as in NVN, so initializing from the builder is nvnProgramInitialize + nvnProgramSetShaders. The builder keeps the caller's array
typedef struct { NVNdevice *dev; const NVNshaderData *sd; int n; } prog_builder_t;
NVN_NOPROTO(nvnProgramBuilderSetDefaults, void, void *b) { memset(b, 0, sizeof(prog_builder_t)); }
NVN_NOPROTO(nvnProgramBuilderSetDevice, void, void *b, NVNdevice *d) { ((prog_builder_t *)b)->dev = d; }
NVN_NOPROTO(nvnProgramBuilderSetShaders, void, void *b, int n, const NVNshaderData *sd) { ((prog_builder_t *)b)->n = n; ((prog_builder_t *)b)->sd = sd; }
NVN_NOPROTO(nvnProgramInitializeFromBuilder, NVNboolean, NVNprogram *p, const void *b) {
    const prog_builder_t *o = (const prog_builder_t *)b;
    return h_nvnProgramInitialize(p, o->dev) && (!o->n || h_nvnProgramSetShaders(p, o->n, o->sd));
}
// A queue tied to a primary queue: every queue's submits run in order on the one device queue, so the tie needs no work
NVN_NOPROTO(nvnQueueBuilderSetPrimaryQueue, void, NVNqueueBuilder *b, NVNqueue *primary) { (void)b; (void)primary; }
NVN_NOPROTO(nvnCommandBufferSetZCullZF32CompressionEnable, void, NVNcommandBuffer *c, uint32_t on) { (void)c; (void)on; }
// How the GPU's front end runs commands, not what they produce
NVN_NOPROTO(nvnCommandBufferSetRecordingMode, void, NVNcommandBuffer *c, int mode) { (void)c; (void)mode; }
NVN_NOPROTO(nvnCommandBufferSetGeometryProcessingMode, void, NVNcommandBuffer *c, int mode) { (void)c; (void)mode; }
NVN_NOPROTO(nvnCommandBufferSetComputeOverlapMode, void, NVNcommandBuffer *c, int mode) { (void)c; (void)mode; }
NVN_NOPROTO(nvnCommandBufferSetStencilCullCriteria, void, NVNcommandBuffer *c, int func, int ref) { (void)c; (void)func; (void)ref; }
// NVN2 push constants: (command buffer, a word seen 0, byte offset, size, data). A constant block of the graphics stages or of compute, read as c[24] (DMSH_PUSH_BANK)
static void push_const(NVNcommandBuffer *c, int compute, uint64_t w, uint64_t off, uint64_t n, const void *data) {
    if (w) { static int once; if (!once++) LOG("[RED] push constants with the word before the offset %#llx: its meaning is not known", (unsigned long long)w); }
    if (off + n > DMSH_PUSH_SIZE || !data) { static int k; if (k++ < 8) LOG("[RED] push constants at %#llx + %#llx: beyond the host's %#x-byte block, dropped", (unsigned long long)off, (unsigned long long)n, DMSH_PUSH_SIZE); return; }
    p_push_const *r = RECN(c, OP_PUSH_CONST, p_push_const, data, n); r->compute = (uint16_t)compute; r->offset = (uint16_t)off; r->size = (uint32_t)n; memcpy(r->data, data, n);
}
NVN_NOPROTO(nvnCommandBufferPushGraphicsConstants, void, NVNcommandBuffer *c, uint64_t w, uint64_t off, uint64_t n, const void *data) { push_const(c, 0, w, off, n, data); }
NVN_NOPROTO(nvnCommandBufferPushComputeConstants, void, NVNcommandBuffer *c, uint64_t w, uint64_t off, uint64_t n, const void *data) { push_const(c, 1, w, off, n, data); }
// Variable rate shading: accepted and ignored, every pixel is shaded (nxnative docs/reference/rendering.md)
NVN_NOPROTO(nvnCommandBufferBindShadingRateImage, void, NVNcommandBuffer *c, const void *view, const void *tex) { (void)c; (void)view; (void)tex; }
NVN_NOPROTO(nvnCommandBufferShadingRateImagePalette, void, NVNcommandBuffer *c, int first, int n, const void *entries) { (void)c; (void)first; (void)n; (void)entries; }
NVN_NOPROTO(nvnCommandBufferSetShadingRatePaletteMode, void, NVNcommandBuffer *c, int mode) { (void)c; (void)mode; }

#include "hle_nvn.adapters.inc"
