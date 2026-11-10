// NVIDIA Nsight Aftermath GPU crash dumps (NXRT_VK_AFTERMATH=1), for faults and hangs the checkpoints cannot name: which shader, which stage, which instruction.
// The SDK (external/aftermath, NVIDIA's licence) is not linked: its DLL is loaded at run time from beside the exe, so without it this is all off.
// Files go to <exe folder>/aftermath/<run time>/: the dumps (*.nv-gpudmp), the driver's shader debug info (*.nvdbg) and shaders.tsv, which maps
// each module's Aftermath hash to our shader key. Every module's SPIR-V goes to aftermath/spv/<hash>.spv (as given to the driver) and
// <hash>.src.spv (before the line markers below): tools/aftermath_read.py decodes a dump with them.
// Line markers (NXRT_VK_AFTERMATH_LINES=0 leaves them out): an OpLine before each instruction whose line is that instruction's word offset in the
// .src.spv, so the decoder can name the SPIR-V instruction behind the faulting GPU instruction.
#include "vk_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if __has_include("GFSDK_Aftermath_GpuCrashDump.h")
#include "GFSDK_Aftermath_Defines.h"
#include "GFSDK_Aftermath_GpuCrashDump.h"
#include "GFSDK_Aftermath_GpuCrashDumpDecoding.h"
#define HAVE_AM 1
#else
#define HAVE_AM 0
#endif

static int g_on;              // 1 = crash dumps enabled before the device
static int g_lines = 1;
static char g_dir[1024], g_spv[1024];
static SRWLOCK g_lk = SRWLOCK_INIT;
static FILE *g_tsv;
static _Atomic int g_dumps, g_dbginfo;

int vk_am_on(void) { return g_on; }
const char *vk_am_dir(void) { return g_on ? g_dir : NULL; }

#if HAVE_AM
static PFN_GFSDK_Aftermath_EnableGpuCrashDumps p_enable;
static PFN_GFSDK_Aftermath_GetCrashDumpStatus p_status;
typedef GFSDK_Aftermath_Result(GFSDK_AFTERMATH_CALL *PFN_hash_spirv)(GFSDK_Aftermath_Version, const GFSDK_Aftermath_SpirvCode *, GFSDK_Aftermath_ShaderBinaryHash *);
typedef GFSDK_Aftermath_Result(GFSDK_AFTERMATH_CALL *PFN_dbg_id)(GFSDK_Aftermath_Version, const void *, const uint32_t, GFSDK_Aftermath_ShaderDebugInfoIdentifier *);
static PFN_hash_spirv p_hash;
static PFN_dbg_id p_dbgid;

static void write_file(const char *path, const void *p, size_t n) {
    FILE *f = fopen(path, "wb"); if (!f) return;
    fwrite(p, 1, n, f); fclose(f);
}

static void GFSDK_AFTERMATH_CALL on_dump(const void *p, const uint32_t n, void *u) {
    (void)u; char path[1200]; int k = atomic_fetch_add(&g_dumps, 1);
    snprintf(path, sizeof path, "%s\\dump%d.nv-gpudmp", g_dir, k);
    write_file(path, p, n);
    nxrt_logf("[vk] [RED] Aftermath: GPU crash dump written, %u bytes: %s", n, path);
}
static void GFSDK_AFTERMATH_CALL on_dbginfo(const void *p, const uint32_t n, void *u) {
    (void)u; GFSDK_Aftermath_ShaderDebugInfoIdentifier id = {{0, 0}};
    if (p_dbgid && p_dbgid(GFSDK_Aftermath_Version_API, p, n, &id) != GFSDK_Aftermath_Result_Success) return;
    char path[1200]; snprintf(path, sizeof path, "%s\\shader-%016llx%016llx.nvdbg", g_dir, (unsigned long long)id.id[0], (unsigned long long)id.id[1]);
    write_file(path, p, n); atomic_fetch_add(&g_dbginfo, 1);
}
static void GFSDK_AFTERMATH_CALL on_desc(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription add, void *u) {
    (void)u; add(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, "Switch2Recomp");
    add(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationVersion, "native-gpu");
}
// Our checkpoints carry their number as the marker: the dump gets the text vk_lost would print for it
static void GFSDK_AFTERMATH_CALL on_marker(const void *p, const uint32_t n, void *u, PFN_GFSDK_Aftermath_ResolveMarker resolve) {
    (void)n; (void)u; static __thread char t[256];
    const char *vk_ck_describe(uint64_t s, char *out, size_t n);
    vk_ck_describe((uint64_t)(uintptr_t)p, t, sizeof t);
    resolve(t, (uint32_t)strlen(t) + 1);
}
#endif

// Before vkCreateDevice: loads the DLL and turns dumps on. 0 = on.
int vk_am_init(void) {
    const char *e = getenv("NXRT_VK_AFTERMATH");
    if (!e || !*e || *e == '0') return -1;
#if !HAVE_AM
    VKRED("NXRT_VK_AFTERMATH is set, but this runtime was built without the Aftermath SDK headers (external/aftermath)");
    return -1;
#else
    const char *l = getenv("NXRT_VK_AFTERMATH_LINES"); if (l && *l == '0') g_lines = 0;
    char exe[MAX_PATH]; GetModuleFileNameA(NULL, exe, sizeof exe);
    char *s = strrchr(exe, '\\'); if (s) *s = 0;
    char dll[MAX_PATH + 64]; snprintf(dll, sizeof dll, "%s\\GFSDK_Aftermath_Lib.x64.dll", exe);
    HMODULE h = LoadLibraryA(dll);
    if (!h) { VKRED("Aftermath: %s not found: crash dumps off", dll); return -1; }
    p_enable = (PFN_GFSDK_Aftermath_EnableGpuCrashDumps)(void *)GetProcAddress(h, "GFSDK_Aftermath_EnableGpuCrashDumps");
    p_status = (PFN_GFSDK_Aftermath_GetCrashDumpStatus)(void *)GetProcAddress(h, "GFSDK_Aftermath_GetCrashDumpStatus");
    p_hash = (PFN_hash_spirv)(void *)GetProcAddress(h, "GFSDK_Aftermath_GetShaderHashSpirv");
    p_dbgid = (PFN_dbg_id)(void *)GetProcAddress(h, "GFSDK_Aftermath_GetShaderDebugInfoIdentifier");
    if (!p_enable || !p_hash) { VKRED("Aftermath: the DLL lacks its entry points: crash dumps off"); return -1; }
    SYSTEMTIME st; GetLocalTime(&st);
    snprintf(g_dir, sizeof g_dir, "%s\\aftermath", exe); CreateDirectoryA(g_dir, NULL);
    snprintf(g_spv, sizeof g_spv, "%s\\spv", g_dir); CreateDirectoryA(g_spv, NULL);
    size_t n = strlen(g_dir);
    snprintf(g_dir + n, sizeof g_dir - n, "\\%04d%02d%02d_%02d%02d%02d", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
    CreateDirectoryA(g_dir, NULL);
    char tsv[1200]; snprintf(tsv, sizeof tsv, "%s\\shaders.tsv", g_dir);
    g_tsv = fopen(tsv, "w");
    if (g_tsv) fprintf(g_tsv, "hash\tkey\twhat\twords\n");
    GFSDK_Aftermath_Result r = p_enable(GFSDK_Aftermath_Version_API, GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan, GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks,
                                        on_dump, on_dbginfo, on_desc, on_marker, NULL);
    if (r != GFSDK_Aftermath_Result_Success) { VKRED("Aftermath: GFSDK_Aftermath_EnableGpuCrashDumps failed (%#x): crash dumps off", (unsigned)r); return -1; }
    g_on = 1;
    VKLOG("Aftermath: GPU crash dumps on, into %s%s", g_dir, g_lines ? "; SPIR-V line markers on" : "");
    return 0;
#endif
}

// After a loss: waits for the dump to be written (the driver may take minutes on a hang). Returns 1 when a dump was written.
int vk_am_wait(int max_s) {
#if HAVE_AM
    if (!g_on || !p_status) return 0;
    uint64_t t0 = now_ns(); GFSDK_Aftermath_CrashDump_Status st = GFSDK_Aftermath_CrashDump_Status_Unknown, last = 99;
    for (;;) {
        p_status(&st);
        if (st != last) { VKLOG("Aftermath: dump status %u after %.1f s (0 not started, 1 collecting, 2 failed, 3 callback, 4 finished)", (unsigned)st, (now_ns() - t0) / 1e9); last = st; }
        if (st == GFSDK_Aftermath_CrashDump_Status_Finished || st == GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed) break;
        if ((now_ns() - t0) / 1000000000ull >= (uint64_t)max_s) break;
        Sleep(250);
    }
    if (g_tsv) fflush(g_tsv);
    VKLOG("Aftermath: %d dumps, %d shader debug info files in %s", atomic_load(&g_dumps), atomic_load(&g_dbginfo), g_dir);
    return atomic_load(&g_dumps) > 0;
#else
    (void)max_s; return 0;
#endif
}

// Device creation: the diagnostics config to chain, NULL when off or not supported (the caller adds VK_NV_device_diagnostics_config)
static VkDeviceDiagnosticsConfigCreateInfoNV g_dcfg;
void *vk_am_device_chain(void *next) {
    if (!g_on) return next;
    g_dcfg = (VkDeviceDiagnosticsConfigCreateInfoNV){VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV, next,
        VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV | VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV |
        VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV | VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_ERROR_REPORTING_BIT_NV};
    return &g_dcfg;
}

// ---------- SPIR-V line markers ----------
// An OpString "s2r.spv" in the debug section, and an OpLine (file, word offset, 1) before each instruction of a function body where one is allowed.
// None before OpLabel, OpPhi, OpVariable, the function's own instructions, merge instructions, or the branch a merge instruction must precede.
static uint32_t *add_lines(const uint32_t *w, size_t n, size_t *out_n) {
    if (n < 5 || w[0] != 0x07230203) return NULL;
    for (size_t i = 5; i < n;) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16; if (!wc) return NULL; if (op == 8 || op == 317) return NULL; i += wc; }   // has lines already
    size_t cap = n * 5 + 16, k = 0; uint32_t *o = malloc(cap * 4); if (!o) return NULL;
    memcpy(o, w, 20); k = 5;
    uint32_t file = w[3]; o[3] = w[3] + 1;   // a new id for the file name
    int str_done = 0, in_fn = 0, after_merge = 0;
    for (size_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
        if (!str_done && op != 17 && op != 10 && op != 11 && op != 14 && op != 15 && op != 16 && op != 331) {   // end of capabilities .. execution modes
            o[k++] = (4u << 16) | 7; o[k++] = file; memcpy(&o[k], "s2r.spv\0", 8); k += 2;   // OpString, 8 bytes = 2 words
            str_done = 1;
        }
        if (op == 54) in_fn = 1;
        int skip = !in_fn || op == 54 || op == 55 || op == 56 || op == 248 || op == 245 || op == 59 || op == 246 || op == 247 || after_merge;
        if (!skip) { o[k++] = (4u << 16) | 8; o[k++] = file; o[k++] = (uint32_t)i; o[k++] = 1; }
        after_merge = op == 246 || op == 247;
        if (op == 56) in_fn = 0;
        memcpy(&o[k], &w[i], wc * 4); k += wc; i += wc;
    }
    *out_n = k; return o;
}

#if HAVE_AM
static uint64_t hash_of(const void *code, size_t bytes) {
    GFSDK_Aftermath_SpirvCode c; memset(&c, 0, sizeof c); c.pData = code; c.size = (uint32_t)bytes;
    GFSDK_Aftermath_ShaderBinaryHash h = {0};
    return p_hash(GFSDK_Aftermath_Version_API, &c, &h) == GFSDK_Aftermath_Result_Success ? h.hash : 0;
}
#endif

// Every shader module goes through here: with Aftermath on it gets its line markers, its hash is noted against our key, and its SPIR-V is kept
VkResult vk_module(const VkShaderModuleCreateInfo *ci, VkShaderModule *m, const char *key, const char *what) {
    if (!g_on) return vkCreateShaderModule(g_vk.dev, ci, NULL, m);
#if HAVE_AM
    size_t nn = 0; uint32_t *lw = g_lines ? add_lines(ci->pCode, ci->codeSize / 4, &nn) : NULL;
    VkShaderModuleCreateInfo c2 = *ci; if (lw) { c2.pCode = lw; c2.codeSize = nn * 4; }
    VkResult r = vkCreateShaderModule(g_vk.dev, &c2, NULL, m);
    uint64_t h = hash_of(c2.pCode, c2.codeSize);
    char path[1200];
    AcquireSRWLockExclusive(&g_lk);
    if (g_tsv) { fprintf(g_tsv, "%016llx\t%s\t%s\t%zu\n", (unsigned long long)h, key ? key : "?", what ? what : "", ci->codeSize / 4); }
    snprintf(path, sizeof path, "%s\\%016llx.spv", g_spv, (unsigned long long)h);
    if (GetFileAttributesA(path) == INVALID_FILE_ATTRIBUTES) {
        write_file(path, c2.pCode, c2.codeSize);
        if (lw) { snprintf(path, sizeof path, "%s\\%016llx.src.spv", g_spv, (unsigned long long)h); write_file(path, ci->pCode, ci->codeSize); }
    }
    ReleaseSRWLockExclusive(&g_lk);
    free(lw);
    return r;
#else
    (void)key; (void)what; return vkCreateShaderModule(g_vk.dev, ci, NULL, m);
#endif
}
