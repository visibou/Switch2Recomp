// The pipeline cache, persisted incrementally (research/01 lesson 16: a cache written only at exit is never written when the process is terminated).
// A saver thread writes it a few seconds after new pipelines stop arriving, through a temporary file and an atomic replace, so a crash or a kill
// at any moment leaves either the previous file or the new one. The file is checked against this device's pipelineCacheUUID before use.
#include "vk_internal.h"
#include <process.h>
#include <shlobj.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void vkpc_note_created(int n);
int vkpc_save_now(void);

static struct {
    wchar_t path[MAX_PATH], tmp[MAX_PATH];
    _Atomic uint64_t created, saved_at_count, last_created_ns, saves, bytes;
    uint64_t loaded;
    HANDLE thread, wake; _Atomic int stop;
    SRWLOCK lk;
    int quiet_ms;
} g_pc;

static int cache_dir(wchar_t *out, size_t n) {
    const char *e = getenv("NXRT_VK_CACHE_DIR");
    if (e && *e) { MultiByteToWideChar(CP_UTF8, 0, e, -1, out, (int)n); CreateDirectoryW(out, NULL); return 0; }
    wchar_t *la = NULL;
    if (SHGetKnownFolderPath(&FOLDERID_LocalAppData, 0, NULL, &la) != S_OK) return -1;
    _snwprintf(out, n, L"%ls\\pokopia-native", la); CreateDirectoryW(out, NULL);
    _snwprintf(out, n, L"%ls\\pokopia-native\\vulkan", la); CreateDirectoryW(out, NULL);
    CoTaskMemFree(la);
    return 0;
}

static unsigned __stdcall saver_main(void *arg) {
    (void)arg;
    SetThreadDescription(GetCurrentThread(), L"vk pipeline cache");
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    while (!atomic_load(&g_pc.stop)) {
        WaitForSingleObject(g_pc.wake, 1000);
        uint64_t c = atomic_load(&g_pc.created), s = atomic_load(&g_pc.saved_at_count);
        if (c == s) continue;
        if (now_ns() - atomic_load(&g_pc.last_created_ns) < (uint64_t)g_pc.quiet_ms * 1000000ull) continue;   // let a burst of pipelines finish first
        vkpc_save_now();
    }
    return 0;
}

// Creates g_vk.pcache from the saved file (when it was made by this device and driver) and starts the saver
int vkpc_init(void) {
    InitializeSRWLock(&g_pc.lk);
    const char *q = getenv("NXRT_VK_PCACHE_QUIET_MS"); g_pc.quiet_ms = q ? atoi(q) : 2000;
    wchar_t dir[MAX_PATH];
    void *data = NULL; size_t size = 0;
    if (!cache_dir(dir, MAX_PATH)) {
        _snwprintf(g_pc.path, MAX_PATH, L"%ls\\pipelines-%04x-%04x.bin", dir, g_vk.props.vendorID, g_vk.props.deviceID);
        _snwprintf(g_pc.tmp, MAX_PATH, L"%ls.tmp", g_pc.path);
        FILE *f = _wfopen(g_pc.path, L"rb");
        if (f) {
            fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
            if (n > (long)sizeof(VkPipelineCacheHeaderVersionOne) && (data = malloc((size_t)n)) && fread(data, 1, (size_t)n, f) == (size_t)n) size = (size_t)n;
            fclose(f);
            const VkPipelineCacheHeaderVersionOne *h = data;
            if (size && (h->headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE || h->vendorID != g_vk.props.vendorID || h->deviceID != g_vk.props.deviceID ||
                         memcmp(h->pipelineCacheUUID, g_vk.props.pipelineCacheUUID, VK_UUID_SIZE))) {
                VKLOG("pipeline cache file is from another device or driver: started empty"); size = 0;
            }
        }
    }
    VkPipelineCacheCreateInfo ci = {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, NULL, 0, size, size ? data : NULL};
    VkResult r = vkCreatePipelineCache(g_vk.dev, &ci, NULL, &g_vk.pcache);
    if (r != VK_SUCCESS && size) { ci.initialDataSize = 0; ci.pInitialData = NULL; size = 0; r = vkCreatePipelineCache(g_vk.dev, &ci, NULL, &g_vk.pcache); }
    free(data);
    if (r != VK_SUCCESS) { VKRED("cannot create the pipeline cache (%d)", r); return -1; }
    g_pc.loaded = size;
    VKLOG("pipeline cache %ls: %zu bytes loaded; saved incrementally (%d ms after the last new pipeline)", g_pc.path[0] ? g_pc.path : L"(no folder)", size, g_pc.quiet_ms);
    if (!g_pc.path[0]) return 0;
    g_pc.wake = CreateEventW(NULL, FALSE, FALSE, NULL);
    atomic_store(&g_pc.stop, 0);
    g_pc.thread = (HANDLE)_beginthreadex(NULL, 0, saver_main, NULL, 0, NULL);
    return 0;
}

void vkpc_note_created(int n) { atomic_fetch_add(&g_pc.created, (uint64_t)n); atomic_store(&g_pc.last_created_ns, now_ns()); }

// Writes the cache now (temporary file, flush, atomic replace); 0 on success
int vkpc_save_now(void) {
    if (!g_vk.pcache || !g_pc.path[0]) return -1;
    AcquireSRWLockExclusive(&g_pc.lk);
    uint64_t c = atomic_load(&g_pc.created);
    size_t n = 0; void *d = NULL; int ok = 0;
    if (vkGetPipelineCacheData(g_vk.dev, g_vk.pcache, &n, NULL) == VK_SUCCESS && n && (d = malloc(n)) && vkGetPipelineCacheData(g_vk.dev, g_vk.pcache, &n, d) == VK_SUCCESS) {
        HANDLE h = CreateFileW(g_pc.tmp, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        if (h != INVALID_HANDLE_VALUE) {
            DWORD w = 0; ok = WriteFile(h, d, (DWORD)n, &w, NULL) && w == n && FlushFileBuffers(h);
            CloseHandle(h);
            ok = ok && MoveFileExW(g_pc.tmp, g_pc.path, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
        }
    }
    free(d);
    if (ok) { atomic_store(&g_pc.saved_at_count, c); atomic_fetch_add(&g_pc.saves, 1); atomic_store(&g_pc.bytes, n); }
    ReleaseSRWLockExclusive(&g_pc.lk);
    return ok ? 0 : -1;
}

void vkpc_shutdown(void) {
    if (g_pc.thread) { atomic_store(&g_pc.stop, 1); SetEvent(g_pc.wake); WaitForSingleObject(g_pc.thread, 5000); CloseHandle(g_pc.thread); g_pc.thread = NULL; CloseHandle(g_pc.wake); }
    if (atomic_load(&g_pc.created) != atomic_load(&g_pc.saved_at_count)) vkpc_save_now();
    if (g_vk.pcache) vkDestroyPipelineCache(g_vk.dev, g_vk.pcache, NULL);
    g_vk.pcache = VK_NULL_HANDLE;
}

void vkpc_stats(uint64_t *created, uint64_t *saves, uint64_t *bytes, uint64_t *loaded) {
    *created = atomic_load(&g_pc.created); *saves = atomic_load(&g_pc.saves); *bytes = atomic_load(&g_pc.bytes); *loaded = g_pc.loaded;
}
const wchar_t *vkpc_path(void) { return g_pc.path; }
