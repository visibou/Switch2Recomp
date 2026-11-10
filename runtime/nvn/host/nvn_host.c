/* The NVN layer's view of the runtime: what pokopia-native's host (nxrt) gave hle_nvn.c, gfx/ and vk/, on our kernel and services.
   Guest threads are real host threads here and the SDK runs for real (docs/PLAN_NATIVE_GPU.md D1), so there is no guest core to give back
   around a host wait and no SDK function to replace: only logging, clocks, guest calls, the fault hooks and the names. */
#include "dm.h"
#include "hle_registry.h"
#include "ctl_gfx.h"
#include <stdarg.h>

/* runtime/nvn_hook.c: the part that sees rt.h */
void swr_nvn_log(const char *s);
void swr_nvn_fatal(const char *s);
void swr_nvn_frame(void);
uint64_t swr_nvn_slot_addr(unsigned index);
const char *swr_nvn_log_dir(void);
void swr_nvn_vi_crop(int c[4]);
NxwGuestCpu *swr_nvn_cur_cpu(void);
uint64_t swr_nvn_guest_call(uint64_t va, int n, const uint64_t *args);
uint64_t swr_nvn_guest_ns(void);
void swr_nvn_symbol(uint64_t a, char *out, size_t n);
int swr_nvn_quiet_faults(void);

uint64_t g_t0_ns;
struct dm_flags g_flag;
_Thread_local _Atomic int dm_host_work;
dm_gfx_ctl_ops dm_gfx_ctl;
__thread void **nvn_guard_cur;

void nxrt_logf(const char *fmt, ...) {
    char b[1600]; va_list a; va_start(a, fmt); int n = vsnprintf(b, sizeof b - 2, fmt, a); va_end(a);
    if (n < 0) return;
    if (n > (int)sizeof b - 2) n = sizeof b - 2;
    b[n] = '\n'; b[n + 1] = 0; swr_nvn_log(b);
}
void nxrt_die(const char *fmt, ...) {
    char b[1600]; va_list a; va_start(a, fmt); vsnprintf(b, sizeof b, fmt, a); va_end(a);
    swr_nvn_fatal(b); ExitProcess(1);
}

uint64_t now_ns(void) { return plat_now_ns(); }
uint64_t guest_ns(void) { return swr_nvn_guest_ns(); }
void dm_guest_frame_tick(void) { swr_nvn_frame(); }
void dm_vi_layer_crop(int c[4]) { swr_nvn_vi_crop(c); }
NxwGuestCpu *nxrt_cur_cpu(void) { return swr_nvn_cur_cpu(); }
uint64_t nxrt_guest_call(uint64_t va, int n, const uint64_t *args) { return swr_nvn_guest_call(va, n, args); }
void dm_backtrace(uint64_t fp) { LOG("guest frame pointer %#llx", (unsigned long long)fp); }
module_t *module_of(const void *p) { (void)p; return NULL; }
int dm_sdk_major(void) { return 19; }
void nxrt_symbolize(uint64_t a, char *out, size_t n) { swr_nvn_symbol(a, out, n); }
void nxrt_log_summary(void) {}
const char *nxrt_log_path(void) { return swr_nvn_log_dir(); }
uint32_t nxrt_hle_calls(const char *name) { (void)name; return 0; }

/* Names: a registry index is a slot of the synthetic range (nvn_hook.c) */
const nxrt_hle_entry *nxrt_hle_find(const char *name, int kind) {
    (void)kind;
    static struct { const char *k; int i; } *h; enum { HN = 4096 };
    if (!h) {
        void *t = calloc(HN, sizeof *h);
        for (unsigned i = 0; i < nxrt_hle_count; i++) {
            uint32_t v = 2166136261u; for (const char *c = nxrt_hle_table[i].name; *c; c++) v = (v ^ (uint8_t)*c) * 16777619u;
            __typeof__(h) e = (__typeof__(h))t + (v & (HN - 1));
            while (e->k) { e++; if (e == (__typeof__(h))t + HN) e = t; }
            e->k = nxrt_hle_table[i].name; e->i = (int)i;
        }
        h = t;
    }
    uint32_t v = 2166136261u; for (const char *c = name; *c; c++) v = (v ^ (uint8_t)*c) * 16777619u;
    for (uint32_t k = v & (HN - 1);; k = (k + 1) & (HN - 1)) { if (!h[k].k) return NULL; if (!strcmp(h[k].k, name)) return &nxrt_hle_table[h[k].i]; }
}
uint64_t nxrt_token_of(const char *name) {
    const nxrt_hle_entry *e = nxrt_hle_find(name, 0);
    return e ? swr_nvn_slot_addr((unsigned)(e - nxrt_hle_table)) : 0;
}
void nxrt_stub_log(const char *name, const char *file, uint8_t *seen, const NxwGuestCpu *cpu) {
    if (*seen) return;
    *seen = 1; LOG("[RED] NVN stub called: %s (%s), from %#llx", name, file, (unsigned long long)cpu->x[30]);
}

/* Fault hooks: the write-protect tracker (nvn_dirty.c), and nvn_guard.h's guarded reads */
enum { MAX_HOOKS = 8 };
static int (*g_hook[MAX_HOOKS])(void *, int, void *); static int g_nhook;
static void (*g_prep[MAX_HOOKS])(void *, size_t); static int g_nprep;
static int (*g_wp)(void *);
void dm_fault_hook_register(int (*fn)(void *, int, void *)) { if (g_nhook < MAX_HOOKS) g_hook[g_nhook++] = fn; }
void dm_guest_write_prep_register(void (*fn)(void *, size_t)) { if (g_nprep < MAX_HOOKS) g_prep[g_nprep++] = fn; }
void dm_fault_wp_register(int (*fn)(void *)) { g_wp = fn; }
int dm_fault_is_wp(void *a) { return g_wp && g_wp(a); }
void dm_guest_write_prepare(void *a, size_t n) { for (int i = 0; i < g_nprep; i++) g_prep[i](a, n); }

LONG plat_lazy_try(EXCEPTION_POINTERS *e);   /* plat.c: memory committed on first touch */
/* In this order: the write-protect tracker, lazily committed memory, then a guarded read gives up. Registered first, so before the fatal handler */
static LONG WINAPI veh_nvn(EXCEPTION_POINTERS *e) {
    if (e->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || e->ExceptionRecord->NumberParameters < 2) return EXCEPTION_CONTINUE_SEARCH;
    void *a = (void *)e->ExceptionRecord->ExceptionInformation[1];
    int wr = e->ExceptionRecord->ExceptionInformation[0] == 1;
    for (int i = 0; i < g_nhook; i++) if (g_hook[i](a, wr, e->ContextRecord)) return EXCEPTION_CONTINUE_EXECUTION;
    void **b = nvn_guard_cur;
    if (!b) return EXCEPTION_CONTINUE_SEARCH;
    if (plat_lazy_try(e) == EXCEPTION_CONTINUE_EXECUTION) return EXCEPTION_CONTINUE_EXECUTION;
    e->ContextRecord->Rbp = (DWORD64)b[0]; e->ContextRecord->Rip = (DWORD64)b[1]; e->ContextRecord->Rsp = (DWORD64)b[2];
    return EXCEPTION_CONTINUE_EXECUTION;
}
void nvn_host_init(void) { g_t0_ns = now_ns(); AddVectoredExceptionHandler(1, veh_nvn); }
void dm_gfx_shot_at(const char *path) { if (dm_gfx_ctl.shot_at) dm_gfx_ctl.shot_at(path); }

/* The clock and sleep of plat.h */
uint64_t plat_now_ns(void) {   // called tens of thousands of times a frame: one multiply when the counter's rate divides a second (10 MHz on current Windows), else two divisions
    static LARGE_INTEGER f; static uint64_t mul; if (!f.QuadPart) { QueryPerformanceFrequency(&f); mul = 1000000000ull % (uint64_t)f.QuadPart ? 0 : 1000000000ull / (uint64_t)f.QuadPart; }
    LARGE_INTEGER c; QueryPerformanceCounter(&c);
    if (mul) return (uint64_t)c.QuadPart * mul;
    return (uint64_t)c.QuadPart / (uint64_t)f.QuadPart * 1000000000ull + (uint64_t)c.QuadPart % (uint64_t)f.QuadPart * 1000000000ull / (uint64_t)f.QuadPart;
}
uint64_t plat_realtime_ns(void) { FILETIME t; GetSystemTimePreciseAsFileTime(&t); return ((((uint64_t)t.dwHighDateTime << 32) | t.dwLowDateTime) - 116444736000000000ull) * 100; }
void plat_sleep_ns(uint64_t ns) {
    static __thread HANDLE h; if (!h) h = CreateWaitableTimerExW(NULL, NULL, 2 /* CREATE_WAITABLE_TIMER_HIGH_RESOLUTION */, 0x1F0003);
    LARGE_INTEGER d; d.QuadPart = -(LONGLONG)(ns / 100);
    if (h && d.QuadPart && SetWaitableTimer(h, &d, 0, NULL, NULL, 0)) WaitForSingleObject(h, INFINITE); else Sleep((DWORD)(ns / 1000000));
}
void plat_yield(void) { SwitchToThread(); }

/* The Switch 2's NVN answers, from the game's sdk (nxnative dm/host/loader.c) */
static uint64_t g_sdk_base, g_sdk_size;
void nvn_host_sdk(uint64_t base, uint64_t size) { g_sdk_base = base; g_sdk_size = size; }
int dm_sdk_ounce(void) {
    static int ounce = -1;
    if (ounce >= 0) return ounce;
    ounce = 0;
    const char *b = (const char *)(uintptr_t)g_sdk_base, *e = b + g_sdk_size;
    for (const char *q = b; q + 11 <= e; q++) if (*q == 's' && !memcmp(q, "spec: Ounce", 11)) { ounce = 1; break; }
    if (ounce) LOG("sdk: built for the Switch 2 (spec: Ounce)");
    return ounce;
}
// The constants NVN2's own nvnDeviceGetInteger answers: the driver is compiled into the game's sdk, and its API and GPU code versions move with the SDK (20.5.19: API 231.30, GPU code minor 17;
// 21.4.4: 231.39 and 20). Games check them: Splatoon Raiders requires minor >= 39 and otherwise creates no graphics system (main+0x31b86b0). The driver's switch over the items loads w0 with the
// major first, then jumps through a byte table: mov w0, #231 / adr / adrp + add (the table) / ldrb / add, lsl #2 / br, with cmp wN, #<last item> before it. A case that is a constant is mov w0, #v; b.
// Returns 1 with *out set when item has such a case; 0 for a case that reads something else (most items go through the driver's table lookup or read the GPU's description), or no such switch
int dm_sdk_nvn2_item(unsigned item, int *out) {
    static const uint8_t *table; static const uint32_t *cases; static unsigned last; static int found = -1;
    if (found < 0) {
        found = 0;
        {
            const uint32_t *t = (const uint32_t *)(uintptr_t)g_sdk_base; size_t n = g_sdk_size / 4;
            for (size_t p = 8; p + 9 < n && !found; p++) if (t[p] == 0x52801CE0) {   // mov w0, #231
                const uint8_t *tb = NULL; const uint32_t *adr = NULL; int lim = -1, ldrb = 0, br = 0;
                for (size_t q = p - 8; q <= p + 8; q++) {
                    uint32_t w = t[q];
                    if ((w & 0xFFC0001F) == 0x7100001F) lim = (w >> 10) & 0xFFF;   // cmp wN, #imm
                    else if ((w & 0x9F000000) == 0x90000000 && (t[q + 1] & 0xFFC00000) == 0x91000000) {   // adrp + add
                        int64_t pg = (int64_t)(((w >> 3) & 0x1FFFFC) | ((w >> 29) & 3)); if (pg & 0x100000) pg -= 0x200000;
                        tb = (const uint8_t *)(((uintptr_t)&t[q] & ~(uintptr_t)0xFFF) + (uintptr_t)(pg * 4096) + ((t[q + 1] >> 10) & 0xFFF));
                    } else if ((w & 0x9F000000) == 0x10000000) {   // adr
                        int64_t off = (int64_t)(((w >> 3) & 0x1FFFFC) | ((w >> 29) & 3)); if (off & 0x100000) off -= 0x200000;
                        adr = (const uint32_t *)((const char *)&t[q] + off);
                    } else if ((w & 0xFFE0FC00) == 0x38606800) ldrb = 1;   // ldrb wK, [xT, xI]
                    else if ((w & 0xFFFFFC1F) == 0xD61F0000) br = 1;       // br
                }
                if (!tb || !adr || lim <= 1 || !ldrb || !br) continue;
                const char *ro = (const char *)(uintptr_t)g_sdk_base, *ro_end = ro + g_sdk_size;
                if ((const char *)tb < ro || (const char *)tb + lim >= ro_end) continue;
                table = tb; cases = adr; last = (unsigned)lim; found = 1;
            }
        }
        if (dm_sdk_ounce() && !found) LOG("[RED] sdk: NVN2's nvnDeviceGetInteger switch not found; the device answers the newest versions known");
    }
    if (!found || item > last) return 0;
    const uint32_t *c = cases + table[item];
    if ((c[0] & 0xFFE0001F) != 0x52800000 || (c[1] & 0xFC000000) != 0x14000000) return 0;   // mov w0, #imm16; b
    *out = (int)((c[0] >> 5) & 0xFFFF);
    return 1;
}

