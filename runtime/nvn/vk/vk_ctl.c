// The control channel and the RenderDoc link (vk_flight.h). One command per connection on \\.\pipe\swr_ctl; tools/gamectl.py is the client.
#include "vk_flight.h"
#include "vk_mem.h"
#include "vk_replay.h"
#include "vk_shot.h"
#include "vk_submit.h"
#include <stdbool.h>
#include <stdarg.h>
#include "third_party/renderdoc/renderdoc_app.h"
#include <ctype.h>
#include <process.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern volatile int g_quit;
extern volatile int g_flight_req;
void dm_guest_write_prepare(void *addr, size_t len);

// ---------- Switches ----------
typedef struct { char name[32]; int v; } sw_t;
static sw_t g_sw[64]; static volatile LONG g_nsw; static SRWLOCK g_swl = SRWLOCK_INIT;
static sw_t *sw_find(const char *name) { for (LONG i = 0; i < g_nsw; i++) if (!strcmp(g_sw[i].name, name)) return &g_sw[i]; return NULL; }
int vkc_int(const char *name, int def) {
    sw_t *s = sw_find(name);
    if (s) return s->v;
    AcquireSRWLockExclusive(&g_swl);
    s = sw_find(name);
    if (!s && g_nsw < 64) { char e[64] = "NXRT_VK_"; size_t n = strlen(e); for (const char *p = name; *p && n < 62; p++) e[n++] = (char)toupper((unsigned char)*p); e[n] = 0;
        const char *v = getenv(e); sw_t *x = &g_sw[g_nsw]; snprintf(x->name, sizeof x->name, "%s", name); x->v = v ? atoi(v) : def; s = x; InterlockedIncrement(&g_nsw); }
    ReleaseSRWLockExclusive(&g_swl);
    return s ? s->v : def;
}
static void sw_set(const char *name, int v) { vkc_int(name, v); sw_t *s = sw_find(name); if (s) s->v = v; }

// ---------- Hidden shaders ----------
static uint64_t g_hide[64]; static volatile LONG g_nhide;
int vkc_any_hidden(void) { return g_nhide != 0; }
static void hide_add(uint64_t k);
// NXRT_VK_HIDE=KEY,KEY: hidden from the first frame, for tests that need many runs (gamectl hide does the same in a running game)
static void hide_env(void) { static int done; if (done) return; done = 1; const char *e = getenv("NXRT_VK_HIDE"); if (!e) return; char b[1024]; snprintf(b, sizeof b, "%s", e); for (char *t = strtok(b, ","); t; t = strtok(NULL, ",")) hide_add(vkf_key(t)); }
int vkc_hidden(uint64_t key) { hide_env(); if (!g_nhide || !key) return 0; for (LONG i = 0; i < g_nhide; i++) if (g_hide[i] == key) return 1; return 0; }
static void hide_add(uint64_t k) { if (!vkc_hidden(k) && g_nhide < 64) { g_hide[g_nhide] = k; InterlockedIncrement(&g_nhide); } }
static void hide_del(uint64_t k) { for (LONG i = 0; i < g_nhide; i++) if (g_hide[i] == k) { g_hide[i] = g_hide[g_nhide - 1]; InterlockedDecrement(&g_nhide); return; } }

// ---------- Pause and step ----------
static volatile LONG g_paused, g_step; static volatile uint64_t g_frame_now, g_held_at;
void vkc_gate(uint64_t frame) {
    g_frame_now = frame;
    if (g_flight_req) { g_flight_req = 0; vkf_trigger(""); }
    if (!g_paused) return;
    if (g_step > 0) { InterlockedDecrement(&g_step); if (g_step > 0) return; }
    vks_flush(); g_held_at = frame;
    while (g_paused && !g_step && !g_quit) Sleep(2);
    g_held_at = 0;
}

// ---------- RenderDoc ----------
static RENDERDOC_API_1_4_0 *g_rd;
void vkc_rdoc_load(void) {
    if (!getenv("SWR_RDOC")) return;
    SetEnvironmentVariableA("ENABLE_VULKAN_RENDERDOC_CAPTURE", "1");
    HMODULE m = GetModuleHandleA("renderdoc.dll");
    if (!m) m = LoadLibraryA("C:\\Program Files\\RenderDoc\\renderdoc.dll");
    pRENDERDOC_GetAPI get = m ? (pRENDERDOC_GetAPI)(void *)GetProcAddress(m, "RENDERDOC_GetAPI") : NULL;
    if (!get || !get(eRENDERDOC_API_Version_1_4_0, (void **)&g_rd)) { VKRED("SWR_RDOC: RenderDoc could not be loaded"); g_rd = NULL; return; }
    g_rd->SetCaptureFilePathTemplate("rdoc/mkw");
    g_rd->MaskOverlayBits(0, 0);
    VKLOG("RenderDoc in-application API on");
}

// ---------- The pipe ----------
static int replyf(char *o, size_t cap, size_t *n, const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); int k = vsnprintf(o + *n, cap - *n, fmt, ap); va_end(ap);
    if (k > 0) *n += (size_t)k < cap - *n ? (size_t)k : cap - *n - 1;
    return k;
}
#define R(...) replyf(o, cap, &n, __VA_ARGS__)
static void command(char *line, char *o, size_t cap) {
    size_t n = 0; o[0] = 0;
    char *av[8]; int ac = 0;
    for (char *t = strtok(line, " \t\r\n"); t && ac < 8; t = strtok(NULL, " \t\r\n")) av[ac++] = t;
    if (!ac) { R("no command\n"); return; }
    const char *c = av[0];
    if (!strcmp(c, "status")) {
        uint64_t ser; const char *last = vkf_last(&ser);
        R("frame %llu\npaused %ld\nheld_at %llu\nsteps_left %ld\ngpu_assigned %llu\ngpu_submitted %llu\nflights %llu\nlast_flight %s\nhidden %ld\nrenderdoc %d\n", (unsigned long long)g_frame_now, g_paused, (unsigned long long)g_held_at, g_step,
          (unsigned long long)atomic_load(&g_vk.tl_assigned), (unsigned long long)atomic_load(&g_vk.tl_submitted), (unsigned long long)ser, last, g_nhide, g_rd != NULL);
    }
    else if (!strcmp(c, "flight")) {
        uint64_t s0; vkf_last(&s0); vkf_trigger(ac > 1 ? av[1] : "");
        for (int i = 0; i < 6000; i++) { uint64_t s; const char *last = vkf_last(&s); if (s != s0) { R("%s\n", last); return; } Sleep(10); }
        R("timed out\n");
    }
    else if (!strcmp(c, "flashes")) { static char list[8192]; long k = ac > 1 ? atol(av[1]) : 60; int mb = ac > 2 ? atoi(av[2]) : 60; int m = vkf_flashes(k, list, sizeof list, mb); R("%d in the last %ld frames: %s\n", m, k, list); }
    else if (!strcmp(c, "pause")) { g_paused = 1; g_step = 0; for (int i = 0; i < 1500 && !g_held_at; i++) Sleep(10); R("paused at frame %llu\n", (unsigned long long)g_held_at); }
    else if (!strcmp(c, "resume")) { g_step = 0; g_paused = 0; R("running\n"); }
    else if (!strcmp(c, "step")) {
        long k = ac > 1 ? atol(av[1]) : 1; if (k < 1) k = 1;
        if (!g_paused) { g_paused = 1; for (int i = 0; i < 1500 && !g_held_at; i++) Sleep(10); }
        uint64_t f0 = g_held_at; g_step = k;
        for (int i = 0; i < 3000 && (g_held_at == 0 || g_held_at == f0); i++) Sleep(10);
        R("paused at frame %llu\n", (unsigned long long)g_held_at);
    }
    else if (!strcmp(c, "hide") && ac > 1) { for (int i = 1; i < ac; i++) hide_add(vkf_key(av[i])); R("%ld hidden\n", g_nhide); }
    else if (!strcmp(c, "show") && ac > 1) { for (int i = 1; i < ac; i++) hide_del(vkf_key(av[i])); R("%ld hidden\n", g_nhide); }
    else if (!strcmp(c, "showall")) { g_nhide = 0; R("0 hidden\n"); }
    else if (!strcmp(c, "hidden")) { for (LONG i = 0; i < g_nhide; i++) R("%016llx\n", (unsigned long long)g_hide[i]); }
    else if (!strcmp(c, "set") && ac > 2) { sw_set(av[1], atoi(av[2])); R("%s = %d\n", av[1], vkc_int(av[1], 0)); }
    else if (!strcmp(c, "get") && ac > 1) { sw_t *s = sw_find(av[1]); if (s) R("%s = %d\n", s->name, s->v); else R("%s has not been used yet\n", av[1]); }
    else if (!strcmp(c, "switches")) { for (LONG i = 0; i < g_nsw; i++) R("%s = %d\n", g_sw[i].name, g_sw[i].v); }
    else if (!strcmp(c, "shot")) { vks_shot_request(ac > 1 ? av[1] : ""); R("the next present\n"); }
    else if (!strcmp(c, "gputime")) { vkr_gputime_next(); R("in the log\n"); }
    else if (!strcmp(c, "passdump")) { vkr_passdump_next(ac > 1 ? atol(av[1]) : 1); R("in the log\n"); }
    else if (!strcmp(c, "rawdump") && ac > 2) { uint64_t ser[6]; int k = 0; for (int i = 2; i < ac && k < 6; i++) ser[k++] = strtoull(av[i], NULL, 0); vkr_rawdump(ser, k, atol(av[1])); R("%d images for %s frames\n", k, av[1]); }
    else if (!strcmp(c, "passview")) {
        uint64_t s0; vkf_last(&s0); vkv_request(ac > 1 ? atol(av[1]) : 16);
        for (int i = 0; i < 12000; i++) { uint64_t s; const char *last = vkf_last(&s); if (s != s0) { R("%s\n", last); return; } Sleep(10); }
        R("timed out\n");
    }
    else if (!strcmp(c, "watch") && ac > 2) {
        uint64_t k[6], s0; int m = 0; for (int i = 2; i < ac && m < 6; i++) k[m++] = vkf_key(av[i]); vkf_last(&s0); vkw_request(atol(av[1]), k, m);
        for (int i = 0; i < 12000; i++) { uint64_t s; const char *last = vkf_last(&s); if (s != s0) { R("%s\n", last); return; } Sleep(10); }
        R("timed out\n");
    }
    else if (!strcmp(c, "drawstats")) {
        uint64_t s0; vkf_last(&s0); vkq_request(ac > 1 ? atol(av[1]) : 16);
        for (int i = 0; i < 12000; i++) { uint64_t s; const char *last = vkf_last(&s); if (s != s0) { R("%s\n", last); return; } Sleep(10); }
        R("timed out\n");
    }
    else if (!strcmp(c, "texdump")) { vkr_texdump_next(); R("beside the log\n"); }
    else if (!strcmp(c, "peek") && ac > 1) {   // guest memory, 32-bit words
        uint64_t a = strtoull(av[1], NULL, 0); int k = ac > 2 ? atoi(av[2]) : 16; if (k > 4096) k = 4096;
        for (int i = 0; i < k; i++) { uint32_t v; if (IsBadReadPtr((const void *)(uintptr_t)(a + 4 * (uint64_t)i), 4)) { R("\nnot readable at %#llx\n", (unsigned long long)(a + 4 * (uint64_t)i)); break; }
            memcpy(&v, (const void *)(uintptr_t)(a + 4 * (uint64_t)i), 4); if (!(i & 7)) R("%s%#llx:", i ? "\n" : "", (unsigned long long)(a + 4 * (uint64_t)i)); R(" %08x", v); }
        R("\n");
    }
    else if (!strcmp(c, "wpage") && ac > 1) { extern void plat_watch_page_arm(unsigned long long); plat_watch_page_arm(strtoull(av[1], NULL, 16)); R("armed: each write to the page goes in the log, up to 600\n"); }
    else if (!strcmp(c, "wtrace") && ac > 2) { extern void plat_watch_range(unsigned long long, unsigned long long, int); plat_watch_range(strtoull(av[1], NULL, 16), strtoull(av[2], NULL, 16), ac > 3 && !strcmp(av[3], "huge")); R("armed\n"); }
    else if (!strcmp(c, "bp") && ac > 1) { extern void plat_bp(unsigned long long, long), plat_bp_min(unsigned long long); plat_bp_min(ac > 3 ? strtoull(av[3], NULL, 0) : 0); plat_bp(strtoull(av[1], NULL, 16), ac > 2 ? atol(av[2]) : 400); R("armed\n"); }
    else if (!strcmp(c, "rdoc")) {
        if (!g_rd) { R("not started with SWR_RDOC=1\n"); return; }
        uint32_t n0 = g_rd->GetNumCaptures(); g_rd->TriggerCapture();
        for (int i = 0; i < 12000 && g_rd->GetNumCaptures() == n0; i++) Sleep(10);
        char path[1024] = ""; uint32_t len = sizeof path; uint64_t ts;
        if (g_rd->GetNumCaptures() > n0 && g_rd->GetCapture(g_rd->GetNumCaptures() - 1, path, &len, &ts)) R("%s\n", path); else R("no capture came\n");
    }
    else if (!strcmp(c, "inpool") && ac > 1) {   // is guest memory [ADDR, ADDR+LEN) in an NVN memory pool the GPU sees (hex)
        uint64_t a = strtoull(av[1], NULL, 16), len = ac > 2 ? strtoull(av[2], NULL, 16) : 4; VkBuffer pb; VkDeviceSize po;   // not o / n: R uses them
        int in1 = vkm_addr(a, 1, &pb, &po), inn = vkm_addr(a, len, &pb, &po);
        R("%#llx+%#llx: %s\n", (unsigned long long)a, (unsigned long long)len, inn ?"in a memory pool" : in1 ? "starts in a memory pool, does not fit in it" : "in no memory pool (CPU memory only)");
    }
    else if (!strcmp(c, "texat") && ac > 1) {   // the textures whose storage covers a guest address (hex)
        int vkt_list_at(uint64_t addr, char *out, int cap); static char buf[16384];
        vkt_list_at(strtoull(av[1], NULL, 16), buf, (int)sizeof buf); R("%s", buf);
    }
    else if (!strcmp(c, "quit")) { g_quit = 1; R("quitting\n"); }
    else R("unknown command: %s\n", c);
}
static unsigned __stdcall ctl_main(void *arg) {
    (void)arg;
    static char in[4096], out[256 * 1024];
    for (;;) {
        HANDLE p = CreateNamedPipeA("\\\\.\\pipe\\swr_ctl", PIPE_ACCESS_DUPLEX, PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT, 1, sizeof out, sizeof in, 0, NULL);
        if (p == INVALID_HANDLE_VALUE) { VKLOG("control pipe: not available (another game is running?)"); return 0; }
        if (ConnectNamedPipe(p, NULL) || GetLastError() == ERROR_PIPE_CONNECTED) {
            DWORD got = 0, k; size_t n = 0;
            while (n < sizeof in - 1 && ReadFile(p, in + n, (DWORD)(sizeof in - 1 - n), &got, NULL) && got) { n += got; if (memchr(in, '\n', n)) break; }
            in[n] = 0;
            command(in, out, sizeof out);
            WriteFile(p, out, (DWORD)strlen(out), &k, NULL);
            FlushFileBuffers(p); DisconnectNamedPipe(p);
        }
        CloseHandle(p);
    }
}
void vkc_start(void) {
    static int on; if (on) return; on = 1;
    HANDLE t = (HANDLE)_beginthreadex(NULL, 0, ctl_main, NULL, 0, NULL);
    if (t) CloseHandle(t);
}
