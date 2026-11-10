// from nxnative dm/host/dm.h (MIT, © seedds): the common header of the host, for Windows x86-64 with recompiled guest code.
// Kept: the profile, host work, LOG/die, module table, guest clocks, SystemEvent bindings, fault and pre-write hooks.
// Gone with native execution: key page and TSD slots, gates and observation gates, patch areas, cheats, Unreal/fps state, Mac/iOS front-end hooks.
#pragma once
#include "posix_win.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "game_profile.h"   // constants of the current game profile (nxnative generates it from dm/games/<id>/game.tsv)
#include "nxw/guest_cpu.h"

// Explicit host work does not occupy a guest core (GPU submission, or a blocked native codec lock). A contended host mutex must count as blocked: a signal's same-core handoff
// can otherwise wait forever for a peer that needs the signaller's native lock. Uncontended locks retain guest ordering; the counter nests inside existing host work.
extern _Thread_local _Atomic int dm_host_work;
static inline int dm_host_mutex_lock(pthread_mutex_t *mutex) {
    int r = pthread_mutex_trylock(mutex);
    if (r != EBUSY) return r;
    dm_host_work++;
    r = pthread_mutex_lock(mutex);
    dm_host_work--;
    return r;
}

// The log (host/log.c): every line to the run's log file (flushed) and to stderr, one lock per line
void nxrt_logf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
_Noreturn void nxrt_die(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
#define LOG(...) nxrt_logf(__VA_ARGS__)
#define die(...) nxrt_die(__VA_ARGS__)

#include "module.h"   // module_t, g_mod, module_sym / guest_sym (guest addresses: call them with GUEST_CALL), module_of
// main's build ID (empty if none): code with fixed addresses in a game's module runs only on the build they were found in
static inline const char *dm_main_build(void) { for (int i = 0; i < g_nmod; i++) if (!strcmp(g_mod[i].name, "main")) return g_mod[i].build_id; return ""; }
#define DM_SDK_OLD(major) ((major) > 0 && (major) <= 10)

// Host-to-guest calls (host/rt.c): x0.. = the arguments, returns x0. On a guest thread only
uint64_t nxrt_guest_call(uint64_t guest_va, int n, const uint64_t *args);
#define GUEST_CALL(fn, ...) nxrt_guest_call((uint64_t)(uintptr_t)(fn), (int)(sizeof((uint64_t[]){__VA_ARGS__}) / 8), (uint64_t[]){__VA_ARGS__})
#define GUEST_CALL0(fn) nxrt_guest_call((uint64_t)(uintptr_t)(fn), 0, NULL)
NxwGuestCpu *nxrt_cur_cpu(void);
// Where a host replacement was called from in the guest: its x30 (nxnative used __builtin_return_address(0), which is the host's here)
#define DM_GUEST_RA() ((void *)(uintptr_t)(nxrt_cur_cpu() ? nxrt_cur_cpu()->x[30] : 0))
#define DM_GUEST_FP() (nxrt_cur_cpu() ? nxrt_cur_cpu()->x[29] : 0)

uint64_t now_ns(void);
uint64_t guest_ns(void);              // Guest clock (hle_os.c; does not advance while frozen, guest_clock_freeze)
uint64_t guest_ticks(void);           // The guest system counter, 31.25 MHz (hle_os.c)
int dm_detclock_enabled(void);
void dm_guest_clock_advanced(void);
void guest_clock_freeze(int on);
void guest_clock_freeze_for(int reason, int on);   // Freeze by reason (1 background, 2 system keyboard); see hle_os.c
void guest_clock_read_wait(void);     // A guest read of the system counter: waits while the app is in the background (hle_os.c)
void dm_guest_frame_tick(void);
uint64_t dm_entropy64(void);
void dm_backtrace(uint64_t fp);       // the guest frame chain from fp (crash.c)
void crash_dump_now(const char *why);
// Diagnostic switches: read once from environment variables at start (host/main.c)
extern uint64_t g_t0_ns;
extern struct dm_flags { int thr, svc, fs, dbg, imports, bootprof; } g_flag;
void *objmap_get(const void *guest); void objmap_put(const void *guest, void *host); void *objmap_del(const void *guest);
void hle_signal_system_event(void *g);   // hle_thread.c: signal a service-layer SystemEvent (any thread)
void hle_bind_system_event(void *g, int autoclear);   // On a guest thread, bind a guest SystemEvent to a kernel event
void hle_clear_system_event(void *g);
int hle_signal_bound_system_event(void *g);
uint64_t hle_system_event_binding(void *g);
int hle_signal_system_event_binding(uint64_t binding);
uint32_t hle_dup_readable_handle(uint32_t h);
// Page fault hook / hook before the host writes guest memory (crash.c; used for write-protect dirty page tracking)
void dm_fault_hook_register(int (*fn)(void *addr, int is_write, void *ctx));
void dm_guest_write_prep_register(void (*fn)(void *addr, size_t len));
void dm_fault_wp_register(int (*fn)(void *addr));   // the write-protect tracker: 1 when it has the page write-protected (or is changing it)
int dm_fault_is_wp(void *addr);
void dm_guest_write_prepare(void *addr, size_t len);
void dm_guest_heap_range(uint64_t *base, size_t *open_len);
int dm_guest_heap_read(uint64_t addr, void *out, size_t len);
int dm_guest_heap_write(uint64_t addr, const void *src, size_t len);
void dm_guest_sleep_ns(int64_t ns);
void dm_host_shutdown(const char *why);
