// from nxnative dm/host/hle.h (MIT, © seedds): registering host implementations by Switch mangled name.
// On Windows the macros only declare the function: the registry, the guest-visible token and the AAPCS64 adapter of each entry are generated
// from these same lines by src/nxrt/gen/gen_hle.py (DESIGN D8), and each ported file includes its generated <file>.adapters.inc at its end.
// The Switch quirks the adapters handle: callers do not extend narrow (<32-bit) arguments (declare them int/unsigned and truncate, as before);
// variadic arguments arrive in registers (declare them as fixed parameters); results larger than 16 bytes go through x8; float aggregates in v0-v3.
// Win64 is LLP64: never use long / unsigned long in a signature (the generator refuses them); the Switch's long is int64_t.
#pragma once
#include "dm.h"

#define HLE(mangled, ret, cname, ...) static ret cname(__VA_ARGS__)

typedef uint32_t Result;   // nn::Result: w0, 0 = success
#define RES_OK 0u
#define RES_FAIL 0x2A2u      // Generic failure (cut features such as networking uniformly return this)

// Guest address → host object side table
void *objmap_get(const void *guest);
void objmap_put(const void *guest, void *host);
void *objmap_del(const void *guest);
uint64_t now_ns(void);
// A blocking native HLE wait releases the guest core. With the condition mutex
// held, enter with 1 before waiting and balance with 0 after reacquiring it.
void dm_thread_native_wait(int entering);
uint64_t dm_guest_frame(void);        // Incremented each time the host presents a frame; unaffected by machine load

// Forced takeover: even if the SDK module has a definition, all references to that symbol at load time (JUMP_SLOT/GLOB_DAT/ABS64) are resolved to the host implementation.
// Used only for the service-layer nn:: (service clients of fs/oe/aoc/pctl...); the SDK's libc++ and nn::os internals do not go into this table.
#define HLE_FORCE(mangled, why, ret, cname, ...) static ret cname(__VA_ARGS__)
// nxnative ran these on a big host stack (hle_bigstack.c); host code always runs on the host stack here
#define HLE_FORCE_BIG HLE_FORCE

// Operation mode (0 handheld / 1 docked) and focus (hle_oe.c when ported; defaults otherwise)
int dm_opmode(void);
void dm_set_opmode(int m);
void dm_focus_set(int st);
// Library applet stand-ins: DM_APPLET("name"); first in the stand-in's body holds the game out of focus until the body returns (hle_oe.c)
void dm_applet_enter(const char *name);
void dm_applet_leave(void);
static inline void dm_applet_leave_(const char *const *name) { (void)name; dm_applet_leave(); }
#define DM_APPLET(name) const char *const dm_applet_ __attribute__((cleanup(dm_applet_leave_))) = (name); dm_applet_enter(dm_applet_)
