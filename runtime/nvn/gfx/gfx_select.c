// The contract functions the recording layer calls (gfx_backend.h, nvn_obj.h), each forwarding to the backend chosen at run time (gfx_ops.h).
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include "gfx_ops.h"
#include "dm.h"

static const struct { const char *name; const gfx_backend_ops *ops; } g_backends[] = {
    {"null", &null_gfx_ops},
#ifdef NXRT_GFX_VK
    {"vulkan", &vk_gfx_ops},
#endif
};
static const gfx_backend_ops *_Atomic g_be;
static _Atomic int g_used;   // a contract function ran: the choice is final

static const gfx_backend_ops *find(const char *name) {
    for (size_t i = 0; i < sizeof g_backends / sizeof *g_backends; i++) if (!strcmp(g_backends[i].name, name)) return g_backends[i].ops;
    return NULL;
}
int nxrt_gfx_select(const char *name) {
    const char *e = getenv("NXRT_GFX");
    if (!name || !*name) name = e && *e ? e : "null";
    const gfx_backend_ops *o = find(name);
    if (!o) {
        LOG("[RED] graphics backend \"%s\" is not built into this nxrt (built in: null%s)", name,
#ifdef NXRT_GFX_VK
            ", vulkan"
#else
            ""
#endif
        );
        return -1;
    }
    const gfx_backend_ops *cur = atomic_load(&g_be);
    if (atomic_load(&g_used) && cur && cur != o) { LOG("[RED] graphics backend already in use (%s): cannot switch to %s", cur->name, o->name); return -1; }
    atomic_store(&g_be, o);
    if (cur != o) LOG("[gfx] backend: %s", o->name);
    return 0;
}
static const gfx_backend_ops *be(void) {
    const gfx_backend_ops *o = atomic_load(&g_be);
    if (!o) {
        if (nxrt_gfx_select(NULL)) { LOG("[RED] NXRT_GFX names no built-in backend: using null"); atomic_store(&g_be, &null_gfx_ops); }
        o = atomic_load(&g_be);
    }
    if (!atomic_load(&g_used)) atomic_store(&g_used, 1);
    return o;
}
const char *nxrt_gfx_name(void) { return be()->name; }

#define GFX_FN(ret, member, name, params, args) ret name params { return be()->member args; }
#define GFX_FNV(ret, member, name, params, args) ret name params { be()->member args; }
#include "gfx_api.h"
