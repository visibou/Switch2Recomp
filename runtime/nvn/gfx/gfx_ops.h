// The backend chosen at run time (nxrt --gfx null|vulkan, or NXRT_GFX). The recording layer calls the contract functions (gfx_backend.h, nvn_obj.h)
// by their names; those are defined once, in gfx_select.c, and forward to the chosen backend's table. A backend defines the same functions under its
// own names: every source of it starts with  #define GFX_BE <tag>  and  #include "gfx_rename.h"  (or is compiled with -DGFX_BE=<tag> -include gfx_rename.h),
// and one of its sources ends with  #include "gfx_ops_table.h", which builds  const gfx_backend_ops <tag>_gfx_ops.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "gfx_backend.h"
#include "nvn_obj.h"

typedef struct gfx_backend_ops {
    const char *name;
#define GFX_FN(ret, member, name, params, args) ret (*member) params;
#define GFX_FNV GFX_FN
#include "gfx_api.h"
#undef GFX_FN
#undef GFX_FNV
} gfx_backend_ops;

extern const gfx_backend_ops null_gfx_ops;   // gfx_null.c
#ifdef NXRT_GFX_VK
extern const gfx_backend_ops vk_gfx_ops;     // src/nxrt/vk
#endif

// Chooses the backend by name ("null", "vulkan"; NULL = NXRT_GFX, else null). Returns 0, or -1 for a name that is not built in. Once a contract function
// has run the choice is fixed: a later call with another name fails
int nxrt_gfx_select(const char *name);
const char *nxrt_gfx_name(void);   // the backend in use (choosing the default if nothing chose yet)
