// Included once, at the end of one source of a backend (GFX_BE defined, gfx_rename.h included): its table of contract functions, <tag>_gfx_ops
#pragma once
#include "gfx_ops.h"
#define GFX_OPS_STR_(x) #x
#define GFX_OPS_STR(x) GFX_OPS_STR_(x)
const gfx_backend_ops GFX_BE_NAME(gfx_ops) = {
    .name = GFX_OPS_STR(GFX_BE),
#define GFX_FN(ret, member, name, params, args) .member = name,
#define GFX_FNV GFX_FN
#include "gfx_api.h"
#undef GFX_FN
#undef GFX_FNV
};
