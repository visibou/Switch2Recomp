// The run-time side of nxshader (src/nxshader, docs/m5-record-shaders.md): a program stage's Vulkan SPIR-V and reflection by its shader key (prog_key), from
// the cache nxshader fills (<cache>/vk/<key>.spv + .json, current while its chain id is <cache>/vk/CHAIN's). A key the cache lacks is never skipped silently:
// it is logged in red once, counted, dumped as nxnative's replay dumps it (DM_NVN_SHDUMP's format: <key>.ctl, the control section, and <key>.code, the code
// area), and handed to nxshader's on-demand translation, one at a time at below-normal priority; a later lookup finds it.
// Environment: NXRT_SHADER_CACHE (default <repo>/_local/nxshader/cache beside the executable's build folder), NXRT_SHADER_DUMP (default <cache>/dumps),
// NXRT_SHADER_TOOL (the command that translates: default py -3 -I <repo>/src/nxshader/nxshader.py --cache <cache>; "off" = dump and log only),
// NXRT_SHADER_BUILD (an installer build folder whose answer table the translation uses).
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "nvn_obj.h"

typedef struct nxs_shader {
    char key[17];
    const uint8_t *spv; size_t spv_size;    // the module (SPIR-V words)
    const char *json; size_t json_size;     // its reflection (nxs_refl.py's format, nul-terminated)
} nxs_shader;
typedef struct { uint64_t lookups, found, keys, missing_keys, dumped, requested, arrived, failed, stale; } nxs_counts;   // keys: distinct shaders loaded

int nxs_rt_enabled(void);                     // a cache to read (the backends ask before resolving)
const char *nxs_rt_cache_dir(void);
const nxs_shader *nxs_get(const char *key);   // NULL while missing (the first time: logged, no dump: the caller has no program to dump from)
const nxs_shader *nxs_peek(const char *key);  // the same, quiet: NULL when missing, nothing logged or marked (the pipeline prewarm)
// Stage i of program p (in SetShaders' order): 1 with *out set, 0 missing (logged, dumped, requested the first time), -1 no such stage
int nxs_resolve(const prog_t *p, unsigned i, const nxs_shader **out);
void nxs_counts_get(nxs_counts *c);
void nxs_rt_wait_idle(uint64_t timeout_ns);   // tests: until the on-demand queue is empty
