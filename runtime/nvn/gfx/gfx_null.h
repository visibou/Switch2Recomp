// The null backend's own interface (outside the contract): its counts, for tests and the frame statistics, and a hook on the programs a replay binds
#pragma once
#include <stdint.h>
#include "gfx_ops.h"

typedef struct {
    uint64_t submits, handles, records, bytes, draws, dispatches, copies, copy_bytes, updates, fills, fences, events, counters, tex_ops, presents, bad, calls;
    uint64_t op[OP_COUNT_];   // records walked, by op
    int depth_zero, origin_ll, queue_origin_ll, tex_pool_n;   // the last state the recording layer set
    int live_tex, reg_tex, reg_samp, reg_img, progs_released, cpu_written;
} gfx_null_counts;
void gfx_null_stats(gfx_null_counts *c);
// Called for every OP_BIND_PROGRAM the walk meets (the shader lookup a replay does there: gfx/nxs_rt.c)
typedef void (*gfx_null_bind_fn)(const prog_t *p, int stages);
void gfx_null_on_bind_program(gfx_null_bind_fn fn);
