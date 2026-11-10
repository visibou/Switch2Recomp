// Shaders for the replay (vk_shader.h). The binding kinds and their meaning are nxnative's (nvn_refl.m, nvn_mtl.m encode_args), as nxshader's reflection
// names them (dmsh_abi.h DMSH_KIND_*); the SPIR-V side is read from the module itself (the SPIR-V specification's physical layout).
#include "vk_shader.h"
#include "vk_json.h"
#include "vk_hmap.h"
#include "nxs_rt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static vk_hmap g_sh = {17};   // key → vks_shader (a failed one stays, marked, so it is not retried)
static SRWLOCK g_shl = SRWLOCK_INIT;   // M6 phase B: the table, the variants and the c1 bank, shared by the replay and the pipeline prewarm (vk_pipe.c)
static VkDescriptorSetLayout g_empty;
static uint64_t g_loaded, g_failed;
static struct { VkBuffer buf; vk_alloc mem; VkDeviceSize used, size; } g_c1;   // the c1 bank (host-visible, written once per shader)
static vks_shader g_failed_mark;

int vksh_set_of(int stage) { return stage == VKS_CS ? 0 : stage == VKS_VS ? 0 : stage == VKS_FS ? 1 : stage == VKS_GS ? 2 : stage == VKS_TCS ? 3 : 4; }
VkShaderStageFlagBits vksh_stage_bit(const vks_shader *s) { return s->mesh == 1 ? VK_SHADER_STAGE_TASK_BIT_EXT : s->mesh == 2 ? VK_SHADER_STAGE_MESH_BIT_EXT : vksh_vk_stage(s->stage); }
VkShaderStageFlagBits vksh_vk_stage(int stage) {
    switch (stage) {
    case VKS_VS: return VK_SHADER_STAGE_VERTEX_BIT; case VKS_FS: return VK_SHADER_STAGE_FRAGMENT_BIT; case VKS_GS: return VK_SHADER_STAGE_GEOMETRY_BIT;
    case VKS_TCS: return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT; case VKS_TES: return VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT; default: return VK_SHADER_STAGE_COMPUTE_BIT;
    }
}
VkDescriptorSetLayout vksh_empty_dsl(void) { return g_empty; }
void vksh_stats(uint64_t *loaded, uint64_t *failed) { *loaded = g_loaded; *failed = g_failed; }

// ---------- SPIR-V ----------
enum { OP_NAME = 5, OP_TYPE_INT = 21, OP_TYPE_FLOAT = 22, OP_TYPE_IMAGE = 25, OP_TYPE_SAMPLED_IMAGE = 27, OP_TYPE_ARRAY = 28, OP_TYPE_RUNTIME_ARRAY = 29,
       OP_TYPE_POINTER = 32, OP_VARIABLE = 59, OP_DECORATE = 71, DEC_BINDING = 33, DEC_SET = 34 };
static int spv_ok(const uint32_t *w, size_t n) { return n >= 5 && w[0] == 0x07230203u && w[3] > 0 && w[3] < (1u << 22); }

int vkspv_set_descriptor_set(uint32_t *w, size_t n, uint32_t set) {
    if (!spv_ok(w, n)) return -1;
    int changed = 0;
    for (size_t i = 5; i < n; ) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (!wc || i + wc > n) return -1;
        if (op == OP_DECORATE && wc >= 4 && w[i + 2] == DEC_SET) { w[i + 3] = set; changed++; }
        i += wc;
    }
    return changed;
}

// gl_PerVertex declares ClipDistance[8] and CullDistance[1] (Maxwell has 8 clip distances): more than a device with maxCombinedClipAndCullDistances 8 takes.
// The cull member goes when nothing reaches it (Pokopia never culls by distance): the block's last member, no access chain names it, the block is never
// loaded, stored or built whole. Returns the new word count (*w may be replaced), n when nothing changed, 0 when it cannot be done
// NXRT_VK_NO_INVARIANT=1 (test, the GPU freeze): the Invariant decorations (OpDecorate / OpMemberDecorate ... Invariant) removed from a module, in place.
// The translator marks every vertex shader's position invariant. Returns the new length
static size_t vkspv_strip_invariant(uint32_t *w, size_t n) {
    size_t i = 5, o = 5;
    while (i < n) { uint32_t op = w[i] & 0xffff, ln = w[i] >> 16; if (!ln || i + ln > n) return n;
        int inv = (op == 71 && ln >= 3 && w[i + 2] == 18) || (op == 72 && ln >= 4 && w[i + 3] == 18);
        if (!inv) { if (o != i) memmove(w + o, w + i, ln * 4); o += ln; }
        i += ln; }
    return o;
}
size_t vkspv_trim_cull(uint32_t **pw, size_t n, uint32_t limit) {
    enum { OP_MEMBER_NAME = 6, OP_TYPE_STRUCT = 30, OP_CONSTANT = 43, OP_LOAD = 61, OP_STORE = 62, OP_COPY_MEMORY = 63, OP_ACCESS_CHAIN = 65, OP_IN_BOUNDS_ACCESS_CHAIN = 66,
           OP_MEMBER_DECORATE = 72, OP_COMPOSITE_CONSTRUCT = 80, OP_COPY_OBJECT = 83, DEC_BUILTIN = 11, BI_CLIP = 3, BI_CULL = 4 };
    uint32_t *w = *pw;
    if (!spv_ok(w, n)) return 0;
    uint32_t bound = w[3], st = 0, clip_m = ~0u, cull_m = ~0u;
    for (size_t i = 5; i < n; ) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (!wc || i + wc > n) return 0;
        if (op == OP_MEMBER_DECORATE && wc >= 5 && w[i + 3] == DEC_BUILTIN && (w[i + 4] == BI_CLIP || w[i + 4] == BI_CULL)) {
            if (st && st != w[i + 1]) return n;   // more than one block declares them: left alone
            st = w[i + 1]; if (w[i + 4] == BI_CLIP) clip_m = w[i + 2]; else cull_m = w[i + 2];
        }
        i += wc;
    }
    if (!st || clip_m == ~0u || cull_m == ~0u || st >= bound) return n;
    uint32_t *def = calloc(bound, sizeof *def), *cval = calloc(bound, sizeof *cval); uint8_t *isc = calloc(bound, 1);
    size_t si = 0; size_t out = 0;
    for (size_t i = 5; i < n; i += w[i] >> 16) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (op >= OP_TYPE_INT && op <= OP_TYPE_POINTER && wc >= 2 && w[i + 1] < bound) def[w[i + 1]] = (uint32_t)i;
        if (op == OP_TYPE_STRUCT && w[i + 1] == st) si = i;
        if (op == OP_CONSTANT && wc >= 4 && w[i + 2] < bound) { cval[w[i + 2]] = w[i + 3]; isc[w[i + 2]] = 1; }
    }
    uint32_t nmem = si ? (w[si] >> 16) - 2 : 0;
    // the members' array lengths
    uint32_t len[2] = {0, 0}, mem[2] = {clip_m, cull_m};
    for (int k = 0; k < 2 && si; k++) if (mem[k] < nmem) { uint32_t t = w[si + 2 + mem[k]];
        if (t < bound && def[t] && (w[def[t]] & 0xffff) == OP_TYPE_ARRAY && w[def[t] + 3] < bound && isc[w[def[t] + 3]]) len[k] = cval[w[def[t] + 3]]; }
    int ok = si && cull_m == nmem - 1 && len[0] && len[1] && len[0] + len[1] > limit && len[0] <= limit;
    // what refers to the block: pointers to it or to arrays of it, the variables of those
    uint8_t *blk = calloc(bound, 1), *bptr = calloc(bound, 1), *bvar = calloc(bound, 1);   // blk: the block or an array of it
    if (ok) {
        blk[st] = 1;
        for (size_t i = 5; i < n; i += w[i] >> 16) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
            if ((op == OP_TYPE_ARRAY || op == OP_TYPE_RUNTIME_ARRAY) && wc >= 3 && w[i + 2] < bound && blk[w[i + 2]] && w[i + 1] < bound) blk[w[i + 1]] = 2;
            if (op == OP_TYPE_POINTER && wc >= 4 && w[i + 3] < bound && blk[w[i + 3]] && w[i + 1] < bound) bptr[w[i + 1]] = blk[w[i + 3]];
            if (op == OP_VARIABLE && wc >= 4 && w[i + 1] < bound && bptr[w[i + 1]] && w[i + 2] < bound) bvar[w[i + 2]] = bptr[w[i + 1]]; }
        for (size_t i = 5; ok && i < n; i += w[i] >> 16) { uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
            if ((op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN) && wc >= 5 && w[i + 3] < bound && bvar[w[i + 3]]) {
                uint32_t k = bvar[w[i + 3]] == 2 ? 5 : 4;   // the member index follows the vertex index of an arrayed block
                if (k < wc && (w[i + k] >= bound || !isc[w[i + k]] || cval[w[i + k]] == cull_m)) ok = 0;
            }
            if ((op == OP_LOAD || op == OP_COMPOSITE_CONSTRUCT || op == OP_COPY_OBJECT) && wc >= 3 && w[i + 1] < bound && blk[w[i + 1]]) ok = 0;
            if ((op == OP_STORE || op == OP_COPY_MEMORY) && wc >= 3 && ((w[i + 1] < bound && bvar[w[i + 1]]) || (w[i + 2] < bound && bvar[w[i + 2]]))) ok = 0;
        }
    }
    if (ok) {   // the struct loses its last word; the member's name and decorations go
        uint32_t *nw = malloc(n * 4);
        memcpy(nw, w, 20); out = 5;
        for (size_t i = 5; i < n; i += w[i] >> 16) {
            uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
            if ((op == OP_MEMBER_DECORATE || op == OP_MEMBER_NAME) && wc >= 3 && w[i + 1] == st && w[i + 2] == cull_m) continue;
            if (i == si) { nw[out] = ((wc - 1) << 16) | op; memcpy(nw + out + 1, w + i + 1, (wc - 2) * 4); out += wc - 1; continue; }
            memcpy(nw + out, w + i, wc * 4); out += wc;
        }
        free(w); *pw = nw;
    }
    free(def); free(cval); free(isc); free(blk); free(bptr); free(bvar);
    return ok ? out : (si && len[0] + len[1] > limit ? 0 : n);
}

typedef struct { uint32_t *w; size_t n, cap; } wbuf;
static void wput(wbuf *b, const uint32_t *x, size_t k) { if (b->n + k > b->cap) { b->cap = (b->n + k) * 2; b->w = realloc(b->w, b->cap * 4); } memcpy(b->w + b->n, x, k * 4); b->n += k; }

// NVN's compiler aggregates an atomic add over a warp: ballot(true), the ballot's highest lane leads and adds its bit count for all of them. After some lanes have
// returned, that ballot never completes on NVIDIA's RTX 50 cards (#12, #21, #22: shader d2c50cca6e386abc, driver 617). Where the add's result is never read, a
// ballot of only the lane's own bit gives the same sum: every lane leads itself and adds 1. Only that shape is rewritten: ballot(true) whose value reaches a
// FindUMsb / FindSMsb and a bit count, the bit count an OpAtomicIAdd whose result goes nowhere (or only into a variable nothing reads). The new word count
// (*pw replaced) and *rewritten ballots; n when nothing changed
size_t vkspv_ballot_per_lane(uint32_t **pw, size_t n, int *rewritten) {
    enum { B_NAME = 5, B_EXT_INST = 12, B_TYPE_INT = 21, B_TYPE_VECTOR = 23, B_TYPE_POINTER = 32, B_CONSTANT_TRUE = 41, B_CONSTANT = 43, B_FUNCTION = 54,
           B_VARIABLE = 59, B_LOAD = 61, B_STORE = 62, B_ACCESS_CHAIN = 65, B_IN_BOUNDS_ACCESS_CHAIN = 66, B_DECORATE = 71, B_COMPOSITE_CONSTRUCT = 80,
           B_COMPOSITE_EXTRACT = 81, B_COPY_OBJECT = 83, B_BITCAST = 124, B_SHL = 196, B_AND = 199, B_BIT_COUNT = 205, B_ATOMIC_IADD = 234, B_BALLOT = 339,
           B_DEC_BUILTIN = 11, B_BI_SUBGROUP_LANE = 41, B_FIND_SMSB = 74, B_FIND_UMSB = 75, MAXB = 32 };
    uint32_t *w = *pw;
    *rewritten = 0;
    if (!spv_ok(w, n)) return n;
    uint32_t bound = w[3], tu = 0, c0 = 0, c1 = 0, lane_var = 0, lane_ptr = 0, nb = 0, bres[MAXB], bat[MAXB];
    size_t first_fn = 0;
    uint32_t *def = calloc(bound, sizeof *def); uint8_t *is_true = calloc(bound, 1);
    for (size_t i = 5; i < n; ) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (!wc || i + wc > n) { free(def); free(is_true); return n; }
        if ((op == B_TYPE_INT || op == B_TYPE_VECTOR || op == B_TYPE_POINTER) && wc >= 2 && w[i + 1] < bound) def[w[i + 1]] = (uint32_t)i;
        if (op == B_VARIABLE && wc >= 4 && w[i + 2] < bound) def[w[i + 2]] = (uint32_t)i;
        if (op == B_TYPE_INT && wc == 4 && w[i + 2] == 32 && !w[i + 3] && !tu) tu = w[i + 1];
        if (op == B_CONSTANT_TRUE && wc == 3 && w[i + 2] < bound) is_true[w[i + 2]] = 1;
        if (op == B_CONSTANT && wc == 4 && tu && w[i + 1] == tu) { if (w[i + 3] == 0 && !c0) c0 = w[i + 2]; if (w[i + 3] == 1 && !c1) c1 = w[i + 2]; }
        if (op == B_DECORATE && wc == 4 && w[i + 2] == B_DEC_BUILTIN && w[i + 3] == B_BI_SUBGROUP_LANE) lane_var = w[i + 1];
        if (op == B_FUNCTION && !first_fn) first_fn = i;
        if (op == B_BALLOT && wc == 5 && w[i + 4] < bound && is_true[w[i + 4]] && nb < MAXB) { bres[nb] = w[i + 2]; bat[nb++] = (uint32_t)i; }
        i += wc;
    }
    // the lane's index must be an Input uint (as NVN's leader test reads it), the ballot a vector of 4 of them
    if (lane_var && lane_var < bound && def[lane_var]) { uint32_t pt = w[def[lane_var] + 1];
        if (pt < bound && def[pt] && (w[def[pt]] & 0xffff) == B_TYPE_POINTER && w[def[pt] + 3] == tu) lane_ptr = pt; }
    if (!nb || !tu || !lane_ptr || !first_fn) { free(def); free(is_true); return n; }
    // what each ballot's value reaches (bit k for ballot k): D its bits, C a bit count of them, through variables, access chains and bit casts
    uint32_t *D = calloc(bound, 4), *C = calloc(bound, 4), msb = 0, atom = 0, atom_used = 0;
    for (uint32_t k = 0; k < nb; k++) D[bres[k]] |= 1u << k;
    for (int pass = 0, changed = 1; changed && pass < 32; pass++) {
        changed = 0; atom = 0;
        #define FLOW(dst, d, c) do { uint32_t d_ = (d), c_ = (c); if ((D[dst] | d_) != D[dst] || (C[dst] | c_) != C[dst]) { D[dst] |= d_; C[dst] |= c_; changed = 1; } } while (0)
        for (size_t i = 5; i < n; i += w[i] >> 16) {
            uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
            if (op == B_STORE && wc >= 3 && w[i + 1] < bound && w[i + 2] < bound) FLOW(w[i + 1], D[w[i + 2]], C[w[i + 2]]);
            else if ((op == B_LOAD || op == B_ACCESS_CHAIN || op == B_IN_BOUNDS_ACCESS_CHAIN || op == B_COMPOSITE_EXTRACT || op == B_COPY_OBJECT || op == B_BITCAST)
                     && wc >= 4 && w[i + 2] < bound && w[i + 3] < bound) FLOW(w[i + 2], D[w[i + 3]], C[w[i + 3]]);
            else if (op == B_AND && wc == 5 && w[i + 2] < bound && w[i + 3] < bound && w[i + 4] < bound) FLOW(w[i + 2], D[w[i + 3]] | D[w[i + 4]], C[w[i + 3]] | C[w[i + 4]]);
            else if (op == B_BIT_COUNT && wc == 4 && w[i + 2] < bound && w[i + 3] < bound) FLOW(w[i + 2], 0, D[w[i + 3]] | C[w[i + 3]]);
            else if (op == B_EXT_INST && wc == 6 && (w[i + 4] == B_FIND_UMSB || w[i + 4] == B_FIND_SMSB) && w[i + 5] < bound) msb |= D[w[i + 5]];
            else if (op == B_ATOMIC_IADD && wc == 7 && w[i + 6] < bound && C[w[i + 6]]) {
                atom |= C[w[i + 6]];
                // its result read anywhere (beyond a store into a variable nothing else names) keeps those ballots as they are
                uint32_t r = w[i + 2], stored = 0;
                for (size_t j = 5; j < n && !(atom_used & C[w[i + 6]]) ; j += w[j] >> 16) {
                    if (j == i) continue;
                    uint32_t jc = w[j] >> 16, jo = w[j] & 0xffff;
                    for (uint32_t a = 1; a < jc; a++) if (w[j + a] == r) {
                        if (jo == B_STORE && a == 2 && !stored) stored = w[j + 1]; else atom_used |= C[w[i + 6]];
                        break;
                    }
                }
                if (stored) for (size_t j = 5; j < n && !(atom_used & C[w[i + 6]]); j += w[j] >> 16) {
                    uint32_t jc = w[j] >> 16, jo = w[j] & 0xffff;
                    if (jo == B_NAME || (jo == B_VARIABLE && w[j + 2] == stored) || (jo == B_STORE && w[j + 1] == stored)) continue;
                    for (uint32_t a = 1; a < jc; a++) if (w[j + a] == stored) { atom_used |= C[w[i + 6]]; break; }
                }
            }
        }
        #undef FLOW
    }
    uint32_t ok = msb & atom & ~atom_used, vt = 0;
    for (uint32_t k = 0; k < nb; k++) if (ok >> k & 1) {   // the ballot's type: a vector of 4 uints
        vt = w[bat[k] + 1];
        if (!(vt < bound && def[vt] && (w[def[vt]] & 0xffff) == B_TYPE_VECTOR && w[def[vt] + 2] == tu && w[def[vt] + 3] == 4)) ok &= ~(1u << k);
    }
    free(D); free(C); free(is_true);
    if (!ok) { free(def); return n; }
    // the rewrite: the constants 0 and 1 when missing (before the first function), each ballot → this lane's bit in .x
    wbuf o = {0};
    wput(&o, w, 5);
    for (size_t i = 5; i < n; ) {
        uint32_t wc = w[i] >> 16;
        if (i == first_fn) {
            if (!c0) { c0 = bound++; uint32_t c[4] = {4u << 16 | B_CONSTANT, tu, c0, 0}; wput(&o, c, 4); }
            if (!c1) { c1 = bound++; uint32_t c[4] = {4u << 16 | B_CONSTANT, tu, c1, 1}; wput(&o, c, 4); }
        }
        int k = -1;
        for (uint32_t b = 0; b < nb; b++) if (bat[b] == i && (ok >> b & 1)) k = (int)b;
        if (k < 0) { wput(&o, w + i, wc); i += wc; continue; }
        uint32_t lid = bound++, bit = bound++;
        uint32_t ld[4] = {4u << 16 | B_LOAD, tu, lid, lane_var}, sh[5] = {5u << 16 | B_SHL, tu, bit, c1, lid},
                 cc[7] = {7u << 16 | B_COMPOSITE_CONSTRUCT, w[i + 1], w[i + 2], bit, c0, c0, c0};
        wput(&o, ld, 4); wput(&o, sh, 5); wput(&o, cc, 7);
        (*rewritten)++;
        i += wc;
    }
    o.w[3] = bound;
    free(def); free(w);
    *pw = o.w;
    return o.n;
}

int vkspv_image_of(const uint32_t *w, size_t n, uint32_t binding, vkspv_img *o) {
    memset(o, 0, sizeof *o); o->binding = binding;
    if (!spv_ok(w, n)) return -1;
    uint32_t bound = w[3], var = 0;
    for (size_t i = 5; i < n; ) {   // the variable decorated with this binding
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (!wc || i + wc > n) return -1;
        if (op == OP_DECORATE && wc >= 4 && w[i + 2] == DEC_BINDING && w[i + 3] == binding) { var = w[i + 1]; break; }
        i += wc;
    }
    if (!var || var >= bound) return -1;
    // types by id: what each defines (opcode, operands' offset)
    uint32_t *def = calloc(bound, sizeof *def);   // index of the defining instruction
    for (size_t i = 5; i < n; ) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (!wc || i + wc > n) break;
        uint32_t res = 0;
        if (op >= OP_TYPE_INT && op <= OP_TYPE_POINTER && wc >= 2) res = w[i + 1];
        else if (op == OP_VARIABLE && wc >= 3) res = w[i + 2];
        if (res && res < bound) def[res] = (uint32_t)i;
        i += wc;
    }
    int rc = -1;
    uint32_t vi = def[var];
    if (vi && (w[vi] & 0xffff) == OP_VARIABLE) {
        uint32_t t = w[vi + 1];
        for (int hop = 0; hop < 8 && t < bound && def[t]; hop++) {
            uint32_t d = def[t], op = w[d] & 0xffff;
            if (op == OP_TYPE_POINTER) t = w[d + 3];
            else if (op == OP_TYPE_ARRAY || op == OP_TYPE_RUNTIME_ARRAY) t = w[d + 2];
            else if (op == OP_TYPE_SAMPLED_IMAGE) t = w[d + 2];
            else if (op == OP_TYPE_IMAGE) {
                o->is_image = 1; o->dim = (uint8_t)w[d + 3]; o->depth = w[d + 4] == 1; o->arrayed = (uint8_t)w[d + 5]; o->ms = (uint8_t)w[d + 6];
                uint32_t st = w[d + 2];
                if (st < bound && def[st]) { uint32_t sd = def[st]; o->ct = (w[sd] & 0xffff) == OP_TYPE_INT ? (w[sd + 3] ? VKS_CT_INT : VKS_CT_UINT) : VKS_CT_FLOAT; }
                break;
            } else break;
        }
        o->found = 1; rc = 0;
        // how the functions sample it: loads of the variable, sampled images made from them, then the sample instructions that read those
        uint32_t ids[64]; int nid = 0; ids[nid++] = var;
        for (size_t i = 5; i < n; ) {
            uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
            if (!wc || i + wc > n) break;
            int in = 0;
            if ((op == 61 /* OpLoad */ || op == 86 /* OpSampledImage */ || op == 100 /* OpImage */ || op == 124 /* OpBitcast */ || op == 83 /* OpCopyObject */) && wc >= 4)
                for (int k = 0; k < nid; k++) in |= w[i + 3] == ids[k];
            if (in && nid < 64) ids[nid++] = w[i + 2];
            if (wc >= 4) {
                int dref = op == 89 || op == 90 || op == 91 || op == 92 || op == 97 || op == 307 || op == 308 || op == 311 || op == 312 || op == 316;
                int plain = op == 87 || op == 88 || op == 93 || op == 94 || op == 96 || op == 305 || op == 306 || op == 309 || op == 310 || op == 315;
                if (dref || plain) for (int k = 1; k < nid; k++) if (w[i + 3] == ids[k]) { if (dref) o->dref = 1; else o->plain = 1; }
            }
            i += wc;
        }
    }
    free(def);
    return rc;
}

static VkImageViewType view_type(uint8_t dim, uint8_t arrayed) {
    switch (dim) {
    case 0: return arrayed ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
    case 2: return VK_IMAGE_VIEW_TYPE_3D;
    case 3: return arrayed ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
    default: return arrayed ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;   // 2D, rect
    }
}

// ---------- Uniform array indexes held to the array (NXRT_VK_UBO_CLAMP) ----------
// An access chain into a uniform block's array with an index computed at run time gets UMin(index, length - 1) in front of it. The translator reads every
// constant buffer as a 4,096-entry vec4 array (64 KB) at an address it computes; past the array is past the whole 64 KB block, where robust buffer access
// gives no promise. Returns a new module, or NULL when nothing needed it.
uint32_t *vkspv_clamp_ubo(const uint32_t *w, size_t n, size_t *out_n, int *count) {
    *count = 0;
    if (!spv_ok(w, n)) return NULL;
    uint32_t bound = w[3];
    uint32_t *ty = calloc(bound, 4), *a1 = calloc(bound, 4), *a2 = calloc(bound, 4);   // per id: opcode that made it (types, constants, variables), and two operands
    uint32_t **members = calloc(bound, sizeof *members), *rt = NULL;
    uint32_t glsl = 0; size_t first_fn = 0, imp_at = 0;
    if (!ty || !a1 || !a2 || !members) { free(ty); free(a1); free(a2); free(members); return NULL; }
    for (size_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16; if (!wc || i + wc > n) goto bad;
        if (op == 11 && wc >= 3 && !memcmp(&w[i + 2], "GLSL.std.450", 12)) glsl = w[i + 1];
        if (op == 10 || op == 17) imp_at = i + wc;   // after the capabilities and extensions: where an import would go
        if (op == 11) imp_at = i + wc;
        if (op == 54 && !first_fn) first_fn = i;
        if (!first_fn) {
            uint32_t r = 0;
            if (op == 21 || op == 28 || op == 30 || op == 32) r = w[i + 1];        // int, array, struct, pointer: result id first
            else if (op == 43 || op == 59) r = w[i + 2];                             // constant, variable: type first
            if (r && r < bound) {
                ty[r] = op;
                if (op == 21) { a1[r] = w[i + 2]; a2[r] = w[i + 3]; }                // width, signedness
                if (op == 28) { a1[r] = w[i + 2]; a2[r] = w[i + 3]; }                // element, length id
                if (op == 32) { a1[r] = w[i + 2]; a2[r] = w[i + 3]; }                // storage class, pointee
                if (op == 43) { a1[r] = w[i + 1]; a2[r] = w[i + 3]; }                // type, value (low word)
                if (op == 59) { a1[r] = w[i + 1]; a2[r] = w[i + 3]; }                // pointer type, storage class
                if (op == 30) { members[r] = calloc(wc, 4); if (members[r]) { members[r][0] = wc - 2; memcpy(members[r] + 1, &w[i + 2], (wc - 2) * 4); } }
            }
        }
        i += wc;
    }
    if (!first_fn) goto bad;
    // the type of each value made in the functions: (type, result) instructions, the first one naming an id (definitions come before their uses)
    rt = calloc(bound, 4); if (!rt) goto bad;
    for (size_t i = first_fn; i < n;) {
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
        if (wc >= 3 && op != 62 && op != 246 && op != 247 && op != 250 && op != 251 && op != 8 && op != 248 && w[i + 2] < bound && !rt[w[i + 2]]) rt[w[i + 2]] = w[i + 1];
        i += wc;
    }
    // new constants (index type, length - 1), made once each; new ids from the bound
    enum { NK = 64 }; uint32_t kt[NK], kv[NK], kid[NK]; int nk = 0;
    size_t cap = n + n / 2 + 64, k = 0; uint32_t *o = malloc(cap * 4); if (!o) goto bad;
    uint32_t nb = bound; int need_imp = !glsl; if (need_imp) glsl = nb++;
    // first pass over the functions: count, to size the output and make the constants
    for (int pass = 0; pass < 2; pass++) {
        k = 0;
        if (pass) {
            memcpy(o, w, 20); k = 5;
            for (size_t i = 5; i < first_fn;) {
                uint32_t wc = w[i] >> 16;
                if (need_imp && i == imp_at) { o[k++] = (6u << 16) | 11; o[k++] = glsl; memcpy(&o[k], "GLSL.std.450\0\0\0\0", 16); k += 4; }
                memcpy(&o[k], &w[i], wc * 4); k += wc; i += wc;
            }
            for (int c = 0; c < nk; c++) { o[k++] = (4u << 16) | 43; o[k++] = kt[c]; o[k++] = kid[c]; o[k++] = kv[c]; }
        }
        for (size_t i = first_fn; i < n;) {
            uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
            size_t at = k;
            if (pass) { memcpy(&o[k], &w[i], wc * 4); k += wc; }
            if ((op == 65 || op == 66) && wc > 4) {   // OpAccessChain, OpInBoundsAccessChain
                uint32_t base = w[i + 3];
                if (base < bound && ty[base] == 59 && a2[base] == 2 /* Uniform */ && a1[base] < bound && ty[a1[base]] == 32) {
                    uint32_t cur = a2[a1[base]];   // the block's type
                    for (uint32_t j = 4; j < wc && cur < bound; j++) {
                        uint32_t idx = w[i + j];
                        if (ty[cur] == 30) {   // struct: a constant member index
                            if (idx >= bound || ty[idx] != 43 || !members[cur] || a2[idx] >= members[cur][0]) { cur = 0; break; }
                            cur = members[cur][1 + a2[idx]]; continue; }
                        if (ty[cur] != 28) break;
                        uint32_t len = a2[cur] < bound && ty[a2[cur]] == 43 ? a2[a2[cur]] : 0;
                        if (len && idx < bound && ty[idx] != 43) {   // computed at run time
                            uint32_t it = rt[idx];   // the index's type
                            if (it && it < bound && ty[it] == 21 && a1[it] == 32) {
                                int c = 0; while (c < nk && !(kt[c] == it && kv[c] == len - 1)) c++;
                                if (c == nk && nk < NK) { kt[nk] = it; kv[nk] = len - 1; kid[nk] = nb++; nk++; }
                                if (c < nk) {
                                    if (!pass) (*count)++;
                                    else {   // UMin before the chain, the chain reads its result
                                        uint32_t t = nb++;
                                        memmove(&o[at + 7], &o[at], wc * 4);
                                        o[at] = (7u << 16) | 12; o[at + 1] = it; o[at + 2] = t; o[at + 3] = glsl; o[at + 4] = 38 /* GLSL.std.450 UMin: a negative index clamps too */; o[at + 5] = idx; o[at + 6] = kid[c];
                                        k += 7; at += 7; o[at + j] = t;
                                        if (k + 64 > cap) { cap *= 2; uint32_t *no = realloc(o, cap * 4); if (!no) { free(o); goto bad; } o = no; }
                                    }
                                }
                            }
                        }
                        cur = a1[cur];
                    }
                }
            }
            if (pass && k + 64 > cap) { cap *= 2; uint32_t *no = realloc(o, cap * 4); if (!no) { free(o); goto bad; } o = no; }
            i += wc;
        }
        if (!*count) { free(o); o = NULL; break; }
        if (!pass) { size_t need = n + (size_t)*count * 7 + (size_t)nk * 4 + 16; if (need > cap) { cap = need; uint32_t *no = realloc(o, cap * 4); if (!no) { free(o); goto bad; } o = no; } }
    }
    if (o) { o[3] = nb; *out_n = k; }
    for (uint32_t i = 0; i < bound; i++) free(members[i]);
    free(ty); free(a1); free(a2); free(members); free(rt);
    return o;
bad:
    if (members) for (uint32_t i = 0; i < bound; i++) free(members[i]);
    free(ty); free(a1); free(a2); free(members); free(rt); *count = 0;
    return NULL;
}

// ---------- Vertex input retyping (vksh_vin_variant) ----------
enum { OP_TYPE_VECTOR = 23, OP_ACCESS_CHAIN = 65, OP_IN_BOUNDS_ACCESS_CHAIN = 66, OP_LOAD = 61, OP_BITCAST = 124, DEC_LOCATION = 30, SC_INPUT = 1 };
uint32_t *vkspv_retype_inputs(const uint32_t *w, size_t n, uint32_t mask, const uint8_t ct[32], size_t *out_n) {
    if (!spv_ok(w, n) || !mask) return NULL;
    uint32_t bound = w[3];
    // what the module defines: scalar types by class, vectors, input pointers, the input variables by location
    uint32_t *def = calloc(bound, sizeof *def), loc_of_var[32] = {0}, scal[3] = {0}, first_var = 0;
    for (size_t i = 5; i < n; ) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (!wc || i + wc > n) { free(def); return NULL; }
        if (op == OP_DECORATE && wc >= 4 && w[i + 2] == DEC_LOCATION && w[i + 3] < 32) loc_of_var[w[i + 3]] = w[i + 1];   // the last one wins; inputs are checked below
        uint32_t res = 0;
        if (op >= OP_TYPE_INT && op <= OP_TYPE_POINTER && wc >= 2) res = w[i + 1];
        else if (op == OP_VARIABLE && wc >= 4) { res = w[i + 2]; if (!first_var && w[i + 3] != 7) first_var = (uint32_t)i; }
        if (res && res < bound) def[res] = (uint32_t)i;
        if (op == OP_TYPE_INT && wc == 4 && w[i + 2] == 32) scal[w[i + 3] ? VKS_CT_INT : VKS_CT_UINT] = w[i + 1];
        if (op == OP_TYPE_FLOAT && wc == 3 && w[i + 2] == 32) scal[VKS_CT_FLOAT] = w[i + 1];
        i += wc;
    }
    // the input variables to retype (their Location decorations may sit on outputs too: only Input variables count)
    uint32_t var_of[32] = {0}, var_ct[32];
    for (int l = 0; l < 32; l++) {
        if (!(mask >> l & 1)) continue;
        uint32_t v = 0;
        for (size_t i = 5; i < n; i += w[i] >> 16) if ((w[i] & 0xffff) == OP_DECORATE && (w[i] >> 16) >= 4 && w[i + 2] == DEC_LOCATION && w[i + 3] == (uint32_t)l) {
            uint32_t c = w[i + 1]; if (c < bound && def[c] && (w[def[c]] & 0xffff) == OP_VARIABLE && w[def[c] + 3] == SC_INPUT) v = c; }
        if (v) { var_of[l] = v; var_ct[l] = ct[l]; }
    }
    (void)loc_of_var;
    // the new types go right before the first variable retyped; the existing types they name must be defined before that
    first_var = 0;
    for (int l = 0; l < 32; l++) if (var_of[l] && (!first_var || def[var_of[l]] < first_var)) first_var = def[var_of[l]];
    if (!first_var) { free(def); return NULL; }
    for (int k = 0; k < 3; k++) if (scal[k] && def[scal[k]] > first_var) { free(def); return NULL; }
    // new types, placed before the first global variable: the scalars of each class, vectors of them, input pointers to both
    wbuf types = {0}; uint32_t vec_of[3][5] = {{0}}, pv[3][5] = {{0}}, ps[3] = {0};
    for (size_t i = 5; i < n; i += w[i] >> 16) {   // existing vectors and input pointers
        uint32_t op = w[i] & 0xffff;
        for (int k = 0; k < 3; k++) {
            if (op == OP_TYPE_VECTOR && w[i + 2] == scal[k] && w[i + 3] <= 4) vec_of[k][w[i + 3]] = w[i + 1];
            if (op == OP_TYPE_POINTER && w[i + 2] == SC_INPUT && w[i + 3] == scal[k] && scal[k]) ps[k] = w[i + 1];
        }
    }
    for (size_t i = 5; i < n; i += w[i] >> 16) if ((w[i] & 0xffff) == OP_TYPE_POINTER && w[i + 2] == SC_INPUT)
        for (int k = 0; k < 3; k++) for (int c = 2; c <= 4; c++) if (vec_of[k][c] && w[i + 3] == vec_of[k][c]) pv[k][c] = w[i + 1];
    #define NEWT(...) do { uint32_t t_[] = {__VA_ARGS__}; wput(&types, t_, sizeof t_ / 4); } while (0)
    for (int l = 0; l < 32; l++) {
        if (!var_of[l]) continue;
        int k = var_ct[l];
        if (!scal[k]) { scal[k] = bound++; if (k == VKS_CT_FLOAT) NEWT(3u << 16 | OP_TYPE_FLOAT, scal[k], 32); else NEWT(4u << 16 | OP_TYPE_INT, scal[k], 32, k == VKS_CT_INT); }
        if (!ps[k]) { ps[k] = bound++; NEWT(4u << 16 | OP_TYPE_POINTER, ps[k], SC_INPUT, scal[k]); }
        uint32_t pt = w[def[var_of[l]] + 1], pointee = w[def[pt] + 3], pop = w[def[pointee]] & 0xffff, nc = pop == OP_TYPE_VECTOR ? w[def[pointee] + 3] : 1;
        if (nc > 1 && nc <= 4) {
            if (!vec_of[k][nc]) { vec_of[k][nc] = bound++; NEWT(4u << 16 | OP_TYPE_VECTOR, vec_of[k][nc], scal[k], nc); }
            if (!pv[k][nc]) { pv[k][nc] = bound++; NEWT(4u << 16 | OP_TYPE_POINTER, pv[k][nc], SC_INPUT, vec_of[k][nc]); }
        }
    }
    #undef NEWT
    // the rewrite: variables take the new pointer type; access chains into them the scalar pointer; loads through either load the new type and bit-cast
    uint32_t *repl = calloc(bound + 1, sizeof *repl);   // pointer id → class + 1 of what it now points to
    wbuf o = {0};
    wput(&o, w, 5);
    for (size_t i = 5; i < n; ) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (i == first_var && types.n) wput(&o, types.w, types.n);
        uint32_t x[64]; memcpy(x, w + i, (wc < 64 ? wc : 64) * 4);
        if (wc > 64) { wput(&o, w + i, wc); i += wc; continue; }
        if (op == OP_VARIABLE) {
            for (int l = 0; l < 32; l++) if (var_of[l] && x[2] == var_of[l]) {
                int k = var_ct[l]; uint32_t pointee = w[def[x[1]] + 3], nc = (w[def[pointee]] & 0xffff) == OP_TYPE_VECTOR ? w[def[pointee] + 3] : 1;
                x[1] = nc > 1 ? pv[k][nc] : ps[k]; repl[x[2]] = (uint32_t)k + 1;
            }
            wput(&o, x, wc);
        } else if ((op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN) && x[3] < bound && repl[x[3]]) {
            int k = (int)repl[x[3]] - 1; x[1] = ps[k]; repl[x[2]] = (uint32_t)k + 1;   // one index into a vector: the scalar
            wput(&o, x, wc);
        } else if (op == OP_LOAD && x[3] < bound && repl[x[3]]) {
            int k = (int)repl[x[3]] - 1; uint32_t want = x[1], old = x[2], nid = bound++;
            uint32_t wop = w[def[want]] & 0xffff, nc = wop == OP_TYPE_VECTOR ? w[def[want] + 3] : 1;
            x[1] = nc > 1 ? vec_of[k][nc] : scal[k]; x[2] = nid;
            wput(&o, x, wc);
            uint32_t bc[4] = {4u << 16 | OP_BITCAST, want, old, nid}; wput(&o, bc, 4);
        } else wput(&o, x, wc);
        i += wc;
    }
    o.w[3] = bound;
    free(def); free(repl); free(types.w);
    *out_n = o.n;
    return o.w;
}
static const vks_shader *vin_variant_l(const vks_shader *vs_, uint32_t mask, const uint8_t ct[32]) {
    vks_shader *vs = (vks_shader *)vs_;
    if (!mask || !vs->words) return vs;
    uint64_t key = mask; for (int l = 0; l < 32; l++) if (mask >> l & 1) key = key * 31 + ct[l] + 1;
    for (int i = 0; i < vs->nvar; i++) if (vs->var[i].key == key) return vs->var[i].s;
    if (vs->nvar == 8) { static int n; if (n++ < 4) VKRED("shader %s: more than 8 vertex input variants", vs->key); return vs; }
    size_t nn; uint32_t *nw = vkspv_retype_inputs(vs->words, vs->nwords, mask, ct, &nn);
    VkShaderModule mod = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo mci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, nn * 4, nw};
    if (!nw || vk_module(&mci, &mod, vs->key, "vertex inputs retyped") != VK_SUCCESS) { free(nw); static int n; if (n++ < 4) VKRED("shader %s: its inputs %#x could not be retyped", vs->key, mask); return vs; }
    vks_shader *s = malloc(sizeof *s); *s = *vs;
    s->mod = mod; s->words = nw; s->nwords = nn; s->base = vs; s->nvar = 0;
    for (int i = 0; i < s->nin; i++) if (mask >> s->in[i].loc & 1) s->in[i].ct = ct[s->in[i].loc];
    s->vin_mask = mask; memcpy(s->vin_ct, ct, sizeof s->vin_ct);
    vs->var[vs->nvar].key = key; vs->var[vs->nvar].s = s; vs->nvar++;
    static int nl; if (nl++ < 16) VKLOG("vertex shader %s: inputs %#x declared as their formats' types (variant %d)", vs->key, mask, vs->nvar);
    return s;
}
const vks_shader *vksh_vin_variant(const vks_shader *vs, uint32_t mask, const uint8_t ct[32]) {
    if (!mask) return vs;
    AcquireSRWLockExclusive(&g_shl); const vks_shader *r = vin_variant_l(vs, mask, ct); ReleaseSRWLockExclusive(&g_shl);
    return r;
}

// ---------- Fragment variants (M6 phase B): dual-source blending, the alpha test ----------
enum { OP_ENTRY_POINT = 15, OP_TYPE_VOID = 19, OP_TYPE_BOOL = 20, OP_TYPE_STRUCT = 30, OP_CONSTANT = 43, OP_FUNCTION = 54, OP_FUNCTION_END = 56, OP_MEMBER_DECORATE = 72,
       OP_ISUB = 130, OP_LOGICAL_OR = 166, OP_LOGICAL_AND = 167, OP_INOT_EQUAL = 171, OP_FORD_EQUAL = 180, OP_FORD_LESS = 184, OP_FORD_GREATER = 186, OP_BITWISE_AND = 199,
       OP_SELECTION_MERGE = 247, OP_LABEL = 248, OP_BRANCH_COND = 250, OP_KILL = 252, OP_RETURN = 253,
       DEC_BLOCK = 2, DEC_BUILTIN = 11, DEC_INDEX = 32, DEC_OFFSET = 35, SC_OUTPUT = 3, SC_PUSH = 9 };
// The non-builtin Output variables by Location (-1 when absent; -2 when a location holds more than one; a second blend source, Index 1, is not counted); returns how many
static int spv_outputs(const uint32_t *w, size_t n, int32_t var_at[8]) {
    uint32_t bound = w[3]; int cnt = 0;
    uint8_t *out = calloc(bound, 1), *bi = calloc(bound, 1);
    for (size_t i = 5; i < n; i += w[i] >> 16) {
        uint32_t op = w[i] & 0xffff, wc = w[i] >> 16;
        if (op == OP_VARIABLE && wc >= 4 && w[i + 3] == SC_OUTPUT && w[i + 2] < bound) out[w[i + 2]] = 1;
        if (op == OP_DECORATE && wc >= 3 && (w[i + 2] == DEC_BUILTIN || (w[i + 2] == DEC_INDEX && wc >= 4 && w[i + 3] == 1)) && w[i + 1] < bound) bi[w[i + 1]] = 1;
    }
    for (int l = 0; l < 8; l++) var_at[l] = -1;
    for (size_t i = 5; i < n; i += w[i] >> 16)
        if ((w[i] & 0xffff) == OP_DECORATE && (w[i] >> 16) >= 4 && w[i + 2] == DEC_LOCATION && w[i + 1] < bound && out[w[i + 1]] && !bi[w[i + 1]] && w[i + 3] < 8) {
            uint32_t l = w[i + 3]; var_at[l] = var_at[l] == -1 ? (int32_t)w[i + 1] : -2; cnt++;
        }
    free(out); free(bi);
    return cnt;
}
// Dual-source blending (nxshader's DualSource pass, nxnative's fs_dual): output 1 becomes Location 0 Index 1, the second source of target 0. Only for a shader writing
// exactly outputs 0 and 1. New words or NULL
uint32_t *vkspv_dual_source(const uint32_t *w, size_t n, size_t *out_n) {
    if (!spv_ok(w, n)) return NULL;
    int32_t at[8]; if (spv_outputs(w, n, at) != 2 || at[0] < 0 || at[1] < 0) return NULL;
    wbuf o = {0}; wput(&o, w, 5);
    for (size_t i = 5; i < n; i += w[i] >> 16) {
        uint32_t wc = w[i] >> 16;
        if ((w[i] & 0xffff) == OP_DECORATE && wc == 4 && w[i + 1] == (uint32_t)at[1] && w[i + 2] == DEC_LOCATION) {
            uint32_t d[8] = {4u << 16 | OP_DECORATE, w[i + 1], DEC_LOCATION, 0, 4u << 16 | OP_DECORATE, w[i + 1], DEC_INDEX, 1}; wput(&o, d, 8);
        } else wput(&o, w + i, wc);
    }
    *out_n = o.n; return o.w;
}
// The alpha test (nxnative's dmsh_alpha.py library, as nvn_mtl.m at_fn / at_bind use it): before each return of the entry point, colour 0's alpha is compared with the
// reference by NVN's alpha function, both in a push constant block {uint function; float reference;} (fragment stage, offset 0, 8 bytes), and the fragment is discarded
// when the comparison fails. The function and reference are data, so one pipeline serves every value. Functions NEVER = 1 … ALWAYS = 8: function − 1 is a mask of
// less (1), equal (2), greater (4). Needs colour 0 as a float vec4 output. New words or NULL
uint32_t *vkspv_alpha_test(const uint32_t *w, size_t n, size_t *out_n) {
    if (!spv_ok(w, n)) return NULL;
    int32_t at[8]; spv_outputs(w, n, at);
    if (at[0] < 0) return NULL;
    uint32_t bound = w[3], *def = calloc(bound, sizeof *def), entry = 0, f32 = 0, u32 = 0, tbool = 0, pof = 0;
    size_t first_type = 0, first_fn = 0;
    for (size_t i = 5; i < n; ) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (!wc || i + wc > n) { free(def); return NULL; }
        if (op >= OP_TYPE_VOID && op <= OP_TYPE_POINTER && wc >= 2 && w[i + 1] < bound) { def[w[i + 1]] = (uint32_t)i; if (!first_type) first_type = i; }
        if (op == OP_VARIABLE && wc >= 4 && w[i + 2] < bound) def[w[i + 2]] = (uint32_t)i;
        if (op == OP_ENTRY_POINT && wc >= 3 && !entry) entry = w[i + 2];
        if (op == OP_TYPE_FLOAT && wc == 3 && w[i + 2] == 32) f32 = w[i + 1];
        if (op == OP_TYPE_INT && wc == 4 && w[i + 2] == 32 && !w[i + 3]) u32 = w[i + 1];
        if (op == OP_TYPE_BOOL && wc == 2) tbool = w[i + 1];
        if (op == OP_FUNCTION && !first_fn) first_fn = i;
        i += wc;
    }
    // colour 0: an Output pointer to a vector of 4 floats
    uint32_t vi = def[at[0]], pt = vi ? w[vi + 1] : 0, pti = pt < bound ? def[pt] : 0, vt = pti && (w[pti] & 0xffff) == OP_TYPE_POINTER ? w[pti + 3] : 0, vti = vt < bound ? def[vt] : 0;
    int ok = entry && f32 && first_type && first_fn && vti && (w[vti] & 0xffff) == OP_TYPE_VECTOR && w[vti + 2] == f32 && w[vti + 3] == 4;
    for (size_t i = 5; ok && i < n; i += w[i] >> 16)   // an existing Output float pointer (a second declaration of a non-aggregate type is invalid)
        if ((w[i] & 0xffff) == OP_TYPE_POINTER && w[i + 2] == SC_OUTPUT && w[i + 3] == f32) pof = w[i + 1];
    free(def);
    if (!ok) return NULL;
    // new ids, declarations and decorations
    wbuf ty = {0}, dec = {0};
    #define T_(...) do { uint32_t t_[] = {__VA_ARGS__}; wput(&ty, t_, sizeof t_ / 4); } while (0)
    if (!u32) { u32 = bound++; T_(4u << 16 | OP_TYPE_INT, u32, 32, 0); }
    if (!tbool) { tbool = bound++; T_(2u << 16 | OP_TYPE_BOOL, tbool); }
    if (!pof) { pof = bound++; T_(4u << 16 | OP_TYPE_POINTER, pof, SC_OUTPUT, f32); }
    uint32_t st = bound++, pst = bound++, pu = bound++, pfl = bound++, var = bound++, c[5];
    T_(4u << 16 | OP_TYPE_STRUCT, st, u32, f32); T_(4u << 16 | OP_TYPE_POINTER, pst, SC_PUSH, st);
    T_(4u << 16 | OP_TYPE_POINTER, pu, SC_PUSH, u32); T_(4u << 16 | OP_TYPE_POINTER, pfl, SC_PUSH, f32);
    T_(4u << 16 | OP_VARIABLE, pst, var, SC_PUSH);
    for (uint32_t k = 0; k < 5; k++) { c[k] = bound++; T_(4u << 16 | OP_CONSTANT, u32, c[k], k); }
    #undef T_
    uint32_t d[] = {3u << 16 | OP_DECORATE, st, DEC_BLOCK, 5u << 16 | OP_MEMBER_DECORATE, st, 0, DEC_OFFSET, 0, 5u << 16 | OP_MEMBER_DECORATE, st, 1, DEC_OFFSET, 4};
    wput(&dec, d, sizeof d / 4);
    // the rewrite: decorations before the first type, declarations before the first function, the test before each return of the entry point
    wbuf o = {0}; wput(&o, w, 5);
    uint32_t in_fn = 0;
    for (size_t i = 5; i < n; i += w[i] >> 16) {
        uint32_t wc = w[i] >> 16, op = w[i] & 0xffff;
        if (i == first_type) wput(&o, dec.w, dec.n);
        if (i == first_fn) wput(&o, ty.w, ty.n);
        if (op == OP_FUNCTION) in_fn = w[i + 2];
        if (op == OP_RETURN && in_fn == entry) {
            uint32_t pf = bound++, f = bound++, pr = bound++, r = bound++, pa = bound++, a = bound++, m = bound++, lt = bound++, eq = bound++, gt = bound++;
            uint32_t b0 = bound++, h0 = bound++, p0 = bound++, b1 = bound++, h1 = bound++, p1 = bound++, b2 = bound++, h2 = bound++, p2 = bound++, q = bound++, pass = bound++;
            uint32_t ok_l = bound++, kill_l = bound++;
            uint32_t s[] = {
                5u << 16 | OP_ACCESS_CHAIN, pu, pf, var, c[0],   4u << 16 | OP_LOAD, u32, f, pf,
                5u << 16 | OP_ACCESS_CHAIN, pfl, pr, var, c[1],  4u << 16 | OP_LOAD, f32, r, pr,
                5u << 16 | OP_ACCESS_CHAIN, pof, pa, (uint32_t)at[0], c[3], 4u << 16 | OP_LOAD, f32, a, pa,
                5u << 16 | OP_ISUB, u32, m, f, c[1],
                5u << 16 | OP_FORD_LESS, tbool, lt, a, r, 5u << 16 | OP_FORD_EQUAL, tbool, eq, a, r, 5u << 16 | OP_FORD_GREATER, tbool, gt, a, r,
                5u << 16 | OP_BITWISE_AND, u32, b0, m, c[1], 5u << 16 | OP_INOT_EQUAL, tbool, h0, b0, c[0], 5u << 16 | OP_LOGICAL_AND, tbool, p0, h0, lt,
                5u << 16 | OP_BITWISE_AND, u32, b1, m, c[2], 5u << 16 | OP_INOT_EQUAL, tbool, h1, b1, c[0], 5u << 16 | OP_LOGICAL_AND, tbool, p1, h1, eq,
                5u << 16 | OP_BITWISE_AND, u32, b2, m, c[4], 5u << 16 | OP_INOT_EQUAL, tbool, h2, b2, c[0], 5u << 16 | OP_LOGICAL_AND, tbool, p2, h2, gt,
                5u << 16 | OP_LOGICAL_OR, tbool, q, p0, p1, 5u << 16 | OP_LOGICAL_OR, tbool, pass, q, p2,
                3u << 16 | OP_SELECTION_MERGE, ok_l, 0, 4u << 16 | OP_BRANCH_COND, pass, ok_l, kill_l,
                2u << 16 | OP_LABEL, kill_l, 1u << 16 | OP_KILL, 2u << 16 | OP_LABEL, ok_l};
            wput(&o, s, sizeof s / 4);
        }
        if (op == OP_FUNCTION_END) in_fn = 0;
        if (op == OP_ENTRY_POINT && w[i + 2] == entry && w[1] >= 0x00010400) {   // SPIR-V 1.4 on: an entry point lists every global it uses
            uint32_t h = (wc + 1) << 16 | op; wput(&o, &h, 1); wput(&o, w + i + 1, wc - 1); wput(&o, &var, 1); continue;
        }
        wput(&o, w + i, wc);
    }
    o.w[3] = bound;
    free(ty.w); free(dec.w);
    *out_n = o.n; return o.w;
}
static const vks_shader *fs_variant_l(const vks_shader *fs_, int flags) {
    vks_shader *fs = (vks_shader *)fs_;
    if (!flags || !fs->words || fs->stage != VKS_FS) return fs;
    uint64_t key = 0x100u | (unsigned)flags;
    for (int i = 0; i < fs->nvar; i++) if (fs->var[i].key == key) return fs->var[i].s;
    if (fs->nvar == 8) return fs;
    size_t n = fs->nwords, nn = 0; uint32_t *w = NULL; int dual = 0, alpha = 0;
    if (flags & VKSH_FS_DUAL) {
        if ((w = vkspv_dual_source(fs->words, n, &nn))) { n = nn; dual = 1; }
        else { static int k; if (k++ < 8) VKRED("shader %s: no dual-source variant (it does not write exactly outputs 0 and 1): the second source counts as zero", fs->key); }
    }
    if (flags & VKSH_FS_ALPHA) {
        uint32_t *a = vkspv_alpha_test(w ? w : fs->words, n, &nn);
        if (a) { free(w); w = a; n = nn; alpha = 1; }
        else { static int k; if (k++ < 8) VKRED("shader %s: no alpha test variant (its colour 0 is not a float vec4): drawn without the test", fs->key); }
    }
    VkShaderModule mod = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo mci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, n * 4, w};
    if (!w || vk_module(&mci, &mod, fs->key, dual && alpha ? "dual source + alpha test" : dual ? "dual source" : "alpha test") != VK_SUCCESS) { free(w); fs->var[fs->nvar].key = key; fs->var[fs->nvar].s = fs; fs->nvar++; return fs; }   // the base, remembered
    vks_shader *s = malloc(sizeof *s); *s = *fs;
    s->mod = mod; s->words = w; s->nwords = n; s->base = fs; s->nvar = 0; s->dual = dual; s->alpha = alpha; s->fs_flags = (uint8_t)flags;
    if (dual) s->out_mask = fs->out_mask & ~2u;   // output 1 is target 0's second source now, not a target of its own
    fs->var[fs->nvar].key = key; fs->var[fs->nvar].s = s; fs->nvar++;
    static int nl; if (nl++ < 16) VKLOG("fragment shader %s: %s%s%s variant", fs->key, dual ? "dual-source" : "", dual && alpha ? " + " : "", alpha ? "alpha test" : "");
    return s;
}
// A pixel shader that draws a number as a flat colour: 1 + n / 32 in each channel, 5 bits a channel, which R11G11B10F keeps exactly (NXRT_VK_DUMP_AT_S, tools/pass_pair.py)
const vks_shader *vksh_id_variant(const vks_shader *fs) {
    enum { MAXID = 4096 }; static const vks_shader *from[MAXID], *to[MAXID]; static int n;
    AcquireSRWLockExclusive(&g_shl);
    const vks_shader *r = NULL;
    for (int i = 0; i < n && !r; i++) if (from[i] == fs) r = to[i];
    if (!r && n < MAXID) {
        uint32_t id = (uint32_t)n, bits[3];
        for (int i = 0; i < 3; i++) { float v = 1.0f + (float)((id >> (5 * i)) & 31) / 32.0f; memcpy(&bits[i], &v, 4); }
        static const uint32_t head[] = {0x07230203, 0x00010300, 0, 14, 0, 0x00020011, 1, 0x0003000E, 0, 1, 0x0006000F, 4, 1, 0x6E69616D, 0, 2, 0x00030010, 1, 7, 0x00040047, 2, 30, 0,
                                        0x00020013, 3, 0x00030021, 4, 3, 0x00030016, 5, 32, 0x00040017, 6, 5, 4, 0x00040020, 7, 3, 6, 0x0004003B, 7, 2, 3};
        enum { HN = sizeof head / 4 };
        uint32_t m[HN + 36], k = HN; memcpy(m, head, sizeof head);
        for (int i = 0; i < 3; i++) { m[k++] = 0x0004002B; m[k++] = 5; m[k++] = 8 + (uint32_t)i; m[k++] = bits[i]; }
        m[k++] = 0x0004002B; m[k++] = 5; m[k++] = 11; m[k++] = 0x3F800000;
        m[k++] = 0x0007002C; m[k++] = 6; m[k++] = 12; m[k++] = 8; m[k++] = 9; m[k++] = 10; m[k++] = 11;
        m[k++] = 0x00050036; m[k++] = 3; m[k++] = 1; m[k++] = 0; m[k++] = 4; m[k++] = 0x000200F8; m[k++] = 13; m[k++] = 0x0003003E; m[k++] = 2; m[k++] = 12; m[k++] = 0x000100FD; m[k++] = 0x00010038;
        VkShaderModuleCreateInfo mci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, k * 4, m};
        VkShaderModule mod;
        if (vk_module(&mci, &mod, fs->key, "flat colour id") == VK_SUCCESS) {
            vks_shader *s = malloc(sizeof *s); *s = *fs; s->mod = mod; s->base = fs; s->nvar = 0; s->words = NULL; s->nwords = 0;
            from[n] = fs; to[n] = s; n++; r = s;
            VKLOG("shader id %u = %s", id, fs->key);
        }
    }
    ReleaseSRWLockExclusive(&g_shl);
    return r ? r : fs;
}
const vks_shader *vksh_fs_variant(const vks_shader *fs, int flags) {
    if (!flags) return fs;
    AcquireSRWLockExclusive(&g_shl); const vks_shader *r = fs_variant_l(fs, flags); ReleaseSRWLockExclusive(&g_shl);
    return r;
}

// ---------- The c1 bank ----------
static int c1_store(vks_shader *s, const uint8_t *data, uint32_t len) {
    VkDeviceSize need = ((VkDeviceSize)len + 255) & ~(VkDeviceSize)255, align = g_vk.props.limits.minUniformBufferOffsetAlignment;
    if (align < 256) align = 256;
    if (!g_c1.buf || g_c1.used + need > g_c1.size) {
        // a new bank; the old one stays (shaders keep their slices for the run)
        VkDeviceSize sz = 4u << 20; if (need > sz) sz = need;
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, sz, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
        VkBuffer b; vk_alloc m;
        if (vkCreateBuffer(g_vk.dev, &bci, NULL, &b) != VK_SUCCESS) return -1;
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g_vk.dev, b, &mr);
        if (vk_alloc_mem(&mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0, &m)) { vkDestroyBuffer(g_vk.dev, b, NULL); return -1; }
        vkBindBufferMemory(g_vk.dev, b, m.mem, m.off);
        memset(m.map, 0, (size_t)sz);
        vk_name((uint64_t)b, VK_OBJECT_TYPE_BUFFER, "c1 bank");
        g_c1.buf = b; g_c1.mem = m; g_c1.used = 0; g_c1.size = sz;
    }
    VkDeviceSize off = (g_c1.used + align - 1) / align * align;
    memcpy((uint8_t *)g_c1.mem.map + off, data, len);
    g_c1.used = off + need;
    s->c1buf = g_c1.buf; s->c1off = off; s->c1_len = len;
    return 0;
}

static int hexval(char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }

// ---------- From a module and its reflection ----------
static int stage_of(const char *s) {
    return !strcmp(s, "vs") ? VKS_VS : !strcmp(s, "ps") ? VKS_FS : !strcmp(s, "gs") ? VKS_GS : !strcmp(s, "cs") ? VKS_CS : !strcmp(s, "hs") || !strcmp(s, "tcs") || !strcmp(s, "ts") ? VKS_TCS : !strcmp(s, "ds") || !strcmp(s, "tes") || !strcmp(s, "ms") ? VKS_TES : -1;
}
static VkDescriptorType desc_type(const char *d) {
    if (!strcmp(d, "uniform_buffer")) return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    if (!strcmp(d, "storage_buffer")) return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    if (!strcmp(d, "combined_image_sampler")) return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    if (!strcmp(d, "sampled_image")) return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    if (!strcmp(d, "sampler")) return VK_DESCRIPTOR_TYPE_SAMPLER;
    if (!strcmp(d, "storage_image")) return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    if (!strcmp(d, "uniform_texel_buffer")) return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    if (!strcmp(d, "storage_texel_buffer")) return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}
static int ct_of(const char *t) { return !t ? VKS_CT_FLOAT : !strcmp(t, "uint") ? VKS_CT_UINT : !strcmp(t, "int") ? VKS_CT_INT : VKS_CT_FLOAT; }

void vksh_free(vks_shader *s) {
    if (!s || s == &g_failed_mark) return;
    if (s->mod) vkDestroyShaderModule(g_vk.dev, s->mod, NULL);
    for (int i = 0; i < s->nvar; i++) if (s->var[i].s != s) vksh_free(s->var[i].s);   // a variant that could not be made is the shader itself
    if (s->dsl && !s->base) vkDestroyDescriptorSetLayout(g_vk.dev, s->dsl, NULL);
    if (!s->base) free(s->b);
    free(s->words); free(s);
}

vks_shader *vksh_make(const char *key, const uint8_t *spv, size_t spv_size, const char *json, size_t json_size, const char **why) {
    *why = NULL;
    vj *j = vj_parse(json, json_size);
    if (!j) { *why = "its reflection is not JSON"; return NULL; }
    vks_shader *s = calloc(1, sizeof *s);
    snprintf(s->key, sizeof s->key, "%s", key);
    s->stage = stage_of(vj_str(vj_get(j, "stage"), ""));
    { const char *st = vj_str(vj_get(j, "stage"), ""); s->mesh = !strcmp(st, "ts") ? 1 : !strcmp(st, "ms") ? 2 : 0; }
    { const vj *fs_ = vj_get(j, "first_store"); for (int i = 0; fs_ && i < fs_->n && s->nfirst < 8; i++) { const vj *e = vj_at(fs_, i);
        s->first[s->nfirst].reg = (int)vj_int(vj_get(e, "reg"), 0); s->first[s->nfirst].word = (uint32_t)vj_int(vj_get(e, "word"), 0); s->first[s->nfirst].value = (uint32_t)vj_int(vj_get(e, "value"), 0); s->nfirst++; } }
    if (s->stage < 0) { *why = "unknown stage"; goto fail; }
    s->set = vksh_set_of(s->stage);
    size_t nw = spv_size / 4;
    uint32_t *w = malloc(spv_size); memcpy(w, spv, spv_size);
    { const char *od = getenv("NXRT_VK_SPV_DIR");   // test switch: <dir>/<key>.spv replaces the translated module (same interface; the edits below still apply)
      if (od && *od) { char p[MAX_PATH]; snprintf(p, sizeof p, "%s\\%s.spv", od, key); FILE *f = fopen(p, "rb");
        if (f) { fseek(f, 0, SEEK_END); long fl = ftell(f); fseek(f, 0, SEEK_SET); uint32_t *ow = fl > 20 && !(fl & 3) ? malloc((size_t)fl) : NULL;
            if (ow && fread(ow, 1, (size_t)fl, f) == (size_t)fl && ow[0] == 0x07230203u) { free(w); w = ow; nw = (size_t)fl / 4; VKLOG("shader %s: SPIR-V replaced from %s (%ld bytes)", key, p, fl); } else free(ow);
            fclose(f); } } }
    if (vkspv_set_descriptor_set(w, nw, (uint32_t)s->set) < 0) { free(w); *why = "malformed SPIR-V"; goto fail; }
    { static int ni = -1; if (ni < 0) { const char *e = getenv("NXRT_VK_NO_INVARIANT"); ni = e && *e == '1'; } if (ni && s->stage == VKS_VS) nw = vkspv_strip_invariant(w, nw); }
    if (s->stage != VKS_FS && s->stage != VKS_CS && g_vk.props.limits.maxCombinedClipAndCullDistances) {
        size_t tn = vkspv_trim_cull(&w, nw, g_vk.props.limits.maxCombinedClipAndCullDistances);
        if (tn) nw = tn; else { static int n; if (n++ < 4) VKRED("shader %s: more clip and cull distances than the device takes, and the cull ones are used", key); }
    }
    static int ballot_fix = -1;   // NXRT_VK_BALLOT_FIX=0 leaves the ballots as translated (to compare)
    if (ballot_fix < 0) { const char *e = getenv("NXRT_VK_BALLOT_FIX"); ballot_fix = !(e && *e == '0'); }
    if (ballot_fix) {
        int nb = 0; size_t bn = vkspv_ballot_per_lane(&w, nw, &nb);
        if (nb) { nw = bn; VKLOG("shader %s: %d subgroup ballot%s for a counter made per lane (RTX 50 hang, #12)", key, nb, nb > 1 ? "s" : ""); }
    }
    static int ubo_clamp = -1;   // NXRT_VK_UBO_CLAMP=1: uniform array indexes computed at run time held to the array (the GPU freeze, step 5b)
    if (ubo_clamp < 0) { const char *e = getenv("NXRT_VK_UBO_CLAMP"); ubo_clamp = e && *e == '1'; }
    if (ubo_clamp) {
        size_t cn = 0; int nc = 0; uint32_t *cw = vkspv_clamp_ubo(w, nw, &cn, &nc);
        if (cw) { free(w); w = cw; nw = cn; static _Atomic uint64_t ns, nidx; uint64_t k2 = atomic_fetch_add(&ns, 1) + 1, i2 = atomic_fetch_add(&nidx, (uint64_t)nc) + (uint64_t)nc;
            if (k2 <= 4 || !(k2 % 500)) VKLOG("uniform indexes held to their arrays: shader %s, %d (%llu shaders, %llu indexes so far)", key, nc, (unsigned long long)k2, (unsigned long long)i2); }
    }
    // bindings
    const vj *bs = vj_get(j, "bindings");
    s->nb = bs ? bs->n : 0; s->b = calloc(s->nb ? s->nb : 1, sizeof *s->b);
    VkDescriptorSetLayoutBinding *lb = calloc(s->nb ? s->nb : 1, sizeof *lb);
    for (int i = 0; i < s->nb; i++) {
        const vj *e = vj_at(bs, i), *nv = vj_get(e, "nvn"); vks_bind *b = &s->b[i];
        b->binding = (uint32_t)vj_int(vj_get(e, "binding"), 0); b->count = (uint32_t)vj_int(vj_get(e, "count"), 1);
        b->type = desc_type(vj_str(vj_get(e, "descriptor"), ""));
        if (vj_int(vj_get(e, "set"), 0) != 0) { *why = "a descriptor outside set 0"; free(w); free(lb); goto fail; }
        if (b->type == VK_DESCRIPTOR_TYPE_MAX_ENUM) { *why = "an unknown descriptor type"; free(w); free(lb); goto fail; }
        const char *k = vj_str(vj_get(nv, "kind"), "?");
        b->reg = (int16_t)vj_int(vj_get(nv, "reg"), 0); b->bank = (uint8_t)vj_int(vj_get(nv, "bank"), 0);
        b->kind = !strcmp(k, "support") ? VKB_SUPPORT : !strcmp(k, "cbuf") ? VKB_CBUF : !strcmp(k, "c1") ? VKB_C1 : !strcmp(k, "drvcb") ? VKB_DRV : !strcmp(k, "sbuf") ? VKB_SBUF :
                  !strcmp(k, "tex") ? VKB_TEX : !strcmp(k, "tbuf") ? VKB_TBUF : !strcmp(k, "img") ? VKB_IMG : !strcmp(k, "ubo_ptr") ? VKB_UBOP : !strcmp(k, "sbuf_ubo") ? VKB_SBUF_UBO : VKB_UNKNOWN;
        if (b->kind == VKB_UNKNOWN) { static int n; if (n++ < 16) VKRED("shader %s binding %u: kind %s is not replayed yet (bound as empty)", key, b->binding, k); }
        b->rw = vj_true(vj_get(nv, "rw")); b->c0_off = (uint16_t)vj_int(vj_get(nv, "c0_offset"), 0);
        if (b->kind == VKB_SBUF_UBO) b->c0_off = (uint16_t)vj_int(vj_get(nv, "off"), 0);   // where in uniform slot reg the buffer's address and size are
        const vj *sm = vj_get(nv, "sampler");
        if (sm) { const char *sk = vj_str(vj_get(sm, "kind"), ""); b->samp_reg = (int16_t)vj_int(vj_get(sm, "reg"), 0);
            b->samp_kind = !strcmp(sk, "samp_tex") ? VKB_SAMP_TEX : !strcmp(sk, "samp") ? VKB_SAMP_SLOT : !strcmp(sk, "samp_id") ? VKB_SAMP_ID : VKB_SAMP_NONE; }
        else if (b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) { b->samp_kind = VKB_SAMP_TEX; b->samp_reg = b->reg; }
        vkspv_img im; b->vtype = VK_IMAGE_VIEW_TYPE_2D;
        if (!vkspv_image_of(w, nw, b->binding, &im) && im.is_image) { b->dim = im.dim; b->vtype = view_type(im.dim, im.arrayed); b->ct = im.ct; b->depth = im.depth; b->ms = im.ms; b->dref = im.dref; b->plain = im.plain; }
        lb[i] = (VkDescriptorSetLayoutBinding){b->binding, b->type, b->count, (VkShaderStageFlags)vksh_stage_bit(s), NULL};
    }
    // interface
    const vj *in = vj_get(j, "inputs");
    for (int i = 0; in && i < in->n && s->stage == VKS_VS && s->nin < 32; i++) {
        const vj *e = vj_at(in, i);
        s->in[s->nin].loc = (uint8_t)vj_int(vj_get(e, "location"), 0); s->in[s->nin].ct = (uint8_t)ct_of(vj_str(vj_get(e, "type"), "float"));
        s->in[s->nin].ncomp = (uint8_t)vj_int(vj_get(e, "components"), 4); s->nin++;
    }
    const vj *out = vj_get(j, "outputs");
    for (int i = 0; out && i < out->n && s->stage == VKS_FS; i++) {
        const vj *e = vj_at(out, i); int l = vj_int(vj_get(e, "location"), -1);
        if (l >= 0 && l < 8) { s->out_mask |= 1u << l; s->out_ct[l] = (uint8_t)ct_of(vj_str(vj_get(e, "type"), "float")); }
    }
    const vj *ls = vj_get(j, "local_size");
    for (int i = 0; i < 3; i++) s->local[i] = (uint32_t)vj_int(vj_at(ls, i), 1);
    const vj *dr = vj_get(j, "drv_reads");
    for (int i = 0; dr && i < dr->n; i++) { int bank = atoi(dr->keys[i]); if (bank >= 0 && bank < 3) s->drv_read[bank] = 1; }
    const vj *s2 = vj_get(j, "s2_compute_layout");
    if (s2) { s->s2cs = 1; s->s2_ssbo_base = (uint32_t)vj_int(vj_get(s2, "ssbo_base"), 0); s->s2_tex_base = (uint32_t)vj_int(vj_get(s2, "tex_base"), 0); s->s2_ntex = (uint32_t)vj_int(vj_get(s2, "n_tex"), 0); }
    // c1: the bytes nxshader took from the shader's record (its translation job), published beside the reflection
    int has_c1 = 0; for (int i = 0; i < s->nb; i++) has_c1 |= s->b[i].kind == VKB_C1;
    if (has_c1) {
        const char *hex = vj_str(vj_get(j, "c1"), NULL);
        if (!hex) { s->c1_missing = 1; static int n; if (n++ < 8) VKRED("shader %s reads its c1 segment but the reflection has no \"c1\" (an older nxshader publish): it reads zeros", key); }
        else {
            size_t hl = strlen(hex) / 2; uint8_t *c1 = malloc(hl ? hl : 1);
            for (size_t k = 0; k < hl; k++) { int a = hexval(hex[2 * k]), b = hexval(hex[2 * k + 1]); c1[k] = (uint8_t)(a < 0 || b < 0 ? 0 : a * 16 + b); }
            if (hl && c1_store(s, c1, (uint32_t)hl)) VKRED("shader %s: no memory for its c1 segment", key);
            free(c1);
        }
    }
    // NXRT_VK_HILITE=<file of keys, one per line>: those pixel shaders draw their line number as a flat colour, read back with NXRT_VK_TD_PASSES: which material is which pixel
    if (s->stage == VKS_FS && getenv("NXRT_VK_HILITE")) {
        static char *keys; static long nk;
        if (!keys) { FILE *h = fopen(getenv("NXRT_VK_HILITE"), "rb"); if (h) { fseek(h, 0, SEEK_END); nk = ftell(h); fseek(h, 0, SEEK_SET); keys = calloc(1, nk + 1); fread(keys, 1, nk, h); fclose(h); } }
        const char *hit = keys ? strstr(keys, key) : NULL;
        if (hit) {
            uint32_t id = (uint32_t)((hit - keys) / 17), bits[3];
            for (int i = 0; i < 3; i++) { float v = 1.0f + (float)((id >> (5 * i)) & 31) / 32.0f; memcpy(&bits[i], &v, 4); }
            static const uint32_t head[] = {0x07230203, 0x00010300, 0, 14, 0, 0x00020011, 1, 0x0003000E, 0, 1, 0x0006000F, 4, 1, 0x6E69616D, 0, 2, 0x00030010, 1, 7, 0x00040047, 2, 30, 0,
                                            0x00020013, 3, 0x00030021, 4, 3, 0x00030016, 5, 32, 0x00040017, 6, 5, 4, 0x00040020, 7, 3, 6, 0x0004003B, 7, 2, 3};
            enum { HN = sizeof head / 4 };
            uint32_t *m = malloc((HN + 36) * 4); memcpy(m, head, sizeof head); uint32_t n = HN;
            for (int i = 0; i < 3; i++) { m[n++] = 0x0004002B; m[n++] = 5; m[n++] = 8 + i; m[n++] = bits[i]; }
            m[n++] = 0x0004002B; m[n++] = 5; m[n++] = 11; m[n++] = 0x3F800000;
            m[n++] = 0x0007002C; m[n++] = 6; m[n++] = 12; m[n++] = 8; m[n++] = 9; m[n++] = 10; m[n++] = 11;
            m[n++] = 0x00050036; m[n++] = 3; m[n++] = 1; m[n++] = 0; m[n++] = 4; m[n++] = 0x000200F8; m[n++] = 13; m[n++] = 0x0003003E; m[n++] = 2; m[n++] = 12; m[n++] = 0x000100FD; m[n++] = 0x00010038;
            free(w); w = m; nw = n;
        }
    }
    // Vulkan objects
    VkShaderModuleCreateInfo mci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, nw * 4, w};
    VkResult r = vk_module(&mci, &s->mod, s->key, "base");
    if (s->stage == VKS_VS || s->stage == VKS_FS) { s->words = w; s->nwords = nw; } else free(w);   // the sources of their variants
    if (r != VK_SUCCESS) { free(lb); *why = "vkCreateShaderModule failed"; goto fail; }
    VkDescriptorSetLayoutCreateInfo dci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, (uint32_t)s->nb, lb};
    r = vkCreateDescriptorSetLayout(g_vk.dev, &dci, NULL, &s->dsl);
    free(lb);
    if (r != VK_SUCCESS) { *why = "vkCreateDescriptorSetLayout failed"; goto fail; }
    vj_free(j);
    return s;
fail:
    vj_free(j); vksh_free(s);
    return NULL;
}

static vks_shader *get_l(const nxs_shader *n) {   // g_shl held
    vks_shader *s = vk_hm_get(&g_sh, n->key);
    if (s == &g_failed_mark) return NULL;
    if (!s) {
        const char *why = NULL;
        s = vksh_make(n->key, n->spv, n->spv_size, n->json, n->json_size, &why);
        if (!s) { g_failed++; VKRED("shader %s cannot be used: %s; its draws are skipped", n->key, why ? why : "?"); vk_hm_put(&g_sh, n->key, &g_failed_mark); return NULL; }
        g_loaded++;
        vk_hm_put(&g_sh, n->key, s);
        { char k[128]; int o = 0; int cnt[VKB_UNKNOWN + 1] = {0}; for (int i = 0; i < s->nb; i++) cnt[s->b[i].kind]++;
          static const char *nm[] = {"", "support", "cbuf", "c1", "drv", "sbuf", "tex", "tbuf", "img", "ubop", "sbuf_ubo", "?"};
          for (int i = 1; i <= VKB_UNKNOWN; i++) if (cnt[i]) o += snprintf(k + o, sizeof k - o, " %s %d", nm[i], cnt[i]);
          VKLOG("shader %s (%s):%s%s%s", n->key, s->stage == VKS_VS ? "vertex" : s->stage == VKS_FS ? "fragment" : s->stage == VKS_CS ? "compute" : s->stage == VKS_GS ? "geometry" : "tessellation", k,
                s->drv_read[0] || s->drv_read[2] ? " (reads driver segments)" : "", s->s2cs ? " (Switch 2 compute bank)" : ""); }
    }
    s->uses++;
    return s;
}
const vks_shader *vksh_get(const prog_t *p, unsigned i) {
    const nxs_shader *n = NULL;
    if (!p || nxs_resolve(p, i, &n) != 1 || !n) return NULL;
    AcquireSRWLockExclusive(&g_shl); vks_shader *s = get_l(n); ReleaseSRWLockExclusive(&g_shl);
    return s;
}
const vks_shader *vksh_by_key(const char *key) {
    const nxs_shader *n = nxs_peek(key);
    if (!n) return NULL;
    AcquireSRWLockExclusive(&g_shl); vks_shader *s = get_l(n); ReleaseSRWLockExclusive(&g_shl);
    return s;
}

int vksh_init(void) {
    VkDescriptorSetLayoutCreateInfo dci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 0, NULL};
    return vkCreateDescriptorSetLayout(g_vk.dev, &dci, NULL, &g_empty) == VK_SUCCESS ? 0 : -1;
}
void vksh_shutdown(void) {
    for (uint32_t i = 0; i < g_sh.cap; i++) if (g_sh.vals[i]) vksh_free(g_sh.vals[i]);
    vk_hm_free(&g_sh);
    if (g_empty) vkDestroyDescriptorSetLayout(g_vk.dev, g_empty, NULL);
    g_empty = VK_NULL_HANDLE;
}
