// The generated registry of host replacements (gen/out/hle_registry.gen.c, by gen/gen_hle.py)
#pragma once
#include "nxw/guest_cpu.h"

enum { NXRT_HLE_FORCED = 1, NXRT_HLE_SWHLE = 2, NXRT_HLE_NVN = 3 };
typedef struct nxrt_hle_entry {
    const char *name;      // the symbol (mangled) or NVN procedure name
    uint8_t kind;          // NXRT_HLE_*
    uint8_t ported;        // 1: the adapter calls nxnative's ported host function; 2: nxnative's constant result; 0: a stub (logs once, neutral result)
    const char *file;      // nxnative's source file
    const char *why;       // nxnative's stated reason for the takeover
    NxwGuestFn adapter;    // adapter_<name>
} nxrt_hle_entry;
extern const nxrt_hle_entry nxrt_hle_table[];
extern const unsigned nxrt_hle_count;

const nxrt_hle_entry *nxrt_hle_find(const char *name, int kind);   // kind 0 = any
uint64_t nxrt_token_addr(unsigned k);
uint64_t resolve_import(const char *name);
void nxrt_tokens_init(void);
void nxrt_hle_note_bound(int n);
uint32_t nxrt_hle_calls(const char *name);
// The once-per-name log of a generated stub (gen/out/hle_stubs.gen.c)
void nxrt_stub_log(const char *name, const char *file, uint8_t *seen, const NxwGuestCpu *cpu);
uint64_t nxrt_token_of(const char *name);
void nxrt_symbolize(uint64_t guest_va, char *out, size_t n);
void nxrt_log_summary(void);
