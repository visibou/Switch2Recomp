// A guest module at run time (host/loader.cpp maps it; nxload links it). Plain C, shared with the C++ loader.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[16];
    uint8_t *base;             // start of the image = its guest address
    size_t size;
    uint64_t text_end, ro_off, data_off;   // segment boundaries (module-relative)
    char build_id[41];         // the NSO build ID (hex, 20 bytes)
    uint64_t init, init_array, init_arraysz;   // DT_INIT (module-relative, as nxnative keeps it), the init array
    uint64_t fini;
    int nro;                   // NRO modules are not supported in v0 (Pokopia lists none)
    int loaded;
} module_t;

#define g_mod nvn_g_mod
extern module_t *g_mod;
extern int g_nmod;
void run_module_inits(void);
void *module_sym(module_t *m, const char *name);   // a guest address (call it with GUEST_CALL), NULL if the module does not export it
void *guest_sym(const char *name);
module_t *module_of(const void *p);
int dm_sdk_major(void);
int dm_main_imports(const char *prefix);

#ifdef __cplusplus
}
#endif
