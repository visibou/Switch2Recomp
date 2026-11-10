/* What runtime/nvn_hook.c (which sees rt.h) needs from the NVN layer (whose headers don't mix with rt.h) */
#pragma once
typedef struct { const char* name; unsigned char kind, ported; const char* file; const char* why; void (*adapter)(void*); } SwrHleEntry;
#define nxrt_hle_table nxrt_hle_table_rt
extern const SwrHleEntry nxrt_hle_table_rt[] __asm__("nxrt_hle_table");
extern const unsigned nxrt_hle_count;
void adapter_nvnBootstrapLoader(void* cpu);
void nvn_host_init(void);
void nvn_window_set(void* hwnd);
void nvn_window_resized(int w, int h);
void dm_gfx_shot_at(const char* path);
