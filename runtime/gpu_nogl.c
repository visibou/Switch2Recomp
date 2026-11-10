/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* no-GL build used by the headless tests */
#include "gpu.h"
int g_gl=0; u64 g_draws;

int gpu_async_submit(u64* e, u32 n, u32 syncpt, u32 incs, u32 wid, u32 wval){ (void)e;(void)n;(void)syncpt;(void)incs;(void)wid;(void)wval; return 0; }

void gpu_gl_present_buffer(u64 a, u32 w, u32 h, u32 p, u32 l, u32 b, u32 f){ (void)a;(void)w;(void)h;(void)p;(void)l;(void)b;(void)f; }

void r3d_clear_gl(GpuChannel* ch, u32 v){ (void)ch;(void)v; }

void r3d_draw_gl(GpuChannel* ch,int i,u32 t,u32 f,u32 c,u32 bv,u32 bi,u32 in){ (void)ch;(void)i;(void)t;(void)f;(void)c;(void)bv;(void)bi;(void)in; }

void r3d_cb_bind_gl(GpuChannel* ch,int s,u32 v){ (void)ch;(void)s;(void)v; }

void gpu_gl_dump_rts(void){}

void gpu_gl_rec_finish(void){}

void gpu_gl_rt_flush(u64 a){ (void)a; }

void gpu_gl_rt_reload(u64 a){ (void)a; }

void gpu_gl_trace(const char* s){ (void)s; }

int r3d_blit_gl(void* s, void* d, int dx, int dy, int dw, int dh, double sx, double sy, double sdx, double sdy){ (void)s;(void)d;(void)dx;(void)dy;(void)dw;(void)dh;(void)sx;(void)sy;(void)sdx;(void)sdy; return 0; }

void gpu_gl_note_compute(void){}

void lsfg_ini(const char* k, const char* v){ (void)k; (void)v; }

u64 gpu_busy_ns(void){ return 0; }

void gpu_gl_dispatch(void* ch, const void* q, unsigned long long code){ (void)ch;(void)q;(void)code; }

volatile int g_cs_any; void cs_mem_need(u64 a, u64 n){ (void)a; (void)n; }

void gpu_gl_before_syncwait(void){}
void gpu_gl_before_wait(int kind, unsigned long long addr){ (void)kind; (void)addr; }

void gpu_gl_stall_report(void){}
int gpu_gl_hold_write(u64 addr, const void* data, u32 len){ (void)addr; (void)data; (void)len; return 0; }
int gpu_gl_hold_reduce(u64 addr, u32 op, u32 payload, int is_signed){ (void)addr; (void)op; (void)payload; (void)is_signed; return 0; }

void gpu_gl_stream_incr(u32 id){ extern void syncpt_incr(u32); syncpt_incr(id); }

/* set by vi QueueBuffer, only the GL presenter reads them */
volatile i32 g_vi_crop[4];
volatile u32 g_vi_transform;
int g_texoff_emul;
int g_opt_frame_skip;   /* fps_target, only used by the GL renderer */

void gpu_crash_note(void){}

unsigned gpu_gl_held_view(int kind, unsigned long long addr, unsigned cur){ (void)kind; (void)addr; return cur; }
int gpu_gl_cs_pending(unsigned long long a, unsigned long long n){ (void)a; (void)n; return 0; }
void gpu_gl_cs_poll(void){}
