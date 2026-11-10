/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* GL call queue: the GPU thread records calls, a dedicated thread replays them */
#ifndef SWR_GLQ_H
#define SWR_GLQ_H
int glq_start(void);                         /* starts the GL thread and makes its context, 0 if that failed */
int glq_active(void);
void glq_excl(void); int glq_drv_lock(void); void glq_drv_unlock(void);
void* glq_proc(const char* name);           /* like backend_gl_proc but hands out the queued wrappers */
void glq_swap(void);                        /* queued swap, never more than one frame in flight */
int glq_worker_current(void);               /* worker context current + bypass the queue on this thread */
u64  glq_thread_cpu_ns(void);   /* CPU time used by the GL thread so far */
void glq_sync_report(char* out, size_t cap, u32 frames);   /* worst blocking GL calls since the last report */
void glq_stall_report(char* out, size_t cap);   /* GL thread state for the stall dump */
void glq_wd_mark(u32 id); int glq_wd_probe(void); int glq_wd_done(u32 id);   /* GPU watchdog fences */
#endif
