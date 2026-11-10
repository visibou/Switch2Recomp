/* SWR_GPU=eden: Eden's GPU side (runtime/eden, Vulkan) instead of the GL renderer, when the runtime was built with
   it (build/eden from tools/build_eden.sh; switchrecomp.py then defines SWR_HAVE_EDEN). Always compiled: without
   Eden these say "not available" and the GL renderer runs. With Eden, GPU maps (nvdrv.c), submissions (gpu.c) and
   presents (main.c) go to it instead of the runtime's own GPU code. */
#include "rt.h"
#ifdef SWR_HAVE_EDEN
#include "eden/eden_gpu.h"
#endif

int g_gpu_eden=0;   /* 1 while Eden draws */

int gpu_eden_wanted(void){
  const char* g=getenv("SWR_GPU"); if(!g || strcmp(g,"eden")) return 0;
#ifdef SWR_HAVE_EDEN
  return 1;
#else
  WARN("SWR_GPU=eden: this runtime was built without the Eden GPU backend (tools/build_eden.sh), using OpenGL\n"); return 0;
#endif
}

#ifdef SWR_HAVE_EDEN
static void replay_map(u64 gva, u64 cpu, u64 size){ eden_gpu_map(gva,cpu,size); }
#endif

/* on the game window; Eden's log and caches go to <game>_eden/ next to the exe */
int gpu_eden_start(void* hwnd, int w, int h){
#ifdef SWR_HAVE_EDEN
  extern u64 g_title_id; extern void gmmu_foreach(void (*cb)(u64,u64,u64));
  char dir[1400]; snprintf(dir,sizeof dir,"%s%s_eden",g_exedir,g_exename);
  { extern float g_opt_resf; eden_gpu_set_resolution(g_opt_resf>0? g_opt_resf : 1.0f); }   /* ini resolution= */
  if(!eden_gpu_start(hwnd,w,h,g_mem,AS_SIZE,dir,g_title_id)){ WARN("gpu: the Eden GPU backend did not start (see %s/log), using OpenGL\n",dir); return 0; }
  gmmu_foreach(replay_map);   /* GPU mappings made before (none expected: the game has not run yet) */
  g_gpu_eden=1;
  LOG("gpu: Eden GPU backend (Vulkan) started, log in %s/log\n",dir);
  return 1;
#else
  (void)hwnd; (void)w; (void)h; return 0;
#endif
}

#ifdef SWR_HAVE_EDEN
void gpu_eden_map(u64 gva, u64 cpu, u64 size){ if(g_gpu_eden) eden_gpu_map(gva,cpu,size); }
void gpu_eden_unmap(u64 gva, u64 size){ if(g_gpu_eden) eden_gpu_unmap(gva,size); }
void gpu_eden_memory(u64 start, u64 size, int created){ if(g_gpu_eden) eden_gpu_memory(start,size,created); }
void gpu_eden_submit(u64* e, u32 n, u32 syncpt, u32 incs, u32 wait_id, u32 wait_val){ eden_gpu_submit((const unsigned long long*)e,n,syncpt,incs,wait_id,wait_val); }
void gpu_eden_present_image(u64 a, u32 w, u32 h, u32 pitch, u32 layout, u32 bhl, u32 fmt){ extern volatile i32 g_vi_crop[4]; eden_gpu_present_image(a,w,h,pitch,layout,bhl,fmt,g_vi_crop[2],g_vi_crop[3]); }   /* crop: svc_ns2.c vi2_present */
void gpu_eden_resize(int w, int h){ if(g_gpu_eden) eden_gpu_resize(w,h); }
u64 gpu_eden_screenshot(const char* path){ return g_gpu_eden? eden_gpu_screenshot(path) : 0; }
void gpu_eden_syncpt_cpu_incr(u32 id){ eden_gpu_syncpt_cpu_incr(id); }
void gpu_eden_gpu_only(u64 cpu, u64 size, int on){ if(g_gpu_eden) eden_gpu_gpu_only(cpu,size,on); }
void gpu_eden_present_wait(u32 id, u32 value){ eden_gpu_present_wait(id,value); }
void gpu_eden_wait_ns(u64* front, u64* back){ if(g_gpu_eden) eden_gpu_wait_ns((unsigned long long*)front,(unsigned long long*)back); else *front=*back=0; }
#else
void gpu_eden_wait_ns(u64* front, u64* back){ *front=*back=0; }
void gpu_eden_map(u64 gva, u64 cpu, u64 size){ (void)gva; (void)cpu; (void)size; }
void gpu_eden_unmap(u64 gva, u64 size){ (void)gva; (void)size; }
void gpu_eden_memory(u64 start, u64 size, int created){ (void)start; (void)size; (void)created; }
void gpu_eden_submit(u64* e, u32 n, u32 syncpt, u32 incs, u32 wait_id, u32 wait_val){ (void)e; (void)n; (void)syncpt; (void)incs; (void)wait_id; (void)wait_val; }
void gpu_eden_present_image(u64 a, u32 w, u32 h, u32 pitch, u32 layout, u32 bhl, u32 fmt){ (void)a; (void)w; (void)h; (void)pitch; (void)layout; (void)bhl; (void)fmt; }
void gpu_eden_resize(int w, int h){ (void)w; (void)h; }
u64 gpu_eden_screenshot(const char* path){ (void)path; return 0; }
void gpu_eden_syncpt_cpu_incr(u32 id){ (void)id; }
void gpu_eden_gpu_only(u64 cpu, u64 size, int on){ (void)cpu; (void)size; (void)on; }
void gpu_eden_present_wait(u32 id, u32 value){ (void)id; (void)value; }
#endif
