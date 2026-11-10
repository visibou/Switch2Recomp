/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* The boundary between the game and NVN (docs/PLAN_NATIVE_GPU.md, D2 and D3). main imports nvnBootstrapLoader and finds every
   other entry by name through it or nvnDeviceGetProcAddress, so those two functions are the one place to stand. What they hand
   back are synthetic addresses in a range added after the code (build_dispatch), each slot one name, all of them nvn_thunk.
   SWR_NVN_CENSUS=1: pass-through. Every slot counts its calls by thread and calls the game's own driver in sdk; nothing else
   changes. nvn_census.tsv is rewritten every 10 s (tools/nvn_census_report.py joins it with what the NVN layer implements). */
#include "rt.h"

#define NVN_SLOTS 2048
#define NVN_THREADS 96
u64 g_nvn_lo, g_nvn_hi;                          /* the synthetic range; set by nvn_hook_reserve */
typedef struct { char name[72]; u64 real; } Slot;
static Slot g_slot[NVN_SLOTS]; static volatile int g_nslot;
static PMutex g_nvn_lock;
static u64 (*g_cnt)[NVN_THREADS];                /* [slot][thread], each thread writes its own column */
static char g_tname[NVN_THREADS][32]; static volatile int g_nthr;
static GuestFn g_real_boot, g_real_gpa;

u64 nvn_hook_reserve(u64 code_hi){ g_nvn_lo=(code_hi+0xFFFull)&~0xFFFull; g_nvn_hi=g_nvn_lo+NVN_SLOTS*4; return g_nvn_hi; }

static int thread_index(CPU* c){
  static __thread int ti=-1; if(ti>=0) return ti;
  pm_lock(&g_nvn_lock);
  int i=g_nthr; if(i>=NVN_THREADS-1) i=NVN_THREADS-1; else g_nthr=i+1;
  KThread* t=(KThread*)c->thread; snprintf(g_tname[i],sizeof g_tname[i],"%s",t && t->name[0]? t->name : "?");
  pm_unlock(&g_nvn_lock);
  return ti=i; }

static void nvn_thunk(CPU* RESTRICT c, u64 e){
  u32 i=(u32)((e-g_nvn_lo)>>2);
  g_cnt[i][thread_index(c)]++;
  u64 t=g_slot[i].real; GuestFn f=swr_lookup(c,t); f(c,t); }

/* the name's slot, made on first sight. real==0 (the driver does not know the name) stays 0: the game sees what it would see */
static u64 slot_of(const char* name, u64 real){
  if(!real) return 0;
  pm_lock(&g_nvn_lock);
  int n=g_nslot, i=0; for(;i<n;i++) if(!strcmp(g_slot[i].name,name)) break;
  if(i==n){ if(n==NVN_SLOTS){ pm_unlock(&g_nvn_lock); WARN("nvn: more than %d entry points, %s goes straight to the driver\n",NVN_SLOTS,name); return real; }
    snprintf(g_slot[n].name,sizeof g_slot[n].name,"%s",name); g_slot[n].real=real; g_nslot=n+1; }
  pm_unlock(&g_nvn_lock);
  return g_nvn_lo+(u64)i*4; }

static void nvn_gpa(CPU* RESTRICT c, u64 e){
  const char* name= c->x[1]? (const char*)GP(c->x[1]) : NULL;
  g_real_gpa(c,e); if(name) c->x[0]=slot_of(name,c->x[0]); }

/* nvnDeviceGetProcAddress is not exported: the loader is the only way to it, so that is where it gets its wrapper */
static void nvn_boot(CPU* RESTRICT c, u64 e){
  extern GuestFn* g_disp; extern u64 g_code_lo;
  const char* name= c->x[0]? (const char*)GP(c->x[0]) : NULL;
  g_real_boot(c,e); if(!name) return;
  if(c->x[0] && !g_real_gpa && !strcmp(name,"nvnDeviceGetProcAddress")){ u64 i=(c->x[0]-g_code_lo)>>2; g_real_gpa=g_disp[i]; g_disp[i]=nvn_gpa; }
  c->x[0]=slot_of(name,c->x[0]); }

static void nvn_hle_install(u64 (*find)(const char*));
void nvn_hook_install(u64 (*find)(const char*)){
#ifdef SWR_HAVE_NVN
  { extern int gpu_nvn_wanted(void); if(gpu_nvn_wanted()){ nvn_hle_install(find); return; } }
#endif
  if(!getenv("SWR_NVN_CENSUS")) return;
  extern GuestFn* g_disp; extern u64 g_code_lo, g_code_hi;
  u64 a=find("nvnBootstrapLoader");
  if(!a){ WARN("nvn: no nvnBootstrapLoader: this game has no NVN boundary to take\n"); return; }
  pm_init(&g_nvn_lock); g_cnt=calloc(NVN_SLOTS,sizeof *g_cnt);
  for(u64 p=g_nvn_lo;p<g_nvn_hi;p+=4) g_disp[(p-g_code_lo)>>2]=nvn_thunk;
  g_real_boot=g_disp[(a-g_code_lo)>>2]; g_disp[(a-g_code_lo)>>2]=nvn_boot;
  LOG("nvn: census on, nvnBootstrapLoader %llx, slots at %llx\n",(unsigned long long)a,(unsigned long long)g_nvn_lo); }

/* main loop, once a second */
void nvn_census_tick(void){
#ifdef SWR_HAVE_NVN
  { extern int g_gpu_nvn; extern void vkt_log_stats(void); static int s; if(g_gpu_nvn && getenv("NXRT_GFX") && !strcmp(getenv("NXRT_GFX"),"vulkan") && !(++s%5)){ vkt_log_stats(); if(getenv("SWR_NVN_STACKS")){ extern void plat_dump_native_stacks(void); plat_dump_native_stacks(); } } }
#endif
  if(!g_cnt) return;
  static int sec; static u64 last[NVN_SLOTS]; static int lastf;
  if(++sec%10) return;
  char p[1200]; snprintf(p,sizeof p,"%snvn_census.tsv",g_exedir); FILE* f=fopen(p,"w"); if(!f) return;
  int nt=g_nthr, df=g_frames-lastf; lastf=g_frames;
  fprintf(f,"# seconds %d, frames %d, frames in the last 10 s %d\nname\treal\tcalls\tper_frame_last_10s\tthreads\n",sec,g_frames,df);
  for(int i=0;i<g_nslot;i++){ u64 tot=0; for(int t=0;t<nt;t++) tot+=g_cnt[i][t];
    fprintf(f,"%s\t%llx\t%llu\t%.2f\t",g_slot[i].name,(unsigned long long)g_slot[i].real,(unsigned long long)tot,df>0? (double)(tot-last[i])/df : 0.0);
    last[i]=tot; int any=0;
    for(int t=0;t<nt;t++) if(g_cnt[i][t]){ fprintf(f,"%s%s=%llu",any? "," : "",g_tname[t],(unsigned long long)g_cnt[i][t]); any=1; }
    fputc('\n',f); }
  fclose(f); }

/* ---------------------------------------------------------------- SWR_GPU=nvn: the game's NVN calls land in runtime/nvn, nothing of the driver in sdk runs */
#ifdef SWR_HAVE_NVN
#include "nvn/host/nvn_rt_api.h"
int g_gpu_nvn;
static GuestFn g_hle_boot;
static void nvn_hle_thunk(CPU* RESTRICT c, u64 e){
  nxrt_hle_table[(e-g_nvn_lo)>>2].adapter((void*)c); c->pc=c->x[30]; }

int gpu_nvn_wanted(void){ const char* g=getenv("SWR_GPU"); return g && !strcmp(g,"nvn"); }

static void nvn_hle_boot(CPU* RESTRICT c, u64 e){ (void)e; adapter_nvnBootstrapLoader((void*)c); c->pc=c->x[30]; }

static void nvn_hle_install(u64 (*find)(const char*)){
  extern GuestFn* g_disp; extern u64 g_code_lo;
  u64 a=find("nvnBootstrapLoader");
  if(!a) fatal("SWR_GPU=nvn: this game has no nvnBootstrapLoader");
  if(nxrt_hle_count>NVN_SLOTS) fatal("too many NVN entries");
  _Static_assert(offsetof(CPU,v)==280,"the layout nvn/host/nxw/guest_cpu.h copies");
  for(u32 i=0;i<nxrt_hle_count;i++) g_disp[((g_nvn_lo-g_code_lo)>>2)+i]=nvn_hle_thunk;
  g_hle_boot=g_disp[(a-g_code_lo)>>2]; g_disp[(a-g_code_lo)>>2]=nvn_hle_boot;
  g_gpu_nvn=1; nvn_host_init();
  { char e[1400]; /* the shader cache from tools/nxshader and the driver's pipeline cache: in the game's folder, C: is not for this */
    if(!getenv("NXRT_SHADER_CACHE")){ snprintf(e,sizeof e,"NXRT_SHADER_CACHE=%s%s_nvn\\cache",g_exedir,g_exename); _putenv(e); }
    if(!getenv("NXRT_SHADER_TOOL")) _putenv("NXRT_SHADER_TOOL=off");
    if(!getenv("NXRT_VK_STAGING_MB")) _putenv("NXRT_VK_STAGING_MB=256");
    if(!getenv("NXRT_VK_CACHE_DIR")){ snprintf(e,sizeof e,"NXRT_VK_CACHE_DIR=%s%s_nvn\\vk",g_exedir,g_exename); _putenv(e); } }
  { extern void nvn_host_sdk(u64,u64); for(u32 m=0;m<g_mod->nmods;m++) if(!strcmp(g_mod->mods[m].name,"sdk")) nvn_host_sdk(g_mod->mods[m].base,g_mod->mods[m].size); }
  LOG("nvn: NVN taken over at %llx, %u entries at %llx, graphics backend %s\n",(unsigned long long)a,nxrt_hle_count,(unsigned long long)g_nvn_lo,getenv("NXRT_GFX")? getenv("NXRT_GFX") : "null"); }

void swr_nvn_log(const char* s){ logf_(1,"%s",s); }
void swr_nvn_fatal(const char* s){ fatal("%s",s); }
u64 swr_nvn_slot_addr(unsigned i){ return g_nvn_lo+(u64)i*4; }
const char* swr_nvn_log_dir(void){ static char p[1200]; if(!p[0]) snprintf(p,sizeof p,"%s%s.log",g_exedir,g_exename); return p; }
void swr_nvn_vi_crop(int c[4]){ extern volatile i32 g_vi_crop[4]; for(int i=0;i<4;i++) c[i]=g_vi_crop[i]; }
void* swr_nvn_cur_cpu(void){ KThread* t=kthread_cur(); return t? (void*)&t->cpu : NULL; }
u64 swr_nvn_guest_ns(void){ extern u64 guest_ns(void); return guest_ns(); }
void swr_nvn_symbol(u64 a, char* out, size_t n){ snprintf(out,n,"%llx",(unsigned long long)a); }
#define RETURN_TO_HOST 0xFFFFFFFFFFFF0000ull
/* a guest function on this thread's CPU. Its ret leaves pc at the sentinel; anything else that comes back (a longjmp, a tail through the trampoline) goes round the loop as in thread_body */
u64 swr_nvn_guest_call(u64 va, int n, const u64* args){
  KThread* t=kthread_cur(); if(!t) fatal("a guest call from a host thread (%llx)",(unsigned long long)va);
  CPU* c=&t->cpu; u64 lr=c->x[30], pc=c->pc;
  for(int i=0;i<n && i<8;i++) c->x[i]=args[i];
  c->x[30]=RETURN_TO_HOST; c->pc=va;
  while(c->pc!=RETURN_TO_HOST){ GuestFn f=swr_lookup(c,c->pc); f(c,c->pc); }
  u64 r=c->x[0]; c->x[30]=lr; c->pc=pc; return r; }

/* a present: what vi_show_addr and vi2_present do for the frame count and the tools (frames.csv) */
void swr_nvn_frame(void){
  extern void swr_vt_frame(void); extern u64 swr_run_ns(void); extern volatile u32 g_swr_present_seq; extern void input_rec_present(u32);
  extern volatile int g_nvn_present_interval; extern void swr_present_pace(u32 iv); extern volatile u64 g_frame_mode;
  /* the frame-rate mode and vblank pacing, as vi2_present does on the Eden path: without them the game kept its 60 FPS mode at about 30 FPS (half speed).
     In the 30 FPS mode the game's clock moves two vblanks a frame whatever interval it last set */
  u32 iv=(u32)(g_frame_mode==3 ? 2 : g_frame_mode==2 ? 1 : g_nvn_present_interval);
  u64 t0=swr_run_ns(); swr_present_pace(iv); u64 waited=swr_run_ns()-t0;
  g_frames++; swr_vt_frame(); g_swr_present_seq++; input_rec_present(iv);
  static FILE* ff; static u32 nf;
  if(!ff){ char p[1400]; snprintf(p,sizeof p,"%sframes.csv",g_exedir); ff=fopen(p,"w"); if(ff) fprintf(ff,"ms,interval,wait_us,n,front_wait_ms,back_wait_ms\n"); }
  if(ff){ fprintf(ff,"%.3f,%u,%llu,%u,0,0\n",swr_run_ns()/1e6,iv,(unsigned long long)(waited/1000),g_swr_present_seq); if(++nf%60==0) fflush(ff); } }
/* the next presented frame as a PNG; waits a moment for it, the final screenshot is followed by ExitProcess */
void nvn_shot(const char* path){ char p[1200]; snprintf(p,sizeof p,"%s.png",path); remove(p); { extern void plat_mkdir_for(const char*); plat_mkdir_for(p); } dm_gfx_shot_at(p);
  for(int i=0;i<600;i++){ FILE* f=fopen(p,"rb"); if(f){ fclose(f); plat_sleep_ns(100000000); return; } plat_sleep_ns(10000000); } }
#else
int g_gpu_nvn; int gpu_nvn_wanted(void){ return 0; }
void nvn_window_set(void* h){ (void)h; }
void nvn_window_resized(int w, int h){ (void)w; (void)h; }
void nvn_shot(const char* path){ (void)path; }
#endif
