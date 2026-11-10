/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
#define _GNU_SOURCE

/* platform glue: threads, timers, VM, files. win32 + posix */
#include "rt.h"
#include <time.h>
#include <math.h>      /* sinf */

__thread void* tls_thread;
int g_verbose = 1;
static FILE* g_logf;
static PMutex g_logm; static int g_logm_ok;

__thread FILE* g_log_redirect;   /* redirect for the threads.txt dumps so they don't end up in the log, it's per
                                    thread so the other threads keep logging like normal */
void logf_(int lvl, const char* fmt, ...){
  if(!g_logm_ok){ pm_init(&g_logm); g_logm_ok=1; }
  char buf[2048]; va_list ap; va_start(ap,fmt); vsnprintf(buf,sizeof buf,fmt,ap); va_end(ap);
  { FILE* rf=g_log_redirect; if(rf){ fputs(buf,rf); return; } }
  static u64 t0; u64 now=plat_ns(); if(!t0) t0=now;
  char tb[24]; snprintf(tb,sizeof tb,"[%7.2f] ",(double)(now-t0)/1e9);
  pm_lock(&g_logm);
  if(!g_logf){ char p[1200]; snprintf(p,sizeof p,"%s%s.log",g_exedir,g_exename[0]?g_exename:"switchrecomp");
    { char o[1220]; snprintf(o,sizeof o,"%s%s.prev.log",g_exedir,g_exename[0]?g_exename:"switchrecomp"); remove(o); rename(p,o); }   /* the previous run's log survives one launch (a crash, then a restart) */
    g_logf=fopen(p,"w"); }
  if(g_logf){ static int bol=1; if(bol) fputs(tb,g_logf); fputs(buf,g_logf); size_t L=strlen(buf); bol= L && buf[L-1]=='\n'; fflush(g_logf); }
  if(lvl<=1 || g_verbose>=2) fputs(buf,stderr);
  pm_unlock(&g_logm);
}

void fatal(const char* fmt, ...){
  char buf[4096]; va_list ap; va_start(ap,fmt); vsnprintf(buf,sizeof buf,fmt,ap); va_end(ap);
  logf_(0,"FATAL: %s\n",buf);
#if defined(_WIN32)
  { extern void win_fatal_box(const char*); win_fatal_box(buf); }
#else
  { char m[4400]; snprintf(m,sizeof m,"%s\n\nDetails are in %s.log",buf,g_exename); plat_message(m,1,1); }
#endif
  fflush(NULL);
  _exit(1);
}

#if defined(_WIN32)
/* ------------------------------------------------------------------ Windows */
#include <process.h>
#include <io.h>        /* _get_osfhandle */
void pm_init(PMutex* m){ InitializeSRWLock(&m->l); }

void pm_lock(PMutex* m){ AcquireSRWLockExclusive(&m->l); }

void pm_unlock(PMutex* m){ ReleaseSRWLockExclusive(&m->l); }

void pc_init(PCond* c){ InitializeConditionVariable(&c->c); }

void pc_signal(PCond* c){ WakeConditionVariable(&c->c); }

void pc_broadcast(PCond* c){ WakeAllConditionVariable(&c->c); }

int pc_wait(PCond* c, PMutex* m, i64 ns){
  DWORD ms= ns<0? INFINITE : (DWORD)((ns+999999)/1000000);
  if(!SleepConditionVariableSRW(&c->c,&m->l,ms,0)) return GetLastError()==ERROR_TIMEOUT? 1 : 0;
  return 0; }

typedef struct { PThreadFn fn; void* arg; } TArg;

static unsigned __stdcall tstart(void* p){ TArg a=*(TArg*)p; free(p); a.fn(a.arg); return 0; }

int plat_thread(PThreadFn fn, void* arg, size_t stack){
  TArg* a=malloc(sizeof *a); a->fn=fn; a->arg=arg;
  uintptr_t h=_beginthreadex(NULL,(unsigned)stack,tstart,a,STACK_SIZE_PARAM_IS_A_RESERVATION,NULL);
  if(!h) return -1;
  { extern void plat_prof_add_thread(HANDLE); plat_prof_add_thread((HANDLE)h); }
  CloseHandle((HANDLE)h); return 0; }

void plat_thread_exit(void){ _endthreadex(0); }

static LONG g_hires;

void plat_sleep_ns(i64 ns){
  if(!g_hires){ typedef UINT (WINAPI*TBP)(UINT); HMODULE w=LoadLibraryA("winmm.dll"); if(w){ TBP f=(TBP)GetProcAddress(w,"timeBeginPeriod"); if(f) f(1);} g_hires=1; }
  if(ns<=0){ SwitchToThread(); return; }
  /* use the high-res waitable timer when the OS has it */
  static __thread HANDLE tmr; if(!tmr){ tmr=CreateWaitableTimerExW(NULL,NULL,0x00000002/*HIGH_RESOLUTION*/,TIMER_ALL_ACCESS); if(!tmr) tmr=CreateWaitableTimerW(NULL,TRUE,NULL); }
  if(tmr){ LARGE_INTEGER due; due.QuadPart=-(ns/100); if(!due.QuadPart) due.QuadPart=-1; SetWaitableTimer(tmr,&due,0,NULL,NULL,FALSE); WaitForSingleObject(tmr,INFINITE); }
  else Sleep((DWORD)(ns/1000000)); }

void plat_yield(void){ SwitchToThread(); }

/* on 2 core CPUs the shader compiler (300-400 ms per shader on a celeron) ate more than a core in busy scenes, at
   the lowest priority it only gets what the game leaves free */
void plat_thread_low_priority(void){ extern int plat_ncpu(void); SetThreadPriority(GetCurrentThread(), plat_ncpu()<=2? THREAD_PRIORITY_LOWEST : THREAD_PRIORITY_BELOW_NORMAL); }

void plat_thread_high_priority(void){ SetThreadPriority(GetCurrentThread(),THREAD_PRIORITY_ABOVE_NORMAL); }
/* SWR_CORE_PIN test: a guest thread on one host CPU per guest core, its Horizon priority (0 best .. 63, 44 the default)
   on Windows' strictly preemptive levels and no dynamic boost, so threads of one guest core run like on the console:
   never two at once, the better priority first */
void plat_thread_pin(void* h, int core, int prio){
  int n=plat_ncpu(); int cpu=n>=12? (2*core)%n : core%n; HANDLE th=h? (HANDLE)h : GetCurrentThread();
  int wp= prio<=30? THREAD_PRIORITY_HIGHEST : prio<=43? THREAD_PRIORITY_ABOVE_NORMAL : prio==44? THREAD_PRIORITY_NORMAL : prio<=47? THREAD_PRIORITY_BELOW_NORMAL : THREAD_PRIORITY_LOWEST;
  SetThreadAffinityMask(th,(DWORD_PTR)1<<cpu); SetThreadPriority(th,wp); SetThreadPriorityBoost(th,TRUE); }

u64 plat_thread_cpu_ns(void){ FILETIME c,e,k,u; if(!GetThreadTimes(GetCurrentThread(),&c,&e,&k,&u)) return 0; return ((((u64)k.dwHighDateTime<<32)|k.dwLowDateTime)+(((u64)u.dwHighDateTime<<32)|u.dwLowDateTime))*100; }

u64 plat_process_cpu_ns(void){ FILETIME c,e,k,u; if(!GetProcessTimes(GetCurrentProcess(),&c,&e,&k,&u)) return 0; return ((((u64)k.dwHighDateTime<<32)|k.dwLowDateTime)+(((u64)u.dwHighDateTime<<32)|u.dwLowDateTime))*100; }

u64 plat_ns(void){ static LARGE_INTEGER f; LARGE_INTEGER t; if(!f.QuadPart) QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t);
  return (u64)((double)t.QuadPart*1e9/(double)f.QuadPart); }

i64 plat_unix_time(void){ return (i64)time(NULL); }

/* UTC offset in seconds, DST included */
i64 plat_tz_offset(void){ TIME_ZONE_INFORMATION z; DWORD r=GetTimeZoneInformation(&z); if(r==TIME_ZONE_ID_INVALID) return 0;
  long b=z.Bias+(r==TIME_ZONE_ID_DAYLIGHT? z.DaylightBias : r==TIME_ZONE_ID_STANDARD? z.StandardBias : 0); return -(i64)b*60; }

int g_ww_ok;   /* MEM_WRITE_WATCH: Windows tracks written pages, so unchanged buffers don't need a reread */
u8* plat_vm_reserve(u64 size){ const char* gb=getenv("SWR_GPU");   /* write watch: SWR_WW=1, and always for SWR_GPU=eden (CPU writes to data Eden caches) */
  void* p=!getenv("SWR_WW") && !(gb && !strcmp(gb,"eden"))? NULL : VirtualAlloc(NULL,size,MEM_RESERVE|MEM_WRITE_WATCH,PAGE_NOACCESS);
  if(p){ g_ww_ok=1; return (u8*)p; } return (u8*)VirtualAlloc(NULL,size,MEM_RESERVE,PAGE_NOACCESS); }

/* the written pages of [base, base+size) as offsets from base; reset: also clear them (else plat_ww_reset later).
   The address buffer is per thread: the Eden bridge reads it from two threads */
int plat_ww_query(u8* base, u64 size, int reset, void (*cb)(u64)){
  static __thread PVOID a[16384]; if(!g_ww_ok) return -1;
  u8* p=base; u8* end=base+size;
  for(int it=0;it<4096 && p<end;it++){ ULONG_PTR n=16384; ULONG gran=0;
    if(GetWriteWatch(reset? WRITE_WATCH_FLAG_RESET : 0,p,(SIZE_T)(end-p),a,&n,&gran)) return -1;
    for(ULONG_PTR i=0;i<n;i++) cb((u64)((u8*)a[i]-base));
    if(n<16384) break;
    p=(u8*)a[n-1]+0x1000; }   /* without reset the same pages would come back: go on after the last one */
  return 0; }
int plat_ww_collect(u8* base, u64 size, void (*cb)(u64)){ return plat_ww_query(base,size,1,cb); }
void plat_ww_reset(u8* base, u64 size){ if(g_ww_ok) ResetWriteWatch(base,(SIZE_T)size); }

int plat_vm_commit(u8* p, u64 size){ return VirtualAlloc(p,size,MEM_COMMIT,PAGE_READWRITE)? 0 : -1; }
/* SWR_GPU=nvn and SWR_IDMEM=1: the game's addresses become the host's (docs/PLAN_NATIVE_GPU.md D4). An exe with high-entropy ASLR has nothing below 128 GB
   but two fixed Windows pages at 7ffe0000 and, now and then, its main stack: Windows draws that from about 0x1000000000 to 0xFFFFFFFFFF, and the guest
   needs its fixed addresses. Then this copy starts a new one with a new draw and passes on its exit code. */
int plat_identity_early(void){
  extern int g_identity_mem; const char* gb=getenv("SWR_GPU"); const char* id=getenv("SWR_IDMEM");
  if(id? !atoi(id) : !(gb && !strcmp(gb,"nvn"))) return 0;
  void* a=VirtualAlloc((void*)AS_BASE,0x7FFE0000ull-AS_BASE,MEM_RESERVE,PAGE_NOACCESS);
  void* b=a? VirtualAlloc((void*)0x7FFF0000ull,AS_SIZE-0x7FFF0000ull,MEM_RESERVE,PAGE_NOACCESS) : NULL;
  if(a==(void*)AS_BASE && b==(void*)0x7FFF0000ull){ g_identity_mem=1; return 1; }
  char tries[16]; DWORD n=GetEnvironmentVariableA("SWR_IDMEM_TRY",tries,sizeof tries); int t= n? atoi(tries) : 0;
  if(t>=16) fatal("Could not reserve the game's addresses (below %llu GB) after %d tries.",(unsigned long long)(AS_SIZE>>30),t);
  snprintf(tries,sizeof tries,"%d",t+1); SetEnvironmentVariableA("SWR_IDMEM_TRY",tries);
  STARTUPINFOA si; PROCESS_INFORMATION pi; memset(&si,0,sizeof si); si.cb=sizeof si;
  if(!CreateProcessA(NULL,GetCommandLineA(),NULL,NULL,TRUE,0,NULL,NULL,&si,&pi)) fatal("Could not restart the game (error %lu).",GetLastError());
  WaitForSingleObject(pi.hProcess,INFINITE); DWORD rc=0; GetExitCodeProcess(pi.hProcess,&rc); ExitProcess(rc); return 0; }

void plat_vm_readonly(u8* p, u64 size){ DWORD o; VirtualProtect(p,size,PAGE_READONLY,&o); }
void plat_vm_decommit(u8* p, u64 size){ if(size) VirtualFree(p,size,MEM_DECOMMIT); }   /* size 0 at the base would decommit the whole reservation */

/* lazily backed ranges: reserved, committed 64 KB at a time on first touch (the fault handler below). For GPU
   regions far bigger than what is ever written (Switch 2 sparse texture pools and render-target areas, 1-6 GB each):
   memory only where the game or the GPU really goes, the rest is never committed */
#define LAZY_N 256
static struct { u8* p; u64 n; } g_lazy[LAZY_N]; static volatile LONG g_lazy_lk; static int g_lazy_vh;
static volatile DWORD g_lazy_err; static u8* volatile g_lazy_err_at;
static LONG WINAPI veh_lazy(EXCEPTION_POINTERS* e){
  if(e->ExceptionRecord->ExceptionCode!=EXCEPTION_ACCESS_VIOLATION || e->ExceptionRecord->NumberParameters<2) return EXCEPTION_CONTINUE_SEARCH;
  u8* a=(u8*)e->ExceptionRecord->ExceptionInformation[1];
  for(int i=0;i<LAZY_N;i++){ u8* p=g_lazy[i].p; if(p && a>=p && a<p+g_lazy[i].n){
    MEMORY_BASIC_INFORMATION mi; if(VirtualQuery(a,&mi,sizeof mi) && mi.State==MEM_COMMIT){
      if(mi.Protect==PAGE_READWRITE) return EXCEPTION_CONTINUE_EXECUTION;   /* another thread committed it after our fault */
      return EXCEPTION_CONTINUE_SEARCH; }   /* committed: a protection fault (write traps), not ours */
    u8* b=p+(((u64)(a-p))&~0xFFFFull); u64 n=(u64)(p+g_lazy[i].n-b); if(n>0x10000) n=0x10000;
    if(VirtualAlloc(b,n,MEM_COMMIT,PAGE_READWRITE)) return EXCEPTION_CONTINUE_EXECUTION;
    g_lazy_err=GetLastError(); g_lazy_err_at=a; return EXCEPTION_CONTINUE_SEARCH; } }
  return EXCEPTION_CONTINUE_SEARCH; }
LONG plat_lazy_try(EXCEPTION_POINTERS* e){ return veh_lazy(e); }
void plat_mkdir_for(const char* path){ char d[1200]; snprintf(d,sizeof d,"%s",path); char* s=strrchr(d,'/'); if(s){ *s=0; CreateDirectoryA(d,NULL); } }
/* the crash report's view of an address: is it in a lazy range, what does Windows say, did a commit fail */
void plat_lazy_note(void* addr){ u8* a=(u8*)addr; int in=-1; for(int i=0;i<LAZY_N;i++){ u8* p=g_lazy[i].p; if(p && a>=p && a<p+g_lazy[i].n) in=i; }
  MEMORY_BASIC_INFORMATION mi={0}; VirtualQuery(a,&mi,sizeof mi);
  logf_(0,"  lazy memory: %s, Windows state %lx protect %lx, last failed commit: error %lu at %p\n",in>=0? "in a lazy range" : "not in a lazy range",(unsigned long)mi.State,(unsigned long)mi.Protect,(unsigned long)g_lazy_err,(void*)g_lazy_err_at); }
/* can this be read without faulting? (crash reports look at game memory that may have no host page) */
int plat_readable(const void* p, u64 n){ const u8* a=(const u8*)p; const u8* e=a+n;
  while(a<e){ MEMORY_BASIC_INFORMATION mi; if(!VirtualQuery(a,&mi,sizeof mi) || mi.State!=MEM_COMMIT || (mi.Protect&(PAGE_NOACCESS|PAGE_GUARD))) return 0;
    a=(const u8*)mi.BaseAddress+mi.RegionSize; }
  return 1; }
/* SWR_WATCH_PAGE=<guest hex address>: debug, logs every write to that 4 KB page (thread, host pc, the 16 bytes at
   the address before the write). The page is read-only; a write is logged, let through, and the page protected
   again 1 ms later by a helper thread */
static u8* g_wpage; static volatile LONG g_wpage_open; static u64 g_waddr; static volatile LONG g_wn;
__attribute__((weak)) void dm_guest_write_prepare(void *addr, size_t len);   /* the renderer's dirty tracker keeps counting the writes this watch takes first */
static LONG WINAPI veh_watch(EXCEPTION_POINTERS* e){
  if(e->ExceptionRecord->ExceptionCode!=EXCEPTION_ACCESS_VIOLATION || e->ExceptionRecord->NumberParameters<2 || e->ExceptionRecord->ExceptionInformation[0]!=1) return EXCEPTION_CONTINUE_SEARCH;
  u8* a=(u8*)e->ExceptionRecord->ExceptionInformation[1]; if(!g_wpage || a<g_wpage || a>=g_wpage+0x1000) return EXCEPTION_CONTINUE_SEARCH;
  if(InterlockedIncrement(&g_wn)<=600){ const u32* w=(const u32*)(g_mem+g_waddr);
    void* bt[24]; USHORT nb=RtlCaptureStackBackTrace(0,24,bt,NULL); char line[600]; int k=0;
    for(USHORT i=0;i<nb && k<(int)sizeof line-20;i++) k+=snprintf(line+k,sizeof line-k," %llx",(unsigned long long)(u64)bt[i]);
    MEMORY_BASIC_INFORMATION rm={0}; VirtualQuery((void*)e->ContextRecord->Rip,&rm,sizeof rm); logf_(0,"watch: write to %llx by thread %lu pc %p (module %p, exe %p), %llx holds %08x %08x %08x %08x; stack%s\n",(unsigned long long)(a-g_mem),GetCurrentThreadId(),(void*)e->ContextRecord->Rip,rm.AllocationBase,(void*)GetModuleHandleA(NULL),(unsigned long long)g_waddr,w[0],w[1],w[2],w[3],line);
    { const u64* c=(const u64*)e->ContextRecord->Rbx; if(plat_readable(c,50*8)){ char h[900]; int m=0; for(int i=0;i<50 && m<(int)sizeof h-20;i++) m+=snprintf(h+m,sizeof h-m," %llx",(unsigned long long)c[i]); logf_(0,"watch: rbx %p:%s\n",c,h); } } }
  DWORD o; VirtualProtect(g_wpage,0x1000,PAGE_READWRITE,&o); if(dm_guest_write_prepare) dm_guest_write_prepare(a,8); InterlockedExchange(&g_wpage_open,1); return EXCEPTION_CONTINUE_EXECUTION; }
static void watch_thread(void* arg){ (void)arg; for(;;){ Sleep(1); if(InterlockedExchange(&g_wpage_open,0)){ MEMORY_BASIC_INFORMATION mi; if(VirtualQuery(g_wpage,&mi,sizeof mi) && mi.State==MEM_COMMIT){ DWORD o; VirtualProtect(g_wpage,0x1000,PAGE_READONLY,&o); } } } }
/* "gamectl wpage ADDR": the same watch, armed while the game runs */
void plat_watch_page_arm(u64 guest){ static int on; g_waddr=guest; g_wn=0; u8* p=g_mem+(guest&~0xFFFull); DWORD o;
  if(g_wpage && g_wpage!=p) VirtualProtect(g_wpage,0x1000,PAGE_READWRITE,&o);
  g_wpage=p; if(!on){ on=1; AddVectoredExceptionHandler(1,veh_watch); _beginthread(watch_thread,1<<16,NULL); }
  VirtualProtect(p,0x1000,PAGE_READONLY,&o); logf_(0,"watch: page %llx armed\n",(unsigned long long)(guest&~0xFFFull)); }
/* "gamectl wtrace ADDR LEN [huge]": every write into [ADDR, ADDR+LEN) is single-stepped and logged with the value it left, its host pc and the guest registers. huge: only floats of 1e5 or more, or NaN */
static u8* g_tlo; static u8* g_thi; static u8* g_tplo; static u8* g_tphi; static int g_thuge; static volatile LONG g_tn; static __thread u8* t_ta;
static int wt_huge(const u8* p){ for(int k=0;k<2;k++){ u32 w; float f; memcpy(&w,p+4*k,4); memcpy(&f,&w,4); if(w && !(f>-1e5f && f<1e5f)) return 1; } return 0; }
static LONG WINAPI veh_trace(EXCEPTION_POINTERS* e){
  DWORD code=e->ExceptionRecord->ExceptionCode;
  if(code==EXCEPTION_SINGLE_STEP){ if(!t_ta) return EXCEPTION_CONTINUE_SEARCH; u8* a=t_ta; t_ta=NULL;
    if(a>=g_tlo && a<g_thi && (!g_thuge || wt_huge((const u8*)((uintptr_t)a&~7ull))) && InterlockedIncrement(&g_tn)<=3000){ const u32* q=(const u32*)((uintptr_t)a&~7ull); float f0,f1; memcpy(&f0,q,4); memcpy(&f1,q+1,4);
      void* bt[16]; USHORT nb=RtlCaptureStackBackTrace(0,16,bt,NULL); char line[300]; int k=0; for(USHORT i=0;i<nb && k<(int)sizeof line-20;i++) k+=snprintf(line+k,sizeof line-k," %llx",(unsigned long long)(u64)bt[i]);
      MEMORY_BASIC_INFORMATION rm={0}; VirtualQuery((void*)e->ContextRecord->Rip,&rm,sizeof rm); char h[420]; int m=0; const u64* c=(const u64*)e->ContextRecord->Rbx; if(plat_readable(c,33*8)) for(int i=0;i<33;i++) m+=snprintf(h+m,sizeof h-m," %llx",(unsigned long long)c[i]); else h[0]=0;
      logf_(0,"wtrace: %llx = %08x %08x (%g %g) thread %lu pc %p module %p; stack%s; rbx%s\n",(unsigned long long)(a-g_mem),q[0],q[1],(double)f0,(double)f1,GetCurrentThreadId(),(void*)e->ContextRecord->Rip,rm.AllocationBase,line,h); }
    DWORD o; VirtualProtect(g_tplo,(size_t)(g_tphi-g_tplo),PAGE_READONLY,&o); return EXCEPTION_CONTINUE_EXECUTION; }
  if(code!=EXCEPTION_ACCESS_VIOLATION || e->ExceptionRecord->NumberParameters<2 || e->ExceptionRecord->ExceptionInformation[0]!=1) return EXCEPTION_CONTINUE_SEARCH;
  u8* a=(u8*)e->ExceptionRecord->ExceptionInformation[1]; if(!g_tplo || a<g_tplo || a>=g_tphi) return EXCEPTION_CONTINUE_SEARCH;
  DWORD o; VirtualProtect(g_tplo,(size_t)(g_tphi-g_tplo),PAGE_READWRITE,&o); if(dm_guest_write_prepare) dm_guest_write_prepare(a,8);
  t_ta=a; e->ContextRecord->EFlags|=0x100; return EXCEPTION_CONTINUE_EXECUTION; }
void plat_watch_range(u64 guest, u64 len, int huge){ static int on; DWORD o; if(g_tplo) VirtualProtect(g_tplo,(size_t)(g_tphi-g_tplo),PAGE_READWRITE,&o);
  g_tn=0; g_thuge=huge; g_tlo=g_mem+guest; g_thi=g_tlo+len; g_tplo=g_mem+(guest&~0xFFFull); g_tphi=g_mem+((guest+len+0xFFF)&~0xFFFull);
  if(!on){ on=1; AddVectoredExceptionHandler(1,veh_trace); }
  VirtualProtect(g_tplo,(size_t)(g_tphi-g_tplo),PAGE_READONLY,&o); logf_(0,"wtrace: %llx+%llx armed%s\n",(unsigned long long)guest,(unsigned long long)len,huge?", huge values only":""); }
/* "gamectl bp RVA [N]": a breakpoint in the game's code (RVA in <game>_code.dll); each hit logs the guest registers, the frame in v0 and memory around x9 and x11, up to N hits */
static u8* g_bp; static u8 g_bp_old; static volatile LONG g_bpn, g_bpmax; static __thread int t_bp;
static void bp_put(u8 v){ *g_bp=v; FlushInstructionCache(GetCurrentProcess(),g_bp,1); }   /* the page stays writable while armed: threads hit this at once, and changing its protection back from one of them faults the other */
static volatile u64 g_bp_min;   /* bp at a function's first instruction: its CPU* is in rcx; only hits with x3 at least this */
void plat_bp_min(u64 m){ g_bp_min=m; }
static LONG WINAPI veh_bp(EXCEPTION_POINTERS* e){
  DWORD code=e->ExceptionRecord->ExceptionCode;
  if(code==EXCEPTION_SINGLE_STEP && t_bp){ t_bp=0; if(g_bp && g_bpn<g_bpmax) bp_put(0xCC); return EXCEPTION_CONTINUE_EXECUTION; }
  if(code!=EXCEPTION_BREAKPOINT || (u8*)e->ExceptionRecord->ExceptionAddress!=g_bp) return EXCEPTION_CONTINUE_SEARCH;
  const u64* c=(const u64*)(g_bp_min? e->ContextRecord->Rcx : e->ContextRecord->Rbx); if(g_bp_min && (!plat_readable(c,33*8) || c[3]<g_bp_min)){ bp_put(g_bp_old); t_bp=1; e->ContextRecord->EFlags|=0x100; return EXCEPTION_CONTINUE_EXECUTION; }
  LONG n=InterlockedIncrement(&g_bpn);
  if(n<=g_bpmax && plat_readable(c,33*8)){ char h[900]; int m=0; for(int i=0;i<31;i++) m+=snprintf(h+m,sizeof h-m," %llx",(unsigned long long)c[i]);
    f32 v0; memcpy(&v0,(const u8*)c+280,4); char d[300]=""; int k=0; u64 x9=c[8], x11=c[10];
    if(x9>0x1000 && plat_readable(g_mem+x9,320)){ const u32* a=(const u32*)(g_mem+x9); static const short w[]={0,1,2,3,10,11,12,13,16,17,66,67,68,69,70,71,72,73,74,75}; for(int i=0;i<20;i++) k+=snprintf(d+k,sizeof d-k>0?sizeof d-k:0,"%x ",a[w[i]]); }
    char q[120]=""; if(x11>0x1000 && plat_readable(g_mem+x11,16)){ const u32* a=(const u32*)(g_mem+x11); snprintf(q,sizeof q,"%08x %08x %08x %08x",a[0],a[1],a[2],a[3]); }
    logf_(0,"bphit: thread %lu v0 %g regs%s | data %s | x9mem %s\n",GetCurrentThreadId(),(double)v0,h,q,d); }
  bp_put(g_bp_old); t_bp=1; e->ContextRecord->EFlags|=0x100; return EXCEPTION_CONTINUE_EXECUTION; }
void plat_bp(u64 rva, long n){ static int on; char cm[300]; snprintf(cm,sizeof cm,"%s_code.dll",g_exename); HMODULE m=GetModuleHandleA(cm); if(!m){ logf_(0,"bp: no code module\n"); return; }
  if(g_bp) bp_put(g_bp_old);
  { DWORD o; VirtualProtect((u8*)m+rva,1,PAGE_EXECUTE_READWRITE,&o); } g_bp=(u8*)m+rva; g_bp_old=*g_bp; g_bpn=0; g_bpmax=n; if(!on){ on=1; AddVectoredExceptionHandler(1,veh_bp); } bp_put(0xCC); logf_(0,"bp: armed at %p, byte %02x\n",g_bp,g_bp_old); }
void plat_watch_page_poll(void){   /* called each frame: arms the watch once the page is committed */
  static int armed; const char* e; if(armed || !(e=getenv("SWR_WATCH_PAGE"))) return;
  g_waddr=strtoull(e,NULL,16); u8* p=g_mem+(g_waddr&~0xFFFull); MEMORY_BASIC_INFORMATION mi;
  if(!VirtualQuery(p,&mi,sizeof mi) || mi.State!=MEM_COMMIT) return;
  armed=1; g_wpage=p; AddVectoredExceptionHandler(1,veh_watch); DWORD o; VirtualProtect(p,0x1000,PAGE_READONLY,&o); _beginthread(watch_thread,1<<16,NULL);
  logf_(0,"watch: page %llx armed\n",(unsigned long long)(g_waddr&~0xFFFull)); }

int plat_vm_lazy_add(u8* p, u64 size){
  while(InterlockedExchange(&g_lazy_lk,1)) Sleep(0);
  if(!g_lazy_vh){ g_lazy_vh=1; AddVectoredExceptionHandler(1,veh_lazy); }
  int ok=-1; for(int i=0;i<LAZY_N;i++) if(!g_lazy[i].p){ g_lazy[i].n=size; MemoryBarrier(); g_lazy[i].p=p; ok=0; break; }
  InterlockedExchange(&g_lazy_lk,0); return ok; }
void plat_vm_lazy_remove(u8* p, u64 size){
  while(InterlockedExchange(&g_lazy_lk,1)) Sleep(0);
  for(int i=0;i<LAZY_N;i++) if(g_lazy[i].p==p){ g_lazy[i].p=NULL; break; }
  InterlockedExchange(&g_lazy_lk,0); plat_vm_decommit(p,size); }

void* plat_dl_open(const char* path){ return (void*)LoadLibraryA(path); }

void* plat_dl_sym(void* h, const char* n){ return (void*)GetProcAddress((HMODULE)h,n); }

int plat_mkdir(const char* p){ return CreateDirectoryA(p,NULL)? 0 : -1; }

int plat_is_dir(const char* p){ DWORD a=GetFileAttributesA(p); return a!=INVALID_FILE_ATTRIBUTES && (a&FILE_ATTRIBUTE_DIRECTORY); }

int plat_dir_open(PDir* d, const char* path){ char p[1100]; snprintf(p,sizeof p,"%s\\*",path); WIN32_FIND_DATAA fd; HANDLE h=FindFirstFileA(p,&fd);
  if(h==INVALID_HANDLE_VALUE) return -1; d->h=h; d->first=1; strncpy(d->name,fd.cFileName,511); d->is_dir=(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)!=0;
  d->size=((u64)fd.nFileSizeHigh<<32)|fd.nFileSizeLow; return 0; }

int plat_dir_next(PDir* d){
  for(;;){
    if(d->first){ d->first=0; }
    else { WIN32_FIND_DATAA fd; if(!FindNextFileA((HANDLE)d->h,&fd)) return 0; strncpy(d->name,fd.cFileName,511); d->is_dir=(fd.dwFileAttributes&FILE_ATTRIBUTE_DIRECTORY)!=0; d->size=((u64)fd.nFileSizeHigh<<32)|fd.nFileSizeLow; }
    if(strcmp(d->name,".") && strcmp(d->name,"..")) return 1; } }

void plat_dir_close(PDir* d){ FindClose((HANDLE)d->h); }

u64 plat_file_size(const char* p){ WIN32_FILE_ATTRIBUTE_DATA a; if(!GetFileAttributesExA(p,GetFileExInfoStandard,&a)) return 0; return ((u64)a.nFileSizeHigh<<32)|a.nFileSizeLow; }

int plat_pread(FILE* f, void* buf, u64 size, u64 off){
  HANDLE h=(HANDLE)_get_osfhandle(_fileno(f)); OVERLAPPED ov; memset(&ov,0,sizeof ov); ov.Offset=(DWORD)off; ov.OffsetHigh=(DWORD)(off>>32);
  DWORD got=0; if(ReadFile(h,buf,(DWORD)size,&got,&ov)) return (int)got;
  if(GetLastError()!=ERROR_NOACCESS) return -1;
  /* the destination has read-only pages (the Eden backend's write traps on memory it caches): the kernel does not
     write those, a copy from a bounce buffer does (and is seen as the CPU write it is) */
  static __thread u8* bb; if(!bb && !(bb=malloc(1<<20))) return -1;
  u64 done=0; while(done<size){ DWORD want=(DWORD)(size-done>(1u<<20)? (1u<<20) : size-done), n=0; u64 at=off+done;
    memset(&ov,0,sizeof ov); ov.Offset=(DWORD)at; ov.OffsetHigh=(DWORD)(at>>32);
    if(!ReadFile(h,bb,want,&n,&ov)) return done? (int)done : -1;
    memcpy((u8*)buf+done,bb,n); done+=n; if(n<want) break; }
  return (int)done; }

int plat_ncpu(void){ SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors; }

void plat_battery(int* pct, int* chg){ SYSTEM_POWER_STATUS s; *pct=100; *chg=1;
  if(!GetSystemPowerStatus(&s) || (s.BatteryFlag&128) || s.BatteryLifePercent>100) return;
  *pct=s.BatteryLifePercent; *chg= s.ACLineStatus==1; }

/* a crash inside a system DLL (memcpy in msvcrt, the video driver) doesn't say anything by itself, so we log the
   return addresses into the runtime that are on the stack, those tell you what code called it */
static void log_runtime_callers(CONTEXT* c){
  extern char __ImageBase; u64 b=(u64)(size_t)&__ImageBase;
  IMAGE_NT_HEADERS* nt=(IMAGE_NT_HEADERS*)(b+(u64)((IMAGE_DOS_HEADER*)b)->e_lfanew); u64 end=b+nt->OptionalHeader.SizeOfImage;
  if(c->Rip>=b && c->Rip<end) return;
  u64* sp=(u64*)(size_t)c->Rsp; u64* top=(u64*)((NT_TIB*)NtCurrentTeb())->StackBase;
  int n=0; for(; sp<top && n<8; sp++) if(*sp>b+0x1000 && *sp<end){ if(!n) logf_(0,"  called from the runtime at"); logf_(0," +%llx",(unsigned long long)(*sp-b)); n++; }
  if(n) logf_(0," (exe offsets)\n"); }

static DWORD g_guard_tls=TLS_OUT_OF_INDEXES;

void plat_guard_enter(void* rec){ if(g_guard_tls==TLS_OUT_OF_INDEXES) return; TlsSetValue(g_guard_tls,rec); }

void* plat_guard_get(void){ return g_guard_tls==TLS_OUT_OF_INDEXES? NULL : TlsGetValue(g_guard_tls); }

volatile int g_driver_recovered;   /* times we jumped out of a driver crash */

/* an uncaught C++ exception (GCC's SEH code 0x20474343): its type and what(). The argument is the _Unwind_Exception;
   the __cxa_exception header ends there, with the type_info first (80 bytes before it), the thrown object after it */
#define CXX_UNWIND_SIZE 64
static void log_cxx_exception(EXCEPTION_RECORD* r){
  if(r->ExceptionCode!=0x20474343 || r->NumberParameters<1) return;
  u8* ue=(u8*)r->ExceptionInformation[0]; if(!plat_readable(ue-80,16)) return;
  u8* ti=*(u8**)(ue-80); if(!plat_readable(ti,16)) return;
  const char* name=*(const char**)(ti+8); if(!plat_readable(name,1)) return;
  logf_(0,"  uncaught C++ exception of type %s\n",name);
  void* obj=ue+CXX_UNWIND_SIZE; void** vt=*(void***)obj;
  if((strstr(name,"rror") || strstr(name,"xception") || !strncmp(name,"St",2)) && plat_readable(vt,24)){
    const char* w=((const char*(*)(void*))vt[2])(obj); if(w && plat_readable(w,1)) logf_(0,"  what(): %s\n",w); } }

static LONG WINAPI crash_filter(EXCEPTION_POINTERS* e){
  plat_jmpbuf* b=(plat_jmpbuf*)plat_guard_get();
  if(b){ /* the video driver crashed in a guarded call (shader compile): drop that shader, keep playing */
    { extern void gpu_crash_note(void); gpu_crash_note(); }
    plat_guard_enter(NULL); g_driver_recovered++;
    logf_(0,"  (exception %08lx at %p inside the video driver, recovered: only that effect is missing)\n",(unsigned long)e->ExceptionRecord->ExceptionCode,(void*)e->ContextRecord->Rip);
    __builtin_longjmp(*b,1); }
  extern void crash_report(void* host_pc, void* fault_addr, int sig);
  log_runtime_callers(e->ContextRecord);
  log_cxx_exception(e->ExceptionRecord);
  { extern void gpu_crash_note(void); gpu_crash_note(); }
  crash_report((void*)e->ContextRecord->Rip,(void*)e->ExceptionRecord->ExceptionInformation[1],(int)e->ExceptionRecord->ExceptionCode);
  return EXCEPTION_EXECUTE_HANDLER; }

/* The last vectored handler sees every hardware fault the lazy-memory and write-trap handlers left, before any SEH
   frame can end the process without a word: the crash of 2026-11-04 left an exit code, no report, no Windows event.
   Faults a driver catches itself pass through here too, so this line alone is not a crash. */
static void log_last_chance(EXCEPTION_POINTERS* e, DWORD c){
  static volatile LONG seen; if(plat_guard_get() || InterlockedIncrement(&seen)>8) return;
  extern char __ImageBase; u64 b=(u64)(size_t)&__ImageBase; CONTEXT* x=e->ContextRecord; EXCEPTION_RECORD* r=e->ExceptionRecord;
  IMAGE_NT_HEADERS* nt=(IMAGE_NT_HEADERS*)(b+(u64)((IMAGE_DOS_HEADER*)b)->e_lfanew); u64 end=b+nt->OptionalHeader.SizeOfImage;
  logf_(0,"\n*** Last chance: exception %08lx at %p (exe offset +%llx), %s %p, thread %lu ***\n",(unsigned long)c,(void*)x->Rip,
        (unsigned long long)(x->Rip>=b && x->Rip<end? x->Rip-b : 0),r->NumberParameters>=2 && r->ExceptionInformation[0]? "write to":"read of",
        r->NumberParameters>=2? (void*)r->ExceptionInformation[1]:NULL,(unsigned long)GetCurrentThreadId());
  logf_(0,"  rax %llx rcx %llx rdx %llx rbx %llx rsi %llx rdi %llx r8 %llx r9 %llx\n",(unsigned long long)x->Rax,(unsigned long long)x->Rcx,(unsigned long long)x->Rdx,
        (unsigned long long)x->Rbx,(unsigned long long)x->Rsi,(unsigned long long)x->Rdi,(unsigned long long)x->R8,(unsigned long long)x->R9);
  u64* sp=(u64*)(size_t)x->Rsp; u64* top=(u64*)((NT_TIB*)NtCurrentTeb())->StackBase; int n=0;
  logf_(0,"  return addresses (exe offsets):");
  for(; sp<top && n<24; sp++) if(*sp>b+0x1000 && *sp<end){ logf_(0," +%llx",(unsigned long long)(*sp-b)); n++; }
  logf_(0,"\n"); }

static LONG WINAPI veh_fatal(EXCEPTION_POINTERS* e){
  DWORD c=e->ExceptionRecord->ExceptionCode;
  if(c==0xC0000005 /* access violation */ || c==0xC0000006 /* in-page error: a mapped file on a full disk */ || c==0xC000001D /* illegal instruction */) log_last_chance(e,c);
  if(c==EXCEPTION_STACK_OVERFLOW || c==0xC0000409 /* STACK_BUFFER_OVERRUN / fastfail */ || c==0xC0000374 /* heap corruption */){
    static volatile LONG once; if(!InterlockedExchange(&once,1)){ logf_(0,"\n*** Shutdown: exception %08lx at %p (stack/heap) ***\n",(unsigned long)c,(void*)e->ContextRecord->Rip);
      /* what lies on the stack that points into the exe: the way here, for nm on a build with symbols */
      { u64 base=(u64)(uintptr_t)GetModuleHandleA(NULL); const IMAGE_NT_HEADERS* nt=(const IMAGE_NT_HEADERS*)(base+((const IMAGE_DOS_HEADER*)(uintptr_t)base)->e_lfanew); u64 hi=base+nt->OptionalHeader.SizeOfImage; u64 st[512]; SIZE_T got=0;
        ReadProcessMemory(GetCurrentProcess(),(const void*)(uintptr_t)e->ContextRecord->Rsp,st,sizeof st,&got);
        char b[900]; int n=0, k=0; for(size_t i=0;i<got/8 && k<28 && n<860;i++) if(st[i]>base && st[i]<hi){ n+=snprintf(b+n,sizeof b-n," %llx",(unsigned long long)(st[i]-base)); k++; }
        logf_(0,"exe offsets on the stack of thread %lu:%s\n",(unsigned long)GetCurrentThreadId(),b); } } }
  return EXCEPTION_CONTINUE_SEARCH; }

#include <signal.h>

static void on_abort(int s){ logf_(0,"\n*** Shutdown: abort() (signal %d), probably the video driver or out of memory ***\n",s); }

static void on_exit_log(void){ logf_(0,"program shutdown\n"); }

static void inv_param(const wchar_t* e, const wchar_t* f, const wchar_t* fi, unsigned l, uintptr_t r){ (void)e;(void)f;(void)fi;(void)l;(void)r; logf_(0,"warning: invalid parameter in the C library\n"); }

void plat_install_crash_handler(void){ g_guard_tls=TlsAlloc(); SetUnhandledExceptionFilter(crash_filter);
  AddVectoredExceptionHandler(0,veh_fatal); signal(SIGABRT,on_abort); atexit(on_exit_log); _set_invalid_parameter_handler(inv_param);
  /* keep the display/system awake while playing */
  SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_DISPLAY_REQUIRED); }

/* every host thread's call stack (return addresses, walked with the x64 unwind data) into the log, for stalls that
   happen outside the game's code (SWR_GPU=eden: Eden's GPU threads). Each thread is suspended only while its
   addresses are copied; nothing that may take a lock runs then. addr2line on an unstripped runtime
   (SWR_KEEP_SYMBOLS=1) names them: address - exe base + 0x140000000 */
#include <tlhelp32.h>
void plat_dump_native_stacks(void){
  typedef HRESULT (WINAPI *GTD)(HANDLE,PWSTR*); static GTD gtd; if(!gtd) gtd=(GTD)GetProcAddress(GetModuleHandleA("kernel32.dll"),"GetThreadDescription");
  DWORD me=GetCurrentThreadId(), pid=GetCurrentProcessId(); HANDLE snap=CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD,0); if(snap==INVALID_HANDLE_VALUE) return;
  logf_(0,"native stacks (exe base %llx):\n",(unsigned long long)(u64)GetModuleHandleA(NULL));
  THREADENTRY32 te; te.dwSize=sizeof te;
  for(BOOL ok=Thread32First(snap,&te); ok; ok=Thread32Next(snap,&te)){
    if(te.th32OwnerProcessID!=pid || te.th32ThreadID==me) continue;
    HANDLE t=OpenThread(THREAD_SUSPEND_RESUME|THREAD_GET_CONTEXT|THREAD_QUERY_INFORMATION|THREAD_QUERY_LIMITED_INFORMATION,FALSE,te.th32ThreadID); if(!t) continue;
    char name[64]=""; if(gtd){ PWSTR w=NULL; if(SUCCEEDED(gtd(t,&w)) && w){ WideCharToMultiByte(CP_UTF8,0,w,-1,name,sizeof name,NULL,NULL); LocalFree(w); } }
    u64 pc[32]; int np=0; CONTEXT c; memset(&c,0,sizeof c); c.ContextFlags=CONTEXT_FULL;
    if(SuspendThread(t)!=(DWORD)-1){
      if(GetThreadContext(t,&c)) for(; np<32 && c.Rip; np++){
        pc[np]=c.Rip; DWORD64 ib=0; PRUNTIME_FUNCTION rf=RtlLookupFunctionEntry(c.Rip,&ib,NULL);
        if(!rf){ if(!c.Rsp) break; c.Rip=*(DWORD64*)c.Rsp; c.Rsp+=8; continue; }   /* leaf function: return address on top */
        PVOID hd=NULL; DWORD64 ef=0; RtlVirtualUnwind(UNW_FLAG_NHANDLER,ib,c.Rip,rf,&c,&hd,&ef,NULL); }
      ResumeThread(t); }
    CloseHandle(t);
    char line[1024]; int n=snprintf(line,sizeof line,"  thread %lu %s:",(unsigned long)te.th32ThreadID,name);
    for(int i=0;i<np && n<(int)sizeof line-20;i++) n+=snprintf(line+n,sizeof line-n," %llx",(unsigned long long)pc[i]);
    logf_(0,"%s\n",line); }
  CloseHandle(snap); }

/* SWR_HOST_PROF=<s>: from second <s> on, 10 s of samples of every host thread (Eden's GPU threads, the driver's,
   the game's): host_profile.txt gets each thread's CPU time in that window and its most frequent pcs (exe offsets,
   or module+offset). tools resolve exe offsets with nm on an unstripped runtime (SWR_KEEP_SYMBOLS=1) */
#define HP_THREADS 256
#define HP_PCS 512
int plat_thread_pc(void* h, u64* pc); int plat_addr_module(u64 a, char* name, int n, u64* off);
/* SWR_HOST_PROF: the render thread's samples outside the exe, by the return addresses into the exe on its stack, so a wait or a driver call has a name */
volatile DWORD g_host_focus_tid; enum { HS_N = 4096 }; static struct { u32 off[4]; u32 n; } g_hs[HS_N];
static void host_stack(HANDLE h, u64 base){
  static u64 lo, hi; if(!hi){ const u8* b=(const u8*)(uintptr_t)base; const IMAGE_NT_HEADERS* nt=(const IMAGE_NT_HEADERS*)(b+((const IMAGE_DOS_HEADER*)b)->e_lfanew); const IMAGE_SECTION_HEADER* sh=IMAGE_FIRST_SECTION(nt);
    for(int i=0;i<nt->FileHeader.NumberOfSections;i++) if(!memcmp(sh[i].Name,".text",5)){ lo=base+sh[i].VirtualAddress; hi=lo+sh[i].Misc.VirtualSize; } if(!hi) hi=1; }
  if(SuspendThread(h)==(DWORD)-1) return; CONTEXT cx; memset(&cx,0,sizeof cx); cx.ContextFlags=CONTEXT_CONTROL; u64 st[640]; SIZE_T got=0;
  int ok=GetThreadContext(h,&cx) && ReadProcessMemory(GetCurrentProcess(),(const void*)(uintptr_t)cx.Rsp,st,sizeof st,&got); if(!got) ok=0; ResumeThread(h);
  if(!ok || (cx.Rip>=lo && cx.Rip<hi)) return;
  u32 off[4]={0,0,0,0}; int no=0; for(size_t i=0;i<got/8 && no<4;i++) if(st[i]>lo && st[i]<hi){ u32 o=(u32)(st[i]-base); if(!no || off[no-1]!=o) off[no++]=o; }
  if(!no) return; u32 k=((off[0]^(off[1]*7)^(off[2]*13)^(off[3]*17))*2654435761u)>>20; for(int t=0;t<HS_N;t++){ u32 q=(k+t)&(HS_N-1); if(!g_hs[q].n || !memcmp(g_hs[q].off,off,sizeof off)){ memcpy(g_hs[q].off,off,sizeof off); g_hs[q].n++; break; } } }
static void host_prof_thread(void* arg){
  double start=atof((const char*)arg); Sleep((DWORD)(start*1000));
  static struct { DWORD tid; HANDLE h; char name[64]; u64 cpu0; u32 n; struct { u64 pc; u32 n; } pc[HP_PCS]; } T[HP_THREADS]; int nt=0;
  typedef HRESULT (WINAPI *GTD)(HANDLE,PWSTR*); GTD gtd=(GTD)GetProcAddress(GetModuleHandleA("kernel32.dll"),"GetThreadDescription");
  DWORD me=GetCurrentThreadId(), pid=GetCurrentProcessId(); u64 base=(u64)(uintptr_t)GetModuleHandleA(NULL);
  HANDLE snap=CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD,0); THREADENTRY32 te; te.dwSize=sizeof te;
  for(BOOL ok=Thread32First(snap,&te); ok && nt<HP_THREADS; ok=Thread32Next(snap,&te)){
    if(te.th32OwnerProcessID!=pid || te.th32ThreadID==me) continue;
    HANDLE h=OpenThread(THREAD_SUSPEND_RESUME|THREAD_GET_CONTEXT|THREAD_QUERY_INFORMATION|THREAD_QUERY_LIMITED_INFORMATION,FALSE,te.th32ThreadID); if(!h) continue;
    memset(&T[nt],0,sizeof T[nt]); T[nt].tid=te.th32ThreadID; T[nt].h=h;
    if(gtd){ PWSTR w=NULL; if(SUCCEEDED(gtd(h,&w)) && w){ WideCharToMultiByte(CP_UTF8,0,w,-1,T[nt].name,64,NULL,NULL); LocalFree(w); } }
    FILETIME c,e,k,u; if(GetThreadTimes(h,&c,&e,&k,&u)) T[nt].cpu0=((u64)k.dwHighDateTime<<32|k.dwLowDateTime)+((u64)u.dwHighDateTime<<32|u.dwLowDateTime);
    nt++; }
  CloseHandle(snap);
  u64 t0=plat_ns();
  while(plat_ns()-t0<10000000000ull){
    for(int i=0;i<nt;i++){ u64 pc; if(T[i].tid==g_host_focus_tid) host_stack(T[i].h,base); if(!plat_thread_pc(T[i].h,&pc)) continue; T[i].n++;
      int j=0; for(;j<HP_PCS && T[i].pc[j].pc;j++) if(T[i].pc[j].pc==pc){ T[i].pc[j].n++; break; }
      if(j<HP_PCS && !T[i].pc[j].pc){ T[i].pc[j].pc=pc; T[i].pc[j].n=1; } }
    Sleep(1); }
  FILE* f=fopen("host_profile.txt","w"); if(!f) return;
  fprintf(f,"host profile, 10 s from second %.0f, exe base %llx\n",start,(unsigned long long)base);
  for(int i=0;i<nt;i++){
    FILETIME c,e,k,u; u64 cpu1=T[i].cpu0; if(GetThreadTimes(T[i].h,&c,&e,&k,&u)) cpu1=((u64)k.dwHighDateTime<<32|k.dwLowDateTime)+((u64)u.dwHighDateTime<<32|u.dwLowDateTime);
    double pct=(double)(cpu1-T[i].cpu0)/1e5/10.0;   /* 100 ns units over 10 s, in % of a core */
    if(pct<2) continue;
    fprintf(f,"thread %lu '%s': %.0f%% of a core, %u samples\n",(unsigned long)T[i].tid,T[i].name,pct,T[i].n);
    for(int r=0;r<HP_PCS;r++){ int best=-1; for(int j=0;j<HP_PCS;j++) if(T[i].pc[j].n && (best<0 || T[i].pc[j].n>T[i].pc[best].n)) best=j; if(best<0) break;
      u64 pc=T[i].pc[best].pc; char mod[64]; u64 off=0;
      if(pc>=base && pc<base+0x8000000) fprintf(f,"   %5u exe+%llx\n",T[i].pc[best].n,(unsigned long long)(pc-base));
      else if(plat_addr_module(pc,mod,sizeof mod,&off)) fprintf(f,"   %5u %s+%llx\n",T[i].pc[best].n,mod,(unsigned long long)off);
      else fprintf(f,"   %5u %llx\n",T[i].pc[best].n,(unsigned long long)pc);
      T[i].pc[best].n=0; } }
  for(int r=0;r<80;r++){ int best=-1; for(int j=0;j<HS_N;j++) if(g_hs[j].n && (best<0 || g_hs[j].n>g_hs[best].n)) best=j; if(best<0) break;
    fprintf(f,"stack %u +%x +%x +%x +%x\n",g_hs[best].n,g_hs[best].off[0],g_hs[best].off[1],g_hs[best].off[2],g_hs[best].off[3]); g_hs[best].n=0; }
  fclose(f); }
void plat_host_prof_start(void){ const char* e=getenv("SWR_HOST_PROF"); if(e) _beginthread(host_prof_thread,1<<16,(void*)e); }

/* ------------------------------------------------------------------ sampling profiler (profile=1) ~1 ms tick:
   suspend each thread, grab rip, map it to a game function / exe offset / module. profile.txt gets rewritten every
   10 s */
#include "rt.h"
static HANDLE g_pth[256]; static volatile LONG g_npth;

void plat_prof_add_thread(HANDLE h){ HANDLE d=NULL; if(!DuplicateHandle(GetCurrentProcess(),h,GetCurrentProcess(),&d,0,FALSE,DUPLICATE_SAME_ACCESS)) return;
  LONG i=InterlockedIncrement(&g_npth)-1; if(i<256) g_pth[i]=d; else CloseHandle(d); }

typedef struct { u64 host; u32 idx; } PEnt;

static PEnt* g_pe; static u32 g_npe; static u64 g_plo, g_phi; static u32* g_pcnt;
#define PMOD 32
static struct { u64 base; char name[64]; u32 n; } g_pmod[PMOD]; static int g_npmod;

#define PEXE 4096
static struct { u32 off; u32 n; } g_pexe[PEXE];

static u64 g_ptot, g_pguest, g_pexe_n;

static int pe_cmp(const void* a, const void* b){ u64 x=((const PEnt*)a)->host, y=((const PEnt*)b)->host; return x<y? -1 : x>y; }

static void prof_note(u64 rip){
  g_ptot++;
  if(rip>=g_plo && rip<g_phi+0x4000){ u32 lo=0, hi=g_npe; while(hi-lo>1){ u32 m=(lo+hi)/2; if(g_pe[m].host<=rip) lo=m; else hi=m; } g_pcnt[lo]++; g_pguest++; return; }
  HMODULE hm=NULL; if(!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,(LPCSTR)(uintptr_t)rip,&hm) || !hm){ hm=NULL; }
  u64 base=(u64)(uintptr_t)hm;
  if(hm==GetModuleHandleA(NULL)){ u32 off=(u32)((rip-base)&~0x3Full); u32 k=(off*2654435761u)>>20; for(int t=0;t<PEXE;t++){ u32 s=(k+t)&(PEXE-1); if(!g_pexe[s].n || g_pexe[s].off==off){ g_pexe[s].off=off; g_pexe[s].n++; break; } } g_pexe_n++; }
  int i; for(i=0;i<g_npmod;i++) if(g_pmod[i].base==base) break;
  if(i==g_npmod && g_npmod<PMOD){ g_pmod[i].base=base; g_pmod[i].n=0; if(!hm) strcpy(g_pmod[i].name,"(no module: generated/JIT code)"); else { char pth[MAX_PATH]; GetModuleFileNameA(hm,pth,MAX_PATH); const char* s=strrchr(pth,'\\'); snprintf(g_pmod[i].name,64,"%s",s? s+1 : pth); } g_npmod++; }
  if(i<g_npmod) g_pmod[i].n++; }

static const char* g_pfile="profile.txt"; static int g_pboot;

/* one thread on its own (the GPU thread): its samples by exe offset (64 byte buckets), and how many were
   elsewhere (driver, C runtime, waits), so "where does the GPU thread spend a frame" has an answer */
static HANDLE g_pfoc; static struct { u32 off; u32 n; } g_pfx[PEXE]; static u64 g_pfx_tot, g_pfx_other, g_pfx_wait;
void plat_prof_focus_self(void){ HANDLE d=NULL;
  if(DuplicateHandle(GetCurrentProcess(),GetCurrentThread(),GetCurrentProcess(),&d,THREAD_SUSPEND_RESUME|THREAD_GET_CONTEXT|THREAD_QUERY_INFORMATION,FALSE,0)) g_pfoc=d; }
/* waits of the focus thread, by the first return address into the exe found on its stack (who waited) */
static struct { u32 off[6]; u32 n; } g_pfw[PEXE];
static void prof_note_wait(u64 rsp){
  static u64 lo, hi; HMODULE me=GetModuleHandleA(NULL);
  if(!hi){ /* the exe's .text only: data pointers on the stack aren't return addresses */
    const u8* b=(const u8*)me; const IMAGE_NT_HEADERS* nt=(const IMAGE_NT_HEADERS*)(b+((const IMAGE_DOS_HEADER*)b)->e_lfanew);
    const IMAGE_SECTION_HEADER* sh=IMAGE_FIRST_SECTION(nt);
    for(int i=0;i<nt->FileHeader.NumberOfSections;i++) if(!memcmp(sh[i].Name,".text",5)){ lo=(u64)(uintptr_t)b+sh[i].VirtualAddress; hi=lo+sh[i].Misc.VirtualSize; }
    if(!hi){ lo=(u64)(uintptr_t)b; hi=lo+0x2000000; } }
  const u64* s=(const u64*)(uintptr_t)rsp; u32 off[6]={0,0,0,0,0,0}; int no=0;
  MEMORY_BASIC_INFORMATION mi; if(!VirtualQuery(s,&mi,sizeof mi) || mi.State!=MEM_COMMIT) return;
  u64 end=(u64)(uintptr_t)mi.BaseAddress+mi.RegionSize; int n=(int)((end-rsp)/8); if(n>4096) n=4096;
  for(int i=0;i<n && no<6;i++){ u64 v=s[i]; if(v>lo && v<hi) off[no++]=(u32)(v-(u64)(uintptr_t)me); }
  if(!no) return; u32 k=((off[0]^(off[1]*7)^(off[2]*13)^(off[3]*17)^(off[4]*19)^(off[5]*23))*2654435761u)>>20;
  for(int t=0;t<PEXE;t++){ u32 q=(k+t)&(PEXE-1); if(!g_pfw[q].n || !memcmp(g_pfw[q].off,off,sizeof off)){ memcpy(g_pfw[q].off,off,sizeof off); g_pfw[q].n++; break; } } }
static void prof_note_focus(u64 rip){
  g_pfx_tot++;
  HMODULE hm=NULL; if(!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,(LPCSTR)(uintptr_t)rip,&hm)) hm=NULL;
  if(hm && hm==GetModuleHandleA(NULL)){ u32 off=(u32)((rip-(u64)(uintptr_t)hm)&~0x3Full); u32 k=(off*2654435761u)>>20;
    for(int t=0;t<PEXE;t++){ u32 s=(k+t)&(PEXE-1); if(!g_pfx[s].n || g_pfx[s].off==off){ g_pfx[s].off=off; g_pfx[s].n++; break; } } return; }
  char pth[MAX_PATH]="?"; if(hm) GetModuleFileNameA(hm,pth,MAX_PATH); const char* s=strrchr(pth,'\\'); s= s? s+1 : pth;
  if(!_stricmp(s,"ntdll.dll")||!_stricmp(s,"kernelbase.dll")||!_stricmp(s,"win32u.dll")||!_stricmp(s,"kernel32.dll")) g_pfx_wait++; else g_pfx_other++; }

static void prof_write(void){
  FILE* f=fopen(g_pfile,"w"); if(!f) return;
  if(g_pfx_tot){ fprintf(f,"GPU thread: %llu samples, %.1f%% waiting, %.1f%% in other modules (driver, C runtime), the rest in the exe:\n",(unsigned long long)g_pfx_tot,100.0*g_pfx_wait/g_pfx_tot,100.0*g_pfx_other/g_pfx_tot);
    for(int r=0;r<60;r++){ int best=-1; u32 bn=0; for(int s=0;s<PEXE;s++) if(g_pfx[s].n>bn && !(g_pfx[s].n&0x80000000u)){ bn=g_pfx[s].n; best=s; }
      if(best<0 || bn*1000ull<g_pfx_tot) break; fprintf(f,"  %6.2f%%  +%x\n",100.0*bn/g_pfx_tot,g_pfx[best].off); g_pfx[best].n|=0x80000000u; }
    for(int s=0;s<PEXE;s++) g_pfx[s].n&=0x7FFFFFFFu;
    fprintf(f,"focus thread outside the exe (waits, driver), by the runtime calls on its stack:\n");
    for(int r=0;r<30;r++){ int best=-1; u32 bn=0; for(int s=0;s<PEXE;s++) if(g_pfw[s].n>bn && !(g_pfw[s].n&0x80000000u)){ bn=g_pfw[s].n; best=s; }
      if(best<0 || bn*1000ull<g_pfx_tot) break; fprintf(f,"  %6.2f%%  +%x +%x +%x +%x +%x +%x\n",100.0*bn/g_pfx_tot,g_pfw[best].off[0],g_pfw[best].off[1],g_pfw[best].off[2],g_pfw[best].off[3],g_pfw[best].off[4],g_pfw[best].off[5]); g_pfw[best].n|=0x80000000u; }
    for(int s=0;s<PEXE;s++) g_pfw[s].n&=0x7FFFFFFFu; fprintf(f,"\n"); }
  /* idle threads sit in ntdll/kernelbase/win32u, so percentages are of non-idle samples */
  u64 idle=0; for(int i=0;i<g_npmod;i++){ const char* n=g_pmod[i].name; if(!_stricmp(n,"ntdll.dll")||!_stricmp(n,"kernelbase.dll")||!_stricmp(n,"win32u.dll")||!_stricmp(n,"kernel32.dll")) idle+=g_pmod[i].n; }
  u64 act= g_ptot>idle? g_ptot-idle : 1;
  fprintf(f,"SwitchRecomp profile: %llu samples, %llu active (excluding waits). %% of active time\n\n",(unsigned long long)g_ptot,(unsigned long long)act);
  fprintf(f,"modules:\n"); for(int i=0;i<g_npmod;i++) { const char* n=g_pmod[i].name; int w=!_stricmp(n,"ntdll.dll")||!_stricmp(n,"kernelbase.dll")||!_stricmp(n,"win32u.dll")||!_stricmp(n,"kernel32.dll"); if(w) fprintf(f,"   (waits) %s: %llu samples\n",n,(unsigned long long)g_pmod[i].n); else fprintf(f,"  %6.2f%%  %s\n",100.0*g_pmod[i].n/act,n); }
  fprintf(f,"  %6.2f%%  game code (functions below)\n",100.0*g_pguest/act); fprintf(f,"\ngame functions (guest address):\n");
  u64 top[6]; int ntop=0;
  for(int r=0;r<60;r++){ u32 best=~0u, bn=0; for(u32 i=0;i<g_npe;i++) if(!(g_pcnt[i]&0x80000000u) && g_pcnt[i]>bn){ bn=g_pcnt[i]; best=i; }
    if(best==~0u || !bn || bn*2000ull<act) break; fprintf(f,"  %6.2f%%  %llx\n",100.0*bn/act,(unsigned long long)g_mod->entry_addr[g_pe[best].idx]);
    if(r<6) top[ntop++]=g_mod->entry_addr[g_pe[best].idx]; g_pcnt[best]|=0x80000000u; }
  for(u32 i=0;i<g_npe;i++) g_pcnt[i]&=0x7FFFFFFFu;
  /* raw A64 words of the top functions, for disassembling later */
  fprintf(f,"\ncode of the hottest game functions (first 96 instructions):\n");
  for(int t=0;t<ntop;t++){ fprintf(f,"  %llx:",(unsigned long long)top[t]); const u32* w=(const u32*)GP(top[t]);
    for(int k=0;k<96;k++){ if(k%16==0) fprintf(f,"\n   "); fprintf(f," %08x",w[k]); } fprintf(f,"\n"); }
  fprintf(f,"\nruntime (offset in the exe):\n");
  for(int r=0;r<40;r++){ int best=-1; u32 bn=0; for(int s=0;s<PEXE;s++) if(g_pexe[s].n>bn && !(g_pexe[s].n&0x80000000u)){ bn=g_pexe[s].n; best=s; }
    if(best<0 || bn*2000ull<act) break; fprintf(f,"  %6.2f%%  +%x\n",100.0*bn/act,g_pexe[best].off); g_pexe[best].n|=0x80000000u; }
  for(int s=0;s<PEXE;s++) g_pexe[s].n&=0x7FFFFFFFu;
  fclose(f); }

static unsigned __stdcall prof_thread(void* a){ (void)a;
  u64 last=GetTickCount64(); DWORD me=GetCurrentThreadId();
  u64 start=last;
  for(;;){ Sleep(g_pboot? 2 : 1);
    LONG n=g_npth; if(n>256) n=256;
    for(LONG i=0;i<n;i++){ HANDLE h=g_pth[i]; if(!h || GetThreadId(h)==me) continue;
      if(SuspendThread(h)==(DWORD)-1) continue; CONTEXT cx; memset(&cx,0,sizeof cx); cx.ContextFlags=CONTEXT_CONTROL;
      if(GetThreadContext(h,&cx)){ ResumeThread(h); prof_note(cx.Rip); } else ResumeThread(h); }
    if(g_pfoc && SuspendThread(g_pfoc)!=(DWORD)-1){ CONTEXT cx; memset(&cx,0,sizeof cx); cx.ContextFlags=CONTEXT_CONTROL;
      if(GetThreadContext(g_pfoc,&cx)){ u64 b=(u64)(uintptr_t)GetModuleHandleA(NULL);
        if(cx.Rip<b || cx.Rip>=b+0x2000000) prof_note_wait(cx.Rsp);   /* outside the exe: read its stack while it's stopped */
        ResumeThread(g_pfoc); prof_note_focus(cx.Rip); } else ResumeThread(g_pfoc); }
    if(GetTickCount64()-last>=(g_pboot? 5000u : 10000u)){ last=GetTickCount64(); prof_write(); }
    if(g_pboot && GetTickCount64()-start>=90000){ prof_write(); break; } }
  return 0; }

void plat_prof_start(int boot){
  if(!g_mod || !g_mod->nentries) return;
  g_pboot=boot; if(boot) g_pfile="startup.txt";
  g_npe=g_mod->nentries; g_pe=malloc(sizeof(PEnt)*g_npe); g_pcnt=calloc(g_npe,4); g_plo=~0ull; g_phi=0;
  for(u32 i=0;i<g_npe;i++){ g_pe[i].host=(u64)(uintptr_t)g_mod->entry_fn[i]; g_pe[i].idx=i; if(g_pe[i].host<g_plo) g_plo=g_pe[i].host; if(g_pe[i].host>g_phi) g_phi=g_pe[i].host; }
  qsort(g_pe,g_npe,sizeof(PEnt),pe_cmp);
  { HANDLE d=NULL; DuplicateHandle(GetCurrentProcess(),GetCurrentThread(),GetCurrentProcess(),&d,0,FALSE,DUPLICATE_SAME_ACCESS); plat_prof_add_thread(d); CloseHandle(d); }
  uintptr_t h=_beginthreadex(NULL,0,prof_thread,NULL,0,NULL); if(h){ SetThreadPriority((HANDLE)h,THREAD_PRIORITY_TIME_CRITICAL); CloseHandle((HANDLE)h); } }

/* stall dump: where a game thread that runs without SVCs really is. a handle to the calling thread, closed on exit */
void* plat_thread_self(void){ HANDLE d=NULL;
  if(!DuplicateHandle(GetCurrentProcess(),GetCurrentThread(),GetCurrentProcess(),&d,THREAD_SUSPEND_RESUME|THREAD_GET_CONTEXT|THREAD_QUERY_INFORMATION,FALSE,0)) return NULL;
  return d; }

void plat_thread_forget(void* h){ if(h) CloseHandle((HANDLE)h); }
/* the caller's stack as exe offsets (debug logs of who did something) */
int plat_backtrace_exe(u32* out, int n){ void* f[24]; int k=RtlCaptureStackBackTrace(1,24,f,NULL), m=0; u64 b=(u64)(uintptr_t)GetModuleHandleA(NULL);
  for(int i=0;i<k && m<n;i++){ u64 v=(u64)(uintptr_t)f[i]; if(v>b && v<b+0x2000000) out[m++]=(u32)(v-b); } return m; }

/* its host pc. nothing between suspend and resume (no allocation, no log), it could be holding any lock */
int plat_thread_pc(void* h, u64* pc){ if(!h || SuspendThread((HANDLE)h)==(DWORD)-1) return 0;
  CONTEXT cx; memset(&cx,0,sizeof cx); cx.ContextFlags=CONTEXT_CONTROL; int ok=GetThreadContext((HANDLE)h,&cx)!=0; ResumeThread((HANDLE)h);
  if(ok) *pc=cx.Rip; return ok; }

/* module name + offset of a host address */
int plat_addr_module(u64 a, char* name, int n, u64* off){ HMODULE hm=NULL;
  if(!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,(LPCSTR)(uintptr_t)a,&hm) || !hm) return 0;
  char full[MAX_PATH]; DWORD k=GetModuleFileNameA(hm,full,sizeof full); if(!k) return 0;
  const char* b=full; for(DWORD i=0;i<k;i++) if(full[i]=='\\' || full[i]=='/') b=full+i+1;
  snprintf(name,n,"%s",hm==GetModuleHandleA(NULL)? "runtime" : b); *off=a-(u64)(uintptr_t)hm; return 1; }
/* text input for the on-screen keyboard (swkbd), drawn like the SwitchRecomp tool's dialogs (dark panel over the
   dimmed game, Hind font, cyan buttons). you type with the PC keyboard, any layout. Enter = OK, Esc = keep the
   default text. returns the length in UTF-16 units, -1 = keep the default */
#include "swkbd_font.h"
static struct { wchar_t txt[1024]; int n, max, all, res, hover, tick; float s; HFONT fb, fh; } KB;
#define KB_W 680
#define KB_H 268
#define KB_BAR 84
static COLORREF kb_pulse(void){ float t=(sinf(KB.tick*0.12f)+1)*0.5f;   /* the tool's selection frame: cyan that breathes */
  return RGB((int)(0+t*90),(int)(200+t*40),(int)(225+t*25)); }
static void kb_rrect(HDC d, int x, int y, int w, int h, int r, COLORREF c){
  HBRUSH b=CreateSolidBrush(c); HPEN p=CreatePen(PS_SOLID,1,c); HGDIOBJ ob=SelectObject(d,b), op=SelectObject(d,p);
  RoundRect(d,x,y,x+w,y+h,r*2,r*2); SelectObject(d,ob); SelectObject(d,op); DeleteObject(b); DeleteObject(p); }
static void kb_frame(HDC d, int x, int y, int w, int h, int r, int th, COLORREF c){
  HPEN p=CreatePen(PS_INSIDEFRAME,th,c); HGDIOBJ op=SelectObject(d,p), ob=SelectObject(d,GetStockObject(NULL_BRUSH));
  RoundRect(d,x,y,x+w,y+h,r*2,r*2); SelectObject(d,op); SelectObject(d,ob); DeleteObject(p); }
static void kb_text(HDC d, HFONT f, const wchar_t* s, int n, int cx, int cy, COLORREF c){
  SelectObject(d,f); SIZE z; GetTextExtentPoint32W(d,s,n,&z); SetTextColor(d,c); TextOutW(d,cx-z.cx/2,cy-z.cy/2,s,n); }
static void kb_paint(HWND h){
  PAINTSTRUCT ps; HDC wd=BeginPaint(h,&ps); RECT rc; GetClientRect(h,&rc); int W=rc.right, H=rc.bottom; float s=KB.s;
  HDC d=CreateCompatibleDC(wd); HBITMAP bm=CreateCompatibleBitmap(wd,W,H); HGDIOBJ obm=SelectObject(d,bm);
  SetBkMode(d,TRANSPARENT);
  kb_rrect(d,0,0,W,H,(int)(6*s),RGB(70,70,70));                                   /* panel */
  kb_text(d,KB.fb,L"Enter text",10,W/2,(int)(54*s),RGB(255,255,255));
  /* text field */
  int fx=(int)(40*s), fy=(int)(84*s), fw=W-2*fx, fh=(int)(56*s), pad=(int)(16*s);
  kb_rrect(d,fx,fy,fw,fh,(int)(4*s),RGB(52,52,52));
  kb_frame(d,fx,fy,fw,fh,(int)(4*s),(int)(2*s)<1?1:(int)(2*s),RGB(0,200,225));
  SelectObject(d,KB.fb); int st=0; SIZE z; GetTextExtentPoint32W(d,KB.txt,KB.n,&z);
  while(st<KB.n && z.cx>fw-2*pad-(int)(8*s)){ st++; GetTextExtentPoint32W(d,KB.txt+st,KB.n-st,&z); }   /* keep the end visible */
  int ty=fy+(fh-z.cy)/2+(int)(2*s);
  if(KB.all && KB.n){ RECT sr={fx+pad-2,fy+(int)(10*s),fx+pad+z.cx+2,fy+fh-(int)(10*s)}; HBRUSH b=CreateSolidBrush(RGB(0,120,140)); FillRect(d,&sr,b); DeleteObject(b); }
  SetTextColor(d,RGB(255,255,255)); TextOutW(d,fx+pad,ty,KB.txt+st,KB.n-st);
  if(!KB.all && (KB.tick/16)%2==0){ RECT cr={fx+pad+z.cx+1,fy+(int)(12*s),fx+pad+z.cx+1+(int)(2*s<2?2:2*s),fy+fh-(int)(12*s)};
    HBRUSH b=CreateSolidBrush(RGB(0,200,225)); FillRect(d,&cr,b); DeleteObject(b); }
  { wchar_t c[32]; int k=swprintf(c,32,L"%d / %d",KB.n,KB.max); kb_text(d,KB.fh,c,k,W/2,fy+fh+(int)(20*s),RGB(160,160,160)); }
  /* button bar, like the tool's dialogs: separator, two halves, cyan labels, pulsing frame on the focused one */
  int by=H-(int)(KB_BAR*s), bw=W/2;
  { RECT l={0,by,W,by+1}; HBRUSH b=CreateSolidBrush(RGB(80,80,80)); FillRect(d,&l,b); RECT v={bw,by,bw+1,H}; FillRect(d,&v,b); DeleteObject(b); }
  const wchar_t* lab[2]={L"Cancel  (Esc)",L"OK  (Enter)"};
  for(int i=0;i<2;i++){ int bx=i*bw; int sel= KB.hover>=0? KB.hover==i : i==1;
    if(sel) kb_frame(d,bx+(int)(6*s),by+(int)(6*s),bw-(int)(12*s),(int)((KB_BAR-12)*s),(int)(4*s),(int)(4*s),kb_pulse());
    kb_text(d,KB.fh,lab[i],(int)wcslen(lab[i]),bx+bw/2,by+(int)(KB_BAR*s/2),RGB(0,200,225)); }
  BitBlt(wd,0,0,W,H,d,0,0,SRCCOPY); SelectObject(d,obm); DeleteObject(bm); DeleteDC(d); EndPaint(h,&ps); }
static LRESULT CALLBACK kb_proc(HWND h, UINT m, WPARAM w, LPARAM l){
  switch(m){
    case WM_PAINT: kb_paint(h); return 0;
    case WM_ERASEBKGND: return 1;
    case WM_TIMER: KB.tick++; InvalidateRect(h,NULL,0); return 0;
    case WM_KEYDOWN:
      if(w==VK_RETURN){ KB.res=1; return 0; } if(w==VK_ESCAPE){ KB.res=-1; return 0; }
      if(w==VK_BACK){ if(KB.all) KB.n=0; else if(KB.n) KB.n--; KB.all=0; InvalidateRect(h,NULL,0); return 0; }
      break;
    case WM_CHAR:
      if(w==0x16 && OpenClipboard(h)){ HANDLE c=GetClipboardData(CF_UNICODETEXT); const wchar_t* p= c? (const wchar_t*)GlobalLock(c) : NULL;   /* Ctrl+V */
        if(p){ if(KB.all) KB.n=0; KB.all=0; for(;*p && KB.n<KB.max;p++) if(*p>=0x20) KB.txt[KB.n++]=*p; GlobalUnlock(c); }
        CloseClipboard(); InvalidateRect(h,NULL,0); return 0; }
      if(w>=0x20 && w!=0x7F){ if(KB.all) KB.n=0; KB.all=0; if(KB.n<KB.max) KB.txt[KB.n++]=(wchar_t)w; InvalidateRect(h,NULL,0); }
      return 0;
    case WM_MOUSEMOVE: { RECT rc; GetClientRect(h,&rc); int y=(short)HIWORD(l), x=(short)LOWORD(l);
      KB.hover= y>=rc.bottom-(int)(KB_BAR*KB.s)? (x<rc.right/2? 0 : 1) : -1; return 0; }
    case WM_LBUTTONUP: if(KB.hover>=0) KB.res= KB.hover? 1 : -1; return 0;
    case WM_CLOSE: KB.res=-1; return 0;
  }
  return DefWindowProcW(h,m,w,l); }
int plat_text_input(const u16* def, u16* out, int max){
  static int font_done; if(!font_done){ DWORD nf=0; AddFontMemResourceEx((void*)g_hind_ttf,sizeof g_hind_ttf,NULL,&nf); font_done=1; }
  HINSTANCE hi=GetModuleHandleW(NULL);
  WNDCLASSW wc={0}; wc.lpfnWndProc=kb_proc; wc.hInstance=hi; wc.lpszClassName=L"SwrKeyboard"; wc.hCursor=LoadCursor(NULL,IDC_ARROW); RegisterClassW(&wc);
  WNDCLASSW wd={0}; wd.lpfnWndProc=DefWindowProcW; wd.hInstance=hi; wd.lpszClassName=L"SwrKeyboardDim"; wd.hbrBackground=(HBRUSH)GetStockObject(BLACK_BRUSH); RegisterClassW(&wd);
  /* over the game window (the foreground one), its height sets the scale like the tool's 1280x720 layout */
  RECT gr; HWND game=GetForegroundWindow(); if(!game || !GetWindowRect(game,&gr)) SetRect(&gr,0,0,GetSystemMetrics(SM_CXSCREEN),GetSystemMetrics(SM_CYSCREEN));
  int gw=gr.right-gr.left, gh=gr.bottom-gr.top; float s=gh/720.0f; if(s<0.75f) s=0.75f; if(s>3) s=3;
  memset(&KB,0,sizeof KB); KB.max= max>0 && max<1000? max : 32; KB.s=s; KB.hover=-1; KB.res=0;
  if(def) while(def[KB.n] && KB.n<KB.max){ KB.txt[KB.n]=def[KB.n]; KB.n++; } KB.all=KB.n>0;
  KB.fb=CreateFontW(-(int)(24*s),0,0,0,FW_NORMAL,0,0,0,DEFAULT_CHARSET,0,0,ANTIALIASED_QUALITY,0,L"Hind");
  KB.fh=CreateFontW(-(int)(22*s),0,0,0,FW_NORMAL,0,0,0,DEFAULT_CHARSET,0,0,ANTIALIASED_QUALITY,0,L"Hind");
  HWND dim=CreateWindowExW(WS_EX_TOPMOST|WS_EX_LAYERED|WS_EX_TOOLWINDOW|WS_EX_NOACTIVATE,L"SwrKeyboardDim",L"",WS_POPUP|WS_VISIBLE,gr.left,gr.top,gw,gh,NULL,NULL,hi,NULL);
  if(dim) SetLayeredWindowAttributes(dim,0,140,LWA_ALPHA);                        /* the tool dims to 55% */
  int w=(int)(KB_W*s), h=(int)(KB_H*s);
  HWND win=CreateWindowExW(WS_EX_TOPMOST|WS_EX_TOOLWINDOW,L"SwrKeyboard",L"Keyboard",WS_POPUP,gr.left+(gw-w)/2,gr.top+(gh-h)/2,w,h,NULL,NULL,hi,NULL);
  if(!win){ if(dim) DestroyWindow(dim); DeleteObject(KB.fb); DeleteObject(KB.fh); return -1; }
  SetWindowRgn(win,CreateRoundRectRgn(0,0,w+1,h+1,(int)(12*s),(int)(12*s)),1);
  ShowWindow(win,SW_SHOW); SetForegroundWindow(win); SetFocus(win); SetTimer(win,1,30,NULL);
  MSG m; while(!KB.res && GetMessageW(&m,NULL,0,0)>0){ TranslateMessage(&m); DispatchMessageW(&m); }
  int n=-1; if(KB.res==1){ n=KB.n; memcpy(out,KB.txt,n*2); }
  KillTimer(win,1); DestroyWindow(win); if(dim) DestroyWindow(dim); DeleteObject(KB.fb); DeleteObject(KB.fh);
  /* wait until Enter/Esc are released, if not the game reads it as a button press when it gets the focus back */
  for(int i=0;i<100 && ((GetAsyncKeyState(VK_RETURN)|GetAsyncKeyState(VK_ESCAPE))&0x8000);i++) Sleep(10);
  if(game) SetForegroundWindow(game);
  return n; }
#else
/* ------------------------------------------------------------------ POSIX */
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>
#include <dlfcn.h>
#include <signal.h>
#include <sched.h>
#include <errno.h>

void pm_init(PMutex* m){ pthread_mutex_init(&m->l,NULL); }

void pm_lock(PMutex* m){ pthread_mutex_lock(&m->l); }

void pm_unlock(PMutex* m){ pthread_mutex_unlock(&m->l); }

void pc_init(PCond* c){ pthread_condattr_t a; pthread_condattr_init(&a); pthread_condattr_setclock(&a,CLOCK_MONOTONIC); pthread_cond_init(&c->c,&a); }

void pc_signal(PCond* c){ pthread_cond_signal(&c->c); }

void pc_broadcast(PCond* c){ pthread_cond_broadcast(&c->c); }

int pc_wait(PCond* c, PMutex* m, i64 ns){
  if(ns<0){ pthread_cond_wait(&c->c,&m->l); return 0; }
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); u64 t=(u64)ts.tv_sec*1000000000ull+ts.tv_nsec+(u64)ns;
  ts.tv_sec=t/1000000000ull; ts.tv_nsec=t%1000000000ull;
  return pthread_cond_timedwait(&c->c,&m->l,&ts)==ETIMEDOUT; }

typedef struct { PThreadFn fn; void* arg; } TArg;

static void* tstart(void* p){ TArg a=*(TArg*)p; free(p); a.fn(a.arg); return NULL; }

int plat_thread(PThreadFn fn, void* arg, size_t stack){
  TArg* a=malloc(sizeof *a); a->fn=fn; a->arg=arg; pthread_attr_t at; pthread_attr_init(&at); pthread_attr_setstacksize(&at,stack);
  pthread_attr_setdetachstate(&at,PTHREAD_CREATE_DETACHED); pthread_t t; int r=pthread_create(&t,&at,tstart,a); pthread_attr_destroy(&at); return r? -1 : 0; }

void plat_thread_exit(void){ pthread_exit(NULL); }

void plat_sleep_ns(i64 ns){ if(ns<=0){ sched_yield(); return; } struct timespec ts={ns/1000000000,ns%1000000000}; while(nanosleep(&ts,&ts)==-1 && errno==EINTR); }

void plat_yield(void){ sched_yield(); }

void plat_thread_low_priority(void){ }

void plat_thread_high_priority(void){ }
void plat_thread_pin(void* h, int core, int prio){ (void)h; (void)core; (void)prio; }

u64 plat_thread_cpu_ns(void){ struct timespec ts; clock_gettime(CLOCK_THREAD_CPUTIME_ID,&ts); return (u64)ts.tv_sec*1000000000ull+ts.tv_nsec; }

u64 plat_process_cpu_ns(void){ struct timespec ts; clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&ts); return (u64)ts.tv_sec*1000000000ull+ts.tv_nsec; }

u64 plat_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return (u64)ts.tv_sec*1000000000ull+ts.tv_nsec; }

i64 plat_unix_time(void){ return (i64)time(NULL); }

i64 plat_tz_offset(void){ time_t t=time(NULL); struct tm lt; localtime_r(&t,&lt); return (i64)lt.tm_gmtoff; }

int g_ww_ok;

int plat_ww_collect(u8* base, u64 size, void (*cb)(u64)){ (void)base; (void)size; (void)cb; return -1; }
int plat_ww_query(u8* base, u64 size, int reset, void (*cb)(u64)){ (void)base; (void)size; (void)reset; (void)cb; return -1; }
void plat_ww_reset(u8* base, u64 size){ (void)base; (void)size; }

int plat_identity_early(void){ return 0; }
u8* plat_vm_reserve(u64 size){ void* p=mmap(NULL,size,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_NORESERVE,-1,0); return p==MAP_FAILED? NULL : (u8*)p; }

void plat_vm_readonly(u8* p, u64 size){ mprotect(p,size,PROT_READ); }
int plat_vm_commit(u8* p, u64 size){ return mprotect(p,size,PROT_READ|PROT_WRITE); }

void plat_vm_decommit(u8* p, u64 size){ madvise(p,size,MADV_DONTNEED); mprotect(p,size,PROT_NONE); }
int plat_vm_lazy_add(u8* p, u64 size){ return plat_vm_commit(p,size); }   /* MAP_NORESERVE: pages appear on first touch anyway */
void plat_lazy_note(void* addr){ (void)addr; }
int plat_readable(const void* p, u64 n){ (void)p; (void)n; return 1; }
void plat_vm_lazy_remove(u8* p, u64 size){ plat_vm_decommit(p,size); }

void* plat_dl_open(const char* path){ return dlopen(path,RTLD_NOW|RTLD_LOCAL); }

void* plat_dl_sym(void* h, const char* n){ return dlsym(h,n); }

int plat_mkdir(const char* p){ return mkdir(p,0755); }

int plat_is_dir(const char* p){ struct stat s; return !stat(p,&s) && S_ISDIR(s.st_mode); }

typedef struct { DIR* d; char path[1024]; } PD_;

int plat_dir_open(PDir* d, const char* path){ PD_* p=malloc(sizeof *p); p->d=opendir(path); if(!p->d){ free(p); return -1; } strncpy(p->path,path,1023); p->path[1023]=0; d->h=p; return 0; }

int plat_dir_next(PDir* d){ PD_* p=d->h; struct dirent* e;
  while((e=readdir(p->d))){ if(!strcmp(e->d_name,".")||!strcmp(e->d_name,"..")) continue; strncpy(d->name,e->d_name,511); d->name[511]=0;
    char f[1600]; snprintf(f,sizeof f,"%s/%s",p->path,e->d_name); struct stat s; if(stat(f,&s)) continue; d->is_dir=S_ISDIR(s.st_mode); d->size=(u64)s.st_size; return 1; }
  return 0; }

void plat_dir_close(PDir* d){ PD_* p=d->h; closedir(p->d); free(p); }

u64 plat_file_size(const char* p){ struct stat s; if(stat(p,&s)) return 0; return (u64)s.st_size; }

int plat_pread(FILE* f, void* buf, u64 size, u64 off){ return (int)pread(fileno(f),buf,size,(off_t)off); }

int plat_ncpu(void){ long n=sysconf(_SC_NPROCESSORS_ONLN); return n>0? (int)n : 1; }

void plat_battery(int* pct, int* chg){ *pct=100; *chg=1;
  FILE* f=fopen("/sys/class/power_supply/BAT0/capacity","r"); if(!f) return;
  if(fscanf(f,"%d",pct)!=1 || *pct<0 || *pct>100) *pct=100; fclose(f);
  f=fopen("/sys/class/power_supply/BAT0/status","r"); if(!f) return;
  char s[32]={0}; if(fgets(s,sizeof s,f)) *chg= strncmp(s,"Discharging",11)!=0; fclose(f); }

__thread void* plat_guard_rec;

void plat_guard_enter(void* rec){ plat_guard_rec=rec; }

void* plat_guard_get(void){ return plat_guard_rec; }

volatile int g_driver_recovered;   /* times we jumped out of a driver crash */

static void sighandler(int sig, siginfo_t* si, void* uc_){
  plat_jmpbuf* b=(plat_jmpbuf*)plat_guard_get();
  if(b){
    { extern void gpu_crash_note(void); gpu_crash_note(); }
    plat_guard_enter(NULL); g_driver_recovered++;
    logf_(0,"  (signal %d inside the video driver, recovered: only that effect is missing)\n",sig);
    siglongjmp(*b,1); }
  extern void crash_report(void* host_pc, void* fault_addr, int sig);
  void* pc=NULL;
#if defined(__x86_64__)
  #include <ucontext.h>
  ucontext_t* uc=(ucontext_t*)uc_; pc=(void*)uc->uc_mcontext.gregs[REG_RIP];
#elif defined(__aarch64__)
  #include <ucontext.h>
  ucontext_t* uc=(ucontext_t*)uc_; pc=(void*)uc->uc_mcontext.pc;
#else
  (void)uc_;
#endif
  crash_report(pc,si->si_addr,sig); _exit(1); }

/* errors and the F1 help as a desktop dialog: zenity (GNOME and most desktops) or kdialog (KDE, steam deck).
   without either the text only goes to the log. wait=0 leaves the dialog open while the game runs */
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;

static void reap(void* p){ int st; waitpid((pid_t)(size_t)p,&st,0); }

void plat_message(const char* text, int error, int wait){
  if(!getenv("DISPLAY") && !getenv("WAYLAND_DISPLAY")) return;
  char* z[]={"zenity",error? "--error" : "--info","--no-markup","--no-wrap","--title=SwitchRecomp","--text",(char*)text,NULL};
  char* k[]={"kdialog","--title","SwitchRecomp",error? "--error" : "--msgbox",(char*)text,NULL};
  pid_t pid;
  if(posix_spawnp(&pid,z[0],NULL,NULL,z,environ) && posix_spawnp(&pid,k[0],NULL,NULL,k,environ)) return;
  if(wait) reap((void*)(size_t)pid); else plat_thread(reap,(void*)(size_t)pid,1<<16); }

void plat_install_crash_handler(void){
  struct sigaction sa; memset(&sa,0,sizeof sa); sa.sa_sigaction=sighandler; sa.sa_flags=SA_SIGINFO|SA_ONSTACK;
  static char altstack[65536]; stack_t ss; ss.ss_sp=altstack; ss.ss_size=sizeof altstack; ss.ss_flags=0; sigaltstack(&ss,NULL);
  sigaction(SIGSEGV,&sa,NULL); sigaction(SIGBUS,&sa,NULL); sigaction(SIGILL,&sa,NULL); sigaction(SIGFPE,&sa,NULL); }
#endif
#ifndef _WIN32
void plat_prof_start(int boot){ (void)boot; }

/* thread sampling on Linux (the stall report shows where a busy thread runs): the thread gets SIGPROF and its
   handler copies the interrupted pc out of the signal context */
#include <pthread.h>
#include <ucontext.h>
#include <dlfcn.h>
#include <semaphore.h>
typedef struct { pthread_t t; } HostThread;
static volatile u64 g_samp_pc; static sem_t g_samp_sem; static int g_samp_init; static pthread_mutex_t g_samp_lk=PTHREAD_MUTEX_INITIALIZER;
static void samp_handler(int s, siginfo_t* si, void* uc){ (void)s; (void)si;
#if defined(__x86_64__)
  g_samp_pc=(u64)((ucontext_t*)uc)->uc_mcontext.gregs[REG_RIP];
#elif defined(__aarch64__)
  g_samp_pc=(u64)((ucontext_t*)uc)->uc_mcontext.pc;
#else
  (void)uc; g_samp_pc=0;
#endif
  sem_post(&g_samp_sem); }
void* plat_thread_self(void){ HostThread* h=malloc(sizeof *h); if(h) h->t=pthread_self(); return h; }

void plat_thread_forget(void* h){ free(h); }

int plat_thread_pc(void* hv, u64* pc){ HostThread* h=hv; if(!h) return 0;
  pthread_mutex_lock(&g_samp_lk);
  if(!g_samp_init){ sem_init(&g_samp_sem,0,0); struct sigaction sa; memset(&sa,0,sizeof sa); sa.sa_sigaction=samp_handler; sa.sa_flags=SA_SIGINFO|SA_RESTART; sigaction(SIGPROF,&sa,NULL); g_samp_init=1; }
  g_samp_pc=0; int ok=0;
  if(!pthread_kill(h->t,SIGPROF)){ struct timespec ts; clock_gettime(CLOCK_REALTIME,&ts); ts.tv_nsec+=50000000; if(ts.tv_nsec>=1000000000){ ts.tv_sec++; ts.tv_nsec-=1000000000; }
    if(!sem_timedwait(&g_samp_sem,&ts)){ *pc=g_samp_pc; ok= g_samp_pc!=0; } }
  pthread_mutex_unlock(&g_samp_lk); return ok; }

int plat_addr_module(u64 a, char* name, int n, u64* off){ Dl_info di; if(!dladdr((void*)(uintptr_t)a,&di) || !di.dli_fname) return 0;
  const char* b=strrchr(di.dli_fname,'/'); snprintf(name,n,"%s%s%s",b? b+1 : di.dli_fname,di.dli_sname? ":" : "",di.dli_sname? di.dli_sname : "");
  *off= di.dli_saddr? a-(u64)(uintptr_t)di.dli_saddr : a-(u64)(uintptr_t)di.dli_fbase; return 1; }
int plat_text_input(const u16* def, u16* out, int max){ (void)def; (void)out; (void)max; return -1; }
#endif
