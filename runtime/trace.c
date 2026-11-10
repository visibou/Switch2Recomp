/* Timeline trace, see trace.h. Each thread appends to its own buffer (no locks while recording); the buffers are
   registered once per thread. Timestamps are TSC ticks, converted to microseconds when the file is written with a
   rate measured against plat_ns over the window. */
#include "rt.h"
#include "trace.h"
#include <x86intrin.h>

volatile int g_swr_trace_on = 0;

typedef struct { unsigned long long t0, t1, arg; const char* name; int kind; } TEv;   /* kind 0 span, 1 counter, 2 gpu span */
typedef struct { TEv* ev; u32 n, cap; u32 tid; const char* name; } TBuf;

#define TRACE_MAX_THREADS 256
#define TRACE_CAP (1u << 20)     /* events per thread in one window */
static TBuf* g_bufs[TRACE_MAX_THREADS];
static volatile int g_nbufs, g_reg_lock;
static __thread TBuf* t_buf;
static __thread const char* t_name;
static double g_start_s = -1, g_len_s = 3;
static u64 g_on_ns, g_on_tsc, g_off_ns, g_off_tsc, g_proc_ns;
static int g_state;              /* 0 waiting, 1 on, 2 written */
static long long g_gpu_off = -(1ll << 62);   /* trace ns minus GPU ns, raised by each sync (swr_trace_gpu_sync) */
static int g_gpu_synced;

unsigned long long swr_trace_ts(void){ return __rdtsc(); }
__attribute__((constructor)) static void trace_init(void){ g_proc_ns=plat_ns(); }   /* SWR_TRACE counts from here */
unsigned long long swr_run_ns(void){ return plat_ns()-g_proc_ns; }

static u32 cur_tid(void){
#ifdef _WIN32
  return (u32)GetCurrentThreadId();
#else
  return (u32)(uintptr_t)&t_buf;
#endif
}

static TBuf* buf(void){
  TBuf* b=t_buf; if(b) return b;
  b=(TBuf*)calloc(1,sizeof *b); b->cap=TRACE_CAP; b->ev=(TEv*)malloc(sizeof(TEv)*b->cap); b->tid=cur_tid(); b->name=t_name;
  if(!b->ev){ free(b); return NULL; }
  while(__sync_lock_test_and_set(&g_reg_lock,1)) ;
  if(g_nbufs<TRACE_MAX_THREADS) g_bufs[g_nbufs++]=b; else { free(b->ev); free(b); b=NULL; }
  __sync_lock_release(&g_reg_lock);
  t_buf=b; return b; }

static void add(const char* name, u64 t0, u64 t1, u64 arg, int kind){
  if(!g_swr_trace_on) return;
  TBuf* b=buf(); if(!b || b->n>=b->cap) return;
  TEv* e=&b->ev[b->n]; e->t0=t0; e->t1=t1; e->arg=arg; e->name=name; e->kind=kind;
  __sync_synchronize(); b->n++; }

void swr_trace_span(const char* name, unsigned long long t0){ if(g_swr_trace_on) add(name,t0,__rdtsc(),0,0); }
void swr_trace_span_arg(const char* name, unsigned long long t0, unsigned long long arg){ if(g_swr_trace_on) add(name,t0,__rdtsc(),arg,0); }
void swr_trace_counter(const char* name, long long value){ if(g_swr_trace_on){ u64 t=__rdtsc(); add(name,t,t,(u64)value,1); } }
void swr_trace_thread_name(const char* name){ t_name=name; if(t_buf) t_buf->name=name; }
void swr_trace_gpu_span(const char* name, unsigned long long g0, unsigned long long g1, unsigned long long arg, int track){
  if(g_swr_trace_on) add(name,g0,g1,arg | ((u64)(track&0xff)<<56),2); }

/* a submission cannot start on the GPU before the CPU submitted it: the largest (cpu - gpu) seen is the offset, to
   within the GPU's start latency when it was idle */
void swr_trace_gpu_sync(unsigned long long cpu_submit, unsigned long long gpu_start_ns){
  if(g_state!=1) return;
  double rate=(double)(plat_ns()-g_on_ns)/(double)(__rdtsc()-g_on_tsc+1);   /* ns per tick so far */
  long long cpu_ns=(long long)((double)(long long)(cpu_submit-g_on_tsc)*rate);
  long long off=cpu_ns-(long long)gpu_start_ns;
  if(off>g_gpu_off) g_gpu_off=off;
  g_gpu_synced=1; }

static void json_name(FILE* f, const char* s){
  fputc('"',f); for(; s && *s; s++){ if(*s=='"' || *s=='\\') fputc('\\',f); fputc(*s,f); } fputc('"',f); }

static void write_file(void){
  char path[1400]; snprintf(path,sizeof path,"%strace.json",g_exedir);
  FILE* f=fopen(path,"wb"); if(!f){ WARN("trace: cannot write %s\n",path); return; }
  double us_per_tick=(double)(g_off_ns-g_on_ns)/1000.0/(double)(g_off_tsc-g_on_tsc);
  u64 total=0;
  fprintf(f,"{\"displayTimeUnit\":\"ms\",\"traceEvents\":[\n");
  fprintf(f,"{\"ph\":\"M\",\"pid\":1,\"name\":\"process_name\",\"args\":{\"name\":\"%s\"}}",g_exename);
  static const char* gpu_tracks[4]={"host GPU: submissions","host GPU: passes","host GPU: other","host GPU"};
  for(int k=0;k<4;k++) fprintf(f,",\n{\"ph\":\"M\",\"pid\":1,\"tid\":%d,\"name\":\"thread_name\",\"args\":{\"name\":\"%s\"}}",900000+k,gpu_tracks[k]);
  int n=g_nbufs;
  for(int i=0;i<n;i++){ TBuf* b=g_bufs[i]; if(!b->n) continue;
    fprintf(f,",\n{\"ph\":\"M\",\"pid\":1,\"tid\":%u,\"name\":\"thread_name\",\"args\":{\"name\":",b->tid);
    char nm[64]; snprintf(nm,sizeof nm,"thread %u",b->tid); json_name(f,b->name? b->name : nm); fprintf(f,"}}");
    for(u32 j=0;j<b->n;j++){ TEv* e=&b->ev[j]; total++;
      if(e->kind==2){
        if(!g_gpu_synced) continue;
        int track=(int)(e->arg>>56); u64 arg=e->arg&((1ull<<56)-1);
        double ts=((double)((long long)e->t0+g_gpu_off))/1000.0, dur=(double)(e->t1-e->t0)/1000.0;
        fprintf(f,",\n{\"ph\":\"X\",\"pid\":1,\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f,\"name\":",900000+track,ts,dur);
        if(arg){ char s[96]; snprintf(s,sizeof s,"%s %llux%llu",e->name,(unsigned long long)(arg>>20),(unsigned long long)(arg&0xFFFFF)); json_name(f,s); }
        else json_name(f,e->name);
        fprintf(f,"}");
        continue; }
      double ts=(double)(long long)(e->t0-g_on_tsc)*us_per_tick;
      if(e->kind==1){ fprintf(f,",\n{\"ph\":\"C\",\"pid\":1,\"ts\":%.3f,\"name\":",ts); json_name(f,e->name); fprintf(f,",\"args\":{\"value\":%lld}}",(long long)e->arg); continue; }
      double dur=(double)(e->t1-e->t0)*us_per_tick;
      fprintf(f,",\n{\"ph\":\"X\",\"pid\":1,\"tid\":%u,\"ts\":%.3f,\"dur\":%.3f,\"name\":",b->tid,ts,dur);
      if(e->arg){ char s[96]; snprintf(s,sizeof s,"%s %llu",e->name,(unsigned long long)e->arg); json_name(f,s); }
      else json_name(f,e->name);
      fprintf(f,"}"); } }
  fprintf(f,"\n]}\n"); fclose(f);
  LOG("trace: %llu events from %d threads in %s (%.1f s from %.1f s)\n",(unsigned long long)total,n,path,g_len_s,g_start_s); }

void swr_trace_poll(void){
  if(g_state==2) return;
  if(g_start_s<0){
    const char* e=getenv("SWR_TRACE"); if(!e){ g_state=2; return; }
    g_start_s=atof(e); const char* c=strchr(e,','); if(c) g_len_s=atof(c+1); if(g_len_s<=0) g_len_s=3; }
  u64 now=plat_ns(); double t=(double)(now-g_proc_ns)/1e9;
  if(g_state==0 && t>=g_start_s){ g_on_ns=plat_ns(); g_on_tsc=__rdtsc(); g_state=1; g_swr_trace_on=1; LOG("trace: on for %.1f s\n",g_len_s); return; }
  if(g_state==1 && t>=g_start_s+g_len_s){
    g_swr_trace_on=0; g_off_ns=plat_ns(); g_off_tsc=__rdtsc(); g_state=2;
    plat_sleep_ns(20000000);   /* spans being appended right now */
    write_file(); } }
