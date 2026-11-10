#include "audio.h"
#include <assert.h>
#include <math.h>
#include <pthread.h>
u8* g_mem;int g_verbose=0;
void logf_(int l,const char* f,...){(void)l;va_list a;va_start(a,f);vfprintf(stderr,f,a);va_end(a);}
void pm_init(PMutex* m){pthread_mutex_init(&m->l,NULL);}void pm_lock(PMutex* m){pthread_mutex_lock(&m->l);}void pm_unlock(PMutex* m){pthread_mutex_unlock(&m->l);}
typedef struct{PThreadFn fn;void* arg;}TArg;static void* tstart(void* p){TArg t=*(TArg*)p;free(p);t.fn(t.arg);return NULL;}
int plat_thread(PThreadFn fn,void* arg,size_t stack){(void)stack;pthread_t th;TArg* t=malloc(sizeof *t);t->fn=fn;t->arg=arg;if(pthread_create(&th,NULL,tstart,t)){free(t);return -1;}pthread_detach(th);return 0;}
volatile int g_quit;u64 plat_ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return (u64)ts.tv_sec*1000000000ull+(u64)ts.tv_nsec;}
void plat_sleep_ns(i64 ns){struct timespec ts={ns/1000000000,ns%1000000000};nanosleep(&ts,NULL);}
void mem_query(u64 a,MemBlock* b){memset(b,0,sizeof *b);if(a<65536){b->size=65536;b->perm=PERM_R;}}
static void w32(u8* p,u32 x){memcpy(p,&x,4);}static void w64(u8* p,u64 x){memcpy(p,&x,8);}static void wf(u8* p,float x){memcpy(p,&x,4);}
static u64 played(u8* s){u64 n;memcpy(&n,s,8);return n;}static u32 consumed(u8* s){u32 n;memcpy(&n,s+8,4);return n;}
static u8 v[0x170],st[16];static float out[480];static SvcObj o;
static void begin(unsigned format,unsigned ch,unsigned count){swr_audio_destroy(&o);memset(&o,0,sizeof o);memset(v,0,sizeof v);memset(st,0,sizeof st);
 v[9]=1;v[0xb]=(u8)format;w32(v+0xc,48000);w32(v+0x18,ch);wf(v+0x1c,1);wf(v+0x20,1);w64(v+0x60,0x1000);w64(v+0x68,count*ch*(format==6?1:format==5?4:2));w32(v+0x74,count);}
static void run(void){swr_audio_mix(&o,v,1,sizeof v,NULL,0,NULL,0,NULL,0,st,out);}
static void near(float x,float y){assert(isfinite(x));assert(fabsf(x-y)<.0001f);}
int main(void){g_mem=calloc(1,65536);assert(g_mem);
 begin(2,1,240);for(int i=0;i<240;i++)((i16*)(g_mem+0x1000))[i]=8192;run();near(out[0],.25f*.70710678f);near(out[1],out[0]);assert(played(st)==240&&consumed(st)==1);
 run();assert(consumed(st)==1&&played(st)==240);near(out[0],0);puts("PASS PCM16 mono, gain, completion and duplicate-descriptor protection");
 begin(2,2,240);for(int i=0;i<240;i++){((i16*)(g_mem+0x1000))[i*2]=8192;((i16*)(g_mem+0x1000))[i*2+1]=-16384;}run();near(out[0],.25);near(out[1],-.5);puts("PASS interleaved stereo channels");
 begin(2,1,120);for(int i=0;i<120;i++){((i16*)(g_mem+0x1000))[i]=8192;((i16*)(g_mem+0x2000))[i]=16384;}
 memcpy(v+0x98,v+0x60,0x38);w64(v+0x98,0x2000);run();assert(consumed(st)==2&&played(st)==240);near(out[238],.25*.70710678);near(out[240],.5*.70710678);puts("PASS streaming across two buffers in a single update");
 begin(2,1,160);w32(v+0xc,32000);run();assert(played(st)==160&&consumed(st)==1);puts("PASS 32kHz resampling and exact end-of-buffer accounting");
 begin(2,1,120);v[0x78]=1;run();assert(played(st)==240&&consumed(st)==0);v[0xa]=2;run();assert(played(st)==240);v[0xa]=1;run();assert(consumed(st)==1&&played(st)==0);puts("PASS looping, pause and stop/flush");
 begin(6,1,14);w64(v+0x68,8);memset(g_mem+0x1000,0,8);memset(g_mem+0x3000,0,32);((i16*)(g_mem+0x3000))[0]=2048;((i16*)(g_mem+0x4000))[1]=1000;
 w64(v+0x48,0x3000);w64(v+0x50,32);w64(v+0x80,0x4000);w64(v+0x88,6);v[0x78]=1;run();near(out[0],1000/32768.f*.70710678f);near(out[478],out[0]);assert(played(st)==240);puts("PASS DSP ADPCM predictor, histories and loop context");
 begin(6,1,14);w64(v+0x68,8);g_mem[0x1000]=15;memset(g_mem+0x1001,0x78,7);run();near(out[0],32767/32768.f*.70710678f);near(out[2],-.70710678f);puts("PASS DSP ADPCM positive/negative saturation");
 begin(5,1,240);((float*)(g_mem+0x1000))[0]=NAN;((float*)(g_mem+0x1000))[1]=INFINITY;run();near(out[0],0);near(out[2],0);puts("PASS non-finite float samples become silence");
 begin(2,1,240);w64(v+0x60,65530);run();assert(played(st)==0&&consumed(st)==0);puts("PASS unmapped/truncated sample memory rejected");
 begin(2,1,240);for(int i=0;i<240;i++)((i16*)(g_mem+0x1000))[i]=16384;wf(v+0x20,.5);w32(v+0x58,1);
 u8 resources[0x70]={0},mix[0x1260]={0};resources[0x64]=1;wf(resources+4,1);wf(resources+8,.5);
 for(int i=0;i<2;i++){u8* p=mix+i*0x930;wf(p,.5);w32(p+8,2);p[0xc]=1;w32(p+0x10,i);w32(p+0x924,i?0:0x7fffffff);}
 wf(mix+0x930+0x24,1);wf(mix+0x930+0x24+(24+1)*4,1);
 swr_audio_mix(&o,v,1,sizeof v,resources,sizeof resources,mix,sizeof mix,NULL,0,st,out);near(out[0],.0625);near(out[1],.03125);puts("PASS per-channel routing, submix matrix and final-mix volumes");
 /* A mono voice routed only into the 5.1 center must reach stereo output. */
 begin(2,1,240);for(int i=0;i<240;i++)((i16*)(g_mem+0x1000))[i]=8192;
 u8 res6[0x70]={0},mix6[0x930]={0},sink6[0x140]={0};res6[0x64]=1;wf(res6+4+2*4,1);
 wf(mix6,1);w32(mix6+8,6);mix6[0xc]=1;w32(mix6+0x924,0x7fffffff);
 sink6[0]=1;sink6[1]=1;w32(sink6+0x120,6);for(int i=0;i<6;i++)sink6[0x124+i]=i;
 swr_audio_mix(&o,v,1,sizeof v,res6,sizeof res6,mix6,sizeof mix6,sink6,sizeof sink6,st,out);
 near(out[0],.25f*.70710678f);near(out[1],out[0]);puts("PASS center-only dialogue survives 5.1 stereo downmix");
 begin(2,1,240);wf(res6+4+2*4,0);wf(res6+4+4*4,1);
 swr_audio_mix(&o,v,1,sizeof v,res6,sizeof res6,mix6,sizeof mix6,sink6,sizeof sink6,st,out);
 near(out[0],.25f*.70710678f);near(out[1],0);puts("PASS rear-left channel downmix preserves stereo side");
 /* Legacy REV1 padding must never discard a queued logo jingle. */
 begin(2,1,480);for(int i=0;i<480;i++)((i16*)(g_mem+0x1000))[i]=8192;
 v[0x159]=0x7f;run();assert(played(st)==240&&consumed(st)==0);near(out[0],.25f*.70710678f);
 run();assert(played(st)==480&&consumed(st)==1);puts("PASS REV1 reserved padding cannot drop a short sound");
 /* audout uses direct PCM16 streams, in stereo (Celeste) or 5.1 (Brawlhalla). */
 for(int i=0;i<240;i++){((i16*)(g_mem+0x1000))[i*2]=8192;((i16*)(g_mem+0x1000))[i*2+1]=-16384;}
 assert(swr_audio_read_pcm16(0x1000,240,2,.5f,out));near(out[0],.125f);near(out[1],-.25f);
 memset(g_mem+0x1000,0,240*6*2);for(int i=0;i<240;i++)((i16*)(g_mem+0x1000))[i*6+2]=8192;
 assert(swr_audio_read_pcm16(0x1000,240,6,1,out));near(out[0],.25f*.70710678f);near(out[1],out[0]);
 assert(!swr_audio_read_pcm16(65530,240,6,1,out));assert(!swr_audio_read_pcm16(0x1000,240,7,1,out));
 puts("PASS audout PCM16 stereo, surround center, volume and memory bounds");
 swr_audio_init();swr_audio_submit_at(out,240,0);puts("PASS host output submission (SDL dummy device)");
 swr_audio_destroy(&o);free(g_mem);return 0;}
