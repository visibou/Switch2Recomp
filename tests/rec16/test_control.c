
#include <pthread.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <wchar.h>
#include <stdarg.h>
#include "rec_flow.h"
typedef uint64_t u64;typedef uint32_t u32;typedef int16_t i16;typedef int32_t HRESULT,LONG;typedef int64_t LONG64;
typedef pthread_mutex_t SRWLOCK;
#define SRWLOCK_INIT PTHREAD_MUTEX_INITIALIZER
#define AcquireSRWLockExclusive pthread_mutex_lock
#define ReleaseSRWLockExclusive pthread_mutex_unlock
#define AcquireSRWLockShared pthread_mutex_lock
#define ReleaseSRWLockShared pthread_mutex_unlock
#define E_FAIL ((HRESULT)0x80004005)
#define WAIT_OBJECT_0 0
static int g_ok=1;static volatile int g_run=1;static int g_ev=2,g_space_ev=3,g_ready_ev=4,g_th=1;
static volatile LONG g_failed;static volatile LONG64 g_progress_ms;
static unsigned waits,closed;static int video_pending,thread_ready,fake_clock;static u64 fake_ms;
static FILE*g_raw;static u64 g_frames_in,g_frames_drop;static wchar_t g_path[8];
static int q_count(void){return video_pending;}
static LONG InterlockedCompareExchange(volatile LONG*p,LONG v,LONG old){return __sync_val_compare_and_swap(p,old,v);}
static LONG64 InterlockedCompareExchange64(volatile LONG64*p,LONG64 v,LONG64 old){return __sync_val_compare_and_swap(p,old,v);}
static LONG64 InterlockedExchange64(volatile LONG64*p,LONG64 v){return __atomic_exchange_n(p,v,__ATOMIC_SEQ_CST);}
static u64 GetTickCount64(void){if(fake_clock)return fake_ms;struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (u64)t.tv_sec*1000+t.tv_nsec/1000000;}
static void logger(const char*f,...){(void)f;}
#define WARN(...) logger(__VA_ARGS__)
#define LOG(...) logger(__VA_ARGS__)
static void SetEvent(int e){}
static void CloseHandle(int e){closed++;}
static int WaitForSingleObject(int e,int ms){if(e==g_th)return thread_ready?0:258;__atomic_add_fetch(&waits,1,__ATOMIC_RELAXED);nanosleep(&(struct timespec){0,1000000},NULL);return 0;}
#include "rec_audio_time.h"
#define AQ 4096
#define AF 240
typedef struct {u64 frame;unsigned count;i16 pcm[AF*2];} AudioPacket;
static AudioPacket g_aq[AQ];static u32 g_ah,g_at;
static SRWLOCK g_alock=SRWLOCK_INIT;
static int g_aactive;static u64 g_aepoch,g_anext,g_audio_drop,g_asession;static float g_again;
int rec_failed(void){return InterlockedCompareExchange(&g_failed,0,0)!=0;}
static void rec_progress(void){InterlockedExchange64(&g_progress_ms,(LONG64)GetTickCount64());}
static void rec_fail(const char* reason,HRESULT hr){
  if(InterlockedCompareExchange(&g_failed,1,0))return;
  WARN("recorder: stopping capture: %s (%08lx); releasing game producers\n",reason,(unsigned long)hr);
  g_ok=0;g_run=0;
  AcquireSRWLockExclusive(&g_alock);g_aactive=0;ReleaseSRWLockExclusive(&g_alock);
  SetEvent(g_ev);SetEvent(g_space_ev);
}
static int rec_wait_stalled(u64 entered){return rec_flow_stalled(GetTickCount64(),entered,(u64)InterlockedCompareExchange64(&g_progress_ms,0,0));}
void rec_audio_start(u64 guest_epoch){
  AcquireSRWLockExclusive(&g_alock);g_asession++;g_aepoch=guest_epoch;g_anext=0;g_again=1;g_aactive=g_ok&&g_run;ReleaseSRWLockExclusive(&g_alock);
}
void rec_audio(const float* stereo,unsigned frames,u64 guest_time){
  if(!frames||frames>AF)return;
  AcquireSRWLockExclusive(&g_alock);
  if(!g_aactive||guest_time<g_aepoch){ReleaseSRWLockExclusive(&g_alock);return;}
  u64 session=g_asession;
  /* Timestamp the rendered quantum, not its delayed IPC delivery time.
     Catch-up quanta are contiguous and share the video's guest-clock epoch. */
  u64 frame=rec_audio_index(guest_time-g_aepoch,frames,g_anext);
  g_anext=frame+frames;
  u64 entered=GetTickCount64();
  /* Keep the packet while the encoder is behind. Discarding a quantum here
     made permanent holes even though live playback retained its own copy. */
  while(g_ah-g_at>=AQ && g_aactive && g_run && session==g_asession){
    ReleaseSRWLockExclusive(&g_alock);
    if(rec_wait_stalled(entered)){rec_fail("encoder stalled with a full audio queue",E_FAIL);return;}
    WaitForSingleObject(g_space_ev,10);AcquireSRWLockExclusive(&g_alock);
  }
  if(!g_aactive || !g_run || session!=g_asession){ReleaseSRWLockExclusive(&g_alock);return;}
  AudioPacket* p=&g_aq[g_ah%AQ];p->frame=frame;p->count=frames;
  float target=rec_audio_gain(stereo,frames,g_again);g_again=target;
  for(unsigned i=0;i<frames*2;i++)p->pcm[i]=rec_audio_pcm(stereo[i]*target);
  g_ah++;SetEvent(g_ev);ReleaseSRWLockExclusive(&g_alock);
}
static int audio_peek(u64* frame){int ok;AcquireSRWLockShared(&g_alock);ok=g_ah!=g_at;if(ok)*frame=g_aq[g_at%AQ].frame;ReleaseSRWLockShared(&g_alock);return ok;}
static int audio_pop(AudioPacket* p){int ok;AcquireSRWLockExclusive(&g_alock);ok=g_ah!=g_at;if(ok){*p=g_aq[g_at%AQ];g_at++;}ReleaseSRWLockExclusive(&g_alock);if(ok)SetEvent(g_space_ev);return ok;}
void rec_watchdog(void){
  if(!g_th||!g_ok||!g_run)return;
  u64 frame;int pending=q_count()>0||audio_peek(&frame);
  if(pending&&rec_wait_stalled(0))rec_fail("encoder made no progress for 5 seconds",E_FAIL);
}
void rec_close(void){
  if(g_raw){ fclose(g_raw); g_raw=NULL; LOG("recorder: %llu raw frames\n",(unsigned long long)g_frames_in); return; }
  if(!g_th) return;
  AcquireSRWLockExclusive(&g_alock);g_aactive=0;ReleaseSRWLockExclusive(&g_alock);
  g_run=0; SetEvent(g_ev);SetEvent(g_space_ev);
  if(WaitForSingleObject(g_th,rec_failed()?1000:5000)!=WAIT_OBJECT_0){
    rec_fail("encoder did not finish; capture disabled until it responds",E_FAIL);
    return; /* thread still owns its handles and MF objects; do not free them */
  }
  CloseHandle(g_th); g_th=NULL; CloseHandle(g_ev); CloseHandle(g_ready_ev); CloseHandle(g_space_ev);
  LOG("recorder: %llu audio packets lost (queue limit)\n",(unsigned long long)g_audio_drop);
  LOG("recorder: saved %ls (%llu frames, %llu dropped)\n",g_path,(unsigned long long)g_frames_in,(unsigned long long)g_frames_drop); }

static void* producer(void*p){float pcm[480]={0};for(unsigned k=0;k<AQ+32;k++)rec_audio(pcm,240,(u64)(k+1)*5000000);return NULL;}
int main(void){rec_progress();rec_audio_start(0);pthread_t p;assert(!pthread_create(&p,NULL,producer,NULL));
nanosleep(&(struct timespec){0,40000000},NULL);
unsigned got=0;while(got<AQ+32){AudioPacket a;if(audio_pop(&a)){assert(a.frame==(u64)got*240);got++;rec_progress();}else nanosleep(&(struct timespec){0,1000000},NULL);}
assert(!pthread_join(p,NULL));assert(waits&&!rec_failed()&&g_audio_drop==0);
puts("PASS healthy encoder backpressure retains all packets and contiguous timestamps");
/* The independent runtime poll detects a hung MF call before queues fill. */
fake_clock=1;fake_ms=6000;InterlockedExchange64(&g_progress_ms,1000);video_pending=1;rec_watchdog();assert(rec_failed()&&!g_ok&&!g_run&&!g_aactive);
puts("PASS independent 5-second watchdog disables stalled capture and releases audio producers");
/* A blocked worker still owns its handles. Closing them here would allow
   a new recording to corrupt the old worker when it eventually returns. */
closed=0;thread_ready=0;rec_close();assert(g_th==1&&closed==0);
puts("PASS stalled encoder close returns without freeing handles still owned by its worker");
thread_ready=1;rec_close();assert(g_th==0&&closed==4);
puts("PASS completed encoder close releases all four handles normally");
}
