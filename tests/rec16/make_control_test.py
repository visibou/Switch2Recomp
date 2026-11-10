from pathlib import Path
s=Path('work/v190/SwitchRecomp/runtime/rec_win32.c').read_text();a=s[s.index('#include "rec_audio_time.h"'):s.index('static HRESULT audio_write(')];close=s[s.index('void rec_close(void)'):]
pre=r'''
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
'''
post=r'''
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
'''
Path('work/rec16/test_control.c').write_text(pre+a+close+post)
