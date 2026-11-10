
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <setjmp.h>
#include "rec_audio_time.h"
#include "rec_flow.h"
typedef uint64_t u64;typedef int64_t i64;typedef int32_t HRESULT;typedef unsigned DWORD;
#define FAILED(x) ((x)<0)
#define SUCCEEDED(x) ((x)>=0)
#define RQ 64
#define AQ 4096
typedef struct {i64 t,d;} IMFSample;
typedef struct {u64 frame;unsigned count;int16_t pcm[480];} AudioPacket;
static IMFSample samples[RQ],*g_q[RQ];static unsigned g_qh,g_qt,g_ah,g_at;static AudioPacket packets[AQ];
static int g_run=1,g_ev,g_space_ev,g_alock,failed,fail_after,writes,progress,idle;
static u64 audio_end;static i64 video_last;static unsigned audio_written,video_written;
static jmp_buf finish;
#define AcquireSRWLockExclusive(x) ((void)0)
#define ReleaseSRWLockExclusive(x) ((void)0)
static void SetEvent(int e){}
static void WaitForSingleObject(int e,unsigned ms){if(++idle==20)longjmp(finish,1);}
static int rec_failed(void){return failed;}
static void rec_progress(void){progress++;}
static void rec_fail(const char*r,HRESULT hr){assert(hr<0);failed=1;g_run=0;}
static int q_count(void){return g_qh-g_qt;}
static unsigned InterlockedIncrement(unsigned*p){return ++*p;}
static int audio_peek(u64*f){if(g_ah==g_at)return 0;*f=packets[g_at%AQ].frame;return 1;}
static int audio_pop(AudioPacket*p){if(g_ah==g_at)return 0;*p=packets[g_at++%AQ];return 1;}
static void IMFSample_GetSampleTime(IMFSample*p,i64*t){*t=p->t;}
static void IMFSample_GetSampleDuration(IMFSample*p,i64*t){*t=p->d;}
static void IMFSample_Release(IMFSample*p){}
static HRESULT write_result(void){writes++;return fail_after&&writes==fail_after?(HRESULT)0x8007000e:0;}
static HRESULT audio_write(void*w,DWORD st,const int16_t*pcm,unsigned n,u64 first){HRESULT r=write_result();if(r<0)return r;assert(first>=audio_end);audio_end=first+n;audio_written+=n;return 0;}
static HRESULT audio_silence(void*w,DWORD st,u64*next,u64 end){if(end>*next){HRESULT r=audio_write(w,st,NULL,(unsigned)(end-*next),*next);if(r<0)return r;*next=end;}return 0;}
static HRESULT IMFSinkWriter_WriteSample(void*w,DWORD st,IMFSample*p){HRESULT r=write_result();if(r<0)return r;assert(p->t>=video_last);video_last=p->t+p->d;video_written++;return 0;}
static void run_loop(void){void*wr=NULL;DWORD si=0,ai=1;HRESULT hr=0;u64 audio_next=0;i64 video_end=0;
  for(;;){
    if(rec_failed())break;
    u64 af=0; int have_audio=audio_peek(&af),have_video=q_count()>0;
    if(have_audio||have_video){
      i64 vt=0;if(have_video)IMFSample_GetSampleTime(g_q[g_qt%RQ],&vt);
      /* A real gap is known only when the next PCM packet has a later
         timestamp. Fill it in bounded pieces as video advances. */
      if(have_audio && af>audio_next){
        u64 limit=(u64)video_end*48000/10000000+4800;
        u64 end=af<limit?af:limit;
        if(end>audio_next){hr=audio_silence(wr,ai,&audio_next,end);if(FAILED(hr)){rec_fail("audio gap encoding failed",hr);break;}rec_progress();continue;}
      }
      if(!g_run&&!have_video && rec_audio_time(audio_next)>=video_end)break;
      int choice=rec_flow_pick(have_audio,rec_audio_time(af),have_video,vt,rec_audio_time(audio_next),video_end,!g_run);
      if(choice==REC_FLOW_AUDIO){
        AudioPacket p;if(audio_pop(&p)){
          hr=audio_silence(wr,ai,&audio_next,p.frame);
          unsigned skip=audio_next>p.frame?(unsigned)(audio_next-p.frame):0;
          if(!g_run&&!have_video){u64 last=(u64)video_end*48000/10000000;if(p.frame>=last)break;if(p.count>last-p.frame)p.count=(unsigned)(last-p.frame);}
          if(SUCCEEDED(hr)&&skip<p.count){hr=audio_write(wr,ai,p.pcm+skip*2,p.count-skip,p.frame+skip);audio_next=p.frame+p.count;}
          if(FAILED(hr)){rec_fail("audio encoding failed",hr);break;}rec_progress();
        }
      }else if(choice==REC_FLOW_VIDEO){
        if(!g_run && !have_audio && rec_audio_time(audio_next)<vt){hr=audio_silence(wr,ai,&audio_next,(u64)vt*48000/10000000);if(FAILED(hr)){rec_fail("closing audio gap failed",hr);break;}}
        IMFSample* sample=g_q[g_qt%RQ];InterlockedIncrement(&g_qt);SetEvent(g_space_ev);
        i64 t=0,d=0;IMFSample_GetSampleTime(sample,&t);IMFSample_GetSampleDuration(sample,&d);if(t+d>video_end)video_end=t+d;
        hr=IMFSinkWriter_WriteSample(wr,si,sample);IMFSample_Release(sample);
        if(FAILED(hr)){rec_fail("video encoding failed",hr);break;}rec_progress();
      }else {WaitForSingleObject(g_ev,10);}
      continue;
    }
    if(!g_run)break;
    WaitForSingleObject(g_ev,100);
  }
  if(!rec_failed()&&video_end>0)audio_silence(wr,ai,&audio_next,(u64)video_end*48000/10000000);
  /* On an explicit recording failure, release queued samples instead of
     feeding the failing encoder repeatedly. Normal close drains them above. */
  while(q_count()>0){IMFSample* sample=g_q[g_qt%RQ];InterlockedIncrement(&g_qt);IMFSample_Release(sample);}
  AcquireSRWLockExclusive(&g_alock);g_at=g_ah;ReleaseSRWLockExclusive(&g_alock);SetEvent(g_space_ev);

}
static void setup(unsigned video,unsigned audio){failed=fail_after=writes=progress=idle=0;g_run=1;g_qh=video;g_qt=0;g_ah=audio;g_at=0;audio_end=0;video_last=0;audio_written=video_written=0;
for(unsigned i=0;i<video;i++){samples[i].t=(u64)i*10000000/60;samples[i].d=(u64)(i+1)*10000000/60-samples[i].t;g_q[i]=samples+i;}
for(unsigned i=0;i<audio;i++){packets[i].frame=(u64)i*240;packets[i].count=240;}}
int main(void){
setup(60,1000);if(!setjmp(finish))run_loop();assert(video_written==60&&audio_end<=48000+4800+240);
puts("PASS real encoder scheduler bounds audio lead to ~100ms when video stops; no unbounded MF audio buffering");
setup(60,20);if(!setjmp(finish))run_loop();assert(video_last<=rec_audio_time(audio_end)+1000000+166667);
puts("PASS real encoder scheduler also bounds video lead when PCM production pauses");
setup(60,200);fail_after=5;if(!setjmp(finish))run_loop();assert(failed&&writes==5&&g_qh==g_qt&&g_ah==g_at);
puts("PASS injected 8007000e stops recording immediately and releases pending queues instead of retrying failed writes");
setup(60,1000);g_run=0;run_loop();assert(!failed&&video_written==60&&audio_end==48000);
puts("PASS normal close drains a balanced one-second movie and trims audio beyond the last captured frame");
}
