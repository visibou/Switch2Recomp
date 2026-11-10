/* SwitchRecomp by MZ Labs, Copyright 2026 MZ Labs. AGPL-3.0, see LICENSE.txt and NOTICE.txt. */
#ifndef SWR_AUDIO_QUEUE_H
#define SWR_AUDIO_QUEUE_H
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* one continuous source timeline, interpolation across IPC packets included. frames that don't fit yet stay with
   the producer */
typedef struct {
  float* pcm;
  unsigned capacity,head,count,waiting;
  uint64_t accepted,consumed;
  double phase,rate;
  float previous[2],envelope;
  int primed;
} SwrAudioQueue;

static int swr_aq_init(SwrAudioQueue* q,unsigned capacity){
  memset(q,0,sizeof *q); q->pcm=calloc((size_t)capacity*2,sizeof(float));
  q->capacity=q->pcm?capacity:0; q->rate=1; return q->pcm!=NULL;
}
static void swr_aq_free(SwrAudioQueue* q){ free(q->pcm); memset(q,0,sizeof *q); }
static unsigned swr_aq_push(SwrAudioQueue* q,const float* pcm,unsigned frames){
  unsigned take=q->capacity-q->count; if(take>frames) take=frames;
  for(unsigned i=0;i<take;i++){
    unsigned at=(q->head+q->count+i)%q->capacity;
    for(unsigned ch=0;ch<2;ch++){ float x=pcm[i*2+ch]; q->pcm[at*2+ch]=isfinite(x)?x:0; }
  }
  q->count+=take; q->accepted+=take; return take;
}
static void swr_aq_pop(SwrAudioQueue* q){
  q->previous[0]=q->pcm[q->head*2]; q->previous[1]=q->pcm[q->head*2+1];
  q->head=(q->head+1)%q->capacity; q->count--; q->consumed++;
}
static void swr_aq_render(SwrAudioQueue* q,float* out,unsigned frames,int smooth){
  /* 100ms is the adaptive reserve. a producer giving half real time ends up around half speed, playback never
     speeds up to throw away backlog */
  double target=smooth?(double)q->count/4800.0:1.0;
  if(target>1) target=1; if(target<.02) target=.02;
  if(!smooth) q->rate=1;
  for(unsigned i=0;i<frames;i++){
    if(!q->primed){
      q->waiting++;
      if(q->count && (q->count>=2400 || q->waiting>=2400)){
        swr_aq_pop(q); q->primed=1; q->envelope=1; q->rate=target;
      } else { out[i*2]=out[i*2+1]=0; continue; }
    }
    if(!q->count){
      /* a real stall can't give future samples, so fade the held end to silence without moving the source cursor
         and pick that cursor back up after */
      out[i*2]=q->previous[0]*q->envelope;
      out[i*2+1]=q->previous[1]*q->envelope;
      q->envelope-=1.f/240.f; if(q->envelope<0) q->envelope=0; continue;
    }
    if(q->envelope<1){
      q->envelope+=1.f/240.f; if(q->envelope>1) q->envelope=1;
      out[i*2]=q->previous[0]*q->envelope; out[i*2+1]=q->previous[1]*q->envelope;
      continue;
    }
    for(unsigned ch=0;ch<2;ch++) out[i*2+ch]=q->previous[ch]+(q->pcm[q->head*2+ch]-q->previous[ch])*(float)q->phase;
    q->rate+=(target-q->rate)*.0005;
    q->phase+=q->rate;
    if(q->phase>=1){ q->phase-=1; swr_aq_pop(q); }
  }
}
#endif
