/* SwitchRecomp by MZ Labs, Copyright 2026 MZ Labs. AGPL-3.0, see LICENSE.txt and NOTICE.txt. */
#ifndef SWR_REC_AUDIO_TIME_H
#define SWR_REC_AUDIO_TIME_H
#include <stdint.h>
#include <math.h>
static inline uint64_t rec_audio_index(uint64_t elapsed_ns,unsigned frames,uint64_t next){
  uint64_t end=(elapsed_ns/1000000000ull)*48000+(elapsed_ns%1000000000ull)*48000/1000000000ull;
  uint64_t start=end>frames?end-frames:0;
  /* 5ms event jitter snaps to the sample clock, real gaps stay */
  return start>next+480?start:next;
}
static inline int64_t rec_audio_time(uint64_t frame){ return (int64_t)(frame/48000*10000000+(frame%48000)*10000000/48000); }
static inline int16_t rec_audio_pcm(float x){ if(!isfinite(x)) x=0; if(x>1) x=1; if(x< -1) x=-1; return (int16_t)lrintf(x*32767); }
#include "audio_limit.h"
static inline float rec_audio_gain(const float* pcm,unsigned frames,float previous){ return swr_audio_gain(pcm,frames,previous); }
#endif
