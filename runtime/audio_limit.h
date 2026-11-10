/* SwitchRecomp by MZ Labs, Copyright 2026 MZ Labs. AGPL-3.0, see LICENSE.txt and NOTICE.txt. */
#ifndef SWR_AUDIO_LIMIT_H
#define SWR_AUDIO_LIMIT_H
#include <math.h>
/* stereo linked peak limiter, fast attack and 100ms release. the headroom goes in before converting so loud mixer
   peaks don't get flattened */
static inline float swr_audio_gain(const float* pcm,unsigned frames,float previous){
  float peak=0;
  for(unsigned i=0;i<frames*2;i++) if(isfinite(pcm[i]) && fabsf(pcm[i])>peak) peak=fabsf(pcm[i]);
  float ceiling=peak>.95f?.95f/peak:1.f;
  float released=1.f-(1.f-previous)*expf(-(float)frames/4800.f);
  return ceiling<released?ceiling:released;
}
#endif
