/* SwitchRecomp by MZ Labs, Copyright 2026 MZ Labs. AGPL-3.0, see LICENSE.txt and NOTICE.txt. */
#ifndef SWR_REC_FLOW_H
#define SWR_REC_FLOW_H
#include <stdint.h>
enum { REC_FLOW_WAIT,REC_FLOW_AUDIO,REC_FLOW_VIDEO };
/* MF buffers the stream that's ahead while it waits for the other one, so never feed it seconds of one stream just
   because the other producer is idle for a moment */
static inline int rec_flow_pick(int audio,int64_t at,int video,int64_t vt,
                               int64_t audio_end,int64_t video_end,int draining){
  int ar=audio && (draining || at<=video_end+1000000);
  int vr=video && (draining || vt<=audio_end+1000000);
  if(ar && (!vr || at<=vt)) return REC_FLOW_AUDIO;
  if(vr) return REC_FLOW_VIDEO;
  return REC_FLOW_WAIT;
}
static inline int rec_flow_stalled(uint64_t now,uint64_t entered,uint64_t progress){
  uint64_t since=progress>entered?progress:entered;
  return now>=since && now-since>=5000;
}
#endif
