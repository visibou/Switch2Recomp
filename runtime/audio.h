/* SwitchRecomp by MZ Labs, Copyright 2026 MZ Labs. AGPL-3.0, see LICENSE.txt and NOTICE.txt. */
#ifndef SWR_AUDIO_H
#define SWR_AUDIO_H
#include "rt.h"
#define SWR_AUDIO_FRAMES 240
extern int g_opt_audio_slowdown;
void swr_audio_init(void);
void swr_audio_submit_at(const float* stereo, unsigned frames, u64 sample_end);
int swr_audio_read_pcm16(u64 address,unsigned frames,unsigned channels,float volume,float* stereo);
void swr_audio_begin_update(SvcObj* o,unsigned blocks);
void swr_audio_configure_effects(SvcObj* o,const u8* effects,u32 count,u32 stride);
void swr_audio_destroy(SvcObj* o);
int swr_audio_mix(SvcObj* o,const u8* voices,u32 count,u32 stride,
  const u8* resources,u32 resources_size,const u8* mixes,u32 mixes_size,
  const u8* sinks,u32 sinks_size,u8* status,float* stereo);
#endif
