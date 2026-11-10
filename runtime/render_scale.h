/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
#ifndef SWR_RENDER_SCALE_H
#define SWR_RENDER_SCALE_H
/* one row/column targets hold data (lighting, exposure), not images. scaling them changes how many entries there
   are and what they cover, even if the size rounds back to native */
static inline float swr_rt_scale(unsigned w, unsigned h, float scene_scale){
  return (w==1 || h==1)? 1.0f : scene_scale;
}
static inline unsigned swr_scaled_dim(unsigned v, float scale){
  unsigned n=(unsigned)(v*scale+0.5f); return n? n : 1;
}
#endif
