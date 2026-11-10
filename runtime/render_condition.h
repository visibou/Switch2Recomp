/* SwitchRecomp by MZ Labs, Copyright 2026 MZ Labs. AGPL-3.0, see LICENSE.txt and NOTICE.txt. maxwell B197 render enable register modes (NVIDIA open-gpu-doc) */
#ifndef SWR_RENDER_CONDITION_H
#define SWR_RENDER_CONDITION_H
static int swr_render_test(const u32* r,unsigned base){
  unsigned mode=r[base+2]&7;
  if(mode<=1) return (int)mode;
  /* reports are 16 bytes, 64 bit value then timestamp. NON_ZERO looks at the value at address, EQUAL/NOT_EQUAL
     compare it with the value 16 bytes later (nvn's two counter reports) */
  u64 address=((u64)(r[base]&255)<<32)|r[base+1]; u64 rep[3]={0};
  if(!address || !gmem_read(address,rep,sizeof rep)) return 1; /* no GPU report to look at, so don't cull anything */
  if(mode==2) return rep[0]!=0;
  /* we don't count samples, every ZPASS report says 1000. two of those match even when the box was on screen and the
     occlusion test culled the object (tomodachi's flickering and missing objects), so those count as visible */
  int eq= rep[0]==rep[2] && rep[0]!=1000;
  if(mode==3) return eq;
  if(mode==4) return !eq;
  return 1;
}
static int swr_render_enabled(const u32* r){
  unsigned override=r[0x651]&3;
  if(override==1) return 1;
  if(override==2) return 0;
  return swr_render_test(r,0x4c) && swr_render_test(r,0x554);
}
#endif
