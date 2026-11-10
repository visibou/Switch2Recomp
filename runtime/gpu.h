/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* GPU state shared between the engines and the backends */
#ifndef SWR_GPU_H
#define SWR_GPU_H
#include "rt.h"
typedef struct GpuChannel {
  u32 id;
  u32 sub_class[8];
  u32 puller[0x40];
  u32 r3d[0x1000];          /* 3D engine regs, 0xE00+ are MME macro calls */
  u32 rcomp[0x1000];
  u32 r2d[0x400];
  u32 rdma[0x200];
  u32 ri2m[0x80];
  /* inline to memory */
  u8* i2m_buf; u32 i2m_buf_cap, i2m_len, i2m_pos, i2m_w, i2m_lines, i2m_pitch, i2m_bh; u64 i2m_addr; int i2m_linear; u32 i2m_bd, i2m_dw, i2m_dh, i2m_x, i2m_y, i2m_z;
  /* macros (MME) */
  u32 pb_mode, pb_sub, pb_m, pb_cnt, pb_k;   /* a method batch can continue into the next GPFIFO entry */
  u32 mme_code[0x3000]; u32 mme_start[0x100]; u32 mme_upload_ptr, mme_start_ptr;
  /* Switch 2: the channel bound an Ampere 3D class (C797/C597). its macros are TU104 MME programs, with a data RAM
     and DMA reads of their parameters (gpu_engines.c mme_tu104) */
  int ampere; u32 mme_dram[0xC00];
  u32 mme_params[0x10000]; u32 mme_nparams; int mme_active; u32 mme_index;
  u32 mme_shadow[0x1000];
  /* Switch 2: the Ampere compute class (C7C0) calls macros too. The macro RAM is the channel's (a channel with a 3D
     class uploads through it and its compute subchannel calls them); the calls have their own parameter queue */
  u32 cmme_params[0x1000]; u32 cmme_nparams, cmme_index; int cmme_active;
  u32 qmd_rel_tmpl[8]; int qmd_has_tmpl;
  u64 qt_addr; u32 qt_need, qt_w15, qt_w16;   /* the fence ring's full QMD: blank ones take its counter and hand-off */
  /* waits between queues (gpu.c gpu_chan_submit): its submits not run yet, and the wait it is parked on */
  void* psq; void* psq_tail; int blk, stop, blk_wide, blk_cs; u32 blk_op; u64 blk_addr, blk_want, blk_t0;   /* the release words of the channel's fence QMD, see qmd3_releases */
} GpuChannel;
typedef struct { GpuChannel* chan[64]; int nchan; } GpuState;
extern GpuState g_gpu;
int gmem_read(u64 gva, void* dst, u64 size);
int gmem_write(u64 gva, const void* src, u64 size);
u8* gmem_ptr(u64 gva, u64 size);
u32 gmem_r32(u64 gva); void gmem_w32(u64 gva, u32 v); void gmem_w64(u64 gva, u64 v);
u64 bl_offset(u32 x_bytes, u32 y, u32 z, u32 width_bytes, u32 height, u32 bh_log2, u32 bd_log2);
/* Switch 2 GOBs: inside the 512 bytes, bits 5 and 6 of the offset trade places when they differ (Eden's notes on the
   native tiling). Without it every compressed texture came out with hatched edges and wrong block colours */
extern int g_gob_native;
static inline int gob_native_on(void){ extern u64 g_title_id; if(g_gob_native<0) g_gob_native= getenv("SWR_GOB_CLASSIC")? 0 : (g_title_id>>56)==0x04; return g_gob_native; }
static inline u32 gob_native(u32 o){ u32 d=((o>>5)^(o>>6))&1; return o^(d<<5)^(d<<6); }
void deswizzle(u8* dst, u32 dst_pitch, const u8* src, u32 w_bytes, u32 h, u32 bh_log2);
void swizzle(u8* dst, const u8* src, u32 src_pitch, u32 w_bytes, u32 h, u32 bh_log2);
void i2m_launch(GpuChannel* ch, u32* regs, int base);
void i2m_data(GpuChannel* ch, u32 v);
void maxwell3d_method(GpuChannel* ch, u32 m, u32 v, int last);
void compute_method(GpuChannel* ch, u32 m, u32 v, int last);
void kepler_i2m_method(GpuChannel* ch, u32 m, u32 v);
void fermi2d_method(GpuChannel* ch, u32 m, u32 v);
void dma_method(GpuChannel* ch, u32 m, u32 v);
void gpu_flush(GpuChannel* ch);
void gpu_mem_written(u64 gva, u64 size);
void gpu3d_init(void);
#endif
