/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* Maxwell (GM20B) shader -> GLSL 4.30 translator */
#ifndef SWR_SHADER_H
#define SWR_SHADER_H
#include "cpu.h"
enum { ST_VS=0, ST_TCS=1, ST_TES=2, ST_GS=3, ST_FS=4, ST_CS=5, ST_TASK=6, ST_MESH=7 };   /* 6/7: Switch 2 task / mesh programs (GL_NV_mesh_shader) */
#define SH_MAXTEX 32
typedef struct {
  u32 cbuf_off;        /* handle offset in the texture cbuf, in words */
  u8 type;             /* 0 1D 1 2D 2 3D 3 CUBE */
  u8 array, shadow, fetch_only, is_int;
  u8 bindless; u8 bl_cbuf; u8 bl2; u8 bl2_cbuf;   /* bl2: handle = read1 | read2 (separate tex/sampler handles) */
  u32 bl2_off;          /* word offset of read2 */
} ShTex;
typedef struct {
  int stage;
  u32 gs_input_vertices;         /* geometry shader input primitive size */
  u32 cbuf_ssbo_mask;            /* cbufs moved to storage blocks when the stage runs out of uniform blocks */
  u32 sph[20];
  u32 cbuf_mask;                 /* cbufs used */
  u32 cbuf_indirect_mask;
  u32 cbuf_max[18];              /* bytes used per cbuf, so we upload less */
  ShTex tex[SH_MAXTEX]; int ntex;
  u32 in_mask, out_mask;         /* generic attribs 0..31 */
  u32 attr_int_mask;             /* key: integer VS attribs */
  u32 attr_sint_mask;
  u32 fs_flat_mask;              /* key */
  int writes_depth, uses_position_in, uses_clip_mask, writes_layer, writes_point_size, uses_front_face;
  int uses_blit_cb;              /* reads NVN2's blit parameter bank (c[24]), synthesized from the render area */
  u32 frag_rt_mask;
  int uses_global, uses_local, local_size;
  u32 gbuf_cbuf_off[16]; int ngbuf;  /* global buffers found (their cbuf0 offset) */
  int unsupported;               /* count of instructions we couldn't translate */
  char first_unsupported[64];
  u32 cs_local[3]; int shared_size;
  u32 gbuf_wmask;                /* gbufs written (compute) */
  struct { u32 cbuf_off; u8 type, bindless, bl_cbuf, pad; } img[8]; int nimg;   /* SUST targets (compute) */
  u32 sutex_mask;                /* SULD.D reads, these need their format class in swr_tc */
  u32 tex_int_mask, tex_sint_mask, rt_int_mask, rt_sint_mask;  /* FS key: integer textures / outputs */
  u8 tex_bank_set, tex_bank;     /* Ampere: the bank named in its TEX instructions, used instead of tex_cb_index */
  /* task / mesh programs (SET_MESH_SHADER_A/B, SET_MESH_INIT_SHADER, the mesh and pass-through headers), filled in by
     the caller: workgroup size, payload words, output limits, and the stage-buffer lines (an attribute per line:
     0-7 = primitive id, layer, viewport, point size, position x y z w; 8+ = generic 0x..31w) */
  u32 mesh_threads, mesh_payload_words, mesh_max_vertices, mesh_max_primitives;
  u8 mesh_vlines[136], mesh_plines[136]; u32 mesh_nvlines, mesh_nplines; int mesh_has_task;
} ShaderInfo;
/* malloc'd GLSL, or NULL */
char* shader_to_glsl(const u8* code, u32 max_bytes, int stage, ShaderInfo* si);
extern int g_sh_ampere;
extern int (*g_shader_cb_read)(u32 idx, u32 off, u32* out);
u64 shader_hash(const u8* code, u32 max_bytes, u32* out_len);
#endif
