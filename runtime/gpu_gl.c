/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* GL 4.3 renderer for the maxwell command stream. it runs on its own thread like the real gpu, the game submits
   pushbuffers, we run them and signal the syncpoints. textures, buffers and shaders get translated once and cached */
#include "gpu.h"
#include "render_condition.h"
#include <time.h>
#include <math.h>
#include "glfn.h"
#include "shader.h"
#include "sm86.h"

int g_gl=0;                     /* active renderer */
static int g_gl_ready;
#include "glq.h"

/* GL calls in this file go through glq when it's on */
#define backend_gl_proc glq_proc
#define backend_gl_swap glq_swap
#define backend_gl_worker_current glq_worker_current
extern int backend_gl_create(void); extern void* backend_gl_proc(const char*); extern void backend_gl_swap(void); extern void backend_window_size(int*,int*);
u64 g_draws;
static __thread int t_gpu_thread;   /* GL context owner */
static const char* volatile g_gpu_phase; static volatile u64 g_gpu_phase_t0;   /* what the GPU thread is on, for the stall report */
extern int gpu_chan_submit(u64* e, u32 n, u32 channel, u32 incs, u32 wait_id, u32 wait_val); extern int gpu_chan_pump_all(void); static int g_parked_any;
static void sub_done(u32 id, u32 n);
static void cs_complete(int wait); static void cs_kick_async(void); static int cs_inflight(void); static int cs_pending_any(void); static void cs_submit_done(u32 id, u32 n); static void cs_stale_check(void);
static u64 g_gpu_busy; static u64 st_tex_ok, st_tex_null, st_tex_mismatch, st_tex_rt;

u64 gpu_busy_ns(void){ return __atomic_load_n(&g_gpu_busy,__ATOMIC_RELAXED); }

extern u32 g_submit_epoch;
static u64 st_compiling, st_cb_skip, st_cb_bytes, st_vb_hashed, st_vb_up, st_tex_up; static u64 st_noshader, st_shfail, st_noprog, st_nort, st_drawn, st_clears, st_pres_rt, st_pres_cpu, st_shaders;
extern int g_opt_res;
#include "render_scale.h"
static float g_draw_scale=1.0f;
#define SH_CODE_MAX 0x20000
static void rs_update(void);
static void ra_update(u32 w, u32 h);
static float g_scale=1.0f;       /* internal res scale, 0.25..4 */
static u32 SC(u32 v){ return swr_scaled_dim(v,g_scale); }

static u32 g_last_cw, g_last_ch;   /* last color target size, used to spot shadow passes */
static float fabsf_(float x){ return x<0? -x : x; }

u32 fmt_bpp_2d(u32 f);

/* ============================================================ work queue */
enum { WK_SUBMIT, WK_PRESENT_CPU, WK_PRESENT_ADDR, WK_SHOT, WK_SYNC, WK_DUMP };

typedef struct { int kind; u64* ent; u32 n, syncpt, incs, wait_id, wait_val; u32* px; int w,h; u64 addr; u32 fmt; PCond* done_cv; volatile int* done; char path[256]; u64 result; u64 sub_ns; } Work;

#define QN 1024
static Work q[QN]; static int qh, qt; static PMutex g_ql; static PCond g_qcv, g_qspace;

static void q_push(Work* w){
  pm_lock(&g_ql); while(((qh+1)%QN)==qt) pc_wait(&g_qspace,&g_ql,1000000); q[qh]=*w; qh=(qh+1)%QN; pc_signal(&g_qcv); pm_unlock(&g_ql); }

static int q_pop(Work* w, i64 timeout){
  pm_lock(&g_ql); if(qh==qt) pc_wait(&g_qcv,&g_ql,timeout); if(qh==qt){ pm_unlock(&g_ql); return 0; }
  *w=q[qt]; qt=(qt+1)%QN; pc_signal(&g_qspace); pm_unlock(&g_ql); return 1; }

/* ============================================================ GL helpers */
static GLuint compile(GLenum type, const char* src, char* err, int errn){
  GLuint s=glCreateShader(type); glShaderSource(s,1,&src,NULL); glCompileShader(s); GLint ok=0; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
  if(!ok){ if(err) glGetShaderInfoLog(s,errn,NULL,err); glDeleteShader(s); return 0; } return s; }

static GLuint link2(GLuint a, GLuint b, GLuint c3){
  GLuint p=glCreateProgram(); if(a) glAttachShader(p,a); if(b) glAttachShader(p,b); if(c3) glAttachShader(p,c3); glLinkProgram(p);
  GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok); if(!ok){ char log[2048]; glGetProgramInfoLog(p,2048,NULL,log); WARN("GL link: %s\n",log); return 0; } return p; }

/* ============================================================ formats */
typedef struct { GLenum ifmt, fmt, type; u8 bpp, bw, bh, compressed, depth, is_int, astc, dec, tobc; } GlFmt;

/* GLES: anything the hw can't sample gets decoded on the CPU */
static int g_has_astc;
/* ASTC is decoded on the CPU and stored as BC3 (1 byte a texel) instead of RGBA8 (4): MKW's ASTC 8x8
   textures came out 16 times their size and it ran out of RAM. Desktop GPUs have no ASTC (Mesa takes it but decodes
   it to RGBA8 inside the driver), Intel Gen9-11 does and keeps it native */
static int g_astc_bc;
extern int swr_bc_decode(int kind, const u8* src, u32 w, u32 h, u32* out);

static GlFmt F(GLenum i, GLenum f, GLenum t, int bpp){ GlFmt r={i,f,t,(u8)bpp,1,1,0,0,0,0,0}; r.is_int=(f==GL_RGBA_INTEGER||f==GL_RG_INTEGER||f==GL_RED_INTEGER); return r; }

static GlFmt FC(GLenum i, int bpp){ GlFmt r={i,0,0,(u8)bpp,4,4,1,0,0,0,0}; return r; }

static GlFmt FASTC(int bw, int bh){ GlFmt r={GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,16,(u8)bw,(u8)bh,1,0,0,1,0}; return r; }

static GlFmt FD(GLenum i, GLenum f, GLenum t, int bpp){ GlFmt r={i,f,t,(u8)bpp,1,1,0,1,0,0,0}; return r; }

/* TIC: format, type of the first component, sRGB */
static GlFmt tic_format(u32 w0, int srgb){
  u32 f=w0&0x7F, ct=(w0>>7)&7;
  int flt= ct==7, sn= ct==1, ui= ct==4, si= ct==3;
  GlFmt r;
  switch(f){
    case 0x01: r= flt? F(GL_RGBA32F,GL_RGBA,GL_FLOAT,16) : ui? F(GL_RGBA32UI,GL_RGBA_INTEGER,GL_UNSIGNED_INT,16) : F(GL_RGBA32I,GL_RGBA_INTEGER,GL_INT,16); break;
    case 0x02: r= F(GL_RGB32F,GL_RGB,GL_FLOAT,12); break;
    case 0x0E: r= FD(GL_DEPTH24_STENCIL8,GL_DEPTH_STENCIL,GL_UNSIGNED_INT_24_8,4); break;   /* G24R8 */
    case 0x73: r= F(GL_RGBA16_SNORM,GL_RGBA,GL_SHORT,8); break;   /* RGBX16SN, the X reads as 1 (swizzle below) */
    case 0x03: r= flt? F(GL_RGBA16F,GL_RGBA,GL_HALF_FLOAT,8) : sn? F(GL_RGBA16_SNORM,GL_RGBA,GL_SHORT,8) : ui? F(GL_RGBA16UI,GL_RGBA_INTEGER,GL_UNSIGNED_SHORT,8) : si? F(GL_RGBA16I,GL_RGBA_INTEGER,GL_SHORT,8) : F(GL_RGBA16,GL_RGBA,GL_UNSIGNED_SHORT,8); break;
    case 0x04: r= flt? F(GL_RG32F,GL_RG,GL_FLOAT,8) : ui? F(GL_RG32UI,GL_RG_INTEGER,GL_UNSIGNED_INT,8) : F(GL_RG32I,GL_RG_INTEGER,GL_INT,8); break;
    /* ETC2/EAC, core since GL 4.3: uploaded as 4x4 blocks */
    case 0x06: r=FC(srgb? 0x9275 : 0x9274,8); break; /* ETC2 RGB8 */
    case 0x07: case 0x08: r= sn? F(GL_RGBA8_SNORM,GL_RGBA,GL_BYTE,4) : ui? F(GL_RGBA8UI,GL_RGBA_INTEGER,GL_UNSIGNED_BYTE,4) : si? F(GL_RGBA8I,GL_RGBA_INTEGER,GL_BYTE,4) : F(srgb? GL_SRGB8_ALPHA8 : GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,4); break;
    case 0x09: r= ui? F(GL_RGB10_A2UI,GL_RGBA_INTEGER,GL_UNSIGNED_INT_2_10_10_10_REV,4) : F(GL_RGB10_A2,GL_RGBA,GL_UNSIGNED_INT_2_10_10_10_REV,4); break;
    case 0x0A: r=FC(srgb? 0x9277 : 0x9276,8); break; /* ETC2 RGB8 punch-through alpha */
    case 0x0B: r=FC(srgb? 0x9279 : 0x9278,16); break; /* ETC2 RGBA8 / EAC alpha */
    case 0x0C: r= flt? F(GL_RG16F,GL_RG,GL_HALF_FLOAT,4) : sn? F(GL_RG16_SNORM,GL_RG,GL_SHORT,4) : ui? F(GL_RG16UI,GL_RG_INTEGER,GL_UNSIGNED_SHORT,4) : si? F(GL_RG16I,GL_RG_INTEGER,GL_SHORT,4) : F(GL_RG16,GL_RG,GL_UNSIGNED_SHORT,4); break;
    case 0x0F: r= flt? F(GL_R32F,GL_RED,GL_FLOAT,4) : ui? F(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT,4) : F(GL_R32I,GL_RED_INTEGER,GL_INT,4); break;
    case 0x12: r= F(GL_RGBA4,GL_RGBA,GL_UNSIGNED_SHORT_4_4_4_4_REV,2); break;
    case 0x13: case 0x14: r= F(GL_RGB5_A1,GL_RGBA,GL_UNSIGNED_SHORT_1_5_5_5_REV,2); break;
    case 0x15: r= F(GL_RGB565,GL_RGB,GL_UNSIGNED_SHORT_5_6_5_REV,2); break;
    case 0x16: r= F(GL_RGB565,GL_RGB,GL_UNSIGNED_SHORT_5_6_5,2); break;
    case 0x18: r= sn? F(GL_RG8_SNORM,GL_RG,GL_BYTE,2) : ui? F(GL_RG8UI,GL_RG_INTEGER,GL_UNSIGNED_BYTE,2) : si? F(GL_RG8I,GL_RG_INTEGER,GL_BYTE,2) : F(GL_RG8,GL_RG,GL_UNSIGNED_BYTE,2); break;
    case 0x19: r=FC(sn? 0x9271 : 0x9270,8); break; /* EAC R11 */
    case 0x1A: r=FC(sn? 0x9273 : 0x9272,16); break; /* EAC RG11 */
    case 0x1B: r= flt? F(GL_R16F,GL_RED,GL_HALF_FLOAT,2) : sn? F(GL_R16_SNORM,GL_RED,GL_SHORT,2) : ui? F(GL_R16UI,GL_RED_INTEGER,GL_UNSIGNED_SHORT,2) : si? F(GL_R16I,GL_RED_INTEGER,GL_SHORT,2) : F(GL_R16,GL_RED,GL_UNSIGNED_SHORT,2); break;
    case 0x1D: case 0x1C: r= sn? F(GL_R8_SNORM,GL_RED,GL_BYTE,1) : ui? F(GL_R8UI,GL_RED_INTEGER,GL_UNSIGNED_BYTE,1) : si? F(GL_R8I,GL_RED_INTEGER,GL_BYTE,1) : F(GL_R8,GL_RED,GL_UNSIGNED_BYTE,1); break;
    case 0x20: r= F(GL_RGB9_E5,GL_RGB,GL_UNSIGNED_INT_5_9_9_9_REV,4); break;
    case 0x21: r= F(GL_R11F_G11F_B10F,GL_RGB,GL_UNSIGNED_INT_10F_11F_11F_REV,4); break;
    case 0x24: r= FC(srgb? 0x8C4D : 0x83F1,8); break;
    case 0x25: r= FC(srgb? 0x8C4E : 0x83F2,16); break;
    case 0x26: r= FC(srgb? 0x8C4F : 0x83F3,16); break;
    case 0x27: r= FC(sn? 0x8DBC : 0x8DBB,8); break;
    case 0x28: r= FC(sn? 0x8DBE : 0x8DBD,16); break;
    case 0x17: r= FC(srgb? 0x8E8D : 0x8E8C,16); break;
    case 0x10: r= FC(0x8E8E,16); break;
    case 0x11: r= FC(0x8E8F,16); break;
    case 0x40: r=FASTC(4,4); break; case 0x50: r=FASTC(5,4); break; case 0x41: r=FASTC(5,5); break; case 0x51: r=FASTC(6,5); break;
    case 0x42: r=FASTC(6,6); break; case 0x55: r=FASTC(8,5); break; case 0x52: r=FASTC(8,6); break; case 0x44: r=FASTC(8,8); break;
    case 0x56: r=FASTC(10,5); break; case 0x57: r=FASTC(10,6); break; case 0x53: r=FASTC(10,8); break; case 0x45: r=FASTC(10,10); break;
    case 0x54: r=FASTC(12,10); break; case 0x46: r=FASTC(12,12); break;
    case 0x29: case 0x2A: case 0x2B: r= FD(GL_DEPTH24_STENCIL8,GL_DEPTH_STENCIL,GL_UNSIGNED_INT_24_8,4); break;
    case 0x2F: r= FD(GL_DEPTH_COMPONENT32F,GL_DEPTH_COMPONENT,GL_FLOAT,4); break;
    case 0x30: r= FD(GL_DEPTH32F_STENCIL8,GL_DEPTH_STENCIL,GL_FLOAT_32_UNSIGNED_INT_24_8_REV,8); break;   /* ZF32_X24S8 */
    case 0x3A: r= FD(GL_DEPTH_COMPONENT16,GL_DEPTH_COMPONENT,GL_UNSIGNED_SHORT,2); break;
    default: r= F(GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,4); r.bpp=0; { static u32 said[4]; int k; for(k=0;k<4&&said[k]!=f+1;k++); if(k==4){ for(k=0;k<4&&said[k];k++); if(k<4){ said[k]=f+1; WARN("gpu: unsupported texture format %x (ignored)\n",f); } } } break;
  }
  if(r.astc && srgb) r.ifmt=GL_SRGB8_ALPHA8;
  if(r.astc && g_astc_bc){ r.ifmt= srgb? 0x8C4F /* SRGB_ALPHA_S3TC_DXT5 */ : 0x83F3 /* RGBA_S3TC_DXT5 */; r.tobc=1; }
  /* hw ASTC (Intel Gen9-11 etc): upload compressed */
  else if(r.astc && g_has_astc && !getenv("SWR_ASTC_CPU")){ static const u8 bwh[14][2]={{4,4},{5,4},{5,5},{6,5},{6,6},{8,5},{8,6},{8,8},{10,5},{10,6},{10,8},{10,10},{12,10},{12,12}};
    for(int k=0;k<14;k++) if(bwh[k][0]==r.bw && bwh[k][1]==r.bh){ r.ifmt=(srgb? 0x93D0 : 0x93B0)+k; r.astc=0; break; } }
  r.is_int= (r.fmt==GL_RGBA_INTEGER||r.fmt==GL_RG_INTEGER||r.fmt==GL_RED_INTEGER);
  return r; }

/* SNORM color targets dont have to be renderable in GL. if the driver gets them wrong (gl_selftest checks) they're
   stored as half float, the transfers still convert from/to the game's snorm data */
static int g_snorm_fb;

static GlFmt rt_format(u32 f){
  switch(f){
    case 0xC0: return F(GL_RGBA32F,GL_RGBA,GL_FLOAT,16);
    case 0xC2: return F(GL_RGBA32UI,GL_RGBA_INTEGER,GL_UNSIGNED_INT,16);
    case 0xC1: case 0xC4: return F(GL_RGBA32I,GL_RGBA_INTEGER,GL_INT,16);
    case 0xC3: return F(GL_RGBA32F,GL_RGBA,GL_FLOAT,16);
    case 0xC5: return F(GL_RGBA32UI,GL_RGBA_INTEGER,GL_UNSIGNED_INT,16);
    case 0xC8: return F(GL_RGBA16I,GL_RGBA_INTEGER,GL_SHORT,8);
    case 0xC9: return F(GL_RGBA16UI,GL_RGBA_INTEGER,GL_UNSIGNED_SHORT,8);
    case 0xCC: return F(GL_RG32I,GL_RG_INTEGER,GL_INT,8);
    case 0xD2: return F(GL_RGB10_A2UI,GL_RGBA_INTEGER,GL_UNSIGNED_INT_2_10_10_10_REV,4);
    case 0xD8: return F(GL_RGBA8I,GL_RGBA_INTEGER,GL_BYTE,4);
    case 0xDC: return F(GL_RG16I,GL_RG_INTEGER,GL_SHORT,4);
    case 0xDD: return F(GL_RG16UI,GL_RG_INTEGER,GL_UNSIGNED_SHORT,4);
    case 0xE3: return F(GL_R32I,GL_RED_INTEGER,GL_INT,4);
    case 0xEC: return F(GL_RG8I,GL_RG_INTEGER,GL_BYTE,2);
    case 0xED: return F(GL_RG8UI,GL_RG_INTEGER,GL_UNSIGNED_BYTE,2);
    case 0xEF: return F(g_snorm_fb? GL_R16F : GL_R16_SNORM,GL_RED,GL_SHORT,2);
    case 0xF0: return F(GL_R16I,GL_RED_INTEGER,GL_SHORT,2);
    case 0xF4: return F(g_snorm_fb? GL_R16F : GL_R8_SNORM,GL_RED,GL_BYTE,1);
    case 0xF5: return F(GL_R8I,GL_RED_INTEGER,GL_BYTE,1);
    case 0xC6: return F(GL_RGBA16,GL_RGBA,GL_UNSIGNED_SHORT,8);
    case 0xC7: return F(g_snorm_fb? GL_RGBA16F : GL_RGBA16_SNORM,GL_RGBA,GL_SHORT,8);
    case 0xCA: case 0xCE: return F(GL_RGBA16F,GL_RGBA,GL_HALF_FLOAT,8);
    case 0xCB: return F(GL_RG32F,GL_RG,GL_FLOAT,8);
    case 0xCD: return F(GL_RG32UI,GL_RG_INTEGER,GL_UNSIGNED_INT,8);
    case 0xCF: case 0xD5: return F(GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,4);
    case 0xD0: case 0xD6: return F(GL_SRGB8_ALPHA8,GL_RGBA,GL_UNSIGNED_BYTE,4);
    case 0xD1: return F(GL_RGB10_A2,GL_RGBA,GL_UNSIGNED_INT_2_10_10_10_REV,4);
    case 0xD7: return F(g_snorm_fb? GL_RGBA16F : GL_RGBA8_SNORM,GL_RGBA,GL_BYTE,4);
    case 0xD9: return F(GL_RGBA8UI,GL_RGBA_INTEGER,GL_UNSIGNED_BYTE,4);
    case 0xDA: return F(GL_RG16,GL_RG,GL_UNSIGNED_SHORT,4);
    case 0xDB: return F(g_snorm_fb? GL_RG16F : GL_RG16_SNORM,GL_RG,GL_SHORT,4);
    case 0xDE: return F(GL_RG16F,GL_RG,GL_HALF_FLOAT,4);
    case 0xE0: return F(GL_R11F_G11F_B10F,GL_RGB,GL_UNSIGNED_INT_10F_11F_11F_REV,4);
    case 0xE4: return F(GL_R32UI,GL_RED_INTEGER,GL_UNSIGNED_INT,4);
    case 0xE5: return F(GL_R32F,GL_RED,GL_FLOAT,4);
    case 0xE8: return F(GL_RGB565,GL_RGB,GL_UNSIGNED_SHORT_5_6_5,2);
    case 0xE9: return F(GL_RGB5_A1,GL_RGBA,GL_UNSIGNED_SHORT_1_5_5_5_REV,2);
    case 0xEA: return F(GL_RG8,GL_RG,GL_UNSIGNED_BYTE,2);
    case 0xEB: return F(g_snorm_fb? GL_RG16F : GL_RG8_SNORM,GL_RG,GL_BYTE,2);
    case 0xEE: return F(GL_R16,GL_RED,GL_UNSIGNED_SHORT,2);
    case 0xF1: return F(GL_R16UI,GL_RED_INTEGER,GL_UNSIGNED_SHORT,2);
    case 0xF2: return F(GL_R16F,GL_RED,GL_HALF_FLOAT,2);
    case 0xF3: return F(GL_R8,GL_RED,GL_UNSIGNED_BYTE,1);
    case 0xF6: return F(GL_R8UI,GL_RED_INTEGER,GL_UNSIGNED_BYTE,1);
    /* depth */
    case 0x0A: return FD(GL_DEPTH_COMPONENT32F,GL_DEPTH_COMPONENT,GL_FLOAT,4);
    case 0x13: return FD(GL_DEPTH_COMPONENT16,GL_DEPTH_COMPONENT,GL_UNSIGNED_SHORT,2);
    case 0x14: case 0x15: case 0x16: case 0x18: return FD(GL_DEPTH24_STENCIL8,GL_DEPTH_STENCIL,GL_UNSIGNED_INT_24_8,4);
    case 0x19: return FD(GL_DEPTH32F_STENCIL8,GL_DEPTH_STENCIL,GL_FLOAT_32_UNSIGNED_INT_24_8_REV,8);
  }
  WARN("gpu: unsupported render target format %x\n",f);
  return F(GL_RGBA8,GL_RGBA,GL_UNSIGNED_BYTE,4); }

/* ============================================================ texture cache */
typedef struct Tex {
  u64 gva; u32 w, h, d, levels, target; u32 fkey; GlFmt f; float scale;
  u32 bl, bh, bd, pitch; u64 size;
  GLuint id; u64 hash; int is_rt, gpu_written; u32 frame_checked, last_used; GLint swz[4]; u64 tr_last; u32 stable, next_check; GLuint view; GLenum view_target; GLint vswz[4]; u32 wgen, crop_gen; u64 wfs; u32 wframe; struct Tex* crop; struct Tex* lay; struct Tex* vol; GLuint fbc; u32 fbc_w, fbc_h, fbc_gen, fbc_draw; int skipclr, zero3d; u64 alias; u64 shash; u32 tfull;
  GLuint zc; u64 zc_hash; u32 zc_wgen, zc_ok;   /* depth copy for shadow samplers (zc_ok: 1 made, 2 failed) */
  u64* seen; u32 nseen;   /* assembled textures: which RT (and write count) each layer/level was copied from */
  struct RtHist* hist; u32 nhist;   /* last draws into this RT (small RTs only), for F5 traces */
  u32 huge_w, huge_h;   /* a texture too big for the host: the size the game gave it */
  u32 smw;        /* small RT: g_frame+1 of the last frame a draw wrote it (see rt_small_writeback) */
  u32 a3d_wgen;   /* 3D texture: write count of the 2D array copied into it (same memory, see sync3d_from_array) */
  u32 skip; u64* lhash; u8* lmask; u32 vt_next;   /* virtual textures: top mips left out, per layer hash, layers to upload */
  struct Tex* next;
} Tex;

/* the real scale is per texture, not for the whole GPU */
static float TS(const Tex* t){ return t->is_rt? t->scale : 1.0f; }
static u32 TW(const Tex* t){ return swr_scaled_dim(t->w,TS(t)); }
static u32 TH(const Tex* t){ return swr_scaled_dim(t->h,TS(t)); }
static u32 TD(const Tex* t, u32 v){ return swr_scaled_dim(v,TS(t)); }

#define TEXH 4096
static Tex* texh[TEXH]; static u32 g_frame;

static u32 hkey(u64 a){ return (u32)((a>>8)^(a>>20))&(TEXH-1); }

/* GPU thread time per category, for performance.txt */
enum { TM_SH, TM_SHREADY, TM_STATE, TM_TEX, TM_BUF, TM_DRAW, TM_SWAP, TM_CB, TM_GB, TM_N };

static u64 g_tm[TM_N], g_tmp[TM_N]; static u64 st_sh_new;
#define TM(k_,code_) do{ u64 t_=plat_ns(); code_; g_tm[k_]+=plat_ns()-t_; }while(0)

/* shadow sampler (compare) on a color texture. the switch compares against the red channel but GL needs a depth
   texture and gives 0 ("all in shadow") otherwise, and splatoon 2 binds a 4x4 R32F "no shadow" texture to every
   lit material, so the whole level was black. we keep a DEPTH_COMPONENT32F copy and redo it when the data changes
   (uploads by hash, render targets by write generation). desktop GL only (glGetTexImage) */
static GLuint shadow_depth_copy(Tex* t){
  if(t->zc_ok==2) return 0;
  if(t->target!=GL_TEXTURE_2D || t->f.depth || t->f.is_int || t->f.compressed) return 0;
  if(t->f.ifmt!=GL_R32F && t->f.ifmt!=GL_R16F && t->f.ifmt!=GL_R16) return 0;
  if(t->is_rt && (u64)t->w*t->h>512u*512u){ t->zc_ok=2; return 0; }   /* a big render target would need a readback every frame */
  if(t->zc && t->zc_hash==t->hash && t->zc_wgen==t->wgen) return t->zc;
  u64 n=(u64)t->w*t->h; float* px=malloc(n*4); if(!px){ t->zc_ok=2; return 0; }
  glBindTexture(GL_TEXTURE_2D,t->id); glPixelStorei(GL_PACK_ALIGNMENT,1); glGetTexImage(GL_TEXTURE_2D,0,GL_RED,GL_FLOAT,px);
  for(u64 k=0;k<n;k++){ if(!(px[k]>=0.0f)) px[k]=0.0f; if(px[k]>1.0f) px[k]=1.0f; }
  if(!t->zc){ glGenTextures(1,&t->zc); glBindTexture(GL_TEXTURE_2D,t->zc); glTexStorage2D(GL_TEXTURE_2D,1,GL_DEPTH_COMPONENT32F,t->w,t->h);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,0); }
  else glBindTexture(GL_TEXTURE_2D,t->zc);
  glPixelStorei(GL_UNPACK_ALIGNMENT,1); glPixelStorei(GL_UNPACK_ROW_LENGTH,0);
  glTexSubImage2D(GL_TEXTURE_2D,0,0,0,t->w,t->h,GL_DEPTH_COMPONENT,GL_FLOAT,px); free(px);
  t->zc_hash=t->hash; t->zc_wgen=t->wgen; t->zc_ok=1;
  { static int said; if(said<sw2_cap(4)){ said++; LOG("gpu: %ux%u %s %s read by a shadow sampler: using a depth copy\n",t->w,t->h,t->f.ifmt==GL_R32F? "R32F" : "R16",t->is_rt? "render target" : "texture"); } }
  return t->zc; }

/* full size of a bl texture, all layers and mips */
static u64 bl_level_size(u32 w, u32 h, u32 d, const GlFmt* f, u32 bh, u32 bd, u32 level, u32* out_bh){
  u32 lw=w>>level, lh=h>>level, ld=d>>level; if(!lw) lw=1; if(!lh) lh=1; if(!ld) ld=1;
  u32 tw=(lw+f->bw-1)/f->bw, th=(lh+f->bh-1)/f->bh; u32 rowb=tw*f->bpp;
  u32 gx=(rowb+63)/64, gy=(th+7)/8;
  u32 b=bh; while(b>0 && gy<=(1u<<(b-1))) b--;
  u32 bdd=bd; while(bdd>0 && ld<=(1u<<(bdd-1))) bdd--;
  if(out_bh) *out_bh=b;
  u64 gys=((gy+(1u<<b)-1)>>b)<<b; u64 lds=((ld+(1u<<bdd)-1)>>bdd)<<bdd;
  return (u64)gx*gys*lds*512; }


/* hack for SMO's cap kingdom sea: the VS moves the surface by (h-0.5)*k with an R16F heightmap that's still all
   zeros here (no idea yet what writes it). zeros push the sea up into the sky and 0.5 is neutral, so only while
   the memory really is all zero */
static void mb_reset(void);

static int tex_zero_height_fix(Tex* t){
  static int off=-1; if(off<0) off=getenv("SWR_NO_SEA_FIX")!=NULL; if(off) return 0;
  if((t->fkey&0x7FFFF)!=0x7ff9b || t->is_rt || t->target!=GL_TEXTURE_2D || t->levels<6 || t->w!=t->h || t->w<256 || (t->w&(t->w-1)) || !t->size) return 0;
  const u8* p=gmem_ptr(t->gva,t->size); if(!p) return 0;
  const u64* q=(const u64*)p; for(u64 k=0;k<t->size/8;k++) if(q[k]) return 0;
  u16* half=malloc((size_t)t->w*t->h*2); if(!half) return 0; for(u64 k=0;k<(u64)t->w*t->h;k++) half[k]=0x3800;
  glBindTexture(GL_TEXTURE_2D,t->id); glPixelStorei(GL_UNPACK_ALIGNMENT,1); glPixelStorei(GL_UNPACK_ROW_LENGTH,0);
  for(u32 l=0;l<t->levels;l++){ u32 lw=t->w>>l, lh=t->h>>l; if(!lw) lw=1; if(!lh) lh=1; glTexSubImage2D(GL_TEXTURE_2D,l,0,0,lw,lh,GL_RED,GL_HALF_FLOAT,half); }
  free(half); mb_reset();
  { static int said; if(said<sw2_cap(8)){ said++; LOG("gpu: R16F heightmap %ux%u at %llx is all zero: filled with neutral height 0.5 (sea workaround, SWR_NO_SEA_FIX=1 disables)\n",t->w,t->h,(unsigned long long)t->gva); } }
  return 1; }

static int getenv_cached_nolut16(void){ static int v=-1; if(v<0) v=getenv("SWR_NO_LUT16")!=NULL; return v; }   /* 1 = upload packed */

static u16 lut_f16(float v){
  u32 x; memcpy(&x,&v,4); u32 sgn=(x>>16)&0x8000; i32 e=(i32)((x>>23)&0xFF)-127+15; u32 m=x&0x7FFFFF;
  if(((x>>23)&0xFF)==0xFF) return (u16)(sgn|0x7C00|(m? 0x200 : 0));
  if(e>=31) return (u16)(sgn|0x7C00);
  if(e<=0){ if(e<-10) return (u16)sgn; m|=0x800000; return (u16)(sgn|(m>>(14-e))); }
  return (u16)(sgn|((u32)e<<10)|(m>>13)); }

/* one channel of R11G11B10F: 5-bit exponent, 6 or 5-bit mantissa, no sign */
static float uf_decode(u32 v, int mbits){
  u32 e=(v>>mbits)&31, m=v&((1u<<mbits)-1);
  if(e==31) return m? 0.0f : 65504.0f;
  if(!e) return (float)m/(float)(1u<<mbits)*(1.0f/16384.0f);
  return ldexpf(1.0f+(float)m/(float)(1u<<mbits),(int)e-15); }

/* ---- parallel for. texture loads (deswizzle + ASTC/BC decode) are split by rows over the cpu cores, on a level
   load this was most of what people call "shader stutter" */
typedef void (*PfFn)(void* c, u32 a, u32 b);
/* one job per call. a thread only takes the job under the lock while it's published, and the caller waits for
   every thread that took it, so a late thread never touches the next job (that race blanked textures) */
typedef struct { PfFn fn; void* ctx; u32 n, ch, refs; volatile u32 next; } PfJob;
static struct { PMutex lk; PCond cv, dn; PfJob* job; u32 gen; int nth, init; } g_pf;
static void pf_work(PfJob* j){ for(;;){ u32 a=__atomic_fetch_add(&j->next,j->ch,__ATOMIC_ACQ_REL); if(a>=j->n) break; u32 b=a+j->ch; if(b>j->n) b=j->n; j->fn(j->ctx,a,b); } }
static void pf_thread(void* arg){ (void)arg; u32 seen=0;
  for(;;){ pm_lock(&g_pf.lk); while(g_pf.gen==seen) pc_wait(&g_pf.cv,&g_pf.lk,1000000000ll);
    seen=g_pf.gen; PfJob* j=g_pf.job; if(!j){ pm_unlock(&g_pf.lk); continue; } j->refs++; pm_unlock(&g_pf.lk);
    pf_work(j);
    pm_lock(&g_pf.lk); if(!--j->refs) pc_broadcast(&g_pf.dn); pm_unlock(&g_pf.lk); } }
static void pf_run(PfFn fn, void* ctx, u32 n, u32 min_par){
  if(!g_pf.init){ g_pf.init=1; pm_init(&g_pf.lk); pc_init(&g_pf.cv); pc_init(&g_pf.dn);
    extern int plat_ncpu(void); int nc=plat_ncpu(); g_pf.nth= nc>1? nc-1 : 0; if(g_pf.nth>7) g_pf.nth=7;
    const char* e=getenv("SWR_DECODE_THREADS"); if(e){ g_pf.nth=atoi(e)-1; if(g_pf.nth<0) g_pf.nth=0; if(g_pf.nth>15) g_pf.nth=15; }
    int ok=0; for(int i=0;i<g_pf.nth;i++) if(!plat_thread(pf_thread,NULL,1u<<20)) ok++; g_pf.nth=ok;
    LOG("textures: %d thread(s) decode and deswizzle texture loads\n",g_pf.nth+1); }
  if(!g_pf.nth || n<min_par){ fn(ctx,0,n); return; }
  PfJob j; j.fn=fn; j.ctx=ctx; j.n=n; j.ch=n/((u32)(g_pf.nth+1)*4); if(!j.ch) j.ch=1; j.refs=0; j.next=0;
  pm_lock(&g_pf.lk); g_pf.job=&j; g_pf.gen++; pc_broadcast(&g_pf.cv); pm_unlock(&g_pf.lk);
  pf_work(&j);
  pm_lock(&g_pf.lk); g_pf.job=NULL; while(j.refs) pc_wait(&g_pf.dn,&g_pf.lk,1000000000ll); pm_unlock(&g_pf.lk); }

/* deswizzle: one call per range of rows (z*th + y) */
typedef struct { u8* lin; const u8* src; u64 lsz; u32 rowb, th, lbh, lbd; } DsJob;
/* inside a GOB the four 16 byte pieces of a row are at fixed offsets (0, 32, 256, 288), so the block/GOB base only
   gets computed once per 64 bytes instead of a dozen divisions per 16 bytes */
static void ds_rows(void* c, u32 a, u32 b){ DsJob* j=c;
  u32 rowb=j->rowb, th=j->th, gob_w=(rowb+63)/64, bh=1u<<j->lbh, bd=1u<<j->lbd, blocks_h=((th+7)/8+bh-1)/bh; u64 bsz=512ull*bh*bd;
  static const u32 seg[4]={0,32,256,288}; int nat=gob_native_on();
  for(u32 r=a;r<b;r++){ u32 z=r/th, y=r%th; u8* dr=j->lin+(u64)r*rowb; u32 gy=y/8, yb=y%8;
    u64 row=((u64)(z/bd)*blocks_h*gob_w + (u64)(gy/bh)*gob_w)*bsz + ((u64)(z%bd)*bh + gy%bh)*512; u32 rin=(yb/2)*64 + (yb%2)*16;
    u32 sk[4]; for(u32 k=0;k<4;k++) sk[k]= nat? gob_native(rin+seg[k]) : rin+seg[k];
    for(u32 gx=0;gx<gob_w;gx++){ u64 base=row+(u64)gx*bsz; u32 x0=gx*64;
      for(u32 k=0;k<4;k++){ u32 x=x0+k*16; if(x>=rowb) break; u32 n=rowb-x<16? rowb-x : 16; u64 o=base+sk[k]; if(o+n<=j->lsz) memcpy(dr+x,j->src+o,n); } } } }
/* ASTC / BCn: ranges of block rows */
typedef struct { const u8* src; u8* dst; u32 w, h, bw, bh, kind, bsz; } DecJob;
extern void astc_decode_rows(const u8* src, u32 w, u32 h, u32 bw, u32 bh, u8* dst, u32 j0, u32 j1);
static void astc_rows(void* c, u32 a, u32 b){ DecJob* j=c; astc_decode_rows(j->src,j->w,j->h,j->bw,j->bh,j->dst,a,b); }
/* RGBA8 -> BC3 (DXT5): bounding box endpoints along the color diagonal the block leans on, nearest palette entry per
   texel; alpha as an 8 value ramp between its min and max */
static u16 to565(int r, int g, int b){ return (u16)(((r*31+127)/255)<<11 | ((g*63+127)/255)<<5 | ((b*31+127)/255)); }
static void from565(u16 c, int* o){ int r=(c>>11)&31, g=(c>>5)&63, b=c&31; o[0]=(r<<3)|(r>>2); o[1]=(g<<2)|(g>>4); o[2]=(b<<3)|(b>>2); }
static void bc3_block(const u8 px[16][4], u8* out){
  /* alpha */
  int amin=255, amax=0; for(int i=0;i<16;i++){ int a=px[i][3]; if(a<amin) amin=a; if(a>amax) amax=a; }
  out[0]=(u8)amax; out[1]=(u8)amin; u64 ab=0;
  if(amax>amin){ int pal[8]={amax,amin,(6*amax+amin)/7,(5*amax+2*amin)/7,(4*amax+3*amin)/7,(3*amax+4*amin)/7,(2*amax+5*amin)/7,(amax+6*amin)/7};
    for(int i=0;i<16;i++){ int a=px[i][3], bi=0, bd=1<<30; for(int k=0;k<8;k++){ int dd=a-pal[k]; dd*=dd; if(dd<bd){ bd=dd; bi=k; } } ab|=(u64)bi<<(3*i); } }
  for(int k=0;k<6;k++) out[2+k]=(u8)(ab>>(8*k));
  /* color: box, flipped on the channels that run against green */
  int mn[3]={255,255,255}, mx[3]={0,0,0}, mean[3]={0,0,0};
  for(int i=0;i<16;i++) for(int c=0;c<3;c++){ int v=px[i][c]; if(v<mn[c]) mn[c]=v; if(v>mx[c]) mx[c]=v; mean[c]+=v; }
  for(int c=0;c<3;c++) mean[c]=(mean[c]+8)/16;
  int cov_rg=0, cov_bg=0; for(int i=0;i<16;i++){ int g=px[i][1]-mean[1]; cov_rg+=(px[i][0]-mean[0])*g; cov_bg+=(px[i][2]-mean[2])*g; }
  int e0[3]={mx[0],mx[1],mx[2]}, e1[3]={mn[0],mn[1],mn[2]};
  if(cov_rg<0){ e0[0]=mn[0]; e1[0]=mx[0]; } if(cov_bg<0){ e0[2]=mn[2]; e1[2]=mx[2]; }
  for(int c=0;c<3;c++){ int in=(e0[c]-e1[c])/16; e0[c]-=in; e1[c]+=in; }   /* inset a little */
  u16 c0=to565(e0[0],e0[1],e0[2]), c1=to565(e1[0],e1[1],e1[2]); u32 idx=0;
  if(c0<c1){ u16 t=c0; c0=c1; c1=t; }
  if(c0!=c1){ int p[4][3]; from565(c0,p[0]); from565(c1,p[1]); for(int c=0;c<3;c++){ p[2][c]=(2*p[0][c]+p[1][c])/3; p[3][c]=(p[0][c]+2*p[1][c])/3; }
    for(int i=0;i<16;i++){ int bi=0, bd=1<<30; for(int k=0;k<4;k++){ int dr=px[i][0]-p[k][0], dg=px[i][1]-p[k][1], db=px[i][2]-p[k][2]; int dd=dr*dr+dg*dg+db*db; if(dd<bd){ bd=dd; bi=k; } } idx|=(u32)bi<<(2*i); } }
  out[8]=(u8)c0; out[9]=(u8)(c0>>8); out[10]=(u8)c1; out[11]=(u8)(c1>>8); out[12]=(u8)idx; out[13]=(u8)(idx>>8); out[14]=(u8)(idx>>16); out[15]=(u8)(idx>>24); }
typedef struct { const u8* src; u8* dst; u32 w, h; } BcEnc;
static void bc3_rows(void* c, u32 a, u32 b){ BcEnc* j=c; u32 bx=(j->w+3)/4;
  for(u32 by=a;by<b;by++) for(u32 x=0;x<bx;x++){ u8 px[16][4];
    for(u32 yy=0;yy<4;yy++) for(u32 xx=0;xx<4;xx++){ u32 sx=x*4+xx, sy=by*4+yy; if(sx>=j->w) sx=j->w-1; if(sy>=j->h) sy=j->h-1; memcpy(px[yy*4+xx],j->src+((size_t)sy*j->w+sx)*4,4); }
    bc3_block((const u8(*)[4])px,j->dst+((size_t)by*bx+x)*16); } }
static void bc_rows(void* c, u32 a, u32 b){ DecJob* j=c; u32 bx=(j->w+3)/4; u32 y0=a*4, y1=b*4; if(y1>j->h) y1=j->h; if(y0>=y1) return;
  swr_bc_decode((int)j->kind,j->src+(u64)a*bx*j->bsz,j->w,y1-y0,(u32*)(j->dst+(u64)y0*j->w*4)); }

/* texture uploads get their own GPU timer entry (one line in performance.txt for all of them) */
#define GP_TEXUP_KEY 0x7E87E87E87E87E80ull
static void tex_upload_body(Tex* t);
static void gp_compute_begin(u64 key, u32 grid, u32 mb); static void gp_compute_end(void); static int gp_cs_on(void);
static void tex_upload(Tex* t){ int g=!gp_cs_on(); if(g) gp_compute_begin(GP_TEXUP_KEY,(u32)(t->size>>10),(u32)(t->size>>20)); tex_upload_body(t); if(g) gp_compute_end(); }
static void tex_upload_body(Tex* t){ st_tex_up+=t->size;
  { static int done; const char* e= done? NULL : getenv("SWR_DUMPTEX"); u32 dw=0, dh=0; if(e && sscanf(e,"%ux%u",&dw,&dh)==2 && t->w==dw && t->h==dh && t->size){   /* debug: raw guest bytes of one texture */
      done=1; u8* b=malloc(t->size); if(b){ gmem_read(t->gva,b,t->size); FILE* f=fopen("captures/tex_raw.bin","wb"); if(f){ fwrite(b,1,t->size,f); fclose(f); } free(b); }
      LOG("dumped texture %ux%u at %llx: %llu bytes, fkey %x, block linear %u, block height %u, astc %d\n",t->w,t->h,(unsigned long long)t->gva,(unsigned long long)t->size,t->fkey,t->bl,t->bh,t->f.astc); } }
  if(tex_zero_height_fix(t)) return;
  const GlFmt* f=&t->f;
  u32 layers= t->target==GL_TEXTURE_3D? 1 : t->d;
  u32 depth3d= t->target==GL_TEXTURE_3D? t->d : 1;
  u32 sk=t->skip, FW= sk? t->huge_w : t->w, FH= sk? t->huge_h : t->h, FL=t->levels+sk;   /* full size in guest memory */
  u64 layer_size=0; for(u32 l=0;l<FL;l++) layer_size+=bl_level_size(FW,FH,depth3d,f,t->bh,t->bd,l,NULL);
  if(t->bl && layers>1){ u64 al=512ull<<t->bh; layer_size=(layer_size+al-1)/al*al; }
  glBindTexture(t->target,t->id);
  glPixelStorei(GL_UNPACK_ALIGNMENT,1); glPixelStorei(GL_UNPACK_ROW_LENGTH,0);
  int was_zero3d= t->target==GL_TEXTURE_3D? t->zero3d : 0; (void)was_zero3d;
  if(t->target==GL_TEXTURE_3D) t->zero3d=1;
  for(u32 layer=0;layer<layers;layer++){
    if(t->lmask && !t->lmask[layer]) continue;
    u64 off=layer*layer_size;
    for(u32 l=0;l<FL;l++){
      u32 lw=FW>>l, lh=FH>>l, ld=depth3d>>l; if(!lw) lw=1; if(!lh) lh=1; if(!ld) ld=1;
      u32 tw=(lw+f->bw-1)/f->bw, th=(lh+f->bh-1)/f->bh; u32 rowb=tw*f->bpp; u32 lbh;
      u64 lsz= t->bl? bl_level_size(FW,FH,depth3d,f,t->bh,t->bd,l,&lbh) : (u64)t->pitch*th;
      if(l<sk){ off+=lsz; continue; }
      u32 gl_l=l-sk;
      u64 lin_size=(u64)rowb*th*ld; u8* lin=malloc(lin_size? lin_size : 1);
      u8* src=malloc(lsz? lsz : 1);
      if(!lin || !src){ free(lin); free(src); static int said; if(said<sw2_cap(4)){ said++; LOG("gpu: out of memory uploading texture %ux%ux%u (%llu MB): skipped\n",t->w,t->h,t->d,(unsigned long long)(lsz>>20)); } off+= t->bl? lsz : (u64)t->pitch*th; continue; }
      gmem_read(t->gva+off,src,lsz);
      if(t->target==GL_TEXTURE_3D && t->zero3d){ for(u64 k=0;k<lsz;k++) if(src[k]){ t->zero3d=0; break; }
        if(!t->zero3d && was_zero3d){ static int said; if(said<sw2_cap(16)){ said++; LOG("gpu: 3D texture %ux%ux%u at %llx got its data in memory (frame %u)\n",t->w,t->h,t->d,(unsigned long long)t->gva,g_frame); } } }
      if(t->bl){ u32 lbd=t->bd; while(lbd>0 && ld<=(1u<<(lbd-1))) lbd--;   /* 3D: block depth (LUTs) */
        DsJob dj={lin,src,lsz,rowb,th,lbh,lbd}; pf_run(ds_rows,&dj,ld*th,(u32)((256u<<10)/(rowb? rowb : 1))); }
      else { for(u32 y=0;y<th;y++) memcpy(lin+(u64)y*rowb,src+(u64)y*t->pitch,rowb); }
      const u8* up=lin; u8* dec=NULL;
      if(f->dec==20){ u64 n=(u64)lw*lh*ld; u16* h=malloc((size_t)n*8+8);
        if(h){ for(u64 k=0;k<n;k++){ u32 v; memcpy(&v,lin+k*4,4); u16* q=h+k*4;
            q[0]=lut_f16(uf_decode(v&0x7FF,6)); q[1]=lut_f16(uf_decode((v>>11)&0x7FF,6)); q[2]=lut_f16(uf_decode(v>>22,5)); q[3]=0x3C00; }
          glTexSubImage3D(GL_TEXTURE_3D,gl_l,0,0,0,lw,lh,ld,GL_RGBA,GL_HALF_FLOAT,h); free(h); }
        free(lin); free(src); off+= t->bl? lsz : (u64)t->pitch*th; continue; }
      if((f->astc||f->dec) && !(dec=malloc((size_t)lw*lh*4+64))){ free(lin); free(src); off+= t->bl? lsz : (u64)t->pitch*th; continue; }
      if(f->astc){ DecJob dj={lin,dec,lw,lh,f->bw,f->bh,0,16}; pf_run(astc_rows,&dj,(lh+f->bh-1)/f->bh,16); up=dec; }
      if(f->astc && f->tobc){ u32 bx=(lw+3)/4, by=(lh+3)/4; GLsizei isz=(GLsizei)((size_t)bx*by*16); u8* bc=malloc((size_t)isz+16);
        if(bc){ BcEnc ej={dec,bc,lw,lh}; pf_run(bc3_rows,&ej,by,8);
          if(t->target==GL_TEXTURE_2D) glCompressedTexSubImage2D(GL_TEXTURE_2D,gl_l,0,0,lw,lh,f->ifmt,isz,bc);
          else if(t->target==GL_TEXTURE_2D_ARRAY || t->target==GL_TEXTURE_CUBE_MAP_ARRAY) glCompressedTexSubImage3D(t->target,gl_l,0,0,layer,lw,lh,1,f->ifmt,isz,bc);
          else if(t->target==GL_TEXTURE_CUBE_MAP) glCompressedTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+layer,gl_l,0,0,lw,lh,f->ifmt,isz,bc);
          free(bc); }
        free(dec); free(lin); free(src); off+= t->bl? lsz : (u64)t->pitch*th; continue; }
      else if(f->dec){ u32* o=(u32*)dec;
        if(f->dec<10){ DecJob dj={lin,dec,lw,lh,4,4,f->dec,(f->dec==2||f->dec==5)? 8u : 16u}; pf_run(bc_rows,&dj,(lh+3)/4,32); }
        else { const u16* q=(const u16*)lin; for(u32 y=0;y<lh;y++) for(u32 x=0;x<lw;x++){ u32 v=q[(size_t)y*(rowb/2)+x], rr,gg,bb,aa;
            if(f->dec==10){ rr=(v&15)*17; gg=((v>>4)&15)*17; bb=((v>>8)&15)*17; aa=((v>>12)&15)*17; }
            else if(f->dec==11){ rr=(v&31)*255/31; gg=((v>>5)&31)*255/31; bb=((v>>10)&31)*255/31; aa=(v>>15)? 255 : 0; }
            else { rr=(v&31)*255/31; gg=((v>>5)&63)*255/63; bb=((v>>11)&31)*255/31; aa=255; }
            o[(size_t)y*lw+x]=rr|(gg<<8)|(bb<<16)|(aa<<24); } }
        up=dec; }
      { static int dmp=-1, nd; if(dmp<0) dmp=getenv("SWR_DUMP_TEX")!=NULL;
        if(dmp && l==0 && layer==0 && nd<20 && (f->astc || (!f->compressed && f->bpp==4))){ char pth[128]; snprintf(pth,sizeof pth,"tex_%02d_%llx_%ux%u.ppm",nd++,(unsigned long long)t->gva,lw,lh);
          FILE* fo=fopen(pth,"wb"); if(fo){ fprintf(fo,"P6\n%u %u\n255\n",lw,lh); for(u64 k=0;k<(u64)lw*lh;k++) fwrite(up+k*4,1,3,fo); fclose(fo); } } }
      if(f->compressed && !f->astc && !f->dec){
        GLsizei isz=(GLsizei)(rowb*th);
        if(t->target==GL_TEXTURE_2D) glCompressedTexSubImage2D(GL_TEXTURE_2D,gl_l,0,0,lw,lh,f->ifmt,isz,up);
        else if(t->target==GL_TEXTURE_2D_ARRAY || t->target==GL_TEXTURE_CUBE_MAP_ARRAY) glCompressedTexSubImage3D(t->target,gl_l,0,0,layer,lw,lh,1,f->ifmt,isz,up);
        else if(t->target==GL_TEXTURE_CUBE_MAP) glCompressedTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+layer,gl_l,0,0,lw,lh,f->ifmt,isz,up);
        else if(t->target==GL_TEXTURE_3D) glCompressedTexSubImage3D(GL_TEXTURE_3D,gl_l,0,0,0,lw,lh,ld,f->ifmt,isz*ld,up);
      } else {
        GLenum fm= (f->astc||f->dec)? GL_RGBA : f->fmt, ty= (f->astc||f->dec)? GL_UNSIGNED_BYTE : f->type;
        if(t->target==GL_TEXTURE_2D) glTexSubImage2D(GL_TEXTURE_2D,gl_l,0,0,lw,lh,fm,ty,up);
        else if(t->target==GL_TEXTURE_2D_ARRAY || t->target==GL_TEXTURE_CUBE_MAP_ARRAY) glTexSubImage3D(t->target,gl_l,0,0,layer,lw,lh,1,fm,ty,up);
        else if(t->target==GL_TEXTURE_CUBE_MAP) glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+layer,gl_l,0,0,lw,lh,fm,ty,up);
        else if(t->target==GL_TEXTURE_3D) glTexSubImage3D(GL_TEXTURE_3D,gl_l,0,0,0,lw,lh,ld,fm,ty,up);
        else if(t->target==GL_TEXTURE_1D) glTexSubImage2D(GL_TEXTURE_2D,gl_l,0,0,lw,1,fm,ty,up);
      }
      free(dec); free(lin); free(src);
      off+= t->bl? lsz : (u64)t->pitch*th;
    }
  }
}

static u32 g_tex_bud_frame; static i64 g_tex_bud; static u64 st_tex_chk;

static u64 fast_hash(const u8* p, u64 size, int sampled){
  const u64* q=(const u64*)p; u64 n=size/8, step= (sampled && n>8192)? 16 : 1;   /* sample one word per 128 bytes */
  u64 a=0x9E3779B97F4A7C15ull, b=0xC2B2AE3D27D4EB4Full, c=0x165667B19E3779F9ull, d=0x27D4EB2F165667C5ull; u64 i=0;
  for(; i+3*step<n; i+=4*step){ a=(a^q[i])*0x100000001B3ull; b=(b^q[i+step])*0x100000001B3ull; c=(c^q[i+2*step])*0x100000001B3ull; d=(d^q[i+3*step])*0x100000001B3ull; }
  for(; i<n; i+=step){ a=(a^q[i])*0x100000001B3ull; }
  for(u64 k=n*8;k<size;k++){ b=(b^p[k])*0x100000001B3ull; }
  return a ^ (b<<1) ^ (c<<2) ^ (d<<3) ^ size; }

static u64 sample_hash(u64 gva, u64 len); static int g_nosample;

static u64 mem_hash(u64 gva, u64 size){
  /* quick sample: one word per 64 bytes + the first 4K */
  u64 h=1469598103934665603ull; u8* p=gmem_ptr(gva,size);
  if(p) return fast_hash(p,size,1);
  for(u64 o=0;o<size;o+=4096){ u64 v[8]; gmem_read(gva+o,v,size-o<64? size-o : 64); for(int i=0;i<8;i++){ h^=v[i]; h*=1099511628211ull; } }
  return h; }

static u64 rt_sig(u64 gva){ u8* p=gmem_ptr(gva,4096); if(p) return fast_hash(p,4096,0); u8 b[4096]; gmem_read(gva,b,4096); return fast_hash(b,4096,0); }

static GLenum tex_target(u32 type, u32 depth){
  switch(type){ case 0: return GL_TEXTURE_2D; case 1: case 7: return depth>1? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D; case 2: return GL_TEXTURE_3D; case 3: return GL_TEXTURE_CUBE_MAP;
    case 4: case 5: return GL_TEXTURE_2D_ARRAY; case 8: return GL_TEXTURE_CUBE_MAP_ARRAY; default: return GL_TEXTURE_2D; } }

/* generation counter for the TIC -> texture mapping. bump it on anything that can change what a TIC points to
   (texture created/freed, RT dropped or recreated, FBO attachments, clears, new frame). tex_from_tic keeps its
   result while this stays the same */
static u32 g_tmap_gen=1;

/* mip levels a w x h (x d for 3D) texture can have. glTexStorage with more fails and leaves the texture empty */
static u32 max_levels(u32 w, u32 h, u32 d){ u32 m= w>h? w : h; if(d>m) m=d; u32 n=1; while(m>1){ m>>=1; n++; } return n; }

/* SWR_WATCH=gva,gva,...: log every texture made, render target bound and compute image bound at those addresses
   (who writes a texture that comes out empty) */
static int watch_gva(u64 a){ static u64 w[16]; static int n=-1;
  if(n<0){ n=0; const char* e=getenv("SWR_WATCH"); while(e && *e && n<16){ char* q; w[n++]=strtoull(e,&q,16); if(*q!=',') break; e=q+1; } }
  for(int i=0;i<n;i++) if(w[i]==a) return 1; return 0; }
static Tex* tex_create(u64 gva, u32 w, u32 h, u32 d, u32 levels, GLenum target, GlFmt f, u32 fkey, u32 bl, u32 bh, u32 bd, u32 pitch, int is_rt){
  g_tmap_gen++;
  if(watch_gva(gva)){ u32 m[8]={0}; gmem_read(gva,m,32); u32 nz=0; { u8* b=malloc(65536); if(b){ gmem_read(gva,b,65536); for(int k=0;k<65536;k++) nz+=b[k]!=0; free(b); } }
    extern u64 gmmu_translate(u64);
    LOG("watch %llx: texture made %ux%ux%u levels %u target %x fmt %x%s (frame %u), memory %08x %08x %08x %08x, %u of the first 64 KB nonzero, cpu address %llx\n",(unsigned long long)gva,w,h,d,levels,target,fkey,is_rt? " RT" : "",g_frame,m[0],m[1],m[2],m[3],nz,(unsigned long long)gmmu_translate(gva)); }
  Tex* t=calloc(1,sizeof *t); t->swz[0]=GL_RED; t->swz[1]=GL_GREEN; t->swz[2]=GL_BLUE; t->swz[3]=GL_ALPHA; t->gva=gva; t->w=w; t->h=h; t->d=d; t->levels=levels; t->target=target; t->f=f; t->fkey=fkey; t->bl=bl; t->bh=bh; t->bd=bd; t->pitch=pitch; t->is_rt=is_rt; t->scale=is_rt? swr_rt_scale(w,h,g_scale) : 1.0f;
  glGenTextures(1,&t->id); glBindTexture(target,t->id);
  GLenum ifmt=f.ifmt; u32 sw=TW(t), sh=TH(t);
  { u32 ml=max_levels(sw,sh,target==GL_TEXTURE_3D? d : 1); if(levels>ml) levels=ml; if(!levels) levels=1; t->levels=levels; }
  if(target==GL_TEXTURE_2D || target==GL_TEXTURE_CUBE_MAP) glTexStorage2D(target,levels,ifmt,sw,sh);
  else if(target==GL_TEXTURE_CUBE_MAP_ARRAY) glTexStorage3D(target,levels,ifmt,sw,sh,d*6>6? d : 6);
  else glTexStorage3D(target,levels,ifmt,sw,sh,d);
  glTexParameteri(target,GL_TEXTURE_MAX_LEVEL,(GLint)levels-1);
  t->last_used=g_frame; u32 k=hkey(gva); t->next=texh[k]; texh[k]=t;
  return t; }

/* virtual textures: hash the mips we keep of each layer (sampled), upload the layers that changed */
static void vt_refresh(Tex* t, int first){
  const GlFmt* f=&t->f; u32 sk=t->skip, FL=t->levels+sk; u64 ls=0, keep0=0;
  for(u32 l=0;l<FL;l++){ if(l==sk) keep0=ls; ls+=bl_level_size(t->huge_w,t->huge_h,1,f,t->bh,t->bd,l,NULL); }
  if(t->bl && t->d>1){ u64 al=512ull<<t->bh; ls=(ls+al-1)/al*al; }
  u64 kl=ls-keep0, n=0; u8* buf=malloc(kl); if(!buf) return;
  for(u32 L=0;L<t->d;L++){ gmem_read(t->gva+L*ls+keep0,buf,kl); u64 h=fast_hash(buf,kl,1);
    if(first || h!=t->lhash[L]){ t->lhash[L]=h; t->lmask[L]=1; n++; } else t->lmask[L]=0; }
  free(buf);
  if(n){ tex_upload(t); static int said; if(!first && said<sw2_cap(16)){ said++; LOG("gpu: virtual texture at %llx: %llu layer(s) changed, re-uploaded (frame %u)\n",(unsigned long long)t->gva,(unsigned long long)n,g_frame); } }
  t->vt_next=g_frame+30; }

static Tex* tex_find_rt(u64 gva){ for(Tex* t=texh[hkey(gva)];t;t=t->next) if(t->gva==gva && t->is_rt) return t; return NULL; }

/* the game samples an RT through a smaller TIC (DRS, crops). on hw uv 1.0 is the TIC edge and not the RT edge, so
   we give it a copy of just that part */
static PFNGLCOPYIMAGESUBDATAPROC p_cis; static u64 st_rt_crop;

static Tex* rt_crop(Tex* rt, u32 w, u32 h){
  if(!p_cis || getenv("SWR_NO_RTCROP")) return rt;
  Tex* c=rt->crop;
  if(c && (c->w!=w || c->h!=h)){ glDeleteTextures(1,&c->id); if(c->view) glDeleteTextures(1,&c->view); free(c); c=rt->crop=NULL; }
  if(!c){ g_tmap_gen++; c=calloc(1,sizeof *c); c->gva=rt->gva; c->w=w; c->h=h; c->d=1; c->levels=1; c->target=GL_TEXTURE_2D; c->f=rt->f; c->fkey=rt->fkey; c->is_rt=1; c->scale=TS(rt); c->bl=rt->bl; c->bh=rt->bh;
    c->swz[0]=GL_RED; c->swz[1]=GL_GREEN; c->swz[2]=GL_BLUE; c->swz[3]=GL_ALPHA;
    glGenTextures(1,&c->id); glBindTexture(GL_TEXTURE_2D,c->id); glTexStorage2D(GL_TEXTURE_2D,1,rt->f.ifmt,TD(rt,w),TD(rt,h));
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,0); c->crop_gen=~0u; rt->crop=c;
    static int said; if(said<sw2_cap(10)){ said++; LOG("gpu: %ux%u texture reads the %ux%u render target (%llx): only that region is used\n",w,h,rt->w,rt->h,(unsigned long long)rt->gva); } }
  if(c->crop_gen!=rt->wgen){ p_cis(rt->id,GL_TEXTURE_2D,0,0,0,0,c->id,GL_TEXTURE_2D,0,0,0,0,(GLsizei)TD(rt,w),(GLsizei)TD(rt,h),1); c->crop_gen=rt->wgen; st_rt_crop++; }
  c->last_used=g_frame; return c; }

/* copies into mip levels of textures built from RTs. drivers disagree here (RX 7600 gives black levels > 0 with
   CopyImageSubData, nvidia is fine), so gl_selftest() picks at startup: 0 = glCopyImageSubData, 1 = blit.
   g_mipgen_manual: glGenerateMipmap from a base level > 0 is broken, the levels get built with linear blits */
static GLuint g_rfbo, g_dfbo; static GLuint g_fbo, g_scratch_vao;
static int g_lvl_mode, g_mipgen_manual;

static void lvl_attach(GLenum fbt, GLuint tex, GLenum target, int level, int layer){
  if(target==GL_TEXTURE_2D || !tex) glFramebufferTexture2D(fbt,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,level);
  else if(target==GL_TEXTURE_CUBE_MAP) glFramebufferTexture2D(fbt,GL_COLOR_ATTACHMENT0,GL_TEXTURE_CUBE_MAP_POSITIVE_X+(GLenum)layer,tex,level);
  else glFramebufferTextureLayer(fbt,GL_COLOR_ATTACHMENT0,tex,level,layer); }

/* color blit src(target/level/layer) -> dst(level/layer); saves and restores scissor, srgb, color mask */
static void lvl_blit(GLuint src, GLenum st, int sl, int slay, GLsizei sw, GLsizei sh, GLuint dst, GLenum dtg, int dl, int dlay, GLsizei dw, GLsizei dh, GLenum filt){
  /* glGetIntegerv and not glGetBooleanv, only the first one goes through the GL call queue */
  GLboolean sc=glIsEnabled(GL_SCISSOR_TEST), sr=glIsEnabled(GL_FRAMEBUFFER_SRGB); GLint cm[4]={1,1,1,1}; glGetIntegerv(GL_COLOR_WRITEMASK,cm);
  glDisable(GL_SCISSOR_TEST); glDisable(GL_FRAMEBUFFER_SRGB); glColorMaski(0,1,1,1,1);
  glBindFramebuffer(GL_READ_FRAMEBUFFER,g_rfbo); lvl_attach(GL_READ_FRAMEBUFFER,src,st,sl,slay); glReadBuffer(GL_COLOR_ATTACHMENT0);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER,g_dfbo); lvl_attach(GL_DRAW_FRAMEBUFFER,dst,dtg,dl,dlay); { GLenum b=GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&b); }
  glBlitFramebuffer(0,0,sw,sh,0,0,dw,dh,GL_COLOR_BUFFER_BIT,filt);
  glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,0,0); glFramebufferTexture2D(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,0,0);
  glBindFramebuffer(GL_FRAMEBUFFER,g_fbo); if(sc) glEnable(GL_SCISSOR_TEST); if(sr) glEnable(GL_FRAMEBUFFER_SRGB); glColorMaski(0,cm[0]!=0,cm[1]!=0,cm[2]!=0,cm[3]!=0); }

/* read one texel of a level through the read fbo (glReadPixels is queued, glGetTextureSubImage isn't) */
static void lvl_read1(GLuint tex, GLenum target, int level, int layer, GLint x, GLint y, float* out){
  glBindFramebuffer(GL_READ_FRAMEBUFFER,g_rfbo); lvl_attach(GL_READ_FRAMEBUFFER,tex,target,level,layer); glReadBuffer(GL_COLOR_ATTACHMENT0);
  glReadPixels(x,y,1,1,GL_RGBA,GL_FLOAT,out);
  glFramebufferTexture2D(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,0,0); glBindFramebuffer(GL_FRAMEBUFFER,g_fbo); }

static void lvl_copy(GLuint src, GLuint dst, GLenum target, int level, int layer, GLsizei w, GLsizei h, int color){
  /* 2D mip chains (bloom) always go through a blit. odyssey's bloom came out blue on some intel drivers with the
     copy even when the startup check passed */
  if((!g_lvl_mode && target!=GL_TEXTURE_2D) || !color){ p_cis(src,GL_TEXTURE_2D,0,0,0,0,dst,target,level,0,0,target==GL_TEXTURE_2D? 0 : layer,w,h,1); return; }
  lvl_blit(src,GL_TEXTURE_2D,0,0,w,h,dst,target,level,layer,w,h,GL_NEAREST); }

/* fill levels from+1..levels-1 from level 'from' */
static void lvl_gen(GLuint tex, GLenum target, int from, int levels, u32 layers, u32 W0, u32 H0){
  if(!g_mipgen_manual){
    glBindTexture(target,tex); glTexParameteri(target,GL_TEXTURE_BASE_LEVEL,from); glTexParameteri(target,GL_TEXTURE_MAX_LEVEL,levels-1);
    glGenerateMipmap(target); glTexParameteri(target,GL_TEXTURE_BASE_LEVEL,0); return; }
  u32 nl= target==GL_TEXTURE_CUBE_MAP? 6 : target==GL_TEXTURE_2D? 1 : layers;
  for(int l=from;l+1<levels;l++){ GLsizei sw=(GLsizei)(W0>>l), sh=(GLsizei)(H0>>l), dw=(GLsizei)(W0>>(l+1)), dh=(GLsizei)(H0>>(l+1));
    if(sw<1) sw=1; if(sh<1) sh=1; if(dw<1) dw=1; if(dh<1) dh=1;
    for(u32 y=0;y<nl;y++) lvl_blit(tex,target,l,(int)y,sw,sh,tex,target,l+1,(int)y,dw,dh,GL_LINEAR); }
  glBindTexture(target,tex); glTexParameteri(target,GL_TEXTURE_MAX_LEVEL,levels-1); }

/* cubemaps / arrays drawn face by face: every face+mip is its own 2D RT at gva + layer*layer_size + mip_offset. we
   put them together in a real cube texture (GPU copies, only when a face changed) so cube samplers get something */
static u64 st_lay_copy; static int g_fb_sig_ok; static void mb_reset(void);
/* 3D texture rendered slice by slice. in 3D block linear the slices of a block interleave per GOB: base + (z %
   block_depth) * (block_height * 512) + (z / block_depth) * block_layer_size. they get gathered into a 3D texture,
   only the slices that changed get copied */
static u64 st_vol_copy;

static u64 slice3d_addr(u64 gva, u32 w, u32 h, u32 d, const GlFmt* f, u32 bh, u32 bd, u32 z){
  u32 tw=(w+f->bw-1)/f->bw, th=(h+f->bh-1)/f->bh; u32 rowb=tw*f->bpp; u32 gx=(rowb+63)/64, gy=(th+7)/8;
  u32 b=bh; while(b>0 && gy<=(1u<<(b-1))) b--;
  u32 bdd=bd; while(bdd>0 && d<=(1u<<(bdd-1))) bdd--;
  u64 gys=((u64)(gy+(1u<<b)-1)>>b)<<b;
  return gva + (u64)(z>>bdd)*gx*gys*(1ull<<bdd)*512ull + ((u64)(z&((1u<<bdd)-1))<<b)*512ull; }

static Tex* vol_from_slices(Tex* rt0, u64 gva, u32 w, u32 h, u32 d, GlFmt f, u32 bh, u32 bd){
  if(!p_cis || d<2 || d>256 || !rt0->bl) return NULL;
  Tex* sl[256]; u32 sig=2166136261u;
  static int dbg=-1; if(dbg<0) dbg=getenv("SWR_LAYDBG")!=0;
  for(u32 z=0;z<d;z++){ u64 sa=slice3d_addr(gva,w,h,d,&f,bh,bd,z); Tex* r=tex_find_rt(sa);
    if(!r || r->target!=GL_TEXTURE_2D || r->w!=w || r->h!=h || r->f.ifmt!=rt0->f.ifmt){
      if(dbg){ static int n; if(n<sw2_cap(20)){ n++; LOG("voldbg %llx %ux%ux%u bh=%u bd=%u z=%u at %llx: %s\n",(unsigned long long)gva,w,h,d,bh,bd,z,(unsigned long long)sa,r? "different rt" : "no rt"); } }
      return NULL; }
    sl[z]=r; sig=(sig^(r->wgen+z*31))*16777619u; sig=(sig^(u32)(uintptr_t)r)*16777619u; }
  Tex* c=rt0->lay;
  if(c && (c->w!=w || c->h!=h || c->d!=d || c->target!=GL_TEXTURE_3D || c->fkey!=rt0->fkey || c->pitch!=TD(rt0,w))){
    glDeleteTextures(1,&c->id); if(c->view) glDeleteTextures(1,&c->view); free(c); c=rt0->lay=NULL; g_fb_sig_ok=0; g_tmap_gen++; }
  if(!c){ g_tmap_gen++; c=calloc(1,sizeof *c); c->gva=gva; c->w=w; c->h=h; c->d=d; c->levels=1; c->target=GL_TEXTURE_3D; c->f=rt0->f; c->fkey=rt0->fkey; c->is_rt=1; c->scale=TS(rt0); c->bl=1; c->bh=bh; c->pitch=TD(rt0,w);
    c->swz[0]=GL_RED; c->swz[1]=GL_GREEN; c->swz[2]=GL_BLUE; c->swz[3]=GL_ALPHA;
    glGenTextures(1,&c->id); glBindTexture(GL_TEXTURE_3D,c->id); glTexStorage3D(GL_TEXTURE_3D,1,rt0->f.ifmt,TD(rt0,w),TD(rt0,h),d);
    glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MAX_LEVEL,0); c->crop_gen=~0u; rt0->lay=c; mb_reset();
    static int said; if(said<sw2_cap(12)){ said++; LOG("gpu: 3D texture %ux%ux%u (%llx) assembled from its %u GPU-rendered slices\n",w,h,d,(unsigned long long)gva,d); } }
  if(c->crop_gen!=sig){
    for(u32 z=0;z<d;z++) p_cis(sl[z]->id,GL_TEXTURE_2D,0,0,0,0,c->id,GL_TEXTURE_3D,0,0,0,(GLint)z,(GLsizei)TD(rt0,w),(GLsizei)TD(rt0,h),1);
    c->crop_gen=sig; st_vol_copy++; }
  c->last_used=g_frame; return c; }

static int getenv_cached_nomip(void){ static int v=-1; if(v<0) v=getenv("SWR_NO_RT_MIPS")!=NULL; return v; }

/* same width, or padded up to the next GOB (64 bytes) */
static int rt_w_fits(Tex* r, u32 w){ u32 g= r->f.bpp? 64u/r->f.bpp : 16u; if(!g) g=1; return r->w==w || (r->w>w && (r->w+g-1)/g==(w+g-1)/g); }

static Tex* layered_from_rts(Tex* rt0, u64 gva, u32 w, u32 h, u32 d, u32 levels, GLenum target, GlFmt f, u32 fkey, u32 bl, u32 bh, u32 bd){
  if(!p_cis || levels>15 || d<1) return NULL;
  { u32 ml=max_levels(TD(rt0,w),TD(rt0,h),1); if(levels>ml) levels=ml; }   /* every level must exist at the scaled size too */
  u64 loff[16], lsz=0; for(u32 l=0;l<levels;l++){ loff[l]=lsz; lsz+= bl? bl_level_size(w,h,1,&f,bh,bd,l,NULL) : 0; }
  if(!bl) return NULL;
  if(d>1){ u64 al=512ull<<bh; lsz=(lsz+al-1)/al*al; }
  Tex* c=rt0->lay;
  if(c && (c->w!=w || c->h!=h || c->d!=d || c->levels!=levels || c->target!=target || c->fkey!=rt0->fkey)){
    glDeleteTextures(1,&c->id); if(c->view) glDeleteTextures(1,&c->view); free(c->seen); free(c); c=rt0->lay=NULL; g_fb_sig_ok=0; g_tmap_gen++; }
  u32 W0=TD(rt0,w), H0=TD(rt0,h);
  if(!c){ g_tmap_gen++; c=calloc(1,sizeof *c); c->gva=gva; c->w=w; c->h=h; c->d=d; c->levels=levels; c->target=target; c->f=rt0->f; c->fkey=rt0->fkey; c->is_rt=1; c->scale=TS(rt0); c->bl=1; c->bh=bh;
    c->swz[0]=GL_RED; c->swz[1]=GL_GREEN; c->swz[2]=GL_BLUE; c->swz[3]=GL_ALPHA;
    glGenTextures(1,&c->id); glBindTexture(target,c->id);
    { u32 ml=max_levels(W0,H0,1); GLsizei sl=(GLsizei)(levels<ml? levels : ml);   /* the scaled size can allow fewer */
      if(target==GL_TEXTURE_CUBE_MAP || target==GL_TEXTURE_2D) glTexStorage2D(target,sl,rt0->f.ifmt,W0,H0); else glTexStorage3D(target,sl,rt0->f.ifmt,W0,H0,d); }
    glTexParameteri(target,GL_TEXTURE_MAX_LEVEL,0); c->crop_gen=~0u; c->stable=0; rt0->lay=c;
    /* levels start at 0, so one the copy misses reads black and not whatever that GPU memory had. NaNs left in there
       turned SMO's tonemap black */
    { static PFNGLCLEARTEXIMAGEPROC p_ct; static int got; if(!got){ got=1; p_ct=(PFNGLCLEARTEXIMAGEPROC)backend_gl_proc("glClearTexImage"); }
      if(p_ct && !rt0->f.depth && !rt0->f.is_int){ u32 ml=max_levels(W0,H0,1); u32 nl= levels<ml? levels : ml; for(u32 l=0;l<nl;l++) p_ct(c->id,(GLint)l,GL_RGBA,GL_FLOAT,NULL); } }
    mb_reset();
    static int said; if(said<sw2_cap(6)){ said++; LOG("gpu: %s %ux%u (%u layers, %u mips) assembled from its render targets (%llx)\n",target==GL_TEXTURE_CUBE_MAP? "cubemap":target==GL_TEXTURE_2D? "mip chain":"array",w,h,d,levels,(unsigned long long)gva); } }
  /* signature: which RTs exist and their write counters */
  u32 sig=2166136261u; int maxl=-1;
  /* arrays and cube arrays can get filled a piece at a time: wonder renders the first cube of its sky cube array
     and builds its mips (sampling the array) before the second cube exists, so we use the layers that are there
     instead of dropping the whole texture */
  u8 have[256]; int part= (target==GL_TEXTURE_2D_ARRAY || target==GL_TEXTURE_CUBE_MAP_ARRAY) && d<=256;
  for(u32 y=0;y<d && y<256;y++) have[y]=1;
  if(part) for(u32 y=1;y<d;y++){ Tex* r=tex_find_rt(gva+(u64)y*lsz); have[y]= r && r->target==GL_TEXTURE_2D && rt_w_fits(r,w) && r->h==h && r->f.ifmt==rt0->f.ifmt; }
  for(u32 l=0;l<levels;l++){ int all=1;
    for(u32 y=0;y<d;y++){ if(part && !have[y]) continue;
      Tex* r=tex_find_rt(gva+(u64)y*lsz+loff[l]); u32 lw=w>>l, lh=h>>l; if(!lw) lw=1; if(!lh) lh=1;
      /* tiny mips get rendered into RTs padded to a GOB row (64 bytes), like 8x4 / 8x2 for the 4x4 / 2x2 mips of
         an RGBA16F chain */
      u32 gobpx= rt0->f.bpp? 64u/rt0->f.bpp : 16u; if(!gobpx) gobpx=1;
      if(r && r->target==GL_TEXTURE_2D && (rt_w_fits(r,lw) || (l>0 && r->w>=lw && (r->w<lw*2 || r->w<=gobpx))) && (r->h==lh || (l>0 && r->h>=lh && r->h<lh*2)) && r->f.ifmt==rt0->f.ifmt){ sig=(sig^(r->wgen+y*31+l*977))*16777619u; sig=(sig^(u32)(uintptr_t)r)*16777619u; } else all=0; }
    if(!all) break; maxl=(int)l; }
  static int lay_last=-2;
  if(getenv("SWR_LAYDBG") && (!strtoull(getenv("SWR_LAYDBG"),NULL,16) || strtoull(getenv("SWR_LAYDBG"),NULL,16)==gva) && maxl!=lay_last){ lay_last=maxl; static int n; if(n<sw2_cap(12)){ n++;   /* =<hex address>: that texture only, each time the levels found change */ LOG("laydbg gva=%llx lsz=%llx bh=%u levels=%u maxl=%d\n",(unsigned long long)gva,(unsigned long long)lsz,bh,levels,maxl);
      for(u32 l=0;l<levels;l++) for(u32 y=0;y<d;y++){ u64 a=gva+(u64)y*lsz+loff[l]; Tex* r=tex_find_rt(a); LOG("  l%u c%u %llx -> %s %ux%u\n",l,y,(unsigned long long)a,r?"rt":"-",r?r->w:0,r?r->h:0); }
      for(int k=0;k<TEXH;k++) for(Tex* r=texh[k];r;r=r->next) if(r->is_rt && r->w<=64 && r->h<=64) LOG("  RT %llx %ux%u\n",(unsigned long long)r->gva,r->w,r->h); } }
  if(maxl<0) return NULL;   /* missing faces -> normal path */
  if(c->crop_gen!=sig){
    int col=!rt0->f.depth && !rt0->f.is_int;
    /* only the layers/levels whose RT changed get copied. tomodachi rebuilds one face of a 72 layer cube array a
       level at a time and samples the array for every next level, copying all 576 pieces each time was ~10 ms per
       pass and 200 ms a frame on an intel iGPU */
    u32 ns=levels*d; if(c->nseen!=ns){ free(c->seen); c->seen=calloc(ns,sizeof(u64)); c->nseen= c->seen? ns : 0; }
    for(int pass=0;pass<2;pass++){
      if(pass && c->seen) memset(c->seen,0,sizeof(u64)*c->nseen);   /* copies switched to blits: redo all of it */
      for(int l=0;l<=maxl;l++) for(u32 y=0;y<d;y++){ if(part && !have[y]) continue; Tex* r=tex_find_rt(gva+(u64)y*lsz+loff[l]);
        if(c->seen){ u64 key=((u64)r->wgen<<40)^(u64)(uintptr_t)r^1; u64* sl=&c->seen[(u32)l*d+y]; if(*sl==key) continue; *sl=key; }
        u32 cw=W0>>l, ch=H0>>l; if(!cw) cw=1; if(!ch) ch=1;
        if(TS(r)!=TS(c)){
          u32 lw=w>>l, lh=h>>l; if(!lw) lw=1; if(!lh) lh=1;
          lvl_blit(r->id,GL_TEXTURE_2D,0,0,(GLsizei)TD(r,lw),(GLsizei)TD(r,lh),
                   c->id,target,l,(int)y,(GLsizei)cw,(GLsizei)ch,col? GL_LINEAR : GL_NEAREST);
        } else {
          u32 rw=TW(r), rh=TH(r); if(rw<cw) cw=rw; if(rh<ch) ch=rh;
          lvl_copy(r->id,c->id,target,l,(int)y,(GLsizei)cw,(GLsizei)ch,col);
        } }
      /* check the first few chains: the center of level 1 has to match its source RT. if the driver dropped the
         copy we switch to blits for good */
      static int nchk; if(pass || g_lvl_mode || !col || maxl<1 || nchk>=6) break;
      { Tex* r1=tex_find_rt(gva+loff[1]); if(r1 && TS(r1)!=TS(c)) break; } nchk++;
      Tex* r1=tex_find_rt(gva+loff[1]); u32 cw=W0>>1, ch=H0>>1; u32 rw=TW(r1), rh=TH(r1); if(rw<cw) cw=rw; if(rh<ch) ch=rh; if(!cw || !ch) break;
      float a4[4]={0}, b4[4]={0}; GLint px=(GLint)cw/2, py=(GLint)ch/2;
      lvl_read1(r1->id,GL_TEXTURE_2D,0,0,px,py,a4); lvl_read1(c->id,target,1,0,px,py,b4);
      int same=1; for(int k=0;k<4;k++){ float e=a4[k]-b4[k]; if(e<0) e=-e; float m=a4[k]<0? -a4[k] : a4[k]; if(e>0.02f+0.02f*m) same=0; }
      if(same) break;
      g_lvl_mode=1; LOG("gpu: mip level copy check failed in game (%.3f %.3f %.3f -> %.3f %.3f %.3f): copies switched to blits\n",a4[0],a4[1],a4[2],b4[0],b4[1],b4[2]); }
    /* mips the game didn't render itself get generated from the last real one. SMO samples its env cubemaps at LOD
       5 (diffuse) and roughness*5 (spec, RGBM), with only level 0 those get the sharp image. a box chain isn't the
       game's prefilter but it's way closer than nothing. cube arrays too: frontiers' sky cubemap is one, and its
       lighting reads the upper levels */
    int topl=maxl; static int nogen=-1; if(nogen<0) nogen=getenv("SWR_NO_CUBE_MIPGEN")!=NULL;
    if(maxl<(int)levels-1 && !nogen && !rt0->f.is_int && !rt0->f.depth && (target==GL_TEXTURE_CUBE_MAP || target==GL_TEXTURE_CUBE_MAP_ARRAY || target==GL_TEXTURE_2D_ARRAY || target==GL_TEXTURE_2D)){
      lvl_gen(c->id,target,maxl,(int)levels,d,W0,H0); topl=(int)levels-1; mb_reset();
      if(c->seen) for(u32 l=(u32)maxl+1;l<levels;l++) memset(c->seen+l*d,0,sizeof(u64)*d);   /* generated now, a real RT there later has to be copied again */
      static int said; if(said<sw2_cap(8)){ said++; LOG("gpu: %s %ux%u at %llx: %d of %u mips rendered by the game, the rest generated\n",target==GL_TEXTURE_CUBE_MAP? "cubemap":target==GL_TEXTURE_2D? "mip chain":"array",w,h,(unsigned long long)gva,maxl+1,levels); } }
    if((int)c->stable!=topl){ glBindTexture(target,c->id); glTexParameteri(target,GL_TEXTURE_MAX_LEVEL,topl); c->stable=(u32)topl; mb_reset(); }
    c->crop_gen=sig; st_lay_copy++; }
  c->last_used=g_frame; rt0->last_used=g_frame; return c; }

/* driver checks at startup for the stuff the post processing needs. results go to the log as "gpu: driver check",
   every test writes a known color and reads it back */
static GLuint st_prog, st_samp, st_out, st_ofbo;

static int st_near(const float* a, const float* b, int n){ for(int i=0;i<n;i++){ float d=a[i]-b[i]; if(d<0) d=-d; if(d>0.03f || a[i]!=a[i]) return 0; } return 1; }

static void st_clear_level(GLuint tex, int level, const float* c){
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER,g_dfbo); glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tex,level);
  { GLenum b=GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&b); } glClearBufferfv(GL_COLOR,0,c);
  glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,0,0); }

/* draw one pixel sampling tex at a fixed lod (min=max=lod, like the bloom passes) and return it */
static void st_sample(GLuint tex, float lod, float* out){
  glSamplerParameterf(st_samp,GL_TEXTURE_MIN_LOD,lod); glSamplerParameterf(st_samp,GL_TEXTURE_MAX_LOD,lod);
  glBindFramebuffer(GL_FRAMEBUFFER,st_ofbo); { GLenum b=GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&b); } glViewport(0,0,1,1);
  glUseProgram(st_prog); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,tex); glBindSampler(0,st_samp);
  glBindVertexArray(g_scratch_vao); glDrawArrays(GL_TRIANGLES,0,3);
  glReadBuffer(GL_COLOR_ATTACHMENT0); float px[4]={-9,-9,-9,-9}; glReadPixels(0,0,1,1,GL_RGBA,GL_FLOAT,px); memcpy(out,px,16);
  glBindSampler(0,0); glUseProgram(0); }

/* mip 1 filled from a separate RT, then sampled at lod 1 */
static int st_mip_copy(GLenum ifmt, int mode, float* got){
  static const float c0[4]={0.1f,0.1f,0.1f,1}, c1[4]={0.5f,0.25f,0.75f,1};
  GLuint dst, src; glGenTextures(1,&dst); glBindTexture(GL_TEXTURE_2D,dst); glTexStorage2D(GL_TEXTURE_2D,3,ifmt,8,8); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,0);
  glGenTextures(1,&src); glBindTexture(GL_TEXTURE_2D,src); glTexStorage2D(GL_TEXTURE_2D,1,ifmt,4,4); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,0);
  st_clear_level(dst,0,c0); st_clear_level(dst,1,c0); st_clear_level(src,0,c1);
  int om=g_lvl_mode; g_lvl_mode=mode; lvl_copy(src,dst,GL_TEXTURE_2D,1,0,4,4,1); g_lvl_mode=om;
  glBindTexture(GL_TEXTURE_2D,dst); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,1);
  st_sample(dst,1.0f,got); glDeleteTextures(1,&dst); glDeleteTextures(1,&src);
  return st_near(got,c1,3); }

/* draw level 1, generate 2.., sample level 2 */
static int st_mip_gen(GLenum ifmt, float* got){
  static const float c0[4]={0.1f,0.1f,0.1f,1}, c1[4]={0.5f,0.25f,0.75f,1};
  GLuint t; glGenTextures(1,&t); glBindTexture(GL_TEXTURE_2D,t); glTexStorage2D(GL_TEXTURE_2D,3,ifmt,8,8); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,0);
  st_clear_level(t,0,c0); st_clear_level(t,1,c1); st_clear_level(t,2,c0);
  lvl_gen(t,GL_TEXTURE_2D,1,3,1,8,8); glBindTexture(GL_TEXTURE_2D,t); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,2);
  st_sample(t,2.0f,got); glDeleteTextures(1,&t);
  return st_near(got,c1,3); }

/* shader writes negative values to an SNORM target */
static int st_snorm(GLenum ifmt, int n, float* got){
  static const char* fs="#version 430 core\nout vec4 o; void main(){ o=vec4(-0.5,0.25,-1.0,0.75); }";
  static const char* vs="#version 430 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID&1)*4-1),float((gl_VertexID>>1)*4-1)); gl_Position=vec4(p,0.0,1.0); }";
  static GLuint pr; if(!pr){ GLuint a=compile(GL_VERTEX_SHADER,vs,NULL,0), b=compile(GL_FRAGMENT_SHADER,fs,NULL,0); if(a&&b) pr=link2(a,b,0); if(!pr) return 1; }
  GLuint t, fb; glGenTextures(1,&t); glBindTexture(GL_TEXTURE_2D,t); glTexStorage2D(GL_TEXTURE_2D,1,ifmt,1,1);
  glGenFramebuffers(1,&fb); glBindFramebuffer(GL_FRAMEBUFFER,fb); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,t,0);
  { GLenum b=GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&b); }
  int ok=0; got[0]=got[1]=got[2]=got[3]=-9;
  if(glCheckFramebufferStatus(GL_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE){
    glViewport(0,0,1,1); glUseProgram(pr); glBindVertexArray(g_scratch_vao); glDrawArrays(GL_TRIANGLES,0,3);
    /* read back with a shader, glReadPixels can clamp fixed point to 0..1 */
    glUseProgram(0); glBindFramebuffer(GL_FRAMEBUFFER,0); st_sample(t,0.0f,got);
    static const float want[4]={-0.5f,0.25f,-1.0f,0.75f}; ok=st_near(got,want,n); }
  glUseProgram(0); glBindFramebuffer(GL_FRAMEBUFFER,0); glDeleteFramebuffers(1,&fb); glDeleteTextures(1,&t);
  return ok; }

int g_texoff_emul;

static void gl_selftest(void){
  /* non-constant texture offsets: nvidia takes them, the spec / AMD / mesa don't */
  { static const char* fs="#version 430 core\nlayout(binding=0) uniform sampler2D t; uniform int k; out vec4 o; void main(){ o=textureOffset(t,vec2(0.5),ivec2(bitfieldExtract(k,0,4),bitfieldExtract(k,4,4))); }";
    GLuint sh=compile(GL_FRAGMENT_SHADER,fs,NULL,0); g_texoff_emul= sh? 0 : 1; if(sh) glDeleteShader(sh);
    if(getenv("SWR_TEXOFF_EMUL")) g_texoff_emul=1;
    LOG("gpu: driver check: texture offsets from a register %s\n",g_texoff_emul? "refused by the driver: added to the coordinates instead" : "accepted"); }
  if(!p_cis || getenv("SWR_NO_DRIVER_CHECK")) return;
  static const char* vs="#version 430 core\nvoid main(){ vec2 p=vec2(float((gl_VertexID&1)*4-1),float((gl_VertexID>>1)*4-1)); gl_Position=vec4(p,0.0,1.0); }";
  static const char* fs="#version 430 core\nlayout(binding=0) uniform sampler2D t; out vec4 o; void main(){ o=texture(t,vec2(0.5)); }";
  GLuint a=compile(GL_VERTEX_SHADER,vs,NULL,0), b=compile(GL_FRAGMENT_SHADER,fs,NULL,0); st_prog= a&&b? link2(a,b,0) : 0;
  if(!st_prog){ LOG("gpu: driver check skipped (test shader did not compile)\n"); return; }
  glDisable(GL_SCISSOR_TEST); glDisable(GL_FRAMEBUFFER_SRGB); glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST); glDisable(GL_STENCIL_TEST); glDisable(GL_CULL_FACE); glColorMaski(0,1,1,1,1);
  glGenSamplers(1,&st_samp); glSamplerParameteri(st_samp,GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_LINEAR); glSamplerParameteri(st_samp,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  glSamplerParameteri(st_samp,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); glSamplerParameteri(st_samp,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
  glGenTextures(1,&st_out); glBindTexture(GL_TEXTURE_2D,st_out); glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA32F,1,1);
  glGenFramebuffers(1,&st_ofbo); glBindFramebuffer(GL_FRAMEBUFFER,st_ofbo); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,st_out,0);
  static const GLenum fmts[2]={GL_R11F_G11F_B10F,GL_RGBA16F}; static const char* fnm[2]={"R11G11B10F","RGBA16F"};
  float g[4]; int force=-1; { const char* e=getenv("SWR_LEVEL_COPY"); if(e) force=atoi(e); }
  /* copies into mip levels */
  int okm[2]={1,1};
  for(int m=0;m<2;m++) for(int k=0;k<2;k++){ int ok=st_mip_copy(fmts[k],m,g); okm[m]&=ok;
    LOG("gpu: driver check: mip level copy (%s, %s): %.2f %.2f %.2f -> %s\n",m? "blit" : "CopyImageSubData",fnm[k],g[0],g[1],g[2],ok? "ok" : "WRONG"); }
  g_lvl_mode= force>=0? (force? 1 : 0) : okm[0]? 0 : okm[1]? 1 : 0;
  /* mips generated from a level above 0 */
  int okg=1; for(int k=0;k<2;k++){ int ok=st_mip_gen(fmts[k],g); okg&=ok; LOG("gpu: driver check: mip generation from level 1 (%s): %.2f %.2f %.2f -> %s\n",fnm[k],g[0],g[1],g[2],ok? "ok" : "WRONG"); }
  if(!okg){ g_mipgen_manual=1; int okg2=1; for(int k=0;k<2;k++){ int ok=st_mip_gen(fmts[k],g); okg2&=ok; LOG("gpu: driver check: mip generation with blits (%s): %.2f %.2f %.2f -> %s\n",fnm[k],g[0],g[1],g[2],ok? "ok" : "WRONG"); }
    if(!okg2) g_mipgen_manual=0; }
  if(getenv("SWR_MIPGEN_BLIT")) g_mipgen_manual=1;
  /* SNORM render targets */
  int oks=st_snorm(GL_RGBA8_SNORM,4,g); LOG("gpu: driver check: SNORM render target: %.2f %.2f %.2f %.2f -> %s\n",g[0],g[1],g[2],g[3],oks? "ok" : "WRONG");
  if(!oks){ int okf=st_snorm(GL_RGBA16F,4,g); g_snorm_fb=okf; LOG("gpu: driver check: half float instead: %.2f %.2f %.2f %.2f -> %s\n",g[0],g[1],g[2],g[3],okf? "ok" : "WRONG"); }
  if(getenv("SWR_SNORM_FLOAT")) g_snorm_fb=1;
  LOG("gpu: driver check: mip level copies by %s, mip generation %s, SNORM targets %s\n",g_lvl_mode? "blit" : "CopyImageSubData",g_mipgen_manual? "with blits" : "by the driver",g_snorm_fb? "stored as half float" : "native");
  glDeleteFramebuffers(1,&st_ofbo); glDeleteTextures(1,&st_out); glDeleteProgram(st_prog);
  glBindFramebuffer(GL_FRAMEBUFFER,0); glBindVertexArray(0); glBindTexture(GL_TEXTURE_2D,0); glUseProgram(0);
  while(glGetError()!=GL_NO_ERROR){} }

/* texture for a TIC */
static Tex* tex_from_tic_(u32* tic);
static struct { u32 gen, frame; u32 tic[8]; Tex* t; } g_tdc[512];

extern u64 g_draws;
static u64 st_tdc_hit, st_tdc_miss;

/* one memory, two views: MKW 1.0 fills its 64x36x256 volumetric fog grid with compute image stores to a
   2D array of 256 layers and samples it as a 3D texture. Here those are two GL textures, so the 3D one only had what was
   in guest memory (-800, inf), the fog pass made NaN of it and the 3D scenes came out white. A 3D texture about to be
   used gets the layers of a GPU written 2D array at its address (same size and format) whenever that array changed */
static void sync3d_from_array(Tex* t){
  if(!t || t->target!=GL_TEXTURE_3D) return;
  for(Tex* a=texh[hkey(t->gva)];a;a=a->next){
    if(a==t || a->gva!=t->gva || a->target!=GL_TEXTURE_2D_ARRAY || !a->gpu_written || a->w!=t->w || a->h!=t->h || a->d!=t->d || a->f.ifmt!=t->f.ifmt) continue;
    /* the image stores must have landed before the copy reads them (an F5 trace syncs enough to hide that) */
    if(a->wgen!=t->a3d_wgen){ glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT|GL_SHADER_IMAGE_ACCESS_BARRIER_BIT); glCopyImageSubData(a->id,GL_TEXTURE_2D_ARRAY,0,0,0,0,t->id,GL_TEXTURE_3D,0,0,0,0,(GLsizei)a->w,(GLsizei)a->h,(GLsizei)a->d);
      t->a3d_wgen=a->wgen; t->gpu_written=1;
      static int said; if(said<4){ said++; LOG("gpu: 3D texture %llx %ux%ux%u takes the layers of the 2D array compute wrote there\n",(unsigned long long)t->gva,t->w,t->h,t->d); } }
    return; } }

static Tex* tex_from_tic(u32* tic){
  u32 k=(tic[1]^(tic[1]>>9)^tic[0]^(tic[2]<<7)^tic[4]^(tic[5]<<3))&511;
  static int off=-1; if(off<0) off=getenv("SWR_NO_TEX_MEMO")!=NULL;
  if(!off && g_tdc[k].gen==g_tmap_gen && g_tdc[k].frame==(u32)g_frame && !memcmp(g_tdc[k].tic,tic,32)){ st_tdc_hit++; sync3d_from_array(g_tdc[k].t); return g_tdc[k].t; }
  Tex* t=tex_from_tic_(tic); st_tdc_miss++; sync3d_from_array(t);
  /* tex_from_tic_ can bump the generation itself, so cache it under the new one */
  g_tdc[k].gen=g_tmap_gen; g_tdc[k].frame=(u32)g_frame; memcpy(g_tdc[k].tic,tic,32); g_tdc[k].t=t; return t; }

static Tex* tex_from_tic_(u32* tic){
  u64 gva=((u64)(tic[2]&0xFFFF)<<32)|tic[1];
  u32 hv=(tic[2]>>21)&7; int bl= hv==3||hv==4;
  u32 w=(tic[4]&0xFFFF)+1, h=(tic[5]&0xFFFF)+1, d=((tic[5]>>16)&0x3FFF)+1; u32 type=(tic[4]>>23)&0xF;
  int srgb=(tic[4]>>22)&1; u32 levels=((tic[3]>>28)&0xF)+1;
  u32 bh=(tic[3]>>3)&7, bd=(tic[3]>>6)&7, pitch=bl? 0 : ((tic[3]&0xFFFF)<<5);
  if(type==0 || type==4) h=1;
  if(type==3){ d=6; } else if(type==8){ d=d*6; }
  /* the TIC's max mip level can be more than the size allows. tomodachi's mii face and skin RTs (384x384, 128x256)
     say 15 levels, the mip chain built from them couldn't be allocated and the miis had black faces */
  { u32 ml=max_levels(w,h,type==2? d : 1); if(levels>ml) levels=ml; }
  GlFmt f=tic_format(tic[0],srgb);
  /* garbage TIC (bad format, null address, silly size): return null instead of uploading hundreds of MB of junk */
  if(!f.bpp || !gva || w>16384 || h>16384 || d>2048 || levels>15){ return NULL; }
  /* a pitch shorter than a row (header version 0 texture buffers keep the high half of the width there) would make
     the upload copy rows past what it read */
  if(!bl){ u32 rowb=((w+f.bw-1)/f.bw)*f.bpp; if(pitch<rowb) pitch=rowb; }
  { u64 est=(u64)w*h*(type==2? d : (d>0? d : 1))*(f.compressed? 1 : f.bpp); if(est>(512ull<<20) && (type==2 || levels<4)) return NULL; }   /* bigger ones with mips: virtual textures, kept without their top mips below */
  u32 fkey=(tic[0]&0x7FFFF)|(srgb<<31);
  /* there's an RT at that address, use it */
  Tex* rt=tex_find_rt(gva);
  /* an RT the GPU wrote isn't dropped because guest memory differs: that memory is stale until the next flush.
     smo's 1600x900 background RT got thrown away every frame that way. SWR_NO_RT_TRUST=1 checks it again */
  static int rt_trust=-1; if(rt_trust<0) rt_trust=getenv("SWR_NO_RT_TRUST")==NULL;
  if(rt && rt->frame_checked!=g_frame){ rt->frame_checked=g_frame; if(!(rt_trust && rt->gpu_written) && rt_sig(gva)!=rt->hash){
      static int said; if(said<sw2_cap(8)){ said++; LOG("gpu: rt %llx %ux%u (format %x) dropped, its memory changed; a %ux%u texture (format %x) reads that memory now\n",(unsigned long long)gva,rt->w,rt->h,rt->fkey&0xFFFF,w,h,fkey&0x7FFFF); }
      rt->gva=~0ull; rt=NULL; g_tmap_gen++; } }
  if(rt && rt->vol && rt->w==w && rt->h==h && d<=rt->vol->d && ((type==2 && rt->vol->target==GL_TEXTURE_3D) || ((type==5 || type==1) && rt->vol->target==GL_TEXTURE_2D_ARRAY))){
    rt->vol->last_used=g_frame; return rt->vol; }
  /* 3D TIC over a 2D RT: build it from slices if they're all RTs, if not the data is in memory */
  if(rt && type==2 && d>1 && !f.compressed && !f.astc){ Tex* v=vol_from_slices(rt,gva,w,h,d,f,bh,bd); if(v) return v; rt=NULL; }
  else if(rt && type==2 && d>1) rt=NULL;
  /* the layers can be RTs a bit wider than the TIC because the game rounds RT widths up to a whole GOB (totk's
     shadow cascades: 736 wide RTs read through a 720 wide array) */
  if(rt && !f.compressed && !f.astc && rt->target==GL_TEXTURE_2D && rt_w_fits(rt,w) && rt->h==h && (type==3 || type==8 || (type==5 && d>1))){
    GLenum lt= type==3? GL_TEXTURE_CUBE_MAP : type==8? GL_TEXTURE_CUBE_MAP_ARRAY : GL_TEXTURE_2D_ARRAY;
    Tex* lc=layered_from_rts(rt,gva,w,h,d,levels,lt,f,fkey,bl,bh,bd); if(lc) return lc; }
  /* mipped 2D TIC over an RT: the game renders each mip to its own RT and picks one with the sampler LOD clamp
     (SMO bloom: 160x180, 5 mips, read at lod 1..4). if we only return level 0 every blur pass reads the sharp
     image */
  if(rt && type==1 && levels>1 && d==1 && !f.compressed && !f.astc && rt->target==GL_TEXTURE_2D && rt->w==w && rt->h==h && !rt->f.depth && !getenv_cached_nomip()){
    Tex* lc=layered_from_rts(rt,gva,w,h,1,levels,GL_TEXTURE_2D,f,fkey,bl,bh,bd); if(lc) return lc; }
  if(rt && !f.compressed && !f.astc && rt->w>=w && rt->h>=h){ rt->last_used=g_frame; return (rt->w==w && rt->h==h)? rt : rt_crop(rt,w,h); }   /* smaller view -> just that region */
  GLenum target=tex_target(type,d);
  /* 3D textures in packed R11G11B10F (color grading LUTs) go up as RGBA16F, decoded on the cpu. SMO's tonemap
     reads an 8x8x8 LUT in that format and on AMD iGPUs (ryzen 5 4600G) the packed 3D upload came back as zeros,
     black screen */
  if(target==GL_TEXTURE_3D && f.ifmt==GL_R11F_G11F_B10F && !f.compressed && !f.dec && !getenv_cached_nolut16()){
    f.ifmt=GL_RGBA16F; f.fmt=GL_RGBA; f.type=GL_HALF_FLOAT; f.dec=20; }
  if(target==GL_TEXTURE_CUBE_MAP && d!=6) target=GL_TEXTURE_2D_ARRAY;
  if(f.tobc && target==GL_TEXTURE_3D){ f.tobc=0; f.ifmt= f.ifmt==0x8C4F? GL_SRGB8_ALPHA8 : GL_RGBA8; }   /* 3D ASTC stays RGBA8 */
  if(f.astc && !f.tobc && levels>1) levels=1;   /* BC3 keeps the game's own mips */
  for(Tex* t=texh[hkey(gva)];t;t=t->next){
    if(t->gva==gva && ((t->w==w && t->h==h) || (t->huge_w==w && t->huge_h==h)) && t->d==d && t->fkey==fkey && t->target==target && !t->is_rt){
      t->last_used=g_frame;
      if(t->skip){ if((i32)(g_frame-t->vt_next)>=0) vt_refresh(t,0); return t; }
      /* change check: often at first, then less while the texture stays the same, capped in MB per frame */
      if(t->frame_checked!=g_frame && (i32)(g_frame-t->next_check)>=0){
        if(g_tex_bud_frame!=g_frame){ g_tex_bud_frame=g_frame; g_tex_bud=24u<<20; }
        /* stable for a long time: cheap sampled check every 20 frames, sampled hash check every 120 */
        if(g_nosample<0) g_nosample=getenv("SWR_NO_SAMPLE_CHECK")!=NULL;
        if(!g_nosample && t->stable>=40 && t->size>=16384 && (i32)(g_frame-t->tfull)<120){
          u64 sh=sample_hash(gva,t->size); t->frame_checked=g_frame;
          if(sh==t->shash){ t->next_check=g_frame+20+((u32)(gva>>9)&7); return t; }
          t->stable=0; }
        if(g_tex_bud>0 || t->stable<4){ t->frame_checked=g_frame; g_tex_bud-=(i64)(t->size>>(t->size>65536? 1 : 0)); st_tex_chk+=t->size;
          u64 hh=mem_hash(gva,t->size); if(hh!=t->hash){ t->hash=hh; tex_upload(t); t->stable=0; } else if(t->stable<100000) t->stable++;
          if(t->stable>=39 && t->size>=16384){ t->shash=sample_hash(gva,t->size); t->tfull=g_frame; }
          t->next_check=g_frame+(t->stable<4? 4 : t->stable<12? 8 : 20)+((u32)(gva>>9)&3); } }
      return t; } }
  { /* debug: a texture that lives inside an RT is being read from memory */
    static int nd; if(nd<24 && g_verbose>=2){ for(int k=0;k<TEXH && nd<24;k++) for(Tex* q=texh[k];q;q=q->next){ if(!q->is_rt || q->gva==~0ull) continue;
        u64 rsz=(u64)q->w*q->h*(q->f.bpp? q->f.bpp : 4);
        if(gva>=q->gva && gva<q->gva+rsz){ nd++; LOG("gpu: texture %ux%u fmt %x at %llx falls inside render target %llx %ux%u (+%llx)%s: read from memory\n",w,h,tic[0]&0x7F,(unsigned long long)gva,
            (unsigned long long)q->gva,q->w,q->h,(unsigned long long)(gva-q->gva),gva==q->gva? " and is larger than it" : ""); break; } } } }
  /* MKW's virtual textures are 8192x8192 arrays of 68 layers (ASTC: 1.5 GB in guest memory, 18 GB
     decoded): textures over 2^28 texels get a 1x1 black stand-in, no uploads (the game's other textures still show) */
  if((u64)w*h*(target==GL_TEXTURE_3D || target==GL_TEXTURE_2D_ARRAY || target==GL_TEXTURE_CUBE_MAP_ARRAY? d : 1)>(1ull<<28)){
    static int vt_off=-1; static u64 vt_max; if(vt_off<0){ vt_off=getenv("SWR_NO_VT")!=NULL; const char* e=getenv("SWR_VT_TEXELS_LOG2"); vt_max=1ull<<(e? atoi(e) : 26); }
    u32 dd= (target==GL_TEXTURE_3D || target==GL_TEXTURE_2D_ARRAY || target==GL_TEXTURE_CUBE_MAP_ARRAY)? d : 1, k=0;
    while(k+1<levels && (u64)(w>>k)*(h>>k)*dd>vt_max) k++;
    if(vt_off || target==GL_TEXTURE_3D || (u64)(w>>k)*(h>>k)*dd>vt_max){
      Tex* t=tex_create(gva,1,1,d,1,target,f,fkey,bl,bh,bd,pitch,0); t->huge_w=w; t->huge_h=h; t->size=0; t->hash=mem_hash(gva,0); t->stable=100000; t->frame_checked=g_frame; t->last_used=g_frame;
      { static int said; if(said<4){ said++; LOG("gpu: texture %ux%ux%u at %llx too big for the host, a black stand-in is used\n",w,h,d,(unsigned long long)gva); } }
      return t; }
    /* virtual texture (sparse, pages streamed in while you drive): kept without its top k mips, layers re-uploaded when their bytes change */
    Tex* t=tex_create(gva,w>>k,h>>k,d,levels-k,target,f,fkey,bl,bh,bd,pitch,0); t->huge_w=w; t->huge_h=h; t->skip=k; t->stable=100000; t->frame_checked=g_frame; t->last_used=g_frame;
    t->lhash=calloc(d,8); t->lmask=calloc(d,1); t->size=0;
    vt_refresh(t,1);
    { static int said; if(said<8){ said++; LOG("gpu: virtual texture %ux%ux%u at %llx kept at %ux%u (top %u mips left out)\n",w,h,d,(unsigned long long)gva,w>>k,h>>k,k); } }
    return t; }
  Tex* t=tex_create(gva,w,h,d,levels,target,f,fkey,bl,bh,bd,pitch,0);
  u64 sz=0; u32 dd= target==GL_TEXTURE_3D? d : 1; for(u32 l=0;l<levels;l++) sz+=bl? bl_level_size(w,h,dd,&f,bh,bd,l,NULL) : (u64)pitch*((h+f.bh-1)/f.bh);
  if(target!=GL_TEXTURE_3D) sz*=d;
  t->size=sz; t->hash=mem_hash(gva,sz); t->frame_checked=g_frame; t->last_used=g_frame;
  tex_upload(t);
  return t; }

/* ============================================================ samplers */
typedef struct { u32 tsc[8]; GLuint id; } Samp;

static Samp samps[1024]; static int nsamps;

static GLenum wrap(u32 m){ switch(m){ case 0: return GL_REPEAT; case 1: return GL_MIRRORED_REPEAT; case 2: case 4: return GL_CLAMP_TO_EDGE; case 3: case 6: return GL_CLAMP_TO_BORDER;
  case 5: case 7: return GL_MIRROR_CLAMP_TO_EDGE; } return GL_REPEAT; }

static short samp_ht[4096];

static GLuint sampler_get(const u32* tsc){
  u32 hk=2166136261u; for(int i=0;i<8;i++){ hk^=tsc[i]; hk*=16777619u; } hk&=4095;
  for(u32 j=hk;;j=(j+1)&4095){ int e=samp_ht[j]; if(!e) break; if(!memcmp(samps[e-1].tsc,tsc,32)) return samps[e-1].id; }
  GLuint s; glGenSamplers(1,&s);
  glSamplerParameteri(s,GL_TEXTURE_WRAP_S,wrap(tsc[0]&7)); glSamplerParameteri(s,GL_TEXTURE_WRAP_T,wrap((tsc[0]>>3)&7)); glSamplerParameteri(s,GL_TEXTURE_WRAP_R,wrap((tsc[0]>>6)&7));
  u32 mag=tsc[1]&3, min=(tsc[1]>>4)&3, mip=(tsc[1]>>6)&3;
  glSamplerParameteri(s,GL_TEXTURE_MAG_FILTER,mag==2? GL_LINEAR : GL_NEAREST);
  GLenum mf= min==2? (mip==3? GL_LINEAR_MIPMAP_LINEAR : mip==2? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR) : (mip==3? GL_NEAREST_MIPMAP_LINEAR : mip==2? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST);
  glSamplerParameteri(s,GL_TEXTURE_MIN_FILTER,mf);
  if((tsc[0]>>9)&1){ glSamplerParameteri(s,GL_TEXTURE_COMPARE_MODE,GL_COMPARE_REF_TO_TEXTURE); glSamplerParameteri(s,GL_TEXTURE_COMPARE_FUNC,GL_NEVER+((tsc[0]>>10)&7)); }
  u32 aniso=(tsc[0]>>20)&7; if(aniso) glSamplerParameterf(s,0x84FE,(float)(1<<aniso));
  float minl=(float)(tsc[2]&0xFFF)/256.f, maxl=(float)((tsc[2]>>12)&0xFFF)/256.f; glSamplerParameterf(s,GL_TEXTURE_MIN_LOD,minl); glSamplerParameterf(s,GL_TEXTURE_MAX_LOD,maxl);
  i32 bias=(i32)((tsc[1]>>12)&0x1FFF); if(bias&0x1000) bias|=~0x1FFF; glSamplerParameterf(s,GL_TEXTURE_LOD_BIAS,(float)bias/256.f);
  float bc[4]; { u32 b4[4]; memcpy(b4,&tsc[4],16); if(b4[3]&0x80000000u){ b4[3]&=0x7FFFFFFFu; glSamplerParameteri(s,0x8A48,0x8A4A); } memcpy(bc,b4,16); } glSamplerParameterfv(s,GL_TEXTURE_BORDER_COLOR,bc);   /* tsc[7] bit 31 = skip sRGB decode */
  if(nsamps<1024){ memcpy(samps[nsamps].tsc,tsc,32); samps[nsamps].id=s; nsamps++; u32 j=hk; while(samp_ht[j]) j=(j+1)&4095; samp_ht[j]=(short)nsamps; }
  return s; }

/* ============================================================ shaders and programs */
extern int g_glsl_version;
struct WJob;
typedef struct Sh { u64 hash; u32 key2; int stage; GLuint gl; ShaderInfo si; struct Sh* next; int failed; char* src;
  GLuint sp; GLint uloc; int sps; struct WJob* sj; u32 clen; u64 qh; float lu[4]; int lu_ok; u64 skey; } Sh;   /* sp: this shader's own separable program */

static Sh* shh[1024];
/* per address cache. shader_hash walks the whole program (~50us per stage per draw on big shaders), inside one
   submit a quick hash of the same range is enough */
typedef struct { u64 gva, cpu; u32 key2, epoch, chk; int stage; Sh* s; } ShA;
static int g_ww; static int ww_dirty(u64 ga, u64 len, u32 since);

static ShA sha[4096]; static u64 st_sh_fast;
static int g_sh_bind; static u32* g_last_r3d; static u32 g_gs_input_vertices;
static int sh_cb_read(u32 idx, u32 off, u32* out);
static void sh_dump_cbs(u64 h);
static u32 g_fs_rtk, g_fs_rtm, g_fs_rts;   /* FS integer output key */
static void aw_write(const char* path, const void* d, size_t n, int append);
static GLenum shader_gl_type(int stage){
  if(stage==ST_TASK) return 0x955A; /* GL_TASK_SHADER_NV */ if(stage==ST_MESH) return 0x9559; /* GL_MESH_SHADER_NV */
  return stage==ST_VS? GL_VERTEX_SHADER : stage==ST_GS? GL_GEOMETRY_SHADER :
         stage==ST_CS? GL_COMPUTE_SHADER : stage==ST_TCS? GL_TESS_CONTROL_SHADER :
         stage==ST_TES? GL_TESS_EVALUATION_SHADER : GL_FRAGMENT_SHADER;
}

/* common end stuff: alpha test for the FS, Y flip for the VS */
static char* sh_post(char* src, int stage){
  /* alpha test: discard before every return */
  if(stage==ST_FS && src && strstr(src,"swr_at.w") && !strstr(src,"out vec4 frag_color0")){ char* p=strstr(src,"void main(){"); if(p){ size_t pre=p-src;
      char* ns=malloc(strlen(src)+64); memcpy(ns,src,pre); ns[pre]=0; strcat(ns,"uniform vec4 swr_at;\n"); strcat(ns,p); free(src); src=ns; } }
  if(stage==ST_FS && src && strstr(src,"out vec4 frag_color0")){ char* p=strstr(src,"void main(){"); if(p){ size_t pre=p-src;
      const char* fn="uniform vec4 swr_at;\nbool swr_atest(float a){ int f=int(swr_at.y); float r=swr_at.z;\n"
        "  if(f==0) return false; if(f==1) return a<r; if(f==2) return a==r; if(f==3) return a<=r; if(f==4) return a>r; if(f==5) return a!=r; if(f==6) return a>=r; return true; }\n";
      char* out=malloc(strlen(src)*2+strlen(fn)+64); memcpy(out,src,pre); out[pre]=0; strcat(out,fn); char* o=out+strlen(out); const char* q2=p;
      while(*q2){ if(!strncmp(q2,"return;",7)){ o+=sprintf(o,"{ if(swr_at.x!=0.0 && !swr_atest(frag_color0.w)) discard; return; }"); q2+=7; } else *o++=*q2++; }
      *o=0; free(src); src=out; } }
  /* geometry shaders: the viewport flip and depth range go on every emitted vertex, the VS one is skipped then */
  if(stage==ST_GS && src){ char* p=strstr(src,"void main(){"); if(p){ size_t pre=p-src; char* ns=malloc(strlen(src)+512);
      memcpy(ns,src,pre); ns[pre]=0; strcat(ns,"uniform vec4 swr_vp;\n"); strcat(ns,p); free(src); src=ns;
      const char* fx="{ gl_Position.y *= swr_vp.x; if(swr_vp.y!=0.0) gl_Position.z = gl_Position.z*2.0 - gl_Position.w; EmitVertex(); }";
      char* out=malloc(strlen(src)*3+64); char* o=out; const char* q2=src;
      while(*q2){ if(!strncmp(q2,"EmitVertex();",13)){ o+=sprintf(o,"%s",fx); q2+=13; } else *o++=*q2++; }
      *o=0; free(src); src=out; } }
  /* Y flip */
  if(stage==ST_VS && src && !(g_last_r3d && (g_last_r3d[0x840]&1))){ char* p=strstr(src,"void main(){"); if(p){ size_t pre=p-src; char* ns=malloc(strlen(src)+512);
      memcpy(ns,src,pre); ns[pre]=0; strcat(ns,"uniform vec4 swr_vp;\n"); strcat(ns,p); free(src); src=ns;
      /* patch every return */
      char* out=malloc(strlen(src)*3+64); char* o=out; const char* q2=src;
      int lk=strstr(src,"bool swr_lk")!=NULL;
      while(*q2){ if(!strncmp(q2,"return;",7)){ o+=sprintf(o,"{ gl_Position.y *= swr_vp.x; if(swr_vp.y!=0.0) gl_Position.z = gl_Position.z*2.0 - gl_Position.w;%s return; }",lk? " if(swr_lk) gl_Position=vec4(2.0,2.0,2.0,1.0);" : ""); q2+=7; } else *o++=*q2++; }
      *o=0; free(src); src=out; } }
  return src; }

/* task / mesh programs: the draw's mesh registers and headers, set before shader_get (part of the cache key) */
static ShaderInfo g_mesh_si; static u32 g_mesh_key, g_mesh_threads_m;   /* mesh_threads: the task's, g_mesh_threads_m: the mesh program's */
static Sh* shader_get(u64 gva_code, int stage, u32 attr_int, u32 attr_sint){
  st_shaders++; g_sh_bind= stage==ST_VS || stage==ST_TASK? 0 : stage==ST_MESH? 1 : stage==ST_GS? 3 : 4; g_shader_cb_read=sh_cb_read; /* BRX table is only used to mark targets, length doesn't change */
  u32 key2= stage>=ST_TASK? g_mesh_key*0x27D4EB2Fu+(u32)stage : stage==ST_VS? (attr_int^(attr_sint*31)^((g_last_r3d && (g_last_r3d[0x840]&1))? 0xA341316Cu : 0)) : stage==ST_GS? (g_gs_input_vertices*0x85EBCA6Bu) : (attr_int*0x9E3779B1u ^ attr_sint*0x85EBCA6Bu ^ g_fs_rtk);
  ShA* ca=&sha[(u32)((gva_code>>6)^(gva_code>>18)^((u64)key2*0x9E3779B1u)^((u32)stage<<11))&4095];
  if(ca->s && ca->gva==gva_code && ca->stage==stage && ca->key2==key2){
    if(ca->epoch==g_submit_epoch){ st_sh_fast++; return ca->s; }
    Sh* s=ca->s;
    /* write tracking says nothing touched the code since the last check, so no need to hash it again (hashing
       every shader of every submit cost ~2 ms a frame on a slow cpu) */
    if(g_ww>0 && ca->cpu && g_submit_epoch-ca->chk<240 && gmmu_translate(gva_code)==ca->cpu && !ww_dirty(ca->cpu,s->clen,ca->chk)){
      ca->epoch=g_submit_epoch; st_sh_fast++; return s; }
    u8* p=gmem_ptr(gva_code,s->clen); u64 q;
    if(p) q=fast_hash(p,s->clen,0); else { u8* t2=malloc(s->clen? s->clen : 1); gmem_read(gva_code,t2,s->clen); q=fast_hash(t2,s->clen,0); free(t2); }
    if(q==s->qh){ ca->epoch=g_submit_epoch; ca->chk=g_submit_epoch; ca->cpu=gmmu_translate(gva_code); st_sh_fast++; return s; } }
  /* 128 KB of Ampere code and of lowered Maxwell code (shader_hash scans 16384 words): Maxwell needs a control
     word per 3 instructions, the lowered code of a ~2500 instruction MKW pixel shader (b784c8e4e9fee949, its
     ambient/luminance chain) passed 64 KB, was cut there and came out 1474 "unknown 0" instructions */
  u8* code=gmem_ptr(gva_code,SH_CODE_MAX); u8* tmp=NULL;
  if(!code){ tmp=malloc(SH_CODE_MAX); gmem_read(gva_code,tmp,SH_CODE_MAX); code=tmp; }
  u32 len; u64 h, qh;
  /* Switch 2: Ampere code goes through sm86.c first, the cache keys and the change check stay on the Ampere bytes */
  u8* low=NULL; Sm86Info sinf; int amp=g_sh_ampere;
  if(amp){ u32 ol, al; u8* m=sm86_lower(code,SH_CODE_MAX,stage!=ST_CS,&ol,&al,&sinf); if(!m){ free(tmp); return NULL; }
    { static int logged; while(logged<g_sm86_nuntr){ LOG("sm86: opcode %03x not translated\n",g_sm86_untr[logged]); logged++; } }
    if(getenv("SWR_DUMP_AMP") && (sinf.untranslated || !strcmp(getenv("SWR_DUMP_AMP"),"all"))){   /* =all: every shader */ static int nd; if(nd<4000){ nd++; char fp[200]; plat_mkdir("captures"); plat_mkdir("captures/amp");   /* raw Ampere code to study the opcodes left */
        snprintf(fp,sizeof fp,"captures/amp/%016llx_%d.bin",(unsigned long long)fast_hash(code,al,0),stage); FILE* f=fopen(fp,"wb"); if(f){ fwrite(code,1,al,f); fclose(f); } } }
    low=calloc(1,SH_CODE_MAX); memcpy(low,m,ol<SH_CODE_MAX? ol : SH_CODE_MAX); free(m);
    len=al; qh=fast_hash(code,len,0); h=qh^0x5A3D86A3C0FFEE11ull; }
  else { h=shader_hash(code,SH_CODE_MAX,&len); qh=0; }
  for(Sh* s=shh[h&1023];s;s=s->next) if(s->hash==h && s->stage==stage && s->key2==key2){
    ca->gva=gva_code; ca->key2=key2; ca->stage=stage; ca->epoch=g_submit_epoch; ca->chk=g_submit_epoch; ca->cpu=gmmu_translate(gva_code); ca->s=s; free(tmp); free(low); return s; }
  Sh* s=calloc(1,sizeof *s); s->hash=h; s->stage=stage; s->key2=key2; st_sh_new++;
  s->clen=len; s->qh= amp? qh : fast_hash(code,len,0);
  if(amp){ static int said; if(said<40 || sinf.untranslated){ said++;
      if(sinf.untranslated || sinf.reg_clash) LOG("gpu: Ampere %s shader %016llx: %u instructions, %u not translated (first #%u, opcode %03x)%s\n",stage==ST_VS?"vs":stage==ST_FS?"fs":"other",
        (unsigned long long)h,sinf.insts,sinf.untranslated,sinf.first_bad,sinf.bad_op,sinf.reg_clash? ", uses R176+ (clashes with the uniform register copies)" : "");
      else LOG("gpu: Ampere %s shader %016llx: %u instructions, all translated\n",stage==ST_VS?"vs":stage==ST_FS?"fs":"other",(unsigned long long)h,sinf.insts); }
    code=low; }
  ca->gva=gva_code; ca->key2=key2; ca->stage=stage; ca->epoch=g_submit_epoch; ca->chk=g_submit_epoch; ca->cpu=gmmu_translate(gva_code); ca->s=s;
  if(stage==ST_FS){ s->si.tex_int_mask=attr_int; s->si.tex_sint_mask=attr_sint; s->si.rt_int_mask=g_fs_rtm; s->si.rt_sint_mask=g_fs_rts; }
  else { s->si.attr_int_mask=attr_int; s->si.attr_sint_mask=attr_sint; }
  if(stage==ST_GS) s->si.gs_input_vertices=g_gs_input_vertices;
  if(stage>=ST_TASK){ ShaderInfo* m=&g_mesh_si; s->si.mesh_threads= stage==ST_TASK? m->mesh_threads : g_mesh_threads_m; s->si.mesh_payload_words=m->mesh_payload_words;
    s->si.mesh_max_vertices=m->mesh_max_vertices; s->si.mesh_max_primitives=m->mesh_max_primitives; s->si.mesh_has_task=m->mesh_has_task;
    memcpy(s->si.mesh_vlines,m->mesh_vlines,sizeof m->mesh_vlines); memcpy(s->si.mesh_plines,m->mesh_plines,sizeof m->mesh_plines); s->si.mesh_nvlines=m->mesh_nvlines; s->si.mesh_nplines=m->mesh_nplines; }
  char* src=shader_to_glsl(code,SH_CODE_MAX,stage,&s->si);
  if(amp && sinf.tex_bank>=0 && sinf.tex_bank<18){ s->si.tex_bank_set=1; s->si.tex_bank=(u8)sinf.tex_bank; }
  /* dump shaders with indirect branches (code + cbufs) for offline analysis */
  { extern int g_opt_diag; if(g_opt_diag && src && strstr(src,"uint t_ = uint(")){ static int nb; if(nb<24){ nb++; plat_mkdir("captures"); plat_mkdir("captures/failed"); char fp[200];
      snprintf(fp,sizeof fp,"captures/failed/%016llx_%s.bin",(unsigned long long)h,stage==ST_VS?"vs":"fs"); FILE* f=fopen(fp,"wb"); if(f){ fwrite(code,1,SH_CODE_MAX,f); fclose(f); }
      sh_dump_cbs(h);
      u32* rr=g_last_r3d; if(rr){ snprintf(fp,sizeof fp,"captures/failed/%016llx_stages.txt",(unsigned long long)h); f=fopen(fp,"w");
        if(f){ for(int k=0;k<6;k++) fprintf(f,"slot %d: cfg=%08x offset=%08x\n",k,rr[0x800+k*0x10],rr[0x801+k*0x10]); fclose(f); } } } } }
  if(stage!=ST_FS) s->si.attr_sint_mask=attr_sint;
  if(s->si.unsupported) WARN("shader %016llx: %d untranslated instructions (first: %s)\n",(unsigned long long)h,s->si.unsupported,s->si.first_unsupported);
  if(s->si.unsupported && strstr(s->si.first_unsupported,"untracked")){ static int nd; if(nd<sw2_cap(8)){ nd++; char fp[200]; plat_mkdir("captures"); plat_mkdir("captures/failed");
      snprintf(fp,sizeof fp,"captures/failed/%016llx_%s.bin",(unsigned long long)h,stage==ST_VS?"vs":"fs"); aw_write(fp,code,SH_CODE_MAX,0);
      LOG("shader %016llx saved to %s (send it so the missing texture can be analyzed)\n",(unsigned long long)h,fp); } }
  src=sh_post(src,stage);
  if(!src) s->failed=1;
  s->src=src; src=NULL;   /* only compiled if it isn't in the disk cache */
  { static int dump=-1; if(dump<0){ dump=getenv("SWR_DUMP_SHADERS")!=NULL; } if(dump && s->src){ char pth[256]; snprintf(pth,sizeof pth,"shader_%016llx_%d.glsl",(unsigned long long)h,stage); FILE* f=fopen(pth,"w"); if(f){ fputs(s->src,f); fclose(f);} } }
  free(src); free(tmp); free(low);
  s->next=shh[h&1023]; shh[h&1023]=s; return s; }

/* deferred compile + on-disk program binaries */
static int sh_ensure(Sh* s){
  if(s->gl) return 1; if(s->failed || !s->src) return 0;
  GLenum ty=shader_gl_type(s->stage); char err[4096]; err[0]=0;
  s->gl=compile(ty,s->src,err,sizeof err);
  if(!s->gl){ s->failed=1; WARN("shader %016llx (stage %d) failed to compile:\n%s\n",(unsigned long long)s->hash,s->stage,err); if(g_verbose>=2) WARN("---- GLSL ----\n%s\n",s->src); return 0; }
  return 1; }

static int g_pcache=-1; static char g_pc_tag[64];

/* the binary cache key is the GLSL text, so a new exe that writes the same source reuses the old binaries */
static u64 src_key(Sh* s){ if(!s->skey){ u64 h= s->src? fast_hash((const u8*)s->src,strlen(s->src),0) : s->hash; s->skey=(h^((u64)(s->stage+1)<<58))|1; } return s->skey; }

static u64 prog_key(Sh* vs, Sh* fs){ return src_key(vs)*0x9E3779B97F4A7C15ull ^ src_key(fs)*0xC2B2AE3D27D4EB4Full ^ 0x5E9A0001ull; }

static u64 sh_key(Sh* s){ return src_key(s)*0x9E3779B97F4A7C15ull ^ 0x5E9A0000ull; }

static int g_par=-1;

static void pcache_init(void){
  if(g_pcache<0){ g_pcache= p_glProgramBinary && p_glGetProgramBinary && p_glProgramParameteri; if(g_pcache){ plat_mkdir("shader_cache");
      const char* rn=(const char*)glGetString(GL_RENDERER); const char* vv=(const char*)glGetString(GL_VERSION); u64 h=1469598103934665603ull;
      for(const char* q=rn;q&&*q;q++){ h^=(u8)*q; h*=1099511628211ull; } for(const char* q=vv;q&&*q;q++){ h^=(u8)*q; h*=1099511628211ull; }
      /* and the runtime build: program binaries from an older translator must not come back after a shader fix */
      for(const char* q=__DATE__ " " __TIME__;*q;q++){ h^=(u8)*q; h*=1099511628211ull; }
      snprintf(g_pc_tag,sizeof g_pc_tag,"%08x",(u32)(h^(h>>32))); } } }

static void pack_add(u64 key, Sh* vs, Sh* fs);
static void pack_add_sh(u64 key, Sh* s);

static int sh_start(Sh* s){ if(s->gl) return 1; if(s->failed || !s->src) return 0;
  s->gl=glCreateShader(shader_gl_type(s->stage)); const char* src=s->src; glShaderSource(s->gl,1,&src,NULL); glCompileShader(s->gl); return 1; }

static GLuint prog_finish(Sh* vs, Sh* fs, GLuint p);
static u64 prog_key(Sh* vs, Sh* fs);
static int prog_cached(Sh* vs, Sh* fs);

static GLuint prog_build(Sh* vs, Sh* fs, int* pending){
  if(!vs || !fs || vs->failed || fs->failed) return 0;
  pcache_init();
  char path[160]; u64 key=prog_key(vs,fs);
  if(g_pcache){ snprintf(path,sizeof path,"shader_cache/%s_%016llx.bin",g_pc_tag,(unsigned long long)key);
    FILE* f=fopen(path,"rb");
    if(f){ u32 fmt=0, len=0; if(fread(&fmt,4,1,f)==1 && fread(&len,4,1,f)==1 && len && len<(64u<<20)){ void* b=malloc(len);
        if(fread(b,1,len,f)==len){ GLuint p=glCreateProgram(); p_glProgramBinary(p,fmt,b,(GLsizei)len); GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
          if(ok){ free(b); fclose(f); pack_add(key,vs,fs); return p; } glDeleteProgram(p); }
        free(b); }
      fclose(f); } }
  if(g_par>0 && pending){ /* parallel compile: kick it off, poll on later draws */
    if(!sh_start(vs) || !sh_start(fs)) return 0;
    GLuint p=glCreateProgram(); glAttachShader(p,vs->gl); glAttachShader(p,fs->gl);
    if(g_pcache) p_glProgramParameteri(p,GL_PROGRAM_BINARY_RETRIEVABLE_HINT,GL_TRUE);
    glLinkProgram(p); *pending=1; return p; }
  if(!sh_ensure(vs) || !sh_ensure(fs)) return 0;
  GLuint p=glCreateProgram(); glAttachShader(p,vs->gl); glAttachShader(p,fs->gl);
  if(g_pcache) p_glProgramParameteri(p,GL_PROGRAM_BINARY_RETRIEVABLE_HINT,GL_TRUE);
  glLinkProgram(p); return prog_finish(vs,fs,p); }

static GLuint prog_finish(Sh* vs, Sh* fs, GLuint p){
  char path[160]; u64 key=prog_key(vs,fs); if(g_pcache) snprintf(path,sizeof path,"shader_cache/%s_%016llx.bin",g_pc_tag,(unsigned long long)key);
  GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
  if(!ok){ Sh* ss[2]={vs,fs}; for(int k=0;k<2;k++){ GLint cok=1; glGetShaderiv(ss[k]->gl,GL_COMPILE_STATUS,&cok);
      if(!cok){ char e[4096]; glGetShaderInfoLog(ss[k]->gl,sizeof e,NULL,e); WARN("shader %016llx (stage %d) failed to compile:\n%s\n",(unsigned long long)ss[k]->hash,ss[k]->stage,e); ss[k]->failed=1; } } }
  if(!ok){ char log[2048]; glGetProgramInfoLog(p,2048,NULL,log); WARN("GL link: %s\n",log); glDeleteProgram(p);
    static int nd; if(nd<12 && vs->src && fs->src){ nd++; plat_mkdir("captures"); plat_mkdir("captures/failed"); char fp[200];
      snprintf(fp,sizeof fp,"captures/failed/%016llx_vs.glsl",(unsigned long long)vs->hash); FILE* f=fopen(fp,"w"); if(f){ fputs(vs->src,f); fclose(f); }
      snprintf(fp,sizeof fp,"captures/failed/%016llx_fs.glsl",(unsigned long long)fs->hash); f=fopen(fp,"w"); if(f){ fputs(fs->src,f); fclose(f); } }
    return 0; }
  pack_add(key,vs,fs);
  if(g_pcache){ GLint len=0; glGetProgramiv(p,GL_PROGRAM_BINARY_LENGTH,&len);
    if(len>0){ void* b=malloc(len); GLenum fmt=0; GLsizei got=0; p_glGetProgramBinary(p,len,&got,&fmt,b);
      if(got>0){ FILE* f=fopen(path,"wb"); if(f){ u32 fm=fmt, ln=(u32)got; fwrite(&fm,4,1,f); fwrite(&ln,4,1,f); fwrite(b,1,got,f); fclose(f); } } free(b); } }
  return p; }

/* ============================================================ shader compile thread
   second shared context that compiles + links, so the GPU thread never blocks on the driver compiler. draws whose
   program isn't ready yet get skipped for a few frames */
static int prog_cached(Sh* vs, Sh* fs){ pcache_init(); if(g_pcache<=0) return 0; char path[160]; snprintf(path,sizeof path,"shader_cache/%s_%016llx.bin",g_pc_tag,(unsigned long long)prog_key(vs,fs)); FILE* f=fopen(path,"rb"); if(!f) return 0; fclose(f); return 1; }

typedef struct WJob { const char* vs; const char* fs; char* own_vs; char* own_fs; u64 key; volatile int state; GLuint prog; int keep; int sep; GLenum stype; char log[1536]; int try_bin; char bin_path[160]; GLint loc[2]; } WJob;

/* uniform locations are looked up here on the compile thread's own context, on the GPU thread every
   glGetUniformLocation waited for the whole GL queue */
static void job_locs(WJob* j, GLuint p){
  if(j->sep){ j->loc[0]= glGetUniformLocation(p, (j->stype==GL_VERTEX_SHADER || j->stype==GL_GEOMETRY_SHADER || j->stype==0x9559 /* mesh */)? "swr_vp" : "swr_at"); j->loc[1]=-1; }
  else { j->loc[0]=glGetUniformLocation(p,"swr_vp"); j->loc[1]=glGetUniformLocation(p,"swr_at"); } }

#define WQN 8192
typedef struct { WJob* volatile e[WQN]; volatile u32 h, t; } WRing;

static WRing wq_hi, wq_lo; static volatile int g_worker; static volatile u64 g_cmp_ns, g_cmp_n, g_cmp_max, g_bin_n, g_bin_ns;   /* 0 not started, 1 running, -1 unavailable */
static int wr_push(WRing* r, WJob* j){ u32 t=__atomic_load_n(&r->t,__ATOMIC_RELAXED), h=__atomic_load_n(&r->h,__ATOMIC_ACQUIRE); if(t-h>=WQN) return 0;
  r->e[t%WQN]=j; __atomic_store_n(&r->t,t+1,__ATOMIC_RELEASE); return 1; }

static WJob* wr_pop(WRing* r){ u32 h=__atomic_load_n(&r->h,__ATOMIC_RELAXED), t=__atomic_load_n(&r->t,__ATOMIC_ACQUIRE); if(h==t) return NULL;
  WJob* j=r->e[h%WQN]; __atomic_store_n(&r->h,h+1,__ATOMIC_RELEASE); return j; }

extern int backend_gl_worker_current(void);

/* some drivers crash while compiling or linking one shader (intel on SMO's cutscenes). we remember which program
   it was and skip it next time, so that effect is missing instead of the game closing */
/* per compile thread, the crash handler on that thread reads it with plat_guard_get() */
typedef struct { plat_jmpbuf jb; u64 key; const char* vs; const char* fs; WJob* job; } CmpGuard;
static u64 g_bad_keys[64]; static int g_nbad=-1; static int g_excl;   /* 1: driver exclusive mode (intel on windows) */
static volatile int g_shp_on;   /* 1: shaders compile in the helper process (shader_proc.c) */
static const char* bad_prefix(void){ return g_shp_on? "crashp_" : g_excl? "crashx_" : "crash_"; }
static void bad_keys_load(void){ if(g_nbad>=0) return; g_nbad=0; PDir d; if(plat_dir_open(&d,"shader_cache")) return;
  /* in exclusive mode the old crash_ files don't count, those came from the compile threads racing each other and
     most of them are fine shaders. with the helper process only crashp_ counts: every version of it crashed there */
  const char* pre=bad_prefix(); size_t pl=strlen(pre);
  while(plat_dir_next(&d)) if(!strncmp(d.name,pre,pl) && g_nbad<64){ unsigned long long k=0; if(sscanf(d.name+pl,"%llx",&k)==1 && k) g_bad_keys[g_nbad++]=k; }
  plat_dir_close(&d); if(g_nbad) LOG("shaders: %d shader(s) crashed the video driver before and will be skipped (shader_cache/crash_*.txt)\n",g_nbad); }

static int bad_key(u64 k){ for(int i=0;i<g_nbad;i++) if(g_bad_keys[i]==k) return 1; return 0; }

static void crash_file(u64 k, const char* a, const char* b){
  plat_mkdir("shader_cache"); char p[200]; snprintf(p,sizeof p,"shader_cache/%s%016llx.txt",bad_prefix(),(unsigned long long)k);
  FILE* f=fopen(p,"w"); if(f){ fprintf(f,"// this program crashed the video driver. delete this file to try it again\n");
    if(a) fputs(a,f); if(b){ fputs("\n// ----\n",f); fputs(b,f); } fclose(f); } }

void gpu_crash_note(void){ CmpGuard* g=(CmpGuard*)plat_guard_get(); u64 k= g? g->key : 0; if(!k) return;
  /* only the first crash of a run goes on the list. after it the driver is broken and anything crashes, those
     shaders are fine */
  static volatile int nsaved; if(__atomic_add_fetch(&nsaved,1,__ATOMIC_RELAXED)>1){ logf_(0,"  the video driver crashed again on %016llx (not saved, the driver was already broken)\n",(unsigned long long)k); return; }
  logf_(0,"  the video driver crashed while compiling/linking shader program %016llx; it will be skipped from now on\n",(unsigned long long)k);
  crash_file(k,g->vs,g->fs); }

/* ---- the helper process (shader_proc.c) */
int shp_compile(int slot, int sep, unsigned stype, const char* vs, const char* fs, unsigned* fmt, void** bin, unsigned* len, char* log, int logn);
int shp_init(const char* id);
static volatile int g_shp_ok, g_shp_crash;
static u64 g_shp_saved[8]; static volatile int g_shp_nsaved;   /* skip list entries from this run */

static char* str_repl(const char* src, const char* from, const char* to){
  size_t fl=strlen(from), tl=strlen(to), n=0; for(const char* q=src;(q=strstr(q,from));q+=fl) n++;
  char* o=malloc(strlen(src)+n*(tl>fl? tl-fl : 0)+1); char* w=o;
  for(const char* q=src;;){ const char* m=strstr(q,from); if(!m){ strcpy(w,q); break; } memcpy(w,q,m-q); w+=m-q; memcpy(w,to,tl); w+=tl; q=m+fl; }
  return o; }

/* simpler versions for a driver that crashes on the real one. every shader intel crashed on was a vertex shader, and
   precise/invariant and the gl_PerVertex block are what only those have. 1: no precise/invariant, 2: no block either */
static char* glsl_relax(const char* s, int lv){
  char* a=str_repl(s,"invariant gl_Position;\n",""); char* b=str_repl(a,"invariant ",""); free(a); a=str_repl(b,"precise ",""); free(b);
  if(lv>=2){ b=str_repl(a,"out gl_PerVertex { vec4 gl_Position; float gl_PointSize; float gl_ClipDistance[8]; };\n","out float gl_ClipDistance[8];\n"); free(a); a=b; }
  return a; }

/* compiles j in the helper: as it is, and the simpler versions if the driver crashes on it.
   1 ok (the binary is in bin), 2 failed or skipped (j->log says why), 0 no helper: compile it here */
static int shp_job(WJob* j, int slot, u32* fmt, void** bin, u32* len){
  const char* a=j->vs; const char* b= j->sep? NULL : j->fs; int r=-1, crashed=0, broken=0;
  for(int t=0;t<3 && r==-1;t++){
    char* ra= t? glsl_relax(a,t) : NULL; char* rb= t && b? glsl_relax(b,t) : NULL;
    r=shp_compile(slot,j->sep,j->stype,ra? ra : a,rb? rb : b,fmt,bin,len,j->log,sizeof j->log);
    free(ra); free(rb);
    if(r==-1){ crashed=1; int nc=__atomic_add_fetch(&g_shp_crash,1,__ATOMIC_RELAXED);
      LOG("shaders: the video driver crashed on %016llx in the helper process (the game is fine)%s\n",(unsigned long long)j->key,t<2? ", trying a simpler version of it" : "");
      if(nc>=6 && !g_shp_ok){ g_shp_on=0; broken=1; LOG("shaders: the helper process crashes on every shader, compiling in the game instead\n");
        /* then those weren't the shaders' fault */
        for(int i=0;i<g_shp_nsaved && i<8;i++){ char p[200]; snprintf(p,sizeof p,"shader_cache/crashp_%016llx.txt",(unsigned long long)g_shp_saved[i]); remove(p); }
        break; } }
    else if(r==1 && t) LOG("shaders: %016llx compiled without %s, the driver only crashed on the full version\n",(unsigned long long)j->key,t==1? "precise/invariant" : "precise/invariant and the gl_PerVertex block"); }
  if(broken) return 0;
  if(r<0 && crashed){ crash_file(j->key,j->vs,j->sep? NULL : j->fs); { int i=__atomic_fetch_add(&g_shp_nsaved,1,__ATOMIC_RELAXED); if(i<8) g_shp_saved[i]=j->key; } snprintf(j->log,sizeof j->log,"skipped: the video driver crashes on every version of it (in the helper process, the game kept going)"); return 2; }
  if(r<0){ if(g_shp_on){ g_shp_on=0; LOG("shaders: the helper process stopped working, compiling in the game instead\n"); } return 0; }
  if(r==0) return 2;
  __atomic_add_fetch(&g_shp_ok,1,__ATOMIC_RELAXED);
  if(g_pcache>0){ char path[160]; snprintf(path,sizeof path,"shader_cache/%s_%s%016llx.bin",g_pc_tag,j->sep? "s" : "",(unsigned long long)j->key);
    FILE* f=fopen(path,"wb"); if(f){ u32 fm=*fmt, ln=*len; fwrite(&fm,4,1,f); fwrite(&ln,4,1,f); fwrite(*bin,1,*len,f); fclose(f); } }
  return 1; }

/* several compile threads pop from the same rings */
static volatile u32 g_wq_lk;
static WJob* wq_next(int* lo){
  while(__atomic_exchange_n(&g_wq_lk,1,__ATOMIC_ACQUIRE)) plat_yield();
  WJob* j=wr_pop(&wq_hi); *lo=0; if(!j){ j=wr_pop(&wq_lo); *lo=1; }
  __atomic_store_n(&g_wq_lk,0,__ATOMIC_RELEASE); return j; }

static volatile int g_nworkers;
static const char* volatile g_wk_phase; static volatile u64 g_wk_t0;   /* first compile thread, for the stall report */
#define WK_PHASE(first,s) do{ if(first){ g_wk_phase=(s); g_wk_t0=plat_ns(); } }while(0)

static void shader_worker(void* arg){
  int first= arg==NULL, slot=(int)(size_t)arg;
  if(!backend_gl_worker_current()){ if(first) __atomic_store_n(&g_worker,-1,__ATOMIC_RELEASE); return; }
  __atomic_add_fetch(&g_nworkers,1,__ATOMIC_ACQ_REL);
  if(first) __atomic_store_n(&g_worker,1,__ATOMIC_RELEASE);
  { extern int g_opt_cmp_low; if(g_opt_cmp_low) plat_thread_low_priority(); }   /* only use the cpu the game isn't using */
  for(;;){
    static __thread u32 idle;
    int lo=0; WJob* j=wq_next(&lo);
    if(!j){ WK_PHASE(first,NULL); plat_sleep_ns(idle<200? 1000000 : 8000000); if(idle<200) idle++; continue; }   /* idle for a while: poll less, the game can use that cpu */
    idle=0; WK_PHASE(first,"waiting for the driver lock");
    /* intel's driver crashes inside glLinkProgram on some of SMO's shaders (null pointer in the driver). the crash
       comes back here, the shader gets marked failed and saved to the skip list and the game keeps going without
       that effect */
    static __thread CmpGuard cg; static __thread int lk; cg.job=j; cg.key=0; cg.vs=NULL; cg.fs=NULL;
    if(plat_guard_set(cg.jb)){
      WJob* cj=cg.job; plat_guard_enter(NULL);
      if(lk){ lk=0; glq_drv_unlock(); }
      snprintf(cj->log,sizeof cj->log,"the video driver crashed compiling it (recovered)");
      cj->prog=0; __atomic_store_n(&cj->state,2,__ATOMIC_RELEASE);
      { static volatile int nrec; if(__atomic_add_fetch(&nrec,1,__ATOMIC_RELAXED)<=8) LOG("shaders: the video driver crashed on program %016llx, skipped it and kept going\n",(unsigned long long)cj->key); }
      continue; }
    lk=glq_drv_lock(); WK_PHASE(first, j->try_bin? "loading a cached shader" : "in the driver");
    plat_guard_enter(&cg);
    u64 tc0=plat_ns();
    if(j->try_bin){ /* disk binary, loaded here instead of on the GPU thread */
      FILE* f=fopen(j->bin_path,"rb"); GLuint p=0; GLint ok=0;
      if(f){ u32 fmt=0, len=0; if(fread(&fmt,4,1,f)==1 && fread(&len,4,1,f)==1 && len && len<(64u<<20)){ void* b=malloc(len);
          if(fread(b,1,len,f)==len){ p=glCreateProgram(); p_glProgramParameteri(p,0x8258,GL_TRUE); p_glProgramBinary(p,fmt,b,(GLsizei)len); glGetProgramiv(p,GL_LINK_STATUS,&ok); if(!ok){ glDeleteProgram(p); p=0; } }
          free(b); }
        fclose(f); }
      if(ok){ job_locs(j,p); glFinish(); u64 dt=plat_ns()-tc0; __atomic_add_fetch(&g_bin_ns,dt,__ATOMIC_RELAXED); __atomic_add_fetch(&g_bin_n,1,__ATOMIC_RELAXED);
        plat_guard_enter(NULL); if(lk){ lk=0; glq_drv_unlock(); } j->prog=p; __atomic_store_n(&j->state,1,__ATOMIC_RELEASE); continue; }
      while(glGetError()!=GL_NO_ERROR){} }
    if(bad_key(j->key)){ plat_guard_enter(NULL); if(lk){ lk=0; glq_drv_unlock(); } snprintf(j->log,sizeof j->log,"skipped: it crashed the video driver in an earlier run"); j->prog=0; __atomic_store_n(&j->state,2,__ATOMIC_RELEASE); continue; }
    if(g_shp_on){ /* compiled in the helper process, only its binary gets loaded here. the driver lock isn't held while we wait */
      plat_guard_enter(NULL); if(lk){ lk=0; glq_drv_unlock(); }
      WK_PHASE(first,"waiting for the helper process");
      u32 fmt=0, len=0; void* bin=NULL; int r=shp_job(j,slot,&fmt,&bin,&len);
      if(r==2 || (r==1 && !j->keep)){ free(bin); j->prog=0; __atomic_store_n(&j->state,r,__ATOMIC_RELEASE); if(lo) plat_sleep_ns(2000000); continue; }
      WK_PHASE(first,"waiting for the driver lock"); lk=glq_drv_lock(); WK_PHASE(first,"loading the helper's binary"); plat_guard_enter(&cg);
      if(r==1){ cg.key=j->key; GLuint p=glCreateProgram(); if(j->sep) p_glProgramParameteri(p,0x8258,GL_TRUE); p_glProgramBinary(p,fmt,bin,(GLsizei)len); free(bin);
        GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
        if(ok){ job_locs(j,p); glFinish(); u64 dt=plat_ns()-tc0; __atomic_add_fetch(&g_cmp_ns,dt,__ATOMIC_RELAXED); __atomic_add_fetch(&g_cmp_n,1,__ATOMIC_RELAXED); if(dt>g_cmp_max) g_cmp_max=dt;
          cg.key=0; plat_guard_enter(NULL); if(lk){ lk=0; glq_drv_unlock(); } j->prog=p; __atomic_store_n(&j->state,1,__ATOMIC_RELEASE); if(lo) plat_sleep_ns(2000000); continue; }
        glDeleteProgram(p); while(glGetError()!=GL_NO_ERROR){} cg.key=0;
        if(g_shp_on){ g_shp_on=0; LOG("shaders: the driver doesn't load the helper's program binaries, compiling in the game instead\n"); } }
      /* no helper: compiled below */ }
    cg.vs=j->vs; cg.fs= j->sep? NULL : j->fs; cg.key=j->key;
    GLuint a=glCreateShader(j->sep? j->stype : GL_VERTEX_SHADER), b= j->sep? 0 : glCreateShader(GL_FRAGMENT_SHADER);
    const char* s1=j->vs; const char* s2=j->fs; glShaderSource(a,1,&s1,NULL); glCompileShader(a); if(b){ glShaderSource(b,1,&s2,NULL); glCompileShader(b); }
    GLuint p=glCreateProgram(); glAttachShader(p,a); if(b) glAttachShader(p,b);
    if(j->sep) p_glProgramParameteri(p,0x8258 /* GL_PROGRAM_SEPARABLE */,GL_TRUE);
    if(g_pcache>0) p_glProgramParameteri(p,GL_PROGRAM_BINARY_RETRIEVABLE_HINT,GL_TRUE);
    glLinkProgram(p); GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ GLint c=1; char e[1024]; e[0]=0; glGetShaderiv(a,GL_COMPILE_STATUS,&c); if(!c){ glGetShaderInfoLog(a,sizeof e,NULL,e); snprintf(j->log,sizeof j->log,"%s: %s",j->sep? (j->stype==GL_VERTEX_SHADER? "VS" : j->stype==GL_GEOMETRY_SHADER? "GS" : "FS") : "VS",e); }
      else if(b && (glGetShaderiv(b,GL_COMPILE_STATUS,&c),!c)){ glGetShaderInfoLog(b,sizeof e,NULL,e); snprintf(j->log,sizeof j->log,"FS: %s",e); } else glGetProgramInfoLog(p,sizeof j->log,NULL,j->log);
      glDeleteProgram(p); p=0; }
    else if(g_pcache>0){ GLint len=0; glGetProgramiv(p,GL_PROGRAM_BINARY_LENGTH,&len);
      if(len>0){ void* bb=malloc(len); GLenum fmt=0; GLsizei got=0; p_glGetProgramBinary(p,len,&got,&fmt,bb);
        char path[160]; snprintf(path,sizeof path,"shader_cache/%s_%s%016llx.bin",g_pc_tag,j->sep? "s" : "",(unsigned long long)j->key);
        if(got>0){ FILE* f=fopen(path,"wb"); if(f){ u32 fm=fmt, ln=(u32)got; fwrite(&fm,4,1,f); fwrite(&ln,4,1,f); fwrite(bb,1,got,f); fclose(f); } } free(bb); } }
    glDeleteShader(a); if(b) glDeleteShader(b);   /* flagged for deletion, freed together with the program */
    if(p) job_locs(j,p);
    if(p && !j->keep){ glDeleteProgram(p); p=0; }
    glFinish();
    { u64 dt=plat_ns()-tc0; __atomic_add_fetch(&g_cmp_ns,dt,__ATOMIC_RELAXED); __atomic_add_fetch(&g_cmp_n,1,__ATOMIC_RELAXED); if(dt>g_cmp_max) g_cmp_max=dt; }
    cg.key=0; plat_guard_enter(NULL); if(lk){ lk=0; glq_drv_unlock(); }
    j->prog=p; __atomic_store_n(&j->state, ok? 1 : 2, __ATOMIC_RELEASE);
    if(lo) plat_sleep_ns(2000000);   /* background work yields to the game */
  } }

static void worker_start(void){
  { extern int g_opt_drv_excl; const char* e=getenv("SWR_DRV_EXCL");
    int want= e? atoi(e) : g_opt_drv_excl>=0? g_opt_drv_excl : 0;
#ifdef _WIN32
    if(!e && g_opt_drv_excl<0){ const char* v=(const char*)glGetString(GL_VENDOR); if(v && strstr(v,"Intel")) want=1; }
#endif
    if(want){ if(glq_active()){ glq_excl(); g_excl=1; LOG("shaders: driver exclusive mode (compiling and drawing take turns inside the driver, intel crashes otherwise)\n"); }
      else { extern int g_opt_shader_thread; g_opt_shader_thread=0; } } }
  /* intel on windows: the shaders compile in a helper process once the driver crashed on one in this game (a crash_
     file in shader_cache), then only the helper goes. it isn't on from the start: tomodachi froze on intel HD a few
     seconds after its first compute with it on, and most games never crash the driver */
  { extern int g_opt_shader_proc; extern int g_opt_shader_thread; const char* e=getenv("SWR_SHADER_PROC");
    int crashed=0; { PDir d; if(!plat_dir_open(&d,"shader_cache")){ while(plat_dir_next(&d)) if(!strncmp(d.name,"crashx_",7) || !strncmp(d.name,"crashp_",7)) crashed=1; plat_dir_close(&d); } }
    int want= e? atoi(e) : g_opt_shader_proc>=0? g_opt_shader_proc : g_excl && crashed;
    pcache_init();
    if(want && g_pcache>0 && g_opt_shader_thread && !getenv("SWR_SYNC_SHADERS")){ char id[512]; const char* v=(const char*)glGetString(GL_VENDOR); const char* r=(const char*)glGetString(GL_RENDERER); const char* x=(const char*)glGetString(GL_VERSION);
      snprintf(id,sizeof id,"%s|%s|%s",v?v:"",r?r:"",x?x:"");
      int st=shp_init(id);
      if(!st){ g_shp_on=1; LOG("shaders: compiled in a helper process (if the video driver crashes on one, only the helper closes and the game keeps going)\n"); }
      else LOG("shaders: the helper process %s, shaders compile in the game\n",st==-3? "got another GPU or no program binaries" : "did not start"); }
    else if(g_excl && !e && g_opt_shader_proc<0) LOG("shaders: compiled in the game (if the video driver ever crashes on one, the next run compiles them in a helper process)\n"); }
  bad_keys_load();
  { extern int g_opt_shader_thread; if(getenv("SWR_SYNC_SHADERS") || !g_opt_shader_thread){ g_worker=-1; LOG("shaders: compiler thread off (shader_thread=0): shaders compile on the GPU thread\n"); return; } }
  if(plat_thread(shader_worker,NULL,8u<<20)){ g_worker=-1; return; }
  u64 t0=plat_ns(); while(!__atomic_load_n(&g_worker,__ATOMIC_ACQUIRE) && plat_ns()-t0<2000000000ull) plat_sleep_ns(1000000);
  if(g_worker!=1) g_worker=-1;
  /* AMD and intel compile one shader per thread and take 100-200 ms each on big ones, so a burst of new shaders
     kept draws waiting for seconds. more compile threads on cpus with the cores for it, each one gets its own
     shared context */
  if(g_worker==1){ extern int plat_ncpu(void); int nc=plat_ncpu(), extra= g_excl && !g_shp_on? 0 : nc>=12? 2 : nc>=6? 1 : 0;
    const char* e=getenv("SWR_SHADER_THREADS"); if(e){ extra=atoi(e)-1; if(extra<0) extra=0; if(extra>2) extra=2; }
    for(int i=0;i<extra;i++) plat_thread(shader_worker,(void*)(size_t)(i+1),8u<<20);
    u64 t1=plat_ns(); while(__atomic_load_n(&g_nworkers,__ATOMIC_ACQUIRE)<1+extra && plat_ns()-t1<500000000ull) plat_sleep_ns(1000000); }
  LOG("shaders: %d compiler thread(s) %s\n",g_worker==1? g_nworkers : 0,g_worker==1? "active (shaders compile in the background without freezing the game)" : "not available (shaders compile on the GPU thread)"); }

/* ============================================================ separable programs
   one program per shader stage, compiled once. draws just bind the stages instead of linking every vs+fs
   combination */
static int g_sep=-1; static GLuint g_pipe; static int g_pipe_on; static GLuint g_pipe_vs, g_pipe_gs, g_pipe_fs;
typedef void (APIENTRY*PFN_UPS_)(GLuint,GLbitfield,GLuint); typedef void (APIENTRY*PFN_BPP_)(GLuint); typedef void (APIENTRY*PFN_GPP_)(GLsizei,GLuint*);
static PFN_UPS_ p_ups; static PFN_BPP_ p_bpp;

static void sh_uloc(Sh* s){ s->uloc= s->sp? glGetUniformLocation(s->sp, (s->stage==ST_VS || s->stage==ST_GS)? "swr_vp" : "swr_at") : -1; }

static int sh_sep_ready(Sh* s){
  if(s->sps>=2) return 1;
  if(s->sps==1){ int st=__atomic_load_n(&s->sj->state,__ATOMIC_ACQUIRE); if(!st) return 0;
    struct WJob* j=s->sj; s->sj=NULL;
    if(st==1){ s->sp=j->prog; s->sps=2; pack_add_sh(j->key,s); s->uloc=j->loc[0]; free(j); return 1; }
    else { s->sp=0; s->sps=3; WARN("shader %016llx (stage %d) failed to compile: %s\n",(unsigned long long)s->hash,s->stage,j->log);
      static int nd; if(nd<16 && s->src){ nd++; plat_mkdir("captures"); plat_mkdir("captures/failed"); char fp[200];
        snprintf(fp,sizeof fp,"captures/failed/%016llx_%s.glsl",(unsigned long long)s->hash,s->stage==ST_VS?"vs":s->stage==ST_GS?"gs":"fs"); FILE* f=fopen(fp,"w"); if(f){ fputs(s->src,f); fclose(f); } } }
    free(j); sh_uloc(s); return 1; }
  s->sps=3; if(s->failed || !s->src) return 1;
  u64 key=sh_key(s); pcache_init();
  if(g_worker==1){ /* disk binary or compile, both on the compile thread */
    struct WJob* j=calloc(1,sizeof *j); j->vs=s->src; j->key=key; j->keep=1; j->sep=1; j->stype=shader_gl_type(s->stage);
    if(g_pcache>0){ j->try_bin=1; snprintf(j->bin_path,sizeof j->bin_path,"shader_cache/%s_s%016llx.bin",g_pc_tag,(unsigned long long)key); }
    if(wr_push(&wq_hi,j)){ s->sj=j; s->sps=1; return 0; }
    free(j); }
  if(g_pcache>0){ char path[160]; snprintf(path,sizeof path,"shader_cache/%s_s%016llx.bin",g_pc_tag,(unsigned long long)key); FILE* f=fopen(path,"rb");
    if(f){ u32 fmt=0, len=0; if(fread(&fmt,4,1,f)==1 && fread(&len,4,1,f)==1 && len && len<(64u<<20)){ void* b=malloc(len);
        if(fread(b,1,len,f)==len){ GLuint p=glCreateProgram(); p_glProgramParameteri(p,0x8258,GL_TRUE); p_glProgramBinary(p,fmt,b,(GLsizei)len); GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
          if(ok){ free(b); fclose(f); s->sp=p; s->sps=2; sh_uloc(s); pack_add_sh(key,s); return 1; } glDeleteProgram(p); }
        free(b); }
      fclose(f); } }
  struct WJob* j=calloc(1,sizeof *j); j->vs=s->src; j->key=key; j->keep=1; j->sep=1; j->stype=shader_gl_type(s->stage);
  if(g_worker==1 && wr_push(&wq_hi,j)){ s->sj=j; s->sps=1; return 0; }
  /* no compile thread, do it inline */
  GLuint a=glCreateShader(j->stype); const char* s1=s->src; glShaderSource(a,1,&s1,NULL); glCompileShader(a);
  GLuint p=glCreateProgram(); glAttachShader(p,a); p_glProgramParameteri(p,0x8258,GL_TRUE); if(g_pcache>0) p_glProgramParameteri(p,GL_PROGRAM_BINARY_RETRIEVABLE_HINT,GL_TRUE);
  glLinkProgram(p); glDeleteShader(a); GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
  if(!ok){ glGetProgramInfoLog(p,sizeof j->log,NULL,j->log); WARN("shader %016llx failed to compile: %s\n",(unsigned long long)s->hash,j->log); glDeleteProgram(p); p=0; }
  else if(g_pcache>0){ GLint len=0; glGetProgramiv(p,GL_PROGRAM_BINARY_LENGTH,&len);
    if(len>0){ void* bb=malloc(len); GLenum fmt=0; GLsizei got=0; p_glGetProgramBinary(p,len,&got,&fmt,bb);
      char path[160]; snprintf(path,sizeof path,"shader_cache/%s_s%016llx.bin",g_pc_tag,(unsigned long long)key);
      if(got>0){ FILE* f=fopen(path,"wb"); if(f){ u32 fm=fmt, ln=(u32)got; fwrite(&fm,4,1,f); fwrite(&ln,4,1,f); fwrite(bb,1,got,f); fclose(f); } } free(bb); } }
  free(j); s->sp=p; s->sps= p? 2 : 3; sh_uloc(s); if(p) pack_add_sh(key,s); return 1; }

static void sep_init(void){
  g_sep=0; if(getenv("SWR_NO_SEP") || !p_glProgramParameteri) goto out;
  p_ups=(PFN_UPS_)backend_gl_proc("glUseProgramStages"); p_bpp=(PFN_BPP_)backend_gl_proc("glBindProgramPipeline"); PFN_GPP_ gpp=(PFN_GPP_)backend_gl_proc("glGenProgramPipelines");
  if(!p_ups || !p_bpp || !gpp) goto out;
  gpp(1,&g_pipe); if(g_pipe) g_sep=1;
out: LOG("shaders: %s\n",g_sep? "one program per shader (each shader is compiled only once)" : "one program per combination"); }

typedef struct Prog { Sh* vs; Sh* fs; GLuint p; GLint vp_loc, at_loc; int pending; WJob* job; struct Prog* next; } Prog;

static Prog* progs[512];

static Prog* program_get(Sh* vs, Sh* fs){
  u32 k=(u32)((((size_t)vs)>>4)^(((size_t)fs)>>6))&511;
  for(Prog* p=progs[k];p;p=p->next) if(p->vs==vs && p->fs==fs) return p;
  Prog* p=calloc(1,sizeof *p); p->vs=vs; p->fs=fs;
  if(g_worker==1 && vs && fs && !vs->failed && !fs->failed && vs->src && fs->src && !prog_cached(vs,fs)){
    WJob* j=calloc(1,sizeof *j); j->vs=vs->src; j->fs=fs->src; j->key=prog_key(vs,fs); j->keep=1;
    if(wr_push(&wq_hi,j)){ p->job=j; p->next=progs[k]; progs[k]=p; return p; } free(j); }
  p->p=prog_build(vs,fs,&p->pending);
  if(!p->pending){ p->vp_loc= p->p? glGetUniformLocation(p->p,"swr_vp") : -1; p->at_loc= p->p? glGetUniformLocation(p->p,"swr_at") : -1; }
  p->next=progs[k]; progs[k]=p; return p; }

/* 1 = ready, 0 = driver still compiling (skip the draw) */
static int program_ready(Prog* p){
  if(p->job){ int s=__atomic_load_n(&p->job->state,__ATOMIC_ACQUIRE); if(!s) return 0;
    WJob* j=p->job; p->job=NULL;
    if(s==1){ p->p=j->prog; pack_add(j->key,p->vs,p->fs); p->vp_loc=j->loc[0]; p->at_loc=j->loc[1]; }
    else { p->p=0; WARN("GL link (thread): %s\n",j->log);
      static int nd; if(nd<sw2_cap(12)){ nd++; plat_mkdir("captures"); plat_mkdir("captures/failed"); char fp[200];
        snprintf(fp,sizeof fp,"captures/failed/%016llx_vs.glsl",(unsigned long long)p->vs->hash); FILE* f=fopen(fp,"w"); if(f){ fputs(p->vs->src,f); fclose(f); }
        snprintf(fp,sizeof fp,"captures/failed/%016llx_fs.glsl",(unsigned long long)p->fs->hash); f=fopen(fp,"w"); if(f){ fputs(p->fs->src,f); fclose(f); } } }
    free(j); return 1; }
  if(!p->pending) return 1;
  GLint done=0; glGetProgramiv(p->p,0x91B1 /* GL_COMPLETION_STATUS_KHR */,&done); if(!done) return 0;
  p->pending=0; p->p=prog_finish(p->vs,p->fs,p->p);
  p->vp_loc= p->p? glGetUniformLocation(p->p,"swr_vp") : -1; p->at_loc= p->p? glGetUniformLocation(p->p,"swr_at") : -1; return 1; }

/* ============================================================ framebuffers */
static GLuint g_fbo, g_vao, g_vbuf[32], g_ibuf, g_ubuf[5][18], g_scratch_vao;
typedef struct { u64 cb[5][18]; u32 cbsize[5][18]; } CbState;

static CbState g_cb;

/* constant buffer bindings are per channel on hw (every NVN queue has its own 3D state). tomodachi life opens 3
   more channels and their queue setup binds bank 2 (the texture handles) to their own empty table every frame,
   with one global set the main channel read ffffffff for every texture. g_cb holds the set of the channel whose
   draw is being set up */
typedef struct { u32 id; int used; CbState s; } ChanCb;
static ChanCb g_chcb[16]; static int g_cb_owner=-1;

static ChanCb* chan_cb(GpuChannel* ch){
  int fr=-1; for(int i=0;i<16;i++){ if(g_chcb[i].used && g_chcb[i].id==ch->id) return &g_chcb[i]; if(!g_chcb[i].used && fr<0) fr=i; }
  if(fr<0) return &g_chcb[15];   /* more than 16 channels: share the last one */
  g_chcb[fr].used=1; g_chcb[fr].id=ch->id; memset(&g_chcb[fr].s,0,sizeof g_chcb[fr].s); return &g_chcb[fr]; }

static int getenv_cached_sharedcb(void){ static int v=-1; if(v<0) v=getenv("SWR_SHARED_CB")!=NULL; return v; }

static void cb_switch(GpuChannel* ch){
  if(g_cb_owner==(int)ch->id) return;
  static int said; if(g_cb_owner>=0 && !said){ said=1; LOG("gpu: several channels draw with 3D state; constant buffer bindings are kept per channel\n"); }
  ChanCb* c=chan_cb(ch); memcpy(&g_cb,&c->s,sizeof g_cb); g_cb_owner=(int)ch->id; }

static void sh_dump_cbs(u64 h){ char fp[200]; FILE* f;
      for(int ci=0;ci<18;ci++){ u64 ca=g_cb.cb[g_sh_bind][ci]; if(!ca) continue; u32 csz=g_cb.cbsize[g_sh_bind][ci]; if(!csz||csz>0x10000) csz=0x10000;
        u8* cbb=malloc(csz); gmem_read(ca,cbb,csz); snprintf(fp,sizeof fp,"captures/failed/%016llx_cb%d.bin",(unsigned long long)h,ci); f=fopen(fp,"wb"); if(f){ fwrite(cbb,1,csz,f); fclose(f); } free(cbb); }
}

static int sh_cb_read(u32 idx, u32 off, u32* out){ if(idx>=18) return 0; u64 a=g_cb.cb[g_sh_bind][idx]; u32 sz=g_cb.cbsize[g_sh_bind][idx];
  if(!a || off+4>(sz? sz : 65536)) return 0; gmem_read(a+off,out,4); return 1; }

static Tex* g_rt[8]; static Tex* g_zt; static int g_nrt;

/* RT memory signature, if the cpu or DMA overwrote it something else lives there now */
static Tex* rt_texture(u64 gva, u32 w, u32 h, u32 fmt, u32 tile, int depth){
  GlFmt f= rt_format(fmt);
  u32 fkey=fmt|(depth? 0x10000 : 0);
  for(Tex* t=texh[hkey(gva)];t;t=t->next) if(t->gva==gva && t->is_rt && t->w==w && t->h==h && t->fkey==fkey){ t->last_used=g_frame; return t; }
  /* games reuse the same memory for RTs of different size/format inside a frame (SMO does it every frame). instead
     of destroying the old one we park it as an alias and reuse it when that size comes back, recreating it meant a
     GL sync plus a deswizzle every time */
  Tex* rv=NULL;
  for(Tex* t=texh[hkey(gva)];t;t=t->next) if(t->gva==~0ull && t->alias==gva && t->is_rt && t->w==w && t->h==h && t->fkey==fkey && !rv) rv=t;
  for(Tex* t=texh[hkey(gva)];t;t=t->next) if(t->gva==gva && t->is_rt){ t->gva=~0ull; t->alias=gva; g_tmap_gen++; }
  if(rv && !getenv("SWR_NO_RT_ALIAS")){ rv->gva=gva; rv->alias=0; rv->last_used=g_frame; rv->hash=rt_sig(gva); rv->frame_checked=g_frame; return rv; }
  Tex* t=tex_create(gva,w,h,1,1,GL_TEXTURE_2D,f,fkey,!((tile>>12)&1),(tile>>4)&15,(tile>>8)&15,0,1);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  /* seed from memory in case the CPU wrote it */
  if(!depth && !f.compressed && f.fmt==GL_RGBA && f.bpp==4){
    u64 sz=(u64)w*h*4; u8* lin=malloc(sz); u8* src=gmem_ptr(gva,sz*2);
    if(src && lin && t->bl){ deswizzle(lin,w*4,src,w*4,h,t->bh); glBindTexture(GL_TEXTURE_2D,t->id); glTexSubImage2D(GL_TEXTURE_2D,0,0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,lin); }
    free(lin); }
  t->hash=rt_sig(gva);
  return t; }

/* 3D RT (tile mode bit 16) or array (layers at 0x206), drawn in one pass with the VS picking the layer or slice by
   slice (base layer at 0x208). it's backed by a 3D/array texture that 3D/array TICs at the same address use
   directly (32^3 LUTs) */
static u64 st_rt_vol, st_waited; int g_draw_layered;

static Tex* rt_vol(Tex* t, u32 dep, int is3d){
  GLenum tg= is3d? GL_TEXTURE_3D : GL_TEXTURE_2D_ARRAY; Tex* v=t->vol; u32 sw=TW(t), sh=TH(t);
  if(v && (v->d!=dep || v->target!=tg || v->pitch!=sw || v->size!=sh)){ glDeleteTextures(1,&v->id); if(v->view) glDeleteTextures(1,&v->view); free(v); v=t->vol=NULL; g_fb_sig_ok=0; g_tmap_gen++; mb_reset(); }
  if(!v){ v=calloc(1,sizeof *v); v->gva=t->gva; v->w=t->w; v->h=t->h; v->d=dep; v->levels=1; v->target=tg; v->f=t->f; v->fkey=t->fkey; v->is_rt=1; v->scale=TS(t); v->gpu_written=1;
    v->bl=t->bl; v->bh=t->bh; v->pitch=sw; v->size=sh; v->swz[0]=GL_RED; v->swz[1]=GL_GREEN; v->swz[2]=GL_BLUE; v->swz[3]=GL_ALPHA;
    glGenTextures(1,&v->id); glBindTexture(tg,v->id); glTexStorage3D(tg,1,t->f.ifmt,sw,sh,dep);
    glTexParameteri(tg,GL_TEXTURE_MAX_LEVEL,0); glTexParameteri(tg,GL_TEXTURE_MIN_FILTER,GL_LINEAR); glTexParameteri(tg,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    t->vol=v; st_rt_vol++;
    static int said; if(said<sw2_cap(8)){ said++; LOG("gpu: %s render target %ux%ux%u at %llx (fmt %x)\n",is3d? "3D" : "array",t->w,t->h,dep,(unsigned long long)t->gva,t->fkey&0xFFFF); } }
  return v; }

/* current g_fbo attachments, skip reattaching if nothing changed since every change makes the driver revalidate */
/* ============================================================ GPU watchdog
   every pass (render target change, compute dispatch, present) gets a number and a fence on the GL thread. the stall
   dump asks the GL thread to check them and names the first pass the GPU never finished, with its shaders. a shader
   that runs into our loop limit on every pixel takes seconds on a small GPU, and odyssey and tomodachi froze
   right when their 3D started on an intel HD */
#define WD_N 64
static struct { u64 gva, cs; u32 w, h, fmt, draws, nk, frame; u64 vs[4], fs[4], t0; } g_wd[WD_N];
static u32 g_wd_n; static u64 g_wd_d0; static int g_wd_off=-1;

static void wd_pass(Tex* t, u64 cs){
  if(g_wd_off<0) g_wd_off= getenv("SWR_NO_GPU_WATCH")!=NULL || !glq_active();
  if(g_wd_off) return;
  if(g_wd_n) g_wd[g_wd_n%WD_N].draws=(u32)(st_drawn-g_wd_d0);
  u32 id=++g_wd_n; __typeof__(g_wd[0])* p=&g_wd[id%WD_N]; memset(p,0,sizeof *p);
  p->gva= t? t->gva : 0; p->w= t? t->w : 0; p->h= t? t->h : 0; p->fmt= t? t->fkey&0xFFFF : 0; p->cs=cs; p->frame=g_frame; p->t0=plat_ns();
  g_wd_d0=st_drawn; glq_wd_mark(id); }

static void wd_note(u64 vs, u64 fs){
  if(g_wd_off || !g_wd_n) return; __typeof__(g_wd[0])* p=&g_wd[g_wd_n%WD_N];
  for(u32 i=0;i<p->nk;i++) if(p->vs[i]==vs && p->fs[i]==fs) return;
  if(p->nk<4){ p->vs[p->nk]=vs; p->fs[p->nk]=fs; p->nk++; } }

static void wd_line(u32 id, u64 now){
  __typeof__(g_wd[0])* p=&g_wd[id%WD_N]; u32 dr= id==g_wd_n? (u32)(st_drawn-g_wd_d0) : p->draws;
  if(p->cs) LOG("    pass %u (frame %u): compute %016llx",id,p->frame,(unsigned long long)p->cs);
  else if(p->gva) LOG("    pass %u (frame %u): rt %llx %ux%u f%x, %u draws",id,p->frame,(unsigned long long)p->gva,p->w,p->h,p->fmt,dr);
  else LOG("    pass %u (frame %u): present/copies",id,p->frame);
  LOG(", sent %.1f s ago\n",(now-p->t0)/1e9);
  for(u32 i=0;i<p->nk;i++) LOG("      vs=%016llx fs=%016llx\n",(unsigned long long)p->vs[i],(unsigned long long)p->fs[i]); }

static void wd_report(void){
  if(g_wd_off || !g_wd_n) return;
  u64 now=plat_ns(); u32 last=g_wd_n;
  if(!glq_wd_probe()){ LOG("  GPU progress: the GL thread is inside the driver, it can't check the GPU. last passes sent:\n");
    for(u32 id= last>3? last-2 : 1; id<=last; id++) wd_line(id,now); return; }
  /* fence k signals when everything before pass k is done: the first one not signaled is right after the pass the
     GPU is still on */
  u32 lo= last>WD_N-2? last-(WD_N-2) : 1, first=0;
  for(u32 id=lo; id<=last; id++){ int d=glq_wd_done(id); if(d==0){ first=id; break; } }
  if(!first){ LOG("  GPU progress: the GPU finished everything sent to it (the stop is on the CPU side)\n"); return; }
  if(first==lo){ LOG("  GPU progress: the GPU is more than %d passes behind, oldest one still running:\n",WD_N-2); wd_line(first,now); return; }
  LOG("  GPU progress: the GPU has not finished this pass:\n"); wd_line(first-1,now);
  if(first<=last){ LOG("    then %u more pass(es) waiting, the next ones:\n",last-first+1); for(u32 id=first; id<=last && id<first+3; id++) wd_line(id,now); } }

/* ============================================================ gpu time per pass
   every 5 s one frame gets a GL_TIME_ELAPSED query per render target change, read back at the next print, and the
   passes that cost the most go in performance.txt. copies and compute in between count for the pass after them */
typedef void (APIENTRY*PFN_GQ_)(GLsizei,GLuint*); typedef void (APIENTRY*PFN_BQ_)(GLenum,GLuint); typedef void (APIENTRY*PFN_EQ_)(GLenum); typedef void (APIENTRY*PFN_QR_)(GLuint,GLenum,GLuint64*);
static PFN_GQ_ p_genq; static PFN_BQ_ p_begq; static PFN_EQ_ p_endq; static PFN_QR_ p_qres;
#define GP_N 4096
static GLuint g_gp_q[GP_N]; static struct { u64 gva; u32 w, h, fmt, draws; u64 cs; u32 grid, mb; u64 sh; } g_gp_p[GP_N];   /* sh: SWR_GP_DRAW=1, one entry per draw (its fragment shader) */   /* cs: a compute dispatch (its key) */
static int g_gp_n, g_gp_state;   /* 0 idle, 1 armed, 2 timing this frame, 3 waiting for the print, -1 off */
static u64 g_gp_d0;

static void gp_pass(Tex* t){
  if(g_gp_state!=2) return;
  if(g_gp_n){ p_endq(0x88BF /* GL_TIME_ELAPSED */); g_gp_p[g_gp_n-1].draws=(u32)(st_drawn-g_gp_d0); }
  if(g_gp_n>=GP_N){ g_gp_state=3; return; }
  g_gp_p[g_gp_n].gva= t? t->gva : 0; g_gp_p[g_gp_n].w= t? t->w : 0; g_gp_p[g_gp_n].h= t? t->h : 0; g_gp_p[g_gp_n].fmt= t? t->fkey : 0; g_gp_p[g_gp_n].cs=0; g_gp_p[g_gp_n].sh=0; g_gp_d0=st_drawn;
  p_begq(0x88BF,g_gp_q[g_gp_n]); g_gp_n++; }

/* a compute dispatch gets its own entry, then the pass it interrupted goes on (tomodachi's 3D map: a compute took
   seconds and it showed up as a 256x256 pass with 2 draws) */
static struct { int on; u64 gva; u32 w, h, fmt; } g_gp_cs;
static void gp_compute_begin(u64 key, u32 grid, u32 mb){
  g_gp_cs.on=0; if(g_gp_state!=2 || g_gp_n>=GP_N-1) return;
  g_gp_cs.gva=0; g_gp_cs.w=g_gp_cs.h=g_gp_cs.fmt=0;
  if(g_gp_n){ g_gp_cs.gva=g_gp_p[g_gp_n-1].gva; g_gp_cs.w=g_gp_p[g_gp_n-1].w; g_gp_cs.h=g_gp_p[g_gp_n-1].h; g_gp_cs.fmt=g_gp_p[g_gp_n-1].fmt; }
  gp_pass(NULL); if(g_gp_state!=2) return;
  g_gp_p[g_gp_n-1].cs=key; g_gp_p[g_gp_n-1].grid=grid; g_gp_p[g_gp_n-1].mb=mb; g_gp_cs.on=1; }
static int gp_cs_on(void){ return g_gp_cs.on; }
/* after the dispatch and its result copies */
static void gp_compute_end(void){
  if(!g_gp_cs.on) return; g_gp_cs.on=0;
  gp_pass(NULL); if(g_gp_state!=2) return;
  g_gp_p[g_gp_n-1].gva=g_gp_cs.gva; g_gp_p[g_gp_n-1].w=g_gp_cs.w; g_gp_p[g_gp_n-1].h=g_gp_cs.h; g_gp_p[g_gp_n-1].fmt=g_gp_cs.fmt; }

/* present: the armed frame starts here, the timed one ends here */
static void gp_frame_end(void){
  if(g_gp_state==2){ if(g_gp_n){ p_endq(0x88BF); g_gp_p[g_gp_n-1].draws=(u32)(st_drawn-g_gp_d0); } g_gp_state=3; }
  else if(g_gp_state==1){ g_gp_state=2; g_gp_n=0; Tex* t=NULL; for(int i=0;i<g_nrt && !t;i++) t=g_rt[i]; gp_pass(t? t : g_zt); } }

static void gp_report(FILE* rf){
  if(g_gp_state==0 || g_gp_state==3){
    /* only once the GPU is done with that frame: GL_QUERY_RESULT waits for it, and with a GPU stuck for seconds
       the GPU thread sat in that wait instead of reporting */
    if(g_gp_state==3 && g_gp_n){ GLuint64 av=0; p_qres(g_gp_q[g_gp_n-1],0x8867 /* GL_QUERY_RESULT_AVAILABLE */,&av); if(!av) return; }
    if(g_gp_state==3 && rf && g_gp_n){
      static struct { u64 gva; u32 w, h, fmt, draws, n; u64 ns; u64 cs; u32 grid, mb; u64 sh; } a[GP_N]; int na=0; u64 tot=0;
      for(int i=0;i<g_gp_n;i++){ GLuint64 ns=0; p_qres(g_gp_q[i],0x8866 /* GL_QUERY_RESULT */,&ns); tot+=ns; int k=0;
        for(;k<na;k++) if(a[k].cs==g_gp_p[i].cs && a[k].sh==g_gp_p[i].sh && (g_gp_p[i].cs || g_gp_p[i].sh || (a[k].gva==g_gp_p[i].gva && a[k].w==g_gp_p[i].w && a[k].h==g_gp_p[i].h && a[k].fmt==g_gp_p[i].fmt))) break;
        if(k==na){ a[na].gva=g_gp_p[i].gva; a[na].w=g_gp_p[i].w; a[na].h=g_gp_p[i].h; a[na].fmt=g_gp_p[i].fmt; a[na].cs=g_gp_p[i].cs; a[na].sh=g_gp_p[i].sh; a[na].draws=0; a[na].n=0; a[na].ns=0; a[na].grid=0; a[na].mb=0; na++; }
        a[k].ns+=ns; a[k].draws+=g_gp_p[i].draws; a[k].n++; if(g_gp_p[i].grid>a[k].grid) a[k].grid=g_gp_p[i].grid; if(g_gp_p[i].mb>a[k].mb) a[k].mb=g_gp_p[i].mb; }
      fprintf(rf,"      GPU time of one frame: %.1f ms in %d passes. most expensive:",tot/1e6,g_gp_n);
      for(int j=0;j<(getenv("SWR_GP_DRAW")? 16 : 8) && j<na;j++){ int b=j; for(int k=j+1;k<na;k++) if(a[k].ns>a[b].ns) b=k; if(b!=j){ __typeof__(a[0]) x=a[j]; a[j]=a[b]; a[b]=x; }
        if(a[j].cs==GP_TEXUP_KEY) fprintf(rf," | texture uploads x%u: %.1f ms",a[j].n,a[j].ns/1e6);
        else if(a[j].cs) fprintf(rf," | compute %016llx x%u (%u groups, %u MB of buffers): %.1f ms",(unsigned long long)a[j].cs,a[j].n,a[j].grid,a[j].mb,a[j].ns/1e6);
        else if(a[j].sh) fprintf(rf," | fs %016llx on %llx %ux%u x%u: %.1f ms",(unsigned long long)a[j].sh,(unsigned long long)a[j].gva,a[j].w,a[j].h,a[j].n,a[j].ns/1e6);
        else if(!a[j].gva) fprintf(rf," | no render target (copies, present): %.1f ms",a[j].ns/1e6);
        else fprintf(rf," | rt %llx %ux%u f%x: %.1f ms, %u draws",(unsigned long long)a[j].gva,a[j].w,a[j].h,a[j].fmt&0xFFFF,a[j].ns/1e6,a[j].draws); }
      fprintf(rf,"\n"); }
    if(!p_genq){ p_genq=(PFN_GQ_)backend_gl_proc("glGenQueries"); p_begq=(PFN_BQ_)backend_gl_proc("glBeginQuery"); p_endq=(PFN_EQ_)backend_gl_proc("glEndQuery"); p_qres=(PFN_QR_)backend_gl_proc("glGetQueryObjectui64v");
      if(!p_genq || !p_begq || !p_endq || !p_qres || getenv("SWR_NO_GPU_PROF")){ g_gp_state=-1; return; } p_genq(GP_N,g_gp_q); }
    g_gp_state=1; } }

static GLuint g_fb_sig[10]; static int g_fb_lay[8]; static int g_fb_sig_ok; static u64 st_fb_skip; static int g_clear_layer=-1;

static void bind_framebuffer(GpuChannel* ch){
  u32* r=ch->r3d;
  u32 ctl=r[0x487]; u32 count=ctl&15; if(count>8) count=8;
  int n=0; g_nrt=0;
  glBindFramebuffer(GL_FRAMEBUFFER,g_fbo);
  GLuint sig[10]; int lay[8]; int anylay=0;
  for(u32 i=0;i<8;i++){
    Tex* t=NULL; lay[i]=-2;
    if(i<count){ u32 idx=(ctl>>(4+i*3))&7; u32* rt=&r[0x200+idx*0x10]; u64 a=((u64)rt[0]<<32)|rt[1];
      { static int dbg=-1; if(dbg<0) dbg=getenv("SWR_RTDBG")!=0; if(dbg) LOG("rtdbg %d: %llx %ux%u fmt=%x tile=%x arr=%x stride=%x base=%x\n",i,(unsigned long long)a,rt[2],rt[3],rt[4],rt[5],rt[6],rt[7],rt[8]); }
      if(a && rt[4] && rt[2] && rt[3]) t=rt_texture(a,rt[2],rt[3],rt[4],rt[5],0);
      if(a && watch_gva(a)) LOG("watch %llx: render target %d, %ux%u fmt %x layer %u depth %u (frame %u)\n",(unsigned long long)a,i,rt[2],rt[3],rt[4],rt[8]&0xFFFF,rt[6]&0xFFFF,g_frame);
      u32 dep=rt[6]&0xFFFF; int is3d=(rt[5]>>16)&1;
      if(t && !t->f.depth && dep>1 && dep<=2048 && !getenv("SWR_NO_RTVOL")){ Tex* v=rt_vol(t,dep,is3d);
        if(v){ v->wgen++; v->last_used=g_frame;
          if(g_clear_layer>=0) lay[i]=(int)(rt[8]&0xFFFF)+g_clear_layer; else lay[i]= g_draw_layered? -1 : (int)(rt[8]&0xFFFF);
          if(lay[i]>=(int)dep) lay[i]=(int)dep-1; if(lay[i]==-1) anylay=1;
          g_rt[i]=t; sig[i]=v->id; n=i+1; t->gpu_written=1; t->wgen++; continue; } } }
    g_rt[i]=t; sig[i]= t? t->id : 0;
    if(t){ n=i+1; t->gpu_written=1; t->wgen++; }
  }
  g_nrt=n;
  g_zt=NULL;
  if(r[0x54E]){ u64 za=((u64)r[0x3F8]<<32)|r[0x3F9]; u32 zw=r[0x48A], zh=r[0x48B];
    if(za && zw && zh) g_zt=rt_texture(za,zw,zh,r[0x3FA],r[0x3FB],1); }
  if(anylay && g_zt){ g_zt=NULL; static int said; if(!said){ said=1; LOG("gpu: layered draw with depth: drawn without depth\n"); } }
  if(g_zt) g_zt->wgen++;
  { Tex* target=NULL;
    for(int i=0;i<g_nrt && !target;i++) target=g_rt[i];
    if(!target) target=g_zt;
    float ds=target? TS(target) : g_scale;
    if(ds!=g_draw_scale){ g_draw_scale=ds; rs_update(); }
    if(target) ra_update(target->w,target->h);
  }
  GLenum att= g_zt? (g_zt->f.fmt==GL_DEPTH_STENCIL? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT) : GL_DEPTH_STENCIL_ATTACHMENT;
  sig[8]= g_zt? g_zt->id : 0; sig[9]=att;
  if(g_fb_sig_ok && !memcmp(sig,g_fb_sig,sizeof sig) && !memcmp(lay,g_fb_lay,sizeof lay)){ st_fb_skip++; return; }
  g_tmap_gen++;
  GLenum bufs[8];
  for(u32 i=0;i<8;i++){ if(!g_fb_sig_ok || g_fb_sig[i]!=sig[i] || g_fb_lay[i]!=lay[i]){
      if(lay[i]==-2) glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0+i,GL_TEXTURE_2D,sig[i],0);
      else if(lay[i]==-1) glFramebufferTexture(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0+i,sig[i],0);
      else glFramebufferTextureLayer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0+i,sig[i],0,lay[i]); }
    bufs[i]= sig[i]? GL_COLOR_ATTACHMENT0+i : GL_NONE; }
  memcpy(g_fb_lay,lay,sizeof lay);
  { static int dbg=-1; if(dbg<0) dbg=getenv("SWR_RTDBG")!=0; if(dbg){ GLenum st=glCheckFramebufferStatus(GL_FRAMEBUFFER); LOG("rtdbg fbo %x lay %d %d layered=%d err=%x\n",st,lay[0],lay[1],g_draw_layered,glGetError()); } }
  glDrawBuffers(8,bufs);
  if(!g_fb_sig_ok || g_fb_sig[8]!=sig[8] || g_fb_sig[9]!=sig[9]){
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_STENCIL_ATTACHMENT,GL_TEXTURE_2D,0,0);
    glFramebufferTexture2D(GL_FRAMEBUFFER,att,GL_TEXTURE_2D,sig[8],0); }
  memcpy(g_fb_sig,sig,sizeof sig); g_fb_sig_ok=1;
  { Tex* t=NULL; for(int i=0;i<g_nrt && !t;i++) t=g_rt[i]; if(!t) t=g_zt; wd_pass(t,0); if(g_gp_state==2) gp_pass(t); }
}

/* ============================================================ RT <-> memory, GPU 2D blits */
typedef struct { u32 format, linear, bh, depth, layer, pitch, width, height; u64 addr; } Surf2Dg;

static GLuint g_rfbo, g_dfbo;

static int rt_plain(Tex* t){ return t && !t->f.depth && !t->f.is_int && t->f.fmt==GL_RGBA && t->f.type==GL_UNSIGNED_BYTE && t->f.bpp==4; }

static void trace_note(const char* fmt, ...);
/* write RT contents back to guest memory for cpu side DMA/2D */
static void state_dirty(void);

/* res scale != 1: GPU copy between the scaled RT and native size */
static GLuint rt_unscaled_copy(Tex* t, int to_real){
  GLuint tmp=0; glGenTextures(1,&tmp); glBindTexture(GL_TEXTURE_2D,tmp); glTexStorage2D(GL_TEXTURE_2D,1,t->f.ifmt,t->w,t->h);
  if(to_real){ glBindFramebuffer(GL_READ_FRAMEBUFFER,g_rfbo); glFramebufferTexture2D(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,t->id,0); glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER,g_dfbo); glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tmp,0); { GLenum b=GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&b); }
    glDisable(GL_SCISSOR_TEST); glDisable(GL_FRAMEBUFFER_SRGB); glColorMaski(0,1,1,1,1); state_dirty();
    glBlitFramebuffer(0,0,(GLint)TW(t),(GLint)TH(t),0,0,(GLint)t->w,(GLint)t->h,GL_COLOR_BUFFER_BIT,GL_LINEAR); }
  return tmp; }

static void rt_scaled_put(Tex* t, GLuint tmp){
  glBindFramebuffer(GL_READ_FRAMEBUFFER,g_rfbo); glFramebufferTexture2D(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tmp,0); glReadBuffer(GL_COLOR_ATTACHMENT0);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER,g_dfbo); glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,t->id,0); { GLenum b=GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&b); }
  glDisable(GL_SCISSOR_TEST); glDisable(GL_FRAMEBUFFER_SRGB); glColorMaski(0,1,1,1,1); state_dirty();
  glBlitFramebuffer(0,0,(GLint)t->w,(GLint)t->h,0,0,(GLint)TW(t),(GLint)TH(t),GL_COLOR_BUFFER_BIT,GL_LINEAR);
  glBindFramebuffer(GL_FRAMEBUFFER,g_fbo); }

/* any uncompressed color RT (R16F heightmaps, RG16F flow maps...), games render these once and DMA them into the
   final texture */
static int rt_flushable(Tex* t){ return t && !t->f.depth && !t->f.compressed && t->f.bpp>0 && t->f.bpp<=16 && t->target==GL_TEXTURE_2D && !t->vol; }

/* games that read a render target back every frame (auto exposure, mario tennis aces' 16x1 target) would make the
   GPU thread wait for the graphics card every single time, cpu and gpu taking turns. once the same target gets
   read 3 frames in a row we read it into a buffer and give the game the newest copy that's already done (1 or 2
   frames old). one-off copies (a heightmap rendered once and copied) still wait and stay exact.
   SWR_SYNC_READBACK=1 turns it off */
#define RBN 3
typedef struct { u64 gva; u32 w, h, bp, fkey, streak, last_fr; GLuint pbo[RBN]; const u8* map[RBN]; GLsync fs[RBN]; int cur; } RBack;
static RBack g_rb[64];
static int g_rb_off=-1;
static PFNGLBUFFERSTORAGEPROC p_rb_bs; static PFNGLMAPBUFFERRANGEPROC p_rb_map; static PFNGLFENCESYNCPROC p_rb_fs; static PFNGLCLIENTWAITSYNCPROC p_rb_cws; static PFNGLDELETESYNCPROC p_rb_ds;

static void rb_free(RBack* rb){
  for(int i=0;i<RBN;i++) if(rb->fs[i]){ p_rb_ds(rb->fs[i]); rb->fs[i]=0; }
  if(rb->pbo[0]){ glBindBuffer(GL_PIXEL_PACK_BUFFER,0); glDeleteBuffers(RBN,rb->pbo); }
  memset(rb,0,sizeof *rb); }

static void rb_store(Tex* t, u64 gva, const u8* px){
  u32 w=t->w, h=t->h, bp=t->f.bpp;
  u64 sz= t->bl? bl_level_size(w,h,1,&t->f,t->bh,0,0,NULL) : (u64)w*h*bp; u8* dst=gmem_ptr(gva,sz);
  if(dst){ if(t->bl) swizzle(dst,px,w*bp,w*bp,h,t->bh); else memcpy(dst,px,(size_t)w*h*bp); } }

static int rb_async(Tex* t, u64 gva){
  if(g_rb_off<0){ g_rb_off= getenv("SWR_SYNC_READBACK")!=NULL;
    if(!g_rb_off){ p_rb_bs=(PFNGLBUFFERSTORAGEPROC)backend_gl_proc("glBufferStorage"); p_rb_map=(PFNGLMAPBUFFERRANGEPROC)backend_gl_proc("glMapBufferRange");
      p_rb_fs=(PFNGLFENCESYNCPROC)backend_gl_proc("glFenceSync"); p_rb_cws=(PFNGLCLIENTWAITSYNCPROC)backend_gl_proc("glClientWaitSync"); p_rb_ds=(PFNGLDELETESYNCPROC)backend_gl_proc("glDeleteSync");
      if(!p_rb_bs || !p_rb_map || !p_rb_fs || !p_rb_cws || !p_rb_ds) g_rb_off=1; } }
  u32 w=t->w, h=t->h, bp=t->f.bpp; u64 size=(u64)w*h*bp;
  if(g_rb_off || size>(4u<<20)) return 0;
  RBack* rb=NULL;
  for(int i=0;i<64;i++) if(g_rb[i].gva==gva){ rb=&g_rb[i]; break; }
  if(!rb){ rb=&g_rb[0]; for(int i=1;i<64;i++) if(g_rb[i].last_fr<rb->last_fr) rb=&g_rb[i]; rb_free(rb); rb->gva=gva; }
  if(rb->w!=w || rb->h!=h || rb->bp!=bp || rb->fkey!=t->fkey){ rb_free(rb); rb->gva=gva; rb->w=w; rb->h=h; rb->bp=bp; rb->fkey=t->fkey; }
  if(rb->streak && (u32)g_frame-rb->last_fr<=1) rb->streak++; else if(rb->last_fr!=(u32)g_frame || !rb->streak) rb->streak=1;
  rb->last_fr=(u32)g_frame;
  if(rb->streak<3) return 0;
  if(!rb->pbo[0]){
    glGenBuffers(RBN,rb->pbo);
    for(int i=0;i<RBN;i++){ glBindBuffer(GL_PIXEL_PACK_BUFFER,rb->pbo[i]);
      p_rb_bs(GL_PIXEL_PACK_BUFFER,(GLsizeiptr)size,NULL,GL_MAP_READ_BIT|GL_MAP_PERSISTENT_BIT|GL_MAP_COHERENT_BIT);
      rb->map[i]=(const u8*)p_rb_map(GL_PIXEL_PACK_BUFFER,0,(GLsizeiptr)size,GL_MAP_READ_BIT|GL_MAP_PERSISTENT_BIT|GL_MAP_COHERENT_BIT); }
    glBindBuffer(GL_PIXEL_PACK_BUFFER,0);
    for(int i=0;i<RBN;i++) if(!rb->map[i]){ rb_free(rb); g_rb_off=1; LOG("gpu: persistent pack buffers unavailable, render target readbacks stay synchronous\n"); return 0; }
    LOG("gpu: rt %llx %ux%u is read back every frame: now asynchronous (the game gets a copy a frame or two old)\n",(unsigned long long)gva,w,h); }
  /* queue this frame's read. if the buffer we're about to reuse still has a copy nobody got (none finished in
     time), wait for it and hand it over, that keeps the data at most RBN-1 frames old */
  int k=rb->cur;
  if(rb->fs[k]){ GLenum r=p_rb_cws(rb->fs[k],GL_SYNC_FLUSH_COMMANDS_BIT,100000000ull);
    if(r==GL_ALREADY_SIGNALED || r==GL_CONDITION_SATISFIED) rb_store(t,gva,rb->map[k]);
    p_rb_ds(rb->fs[k]); rb->fs[k]=0; }
  GLuint tmp= TS(t)!=1.0f? rt_unscaled_copy(t,1) : 0;
  glBindFramebuffer(GL_READ_FRAMEBUFFER,g_rfbo); glFramebufferTexture2D(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tmp? tmp : t->id,0); glReadBuffer(GL_COLOR_ATTACHMENT0);
  glPixelStorei(GL_PACK_ALIGNMENT,1); glPixelStorei(GL_PACK_ROW_LENGTH,0);
  glBindBuffer(GL_PIXEL_PACK_BUFFER,rb->pbo[k]);
  if(rt_plain(t)) glReadPixels(0,0,(GLsizei)w,(GLsizei)h,GL_RGBA,GL_UNSIGNED_BYTE,(void*)0);
  else { glBindTexture(GL_TEXTURE_2D,tmp? tmp : t->id); glGetTexImage(GL_TEXTURE_2D,0,t->f.fmt,t->f.type,(void*)0); }
  glBindBuffer(GL_PIXEL_PACK_BUFFER,0);
  rb->fs[k]=p_rb_fs(GL_SYNC_GPU_COMMANDS_COMPLETE,0); rb->cur=(k+1)%RBN;
  glBindFramebuffer(GL_FRAMEBUFFER,g_fbo); if(tmp) glDeleteTextures(1,&tmp); state_dirty();
  /* hand over the newest finished one, the older ones aren't needed anymore */
  for(int n=1;n<RBN;n++){ int j=(k-n+RBN)%RBN; if(!rb->fs[j]) continue;
    GLenum r=p_rb_cws(rb->fs[j],0,0); if(r!=GL_ALREADY_SIGNALED && r!=GL_CONDITION_SATISFIED) continue;
    rb_store(t,gva,rb->map[j]);
    for(int m=n;m<RBN;m++){ int o=(k-m+RBN)%RBN; if(rb->fs[o]){ p_rb_ds(rb->fs[o]); rb->fs[o]=0; } }
    break; }
  t->hash=rt_sig(gva); trace_note("  (rt %llx -> memory, asynchronous)\n",(unsigned long long)gva);
  return 1; }

void gpu_gl_rt_flush(u64 gva){
  Tex* t=tex_find_rt(gva); if(!rt_flushable(t)) return;
  { static int nlog; if(nlog<40 && !rt_plain(t)){ nlog++; LOG("rt %llx %ux%u (fkey %x) copied to memory for a DMA/2D copy\n",(unsigned long long)gva,t->w,t->h,t->fkey); } }
  /* right before a DMA/2D copy reads or overwrites that memory, so it's synchronous: an async readback landing after
     the copy wrote an old RT image over smo's newer video/background pixels. SWR_ASYNC_RT_FLUSH=1 goes back */
  { static int async=-1; if(async<0) async=getenv("SWR_ASYNC_RT_FLUSH")!=NULL; if(async && rb_async(t,gva)) return; }
  u32 w=t->w, h=t->h, bp=t->f.bpp; u8* px=malloc((size_t)w*h*bp); if(!px) return;
  GLuint tmp= TS(t)!=1.0f? rt_unscaled_copy(t,1) : 0;
  glBindFramebuffer(GL_READ_FRAMEBUFFER,g_rfbo); glFramebufferTexture2D(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tmp? tmp : t->id,0); glReadBuffer(GL_COLOR_ATTACHMENT0);
  glPixelStorei(GL_PACK_ALIGNMENT,1); glPixelStorei(GL_PACK_ROW_LENGTH,0);
  if(rt_plain(t)) glReadPixels(0,0,(GLsizei)w,(GLsizei)h,GL_RGBA,GL_UNSIGNED_BYTE,px);
  else { glBindTexture(GL_TEXTURE_2D,tmp? tmp : t->id); glGetTexImage(GL_TEXTURE_2D,0,t->f.fmt,t->f.type,px); }
  glBindFramebuffer(GL_FRAMEBUFFER,g_fbo); if(tmp) glDeleteTextures(1,&tmp); state_dirty();
  u64 sz= t->bl? bl_level_size(w,h,1,&t->f,t->bh,0,0,NULL) : (u64)w*h*bp; u8* dst=gmem_ptr(gva,sz);
  if(dst){ if(t->bl) swizzle(dst,px,w*bp,w*bp,h,t->bh); else memcpy(dst,px,(size_t)w*h*bp); }
  free(px); t->hash=rt_sig(gva); t->gpu_written=0; trace_note("  (rt %llx -> memory, %u bytes/px)\n",(unsigned long long)gva,bp); }

/* CPU copy changed the memory, reload the RT */
void gpu_gl_rt_reload(u64 gva){
  Tex* t=tex_find_rt(gva);
  /* DMA over an RT we can't reload (R8 video luma etc): drop the RT and read memory instead */
  if(t && !rt_plain(t)){ t->gva=~0ull; g_tmap_gen++; trace_note("  (new memory over rt %llx: rt discarded)\n",(unsigned long long)gva); return; }
  if(!rt_plain(t)) return;
  u32 w=t->w, h=t->h; u64 sz= t->bl? bl_level_size(w,h,1,&t->f,t->bh,0,0,NULL) : (u64)w*h*4; u8* src=gmem_ptr(gva,sz); if(!src) return;
  u8* lin=malloc((size_t)w*h*4); if(!lin) return;   /* out of memory: keep what the RT has */
  if(t->bl) deswizzle(lin,w*4,src,w*4,h,t->bh); else memcpy(lin,src,(size_t)w*h*4);
  GLuint tmp= TS(t)!=1.0f? rt_unscaled_copy(t,0) : 0;
  glBindTexture(GL_TEXTURE_2D,tmp? tmp : t->id); glPixelStorei(GL_UNPACK_ALIGNMENT,1); glPixelStorei(GL_UNPACK_ROW_LENGTH,0); glTexSubImage2D(GL_TEXTURE_2D,0,0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,lin);
  if(tmp){ rt_scaled_put(t,tmp); glDeleteTextures(1,&tmp); }
  free(lin); t->hash=rt_sig(gva); t->gpu_written=0; t->wgen++; trace_note("  (memory -> rt %llx)\n",(unsigned long long)gva); }

/* 2D blit from an RT, done on the GPU */
static void state_dirty(void);

int r3d_blit_gl(Surf2Dg* s, Surf2Dg* d, int dx, int dy, int dw, int dh, double sx, double sy, double sdx, double sdy){ state_dirty();
  Tex* st=tex_find_rt(s->addr);
  trace_note("BLIT2D %llx(%ux%u f%x%s) -> %llx(%ux%u f%x) dst(%d,%d %dx%d) src(%.1f,%.1f step %.3f,%.3f)\n",(unsigned long long)s->addr,s->width,s->height,s->format,st?" RT":"",
    (unsigned long long)d->addr,d->width,d->height,d->format,dx,dy,dw,dh,sx,sy,sdx,sdy);
  if(!st) return 0;
  u32 tile=(d->linear? 0x1000u : 0)|(d->bh<<4);
  if(st->f.depth){
    /* a depth buffer copied with the 2D engine (as a 32 bit color surface) to be read as a texture later. sonic
       frontiers does it right after the gbuffer and all its lighting reads that copy: done from memory it was all
       "far", so the lighting came out black. the copy stays on the GPU as a depth RT with the same format */
    if(sdx!=1.0 || sdy!=1.0 || !p_cis) return 0;
    Tex* dt=rt_texture(d->addr,d->width,d->height,st->fkey&0xFFFF,tile,1); if(!dt || dt==st || !dt->f.depth || dt->f.ifmt!=st->f.ifmt) return 0;
    int x0=(int)sx, y0=(int)sy, w=dw, h=dh;
    if(x0<0 || y0<0 || dx<0 || dy<0) return 0;
    if(x0+w>(int)st->w) w=(int)st->w-x0; if(y0+h>(int)st->h) h=(int)st->h-y0;
    if(dx+w>(int)dt->w) w=(int)dt->w-dx; if(dy+h>(int)dt->h) h=(int)dt->h-dy;
    if(w<=0 || h<=0) return 0;
    if(TS(st)!=TS(dt)) return 0;
    p_cis(st->id,GL_TEXTURE_2D,0,(GLint)TD(st,x0),(GLint)TD(st,y0),0,dt->id,GL_TEXTURE_2D,0,(GLint)TD(dt,dx),(GLint)TD(dt,dy),0,(GLsizei)TD(st,w),(GLsizei)TD(st,h),1);
    dt->last_used=g_frame; dt->wgen++; dt->gpu_written=1;
    static int said; if(!said){ said=1; LOG("gpu: depth buffer %llx copied to %llx on the GPU (2D engine copy of a depth RT)\n",(unsigned long long)st->gva,(unsigned long long)d->addr); }
    return 1; }
  Tex* dt=rt_texture(d->addr,d->width,d->height,d->format,tile,0); if(!dt || dt==st || dt->f.depth) return 0;
  if(st->f.is_int!=dt->f.is_int) return 0;
  glBindFramebuffer(GL_READ_FRAMEBUFFER,g_rfbo); glFramebufferTexture2D(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,st->id,0); glReadBuffer(GL_COLOR_ATTACHMENT0);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER,g_dfbo); glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,dt->id,0); { GLenum b=GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&b); }
  glDisable(GL_SCISSOR_TEST); glDisable(GL_FRAMEBUFFER_SRGB); glColorMaski(0,1,1,1,1);
  float S=TS(st), D=TS(dt); int scaled= sdx!=1.0 || sdy!=1.0 || S!=D;
  glBlitFramebuffer((GLint)(sx*S),(GLint)(sy*S),(GLint)((sx+sdx*dw)*S),(GLint)((sy+sdy*dh)*S),(GLint)(dx*D),(GLint)(dy*D),(GLint)((dx+dw)*D),(GLint)((dy+dh)*D),GL_COLOR_BUFFER_BIT,(scaled && !st->f.is_int)? GL_LINEAR : GL_NEAREST);
  glBindFramebuffer(GL_FRAMEBUFFER,g_fbo); dt->last_used=g_frame; dt->wgen++;
  return 1; }

/* ============================================================ fixed state */
static GLenum blend_factor(u32 v){
  if(v&0x4000) return v&~0x4000u;
  switch(v){ case 1: return GL_ZERO; case 2: return GL_ONE; case 3: return GL_SRC_COLOR; case 4: return GL_ONE_MINUS_SRC_COLOR; case 5: return GL_SRC_ALPHA; case 6: return GL_ONE_MINUS_SRC_ALPHA;
    case 7: return GL_DST_ALPHA; case 8: return GL_ONE_MINUS_DST_ALPHA; case 9: return GL_DST_COLOR; case 10: return GL_ONE_MINUS_DST_COLOR; case 11: return GL_SRC_ALPHA_SATURATE;
    case 14: return GL_CONSTANT_COLOR; case 15: return GL_ONE_MINUS_CONSTANT_COLOR; case 16: return GL_SRC1_COLOR; case 17: return GL_ONE_MINUS_SRC1_COLOR; case 18: return GL_SRC1_ALPHA; case 19: return GL_ONE_MINUS_SRC1_ALPHA; }
  return GL_ONE; }

static GLenum blend_eq(u32 v){ if(v>=0x8000) return v; switch(v){ case 1: return GL_FUNC_ADD; case 2: return GL_FUNC_SUBTRACT; case 3: return GL_FUNC_REVERSE_SUBTRACT; case 4: return GL_MIN; case 5: return GL_MAX; } return GL_FUNC_ADD; }

static GLenum cmp_func(u32 v){ if(v>=0x200 && v<=0x207) return v; if(v>=1 && v<=8) return GL_NEVER+(v-1); return GL_ALWAYS; }

static GLenum stencil_op(u32 v){ switch(v){ case 1: return GL_KEEP; case 2: return GL_ZERO; case 3: return GL_REPLACE; case 4: return GL_INCR; case 5: return GL_DECR; case 6: return GL_INVERT; case 7: return GL_INCR_WRAP; case 8: return GL_DECR_WRAP; case 0: return GL_ZERO; } return v; }

/* fixed state signature: if no reg changed since the last draw we skip ~70 GL calls (every call costs on intel's
   driver and we do ~1000 draws a frame) */
static int g_state_ok; static u32 g_state_sig[256]; static u64 st_state_skip;
/* native 0..1 depth (GL 4.5 clip control), remapping z*2-w in the shader loses precision far away */
static PFNGLCLIPCONTROLPROC p_clip; static int g_clip_mode=-1;
static int g_gs_ok;

static void state_dirty(void){ g_state_ok=0; g_gs_ok=0; }

static int getenv_cached_noorigin(void){ static int v=-1; if(v<0) v= getenv("SWR_NO_ORIGIN")? 1 : 0; return v; }

/* WINDOW_ORIGIN (0x4EB) bit 0: window Y counts from the bottom of the target. almost every game uses the top, but
   games on the switch's OpenGL driver (deltarune, dead cells) draw their final pass with the bottom one. returns
   the target height when the flip applies, 0 if not */
static u32 origin_flip_h(const u32* r){
  if(!(r[0x4EB]&1) || getenv_cached_noorigin()) return 0;
  if(g_nrt && g_rt[0]) return g_rt[0]->h;
  return g_zt? g_zt->h : 0; }

/* scissor Y range (min | max<<16) mirrored for a bottom-origin window */
static u32 origin_scissor_y(u32 ys, u32 oh){
  if(!oh) return ys;
  u32 y0=ys&0xFFFF, y1=ys>>16; if(y1>oh) y1=oh; if(y0>y1) y0=y1;
  return (oh-y1) | ((oh-y0)<<16); }

static void apply_state(GpuChannel* ch){
  u32* r=ch->r3d;
  { u32 sig[256]; int n=0;
#define SR(a,b) for(u32 q_=(a);q_<=(b);q_++) sig[n++]=r[q_];
    SR(0x280,0x285) SR(0x300,0x303) SR(0x35F,0x35F) SR(0x36B,0x36B) SR(0x372,0x372) SR(0x380,0x382) SR(0x3D5,0x3D7) SR(0x3E4,0x3E4)
    SR(0x4B3,0x4B3) SR(0x4B8,0x4BA) SR(0x4C3,0x4C3) SR(0x4C7,0x4CA) SR(0x4D0,0x4DF) SR(0x4E0,0x4E7) SR(0x4ED,0x4ED) SR(0x546,0x546)
    SR(0x55B,0x55B) SR(0x565,0x569) SR(0x56F,0x56F) SR(0x591,0x592) SR(0x646,0x648) SR(0x680,0x687) SR(0x780,0x7BF) SR(0x4EB,0x4EB)
#undef SR
    sig[n++]= g_zt? 1u : 0u; sig[n++]=origin_flip_h(r); memcpy(&sig[n++],&g_draw_scale,4);
    if(g_state_ok && !memcmp(sig,g_state_sig,(size_t)n*4)){ st_state_skip++; return; }
    memcpy(g_state_sig,sig,(size_t)n*4); g_state_ok=1; }
  if(p_clip){ int m=r[0x35F]&1; if(m!=g_clip_mode){ p_clip(0x8CA1 /* LOWER_LEFT */, m? 0x935F /* ZERO_TO_ONE */ : 0x935E /* NEGATIVE_ONE_TO_ONE */); g_clip_mode=m; } }
  /* grouped so only what changed hits GL */
  static u32 gs[16][20]; int ok=g_gs_ok; g_gs_ok=1;
#define GRP(id_, ...) u32 k_##id_[]={__VA_ARGS__}; if(!ok || memcmp(gs[id_],k_##id_,sizeof k_##id_)) if(memcpy(gs[id_],k_##id_,sizeof k_##id_))
  u32 gsc; memcpy(&gsc,&g_draw_scale,4);
  float vt[6]; memcpy(vt,&r[0x280],24);
  float sx=vt[0], sy=vt[1], tx=vt[3], ty=vt[4];
  u32 oh=origin_flip_h(r);
  { GRP(0, r[0x280],r[0x281],r[0x282],r[0x283],r[0x284],r[0x285],r[0x35F],r[0x300],r[0x301],r[0x302],r[0x303],r[0x380],r[0x381],r[0x382],gsc,oh){
    float x0=tx-fabsf_(sx), y0=ty-fabsf_(sy), w=2*fabsf_(sx), h=2*fabsf_(sy);
    if(w<1||h<1){ u32 vx=r[0x300], vy=r[0x301]; x0=(float)(vx&0xFFFF); w=(float)(vx>>16); y0=(float)(vy&0xFFFF); h=(float)(vy>>16); }
    if(oh) y0=(float)oh-(y0+h);
    glViewportIndexedf(0,x0*g_draw_scale,y0*g_draw_scale,w*g_draw_scale,h*g_draw_scale);
    float n, f; memcpy(&n,&r[0x302],4); memcpy(&f,&r[0x303],4);
    /* depth goes through the viewport transform (z*scale_z+translate_z) like on hw / yuzu's vk backend. SMO's fog
       pass draws at z=1 with NOTEQUAL to skip the sky, so this has to be exact */
    if((vt[2]!=0.0f || vt[5]!=0.0f) && !getenv("SWR_DEPTH_REGS")){ float red= (r[0x35F]&1)? 0.0f : 1.0f; n=vt[5]-vt[2]*red; f=vt[5]+vt[2];
      if(fabsf_(n)<1e-6f) n=0.0f; if(fabsf_(n-1.0f)<1e-6f) n=1.0f; if(fabsf_(f)<1e-6f) f=0.0f; if(fabsf_(f-1.0f)<1e-6f) f=1.0f; }
    glDepthRangeIndexed(0,n,f);
    if(r[0x380]&1){ u32 xs=r[0x381], ys=origin_scissor_y(r[0x382],oh); glEnable(GL_SCISSOR_TEST); glScissorIndexed(0,(GLint)((xs&0xFFFF)*g_draw_scale),(GLint)((ys&0xFFFF)*g_draw_scale),(GLsizei)(((xs>>16)-(xs&0xFFFF))*g_draw_scale+0.5f),(GLsizei)(((ys>>16)-(ys&0xFFFF))*g_draw_scale+0.5f)); }
    else glDisable(GL_SCISSOR_TEST); } }
  /* raster state (winding flips with the Y flip) */
  { float w=2*fabsf_(sx), h=2*fabsf_(sy); int yflip= ((w<1||h<1)? 0 : (sy<0)) ^ (oh? 1 : 0);
    GRP(1, r[0x646],r[0x647],r[0x648],r[0x36B],r[0x372],r[0x55B],r[0x56F],(u32)yflip){
    if(r[0x646]){ glEnable(GL_CULL_FACE); glCullFace(r[0x648]==0x404? GL_FRONT : r[0x648]==0x408? GL_FRONT_AND_BACK : GL_BACK); } else glDisable(GL_CULL_FACE);
    { GLenum ff= r[0x647]==0x900? GL_CW : GL_CCW; if(yflip) ff= ff==GL_CW? GL_CCW : GL_CW; glFrontFace(ff); }
    glPolygonMode(GL_FRONT_AND_BACK, r[0x36B]==0x1B01? GL_LINE : r[0x36B]==0x1B00? GL_POINT : GL_FILL);
    if(r[0x372]){ float fac, un; memcpy(&fac,&r[0x55B],4); memcpy(&un,&r[0x56F],4); glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(fac,un); } else glDisable(GL_POLYGON_OFFSET_FILL); } }
  /* depth / stencil */
  { u32 z= g_zt? 1u : 0u;
    GRP(2, z,r[0x4B3],r[0x4C3],r[0x4BA],r[0x4E0],r[0x4E1],r[0x4E2],r[0x4E3],r[0x4E4],r[0x4E5],r[0x4E6],r[0x4E7],r[0x565],r[0x566],r[0x567],r[0x568],r[0x569],r[0x3D5],r[0x3D6],r[0x3D7]){
    if(r[0x4B3] && g_zt){ glEnable(GL_DEPTH_TEST); GLenum df=cmp_func(r[0x4C3]);
      { static int eq=-1; if(eq<0) eq=getenv("SWR_ZEQUAL_LEQUAL")!=NULL; if(eq && df==GL_EQUAL) df=GL_LEQUAL; }   /* test: depth EQUAL passes */
      glDepthFunc(df); } else glDisable(GL_DEPTH_TEST);
    glDepthMask(r[0x4BA] && g_zt? GL_TRUE : GL_FALSE);
    if(r[0x4E0] && g_zt){ glEnable(GL_STENCIL_TEST);
      glStencilFuncSeparate(GL_FRONT,cmp_func(r[0x4E4]),r[0x4E5],r[0x4E6]); glStencilOpSeparate(GL_FRONT,stencil_op(r[0x4E1]),stencil_op(r[0x4E2]),stencil_op(r[0x4E3])); glStencilMaskSeparate(GL_FRONT,r[0x4E7]);
      if(r[0x565]){ glStencilFuncSeparate(GL_BACK,cmp_func(r[0x569]),r[0x3D5],r[0x3D7]); glStencilOpSeparate(GL_BACK,stencil_op(r[0x566]),stencil_op(r[0x567]),stencil_op(r[0x568])); glStencilMaskSeparate(GL_BACK,r[0x3D6]); }
      else { glStencilFuncSeparate(GL_BACK,cmp_func(r[0x4E4]),r[0x4E5],r[0x4E6]); glStencilOpSeparate(GL_BACK,stencil_op(r[0x4E1]),stencil_op(r[0x4E2]),stencil_op(r[0x4E3])); glStencilMaskSeparate(GL_BACK,r[0x4E7]); } }
    else glDisable(GL_STENCIL_TEST); } }
  /* blend + color masks */
  { GRP(3, r[0x4B8]&1,r[0x4C7],r[0x4C8],r[0x4C9],r[0x4CA]){
    if(r[0x4B8]&1) glEnable(GL_FRAMEBUFFER_SRGB); else glDisable(GL_FRAMEBUFFER_SRGB);   /* framebuffer_srgb */
    float bc[4]; memcpy(bc,&r[0x4C7],16); glBlendColor(bc[0],bc[1],bc[2],bc[3]); } }
  for(int i=0;i<8;i++){
    u32 cm= r[0x3E4]? r[0x680] : r[0x680+i]; int en= r[0x4D8+i]; u32* b=&r[0x780+i*8];
    u32 k[12]={cm&0xFFFF,(u32)en,0,0,0,0,0,0,0,0,0,0};
    if(en){ k[2]=r[0x4B9]; if(r[0x4B9]){ k[3]=b[1]; k[4]=b[2]; k[5]=b[3]; k[6]=b[4]; k[7]=b[5]; k[8]=b[6]; } else { k[3]=r[0x4D0]; k[4]=r[0x4D1]; k[5]=r[0x4D2]; k[6]=r[0x4D3]; k[7]=r[0x4D4]; k[8]=r[0x4D6]; } }
    if(ok && !memcmp(gs[4+i],k,sizeof k)) continue; memcpy(gs[4+i],k,sizeof k);
    glColorMaski(i,(cm&0xF)!=0,(cm&0xF0)!=0,(cm&0xF00)!=0,(cm&0xF000)!=0);
    if(en){ glEnablei(GL_BLEND,i);
      /* hw always uses the alpha fields, separate_alpha doesn't turn them off (yuzu does the same). menu alpha
         masks need this */
      if(r[0x4B9]){ glBlendEquationSeparatei(i,blend_eq(b[1]),blend_eq(b[4]? b[4] : b[1]));
        glBlendFuncSeparatei(i,blend_factor(b[2]),blend_factor(b[3]),blend_factor(b[5]? b[5] : b[2]),blend_factor(b[6]? b[6] : b[3])); }
      else { glBlendEquationSeparatei(i,blend_eq(r[0x4D0]),blend_eq(r[0x4D3]? r[0x4D3] : r[0x4D0]));
        glBlendFuncSeparatei(i,blend_factor(r[0x4D1]),blend_factor(r[0x4D2]),blend_factor(r[0x4D4]? r[0x4D4] : r[0x4D1]),blend_factor(r[0x4D6]? r[0x4D6] : r[0x4D2])); } }
    else glDisablei(GL_BLEND,i);
  }
  { GRP(12, r[0x591],r[0x592],r[0x4ED],r[0x546]){
    if(r[0x591]){ glEnable(GL_PRIMITIVE_RESTART); glPrimitiveRestartIndex(r[0x592]); } else glDisable(GL_PRIMITIVE_RESTART);
    { float lw; memcpy(&lw,&r[0x4ED],4); if(lw>0 && lw<64) glLineWidth(lw); }
    { float ps; memcpy(&ps,&r[0x546],4); if(ps>0 && ps<256) glPointSize(ps); } } }
#undef GRP
}

/* ============================================================ drawing */
static u32 attr_count(u32 size){ switch(size){ case 0x01: return 4; case 0x02: return 3; case 0x03: return 4; case 0x04: return 2; case 0x05: return 3; case 0x0A: return 4; case 0x0F: return 2;
  case 0x12: return 1; case 0x13: return 3; case 0x18: return 2; case 0x1B: return 1; case 0x1D: return 1; case 0x30: return 4; case 0x31: return 3; } return 4; }

static u32 attr_bits(u32 size){ switch(size){ case 0x01: case 0x02: case 0x04: case 0x12: return 32; case 0x03: case 0x05: case 0x0F: case 0x1B: return 16; case 0x30: case 0x31: return 0; } return 8; }

static GLenum topo(u32 t){ switch(t){ case 0: return GL_POINTS; case 1: return GL_LINES; case 2: return GL_LINE_LOOP; case 3: return GL_LINE_STRIP; case 4: return GL_TRIANGLES; case 5: return GL_TRIANGLE_STRIP;
  case 6: return GL_TRIANGLE_FAN; case 10: return GL_LINES_ADJACENCY; case 11: return GL_LINE_STRIP_ADJACENCY; case 12: return GL_TRIANGLES_ADJACENCY; case 13: return GL_TRIANGLE_STRIP_ADJACENCY; case 14: return GL_PATCHES; } return GL_TRIANGLES; }

void r3d_cb_bind_gl(GpuChannel* ch, int stage, u32 v){
  u32* r=ch->r3d; u32 idx=(v>>4)&31;
  if(stage<=4 && idx>=18){ static u32 said[5]; if(!(said[stage]>>idx&1)){ said[stage]|=1u<<idx;   /* Switch 2 shaders read c[24] */
      LOG("gpu: ch%u stage %d %s cbuf %u at %llx size %x: past the 18 Maxwell slots, dropped\n",ch->id,stage,v&1? "binds" : "unbinds",idx,
          (unsigned long long)(((u64)r[0x8E1]<<32)|r[0x8E2]),r[0x8E0]); } }
  if(stage>4 || idx>=18) return;
  static int per_ch=-1; if(per_ch<0) per_ch=getenv("SWR_SHARED_CB")==NULL;   /* SWR_SHARED_CB=1: one set for all channels */
  CbState* cs= per_ch? &chan_cb(ch)->s : &g_cb;
  if(v&1){ cs->cb[stage][idx]=((u64)r[0x8E1]<<32)|r[0x8E2]; cs->cbsize[stage][idx]=r[0x8E0];
    if(ch->ampere){ extern u64 gmmu_translate(u64); u64 a=cs->cb[stage][idx]; static u64 seen[32]; static int ns; int k=0; for(;k<ns;k++) if(seen[k]==a) break;
      { static int nb3; extern u64 plat_ns(void); static u64 t0; if(!t0) t0=plat_ns(); if(nb3<200 && getenv("SWR_CB3_LOG") && plat_ns()-t0>14000000000ull){ nb3++; LOG("cb bind: ch%u stage %d bank %u at %llx size %x\n",ch->id,stage,idx,(unsigned long long)a,r[0x8E0]); } }
      if(k==ns && ns<32){ seen[ns++]=a; LOG("gpu: ch%u stage %d binds cbuf %u at %llx size %x (cpu %llx)\n",ch->id,stage,idx,(unsigned long long)a,r[0x8E0],(unsigned long long)gmmu_translate(a)); } } }
  else { cs->cb[stage][idx]=0; cs->cbsize[stage][idx]=0; }
  if(per_ch && g_cb_owner==(int)ch->id){ g_cb.cb[stage][idx]=cs->cb[stage][idx]; g_cb.cbsize[stage][idx]=cs->cbsize[stage][idx]; }
  else if(per_ch && g_cb_owner<0) cb_switch(ch); }

/* NVN ssbo table at c0[0x110+16k]: u64 addr + u32 size, for LDG */
#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#endif
static GLuint g_sbuf[3][8];
static GLuint buf_cached(GLenum target, u64 start, u64 len);
static int cs_gpu_src(u64 a, u64 n, GLuint* buf, u32* off); static void cs_stage_push(GLuint src, u32 size, u64 addr, GLuint wmb, u32 wbase);
extern volatile int g_cs_any;
/* buffers a draw's shaders write (atomics, stores) go back to guest memory after the draw, same as compute output.
   tomodachi's depth pass does atomicMax per screen tile and the culling compute after it reads that, without it the
   compute only saw the reset values and objects popped in and out */
static struct { GLuint buf; u64 addr; u32 size, wbase; int slot; } g_gw[24]; static int g_ngw; static GLuint g_gwmb[3]; static int g_gw_full[3];
static void gbuf_writeback(void){
  static int off=-1; if(off<0) off=getenv("SWR_NO_DRAW_WRITEBACK")!=NULL;   /* =1: draws' writes stay on the GPU */
  if(!g_ngw || off){ g_ngw=0; return; }
  glMemoryBarrier(0x00000200|0x00002000 /* BUFFER_UPDATE | SHADER_STORAGE */);
  for(int k=0;k<g_ngw;k++) cs_stage_push(g_gw[k].buf,g_gw[k].size,g_gw[k].addr,g_gw_full[g_gw[k].slot]? 0 : g_gwmb[g_gw[k].slot],g_gw[k].wbase);
  for(int k=0;k<g_ngw;k++){ /* a fresh buffer for the next draw so it doesn't wait on this one */
    for(int s=0;s<3;s++) for(int i=0;i<8;i++) if(g_sbuf[s][i]==g_gw[k].buf){ glDeleteBuffers(1,&g_sbuf[s][i]); g_sbuf[s][i]=0; } }
  g_ngw=0; }

static void bind_gbufs(int stage, Sh* s){
  if(!s->si.ngbuf) return; int slot= stage==0? 0 : stage==4? 1 : 2; u64 cb0=g_cb.cb[stage][0];
  static u8* tmp; static size_t cap;
  static u32* wm; static u32 wcap; u32 nw=32, wbase[8]={0}, wsize[8]={0};
  for(int i=0;i<s->si.ngbuf && i<8;i++){
    if(!g_sbuf[slot][i]) glGenBuffers(1,&g_sbuf[slot][i]);
    u64 addr=0; u32 size=16; if(cb0){ u32 v[3]; gmem_read(cb0+s->si.gbuf_cbuf_off[i],v,12); addr=((u64)v[1]<<32)|v[0]; size=v[2]; }
    if(!size) size=16; if(size>(16u<<20)) size=16u<<20; size=(size+3)&~3u;
    { static int nc=-1; if(nc<0) nc=getenv("SWR_NO_SSBO_CACHE")!=NULL;
      if(!nc && addr && !((s->si.gbuf_wmask>>i)&1)){ GLuint id=buf_cached(GL_SHADER_STORAGE_BUFFER,addr,size); if(id){ glBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot*8+i,id); continue; } } }
    int wr= addr && ((s->si.gbuf_wmask>>i)&1);
    { GLuint gsb; u32 gso;   /* a compute (or draw) result not read back yet: copy it on the GPU */
      if(wr && g_cs_any && cs_gpu_src(addr,size,&gsb,&gso)){
        glBindBuffer(GL_COPY_WRITE_BUFFER,g_sbuf[slot][i]); glBufferData(GL_COPY_WRITE_BUFFER,size,NULL,GL_DYNAMIC_COPY);
        glBindBuffer(GL_COPY_READ_BUFFER,gsb); glCopyBufferSubData(GL_COPY_READ_BUFFER,GL_COPY_WRITE_BUFFER,gso,0,size); }
      else {
        if(cap<size){ tmp=realloc(tmp,size); cap=size; }
        if(addr) gmem_read(addr,tmp,size); else memset(tmp,0,size);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER,g_sbuf[slot][i]); glBufferData(GL_SHADER_STORAGE_BUFFER,size,tmp,GL_STREAM_DRAW); } }
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot*8+i,g_sbuf[slot][i]);
    if(wr && g_ngw<24){ wbase[i]=nw; wsize[i]=size/4; nw+=(size/4+31)/32; g_gw[g_ngw].buf=g_sbuf[slot][i]; g_gw[g_ngw].addr=addr; g_gw[g_ngw].size=size; g_gw[g_ngw].wbase=wbase[i]; g_gw[g_ngw].slot=slot; g_ngw++; } }
  if(s->si.gbuf_wmask){   /* header + bits (zeroed) for swr_wmark; with 8 buffers there's no binding left: whole ranges go back */
    g_gw_full[slot]= s->si.ngbuf>=8;
    if(!g_gw_full[slot]){
      if(wcap<nw){ wm=realloc(wm,nw*4); wcap=nw; }
      memset(wm,0,nw*4); for(int i=0;i<8;i++){ wm[i]=wbase[i]; wm[16+i]=wsize[i]; }   /* size 0: marks ignored */
      if(!g_gwmb[slot]) glGenBuffers(1,&g_gwmb[slot]);
      glBindBuffer(GL_SHADER_STORAGE_BUFFER,g_gwmb[slot]); glBufferData(GL_SHADER_STORAGE_BUFFER,nw*4,wm,GL_DYNAMIC_COPY);
      glBindBufferBase(GL_SHADER_STORAGE_BUFFER,slot*8+s->si.ngbuf,g_gwmb[slot]); } } }

/* persistent ring for cbufs (GL 4.4 buffer_storage): memcpy + glBindBufferRange, no 64K realloc on every update.
   fences per segment so we never overwrite something the gpu is still reading */
#define UR_SEGS 4
static GLuint g_ur; static u8* g_ur_ptr; static u32 g_ur_size, g_ur_head, g_ur_align=256; static u64 g_ur_abs; static GLsync g_ur_f[UR_SEGS];
static PFNGLFENCESYNCPROC p_fs; static PFNGLCLIENTWAITSYNCPROC p_cws; static PFNGLDELETESYNCPROC p_ds;
static u64 st_ur_wait; static u32 g_ub_bound[64];
static u32 g_sb_bound[64];

static int cb_ssbo_binding(int stage, Sh* s, int cb){
  int slot=stage==0? 0 : stage==4? 1 : 2, rank=0;
  for(int i=0;i<cb;i++) if(s->si.cbuf_ssbo_mask&(1u<<i)) rank++;
  return 24+slot*4+rank;
}

static void ur_init(void){
  if(getenv("SWR_NO_UBO_RING")) return;
  PFNGLBUFFERSTORAGEPROC bs=(PFNGLBUFFERSTORAGEPROC)backend_gl_proc("glBufferStorage"); PFNGLMAPBUFFERRANGEPROC mbr=(PFNGLMAPBUFFERRANGEPROC)backend_gl_proc("glMapBufferRange");
  p_fs=(PFNGLFENCESYNCPROC)backend_gl_proc("glFenceSync"); p_cws=(PFNGLCLIENTWAITSYNCPROC)backend_gl_proc("glClientWaitSync"); p_ds=(PFNGLDELETESYNCPROC)backend_gl_proc("glDeleteSync");
  if(!bs || !mbr || !p_fs || !p_cws || !p_ds) return;
  GLint al=256, sal=16; glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT,&al); glGetIntegerv(0x90DF /* SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT */,&sal);
  if(sal>al) al=sal; if(al<16) al=16; g_ur_align=(u32)al;   /* cbufs past the UBO limit get bound as storage blocks from the same ring */
  g_ur_size=32u<<20;
  GLuint b=0; glGenBuffers(1,&b); glBindBuffer(GL_UNIFORM_BUFFER,b);
  GLbitfield fl=GL_MAP_WRITE_BIT|GL_MAP_PERSISTENT_BIT|GL_MAP_COHERENT_BIT;
  bs(GL_UNIFORM_BUFFER,(GLsizeiptr)g_ur_size+65536,NULL,fl);
  u8* ptr= glGetError()==GL_NO_ERROR? (u8*)mbr(GL_UNIFORM_BUFFER,0,(GLsizeiptr)g_ur_size+65536,fl) : NULL;
  while(glGetError()!=GL_NO_ERROR){}
  if(!ptr) return;
  g_ur=b; g_ur_ptr=ptr; }

static u32 ur_alloc(u32 sz){
  u32 need=(sz+g_ur_align-1)/g_ur_align*g_ur_align, seg=g_ur_size/UR_SEGS;
  u32 off=g_ur_head; if(off+need>g_ur_size){ g_ur_abs+=g_ur_size-off; off=0; }
  u32 cs=g_ur_head/seg; if(cs>=UR_SEGS) cs=UR_SEGS-1;
  u32 es=(off+need-1)/seg; if(es>=UR_SEGS) es=UR_SEGS-1;
  while(cs!=es){ if(g_ur_f[cs]) p_ds(g_ur_f[cs]); g_ur_f[cs]=p_fs(GL_SYNC_GPU_COMMANDS_COMPLETE,0);
    cs=(cs+1)%UR_SEGS;
    if(g_ur_f[cs]){ GLenum rr=p_cws(g_ur_f[cs],GL_SYNC_FLUSH_COMMANDS_BIT,0);
      if(rr==GL_TIMEOUT_EXPIRED){ st_ur_wait++; p_cws(g_ur_f[cs],GL_SYNC_FLUSH_COMMANDS_BIT,2000000000ull); }
      p_ds(g_ur_f[cs]); g_ur_f[cs]=0; } }
  g_ur_head=off+need; g_ur_abs+=need; return off; }

typedef struct { u64 a, gen, abs; u32 sz, valid, epoch, off; } UrC;

static UrC urc[4096];

static void upload_cbufs_ring(int stage, Sh* s){
  extern u64 g_gw_gen; extern int gpu_mem_dirty_since(u64,u64,u64);
  int bb=(stage==0? 0 : stage==4? 1 : 2)*18;
  for(int i=0;i<18;i++){ if(!(s->si.cbuf_mask&(1u<<i))) continue;
    u64 a=g_cb.cb[stage][i]; u32 sz=g_cb.cbsize[stage][i]; if(sz>65536) sz=65536; if(!sz) sz=16;
    if(s->si.cbuf_max[i] && sz>s->si.cbuf_max[i]) sz=s->si.cbuf_max[i];   /* only the bytes the shader reads */
    /* reads past the bound size give 0 on hw. the ring had an older upload there: tomodachi's mii shaders read past
       a small bound buffer and got leftovers from other draws. indexed reads need the full 64K zero tail too, otherwise they sample the next upload */
    u32 tail= s->si.cbuf_max[i]>sz? s->si.cbuf_max[i]-sz : 0;
    UrC* e=&urc[(u32)((a>>4)^(a>>16)^(a>>28))&4095]; u32 off;
    /* same submit/address, not GPU-written, ring hasn't wrapped: reuse */
    if(a && e->a==a && e->epoch==g_submit_epoch && e->valid==sz && e->sz>=sz+tail && g_ur_abs-e->abs<g_ur_size/2 && !gpu_mem_dirty_since(e->gen,a,sz)){ off=e->off; st_cb_skip++; }
    else { st_cb_bytes+=sz; off=ur_alloc(sz+tail); u8* dst=g_ur_ptr+off; u8* src= a? gmem_ptr(a,sz) : NULL;
      if(src) memcpy(dst,src,sz); else if(a) gmem_read(a,dst,sz); else memset(dst,0,sz);
      if(tail) memset(dst+sz,0,tail);
      if(a){ e->a=a; e->valid=sz; e->sz=sz+tail; e->epoch=g_submit_epoch; e->gen=g_gw_gen; e->off=off; e->abs=g_ur_abs; } }
    if(s->si.cbuf_ssbo_mask&(1u<<i)){
      int bind=cb_ssbo_binding(stage,s,i);
      if(g_sb_bound[bind]!=off+1){ glBindBufferRange(GL_SHADER_STORAGE_BUFFER,bind,g_ur,off,65536); g_sb_bound[bind]=off+1; }
    } else if(g_ub_bound[bb+i]!=off+1){ glBindBufferRange(GL_UNIFORM_BUFFER,bb+i,g_ur,off,65536); g_ub_bound[bb+i]=off+1; } } }

static void upload_cbufs(int stage, Sh* s){
  if(g_ur_ptr){ upload_cbufs_ring(stage,s); return; }
  for(int i=0;i<18;i++){ if(!(s->si.cbuf_mask&(1u<<i))) continue;
    u64 a=g_cb.cb[stage][i]; u32 sz=g_cb.cbsize[stage][i]; if(sz>65536) sz=65536; if(!sz) sz=16;
    if(s->si.cbuf_max[i] && sz>s->si.cbuf_max[i]) sz=s->si.cbuf_max[i];   /* only the bytes the shader reads */
    static u8* last[5][18]; static u32 last_sz[5][18];
    if(!last[stage][i]) last[stage][i]=calloc(1,65536);
    { /* same submit/address, not written by the gpu: unchanged (games don't write while the gpu reads) */
      extern u64 g_gw_gen; extern int gpu_mem_dirty_since(u64,u64,u64); static u64 la[5][18], lg[5][18], le[5][18]; static u32 lv[5][18];
      if(a && la[stage][i]==a && lv[stage][i]==sz && le[stage][i]==g_submit_epoch && last_sz[stage][i]==sz+(s->si.cbuf_max[i]>sz?s->si.cbuf_max[i]-sz:0) && !gpu_mem_dirty_since(lg[stage][i],a,sz)){
        st_cb_skip++;
        if(s->si.cbuf_ssbo_mask&(1u<<i)) glBindBufferBase(GL_SHADER_STORAGE_BUFFER,cb_ssbo_binding(stage,s,i),g_ubuf[stage][i]);
        else glBindBufferBase(GL_UNIFORM_BUFFER,(stage==0? 0 : stage==4? 1 : 2)*18+i,g_ubuf[stage][i]);
        continue; }
      la[stage][i]=a; lv[stage][i]=sz; lg[stage][i]=g_gw_gen; le[stage][i]=g_submit_epoch; }
    st_cb_bytes+=sz;
    u32 tail= s->si.cbuf_max[i]>sz? s->si.cbuf_max[i]-sz : 0;   /* past the bound size reads 0 */
    u8* src= a? gmem_ptr(a,sz) : NULL; u8 tmp_local[256]; u8* tmp=NULL;
    if(!src || tail){ static u8 big[65536]; tmp= sz+tail<=256? tmp_local : big; if(src) memcpy(tmp,src,sz); else if(a) gmem_read(a,tmp,sz); else memset(tmp,0,sz);
      memset(tmp+sz,0,tail); src=tmp; sz+=tail; }
    if(last_sz[stage][i]<sz || memcmp(last[stage][i],src,sz)){      /* exact compare */
      memcpy(last[stage][i],src,sz); last_sz[stage][i]=sz;
      glBindBuffer(GL_UNIFORM_BUFFER,g_ubuf[stage][i]); glBufferData(GL_UNIFORM_BUFFER,65536,NULL,GL_STREAM_DRAW); glBufferSubData(GL_UNIFORM_BUFFER,0,sz,src); }
    if(s->si.cbuf_ssbo_mask&(1u<<i)) glBindBufferBase(GL_SHADER_STORAGE_BUFFER,cb_ssbo_binding(stage,s,i),g_ubuf[stage][i]);
    else glBindBufferBase(GL_UNIFORM_BUFFER,(stage==0? 0 : stage==4? 1 : 2)*18+i,g_ubuf[stage][i]); } }

/* multi-bind (GL 4.4): a stage's textures + samplers in two calls, none if nothing changed. other glBindTexture
   users (uploads, creation) use g_scratch_unit */
typedef void (APIENTRY*PFN_BT_)(GLuint,GLsizei,const GLuint*);
static PFN_BT_ p_bt, p_bs; static int g_scratch_unit=-1;
static GLuint g_tb_last[3][32], g_sb_last[3][32]; static int g_tb_n[3], g_tb_ok[3]; static u64 st_tb_skip;

static void mb_init(void){
  if(getenv("SWR_NO_MULTIBIND")) return;
  GLint mu=0; glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS,&mu); if(mu<65) return;
  p_bt=(PFN_BT_)backend_gl_proc("glBindTextures"); p_bs=(PFN_BT_)backend_gl_proc("glBindSamplers");
  if(!p_bt || !p_bs){ p_bt=p_bs=NULL; return; }
  g_scratch_unit=mu-1; glActiveTexture(GL_TEXTURE0+g_scratch_unit); }

/* sampler type mismatch (2D vs array, cube vs array): use a texture view */
static PFNGLTEXTUREVIEWPROC p_tview; static u64 st_tex_view;

static GLuint tex_view(Tex* t, GLenum want){
  if(!p_tview || !t) return 0;
  if(t->view && t->view_target==want) return t->view;
  GLenum a=t->target; u32 layers= a==GL_TEXTURE_2D? 1 : a==GL_TEXTURE_CUBE_MAP? 6 : t->d; u32 nl;
  int ok2d= a==GL_TEXTURE_2D||a==GL_TEXTURE_2D_ARRAY||a==GL_TEXTURE_CUBE_MAP||a==GL_TEXTURE_CUBE_MAP_ARRAY;
  if(!ok2d) return 0;
  if(want==GL_TEXTURE_2D) nl=1; else if(want==GL_TEXTURE_2D_ARRAY) nl=layers;
  else if(want==GL_TEXTURE_CUBE_MAP){ if(layers<6) return 0; nl=6; }
  else if(want==GL_TEXTURE_CUBE_MAP_ARRAY){ if(layers<6) return 0; nl=layers/6*6; }
  else return 0;
  if(t->f.fmt==0 && !t->f.compressed && !(t->f.astc && t->f.tobc)) return 0;   /* ASTC stored as BC3 views fine */
  if(t->view){ glDeleteTextures(1,&t->view); t->view=0; g_fb_sig_ok=0; }
  GLuint v=0; glGenTextures(1,&v); p_tview(v,want,t->id,t->f.ifmt,0,t->levels,0,nl);
  if(glGetError()!=GL_NO_ERROR){ glDeleteTextures(1,&v); return 0; }
  t->view=v; t->view_target=want; st_tex_view++;
  t->vswz[0]=GL_RED; t->vswz[1]=GL_GREEN; t->vswz[2]=GL_BLUE; t->vswz[3]=GL_ALPHA;
  return v; }

static void mb_reset(void){ g_tb_ok[0]=g_tb_ok[1]=g_tb_ok[2]=0; if(g_scratch_unit>=0) glActiveTexture(GL_TEXTURE0+g_scratch_unit); }

static int ifmt_signed(GLenum f){ return f==GL_RGBA8I||f==GL_RGBA16I||f==GL_RGBA32I||f==GL_RG8I||f==GL_RG16I||f==GL_RG32I||f==GL_R8I||f==GL_R16I||f==GL_R32I; }

/* FS key: integer textures (u/isampler) and integer RT outputs (u/ivec4) */
/* texture handle for a slot, from the texture cbuf or for bindless from whatever bank the shader read. split
   handles (tex | sampler) get combined the same way the shader does it */
/* TIC pool of the draw being set up (for checking handles) */
static u64 g_tic_base; static u32 g_tic_lim; static int g_tex_any_ok;   /* a texture has resolved at least once */
/* test view (show_game_rt=1 in the ini, off by default): while the image a game presents stays black
   because its last composition passes don't work yet, the window shows the biggest render target the game drew
   with its own textures (MKW: the world map) */
int g_opt_show_rt=-1; static int g_draw_asset; static Tex* g_view_rt; static u32 g_view_frame;

static int tic_ok(u32 h){
  if((h&0xFFFFF)>g_tic_lim) return 0;
  u32 w[8]; gmem_read(g_tic_base+(h&0xFFFFF)*32ull,w,32); return (w[1]|w[2]|w[4])!=0; }

/* a bound handle past the end of the TIC pool can't give a texture on hw either, so on hw the handles have to be
   somewhere we don't read. tomodachi life (2026 SDK) reads ffffffff from the texture bank for every texture. we
   look for the handle in the other constant banks of the stage (same offset) and in the texture bank of the other
   stages, keep the first place that gives valid entries and log what we found. SWR_NO_HANDLE_FIX=1 turns it off */
static int g_hfix=-2;   /* -2 not decided, -1 nothing found, else stage*32+bank */
static u32 handle_fallback(int stage, u32 bank, u32 off, u32 h){
  static int off_env=-1; if(off_env<0) off_env=getenv("SWR_NO_HANDLE_FIX")!=NULL; if(off_env || !g_tic_base) return h;
  if(g_hfix>=0){ int s2=g_hfix>>5, b2=g_hfix&31; if(s2==9) s2=stage; u64 c=g_cb.cb[s2][b2]; u32 v= c? gmem_r32(c+off) : h; return tic_ok(v)? v : h; }
  if(g_hfix==-1) return h;
  if(g_tex_any_ok){ g_hfix=-1; return h; }   /* handles work in this game: an invalid one is just an unused slot */
  static int tries; if(tries>=64){ g_hfix=-1; return h; } tries++;
  if(tries==1){ /* what the stage's banks hold around the handle, once */
    LOG("gpu: texture handle %x at bank %u+%#x is outside the TIC pool (limit %x); constant banks of stage %d:\n",h,bank,off,g_tic_lim,stage);
    for(int b=0;b<18;b++){ u64 c=g_cb.cb[stage][b]; if(!c) continue; u32 w[24]; gmem_read(c,w,sizeof w);
      LOG("   bank %2d at %llx: %08x %08x %08x %08x %08x %08x %08x %08x | %08x %08x %08x %08x %08x %08x %08x %08x | %08x %08x %08x %08x %08x %08x %08x %08x\n",b,(unsigned long long)c,
        w[0],w[1],w[2],w[3],w[4],w[5],w[6],w[7],w[8],w[9],w[10],w[11],w[12],w[13],w[14],w[15],w[16],w[17],w[18],w[19],w[20],w[21],w[22],w[23]); } }
  for(int b=0;b<18;b++){ if((u32)b==bank) continue; u64 c=g_cb.cb[stage][b]; if(!c) continue; u32 v=gmem_r32(c+off);
    if(v!=h && tic_ok(v)){ g_hfix=9*32+b; LOG("gpu: texture handles found in constant bank %d instead of %u (handle %x): using that bank\n",b,bank,v); return v; } }
  for(int s2=0;s2<5;s2++){ if(s2==stage) continue; u64 c=g_cb.cb[s2][bank]; if(!c || c==g_cb.cb[stage][bank]) continue; u32 v=gmem_r32(c+off);
    if(v!=h && tic_ok(v)){ g_hfix=s2*32+(int)bank; LOG("gpu: texture handles found in bank %u of stage %d instead of stage %d (handle %x): using it\n",bank,s2,stage,v); return v; } }
  return h; }

/* the bank bound texture handles come from: tex_cb_index (0x982) on Maxwell, the one the TEX instructions name on
   Ampere (NVN2 leaves 0x982 at 0 and keeps them in bank 2; read from bank 0 every handle was 0 and every texture black) */
static u32 sh_tcb(const Sh* s, u32 reg){ return s && s->si.tex_bank_set? s->si.tex_bank : reg; }

static u32 sh_tex_handle(int stage, u32 tcb, const ShTex* st){
  u32 bank=(st->bindless? st->bl_cbuf : tcb)&31; if(bank>=18) return 0;   /* 18 banks per stage */
  u64 cba=g_cb.cb[stage][bank]; u32 h= cba? gmem_r32(cba+st->cbuf_off*4) : 0;
  if(st->bindless && st->bl2){ u32 b2=st->bl2_cbuf&31; u64 c2= b2<18? g_cb.cb[stage][b2] : 0; if(c2) h|=gmem_r32(c2+st->bl2_off*4); }
  if(!st->bindless && g_tic_base && g_tic_lim && (h&0xFFFFF)>g_tic_lim && g_hfix!=-1) h=handle_fallback(stage,bank,st->cbuf_off*4,h);
  return h; }

/* a stage's program: program region + offset on Maxwell. Ampere gives every stage its own 64 bit address
   (SET_PIPELINE_PROGRAM_ADDRESS_A/B, 0x805/0x806 + 16*stage): MKW keeps its vertex and fragment programs
   in different 4 GB windows, through one shared region the fragment programs were read from unmapped memory */
static u64 stage_prog(const GpuChannel* ch, const u32* r, u64 region, int i){
  if(ch && ch->ampere){ u64 a=((u64)(r[0x805+i*0x10]&0xFF)<<32)|r[0x806+i*0x10]; if(a) return a; }
  return region+r[0x801+i*0x10]; }

static u64 g_fsk_tex_ns;
static Sh* fs_get_keyed(GpuChannel* ch, u64 addr){
  u32* r=ch->r3d; u32 ctl=r[0x487]; u32 count=ctl&15; if(count>8) count=8; u32 rm=0, rs=0;
  for(u32 i=0;i<count;i++){ u32 idx=(ctl>>(4+i*3))&7; u32* rt=&r[0x200+idx*0x10]; if(!rt[4]) continue; GlFmt f=rt_format(rt[4]); if(f.is_int){ rm|=1u<<i; if(ifmt_signed(f.ifmt)) rs|=1u<<i; } }
  g_fs_rtk=0; g_fs_rtm=0; g_fs_rts=0;
  Sh* s0=shader_get(addr,ST_FS,0,0); if(!s0) return s0;
  u32 tm=0, ts=0; int nt=s0->si.ntex; if(nt>32) nt=32;
  if(nt){ u32 tcb=sh_tcb(s0,r[0x982]); u64 tic_base=((u64)r[0x55D]<<32)|r[0x55E]; u64 cba=g_cb.cb[4][tcb&31]; g_tic_base=tic_base; g_tic_lim=r[0x55F];
    for(int i=0;i<nt;i++){ ShTex* st=&s0->si.tex[i]; if(st->shadow) continue; u32 handle=sh_tex_handle(4,tcb,st); (void)cba;
      u32 tic[8]; gmem_read(tic_base+(handle&0xFFFFF)*32ull,tic,32); u32 ty=(tic[0]>>7)&7;
      u64 tt0=plat_ns(); Tex* t=tex_from_tic(tic); g_fsk_tex_ns+=plat_ns()-tt0;   /* texture loads: counted as textures, not shaders */
      if(t? t->f.is_int : (ty==3 || ty==4)){ tm|=1u<<i; if((t && ifmt_signed(t->f.ifmt)) || (!t && ty==3)) ts|=1u<<i; } } }
  rm&=s0->si.sph[18]? 0xFF : 0; if(!tm && !rm) return s0;
  g_fs_rtm=rm; g_fs_rts=rs; g_fs_rtk=rm*0x27D4EB2Fu ^ rs*0x165667B1u ^ 0x5A5A0000u;
  Sh* s=shader_get(addr,ST_FS,tm,ts); g_fs_rtk=0; g_fs_rtm=0; g_fs_rts=0; return s; }

/* transparent 0,0,0,0 for bad TICs, an empty GL unit gives opaque black */
static GLuint null_tex(GLenum tg){
  static GLuint n2, na, nc, nca; static const u8 z[4*6]={0};
  GLuint* p= tg==GL_TEXTURE_2D? &n2 : tg==GL_TEXTURE_2D_ARRAY? &na : tg==GL_TEXTURE_CUBE_MAP? &nc : tg==GL_TEXTURE_CUBE_MAP_ARRAY? &nca : NULL;
  if(!p) return 0; if(*p) return *p;
  glGenTextures(1,p); glBindTexture(tg,*p);
  if(tg==GL_TEXTURE_2D){ glTexStorage2D(tg,1,GL_RGBA8,1,1); glTexSubImage2D(tg,0,0,0,1,1,GL_RGBA,GL_UNSIGNED_BYTE,z); }
  else if(tg==GL_TEXTURE_CUBE_MAP){ glTexStorage2D(tg,1,GL_RGBA8,1,1); for(int f=0;f<6;f++) glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+f,0,0,0,1,1,GL_RGBA,GL_UNSIGNED_BYTE,z); }
  else { int d= tg==GL_TEXTURE_CUBE_MAP_ARRAY? 6 : 1; glTexStorage3D(tg,1,GL_RGBA8,1,1,d); glTexSubImage3D(tg,0,0,0,0,1,1,d,GL_RGBA,GL_UNSIGNED_BYTE,z); }
  mb_reset(); return *p; }

static int getenv_cached_nolodrt(void){ static int v=-1; if(v<0) v=getenv("SWR_NO_LOD_RT")!=NULL; return v; }   /* 1 = always read the assembled chain */

static int getenv_cached_rtswz(void){ static int v=-1; if(v<0) v=getenv("SWR_OLD_RT_SWIZZLE")!=NULL; return v; }   /* 1 = ignore reorders on RTs */

static GLenum img_fmt(GLenum f);
/* an image (SUST) TIC can point at one mip of a chain: res_min_mip_level, TIC word 7 bits 0..3. sonic frontiers
   builds its depth pyramid (2D) and its light probe cubemaps (6 mips per dispatch, bound as 2D arrays) that way.
   writing every mip into mip 0 left the probe mips black, and the deferred pass reads rough surfaces from them */
static GLint img_level(const u32* tic, const Tex* t){
  static int off=-1, only2d=-1; if(off<0) off=getenv("SWR_NO_IMG_LEVEL")!=NULL; if(only2d<0) only2d=getenv("SWR_IMG_LEVEL_2D_ONLY")!=NULL;
  u32 l=tic[7]&0xF; if(off || !l || !t || l>=t->levels || (only2d && t->target!=GL_TEXTURE_2D)) return 0;
  return (GLint)l; }

/* images a fragment shader writes with SUST: bound like compute does, from the stage's texture cbuf */
static int bind_images(GpuChannel* ch, Sh* s){
  if(!s || !s->si.nimg) return 0; u32* r=ch->r3d; u32 tcb=sh_tcb(s,r[0x982]); u64 tic_base=((u64)r[0x55D]<<32)|r[0x55E]; int any=0;
  for(int i=0;i<s->si.nimg && i<8;i++){ u32 bank=(s->si.img[i].bindless? s->si.img[i].bl_cbuf : tcb)&31; u64 cba= bank<18? g_cb.cb[4][bank] : 0;
    u32 h= cba? gmem_r32(cba+s->si.img[i].cbuf_off*4) : 0; u32 tic[8]; gmem_read(tic_base+(h&0xFFFFF)*32ull,tic,32);
    Tex* t=tex_from_tic(tic); GLenum f= t? img_fmt(t->f.ifmt) : 0;
    if(t && f){ int ty=s->si.img[i].type; glBindImageTexture(i,t->id,img_level(tic,t),(ty==2||ty==4||ty==5)? GL_TRUE : GL_FALSE,0,GL_WRITE_ONLY,f); t->gpu_written=1; t->wgen++; any=1;
      static int said; if(said<sw2_cap(4)){ said++; LOG("gpu: fragment shader writes image %ux%u at %llx (format %x)\n",t->w,t->h,(unsigned long long)t->gva,f); } }
    else glBindImageTexture(i,0,0,GL_FALSE,0,GL_WRITE_ONLY,GL_R32F); }
  return any; }

/* a draw that samples one of its own colour targets (MKW's tonemap reads the HDR scene and writes the result over it)
   gets a copy of it. On the console that works pixel for pixel; in GL it is a feedback loop with undefined results,
   and NVIDIA returned wrong values for the packed R11G11B10F scene. Copied once per draw. SWR_NO_FEEDBACK_COPY=1: off */
static u64 st_fb_copy;
static Tex* rt_feedback_copy(Tex* t){
  static int off=-1; if(off<0) off=getenv("SWR_NO_FEEDBACK_COPY")!=NULL;
  if(off || !t || !t->is_rt || !p_cis || t->target!=GL_TEXTURE_2D || t->f.depth) return t;
  int bound=0; for(int i=0;i<g_nrt;i++) if(g_rt[i]==t) bound=1; if(!bound) return t;
  static Tex shadow[8]; static int ns; u32 W=TD(t,t->w), H=TD(t,t->h);
  if(!t->fbc || t->fbc_w!=W || t->fbc_h!=H){ if(t->fbc) glDeleteTextures(1,&t->fbc);
    glGenTextures(1,&t->fbc); glBindTexture(GL_TEXTURE_2D,t->fbc); glTexStorage2D(GL_TEXTURE_2D,1,t->f.ifmt,W,H);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,0); t->fbc_w=W; t->fbc_h=H; t->fbc_draw=~0u;
    static int said; if(said<sw2_cap(6)){ said++; LOG("gpu: a draw samples its own render target %llx %ux%u: it reads a copy\n",(unsigned long long)t->gva,t->w,t->h); } }
  if(t->fbc_draw!=(u32)st_drawn){ p_cis(t->id,GL_TEXTURE_2D,0,0,0,0,t->fbc,GL_TEXTURE_2D,0,0,0,0,(GLsizei)W,(GLsizei)H,1); t->fbc_draw=(u32)st_drawn; st_fb_copy++; }
  glBindTexture(GL_TEXTURE_2D,t->fbc); glTexParameteriv(GL_TEXTURE_2D,GL_TEXTURE_SWIZZLE_RGBA,t->swz);
  /* a stand-in Tex with the copy's id; everything else as the RT */
  Tex* c=&shadow[ns++&7]; *c=*t; c->id=t->fbc; c->view=0; return c; }

static void bind_textures(GpuChannel* ch, int stage, Sh* s){
  u32* r=ch->r3d; u32 tcb=sh_tcb(s,r[0x982]);
  u64 tic_base=((u64)r[0x55D]<<32)|r[0x55E], tsc_base=((u64)r[0x557]<<32)|r[0x558]; g_tic_base=tic_base; g_tic_lim=r[0x55F];
  int mb= p_bt!=NULL, slot=(stage==0? 0 : stage==4? 1 : 2), nt=s->si.ntex; if(nt>32) nt=32;
  GLuint tids[32], sids[32];
  for(int i=0;i<nt;i++){ ShTex* st=&s->si.tex[i];
    u32 handle=sh_tex_handle(stage,tcb,st);
    u32 tic_id=handle&0xFFFFF, tsc_id=(handle>>20)&0xFFF;
    u32 tic[8], tsc[8]; gmem_read(tic_base+tic_id*32ull,tic,32); gmem_read(tsc_base+tsc_id*32ull,tsc,32);
    Tex* t=tex_from_tic(tic); if(stage==4) t=rt_feedback_copy(t);
    /* SMO's bloom reads one mip of a chain it renders level by level, with the sampler clamped to that level (min
       LOD = max LOD = L). we bind that level's own render target instead of going through the chain, on some
       drivers the level copies come back as garbage (blue screen) */
    if(t && t->is_rt && t->target==GL_TEXTURE_2D && t->levels>1 && st->type<=1 && !st->array && !getenv_cached_nolodrt()){
      u32 mn=tsc[2]&0xFFF, mx=(tsc[2]>>12)&0xFFF;
      if(mn==mx && mn && !(mn&0xFF) && (mn>>8)<t->levels){
        u32 L=mn>>8, lw=t->w>>L, lh=t->h>>L; u64 off=0;
        for(u32 l=0;l<L;l++) off+=bl_level_size(t->w,t->h,1,&t->f,t->bh,0,l,NULL);
        Tex* lv=tex_find_rt(t->gva+off); if(!lw) lw=1; if(!lh) lh=1;
        /* the level RTs can be wider than the level: SMO renders the 200x114 level into a 208x114 RT (rounded to a
           GOB row), 50x28 into 64x28 and so on, with the viewport at the real size, so only that corner gets read.
           the level copies are the other way and they come out as garbage on some GPUs: blue on some, NaN on others,
           and then the tonemap turns the whole screen black */
        if(lv && lv!=t && lv->target==GL_TEXTURE_2D && lv->f.ifmt==t->f.ifmt && rt_w_fits(lv,lw) && lv->h==lh){
          lv->last_used=g_frame; t= lv->w==lw? lv : rt_crop(lv,lw,lh); tsc[2]&=~0xFFFFFFu;
          static int said; if(!said){ said=1; LOG("gpu: mip %u of %ux%u read from its own render target (sampler pinned to that level)\n",L,lv->w,lv->h); } } } }
    int unit=(stage==0? 0 : stage==4? 1 : 2)*32+i; if(!mb) glActiveTexture(GL_TEXTURE0+unit);
    /* the shader's sampler type wins */
    GLenum want= st->type==2? GL_TEXTURE_3D : st->type==3? (st->array? GL_TEXTURE_CUBE_MAP_ARRAY : GL_TEXTURE_CUBE_MAP) : st->array? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;
    tids[i]=0;
    GLuint zc_id= (st->shadow && t && want==GL_TEXTURE_2D && !t->f.depth)? shadow_depth_copy(t) : 0;
    if(zc_id){ if(mb) tids[i]=zc_id; else glBindTexture(want,zc_id); st_tex_ok++; g_tex_any_ok=1; }
    /* a 3D texture with nothing in memory only gets the neutral LUT when it has a LUT's shape (N x N x N): MKW's 64x36x256 volumetric fog grid, written by compute, got an identity table and every pixel was tinted by
       its own screen position (blue to yellow) */
    else if(t && t->target==want && !(want==GL_TEXTURE_3D && t->zero3d && !t->is_rt && !t->gpu_written && t->w==t->h && t->d==t->w && t->w<=64)){ if(mb) tids[i]=t->id; else glBindTexture(want,t->id); if(t->is_rt) st_tex_rt++; else { st_tex_ok++; g_draw_asset=1; } g_tex_any_ok=1;
      if(want==GL_TEXTURE_3D){ static u64 seen3[16]; static int n3; int k=0; for(;k<n3;k++) if(seen3[k]==t->gva) break;
        if(k==n3 && n3<16){ seen3[n3++]=t->gva; LOG("gpu: 3D texture %ux%ux%u at %llx %s\n",t->w,t->h,t->d,(unsigned long long)t->gva,t->is_rt? "rendered by the GPU (3D render target)" : "read from memory (has data)"); } } }
    else if(want==GL_TEXTURE_3D){ /* 3D LUT with no data (whatever writes it isn't emulated yet): use a neutral
                                     table instead of black */
      /* identity table at the LUT's size. an sRGB->linear curve is the right thing in theory for float LUTs, but
         next to real hw it was way too dark and identity is close. SWR_LUT_LINEAR=1 for the curve */
      int lin_out= 0;
      static int force_lin=-1; if(force_lin<0) force_lin=getenv("SWR_LUT_LINEAR")!=NULL;
      if(force_lin && t && (t->f.ifmt==GL_R11F_G11F_B10F || t->f.ifmt==GL_RGBA16F || t->f.ifmt==GL_RGBA32F)) lin_out=1;
      /* has to match the LUT's real size: the shader samples texel centers (c*(N-1)+0.5)/N, so a 32^3 read as 8^3
         lifts black to ~6% gray */
      int NN= t && t->w>=2 && t->w<=64 && t->w==t->h && t->d==t->w? (int)t->w : 32;
      static GLuint lut_tex[65][2]; GLuint* lt=&lut_tex[NN][lin_out];
      if(!*lt){ const int N=NN; float* px=malloc((size_t)N*N*N*4*4);
        for(int z=0;z<N;z++) for(int y=0;y<N;y++) for(int x=0;x<N;x++){ float* q=px+((size_t)(z*N+y)*N+x)*4; int c3[3]={x,y,z};
          for(int k=0;k<3;k++){ float v=(float)c3[k]/(float)(N-1); if(lin_out) v= v<=0.04045f? v/12.92f : powf((v+0.055f)/1.055f,2.4f); q[k]=v; } q[3]=1.f; }
        glGenTextures(1,lt); glBindTexture(GL_TEXTURE_3D,*lt); glTexStorage3D(GL_TEXTURE_3D,1,GL_RGBA16F,N,N,N); glTexParameteri(GL_TEXTURE_3D,GL_TEXTURE_MAX_LEVEL,0);
        glPixelStorei(GL_UNPACK_ALIGNMENT,1); glPixelStorei(GL_UNPACK_ROW_LENGTH,0);
        glTexSubImage3D(GL_TEXTURE_3D,0,0,0,0,N,N,N,GL_RGBA,GL_FLOAT,px); free(px); }
      if(mb) tids[i]=*lt; else glBindTexture(GL_TEXTURE_3D,*lt); st_tex_mismatch++;
      { static u64 seenz[8]; static int nz; int k=0; u64 g= t? t->gva : 0; for(;k<nz;k++) if(seenz[k]==g) break;
        if(k==nz && nz<8){ seenz[nz++]=g; LOG("gpu: 3D texture (color LUT) %ux%ux%u at %llx (fmt %x) has no data: using a %s LUT (SWR_LUT_LINEAR=1: sRGB->linear)\n",
            t? t->w:0,t? t->h:0,t? t->d:0,(unsigned long long)g,t? t->fkey:0,lin_out? "neutral sRGB->linear" : "identity"); } } }
    else if(t && tex_view(t,want)){ GLuint v=t->view; if(mb) tids[i]=v; else glBindTexture(want,v); }
    else if(!t && null_tex(want)){ GLuint z=null_tex(want); if(mb) tids[i]=z; else glBindTexture(want,z); st_tex_null++;
      /* why a texture resolves to nothing: the handle (and where it came from), the TIC pool and if it's mapped.
         once per shader slot, only the first few */
      { static u64 seenn[12]; static int nn; u64 sk=s->hash*31+(u64)i; int k=0; for(;k<nn;k++) if(seenn[k]==sk) break;
        if(k==nn && nn<12){ seenn[nn++]=sk; extern u64 gmmu_translate(u64);
          u32 bank=(st->bindless? st->bl_cbuf : tcb)&31; u64 cba= bank<18? g_cb.cb[stage][bank] : 0;
          LOG("gpu: texture slot %d of shader %016llx (stage %d) has no texture: handle %x from cbuf %u+%#x%s (cbuf at %llx, %s), TIC pool %llx (cpu %llx, limit %x) entry %x: %08x %08x %08x %08x %08x %08x %08x %08x\n",
            i,(unsigned long long)s->hash,stage,handle,bank,st->cbuf_off*4,st->bindless? (st->bl2? " bindless, split" : " bindless") : "",(unsigned long long)cba,cba? (gmmu_translate(cba)? "mapped" : "NOT mapped") : "not bound",
            (unsigned long long)tic_base,(unsigned long long)(tic_base? gmmu_translate(tic_base) : 0),r[0x55F],tic_id,tic[0],tic[1],tic[2],tic[3],tic[4],tic[5],tic[6],tic[7]);
          if(nn==1 && tic_base){ /* what the pool holds: the first valid looking entries */
            int nz=0, shown=0; for(u32 e=0;e<4096;e++){ u32 w8[8]; gmem_read(tic_base+e*32ull,w8,32); if(!(w8[1]|w8[2]|w8[4])) continue; nz++;
              if(shown<4){ shown++; LOG("      TIC %x: %08x %08x %08x %08x %08x %08x %08x %08x\n",e,w8[0],w8[1],w8[2],w8[3],w8[4],w8[5],w8[6],w8[7]); } }
            LOG("      %d of the first 4096 TIC entries are non-zero\n",nz); } } } }
    else { if(!mb) glBindTexture(want,0); if(!t) st_tex_null++; else { st_tex_mismatch++;
        static int said; if(said<sw2_cap(6)){ said++; LOG("gpu: texture %ux%ux%u TIC type %u does not match the shader sampler (wants %x, has %x)\n",t->w,t->h,t->d,(tic[4]>>23)&0xF,want,t->target); } } }
    /* sRGB storage read through a non-sRGB TIC returns raw values on hw */
    /* sRGB storage read through a non-sRGB TIC gives raw values on hw. in GL the sampler's SRGB_DECODE wins over
       the texture's, so it's a sampler flag (tsc[7] bit 31) */
    int srgb_raw= t && t->f.ifmt==GL_SRGB8_ALPHA8 && !((tic[4]>>22)&1);
    /* compare mode comes from the shader's sampler type, a non-shadow sampler with compare on reads 0 in GL */
    if(st->shadow){ if(!((tsc[0]>>9)&1)){ tsc[0]|=1u<<9; tsc[0]=(tsc[0]&~(7u<<10))|(3u<<10); } } else tsc[0]&=~(1u<<9);
    if(t && t->f.is_int) tsc[1]=(tsc[1]&~0xF3u)|0x51u;   /* int textures must be NEAREST, LINEAR makes them incomplete */
    if(srgb_raw) tsc[7]|=0x80000000u;
    if(mb) sids[i]=sampler_get(tsc); else glBindSampler(unit,sampler_get(tsc));
    /* TIC swizzle */
    if(!zc_id && t && (t->target==want || (t->view && t->view_target==want))){ static const GLint sw[8]={GL_ZERO,GL_ZERO,GL_RED,GL_GREEN,GL_BLUE,GL_ALPHA,GL_ONE,GL_ONE};
      GLint m[4]={sw[(tic[0]>>19)&7],sw[(tic[0]>>22)&7],sw[(tic[0]>>25)&7],sw[(tic[0]>>28)&7]};
      if((t->fkey&0x7F)==0x73) m[3]=GL_ONE; /* RGBX's padding component is opaque */
      /* RTs: only swizzles with constant 0/1 get applied (like an R8 HUD mask read as 1,1,1,R). pure reorders are
         left alone since the RT format already has the host order */
      /* RTs: the TIC swizzle is in memory component order but the GL texture is in logical order, so only RTs
         stored BGRA (A8R8G8B8 / X8R8G8B8) need R/B swapped. SMO reads its light buffer as GBAR and R8 masks as
         RRRR */
      if(t->is_rt && !getenv_cached_rtswz()){ u32 rf=t->fkey&0xFF; int bgra= rf==0xCF || rf==0xD0 || rf==0xE6 || rf==0xE7;
        if(bgra) for(int k=0;k<4;k++){ if(m[k]==GL_RED) m[k]=GL_BLUE; else if(m[k]==GL_BLUE) m[k]=GL_RED; } }
      else if(t->is_rt){ int hc=0; for(int k=0;k<4;k++) if(m[k]==GL_ZERO||m[k]==GL_ONE) hc=1;
        if(!hc){ m[0]=GL_RED; m[1]=GL_GREEN; m[2]=GL_BLUE; m[3]=GL_ALPHA; } }
      if(t->f.depth){ m[0]=GL_RED; m[1]=GL_RED; m[2]=GL_RED; m[3]=GL_RED; }
      int isv= t->target!=want; GLint* cur= isv? t->vswz : t->swz; GLuint tid= isv? t->view : t->id;
      if(memcmp(cur,m,sizeof m)){ memcpy(cur,m,sizeof m); if(mb) glBindTexture(want,tid); /* use the scratch unit */ else glBindTexture(want,tid); glTexParameteriv(want,GL_TEXTURE_SWIZZLE_RGBA,m); } }
  }
  if(mb && nt>0){ int base=slot*32;
    if(g_tb_ok[slot] && g_tb_n[slot]==nt && !memcmp(g_tb_last[slot],tids,(size_t)nt*4) && !memcmp(g_sb_last[slot],sids,(size_t)nt*4)){ st_tb_skip++; return; }
    if(!g_tb_ok[slot] || g_tb_n[slot]!=nt || memcmp(g_tb_last[slot],tids,(size_t)nt*4)) p_bt(base,nt,tids);
    if(!g_tb_ok[slot] || g_tb_n[slot]!=nt || memcmp(g_sb_last[slot],sids,(size_t)nt*4)) p_bs(base,nt,sids);
    memcpy(g_tb_last[slot],tids,(size_t)nt*4); memcpy(g_sb_last[slot],sids,(size_t)nt*4); g_tb_n[slot]=nt; g_tb_ok[slot]=1; } }

static u8* g_stage_buf; static size_t g_stage_cap;
/* buffer cache keyed by guest range, validated once per submit */
u32 g_submit_epoch;
typedef struct VbC { u64 start, len, hash, cpu; u32 epoch, used; GLuint id; struct VbC* next; u32 stable, chk_epoch; int dirty; u32 csseen; u64 shash; u32 schk; } VbC;

/* 64K blocks (by gpu va, hashed) that got compute output, with a generation. a cached buffer on one of them gets
   read again, so we don't walk the whole cache on every compute result */
static u32 g_csblk[65536], g_csgen;

static int cs_blk_dirty(u64 a, u64 n, u32 seen){ if(g_csgen==seen) return 0; for(u64 b=a>>16, e=(a+n-1)>>16; b<=e; b++) if(g_csblk[b&0xFFFF]>seen) return 1; return 0; }

static VbC* vbc[2048];
static u64 g_vbc_bytes, g_vbc_n;   /* what the buffer cache holds (driver memory) */
/* write-watch generation per 64K block; untouched ranges skip the reread/hash */
static u32* g_cgen; static int g_ww=-1; static u64 st_ww_skip; static int g_va_inval;
static void ww_touch(u64 cpu);

static void ww_mark(u64 off){ if(off<AS_SIZE) g_cgen[off>>16]=g_submit_epoch; }

static void ww_poll(void){
  if(!g_ww) return;
  if(g_ww<0){ g_ww= g_ww_ok? 1 : 0; if(g_ww){ g_cgen=calloc((size_t)(AS_SIZE>>16),4); if(!g_cgen) g_ww=0; else { extern void (*g_ww_touch)(u64); g_ww_touch=ww_touch; } } if(!g_ww) return; LOG("gpu: write tracking active (unchanged buffers are not re-read)\n"); }
  if(plat_ww_collect(g_mem,AS_SIZE,ww_mark)){ g_ww=0; LOG("gpu: write watch not available\n"); } }

static void ww_touch(u64 cpu){ if(g_cgen && cpu<AS_SIZE) g_cgen[cpu>>16]=g_submit_epoch+1; }

static int ww_dirty(u64 ga, u64 len, u32 since){
  for(u64 c=ga>>16, e=(ga+len-1)>>16; c<=e; c++) if(g_cgen[c]>since) return 1;
  return 0; }

void cs_mem_need(u64 a, u64 n); extern volatile int g_cs_any;

/* cheap check for big ranges: first/last 64 bytes + 62 spread out 32 byte blocks (~2K read). good for catching a
   new level loaded over old data, not small edits, the full check handles those */
static u64 sample_hash(u64 gva, u64 len){
  u8 tmp[2048+128]; u32 n=0;
  if(len<=sizeof tmp){ gmem_read(gva,tmp,len); return fast_hash(tmp,len,7); }
  gmem_read(gva,tmp,64); n=64; gmem_read(gva+len-64,tmp+n,64); n+=64;
  u64 step=(len-32)/62; for(int i=0;i<62;i++){ gmem_read(gva+(u64)i*step,tmp+n,32); n+=32; }
  return fast_hash(tmp,n,7); }

static int g_nosample=-1;
/* range fully inside a compute result that didn't land in guest memory yet: copy gpu to gpu instead of waiting,
   reading back and uploading again (SMO particles hit this every frame) */
static int cs_gpu_src(u64 a, u64 n, GLuint* buf, u32* off);
static u64 st_cs_gpucopy, st_cs_sync, st_cs_sync_ns;

static GLuint buf_cached(GLenum target, u64 start, u64 len){
  if(g_cs_any){ GLuint sb; u32 so;
    if(cs_gpu_src(start,len,&sb,&so)){
      u32 k=(u32)((start>>6)^(start>>20)^len)&2047;
      VbC* e=vbc[k]; for(;e;e=e->next) if(e->start==start && e->len==len) break;
      if(!e){ e=calloc(1,sizeof *e); e->start=start; e->len=len; glGenBuffers(1,&e->id); e->next=vbc[k]; vbc[k]=e; g_vbc_bytes+=len; g_vbc_n++; }
      else if(e->epoch!=g_submit_epoch) e->used++;
      glBindBuffer(GL_COPY_WRITE_BUFFER,e->id); glBufferData(GL_COPY_WRITE_BUFFER,(GLsizeiptr)len,NULL,GL_STATIC_DRAW);
      glBindBuffer(GL_COPY_READ_BUFFER,sb); glCopyBufferSubData(GL_COPY_READ_BUFFER,GL_COPY_WRITE_BUFFER,so,0,(GLsizeiptr)len);
      /* the cpu side copy lands later (cs_apply marks the block) and the reread finds the same data */
      e->hash=~0ull; e->stable=0; e->epoch=g_submit_epoch; e->chk_epoch=g_submit_epoch; e->csseen=g_csgen; e->cpu=0;
      st_cs_gpucopy++; { static int said; if(!said){ said=1; LOG("gpu: compute results reused on the GPU (no wait for the readback)\n"); } }
      glBindBuffer(target,e->id); return e->id; }
    cs_mem_need(start,len); }
  u32 k=(u32)((start>>6)^(start>>20)^len)&2047;
  VbC* e=vbc[k]; for(;e;e=e->next) if(e->start==start && e->len==len) break;
  if(e && e->csseen!=g_csgen){ if(cs_blk_dirty(start,len,e->csseen)){ e->epoch=0; e->chk_epoch=g_submit_epoch-100000u; e->stable=0; } e->csseen=g_csgen; }   /* written by a compute, reread */
  if(e && e->epoch!=g_submit_epoch && e->used<0xFFFFFFFFu) e->used++;   /* submits it was used in: 1 = a one-off range */
  if(e && e->epoch==g_submit_epoch){ glBindBuffer(target,e->id); return e->id; }
  /* nobody touched it since last time (still checked again every 240 submits) */
  if(e && g_ww>0 && e->cpu && g_submit_epoch-e->chk_epoch<240 && gmmu_translate(start)==e->cpu && !ww_dirty(e->cpu,len,e->chk_epoch)){
    e->epoch=g_submit_epoch; st_ww_skip++; glBindBuffer(target,e->id); return e->id; }
  /* big buffers that never changed (static geometry): check every 8 submits */
  if(e && g_ww<=0 && e->stable>=16 && len>=(64u<<10) && g_submit_epoch-e->chk_epoch<8){ e->epoch=g_submit_epoch; glBindBuffer(target,e->id); return e->id; }
  if(e && g_ww<=0 && e->stable>=32 && len<(64u<<10) && g_submit_epoch-e->chk_epoch<4){ e->epoch=g_submit_epoch; glBindBuffer(target,e->id); return e->id; }
  /* buffers stable for a long time: full check every 64/240 submits, sampled every 4 in between. there's no write
     tracking on proton/linux and without this the deck hashed ~9 MB of vertices a frame */
  if(g_nosample<0) g_nosample=getenv("SWR_NO_SAMPLE_CHECK")!=NULL;
  if(e && g_ww<=0 && !g_nosample && e->stable>=64 && len>=4096 && g_submit_epoch-e->chk_epoch<(e->stable>=256? 240u : 64u)){
    if(g_submit_epoch-e->schk<4){ e->epoch=g_submit_epoch; glBindBuffer(target,e->id); return e->id; }
    u64 sh=sample_hash(start,len); e->schk=g_submit_epoch; st_vb_hashed+=2048;
    if(sh==e->shash){ e->epoch=g_submit_epoch; glBindBuffer(target,e->id); return e->id; } }
  u8* src=gmem_ptr(start,len); u64 cpu= src? (u64)(src-g_mem) : 0;
  if(!src){ if(g_stage_cap<len){ u8* nb=realloc(g_stage_buf,len); if(!nb) return 0; g_stage_buf=nb; g_stage_cap=len; } gmem_read(start,g_stage_buf,len); src=g_stage_buf; }
  u64 h=fast_hash(src,len,0); st_vb_hashed+=len;
  if(!e){ e=calloc(1,sizeof *e); e->start=start; e->len=len; e->csseen=g_csgen; e->used=1; glGenBuffers(1,&e->id); e->next=vbc[k]; vbc[k]=e; e->hash=h^1; g_vbc_bytes+=len; g_vbc_n++; }
  glBindBuffer(target,e->id);
  if(e->hash!=h){ e->hash=h; glBufferData(target,(GLsizeiptr)len,src,GL_STATIC_DRAW); st_vb_up+=len; e->stable=0; } else if(e->stable<100000) e->stable++;
  if(e->stable>=63 && len>=4096){ e->shash=sample_hash(start,len); e->schk=g_submit_epoch; }
  e->cpu=cpu; e->epoch=g_submit_epoch; e->chk_epoch=g_submit_epoch; return e->id; }

/* ============================================================ F5 frame trace */
static int g_tr_arm, g_tr_on, g_tr_n, g_tr_img; static FILE* g_tr;
static Tex* g_tr_tex[2048]; static int g_tr_ntex;
static u64 g_tr_sh[1024]; static int g_tr_nsh; static int g_tr_gen;
/* every F5 gets its own folder: captures/traces/1, 2... */
static char g_tr_dir[64]="captures/traces";
/* constant buffers of every draw, each distinct content once */
static FILE* g_tr_cbf; static u64 g_tr_cbpos; static u64 g_tr_cbh[8192]; static u64 g_tr_cbo[8192];
static GLuint g_tr_fb, g_tr_tx;
#define TRW 256
#define TRH 144

static void trace_note(const char* fmt, ...){ if(!g_tr_on || !g_tr) return; va_list ap; va_start(ap,fmt); vfprintf(g_tr,fmt,ap); va_end(ap); }

void gpu_gl_note_compute(void){ }

void gpu_gl_trace(const char* s){ trace_note("%s",s); }

/* color | alpha side by side, row 0 at the top */
static void tr_save(const char* path, const u8* rgba, u32 w, u32 h){
  FILE* f=fopen(path,"wb"); if(!f) return; fprintf(f,"P6\n%u %u\n255\n",w*2,h);
  u8* row=malloc((size_t)w*6);
  for(u32 y=0;y<h;y++){ const u8* q=rgba+(size_t)y*w*4;
    for(u32 x=0;x<w;x++){ row[x*3]=q[x*4]; row[x*3+1]=q[x*4+1]; row[x*3+2]=q[x*4+2]; u8 a=q[x*4+3]; row[(w+x)*3]=a; row[(w+x)*3+1]=a; row[(w+x)*3+2]=a; }
    fwrite(row,1,(size_t)w*6,f); }
  free(row); fclose(f); }

static void tr_thumb(Tex* t, char* out){ state_dirty();
  strcpy(out,"-"); if(!t || t->f.depth || t->f.is_int || g_tr_img>=2500) return;
  if(!g_tr_fb){ glGenFramebuffers(1,&g_tr_fb); glGenTextures(1,&g_tr_tx); glBindTexture(GL_TEXTURE_2D,g_tr_tx); glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA16F,TRW,TRH); }
  GLboolean sc=glIsEnabled(GL_SCISSOR_TEST), sr=glIsEnabled(GL_FRAMEBUFFER_SRGB);
  glDisable(GL_SCISSOR_TEST); glDisable(GL_FRAMEBUFFER_SRGB); glColorMaski(0,1,1,1,1);
  glBindFramebuffer(GL_READ_FRAMEBUFFER,g_rfbo); glFramebufferTexture2D(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,t->id,0); glReadBuffer(GL_COLOR_ATTACHMENT0);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER,g_tr_fb); glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,g_tr_tx,0); { GLenum b=GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&b); }
  glBlitFramebuffer(0,0,(GLint)TW(t),(GLint)TH(t),0,0,TRW,TRH,GL_COLOR_BUFFER_BIT,GL_LINEAR);
  /* read as float: the image keeps HDR values past 1, so the mean/max of every result shows where an HDR chain
     blows up, and a change above 1 counts as a change (8 bit thumbnails called those "no change") */
  static float pf[TRW*TRH*4]; static u8 px[TRW*TRH*4];
  glBindFramebuffer(GL_READ_FRAMEBUFFER,g_tr_fb); glReadBuffer(GL_COLOR_ATTACHMENT0); glPixelStorei(GL_PACK_ALIGNMENT,4); glReadPixels(0,0,TRW,TRH,GL_RGBA,GL_FLOAT,pf);
  glBindFramebuffer(GL_FRAMEBUFFER,g_fbo); if(sc) glEnable(GL_SCISSOR_TEST); if(sr) glEnable(GL_FRAMEBUFFER_SRGB);
  u64 h=fast_hash((const u8*)pf,sizeof pf,0); if(h==t->tr_last){ strcpy(out,"(no change)"); return; }
  double m[3]={0}; float mx=0; u32 nan=0;
  for(int k=0;k<TRW*TRH*4;k++){ float v=pf[k]; if(v!=v){ nan++; v=0; } if((k&3)<3){ m[k&3]+=v; if(v>mx) mx=v; } px[k]= v<=0? 0 : v>=1? 255 : (u8)(v*255.0f+0.5f); }
  t->tr_last=h; char p[128]; snprintf(p,sizeof p,"%s/d%04d.ppm",g_tr_dir,g_tr_n); tr_save(p,px,TRW,TRH); g_tr_img++;
  /* and the bottom-centre region (in a race: the road in front of the kart), where colour casts show on grey */
  double rm[3]={0}; int rn=0; for(int y=0;y<TRH/4;y++) for(int x=TRW/3;x<2*TRW/3;x++){ const float* q=pf+((size_t)y*TRW+x)*4; for(int c=0;c<3;c++) rm[c]+= q[c]==q[c]? q[c] : 0; rn++; }
  snprintf(out,160,"d%04d.ppm mean=(%.3g,%.3g,%.3g) max=%.3g%s low=(%.3g,%.3g,%.3g)",g_tr_n,m[0]/(TRW*TRH),m[1]/(TRW*TRH),m[2]/(TRW*TRH),mx,nan? " NaN" : "",rm[0]/rn,rm[1]/rn,rm[2]/rn); }

static int tr_tex_id(Tex* t){ if(!t) return -1; for(int i=0;i<g_tr_ntex;i++) if(g_tr_tex[i]==t) return i; if(g_tr_ntex<2048){ g_tr_tex[g_tr_ntex]=t; return g_tr_ntex++; } return -1; }

static void tr_shader(u64 addr, Sh* sh, int stage){
  if(!sh) return; for(int i=0;i<g_tr_nsh;i++) if(g_tr_sh[i]==sh->hash) return; if(g_tr_nsh>=1024) return; g_tr_sh[g_tr_nsh++]=sh->hash;
  u8* b=malloc(0x8000); gmem_read(addr,b,0x8000); char p[128];
  snprintf(p,sizeof p,"%s/sh_%016llx_%s.bin",g_tr_dir,(unsigned long long)sh->hash,stage==ST_VS?"vs":"fs"); FILE* f=fopen(p,"wb"); if(f){ fwrite(b,1,0x8000,f); fclose(f); }
  /* the GLSL the draw really used (Switch 2 code goes through sm86.c first; translating the raw bytes here gave garbage) */
  char* g=NULL; if(sh->src) g=strdup(sh->src); else if(!g_sh_ampere){ ShaderInfo si; memset(&si,0,sizeof si); si.attr_int_mask=sh->si.attr_int_mask; si.attr_sint_mask=sh->si.attr_sint_mask; g=shader_to_glsl(b,0x8000,stage,&si); }
  if(g){ snprintf(p,sizeof p,"%s/sh_%016llx_%s.glsl",g_tr_dir,(unsigned long long)sh->hash,stage==ST_VS?"vs":"fs"); f=fopen(p,"wb"); if(f){ fputs(g,f); fclose(f); } free(g); }
  free(b);
  trace_note("  shader %016llx %s: %d untranslated%s%s, failed=%d\n",(unsigned long long)sh->hash,stage==ST_VS?"vs":"fs",sh->si.unsupported,sh->si.unsupported?" first: ":"",sh->si.first_unsupported,sh->failed); }

static void tr_rt_history(Tex* t);

static void tr_shader_at(u64 addr, u64 hash, int stage){
  for(int i=0;i<g_tr_nsh;i++) if(g_tr_sh[i]==hash) return; if(g_tr_nsh>=1024) return; g_tr_sh[g_tr_nsh++]=hash;
  u8* b=malloc(0x8000); if(!b) return; gmem_read(addr,b,0x8000);
  ShaderInfo si; memset(&si,0,sizeof si); char* gl= g_sh_ampere? NULL : shader_to_glsl(b,0x8000,stage,&si); free(b);
  if(gl){ char p[128]; snprintf(p,sizeof p,"%s/sh_%016llx_%s.glsl",g_tr_dir,(unsigned long long)hash,stage==ST_VS?"vs":"fs"); FILE* f=fopen(p,"wb"); if(f){ fputs(gl,f); fclose(f); } free(gl); } }

static void tr_textures(GpuChannel* ch, int stage, Sh* sh){
  u32* r=ch->r3d; u32 tcb=sh_tcb(sh,r[0x982]);
  u64 tic_base=((u64)r[0x55D]<<32)|r[0x55E], tsc_base=((u64)r[0x557]<<32)|r[0x558];
  for(int i=0;i<sh->si.ntex;i++){ ShTex* st=&sh->si.tex[i];
    u32 handle=sh_tex_handle(stage,tcb,st);
    u32 tic[8], tsc[8]; gmem_read(tic_base+(handle&0xFFFFF)*32ull,tic,32); gmem_read(tsc_base+((handle>>20)&0xFFF)*32ull,tsc,32);
    Tex* t=tex_from_tic(tic);
    trace_note("  %s tex%d: sampler type%u%s%s  tic fmt=%05x %ux%ux%u type=%u levels=%u srgb=%u addr=%llx swz=%03x -> %s #%d %ux%u  tsc %08x %08x %08x\n",stage==0?"vs":"fs",i,
      st->type,st->array?" array":"",st->shadow?" shadow":"",tic[0]&0x7FFFF,(tic[4]&0xFFFF)+1,(tic[5]&0xFFFF)+1,((tic[5]>>16)&0x3FFF)+1,(tic[4]>>23)&0xF,((tic[3]>>28)&0xF)+1,(tic[4]>>22)&1,
      (unsigned long long)(((u64)(tic[2]&0xFFFF)<<32)|tic[1]),(tic[0]>>19)&0xFFF,t? (t->is_rt? "RT" : "tex") : "NULL",tr_tex_id(t),t?t->w:0,t?t->h:0,tsc[0],tsc[1],tsc[2]);
    if(t && t->is_rt){ Tex* w= t->wfs? t : t->lay && t->lay->wfs? t->lay : NULL; Tex* src= w? w : tex_find_rt(t->gva);
      if(src && src->wfs) trace_note("     written by fs %016llx, frame %u (now %u)\n",(unsigned long long)src->wfs,src->wframe,g_frame);
      tr_rt_history(src? src : t); } } }

/* diag: remember which fragment shader last wrote each RT, and keep the code of the ones that write small targets
   (cubemap faces, LUTs), those are usually drawn once at load way before a capture */
static float g_last_at[4];   /* last alpha test uniform, the layer-by-layer path reuses it */
static void rt_note_writer(GpuChannel* ch, Sh* fs, u64 code){
  u32* r=ch->r3d; static u64 dumped[64]; static int nd;
  for(int i=0;i<g_nrt;i++){ Tex* t=g_rt[i]; if(!t) continue; t->wfs=fs->hash; t->wframe=g_frame; if(t->vol){ t->vol->wfs=fs->hash; t->vol->wframe=g_frame; }
    if(t->w>256 || t->h>256 || !(r[0x850]&1)) continue;
    int seen=0; for(int k=0;k<nd;k++) if(dumped[k]==fs->hash) seen=1;
    if(seen || nd>=64) continue; dumped[nd++]=fs->hash;
    plat_mkdir("captures"); plat_mkdir("captures/rtw"); char p[200]; snprintf(p,sizeof p,"captures/rtw/sh_%016llx_fs.bin",(unsigned long long)fs->hash);
    u8* b=malloc(0x10000); if(!b) return; gmem_read(stage_prog(ch,r,code,5),b,0x10000); FILE* f=fopen(p,"wb"); if(f){ fwrite(b,1,0x10000,f); fclose(f); } free(b);
    LOG("diag: fs %016llx writes %ux%u RT %llx (frame %u)\n",(unsigned long long)fs->hash,t->w,t->h,(unsigned long long)t->gva,g_frame); } }

/* every constant buffer the shader reads: where, bound size, how far the shader reads, and its bytes in cbdata.bin
   (same content stored once). "reads past the end" means the shader goes beyond what the game bound (hw gives 0) */
static void tr_cb_at(const char* who, int ci, u64 ca, u32 bound, u32 rd){
    u32 n= bound && bound<=0x10000? bound : 0x10000; if(rd && rd<65536 && n>rd) n=rd;
    u8* b=malloc(n); if(!b) return; gmem_read(ca,b,n);
    u64 h=fast_hash(b,n,0)^n; u32 k=(u32)(h%8192); u64 off=~0ull;
    for(u32 j=0;j<8192;j++){ u32 s=(k+j)%8192; if(g_tr_cbh[s]==h){ off=g_tr_cbo[s]; break; }
      if(!g_tr_cbh[s]){ if(g_tr_cbf && g_tr_cbpos<(64u<<20)){ fwrite(b,1,n,g_tr_cbf); g_tr_cbh[s]=h; g_tr_cbo[s]=off=g_tr_cbpos; g_tr_cbpos+=n; } break; } }
    free(b);
    trace_note("  %s cb%d @%llx bound=%x reads=%x%s data=cbdata.bin+%llx (%x bytes)\n",who,ci,(unsigned long long)ca,bound,rd,
      rd>bound && rd<65536? " (reads past the end)" : rd>=65536? " (indexed)" : "",(unsigned long long)off,n); }

static void tr_cbufs(Sh* sh, int stg){
  if(!sh) return;
  for(int ci=0;ci<18;ci++) if(sh->si.cbuf_mask&(1u<<ci)){
    u64 ca=g_cb.cb[stg][ci]; u32 bound=g_cb.cbsize[stg][ci], rd=sh->si.cbuf_max[ci];
    if(!ca){ trace_note("  %s cb%d: not bound (reads %x)\n",stg? "fs" : "vs",ci,rd); continue; }
    tr_cb_at(stg? "fs" : "vs",ci,ca,bound,rd); } }

/* the last draws into each small render target. a trace only sees one frame, and things like the mii face texture
   are drawn once in some earlier frame; when a trace samples such a target it prints how it was made */
#define RTH_N 12
typedef struct RtHist { u32 frame, count; u64 vs, fs, vsa, fsa; u64 cb[6]; u32 cbsz[6], cbrd[6]; u8 cbi[6], ncb, ntx;
  u64 tx[6]; u32 txf[6], txw[6], txh[6]; u32 blend[8]; } RtHist;
static void rt_hist_note(GpuChannel* ch, Sh* vs, Sh* fs, u64 vsa, u64 fsa, u32 count){
  for(int i=0;i<g_nrt && i<8;i++){ Tex* t=g_rt[i]; if(!t || (u64)t->w*t->h>768u*768u) continue;
    if(!t->hist){ t->hist=calloc(RTH_N,sizeof(RtHist)); if(!t->hist) continue; }
    RtHist* h=&t->hist[t->nhist++%RTH_N]; memset(h,0,sizeof *h); u32* r=ch->r3d;
    h->frame=g_frame; h->count=count; h->vs=vs? vs->hash : 0; h->fs=fs? fs->hash : 0; h->vsa=vsa; h->fsa=fsa;
    h->blend[0]=r[0x4B9]; h->blend[1]=r[0x4D8+i]; h->blend[2]=r[0x780+i*8+1]; h->blend[3]=r[0x780+i*8+2]; h->blend[4]=r[0x780+i*8+3];
    h->blend[5]=r[0x780+i*8+4]; h->blend[6]=r[0x780+i*8+5]; h->blend[7]=r[0x780+i*8+6];
    if(fs && fsa){ for(int ci=0;ci<18 && h->ncb<6;ci++) if(fs->si.cbuf_mask&(1u<<ci)){ int k=h->ncb++; h->cbi[k]=(u8)ci; h->cb[k]=g_cb.cb[4][ci]; h->cbsz[k]=g_cb.cbsize[4][ci]; h->cbrd[k]=fs->si.cbuf_max[ci]; }
      u32 tcb=sh_tcb(fs,r[0x982]); u64 tic_base=((u64)r[0x55D]<<32)|r[0x55E];   /* the shader's own bank, like the draw */
      for(int k=0;k<fs->si.ntex && k<6;k++){ u32 hd=sh_tex_handle(4,tcb,&fs->si.tex[k]); u32 tic[8]; gmem_read(tic_base+(hd&0xFFFFF)*32ull,tic,32);
        h->tx[k]=((u64)(tic[2]&0xFFFF)<<32)|tic[1]; h->txf[k]=tic[0]&0x7FFFF; h->txw[k]=(tic[4]&0xFFFF)+1; h->txh[k]=(tic[5]&0xFFFF)+1; h->ntx=(u8)(k+1); } } } }

static void tr_cb_at(const char* who, int ci, u64 ca, u32 bound, u32 rd);
static void tr_shader_at(u64 addr, u64 hash, int stage);

/* a sampled render target that wasn't drawn in this frame: how it was drawn last time */
static void tr_rt_history(Tex* t){
  if(!t || !t->hist || !t->nhist || t->hist[(t->nhist-1)%RTH_N].frame==g_frame) return;   /* drawn this frame: the trace has it */
  static Tex* done[256]; static int ndone, gen; if(gen!=g_tr_gen){ gen=g_tr_gen; ndone=0; }
  for(int i=0;i<ndone;i++) if(done[i]==t) return; if(ndone<256) done[ndone++]=t;
  u32 n= t->nhist<RTH_N? t->nhist : RTH_N;
  trace_note("     last %u draws into %llx (%ux%u fmt=%x), oldest first:\n",n,(unsigned long long)t->gva,t->w,t->h,t->fkey);
  for(u32 j=0;j<n;j++){ RtHist* h=&t->hist[(t->nhist-n+j)%RTH_N];
    trace_note("     - frame %u count=%u vs=%016llx fs=%016llx blend indep=%u en=%u eq=%x src=%x dst=%x eqa=%x srca=%x dsta=%x\n",h->frame,h->count,(unsigned long long)h->vs,(unsigned long long)h->fs,
      h->blend[0],h->blend[1],h->blend[2],h->blend[3],h->blend[4],h->blend[5],h->blend[6],h->blend[7]);
    for(int k=0;k<h->ntx;k++){ trace_note("         tex%d %llx %ux%u fmt=%05x\n",k,(unsigned long long)h->tx[k],h->txw[k],h->txh[k],h->txf[k]);
      Tex* x=tex_find_rt(h->tx[k]); if(x) tr_tex_id(x); }   /* its stats at the end of the trace */
    for(int k=0;k<h->ncb;k++) if(h->cb[k]) tr_cb_at("    fs",h->cbi[k],h->cb[k],h->cbsz[k],h->cbrd[k]);
    if(h->vsa) tr_shader_at(h->vsa,h->vs,ST_VS); if(h->fsa) tr_shader_at(h->fsa,h->fs,ST_FS); } }

static void tr_draw(GpuChannel* ch, Sh* vs, Sh* fs, u64 vsa, u64 fsa, int indexed, u32 topology, u32 count, u32 instances){
  if(!g_tr_on || g_tr_n>=6000) return; u32* r=ch->r3d; g_tr_n++;
  float vt[6]; memcpy(vt,&r[0x280],24); float ref; memcpy(&ref,&r[0x4C4],4);
  trace_note("#%d DRAW vs=%016llx fs=%016llx count=%u inst=%u topo=%u idx=%d\n",g_tr_n,(unsigned long long)vs->hash,(unsigned long long)fs->hash,count,instances,topology,indexed);
  for(int i=0;i<g_nrt;i++) if(g_rt[i]) trace_note("  rt%d %llx %ux%u fmt=%x%s\n",i,(unsigned long long)g_rt[i]->gva,g_rt[i]->w,g_rt[i]->h,g_rt[i]->fkey,
      g_rt[i]->vol? (g_fb_lay[i]==-1? " (volume: all layers)" : " (volume: one layer)") : "");
  if(g_zt){ float dn,df; memcpy(&dn,&r[0x302],4); memcpy(&df,&r[0x303],4);
    trace_note("  z %llx %ux%u fmt=%x test=%u func=%x write=%u stencil=%u zscale=%.9g ztrans=%.9g range=%.9g..%.9g mode=%u\n",(unsigned long long)g_zt->gva,g_zt->w,g_zt->h,g_zt->fkey,r[0x4B3],r[0x4C3],r[0x4BA],r[0x4E0],vt[2],vt[5],dn,df,r[0x35F]&1); }
  trace_note("  blend en=%u,%u indep=%u common[sep=%u eq=%x src=%x dst=%x eqa=%x srca=%x dsta=%x] ind0[%x %x %x %x %x %x %x] mask=%x/%x common=%u srgb=%u alphatest=%u f=%x ref=%.3f cull=%u/%x front=%x vp(%.0f,%.0f,%.0f,%.0f) origin=%x yctl=%x vpctl=%x\n",
    r[0x4D8],r[0x4D9],r[0x4B9],r[0x4CF],r[0x4D0],r[0x4D1],r[0x4D2],r[0x4D3],r[0x4D4],r[0x4D6],r[0x780],r[0x781],r[0x782],r[0x783],r[0x784],r[0x785],r[0x786],r[0x680],r[0x681],r[0x3E4],r[0x4B8],r[0x4BB],r[0x4C5],ref,r[0x646],r[0x648],r[0x647],vt[0],vt[1],vt[3],vt[4],r[0x4EB],r[0x5E4],r[0x35F]);
  tr_shader(vsa,vs,ST_VS); if(fsa) tr_shader(fsa,fs,ST_FS);
  tr_textures(ch,0,vs); if(fsa) tr_textures(ch,4,fs);
  if(count<=6){ /* fullscreen passes (tonemap, fog, lighting): dump their constants */
    for(int st=0;st<2;st++){ Sh* sh= st? fs : vs; int stg= st? 4 : 0; if(!sh || (st && !fsa)) continue;
      for(int ci=0;ci<18;ci++) if(sh->si.cbuf_mask&(1u<<ci)){ u64 ca=g_cb.cb[stg][ci]; if(!ca) continue; float v[64]; gmem_read(ca,v,sizeof v);
        trace_note("  %s cb%d:",st? "fs" : "vs",ci); for(int k=0;k<16;k++) trace_note(" d[%d]=(%.4g,%.4g,%.4g,%.4g)",k,v[k*4],v[k*4+1],v[k*4+2],v[k*4+3]); trace_note("\n"); } } }
  tr_cbufs(vs,0); if(fsa) tr_cbufs(fs,4);
  char nm[160]; tr_thumb(g_rt[0],nm); trace_note("  result rt0: %s\n",nm); }

/* the GL texture can be bigger than the guest one (a texture that reads part of a bigger render target, or BCn
   stored rounded up to whole blocks), and glGetTexImage wrote past the buffer and broke the heap (BOTW and others
   crashed right after F5). glGetnTexImage takes the buffer size and refuses instead */
static int tr_teximage(GLenum tg, GLenum fmt, GLenum type, size_t cap, void* px){
  typedef void (APIENTRY *PFN_GNTI)(GLenum,GLint,GLenum,GLenum,GLsizei,void*);
  static PFN_GNTI p; static int init; if(!init){ init=1; p=(PFN_GNTI)backend_gl_proc("glGetnTexImage"); }
  memset(px,0,cap); if(!p || cap>0x7FFFFFFF) return 0;
  while(glGetError()!=GL_NO_ERROR){}
  p(tg,0,fmt,type,(GLsizei)cap,px);
  return glGetError()==GL_NO_ERROR; }

/* small textures go through an FBO and glReadPixels, layer by layer for 3D. glGetnTexImage gave zeros with no error,
   so white exposure textures and LUTs all read 0 */
static int tr_read_small(Tex* t, u32 W, u32 H, u32 D, float* px){
  memset(px,0,(size_t)W*H*D*16);
  GLuint fb=0; glGenFramebuffers(1,&fb); glBindFramebuffer(GL_READ_FRAMEBUFFER,fb); int ok=1;
  for(u32 z=0;z<D && ok;z++){
    if(t->target==GL_TEXTURE_3D) glFramebufferTextureLayer(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,t->id,0,(GLint)z);
    else glFramebufferTexture2D(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,t->id,0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    if(glCheckFramebufferStatus(GL_READ_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE){ ok=0; break; }
    glPixelStorei(GL_PACK_ALIGNMENT,4); glReadPixels(0,0,(GLsizei)W,(GLsizei)H,GL_RGBA,GL_FLOAT,px+(size_t)z*W*H*4); }
  glBindFramebuffer(GL_FRAMEBUFFER,g_fbo); glDeleteFramebuffers(1,&fb); state_dirty(); mb_reset(); return ok; }

static void trace_frame(void){
  /* SWR_WATCH: every 60 frames, how much of the first 64 KB at each watched address is nonzero (when does a
     table nobody seems to write get its data) */
  if(getenv("SWR_WATCH") && g_frame%60==0){ static u64 last[16]; const char* e=getenv("SWR_WATCH"); int n=0;
    while(e && *e && n<16){ char* q; u64 a=strtoull(e,&q,16); u8* b=malloc(65536); u32 nz=0;
      if(b){ gmem_read(a,b,65536); for(int k=0;k<65536;k++) nz+=b[k]!=0; free(b); }
      if(nz!=last[n]){ last[n]=nz; LOG("watch %llx: frame %u, %u of the first 64 KB nonzero\n",(unsigned long long)a,g_frame,nz); }
      n++; if(*q!=',') break; e=q+1; } }
  if(g_tr_on){
    trace_note("==== end of frame. textures used: %d ====\n",g_tr_ntex);
    /* every volume (3D or array RT, slices put together) too: they're often built once at load and only their
       result is sampled later (MKW's sky scattering tables) */
    for(int b=0;b<TEXH;b++) for(Tex* t=texh[b];t;t=t->next){ if(t->vol){ tr_tex_id(t->vol); tr_rt_history(t); } if(t->lay && t->lay->target==GL_TEXTURE_3D){ tr_tex_id(t->lay); tr_rt_history(t); } }
    for(int i=0;i<g_tr_ntex;i++){ Tex* t=g_tr_tex[i];
      trace_note("tex #%d %llx %ux%ux%u fmt=%x target=%x levels=%u rt=%d\n",i,(unsigned long long)t->gva,t->w,t->h,t->d,t->fkey,t->target,t->levels,t->is_rt);
      /* cubemaps and arrays: mean and max of every mip. a level left black (or NaN) shows up here, frontiers lights
         everything from a 128x128 sky cubemap and its shadow cascades are an array */
      if(!t->f.is_int && !t->f.compressed && (t->target==GL_TEXTURE_CUBE_MAP || t->target==GL_TEXTURE_CUBE_MAP_ARRAY || t->target==GL_TEXTURE_2D_ARRAY) && (u64)TW(t)*TH(t)*t->d<=1024*1024*4){
        glBindTexture(t->target,t->id); GLint ml=(GLint)t->levels-1;   /* the max level we set; glGetTexParameteriv is not in the GL queue */
        for(u32 l=0;l<t->levels && l<16;l++){ u32 w=TW(t)>>l, h=TH(t)>>l; if(!w) w=1; if(!h) h=1; u32 nl= t->target==GL_TEXTURE_CUBE_MAP? 6 : t->d;
          size_t n=(size_t)w*h*nl; float* px=malloc(n*16); if(!px) break;
          if(t->f.depth){ glGetTexImage(t->target==GL_TEXTURE_CUBE_MAP? GL_TEXTURE_CUBE_MAP_POSITIVE_X : t->target,(GLint)l,GL_DEPTH_COMPONENT,GL_FLOAT,px); for(size_t k=n;k-->0;){ float v=px[k]; px[k*4]=v; px[k*4+1]=v; px[k*4+2]=v; px[k*4+3]=1; } if(t->target==GL_TEXTURE_CUBE_MAP) n/=6; }
          else if(t->target==GL_TEXTURE_CUBE_MAP){ n/=6; glGetTexImage(GL_TEXTURE_CUBE_MAP_POSITIVE_X,(GLint)l,GL_RGBA,GL_FLOAT,px); }
          else glGetTexImage(t->target,(GLint)l,GL_RGBA,GL_FLOAT,px);
          double m[3]={0}; float mx=0; u32 nan=0;
          for(size_t k=0;k<n;k++) for(int c=0;c<3;c++){ float v=px[k*4+c]; if(v!=v){ nan++; continue; } m[c]+=v; if(v>mx) mx=v; }
          trace_note("   mip %u%s: mean=(%.4g,%.4g,%.4g) max=%.4g%s\n",l,(int)l>ml? " (past max level)" : "",m[0]/n,m[1]/n,m[2]/n,mx,nan? " NaN" : "");
          if(t->target==GL_TEXTURE_2D_ARRAY && l==0 && nl<=8){ size_t per=(size_t)w*h;   /* per layer: one shadow cascade each */
            for(u32 y=0;y<nl;y++){ double s=0; float lo=1e30f, hi=-1e30f; for(size_t k=0;k<per;k++){ float v=px[(y*per+k)*4]; s+=v; if(v<lo) lo=v; if(v>hi) hi=v; }
              trace_note("     layer %u: mean=%.4g min=%.4g max=%.4g\n",y,s/per,lo,hi); } }
          free(px); }
        mb_reset(); }
      if(!t->f.depth && !t->f.is_int && !t->f.compressed && (t->target==GL_TEXTURE_2D || t->target==GL_TEXTURE_3D) &&
         (t->target==GL_TEXTURE_3D? (u64)t->w*t->h*t->d<=(1u<<20) : (u64)t->w*t->h<=(t->is_rt? 4u<<20 : 4096u))){
        /* small textures (exposure), 3D LUTs and 2D render targets up to 4M texels: first texel, average, max */
        u32 W=TW(t), H=TH(t), D= t->target==GL_TEXTURE_3D? t->d : 1; float* px=malloc((size_t)W*H*D*16);
        if(px && !tr_read_small(t,W,H,D,px)){ free(px); px=NULL; trace_note("   (not read: the driver refused it as a framebuffer)\n"); }
        if(px){ double m[4]={0}; float mx=-1e30f; u32 nan=0;
          for(u32 k=0;k<W*H*D;k++) for(int c=0;c<4;c++){ float v=px[k*4+c]; if(v!=v){ nan++; continue; } m[c]+=v; if(c<3 && v>mx) mx=v; }
          trace_note("   texel0=(%.4g,%.4g,%.4g,%.4g) mean=(%.4g,%.4g,%.4g,%.4g) max rgb=%.4g%s\n",px[0],px[1],px[2],px[3],m[0]/(W*H*D),m[1]/(W*H*D),m[2]/(W*H*D),m[3]/(W*H*D),mx,nan? " NaN" : "");
          if(D>8) for(u32 z=0;z<D;z+=(D+15)/16){ double s[4]={0}; u32 bad=0; size_t per=(size_t)W*H;   /* volumes: which slices go wrong */
            for(size_t k=0;k<per;k++) for(int c=0;c<4;c++){ float v=px[((size_t)z*per+k)*4+c]; if(v!=v || v>6e4f || v<-6e4f){ bad++; continue; } s[c]+=v; }
            /* cells at x%4==0 && y%4==0 against the rest: one odd thread per 4x4 workgroup shows up as a regular grid of dots */
            double c0=0, c1=0; u32 n0=0, n1=0; for(u32 y=0;y<H;y++) for(u32 x=0;x<W;x++){ float v=px[((size_t)z*per+(size_t)y*W+x)*4]; if(v!=v || v>6e4f || v<-6e4f) continue; if(!(x&3) && !(y&3)){ c0+=v; n0++; } else { c1+=v; n1++; } }
            trace_note("     slice %u: mean=(%.4g,%.4g,%.4g,%.4g) inf/NaN %u, r at x,y%%4==0 %.4g vs others %.4g\n",z,s[0]/per,s[1]/per,s[2]/per,s[3]/per,bad,n0? c0/n0 : 0,n1? c1/n1 : 0); }
          /* cube-shaped volumes (colour LUTs): the diagonal i,i,i, where a grey input lands; a tint shows up here */
          if(W==H && H==D && D>=8){ trace_note("     diagonal:"); for(u32 i=0;i<D;i+=D/8){ const float* q=px+(((size_t)i*H+i)*W+i)*4; trace_note(" %u:(%.3g %.3g %.3g)",i,q[0],q[1],q[2]); } trace_note("\n");
            /* along each axis from the 3/4 corner: a colour LUT's x changes red, y green, z blue */
            { u32 c=D*3/4; static const char* an[3]={"x","y","z"};
              for(int ax=0;ax<3;ax++){ trace_note("     along %s:",an[ax]); for(u32 i=0;i<D;i+=D/8){ u32 x= ax==0? i : c, y= ax==1? i : c, z= ax==2? i : c;
                  const float* q=px+(((size_t)z*H+y)*W+x)*4; trace_note(" %u:(%.3g %.3g %.3g)",i,q[0],q[1],q[2]); } trace_note("\n"); } } }
          free(px); mb_reset(); } }
      if(t->f.depth || t->f.is_int || (t->target!=GL_TEXTURE_2D && t->target!=GL_TEXTURE_2D_ARRAY)) continue;
      /* GPU downsizes to max 512 and we read only that; layer 0 for arrays */
      u32 w=TW(t), h=TH(t);
      u32 st=1; while(w/st>512 || h/st>512) st*=2; u32 ow=w/st? w/st : 1, oh=h/st? h/st : 1;
      u8* o=malloc((size_t)ow*oh*4); if(!o) continue; int ok=0;
      if(!t->f.compressed){ GLuint tt=0, fb=0; glGenTextures(1,&tt); glBindTexture(GL_TEXTURE_2D,tt); glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA8,ow,oh);
        glGenFramebuffers(1,&fb); glBindFramebuffer(GL_READ_FRAMEBUFFER,g_rfbo);
        if(t->target==GL_TEXTURE_2D_ARRAY) glFramebufferTextureLayer(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,t->id,0,0); else glFramebufferTexture2D(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,t->id,0);
        glReadBuffer(GL_COLOR_ATTACHMENT0); glBindFramebuffer(GL_DRAW_FRAMEBUFFER,fb); glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,tt,0);
        if(glCheckFramebufferStatus(GL_READ_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE && glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER)==GL_FRAMEBUFFER_COMPLETE){
          glDisable(GL_SCISSOR_TEST); glDisable(GL_FRAMEBUFFER_SRGB); glColorMaski(0,1,1,1,1); { GLenum bb=GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&bb); }
          glBlitFramebuffer(0,0,(GLint)w,(GLint)h,0,0,(GLint)ow,(GLint)oh,GL_COLOR_BUFFER_BIT,GL_LINEAR);
          glBindFramebuffer(GL_READ_FRAMEBUFFER,fb); glReadBuffer(GL_COLOR_ATTACHMENT0); glPixelStorei(GL_PACK_ALIGNMENT,1); glReadPixels(0,0,ow,oh,GL_RGBA,GL_UNSIGNED_BYTE,o); ok=1; }
        glBindFramebuffer(GL_FRAMEBUFFER,g_fbo); glDeleteFramebuffers(1,&fb); glDeleteTextures(1,&tt); state_dirty(); mb_reset(); }
      if(!ok && t->target==GL_TEXTURE_2D && (u64)w*h<=(1u<<20)){ u8* px=malloc((size_t)w*h*4);
        if(px){ glBindTexture(GL_TEXTURE_2D,t->id); glPixelStorei(GL_PACK_ALIGNMENT,1);
          if(tr_teximage(GL_TEXTURE_2D,GL_RGBA,GL_UNSIGNED_BYTE,(size_t)w*h*4,px)){ for(u32 y=0;y<oh;y++) for(u32 x=0;x<ow;x++) memcpy(o+((size_t)y*ow+x)*4,px+((size_t)y*st*w+x*st)*4,4); ok=1; }
          free(px); } }
      if(ok){ char p[160]; snprintf(p,sizeof p,"%s/tex%03d_%llx_%ux%u_f%x%s.ppm",g_tr_dir,i,(unsigned long long)t->gva,t->w,t->h,t->fkey,t->is_rt?"_rt":""); tr_save(p,o,ow,oh); }
      free(o); }
    fclose(g_tr); g_tr=NULL; g_tr_on=0; if(g_tr_cbf){ fclose(g_tr_cbf); g_tr_cbf=NULL; }
    LOG("trace: frame saved in %s (%d draws, %d images)\n",g_tr_dir,g_tr_n,g_tr_img); fflush(NULL); }
  { static int envf=-2; if(envf==-2){ const char* e=getenv("SWR_TRACE"); envf= e? atoi(e) : -1; } if(envf>=0 && (int)g_frame==envf) g_tr_arm=1; }
  if(g_tr_arm){ g_tr_arm=0; plat_mkdir("captures"); plat_mkdir("captures/traces");
    for(int n=1;n<10000;n++){ char p[96]; snprintf(g_tr_dir,sizeof g_tr_dir,"captures/traces/%d",n); snprintf(p,sizeof p,"%s/trace.txt",g_tr_dir);
      FILE* e=fopen(p,"rb"); if(!e) break; fclose(e); }
    plat_mkdir(g_tr_dir); { char p[96]; snprintf(p,sizeof p,"%s/trace.txt",g_tr_dir); g_tr=fopen(p,"w"); snprintf(p,sizeof p,"%s/cbdata.bin",g_tr_dir); g_tr_cbf=fopen(p,"wb"); }
    g_tr_cbpos=0; memset(g_tr_cbh,0,sizeof g_tr_cbh);
    g_tr_on= g_tr!=NULL; g_tr_n=0; g_tr_img=0; g_tr_ntex=0; g_tr_nsh=0; g_tr_gen++;
    { extern const char* g_runtime_version; const char* v=(const char*)glGetString(GL_VENDOR); const char* r=(const char*)glGetString(GL_RENDERER); const char* x=(const char*)glGetString(GL_VERSION);
      trace_note("runtime %s, GL: %s / %s / %s, render scale %.2f, frame %u\n",g_runtime_version,v?v:"?",r?r:"?",x?x:"?",g_draw_scale,g_frame); }
    for(int k=0;k<TEXH;k++) for(Tex* t=texh[k];t;t=t->next) t->tr_last=0;
    LOG("trace: recording the next frame into %s...\n",g_tr_dir); }
}

static void lsfg_pump(void); static int g_lfn;
void cs_flush_pending(void);

static u32 gs_vertices_for_topology(u32 topology){
  switch(topology){
    case 0: return 1;
    case 1: case 2: case 3: return 2;
    case 10: case 11: return 4;
    case 12: case 13: return 6;
    default: return 3;
  }
}

/* fps_target 30 / 20: the game still runs 60 steps a second (normal speed) but only every 2nd / 3rd frame gets
   drawn and shown. compute, copies and clears still run, so the game's state on the gpu stays right */
int g_opt_frame_skip;
static int g_skip_frame;
/* fps_limit=30: every other image is skipped too, the game keeps its 60 steps a second */
static int frame_skip_count(void){ extern int g_opt_fps; int n=g_opt_frame_skip; if(g_opt_fps==30 && n<1) n=1; return n; }

/* ============================================================ task / mesh draws (Switch 2)
   while SET_MESH_CONTROL (0x453) is on, a draw runs a task program (VertexB slot) and a mesh program (TessCtrl slot),
   or only a mesh program (VertexB) with no task; the Geometry slot holds a header naming the per-primitive outputs.
   Translated to GL_NV_mesh_shader stages in the separable pipeline; the draw count is the number of task (or mesh)
   workgroups. Layout and registers as in Eglantine (ROADMAP.md 6.3). SWR_NO_MESH=1 skips these draws as before */
int g_has_mesh;
typedef void (APIENTRY*PFN_DMT_)(GLuint,GLuint);
/* the stage-buffer lines a program header's output map names: 0-7 system (primitive id, layer, viewport, point size,
   position xyzw), 8+ generic components */
static u32 mesh_lines(u64 hdr_addr, u8* out){
  u8 h[0x50]; gmem_read(hdr_addr,h,sizeof h); u32 n=0;
  for(u32 b=0;b<8;b++) if((h[0x35]>>b)&1) out[n++]=(u8)b;
  for(u32 g=0;g<32;g++){ u32 bits=h[0x36+g/2]>>((g%2)*4); for(u32 e=0;e<4;e++) if((bits>>e)&1 && n<136) out[n++]=(u8)(8+g*4+e); }
  return n; }

static void draw_mesh(GpuChannel* ch, u64 code, u32 count){
  u32* r=ch->r3d;
  static int off=-1; if(off<0) off=getenv("SWR_NO_MESH")!=NULL;
  static PFN_DMT_ p_dmt; if(!p_dmt && g_has_mesh) p_dmt=(PFN_DMT_)backend_gl_proc("glDrawMeshTasksNV");
  if(off || !g_has_mesh || !p_dmt || g_sep<=0){ static int said; if(!said){ said=1; LOG("gpu: task/mesh draws skipped (%s)\n",off? "SWR_NO_MESH" : g_sep<=0? "no separable programs" : "no GL_NV_mesh_shader"); } trace_note("DRAW SKIPPED: mesh\n"); return; }
  u32 A=r[0x454], B=r[0x455], I=r[0x456];
  if((A&15)!=4){ static int said; if(said<4){ said++; LOG("gpu: mesh draw with output topology %u (only triangles): skipped\n",A&15); } return; }
  int has_task=(r[0x820]&1)!=0;   /* the mesh program in the TessCtrl slot: VertexB holds a task */
  u64 task_a= has_task? stage_prog(ch,r,code,1) : 0, mesh_a= stage_prog(ch,r,code,has_task? 2 : 1), prim_a= (r[0x840]&1)? stage_prog(ch,r,code,4) : 0;
  ShaderInfo* m=&g_mesh_si; memset(m,0,sizeof *m);
  u32 lines=(I>>12)&0x1FF; if(((I>>22)&0x1FF)>lines) lines=(I>>22)&0x1FF;
  m->mesh_payload_words=(lines? lines : 1)*32; m->mesh_max_vertices=(A>>4)&0x1FF; m->mesh_max_primitives=(A>>13)&0x3FF;
  m->mesh_threads=I&0x3F; g_mesh_threads_m=(B>>20)&0x3F; m->mesh_has_task=has_task;
  m->mesh_nvlines=mesh_lines(mesh_a,m->mesh_vlines); if(prim_a) m->mesh_nplines=mesh_lines(prim_a,m->mesh_plines);
  g_mesh_key=(u32)fast_hash((const u8*)&m->mesh_payload_words,4*4+sizeof m->mesh_vlines+sizeof m->mesh_plines+8,0)^(A*31)^(B*17)^I;
  Sh* task= has_task? shader_get(task_a,ST_TASK,0,0) : NULL; Sh* mesh=shader_get(mesh_a,ST_MESH,0,0);
  Sh* fs= (r[0x850]&1)? fs_get_keyed(ch,stage_prog(ch,r,code,5)) : NULL;
  { static u64 seen[16]; static int ns; int k=0; for(;k<ns;k++) if(seen[k]==mesh_a) break;
    if(k==ns && ns<16){ seen[ns++]=mesh_a; LOG("gpu: mesh draw: task %llx mesh %llx, %u vertex / %u primitive lines, max %u vertices %u primitives, threads %u/%u, payload %u words\n",
      (unsigned long long)task_a,(unsigned long long)mesh_a,m->mesh_nvlines,m->mesh_nplines,m->mesh_max_vertices,m->mesh_max_primitives,m->mesh_threads,g_mesh_threads_m,m->mesh_payload_words); } }
  if(!mesh || !fs || (has_task && !task)){ st_noshader++; trace_note("DRAW SKIPPED: mesh draw without its programs\n"); return; }
  if(mesh->failed || fs->failed || (task && task->failed)){ st_shfail++; trace_note("DRAW SKIPPED: mesh draw with a failed program\n"); return; }
  if(!sh_sep_ready(mesh) || !sh_sep_ready(fs) || (task && !sh_sep_ready(task))){ st_compiling++; trace_note("DRAW SKIPPED: compiling\n"); return; }
  if(!mesh->sp || !fs->sp || (task && !task->sp)){ st_noprog++; trace_note("DRAW SKIPPED: mesh program not linked\n"); return; }
  g_draw_layered=0;
  bind_framebuffer(ch);
  if(!g_nrt && !g_zt){ st_nort++; trace_note("DRAW SKIPPED: no target\n"); return; }
  st_drawn++; wd_note(mesh->hash,fs->hash);
  if(!g_pipe_on){ glUseProgram(0); p_bpp(g_pipe); g_pipe_on=1; }
  p_ups(g_pipe,0x00000001|0x00000004 /* VERTEX | GEOMETRY */,0); g_pipe_vs=g_pipe_gs=0;
  p_ups(g_pipe,0x00000080 /* TASK_SHADER_BIT_NV */,task? task->sp : 0); p_ups(g_pipe,0x00000040 /* MESH_SHADER_BIT_NV */,mesh->sp);
  if(g_pipe_fs!=fs->sp){ p_ups(g_pipe,0x00000002,fs->sp); g_pipe_fs=fs->sp; }
  apply_state(ch);
  { float vt[2]; memcpy(vt,&r[0x280],8); float fl[4]={ (vt[1]<0)!=(origin_flip_h(r)!=0)? -1.f : 1.f, p_clip? 0.f : (float)(r[0x35F]&1), 0, 0 };
    if(mesh->uloc>=0) glProgramUniform4fv(mesh->sp,mesh->uloc,1,fl); mesh->lu_ok=0; }
  if(fs->uloc>=0){ u32 fn=r[0x4C5]; int f= fn>=0x200 && fn<=0x207? (int)(fn-0x200) : fn>=1 && fn<=8? (int)fn-1 : 7; float ref; memcpy(&ref,&r[0x4C4],4);
    float at[4]={ (r[0x4BB]&1)? 1.f : 0.f, (float)f, ref, 0 }; glProgramUniform4fv(fs->sp,fs->uloc,1,at); fs->lu_ok=0; }
  if(task) upload_cbufs(0,task); upload_cbufs(1,mesh); upload_cbufs(4,fs);
  g_ngw=0; if(task) bind_gbufs(0,task); bind_gbufs(1,mesh); bind_gbufs(4,fs);
  if(task) bind_textures(ch,0,task); bind_textures(ch,1,mesh); bind_textures(ch,4,fs); bind_images(ch,fs);
  p_dmt(0,count);
  /* back to the vertex pipeline for the next draw */
  p_ups(g_pipe,0x00000080|0x00000040,0);
  trace_note("  MESH DRAW task=%016llx mesh=%016llx fs=%016llx groups=%u\n",(unsigned long long)(task? task->hash : 0),(unsigned long long)mesh->hash,(unsigned long long)fs->hash,count); }

void r3d_draw_gl(GpuChannel* ch, int indexed, u32 topology, u32 first, u32 count, u32 base_vertex, u32 base_instance, u32 instances){
  if(!swr_render_enabled(ch->r3d)){ static int said; if(said<sw2_cap(4)){ said++; u32* q=ch->r3d; LOG("gpu: draw skipped by render enable (modes %u/%u, override %u)\n",q[0x4e]&7,q[0x556]&7,q[0x651]&3); } return; }
  if(g_cb_owner!=(int)ch->id && !getenv_cached_sharedcb()) cb_switch(ch);
  u32* r=ch->r3d; g_draws++;
  if(g_skip_frame) return;
  if(g_lfn && !(g_draws&15)) lsfg_pump();   /* push pending generated frames out on time even mid-frame */
  if(!count) return;
  /* shaders */
  u64 code=((u64)r[0x582]<<32)|r[0x583];
  g_sh_ampere=ch->ampere;   /* Switch 2: shaders are Ampere SASS, shader_get lowers them with sm86.c */
  if(ch->ampere && getenv("SWR_CB3_LOG")){ static int nl; if(nl<30){ nl++; LOG("slots: 0 cfg %08x at %08x%08x | 1 cfg %08x at %08x%08x | 5 cfg %08x | vs bank3 %llx/%x\n",r[0x800],r[0x805],r[0x806],r[0x810],r[0x815],r[0x816],r[0x850],(unsigned long long)g_cb.cb[0][3],g_cb.cbsize[0][3]); } }
  if(ch->ampere){ /* the first programs also go to captures/ampere_* to debug the translation offline */
    static u64 seen[64]; static int ns;
    for(int i=1;i<6;i+=4){ u64 a=stage_prog(ch,r,code,i); int k=0; for(;k<ns;k++) if(seen[k]==a) break;
      if(k==ns && ns<64){ seen[ns++]=a; { extern u64 gmmu_translate(u64); LOG("gpu: ampere %s program at %llx = region %llx + %x (cfg %08x, regs %08x %08x %08x), cpu %llx\n",i==1? "vs" : "fs",(unsigned long long)a,(unsigned long long)code,r[0x801+i*0x10],r[0x800+i*0x10],r[0x802+i*0x10],r[0x803+i*0x10],r[0x804+i*0x10],(unsigned long long)gmmu_translate(a)); if(!gmmu_translate(a)) LOG("      same offset under region 1/2: %llx / %llx, method 0x1608 regs %08x %08x\n",(unsigned long long)gmmu_translate(0x100000000ull+r[0x801+i*0x10]),(unsigned long long)gmmu_translate(0x200000000ull+r[0x801+i*0x10]),r[0x582],r[0x583]); }
        plat_mkdir("captures"); char fp[160]; snprintf(fp,sizeof fp,"captures/ampere_%s_%02d.bin",i==1? "vs" : "fs",ns);
        u8* b=malloc(0x10000); if(b){ gmem_read(a,b,0x10000); FILE* f=fopen(fp,"wb"); if(f){ fwrite(b,1,0x10000,f); fclose(f); } free(b); } } } }
  /* task / mesh draw: opt-in (SWR_MESH=1) until glDrawMeshTasksNV goes through the GL queue (its lookup on this
     thread fails and every mesh draw was skipped); by default these draws take the vertex path as before */
  { static int mesh=-1; if(mesh<0) mesh=getenv("SWR_MESH")!=NULL;
    if(mesh && ch->ampere && (r[0x453]&1)){ g_last_r3d=r; draw_mesh(ch,code,count); return; } }
  Sh* vs=NULL; Sh* gs=NULL; Sh* fs=NULL;
  u32 attr_int=0, attr_sint=0;
  for(int i=0;i<32;i++){ u32 a=r[0x458+i]; u32 ty=(a>>27)&7; if(ty==3||ty==4){ attr_int|=1u<<i; if(ty==3) attr_sint|=1u<<i; } }
  /* stage slots: 0 VertexA, 1 VertexB, 2 TessCtrl, 3 TessEval, 4 Geometry, 5 Fragment. VertexB is always on even
     with the enable bit clear (NVN doesn't set it) */
  g_last_r3d=r;
  g_gs_input_vertices=gs_vertices_for_topology(topology);
  for(int i=1;i<6;i+=4){ u32 cfg=r[0x800+i*0x10]; if(i!=1 && !(cfg&1)) continue; u64 addr=stage_prog(ch,r,code,i);
    if(i==1) TM(TM_SH,vs=shader_get(addr,ST_VS,attr_int,attr_sint));
    else { g_fsk_tex_ns=0; TM(TM_SH,fs=fs_get_keyed(ch,addr)); g_tm[TM_SH]-=g_fsk_tex_ns; g_tm[TM_TEX]+=g_fsk_tex_ns; } }
  int gs_on=(r[0x840]&1)!=0;
  if(gs_on && g_sep>0) TM(TM_SH,gs=shader_get(stage_prog(ch,r,code,4),ST_GS,0,0));
  if((r[0x820]|r[0x830])&1 || (gs_on && g_sep<=0)){ /* tessellation is not translated; geometry needs separable programs */
    static u64 gseen[16]; static int ng; u64 ga=stage_prog(ch,r,code,4); int k=0; for(;k<ng;k++) if(gseen[k]==ga) break;
    if(k==ng && ng<16){ gseen[ng++]=ga; LOG("gpu: draw uses unsupported stages:%s%s%s (gs at %llx, rt0 %ux%u layers %u 3d %u) - drawn without them\n",(r[0x820]&1)?" TessCtrl":"",(r[0x830]&1)?" TessEval":"",(r[0x840]&1)?" Geometry":"",
        (unsigned long long)ga,r[0x202],r[0x203],r[0x206]&0xFFFF,(r[0x205]>>16)&1);
      if(gs_on && g_sep<=0){ plat_mkdir("captures"); char fp[160]; snprintf(fp,sizeof fp,"captures/gs_%02d.bin",ng); u8* b=malloc(0x8000); if(b){ gmem_read(ga,b,0x8000); FILE* f=fopen(fp,"wb"); if(f){ fwrite(b,1,0x8000,f); fclose(f); } free(b); } } }
    trace_note("WARNING: unsupported stages enabled (tess/geometry)\n"); }
  if(vs && !fs){ /* depth-only passes: use an empty FS */
    static Sh nullfs; static char nsrc[64]; if(!nullfs.gl && !nullfs.failed){ char e[512]; snprintf(nsrc,sizeof nsrc,"#version %d core\nvoid main(){}\n",g_glsl_version); nullfs.stage=ST_FS; nullfs.hash=0x4E554C4C46530001ull; nullfs.src=nsrc; nullfs.gl=compile(GL_FRAGMENT_SHADER,nsrc,e,sizeof e); if(!nullfs.gl) nullfs.failed=1; }
    fs=&nullfs; }
  if(!vs || !fs){ st_noshader++; trace_note("DRAW SKIPPED: no shader\n"); return; }
  if(vs->failed || fs->failed || (gs && gs->failed)){ st_shfail++; trace_note("DRAW SKIPPED: failed shader vs=%016llx(%d) gs=%016llx(%d) fs=%016llx(%d)\n",(unsigned long long)vs->hash,vs->failed,(unsigned long long)(gs?gs->hash:0),gs?gs->failed:0,(unsigned long long)fs->hash,fs->failed); return; }
  Prog* p=NULL; GLuint vpp=0, gsp=0, atp=0; GLint vpl=-1, glp=-1, atl=-1;
  /* draws into tables (3D/array RTs, or up to 512x512 texels) usually happen once at level load. skipping one while
     its shader compiles leaves the table empty, so we wait for the compile here (one hitch). MKW's
     256x64 sky transmittance table (past the old 128x128 limit) was drawn once, skipped, and every scattering volume
     made from it came out zero (colour cast, overexposed sky) */
  int must_wait=0;
  { u32 ctl=r[0x487]; if((ctl&15)>=1){ u32* q=&r[0x200+((ctl>>4)&7)*0x10];
      if(q[0]|q[1]){ if(((q[5]>>16)&1) || (q[6]&0xFFFF)>1 || (q[2] && q[3] && (u64)q[2]*q[3]<=512u*512u)) must_wait=1; } } }
  if(must_wait && g_sep>0){ u64 t0=plat_ns(); while((!sh_sep_ready(vs) || (gs && !sh_sep_ready(gs)) || !sh_sep_ready(fs)) && plat_ns()-t0<3000000000ull) plat_sleep_ns(500000); st_waited++; }
  if(g_sep>0){ int rv, rg=1, rf; TM(TM_SHREADY,(rv=sh_sep_ready(vs), rg=gs?sh_sep_ready(gs):1, rf=sh_sep_ready(fs)));
    if(!rv || !rg || !rf){ st_compiling++; trace_note("DRAW SKIPPED: compiling\n"); return; }
    if(!vs->sp || (gs && !gs->sp) || !fs->sp){ st_noprog++; trace_note("DRAW SKIPPED: shader without program vs=%016llx gs=%016llx fs=%016llx\n",(unsigned long long)vs->hash,(unsigned long long)(gs?gs->hash:0),(unsigned long long)fs->hash); return; }
    vpp=vs->sp; vpl=vs->uloc; if(gs){ gsp=gs->sp; glp=gs->uloc; } atp=fs->sp; atl=fs->uloc; }
  else {
  p=program_get(vs,fs);
  if(must_wait){ u64 t0=plat_ns(); while(!program_ready(p) && plat_ns()-t0<3000000000ull) plat_sleep_ns(500000); st_waited++; }
  if(!program_ready(p)){ st_compiling++; trace_note("DRAW SKIPPED: compiling\n"); return; }
  vpp=atp=p->p; vpl=p->vp_loc; atl=p->at_loc;
  if(!p->p){ st_noprog++;
    { static int said; if(said<sw2_cap(5)){ said++; LOG("gpu: program not linked vs=%016llx fs=%016llx active stages:%s%s%s%s%s%s\n",(unsigned long long)vs->hash,(unsigned long long)fs->hash,
        (r[0x800]&1)?" VertexA":"",(r[0x810]&1)?" VertexB":"",(r[0x820]&1)?" TessCtrl":"",(r[0x830]&1)?" TessEval":"",(r[0x840]&1)?" Geometry":"",(r[0x850]&1)?" Fragment":""); } } trace_note("DRAW SKIPPED: link failed vs=%016llx fs=%016llx\n",(unsigned long long)vs->hash,(unsigned long long)fs->hash); return; } }
  g_draw_layered=vs->si.writes_layer || (gs && gs->si.writes_layer);
  TM(TM_STATE,bind_framebuffer(ch));
  { extern int g_opt_diag; if(g_opt_diag) rt_note_writer(ch,fs,code); }
  for(int i=0;i<g_nrt;i++){ Tex* t=g_rt[i]; if(t && (u64)t->w*t->h<=4096) t->smw=g_frame+1; }   /* rt_small_writeback */
  if(g_nrt && g_rt[0]){ g_last_cw=g_rt[0]->w; g_last_ch=g_rt[0]->h; }
  { extern int g_opt_shadows; /* shadows=0: skip depth-only passes at a different size (shadow maps) */
    if(!g_opt_shadows && !g_nrt && g_zt && (g_zt->w!=g_last_cw || g_zt->h!=g_last_ch)){ trace_note("DRAW SKIPPED: shadow\n");
      /* ...and clear the map to far once a frame so it reads as fully lit and not black. far is 0 with reversed Z
         (GREATER/GEQUAL test): sonic frontiers' shadow maps cleared to 1.0 put the whole world in shadow */
      { extern int g_frames; if(g_zt->skipclr!=g_frames+1){ g_zt->skipclr=g_frames+1;
        GLenum zf= r[0x4B3]? cmp_func(r[0x4C3]) : GL_LESS; float fz= (zf==GL_GREATER || zf==GL_GEQUAL)? 0.0f : 1.0f;
        static PFNGLCLEARTEXIMAGEPROC p_cz; if(!p_cz) p_cz=(PFNGLCLEARTEXIMAGEPROC)backend_gl_proc("glClearTexImage");
        if(p_cz){ if(g_zt->f.fmt==GL_DEPTH_STENCIL){ struct { float d; u32 s; } ds={fz,0}; p_cz(g_zt->id,0,GL_DEPTH_STENCIL,GL_FLOAT_32_UNSIGNED_INT_24_8_REV,&ds); }
          else p_cz(g_zt->id,0,GL_DEPTH_COMPONENT,GL_FLOAT,&fz); g_zt->wgen++; } } }
      return; } }
  if(!g_nrt && !g_zt){ st_nort++; trace_note("DRAW SKIPPED: no target\n"); return; }
  st_drawn++; wd_note(vs->hash,fs->hash);
  if(g_gp_state==2){ static int pd=-1; if(pd<0) pd=getenv("SWR_GP_DRAW")!=NULL;   /* GPU time per draw, by fragment shader */
    if(pd){ Tex* t=NULL; for(int i=0;i<g_nrt && !t;i++) t=g_rt[i]; if(!t) t=g_zt; gp_pass(t); if(g_gp_state==2 && g_gp_n) g_gp_p[g_gp_n-1].sh=fs->hash; } }
  if(g_draw_asset && g_nrt && g_rt[0] && !g_rt[0]->f.depth && g_rt[0]->w>=256 && g_rt[0]->h>=256 && (u64)g_rt[0]->w*g_rt[0]->h<1920ull*1080){ Tex* c=g_rt[0];
    if(!g_view_rt || g_frame-g_view_frame>300 || (u64)c->w*c->h>=(u64)g_view_rt->w*g_view_rt->h){ if(g_view_rt!=c){ static int said; if(said<6){ said++; LOG("gpu: test view candidate: render target %llx %ux%u\n",(unsigned long long)c->gva,c->w,c->h); } } g_view_rt=c; g_view_frame=g_frame; } }
  g_draw_asset=0;
  if(g_sep>0){ if(!g_pipe_on){ glUseProgram(0); p_bpp(g_pipe); g_pipe_on=1; }
    if(g_pipe_vs!=vpp){ p_ups(g_pipe,0x00000001 /* VERTEX_SHADER_BIT */,vpp); g_pipe_vs=vpp; }
    if(g_pipe_gs!=gsp){ p_ups(g_pipe,0x00000004 /* GEOMETRY_SHADER_BIT */,gsp); g_pipe_gs=gsp; }
    if(g_pipe_fs!=atp){ p_ups(g_pipe,0x00000002 /* FRAGMENT_SHADER_BIT */,atp); g_pipe_fs=atp; } }
  else glUseProgram(p->p);
  TM(TM_STATE,apply_state(ch));
  { float vt[2]; memcpy(vt,&r[0x280],8); float fl[4]={ (vt[1]<0)!=(origin_flip_h(r)!=0)? -1.f : 1.f, p_clip? 0.f : (float)(r[0x35F]&1), 0, 0 };
    if(vpl>=0){ if(g_sep>0){ if(!vs->lu_ok || memcmp(vs->lu,fl,16)){ memcpy(vs->lu,fl,16); vs->lu_ok=1; glProgramUniform4fv(vpp,vpl,1,fl); } } else glProgramUniform4fv(vpp,vpl,1,fl); }
    if(glp>=0 && gs && (!gs->lu_ok || memcmp(gs->lu,fl,16))){ memcpy(gs->lu,fl,16); gs->lu_ok=1; glProgramUniform4fv(gsp,glp,1,fl); } }
  if(atl>=0){ /* alpha test: 0x4BB enable, 0x4C4 ref, 0x4C5 func (GL 0x200.. or D3D 1..8) */
    u32 fn=r[0x4C5]; int f= fn>=0x200 && fn<=0x207? (int)(fn-0x200) : fn>=1 && fn<=8? (int)fn-1 : 7; float ref; memcpy(&ref,&r[0x4C4],4);
    float at[4]={ (r[0x4BB]&1)? 1.f : 0.f, (float)f, ref, 0 };
    memcpy(g_last_at,at,16);
    if(g_sep>0){ if(!fs->lu_ok || memcmp(fs->lu,at,16)){ memcpy(fs->lu,at,16); fs->lu_ok=1; glProgramUniform4fv(atp,atl,1,at); } } else glProgramUniform4fv(atp,atl,1,at); }
  TM(TM_CB,(upload_cbufs(0,vs), gs?upload_cbufs(3,gs):(void)0, upload_cbufs(4,fs)));
  g_ngw=0; TM(TM_GB,(bind_gbufs(0,vs), gs?bind_gbufs(3,gs):(void)0, bind_gbufs(4,fs)));
  int fs_imgs=0; TM(TM_TEX,(bind_textures(ch,0,vs), gs?bind_textures(ch,3,gs):(void)0, bind_textures(ch,4,fs), fs_imgs=bind_images(ch,fs)));
  /* vertex attribs */
  u64 tb0_=plat_ns();
  glBindVertexArray(g_vao);
  u32 used=vs->si.in_mask;
  /* attrib state cache for g_vao */
  static u32 va_en; static u32 va_key[16][8]; static int va_const[16];
  if(g_va_inval){ g_va_inval=0; memset(va_key,0,sizeof va_key); }
#define VA_OFF(i_) do{ if(va_en&(1u<<(i_))){ glDisableVertexAttribArray(i_); va_en&=~(1u<<(i_)); } }while(0)
  for(int i=0;i<16;i++){
    if(!(used&(1u<<i))){ VA_OFF(i); continue; }
    u32 a=r[0x458+i]; u32 buf=a&31; int constant=(a>>6)&1; u32 off=(a>>7)&0x3FFF; u32 size=(a>>21)&0x3F, ty=(a>>27)&7; int bgra=(a>>31)&1;
    u32* va=&r[0x700+buf*4];
    if(constant || !(va[0]&0x1000)){ VA_OFF(i); if(!va_const[i]){ glVertexAttrib4f(i,0,0,0,1); va_const[i]=1; } continue; }
    u64 start=((u64)va[1]<<32)|va[2]; u64 lim=((u64)r[0x7C0+buf*2]<<32)|r[0x7C1+buf*2]; u32 stride=va[0]&0xFFF;
    u64 len= lim>=start? lim-start+1 : 0; if(len>64u<<20) len=64u<<20;
    if(!len) { VA_OFF(i); continue; }
    GLuint vbid=buf_cached(GL_ARRAY_BUFFER,start,len);
    u32 cnt=attr_count(size), bits=attr_bits(size);
    GLenum gt; GLboolean norm= ty==1||ty==2;
    if(size==0x30) gt= ty==1? GL_INT_2_10_10_10_REV : GL_UNSIGNED_INT_2_10_10_10_REV;
    else if(size==0x31) gt=GL_UNSIGNED_INT_10F_11F_11F_REV;
    else if(ty==7) gt= bits==32? GL_FLOAT : GL_HALF_FLOAT;
    else if(ty==1||ty==3||ty==6) gt= bits==32? GL_INT : bits==16? GL_SHORT : GL_BYTE;
    else gt= bits==32? GL_UNSIGNED_INT : bits==16? GL_UNSIGNED_SHORT : GL_UNSIGNED_BYTE;
    GLint gcnt= (bgra && cnt==4)? GL_BGRA : (GLint)cnt; if(gcnt==GL_BGRA) norm=GL_TRUE;
    u32 div= r[0x620+buf]? (va[3]? va[3] : 1) : 0;
    u32 key[8]={vbid,(u32)gcnt,gt,(u32)norm|((ty==3||ty==4)? 2u : 0u),stride,off,div,cnt};
    if(!(va_en&(1u<<i))){ glEnableVertexAttribArray(i); va_en|=1u<<i; }
    va_const[i]=0;
    if(memcmp(va_key[i],key,sizeof key)){ memcpy(va_key[i],key,sizeof key);
      if(ty==3||ty==4) glVertexAttribIPointer(i,cnt,gt,stride,(void*)(size_t)off);
      else glVertexAttribPointer(i,gcnt,gt,norm,stride,(void*)(size_t)off);
      glVertexAttribDivisor(i,div); }
  }
  g_tm[TM_BUF]+=plat_ns()-tb0_; tb0_=plat_ns();
  /* layered draw (3D LUTs, cube faces) without gl_Layer in the VS: one pass per layer and the VS drops the
     vertices of the other layers */
  int nlay=1; { extern int g_vs_layer_ext; if(g_draw_layered && !g_vs_layer_ext && vpl>=0)
    for(int i=0;i<g_nrt;i++) if(g_rt[i] && g_rt[i]->vol && (int)g_rt[i]->vol->d>nlay) nlay=(int)g_rt[i]->vol->d; }
  for(int L_=0;L_<nlay;L_++){
  if(nlay>1){ for(int i=0;i<g_nrt;i++) if(g_rt[i] && g_rt[i]->vol && L_<(int)g_rt[i]->vol->d) glFramebufferTextureLayer(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0+i,g_rt[i]->vol->id,0,L_);
    float vt[2]; memcpy(vt,&r[0x280],8); float fl[4]={ (vt[1]<0)!=(origin_flip_h(r)!=0)? -1.f : 1.f, p_clip? 0.f : (float)(r[0x35F]&1), (float)L_, 0 };
    glProgramUniform4fv(vpp,vpl,1,fl); vs->lu_ok=0; g_fb_sig_ok=0;
    if(atl>=0){ float a4[4]; memcpy(a4,g_last_at,16); a4[3]=(float)L_; glProgramUniform4fv(atp,atl,1,a4); fs->lu_ok=0; } }
  GLenum mode=topo(topology);
  if(topology==7 || topology==8 || topology==9){ mode=GL_TRIANGLE_FAN; if(topology==7) mode=GL_TRIANGLES; if(topology==8) mode=GL_TRIANGLE_STRIP; }   /* quad strip = tri strip */
  if(indexed){
    u32 fmt=r[0x5F6]; u32 isz= fmt==0? 1 : fmt==1? 2 : 4; u64 ia=((u64)r[0x5F2]<<32)|r[0x5F3];
    u64 ilen=((u64)first+count)*isz; u8* src=NULL;   /* 64-bit: first+count can wrap */
    if(topology==7){ src=gmem_ptr(ia,ilen); if(!src){ if(g_stage_cap<ilen){ u8* nb=realloc(g_stage_buf,ilen); if(!nb) return; g_stage_buf=nb; g_stage_cap=ilen; } gmem_read(ia,g_stage_buf,ilen); src=g_stage_buf; } }
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,g_ibuf);
    GLenum it= isz==1? GL_UNSIGNED_BYTE : isz==2? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT;
    if(topology==7){ /* quads -> tris */
      u32 nq=count/4; u32* idx=malloc((size_t)nq*6*4); if(!idx) return;
      for(u32 k=0;k<nq;k++){ u32 v[4]; for(int j=0;j<4;j++){ u64 o=((u64)first+(u64)k*4+j)*isz; v[j]= isz==1? src[o] : isz==2? *(u16*)(src+o) : *(u32*)(src+o); }
        u32 t6[6]={v[0],v[1],v[2],v[0],v[2],v[3]}; memcpy(idx+k*6,t6,24); }
      glBufferData(GL_ELEMENT_ARRAY_BUFFER,(GLsizeiptr)nq*24,idx,GL_STREAM_DRAW); free(idx);
      glDrawElementsInstancedBaseVertexBaseInstance(GL_TRIANGLES,nq*6,GL_UNSIGNED_INT,0,instances,(GLint)base_vertex,base_instance);
    } else {
      buf_cached(GL_ELEMENT_ARRAY_BUFFER,ia+(u64)first*isz,(u64)count*isz);   /* just this draw's range */
      glDrawElementsInstancedBaseVertexBaseInstance(mode,count,it,(void*)0,instances,(GLint)base_vertex,base_instance);
    }
  } else {
    if(topology==7){ u32 nq=count/4; u32* idx=malloc((size_t)nq*24+4); for(u32 k=0;k<nq;k++){ u32 b=first+k*4; u32 t6[6]={b,b+1,b+2,b,b+2,b+3}; memcpy(idx+k*6,t6,24); }
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,g_ibuf); glBufferData(GL_ELEMENT_ARRAY_BUFFER,(GLsizeiptr)nq*24,idx,GL_STREAM_DRAW); free(idx);
      glDrawElementsInstancedBaseVertexBaseInstance(GL_TRIANGLES,nq*6,GL_UNSIGNED_INT,0,instances,0,base_instance); }
    else glDrawArraysInstancedBaseInstance(mode,first,count,instances,base_instance);
  }
  if(fs_imgs) glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT|GL_SHADER_IMAGE_ACCESS_BARRIER_BIT|GL_FRAMEBUFFER_BARRIER_BIT);   /* later draws read what SUST wrote */
  }
  gbuf_writeback();
  if(nlay>1){ static int said; if(!said){ said=1; LOG("gpu: layered draw without VS layer support: drawn layer by layer (%d layers)\n",nlay); } }
  g_tm[TM_DRAW]+=plat_ns()-tb0_;
  rt_hist_note(ch,vs,fs,stage_prog(ch,r,code,1),(r[0x800+5*0x10]&1)? stage_prog(ch,r,code,5) : 0,count);
  tr_draw(ch,vs,fs,stage_prog(ch,r,code,1),(r[0x800+5*0x10]&1)? stage_prog(ch,r,code,5) : 0,indexed,topology,count,instances);
}

void r3d_clear_gl(GpuChannel* ch, u32 v){ if(!swr_render_enabled(ch->r3d)) return; state_dirty(); g_tmap_gen++; if(g_cb_owner!=(int)ch->id && !getenv_cached_sharedcb()) cb_switch(ch);
  u32* r=ch->r3d; st_clears++;
  u32 rt=(v>>6)&15;
  g_clear_layer=(int)((v>>10)&0x7FF); bind_framebuffer(ch); g_clear_layer=-1;
  /* NVB197 SET_CLEAR_SURFACE_CONTROL: bit4 clear rect, bit8 scissor0, bit12 viewport clip0. each one is its own
     limit */
  u32 cxs=0xFFFF0000u, cys=0xFFFF0000u; int crect=0;
  if(r[0x43E]&0x10){ cxs=r[0x35B]; cys=r[0x35C]; crect=1; }
  if((r[0x43E]&0x100) && (r[0x380]&1)){
    u32 sx=r[0x381],sy=r[0x382];
    u32 x0=sx&0xffff,x1=sx>>16,y0=sy&0xffff,y1=sy>>16;
    if(x0<(cxs&0xffff)) x0=cxs&0xffff; if(x1>(cxs>>16)) x1=cxs>>16;
    if(y0<(cys&0xffff)) y0=cys&0xffff; if(y1>(cys>>16)) y1=cys>>16;
    if(x1<x0) x1=x0; if(y1<y0) y1=y0; cxs=x0|(x1<<16); cys=y0|(y1<<16); crect=1;
  }
  if(r[0x43E]&0x1000){ u32 vx=r[0x300],vy=r[0x301];
    u32 x0=vx&0xffff,x1=x0+(vx>>16),y0=vy&0xffff,y1=y0+(vy>>16);
    if(x0<(cxs&0xffff)) x0=cxs&0xffff; if(x1>(cxs>>16)) x1=cxs>>16;
    if(y0<(cys&0xffff)) y0=cys&0xffff; if(y1>(cys>>16)) y1=cys>>16;
    if(x1<x0) x1=x0; if(y1<y0) y1=y0; cxs=x0|(x1<<16); cys=y0|(y1<<16); crect=1;
  }
  if(crect){ u32 xs=cxs, ys=origin_scissor_y(cys,origin_flip_h(r)); glEnable(GL_SCISSOR_TEST); glScissorIndexed(0,(GLint)((xs&0xFFFF)*g_draw_scale),(GLint)((ys&0xFFFF)*g_draw_scale),(GLsizei)(((xs>>16)-(xs&0xFFFF))*g_draw_scale+0.5f),(GLsizei)(((ys>>16)-(ys&0xFFFF))*g_draw_scale+0.5f)); }
  else glDisable(GL_SCISSOR_TEST);
  if(v&0x3C){ Tex* t= rt<8? g_rt[rt] : NULL;
    if(!t && rt<8){ /* the clear targets an RT that isn't bound (kirby does this), so clear the texture directly */
      u32* q=&r[0x200+rt*0x10]; u64 a=((u64)q[0]<<32)|q[1];
      if(a && q[2] && q[3] && q[4]){ Tex* ct=rt_texture(a,q[2],q[3],q[4],q[5],0);
        static PFNGLCLEARTEXIMAGEPROC p_cti; if(!p_cti) p_cti=(PFNGLCLEARTEXIMAGEPROC)backend_gl_proc("glClearTexImage");
        if(ct && p_cti && !ct->f.depth){ float cc[4]; memcpy(cc,&r[0x360],16);
          if(ct->f.is_int){ GLuint ui[4]={(GLuint)cc[0],(GLuint)cc[1],(GLuint)cc[2],(GLuint)cc[3]}; p_cti(ct->id,0,ct->f.fmt,GL_UNSIGNED_INT,ui); }
          else p_cti(ct->id,0,GL_RGBA,GL_FLOAT,cc);
          ct->wgen++; static int said; if(!said){ said=1; LOG("gpu: clear of a RT outside the active list: cleared directly\n"); } } } }
    /* single float RT with color + depth (+stencil): one glClear so the driver can fast clear it all at once */
    if(t && rt==0 && g_nrt==1 && !t->f.is_int && (v&0x3C)==0x3C && (v&1) && g_zt && !getenv("SWR_NO_CLEARMERGE")){
      float cc[4], dz; memcpy(cc,&r[0x360],16); memcpy(&dz,&r[0x364],4);
      glColorMaski(0,1,1,1,1); glClearColor(cc[0],cc[1],cc[2],cc[3]); glDepthMask(GL_TRUE); glClearDepth((double)dz);
      GLbitfield m=GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT;
      if((v&2) && g_zt->f.fmt==GL_DEPTH_STENCIL){ glStencilMaskSeparate(GL_FRONT_AND_BACK,0xFF); glClearStencil((GLint)r[0x368]); m|=GL_STENCIL_BUFFER_BIT; }
      glClear(m); v&=~3u; t=NULL; }
    if(t){ glColorMaski(rt,(v>>2)&1,(v>>3)&1,(v>>4)&1,(v>>5)&1); float cc[4]; memcpy(cc,&r[0x360],16);
      if(t->f.is_int && ifmt_signed(t->f.ifmt)){ GLint si[4]={(GLint)cc[0],(GLint)cc[1],(GLint)cc[2],(GLint)cc[3]}; glClearBufferiv(GL_COLOR,rt,si); }
      else if(t->f.is_int){ GLuint ui[4]={(GLuint)cc[0],(GLuint)cc[1],(GLuint)cc[2],(GLuint)cc[3]}; glClearBufferuiv(GL_COLOR,rt,ui); }
      else glClearBufferfv(GL_COLOR,rt,cc); } }
  if(g_tr_on && g_tr_n<6000){ g_tr_n++; float cc[4]; memcpy(cc,&r[0x360],16); char nm[160]; Tex* t= rt<8? g_rt[rt] : NULL;
    { float dz; memcpy(&dz,&r[0x364],4); trace_note("#%d CLEAR rt%u %llx mask=%x color=(%.2f,%.2f,%.2f,%.2f) depth=%.9g z=%llx ch=%u layer=%u flags=%x rect=%s x=%x y=%x rtctl=%x\n",g_tr_n,rt,(unsigned long long)(t?t->gva:0),v&0x3F,cc[0],cc[1],cc[2],cc[3],dz,(unsigned long long)(g_zt?g_zt->gva:0),(unsigned)ch->id,(unsigned)((v>>10)&0x7FF),r[0x43E],crect? "on" : "off",cxs,cys,r[0x487]); }
    if(v&0x3C){ tr_thumb(t,nm); trace_note("  result: %s\n",nm); } }
  if((v&3) && g_zt){ float dz; memcpy(&dz,&r[0x364],4); glDepthMask(GL_TRUE); glStencilMaskSeparate(GL_FRONT_AND_BACK,0xFF);
    if((v&3)==3 && g_zt->f.fmt==GL_DEPTH_STENCIL) glClearBufferfi(GL_DEPTH_STENCIL,0,dz,(GLint)r[0x368]);
    else if(v&1) glClearBufferfv(GL_DEPTH,0,&dz); else { GLint s=(GLint)r[0x368]; glClearBufferiv(GL_STENCIL,0,&s); } }
}

/* ============================================================ present */
static GLuint g_blit_prog, g_cpu_tex; static int g_cpu_w, g_cpu_h;
static GLuint g_last_present_tex; static int g_last_flip, g_last_w, g_last_h; static int g_last_is_rt;
static u64 g_present_guest_ns; /* when the game presented it, before the GPU queue delay */
static const char* BLIT_VS="#version 430 core\nout vec2 uv; uniform vec4 xf; void main(){ vec2 p=vec2((gl_VertexID&1)*2.0-1.0,(gl_VertexID>>1)*2.0-1.0); uv=(p*0.5+0.5)*xf.zw+xf.xy; gl_Position=vec4(p,0.0,1.0); }";
static const char* BLIT_FS="#version 430 core\nin vec2 uv; out vec4 o; layout(binding=0) uniform sampler2D t; void main(){ o=vec4(texture(t,uv).rgb,1.0); }";
volatile i32 g_vi_crop[4]; volatile u32 g_vi_transform;   /* crop/transform from QueueBuffer */
#ifdef _WIN32
/* ============================================================ F7 recorder
   every presented frame gets converted to NV12 (BT.709) on the gpu at window size, same letterboxing as on screen.
   it's read back through a PBO two frames later so the GPU thread never waits, windows does the H.264 encode on
   its own thread */
extern volatile int g_rec_toggle; extern void backend_rec_state(int on);
extern int rec_open(const wchar_t* path, int w, int h, int fps); extern void rec_frame(const u8* nv12, i64 t100, i64 dur100); extern void rec_close(void); extern void rec_audio_start(u64);
extern int rec_failed(void);
#define REC_N 3
extern volatile int g_vt_on; extern int g_opt_rec_fps; extern u64 guest_ns(void); static u64 g_rec_n, g_rec_vt0, g_rec_lastts;
static int g_rec_on, g_rec_w, g_rec_h; static u64 g_rec_t0, g_rec_last; static u32 g_rec_k; static u64 g_rec_out, g_rec_base; static int g_rec_have;
static GLuint g_rec_fbo[2], g_rec_tex[2], g_rec_prog[2], g_rec_pbo[REC_N]; static GLsync g_rec_sync[REC_N]; static u64 g_rec_ts[REC_N];
static GLint g_rec_xf[2];
static PFNGLMAPBUFFERRANGEPROC p_rmap; static PFNGLUNMAPBUFFERPROC p_runmap; static PFNGLFENCESYNCPROC p_rfs; static PFNGLCLIENTWAITSYNCPROC p_rcws; static PFNGLDELETESYNCPROC p_rds;
static const char* REC_VS="#version 430 core\nout vec2 uv; uniform vec4 xf; void main(){ vec2 p=vec2((gl_VertexID&1)*2.0-1.0,(gl_VertexID>>1)*2.0-1.0); vec2 q=p*0.5+0.5; q.y=1.0-q.y; uv=q*xf.zw+xf.xy; gl_Position=vec4(p,0.0,1.0); }";
static const char* REC_FS_Y="#version 430 core\nin vec2 uv; out vec4 o; layout(binding=0) uniform sampler2D t; void main(){ vec3 c=clamp(texture(t,uv).rgb,0.0,1.0); o=vec4(16.0/255.0+dot(c,vec3(0.2126,0.7152,0.0722))*(219.0/255.0)); }";
static const char* REC_FS_UV="#version 430 core\nin vec2 uv; out vec4 o; layout(binding=0) uniform sampler2D t; void main(){ vec3 c=clamp(texture(t,uv).rgb,0.0,1.0);\n"
  " o=vec4(128.0/255.0+dot(c,vec3(-0.1146,-0.3854,0.5))*(224.0/255.0), 128.0/255.0+dot(c,vec3(0.5,-0.4542,-0.0458))*(224.0/255.0), 0.0, 0.0); }";

static void rec_gl_free(void){
  for(int i=0;i<REC_N;i++){ if(g_rec_sync[i]){ p_rds(g_rec_sync[i]); g_rec_sync[i]=0; } }
  glDeleteBuffers(REC_N,g_rec_pbo); glDeleteFramebuffers(2,g_rec_fbo); glDeleteTextures(2,g_rec_tex); memset(g_rec_pbo,0,sizeof g_rec_pbo); memset(g_rec_fbo,0,sizeof g_rec_fbo); memset(g_rec_tex,0,sizeof g_rec_tex); }

static void rec_start(void){
  if(!p_rmap){ p_rmap=(PFNGLMAPBUFFERRANGEPROC)backend_gl_proc("glMapBufferRange"); p_runmap=(PFNGLUNMAPBUFFERPROC)backend_gl_proc("glUnmapBuffer");
    p_rfs=(PFNGLFENCESYNCPROC)backend_gl_proc("glFenceSync"); p_rcws=(PFNGLCLIENTWAITSYNCPROC)backend_gl_proc("glClientWaitSync"); p_rds=(PFNGLDELETESYNCPROC)backend_gl_proc("glDeleteSync"); }
  if(!p_rmap || !p_runmap || !p_rfs || !p_rcws || !p_rds){ WARN("recorder: the video driver lacks the required features\n"); return; }
  if(!g_rec_prog[0]){ GLuint v=compile(GL_VERTEX_SHADER,REC_VS,NULL,0);
    g_rec_prog[0]=link2(v,compile(GL_FRAGMENT_SHADER,REC_FS_Y,NULL,0),0); g_rec_prog[1]=link2(compile(GL_VERTEX_SHADER,REC_VS,NULL,0),compile(GL_FRAGMENT_SHADER,REC_FS_UV,NULL,0),0);
    for(int k=0;k<2;k++) g_rec_xf[k]=g_rec_prog[k]?glGetUniformLocation(g_rec_prog[k],"xf"):-1; }
  int ww,wh; backend_window_size(&ww,&wh); g_rec_w=ww&~3; g_rec_h=wh&~3;
  if(g_rec_w<64 || g_rec_h<64 || !g_rec_prog[0] || !g_rec_prog[1]) return;
  wchar_t dir[MAX_PATH], path[MAX_PATH]; GetModuleFileNameW(NULL,dir,MAX_PATH); wchar_t* sl=wcsrchr(dir,L'\\'); wchar_t name[128]=L"game";
  if(sl){ wcsncpy(name,sl+1,127); name[127]=0; wchar_t* dot=wcsrchr(name,L'.'); if(dot) *dot=0; sl[1]=0; } else dir[0]=0;
  wcscat(dir,L"recordings"); CreateDirectoryW(dir,NULL);
  SYSTEMTIME st; GetLocalTime(&st);
  _snwprintf(path,MAX_PATH,L"%ls\\%ls_%04d-%02d-%02d_%02d-%02d-%02d.mp4",dir,name,st.wYear,st.wMonth,st.wDay,st.wHour,st.wMinute,st.wSecond);
  if(!rec_open(path,g_rec_w,g_rec_h,g_opt_rec_fps)) return;
  glGenTextures(2,g_rec_tex); glGenFramebuffers(2,g_rec_fbo);
  for(int i=0;i<2;i++){ glBindTexture(GL_TEXTURE_2D,g_rec_tex[i]); glTexStorage2D(GL_TEXTURE_2D,1,i? GL_RG8 : GL_R8,i? g_rec_w/2 : g_rec_w,i? g_rec_h/2 : g_rec_h);
    glBindFramebuffer(GL_FRAMEBUFFER,g_rec_fbo[i]); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,g_rec_tex[i],0); }
  glGenBuffers(REC_N,g_rec_pbo);
  for(int i=0;i<REC_N;i++){ glBindBuffer(GL_PIXEL_PACK_BUFFER,g_rec_pbo[i]); glBufferData(GL_PIXEL_PACK_BUFFER,(GLsizeiptr)g_rec_w*g_rec_h*3/2,NULL,GL_STREAM_READ); g_rec_sync[i]=0; }
  glBindBuffer(GL_PIXEL_PACK_BUFFER,0); glBindFramebuffer(GL_FRAMEBUFFER,0);
  g_rec_t0=plat_ns(); g_rec_last=0; g_rec_k=0; g_rec_on=1; g_rec_n=0; g_rec_out=0; g_rec_base=0; g_rec_have=0; backend_rec_state(1);
  { extern void vt_set(int); extern volatile int g_vt_on; extern volatile int g_rec_active; g_rec_active=1; vt_set(1); g_rec_vt0=guest_ns(); rec_audio_start(g_rec_vt0); g_rec_lastts=0; if(g_vt_on) LOG("recorder: smooth mode (the game runs in slow motion if the PC can't keep up; the video comes out at a steady %d FPS)\n",g_opt_rec_fps); } }

/* PBO from REC_N-1 frames ago: hand it over if it's ready, if not skip it (never wait) */
static void rec_drain(int all){
  for(u32 n=0;n<REC_N;n++){ u32 i=(g_rec_k+n)%REC_N; if(!g_rec_sync[i]) continue;
    u64 wait_start=plat_ns(); GLenum r;
    do { r=p_rcws(g_rec_sync[i],0,all?100000000ull:0); }
    while(all && r==GL_TIMEOUT_EXPIRED && plat_ns()-wait_start<30000000000ull);
    if(r!=GL_ALREADY_SIGNALED && r!=GL_CONDITION_SATISFIED){
      if(all || r!=GL_TIMEOUT_EXPIRED) WARN("recorder: GPU readback did not complete (%x); frame retained for retry\n",r);
      return;
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER,g_rec_pbo[i]); size_t n12=(size_t)g_rec_w*g_rec_h*3/2;
    const u8* p=(const u8*)p_rmap(GL_PIXEL_PACK_BUFFER,0,(GLsizeiptr)n12,GL_MAP_READ_BIT);
    if(p){ /* constant fps output: output frame k sits at k/fps. every captured frame covers its guest time up to
              the next one, so it gets repeated or dropped. weird clock jumps are clamped */
      u64 per=1000000000ull/(u64)(g_opt_rec_fps>0? g_opt_rec_fps : 60);
      u64 ts=g_rec_ts[i];
      /* audio and video both start at the recorder's guest clock start */
      u64 rel= ts>g_rec_base? ts-g_rec_base : 0;
      u64 want=rel/per+1;                                   /* output frames that should exist up to this one */
      /* keep the gaps so audio and video stay on the same timeline */   /* jump of more than 0.5s: put out one frame */
      if(!g_rec_have) want= want>g_rec_out? want : g_rec_out+1;
      u64 fps=(u64)(g_opt_rec_fps>0? g_opt_rec_fps : 60); while(g_rec_out<want){ rec_frame(p,(i64)(g_rec_out*10000000ull/fps),(i64)((g_rec_out+1)*10000000ull/fps-g_rec_out*10000000ull/fps)); g_rec_out++; }
      g_rec_have=1; p_runmap(GL_PIXEL_PACK_BUFFER); p_rds(g_rec_sync[i]); g_rec_sync[i]=0;
    } else { WARN("recorder: readback mapping failed; frame retained for retry\n"); glBindBuffer(GL_PIXEL_PACK_BUFFER,0); return; }
    glBindBuffer(GL_PIXEL_PACK_BUFFER,0); } }

static void rec_stop(void){ if(!g_rec_on) return; rec_drain(1); rec_close(); rec_gl_free(); g_rec_on=0; backend_rec_state(0); { extern void vt_set(int); extern volatile int g_rec_active; vt_set(0); g_rec_active=0; } }

volatile int g_rec_exit;   /* at exit: 1 = finalize, 2 = done */
void gpu_gl_rec_finish(void){ if(!g_rec_on) return; g_rec_exit=1; u64 t0=plat_ns(); while(g_rec_exit!=2 && plat_ns()-t0<30000000000ull) plat_sleep_ns(10000000); }

static void rec_capture(GLuint tex, const float* xf, int tw, int th){
  if(g_rec_on && rec_failed()){ rec_stop(); return; }
  if(g_rec_toggle){ g_rec_toggle=0; if(g_rec_on) rec_stop(); else rec_start(); }
  if(!g_rec_on) return;
  /* a frame queued before F7 was pressed belongs to the old timeline */
  if(g_present_guest_ns && g_present_guest_ns<g_rec_vt0) return;
  u64 now=plat_ns(); if(!g_vt_on && g_rec_last && now-g_rec_last<15500000ull) return;   /* max 60/s (smooth mode: every real frame) */
  g_rec_last=now;
  rec_drain(0);
  u32 i=g_rec_k%REC_N; if(g_rec_sync[i]){ if(!g_vt_on) return; rec_drain(1); if(g_rec_sync[i]) return; }   /* never write over a readback that's still pending */
  int dw=g_rec_w, dh=g_rec_w*th/(tw?tw:1); if(dh>g_rec_h){ dh=g_rec_h; dw=g_rec_h*tw/(th?th:1); } dw&=~1; dh&=~1;
  int ox=((g_rec_w-dw)/2)&~1, oy=((g_rec_h-dh)/2)&~1;
  glDisable(GL_SCISSOR_TEST); glDisable(GL_BLEND); glDisable(GL_FRAMEBUFFER_SRGB);
  glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,tex); glBindSampler(0,0);
  for(int k=0;k<2;k++){ int sx= k? 2 : 1;
    glBindFramebuffer(GL_FRAMEBUFFER,g_rec_fbo[k]); { GLenum b=GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&b); }
    glViewport(0,0,g_rec_w/sx,g_rec_h/sx);
    if(k) glClearColor(128.0f/255,128.0f/255,0,0); else glClearColor(16.0f/255,0,0,0); glClear(GL_COLOR_BUFFER_BIT);
    glViewport(ox/sx,oy/sx,dw/sx,dh/sx);
    glUseProgram(g_rec_prog[k]); glUniform4fv(g_rec_xf[k],1,xf);
    glBindVertexArray(g_scratch_vao); glDrawArrays(GL_TRIANGLE_STRIP,0,4); }
  glBindBuffer(GL_PIXEL_PACK_BUFFER,g_rec_pbo[i]); glPixelStorei(GL_PACK_ALIGNMENT,1);
  glBindFramebuffer(GL_READ_FRAMEBUFFER,g_rec_fbo[0]); glReadPixels(0,0,g_rec_w,g_rec_h,GL_RED,GL_UNSIGNED_BYTE,(void*)0);
  glBindFramebuffer(GL_READ_FRAMEBUFFER,g_rec_fbo[1]); glReadPixels(0,0,g_rec_w/2,g_rec_h/2,GL_RG,GL_UNSIGNED_BYTE,(void*)(size_t)((size_t)g_rec_w*g_rec_h));
  glBindBuffer(GL_PIXEL_PACK_BUFFER,0); glPixelStorei(GL_PACK_ALIGNMENT,4);
  g_rec_sync[i]=p_rfs(GL_SYNC_GPU_COMMANDS_COMPLETE,0); { u64 source=g_present_guest_ns?g_present_guest_ns:guest_ns(); u64 ts=source>g_rec_vt0?source-g_rec_vt0:0; if(g_rec_n && ts<g_rec_lastts) ts=g_rec_lastts; g_rec_lastts=ts; g_rec_ts[i]=ts; } g_rec_n++; g_rec_k++;
  glBindFramebuffer(GL_FRAMEBUFFER,0); glUseProgram(g_blit_prog); }
#else
static void rec_capture(GLuint tex, const float* xf, int tw, int th){ (void)tex; (void)xf; (void)tw; (void)th; }

void gpu_gl_rec_finish(void){}
#endif
/* ============================================================ F8 full res screenshot (png)
   the gpu copies the frame into a PBO, we map it a frame or two later and a worker thread writes the png */
volatile int g_shot8_req;
static GLuint g_s8_fbo, g_s8_tex, g_s8_pbo; static GLsync g_s8_sync; static int g_s8_w, g_s8_h, g_s8_tw, g_s8_th;
static PFNGLMAPBUFFERRANGEPROC p_s8map; static PFNGLUNMAPBUFFERPROC p_s8unmap; static PFNGLFENCESYNCPROC p_s8fs; static PFNGLCLIENTWAITSYNCPROC p_s8cws; static PFNGLDELETESYNCPROC p_s8ds;
typedef struct { u8* px; int w, h; char path[320]; } S8Job;

static u32 s8_crc_tab[256];

static u32 s8_crc(u32 c, const u8* d, size_t n){ if(!s8_crc_tab[1]) for(u32 i=0;i<256;i++){ u32 k=i; for(int j=0;j<8;j++) k= (k&1)? 0xEDB88320u^(k>>1) : k>>1; s8_crc_tab[i]=k; }
  c=~c; for(size_t i=0;i<n;i++) c=s8_crc_tab[(c^d[i])&255]^(c>>8); return ~c; }

static void s8_be32(u8* p, u32 v){ p[0]=(u8)(v>>24); p[1]=(u8)(v>>16); p[2]=(u8)(v>>8); p[3]=(u8)v; }

static void s8_chunk(FILE* f, const char* t, const u8* d, u32 n){ u8 h[8]; s8_be32(h,n); memcpy(h+4,t,4); fwrite(h,1,8,f); if(n) fwrite(d,1,n,f);
  u32 c=s8_crc(0,(const u8*)t,4); c=s8_crc(c,d,n); u8 cc[4]; s8_be32(cc,c); fwrite(cc,1,4,f); }

/* RGB png, Sub filter per row, stored (uncompressed) deflate. fast, ~3 bytes/px */
static void s8_thread(void* arg){
  S8Job* j=(S8Job*)arg; int w=j->w, h=j->h; size_t row=(size_t)w*3+1, raw=row*h;
  u8* r=malloc(raw); if(!r){ free(j->px); free(j); return; }
  for(int y=0;y<h;y++){ const u8* s=j->px+(size_t)(h-1-y)*w*4; u8* o=r+(size_t)y*row; o[0]=1; u8 pr=0,pg=0,pb=0;
    for(int x=0;x<w;x++){ u8 R=s[x*4],G=s[x*4+1],B=s[x*4+2]; o[1+x*3]=(u8)(R-pr); o[2+x*3]=(u8)(G-pg); o[3+x*3]=(u8)(B-pb); pr=R; pg=G; pb=B; } }
  free(j->px);
  size_t nb=(raw+65534)/65535; size_t zn=2+raw+nb*5+4; u8* z=malloc(zn);
  if(z){ size_t o=0; z[o++]=0x78; z[o++]=0x01; u32 a=1,b=0;
    for(size_t i=0;i<raw;i+=65535){ size_t n=raw-i; if(n>65535) n=65535; z[o++]= (i+n>=raw)? 1 : 0; z[o++]=(u8)n; z[o++]=(u8)(n>>8); z[o++]=(u8)~n; z[o++]=(u8)(~n>>8);
      memcpy(z+o,r+i,n); o+=n; for(size_t k=0;k<n;k++){ a=(a+r[i+k])%65521u; b=(b+a)%65521u; } }
    s8_be32(z+o,(b<<16)|a); o+=4;
    FILE* f=fopen(j->path,"wb");
    if(f){ static const u8 sig[8]={0x89,'P','N','G',13,10,26,10}; fwrite(sig,1,8,f);
      u8 ih[13]; s8_be32(ih,(u32)w); s8_be32(ih+4,(u32)h); ih[8]=8; ih[9]=2; ih[10]=0; ih[11]=0; ih[12]=0; s8_chunk(f,"IHDR",ih,13);
      s8_chunk(f,"IDAT",z,(u32)o); s8_chunk(f,"IEND",NULL,0); fclose(f); LOG("F8 screenshot: %s (%dx%d)\n",j->path,w,h); }
    else WARN("F8 screenshot: could not write %s\n",j->path);
    free(z); }
  free(r); free(j); }

static void shot8_poll(void){
  if(!g_s8_sync) return;
  GLenum st=p_s8cws(g_s8_sync,0,0); if(st!=GL_ALREADY_SIGNALED && st!=GL_CONDITION_SATISFIED) return;
  p_s8ds(g_s8_sync); g_s8_sync=0;
  size_t n=(size_t)g_s8_w*g_s8_h*4; glBindBuffer(GL_PIXEL_PACK_BUFFER,g_s8_pbo);
  const u8* p=(const u8*)p_s8map(GL_PIXEL_PACK_BUFFER,0,(GLsizeiptr)n,GL_MAP_READ_BIT);
  S8Job* j=calloc(1,sizeof *j);
  if(p && j && (j->px=malloc(n))){ memcpy(j->px,p,n); j->w=g_s8_w; j->h=g_s8_h;
    time_t tt=time(NULL); struct tm* tm=localtime(&tt); static int seq;
    plat_mkdir("captures"); snprintf(j->path,sizeof j->path,"captures/F8_%04d%02d%02d_%02d%02d%02d_%d.png",tm->tm_year+1900,tm->tm_mon+1,tm->tm_mday,tm->tm_hour,tm->tm_min,tm->tm_sec,++seq);
    if(plat_thread(s8_thread,j,4u<<20)){ s8_thread(j); } j=NULL; }
  if(j){ free(j->px); free(j); }
  if(p) p_s8unmap(GL_PIXEL_PACK_BUFFER);
  glBindBuffer(GL_PIXEL_PACK_BUFFER,0); }

static void shot8_capture(GLuint tex, const float* xf, int tw, int th){
  shot8_poll();
  { static int tf; if(!tf){ tf= getenv("SWR_SHOT8_TEST")? 1 : -1; } if(tf>0 && ++tf==40) g_shot8_req=1; }   /* automated test hook */
  { static int ev=-1, n, shots; if(ev<0){ const char* e=getenv("SWR_SHOT8_EVERY"); ev= e? atoi(e) : 0; } if(ev>0 && shots<40 && ++n%ev==0){ g_shot8_req=1; shots++; } }   /* every N presents (tests) */
  if(!g_shot8_req || g_s8_sync) return;
  g_shot8_req=0;
  if(!p_s8map){ p_s8map=(PFNGLMAPBUFFERRANGEPROC)backend_gl_proc("glMapBufferRange"); p_s8unmap=(PFNGLUNMAPBUFFERPROC)backend_gl_proc("glUnmapBuffer");
    p_s8fs=(PFNGLFENCESYNCPROC)backend_gl_proc("glFenceSync"); p_s8cws=(PFNGLCLIENTWAITSYNCPROC)backend_gl_proc("glClientWaitSync"); p_s8ds=(PFNGLDELETESYNCPROC)backend_gl_proc("glDeleteSync"); }
  if(!p_s8map || !p_s8unmap || !p_s8fs || !p_s8cws || !p_s8ds){ WARN("F8 screenshot: the video driver lacks the required features\n"); return; }
  /* actual size of the shown content (visible part of the texture) */
  int w=(int)(tw*fabsf(xf[2])+0.5f), h=(int)(th*fabsf(xf[3])+0.5f); if(w<16||h<16||w>8192||h>8192) return;
  if(w!=g_s8_w || h!=g_s8_h){
    if(g_s8_tex){ glDeleteTextures(1,&g_s8_tex); glDeleteFramebuffers(1,&g_s8_fbo); glDeleteBuffers(1,&g_s8_pbo); }
    glGenTextures(1,&g_s8_tex); glBindTexture(GL_TEXTURE_2D,g_s8_tex); glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA8,w,h);
    glGenFramebuffers(1,&g_s8_fbo); glBindFramebuffer(GL_FRAMEBUFFER,g_s8_fbo); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,g_s8_tex,0);
    glGenBuffers(1,&g_s8_pbo); glBindBuffer(GL_PIXEL_PACK_BUFFER,g_s8_pbo); glBufferData(GL_PIXEL_PACK_BUFFER,(GLsizeiptr)w*h*4,NULL,GL_STREAM_READ); glBindBuffer(GL_PIXEL_PACK_BUFFER,0);
    g_s8_w=w; g_s8_h=h; }
  (void)g_s8_tw; (void)g_s8_th;
  glBindFramebuffer(GL_FRAMEBUFFER,g_s8_fbo); { GLenum b=GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&b); }
  glDisable(GL_SCISSOR_TEST); glDisable(GL_BLEND); glDisable(GL_FRAMEBUFFER_SRGB); glColorMaski(0,1,1,1,1);
  glViewport(0,0,w,h); glUseProgram(g_blit_prog); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,tex); glBindSampler(0,0);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
  { static GLint xl=-2; static GLuint xp; if(xl==-2 || xp!=g_blit_prog){ xl=glGetUniformLocation(g_blit_prog,"xf"); xp=g_blit_prog; } glUniform4fv(xl,1,xf); }
  glBindVertexArray(g_scratch_vao); glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  glBindBuffer(GL_PIXEL_PACK_BUFFER,g_s8_pbo); glPixelStorei(GL_PACK_ALIGNMENT,4);
  glBindFramebuffer(GL_READ_FRAMEBUFFER,g_s8_fbo); glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,(void*)0);
  glBindBuffer(GL_PIXEL_PACK_BUFFER,0); glBindFramebuffer(GL_FRAMEBUFFER,0);
  g_s8_sync=p_s8fs(GL_SYNC_GPU_COMMANDS_COMPLETE,0); glFlush(); }

/* blit a texture to the window, 16:9 letterboxed. xf = uv transform (crop/flip) */
static void screen_quad(GLuint tex, const float* xf, int tw, int th){
  int ww,wh; backend_window_size(&ww,&wh);
  glBindFramebuffer(GL_FRAMEBUFFER,0); glDisable(GL_SCISSOR_TEST); glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST); glDisable(GL_CULL_FACE); glDisable(GL_STENCIL_TEST);
  glDisable(GL_FRAMEBUFFER_SRGB);
  glBindTexture(GL_TEXTURE_2D,tex); glTexParameteri(GL_TEXTURE_2D,0x8A48,0x8A4A);   /* TEXTURE_SRGB_DECODE = SKIP_DECODE */
  glColorMaski(0,1,1,1,1);
  glViewport(0,0,ww,wh); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
  int dw=ww, dh=ww*th/(tw?tw:1); if(dh>wh){ dh=wh; dw=wh*tw/(th?th:1); }
  glViewport((ww-dw)/2,(wh-dh)/2,dw,dh);
  glUseProgram(g_blit_prog); g_pipe_on=0; glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,tex); glBindSampler(0,0);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  { static const GLint idm[4]={GL_RED,GL_GREEN,GL_BLUE,GL_ALPHA}; glTexParameteriv(GL_TEXTURE_2D,GL_TEXTURE_SWIZZLE_RGBA,idm); }
  { static GLint xl=-2; static GLuint xp; if(xl==-2 || xp!=g_blit_prog){ xl=glGetUniformLocation(g_blit_prog,"xf"); xp=g_blit_prog; } glUniform4fv(xl,1,xf); }
  glBindVertexArray(g_scratch_vao); glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  rec_capture(tex,xf,tw,th); shot8_capture(tex,xf,tw,th); glUseProgram(g_blit_prog); glBindFramebuffer(GL_FRAMEBUFFER,0); glBindTexture(GL_TEXTURE_2D,tex);
  glTexParameteri(GL_TEXTURE_2D,0x8A48,0x8A49); }   /* restore DECODE for when the game samples it */

static void screen_swap(void){ TM(TM_SWAP,backend_gl_swap()); state_dirty(); g_pipe_on=0; mb_reset(); }

/* ============================================================ LSFG frame gen (user's own Lossless.dll) */
extern int lsfg_init(void); extern int lsfg_ready(void); extern int lsfg_context(u32,u32); extern GLuint lsfg_in_tex(void); extern void lsfg_reset(void); extern int lsfg_process(int, GLuint*);
extern int g_lsfg_mult, g_lsfg_height; extern float g_lsfg_target;
static GLuint g_lf_fbo; static u64 g_lf_last, g_lf_int=33333333ull; static int g_lf_on, g_lf_W, g_lf_H;
typedef struct { GLuint tex; u64 t; } LfQ; static LfQ g_lfq[6]; static int g_lfn; static u64 st_lsfg_gen, st_lsfg_ns;

static const float XF_ID[4]={0,0,1,1};

static void lf_flush(void){ for(int i=0;i<g_lfn;i++){ screen_quad(g_lfq[i].tex,XF_ID,g_lf_W,g_lf_H); screen_swap(); } g_lfn=0; }

static void lsfg_pump(void){ if(!g_lfn) return; u64 now=plat_ns(); int k=0;
  while(k<g_lfn && now>=g_lfq[k].t){ screen_quad(g_lfq[k].tex,XF_ID,g_lf_W,g_lf_H); screen_swap(); k++; }
  if(k){ memmove(g_lfq,g_lfq+k,(size_t)(g_lfn-k)*sizeof *g_lfq); g_lfn-=k; } }

/* real frame: copy it into the LSFG input, generate the in-between frames and show them spread over the interval,
   the real one last. 0 = not active (already at target, or no dll) */
static int g_blit_rt;   /* shown image is an RT, i.e. scaled by the res scale */
static int lsfg_frame(GLuint tex, const float* xf, int tw, int th){
  if(!lsfg_ready() || tw<=0 || th<=0) return 0;
#ifdef _WIN32
  { extern volatile int g_vt_on; if(g_vt_on){ lf_flush(); g_lf_on=0; return 0; } }   /* smooth recording: only real frames */
#endif
  u64 now=plat_ns();
  if(g_lf_last){ u64 d=now-g_lf_last; if(d>250000000ull) d=250000000ull; g_lf_int=(g_lf_int*7+d)/8; }
  g_lf_last=now;
  double base=1e9/(double)g_lf_int;
  int m=(int)(g_lsfg_target/base+0.5); if(m>g_lsfg_mult) m=g_lsfg_mult;
  if(base>g_lsfg_target*(g_lf_on? 0.92 : 0.85)) m=1;     /* already at target */
  { static int ls=-1; int st= m>=2? m : 0; if(st!=ls){ ls=st;
#ifdef _WIN32
      extern void backend_lsfg_state(int,int); backend_lsfg_state(st!=0,st);
#endif
      LOG("lsfg: %s\n",st? "ON" : "standby (the game already reaches the target)"); } }
  if(m<2){ lf_flush(); g_lf_on=0; return 0; }
  /* at a low res scale the real image is smaller than the window, and generating bigger just burns gpu on
     stretched pixels (x0.25 at 720p cost ~29 ms on an iGPU). min 360 */
  int content= g_blit_rt? (int)(th*g_scale+0.5f) : th; if(content<360) content=360; if(content>th) content=th;
  u32 H=(u32)(content<g_lsfg_height? content : g_lsfg_height), W=(u32)((u64)H*(u32)tw/(u32)th)&~1u; if(W<16 || H<16) return 0;
  if(!lsfg_context(W,H)){ return 0; }
  if(!g_lf_on){ g_lf_on=1; lsfg_reset(); }
  lf_flush();
  g_lf_W=(int)W; g_lf_H=(int)H;
  GLuint in=lsfg_in_tex();
  if(!g_lf_fbo) glGenFramebuffers(1,&g_lf_fbo);
  glBindFramebuffer(GL_FRAMEBUFFER,g_lf_fbo); glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,in,0); { GLenum b=GL_COLOR_ATTACHMENT0; glDrawBuffers(1,&b); }
  glDisable(GL_SCISSOR_TEST); glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST); glDisable(GL_CULL_FACE); glDisable(GL_STENCIL_TEST); glDisable(GL_FRAMEBUFFER_SRGB); glColorMaski(0,1,1,1,1);
  glViewport(0,0,(GLsizei)W,(GLsizei)H);
  glUseProgram(g_blit_prog); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,tex); glBindSampler(0,0);
  glTexParameteri(GL_TEXTURE_2D,0x8A48,0x8A4A); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  { static const GLint idm[4]={GL_RED,GL_GREEN,GL_BLUE,GL_ALPHA}; glTexParameteriv(GL_TEXTURE_2D,GL_TEXTURE_SWIZZLE_RGBA,idm); }
  { static GLint xl=-2; static GLuint xp; if(xl==-2 || xp!=g_blit_prog){ xl=glGetUniformLocation(g_blit_prog,"xf"); xp=g_blit_prog; } glUniform4fv(xl,1,xf); }
  glBindVertexArray(g_scratch_vao); glDrawArrays(GL_TRIANGLE_STRIP,0,4);
  glTexParameteri(GL_TEXTURE_2D,0x8A48,0x8A49);
  GLuint outs[4]; u64 t0=plat_ns(); int n=lsfg_process(m-1,outs); st_lsfg_ns+=plat_ns()-t0;
  { static int nl; if(nl<4 && n>0){ nl++; glFinish(); LOG("lsfg: frame generated in %.1f ms (GPU included, x%d, base %.1f FPS)\n",(plat_ns()-t0)/1e6,m,base); } }
  { static int shot=-1, cnt; if(shot<0){ const char* e=getenv("SWR_LSFG_SHOT"); shot= e? atoi(e) : 0; }
    if(shot>0 && n>0 && ++cnt==shot){ extern void lsfg_debug_tex(GLuint*,GLuint*,u32*,u32*); GLuint a,b; u32 w,h; lsfg_debug_tex(&a,&b,&w,&h);
      GLuint tx[3]={a,outs[0],b}; u8* px=malloc((size_t)w*h*4*3); u8* row=malloc((size_t)w*3*3);
      for(int k=0;k<3;k++){ glBindTexture(GL_TEXTURE_2D,tx[k]); glPixelStorei(GL_PACK_ALIGNMENT,1); glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_UNSIGNED_BYTE,px+(size_t)k*w*h*4); }
      FILE* f=fopen("lsfg_shot.ppm","wb"); if(f){ fprintf(f,"P6\n%u %u\n255\n",w*3,h);
        for(int y=(int)h-1;y>=0;y--){ for(int k=0;k<3;k++) for(u32 x=0;x<w;x++){ u8* q=px+(size_t)k*w*h*4+((size_t)y*w+x)*4; memcpy(row+((size_t)k*w+x)*3,q,3); } fwrite(row,1,(size_t)w*9,f); }
        fclose(f); LOG("lsfg: previous | generated | current saved to lsfg_shot.ppm\n"); }
      free(px); free(row); mb_reset(); } }
  /* the LSFG dispatches mess up programs, textures, samplers and UBO 0 */
  g_pipe_on=0; state_dirty(); mb_reset(); memset(g_ub_bound,0,sizeof g_ub_bound); glBindFramebuffer(GL_FRAMEBUFFER,g_fbo);
  g_last_present_tex=in; g_last_flip=0; g_last_w=(int)W; g_last_h=(int)H; g_last_is_rt=0;
  if(n<=0){ screen_quad(in,XF_ID,(int)W,(int)H); screen_swap(); return 1; }
  u64 step=g_lf_int/(u64)(n+1);
  screen_quad(outs[0],XF_ID,(int)W,(int)H); screen_swap(); st_lsfg_gen+=(u64)n;
  for(int k=1;k<n;k++){ g_lfq[g_lfn].tex=outs[k]; g_lfq[g_lfn].t=now+step*(u64)k; g_lfn++; }
  g_lfq[g_lfn].tex=in; g_lfq[g_lfn].t=now+step*(u64)n; g_lfn++;
  return 1; }

static void blit_to_screen(GLuint tex, int flip, int tw, int th){ state_dirty();
  /* game-requested crop (e.g. 1280x720 inside a 1920x1080 buffer) */
  float u0=0, v0=0, u1=1, v1=1; { i32 l=g_vi_crop[0], t=g_vi_crop[1], r=g_vi_crop[2], b=g_vi_crop[3];
    if(r>l && b>t && tw>0 && th>0 && r<=tw*4 && b<=th*4){ u0=(float)l/tw; u1=(float)r/tw; v0=(float)t/th; v1=(float)b/th; tw=r-l; th=b-t; } }
  if(g_vi_transform&2){ float x=v0; v0=v1; v1=x; }   /* FLIP_V */
  if(g_vi_transform&1){ float x=u0; u0=u1; u1=x; }   /* FLIP_H */
  float xf[4]; if(flip){ xf[0]=u0; xf[1]=v1; xf[2]=u1-u0; xf[3]=v0-v1; } else { xf[0]=u0; xf[1]=1-v1; xf[2]=u1-u0; xf[3]=v1-v0; }
  if(lsfg_frame(tex,xf,tw,th)) return;
  screen_quad(tex,xf,tw,th); screen_swap();
  g_last_present_tex=tex; g_last_flip=flip; g_last_w=tw; g_last_h=th;
}

static void present_cpu(u32* px, int w, int h){
  if(!g_cpu_tex || g_cpu_w!=w || g_cpu_h!=h){ if(g_cpu_tex){ glDeleteTextures(1,&g_cpu_tex); g_fb_sig_ok=0; } glGenTextures(1,&g_cpu_tex); glBindTexture(GL_TEXTURE_2D,g_cpu_tex); glTexStorage2D(GL_TEXTURE_2D,1,GL_RGBA8,w,h); g_cpu_w=w; g_cpu_h=h; }
  glBindTexture(GL_TEXTURE_2D,g_cpu_tex); glPixelStorei(GL_UNPACK_ROW_LENGTH,0); glTexSubImage2D(GL_TEXTURE_2D,0,0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,px);
  blit_to_screen(g_cpu_tex,1,w,h); g_last_is_rt=0; }

/* eviction: free memory backed textures unused for ~40 s and orphaned RTs, if not switching worlds slowly eats the
   iGPU's shared ram. walks part of the table every frame, textures get recreated from memory if needed */
static u64 st_tex_evict, st_vb_evict;

static void tex_free(Tex* t){
  if(t==g_view_rt) g_view_rt=NULL;
  glDeleteTextures(1,&t->id); if(t->view) glDeleteTextures(1,&t->view); if(t->zc){ glDeleteTextures(1,&t->zc); t->zc=0; } if(t->fbc){ glDeleteTextures(1,&t->fbc); t->fbc=0; }
  if(t->crop){ glDeleteTextures(1,&t->crop->id); if(t->crop->view) glDeleteTextures(1,&t->crop->view); free(t->crop); }
  if(t->vol){ glDeleteTextures(1,&t->vol->id); if(t->vol->view) glDeleteTextures(1,&t->vol->view); free(t->vol); }
  if(t->lay){ glDeleteTextures(1,&t->lay->id); if(t->lay->view) glDeleteTextures(1,&t->lay->view); free(t->lay->seen); free(t->lay); }
  free(t); }

/* host memory a texture takes (BC3 from ASTC: 1 byte a texel) */
static u64 tex_host_bytes(const Tex* t){ u64 n=(u64)t->w*t->h*(t->d? t->d : 1); u32 bpt= t->f.tobc? 1 : (t->f.bpp && !t->f.compressed)? t->f.bpp : 4;
  return n*bpt*(t->levels>1? 4 : 3)/3; }
static void gpu_evict_step(void){
  static u32 cur, vcur; if(g_tr_on) return; int any=0;
  /* textures age in seconds, not frames: at 1-2 FPS 2400 frames was 20 minutes and nothing ever left (9 GB of
     textures in MKW). Over the budget (SWR_TEX_BUDGET_MB, 2048 by default) whatever went unused for 5 s
     goes first. Textures the GPU wrote (compute images) stay, their data isn't in memory */
  static u64 fns[1024]; u64 now=plat_ns(); fns[g_frame&1023]=now;
  static u64 tbudget, total; static u32 tot_frame;
  if(!tbudget){ const char* b=getenv("SWR_TEX_BUDGET_MB"); tbudget=(u64)(b? atoi(b) : 2048)<<20; if(tbudget<(256u<<20)) tbudget=256u<<20; }
  if(g_frame-tot_frame>=32 || !tot_frame){ tot_frame=g_frame? g_frame : 1; total=0; for(int k=0;k<TEXH;k++) for(Tex* t=texh[k];t;t=t->next) if(!t->is_rt) total+=tex_host_bytes(t); }
  int tover= total>tbudget;
  for(int n=0;n<(tover? TEXH/8 : TEXH/64);n++,cur=(cur+1)&(TEXH-1)){
    for(Tex** pp=&texh[cur];*pp;){ Tex* t=*pp; u32 age=g_frame-t->last_used; int busy=(t==g_zt); for(int i=0;i<8;i++) if(g_rt[i]==t) busy=1;
      u64 age_ns= age<1024? now-fns[t->last_used&1023] : ~0ull;
      int old= t->is_rt? (t->gva==~0ull && age>300) : (!t->gpu_written && (age>2400 || age_ns>40000000000ull || (tover && age>2 && age_ns>5000000000ull)));
      if(old && !busy && !t->is_rt){ u64 b=tex_host_bytes(t); total= total>b? total-b : 0; }
      if(old && !busy){ *pp=t->next; tex_free(t); st_tex_evict++; any=1; } else pp=&t->next; } }
  if(any){ mb_reset(); g_fb_sig_ok=0; memset(g_tdc,0,sizeof g_tdc); g_tmap_gen++; }
  /* buffer cache, counted in frames: a range that never gets used again goes after 8 frames, anything unused for
     600, and over the budget anything unused for 30. xenoblade 2 submits once a frame and streams ~16 MB of new
     vertex ranges every frame, counting submits it got to 16 GB of ram in 3 minutes */
  static u32 ep_hist[1024]; ep_hist[g_frame&1023]=g_submit_epoch;
  static u64 budget; if(!budget){ const char* b=getenv("SWR_VB_BUDGET_MB"); budget=(u64)(b? atoi(b) : 1536)<<20; if(budget<(64u<<20)) budget=64u<<20; }
  int over= g_vbc_bytes>budget;
  u32 thr1= g_frame>8? ep_hist[(g_frame-8)&1023] : 0, thr2= g_frame>600? ep_hist[(g_frame-600)&1023] : 0, thr3= g_frame>30? ep_hist[(g_frame-30)&1023] : 0;
  int vany=0, nb= over? 2048 : 512;
  for(int n=0;n<nb;n++,vcur=(vcur+1)&2047){
    for(VbC** pp=&vbc[vcur];*pp;){ VbC* e=*pp;
      int stale= g_submit_epoch-e->epoch>4800 || (thr2 && (i32)(e->epoch-thr2)<0) || (e->used<=1 && thr1 && (i32)(e->epoch-thr1)<0) || (over && thr3 && (i32)(e->epoch-thr3)<0);
      if(stale && e->epoch!=g_submit_epoch){ *pp=e->next; glDeleteBuffers(1,&e->id); g_vbc_bytes-=e->len; g_vbc_n--; free(e); st_vb_evict++; vany=1; } else pp=&e->next; } }
  if(over){ static int said; if(said<sw2_cap(3)){ said++; LOG("gpu: buffer cache over %llu MB, dropping what wasn't used in the last 30 frames\n",(unsigned long long)(budget>>20)); } }
  if(vany) g_va_inval=1; }

/* small render targets a frame drew go back to guest memory (asynchronously, a frame or two late, like a fence on
   hardware): games read them with the CPU and no GPU copy tells us. MKW reads a 64x32 R32F luminance
   target. SWR_SMALL_RT_WB=1 turns it on (it did not change MKW's exposure, kept for games that need it) */
static void rt_small_writeback(void){
  static int on=-1; if(on<0) on=getenv("SWR_SMALL_RT_WB")!=NULL; if(!on) return;   /* opt-in until a game needs it */
  for(int k=0;k<TEXH;k++) for(Tex* t=texh[k];t;t=t->next) if(t->is_rt && t->smw==(u32)g_frame+1 && t->gva!=~0ull && rt_flushable(t)) rb_async(t,t->gva); }

static int present_addr(u64 cpu_addr, int w, int h){
  rt_small_writeback();
  /* find the RT whose gva maps this memory */
  Tex* best=NULL;
  for(int k=0;k<TEXH;k++) for(Tex* t=texh[k];t;t=t->next) if(t->is_rt && gmmu_translate(t->gva)==cpu_addr){ if(!best || (t->w==(u32)w && t->h==(u32)h) || t->last_used>best->last_used) best=t; }
  trace_note("PRESENT %llx %dx%d -> %s %llx\n",(unsigned long long)cpu_addr,w,h,best?"rt":"(none, CPU)",(unsigned long long)(best?best->gva:0));
  if(g_opt_show_rt<0) g_opt_show_rt= getenv("SWR_SHOW_RT")? atoi(getenv("SWR_SHOW_RT")) : 0;
  if(best && g_opt_show_rt>0 && g_view_rt && g_view_rt!=best && g_frame-g_view_frame<600){ Tex* v=g_view_rt; v->last_used=g_frame; g_blit_rt=1; blit_to_screen(v->id,1,(int)v->w,(int)v->h); g_blit_rt=0; g_last_is_rt=1; st_pres_rt++; return 1; }
  if(best){ best->last_used=g_frame; g_blit_rt=1; blit_to_screen(best->id,1,(int)best->w,(int)best->h); g_blit_rt=0; g_last_is_rt=1; st_pres_rt++; return 1; }
  st_pres_cpu++;
  static int said; if(said<sw2_cap(3)){ said++;
    LOG("gpu: the image to present (%llx %dx%d) is not a known render target. Render targets:\n",(unsigned long long)cpu_addr,w,h);
    int n=0; for(int k=0;k<TEXH && n<12;k++) for(Tex* t=texh[k];t && n<12;t=t->next) if(t->is_rt){ n++;
      LOG("   rt gva=%llx cpu=%llx %ux%u fmt=%x used=%u\n",(unsigned long long)t->gva,(unsigned long long)gmmu_translate(t->gva),t->w,t->h,t->fkey,t->last_used); }
    if(!n) LOG("   (none: the game has not drawn with the GPU yet)\n"); }
  return 0; }

/* ============================================================ GPU thread */
extern void gpu_submit_now(u64* e, u32 n, u32 channel);
static u64 g_last_shot_hash; static u32 g_shot_done;   /* sequence number of the last screenshot taken */
static void do_screenshot(const char* path){
  int w=g_last_w? g_last_w : 1280, h=g_last_h? g_last_h : 720; if(g_last_is_rt){ w=(int)SC((u32)w); h=(int)SC((u32)h); }
  u8* px=malloc((size_t)w*h*4);
  if(g_last_present_tex){ GLuint fb; glGenFramebuffers(1,&fb); glBindFramebuffer(GL_FRAMEBUFFER,fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,g_last_present_tex,0); glReadBuffer(GL_COLOR_ATTACHMENT0);
    glPixelStorei(GL_PACK_ALIGNMENT,1); glReadPixels(0,0,w,h,GL_RGBA,GL_UNSIGNED_BYTE,px); glBindFramebuffer(GL_FRAMEBUFFER,0); glDeleteFramebuffers(1,&fb); }
  else memset(px,0,(size_t)w*h*4);
  u64 hh=1469598103934665603ull;
  for(int y=0;y<h;y++){ int sy= g_last_flip? y : h-1-y; for(int x=0;x<w;x++){ u32 v; memcpy(&v,px+((size_t)sy*w+x)*4,4); v|=0xFF000000u; hh^=v; hh*=1099511628211ull; } }
  if(path){ char p[300]; snprintf(p,sizeof p,"%s.ppm",path); FILE* f=fopen(p,"wb");
    if(f){ fprintf(f,"P6\n%d %d\n255\n",w,h); for(int y=0;y<h;y++){ int sy= g_last_flip? y : h-1-y; for(int x=0;x<w;x++) fwrite(px+((size_t)sy*w+x)*4,1,3,f); } fclose(f); } }
  free(px); g_last_shot_hash=hh; }

/* debug: dump the recently used RTs to screenshots/ with some stats */
static void dump_rts(void){
  Tex* list[16]; int n=0;
  for(int k=0;k<TEXH;k++) for(Tex* t=texh[k];t;t=t->next) if(t->is_rt && !t->f.depth && !t->f.is_int){
    if(n<16){ list[n++]=t; } else { int mi=0; for(int i=1;i<16;i++) if(list[i]->last_used<list[mi]->last_used) mi=i; if(t->last_used>list[mi]->last_used) list[mi]=t; } }
  static int ndump; ndump++;
  LOG("---- render targets (dump %d, frame %u) ----\n",ndump,g_frame);
  plat_mkdir("captures");
  GLuint fb; glGenFramebuffers(1,&fb); glBindFramebuffer(GL_READ_FRAMEBUFFER,fb);
  for(int i=0;i<n;i++){ Tex* t=list[i]; u32 w=TW(t), h=TH(t); if(!w || !h || w>8192 || h>8192) continue;
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,t->id,0);
    if(glCheckFramebufferStatus(GL_READ_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE){ LOG("  rt %llx %ux%u fmt=%x: cannot be read\n",(unsigned long long)t->gva,t->w,t->h,t->fkey); continue; }
    u8* px=malloc((size_t)w*h*4); glReadBuffer(GL_COLOR_ATTACHMENT0); glPixelStorei(GL_PACK_ALIGNMENT,1); glReadPixels(0,0,(GLsizei)w,(GLsizei)h,GL_RGBA,GL_UNSIGNED_BYTE,px);
    u64 sr=0,sg=0,sb=0,sa=0,nz=0; size_t np=(size_t)w*h;
    for(size_t k=0;k<np;k++){ u8* q=px+k*4; sr+=q[0]; sg+=q[1]; sb+=q[2]; sa+=q[3]; if(q[0]+q[1]+q[2]>12) nz++; }
    LOG("  rt %llx %ux%u fmt=%x used=%u  average color=(%llu,%llu,%llu,a%llu)  non-black=%llu%%\n",(unsigned long long)t->gva,t->w,t->h,t->fkey,t->last_used,
      (unsigned long long)(sr/np),(unsigned long long)(sg/np),(unsigned long long)(sb/np),(unsigned long long)(sa/np),(unsigned long long)(nz*100/np));
    /* downscaled, max 480 wide */
    u32 step=(w+479)/480; if(!step) step=1; u32 ow=w/step, oh=h/step; char path[256]; snprintf(path,sizeof path,"captures/rt_%d_%02d_%llx_%ux%u.ppm",ndump,i,(unsigned long long)t->gva,t->w,t->h);
    FILE* f=fopen(path,"wb"); if(f){ fprintf(f,"P6\n%u %u\n255\n",ow,oh); for(u32 y=0;y<oh;y++) for(u32 x=0;x<ow;x++){ u8* q=px+(((size_t)(h-1-y*step))*w+x*step)*4; fwrite(q,1,3,f); } fclose(f); }
    free(px); }
  glBindFramebuffer(GL_READ_FRAMEBUFFER,0); glDeleteFramebuffers(1,&fb);
  LOG("--------------------------------------------\n"); fflush(NULL);
}

static void APIENTRY gl_dbg(GLenum src, GLenum type, GLuint id, GLenum sev, GLsizei len, const GLchar* msg, const void* up){
  (void)src;(void)id;(void)len;(void)up; if(sev==GL_DEBUG_SEVERITY_NOTIFICATION) return; if(type==GL_DEBUG_TYPE_ERROR) WARN("GL: %s\n",msg); }

/* ============================================================ shader pack
   the GLSL of every new program (doesn't depend on the driver) gets added to shader_cache/shader_pack.bin. at
   startup whatever isn't in the binary cache yet gets compiled in the background, the pack can be copied to
   another pc */
#define PACK_PATH "shader_cache/shader_pack_v3.bin"   /* versioned with the translator */
static u64* pk_keys; static u32 pk_n, pk_cap;

static int pk_has(u64 k){ if(!pk_cap) return 0; u32 i=(u32)(k^(k>>29))&(pk_cap-1); while(pk_keys[i]){ if(pk_keys[i]==k) return 1; i=(i+1)&(pk_cap-1); } return 0; }

static void pk_put(u64 k){ if(!k) k=1; if(pk_n*2>=pk_cap){ u32 oc=pk_cap; u64* ok=pk_keys; pk_cap= oc? oc*2 : 4096; pk_keys=calloc(pk_cap,8); pk_n=0; for(u32 i=0;i<oc;i++) if(ok[i]) pk_put(ok[i]); free(ok); }
  u32 i=(u32)(k^(k>>29))&(pk_cap-1); while(pk_keys[i]){ if(pk_keys[i]==k) return; i=(i+1)&(pk_cap-1); } pk_keys[i]=k; pk_n++; }

/* disk writes happen on a background thread, with an antivirus scanning an fopen on windows can take tens of ms */
typedef struct AwJob { struct AwJob* next; char path[200]; u8* data; size_t n; int append; } AwJob;

static AwJob* volatile g_aw_head; static AwJob* g_aw_tail; static PMutex g_aw_m; static volatile int g_aw_on;

static void aw_thread(void* arg){ (void)arg; plat_thread_low_priority();
  for(;;){ pm_lock(&g_aw_m); AwJob* j=g_aw_head; if(j){ g_aw_head=j->next; if(!g_aw_head) g_aw_tail=NULL; } pm_unlock(&g_aw_m);
    if(!j){ plat_sleep_ns(20000000); continue; }
    FILE* f=fopen(j->path,j->append? "ab" : "wb"); if(f){ fwrite(j->data,1,j->n,f); fclose(f); } free(j->data); free(j); } }

static void aw_write(const char* path, const void* d, size_t n, int append){
  if(!g_aw_on){ pm_init(&g_aw_m); g_aw_on= plat_thread(aw_thread,NULL,256*1024)? -1 : 1; }
  if(g_aw_on<0){ FILE* f=fopen(path,append? "ab" : "wb"); if(f){ fwrite(d,1,n,f); fclose(f); } return; }
  AwJob* j=calloc(1,sizeof *j); snprintf(j->path,sizeof j->path,"%s",path); j->data=malloc(n?n:1); memcpy(j->data,d,n); j->n=n; j->append=append;
  pm_lock(&g_aw_m); if(g_aw_tail) g_aw_tail->next=j; else g_aw_head=j; g_aw_tail=j; pm_unlock(&g_aw_m); }

static void pack_add(u64 key, Sh* vs, Sh* fs){
  if(!vs->src || !fs->src || pk_has(key)) return; pk_put(key);
  u32 a=(u32)strlen(vs->src), b=(u32)strlen(fs->src), mg=0x4B435053; size_t n=4+8+4+a+4+b; u8* d=malloc(n), *q=d;
  memcpy(q,&mg,4); q+=4; memcpy(q,&key,8); q+=8; memcpy(q,&a,4); q+=4; memcpy(q,vs->src,a); q+=a; memcpy(q,&b,4); q+=4; memcpy(q,fs->src,b);
  aw_write(PACK_PATH,d,n,1); free(d); }

static void pack_add_sh(u64 key, Sh* s){
  if(!s->src || pk_has(key)) return; pk_put(key);
  u32 a=(u32)strlen(s->src), mg=0x33504853, st=(u32)s->stage; u8* d=malloc(20+a); { u8* q=d; memcpy(q,&mg,4); q+=4; memcpy(q,&key,8); q+=8; memcpy(q,&st,4); q+=4; memcpy(q,&a,4); q+=4; memcpy(q,s->src,a); }
  aw_write(PACK_PATH,d,20+a,1); free(d); }

typedef struct { u64 key; char* vs; char* fs; int sep; u32 stage; } PackEnt;

static PackEnt* wq; static u32 wq_n, wq_i, wq_done;
typedef struct { GLuint p, a, b; u64 key; } WarmJob;

static WarmJob wj[6]; static int wj_n;

static void warm_load(void){
  pcache_init(); FILE* f=fopen(PACK_PATH,"rb"); if(!f) return; u32 cap=0;
  for(;;){ u32 mg,a,b; u64 key; if(fread(&mg,4,1,f)!=1 || (mg!=0x4B435053 && mg!=0x33504853) || fread(&key,8,1,f)!=1) break;
    if(mg==0x33504853){ u32 st=0; if(fread(&st,4,1,f)!=1 || fread(&a,4,1,f)!=1 || a>(8u<<20)) break;
      char* src=malloc(a+1); if(fread(src,1,a,f)!=a){ free(src); break; } src[a]=0;
      if(pk_has(key) || g_worker!=1){ free(src); continue; } pk_put(key);
      char path[160]; snprintf(path,sizeof path,"shader_cache/%s_s%016llx.bin",g_pc_tag,(unsigned long long)key);
      FILE* t= g_pcache? fopen(path,"rb") : NULL; if(t){ fclose(t); free(src); continue; }
      if(wq_n>=cap){ cap= cap? cap*2 : 256; wq=realloc(wq,cap*sizeof *wq); } wq[wq_n].key=key; wq[wq_n].vs=src; wq[wq_n].fs=NULL; wq[wq_n].sep=1; wq[wq_n].stage=st; wq_n++; continue; }
    if(fread(&a,4,1,f)!=1 || a>(8u<<20)) break;
    char* vs=malloc(a+1); if(fread(vs,1,a,f)!=a){ free(vs); break; } vs[a]=0;
    if(fread(&b,4,1,f)!=1 || b>(8u<<20)){ free(vs); break; } char* fs=malloc(b+1); if(fread(fs,1,b,f)!=b){ free(vs); free(fs); break; } fs[b]=0;
    if(pk_has(key)){ free(vs); free(fs); continue; } pk_put(key);
    char path[160]; snprintf(path,sizeof path,"shader_cache/%s_%016llx.bin",g_pc_tag,(unsigned long long)key);
    FILE* t= g_pcache? fopen(path,"rb") : NULL; if(t){ fclose(t); free(vs); free(fs); continue; }   /* already have a binary for this PC */
    if(wq_n>=cap){ cap= cap? cap*2 : 256; wq=realloc(wq,cap*sizeof *wq); } wq[wq_n].key=key; wq[wq_n].vs=vs; wq[wq_n].fs=fs; wq[wq_n].sep=0; wq[wq_n].stage=0; wq_n++; }
  fclose(f);
  if(wq_n) LOG("shaders: %u from the pack are precompiled in the background while you play\n",wq_n); }

static WJob* wlo[16]; static int wlo_n;

static void warm_step(void){
  if(g_worker==1){   /* compile thread available: give it a few low priority jobs at a time */
    for(int j=0;j<wlo_n;){ if(__atomic_load_n(&wlo[j]->state,__ATOMIC_ACQUIRE)){ free(wlo[j]->own_vs); free(wlo[j]->own_fs); free(wlo[j]); wlo[j]=wlo[--wlo_n]; wq_done++;
        if(wq_done==wq_n) LOG("shaders: precompiled pack (%u programs)\n",wq_n); } else j++; }
    while(wlo_n<8 && wq_i<wq_n){ PackEnt* e=&wq[wq_i]; WJob* j=calloc(1,sizeof *j); j->own_vs=e->vs; j->own_fs=e->fs; j->vs=e->vs; j->fs=e->fs; j->key=e->key; j->keep=0; j->sep=e->sep; j->stype=shader_gl_type(e->stage);
      if(!wr_push(&wq_lo,j)){ free(j); break; } e->vs=e->fs=NULL; wq_i++; wlo[wlo_n++]=j; }
    return; }
  if(g_par<=0 || !g_pcache) return;
  for(int j=0;j<wj_n;){ GLint done=0; glGetProgramiv(wj[j].p,0x91B1,&done); if(!done){ j++; continue; }
    GLint ok=0; glGetProgramiv(wj[j].p,GL_LINK_STATUS,&ok);
    if(ok){ GLint len=0; glGetProgramiv(wj[j].p,GL_PROGRAM_BINARY_LENGTH,&len);
      if(len>0){ void* bb=malloc(len); GLenum fmt=0; GLsizei got=0; p_glGetProgramBinary(wj[j].p,len,&got,&fmt,bb);
        char path[160]; snprintf(path,sizeof path,"shader_cache/%s_%016llx.bin",g_pc_tag,(unsigned long long)wj[j].key);
        if(got>0){ FILE* f=fopen(path,"wb"); if(f){ u32 fm=fmt, ln=(u32)got; fwrite(&fm,4,1,f); fwrite(&ln,4,1,f); fwrite(bb,1,got,f); fclose(f); } } free(bb); } }
    glDeleteProgram(wj[j].p); glDeleteShader(wj[j].a); glDeleteShader(wj[j].b); wj[j]=wj[--wj_n]; wq_done++;
    if(wq_done==wq_n) LOG("shaders: precompiled pack (%u programs)\n",wq_n); }
  while(wj_n<4 && wq_i<wq_n){ PackEnt* e=&wq[wq_i++]; if(e->sep){ free(e->vs); e->vs=NULL; wq_done++; continue; } WarmJob* j=&wj[wj_n++]; j->key=e->key;
    j->a=glCreateShader(GL_VERTEX_SHADER); const char* s1=e->vs; glShaderSource(j->a,1,&s1,NULL); glCompileShader(j->a);
    j->b=glCreateShader(GL_FRAGMENT_SHADER); const char* s2=e->fs; glShaderSource(j->b,1,&s2,NULL); glCompileShader(j->b);
    j->p=glCreateProgram(); glAttachShader(j->p,j->a); glAttachShader(j->p,j->b); p_glProgramParameteri(j->p,GL_PROGRAM_BINARY_RETRIEVABLE_HINT,GL_TRUE); glLinkProgram(j->p);
    free(e->vs); free(e->fs); e->vs=e->fs=NULL; } }

static FILE* g_rf; static u64 g_rf_t0;
static u64 g_fr_t0, g_fr_max; static int g_fr_slow; static u64 g_fr_shnew;
/* ---- dynamic res: move every RT to the new scale with a GPU copy */
/* UBO with 1/scale for gl_FragCoord math (shaders do uv = pos * 1/native size) */
static GLuint g_rs_ubo;

/* UBO with the render area (w, h, 1/w, 1/h in guest pixels), for NVN2's blit parameter bank (shader.c swr_blit);
   bound at g_rs_binding-1 */
static GLuint g_ra_ubo; static u32 g_ra_w, g_ra_h;
static void ra_update(u32 w, u32 h){ extern int g_rs_binding;
  if(!g_ra_ubo || !w || !h || (w==g_ra_w && h==g_ra_h)) return;
  g_ra_w=w; g_ra_h=h; float v[4]={(float)w,(float)h,1.0f/w,1.0f/h};
  glBindBuffer(GL_UNIFORM_BUFFER,g_ra_ubo); glBufferSubData(GL_UNIFORM_BUFFER,0,16,v); glBindBufferBase(GL_UNIFORM_BUFFER,(GLuint)g_rs_binding-1,g_ra_ubo); }

static void rs_update(void){ extern int g_rs_binding;
  if(!g_rs_ubo){ GLint mx=84; glGetIntegerv(GL_MAX_UNIFORM_BUFFER_BINDINGS,&mx); if(mx<3) return; g_rs_binding=mx-1; glGenBuffers(1,&g_rs_ubo); glBindBuffer(GL_UNIFORM_BUFFER,g_rs_ubo); glBufferData(GL_UNIFORM_BUFFER,16,NULL,GL_DYNAMIC_DRAW);
    glGenBuffers(1,&g_ra_ubo); glBindBuffer(GL_UNIFORM_BUFFER,g_ra_ubo); glBufferData(GL_UNIFORM_BUFFER,16,NULL,GL_DYNAMIC_DRAW); ra_update(1280,720); }
  float v[4]={1.0f/g_draw_scale,1.0f/g_draw_scale,g_draw_scale,0}; glBindBuffer(GL_UNIFORM_BUFFER,g_rs_ubo); glBufferSubData(GL_UNIFORM_BUFFER,0,16,v); glBindBufferBase(GL_UNIFORM_BUFFER,(GLuint)g_rs_binding,g_rs_ubo); }

static void rt_rescale(float ns){
  float old=g_scale; if(ns==old) return;
  glDisable(GL_SCISSOR_TEST); glDisable(GL_FRAMEBUFFER_SRGB); glColorMaski(0,1,1,1,1); glDepthMask(GL_TRUE); glStencilMaskSeparate(GL_FRONT_AND_BACK,0xFF);
  int n=0;
  /* parked aliases (gva ~0, alias set) come back through rt_texture without a size check, so resize them too */
  for(int k=0;k<TEXH;k++) for(Tex* t=texh[k];t;t=t->next){ if(!t->is_rt || (t->gva==~0ull && !t->alias)) continue;
    if(swr_rt_scale(t->w,t->h,old)==1.0f && swr_rt_scale(t->w,t->h,ns)==1.0f) continue;
    u32 ow=(u32)(t->w*old+0.5f), oh=(u32)(t->h*old+0.5f), nw=(u32)(t->w*ns+0.5f), nh=(u32)(t->h*ns+0.5f); if(!ow) ow=1; if(!oh) oh=1; if(!nw) nw=1; if(!nh) nh=1;
    GLuint nt=0; glGenTextures(1,&nt); glBindTexture(GL_TEXTURE_2D,nt); glTexStorage2D(GL_TEXTURE_2D,1,t->f.ifmt,nw,nh);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_LEVEL,0); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    int ds= t->f.depth && t->f.fmt==GL_DEPTH_STENCIL;
    GLenum att= t->f.depth? (ds? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT) : GL_COLOR_ATTACHMENT0;
    GLbitfield mask= t->f.depth? (GL_DEPTH_BUFFER_BIT|(ds? GL_STENCIL_BUFFER_BIT : 0)) : GL_COLOR_BUFFER_BIT;
    GLenum none=GL_NONE, c0=GL_COLOR_ATTACHMENT0;
    glBindFramebuffer(GL_READ_FRAMEBUFFER,g_rfbo); glFramebufferTexture2D(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,0,0); glFramebufferTexture2D(GL_READ_FRAMEBUFFER,GL_DEPTH_STENCIL_ATTACHMENT,GL_TEXTURE_2D,0,0);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER,att,GL_TEXTURE_2D,t->id,0); glReadBuffer(t->f.depth? GL_NONE : GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER,g_dfbo); glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,0,0); glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,GL_DEPTH_STENCIL_ATTACHMENT,GL_TEXTURE_2D,0,0);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,att,GL_TEXTURE_2D,nt,0); glDrawBuffers(1,t->f.depth? &none : &c0);
    glBlitFramebuffer(0,0,(GLint)ow,(GLint)oh,0,0,(GLint)nw,(GLint)nh,mask,(t->f.depth||t->f.is_int)? GL_NEAREST : GL_LINEAR);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER,att,GL_TEXTURE_2D,0,0); glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER,att,GL_TEXTURE_2D,0,0);
    if(g_last_present_tex==t->id) g_last_present_tex=nt;
    glDeleteTextures(1,&t->id); t->id=nt; t->scale=ns; t->wgen++; if(t->zc){ glDeleteTextures(1,&t->zc); t->zc=0; t->zc_ok=0; }
    if(t->crop){ glDeleteTextures(1,&t->crop->id); if(t->crop->view) glDeleteTextures(1,&t->crop->view); free(t->crop); t->crop=NULL; }
    /* built cubemaps/arrays/mip chains have scaled storage too, rebuild them at the new size */
    if(t->lay){ glDeleteTextures(1,&t->lay->id); if(t->lay->view) glDeleteTextures(1,&t->lay->view); free(t->lay->seen); free(t->lay); t->lay=NULL; }
    if(t->view){ glDeleteTextures(1,&t->view); t->view=0; }
    n++; }
  { GLenum c0=GL_COLOR_ATTACHMENT0; glBindFramebuffer(GL_READ_FRAMEBUFFER,g_rfbo); glReadBuffer(GL_COLOR_ATTACHMENT0); glBindFramebuffer(GL_DRAW_FRAMEBUFFER,g_dfbo); glDrawBuffers(1,&c0); }
  while(glGetError()!=GL_NO_ERROR){}
  g_scale=ns; g_draw_scale=ns; rs_update(); g_fb_sig_ok=0; g_tmap_gen++; mb_reset(); state_dirty(); glBindFramebuffer(GL_FRAMEBUFFER,g_fbo);
  LOG("gpu: internal resolution %.0f%% -> %.0f%% (%d render targets)\n",old*100,ns*100,n);
  if(g_rf){ fprintf(g_rf,"   internal resolution %.0f%% -> %.0f%%\n",old*100,ns*100); fflush(g_rf); } }

static u64 rd_t0, rd_f0, rd_busy0, rd_cpu0, rd_swap; static int rd_hold, rd_up; static u64 rd_lastdown;

static void res_dyn_step(void){
  extern int g_opt_resdyn; extern float g_opt_resmin, g_opt_resf;
  if(!g_opt_resdyn || !p_cis) return;
  u64 now=plat_ns(); extern u64 plat_thread_cpu_ns(void);
  if(!rd_t0){ rd_t0=now; rd_f0=g_frame; rd_busy0=g_gpu_busy; rd_cpu0=plat_thread_cpu_ns(); rd_swap=0; return; }
  u64 dt=now-rd_t0; if(dt<1000000000ull) return;
  double fps=(double)(g_frame-rd_f0)*1e9/dt, swapw=(double)rd_swap/dt, busy=(double)(g_gpu_busy-rd_busy0)/dt; u64 c=plat_thread_cpu_ns(); double cpu=(double)(c-rd_cpu0)/dt;
  rd_t0=now; rd_f0=g_frame; rd_busy0=g_gpu_busy; rd_cpu0=c; rd_swap=0;
  { static int tst=-1; if(tst<0) tst=getenv("SWR_RES_TEST")!=NULL; if(tst){ rt_rescale(g_scale>0.95f? 0.7f : 1.0f); return; } }   /* test: flip the scale back and forth */
  { /* compiling shaders / loading: those hitches aren't gpu load, don't drop res */
    static u64 last_new; u32 pend=__atomic_load_n(&wq_hi.t,__ATOMIC_RELAXED)-__atomic_load_n(&wq_hi.h,__ATOMIC_RELAXED);
    if(pend>0 || st_sh_new!=last_new){ last_new=st_sh_new; rd_hold=2; return; } }
  if(rd_hold>0){ rd_hold--; return; }
  float mx=g_opt_resf, mn=g_opt_resmin; if(mn>mx) mn=mx;
  /* gpu bound if the GPU thread spends its time in swap, or busy without using cpu */
  int gpu_bound= swapw>0.20 || (busy>0.90 && cpu<0.55*busy);
  int headroom= swapw<0.04 && busy<0.65;
  if(fps<28.5 && gpu_bound && g_scale>mn+0.01f){ float ns=g_scale-0.1f; if(ns<mn) ns=mn; rt_rescale(ns); rd_hold=2; rd_up=0; rd_lastdown=now; }
  else if(headroom && g_scale<mx-0.01f && now-rd_lastdown>20000000000ull){ if(++rd_up>=4){ float ns=g_scale+0.1f; if(ns>mx) ns=mx; rt_rescale(ns); rd_hold=3; rd_up=0; } }
  else rd_up=0; }

/* end of frame: was it slow, and where did the time go */
static void frame_stats(void){
  gp_frame_end(); wd_pass(NULL,0);
  u64 now=plat_ns(); if(!g_fr_t0){ g_fr_t0=now; memset(g_tm,0,sizeof g_tm); g_fr_shnew=st_sh_new; return; }
  u64 dt=now-g_fr_t0; g_fr_t0=now; if(dt>g_fr_max) g_fr_max=dt;
  if(dt>120000000ull && g_rf && g_fr_slow<300){ g_fr_slow++; u32 pend=__atomic_load_n(&wq_hi.t,__ATOMIC_RELAXED)-__atomic_load_n(&wq_hi.h,__ATOMIC_RELAXED);
    fprintf(g_rf,"   slow frame %.0f ms (GPU thread: shaders %.0f+%.0f, state %.0f, textures %.0f, buffers %.0f, driver %.0f, present %.0f ms) new shaders %llu, queued %u\n",
      dt/1e6,g_tm[TM_SH]/1e6,g_tm[TM_SHREADY]/1e6,g_tm[TM_STATE]/1e6,g_tm[TM_TEX]/1e6,(g_tm[TM_BUF]+g_tm[TM_CB]+g_tm[TM_GB])/1e6,g_tm[TM_DRAW]/1e6,g_tm[TM_SWAP]/1e6,(unsigned long long)(st_sh_new-g_fr_shnew),pend); }
  rd_swap+=g_tm[TM_SWAP];
  for(int k=0;k<TM_N;k++){ g_tmp[k]+=g_tm[k]; g_tm[k]=0; } g_fr_shnew=st_sh_new;
  { /* resolution edited in the ini while running: apply now */
    extern int g_opt_resdyn; extern float g_opt_resf; if(!g_opt_resdyn && p_cis && (g_scale>g_opt_resf+0.01f || g_scale<g_opt_resf-0.01f)) rt_rescale(g_opt_resf); }
  res_dyn_step(); }

/* ============================================================ startup screen
   SwitchRECOMP's logo and the copyright notice on black for a second, then it vanishes: it fades out while it grows
   a little. NOTICE.txt (AGPL section 7 terms) says copies and forks keep it */
#include "splash_img.h"
static void splash_play(double secs){
  int ww=0, wh=0; backend_window_size(&ww,&wh); if(ww<=0 || wh<=0) return;
  u8* px=malloc(SPLASH_W*SPLASH_H); if(!px) return; splash_unpack(px);
  GLuint tex=0; glGenTextures(1,&tex); glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,tex); glTexStorage2D(GL_TEXTURE_2D,1,GL_R8,SPLASH_W,SPLASH_H);
  glPixelStorei(GL_UNPACK_ALIGNMENT,1); glPixelStorei(GL_UNPACK_ROW_LENGTH,0); glTexSubImage2D(GL_TEXTURE_2D,0,0,0,SPLASH_W,SPLASH_H,GL_RED,GL_UNSIGNED_BYTE,px); free(px);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
  static const char* vs="#version 430 core\nuniform float sc; out vec2 uv; void main(){ vec2 p=vec2(float(gl_VertexID&1)*2.0-1.0,float(gl_VertexID>>1)*2.0-1.0);"
    " uv=(vec2(p.x*0.5+0.5,0.5-p.y*0.5)-0.5)/sc+0.5; gl_Position=vec4(p,0.0,1.0); }";
  static const char* fs="#version 430 core\nuniform float a; layout(binding=0) uniform sampler2D t; in vec2 uv; out vec4 o;"
    " void main(){ float v= any(lessThan(uv,vec2(0.0))) || any(greaterThan(uv,vec2(1.0)))? 0.0 : texture(t,uv).r; o=vec4(vec3(v*a),1.0); }";
  GLuint a=compile(GL_VERTEX_SHADER,vs,NULL,0), b=compile(GL_FRAGMENT_SHADER,fs,NULL,0), pr= a&&b? link2(a,b,0) : 0;
  if(pr){ GLint la=glGetUniformLocation(pr,"a"), ls=glGetUniformLocation(pr,"sc");
    double fin=0.12*secs, vstart=0.55*secs; u64 t0=plat_ns(), next=t0;
    for(;;){ double t=(double)(plat_ns()-t0)/1e9; if(t>=secs) break;
      double al=1.0, e=0.0;
      if(t<fin) al=t/fin;
      else if(t>vstart){ e=(t-vstart)/(secs-vstart); e=e*e*(3.0-2.0*e); al=1.0-e; }
      backend_window_size(&ww,&wh);
      glBindFramebuffer(GL_FRAMEBUFFER,0); glDisable(GL_SCISSOR_TEST); glDisable(GL_BLEND); glDisable(GL_DEPTH_TEST); glDisable(GL_STENCIL_TEST); glDisable(GL_CULL_FACE);
      glDisable(GL_FRAMEBUFFER_SRGB); glColorMaski(0,1,1,1,1);
      glViewport(0,0,ww,wh); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT);
      int dw=ww, dh=ww*SPLASH_H/SPLASH_W; if(dh>wh){ dh=wh; dw=wh*SPLASH_W/SPLASH_H; }
      glViewport((ww-dw)/2,(wh-dh)/2,dw,dh);
      glUseProgram(pr); glUniform1f(la,(float)al); glUniform1f(ls,(float)(1.0+0.06*e));
      glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D,tex); glBindSampler(0,0);
      glBindVertexArray(g_scratch_vao); glDrawArrays(GL_TRIANGLE_STRIP,0,4);
      screen_swap();
      next+=16666667ull; u64 now=plat_ns(); if(next>now) plat_sleep_ns((i64)(next-now)); else next=now; }
    glBindFramebuffer(GL_FRAMEBUFFER,0); glViewport(0,0,ww,wh); glClearColor(0,0,0,1); glClear(GL_COLOR_BUFFER_BIT); screen_swap(); }
  if(a) glDeleteShader(a); if(b) glDeleteShader(b); if(pr) glDeleteProgram(pr);
  glDeleteTextures(1,&tex); glUseProgram(0); state_dirty(); mb_reset(); }

static void gpu_thread(void* arg){ t_gpu_thread=1; { const char* e=getenv("SWR_PROF_FOCUS"); if(!e || strcmp(e,"gl")){ extern void plat_prof_focus_self(void); plat_prof_focus_self(); } }   /* profile=1: sampled on its own */
  (void)arg;
  { extern void plat_thread_high_priority(void); plat_thread_high_priority(); }
  { int qs=glq_start();   /* 1: GL lives on its own thread (context + fn pointers made there), 0: this thread */
    if(qs<0 || (qs==0 && (!backend_gl_create() || !swr_gl_load(backend_gl_proc)))){ WARN("No OpenGL 4.3: running without a graphics card\n"); g_gl=0; g_gl_ready=1; return; } }
  { extern int g_glsl_version; GLint mj=4, mn=3; glGetIntegerv(GL_MAJOR_VERSION,&mj); glGetIntegerv(GL_MINOR_VERSION,&mn); g_glsl_version= (mj>4||(mj==4&&mn>=4))? 440 : 430;
  }
  LOG("OpenGL: %s / %s / %s\n",(const char*)glGetString(GL_VENDOR),(const char*)glGetString(GL_RENDERER),(const char*)glGetString(GL_VERSION));
  { /* compile on the driver's threads so we don't freeze */
    g_par=0; int vl=0; GLint n=0; glGetIntegerv(GL_NUM_EXTENSIONS,&n); typedef const GLubyte* (APIENTRY*GSI2)(GLenum,GLuint); GSI2 gsi=(GSI2)backend_gl_proc("glGetStringi");
    for(GLint k=0;k<n && gsi;k++){ const char* e=(const char*)gsi(GL_EXTENSIONS,(GLuint)k); if(!e) continue;
      if(!strcmp(e,"GL_KHR_parallel_shader_compile")||!strcmp(e,"GL_ARB_parallel_shader_compile")) g_par=1;
      if(!strcmp(e,"GL_KHR_texture_compression_astc_ldr")) g_has_astc=1;
      if(!strcmp(e,"GL_ARB_shader_viewport_layer_array")) vl|=1; if(!strcmp(e,"GL_AMD_vertex_shader_layer")) vl|=2; if(!strcmp(e,"GL_NV_viewport_array2")) vl|=4;
      if(!strcmp(e,"GL_KHR_shader_subgroup")) vl|=8;
      if(!strcmp(e,"GL_NV_mesh_shader")){ extern int g_has_mesh; g_has_mesh=1; }
      if(strstr(e,"mesh") || strstr(e,"MESH")) LOG("gpu: extension %s\n",e); }
    /* warp ops (SHFL, VOTE, REDUX, MATCH, FSWZADD, lane id/masks) as GL subgroup ops: only with 32 wide subgroups,
       the guest's warp size. SWR_NO_SUBGROUPS=1 keeps the old one-lane emulation */
    { extern int g_subgroups; GLint sz=0; if(vl&8){ glGetIntegerv(0x9532 /* GL_SUBGROUP_SIZE_KHR */,&sz); while(glGetError()!=GL_NO_ERROR){} }
      g_subgroups= (vl&8) && sz==32 && !getenv("SWR_NO_SUBGROUPS");
      LOG("gpu: warp ops: %s (subgroup size %d)\n",g_subgroups? "GL_KHR_shader_subgroup" : "one-lane emulation",(int)sz); }
    { extern int g_vs_layer_ext; g_vs_layer_ext= (vl&1)? 1 : (vl&2)? 2 : (vl&4)? 3 : 0; if(getenv("SWR_NO_VS_LAYER")) g_vs_layer_ext=0;
      LOG("gpu: render target layer from the vertex shader: %s\n",g_vs_layer_ext==1? "ARB_shader_viewport_layer_array" : g_vs_layer_ext==2? "AMD_vertex_shader_layer" : g_vs_layer_ext==3? "NV_viewport_array2" : "not available (drawn to layer 0)"); }
    { const char* rn=(const char*)glGetString(GL_RENDERER); int s3tc=0;
      for(GLint k=0;k<n && gsi;k++){ const char* e=(const char*)gsi(GL_EXTENSIONS,(GLuint)k); if(e && (!strcmp(e,"GL_EXT_texture_compression_s3tc") || !strcmp(e,"GL_EXT_texture_compression_dxt1"))) s3tc=1; }
      int native= g_has_astc && rn && strstr(rn,"Intel") && !getenv("SWR_ASTC_BC");
      g_astc_bc= s3tc && !native && !getenv("SWR_ASTC_RGBA") && !getenv("SWR_ASTC_NATIVE");
      if(g_astc_bc) g_has_astc=0; }
    LOG("ASTC textures: %s\n",g_astc_bc? "decoded on the CPU and stored as BC3 (4x less memory than RGBA8, with the game's mipmaps)" : g_has_astc && !getenv("SWR_ASTC_CPU")? "native ASTC (the GPU reads them as they are)" : "decoded on the CPU to RGBA8");
    if(getenv("SWR_SYNC_SHADERS")) g_par=0;
    if(g_par){ typedef void (APIENTRY*MSCT)(GLuint); MSCT f=(MSCT)backend_gl_proc("glMaxShaderCompilerThreadsKHR"); if(!f) f=(MSCT)backend_gl_proc("glMaxShaderCompilerThreadsARB"); if(f) f(0xFFFFFFFFu); }
    LOG("shaders: parallel compilation %s\n",g_par? "enabled (no stutter: anything still compiling is drawn as soon as it is ready)" : "not available in this driver"); }
  { GLint v[6]={0}; glGetIntegerv(0x8B4D,&v[0]); glGetIntegerv(0x8872,&v[1]); glGetIntegerv(0x8A2F,&v[2]); glGetIntegerv(0x8A2B,&v[3]); glGetIntegerv(0x8A2D,&v[4]); glGetIntegerv(0x826E,&v[5]);
    extern int g_gl_ubo_blocks[6];
    g_gl_ubo_blocks[ST_VS]=v[3]; glGetIntegerv(0x8A2C,&g_gl_ubo_blocks[ST_GS]);
    g_gl_ubo_blocks[ST_FS]=v[4]; glGetIntegerv(0x8E89,&g_gl_ubo_blocks[ST_TCS]);
    glGetIntegerv(0x8E8A,&g_gl_ubo_blocks[ST_TES]); glGetIntegerv(0x91BB,&g_gl_ubo_blocks[ST_CS]);
    LOG("OpenGL limits: textures %d (per stage %d), UBO %d (VS %d, FS %d), uniform locations %d\n",v[0],v[1],v[2],v[3],v[4],v[5]); }
  if(g_verbose>=2){ typedef void (APIENTRY*DBGCB)(GLDEBUGPROC,const void*); DBGCB f=(DBGCB)backend_gl_proc("glDebugMessageCallback"); if(f){ glEnable(GL_DEBUG_OUTPUT); f(gl_dbg,NULL); } }
  glGenFramebuffers(1,&g_fbo); glGenFramebuffers(1,&g_rfbo); glGenFramebuffers(1,&g_dfbo); glGenVertexArrays(1,&g_vao); glGenVertexArrays(1,&g_scratch_vao); glGenBuffers(32,g_vbuf); glGenBuffers(1,&g_ibuf);
  for(int s=0;s<5;s++) glGenBuffers(18,g_ubuf[s]);
  { GLuint a=compile(GL_VERTEX_SHADER,BLIT_VS,NULL,0), b=compile(GL_FRAGMENT_SHADER,BLIT_FS,NULL,0); g_blit_prog=link2(a,b,0); }
  { glEnable(GL_FRAMEBUFFER_SRGB); glDisable(GL_FRAMEBUFFER_SRGB); glEnable(0x8642); }   /* GL_PROGRAM_POINT_SIZE, gl_PointSize comes from the shader */
  { extern float g_opt_resf; g_scale=g_opt_resf; g_draw_scale=g_scale; if(g_scale!=1.0f) LOG("internal resolution x%.2f\n",g_scale); }
  rs_update();
  worker_start();
  sep_init();
  mb_init(); ur_init();
  if(!getenv("SWR_NO_CLIPCONTROL")) p_clip=(PFNGLCLIPCONTROLPROC)backend_gl_proc("glClipControl");
  if(!getenv("SWR_NO_TEXVIEW")) p_tview=(PFNGLTEXTUREVIEWPROC)backend_gl_proc("glTextureView");
  lsfg_init();
  p_cis=(PFNGLCOPYIMAGESUBDATAPROC)backend_gl_proc("glCopyImageSubData");
  gl_selftest(); state_dirty(); mb_reset();
  LOG("gpu: native 0..1 depth %s, texture views %s\n",p_clip? "yes" : "no",p_tview? "yes" : "no");
  LOG("gpu: texture multi-bind %s, persistent constant ring %s\n",p_bt? "yes" : "no",g_ur_ptr? "yes" : "no");
  warm_load();
  g_gl=1; g_gl_ready=1;
  { extern void (*g_sub_done)(u32,u32); g_sub_done=sub_done; }
  splash_play(1.0);
  for(;;){
    warm_step();
    { /* loading screens etc: log how long we went without a new image */
      static u64 lp_t, lp_f; u64 tn=plat_ns(); if(g_frame!=lp_f){ if(lp_t && tn-lp_t>1500000000ull && g_rf){ fprintf(g_rf,"   -> %.1f s without a new image (loading or waiting)\n",(double)(tn-lp_t)/1e9); fflush(g_rf); } lp_f=g_frame; lp_t=tn; } }
    { static u64 next_st; u64 now=plat_ns(); if(!next_st) next_st=now+5000000000ull;
      if(now>=next_st){ static int nst; nst++; next_st=now+(nst<24? 5000000000ull : 30000000000ull);
        extern volatile u64 g_gpu_submits; extern int g_frames; extern volatile u64 g_dma_copies; extern volatile u64 g_compute_launches;
        LOG("gpu: frames=%d submits=%llu draws=%llu (drawn %llu, no shader %llu, failed shader %llu, no program %llu, no target %llu) clears=%llu shaders=%llu present: rt=%llu cpu=%llu | textures ok=%llu rt=%llu null=%llu type-mismatch=%llu, dma copies=%llu, compute=%llu, waiting for shader=%llu\n",
          g_frames,(unsigned long long)g_gpu_submits,(unsigned long long)g_draws,(unsigned long long)st_drawn,(unsigned long long)st_noshader,(unsigned long long)st_shfail,(unsigned long long)st_noprog,(unsigned long long)st_nort,
          (unsigned long long)st_clears,(unsigned long long)st_shaders,(unsigned long long)st_pres_rt,(unsigned long long)st_pres_cpu,
          (unsigned long long)st_tex_ok,(unsigned long long)st_tex_rt,(unsigned long long)st_tex_null,(unsigned long long)st_tex_mismatch,(unsigned long long)g_dma_copies,(unsigned long long)g_compute_launches,(unsigned long long)st_compiling);
        { u64 tn=0, tb=0, rn=0, rb=0; for(int k=0;k<TEXH;k++) for(Tex* t=texh[k];t;t=t->next){ u64 b= t->is_rt? (u64)t->w*t->h*(t->d? t->d : 1)*(t->f.bpp && !t->f.compressed? t->f.bpp : 4) : tex_host_bytes(t); if(t->is_rt){ rn++; rb+=b; } else { tn++; tb+=b; } }
          LOG("gpu memory: buffer cache %llu buffers %llu MB, textures %llu (~%llu MB), render targets %llu (~%llu MB at x1)\n",(unsigned long long)g_vbc_n,(unsigned long long)(g_vbc_bytes>>20),
            (unsigned long long)tn,(unsigned long long)(tb>>20),(unsigned long long)rn,(unsigned long long)(rb>>20)); }
        static u64 last_frames_pf;
        { static u64 last_t, last_busy, last_frames; u64 dt=now-last_t; if(last_t && dt){
            LOG("performance: %.1f FPS, GPU thread busy %d%% (near 100%% = the bottleneck is the graphics card; low = it is the CPU running the game)\n",
              (double)(g_frame-last_frames)*1e9/(double)dt,(int)((g_gpu_busy-last_busy)*100/dt)); }
          last_t=now; last_busy=g_gpu_busy; last_frames=g_frame; }
          { /* performance.txt: short summary to figure out problems on someone else's pc */
            static u64 l_now, l_pcpu, l_gcpu, l_cn, l_cns, l_fr2, l_skip, l_dr2; FILE* rf=g_rf; u64 t_start=g_rf_t0; if(!rf){ rf=g_rf=fopen("performance.txt","w"); t_start=g_rf_t0=now; if(rf){ extern int plat_ncpu(void);
                fprintf(rf,"SwitchRecomp - performance (each line = last few seconds). CPU: %d cores. GPU: %s\n",plat_ncpu(),(const char*)glGetString(GL_RENDERER));
                { extern int g_code_tcc; if(g_code_tcc) fprintf(rf,"WARNING: the game code was compiled with TCC and runs several times slower than with GCC. Recompile with GCC.\n"); }
                fprintf(rf," sec |  FPS | process CPU | GPU thread (CPU) | draws/frame | shaders compiled (avg ms, max) | draws waiting for shader\n"); } }
            u64 pc=plat_process_cpu_ns(), gc=plat_thread_cpu_ns(); u64 dtn=now-l_now;
            if(rf && l_now && dtn){ extern int plat_ncpu(void); u64 cn=g_cmp_n-l_cn, cns=g_cmp_ns-l_cns; u64 frn=g_frame-l_fr2;
              fprintf(rf,"%4llu | %4.1f | %3d%% of %d cores | %3d%% | %6llu | %4llu (%5.1f, %5.1f) | %llu\n",(unsigned long long)((now-t_start)/1000000000ull),(double)frn*1e9/(double)dtn,
                (int)((pc-l_pcpu)*100/dtn),plat_ncpu(),(int)((gc-l_gcpu)*100/dtn),(unsigned long long)(frn? (st_drawn-l_dr2)/frn : 0),(unsigned long long)cn,cn? (double)cns/cn/1e6 : 0.0,(double)g_cmp_max/1e6,(unsigned long long)(st_compiling-l_skip));
              { double fr= frn? (double)frn : 1.0; static u64 l_bn, l_bns, l_csw;
                fprintf(rf,"      GPU thread breakdown (ms per frame): shaders %.1f + load %.1f | state %.1f | textures %.1f | buffers %.1f (constants %.1f, storage %.1f, vertices %.1f, compute waits %.1f) | driver draw %.1f | present/wait GPU %.1f | slowest frame %.0f ms | cache binaries %llu (%.1f ms)\n",
                  g_tmp[TM_SH]/1e6/fr,g_tmp[TM_SHREADY]/1e6/fr,g_tmp[TM_STATE]/1e6/fr,g_tmp[TM_TEX]/1e6/fr,(g_tmp[TM_BUF]+g_tmp[TM_CB]+g_tmp[TM_GB])/1e6/fr,
                  g_tmp[TM_CB]/1e6/fr,g_tmp[TM_GB]/1e6/fr,g_tmp[TM_BUF]/1e6/fr,(double)(st_cs_sync_ns-l_csw)/1e6/fr,g_tmp[TM_DRAW]/1e6/fr,g_tmp[TM_SWAP]/1e6/fr,g_fr_max/1e6,
                  (unsigned long long)(g_bin_n-l_bn),(g_bin_n-l_bn)? (double)(g_bin_ns-l_bns)/(g_bin_n-l_bn)/1e6 : 0.0);
                l_bn=g_bin_n; l_bns=g_bin_ns; l_csw=st_cs_sync_ns; memset(g_tmp,0,sizeof g_tmp); g_fr_max=0;
                gp_report(rf);
                { char sr[400]; glq_sync_report(sr,sizeof sr,(u32)fr); if(sr[0]) fprintf(rf,"      GPU thread waiting on the GL thread: %s\n",sr); }
                { static u64 l_gl; u64 g=glq_thread_cpu_ns(); if(g && l_gl && dtn) fprintf(rf,"      GL thread (video driver) CPU: %d%% of a core\n",(int)((g-l_gl)*100/dtn)); l_gl=g; }
                { static u64 l_lg, l_lns; u64 ng=st_lsfg_gen-l_lg; if(ng){ fprintf(rf,"      frame generation (LSFG): +%llu frames -> %.1f FPS on screen, %.1f ms of GPU thread per generated frame\n",
                    (unsigned long long)ng,(double)(frn+ng)*1e9/(double)dtn,(double)(st_lsfg_ns-l_lns)/(double)ng/1e6); } l_lg=st_lsfg_gen; l_lns=st_lsfg_ns; } }
              fflush(rf); g_cmp_max=0; }
            l_now=now; l_pcpu=pc; l_gcpu=gc; l_cn=g_cmp_n; l_cns=g_cmp_ns; l_fr2=g_frame; l_skip=st_compiling; l_dr2=st_drawn; }
          { static u64 l_cbs, l_cbb, l_vbh, l_vbu, l_tex, l_dr; u64 fr=g_frame-last_frames_pf; if(!fr) fr=1;
            static u64 l_shf, l_sh, l_fbs, l_tbs, l_urw, l_tck, l_ss, l_dm, l_csg, l_css, l_csn, l_th, l_tm; extern u64 g_draws_merged;
            LOG("per-frame profile: %llu draws | constants: %.1f KB copied, %llu reused | vertices/indices: %.1f KB checked, %.1f KB uploaded | textures: %.1f KB checked, %.1f KB uploaded | shaders via fast cache %llu/%llu | same fbo %llu, same textures %llu, same state %llu, ring waits %llu, merged instances %llu | compute: %llu GPU copies, %llu waits (%.1f ms) | texture lookups reused %.0f%%\n",
              (unsigned long long)((st_drawn-l_dr)/fr),(double)(st_cb_bytes-l_cbb)/1024.0/fr,(unsigned long long)((st_cb_skip-l_cbs)/fr),(double)(st_vb_hashed-l_vbh)/1024.0/fr,(double)(st_vb_up-l_vbu)/1024.0/fr,
              (double)(st_tex_chk-l_tck)/1024.0/fr,(double)(st_tex_up-l_tex)/1024.0/fr,(unsigned long long)((st_sh_fast-l_shf)/fr),(unsigned long long)((st_shaders-l_sh)/fr),
              (unsigned long long)((st_fb_skip-l_fbs)/fr),(unsigned long long)((st_tb_skip-l_tbs)/fr),(unsigned long long)((st_state_skip-l_ss)/fr),(unsigned long long)(st_ur_wait-l_urw),(unsigned long long)((g_draws_merged-l_dm)/fr),
              (unsigned long long)((st_cs_gpucopy-l_csg)/fr),(unsigned long long)((st_cs_sync-l_css)/fr),(double)(st_cs_sync_ns-l_csn)/1e6/fr,
              (st_tdc_hit-l_th)+(st_tdc_miss-l_tm)? 100.0*(st_tdc_hit-l_th)/(double)((st_tdc_hit-l_th)+(st_tdc_miss-l_tm)) : 0.0);
            l_th=st_tdc_hit; l_tm=st_tdc_miss;
            l_csg=st_cs_gpucopy; l_css=st_cs_sync; l_csn=st_cs_sync_ns;
            l_dm=g_draws_merged; l_shf=st_sh_fast; l_sh=st_shaders; l_fbs=st_fb_skip; l_tbs=st_tb_skip; l_urw=st_ur_wait; l_tck=st_tex_chk; l_ss=st_state_skip;
            l_cbs=st_cb_skip; l_cbb=st_cb_bytes; l_vbh=st_vb_hashed; l_vbu=st_vb_up; l_tex=st_tex_up; l_dr=st_drawn; last_frames_pf=g_frame; } } }
    lsfg_pump();
    i64 to_=100000000; if(g_lfn){ u64 nw=plat_ns(); to_= g_lfq[0].t>nw? (i64)(g_lfq[0].t-nw) : 0; if(to_<200000) to_=200000; }
#ifdef _WIN32
    if(g_rec_on && rec_failed()) rec_stop();
    if(g_rec_exit==1){ rec_stop(); g_rec_exit=2; }
    if(g_rec_toggle && g_rec_on){ g_rec_toggle=0; rec_stop(); }   /* let F7 stop even when nothing is being presented */
#endif
    /* parked channels first: one resuming here can finish a frame and send its last compute readback. That used to
       run after the readback check below, so the thread then slept the full idle wait (100 ms) with the readback in
       flight and the frame's fence held behind it: every frame of MKW took 100 ms more than its work */
    if(g_parked_any){ g_parked_any=gpu_chan_pump_all(); if(g_parked_any && to_>500000) to_=500000; }
    /* compute results on their way back (or copies not sent yet, or fence releases held for them): look again in 1 ms */
    if(cs_pending_any()){ cs_kick_async(); cs_complete(0); cs_stale_check(); if(cs_pending_any() && to_>1000000) to_=1000000; }
    g_gpu_phase=NULL; Work w; if(!q_pop(&w,to_)){ lsfg_pump(); continue; }
    u64 t_busy0=plat_ns(); g_gpu_phase_t0=t_busy0; g_gpu_phase= w.kind==WK_SUBMIT? "running a submit" : "presenting";
    switch(w.kind){
      case WK_SUBMIT: g_submit_epoch++; ww_poll(); { extern volatile u64 g_gpu_sub_ns; static u32 fr=~0u; if(fr!=g_frame || !g_gpu_sub_ns){ fr=g_frame; g_gpu_sub_ns=w.sub_ns; } }   /* stamp the whole frame with the time of its first submit */
        /* each channel runs its own queue: a submit whose fence or semaphore isn't there yet waits without holding up
           the others (gpu.c gpu_chan_submit) */
        { extern void swr_tl(const char*,...); swr_tl("gpu thread takes submit sp%u",w.syncpt); } g_parked_any=gpu_chan_submit(w.ent,w.n,w.syncpt,w.incs,w.wait_id,w.wait_val); break;
      case WK_PRESENT_CPU: if(g_skip_frame){ free(w.px); g_frame++; g_skip_frame= g_frame%(frame_skip_count()+1); break; }
        g_present_guest_ns=w.sub_ns; trace_note("PRESENT CPU image %dx%d\n",w.w,w.h); present_cpu(w.px,w.w,w.h); free(w.px); g_frame++; g_skip_frame= frame_skip_count() && g_frame%(frame_skip_count()+1); gpu_evict_step(); frame_stats(); trace_frame(); break;
      case WK_PRESENT_ADDR: if(g_skip_frame){ g_frame++; g_skip_frame= frame_skip_count() && g_frame%(frame_skip_count()+1); break; }
        g_present_guest_ns=w.sub_ns;
        if(!present_addr(w.addr,w.w,w.h)){ extern void vi_convert_cpu(u64 a, u32 w, u32 h, u32 pitch, u32 layout, u32 bhl, u32 fmt, u32* out);
          u32* px=malloc((size_t)w.w*w.h*4); vi_convert_cpu(w.addr,w.w,w.h,w.n,(w.fmt>>8)&15,(w.fmt>>12)&15,w.fmt&0xFF,px); present_cpu(px,w.w,w.h); free(px); } g_frame++; g_skip_frame= frame_skip_count() && g_frame%(frame_skip_count()+1); gpu_evict_step(); frame_stats(); trace_frame(); break;
      case WK_DUMP: dump_rts(); g_tr_arm=1; break;
      case WK_SHOT: do_screenshot(w.path[0]? w.path : NULL); __atomic_store_n(&g_shot_done,w.n,__ATOMIC_SEQ_CST); break;
    }
    g_gpu_busy+=plat_ns()-t_busy0;
  }
}

void gpu_gl_start(void){
  pm_init(&g_ql); pc_init(&g_qcv); pc_init(&g_qspace);
  if(plat_thread(gpu_thread,NULL,64u<<20)) fatal("could not create the GPU thread");
  while(!g_gl_ready) plat_sleep_ns(1000000);
}

/* requests from other threads */
int gpu_async_submit(u64* e, u32 n, u32 syncpt, u32 incs, u32 wait_id, u32 wait_val){
  if(!g_gl) return 0;
  Work w; memset(&w,0,sizeof w); w.kind=WK_SUBMIT; { extern u64 guest_ns(void); w.sub_ns=guest_ns(); } w.ent=malloc((size_t)n*8+8); memcpy(w.ent,e,(size_t)n*8); w.n=n; w.syncpt=syncpt; w.incs=incs; w.wait_id=wait_id; w.wait_val=wait_val; q_push(&w); return 1; }

void gpu_gl_present_cpu(const u32* px, int w, int h, int stride){
  if(!g_gl) return;
  Work k; memset(&k,0,sizeof k); k.kind=WK_PRESENT_CPU; {extern u64 guest_ns(void); k.sub_ns=guest_ns(); } k.px=malloc((size_t)w*h*4); for(int y=0;y<h;y++) memcpy(k.px+(size_t)y*w,px+(size_t)y*stride,(size_t)w*4); k.w=w; k.h=h; q_push(&k); }

void gpu_gl_present_buffer(u64 cpu_addr, u32 w, u32 h, u32 pitch, u32 layout, u32 bhl, u32 fmt){
  Work k; memset(&k,0,sizeof k); k.kind=WK_PRESENT_ADDR; {extern u64 guest_ns(void); k.sub_ns=guest_ns(); } k.addr=cpu_addr; k.w=(int)w; k.h=(int)h; k.fmt=fmt|(layout<<8)|(bhl<<12); k.n=pitch; q_push(&k); }

u64 gpu_gl_screenshot(const char* path){
  { extern int g_gpu_nvn; extern void nvn_shot(const char*); if(g_gpu_nvn){ if(path) nvn_shot(path); return 0; } }
  { extern int g_gpu_eden; extern u64 gpu_eden_screenshot(const char*); if(g_gpu_eden) return gpu_eden_screenshot(path); }   /* SWR_GPU=eden: Eden's frame */
  if(!g_gl) return 0;
  /* wait on a sequence number and not a flag on this stack, after a timeout the GPU thread would write into a
     stack frame that's gone */
  static u32 seq; u32 my=++seq;
  Work k; memset(&k,0,sizeof k); k.kind=WK_SHOT; k.n=my; if(path) strncpy(k.path,path,255); q_push(&k);
  u64 t0=plat_ns(); while((i32)(__atomic_load_n(&g_shot_done,__ATOMIC_SEQ_CST)-my)<0 && plat_ns()-t0<30000000000ull) plat_sleep_ns(2000000);
  return g_last_shot_hash; }

void gpu_gl_dump_rts(void){ if(!g_gl) return; Work k; memset(&k,0,sizeof k); k.kind=WK_DUMP; q_push(&k); }

/* ============================================================ compute (kepler compute class, QMD)
   SMO fires ~25 dispatches a frame (particles, sims) that touch global memory and a 3D texture. QMD -> compute
   GLSL, bind cbufs, textures and ssbos ({addr,size} in c0), and copy back whatever got written to guest memory
   after */
/* compute output goes gpu side into a staging buffer and gets read back in one go: before the next draw, at the
   end of the submit, or earlier if another dispatch reads that memory */
/* moff: where the dispatch's write bits for this buffer sit in the same staging buffer (0 = none, copy it all) */
typedef struct { u64 addr; u32 off, size, moff; } CsPend;
static GLuint g_cs_stage; static u32 g_cs_stage_cap, g_cs_stage_used; static int g_cs_npend;
static CsPend g_cs_pend[256];

/* only the words a dispatch stored to go back to guest memory, like on hw. the rest of the bound range is a copy from
   dispatch time and the cpu may have written there since, copying it all back undid those writes a frame late. the
   shader sets one bit per stored word (swr_wmark). SWR_CS_FULL_COPY=1 copies the whole range */

/* the readback is async: at the end of every submit the staging copy gets a fence and we pick it up when it's
   signaled. only a real read of that memory (draw, texture, DMA, another compute) has to wait */
/* up to CS_FLN readbacks in flight, landed oldest first. With one, every submit with compute output waited for the
   host GPU to finish the previous one (MKW: ~65% of the GPU thread's time, ~24 ms per submit) */
#define CS_FLN 16
typedef struct { GLuint buf; u32 cap, used; GLsync fence; int n; CsPend p[256]; u64 t0, seq; } CsFl;
static CsFl g_csq[CS_FLN]; static int g_csq_h, g_csq_n;   /* in flight: g_csq_n of them from g_csq_h */
static u64 g_cs_kseq, g_cs_dseq;   /* batches kicked / landed */
#define g_cs_fl (g_csq[g_csq_h])   /* the oldest in flight (fence 0: nothing in flight) */

static u64 g_cs_lo=~0ull, g_cs_hi;   /* bounds of everything pending/in flight, for a quick reject */
static int g_cs_busy; volatile int g_cs_any;   /* g_cs_any: anything pending/in flight (gmem_ptr/gmem_read check it) */
/* what has to wait for the results in flight, in stream order: syncpoint incrs (kind 0), semaphore writes (1) and
   semaphore reductions (2: id=op, n=payload, len=signed) */
typedef struct { u8 kind, len; u32 id, n; u64 addr; u8 data[16]; u64 seq; } CsHeld;   /* seq: the newest batch kicked before it */
#define CS_HELDN 1024   /* with 16 batches in flight many more writes wait; a full list drained everything */
static CsHeld g_cs_incs[CS_HELDN]; static int g_cs_nincs;

/* the words whose bit is set, in runs */
static void cs_put_marked(u64 addr, const u8* src, const u32* bits, u32 size){
  u8* dst=gmem_ptr(addr,size); u32 nw=size/4;
  for(u32 w=0;w<nw;){
    if(!bits[w>>5]){ w=(w|31)+1; continue; }
    if(!((bits[w>>5]>>(w&31))&1)){ w++; continue; }
    u32 e=w+1; while(e<nw && ((bits[e>>5]>>(e&31))&1)) e++;
    if(dst) memcpy(dst+w*4,src+w*4,(e-w)*4); else gmem_write(addr+w*4,src+w*4,(e-w)*4);
    w=e; } }

static void cs_apply(const u8* tmp, const CsPend* p, int n){
  extern void gpu_mem_written(u64,u64);
  static int full=-1; if(full<0) full=getenv("SWR_CS_FULL_COPY")!=NULL;
  g_cs_busy=1;
  /* cached vertex/storage buffers on this range are stale */
  g_csgen++; for(int i=0;i<n;i++) for(u64 b=p[i].addr>>16, e=(p[i].addr+p[i].size-1)>>16; b<=e; b++) g_csblk[b&0xFFFF]=g_csgen;
  for(int i=0;i<n;i++){
    u8* dst=gmem_ptr(p[i].addr,p[i].size);
    if(p[i].moff && !full) cs_put_marked(p[i].addr,tmp+p[i].off,(const u32*)(tmp+p[i].moff),p[i].size);
    else if(dst) memcpy(dst,tmp+p[i].off,p[i].size); else gmem_write(p[i].addr,tmp+p[i].off,p[i].size);
    if(dst) for(u64 c=(u64)(dst-g_mem), ce=c+p[i].size; c<ce; c=(c|0xFFFF)+1) ww_touch(c);
    gpu_mem_written(p[i].addr,p[i].size); }   /* constant buffers cached this submit too */
  g_cs_busy=0; }

static u8* cs_tmp(u32 n){ static u8* tmp; static u32 cap; if(cap<n){ tmp=realloc(tmp,n); cap=n; } return tmp; }

static void cs_release_held(void){
  extern void syncpt_incr(u32 id); extern void gpu_sem_reduce(u64,u32,u32,int); extern void swr_tl(const char*,...); if(g_cs_nincs) swr_tl("release %d held",g_cs_nincs);
  for(int i=0;i<g_cs_nincs;i++){ CsHeld* h=&g_cs_incs[i];
    if(h->kind==0) for(u32 k=0;k<h->n;k++) syncpt_incr(h->id);
    else if(h->kind==1) gmem_write(h->addr,h->data,h->len);
    else gpu_sem_reduce(h->addr,h->id,h->n,h->len); }
  g_cs_nincs=0; }
/* the held writes whose batches have all landed (they're in stream order, seq never goes down) */
static void cs_release_upto(u64 seq){
  int k=0; while(k<g_cs_nincs && g_cs_incs[k].seq<=seq) k++; if(!k) return;
  int rest=g_cs_nincs-k; g_cs_nincs=k; cs_release_held(); memmove(g_cs_incs,g_cs_incs+k,sizeof(CsHeld)*rest); g_cs_nincs=rest; }

/* next free slot while something is in flight. still full after a timeout: let go of what's held instead of writing
   past g_cs_incs */
static CsHeld* cs_hold_slot(void){
  if(!g_cs_fl.fence) return NULL;
  { static int nh=-1; if(nh<0) nh=getenv("SWR_NO_FENCE_HOLD")!=NULL; if(nh) return NULL; }   /* test: fences don't wait for compute results */
  if(g_cs_nincs>=CS_HELDN){ cs_complete(1); if(g_cs_nincs>=CS_HELDN) cs_release_held(); if(!g_cs_fl.fence) return NULL; }
  CsHeld* h=&g_cs_incs[g_cs_nincs++]; memset(h,0,sizeof *h); h->seq=g_cs_kseq; return h; }

/* wait=0: only those the GPU already finished. max: how many to land at most (oldest first) */
static void cs_complete_n(int wait, int max){
  while(g_csq_n && max-->0){ CsFl* q=&g_csq[g_csq_h];
    GLenum r=p_cws? p_cws(q->fence,GL_SYNC_FLUSH_COMMANDS_BIT,wait? 4000000000ull : 0) : GL_ALREADY_SIGNALED;
    if(r==GL_TIMEOUT_EXPIRED){
      /* 4 s and the GPU still isn't done: let the held fences go anyway, the game waits on them and would sit there
         forever. those results come in late */
      if(wait && g_cs_nincs){ static int nl; if(nl<sw2_cap(4)){ nl++; LOG("gpu: compute results still not back after 4 s, letting %d held fence write(s) go\n",g_cs_nincs); } cs_release_held(); }
      return; }
    { extern void swr_tl(const char*,...); swr_tl("readback seq %llu landed %.1f ms after its fence (%d in flight)",(unsigned long long)q->seq,(plat_ns()-q->t0)/1e6,g_csq_n); }
    if(p_ds) p_ds(q->fence); q->fence=0;
    u8* tmp=cs_tmp(q->used); glBindBuffer(GL_COPY_READ_BUFFER,q->buf); glGetBufferSubData(GL_COPY_READ_BUFFER,0,q->used,tmp);
    cs_apply(tmp,q->p,q->n); q->n=0; q->used=0; g_cs_dseq=q->seq; g_csq_h=(g_csq_h+1)%CS_FLN; g_csq_n--;
    /* submits that finished while this was in flight only count as done now, the game waits on them before reading */
    cs_release_upto(g_cs_dseq); }
  if(!g_csq_n){ g_cs_any= g_cs_npend>0; if(!g_cs_any){ g_cs_lo=~0ull; g_cs_hi=0; } } }
static void cs_complete(int wait){ cs_complete_n(wait,CS_FLN); }
/* blocking waits for compute results by reason, logged every 10 s: 0 a read of compute output, 1 a cross-queue wait
   on a held release, 2 the in-flight ring full */
static u64 g_csw_ns[3], g_csw_n[3], g_csw_t;
static void cs_wait_timed(int why, int max){ u64 t0=plat_ns(); cs_complete_n(1,max); u64 t1=plat_ns(); g_csw_ns[why]+=t1-t0; g_csw_n[why]++;
  if(!g_csw_t) g_csw_t=t1; if(t1-g_csw_t>10000000000ull){ LOG("gpu: compute result waits (10 s): reads of compute output %llu, %.0f ms | cross-queue waits %llu, %.0f ms | ring full %llu, %.0f ms\n",
    (unsigned long long)g_csw_n[0],g_csw_ns[0]/1e6,(unsigned long long)g_csw_n[1],g_csw_ns[1]/1e6,(unsigned long long)g_csw_n[2],g_csw_ns[2]/1e6); memset(g_csw_ns,0,sizeof g_csw_ns); memset(g_csw_n,0,sizeof g_csw_n); g_csw_t=t1; } }

/* syncpoint incrs of a submit get held back (in order) while compute results are in flight */
static void cs_submit_done(u32 id, u32 n){
  extern void syncpt_incr(u32 id); extern void swr_tl(const char*,...);
  CsHeld* h=cs_hold_slot(); if(h){ h->id=id; h->n=n; swr_tl("hold sp%u +%u (%d readbacks in flight)",id,n,g_csq_n); return; }
  for(u32 k=0;k<n;k++) syncpt_incr(id); }
static void sub_done(u32 id, u32 n){ cs_kick_async(); cs_submit_done(id,n); }

/* semaphore writes from the stream wait for the compute results before them too. NVN fences are semaphores: the cpu
   checks the word and when it moved takes the GPU's output as done. tomodachi got its culling results that way before
   our readback landed, read nothing visible and dropped every house, tree and mii for a frame (only the island and
   some shadows left). 1 = held, the caller doesn't write. SWR_NO_SEM_HOLD=1 writes them right away */
static CsHeld* cs_hold_sem(void){
  static int off=-1; if(off<0) off=getenv("SWR_NO_SEM_HOLD")!=NULL;
  if(off || !t_gpu_thread) return NULL;
  if(g_cs_npend) cs_kick_async();
  return cs_hold_slot(); }

int gpu_gl_hold_write(u64 addr, const void* data, u32 len){
  if(len>16) return 0;
  CsHeld* h=cs_hold_sem(); if(!h) return 0;
  h->kind=1; h->len=(u8)len; h->addr=addr; memcpy(h->data,data,len); return 1; }

int gpu_gl_hold_reduce(u64 addr, u32 op, u32 payload, int is_signed){
  CsHeld* h=cs_hold_sem(); if(!h) return 0;
  h->kind=2; h->addr=addr; h->id=op; h->n=payload; h->len=(u8)(is_signed!=0); return 1; }

/* an incr inside the stream means everything before it is done, so it has to go after any pending compute results */
void gpu_gl_stream_incr(u32 id){
  extern void syncpt_incr(u32 id);
  if(!t_gpu_thread){ syncpt_incr(id); return; }
  if(g_cs_npend) cs_kick_async();
  cs_submit_done(id,1); }

/* the stream is about to wait on a syncpoint: flush everything in flight */
void gpu_gl_before_syncwait(void){ if(t_gpu_thread && g_cs_fl.fence) cs_complete(1); }
/* the stream waits on something (kind 3: syncpoint id, 1/2: semaphore at addr): wait for the compute results in flight
   only when that release is one of the writes held back for them. Otherwise the producer hasn't run yet; waiting for
   the host GPU to finish everything was a full drain at every cross-queue wait (MKW: ~45% of the GPU
   thread's time, the game's frames serialized on the host GPU) */
void gpu_gl_before_wait(int kind, u64 addr){
  if(!t_gpu_thread || !g_cs_fl.fence) return;
  for(int i=0;i<g_cs_nincs;i++){ CsHeld* h=&g_cs_incs[i];
    if(kind==3? (h->kind==0 && h->id==(u32)addr) : (h->kind!=0 && h->addr==addr)){ cs_wait_timed(1,CS_FLN); return; } } }

/* what a waiter on the GPU side sees: the word (kind 1/2: semaphore at addr) or the syncpoint (kind 3: id) with the
   releases held back for compute results applied in order. GPU work after a release already runs after the work before
   it (one GL context), and reads of compute output from guest memory go through cs_mem_need, so only CPU readers need
   the hold. Waiting for the readback instead serialized MKW's queues on the host GPU */
u32 gpu_gl_held_view(int kind, u64 addr, u32 cur){
  if(!t_gpu_thread) return cur;
  for(int i=0;i<g_cs_nincs;i++){ CsHeld* h=&g_cs_incs[i];
    if(kind==3){ if(h->kind==0 && h->id==(u32)addr) cur+=h->n; continue; }
    if(h->addr!=addr) continue;
    if(h->kind==1){ memcpy(&cur,h->data,4); continue; }
    if(h->kind==2){ u32 pl=h->n; int sg=h->len;
      switch(h->id){ case 0: cur= sg? (u32)((i32)cur<(i32)pl? (i32)cur : (i32)pl) : (cur<pl? cur : pl); break;
        case 1: cur= sg? (u32)((i32)cur>(i32)pl? (i32)cur : (i32)pl) : (cur>pl? cur : pl); break;
        case 2: cur^=pl; break; case 3: cur&=pl; break; case 4: cur|=pl; break; case 5: cur+=pl; break;
        case 6: cur= cur>=pl? 0 : cur+1; break; case 7: cur= (cur==0 || cur>pl)? pl : cur-1; break; } } }
  return cur; }

static int cs_inflight(void){ return g_cs_fl.fence!=0; }
/* anything still owed to guest memory or held behind it: readbacks in flight, copies not sent, held releases */
static int cs_pending_any(void){ return g_cs_fl.fence!=0 || g_csq_n || g_cs_npend || g_cs_nincs; }

/* idle GPU thread with a readback that never comes back: after 4 s the held fence writes go anyway */
static void cs_stale_check(void){
  if(!g_cs_fl.fence || !g_cs_nincs || plat_ns()-g_cs_fl.t0<4000000000ull) return;
  static int nl; if(nl<sw2_cap(4)){ nl++; LOG("gpu: compute readback in flight for 4 s, letting %d held fence write(s) go\n",g_cs_nincs); }
  cs_release_held(); }

/* stall dump: what the GPU side is doing when the game stops presenting */
void gpu_gl_stall_report(void){
  if(!g_gl) return;
  u64 now=plat_ns(); const char* ph=g_gpu_phase; u64 p0=g_gpu_phase_t0;
  LOG("  gpu thread: %s", ph? ph : "idle"); if(ph) LOG(" for %.0f ms",(now-p0)/1e6);
  LOG(" | compute: %d held fence write(s), readback %s",g_cs_nincs,g_cs_fl.fence? "in flight" : "none");
  if(g_cs_fl.fence) LOG(" for %.0f ms (%d in flight)",(now-g_cs_fl.t0)/1e6,g_csq_n);
  LOG("\n");
  { char b[300]; glq_stall_report(b,sizeof b); LOG("  GL thread: %s\n",b); }
  { const char* w=g_wk_phase; u64 w0=g_wk_t0; if(g_worker==1) { LOG("  compile thread: %s",w? w : "idle"); if(w) LOG(" for %.0f ms",(now-w0)/1e6); LOG("%s\n",g_excl? " (driver exclusive mode)" : ""); } }
  wd_report(); }

/* sync path: land everything pending/in flight in guest memory now */
void cs_flush_pending(void){
  if(g_csq_n) cs_wait_timed(0,CS_FLN);
  if(!g_cs_npend){ g_cs_any=0; g_cs_lo=~0ull; g_cs_hi=0; return; }
  glMemoryBarrier(0x00000200 /* GL_BUFFER_UPDATE_BARRIER_BIT */); glBindBuffer(GL_COPY_READ_BUFFER,g_cs_stage);
  u8* tmp=cs_tmp(g_cs_stage_used); glGetBufferSubData(GL_COPY_READ_BUFFER,0,g_cs_stage_used,tmp);
  cs_apply(tmp,g_cs_pend,g_cs_npend);
  g_cs_npend=0; g_cs_stage_used=0; g_cs_any=0; g_cs_lo=~0ull; g_cs_hi=0; }

/* end of submit: pending copy goes in flight with a fence */
static void cs_kick_async(void){
  if(!g_cs_npend) return;
  static int sync=-1; if(sync<0) sync=getenv("SWR_CS_SYNC")!=NULL || !p_fs || !p_cws || !p_ds;
  if(sync){ cs_flush_pending(); return; }
  cs_complete(0);   /* land what's already done */
  if(g_csq_n==CS_FLN){ cs_wait_timed(2,1);
    if(g_csq_n==CS_FLN){ CsFl* q=&g_csq[g_csq_h]; if(p_ds) p_ds(q->fence); q->fence=0; q->n=0; g_cs_dseq=q->seq; g_csq_h=(g_csq_h+1)%CS_FLN; g_csq_n--; cs_release_upto(g_cs_dseq); } }   /* timed out: drop it */
  glMemoryBarrier(0x00000200);
  CsFl* q=&g_csq[(g_csq_h+g_csq_n)%CS_FLN];
  GLuint b=q->buf; u32 c=q->cap;
  q->buf=g_cs_stage; q->cap=g_cs_stage_cap; q->used=g_cs_stage_used; q->n=g_cs_npend; memcpy(q->p,g_cs_pend,sizeof(CsPend)*g_cs_npend);
  q->fence=p_fs(GL_SYNC_GPU_COMMANDS_COMPLETE,0); q->t0=plat_ns(); q->seq=++g_cs_kseq; g_csq_n++; glFlush();
  g_cs_stage=b; g_cs_stage_cap=c;   /* that slot's old buffer (0 = create on next use) */
  g_cs_npend=0; g_cs_stage_used=0; g_cs_any=1; }

/* parked channels: send what's pending on its way and land what the host GPU finished, without waiting */
static void cs_kick_async(void);
void gpu_gl_cs_poll(void){ if(!t_gpu_thread) return; if(g_cs_npend) cs_kick_async(); cs_complete(0); }

/* does [a,a+n) overlap compute output that hasn't landed in guest memory yet */
int gpu_gl_cs_pending(u64 a, u64 n){
  if(!t_gpu_thread || a>=g_cs_hi || a+n<=g_cs_lo) return 0;
  for(int k=0;k<g_csq_n;k++){ CsFl* q=&g_csq[(g_csq_h+k)%CS_FLN]; for(int i=0;i<q->n;i++) if(a<q->p[i].addr+q->p[i].size && q->p[i].addr<a+n) return 1; }
  for(int i=0;i<g_cs_npend;i++) if(a<g_cs_pend[i].addr+g_cs_pend[i].size && g_cs_pend[i].addr<a+n) return 1;
  return 0; }

/* gmem_ptr/gmem_read on the GPU thread hit memory a compute wrote that didn't land yet, pull it now */
void cs_mem_need(u64 a, u64 n){
  if(a>=g_cs_hi || a+n<=g_cs_lo) return;
  if(g_cs_busy || !t_gpu_thread) return;   /* GL from the GPU thread only */
  { static int st=-1; if(st<0) st=getenv("SWR_CS_STALE_READS")!=NULL; if(st && n<=64) return; }   /* test: small reads (indirect args) take what memory has */
  int hit=0; for(int k=0;k<g_csq_n && !hit;k++){ CsFl* q=&g_csq[(g_csq_h+k)%CS_FLN]; for(int i=0;i<q->n && !hit;i++) if(a<q->p[i].addr+q->p[i].size && q->p[i].addr<a+n) hit=1; }
  for(int i=0;i<g_cs_npend && !hit;i++) if(a<g_cs_pend[i].addr+g_cs_pend[i].size && g_cs_pend[i].addr<a+n) hit=1;
  if(hit){ { static int nb; if(nb<6){ nb++; extern int plat_backtrace_exe(u32*,int); u32 bt[8]; int k=plat_backtrace_exe(bt,8); char line[256]; int o=0;
        for(int i=0;i<k;i++) o+=snprintf(line+o,sizeof line-o," +%x",bt[i]); LOG("gpu: read of compute output at %llx (%llx bytes) from%s\n",(unsigned long long)a,(unsigned long long)n,line); } }
    u64 t0=plat_ns(); cs_flush_pending(); st_cs_sync++; st_cs_sync_ns+=plat_ns()-t0;
    static int nl; if(nl<sw2_cap(6)){ nl++; LOG("gpu: waited %.1f ms for compute output at %llx (%llx bytes read)\n",(plat_ns()-t0)/1e6,(unsigned long long)a,(unsigned long long)n); }
    trace_note("  (waited %.1f ms for compute output at %llx, %llx bytes)\n",(plat_ns()-t0)/1e6,(unsigned long long)a,(unsigned long long)n); } }

static int cs_pending_overlaps(u64 a, u32 n){ for(int i=0;i<g_cs_npend;i++) if(a<g_cs_pend[i].addr+g_cs_pend[i].size && g_cs_pend[i].addr<a+n) return 1; return 0; }

/* the newest overlapping result has to cover all of [a,a+n) (pending is newer than in flight) */
static int cs_gpu_src(u64 a, u64 n, GLuint* buf, u32* off){
  static int off_=-1; if(off_<0) off_=getenv("SWR_CS_NO_GPUCOPY")!=NULL; if(off_) return 0;
  if(!t_gpu_thread || g_cs_busy || a>=g_cs_hi || a+n<=g_cs_lo || n==0) return 0;
  for(int i=g_cs_npend-1;i>=0;i--){ u64 pa=g_cs_pend[i].addr, pe=pa+g_cs_pend[i].size;
    if(a<pe && pa<a+n){ if(a>=pa && a+n<=pe && g_cs_stage){ *buf=g_cs_stage; *off=g_cs_pend[i].off+(u32)(a-pa); return 1; } return 0; } }
  for(int k=g_csq_n-1;k>=0;k--){ CsFl* q=&g_csq[(g_csq_h+k)%CS_FLN];   /* newest first */
    for(int i=q->n-1;i>=0;i--){ u64 pa=q->p[i].addr, pe=pa+q->p[i].size;
      if(a<pe && pa<a+n){ if(a>=pa && a+n<=pe && q->buf){ *buf=q->buf; *off=q->p[i].off+(u32)(a-pa); return 1; } return 0; } } }
  return 0; }

/* a buffer the GPU wrote (compute or a draw) goes into the staging buffer to be read back: the data, then its write
   bits (wmb at word wbase, 0 = no bits: the whole range goes back) */
static void cs_stage_push(GLuint src, u32 size, u64 addr, GLuint wmb, u32 wbase){
  u32 mb= wmb? (size/4+31)/32*4 : 0, dsz=(size+15)&~15u;
  if(g_cs_npend>=256 || g_cs_stage_used+dsz+mb>(32u<<20)) cs_flush_pending();
  u32 need=g_cs_stage_used+dsz+mb;
  if(!g_cs_stage || need>g_cs_stage_cap){ /* growing: read back what's pending first */
    cs_flush_pending(); u32 nc=g_cs_stage_cap? g_cs_stage_cap*2 : (1u<<20); while(nc<dsz+mb) nc*=2;
    if(!g_cs_stage) glGenBuffers(1,&g_cs_stage); glBindBuffer(GL_COPY_WRITE_BUFFER,g_cs_stage); glBufferData(GL_COPY_WRITE_BUFFER,nc,NULL,GL_DYNAMIC_READ); g_cs_stage_cap=nc; }
  glBindBuffer(GL_COPY_READ_BUFFER,src); glBindBuffer(GL_COPY_WRITE_BUFFER,g_cs_stage);
  glCopyBufferSubData(GL_COPY_READ_BUFFER,GL_COPY_WRITE_BUFFER,0,g_cs_stage_used,size);
  if(wmb){ glBindBuffer(GL_COPY_READ_BUFFER,wmb); glCopyBufferSubData(GL_COPY_READ_BUFFER,GL_COPY_WRITE_BUFFER,(GLintptr)wbase*4,g_cs_stage_used+dsz,mb); }
  CsPend* q=&g_cs_pend[g_cs_npend++]; q->addr=addr; q->off=g_cs_stage_used; q->size=size; q->moff= wmb? g_cs_stage_used+dsz : 0; g_cs_any=1;
  if(addr<g_cs_lo) g_cs_lo=addr; if(addr+size>g_cs_hi) g_cs_hi=addr+size;
  g_cs_stage_used=(g_cs_stage_used+dsz+mb+255)&~255u; }

/* format class for raw SUST.D / SULD.D: 0 f32, 1 f16, 2 unorm8, 3 unorm16 */
static int fmt_cls(GLenum f){ switch(f){ case GL_R16F: case GL_RG16F: case GL_RGBA16F: return 1;
  case GL_R8: case GL_RG8: case GL_RGBA8: case GL_SRGB8_ALPHA8: return 2; case GL_R16: case GL_RG16: case GL_RGBA16: return 3; default: return 0; } }

static GLenum img_fmt(GLenum f){ switch(f){ case GL_SRGB8_ALPHA8: return GL_RGBA8;
  case GL_RGBA32F: case GL_RGBA16F: case GL_RG32F: case GL_RG16F: case GL_R11F_G11F_B10F: case GL_R32F: case GL_R16F: case GL_RGBA32UI: case GL_RGBA16UI:
  case GL_RGB10_A2UI: case GL_RGBA8UI: case GL_RG32UI: case GL_RG16UI: case GL_RG8UI: case GL_R32UI: case GL_R16UI: case GL_R8UI: case GL_RGBA32I: case GL_RGBA16I:
  case GL_RGBA8I: case GL_RG32I: case GL_RG16I: case GL_RG8I: case GL_R32I: case GL_R16I: case GL_R8I: case GL_RGBA16: case GL_RGB10_A2: case GL_RGBA8: case GL_RG16:
  case GL_RG8: case GL_R16: case GL_R8: case GL_RGBA16_SNORM: case GL_RGBA8_SNORM: case GL_RG16_SNORM: case GL_RG8_SNORM: case GL_R16_SNORM: case GL_R8_SNORM: return f;
  default: return 0; } }

typedef struct CsProg { u64 key; GLuint prog; int failed; ShaderInfo si; struct CsProg* next; GLint loc[4]; int loc_ok; char* src; int tr_dumped; } CsProg;

static CsProg* g_csp[256]; static u64 st_cs_run, st_cs_fail;
static u8 g_cs_cb_[8][0x10000]; static u32 g_cs_cbn[8];

static int cs_cb_read(u32 idx, u32 off, u32* out){ if(idx<8 && off+4<=g_cs_cbn[idx]){ memcpy(out,g_cs_cb_[idx]+off,4); return 1; } return 0; }

void gpu_gl_dispatch(GpuChannel* ch, const u32* q, u64 code){
  if(getenv("SWR_NO_COMPUTE")) return;
  extern void r3d_pending_flush(void); r3d_pending_flush();
  u32* rc=ch->rcomp; u64 prog=code+q[8];
  u32 gx=q[12]&0x7FFFFFFF, gy=q[13]&0xFFFF, gz=q[13]>>16, lx=q[18]>>16, ly=q[19]&0xFFFF, lz=q[19]>>16, shm=q[17]&0x3FFFF;
  if(!gx || !gy || !gz || !lx || !ly || !lz) return;
  /* cbufs from the QMD */
  u64 cba[8]={0};
  for(int i=0;i<8;i++) if((q[20]>>i)&1){ cba[i]=((u64)(q[30+2*i]&0xFF)<<32)|q[29+2*i]; u32 sz=(q[30+2*i]>>15)&0x1FFFF; if(!sz || sz>0x10000) sz=0x10000;
    g_cs_cbn[i]=sz; gmem_read(cba[i],g_cs_cb_[i],sz); } else g_cs_cbn[i]=0;
  /* 256 KB: a MKW compute (f0b45cf863f3f0f9) is longer than 64 KB of Ampere code, cut there its branch
     targets fell outside and 1701 instructions were left untranslated; at 128 KB its lowered code still didn't fit */
  #define CS_CODE_MAX 0x40000
  static u8 buf[CS_CODE_MAX+80];
  /* the key needs the first 16 KB; the rest is read on a miss (reading it all cost ~46 MB of copies a frame) */
  memset(buf,0,80); gmem_read(prog,buf+80,0x4000);
  u64 key=fast_hash(buf+80,0x4000,0)^((u64)lx<<40)^((u64)ly<<48)^((u64)lz<<56)^((u64)shm<<20);
  if(ch->ampere) key^=0x5A3D86A3C0FFEE22ull;
  CsProg* p=g_csp[key&255]; while(p && p->key!=key) p=p->next;
  if(!p){ gmem_read(prog+0x4000,buf+80+0x4000,CS_CODE_MAX-0x4000);
    p=calloc(1,sizeof *p); p->key=key; p->next=g_csp[key&255]; g_csp[key&255]=p;
    memset(&p->si,0,sizeof p->si); p->si.cs_local[0]=lx; p->si.cs_local[1]=ly; p->si.cs_local[2]=lz; p->si.shared_size=(int)shm;
    int (*old)(u32,u32,u32*)=g_shader_cb_read; g_shader_cb_read=cs_cb_read;
    /* Switch 2: SM86 code lowered to Maxwell first, like the 3D stages */
    int lowered=1; if(ch->ampere){ u32 ol=0, al=0; Sm86Info sinf; u8* m=sm86_lower(buf+80,CS_CODE_MAX,0,&ol,&al,&sinf);
      if(m && getenv("SWR_DUMP_AMP") && (sinf.untranslated || !strcmp(getenv("SWR_DUMP_AMP"),"all"))){   /* =all: every compute */ static int nd; if(nd<4000){ nd++; char fp[200]; plat_mkdir("captures"); plat_mkdir("captures/amp");
          snprintf(fp,sizeof fp,"captures/amp/%016llx_cs.bin",(unsigned long long)key); FILE* f=fopen(fp,"wb"); if(f){ fwrite(buf+80,1,al,f); fclose(f); }
          snprintf(fp,sizeof fp,"captures/amp/%016llx_cs.maxwell.bin",(unsigned long long)key); f=fopen(fp,"wb"); if(f){ fwrite(m,1,ol,f); fclose(f); } } }   /* what sm86 made of it */
      /* too long for the host driver: NVIDIA rejects programs past ~64k assembly instructions ("too many
         instructions") and our GLSL is ~4x the Ampere code. MKW's 16010 instruction compute took the GL thread 37 s
         to fail, a 15 s freeze in every run. SWR_CS_MAX_INSTS overrides the limit */
      { static u32 lim=0; if(!lim){ const char* e=getenv("SWR_CS_MAX_INSTS"); lim= e? (u32)atoi(e) : 12000; }
        if(m && sinf.insts>lim){ static int said; if(said<sw2_cap(4)){ said++; LOG("gpu: Ampere compute %016llx: %u instructions, too long for the host driver (limit %u): skipped\n",(unsigned long long)key,sinf.insts,lim); } free(m); m=NULL; } }
      if(!m) lowered=0; else { memset(buf+80,0,CS_CODE_MAX); memcpy(buf+80,m,ol<CS_CODE_MAX? ol : CS_CODE_MAX); free(m);
        if(sinf.tex_bank>=0 && sinf.tex_bank<8){ p->si.tex_bank_set=1; p->si.tex_bank=(u8)sinf.tex_bank; }
        static int said; if(said<sw2_cap(6) || sinf.untranslated){ if(said<sw2_cap(40)){ said++; LOG("gpu: Ampere compute %016llx: %u instructions, %u not translated\n",(unsigned long long)key,sinf.insts,sinf.untranslated); } } } }
    g_sh_ampere=ch->ampere;
    char* src= lowered? shader_to_glsl(buf,sizeof buf,ST_CS,&p->si) : NULL; g_shader_cb_read=old;
    /* a 3D texture written through an image the shader declares as a 2D array (SUST .2D_ARRAY, the layer is the z
       slice): the hardware takes it, GL drops every store when the image unit's type doesn't match the texture.
       MKW 1.0 fills its 64x36x256 volumetric fog grid that way; the grid stayed garbage (-800, inf), the
       fog pass turned it into NaN and the 3D scenes came out white. Declared image3D, same ivec3 coordinates */
    if(src) for(int i=0;i<p->si.nimg && i<8;i++) if(p->si.img[i].type==4){
      u64 tic_base=((u64)rc[0x55D]<<32)|rc[0x55E]; u32 tcb=(p->si.tex_bank_set? p->si.tex_bank : rc[0x982])&7;
      u32 cb= p->si.img[i].bindless? (p->si.img[i].bl_cbuf&7) : tcb; u32 h=0; cs_cb_read(cb,p->si.img[i].cbuf_off*4,&h);
      u32 tic[8]; gmem_read(tic_base+(h&0xFFFFF)*32ull,tic,32); Tex* t=tex_from_tic(tic);
      char pat[48]; snprintf(pat,sizeof pat,"uniform image2DArray img%d;",i); char* s=strstr(src,pat);
      { static int dbg; if(dbg<16){ dbg++; LOG("compute %016llx: image %d (2D array) -> tic %x tex %s target %x %ux%ux%u\n",(unsigned long long)key,i,h,
          t? "found" : "none",t? t->target : 0,t? t->w : 0,t? t->h : 0,t? t->d : 0); } }
      if(t && t->target==GL_TEXTURE_3D && s){ memcpy(s+8,"image3D     ",12);   /* same length, keeps the string in place */
        static int said; if(said<8){ said++; LOG("compute %016llx: image %d is a 2D array over a 3D texture (%ux%ux%u), declared image3D\n",(unsigned long long)key,i,t->w,t->h,t->d); } } }
    if(!src || p->si.unsupported){ p->failed=1; LOG("compute %016llx: not translated (%d instructions, first %s)\n",(unsigned long long)key,p->si.unsupported,p->si.first_unsupported);
      extern int g_opt_diag; static int nd; if(g_opt_diag && nd<16){ nd++; char fp[200]; plat_mkdir("captures"); plat_mkdir("captures/failed");
        snprintf(fp,sizeof fp,"captures/failed/%016llx_cs.bin",(unsigned long long)key); FILE* f=fopen(fp,"wb"); if(f){ fwrite(buf+80,1,0x10000,f); fclose(f); } } }
    else { char err[4096]; GLuint sh=compile(GL_COMPUTE_SHADER,src,err,sizeof err);
      if(!sh){ p->failed=1; LOG("compute %016llx failed to compile: %.600s\n",(unsigned long long)key,err); }
      else { GLuint pr=glCreateProgram(); glAttachShader(pr,sh); glLinkProgram(pr); GLint ok=0; glGetProgramiv(pr,GL_LINK_STATUS,&ok); glDeleteShader(sh);
        if(!ok){ glGetProgramInfoLog(pr,sizeof err,NULL,err); LOG("compute %016llx failed to link: %.600s\n",(unsigned long long)key,err); glDeleteProgram(pr); p->failed=1; }
        else { p->prog=pr; static int said; if(said<sw2_cap(12)){ said++; LOG("compute %016llx ready: block %ux%ux%u, %d global buffers, %d textures\n",(unsigned long long)key,lx,ly,lz,p->si.ngbuf,p->si.ntex); } } } }
    if(src && getenv("SWR_DUMP_SHADERS")){ char pth[128]; snprintf(pth,sizeof pth,"shader_%016llx_cs.glsl",(unsigned long long)key); FILE* f=fopen(pth,"w"); if(f){ fputs(src,f); fclose(f); } }
    p->src=src; }   /* kept for F5 traces */
  if(p->failed || !p->prog){ st_cs_fail++; trace_note("COMPUTE %016llx (not translated: %d, first %s)\n",(unsigned long long)key,p->si.unsupported,p->si.first_unsupported); return; }
  glUseProgram(p->prog); g_pipe_on=0;
  /* cbufs at binding 36+i (translator's 'other stages' range) */
  static GLuint ub[8]; for(int i=0;i<8;i++) if(p->si.cbuf_mask&(1u<<i)){ if(!ub[i]) glGenBuffers(1,&ub[i]);
      glBindBuffer(GL_UNIFORM_BUFFER,ub[i]); static u8 z[0x10000]; u32 n=g_cs_cbn[i];
      if(n){ if(n<0x10000){ memcpy(z,g_cs_cb_[i],n); memset(z+n,0,0x10000-n); glBufferData(GL_UNIFORM_BUFFER,0x10000,z,GL_STREAM_DRAW); } else glBufferData(GL_UNIFORM_BUFFER,0x10000,g_cs_cb_[i],GL_STREAM_DRAW); }
      else { memset(z,0,sizeof z); glBufferData(GL_UNIFORM_BUFFER,0x10000,z,GL_STREAM_DRAW); }
      glBindBufferRange(GL_UNIFORM_BUFFER,36+i,ub[i],0,0x10000); g_ub_bound[36+i]=0; }
  /* texture handles come from bank tex_cb_index (0x982) of the compute engine */
  GLint tcls[32]={0}, tbpp[32]={0}, icls[8]={0}, ibpp[8]={0}; int nimg_ok=0;
  { u64 tic_base=((u64)rc[0x55D]<<32)|rc[0x55E], tsc_base=((u64)rc[0x557]<<32)|rc[0x558]; u32 tcb=(p->si.tex_bank_set? p->si.tex_bank : rc[0x982])&7;
    for(int i=0;i<p->si.ntex && i<32;i++){ ShTex* st=&p->si.tex[i]; u32 cb= st->bindless? (st->bl_cbuf&7) : tcb; u32 h=0; cs_cb_read(cb,st->cbuf_off*4,&h);
      if(st->bindless && st->bl2){ u32 h2=0; cs_cb_read(st->bl2_cbuf&7,st->bl2_off*4,&h2); h|=h2; }
      u32 tic[8], tsc[8]; gmem_read(tic_base+(h&0xFFFFF)*32ull,tic,32); gmem_read(tsc_base+((h>>20)&0xFFF)*32ull,tsc,32);
      GLenum want= st->type==2? GL_TEXTURE_3D : st->type==3? (st->array? GL_TEXTURE_CUBE_MAP_ARRAY : GL_TEXTURE_CUBE_MAP) : st->array? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;
      Tex* t=tex_from_tic(tic); GLuint id= t && t->target==want? t->id : (t && tex_view(t,want)? t->view : null_tex(want));
      glActiveTexture(GL_TEXTURE0+64+i); glBindTexture(want,id); glBindSampler(64+i,sampler_get(tsc));
      tcls[i]= t? fmt_cls(t->f.ifmt) : 0; tbpp[i]= t && t->f.bpp? (int)t->f.bpp : 4;
      if(g_tr_on){ trace_note("  tex %d: %llx %ux%ux%u target %x%s\n",i,(unsigned long long)(t? t->gva : 0),t? t->w : 0,t? t->h : 0,t? t->d : 0,t? t->target : 0,t && t->is_rt? " RT" : ""); if(t) tr_tex_id(t); } }
    /* SUST targets */
    for(int i=0;i<p->si.nimg && i<8;i++){ u32 cb= p->si.img[i].bindless? (p->si.img[i].bl_cbuf&7) : tcb; u32 h=0; cs_cb_read(cb,p->si.img[i].cbuf_off*4,&h);
      u32 tic[8]; gmem_read(tic_base+(h&0xFFFFF)*32ull,tic,32);
      /* SUST .3D through a TIC that describes the memory as a 2D array (its layers are the z slices): use the 3D
         texture there. A 2D array texture bound to an image3D unit drops every store, and the 3D texture is the one
         the samplers read (MKW's volumetric fog grids, aff4a39505b5e673) */
      if(p->si.img[i].type==5 && ((tic[4]>>23)&0xF)==5) tic[4]=(tic[4]&~(0xFu<<23))|(2u<<23);
      Tex* t=tex_from_tic(tic); GLenum f= t? img_fmt(t->f.ifmt) : 0;
      icls[i]= t? fmt_cls(t->f.ifmt) : 0; ibpp[i]= t && t->f.bpp? (int)t->f.bpp : 4;
      { u64 ta=((u64)(tic[2]&0xFFFF)<<32)|tic[1]; if(watch_gva(ta)) LOG("watch %llx: compute %016llx writes it as image %d (tic %x, format %x, %s) (frame %u)\n",(unsigned long long)ta,(unsigned long long)p->key,i,h,tic[0]&0x7FFFF,t? "texture found" : "NO texture",g_frame); }
      if(t && f){ int ty=p->si.img[i].type; GLint lv=img_level(tic,t); int at=p->si.img[i].pad&1;   /* SURED: atomics on r32ui, read and written */
        glBindImageTexture(i,t->id,lv,(ty==2||ty==4||ty==5)? GL_TRUE : GL_FALSE,0,at? GL_READ_WRITE : GL_WRITE_ONLY,at? GL_R32UI : f); t->gpu_written=1; t->zero3d=0; t->wgen++; nimg_ok++;
        trace_note("  image %d: %llx %ux%ux%u target %x, shader type %d, mip %d of %u (tic levels %u..%u)%s\n",i,(unsigned long long)t->gva,t->w,t->h,t->d,t->target,ty,lv,t->levels,tic[7]&0xF,(tic[7]>>4)&0xF,t->is_rt? " RT" : "");
        static int said; if(said<sw2_cap(8)){ said++; LOG("compute: writes image %ux%ux%u at %llx (format %x, mip %d)\n",t->w,t->h,t->d,(unsigned long long)t->gva,f,lv); } }
      else { trace_note("  image %d: NOT BOUND (tic %x, %s, tic format %x)\n",i,h,t? "no image format for it" : "no texture",tic[0]&0x7FFFF);
        glBindImageTexture(i,0,0,GL_FALSE,0,GL_WRITE_ONLY,GL_R32F); static int said; if(said<sw2_cap(4)){ said++; LOG("compute %016llx: image %d without a usable texture (tic %x from c%u[%x]%s)\n",(unsigned long long)p->key,i,h,cb,p->si.img[i].cbuf_off*4,p->si.img[i].bindless? ", bindless" : "");
          if(said==1){ u32 v[24]={0}; for(int k=0;k<24;k++) cs_cb_read(cb,k*4+0x20,&v[k]); LOG("  c%u[20..7c]:",cb); for(int k=0;k<24;k++) LOG(" %x",v[k]); LOG("\n"); } } } }
    { /* cache the uniform locations per program, a glGet per dispatch stalls mesa_glthread */
      if(!p->loc_ok){ p->loc[0]=glGetUniformLocation(p->prog,"swr_ic"); p->loc[1]=glGetUniformLocation(p->prog,"swr_ib"); p->loc[2]=glGetUniformLocation(p->prog,"swr_tc"); p->loc[3]=glGetUniformLocation(p->prog,"swr_tb"); p->loc_ok=1; }
      if(p->loc[0]>=0) glUniform1iv(p->loc[0],8,icls); if(p->loc[1]>=0) glUniform1iv(p->loc[1],8,ibpp);
      if(p->loc[2]>=0) glUniform1iv(p->loc[2],32,tcls); if(p->loc[3]>=0) glUniform1iv(p->loc[3],32,tbpp); }
    mb_reset(); }
  wd_pass(NULL,key); gp_compute_begin(key,gx*gy*gz,0);   /* the buffer uploads below count for the compute too */
  /* ssbos */
  static GLuint sb[16]; u64 sa[16]; u32 ss[16];
  for(int i=0;i<p->si.ngbuf && i<16;i++){ if(!sb[i]) glGenBuffers(1,&sb[i]);
    u32 v[3]={0,0,0}; cs_cb_read(0,p->si.gbuf_cbuf_off[i],&v[0]); cs_cb_read(0,p->si.gbuf_cbuf_off[i]+4,&v[1]); cs_cb_read(0,p->si.gbuf_cbuf_off[i]+8,&v[2]);
    sa[i]=((u64)v[1]<<32)|v[0]; ss[i]=v[2];
    if(ch->ampere){ static int nl; if(nl<sw2_cap(24)){ nl++; u32 v3=0; cs_cb_read(0,p->si.gbuf_cbuf_off[i]+12,&v3); LOG("compute %016llx: buffer %d from c0[%x] = %llx, next words %x %x\n",(unsigned long long)key,i,p->si.gbuf_cbuf_off[i],(unsigned long long)sa[i],v[2],v3); } }
    if(!ss[i]) ss[i]=16; if(ss[i]>(16u<<20)) ss[i]=16u<<20; ss[i]=(ss[i]+3)&~3u;
    { GLuint gsb; u32 gso; if(sa[i] && g_cs_any && cs_gpu_src(sa[i],ss[i],&gsb,&gso)){
        glBindBuffer(GL_COPY_WRITE_BUFFER,sb[i]); glBufferData(GL_COPY_WRITE_BUFFER,ss[i],NULL,GL_DYNAMIC_COPY);
        glBindBuffer(GL_COPY_READ_BUFFER,gsb); glCopyBufferSubData(GL_COPY_READ_BUFFER,GL_COPY_WRITE_BUFFER,gso,0,ss[i]);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER,i,sb[i]); st_cs_gpucopy++; continue; } }
    if(sa[i] && cs_pending_overlaps(sa[i],ss[i])){ u64 t0=plat_ns(); cs_flush_pending(); st_cs_sync++; st_cs_sync_ns+=plat_ns()-t0; }
    static u8* tmp; static u32 cap; if(cap<ss[i]){ tmp=realloc(tmp,ss[i]); cap=ss[i]; }
    if(sa[i]) gmem_read(sa[i],tmp,ss[i]); else memset(tmp,0,ss[i]);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,sb[i]); glBufferData(GL_SHADER_STORAGE_BUFFER,ss[i],tmp,GL_DYNAMIC_COPY); glBindBufferBase(GL_SHADER_STORAGE_BUFFER,i,sb[i]); }
  /* write bits (swr_wmark): header with each buffer's bit offset and size in words, then the bits, all zero */
  static GLuint wmb; u32 wbase[16]={0};
  if(p->si.gbuf_wmask){ static u32* wm; static u32 wcap; u32 nw=32;
    for(int i=0;i<p->si.ngbuf && i<16;i++) if((p->si.gbuf_wmask>>i)&1){ wbase[i]=nw; nw+=(ss[i]/4+31)/32; }
    if(wcap<nw){ wm=realloc(wm,nw*4); wcap=nw; }
    memset(wm,0,nw*4); for(int i=0;i<p->si.ngbuf && i<16;i++) if((p->si.gbuf_wmask>>i)&1){ wm[i]=wbase[i]; wm[16+i]=ss[i]/4; }
    if(!wmb) glGenBuffers(1,&wmb);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER,wmb); glBufferData(GL_SHADER_STORAGE_BUFFER,nw*4,wm,GL_DYNAMIC_COPY); glBindBufferBase(GL_SHADER_STORAGE_BUFFER,p->si.ngbuf,wmb); }
  if(g_gp_cs.on){ u64 mbs=0; for(int i=0;i<p->si.ngbuf && i<16;i++) mbs+=ss[i]; g_gp_p[g_gp_n-1].mb=(u32)((mbs+(1u<<19))>>20); }
  glDispatchCompute(gx,gy,gz); st_cs_run++;
  if(nimg_ok) glMemoryBarrier(0x20|0x08|0x100|0x400|0x80);   /* images -> textures, copies, fbos */
  trace_note("COMPUTE %016llx grid %ux%ux%u block %ux%ux%u buffers %d (writes %x)\n",(unsigned long long)key,gx,gy,gz,lx,ly,lz,p->si.ngbuf,p->si.gbuf_wmask);
  if(getenv("SWR_WATCH")) for(int i=0;i<p->si.ngbuf && i<16;i++) for(u64 a=sa[i]&~0xFFFull;a<sa[i]+ss[i];a+=0x1000) if(watch_gva(a)){
      LOG("watch %llx: compute %016llx buffer %d %llx size %x%s (frame %u)\n",(unsigned long long)a,(unsigned long long)p->key,i,(unsigned long long)sa[i],ss[i],((p->si.gbuf_wmask>>i)&1)? " written" : " read",g_frame); break; }
  if(g_tr_on){ for(int i=0;i<p->si.ngbuf && i<16;i++) trace_note("  buffer %d: %llx size %x%s\n",i,(unsigned long long)sa[i],ss[i],((p->si.gbuf_wmask>>i)&1)? " (written)" : "");
    if(p->src && p->tr_dumped!=g_tr_gen){ p->tr_dumped=g_tr_gen; char pth[200]; snprintf(pth,sizeof pth,"%s/cs_%016llx.glsl",g_tr_dir,(unsigned long long)key); FILE* f=fopen(pth,"w"); if(f){ fputs(p->src,f); fclose(f); } } }
  if(p->si.gbuf_wmask){ glMemoryBarrier(0x00000200|0x00002000 /* BUFFER_UPDATE | SHADER_STORAGE */);
    for(int i=0;i<p->si.ngbuf && i<16;i++) if(((p->si.gbuf_wmask>>i)&1) && sa[i]) cs_stage_push(sb[i],ss[i],sa[i],wmb,wbase[i]);
    /* fresh SSBOs for the next dispatch so it doesn't wait on these */
    for(int i=0;i<p->si.ngbuf && i<16;i++) if((p->si.gbuf_wmask>>i)&1){ glDeleteBuffers(1,&sb[i]); sb[i]=0; } }
  gp_compute_end();
  glUseProgram(0); state_dirty(); }
