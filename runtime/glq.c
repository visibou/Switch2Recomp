/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* GL call queue. the GPU thread records GL calls into a ring (pointer args get copied) and a separate thread
   replays them. the main reason is proton: every GL call goes through wine, that was like 26% of the GPU thread on
   the deck, and mesa's glthread can't help because it sits below wine. calls that return something (glGet*,
   glGen*, glMapBuffer...) sync with the GL thread. glFenceSync is virtual, the GL thread owns the real fences.
   SWR_NO_GLQ=1 turns it off */
#include "glq_int.h"
#include "glq.h"
#if defined(__x86_64__) || defined(__i386__)
#define cpu_relax() __builtin_ia32_pause()
#else
#define cpu_relax() __asm__ volatile("yield")
#endif
__thread int t_glq_direct;
extern void* backend_gl_proc(const char*); extern int backend_gl_create(void); extern void backend_gl_swap(void); extern int backend_gl_worker_current(void);
extern int backend_glq_ok(void);

#define Q_CAP (64u<<20)
#define Q_BIG (8u<<20)
typedef struct { u32 size, big; void (*ex)(void*); void* data; } QHdr;   /* record size in the ring, bytes */

static u8* g_q; static volatile u64 g_qh, g_qt;   /* head = producer, tail = consumer */
static u64 g_qh_pend; static int g_active; static volatile int g_ready;
static volatile int g_sleeping; static PMutex g_qm; static PCond g_qc;
static volatile u32 g_lock;
static volatile u64 g_swaps_sent, g_swaps_done;

static void q_wrap_ex(void* p){ (void)p; }

static void q_lock(void){ while(__atomic_exchange_n(&g_lock,1,__ATOMIC_ACQUIRE)){ while(__atomic_load_n(&g_lock,__ATOMIC_RELAXED)) ; } }

static void q_unlock(void){ __atomic_store_n(&g_lock,0,__ATOMIC_RELEASE); }

int glq_active(void){ return g_active; }

/* ---------- producer */
static u8* q_reserve(u32 total){
  for(int spins=0;;spins++){
    u64 h=g_qh, t=__atomic_load_n(&g_qt,__ATOMIC_ACQUIRE); u64 pos=h%Q_CAP;
    u64 need= (pos+total>Q_CAP)? (Q_CAP-pos)+total : total;
    if(h+need-t<=Q_CAP){
      if(pos+total>Q_CAP){ QHdr* w=(QHdr*)(g_q+pos); w->size=(u32)(Q_CAP-pos); w->big=0; w->ex=q_wrap_ex; w->data=NULL; h+=Q_CAP-pos; pos=0; }
      g_qh_pend=h+total; return g_q+pos; }
    if(spins>64) plat_sleep_ns(200000); else cpu_relax(); }   /* queue full: the GL thread needs the CPU to drain it */
}

void* glq_alloc(size_t sz, size_t extra, void (*ex)(void*)){
  q_lock();
  size_t pay=q_al16(sz)+q_al16(extra);
  static u8* sink; static size_t sink_n;   /* where a dropped command writes, set aside while memory is still there */
  if(pay>Q_BIG && !sink){ sink=malloc(64u<<20); sink_n= sink? 64u<<20 : 0; }
  if(pay>Q_BIG){ QHdr* hd=(QHdr*)q_reserve((u32)sizeof(QHdr)); u8* d=malloc(pay);
    /* out of memory (MKW on a PC near its commit limit crashed here): wait for the GL thread to free the
       uploads still queued and try again, then drop this one command instead of writing through NULL */
    for(int k=0;!d && k<100;k++){ plat_sleep_ns(20000000); d=malloc(pay); }
    if(!d){ static int said; if(said<4){ said++; WARN("glq: no memory for a %llu KB GL command, it is dropped\n",(unsigned long long)(pay>>10)); }
      if(pay>sink_n){ u8* b=malloc(pay); if(b){ free(sink); sink=b; sink_n=pay; } }
      if(!sink || pay>sink_n){ WARN("glq: out of memory (%llu KB)\n",(unsigned long long)(pay>>10)); abort(); }
      hd->size=sizeof(QHdr); hd->big=0; hd->ex=q_wrap_ex; hd->data=NULL; return sink; }
    hd->size=sizeof(QHdr); hd->big=1; hd->ex=ex; hd->data=d; return d; }
  u32 total=(u32)(sizeof(QHdr)+pay); QHdr* hd=(QHdr*)q_reserve(total); hd->size=total; hd->big=0; hd->ex=ex; hd->data=(u8*)(hd+1); return hd->data; }

static void q_wake(void){ if(__atomic_load_n(&g_sleeping,__ATOMIC_ACQUIRE)){ pm_lock(&g_qm); pc_signal(&g_qc); pm_unlock(&g_qm); } }

void glq_commit(void){ __atomic_store_n(&g_qh,g_qh_pend,__ATOMIC_RELEASE); q_unlock(); q_wake(); }

/* waiting for the GL thread: a short spin and then a real sleep that the GL thread ends with a signal. only
   yielding is no good on windows, SwitchToThread returns right away when nothing else is ready on that core, so
   the GPU thread burns a whole core while it waits (up to 30 ms a frame for the swap on a 2 core celeron) and
   that's the core the driver and the game need */
static PMutex g_dm; static PCond g_dc; static volatile int g_dwaiters;

static void q_done_signal(void){
  if(__atomic_load_n(&g_dwaiters,__ATOMIC_SEQ_CST)){ pm_lock(&g_dm); pc_broadcast(&g_dc); pm_unlock(&g_dm); } }

static void q_wait(int (*ready)(void*), void* a){
  for(int i=0;i<100;i++){ if(ready(a)) return; cpu_relax(); }
  pm_lock(&g_dm); __atomic_add_fetch(&g_dwaiters,1,__ATOMIC_SEQ_CST);
  while(!ready(a)) pc_wait(&g_dc,&g_dm,2000000);   /* 2 ms cap, in case a signal is ever missed */
  __atomic_sub_fetch(&g_dwaiters,1,__ATOMIC_SEQ_CST); pm_unlock(&g_dm); }

typedef struct { void (*ex)(void*); void* arg; volatile int done; } QSync;

static void q_sync_ex(void* p){ QSync* s=*(QSync**)p; s->ex(s->arg); __atomic_store_n(&s->done,1,__ATOMIC_SEQ_CST); q_done_signal(); }

static int q_sync_ready(void* p){ return __atomic_load_n(&((QSync*)p)->done,__ATOMIC_SEQ_CST); }

static volatile u64 g_wait_t0; static const char* volatile g_wait_name;   /* the GPU thread waiting on a sync call */

void glq_sync(void (*ex)(void*), void* arg){
  if(t_glq_direct || !g_active){ ex(arg); return; }
  QSync s={ex,arg,0}; QSync** p=(QSync**)glq_alloc(sizeof(QSync*),0,q_sync_ex); *p=&s; glq_commit();
  g_wait_t0=plat_ns(); q_wait(q_sync_ready,&s); g_wait_t0=0; g_wait_name=NULL; }

/* every call that returns something waits until the GL thread ran everything queued before it, so one in the
   middle of a frame costs the whole queue. I count them per function (time spent waiting) for the performance
   report so the expensive ones show up in user logs */
typedef struct { const char* n; u64 cnt, ns; } SyncStat;
static SyncStat g_ss[48];

void glq_sync_n(void (*ex)(void*), void* arg, const char* name){
  if(t_glq_direct || !g_active){ ex(arg); return; }
  u64 t0=plat_ns(); g_wait_name=name; glq_sync(ex,arg); u64 dt=plat_ns()-t0;
  for(int i=0;i<48;i++){ if(g_ss[i].n==name || !g_ss[i].n){ g_ss[i].n=name; g_ss[i].cnt++; g_ss[i].ns+=dt; break; } } }

/* "glGetUniformLocation 12x 3.1ms, ..." for the worst ones since the last call, then reset */
void glq_sync_report(char* out, size_t cap, u32 frames){
  out[0]=0; if(!g_active || !frames) return;
  size_t o=0;
  for(int k=0;k<4;k++){ int b=-1; for(int i=0;i<48 && g_ss[i].n;i++) if(g_ss[i].cnt && (b<0 || g_ss[i].ns>g_ss[b].ns)) b=i;
    if(b<0) break;
    int w=snprintf(out+o,cap-o,"%s%s %.1f/frame %.2f ms",o? ", " : "",g_ss[b].n+2,(double)g_ss[b].cnt/frames,(double)g_ss[b].ns/1e6/frames);
    if(w<0 || (size_t)w>=cap-o) break; o+=(size_t)w; g_ss[b].cnt=0; g_ss[b].ns=0; }
  for(int i=0;i<48 && g_ss[i].n;i++){ g_ss[i].cnt=0; g_ss[i].ns=0; } }

/* glGen*: object names don't need the GL thread's answer right away. I grab 256 at a time, so making textures and
   buffers mid frame doesn't drain the queue every time */
typedef void (APIENTRY *T_gen_)(GLsizei, GLuint*);
extern T_gen_ r_glGenTextures, r_glGenBuffers, r_glGenSamplers, r_glGenFramebuffers, r_glGenVertexArrays;
typedef struct { T_gen_* fn; GLuint ids[2048]; int n; } NamePool;   /* 2048: MKW makes ~280 buffers a frame */
static NamePool g_np[5]={{&r_glGenTextures},{&r_glGenBuffers},{&r_glGenSamplers},{&r_glGenFramebuffers},{&r_glGenVertexArrays}};

static void e_fill(void* p){ NamePool* np=p; (*np->fn)(2048,np->ids); np->n=2048; }

static void q_gen(int k, GLsizei n, GLuint* out){
  NamePool* np=&g_np[k];
  if(t_glq_direct || !g_active){ (*np->fn)(n,out); return; }
  static volatile u32 lk; while(__atomic_exchange_n(&lk,1,__ATOMIC_ACQUIRE)) plat_yield();
  for(GLsizei i=0;i<n;i++){ if(!np->n) glq_sync_n(e_fill,np, k==0? "glGenTextures" : k==1? "glGenBuffers" : k==2? "glGenSamplers" : k==3? "glGenFramebuffers" : "glGenVertexArrays");
    out[i]=np->ids[--np->n]; }
  __atomic_store_n(&lk,0,__ATOMIC_RELEASE); }

void APIENTRY q_glGenTextures(GLsizei n, GLuint* p){ q_gen(0,n,p); }
void APIENTRY q_glGenBuffers(GLsizei n, GLuint* p){ q_gen(1,n,p); }
void APIENTRY q_glGenSamplers(GLsizei n, GLuint* p){ q_gen(2,n,p); }
void APIENTRY q_glGenFramebuffers(GLsizei n, GLuint* p){ q_gen(3,n,p); }
void APIENTRY q_glGenVertexArrays(GLsizei n, GLuint* p){ q_gen(4,n,p); }

/* ---------- unpack state, needed to know how many bytes to copy */
static GLint g_un_align=4, g_un_row=0, g_un_imgh=0, g_un_skp=0, g_un_skr=0, g_un_ski=0; static GLuint g_un_pbo;

int q_unpack_pbo(void){ return g_un_pbo!=0; }

static size_t q_px(GLenum f, GLenum t){
  int c=4;
  switch(f){ case GL_RED: case GL_RED_INTEGER: case GL_DEPTH_COMPONENT: case GL_STENCIL_INDEX: case GL_GREEN: case GL_BLUE: c=1; break;
    case GL_RG: case GL_RG_INTEGER: c=2; break; case GL_RGB: case GL_BGR: case GL_RGB_INTEGER: case GL_BGR_INTEGER: c=3; break;
    case GL_DEPTH_STENCIL: c=1; break; default: c=4; }
  switch(t){ case GL_UNSIGNED_BYTE: case GL_BYTE: return c; case GL_UNSIGNED_SHORT: case GL_SHORT: case GL_HALF_FLOAT: return 2*c;
    case GL_UNSIGNED_INT: case GL_INT: case GL_FLOAT: return 4*c;
    case GL_UNSIGNED_SHORT_5_6_5: case GL_UNSIGNED_SHORT_5_6_5_REV: case GL_UNSIGNED_SHORT_4_4_4_4: case GL_UNSIGNED_SHORT_4_4_4_4_REV:
    case GL_UNSIGNED_SHORT_5_5_5_1: case GL_UNSIGNED_SHORT_1_5_5_5_REV: return 2;
    case GL_UNSIGNED_BYTE_3_3_2: case GL_UNSIGNED_BYTE_2_3_3_REV: return 1;
    case GL_FLOAT_32_UNSIGNED_INT_24_8_REV: return 8;
    default: return 4; } }   /* 8_8_8_8, 10_10_10_2, 2_10_10_10_REV, 24_8, 10F_11F_11F_REV, 5_9_9_9_REV */

size_t q_texel_size(GLenum f, GLenum t){ return q_px(f,t); }

size_t q_img_size(GLsizei w, GLsizei h, GLsizei d, GLenum f, GLenum t){
  if(w<=0 || h<=0 || d<=0) return 0;
  size_t px=q_px(f,t), rl=(size_t)(g_un_row>0? g_un_row : w), al=(size_t)(g_un_align>0? g_un_align : 1);
  size_t row=(rl*px+al-1)/al*al, ih=(size_t)(g_un_imgh>0? g_un_imgh : h);
  return ((size_t)g_un_ski*ih + (size_t)g_un_skr)*row + (size_t)g_un_skp*px + ((size_t)(d-1)*ih + (size_t)(h-1))*row + (size_t)w*px; }

/* ---------- hand-written wrappers (glq_gen.c has the r_ pointers) */
typedef void (APIENTRY *T_ps)(GLenum,GLint); extern T_ps r_glPixelStorei;
typedef void (APIENTRY *T_bb)(GLenum,GLuint); extern T_bb r_glBindBuffer;
typedef GLsync (APIENTRY *T_fs)(GLenum,GLbitfield); extern T_fs r_glFenceSync;
typedef GLenum (APIENTRY *T_cws)(GLsync,GLbitfield,GLuint64); extern T_cws r_glClientWaitSync;
typedef void (APIENTRY *T_ds)(GLsync); extern T_ds r_glDeleteSync;
typedef struct { GLenum a; GLint b; } A2;

static void e_ps(void* p){ A2* a=p; r_glPixelStorei(a->a,a->b); }

void APIENTRY q_glPixelStorei(GLenum pname, GLint param){
  if(t_glq_direct){ r_glPixelStorei(pname,param); return; }
  switch(pname){ case GL_UNPACK_ALIGNMENT: g_un_align=param; break; case GL_UNPACK_ROW_LENGTH: g_un_row=param; break; case GL_UNPACK_IMAGE_HEIGHT: g_un_imgh=param; break;
    case GL_UNPACK_SKIP_PIXELS: g_un_skp=param; break; case GL_UNPACK_SKIP_ROWS: g_un_skr=param; break; case GL_UNPACK_SKIP_IMAGES: g_un_ski=param; break; }
  A2* a=glq_alloc(sizeof *a,0,e_ps); a->a=pname; a->b=param; glq_commit(); }

static void e_bb(void* p){ A2* a=p; r_glBindBuffer(a->a,(GLuint)a->b); }

static GLuint g_pk_pbo;   /* pack buffer bound: glReadPixels / glGetTexImage write into it and don't need to wait */

void APIENTRY q_glBindBuffer(GLenum target, GLuint buffer){
  if(t_glq_direct){ r_glBindBuffer(target,buffer); return; }
  if(target==GL_PIXEL_UNPACK_BUFFER) g_un_pbo=buffer;
  if(target==GL_PIXEL_PACK_BUFFER) g_pk_pbo=buffer;
  A2* a=glq_alloc(sizeof *a,0,e_bb); a->a=target; a->b=(GLint)buffer; glq_commit(); }

/* ---------- shadowed state the runtime saves and restores scissor, sRGB and the color mask around its own blits
   (mip chains, copies). asking the driver would drain the queue up to 3 times per blit, so I keep a copy of what
   was last set on this side and only ask GL when I don't know it yet */
enum { SS_SCISSOR, SS_SRGB, SS_N };
static int g_ss_known[SS_N], g_ss_on[SS_N], g_cm_known; static GLint g_cm[4];

static int ss_idx(GLenum c){ return c==GL_SCISSOR_TEST? SS_SCISSOR : c==GL_FRAMEBUFFER_SRGB? SS_SRGB : -1; }

typedef void (APIENTRY *T_cap)(GLenum); typedef void (APIENTRY *T_capi)(GLenum,GLuint);
typedef void (APIENTRY *T_cmi)(GLuint,GLboolean,GLboolean,GLboolean,GLboolean); typedef void (APIENTRY *T_cm)(GLboolean,GLboolean,GLboolean,GLboolean);
typedef GLboolean (APIENTRY *T_ise)(GLenum); typedef void (APIENTRY *T_gi)(GLenum,GLint*);
extern T_cap r_glEnable, r_glDisable; extern T_capi r_glEnablei, r_glDisablei; extern T_cmi r_glColorMaski; extern T_cm r_glColorMask;
extern T_ise r_glIsEnabled; extern T_gi r_glGetIntegerv;

typedef struct { GLenum c; GLuint i; GLboolean m[4]; } ACap;
static void e_en(void* p){ ACap* a=p; r_glEnable(a->c); }
static void e_dis(void* p){ ACap* a=p; r_glDisable(a->c); }
static void e_eni(void* p){ ACap* a=p; r_glEnablei(a->c,a->i); }
static void e_disi(void* p){ ACap* a=p; r_glDisablei(a->c,a->i); }
static void e_cmi(void* p){ ACap* a=p; r_glColorMaski(a->i,a->m[0],a->m[1],a->m[2],a->m[3]); }
static void e_cm(void* p){ ACap* a=p; r_glColorMask(a->m[0],a->m[1],a->m[2],a->m[3]); }

static void q_cap(void (*ex)(void*), GLenum c, GLuint i, int on, int indexed){
  int k=ss_idx(c); if(k>=0 && (!indexed || i==0)){ g_ss_known[k]=1; g_ss_on[k]=on; }
  ACap* a=glq_alloc(sizeof *a,0,ex); a->c=c; a->i=i; glq_commit(); }

void APIENTRY q_glEnable(GLenum c){ if(t_glq_direct){ r_glEnable(c); return; } q_cap(e_en,c,0,1,0); }
void APIENTRY q_glDisable(GLenum c){ if(t_glq_direct){ r_glDisable(c); return; } q_cap(e_dis,c,0,0,0); }
void APIENTRY q_glEnablei(GLenum c, GLuint i){ if(t_glq_direct){ r_glEnablei(c,i); return; } q_cap(e_eni,c,i,1,1); }
void APIENTRY q_glDisablei(GLenum c, GLuint i){ if(t_glq_direct){ r_glDisablei(c,i); return; } q_cap(e_disi,c,i,0,1); }

void APIENTRY q_glColorMaski(GLuint i, GLboolean r, GLboolean g, GLboolean b, GLboolean al){
  if(t_glq_direct){ r_glColorMaski(i,r,g,b,al); return; }
  if(i==0){ g_cm_known=1; g_cm[0]=r; g_cm[1]=g; g_cm[2]=b; g_cm[3]=al; }
  ACap* a=glq_alloc(sizeof *a,0,e_cmi); a->i=i; a->m[0]=r; a->m[1]=g; a->m[2]=b; a->m[3]=al; glq_commit(); }

void APIENTRY q_glColorMask(GLboolean r, GLboolean g, GLboolean b, GLboolean al){
  if(t_glq_direct){ r_glColorMask(r,g,b,al); return; }
  g_cm_known=1; g_cm[0]=r; g_cm[1]=g; g_cm[2]=b; g_cm[3]=al;
  ACap* a=glq_alloc(sizeof *a,0,e_cm); a->m[0]=r; a->m[1]=g; a->m[2]=b; a->m[3]=al; glq_commit(); }

typedef struct { GLenum c; GLboolean r; GLint* d; } AIs;
static void e_ise(void* p){ AIs* a=p; a->r=r_glIsEnabled(a->c); }
static void e_gi(void* p){ AIs* a=p; r_glGetIntegerv(a->c,a->d); }

GLboolean APIENTRY q_glIsEnabled(GLenum c){
  if(t_glq_direct) return r_glIsEnabled(c);
  int k=ss_idx(c); if(k>=0 && g_ss_known[k]) return (GLboolean)g_ss_on[k];
  AIs a={c,0,NULL}; glq_sync_n(e_ise,&a,"glIsEnabled");
  if(k>=0){ g_ss_known[k]=1; g_ss_on[k]=a.r!=0; }
  return a.r; }

void APIENTRY q_glGetIntegerv(GLenum pname, GLint* d){
  if(t_glq_direct){ r_glGetIntegerv(pname,d); return; }
  if(pname==GL_COLOR_WRITEMASK && g_cm_known){ memcpy(d,g_cm,sizeof g_cm); return; }
  AIs a={pname,0,d}; glq_sync_n(e_gi,&a,"glGetIntegerv"); }

/* ---------- readbacks: into a pack buffer they're ordinary queued commands, only client memory waits */
typedef void (APIENTRY *T_rp)(GLint,GLint,GLsizei,GLsizei,GLenum,GLenum,void*); typedef void (APIENTRY *T_gti)(GLenum,GLint,GLenum,GLenum,void*);
extern T_rp r_glReadPixels; extern T_gti r_glGetTexImage;
typedef struct { GLint x, y; GLsizei w, h; GLenum f, t, tg; GLint l; void* d; } ARd;
static void e_rp(void* p){ ARd* a=p; r_glReadPixels(a->x,a->y,a->w,a->h,a->f,a->t,a->d); }
static void e_gti(void* p){ ARd* a=p; r_glGetTexImage(a->tg,a->l,a->f,a->t,a->d); }

void APIENTRY q_glReadPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum f, GLenum t, void* d){
  if(t_glq_direct){ r_glReadPixels(x,y,w,h,f,t,d); return; }
  ARd a={x,y,w,h,f,t,0,0,d};
  if(g_pk_pbo){ ARd* q=glq_alloc(sizeof *q,0,e_rp); *q=a; glq_commit(); return; }
  glq_sync_n(e_rp,&a,"glReadPixels"); }

void APIENTRY q_glGetTexImage(GLenum tg, GLint l, GLenum f, GLenum t, void* d){
  if(t_glq_direct){ r_glGetTexImage(tg,l,f,t,d); return; }
  ARd a={0,0,0,0,f,t,tg,l,d};
  if(g_pk_pbo){ ARd* q=glq_alloc(sizeof *q,0,e_gti); *q=a; glq_commit(); return; }
  glq_sync_n(e_gti,&a,"glGetTexImage"); }

/* virtual fences */
typedef struct VF { GLsync real; volatile int sig; int polled; struct VF* next; } VF;

static VF* g_vf_list;   /* GL thread side: fences still pending */
typedef struct { VF* v; GLenum c; GLbitfield f; } AF;

static void e_fs(void* p){ AF* a=p; a->v->real=r_glFenceSync(a->c,a->f); a->v->next=g_vf_list; g_vf_list=a->v; }

GLsync APIENTRY q_glFenceSync(GLenum c, GLbitfield f){
  VF* v=calloc(1,sizeof *v);
  if(t_glq_direct){ v->real=r_glFenceSync(c,f); return (GLsync)v; }
  AF* a=glq_alloc(sizeof *a,0,e_fs); a->v=v; a->c=c; a->f=f; glq_commit(); return (GLsync)v; }

typedef struct { VF* v; GLbitfield f; GLuint64 t; GLenum r; } AW;

static void e_cws(void* p){ AW* a=p; if(a->v->sig){ a->r=GL_ALREADY_SIGNALED; return; } a->r= a->v->real? r_glClientWaitSync(a->v->real,a->f,a->t) : GL_WAIT_FAILED;
  if(a->r==GL_ALREADY_SIGNALED || a->r==GL_CONDITION_SATISFIED) __atomic_store_n(&a->v->sig,1,__ATOMIC_RELEASE); }

GLenum APIENTRY q_glClientWaitSync(GLsync s, GLbitfield f, GLuint64 t){
  VF* v=(VF*)s; if(!v) return GL_WAIT_FAILED;
  if(__atomic_load_n(&v->sig,__ATOMIC_ACQUIRE)) return GL_ALREADY_SIGNALED;
  AW a={v,f,t,0};
  if(t_glq_direct){ e_cws(&a); return a.r; }
  if(t==0) return GL_TIMEOUT_EXPIRED;   /* GL thread polls it, don't block here */
  /* blocking waits: time per caller, logged every 10 s (who makes the GPU thread wait for the host GPU) */
  static void* who[8]; static u64 ns[8], cnt[8], last; void* ra=__builtin_return_address(0); u64 t0=plat_ns();
  glq_sync(e_cws,&a);
  { u64 dt=plat_ns()-t0; int k=0; for(;k<8;k++) if(who[k]==ra || !who[k]) break; if(k<8){ who[k]=ra; ns[k]+=dt; cnt[k]++; }
    if(t0-last>10000000000ull){ last=t0; char m[64]; for(k=0;k<8 && who[k];k++){ u64 off=0; if(!plat_addr_module((u64)(uintptr_t)who[k],m,sizeof m,&off)) snprintf(m,sizeof m,"?");
        LOG("glq: blocking fence waits from %s+%llx: %llu, %.0f ms\n",m,(unsigned long long)off,(unsigned long long)cnt[k],ns[k]/1e6); ns[k]=0; cnt[k]=0; } } }
  return a.r; }

static void e_ds(void* p){ VF* v=*(VF**)p; for(VF** pp=&g_vf_list;*pp;pp=&(*pp)->next) if(*pp==v){ *pp=v->next; break; } if(v->real) r_glDeleteSync(v->real); free(v); }

void APIENTRY q_glDeleteSync(GLsync s){
  VF* v=(VF*)s; if(!v) return;
  if(t_glq_direct){ if(v->real) r_glDeleteSync(v->real); free(v); return; }
  VF** p=glq_alloc(sizeof *p,0,e_ds); *p=v; glq_commit(); }

/* the first poll of a fence flushes (GL_SYNC_FLUSH_COMMANDS_BIT): a fence at the end of a frame, with nothing sent
   after it, otherwise sat in the driver's command buffer until its own ~100 ms idle flush. MKW's last
   compute readback of every frame landed exactly 100 ms after its fence, and the game's GPU wait with it */
static void q_poll_fences(void){
  for(VF* v=g_vf_list;v;v=v->next) if(!v->sig && v->real){ GLenum r=r_glClientWaitSync(v->real,v->polled? 0 : GL_SYNC_FLUSH_COMMANDS_BIT,0); v->polled=1;
    if(r==GL_ALREADY_SIGNALED || r==GL_CONDITION_SATISFIED) __atomic_store_n(&v->sig,1,__ATOMIC_RELEASE); } }

/* ---------- GPU watchdog
   a real fence at the start of every pass (render target change, compute, present). they stay out of the poll list
   so they cost nothing per frame: the GL thread checks them only when the stall dump asks. that tells how far the
   GPU got when the image stops, without a blocking driver call */
#define WD_N 64
static GLsync g_wdf[WD_N]; static volatile u32 g_wdid[WD_N]; static volatile u8 g_wds[WD_N]; static volatile int g_wd_probe;

static void e_wdm(void* p){ u32 id=*(u32*)p, k=id%WD_N;
  if(g_wdf[k]) r_glDeleteSync(g_wdf[k]);
  g_wdf[k]=r_glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE,0); g_wds[k]=0; __atomic_store_n(&g_wdid[k],id,__ATOMIC_RELEASE); }

void glq_wd_mark(u32 id){ if(!g_active || t_glq_direct) return; u32* p=glq_alloc(sizeof(u32),0,e_wdm); *p=id; glq_commit(); }

static void q_wd_poll(void){
  for(int k=0;k<WD_N;k++) if(g_wdf[k] && !g_wds[k]){ GLenum r=r_glClientWaitSync(g_wdf[k],GL_SYNC_FLUSH_COMMANDS_BIT,0);
    if(r==GL_ALREADY_SIGNALED || r==GL_CONDITION_SATISFIED) g_wds[k]=1; }
  __atomic_store_n(&g_wd_probe,0,__ATOMIC_RELEASE); }

/* stall dump side: ask the GL thread for a poll, 1 if it answered within 200 ms */
int glq_wd_probe(void){
  if(!g_active) return 0;
  __atomic_store_n(&g_wd_probe,1,__ATOMIC_RELEASE); q_wake();
  for(int i=0;i<200 && __atomic_load_n(&g_wd_probe,__ATOMIC_ACQUIRE);i++) plat_sleep_ns(1000000);
  return !__atomic_load_n(&g_wd_probe,__ATOMIC_ACQUIRE); }

/* 1 done, 0 not done, -1 that fence is gone (too old) */
int glq_wd_done(u32 id){ u32 k=id%WD_N; if(__atomic_load_n(&g_wdid[k],__ATOMIC_ACQUIRE)!=id) return -1; return g_wds[k]? 1 : 0; }

/* ---------- swap */
static volatile u64 g_gl_cpu;   /* CPU time of the GL thread (the driver's work), sampled at each swap */

u64 glq_thread_cpu_ns(void){ return __atomic_load_n(&g_gl_cpu,__ATOMIC_RELAXED); }

static void e_swap(void* p){ (void)p; backend_gl_swap(); __atomic_store_n(&g_gl_cpu,plat_thread_cpu_ns(),__ATOMIC_RELAXED); __atomic_add_fetch(&g_swaps_done,1,__ATOMIC_SEQ_CST); q_done_signal(); }

static int q_swap_ready(void* p){ (void)p; return g_swaps_sent-__atomic_load_n(&g_swaps_done,__ATOMIC_SEQ_CST)<=1; }

void glq_swap(void){
  if(!g_active || t_glq_direct){ backend_gl_swap(); return; }
  g_swaps_sent++; glq_alloc(16,0,e_swap); glq_commit();
  /* max one frame ahead, more just adds input lag */
  q_wait(q_swap_ready,NULL); }

int glq_worker_current(void){ int r=backend_gl_worker_current(); if(r) t_glq_direct=1; return r; }

/* ---------- proc lookup */
void* glq_proc(const char* n){
  if(!g_active) return backend_gl_proc(n);
  for(const GlqEnt* e=g_glq_tab;e->name;e++) if(!strcmp(e->name,n)) return *e->real? e->q : NULL;
  { static int said; if(said<8){ said++; LOG("glq: %s is not in the queue table (called directly)\n",n); } }
  return backend_gl_proc(n); }

/* ---------- driver exclusive mode
   intel's windows driver crashes when the compile thread and the GL thread are inside it at the same time (smo died
   on a different shader every run, then everything after the first crash crashed too). with this on they take
   turns: the GL thread holds the lock while it runs a batch, the compile thread while it compiles */
static PMutex g_drv; static volatile int g_drv_on;
static volatile int g_drv_who;   /* 1 while the GL thread is inside the driver, for the stall report */
static volatile u64 g_batch_t0; static void (* volatile g_cur_ex)(void*);
void glq_excl(void){ if(g_drv_on) return; pm_init(&g_drv); __atomic_store_n(&g_drv_on,1,__ATOMIC_RELEASE); }
int glq_drv_lock(void){ if(!__atomic_load_n(&g_drv_on,__ATOMIC_ACQUIRE)) return 0; pm_lock(&g_drv); return 1; }
void glq_drv_unlock(void){ pm_unlock(&g_drv); }

/* ---------- GL thread */
static int g_start_ok;

static void gl_thread(void* arg){ (void)arg; { const char* e=getenv("SWR_PROF_FOCUS"); if(e && !strcmp(e,"gl")){ extern void plat_prof_focus_self(void); plat_prof_focus_self(); } }   /* profile=1 + SWR_PROF_FOCUS=gl */
  { extern void plat_thread_high_priority(void); plat_thread_high_priority(); }
  t_glq_direct=1;
  int ok= backend_gl_create() && swr_gl_load(backend_gl_proc);
  if(ok){ for(const GlqEnt* e=g_glq_tab;e->name;e++){ void* r= e->pvar? *e->pvar : NULL; if(!r) r=backend_gl_proc(e->name); *e->real=r; }
    for(const GlqEnt* e=g_glq_tab;e->name;e++) if(e->pvar && *e->real) *e->pvar=e->q; }
  g_start_ok=ok; __atomic_store_n(&g_ready,1,__ATOMIC_RELEASE);
  if(!ok) return;
  u32 n=0;
  for(;;){
    u64 h=__atomic_load_n(&g_qh,__ATOMIC_ACQUIRE), t=g_qt;
    if(t==h){
      /* the fence poll is a driver call too, so it takes the lock like a batch. without it the poll ran while the
         compile thread was inside the driver and tomodachi froze on intel a few seconds after its first compute */
      if(g_vf_list){ int lk=glq_drv_lock(); g_drv_who=1; q_poll_fences(); g_drv_who=0; if(lk) glq_drv_unlock(); }
      if(__atomic_load_n(&g_wd_probe,__ATOMIC_ACQUIRE)){ int lk=glq_drv_lock(); g_drv_who=1; q_wd_poll(); g_drv_who=0; if(lk) glq_drv_unlock(); }
      { extern int plat_ncpu(void); static u64 spin; if(!spin) spin= plat_ncpu()<=2? 15000 : 50000;   /* less on 2 cores, the game needs them */
        u64 t0=plat_ns(); while(__atomic_load_n(&g_qh,__ATOMIC_ACQUIRE)==t && plat_ns()-t0<spin) cpu_relax(); }   /* then sleep on g_qc (woken by glq_commit) */
      if(__atomic_load_n(&g_qh,__ATOMIC_ACQUIRE)!=t) continue;
      pm_lock(&g_qm); __atomic_store_n(&g_sleeping,1,__ATOMIC_RELEASE);
      if(__atomic_load_n(&g_qh,__ATOMIC_ACQUIRE)==t) pc_wait(&g_qc,&g_qm, g_vf_list? 500000 : 2000000);
      __atomic_store_n(&g_sleeping,0,__ATOMIC_RELEASE); pm_unlock(&g_qm); continue; }
    int lk=glq_drv_lock(); g_drv_who=1; g_batch_t0=plat_ns();
    while(t!=h){
      QHdr* hd=(QHdr*)(g_q+t%Q_CAP); u32 sz=hd->size;
      g_cur_ex=hd->ex; { u64 c0=plat_ns(); hd->ex(hd->data); u64 dc=plat_ns()-c0; if(dc>10000000ull){ extern void swr_tl(const char*,...); char m[64]; u64 off=0; if(!plat_addr_module((u64)(uintptr_t)hd->ex,m,sizeof m,&off)) m[0]=0; swr_tl("GL call +%llx took %.1f ms",(unsigned long long)off,dc/1e6); } } if(hd->big) free(hd->data);
      t+=sz; __atomic_store_n(&g_qt,t,__ATOMIC_RELEASE);
      if(((++n)&255)==0){ if(g_vf_list) q_poll_fences(); if(g_wd_probe) q_wd_poll(); } }
    g_batch_t0=0; g_drv_who=0; if(lk) glq_drv_unlock();
  } }

/* one line for the stall dump: is the GL thread stuck inside the driver, idle, or waiting for the driver lock */
void glq_stall_report(char* out, size_t cap){
  out[0]=0; if(!g_active){ snprintf(out,cap,"off"); return; }
  u64 h=__atomic_load_n(&g_qh,__ATOMIC_ACQUIRE), t=__atomic_load_n(&g_qt,__ATOMIC_ACQUIRE), b0=g_batch_t0;
  int o=snprintf(out,cap,"%.1f KB queued, ",(h-t)/1024.0);
  if(b0){ char m[64]; u64 off=0; void* ex=(void*)g_cur_ex;
    if(!plat_addr_module((u64)(uintptr_t)ex,m,sizeof m,&off)) snprintf(m,sizeof m,"?");
    snprintf(out+o,cap-o,"in a batch for %.0f ms (call at %s+%llx)",(plat_ns()-b0)/1e6,m,(unsigned long long)off); }
  else if(h!=t) snprintf(out+o,cap-o,"waiting for the driver lock");
  else snprintf(out+o,cap-o,"idle%s",g_vf_list? ", polling fences" : "");
  u64 w0=g_wait_t0; const char* wn=g_wait_name; size_t l=strlen(out);
  if(w0 && l<cap) snprintf(out+l,cap-l,", the GPU thread waits on it for %.0f ms (%s)",(plat_ns()-w0)/1e6,wn? wn : "sync call"); }

int glq_start(void){
  if(getenv("SWR_NO_GLQ") || !backend_glq_ok()) return 0;
  g_q=malloc(Q_CAP+256);   /* slack so a header written at the end can't run past the buffer */ if(!g_q) return 0;
  pm_init(&g_qm); pc_init(&g_qc); pm_init(&g_dm); pc_init(&g_dc);
  if(plat_thread(gl_thread,NULL,8u<<20)) return 0;
  while(!__atomic_load_n(&g_ready,__ATOMIC_ACQUIRE)) plat_sleep_ns(1000000);
  if(!g_start_ok) return -1;
  g_active=1; LOG("gpu: OpenGL calls run on their own thread (the GPU thread and the driver work at the same time)\n");
  return 1; }
