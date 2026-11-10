/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* Switch 2 graphics services. A Switch 2 game reaches the GPU through ISelfController 2000 (GetGraphicsSupervisor),
   not through nvdrv: the supervisor (TIPC) bootstraps, creates native memory and images, and opens sessions for the
   GPU (address spaces, devices, a worker that maps GPU memory), syncpoints, display ("vi2") and an nvdrv session of
   its own (1402). Underneath it all is our nvdrv: its nvmap handles, GPU page table and channels.
   Requests are TIPC: W(c,0) is the header word (16 + command), W(c,1) the size word, the arguments follow. Results go
   back as the first payload word (the dispatch return value) */
#include "rt.h"

extern SvcObj* nvdrv_obj_new(const char* name);
extern u32 nv_native_handle(u64 addr, u64 size);
extern void nv_gpu_map_raw(u64 gva, u64 cpu, u64 size), nv_gpu_unmap_raw(u64 gva, u64 size);
extern int nv_native_bind(u32 h, u64 addr), nv_native_alive(u32 h), nv_native_map(u64 gva, u32 h, u64 off, u64 size);
extern void nv_native_free(u32 h), nv_native_unmap(u64 gva, u64 size);
extern int nv_open_dev(const char* name), nv_fd_is_as(u32 fd);
extern u32 syncpt_read(u32 id); extern int syncpt_wait(u32 id, u32 thresh, i64 timeout);
extern u64 nvmap_addr(u32 h); extern u32 nvmap_size(u32 h);
extern void vi_show_addr(u64 a, u32 w, u32 h, u32 pitch, u32 layout, u32 bhl, u32 fmt);
extern int ipc_warn_once(const char* name, u32 cmd);

/* kernel results (module 1) */
#define K_INVSIZE   0xCA01
#define K_INVADDR   0xCC01
#define K_OUTOFRES  0xCE01
#define K_INVMEM    0xD401
#define K_NOTIMPL   0xF001
#define K_INVSTATE  0xFA01
#define K_NOTFOUND  0xF201
#define K_INVCOMB   0xE801
#define K_BUSY      0xFC01

static u32 W(IpcCtx* c, u32 i){ u32 v; memcpy(&v,c->msg+4*i,4); return v; }
static u64 W64(IpcCtx* c, u32 i){ return (u64)W(c,i)|((u64)W(c,i+1)<<32); }
static u32 ev_h(KEvent* e){ return h_new(&e->o); }
/* Switch 2 services fail soft: what this file doesn't know is logged and answered "ok" with zeros, so the game goes
   on and one log shows several missing pieces instead of stopping at the first one. SWR_NS2_STRICT=1 answers
   NotImplemented like the system */
static int ns2_strict(void){ static int v=-1; if(v<0) v=getenv("SWR_NS2_STRICT")!=NULL; return v; }
static int soft_worker_disp(SvcObj*,IpcCtx*); static int soft_sync_disp(SvcObj*,IpcCtx*); static int soft_vi2_disp(SvcObj*,IpcCtx*);
static int soft_gpusess_disp(SvcObj*,IpcCtx*); static int soft_gsv_disp(SvcObj*,IpcCtx*);
/* a command this file doesn't know */
static int ns2_unknown(SvcObj* o, IpcCtx* c){
  /* command 0 with a 64 bit word is a version handshake: the sdk sends its version and aborts unless the same value
     comes back (nn::syncpt::sdk::Initialize does it on the syncpoint session) */
  if(c->cmd==0 && (W(c,1)&0x3FF)>=2){ u64 v=W64(c,2);
    if(ipc_warn_once(o->name,0)) LOG("switch 2: %s handshake %llx\n",o->name,(unsigned long long)v);
    out64(c,v); return 0; }
  if(ipc_warn_once(o->name,c->cmd)){ u32 n=W(c,1)&0x3FF; if(n>12) n=12; char b[160]; int k=0; for(u32 i=0;i<n;i++) k+=snprintf(b+k,sizeof b-k," %x",W(c,2+i));
    WARN("  [switch 2] %s cmd %u not implemented (words%s)%s\n",o->name,c->cmd,k? b : " none",ns2_strict()? "" : ", answered ok with zeros"); }
  if(ns2_strict()) return K_NOTIMPL;
  for(int i=0;i<4;i++) out64(c,0);   /* the game reads its outputs as zeros instead of aborting on NotImplemented */
  return 0; }
/* a check this file doesn't know how to answer (NotImplemented) becomes success too, logged once per command */
static int ns2_soft(SvcObj* o, IpcCtx* c, int rc){
  if(rc!=K_NOTIMPL || ns2_strict()) return rc;
  if(ipc_warn_once("ns2-soft",((u32)(uintptr_t)o->name<<8)^c->cmd)) WARN("  [switch 2] %s cmd %u: case not handled, answered ok\n",o->name,c->cmd);
  return 0; }
/* a session handed back as a move handle; on failure the slot still goes out (the SDK reads it before the result) */
static int move_obj(IpcCtx* c, SvcObj* o, int rc){ out_move_h(c, rc? 0 : session_new(o)); if(rc && o) svcobj_unref(o); return rc; }

/* ============================================================ shared state (one process) */
static PMutex g_ns2; static int g_ns2_init;
static void ns2_lock(void){ if(!g_ns2_init){ pm_init(&g_ns2); g_ns2_init=1; } pm_lock(&g_ns2); }
static void ns2_unlock(void){ pm_unlock(&g_ns2); }

typedef struct { u32 id, mem; u32 pref[16]; u64 size; int window_pool; u64 off; } NsImage;   /* off: where in mem (image pools) */
static NsImage g_img[1024]; static u32 g_nimg, g_next_img=1;
static u32 g_native_mem[1024]; static u32 g_nnative;
static int g_phase;              /* 0 created, 1 initialized, 2 process bound, 3 ready */
static int g_gpu_ctx, g_gpu_ready, g_frame_ctx, g_img_ctx, g_img_formats, g_disp_ctx, g_sync_ctx;
/* MKW 1.0's sdk never opens the image context (1501/1502) before it sizes and creates its window images
   (435, 429). Later sdks open it first; the formats count as ready until a game starts opening the context itself */
static int img_formats_ok(void){ return g_img_formats || !g_img_ctx; }

static int native_mem_has(u32 id){ for(u32 i=0;i<g_nnative;i++) if(g_native_mem[i]==id) return 1; return 0; }
static NsImage* image_find(u32 id){ for(u32 i=0;i<g_nimg;i++) if(g_img[i].id==id) return &g_img[i]; return NULL; }

/* the size of a native image: four-byte color, row pitch in GOBs, block height 2^pref[4] pixels */
/* a layout image_size doesn't know: a size that's surely big enough (16 bytes a pixel unless the pitch says less, 128
   line blocks, every layer and twice that for mips), so the game gets memory instead of an abort */
static int image_size(const u32* p, u32 count, u64* size, u64* align);
static int image_size_any(const u32* p, u32 count, u64* size, u64* align){
  if(image_size(p,count,size,align)) return 1;
  { static int said; if(said<sw2_cap(16)){ said++; WARN("switch 2: image layout not handled (count %u):",count); for(int i=0;i<16;i++) WARN(" %x",p[i]); WARN("%s\n",ns2_strict()? "" : ", size estimated"); } }
  if(ns2_strict()) return 0;
  u64 w=(p[2]&0xFFFF)+1, h=(p[2]>>16)+1, pitch= p[5]>=(w*4+63)/64 && p[5]<=4096? (u64)p[5]*64 : w*16;
  u64 layers=(count? count : 1), one=pitch*((h+127)&~127ull);
  *size=((one*layers*(p[7]>1? 2 : 1))+0xFFFF)&~0xFFFFull; if(align) *align=0x10000; return 1; }

/* p[6] isn't always 1: MKW's window asks for its 3840x2160 textures with 4 there. It doesn't make the
   image bigger: the window places its textures one image size apart, and with 4x the size its memory pool came to
   382 MB, more than the game's 117 MB GraphicsSystem heap (HeapDef), so the pool got no memory and the window aborted.
   SWR_NS2_IMG_LAYERS=1 multiplies by it again */
static int image_size(const u32* p, u32 count, u64* size, u64* align){
  if(count!=1 || p[0]!=0 || (p[1]!=1 && p[1]!=2) || p[3]!=0 || !p[6] || p[6]>16 || p[7]!=1 || p[4]>8) return 0;
  for(int i=8;i<16;i++) if(p[i]) return 0;
  u64 w=(p[2]&0xFFFF)+1, h=(p[2]>>16)+1, gobs=p[5]; if(gobs<(w*4+63)/64 || gobs>4096) return 0;
  u64 bh=1ull<<p[4]; u64 one=gobs*64*((h+bh-1)&~(bh-1)); one=(one+0xFFF)&~0xFFFull;
  *size= getenv("SWR_NS2_IMG_LAYERS")? one*p[6] : one; if(align){ *align=64*bh; if(*align<0x1000) *align=0x1000; } return 1; }

/* ============================================================ syncpoint cache: 192 counters the GPU's syncpoints
   are copied into, in two transfer memories (requests in rw, values and acks in ro), refreshed every millisecond */
#define SC_N 192
static u64 g_sc_rw, g_sc_ro; static KEvent *g_sc_req, *g_sc_done; static u64 g_sc_prev[SC_N];
static void sc_refresh(void){
  if(!g_sc_rw) return;
  u64 req=gr64(g_sc_rw), ack=gr64(g_sc_ro+0x640), groups=(req^ack)&7; if(!groups) return;
  for(u32 g=0;g<3;g++){ if(!(groups>>g&1)) continue;
    u64 want=gr64(g_sc_rw+8+g*8), slots=want^gr64(g_sc_ro+0x648+g*8);
    for(u32 b=0;b<64;b++) if(slots>>b&1){ u32 id=g*64+b; u32 cur=syncpt_read(id); u64 prev=g_sc_prev[id];
      u64 v= prev==~0ull? cur : ((prev&0xFFFFFFFF00000000ull)+((u32)cur<(u32)prev? 1ull<<32 : 0))|cur; g_sc_prev[id]=v; gw64(g_sc_ro+0x40+id*8,v); }
    __atomic_thread_fence(__ATOMIC_RELEASE); gw64(g_sc_ro+0x648+g*8,want); }
  __atomic_thread_fence(__ATOMIC_RELEASE); gw64(g_sc_ro+0x640,req);
  kevent_clear(g_sc_req); kevent_signal(g_sc_done); }
/* syncpoint event slots (ISyncpointSupervisor 22/201/202): 201 asks for the slot's event once a syncpoint reaches a
   value, 202 cancels and always signals (nn::syncpt::sdk::CancelSyncPointEvent waits on it either way) */
static struct { KEvent* ev; u32 id, val; int armed; } g_sps[64];
static KEvent* sps_ev(u32 s){ s&=63; if(!g_sps[s].ev){ g_sps[s].ev=kevent_new("syncpoint slot",0); kobj_ref(&g_sps[s].ev->o); } return g_sps[s].ev; }
static void sps_poll(void){ for(int i=0;i<64;i++) if(g_sps[i].armed && (i32)(syncpt_read(g_sps[i].id)-g_sps[i].val)>=0){ extern void swr_tl(const char*,...); swr_tl("slot %d fires (sp%u >= %u)",i,g_sps[i].id,g_sps[i].val); g_sps[i].armed=0; kevent_signal(sps_ev(i)); } }
/* vi2 31/32 events: the display's vsync, every 16.7 ms (the main thread waits on 32 before its first frame) */
static KEvent* g_vsync[2];
static KEvent* g_vdec_ev[16];   /* signalled at vsync too: the client attaches its ready event auto-clearing */
static void sc_thread(void* a){ (void)a; u64 next=plat_ns()+16666667;
  for(;;){ plat_sleep_ns(1000000); ns2_lock(); sc_refresh(); sps_poll();
    if(plat_ns()>=next){ next+=16666667; if(plat_ns()>next) next=plat_ns()+16666667; for(int i=0;i<2;i++) if(g_vsync[i]) kevent_signal(g_vsync[i]); for(int i=0;i<16;i++) if(g_vdec_ev[i]) kevent_signal(g_vdec_ev[i]); }
    ns2_unlock(); } }
/* started by whichever comes first: vsync events, an armed syncpoint slot or the syncpoint cache (gsv 204). MKW 1.0's sdk never binds the cache, so with the thread started only there vsync never came and its main thread
   waited forever before the first frame. Called with ns2_lock held */
static void sc_start(void){ static int started; if(!started){ started=1; plat_thread(sc_thread,NULL,256*1024); } }

static int sync_disp(SvcObj* o, IpcCtx* c){   /* ISyncpointSupervisor */
  switch(c->cmd){
    case 1: out32(c,63); return 0;                                      /* GetMaximumEventCount */
    case 21: return W(c,2)<=63? 0 : K_INVCOMB;                          /* SetEventCount */
    case 103: { u32 id=~W(c,2); out64(c,syncpt_read(id)); out32(c,0); return 0; }   /* QueryCounter */
    case 22: ns2_lock(); out_move_h(c,ev_h(sps_ev(W(c,2)))); ns2_unlock(); return 0;   /* the slot's event */
    case 201: { u32 s=W(c,2)&63; { extern void swr_tl(const char*,...); swr_tl("arm slot %u: sp%u >= %u (now %u)",s,~W(c,3),W(c,4),syncpt_read(~W(c,3))); } ns2_lock(); sc_start(); KEvent* e=sps_ev(s); kevent_clear(e); g_sps[s].id=~W(c,3); g_sps[s].val=W(c,4); g_sps[s].armed=1;
      { static int dbg=-1; if(dbg<0) dbg=getenv("SWR_SPS_LOG")? atoi(getenv("SWR_SPS_LOG"))*400 : 0; static int n; if(dbg && n<dbg){ n++; LOG("sps arm slot %u: syncpoint %u >= %u (now %u) words %x %x %x %x %x %x\n",s,g_sps[s].id,g_sps[s].val,syncpt_read(g_sps[s].id),W(c,2),W(c,3),W(c,4),W(c,5),W(c,6),W(c,7)); } }
      sps_poll(); ns2_unlock(); return 0; }
    case 202: { u32 s=W(c,2)&63; { extern void swr_tl(const char*,...); if(g_sps[s].armed) swr_tl("cancel slot %u (sp%u now %u, wanted %u)",s,g_sps[s].id,syncpt_read(g_sps[s].id),g_sps[s].val); } ns2_lock();
      { static int dbg=-1; if(dbg<0) dbg=getenv("SWR_SPS_LOG")? atoi(getenv("SWR_SPS_LOG"))*400 : 0; static int n; if(dbg && n<dbg && g_sps[s].armed){ n++; LOG("sps cancel slot %u: syncpoint %u still %u (wanted %u)\n",s,g_sps[s].id,syncpt_read(g_sps[s].id),g_sps[s].val); } }
      g_sps[s].armed=0; kevent_signal(sps_ev(s)); ns2_unlock(); return 0; }
  }
  return ns2_unknown(o,c); }

/* ============================================================ GPU session, worker */
typedef struct { int as[8], as_reg[8]; u64 dev_name[256]; u8 dev_used[256]; KEvent* ev[64]; SvcObj* worker; } GpuDevs;
static GpuDevs g_dev;
typedef struct { u32 as; u64 gva, size; u32 mem; } GpuMap;
static GpuMap g_maps[32768]; static u32 g_nmaps;
typedef struct { u64 gva, size, cpu; } Sparse; static Sparse g_sparse[256]; static u32 g_nsparse;
/* memory behind sparse regions, at most 2 GB in all (SWR_SPARSE_MB): Windows commits all of it, and MKW's
   ~8 GB of regions on top of its 7.6 GB heap ran PCs out of commit, the next allocations failed and the runtime crashed.
   The renderer keeps render targets in host textures, a region left unbacked only loses CPU reads of them */
extern int g_gpu_eden; extern int mem_map_lazy(u64,u64,u32,u32);
static u64 g_sparse_mb; static u64 sparse_budget_mb(void){ static u64 v; if(!v){ const char* e=getenv("SWR_SPARSE_MB"); v= e? strtoull(e,NULL,10) : 2048; if(!v) v=1; } return v; } static u32 g_wmem[16384]; static u32 g_nwmem;

static int as_owned(u32 id){ for(int i=0;i<8;i++) if(g_dev.as[i]==(int)id && g_dev.as_reg[i]) return 1; return 0; }
static void ev_sig(u32 i){ if(i<64 && g_dev.ev[i]) kevent_signal(g_dev.ev[i]); }
/* memory kinds: equal requested/fallback ones need no compression; compressible ones fall back */
static int kind_select(u8 req, u8 fb, u8* sel){
  if(req==fb && (req==0 || req==1 || req==3 || req==6)){ *sel=req; return 1; }
  if((req==8 && fb==6) || (req==0xB && fb==1) || (req==0xC && fb==3)){ *sel=fb; return 1; }
  return 0; }
/* guest memory the GPU may use carries attribute 0x20 (svc 0x80) */
static int gpu_accessible(u64 a, u64 end){
  for(u64 p=a;p<=end;){ MemBlock b; mem_query(p,&b); if(!b.size || !(b.attr&0x20)) return 0; u64 n=b.addr+b.size; if(n<=p) return 0; p=n; }
  return 1; }

static int worker_disp(SvcObj* o, IpcCtx* c){   /* IGpuWorker */
  switch(c->cmd){
    case 101: { /* CreateGpuAccessibleMemory {begin, end, flags u8, device u8} */
      u64 b=W64(c,2), e=W64(c,4); u32 fl=W(c,6)&0xFF, dv=W(c,7)&0xFF; u32 id=0; int rc;
      if(fl) rc=K_NOTIMPL; else if(!g_dev.dev_used[dv]) rc=K_NOTFOUND; else if(e<b || e==~0ull) rc=K_INVSIZE;
      else if((b|(e-b+1))&0xFFF) rc=(b&0xFFF)? K_INVADDR : K_INVSIZE; else if(g_nwmem>=16384) rc=K_OUTOFRES;
      else if(!gpu_accessible(b,e)) rc=K_INVMEM;
      else { id=nv_native_handle(b,e-b+1); rc= id? 0 : K_OUTOFRES; if(id){ g_wmem[g_nwmem++]=id; extern void gpu_eden_memory(u64,u64,int); gpu_eden_memory(b,e-b+1,1); } }
      if(rc){ static int said; if(said<sw2_cap(8)){ said++; LOG("switch 2: GPU-accessible memory %llx-%llx refused (%x, %u registered)\n",(unsigned long long)b,(unsigned long long)e,rc,g_nwmem); } }
      out32(c,id); return rc; }
    case 102: { /* DestroyGpuAccessibleMemory {id, notification} */
      u32 id=W(c,2), n=W(c,3); for(u32 i=0;i<g_nwmem;i++) if(g_wmem[i]==id){ { extern void gpu_eden_memory(u64,u64,int); gpu_eden_memory(nvmap_addr(id),nvmap_size(id),0); } nv_native_free(id); g_wmem[i]=g_wmem[--g_nwmem]; ev_sig(n); return 0; }
      return K_NOTFOUND; }
    case 211: { /* NormalMapGpuMemory4K {as, begin, end, memory, offset, ..., kind, fallback, ..., device, ..., notification} */
      u32 as=W(c,2), mem=W(c,7), dv=W(c,15)&0xFF, note=W(c,17); u64 b=W64(c,3), e=W64(c,5), off=W64(c,8); u8 sel=0; int rc;
      if(!as_owned(as) || !nv_native_alive(mem) || !g_dev.dev_used[dv]) rc=K_NOTFOUND;
      else if(!kind_select((u8)W(c,12),(u8)W(c,13),&sel)) rc=K_NOTIMPL;
      else if(e<b || e==~0ull) rc=K_INVSIZE; else if(g_nmaps>=32768) rc=K_OUTOFRES;
      else { rc= nv_native_map(b,mem,off,e-b+1)? K_INVMEM : 0; if(!rc) g_maps[g_nmaps++]=(GpuMap){as,b,e-b+1,mem}; }
      ev_sig(note);
      { static int said; if(said<sw2_cap(6)){ said++; LOG("switch 2: GPU map %llx-%llx -> memory %u + %llx, kind %x (%x)\n",(unsigned long long)b,(unsigned long long)e,mem,(unsigned long long)off,sel,rc); } }
      out32(c,sel); return rc; }
    case 212: { /* NormalModifyGpuMemory4K: the kind only (our page table has none) */
      u64 b=W64(c,3), e=W64(c,5); u8 sel=0; int rc=K_NOTFOUND;
      if(kind_select((u8)W(c,7),(u8)W(c,8),&sel)) for(u32 i=0;i<g_nmaps;i++) if(g_maps[i].as==W(c,2) && b>=g_maps[i].gva && e<g_maps[i].gva+g_maps[i].size){ rc=0; break; }
      ev_sig(W(c,10)); out32(c,sel); return rc; }
    /* 214 CreateSparseRegion {as, first, last, ...}, done through the event in word 11; 215 destroys it. The graphics
       server backs these itself on the console (render targets live there, MKW makes several). Here
       each gets memory of its own up to 2 GB a region; a bigger one stays unbacked (logged), maps into it still work */
    case 214: { u64 b=W64(c,3), e=W64(c,5), sz= e>=b? e-b+1 : 0; int rc=0; u64 cpu=0;
      if(!sz || (b&0xFFF) || (sz&0xFFF)) rc=K_INVSIZE;
      else if(g_gpu_eden && g_nsparse<256){
        /* Eden: every region, memory committed only where something writes or reads (plat_vm_lazy_add), placed
           below 64 GB (Eden's device addresses are guest addresses, 36 bits). Eglantine backs them per block too */
        cpu=mem_find_free(0x400000000ull,0x1000000000ull,sz,0x10000);
        if(cpu && !mem_map_lazy(cpu,sz,MS_NORMAL,PERM_RW)){ extern void gpu_eden_gpu_only(u64,u64,int); nv_gpu_map_raw(b,cpu,sz); gpu_eden_gpu_only(cpu,sz,1); g_sparse[g_nsparse++]=(Sparse){b,sz,cpu}; } else cpu=0; }
      else if(g_nsparse<64 && sz<=(2ull<<30) && g_sparse_mb+(sz>>20)<=sparse_budget_mb()){ cpu=mem_find_free(0xC00000000ull,0x1300000000ull,sz,0x10000);
        if(cpu && !mem_map(cpu,sz,MS_NORMAL,PERM_RW)){ nv_gpu_map_raw(b,cpu,sz); g_sparse[g_nsparse++]=(Sparse){b,sz,cpu}; g_sparse_mb+=sz>>20; } else cpu=0; }
      { static int said; if(said<sw2_cap(8)){ said++; LOG("switch 2: sparse GPU region %llx-%llx (%llu MB) %s\n",(unsigned long long)b,(unsigned long long)e,(unsigned long long)(sz>>20),cpu? "backed" : "left unbacked"); } }
      ev_sig(W(c,11)); return rc; }
    case 215: { u64 b=W64(c,3);
      for(u32 i=0;i<g_nsparse;i++) if(g_sparse[i].gva==b){ nv_gpu_unmap_raw(b,g_sparse[i].size); if(g_gpu_eden){ extern void gpu_eden_gpu_only(u64,u64,int); gpu_eden_gpu_only(g_sparse[i].cpu,g_sparse[i].size,0); plat_vm_lazy_remove(GP(g_sparse[i].cpu),0); } mem_unmap(g_sparse[i].cpu,g_sparse[i].size); if(!g_gpu_eden) g_sparse_mb-=g_sparse[i].size>>20; g_sparse[i]=g_sparse[--g_nsparse]; break; }
      /* 215's job slot is word 8 (like 213), not 214's word 11: the wrong event got signalled and the caller waited
         forever with the GPU address-space lock held (MKW froze going from the menus into a race; Eglantine's notes) */
      ev_sig(W(c,8)); return 0; }
    case 213: { /* NormalUnmapGpuMemory4K */
      u64 b=W64(c,3), e=W64(c,5), n=0; int rc=K_NOTFOUND;
      for(u32 i=0;i<g_nmaps;i++) if(g_maps[i].as==W(c,2) && g_maps[i].gva==b && e-b+1==g_maps[i].size){ nv_native_unmap(b,g_maps[i].size); n=g_maps[i].size; g_maps[i]=g_maps[--g_nmaps]; rc=0; break; }
      ev_sig(W(c,8)); out64(c,n); return rc; }
  }
  return ns2_unknown(o,c); }

static int gpusess_disp(SvcObj* o, IpcCtx* c){   /* IGpuSession */
  ns2_lock(); int rc=0;
  switch(c->cmd){
    case 11: out32(c,0); out32(c,63); break;                            /* GetNotificationRange */
    case 12: { u32 i=W(c,2); if(i>=64){ rc=K_INVCOMB; out_move_h(c,0); break; }   /* GetNotificationEvent */
      if(!g_dev.ev[i]){ g_dev.ev[i]=kevent_new("gpu notification",0); kobj_ref(&g_dev.ev[i]->o); }
      out_move_h(c,ev_h(g_dev.ev[i])); break; }
    case 21: { u32 i=W(c,2); if(i>=64 || !g_dev.ev[i]){ rc=K_INVSTATE; break; }   /* RequestWorker */
      if(!g_dev.worker) g_dev.worker=svcobj_new("IGpuWorker",soft_worker_disp,0); kevent_signal(g_dev.ev[i]); break; }
    case 22: if(!g_dev.worker){ rc=K_INVSTATE; out_move_h(c,0); break; }   /* AcquireWorker */
      g_dev.worker->refs++; out_move_h(c,session_new(g_dev.worker)); break;
    case 33: { rc=K_OUTOFRES; for(int i=0;i<8;i++) if(!g_dev.as[i]){ int fd=nv_open_dev("/dev/nvhost-as-gpu");   /* CreateAddressSpace */
        if(fd>0){ g_dev.as[i]=fd; out32(c,(u32)fd); rc=0; } break; }
      if(rc) out32(c,0); break; }
    case 41: { u64 name=W64(c,2), flags=W64(c,4); u32 id=0; rc= flags? K_NOTIMPL : K_OUTOFRES;   /* CreateGraphicsDevice */
      if(!flags) for(u32 i=1;i<256;i++) if(!g_dev.dev_used[i]){ g_dev.dev_used[i]=1; g_dev.dev_name[i]=name; id=i; rc=0; break; }
      out32(c,id); break; }
    case 42: { u32 i=W(c,2)&0xFF; rc= g_dev.dev_used[i]? 0 : K_NOTFOUND; g_dev.dev_used[i]=0; break; }   /* DestroyGraphicsDevice */
    /* 201 {as, kind}: the user partition on MKW 1.0's sdk (later sdks ask with 203), answered {min, max}
       without 203's valid word. Answered "ok" with zeros, NVN took 0..0 as its range, the 512 GB reservation failed
       (0x11) and NVN closed its device */
    case 201: { u32 id=W(c,2); int ok=0; for(int i=0;i<8;i++) if(g_dev.as[i]==(int)id) ok=1;
      if(ipc_warn_once("gpusess-201",W(c,3))) LOG("switch 2: GPU partition query 201, kind %u (answered as the user partition)\n",W(c,3));
      rc= ok? 0 : K_NOTFOUND; out64(c,0x04000000); out64(c,0xFFFFFFFFFFull); break; }
    case 202: case 203: { u32 id=W(c,2); int ok=0; for(int i=0;i<8;i++) if(g_dev.as[i]==(int)id) ok=1;   /* Get{Gsv,User}Partition */
      rc= ok? 0 : K_NOTFOUND;
      if(c->cmd==202){ out64(c,~0ull); out64(c,0); }
      else { /* the user partition: from the 4K region's start to the end of the 40 bit space (the 4K region of ioctl_as 0x08) */
        out64(c,ok? 1 : 0); out64(c,0x04000000); out64(c,0xFFFFFFFFFFull);   /* the whole 40 bit space: NVN reserves 0x8000000..512 GB of it and takes more past that */ for(int k=0;k<6;k++) out64(c,0); }
      break; }
    default: ns2_unlock(); return ns2_unknown(o,c);
  }
  ns2_unlock(); return rc; }

/* ============================================================ vi2: display sessions, presenting native images */
typedef struct { u64 token; u32 slots; int has_display, bound_default; u64 available; u32 retire_note, avail_note; u64 encoding; u32 queued; } Vi2Client;
typedef struct { Vi2Client cl[16]; u32 client_slots; int display_refs; u32 display_note; u32 notes[32]; KEvent* ev; int endpoint; u64 endpoint_token; } Vi2;
#define NAME_DEFAULT 0x00746C7561666564ull   /* "default" */
static Vi2Client* vi2_find(Vi2* v, u64 t){ if(!t) return NULL; for(int i=0;i<16;i++) if(v->cl[i].token==t) return &v->cl[i]; return NULL; }
static void vi2_note(Vi2* v, u32 n){ v->notes[(n/32)&31]|=1u<<(n%32); if(v->ev) kevent_signal(v->ev); }

volatile u32 g_swr_present_seq;   /* presents so far (frames.csv column n; Eden gets it with the image) */
int g_opt_tile_skip=-1;   /* runtime option texture_quality: tile levels left out, -1 auto (vk_sparse.inc) */
int g_frame_pacing;   /* runtime option frame_pacing: 0 game, 1 game_dynamic, 2 60 FPS, 3 30 FPS, 4 dynamic */
/* A video plays at its own speed. The game takes one picture a frame, so at interval 2 a 60 FPS movie ran at half speed while its sound went on at full speed.
   Only the display wait changes (one vblank a frame while pictures come): nothing but the movie moves while it plays. SWR_VIDEO_PACE=0: off */
static volatile u64 g_vdec_out_ns;   /* when the decoder last handed out a picture */
int vdec_playing(void){ static int vp=-1; if(vp<0){ const char* e=getenv("SWR_VIDEO_PACE"); vp= !e || *e!='0'; } return vp && g_vdec_out_ns && plat_ns()-g_vdec_out_ns<200000000ull; }
volatile u64 g_frame_mode;   /* what the recompiled frame-rate mode reads (c->user[6]): 0 game, 1 game_dynamic, 2 60, 3 30 */
/* frame_pacing=dynamic: the game's 60 FPS mode while frames keep up, its 30 FPS mode (two vblanks a frame, full game
   speed at 30) when they don't. A frame's work is the longest of: the game's (present to present minus the pacing
   wait) and the busy time of Eden's GPU thread and back thread over that frame (they run a frame behind the game, so
   the game's own time can look short while they are the slow part). To 30 when the average work is over 17.5 ms
   for half a second; back to 60 when it stays under 14 ms for three seconds */
static void frame_mode_update(u64 work_ns, u64 frame_ns, u32 iv){
  static double avg; static u64 above, below; static int inited; static u64 last_front, last_back;
  if(g_frame_pacing!=4){ g_frame_mode=(u64)g_frame_pacing; return; }
  { extern void gpu_eden_wait_ns(u64*,u64*); u64 f,b; gpu_eden_wait_ns(&f,&b);
    if(last_front || last_back){ u64 fw=f-last_front, bw=b-last_back;
      u64 fb= frame_ns>fw? frame_ns-fw : 0, bb= frame_ns>bw? frame_ns-bw : 0;
      if(fb>work_ns) work_ns=fb; if(bb>work_ns) work_ns=bb; }
    last_front=f; last_back=b; }
  if(!inited){ inited=1; g_frame_mode=2; avg=16e6; }
  if(work_ns>200000000ull) work_ns=200000000ull;   /* a loading stall says nothing about the next frames */
  avg=avg*0.9+(double)work_ns*0.1;
  u64 per=(u64)iv*1000000000ull/60;
  if(g_frame_mode==2){ if(avg>17.5e6){ above+=per; if(above>=500000000ull){ g_frame_mode=3; above=0; below=0; LOG("frame pacing: 30 FPS mode (frames take %.1f ms)\n",avg/1e6); } } else above=0; }
  else { if(avg<14e6){ below+=per; if(below>=3000000000ull){ g_frame_mode=2; above=0; below=0; LOG("frame pacing: 60 FPS mode (frames take %.1f ms)\n",avg/1e6); } } else below=0; } }
/* At each present, from both present paths (vi2_present for Eden, swr_nvn_frame for the native GPU): the frame-rate mode (frame_pacing) and
   vblank pacing. The game advances its clock by its present interval (vblanks of 60 Hz) each frame, so a frame may not be shown before that much
   time has passed since the last one, or the game runs fast. Slower than that: shown at once, the lost time is not made up. SWR_PACE=0: off.
   The native path never called this until 2026-11-07: MKW stayed in its 60 FPS mode at about 30 FPS and its races ran at 55% speed */
void swr_present_pace(u32 iv){
  { static int on=-1; if(on<0){ const char* e=getenv("SWR_PACE"); on= !e || *e!='0'; }
    static u64 next; if(iv<1) iv=1; if(iv>4) iv=4;
    /* dynamic mode: the game times its own frames to choose 1 or 2 vblanks; pacing every frame to 2 would keep it
       at 30 for good, so pace to each vblank and let it catch up through its interval */
    if(g_frame_pacing==1) iv=1;
    if(vdec_playing()) iv=1;
    static u64 last_present, last_wait;
    u64 now0=swr_run_ns(); if(last_present) frame_mode_update(now0-last_present-last_wait,now0-last_present,iv);
    last_wait=0;
    if(on){ u64 per=(u64)iv*1000000000ull/60, now=now0;
      if(!next || now>next+per) next=now;
      else if(now<next){ u64 tr=swr_trace_ts(); if(next-now>1500000) plat_sleep_ns((i64)(next-now-1000000)); while(swr_run_ns()<next) plat_yield(); swr_trace_span("vblank pacing",tr); last_wait=swr_run_ns()-now; }
      next+=per; }
    last_present=now0; }
}
static int vi2_present(Vi2* v, Vi2Client* s, IpcCtx* c){
  u32 slot=W(c,4)&0xFF; if(!s->has_display || slot>=15) return K_INVSTATE;
  NsImage* im=image_find(W(c,5)); if(!im || !im->window_pool) return K_NOTFOUND;
  u32 fence=~W(c,22), fval=W(c,24);
  /* word 7: the crop, (bottom << 16) | right, inclusive. MKW renders 1920x1080 into its 3840x2160 window
     images and shows only that corner (it filled a quarter of the screen without this) */
  { extern volatile i32 g_vi_crop[4]; u32 cr=W(c,7); if(cr){ g_vi_crop[0]=0; g_vi_crop[1]=0; g_vi_crop[2]=(i32)(cr&0xFFFF)+1; g_vi_crop[3]=(i32)(cr>>16)+1; } else memset((void*)g_vi_crop,0,sizeof g_vi_crop); }
  /* the image is complete once the GPU reached the fence. MKW passes no fence (its NativeSyncHandle is -1,
     which reads as syncpoint 0 here): that wait never ended and cost 200 ms each frame (5 FPS). The present goes to the
     GPU thread behind the submits that draw it anyway */
  { static int nl; if(nl<sw2_cap(3)){ nl++; LOG("switch 2: present words 20-25: %08x %08x %08x %08x %08x %08x\n",W(c,20),W(c,21),W(c,22),W(c,23),W(c,24),W(c,25)); } }
  /* frames.csv next to the exe, one line per present: ms since start, the game's present interval (vblanks per frame,
     high half of word 12, as nvnWindowSetPresentInterval set it: the game's clock advances that many vblanks), and how
     long this present waited for the GPU side. tools/bench.py reads it (frame times, game speed) */
  u64 t_present=swr_run_ns(), t_tr=swr_trace_ts();
  swr_trace_poll();
  { extern int g_gpu_eden; extern void gpu_eden_present_wait(u32,u32);
    if(g_gpu_eden) gpu_eden_present_wait(fence,fval);   /* Eden: pace the game to the GPU thread (gpu_eden.c), up to 4 s */
    else if(fence!=~0u && fence!=0) syncpt_wait(fence,fval,50000000); }
  /* vblank pacing: the game advances its clock by its present interval (vblanks of 60 Hz) each frame, so a frame may
     not be shown before that much time has passed since the last one, or the game runs fast (MKW races
     at interval 2 reached 109% speed at 32 FPS). Slower than that: shown at once, the lost time is not made up.
     SWR_PACE=0: off */
  swr_present_pace((W(c,12)>>16)&0xFF);
  { static FILE* ff; static u32 nf; u32 iv=(W(c,12)>>16)&0xFF; u64 waited=swr_run_ns()-t_present;
    swr_trace_span_arg("present wait (interval)",t_tr,iv);
    { extern void input_rec_present(u32); input_rec_present(iv); }
    if(!ff){ char p[1400]; snprintf(p,sizeof p,"%sframes.csv",g_exedir); ff=fopen(p,"w"); if(ff) fprintf(ff,"ms,interval,wait_us,n,front_wait_ms,back_wait_ms\n"); }
    /* n: the present number, also handed to Eden with the image (composites.csv, captures/frames.bin) */
    { extern volatile u32 g_swr_present_seq; g_swr_present_seq++;
      /* the last two: how long Eden's front and back have waited for work so far (tools/bench.py: back ms a frame) */
      if(ff){ extern void gpu_eden_wait_ns(u64*,u64*); u64 fw,bw; gpu_eden_wait_ns(&fw,&bw);
        fprintf(ff,"%.3f,%u,%llu,%u,%.3f,%.3f\n",t_present/1e6,iv,(unsigned long long)(waited/1000),g_swr_present_seq,fw/1e6,bw/1e6); if(++nf%60==0) fflush(ff); } } }
  u32 w=(im->pref[2]&0xFFFF)+1, h=(im->pref[2]>>16)+1, pitch=im->pref[5]*64;
  u64 a=nvmap_addr(im->mem); if(!a) return K_NOTFOUND;
  { static u64 pk=~0ull; static int np; if(pk==~0ull){ const char* e=getenv("SWR_PEEK"); pk= e? strtoull(e,NULL,16) : 0; }   /* debug: guest words at a GPU address, each present */
    { static u32 fw; static int done; if(!done && np==30){ done=1; const char* e=getenv("SWR_FIND"); if(e){ fw=(u32)strtoul(e,NULL,16); int hits=0;   /* debug: where a u32 sits in guest memory */
        for(u64 a=0x10000; a<(1ull<<39) && hits<24;){ MemBlock b; mem_query(a,&b); if(!b.size) break; u64 n=b.addr+b.size; if(n<=a) break;
          if(b.state!=MS_FREE && b.perm){ const u32* w=(const u32*)GP(b.addr); for(u64 i=0;i<b.size/4 && hits<24;i++) if(w[i]==fw){ hits++; LOG("find %08x at %llx\n",fw,(unsigned long long)(b.addr+i*4)); } }
          a=n; } LOG("find %08x: %d hits\n",fw,hits); } } }
    { static int done; if(!done && np==25){ const char* e=getenv("SWR_DUMPMEM"); if(e){ done=1; u64 da=strtoull(e,NULL,16), dn=4u<<20; extern u64 gmmu_translate(u64); u64 c0=gmmu_translate(da);   /* debug: 4 MB of guest memory at a GPU address */
        if(c0){ FILE* f=fopen("captures/memdump.bin","wb"); if(f){ fwrite(GP(c0),1,dn,f); fclose(f); LOG("dumped 4 MB at %llx\n",(unsigned long long)da); } } } } }
    if(pk && np%20==0){ extern u64 gmmu_translate(u64); u64 c0=gmmu_translate(pk); if(c0){ int nz=0, first=-1; for(int i=0;i<0x1000;i++) if(gr32(c0+i*4)){ nz++; if(first<0) first=i*4; } LOG("peek %llx (present %d): %d non-zero words in 16 KB (first at +%x)\n",(unsigned long long)pk,np,nz,first);
      if(np==20){ for(int r=0;r<0x240/16;r++){ float f[4]; memcpy(f,GP(c0+r*16),16); LOG("  +%03x: %g %g %g %g\n",r*16,f[0],f[1],f[2],f[3]); } } } }
    if(pk && np++<0){ extern u64 gmmu_translate(u64); u64 c=gmmu_translate(pk); if(c) LOG("peek %llx: %08x %08x %08x %08x  +0x40: %08x %08x %08x %08x  +0x210: %08x %08x %08x %08x\n",(unsigned long long)pk,gr32(c),gr32(c+4),gr32(c+8),gr32(c+12),gr32(c+0x40),gr32(c+0x44),gr32(c+0x48),gr32(c+0x4c),gr32(c+0x210),gr32(c+0x214),gr32(c+0x218),gr32(c+0x21c)); } }
  { static int said; if(said<sw2_cap(3)){ said++; LOG("switch 2: present image %u (%ux%u, pitch %u, block height 2^%u) slot %u, fence %d/%u\n",im->id,w,h,pitch,im->pref[4],slot,(int)fence,fval); } }
  vi_show_addr(a,w,h,pitch,3,im->pref[4]>=3? im->pref[4]-3 : 0,532120);   /* RGBA8, block linear, block height in GOBs */
  s->available|=1ull<<slot; vi2_note(v,s->avail_note);   /* shown already: the slot is free again */
  (void)fval; return 0; }

/* the chat context's session (GameChat camera images): command 11 hands out its notification event like vi2's;
   no camera, so the event never fires. The rest answers ok */
static int chat_disp(SvcObj* o, IpcCtx* c){
  if(c->cmd==11){ static KEvent* ev; if(!ev){ ev=kevent_new("chat notification",0); kobj_ref(&ev->o); } out_move_h(c,ev_h(ev)); return 0; }
  return ns2_unknown(o,c); }

/* video decoder output images (gsv 911/961): preference word 1 the layout code (1/4 NV12, 2/5 BGRA, 3/6 RGBA), word 2
   (w-1)|(h-1)<<16. Pitch-linear rows, 64-byte pitch, each plane padded to 128 rows, NV12 chroma below the luma */
typedef struct { u32 id, code, w, h, mem; u64 off; } DImg;
static DImg g_dimg[256]; static u32 g_ndimg;
typedef struct { int nv12; u64 pitch, chroma, size; } DImgLayout;
static DImgLayout dimg_layout(u32 code, u32 w, u32 h){ DImgLayout L; L.nv12= code==1 || code==4;
  L.pitch=((u64)w*(L.nv12? 1 : 4)+63)&~63ull; L.chroma=L.pitch*(((u64)h+127)&~127ull);
  L.size=L.chroma+(L.nv12? L.pitch*((((u64)h+1)/2+127)&~127ull) : 0); return L; }

/* video decoder (nn::codec::video): gsv 902/906/1002 hand out a session whose command 1 (CreateVideoDecoderService)
   hands out another. That one follows the protocol without decoding (movies get no picture yet): submitted frames
   (109/400) come out in order (213 GetOutputInfo) into the images the client registers or releases (311/314), each
   handed-out image's NativeSyncHandle is a value on the session's own syncpoint, raised once per frame. Answering 213
   with zeros made the movie player poll it forever: MKW froze at its title movie and before every race.
   Protocol from Eglantine's notes (row 148) */
#define VDEC_QN 4096   /* MKW submits a whole 1424-frame movie at once; 64 dropped most of it */
typedef struct { u32 sp, base, handed; u64 pts[VDEC_QN]; u8 end[VDEC_QN]; u32 nin, hin; u32 img[32]; u32 nimg; u32 w, h; KEvent* ev;
  u32 oimg[64]; u64 opts[64]; u8 oend[64]; u32 nout, hout;
  u8* ind[VDEC_QN]; u32 inn[VDEC_QN]; PCond cv; int thr; volatile int quit; int reset, vp8; u32 gen; } Vdec;
/* "decode": a submitted frame goes into the next free image right away and that image's sync value is reached (the
   client waits for it before asking for the output info) */
static void vdec_pump(Vdec* v){
  while(v->nin && v->nimg && v->nout<64){ u32 i=v->hin; v->hin=(v->hin+1)%VDEC_QN; v->nin--; u32 img=v->img[0]; memmove(v->img,v->img+1,(v->nimg-1)*4); v->nimg--;
    u32 o=(v->hout+v->nout)%64; v->oimg[o]=img; v->opts[o]=v->pts[i]; v->oend[o]=v->end[i]; v->nout++;
    extern void syncpt_incr(u32); syncpt_incr(v->sp); } }
#ifdef SWR_HAVE_VP9
#include "vpx/vpx_decoder.h"
#include "vpx/vp8dx.h"
/* The real decoder: one thread per session, as Eglantine's nx2_video_decoder (row 148 of its notes). Compressed frames are copied from the A buffer of 109/400 and decoded in order with libvpx; a picture goes
   into the next image the client gave (311/314) as NV12, BGRA or RGBA, and only then does the session's syncpoint move, so the client reads a finished image. SWR_VDEC=0: back to the protocol alone */
__attribute__((weak)) void dm_guest_write_prepare(void *addr, size_t len);
extern u64 nvmap_addr_by_id(u32 id);
static int vdec_real(void){ static int on=-1; if(on<0){ const char* e=getenv("SWR_VDEC"); on= !e || *e!='0'; } return on; }
static void vdec_write(u32 id, const vpx_image_t* f){
  DImg* d=NULL; for(int i=0;i<256;i++) if(g_dimg[i].id==id && g_dimg[i].w){ d=&g_dimg[i]; break; }
  if(!d || !f) return;
  u64 base=nvmap_addr(d->mem); if(!base) base=nvmap_addr_by_id(d->mem); if(!base){ static int nl; if(nl++<4) WARN("vdec: image %u has no memory (memory %u)\n",id,d->mem); return; }
  DImgLayout L=dimg_layout(d->code,d->w,d->h); u8* dst=g_mem+base+d->off;
  u32 w=f->d_w<d->w? f->d_w : d->w, h=f->d_h<d->h? f->d_h : d->h;
  if(dm_guest_write_prepare) dm_guest_write_prepare(dst,(size_t)L.size);
  const u8* py=f->planes[VPX_PLANE_Y]; const u8* pu=f->planes[VPX_PLANE_U]; const u8* pv=f->planes[VPX_PLANE_V];
  if(L.nv12){
    for(u32 y=0;y<h;y++) memcpy(dst+y*L.pitch,py+(size_t)y*f->stride[VPX_PLANE_Y],w);
    u8* uv=dst+L.chroma; u32 cw=(w+1)/2, ch=(h+1)/2;
    for(u32 y=0;y<ch;y++){ u8* o=uv+y*L.pitch; const u8* a=pu+(size_t)y*f->stride[VPX_PLANE_U]; const u8* b=pv+(size_t)y*f->stride[VPX_PLANE_V]; for(u32 x=0;x<cw;x++){ o[2*x]=a[x]; o[2*x+1]=b[x]; } }
    return; }
  int bgra= d->code==2 || d->code==5;   /* BT.709, limited range */
  for(u32 y=0;y<h;y++){ u8* o=dst+y*L.pitch; const u8* ly=py+(size_t)y*f->stride[VPX_PLANE_Y]; const u8* lu=pu+(size_t)(y/2)*f->stride[VPX_PLANE_U]; const u8* lv=pv+(size_t)(y/2)*f->stride[VPX_PLANE_V];
    for(u32 x=0;x<w;x++){ int Y=(ly[x]-16)*298, U=lu[x/2]-128, V=lv[x/2]-128; int r=(Y+459*V+128)>>8, g=(Y-55*U-136*V+128)>>8, b=(Y+541*U+128)>>8;
      r=r<0?0:r>255?255:r; g=g<0?0:g>255?255:g; b=b<0?0:b>255?255:b; o[4*x]=(u8)(bgra? b:r); o[4*x+1]=(u8)g; o[4*x+2]=(u8)(bgra? r:b); o[4*x+3]=255; } } }
/* lock held: one more picture is out. The syncpoint moves last: the client looks at the output only when it has */
static void vdec_out(Vdec* v, u32 img, u64 pts, u8 end){ extern void syncpt_incr(u32); g_vdec_out_ns=plat_ns(); u32 o=(v->hout+v->nout)%64; v->oimg[o]=img; v->opts[o]=pts; v->oend[o]=end; v->nout++; syncpt_incr(v->sp); }
static void vdec_thread(void* arg){ Vdec* v=(Vdec*)arg; vpx_codec_ctx_t ctx; int open=0; u64 frames=0;
  ns2_lock();
  for(;;){
    while(!v->nin && !v->quit && !v->reset) pc_wait(&v->cv,&g_ns2,20000000);
    if(v->quit) break;
    if(v->reset){ v->reset=0; if(open){ vpx_codec_destroy(&ctx); open=0; } continue; }
    u32 i=v->hin; v->hin=(v->hin+1)%VDEC_QN; v->nin--; u8* data=v->ind[i]; u32 n=v->inn[i]; u64 pts=v->pts[i]; u8 end=v->end[i]; v->ind[i]=NULL; u32 gen=v->gen;
    ns2_unlock();
    if(!open){ vpx_codec_dec_cfg_t cfg; memset(&cfg,0,sizeof cfg); cfg.threads=4; open= !vpx_codec_dec_init(&ctx,v->vp8? vpx_codec_vp8_dx() : vpx_codec_vp9_dx(),&cfg,0);
      if(!open) WARN("vdec: libvpx would not open: %s\n",vpx_codec_error(&ctx)); }
    const vpx_image_t* f=NULL; vpx_codec_iter_t it=NULL; u64 t0=plat_ns();
    if(open && data && n){ if(vpx_codec_decode(&ctx,data,n,NULL,0)){ static int nl; if(nl++<4) WARN("vdec: frame not decoded: %s\n",vpx_codec_error(&ctx)); } else f=vpx_codec_get_frame(&ctx,&it); }
    free(data); u64 t1=plat_ns();
    ns2_lock();
    if(f || end){   /* one image per picture: the thread holds the picture until the client has given an image */
      while(!v->nimg && !v->quit && !v->reset && v->gen==gen) pc_wait(&v->cv,&g_ns2,20000000);
      if(v->quit) break; if(v->reset || v->gen!=gen) continue;
      while(v->nout>=64 && !v->quit && !v->reset && v->gen==gen) pc_wait(&v->cv,&g_ns2,20000000);
      if(v->quit) break; if(v->reset || v->gen!=gen) continue;
      u32 img=v->img[0]; memmove(v->img,v->img+1,(v->nimg-1)*4); v->nimg--;
      u64 t2=plat_ns(); ns2_unlock(); vdec_write(img,f); ns2_lock();
      if(v->quit) break;
      { static u64 dec, wai, wri, nn, w0; u64 t3=plat_ns(); dec+=t1-t0; wai+=t2-t1; wri+=t3-t2; if(++nn==300){ LOG("vdec: last 300 pictures: decoding %.1f ms each, waiting for an image %.1f ms, writing %.1f ms, %.1f pictures a second\n",dec/3e8,wai/3e8,wri/3e8,w0? 3e11/(double)(t3-w0) : 0.0); dec=wai=wri=nn=0; w0=t3; } }
      vdec_out(v,img,pts,end);
      frames++; { static int nl; if(nl<12 && (frames==1 || !(frames%300) || end)){ nl++; LOG("vdec: picture %llu, %ux%u, pts %.2f s%s\n",(unsigned long long)frames,f? f->d_w : 0,f? f->d_h : 0,pts/1e6,end? ", end of stream" : ""); } }
      if(v->ev) kevent_signal(v->ev); } }
  /* the session is gone: this thread owns its memory now (vdec_svc_destroy) */
  for(u32 i=0;i<VDEC_QN;i++) free(v->ind[i]);
  ns2_unlock(); if(open) vpx_codec_destroy(&ctx);
  LOG("vdec: session closed after %llu pictures, its thread ends\n",(unsigned long long)frames); free(v); }
#endif
/* Closing the session frees o->data. With a decoder thread running that memory is the thread's until it ends: it was left writing into freed memory (heap corruption, c0000374) */
static void vdec_svc_destroy(SvcObj* o){ Vdec* v=(Vdec*)o->data; if(v && v->thr){ v->quit=1; o->data=NULL; } }
static int vdec_svc_disp(SvcObj* o, IpcCtx* c){
  Vdec* v=(Vdec*)o->data; int rc=0;
  ns2_lock();
  if(!v->sp){ extern u32 syncpt_alloc(void); extern u32 syncpt_read(u32); v->sp=syncpt_alloc(); v->base=syncpt_read(v->sp); }
  { static u64 t0, cnt[8]; static const u32 ids[8]={109,213,311,314,212,8,207,400}; u64 now=plat_ns();
    for(int k=0;k<8;k++) if(c->cmd==ids[k]) cnt[k]++;
    if(!t0) t0=now; if(now-t0>10000000000ull){ t0=now; LOG("vdec: 10 s: submit %llu, output info %llu, register %llu, release %llu, unregister %llu, flush %llu, stop %llu | queued %u, ready %u, free images %u\n",
      (unsigned long long)(cnt[0]+cnt[7]),(unsigned long long)cnt[1],(unsigned long long)cnt[2],(unsigned long long)cnt[3],(unsigned long long)cnt[4],(unsigned long long)cnt[5],(unsigned long long)cnt[6],v->nin,v->nout,v->nimg); memset(cnt,0,sizeof cnt); } }
  { static int nl; if(nl<80 && c->cmd!=213){ nl++; LOG("vdec: cmd %u (words %x %x %x %x %x %x), queued %u, images %u\n",(unsigned)c->cmd,W(c,2),W(c,3),W(c,4),W(c,5),W(c,6),W(c,7),v->nin,v->nimg); } }
  switch(c->cmd){
    case 3: if(!v->ev){ v->ev=kevent_new("vdec ready",0); kobj_ref(&v->ev->o); for(int i=0;i<16;i++) if(!g_vdec_ev[i]){ g_vdec_ev[i]=v->ev; break; } }
      kevent_signal(v->ev); out_copy_h(c,ev_h(v->ev)); break;
    case 101: v->w=W(c,5); v->h=W(c,6); { static int nl; if(nl<4){ nl++; LOG("vdec: configured %ux%u\n",v->w,v->h); } }
#ifdef SWR_HAVE_VP9
      if(vdec_real() && !v->thr){ char mime[16]={0}; u64 na=(u64)W(c,3)|(u64)((W(c,4)>>28)&15)<<32|(u64)((W(c,4)>>2)&7)<<36; u32 nn=W(c,2)<15? W(c,2) : 15; if(na>=AS_BASE && na<AS_SIZE) memcpy(mime,g_mem+na,nn);
        if(strstr(mime,"vp9") || strstr(mime,"vp8")){ v->vp8= strstr(mime,"vp8")!=NULL; v->thr=1; pc_init(&v->cv); plat_thread(vdec_thread,v,1<<20); LOG("vdec: decoding %s on a thread of its own\n",mime); }
        else { static int nl; if(nl<4){ nl++; WARN("vdec: %s is not decoded: this session gives no picture\n",mime); } } }
#endif
      break;
    case 109: case 400: { u32 na=(W(c,0)>>20)&15, k=2+3*na; u64 pts=W64(c,k); u8 end=(u8)(W(c,k+2)&0xFF);
      if(v->nin<VDEC_QN){ u32 i=(v->hin+v->nin)%VDEC_QN; v->pts[i]=pts; v->end[i]=end; v->nin++;
#ifdef SWR_HAVE_VP9
        if(v->thr){ u64 ad=(u64)W(c,3)|(u64)((W(c,4)>>28)&15)<<32|(u64)((W(c,4)>>2)&7)<<36; u32 sz=W(c,2); v->ind[i]=NULL; v->inn[i]=0; if(sz && ad>=AS_BASE && ad<AS_SIZE && (v->ind[i]=malloc(sz))){ memcpy(v->ind[i],g_mem+ad,sz); v->inn[i]=sz; } pc_signal(&v->cv); }
#endif
      } if(!v->thr) vdec_pump(v);
      static u64 id; id++; out32(c,(u32)id); out32(c,(u32)(id>>32)); if(v->ev) kevent_signal(v->ev); break; }
    case 311: case 314: { u32 img=W(c,2); if(v->nimg<32) v->img[v->nimg++]=img; v->handed++;
      out32(c,~v->sp); out32(c,0); out32(c,v->base+v->handed); out32(c,0); if(v->thr) pc_signal(&v->cv); else vdec_pump(v); break; }
    case 212: { u32 img=W(c,2); for(u32 i=0;i<v->nimg;i++) if(v->img[i]==img){ memmove(v->img+i,v->img+i+1,(v->nimg-i-1)*4); v->nimg--; break; } break; }
    case 213: { static int nl; if(nl<20){ nl++; LOG("vdec: 213 with %u outputs ready, queued %u, images %u, syncpoint %u = %u\n",v->nout,v->nin,v->nimg,v->sp,syncpt_read(v->sp)); } }
      if(v->nout){ u32 o=v->hout; v->hout=(v->hout+1)%64; v->nout--; u64 ns=v->opts[o]*1000;
        { static u64 nout; nout++; if(nout%200==0 || v->oend[o]){ static int nl; if(nl<40){ nl++; LOG("vdec: %llu frames out (pts %.2f s)%s\n",(unsigned long long)nout,ns/1e9,v->oend[o]? ", end of stream" : ""); } } }
        u32 r[16]={v->oimg[o],v->w,v->h,v->w,v->h,0,(u32)ns,(u32)(ns>>32),v->oend[o]? 1u : 0u,0,0,0,0,0,0,0}; for(int k=0;k<16;k++) out32(c,r[k]); }
      else rc=0x1c286f;   /* nothing finished: what the client's own Acquire answers for an empty ring */
      break;
    case 8: if(v->thr){ for(u32 i=0;i<v->nin;i++){ u32 k=(v->hin+i)%VDEC_QN; free(v->ind[k]); v->ind[k]=NULL; } v->gen++; v->reset=1; pc_signal(&v->cv); } v->nin=0; break;   /* Flush: frames already finished stay, the client saw their syncpoints reached and will take them (dropped, its 213 found nothing and it aborted) */
    case 207: if(v->thr){ for(u32 i=0;i<v->nin;i++){ u32 k=(v->hin+i)%VDEC_QN; free(v->ind[k]); v->ind[k]=NULL; } v->gen++; v->reset=1; pc_signal(&v->cv); } v->nin=0; v->nout=0; break;   /* Stop */
    default: break;   /* Start, Finalize, Initialize*: ok */
  }
  ns2_unlock(); return rc; }
static int vdec_disp(SvcObj* o, IpcCtx* c){
  if(c->cmd==1){ SvcObj* so=svcobj_new("IVideoDecoderService",vdec_svc_disp,sizeof(Vdec)); so->destroy=vdec_svc_destroy; return move_obj(c,so,0); }
  return ns2_unknown(o,c); }

static int vi2_disp(SvcObj* o, IpcCtx* c){   /* IVi2Session */
  Vi2* v=(Vi2*)o->data; ns2_lock(); int rc=0; u64 tok=W64(c,2); Vi2Client* s=vi2_find(v,tok);
  switch(c->cmd){
    case 1: case 3: case 5: out32(c,c->cmd==5? 7 : 0); break;           /* notification / session / request group limits */
    case 11: if(!v->ev){ v->ev=kevent_new("vi2 notification",0); kobj_ref(&v->ev->o); }   /* GetNotificationEvent */
      { int any=0; for(int i=0;i<32;i++) any|=v->notes[i]!=0; if(any) kevent_signal(v->ev); }
      out_move_h(c,ev_h(v->ev)); break;
    case 13: break;                                                     /* ReleaseNotificationEvent */
    case 31: case 32: { int i=c->cmd-31; sc_start(); if(!g_vsync[i]){ g_vsync[i]=kevent_new(i? "vi2 vsync 32" : "vi2 vsync 31",0); kobj_ref(&g_vsync[i]->o); }
      out_move_h(c,ev_h(g_vsync[i])); break; }                            /* vsync events, signalled at 60 Hz */
    case 21: { u64 a,sz; buf_out(c,0,&a,&sz); if(sz<sizeof v->notes){ rc=K_INVSIZE; break; }   /* GetNotifications */
      memcpy(GP(a),v->notes,sizeof v->notes); memset(v->notes,0,sizeof v->notes); if(v->ev) kevent_clear(v->ev); break; }
    case 51: rc= W(c,2)<8? 0 : K_INVCOMB; break;                        /* ConfigureRequestGroups */
    case 101: { u64 name=W64(c,2); u32 note=W(c,4);                       /* OpenDisplay */
      if(name!=NAME_DEFAULT){ rc=0; break; }   /* GameShare's player displays: accepted, nothing shown there */
      if(note!=1){ rc=K_NOTIMPL; break; } v->display_refs++; v->display_note=note; break; }
    case 102: if(W64(c,2)==NAME_DEFAULT && v->display_refs){ if(!--v->display_refs) vi2_note(v,v->display_note); } break;   /* CloseDisplay */
    case 111: if(!s && tok==NAME_DEFAULT){ v->endpoint=1; v->endpoint_token=tok; break; }   /* MKW opens it by display name ("default", "default") */
      if(!s || !s->has_display || v->endpoint){ rc= s? K_INVSTATE : K_NOTFOUND; break; } v->endpoint=1; v->endpoint_token=tok; break;   /* OpenEndpoint */
    case 112: if(!v->endpoint){ rc=K_NOTFOUND; break; } v->endpoint=0; break;   /* CloseEndpoint */
    /* display queries (name "default"), ~75 times a second in MKW. Answered with no output words, the
       game read its paper white and HDR status from the message buffer's leftovers */
    case 103: out32(c,60); out32(c,0); break;                         /* QueryDisplayRefreshRate: Hz, then a bool */
    case 104: out64(c,1); out32(c,400); break;                        /* QueryDisplayHdrStatus: format 1 = SDR, peak nits */
    case 105: out32(c,203); break;                                    /* QueryDisplayHdrPaperWhite: nits (BT.2408 reference white) */
    case 201: { u32 a=W(c,4), b=W(c,5);                                 /* RegisterClientState {token, retire, available} */
      if(!tok || a>=32 || b>=32 || a==b){ rc=K_INVCOMB; break; } u32 sl=(1u<<a)|(1u<<b);
      if(s || (v->client_slots&sl)){ rc=K_INVSTATE; break; }
      Vi2Client* f=vi2_find(v,0); for(int i=0;i<16 && !f;i++) if(!v->cl[i].token) f=&v->cl[i];
      if(!f){ rc=K_OUTOFRES; break; }
      memset(f,0,sizeof *f); f->token=tok; f->slots=sl; f->retire_note=a; f->avail_note=b; f->available=0x7FFF; f->encoding=0x14; v->client_slots|=sl; break; }
    case 202: if(!s){ rc=K_NOTFOUND; break; } vi2_note(v,s->retire_note); v->client_slots&=~s->slots; memset(s,0,sizeof *s); break;   /* UnregisterClientState */
    case 212: if(!s){ rc=K_NOTFOUND; out64(c,0); break; } out64(c,s->available); break;   /* GetClientAvailability */
    case 228: { u32 slot=W(c,4)&0xFF; if(!s || slot>=15){ rc=K_NOTFOUND; } else s->available|=1ull<<slot;   /* CancelClientBuffer */
      out64(c,s? s->available : 0); for(int i=0;i<4;i++) out32(c,0); break; }
    case 229: { if(!s) rc=K_NOTFOUND; else { s->available&=~(1ull<<(W(c,4)&0xFF)); ns2_unlock(); rc=vi2_present(v,s,c); ns2_lock(); }   /* PresentClientBuffer */
      out64(c,s? s->available : 0); for(int i=0;i<4;i++) out32(c,0); break; }
    case 251: if(!s){ rc=K_NOTFOUND; break; } if(W64(c,4)==NAME_DEFAULT && !v->display_refs){ rc=K_INVSTATE; break; } s->has_display=1; break;   /* BindClientDisplay */
    case 252: if(!s || !s->has_display){ rc=K_NOTFOUND; break; } s->has_display=0; break;   /* UnbindClientDisplay */
    case 351: if(!s){ rc=K_NOTFOUND; break; } rc= W64(c,4)==0x14? 0 : K_NOTIMPL; break;   /* SetClientOutputEncoding */
    default: ns2_unlock(); return ns2_unknown(o,c);
  }
  if(rc){ static int nl; if(nl<40){ nl++; LOG("vi2: cmd %u failed (%x), token %llx, words %x %x %x %x\n",(unsigned)c->cmd,rc,(unsigned long long)tok,W(c,2),W(c,3),W(c,4),W(c,5)); } }
  ns2_unlock(); return rc; }

/* ============================================================ IGraphicsSupervisor */
static int gsv_disp(SvcObj* o, IpcCtx* c){
  int cmd=(int)c->cmd, rc=0;
  ns2_lock();
  switch(cmd){
    case 4: { u64 flags=W64(c,6); rc= g_phase? K_INVSTATE : 0; if(!rc) g_phase=1;   /* Initialize {process, aruid, flags, remote memory} */
      LOG("switch 2: graphics supervisor bootstrap (flags %llx)\n",(unsigned long long)flags); break; }
    case 11: rc= g_phase==1? 0 : K_INVSTATE; if(!rc) g_phase=2; break;  /* BindProcess */
    case 101: rc= g_phase==2? 0 : K_INVSTATE; if(!rc){ g_phase=3; LOG("switch 2: graphics supervisor ready\n"); } break;   /* CompleteBootstrap */
    case 111: { u64 sz=W64(c,3); u32 id=0;                             /* CreateNativeMemory {kind, size} */
      if(g_phase!=3) rc=K_INVSTATE; else if(!sz || sz>(1ull<<32) || (sz&0xFFF)) rc=K_INVSIZE; else if(g_nnative>=1024) rc=K_OUTOFRES;
      else { id=nv_native_handle(0,sz); if(!id) rc=K_OUTOFRES; else g_native_mem[g_nnative++]=id; }
      out32(c,id); break; }
    case 112: { u32 id=W(c,2); rc=K_NOTFOUND; for(u32 i=0;i<g_nnative;i++) if(g_native_mem[i]==id){ nv_native_free(id); g_native_mem[i]=g_native_mem[--g_nnative]; rc=0; break; } break; }   /* DestroyNativeMemory */
    case 114: { u32 id=W(c,2); u64 a=W64(c,3), sz=W64(c,5), fl=W64(c,7), off=W64(c,9);   /* BindNativeMemory */
      if(!native_mem_has(id)) rc=K_NOTFOUND; else if((fl!=0 && fl!=2) || off) rc=K_NOTIMPL; else if(!a || sz!=nvmap_size(id)) rc=K_INVSIZE;
      else if(a&0xFFF) rc=K_INVADDR; else if(fl==2 && !gpu_accessible(a,a+sz-1)) rc=K_INVMEM; else nv_native_bind(id,a);
      break; }
    case 115: rc= g_phase==3? 0 : K_INVSTATE; out64(c,0); break;       /* ConfigureNativeMemory: the nvdrv-compatible mode */
    case 201: rc= g_phase==3? 0 : K_INVSTATE; if(!rc) g_sync_ctx=1; break;   /* InitializeSyncpointContext */
    case 202: if(!g_sync_ctx){ rc=K_INVSTATE; break; } rc=move_obj(c,svcobj_new("ISyncpointSupervisor",soft_sync_disp,0),0); break;   /* GetSyncpointSession */
    case 204: { /* BindSyncpointMemory: two transfer memories, 0x1000 bytes written by the game, 0x2000 read by it */
      KTmem* rw= c->ncopy>0? (KTmem*)h_get_type(c->copy_h[0],KO_TMEM) : NULL; KTmem* ro= c->ncopy>1? (KTmem*)h_get_type(c->copy_h[1],KO_TMEM) : NULL;
      if(!g_sync_ctx || g_sc_rw) rc=K_INVSTATE; else if(!rw || !ro || rw==ro) rc=0xE401; else if(rw->size!=0x1000 || ro->size!=0x2000) rc=K_INVSIZE;
      if(rc){ out_move_h(c,0); out_move_h(c,0); break; }
      memset(GP(rw->addr),0,0x1000); memset(GP(ro->addr),0,0x2000);
      static const u32 hdr[8]={1,0,SC_N-1,0x40,0,0x640,8,0x648}; memcpy(GP(ro->addr),hdr,sizeof hdr);
      for(u32 i=0;i<SC_N;i++){ gw64(ro->addr+0x40+i*8,~0ull); g_sc_prev[i]=~0ull; }
      g_sc_rw=rw->addr; g_sc_ro=ro->addr; g_sc_req=kevent_new("syncpoint cache request",0); g_sc_done=kevent_new("syncpoint cache",0);
      kobj_ref(&g_sc_req->o); kobj_ref(&g_sc_done->o); sc_start();
      out_move_h(c,ev_h(g_sc_req)); out_move_h(c,ev_h(g_sc_done));
      LOG("switch 2: syncpoint cache at %llx / %llx\n",(unsigned long long)g_sc_rw,(unsigned long long)g_sc_ro); break; }
    case 301: rc= g_phase==3? 0 : K_INVSTATE; if(!rc) g_disp_ctx=1; break;   /* InitializeDisplayContext */
    case 305: if(!g_disp_ctx){ rc=K_INVSTATE; out_move_h(c,0); break; }       /* OpenDisplaySession */
      rc=move_obj(c,svcobj_new("IVi2Session",soft_vi2_disp,sizeof(Vi2)),0); break;
    case 314: case 315: { NsImage* im=image_find(W(c,2)); int create=cmd==315;   /* {Destroy,Create}NativeWindowAliasImagePool */
      if(!im) rc=K_NOTFOUND; else if(create && W(c,3)!=1) rc=K_NOTIMPL; else if(im->window_pool==create) rc=K_INVSTATE; else im->window_pool=create; break; }
    case 401: rc= g_phase==3? 0 : K_INVSTATE; if(!rc) g_gpu_ctx=1; break;   /* InitializeGpuContext */
    case 402: rc= g_gpu_ctx? 0 : K_INVSTATE; if(!rc) g_gpu_ready=1; break;  /* ActivateGpuContext */
    case 423: { u32 id=W(c,2), parent=W(c,7); u64 a=W64(c,3), sz=W64(c,5), off=W64(c,8);   /* BindNativeGpuMemory: a range of other native memory */
      u64 pa=nvmap_addr(parent);
      if(!pa || !native_mem_has(id) || off>nvmap_size(parent) || sz>nvmap_size(parent)-off || a!=pa+off) rc=K_NOTFOUND;
      else if(sz!=nvmap_size(id)) rc=K_INVSIZE; else nv_native_bind(id,a);
      break; }
    case 429: { u32 id=0; u64 sz=0;                                    /* CreateNativeImage {preference[16], count, memory} */
      u32 p[16]; for(int i=0;i<16;i++) p[i]=W(c,2+i);
      if(g_phase!=3 || !g_gpu_ready || !img_formats_ok()) rc=K_INVSTATE;
      else if(!image_size_any(p,W(c,18),&sz,NULL)) rc=K_NOTIMPL;
      else if(!native_mem_has(W(c,19)) || !nvmap_addr(W(c,19))) rc=K_NOTFOUND;
      else if(nvmap_size(W(c,19))<sz) rc=K_INVSIZE; else if(g_nimg>=1024) rc=K_OUTOFRES;
      else { NsImage* im=&g_img[g_nimg++]; memset(im,0,sizeof *im); im->id=id=g_next_img++; im->mem=W(c,19); memcpy(im->pref,p,sizeof p); im->size=sz; }
      { static int said; if(said<sw2_cap(4)){ said++; LOG("switch 2: native image %u: %ux%u, pitch %u GOBs, memory %u (%x)\n",id,(p[2]&0xFFFF)+1,(p[2]>>16)+1,p[5],W(c,19),rc); } }
      out32(c,id); break; }
    case 430: { NsImage* im=image_find(W(c,2)); if(!im) rc=K_NOTFOUND; else if(im->window_pool) rc=K_BUSY; else *im=g_img[--g_nimg]; break; }   /* DestroyNativeImage */
    case 434: { u32 id=W(c,2); rc= g_gpu_ready? K_NOTFOUND : K_INVSTATE;   /* RegisterGpuAddressSpace */
      if(g_gpu_ready) for(int i=0;i<8;i++) if(g_dev.as[i]==(int)id){ g_dev.as_reg[i]=1; rc=0; } break; }
    case 435: case 1510: { u32 p[16]; for(int i=0;i<16;i++) p[i]=W(c,2+i); u64 sz=0, al=0;   /* CalculateNativeImageSize / ImageRequirements */
      if(g_phase!=3 || !img_formats_ok()) rc=K_INVSTATE; else if(!image_size_any(p,W(c,18),&sz,&al)) rc=K_NOTIMPL;
      { static int said; if(said<sw2_cap(2)){ said++; LOG("switch 2: %s {",cmd==1510? "image requirements" : "native image size"); for(int i=0;i<16;i++) LOG("%s%x",i? " " : "",p[i]); LOG("} count %u -> size %llx align %llx (%x)\n",W(c,18),(unsigned long long)sz,(unsigned long long)al,rc); } }
      out64(c,sz); if(cmd==1510) out64(c,al); break; }
    case 497: if(g_phase!=3){ rc=K_INVSTATE; out_move_h(c,0); break; }       /* OpenGpuSession */
      rc=move_obj(c,svcobj_new("IGpuSession",soft_gpusess_disp,0),0); break;
    /* 1201-1203: the chat (GameChat camera) context, nn::chat::gsv::Initialize. 1202 hands out a session the sdk
       calls next (it sent requests to handle 0 without one); everything on it answers ok for now */
    /* 601/602: another context with its session (the game sends requests on 602's right away); the session answers
       ok to everything for now, its commands show up in the log */
    case 901: case 1001: break;                                        /* the video decoder's contexts */
    case 902: case 906: case 1002: rc=move_obj(c,svcobj_new("IVideoDecoderSession",vdec_disp,0),0); break;
    /* 991: nn::codec::CalculateDecoderImagePoolBuffer takes the same 16-byte preference as 911 (code at word 3, width and height - 1 at word 4)
       and answers the memory for one frame: block-linear NV12 planes with their heights rounded to 128 lines, so 640x360 -> 0x64000 and
       2560x1440 -> 0x5a0000. Answering 640x360 for every movie left the game a pool too small for the big one, and the decoder wrote over what lay behind it */
    case 991: { u32 code=W(c,3), w=(W(c,4)&0xFFFF)+1, h=(W(c,4)>>16)+1; DImgLayout L=dimg_layout(code,w,h);
      { static int said; if(said<sw2_cap(2)){ said++; LOG("switch 2: decoder image size {%x %x %x %x} -> %ux%u, %llx bytes\n",W(c,2),W(c,3),W(c,4),W(c,5),w,h,(unsigned long long)L.size); } }
      out64(c,L.size); out64(c,0x10000); break; }
    /* 911 CreateDecoderImage {preference[16], u32 count, u32 memory}: the id of the first of `count` decoder output
       images, the client names image i pool id + i. Answered 0, every image of MKW's movie pools was image 0 and its
       movie player stopped after two frames (from Eglantine's notes). 912 destroys one */
    case 911: { u32 n=W(c,18); if(!n) n=1; if(n>64) n=64; u32 first=g_next_img; g_next_img+=n; out32(c,first);
      u32 code=W(c,3), w=(W(c,4)&0xFFFF)+1, h=(W(c,4)>>16)+1; DImgLayout L=dimg_layout(code,w,h);
      for(u32 i=0;i<n;i++){ DImg* d=&g_dimg[g_ndimg++%256]; d->id=first+i; d->code=code; d->w=w; d->h=h; d->mem=W(c,19); d->off=L.size*i; }
      { static int said; if(said<sw2_cap(4)){ said++; LOG("switch 2: decoder images %u..%u in memory %u, code %u, %ux%u\n",first,first+n-1,W(c,19),code,w,h); } } break; }
    case 912: break;
    /* 961 GetCpuImageDescriptor(u32 image): the 128-byte nn::gsv::CpuUserImageDescriptor ReadDecoderOutputImage copies
       an image out with: kind at +0 (8 RGBA, 9 BGRA, 10 NV12, pitch linear), plane offset +0x8 and pitch +0x10, NV12's
       chroma plane +0x18/+0x20. Zeros (kind 0) stopped MKW's movie player after two frames */
    case 961: { u32 r[32]; memset(r,0,sizeof r); u32 id=W(c,2);
      for(int i=0;i<256;i++) if(g_dimg[i].id==id && (g_dimg[i].w || g_dimg[i].h)){ DImg* d=&g_dimg[i]; DImgLayout L=dimg_layout(d->code,d->w,d->h);
        r[0]= L.nv12? 0xa : (d->code==2 || d->code==5)? 0x9 : 0x8; r[2]=(u32)d->off; r[4]=(u32)L.pitch;
        if(L.nv12){ r[6]=(u32)(d->off+L.chroma); r[8]=(u32)L.pitch; } break; }
      for(int k=0;k<32;k++) out32(c,r[k]); break; }
    case 601: break;
    case 602: rc=move_obj(c,svcobj_new("IGsvSession602",ns2_unknown,0),0); break;
    case 1201: break;
    /* 1211: the chat camera's image pool {preference[16], count, memory}: one native image in that memory, its id is
       the pool's (nn::chat::gsv::GetImagePoolImageHandle hands it to QueryNativeImage, 428) */
    /* the pool's images get consecutive ids (nn::gs::sdk::CalculateNativeImageHandle is the first one + index), each
       at its own offset in the memory: as many as fit (the memory came from 1291's size) */
    case 1211: { u32 p[16]; for(int i=0;i<16;i++) p[i]=W(c,2+i); u32 mem=0; u64 sz=0;
      for(int i=18;i<22;i++) if(native_mem_has(W(c,i))){ mem=W(c,i); break; }
      image_size_any(p,1,&sz,NULL); u64 msz= mem? nvmap_size(mem) : 0; if(!sz || sz>msz) sz=msz;
      u32 n= sz? (u32)(msz/sz) : 0; if(n>16) n=16;
      if(!mem || !n) rc=K_NOTFOUND; else if(g_nimg+n>1024) rc=K_OUTOFRES;
      else { u32 first=g_next_img; for(u32 i=0;i<n;i++){ NsImage* im=&g_img[g_nimg++]; memset(im,0,sizeof *im); im->id=g_next_img++; im->mem=mem; memcpy(im->pref,p,sizeof p); im->size=sz; im->off=(u64)i*sz; }
        out32(c,first); }
      { static int said; if(said<sw2_cap(2)){ said++; LOG("switch 2: chat image pool in memory %u: %u images of %llx bytes (%x)\n",mem,n,(unsigned long long)sz,rc); } }
      if(rc) out32(c,0); break; }
    /* 428: QueryNativeImage. The sdk reads the address at data+8, then a 128 byte descriptor: two kind bytes that have
       to be 1 (data+16, data+48), the image's offset and size (data+56) */
    /* 426: MapNativeImage {image, address space, permission, ...}: the same answer as 428 with the GPU address. GPU
       addresses are the CPU ones here, the image's memory gets mapped at its own address. 427 unmaps (ok) */
    case 426: case 428: { NsImage* im=image_find(W(c,2)); if(!im){ rc=K_NOTFOUND; break; }
      u64 a=nvmap_addr(im->mem); if(a) a+=im->off;
      if(cmd==426 && a){ for(u32 i=0;i<g_nmaps;i++) if(g_maps[i].gva==a){ a=0; break; }
        if(a && !nv_native_map(a,im->mem,im->off,(im->size+0xFFF)&~0xFFFull) && g_nmaps<4096) g_maps[g_nmaps++]=(GpuMap){W(c,3),a,(im->size+0xFFF)&~0xFFFull,im->mem};
        a=nvmap_addr(im->mem)+im->off; }
      /* descriptor (nvnTextureInitializeFromNativeImage reads it): byte 0 = 1, +8 the size word ((w-1) | (h-1) << 16),
         +32 = 1 with the image kind above it (the preference's p[1]: 1 for color formats, 2 for the YUV ones),
         +40 offset and size */
      u32 kind= im->pref[1]==1 || im->pref[1]==2? im->pref[1] : 1;
      out64(c,a); out64(c,a); out64(c,1); out64(c,im->pref[2]);   /* data+0: the GPU address (MapNativeImage keys its mappings by it) */ out64(c,0); out64(c,0); out64(c,1|((u64)kind<<32));
      out64(c,(u64)(u32)im->size<<32); for(int i=0;i<10;i++) out64(c,0); out32(c,0);
      { static int said; if(said<sw2_cap(4)){ said++; LOG("switch 2: native image %u %s (%llx, %llx bytes)\n",im->id,cmd==426? "mapped" : "queried",(unsigned long long)a,(unsigned long long)im->size); } }
      break; }
    case 427: break;
    /* 1291: the chat camera's video memory for an image preference ({0, 1, size, ...} like 1510's): a size that's
       surely big enough, so CreateNativeMemory (111) gets a real size instead of 0 */
    case 1291: { u32 p[16]; for(int i=0;i<16;i++) p[i]=W(c,2+i); u64 sz=0, al=0x10000;
      image_size_any(p,1,&sz,&al); if(!sz) sz=0x400000; sz=(sz*4+0xFFFF)&~0xFFFFull;
      { static int said; if(said<sw2_cap(2)){ said++; LOG("switch 2: chat video memory %ux%u -> %llx bytes\n",(p[2]&0xFFFF)+1,(p[2]>>16)+1,(unsigned long long)sz); } }
      out64(c,sz); out64(c,al); break; }
    case 1202: rc=move_obj(c,svcobj_new("IChatSession",chat_disp,0),0); break;
    case 1101: rc= g_phase==3? 0 : K_INVSTATE; if(!rc) g_frame_ctx=1; break;   /* InitializeFrameContext */
    case 1102: rc= g_frame_ctx? 0 : K_INVSTATE; break;                         /* ActivateFrameContext */
    /* GetNvidiaDriver: an nvdrv session. games that bootstrap with flag bit 0 (nvdrv mode) ask for it right after
       CompleteBootstrap, without setting up the image context first */
    case 1402: if(g_phase!=3){ rc=K_INVSTATE; break; }
      out_move_h(c,session_new(nvdrv_obj_new("nvdrv:gsv"))); LOG("switch 2: nvdrv session for the game's NVN\n"); break;
    case 1501: rc= g_phase==3? 0 : K_INVSTATE; if(!rc) g_img_ctx=1; break;   /* InitializeImageContext */
    case 1502: rc= g_img_ctx? 0 : K_INVSTATE; if(!rc) g_img_formats=1; break;
    default: ns2_unlock(); return ns2_unknown(o,c);
  }
  ns2_unlock();
  if(rc && ipc_warn_once("gsv-fail",(u32)cmd)) WARN("switch 2: graphics supervisor cmd %d -> %x\n",cmd,rc);
  return rc; }

/* ISelfController 2000: the session is made once and handed out as a copy handle */
#define NS2_SOFT(f) static int soft_##f(SvcObj* o, IpcCtx* c){ return ns2_soft(o,c,f(o,c)); }
NS2_SOFT(worker_disp) NS2_SOFT(sync_disp) NS2_SOFT(vi2_disp) NS2_SOFT(gpusess_disp) NS2_SOFT(gsv_disp)

void ns2_graphics_supervisor(IpcCtx* c){
  static u32 h; static SvcObj* o;
  if(!o){ o=svcobj_new("IGraphicsSupervisor",soft_gsv_disp,0); h=session_new(o); LOG("switch 2: graphics supervisor opened\n"); }
  out_copy_h(c,h_new(h_get(h))); }   /* a handle of its own each time, to the same session */
