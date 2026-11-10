/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
#include <stdio.h>
#include <stdlib.h>
/* engines: MME, 3D (regs, cb data, reports, clears), 2D, DMA, compute */
#include "gpu.h"

/* ============================================================ MME (macros) */
typedef struct { u32 regs[8]; u32 carry; u32 method, incr, idx; const u32* params; u32 np, pi; GpuChannel* ch; u32 pc; int delayed; u32 dpc; } Mme;

static void mme_send(Mme* m, u32 v){ maxwell3d_method(m->ch,m->method,v,1); m->method+=m->incr; m->method&=0xFFF; }

static void mme_setmethod(Mme* m, u32 v){ m->method=v&0xFFF; m->incr=(v>>12)&0x3F; }

static u32 mme_fetch(Mme* m){ return m->pi<m->np? m->params[m->pi++] : 0; }

static void mme_setreg(Mme* m, u32 r, u32 v){ if(r) m->regs[r]=v; }

static void mme_result(Mme* m, u32 op, u32 dst, u32 res){
  switch(op){
    case 0: mme_setreg(m,dst,mme_fetch(m)); break;
    case 1: mme_setreg(m,dst,res); break;
    case 2: mme_setreg(m,dst,res); mme_setmethod(m,res); break;
    case 3: mme_setreg(m,dst,mme_fetch(m)); mme_send(m,res); break;
    case 4: mme_setreg(m,dst,res); mme_send(m,res); break;
    case 5: mme_setreg(m,dst,mme_fetch(m)); mme_setmethod(m,res); break;
    case 6: mme_setreg(m,dst,res); mme_setmethod(m,res); mme_send(m,mme_fetch(m)); break;
    case 7: mme_setreg(m,dst,res); mme_setmethod(m,res); mme_send(m,(res>>12)&0x3F); break;
  }
}

static u32 mme_alu(Mme* m, u32 op, u32 a, u32 b){
  u64 r;
  switch(op){
    case 0: r=(u64)a+b; m->carry=(u32)(r>>32); return (u32)r;
    case 1: r=(u64)a+b+m->carry; m->carry=(u32)(r>>32); return (u32)r;
    case 2: r=(u64)a-b; m->carry=r<0x100000000ull; return (u32)r;
    case 3: r=(u64)a-b-(m->carry?0:1); m->carry=r<0x100000000ull; return (u32)r;
    case 8: return a^b; case 9: return a|b; case 10: return a&b; case 11: return a&~b; case 12: return ~(a&b);
  }
  { static int said; if(said<sw2_cap(8)){ said++; WARN("mme: alu %u (not a Maxwell ALU operation) in macro %u at pc %x\n",op,m->idx,m->pc-4); }
    static int dumped; if(!dumped){ dumped=1; u32 st=m->ch->mme_start[m->idx&0xFF]; LOG("mme: macro %u starts at word %x (upload pointer %x); code from there:\n",m->idx,st,m->ch->mme_upload_ptr);
      for(u32 i=0;i<64;i+=8){ const u32* c=&m->ch->mme_code[(st+i)&0x1FFF]; LOG("  %04x: %08x %08x %08x %08x %08x %08x %08x %08x\n",st+i,c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]); }
      LOG("  macro start table:"); for(u32 i=0;i<0x80;i++){ if(i%16==0) LOG("\n   "); LOG(" %x",m->ch->mme_start[i]); } LOG("\n"); } }
  return 0; }

static int mme_step(Mme* m, int delay){
  u32 base=m->pc; u32 w=m->ch->mme_code[(m->pc/4)&0x1FFF];
  if(m->delayed){ m->pc=m->dpc; m->delayed=0; } else m->pc+=4;
  u32 op=w&7, rop=(w>>4)&7, dst=(w>>8)&7, sa=(w>>11)&7, sb=(w>>14)&7;
  i32 imm=((i32)(w<<0))>>14; /* imm: 18 bits signed @14 */
  imm=(i32)(w>>14); if(imm&0x20000) imm|=~0x3FFFF;
  u32 bf_src=(w>>17)&31, bf_size=(w>>22)&31, bf_dst=(w>>27)&31; u32 mask=(1u<<bf_size)-1; if(bf_size==0) mask=0;
  switch(op){
    case 0: mme_result(m,rop,dst,mme_alu(m,(w>>17)&31,m->regs[sa],m->regs[sb])); break;
    case 1: mme_result(m,rop,dst,m->regs[sa]+(u32)imm); break;
    case 2: { u32 d=m->regs[sa], s=m->regs[sb]; s=(s>>bf_src)&mask; d&=~(mask<<bf_dst); d|=s<<bf_dst; mme_result(m,rop,dst,d); break; }
    case 3: { u32 d=m->regs[sa], s=m->regs[sb]; mme_result(m,rop,dst,((s>>d)&mask)<<bf_dst); break; }
    case 4: { u32 d=m->regs[sa], s=m->regs[sb]; mme_result(m,rop,dst,((s>>bf_src)&mask)<<d); break; }
    case 5: { u32 a=(m->regs[sa]+(u32)imm)&0xFFF; mme_result(m,rop,dst,m->ch->r3d[a]); break; }
    case 7: { u32 v=m->regs[sa]; int nz=(w>>4)&1, annul=(w>>5)&1; int taken= nz? v!=0 : v==0;
      if(taken){ if(annul){ m->pc=base+(u32)(imm*4); return 1; } m->delayed=1; m->dpc=base+(u32)(imm*4); mme_step(m,1); return 1; }
      return 1; }   /* bit 7 is not exit on branches (yuzu does the same) */
    default: WARN("mme: op %u\n",op); break;
  }
  if(((w>>7)&1) && !delay){ mme_step(m,1); return 0; }
  return 1; }

/* ---- TU104 MME (Turing and Ampere, Switch 2's 3D class): two ALUs and two method outputs per 96-bit instruction.
   Ported from Mesa's mme_tu104_sim.c / mme_tu104.c (Copyright 2022 Collabora Ltd., MIT license), with the
   instruction fields from mme_tu104_decode. The parameter FIFO can be refilled from memory (MME_DMA_READ_FIFOED,
   then the EXTENDED barrier). Turing method numbers from NVIDIA's clc597.h (MIT) */
typedef struct { GpuChannel* ch; const u32* p; u32 np, pi; u32 fifo[1024]; u32 dma_count; int comp; int ind; } Tu;   /* ind: loaded compute output (SWR_MME_IND_LOG) */   /* comp: the compute engine's MME */
static u32 tu_load(Tu* t){ return t->pi<t->np? t->p[t->pi++] : 0; }
static u32 tu_field(const u32 b[3], u32 lo, u32 n){ u64 v=b[lo/32]; if(lo/32<2) v|=(u64)b[lo/32+1]<<32; return (u32)((v>>(lo%32))&((1ull<<n)-1)); }
static void tu_barrier(Tu* t){   /* load the FIFO from SET_MME_MEM_ADDRESS (0x154/0x155), MME_DMA_READ_FIFOED (0x158) words */
  GpuChannel* ch=t->ch; u32 n=t->dma_count; if(!n) return; if(n>1024) n=1024;
  const u32* rg= t->comp? ch->rcomp : ch->r3d; u64 a=((u64)rg[0x154]<<32)|rg[0x155];
  { static int lg=-1, nl; if(lg<0) lg=getenv("SWR_MME_IND_LOG")!=NULL; extern int gpu_gl_cs_pending(u64,u64);
    if(lg && nl<12 && gpu_gl_cs_pending(a,(u64)n*4)){ nl++; t->ind=1; LOG("mme-ind: %s macro loads %u words of compute output at %llx\n",t->comp? "compute" : "3D",n,(unsigned long long)a); } }
  gmem_read(a,t->fifo,(u64)n*4); t->p=t->fifo; t->np=n; t->pi=0; t->dma_count=0; }
static void tu_mthd(Tu* t, u32 m, u32 v){
  GpuChannel* ch=t->ch;
  if(m==0x158){ t->dma_count=v; (t->comp? ch->rcomp : ch->r3d)[m]=v; return; }   /* MME_DMA_READ_FIFOED */
  if(m>=0xE00){ static int said; if(!said){ said=1; WARN("gpu: TU104 macro emits a macro call (%x), ignored\n",m); } return; }
  { static int lg=-1, n, cap; if(lg<0){ extern u64 g_title_id; lg= !getenv("SWR_TRACE_OFF") && (getenv("SWR_METH_LOG") || (g_title_id>>56)==0x04); cap= getenv("SWR_METH_LOG")? 40000 : 1000; }
    static int nc; if(lg && (t->comp? nc++<cap : n++<cap)) LOG("  %s-> m %x v %x\n",t->comp? "cmme" : "mme",m,v); }
  { static int on=-1, n2; static u64 t0, after; if(on<0){ const char* e=getenv("SWR_METH_AFTER"); on= e!=NULL; after= e? strtoull(e,NULL,10)*1000000000ull : 0; t0=plat_ns(); }
    if(on && !t->comp && n2<8000 && plat_ns()-t0>after){ n2++; LOG("  O ch%u %x %x\n",ch->id,m,v); } }
  if(t->ind) LOG("mme-ind:   -> m %x v %x\n",m,v);
  if(t->comp) compute_method(ch,m,v,1); else maxwell3d_method(ch,m,v,1); }
static void mme_tu104(GpuChannel* ch, u32 start, const u32* params, u32 np, int comp){
  Tu t; memset(&t,0,sizeof t); t.ch=ch; t.p=params; t.np=np; t.comp=comp;
  const u32* code=ch->mme_code; u32* dram=ch->mme_dram; const u32* state= comp? ch->rcomp : ch->r3d;
  u32 regs[24]={0}, ld[2]={0}, res[2]={0}, carry=0, mthd=0, inc=0, ip=start, loop_n=0, loop_s=0, loop_e=0; int end_next=0, stop=0;
  static const char* const pred_s[16]={"UUUU","TTTT","FFFF","TTUU","FFUU","TFUU","TUUU","FUUU","UUTT","UUTF","UUTU","UUFT","UUFF","UUFU","UUUT","UUUF"};
  for(u32 steps=0; steps<2000000; steps++){
    if(ip*3+2>=0x3000){ static int said; if(said<sw2_cap(4)){ said++; WARN("gpu: TU104 macro ran outside its code (ip %x)\n",ip); } return; }
    const u32 b[3]={code[ip*3+2],code[ip*3+1],code[ip*3]};
    u32 next=ip+1, imm[2]={tu_field(b,30,16),tu_field(b,66,16)}, op[2]={tu_field(b,10,5),tu_field(b,46,5)}, dst[2]={tu_field(b,15,5),tu_field(b,51,5)};
    u32 sa[2]={tu_field(b,20,5),tu_field(b,56,5)}, sb[2]={tu_field(b,25,5),tu_field(b,61,5)}, om[2]={tu_field(b,82,3),tu_field(b,89,3)}, oe[2]={tu_field(b,85,4),tu_field(b,92,4)};
    u32 pmode=tu_field(b,1,4), preg=tu_field(b,5,5);
    { static int tr=-1, tr3=-1, n3; if(tr<0) tr=getenv("SWR_MME_TRACE")!=NULL; if(tr3<0) tr3=getenv("SWR_MME_TRACE3D")!=NULL; if((tr && comp) || (tr3 && !comp && np==4 && params[0]==8 && n3++<300)) LOG("  cmme ip %x: op %u/%u dst %u/%u src %u,%u/%u,%u imm %x/%x out %u:%u %u:%u pred %u r%u\n",ip,op[0],op[1],dst[0],dst[1],sa[0],sb[0],sa[1],sb[1],imm[0],imm[1],om[0],oe[0],om[1],oe[1],pmode,preg); }
    for(int i=0;i<2;i++){ u32 r=28+i, o=3+i;   /* LOAD0/LOAD1 pop the FIFO when the instruction reads them */
      if(preg==r || sa[0]==r || sa[1]==r || sb[0]==r || sb[1]==r || om[0]==o || om[1]==o || oe[0]==o || oe[1]==o) ld[i]=tu_load(&t); }
    u32 in[24]; memcpy(in,regs,sizeof in);   /* both ALUs read the registers as they were before the instruction; ALU1
                                                seeing ALU0's write made NVN2's uniform-bind macro read the wrong data RAM word
                                                and skip every vertex stage bind (black UI in MKW) */
    #define TU_GET(r,lane) ((r)<24? in[r] : (r)==24? 0u : (r)==25? (u32)(i32)(i16)imm[lane] : (r)==26? (u32)(i32)(i16)imm[1-(lane)] : (r)==27? (imm[0]<<16)|imm[1] : (r)==28? ld[0] : (r)==29? ld[1] : 0u)
    u32 pm=0xF; if(pmode){ u32 pv=TU_GET(preg,0); pm=0; for(int i=0;i<4;i++) if(pred_s[pmode][i]!=(pv? 'T' : 'F')) pm|=1u<<i; }
    for(int a=0;a<2;a++){ if(!(pm>>a&1)) continue;
      u32 x=TU_GET(sa[a],a), y=TU_GET(sb[a],a), r=0;
      switch(op[a]){
        case 0: r=x+y; carry=r<x; break;                                    /* ADD */
        case 1: r=x+y+carry; break;                                         /* ADDC */
        case 2: r=x-y; carry=r>x; break;                                    /* SUB */
        case 3: r=x-y-carry; break;                                         /* SUBB */
        case 4: { u64 p=(u64)(i64)(i32)x*(u64)(i64)(i32)y; r=(u32)p; carry=(u32)(p>>32); break; }   /* MUL */
        case 5: r=carry; break;                                             /* MULH */
        case 6: { u64 p=(u64)x*y; r=(u32)p; carry=(u32)(p>>32); break; }     /* MULU */
        case 7: if(x==0x1000 && y==1) tu_barrier(&t); break;                /* EXTENDED: the READ_FIFOED barrier */
        case 8: r= x? (u32)__builtin_clz(x) : 32; break;
        case 9: r=x<<(y&31); break; case 10: r=x>>(y&31); break; case 11: r=(u32)((i32)x>>(y&31)); break;
        case 12: r=x&y; break; case 13: r=~(x&y); break; case 14: r=x|y; break; case 15: r=x^y; break;
        case 16: { u32 dp=(imm[a]>>10)&0x3F, nb=(imm[a]>>5)&0x1F, sp=imm[a]&0x1F, mk= nb>=32? ~0u : (1u<<nb)-1;   /* MERGE */
          r=(x&~(mk<<dp))|(((y>>sp)&mk)<<dp); break; }
        case 17: r=(i32)x<(i32)y? ~0u : 0; break; case 18: r=x<y? ~0u : 0; break;
        case 19: r=(i32)x<=(i32)y? ~0u : 0; break; case 20: r=x<=y? ~0u : 0; break; case 21: r=x==y? ~0u : 0; break;
        case 22: r=state[(u16)(x+y)&0xFFF]; break;                          /* STATE: a register of the engine (its shadow) */
        case 23: loop_n=(x? x : 1)-1; loop_s=ip; loop_e=ip+imm[a]-1; break;   /* LOOP: the body runs at least once */
        case 24: next=ip+(imm[a]&0x7FFF); r=0; break;                       /* JAL */
        case 25: case 26: case 27: case 28: case 29: {                      /* BLT BLTU BLE BLEU BEQ */
          int c= op[a]==25? (i32)x<(i32)y : op[a]==26? x<y : op[a]==27? (i32)x<=(i32)y : op[a]==28? x<=y : x==y;
          if(c==((imm[a]>>15)&1)){ i32 off=(i32)((u32)imm[a]<<19)>>19; if(off==-4096) stop=1; else next=ip+off; }
          break; }
        case 30: r=dram[x%0xC00]; break;                                    /* DREAD */
        case 31: dram[x%0xC00]=y; break;                                    /* DWRITE */
      }
      res[a]=r; if(dst[a]<24) regs[dst[a]]=r; }
    for(int o=0;o<2;o++){ if(!(pm>>(2+o)&1)) continue;
      #define TU_OUT(k) ((k)==1? res[0] : (k)==2? res[1] : (k)==3? ld[0] : (k)==4? ld[1] : (k)==5? imm[0] : (k)==6? imm[1] : (k)==8? imm[0]>>12 : (k)==9? imm[1]>>12 : (k)==10? (imm[0]<<16)|imm[1] : 0u)
      if(om[o]){ u32 d=TU_OUT(om[o]); mthd=d&0xFFF; inc=(d>>12)&0xF; }
      if(oe[o]){ tu_mthd(&t,mthd,TU_OUT(oe[o])); mthd=(mthd+inc)&0xFFF; } }
    #undef TU_GET
    #undef TU_OUT
    if(end_next || stop) return;
    end_next=tu_field(b,0,1);
    if(loop_n && ip==loop_e){ loop_n--; next=loop_s+1; }
    ip=next; }
  { static int said; if(said<sw2_cap(4)){ said++; WARN("gpu: TU104 macro at %x ran 2M steps, stopped\n",start); } } }

static void mme_run(GpuChannel* ch, u32 idx, const u32* params, u32 np){
  if(ch->ampere){ mme_tu104(ch,ch->mme_start[idx&0xFF],params,np,0); return; }
  Mme m; memset(&m,0,sizeof m); m.ch=ch; m.idx=idx; m.params=params; m.np=np; m.pi=1; m.regs[1]=np? params[0] : 0;
  m.pc=ch->mme_start[idx&0xFF]*4;
  for(int guard=0; guard<5000000; guard++) if(!mme_step(&m,0)) break;
}

/* ============================================================ Maxwell 3D */
#define R3(i) (ch->r3d[i])
extern void r3d_draw(GpuChannel* ch, int indexed, u32 topology, u32 first, u32 count, u32 base_vertex, u32 base_instance, u32 instances);
extern void r3d_clear(GpuChannel* ch, u32 v);
extern void r3d_const_upload(GpuChannel* ch, u32 v);
extern void r3d_cb_bind(GpuChannel* ch, int stage, u32 v);
extern u64 g_draws;
static u32 g_instance_count, g_draw_inline_count;
/* instancing comes in as N BEGIN/END pairs with the next instance bit. we keep adding them up and emit one draw of
   N instances once something else shows up */
static struct { int on, indexed; GpuChannel* ch; u32 topo, first, count, bv, bi, inst; } g_pd;

u64 g_draws_merged;

void r3d_pending_flush(void){ if(!g_pd.on) return; g_pd.on=0;
  r3d_draw(g_pd.ch,g_pd.indexed,g_pd.topo,g_pd.first,g_pd.count,g_pd.bv,g_pd.bi,g_pd.inst); }

int r3d_pending_on(void){ return g_pd.on; }

/* timestamps for report semaphores. games with DRS (smo with al::GpuPerf / agl::perf::GPUTimeStamp) time the GPU
   with these and drop resolution when it looks slow. our "GPU" timing means nothing for that, see below */
/* sead::GraphicsNvn::convertGPUTimeStampToSystemTick compares them with the CPU tick so they have to use the same
   base (guest_ns). we stamp each report with the time the work got submitted and run that clock 256x slower (yuzu's
   fast gpu time). submit times alone weren't enough: on a slow cpu the first and last submit of a frame are 30-50ms
   apart, smo took that as gpu time, dropped to half width and the half width path came out blue or black.
   game_dynamic_resolution=1 stamps real time instead */
volatile u64 g_gpu_sub_ns;

u64 gpu_ticks(void){
  extern int g_opt_drs; extern u64 guest_ns(void); static u64 last;
  u64 n= g_opt_drs? guest_ns() : (g_gpu_sub_ns? g_gpu_sub_ns : guest_ns())/256;
  if(n<=last) n=last+16; last=n;   /* monotonic, nudged by 1ns within a frame */
  return n*384/625; }

int gpu_gl_hold_write(u64 addr, const void* data, u32 len); int gpu_gl_hold_reduce(u64 addr, u32 op, u32 payload, int is_signed); void gpu_gl_before_syncwait(void);

static void query_get(GpuChannel* ch, u32* r, int base, u32 v){
  u64 addr=((u64)(r[base]&0xFF)<<32)|r[base+1]; u32 seq=r[base+2];
  u32 op=v&3;
  { extern void semdbg(const char*,u32,u64,u64,u32); semdbg(op==0? "rep-rel" : op==1? "rep-acq" : "rep-cnt",ch->id,addr,seq,v); }
  { static int nl; if(nl<sw2_cap(24)){ nl++; LOG("gpu: report semaphore op %u at %llx payload %x (word %x, now %x)\n",op,(unsigned long long)addr,seq,v,gmem_r32(addr)); } }
  if(op==0 && ((v>>3)&1)){ /* release with REDUCTION_ENABLE: atomic op on the word. 3D encoding: ADD MIN MAX INC DEC AND OR XOR */
    static const u8 map[8]={5,0,1,6,7,3,4,2}; extern void gpu_sem_reduce(u64,u32,u32,int);
    if(!gpu_gl_hold_reduce(addr,map[(v>>9)&7],seq,(v>>17)&1)) gpu_sem_reduce(addr,map[(v>>9)&7],seq,(v>>17)&1);
    return; }
  if(op==0){ /* release, after any compute results still on their way (gpu_gl_hold_write) */
    if(((v>>4)&1) || ((v>>28)&1)){ if(!gpu_gl_hold_write(addr,&seq,4)) gmem_w32(addr,seq); }
    else { u64 q[2]={seq,gpu_ticks()}; if(!gpu_gl_hold_write(addr,q,16)) gmem_write(addr,q,16); }
  } else if(op==1){ /* acquire */
    extern u32 gpu_gl_held_view(int,u64,u32);   /* a release held for compute results counts for the GPU */
    u64 end=plat_ns()+1000000000ull;
    u32 bo=0; while(gpu_gl_held_view(1,addr,gmem_r32(addr))!=seq && plat_ns()<end) plat_backoff(&bo);
  } else if(op==2){ /* counter */
    u32 sel=(v>>23)&31; u64 val= sel==0? 0 : (sel==1? g_draws*3 : 1000);
    if(sel==2 || sel==21 || sel==23){ static int said; if(!said){ said=1; LOG("gpu: the game reads occlusion queries (counter %u)\n",sel); } }
    if((v>>28)&1){ u32 w=(u32)val; if(!gpu_gl_hold_write(addr,&w,4)) gmem_w32(addr,w); }
    else { u64 q[2]={val,gpu_ticks()}; if(!gpu_gl_hold_write(addr,q,16)) gmem_write(addr,q,16); }
  }
}

/* ---- Ampere 3D (C797, Switch 2) on the Maxwell engine: same method numbers, with some Maxwell state now given another
   way (NVIDIA clc797.h). Those writes are turned into the Maxwell registers the renderer reads:
   - shader programs: an absolute address per stage (0x805/0x806 + 16*stage), not program region + offset
   - index buffer and vertex stream sizes (0x8E/0x8F, 0x180 + 2*stream) instead of limit addresses
   - draws in one method pair (DRAW_*_BEGIN_END, 0x9A-0x9D) with SET_DRAW_CONTROL_A/B (0x98/0x99)
   - REPORT_SEMAPHORE_EXECUTE (0x5A) with its payload and address in 0x56-0x59 */
static void amp_ib(GpuChannel* ch){ u64 a=((u64)R3(0x5F2)<<32)|R3(0x5F3), n=((u64)(R3(0x8E)&0xFF)<<32)|R3(0x8F), e=a+(n? n-1 : 0); R3(0x5F4)=(u32)(e>>32); R3(0x5F5)=(u32)e; }
static void amp_vs(GpuChannel* ch, u32 j){ u64 a=((u64)R3(0x701+4*j)<<32)|R3(0x702+4*j), n=((u64)(R3(0x180+2*j)&0xFF)<<32)|R3(0x181+2*j), e=a+(n? n-1 : 0);
  R3(0x7C0+2*j)=(u32)(e>>32); R3(0x7C1+2*j)=(u32)e; }
static void ampere_method(GpuChannel* ch, u32 m, u32 v){
  if(m>=0x800 && m<0x860 && ((m&15)==5 || (m&15)==6)){ u32 s=(m-0x800)>>4; u64 a=((u64)(R3(0x805+16*s)&0xFF)<<32)|R3(0x806+16*s);
    if(a>>32!=R3(0x582)){ static int said; if(said<4 && R3(0x801+16*s)){ said++; LOG("gpu: Ampere shader stage %u at %llx (other stages share %x)\n",s,(unsigned long long)a,R3(0x582)); } }
    R3(0x582)=(u32)(a>>32); R3(0x583)=0; R3(0x801+16*s)=(u32)a; return; }
  if(m==0x8E || m==0x8F || m==0x5F2 || m==0x5F3){ amp_ib(ch); return; }
  if(m>=0x180 && m<0x1C0){ amp_vs(ch,(m-0x180)>>1); return; }
  if(m>=0x701 && m<0x780 && ((m-0x701)&3)<2){ amp_vs(ch,(m-0x701)>>2); return; }
  switch(m){
    case 0x9B: case 0x9D: { int idx= m==0x9B; u32 ca=R3(0x98), inst= (ca>>9&1)? R3(0x99) : 1; if(!inst) inst=1;   /* DRAW_{INDEX_BUFFER,VERTEX_ARRAY}_BEGIN_END */
      if(idx) amp_ib(ch);
      { static int said; if(said<sw2_cap(4)){ said++; LOG("gpu: Ampere %s draw, topology %u, first %u, count %u, instances %u\n",idx? "indexed" : "array",ca&15,R3(m-1),v,inst); } }
      r3d_draw(ch,idx,ca&15,R3(m-1),v,idx? R3(0x50D) : 0,R3(0x50E),inst); return; }
    case 0x5A: { u64 a=((u64)(R3(0x59)&0xFF)<<32)|R3(0x58); u32 op=v&3, sz=(v>>13)&3, rep=(v>>7)&31;   /* REPORT_SEMAPHORE_EXECUTE */
      { extern void semdbg(const char*,u32,u64,u64,u32); semdbg("3d-5a",ch->id,a,((u64)R3(0x57)<<32)|R3(0x56),v); }
      { static int nl; if(nl<sw2_cap(2) && getenv("SWR_CB3_LOG")){ nl++; LOG("gpu: ch%u 3D report semaphore %llx exec %08x payload %08x %08x\n",ch->id,(unsigned long long)a,v,R3(0x56),R3(0x57)); } }
      if(op==0 || op==2){ u64 val= rep? 0 : (((u64)R3(0x57)<<32)|R3(0x56));
        if(sz==1) gmem_w32(a,(u32)val); else if(sz==2) gmem_w64(a,val); else { u64 q[2]={val,gpu_ticks()}; gmem_write(a,q,16); } }
      return; }
  }
}


void maxwell3d_method(GpuChannel* ch, u32 m, u32 v, int last){
  if(m>=0xE00){ /* macro call */
    if(!ch->mme_active){ ch->mme_active=1; ch->mme_index=(m-0xE00)/2; ch->mme_nparams=0; }
    if(ch->mme_nparams<0x10000) ch->mme_params[ch->mme_nparams++]=v;
    if(last){ ch->mme_active=0; mme_run(ch,ch->mme_index,ch->mme_params,ch->mme_nparams); ch->mme_nparams=0; }
    return;
  }
  m&=0xFFF;
  if(g_pd.on){ /* keep accumulating only for BEGIN(next instance), END, ranges, or writes that don't change a reg */
    int keep;
    switch(m){
      case 0x586: keep=(v>>26)&1; break;
      case 0x585: case 0x35D: case 0x35E: case 0x5F7: case 0x5F8: keep=1; break;
      case 0x45: case 0x46: case 0x47: case 0x48: case 0x6C: case 0x6D: case 0xB2: case 0x674: case 0x6C3:
      case 0x904: case 0x90C: case 0x914: case 0x91C: case 0x924: keep=0; break;
      default: keep= (m<0x8E4 || m>0x8F3) && R3(m)==v; break; }
    if(!keep || g_pd.ch!=ch) r3d_pending_flush(); }
  R3(m)=v;
  if(ch->ampere) ampere_method(ch,m,v);
  switch(m){
    case 0x45: ch->mme_upload_ptr= ch->ampere? v*3 : v; break;   /* TU104: the pointer (and the start addresses) count 96 bit instructions */
    case 0x46: ch->mme_code[ch->mme_upload_ptr++%0x3000]=v; break;
    case 0x47: ch->mme_start_ptr=v; break;
    case 0x48: ch->mme_start[ch->mme_start_ptr++&0xFF]=v; break;
    case 0x6C: i2m_launch(ch,ch->r3d,0x60); break;
    case 0x6D: i2m_data(ch,v); break;
    case 0xB2: if(v&(1u<<20)){ extern void gpu_gl_stream_incr(u32); gpu_gl_stream_incr(v&0xFFFF); } break;              /* SyncInfo */
    case 0x674: r3d_clear(ch,v); break;                                    /* ClearBuffers */
    case 0x6C3: query_get(ch,ch->r3d,0x6C0,v); break;
    case 0x585: { /* END: flush */
      u32 topo=R3(0x586)&0xFFFF; int idx= R3(0x5F8)!=0;
      u32 first= idx? R3(0x5F7) : R3(0x35D), count= idx? R3(0x5F8) : R3(0x35E), bv= idx? R3(0x50D) : 0, bi=R3(0x50E);
      if(g_pd.on && g_pd.ch==ch && ((R3(0x586)>>26)&1) && g_pd.indexed==idx && g_pd.topo==topo && g_pd.first==first && g_pd.count==count && g_pd.bv==bv && g_pd.bi==bi){
        g_pd.inst=g_instance_count+1; g_draws_merged++; }   /* one more instance */
      else { r3d_pending_flush(); g_pd.on=1; g_pd.ch=ch; g_pd.indexed=idx; g_pd.topo=topo; g_pd.first=first; g_pd.count=count; g_pd.bv=bv; g_pd.bi=bi; g_pd.inst=g_instance_count+1; }
      /* hw eats the counts on each draw, if we don't the next non indexed draw picks up old indices */
      R3(0x5F8)=0; R3(0x35E)=0;
      break; }
    case 0x586: /* BEGIN, bit 26 = next instance */
      if(v&(1u<<26)) g_instance_count++; else if(!(v&(1u<<27))) g_instance_count=0; break;
    case 0x8E4: case 0x8E5: case 0x8E6: case 0x8E7: case 0x8E8: case 0x8E9: case 0x8EA: case 0x8EB:
    case 0x8EC: case 0x8ED: case 0x8EE: case 0x8EF: case 0x8F0: case 0x8F1: case 0x8F2: case 0x8F3: r3d_const_upload(ch,v); break;
    case 0x904: case 0x90C: case 0x914: case 0x91C: case 0x924: r3d_cb_bind(ch,(int)((m-0x904)/8),v); break;
    case 0x6F8: case 0x6F9: case 0x6FA: case 0x6FB: case 0x6FC: case 0x6FD: case 0x6FE: case 0x6FF: /* draw inline */ break;
    default: break;
  }
  (void)g_draw_inline_count;
}

/* ============================================================ compute + I2M */
volatile u64 g_compute_launches;
/* Ampere QMD (V03_00, NVIDIA clc7c0qmd.h): the two release semaphores the dispatch writes when it's done. NVN's fences
   (nvnSyncWait, the queue's own waits) come from these, so they're written even while the dispatch itself is skipped */
/* NVN's fences are a ring of QMDs: the first carries everything (DEPENDENCE_COUNTER, a dependent QMD, release 0 to the
   queue's fence word), the next ones are left blank except for their new payload and run after a DECREMENT_DEPENDENCE.
   A blank QMD (major version 0) that runs takes the release words of the channel's last full one */
static int getenv_cached_nocs3(void){ static int v=-1; if(v<0) v=getenv("SWR_NO_AMPERE_COMPUTE")!=NULL; return v; }   /* SWR_NO_AMPERE_COMPUTE=1: skip them again */
static void qmd3_releases(GpuChannel* ch, const u32* q, u32 action){
  u32 qq[64]; memcpy(qq,q,256);
  /* the template is the fence ring's full QMD (it has a DEPENDENCE_COUNTER): work QMDs with their own releases replaced
     it and the blank fences then released the work counter instead of the queue's fence word */
  if(((q[18]>>4)&0xF)==3 && ((q[25]>>23)&1) && ((q[4]>>14)&0xFFFF)){ memcpy(ch->qmd_rel_tmpl,q+24,32); ch->qmd_has_tmpl=1; }
  else if(((q[18]>>4)&0xF)==0 && ch->qmd_has_tmpl){ if(action!=7) return;
    u32 p0=q[26], p1=q[27]; memcpy(qq+24,ch->qmd_rel_tmpl,32); qq[26]=p0; qq[27]=p1; q=qq; }
  if(((q[4]>>14)&0xFFFF) && action!=7) return;   /* DEPENDENCE_COUNTER: it waits for its decrements */
  /* releases 0 and 1 (words 24, 28), and 2 at word 52 (NVN patches it for syncs between queues) */
  for(int r=0;r<3;r++){ const u32* w=q+(r==2? 52 : 24+4*r); u32 f=w[1]; if(!((f>>23)&1)) continue;   /* RELEASEn_ENABLE */
    u64 a=((u64)(f&0xFF)<<32)|w[0]; u32 ssz=(f>>30)&3; u64 pay=((u64)w[3]<<32)|w[2];
    { static int nl; if(nl<sw2_cap(8)){ nl++; LOG("gpu: ch%u compute release %d at %llx payload %llx (%s%s)\n",ch->id,r,(unsigned long long)a,(unsigned long long)pay,(f>>26)&1? "reduction " : "",ssz==1? "one word" : ssz==2? "two words" : "four words"); } }
    { extern void semdbg(const char*,u32,u64,u64,u32); semdbg("qmd-rel",ch->id,a,pay,f|(r<<8)); }
    if(!a) continue;
    /* the QMD numbers its reductions ADD MIN MAX INC DEC AND OR XOR, gpu_sem_reduce takes the GPFIFO ones: with ADD read
       as MIN, NVN2's +1 sync releases left 0 and the waiting queue gave up after 2 s each time */
    if((f>>26)&1){ extern void gpu_sem_reduce(u64 addr, u32 redop, u32 payload, int is_signed); static const u8 qmap[8]={5,0,1,6,7,3,4,2}; gpu_sem_reduce(a,qmap[(f>>20)&7],(u32)pay,(f>>24)&1); }
    /* bit 19: a release a 3D macro also advances, it only ever moves forward (Eden's notes, title 1) */
    else if((f>>19)&1 && ssz==1){ u32 old=gmem_r32(a); if((i32)((u32)pay-old)>0) gmem_w32(a,(u32)pay); }
    else if(ssz==1) gmem_w32(a,(u32)pay);
    else if(ssz==2 || ((f>>29)&1)) gmem_w64(a, (f>>29)&1? pay : (u32)pay);
    else { u64 qq[2]={(f>>29)&1? pay : (u32)pay,gpu_ticks()}; gmem_write(a,qq,16); } } }

/* QMD dependencies (clc7c0qmd.h): a QMD with a DEPENDENCE_COUNTER runs once that many decrements reached it, from
   PCAS2 action 7 or from another QMD that finished with DEPENDENT_QMD0 = DECREMENT_DEPENDENCE (word 16 bits 1-3,
   pointer in word 15). NVN2's queue fences are QMDs like that (counter 2): the work QMD before them decrements them
   and so does the submit. Only action 7 used to count, and a fence whose decrements all came from other QMDs never
   released: MKW's compute queue waited 1 s each time and the race ran at a frame every 30 s.
   A blank QMD (major version 0) of NVN's fence ring counts as waiting for one decrement */
typedef struct { u64 qmd; int sched; int decs; int incs; int pcas; } QmdDep;
static QmdDep g_qdep[256]; static u32 g_qdep_next;
static QmdDep* qdep_find(u64 a, int make){
  for(int i=0;i<256;i++) if(g_qdep[i].qmd==a) return &g_qdep[i];
  if(!make) return NULL;
  QmdDep* d=NULL; for(int i=0;i<256 && !d;i++){ QmdDep* e=&g_qdep[(g_qdep_next+i)&255]; if(!e->qmd) d=e; }
  if(!d) d=&g_qdep[g_qdep_next&255]; g_qdep_next++; d->qmd=a; d->sched=0; d->decs=0; d->incs=0; d->pcas=0; return d; }
/* ready: every decrement it counts on arrived, or the queue's own DECREMENT_DEPENDENCE came after all the work linked
   with INCREMENT_DEPENDENCE finished. The second covers the ring's full fence, whose other decrement comes from the
   last ring entry: blank entries don't carry that pointer. Everything here runs in submission order, so an earlier
   fence of the queue has already run by then */
static int qdep_ready(const QmdDep* d, int need){ return d->decs>=need+d->incs || (d->pcas && d->decs-d->pcas>=d->incs); }
/* a blank QMD (major version 0) of NVN's fence ring behaves like the ring's full one: same counter, and it hands off
   to the entry the same distance away (the next one) */
static int qmd_dep_count(GpuChannel* ch, const u32* q){
  if(((q[18]>>4)&0xF)==0) return ch->qt_addr? (int)ch->qt_need : 1;
  return (int)((q[4]>>14)&0xFFFF); }
static void qmd_note_tmpl(GpuChannel* ch, u64 qa, const u32* q){
  if(((q[18]>>4)&0xF)==3 && ((q[4]>>14)&0xFFFF) && (q[16]&1) && q[15]){ ch->qt_addr=qa; ch->qt_need=(q[4]>>14)&0xFFFF; ch->qt_w15=q[15]; ch->qt_w16=q[16]; } }
static void qmd_schedule(GpuChannel* ch, u64 qa, int depth);
/* PCAS2 action 12 (INCREMENT_DEPENDENCE): one more decrement to wait for. NVN2 adds one to its queue fence for every
   work QMD it links to it */
static int qdep_log(void);
static void qmd_increment(GpuChannel* ch, u64 qa){ if(qa){ QmdDep* d=qdep_find(qa,1); d->incs++; if(qdep_log()) LOG("qmd ch%u increment %llx -> %d\n",ch->id,(unsigned long long)qa,d->incs); } }
static void qmd_decrement(GpuChannel* ch, u64 qa, int depth);
static void qmd_exec(GpuChannel* ch, u64 qmd, int depth){
  g_compute_launches++; extern int g_gl; extern void gpu_gl_note_compute(void); if(g_gl) gpu_gl_note_compute();
  u32 q[64]; gmem_read(qmd,q,256);
  { int m=0; (void)m;
      if(ch->ampere){ /* Switch 2: the first Ampere QMDs (clc7c0qmd.h) are saved for debugging */
        static int n; if(n<sw2_cap(8)){ n++; plat_mkdir("captures"); char fp[96]; snprintf(fp,sizeof fp,"captures/ampere_qmd_%02d.bin",n); FILE* f=fopen(fp,"wb"); if(f){ fwrite(q,1,256,f); fclose(f); }
          LOG("gpu: Ampere compute dispatch (QMD at %llx saved in %s)\n",(unsigned long long)qmd,fp); }
        /* QMD 3.0 (clc7c0qmd.h) -> the 2.x words gpu_gl_dispatch reads: program address in 48/49 (absolute), grid z in 14,
           constant buffers at 32+2i/33+2i with the size in 16 byte units at bit 19 */
        if(g_gl && ((q[18]>>4)&0xF)==3 && !getenv_cached_nocs3()){ u32 q2[64]; memcpy(q2,q,256);
          q2[8]=q[48]; q2[13]=(q[13]&0xFFFF)|((q[14]&0xFFFF)<<16);
          for(int i=0;i<8;i++){ u32 hi=q[33+2*i], sz=((hi>>19)&0x1FFF)<<4; q2[29+2*i]=q[32+2*i]; q2[30+2*i]=(hi&0xFF)|(sz<<15); }
          extern void gpu_gl_dispatch(GpuChannel*, const u32*, u64); gpu_gl_dispatch(ch,q2,(u64)(q[49]&0x1FFFF)<<32); }
        qmd3_releases(ch,q,7); }
  }
  /* the QMD this one hands off to */
  u32 w15=q[15], w16=q[16]; u64 dq=(u64)w15<<8;
  if(((q[18]>>4)&0xF)==0 && ch->qt_addr){ w16=ch->qt_w16; dq=qmd+(((u64)ch->qt_w15<<8)-ch->qt_addr); w15=1; }
  if(depth<16 && (w16&1) && w15){ u32 act=(w16>>1)&7;
    if(act==4) qmd_decrement(ch,dq,depth+1); else if(act==1 || act==3) qmd_schedule(ch,dq,depth+1); } }
static int qdep_log(void){ static int v=-1; if(v<0) v=getenv("SWR_QMD_LOG")!=NULL; return v; }
static void qmd_schedule(GpuChannel* ch, u64 qa, int depth){
  if(!qa) return; u32 q[64]; gmem_read(qa,q,256); qmd_note_tmpl(ch,qa,q); int need=qmd_dep_count(ch,q);
  QmdDep* d=qdep_find(qa,0);
  /* counts left from the ring's previous lap (a late DECREMENT_DEPENDENCE) would fire this one early and lose this
     lap's: a schedule starts it over */
  if(d && !d->sched){ d->decs=d->incs=d->pcas=0; }
  int have= d? d->decs : 0; int rdy= d? qdep_ready(d,need) : need<=0; if(d) need+=d->incs;
  if(qdep_log()) LOG("qmd ch%u schedule %llx major %u need %d have %d dep0 %x->%llx (depth %d) rel0 %x %x %x rel2 %x %x %x\n",ch->id,(unsigned long long)qa,(q[18]>>4)&15,need,have,q[16]&15,(unsigned long long)q[15]<<8,depth,q[24],q[25],q[26],q[52],q[53],q[54]);
  if(rdy || have>=need){ if(d) d->qmd=0; qmd_exec(ch,qa,depth); return; }
  d=qdep_find(qa,1); d->sched=1; }
static void qmd_decrement(GpuChannel* ch, u64 qa, int depth){
  if(!qa) return; QmdDep* d=qdep_find(qa,1); d->decs++; if(!depth) d->pcas++;
  { static int nl; if(nl<sw2_cap(6) && !d->sched){ nl++; LOG("gpu: ch%u QMD %llx decremented before it was scheduled (%d)\n",ch->id,(unsigned long long)qa,d->decs); } }
  if(qdep_log()) LOG("qmd ch%u decrement %llx -> %d (scheduled %d, depth %d)\n",ch->id,(unsigned long long)qa,d->decs,d->sched,depth);
  if(!d->sched) return;
  u32 q[64]; gmem_read(qa,q,256); if(!qdep_ready(d,qmd_dep_count(ch,q))) return;
  d->qmd=0; qmd_exec(ch,qa,depth); }

void compute_method(GpuChannel* ch, u32 m, u32 v, int last){
  if(m>=0xE00 && ch->ampere){ /* a compute macro call (CALL_MME_MACRO 0x3800 + 8j) */
    if(!ch->cmme_active){ ch->cmme_active=1; ch->cmme_index=(m-0xE00)/2; ch->cmme_nparams=0; }
    if(ch->cmme_nparams<0x1000) ch->cmme_params[ch->cmme_nparams++]=v;
    if(last){ ch->cmme_active=0; { static int said; if(said<sw2_cap(4)){ said++; LOG("gpu: ch%u compute macro %u (%u parameters)\n",ch->id,ch->cmme_index,ch->cmme_nparams); } }
      mme_tu104(ch,ch->mme_start[ch->cmme_index&0xFF],ch->cmme_params,ch->cmme_nparams,1); ch->cmme_nparams=0; }
    return; }
  m&=0xFFF; ch->rcomp[m]=v;
  switch(m){
    case 0x107: /* Ampere LOAD_INLINE_QMD_DATA(63): the QMD written word by word (0xC8..0x107) goes to the address in
                   SET_INLINE_QMD_ADDRESS (0xC6/0xC7) and is scheduled unless it waits for a dependency (word 4 bit 12),
                   0xA7 bit 24 forces it. NVN2's queue-to-queue syncs come this way: MKW's compute queue
                   waited on a release (word 24) that never happened */
      if(ch->ampere){ u64 qa=(((u64)ch->rcomp[0xC6]<<32)|ch->rcomp[0xC7])<<8; if(qa) gmem_write(qa,&ch->rcomp[0xC8],256);
        { static int said; if(said<sw2_cap(4)){ said++; LOG("gpu: ch%u inline QMD at %llx (release %08x %08x payload %x)\n",ch->id,(unsigned long long)qa,ch->rcomp[0xC8+24],ch->rcomp[0xC8+25],ch->rcomp[0xC8+26]); } }
        if(qa && (((ch->rcomp[0xA7]>>24)&1) || !((ch->rcomp[0xCC]>>12)&1))){ u32 prev=ch->rcomp[0xAD]; ch->rcomp[0xAD]=(u32)(qa>>8); compute_method(ch,0xB0,3,last); ch->rcomp[0xAD]=prev; ch->rcomp[0xB0]=3; } }
      break;
    case 0x45: if(ch->ampere) ch->mme_upload_ptr=v*3; break;   /* macro uploads through the compute class (TU104 programs) */
    case 0x46: if(ch->ampere) ch->mme_code[ch->mme_upload_ptr++%0x3000]=v; break;
    case 0x47: if(ch->ampere) ch->mme_start_ptr=v; break;
    case 0x48: if(ch->ampere) ch->mme_start[ch->mme_start_ptr++&0xFF]=v; break;
    case 0x6C: i2m_launch(ch,ch->rcomp,0x60); break;
    case 0x6D: i2m_data(ch,v); break;
    case 0xB0: /* SEND_SIGNALING_PCAS2_B (Ampere): actions 2, 3, 9, 10 schedule the QMD; 7 (decrement dependence) lets a
                  QMD that waited on another run, NVN's fence QMD goes this way */
      if(ch->ampere && (v&15)==12){ qmd_increment(ch,(u64)ch->rcomp[0xAD]<<8); break; }
      if(!ch->ampere || !((v&15)==2 || (v&15)==3 || (v&15)==7 || (v&15)==9 || (v&15)==10)) break;
      /* fallthrough */
    case 0xAF: /* launch; the first few programs get dumped for debugging */
      if(m==0xAF && ch->ampere && !(v&2)) break;   /* Ampere SEND_SIGNALING_PCAS_B: bit 1 schedules */
      if(ch->ampere){ u64 qa=(u64)ch->rcomp[0xAD]<<8; if(m==0xB0 && (v&15)==7) qmd_decrement(ch,qa,0); else qmd_schedule(ch,qa,0); break; }
      { g_compute_launches++; extern int g_gl; extern void gpu_gl_note_compute(void); if(g_gl) gpu_gl_note_compute();
      u64 qmd=(u64)ch->rcomp[0xAD]<<8; u64 code=((u64)ch->rcomp[0x582]<<32)|ch->rcomp[0x583];
      u32 q[64]; gmem_read(qmd,q,256);
      if(g_gl){ extern void gpu_gl_dispatch(GpuChannel*, const u32*, u64); gpu_gl_dispatch(ch,q,code); }
      u64 prog=code+q[8]; static u64 seen[32]; static int ns; int known=0; for(int i=0;i<ns;i++) if(seen[i]==prog) known=1;
      extern int g_opt_diag; if(!known && ns<32 && g_opt_diag){ seen[ns++]=prog; extern int plat_mkdir(const char*); plat_mkdir("captures"); plat_mkdir("captures/compute"); char p[160];
        snprintf(p,sizeof p,"captures/compute/c%02d_qmd.bin",ns); FILE* f=fopen(p,"wb"); if(f){ fwrite(q,1,256,f); fclose(f); }
        u8* b=malloc(0x10000); gmem_read(prog,b,0x10000); snprintf(p,sizeof p,"captures/compute/c%02d_cs.bin",ns); f=fopen(p,"wb"); if(f){ fwrite(b,1,0x10000,f); fclose(f); } free(b);
        for(int i=0;i<8;i++) if((q[20]>>i)&1){ u64 a=((u64)(q[30+2*i]&0xFF)<<32)|q[29+2*i]; u32 sz=(q[30+2*i]>>15)&0x1FFFF; if(!sz||sz>0x10000) sz=0x10000;
          u8* cb=malloc(sz); gmem_read(a,cb,sz); snprintf(p,sizeof p,"captures/compute/c%02d_cb%d.bin",ns,i); f=fopen(p,"wb"); if(f){ fwrite(cb,1,sz,f); fclose(f); } free(cb); }
        LOG("compute %d: grid %ux%ux%u, block %ux%ux%u, shared %u bytes, cbufs %02x (saved in captures/compute)\n",ns,q[12]&0x7FFFFFFF,q[13]&0xFFFF,q[13]>>16,q[18]>>16,q[19]&0xFFFF,q[19]>>16,q[17]&0x3FFFF,q[20]&0xFF); } } break;
    case 0x6C3: query_get(ch,ch->rcomp,0x6C0,v); break;   /* release the semaphore, the game may be waiting */
  }
}

void kepler_i2m_method(GpuChannel* ch, u32 m, u32 v){
  m&=0x7F; ch->ri2m[m]=v;
  if(m==0x6C) i2m_launch(ch,ch->ri2m,0x60);
  else if(m==0x6D) i2m_data(ch,v);
}

/* ============================================================ Fermi 2D (blits) */
typedef struct { u32 format, linear, bh, depth, layer, pitch, width, height; u64 addr; } Surf2D;

static void surf2d(u32* r, int b, Surf2D* s){ s->format=r[b]; s->linear=r[b+1]; s->bh=(r[b+2]>>4)&15; s->depth=r[b+3]; s->layer=r[b+4]; s->pitch=r[b+5]; s->width=r[b+6]; s->height=r[b+7]; s->addr=((u64)r[b+8]<<32)|r[b+9]; }

u32 fmt_bpp_2d(u32 f){
  switch(f){ case 0xE6: case 0xE9: case 0xD5: case 0xCF: case 0xD8: case 0xD9: case 0xDF: case 0xC2: case 0xCA: case 0xDE: case 0xC8: case 0xE0: case 0xE2: return 4;
    case 0xF3: case 0xE8: case 0xF8: case 0xEA: case 0xEB: case 0xEC: case 0xFE: case 0xFF: return 1;
    case 0xC0: case 0xC1: case 0xC3: case 0xC4: case 0xC5: case 0xC6: case 0xC7: return 16;
    case 0xCB: case 0xCC: case 0xCE: return 8;
    default: return 4; } }

extern int r3d_blit(Surf2D* src, Surf2D* dst, int dx, int dy, int dw, int dh, double sx, double sy, double sdx, double sdy);

void fermi2d_method(GpuChannel* ch, u32 m, u32 v){
  m&=0x3FF; ch->r2d[m]=v;
  if(m==0x237){ /* src_y0 write kicks the blit */
    u32* r=ch->r2d; Surf2D dst, src; surf2d(r,0x80,&dst); surf2d(r,0x8C,&src);
    i32 dx0=(i32)r[0x22C], dy0=(i32)r[0x22D], dw=(i32)r[0x22E], dh=(i32)r[0x22F];
    /* 32.32 fixed point, frac reg comes before int reg (DU_DX_FRAC, DU_DX_INT...) */
    i64 dudx=(i64)(((u64)r[0x231]<<32)|r[0x230]); i64 dvdy=(i64)(((u64)r[0x233]<<32)|r[0x232]);
    i64 sx0=(i64)(((u64)r[0x235]<<32)|r[0x234]); i64 sy0=(i64)(((u64)r[0x237]<<32)|r[0x236]);
    double sx=(double)sx0/4294967296.0, sy=(double)sy0/4294967296.0, sdx=(double)dudx/4294967296.0, sdy=(double)dvdy/4294967296.0;
    /* CENTER origin (sample_mode bit0 == 0): coords address texel centers */
    if(!(r[0x223]&1)){ sx-=sdx*0.5; sy-=sdy*0.5; if(sx<0) sx=0; if(sy<0) sy=0; }
    if(r3d_blit(&src,&dst,dx0,dy0,dw,dh,sx,sy,sdx,sdy)) return;
    extern int g_gl; extern void gpu_gl_rt_flush(u64); extern void gpu_gl_rt_reload(u64);
    if(g_gl){ gpu_gl_rt_flush(src.addr); gpu_gl_rt_flush(dst.addr); }
    u32 bpp=fmt_bpp_2d(dst.format), sbpp=fmt_bpp_2d(src.format); if(bpp!=sbpp) bpp=sbpp;
    for(i32 y=0;y<dh;y++){ u32 syy=(u32)(sy+sdy*y);
      for(i32 x=0;x<dw;x++){ u32 sxx=(u32)(sx+sdx*x); u8 px[16];
        u64 so= src.linear? (u64)syy*src.pitch+sxx*bpp : bl_offset(sxx*bpp,syy,0,src.width*bpp,src.height,src.bh,0);
        u64 dof= dst.linear? (u64)(dy0+y)*dst.pitch+(dx0+x)*bpp : bl_offset((dx0+x)*bpp,dy0+y,0,dst.width*bpp,dst.height,dst.bh,0);
        gmem_read(src.addr+so,px,bpp); gmem_write(dst.addr+dof,px,bpp); } }
    if(g_gl) gpu_gl_rt_reload(dst.addr);
  }
}

/* ============================================================ DMA */
volatile u64 g_dma_copies;

/* host pointer to a whole bl surface (slice dz, block depth bd) if it's contiguous, so DMA can memcpy it instead
   of translating + tracking every 16 bytes */
static u8* bl_span_ptr(u64 base, u32 wbytes, u32 h, u32 bh, u32 z, u32 bd, u64* span){
  static int off=-1; if(off<0) off=getenv("SWR_DMA_SLOW")!=NULL; if(off) return NULL;
  u64 gh=8ull<<bh, gd=1ull<<bd; u64 sz=(((u64)wbytes+63)&~63ull)*((h+gh-1)/gh*gh)*((z/gd+1)*gd);
  if(!sz || sz>(256ull<<20)) return NULL;
  u8* p=gmem_ptr(base,sz); if(p) *span=sz; return p; }

static void bl_span_done(u64 base, u64 span){
  extern void (*g_ww_touch)(u64); extern u8* g_mem;
  if(g_ww_touch){ u8* p=gmem_ptr(base,span); if(p) for(u64 o=0;o<span;o+=0x10000) g_ww_touch((u64)(p-g_mem)+o); }
  gpu_mem_written(base,span); }

void dma_method(GpuChannel* ch, u32 m, u32 v){
  m&=0x1FF; ch->rdma[m]=v;
  { static int nl; if(nl<40 && getenv("SWR_CB3_LOG")){ nl++; LOG("dma method ch%u m %x v %x\n",ch->id,m,v); } }
  if(m!=0xC0) return;             /* launch_dma */
  u32* r=ch->rdma;
  u64 src=((u64)r[0x100]<<32)|r[0x101], dst=((u64)r[0x102]<<32)|r[0x103];
  u32 pitch_in=r[0x104], pitch_out=r[0x105], line_len=r[0x106], lines=r[0x107];
  int src_linear=(v>>7)&1, dst_linear=(v>>8)&1, multi=(v>>9)&1, remap=(v>>10)&1;
  if(!multi) lines=1;
  u32 bpp=1;
  if(remap){ u32 rc=r[0x1C2]; u32 comp=((rc>>16)&3)+1, csz=((rc>>20)&3)+1; bpp=comp*csz; line_len*=bpp; }
  /* bl params: dst 0x1C3.., src 0x1CA.. */
  u32 dbh=(r[0x1C3]>>4)&15, dw=r[0x1C4], dh=r[0x1C5], dd=r[0x1C6]; (void)dd;
  u32 sbh=(r[0x1CA]>>4)&15, sw=r[0x1CB], sh=r[0x1CC];
  u32 sx=r[0x1CF]&0xFFFF, sy=r[0x1CF]>>16; u32 dx=r[0x1C8]&0xFFFF, dy=r[0x1C8]>>16;
  /* array layer = whole 2D bl surface stride */
  /* 3D (depth>1): layer reg is the z slice inside the volume, block depth at 11:8 */
  u32 dz=0, sz_=0, dbd=0, sbd=0;
  if(!dst_linear && r[0x1C6]>1){ dz=r[0x1C7]; dbd=(r[0x1C3]>>8)&15; }
  if(!src_linear && r[0x1CD]>1){ sz_=r[0x1CE]; sbd=(r[0x1CA]>>8)&15; }
  { u32 dl= dz? 0 : r[0x1C7], sl= sz_? 0 : r[0x1CE];
    if(r[0x1C6]>1) dl=0; if(r[0x1CD]>1) sl=0;
    if(!dst_linear && dl){ u32 wbb=dw*(remap?bpp:1); u64 gh=8u<<dbh; u64 lsz=(((u64)wbb+63)&~63ull)*((dh+gh-1)/gh*gh); dst+=lsz*dl; }
    if(!src_linear && sl){ u32 wbb=sw*(remap?bpp:1); u64 gh=8u<<sbh; u64 lsz=(((u64)wbb+63)&~63ull)*((sh+gh-1)/gh*gh); src+=lsz*sl; } }
  { static u64 ndma; ndma++; extern volatile u64 g_dma_copies; g_dma_copies=ndma; }
  { static int said; if(said<sw2_cap(6)){ said++; LOG("dma: %s->%s src=%llx dst=%llx pitch %u/%u len=%u lines=%u multi=%d remap=%d(%x) dst[bh=%u w=%u h=%u d=%u l=%u o=%x] src[bh=%u w=%u h=%u o=%x] launch=%x\n",
      src_linear?"lin":"bl",dst_linear?"lin":"bl",(unsigned long long)src,(unsigned long long)dst,pitch_in,pitch_out,line_len,lines,multi,remap,r[0x1C2],dbh,dw,dh,r[0x1C6],r[0x1C7],r[0x1C8],sbh,sw,sh,r[0x1CF],v); } }
  extern int g_gl; extern void gpu_gl_rt_flush(u64); extern void gpu_gl_rt_reload(u64); extern void gpu_gl_trace(const char*);
  if(g_gl){ char tb[256]; snprintf(tb,sizeof tb,"DMA %s %llx -> %s %llx len=%u lines=%u remap=%d\n",src_linear?"lin":"bl",(unsigned long long)src,dst_linear?"lin":"bl",(unsigned long long)dst,line_len,lines,remap); gpu_gl_trace(tb);
    gpu_gl_rt_flush(src); gpu_gl_rt_flush(dst); }
  u8* dpp=NULL; u8* spp=NULL; u64 dspan=0, sspan=0; (void)spp; (void)sspan;
  if(remap && (r[0x1C2]&7)==4 && dst_linear){
    /* DST_X = CONST_A: a buffer fill with the constant in 0x1C0, the source isn't read */
    u32 cv=r[0x1C0]; u8* f=malloc((size_t)line_len+4); for(u32 i=0;i<line_len;i+=4) memcpy(f+i,&cv,4);
    for(u32 y=0;y<lines;y++) gmem_write(dst+(u64)y*pitch_out,f,line_len);
    free(f);
  } else if(src_linear && dst_linear){
    if(!multi){ u8* tmp=malloc(line_len); gmem_read(src,tmp,line_len); gmem_write(dst,tmp,line_len); free(tmp); }
    else { u8* tmp=malloc(line_len); for(u32 y=0;y<lines;y++){ gmem_read(src+(u64)y*pitch_in,tmp,line_len); gmem_write(dst+(u64)y*pitch_out,tmp,line_len); } free(tmp); }
  } else if(src_linear && !dst_linear && (dpp=bl_span_ptr(dst,dw*(remap?bpp:1),dh,dbh,dz,dbd,&dspan))){
    /* contiguous dst: copy directly, one write record at the end */
    u32 wb=dw*(remap?bpp:1); u8* buf=malloc(line_len);
    for(u32 y=0;y<lines;y++){ gmem_read(src+(u64)y*pitch_in,buf,line_len);
      for(u32 x=0;x<line_len;){ u32 xb=dx*(remap?bpp:1)+x, n=16-(xb&15); if(n>line_len-x) n=line_len-x;
        u64 o=bl_offset(xb,dy+y,dz,wb,dh,dbh,dbd); if(o+n<=dspan) memcpy(dpp+o,buf+x,n); else gmem_write(dst+o,buf+x,n); x+=n; } }
    free(buf); bl_span_done(dst,dspan);
  } else if(src_linear && !dst_linear){
    u32 wb=dw*(remap?bpp:1); u8* buf=malloc(line_len);
    for(u32 y=0;y<lines;y++){ gmem_read(src+(u64)y*pitch_in,buf,line_len);
      /* bl keeps 16 byte groups contiguous so copy per group and not per byte (per byte was ~11% in profiles) */
      for(u32 x=0;x<line_len;){ u32 xb=dx*(remap?bpp:1)+x, n=16-(xb&15); if(n>line_len-x) n=line_len-x;
        u64 o=bl_offset(xb,dy+y,dz,wb,dh,dbh,dbd); gmem_write(dst+o,buf+x,n); x+=n; } }
    free(buf);
  } else if(!src_linear && dst_linear){
    u32 wb=sw*(remap?bpp:1); u8* buf=malloc(line_len);
    for(u32 y=0;y<lines;y++){ for(u32 x=0;x<line_len;){ u32 xb=sx*(remap?bpp:1)+x, n=16-(xb&15); if(n>line_len-x) n=line_len-x;
        u64 o=bl_offset(xb,sy+y,sz_,wb,sh,sbh,sbd); gmem_read(src+o,buf+x,n); x+=n; }
      gmem_write(dst+(u64)y*pitch_out,buf,line_len); }
    free(buf);
  } else if(!src_linear && !dst_linear && (dpp=bl_span_ptr(dst,dw*(remap?bpp:1),dh,dbh,dz,dbd,&dspan)) && (spp=bl_span_ptr(src,sw*(remap?bpp:1),sh,sbh,sz_,sbd,&sspan))){
    u32 swb=sw*(remap?bpp:1), dwb=dw*(remap?bpp:1);
    for(u32 y=0;y<lines;y++) for(u32 x=0;x<line_len;){ u32 xs=sx*(remap?bpp:1)+x, xd=dx*(remap?bpp:1)+x, n=16-(xs&15), m=16-(xd&15); if(m<n) n=m; if(n>line_len-x) n=line_len-x;
      u64 so=bl_offset(xs,sy+y,sz_,swb,sh,sbh,sbd), dof=bl_offset(xd,dy+y,dz,dwb,dh,dbh,dbd);
      if(so+n<=sspan && dof+n<=dspan) memmove(dpp+dof,spp+so,n); else { u8 b[16]; gmem_read(src+so,b,n); gmem_write(dst+dof,b,n); } x+=n; }
    bl_span_done(dst,dspan);
  } else {
    u32 swb=sw*(remap?bpp:1), dwb=dw*(remap?bpp:1);
    for(u32 y=0;y<lines;y++) for(u32 x=0;x<line_len;){ u32 xs=sx*(remap?bpp:1)+x, xd=dx*(remap?bpp:1)+x, n=16-(xs&15), m=16-(xd&15); if(m<n) n=m; if(n>line_len-x) n=line_len-x;
      u8 b[16]; gmem_read(src+bl_offset(xs,sy+y,sz_,swb,sh,sbh,sbd),b,n); gmem_write(dst+bl_offset(xd,dy+y,dz,dwb,dh,dbh,dbd),b,n); x+=n; }
  }
  if(g_gl) gpu_gl_rt_reload(dst);
  /* completion semaphore */
  u32 sem=(v>>3)&3;
  if(sem){ u64 sa=((u64)r[0x90]<<32)|r[0x91]; { extern void semdbg(const char*,u32,u64,u64,u32); semdbg("dma-rel",ch->id,sa,r[0x92],v); }
    if(sem==1 || sem==2){ if(!gpu_gl_hold_write(sa,&r[0x92],4)) gmem_w32(sa,r[0x92]); }
    if(sem==2){ u64 t=gpu_ticks(); if(!gpu_gl_hold_write(sa+8,&t,8)) gmem_w64(sa+8,t); } }
}

void gpu_flush(GpuChannel* ch){ extern void r3d_flush(GpuChannel* ch); r3d_flush(ch); }
