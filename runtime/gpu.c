/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* GM20B front end: GPFIFO, puller, semaphores/syncpoints, I2M, DMA, 2D and the parts of 3D that write memory
   (clears, reports). the drawing itself is the renderer's job */
#include "rt.h"
/* how long a semaphore acquire waits before going on anyway: 2 s, 200 ms on Switch 2 (some of NVN2's queue syncs
   aren't paired up yet and each wait held the game back), SWR_SEM_TIMEOUT_MS overrides */
static u64 sem_timeout_ns(void){ static u64 v; if(!v){ const char* e=getenv("SWR_SEM_TIMEOUT_MS"); u64 ms= e? strtoull(e,NULL,10) : 0; if(!ms) ms= (g_title_id>>56)==0x04? 200 : 2000; v=ms*1000000ull; } return v; }

#include "gpu.h"
void swr_tl(const char* fmt, ...);

GpuState g_gpu;
static PMutex g_gpulock;

/* ============================================================ GPU memory */
extern volatile int g_cs_any; void cs_mem_need(u64 a, u64 n);

int gmem_read(u64 gva, void* dst, u64 size){
  u8* d=dst; if(g_cs_any) cs_mem_need(gva,size);
  while(size){ u64 cpu=gmmu_translate(gva); u64 n=0x1000-(gva&0xFFF); if(n>size) n=size;
    if(!cpu){ memset(d,0,n); } else memcpy(d,GP(cpu),n);
    d+=n; gva+=n; size-=n; }
  return 0; }

void (*g_ww_touch)(u64 cpu);   /* GPU wrote guest memory, record it for write tracking */
int gmem_write(u64 gva, const void* src, u64 size){
  const u8* s=src; u64 g0=gva, s0=size;
  while(size){ u64 cpu=gmmu_translate(gva); u64 n=0x1000-(gva&0xFFF); if(n>size) n=size;
    if(cpu){ memcpy(GP(cpu),s,n); if(g_ww_touch) g_ww_touch(cpu); } s+=n; gva+=n; size-=n; }
  gpu_mem_written(g0,s0);
  return 0; }

u8* gmem_ptr(u64 gva, u64 size){
  /* host pointer, only if the whole range is contiguous */
  if(g_cs_any) cs_mem_need(gva,size);
  u64 c0=gmmu_translate(gva); if(!c0) return NULL;
  for(u64 o=(0x1000-(gva&0xFFF));o<size;o+=0x1000){ if(gmmu_translate(gva+o)!=c0+o) return NULL; }
  return GP(c0); }

u32 gmem_r32(u64 gva){ u32 v=0; gmem_read(gva,&v,4); return v; }

void gmem_w32(u64 gva, u32 v){ gmem_write(gva,&v,4); }

void gmem_w64(u64 gva, u64 v){ gmem_write(gva,&v,8); }

/* ============================================================ block linear */
u64 bl_offset(u32 x_bytes, u32 y, u32 z, u32 width_bytes, u32 height, u32 bh_log2, u32 bd_log2){
  u32 gob_w=(width_bytes+63)/64; u32 bh=1u<<bh_log2, bd=1u<<bd_log2;
  u32 blocks_h=((height+7)/8+bh-1)/bh;
  u32 gx=x_bytes/64, gy=y/8;
  u32 block_y=gy/bh, gob_in=gy%bh; u32 block_z=z/bd, z_in=z%bd;
  u64 block_size=512ull*bh*bd;
  u64 off=((u64)block_z*blocks_h*gob_w + (u64)block_y*gob_w + gx)*block_size + ((u64)z_in*bh + gob_in)*512;
  u32 xb=x_bytes%64, yb=y%8;
  u32 in= ((xb%64)/32)*256 + (yb/2)*64 + ((xb%32)/16)*32 + (yb%2)*16 + (xb%16);
  off+= gob_native_on()? gob_native(in) : in;
  return off; }
int g_gob_native=-1;

void deswizzle(u8* dst, u32 dst_pitch, const u8* src, u32 w_bytes, u32 h, u32 bh_log2){
  for(u32 y=0;y<h;y++){
    u8* d=dst+(u64)y*dst_pitch;
    for(u32 x=0;x<w_bytes;x+=16){
      u64 o=bl_offset(x,y,0,w_bytes,h,bh_log2,0); u32 n=w_bytes-x<16? w_bytes-x : 16; memcpy(d+x,src+o,n); } } }

void swizzle(u8* dst, const u8* src, u32 src_pitch, u32 w_bytes, u32 h, u32 bh_log2){
  for(u32 y=0;y<h;y++){ const u8* s=src+(u64)y*src_pitch;
    for(u32 x=0;x<w_bytes;x+=16){ u64 o=bl_offset(x,y,0,w_bytes,h,bh_log2,0); u32 n=w_bytes-x<16? w_bytes-x : 16; memcpy(dst+o,s+x,n); } } }

/* ============================================================ puller */
/* Turing/Ampere engine classes (Switch 2 binds these) run on the Maxwell engines: their methods keep the same
   offsets for what the Maxwell engines implement */
static u32 gpu_class_norm(u32 c){
  switch(c){ case 0xC797: case 0xC597: case 0xC397: return 0xB197;
    case 0xC7C0: case 0xC5C0: case 0xC3C0: return 0xB1C0;
    case 0xC7B5: case 0xC5B5: case 0xC3B5: return 0xB0B5; }
  return c; }

/* Switch 2 queues wait on each other with semaphores: NVN2 has the 3D queue release a word that the compute queue
   acquires, and the compute queue release the one 0x40 after it for the 3D queue. Every channel runs on the one GPU
   thread, so a blocking wait there could only end by timing out (MKW lost 200 ms on each, 1 FPS on the
   title map). An acquire that isn't met yet parks its channel instead; the thread goes on with the other channels and
   the parked one resumes once the word is there (gpu_chan_pump). kinds: 1 host semaphore (ops 1 equal, 4 >=, 8 and),
   2 C56F SEM_EXECUTE (0 equal, 2 >=, 3 circular >=, 4 and, 5 nor), 3 syncpoint */
static __thread int t_defer;
extern u32 gpu_gl_held_view(int kind, u64 addr, u32 cur);   /* with the releases held for compute results */
static int wait_met(int kind, u64 addr, u64 want, u32 op, int wide){
  /* 4: commands at addr (want bytes) a compute wrote that haven't come back from the host GPU yet */
  if(kind==4){ extern int gpu_gl_cs_pending(u64,u64); extern void gpu_gl_cs_poll(void); gpu_gl_cs_poll(); return !gpu_gl_cs_pending(addr,want); }
  if(kind==3){ extern u32 syncpt_read(u32); return (i32)(gpu_gl_held_view(3,addr,syncpt_read((u32)addr))-(u32)want)>=0; }
  if(kind==1){ u32 cur=gpu_gl_held_view(1,addr,gmem_r32(addr)); return op==1? cur==(u32)want : op==8? (cur&(u32)want)!=0 : (i32)(cur-(u32)want)>=0; }
  u64 cur= wide? (gmem_r32(addr)|((u64)gmem_r32(addr+4)<<32)) : gpu_gl_held_view(2,addr,gmem_r32(addr)); if(!wide) want=(u32)want;
  return op==0? cur==want : op==2? cur>=want : op==3? (wide? (i64)(cur-want)>=0 : (i32)((u32)cur-(u32)want)>=0) : op==4? (cur&want)!=0 : (~(cur|want))!=0; }
/* 1 = met now (after letting held compute results land), else parks the channel when the caller can resume it later */
static int wait_or_park(GpuChannel* ch, int kind, u64 addr, u64 want, u32 op, int wide){
  if(wait_met(kind,addr,want,op,wide)) return 1;
  if(!t_defer) return 0;
  ch->blk=kind; ch->blk_addr=addr; ch->blk_want=want; ch->blk_op=op; ch->blk_wide=wide; ch->blk_t0=plat_ns(); ch->stop=1; return 1; }

static void puller_method(GpuChannel* ch, u32 m, u32 v){
  u32* r=ch->puller;
  r[m]=v;
  switch(m){
    case 0x00: break;
    case 0x07: { /* SemaphoreOperation */
      u64 addr=((u64)(r[4]&0xFF)<<32)|r[5]; u32 op=v&0x1F;
      { static int nl; if(nl<sw2_cap(24)){ nl++; LOG("gpu: ch%u channel semaphore op %u at %llx payload %x (word %x, now %x)\n",ch->id,op,(unsigned long long)addr,r[6],v,gmem_r32(addr)); } }
      { extern void semdbg(const char*,u32,u64,u64,u32); semdbg(op==0x10? "host-red" : op==2? "host-rel" : "host-acq",ch->id,addr,r[6],v); }
      if(op==0x10){ /* reduction: an atomic op on the word instead of a write (fence counters) */
        extern void gpu_sem_reduce(u64 addr, u32 redop, u32 payload, int is_signed); extern int gpu_gl_hold_reduce(u64,u32,u32,int);
        if(!gpu_gl_hold_reduce(addr,(v>>27)&0xF,r[6],!((v>>31)&1))) gpu_sem_reduce(addr,(v>>27)&0xF,r[6],!((v>>31)&1));
      } else if(op==8){ /* acquire AND: wait until (mem & payload) != 0 */
        if(wait_or_park(ch,1,addr,r[6],8,0)) break;
        u64 end=plat_ns()+sem_timeout_ns();
        u32 bo=0; while(!(gmem_r32(addr)&r[6]) && plat_ns()<end) plat_backoff(&bo);
      } else
      if(op==2){ /* release */
        extern int gpu_gl_hold_write(u64,const void*,u32);   /* after compute results still on their way */
        if(v&(1u<<24)) { if(!gpu_gl_hold_write(addr,&r[6],4)) gmem_w32(addr,r[6]); }                     /* 4 bytes only */
        else { extern u64 gpu_ticks(void); u32 q[4]={r[6],0,0,0}; u64 t=gpu_ticks(); memcpy(&q[2],&t,8); if(!gpu_gl_hold_write(addr,q,16)) gmem_write(addr,q,16); }
      } else if(op==1 || op==4){ /* acquire, with a timeout so a bad value can't hang us */
        if(wait_or_park(ch,1,addr,r[6],op,0)) break;
        u64 end=plat_ns()+sem_timeout_ns();
        u32 bo=0; for(;;){ u32 cur=gmem_r32(addr); if(op==1? cur==r[6] : (i32)(cur-r[6])>=0) break; if(plat_ns()>end){ WARN("gpu: ch%u semaphore %llx waits for %x (has %x), gave up after %llu ms\n",ch->id,(unsigned long long)addr,r[6],cur,(unsigned long long)(sem_timeout_ns()/1000000)); break; } plat_backoff(&bo); }
      }
      break; }
    case 0x1B: { /* SEM_EXECUTE (Volta+ host class C56F, Switch 2): address in 0x17/0x18, payload in 0x19/0x1A (NVIDIA clc56f.h).
                    op 0 acquire equal, 1 release, 2 strict >=, 3 circular >=, 4 and, 5 nor, 6 reduction; bit 24 64-bit payload,
                    bit 25 release with a timestamp */
      u64 addr=((u64)(r[0x18]&0x1FFFFFF)<<32)|(r[0x17]&~3u); u32 op=v&7; int wide=(v>>24)&1; u64 pay=((u64)r[0x1A]<<32)|r[0x19];
      { extern void semdbg(const char*,u32,u64,u64,u32); semdbg(op==6? "c56f-red" : op==1? "c56f-rel" : "c56f-acq",ch->id,addr,pay,v); }
      if(op==1){ extern int gpu_gl_hold_write(u64,const void*,u32);
        if((v>>25)&1){ extern u64 gpu_ticks(void); u64 q[2]={pay,gpu_ticks()}; if(!wide) q[0]=(u32)pay; if(!gpu_gl_hold_write(addr,q,16)) gmem_write(addr,q,16); }
        else if(wide){ if(!gpu_gl_hold_write(addr,&pay,8)) gmem_w64(addr,pay); }
        else { u32 p32=(u32)pay; if(!gpu_gl_hold_write(addr,&p32,4)) gmem_w32(addr,p32); } }
      else if(op==6){ extern void gpu_sem_reduce(u64 addr, u32 redop, u32 payload, int is_signed); extern int gpu_gl_hold_reduce(u64,u32,u32,int);
        if(!gpu_gl_hold_reduce(addr,(v>>27)&0xF,(u32)pay,!((v>>31)&1))) gpu_sem_reduce(addr,(v>>27)&0xF,(u32)pay,!((v>>31)&1)); }
      else if(op<=5 && wait_or_park(ch,2,addr,pay,op,wide)){}
      else if(op<=5){
        u64 end=plat_ns()+sem_timeout_ns(); u32 bo=0;
        for(;;){ u64 cur= wide? (gmem_r32(addr)|((u64)gmem_r32(addr+4)<<32)) : gmem_r32(addr); u64 want= wide? pay : (u32)pay;
          int ok= op==0? cur==want : op==2? cur>=want : op==3? (wide? (i64)(cur-want)>=0 : (i32)((u32)cur-(u32)want)>=0) : op==4? (cur&want)!=0 : (~(cur|want))!=0;
          if(ok) break; if(plat_ns()>end){ WARN("gpu: ch%u semaphore %llx op %u waits for %llx (has %llx), gave up after %llu ms\n",ch->id,(unsigned long long)addr,op,(unsigned long long)want,(unsigned long long)cur,(unsigned long long)(sem_timeout_ns()/1000000)); break; } plat_backoff(&bo); } }
      { static int nl; if(nl<sw2_cap(12)){ nl++; LOG("gpu: ch%u channel semaphore (C56F) op %u at %llx payload %llx\n",ch->id,op,(unsigned long long)addr,(unsigned long long)pay); } }
      break; }
    case 0x1D: { /* Syncpoint */
      u32 id=(v>>8)&0xFFF; { static int nl; if((g_title_id>>56)==0x04 && nl<600){ nl++; LOG("gpu: ch%u syncpoint %u %s %x (now %x)\n",ch->id,id,(v&1)? "incr" : "wait for",r[0x1C],syncpt_read(id)); } }
      if(v&1){ extern void gpu_gl_stream_incr(u32); gpu_gl_stream_incr(id); } else if(!wait_or_park(ch,3,id,r[0x1C],0,0)) syncpt_wait(id,r[0x1C],2000000000); break; }
    default: break;
  }
}

/* ============================================================ inline to memory */
static int i2m_plain(GpuChannel* ch){ return (ch->i2m_linear || (ch->i2m_lines<=1 && !ch->i2m_dw)) && (ch->i2m_lines<=1 || !ch->i2m_pitch || ch->i2m_pitch==ch->i2m_w); }
void i2m_launch(GpuChannel* ch, u32* regs, int base){
  /* a plain upload the game stopped short of its length (Switch 2's compute macros give 0x1000 and send a 256 byte
     QMD): what came is already in memory, tell the caches */
  if(ch->i2m_pos && ch->i2m_pos<ch->i2m_len && i2m_plain(ch)) gpu_mem_written(ch->i2m_addr,ch->i2m_pos);
  u32 line_len=regs[base+0], lines=regs[base+1]; u64 addr=((u64)regs[base+2]<<32)|regs[base+3];
  /* 64-bit: line_len*lines can wrap in 32 bits and leave a tiny buffer for a huge upload */
  u64 len=(u64)line_len*(lines?lines:1); if(len>(256ull<<20)) len=0;
  ch->i2m_len=(u32)len; ch->i2m_pos=0; ch->i2m_addr=addr; ch->i2m_linear=(regs[base+0xC]&1);
  ch->i2m_pitch=regs[base+4]; ch->i2m_w=line_len; ch->i2m_lines=lines;
  { static u64 seen[40]; static int ns; static int on=-1; if(on<0) on=getenv("SWR_CB3_LOG")!=NULL; u64 pg=addr>>12; int k=0; for(;k<ns;k++) if(seen[k]==pg) break; if(on && k==ns && ns<40){ seen[ns++]=pg; LOG("i2m: ch%u base %x to %llx len %llu\n",ch->id,base,(unsigned long long)addr,(unsigned long long)len); } }
  /* block linear dst. 0x65 = block dims (w 3:0, h 7:4, d 11:8), 0x66..0x68 size, 0x69 z, 0x6A x, 0x6B y. yes the
     width field is bits 3:0 and height is 7:4 */
  ch->i2m_bh=(regs[base+5]>>4)&0xF; ch->i2m_bd=(regs[base+5]>>8)&0xF;
  ch->i2m_dw=regs[base+6]; ch->i2m_dh=regs[base+7]; ch->i2m_z=regs[base+9]; ch->i2m_x=regs[base+0xA]; ch->i2m_y=regs[base+0xB];
  if(ch->i2m_buf_cap<ch->i2m_len){ u8* nb=realloc(ch->i2m_buf,(size_t)ch->i2m_len+4); if(!nb){ ch->i2m_len=0; return; } ch->i2m_buf=nb; ch->i2m_buf_cap=ch->i2m_len; }
}

void i2m_data(GpuChannel* ch, u32 v){
  if(ch->i2m_pos>=ch->i2m_len) return;
  u32 n=ch->i2m_len-ch->i2m_pos; if(n>4) n=4; memcpy(ch->i2m_buf+ch->i2m_pos,&v,n);
  int plain=i2m_plain(ch);
  if(plain) gmem_write(ch->i2m_addr+ch->i2m_pos,&v,n);   /* plain uploads land word by word, like the hardware's */
  ch->i2m_pos+=n;
  if(ch->i2m_pos>=ch->i2m_len){
    if(ch->i2m_linear || (ch->i2m_lines<=1 && !ch->i2m_dw)){   /* single line, no surface: plain copy */
      if(plain) {}
      else for(u32 y=0;y<ch->i2m_lines;y++) gmem_write(ch->i2m_addr+(u64)y*ch->i2m_pitch,ch->i2m_buf+(u64)y*ch->i2m_w,ch->i2m_w);
      gpu_mem_written(ch->i2m_addr,ch->i2m_lines>1 && ch->i2m_pitch? (u64)ch->i2m_pitch*ch->i2m_lines : ch->i2m_len); }
    else { /* bl dst */
      u32 w=ch->i2m_w, h=ch->i2m_lines? ch->i2m_lines : 1; u32 wb=ch->i2m_dw? ch->i2m_dw : w, hh=ch->i2m_dh? ch->i2m_dh : h;
      u64 hi=0;
      for(u32 y=0;y<h;y++) for(u32 x=0;x<w;){ u32 xb=ch->i2m_x+x, n=16-(xb&15); if(n>w-x) n=w-x;
        u64 o=bl_offset(xb,ch->i2m_y+y,ch->i2m_z,wb,hh,ch->i2m_bh,ch->i2m_bd); gmem_write(ch->i2m_addr+o,ch->i2m_buf+(u64)y*w+x,n); if(o+n>hi) hi=o+n; x+=n; }
      gpu_mem_written(ch->i2m_addr,hi); }
  }
}

/* ============================================================ dispatch */
extern void r3d_pending_flush(void); extern int r3d_pending_on(void);

static void call_method(GpuChannel* ch, u32 sub, u32 m, u32 v, int last){
  { static int lg=-1, n, cap; if(lg<0){ extern u64 g_title_id; lg= !getenv("SWR_TRACE_OFF") && (getenv("SWR_METH_LOG") || (g_title_id>>56)==0x04); cap= getenv("SWR_METH_LOG")? 40000 : 3000; }   /* Switch 2: the first GPU methods go to the log */
    if(lg && n<cap && m!=0x46){ n++; LOG("meth ch%u sub%u cls %x m %x v %x\n",ch->id,sub,ch->sub_class[sub],m,v); } }
  { static int on=-1, n2; static u64 t0, after; if(on<0){ const char* e=getenv("SWR_METH_AFTER"); on= e!=NULL; after= e? strtoull(e,NULL,10)*1000000000ull : 0; t0=plat_ns(); }   /* debug: methods of one channel after N seconds */
    if(on && n2<8000 && plat_ns()-t0>after && sub==0 && m!=0x46 && m!=0x6D){ n2++; LOG("M ch%u %x %x\n",ch->id,m,v); } }
  if(m<0x40){ if(r3d_pending_on()) r3d_pending_flush(); puller_method(ch,m,v); if(m==0){ u32 cl=v&0xFFFF; if(cl==0xC797 || cl==0xC597 || cl==0xC7C0 || cl==0xC5C0){ if(!ch->ampere) LOG("gpu: channel %u binds Ampere %s (class %x)\n",ch->id,(cl&0xFF)==0xC0? "compute" : "3D",cl); ch->ampere=1; } ch->sub_class[sub]=gpu_class_norm(cl); } return; }
  u32 cls=ch->sub_class[sub];
  if(r3d_pending_on() && !(cls==0xB197 || (sub==0 && cls!=0xB1C0 && cls!=0xA140 && cls!=0x902D && cls!=0xB0B5))) r3d_pending_flush();   /* switching engines: flush pending draws */
  switch(cls){
    case 0xB197: maxwell3d_method(ch,m,v,last); break;
    case 0xB1C0: compute_method(ch,m,v,last); break;
    case 0xA140: kepler_i2m_method(ch,m,v); break;
    case 0x902D: fermi2d_method(ch,m,v); break;
    case 0xB0B5: dma_method(ch,m,v); break;
    default:
      if(sub==0) maxwell3d_method(ch,m,v,last);
      else if(sub==1) compute_method(ch,m,v,last);
      else if(sub==2) kepler_i2m_method(ch,m,v);
      else if(sub==3) fermi2d_method(ch,m,v);
      else if(sub==4) dma_method(ch,m,v);
      break;
  }
}

/* returns the words used: all of them, or fewer when a wait parked the channel (ch->stop) */
static u32 run_pushbuffer(GpuChannel* ch, const u32* w, u32 n){
  /* method batches can spill into the next GPFIFO entry (hw does this, yuzu's DmaPusher too). keep the batch state
     across entries or the leftover data gets parsed as headers */
  u32 i=0;
  for(;;){
    if(ch->pb_k<ch->pb_cnt){
      u32 mode=ch->pb_mode, sub=ch->pb_sub, m=ch->pb_m;
      while(ch->pb_k<ch->pb_cnt && i<n){ u32 k=ch->pb_k++; u32 mm= mode==1? m+k : mode==5? (k==0? m : m+1) : m;
        call_method(ch,sub,mm,w[i++],ch->pb_k==ch->pb_cnt); if(ch->stop) return i; }
      if(ch->pb_k<ch->pb_cnt){ static int said; if(said<sw2_cap(6)){ said++; LOG("gpu: method batch continues in the next GPFIFO entry (sub %u method %x mode %u, %u of %u words done)\n",ch->pb_sub,ch->pb_m,ch->pb_mode,ch->pb_k,ch->pb_cnt); } return i; }    /* rest comes in the next entry */
    }
    if(i>=n) return i;
    u32 h=w[i++];
    u32 mode=h>>29, sub=(h>>13)&7, m=h&0x1FFF, cnt=(h>>16)&0x1FFF;
    if(h && (mode==0 || mode==2 || mode>=6)){ /* shouldn't happen in a sane stream, log it */
      static int said; if(said<sw2_cap(6)){ said++; LOG("gpu: odd command header %08x (mode %u) at word %u of %u in channel %u; words around:",h,mode,i-1,n,ch->id);
        for(u32 j= i>5? i-5 : 0; j<i+6 && j<n; j++) LOG(" %s%08x",j==i-1? "*" : "",w[j]); LOG("\n"); } }
    switch(mode){
      case 1: case 3: case 5: ch->pb_mode=mode; ch->pb_sub=sub; ch->pb_m=m; ch->pb_cnt=cnt; ch->pb_k=0; break;
      /* modes 0/2 (fermi), 6 and 7 carry no data words (same as DmaPusher). don't treat them as batches, a stray
         one would eat the methods after it, syncpoint incrs included */
      case 4: call_method(ch,sub,m,cnt,1); if(ch->stop) return i; break;
      default: break;
    }
  }
}

static GpuChannel* chan_get(u32 id){
  for(int i=0;i<g_gpu.nchan;i++) if(g_gpu.chan[i]->id==id) return g_gpu.chan[i];
  if(g_gpu.nchan>=64){ static int said; if(!said){ said=1; WARN("gpu: more than 64 channels, sharing the last one\n"); } return g_gpu.chan[63]; }
  GpuChannel* c=calloc(1,sizeof *c); c->id=id; c->sub_class[0]=0xB197; c->sub_class[1]=0xB1C0; c->sub_class[2]=0xA140; c->sub_class[3]=0x902D; c->sub_class[4]=0xB0B5;
  /* hw reset values that NVN/deko3d rely on and never write. sRGB writes are on by default, if not sRGB targets
     get stored linear and look too dark */
  c->r3d[0x4B8]=1;
  /* gba classics leaves the color masks at reset. zero masks discarded every sprite, including the loading UI.
     initialize once; explicit masks from the game still replace these values */
  for(int i=0;i<8;i++) c->r3d[0x680+i]=0x1111;
  c->r3d[0x4E]=1; c->r3d[0x556]=1; /* render-enable reset: true */
  g_gpu.chan[g_gpu.nchan++]=c; return c; }

void gpu_submit_now(u64* e, u32 n, u32 channel){
  pm_lock(&g_gpulock);
  GpuChannel* ch=chan_get(channel);
  for(u32 k=0;k<n;k++){
    u64 ent=e[k]; u64 va=ent&0xFFFFFFFFFCull; u32 words=(u32)((ent>>42)&0x1FFFFF);
    if(!words) continue;
    u32* buf=(u32*)gmem_ptr(va,(u64)words*4);
    if(buf) run_pushbuffer(ch,buf,words);
    else { u32* tmp=malloc((u64)words*4); gmem_read(va,tmp,(u64)words*4); run_pushbuffer(ch,tmp,words); free(tmp); }
  }
  r3d_pending_flush();
  gpu_flush(ch);
  pm_unlock(&g_gpulock);
}

void gpu_init(void){ pm_init(&g_gpulock); gpu3d_init(); }

/* ---- per channel submit queues on the GPU thread (see wait_or_park) */
typedef struct PSub { u64* ent; u32 n, k, i, syncpt, incs, wait_id, wait_val; int started; u64 wt0; struct PSub* next; } PSub;
void (*g_sub_done)(u32 syncpt, u32 incs);
static GpuChannel* g_parked[64]; static int g_nparked;
static u64 park_timeout_ns(void){ static u64 v; if(!v){ const char* e=getenv("SWR_PARK_MS"); v=(e? strtoull(e,NULL,10) : 1000)*1000000ull; if(!v) v=1; } return v; }

/* runs what it can of the channel's queue. 1 = it's still waiting on something */
static int chan_pump(GpuChannel* ch){
  while(ch->psq){ PSub* s=ch->psq;
    if(ch->blk){
      if(!wait_met(ch->blk,ch->blk_addr,ch->blk_want,ch->blk_op,ch->blk_wide)){
        if(plat_ns()-ch->blk_t0<park_timeout_ns()) return 1;
        WARN("gpu: ch%u %s %llx waits for %llx (has %x), gave up after %llu ms\n",ch->id,ch->blk==3? "syncpoint" : "semaphore",(unsigned long long)ch->blk_addr,(unsigned long long)ch->blk_want,
          ch->blk==3? 0u : gmem_r32(ch->blk_addr),(unsigned long long)(park_timeout_ns()/1000000)); }
      ch->blk=0; }
    if(!s->started){
      if(s->wait_id!=~0u){ extern u32 syncpt_read(u32);
        if((i32)(gpu_gl_held_view(3,s->wait_id,syncpt_read(s->wait_id))-s->wait_val)<0){
          { u64 now=plat_ns(); if(!s->wt0) s->wt0=now;
            if(now-s->wt0<2000000000ull) return 1;
            { static int nl; if(nl<sw2_cap(6)){ nl++; LOG("gpu: submit fence %u >= %u never came (2 s), running it anyway\n",s->wait_id,s->wait_val); } } } } }
      s->started=1; }
    for(; s->k<s->n; s->k++, s->i=0){
      u64 ent=s->ent[s->k]; u64 va=ent&0xFFFFFFFFFCull; u32 words=(u32)((ent>>42)&0x1FFFFF);
      if(!words || s->i>=words) continue;
      /* commands a compute wrote (GPU-made command buffers, MKW's culling): park the channel until the
         readback lands instead of blocking the GPU thread on the host GPU; other channels go on meanwhile. blk_cs:
         this segment was parked once already (landed or timed out), run it */
      if(!ch->blk_cs){ extern int gpu_gl_cs_pending(u64,u64); u64 ca=va+(u64)s->i*4, cn=(u64)(words-s->i)*4;
        if(gpu_gl_cs_pending(ca,cn) && !wait_met(4,ca,cn,0,0)){ swr_tl("park ch%u on compute output %llx",ch->id,(unsigned long long)ca); ch->blk=4; ch->blk_addr=ca; ch->blk_want=cn; ch->blk_op=0; ch->blk_wide=0; ch->blk_t0=plat_ns(); ch->blk_cs=1; return 1; } }
      ch->blk_cs=0;
      u32* buf=(u32*)gmem_ptr(va,(u64)words*4); u32 used;
      t_defer=1;
      if(buf) used=run_pushbuffer(ch,buf+s->i,words-s->i);
      else { u32* tmp=malloc((u64)(words-s->i)*4); gmem_read(va+(u64)s->i*4,tmp,(u64)(words-s->i)*4); used=run_pushbuffer(ch,tmp,words-s->i); free(tmp); }
      t_defer=0;
      s->i+=used;
      if(ch->stop){ ch->stop=0; r3d_pending_flush(); gpu_flush(ch); return 1; } }
    r3d_pending_flush(); gpu_flush(ch);
    ch->psq=s->next; if(!ch->psq) ch->psq_tail=NULL;
    swr_tl("ran submit sp%u",s->syncpt); free(s->ent); if(g_sub_done) g_sub_done(s->syncpt,s->incs); free(s); }
  return 0; }

/* GPU thread: queue a submit on its channel (takes e) and run what can run. 1 = some channel is parked */
int gpu_chan_submit(u64* e, u32 n, u32 channel, u32 incs, u32 wait_id, u32 wait_val){
  pm_lock(&g_gpulock);
  GpuChannel* ch=chan_get(channel);
  PSub* s=calloc(1,sizeof *s); s->ent=e; s->n=n; s->syncpt=channel; s->incs=incs; s->wait_id=wait_id; s->wait_val=wait_val;
  if(ch->psq_tail) ((PSub*)ch->psq_tail)->next=s; else ch->psq=s; ch->psq_tail=s;
  int busy=chan_pump(ch);
  if(busy){ int k=0; for(;k<g_nparked;k++) if(g_parked[k]==ch) break; if(k==g_nparked && g_nparked<64) g_parked[g_nparked++]=ch; }
  pm_unlock(&g_gpulock);
  extern int gpu_chan_pump_all(void); return gpu_chan_pump_all(); }

/* GPU thread: resume parked channels whose wait is met. 1 = some are still parked */
int gpu_chan_pump_all(void){
  if(!g_nparked) return 0;
  pm_lock(&g_gpulock);
  for(int again=1; again; ){ again=0;
    for(int k=0;k<g_nparked;k++){ GpuChannel* ch=g_parked[k]; PSub* before=ch->psq; u32 bk=before? before->k : 0, bi=before? before->i : 0;
      if(!chan_pump(ch)){ g_parked[k]=g_parked[--g_nparked]; k--; again=1; }
      else if(ch->psq!=before || (before && (before->k!=bk || before->i!=bi))) again=1; } }   /* progress can free another one */
  int n=g_nparked; pm_unlock(&g_gpulock); return n>0; }

/* SWR_TIMELINE=<s>: a 3 s timeline of one stretch of frames from second <s> on (submits, GPU-thread runs, fence
   releases, syncpoint waits), each line stamped in ms with 0.01 resolution */
void swr_tl(const char* fmt, ...){
  static double from=-1; if(from<0){ const char* e=getenv("SWR_TIMELINE"); from= e? atof(e) : 1e30; }
  static u64 t0; if(!t0) t0=plat_ns(); double s=(plat_ns()-t0)/1e9; if(s<from || s>from+3) return;
  char b[256]; va_list ap; va_start(ap,fmt); vsnprintf(b,sizeof b,fmt,ap); va_end(ap); LOG("tl %10.2f %s\n",(plat_ns()-t0)/1e6,b); }

/* with GL, submissions go to the GPU thread */
volatile u64 g_gpu_submits;

void gpu_submit(u64* e, u32 n, u32 channel, u32 incs, u32 wait_id, u32 wait_val){ g_gpu_submits++; swr_tl("submit sp%u n=%u incs=%u",channel,n,incs);
  { extern int g_gpu_eden; if(g_gpu_eden){ extern void gpu_eden_submit(u64*,u32,u32,u32,u32,u32); gpu_eden_submit(e,n,channel,incs,wait_id,wait_val); return; } }   /* SWR_GPU=eden: Eden runs it and raises the syncpoint */
  extern int gpu_async_submit(u64* e, u32 n, u32 syncpt, u32 incs, u32 wait_id, u32 wait_val);
  if(gpu_async_submit(e,n,channel,incs,wait_id,wait_val)) return;
  if(wait_id!=~0u) syncpt_wait(wait_id,wait_val,2000000000);
  gpu_submit_now(e,n,channel);
  for(u32 i=0;i<incs;i++) syncpt_incr(channel);
}

/* semaphore reductions, the GPFIFO encoding: MIN MAX XOR AND OR ADD INC DEC (INC wraps to 0 past the payload,
   DEC wraps to the payload at 0, like the hw) */
void semdbg(const char* tag, u32 ch, u64 addr, u64 val, u32 word){ static int on=-1, n; if(on<0) on=getenv("SWR_SEMDBG")!=NULL; if(!on || n>6000) return; n++; LOG("semdbg %s ch%u %llx val %llx word %x (now %x)\n",tag,ch,(unsigned long long)addr,(unsigned long long)val,word,gmem_r32(addr)); }
void gpu_sem_reduce(u64 addr, u32 redop, u32 payload, int is_signed){
  u32 cur=gmem_r32(addr), nv=cur;
  switch(redop){
    case 0: nv= is_signed? (u32)((i32)cur<(i32)payload? (i32)cur : (i32)payload) : (cur<payload? cur : payload); break;
    case 1: nv= is_signed? (u32)((i32)cur>(i32)payload? (i32)cur : (i32)payload) : (cur>payload? cur : payload); break;
    case 2: nv=cur^payload; break;
    case 3: nv=cur&payload; break;
    case 4: nv=cur|payload; break;
    case 5: nv=cur+payload; break;
    case 6: nv= cur>=payload? 0 : cur+1; break;
    case 7: nv= (cur==0 || cur>payload)? payload : cur-1; break;
    default: break; }
  gmem_w32(addr,nv); }
