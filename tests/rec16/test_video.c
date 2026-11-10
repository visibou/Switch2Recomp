
#include "rt.h"
#include <assert.h>
#include <stdio.h>
typedef unsigned GLuint;typedef int GLint;typedef unsigned GLenum;typedef unsigned GLbitfield;typedef intptr_t GLsizeiptr;typedef uintptr_t GLsync;
#define REC_N 3
static int g_rec_toggle,g_rec_on=1,g_vt_on=1,g_opt_rec_fps=60;
static int g_rec_w=640,g_rec_h=480,g_rec_have;
static u64 g_rec_vt0=1000000000,g_rec_lastts,g_rec_n,g_rec_t0,g_rec_last,g_rec_out,g_rec_base,g_present_guest_ns;
static u32 g_rec_k;static GLuint g_rec_fbo[2],g_rec_prog[2],g_rec_tex[2],g_rec_pbo[3]={1,2,3},g_scratch_vao,g_blit_prog;
static GLint g_rec_xf[2];static GLsync g_rec_sync[3];static u64 g_rec_ts[3];
static u64 wall,clock_ns;static unsigned current_id,output_frames,bound;static unsigned tags[3],remaining[3];static unsigned waits;
u64 plat_ns(void){return wall;}u64 guest_ns(void){return clock_ns;}
#undef WARN
#define WARN(...) ((void)0)
static int rec_failed(void){return 0;}
static void rec_start(void){}static void rec_stop(void){}
static void glBindBuffer(GLenum t,GLuint id){bound=id;}
static GLsync mock_fence(GLenum c,unsigned f){tags[g_rec_k%3]=current_id;remaining[g_rec_k%3]=3;return g_rec_k%3+1;}
static GLenum mock_wait(GLsync s,unsigned f,u64 ns);
static void mock_delete(GLsync s){}
static const void*mock_map(GLenum t,intptr_t o,GLsizeiptr n,unsigned f){assert(bound>=1&&bound<=3);return &tags[bound-1];}
static int mock_unmap(GLenum t){return 1;}
#define p_rfs mock_fence
#define p_rcws mock_wait
#define p_rds mock_delete
#define p_rmap mock_map
#define p_runmap mock_unmap
void rec_frame(const u8*p,i64 pts,i64 duration){assert(*(const unsigned*)p==output_frames);assert(pts==(i64)((u64)output_frames*10000000/60));assert(duration==(i64)(((u64)output_frames+1)*10000000/60-(u64)output_frames*10000000/60));output_frames++;}
#define GL_ALREADY_SIGNALED 1
#define GL_BLEND 2
#define GL_COLOR_ATTACHMENT0 3
#define GL_COLOR_BUFFER_BIT 4
#define GL_CONDITION_SATISFIED 5
#define GL_FRAMEBUFFER 6
#define GL_FRAMEBUFFER_SRGB 7
#define GL_MAP_READ_BIT 8
#define GL_PACK_ALIGNMENT 9
#define GL_PIXEL_PACK_BUFFER 10
#define GL_READ_FRAMEBUFFER 11
#define GL_RED 12
#define GL_RG 13
#define GL_SCISSOR_TEST 14
#define GL_SYNC_GPU_COMMANDS_COMPLETE 15
#define GL_TEXTURE0 16
#define GL_TEXTURE_2D 17
#define GL_TIMEOUT_EXPIRED 18
#define GL_TRIANGLE_STRIP 19
#define GL_UNSIGNED_BYTE 20
#define glActiveTexture(...) ((void)0)
#define glBindFramebuffer(...) ((void)0)
#define glBindSampler(...) ((void)0)
#define glBindTexture(...) ((void)0)
#define glBindVertexArray(...) ((void)0)
#define glClear(...) ((void)0)
#define glClearColor(...) ((void)0)
#define glDisable(...) ((void)0)
#define glDrawArrays(...) ((void)0)
#define glDrawBuffers(...) ((void)0)
#define glPixelStorei(...) ((void)0)
#define glReadPixels(...) ((void)0)
#define glUniform4fv(...) ((void)0)
#define glUseProgram(...) ((void)0)
#define glViewport(...) ((void)0)
static GLenum mock_wait(GLsync s,unsigned f,u64 ns){if(remaining[s-1]){remaining[s-1]--;waits++;return GL_TIMEOUT_EXPIRED;}return GL_ALREADY_SIGNALED;}
static void rec_drain(int all){
  for(u32 n=0;n<REC_N;n++){ u32 i=(g_rec_k+n)%REC_N; if(!g_rec_sync[i]) continue;
    u64 wait_start=plat_ns();GLenum r;
    do {r=p_rcws(g_rec_sync[i],0,all?100000000ull:0);}
    while(all && r==GL_TIMEOUT_EXPIRED && plat_ns()-wait_start<30000000000ull);
    if(r!=GL_ALREADY_SIGNALED && r!=GL_CONDITION_SATISFIED){
      if(all || r!=GL_TIMEOUT_EXPIRED)WARN("recorder: GPU readback did not complete (%x); frame retained for retry\n",r);
      return;
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER,g_rec_pbo[i]); size_t n12=(size_t)g_rec_w*g_rec_h*3/2;
    const u8* p=(const u8*)p_rmap(GL_PIXEL_PACK_BUFFER,0,(GLsizeiptr)n12,GL_MAP_READ_BIT);
    if(p){ /* constant fps output: output frame k sits at k/fps. every captured frame covers its guest time up to
              the next one, so it gets repeated or dropped. weird clock jumps are clamped */
      u64 per=1000000000ull/(u64)(g_opt_rec_fps>0? g_opt_rec_fps : 60);
      u64 ts=g_rec_ts[i];
      /* Audio and video both start at the recorder guest-clock epoch. */
      u64 rel= ts>g_rec_base? ts-g_rec_base : 0;
      u64 want=rel/per+1;                                   /* output frames that should exist up to this one */
      /* Preserve gaps on the shared audio/video timeline. */   /* > 0.5s jump: emit one frame */
      if(!g_rec_have) want= want>g_rec_out? want : g_rec_out+1;
      u64 fps=(u64)(g_opt_rec_fps>0? g_opt_rec_fps : 60); while(g_rec_out<want){ rec_frame(p,(i64)(g_rec_out*10000000ull/fps),(i64)((g_rec_out+1)*10000000ull/fps-g_rec_out*10000000ull/fps)); g_rec_out++; }
      g_rec_have=1; p_runmap(GL_PIXEL_PACK_BUFFER);p_rds(g_rec_sync[i]);g_rec_sync[i]=0;
    }else {WARN("recorder: readback mapping failed; frame retained for retry\n");glBindBuffer(GL_PIXEL_PACK_BUFFER,0);return;}
    glBindBuffer(GL_PIXEL_PACK_BUFFER,0); } }

static void rec_capture(GLuint tex, const float* xf, int tw, int th){
  if(g_rec_on&&rec_failed()){rec_stop();return;}
  if(g_rec_toggle){ g_rec_toggle=0; if(g_rec_on) rec_stop(); else rec_start(); }
  if(!g_rec_on) return;
  /* A frame queued before F7 was pressed belongs to the previous timeline. */
  if(g_present_guest_ns && g_present_guest_ns<g_rec_vt0)return;
  u64 now=plat_ns(); if(!g_vt_on && g_rec_last && now-g_rec_last<15500000ull) return;   /* max 60/s (smooth mode: every real frame) */
  g_rec_last=now;
  rec_drain(0);
  u32 i=g_rec_k%REC_N; if(g_rec_sync[i]){ if(!g_vt_on) return; rec_drain(1);if(g_rec_sync[i])return; }   /* never overwrite a pending readback */
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
  g_rec_sync[i]=p_rfs(GL_SYNC_GPU_COMMANDS_COMPLETE,0); { u64 source=g_present_guest_ns?g_present_guest_ns:guest_ns();u64 ts=source>g_rec_vt0?source-g_rec_vt0:0; if(g_rec_n && ts<g_rec_lastts) ts=g_rec_lastts; g_rec_lastts=ts; g_rec_ts[i]=ts; } g_rec_n++; g_rec_k++;
  glBindFramebuffer(GL_FRAMEBUFFER,0); glUseProgram(g_blit_prog); }
static int g_gl=1;enum {WK_PRESENT_CPU=1,WK_PRESENT_ADDR=2};
typedef struct { int kind; u64* ent; u32 n, syncpt, incs; u32* px; int w,h; u64 addr; u32 fmt; PCond* done_cv; volatile int* done; char path[256]; u64 result; u64 sub_ns; } Work;


static Work pending;
static void q_push(Work*w){pending=*w;}
void gpu_gl_present_cpu(const u32* px, int w, int h, int stride){
  if(!g_gl) return;
  Work k; memset(&k,0,sizeof k); k.kind=WK_PRESENT_CPU;{extern u64 guest_ns(void);k.sub_ns=guest_ns();} k.px=malloc((size_t)w*h*4); for(int y=0;y<h;y++) memcpy(k.px+(size_t)y*w,px+(size_t)y*stride,(size_t)w*4); k.w=w; k.h=h; q_push(&k); }

void gpu_gl_present_buffer(u64 cpu_addr, u32 w, u32 h, u32 pitch, u32 layout, u32 bhl, u32 fmt){
  Work k; memset(&k,0,sizeof k); k.kind=WK_PRESENT_ADDR;{extern u64 guest_ns(void);k.sub_ns=guest_ns();} k.addr=cpu_addr; k.w=(int)w; k.h=(int)h; k.fmt=fmt|(layout<<8)|(bhl<<12); k.n=pitch; q_push(&k); }


int main(void){float xf[4]={0,0,1,1};
for(unsigned frame=0;frame<600;frame++){
 current_id=frame;u64 source=g_rec_vt0+(u64)frame*1000000000/60;clock_ns=source;
 if(frame&1){u32 px[4]={0};gpu_gl_present_cpu(px,2,2,2);assert(pending.sub_ns==source);free(pending.px);}
 else {gpu_gl_present_buffer(0x1000,640,480,2560,0,0,0);assert(pending.sub_ns==source);}
 /* The GL/GPU queue finishes much later, with 500ms spikes and no stable latency. */
 clock_ns=source+((frame%41)==0?500000000ull:(u64)(frame%7)*9000000);wall+=50000000;
 g_present_guest_ns=pending.sub_ns;rec_capture(1,xf,640,480);
 assert(g_rec_ts[(g_rec_k-1)%3]==source-g_rec_vt0);
}
rec_drain(1);assert(output_frames==600&&g_rec_out==600&&waits==1800);
puts("PASS actual F7 capture/drain and CPU/GPU present producers: 600 distinct frames, exact 60fps timestamps, no duplicates/freezes from 500ms GPU queue jitter");
unsigned before=g_rec_k;g_present_guest_ns=g_rec_vt0-1;rec_capture(1,xf,640,480);assert(g_rec_k==before);
puts("PASS frames queued before F7 starts are excluded from the new audio/video epoch");}
