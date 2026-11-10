from pathlib import Path
import re
s=Path('work/v190/SwitchRecomp/runtime/gpu_gl.c').read_text()
a=s[s.index('static void rec_capture('):s.index('\n#else\nstatic void rec_capture(')]
b=s[s.index('static void rec_drain('):s.index('static void rec_stop(')]
w=s[s.index('typedef struct { int kind;'):s.index('#define QN')]
c=s[s.index('void gpu_gl_present_cpu('):s.index('u64 gpu_gl_screenshot(')]
const=sorted(set(re.findall(r'\bGL_[A-Z0-9_]+',a+b)))
funcs=sorted(set(re.findall(r'\b(gl[A-Z][A-Za-z0-9_]*)\(',a+b))-{'glBindBuffer'})
pre=r'''
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
'''
pre+='\n'.join('#define '+n+' '+str(i+1) for i,n in enumerate(const))+'\n'
pre+='\n'.join('#define '+n+'(...) ((void)0)' for n in funcs)+'\n'
pre+='static GLenum mock_wait(GLsync s,unsigned f,u64 ns){if(remaining[s-1]){remaining[s-1]--;waits++;return GL_TIMEOUT_EXPIRED;}return GL_ALREADY_SIGNALED;}\n'
post=r'''
static int g_gl=1;enum {WK_PRESENT_CPU=1,WK_PRESENT_ADDR=2};
'''+w+r'''
static Work pending;
static void q_push(Work*w){pending=*w;}
'''+c+r'''
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
'''
Path('work/rec16/test_video.c').write_text(pre+b+a+post)
