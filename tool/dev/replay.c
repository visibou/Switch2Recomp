/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* offline replay of one captured fullscreen draw on Mesa (surfaceless EGL)
   usage: replay <fs.glsl> <vs.glsl> <cbdata.bin> <config> <out.ppm> */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#define GL_GLEXT_PROTOTYPES
#include <GL/glcorearb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static char* slurp(const char* p, long* n){ FILE* f=fopen(p,"rb"); if(!f){ perror(p); exit(1); } fseek(f,0,2); long s=ftell(f); fseek(f,0,0);
  char* b=malloc(s+1); fread(b,1,s,f); b[s]=0; fclose(f); if(n) *n=s; return b; }

static GLuint sh(GLenum t, const char* src){ GLuint s=glCreateShader(t); glShaderSource(s,1,&src,0); glCompileShader(s);
  GLint ok; glGetShaderiv(s,GL_COMPILE_STATUS,&ok); if(!ok){ char log[8192]; glGetShaderInfoLog(s,sizeof log,0,log); fprintf(stderr,"compile: %s\n",log); exit(1); } return s; }

/* thumbnail ppm: color | alpha side by side */
static float* load_thumb(const char* p, int* w, int* h){
  long n; unsigned char* b=(unsigned char*)slurp(p,&n); int W,H,mx,off=0; char* s=(char*)b;
  if(sscanf(s,"P6 %d %d %d%n",&W,&H,&mx,&off)<3){ fprintf(stderr,"bad ppm %s\n",p); exit(1); } off++;
  *w=W/2; *h=H; float* o=malloc(sizeof(float)*4*(*w)*H);
  for(int y=0;y<H;y++) for(int x=0;x<*w;x++){ unsigned char* c=b+off+(y*W+x)*3; unsigned char* a=b+off+(y*W+x+*w)*3;
    int yy=H-1-y; float* d=o+4*(yy*(*w)+x); d[0]=c[0]/255.f; d[1]=c[1]/255.f; d[2]=c[2]/255.f; d[3]=a[0]/255.f; }
  return o; }

int main(int argc, char** argv){
  if(argc<6){ fprintf(stderr,"usage\n"); return 1; }
  EGLDisplay dpy=eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,EGL_DEFAULT_DISPLAY,0);
  if(!eglInitialize(dpy,0,0)){ fprintf(stderr,"egl init\n"); return 1; }
  eglBindAPI(EGL_OPENGL_API);
  EGLint ca[]={EGL_CONTEXT_MAJOR_VERSION,4,EGL_CONTEXT_MINOR_VERSION,5,EGL_CONTEXT_OPENGL_PROFILE_MASK,EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,EGL_NONE};
  EGLContext ctx=eglCreateContext(dpy,EGL_NO_CONFIG_KHR,EGL_NO_CONTEXT,ca); if(!ctx){ fprintf(stderr,"ctx %x\n",eglGetError()); return 1; }
  eglMakeCurrent(dpy,EGL_NO_SURFACE,EGL_NO_SURFACE,ctx);
  fprintf(stderr,"GL %s / %s\n",glGetString(GL_VERSION),glGetString(GL_RENDERER));

  char* fs=slurp(argv[1],0); char* vs=slurp(argv[2],0); long cbn; char* cb=slurp(argv[3],&cbn);
  GLuint prog=glCreateProgram(); glAttachShader(prog,sh(GL_VERTEX_SHADER,vs)); glAttachShader(prog,sh(GL_FRAGMENT_SHADER,fs)); glLinkProgram(prog);
  GLint ok; glGetProgramiv(prog,GL_LINK_STATUS,&ok); if(!ok){ char log[8192]; glGetProgramInfoLog(prog,sizeof log,0,log); fprintf(stderr,"link: %s\n",log); return 1; }

  int W=320,H=180; FILE* cf=fopen(argv[4],"r"); char line[1024];
  static char zero[65536];
  while(fgets(line,sizeof line,cf)){
    char k[32]; int bind; if(line[0]=='#'||sscanf(line,"%31s",k)<1) continue;
    if(!strcmp(k,"size")){ sscanf(line,"%*s %d %d",&W,&H); }
    else if(!strcmp(k,"ubo")){ unsigned off,sz; sscanf(line,"%*s %d %x %x",&bind,&off,&sz);
      char* buf=calloc(1,65536); if(off!=0xffffffffu) memcpy(buf,cb+off,sz); GLuint b; glGenBuffers(1,&b); glBindBuffer(GL_UNIFORM_BUFFER,b);
      glBufferData(GL_UNIFORM_BUFFER,65536,buf,GL_STATIC_DRAW); glBindBufferBase(GL_UNIFORM_BUFFER,bind,b); }
    else if(!strcmp(k,"ubof")){ float v[4]={0}; sscanf(line,"%*s %d %f %f %f %f",&bind,v,v+1,v+2,v+3);
      GLuint b; glGenBuffers(1,&b); glBindBuffer(GL_UNIFORM_BUFFER,b); char* buf=calloc(1,65536); memcpy(buf,v,16);
      glBufferData(GL_UNIFORM_BUFFER,65536,buf,GL_STATIC_DRAW); glBindBufferBase(GL_UNIFORM_BUFFER,bind,b); }
    else if(!strcmp(k,"ssbo")){ sscanf(line,"%*s %d",&bind); GLuint b; glGenBuffers(1,&b); glBindBuffer(GL_SHADER_STORAGE_BUFFER,b);
      glBufferData(GL_SHADER_STORAGE_BUFFER,65536,zero,GL_STATIC_DRAW); glBindBufferBase(GL_SHADER_STORAGE_BUFFER,bind,b); }
    else { /* textures: <kind> <unit> <file|c> [r g b a] */
      char src[512]; float v[4]={0,0,0,0}; sscanf(line,"%*s %d %511s %f %f %f %f",&bind,src,v,v+1,v+2,v+3);
      GLuint t; glGenTextures(1,&t); glActiveTexture(GL_TEXTURE0+bind);
      GLenum tg = !strcmp(k,"tex2d")? GL_TEXTURE_2D : !strcmp(k,"texcube")? GL_TEXTURE_CUBE_MAP : !strcmp(k,"texcubearr")? GL_TEXTURE_CUBE_MAP_ARRAY : !strcmp(k,"tex2darr")? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_3D;
      glBindTexture(tg,t);
      int w=4,h=4; float* px; if(tg==GL_TEXTURE_CUBE_MAP){ w=64; h=64; }
      if(strcmp(src,"c")){ px=load_thumb(src,&w,&h); }
      else { px=malloc(sizeof(float)*4*w*h*6); for(int i=0;i<w*h*6;i++) memcpy(px+4*i,v,16); }
      if(tg==GL_TEXTURE_2D) glTexImage2D(tg,0,GL_RGBA32F,w,h,0,GL_RGBA,GL_FLOAT,px);
      else if(tg==GL_TEXTURE_CUBE_MAP){ int mips=getenv("CUBE_MIP0ONLY")? 7:1; float* zz=calloc(4*w*h*64,4);
        for(int l=0;l<mips;l++) for(int f=0;f<6;f++) glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X+f,l,GL_RGBA32F,w>>l?w>>l:1,h>>l?h>>l:1,0,GL_RGBA,GL_FLOAT,l?zz:px);
        glTexParameteri(tg,GL_TEXTURE_MAX_LEVEL,mips-1); }
      else if(tg==GL_TEXTURE_CUBE_MAP_ARRAY) glTexImage3D(tg,0,GL_RGBA32F,w,h,6,0,GL_RGBA,GL_FLOAT,px);
      else glTexImage3D(tg,0,GL_RGBA32F,w,h,1,0,GL_RGBA,GL_FLOAT,px);
      glTexParameteri(tg,GL_TEXTURE_MIN_FILTER,tg==GL_TEXTURE_CUBE_MAP? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR); glTexParameteri(tg,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
      glTexParameteri(tg,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE); glTexParameteri(tg,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
    }
  }
  GLuint fb,c0,c1; glGenFramebuffers(1,&fb); glBindFramebuffer(GL_FRAMEBUFFER,fb);
  glGenTextures(1,&c0); glBindTexture(GL_TEXTURE_2D,c0); glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,W,H,0,GL_RGBA,GL_FLOAT,0);
  glGenTextures(1,&c1); glBindTexture(GL_TEXTURE_2D,c1); glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA32F,W,H,0,GL_RGBA,GL_FLOAT,0);
  glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,c0,0);
  glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT1,GL_TEXTURE_2D,c1,0);
  GLenum dbs[2]={GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1}; glDrawBuffers(2,dbs);
  GLuint vao; glGenVertexArrays(1,&vao); glBindVertexArray(vao);
  glViewport(0,0,W,H); glUseProgram(prog); glDrawArrays(GL_TRIANGLES,0,3); glFinish();
  float* out=malloc(sizeof(float)*4*W*H); glReadBuffer(GL_COLOR_ATTACHMENT0); glReadPixels(0,0,W,H,GL_RGBA,GL_FLOAT,out);
  double s[4]={0}; float mx[4]={0}; int nz=0, nan=0;
  for(int i=0;i<W*H;i++){ for(int c=0;c<4;c++){ float v=out[4*i+c]; if(isnan(v)){ nan++; continue; } s[c]+=v; if(v>mx[c]) mx[c]=v; } if(out[4*i]>0||out[4*i+1]>0||out[4*i+2]>0) nz++; }
  printf("mean %.4f %.4f %.4f %.4f  max %.3f %.3f %.3f %.3f  nonzero %d/%d nan %d\n",s[0]/(W*H),s[1]/(W*H),s[2]/(W*H),s[3]/(W*H),mx[0],mx[1],mx[2],mx[3],nz,W*H,nan);
  printf("center px: %g %g %g %g\n",out[4*(H/2*W+W/2)],out[4*(H/2*W+W/2)+1],out[4*(H/2*W+W/2)+2],out[4*(H/2*W+W/2)+3]);
  FILE* o=fopen(argv[5],"wb"); fprintf(o,"P6\n%d %d\n255\n",W,H);
  for(int y=H-1;y>=0;y--) for(int x=0;x<W;x++) for(int c=0;c<3;c++){ float v=out[4*(y*W+x)+c]; v=v/(1+v); v=powf(v<0?0:v,1/2.2f); fputc((int)(v*255),o); }
  fclose(o); return 0; }
