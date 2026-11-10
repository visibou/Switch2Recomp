/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* X11 + GLX 4.3 backend, the linux version of backend_win32.c (wayland desktops run it through XWayland).
   controllers come from SDL2, loaded at run time like XInput on windows. F1 help F3 screenshot F4 fps limit F5
   trace F8 full res screenshot F11/Alt+Enter fullscreen Esc quit. the Xvfb tests run on this too (screenshots,
   scripted input) */
#include "rt.h"
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/XKBlib.h>
#include <X11/keysym.h>
#include <GL/glx.h>
#include <dlfcn.h>
#include <time.h>

static Display* dpy; static Window win; static Atom wm_delete, a_state, a_fs; static Cursor g_blank;
static volatile int g_ww=1280, g_wh=720; static int g_fullscreen=0, g_want_fs=1;
static int g_seconds=0, g_maxframes=-1; static char g_shot[1024]; static u64 g_start; static u64 g_press=0, g_press_at=0, g_press_len=300;
static char g_script[8192]; static char g_title[256]; static volatile int g_shot_req;
static u64 keys_held;
extern int g_opt_fps;

void backend_ini(const char* k, const char* v){
  if(!strcmp(k,"fullscreen")) g_want_fs=atoi(v);
  else if(!strcmp(k,"seconds")) g_seconds=atoi(v);
  else if(!strcmp(k,"frames")) g_maxframes=atoi(v);
  else if(!strcmp(k,"shot")) strncpy(g_shot,v,1023);
  else if(!strcmp(k,"press")) g_press=strtoull(v,NULL,16);
  else if(!strcmp(k,"press_at")) g_press_at=strtoull(v,NULL,10);
  else if(!strcmp(k,"press_len")) g_press_len=strtoull(v,NULL,10);
  else if(!strcmp(k,"script")) strncpy(g_script,v,8191);
}

int video_has_gl(void){ return 1; }

/* ---------------------------------------------------------------- SDL2 game controllers, loaded dynamically */
typedef void SDLGC;
static struct {
  int (*Init)(u32); int (*SetHint)(const char*, const char*); int (*NumJoysticks)(void); int (*IsGameController)(int);
  SDLGC* (*Open)(int); int (*Attached)(SDLGC*); void (*Close)(SDLGC*); void (*Update)(void);
  u8 (*Button)(SDLGC*, int); i16 (*Axis)(SDLGC*, int); const char* (*Name)(SDLGC*);
} sdl;

static SDLGC* g_pad_gc; static u64 g_pad_scan;

static void sdl_load(void){
  void* h=dlopen("libSDL2-2.0.so.0",RTLD_NOW|RTLD_LOCAL); if(!h) h=dlopen("libSDL2.so",RTLD_NOW|RTLD_LOCAL);
  if(!h){ LOG("controllers: SDL2 not found (install libsdl2), keyboard only\n"); return; }
  void** f[]={(void**)&sdl.Init,(void**)&sdl.SetHint,(void**)&sdl.NumJoysticks,(void**)&sdl.IsGameController,(void**)&sdl.Open,(void**)&sdl.Attached,
              (void**)&sdl.Close,(void**)&sdl.Update,(void**)&sdl.Button,(void**)&sdl.Axis,(void**)&sdl.Name};
  const char* n[]={"SDL_Init","SDL_SetHint","SDL_NumJoysticks","SDL_IsGameController","SDL_GameControllerOpen","SDL_GameControllerGetAttached",
                   "SDL_GameControllerClose","SDL_GameControllerUpdate","SDL_GameControllerGetButton","SDL_GameControllerGetAxis","SDL_GameControllerName"};
  for(int i=0;i<11;i++){ *f[i]=dlsym(h,n[i]); if(!*f[i]){ memset(&sdl,0,sizeof sdl); return; } }
  /* buttons by position like on windows, also for nintendo controllers that SDL would map by label */
  sdl.SetHint("SDL_GAMECONTROLLER_USE_BUTTON_LABELS","0");
  sdl.SetHint("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS","1");
  if(sdl.Init(0x2000 /* SDL_INIT_GAMECONTROLLER */)){ memset(&sdl,0,sizeof sdl); return; }
}

static i32 dz(i32 v){ return (v>-7000 && v<7000)? 0 : v; }

static void pad_poll(u64* b, i32* lx, i32* ly, i32* rx, i32* ry){
  if(!sdl.Init) return;
  sdl.Update();
  if(g_pad_gc && !sdl.Attached(g_pad_gc)){ sdl.Close(g_pad_gc); g_pad_gc=NULL; LOG("controllers: disconnected\n"); }
  if(!g_pad_gc){ u64 now=plat_ns(); if(now-g_pad_scan<1000000000ull) return; g_pad_scan=now;
    for(int i=0;i<sdl.NumJoysticks() && !g_pad_gc;i++) if(sdl.IsGameController(i)) g_pad_gc=sdl.Open(i);
    if(!g_pad_gc) return;
    LOG("controllers: %s\n",sdl.Name(g_pad_gc)? sdl.Name(g_pad_gc) : "connected"); }
  SDLGC* g=g_pad_gc;
  /* SDL names buttons by Xbox position: A bottom, B right, X left, Y top */
  static const struct { int sdl, bit; } m[]={{1,0},{0,1},{3,2},{2,3},{7,4},{8,5},{9,6},{10,7},{6,10},{4,11},{13,12},{11,13},{14,14},{12,15}};
  for(unsigned i=0;i<sizeof m/sizeof m[0];i++) if(sdl.Button(g,m[i].sdl)) *b|=1ull<<m[i].bit;
  if(sdl.Axis(g,4)>7700) *b|=1ull<<8; if(sdl.Axis(g,5)>7700) *b|=1ull<<9;
  /* SDL's Y axes point down, the Switch's up */
  i32 x=sdl.Axis(g,0), y=-(i32)sdl.Axis(g,1), x2=sdl.Axis(g,2), y2=-(i32)sdl.Axis(g,3);
  if(y>32767) y=32767; if(y2>32767) y2=32767;
  if(!*lx && !*ly){ *lx=dz(x); *ly=dz(y); } if(!*rx && !*ry){ *rx=dz(x2); *ry=dz(y2); }
}

/* ---------------------------------------------------------------- window */
static void set_fullscreen(int on){
  if(on==g_fullscreen) return;
  XEvent e; memset(&e,0,sizeof e); e.xclient.type=ClientMessage; e.xclient.window=win; e.xclient.message_type=a_state; e.xclient.format=32;
  e.xclient.data.l[0]=on? 1 : 0; e.xclient.data.l[1]=(long)a_fs; e.xclient.data.l[3]=1;
  XSendEvent(dpy,DefaultRootWindow(dpy),False,SubstructureRedirectMask|SubstructureNotifyMask,&e);
  if(on) XDefineCursor(dpy,win,g_blank); else XUndefineCursor(dpy,win);
  XFlush(dpy); g_fullscreen=on; }

static void set_title(const char* t){ XStoreName(dpy,win,t); XFlush(dpy); }

static const char* HELP=
  "SwitchRecomp - controls\n\n"
  "Controller: same as the Switch controller (buttons by position).\n\n"
  "Keyboard:\n"
  "  Arrows = D-Pad     W A S D = Left stick    I J K L = Right stick\n"
  "  X / Space = A    Z = B    C = X    V = Y\n"
  "  Q = L   E = R   1 = ZL   3 = ZR   Enter = +   Backspace = -\n\n"
  "F1 = this help   F3 = screenshot   F4 = FPS limit 60 / unlimited   F5 = diagnostic dump (trace)\n"
  "F8 = instant full-resolution screenshot (PNG in the captures folder)\n"
  "F11 / Alt+Enter = fullscreen   Esc = quit";

void video_init(const char* title){
  snprintf(g_title,sizeof g_title,"%s - SwitchRecomp  (F1 help)",title);
  XInitThreads();
  /* let mesa run its own glthread so the driver work isn't on our GPU thread (deck, AMD/intel on linux) */
  if(!getenv("SWR_NO_GLTHREAD")) setenv("mesa_glthread","true",0);
  dpy=XOpenDisplay(NULL); if(!dpy) fatal("No X11 display (DISPLAY is not set). Start the game from a desktop session.");
  int scr=DefaultScreen(dpy);
  static int att[]={GLX_X_RENDERABLE,True,GLX_DRAWABLE_TYPE,GLX_WINDOW_BIT,GLX_RENDER_TYPE,GLX_RGBA_BIT,GLX_RED_SIZE,8,GLX_GREEN_SIZE,8,GLX_BLUE_SIZE,8,GLX_ALPHA_SIZE,8,GLX_DEPTH_SIZE,24,GLX_DOUBLEBUFFER,True,None};
  int n=0; GLXFBConfig* fc=glXChooseFBConfig(dpy,scr,att,&n); if(!fc||!n) fatal("Your graphics driver has no OpenGL support for this display (GLX).");
  XVisualInfo* vi=glXGetVisualFromFBConfig(dpy,fc[0]);
  XSetWindowAttributes swa; memset(&swa,0,sizeof swa); swa.colormap=XCreateColormap(dpy,RootWindow(dpy,vi->screen),vi->visual,AllocNone);
  swa.background_pixel=BlackPixel(dpy,scr);
  swa.event_mask=KeyPressMask|KeyReleaseMask|StructureNotifyMask|FocusChangeMask;
  win=XCreateWindow(dpy,RootWindow(dpy,vi->screen),0,0,g_ww,g_wh,0,vi->depth,InputOutput,vi->visual,CWColormap|CWEventMask|CWBackPixel,&swa);
  XStoreName(dpy,win,g_title);
  { XClassHint ch={(char*)"switchrecomp",(char*)"SwitchRecomp"}; XSetClassHint(dpy,win,&ch); }
  wm_delete=XInternAtom(dpy,"WM_DELETE_WINDOW",False); XSetWMProtocols(dpy,win,&wm_delete,1);
  a_state=XInternAtom(dpy,"_NET_WM_STATE",False); a_fs=XInternAtom(dpy,"_NET_WM_STATE_FULLSCREEN",False);
  { static char none[8]; XColor black; memset(&black,0,sizeof black);
    Pixmap p=XCreateBitmapFromData(dpy,win,none,8,8); g_blank=XCreatePixmapCursor(dpy,p,p,&black,&black,0,0); XFreePixmap(dpy,p); }
  /* without this X11 sends a release before every auto-repeated press */
  XkbSetDetectableAutoRepeat(dpy,True,NULL);
  /* start fullscreen: the window manager reads the state when the window is mapped */
  if(g_want_fs){ XChangeProperty(dpy,win,a_state,XA_ATOM,32,PropModeReplace,(unsigned char*)&a_fs,1); XDefineCursor(dpy,win,g_blank); g_fullscreen=1; }
  XMapWindow(dpy,win); XFlush(dpy);
  sdl_load();
  g_start=plat_ns();
  extern void gpu_gl_start(void); gpu_gl_start();
}

/* GPU thread */
static GLXContext ctx, wctxs[3]; static GLXFBConfig wfc; static volatile int g_wnext;

static int xerr_ignore(Display* d, XErrorEvent* e){ (void)d;(void)e; return 0; }

int backend_gl_create(void){
  int scr=DefaultScreen(dpy);
  static int att[]={GLX_X_RENDERABLE,True,GLX_DRAWABLE_TYPE,GLX_WINDOW_BIT,GLX_RENDER_TYPE,GLX_RGBA_BIT,GLX_RED_SIZE,8,GLX_GREEN_SIZE,8,GLX_BLUE_SIZE,8,GLX_ALPHA_SIZE,8,GLX_DOUBLEBUFFER,True,None};
  int n=0; GLXFBConfig* fc=glXChooseFBConfig(dpy,scr,att,&n); if(!fc||!n) return 0;
  typedef GLXContext (*CCA)(Display*,GLXFBConfig,GLXContext,Bool,const int*);
  CCA cca=(CCA)glXGetProcAddressARB((const GLubyte*)"glXCreateContextAttribsARB");
  static const int vers[4][2]={{4,6},{4,5},{4,4},{4,3}};
  int (*old)(Display*,XErrorEvent*)=XSetErrorHandler(xerr_ignore);
  for(int i=0;i<4 && !ctx && cca;i++){ int ca[]={0x2091,vers[i][0],0x2092,vers[i][1],0x9126,0x00000001,None}; ctx=cca(dpy,fc[0],NULL,True,ca); XSync(dpy,False);
    if(ctx){ for(int k=0;k<3;k++) wctxs[k]=cca(dpy,fc[0],ctx,True,ca); XSync(dpy,False); } }
  wfc=fc[0];
  XSetErrorHandler(old);
  if(!ctx){ fatal("Your graphics driver does not support OpenGL 4.3.\nPlease update Mesa or your GPU driver."); return 0; }
  glXMakeCurrent(dpy,win,ctx);
  extern int g_opt_vsync;
  typedef void (*SIE)(Display*,GLXDrawable,int); typedef int (*SIM)(unsigned);
  SIE sie=(SIE)glXGetProcAddressARB((const GLubyte*)"glXSwapIntervalEXT"); SIM sim=(SIM)glXGetProcAddressARB((const GLubyte*)"glXSwapIntervalMESA");
  if(sie) sie(dpy,win,g_opt_vsync? 1 : 0); else if(sim) sim(g_opt_vsync? 1 : 0);
  return 1; }

void* backend_gl_proc(const char* n){ return (void*)glXGetProcAddressARB((const GLubyte*)n); }

void backend_gl_swap(void){ glXSwapBuffers(dpy,win); }

/* each compile thread takes the next shared context */
int backend_gl_worker_current(void){
  int k=__atomic_fetch_add(&g_wnext,1,__ATOMIC_ACQ_REL); GLXContext wctx= k<3? wctxs[k] : NULL;
  if(!wctx) return 0; int pa[]={GLX_PBUFFER_WIDTH,4,GLX_PBUFFER_HEIGHT,4,None};
  int (*old)(Display*,XErrorEvent*)=XSetErrorHandler(xerr_ignore); GLXPbuffer pb=glXCreatePbuffer(dpy,wfc,pa); XSync(dpy,False); XSetErrorHandler(old);
  if(pb && glXMakeContextCurrent(dpy,pb,pb,wctx)) return 1;
  return glXMakeContextCurrent(dpy,None,None,wctx)? 1 : 0; }

void backend_window_size(int* w, int* h){ *w=g_ww; *h=g_wh; }

void video_present_rgba(const u32* px, int w, int h, int stride){ extern void gpu_gl_present_cpu(const u32*,int,int,int); gpu_gl_present_cpu(px,w,h,stride); }

/* keys -> HidNpadButton bits, the same layout as on Windows */
static u64 keymap(KeySym k){
  switch(k){ case XK_x: case XK_space: return 1ull<<0; /* A */ case XK_z: return 1ull<<1; /* B */ case XK_c: return 1ull<<2; /* X */ case XK_v: return 1ull<<3; /* Y */
    case XK_q: return 1ull<<6; /* L */ case XK_e: return 1ull<<7; /* R */ case XK_1: return 1ull<<8; /* ZL */ case XK_3: return 1ull<<9; /* ZR */
    case XK_Return: case XK_KP_Enter: return 1ull<<10; /* + */ case XK_BackSpace: return 1ull<<11; /* - */
    case XK_Left: return 1ull<<12; case XK_Up: return 1ull<<13; case XK_Right: return 1ull<<14; case XK_Down: return 1ull<<15;
    /* stick keys, turned into axes later */
    case XK_a: return 1ull<<40; case XK_w: return 1ull<<41; case XK_d: return 1ull<<42; case XK_s: return 1ull<<43;
    case XK_j: return 1ull<<44; case XK_i: return 1ull<<45; case XK_l: return 1ull<<46; case XK_k: return 1ull<<47; }
  return 0; }

/* one-shot keys act on the first press only, not on auto-repeat */
static int key_once(KeySym k, int down){
  static KeySym held[8]; int i;
  for(i=0;i<8;i++) if(held[i]==k) break;
  if(!down){ if(i<8) held[i]=0; return 0; }
  if(i<8) return 0;
  for(i=0;i<8;i++) if(!held[i]){ held[i]=k; break; }
  return 1; }

static void key_down(XKeyEvent* e){
  KeySym k=XLookupKeysym(e,0);
  int once=key_once(k,1);
  if(k==XK_Return && (e->state&Mod1Mask)){ if(once) set_fullscreen(!g_fullscreen); return; }
  switch(k){
    case XK_Escape: g_quit=1; return;
    case XK_F11: if(once) set_fullscreen(!g_fullscreen); return;
    case XK_F1: if(once){ LOG("%s\n",HELP); plat_message(HELP,0,0); } return;
    case XK_F3: if(once) g_shot_req=1; return;
    case XK_F7: if(once) LOG("F7 video recording is only available on Windows\n"); return;
    case XK_F8: if(once){ extern volatile int g_shot8_req; g_shot8_req=1; } return;
    case XK_F5: if(once){ extern void gpu_gl_dump_rts(void); extern void kernel_dump_threads(void); extern __thread FILE* g_log_redirect;
        plat_mkdir("captures"); FILE* f=fopen("captures/threads.txt","w");   /* written next to the trace */
        if(f){ g_log_redirect=f; kernel_dump_threads(); g_log_redirect=NULL; fclose(f); }
        kernel_dump_threads(); gpu_gl_dump_rts(); } return;
    case XK_F4: if(once){ g_opt_fps= g_opt_fps==0? 1 : 0; char t[300]; snprintf(t,sizeof t,"%s  |  FPS limit: %s",g_title,g_opt_fps==0? "unlimited" : "60"); set_title(t);
        extern void ini_set(const char*, const char*); char v[8]; snprintf(v,sizeof v,"%d",g_opt_fps); ini_set("fps_limit",v); } return;
  }
  keys_held|=keymap(k); }

static i32 axis(u64 b, int neg, int pos){ return ((b>>pos)&1? 32767:0) - ((b>>neg)&1? 32767:0); }

void video_poll(void){
  while(XPending(dpy)){ XEvent e; XNextEvent(dpy,&e);
    if(e.type==KeyPress) key_down(&e.xkey);
    else if(e.type==KeyRelease){ KeySym k=XLookupKeysym(&e.xkey,0); key_once(k,0); keys_held&=~keymap(k); }
    else if(e.type==FocusOut) keys_held=0;
    else if(e.type==ConfigureNotify){ g_ww=e.xconfigure.width>0? e.xconfigure.width : 1; g_wh=e.xconfigure.height>0? e.xconfigure.height : 1; }
    else if(e.type==ClientMessage && (Atom)e.xclient.data.l[0]==wm_delete){ LOG("window: closed by the user\n"); g_quit=1; } }
  u64 k=keys_held, b=k&0xFFFFull;
  i32 lx=axis(k,40,42), ly=axis(k,43,41), rx=axis(k,44,46), ry=axis(k,47,45);
  pad_poll(&b,&lx,&ly,&rx,&ry);
  /* stick-as-button bits 16..23 */
  if(lx<-16000) b|=1ull<<16; if(ly>16000) b|=1ull<<17; if(lx>16000) b|=1ull<<18; if(ly<-16000) b|=1ull<<19;
  if(rx<-16000) b|=1ull<<20; if(ry>16000) b|=1ull<<21; if(rx>16000) b|=1ull<<22; if(ry<-16000) b|=1ull<<23;
  /* test input: --press=hex --press_at=ms, or --script=ms:hex,ms:hex,... */
  u64 el=(plat_ns()-g_start)/1000000ull;
  { static u64 at=~0ull; if(at==~0ull){ const char* e=getenv("SWR_TDUMP_AT"); at= e? strtoull(e,NULL,10)*1000ull : 0; }   /* tests: game thread dump at N s */
    if(at && el>=at){ static int n; const char* ev=getenv("SWR_TDUMP_EVERY"); at= ev && ++n<6? el+strtoull(ev,NULL,10)*1000ull : 0; extern void kernel_dump_threads(void); kernel_dump_threads(); } }
  if(g_press && el>=g_press_at && el<g_press_at+g_press_len) b|=g_press;
  if(g_script[0]){ const char* p=g_script; while(*p){ u64 t=strtoull(p,(char**)&p,10); if(*p!=':') break; p++; u64 m=strtoull(p,(char**)&p,16); if(el>=t && el<t+g_press_len) b|=m; if(*p==',') p++; } }
  g_pad.buttons=b; g_pad.lx=lx; g_pad.ly=ly; g_pad.rx=rx; g_pad.ry=ry;
}

void frontend_frame(void){
  extern u64 gpu_gl_screenshot(const char* path);
  if(g_shot_req){ g_shot_req=0; char p[300]; time_t now=time(NULL); struct tm t; localtime_r(&now,&t); plat_mkdir("captures");
    snprintf(p,sizeof p,"captures/screenshot_%04d%02d%02d_%02d%02d%02d",t.tm_year+1900,t.tm_mon+1,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec); gpu_gl_screenshot(p); }
  u64 el=plat_ns()-g_start;
  if((g_maxframes>=0 && g_frames>=g_maxframes) || (g_seconds && el>=(u64)g_seconds*1000000000ull)){
    u64 h=gpu_gl_screenshot(g_shot[0]? g_shot : NULL);
    fprintf(stderr,"\nframes=%d hash=%016llx\n",g_frames,(unsigned long long)h);
    extern void save_hints(void); save_hints(); fflush(NULL); _exit(0); }
}

int backend_glq_ok(void){ return 1; }   /* GL calls go through glq */
