/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* win32 backend: window, WGL, keyboard, XInput.
   F1 help  F3 screenshot  F4 fps limit  F11 fullscreen  Shift+Esc quit */
#include "rt.h"
#include "audio.h"
#include <windows.h>
#include <math.h>
#include <mmsystem.h>  /* timeBeginPeriod (windows.h leaves it out with WIN32_LEAN_AND_MEAN) */

/* laptops with two GPUs: ask the NVIDIA/AMD drivers for the dedicated one, not the integrated one */
__declspec(dllexport) DWORD NvOptimusEnablement=1;
__declspec(dllexport) int AmdPowerXpressRequestHighPerformance=1;

static HWND g_hwnd; static HDC g_hdc; static HGLRC g_ctx, g_wctx[3]; static volatile int g_wnext;
static volatile int g_ww=1280, g_wh=720; static u64 keys_held; static int g_fullscreen=0, g_want_fs=1;
static WINDOWPLACEMENT g_wp={sizeof(WINDOWPLACEMENT)};

static const char* g_title_was;
volatile int g_flight_req;   /* F5: vk_flight.c writes the flight recording */
static int g_seconds=0; static char g_shot[1024]; static u64 g_start; static volatile int g_shot_req;
static char g_script[8192];
/* unattended runs: "ms,ms,..." since start, the F5 trace (trace_at) and the F3 screenshot (shots_at) at those times */
static char g_trace_at[1024], g_shots_at[1024];
static HANDLE g_ready;
static volatile LONG g_mouse_mode, g_mouse_invert, g_mouse_dx, g_mouse_dy;
static volatile LONG g_mouse_sensitivity=100; /* F1 scale, curved: 10% = 5x, 500% = 1250x */
static int g_controls_open;
static u64 g_mouse_time;
static i32 g_mouse_rx, g_mouse_ry;
static void mouse_capture(void){
  int active=g_mouse_mode && !g_controls_open && GetForegroundWindow()==g_hwnd && !IsIconic(g_hwnd);
  if(active){ RECT r; GetClientRect(g_hwnd,&r); MapWindowPoints(g_hwnd,NULL,(POINT*)&r,2); ClipCursor(&r); }
  else ClipCursor(NULL);
  InterlockedExchange(&g_mouse_dx,0); InterlockedExchange(&g_mouse_dy,0);
}

extern int g_opt_fps;

static int g_ls_mode=0; static char g_ls_path[1024];

void backend_ini(const char* k, const char* v){
  if(!strcmp(k,"fullscreen")) g_want_fs=atoi(v);
  else if(!strcmp(k,"lossless_scaling")) g_ls_mode=atoi(v);
  else if(!strcmp(k,"lossless_path")) strncpy(g_ls_path,v,1023);
  else if(!strcmp(k,"seconds")) g_seconds=atoi(v);
  else if(!strcmp(k,"shot")) strncpy(g_shot,v,1023);
  else if(!strcmp(k,"audio_slowdown")) g_opt_audio_slowdown=atoi(v)!=0;
  else if(!strcmp(k,"mouse_right_stick")) g_mouse_mode=atoi(v)!=0;
  else if(!strcmp(k,"mouse_sensitivity")){
    double n=atof(v); if(!(n>0)) n=1; if(n<.1) n=.1; if(n>5) n=5; g_mouse_sensitivity=(LONG)(n*100+.5);
  }
  else if(!strcmp(k,"mouse_invert")) g_mouse_invert=atoi(v)&3;
  else if(!strcmp(k,"script")) strncpy(g_script,v,8191);
  else if(!strcmp(k,"trace_at")) strncpy(g_trace_at,v,1023);
  else if(!strcmp(k,"shots_at")) strncpy(g_shots_at,v,1023);
}

int video_has_gl(void){ return 1; }

void win_fatal_box(const char* msg){ MessageBoxA(g_hwnd,msg,"SwitchRecomp",MB_OK|MB_ICONERROR); }

/* ---------------------------------------------------------------- XInput, loaded dynamically */
typedef struct { WORD wButtons; BYTE bLeftTrigger, bRightTrigger; SHORT sThumbLX, sThumbLY, sThumbRX, sThumbRY; } XGP;

typedef struct { DWORD dwPacketNumber; XGP Gamepad; } XST;

typedef DWORD (WINAPI *XGetState)(DWORD, XST*);
static XGetState xget;

static void xinput_load(void){
  const char* dl[]={"xinput1_4.dll","xinput1_3.dll","xinput9_1_0.dll"};
  for(int i=0;i<3 && !xget;i++){ HMODULE h=LoadLibraryA(dl[i]); if(h) xget=(XGetState)GetProcAddress(h,"XInputGetState"); }
}

static void set_fullscreen(int on){
  DWORD st=GetWindowLong(g_hwnd,GWL_STYLE);
  if(on && !g_fullscreen){
    MONITORINFO mi={sizeof mi};
    if(GetWindowPlacement(g_hwnd,&g_wp) && GetMonitorInfo(MonitorFromWindow(g_hwnd,MONITOR_DEFAULTTOPRIMARY),&mi)){
      SetWindowLong(g_hwnd,GWL_STYLE,st&~WS_OVERLAPPEDWINDOW);
      SetWindowPos(g_hwnd,HWND_TOP,mi.rcMonitor.left,mi.rcMonitor.top,mi.rcMonitor.right-mi.rcMonitor.left,mi.rcMonitor.bottom-mi.rcMonitor.top,SWP_NOOWNERZORDER|SWP_FRAMECHANGED); }
    ShowCursor(FALSE); g_fullscreen=1;
  } else if(!on && g_fullscreen){
    SetWindowLong(g_hwnd,GWL_STYLE,st|WS_OVERLAPPEDWINDOW); SetWindowPlacement(g_hwnd,&g_wp);
    SetWindowPos(g_hwnd,NULL,0,0,0,0,SWP_NOMOVE|SWP_NOSIZE|SWP_NOZORDER|SWP_NOOWNERZORDER|SWP_FRAMECHANGED);
    ShowCursor(TRUE); g_fullscreen=0; }
}

/* optional: launch the lossless scaling app next to the game (lossless_scaling=1). this path doesn't touch its
   dll, the app just captures our window */
static int file_exists_(const char* p){ DWORD a=GetFileAttributesA(p); return a!=INVALID_FILE_ATTRIBUTES && !(a&FILE_ATTRIBUTE_DIRECTORY); }

static int ls_find(char* out){
  if(g_ls_path[0] && file_exists_(g_ls_path)){ strcpy(out,g_ls_path); return 1; }
  static char libs[10][600]; int nl=0;
  typedef LONG (WINAPI*RGV)(HKEY,LPCSTR,LPCSTR,DWORD,LPDWORD,PVOID,LPDWORD);
  HMODULE a=LoadLibraryA("advapi32.dll"); RGV rgv= a? (RGV)GetProcAddress(a,"RegGetValueA") : NULL;
  char sp[600]; DWORD sz=sizeof sp;
  if(rgv && rgv(HKEY_CURRENT_USER,"Software\\Valve\\Steam","SteamPath",0x0000ffff,NULL,sp,&sz)==0 && sp[0]){ strncpy(libs[nl],sp,599); libs[nl][599]=0; nl++; }
  strcpy(libs[nl++],"C:\\Program Files (x86)\\Steam"); strcpy(libs[nl++],"C:\\Program Files\\Steam");
  { char vdf[700]; snprintf(vdf,sizeof vdf,"%s\\steamapps\\libraryfolders.vdf",libs[0]); FILE* f=fopen(vdf,"r");
    if(f){ char ln[800]; while(fgets(ln,sizeof ln,f) && nl<10){ char* p=strstr(ln,"\"path\""); if(!p) continue; p=strchr(p+6,'"'); if(!p) continue; p++; char* e=strchr(p,'"'); if(!e) continue; *e=0;
        int k=0; for(char* q=p;*q && k<599;q++){ if(q[0]=='\\' && q[1]=='\\') q++; libs[nl][k++]=*q; } libs[nl][k]=0; nl++; }
      fclose(f); } }
  for(int i=0;i<nl;i++){ char c[800]; snprintf(c,sizeof c,"%s\\steamapps\\common\\Lossless Scaling\\LosslessScaling.exe",libs[i]); for(char* q=c;*q;q++) if(*q=='/') *q='\\';
    if(file_exists_(c)){ strcpy(out,c); return 1; } }
  return 0; }

static void ls_launch(void){
  if(!g_ls_mode) return;
  if(FindWindowA(NULL,"Lossless Scaling")){ LOG("lossless scaling: already running\n"); return; }
  char p[1024]; if(!ls_find(p)){ LOG("lossless scaling: not found (install it from Steam, or set lossless_path=C:\\...\\LosslessScaling.exe in the .ini)\n"); return; }
  char dir[1024]; strcpy(dir,p); char* sl=strrchr(dir,'\\'); if(sl) *sl=0;
  STARTUPINFOA si; memset(&si,0,sizeof si); si.cb=sizeof si; si.dwFlags=STARTF_USESHOWWINDOW; si.wShowWindow=SW_SHOWMINNOACTIVE; PROCESS_INFORMATION pi;
  if(CreateProcessA(p,NULL,NULL,NULL,FALSE,0,NULL,dir,&si,&pi)){ CloseHandle(pi.hThread); CloseHandle(pi.hProcess); LOG("lossless scaling: started (%s)\n",p); }
  else LOG("lossless scaling: could not open %s (error %lu)\n",p,(unsigned long)GetLastError());
  SetForegroundWindow(g_hwnd); }

/* window title update for lsfg on/off, posted from the GPU thread */
static char g_title[256];

void backend_lsfg_state(int on, int mult){ if(g_hwnd) PostMessageA(g_hwnd,WM_APP+7,(WPARAM)on,(LPARAM)mult); }

/* F7 recording state in the title bar */
volatile int g_rec_toggle;

void backend_rec_state(int on){ if(g_hwnd) PostMessageA(g_hwnd,WM_APP+8,(WPARAM)on,0); }

/* F1: controls and the mouse/audio options, drawn like the recompiler's settings page (same colors and layout: light
   gray page, rows with the value in cyan, cyan frame on the selected row, button hints at the bottom). the options
   get saved in the game's ini */
#define CW_W 760
#define CW_H 620
#define CW_ROW 46
#define CW_LIST_Y 84
#define CW_NOPT 4
static const char* inversion_names[]={"Normal", "Left / right inverted", "Up / down inverted", "Both inverted"};
static const char* const cw_keys[][2]={
  {"D-Pad","Arrows"}, {"Left stick","W A S D"}, {"Right stick","I J K L (or the mouse)"}, {"A  B  X  Y","X/Space  Z  C  V"},
  {"L  R  ZL  ZR","Q  E  1  3"}, {"+  -","Enter  Backspace"}, {"Controller","Xbox (XInput) pad"}};
static const char* const cw_fkeys[][2]={
  {"F1","This page"}, {"F3","Screenshot"}, {"F4","FPS limit on/off"}, {"F5","Save the last frames (flight recorder)"},
  {"F7","Record video"}, {"F8","Full resolution capture"}, {"F11","Fullscreen"}, {"Shift+Esc","Quit the game"}};
static int cw_sel;
static HFONT cw_font(int px, int weight){ return CreateFontA(-px,0,0,0,weight,0,0,0,DEFAULT_CHARSET,OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,CLEARTYPE_QUALITY,DEFAULT_PITCH,"Segoe UI"); }
static void cw_value(int i, char* o, size_t n){
  switch(i){
    case 0: snprintf(o,n,"%s",g_mouse_mode?"Mouse":"I J K L keys"); break;
    case 1: snprintf(o,n,"%s",inversion_names[g_mouse_invert&3]); break;
    case 2: snprintf(o,n,"%s%ld%%%s",g_mouse_sensitivity>10?"<  ":"",g_mouse_sensitivity,g_mouse_sensitivity<500?"  >":""); break;
    default: snprintf(o,n,"%s",g_opt_audio_slowdown?"On":"Off"); } }
static void cw_change(int i, int dir){
  extern void ini_set(const char*,const char*); char value[8];
  if(i==0){ g_mouse_mode=!g_mouse_mode; snprintf(value,sizeof value,"%ld",g_mouse_mode); ini_set("mouse_right_stick",value); }
  else if(i==1){ g_mouse_invert=(g_mouse_invert+(dir<0?3:1))&3; snprintf(value,sizeof value,"%ld",g_mouse_invert); ini_set("mouse_invert",value); }
  else if(i==2){ LONG n= dir==0? 100 : g_mouse_sensitivity+dir*10; if(n<10) n=10; if(n>500) n=500; g_mouse_sensitivity=n;
    snprintf(value,sizeof value,"%.2f",n/100.0); ini_set("mouse_sensitivity",value); }
  else { g_opt_audio_slowdown=!g_opt_audio_slowdown; ini_set("audio_slowdown",g_opt_audio_slowdown?"1":"0"); } }
static void cw_rect(HDC dc, int x, int y, int w, int h, COLORREF c){ RECT r={x,y,x+w,y+h}; HBRUSH b=CreateSolidBrush(c); FillRect(dc,&r,b); DeleteObject(b); }
static void cw_text(HDC dc, HFONT f, const char* t, int x, int y, int w, int h, UINT fmt, COLORREF c){
  RECT r={x,y,x+w,y+h}; SelectObject(dc,f); SetTextColor(dc,c); DrawTextA(dc,t,-1,&r,fmt|DT_SINGLELINE|DT_VCENTER|DT_NOPREFIX); }
/* a round button hint like the recompiler's bottom bar: dark circle with the letter, then the action */
static int cw_hint(HDC dc, HFONT fl, HFONT ft, int right, int y, const char* key, const char* what){
  SIZE sz; SelectObject(dc,ft); GetTextExtentPoint32A(dc,what,(int)strlen(what),&sz);
  int x=right-sz.cx; cw_text(dc,ft,what,x,y-16,sz.cx+2,32,DT_LEFT,RGB(45,45,45));
  HBRUSH b=CreateSolidBrush(RGB(45,45,45)); HPEN pn=(HPEN)GetStockObject(NULL_PEN); SelectObject(dc,b); SelectObject(dc,pn);
  int cx=x-24; Ellipse(dc,cx-14,y-14,cx+15,y+15); DeleteObject(b);
  cw_text(dc,fl,key,cx-14,y-15,29,29,DT_CENTER,RGB(255,255,255));
  return cx-30; }
static void cw_paint(HWND h){
  PAINTSTRUCT ps; HDC wdc=BeginPaint(h,&ps);
  HDC dc=CreateCompatibleDC(wdc); HBITMAP bm=CreateCompatibleBitmap(wdc,CW_W,CW_H); HGDIOBJ obm=SelectObject(dc,bm);
  SetBkMode(dc,TRANSPARENT);
  HFONT fh=cw_font(30,FW_NORMAL), fb=cw_font(21,FW_NORMAL), fs=cw_font(17,FW_NORMAL), fk=cw_font(16,FW_SEMIBOLD);
  const COLORREF bg=RGB(235,235,235), text=RGB(45,45,45), dim=RGB(130,130,130), accent=RGB(8,175,200), sep=RGB(205,205,205);
  cw_rect(dc,0,0,CW_W,CW_H,bg);
  cw_text(dc,fh,"Controls",40,18,500,48,DT_LEFT,text);
  cw_rect(dc,20,66,CW_W-40,1,text);
  static const char* const names[CW_NOPT]={"Right stick","Mouse direction","Mouse sensitivity","Smooth audio slowdown"};
  for(int i=0;i<CW_NOPT;i++){ int y=CW_LIST_Y+i*CW_ROW; char v[64]; cw_value(i,v,sizeof v);
    if(i) cw_rect(dc,40,y,CW_W-80,1,sep);
    cw_text(dc,fb,names[i],64,y,360,CW_ROW,DT_LEFT,text);
    cw_text(dc,fb,v,CW_W-380,y,316,CW_ROW,DT_RIGHT,accent); }
  { /* the selected row: cyan frame with round corners, like the recompiler's cursor */
    int y=CW_LIST_Y+cw_sel*CW_ROW; HPEN pn=CreatePen(PS_SOLID,4,accent); SelectObject(dc,pn); SelectObject(dc,GetStockObject(NULL_BRUSH));
    RoundRect(dc,38,y-1,CW_W-38,y+CW_ROW+2,10,10); DeleteObject(pn); }
  int ky=CW_LIST_Y+CW_NOPT*CW_ROW+22;
  cw_text(dc,fs,"KEYBOARD",64,ky,300,24,DT_LEFT,dim); cw_text(dc,fs,"KEYS",410,ky,300,24,DT_LEFT,dim);
  for(int i=0;i<7;i++){ int y=ky+28+i*28;
    cw_text(dc,fs,cw_keys[i][0],64,y,120,28,DT_LEFT,text); cw_text(dc,fs,cw_keys[i][1],170,y,220,28,DT_LEFT,accent); }
  for(int i=0;i<8;i++){ int y=ky+28+i*28;
    cw_text(dc,fs,cw_fkeys[i][0],410,y,50,28,DT_LEFT,accent); cw_text(dc,fs,cw_fkeys[i][1],460,y,260,28,DT_LEFT,text); }
  cw_rect(dc,30,CW_H-60,CW_W-60,1,text);
  int x=cw_hint(dc,fk,fb,CW_W-40,CW_H-30,"A","Change");
  cw_hint(dc,fk,fb,x-20,CW_H-30,"B","Back");
  cw_text(dc,fs,"Esc or F1: back to the game",40,CW_H-46,300,32,DT_LEFT,dim);
  BitBlt(wdc,0,0,CW_W,CW_H,dc,0,0,SRCCOPY);
  DeleteObject(fh); DeleteObject(fb); DeleteObject(fs); DeleteObject(fk);
  SelectObject(dc,obm); DeleteObject(bm); DeleteDC(dc); EndPaint(h,&ps); }
static LRESULT CALLBACK controls_proc(HWND h,UINT m,WPARAM w,LPARAM l){
  switch(m){
    case WM_ERASEBKGND: return 1;
    case WM_PAINT: cw_paint(h); return 0;
    case WM_KEYDOWN:
      if(w==VK_ESCAPE || w==VK_F1 || w==VK_BACK || w=='Z'){ DestroyWindow(h); return 0; }
      if(w==VK_UP || w=='W'){ if(cw_sel>0) cw_sel--; }
      else if(w==VK_DOWN || w=='S'){ if(cw_sel<CW_NOPT-1) cw_sel++; }
      else if(w==VK_LEFT || w=='A') cw_change(cw_sel,-1);
      else if(w==VK_RIGHT || w=='D' || w==VK_RETURN || w==VK_SPACE || w=='X') cw_change(cw_sel,+1);
      else if(w=='R' && cw_sel==2) cw_change(2,0);
      InvalidateRect(h,NULL,FALSE); return 0;
    case WM_MOUSEMOVE: case WM_LBUTTONDOWN: case WM_RBUTTONDOWN: {
      int x=(short)LOWORD(l), y=(short)HIWORD(l), i=(y-CW_LIST_Y)/CW_ROW;
      if(y>=CW_LIST_Y && i>=0 && i<CW_NOPT && x>=38 && x<CW_W-38){
        if(i!=cw_sel){ cw_sel=i; InvalidateRect(h,NULL,FALSE); }
        /* sensitivity: the left half lowers it, the right half raises it */
        if(m!=WM_MOUSEMOVE){ cw_change(i, m==WM_RBUTTONDOWN? -1 : i==2 && x<CW_W/2? -1 : +1); InvalidateRect(h,NULL,FALSE); } }
      else if(m==WM_LBUTTONDOWN && y>CW_H-60) DestroyWindow(h);
      return 0; }
    case WM_CLOSE: DestroyWindow(h); return 0; }
  return DefWindowProcA(h,m,w,l); }
static void show_controls(void){
  int fs=g_fullscreen; g_controls_open=1; keys_held=0; mouse_capture(); if(fs) set_fullscreen(0);
  HINSTANCE hi=GetModuleHandle(NULL); WNDCLASSA wc={0}; wc.lpfnWndProc=controls_proc; wc.hInstance=hi;
  wc.hCursor=LoadCursor(NULL,IDC_ARROW); wc.hbrBackground=NULL; wc.lpszClassName="SwitchRecompControls"; RegisterClassA(&wc);
  RECT parent; GetWindowRect(g_hwnd,&parent); RECT r={0,0,CW_W,CW_H}; AdjustWindowRect(&r,WS_CAPTION|WS_SYSMENU,FALSE);
  int ww=r.right-r.left, wh=r.bottom-r.top, x=(parent.left+parent.right-ww)/2, y=(parent.top+parent.bottom-wh)/2; if(x<0) x=0; if(y<0) y=0;
  HWND h=CreateWindowA(wc.lpszClassName,"SwitchRecomp - Controls",WS_CAPTION|WS_SYSMENU,x,y,ww,wh,g_hwnd,NULL,hi,NULL);
  EnableWindow(g_hwnd,FALSE); ShowWindow(h,SW_SHOW); SetFocus(h);
  MSG msg; while(IsWindow(h) && GetMessageA(&msg,NULL,0,0)>0){ TranslateMessage(&msg); DispatchMessageA(&msg); }
  EnableWindow(g_hwnd,TRUE); SetForegroundWindow(g_hwnd); if(fs) set_fullscreen(1); g_controls_open=0; keys_held=0; mouse_capture();
}
static u64 keymap(WPARAM k){
  switch(k){ case 'X': case VK_SPACE: return 1ull<<0; case 'Z': return 1ull<<1; case 'C': return 1ull<<2; case 'V': return 1ull<<3;
    case 'Q': return 1ull<<6; case 'E': return 1ull<<7; case '1': return 1ull<<8; case '3': return 1ull<<9;
    case VK_RETURN: return 1ull<<10; case VK_BACK: return 1ull<<11;
    case VK_LEFT: return 1ull<<12; case VK_UP: return 1ull<<13; case VK_RIGHT: return 1ull<<14; case VK_DOWN: return 1ull<<15;
    /* stick keys, turned into axes later */
    case 'A': return 1ull<<40; case 'W': return 1ull<<41; case 'D': return 1ull<<42; case 'S': return 1ull<<43;
    case 'J': return 1ull<<44; case 'I': return 1ull<<45; case 'L': return 1ull<<46; case 'K': return 1ull<<47; }
  return 0; }

static LRESULT CALLBACK wndproc(HWND h, UINT m, WPARAM w, LPARAM l){
  switch(m){
  case WM_TIMER: if(w==77){ KillTimer(h,77); if(g_title_was){ SetWindowTextA(h,g_title_was); g_title_was=NULL; } return 0; } break;
  case WM_CLOSE: LOG("window: closed by the user or the system\n"); g_quit=1; return 0;
  case WM_QUERYENDSESSION: LOG("window: Windows is logging off / shutting down\n"); break;
  case WM_POWERBROADCAST: if(w==PBT_APMSUSPEND) LOG("window: the PC is going to sleep\n"); else if(w==PBT_APMRESUMEAUTOMATIC) LOG("window: the PC resumed from sleep\n"); break;
  case WM_MOVE: mouse_capture(); break;
  case WM_SETFOCUS: mouse_capture(); return 0;
  case WM_SIZE: mouse_capture(); g_ww=LOWORD(l); g_wh=HIWORD(l); if(g_ww<1) g_ww=1; if(g_wh<1) g_wh=1; { extern void gpu_eden_resize(int,int), nvn_window_resized(int,int); gpu_eden_resize(g_ww,g_wh); nvn_window_resized(g_ww,g_wh); } return 0;
  case WM_SYSKEYDOWN: if(w==VK_RETURN){ if(!(l&(1<<30))) set_fullscreen(!g_fullscreen); return 0; } if(w==VK_F4) break; /* let Alt+F4 through */
  /* fallthrough */
  case WM_KEYDOWN:
    /* a held key auto repeats (bit 30), the one shot actions below only fire on the first press or holding F7
       would start and stop the recording every frame */
    if((l&(1<<30)) && (w==VK_ESCAPE||w==VK_F1||w==VK_F3||w==VK_F4||w==VK_F5||w==VK_F7||w==VK_F8||w==VK_F11)) return 0;
    if(w==VK_ESCAPE){ if(GetKeyState(VK_SHIFT)&0x8000) g_quit=1; return 0; }   /* Shift+Esc: Esc alone is one key away from the F keys */
    if(w==VK_F11){ set_fullscreen(!g_fullscreen); return 0; }
    if(w==VK_F1){ show_controls(); return 0; }
    if(w==VK_F3){ g_shot_req=1; return 0; }
    if(w==VK_F7){ g_rec_toggle=1; return 0; }
    if(w==VK_F8){ extern volatile int g_shot8_req; g_shot8_req=1; return 0; }
    if(w==VK_F5){ static char was[256]; g_flight_req=1; if(!g_title_was){ GetWindowTextA(h,was,sizeof was); g_title_was=was; } SetWindowTextA(h,"Saving the last frames..."); SetTimer(h,77,3000,NULL); extern void gpu_gl_dump_rts(void); extern void kernel_dump_threads(void); extern __thread FILE* g_log_redirect;
      CreateDirectoryA("captures",NULL); FILE* f=fopen("captures/threads.txt","w");   /* written next to the trace */
      if(f){ g_log_redirect=f; kernel_dump_threads(); g_log_redirect=NULL; fclose(f); }
      kernel_dump_threads(); gpu_gl_dump_rts(); return 0; }
    if(w==VK_F4){ g_opt_fps= g_opt_fps==0? 1 : 0; char t[64]; snprintf(t,sizeof t,"FPS limit: %s",g_opt_fps==0? "unlimited" : "60"); SetWindowTextA(h,t); { extern void ini_set(const char*, const char*); char v[8]; snprintf(v,sizeof v,"%d",g_opt_fps); ini_set("fps_limit",v); } return 0; }
    keys_held|=keymap(w); return 0;
  case WM_SYSKEYUP: case WM_KEYUP: keys_held&=~keymap(w); return 0;
  case WM_KILLFOCUS: keys_held=0; ClipCursor(NULL); InterlockedExchange(&g_mouse_dx,0); InterlockedExchange(&g_mouse_dy,0); return 0;
  case WM_INPUT: {
    RAWINPUT raw; UINT size=sizeof raw;
    if(g_mouse_mode && !g_controls_open && GetForegroundWindow()==h && GetRawInputData((HRAWINPUT)l,RID_INPUT,&raw,&size,sizeof(RAWINPUTHEADER))==sizeof raw && raw.header.dwType==RIM_TYPEMOUSE && !(raw.data.mouse.usFlags&MOUSE_MOVE_ABSOLUTE)){
      InterlockedExchangeAdd(&g_mouse_dx,raw.data.mouse.lLastX); InterlockedExchangeAdd(&g_mouse_dy,raw.data.mouse.lLastY);
    }break;
  }
  case WM_APP+8: { char t[300]; if(w) snprintf(t,sizeof t,"%s  |  RECORDING (F7 to stop)",g_title); else snprintf(t,sizeof t,"%s  |  Video saved in the recordings folder",g_title); SetWindowTextA(h,t); return 0; }
  case WM_APP+7: { char t[160]; if(w) snprintf(t,sizeof t,"%s  |  LSFG ON x%d (frame generation)",g_title,(int)l); else snprintf(t,sizeof t,"%s  |  LSFG standby",g_title); SetWindowTextA(h,t); return 0; }
  case WM_SETCURSOR: if((g_fullscreen || (g_mouse_mode && !g_controls_open && GetForegroundWindow()==h)) && LOWORD(l)==HTCLIENT){ SetCursor(NULL); return TRUE; } break;
  }
  return DefWindowProcA(h,m,w,l);
}

static char g_title[256];

/* window + message pump on their own thread so nothing stalls them */
static void win_thread(void* arg){ (void)arg;
  HINSTANCE hi=GetModuleHandle(NULL);
  WNDCLASSA wc={0}; wc.style=CS_OWNDC; wc.lpfnWndProc=wndproc; wc.hInstance=hi; wc.hCursor=LoadCursor(NULL,IDC_ARROW);
  wc.hIcon=LoadIcon(hi,MAKEINTRESOURCE(1)); if(!wc.hIcon) wc.hIcon=LoadIcon(NULL,IDI_APPLICATION);
  wc.hbrBackground=(HBRUSH)GetStockObject(BLACK_BRUSH); wc.lpszClassName="SwitchRecompWnd";
  RegisterClassA(&wc);
  RECT r={0,0,1280,720}; AdjustWindowRect(&r,WS_OVERLAPPEDWINDOW,FALSE);
  g_hwnd=CreateWindowA("SwitchRecompWnd",g_title,WS_OVERLAPPEDWINDOW|WS_VISIBLE,CW_USEDEFAULT,CW_USEDEFAULT,r.right-r.left,r.bottom-r.top,NULL,NULL,hi,NULL);
  RAWINPUTDEVICE rid={1,2,0,g_hwnd}; RegisterRawInputDevices(&rid,1,sizeof rid);
  g_hdc=GetDC(g_hwnd);
  PIXELFORMATDESCRIPTOR pfd={sizeof pfd,1,PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER,PFD_TYPE_RGBA,32};
  pfd.cDepthBits=24; pfd.cStencilBits=8; pfd.iLayerType=PFD_MAIN_PLANE;
  SetPixelFormat(g_hdc,ChoosePixelFormat(g_hdc,&pfd),&pfd);
  if(g_want_fs) set_fullscreen(1);
  ls_launch(); mouse_capture();
  SetEvent(g_ready);
  MSG msg; while(GetMessageA(&msg,NULL,0,0)>0){ TranslateMessage(&msg); DispatchMessageA(&msg); }
  g_quit=1;
}

void video_init(const char* title){
  snprintf(g_title,sizeof g_title,"%s - SwitchRecomp  (F1 help)",title);
  xinput_load(); timeBeginPeriod(1);
  g_ready=CreateEventA(NULL,TRUE,FALSE,NULL);
  _beginthread(win_thread,1<<20,NULL);
  WaitForSingleObject(g_ready,INFINITE);
  g_start=plat_ns(); if(getenv("SWR_REC")) g_rec_toggle=1;
  { /* under proton the driver is mesa, glthread has to be turned on from the steam launch options */
    HMODULE nt=GetModuleHandleA("ntdll.dll"); if(nt && GetProcAddress(nt,"wine_get_version"))
      LOG("Proton/Wine detected: for more FPS put   mesa_glthread=true %%command%%   in the game's Steam launch options (moves the OpenGL driver work to another core)\n"); }
  { extern int g_gpu_nvn; extern void nvn_window_set(void*); if(g_gpu_nvn){ nvn_window_set(g_hwnd); return; } }   /* SWR_GPU=nvn: the NVN layer's backend draws on this window, no OpenGL, no Eden */
  { /* SWR_GPU=eden: Eden's Vulkan GPU on this window (gpu_eden.c); OpenGL when it is off or did not start */
    extern int gpu_eden_wanted(void), gpu_eden_start(void*,int,int); extern void gpu_gl_start(void);
    RECT cr; GetClientRect(g_hwnd,&cr); int cw=cr.right-cr.left, ch=cr.bottom-cr.top;
    if(!(gpu_eden_wanted() && gpu_eden_start(g_hwnd,cw>0? cw : 1280,ch>0? ch : 720))) gpu_gl_start(); }
}

/* ---------------------------------------------------------------- GL, GPU thread */
typedef HGLRC (WINAPI *CCA)(HDC,HGLRC,const int*);
typedef BOOL (WINAPI *SWI)(int);

int backend_gl_create(void){
  HGLRC tmp=wglCreateContext(g_hdc); if(!tmp) return 0;
  wglMakeCurrent(g_hdc,tmp);
  CCA cca=(CCA)wglGetProcAddress("wglCreateContextAttribsARB");
  static const int vers[4][2]={{4,6},{4,5},{4,4},{4,3}};
  for(int i=0;i<4 && !g_ctx && cca;i++){ int ca[]={0x2091,vers[i][0],0x2092,vers[i][1],0x9126,0x00000001,0}; g_ctx=cca(g_hdc,NULL,ca);
    if(g_ctx) for(int k=0;k<3;k++) g_wctx[k]=cca(g_hdc,g_ctx,ca); }   /* shared contexts for the shader compile threads */
  wglMakeCurrent(NULL,NULL); wglDeleteContext(tmp);
  if(!g_ctx){ fatal("Your graphics card does not support OpenGL 4.3.\nPlease update your GPU driver."); return 0; }
  wglMakeCurrent(g_hdc,g_ctx);
  SWI si=(SWI)wglGetProcAddress("wglSwapIntervalEXT"); extern int g_opt_vsync; if(si) si(g_opt_vsync?1:0);
  return 1; }

void* backend_gl_proc(const char* n){
  void* p=(void*)wglGetProcAddress(n);
  if(!p || p==(void*)1 || p==(void*)2 || p==(void*)3 || p==(void*)-1){ static HMODULE gl; if(!gl) gl=LoadLibraryA("opengl32.dll"); p=(void*)GetProcAddress(gl,n); }
  return p; }

void backend_gl_swap(void){ SwapBuffers(g_hdc); }

/* each compile thread takes the next shared context */
int backend_gl_worker_current(void){ int k=__atomic_fetch_add(&g_wnext,1,__ATOMIC_ACQ_REL); return k<3 && g_wctx[k] && wglMakeCurrent(g_hdc,g_wctx[k]); }

void backend_window_size(int* w, int* h){ *w=g_ww; *h=g_wh; }

void video_present_rgba(const u32* px, int w, int h, int stride){ extern void gpu_gl_present_cpu(const u32*,int,int,int); gpu_gl_present_cpu(px,w,h,stride); }

static i32 axis(u64 b, int neg, int pos){ return ((b>>pos)&1? 32767:0) - ((b>>neg)&1? 32767:0); }

static i32 dz(i32 v){ return (v>-7000 && v<7000)? 0 : v; }

void video_poll(void){
  int focused=GetForegroundWindow()==g_hwnd && !g_controls_open;
  u64 k=focused?keys_held:0, b=k&0xFFFFull;
  i32 lx=axis(k,40,42), ly=axis(k,43,41), rx=axis(k,44,46), ry=axis(k,47,45);
  if(g_mouse_mode){
    u64 now=plat_ns(); u64 elapsed=now-g_mouse_time;
    if(!focused){ g_mouse_rx=g_mouse_ry=0; g_mouse_time=now; }
    else if(elapsed>=8000000ull){
      LONG dx=InterlockedExchange(&g_mouse_dx,0),dy=InterlockedExchange(&g_mouse_dy,0);
      double sensitivity=5.0*pow(250.0,(g_mouse_sensitivity-10)/490.0);   /* 10% = 5x, 100% ~ 14x, 500% = 1250x */
      double scale=32767.0*500000.0*sensitivity/(double)elapsed;
      double x=dx*scale,y=-dy*scale;
      g_mouse_rx=(i32)(x>32767?32767:x< -32767?-32767:x);
      g_mouse_ry=(i32)(y>32767?32767:y< -32767?-32767:y); g_mouse_time=now;
    }
    rx=(g_mouse_invert&1)?-g_mouse_rx:g_mouse_rx; ry=(g_mouse_invert&2)?-g_mouse_ry:g_mouse_ry;
  }
  if(xget && focused){
    for(DWORD i=0;i<4;i++){ XST s; if(xget(i,&s)) continue; XGP* g=&s.Gamepad; WORD x=g->wButtons;
      /* map by position like a Switch pad: A right (xbox B), B bottom (xbox A), X top, Y left */
      if(x&0x2000) b|=1<<0; if(x&0x1000) b|=1<<1; if(x&0x8000) b|=1<<2; if(x&0x4000) b|=1<<3;
      if(x&0x0040) b|=1<<4; if(x&0x0080) b|=1<<5; if(x&0x0100) b|=1<<6; if(x&0x0200) b|=1<<7;
      if(g->bLeftTrigger>60) b|=1<<8; if(g->bRightTrigger>60) b|=1<<9;
      if(x&0x0010) b|=1<<10; if(x&0x0020) b|=1<<11;
      if(x&0x0004) b|=1<<12; if(x&0x0001) b|=1<<13; if(x&0x0008) b|=1<<14; if(x&0x0002) b|=1<<15;
      if(!lx&&!ly){ lx=dz(g->sThumbLX); ly=dz(g->sThumbLY); } if(!rx&&!ry){ rx=dz(g->sThumbRX); ry=dz(g->sThumbRY); }
      break; }
  }
  /* stick-as-button bits 16..23 */
  if(lx<-16000) b|=1ull<<16; if(ly>16000) b|=1ull<<17; if(lx>16000) b|=1ull<<18; if(ly<-16000) b|=1ull<<19;
  if(rx<-16000) b|=1ull<<20; if(ry>16000) b|=1ull<<21; if(rx>16000) b|=1ull<<22; if(ry<-16000) b|=1ull<<23;
  if(g_script[0]){ /* test input like the X11 backend: "ms:hex,ms:hex,..." holds those buttons for 300 ms */
    u64 el=(plat_ns()-g_start)/1000000ull; const char* p=g_script;
    while(*p){ u64 t=strtoull(p,(char**)&p,10); if(*p!=':') break; p++; u64 m=strtoull(p,(char**)&p,16); if(el>=t && el<t+300) b|=m; if(*p==',') p++; } }
  g_pad.buttons=b; g_pad.lx=lx; g_pad.ly=ly; g_pad.rx=rx; g_pad.ry=ry;
  { extern void input_rec_poll(PadState*); input_rec_poll(&g_pad); }   /* SWR_INPUT_REC / SWR_INPUT_PLAY */
  /* timed F5 / F3: polled here because input keeps being read while the game shows nothing new. Each time fires
     once, as the key would (F5 runs in the window thread, so it gets the key) */
  if(g_trace_at[0] || g_shots_at[0]){
    static int fired_tr, fired_sh;   /* bit n: the nth time already went off */
    u64 el=(plat_ns()-g_start)/1000000ull;
    const char* p=g_trace_at; for(int n=0;*p && n<32;n++){ u64 t=strtoull(p,(char**)&p,10); if(el>=t && !(fired_tr>>n&1)){ fired_tr|=1<<n; PostMessageA(g_hwnd,WM_KEYDOWN,VK_F5,0); } if(*p!=',') break; p++; }
    p=g_shots_at; for(int n=0;*p && n<32;n++){ u64 t=strtoull(p,(char**)&p,10); if(el>=t && !(fired_sh>>n&1)){ fired_sh|=1<<n; g_shot_req=1; } if(*p!=',') break; p++; }
  }
}

void frontend_frame(void){
  extern u64 gpu_gl_screenshot(const char* path);
  if(g_shot_req){ g_shot_req=0; char p[MAX_PATH]; SYSTEMTIME t; GetLocalTime(&t); CreateDirectoryA("captures",NULL);
    snprintf(p,sizeof p,"captures/screenshot_%04d%02d%02d_%02d%02d%02d",t.wYear,t.wMonth,t.wDay,t.wHour,t.wMinute,t.wSecond); gpu_gl_screenshot(p); }
  if(g_seconds && plat_ns()-g_start>=(u64)g_seconds*1000000000ull){
    u64 h=gpu_gl_screenshot(g_shot[0]? g_shot : NULL); fprintf(stderr,"\nframes=%d hash=%016llx\n",g_frames,(unsigned long long)h);
    extern void save_hints(void); save_hints(); { extern void gpu_gl_rec_finish(void); gpu_gl_rec_finish(); } fflush(NULL); ExitProcess(0); }
}

int backend_glq_ok(void){ return 1; }   /* GL calls go through glq */
