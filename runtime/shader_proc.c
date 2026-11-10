/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* shader helper process. a second copy of the exe (started with --shader-helper) opens its own OpenGL context,
   compiles the shaders and sends back program binaries that the game loads with glProgramBinary. intel's windows
   driver crashes on some of SMO's vertex shaders, and jumping out of the crash isn't enough: the driver keeps a lock
   it took inside that call and a few seconds later the whole game freezes on it. when the helper crashes only the
   helper dies, we start a new one and the game never notices */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int shp_compile(int slot, int sep, unsigned stype, const char* vs, const char* fs, unsigned* fmt, void** bin, unsigned* len, char* log, int logn);
int shp_init(const char* id);
int shp_main(void);

#if defined(_WIN32)
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601
#endif
#include <windows.h>
#include <wchar.h>

typedef unsigned int u32;
typedef unsigned char u8;
#define SHP_MAGIC 0x48525753u   /* "SWRH" */
#define SHP_SLOTS 4

static int rd_all(HANDLE h, void* p, u32 n){ u8* b=p; while(n){ DWORD got=0; if(!ReadFile(h,b,n,&got,NULL) || !got) return 0; b+=got; n-=got; } return 1; }
static int wr_all(HANDLE h, const void* p, u32 n){ const u8* b=p; while(n){ DWORD put=0; if(!WriteFile(h,b,n,&put,NULL) || !put) return 0; b+=put; n-=put; } return 1; }

/* ---------------------------------------------------------------- the helper itself */
typedef unsigned (WINAPI *F_CS)(unsigned);
typedef void (WINAPI *F_SS)(unsigned,int,const char* const*,const int*);
typedef void (WINAPI *F_U)(unsigned);
typedef unsigned (WINAPI *F_CP)(void);
typedef void (WINAPI *F_UU)(unsigned,unsigned);
typedef void (WINAPI *F_UEI)(unsigned,unsigned,int);
typedef void (WINAPI *F_GIV)(unsigned,unsigned,int*);
typedef void (WINAPI *F_LOG)(unsigned,int,int*,char*);
typedef void (WINAPI *F_GPB)(unsigned,int,int*,unsigned*,void*);
typedef const u8* (WINAPI *F_GS)(unsigned);
typedef void (WINAPI *F_GI)(unsigned,int*);
typedef HGLRC (WINAPI *F_CCA)(HDC,HGLRC,const int*);

/* a crash in the driver ends the helper right away, no error box. the game sees the pipe close */
static LONG WINAPI shp_die(EXCEPTION_POINTERS* e){ (void)e; TerminateProcess(GetCurrentProcess(),0xC0DE); return EXCEPTION_EXECUTE_HANDLER; }

static void* glp(const char* n){ void* p=(void*)wglGetProcAddress(n);
  if(!p || p==(void*)1 || p==(void*)2 || p==(void*)3 || p==(void*)-1){ static HMODULE gl; if(!gl) gl=LoadLibraryA("opengl32.dll"); p=(void*)GetProcAddress(gl,n); }
  return p; }

static void shp_id(char* o, int n, F_GS gs){ const char* v=(const char*)gs(0x1F00); const char* r=(const char*)gs(0x1F01); const char* x=(const char*)gs(0x1F02);
  snprintf(o,n,"%s|%s|%s",v?v:"",r?r:"",x?x:""); }

int shp_main(void){
  SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX);
  SetUnhandledExceptionFilter(shp_die);
  HANDLE in=GetStdHandle(STD_INPUT_HANDLE), out=GetStdHandle(STD_OUTPUT_HANDLE);
  HINSTANCE hi=GetModuleHandle(NULL);
  WNDCLASSA wc={0}; wc.style=CS_OWNDC; wc.lpfnWndProc=DefWindowProcA; wc.hInstance=hi; wc.lpszClassName="SwitchRecompShaders";
  RegisterClassA(&wc);
  HWND w=CreateWindowA("SwitchRecompShaders","",WS_OVERLAPPEDWINDOW,0,0,64,64,NULL,NULL,hi,NULL);   /* never shown */
  if(!w) return 3;
  HDC dc=GetDC(w);
  PIXELFORMATDESCRIPTOR pfd={sizeof pfd,1,PFD_DRAW_TO_WINDOW|PFD_SUPPORT_OPENGL|PFD_DOUBLEBUFFER,PFD_TYPE_RGBA,32};
  pfd.cDepthBits=24; pfd.cStencilBits=8; pfd.iLayerType=PFD_MAIN_PLANE;
  if(!SetPixelFormat(dc,ChoosePixelFormat(dc,&pfd),&pfd)) return 3;
  HGLRC tmp=wglCreateContext(dc); if(!tmp) return 3;
  wglMakeCurrent(dc,tmp);
  /* same versions and profile as the game's context, a binary from another kind of context may not load */
  F_CCA cca=(F_CCA)wglGetProcAddress("wglCreateContextAttribsARB"); HGLRC ctx=NULL;
  static const int vers[4][2]={{4,6},{4,5},{4,4},{4,3}};
  for(int i=0;i<4 && !ctx && cca;i++){ int ca[]={0x2091,vers[i][0],0x2092,vers[i][1],0x9126,0x00000001,0}; ctx=cca(dc,NULL,ca); }
  wglMakeCurrent(NULL,NULL); wglDeleteContext(tmp);
  if(!ctx || !wglMakeCurrent(dc,ctx)) return 3;
  F_CS createShader=(F_CS)glp("glCreateShader"); F_SS shaderSource=(F_SS)glp("glShaderSource"); F_U compileShader=(F_U)glp("glCompileShader");
  F_CP createProgram=(F_CP)glp("glCreateProgram"); F_UU attachShader=(F_UU)glp("glAttachShader"); F_UEI programParameteri=(F_UEI)glp("glProgramParameteri");
  F_U linkProgram=(F_U)glp("glLinkProgram"); F_GIV getProgramiv=(F_GIV)glp("glGetProgramiv"); F_GIV getShaderiv=(F_GIV)glp("glGetShaderiv");
  F_LOG programLog=(F_LOG)glp("glGetProgramInfoLog"); F_LOG shaderLog=(F_LOG)glp("glGetShaderInfoLog"); F_GPB getProgramBinary=(F_GPB)glp("glGetProgramBinary");
  F_U deleteShader=(F_U)glp("glDeleteShader"); F_U deleteProgram=(F_U)glp("glDeleteProgram"); F_GS getString=(F_GS)glp("glGetString"); F_GI getIntegerv=(F_GI)glp("glGetIntegerv");
  if(!createShader || !shaderSource || !compileShader || !createProgram || !attachShader || !programParameteri || !linkProgram || !getProgramiv || !getShaderiv
     || !programLog || !shaderLog || !getProgramBinary || !deleteShader || !deleteProgram || !getString || !getIntegerv) return 3;
  /* hello: which driver this is, the game only uses us if it's the same one it has */
  char id[512]; shp_id(id,sizeof id,getString); int nfmt=0; getIntegerv(0x87FE,&nfmt);
  u32 hd[3]={SHP_MAGIC,(u32)nfmt,(u32)strlen(id)}; if(!wr_all(out,hd,12) || !wr_all(out,id,hd[2])) return 0;
  for(;;){
    u32 rq[5]; if(!rd_all(in,rq,20) || rq[0]!=SHP_MAGIC || rq[3]>(16u<<20) || rq[4]>(16u<<20)) return 0;   /* the game closed the pipe */
    char* s1=malloc(rq[3]+1); char* s2= rq[4]? malloc(rq[4]+1) : NULL; if(!s1 || (rq[4] && !s2)) return 0;
    if(!rd_all(in,s1,rq[3]) || (s2 && !rd_all(in,s2,rq[4]))) return 0;
    s1[rq[3]]=0; if(s2) s2[rq[4]]=0;
    u32 sep=rq[1], stype=rq[2];
    { static const char* ct; static int ctr; if(!ctr){ ctr=1; ct=getenv("SWR_SHP_CRASH_TEST"); }   /* testing: 1 crash on precise, 2 on everything */
      if(ct && (atoi(ct)==2 || (atoi(ct)==1 && strstr(s1,"precise ")))) *(volatile int*)(size_t)rq[0]=0; }
    unsigned a=createShader(sep? stype : 0x8B31), b= s2? createShader(0x8B30) : 0;
    const char* p1=s1; shaderSource(a,1,&p1,NULL); compileShader(a);
    if(b){ const char* p2=s2; shaderSource(b,1,&p2,NULL); compileShader(b); }
    unsigned p=createProgram(); attachShader(p,a); if(b) attachShader(p,b);
    if(sep) programParameteri(p,0x8258,1);
    programParameteri(p,0x8257,1);   /* binary retrievable */
    linkProgram(p); int ok=0; getProgramiv(p,0x8B82,&ok);
    u32 rp[3]={0,0,0}; void* data=NULL;
    if(ok){ int len=0; getProgramiv(p,0x8741,&len);
      if(len>0 && (data=malloc(len))){ int got=0; unsigned fm=0; getProgramBinary(p,len,&got,&fm,data); if(got>0){ rp[0]=1; rp[1]=fm; rp[2]=(u32)got; } } }
    if(!rp[0]){ char* e=malloc(4096); e[0]=0; int c=1;
      getShaderiv(a,0x8B81,&c);
      if(!c){ snprintf(e,64,"%s: ",sep? (stype==0x8B31? "VS" : "FS") : "VS"); shaderLog(a,4000,NULL,e+strlen(e)); }
      else if(b && (getShaderiv(b,0x8B81,&c),!c)){ strcpy(e,"FS: "); shaderLog(b,4000,NULL,e+4); }
      else if(!ok) programLog(p,4000,NULL,e);
      else strcpy(e,"the driver gave no program binary");
      free(data); data=e; rp[2]=(u32)strlen(e); }
    deleteShader(a); if(b) deleteShader(b); deleteProgram(p);
    free(s1); free(s2);
    if(!wr_all(out,rp,12) || (rp[2] && !wr_all(out,data,rp[2]))) return 0;
    free(data); } }

/* ---------------------------------------------------------------- game side, one helper per compile thread */
typedef struct { HANDLE proc, wr, rd; } ShpSlot;
static ShpSlot g_slot[SHP_SLOTS];
static char g_id[512];
static HANDLE g_jobobj;
static volatile int g_spawned;

static void shp_kill(ShpSlot* s){
  if(s->proc){ TerminateProcess(s->proc,1); WaitForSingleObject(s->proc,2000); CloseHandle(s->proc); }
  if(s->wr) CloseHandle(s->wr);
  if(s->rd) CloseHandle(s->rd);
  s->proc=s->wr=s->rd=NULL; }

/* 1 when n bytes came, 0 when the helper died or took longer than ms */
static int shp_read(ShpSlot* s, void* p, u32 n, DWORD ms){
  u8* b=p; DWORD t0=GetTickCount(); int spin=0;
  while(n){
    DWORD av=0; if(!PeekNamedPipe(s->rd,NULL,0,NULL,&av,NULL)) return 0;   /* broken pipe: it's gone */
    if(av){ DWORD got=0; if(!ReadFile(s->rd,b,av<n? av : n,&got,NULL) || !got) return 0; b+=got; n-=got; spin=0; continue; }
    if(GetTickCount()-t0>ms) return 0;
    if(++spin<200) SwitchToThread(); else Sleep(1); }
  return 1; }

static int shp_spawn(ShpSlot* s){
  SECURITY_ATTRIBUTES sa={sizeof sa,NULL,TRUE};
  HANDLE in_r=NULL, in_w=NULL, out_r=NULL, out_w=NULL;
  if(!CreatePipe(&in_r,&in_w,&sa,1u<<20)) return -2;
  if(!CreatePipe(&out_r,&out_w,&sa,1u<<20)){ CloseHandle(in_r); CloseHandle(in_w); return -2; }
  SetHandleInformation(in_w,HANDLE_FLAG_INHERIT,0); SetHandleInformation(out_r,HANDLE_FLAG_INHERIT,0);
  static wchar_t exe[1024]; if(!exe[0] && !GetModuleFileNameW(NULL,exe,1024)) exe[0]=0;
  wchar_t cmd[1100]; _snwprintf(cmd,1100,L"\"%ls\" --shader-helper",exe); cmd[1099]=0;
  STARTUPINFOEXW si; memset(&si,0,sizeof si); si.StartupInfo.cb=sizeof si; si.StartupInfo.dwFlags=STARTF_USESTDHANDLES;
  si.StartupInfo.hStdInput=in_r; si.StartupInfo.hStdOutput=out_w; si.StartupInfo.hStdError=NULL;
  /* only the two pipe ends go to the helper, not every handle the game has open */
  SIZE_T asz=0; InitializeProcThreadAttributeList(NULL,1,0,&asz); si.lpAttributeList=asz? malloc(asz) : NULL;
  HANDLE inh[2]={in_r,out_w};
  int attr= si.lpAttributeList && InitializeProcThreadAttributeList(si.lpAttributeList,1,0,&asz);
  if(attr && !UpdateProcThreadAttribute(si.lpAttributeList,0,PROC_THREAD_ATTRIBUTE_HANDLE_LIST,inh,sizeof inh,NULL,NULL)){ DeleteProcThreadAttributeList(si.lpAttributeList); attr=0; }
  if(!attr) si.StartupInfo.cb=sizeof si.StartupInfo;
  PROCESS_INFORMATION pi; memset(&pi,0,sizeof pi);
  BOOL ok= exe[0] && CreateProcessW(exe,cmd,NULL,NULL,TRUE,CREATE_NO_WINDOW|(attr? EXTENDED_STARTUPINFO_PRESENT : 0),NULL,NULL,(STARTUPINFOW*)&si,&pi);
  if(attr) DeleteProcThreadAttributeList(si.lpAttributeList);
  free(si.lpAttributeList);
  CloseHandle(in_r); CloseHandle(out_w);
  if(!ok){ CloseHandle(in_w); CloseHandle(out_r); return -2; }
  if(g_jobobj) AssignProcessToJobObject(g_jobobj,pi.hProcess);   /* dies with the game */
  CloseHandle(pi.hThread);
  s->proc=pi.hProcess; s->wr=in_w; s->rd=out_r;
  u32 hd[3]; char id[512];
  if(!shp_read(s,hd,12,20000) || hd[0]!=SHP_MAGIC || hd[2]>=sizeof id || !shp_read(s,id,hd[2],5000)){ shp_kill(s); return -2; }
  id[hd[2]]=0;
  if(!hd[1] || strcmp(id,g_id)){ shp_kill(s); return -3; }   /* no binaries, or another GPU (laptops with two) */
  __atomic_add_fetch(&g_spawned,1,__ATOMIC_RELAXED);
  return 0; }

/* 1 ok (bin/fmt/len set, free bin), 0 the shader doesn't compile (log set), -1 the helper crashed or hung on it,
   -2 no helper available */
int shp_compile(int slot, int sep, unsigned stype, const char* vs, const char* fs, unsigned* fmt, void** bin, unsigned* len, char* log, int logn){
  if(slot<0 || slot>=SHP_SLOTS) return -2;
  ShpSlot* s=&g_slot[slot]; *bin=NULL;
  if(!s->proc){ int r=shp_spawn(s); if(r<0) return -2; }
  u32 l1=(u32)strlen(vs), l2= fs? (u32)strlen(fs) : 0;
  u32 rq[5]={SHP_MAGIC,(u32)sep,stype,l1,l2};
  if(!wr_all(s->wr,rq,20) || !wr_all(s->wr,vs,l1) || (l2 && !wr_all(s->wr,fs,l2))){ shp_kill(s); return -1; }
  u32 rp[3];
  if(!shp_read(s,rp,12,20000) || rp[2]>(64u<<20)){ shp_kill(s); return -1; }   /* SMO's biggest take ~1 s on intel, 20 s is a hang */
  u8* d=malloc(rp[2]+1); if(!d){ shp_kill(s); return -1; }
  if(rp[2] && !shp_read(s,d,rp[2],10000)){ free(d); shp_kill(s); return -1; }
  if(rp[0]==1){ *fmt=rp[1]; *len=rp[2]; *bin=d; return 1; }
  d[rp[2]]=0; if(logn>0) snprintf(log,logn,"%s",(char*)d); free(d); return 0; }

/* starts the first helper to see if it works here. id is "vendor|renderer|version" of the game's context */
int shp_init(const char* id){
  snprintf(g_id,sizeof g_id,"%s",id);
  if(!g_jobobj){ g_jobobj=CreateJobObjectW(NULL,NULL);
    if(g_jobobj){ JOBOBJECT_EXTENDED_LIMIT_INFORMATION li; memset(&li,0,sizeof li); li.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
      SetInformationJobObject(g_jobobj,JobObjectExtendedLimitInformation,&li,sizeof li); } }
  return shp_spawn(&g_slot[0]); }

#else
int shp_compile(int slot, int sep, unsigned stype, const char* vs, const char* fs, unsigned* fmt, void** bin, unsigned* len, char* log, int logn){
  (void)slot; (void)sep; (void)stype; (void)vs; (void)fs; (void)fmt; (void)len; (void)log; (void)logn; *bin=NULL; return -2; }
int shp_init(const char* id){ (void)id; return -2; }
int shp_main(void){ return 3; }
#endif
