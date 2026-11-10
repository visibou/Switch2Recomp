/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* controller recording and replay, to drive the same race again in unattended runs.
   SWR_INPUT_REC=1 (or a folder): records into recordings/rec_<date>_<time>/ next to the exe:
     input.txt  every change of the controller state (after keyboard, XInput and mouse are merged) and the first read
                of each RomFS file, stamped with game time (vblanks the game advanced: the sum of its present
                intervals), the present count and ms since start
     save/      a copy of the save folder as it was at start (replays start from it, see SWR_SAVE_DIR in svc_fs.c)
   SWR_INPUT_PLAY=<folder>: plays <folder>/input.txt back; the real controller is ignored.
   Loading takes a different time every run, so a press is not keyed to the time since start: it is keyed to the last
   "sync file" read before it plus the game time since. A replay waits for that file's first read, then applies the
   press after the same game time. Sync files: for each comma separated SWR_INPUT_SYNC part, the first file whose path
   contains it. The default is what starts at fixed points before the race: the title menu, the cup's opening movie, the
   course intro, the race intro flyover and the course music at the countdown (later music, laps, items and the goal
   depend on how the race goes). A sync file that never comes is given up 2 minutes past its recorded time (logged).
   A menu press can land in a screen transition and be lost, and the replay then waits on that menu for good (about 1 run
   in 10): see replay_tap. */
#include "rt.h"
#include "trace.h"
#include <time.h>

#define SYNC_DEFAULT "BGM_Seq_Title,/Movie/,BGM_Seq_CourseTelop,BGM_Seq_RaceIntro,BGM_Stage_"

static int g_mode;            /* 0 off, 1 record, 2 replay */
static PMutex g_lk;
static volatile u64 g_vb;     /* game time, vblanks */
static volatile u32 g_presents;
static FILE* g_rf;

typedef struct { u64 vb, ms, d; int anchor; PadState s; } RecPad;
typedef struct { char path[256]; u64 vb, ms, vb_play; int seen; } RecSync;
static RecPad* g_pads; static int g_npad, g_cur;
static RecSync g_syncs[64]; static int g_nsync;
static PadState g_last;

static u64 now_ms(void){ return swr_run_ns()/1000000ull; }

void fs_copy_tree(const char* src, const char* dst);

static void rec_start(const char* v){
  char dir[1200];
  if(!strcmp(v,"1")){ time_t t=time(NULL); struct tm* m=localtime(&t); char rd[1100];
    snprintf(rd,sizeof rd,"%srecordings",g_exedir); plat_mkdir(rd);
    snprintf(dir,sizeof dir,"%s/rec_%04d%02d%02d_%02d%02d%02d",rd,m->tm_year+1900,m->tm_mon+1,m->tm_mday,m->tm_hour,m->tm_min,m->tm_sec); }
  else snprintf(dir,sizeof dir,"%s",v);
  plat_mkdir(dir);
  char p[1300]; snprintf(p,sizeof p,"%s/input.txt",dir);
  g_rf=fopen(p,"w"); if(!g_rf){ WARN("input recording: cannot write %s\n",p); return; }
  extern int g_frame_pacing;
  fprintf(g_rf,"swr-input 1\ninfo frame_pacing %d\n",g_frame_pacing); fflush(g_rf);
  char s[1200], d[1300]; snprintf(s,sizeof s,"%ssave",g_datadir); snprintf(d,sizeof d,"%s/save",dir);
  if(plat_is_dir(s)) fs_copy_tree(s,d);
  g_mode=1; LOG("input recording: %s\n",dir); }

static void play_start(const char* dir){
  char p[1300]; snprintf(p,sizeof p,"%s/input.txt",dir);
  FILE* f=fopen(p,"r"); if(!f){ WARN("input replay: cannot read %s\n",p); return; }
  const char* e=getenv("SWR_INPUT_SYNC"); if(!e) e=SYNC_DEFAULT;
  char part[16][128]; int nparts=0, used[16]={0};
  for(const char* q=e;*q && nparts<16;){ const char* c=strchr(q,','); int n= c? (int)(c-q) : (int)strlen(q); if(n>127) n=127;
    if(n){ memcpy(part[nparts],q,n); part[nparts][n]=0; nparts++; } q+= c? n+1 : n; }
  int cap=1024; g_pads=malloc(sizeof(RecPad)*cap);
  char ln[600]; int nfiles=0;
  while(fgets(ln,sizeof ln,f)){
    unsigned long long vb, pr, ms, b; int lx,ly,rx,ry; char path[300];
    if(sscanf(ln,"pad %llu %llu %llu %llx %d %d %d %d",&vb,&pr,&ms,&b,&lx,&ly,&rx,&ry)==8){
      if(g_npad==cap){ cap*=2; g_pads=realloc(g_pads,sizeof(RecPad)*cap); }
      RecPad* r=&g_pads[g_npad++]; memset(r,0,sizeof *r);
      r->vb=vb; r->ms=ms; r->anchor=g_nsync-1; r->d= g_nsync? vb-g_syncs[g_nsync-1].vb : vb;
      r->s.buttons=b; r->s.lx=lx; r->s.ly=ly; r->s.rx=rx; r->s.ry=ry; }
    else if(sscanf(ln,"file %llu %llu %llu %299[^\n]",&vb,&pr,&ms,path)==4){
      nfiles++;
      /* a sync file: the first file whose path contains one of the parts (only the first: BGM_Stage_ also matches the
         final lap's faster version, which comes at a different time once a replayed race goes differently) */
      int hit=0; for(int k=0;k<nparts;k++) if(!used[k] && strstr(path,part[k])){ used[k]=1; hit=1; }
      if(hit && g_nsync<64){ RecSync* s=&g_syncs[g_nsync++]; snprintf(s->path,sizeof s->path,"%s",path); s->vb=vb; s->ms=ms; } } }
  fclose(f);
  g_mode=2;
  LOG("input replay: %s, %d controller changes, %d sync files (of %d first reads, SWR_INPUT_SYNC=%s)\n",dir,g_npad,g_nsync,nfiles,e);
  for(int i=0;i<g_nsync;i++) LOG("input replay: sync %d: %s (recorded at %.1f s)\n",i,g_syncs[i].path,g_syncs[i].ms/1000.0); }

void input_rec_init(void){
  pm_init(&g_lk);
  const char* r=getenv("SWR_INPUT_REC"); const char* p=getenv("SWR_INPUT_PLAY");
  if(p && *p) play_start(p); else if(r && *r) rec_start(r); }

/* replay: applies every change due by game time vb to g_last. Buttons pressed by a change that a later one in the same
   step replaces are held in g_extra until the next step (hid_update then keeps a press for 2 presents), so a tap shorter
   than a frame is not lost. Returns whether anything changed. Called under g_lk */
static u64 g_extra; static u64 g_last_step_ms;
/* SWR_INPUT_PLAY_UNTIL=<s>: the replay hands the controller back at that time since start (the keyboard, XInput and
   the runtime's `script` option drive from then on: a replayed race can be finished with the game's driving assists) */
static int replay_handed(void){
  static u64 until=~0ull; static int said;
  if(until==~0ull){ const char* e=getenv("SWR_INPUT_PLAY_UNTIL"); until= e? (u64)(atof(e)*1000.0) : 0; }
  if(!until || now_ms()<until) return 0;
  if(!said){ said=1; LOG("input replay: controller handed back at %.1f s (SWR_INPUT_PLAY_UNTIL)\n",now_ms()/1000.0); }
  return 1; }
/* The menu presses are all done and what follows them (the first sync after the title: the cup movie, or the race
   intro of a time trial) is 8 s later than it was in the recording: tap A until it comes. Not before: A would skip a
   movie that is playing. Returns the buttons to add */
static u64 replay_tap(void){
  static u64 ready; static int said;
  if(g_nsync<2 || !g_syncs[0].seen || g_syncs[1].seen || !g_cur || g_cur>=g_npad || g_pads[g_cur].anchor<1) return 0;
  u64 now=now_ms(); if(!ready) ready=now;
  u64 last=g_pads[g_cur-1].ms, gap= g_syncs[1].ms>last? g_syncs[1].ms-last : 0;
  if(now<ready+gap+8000) return 0;
  if(!said){ said=1; LOG("input replay: tapping A at %.1f s, %s did not come after the menu presses\n",now/1000.0,g_syncs[1].path); }
  return (now-ready)%1000<200? 1 : 0; }

static int replay_step(u64 vb){
  u64 extra=0; int applied=0;
  g_last_step_ms=now_ms();
  while(g_cur<g_npad){ RecPad* r=&g_pads[g_cur]; u64 base=0;
    if(r->anchor>=0){ RecSync* s=&g_syncs[r->anchor];
      if(!s->seen){ static u64 wait; if(!wait){ const char* e=getenv("SWR_INPUT_SYNC_WAIT_S"); wait=(e? (u64)atoi(e) : 120)*1000; } if(now_ms()<s->ms+wait) break;
        s->seen=1; s->vb_play=vb; WARN("input replay: sync %d (%s) never came, going on from now\n",r->anchor,s->path); }
      base=s->vb_play; }
    if(vb<base+r->d) break;
    if(applied) extra|=g_last.buttons;
    g_last=r->s; g_cur++; applied++;
    if(g_cur==g_npad) LOG("input replay: last change applied at %.1f s; the controller stays neutral from now\n",now_ms()/1000.0); }
  if(g_cur==g_npad && !applied && (g_last.buttons|g_last.lx|g_last.ly|g_last.rx|g_last.ry)){ memset(&g_last,0,sizeof g_last); applied=1; }
  extra|=replay_tap();
  int changed= applied || g_extra!=extra;
  g_extra=extra;
  return changed; }

/* each present: the game's present interval = the vblanks its clock advances this frame. A replay applies its changes
   here, at the frame boundary, and writes them to the shared memory at once, so the game reads each change in the same
   frame (by game time) every run: applied by the 60 Hz main loop instead, a change landed a frame early or late
   depending on timing, and the replayed race drifted further from the recording (stuck in 24th, never finishing) */
void input_rec_present(u32 interval){
  u64 vb=__atomic_add_fetch(&g_vb,interval,__ATOMIC_RELAXED); __atomic_add_fetch(&g_presents,1,__ATOMIC_RELAXED);
  if(g_mode!=2 || replay_handed()) return;
  pm_lock(&g_lk); int changed=replay_step(vb);
  if(changed){ extern PadState g_pad; g_pad.buttons=g_last.buttons|g_extra; g_pad.lx=g_last.lx; g_pad.ly=g_last.ly; g_pad.rx=g_last.rx; g_pad.ry=g_last.ry; g_pad.touch=0; }
  pm_unlock(&g_lk);
  if(changed){ extern void hid_update(void); hid_update(); } }

/* first read of a RomFS file (path inside the RomFS) */
void input_rec_file(const char* path){
  if(!g_mode) return;
  u64 vb=__atomic_load_n(&g_vb,__ATOMIC_RELAXED);
  pm_lock(&g_lk);
  if(g_mode==1 && g_rf){ fprintf(g_rf,"file %llu %u %llu %s\n",(unsigned long long)vb,g_presents,(unsigned long long)now_ms(),path); fflush(g_rf); }
  else if(g_mode==2) for(int i=0;i<g_nsync;i++) if(!g_syncs[i].seen && !strcmp(g_syncs[i].path,path)){
    g_syncs[i].seen=1; g_syncs[i].vb_play=vb;
    LOG("input replay: sync %d (%s) at %.1f s (recorded at %.1f s)\n",i,path,now_ms()/1000.0,g_syncs[i].ms/1000.0); break; }
  pm_unlock(&g_lk); }

/* each poll of the controller (60 Hz, main loop): record the state, or replace it with the recorded one */
void input_rec_poll(PadState* pad){
  if(!g_mode || (g_mode==2 && replay_handed())) return;
  u64 vb=__atomic_load_n(&g_vb,__ATOMIC_RELAXED);
  pm_lock(&g_lk);
  if(g_mode==1){
    if(g_rf && (pad->buttons!=g_last.buttons || pad->lx!=g_last.lx || pad->ly!=g_last.ly || pad->rx!=g_last.rx || pad->ry!=g_last.ry)){
      g_last=*pad;
      fprintf(g_rf,"pad %llu %u %llu %llx %d %d %d %d\n",(unsigned long long)vb,g_presents,(unsigned long long)now_ms(),
              (unsigned long long)pad->buttons,pad->lx,pad->ly,pad->rx,pad->ry); fflush(g_rf); } }
  else {
    /* the state the last present applied; while no present comes (loading), steps here so sync timeouts still run */
    if(now_ms()-g_last_step_ms>200) replay_step(vb);
    pad->buttons=g_last.buttons|g_extra; pad->lx=g_last.lx; pad->ly=g_last.ly; pad->rx=g_last.rx; pad->ry=g_last.ry; pad->touch=0; }
  pm_unlock(&g_lk); }
