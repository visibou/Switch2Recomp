/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* F7 recorder. H.264/mp4 through Media Foundation (uses QuickSync/NVENC/VCE when it's there). frames come in as
   NV12 already converted on the GPU, we just copy them into an MF buffer and a separate thread does the encoding
   so the game thread never waits on it */
#define COBJMACROS
#include <initguid.h>
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <codecapi.h>
#include <stdio.h>
#include "rt.h"
#include "rec_flow.h"

/* big queue, never drop frames. dropped frames leave holes that some players won't play (laggy parts push several
   repeated frames at once) */
#define RQ 64
DEFINE_GUID(SWR_MF_TRANSCODE_CONTAINERTYPE, 0x150ff23f, 0x4abc, 0x478b, 0xac, 0x4f, 0xe1, 0x91, 0x6f, 0xba, 0x1c, 0xca);
DEFINE_GUID(SWR_MFTranscodeContainerType_FMPEG4, 0x9ba876f1, 0x419f, 0x4b77, 0xa1, 0xe0, 0x35, 0x95, 0x9d, 0x9d, 0x40, 0x04);
static IMFSample* g_q[RQ]; static volatile LONG g_qh, g_qt;
static HANDLE g_th, g_ev, g_ready_ev, g_space_ev; static volatile int g_run, g_ok;
static int g_w, g_h, g_fps; static u32 g_bitrate; static wchar_t g_path[MAX_PATH];
static u64 g_frames_in, g_frames_drop;
static volatile LONG g_failed;
static volatile LONG64 g_progress_ms;

static int q_count(void){ return (int)(g_qh-g_qt); }

/* the sound comes from the mixer before waveOut. never record what the PC is playing, smooth recording slows the
   game's clock on purpose */
#include "rec_audio_time.h"
#define AQ 4096
#define AF 240
typedef struct { u64 frame; unsigned count; i16 pcm[AF*2]; } AudioPacket;
static AudioPacket g_aq[AQ]; static u32 g_ah,g_at;
static SRWLOCK g_alock=SRWLOCK_INIT;
static int g_aactive; static u64 g_aepoch,g_anext,g_audio_drop,g_asession; static float g_again;
int rec_failed(void){ return InterlockedCompareExchange(&g_failed,0,0)!=0; }
static void rec_progress(void){ InterlockedExchange64(&g_progress_ms,(LONG64)GetTickCount64()); }
static void rec_fail(const char* reason,HRESULT hr){
  if(InterlockedCompareExchange(&g_failed,1,0)) return;
  WARN("recorder: stopping capture: %s (%08lx); releasing game producers\n",reason,(unsigned long)hr);
  g_ok=0; g_run=0;
  AcquireSRWLockExclusive(&g_alock); g_aactive=0; ReleaseSRWLockExclusive(&g_alock);
  SetEvent(g_ev); SetEvent(g_space_ev);
}
static int rec_wait_stalled(u64 entered){ return rec_flow_stalled(GetTickCount64(),entered,(u64)InterlockedCompareExchange64(&g_progress_ms,0,0)); }
void rec_audio_start(u64 guest_epoch){
  AcquireSRWLockExclusive(&g_alock); g_asession++; g_aepoch=guest_epoch; g_anext=0; g_again=1; g_aactive=g_ok && g_run; ReleaseSRWLockExclusive(&g_alock);
}
void rec_audio(const float* stereo,unsigned frames,u64 guest_time){
  if(!frames || frames>AF) return;
  AcquireSRWLockExclusive(&g_alock);
  if(!g_aactive || guest_time<g_aepoch){ ReleaseSRWLockExclusive(&g_alock); return; }
  u64 session=g_asession;
  /* stamp the quantum with when it was rendered, not when the IPC delivered it. catch up quanta are back to back
     and use the same guest clock start as the video */
  u64 frame=rec_audio_index(guest_time-g_aepoch,frames,g_anext);
  g_anext=frame+frames;
  u64 entered=GetTickCount64();
  /* keep the packet while the encoder is behind. dropping a quantum here left holes for good even though live
     playback had its own copy */
  while(g_ah-g_at>=AQ && g_aactive && g_run && session==g_asession){
    ReleaseSRWLockExclusive(&g_alock);
    if(rec_wait_stalled(entered)){ rec_fail("encoder stalled with a full audio queue",E_FAIL); return; }
    WaitForSingleObject(g_space_ev,10); AcquireSRWLockExclusive(&g_alock);
  }
  if(!g_aactive || !g_run || session!=g_asession){ ReleaseSRWLockExclusive(&g_alock); return; }
  AudioPacket* p=&g_aq[g_ah%AQ]; p->frame=frame; p->count=frames;
  float target=rec_audio_gain(stereo,frames,g_again); g_again=target;
  for(unsigned i=0;i<frames*2;i++) p->pcm[i]=rec_audio_pcm(stereo[i]*target);
  g_ah++; SetEvent(g_ev); ReleaseSRWLockExclusive(&g_alock);
}
static int audio_peek(u64* frame){ int ok; AcquireSRWLockShared(&g_alock); ok=g_ah!=g_at; if(ok)*frame=g_aq[g_at%AQ].frame; ReleaseSRWLockShared(&g_alock); return ok; }
static int audio_pop(AudioPacket* p){ int ok; AcquireSRWLockExclusive(&g_alock); ok=g_ah!=g_at; if(ok){ *p=g_aq[g_at%AQ]; g_at++; }ReleaseSRWLockExclusive(&g_alock); if(ok) SetEvent(g_space_ev); return ok; }
void rec_watchdog(void){
  if(!g_th || !g_ok || !g_run) return;
  u64 frame; int pending=q_count()>0 || audio_peek(&frame);
  if(pending && rec_wait_stalled(0)) rec_fail("encoder made no progress for 5 seconds",E_FAIL);
}
/* the sound goes to a .wav next to the .mp4. with AAC inside the mp4, media foundation kept buffering one stream
   while it waited for the other until it ran out of memory (8007000e) and the encoder stopped */
static FILE* g_wav; static u64 g_wav_frames;
static void wav_header(FILE* f, u64 frames){
  u32 data=(u32)(frames*4), riff=36+data; u16 fmt=1, ch=2, align=4, bits=16; u32 rate=48000, bps=192000, fsz=16;
  fseek(f,0,SEEK_SET); fwrite("RIFF",1,4,f); fwrite(&riff,4,1,f); fwrite("WAVEfmt ",1,8,f); fwrite(&fsz,4,1,f);
  fwrite(&fmt,2,1,f); fwrite(&ch,2,1,f); fwrite(&rate,4,1,f); fwrite(&bps,4,1,f); fwrite(&align,2,1,f); fwrite(&bits,2,1,f);
  fwrite("data",1,4,f); fwrite(&data,4,1,f); fseek(f,0,SEEK_END); }
static void wav_open(void){
  wchar_t p[MAX_PATH]; wcsncpy(p,g_path,MAX_PATH-1); p[MAX_PATH-1]=0;
  wchar_t* dot=wcsrchr(p,L'.'); if(dot && dot-p<MAX_PATH-5) wcscpy(dot,L".wav"); else return;
  g_wav=_wfopen(p,L"wb"); g_wav_frames=0; if(g_wav) wav_header(g_wav,0); }
static void wav_silence(u64 end){
  static const i16 zero[AF*2]={0};
  while(g_wav && g_wav_frames<end){ u64 n=end-g_wav_frames; if(n>AF) n=AF; fwrite(zero,4,(size_t)n,g_wav); g_wav_frames+=n; } }
static void wav_close(void){ if(!g_wav) return; wav_header(g_wav,g_wav_frames); fclose(g_wav); g_wav=NULL; }

static DWORD WINAPI enc_thread(LPVOID arg){
  (void)arg;
  CoInitializeEx(NULL,COINIT_MULTITHREADED);
  IMFSinkWriter* wr=NULL; IMFMediaType* out=NULL; IMFMediaType* in=NULL; IMFAttributes* at=NULL; DWORD si=0; HRESULT hr; i64 video_end=0;
  hr=MFStartup(MF_VERSION,MFSTARTUP_LITE);
  if(SUCCEEDED(hr)) hr=MFCreateAttributes(&at,3);
  if(SUCCEEDED(hr)) hr=IMFAttributes_SetUINT32(at,&MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS,TRUE);
  if(SUCCEEDED(hr)) hr=IMFAttributes_SetUINT32(at,&MF_SINK_WRITER_DISABLE_THROTTLING,FALSE);
  /* fragmented mp4 (win10+) so the file is still valid if we crash mid-recording */
  if(SUCCEEDED(hr) && !getenv("SWR_REC_MP4_CLASICO")){ IMFAttributes_SetGUID(at,&SWR_MF_TRANSCODE_CONTAINERTYPE,&SWR_MFTranscodeContainerType_FMPEG4);
    hr=MFCreateSinkWriterFromURL(g_path,NULL,at,&wr);
    if(FAILED(hr)){ IMFAttributes_DeleteItem(at,&SWR_MF_TRANSCODE_CONTAINERTYPE); hr=S_OK; wr=NULL; } else LOG("recorder: fragmented mp4 (not corrupted if the game closes)\n"); }
  if(SUCCEEDED(hr) && !wr) hr=MFCreateSinkWriterFromURL(g_path,NULL,at,&wr);
  if(SUCCEEDED(hr)) hr=MFCreateMediaType(&out);
  if(SUCCEEDED(hr)){
    IMFMediaType_SetGUID(out,&MF_MT_MAJOR_TYPE,&MFMediaType_Video);
    IMFMediaType_SetGUID(out,&MF_MT_SUBTYPE,&MFVideoFormat_H264);
    IMFMediaType_SetUINT32(out,&MF_MT_AVG_BITRATE,g_bitrate);
    IMFMediaType_SetUINT32(out,&MF_MT_INTERLACE_MODE,MFVideoInterlace_Progressive);
    IMFMediaType_SetUINT32(out,&MF_MT_MPEG2_PROFILE,eAVEncH264VProfile_High);
    IMFMediaType_SetUINT32(out,&MF_MT_YUV_MATRIX,MFVideoTransferMatrix_BT709);
    IMFMediaType_SetUINT32(out,&MF_MT_VIDEO_PRIMARIES,MFVideoPrimaries_BT709);
    IMFMediaType_SetUINT32(out,&MF_MT_TRANSFER_FUNCTION,MFVideoTransFunc_709);
    IMFMediaType_SetUINT32(out,&MF_MT_VIDEO_NOMINAL_RANGE,MFNominalRange_16_235);
    IMFMediaType_SetUINT64(out,&MF_MT_FRAME_SIZE,((UINT64)g_w<<32)|(UINT32)g_h);
    IMFMediaType_SetUINT64(out,&MF_MT_FRAME_RATE,((UINT64)g_fps<<32)|1);
    IMFMediaType_SetUINT64(out,&MF_MT_PIXEL_ASPECT_RATIO,((UINT64)1<<32)|1);
    hr=IMFSinkWriter_AddStream(wr,out,&si); }
  if(SUCCEEDED(hr)) hr=MFCreateMediaType(&in);
  if(SUCCEEDED(hr)){
    IMFMediaType_SetGUID(in,&MF_MT_MAJOR_TYPE,&MFMediaType_Video);
    IMFMediaType_SetGUID(in,&MF_MT_SUBTYPE,&MFVideoFormat_NV12);
    IMFMediaType_SetUINT32(in,&MF_MT_INTERLACE_MODE,MFVideoInterlace_Progressive);
    IMFMediaType_SetUINT32(in,&MF_MT_YUV_MATRIX,MFVideoTransferMatrix_BT709);
    IMFMediaType_SetUINT32(in,&MF_MT_VIDEO_NOMINAL_RANGE,MFNominalRange_16_235);
    IMFMediaType_SetUINT32(in,&MF_MT_DEFAULT_STRIDE,(UINT32)g_w);
    IMFMediaType_SetUINT64(in,&MF_MT_FRAME_SIZE,((UINT64)g_w<<32)|(UINT32)g_h);
    IMFMediaType_SetUINT64(in,&MF_MT_FRAME_RATE,((UINT64)g_fps<<32)|1);
    IMFMediaType_SetUINT64(in,&MF_MT_PIXEL_ASPECT_RATIO,((UINT64)1<<32)|1);
    hr=IMFSinkWriter_SetInputMediaType(wr,si,in,NULL); }
  if(SUCCEEDED(hr)) hr=IMFSinkWriter_BeginWriting(wr);
  if(in) IMFMediaType_Release(in); if(out) IMFMediaType_Release(out); if(at) IMFAttributes_Release(at);
  if(FAILED(hr)){ WARN("recorder: Windows could not open the H.264 encoder (error %08lx)\n",(unsigned long)hr);
    if(wr) IMFSinkWriter_Release(wr); MFShutdown(); CoUninitialize(); g_ok=0; SetEvent(g_ready_ev); return 0; }
  if(!g_run || rec_failed()){ IMFSinkWriter_Release(wr); MFShutdown(); CoUninitialize(); SetEvent(g_ready_ev); return 0; }
  g_ok=1; SetEvent(g_ready_ev);
  rec_progress();
  wav_open();
  LOG("recorder: sound goes to a .wav next to the video (48 kHz stereo, same clock as the video)\n");
  for(;;){
    if(rec_failed()) break;
    AudioPacket p;
    while(audio_pop(&p)){
      wav_silence(p.frame);
      unsigned skip= g_wav_frames>p.frame? (unsigned)(g_wav_frames-p.frame) : 0;
      if(g_wav && skip<p.count){ fwrite(p.pcm+skip*2,4,p.count-skip,g_wav); g_wav_frames=p.frame+p.count; }
      rec_progress(); }
    int wrote=0;
    while(q_count()>0 && !rec_failed()){ IMFSample* sample=g_q[g_qt%RQ]; InterlockedIncrement(&g_qt); SetEvent(g_space_ev);
      i64 t=0,d=0; IMFSample_GetSampleTime(sample,&t); IMFSample_GetSampleDuration(sample,&d); if(t+d>video_end) video_end=t+d;
      hr=IMFSinkWriter_WriteSample(wr,si,sample); IMFSample_Release(sample);
      if(FAILED(hr)){ rec_fail("video encoding failed",hr); break; }
      rec_progress(); wrote=1; }
    if(wrote) continue;
    if(!g_run) break;
    WaitForSingleObject(g_ev,100); }
  if(video_end>0) wav_silence((u64)video_end*48000/10000000);
  wav_close();
  /* when recording fails, let go of the queued samples instead of feeding the failing encoder again and again. a
     normal close drains them above */
  while(q_count()>0){ IMFSample* sample=g_q[g_qt%RQ]; InterlockedIncrement(&g_qt); IMFSample_Release(sample); }
  AcquireSRWLockExclusive(&g_alock); g_at=g_ah; ReleaseSRWLockExclusive(&g_alock); SetEvent(g_space_ev);
  hr=IMFSinkWriter_Finalize(wr);
  if(FAILED(hr)) WARN("recorder: could not finalize the .mp4 (error %08lx)\n",(unsigned long)hr);
  IMFSinkWriter_Release(wr); MFShutdown(); CoUninitialize();
  return 0; }

static FILE* g_raw;   /* SWR_REC_RAW=1 dumps raw NV12, handy to test capture without MF */

int rec_open(const wchar_t* path, int w, int h, int fps){
  if(g_th && WaitForSingleObject(g_th,0)==WAIT_OBJECT_0){ CloseHandle(g_th); g_th=NULL; CloseHandle(g_ev); CloseHandle(g_ready_ev); CloseHandle(g_space_ev); }
  if(g_th || g_raw){ WARN("recorder: previous encoder is still closing; capture remains disabled until it responds\n"); return 0; }
  g_w=w; g_h=h; g_fps=fps; wcsncpy(g_path,path,MAX_PATH-1);
  if(getenv("SWR_REC_RAW")){ wchar_t p2[MAX_PATH]; _snwprintf(p2,MAX_PATH,L"%ls.nv12",path); g_raw=_wfopen(p2,L"wb"); g_frames_in=0; return g_raw!=NULL; }
  /* ~0.12 bpp, 1080p60 ends up around 15 Mb/s */
  double br=(double)w*h*fps*0.12; if(br<8e6) br=8e6; if(br>40e6) br=40e6; g_bitrate=(u32)br;
  AcquireSRWLockExclusive(&g_alock); g_ah=g_at=0; g_aactive=0; g_audio_drop=0; ReleaseSRWLockExclusive(&g_alock);
  g_qh=g_qt=0; g_frames_in=g_frames_drop=0; g_run=1; g_ok=0;
  InterlockedExchange(&g_failed,0); rec_progress();
  g_ev=CreateEventA(NULL,FALSE,FALSE,NULL); g_ready_ev=CreateEventA(NULL,TRUE,FALSE,NULL); g_space_ev=CreateEventA(NULL,FALSE,FALSE,NULL);
  g_th=CreateThread(NULL,0,enc_thread,NULL,0,NULL);
  if(!g_th){ CloseHandle(g_ev); CloseHandle(g_ready_ev); CloseHandle(g_space_ev); return 0; }
  WaitForSingleObject(g_ready_ev,10000);
  if(!g_ok){ g_run=0;
    if(WaitForSingleObject(g_th,5000)!=WAIT_OBJECT_0){ rec_fail("encoder startup timed out",E_FAIL); return 0; }
    CloseHandle(g_th); g_th=NULL; CloseHandle(g_ev); CloseHandle(g_ready_ev); CloseHandle(g_space_ev); DeleteFileW(g_path); return 0; }
  LOG("recorder: recording %dx%d at %d FPS, %.0f Mb/s\n",w,h,fps,g_bitrate/1e6);
  return 1; }

/* NV12 frame: Y plane then interleaved UV at half res. t is in 100ns units */
void rec_frame(const u8* nv12, i64 t100, i64 dur100){
  if(g_raw){ fwrite(nv12,1,(size_t)g_w*g_h*3/2,g_raw); g_frames_in++; return; }
  if(!g_th || !g_ok) return;
  g_frames_in++;
  /* the encoder can't keep up so wait. smooth mode already slows the game down so this barely happens */
  {u64 entered=GetTickCount64(); while(q_count()>=RQ && g_run){
    if(rec_wait_stalled(entered)){ rec_fail("encoder stalled with a full video queue",E_FAIL); return; }
    WaitForSingleObject(g_space_ev,50);
  }}
  if(!g_run) return;
  DWORD n=(DWORD)((u64)g_w*g_h*3/2); IMFMediaBuffer* b=NULL; IMFSample* s=NULL; BYTE* p=NULL;
  if(FAILED(MFCreateMemoryBuffer(n,&b))) return;
  if(SUCCEEDED(IMFMediaBuffer_Lock(b,&p,NULL,NULL))){ memcpy(p,nv12,n); IMFMediaBuffer_Unlock(b); }
  IMFMediaBuffer_SetCurrentLength(b,n);
  if(FAILED(MFCreateSample(&s))){ IMFMediaBuffer_Release(b); return; }
  IMFSample_AddBuffer(s,b); IMFMediaBuffer_Release(b);
  IMFSample_SetSampleTime(s,t100); IMFSample_SetSampleDuration(s,dur100);
  g_q[g_qh%RQ]=s; InterlockedIncrement(&g_qh); SetEvent(g_ev); }

void rec_close(void){
  if(g_raw){ fclose(g_raw); g_raw=NULL; LOG("recorder: %llu raw frames\n",(unsigned long long)g_frames_in); return; }
  if(!g_th) return;
  AcquireSRWLockExclusive(&g_alock); g_aactive=0; ReleaseSRWLockExclusive(&g_alock);
  g_run=0; SetEvent(g_ev); SetEvent(g_space_ev);
  if(WaitForSingleObject(g_th,rec_failed()?1000:5000)!=WAIT_OBJECT_0){
    rec_fail("encoder did not finish; capture disabled until it responds",E_FAIL);
    return; /* the thread still owns its handles and MF objects, don't free them */
  }
  CloseHandle(g_th); g_th=NULL; CloseHandle(g_ev); CloseHandle(g_ready_ev); CloseHandle(g_space_ev);
  LOG("recorder: %llu audio packets lost (queue limit)\n",(unsigned long long)g_audio_drop);
  LOG("recorder: saved %ls (%llu frames, %llu dropped)\n",g_path,(unsigned long long)g_frames_in,(unsigned long long)g_frames_drop); }
