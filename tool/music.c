/* SwitchRecomp by MZ Labs, Copyright 2026 MZ Labs. AGPL-3.0, see LICENSE.txt and NOTICE.txt. the decoders in audio_decode/ keep their own licenses */
#include <SDL.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "music.h"
#define DR_WAV_IMPLEMENTATION
#include "audio_decode/dr_wav.h"
#define DR_MP3_IMPLEMENTATION
#include "audio_decode/dr_mp3.h"
#include "audio_decode/stb_vorbis.c"
static SDL_AudioDeviceID device;
static float* samples;
static size_t count,cursor;
static float gain=.25f;
static int playing;
static char loaded[4096],music_err[256];
static void callback(void* user,Uint8* stream,int len){
    (void)user; float* dst=(float*)stream; int n=len/(int)sizeof(float);
    if(!playing || !count){ memset(stream,0,len); return; }
    for(int i=0;i<n;i++){ float x=samples[cursor++]*gain; if(cursor==count) cursor=0; dst[i]=isfinite(x)?fmaxf(-1,fminf(1,x)):0; }
}
const char* music_error(void){ return music_err; }
int music_load(const char* path){
    if(!path || !path[0]) return 0;
    if(samples && !strcmp(path,loaded)) return 1;
    unsigned channels=0,rate=0; drwav_uint64 frames=0; float* decoded=NULL;
    const char* ext=strrchr(path,'.');
    if(ext && !SDL_strcasecmp(ext,".wav")) decoded=drwav_open_file_and_read_pcm_frames_f32(path,&channels,&rate,&frames,NULL);
    else if(ext && !SDL_strcasecmp(ext,".mp3")){ drmp3_config cfg={0}; drmp3_uint64 f=0; decoded=drmp3_open_file_and_read_pcm_frames_f32(path,&cfg,&f,NULL); channels=cfg.channels; rate=cfg.sampleRate; frames=f; }
    else if(ext && !SDL_strcasecmp(ext,".ogg")){
        short* pcm=NULL; int ch=0,sr=0,f=stb_vorbis_decode_filename(path,&ch,&sr,&pcm);
        if(f>0 && ch>0 && ch<=8){ frames=f; channels=ch; rate=sr; decoded=malloc((size_t)f*ch*sizeof(float)); if(decoded) for(size_t i=0;i<(size_t)f*ch;i++) decoded[i]=pcm[i]/32768.f; }
        free(pcm);
    }
    if(!decoded || !frames || !channels || channels>8 || !rate || frames*channels>0x1fffffff){ free(decoded); SDL_strlcpy(music_err,"Invalid or unsupported WAV, OGG or MP3.",sizeof music_err); return 0; }
    SDL_AudioStream* convert=SDL_NewAudioStream(AUDIO_F32SYS,(Uint8)channels,(int)rate,AUDIO_F32SYS,2,48000);
    int bytes=(int)(frames*channels*sizeof(float));
    if(!convert || SDL_AudioStreamPut(convert,decoded,bytes)<0 || SDL_AudioStreamFlush(convert)<0){ free(decoded); if(convert) SDL_FreeAudioStream(convert); SDL_strlcpy(music_err,SDL_GetError(),sizeof music_err); return 0; }
    free(decoded); int size=SDL_AudioStreamAvailable(convert); float* pcm=size>0?malloc((size_t)size):NULL;
    if(!pcm || SDL_AudioStreamGet(convert,pcm,size)!=size){ free(pcm); SDL_FreeAudioStream(convert); SDL_strlcpy(music_err,"Could not convert the music file.",sizeof music_err); return 0; }
    SDL_FreeAudioStream(convert);
    if(!device){ SDL_AudioSpec want={0}; want.freq=48000; want.format=AUDIO_F32SYS; want.channels=2; want.samples=1024; want.callback=callback;
        device=SDL_OpenAudioDevice(NULL,0,&want,NULL,0); if(!device){ free(pcm); SDL_strlcpy(music_err,SDL_GetError(),sizeof music_err); return 0; }}
    SDL_LockAudioDevice(device); free(samples); samples=pcm; count=(size_t)size/sizeof(float); cursor=0; SDL_strlcpy(loaded,path,sizeof loaded); SDL_UnlockAudioDevice(device);
    SDL_PauseAudioDevice(device,0); music_err[0]=0; return 1;
}
void music_config(const char* path,int enabled,float volume){
    if(enabled && path && path[0] && !music_load(path)) enabled=0;
    if(device) SDL_LockAudioDevice(device);
    playing=enabled; gain=isfinite(volume)?fmaxf(0,fminf(1,volume)):0;
    if(device) SDL_UnlockAudioDevice(device);
}
void music_close(void){ if(device) SDL_CloseAudioDevice(device); device=0; free(samples); samples=NULL; count=cursor=0; loaded[0]=0; }
