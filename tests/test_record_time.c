#include "rec_audio_time.h"
#include <assert.h>
#include <stdio.h>
int main(void){
 uint64_t next=0;
 for(unsigned k=1;k<=200;k++){uint64_t now=(uint64_t)k*5000000;uint64_t f=rec_audio_index(now,240,next);assert(f==next);next=f+240;}
 assert(next==48000);assert(rec_audio_time(next)==10000000);
 /* 4 seconds of real work at a .25 guest-clock rate produce 1s of audio/video. */
 uint64_t wall=4000000000ull,guest=wall/4;assert(guest==1000000000ull);
 assert(rec_audio_index(guest,240,47760)==47760);assert(60*10000000ull/60==(uint64_t)rec_audio_time(48000));
 for(unsigned k=1;k<=200;k++){uint64_t jitter=(uint64_t)k*5000000+1000000;assert(rec_audio_index(jitter,240,(k-1)*240)==(k-1)*240);}
 assert(rec_audio_index(1000000000,240,0)==47760); /* a gap must stay a gap */
 assert(rec_audio_pcm(NAN)==0&&rec_audio_pcm(INFINITY)==0);
 assert(rec_audio_pcm(2)==32767&&rec_audio_pcm(-2)==-32767);
 /* An overloaded mix retains its sine wave rather than clipping to a square. */
 float pcm[480];for(unsigned i=0;i<240;i++)pcm[i*2]=pcm[i*2+1]=4.f*sinf((float)i*6.283185307f/48.f);
 float g=rec_audio_gain(pcm,240,1);assert(g>.23f&&g<.24f);
 for(unsigned i=0;i<480;i++)assert(fabsf(pcm[i]*g)<=.95001f);
 assert(rec_audio_gain(pcm,240,g)==g);
 for(unsigned i=0;i<480;i++)pcm[i]=.1f;
 float release=rec_audio_gain(pcm,240,g);assert(release>g&&release<g+.05f);
 pcm[0]=NAN;pcm[1]=INFINITY;assert(isfinite(rec_audio_gain(pcm,240,g)));
 for(uint64_t n=0;n<100000;n+=240)assert(rec_audio_time(n+240)>rec_audio_time(n));
 puts("PASS: 48kHz timestamps, 60fps/slow guest-clock alignment, jitter snapping, preserved gaps and PCM clipping");
}
