#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include <stdio.h>
#include "gen.c"
#include "ref_table.c"
static u64 rnd(u64* s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }
int main(void){ u64 seed=0x1234567; static u8 buf[64] __attribute__((aligned(64))); static CPU cpu; CPU* c=&cpu;
  u8* M=buf-0x10000;   /* guest address 0x10010 is buf+16 */
  for(int i=0;i<NT;i++){ u64 x[16]; for(int r=0;r<16;r++) x[r]=rnd(&seed)%7==0? rnd(&seed)&0xFF : rnd(&seed); for(int b=0;b<64;b++) buf[b]=(u8)rnd(&seed);
    if(T[i].kind && (i&1)){ int n= T[i].kind==1? 8 : 16; memcpy(buf+16,&x[T[i].rs],8); if(n==16) memcpy(buf+24,&x[T[i].rs+1],8);
      if(T[i].kind==2){ unsigned a32=(unsigned)x[T[i].rs], b32=(unsigned)x[T[i].rs+1]; memcpy(buf+16,&a32,4); memcpy(buf+20,&b32,4); } }   /* half the CAS cases match */
    memset(c,0,sizeof *c); for(int r=0;r<16;r++) c->x[r]=x[r]; if(T[i].rn>=0) c->x[T[i].rn]=0x10010;
    GF[i](c,M); if(T[i].rn>=0) c->x[T[i].rn]=0;
    printf("%d %08x",i,T[i].w); for(int r=0;r<16;r++) printf(" %llx",(unsigned long long)c->x[r]); printf(" |"); for(int b=0;b<64;b++) printf("%02x",buf[b]); printf("\n"); }
  return 0; }
