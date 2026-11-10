/* SwitchRecomp - Copyright 2026 MZ Labs
   Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
   Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen. */
/* Switch 2 (Ampere, SM 8.x) SASS -> Maxwell (GM20B) SASS, so shader.c can turn it into GLSL like any Switch 1
   shader. every 128 bit Ampere instruction becomes one or more 64 bit Maxwell ones; branch targets are fixed up
   at the end. Ampere has no Maxwell form for some things, those go through scratch registers:
     UR0..UR62 (uniform registers)  -> R176..R238
     texture sources/results        -> R239..R250
     immediates that don't fit       -> R253, R254
   a shader that already uses R176+ can't be lowered this way and is reported.
   Ampere field positions come from Mesa's NAK encoder (src/nouveau/compiler/nak/sm70_encode.rs, MIT license,
   Copyright Collabora Ltd. and Red Hat Inc.); the Maxwell side is what shader.c decodes */
#include "sm86.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UBASE  176u
#define TA     239u    /* tex source list A (4) */
#define TB     243u    /* tex source list B (4) */
#define TO     247u    /* tex result (4) */
#define SA     251u    /* operand fix-ups (abs/neg Maxwell can't take there) */
#define SB     252u
#define S1     253u
#define S2     254u
#define RZ     255u

typedef struct { u64 lo, hi; } W128;
static int g_tex_bank=-1;
static int* g_ret; static u32 g_nret;   /* RET -> instruction after its one CALL */
unsigned short g_sm86_untr[256]; int g_sm86_nuntr;   /* every opcode left untranslated so far (the runtime logs them) */
static u32 F(const W128* w, int lo, int n){
  u64 v; if(lo>=64) v=w->hi>>(lo-64); else if(lo+n<=64) v=w->lo>>lo; else v=(w->lo>>lo)|(w->hi<<(64-lo));
  return (u32)(v & ((n>=32)? 0xFFFFFFFFull : ((1ull<<n)-1))); }

/* ---- output */
typedef struct { u64* m; u32 n, cap; u32* amap; u32 namp; struct { u32 at, tgt, kind; } *fx; u32 nfx, cfx;
  int bad; u32 first_bad; u32 bad_op; u64 guard; int maxreg; } Lw;

static void put(Lw* L, u64 v){ if(L->n==L->cap){ L->cap=L->cap? L->cap*2 : 256; L->m=realloc(L->m,L->cap*8); } L->m[L->n++]=v|L->guard; }

/* opcode bits 63..48 from shader.c's pattern ('-' = 0) */
static u64 P(const char* s){ u64 v=0; for(int i=0;i<16 && s[i];i++) if(s[i]=='1') v|=1ull<<(63-i); return v; }
#define FLD(v,lo) ((u64)(v)<<(lo))

static const W128* g_cur;
/* uniform predicates UP0..UP6 live in Maxwell predicates the shader doesn't use (picked per shader in sm86_lower),
   8 = no free one left */
static int g_upmap[8];
static int g_upspill;   /* bit k: UPk lives in register 175-k */
static u32 spr(u32 k){ return 175u-k; }
static u32 upm(u32 x){ x&=7; return x==7? 7u : (u32)g_upmap[x]; }   /* the instruction being lowered, for SWR_SM86_TRACE */
static void untr(Lw* L, u32 i, u32 op){ if(!L->bad){ L->first_bad=i; L->bad_op=op; } L->bad++;
  { static int on=-1; static unsigned short cnt[4096]; if(on<0) on=getenv("SWR_SM86_TRACE")!=NULL;
    if(on && g_cur && cnt[op&0xFFF]<6){ cnt[op&0xFFF]++; fprintf(stderr,"sm86 untr %03x #%u lo %016llx hi %016llx\n",op,i,(unsigned long long)g_cur->lo,(unsigned long long)g_cur->hi); } }
  if(op){ int k=0; for(;k<g_sm86_nuntr;k++) if(g_sm86_untr[k]==op) break; if(k==g_sm86_nuntr && g_sm86_nuntr<256) g_sm86_untr[g_sm86_nuntr++]=(unsigned short)op; (void)i; }
  /* an opcode shader.c doesn't know: reported as "unknown" and counted as unsupported */
  put(L,0xFFFF000000000000ull|((u64)op<<32)|i); }

/* ---- registers */
static u32 gr(Lw* L, u32 r){ if(r!=RZ && (int)r>L->maxreg) L->maxreg=(int)r; return r; }
static u32 ur(u32 u){ u&=63; return u==63? RZ : UBASE+u; }

/* ---- operands */
enum { K_R, K_I, K_C };
typedef struct { int k; u32 v; u32 idx, off; int neg, abs; } Op;

/* slot at bit 32 for the given form (1 reg, 4 imm, 5 cbuf, 6 ureg; 2/3/7 = imm/cbuf/ureg too) */
static Op slot32(Lw* L, const W128* w, u32 form){
  Op o={0}; o.abs=F(w,62,1); o.neg=F(w,63,1);
  switch(form){
  case 1: o.k=K_R; o.v=gr(L,F(w,32,8)); break;
  case 2: case 4: o.k=K_I; o.v=F(w,32,32); o.abs=o.neg=0; break;
  case 3: case 5: o.k=K_C; o.off=F(w,38,16); o.idx=F(w,54,5);
    if(o.idx>=18){ static int said; if(said<24){ said++; extern void logf_(int,const char*,...);   /* Switch 2 shaders that end up reading an undeclared cb24 */
        logf_(1,"sm86: opcode %03x form %u reads c[%u][%x]: %016llx %016llx\n",F(w,0,12),form,o.idx,o.off,(unsigned long long)w->hi,(unsigned long long)w->lo); } }
    break;
  default: o.k=K_R; o.v=ur(F(w,32,6)); break; }
  return o; }
static Op slot64(Lw* L, const W128* w){ Op o={0}; o.k=K_R; o.v=gr(L,F(w,64,8)); o.abs=F(w,74,1); o.neg=F(w,75,1); return o; }
static Op src0(Lw* L, const W128* w, int uni){ Op o={0}; o.k=K_R; o.v= uni? ur(F(w,24,6)) : gr(L,F(w,24,8)); o.abs=F(w,73,1); o.neg=F(w,72,1); return o; }

/* B and C of a 3 source ALU op */
static void bc(Lw* L, const W128* w, u32 form, int uni, Op* b, Op* c){
  if(form==2||form==3||form==7){ *b=slot64(L,w); *c=slot32(L,w,form); }
  else { *b=slot32(L,w,form); *c=slot64(L,w); }
  if(uni){ if(b->k==K_R && form==1) b->v=ur(F(w,32,6)); if(c->k==K_R && form!=7) c->v=ur(F(w,64,6)); } }

static void mov32i(Lw* L, u32 d, u32 v){ put(L,P("000000010000----")|FLD(0xF,12)|FLD(v,20)|d); }
static void movr(Lw* L, u32 d, u32 s){ put(L,P("0101110010011---")|FLD(0xF,39)|FLD(s,20)|d); }
static void movc(Lw* L, u32 d, u32 idx, u32 off){ put(L,P("0100110010011---")|FLD(0xF,39)|FLD(idx,34)|FLD(off>>2,20)|d); }
/* an operand into a register (scratch if needed) */
static u32 toreg(Lw* L, Op* o, u32 scratch){
  if(o->k==K_R) return o->v;
  if(o->k==K_I) mov32i(L,scratch,o->v); else movc(L,scratch,o->idx,o->off);
  o->k=K_R; o->v=scratch; return scratch; }
static int fits20i(u32 v){ int32_t s=(int32_t)v; return s>=-(1<<19) && s<(1<<19); }
static int fits20f(u32 v){ return (v&0xFFF)==0; }
/* a float operand with |x| or -x applied, for the places the Maxwell form has no modifier bits: FADD into a scratch */
static void fmod_fix(Lw* L, Op* o, u32 scratch){
  if(!o->abs && !o->neg) return;
  if(o->k==K_I){ if(o->abs) o->v&=0x7FFFFFFFu; if(o->neg) o->v^=0x80000000u; o->abs=o->neg=0; return; }
  int ab=o->abs, ng=o->neg; o->abs=o->neg=0; u32 src=toreg(L,o,scratch);
  put(L,P("0101110001011---")|FLD(RZ,20)|FLD(src,8)|scratch|FLD(ng,48)|FLD(ab,46)); o->k=K_R; o->v=scratch; }

/* Maxwell B operand for the R/C/I forms: returns the form index (0 R, 1 C, 2 I) and the field bits */
static int mb(Lw* L, Op* o, int fl, u64* bits, u32 scratch){
  if(o->k==K_I && !(fl? fits20f(o->v) : fits20i(o->v))) toreg(L,o,scratch);
  if(o->k==K_R){ *bits=FLD(o->v,20); return 0; }
  if(o->k==K_C){ *bits=FLD(o->idx,34)|FLD((o->off>>2)&0x3FFF,20); return 1; }
  if(fl) *bits=FLD((o->v>>12)&0x7FFFF,20)|FLD(o->v>>31,56); else *bits=FLD(o->v&0x7FFFF,20)|FLD((o->v>>19)&1,56);
  return 2; }

/* ---- one instruction */
static void lower1(Lw* L, const W128* w, u32 i, u32 n);
static void lower1i(Lw* L, const W128* w, u32 i, u32 n){
  u32 op12=F(w,0,12), op9=op12&0x1FF, form=F(w,9,3);
  u32 pg=F(w,12,3), pn=F(w,15,1);
  u32 d=F(w,16,8);
  if(!w->lo && !w->hi){ put(L,P("0101000010110---")); return; }   /* zero padding (NVN2 leaves 0x30 bytes after the header) */
  int uni= ((op9&0x180)==0x080 && F(w,91,1)) || op12==0xc82 || op12==0x882;   /* uniform ALU copies (0x08x..0x0bx), UMOV */
  u32 o9= uni? (op9&0x17F) : op9;
  { int uop= uni || op12==0x886 || op12==0x89c || op12==0x883 || op12==0x9c3 || op12==0xab9 || op12==0xabb;
    if(uop && pg!=7){ pg=upm(pg); if(pg>7){ untr(L,i,op12); return; } } }
  L->guard=FLD(pg,16)|FLD(pn,19);
  if(uni || op12==0x9c3 || op12==0xab9 || op12==0xabb) d=ur(d); else gr(L,d);   /* S2UR, ULDC write uniform registers too */
  Op a=src0(L,w,uni), b, c;
  /* ------------------------------------------------ ALU (9 bit opcode + form) */
  if(op12<0x200 || (op12&0xE00)==0x200 || (op12&0xE00)==0x400 || (op12&0xE00)==0x600 || (op12&0xE00)==0x800 || (op12&0xE00)==0xA00 || (op12&0xE00)==0xC00 || (op12&0xE00)==0xE00){
    switch(o9){
    case 0x002: /* MOV */
      b=slot32(L,w,form);
      if(b.k==K_I) mov32i(L,d,b.v); else if(b.k==K_C) movc(L,d,b.idx,b.off); else movr(L,d,b.v);
      return;
    case 0x010: { /* IADD3 (.X at 74, carry out pred at 81) */
      bc(L,w,form,uni,&b,&c); int x=F(w,74,1); u32 cop=F(w,81,3);
      if(c.k==K_R && c.v==RZ && (x || cop!=7)){   /* 2 sources with carry: IADD.CC / IADD.X */
        u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"0101110000010---","0100110000010---","0011100-00010---"};
        put(L,P(pt[f])|bb|FLD(a.v,8)|d|FLD(a.neg,49)|FLD(b.neg,48)|FLD(x,43)|FLD(cop!=7,47)); return; }
      if(c.k!=K_R) toreg(L,&c,S2);
      u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"010111001100----","010011001100----","0011100-1100----"};
      put(L,P(pt[f])|bb|FLD(c.v,39)|FLD(a.v,8)|d|FLD(a.neg,51)|FLD(b.neg,50)|FLD(c.neg,49)); return; }
    case 0x012: { /* LOP3 */
      bc(L,w,form,uni,&b,&c); u32 lut=F(w,72,8), po=F(w,81,3); if(uni){ po=upm(po); if(po>7){ untr(L,i,op12); return; } }
      if(c.k!=K_R) toreg(L,&c,S2);
      if(po!=7 && b.k!=K_R) toreg(L,&b,S1);
      u64 bb; int f=mb(L,&b,0,&bb,S1);
      if(f==0) put(L,P("0101101111100---")|bb|FLD(c.v,39)|FLD(a.v,8)|d|FLD(lut,28)|FLD(po,48)|FLD(3,36));
      else put(L,P(f==1? "0000001---------" : "001111----------")|bb|FLD(c.v,39)|FLD(a.v,8)|d|FLD(lut,48));
      return; }
    case 0x019: { /* SHF */
      bc(L,w,form,uni,&b,&c); u32 ty=F(w,73,2); int right=F(w,76,1), hi=F(w,80,1), wrap=F(w,75,1);
      if(c.k!=K_R) toreg(L,&c,S2);
      if(b.k==K_C) toreg(L,&b,S1);
      if(!right && !hi){ /* lo << s */
        u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"0101110001001---","0100110001001---","0011100-01001---"};
        put(L,P(pt[f])|bb|FLD(a.v,8)|d|FLD(wrap,39)); return; }
      if(right && hi){ /* hi >> s */
        u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"0101110000101---","0100110000101---","0011100-00101---"};
        put(L,P(pt[f])|bb|FLD(c.v,8)|d|FLD(wrap,39)|FLD(ty==0||ty==2,48)); return; }
      if(b.k==K_I){ put(L,P(right? "0011100-11111---" : "0011011-11111---")|FLD(b.v&63,20)|FLD(c.v,39)|FLD(a.v,8)|d); return; }
      put(L,P(right? "0101110011111---" : "0101101111111---")|FLD(b.v,20)|FLD(c.v,39)|FLD(a.v,8)|d); return; }
    case 0x030: if(form==1 && !uni){ /* HADD2 R, R: packed half add, Maxwell HADD2 (field positions as Mesa NAK / Eden-NS2) */
      if(F(w,76,1) || F(w,79,1) || F(w,81,10)){ untr(L,i,op12); return; }
      u32 ra=gr(L,F(w,24,8)), as=F(w,74,2), aa=F(w,73,1), an=F(w,72,1);
      u32 rb=gr(L,F(w,32,8)), bs=F(w,60,2), ba=F(w,62,1), bn=F(w,63,1);
      if(F(w,78,1)){ /* .F32 out: -RZ + one half of b = that half as a float (F2F.F32.F16) */
        if(ra!=RZ || !an || aa || as || bs<2 || F(w,77,1) || F(w,80,1)){ untr(L,i,op12); return; }
        put(L,P("0101110010101---")|d|FLD(2,8)|FLD(bs==1? 2 : 1,10)|FLD(rb,20)|FLD(bs==3,41)|FLD(bn,45)|FLD(ba,49)); return; }
      put(L,P("0101110100010---")|d|FLD(ra,8)|FLD(rb,20)|FLD(bs,28)|FLD(ba,30)|FLD(bn,31)|FLD(F(w,77,1),32)|FLD(F(w,80,1),39)|
          FLD(an,43)|FLD(aa,44)|FLD(as,47)); return; }
      untr(L,i,op12); return;
    case 0x025: { /* IMAD.WIDE (UIMAD.WIDE too): d:d+1 = a * b (32x32 -> 64, signed unless .U32) + c:c+1.
                     Product halves with IMAD / IMAD.HI, then IADD.CC + IADD.X. Address math of NVN2 compute shaders */
      bc(L,w,form,uni,&b,&c); int sg=F(w,73,1);
      if(c.k!=K_R){ untr(L,i,op12); return; }
      u32 clo=c.v, chi= c.v==RZ? RZ : c.v+1, dhi= d==RZ? RZ : d+1; if(!uni && dhi!=RZ) gr(L,F(w,16,8)+1);
      u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"010110100-------","010010100-------","0011010-0-------"};
      put(L,P(pt[f])|bb|FLD(RZ,39)|FLD(a.v,8)|SB|FLD(1,54)|FLD(sg,48)|FLD(sg,53));   /* SB = hi(a*b) */
      f=mb(L,&b,0,&bb,S1); put(L,P(pt[f])|bb|FLD(RZ,39)|FLD(a.v,8)|SA);              /* SA = lo(a*b) */
      if(chi!=RZ && chi==d){ movr(L,S2,chi); chi=S2; }                                 /* d is written before c's high half is read */
      put(L,P("0101110000010---")|FLD(clo,20)|FLD(SA,8)|d|FLD(1,47));                   /* d = SA + c.lo, carry out */
      if(dhi!=RZ) put(L,P("0101110000010---")|FLD(chi,20)|FLD(SB,8)|dhi|FLD(1,43));     /* d+1 = SB + c.hi + carry */
      return; }
    case 0x024: case 0x027: { /* IMAD / IMAD.HI */
      bc(L,w,form,uni,&b,&c); int sg=F(w,73,1), hi= o9==0x027;
      if(!hi && a.v==RZ && b.k==K_R && b.v==RZ){ /* IMAD.MOV */
        if(c.k==K_I) mov32i(L,d,c.v); else if(c.k==K_C) movc(L,d,c.idx,c.off); else movr(L,d,c.v); return; }
      if(c.k==K_I || (c.k==K_C && b.k!=K_R)) toreg(L,&c,S2);
      u64 hb= hi? FLD(1,54)|FLD(sg,48)|FLD(sg,53) : 0;
      if(c.k==K_C){ put(L,P("010100100-------")|FLD(b.v,39)|FLD(c.idx,34)|FLD((c.off>>2)&0x3FFF,20)|FLD(a.v,8)|d|hb); return; }
      u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"010110100-------","010010100-------","0011010-0-------"};
      put(L,P(pt[f])|bb|FLD(c.v,39)|FLD(a.v,8)|d|hb); return; }
    case 0x00c: { /* ISETP */
      if(F(w,72,1)){ untr(L,i,op12); return; }    /* .EX (64 bit compare) */
      b=slot32(L,w,form); if(uni && form==1) b.v=ur(F(w,32,6));
      u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"010110110110----","010010110110----","0011011-0110----"};
      u32 p1=F(w,81,3), p2=F(w,84,3), pc=F(w,87,3), pcn=F(w,90,1);
      if(uni){ p1=upm(p1); p2=upm(p2); pc=upm(pc); if(p1>7||p2>7||pc>7){ untr(L,i,op12); return; } }
      put(L,P(pt[f])|bb|FLD(a.v,8)|FLD(p1,3)|FLD(p2,0)|FLD(pc,39)|FLD(pcn,42)|FLD(F(w,74,2),45)|FLD(F(w,73,1),48)|FLD(F(w,76,3),49)); return; }
    case 0x00b: { /* FSETP */
      b=slot32(L,w,form); u64 bb; int f=mb(L,&b,1,&bb,S1); static const char* pt[3]={"010110111011----","010010111011----","0011011-1011----"};
      put(L,P(pt[f])|bb|FLD(a.v,8)|FLD(F(w,81,3),3)|FLD(F(w,84,3),0)|FLD(F(w,87,3),39)|FLD(F(w,90,1),42)|FLD(F(w,74,2),45)|FLD(F(w,76,4),48)|
          FLD(a.neg,43)|FLD(a.abs,7)|FLD(b.neg,6)|FLD(b.abs,44)|FLD(F(w,80,1),47)); return; }
    case 0x01a: { /* SGXT: sign/zero extend from b bits -> BFE with position 0, length b */
      b=slot32(L,w,form); if(uni && form==1) b.v=ur(F(w,32,6)); int sg=F(w,73,1);
      if(b.k==K_I){ put(L,P("0011100-00000---")|FLD(((b.v&0xFF)<<8)&0x7FFFF,20)|FLD(a.v,8)|d|FLD(sg,48)); return; }
      u32 r=toreg(L,&b,S1); put(L,P("0011100-01001---")|FLD(8,20)|FLD(r,8)|S1);   /* S1 = b << 8 */
      put(L,P("0101110000000---")|FLD(S1,20)|FLD(a.v,8)|d|FLD(sg,48)); return; }
    case 0x03e: if(uni){ /* UBREV */
      u32 s=ur(F(w,32,6)); put(L,P("0011100-00101---")|FLD(s,8)|d|FLD(1,39)|FLD(1,40)); return; }
      { /* F2FP.PACK_AB: d = half(a) << 16 | half(b) */
      b=slot32(L,w,form); u32 rb=toreg(L,&b,S1);
      put(L,P("0101110010101---")|FLD(rb,20)|SA|FLD(1,8)|FLD(2,10));            /* SA = f16(b) */
      if(a.v==RZ){ movr(L,d,SA); return; }
      put(L,P("0101110010101---")|FLD(a.v,20)|SB|FLD(1,8)|FLD(2,10));           /* SB = f16(a) */
      put(L,P("0011100-01001---")|FLD(16,20)|FLD(SB,8)|SB);                     /* SB <<= 16 */
      put(L,P("0101101111100---")|FLD(SB,20)|FLD(RZ,39)|FLD(SA,8)|d|FLD(0xFC,28)|FLD(7,48)|FLD(3,36)); return; }
    case 0x00a: { /* FSET (.BF: 1.0 / 0.0) */
      b=slot32(L,w,form); u64 bb; int f=mb(L,&b,1,&bb,S1); static const char* pt[3]={"01011000--------","01001000--------","0011000---------"};
      put(L,P(pt[f])|bb|FLD(a.v,8)|d|FLD(F(w,87,3),39)|FLD(F(w,90,1),42)|FLD(F(w,74,2),45)|FLD(F(w,76,4),48)|FLD(1,52)|
          FLD(a.neg,43)|FLD(a.abs,54)|FLD(b.neg,53)|FLD(b.abs,44)|FLD(F(w,80,1),55)); return; }
    case 0x03d: if(uni){ /* UFLO */
      u32 s=ur(F(w,32,6)); put(L,P("0101110000110---")|FLD(s,20)|d|FLD(F(w,63,1),40)|FLD(F(w,73,1),48)|FLD(F(w,74,1),41)); return; } break;
    case 0x03f: if(uni){ /* UPOPC */
      u32 s=ur(F(w,32,6)); put(L,P("0101110000001---")|FLD(s,20)|d|FLD(F(w,63,1),40)); return; } break;
    case 0x021: { /* FADD */
      b=slot32(L,w,form); u32 sat=F(w,77,1);
      if(b.k==K_I && sat) toreg(L,&b,S1);   /* FADD32I has no .SAT */
      if(b.k==K_I){ put(L,P("000010----------")|FLD(b.v,20)|FLD(a.v,8)|d|FLD(a.neg,56)|FLD(a.abs,54)); return; }
      u64 bb; int f=mb(L,&b,1,&bb,S1); static const char* pt[3]={"0101110001011---","0100110001011---","0011100-01011---"};
      put(L,P(pt[f])|bb|FLD(a.v,8)|d|FLD(a.neg,48)|FLD(a.abs,46)|FLD(b.neg,45)|FLD(b.abs,49)|FLD(sat,50)); return; }
    case 0x020: { /* FMUL */
      b=slot32(L,w,form); u32 sat=F(w,77,1);
      if(a.abs){ fmod_fix(L,&a,SA); } if(b.abs){ fmod_fix(L,&b,SB); }
      int ng=a.neg^b.neg;
      /* scale (bits 84-86, 4 = none): 3/2/1 = /2 /4 /8, 5/6/7 = *2 *4 *8; Maxwell's field (41-43) is 1-3 /2../8, 6/5/4
         *2..*8 (mapping as in Eglantine's lowering). Dropped, ~1% of MKW's multiplies were 2-8x off */
      static const u32 smap[8]={0,3,2,1,0,6,5,4}; u32 sc=smap[F(w,84,3)&7];
      if(b.k==K_I){ u32 v=b.v^(ng? 0x80000000u : 0);
        if(sc && ((v>>23)&0xFF) && ((v>>23)&0xFF)!=0xFF){ int k=F(w,84,3)-4; i32 e=(i32)((v>>23)&0xFF)+k; if(e>0 && e<255) v=(v&~(0xFFu<<23))|((u32)e<<23); else sc=0x80; }
        if(sc!=0x80){ put(L,P("00011110--------")|FLD(v,20)|FLD(a.v,8)|d|FLD(sat,55)); return; }
        toreg(L,&b,S1); b.v=S1; sc=smap[F(w,84,3)&7]; }   /* exponent out of range: register form keeps the scale */
      u64 bb; int f=mb(L,&b,1,&bb,S1); static const char* pt[3]={"0101110001101---","0100110001101---","0011100-01101---"};
      put(L,P(pt[f])|bb|FLD(a.v,8)|d|FLD(ng,48)|FLD(sat,50)|FLD(F(w,76,1)? 2 : 0,44)|FLD(sc,41)); return; }
    case 0x023: { /* FFMA */
      bc(L,w,form,0,&b,&c); u32 sat=F(w,77,1); u64 fmz=FLD(F(w,76,1)? 2 : 0,53);
      if(a.abs) fmod_fix(L,&a,SA);
      if(b.abs) fmod_fix(L,&b,SB);
      if(c.abs) fmod_fix(L,&c,S2);
      int ng=a.neg^b.neg;
      if(c.k==K_I) toreg(L,&c,S2);
      if(c.k==K_C){ if(b.k!=K_R) toreg(L,&b,S1);
        put(L,P("010100011-------")|FLD(b.v,39)|FLD(c.idx,34)|FLD((c.off>>2)&0x3FFF,20)|FLD(a.v,8)|d|FLD(ng,48)|FLD(c.neg,49)|FLD(sat,50)|fmz); return; }
      if(b.k==K_I && !fits20f(b.v) && c.v==d && !b.neg){
        put(L,P("000011----------")|FLD(b.v,20)|FLD(a.v,8)|d|FLD(a.neg,56)|FLD(c.neg,57)|FLD(sat,55)|fmz); return; }
      u64 bb; int f=mb(L,&b,1,&bb,S1); static const char* pt[3]={"010110011-------","010010011-------","0011001-1-------"};
      put(L,P(pt[f])|bb|FLD(c.v,39)|FLD(a.v,8)|d|FLD(ng,48)|FLD(c.neg,49)|FLD(sat,50)|fmz); return; }
    case 0x009: { /* FMNMX */
      b=slot32(L,w,form); u64 bb; int f=mb(L,&b,1,&bb,S1); static const char* pt[3]={"0101110001100---","0100110001100---","0011100-01100---"};
      put(L,P(pt[f])|bb|FLD(a.v,8)|d|FLD(F(w,87,3),39)|FLD(F(w,90,1),42)|FLD(a.neg,48)|FLD(a.abs,46)|FLD(b.neg,45)|FLD(b.abs,49)); return; }
    case 0x007: case 0x008: { /* SEL, FSEL */
      b=slot32(L,w,form); if(uni && form==1) b.v=ur(F(w,32,6));
      if(o9==0x008){ fmod_fix(L,&a,SA); fmod_fix(L,&b,SB); }
      u32 sp=F(w,87,3); if(uni){ sp=upm(sp); if(sp>7){ untr(L,i,op12); return; } }
      /* FSEL is a bit select too (Maxwell has only SEL): its immediate goes in as an integer. With the float layout
         (bits 31:12) FSEL R, RZ, 2.0 selected 0x40000, a denormal: MKW's docked-mode SMAA blend drew its full-screen
         triangle at the origin and the picture came out black */
      u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"0101110010100---","0100110010100---","0011100-10100---"};
      put(L,P(pt[f])|bb|FLD(a.v,8)|d|FLD(sp,39)|FLD(F(w,90,1),42)); return; }
    case 0x017: { /* IMNMX */
      b=slot32(L,w,form); u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"0101110000100---","0100110000100---","0011100-00100---"};
      put(L,P(pt[f])|bb|FLD(a.v,8)|d|FLD(F(w,87,3),39)|FLD(F(w,90,1),42)|FLD(F(w,73,1),48)); return; }
    case 0x016: { /* PRMT (index mode) */
      if(F(w,72,3)){ untr(L,i,op12); return; }
      bc(L,w,form,uni,&b,&c); if(c.k!=K_R) toreg(L,&c,S2);
      u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"010110111100----","010010111100----","0011011-1100----"};
      put(L,P(pt[f])|bb|FLD(c.v,39)|FLD(a.v,8)|d); return; }
    case 0x011: { /* LEA / LEA.HI (the carry of .X is dropped) */
      bc(L,w,form,uni,&b,&c); u32 sh=F(w,75,5); int hi=F(w,80,1);
      if(hi){ if(c.k!=K_R) toreg(L,&c,S2); if(b.k!=K_R && b.k!=K_C) toreg(L,&b,S1);
        if(b.k==K_C) put(L,P("000110----------")|FLD(b.idx,34)|FLD((b.off>>2)&0x3FFF,20)|FLD(c.v,39)|FLD(a.v,8)|d|FLD(sh,28)|FLD(F(w,72,1),45));
        else put(L,P("0101101111011---")|FLD(b.v,20)|FLD(c.v,39)|FLD(a.v,8)|d|FLD(sh,28)|FLD(F(w,72,1),45));
        return; }
      u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"0101101111010---","010010111101----","0011011-11010---"};
      put(L,P(pt[f])|bb|FLD(a.v,8)|d|FLD(sh,39)|FLD(F(w,72,1),45)|FLD(7,48)); return; }
    case 0x013: { /* IABS -> I2I.S32.S32 |x| */
      b=slot32(L,w,form); u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"0101110011100---","0100110011100---","0011100-11100---"};
      put(L,P(pt[f])|bb|d|FLD(2,8)|FLD(2,10)|FLD(1,12)|FLD(1,13)|FLD(1,49)); return; }
    case 0x108: { /* MUFU */
      b=slot32(L,w,form); u32 fn=F(w,74,6);
      if(fn==6||fn==7||fn>8||F(w,73,1)){ untr(L,i,op12); return; }
      u32 s=toreg(L,&b,S1);
      if(fn<=1){ put(L,P("00011110--------")|FLD(0x40C90FDBu,20)|FLD(s,8)|S2); s=S2; } /* Ampere takes x/2pi, shader.c's sin/cos radians */
      put(L,P("0101000010000---")|FLD(fn,20)|FLD(s,8)|d|FLD(fn<=1? 0 : b.neg,48)|FLD(fn<=1? 0 : b.abs,46)); return; }
    case 0x106: { /* I2F. A register source has a byte/half selector in bits 60-61 (Rs.B0-B3, U16 Rs.H1), Maxwell's
                     byte selector is bits 41-42 (H1 = byte 2). Dropped, MKW's tone map indexed its LUT with (r, r, b) */
      u32 sel= (form==1 || form==6)? F(w,60,2) : 0; if(F(w,84,2)==1) sel=(sel&1)*2;
      b=slot32(L,w,form); u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"0101110010111---","0100110010111---","0011100-10111---"};
      put(L,P(pt[f])|bb|d|FLD(F(w,75,2),8)|FLD(F(w,84,2),10)|FLD(F(w,74,1),13)|FLD(F(w,78,2),39)|FLD(sel,41)); return; }
    case 0x045: { /* I2FP.F32.S32/U32 (Ampere): I2F with the sizes fixed at 32 bits, signed in bit 74, rounding 78-79.
                    MKW's last pass builds its texcoords with it (the screen came out one flat colour) */
      b=slot32(L,w,form); u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"0101110010111---","0100110010111---","0011100-10111---"};
      put(L,P(pt[f])|bb|d|FLD(2,8)|FLD(2,10)|FLD(F(w,74,1),13)|FLD(F(w,78,2),39)); return; }
    case 0x105: { /* F2I */
      b=slot32(L,w,form); u64 bb; int f=mb(L,&b,1,&bb,S1); static const char* pt[3]={"0101110010110---","0100110010110---","0011100-10110---"};
      put(L,P(pt[f])|bb|d|FLD(F(w,75,2),8)|FLD(F(w,84,2),10)|FLD(F(w,72,1),12)|FLD(F(w,78,2),39)|FLD(b.neg,49)|FLD(b.abs,45)); return; }
    case 0x104: case 0x107: { /* F2F, FRND */
      b=slot32(L,w,form); u64 bb; int f=mb(L,&b,1,&bb,S1); static const char* pt[3]={"0101110010101---","0100110010101---","0011100-10101---"};
      put(L,P(pt[f])|bb|d|FLD(F(w,75,2),8)|FLD(F(w,84,2),10)|FLD(F(w,78,2),39)|FLD(o9==0x107,42)|FLD(b.neg,45)|FLD(b.abs,49)); return; }
    case 0x109: { /* POPC */
      b=slot32(L,w,form); int nt=b.neg; b.neg=0; u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"0101110000001---","0100110000001---","0011100-00001---"};
      put(L,P(pt[f])|bb|d|FLD(nt,40)); return; }
    case 0x100: { /* FLO */
      b=slot32(L,w,form); int nt=b.neg; b.neg=0; u64 bb; int f=mb(L,&b,0,&bb,S1); static const char* pt[3]={"0101110000110---","0100110000110---","0011100-00110---"};
      put(L,P(pt[f])|bb|d|FLD(nt,40)|FLD(F(w,73,1),48)|FLD(F(w,74,1),41)); return; }
    case 0x101: { /* BREV -> SHR.W with the reverse bit, by 0 */
      b=slot32(L,w,form); u32 s=toreg(L,&b,S1); put(L,P("0011100-00101---")|FLD(s,8)|d|FLD(1,39)|FLD(1,40)); return; }
    case 0x031: case 0x035: { /* HFMA2 used as a move: HFMA2.MMA Rd, -RZ, RZ, imm */
      bc(L,w,form,0,&b,&c);
      if(a.v==RZ && b.k==K_R && b.v==RZ){ if(c.k==K_I) mov32i(L,d,c.v); else if(c.k==K_C) movc(L,d,c.idx,c.off); else movr(L,d,c.v); return; }
      if(a.v==RZ && b.k==K_I && b.v==0){ if(c.k==K_R) movr(L,d,c.v); else untr(L,i,op12); return; }
      untr(L,i,op12); return; }
    case 0x148: return;   /* WARPSYNC */
    default: break; }
  }
  /* ------------------------------------------------ everything else (12 bit opcodes) */
  switch(op12){
  case 0x919: case 0x9c3: put(L,P("1111000011001---")|FLD(F(w,72,8),20)|d); return;               /* S2R, S2UR */
  case 0x805: put(L,P("0101000011001---")|FLD(F(w,72,8),20)|d); return;                         /* CS2R */
  case 0x822: { /* FSWZADD Rd, Ra, Rb, lanes (bits 32-39, per lane 0 = a+b, 1 = a-b, 2 = b-a, 3 = b): Maxwell has
                   codes 1 and 2 the other way round (mapping as in Eglantine's lowering). Screen-space derivatives */
    u32 sw=F(w,32,8), ra=gr(L,F(w,24,8)), rb=gr(L,F(w,64,8));
    if(F(w,40,24) || F(w,72,24)){ untr(L,i,op12); return; }
    u32 mw=((sw&0x55)<<1)|((sw&0xAA)>>1);
    put(L,P("0101000011111---")|d|FLD(ra,8)|FLD(rb,20)|FLD(mw,28)); return; }
  case 0xb82: case 0xab9: case 0xabb: { /* LDC, ULDC, ULDC indexed by a uniform register (c[bank][URi + off]) */
    u32 sz=F(w,73,3), ra= op12==0xab9? RZ : op12==0xabb? (F(w,24,6)==63? RZ : ur(F(w,24,6))) : gr(L,F(w,24,8)); i32 off=(int16_t)F(w,38,16); u32 idx=F(w,54,5);
    if(sz==6){ for(int k=0;k<2;k++) put(L,P("1110111110010---")|FLD(ra,8)|(d==RZ? RZ : d+2*k)|FLD(idx,36)|FLD((u32)(off+8*k)&0xFFFF,20)|FLD(5,48)); return; }
    put(L,P("1110111110010---")|FLD(ra,8)|d|FLD(idx,36)|FLD((u32)off&0xFFFF,20)|FLD(sz,48)); return; }
  case 0x326: { /* IPA */
    u32 freq=F(w,78,2); put(L,P("11100000--------")|FLD(F(w,64,8),30)|FLD(RZ,8)|FLD(RZ,20)|FLD(RZ,39)|d|FLD(freq==1? 2 : 0,54)|FLD(F(w,76,2)==1,52)); return; }
  case 0x321: { /* ALD */
    u32 vtx=gr(L,F(w,32,8)), off=gr(L,F(w,24,8));
    put(L,P("1110111111011---")|FLD(F(w,40,10),20)|FLD(F(w,74,2),47)|FLD(F(w,79,1),32)|FLD(vtx!=RZ? vtx : off,8)|FLD(vtx,39)|d|FLD(F(w,77,1),31)); return; }
  case 0x322: { /* AST */
    u32 data=gr(L,F(w,32,8)), vtx=gr(L,F(w,64,8)), off=gr(L,F(w,24,8));
    put(L,P("1110111111110---")|FLD(F(w,40,10),20)|FLD(F(w,74,2),47)|FLD(off,8)|FLD(vtx,39)|data|FLD(F(w,77,1),31)); return; }
  case 0x95b: put(L,P("111000110011----")|0xF); return;                                         /* KILL */
  case 0x94d: put(L,P("111000110000----")|0xF); return;                                         /* EXIT */
  case 0x547: { u32 up=upm(F(w,24,3)); if(up>7){ untr(L,i,op12); return; }   /* BRA.U with a uniform predicate */
    if(pg!=7 || pn){ untr(L,i,op12); return; } L->guard=FLD(up,16)|FLD(F(w,27,1),19); op12=0x947; }
    /* fall through */
  case 0x944: case 0x947: case 0x945: { /* CALL (MKW only calls blocks that never return: a jump), BRA, BSSY */
    if(op12==0x944) op12=0x947;
    int64_t off; if(op12==0x947){ off=(int64_t)((((u64)F(w,34,30))|((u64)F(w,64,18)<<30))<<16)>>16; }
    else off=(int64_t)((u64)F(w,34,30)<<34)>>34;
    int64_t t=(int64_t)i+1+off/4;      /* offsets count 32 bit words from the next instruction (NAK's get_rel_offset) */
    if(op12==0x947){ u32 cp=F(w,87,3), cn=F(w,90,1);
      if(cp!=7 || cn){ /* @Pg BRA Pc: taken when both hold. Maxwell has one predicate: @!Pg BRA over it, then @Pc BRA */
        if(pg!=7 || pn){ if(i+1>n){ untr(L,i,op12); return; }
          if(L->nfx==L->cfx){ L->cfx=L->cfx? L->cfx*2 : 64; L->fx=realloc(L->fx,L->cfx*sizeof *L->fx); }
          L->guard=FLD(pg,16)|FLD(!pn,19); L->fx[L->nfx].at=L->n; L->fx[L->nfx].tgt=i+1; L->fx[L->nfx].kind=0; L->nfx++; put(L,P("111000100100----")|0xF); }
        L->guard=FLD(cp,16)|FLD(cn,19); }
      if(F(w,32,1) && F(w,0,12)!=0x547){ untr(L,i,op12); return; } }
    if(t<0 || t>(int64_t)n){ untr(L,i,op12); return; }
    if(L->nfx==L->cfx){ L->cfx=L->cfx? L->cfx*2 : 64; L->fx=realloc(L->fx,L->cfx*sizeof *L->fx); }
    L->fx[L->nfx].at=L->n; L->fx[L->nfx].tgt=(u32)t; L->fx[L->nfx].kind=op12==0x945; L->nfx++;
    put(L, op12==0x947? P("111000100100----")|0xF : P("111000101001----")); return; }
  case 0x950: { /* RET: CALL is lowered as a jump, so a function called from one place returns with a jump back there */
    int t= (g_ret && i<g_nret)? g_ret[i] : -1; if(t<0 || pg!=7){ untr(L,i,op12); return; }
    if(L->nfx==L->cfx){ L->cfx=L->cfx? L->cfx*2 : 64; L->fx=realloc(L->fx,L->cfx*sizeof *L->fx); }
    L->fx[L->nfx].at=L->n; L->fx[L->nfx].tgt=(u32)t; L->fx[L->nfx].kind=0; L->nfx++; put(L,P("111000100100----")|0xF); return; }
  case 0x941: put(L,P("1111000011111---")|0xF); return;                                         /* BSYNC -> SYNC */
  case 0x804: { /* R2P PR, Ra, immediate mask, byte select (bits 76-77): predicates from register bits (UI vertex shaders
                   of MKW pick their texcoord path this way) */
    u32 mask=F(w,32,32); if(F(w,16,8) || mask>0x7F){ untr(L,i,op12); return; }
    put(L,P("0011100011110---")|FLD(gr(L,F(w,24,8)),8)|FLD(mask,20)|FLD(F(w,76,2),41)); return; }
  case 0x98e: case 0x3a8: case 0x9a8: { /* RED / ATOMG: global atomics (types 0 u32, 1 s32, 3 f32 go through; ops ADD MIN MAX INC DEC AND OR XOR as Maxwell) */
    /* the op is bits 87-89 (bit 90 is another flag, set in RED.MAX through a uniform pointer: read as 10, refused) */
    u32 addr=gr(L,F(w,24,8)), data=gr(L,F(w,32,8)), ty=F(w,73,3), aop=F(w,87,3); i32 off=(i32)(F(w,40,24)<<8)>>8;
    if((ty!=0 && ty!=1 && ty!=3) || off<-(1<<19) || off>=(1<<19)){ untr(L,i,op12); return; }
    if(op12==0x98e && F(w,91,1)){ /* + a 64 bit uniform pointer (bits 64-71): into R250:R251, Maxwell RED.E takes a pair */
      u32 u=ur(F(w,64,6)); if(u==RZ){ untr(L,i,op12); return; }
      if(addr==RZ){ movr(L,250,u); movr(L,251,u+1); }
      else { put(L,P("0101110000010---")|FLD(u,20)|FLD(addr,8)|250|FLD(1,47)); put(L,P("0101110000010---")|FLD(u+1,20)|FLD(addr+1,8)|251|FLD(1,43)); }
      addr=250; }
    if(op12==0x9a8 && F(w,91,1) && (F(w,64,6)&63)!=63){ put(L,P("0101110000010---")|FLD(addr,8)|FLD(ur(F(w,64,6)),20)|S2); addr=S2; }
    if(op12==0x98e) put(L,P("1110101111111---")|FLD(addr,8)|data|FLD(ty,20)|FLD(aop,23)|FLD((u32)off&0xFFFFF,28)|FLD(1,48));
    else put(L,P("11101101--------")|FLD(addr,8)|d|FLD(data,20)|FLD(ty,49)|FLD(aop,52)|FLD((u32)off&0xFFFFF,28)|FLD(1,48));
    return; }
  case 0x59f: { /* SURED.D.ADD.U32 with a bound handle (bits 40-53), value 32-39, coordinates 24-31 -> Maxwell SURED
                   (op ADD and type U32 are zero, dim 33, offset 36, bound 51). Other ops/types aren't seen yet */
    u32 dim=F(w,61,3), hoff=F(w,40,14), ra=gr(L,F(w,24,8)), rv=gr(L,F(w,32,8));
    if((w->hi&0xFFFFFFFFu)!=0x0010e000u || (dim!=0 && dim!=3) || hoff>0x1FFF){ untr(L,i,op12); return; }
    put(L,P("11101011010-----")|FLD(rv,0)|FLD(ra,8)|FLD(dim,33)|FLD(hoff,36)|FLD(1,51)); return; }
  case 0xb97: case 0xb9b: { /* SULD / SUST with a bound handle (c[bank][offset]), formatted with a component mask */
    u32 dim=F(w,61,3), hoff=F(w,40,14), mask=F(w,72,4); if(!mask) mask=1;
    u32 ra=gr(L,F(w,24,8)), rd= op12==0xb9b? gr(L,F(w,32,8)) : d;
    if(dim==1 || dim>5){ untr(L,i,op12); return; }
    put(L,P(op12==0xb97? "11101011000-----" : "11101011001-----")|FLD(1,51)|FLD(dim,33)|FLD(hoff&0x1FFF,36)|FLD(mask,20)|FLD(ra,8)|rd); return; }
  case 0x806: case 0x886: { /* VOTE / VOTEU (into a uniform register): the ballot and the vote */
    u32 pd=F(w,81,3); if(op12==0x886){ d=ur(F(w,16,6)); pd=upm(pd); if(pd>7){ untr(L,i,op12); return; } }
    put(L,P("0101000011011---")|d|FLD(F(w,87,3),39)|FLD(F(w,90,1),42)|FLD(F(w,72,2),48)|FLD(pd,45)); return; }
  case 0x81c: case 0x89c: { /* PLOP3 / UPLOP3: each predicate as a 0/~0 mask, LOP3 with the same table, then != 0 */
    /* a uniform predicate that lives in a register (g_upspill) is already that mask: its register is read and written
       directly. Without that, the sky shaders of MKW's 3D scenes kept 62 PLOP3s untranslated each */
    u32 sr[3]={F(w,87,3),F(w,77,3),F(w,68,3)}, sn[3]={F(w,90,1),F(w,80,1),F(w,71,1)}, dr[2]={F(w,81,3),F(w,84,3)};
    u32 lut[2]={(F(w,72,5)<<3)|F(w,64,3),F(w,16,8)};
    int su[3]={op12==0x89c,op12==0x89c,op12==0x89c || F(w,67,1)}, du=op12==0x89c;
    u32 rr[3]={S1,S2,SA};
    for(int k=0;k<3;k++){ u32 p=sr[k];
      if(su[k] && p!=7 && (g_upspill>>p&1)){ put(L,P("0101101111100---")|FLD(RZ,20)|FLD(RZ,39)|FLD(spr(p),8)|rr[k]|FLD(sn[k]? 0x0F : 0xF0,28)); continue; }   /* r = (~)reg */
      if(su[k]) p=upm(p); if(p>7){ untr(L,i,op12); return; }
      put(L,P("0101000010001---")|rr[k]|FLD(p,12)|FLD(sn[k],15)|FLD(7,29)|FLD(7,39)); }   /* PSET r = p AND PT AND PT */
    for(int k=0;k<2;k++){ u32 pdst=dr[k]; if(pdst==7) continue;
      if(du && (g_upspill>>pdst&1)){ put(L,P("0101101111100---")|FLD(S2,20)|FLD(SA,39)|FLD(S1,8)|spr(pdst)|FLD(lut[k],28)); continue; }
      if(du) pdst=upm(pdst); if(pdst>7){ untr(L,i,op12); return; }
      put(L,P("0101101111100---")|FLD(S2,20)|FLD(SA,39)|FLD(S1,8)|SB|FLD(lut[k],28));
      put(L,P("0011011-0110----")|FLD(SB,8)|FLD(pdst,3)|FLD(7,0)|FLD(7,39)|FLD(5,49)); }
    return; }
  case 0x803: { /* P2R Rd, PR, Ra, immediate mask (byte select 76-77) */
    u32 mask=F(w,32,32); if(mask>0x7F){ untr(L,i,op12); return; }
    put(L,P("0011100011101---")|FLD(gr(L,F(w,24,8)),8)|d|FLD(mask,20)|FLD(F(w,76,2),41)); return; }
  case 0xb69: { /* TMML with a bound handle: LOD query (MKW's terrain shaders pick their virtual texture
                   pages with it). Maxwell TMML takes the same coordinates and handle index */
    u32 dim=F(w,61,3), arr=(dim>>2)&1, type=dim&3, hoff=F(w,40,14), mask=F(w,72,4); if(!mask) mask=3;
    u32 ra=gr(L,F(w,24,8)), d1=gr(L,F(w,64,8));
    if(g_tex_bank<0) g_tex_bank=(int)F(w,54,5);
    put(L,P("1101111101011---")|FLD(ra,8)|FLD(RZ,20)|TO|FLD(type<<1|arr,28)|FLD(mask,31)|FLD(hoff&0x1FFF,36));
    u32 to[4]={d,d==RZ? RZ : d+1,d1,d1==RZ? RZ : d1+1}; u32 nr=0; for(int q=0;q<4;q++) if(mask>>q&1) nr++;
    for(u32 q=0;q<nr;q++) if(to[q]!=RZ) movr(L,to[q],TO+q);
    return; }
  case 0x883: { /* UP2UR URd, URZ, mask: the uniform predicates (in the predicates picked for them) as bits */
    u32 mask=F(w,32,32), dd=ur(F(w,16,6)); if(F(w,24,6)!=63 || mask>0x7F || F(w,76,2)){ untr(L,i,op12); return; }
    mov32i(L,dd,0);
    for(u32 k=0;k<7;k++) if(mask>>k&1){ u32 pp=upm(k); if(pp>7) continue;   /* never written or read: stays 0 */
      put(L,P("0101000010001---")|S1|FLD(pp,12)|FLD(7,29)|FLD(7,39));                              /* S1 = UPk ? ~0 : 0 */
      put(L,P("001111----------")|FLD(1u<<k,20)|FLD(dd,39)|FLD(S1,8)|dd|FLD(0xEA,48)); }         /* dd |= S1 & bit */
    return; }
  /* MATCH (.U64 bit 73, Pd 81-83) and REDUX (op 78-80: 0 AND 1 OR 2 XOR 3 SUM 4 MIN 5 MAX, .S32 bit 73, into a uniform
     register) become shader.c's SWR.WARP pseudo op (1111111100000: kind 20-23, signed 24, Pd 48-50), a subgroup op there */
  case 0x3a1: { u32 ra=gr(L,F(w,24,8)); if(F(w,73,1)) gr(L,F(w,24,8)+1);
    put(L,P("1111111100000---")|d|FLD(ra,8)|FLD(F(w,73,1)? 9 : 8,20)|FLD(F(w,81,3),48)); return; }
  case 0x3c4: { u32 op=F(w,78,3); if(op>5){ untr(L,i,op12); return; }
    put(L,P("1111111100000---")|ur(F(w,16,6))|FLD(gr(L,F(w,24,8)),8)|FLD(op,20)|FLD(F(w,73,1),24)|FLD(7,48)); return; }
  case 0x923: case 0x927: { /* ISBERD / ISBEWR: the task/mesh stage buffer. Kept as the pseudo op Eglantine's lowering
       uses (Maxwell 0xEFC0, free there): data/dest 0-7, Ra 8-15, offset 20-43, .32 44, .ATTR 45, .SKEW 46, .O 47,
       write 48. Ampere: Ra 24, offset 40-63, dest 16 / data 32, 75 .32, 76-77 region (3 = .ATTR), 78 .SKEW, 79 .O.
       shader.c has no task/mesh stages yet (ROADMAP 6.3): there it is still untranslated */
    int wr= op12==0x927; u32 region=F(w,76,2), out=F(w,79,1);
    if(F(w,64,11) || F(w,80,16) || (region!=0 && region!=3) || wr!=(int)out || (!wr && region!=3)){ untr(L,i,op12); return; }
    put(L,FLD(0xEFC,52)|(wr? FLD(gr(L,F(w,32,8)),0) : d)|FLD(gr(L,F(w,24,8)),8)|FLD(F(w,40,24),20)|FLD(F(w,75,1),44)|FLD(region==3,45)|
        FLD(F(w,78,1),46)|FLD(out,47)|FLD(wr,48)); return; }
  case 0x3c2: /* R2UR (the value of the first active lane; uniform in practice) */
    movr(L,ur(F(w,16,6)),gr(L,F(w,24,8))); return;
  case 0x389: case 0x589: case 0x989: case 0xf89: { /* SHFL: mode 58-59, lane immediate 53-57 (0x9/0xf) or register 32
       (0x3/0x5), clamp/segment mask immediate 40-52 (0x5/0xf) or register 64 (0x3/0x9), Pd 81-83. Was lane 0 only:
       warp reductions summed one lane 32 times */
    int limm= op12==0x989 || op12==0xf89, cimm= op12==0x589 || op12==0xf89;
    u64 lb= limm? FLD(F(w,53,5),20)|FLD(1,28) : FLD(gr(L,F(w,32,8)),20);
    u64 cb= cimm? FLD(F(w,40,13),34)|FLD(1,29) : FLD(gr(L,F(w,64,8)),39);
    put(L,P("1110111100010---")|FLD(gr(L,F(w,24,8)),8)|d|lb|cb|FLD(F(w,58,2),30)|FLD(F(w,81,3),48)); return; }
  case 0x942: case 0x918: case 0x91a: case 0x992: case 0x9ab: case 0x98f: case 0x946: case 0x355: case 0x356: case 0x95d:
    put(L,P("0101000010110---")); return;                                                       /* BREAK, NOP, DEPBAR, MEMBAR, ERRBAR, CCTL, YIELD, BMOV, NANOSLEEP */
  case 0xb1d: put(L,P("1111000010101---")); return;                                             /* BAR.SYNC */
  case 0x381: case 0x386: case 0x383: case 0x384: case 0x387: case 0x388:   /* the same without a uniform register in the address */
  case 0x981: case 0x984: case 0x983: case 0x986: case 0x988: case 0x987: { /* LDG LDS LDL STG STS STL */
    /* 0x38x -> 0x98x. It was op12|=0x900, which makes 0x388 0xB88: every LDG/STG/LDS/STS/LDL/STL without a uniform
       register fell to the last case and became STL storing R0 (MKW's fog compute lost its shared memory) */
    int nou= op12<0x900; if(nou) op12+=0x600;
    int st= op12==0x986||op12==0x988||op12==0x987; u32 addr=gr(L,F(w,24,8)); i32 off=(i32)(F(w,40,24)<<8)>>8; u32 sz=F(w,73,3);
    u32 ua= nou? 63 : st? F(w,64,6) : F(w,32,6); u32 data= st? gr(L,F(w,32,8)) : d;
    /* shared memory scales the address register: [R.X4], [R.X8], [R.X16] in bits 78-79 (1, 2, 3). Left out, every
       thread but the first of a workgroup hit the wrong slot (MKW's fog: one bright cell per 4x4 group) */
    if((op12==0x984 || op12==0x988) && F(w,78,2)){ put(L,P("0011100-01001---")|FLD(addr,8)|FLD(F(w,78,2)+1,20)|S1); addr=S1; }
    if((ua&63)!=63){ /* + uniform address */ put(L,P("0101110000010---")|FLD(addr,8)|FLD(ur(ua),20)|S2); addr=S2; }
    const char* pt= op12==0x981? "1110111011010---" : op12==0x984? "1110111101001---" : op12==0x983? "1110111101000---" :
                    op12==0x986? "1110111011011---" : op12==0x988? "1110111101011---" : "1110111101010---";
    put(L,P(pt)|FLD(addr,8)|data|FLD((u32)off&0xFFFFFF,20)|FLD(sz,48)|FLD(op12==0x981||op12==0x986? F(w,90,1) : 0,45)); return; }
  case 0xb60: case 0x361: case 0xb66: case 0x367: case 0xb63: case 0x364: case 0xb6f: case 0x370: { /* TEX TLD TLD4 TXQ */
    int bl= (op12&0xF00)==0x300; int kind= (op12&0xFF)==0x60||(op12&0xFF)==0x61? 0 : (op12&0xFF)==0x66||(op12&0xFF)==0x67? 1 : (op12&0xFF)==0x63||(op12&0xFF)==0x64? 2 : 3;
    u32 hoff= bl? 0 : F(w,40,14), hidx=F(w,54,5);
    u32 ra=gr(L,F(w,24,8)), rb=gr(L,F(w,32,8)), d1=gr(L,F(w,64,8)); u32 mask=F(w,72,4); if(!mask) mask=1;
    if(!bl && g_tex_bank<0) g_tex_bank=(int)hidx;   /* the bank of the bound handle (NVN2: 2), Maxwell takes it from tex_cb_index instead */
    if(kind==3){ put(L,P(bl? "1101111101010---" : "1101111101001---")|FLD(ra,8)|d|FLD(F(w,62,2)==0? 1 : 2,22)|FLD(mask,31)|FLD(hoff,36)); return; }
    u32 dim=F(w,61,3); int arr=dim>>2&1, type=dim&3; u32 lodm=kind==2? 0 : F(w,87,3); int aoffi= kind==2? F(w,76,2)!=0 : F(w,76,1); int dc= kind==1? 0 : F(w,78,1);
    if(lodm>3 || (kind==2 && F(w,76,2)==2)){ untr(L,i,op12); return; }
    if(F(w,60,1)){ /* scalar form: the sources are spread over Ra/Rb, gather them into the Maxwell lists */
      u32 nc=(type==3? 3 : type+1)+arr+bl, nb=(lodm>=2)+aoffi+dc; u32 tot=nc+nb, la[8]; u32 k=0;
      if(tot>4){ untr(L,i,op12); return; }
      u32 n0= tot==4? 2 : tot==3? 2 : 1;
      for(u32 q=0;q<n0;q++) la[k++]=ra+q; for(u32 q=n0;q<tot;q++) la[k++]=rb+(q-n0);
      /* bindless: the handle is the first source on Ampere, Maxwell has it first in Rb */
      u32 ia=0; if(bl){ movr(L,TB,la[0]); ia=1; }
      for(u32 q=0;q<nc-bl;q++) movr(L,TA+q,la[ia+q]);
      for(u32 q=0;q<nb;q++) movr(L,TB+bl+q,la[nc+q]);
      ra=TA; rb=TB; }
    else if(bl){ /* handle first in Ra on Ampere (src0) vs Maxwell's Rb list: rebuild */
      untr(L,i,op12); return; }
    u32 dst=TO;
    u64 base= kind==1? P("11011100--------")|FLD(lodm==3,55)|FLD(aoffi,35) :
              kind==2? P("110010----------")|FLD(dc,50)|FLD(aoffi,54)|FLD(F(w,87,2),56) :
                       P(bl? "1101111010------" : "11000-----------")|FLD(dc,50)|FLD(aoffi,54)|FLD(lodm,55);
    put(L,base|FLD(ra,8)|FLD(rb,20)|dst|FLD((u32)(type<<1|arr),28)|FLD(mask,31)|FLD(hoff,36));
    /* results: packed components go to d, d+1, d1, d1+1 */
    u32 nr=0; for(int q=0;q<4;q++) if(mask>>q&1) nr++;
    if(F(w,79,1)){ /* .F16 results (bit 79): two per register, the first in the low half, into d then d1. Written as
                      floats, HADD2.F32 unpacked garbage: MKW's UI text (an R8 glyph atlas) came out as NaN dots */
      u32 to2[2]={d,d1};
      for(u32 p=0;p<(nr+1)/2;p++){ if(to2[p]==RZ) continue; u32 lo=TO+2*p;
        put(L,P("0101110010101---")|FLD(lo,20)|SA|FLD(1,8)|FLD(2,10));                     /* SA = f16(first) */
        if(2*p+1<nr){ put(L,P("0101110010101---")|FLD(lo+1,20)|SB|FLD(1,8)|FLD(2,10));     /* SB = f16(second) */
          put(L,P("0011100-01001---")|FLD(16,20)|FLD(SB,8)|SB);                            /* SB <<= 16 */
          put(L,P("0101101111100---")|FLD(SB,20)|FLD(RZ,39)|FLD(SA,8)|to2[p]|FLD(0xFC,28)|FLD(7,48)|FLD(3,36)); }
        else movr(L,to2[p],SA); }
      return; }
    u32 to[4]={d,d==RZ? RZ : d+1,d1,d1==RZ? RZ : d1+1};
    for(u32 q=0;q<nr;q++) if(to[q]!=RZ) movr(L,to[q],TO+q);
    return; }
  default: break; }
  untr(L,i,op12); }

/* lowered Maxwell program (SPH + code with control words), malloc'd. alen = Ampere bytes (header included) */

/* uniform predicates left without a free predicate live in registers (0 / ~0): around the one instruction that reads
   or writes one, a predicate it doesn't touch is saved, used for it, and put back. MKW's vertex shaders
   pick cbuf values with seven UISETP/ULDC pairs on the same uniform index, seven uniform predicates alive at once */
#define SPILL_SAVE 168u
static void lower1(Lw* L, const W128* w, u32 i, u32 n){
  if(!g_upspill){ lower1i(L,w,i,n); return; }
  u32 op12=F(w,0,12), op9=op12&0x1FF, pg=F(w,12,3);
  int uni= ((op9&0x180)==0x080 && F(w,91,1)) || op12==0xc82 || op12==0x882;
  int uop= uni || op12==0x886 || op12==0x89c || op12==0x883 || op12==0x9c3 || op12==0xab9 || op12==0xabb;
  u32 o9= uni? (op9&0x17F) : op9;
  int gsp= uop && pg!=7 && (g_upspill>>pg&1);
  int dsp=-1; if(uni && (o9==0x00c || o9==0x012) && (o9==0x012 || F(w,84,3)==7) && F(w,81,3)!=7 && (g_upspill>>F(w,81,3)&1)) dsp=(int)F(w,81,3);
  if(!gsp && dsp<0){ lower1i(L,w,i,n); return; }
  if(gsp && dsp>=0){ untr(L,i,op12); return; }
  /* a predicate this instruction doesn't name */
  u32 busy=0; if(uop && pg!=7 && !gsp) busy|=1u<<upm(pg); if(!uop && pg!=7) busy|=1u<<pg;
  if(uni && o9==0x00c){ u32 pc=F(w,87,3); if(pc!=7) busy|=1u<<upm(pc); }
  int v=-1; for(int q=6;q>=0;q--) if(!(busy>>q&1)){ v=q; break; } if(v<0){ untr(L,i,op12); return; }
  u32 k= gsp? pg : (u32)dsp; int old=g_upmap[k];
  L->guard=FLD(7,16);   /* PT */
  put(L,P("0011100011101---")|FLD(RZ,8)|SPILL_SAVE|FLD(1u<<v,20));                                                  /* save v */
  if(gsp) put(L,P("010110110110----")|FLD(RZ,20)|FLD(spr(k),8)|FLD(v,3)|FLD(7,0)|FLD(7,39)|FLD(5,49));             /* v = reg != 0 */
  g_upmap[k]=v; lower1i(L,w,i,n); g_upmap[k]=old; L->guard=FLD(7,16);
  if(dsp>=0) put(L,P("0101000010001---")|spr(k)|FLD(v,12)|FLD(7,29)|FLD(7,39));                                     /* reg = v ? ~0 : 0 */
  put(L,P("0011100011110---")|FLD(SPILL_SAVE,8)|FLD(1u<<v,20));                                                     /* restore v */
  if(L->maxreg<(int)SPILL_SAVE) L->maxreg=(int)SPILL_SAVE; }

/* SWR_LOWER=egl: Eglantine's lowering first (egl_lower.cpp), ours for the programs it rejects; =ours: ours only */
extern int egl_lower_prog(const unsigned char* code, unsigned max_insts, unsigned long long** out, unsigned* nout,
                          int* tex_bank, unsigned* consumed, const char** err, unsigned* err_index);
static int lower_mode(void){ static int m=-1; if(m<0){ const char* e=getenv("SWR_LOWER"); m= e && !strcmp(e,"egl")? 1 : 0; } return m; }
static u64 st_egl_ok, st_egl_fail;
static u8* egl_try(const u8* code, u32 max_bytes, u32 hdr, u32* out_len, u32* alen, Sm86Info* inf){
  unsigned long long* m=NULL; unsigned nm=0, used=0, ei=0; int tb=-1; const char* err=NULL;
  if(!egl_lower_prog(code+hdr,(max_bytes-hdr)/16,&m,&nm,&tb,&used,&err,&ei)){
    st_egl_fail++; { extern void logf_(int,const char*,...); static int said; if(said<60){ said++; u32 op=0; if(ei<(max_bytes-hdr)/16){ u32 w0; memcpy(&w0,code+hdr+ei*16,4); op=w0&0xFFF; }
      logf_(1,"sm86: Eglantine lowering refused a program (%s at instruction %u, op %03x): ours instead (%llu lowered, %llu refused)\n",err? err : "?",ei,op,(unsigned long long)st_egl_ok,(unsigned long long)st_egl_fail); } }
    return NULL; }
  st_egl_ok++;
  /* same layout as ours below: header, then a control word + 3 instructions; EXIT after the end */
  u32 n=nm+1, base=hdr/8, words=base+((n+2)/3)*4+4;
  u64* o=calloc(words+8,8); if(hdr) memcpy(o,code,hdr);
#define ESLOT(k) (base+((k)/3)*4+1+((k)%3))
  for(u32 k=0;k<nm;k++) o[ESLOT(k)]=m[k];
  o[ESLOT(nm)]=P("111000110000----")|0xF;
  for(u32 g=0; g<(n+2)/3; g++) o[base+g*4]=0x001F8000FC0007E0ull;
  *out_len=(ESLOT(n-1)+1)*8; if(alen) *alen=hdr+used*16;
#undef ESLOT
  if(inf){ memset(inf,0,sizeof *inf); inf->insts=used; inf->maxreg=-1; inf->tex_bank=tb; }
  free(m); return (u8*)o; }

u8* sm86_lower(const u8* code, u32 max_bytes, int has_sph, u32* out_len, u32* alen, Sm86Info* inf){
  u32 hdr= has_sph? 0x50 : 0; if(max_bytes<hdr+16) return NULL;
  if(lower_mode()==1){ u8* r=egl_try(code,max_bytes,hdr,out_len,alen,inf); if(r) return r; }
  u32 n=(max_bytes-hdr)/16; g_tex_bank=-1;
  /* length: past the last EXIT / self branch that's beyond every forward target */
  u32 end=0, maxt=0;
  for(u32 i=0;i<n;i++){ W128 w; memcpy(&w,code+hdr+i*16,16); u32 op=F(&w,0,12);
    if(op==0x944) op=0x947;
    if(op==0x947||op==0x945){ int64_t off= op==0x947? (int64_t)((((u64)F(&w,34,30))|((u64)F(&w,64,18)<<30))<<16)>>16 : (int64_t)((u64)F(&w,34,30)<<34)>>34;
      int64_t t=(int64_t)i+1+off/4; if(t>(int64_t)maxt && t<(int64_t)n) maxt=(u32)t;
      if(op==0x947 && t==(int64_t)i && i>=maxt && F(&w,12,4)==7){ end=i+1; break; } }
    if(op==0x94d && F(&w,12,4)==7 && i>=maxt){ end=i+1; /* NVIDIA ends with EXIT then BRA to self */ continue; }
    if(end && i>end+2) break;
    if(i>=32768) break; }   /* 32768: MKW has a 15671 instruction compute (f0b45cf863f3f0f9) */
  if(!end) end= n<32768? n : 32768;
  { /* which predicates the shader uses, and which uniform ones: those get the free ones */
    u32 usedp=0, usedu=0, uall=0; int ufirst[8], ulast[8]; for(int k=0;k<8;k++){ ufirst[k]=-1; ulast[k]=-1; }
    for(u32 i=0;i<end;i++){ W128 w; memcpy(&w,code+hdr+i*16,16); u32 op=F(&w,0,12), o9=op&0x1FF; int un=(o9&0x180)==0x080 && F(&w,91,1);
      uall|=usedu; usedu=0;
      u32 a9= un? (o9&0x17F) : o9;
      int uop= un || op==0x886 || op==0x89c || op==0x883 || op==0x9c3 || op==0xab9 || op==0xabb || op==0xc82 || op==0x882;
      u32* set= uop? &usedu : &usedp;
      *set|=1u<<F(&w,12,3);
      int alu= op<0x200 || (op&0xE00)!=0x000;
      if(alu && (a9==0x00c||a9==0x00b||a9==0x00a||a9==0x02a||a9==0x034||a9==0x010||a9==0x011||a9==0x012||a9==0x024||a9==0x025||a9==0x007||a9==0x008||a9==0x009||a9==0x017)){
        *set|=1u<<F(&w,81,3); *set|=1u<<F(&w,84,3); *set|=1u<<F(&w,87,3); if(a9==0x010) *set|=1u<<F(&w,77,3); }
      if(op==0x81c){ usedp|=1u<<F(&w,81,3)|1u<<F(&w,84,3)|1u<<F(&w,87,3)|1u<<F(&w,77,3); if(F(&w,67,1)) usedu|=1u<<F(&w,68,3); else usedp|=1u<<F(&w,68,3); }
      if(op==0x89c) usedu|=1u<<F(&w,81,3)|1u<<F(&w,84,3)|1u<<F(&w,87,3)|1u<<F(&w,77,3)|1u<<F(&w,68,3);
      if(op==0x886){ usedu|=1u<<F(&w,81,3); usedp|=1u<<F(&w,87,3); }
      if(op==0x806 || op==0x3a1 || (op&0x1FF)==0x189 || op==0xb97) usedp|=1u<<F(&w,81,3)|1u<<F(&w,87,3);
      if(op==0x947 || op==0x944) usedp|=1u<<F(&w,87,3);
      if(op==0x547) usedu|=1u<<F(&w,24,3);
      if(op==0x804 || op==0x803) usedp|=F(&w,32,7);
      /* UP2UR saves all 7 uniform predicates (mask 0x7F) even when the shader uses two: those don't take a predicate */
      for(int k=0;k<7;k++) if(usedu>>k&1){ if(ufirst[k]<0) ufirst[k]=(int)i; ulast[k]=(int)i; } }
    usedu|=uall;
    usedp&=0x7F; usedu&=0x7F;
    { static int on=-1; if(on<0) on=getenv("SWR_SM86_TRACE")!=NULL; if(on) fprintf(stderr,"sm86 preds used %02x uniform %02x\n",usedp,usedu); }
    for(int k=0;k<8;k++) g_upmap[k]=8;
    /* uniform predicates share a free predicate when their ranges (first to last mention) don't overlap. a backward
       branch makes the ranges unreliable, then each needs its own */
    int back=0; for(u32 i=0;i<end;i++){ W128 w; memcpy(&w,code+hdr+i*16,16); u32 op=F(&w,0,12);
      if(op==0x947 || op==0x547){ int64_t off=(int64_t)((((u64)F(&w,34,30))|((u64)F(&w,64,18)<<30))<<16)>>16; if(off<0) back=1; } }
    int qlast[7]; for(int q=0;q<7;q++) qlast[q]=-2;
    for(int it=0;it<7;it++){ int k=-1; for(int j=0;j<7;j++) if((usedu>>j&1) && g_upmap[j]==8 && (k<0 || ufirst[j]<ufirst[k])) k=j; if(k<0) break;
      int pick=-1; for(int q=0;q<7;q++) if(!((usedp>>q)&1) && qlast[q]==-2){ pick=q; break; }
      if(pick<0 && !back) for(int q=0;q<7;q++) if(!((usedp>>q)&1) && qlast[q]>=0 && qlast[q]<ufirst[k]){ pick=q; break; }
      if(pick<0){ g_upmap[k]=9; continue; }
      g_upmap[k]=pick; qlast[pick]=ulast[k]; }
    { static int on=-1; if(on<0) on=getenv("SWR_SM86_TRACE")!=NULL; if(on){ fprintf(stderr,"sm86 back %d",back); for(int k=0;k<7;k++) fprintf(stderr," UP%d[%d..%d]->%d",k,ufirst[k],ulast[k],g_upmap[k]); fprintf(stderr,"\n"); } }
    g_upspill=0; for(int k=0;k<7;k++) if(g_upmap[k]==9) g_upspill|=1<<k;
    for(int k=0;k<8;k++) if(g_upmap[k]==9) g_upmap[k]=8; }
  { /* RET targets: each function's start is a CALL target; one call site -> return right after it */
    free(g_ret); g_ret=malloc(sizeof(int)*(end+1)); g_nret=end; for(u32 i=0;i<end;i++) g_ret[i]=-1;
    u32 tg[64], site[64], cnt[64]; int nt=0;
    for(u32 i=0;i<end;i++){ W128 w; memcpy(&w,code+hdr+i*16,16); if(F(&w,0,12)!=0x944) continue;
      int64_t off=(int64_t)((((u64)F(&w,34,30))|((u64)F(&w,64,18)<<30))<<16)>>16; int64_t t=(int64_t)i+1+off/4; if(t<0 || t>=(int64_t)end) continue;
      int k=0; for(;k<nt;k++) if(tg[k]==(u32)t) break; if(k==nt){ if(nt==64) continue; tg[nt]=(u32)t; cnt[nt]=0; nt++; } cnt[k]++; site[k]=i; }
    for(u32 i=0;i<end;i++){ W128 w; memcpy(&w,code+hdr+i*16,16); if(F(&w,0,12)!=0x950) continue;
      int best=-1; for(int k=0;k<nt;k++) if(tg[k]<=i && (best<0 || tg[k]>tg[best])) best=k;
      if(best>=0 && cnt[best]==1) g_ret[i]=(int)site[best]+1; } }
  Lw L={0}; L.maxreg=-1; L.amap=malloc(sizeof(u32)*(end+1)); L.namp=end;
  for(u32 i=0;i<end;i++){ W128 w; memcpy(&w,code+hdr+i*16,16); L.amap[i]=L.n; g_cur=&w; lower1(&L,&w,i,end); g_cur=NULL; L.guard=0; }
  L.amap[end]=L.n; put(&L,P("111000110000----")|0xF);   /* EXIT at the end in case the last block falls through */
  /* layout: SPH, then groups of control word + 3 instructions */
  u32 nm=L.n, words=hdr/8 + ((nm+2)/3)*4 + 4;
  u64* o=calloc(words+8,8); if(hdr) memcpy(o,code,hdr);
  u32 base=hdr/8;
#define SLOT(k) (base+((k)/3)*4+1+((k)%3))
  for(u32 k=0;k<nm;k++) o[SLOT(k)]=L.m[k];
  for(u32 g=0; g<(nm+2)/3; g++) o[base+g*4]=0x001F8000FC0007E0ull;   /* control word: no stalls/barriers */
  for(u32 f=0;f<L.nfx;f++){ u32 src=SLOT(L.fx[f].at), dst=SLOT(L.amap[L.fx[f].tgt]); int32_t off=(int32_t)(dst-src-1)*8;
    o[src]=(o[src]&~(0xFFFFFFull<<20))|((u64)((u32)off&0xFFFFFF)<<20); }
  *out_len=(SLOT(nm>0? nm-1 : 0)+1)*8; if(alen) *alen=hdr+end*16;
  if(inf){ inf->insts=end; inf->untranslated=L.bad; inf->first_bad=L.first_bad; inf->bad_op=L.bad_op; inf->maxreg=L.maxreg; inf->reg_clash= L.maxreg>=(int)UBASE; inf->tex_bank=g_tex_bank; }
  free(L.m); free(L.amap); free(L.fx);
  return (u8*)o; }
