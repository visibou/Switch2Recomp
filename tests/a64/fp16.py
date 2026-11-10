# SwitchRecomp - Copyright 2026 MZ Labs
# Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
# Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen.
"""FP16 FABS/FNEG (a64.py FP16_ABSNEG): writes ref.c (qemu side, the real instructions) and gen.c (a64.py's C)"""
import sys, importlib.util
sp = importlib.util.spec_from_file_location('a64', sys.argv[1] + '/a64.py'); a = importlib.util.module_from_spec(sp); sp.loader.exec_module(a)
out = sys.argv[2]
ws = [(q << 30) | (u << 29) | 0x0EF8F800 | (1 << 5) | 0 for q in (0, 1) for u in (0, 1)]   # Vd=0, Vn=1
ref = open(out + '/ref.c', 'w'); gen = open(out + '/gen.c', 'w')
ref.write('#include <stdio.h>\n#include <string.h>\n')
for i, w in enumerate(ws):
    ref.write('static void r%d(unsigned char* io){ __asm__ volatile("ldr q1,[%%0]\\n movi v0.16b,#0x5a\\n .inst 0x%08x\\n str q0,[%%0,#16]\\n" :: "r"(io) : "v0","v1","memory"); }\n' % (i, w))
ref.write('static void (*R[])(unsigned char*)={%s};\n' % ','.join('r%d' % i for i in range(len(ws))))
gen.write('#include "cpu.h"\n#include <stdio.h>\n')
for i, w in enumerate(ws):
    gen.write('static void g%d(CPU* c, u8* M){ %s }\n' % (i, a.emit_static(w, 0)))
gen.write('static void (*G[])(CPU*,u8*)={%s};\n' % ','.join('g%d' % i for i in range(len(ws))))
for f, side in ((ref, 'R[i](io);'), (gen, 'static CPU c; memset(&c,0,sizeof c); memcpy(&c.v[1],io,16); memset(&c.v[0],0x5a,16); G[i](&c,0); memcpy(io+16,&c.v[0],16);')):
    f.write('''int main(void){ unsigned long long s=99; unsigned char io[32];
  static const unsigned short sp[]={0x7C00,0xFC00,0x7E00,0xFE01,0x0000,0x8000,0x3C00,0x0001};
  for(int k=0;k<200;k++) for(int i=0;i<%d;i++){ for(int b=0;b<16;b++){ s^=s<<13; s^=s>>7; s^=s<<17; io[b]=(unsigned char)s; }
    if(k&1) for(int l=0;l<8;l++) memcpy(io+2*l,&sp[(l+k)&7],2);
    %s printf("%%d %%d",k,i); for(int b=0;b<32;b++) printf(" %%02x",io[b]); printf("\\n"); } return 0; }
''' % (len(ws), side))
