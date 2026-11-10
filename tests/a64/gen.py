# SwitchRecomp - Copyright 2026 MZ Labs
# Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
# Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen.
"""random LSE / LDAPR / PAC words, run on qemu-aarch64 -cpu max and through a64.py's C; prints C files"""
import sys, random, importlib.util
sp = importlib.util.spec_from_file_location('a64', sys.argv[1] + '/a64.py'); a = importlib.util.module_from_spec(sp); sp.loader.exec_module(a)
out = sys.argv[2]; random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
cases = []
def regs3():
    while True:
        rn = random.randrange(16); rt = random.choice(list(range(16)) + [31]); rs = random.choice(list(range(16)) + [31])
        if rn not in (rt, rs): return rn, rt, rs
for size in range(4):
    for op in range(9):
        for ar in range(4):
            for k in range(3):
                rn, rt, rs = regs3()
                o3, opc = (1, 0) if op == 8 else (0, op)
                w = (size << 30) | (0b111000 << 24) | ((ar >> 1) << 23) | ((ar & 1) << 22) | (1 << 21) | (rs << 16) | (o3 << 15) | (opc << 12) | (rn << 5) | rt
                cases.append((w, rn, 0, 0))
for size in range(4):
    for k in range(4):
        rn, rt, _ = regs3()
        if rt == 31: rt = 0 if rn else 1
        cases.append(((size << 30) | 0x38BFC000 | (rn << 5) | rt, rn, 0, 0))
for size in range(4):   # CAS, CASA, CASL, CASAL: rs, rt, rn distinct, rt2 = 11111
    for lo in range(4):
        for k in range(4):
            rn, rt, rs = regs3()
            if rs == 31: rs = (rn + 1) % 16
            if rs == rt: rt = 31
            w = (size << 30) | 0x08A07C00 | ((lo >> 1) << 22) | ((lo & 1) << 15) | (rs << 16) | (rn << 5) | rt
            cases.append((w, rn, rs, 1))
for sz in range(2):     # CASP: even pairs, apart from rn and from each other
    for lo in range(4):
        for k in range(4):
            while True:
                rs = random.randrange(0, 16, 2); rt = random.randrange(0, 16, 2); rn = random.randrange(16)
                if rs != rt and rn not in (rs, rs + 1, rt, rt + 1): break
            w = (sz << 30) | 0x08207C00 | ((lo >> 1) << 22) | ((lo & 1) << 15) | (rs << 16) | (rn << 5) | rt
            cases.append((w, rn, rs, 2 + sz))
for k in range(0):   # PAC/AUT: FPAC on qemu -cpu max faults on unsigned pointers; ours are NOPs by design
    rd = random.randrange(16); rn = random.randrange(16)
    cases.append((0xDAC10000 | (random.choice([0, 1, 4, 5]) << 10) | (rn << 5) | rd, None))
gen = open(out + '/gen.c', 'w'); ref = open(out + '/ref.c', 'w')
hdr = '#include <stdint.h>\n#include <string.h>\n#include <stdio.h>\ntypedef uint64_t u64;\n'
ref.write(hdr + 'struct T { unsigned w; int rn, rs, kind; } T[] = {' + ','.join('{0x%08x,%d,%d,%d}' % (w, -1 if rn is None else rn, rs, kind) for w, rn, rs, kind in cases) + '};\nint NT=%d;\n' % len(cases))
for i, (w, rn, rs, kind) in enumerate(cases):
    loads = ''.join('"ldr x%d,[x17,#%d]\\n"' % (r, r * 8) for r in range(16))
    stores = ''.join('"str x%d,[x17,#%d]\\n"' % (r, r * 8) for r in range(16))
    ref.write('static void r%d(u64* x){ __asm__ volatile("mov x17,%%0\\n" %s ".inst 0x%08x\\n" %s :: "r"(x) : "x0","x1","x2","x3","x4","x5","x6","x7","x8","x9","x10","x11","x12","x13","x14","x15","x17","memory"); }\n' % (i, loads, w, stores))
ref.write('typedef void (*F)(u64*); F R[] = {' + ','.join('r%d' % i for i in range(len(cases))) + '};\n')
gen.write('#include "cpu.h"\n')
for i, (w, rn, rs, kind) in enumerate(cases):
    c = a.emit_static(w, 0x1000)
    if not isinstance(c, str): sys.exit('case %d %08x: %r' % (i, w, c))
    gen.write('static void g%d(CPU* c, u8* M){ %s }\n' % (i, c))
gen.write('typedef void (*G)(CPU*,u8*); G GF[] = {' + ','.join('g%d' % i for i in range(len(cases))) + '};\n')
print(len(cases), 'cases')
