"""exe offsets to the nearest function, from the linker map: python tools/exe_syms.py +bbf3 +bb50 ..."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
syms = []
for l in open(os.path.join(ROOT, 'bin', 'runtime_win64_win32.map'), errors='replace'):
    m = re.match(r'\s+0x0*([0-9a-f]+)\s+([A-Za-z_.][\w.$@?]*)\s*$', l)
    if m: syms.append((int(m.group(1), 16), m.group(2)))
syms.sort()
base = 0x140000000
import bisect
keys = [s[0] for s in syms]
for a in sys.argv[1:]:
    o = int(a.lstrip('+'), 16) + base
    i = bisect.bisect_right(keys, o) - 1
    print(a, syms[i][1] if i >= 0 else '?', '+%x' % (o - syms[i][0]) if i >= 0 else '')
