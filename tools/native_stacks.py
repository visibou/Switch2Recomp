"""The stall dumps of a run's game.log (SWR_NATIVE_STACKS=1, build with SWR_KEEP_SYMBOLS=1) with exe addresses named: python tools/native_stacks.py <run folder> [dump number from the end, default 1]
Only threads that have a frame in the exe are shown, innermost first."""
import bisect, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import s2r_paths as P
import subprocess
syms = []
for l in subprocess.run([P.gcc_tool('nm'), os.path.join(ROOT, 'bin', 'runtime_win64_win32.exe')], capture_output=True, text=True).stdout.splitlines():
    m = re.match(r'([0-9a-f]+) [tT] (\S+)', l)
    if m: syms.append((int(m.group(1), 16), m.group(2)))
syms.sort(); keys = [s[0] for s in syms]
L = open(os.path.join(sys.argv[1], 'game.log'), encoding='utf-8', errors='replace').read().splitlines()
ids = [i for i, l in enumerate(L) if 'native stacks (exe base' in l]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
i = ids[-n]; base = int(re.search(r'exe base ([0-9a-f]+)', L[i]).group(1), 16)
print(L[i][:60])
for l in L[i + 1:i + 160]:
    m = re.match(r'\[[^]]*\]\s+thread (\d+) ?(.*?): (.*)', l)
    if not m: continue
    out = []
    for a in m.group(3).split():
        try: v = int(a, 16)
        except ValueError: continue
        if base <= v < base + 0x8000000:
            k = bisect.bisect_right(keys, v - base + 0x140000000) - 1
            out.append(syms[k][1] if k >= 0 else '?')
        elif out and out[-1] != '.': out.append('.')
    if any(o != '.' for o in out): print(m.group(1), m.group(2), ' '.join(out)[:220])
