"""What the threads were doing across all stall dumps of a run (SWR_NVN_STACKS=1, build with SWR_KEEP_SYMBOLS=1): python tools/native_profile.py <run folder> [substring of the stack, default vkr_exec]
Counts, for the stacks that hold the substring, the innermost exe functions (the first three)."""
import bisect, collections, os, re, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import s2r_paths as P
syms = []
for l in subprocess.run([P.gcc_tool('nm'), os.path.join(ROOT, 'bin', 'runtime_win64_win32.exe')], capture_output=True, text=True).stdout.splitlines():
    m = re.match(r'([0-9a-f]+) [tT] (\S+)', l)
    if m: syms.append((int(m.group(1), 16), m.group(2)))
syms.sort(); keys = [s[0] for s in syms]
want = sys.argv[2] if len(sys.argv) > 2 else 'vkr_exec'
c = collections.Counter(); n = 0; base = 0
for l in open(os.path.join(sys.argv[1], 'game.log'), encoding='utf-8', errors='replace'):
    m = re.search(r'exe base ([0-9a-f]+)', l)
    if m: base = int(m.group(1), 16); continue
    m = re.match(r'\[[^]]*\]\s+thread (\d+) ?(.*?): (.*)', l)
    if not m or not base: continue
    fr = []
    for a in m.group(3).split():
        try: v = int(a, 16)
        except ValueError: continue
        if base <= v < base + 0x8000000:
            k = bisect.bisect_right(keys, v - base + 0x140000000) - 1
            fr.append(syms[k][1] if k >= 0 else '?')
    if want in ' '.join(fr): c[' < '.join(fr[:3])] += 1; n += 1
print(n, 'stacks with', want)
for k, v in c.most_common(14): print(v, k)
