"""The hottest exe offsets of a named function, with the disassembly around them, summed over one or more host profiles
(the thread that spends the most samples in the function, per profile).

  python tools/host_prof_hot.py <exe> <function regex> <host_profile.txt> [<host_profile.txt> ...]
"""
import bisect, collections, re, subprocess, sys

exe, pattern = sys.argv[1:3]
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import s2r_paths as P
B = P.gcc_tool('')
out = subprocess.run([B + 'nm.exe', '-n', '-C', exe], capture_output=True, text=True, errors='replace').stdout
syms = []
for line in out.splitlines():
    m = re.match(r'([0-9a-fA-F]+) [tTwW] (.*)', line)
    if m:
        syms.append((int(m.group(1), 16), m.group(2)))
syms.sort()
keys = [s[0] for s in syms]
c = collections.Counter()
for prof in sys.argv[3:]:
    best = collections.Counter()
    for b in re.split(r'(?m)^(?=thread )', open(prof, errors='replace').read())[1:]:
        here = collections.Counter()
        for k, off in re.findall(r'(?m)^\s+(\d+) exe\+([0-9a-f]+)\s*$', b):
            a = 0x140000000 + int(off, 16)
            i = bisect.bisect_right(keys, a) - 1
            if i >= 0 and re.search(pattern, syms[i][1]):
                here[a] += int(k)
        if sum(here.values()) > sum(best.values()):
            best = here
    c.update(best)
if not c:
    sys.exit('no samples in ' + pattern)
print('samples in the function', sum(c.values()))
for a, k in c.most_common(10):
    print(hex(a), k)
top = [a for a, _ in c.most_common(4)]
dis = subprocess.run([B + 'objdump.exe', '-d', '--no-show-raw-insn', '--start-address=%d' % (min(top) - 40),
                      '--stop-address=%d' % (max(top) + 24), exe], capture_output=True, text=True, errors='replace').stdout
for line in dis.splitlines():
    m = re.match(r'\s*([0-9a-f]+):', line)
    print(('%4d ' % c.get(int(m.group(1), 16), 0) if m else '     ') + line[:90])
