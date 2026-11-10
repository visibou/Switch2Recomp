"""Resolve runtime offsets (profile.txt "runtime (offset in the exe)" lines, or offsets given on the command line)
to function names, using nm on an exe built with SWR_KEEP_SYMBOLS=1.
Usage: python prof_syms.py <exe> <nm.exe> profile.txt     (sums the profile per function)
       python prof_syms.py <exe> <nm.exe> 8ed70 50b00 ...   (one offset each)"""
import bisect, re, subprocess, sys

exe, nm = sys.argv[1], sys.argv[2]
syms = []
for line in subprocess.run([nm, '-n', exe], capture_output=True, text=True).stdout.splitlines():
    p = line.split()
    if len(p) == 3 and p[1] in 'tT':
        syms.append((int(p[0], 16), p[2]))
syms.sort()
keys = [s[0] for s in syms]
BASE = 0x140000000


def name(off):
    a = BASE + off
    i = bisect.bisect_right(keys, a) - 1
    return '%s+%x' % (syms[i][1], a - syms[i][0]) if i >= 0 else '?'


args = sys.argv[3:]
if len(args) == 1 and args[0].endswith('.txt'):
    # sections: "GPU thread: ..." (one thread on its own) and "runtime (offset in the exe):" (all threads)
    sections, cur = [], None
    for line in open(args[0]):
        if line.startswith('GPU thread') or line.startswith('runtime'):
            cur = (line.strip(), {})
            sections.append(cur)
            continue
        if cur:
            m = re.match(r'\s+([\d.]+)%\s+\+([0-9a-f]+)', line)
            if not m:
                if line.strip():
                    cur = None
                continue
            fn = name(int(m.group(2), 16)).split('+')[0]
            cur[1][fn] = cur[1].get(fn, 0) + float(m.group(1))
    for title, agg in sections:
        print('==', title)
        for fn, v in sorted(agg.items(), key=lambda x: -x[1])[:30]:
            print('%6.2f%%  %s' % (v, fn))
    # wait chains: "  12.3%  +a +b +c" -> names
    sec = False
    for line in open(args[0]):
        if line.startswith('focus thread outside'):
            sec = True
            print('== waits of the focus thread (innermost first)')
            continue
        if sec:
            m = re.match(r'\s+([\d.]+)%\s+((?:\+[0-9a-f]+\s*)+)$', line)
            if not m:
                break
            chain = ' <- '.join(name(int(x[1:], 16)).split('+')[0] for x in m.group(2).split() if x != '+0')
            print('%6.2f%%  %s' % (float(m.group(1)), chain))
else:
    for a in args:
        print(a, name(int(a, 16)))
