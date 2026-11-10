"""Per-function view of a host profile (SWR_HOST_PROF=<s> -> host_profile.txt): for each thread above a CPU share,
its samples summed by function, resolved with nm on a runtime built with SWR_KEEP_SYMBOLS=1.

  python tools/host_prof_syms.py <host_profile.txt> <runtime exe> [--min 20] [--top 30]
"""
import argparse, bisect, collections, os, re, subprocess

import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2r_paths as P
NM = P.gcc_tool('nm')
BASE = 0x140000000


def load_syms(exe):
    out = subprocess.run([NM, '-n', '-C', exe], capture_output=True, text=True, errors='replace').stdout
    syms = []
    for line in out.splitlines():
        m = re.match(r'([0-9a-fA-F]+) [tTwW] (.*)', line)
        if m: syms.append((int(m.group(1), 16), m.group(2)))
    syms.sort()
    return [s[0] for s in syms], [s[1] for s in syms]


def short(name):
    name = re.sub(r'\(.*', '', name)               # drop the parameter list
    name = re.sub(r'<[^<>]*(<[^<>]*>[^<>]*)*>', '<>', name)   # templates
    return name[-110:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('profile'); ap.add_argument('exe')
    ap.add_argument('--min', type=float, default=20); ap.add_argument('--top', type=int, default=30)
    a = ap.parse_args()
    keys, names = load_syms(a.exe)
    threads = []; cur = None
    for line in open(a.profile, errors='replace'):
        m = re.match(r"thread (\d+) '([^']*)': (\d+)% of a core, (\d+) samples", line)
        if m:
            cur = {'id': m.group(1), 'name': m.group(2), 'pct': int(m.group(3)), 'n': int(m.group(4)),
                   'hits': collections.Counter()}
            threads.append(cur); continue
        m = re.match(r'\s+(\d+) (\S+)\+([0-9a-f]+)', line)
        if m and cur:
            cnt, mod, off = int(m.group(1)), m.group(2), int(m.group(3), 16)
            if mod == 'exe':
                i = bisect.bisect_right(keys, BASE + off) - 1
                fn = short(names[i]) if i >= 0 else 'exe?'
            else:
                fn = mod
            cur['hits'][fn] += cnt
    for t in sorted(threads, key=lambda t: -t['pct']):
        if t['pct'] < a.min: continue
        total = sum(t['hits'].values()) or 1
        print('\n== thread %s %s: %d%% of a core' % (t['id'], t['name'], t['pct']))
        for fn, c in t['hits'].most_common(a.top):
            print('  %5.1f%%  %s' % (100.0 * c / total, fn))


if __name__ == '__main__':
    main()
