"""Two sets of host profiles compared on the GPU back thread, in ms a frame by function: share of the 10 s sample / FPS.
Shares alone mislead when the frame rates differ. ntdll and ucrtbase are named by their nearest export (a rough name:
internal functions take the name of the export before them).

  python tools/host_prof_cmp.py <exe> on=<profile>@<fps> ... off=<profile>@<fps> ... [--top 40] [--match regex]

Arm names are free, there have to be two. <fps> is the run's FPS as bench.py printed it. Each sample is one profile:
SWR_HOST_PROF=<second> python tools/bench.py <name> --runs 1 --replay ... --race-window 23:193 --seconds 350 [SWR_EDEN_F_OFF=abcdef]
leaves it in games/MKW/host_profile.txt. The sampler lists 512 pcs a thread: about 80% of the samples get a name.
"""
import argparse, bisect, collections, os, re, statistics, struct, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host_prof_syms import load_syms, short, BASE

BACK = 'VideoCommon::GpuSplit::BackLoop'


def exports(p):
    d = open(p, 'rb').read()
    pe = struct.unpack_from('<I', d, 0x3c)[0]; nsec = struct.unpack_from('<H', d, pe + 6)[0]
    optsz = struct.unpack_from('<H', d, pe + 20)[0]
    opt = pe + 24; ex = struct.unpack_from('<I', d, opt + 112)[0]
    secs = [struct.unpack_from('<8sIIII', d, opt + optsz + 40 * i) for i in range(nsec)]
    def off(r):
        for n, vs, va, rs, rp in secs:
            if va <= r < va + max(vs, rs): return r - va + rp
    e = off(ex); nf, nn, af, an, ao = struct.unpack_from('<5I', d, e + 20)
    out = []
    for i in range(nn):
        nr = struct.unpack_from('<I', d, off(an) + 4 * i)[0]; o = struct.unpack_from('<H', d, off(ao) + 2 * i)[0]
        out.append((struct.unpack_from('<I', d, off(af) + 4 * o)[0], d[off(nr):d.index(b'\0', off(nr))].decode()))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exe'); ap.add_argument('samples', nargs='+')
    ap.add_argument('--top', type=int, default=40); ap.add_argument('--match', default=None)
    a = ap.parse_args()
    keys, names = load_syms(a.exe)
    mods = {m: exports(os.path.join(r'C:\Windows\System32', m)) for m in ('ntdll.dll', 'ucrtbase.dll')}
    mkeys = {m: [e[0] for e in x] for m, x in mods.items()}

    def name(mod, off):
        if mod == 'exe':
            i = bisect.bisect_right(keys, BASE + off) - 1
            return short(names[i]) if i >= 0 else 'exe?'
        if mod in mods:
            i = bisect.bisect_right(mkeys[mod], off) - 1
            return mod[:5] + ':' + mods[mod][i][1]
        return mod

    def back_thread(path):
        best = None
        for block in re.split(r'(?m)^(?=thread )', open(path, errors='replace').read())[1:]:
            h = re.match(r"thread \d+ '[^']*': (\d+)% of a core, (\d+) samples", block)
            hits = collections.Counter()
            for n, mod, off in re.findall(r'(?m)^\s+(\d+) (\S+)\+([0-9a-f]+)', block):
                hits[name(mod, int(off, 16))] += int(n)
            if hits[BACK] and (best is None or hits[BACK] > best[0][BACK]):
                best = (hits, int(h.group(1)), int(h.group(2)))
        return best

    arms = collections.OrderedDict()
    for s in a.samples:
        arm, rest = s.split('=', 1); path, fps = rest.rsplit('@', 1)
        hits, pct, n = back_thread(path)
        print('%-4s %s: %d%% of a core, %d of %d samples named, %s FPS' % (arm, os.path.basename(path), pct, sum(hits.values()), n, fps))
        arms.setdefault(arm, []).append({k: v / n * 1000.0 / float(fps) for k, v in hits.items()})
    if len(arms) != 2: sys.exit('two arms, please')
    (na, xs), (nb, ys) = arms.items()
    rows = []
    for k in set(k for r in xs + ys for k in r):
        x = [r.get(k, 0) for r in xs]; y = [r.get(k, 0) for r in ys]
        rows.append((statistics.mean(x) - statistics.mean(y), k, x, y))
    line = lambda d, k, x, y: '%+6.2f  %5.2f %5.2f   %.2f-%.2f  %.2f-%.2f  %s' % (
        d, statistics.mean(x), statistics.mean(y), min(x), max(x), min(y), max(y), k)
    print('\nms a frame: %s-%s, %s, %s, range %s, range %s' % (na, nb, na, nb, na, nb))
    if a.match:
        for r in sorted((r for r in rows if re.search(a.match, r[1])), key=lambda r: r[0]): print(line(*r))
        return
    for r in sorted(rows, key=lambda r: r[0])[:a.top]: print(line(*r))
    print('... the biggest rises:')
    for r in sorted(rows, key=lambda r: -r[0])[:12]: print(line(*r))


if __name__ == '__main__':
    main()
