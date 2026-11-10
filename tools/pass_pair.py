"""Which shader drew the black pixels: python tools/pass_pair.py <run folder> [folder of the raw files]
NXRT_VK_DUMP_AT_S and NXRT_VK_TD_PASSES=1 leave, for each moment, a frame's 2560x1440 colour targets as its passes left them, and the next frame's with every pixel shader drawing its number.
For each pair: the pixels of the scene target that are black in the plain frame, by the shader number the next frame shows there. Black is relative: under 2% of the frame's mean."""
import glob, os, re, sys
from collections import defaultdict
import numpy as np


def r11(p):
    a = np.frombuffer(open(p, 'rb').read(), np.uint32).reshape(1440, 2560)
    def f(v, mb):
        m = (v & ((1 << mb) - 1)).astype(np.float64); e = (v >> mb).astype(np.int64)
        return np.where(e == 0, m / (1 << mb) * 2.0 ** -14, (1 + m / (1 << mb)) * 2.0 ** (e - 15.0))
    return f(a & 0x7ff, 6), f((a >> 11) & 0x7ff, 6), f((a >> 22) & 0x3ff, 5)


run = sys.argv[1]; raw = sys.argv[2] if len(sys.argv) > 2 else run
ids = {}
for l in open(os.path.join(run, 'game.log'), encoding='utf-8', errors='replace'):
    m = re.search(r'shader id (\d+) = (\w+)', l)
    if m: ids[int(m.group(1))] = m.group(2)
frames = defaultdict(list)
for p in sorted(glob.glob(os.path.join(raw, '*.pass-*R11G11B10F-2560x1440.raw'))):
    m = re.search(r'\.pass-(\d+)-(\d+)-c0-', p)
    frames[int(m.group(1))].append((int(m.group(2)), p))
for f in sorted(frames):
    if f + 1 not in frames: continue
    plain, idf = frames[f], frames[f + 1]
    # the scene: the plain frame's target that first has every pixel written, with the same place in the frame's list as the numbered one
    for k in range(min(len(plain), len(idf))):
        r, g, b = r11(plain[k][1])
        if (r + g + b > 0).mean() < 0.5: continue
        ir, ig, ib = r11(idf[k][1])
        n = (np.round((ir - 1) * 32).astype(int) & 31) | ((np.round((ig - 1) * 32).astype(int) & 31) << 5) | ((np.round((ib - 1) * 32).astype(int) & 31) << 10)
        ok = (ir >= 1) & (ir < 2) & (ig >= 1) & (ig < 2) & (ib >= 1) & (ib < 2)
        lum = r + g + b; black = (lum < 0.03 * lum.mean()) & ok
        print('frame %d pass %d: mean %.3f, black pixels %d of %d' % (f, plain[k][0], lum.mean() / 3, black.sum(), lum.size))
        u, c = np.unique(n[black], return_counts=True)
        all_u, all_c = np.unique(n[ok], return_counts=True); tot = dict(zip(all_u.tolist(), all_c.tolist()))
        for i in np.argsort(-c)[:12]:
            ys, xs = np.where(black & (n == u[i]))
            print('  %7d black of %7d  shader %s  around %d,%d' % (c[i], tot.get(int(u[i]), 0), ids.get(int(u[i]), '?'), xs[len(xs) // 2], ys[len(ys) // 2]))
        break
