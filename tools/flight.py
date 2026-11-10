"""Read a flight recording (F5 or gamectl flight; <exe folder>/flight/<time>/).

  python tools/flight.py <folder>                  what blinked, from the draws of each frame and from the pictures
  python tools/flight.py <folder> --sheet F0 F1    a contact sheet of frames F0..F1, sheet_F0_F1.png in the folder
  python tools/flight.py <folder> --png F          frame F as frame_F.png
  python tools/flight.py <folder> --draws F [KEY]  the rows of frame F (of one shader)
  python tools/flight.py <folder> --counts         indirect draws whose count or record the CPU saw differently from the GPU
  python tools/flight.py [--latest DIR]            as the first, on the newest recording under DIR/flight (default games/MKWTest)

With --a/--b frame numbers the two lists of draws are compared. A shader is "blinking" when its draws change between frames and come back: drawn, gone, drawn."""
import argparse, glob, os, sys
from collections import defaultdict
import numpy as np

ROW = np.dtype([('vs', '<u8'), ('ps', '<u8'), ('pass', '<u4'), ('flags', '<u4'), ('rt', '<u4'), ('mdi_i', '<u4'), ('mdi_cnt', '<u4'), ('binst', '<u4'), ('args_i', '<u4'), ('pad', '<u4'),
                ('a', '<u4', 5), ('pad2', '<u4'), ('g', '<u4', 6)])
DRAWN, INDEXED, INDIRECT, MDI, MESH, RC_SKIP, HIDDEN, MDI_CUT, DISPATCH = 1, 2, 4, 8, 16, 32, 64, 128, 256


class Rec:
    def __init__(s, d):
        s.d = d; s.imgs = []; s.frames = []
        for l in open(os.path.join(d, 'index.txt')):
            p = l.split()
            if p[0] == 'image': s.w, s.h, s.fmt = int(p[1]), int(p[2]), int(p[4]); assert int(p[6]) == ROW.itemsize, 'row size'
            elif p[0] == 'i': s.imgs.append((int(p[2]), int(p[3])))
            elif p[0] == 'f': s.frames.append((int(p[1]), int(p[2]), int(p[3]), int(p[4])))
        s.rows = np.fromfile(os.path.join(d, 'rows.bin'), ROW)
        s.by = {f: s.rows[a:a + n] for f, a, n, _ in s.frames}
        s.img_index = {f: i for i, (f, _) in enumerate(s.imgs)}

    def image(s, f):
        n = s.w * s.h * 4
        with open(os.path.join(s.d, 'frames.rgba'), 'rb') as h:
            h.seek(s.img_index[f] * n); a = np.frombuffer(h.read(n), np.uint8).reshape(s.h, s.w, 4)
        if s.fmt in (44, 50): a = a[:, :, [2, 1, 0, 3]]   # B8G8R8A8
        return a[:, :, :3]


def png(a, path):
    from PIL import Image
    Image.fromarray(a).save(path)


def blinks(seq):
    """times a sequence of per-frame values left a value and came back to it"""
    n = 0
    for i in range(1, len(seq) - 1):
        if seq[i] != seq[i - 1] and seq[i + 1] == seq[i - 1]: n += 1
    return n


def report(r):
    fs = [f for f, _, _, _ in r.frames if f in r.img_index]
    print('%s\n%d frames with pictures, %d with draws, frames %d..%d, %.1f FPS' % (r.d, len(fs), len(r.frames), r.frames[0][0], r.frames[-1][0], (len(r.frames) - 1) / max(1e-9, (r.frames[-1][3] - r.frames[0][3]) / 1e9)))
    # the draws of each program, frame by frame
    drawn = defaultdict(lambda: defaultdict(int)); inst = defaultdict(lambda: defaultdict(int)); skip = defaultdict(lambda: defaultdict(int))
    fl = [f for f, _, _, _ in r.frames]
    for f in fl:
        x = r.by[f]; x = x[(x['flags'] & DISPATCH) == 0]
        for vs, ps, fg, a in zip(x['vs'].tolist(), x['ps'].tolist(), x['flags'].tolist(), x['a'][:, 1].tolist()):
            k = (vs, ps)
            if fg & DRAWN: drawn[k][f] += 1; inst[k][f] += a if a else 1
            else: skip[k][f] += 1
    rows = []
    for k in set(drawn) | set(skip):
        d = [drawn[k].get(f, 0) for f in fl]; i = [inst[k].get(f, 0) for f in fl]; s = [skip[k].get(f, 0) for f in fl]
        on = [1 if v else 0 for v in d]
        rows.append((blinks(on), blinks(d), blinks(i), sum(on), k, sum(s), max(d)))
    print('\nprograms that came and went between frames (frames with draws of %d):' % len(fl))
    print('  on/off  draws  instances  frames  skipped  max   vertex shader    pixel shader')
    for b, bd, bi, n, k, sk, mx in sorted(rows, key=lambda x: (-x[0], -x[1], -x[2]))[:30]:
        if b + bd + bi == 0: continue
        print('  %6d %6d %10d %7d %8d %4d   %016x %016x' % (b, bd, bi, n, sk, mx, k[0], k[1]))
    # the pictures: a region that is different for one frame and returns
    print('\nframes where part of the picture changed for one frame and came back:')
    prev2 = prev = None; hits = []; S = 8
    for f in fs:
        a = r.image(f).astype(np.int16)
        a = a[: a.shape[0] // S * S, : a.shape[1] // S * S]
        a = a.reshape(a.shape[0] // S, S, a.shape[1] // S, S, 3).mean(axis=(1, 3))
        if prev2 is not None:
            big = (np.abs(prev - prev2).max(axis=2) > 24) & (np.abs(prev - a).max(axis=2) > 24) & (np.abs(a - prev2).max(axis=2) < 10)
            n = int(big.sum())
            if n: ys, xs = np.where(big); hits.append((n, f - 1, int(xs.mean()) * S, int(ys.mean()) * S))
        prev2, prev = prev, a
    for n, f, x, y in sorted(hits, reverse=True)[:12]: print('  frame %d: %d blocks of 8x8, around %d,%d' % (f, n, x, y))
    if not hits: print('  none')
    counts(r, 8)


def counts(r, top=40):
    x = r.rows[(r.rows['args_i'] != 0)]
    print('\nindirect draws read on the GPU: %d' % len(x))
    if not len(x): return
    mdi = x[(x['flags'] & MDI) != 0]
    if len(mdi):
        wrong = mdi[mdi['mdi_cnt'] != mdi['g'][:, 5]]
        cut = mdi[((mdi['flags'] & MDI_CUT) != 0) & (mdi['mdi_i'] < mdi['g'][:, 5])]
        extra = mdi[((mdi['flags'] & DRAWN) != 0) & (mdi['mdi_i'] >= mdi['g'][:, 5])]
        print('  multi-draw rows %d: count differs from the GPU in %d, dropped although the GPU count has them %d, made although the GPU count ends before them %d' % (len(mdi), len(wrong), len(cut), len(extra)))
    n = 5
    diff = (x['a'][:, :n] != x['g'][:, :n]).any(axis=1) & ((x['flags'] & INDEXED) != 0) | (x['a'][:, :4] != x['g'][:, :4]).any(axis=1) & ((x['flags'] & INDEXED) == 0)
    print('  records that differ between what the CPU read and what the GPU read: %d' % int(diff.sum()))
    binst = x[((x['flags'] & INDEXED) != 0) & (x['a'][:, 4] != x['g'][:, 4]) | ((x['flags'] & INDEXED) == 0) & (x['a'][:, 3] != x['g'][:, 3])]
    print('  of them with another base instance: %d' % len(binst))
    x['g'][(x['flags'] & MDI) == 0, 5] = 0
    for v in x[diff][:top]:
        print('    vs %016x ps %016x i %d cpu %s gpu %s count cpu %d gpu %d' % (v['vs'], v['ps'], v['mdi_i'], v['a'].tolist(), v['g'][:5].tolist(), v['mdi_cnt'], v['g'][5]))


def sheet(r, f0, f1):
    from PIL import Image
    fs = [f for f, _ in r.imgs if f0 <= f <= f1][:24]
    W, H = 426, 240; cols = 4; rows = (len(fs) + cols - 1) // cols
    s = Image.new('RGB', (cols * W, rows * H))
    for i, f in enumerate(fs): s.paste(Image.fromarray(r.image(f)).resize((W, H)), (i % cols * W, i // cols * H))
    p = os.path.join(r.d, 'sheet_%d_%d.png' % (f0, f1)); s.save(p); print(p)


def draws(r, f, key):
    x = r.by[f]
    if key:
        k = int(key, 16); x = x[(x['vs'] == k) | (x['ps'] == k)]
    names = ((DRAWN, 'drawn'), (INDEXED, 'idx'), (INDIRECT, 'ind'), (MDI, 'mdi'), (MESH, 'mesh'), (RC_SKIP, 'rc-skip'), (HIDDEN, 'hidden'), (MDI_CUT, 'cut'), (DISPATCH, 'dispatch'))
    for v in x:
        fl = ','.join(n for b, n in names if v['flags'] & b)
        print('pass %5d rt %6d vs %016x ps %016x %-22s a %s g %s i %d cnt %d binst %d' % (v['pass'], v['rt'], v['vs'], v['ps'], fl, v['a'].tolist(), v['g'].tolist() if v['args_i'] else '-', v['mdi_i'], v['mdi_cnt'], v['binst']))


def pv_load(d):
    L = open(os.path.join(d, 'tiles.txt')).read().splitlines(); _, w, h = L[0].split(); w = int(w); h = int(h)
    T = []
    for i, l in enumerate(L[1:]):
        p = l.split(); T.append(dict(i=i, frame=int(p[0]), n=int(p[1]), compute=int(p[2]), slot=int(p[3]), ser=int(p[4]), key=p[5], size=p[6], level=int(p[7]), layer=int(p[8]), fmt=p[9]))
    a = np.memmap(os.path.join(d, 'tiles.rgba'), np.uint8, 'r').reshape(-1, h, w, 4)
    return T, a, w, h


def pv_sheet(d, frame):
    """every thumbnail of a frame"""
    from PIL import Image, ImageDraw
    T, a, w, h = pv_load(d); t = [x for x in T if x['frame'] == frame]
    cols = 12; rows = (len(t) + cols - 1) // cols
    s = Image.new('RGB', (cols * w, rows * (h + 11)), (30, 30, 30)); dr = ImageDraw.Draw(s)
    for k, x in enumerate(t):
        s.paste(Image.fromarray(a[x['i']][:, :, :3]), (k % cols * w, k // cols * (h + 11) + 11))
        dr.text((k % cols * w + 2, k // cols * (h + 11)), '%d %s%s %s' % (x['n'], 'C ' if x['compute'] else '', x['fmt'][:9], x['key'][:6]), fill=(255, 255, 0))
    p = os.path.join(d, 'frame_%d.png' % frame); s.save(p); print(p)


def pv_diff(d, f0, f1):
    """the two frames, thumbnail by thumbnail in order: where do they part, compared with how much two good frames differ"""
    from PIL import Image
    T, a, w, h = pv_load(d); x0 = [x for x in T if x['frame'] == f0]; x1 = [x for x in T if x['frame'] == f1]
    print(len(x0), len(x1), 'thumbnails')
    out = []
    for p, q in zip(x0, x1):
        e = float(np.abs(a[p['i']][:, :, :3].astype(np.int16) - a[q['i']][:, :, :3].astype(np.int16)).mean())
        same = p['fmt'] == q['fmt'] and p['size'] == q['size'] and p['key'] == q['key']
        out.append((e, p, q, same))
        print('%4d %s %-12s %-10s %s l%d ser %d/%d  diff %.1f%s' % (p['n'], 'C' if p['compute'] else ' ', p['fmt'][:12], p['size'], p['key'][:8], p['layer'], p['ser'], q['ser'], e, '' if same else '  (different pass)'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder', nargs='?'); ap.add_argument('--latest', default=os.path.join('games', 'MKWTest'))
    ap.add_argument('--pv', type=int); ap.add_argument('--pvdiff', nargs=2, type=int); ap.add_argument('--sheet', nargs=2, type=int); ap.add_argument('--png', type=int); ap.add_argument('--draws', nargs='+'); ap.add_argument('--counts', action='store_true')
    a = ap.parse_args()
    d = a.folder or sorted(glob.glob(os.path.join(a.latest, 'flight', '*')))[-1]
    if a.pv is not None: pv_sheet(d, a.pv); return 0
    if a.pvdiff: pv_diff(d, *a.pvdiff); return 0
    r = Rec(d)
    if a.sheet: sheet(r, *a.sheet)
    elif a.png is not None: png(r.image(a.png), os.path.join(d, 'frame_%d.png' % a.png)); print(os.path.join(d, 'frame_%d.png' % a.png))
    elif a.draws: draws(r, int(a.draws[0]), a.draws[1] if len(a.draws) > 1 else None)
    elif a.counts: counts(r)
    else: report(r)
    return 0


if __name__ == '__main__':
    sys.exit(main())
