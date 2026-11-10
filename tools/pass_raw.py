"""The colour targets as each pass of the dump frame left them (NXRT_VK_TD_PASSES=1 with NXRT_VK_TEXDUMP=<frame>): python tools/pass_raw.py <folder> [substring] [out folder]
Prints the mean, maximum and share of non-zero pixels of each file; with an out folder, writes PNGs scaled by the file's own 99th percentile."""
import glob, os, sys
import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None
def rgb(p, fmt, w, h):
    b = open(p, 'rb').read()
    if fmt == 'R11G11B10F':
        a = np.frombuffer(b, np.uint32).reshape(h, w)
        def f(v, mb, eb):
            m = (v & ((1 << mb) - 1)).astype(np.float64); e = (v >> mb).astype(np.int64)
            return np.where(e == 0, m / (1 << mb) * 2.0 ** -14, (1 + m / (1 << mb)) * 2.0 ** (e - 15.0))
        return np.stack([f(a & 0x7ff, 6, 5), f((a >> 11) & 0x7ff, 6, 5), f((a >> 22) & 0x3ff, 5, 5)], -1)
    if fmt in ('RGBA16F', 'RG16F', 'R16F', 'RGBA32F', 'RG32F', 'R32F'):
        n = {'R': 1, 'G': 2, 'B': 4}[fmt[:-3][-1] if False else ('B' if fmt.startswith('RGBA') else 'G' if fmt.startswith('RG') else 'R')]
        t = np.float16 if '16' in fmt else np.float32
        a = np.frombuffer(b, t).reshape(h, w, n).astype(np.float64)
        return a
    n = len(b) // (w * h)
    return np.frombuffer(b, np.uint8).reshape(h, w, n).astype(np.float64) / 255
want = sys.argv[2] if len(sys.argv) > 2 else ''
out = sys.argv[3] if len(sys.argv) > 3 else None
for p in sorted(glob.glob(os.path.join(sys.argv[1], '*.pass-*.raw'))):
    if want not in p: continue
    parts = os.path.basename(p)[:-4].split('.pass-')[1].split('-')
    n, slot, ser, fmt, size = parts[1], parts[2], parts[3], parts[4], parts[5]
    w, h = map(int, size.split('x'))
    a = rgb(p, fmt, w, h)
    c = a[..., :3]
    print('%s %s %s %-12s %s mean %s max %s nonzero %.3f' % (n, slot, ser, fmt, size, np.round(c.mean((0, 1)), 4), np.round(c.max((0, 1)), 3), (c.sum(-1) > 0).mean()))
    if out and Image:
        os.makedirs(out, exist_ok=True)
        s = np.percentile(c, 99) or 1
        v = np.clip(c / s, 0, 1) ** (1 / 2.2)
        if v.shape[2] < 3: v = np.concatenate([v, np.zeros(v.shape[:2] + (3 - v.shape[2],))], -1)
        Image.fromarray((v * 255).astype(np.uint8)).save(os.path.join(out, '%s-%s-%s.png' % (n, ser, fmt)))
