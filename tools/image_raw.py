"""What the small images of a dumped frame hold (NXRT_VK_DUMP_AT_S with NXRT_VK_DUMP_RAW=1): python tools/image_raw.py <folder> [substring]
A line per image and layer: format, size, per channel min / mean / max, and how many texels are NaN, infinite or zero."""
import glob, os, re, sys
import numpy as np
want = sys.argv[2] if len(sys.argv) > 2 else ''
T = {'RGBA16SN': (np.int16, 4), 'R8SN': (np.int8, 1), 'R32UI': (np.uint32, 1), 'RGBA16F': (np.float16, 4), 'RG16F': (np.float16, 2), 'R16F': (np.float16, 1), 'RGBA32F': (np.float32, 4), 'RG32F': (np.float32, 2), 'R32F': (np.float32, 1), 'RGBA16': (np.uint16, 4), 'RG16': (np.uint16, 2), 'R16': (np.uint16, 1), 'RGBA8': (np.uint8, 4), 'RG8': (np.uint8, 2), 'R8': (np.uint8, 1)}
for p in sorted(glob.glob(os.path.join(sys.argv[1], '*.img-*.raw'))):
    if want not in p: continue
    m = re.search(r'\.img-(\d+)-ser(\d+)-(\w+)-(\d+)x(\d+)-l(\d+)\.raw', p)
    if not m: continue
    fr, ser, fmt, w, h, l = m.group(1), m.group(2), m.group(3), int(m.group(4)), int(m.group(5)), m.group(6)
    if fmt not in T: print(fr, ser, fmt, 'not read'); continue
    t, n = T[fmt]
    a = np.frombuffer(open(p, 'rb').read(), t)
    if a.size < w * h * n: continue
    a = a[:w * h * n].reshape(h, w, n).astype(np.float64)
    bad = np.isnan(a).sum(); inf = np.isinf(a).sum(); ok = a[np.isfinite(a)]
    print('%s ser%s %-8s %dx%d l%s  min %-10.4g max %-10.4g mean %-10.4g nan %-6d inf %-6d zero %.2f' % (fr, ser, fmt, w, h, l, ok.min() if ok.size else 0, ok.max() if ok.size else 0, ok.mean() if ok.size else 0, bad, inf, (a == 0).mean()))
