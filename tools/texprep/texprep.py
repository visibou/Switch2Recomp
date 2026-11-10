"""texprep.py: convert every texture of a game into the converted-texture pack before playing, so new places never convert
textures while you play (they did: the pack filled up, and every new track converted thousands of textures each session).

  python tools/texprep/texprep.py --romfs "D:/path/to/romfs" [--pack games/MKWTest/astc_cache] [--jobs 12] [--limit N]

Reads every .bntx / .bntx.zs file (BNTX: NintendoWare's texture container), and for each ASTC texture converts each level the
runtime keeps in the pack (128x128 and up) exactly as the runtime would (texprep.dll is built from the runtime's own code).
Levels already in the pack are skipped, so it can be stopped and run again. Close the game first (the pack is opened for writing).
Stops if the drive would drop below --keep-free GB. Build the DLL first: python tools/texprep/build.py. Nothing here knows a game."""
import argparse, ctypes, os, shutil, sys, threading, time, concurrent.futures as cf
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bntx

try:
    import zstandard
except ImportError:
    sys.exit('Python package "zstandard" is missing: pip install zstandard')


def load_dll(pack):
    dll = ctypes.CDLL(os.path.join(HERE, 'texprep.dll'))
    dll.tp_set_pack_dir.argtypes = [ctypes.c_char_p]
    dll.tp_texture.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int,
                               ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint64)]
    dll.tp_texture.restype = ctypes.c_int
    dll.tp_pack_bytes.restype = ctypes.c_uint64
    dll.tp_set_min.argtypes = [ctypes.c_uint32]
    os.makedirs(pack, exist_ok=True)
    dll.tp_set_pack_dir(os.path.abspath(pack).encode('utf-8'))
    if not dll.tp_ready(): sys.exit('Cannot open the pack in %s: is the game running? Close it and try again.' % pack)
    return dll


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--romfs', required=True)
    ap.add_argument('--pack', default=os.path.join(HERE, '..', '..', 'games', 'MKWTest', 'astc_cache'))
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument('--limit', type=int, default=0, help='only the first N files (to try it)')
    ap.add_argument('--keep-free', type=float, default=20.0, help='stop when the drive would have less than this many GB free')
    ap.add_argument('--min', type=int, default=128, help='smallest level edge the runtime keeps in the pack (NXRT_VK_ASYNC_MIN)')
    a = ap.parse_args()

    files = []
    for r, _, fs in os.walk(a.romfs):
        for n in fs:
            if n.endswith('.bntx.zs') or n.endswith('.bntx'): files.append(os.path.join(r, n))
    files.sort()
    if a.limit: files = files[:a.limit]
    if not files: sys.exit('No .bntx files under %s' % a.romfs)
    dll = load_dll(a.pack); dll.tp_set_min(a.min)
    start_bytes = dll.tp_pack_bytes()
    print('%d texture files; pack %s holds %.1f GB; %d threads' % (len(files), os.path.abspath(a.pack), start_bytes / 2**30, a.jobs), flush=True)

    lk = threading.Lock(); st = dict(files=0, tex=0, wrote=0, have=0, skip=0, raw=0, bad=0); stop = threading.Event()
    def one(path):
        if stop.is_set(): return
        try:
            with open(path, 'rb') as f: d = f.read()
            if path.endswith('.zs'): d = zstandard.ZstdDecompressor().stream_reader(d).read()
        except Exception:
            with lk: st['bad'] += 1; st['files'] += 1
            return
        w = h = s = 0
        for t in bntx.textures(d):
            f = bntx.nvn_format(t)
            if f is None or not t['data'] or t['depth'] > 1: continue
            have = ctypes.c_int(0); skip = ctypes.c_int(0); by = ctypes.c_uint64(0)
            r = dll.tp_texture(t['data'], len(t['data']), f, t['w'], t['h'], max(t['mips'], 1), max(t['layers'], 1), (t['lay0'] & 0xff) + 1,
                               ctypes.byref(have), ctypes.byref(skip), ctypes.byref(by))
            w += max(r, 0); h += have.value; s += skip.value
        with lk:
            st['files'] += 1; st['wrote'] += w; st['have'] += h; st['skip'] += s

    t0 = time.time(); last = 0
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(one, p) for p in files]
        while True:
            done = sum(f.done() for f in futs)
            now = time.time()
            if now - last >= 10 or done == len(futs):
                last = now; el = now - t0; gb = (dll.tp_pack_bytes() - start_bytes) / 2**30
                free = shutil.disk_usage(os.path.abspath(a.pack)).free / 2**30
                eta = el / done * (len(futs) - done) if done else 0
                print('%5.1f%%  %d/%d files  %d levels converted, %d already there  +%.2f GB  %.1f GB free  %s left' % (
                    100.0 * done / len(futs), done, len(futs), st['wrote'], st['have'], gb, free, time.strftime('%H:%M:%S', time.gmtime(eta))), flush=True)
                if free < a.keep_free and not stop.is_set():
                    print('Stopping: less than %.0f GB would be left on the drive.' % a.keep_free, flush=True); stop.set()
            if done == len(futs): break
            time.sleep(1)
    print('Done in %s: %d levels converted, %d were already there, %d skipped, %d unreadable files; pack now %.1f GB.' % (
        time.strftime('%H:%M:%S', time.gmtime(time.time() - t0)), st['wrote'], st['have'], st['skip'], st['bad'], dll.tp_pack_bytes() / 2**30), flush=True)


if __name__ == '__main__':
    main()
