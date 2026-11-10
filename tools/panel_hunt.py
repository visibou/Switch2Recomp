"""Catch a visual flash in a running replay and act on it at once: scores the frames every N frames and runs gamectl commands the first time a stretch of flashes starts.

  python tools/panel_hunt.py [--replay REC] [--name X] [--from-frame 2300] [--to-frame 4200] [--every 100] [--need 3] [--limit-s 560] [--env K=V ...] [--on "CMD" ...]

--on is a gamectl command line; each runs in order when the frames among the last --every show at least --need flashes. Prints one line per check, so the run's flash timeline is the output,
and "frozen" if the frame count stops. Writes nothing but what the commands write. The game ends with the run, at --to-frame or after --limit-s seconds."""
import argparse, os, shlex, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gamectl as G


def ctl(*a, timeout=200):
    try:
        return subprocess.run([sys.executable, os.path.join(HERE, 'gamectl.py')] + [str(x) for x in a], capture_output=True, text=True, timeout=timeout).stdout.strip()
    except subprocess.TimeoutExpired:
        return 'timed out'


def last_buffer(key):
    """the last buffer of 4 KB or more that flight/watch.bin holds for a shader"""
    import struct
    k = int(key, 16); b = open(os.path.join(G.TEST, 'flight', 'watch.bin'), 'rb').read(); i = 0; a = None
    while i + 32 <= len(b):
        f, kk, ad, s = struct.unpack_from('<4Q', b, i); i += 32 + 3 * s
        if kk == k and s >= 4096: a = ad
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--replay', default='rec_20261103_214452'); ap.add_argument('--name', default='hunt'); ap.add_argument('--from-frame', type=int, default=2300); ap.add_argument('--to-frame', type=int, default=4200)
    ap.add_argument('--every', type=int, default=100); ap.add_argument('--need', type=int, default=3); ap.add_argument('--limit-s', type=int, default=560); ap.add_argument('--env', action='append', default=[]); ap.add_argument('--extra', action='append', default=[]); ap.add_argument('--on', action='append', default=[])
    a = ap.parse_args()
    t0 = time.time()
    st = ['start', '--replay', a.replay, '--seconds', str(a.limit_s), '--name', a.name]
    for e in a.env: st += ['--env', e]
    for e in a.extra: st += ['--extra', e]
    print(ctl(*st), flush=True)
    print(ctl('wait-frame', a.from_frame, '--timeout', a.limit_s - 40, timeout=a.limit_s), flush=True)
    done = not a.on; total = 0; last = -1; same = 0
    while time.time() - t0 < a.limit_s - 10:
        s = G.status()
        if not s: print('the game has gone', flush=True); return 1
        f = int(s['frame'])
        if f >= a.to_frame: break
        if f == last:
            same += 1
            if same > 40: print('frozen at frame', f, flush=True); return 2
            time.sleep(1); continue
        last = f; same = 0
        b = ctl('flashes', min(a.every + 18, 118))
        print('%6.0f s frame %d: %s' % (time.time() - t0, f, b[:150]), flush=True)
        n = int(b.split()[0]) if b[:1].isdigit() else 0
        total += n
        if n >= a.need and not done:
            done = True
            for c in a.on:
                if c.startswith('wpage-of '):   # the page of a shader's bone buffer: its address moves from run to run
                    k = c.split()[1]; ctl('watch', 6, k); addr = last_buffer(k)
                    print('  buffer of', k, 'at', hex(addr or 0), flush=True)
                    if addr: c = 'wpage %#x' % (addr + int(os.environ.get('HUNT_PAGE_OFF', '0x200'), 0))
                print('  >', c, flush=True)
                print('  ' + ctl(*shlex.split(c)).replace('\n', '\n  '), flush=True)
        t1 = time.time()
        while time.time() - t1 < 60:
            s = G.status()
            if not s or int(s['frame']) >= f + a.every: break
            time.sleep(0.25)
    print('flashes counted (windows overlap):', total, flush=True)
    ctl('quit')
    return 0


if __name__ == '__main__':
    sys.exit(main())
