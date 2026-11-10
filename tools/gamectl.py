"""Talk to the running game (runtime/nvn/vk/vk_ctl.c), or start one to talk to.

  python tools/gamectl.py start [--seconds N] [--replay REC] [--name X] [--rdoc] [--env K=V ...] [--extra key=value ...] [--script ms:hex,...]
  python tools/gamectl.py wait-frame N [--timeout S]    wait until the game reaches a frame
  python tools/gamectl.py status
  python tools/gamectl.py flight [label]                save the last frames; prints the folder (read it with tools/flight.py)
  python tools/gamectl.py flashes [N] [BLOCKS]          frames among the last N (up to 118) with a one-frame change over more than BLOCKS 8x8 blocks (default 60; panels far away are under 100); writes nothing
  python tools/gamectl.py pause | resume | step [N]     hold the game at a present, or advance it N frames
  python tools/gamectl.py hide KEY... | show KEY... | showall | hidden     shaders whose draws are skipped
  python tools/gamectl.py set NAME VALUE | get NAME | switches             switches the renderer reads while it runs
  python tools/gamectl.py shot [PATH] | gputime | passdump [N] | texdump | rawdump FRAMES SERIAL... | passview [FRAMES] | drawstats [FRAMES] | watch FRAMES KEY... | peek ADDR [WORDS] | wpage ADDR | wtrace ADDR LEN [huge] | bp RVA [N] | rdoc | quit

start runs games/MKWTest (a copy of the game beside the real one) on the saved-game copy, never the real save. A replay brings its own save."""
import argparse, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = r'\\.\pipe\swr_ctl'
TEST = os.path.join(ROOT, 'games', 'MKWTest')


_other = [0.0, False]
def other_game_running():
    """A MKW.exe from outside games/MKWTest (the user playing their install): it would own the control pipe. Cached 3 s."""
    if time.time() - _other[0] < 3: return _other[1]
    ps = subprocess.run(['powershell', '-NoProfile', '-Command', "Get-Process -Name 'MKW' -ErrorAction SilentlyContinue | ForEach-Object { $_.Path }"],
                        capture_output=True, text=True).stdout
    test = os.path.normcase(os.path.abspath(TEST))
    _other[0] = time.time(); _other[1] = any(p.strip() and os.path.normcase(os.path.dirname(os.path.abspath(p.strip()))) != test for p in ps.splitlines())
    return _other[1]


def send(line, timeout=5.0):
    """One command, one reply. None when no game is listening, or when the user's own game is running (never send it anything)."""
    if other_game_running(): return None
    t0 = time.time()
    while True:
        try:
            f = open(PIPE, 'r+b', buffering=0)
            break
        except OSError:
            if time.time() - t0 > timeout: return None
            time.sleep(0.05)
    try:
        f.write((line + '\n').encode())
        out = b''
        while True:
            try:
                b = f.read(65536)
            except OSError:
                break
            if not b: break
            out += b
        return out.decode('utf8', 'replace')
    finally:
        try: f.close()
        except OSError: pass


def status():
    r = send('status', 0.2)
    return dict(l.split(' ', 1) for l in r.splitlines() if ' ' in l) if r else None


def start(a):
    env = dict(os.environ, SWR_GPU='nvn', NXRT_GFX='vulkan', SWR_INPUT_SYNC_WAIT_S='900')
    if not a.replay: env['SWR_SAVE_DIR'] = os.path.join(TEST, 'save_copy')
    if a.rdoc: env['SWR_RDOC'] = '1'
    for kv in a.env:
        k, v = kv.split('=', 1); env[k] = v
    # The default pipeline cache sits under MKW_nvn, which is a link into the installed game's folder: test runs keep their own in games/MKWTest/vk_test
    env.setdefault('NXRT_VK_CACHE_DIR', os.path.join(TEST, 'vk_test'))
    cmd = [sys.executable, os.path.join(ROOT, 'tools', 'run_game.py'), '--game', os.path.join(TEST, 'MKW.exe'), '--name', a.name, '--seconds', str(a.seconds)]
    for e in a.extra: cmd += ['--extra', e]
    if a.script: cmd += ['--script', a.script]
    if a.replay: cmd += ['--replay', a.replay if os.path.isabs(a.replay) else os.path.join(ROOT, 'games', 'MKW', 'recordings', a.replay)]
    out = open(os.path.join(TEST, 'gamectl_start.log'), 'w')
    subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=out, stderr=out, creationflags=0x00000008 | 0x00000200)   # detached, its own process group
    print('started; wait-frame tells when it is there')


def main():
    if len(sys.argv) < 2: print(__doc__); return 1
    c = sys.argv[1]
    if c == 'start':
        ap = argparse.ArgumentParser()
        ap.add_argument('--seconds', type=int, default=1800); ap.add_argument('--replay', default=''); ap.add_argument('--name', default='ctl')
        ap.add_argument('--rdoc', action='store_true'); ap.add_argument('--env', action='append', default=[]); ap.add_argument('--extra', action='append', default=[])
        ap.add_argument('--script', default='', help='timed buttons "ms:hex,..." held 300 ms each (run_game.py --script)')
        start(ap.parse_args(sys.argv[2:])); return 0
    if c == 'wait-frame':
        ap = argparse.ArgumentParser(); ap.add_argument('frame', type=int); ap.add_argument('--timeout', type=float, default=300)
        a = ap.parse_args(sys.argv[2:]); t0 = time.time()
        while time.time() - t0 < a.timeout:
            s = status()
            if s and int(s['frame']) >= a.frame: print('frame', s['frame'], 'after %.0f s' % (time.time() - t0)); return 0
            time.sleep(0.5)
        print('timed out', status()); return 1
    r = send(' '.join(sys.argv[1:]), 3.0)
    if r is None: print('no game is running with the control pipe'); return 1
    sys.stdout.write(r)
    return 0


if __name__ == '__main__':
    sys.exit(main())
