"""Which shader makes a one-frame flash: hides groups of shaders in turn while the game runs, with a window of nothing hidden after each.

  python tools/hide_test.py [--replay REC] [--from-frame 3270] [--window 62] [--minutes 25] GROUP [GROUP...]

A GROUP is shader keys joined by commas, up to seven. Every group gets a window with it hidden and a window with nothing hidden, and the number of flashes in each is printed. A run that shows
fewer than 3 flashes in the 60 frames before the first window proves nothing, so the game is closed and started again until --minutes runs out. Writes no recordings.

Read two results before believing one: the flashes come in stretches and can stop by themselves. A group counts only if the windows on both sides of it flash."""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gamectl as G


def ctl(*a):
    return subprocess.run([sys.executable, os.path.join(HERE, 'gamectl.py')] + [str(x) for x in a], capture_output=True, text=True).stdout.strip()


def running():
    return subprocess.run('tasklist | findstr /i MKW', shell=True, capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('groups', nargs='+'); ap.add_argument('--replay', default='rec_20261103_214452'); ap.add_argument('--from-frame', type=int, default=3270)
    ap.add_argument('--window', type=int, default=62); ap.add_argument('--minutes', type=float, default=25)
    a = ap.parse_args()
    end = time.time() + a.minutes * 60
    n = min(a.window - 2, 118)
    while time.time() < end:
        while running(): time.sleep(2)
        ctl('start', '--replay', a.replay, '--seconds', '560', '--name', 'hide')
        print(ctl('wait-frame', a.from_frame, '--timeout', 500), flush=True)
        if not G.status():
            print('the game did not start:', open(os.path.join(G.TEST, 'gamectl_start.log'), errors='replace').read()[-200:]); return 1
        b = ctl('flashes', 60); print('before:', b, flush=True)
        if 'in the last' not in b:
            print('frozen, next run', flush=True); subprocess.run('taskkill /F /IM MKW.exe', shell=True, capture_output=True); time.sleep(6); continue
        if int(b.split()[0]) < 3:
            print('no flashes, next run', flush=True); ctl('quit'); time.sleep(6); continue
        f = int(G.status()['frame'])
        for g in a.groups:
            for name, keys in ((g, g.split(',')), ('nothing hidden', [])):
                ctl('showall')
                if keys: ctl('hide', *keys)
                f += a.window; ctl('wait-frame', f, '--timeout', 60)
                print('%-40s to frame %d: %s' % (name[:40], f, ctl('flashes', n)), flush=True)
                f = int(G.status()['frame'])
        ctl('quit'); return 0
    print('no run with flashes before the time limit'); return 1


if __name__ == '__main__':
    sys.exit(main())
