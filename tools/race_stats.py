"""Frame times of a run's race, from its frames.csv and game.log: python tools/race_stats.py <run folder> [more folders]

The race starts at the replay's BGM_Stage sync, or --from S seconds into the run; --len S is how much of it counts (default 120). Prints average FPS, 1% low, the longest stop, stops over 200 ms, how long the window took against
the game time it covered, and the upload wait and video memory from the log."""
import argparse, csv, os, re, sys


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs', nargs='+'); ap.add_argument('--from', dest='start', type=float); ap.add_argument('--len', dest='length', type=float, default=120)
    a = ap.parse_args()
    print('%-34s %6s %6s %7s %7s %6s %7s %8s %8s' % ('run', 'frames', 'FPS', '1% low', 'worst', '>200ms', 'wait s', 'mem MB', 'p50 ms'))
    for r in a.runs:
        log = open(os.path.join(r, 'game.log'), errors='replace').read().split('\n')
        t0 = a.start
        if t0 is None:
            for l in log:
                m = re.match(r'\[\s*([\d.]+)\].*input replay: sync \d+ \(.*BGM_Stage.*\) at', l)
                if m: t0 = float(m.group(1)); break
        if t0 is None: print(os.path.basename(r), 'no race in this run'); continue
        ms = [float(x['ms']) / 1000 for x in csv.DictReader(open(os.path.join(r, 'frames.csv')))]
        t1 = t0 + a.length
        w = [t for t in ms if t0 <= t < t1]
        if len(w) < 10: print(os.path.basename(r), 'too few frames after %.0f s' % t0); continue
        dt = sorted(1000 * (y - x) for x, y in zip(w, w[1:]))
        up = 0.0; mem = 0
        for l in log:
            m = re.match(r'\[\s*([\d.]+)\] \[vk\] uploads, last 5 s:.*workers (\d+) ms', l)
            if m and t0 <= float(m.group(1)) < t1: up += int(m.group(2)) / 1000
            m = re.match(r'\[\s*([\d.]+)\] \[vk\] textures:.*device memory (\d+) of', l)
            if m and t0 <= float(m.group(1)) < t1: mem = max(mem, int(m.group(2)))
        print('%-34s %6d %6.1f %7.1f %7.0f %6d %7.1f %8d %8.1f' % (os.path.basename(r)[-34:], len(w), (len(w) - 1) / (w[-1] - w[0]), 1000 / dt[int(len(dt) * .99)], dt[-1], sum(d > 200 for d in dt), up, mem, dt[len(dt) // 2]))


if __name__ == '__main__':
    sys.exit(main())
