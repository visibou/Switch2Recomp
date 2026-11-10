"""What took the time in a run's slow frames: python tools/slow_frames.py <run folder> [from second]. Reads the "slow frame" lines of game.log and sums them by cause."""
import re, sys

r = sys.argv[1]; t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0
names = ['sparse binds', 'GPU waits', 'texture uploads', 'pipelines', 'pool copies']
tot = [0.0] * 5; n = 0; all_ms = 0.0; rest = 0.0; worst = []
for l in open(r + '/game.log', errors='replace'):
    m = re.match(r'\[\s*([\d.]+)\] \[vk\] slow frame (\d+): (\d+) ms: sparse binds (\d+) ms \((\d+)\), GPU waits (\d+) ms, texture uploads (\d+) ms, pipelines (\d+) ms, pool copies (\d+) ms', l)
    if not m or float(m.group(1)) < t0: continue
    ms = int(m.group(3)); p = [int(m.group(4)), int(m.group(6)), int(m.group(7)), int(m.group(8)), int(m.group(9))]
    n += 1; all_ms += ms
    for i in range(5): tot[i] += p[i]
    rest += ms - sum(p); worst.append((ms, m.group(1), p))
print(n, 'slow frames,', round(all_ms / 1000, 1), 's in them')
for i in range(5): print('%8.1f s  %s' % (tot[i] / 1000, names[i]))
print('%8.1f s  the rest' % (rest / 1000))
for w in sorted(worst, reverse=True)[:12]: print(w)
