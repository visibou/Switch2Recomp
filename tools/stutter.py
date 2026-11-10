"""stutter.py: how often a run stutters, and why (any game; reads a run folder's frames.csv and game.log).

  python tools/stutter.py RUN_DIR [RUN_DIR ...]          run folders (games/<test>/runs/<stamp>_<name>), or
  python tools/stutter.py --last N [--test games/MKWTest] the N newest runs of a test folder

For each window it prints frames, fps, frame-time p50 / p95 / p99 / max, and the frames over 50 ms and over 100 ms per minute.
Windows come from the input replay's sync lines in game.log ("menu": sync 0 to sync 3, "race": sync 3 + 15 s to the end); without
them the whole run after its first 30 s is one window. The causes of the slow frames come from the runtime's "slow frame" lines
(NXRT_VK_SLOW_MS, default 50): sparse binds, GPU waits, texture uploads, pipeline builds, pool copies, and the rest unexplained.
A frame-time here is the present-to-present interval the game saw (frames.csv)."""
import argparse, csv, glob, os, re, sys

CAUSES = ('sparse binds', 'GPU waits', 'texture uploads', 'pipelines', 'pool copies')


def pct(v, q):
    if not v: return float('nan')
    v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]


def windows(log):
    sync = {int(n): float(t) for t, n in re.findall(r'\[\s*([\d.]+)\] input replay: sync (\d+) \(', log)}
    end = max([float(t) for t in re.findall(r'(?m)^\[\s*([\d.]+)\]', log[-20000:])] or [0])
    if 0 in sync and 3 in sync: return [('menu', sync[0] + 5, sync[3]), ('race', sync[3] + 15, end)]
    return [('all', 30, end)]


def one(run):
    fp, lp = os.path.join(run, 'frames.csv'), os.path.join(run, 'game.log')
    if not os.path.isfile(fp) or not os.path.isfile(lp): return None
    t = [float(r['ms']) / 1000 for r in csv.DictReader(open(fp))]
    log = open(lp, encoding='utf-8', errors='replace').read()
    slow = [(float(ts), float(ms), None, rest)
            for ts, ms, rest in re.findall(r'\[\s*([\d.]+)\] \[vk\] slow frame \d+: ([\d.]+) ms: (.*)', log)]
    out = []
    for name, lo, hi in windows(log):
        iv = [(b, (b - a) * 1000) for a, b in zip(t, t[1:]) if lo <= b <= hi]
        if len(iv) < 30: continue
        d = [x for _, x in iv]; mins = (iv[-1][0] - iv[0][0]) / 60 or 1
        cause = dict.fromkeys(CAUSES + ('unexplained',), 0.0); nslow = 0
        for ts, ms, _, rest in slow:
            if not (lo <= ts <= hi): continue
            nslow += 1; acc = 0
            for c in CAUSES:
                m = re.search(re.escape(c) + r' ([\d.]+) ms', rest)
                if m: cause[c] += float(m.group(1)); acc += float(m.group(1))
            cause['unexplained'] += max(0.0, ms - acc)
        back = sum(int(n) for ts, n in re.findall(r'\[\s*([\d.]+)\] \[vk\] uploads, last 5 s:.*?; (\d+) textures back after eviction', log) if lo <= float(ts) <= hi)
        upms = sum(float(m) for ts, m in re.findall(r'\[\s*([\d.]+)\] \[vk\] textures: .*? ([\d.]+) ms in upload', log) if lo <= float(ts) <= hi)
        out.append(dict(back=back / mins, upms=upms / mins, run=os.path.basename(run), win=name, frames=len(d), fps=len(d) / (mins * 60), p50=pct(d, .5), p95=pct(d, .95), p99=pct(d, .99), mx=max(d),
                        s50=sum(x > 50 for x in d) / mins, s100=sum(x > 100 for x in d) / mins, nslow=nslow, cause=cause, mins=mins))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs', nargs='*'); ap.add_argument('--last', type=int, default=0); ap.add_argument('--test', default='games/MKWTest')
    ap.add_argument('--match', default='', help='only runs whose folder name contains this')
    a = ap.parse_args()
    runs = list(a.runs)
    if a.last:
        all_ = sorted((d for d in glob.glob(os.path.join(a.test, 'runs', '*')) if os.path.isdir(d) and a.match in os.path.basename(d)), key=os.path.getmtime)
        runs += all_[-a.last:]
    agg = {}
    for r in runs:
        for w in one(r) or []:
            c = ', '.join('%s %.0f' % (k, v / w['mins']) for k, v in w['cause'].items() if v >= 1)
            print('%-34s %-5s %5d frames %5.1f fps  p50 %5.1f p95 %5.1f p99 %6.1f max %7.1f ms  >50ms %5.1f/min >100ms %5.1f/min  reloads %4.0f/min uploads %5.0f ms/min  slow-frame ms/min: %s' % (
                w['run'][:34], w['win'], w['frames'], w['fps'], w['p50'], w['p95'], w['p99'], w['mx'], w['s50'], w['s100'], w['back'], w['upms'], c or '-'))
            agg.setdefault(w['win'], []).append(w)
    for win, ws in agg.items():
        if len(ws) < 2: continue
        m = lambda k: sum(x[k] for x in ws) / len(ws)
        print('mean of %d %-5s: %.1f fps  p50 %.1f p95 %.1f p99 %.1f max %.1f ms  >50ms %.1f/min >100ms %.1f/min  reloads %.0f/min uploads %.0f ms/min' % (len(ws), win, m('fps'), m('p50'), m('p95'), m('p99'), m('mx'), m('s50'), m('s100'), m('back'), m('upms')))


if __name__ == '__main__':
    sys.exit(main())
