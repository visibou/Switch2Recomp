"""racebench.py: the race replay's speed in one number set (games/MKWTest, the native GPU path).

  python tools/racebench.py NAME [--runs 1] [--skip 15] [--secs 30] [--env K=V ...] [--b-env K=V ...] [--extra k=v ...] [--b-extra k=v ...]

Each run starts the replay through gamectl (the save copy, SWR_GPU=nvn, the Vulkan backend, flight recorder off), waits for the
race intro ("sync 3 (" in the log), skips --skip seconds, then averages --secs seconds of the log's lines: presents per second,
present interval p50 / p95, submit+present ms a frame (and its CPU time), and the replay phases (NXRT_VK_PHASES=1, set here unless --no-phases). --b-env alternates
an on/off test. Prints one line per run and the averages per group."""
import argparse, os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gamectl as G
from panelcheck import games


def ctl(*a):
    return subprocess.run([sys.executable, os.path.join(HERE, 'gamectl.py')] + list(a), capture_output=True, text=True, timeout=60).stdout


def one(a, tag, env, extra=()):
    t_wait = time.time()
    while G.other_game_running() or games():
        if time.time() - t_wait > 7200: return None
        time.sleep(15)
    st = ['start', '--replay', 'rec_20261103_214452', '--seconds', '900', '--name', tag, '--env', 'NXRT_VK_FLIGHT=0', '--env', 'NXRT_VK_PHASES=%d' % (0 if a.no_phases else 1)]
    for e in env: st += ['--env', e]
    for e in extra: st += ['--extra', e]
    ctl(*st)
    logp = os.path.join(G.TEST, 'MKW.log'); t0 = time.time(); time.sleep(30); s3 = None
    while time.time() - t0 < 600 and games():
        m = re.search(r'\[\s*([\d.]+)\] input replay: sync 3 \(', open(logp, encoding='utf-8', errors='replace').read())
        if m: s3 = float(m.group(1)); break
        time.sleep(1)
    if s3 is None: ctl('quit'); return None
    lo, hi = s3 + a.skip, s3 + a.skip + a.secs
    while games():
        txt = open(logp, encoding='utf-8', errors='replace').read()
        last = re.findall(r'\[\s*([\d.]+)\]', txt[-3000:])
        if last and float(last[-1]) > hi + 6: break
        time.sleep(2)
    ctl('quit'); t1 = time.time()
    while games() and time.time() - t1 < 120: time.sleep(2)
    txt = open(logp, encoding='utf-8', errors='replace').read()
    native = 'NVN taken over' in txt and 'graphics backend vulkan' in txt
    fp = [(float(t), float(f), float(p50), float(p95), float(sp)) for t, f, p50, p95, sp in re.findall(
        r'\[\s*([\d.]+)\] \[nvn\] frame pacing up to frame \d+: ([\d.]+) presents/s; present interval p50 ([\d.]+) / p95 ([\d.]+).*?submit\+present ([\d.]+) ms', txt)]
    fp = [x for x in fp if lo <= x[0] <= hi]
    ph = [(float(t), *map(float, v)) for t, *v in re.findall(
        r'\[\s*([\d.]+)\] \[vk\] replay phases per frame: records ([\d.]+) ms \(draws ([\d.]+), of them descriptor sets ([\d.]+); dispatches ([\d.]+)\), commits inside submissions ([\d.]+)', txt)]
    ph = [x for x in ph if lo <= x[0] <= hi]
    gp = [(float(t), float(g)) for t, g in re.findall(r'\[\s*([\d.]+)\] \[vk\] GPU time up to frame \d+: ([\d.]+) ms per frame', txt)]
    gp = [x for x in gp if lo <= x[0] <= hi]
    cp = [(float(t), float(c)) for t, c in re.findall(r'\[\s*([\d.]+)\] \[nvn\] submit CPU time up to frame \d+: ([\d.]+) ms', txt)]
    cp = [x for x in cp if lo <= x[0] <= hi]
    sw = [(float(t), float(c)) for t, c in re.findall(r'\[\s*([\d.]+)\] \[nvn\] submit CPU time up to frame \d+: [\d.]+ ms .*?SyncWait ([\d.]+) ms', txt)]
    sw = [x for x in sw if lo <= x[0] <= hi]
    tt = [(float(t), float(w), float(c)) for t, w, c in re.findall(r'\[\s*([\d.]+)\] \[nvn\] translation thread up to frame \d+: ([\d.]+) ms a frame translating \(CPU ([\d.]+) ms\)', txt)]
    tt = [x for x in tt if lo <= x[0] <= hi]
    dc = [(float(t), float(a), float(b), float(v)) for t, a, b, v in re.findall(r'\[\s*([\d.]+)\] \[vk\] replay calls per frame: dynamic state groups set ([\d.]+), already set ([\d.]+) \(dyn_dedupe\); buffer views made ([\d.]+)', txt)]
    dc = [x for x in dc if lo <= x[0] <= hi]
    avg = lambda rows, i: sum(r[i] for r in rows) / len(rows) if rows else float('nan')
    return dict(native=native, fps=avg(fp, 1), p50=avg(fp, 2), p95=avg(fp, 3), submit=avg(fp, 4), rec=avg(ph, 1), draws=avg(ph, 2), wset=avg(ph, 3), disp=avg(ph, 4), split=avg(ph, 5), gpu=avg(gp, 1), cpu=avg(cp, 1), sync=avg(sw, 1), tr=avg(tt, 1), trc=avg(tt, 2), dset=avg(dc, 1), dskip=avg(dc, 2), bview=avg(dc, 3))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--runs', type=int, default=1); ap.add_argument('--skip', type=float, default=15)
    ap.add_argument('--secs', type=float, default=30); ap.add_argument('--env', action='append', default=[]); ap.add_argument('--b-env', action='append', default=[])
    ap.add_argument('--extra', action='append', default=[]); ap.add_argument('--b-extra', action='append', default=[])
    ap.add_argument('--no-phases', action='store_true', help='NXRT_VK_PHASES=0: its timers cost ~100k clock reads a frame, which inflates every number')
    a = ap.parse_args(); res = {'A': [], 'B': []}
    for i in range(a.runs):
        g = 'B' if (a.b_env or a.b_extra) and i % 2 == 1 else 'A'
        r = one(a, '%s_%d' % (a.name, i), a.env + (a.b_env if g == 'B' else []), a.extra + (a.b_extra if g == 'B' else []))
        if not r: print('%s_%d %s: no result' % (a.name, i, g), flush=True); continue
        res[g].append(r)
        print('%s_%d %s: %s fps %.1f p50 %.1f p95 %.1f ms, GPU %.1f ms, submit+present %.1f ms (CPU %.1f); records %.1f (draws %.1f, sets %.1f, dispatches %.1f), commits %.1f; SyncWait %.1f, translation thread %.1f (CPU %.1f); state calls set %.0f skipped %.0f, buffer views %.0f' % (
            a.name, i, g, 'native' if r['native'] else 'NOT NATIVE', r['fps'], r['p50'], r['p95'], r['gpu'], r['submit'], r['cpu'], r['rec'], r['draws'], r['wset'], r['disp'], r['split'], r['sync'], r['tr'], r['trc'], r['dset'], r['dskip'], r['bview']), flush=True)
    for g, rows in res.items():
        if rows:
            m = lambda k: sum(r[k] for r in rows) / len(rows)
            print('%s %s%s (%d runs): fps %.1f, p50 %.1f ms, p95 %.1f ms, GPU %.1f ms, submit+present %.1f ms (CPU %.1f), records %.1f, draws %.1f, sets %.1f, dispatches %.1f, commits %.1f, SyncWait %.1f, translation thread %.1f (CPU %.1f)' % (
                a.name, g, ' ' + ' '.join(a.b_env + a.b_extra) if g == 'B' else '', len(rows), m('fps'), m('p50'), m('p95'), m('gpu'), m('submit'), m('cpu'), m('rec'), m('draws'), m('wset'), m('disp'), m('split'), m('sync'), m('tr'), m('trc')), flush=True)


if __name__ == '__main__':
    sys.exit(main())
