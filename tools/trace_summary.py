"""Summarize a trace.json (runtime/trace.c, SWR_TRACE): per thread, the time in each span name (share of the window,
count, mean), the host GPU's busy and idle time, and frame times from the presents.

  python tools/trace_summary.py <trace.json> [--top 14]
"""
import argparse, collections, json


def union(intervals):
    """total length of the union of (start, end) intervals"""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None: total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None: total += cur_e - cur_s
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace')
    ap.add_argument('--top', type=int, default=14)
    a = ap.parse_args()
    ev = json.load(open(a.trace))['traceEvents']
    names = {e.get('tid'): e['args']['name'] for e in ev if e.get('ph') == 'M' and e['name'] == 'thread_name'}
    xs = [e for e in ev if e.get('ph') == 'X']
    cpu = [e for e in xs if e['tid'] < 900000]
    t0 = min(e['ts'] for e in cpu); t1 = max(e['ts'] + e['dur'] for e in cpu)
    win = t1 - t0
    print('window %.0f ms' % (win / 1000))
    by_thread = collections.defaultdict(list)
    for e in xs: by_thread[e['tid']].append(e)
    for tid in sorted(by_thread, key=lambda t: (t >= 900000, names.get(t, str(t)))):
        es = by_thread[tid]
        tot = collections.Counter(); cnt = collections.Counter()
        for e in es:
            k = e['name'].split(' ')[0] if tid >= 900000 else e['name']
            if tid < 900000 and k.startswith('present wait'): k = 'present wait'
            tot[k] += e['dur']; cnt[k] += 1
        covered = union([(e['ts'], e['ts'] + e['dur']) for e in es])
        print('\n== %s: spans cover %.0f%% of the window' % (names.get(tid, 'thread %d' % tid), 100.0 * covered / win))
        for k, v in tot.most_common(a.top):
            print('   %-40s %5.1f%%  %7d x  %8.3f ms avg' % (k[:40], 100.0 * v / win, cnt[k], v / cnt[k] / 1000))
    gpu = [e for e in xs if e['tid'] == 900000]
    if gpu:
        busy = union([(e['ts'], e['ts'] + e['dur']) for e in gpu])
        g0 = min(e['ts'] for e in gpu); g1 = max(e['ts'] + e['dur'] for e in gpu)
        print('\n== host GPU: busy %.0f%% of %.0f ms (submissions)' % (100.0 * busy / (g1 - g0), (g1 - g0) / 1000))
        gaps = []; last = None
        for e in sorted(gpu, key=lambda e: e['ts']):
            if last is not None and e['ts'] > last: gaps.append(e['ts'] - last)
            last = max(last or 0, e['ts'] + e['dur'])
        if gaps:
            gaps.sort()
            print('   idle gaps: %d, median %.2f ms, p90 %.2f ms, total %.0f ms' % (
                len(gaps), gaps[len(gaps) // 2] / 1000, gaps[int(len(gaps) * 0.9)] / 1000, sum(gaps) / 1000))
    pres = sorted(e['ts'] for e in cpu if e['name'].startswith('present wait'))
    if len(pres) > 2:
        dt = [(b - a) / 1000 for a, b in zip(pres, pres[1:])]
        print('\n== presents: %d, frame %.1f ms avg (%.1f FPS), max %.1f ms' % (len(pres), sum(dt) / len(dt),
              1000 * len(dt) / sum(dt), max(dt)))


if __name__ == '__main__':
    main()
