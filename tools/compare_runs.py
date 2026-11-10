"""Side-by-side numbers for captured runs (after tools/frame_check.py has run on each).

  python tools/compare_runs.py <run folder or name part> [...] [--from 41] [--to 200]

Per run, over composites between --from and --to seconds (composites.csv): composite interval median and 99th
percentile, longest, frames over 50 and 100 ms; from frame_check.txt: jump-backs and spikes per minute moving,
camera cuts and the slowest frame after each (median, max); GPU-thread texture upload time and host GPU busy time
per second.
"""
import argparse, csv, glob, os, re, statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find(run):
    if os.path.isdir(run): return run
    hits = sorted(glob.glob(os.path.join(ROOT, 'games', 'MKW', 'runs', '*' + run + '*')))
    if not hits: raise SystemExit('no run matches %s' % run)
    return hits[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--from', dest='t0', type=float, default=0)
    ap.add_argument('--to', dest='t1', type=float, default=1e9)
    a = ap.parse_args()
    print('%-34s %6s %6s %6s %6s %6s %6s %6s %5s %7s %7s %8s %8s' % (
        'run', 'comps', 'med', 'p99', 'max', '>50ms', '>100ms', 'back/m', 'cuts', 'cut med', 'cut max',
        'tex ms/s', 'gpu ms/s'))
    for r in a.runs:
        d = find(r)
        rows = [x for x in csv.DictReader(open(os.path.join(d, 'composites.csv')))
                if a.t0 * 1000 <= float(x['ms']) <= a.t1 * 1000]
        ints = sorted(float(y['ms']) - float(x['ms']) for x, y in zip(rows, rows[1:]))
        span = (float(rows[-1]['ms']) - float(rows[0]['ms'])) / 1000.0
        def per_s(col): return sum(float(x.get(col) or 0) for x in rows) / 1000.0 / span if col in rows[0] else float('nan')
        fc = open(os.path.join(d, 'frame_check.txt')).read() if os.path.isfile(os.path.join(d, 'frame_check.txt')) else ''
        back = re.search(r'^\s+back\s+\d+\s+([0-9.]+) per minute', fc, re.M)
        cuts = re.search(r'camera cuts: (\d+); slowest frame in the 10 from the cut: median (\d+) ms, max (\d+) ms', fc)
        print('%-34s %6d %6.1f %6.1f %6.0f %6d %6d %6s %5s %7s %7s %8.1f %8.1f' % (
            os.path.basename(d)[-34:], len(rows), ints[len(ints) // 2], ints[int(len(ints) * 0.99)], ints[-1],
            sum(1 for x in ints if x > 50), sum(1 for x in ints if x > 100), back.group(1) if back else '-',
            cuts.group(1) if cuts else '-', cuts.group(2) if cuts else '-', cuts.group(3) if cuts else '-',
            per_s('tex_upload_us'), per_s('gpu_busy_us')))


if __name__ == '__main__':
    main()
