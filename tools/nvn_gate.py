"""A run on the NVN path with memory sampled: video memory (nvidia-smi) and the game's working set and commit, once a second, into <run>/memory.csv; a summary at the end.

  python tools/nvn_gate.py <name> <seconds> [--shots-at ms,ms] [--replay folder]

The game needs SWR_GPU=nvn NXRT_GFX=vulkan, which this sets. The run itself is tools/run_game.py."""
import argparse, csv, glob, os, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sample(stop, rows):
    ps = "(Get-Process MKW -ErrorAction SilentlyContinue | Select-Object -First 1 | ForEach-Object { '{0},{1}' -f $_.WorkingSet64, $_.PagedMemorySize64 })"
    t0 = time.time()
    while not stop.wait(1.0):
        vram = ''
        try:
            vram = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL).stdout.strip()
        except Exception: pass
        mem = ''
        try:
            mem = subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL).stdout.strip()
        except Exception: pass
        rows.append((round(time.time() - t0, 1), vram, mem))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('seconds', type=int); ap.add_argument('--shots-at', default=''); ap.add_argument('--replay', default=''); a = ap.parse_args()
    os.environ.update(SWR_GPU='nvn', NXRT_GFX='vulkan')
    rows = []; stop = threading.Event(); t = threading.Thread(target=sample, args=(stop, rows), daemon=True); t.start()
    cmd = [sys.executable, os.path.join(ROOT, 'tools', 'run_game.py'), '--name', a.name, '--seconds', str(a.seconds)]
    if a.shots_at: cmd += ['--shots-at', a.shots_at]
    if a.replay: cmd += ['--replay', a.replay]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.seconds + 300)
    stop.set(); t.join(10)
    print(r.stdout[-300:])
    run = sorted(glob.glob(os.path.join(ROOT, 'games', 'MKW', 'runs', '*_' + a.name)))[-1]
    with open(os.path.join(run, 'memory.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['s', 'vram_mb,gpu%', 'working_set,commit'])
        w.writerows(rows)
    vr = [int(x[1].split(',')[0]) for x in rows if x[1]]; ws = [int(x[2].split(',')[0]) for x in rows if x[2]]; cm = [int(x[2].split(',')[1]) for x in rows if x[2]]
    gp = [int(x[1].split(',')[1]) for x in rows if x[1]]
    print('video memory peak %d MB; working set peak %.1f GB; commit peak %.1f GB; GPU load mean %.0f%%' % (max(vr or [0]), max(ws or [0]) / 2**30, max(cm or [0]) / 2**30, sum(gp) / max(1, len(gp))))
    print(run)


if __name__ == '__main__':
    main()
