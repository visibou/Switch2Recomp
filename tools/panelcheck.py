"""panelcheck.py: one replay scored for visual bugs: one-frame flashes over the whole race (gamectl flashes, the flight recorder's frames, kept in memory) and
screenshots at chosen moments, for the panels / stretched-mesh bugs (docs/PLAN_NEXT.md step 2).

  python tools/panelcheck.py NAME [--shots 0,60,120,180,240,300,600] [--blocks 118] [--env K=V ...] [--extra k=v ...] [--frames-after 6000]

Waits for the race intro's sync line, then every 100 frames asks for flashes in those 100 frames (a flash: a piece of at least BLOCKS 8x8 blocks that
changed for one frame only), and takes a screenshot each time the frame count passes intro + N. Prints the flash count per window and the total, and
leaves the shots in games/MKWTest/panelshots/NAME_<N>.png. The flight recorder must be on (it is by default; race_runs.py turns it off)."""
import argparse, os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gamectl as G

def games():
    return subprocess.run('tasklist /FI "IMAGENAME eq MKW.exe"', shell=True, capture_output=True, text=True).stdout.lower().count('marioka')

def frame():
    s = G.status()
    return int(s['frame']) if s else -1

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--shots', default='0,60,120,180,240,300,600'); ap.add_argument('--blocks', type=int, default=118)
    ap.add_argument('--env', action='append', default=[]); ap.add_argument('--extra', action='append', default=[]); ap.add_argument('--frames-after', type=int, default=6000)
    ap.add_argument('--sync', default='sync 3 ('); ap.add_argument('--replay', default='rec_20261103_214452')
    a = ap.parse_args()
    out = os.path.join(G.TEST, 'panelshots'); os.makedirs(out, exist_ok=True)
    if games(): print('a game is running: stop it first'); return 1
    cmd = [sys.executable, os.path.join(HERE, 'gamectl.py'), 'start', '--replay', a.replay, '--seconds', '900', '--name', a.name]
    for e in a.env: cmd += ['--env', e]
    for e in a.extra: cmd += ['--extra', e]
    subprocess.run(cmd, capture_output=True, timeout=60)
    time.sleep(20)
    logp, t0 = os.path.join(G.TEST, 'MKW.log'), time.time()
    while a.sync not in open(logp, encoding='utf-8', errors='replace').read():
        if time.time() - t0 > 400 or not games(): print('no sync line'); return 1
        time.sleep(1)
    f0 = frame(); shots = sorted(int(x) for x in a.shots.split(',') if x); total, windows, nxt = 0, [], f0 + 100
    while frame() < f0 + a.frames_after:
        if not games(): print('the game ended'); break
        f = frame()
        while shots and f >= f0 + shots[0]:
            subprocess.run([sys.executable, os.path.join(HERE, 'gamectl.py'), 'shot', os.path.join(out, '%s_%d.png' % (a.name, shots[0]))], capture_output=True, timeout=20); shots.pop(0)
        if f >= nxt:
            r = G.send('flashes 100 %d' % a.blocks) or ''
            m = re.match(r'\s*(\d+)', r); n = int(m.group(1)) if m else -1
            windows.append((f - f0, n)); total += max(n, 0); nxt = f + 100
            if n: print('  frames %d-%d after the intro: %d flash(es): %s' % (f - f0 - 100, f - f0, n, r.strip()[:160]), flush=True)
        time.sleep(0.5)
    subprocess.run([sys.executable, os.path.join(HERE, 'gamectl.py'), 'quit'], capture_output=True, timeout=20)
    t1 = time.time()
    while games() and time.time() - t1 < 120: time.sleep(2)
    s = open(logp, encoding='utf-8', errors='replace').read()
    print('%s: %d flash(es) in %d windows of 100 frames; native %s; shots in %s' % (a.name, total, len(windows), 'yes' if 'NVN taken over' in s else 'NO', out), flush=True)
    return 0 if total == 0 else 2

if __name__ == '__main__':
    sys.exit(main())
