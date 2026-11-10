"""menucheck.py: the stretched-mesh test in the main menu's free roam, a few minutes a run, no replay.

  python tools/menucheck.py NAME [--runs 5] [--frames 1500] [--blocks 300] [--until LOGTEXT] [--press-ms MS] [--env K=V ...] [--extra k=v ...] [--flight]

Each run boots games/MKWTest on the saved-game copy and waits, untouched, for the title screen's attract mode to reach its demo race
(--until: the first audience sound stream in the log, about 220 s after start; --until '' --press-ms 100000 presses L+R for the menu
instead), takes a screenshot when scoring starts, then every 100 frames asks the game for one-frame flashes over BLOCKS 8x8 blocks
(gamectl flashes; the stretched meshes cover hundreds, the kart's own movement up to ~170). --flight saves the recorder (550 MB)
once, at the first window with a flash. Results go to games/MKWTest/menucheck.csv: name, run, frames scored, flashes, windows with any.
The menu and the title screen are both live free roam: the driving goes on behind the UI."""
import argparse, os, re, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gamectl as G
from panelcheck import games, frame


def one(a, run):
    tag = '%s_%d' % (a.name, run)
    t_wait = time.time()
    while G.other_game_running():   # the user is playing their install: wait, never start beside it (up to 2 hours)
        if time.time() - t_wait > 7200: return tag, -1, -1, -1, 'the user played for 2 hours: stopped'
        time.sleep(30)
    cmd = [sys.executable, os.path.join(HERE, 'gamectl.py'), 'start', '--seconds', str(a.seconds), '--name', tag]
    if a.press_ms: cmd += ['--script', '%d:C0' % a.press_ms]
    if a.replay: cmd += ['--replay', a.replay]
    for e in a.env: cmd += ['--env', e]
    for e in a.extra: cmd += ['--extra', e]
    subprocess.run(cmd, capture_output=True, timeout=60)
    t0 = time.time(); logp = os.path.join(G.TEST, 'MKW.log')
    # scoring starts at the line --until in the log (the attract mode's demo race: its audience sounds), or once the press is 15 s behind us
    while True:
        time.sleep(1)
        if time.time() - t0 > 30 and not games(): return tag, -1, -1, -1, 'the game ended first'
        if a.until:
            if time.time() - t0 > a.until_s: G.send('quit', 3); time.sleep(20); return tag, -1, -1, -1, 'no "%s" line' % a.until
            if time.time() - t0 > 30 and a.until in open(logp, encoding='utf-8', errors='replace').read(): break
        elif time.time() - t0 > a.press_ms / 1000 + 15: break
    native = 'NVN taken over' in open(logp, encoding='utf-8', errors='replace').read()
    shots = os.path.join(G.TEST, 'panelshots'); os.makedirs(shots, exist_ok=True)
    G.send('shot ' + os.path.join(shots, tag + '_menu.png'), 3)
    f0 = frame(); nxt = f0 + 100; total = windows = 0; flight = None; lst = []
    while frame() < f0 + a.frames:
        if not games(): break
        f = frame()
        if f >= nxt:
            r = G.send('flashes 100 %d' % a.blocks) or ''
            m = re.match(r'\s*(\d+)', r); n = int(m.group(1)) if m else 0
            if n:
                total += n; windows += 1; lst.append(r.split(':', 1)[-1].strip())
                if a.flight and not flight: flight = (G.send('flight ' + tag, 30) or '').strip()
            nxt = f + 100
        time.sleep(0.5)
    scored = frame() - f0
    G.send('quit', 3); t1 = time.time()
    while games() and time.time() - t1 < 120: time.sleep(2)
    note = ('native' if native else 'NOT NATIVE') + ('; frames ' + ' '.join(lst))[:300] + ('; flight ' + flight if flight else '')
    return tag, scored, total, windows, note


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--runs', type=int, default=5); ap.add_argument('--frames', type=int, default=1500)
    ap.add_argument('--blocks', type=int, default=300); ap.add_argument('--press-ms', type=int, default=0)
    ap.add_argument('--until', default='Stream/Audience_Normal_Small'); ap.add_argument('--until-s', type=int, default=420); ap.add_argument('--seconds', type=int, default=600)
    ap.add_argument('--env', action='append', default=[]); ap.add_argument('--extra', action='append', default=[]); ap.add_argument('--flight', action='store_true')
    ap.add_argument('--replay', default='', help='drive with an input recording; --until "sync 3 (" then starts scoring at its race intro')
    ap.add_argument('--b-env',action='append', default=[], help='K=V added on every second run (an alternating on/off test)')
    a = ap.parse_args()
    if games(): print('a game is running: stop it first'); return 1
    csv = os.path.join(G.TEST, 'menucheck.csv'); res = []
    base_env = list(a.env)
    for run in range(a.runs):
        b = bool(a.b_env) and run % 2 == 1          # --b-env: every second run adds these, an alternating on/off test
        a.env = base_env + (a.b_env if b else [])
        r = one(a, run) + ('B' if b else 'A',); res.append(r)
        line = '%s,%d,%d,%d,%s,%s' % (r[0], r[1], r[2], r[3], r[5], r[4])
        print(line, flush=True); open(csv, 'a').write(time.strftime('%Y%m%d_%H%M%S') + ',' + line + '\n')
    for g in (('A', 'B') if a.b_env else ('A',)):
        rr = [r for r in res if r[5] == g and r[1] > 0]
        print('%s %s%s: %d of %d runs with flashes, %d flashes in all' % (a.name, g, ' (' + ' '.join(a.b_env) + ')' if g == 'B' else '',
              sum(1 for r in rr if r[2] > 0), len(rr), sum(r[2] for r in rr)), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
