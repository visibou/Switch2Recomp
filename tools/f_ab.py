"""Same-conditions A/B of Eden builds and SWR_EDEN_F_OFF switches: arms interleaved, so heat and warm caches fall on all
of them, one run at a time through bench.py, resumable, and it stops on the first bad run.

  python tools/f_ab.py run <batch> --scenes koopa,bowser --arms on,off,ours --counts 6,6,3
  python tools/f_ab.py run stab --scenes drive --arms on --counts 6 --capture
  python tools/f_ab.py run ab2 --scenes drive --arms on,off,ours --counts 4,4,2 --until 13:00
  python tools/f_ab.py report <batch> [<batch> ...]
  python tools/f_ab.py report <batch> --base off

An arm is "name=exe" or "name=exe:letters" (letters go to SWR_EDEN_F_OFF); shorthands: on=wip, off=wip:abcdef,
ours=oursrt. exe wip/oursrt = games/MKW/MKW.exe.wip/.oursrt (Eden branches phase-f-wip and ours on the
same runtime; kept = the fixes that were kept, on ours), copied over MKW.exe for the run: the name has to stay, the save, shader cache and log folders
follow it. Other names: noa=wip:a etc. (one candidate off).

The batch's state is games/MKW/ab_<batch>.jsonl, one line per finished run. Runs in 60 FPS mode are left out of
the numbers and counted by arm. Run it again to go on after a stop; a bad run stays in the file, and its slot is run again.
Compare back ms: the two courses hold 30 FPS, so their FPS says nothing.
--until HH:MM (today): no run starts that could end after it (exit 3), for when the machine is needed back; go on with the
same command later.
"""
import argparse, datetime, json, math, os, shutil, statistics, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GDIR = os.path.join(ROOT, 'games', 'MKW')
EXE = os.path.join(GDIR, 'MKW.exe')
REC = 'games/MKW/recordings/'
SCENES = {
    # Windows from the race music; the runs are long enough for a late boot (race music at up to ~150 s)
    'drive': ['--replay', REC + 'rec_20261103_214452', '--race-window', '23:193', '--seconds', '350'],
    'koopa': ['--replay', REC + 'rec_20261104_145600', '--race-window', '5:85', '--seconds', '240'],
    'bowser': ['--replay', REC + 'rec_20261104_143423', '--race-window', '5:150', '--seconds', '305'],
}
SHORT = {'on': 'wip', 'off': 'wip:abcdef', 'ours': 'oursrt', 'kept': 'kept'}
RUN_LIMIT = 1500    # s, the longest scene with its frame_check and then some: past this a run is bad and the batch stops


def parse_arm(text):
    name, _, spec = text.partition('=')
    if not spec: spec = SHORT.get(name) or ('wip:' + name[2:] if name.startswith('no') else '')
    exe, _, letters = spec.partition(':')
    if exe not in ('wip', 'oursrt', 'kept'): sys.exit('arm %s: no such build' % text)
    return name, exe, letters


def schedule(arms, counts):
    """rounds in turn, the order flipped every round; an arm with fewer runs is spread over the rounds"""
    rounds = max(counts)
    seq = []
    for i in range(rounds):
        round_arms = [a for a, n in zip(arms, counts) if int((i + 1) * n / rounds) > int(i * n / rounds)]
        seq += round_arms if i % 2 == 0 else round_arms[::-1]
    return seq


def state_path(batch): return os.path.join(GDIR, 'ab_%s.jsonl' % batch)


def load(batch):
    path = state_path(batch)
    if not os.path.isfile(path): return []
    return [json.loads(line) for line in open(path) if line.strip()]


def cmd_run(a):
    arms = [parse_arm(x) for x in a.arms.split(',')]
    counts = [int(x) for x in a.counts.split(',')]
    if len(arms) != len(counts): sys.exit('--arms and --counts differ')
    for exe in set(x[1] for x in arms):
        if not os.path.isfile(EXE + '.' + exe): sys.exit('missing ' + EXE + '.' + exe)
    until = None
    if a.until:
        h, m = a.until.split(':')
        until = datetime.datetime.now().replace(hour=int(h), minute=int(m), second=0).timestamp()
    for scene in a.scenes.split(','):
        for slot, arm in enumerate(schedule(arms, counts)):
            done = load(a.batch)
            if any(d['scene'] == scene and d['slot'] == slot and d['state'] != 'bad' for d in done): continue
            name, exe, letters = arm
            if until and time.time() + float(SCENES[scene][SCENES[scene].index('--seconds') + 1]) + 60 > until:
                print('%s stopped for --until %s, before %s slot %d' % (time.strftime('%H:%M:%S'), a.until, scene, slot), flush=True)
                sys.exit(3)
            shutil.copyfile(EXE + '.' + exe, EXE)
            result = os.path.join(GDIR, 'ab_%s.tmp' % a.batch)
            if os.path.isfile(result): os.remove(result)
            cmd = [sys.executable, os.path.join(ROOT, 'tools', 'bench.py'), '%s_%s_%s_s%d' % (a.batch, scene, name, slot),
                   '--runs', '1', '--result-file', result] + SCENES[scene]
            if a.capture: cmd.append('SWR_CAPTURE=150,170')
            if letters: cmd.append('SWR_EDEN_F_OFF=' + letters)
            print('%s %s slot %d arm %s' % (time.strftime('%H:%M:%S'), scene, slot, name), flush=True)
            # To a file, not a pipe: anything the child leaves behind would keep a pipe open, and the batch would wait on it.
            # An unnamed one: such a process locks a named file on Windows, and this goes away with its last handle.
            with tempfile.TemporaryFile('w+', encoding='utf-8', errors='replace') as out:
                late = ''
                try: subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=RUN_LIMIT)
                except subprocess.TimeoutExpired: late = 'bench.py took over %d s; ' % RUN_LIMIT
                out.seek(0)
                text = out.read()
            print('\n'.join(l for l in text.splitlines() if not l.startswith('==')), flush=True)
            r = json.loads(open(result).readline()) if os.path.isfile(result) else {'why': late + text[-600:]}
            if os.path.isfile(result): os.remove(result)
            r.update(scene=scene, slot=slot, arm=name, time=time.strftime('%Y-%m-%d %H:%M'),
                     state='mode60' if r.get('soft') else 'bad' if r.get('hung', True) else 'ok')
            with open(state_path(a.batch), 'a') as f: f.write(json.dumps(r) + '\n')
            if r['state'] == 'bad':
                print('STOPPED %s slot %d arm %s: %s' % (scene, slot, name, r.get('why', '?')), flush=True)
                sys.exit(2)
    print('batch %s finished' % a.batch, flush=True)


def t95(df):
    """two-sided 95% Student t, close enough between table points"""
    table = {1: 12.7, 2: 4.3, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 12: 2.18, 15: 2.13, 20: 2.09}
    keys = [k for k in table if k <= df]
    return table[max(keys)] if keys else 12.7


def diff(x, y):
    """mean(x) - mean(y) with a Welch 95% interval"""
    if len(x) < 2 or len(y) < 2: return None
    d = statistics.mean(x) - statistics.mean(y)
    vx, vy = statistics.variance(x) / len(x), statistics.variance(y) / len(y)
    if vx + vy == 0: return d, 0.0
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    return d, t95(math.floor(df)) * math.sqrt(vx + vy)


def cmd_report(a):
    rows = []
    for b in a.batch: rows += load(b)
    for scene in sorted(set(r['scene'] for r in rows)):
        sr = [r for r in rows if r['scene'] == scene]
        arms = sorted(set(r['arm'] for r in sr))
        print('== %s' % scene)
        print('%-8s %3s %5s %6s %6s %6s %6s  %s' % ('arm', 'n', '60fps', 'back', 'sd', 'FPS', '1%low', 'back ms a run'))
        ok = {x: [r for r in sr if r['arm'] == x and r['state'] == 'ok'] for x in arms}
        for x in arms:
            b = [r['back_ms'] for r in ok[x]]
            if not b:
                print('%-8s   0' % x); continue
            print('%-8s %3d %5d %6.2f %6.2f %6.1f %6.1f  %s' % (
                x, len(b), sum(1 for r in sr if r['arm'] == x and r['state'] == 'mode60'), statistics.mean(b),
                statistics.stdev(b) if len(b) > 1 else 0, statistics.mean(r['fps'] for r in ok[x]),
                statistics.mean(r['low1'] for r in ok[x]), ' '.join('%.1f' % v for v in b)))
        base = a.base if a.base in arms else None
        for x in arms:
            if not base or x == base: continue
            d = diff([r['back_ms'] for r in ok[x]], [r['back_ms'] for r in ok[base]])
            if d: print('  %s - %s: %+.2f ms +- %.2f (95%%)%s' % (x, base, d[0], d[1], '' if abs(d[0]) > d[1] else ', not separated'))
    bad = [r for r in rows if r['state'] == 'bad']
    print('\nruns: %d, bad %d, 60 FPS mode %d' % (len(rows), len(bad), sum(1 for r in rows if r['state'] == 'mode60')))
    for r in bad: print('  BAD %s %s slot %s: %s' % (r['scene'], r['arm'], r['slot'], r.get('why', '?')[:220]))


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    r = sp.add_parser('run')
    r.add_argument('batch'); r.add_argument('--scenes', default='koopa,bowser'); r.add_argument('--arms', default='on,off,ours')
    r.add_argument('--counts', default='6,6,3'); r.add_argument('--capture', action='store_true')
    r.add_argument('--until', default=None, help='HH:MM today: start no run that could end after it')
    r = sp.add_parser('report')
    r.add_argument('batch', nargs='+'); r.add_argument('--base', default='off')
    a = ap.parse_args()
    (cmd_run if a.cmd == 'run' else cmd_report)(a)


if __name__ == '__main__':
    main()
