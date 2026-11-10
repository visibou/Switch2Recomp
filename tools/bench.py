"""MKW race benchmark: N scripted runs into a race (Grand Prix, Mario Bros. Circuit) on the Eden backend,
measured over a window of the race.

  python tools/bench.py <name> [--runs 3] [--window 215:255] [--trace 230,3] [KEY=VALUE ...]
  python tools/bench.py <name> --replay games/MKW/recordings/rec_20261103_214452 [SWR_CAPTURE=150,170]

--replay drives with an input recording (runtime/input_rec.c) instead of the A presses, which never drove the kart:
the default window is then 150-320 s (the reference recording's race) and the run 335 s. With SWR_CAPTURE each run
also gets tools/frame_check.py; its glitch rates go to games/MKW/glitch_results.csv.
--race-window a:b counts the window from the race music (the BGM_Stage_ sync of the replay) instead of from the start:
boot takes 35-80 s, so a window fixed from the start covers a different part of the race in every run, or the menus.

KEY=VALUE: SWR_* / EDEN_* go to the environment, anything else is a runtime option (resolution=0.75, docked=0).
Per run, from frames.csv (one line per present, svc_ns2.c): average FPS, 1% low (99th percentile frame time),
frames over 33 ms, game speed (vblanks the game advanced per second / 60: its clock follows its present interval),
present wait; nvidia-smi GPU load and VRAM in the window; Eden's per-second lines (host GPU timing, GPU thread
profile when EDEN_NX2_GPU_PROFILE=1). A run whose frames stop before the window ends is marked HUNG and left out
of the median, and so is one that crashed (by its log: the exit code says 0 after most crashes), stayed in the menus
or spent more than 2% of the window in the game's 60 FPS mode. The last kind is left out but does not stop the runs;
any other bad run stops them, exit status 2. Do not measure with EDEN_NX2_GPU_PROFILE=1: its timers cost ~20%.
Results: the run folders (bench.txt) and games/MKW/bench_results.csv; --result-file adds a JSON line a run.
"""
import argparse, csv, json, os, re, statistics, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GDIR = os.path.join(ROOT, 'games', 'MKW')
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import s2r_paths as P
PY = P.python()


def sample_gpu(t_start, t_end, out, stop):
    """nvidia-smi utilization and memory every second between t_start and t_end (time.time()), or until stop: the game can
    end before the window does, and a nvidia-smi -l left behind runs for good. It held f_ab's pipe through its inherited
    stderr for 9 hours on 2026-11-05, so it gets none."""
    while time.time() < t_start:
        if stop.wait(0.5): return
    p = subprocess.Popen(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits',
                          '-l', '1'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, text=True)
    threading.Thread(target=lambda: (stop.wait(max(0, t_end - time.time())), p.kill()), daemon=True).start()
    for line in p.stdout:
        try: out.append(tuple(int(x) for x in line.split(',')))
        except ValueError: pass


def crash_of(run_dir):
    """the first crash line of a run's log, with the report after it, or ''"""
    try: lines = open(os.path.join(run_dir, 'game.log'), encoding='utf-8', errors='replace').read().splitlines()
    except OSError: return ''
    for i, line in enumerate(lines):
        if re.search(r'\*\*\* (Crash|The game called|Shutdown)|FATAL:', line):
            return ' | '.join(l.strip()[:150] for l in lines[i:i + 6])
    return ''


def exit_of(run_dir):
    try: m = re.match(r'exit: (.*)', open(os.path.join(run_dir, 'summary.txt'), encoding='utf-8', errors='replace').readline())
    except OSError: return None
    return m.group(1).strip() if m else None


def race_start(run_dir):
    """when the replay's race music started, s since start, or None"""
    try: log = open(os.path.join(run_dir, 'game.log'), encoding='utf-8', errors='replace').read()
    except OSError: return None
    m = re.search(r'input replay: sync \d+ \(\S*BGM_Stage_\S*\) at ([0-9.]+) s', log)
    return float(m.group(1)) if m else None


def replay_notes(run_dir):
    try: log = open(os.path.join(run_dir, 'game.log'), encoding='utf-8', errors='replace').read()
    except OSError: return {}
    return {'tapped': 'input replay: tapping A' in log, 'lost_sync': 'never came' in log}


def analyze(run_dir, w0, w1, race=False):
    t = race_start(run_dir) if race else 0.0
    if t is None:
        r = {'dir': run_dir, 'hung': True, 'soft': False, 'why': 'the replay did not reach the race music'}
    else:
        r = analyze_frames(run_dir, w0 + t, w1 + t)
        r['t_race'] = t
    r['exit'] = exit_of(run_dir)
    r.update(replay_notes(run_dir))
    if r.get('lost_sync'):      # the menus went differently: another course, or none
        r['hung'] = True; r['soft'] = False; r['why'] = 'a replay sync never came: not the recorded race'
    crash = crash_of(run_dir)
    if crash or r['exit'] not in ('0', None):
        r['hung'] = True; r['soft'] = False
        r['crash'] = crash or 'no report in the log'
        r['why'] = 'CRASH (exit %s): %s' % (r['exit'], r['crash'])
    return r


def analyze_frames(run_dir, w0, w1):
    r = {'dir': run_dir, 'hung': True}
    path = os.path.join(run_dir, 'frames.csv')
    if not os.path.isfile(path): r['why'] = 'no frames.csv'; return r
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try: rows.append((float(row['ms']) / 1000.0, int(row['interval']), int(row['wait_us']),
                              float(row.get('front_wait_ms') or 0), float(row.get('back_wait_ms') or 0)))
            except (ValueError, KeyError, TypeError): pass
    win = [x for x in rows if w0 <= x[0] <= w1]
    if len(win) < 10: r['why'] = '%d frames in the window' % len(win); return r
    dts = [(b[0] - a[0]) * 1000.0 for a, b in zip(win, win[1:])]
    span = win[-1][0] - win[0][0]
    r['fps'] = (len(win) - 1) / span
    p99 = sorted(dts)[int(len(dts) * 0.99)]
    r['low1'] = 1000.0 / p99
    r['p99_ms'] = p99
    r['max_ms'] = max(dts)
    r['over33'] = sum(1 for d in dts if d > 33.4)
    r['speed'] = 100.0 * sum(x[1] for x in win[1:]) / (span * 60.0)
    r['interval2'] = sum(1 for x in win if x[1] >= 2)
    # Work per frame of Eden's two GPU threads: the window minus what they waited for work (cheap: no zone timers)
    if win[-1][4] > 0:
        frames = len(win) - 1
        r['back_ms'] = (span * 1000.0 - (win[-1][4] - win[0][4])) / frames
        r['front_ms'] = (span * 1000.0 - (win[-1][3] - win[0][3])) / frames
    r['wait_ms'] = statistics.mean(x[2] for x in win) / 1000.0
    r['hung'] = win[-1][0] < w1 - 5 or max(dts) > 3000
    if r['hung']: r['why'] = 'frames stop at %.0f s (max frame %.0f ms)' % (win[-1][0], max(dts))
    # Frames at interval 1 in a race window: the game's dynamic pacing left its 30 FPS mode, and its frames are lighter
    # there, so back ms from such a run is another measure. Not a broken run: count them by arm.
    r['mode60'] = 100.0 * sum(1 for x in win if x[1] < 2) / len(win)
    if r['mode60'] > 2.0:
        r['hung'] = True; r['soft'] = True
        r['why'] = '%.0f%% of the window in 60 FPS mode' % r['mode60']
    # A replay that missed its menu presses sits in the menus at 60 FPS: no race to measure. Last: it wins over the above
    if r.get('back_ms', 99.0) < 8.0:
        r['hung'] = True; r['soft'] = False
        r['why'] = 'back thread %.1f ms: the replay did not reach the race' % r['back_ms']
    # Eden's per-second lines inside the window (the eden log's own clock starts with Eden, close to the process)
    lines = []
    el = os.path.join(run_dir, 'eden_log.txt')
    if os.path.isfile(el):
        for line in open(el, encoding='utf-8', errors='replace'):
            m = re.match(r'\[\s*([0-9.]+)\]', line)
            if m and w0 <= float(m.group(1)) <= w1 + 10 and (
                    'per second' in line or 'host GPU' in line or 'submissions (' in line):
                lines.append(line.rstrip())
    r['eden'] = lines
    # EDEN_NX2_GPU_PROFILE=1: the back thread's work per frame (its second minus "split wait"), and its zones
    waits, zones = [], {}
    for line in lines:
        if 'GPU back thread per second' not in line: continue
        m = re.search(r'split wait ([0-9.]+) ms', line)
        if not m: continue
        waits.append(float(m.group(1)))
        for name, ms in re.findall(r'\| ([a-zA-Z ]+?) ([0-9.]+) ms', line):
            zones.setdefault(name, []).append(float(ms))
    if len(waits) >= 3:
        r['zones'] = {k: statistics.mean(v) / r['fps'] for k, v in zones.items()}
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--window', default=None)
    ap.add_argument('--race-window', default=None, help='a:b seconds from the race music, with --replay')
    ap.add_argument('--seconds', type=int, default=None)
    ap.add_argument('--replay', default='', help='an input recording folder to drive with')
    ap.add_argument('--trace', default='', help='SWR_TRACE for the first run, e.g. 230,3')
    ap.add_argument('--result-file', default='', help='add one JSON line per run to this file')
    ap.add_argument('kv', nargs='*')
    a = ap.parse_args()
    if a.window is None: a.window = '150:320' if a.replay else '215:255'
    if a.seconds is None: a.seconds = 335 if a.replay else 260
    w0, w1 = (float(x) for x in a.window.split(':'))
    race = bool(a.race_window)
    if race: w0, w1 = (float(x) for x in a.race_window.split(':'))
    nominal = 125.0 if race else 0.0     # for what only needs the time roughly: GPU load samples and screenshots
    env = dict(os.environ)
    env.setdefault('SWR_GPU', 'eden')
    env.setdefault('SWR_GPU_TIMING', '1')
    extra = []
    for kv in a.kv:
        k, _, v = kv.partition('=')
        if k.startswith('SWR_') or k.startswith('EDEN_'): env[k] = v
        else: extra += ['--extra', kv]
    # A every 4 s from 60 s to the end: menus advance whenever loading allows, in the race it taps accelerate
    script = ','.join('%d:1,%d:0' % (t, t + 300) for t in range(60000, a.seconds * 1000 - 3000, 4000))
    drive = ['--replay', a.replay] if a.replay else ['--script', script]
    results = []
    stopped = ''
    for i in range(a.runs):
        renv = dict(env)
        if a.trace and i == 0: renv['SWR_TRACE'] = a.trace
        gpu = []
        t0 = time.time()
        stop_gpu = threading.Event()
        th = threading.Thread(target=sample_gpu, args=(t0 + nominal + w0 + 3, t0 + nominal + w1, gpu, stop_gpu), daemon=True)
        th.start()
        out = subprocess.run([PY, os.path.join(ROOT, 'tools', 'run_game.py'), '--seconds', str(a.seconds),
                              '--shots-at', '%d,%d' % (int((nominal + w0) * 1000) + 10000, int((nominal + w1) * 1000) + 2000)] + drive + [
                              '--name', 'bench_%s_%d' % (a.name, i + 1)] + (['--game', os.environ['SWR_BENCH_EXE']] if os.environ.get('SWR_BENCH_EXE') else []) + extra,
                             env=renv, capture_output=True, text=True)
        stop_gpu.set()
        th.join(10)
        m = re.search(r'results: (.*)', out.stdout)
        if not m:
            print('run %d: run_game failed:\n%s%s' % (i + 1, out.stdout[-2000:], out.stderr[-2000:]))
            stopped = 'run_game failed: ' + (out.stdout + out.stderr)[-300:]
            break
        r = analyze(m.group(1).strip(), w0, w1, race)
        if gpu:
            r['gpu_util'] = statistics.mean(g[0] for g in gpu)
            r['vram'] = max(g[1] for g in gpu)
        if os.path.isfile(os.path.join(r['dir'], 'frames.bin')):   # SWR_CAPTURE: glitches per minute while moving
            fc = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'frame_check.py'), r['dir'], '--strips', '16'],
                                capture_output=True, text=True).stdout
            for k in ('back', 'flicker', 'repeat', 'black', 'spike'):
                m2 = re.search(r'^\s+%s\s+(\d+)\s+([0-9.]+) per minute' % k, fc, re.M)
                if m2: r['g_' + k] = float(m2.group(2))
            # Captures are ~1 GB a run and the strips stay: 99 of them filled the disk on 2026-11-04
            os.remove(os.path.join(r['dir'], 'frames.bin'))
        results.append(r)
        with open(os.path.join(r['dir'], 'bench.txt'), 'w') as f:
            f.write(fmt(r) + '\n\n' + '\n'.join(r.get('eden', [])) + '\n')
        if a.result_file:
            with open(a.result_file, 'a') as f:
                f.write(json.dumps({k: v for k, v in r.items() if k not in ('eden', 'zones')}) + '\n')
        print('run %d: %s%s' % (i + 1, fmt(r), ', back %.1f ms, front %.1f ms' % (r['back_ms'], r['front_ms'])
                                if 'back_ms' in r else ''), flush=True)
        if 'zones' in r:
            print('   back zones, ms a frame: ' + ', '.join('%s %.1f' % kv for kv in sorted(
                r['zones'].items(), key=lambda kv: -kv[1])[:12]), flush=True)
        if r['hung'] and not r.get('soft'):
            stopped = r.get('why', 'bad run')
            print('STOPPED after run %d: %s' % (i + 1, stopped), flush=True)
            break
    good = [r for r in results if not r['hung']]
    keys = ('fps', 'low1', 'over33', 'speed', 'wait_ms', 'gpu_util', 'vram', 'back_ms', 'front_ms')
    med = {k: statistics.median(r[k] for r in good if k in r) for k in keys if any(k in r for r in good)}
    print('== %s: %d of %d runs valid; median: %s' % (a.name, len(good), len(results),
          ', '.join('%s %.1f' % (k, v) for k, v in med.items())))
    res = os.path.join(GDIR, 'bench_results.csv')
    new = not os.path.isfile(res)
    with open(res, 'a', newline='') as f:
        w = csv.writer(f)
        if new: w.writerow(['time', 'name', 'settings', 'valid', 'runs'] + list(keys))
        w.writerow([time.strftime('%Y-%m-%d %H:%M'), a.name, ' '.join(a.kv), len(good), len(results)] +
                   ['%.1f' % med[k] if k in med else '' for k in keys])
    gk = ('g_back', 'g_flicker', 'g_repeat', 'g_black', 'g_spike')
    if any(k in r for r in results for k in gk):   # per minute while moving, every run (a hung run too)
        print('== %s glitches per minute (back flicker repeat black spike): %s' % (a.name, '; '.join(
            ' '.join('%.1f' % r.get(k, -1) for k in gk) for r in results)))
        res = os.path.join(GDIR, 'glitch_results.csv')
        new = not os.path.isfile(res)
        with open(res, 'a', newline='') as f:
            w = csv.writer(f)
            if new: w.writerow(['time', 'name', 'settings', 'run'] + [k[2:] + '_per_min' for k in gk])
            for i, r in enumerate(results):
                w.writerow([time.strftime('%Y-%m-%d %H:%M'), a.name, ' '.join(a.kv), os.path.basename(r['dir'])] +
                           ['%.2f' % r[k] if k in r else '' for k in gk])
    if stopped: sys.exit(2)


def fmt(r):
    if 'fps' not in r: return 'HUNG/INVALID (%s) %s' % (r.get('why', ''), r['dir'])
    s = '%5.1f FPS, 1%% low %5.1f (p99 %.1f ms, max %.0f ms), %d frames >33 ms, game speed %.0f%%, present wait %.1f ms, 60 FPS mode %.0f%%' % (
        r['fps'], r['low1'], r['p99_ms'], r['max_ms'], r['over33'], r['speed'], r['wait_ms'], r['mode60'])
    if 'gpu_util' in r: s += ', GPU %.0f%%, VRAM %d MiB' % (r['gpu_util'], r['vram'])
    if 'g_back' in r: s += ', per minute: back %.1f, flicker %.1f, repeat %.1f' % (r['g_back'], r['g_flicker'], r['g_repeat'])
    if r['hung']: s = 'BAD (%s) ' % r.get('why', '') + s
    return s + '  [%s]' % os.path.basename(r['dir'])


if __name__ == '__main__':
    main()
