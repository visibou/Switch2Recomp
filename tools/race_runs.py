"""Runs the benchmark race N times, one at a time, and says how each went: python tools/race_runs.py NAME [--runs 2] [--seconds 420] [--env K=V ...] [--extra k=v ...]

A run whose frame number stands still for 40 s after frame 500 is a freeze: its game is killed and the next run waits for the driver to let go of it (up to 8 minutes). Prints race_stats for each run at the end.
  --rig        the freeze rig: NXRT_VK_SPLIT_MS=2 NXRT_VK_CHECKPOINTS=all
  --aftermath  NXRT_VK_AFTERMATH=1 with its own pipeline cache (games/MKWTest/vk_am); a frozen game is left alone until Aftermath says whether a dump came (up to 7 minutes)
  --stop-on-freeze  ends the series at the first freeze"""
import argparse, glob, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gamectl as G


def games():
    r = subprocess.run('tasklist /FI "IMAGENAME eq MKW.exe"', shell=True, capture_output=True, text=True, timeout=20).stdout
    return r.lower().count('marioka')


def wait_gone(limit):
    t0 = time.time()
    while time.time() - t0 < limit:
        if not games(): return True
        subprocess.run('taskkill /F /IM MKW.exe', shell=True, capture_output=True, timeout=30); time.sleep(5)
    return False


def gpu_lost():
    try: return 'the GPU stopped responding' in open(os.path.join(G.TEST, 'MKW.log'), encoding='utf-8', errors='replace').read()[-400000:]
    except OSError: return False


def wait_aftermath(limit):
    """The game's live log says when Aftermath has its dump, or that none came (vk_dev.c, after the 15 s wait report)."""
    log = os.path.join(G.TEST, 'MKW.log'); t0 = time.time()
    while time.time() - t0 < limit:
        try: txt = open(log, encoding='utf-8', errors='replace').read()[-200000:]
        except OSError: txt = ''
        for key in ('a dump was written', 'no dump came'):
            if 'Aftermath: ' + key in txt: return '%s after %.0f s' % (key, time.time() - t0)
        if not games(): return 'the game ended after %.0f s' % (time.time() - t0)
        time.sleep(5)
    return 'nothing after %.0f s' % limit


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--runs', type=int, default=2); ap.add_argument('--seconds', type=int, default=420)
    ap.add_argument('--env', action='append', default=[]); ap.add_argument('--extra', action='append', default=[])
    ap.add_argument('--rig', action='store_true'); ap.add_argument('--aftermath', action='store_true'); ap.add_argument('--stop-on-freeze', action='store_true')
    ap.add_argument('--cache', default='', help='a test shader cache (NXRT_SHADER_CACHE) and its own pipeline cache beside it (<cache>_vk): the installed game\'s are not touched')
    a = ap.parse_args()
    if a.cache:
        c = os.path.abspath(a.cache); a.env += ['NXRT_SHADER_CACHE=' + c]
        if not a.aftermath: a.env += ['NXRT_VK_CACHE_DIR=' + c + '_vk']
    if a.rig: a.env += ['NXRT_VK_SPLIT_MS=2', 'NXRT_VK_CHECKPOINTS=all']
    if a.aftermath: a.env += ['NXRT_VK_AFTERMATH=1', 'NXRT_VK_CACHE_DIR=' + os.path.join(G.TEST, 'vk_am')]
    print('series %s: %d runs of %d s, env %s; started %s' % (a.name, a.runs, a.seconds, ' '.join(a.env) or '(default)', time.strftime('%H:%M:%S')), flush=True)
    freezes = 0
    if games(): print('a game is running: stop it first'); return 1
    done = []
    for n in range(a.runs):
        name = '%s%d' % (a.name, n + 1)
        st = [sys.executable, os.path.join(HERE, 'gamectl.py'), 'start', '--replay', 'rec_20261103_214452', '--seconds', str(a.seconds), '--name', name, '--env', 'NXRT_VK_FLIGHT=0']
        for e in a.env: st += ['--env', e]
        for e in a.extra: st += ['--extra', e]
        subprocess.run(st, capture_output=True, timeout=30)
        t0 = time.time(); last = -1; t_last = time.time(); frozen = False; started = False
        while time.time() - t0 < a.seconds + 90:
            time.sleep(4)
            s = G.status()
            if s:
                started = True; f = int(s['frame'])
                if f != last: last = f; t_last = time.time()
                elif f > 500 and time.time() - t_last > 40: frozen = True; break
                elif f > 0 and time.time() - t_last > 40 and gpu_lost(): frozen = True; break   # a freeze at boot: the frame count stands still and the log says the GPU stopped
            elif started and not games(): break
            elif not started and time.time() - t0 > 60: break
        if not frozen and gpu_lost(): frozen = True   # the game ended on its own after a device loss (Aftermath lets it go)
        print('%s: %s at frame %s after %.0f s (%s)' % (name, 'FROZEN' if frozen else 'ended', last, time.time() - t0, time.strftime('%H:%M:%S')), flush=True)
        if frozen: freezes += 1
        if frozen and a.aftermath:
            print('  Aftermath:', wait_aftermath(420), flush=True)
        if frozen or games():
            print('  driver let go:', wait_gone(480), flush=True)
        time.sleep(8)
        runs = sorted(glob.glob(os.path.join(G.TEST, 'runs', '*_' + name)))
        if runs: done.append(runs[-1])
        if frozen and a.stop_on_freeze: break
    print('series %s: %d freezes in %d runs; ended %s' % (a.name, freezes, len(done), time.strftime('%H:%M:%S')), flush=True)
    have = [r for r in done if os.path.exists(os.path.join(r, 'game.log'))]
    if have:
        print(subprocess.run([sys.executable, os.path.join(HERE, 'race_stats.py')] + have + ['--len', '100'], capture_output=True, text=True).stdout, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
