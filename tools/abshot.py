"""abshot.py: screenshots at the same moments of a replay, for comparing two setups (shader caches, switches) side by side.

  python tools/abshot.py NAME --frames 100,400,1000 [--sync "sync 3 ("] [--env K=V ...] [--out DIR]

Starts the replay, waits for the sync line in the log (default: the race intro), then takes a screenshot each time the frame count passes
sync + N, and quits. Counting frames, not seconds, keeps the moments the same when the speed differs (the game advances per frame). Shots go to
DIR/NAME_<N>.png (default games/MKWTest/abshots). Compare two runs with tools/abshot.py --stack A B: one image with A's shots over B's."""
import argparse, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gamectl as G

def games():
    return subprocess.run('tasklist /FI "IMAGENAME eq MKW.exe"', shell=True, capture_output=True, text=True).stdout.lower().count('marioka')

def frame():
    s = G.status()
    return int(s['frame']) if s else -1

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('name', nargs='?'); ap.add_argument('--frames', default='100,400,1000'); ap.add_argument('--sync', default='sync 3 (')
    ap.add_argument('--env', action='append', default=[]); ap.add_argument('--out', default=os.path.join(G.TEST, 'abshots')); ap.add_argument('--replay', default='rec_20261103_214452')
    ap.add_argument('--stack', nargs=2, metavar=('A', 'B'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    frames = [int(x) for x in a.frames.split(',')]
    if a.stack:
        from PIL import Image
        for n in frames:
            ims = [Image.open(os.path.join(a.out, '%s_%d.png' % (k, n))).convert('RGB').resize((960, 540)) for k in a.stack]
            out = Image.new('RGB', (960, 1080)); out.paste(ims[0], (0, 0)); out.paste(ims[1], (0, 540))
            p = os.path.join(a.out, '%s_vs_%s_%d.png' % (a.stack[0], a.stack[1], n)); out.save(p); print(p)
        return 0
    if games(): print('a game is running: stop it first'); return 1
    cmd = [sys.executable, os.path.join(HERE, 'gamectl.py'), 'start', '--replay', a.replay, '--seconds', '600', '--name', a.name, '--env', 'NXRT_VK_FLIGHT=0']
    for e in a.env: cmd += ['--env', e]
    subprocess.run(cmd, capture_output=True, timeout=60)
    time.sleep(20)
    logp, t0 = os.path.join(G.TEST, 'MKW.log'), time.time()
    while a.sync not in open(logp, encoding='utf-8', errors='replace').read():
        if time.time() - t0 > 400 or not games(): print('no sync line'); return 1
        time.sleep(1)
    f0 = frame()
    for n in frames:
        while frame() < f0 + n:
            if not games(): print('the game ended'); return 1
            time.sleep(0.3)
        p = os.path.join(a.out, '%s_%d.png' % (a.name, n))
        subprocess.run([sys.executable, os.path.join(HERE, 'gamectl.py'), 'shot', p], capture_output=True, timeout=20)
    time.sleep(3)
    subprocess.run([sys.executable, os.path.join(HERE, 'gamectl.py'), 'quit'], capture_output=True, timeout=20)
    t1 = time.time()
    while games() and time.time() - t1 < 120: time.sleep(2)
    s = open(logp, encoding='utf-8', errors='replace').read()
    print('%s: native %s, %d shaders missing; shots in %s' % (a.name, 'yes' if 'NVN taken over' in s else 'NO', s.count('is not in the shader cache'), a.out))
    return 0

if __name__ == '__main__':
    sys.exit(main())
