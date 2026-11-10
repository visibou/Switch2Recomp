"""Tests a renderer switch inside one race, with windows on both sides: python tools/ab_switch.py SWITCH VALUE_A VALUE_B [--window 15] [--windows 8] [--wait-frame 3000] [--limit-s 420] [--name ab] [--env K=V] [--extra k=v]

Starts the benchmark replay, waits for --wait-frame, then sets the switch to A, B, A, B... for --window seconds each and prints the FPS of each window (frames counted from the game's own frame number).
Windows differ for other reasons too, so read pairs next to each other, not the averages alone."""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gamectl as G
import panel_hunt as H


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('switch'); ap.add_argument('a'); ap.add_argument('b'); ap.add_argument('--window', type=float, default=15); ap.add_argument('--windows', type=int, default=8)
    ap.add_argument('--wait-frame', type=int, default=3000); ap.add_argument('--limit-s', type=int, default=420); ap.add_argument('--name', default='ab'); ap.add_argument('--env', action='append', default=[]); ap.add_argument('--extra', action='append', default=[])
    a = ap.parse_args()
    st = ['start', '--replay', 'rec_20261103_214452', '--seconds', str(a.limit_s), '--name', a.name, '--env', 'NXRT_VK_FLIGHT=0']
    for e in a.env: st += ['--env', e]
    for e in a.extra: st += ['--extra', e]
    print(H.ctl(*st), flush=True)
    print(H.ctl('wait-frame', a.wait_frame, '--timeout', a.limit_s - 100, timeout=a.limit_s), flush=True)
    res = {a.a: [], a.b: []}
    for w in range(a.windows):
        v = a.a if w % 2 == 0 else a.b
        H.ctl('set', a.switch, v); time.sleep(1.5)
        s = G.status()
        if not s: print('the game has gone'); break
        f0 = int(s['frame']); t0 = time.time(); time.sleep(a.window)
        s = G.status()
        if not s: print('the game has gone'); break
        fps = (int(s['frame']) - f0) / (time.time() - t0); res[v].append(fps)
        print('%s=%s: %.1f FPS (frames %d..%s)' % (a.switch, v, fps, f0, s['frame']), flush=True)
    for v, x in res.items():
        if x: print('%s=%s: mean %.1f over %d windows' % (a.switch, v, sum(x) / len(x), len(x)))
    H.ctl('quit')


if __name__ == '__main__':
    sys.exit(main())
