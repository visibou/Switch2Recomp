"""drawtest.py: does the GPU survive the first draw of a shader? One replay, watched until that shader's first draw and a little after.

  python tools/drawtest.py NAME --key SHADERKEY [--cache DIR] [--spv-dir DIR] [--env K=V ...] [--after 25] [--limit 480]

Prints one line: SURVIVED, FROZEN, NOT DRAWN or GAME ENDED, whether a module was swapped (NXRT_VK_SPV_DIR) and whether the native path ran. A shader
whose first draw hangs the GPU every time makes this a yes/no test of an idea in a few minutes (docs/DEBUG_TOOLS.md). Paths are made absolute (the game
does not resolve relative NXRT_VK_SPV_DIR paths). The game is ended afterwards, waiting for a frozen one to go."""
import argparse, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gamectl as G

def games():
    return subprocess.run('tasklist /FI "IMAGENAME eq MKW.exe"', shell=True, capture_output=True, text=True).stdout.lower().count('marioka')

def log():
    try: return open(os.path.join(G.TEST, 'MKW.log'), encoding='utf-8', errors='replace').read()
    except OSError: return ''

def frame():
    s = G.status()
    return int(s['frame']) if s else -1

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--key', required=True); ap.add_argument('--cache', default='')
    ap.add_argument('--spv-dir', default=''); ap.add_argument('--env', action='append', default=[]); ap.add_argument('--after', type=int, default=25)
    ap.add_argument('--limit', type=int, default=480); ap.add_argument('--replay', default='rec_20261103_214452')
    a = ap.parse_args()
    env = ['NXRT_VK_DRAWLOG=' + a.key, 'NXRT_VK_FLIGHT=0'] + a.env
    if a.cache: env.append('NXRT_SHADER_CACHE=' + os.path.abspath(a.cache))
    if a.spv_dir: env.append('NXRT_VK_SPV_DIR=' + os.path.abspath(a.spv_dir))
    while games(): subprocess.run('taskkill /F /IM MKW.exe', shell=True, capture_output=True); time.sleep(5)
    cmd = [sys.executable, os.path.join(HERE, 'gamectl.py'), 'start', '--replay', a.replay, '--seconds', str(a.limit), '--name', a.name]
    for e in env: cmd += ['--env', e]
    subprocess.run(cmd, capture_output=True, timeout=60)
    time.sleep(25)
    t0, res, mark = time.time(), 'NOT DRAWN', 'drawlog: first draw of ' + a.key
    while time.time() - t0 < a.limit:
        s = log()
        if 'stopped responding' in s: res = 'FROZEN (before a first-draw line)'; break
        if mark in s:
            t1, f0, res = time.time(), frame(), 'SURVIVED'
            while time.time() - t1 < a.after:
                time.sleep(3)
                if 'stopped responding' in log(): res = 'FROZEN'; break
            if res == 'SURVIVED' and frame() <= f0: res = 'FROZEN (the frame stood still)'
            break
        if not games() and time.time() - t0 > 60: res = 'GAME ENDED'; break
        time.sleep(3)
    s = log()
    print('%s: %s; swap %s; native %s' % (a.name, res, 'yes' if 'SPIR-V replaced' in s else 'no', 'yes' if 'NVN taken over' in s else 'NO'), flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, 'gamectl.py'), 'quit'], capture_output=True, timeout=20)
    t2 = time.time()
    while games() and time.time() - t2 < 540:
        subprocess.run('taskkill /F /IM MKW.exe', shell=True, capture_output=True); time.sleep(10)
    return 0 if res == 'SURVIVED' else 1

if __name__ == '__main__':
    sys.exit(main())
