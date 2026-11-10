"""One command from a dumped game to a build that runs on the native GPU path, on any Switch 2 game:

  python tools/port_game.py "<dump>/base" NAME --outdir games/NAME [--skip-cpu] [--skip-shaders] [--jobs 8]

<dump>/base holds exefs/ and romfs/ from your own console. The steps, each kept if its result is there:
  1. CPU: switchrecomp.py lifts every module to C and GCC builds NAME_code.dll, about 30 minutes for a large game.
  2. Shaders found: tools/nxshader/run_split.py, every shader in the RomFS and in main, into NAME_nvn/split.
  3. Shaders translated: tools/nxshader/nxshader.py, to SPIR-V in NAME_nvn/cache.
  4. The ini gets gpu=nvn, so the exe needs no environment.
The title id comes from main.npdm and names the nxnative game profile the shader split reads. A game with no profile stops at step 2 and says so.
Nothing here is written per game: what a game needs goes in a config file beside it."""
import argparse, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARENT = os.path.dirname(ROOT)
sys.path.insert(0, HERE)
import s2r_paths as P
GCC = P.gcc_bin() or ''
PY = P.python()
NXN = P.ext('nxnative', 'dm')
PYLIBS = P.ext('pylibs')


def title_id(base):
    r = subprocess.run([sys.executable, os.path.join(HERE, 'npdm_info.py'), os.path.join(base, 'exefs', 'main.npdm')], capture_output=True, text=True)
    m = re.search(r'title ([0-9A-Fa-f]{16})', r.stdout)
    if not m: sys.exit('no title id in main.npdm: ' + r.stdout[:200])
    return m[1].upper()


def run(cmd, log, env=None, cwd=None):
    print('  ' + os.path.basename(cmd[0]) + ' ' + ' '.join(os.path.basename(c) for c in cmd[1:3]) + ' -> ' + log, flush=True)
    t = time.time()
    with open(log, 'w', encoding='utf8', errors='replace') as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    print('  exit %d after %.0f s' % (r.returncode, time.time() - t), flush=True)
    if r.returncode: sys.exit('failed, see ' + log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base'); ap.add_argument('name'); ap.add_argument('--outdir')
    ap.add_argument('--jobs', type=int, default=8); ap.add_argument('--skip-cpu', action='store_true'); ap.add_argument('--skip-shaders', action='store_true')
    ap.add_argument('--runtime', default=os.path.join(ROOT, 'bin', 'runtime_win64_win32.exe'))
    a = ap.parse_args()
    out = os.path.abspath(a.outdir or os.path.join(ROOT, 'games', a.name)); base = os.path.abspath(a.base); os.makedirs(out, exist_ok=True)
    tid = title_id(base); print(a.name, 'title', tid, flush=True)
    g = os.path.join(out, a.name + '_nvn'); split = os.path.join(g, 'split'); os.makedirs(os.path.join(split, 'pkg'), exist_ok=True)
    env = dict(os.environ, PATH=GCC + os.pathsep + os.environ.get('PATH', ''), PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    if not a.skip_cpu and not os.path.exists(os.path.join(out, a.name + '_code.dll')):
        print('1. CPU', flush=True)
        run([PY, os.path.join(ROOT, 'switchrecomp.py'), base, '-o', a.name, '--outdir', out, '--cc', 'x86_64-w64-mingw32-gcc', '-O', '2', '--jobs', str(a.jobs), '--runtime', a.runtime],
            os.path.join(out, 'port_cpu.log'), env, ROOT)
    if a.skip_shaders: return
    prof = os.path.join(NXN, 'games', tid)
    if not os.path.isdir(prof): sys.exit('no nxnative profile for %s: add %s (game.tsv) and run this again' % (tid, prof))
    main_mod = os.path.join(base, 'exefs', 'main')
    if not os.path.exists(os.path.join(split, 'shader_map.tsv')):
        print('2. Shaders found', flush=True)
        e = dict(env, DM_GAME=tid, J=str(a.jobs), PYTHONPATH=PYLIBS)
        run([sys.executable, os.path.join(HERE, 'nxshader', 'run_split.py'), os.path.join(base, 'romfs'), main_mod, os.path.join(split, 'pkg'), os.path.join(split, 'shader_map.tsv'),
             os.path.join(split, 'shader_facts.tsv'), os.path.join(split, 'shader_pairs.tsv'), os.path.join(split, 'pso_pairs.tsv')], os.path.join(split, 'split.log'), e, os.path.join(NXN, 'gfx', 'tools'))
    print('3. Shaders translated', flush=True)
    e = dict(env, PYTHONPATH=os.path.join(HERE, 'nxshader'))
    run([sys.executable, os.path.join(HERE, 'nxshader', 'nxshader.py'), '--cache', os.path.join(g, 'cache'), 'build', split, '--jobs', str(max(2, a.jobs // 2)), '--report', os.path.join(g, 'shader_report.txt')],
        os.path.join(g, 'shader_build.log'), e, ROOT)
    ini = os.path.join(out, a.name + '.ini')
    s = open(ini, encoding='utf8', errors='replace').read() if os.path.exists(ini) else '[Graphics]\n'
    s = re.sub(r'(?m)^gpu=.*$', 'gpu=nvn', s) if re.search(r'(?m)^gpu=', s) else s.replace('[Graphics]\n', '[Graphics]\ngpu=nvn\n', 1)
    open(ini, 'w', encoding='utf8').write(s)
    print('4. Done:', os.path.join(out, a.name + '.exe'), flush=True)


if __name__ == '__main__':
    main()
