"""build_probe.py: a test build of a game's recompiled code with probes (a debugging aid), into a test folder only.

  python tools/build_probe.py --probe 83A8F60[,ADDR...] --game "<your dump>/base" [--out games/MKWProbe]
                              [--from games/MKW] [--name MKW]

A probe makes the generated code call the runtime before that guest instruction (runtime/main.c swr_probe; it logs registers
and memory, see the SWR_PROBE_* options there). Same steps as switchrecomp.py (load, analyze with the same hints, generate C,
compile with GCC, link), but it copies no game data and writes nothing but <out>/<name>_build and <out>/<name>_code.dll. The
first run seeds <out>/<name>_build from <from>/<name>_build with timestamps kept, so only the chunks a probe changes are
compiled again. To try the DLL, copy it over games/MKWTest/<name>_code.dll after backing that up (never the installed game's)."""
import argparse, os, shutil, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import codegen, nxloader, switchrecomp as S
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import s2r_paths as P
os.environ['PATH'] = (P.gcc_bin() or '') + os.pathsep + os.environ.get('PATH', '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', action='append', default=[]); ap.add_argument('--game', required=True, help='the dump folder (exefs and romfs inside)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'games', 'MKWProbe')); ap.add_argument('--from', dest='src', default=os.path.join(ROOT, 'games', 'MKW'))
    ap.add_argument('--name', default='MKW'); ap.add_argument('--jobs', type=int, default=12)
    a = ap.parse_args()
    say = lambda s: print(s, flush=True)
    os.makedirs(a.out, exist_ok=True)
    build = os.path.join(a.out, a.name + '_build')
    if not os.path.isdir(build):
        say('seeding %s from %s (timestamps kept)' % (build, a.src))
        shutil.copytree(os.path.join(a.src, a.name + '_build'), build, copy_function=shutil.copy2)
    cc = S.find_cc('x86_64-w64-mingw32-gcc', True)
    if not cc: raise SystemExit('GCC not found: run Install.bat once')
    mods, npdm, romfs = nxloader.load_program(a.game)
    hints = []
    hp = os.path.join(a.src, a.name + '.hints')
    if os.path.exists(hp):
        hints = [int(l.strip(), 16) for l in open(hp) if l.strip()]
    prog = codegen.Program(mods, hints)
    codegen.find_frame_mode_patches(prog, say)
    for p in a.probe:
        for s in p.split(','): prog.probes.add(int(s, 16))
    say('probes at %s' % ', '.join('%X' % x for x in sorted(prog.probes)))
    prog.analyze(say)
    t0 = time.time()
    files = S.gen_module(prog, mods, npdm, build, a.name, 30000, say)
    objs = S.compile_all(files, cc, '2', build, True, a.jobs, say)
    out = os.path.join(a.out, a.name + '_code.dll')
    S.link_module(objs, cc, out, True, say)
    say('done in %.0f s: %s' % (time.time() - t0, out))


if __name__ == '__main__':
    main()
