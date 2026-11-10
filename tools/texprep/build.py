"""Builds tools/texprep/texprep.dll and texkey.dll with GCC from the runtime's own code (vk_astc.cpp, astcenc, nvn_tex.c, s2tex.c).

  python tools/texprep/build.py

Compiles the NVN layer first if it is not built (tools/build_nvn.py makes the vk_astc and astcenc objects this links)."""
import glob, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
import s2r_paths as P


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, env=P.env_with_gcc(), capture_output=True, text=True)
    if r.returncode: sys.exit('failed: %s\n%s' % (' '.join(os.path.basename(c) for c in cmd[:2]), (r.stdout + r.stderr)[-3000:]))


def main():
    if not P.gcc_bin(): sys.exit('GCC is missing: double-click Install.bat once, then run this again.')
    nvn = os.path.join(ROOT, 'build', 'nvn')
    objs = [os.path.join(nvn, 'vk_vk_astc_cpp.o')] + sorted(glob.glob(os.path.join(nvn, 'vk_third_party_*.o')))
    if not os.path.isfile(objs[0]) or len(objs) < 2:
        print('building the NVN layer first (tools/build_nvn.py)...', flush=True); run([sys.executable, os.path.join(ROOT, 'tools', 'build_nvn.py')])
        objs = [objs[0]] + sorted(glob.glob(os.path.join(nvn, 'vk_third_party_*.o')))
    r = os.path.join(ROOT, 'runtime', 'nvn'); t = os.path.join(ROOT, 'build', 'texprep'); os.makedirs(t, exist_ok=True)
    inc = ['-I' + os.path.join(r, 'gfx'), '-I' + os.path.join(r, 's2tex'), '-I' + os.path.join(r, 'vk')]
    mine = []
    for src in (os.path.join(HERE, 'texprep.c'), os.path.join(r, 'gfx', 'nvn_tex.c'), os.path.join(r, 's2tex', 's2tex.c')):
        o = os.path.join(t, os.path.splitext(os.path.basename(src))[0] + '.o'); run([P.gcc_tool('gcc'), '-O2', '-c', src, '-o', o] + inc); mine.append(o)
    run([P.gcc_tool('g++'), '-O2', '-shared', '-static', '-o', os.path.join(HERE, 'texprep.dll')] + mine + objs + ['-lshell32', '-lole32', '-luuid'])
    run([P.gcc_tool('gcc'), '-O2', '-shared', '-o', os.path.join(HERE, 'texkey.dll'), os.path.join(HERE, 'texkey.c'), os.path.join(r, 'gfx', 'nvn_tex.c'), os.path.join(r, 's2tex', 's2tex.c')] + inc[:2])
    print('built', os.path.join(HERE, 'texprep.dll'))


if __name__ == '__main__':
    main()
