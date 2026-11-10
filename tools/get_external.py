"""Fetch Switch2Recomp's outside dependencies into external/ at the exact versions it is tested with. Run once after cloning:

  python tools/get_external.py            (needs git and Python 3.10+ on the PATH)

  nxnative        github.com/seedds/nxnative (MIT): game shader profiles, the shader split and the shader translator (dmsh_xlate_bin)
  pokopia-native  github.com/xFractals/pokopia-native (MIT; its ampere-lower is GPL): the shader installer pieces nxshader can use
  pylibs          capstone, lz4, backports.zstd from PyPI, for the shader split

Also checks for the Vulkan SDK (vulkan.lunarg.com: glslang, spirv-val, spirv-opt and the Vulkan headers), which is installed
separately, and GCC and Python, which Install.bat downloads. Already present and at the right version: left alone."""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2r_paths as P

REPOS = {   # folder: (url, commit)
    'nxnative': ('https://github.com/seedds/nxnative.git', '865cc35fad7034d17c2190f8ce349f61f5bcaba1'),
    'pokopia-native': ('https://github.com/xFractals/pokopia-native.git', '302a1ddf114e4e3c096c98bf9d578c33ec80e64f'),
}
PIP = ['capstone==5.0.9', 'lz4==4.4.5', 'backports.zstd==1.8.0']


def git(*a, cwd=None):
    return subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True)


def fetch_repo(name, url, commit):
    d = P.ext(name)
    if os.path.isdir(os.path.join(d, '.git')):
        head = git('rev-parse', 'HEAD', cwd=d).stdout.strip()
        if head == commit: print('  %-15s at %s: ok' % (name, commit[:9])); return True
        print('  %-15s at %s, wanted %s: fetching it' % (name, head[:9], commit[:9]))
    else:
        print('  %-15s fetching %s at %s' % (name, url, commit[:9]), flush=True)
        os.makedirs(d, exist_ok=True); git('init', '-q', cwd=d); git('remote', 'add', 'origin', url, cwd=d)
    r = git('fetch', '-q', '--depth', '1', 'origin', commit, cwd=d)
    if r.returncode: print('    failed: %s' % (r.stderr.strip()[:300])); return False
    r = git('checkout', '-q', '--detach', 'FETCH_HEAD', cwd=d)
    if r.returncode: print('    failed: %s' % (r.stderr.strip()[:300])); return False
    print('    done'); return True


def main():
    print('Switch2Recomp: outside dependencies into %s' % P.EXTERNAL, flush=True)
    if subprocess.run(['git', '--version'], capture_output=True).returncode:
        sys.exit('git is not on the PATH: install it (git-scm.com) and run this again.')
    os.makedirs(P.EXTERNAL, exist_ok=True)
    ok = all([fetch_repo(n, u, c) for n, (u, c) in REPOS.items()])
    pyl = P.ext('pylibs')
    have = os.path.isdir(pyl) and all(any(x.lower().startswith(p.split('==')[0].replace('.', '_').lower() + '-' + p.split('==')[1]) for x in os.listdir(pyl)) for p in PIP)
    if have: print('  %-15s %s: ok' % ('pylibs', ', '.join(PIP)))
    else:
        print('  %-15s installing %s' % ('pylibs', ', '.join(PIP)), flush=True)
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade', '--target', pyl] + PIP)
        ok = ok and r.returncode == 0
    print('\nAlso needed:')
    print('  Vulkan SDK      %s' % ('found (%s)' % P.VULKAN_SDK if P.VULKAN_SDK else ('not installed, but local tools found' if P.spirv_tool('spirv-val') and P.glslang() and P.vulkan_include()
                                                                                    else 'MISSING: install it from vulkan.lunarg.com (glslang, spirv-val, spirv-opt, headers)')))
    print('  GCC             %s' % (P.gcc_bin() or 'MISSING: double-click Install.bat'))
    print('  Python          %s' % P.python())
    print('\n' + ('Ready.' if ok else 'Something failed: see above, then run this again.'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
