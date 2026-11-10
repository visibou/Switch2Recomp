"""Builds the NVN layer (runtime/nvn) into build/nvn/libnvn.a: only what changed. switchrecomp.py links it when it exists (SWR_HAVE_NVN).

  python tools/build_nvn.py [--jobs 8]

Stops at the first failure with the compiler's output. See docs/PLAN_NATIVE_GPU.md M3 and M5."""
import argparse, concurrent.futures as cf, glob, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2r_paths as P
EXT = P.EXTERNAL
N = os.path.join(ROOT, 'runtime', 'nvn')
OUT = os.path.join(ROOT, 'build', 'nvn')
BIN = P.gcc_bin() or ''
GLSLANG = P.glslang() or 'glslang'
os.environ['PATH'] = BIN + os.pathsep + os.environ['PATH']
INC = ['-I' + os.path.join(N, d) for d in ('host', 'gfx', 's2tex', 'gen', 'vk')] + ['-I' + OUT, '-I' + (P.vulkan_include() or os.path.join(EXT, 'deps', 'Vulkan-Headers', 'include'))]
AFTERMATH = os.path.join(EXT, 'aftermath', 'include')   # NVIDIA Nsight Aftermath headers (vk_aftermath.c); without them it builds as off
if os.path.isdir(AFTERMATH): INC.append('-I' + AFTERMATH)
CF = ['-O2', '-g0', '-w', '-fno-strict-aliasing', '-fms-extensions', '-D_USE_MATH_DEFINES', '-DWIN32_LEAN_AND_MEAN', '-DNXRT_GFX_VK', '-DVK_USE_PLATFORM_WIN32_KHR', '-include', 'nvn_guard.h',
      '-Dplat_sleep_ns=nvn_plat_sleep_ns', '-Dplat_yield=nvn_plat_yield', '-Dguest_ns=nvn_guest_ns', '-Dg_t0_ns=nvn_g_t0_ns']
ASTC = ['-DASTCENC_NEON=0', '-DASTCENC_SVE=0', '-DASTCENC_SSE=41', '-DASTCENC_AVX=2', '-DASTCENC_X86_GATHERS=0', '-DASTCENC_POPCNT=1', '-DASTCENC_F16C=1',
        '-mavx2', '-mfma', '-mpopcnt', '-mf16c', '-O2', '-w', '-g0']
SKIP = {'vk_window.c'}   # SDL3: replaced by host/vk_window_hwnd.c


def sources():
    s = [os.path.join(N, 'hle_nvn.c')]
    for d in ('host', 'gfx', 's2tex', 'gen', 'vk'):
        s += [f for f in glob.glob(os.path.join(N, d, '*.c')) + glob.glob(os.path.join(N, d, '*.cpp')) if os.path.basename(f) not in SKIP]
    s += glob.glob(os.path.join(N, 'vk', 'third_party', '*', '*.cpp'))
    return s


def shaders():
    made = []
    for f in sorted(glob.glob(os.path.join(N, 'vk', 'shaders', '*'))):
        name = os.path.basename(f)
        if name.startswith('test_') and name not in ('test_fill.comp',): continue
        var = name.replace('.', '_')
        out = os.path.join(OUT, var + '.h')
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(f): continue
        r = subprocess.run([GLSLANG, '-V', '--target-env', 'vulkan1.3', '--vn', var + '_spv', '-o', out, f], capture_output=True, text=True)
        if r.returncode: sys.exit('glslang failed on %s:\n%s' % (name, r.stdout + r.stderr))
        made.append(name)
    return made


def compile1(src):
    obj = os.path.join(OUT, os.path.relpath(src, N).replace(os.sep, '_').replace('.', '_') + '.o')
    deps = obj[:-2] + '.d'
    if os.path.exists(obj) and os.path.exists(deps):
        t = os.path.getmtime(obj)
        try: names = open(deps).read().replace(chr(92) + chr(10), ' ').split()[1:]
        except OSError: names = [src]
        names = [n for n in names if n.endswith(('.c', '.h', '.cpp', '.inc', '.hpp'))]
        if all(os.path.getmtime(n) <= t for n in names if os.path.exists(n)): return obj, None
    cpp = src.endswith('.cpp')
    tp = 'third_party' in src
    if tp and 'astcenc' in src: cmd = ['x86_64-w64-mingw32-g++', '-std=gnu++17'] + ASTC
    elif cpp: cmd = ['x86_64-w64-mingw32-g++', '-std=gnu++17', '-O2', '-w', '-g0', '-mavx2', '-mfma'] + INC
    else: cmd = ['x86_64-w64-mingw32-gcc', '-std=gnu17'] + CF + INC
    if os.path.sep + 'vk' + os.path.sep in src and not tp and not cpp: pass
    r = subprocess.run(cmd + ['-MMD', '-MF', deps, '-c', src, '-o', obj], capture_output=True, text=True)
    return obj, (src + ':\n' + (r.stdout + r.stderr)[-3000:]) if r.returncode else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=8); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    shaders()
    with cf.ThreadPoolExecutor(a.jobs) as ex: res = list(ex.map(compile1, sources()))
    bad = [e for _, e in res if e]
    if bad: print('\n'.join(bad[:6])); sys.exit('%d files failed' % len(bad))
    lib = os.path.join(OUT, 'libnvn.a')
    if not os.path.exists(lib) or any(os.path.getmtime(o) > os.path.getmtime(lib) for o, _ in res):
        if os.path.exists(lib): os.remove(lib)
        rsp = os.path.join(OUT, 'objs.rsp'); open(rsp, 'w').write('\n'.join(o.replace(chr(92), '/') for o, _ in res))
        subprocess.run(['x86_64-w64-mingw32-ar', 'rcs', lib, '@' + rsp], check=True)
    print('libnvn.a: %d objects' % len(res))


if __name__ == '__main__':
    main()
