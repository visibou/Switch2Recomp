"""Where Switch2Recomp's tools find GCC, Python and the outside dependencies, on any PC.

Looked for in this order: inside the repository (as Install.bat and tools/get_external.py put them: gcc/, python/, external/),
then beside it (../gcc, ../SwitchRecomp/python, ../external: an older layout), then the PATH and the Vulkan SDK (%VULKAN_SDK%).
S2R_EXTERNAL picks another external folder. Every tool imports this instead of naming a folder of its own."""
import os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)


def _first(*paths):
    for p in paths:
        if p and os.path.exists(p): return os.path.normpath(p)
    return None


EXTERNAL = os.path.normpath(os.environ.get('S2R_EXTERNAL') or _first(os.path.join(ROOT, 'external'), os.path.join(PARENT, 'external')) or os.path.join(ROOT, 'external'))
VULKAN_SDK = os.environ.get('VULKAN_SDK', '')


def gcc_bin():
    """The folder holding x86_64-w64-mingw32-gcc.exe, or None."""
    d = _first(os.path.join(ROOT, 'gcc', 'Library', 'bin', 'x86_64-w64-mingw32-gcc.exe'), os.path.join(PARENT, 'gcc', 'Library', 'bin', 'x86_64-w64-mingw32-gcc.exe'))
    if d: return os.path.dirname(d)
    w = shutil.which('x86_64-w64-mingw32-gcc')
    return os.path.dirname(w) if w else None


def gcc_tool(name):
    """A GCC tool (gcc, g++, nm, ar...) by its short name, with the mingw prefix."""
    b = gcc_bin()
    return os.path.join(b, 'x86_64-w64-mingw32-' + name + ('.exe' if os.name == 'nt' else '')) if b else 'x86_64-w64-mingw32-' + name


def python():
    """The Python that builds games: the one Install.bat downloads, else this one."""
    return _first(os.path.join(ROOT, 'python', 'python.exe'), os.path.join(PARENT, 'SwitchRecomp', 'python', 'python.exe')) or sys.executable


def ext(*parts):
    return os.path.join(EXTERNAL, *parts)


def glslang():
    return _first(ext('tools', 'glslang', 'bin', 'glslang.exe'), os.path.join(VULKAN_SDK, 'Bin', 'glslang.exe') if VULKAN_SDK else None) or shutil.which('glslang') or shutil.which('glslangValidator')


def vulkan_include():
    return _first(ext('deps', 'Vulkan-Headers', 'include'), os.path.join(VULKAN_SDK, 'Include') if VULKAN_SDK else None)


def spirv_tool(name):
    """spirv-val, spirv-opt, spirv-dis: NXSHADER_TOOLS, a local SPIRV-Tools build, the Vulkan SDK, then the PATH."""
    exe = name + ('.exe' if os.name == 'nt' else '')
    return _first(os.path.join(os.environ['NXSHADER_TOOLS'], exe) if os.environ.get('NXSHADER_TOOLS') else None, ext('spirv-tools-build', 'tools', exe),
                  os.path.join(VULKAN_SDK, 'Bin', exe) if VULKAN_SDK else None) or shutil.which(name)


def env_with_gcc(env=None):
    e = dict(env if env is not None else os.environ)
    b = gcc_bin()
    if b: e['PATH'] = b + os.pathsep + e.get('PATH', '')
    return e


if __name__ == '__main__':   # python tools/s2r_paths.py: what this PC has
    print('repository  ', ROOT); print('external    ', EXTERNAL, '(found)' if os.path.isdir(EXTERNAL) else '(missing: python tools/get_external.py)')
    print('GCC         ', gcc_bin() or 'missing: run Install.bat'); print('Python      ', python())
    print('glslang     ', glslang() or 'missing: install the Vulkan SDK'); print('Vulkan headers', vulkan_include() or 'missing: install the Vulkan SDK')
    for t in ('spirv-val', 'spirv-opt'): print('%-12s' % t, spirv_tool(t) or 'missing: install the Vulkan SDK')
