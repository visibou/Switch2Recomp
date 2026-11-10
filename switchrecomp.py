#!/usr/bin/env python3
# SwitchRecomp - Copyright 2026 MZ Labs
# Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
# Forks and derived works have to keep the credit to SwitchRecomp by MZ Labs and the startup screen.
"""
SwitchRecomp: static recompiler for Switch games, AArch64 -> C -> native.

The game code is translated to C ahead of time (plus a fallback interpreter built from the same
instruction templates). The runtime provides HLE kernel/services and a GL renderer for the GPU.

Usage:
  python switchrecomp.py GAME_FOLDER                   (with exefs/ and romfs/ extracted from your own dump)
  python switchrecomp.py game.nro                      (homebrew)
  python switchrecomp.py FOLDER -o Game --cc gcc -O 2 --jobs 8

Output (next to the executable):
  Game.exe             runtime (Windows) or Game (Linux)
  Game_code.dll/.so    the game code already translated to native
  Game_data/           exefs + romfs + save data
  Game.ini             options

LEGAL: does not include or download games, keys or firmware. Use your own copy.
"""
import argparse, os, sys, shutil, subprocess, time, hashlib, filecmp, re, struct, threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nxloader, codegen

RT = os.path.join(HERE, 'runtime')
RUNTIME_SRC = ['gpu_eden', 'trace', 'plat', 'mem', 'kernel', 'ipc', 'svc_core', 'audio', 'svc_hid', 'input_rec', 'svc_fs', 'svc_vi', 'nvdrv', 'svc_ns2', 'sm86', 'gpu', 'gpu_engines',
               'gpu_3d', 'loader', 'main', 'nvn_hook']
ABI = 1
# portable compilers next to us: the GCC package (gcc\, from Install.bat, here or beside this folder), w64devkit.
# Switch2Recomp builds with GCC only: TCC (SwitchRecomp's bundled default) built games that crash, and is not shipped
CC_DIRS = [os.path.join(HERE, *d) for d in (('gcc', 'bin'), ('gcc', 'Library', 'bin'), ('..', 'gcc', 'Library', 'bin'), ('w64devkit', 'bin'))]


def find_cc(pref=None, target_win=False):
    exe = '.exe' if sys.platform.startswith('win') else ''

    def look(c):
        p = shutil.which(c) or (c if os.path.isfile(c) else None)
        for d in CC_DIRS:
            if p: break
            q = os.path.join(d, c + exe)
            if os.path.isfile(q): p = q
        return p
    if pref:
        # an explicit --cc must not silently turn into another compiler
        if is_tcc(pref): raise SystemExit('TCC is not supported: games built with it crash. Use GCC (Install.bat downloads it).')
        return look(pref)
    if target_win:
        cands = ['x86_64-w64-mingw32-gcc', 'gcc', 'clang']
    else:
        cands = ['gcc', 'clang', 'cc']
    for c in cands:
        p = look(c)
        if p: return p
    return None


def is_tcc(cc): return 'tcc' in os.path.basename(cc).lower()


def tcc_b(cc):
    # bundled tcc keeps include\ and lib\ next to the exe
    d = os.path.dirname(os.path.abspath(cc))
    return ['-B', d] if os.path.isdir(os.path.join(d, 'include')) else []


def cc_env(cc):
    # a portable gcc runs cc1/as/ld from its own folder and needs its DLLs on PATH
    d = os.path.dirname(os.path.abspath(cc))
    return dict(os.environ, PATH=d + os.pathsep + os.environ.get('PATH', ''))


def run_cc(cmd, cc, **kw):
    # compilers print in the console code page: never let a decode error hide the real message
    r = subprocess.run(cmd, capture_output=True, text=True, errors='replace', env=cc_env(cc), **kw)
    return (r.stderr or r.stdout or 'exit code %d' % r.returncode) if r.returncode else None


# ======================================================================= codegen
def write_if_changed(p, text):
    # don't touch unchanged files so make-style reuse of the .o works
    try:
        with open(p, 'r', encoding='utf-8') as fi:
            if fi.read() == text: return False
    except (OSError, UnicodeDecodeError):
        pass
    with open(p, 'w', encoding='utf-8') as fo: fo.write(text)
    return True


def gen_module(prog, mods, npdm, outdir, title, chunk_instrs, say):
    funcs = prog.funcs
    order = sorted(funcs)
    # split by instruction count
    chunks = []; cur = []; n = 0
    for a in order:
        f = funcs[a]; cur.append(f); n += len(f.addrs)
        if n >= chunk_instrs: chunks.append(cur); cur = []; n = 0
    if cur: chunks.append(cur)
    files = []
    t0 = time.time()
    for ci, ch in enumerate(chunks):
        text = '\n'.join(codegen.emit_function(prog, f, funcs) for f in ch)
        # only prototype what this chunk references
        refs = set(re.findall(r'\bF_[0-9A-F]+\b', text))
        protos = ''.join('void %s(CPU* RESTRICT c, u64 e);\n' % r for r in sorted(refs))
        p = os.path.join(outdir, 'code_%03d.c' % ci)
        write_if_changed(p, '#include "cpu.h"\n' + protos + text + '\n')
        files.append(p)
        if ci % 20 == 0 or ci == len(chunks) - 1:
            say('  C generated: %d/%d chunks (%.0fs)' % (ci + 1, len(chunks), time.time() - t0))
    # main module: tables + interpreter
    ents = []
    for a in order:
        f = funcs[a]
        for l in sorted(f.leaders): ents.append((l, f.name))
    names = sorted({f.name for f in funcs.values()})
    idx = {n: i for i, n in enumerate(names)}
    L = ['#include "cpu.h"', '#include "cpu_helpers.c"']
    L += ['void %s(CPU* RESTRICT c, u64 e);' % n for n in names]
    L.append(codegen.gen_interp(step_mode=False))
    L.append('GuestFn swr_interp_entry;')
    L.append('static const GuestFn FNS[%d]={%s};' % (max(1, len(names)), ','.join(names) or '0'))
    L.append('static const u32 ENT_OFF[%d]={%s};' % (max(1, len(ents)), ','.join('%d' % (a - nxloader.CODE_BASE) for a, _ in ents) or '0'))
    L.append('static const u32 ENT_FN[%d]={%s};' % (max(1, len(ents)), ','.join(str(idx[n]) for _, n in ents) or '0'))
    L.append('static u64 ENT_ADDR[%d]; static GuestFn ENT_PTR[%d];' % (max(1, len(ents)), max(1, len(ents))))
    L.append('typedef struct SwrModInfo { const char* name; u64 base, size; u8 build_id[32]; u32 text_off, text_size, ro_off, ro_size, data_off, data_size, bss_size; u32 is_nro; } SwrModInfo;')
    mi = []
    for m in mods:
        bid = (m.build_id + b'\0' * 32)[:32]
        mi.append('{"%s",0x%XULL,0x%XULL,{%s},%d,%d,%d,%d,%d,%d,%d,%d}' % (
            m.name, m.base, m.size, ','.join(str(b) for b in bid), m.text[0], m.text[1], m.ro[0], m.ro[1], m.data[0], m.data[1],
            m.bss, 2 if getattr(m, 'deferred', False) else 1 if getattr(m, 'nro', False) else 0))
    L.append('static const SwrModInfo MODS[%d]={%s};' % (len(mods), ','.join(mi)))
    lo = min(m.base + m.text[0] for m in mods); hi = max(m.base + m.text[0] + m.text[1] for m in mods)
    np_ = nxloader.parse_npdm(npdm)
    L.append('''typedef struct SwrModule { u32 abi; u32 nmods; const SwrModInfo* mods; u32 nentries; const u64* entry_addr; const GuestFn* entry_fn;
  GuestFn interp; u64 code_lo, code_hi; const char* title; u64 title_id; u32 main_stack, main_prio, main_core;
  void (*init)(const RtAPI* rt); u64 (*step)(CPU* c, u32 w, u64 pc); } SwrModule;''')
    L.append('static const RtAPI* g_rt;')
    L.append('static void mod_init(const RtAPI* rt){ g_rt=rt; }')
    L.append('static SwrModule MOD;')
    L.append('SWR_EXPORT const SwrModule* swr_module(void){')
    L.append(' for(u32 i=0;i<%d;i++){ ENT_ADDR[i]=0x%XULL+ENT_OFF[i]; ENT_PTR[i]=FNS[ENT_FN[i]]; }' % (len(ents), nxloader.CODE_BASE))
    L.append(' swr_interp_entry=swr_interp;')
    L.append(' MOD.abi=%d; MOD.nmods=%d; MOD.mods=MODS; MOD.nentries=%d; MOD.entry_addr=ENT_ADDR; MOD.entry_fn=ENT_PTR; MOD.interp=swr_interp;' % (ABI, len(mods), len(ents)))
    L.append(' MOD.code_lo=0x%XULL; MOD.code_hi=0x%XULL; MOD.title="%s"; MOD.title_id=0x%XULL;' % (lo, hi, title.replace('"', ''), np_['title_id']))
    L.append(' MOD.main_stack=%d; MOD.main_prio=%d; MOD.main_core=%d; MOD.init=mod_init; return &MOD; }' % (np_['stack'], np_['prio'], np_['core']))
    p = os.path.join(outdir, 'module.c')
    write_if_changed(p, '\n'.join(L) + '\n')
    files.insert(0, p)
    say('C code: %d files, %.1f MB, %d entries' % (len(files), sum(os.path.getsize(x) for x in files) / 1e6, len(ents)))
    return files


def compile_all(files, cc, opt, outdir, target_win, jobs, say, extra_flags=()):
    objs = []
    base = [cc]
    if is_tcc(cc):
        flags = ['-w', '-I', RT] + tcc_b(cc)
    else:
        flags = ['-O' + opt, '-fno-strict-aliasing', '-fwrapv', '-w', '-I', RT, '-foptimize-sibling-calls', '-fno-semantic-interposition']
        if not target_win: flags += ['-fPIC', '-fvisibility=hidden']
        flags += list(extra_flags)
    t0 = time.time()
    done = [0]; lock = threading.Lock()

    # object names are keyed on compiler + flags + cpu.h, so tcc and gcc objects never mix
    with open(os.path.join(RT, 'cpu.h'), 'rb') as f1, open(os.path.join(RT, 'cpu_helpers.c'), 'rb') as f2:
        key = '|'.join(base + flags) + '|' + f1.read().decode('latin1') + '|' + f2.read().decode('latin1')
    try:
        key = key.encode('latin1')
    except UnicodeEncodeError:      # compiler path outside latin-1, e.g. a Cyrillic or CJK user name
        key = key.encode('utf-8')
    sig = hashlib.sha1(key).hexdigest()[:10]

    def one(src):
        obj = os.path.splitext(src)[0] + '.' + sig + '.o'
        err = None
        if not (os.path.exists(obj) and os.path.getmtime(obj) >= os.path.getmtime(src)):
            # write to a temp name: a build that is cancelled or crashes must not leave a
            # truncated .o that looks newer than its .c and gets reused next time
            tmp = obj + '.tmp'
            err = run_cc(base + flags + ['-c', src, '-o', tmp], cc)
            if err is None: os.replace(tmp, obj)
            elif os.path.exists(tmp): os.remove(tmp)
        with lock:
            done[0] += 1; n = done[0]
        if n % 10 == 0 or n == len(files):
            say('  compiled %d/%d (%.0fs)' % (n, len(files), time.time() - t0))
        return obj, err
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for obj, err in ex.map(one, files):
            if err:
                ex.shutdown(cancel_futures=True)   # don't keep compiling the rest
                raise SystemExit('Compilation failed:\n' + err[-4000:])
            objs.append(obj)
    return objs


def link_module(objs, cc, out, target_win, say):
    # big games have hundreds of objects and windows caps command lines at ~32K, so use an @rsp file
    # (gcc and tcc both take it) with short relative names
    bdir = os.path.dirname(os.path.abspath(objs[0])) if objs else '.'
    names = [os.path.relpath(os.path.abspath(o), bdir).replace('\\', '/') for o in objs]
    rsp = os.path.join(bdir, 'objects.rsp')
    with open(rsp, 'w') as fo:
        fo.write('\n'.join('"%s"' % n for n in names) + '\n')
    out = os.path.abspath(out)
    if os.path.isfile(cc): cc = os.path.abspath(cc)
    if is_tcc(cc):
        cmd = [cc, '-shared', '-o', out] + tcc_b(cc) + ['@objects.rsp']
    else:
        cmd = [cc, '-shared', '-o', out, '@objects.rsp']
        if not target_win: cmd += ['-lm']
        else: cmd += ['-static-libgcc']
    err = run_cc(cmd, cc, cwd=bdir)
    if err: raise SystemExit('Linking failed:\n' + err[-4000:])
    if is_tcc(cc):
        # tcc also writes an export list next to the dll that nothing uses
        d = os.path.splitext(out)[0] + '.def'
        if os.path.isfile(d): os.remove(d)
        if not target_win: elf_stack_not_exec(out)


def elf_stack_not_exec(path):
    """tcc 0.9.27 writes no PT_GNU_STACK, so the loader thinks the library needs an executable stack
    and glibc 2.41+ refuses to dlopen it. we append a copy of the program headers plus a
    read/write PT_GNU_STACK at the end of the file and point the ELF header there, glibc copies
    program headers that no segment maps"""
    with open(path, 'r+b') as f:
        h = f.read(64)
        if h[:4] != b'\x7fELF' or h[4] != 2: return
        phoff, = struct.unpack_from('<Q', h, 32)
        phentsize, phnum = struct.unpack_from('<HH', h, 54)
        f.seek(phoff)
        ph = f.read(phentsize * phnum)
        types = [struct.unpack_from('<I', ph, i * phentsize)[0] for i in range(phnum)]
        # already there, or a PT_PHDR that would point at the old copy
        if 0x6474e551 in types or 6 in types: return
        end = f.seek(0, 2)
        pad = -end % 8
        f.write(bytes(1) * pad + ph + struct.pack('<IIQQQQQQ', 0x6474e551, 6, 0, 0, 0, 0, 0, 16))
        f.seek(32)
        f.write(struct.pack('<Q', end + pad))
        f.seek(56)
        f.write(struct.pack('<H', phnum + 1))


def runtime_path(target_win, backend):
    return os.path.join(HERE, 'bin', 'runtime_%s_%s%s' % ('win64' if target_win else 'linux', backend,
                                                           '.exe' if target_win else ''))


def build_runtime(cc, out, target_win, backend, say, opt='2'):
    if is_tcc(cc):
        # bcn, SPIRV-Cross and OpenH264 are C++
        raise SystemExit('TCC cannot build the runtime: build it once with GCC (--build-runtime --cc gcc), '
                         'or use the prebuilt one from the release (%s).' % os.path.relpath(runtime_path(target_win, backend), HERE))
    gl = backend != 'headless'
    extra = ['gpu_gl', 'gl_load', 'shader', 'shader_proc', 'astc', 'glq', 'glq_gen'] if gl else ['gpu_nogl', 'shader_proc']
    srcs = [os.path.join(RT, s + '.c') for s in RUNTIME_SRC + extra] + [os.path.join(RT, 'backend_%s.c' % backend), os.path.join(RT, 'entry.c')]
    if gl: srcs += [os.path.join(RT, 'bcn', 'bcn.cpp'), os.path.join(RT, 'bcn', 'bcdec.cpp')]
    # Eglantine's Ampere -> Maxwell lowering (sm86.c, SWR_LOWER=egl) needs C++20, so it is compiled on its own
    egl_obj = None
    if gl:
        egl_obj = os.path.splitext(out)[0] + '_egl_lower.o'
        err = run_cc([cc, '-std=gnu++20', '-O' + opt, '-w', '-c', os.path.join(RT, 'egl_lower.cpp'), '-I', RT, '-o', egl_obj], cc)
        if err: raise SystemExit('Runtime build failed (egl_lower.cpp):\n' + err[-4000:])
        srcs += [egl_obj]
    # LSFG (uses the user's Lossless.dll), desktop GL only
    spvc = None
    if gl and backend in ('win32', 'x11'):
        spvc = os.path.join(RT, 'spvc', 'win' if target_win else 'lin', 'libspvc.a')
        if os.path.isfile(spvc): srcs += [os.path.join(RT, 'lsfg.c'), os.path.join(RT, 'lsfg_spvc.cpp')]
        else: spvc = None
    if gl and not spvc: srcs += [os.path.join(RT, 'lsfg_stub.c')]
    if target_win and backend == 'win32': srcs += [os.path.join(RT, 'rec_win32.c')]   # F7 recorder, Media Foundation
    # NVDEC/VIC video: link OpenH264 (BSD) statically if we have it for this target
    oh = os.path.join(RT, 'oh264', 'win' if target_win else 'lin')
    h264_libs = [os.path.join(oh, 'libdecoder.a'), os.path.join(oh, 'libcommon.a')]
    if not all(os.path.isfile(lib) for lib in h264_libs):
        h264_libs = [os.path.join(oh, 'libopenh264.a')]
    h264 = all(os.path.isfile(lib) for lib in h264_libs) and backend in ('win32', 'x11', 'headless')
    srcs += [os.path.join(RT, 'host1x.c')]
    if not target_win: srcs += [os.path.join(RT, 'glibc_compat.c')]   # runs on glibc 2.34+ (Ubuntu 22.04, Debian 12)
    # hwopus: libopus (BSD), same rule as OpenH264
    op = os.path.join(RT, 'opus', 'win' if target_win else 'lin')
    opus = os.path.isfile(os.path.join(op, 'libopus.a')) and backend in ('win32', 'x11', 'headless')
    # VP9 WebM movies: indexed from RomFS and decoded with libvpx (BSD), linked when runtime/vp9/<os>/libvpx.a exists
    vp = os.path.join(RT, 'vp9', 'win' if target_win else 'lin')
    vp9 = os.path.isfile(os.path.join(vp, 'libvpx.a')) and backend in ('win32', 'x11', 'headless')
    # Eden's GPU side (runtime/eden, SWR_GPU=eden): linked when tools/build_eden.sh has built it into build/eden
    eden_libs = []
    if target_win and backend == 'win32':
        eb = os.path.join(HERE, 'build', 'eden_' + os.environ.get('EDEN_FLAVOR', 'eden-ns2'))   # tools/build_eden.sh's output (EDEN_FLAVOR=eglantine: the experimental one)
        names = ['libswr_eden.a', 'video_core/libvideo_core.a', 'shader_recompiler/libshader_recompiler.a',
                 'video_core/gpu_logging/libgpu_logging.a', 'libeden_common.a', 'libeden_ext.a', 'libsirit_lib.a']
        if all(os.path.isfile(os.path.join(eb, n)) for n in names): eden_libs = [os.path.join(eb, n) for n in names]
    cmd = [cc, '-std=gnu17'] + ['-O' + opt, '-w', '-fno-strict-aliasing', *([] if os.environ.get('SWR_KEEP_SYMBOLS') else ['-Wl,-S']), '-Wl,-Map=' + os.path.splitext(out)[0] + '.map', '-I', RT, '-I', os.path.join(RT, 'bcn'), '-I', os.path.join(RT, 'spvc'), '-I', os.path.join(RT, 'oh264', 'include'), '-I', os.path.join(RT, 'opus', 'include')] + srcs + ['-o', out]
    if target_win and backend == 'win32': cmd += ['-DSWR_RECORD_AUDIO']
    if not h264: cmd += ['-DSWR_NO_H264']
    if not opus: cmd += ['-DSWR_NO_OPUS']
    if vp9: cmd += ['-DSWR_HAVE_VP9', '-I', os.path.join(RT, 'vp9', 'include')]
    if spvc: cmd += [spvc]
    if eden_libs: cmd += ['-DSWR_HAVE_EDEN', '-Wl,--start-group'] + eden_libs + ['-Wl,--end-group', '-lws2_32', '-ldwmapi', '-lntdll', '-luuid']
    nvn_lib = os.path.join(HERE, 'build', 'nvn', 'libnvn.a')   # tools/build_nvn.py: SWR_GPU=nvn
    if target_win and backend == 'win32' and os.path.isfile(nvn_lib): cmd += ['-DSWR_HAVE_NVN', nvn_lib, '-lbcrypt', '-luuid']
    if h264: cmd += h264_libs
    if opus: cmd += [os.path.join(op, 'libopus.a')]
    if vp9: cmd += [os.path.join(vp, 'libvpx.a')] + (['-lpthread'] if target_win else [])   # the MSYS2 build of win/libvpx.a uses winpthreads
    # the bundled libvpx.a was built against msvcrt and calls _setjmp; a ucrt toolchain (conda-forge gcc) only has
    # __intrinsic_setjmpex, which takes the same arguments
    if vp9 and target_win and 'ucrt' in subprocess.run([cc, '-dumpspecs'], capture_output=True, text=True, env=cc_env(cc)).stdout:
        cmd += ['-Wl,--defsym,_setjmp=__intrinsic_setjmpex']
    if spvc or h264 or egl_obj or eden_libs:
        # static on Linux too, so the runtime doesn't depend on the libstdc++ version of the distro
        cmd += ['-lstdc++'] if target_win else ['-Wl,-Bstatic', '-lstdc++', '-Wl,-Bdynamic', '-static-libgcc']
    if target_win:
        cmd += ['-static', '-lgdi32', '-lwinmm', '-luser32', '-lopengl32', '-lcomdlg32', '-lshell32', '-mwindows' if backend == 'win32' else '-mconsole']
        if backend == 'win32': cmd += ['-lmfplat', '-lmfreadwrite', '-lmfuuid', '-lole32']
    else:
        cmd += ['-lpthread', '-ldl', '-lm', '-rdynamic']
        if backend == 'x11': cmd += ['-lX11', '-lGL']
    err = run_cc(cmd, cc)
    if err: raise SystemExit('Runtime build failed:\n' + err[-4000:])


def sync_tree(src, dst):
    """copy the files that are new or changed (size or time); copy2 keeps the time for the next check"""
    # ROMFS trees can contain many large files. On Linux, let rsync compare and
    # transfer only changed files instead of opening and copying each one here.
    rsync = shutil.which('rsync') if sys.platform.startswith('linux') else None
    if rsync:
        os.makedirs(dst, exist_ok=True)
        cmd = [rsync, '-rlt', '--modify-window=2', '--itemize-changes',
               '--out-format=%i', '--', os.path.join(os.path.abspath(src), ''),
               os.path.join(os.path.abspath(dst), '')]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, errors='replace')
        if result.returncode:
            detail = result.stderr.strip() or 'no error details'
            raise RuntimeError('rsync failed while copying ROMFS: ' + detail)
        # Itemized regular files and symlinks count as copied entries; directory
        # creation lines do not.
        return sum(line.startswith(('>f', '<f', '>L', '<L', 'cL'))
                   for line in result.stdout.splitlines())

    # Keep the portable path for Windows and Linux installs without rsync.
    copied = 0
    for root, dirs, names in os.walk(src):
        droot = os.path.join(dst, os.path.relpath(root, src))
        os.makedirs(droot, exist_ok=True)
        for n in names:
            s = os.path.join(root, n); d = os.path.join(droot, n)
            ss = os.stat(s)
            try:
                ds = os.stat(d)
                if ds.st_size == ss.st_size and abs(ds.st_mtime - ss.st_mtime) < 2: continue
            except OSError:
                pass
            # a hard link when the dump is on the same drive: the game's data then takes no space twice (MKW: 21.7 GB). The game
            # only reads its romfs. Another drive, or a file system without links: a copy, as before
            try:
                if os.path.lexists(d): os.remove(d)
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
            copied += 1
    return copied


def copy_data(mods, npdm, romfs, entry, datadir, say):
    ex = os.path.join(datadir, 'exefs'); os.makedirs(ex, exist_ok=True)
    if os.path.isfile(entry) and entry.lower().endswith('.nro'):
        shutil.copyfile(entry, os.path.join(ex, 'main.nro'))
        if isinstance(romfs, (bytes, bytearray)):
            with open(os.path.join(datadir, 'romfs.bin'), 'wb') as fo: fo.write(romfs)
        # icon and title for the tool
        for which, name in ((0, 'icon.jpg'), (1, 'control.nacp')):
            blob = nxloader.nro_asset(mods[0], which)
            if blob: open(os.path.join(datadir, name), 'wb').write(blob)
        return
    # compare contents, not sizes: after a game update main.npdm (and sometimes an NSO) keeps its
    # size, and the runtime refuses an NSO whose build id differs from the recompiled one
    src = nxloader.find_exefs(entry)
    for n in os.listdir(src):
        s = os.path.join(src, n); d = os.path.join(ex, n)
        if os.path.isfile(s) and not (os.path.isfile(d) and filecmp.cmp(s, d, shallow=False)):
            shutil.copyfile(s, d)
    # icon and title for the tool, if the dump has the control data (control/ or next to exefs/)
    root = os.path.dirname(os.path.normpath(src))
    for d in (root, os.path.join(root, 'control')):
        for name in ('control.nacp', 'icon_AmericanEnglish.dat', 'icon.jpg', 'icon.png'):
            s = os.path.join(d, name)
            if os.path.isfile(s): shutil.copyfile(s, os.path.join(datadir, name))
    if isinstance(romfs, str):
        dst = os.path.join(datadir, 'romfs')
        if os.path.abspath(romfs) != os.path.abspath(dst):
            first = not os.path.exists(dst)
            if first: say('Copying romfs (may take a while)...')
            if sys.platform.startswith('linux') and not shutil.which('rsync'):
                say('rsync is unavailable; using the built-in ROMFS copy fallback')
            n = sync_tree(romfs, dst)
            if n and not first: say('romfs: %d changed files copied' % n)


def check_writable(*paths):
    # a running game locks its exe and dll: say so before spending minutes compiling
    for p in paths:
        if os.path.exists(p):
            try:
                with open(p, 'ab'): pass
            except OSError:
                raise SystemExit('%s is in use. Close the game and try again.' % p)


INI = """; %(title)s - SwitchRecomp settings
; This file is rewritten (organized, with every option explained) the first time the game starts.
[Display]
language=1
"""


def main():
    ap = argparse.ArgumentParser(description='Recompiles a Switch game into a native executable.')
    ap.add_argument('input', nargs='?', help='game folder (exefs/ + romfs/) or .nro file')
    ap.add_argument('-o', '--out', help='output name (no extension)')
    ap.add_argument('--outdir', help='output folder')
    ap.add_argument('--cc', help='C compiler (gcc, clang, tcc, x86_64-w64-mingw32-gcc)')
    ap.add_argument('-O', default='2', help='optimization level (0-3)')
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 4)
    ap.add_argument('--chunk', type=int, default=30000, help='instructions per C file')
    ap.add_argument('--target', choices=['auto', 'win64', 'linux'], default='auto')
    ap.add_argument('--backend', choices=['auto', 'win32', 'x11', 'headless'], default='auto')
    ap.add_argument('--native', action='store_true', help='optimize for THIS processor (-march=native)')
    ap.add_argument('--no-build', action='store_true', help='only generate the C code and copy the game data')
    ap.add_argument('--keep-c', action='store_true', help=argparse.SUPPRESS)   # the C is always kept, for rebuilds
    ap.add_argument('--interp-only', action='store_true', help='(diagnostics) no recompilation, everything interpreted')
    ap.add_argument('--probe', action='append', default=[], help='(diagnostics) guest address (hex) whose instruction first calls the runtime probe (SWR_PROBE_* options); repeatable')
    ap.add_argument('--runtime', help='use an already built runtime (.exe) instead of building it')
    ap.add_argument('--build-runtime', action='store_true', help='build bin/runtime_<target>_<backend> with GCC and exit')
    ap.add_argument('-q', '--quiet', action='store_true')
    a = ap.parse_args()
    say = (lambda m: None) if a.quiet else (lambda m: print(m, flush=True))
    target_win = a.target == 'win64' or (a.target == 'auto' and sys.platform.startswith('win'))
    backend = a.backend if a.backend != 'auto' else ('win32' if target_win else 'x11')
    if a.build_runtime:
        cc = find_cc(a.cc, target_win)
        if not cc: raise SystemExit('C compiler not found: %s' % (a.cc or 'gcc'))
        out = runtime_path(target_win, backend)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        check_writable(out)
        say('Compiling the runtime (%s) with %s...' % (backend, os.path.basename(cc)))
        build_runtime(cc, out, target_win, backend, say, a.O)
        say('Done: %s' % out)
        return
    if not a.input: ap.error('the game folder or .nro file is required')
    # check the compiler first, not after minutes of translation
    cc = None
    if not a.no_build:
        cc = find_cc(a.cc, target_win)
        if not cc: raise SystemExit('C compiler not found: %s' % a.cc if a.cc else 'No C compiler found: double-click Install.bat once (it downloads GCC).')
    say('SwitchRecomp - use your own copy of the game. No games or keys included.')
    try:
        mods, npdm, romfs = nxloader.load_program(a.input)
    except (ValueError, OSError) as e:
        raise SystemExit('Cannot load the game: %s' % e)
    for m in mods:
        say('  %-8s %010X  code %.1f MB  .eh_frame functions %d' % (m.name, m.base, m.text[1] / 1048576.0, len(m.fde)))
    info = nxloader.parse_npdm(npdm)
    # a folder name is used as is ("Game v1.0" must not lose ".0"), a file loses its extension
    src_name = os.path.basename(os.path.normpath(a.input))
    if os.path.isfile(a.input): src_name = os.path.splitext(src_name)[0]
    title = a.out or (info['name'] if info['name'] not in ('Application', '') else None) or src_name
    title = ''.join(ch for ch in title if ch.isalnum() or ch in '-_ ').strip() or 'Game'
    outdir = os.path.abspath(a.outdir or os.getcwd())
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, title)
    exe = base + ('.exe' if target_win else '')
    modout = base + ('_code.dll' if target_win else '_code.so')

    if not a.no_build: check_writable(exe, modout)
    hints = []
    hp = base + '.hints'
    if os.path.exists(hp):
        with open(hp) as fi:
            for ln in fi:
                try: hints.append(int(ln.strip(), 16))
                except ValueError: pass
        say('Previous execution hints: %d' % len(hints))
    prog = codegen.Program(mods, hints)
    # a game's own patches, beside its build like its hints (<name>.patches.py with apply(prog, say)): none are built into Switch2Recomp
    pp = base + '.patches.py'
    if os.path.exists(pp):
        import importlib.util
        spec = importlib.util.spec_from_file_location('game_patches', pp); gp = importlib.util.module_from_spec(spec); spec.loader.exec_module(gp)
        gp.apply(prog, say)
    for p in a.probe:
        for s in p.split(','): prog.probes.add(int(s, 16))
    if prog.probes: say('Probes at %s (diagnostics)' % ', '.join('%X' % x for x in sorted(prog.probes)))
    if a.interp_only:
        prog.funcs = {}; prog.extra_entries = set()
    else:
        say('Analyzing the code...')
        prog.analyze(say)
    build = base + '_build'
    os.makedirs(build, exist_ok=True)
    say('Translating ARM64 to C...')
    files = gen_module(prog, mods, npdm, build, title, a.chunk, say)
    datadir = base + '_data'
    copy_data(mods, npdm, romfs, a.input, datadir, say)
    if not os.path.exists(base + '.ini'):
        with open(base + '.ini', 'w', encoding='utf-8') as fo: fo.write(INI % {'title': title})
    if a.no_build: return
    say('Compiling with %s (-O%s, %d at a time)...' % (os.path.basename(cc), a.O, a.jobs))
    extra = []
    if a.native and not is_tcc(cc): extra.append('-march=native')
    if os.environ.get('SWR_CFLAGS') and not is_tcc(cc): extra += os.environ['SWR_CFLAGS'].split()   # e.g. -march for someone else's cpu
    objs = compile_all(files, cc, a.O, build, target_win, a.jobs, say, extra)
    say('Linking %s ...' % os.path.basename(modout))
    link_module(objs, cc, modout, target_win, say)
    if a.runtime:
        shutil.copyfile(a.runtime, exe)
    else:
        pre = runtime_path(target_win, backend)
        if os.path.exists(pre):
            shutil.copyfile(pre, exe)
        else:
            say('Compiling the runtime (%s)...' % backend)
            build_runtime(cc, exe, target_win, backend, say)
    if not target_win: os.chmod(exe, 0o755)
    say('Done: %s' % exe)


if __name__ == '__main__':
    main()
