"""recompile_game.py: recompile a Switch 2 (or Switch) game you dumped yourself, step by step, with plain-English messages.

  python tools/recompile_game.py "D:\\dumps\\My Game"                 recompile into games\\<name>
  python tools/recompile_game.py "D:\\dumps\\My Game" --name MyGame   choose the name
  python tools/recompile_game.py "D:\\dumps\\My Game" --check         only check that everything is ready (nothing is built)

What it does, in order:
  1. checks the dump: a folder with exefs\\ (main, main.npdm) and romfs\\ (or a folder holding one such folder, e.g. "base")
  2. checks the tools: GCC (always GCC, never TCC: TCC builds games that crash), Python, free disk space
  3. builds the runtime (the part shared by every game) if it is not built yet
  4. translates the game's ARM64 code to C and compiles it with GCC (the first time takes ~30-40 minutes for a big game; a progress
     bar shows how far it is and about how long is left); a second run only compiles what changed
  5. turns on the native GPU path (gpu=nvn) in the game's .ini
The whole output also goes to <game folder>\\recompile.log. Nothing here is specific to one game.
No games, keys or firmware are included: use only a dump of a game you own."""
import argparse, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import s2r_paths as P
GCC_DIR = P.gcc_bin() or os.path.join(ROOT, 'gcc', 'Library', 'bin')
GCC = os.path.join(GCC_DIR, 'x86_64-w64-mingw32-gcc.exe')
PY = P.python()
RUNTIME = os.path.join(ROOT, 'bin', 'runtime_win64_win32.exe')
NEED_GB = 25   # translated C, objects and the game's data link: ~1-2 GB for MKW's build, with room to spare; D: should also stay above 20 GB


class Stop(Exception):
    pass


def say(msg, log=None):
    print(msg, flush=True)
    if log: log.write(msg + '\n'); log.flush()


def find_dump(path):
    """The folder holding exefs\\ and romfs\\ (the given one, or one level down), or a reason why there is none."""
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise Stop('The game folder "%s" does not exist. Give the folder you extracted the game into (it has exefs and romfs inside).' % path)
    def ok(p): return os.path.isdir(os.path.join(p, 'exefs')) and os.path.isfile(os.path.join(p, 'exefs', 'main'))
    if ok(path): return path
    subs = [os.path.join(path, d) for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]
    found = [s for s in subs if ok(s)]
    if len(found) == 1: return found[0]
    if len(found) > 1:
        raise Stop('"%s" holds several game folders (%s). Give the one to recompile (with an update, the folder with the combined files).' % (path, ', '.join(os.path.basename(f) for f in found)))
    raise Stop('No exefs\\main in "%s". Point to the folder you extracted your dump into: it holds exefs\\ (with "main") and romfs\\.' % path)


def find_layers(given, dump):
    """An update next to the base (a dump laid out as <game>\\base, <game>\\updates\\<version> or update, <game>\\dlc): the newest update with
    its own exefs\\main, or None; and the DLC folders found (reported, not merged yet)"""
    root = os.path.dirname(dump) if os.path.normcase(os.path.abspath(given)) != os.path.normcase(dump) else dump
    ups = []
    for d in ('updates', 'update', 'patch'):
        p = os.path.join(root, d)
        if not os.path.isdir(p): continue
        if os.path.isfile(os.path.join(p, 'exefs', 'main')): ups.append((0, p))
        for v in os.listdir(p):
            q = os.path.join(p, v)
            if os.path.isfile(os.path.join(q, 'exefs', 'main')): ups.append((int(v) if v.isdigit() else 0, q))
    dlc = []
    for d in ('dlc', 'aoc'):
        p = os.path.join(root, d)
        if os.path.isdir(p): dlc += [os.path.join(p, v) for v in os.listdir(p)]
    return (max(ups)[1] if ups else None), dlc


def tree_bytes(p):
    n = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try: n += os.path.getsize(os.path.join(r, f))
            except OSError: pass
    return n


def layer(base, update, dst, log):
    """dst = base with update over it: exefs from the update (the code the update ships), romfs = base's files with the update's added or replaced.
    Files are hard links (same disk: nothing large is copied), copies otherwise. Built again each time, so a newer update is picked up"""
    if os.path.lexists(dst):
        is_junction = getattr(os.path, 'isjunction', lambda q: False)(dst)
        if os.path.islink(dst) or is_junction or not os.path.isfile(os.path.join(dst, '.s2r_layered')):
            raise Stop('%s exists and was not made by this tool: move it away (it is never deleted here), then run again.' % dst)
        shutil.rmtree(dst)   # this tool's own folder of hard links: removing a link leaves the dump's file as it is
    os.makedirs(os.path.join(dst, 'exefs'))
    open(os.path.join(dst, '.s2r_layered'), 'w').write('made by tools/recompile_game.py: base + update as hard links; safe to delete\n')
    for f in os.listdir(os.path.join(update, 'exefs')):
        shutil.copyfile(os.path.join(update, 'exefs', f), os.path.join(dst, 'exefs', f))
    copied = [0]
    def put(src, d):
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.exists(d): os.remove(d)
        try: os.link(src, d)
        except OSError: shutil.copyfile(src, d); copied[0] += 1
    n = 0
    for top in (os.path.join(base, 'romfs'), os.path.join(update, 'romfs')):
        for r, _, fs in os.walk(top):
            for f in fs:
                put(os.path.join(r, f), os.path.join(dst, 'romfs', os.path.relpath(os.path.join(r, f), top))); n += 1
    msg = '   layered %d files (%d copied instead of linked)' % (n, copied[0])
    print(msg, flush=True); log.write(msg + '\n')
    for top in (base, update):
        if os.path.isdir(os.path.join(top, 'control')): shutil.copytree(os.path.join(top, 'control'), os.path.join(dst, 'control'), dirs_exist_ok=True)


def free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return -1


def explain(line):
    """A plain-English reason for a known kind of failure, or None."""
    l = line.lower()
    if 'bad_alloc' in l or 'out of memory' in l or 'cannot allocate memory' in l:
        return 'The computer ran out of memory while compiling. Close other programs, or run again with fewer jobs (--jobs 4).'
    if 'no space left' in l or 'disk full' in l or 'not enough space' in l:
        return 'The disk is full. Free some space (this needs about %d GB) and run again: finished parts are kept.' % NEED_GB
    if 'cannot load the game' in l:
        return 'The game files could not be read. Check that exefs\\main is the game\'s own code (with an update: the update\'s exefs).'
    if 'permission denied' in l or 'access is denied' in l or 'busy' in l:
        return 'A file is in use. Close the game (or anything that has its folder open) and run again.'
    return None


def run(cmd, log, progress=False, env=None):
    """Runs cmd, showing its output (with a progress bar for the compile step); raises Stop with an explanation when it fails."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace', env=env)
    t0 = time.time(); last_bar = 0; reason = None; tail = []
    for line in p.stdout:
        line = line.rstrip('\n'); log.write(line + '\n'); tail = (tail + [line])[-30:]
        reason = reason or explain(line)
        m = re.search(r'(compiled|C generated:) (\d+)/(\d+)', line)
        if progress and m:
            n, total = int(m.group(2)), int(m.group(3))
            if time.time() - last_bar > 2 or n == total:
                last_bar = time.time(); el = time.time() - t0; frac = n / max(total, 1)
                left = el / frac - el if frac > 0.02 else 0
                what = 'compiling' if m.group(1) == 'compiled' else 'translating to C'
                bar = '#' * int(30 * frac) + '-' * (30 - int(30 * frac))
                print('\r  %s [%s] %d/%d%s   ' % (what, bar, n, total, ', about %d min left' % (left / 60 + 1) if left > 60 else ''), end='', flush=True)
                if n == total: print(flush=True)
            continue
        if not re.search(r'compiled \d+/|C generated', line): print('  ' + line, flush=True)
    rc = p.wait()
    if rc:
        print(flush=True)
        raise Stop((reason or 'The step failed (exit code %d). The last lines are above; the whole output is in %s.' % (rc, log.name)) + '\n' + '\n'.join('    ' + t for t in tail[-8:]))


def set_ini(ini, section, key, value):
    """key=value in [section] of an ini file (added if missing), keeping everything else as it is."""
    lines = open(ini, encoding='utf-8', errors='replace').read().splitlines() if os.path.exists(ini) else []
    out, cur, done, seen = [], None, False, False
    for ln in lines:
        s = ln.strip()
        if s.startswith('[') and s.endswith(']'):
            if cur == section and not done: out.append('%s=%s' % (key, value)); done = True
            cur = s[1:-1]; seen = seen or cur == section
        elif cur == section and re.match(r'\s*%s\s*=' % re.escape(key), ln):
            ln = '%s=%s' % (key, value); done = True
        out.append(ln)
    if not done:
        if not seen: out += ['', '[%s]' % section]
        out.append('%s=%s' % (key, value))
    open(ini, 'w', encoding='utf-8').write('\n'.join(out) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dump', help='the folder you extracted the game into (exefs\\ and romfs\\ inside)')
    ap.add_argument('--name', help='the game\'s name (letters, digits, - and _); default: the game\'s own name')
    ap.add_argument('--out', help='output folder (default games\\<name>)')
    ap.add_argument('--jobs', type=int, default=max(2, (os.cpu_count() or 4) - 2), help='compilers at a time (fewer if memory runs out)')
    ap.add_argument('--check', action='store_true', help='only check that everything is ready')
    ap.add_argument('--replace', action='store_true', help='rebuild a game that is already in the output folder (its saves and settings stay)')
    a = ap.parse_args()
    try:
        print('Switch2Recomp: recompile a game you dumped yourself. No games, keys or firmware are included.\n', flush=True)
        print('1. The dump', flush=True)
        dump = find_dump(a.dump)
        romfs = os.path.isdir(os.path.join(dump, 'romfs'))
        print('   %s: exefs found%s' % (dump, ', romfs found' if romfs else ' (no romfs: the game will find none of its data)'), flush=True)
        update, dlc = find_layers(a.dump, dump)
        if update: print("   update found: %s (its code is recompiled, its files go over the base's)" % update, flush=True)
        if dlc: print('   DLC found (%d): not added yet; the game runs without it' % len(dlc), flush=True)
        name = a.name or os.path.basename(os.path.normpath(a.dump if os.path.basename(dump).lower() in ('base', 'game', 'dump') else dump))
        name = ''.join(ch for ch in name if ch.isalnum() or ch in '-_') or 'Game'
        out = os.path.abspath(a.out or os.path.join(ROOT, 'games', name))
        print('   name %s, output %s' % (name, out), flush=True)
        if os.path.isfile(os.path.join(out, name + '.exe')) and not a.replace:
            raise Stop('A recompiled %s is already in %s. To rebuild it there, add --replace (its saves and settings stay; close the game first). '
                       'To keep it, choose another folder with --out.' % (name, out))

        print('2. The tools', flush=True)
        if not os.path.isfile(GCC):
            raise Stop('GCC is not installed (%s). Double-click Install.bat in the SwitchRecomp folder once (it downloads Python and GCC), then run this again.' % GCC)
        py = PY if os.path.isfile(PY) else sys.executable
        print('   GCC %s\n   Python %s' % (GCC, py), flush=True)
        gb = free_gb(os.path.dirname(out) if os.path.isdir(os.path.dirname(out)) else ROOT)
        data_gb = tree_bytes(os.path.join(dump, 'romfs')) / 1e9 if romfs and not os.path.isdir(os.path.join(out, name + '_data', 'romfs')) else 0   # copied into the game folder the first time
        need = NEED_GB + data_gb
        if 0 <= gb < need:
            raise Stop('Only %.0f GB free on the output disk; this game needs about %.0f GB (%.0f GB of game data + %d GB to build, and the disk should keep 20 GB free). '
                       'Free some space or choose another disk with --out.' % (gb, need, data_gb, NEED_GB))
        print('   %.0f GB free' % gb, flush=True)
        if a.check:
            print('\nReady. Run the same command without --check to recompile.', flush=True)
            return 0

        os.makedirs(out, exist_ok=True)
        log = open(os.path.join(out, 'recompile.log'), 'a', encoding='utf-8')
        log.write('\n==== %s: %s\n' % (time.strftime('%Y-%m-%d %H:%M'), ' '.join(sys.argv)))
        env = dict(os.environ, PATH=GCC_DIR + os.pathsep + os.environ.get('PATH', ''))

        print('3. The runtime (shared by every game)', flush=True)
        if os.path.isfile(RUNTIME):
            print('   already built: %s' % RUNTIME, flush=True)
        else:
            if os.path.isfile(os.path.join(HERE, 'build_nvn.py')):
                print('   building the graphics layer...', flush=True); run([sys.executable, os.path.join(HERE, 'build_nvn.py')], log, env=env)
            print('   building the runtime...', flush=True)
            run([py, os.path.join(ROOT, 'switchrecomp.py'), '--build-runtime', '--target', 'win64', '--backend', 'win32', '--cc', 'x86_64-w64-mingw32-gcc'], log, env=env)

        print('4. The game\'s code (ARM64 -> C -> native, with GCC)', flush=True)
        src = dump
        if update:
            print('   putting base and update together...', flush=True)
            src = os.path.join(out, name + '_dump'); layer(dump, update, src, log)
        t0 = time.time()
        run([py, os.path.join(ROOT, 'switchrecomp.py'), src, '-o', name, '--outdir', out, '--cc', 'x86_64-w64-mingw32-gcc', '-O', '2',
             '--jobs', str(a.jobs), '--runtime', RUNTIME], log, progress=True, env=env)
        print('   done in %d min' % ((time.time() - t0) / 60 + 0.5), flush=True)

        print('5. Settings', flush=True)
        ini = os.path.join(out, name + '.ini')
        set_ini(ini, 'Graphics', 'gpu', 'nvn')
        print('   %s: gpu=nvn (the native GPU path)' % ini, flush=True)

        cache = os.path.join(out, name + '_nvn', 'cache', 'vk')
        if not os.path.isdir(cache):
            print('6. Shaders\n   No translated shader cache in %s yet: the game draws nothing that needs a missing shader. Build the cache with the '
                  'installer step and tools/nxshader/nxshader.py build (docs/DEBUG_TOOLS.md).' % os.path.dirname(cache), flush=True)
        elif update:
            print("6. Shaders\n   A cache is there (%s). If it was built before this update, the update's new shaders are not in it: the game then skips "
                  'what they draw (the log names them: "is not in the shader cache"). Build the cache again from the updated game.' % os.path.dirname(cache), flush=True)
        exe = os.path.join(out, name + '.exe')
        print('\nFinished: %s\n'
              'The first time you play, every new shader and texture is prepared the first time it appears, so new places stutter once; '
              'later runs reuse what was prepared. The log of each play is %s.' % (exe, os.path.join(out, name + '.log')), flush=True)
        return 0
    except Stop as e:
        print('\nStopped: %s' % e, flush=True)
        return 1
    except KeyboardInterrupt:
        print('\nStopped by you. Run the same command again to continue: finished parts are kept.', flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
