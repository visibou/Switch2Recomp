"""Unattended game run: launch, take timed screenshots and F5 traces, quit, collect everything into one folder.

  python tools/run_game.py [--seconds 90] [--shots-at 20000,40000] [--trace-at 60000] [--script 30000:1] [--name tag]
                           [--replay games/MKW/recordings/rec_...]

The game exits by itself after --seconds (runtime option seconds=). Times are milliseconds since start. --script
holds controller buttons for 300 ms ("ms:hex,..."; bit 0 = A, see runtime/hid_layout.h). --replay drives with an
input recording instead (runtime/input_rec.c), from a copy of the save it was made with. The run ends up in
games/<Game>/runs/<time>_<name>/ with:
  summary.txt          crashes, aborts, stalls, GPU stats, untranslated shader opcodes, FPS
  game.log, performance.txt, threads.txt
  shot_*.png           screenshots (F3 and the final one)
  traces/N/            the F5 trace (trace.txt, shaders) + sheet_XX.png contact sheets of every draw's result
"""
import argparse, glob, os, re, shutil, struct, subprocess, sys, time, zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- images (no PIL: PPM in, PNG out)
def read_ppm(path):
    d = open(path, 'rb').read()
    tok = []; i = 0
    while len(tok) < 4:
        while d[i:i + 1].isspace(): i += 1
        if d[i:i + 1] == b'#':
            while d[i:i + 1] not in (b'\n', b''): i += 1
            continue
        j = i
        while not d[j:j + 1].isspace(): j += 1
        tok.append(d[i:j]); i = j
    if tok[0] != b'P6' or int(tok[3]) != 255: raise ValueError('%s: not an 8 bit P6 ppm' % path)
    w, h = int(tok[1]), int(tok[2])
    return w, h, d[i + 1:i + 1 + w * h * 3]


def write_png(path, w, h, rgb):
    raw = b''.join(b'\0' + rgb[y * w * 3:(y + 1) * w * 3] for y in range(h))
    def chunk(t, b): return struct.pack('>I', len(b)) + t + b + struct.pack('>I', zlib.crc32(t + b) & 0xFFFFFFFF)
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def scale(w, h, rgb, tw):
    """nearest neighbour to width tw"""
    th = max(1, h * tw // w)
    out = bytearray(tw * th * 3)
    for y in range(th):
        sy = y * h // th; row = sy * w * 3; o = y * tw * 3
        for x in range(tw):
            s = row + (x * w // tw) * 3; out[o:o + 3] = rgb[s:s + 3]; o += 3
    return tw, th, bytes(out)


def contact_sheets(files, out_prefix, cols=6, tw=256, per=48):
    """files: ppm paths -> out_prefix_XX.png grids with a white strip between cells (labels are in index.txt)"""
    sheets = []
    for k in range(0, len(files), per):
        cells = []
        for p in files[k:k + per]:
            try: cells.append(scale(*read_ppm(p), tw))
            except Exception: cells.append((tw, 8, bytes(tw * 8 * 3)))
        ch = max(c[1] for c in cells); rows = (len(cells) + cols - 1) // cols
        W = cols * (tw + 2); H = rows * (ch + 2)
        img = bytearray(b'\xff' * (W * H * 3))
        for n, (w, h, rgb) in enumerate(cells):
            x0 = (n % cols) * (tw + 2); y0 = (n // cols) * (ch + 2)
            for y in range(h):
                o = ((y0 + y) * W + x0) * 3; img[o:o + w * 3] = rgb[y * w * 3:(y + 1) * w * 3]
        name = '%s_%02d.png' % (out_prefix, k // per)
        write_png(name, W, H, bytes(img)); sheets.append(name)
        with open(os.path.splitext(name)[0] + '.txt', 'w') as f:
            f.write('row-major, %d per row:\n' % cols + '\n'.join(os.path.basename(p) for p in files[k:k + per]) + '\n')
    return sheets


# ---------------------------------------------------------------- log summary
def summarize(log, perf):
    L = open(log, encoding='utf-8', errors='replace').read().splitlines() if os.path.isfile(log) else []
    s = []
    def grab(title, rx, n=12, tail=False):
        m = [l for l in L if re.search(rx, l)]
        if m: s.append('== %s (%d)' % (title, len(m))); s.extend((m[-n:] if tail else m[:n])); s.append('')
    grab('crashes / aborts', r'\*\*\* (Crash|The game called)|FATAL', 6)
    for i, l in enumerate(L):   # the guest backtrace under the first crash/abort
        if re.search(r'\*\*\* (Crash|The game called)', l):
            s.append('== first crash context'); s.extend(L[max(0, i - 8):i + 22]); s.append(''); break
    grab('stalls', r'without a new image', 6, tail=True)
    grab('gpu stats', r'gpu: frames=', 4, tail=True)
    ops = {}
    for l in L:
        m = re.search(r'sm86: opcode ([0-9a-f]+) not translated', l)
        if m: ops[m.group(1)] = ops.get(m.group(1), 0) + 1
    if ops: s += ['== untranslated Ampere opcodes: ' + ' '.join(sorted(ops)), '']
    grab('switch 2 commands answered with zeros', r'\[switch 2\].*not (implemented|handled)', 40)
    grab('unknown services', r'not implemented\]', 30)
    grab('failed file opens', r'fs: Open.*-> (?!0\b)', 10)
    if os.path.isfile(perf):
        s.append('== performance.txt (last lines)')
        s.extend(open(perf, encoding='utf-8', errors='replace').read().splitlines()[-12:])
    return '\n'.join(s) + '\n'


# ---------------------------------------------------------------- run
# Written when something went wrong for real; the game keeps running after most of these (a crash on a runtime thread
# stops at a message box while the rest goes on), and ends with exit 0 when --seconds run out.
CRASH_RX = re.compile(rb'\*\*\* (Crash|The game called|Shutdown)|FATAL:')
MIN_FREE_GB = {'D:': 20, 'C:': 5}   # a full D: cost us the log of the only crash of c145e6e (2026-11-04)


def low_disk(path):
    for drive in set([os.path.splitdrive(os.path.abspath(path))[0].upper()]) | set(MIN_FREE_GB):
        if not os.path.exists(drive + os.sep): continue
        free = shutil.disk_usage(drive + os.sep).free / 2**30
        if free < MIN_FREE_GB.get(drive, 20): return '%s has %.1f GB free, %d needed' % (drive, free, MIN_FREE_GB.get(drive, 20))
    return ''


def keep(src, dst):
    """copy a run's file into its folder; a failure must not lose the rest (the source stays where it is)"""
    try:
        shutil.copyfile(src, dst)
    except OSError as e:
        print('COULD NOT KEEP %s: %s (the original stays, do not start another run before saving it)' % (src, e), flush=True)


def wait_watching(p, log, t0, timeout):
    """wait for the game, ending it as soon as its log reports a crash: returns the exit code or a text"""
    pos = 0
    while True:
        try: return p.wait(timeout=2)
        except subprocess.TimeoutExpired: pass
        if time.time() - t0 > timeout:
            subprocess.run(['taskkill', '/PID', str(p.pid), '/F'], capture_output=True)
            return 'timeout (killed)'
        try:
            if os.path.getmtime(log) < t0: continue     # still the last run's log
            with open(log, 'rb') as f:
                f.seek(pos); new = f.read()
            pos += len(new)
        except OSError:
            continue
        if CRASH_RX.search(new):
            time.sleep(1)                               # the rest of the report
            subprocess.run(['taskkill', '/PID', str(p.pid), '/F'], capture_output=True)
            return 'crash (killed)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', default=os.path.join(ROOT, 'games', 'MKW', 'MKW.exe'))
    ap.add_argument('--seconds', type=int, default=90)
    ap.add_argument('--shots-at', default='')
    ap.add_argument('--trace-at', default='')
    ap.add_argument('--script', default='')
    ap.add_argument('--name', default='run')
    ap.add_argument('--replay', default='', help='an input recording folder (SWR_INPUT_REC=1): drive with it')
    ap.add_argument('--extra', action='append', default=[], help='more runtime options, key=value')
    a = ap.parse_args()
    exe = os.path.abspath(a.game); gdir = os.path.dirname(exe); base = os.path.splitext(os.path.basename(exe))[0]
    why = low_disk(gdir)
    if why: print('NOT STARTED: ' + why, flush=True); sys.exit(3)
    # Only a leftover game from this folder is ended: a game started from another folder (the user's install while they play) is
    # never touched, and nothing starts beside it (it owns the control pipe and the GPU)
    ps = subprocess.run(['powershell', '-NoProfile', '-Command', "Get-Process -Name '%s' -ErrorAction SilentlyContinue | ForEach-Object { \"$($_.Id)|$($_.Path)\" }" % base],
                        capture_output=True, text=True).stdout
    for ln in ps.splitlines():
        if '|' not in ln: continue
        pid, path = ln.split('|', 1)
        if path and os.path.normcase(os.path.dirname(os.path.abspath(path))) != os.path.normcase(gdir):
            print('NOT STARTED: %s is running from %s (the user may be playing): not touched' % (base, os.path.dirname(path)), flush=True); sys.exit(4)
        subprocess.run(['taskkill', '/PID', pid.strip(), '/F'], capture_output=True)
    time.sleep(1)
    for old in ['frames.csv', 'trace.json', 'composites.csv', 'nvn_census.tsv'] + glob.glob(os.path.join(gdir, 'vma_*.json')):   # this run's only
        try: os.remove(os.path.join(gdir, old))
        except OSError: pass
    out = os.path.join(gdir, 'runs', time.strftime('%Y%m%d_%H%M%S') + '_' + a.name)
    os.makedirs(out)
    cap = os.path.join(gdir, 'captures')
    if os.path.isdir(cap):   # keep, don't mix with this run
        try: shutil.move(cap, os.path.join(out, 'captures_before'))
        except OSError:      # a folder in it is open elsewhere (a shell, an explorer window): leave it and empty what we can
            for f in glob.glob(os.path.join(cap, '*')):
                try: shutil.rmtree(f) if os.path.isdir(f) else os.remove(f)
                except OSError: pass
    args = [exe, '--seconds=%d' % a.seconds, '--shot=captures/final', '--fullscreen=0']
    if a.shots_at: args.append('--shots_at=' + a.shots_at)
    if a.trace_at: args.append('--trace_at=' + a.trace_at)
    if a.script: args.append('--script=' + a.script)
    if a.replay:   # from a copy of the recording's save, in the run folder: the player's own save stays untouched
        rec = os.path.abspath(a.replay)
        if os.path.isdir(os.path.join(rec, 'save')): shutil.copytree(os.path.join(rec, 'save'), os.path.join(out, 'save'))
        else: os.makedirs(os.path.join(out, 'save'))
        shutil.copyfile(os.path.join(rec, 'input.txt'), os.path.join(out, 'input.txt'))
        os.environ['SWR_INPUT_PLAY'] = rec
        os.environ['SWR_SAVE_DIR'] = os.path.join(out, 'save')
    elif not os.environ.get('SWR_SAVE_DIR') and os.path.isdir(os.path.join(gdir, 'save_copy')):
        # a test folder's data may be a link to the installed game's: without this, a run would open the player's own save
        os.environ['SWR_SAVE_DIR'] = os.path.join(gdir, 'save_copy')
    args += ['--' + e for e in a.extra]
    print('run:', ' '.join(args[1:]), flush=True)
    t0 = time.time()
    p = subprocess.Popen(args, cwd=gdir)
    rc = wait_watching(p, os.path.join(gdir, base + '.log'), t0, a.seconds + 120)
    print('exit: %s after %.0f s' % (rc, time.time() - t0), flush=True)
    for src, dst in ((base + '.log', 'game.log'), ('performance.txt', 'performance.txt'), ('threads.txt', 'threads.txt'),
                     ('frames.csv', 'frames.csv'), ('trace.json', 'trace.json'), ('composites.csv', 'composites.csv'), ('nvn_census.tsv', 'nvn_census.tsv'),
                     (os.path.join(base + '_eden', 'log', 'eden_log.txt'), 'eden_log.txt')):
        if os.path.isfile(os.path.join(gdir, src)): keep(os.path.join(gdir, src), os.path.join(out, dst))
    for f in glob.glob(os.path.join(gdir, 'vma_*.json')):   # SWR_VMA_DUMP
        shutil.move(f, os.path.join(out, os.path.basename(f)))
    if os.path.isfile(os.path.join(cap, 'frames.bin')):   # SWR_CAPTURE: tools/frame_check.py <run folder>
        shutil.move(os.path.join(cap, 'frames.bin'), os.path.join(out, 'frames.bin'))
    if os.path.isdir(cap):
        for f in glob.glob(os.path.join(cap, '*.png')):   # SWR_GPU=nvn writes PNGs
            shutil.copyfile(f, os.path.join(out, 'shot_' + os.path.basename(f)))
        for f in glob.glob(os.path.join(cap, '*.ppm')):
            try: write_png(os.path.join(out, 'shot_' + os.path.splitext(os.path.basename(f))[0] + '.png'), *read_ppm(f))
            except Exception as e: print('  %s: %s' % (f, e))
        # every screenshot in time order on one small sheet (shots.png + shots.txt), the one to look at first
        shots = sorted(glob.glob(os.path.join(cap, 'screenshot_*.ppm'))) + glob.glob(os.path.join(cap, 'final.ppm'))
        if shots: contact_sheets(shots, os.path.join(out, 'shots'), cols=3, tw=480, per=12)
        tr = os.path.join(cap, 'traces')
        if os.path.isdir(tr):
            shutil.copytree(tr, os.path.join(out, 'traces'))
            for d in sorted(glob.glob(os.path.join(out, 'traces', '*'))):
                draws = sorted(glob.glob(os.path.join(d, 'd*.ppm')))
                if draws: print('  trace %s: %d draws, %d sheets' % (os.path.basename(d), len(draws), len(contact_sheets(draws, os.path.join(d, 'sheet')))))
        if os.path.isfile(os.path.join(cap, 'threads.txt')): shutil.copyfile(os.path.join(cap, 'threads.txt'), os.path.join(out, 'threads_f5.txt'))
    try:
        with open(os.path.join(out, 'summary.txt'), 'w', encoding='utf-8') as f:
            f.write('exit: %s\nargs: %s\n\n' % (rc, ' '.join(args[1:])) + summarize(os.path.join(out, 'game.log'), os.path.join(out, 'performance.txt')))
    except OSError as e:
        print('COULD NOT WRITE summary.txt: %s' % e, flush=True)
    print('results:', out)


if __name__ == '__main__':
    main()
