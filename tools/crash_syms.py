"""Name the places in a run's crash reports: every exe offset in the game.log's "*** Last chance" and "*** Crash" reports,
resolved with nm on the exe that ran (a build with SWR_KEEP_SYMBOLS=1).

  python tools/crash_syms.py <run folder> [exe]
"""
import bisect, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host_prof_syms import load_syms, short, BASE


def main():
    run = sys.argv[1]
    exe = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'games', 'MKW', 'MKW.exe')
    lines = open(os.path.join(run, 'game.log'), encoding='utf-8', errors='replace').read().splitlines()
    addrs, names = load_syms(exe)
    if not addrs: sys.exit('no symbols in ' + exe)
    def name(off):
        a = BASE + off
        i = bisect.bisect_right(addrs, a) - 1
        return '%s +%#x' % (short(names[i]), a - addrs[i]) if i >= 0 else '?'
    for i, line in enumerate(lines):
        if not re.search(r'\*\*\* (Last chance|Crash|Shutdown)|exe offset|exe offsets|return addresses', line): continue
        print(line.strip()[:170])
        for m in re.finditer(r'\+([0-9a-f]{3,})', line):
            print('    +%-9s %s' % (m.group(1), name(int(m.group(1), 16))))


if __name__ == '__main__':
    main()
