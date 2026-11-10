"""The render thread's time outside the exe (waits, driver, C runtime) from a host_profile.txt, by the runtime functions on its stack:

  python tools/prof_focus.py <host_profile.txt> <exe built with SWR_KEEP_SYMBOLS=1> [frames, default 3]"""
import bisect, collections, re, subprocess, sys

NM = r'D:\\NS2\\Switch2Recomp\\gcc\\Library\\bin\\x86_64-w64-mingw32-nm.exe'


def main():
    prof, exe = sys.argv[1], sys.argv[2]; depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    syms = []
    for l in subprocess.run([NM, '-n', exe], capture_output=True, text=True, errors='replace').stdout.splitlines():
        p = l.split()
        if len(p) == 3 and p[1] in 'tT': syms.append((int(p[0], 16), p[2]))
    ad = [a for a, _ in syms]
    def nm(off):
        i = bisect.bisect_right(ad, 0x140000000 + off) - 1
        return syms[i][1] if i >= 0 else hex(off)
    c = collections.Counter(); tot = 0
    for l in open(prof, errors='replace'):
        m = re.match(r'stack (\d+) (.*)', l)
        if m:
            n = int(m.group(1)); tot += n
            c[' < '.join(nm(int(x, 16)) for x in re.findall(r'\+([0-9a-f]+)', m.group(2))[:depth] if int(x, 16))] += n
    print(tot, 'samples outside the exe')
    for k, v in c.most_common(30): print('%6d  %s' % (v, k))


if __name__ == '__main__':
    main()
