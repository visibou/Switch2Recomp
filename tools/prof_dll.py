"""Names the samples a host profile has in a Windows DLL: python tools/prof_dll.py <host_profile.txt> <thread id> [dll, default ntdll.dll]. Uses the DLL's export table, so an offset gets the nearest export before it."""
import bisect, collections, os, re, struct, sys


def exports(path):
    b = open(path, 'rb').read()
    pe = struct.unpack_from('<I', b, 0x3c)[0]; nsec = struct.unpack_from('<H', b, pe + 6)[0]; opt = struct.unpack_from('<H', b, pe + 20)[0]
    secs = [struct.unpack_from('<IIII', b, pe + 24 + opt + 40 * i + 8) for i in range(nsec)]
    def off(rva):
        for vs, va, rs, ro in secs:
            if va <= rva < va + max(vs, rs): return rva - va + ro
        return None
    d = struct.unpack_from('<I', b, pe + 24 + 112)[0]; o = off(d)
    n_fn, n_nm, a_fn, a_nm, a_ord = struct.unpack_from('<IIIII', b, o + 20)
    out = []
    for i in range(n_nm):
        nm = off(struct.unpack_from('<I', b, off(a_nm) + 4 * i)[0]); e = b.index(0, nm)
        od = struct.unpack_from('<H', b, off(a_ord) + 2 * i)[0]; rva = struct.unpack_from('<I', b, off(a_fn) + 4 * od)[0]
        out.append((rva, b[nm:e].decode()))
    return sorted(out)


def main():
    prof, th = sys.argv[1], sys.argv[2]; dll = sys.argv[3] if len(sys.argv) > 3 else 'ntdll.dll'
    ex = exports(os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32', dll)); ad = [a for a, _ in ex]
    c = collections.Counter(); on = False; tot = 0
    for l in open(prof, errors='replace'):
        if l.startswith('thread '): on = l.startswith('thread %s ' % th); continue
        if not on: continue
        m = re.match(r'\s*(\d+) (\S+?)\+([0-9a-f]+)', l)
        if m:
            tot += int(m.group(1))
            if m.group(2).lower() == dll.lower():
                i = bisect.bisect_right(ad, int(m.group(3), 16)) - 1
                c[ex[i][1]] += int(m.group(1))
    print(tot, 'samples listed')
    for k, v in c.most_common(16): print('%6d %s' % (v, k))


if __name__ == '__main__':
    main()
