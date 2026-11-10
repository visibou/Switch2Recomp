import struct, sys
d = open(sys.argv[1], 'rb').read()
aoff, asz, coff, csz = struct.unpack_from('<IIII', d, 0x70)
a = d[aoff:aoff+asz]
print('ACI0' , a[:4], 'title %016X' % struct.unpack_from('<Q', a, 0x10)[0])
fo, fs, so, ss, ko, ks = struct.unpack_from('<IIIIII', a, 0x20)
sac = a[so:so+ss]; i = 0; names = []
while i < len(sac):
    c = sac[i]; n = (c & 7) + 1; names.append(('*' if c & 0x80 else '') + sac[i+1:i+1+n].decode()); i += 1 + n
print('services:', ' '.join(names))
kc = a[ko:ko+ks]
for j in range(0, len(kc), 4):
    v = struct.unpack_from('<I', kc, j)[0]
    t = (~v & (v + 1)).bit_length() - 1
    if t == 4:  # svc mask
        idx = v >> 29; bits = (v >> 5) & 0xFFFFFF
        print('svc group %d: %s' % (idx, ' '.join('%02X' % (idx*24+b) for b in range(24) if bits >> b & 1)))
    elif t == 3: print('thread info prio %d-%d cores %d-%d' % ((v>>4)&63, (v>>10)&63, (v>>16)&255, (v>>24)&255))
    elif t not in (32,): print('kcap type %d %08X' % (t, v))
