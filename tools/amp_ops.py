"""List the 128 bit Ampere instructions of a dumped shader (captures/amp/*.bin): index, op12, raw words.
Usage: python amp_ops.py file.bin [op12,op12,...]   (no filter: opcode histogram)"""
import struct, sys, collections
d = open(sys.argv[1], 'rb').read()
want = set(int(x, 16) for x in sys.argv[2].split(',')) if len(sys.argv) > 2 else None
ins = [struct.unpack_from('<QQ', d, o) for o in range(0, len(d) - 15, 16)]
if want is None:
    c = collections.Counter(lo & 0xFFF for lo, hi in ins)
    print(' '.join('%03x:%d' % (k, v) for k, v in sorted(c.items())))
else:
    for i, (lo, hi) in enumerate(ins):
        if (lo & 0xFFF) in want:
            print('#%-5d op %03x  %016x %016x  ra=%d rb=%d d=%d ur32=%d ur64=%d' % (i, lo & 0xFFF, hi, lo, (lo >> 24) & 255, (lo >> 32) & 255, (lo >> 16) & 255, (lo >> 32) & 63, hi & 63))
