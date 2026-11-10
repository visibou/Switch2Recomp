"""Make a pixel shader output its own temporaries: python tools/spv_probe.py <cache vk folder> <key> <var r> <var g> <var b>   (ids as in spirv-dis, no %), or --undo.
The output stores of the translator load float temporaries; those three loads are pointed at the temporaries asked for. The original stays in <key>.spv.orig."""
import os, shutil, struct, sys
folder, key = sys.argv[1], sys.argv[2]
p = os.path.join(folder, key + '.spv'); o = p + '.orig'
if sys.argv[3] == '--undo':
    shutil.move(o, p); sys.exit()
if not os.path.exists(o): shutil.copy(p, o)
b = open(o, 'rb').read(); w = list(struct.unpack('<%dI' % (len(b) // 4), b))
new = [int(x) for x in sys.argv[3:6]]
# pointers of the output variables of float: OpAccessChain in the last stretch of main, then OpLoad, then OpStore
i = 5; loads = []; lastop = {}
while i < len(w):
    n, op = w[i] >> 16, w[i] & 0xffff
    if op == 65: lastop[w[i + 2]] = w[i + 3]          # OpAccessChain: result -> base
    if op == 61: lastop[-w[i + 2]] = i                # OpLoad: -result -> position
    if op == 62:                                      # OpStore pointer value
        base = lastop.get(w[i + 1]); at = lastop.get(-w[i + 2])
        if base is not None and at is not None: loads.append((base, at))
    i += n
# the first output variable's three colour stores are the last run of stores whose base is one id
bases = {}
for b0, at in loads: bases.setdefault(b0, []).append(at)
out = [v for k, v in bases.items() if len(v) in (3, 4)]
ats = out[0][:3]
print('patching loads at', ats, 'old vars', [w[a + 3] for a in ats])
for a, v in zip(ats, new): w[a + 3] = v
open(p, 'wb').write(struct.pack('<%dI' % len(w), *w))
