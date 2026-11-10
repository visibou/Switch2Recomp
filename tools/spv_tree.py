"""Read a shader backwards: python tools/spv_tree.py <disassembly> <id> [depth]
Prints the expression that gives a value. The translator's temporaries are function variables that are mostly stored once, so a load is followed
through its stores (more than one store is flagged). Calls, samples and uniform reads end a branch."""
import re, sys
L = open(sys.argv[1], encoding='utf-8', errors='replace').read().splitlines()
defs, stores, names = {}, {}, {}
for i, l in enumerate(L):
    m = re.match(r'\s*(%\S+) = (Op\S+) ?(.*)', l)
    if m: defs[m.group(1)] = (m.group(2), m.group(3).split(), i)
    m = re.match(r'\s*OpStore (%\S+) (%\S+)', l)
    if m: stores.setdefault(m.group(1), []).append((m.group(2), i))
def short(i):
    return i
seen = {}
def walk(v, d, ind):
    pad = '  ' * ind
    if v not in defs: print(pad + v); return
    op, a, ln = defs[v]
    if op == 'OpLoad':
        p = a[1]
        if p in stores:
            s = stores[p]
            print(pad + '%s = var %s%s' % (v, p, ' (%d stores)' % len(s) if len(s) > 1 else ''))
            if d > 0:
                if p in seen: print(pad + '  ^ seen'); return
                seen[p] = 1
                for x, _ in s[:3]: walk(x, d - 1, ind + 1)
            return
        walk(p, d, ind); return
    if op == 'OpAccessChain':
        print(pad + '%s = %s[%s]' % (v, a[1], ','.join(a[2:]))); return
    if op in ('OpConstant', 'OpConstantComposite'):
        print(pad + '%s = const %s' % (v, ' '.join(a[1:]))); return
    if op == 'OpVariable': print(pad + '%s = variable %s' % (v, ' '.join(a))); return
    if op == 'OpExtInst':
        print(pad + '%s = %s' % (v, a[2])); args = a[3:]
    else:
        print(pad + '%s = %s %s' % (v, op[2:], ' '.join(x for x in a[1:] if not x.startswith('%')))); args = [x for x in a[1:] if x.startswith('%')]
    if d > 0 and not op.startswith('OpImage'):
        for x in args: walk(x, d - 1, ind + 1)
    elif op.startswith('OpImage'):
        for x in args[:4]: walk(x, min(d - 1, 2), ind + 1)
start = sys.argv[2]
if start in stores:
    for x, _ in stores[start]: walk(x, int(sys.argv[3]) if len(sys.argv) > 3 else 6, 0)
else: walk(start, int(sys.argv[3]) if len(sys.argv) > 3 else 6, 0)
