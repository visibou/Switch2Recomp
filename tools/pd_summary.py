"""A frame's pass dump (NXRT_VK_PASSDUMP) as a timeline: python tools/pd_summary.py <run folder> [substring]
Lines: passes with their draws, dispatches with the images they write, copies. Consecutive like lines fold."""
import os, re, sys
L = [re.sub(r'^\[[^]]*\] \[pd\] ', '', l.rstrip()) for l in open(os.path.join(sys.argv[1], 'game.log'), encoding='utf-8', errors='replace') if '[pd]' in l]
want = sys.argv[2] if len(sys.argv) > 2 else ''
out = []; cur = None
for i, l in enumerate(L):
    if l.startswith('pass '):
        cur = {'k': 'pass', 'l': l[:110], 'n': 0, 'vs': {}, 'prim': 0}; out.append(cur)
    elif l.startswith('draw') and cur and cur['k'] == 'pass':
        cur['n'] += 1; m = re.search(r'fs=(\w+)', l); k = m.group(1) if m else '?'; cur['vs'][k] = cur['vs'].get(k, 0) + 1
    elif l.startswith('dispatch'):
        cur = {'k': 'cs', 'l': l[:80], 'imgs': []}; out.append(cur)
    elif cur and cur['k'] == 'cs' and ' img ' in l:
        m = re.search(r'img slot(\d+) tid=\d+ (ser\d+) fmt(\w+) (\S+)', l);
        if m: cur['imgs'].append(m.group(2) + ':' + m.group(4))
    elif l.startswith(('copy', 'clear', 'blit', 'resolve')):
        out.append({'k': 'x', 'l': l[:100]})
for e in out:
    if e['k'] == 'pass': s = '%s | %d draws, shaders %s' % (e['l'], e['n'], ' '.join('%s*%d' % (k[:8], v) for k, v in sorted(e['vs'].items(), key=lambda kv: -kv[1])[:4]))
    elif e['k'] == 'cs': s = e['l'] + ' -> ' + ' '.join(e['imgs'])
    else: s = e['l']
    if want in s: print(s[:210])
