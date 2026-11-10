# SwitchRecomp - Copyright 2026 MZ Labs
# Licensed under the GNU Affero General Public License v3.0, see LICENSE.txt and NOTICE.txt.
"""compares the TAG.ppm of every test with the BASE.ppm of the same test, pixel by pixel.
usage: python3 tests/regress/cmpreg.py TAG BASE [tests folder]
a test passes when no pixel differs by more than 2/255 (driver rounding), it prints how many pixels changed otherwise"""
import os, sys

def ppm(p):
    b = open(p, 'rb').read()
    parts = b.split(maxsplit=4)
    w, h = int(parts[1]), int(parts[2])
    return w, h, parts[4][:w * h * 3]

tag, base = sys.argv[1], sys.argv[2]
root = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(__file__), 'programs')
same = diff = 0
for t in sorted(os.listdir(root)):
    a, b = os.path.join(root, t, tag + '.ppm'), os.path.join(root, t, base + '.ppm')
    if not os.path.exists(b): continue
    if not os.path.exists(a): print(f'{t:24} NO IMAGE (crashed or never presented, see {tag}.out)'); diff += 1; continue
    wa, ha, pa = ppm(a); wb, hb, pb = ppm(b)
    if (wa, ha) != (wb, hb): print(f'{t:24} SIZE {wa}x{ha} vs {wb}x{hb}'); diff += 1; continue
    n = sum(1 for i in range(0, len(pa), 3) if max(abs(pa[i + k] - pb[i + k]) for k in range(3)) > 2)
    if n: print(f'{t:24} DIFFERENT {n} pixels ({100.0 * n / (wa * ha):.2f}%)'); diff += 1
    else: print(f'{t:24} same'); same += 1
print(f'{same} same, {diff} different')
