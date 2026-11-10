"""Which instructions in a game's exefs SwitchRecomp's a64.py can't translate yet.
Usage: python a64_coverage.py GAME_FOLDER [RECOMPILER_FOLDER [SIMD_WORDS_OUT]]"""
import os, sys, array, collections, time
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import nxloader, a64
mods, npdm, romfs = nxloader.load_program(sys.argv[1])
CLS = [  # (name, mask, value)
 ('LSE atomic LDxxx/SWP', 0x3F200C00, 0x38200000),
 ('LSE CAS', 0x3FA07C00, 0x08A07C00),
 ('LSE CASP', 0xBFA07C00, 0x08207C00),
 ('LDAPR (RCpc)', 0x3FE0FC00, 0x38BFC000),
 ('LDAPUR/STLUR (RCpc2)', 0x3F200C00, 0x19000000),
 ('SDOT/UDOT vec', 0xBF20FC00, 0x0E809400),
 ('SDOT/UDOT elem', 0xBF00F400, 0x0F00E000),
 ('FP16 scalar', 0xFF200000, 0x1EE00000),
 ('AdvSIMD FP16 3same', 0x9F20C400, 0x0E400400),
 ('FJCVTZS', 0xFFFFFC00, 0x1E7E0000),
 ('FRINT32/64', 0xBF3FFC00|0, 0x1E284000),
 ('CRC32', 0x7FE0F000, 0x1AC04000),
 ('SHA/AES', 0xFF000000, 0x4E000000),
 ('BTI/PAC hint', 0xFFFFF01F, 0xD503201F),
 ('PAC data', 0xFFFFC000, 0xDAC10000),
 ('SYS/MRS/MSR', 0xFFC00000, 0xD5000000),
]
bad = collections.Counter(); badw = collections.defaultdict(list)
simd = set()       # words left to the runtime's swr_simd(): decoded, but only correct if the helper knows them
for m in mods:
    o = m.text[0]; n = m.text[1] // 4
    words = array.array('I', bytes(m.image[o:o + n * 4]))
    cnt = collections.Counter(words)
    t0 = time.time(); nb = 0; ns = 0
    for w, k in cnt.items():
        try: r = a64.emit_static(w, m.base + o)
        except Exception as e: r = None
        if isinstance(r, str) and 'swr_simd' in r:
            ns += k; simd.add(w)
        elif r is None or (isinstance(r, str) and 'UNDEF()' in r):
            nb += k
            cls = next((c for c, mk, v in CLS if w & mk == v), 'other %08X' % (w & 0xFF000000))
            bad[(m.name, cls)] += k
            if len(badw[(m.name, cls)]) < 6: badw[(m.name, cls)].append('%08X' % w)
    print('%-8s words %d unique %d undecoded %d simd-helper %d (%.1fs)' % (m.name, n, len(cnt), nb, ns, time.time() - t0), flush=True)
if len(sys.argv) > 3:
    with open(sys.argv[3], 'w') as f: f.write('\n'.join('%08X' % w for w in sorted(simd)) + '\n')
    print('%d unique simd-helper words written to %s' % (len(simd), sys.argv[3]))
print('\nUndecoded by class (counts include data-in-text noise):')
for (mn, c), k in sorted(bad.items(), key=lambda x: -x[1])[:60]:
    print('  %-8s %-28s %8d  e.g. %s' % (mn, c, k, ' '.join(badw[(mn, c)])))
