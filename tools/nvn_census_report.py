"""Join a run's nvn_census.tsv with what the NVN layer implements (D3 of docs/PLAN_NATIVE_GPU.md).

  python tools/nvn_census_report.py <run folder>

Writes nvn_census_report.txt in the run: every entry the game asked for, used or not, whether runtime/nvn/hle_nvn.c, or else
pokopia-native's or nxnative's layer, implements it, calls per frame and the threads."""
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import s2r_paths as P


def impl(path):
    try: src = open(path, encoding='utf-8', errors='replace').read()
    except OSError: return set()
    return set(re.findall(r'^NVN(?:_NOPROTO)?\((\w+)', src, re.M))


def main():
    run = sys.argv[1]
    ours = impl(os.path.join(ROOT, 'runtime', 'nvn', 'hle_nvn.c'))
    poko = impl(P.ext('pokopia-native', 'windows', 'src', 'nxrt', 'hle', 'hle_nvn.c'))
    nxn = impl(P.ext('nxnative', 'dm', 'gfx', 'nvn', 'hle_nvn.c'))
    rows = []; head = ''
    for line in open(os.path.join(run, 'nvn_census.tsv'), encoding='utf-8'):
        if line.startswith('#'): head = line.strip(); continue
        if line.startswith('name\t'): continue
        name, real, calls, pf, threads = line.rstrip('\n').split('\t')
        rows.append((name, int(calls), float(pf), threads))
    used = [r for r in rows if r[1]]
    where = lambda n: 'ours' if n in ours else 'pokopia' if n in poko else 'nxnative' if n in nxn else 'NONE'
    missing = [r for r in used if where(r[0]) == 'NONE']
    out = [head, '%d names asked for, %d called; of those: %d in pokopia-native, %d only in nxnative, %d in neither'
           % (len(rows), len(used), sum(r[0] in poko for r in used), sum(r[0] in nxn and r[0] not in poko for r in used), len(missing)),
           '', 'used, most called first:', 'calls\tper frame\timpl\tname\tthreads']
    for r in sorted(used, key=lambda r: -r[1]):
        out.append('%d\t%.1f\t%s\t%s\t%s' % (r[1], r[2], where(r[0]), r[0], r[3][:90]))
    out += ['', 'used and implemented nowhere:'] + [r[0] for r in missing]
    open(os.path.join(run, 'nvn_census_report.txt'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
    print('\n'.join(out[:2]))


if __name__ == '__main__':
    main()
