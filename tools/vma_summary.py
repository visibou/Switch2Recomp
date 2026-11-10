"""Summarize a VMA memory map (SWR_VMA_DUMP -> vma_<s>.json): per memory type, bytes by kind (images by format and
size class, buffers by memory usage, raw commits), block space left unused, and the largest allocations.

  python tools/vma_summary.py <vma_N.json> [--top 15]
"""
import argparse, collections, json, re


def kind(name, typ):
    if not name: return typ or '?'
    m = re.match(r'img (\d+)x(\d+)x(\d+) levels (\d+) layers (\d+) format (\d+) usage (0x[0-9a-f]+)', name)
    if m:
        w, h, d, lv, ly, fmt, usage = m.groups()
        rt = int(usage, 16) & 0x30   # color / depth attachment
        return 'image%s fmt %s' % (' (render target)' if rt else '', fmt)
    m = re.match(r'buf memusage (\d+)', name)
    if m: return 'buffer memusage %s' % m.group(1)
    return re.sub(r'\d+x\d+.*', '', name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dump')
    ap.add_argument('--top', type=int, default=15)
    a = ap.parse_args()
    d = json.load(open(a.dump))
    flags = {}
    for heap, hv in d.get('MemoryInfo', {}).items():
        for t, tv in hv.get('MemoryPools', {}).items():
            flags[t] = '%s %s' % (heap, ','.join(f.replace('DEVICE_LOCAL', 'DL').replace('HOST_', 'H_') for f in tv.get('Flags', [])))
    allocs = []   # (type, size, name, typ, dedicated)
    unused = collections.Counter(); blocks = collections.Counter()
    for t, pool in d.get('DefaultPools', {}).items():
        for b in pool.get('Blocks', {}).values():
            blocks[t] += b.get('TotalBytes', 0); unused[t] += b.get('UnusedBytes', 0)
            for s in b.get('Suballocations', []):
                if s.get('Type') == 'FREE': continue
                allocs.append((t, s['Size'], s.get('Name', ''), s.get('Type', ''), False))
        for s in pool.get('DedicatedAllocations', []):
            blocks[t] += s['Size']
            allocs.append((t, s['Size'], s.get('Name', ''), s.get('Type', ''), True))
    by_type = collections.defaultdict(collections.Counter); cnt = collections.defaultdict(collections.Counter)
    for t, size, name, typ, ded in allocs:
        k = kind(name, typ) + (' [dedicated]' if ded else '')
        by_type[t][k] += size; cnt[t][k] += 1
    for t in sorted(by_type, key=lambda t: -sum(by_type[t].values())):
        tot = sum(by_type[t].values())
        print('== %s (%s): %d MiB in allocations, %d MiB of blocks, %d MiB unused in blocks' % (
            t, flags.get(t, ''), tot >> 20, blocks[t] >> 20, unused[t] >> 20))
        for k, v in by_type[t].most_common(12):
            print('   %-45s %6d MiB  %5d' % (k[:45], v >> 20, cnt[t][k]))
    print('\n== largest allocations')
    for t, size, name, typ, ded in sorted(allocs, key=lambda x: -x[1])[:a.top]:
        print('   %-7s %6d MiB  %s%s' % (t, size >> 20, name or typ, ' [dedicated]' if ded else ''))


if __name__ == '__main__':
    main()
