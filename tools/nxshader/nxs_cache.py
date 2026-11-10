"""The versioned shader cache (research/01 lesson 2). Content-addressed, in two stages, so a change invalidates exactly what it affects:
  xlate/<d1>.spv.z + .res.json   d1 = sha256(translator identity, the job: code bytes, c1, answer-table facts, translation options)
  final/<d2>.spv.z + .json       d2 = sha256(d1, the name, version and options of every edit that applied to this shader, the reflection's version)
  vk/<key>.spv + <key>.json      what the runtime loads, by shader key; the json carries the chain id, d1, d2 and the inputs it was made from
  vk/CHAIN                       the chain id of the current tools (translator, edits, reflection): an entry whose chain differs is stale
A translator rebuild changes d1 for every shader; a new version of one edit changes d2 only for the shaders it applies to (vertex outputs: vertex and geometry
shaders); new facts for one shader change that shader's d1. The two inner stages are zlib-compressed (the translator's debug-mode SPIR-V shrinks about
fivefold), the runtime's view is plain. Writes go through a temporary file and a rename, so a killed run leaves no half entry; `gc` drops the inner entries
no published shader refers to (older generations)."""
import hashlib, json, os, zlib

LAYOUT = 'nxshader-cache 2'   # 2: compressed inner stages
REFLECT_VERSION = 2   # 2: uses_subgroups from the operations a module runs, not the capabilities it declares


def sha(*parts):
    h = hashlib.sha256()
    for p in parts: h.update(p if isinstance(p, (bytes, bytearray)) else str(p).encode()); h.update(b'\0')
    return h.hexdigest()


def atomic_write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp-%d' % os.getpid()
    with open(tmp, 'wb') as f: f.write(data if isinstance(data, (bytes, bytearray)) else data.encode('utf-8'))
    os.replace(tmp, path)


class Cache:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        lp = os.path.join(root, 'LAYOUT')
        if os.path.exists(lp) and open(lp).read().strip() != LAYOUT: raise RuntimeError(f'{root}: cache layout {open(lp).read().strip()!r}, this nxshader writes {LAYOUT!r} (delete the folder)')
        if not os.path.exists(lp): atomic_write(lp, LAYOUT + '\n')
        self.stats = {'xlate hit': 0, 'xlate miss': 0, 'final hit': 0, 'final miss': 0}

    def _p(self, kind, d, ext): return os.path.join(self.root, kind, d[:2], d + ext)

    def _get(self, kind, d, side):
        s, j = self._p(kind, d, '.spv.z'), self._p(kind, d, side)
        if not (os.path.exists(s) and os.path.exists(j)): return None
        return zlib.decompress(open(s, 'rb').read()), json.load(open(j, encoding='utf-8'))

    def _put(self, kind, d, side, spv, meta):
        atomic_write(self._p(kind, d, side), json.dumps(meta, indent=1))
        atomic_write(self._p(kind, d, '.spv.z'), zlib.compress(spv, 6))

    # stage 1: the translator's output
    def xlate_get(self, d1, count=True):
        r = self._get('xlate', d1, '.res.json')
        if count: self.stats['xlate hit' if r else 'xlate miss'] += 1
        return r

    def xlate_put(self, d1, spv, res): self._put('xlate', d1, '.res.json', spv, res)

    # stage 2: after the edits, with the reflection
    def final_get(self, d2):
        r = self._get('final', d2, '.json')
        self.stats['final hit' if r else 'final miss'] += 1
        return r

    def final_put(self, d2, spv, meta): self._put('final', d2, '.json', spv, meta)

    # the runtime's view
    def publish(self, key, spv, meta, variant=''):
        base = os.path.join(self.root, 'vk', key + (('.' + variant) if variant else ''))
        atomic_write(base + '.json', json.dumps(meta, indent=1))
        atomic_write(base + '.spv', spv)

    def set_chain(self, chain): atomic_write(os.path.join(self.root, 'vk', 'CHAIN'), chain + '\n')

    def lookup(self, key, chain=None):
        """the runtime's lookup: (spv, meta) when vk/<key> exists and was made by the current chain, else None"""
        base = os.path.join(self.root, 'vk', key)
        if not (os.path.exists(base + '.spv') and os.path.exists(base + '.json')): return None
        meta = json.load(open(base + '.json', encoding='utf-8'))
        if chain is None:
            cp = os.path.join(self.root, 'vk', 'CHAIN')
            chain = open(cp).read().strip() if os.path.exists(cp) else None
        if meta.get('cache', {}).get('chain') != chain: return None
        return open(base + '.spv', 'rb').read(), meta

    def gc(self):
        """drop the inner entries that no published shader refers to: → (files removed, bytes freed)"""
        keep = set()
        vk = os.path.join(self.root, 'vk')
        for f in os.listdir(vk) if os.path.isdir(vk) else []:
            if f.endswith('.json'):
                c = json.load(open(os.path.join(vk, f), encoding='utf-8')).get('cache', {})
                keep |= {c.get(k) for k in ('d1', 'd1_first', 'd2')}
        n = freed = 0
        for kind in ('xlate', 'final'):
            top = os.path.join(self.root, kind)
            for sub in os.listdir(top) if os.path.isdir(top) else []:
                for f in os.listdir(os.path.join(top, sub)):
                    if f.split('.')[0] not in keep:
                        p = os.path.join(top, sub, f); freed += os.path.getsize(p); os.remove(p); n += 1
        return n, freed
