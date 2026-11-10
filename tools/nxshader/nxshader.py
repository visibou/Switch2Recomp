#!/usr/bin/env python3
"""nxshader: Vulkan SPIR-V and reflection for every shader a game uses (M5 track A; docs/m5-record-shaders.md).

  nxshader.py build <build folder> [--keys K ...] [--jobs N] [--cache DIR] [--report FILE] [--variants gl,dual] [--no-validate]
      the shader package of an installer build folder (shader/pkg, profile/<title>/shader_*.tsv) → the front half of nxnative's chain (nxs_front.py) →
      the Vulkan edits (nxs_spv.py) → reflection (nxs_refl.py) → spirv-val → the cache (nxs_cache.py). Single-threaded by default; at most 4 workers, at
      below-normal priority (this PC's rule for long all-core loads)
  nxshader.py ondemand --dump DIR --key K [--build FOLDER] [--installer DIR] [--cache DIR]
      a shader the game made at run time that no build has: its DM_NVN_SHDUMP dump (<key>.ctl, <key>.code, as nxnative's replay and nxrt write them)
      → the installer's own steps (s2_dumps.py, lower_corpus.py with ampere-lower, s2_pkg.py) in a scratch folder → the same chain → the cache
  nxshader.py variant K gl|dual|pad=L,L... [--cache DIR]      a variant of a cached shader (the renderer asks for these by pipeline state)
  nxshader.py validate [--cache DIR]                          spirv-val on every module the cache publishes
  nxshader.py lookup K [--cache DIR]                          what the runtime's lookup gives
  nxshader.py gc [--cache DIR]                                drop inner cache entries no published shader refers to
Defaults: --cache <repo>/_local/nxshader/cache, the translator in <installer>/host/windows/dmsh_xlate_bin, the Vulkan SDK's tools under %VULKAN_SDK%."""
import argparse, collections, concurrent.futures as cf, ctypes, hashlib, json, os, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nxs_front as F, nxs_spv as S, nxs_refl as R, nxs_cache as C

REPO = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tools'))
import s2r_paths as P
WORK = P.ext('pokopia-native')
DEFAULT_CACHE = os.path.join(REPO, 'games', 'MKW', 'MKW_nvn', 'cache')
# the installer: this tree's parent inside pokopia-native (windows/ beside nxs2.py), else the sibling folder of the private layout
DEFAULT_INSTALLER = WORK if os.path.exists(os.path.join(WORK, 'nxs2.py')) else os.path.join(WORK, 'installer')
FRONT_VERSION = 2   # nxs_front.py's job building and the integer-input rule (2: uses inside functions only)
BELOW_NORMAL = 0x00004000


def below_normal():
    if sys.platform == 'win32':
        try: ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)
        except Exception: pass


def vk_tool(name):
    """SPIRV-Tools: NXSHADER_TOOLS (the folder the installer's Windows download ships them in), then the Vulkan SDK, then the PATH"""
    return P.spirv_tool(name)


def chain_id(xl):
    return C.sha(C.LAYOUT, xl.id, 'front', FRONT_VERSION, *[p.ident() for p in S.DEFAULT_PASSES], 'reflect', C.REFLECT_VERSION)[:16]


def spirv_val(path):
    r = subprocess.run([vk_tool('spirv-val'), '--target-env', 'vulkan1.3', path], capture_output=True, text=True)
    return 'ok' if r.returncode == 0 else (r.stdout + r.stderr).strip().splitlines()[0][:300] if (r.stdout + r.stderr).strip() else 'exit %d' % r.returncode


def ssa_copy(spv, w, key):
    """the module after spirv-opt's SSA rewrite, for the integer-input analysis only (the published module is the translator's, with nxshader's edits)"""
    a, b = os.path.join(w, key + '.an.spv'), os.path.join(w, key + '.an.opt.spv')
    open(a, 'wb').write(spv)
    r = subprocess.run([vk_tool('spirv-opt'), '--ssa-rewrite', '--eliminate-dead-code-aggressive', '--target-env=vulkan1.3', a, '-o', b], capture_output=True, text=True)
    if r.returncode or not os.path.exists(b): return None
    return S.Module(open(b, 'rb').read())


class Builder:
    def __init__(self, corpus, cache, xl, w, validate=True, variants=()):
        self.corpus, self.cache, self.xl, self.w, self.validate, self.variants = corpus, cache, xl, w, validate, variants
        self.chain = chain_id(xl)
        self.notes = collections.Counter()
        self.first = {}   # key → the d1 of its first translation, when the integer-input answers translated it again

    def make(self, keys):
        """keys → {key: result}; translates the cache misses in one translator run (and the integer-input retranslations in a second)"""
        jobs, todo, out = {}, [], {}
        for k in keys:
            try:
                name, rec = self.corpus.record(k)
                # NXSHADER_XLATE_OPT=all or stages (vs,ps,cs,...; test, the GPU freeze): those shaders in the translator's optimizing mode, as Ryujinx runs it, instead
                # of nxnative's debug mode (every register a variable, a load and a store around each operation). It enters the job, so the cache key
                opt = [s for s in os.environ.get('NXSHADER_XLATE_OPT', '').split(',') if s]
                j, c1 = F.job_for(k, name, rec, self.w, self.corpus.facts, debug=not ('all' in opt or name[:2] in opt))
            except Exception as e:
                out[k] = {'key': k, 'status': 'job failed', 'error': str(e)[:300]}; continue
            d1 = C.sha('xlate', self.xl.id, F.job_digest(j))
            jobs[k] = (j, d1, name, rec)
            if not self.cache.xlate_get(d1): todo.append(j)
        err = self.xl.run(todo, self.w) if todo else {}
        for j in todo:
            k = j['key']
            if k in err: continue
            self.cache.xlate_put(jobs[k][1], open(os.path.join(self.w, k + '.spv'), 'rb').read(), json.load(open(os.path.join(self.w, k + '.res.json'), encoding='utf-8')))
        # the integer-input retranslation of vertex shaders (nxnative's second pass, int_inputs)
        retry = []
        for k, (j, d1, name, rec) in jobs.items():
            if k in err or j['stage'] != 'vs': continue
            spv, res = self.cache.xlate_get(d1, count=False)
            m = ssa_copy(spv, self.w, k)
            extra = {l: t for l, t in (S.int_only_inputs(m).items() if m else ()) if l not in j['attr']}
            if not m: self.notes['integer-input analysis failed (spirv-opt)'] += 1
            if extra:
                j2 = dict(j); j2['attr'] = dict(j['attr'], **extra)
                d1b = C.sha('xlate', self.xl.id, F.job_digest(j2))
                jobs[k] = (j2, d1b, name, rec); self.first[k] = d1
                if not self.cache.xlate_get(d1b, count=False): retry.append(j2)
        err2 = self.xl.run(retry, self.w) if retry else {}
        for j in retry:
            k = j['key']
            if k in err2: err[k] = 'integer-input retranslation: ' + err2[k]; continue
            self.cache.xlate_put(jobs[k][1], open(os.path.join(self.w, k + '.spv'), 'rb').read(), json.load(open(os.path.join(self.w, k + '.res.json'), encoding='utf-8')))
        for k, (j, d1, name, rec) in jobs.items():
            if k in err: out[k] = {'key': k, 'stage': j['stage'], 'status': 'translation failed', 'error': err[k]}; continue
            out[k] = self.finish(k, j, d1, name, rec)
        return out

    def finish(self, k, j, d1, name, rec):
        spv, res = self.cache.xlate_get(d1, count=False)
        m = S.Module(spv)
        m.fetch1d_vars = R.fetch1d_vars(m, j['stage'], res)   # for the fetch1d_tbuf edit
        passes = [p for p in S.DEFAULT_PASSES if p.applies(m, j['stage'])] + ([S.MeshStage(j['mesh_stage'])] if j.get('mesh_stage') else [])
        d2 = C.sha('final', d1, *[p.ident() for p in passes], 'reflect', C.REFLECT_VERSION, 'validate', self.validate)
        hit = self.cache.final_get(d2)
        if hit: fspv, meta = hit
        else:
            applied = [p.ident() for p in passes if p.run(m, j['stage'])]
            fspv = m.bytes()
            meta = R.reflect(k, j['mesh_stage']['kind'] if j.get('mesh_stage') else j['stage'], S.Module(fspv), res)
            meta['edits'] = applied
            if getattr(m, 'first_stores', None):   # the replay makes these before the dispatch (nxs_spv.py FirstStore)
                by = {b['binding']: b for b in meta['bindings']}
                bindof = lambda v: m.deco(v, S.DecBinding)[0]
                meta['first_store'] = [{'reg': by[bindof(f['var'])]['nvn']['reg'], 'word': f['word'], 'value': f['value']} for f in m.first_stores]
            if self.validate:
                vp = os.path.join(self.w, k + '.final.spv'); open(vp, 'wb').write(fspv)
                meta['spirv_val'] = spirv_val(vp)
            self.cache.final_put(d2, fspv, meta)
        meta = dict(meta)
        meta['cache'] = {'chain': self.chain, 'd1': d1, **({'d1_first': self.first[k]} if k in self.first else {}), 'd2': d2, 'translator': self.xl.id}
        if j.get('mesh_stage'): meta['stage'] = j['mesh_stage']['kind']; meta['mesh'] = j['mesh_stage']   # 'ts' / 'ms': the replay's stages 3 and 4, as the host keeps them (nvn_mtl.m gfx_stage_bits)
        meta['c1'] = j.get('c1', '')   # the shader's own constant segment c[1] (part of the job, so of d1): the replay binds it (vk_shader.c)
        meta['made_from'] = {'package': name, 'record_sha256': hashlib.sha256(rec).hexdigest(), 'facts': self.corpus.facts_raw.get(k), 'source': self.corpus.source.get(k),
                          'job': {x: j[x] for x in ('attr', 'tex', 'img', 'fout', 'local', 'shared', 'tf', 'early_fragment_tests', 'no_facts', 'debug', 'reduced') if x in j}}
        self.cache.publish(k, fspv, meta)
        for v in self.variants: self.variant(k, v, fspv, meta)
        return {'key': k, 'stage': j['stage'], 'status': 'ok', 'spirv_val': meta.get('spirv_val'), 'edits': meta.get('edits', []), 'caps': meta['capabilities'],
                'kinds': [b['nvn'].get('kind') for b in meta['bindings']], 'int_inputs': sorted(l for l, t in j['attr'].items() if t == 'uint'), 'tf': bool(j.get('tf')),
                'subgroups': meta['uses_subgroups'], 'size': len(fspv)}

    def variant(self, k, v, fspv, meta):
        name, _, arg = v.partition('=')
        opts = [int(x) for x in arg.split(',') if x] if arg else []
        m = S.Module(fspv)
        ps = S.VARIANTS[name](opts)
        if not any(p.applies(m, meta['stage']) for p in ps): return None
        applied = [p.ident() for p in ps if p.applies(m, meta['stage']) and p.run(m, meta['stage'])]
        if not applied: return None
        vspv = m.bytes(); vmeta = R.reflect(k, meta['stage'], S.Module(vspv), meta.get('translator', {}) | {'sbufs': [], 'textures': [], 'images': []})
        vmeta['bindings'] = meta['bindings']   # the edits never touch descriptors
        vmeta['edits'] = meta.get('edits', []) + applied; vmeta['cache'] = dict(meta['cache'], variant=v)
        if self.validate:
            vp = os.path.join(self.w, k + '.' + name + '.spv'); open(vp, 'wb').write(vspv); vmeta['spirv_val'] = spirv_val(vp)
        tag = name + ('-' + '-'.join(map(str, opts)) if opts else '')
        self.cache.publish(k, vspv, vmeta, tag)
        return vmeta


def worker(args):
    build, profile, cache_root, xldir, keys, validate, variants = args
    below_normal()
    w = tempfile.mkdtemp(prefix='nxshader_')
    try:
        b = Builder(F.Corpus(build, profile), C.Cache(cache_root), F.Translator(xldir), w, validate, variants)
        out = {}
        for i in range(0, len(keys), 64): out.update(b.make(keys[i:i + 64]))
        return out, dict(b.cache.stats), dict(b.notes)
    finally: shutil.rmtree(w, ignore_errors=True)


def summarize(res, cache_stats, notes, t, corpus, runtime_keys):
    st = collections.Counter(r['status'] for r in res.values())
    by_stage = collections.Counter(r.get('stage', '?') for r in res.values())
    ok = [r for r in res.values() if r['status'] == 'ok']
    val = collections.Counter('ok' if r['spirv_val'] == 'ok' else 'fail' for r in ok if r.get('spirv_val'))
    verr = collections.Counter(r['spirv_val'].split(':', 1)[-1].strip()[:110] for r in ok if r.get('spirv_val') not in (None, 'ok'))
    edits = collections.Counter(e.split(' v')[0] for r in ok for e in r['edits'])
    caps = collections.Counter(c for r in ok for c in r['caps'])
    kinds = collections.Counter(k for r in ok for k in r['kinds'])
    rk = [k for k in runtime_keys if k in res]
    rk_ok = [k for k in rk if res[k]['status'] == 'ok' and res[k].get('spirv_val') in ('ok', None)]
    lines = [f'shaders {len(res)} ({", ".join(f"{s} {n}" for s, n in sorted(by_stage.items()))}) in {t:.0f} s: ' + ', '.join(f'{s} {n}' for s, n in st.most_common()),
             f'spirv-val (vulkan1.3): {val["ok"]} pass, {val["fail"]} fail',
             *[f'  [{n:4d}] {e}' for e, n in verr.most_common(12)],
             f'cache: ' + ', '.join(f'{k} {v}' for k, v in cache_stats.items()),
             'edits applied: ' + (', '.join(f'{k} {v}' for k, v in edits.most_common()) or 'none'),
             f'integer-input retranslations: {sum(1 for r in ok if r["int_inputs"])} vertex shaders ({sum(len(r["int_inputs"]) for r in ok)} inputs)',
             f'transform feedback layouts: {sum(1 for r in ok if r["tf"])}; modules using subgroup operations: {sum(1 for r in ok if r["subgroups"])} '
             f'(vertex {sum(1 for r in ok if r["subgroups"] and r["stage"] == "vs")})',
             'capabilities: ' + ', '.join(f'{k} {v}' for k, v in caps.most_common()),
             'binding kinds: ' + ', '.join(f'{k} {v}' for k, v in kinds.most_common()),
             f'run-time keys ({len(runtime_keys)} in the profile): {len(rk)} in this build, {len(rk_ok)} translated and valid',
             *[f'  {k}: {n}' for k, n in notes.items()],
             *[f'  [{r["status"]}] {r["key"]}: {r.get("error", "")[:160]}' for r in list(res.values()) if r['status'] != 'ok'][:20]]
    return '\n'.join(lines)


def runtime_keys(installer):
    p = os.path.join(installer, 'profile', '0400A3A01EE4C000', 'runtime_keys.tsv')
    return [l.split('\t')[0] for l in open(p, encoding='utf-8') if l.strip() and not l.startswith('#')] if os.path.exists(p) else []


def cmd_build(a):
    below_normal()
    corpus = F.Corpus(a.build, a.profile)
    keys = a.keys or ([l.strip() for l in open(a.keys_file, encoding='utf-8') if l.strip()] if a.keys_file else corpus.keys())
    n = max(1, min(4, a.jobs))
    cache = C.Cache(a.cache)
    xl = F.Translator(a.xlate)
    cache.set_chain(chain_id(xl))
    t0 = time.time()
    chunks = [keys[i::n] for i in range(n)]
    res, stats, notes = {}, collections.Counter(), collections.Counter()
    variants = [v for v in (a.variants or '').split(',') if v]
    wargs = [(a.build, a.profile, a.cache, a.xlate, c, not a.no_validate, variants) for c in chunks if c]
    if n == 1: outs = [worker(wargs[0])]
    else:
        with cf.ProcessPoolExecutor(n) as ex: outs = list(ex.map(worker, wargs))
    for o, s, nt in outs: res.update(o); stats.update(s); notes.update(nt)
    text = summarize(res, dict(stats), notes, time.time() - t0, corpus, runtime_keys(a.installer))
    print(text)
    if a.report:
        os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
        open(a.report, 'w', encoding='utf-8').write(text + '\n')
        json.dump(res, open(os.path.splitext(a.report)[0] + '.json', 'w', encoding='utf-8'), indent=1)
    bad = sum(1 for r in res.values() if r['status'] != 'ok' or r.get('spirv_val') not in ('ok', None))
    return 1 if bad else 0


def cmd_ondemand(a):
    """the run-time path: a dump the game made → the installer's lowering steps in a scratch build folder → the chain → the cache"""
    below_normal()
    key = a.key.lower()
    ctl_p, code_p = os.path.join(a.dump, key + '.ctl'), os.path.join(a.dump, key + '.code')
    if not (os.path.exists(ctl_p) and os.path.exists(code_p)): print(f'[RED] ondemand {key}: no dump {ctl_p} / .code'); return 2
    tools, py = os.path.join(a.installer, 'tools'), sys.executable
    ampere = os.environ.get('AMPERE_LOWER') or next((p for p in (os.path.join(a.installer, 'ampere-lower', 'windows', 'ampere-lower.exe'),   # as the installer ships it
                                                             os.path.join(a.installer, 'ampere-lower', 'ampere-lower.exe')) if os.path.exists(p)), '')
    t0 = time.time()
    s = tempfile.mkdtemp(prefix='nxshader_od_')
    try:
        os.makedirs(os.path.join(s, 'dumps'))
        for p in (ctl_p, code_p): shutil.copy(p, os.path.join(s, 'dumps'))
        env = dict(os.environ, AMPERE_LOWER=ampere, J='1')
        steps = [([py, '-I', os.path.join(tools, 's2_dumps.py'), 'dumps', '--corpus', 'corpus', '--ctl', 'ctl'], 'record'),
                 ([py, '-I', os.path.join(tools, 'lower_corpus.py'), 'lowered', 'corpus'], 'lower (ampere-lower)'),
                 ([py, '-I', os.path.join(tools, 's2_pkg.py'), 'ctl', 'lowered', os.path.join('b', 'shader', 'pkg'), os.path.join('b', 'shader', 'tables'), 'corpus'], 'package')]
        for cmd, what in steps:
            r = subprocess.run(cmd, cwd=s, env=env, capture_output=True, text=True, creationflags=BELOW_NORMAL if sys.platform == 'win32' else 0)
            if r.returncode: print(f'[RED] ondemand {key}: {what} failed: {(r.stderr or r.stdout).strip()[-300:]}'); return 3
        corpus = F.Corpus(os.path.join(s, 'b'), os.path.join(s, 'b', 'shader', 'tables'))
        if key not in corpus.loc:
            got = corpus.keys()
            if not (a.rekey and len(got) == 1):
                print(f'[RED] ondemand {key}: the dump keys as {got[0] if got else "nothing"} (its control block or code differs from what the runtime hashed)'); return 4
            print(f'ondemand {key}: the dump keys as {got[0]} under the current rule (--rekey: translated as {got[0]})'); key = got[0]
        if a.build:   # the build's hand-made answers (shader_facts.seed.tsv merged into its profile) win, as nxs2.py merges them
            try:
                fb = F.Corpus(a.build).facts_raw
                if key in fb: corpus.facts_raw[key] = fb[key]; corpus.facts[key] = json.loads(fb[key])
            except FileNotFoundError: pass
        w = os.path.join(s, 'xl'); os.makedirs(w)
        xl = F.Translator(a.xlate); cache = C.Cache(a.cache); cache.set_chain(chain_id(xl))
        r = Builder(corpus, cache, xl, w, validate=True).make([key])[key]
        print(f'ondemand {key}: {r["status"]}' + (f', spirv-val {r.get("spirv_val")}' if r['status'] == 'ok' else f': {r.get("error")}') + f' ({time.time() - t0:.1f} s)')
        return 0 if r['status'] == 'ok' and r.get('spirv_val') == 'ok' else 5
    finally:
        if not a.keep: shutil.rmtree(s, ignore_errors=True)
        else: print('scratch folder kept:', s)


def cmd_variant(a):
    cache = C.Cache(a.cache)
    hit = cache.lookup(a.key)
    if not hit: print(f'{a.key}: not in the cache (or stale)'); return 2
    spv, meta = hit
    w = tempfile.mkdtemp(prefix='nxshader_v_')
    try:
        b = Builder.__new__(Builder); b.cache, b.w, b.validate = cache, w, True
        v = b.variant(a.key, a.variant, spv, meta)
        print(f'{a.key} {a.variant}: ' + (f'edits {v["edits"]}, spirv-val {v.get("spirv_val")}' if v else 'nothing to do for this shader'))
        return 0 if v is None or v.get('spirv_val') == 'ok' else 1
    finally: shutil.rmtree(w, ignore_errors=True)


def cmd_validate(a):
    below_normal()
    vk = os.path.join(a.cache, 'vk'); files = sorted(f for f in os.listdir(vk) if f.endswith('.spv'))
    n = max(1, min(4, a.jobs))
    with cf.ThreadPoolExecutor(n) as ex: rs = list(ex.map(lambda f: (f, spirv_val(os.path.join(vk, f))), files))
    bad = [(f, r) for f, r in rs if r != 'ok']
    print(f'spirv-val (vulkan1.3) on {len(rs)} modules in {vk}: {len(rs) - len(bad)} pass, {len(bad)} fail')
    for f, r in bad[:30]: print(f'  {f}: {r}')
    return 1 if bad else 0


def cmd_gc(a):
    n, freed = C.Cache(a.cache).gc()
    print(f'gc: {n} files no published shader refers to, {freed / 1048576:.0f} MB freed'); return 0


def cmd_lookup(a):
    cache = C.Cache(a.cache); hit = cache.lookup(a.key)
    if not hit: print(f'{a.key}: missing or stale'); return 1
    spv, meta = hit
    print(json.dumps({k: meta[k] for k in ('key', 'stage', 'cache', 'edits', 'spirv_val', 'capabilities') if k in meta}, indent=1)); return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--cache', default=os.environ.get('NXSHADER_CACHE', DEFAULT_CACHE))
    ap.add_argument('--installer', default=DEFAULT_INSTALLER)
    ap.add_argument('--xlate', default=None, help='the translator folder (default <installer>/host/windows/dmsh_xlate_bin)')
    sp = ap.add_subparsers(dest='cmd', required=True)
    b = sp.add_parser('build'); b.add_argument('build'); b.add_argument('--profile'); b.add_argument('--keys', nargs='*'); b.add_argument('--keys-file', help='a file of keys, one per line (for lists too long for a command line)'); b.add_argument('--jobs', type=int, default=1)
    b.add_argument('--report'); b.add_argument('--variants'); b.add_argument('--no-validate', action='store_true')
    o = sp.add_parser('ondemand'); o.add_argument('--dump', required=True); o.add_argument('--key', required=True); o.add_argument('--build'); o.add_argument('--keep', action='store_true')
    o.add_argument('--rekey', action='store_true', help='a dump an older host named by an older key rule: translate it under the key the current rule gives')
    v = sp.add_parser('variant'); v.add_argument('key'); v.add_argument('variant')
    va = sp.add_parser('validate'); va.add_argument('--jobs', type=int, default=2)
    lk = sp.add_parser('lookup'); lk.add_argument('key')
    sp.add_parser('gc')
    a = ap.parse_args()
    if not a.xlate: a.xlate = P.ext('nxnative', 'dm', 'gfx', 'tools', 'dmsh_xlate_bin')
    return {'build': cmd_build, 'ondemand': cmd_ondemand, 'variant': cmd_variant, 'validate': cmd_validate, 'lookup': cmd_lookup, 'gc': cmd_gc}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
