"""The front half of nxnative's shader chain (dm/gfx/tools/dmsh_build.py, MIT, (c) seedds): a build folder's shader package and tables → translation jobs → the
translator (dmsh_xlate: Ryujinx.Graphics.Shader, MIT, with TargetLanguage.Spirv / TargetApi.Vulkan) → <key>.spv and <key>.res.json. Ported: records,
job_for (with vertex_ab, early_fragment_tests, tf_layout) and the integer-input retranslation, the last read from the SPIR-V instead of the MSL. Not ported,
being Metal's: spv_single_lane, the geometry-to-compute lowering, spirv-cross and everything after it (docs/m5-record-shaders.md lists each edit)."""
import hashlib, json, os, re, struct, subprocess, sys, tempfile
import dmsh_ampere, dmsh_meshstage

CTL_MAGIC = bytes.fromhex('34127698')


def records(d):
    """A package file's records, each from its 'BTCDBTCD' to the next (dmsh_build.py records)"""
    r = [m.start() for m in re.finditer(b'BTCDBTCD', d)]
    return [d[x:(r[k + 1] if k + 1 < len(r) else len(d))] for k, x in enumerate(r)]


class Corpus:
    """A build folder's shader package (shader/pkg/<stage>_<game>.bin) and the profile's tables (shader_map.tsv, shader_facts.tsv with the hand-made answers
    merged, shader_pairs.tsv), as nxs2.py lays them out"""
    def __init__(self, build, profile=None):
        self.build = build
        self.pkg = os.path.join(build, 'pkg') if os.path.isdir(os.path.join(build, 'pkg')) else os.path.join(build, 'shader', 'pkg')   # shader_split.py's output folder, or nxs2.py's
        prof = profile or (build if os.path.isfile(os.path.join(build, 'shader_map.tsv')) else self._profile_dir(build))
        self.tables = prof
        self.map_path = os.path.join(prof, 'shader_map.tsv')
        self.loc, self.map_rows, self.source = {}, {}, {}
        for raw in open(self.map_path, 'rb'):
            c = raw.rstrip(b'\r\n').split(b'\t')
            if len(c) > 4 and c[0].endswith(b'.bin'):
                k = c[3].decode()
                self.loc.setdefault(k, (c[0].decode(), int(c[1])))
                self.map_rows.setdefault(k, raw.rstrip(b'\r\n'))
                self.source.setdefault(k, c[4].decode(errors='replace'))
        self.facts_raw, self.facts = {}, {}
        fp = os.path.join(prof, 'shader_facts.tsv')
        if os.path.exists(fp):
            for l in open(fp, encoding='utf-8'):
                if l.strip() and not l.startswith('#'):
                    k, v = l.rstrip('\r\n').split('\t', 1)
                    self.facts_raw[k] = v; self.facts[k] = json.loads(v)
        self.pairs = {}
        pp = os.path.join(prof, 'shader_pairs.tsv')
        if os.path.exists(pp):
            for l in open(pp, encoding='utf-8'):
                if l.strip() and not l.startswith('#'):
                    v, p = l.split()[:2]; self.pairs.setdefault(v, set()).add(p)
        self._packs = {}

    @staticmethod
    def _profile_dir(build):
        prof = os.path.join(build, 'profile')
        subs = [os.path.join(prof, d) for d in sorted(os.listdir(prof))] if os.path.isdir(prof) else []
        for s in subs:
            if os.path.exists(os.path.join(s, 'shader_map.tsv')): return s
        t = os.path.join(build, 'shader', 'tables')
        if os.path.exists(os.path.join(t, 'shader_map.tsv')): return t
        raise FileNotFoundError(f'{build}: no shader_map.tsv under profile/ or shader/tables/')

    def keys(self):
        return sorted(self.loc)

    def record(self, key):
        """(package name, the shader's record bytes)"""
        name, k = self.loc[key]
        if name not in self._packs:
            self._packs[name] = records(open(os.path.join(self.pkg, name), 'rb').read())
        return name, self._packs[name][k]


def vertex_ab(key, code, ctl):
    """dmsh_build.py vertex_ab, unchanged: a vertex shader in two parts (VertexA + VertexB) merged into one program, as Ryujinx's Translate(VertexA) builds it.
    → (code area with its 0x30 header, c1), or None for any other vertex shader"""
    if len(code) < 0x80 or len(ctl) < 0x760 or code[:4] != bytes.fromhex('78563412'): return None
    w0 = struct.unpack_from('<I', code, 0x30)[0]
    if w0 & 0x1f != 1 or (w0 >> 10) & 0xf != 0: return None
    a_sph, a_len, _, b_sph, b_len, _ = struct.unpack_from('<6I', ctl, 0x748)
    if not any((a_sph, a_len, b_sph, b_len)): return None
    bw0 = struct.unpack_from('<I', code, b_sph)[0] if 0x80 <= b_sph <= len(code) - 0x50 else 0
    if a_sph != 0x30 or b_sph != struct.unpack_from('<I', ctl, 0x700)[0] + 0x30 or bw0 & 0x1f != 1 or (bw0 >> 10) & 0xf != 1 or (a_len - 0x50) % 32 or (b_len - 0x50) % 32 or b_sph + b_len > len(code):
        raise ValueError(f'{key}: vertex shader of shader type 0 without a VertexB part in the expected place (control +0x748 {a_sph:#x} {a_len:#x} {b_sph:#x} {b_len:#x})')
    sa, sb = list(struct.unpack_from('<20I', code, a_sph)), list(struct.unpack_from('<20I', code, b_sph))
    sph = [sb[0] | sa[0] & ~(0xf << 10)] + [max(sa[i] & 0xffffff, sb[i] & 0xffffff) | sb[i] & ~0xffffff for i in (1, 2, 3)] + [sb[4]] + [sa[i] | sb[i] for i in range(5, 20)]
    a = bytearray(code[a_sph + 0x50:a_sph + a_len]); b = code[b_sph + 0x50:b_sph + b_len]
    for o in range(0, len(a), 8):
        if o % 32 == 0: continue
        ins = struct.unpack_from('<Q', a, o)[0]
        if ins >> 52 == 0xe30 and ins & 0x1f == 0xf:
            struct.pack_into('<Q', a, o, 0xe24 << 52 | ((len(a) - (o + 8)) & 0xffffff) << 20 | ins & 0xfffff)
    return code[:0x30] + struct.pack('<20I', *sph) + bytes(a) + b, code[b_sph + b_len:]


def early_fragment_tests(ctl, code):
    """dmsh_build.py early_fragment_tests, unchanged: control +0x718 set and the SPH does not write depth"""
    return len(ctl) > 0x718 and ctl[0x718] != 0 and len(code) >= 0x50 and not struct.unpack_from('<I', code, 19 * 4)[0] & 2


def tf_layout(key, ctl):
    """dmsh_build.py tf_layout, unchanged: the four transform feedback streams of the control section, or None"""
    strides = struct.unpack_from('<4H', ctl, 0x6dc)
    if not any(strides): return None
    out = []
    for b in range(4):
        if not strides[b]: out.append([]); continue
        count = ctl[0x6e8 + b]
        locs = list(ctl[0xdc + 0x80 * b:0xdc + 0x80 * b + count])
        sizes = list(ctl[0x4dc + 0x80 * b:0x4dc + 0x80 * b + count])
        if not 1 <= count <= 128 or any(not 0x20 <= x < 0x40 for x in locs) or any(not 1 <= n <= 4 for n in sizes) or strides[b] != 4 * sum(sizes):
            return None
        out.append({'stride': strides[b], 'outputs': [[o - 0x20, n] for o, n in zip(locs, sizes)]})
    return out


def job_for(key, name, rec, w, facts, debug=True, reduced=True):
    """dmsh_build.py job_for: the translator's job and c1. The code file goes to w/<key>.bin. debug / reduced: XLATE_DEBUG / XLATE_REDUCED (both on by default there)"""
    ci = rec.find(CTL_MAGIC); code = rec[0x38:ci]; stage = name[:2]
    extra = {}
    ctl = rec[ci:]
    nvn2 = dmsh_ampere.is_nvn2(ctl)
    if nvn2:
        # NVN2: Ampere code, which nxnative's translator reads itself (job arch sm86). dmsh_build.py job_for, MIT, (c) seedds
        c1 = dmsh_ampere.c1_segment(ctl, code)
        code = code[:dmsh_ampere.inst_len(ctl)]
        extra['arch'] = 'sm86'
        if stage == 'cs': extra['nvn2_cs'] = [rec[ci + 0xb1], rec[ci + 0xb2], rec[ci + 0xb3], struct.unpack_from('<H', rec, ci + 0xba)[0], rec[ci + 0xa9] * 4, rec[ci + 0xb4], rec[ci + 0xb5]]
        if stage in ('ts', 'ms'):
            mesh = dmsh_meshstage.info(ctl, rec[0x38:ci])
            lay, size = dmsh_meshstage.memory(mesh)
            extra.update(mesh_stage=mesh, stage_memory=lay, local=[mesh['threads'], 1, 1], shared=size); code = code[0x80:]; stage = 'cs'
    else:
        c1o, c1t = struct.unpack_from('<2I', rec, ci + 0x700)
        c1 = rec[0x38 + c1o:0x38 + c1t]
        ab = vertex_ab(key, code, ctl) if stage == 'vs' else None
        if ab: code, c1 = ab
        if stage != 'cs': code = code[0x30:]
        elif code[:4] == bytes.fromhex('79563412'):
            if any(code[0x30:0x100]): raise ValueError(f'{key}: compute code area +0x30..+0x100 is not all zero, so the instructions do not start at +0x100')
            code = code[0x100:]
        tf = tf_layout(key, ctl) if stage in ('vs', 'gs') else None
        if tf: extra['tf'] = tf
        if stage == 'ps' and early_fragment_tests(ctl, code): extra['early_fragment_tests'] = True
    cp = os.path.join(w, key + '.bin'); open(cp, 'wb').write(code)
    j = {'key': key, 'stage': stage, 'code': cp, 'tex': {}, 'img': {}, 'attr': {}, 'fout': {}, 'c1': c1.hex(), **extra, 'debug': debug, 'reduced': reduced}
    if extra.get('mesh_stage'): return j, c1
    if key not in facts: j['no_facts'] = True
    else: j.update(facts[key])
    return j, c1


def job_digest(job):
    """What the translator's output depends on: the job with the code's bytes in place of its path"""
    j = dict(job); code = open(j.pop('code'), 'rb').read()
    return hashlib.sha256(json.dumps(j, sort_keys=True, separators=(',', ':')).encode() + b'\0' + code).hexdigest()


class Translator:
    """dmsh_xlate (the installer's Windows build) run with dotnet; its identity is its DLLs' and its embedded sources' hashes"""
    def __init__(self, bindir, dotnet='dotnet'):
        self.bindir, self.dotnet = bindir, dotnet
        self.dll = os.path.join(bindir, 'dmsh_xlate.dll')
        if not os.path.exists(self.dll): raise FileNotFoundError(self.dll)
        h = hashlib.sha256()
        for f in sorted(os.listdir(bindir)):
            if f.endswith(('.dll', '.json')): h.update(f.encode() + b'\0' + open(os.path.join(bindir, f), 'rb').read() + b'\0')
        r = subprocess.run([dotnet, self.dll, '--source-sha'], capture_output=True, text=True, timeout=120)
        self.source_sha = r.stdout.strip()
        if r.returncode or not re.fullmatch(r'[0-9a-f]{16}', self.source_sha): raise RuntimeError(f'dmsh_xlate --source-sha failed: {r.stderr.strip()[:200]}')
        self.id = 'dmsh_xlate ' + self.source_sha + ' bin ' + h.hexdigest()[:16]

    def run(self, jobs, w, low_priority=True):
        """Translate a list of jobs into w; → {key: error} for the ones that failed"""
        jp = os.path.join(w, 'job_%d_%s.json' % (os.getpid(), jobs[0]['key']))
        json.dump(jobs, open(jp, 'w'))
        flags = 0x00004000 if low_priority and sys.platform == 'win32' else 0   # BELOW_NORMAL_PRIORITY_CLASS
        r = subprocess.run([self.dotnet, self.dll, jp, w], capture_output=True, text=True, creationflags=flags)
        err = {}
        for ln in r.stderr.splitlines():
            m = re.match(r'\[RED\] (\w+): (.*)', ln)
            if m: err[m.group(1)] = m.group(2)[:300]
        for j in jobs:
            if j['key'] not in err and not os.path.exists(os.path.join(w, j['key'] + '.spv')): err[j['key']] = 'no spv (' + (r.stderr.strip()[-200:] or 'exit %d' % r.returncode) + ')'
        return err
