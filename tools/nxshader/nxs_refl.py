"""Reflection from the SPIR-V (nxnative reflected from the MSL: dmsh_build.py reflect, decode_tex, decode_ubo_tex, c2_array, pool_arrays, drv_reads; MIT, (c) seedds).
For each descriptor the Vulkan binding (set, binding, descriptor type, count) and what it is in NVN terms (the kinds of dmsh_abi.h: uniform slot, driver segment,
storage slot, texture / image / sampler slot, bindless handle, c2 array, pool), decoded from the translator's resource names and its res.json as nxnative did;
then the stage interface (vertex attributes, varyings, colour outputs, builtins), execution modes, capabilities and the driver-segment rows the shader reads."""
import os, re
import nxs_spv as S

HERE = os.path.dirname(os.path.abspath(__file__))
ABI_H = os.path.join(HERE, '..', '..', 'runtime', 'nvn', 'gfx', 'dmsh_abi.h')
_DEFINE = re.compile(r'#define DMSH_([A-Z][A-Z0-9_]*)[ \t]+(0x[0-9A-Fa-f]+|[1-9][0-9]*|0|"[^"\\]*")(?:[ \t]+//.*)?')


def load_abi(path=ABI_H):
    """dmsh_abi.h as nxnative's dmsh_abi.py reads it (the strict form its header describes)"""
    v = {}
    for n, line in enumerate(open(path, encoding='utf-8').read().splitlines(), 1):
        s = line.strip()
        if not s or s.startswith('//') or s == '#pragma once': continue
        m = _DEFINE.fullmatch(s)
        if not m: raise ValueError(f'{path}:{n}: not a dmsh_abi.h line: {s}')
        v[m.group(1)] = m.group(2)[1:-1] if m.group(2).startswith('"') else int(m.group(2), 0)
    return v


A = load_abi()
TEX_BASE, IMG_BASE, IMG_SLOTS, SEPTEX_BASE, SEPTEX_SLOT0, SAMP_BASE, ROW = A['C2_TEX'], A['C2_IMG'], A['C2_IMG_SLOTS'], A['C2_SEPTEX'], A['SEPTEX_SLOT0'], A['C2_SAMP'], A['C2_ROW']
SSBO = {'vs': A['C0_SSBO_VS'], 'cs': A['C0_SSBO_CS'], 'gs': A['C0_SSBO_GS'], 'ps': A['C0_SSBO_PS'], 'ts': A['C0_SSBO_TS'], 'ms': A['C0_SSBO_MS']}
POOL_MIN = 33
DIM_BUFFER, DIM_SUBPASS = 5, 6


def decode_tex(name, s2layout=None):
    """dmsh_build.py decode_tex: (texture slot, sampler slot, handle type) of fp_t_cb{slots}_{handle} / fp_t_tcb_{handle}; the Switch 2 compute layout when given"""
    m = re.match(r'[a-z]+_[ti]\w*?_(?:cb(\d+)|tcb)_([0-9A-F]+)', name)
    if not m: return None
    slots = int(m.group(1)) if m.group(1) else -1; h = int(m.group(2), 16)
    if s2layout is not None and name.startswith('cp_'):
        o0, o1, ty = h & 0x3fff, (h >> 14) & 0x3fff, h >> 28
        tb = (slots & 0xffff) if slots >= 0 else 2; L = s2layout
        if tb != 2 or o0 * 4 < L['tex_base'] or (o0 * 4 - L['tex_base']) % ROW: return None, None, ty
        idx = (o0 * 4 - L['tex_base']) // ROW
        if re.match(r'[a-z]+_i', name): idx -= L['n_tex']
        if idx < 0 or idx >= 64: return None, None, ty
        return idx, None, ty
    tb = (slots & 0xffff) if slots >= 0 else 2; sb = ((slots >> 16) - 1) if slots >= 0 and (slots >> 16) else tb
    o0, o1, ty = h & 0x3fff, (h >> 14) & 0x3fff, h >> 28
    def slot(word, base, cb): return (word * 4 - base) // ROW if cb == 2 and word * 4 >= base and (word * 4 - base) % ROW == 0 else None
    words = [o0] + ([o1] if ty else [])
    if re.match(r'[a-z]+_i', name):
        return next((slot(x, IMG_BASE, tb) for x in words if IMG_BASE <= x * 4 < IMG_BASE + ROW * IMG_SLOTS and slot(x, IMG_BASE, tb) is not None), None), None, ty
    tw = [o0] if ty == 3 else words
    t = next((slot(x, TEX_BASE, tb) for x in tw if slot(x, TEX_BASE, tb) is not None and x * 4 < SEPTEX_BASE), None)
    if t is None: t = next((SEPTEX_SLOT0 + slot(x, SEPTEX_BASE, tb) for x in tw if slot(x, SEPTEX_BASE, tb) is not None and x * 4 < SAMP_BASE), None)
    s = next((slot(x, SAMP_BASE, sb) for x in words if x * 4 >= SAMP_BASE and slot(x, SAMP_BASE, sb) is not None), None) if ty != 3 else None
    return t, s, ty


def const_sampler(name):
    m = re.match(r'[a-z]+_t\w*?_(?:cb(\d+)|tcb)_([0-9A-F]+)', name)
    if not m: return None
    h = int(m.group(2), 16)
    return (h >> 14) & 0x3fff if h >> 28 == 3 else None


def decode_ubo_tex(name):
    """dmsh_build.py decode_ubo_tex: a bindless handle in the game's own uniform buffer → ((uniform slot, byte offset), sampler part)"""
    m = re.match(r'[a-z]+_t\w*?_cb(\d+)_([0-9A-F]+)', name)
    if not m: return None
    slots, h = int(m.group(1)), int(m.group(2), 16)
    tb = slots & 0xffff; sb = (slots >> 16) - 1 if slots >> 16 else tb
    o0, o1, ty = h & 0x3fff, (h >> 14) & 0x3fff, h >> 28
    if not 3 <= tb <= 16 or ty not in (0, 1, 3) or (ty == 1 and not 3 <= sb <= 16): return None
    return (tb - 3, o0 * 4), (('id', o1) if ty == 3 else (sb - 3, o1 * 4) if ty else (tb - 3, o0 * 4))


def c2_array(name, is_array):
    """dmsh_build.py c2_array: an array of c2 handles indexed at run time → (byte offset of element 0, byte offset of the sampler word or None)"""
    m = re.fullmatch(r'[a-z]+_t\w*?_cb(\d+)_([0-9A-F]+)', name)
    if not m or not is_array: return None
    slots, h = int(m.group(1)), int(m.group(2), 16)
    if h >> 28 == 0 and slots == 2: return (h & 0x3fff) * 4, None
    if h >> 28 == 1 and slots & 0xffff == 2 and (slots >> 16) - 1 == 2: return (h & 0x3fff) * 4, ((h >> 14) & 0x3fff) * 4
    return None


def s2_compute_layout(names, res):
    """dmsh_build.py reflect, SWITCH2 branch (DMSH_S2_CS=1, as the installer builds Pokopia): the per-program packed NVN2 compute driver bank, from what the code reads"""
    max_u = max([int(n[n.rindex('_c') + 2:]) - 3 for n in names if re.fullmatch(r'[a-z]+_c(\d+)', n) and int(n[n.rindex('_c') + 2:]) >= 3], default=-1)
    ssbo_base = 0x10 + A['C0_ROW'] * (max_u + 1)
    soffs = [b['cb_offset'] * 4 for b in res['sbufs'] if b['cb_slot'] == 0]
    max_s = max([(o - ssbo_base) // A['C0_ROW'] for o in soffs if o >= ssbo_base], default=-1)
    tex_base = ssbo_base + A['C0_ROW'] * (max_s + 1)
    toffs = [(int(m.group(2), 16) & 0x3fff) * 4 for m in (re.match(r'[a-z]+_([ti])\w*?_tcb_([0-9A-F]+)', n) for n in names) if m and m.group(1) == 't']
    max_t = max([(o - tex_base) // ROW for o in toffs if o >= tex_base], default=-1)
    return {'ssbo_base': ssbo_base, 'tex_base': tex_base, 'n_tex': max_t + 1}


def descriptor_of(m, ptr_type):
    """(descriptor type, element count or None for a runtime array, image dim) of a resource variable's pointer type"""
    t = m.type_desc(ptr_type)
    sc, x = t['sc'], t['of']
    count = 1
    while x['t'] == 'array':
        count = None if x['n'] is None or count is None else count * x['n']; x = x['of']
    if x['t'] == 'struct':
        blk = m.deco(x['id'], S.DecBlock) is not None
        if sc == S.SC_StorageBuffer or (sc == S.SC_Uniform and m.deco(x['id'], 3) is not None): return 'storage_buffer', count, None   # 3 = BufferBlock
        return ('uniform_buffer' if blk else 'push_constant' if sc == S.SC_PushConstant else 'struct'), count, None
    if x['t'] == 'sampler': return 'sampler', count, None
    if x['t'] == 'sampled_image':
        im = x['image']
        return ('uniform_texel_buffer' if im['dim'] == DIM_BUFFER else 'combined_image_sampler'), count, im['dim']
    if x['t'] == 'image':
        if x['dim'] == DIM_BUFFER: return ('storage_texel_buffer' if x['sampled'] == 2 else 'uniform_texel_buffer'), count, x['dim']
        if x['dim'] == DIM_SUBPASS: return 'input_attachment', count, x['dim']
        return ('storage_image' if x['sampled'] == 2 else 'sampled_image'), count, x['dim']
    return 'unknown', count, None


def comp_type(m, tid):
    """'float' / 'int' / 'uint' and the component count of an interface variable's type (arrays give their element's, and their length)"""
    t = m.type_desc(tid); n_arr = None
    if t['t'] == 'array': n_arr = t['n']; t = t['of']
    n = 1
    if t['t'] == 'vec': n = t['n']; t = t['of']
    return {'type': t['t'] + ('' if t.get('w', 32) == 32 else str(t.get('w'))), 'components': n, **({'array': n_arr} if n_arr else {})}


def fetch1d_vars(m, stage, res, s2_cs=False):
    """the combined image samplers the translator reads with 1D fetches as rows of 16384 texels (res.json fetch1d): their variable ids, for the fetch1d_tbuf edit"""
    slots = set(res.get('fetch1d', ()))
    if not slots: return set()
    rv = [(v, sc, p) for v, sc, p in m.global_vars() if sc in (S.SC_UniformConstant, S.SC_Uniform, S.SC_StorageBuffer, S.SC_PushConstant)]
    layout = s2_compute_layout([m.names.get(v, '') for v, _, _ in rv], res) if s2_cs and stage == 'cs' else None
    out = set()
    for v, sc, p in rv:
        nm = m.names.get(v, ''); desc, count, dim = descriptor_of(m, p)
        if sc != S.SC_UniformConstant or desc != 'combined_image_sampler' or count != 1 or re.match(r'[a-z]+_i', nm) or decode_ubo_tex(nm) or c2_array(nm, False) is not None: continue
        d = decode_tex(nm, layout)
        if d and d[0] in slots: out.add(v)
    return out


def reflect(key, stage, spv, res, s2_cs=False):
    m = S.Module(spv) if isinstance(spv, (bytes, bytearray)) else spv
    eps = m.entry_points(); ep = eps[0] if eps else None
    rv = [(v, sc, p) for v, sc, p in m.global_vars() if sc in (S.SC_UniformConstant, S.SC_Uniform, S.SC_StorageBuffer, S.SC_PushConstant)]
    names = [m.names.get(v, '') for v, _, _ in rv]
    layout = s2_compute_layout(names, res) if s2_cs and stage == 'cs' else None
    sbufs = {b['binding']: b for b in res.get('sbufs', [])}
    texr = {t['binding']: t for t in res.get('textures', []) + res.get('images', [])}
    atomic_vars = {x[3] for x in m.ins if x[0] & 0xffff == 60}   # OpImageTexelPointer: image atomics on that variable
    binds = []
    for v, sc, p in rv:
        nm = m.names.get(v, '')
        desc, count, dim = descriptor_of(m, p)
        b = {'set': (m.deco(v, S.DecDescriptorSet) or [0])[0], 'binding': (m.deco(v, S.DecBinding) or [None])[0], 'name': nm, 'descriptor': desc, 'count': count}
        n = {}
        if nm == 'support_buffer': n = {'kind': A['KIND_SUPPORT']}   # the translator's support buffer (Ryujinx's layout: render scale, viewport, fragment flags)
        elif re.fullmatch(r'[a-z]+_c(\d+)', nm):
            k = int(re.fullmatch(r'[a-z]+_c(\d+)', nm).group(1))
            n = {'kind': A['KIND_CBUF'], 'reg': k - 3} if 3 <= k <= 16 else {'kind': A['KIND_C1'] if k == 1 else A['KIND_DRVCB'], 'bank': k}
        elif desc == 'storage_buffer' or re.fullmatch(r'[a-z]+_s(\d+)', nm):
            sb = sbufs.get(b['binding'])
            base = SSBO.get(stage)
            if layout is not None: base = layout['ssbo_base']
            off = sb['cb_offset'] * 4 if sb and sb['cb_slot'] == 0 else None
            if stage == 'cs' and layout is None and off is not None and A['C0_CS_UBO'] <= off < A['C0_CS_UBO'] + A['C0_ROW'] * A['C0_CS_UBO_SLOTS'] and (off - A['C0_CS_UBO']) % A['C0_ROW'] == 0:
                n = {'kind': A['KIND_UBO_PTR'], 'reg': (off - A['C0_CS_UBO']) // A['C0_ROW']}
            elif sb and sb['cb_slot'] >= 3: n = {'kind': A['KIND_SBUF_UBO'], 'reg': sb['cb_slot'] - 3, 'off': sb['cb_offset'] * 4}
            else: n = {'kind': A['KIND_SBUF'], 'reg': (off - base) // A['C0_ROW'] if off is not None and base is not None and 0 <= off - base < A['C0_ROW'] * A['C0_SSBO_SLOTS'] and (off - base) % A['C0_ROW'] == 0 else None, 'c0_offset': off}
            n['rw'] = bool(sb and sb['flags'] & 1)
        elif desc in ('combined_image_sampler', 'sampled_image', 'uniform_texel_buffer', 'storage_image', 'storage_texel_buffer', 'sampler'):
            is_arr = count is None or count > 1
            pool = nm.endswith('_cb0_40000000') and (count is None or count >= POOL_MIN)
            ub = decode_ubo_tex(nm)
            c2a = c2_array(nm, is_arr)
            if pool: n = {'kind': A['KIND_SAMP_POOL'] if desc == 'sampler' else A['KIND_TEX_POOL']}
            elif ub and desc != 'sampler':
                ut, us = ub
                n = {'kind': A['KIND_UNKNOWN']} if dim == DIM_BUFFER else {'kind': A['KIND_TEX_UBO'], 'reg': ut[0], 'off': ut[1]}
                if desc == 'combined_image_sampler': n['sampler'] = {'kind': A['KIND_SAMP_ID'], 'reg': us[1]} if us[0] == 'id' else {'kind': A['KIND_SAMP_UBO'], 'reg': us[0], 'off': us[1]}
            elif c2a is not None:
                off, soff = c2a
                n = {'kind': A['KIND_SAMP_C2A'] if desc == 'sampler' else A['KIND_TEX_C2A'], 'off': off, **({'soff': soff} if soff is not None else {})}
            else:
                d = decode_tex(nm, layout)
                if not d: n = {'kind': A['KIND_UNKNOWN']}
                elif desc == 'sampler':
                    t, s, ty = d
                    n = {'kind': A['KIND_SAMP_ID'], 'reg': const_sampler(nm)} if ty == 3 else {'kind': A['KIND_SAMP'], 'reg': s} if ty else {'kind': A['KIND_SAMP_TEX'], 'reg': t}
                else:
                    t, s, ty = d
                    img = bool(re.match(r'[a-z]+_i', nm))
                    n = {'kind': A['KIND_IMG'] if img else (A['KIND_TBUF'] if dim == DIM_BUFFER else A['KIND_TEX']), 'reg': t}
                    if img and v in atomic_vars: n['atomic'] = True
                    if desc == 'combined_image_sampler':   # one Vulkan binding for texture and sampler: where the sampler comes from
                        n['sampler'] = {'kind': A['KIND_SAMP_ID'], 'reg': const_sampler(nm)} if ty == 3 else {'kind': A['KIND_SAMP'], 'reg': s} if ty else {'kind': A['KIND_SAMP_TEX'], 'reg': t}
                    if n['kind'] == A['KIND_TEX'] and t in res.get('fetch1d', ()): n['fetch1d'] = True
            tr = texr.get(b['binding'])
            if tr: n['translator'] = {'type': tr['type'], 'format': tr['format'], 'handle': tr['handle'], 'cbuf_slot': tr['cbuf_slot'], 'array': tr['array'], 'separate': tr['separate']}
        else: n = {'kind': A['KIND_UNKNOWN']}
        b['nvn'] = n
        binds.append(b)
    binds.sort(key=lambda b: (b['set'], b['binding'] if b['binding'] is not None else -1))
    # the stage interface
    def iface(sc):
        out, builtins = [], []
        for v, s, p in m.global_vars():
            if s != sc: continue
            bi = m.deco(v, S.DecBuiltIn)
            pointee = m.inst(p)[3]
            if bi is not None: builtins.append(bi[0]); continue
            st = m.inst(pointee)
            if st[0] & 0xffff == S.OpTypeStruct or (st[0] & 0xffff == S.OpTypeArray and m.inst(st[2])[0] & 0xffff == S.OpTypeStruct):
                sid = st[1] if st[0] & 0xffff == S.OpTypeStruct else st[2]
                for mi, ds in sorted(m.member_decor.get(sid, {}).items()):
                    builtins += [a[0] for d, a in ds if d == S.DecBuiltIn and a]
                continue
            loc = m.deco(v, S.DecLocation)
            if loc is None: continue
            e = {'location': loc[0], **comp_type(m, pointee)}
            c = m.deco(v, S.DecComponent)
            if c: e['component'] = c[0]
            ix = m.deco(v, S.DecIndex)
            if ix: e['index'] = ix[0]
            for dn, dv in (('flat', S.DecFlat), ('noperspective', 13), ('centroid', 16), ('sample', 17)):
                if m.deco(v, dv) is not None: e[dn] = True
            out.append(e)
        return sorted(out, key=lambda e: (e['location'], e.get('component', 0))), sorted(set(builtins))
    ins, bin_ = iface(S.SC_Input)
    outs, bout = iface(S.SC_Output)
    modes = []
    for x in m.ins:
        if x[0] & 0xffff in (S.OpExecutionMode, S.OpExecutionModeId): modes.append([x[2]] + x[3:])
    local = next(([md[1], md[2], md[3]] for md in modes if md[0] == 17), None)   # LocalSize
    caps = [S.CAP_NAMES.get(c, str(c)) for c in m.caps()]
    exts = [S.string_of(x[1:]) for x in m.ins if x[0] & 0xffff == S.OpExtension]
    return {'key': key, 'stage': stage, 'entry': ep['name'] if ep else None, 'execution_model': S.EXEC_MODELS.get(ep['model']) if ep else None,
            'spirv_version': '%d.%d' % m.version, 'execution_modes': modes, **({'local_size': local} if local else {}),
            'capabilities': caps, 'extensions': exts, 'uses_subgroups': any(333 <= x[0] & 0xffff <= 366 or x[0] & 0xffff in (4421, 4428, 4429, 4430, 4431, 4432) for x in m.ins),
            'bindings': binds, 'inputs': ins, 'outputs': outs, 'builtins_in': bin_, 'builtins_out': bout,
            'drv_reads': drv_reads(m, rv), **({'s2_compute_layout': layout} if layout else {}),
            'translator': {k: res[k] for k in ('notes', 'geometry_stride', 'uses_frag_coord', 'frag_out_map', 'fetch1d') if k in res}}


def drv_reads(m, rv):
    """Rows of the driver segments c0 / c2 the shader reads (dmsh_build.py drv_reads, from the access chains): constant row → byte offset; any other index is dynamic"""
    drv = {v: int(re.fullmatch(r'[a-z]+_c(\d+)', m.names.get(v, '')).group(1)) for v, _, _ in rv if re.fullmatch(r'[a-z]+_c[02]', m.names.get(v, ''))}
    out = {}
    for x in m.ins:
        if x[0] & 0xffff in (S.OpAccessChain, S.OpInBoundsAccessChain) and x[3] in drv and len(x) > 5:
            bank = str(drv[x[3]]); d = out.setdefault(bank, {'vec4_offsets': set(), 'dynamic': False})
            row = m.const_value(x[5])
            if row is None: d['dynamic'] = True
            else: d['vec4_offsets'].add(row * 16)
    return {k: {'vec4_offsets': sorted(v['vec4_offsets']), 'dynamic': v['dynamic']} for k, v in out.items()}
