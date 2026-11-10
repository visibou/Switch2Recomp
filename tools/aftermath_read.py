"""Reads an NVIDIA Nsight Aftermath GPU crash dump written with NXRT_VK_AFTERMATH=1 (runtime/nvn/vk/vk_aftermath.c).

  python tools/aftermath_read.py [DUMP or run folder]      (default: the newest dump under games/MKWTest/aftermath)

Writes <dump>.json (the SDK decoder's full report) and prints: the device state, page fault, active shaders with our shader keys
(from shaders.tsv), faulted warps, and for each faulting instruction the SPIR-V around it, from the line markers (the line of an
OpLine is the word offset of the instruction in spv/<hash>.src.spv). Needs the SDK's DLL (external/aftermath) and spirv-dis."""
import ctypes as C, glob, json, os, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AM = os.path.join(ROOT, 'games', 'MKWTest', 'aftermath')
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import s2r_paths as P
DLL = P.ext('aftermath', 'lib', 'x64', 'GFSDK_Aftermath_Lib.x64.dll')
DIS = P.spirv_tool('spirv-dis') or 'spirv-dis'
VERSION = 0x21B
SUCCESS = 1

SetData = C.CFUNCTYPE(None, C.c_void_p, C.c_uint32)
DbgLookup = C.CFUNCTYPE(None, C.POINTER(C.c_uint64 * 2), SetData, C.c_void_p)
ShLookup = C.CFUNCTYPE(None, C.POINTER(C.c_uint64), SetData, C.c_void_p)
SrcLookup = C.CFUNCTYPE(None, C.c_void_p, SetData, C.c_void_p)


def newest_dump(arg):
    if arg and os.path.isfile(arg): return arg
    base = arg if arg and os.path.isdir(arg) else AM
    d = sorted(glob.glob(os.path.join(base, '**', '*.nv-gpudmp'), recursive=True) + glob.glob(os.path.join(base, '**', 'vendor_fault.bin'), recursive=True), key=os.path.getmtime)
    if not d: sys.exit('no dump under ' + base)
    return d[-1]


def tsv_map(run):
    m = {}
    p = os.path.join(run, 'shaders.tsv')
    if os.path.exists(p):
        for ln in open(p, encoding='utf-8', errors='replace').read().splitlines()[1:]:
            f = ln.split('\t')
            if len(f) >= 3: m.setdefault(int(f[0], 16), set()).add('%s (%s)' % (f[1], f[2]))
    return m


def decode(path):
    run = os.path.dirname(path)
    spv = os.path.join(AM, 'spv')
    lib = C.CDLL(DLL)
    data = open(path, 'rb').read()
    dec = C.c_void_p()
    r = lib.GFSDK_Aftermath_GpuCrashDump_CreateDecoder(C.c_uint32(VERSION), data, C.c_uint32(len(data)), C.byref(dec))
    if r != SUCCESS: sys.exit('the decoder refused %s (%#x)' % (path, r & 0xffffffff))
    keep = []

    def give(setter, p):
        if p and os.path.exists(p):
            b = C.create_string_buffer(open(p, 'rb').read()); keep.append(b); setter(C.cast(b, C.c_void_p), len(b.raw) - 1)

    @DbgLookup
    def dbg(ident, setter, u):
        i = ident.contents
        give(setter, os.path.join(run, 'shader-%016x%016x.nvdbg' % (i[0], i[1])))

    @ShLookup
    def sh(h, setter, u):
        give(setter, os.path.join(spv, '%016x.spv' % h.contents.value))

    @SrcLookup
    def src(name, setter, u):
        pass

    n = C.c_uint32()
    r = lib.GFSDK_Aftermath_GpuCrashDump_GenerateJSON(dec, C.c_uint32(0x3FFF), C.c_uint32(0x2), dbg, sh, src, None, C.byref(n))
    if r != SUCCESS: sys.exit('GenerateJSON failed (%#x)' % (r & 0xffffffff))
    buf = C.create_string_buffer(n.value)
    lib.GFSDK_Aftermath_GpuCrashDump_GetJSON(dec, n, buf)
    lib.GFSDK_Aftermath_GpuCrashDump_DestroyDecoder(dec)
    txt = buf.value.decode('utf-8', 'replace')
    out = path + '.json'
    open(out, 'w', encoding='utf-8').write(txt)
    return json.loads(txt), out, run


def walk(o, key):
    """Every value under a key of this name, anywhere in the report."""
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key: yield v
            yield from walk(v, key)
    elif isinstance(o, list):
        for v in o: yield from walk(v, key)


def spirv_around(h, word, ctx=8):
    p = os.path.join(AM, 'spv', '%016x.src.spv' % h)
    if not os.path.exists(p) or not os.path.exists(DIS): return ['(no %s)' % p]
    r = subprocess.run([DIS, '--offsets', '--no-color', p], capture_output=True, text=True)
    lines = [' '.join(ln.split()) for ln in r.stdout.splitlines()]
    at = next((i for i, ln in enumerate(lines) if ln.endswith('; 0x%08x' % (word * 4))), None)
    if at is None: return ['(word %d not found in %s)' % (word, p)]
    return [('>> ' if i == at else '   ') + lines[i] for i in range(max(0, at - ctx), min(len(lines), at + 3))]


def section(rep, name):
    for sec in rep if isinstance(rep, list) else [rep]:
        if isinstance(sec, dict) and name in sec: return sec[name]
    return None


def main():
    path = newest_dump(sys.argv[1] if len(sys.argv) > 1 else '')
    rep, out, run = decode(path)
    names = tsv_map(run)
    print('dump', path)
    print('full report', out)
    print('device:', json.dumps(section(rep, 'Device info')))
    pf = section(rep, 'Page fault info')
    if pf: print('page fault:', json.dumps(pf)[:1500])
    infos = section(rep, 'Shader infos') or {}
    infos = infos.get('Info') if isinstance(infos, dict) else infos
    infos = infos if isinstance(infos, list) else [infos] if infos else []
    byname = {}
    for i in infos:
        h = int(i.get('Shader hash', 0)); byname[i.get('Shader name')] = h
        print('shader %s: %s, %s bytes, hash %016x = %s' % (i.get('Shader name'), i.get('Shader type'), i.get('Shader size'), h, ', '.join(sorted(names.get(h, {'(not in shaders.tsv)'})))))
    for kind in ('Faulted Warps', 'Active Warps'):
        for w in section(rep, kind) or []:
            print('%s: %s' % (kind[:-1].lower(), json.dumps(w)))
            pc, mp = w.get('Shader GPU PC Address') or w.get('GPU PC Address') or '', w.get('Shader mapping') or ''
            h = byname.get(pc.split(' @')[0])
            if h and mp.startswith('s2r.spv:'): print('\n'.join(spirv_around(h, int(mp.split(':')[1]))))
    ev = ((section(rep, 'Aftermath markers') or {}).get('Context') or {}).get('Events') or []
    for e in ev:
        E = e.get('Event', {}); d = (E.get('Library') or {}).get('Data', {}).get('Data chunk')
        if d: print('marker %-10s %s' % (E.get('Status'), bytes(d).rstrip(b'\0').decode('utf-8', 'replace')))
    return 0


if __name__ == '__main__':
    sys.exit(main())
