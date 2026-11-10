#!/usr/bin/env python3
"""gen_hle.py: the registry of nxnative's host replacements and their AAPCS64 adapters (DESIGN D8, research/02 §3.3).

Reads nxnative's sources as text, as nxanalyze does (src/nxanalyze/src/hle.cpp): every HLE_FORCE / HLE_FORCE_BIG and the file-local macros built on
them, HLE (the swhle table), hle_table.c's L() passthroughs, the NVN entry points (dm/gfx/nvn/names.txt, with signatures from hle_nvn.c's NVN /
NVN_NOPROTO lines or nvn_protos.h). A file that nxrt has ported (same name in one of the --port folders) is read from the port instead, and its
adapters go into <file>.adapters.inc, which the port includes at its end: the adapter then calls the ported static host function directly.
Every other entry gets a stub adapter: it logs its name once and returns a neutral result (0 in x0/x1 and v0-v3, a zeroed x8 buffer); when
nxnative's body is only `return <constant>;` (after (void) casts), the stub returns that constant instead ("trivial").

Outputs (in --out): hle_registry.gen.c, hle_stubs.gen.c, hle_adapters.h, <file>.adapters.inc per ported file, hle_inventory.tsv.
--check compares instead of writing (exit 1 on a difference). Exit 77 when the nxnative tree is missing.

An adapter `void adapter_<symbol>(NxwGuestCpu *cpu)` reads its arguments as AAPCS64 places them: integers and pointers in x0-x7, then 8-byte stack
slots at sp; floats and doubles in v0-v7 (s/d), then the stack; composites up to 16 bytes in one or two x registers, homogeneous float aggregates in
consecutive v registers, larger composites by reference. Narrow integers arrive unextended (the cast truncates); variadic arguments arrive in
registers (nxnative declares them as fixed parameters). Results: x0 (x1), v0-v3 for floats and float aggregates, *x8 for composites over 16 bytes.
Win64 is LLP64, so `long` is refused in ported signatures (the Switch's long is int64_t).
"""
import argparse
import os
import re
import sys

# ---------------------------------------------------------------------------------------------------------------------------------- types

INT = {  # name -> (bytes, signed)
    'char': (1, True), 'signed char': (1, True), 'unsigned char': (1, False), 'short': (2, True), 'short int': (2, True),
    'unsigned short': (2, False), 'unsigned short int': (2, False), 'int': (4, True), 'signed': (4, True), 'signed int': (4, True),
    'unsigned': (4, False), 'unsigned int': (4, False), 'long long': (8, True), 'long long int': (8, True), 'unsigned long long': (8, False),
    'int8_t': (1, True), 'uint8_t': (1, False), 'int16_t': (2, True), 'uint16_t': (2, False), 'int32_t': (4, True), 'uint32_t': (4, False),
    'int64_t': (8, True), 'uint64_t': (8, False), 'size_t': (8, False), 'ssize_t': (8, True), 'intptr_t': (8, True), 'uintptr_t': (8, False),
    'ptrdiff_t': (8, True), 'bool': (1, False), '_Bool': (1, False), 'Result': (4, False), 'time_t': (8, True), 'char16_t': (2, False),
}
LONG = {'long': (8, True), 'long int': (8, True), 'unsigned long': (8, False), 'unsigned long int': (8, False), 'signed long': (8, True)}
FLOAT = {'float': 4, 'double': 8}
# Composites passed or returned by value: name -> (bytes, kind) with kind 'int' (x registers), 'f32'/'f64' (a homogeneous float aggregate)
STRUCTS = {
    'NVNseparateTextureHandle': (8, 'int'), 'NVNseparateSamplerHandle': (8, 'int'), 'NVNcounterData': (16, 'int'),
    'GLSLCversion': (0x34, 'int'),                                            # glslcGetVersion's result, through x8 (hle_glslc.c)
    'NxrtTestFloat4': (16, 'f32'), 'NxrtTestDouble2': (16, 'f64'),            # tests only: the float-aggregate paths
}


class TypeError_(Exception):
    pass


class CType:
    """cls: void, int, ptr, f32, f64, struct; size; signed; hfa: element count for float aggregates"""

    def __init__(self, text, cls, size=8, signed=False, kind='int'):
        self.text, self.cls, self.size, self.signed, self.kind = text, cls, size, signed, kind

    @property
    def hfa(self):
        return self.cls == 'struct' and self.kind in ('f32', 'f64')

    @property
    def nregs(self):
        if self.hfa:
            return self.size // (4 if self.kind == 'f32' else 8)
        return (self.size + 7) // 8


def strip_quals(t):
    t = re.sub(r'\b(const|volatile|restrict|__restrict|struct|enum)\b', ' ', t)
    return ' '.join(t.split())


class Types:
    def __init__(self, nvn_h=None):
        self.alias = {}       # typedef name -> base text
        self.enums = set()
        self.fnptr = {'nv_alloc_fn', 'nv_free_fn', 'nv_realloc_fn',   # hle_nv.c's allocator callbacks
                      'ikb_cb0', 'ikb_cb1'}                           # hle_swkbd.c's inline keyboard callbacks (guest addresses)
        self.structs = dict(STRUCTS)
        if nvn_h and os.path.exists(nvn_h):
            self.read_typedefs(open(nvn_h, encoding='utf-8', errors='replace').read())

    def read_typedefs(self, s):
        s = re.sub(r'/\*.*?\*/', ' ', s, flags=re.S)
        s = re.sub(r'//[^\n]*', ' ', s)
        for m in re.finditer(r'typedef\s+enum\s*\w*\s*\{[^}]*\}\s*(\w+)\s*;', s):
            self.enums.add(m.group(1))
        for m in re.finditer(r'typedef\s+[^;{}]*?\(\s*\*\s*(\w+)\s*\)\s*\([^;]*\)\s*;', s):
            self.fnptr.add(m.group(1))
        for m in re.finditer(r'typedef\s+(?!enum\b|struct\b)([\w \t]+?\**)\s*\b(\w+)\s*;', s):
            self.alias[m.group(2)] = m.group(1).strip()

    def classify(self, text, where, ported):
        t = text.strip()
        if '(' in t or '[' in t or '*' in t:
            return CType(text, 'ptr')
        base = strip_quals(t)
        for _ in range(8):
            if base in self.alias:
                base = strip_quals(self.alias[base])
            else:
                break
        if '*' in base:
            return CType(text, 'ptr')
        if base == 'void':
            return CType(text, 'void', 0)
        if base in INT:
            n, s = INT[base]
            return CType(text, 'int', n, s)
        if base in LONG:
            if ported:
                raise TypeError_(f'{where}: `{text}` in a ported signature: long is 32-bit on Windows, use int64_t / uint64_t')
            n, s = LONG[base]
            return CType(text, 'int', n, s)
        if base in FLOAT:
            return CType(text, 'f32' if base == 'float' else 'f64', FLOAT[base])
        if base in self.enums:
            return CType(text, 'int', 4, True)
        if base in self.fnptr or (base.startswith('PFN') and base.endswith('PROC')):
            return CType(text, 'ptr')
        if base in self.structs:
            n, k = self.structs[base]
            return CType(text, 'struct', n, False, k)
        raise TypeError_(f'{where}: unknown type `{text}` (add it to gen_hle.py\'s tables)')


# ---------------------------------------------------------------------------------------------------------------------------------- C text

def strip_c(s):
    """Comments out (newlines kept), string and character literals kept, then #if 0 blocks out (as nxanalyze's strip)"""
    o, i, n = [], 0, len(s)
    while i < n:
        if s.startswith('//', i):
            while i < n and s[i] != '\n':
                i += 1
        elif s.startswith('/*', i):
            e = s.find('*/', i + 2)
            e = n if e < 0 else e + 2
            o.append('\n' * s.count('\n', i, e))
            i = e
        elif s[i] in '"\'':
            q = s[i]
            j = i + 1
            while j < n and s[j] != q:
                j += 2 if s[j] == '\\' else 1
            o.append(s[i:j + 1])
            i = j + 1
        else:
            o.append(s[i])
            i += 1
    out, skip = [], 0
    for line in ''.join(o).split('\n'):
        t = line.lstrip()
        if skip:
            if t.startswith('#if'):
                skip += 1
            elif t.startswith('#endif'):
                skip -= 1
            out.append('')
            continue
        if t.startswith('#if 0'):
            skip = 1
            out.append('')
            continue
        out.append(line)
    return '\n'.join(out)


def balanced(s, p):
    """s[p] == '(': the index of the matching ')'"""
    depth, i = 0, p
    while i < len(s):
        c = s[i]
        if c in '"\'':
            j = i + 1
            while j < len(s) and s[j] != c:
                j += 2 if s[j] == '\\' else 1
            i = j
        elif c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError('unbalanced parentheses')


def split_args(s):
    out, depth, cur, i = [], 0, [], 0
    while i < len(s):
        c = s[i]
        if c in '"\'':
            j = i + 1
            while j < len(s) and s[j] != c:
                j += 2 if s[j] == '\\' else 1
            cur.append(s[i:j + 1])
            i = j + 1
            continue
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        if c == ',' and depth == 0:
            out.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    if ''.join(cur).strip() or out:
        out.append(''.join(cur).strip())
    return out


def string_value(s):
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', s)
    return ''.join(parts)


def param_name(p):
    """'const char *name' -> ('const char *', 'name'); a function pointer keeps its whole text as the type"""
    p = ' '.join(p.split())
    m = re.match(r'^(.*\(\s*\*\s*)(\w+)(\s*\).*)$', p)
    if m:
        return (m.group(1) + m.group(3)).strip(), m.group(2)
    m = re.match(r'^(.*?)(\w+)\s*(\[[^\]]*\])$', p)
    if m:
        return m.group(1).strip() + ' *', m.group(2)
    m = re.match(r'^(.*?[\s*])(\w+)$', p)
    if m and m.group(1).strip():
        return m.group(1).strip(), m.group(2)
    return p, None


def ident(c):
    return c.isalnum() or c == '_'


class Entry:
    def __init__(self, name, kind, file, why, ret, cname, params, line, body=None):
        self.name, self.kind, self.file, self.why, self.ret, self.cname, self.params, self.line, self.body = name, kind, file, why, ret, cname, params, line, body
        self.status = 'stub'
        self.trivial = None
        self.sig_known = True


def find_invocations(s, macros):
    """(macro, args text, start, end, line) for NAME( ... ) at identifier boundaries, outside #define lines"""
    names = '|'.join(sorted(map(re.escape, macros), key=len, reverse=True))
    for m in re.finditer(r'(?<![\w])(' + names + r')\s*\(', s):
        ls = s.rfind('\n', 0, m.start()) + 1
        if s[ls:m.start()].lstrip().startswith('#define'):
            continue
        p = s.index('(', m.start())
        e = balanced(s, p)
        yield m.group(1), s[p + 1:e], m.start(), e + 1, s.count('\n', 0, m.start()) + 1


def body_after(s, e):
    """The { ... } body after an invocation, or None"""
    j = e
    while j < len(s) and s[j] in ' \t\r\n':
        j += 1
    if j < len(s) and s[j] == '{':
        depth, k = 0, j
        while k < len(s):
            if s[k] == '{':
                depth += 1
            elif s[k] == '}':
                depth -= 1
                if depth == 0:
                    return s[j:k + 1]
            elif s[k] in '"\'':
                q = s[k]
                k += 1
                while k < len(s) and s[k] != q:
                    k += 2 if s[k] == '\\' else 1
            k += 1
    return None


def local_macros(s):
    """file-local function-like macros whose body invokes HLE_FORCE: name -> (params, body)"""
    out = {}
    for m in re.finditer(r'^[ \t]*#define[ \t]+(\w+)\(([^)]*)\)[ \t]*(.*)$', s, re.M):
        name, params, body = m.group(1), [x.strip() for x in m.group(2).split(',')], m.group(3)
        if re.match(r'(HLE_FORCE|HLE_FORCE_BIG)\s*\(', body.strip()):
            out[name] = (params, body)
    return out


def expand(macro, args, params, body):
    vals = split_args(args)
    named = [p for p in params if p != '...']
    rep = dict(zip(named, vals))
    if '...' in params:
        rep['__VA_ARGS__'] = ', '.join(vals[len(named):])
    # #param stringifies and a ## b pastes (hle_hid.c's TOUCH(N)); plain parameters are replaced
    body = re.sub(r'(?<!#)#\s*(\w+)', lambda m: '"' + rep[m.group(1)] + '"' if m.group(1) in rep else m.group(0), body)
    body = re.sub(r'\b(\w+)\b', lambda m: rep.get(m.group(1), m.group(1)), body)
    return re.sub(r'\s*##\s*', '', body)


def parse_file(path, fname, ported):
    s = strip_c(open(path, encoding='utf-8', errors='replace').read())
    s = re.sub(r'\\\n', '  ', s)   # a #define continued over several lines is one line (hle_hid.c's TOUCH(N))
    out = []
    macros = local_macros(s)
    names = {'HLE_FORCE', 'HLE_FORCE_BIG', 'HLE', 'NVN', 'NVN_NOPROTO', *macros} | ({'L'} if fname == 'hle_table.c' else set())
    for mac, args, st, en, line in find_invocations(s, names):
        if mac in macros:
            # every HLE_FORCE the expansion holds (hle_hid.c's TOUCH(N) defines two); a body outside the expansion belongs to the last one
            text = expand(mac, args, *macros[mac])
            uses = list(re.finditer(r'(?<![\w])(HLE_FORCE_BIG|HLE_FORCE)\s*\(', text))
            for k, m2 in enumerate(uses):
                p = text.index('(', m2.start())
                e = balanced(text, p)
                inner = text[p + 1:e]
                body = body_after(text, e + 1) or (body_after(s, en) if k == len(uses) - 1 else None)
                a = split_args(inner)
                out.append(Entry(string_value(a[0]), 'forced', fname, string_value(a[1]), a[2], a[3], a[4:], line, body))
            continue
        a = split_args(args)
        body = body_after(s, en)
        if mac in ('HLE_FORCE', 'HLE_FORCE_BIG'):
            if not a or not a[0].startswith('"'):
                continue
            out.append(Entry(string_value(a[0]), 'forced', fname, string_value(a[1]), a[2], a[3], a[4:], line, body))
        elif mac == 'HLE':
            if not a or not a[0].startswith('"'):
                continue
            out.append(Entry(string_value(a[0]), 'swhle', fname, 'C library passthrough', a[1], a[2], a[3:], line, body))
        elif mac in ('NVN', 'NVN_NOPROTO'):
            if not a or not re.match(r'^nvn\w+$', a[0]):
                continue
            out.append(Entry(a[0], 'nvn', fname, 'NVN entry point', a[1], 'h_' + a[0], a[2:], line, body))
        elif mac == 'L' and fname == 'hle_table.c' and len(a) == 1 and re.match(r'^\w+$', a[0]):
            e = Entry(a[0], 'swhle', fname, 'C library passthrough', None, None, None, line)
            e.sig_known = False
            out.append(e)
    return out


TRIVIAL_RET = r'(RES_OK|RES_FAIL|NULL|true|false|-?\d+|0x[0-9a-fA-F]+u?|-?\d+u?)'


def trivial_value(body):
    if body is None:
        return None
    b = ' '.join(body.split())
    m = re.match(r'^\{ ((?:\(void\) ?\w+; ?)*)(?:return ' + TRIVIAL_RET + r'; ?)?\}$', b)
    if not m:
        return None
    v = m.group(2)
    if v is None:
        return 'void'
    return {'RES_OK': '0', 'RES_FAIL': '0x2A2', 'NULL': '0', 'true': '1', 'false': '0'}.get(v, v.rstrip('u'))


def parse_protos(path):
    out = {}
    if not os.path.exists(path):
        return out
    s = open(path, encoding='utf-8').read()
    for m in re.finditer(r'typedef (.+?) \(\*NVNT_(\w+)\)\((.*?)\);', s):
        ret, name, args = m.groups()
        params = [] if args.strip() in ('', 'void') else [a.strip() for a in split_args(args)]
        out[name] = (ret.strip(), params)
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- code

def c_str(s):
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', ' ') + '"'


def arg_reads(params, where, types, ported):
    """C statements reading each parameter as AAPCS64 passes it; returns (lines, names)"""
    lines, names = [], []
    ngrn = nsrn = nsaa = 0
    for i, p in enumerate(params):
        t_text, name = param_name(p)
        t = types.classify(t_text, f'{where} parameter {i + 1}', ported)
        v = f'a{i}'
        names.append(v)
        decl = t_text
        if t.cls in ('int', 'ptr'):
            src = f'x[{ngrn}]' if ngrn < 8 else f'nxrt_stack64(cpu, {nsaa})'
            if ngrn < 8:
                ngrn += 1
            else:
                nsaa += 8
            if '(' in t_text:   # a function pointer: the name goes inside its declarator
                lines.append(f'    {re.sub(r"[(][*][)]", "(*" + v + ")", t_text.replace(" ", ""), count=1)} = ({t_text})(uintptr_t){src};')
            else:
                cast = f'({decl})(uintptr_t){src}' if t.cls == 'ptr' else f'({decl}){src}'
                lines.append(f'    {decl} {v} = {cast};')
        elif t.cls in ('f32', 'f64'):
            fn = 'nxrt_ldf32' if t.cls == 'f32' else 'nxrt_ldf64'
            if nsrn < 8:
                lines.append(f'    {decl} {v} = ({decl}){fn}(cpu->q[{nsrn}]);')
                nsrn += 1
            else:
                lines.append(f'    {decl} {v} = ({decl}){fn}((const uint8_t *)(uintptr_t)(cpu->sp + {nsaa}));')
                nsaa += 8
        elif t.cls == 'struct':
            lines.append(f'    {decl} {v}; _Static_assert(sizeof {v} == {t.size}, "{t.text}: the size gen_hle.py assumes");')
            if t.hfa:
                k, es = t.nregs, (4 if t.kind == 'f32' else 8)
                if nsrn + k <= 8:
                    for j in range(k):
                        lines.append(f'    memcpy((uint8_t *)&{v} + {j * es}, cpu->q[{nsrn + j}], {es});')
                    nsrn += k
                else:
                    nsrn = 8
                    lines.append(f'    memcpy(&{v}, (const void *)(uintptr_t)(cpu->sp + {nsaa}), {t.size});')
                    nsaa += (t.size + 7) // 8 * 8
            elif t.size <= 16:
                k = t.nregs
                if ngrn + k <= 8:
                    lines.append(f'    memcpy(&{v}, &x[{ngrn}], {t.size});')
                    ngrn += k
                else:
                    ngrn = 8
                    lines.append(f'    memcpy(&{v}, (const void *)(uintptr_t)(cpu->sp + {nsaa}), {t.size});')
                    nsaa += (t.size + 7) // 8 * 8
            else:   # by reference
                src = f'x[{ngrn}]' if ngrn < 8 else f'nxrt_stack64(cpu, {nsaa})'
                if ngrn < 8:
                    ngrn += 1
                else:
                    nsaa += 8
                lines.append(f'    memcpy(&{v}, (const void *)(uintptr_t){src}, {t.size});')
        else:
            raise TypeError_(f'{where}: parameter `{p}` of type void')
    return lines, names


def result_write(t, expr):
    if t.cls == 'void':
        return [f'    {expr};']
    if t.cls == 'int':
        conv = '(uint64_t)(int64_t)r' if t.signed else '(uint64_t)r'
        return [f'    {t.text} r = {expr};', f'    x[0] = {conv};']
    if t.cls == 'ptr':
        return [f'    {t.text} r = {expr};', '    x[0] = (uint64_t)(uintptr_t)r;']
    if t.cls in ('f32', 'f64'):
        fn = 'nxrt_stf32' if t.cls == 'f32' else 'nxrt_stf64'
        return [f'    {t.text} r = {expr};', f'    {fn}(cpu->q[0], r);']
    out = [f'    {t.text} r = {expr};', f'    _Static_assert(sizeof r == {t.size}, "{t.text}: the size gen_hle.py assumes");']
    if t.hfa:
        es = 4 if t.kind == 'f32' else 8
        for j in range(t.nregs):
            out.append(f'    memset(cpu->q[{j}], 0, 16); memcpy(cpu->q[{j}], (const uint8_t *)&r + {j * es}, {es});')
    elif t.size <= 16:
        out += ['    x[0] = x[1] = 0;', f'    memcpy(&x[0], &r, {t.size});']
    else:
        out.append(f'    memcpy((void *)(uintptr_t)x[8], &r, {t.size});')
    return out


def adapter_ported(e, types):
    where = f'{e.file}:{e.line} {e.name}'
    ret = types.classify(e.ret, f'{where} result', True)
    params = [] if len(e.params) == 1 and e.params[0].strip() == 'void' or not e.params else e.params
    reads, names = arg_reads(params, where, types, True)
    out = [f'NXRT_API void adapter_{e.name}(NxwGuestCpu *cpu) {{', '    uint64_t *x = cpu->x; (void)x;']
    out += reads
    out += result_write(ret, f'{e.cname}({", ".join(names)})')
    out.append('}')
    return out


def adapter_stub(e, types):
    out = [f'NXRT_API void adapter_{e.name}(NxwGuestCpu *cpu) {{']
    big = 0
    if e.sig_known:
        ret = types.classify(e.ret, f'{e.file}:{e.line} {e.name} result', False)
        if ret.cls == 'struct' and not ret.hfa and ret.size > 16:
            big = ret.size
    out.append(f'    static uint8_t seen; nxrt_stub_log({c_str(e.name)}, {c_str(e.file)}, &seen, cpu);')
    if big:
        out.append(f'    if (cpu->x[8]) memset((void *)(uintptr_t)cpu->x[8], 0, {big});')
    out.append('    cpu->x[0] = cpu->x[1] = 0; memset(cpu->q, 0, 4 * 16);')
    out.append('}')
    return out


def adapter_trivial(e, types):
    ret = types.classify(e.ret, f'{e.file}:{e.line} {e.name} result', False)
    out = [f'NXRT_API void adapter_{e.name}(NxwGuestCpu *cpu) {{   /* nxnative: {e.trivial if e.trivial != "void" else "no-op"} */']
    if ret.cls == 'void':
        out.append('    (void)cpu;')
    elif ret.cls == 'int':   # the value in the result's own width (its C name may not be declared here)
        w = f'{"" if ret.signed else "u"}int{ret.size * 8}_t'
        out.append(f'    cpu->x[0] = (uint64_t){"(int64_t)" if ret.signed else ""}({w})({e.trivial});')
    elif ret.cls == 'ptr':
        out.append(f'    cpu->x[0] = (uint64_t)(uintptr_t)({e.trivial});')
    else:
        return adapter_stub(e, types)
    out.append('}')
    return out


HEADER = '/* Generated by src/nxrt/gen/gen_hle.py from nxnative (MIT, © seedds); do not edit. */\n'
HELPERS = '''#include <string.h>
#include "hle_registry.h"
#include "nxrt/rt_abi.h"
#ifndef NXRT_HLE_HELPERS
#define NXRT_HLE_HELPERS
static inline uint64_t nxrt_stack64(const NxwGuestCpu *cpu, uint64_t off) { uint64_t v; memcpy(&v, (const void *)(uintptr_t)(cpu->sp + off), 8); return v; }
static inline float nxrt_ldf32(const uint8_t *q) { float f; memcpy(&f, q, 4); return f; }
static inline double nxrt_ldf64(const uint8_t *q) { double d; memcpy(&d, q, 8); return d; }
static inline void nxrt_stf32(uint8_t *q, float f) { memset(q, 0, 16); memcpy(q, &f, 4); }
static inline void nxrt_stf64(uint8_t *q, double d) { memset(q, 0, 16); memcpy(q, &d, 8); }
#endif
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nxnative', required=True)
    ap.add_argument('--port', action='append', default=[])
    ap.add_argument('--extra', action='append', default=[], help='an extra ported file not in nxnative (tests)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    root = a.nxnative
    host, nvndir = os.path.join(root, 'dm', 'host'), os.path.join(root, 'dm', 'gfx', 'nvn')
    if not os.path.isdir(host) or not os.path.isdir(nvndir):
        print(f'gen_hle: no nxnative tree at {root}', file=sys.stderr)
        return 77
    types = Types(os.path.join(nvndir, 'include', 'nvn.h'))
    ports = {}
    for d in a.port:
        for f in sorted(os.listdir(d)):
            if f.endswith('.c'):
                ports[f] = os.path.join(d, f)
    gfx = os.path.join(root, 'dm', 'gfx')   # hle_nv.c and hle_vi.c take over nv:: and vi:: (nxanalyze reads them too)
    files = [(f, os.path.join(host, f)) for f in sorted(os.listdir(host)) if f.endswith('.c')] +             [(f, os.path.join(gfx, f)) for f in sorted(os.listdir(gfx)) if f.startswith('hle_') and f.endswith('.c')] + [('hle_nvn.c', os.path.join(nvndir, 'hle_nvn.c'))]
    entries, problems, ported_files = [], [], []
    original = {}
    for f, p in files:
        if f in ports:
            original[f] = parse_file(p, f, False)
            got = parse_file(ports[f], f, True)
            if got:
                ported_files.append(f)
            for e in got:
                e.status = 'ported'
            entries += got
            have = {e.name for e in got}
            for e in original[f]:   # entries the port does not carry stay registered, as stubs
                if e.name not in have:
                    entries.append(e)
        else:
            entries += parse_file(p, f, False)
    extras = {}
    for p in a.extra:   # a test's own entries: adapters only, never in the registry
        f = os.path.basename(p)
        extras[f] = parse_file(p, f, True)
    # NVN: every name main's table knows; signatures from the NVN / NVN_NOPROTO lines, else nvn_protos.h
    protos = parse_protos(os.path.join(nvndir, 'nvn_protos.h'))
    nvn_lines = {e.name: e for e in parse_file(os.path.join(nvndir, 'hle_nvn.c'), 'hle_nvn.c', False) if e.kind == 'nvn'}
    have_nvn = {e.name for e in entries if e.kind == 'nvn' and e.status == 'ported'}
    entries = [e for e in entries if not (e.kind == 'nvn' and e.status != 'ported')]
    for name in [l.strip() for l in open(os.path.join(nvndir, 'names.txt')) if l.strip()]:
        if name in have_nvn:
            continue
        if name in nvn_lines:
            entries.append(nvn_lines[name])
        elif name in protos:
            ret, params = protos[name]
            entries.append(Entry(name, 'nvn', 'nvn_protos.h', 'NVN entry point', ret, None, params, 0))
        else:
            e = Entry(name, 'nvn', 'names.txt', 'NVN entry point', None, None, None, 0)
            e.sig_known = False
            entries.append(e)
    # One entry per name: forced first (nxnative's binding order), then the first of the others
    seen, uniq, dups = {}, [], []
    for e in sorted(entries, key=lambda e: ({'forced': 0, 'swhle': 1, 'nvn': 2}[e.kind])):
        if e.name in seen:
            dups.append(f'{e.name}: {e.kind} in {e.file} shadowed by {seen[e.name].kind} in {seen[e.name].file}')
            continue
        if not re.match(r'^[A-Za-z_]\w*$', e.name):
            problems.append(f'{e.file}:{e.line}: symbol {e.name!r} is not a C identifier')
            continue
        seen[e.name] = e
        uniq.append(e)
    order = {'forced': 0, 'swhle': 1, 'nvn': 2}
    uniq.sort(key=lambda e: (order[e.kind], e.file, e.line, e.name))
    # Generate
    out = {}
    per_file = {f: [] for f in ported_files}
    stubs = []
    for e in uniq:
        try:
            if e.status == 'ported':
                per_file[e.file] += adapter_ported(e, types) + ['']
            else:
                tv = trivial_value(e.body) if e.sig_known and e.ret else None
                if tv is not None and e.kind != 'nvn':
                    e.status, e.trivial = 'trivial', tv
                    stubs += adapter_trivial(e, types) + ['']
                else:
                    stubs += adapter_stub(e, types) + ['']
        except TypeError_ as x:
            problems.append(str(x))
    for f, got in extras.items():
        per_file[f] = []
        for e in got:
            try:
                per_file[f] += adapter_ported(e, types) + ['']
            except TypeError_ as x:
                problems.append(str(x))
    for f in ported_files + list(extras):
        stem = f[:-2]
        out[f'{stem}.adapters.inc'] = HEADER + f'/* The AAPCS64 adapters of {f}\'s host replacements: included at the end of the port, next to the static functions they call. */\n' + \
            HELPERS + '\n'.join(per_file[f]) + '\n'
    out['hle_stubs.gen.c'] = HEADER + '/* Stub adapters: the entries nxrt has not ported yet. Each logs its name once and returns a neutral result,\n   or nxnative\'s constant when its body is only that ("trivial"). */\n' + \
        HELPERS + '\n'.join(stubs) + '\n'
    reg = [HEADER.rstrip(), '#include "hle_registry.h"', '#include "nxrt/rt_abi.h"', '']
    reg += [f'NXRT_API void adapter_{e.name}(NxwGuestCpu *cpu);' for e in uniq]
    reg += ['', 'const nxrt_hle_entry nxrt_hle_table[] = {']
    kinds = {'forced': 'NXRT_HLE_FORCED', 'swhle': 'NXRT_HLE_SWHLE', 'nvn': 'NXRT_HLE_NVN'}
    for e in uniq:
        st = {'ported': 1, 'trivial': 2}.get(e.status, 0)
        reg.append(f'    {{{c_str(e.name)}, {kinds[e.kind]}, {st}, {c_str(e.file)}, {c_str(e.why)}, adapter_{e.name}}},')
    reg += ['};', f'const unsigned nxrt_hle_count = {len(uniq)};', '']
    out['hle_registry.gen.c'] = '\n'.join(reg)
    hdr = [HEADER.rstrip(), '/* Every host replacement\'s adapter, as the game DLL imports them from nxrt.exe. */', '#pragma once', '#include "nxrt/rt_abi.h"',
           '#ifdef __cplusplus', 'extern "C" {', '#endif']
    hdr += [f'NXRT_API void adapter_{e.name}(NxwGuestCpu *cpu);' for e in uniq]
    hdr += ['#ifdef __cplusplus', '}', '#endif', '']
    out['hle_adapters.h'] = '\n'.join(hdr)
    inv = ['# name\tkind\tfile\tline\tstatus\tsignature']
    for e in uniq:
        sig = f'{e.ret} ({", ".join(e.params or [])})' if e.sig_known and e.ret else '(no signature)'
        inv.append(f'{e.name}\t{e.kind}\t{e.file}\t{e.line}\t{e.status}{"=" + e.trivial if e.trivial else ""}\t{" ".join(sig.split())}')
    out['hle_inventory.tsv'] = '\n'.join(inv) + '\n'
    if problems:
        for p in problems:
            print('gen_hle: ' + p, file=sys.stderr)
        return 2
    counts = {}
    for e in uniq:
        counts[(e.kind, e.status)] = counts.get((e.kind, e.status), 0) + 1
    summary = ', '.join(f'{k} {s} {n}' for (k, s), n in sorted(counts.items()))
    if a.check:
        bad = [n for n, t in out.items() if not os.path.exists(os.path.join(a.out, n)) or open(os.path.join(a.out, n), encoding='utf-8', newline='').read() != t]
        if bad:
            print('gen_hle: out of date: ' + ', '.join(bad), file=sys.stderr)
            return 1
        print(f'gen_hle: up to date ({len(uniq)} entries: {summary})')
        return 0
    os.makedirs(a.out, exist_ok=True)
    for n, t in out.items():
        with open(os.path.join(a.out, n), 'w', encoding='utf-8', newline='\n') as f:
            f.write(t)
    print(f'gen_hle: {len(uniq)} entries ({summary}); {len(dups)} shadowed duplicates; ported files: {", ".join(ported_files)}')
    for d in dups:
        print('  shadowed: ' + d)
    return 0


if __name__ == '__main__':
    sys.exit(main())
