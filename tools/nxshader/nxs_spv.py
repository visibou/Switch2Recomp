import os
"""A small SPIR-V module model (words in, words out) and the Vulkan edits nxshader makes to the translator's SPIR-V. Written from the SPIR-V 1.6 specification
(Khronos, the binary form and the instruction table); the edits mirror what nxnative's Metal chain did to its MSL where Vulkan needs an equivalent
(docs/m5-record-shaders.md, "The SPIR-V edits"). Every pass has a name and a version: both go into the cache key of what it produces (nxs_cache.py)."""
import shutil, struct, subprocess, tempfile

MAGIC = 0x07230203
# opcodes used here (SPIR-V specification, 3.52)
OpName, OpMemberName, OpExtension, OpExtInstImport, OpMemoryModel, OpEntryPoint, OpExecutionMode, OpCapability = 5, 6, 10, 11, 14, 15, 16, 17
OpTypeVoid, OpTypeBool, OpTypeInt, OpTypeFloat, OpTypeVector, OpTypeMatrix, OpTypeImage, OpTypeSampler, OpTypeSampledImage = 19, 20, 21, 22, 23, 24, 25, 26, 27
OpTypeArray, OpTypeRuntimeArray, OpTypeStruct, OpTypePointer, OpTypeFunction = 28, 29, 30, 32, 33
OpConstantTrue, OpConstantFalse, OpConstant, OpConstantComposite, OpConstantNull = 41, 42, 43, 44, 46
OpFunction, OpFunctionParameter, OpFunctionEnd, OpVariable, OpLoad, OpStore, OpAccessChain, OpInBoundsAccessChain = 54, 55, 56, 59, 61, 62, 65, 66
OpDecorate, OpMemberDecorate, OpCompositeConstruct, OpCompositeExtract, OpVectorShuffle, OpCopyObject = 71, 72, 80, 81, 79, 83
OpImageRead, OpImageWrite, OpBitcast, OpFAdd, OpFMul, OpLabel, OpReturn, OpReturnValue, OpEmitVertex, OpEmitStreamVertex = 98, 99, 124, 129, 133, 248, 253, 254, 218, 220
OpExecutionModeId, OpModuleProcessed = 331, 330
# decorations, builtins, storage classes, capabilities, execution models
DecBuiltIn, DecLocation, DecComponent, DecIndex, DecBinding, DecDescriptorSet, DecBlock, DecFlat = 11, 30, 31, 32, 33, 34, 2, 14
BuiltInPosition, BuiltInPointSize, BuiltInLayer, BuiltInBaseInstance = 0, 1, 9, 4425
SC_UniformConstant, SC_Input, SC_Uniform, SC_Output, SC_Workgroup, SC_Private, SC_Function, SC_PushConstant, SC_StorageBuffer = 0, 1, 2, 3, 4, 6, 7, 9, 12
EXEC_MODELS = {0: 'vertex', 1: 'tess_control', 2: 'tess_eval', 3: 'geometry', 4: 'fragment', 5: 'compute'}
# capability names (spirv.h of the Vulkan SDK 1.4.363, the ones these modules can carry)
CAP_NAMES = {0: 'Matrix', 1: 'Shader', 2: 'Geometry', 3: 'Tessellation', 4: 'Addresses', 5: 'Linkage', 6: 'Kernel', 7: 'Vector16', 8: 'Float16Buffer', 9: 'Float16', 10: 'Float64',
             11: 'Int64', 12: 'Int64Atomics', 13: 'ImageBasic', 14: 'ImageReadWrite', 15: 'ImageMipmap', 17: 'Pipes', 18: 'Groups', 19: 'DeviceEnqueue', 20: 'LiteralSampler', 21:
             'AtomicStorage', 22: 'Int16', 23: 'TessellationPointSize', 24: 'GeometryPointSize', 25: 'ImageGatherExtended', 27: 'StorageImageMultisample', 28:
             'UniformBufferArrayDynamicIndexing', 29: 'SampledImageArrayDynamicIndexing', 30: 'StorageBufferArrayDynamicIndexing', 31: 'StorageImageArrayDynamicIndexing', 32:
             'ClipDistance', 33: 'CullDistance', 34: 'ImageCubeArray', 35: 'SampleRateShading', 36: 'ImageRect', 37: 'SampledRect', 38: 'GenericPointer', 39: 'Int8', 40:
             'InputAttachment', 41: 'SparseResidency', 42: 'MinLod', 43: 'Sampled1D', 44: 'Image1D', 45: 'SampledCubeArray', 46: 'SampledBuffer', 47: 'ImageBuffer', 48:
             'ImageMSArray', 49: 'StorageImageExtendedFormats', 50: 'ImageQuery', 51: 'DerivativeControl', 52: 'InterpolationFunction', 53: 'TransformFeedback', 54:
             'GeometryStreams', 55: 'StorageImageReadWithoutFormat', 56: 'StorageImageWriteWithoutFormat', 57: 'MultiViewport', 58: 'SubgroupDispatch', 59: 'NamedBarrier', 60:
             'PipeStorage', 61: 'GroupNonUniform', 62: 'GroupNonUniformVote', 63: 'GroupNonUniformArithmetic', 64: 'GroupNonUniformBallot', 65: 'GroupNonUniformShuffle', 66:
             'GroupNonUniformShuffleRelative', 67: 'GroupNonUniformClustered', 68: 'GroupNonUniformQuad', 69: 'ShaderLayer', 70: 'ShaderViewportIndex', 71: 'UniformDecoration',
             4422: 'FragmentShadingRateKHR', 4423: 'SubgroupBallotKHR', 4427: 'DrawParameters', 4428: 'WorkgroupMemoryExplicitLayoutKHR', 4429:
             'WorkgroupMemoryExplicitLayout8BitAccessKHR', 4430: 'WorkgroupMemoryExplicitLayout16BitAccessKHR', 4431: 'SubgroupVoteKHR', 4433: 'StorageBuffer16BitAccess', 4434:
             'StorageUniform16', 4435: 'StoragePushConstant16', 4436: 'StorageInputOutput16', 4437: 'DeviceGroup', 4439: 'MultiView', 4441: 'VariablePointersStorageBuffer', 4442:
             'VariablePointers', 4445: 'AtomicStorageOps', 4447: 'SampleMaskPostDepthCoverage', 4448: 'StorageBuffer8BitAccess', 4449: 'UniformAndStorageBuffer8BitAccess', 4450:
             'StoragePushConstant8', 4464: 'DenormPreserve', 4465: 'DenormFlushToZero', 4466: 'SignedZeroInfNanPreserve', 4467: 'RoundingModeRTE', 4468: 'RoundingModeRTZ', 4471:
             'RayQueryProvisionalKHR', 4472: 'RayQueryKHR', 4473: 'UntypedPointersKHR', 4478: 'RayTraversalPrimitiveCullingKHR', 4479: 'RayTracingKHR', 4484:
             'TextureSampleWeightedQCOM', 4485: 'TextureBoxFilterQCOM', 4486: 'TextureBlockMatchQCOM', 4495: 'TileShadingQCOM', 4496: 'CooperativeMatrixConversionQCOM', 4498:
             'TextureBlockMatch2QCOM', 5009: 'ImageGatherBiasLodAMD', 5013: 'StencilExportEXT', 5015: 'ImageReadWriteLodAMD', 5016: 'Int64ImageEXT', 5249:
             'SampleMaskOverrideCoverageNV', 5251: 'GeometryShaderPassthroughNV', 5254: 'ShaderViewportIndexLayerEXT', 5255: 'ShaderViewportMaskNV', 5297:
             'GroupNonUniformPartitionedEXT', 5301: 'ShaderNonUniform', 5306: 'UniformBufferArrayNonUniformIndexing', 5345: 'VulkanMemoryModel', 5378:
             'FragmentShaderPixelInterlockEXT'}


def string_of(words):
    b = struct.pack(f'<{len(words)}I', *words)
    return b[:b.index(0)].decode('utf-8', 'replace') if 0 in b else b.decode('utf-8', 'replace')


def string_words(s):
    b = s.encode() + b'\0'
    b += b'\0' * (-len(b) % 4)
    return list(struct.unpack(f'<{len(b) // 4}I', b))


def string_len(words):   # words a nul-terminated literal string takes, from its start
    for i, w in enumerate(words):
        if any((w >> (8 * k)) & 0xff == 0 for k in range(4)): return i + 1
    return len(words)


class Module:
    def __init__(self, data):
        if len(data) % 4 or len(data) < 20: raise ValueError('not a SPIR-V module (size)')
        w = list(struct.unpack(f'<{len(data) // 4}I', data))
        if w[0] != MAGIC: raise ValueError('not a SPIR-V module (magic)')
        self.header = w[:5]
        self.ins = []
        i = 5
        while i < len(w):
            n = w[i] >> 16
            if n == 0 or i + n > len(w): raise ValueError(f'bad instruction length at word {i}')
            self.ins.append(w[i:i + n]); i += n
        self.reindex()

    def bytes(self):
        flat = self.header + [x for i in self.ins for x in i]
        return struct.pack(f'<{len(flat)}I', *flat)

    @property
    def version(self): return (self.header[1] >> 16 & 0xff, self.header[1] >> 8 & 0xff)

    def new_id(self):
        i = self.header[3]; self.header[3] += 1; return i

    def reindex(self):
        self.names, self.decor, self.member_decor, self.defs = {}, {}, {}, {}
        for k, x in enumerate(self.ins):
            op = x[0] & 0xffff
            if op == OpName: self.names[x[1]] = string_of(x[2:])
            elif op == OpDecorate: self.decor.setdefault(x[1], []).append((x[2], x[3:]))
            elif op == OpMemberDecorate: self.member_decor.setdefault(x[1], {}).setdefault(x[2], []).append((x[3], x[4:]))
            else:
                rid = result_id(x)
                if rid: self.defs[rid] = k

    def op(self, i): return self.ins[i][0] & 0xffff
    def inst(self, rid): return self.ins[self.defs[rid]] if rid in self.defs else None

    def deco(self, rid, d):
        for dd, args in self.decor.get(rid, ()):
            if dd == d: return args or [1]
        return None

    def caps(self): return [x[1] for x in self.ins if x[0] & 0xffff == OpCapability]
    def has_op(self, opcode): return any(x[0] & 0xffff == opcode for x in self.ins)

    def entry_points(self):
        out = []
        for k, x in enumerate(self.ins):
            if x[0] & 0xffff == OpEntryPoint:
                n = string_len(x[3:])
                out.append({'index': k, 'model': x[1], 'func': x[2], 'name': string_of(x[3:3 + n]), 'interface': x[3 + n:]})
        return out

    def type_desc(self, tid, depth=0):
        """a type as a small dict (recursive); pointers give their storage class and pointee"""
        x = self.inst(tid)
        if x is None or depth > 16: return {'t': '?'}
        op = x[0] & 0xffff
        if op == OpTypeVoid: return {'t': 'void'}
        if op == OpTypeBool: return {'t': 'bool'}
        if op == OpTypeInt: return {'t': 'int' if x[3] else 'uint', 'w': x[2]}
        if op == OpTypeFloat: return {'t': 'float', 'w': x[2]}
        if op == OpTypeVector: return {'t': 'vec', 'n': x[3], 'of': self.type_desc(x[2], depth + 1)}
        if op == OpTypeMatrix: return {'t': 'mat', 'n': x[3], 'of': self.type_desc(x[2], depth + 1)}
        if op == OpTypeArray: return {'t': 'array', 'n': self.const_value(x[3]), 'of': self.type_desc(x[2], depth + 1)}
        if op == OpTypeRuntimeArray: return {'t': 'array', 'n': None, 'of': self.type_desc(x[2], depth + 1)}
        if op == OpTypeStruct: return {'t': 'struct', 'id': tid, 'members': [self.type_desc(m, depth + 1) for m in x[2:]]}
        if op == OpTypePointer: return {'t': 'ptr', 'sc': x[2], 'of': self.type_desc(x[3], depth + 1)}
        if op == OpTypeImage: return {'t': 'image', 'sampled_type': self.type_desc(x[2], depth + 1), 'dim': x[3], 'depth': x[4], 'arrayed': x[5], 'ms': x[6], 'sampled': x[7], 'format': x[8]}
        if op == OpTypeSampler: return {'t': 'sampler'}
        if op == OpTypeSampledImage: return {'t': 'sampled_image', 'image': self.type_desc(x[2], depth + 1)}
        return {'t': 'op%d' % op}

    def const_value(self, cid):
        x = self.inst(cid)
        if x is None: return None
        op = x[0] & 0xffff
        if op == OpConstant: return x[3] if len(x) == 4 else x[3] | x[4] << 32
        return None

    def global_vars(self):
        """module-scope OpVariables: (id, storage class, pointer type id)"""
        out, in_func = [], False
        for x in self.ins:
            op = x[0] & 0xffff
            if op == OpFunction: in_func = True
            elif op == OpFunctionEnd: in_func = False
            elif op == OpVariable and not in_func: out.append((x[2], x[3], x[1]))
        return out

    # ---- building blocks for the edits ----
    def first_function_index(self):
        return next(k for k, x in enumerate(self.ins) if x[0] & 0xffff == OpFunction)

    def find_type(self, words_tail, op):
        """an existing type instruction with exactly these operands (after the result id), else None"""
        for x in self.ins:
            if x[0] & 0xffff == op and x[2:] == words_tail: return x[1]
        return None

    def ensure_type(self, op, tail):
        t = self.find_type(tail, op)
        if t: return t
        t = self.new_id(); k = self.first_function_index()
        self.ins.insert(self.types_end(), [(2 + len(tail)) << 16 | op, t] + tail); self.reindex(); return t

    def types_end(self):
        """where new types and constants go: after the last type, constant or global variable before the first function"""
        k = self.first_function_index()
        return k

    def ensure_const(self, type_id, value_words):
        for x in self.ins:
            if x[0] & 0xffff == OpConstant and x[1] == type_id and x[3:] == value_words: return x[2]
        c = self.new_id()
        self.ins.insert(self.types_end(), [(3 + len(value_words)) << 16 | OpConstant, type_id, c] + value_words); self.reindex(); return c

    def ensure_composite(self, type_id, parts):
        for x in self.ins:
            if x[0] & 0xffff == OpConstantComposite and x[1] == type_id and x[3:] == parts: return x[2]
        c = self.new_id()
        self.ins.insert(self.types_end(), [(3 + len(parts)) << 16 | OpConstantComposite, type_id, c] + parts); self.reindex(); return c

    def add_decoration(self, words):
        """an annotation (OpDecorate ...) at the end of the annotation section"""
        ann = [k for k, x in enumerate(self.ins) if x[0] & 0xffff in (71, 72, 73, 74, 332, 5632, 5633)]
        k = ann[-1] + 1 if ann else next(k for k, x in enumerate(self.ins) if x[0] & 0xffff in range(19, 47))
        self.ins.insert(k, words); self.reindex()

    def add_capability(self, cap):
        if cap in self.caps(): return False
        k = max(i for i, x in enumerate(self.ins) if x[0] & 0xffff == OpCapability) + 1
        self.ins.insert(k, [2 << 16 | OpCapability, cap]); self.reindex(); return True

    def function_body_start(self, func_id):
        """index of the first instruction of the function's first block after its OpLabel and leading OpVariables"""
        k = self.defs[func_id]
        while self.ins[k][0] & 0xffff != OpLabel: k += 1
        k += 1
        while self.ins[k][0] & 0xffff == OpVariable: k += 1
        return k

    def function_range(self, func_id):
        k = self.defs[func_id]; e = k
        while self.ins[e][0] & 0xffff != OpFunctionEnd: e += 1
        return k, e


def result_id(x):
    """the result id an instruction defines, or 0 (types and the instructions with a result type)"""
    op = x[0] & 0xffff
    if op in (OpTypeVoid, OpTypeBool, OpTypeInt, OpTypeFloat, OpTypeVector, OpTypeMatrix, OpTypeImage, OpTypeSampler, OpTypeSampledImage, OpTypeArray, OpTypeRuntimeArray,
              OpTypeStruct, 31, OpTypePointer, OpTypeFunction, 34, 35, 36, 37, 38, 39, OpLabel, OpExtInstImport, 322, 327, 5341):
        return x[1] if len(x) > 1 else 0
    if op in RESULT_TYPED and len(x) > 2: return x[2]
    return 0


# instructions with <result type> <result id> (the ones these modules use; the rest of the table only matters for ids we never look up)
RESULT_TYPED = set(range(41, 53)) | {1, 54, 55, 57, 59, 60, 61, 64, 65, 66, 67, 68, 69, 70, 73, 77, 78, 79, 80, 81, 82, 83, 84} | set(range(86, 106)) | set(range(109, 205)) | \
    set(range(207, 216)) | set(range(227, 243)) | {245, 246, 247, 251, 252} | set(range(262, 316)) | {12} | set(range(333, 367)) | {4421, 4424, 4429, 4430, 4432, 4433, 4434, 5249, 5250, 5251, 5252, 5253, 5296, 5297, 5333, 5334, 5335, 5336, 5337, 5338, 5339, 5340, 5380, 5381, 5383, 6086, 4450, 4453, 4454, 4455, 4456, 5000, 5001}


def float_bits(f): return struct.unpack('<I', struct.pack('<f', f))[0]


# ===================================================================================== the edits

class Pass:
    name, version, options = '', 1, ''
    def applies(self, m, stage): return False
    def run(self, m, stage): raise NotImplementedError
    def ident(self): return f'{self.name} v{self.version} {self.options}'.strip()


class VertexOutputInit(Pass):
    """research/01 lesson 3 (nxnative's vs_init in MSL): every user output of a vertex, tessellation-evaluation or geometry stage starts as (0,0,0,1), the value an
    unwritten attribute component reads on the hardware; the translator's outputs start undefined in Vulkan (RADV gives NaN, NVIDIA and AMD Windows 0). The stores
    go at the start of the entry function, before the translated body, which overwrites what it writes. A geometry shader is initialised once, at its start: after
    an EmitVertex the outputs are undefined again, and the translator's code writes every output it emits. Integer outputs get the bits of the same values
    (1.0f = 0x3f800000), as the hardware's untyped attribute slots hold them. A component-decorated output takes its part of the vector."""
    name, version = 'vout_init', 1

    def applies(self, m, stage): return stage in ('vs', 'gs', 'ds')

    def run(self, m, stage):
        ep = m.entry_points()[0]
        stores = []
        for vid, sc, ptr in m.global_vars():
            if sc != SC_Output or m.deco(vid, DecBuiltIn) is not None or m.deco(vid, DecLocation) is None: continue
            x = m.inst(vid)
            if len(x) > 4: continue   # has an initializer
            pointee = m.inst(ptr)[3]
            comp = (m.deco(vid, DecComponent) or [0])[0]
            c = self.value(m, pointee, comp)
            if c is not None: stores.append((vid, c))
        if not stores: return False
        k = m.function_body_start(ep['func'])
        m.ins[k:k] = [[3 << 16 | OpStore, v, c] for v, c in stores]
        m.reindex()
        return True

    def value(self, m, tid, comp):
        x = m.inst(tid); op = x[0] & 0xffff
        dflt = [0.0, 0.0, 0.0, 1.0]
        def scalar(st, i):
            s = m.inst(st); sop = s[0] & 0xffff
            v = dflt[i] if i < 4 else 0.0
            if sop == OpTypeFloat and s[2] == 32: return m.ensure_const(st, [float_bits(v)])
            if sop == OpTypeInt and s[2] == 32: return m.ensure_const(st, [float_bits(v)])
            return None
        if op in (OpTypeFloat, OpTypeInt): return scalar(tid, comp)
        if op == OpTypeVector:
            parts = [scalar(x[2], comp + i) for i in range(x[3])]
            return None if None in parts else m.ensure_composite(tid, parts)
        if op == OpTypeArray:
            e = self.value(m, x[2], comp)
            n = m.const_value(x[3])
            return None if e is None or not n or n > 64 else m.ensure_composite(tid, [e] * n)
        return None


class TrimCapabilities(Pass):
    """The translator declares capabilities a module never uses: Float64 and the other widths in modules without such a type, GeometryStreams without a stream,
    pixel interlock without an interlock. Vulkan requires the device feature for every declared capability before it creates the module (VkShaderModuleCreateInfo
    pCode-08740), and some GPUs lack them (no shaderFloat64 on Intel Arc, no geometryStreams without VK_EXT_transform_feedback, no fragment shader interlock on AMD).
    Removed when the module has nothing that needs it; a module that does use one keeps it, and its reflection says so"""
    name, version = 'caps_trim', 1
    WIDTH = {10: (OpTypeFloat, 64), 11: (OpTypeInt, 64), 9: (OpTypeFloat, 16), 22: (OpTypeInt, 16), 39: (OpTypeInt, 8)}
    KEEP_IF = {11: {12}}   # Int64Atomics implies Int64
    USES = {54: ({220, 221}, {29}, set()),                              # GeometryStreams: OpEmitStreamVertex / OpEndStreamPrimitive, the Stream decoration
            5378: ({5364, 5365}, set(), {5366, 5367, 5368, 5369, 5370, 5371})}   # FragmentShaderPixelInterlockEXT: Begin / EndInvocationInterlock, the interlock modes

    def applies(self, m, stage): return any(c in self.WIDTH or c in self.USES for c in m.caps())

    def run(self, m, stage):
        caps = set(m.caps()); drop = set()
        for c, (op, w) in self.WIDTH.items():
            if c in caps and not (self.KEEP_IF.get(c, set()) & caps) and not any(x[0] & 0xffff == op and x[2] == w for x in m.ins): drop.add(c)
        for c, (ops, decos, modes) in self.USES.items():
            if c in caps and not any(x[0] & 0xffff in ops or (x[0] & 0xffff == OpDecorate and x[2] in decos) or (x[0] & 0xffff == OpExecutionMode and x[2] in modes) for x in m.ins): drop.add(c)
        if not drop: return False
        m.ins = [x for x in m.ins if not (x[0] & 0xffff == OpCapability and x[1] in drop)]; m.reindex()
        return True


class StorageFormatCaps(Pass):
    """Reading or writing a storage image declared without a format needs StorageImageReadWithoutFormat / WriteWithoutFormat in Vulkan (Metal needed nothing):
    the translator declares Unknown for an image the answer table gives no format. The capabilities are added where the module does it"""
    name, version = 'storage_fmt_caps', 1

    def applies(self, m, stage):
        return any(x[0] & 0xffff == OpTypeImage and x[7] == 2 and x[8] == 0 for x in m.ins)

    def run(self, m, stage):
        ch = False
        if m.has_op(OpImageRead): ch |= m.add_capability(55)    # StorageImageReadWithoutFormat
        if m.has_op(OpImageWrite): ch |= m.add_capability(56)   # StorageImageWriteWithoutFormat
        return ch


class DepthGL(Pass):
    """Variant .gl: NVN's default depth convention (near = -w, the GL clip space) for a device without VK_EXT_depth_clip_control: z' = (z + w) / 2 before every
    return of the last pre-rasterisation stage (nxnative's main0 entry point in MSL, vs_depth). Pokopia sets near = 0 (nvnDeviceSetDepthMode), Vulkan's own convention,
    so its renderer needs no variant; the pass is here for the convention's other value"""
    name, version = 'depth_gl', 1

    def applies(self, m, stage): return stage in ('vs', 'gs', 'ds')

    def run(self, m, stage):
        pos = self.position(m)
        if not pos: return False
        var, chain = pos
        f32 = m.ensure_type(OpTypeFloat, [32])
        u32 = m.ensure_type(OpTypeInt, [32, 0])
        pf = m.ensure_type(OpTypePointer, [SC_Output, f32])
        half = m.ensure_const(f32, [float_bits(0.5)])
        idx = [m.ensure_const(u32, [i]) for i in chain]
        i2, i3 = m.ensure_const(u32, [2]), m.ensure_const(u32, [3])
        ep = m.entry_points()[0]
        lo, hi = m.function_range(ep['func'])
        stop = (OpEmitVertex, OpEmitStreamVertex) if stage == 'gs' else (OpReturn,)
        k, n = lo, 0
        while k < hi:
            if m.ins[k][0] & 0xffff in stop:
                pz, pw, z, w, s, h = (m.new_id() for _ in range(6))
                seq = [[(4 + len(idx) + 1) << 16 | OpAccessChain, pf, pz, var] + idx + [i2], [(4 + len(idx) + 1) << 16 | OpAccessChain, pf, pw, var] + idx + [i3],
                       [4 << 16 | OpLoad, f32, z, pz], [4 << 16 | OpLoad, f32, w, pw], [5 << 16 | OpFAdd, f32, s, z, w], [5 << 16 | OpFMul, f32, h, s, half], [3 << 16 | OpStore, pz, h]]
                m.ins[k:k] = seq; k += len(seq); hi += len(seq); n += 1
            k += 1
        m.reindex()
        return n > 0

    @staticmethod
    def position(m):
        """(variable, member indices of the vec4) of the Position output: a gl_PerVertex block member or a variable of its own"""
        for vid, sc, ptr in m.global_vars():
            if sc != SC_Output: continue
            if (m.deco(vid, DecBuiltIn) or [None])[0] == BuiltInPosition: return vid, []
            st = m.inst(m.inst(ptr)[3])
            if st[0] & 0xffff == OpTypeStruct:
                for mi, ds in m.member_decor.get(st[1], {}).items():
                    if any(d == DecBuiltIn and a and a[0] == BuiltInPosition for d, a in ds): return vid, [mi]
        return None


class PadOutputs(Pass):
    """Variant .pad<mask>: outputs at the given locations that the paired fragment shader reads but this vertex shader does not write, (0,0,0,1) (nxnative's vs_pad).
    Vulkan links such a pair (the input reads an undefined value); the renderer asks for the variant when a pipeline's pair needs it (the reflection names both sides' locations)"""
    name, version = 'pad_outputs', 1

    def __init__(self, locations): self.locations = sorted(set(locations)); self.options = 'locations=' + ','.join(map(str, self.locations))

    def applies(self, m, stage): return stage in ('vs', 'gs', 'ds') and bool(self.locations)

    def run(self, m, stage):
        have = {(m.deco(v, DecLocation) or [None])[0] for v, sc, _ in m.global_vars() if sc == SC_Output}
        miss = [l for l in self.locations if l not in have]
        if not miss: return False
        f32 = m.ensure_type(OpTypeFloat, [32]); v4 = m.ensure_type(OpTypeVector, [f32, 4]); pv = m.ensure_type(OpTypePointer, [SC_Output, v4])
        c = VertexOutputInit().value(m, v4, 0)
        ep = m.entry_points()[0]
        new = []
        for l in miss:
            v = m.new_id(); new.append(v)
            k = m.first_function_index(); m.ins.insert(k, [4 << 16 | OpVariable, pv, v, SC_Output])
            m.add_decoration([4 << 16 | OpDecorate, v, DecLocation, l])
        m.reindex()
        epi = m.entry_points()[0]['index']; x = m.ins[epi]; m.ins[epi] = [(len(x) + len(new)) << 16 | OpEntryPoint] + x[1:] + new
        k = m.function_body_start(ep['func'])
        m.ins[k:k] = [[3 << 16 | OpStore, v, c] for v in new]
        m.reindex()
        return True


class DualSource(Pass):
    """Variant .dual: NVN's SRC1 blend factors read the pixel shader's output 1 as target 0's second source (Maxwell feeds MRT 1 to it); Vulkan reads it from
    Location 0 Index 1 (nxnative's fs_dual for Metal's [[color(0), index(1)]]). For a shader that writes exactly outputs 0 and 1"""
    name, version = 'dual_source', 1

    def applies(self, m, stage): return stage == 'ps'

    def run(self, m, stage):
        outs = {(m.deco(v, DecLocation) or [None])[0]: v for v, sc, _ in m.global_vars() if sc == SC_Output and m.deco(v, DecBuiltIn) is None}
        if sorted(k for k in outs if k is not None) != [0, 1]: return False
        v1 = outs[1]
        for x in m.ins:
            if x[0] & 0xffff == OpDecorate and x[1] == v1 and x[2] == DecLocation: x[3] = 0
        k = next(i for i, x in enumerate(m.ins) if x[0] & 0xffff == OpDecorate and x[1] == v1 and x[2] == DecLocation)
        m.ins.insert(k + 1, [4 << 16 | OpDecorate, v1, DecIndex, 1]); m.reindex()
        return True


class Fetch1DTexelBuffer(Pass):
    """The translator reads a 1D integer fetch (texelFetch of a buffer texture) as a 2D fetch at (x & 16383, x >> 14) of a combined image sampler, for Metal,
    whose host binds a texture view of the buffer with rows of 16384 texels there (nvn_mtl.m tbuf_as_2d). Vulkan cannot view a buffer as an image: such a
    binding becomes a uniform texel buffer and each fetch reads element x + (y << 14) of it, the same element. The bindings are the translator's fetch1d slots
    (its res.json, given to the module as m.fetch1d_vars before the passes run); a variable used any other way than load → OpImage → OpImageFetch is left as it is"""
    name, version = 'fetch1d_tbuf', 1
    OpImageFetch, OpImage, OpShiftLeftLogical, OpBitwiseOr = 95, 100, 196, 197

    def applies(self, m, stage): return bool(getattr(m, 'fetch1d_vars', None))

    def uses(self, m, rid):   # instructions that take rid as an operand (debug, name, decoration and constant literals are not operands)
        return [k for k, x in enumerate(m.ins) if x[0] & 0xffff not in (3, 5, 6, 7, 8, 43, 71, 72) and rid in x[1:] and result_id(x) != rid]

    def run(self, m, stage):
        changed = False
        for v in sorted(m.fetch1d_vars):
            var = m.inst(v)
            if not var or var[0] & 0xffff != OpVariable: continue
            si = m.inst(m.inst(var[1])[3])
            if si[0] & 0xffff != OpTypeSampledImage: continue
            img = m.inst(si[2])
            if img[3] != 1 or img[5] or img[6]: continue   # 2D, not arrayed, not multisampled
            loads = [k for k, x in enumerate(m.ins) if x[0] & 0xffff == OpLoad and x[3] == v]
            ok, plan = True, []
            for k in loads:
                for u in self.uses(m, m.ins[k][2]):
                    x = m.ins[u]
                    if x[0] & 0xffff != self.OpImage or x[3] != m.ins[k][2]: ok = False; break
                    for f in self.uses(m, x[2]):
                        y = m.ins[f]
                        if y[0] & 0xffff != self.OpImageFetch or y[3] != x[2] or (len(y) > 5 and y[5] != 2) or len(y) > 7: ok = False; break
                        plan.append(f)
                    plan.append(u)
                if not ok: break
            if not ok or not loads: continue
            fetches = [m.ins[f] for f in plan if m.ins[f][0] & 0xffff == self.OpImageFetch]   # taken now: the insertions below move indices
            tb = m.ensure_type(OpTypeImage, [img[2], 5, 0, 0, 0, 1, img[8]])   # Dim Buffer, sampled
            ptb = m.ensure_type(OpTypePointer, [SC_UniformConstant, tb])
            # the coordinate's component type (int or uint) for the index arithmetic
            ctype = None
            for y in fetches:
                c = m.inst(y[4]); vt = m.inst(c[1]) if c else None
                if vt and vt[0] & 0xffff == OpTypeVector and vt[3] == 2: ctype = vt[2]; break
            if fetches and ctype is None: continue
            c14 = m.ensure_const(ctype, [14]) if ctype else None
            for t in (tb, ptb):   # new types go after the module's globals (ensure_type): the variable needs them before it
                if m.defs[t] > m.defs[v]: x = m.ins.pop(m.defs[t]); m.ins.insert(m.defs[v], x); m.reindex()
            var = m.inst(v); var[1] = ptb
            for k in [k for k, x in enumerate(m.ins) if x[0] & 0xffff == OpLoad and x[3] == v]: m.ins[k][1] = tb
            for k, x in enumerate(m.ins):
                if x[0] & 0xffff == self.OpImage and m.inst(x[3]) is not None and m.inst(x[3])[0] & 0xffff == OpLoad and m.inst(x[3])[3] == v:
                    m.ins[k] = [4 << 16 | OpCopyObject, tb, x[2], x[3]]
            out = []
            fetch_ids = {y[2] for y in fetches}
            for x in m.ins:
                if x[0] & 0xffff == self.OpImageFetch and x[2] in fetch_ids:
                    e0, e1, sh, idx = m.new_id(), m.new_id(), m.new_id(), m.new_id()
                    out += [[5 << 16 | OpCompositeExtract, ctype, e0, x[4], 0], [5 << 16 | OpCompositeExtract, ctype, e1, x[4], 1],
                            [5 << 16 | self.OpShiftLeftLogical, ctype, sh, e1, c14], [5 << 16 | self.OpBitwiseOr, ctype, idx, e0, sh],
                            [5 << 16 | self.OpImageFetch, x[1], x[2], x[3], idx]]
                else: out.append(x)
            m.ins = out; m.reindex(); changed = True
        return changed


class LoopGuard(Pass):
    """A loop's count can come from data: MKW's forward shaders walk a light list whose length they read from a storage buffer at an index worked out per pixel (fp_s1, GridifyGeo).
    One bad word there is a loop of up to 2^32 turns in a pixel shader, which on a PC is a device lost. So every loop leaves after a bound no real list reaches: a counter, kept in
    the function, checked in the loop's header"""
    name, version = 'loop_guard', 1
    LIMIT = {'ps': 4096, 'fs': 4096, 'vs': 65536, 'gs': 65536, 'hs': 65536, 'ds': 65536, 'cs': 1 << 20}
    OpLoopMerge, OpBranch, OpBranchConditional, OpIAdd, OpUGreaterThanEqual, OpLoad, OpStore = 246, 249, 250, 128, 174, 61, 62

    def applies(self, m, stage): return any(x[0] & 0xffff == self.OpLoopMerge for x in m.ins)

    @property
    def options(self):   # NXSHADER_LOOP_LIMIT=N caps every pixel shader loop at N turns instead (a test; it enters the cache key)
        e = os.environ.get('NXSHADER_LOOP_LIMIT'); return f'ps{e}' if e else ''

    def run(self, m, stage):
        lim = self.LIMIT.get(stage, 65536)
        if os.environ.get('NXSHADER_LOOP_LIMIT') and stage in ('ps', 'fs'): lim = int(os.environ['NXSHADER_LOOP_LIMIT'])
        uint = m.ensure_type(OpTypeInt, [32, 0]); boolt = m.ensure_type(OpTypeBool, []); ptr = m.ensure_type(OpTypePointer, [7, uint])
        c0 = m.ensure_const(uint, [0]); c1 = m.ensure_const(uint, [1]); cl = m.ensure_const(uint, [lim])
        ch = False; k = len(m.ins) - 1
        while k >= 0:
            x = m.ins[k]
            if x[0] & 0xffff != self.OpLoopMerge or m.ins[k + 1][0] & 0xffff != self.OpBranch: k -= 1; continue
            merge, body = x[1], m.ins[k + 1][1]
            var, a, b, ov = m.new_id(), m.new_id(), m.new_id(), m.new_id()
            m.ins[k + 1] = [4 << 16 | self.OpBranchConditional, ov, merge, body]
            m.ins[k:k] = [[4 << 16 | self.OpLoad, uint, a, var], [5 << 16 | self.OpIAdd, uint, b, a, c1], [3 << 16 | self.OpStore, var, b], [5 << 16 | self.OpUGreaterThanEqual, boolt, ov, b, cl]]
            f = k
            while m.ins[f][0] & 0xffff != OpFunction: f -= 1
            while m.ins[f][0] & 0xffff != OpLabel: f += 1
            m.ins.insert(f + 1, [5 << 16 | OpVariable, ptr, var, 7, c0])
            ch = True; k = f - 1 if False else k - 1
        m.reindex()
        return ch


class VsBaseInstance(Pass):
    """The draw's base instance from Vulkan, not from the driver segment (nxnative: dmsh_build.py vs_base_instance). GLSLC adds c0 + 0x644 to Maxwell's instance ID where a shader needs the
    absolute instance; the word comes from the draw command, but an indirect draw's base instance is in its record, which compute shaders earlier in the same submission write. Read on the CPU
    it is the previous frame's or older (a flight recording: 14,151 of 206,902 indirect draws had another value than the GPU's), so MKW's GPU-driven objects took other objects' records and
    blinked. The segment's rows hold raw bits as floats, so the load becomes the BaseInstance builtin's bits as a float"""
    name, version = 'vs_base_instance', 1

    def applies(self, m, stage): return stage == 'vs' and 'vp_c0' in m.names.values()

    def run(self, m, stage):
        c0 = next(k for k, v in m.names.items() if v == 'vp_c0')
        cv = {x[2]: x[3] for x in m.ins if x[0] & 0xffff == OpConstant and len(x) > 3}
        chains = {x[2] for x in m.ins if x[0] & 0xffff in (OpAccessChain, OpInBoundsAccessChain) and x[3] == c0 and [cv.get(v) for v in x[4:]] == [0, 100, 1]}
        loads = [x for x in m.ins if x[0] & 0xffff == OpLoad and x[3] in chains]
        if not loads: return False
        var = next((v for v, sc, ptr in m.global_vars() if m.deco(v, DecBuiltIn) == [BuiltInBaseInstance]), None)
        if var is None:
            uint = m.ensure_type(OpTypeInt, [32, 0]); ptr = m.ensure_type(OpTypePointer, [SC_Input, uint])
            var = m.new_id(); m.add_capability(4427)
            m.ins.insert(m.types_end(), [4 << 16 | OpVariable, ptr, var, SC_Input])
            m.add_decoration([4 << 16 | OpDecorate, var, DecBuiltIn, BuiltInBaseInstance])
            m.reindex()
            epi = m.entry_points()[0]['index']; x = m.ins[epi]; m.ins[epi] = [(len(x) + 1) << 16 | OpEntryPoint] + x[1:] + [var]
            m.reindex()
        vt = m.inst(m.inst(var)[1])[3]
        for x in [i for i in m.ins if i[0] & 0xffff == OpLoad and i[3] in chains]:
            t = m.new_id(); k = m.ins.index(x)
            m.ins[k:k + 1] = [[4 << 16 | OpLoad, vt, t, var], [4 << 16 | OpBitcast, x[1], x[2], t]]
        m.reindex()
        return True


class NanMinMax(Pass):
    """min, max and clamp with a NaN operand (nxnative: docs/reference/shaders.md, under that title). Maxwell's FMNMX returns the other operand, and its saturation gives 0. GLSL.std.450's FMin, FMax
    and FClamp leave a NaN operand's result undefined, and a NaN that survives lands in an R11G11B10F target as black. MKW's forward shaders end in max(0, ...), and its wind-animated meshes
    (trees, palms, cacti) came out black on some frames and not on others. NMin, NMax and NClamp are the hardware's rule"""
    name, version = 'nan_min_max', 1
    MAP = {37: 79, 40: 80, 43: 81}   # FMin, FMax, FClamp -> NMin, NMax, NClamp

    def applies(self, m, stage): return any(x[0] & 0xffff == 12 and x[4] in self.MAP for x in m.ins)

    def run(self, m, stage):
        glsl = {x[1] for x in m.ins if x[0] & 0xffff == OpExtInstImport and string_of(x[2:]) == 'GLSL.std.450'}
        ch = False
        for x in m.ins:
            if x[0] & 0xffff == 12 and x[3] in glsl and x[4] in self.MAP: x[4] = self.MAP[x[4]]; ch = True
        return ch


class AbsGuard(Pass):
    """Integer abs on NVIDIA: the GPU freeze (docs/HANDOFF_NATIVE_GPU.md, 2026-11-08). NVIDIA's shader compiler (driver 617.42) simplifies abs() of a value it can prove small,
    such as abs(c ? -(c2 ? 0 : 1) : 1), into machine code that raises a "Shader Instruction Error" when that path runs: the GPU stops, the game freezes and closes. Maxwell code
    computes such shapes (integer division, wrapping), and nxnative's translation keeps them. Reproduced outside the game with tools/drawrepro (MKW's 87c52decaf1a8e68 and
    5c9e6c5662d6daa5 hang with every buffer word 1.0 and finish with 0; spirv-reduce left abs of a select of a negation); hiding the operand behind a value the compiler cannot
    know cures it, and no rewriting of abs itself does (the compiler sees through select, max and shift forms). Every scalar integer SAbs takes x ^ z, z a word the runtime keeps
    at zero: a member appended to the support buffer at ZERO_OFFSET, inside what the runtime binds and zero-fills (vk_replay.c SUPPORT_SIZE). Same result for every input"""
    name, version = 'abs_guard', 1
    ZERO_OFFSET = 1392
    SAbs, OpBitwiseXor, DecOffset = 5, 198, 35

    def _glsl(self, m): return {x[1] for x in m.ins if x[0] & 0xffff == OpExtInstImport and string_of(x[2:]) == 'GLSL.std.450'}

    def _support(self, m):
        sb = next((k for k, v in m.names.items() if v == 'support_buffer'), None)
        if sb is None: sb = next((v for v, sc, ptr in m.global_vars() if sc == SC_Uniform and m.deco(v, DecBinding) == [0]), None)
        return sb

    def applies(self, m, stage):
        g = self._glsl(m)
        return any(x[0] & 0xffff == 12 and x[3] in g and x[4] == self.SAbs for x in m.ins) and self._support(m) is not None

    def run(self, m, stage):
        g, sb = self._glsl(m), self._support(m)
        struct = m.inst(m.inst(sb)[1])[3]
        offs = [a[0] for mi, ds in m.member_decor.get(struct, {}).items() for d, a in ds if d == self.DecOffset]
        if not offs or max(offs) >= self.ZERO_OFFSET: return False
        int_t = m.ensure_type(OpTypeInt, [32, 1]); uint_t = m.find_type([32, 0], OpTypeInt)
        k = m.defs[struct]; x = m.ins[k]; idx = len(x) - 2
        m.ins[k] = [(len(x) + 1) << 16 | OpTypeStruct] + x[1:] + [int_t]; m.reindex()
        m.add_decoration([5 << 16 | OpMemberDecorate, struct, idx, self.DecOffset, self.ZERO_OFFSET])
        ptr_i = m.ensure_type(OpTypePointer, [SC_Uniform, int_t]); cidx = m.ensure_const(int_t, [idx])
        funcs = [x[2] for x in m.ins if x[0] & 0xffff == OpFunction]
        n = 0
        for f in funcs:
            a, e = m.function_range(f)
            sabs = [i for i in range(a, e) if m.ins[i][0] & 0xffff == 12 and m.ins[i][3] in g and m.ins[i][4] == self.SAbs and len(m.ins[i]) == 6
                    and m.ins[i][1] in (int_t, uint_t)]
            if not sabs: continue
            p, z, zu = m.new_id(), m.new_id(), m.new_id()
            new, start = [], m.function_body_start(f)
            for i in range(a, e + 1):
                if i == start:
                    new += [[5 << 16 | OpAccessChain, ptr_i, p, sb, cidx], [4 << 16 | OpLoad, int_t, z, p]]
                    if uint_t: new.append([4 << 16 | OpBitcast, uint_t, zu, z])
                x = m.ins[i]
                if i in sabs:
                    gid = m.new_id(); t = x[1]
                    new.append([5 << 16 | self.OpBitwiseXor, t, gid, x[5], z if t == int_t else zu]); x = x[:5] + [gid]; n += 1
                new.append(x)
            m.ins[a:e + 1] = new; m.reindex()
        return n > 0


class FirstStore(Pass):
    """A store only a dispatch's very first invocation makes (nxnative's first_store, dmsh_build.py; docs/reference/shaders.md "A store only a dispatch's first invocation makes"). A compute
    shader's preamble resets a list on the first invocation while the others add to it later. Nothing orders them in the API; it works on the console because the first workgroup starts ahead
    of the rest. Elsewhere a workgroup that appended before the reset loses its entries, a different few each frame: MKW's scenery lists (2ce9071c2f22d74f, 5fea9b7ddb30e8a8,
    0823b08c2027a38d) flickered, trees and bushes in blocks. So the stores leave the shader and the replay makes them before the dispatch. nxnative matches the shape in the MSL; here the
    preamble is run on small inputs: the stores it makes for workgroup 0 invocation 0 and for no other input it is given are those stores, whatever the expression looks like"""
    name, version = 'first_store', 1
    OpLogicalNot, OpLogicalAnd, OpLogicalOr, OpIEqual, OpINotEqual, OpNot, OpIAdd, OpShl, OpBitcast, OpIMul = 168, 167, 166, 170, 171, 200, 128, 196, 124, 132
    OpSelectionMerge, OpBranch, OpBranchConditional, OpControlBarrier = 247, 249, 250, 224

    def applies(self, m, stage):
        return stage == 'cs' and any(x[0] & 0xffff == OpDecorate and x[2] == 11 and x[3] == 26 for x in m.ins) and any(x[0] & 0xffff == OpDecorate and x[2] == 11 and x[3] == 27 for x in m.ins)

    def run(self, m, stage):
        self.stores = []
        ep = m.entry_points()[0]; size = next((x[3:6] for x in m.ins if x[0] & 0xffff == OpExecutionMode and x[2] == 17), None)
        if not size: return False
        wg = next(x[1] for x in m.ins if x[0] & 0xffff == OpDecorate and x[2] == 11 and x[3] == 26)
        lid = next(x[1] for x in m.ins if x[0] & 0xffff == OpDecorate and x[2] == 11 and x[3] == 27)
        first = self.walk(m, ep['func'], wg, lid, (0, 0, 0), (0, 0, 0))
        if first is None or not first[0] or len(first[0]) > 8: return False
        # with a flat local size the guard looks at x alone, and the dispatch is flat too: y and z are only tried where the local size has rows
        tries = [((1, 0, 0), (0, 0, 0)), ((3, 0, 0), (size[0] - 1, 0, 0)), ((0, 0, 0), (1, 0, 0))]
        if size[1] > 1: tries += [((0, 1, 0), (0, 0, 0)), ((0, 0, 0), (0, 1, 0)), ((2, 1, 0), (1, 1, 0)), ((0, 3, 0), (0, size[1] - 1, 0)), ((0, 0, 1), (0, 0, 0))]
        for w, l in tries:
            r = self.walk(m, ep['func'], wg, lid, w, tuple(min(l[i], size[i] - 1) for i in range(3)))
            if r is None or r[0]: return False
        boolt = m.ensure_type(OpTypeBool, [])
        no = next((x[2] for x in m.ins if x[0] & 0xffff == OpConstantFalse and x[1] == boolt), None)
        if no is None:
            no = m.new_id(); m.ins.insert(m.first_function_index(), [3 << 16 | OpConstantFalse, boolt, no]); m.reindex()
        first = self.walk(m, ep['func'], wg, lid, (0, 0, 0), (0, 0, 0))   # the indices after the constant went in
        for at in first[1]: at[1] = no
        self.stores = first[0]
        m.first_stores = [{'var': v, 'word': w, 'value': val} for v, w, val in first[0]]
        return True

    def walk(self, m, fn, wg, lid, w, l):
        """The function's start for one input, up to the first thing it cannot follow. ([(variable, word, value)], [the branches that lead to those stores]), or None"""
        k = m.defs[fn]; ids, mem, chain = {}, {}, {}; out, br = [], []
        labels = {x[1]: i for i, x in enumerate(m.ins) if x[0] & 0xffff == OpLabel}
        const = {x[2]: x[3] for x in m.ins if x[0] & 0xffff == OpConstant}
        mask = lambda v: v & 0xffffffff
        val = lambda i: ids[i] if i in ids else const.get(i)
        last_cond = None
        for _ in range(4000):
            x = m.ins[k]; op = x[0] & 0xffff; k += 1
            if op in (OpFunction, OpLabel, OpVariable, self.OpSelectionMerge): continue
            if op in (OpAccessChain, OpInBoundsAccessChain):
                if x[3] in (wg, lid) and len(x) == 5 and x[4] in const: ids[x[2]] = ('in', (w if x[3] == wg else l)[const[x[4]]])
                elif len(x) == 6 and x[4] in const and x[5] in const and const[x[4]] == 0: chain[x[2]] = (x[3], const[x[5]])
                else: return (out, br)
            elif op == OpLoad:
                p = x[3]
                if isinstance(ids.get(p), tuple): ids[x[2]] = ids[p][1]
                elif p in mem: ids[x[2]] = mem[p]
                else: return (out, br)
            elif op == OpStore:
                v = val(x[2])
                if v is None: return (out, br)
                if x[1] in chain:
                    out.append((chain[x[1]][0], chain[x[1]][1], mask(int(v))))
                    if last_cond is not None and last_cond not in br: br.append(last_cond)
                else: mem[x[1]] = v
            elif op == self.OpBitcast:
                v = val(x[3])
                if v is None: return (out, br)
                ids[x[2]] = v
            elif op == self.OpNot: ids[x[2]] = mask(~val(x[3]))
            elif op == self.OpLogicalNot: ids[x[2]] = not val(x[3])
            elif op in (self.OpLogicalAnd, self.OpLogicalOr, self.OpIEqual, self.OpINotEqual, self.OpIAdd, self.OpShl, self.OpIMul):
                a, b = val(x[3]), val(x[4])
                if a is None or b is None: return (out, br)
                ids[x[2]] = {self.OpLogicalAnd: lambda: bool(a) and bool(b), self.OpLogicalOr: lambda: bool(a) or bool(b), self.OpIEqual: lambda: mask(a) == mask(b), self.OpINotEqual: lambda: mask(a) != mask(b),
                             self.OpIAdd: lambda: mask(a + b), self.OpShl: lambda: mask(a << b), self.OpIMul: lambda: mask(a * b)}[op]()
            elif op == self.OpBranch: k = labels[x[1]]; last_cond = None if x[1] not in labels else last_cond
            elif op == self.OpBranchConditional:
                c = val(x[1])
                if c is None: return (out, br)
                last_cond = x if c else None
                k = labels[x[2] if c else x[3]]
            else: return (out, br)
        return (out, br)


class MeshStage(Pass):
    """NVN2 task and mesh shaders as Vulkan's (VK_EXT_mesh_shader). The translator reads one as a compute shader with its stage memory in shared memory (dmsh_meshstage.py has the layout;
    the Metal wrapper there, to_stage, is what this does in SPIR-V). A new entry point starts the stage memory, calls the kernel's body as a function, so its early returns come back here,
    and the workgroup's first invocation hands the outputs over. The shared memory starts at zero by its initializer. The draw's first task or mesh group is 0 (Vulkan has no parameter for
    it): hle_nvn.c says so in red if a game asks for another"""
    name, version = 'mesh_stage', 6
    OpExtension_, OpConstantNull_, OpReturn_, OpLabel_ = 10, 46, 253, 248
    OpFunctionCall, OpControlBarrier, OpBranch, OpBranchConditional, OpLoopMerge, OpSelectionMerge = 57, 224, 249, 250, 246, 247
    OpIAdd, OpIMul, OpUDiv, OpUMod, OpIEqual, OpULessThan, OpShiftRightLogical, OpBitwiseAnd, OpBitcast = 128, 132, 134, 137, 170, 176, 194, 199, 124
    OpSetMeshOutputsEXT, OpEmitMeshTasksEXT, OpUMin_ = 5295, 5294, 38
    BLOCK = 32

    def __init__(self, info):
        self.i = info; self.options = hex(hash_info(info))

    def applies(self, m, stage): return True

    def run(self, m, stage):
        import dmsh_meshstage as ms
        i = self.i; kind = i['kind']; self.m = m; self.code = []
        lay, _ = ms.memory(i)
        wg, inp, out, prim = lay[0] >> 2, lay[1] >> 2, lay[2] >> 2, lay[3]
        ep = m.entry_points()[0]; old = ep['func']
        wv = [x for x in m.ins if x[0] & 0xffff == OpVariable and x[3] == 4]
        if len(wv) != 1: raise ValueError('%s: %d workgroup variables' % (kind, len(wv)))
        mem = wv[0][2]
        uint, boolt, voidt, flt = m.ensure_type(OpTypeInt, [32, 0]), m.ensure_type(OpTypeBool, []), m.ensure_type(OpTypeVoid, []), m.ensure_type(OpTypeFloat, [32])
        vec4 = m.ensure_type(OpTypeVector, [flt, 4]); uvec3 = m.ensure_type(OpTypeVector, [uint, 3]); uvec3p = m.ensure_type(OpTypePointer, [1, uvec3]); uintp_in = m.ensure_type(OpTypePointer, [1, uint])
        self.uint, self.boolt, self.voidt = uint, boolt, voidt
        memp = m.ensure_type(OpTypePointer, [4, uint])
        fnt = m.ensure_type(OpTypeFunction, [voidt])
        c = lambda v: m.ensure_const(uint, [v])
        self.c = c
        # the stage memory starts at zero
        null = m.new_id(); wtype = m.inst(wv[0][1])[3]
        m.ins.insert(m.defs[mem], [3 << 16 | self.OpConstantNull_, wtype, null]); m.reindex()
        x = m.inst(mem); x[0] = 5 << 16 | OpVariable; x.append(null) if len(x) == 4 else None
        new_vars = []
        def var(ptr, sc):
            v = m.new_id(); m.ins.insert(m.first_function_index(), [4 << 16 | OpVariable, ptr, v, sc]); m.reindex(); new_vars.append(v); return v
        # The body numbers its threads by their place in the subgroup: on the console a workgroup is one warp, on a PC's mesh stage nothing says so, and two threads with one number leave
        # vertices unwritten, which reach the screen as huge triangles. The workgroup's own index is what it means
        sub = next((x for x in m.ins if x[0] & 0xffff == OpDecorate and x[2] == 11 and x[3] == 41), None)
        if sub: sub[3] = 29; lid = sub[1]; m.reindex()
        else: lid = var(uintp_in, 1); m.add_decoration([4 << 16 | OpDecorate, lid, 11, 29])   # LocalInvocationIndex
        wid = var(uvec3p, 1); m.add_decoration([4 << 16 | OpDecorate, wid, 11, 26])     # WorkgroupId
        pay_words = max(i['payload'] + 3 >> 2, 1)
        payt = m.ensure_type(OpTypeArray, [uint, c(pay_words)])
        payp = m.ensure_type(OpTypePointer, [5402, payt]); payi = m.ensure_type(OpTypePointer, [5402, uint])
        payload = var(payp, 5402) if kind == 'ts' or i.get('task') else 0
        sets = []
        if kind == 'ms':
            comps, pcomps = i['vertex'], i['primitive']
            max_v, max_p = i['max_v'], i['max_p']
            stride = max(len(comps) + len(pcomps), 1) * self.BLOCK
            vat = m.ensure_type(OpTypeArray, [vec4, c(max_v)]); pat = m.ensure_type(OpTypeArray, [vec4, c(max_p)])
            ovp = m.ensure_type(OpTypePointer, [3, vat]); opp = m.ensure_type(OpTypePointer, [3, pat]); oe = m.ensure_type(OpTypePointer, [3, vec4])
            idt = m.ensure_type(OpTypeVector, [uint, 3]); iat = m.ensure_type(OpTypeArray, [idt, c(max_p)]); iap = m.ensure_type(OpTypePointer, [3, iat]); ie = m.ensure_type(OpTypePointer, [3, idt])
            members = {}
            for k, (a, comp) in enumerate(comps): members.setdefault(a, [None] * 4)[comp] = k
            pm = {}
            for k, (a, comp) in enumerate(pcomps): pm.setdefault(a, [None] * 4)[comp] = k + len(comps)
            if None not in members: raise ValueError('ms: no position')
            for a, w in members.items():
                v = var(ovp, 3)
                m.add_decoration([4 << 16 | OpDecorate, v, 11, 0] if a is None else [4 << 16 | OpDecorate, v, 30, a])
                sets.append((v, w))
            psets = []   # per-primitive outputs reach the pixel shader as per-vertex ones, written at the primitive's three vertices: its input is not declared per primitive, and a variant of
                         # each pixel shader for this would be a second key. Right where a vertex's primitives agree, as in a tuft's cluster
            for a, w in pm.items():
                if a in members: continue
                v = var(ovp, 3); m.add_decoration([4 << 16 | OpDecorate, v, 30, a]); psets.append((v, w))
            tri = var(iap, 3); m.add_decoration([4 << 16 | OpDecorate, tri, 11, 5296])
        self.spirv_14(m)
        k1 = m.add_capability(5283)
        at = max(k for k, x in enumerate(m.ins) if x[0] & 0xffff == OpCapability) + 1
        name = encode_string('SPV_EXT_mesh_shader')
        m.ins.insert(at, [(1 + len(name)) << 16 | self.OpExtension_] + name); m.reindex()
        # the entry point and its modes
        entry = m.new_id(); ep = m.entry_points()[0]
        word_name = m.ins[ep['index']][3:3 + string_len(m.ins[ep['index']][3:])]
        iface = list(dict.fromkeys(x[2] for x in m.ins if x[0] & 0xffff == OpVariable and x[3] != 7 and m.defs[x[2]] < m.first_function_index()))
        m.ins[ep['index']] = [(3 + len(word_name) + len(iface)) << 16 | OpEntryPoint, 5364 if kind == 'ts' else 5365, entry] + word_name + iface
        for x in m.ins:
            if x[0] & 0xffff == OpExecutionMode and x[1] == old: x[1] = entry
        k = max(q for q, x in enumerate(m.ins) if x[0] & 0xffff == OpExecutionMode) + 1
        if kind == 'ms':
            m.ins[k:k] = [[3 << 16 | 16 | 4 << 16 & 0, 0]][:0] + [[4 << 16 | OpExecutionMode, entry, 26, max_v], [4 << 16 | OpExecutionMode, entry, 5270, max_p], [3 << 16 | OpExecutionMode, entry, 5298]]
        m.reindex()
        # the body
        e = self.emit; lbl = self.label
        first = m.new_id()
        self.code += [[5 << 16 | OpFunction, voidt, entry, 0, fnt], [2 << 16 | self.OpLabel_, first]]
        ivar = m.new_id(); uintf = m.ensure_type(OpTypePointer, [7, uint])
        self.code.append([4 << 16 | OpVariable, uintf, ivar, 7])
        self.ivar = ivar
        lv = e(61, uint, lid); wv3 = e(61, uvec3, wid); tg = e(81, uint, wv3, 0)
        is0 = e(self.OpIEqual, boolt, lv, c(0))
        memw = lambda idx, val: self.store(e(65, memp, mem, idx), val)
        memr = lambda idx: e(61, uint, e(65, memp, mem, idx))
        def first_only(fn):
            a, b = m.new_id(), m.new_id()
            self.code += [[3 << 16 | self.OpSelectionMerge, b, 0], [4 << 16 | self.OpBranchConditional, is0, a, b], [2 << 16 | self.OpLabel_, a]]
            fn(); self.code += [[2 << 16 | self.OpBranch, b], [2 << 16 | self.OpLabel_, b]]
        barrier = lambda: self.code.append([4 << 16 | self.OpControlBarrier, c(2), c(2), c(0x108)])
        if kind == 'ms' and i.get('task'):
            def fill():
                self.loop(c(pay_words), lambda n: memw(e(self.OpIAdd, uint, c(inp), n), e(61, uint, e(65, payi, payload, n))))
                memw(c(wg), tg); memw(c(inp), tg)
            first_only(fill)
        else:
            first_only(lambda: memw(c(wg), tg))
        barrier()
        self.code.append([4 << 16 | self.OpFunctionCall, voidt, m.new_id(), old])
        barrier()
        a, b = m.new_id(), m.new_id()
        self.code += [[3 << 16 | self.OpSelectionMerge, b, 0], [4 << 16 | self.OpBranchConditional, is0, a, b], [2 << 16 | self.OpLabel_, a]]
        if kind == 'ts':
            self.loop(c(pay_words), lambda n: self.store(e(65, payi, payload, n), memr(e(self.OpIAdd, uint, c(out), n))))
            count = e(12, uint, m_ext(m), 38, memr(c(out + 1)), c(1024))   # UMin
            self.code.append([5 << 16 | self.OpEmitMeshTasksEXT, count, c(1), c(1), payload])
        else:
            pc = e(self.OpShiftRightLogical, uint, memr(c(prim >> 2)), c(24))
            np = e(12, uint, m_ext(m), 38, pc, c(max_p))
            self.code.append([3 << 16 | self.OpSetMeshOutputsEXT, c(max_v), np])
            f0, f1 = m.ensure_const(flt, [float_bits(0.0)]), m.ensure_const(flt, [float_bits(1.0)])
            def base(n):
                blk = e(self.OpIMul, uint, e(self.OpUDiv, uint, n, c(self.BLOCK)), c(stride))
                return e(self.OpIAdd, uint, e(self.OpIAdd, uint, c(out), blk), e(self.OpUMod, uint, n, c(self.BLOCK)))
            def comp4(b, w):
                vals = []
                for kk in range(4):
                    if w[kk] is None: vals.append(f1 if kk == 3 else f0)
                    else: vals.append(e(self.OpBitcast, flt, memr(e(self.OpIAdd, uint, b, c(w[kk] * self.BLOCK)))))
                return e(80, vec4, *vals)
            def vert(n):
                b_ = base(n)
                for v, w in sets: self.store(e(65, oe, v, n), comp4(b_, w))
            self.loop(c(max_v), vert)
            def idx(n):
                def byte(q):
                    at_ = e(self.OpIAdd, uint, c(prim + 4), e(self.OpIAdd, uint, e(self.OpIMul, uint, n, c(3)), c(q)))
                    w_ = memr(e(self.OpShiftRightLogical, uint, at_, c(2)))
                    return e(self.OpBitwiseAnd, uint, e(self.OpShiftRightLogical, uint, w_, e(self.OpIMul, uint, e(self.OpBitwiseAnd, uint, at_, c(3)), c(8))), c(255))
                i0, i1, i2 = byte(0), byte(1), byte(2)
                self.store(e(65, ie, tri, n), e(80, idt, i0, i1, i2))
                if psets:
                    b_ = base(n)
                    for v, w in psets:
                        val = comp4(b_, w)
                        for q in (i0, i1, i2): self.store(e(65, oe, v, q), val)
            self.loop(np, idx)
            self.code += [[2 << 16 | self.OpBranch, b]]
        if kind == 'ms': self.code += [[2 << 16 | self.OpLabel_, b]]
        else: self.code += [[2 << 16 | self.OpLabel_, b]]
        self.code += [[1 << 16 | self.OpReturn_], [1 << 16 | OpFunctionEnd]]
        m.ins += self.code; m.reindex()
        return True

    def spirv_14(self, m):
        """BufferBlock left the language after 1.3: Block, in the StorageBuffer class, with the access chains that start at those variables"""
        m.header[1] = 0x00010400
        structs = {x[1] for x in m.ins if x[0] & 0xffff == OpDecorate and x[2] == 3}
        for x in m.ins:
            if x[0] & 0xffff == OpDecorate and x[2] == 3: x[2] = 2
        m.reindex()
        sb = set()
        for x in list(m.ins):
            if x[0] & 0xffff == OpVariable and x[3] == 2:
                p = m.inst(x[1])
                if p[3] in structs:
                    t = m.new_id(); m.ins.insert(m.defs[x[2]], [4 << 16 | OpTypePointer, t, 12, p[3]]); m.reindex()   # before the variable: types come first
                    x[1] = t; x[3] = 12; sb.add(x[2])
        for x in list(m.ins):
            if x[0] & 0xffff in (OpAccessChain, OpInBoundsAccessChain) and x[3] in sb:
                x[1] = m.ensure_type(OpTypePointer, [12, m.inst(x[1])[3]]); sb.add(x[2])

    def emit(self, op, t, *a):
        r = self.m.new_id(); self.code.append([(3 + len(a)) << 16 | op, t, r] + list(a)); return r

    def store(self, p, v): self.code.append([3 << 16 | OpStore, p, v])

    def label(self):
        r = self.m.new_id(); self.code.append([2 << 16 | self.OpLabel_, r]); return r

    def loop(self, n, body):
        """for (i = 0; i < n; i++) body(i), with a function variable of its own (the entry block holds them)"""
        m, uint = self.m, self.uint
        v = m.new_id(); uintf = m.ensure_type(OpTypePointer, [7, uint])
        at = next(k for k, x in enumerate(self.code) if x[0] & 0xffff == OpVariable)
        self.code.insert(at, [4 << 16 | OpVariable, uintf, v, 7])
        self.store(v, self.c(0))
        h, b_, cont, end = m.new_id(), m.new_id(), m.new_id(), m.new_id()
        self.code += [[2 << 16 | self.OpBranch, h], [2 << 16 | self.OpLabel_, h]]
        i = self.emit(OpLoad, uint, v); t = self.emit(self.OpULessThan, self.boolt, i, n)
        self.code += [[4 << 16 | self.OpLoopMerge, end, cont, 0], [4 << 16 | self.OpBranchConditional, t, b_, end], [2 << 16 | self.OpLabel_, b_]]
        body(i)
        self.code += [[2 << 16 | self.OpBranch, cont], [2 << 16 | self.OpLabel_, cont]]
        self.store(v, self.emit(self.OpIAdd, uint, self.emit(OpLoad, uint, v), self.c(1)))
        self.code += [[2 << 16 | self.OpBranch, h], [2 << 16 | self.OpLabel_, end]]


def hash_info(i):
    import zlib, json
    return zlib.crc32(json.dumps(i, sort_keys=True).encode())


def encode_string(s):
    b = s.encode() + b'\0'; b += b'\0' * (-len(b) % 4)
    return list(struct.unpack('<%dI' % (len(b) // 4), b))


def m_ext(m):
    """the id of the GLSL.std.450 import"""
    return next(x[1] for x in m.ins if x[0] & 0xffff == OpExtInstImport)


class SpirvOptimize(Pass):
    """Off unless NXSHADER_VS_OPT=1 (it enters the cache key): vertex, tessellation and geometry shaders through spirv-opt -O. Made for the GPU freeze (docs/PLAN_NEXT.md
    step 5b): with 87c52decaf1a8e68 optimized the race stopped freezing (6 runs of 6), but an optimized vertex shader then faulted the same way at boot, so this
    is not the fix. Kept for tests and for its smaller modules. Not compute: FirstStore keeps word offsets into those modules."""
    name, version = 'spirv_opt', 1
    STAGES = ('vs', 'hs', 'ds', 'gs')
    _tool = None

    @classmethod
    def tool(cls):
        if cls._tool is None:
            here = os.path.dirname(os.path.abspath(__file__))
            import sys; sys.path.insert(0, os.path.dirname(here))
            import s2r_paths
            cls._tool = s2r_paths.spirv_tool('spirv-opt') or ''
        return cls._tool

    _ver = None

    @property
    def options(self):
        if os.environ.get('NXSHADER_VS_OPT') != '1': return 'off'
        if SpirvOptimize._ver is None:
            t = self.tool()
            r = subprocess.run([t, '--version'], capture_output=True, text=True) if t else None
            SpirvOptimize._ver = next((l.strip() for l in ((r.stdout + r.stderr) if r else '').splitlines() if l.strip()), 'missing')
        return '-O ' + SpirvOptimize._ver

    def applies(self, m, stage): return stage in self.STAGES and os.environ.get('NXSHADER_VS_OPT') == '1' and bool(self.tool())

    def run(self, m, stage):
        fd, a = tempfile.mkstemp(suffix='.spv'); os.close(fd); b = a + '.opt.spv'
        try:
            open(a, 'wb').write(m.bytes())
            r = subprocess.run([self.tool(), '-O', '--target-env=vulkan1.3', a, '-o', b], capture_output=True, text=True)
            if r.returncode or not os.path.exists(b): return False   # left as translated
            o = Module(open(b, 'rb').read())
            m.header, m.ins = o.header, o.ins; m.reindex()
            return True
        finally:
            for p in (a, b):
                try: os.remove(p)
                except OSError: pass


DEFAULT_PASSES = [TrimCapabilities(), VsBaseInstance(), StorageFormatCaps(), VertexOutputInit(), Fetch1DTexelBuffer(), NanMinMax(), AbsGuard(), FirstStore(), LoopGuard(), SpirvOptimize()]
VARIANTS = {'gl': lambda opts: [DepthGL()], 'dual': lambda opts: [DualSource()], 'pad': lambda opts: [PadOutputs(opts)]}


# ===================================================================================== analysis

def int_only_inputs(m):
    """Vertex inputs used only as integers (every use of a loaded value is a bit cast to an integer type, possibly through extracts, shuffles and copies; a store
    of the value somewhere is neutral): {location: 'uint'}. nxnative's int_inputs did this on the MSL; here on the SPIR-V after spirv-opt's SSA rewrite (a copy
    made for the analysis only). The translator declares every attribute float without an answer, and Vulkan then converts an integer vertex format to float
    values instead of passing its bits (bone indices: skinned characters stretched into streaks)"""
    uses, in_func = {}, False   # uses inside functions only: annotations and types hold literals that look like ids
    for x in m.ins:
        op = x[0] & 0xffff
        if op == OpFunction: in_func = True
        if not in_func: continue
        operands = x[3:] if op in RESULT_TYPED else x[1:]
        for o in operands:
            if o in m.defs: uses.setdefault(o, []).append(x)
    def classify(rid, seen):
        n_int, bad = 0, False
        for u in uses.get(rid, []):
            op = u[0] & 0xffff
            if op == OpBitcast:
                t = m.inst(u[1]); tt = m.inst(t[2]) if t and t[0] & 0xffff == OpTypeVector else t
                if tt and tt[0] & 0xffff == OpTypeInt: n_int += 1
                else: bad = True
            elif op in (OpCompositeExtract, OpVectorShuffle, OpCopyObject, OpCompositeConstruct) and u[2] not in seen:
                seen.add(u[2]); a, b = classify(u[2], seen); n_int += a; bad |= b
            elif op == OpStore and u[2] == rid: pass
            else: bad = True
            if bad: break
        return n_int, bad
    out = {}
    for vid, sc, ptr in m.global_vars():
        loc = m.deco(vid, DecLocation)
        if sc != SC_Input or loc is None: continue
        ptrs, loads = {vid}, []
        for x in m.ins:
            op = x[0] & 0xffff
            if op in (OpAccessChain, OpInBoundsAccessChain) and x[3] in ptrs: ptrs.add(x[2])
        for x in m.ins:
            if x[0] & 0xffff == OpLoad and x[3] in ptrs: loads.append(x[2])
        n_int, bad = 0, False
        for r in loads:
            a, b = classify(r, {r}); n_int += a; bad |= b
        if n_int and not bad: out[str(loc[0])] = 'uint'
    return out
