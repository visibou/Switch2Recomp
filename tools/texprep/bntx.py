"""bntx.py: read the textures of a BNTX file (NintendoWare texture container, version 4.x as in Switch 2 games): each texture's
format, size, levels, layers, tiling byte and storage bytes. Used by texprep.py."""
import struct

# BNTX format type (high byte of the format word) -> ASTC index 0..13 (4x4 .. 12x12)
ASTC_TYPES = {0x2d + k: k for k in range(14)}

def textures(d):
    """Yield dicts for each BRTI block: name offset not decoded; data = the texture's storage bytes (all levels and layers)."""
    i = 0
    while True:
        i = d.find(b'BRTI', i)
        if i < 0: return
        try:
            mips = struct.unpack_from('<H', d, i + 0x16)[0]
            fmt = struct.unpack_from('<I', d, i + 0x1c)[0]
            w, h, dep, arr, lay0, lay1 = struct.unpack_from('<IIIIII', d, i + 0x24)
            size = struct.unpack_from('<I', d, i + 0x50)[0]
            mp = struct.unpack_from('<Q', d, i + 0x70)[0]
            off0 = struct.unpack_from('<Q', d, mp)[0] if 0 < mp < len(d) - 8 else 0
        except struct.error:
            i += 4; continue
        yield dict(at=i, mips=mips, fmt=fmt, type=fmt >> 8, kind=fmt & 0xff, w=w, h=h, depth=dep, layers=arr, lay0=lay0, lay1=lay1,
                   size=size, off=off0, data=d[off0:off0 + size] if off0 and off0 + size <= len(d) else b'')
        i += 4

def nvn_format(t):
    """The NVN format the game creates for an ASTC texture (0x79.. unorm, 0x87.. sRGB), or None."""
    k = ASTC_TYPES.get(t['type'])
    if k is None: return None
    return (0x87 if t['kind'] == 6 else 0x79) + k
