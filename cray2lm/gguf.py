"""Minimal GGUF v3 reader (metadata and tensor table), Python standard library only."""
import math
import struct

_SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}
# bytes per block and elements per block for the tensor types used here
_BLOCK = {0: (4, 1),        # F32
          41: (18, 128),    # Q1_0: fp16 scale + 128 sign bits
          142: (34, 128)}   # PQ2_0: fp16 scale + 128 2-bit codes


class GGUF:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            if f.read(4) != b'GGUF':
                raise ValueError('not a GGUF file')
            version, nt, nm = struct.unpack('<IQQ', f.read(20))
            if version != 3:
                raise ValueError('GGUF version %d not supported' % version)
            self.metadata = {}
            for _ in range(nm):
                key = self._str(f)
                self.metadata[key] = self._value(f, struct.unpack('<I', f.read(4))[0])
            self.tensors = {}
            for _ in range(nt):
                name = self._str(f)
                nd = struct.unpack('<I', f.read(4))[0]
                shape = list(struct.unpack('<%dQ' % nd, f.read(8 * nd)))
                typ, off = struct.unpack('<IQ', f.read(12))
                self.tensors[name] = dict(shape=shape, type=typ, offset=off, elements=math.prod(shape))
            align = self.metadata.get('general.alignment', 32)
            start = (f.tell() + align - 1) // align * align
        for t in self.tensors.values():
            t['absolute_offset'] = start + t['offset']
            if t['type'] in _BLOCK:
                b, n = _BLOCK[t['type']]
                t['bytes'] = t['elements'] // n * b

    @staticmethod
    def _str(f):
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8')

    def _value(self, f, t):
        if t in _SCALAR:
            fmt = _SCALAR[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        if t == 8:
            return self._str(f)
        if t == 9:
            et, n = struct.unpack('<IQ', f.read(12))
            if et in _SCALAR:
                fmt = _SCALAR[et]
                size = struct.calcsize(fmt)
                return list(struct.unpack('<%d%s' % (n, fmt[1]), f.read(size * n)))
            if et == 8:
                return [self._str(f) for _ in range(n)]
            raise ValueError('nested GGUF arrays not supported')
        raise ValueError('unknown GGUF value type %d' % t)
