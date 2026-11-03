"""Host side of the emulated CRAY-2: plays the Foreground Processor.

It starts cray2sim, loads Common Memory (program, weight image, RoPE table,
layer descriptors), deadstarts the four Background Processors, and services
their exits through cray2sim's line protocol:

    EXIT 1   CPU0 asks for the next token: the host writes token and position
             into the mailbox (and, for the ternary model, reads that token's
             embedding row from the disk image into Common Memory)
    EXIT 2   CPU0 has selected the next token; the host reads it back
    EXIT 20  optional stop after each layer (debug flag in the mailbox)

The host never does model arithmetic.  Times are emulated CRAY-2 clock
periods (4.1 ns), not host time.
"""
import json
import pathlib
import struct
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
EMU = ROOT / 'external' / 'cray2sim'
sys.path.insert(0, str(EMU))
from cray2.client import CLOCK_NS, Sim as _Sim, cray_to_float, parse_exit  # noqa: E402,F401


def Sim(args=()):
    return _Sim(args, exe=EMU / 'sim' / 'cray2sim')

DISK_MS = 25.0   # assumed single-record disk access (seek + rotation); see docs/TIMING.md


class Machine:
    """A loaded, deadstarted CRAY-2 running one model program.

    build_dir: output of build.py (program.bin, rope.img, desc.img, meta.json)
    image_dir: weight image (weights.img, layout.json, embd.disk for ternary)
    fast:      multithreaded emulation (bit-identical results; Cray clock about 1-6% low)
    """

    def __init__(self, build_dir, image_dir, fast=False, qwin=2000, timing=(), debug=False, log=print):
        self.build_dir, self.image_dir = pathlib.Path(build_dir), pathlib.Path(image_dir)
        self.meta = json.loads((self.build_dir / 'meta.json').read_text())
        self.layout = json.loads((self.image_dir / 'layout.json').read_text())
        args = list(timing) + (['mt=1', 'qwin=%d' % qwin] if fast else [])
        self.s = Sim(args)
        t0 = time.time()
        for f, at in [(self.build_dir / 'program.bin', 0), (self.image_dir / 'weights.img', self.layout['weights_base']),
                      (self.build_dir / 'rope.img', self.meta['ROPE']), (self.build_dir / 'desc.img', self.meta['DESC'])]:
            r = self.s.cmd('load %s %d' % (f, at))
            assert r.startswith('OK'), r
        log('Common Memory loaded in %.1fs' % (time.time() - t0))
        self.disk = None
        if 'embedding_disk' in self.layout:
            self.disk = open(self.image_dir / self.layout['embedding_disk']['file'], 'rb')
            self.row_words = self.layout['embedding_disk']['row_words']
        self.ctx = self.meta['CTX']
        self.s.write(self.meta['MBOX'] + 3, [1 if debug else 0])
        for c in range(4):
            assert self.s.cmd('start %d 0' % c) == 'OK'
        self.pending = self._run_until(1)

    def _run_until(self, want, on_layer=None):
        while True:
            exits = [parse_exit(l) for l in self.s.go() if l.startswith('EXIT')]
            if not exits:
                raise RuntimeError('all processors idle')
            cpu, code, t, A = exits[0]
            if cpu == 0 and code == want:
                return t
            if cpu == 0 and code == 20:
                if on_layer:
                    on_layer(36 - A[1], t)
                assert self.s.cmd('resume 0') == 'OK'
                continue
            raise RuntimeError('unexpected exit: CPU %d code %d' % (cpu, code))

    def step(self, token, pos, on_layer=None):
        """Run one forward pass.  Returns (next_token, its logit, Cray clock periods)."""
        if pos >= self.ctx:
            raise ValueError('position %d exceeds the %d-position KV cache' % (pos, self.ctx))
        extra = 0
        if self.disk:
            # Foreground Processor disk read of this token's embedding row
            self.disk.seek(token * self.row_words * 8)
            row = struct.unpack('<%dQ' % self.row_words, self.disk.read(self.row_words * 8))
            self.s.write(self.meta['EMB'], list(row))
            extra = int(DISK_MS * 1e-3 / (CLOCK_NS * 1e-9))
        mb = self.meta['MBOX']
        self.s.write(mb, [token, pos])
        t0 = self.pending
        assert self.s.cmd('resume 0') == 'OK'
        t1 = self._run_until(2, on_layer)
        out = self.s.read(mb + 2, 3)
        assert self.s.cmd('resume 0') == 'OK'
        self.pending = self._run_until(1)
        return out[0], cray_to_float(out[2]), t1 - t0 + extra

    def read_floats(self, addr, n):
        out = []
        for i in range(0, n, 4096):
            out += [cray_to_float(w) for w in self.s.read(addr + i, min(4096, n - i))]
        return out

    def close(self):
        self.s.close()
        if self.disk:
            self.disk.close()


def default_dirs(model):
    """Conventional locations used by build.py."""
    return ROOT / 'build' / model, ROOT / 'images' / ('binary' if model == 'binary' else 'ternary')
