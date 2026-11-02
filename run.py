#!/usr/bin/env python3
"""Run token IDs through the emulated CRAY-2: timing probes and verification.

    python3 run.py --model ternary6 --tokens 198                  # one token, exact timing
    python3 run.py --model ternary6 --tokens 198 --pos 150        # timing at a later position
    python3 run.py --model ternary6 --tokens 198 --ref DIR        # per-layer comparison
    python3 run.py --model binary --tokens 198 --timing v_log=10,v_int=10   # sensitivity

--ref DIR holds a native capture (l_out-<layer>.f32.bin, final_logits.f32.bin)
made with tools/native_capture.cpp for the same token at position 0.  The per-layer
stops used by --ref shift the clock slightly; quote timings from runs without it.  Timing
is the exact single-threaded emulation unless --fast is given.
"""
import argparse
import math
import pathlib
import struct
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cray2lm.machine import CLOCK_NS, Machine  # noqa: E402


def compare(x, ref):
    rr = sum(b * b for b in ref) or 1e-300
    err = math.sqrt(sum((a - b) ** 2 for a, b in zip(x, ref)) / rr)
    xx = sum(a * a for a in x) or 1e-300
    return err, sum(a * b for a, b in zip(x, ref)) / math.sqrt(xx * rr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='ternary6', choices=['ternary6', 'ternary8', 'binary'])
    ap.add_argument('--tokens', default='198', help='comma-separated token IDs fed at successive positions')
    ap.add_argument('--pos', type=int, default=0, help='first position (timing probes)')
    ap.add_argument('--ref', default='', help='native capture directory for per-layer comparison')
    ap.add_argument('--fast', action='store_true', help='multithreaded emulation')
    ap.add_argument('--timing', default='', help='comma list of emulator timing overrides name=value')
    a = ap.parse_args()
    build = ROOT / 'build' / a.model
    image = ROOT / 'images' / ('binary' if a.model == 'binary' else 'ternary')
    m = Machine(build, image, fast=a.fast, timing=[x for x in a.timing.split(',') if x], debug=bool(a.ref))

    def on_layer(layer, t):
        x = m.read_floats(m.meta['X'], 4096)
        ref = struct.unpack('<4096f', (pathlib.Path(a.ref) / ('l_out-%d.f32.bin' % layer)).read_bytes())
        err, cos = compare(x, ref)
        print('layer %2d  rel.L2 %.2e  cos %.8f' % (layer, err, cos), flush=True)

    failed = False
    pos = a.pos
    for t in [int(v) for v in a.tokens.split(',')]:
        h0 = time.time()
        nxt, logit, cp = m.step(t, pos, on_layer if a.ref else None)
        print('pos %d: token %d -> %d (logit %.4f)  %d CP = %.3f s CRAY-2  [host %.1f s]'
              % (pos, t, nxt, logit, cp, cp * CLOCK_NS * 1e-9, time.time() - h0), flush=True)
        if a.ref and pos == 0:
            ref = struct.unpack('<151669f', (pathlib.Path(a.ref) / 'final_logits.f32.bin').read_bytes())
            x = m.read_floats(m.meta['LOGITS'], 151669)
            err, cos = compare(x, ref)
            best = max(range(len(ref)), key=ref.__getitem__)
            print('logits    rel.L2 %.2e  cos %.8f  native argmax %d %s' % (err, cos, best,
                  'MATCH' if best == nxt else 'DIFFERENT'))
            failed |= best != nxt
        pos += 1
    for line in m.s.stats():
        print(line)
    m.close()
    if failed:
        sys.exit('FAIL: selected token differs from the native reference')


if __name__ == '__main__':
    main()
