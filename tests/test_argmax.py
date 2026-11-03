#!/usr/bin/env python3
"""Run the generated 64-lane argmax reduction (calgen.Gen.argmax_lanes) on the
emulator with ties and check it picks the largest logit, lowest token on a tie."""
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cray2lm.calgen import Gen, cf  # noqa: E402
from cray2lm.machine import Sim, parse_exit  # noqa: E402
from cray2 import cal2  # noqa: E402  (on the path via cray2lm.machine)

LM = {'am': 100, 'ai': 200}
SRC, OUT = 0x2000, 0x3000            # Common Memory: 64 logits, 64 tokens; result


def run(logits, tokens):
    g = Gen(LM, 0)
    e = g.e
    e('A1 64')
    e('VL A1')
    e('A2 1')
    for src, lm in ((SRC, 'am'), (SRC + 64, 'ai')):
        e('A3 %d' % src)
        e('V1 (A3,A2)')
        e('A3 %d' % LM[lm])
        e('[A3] V1')
    g.argmax_lanes('lane')
    e('(%d) S5' % OUT)
    e('(%d) S6' % (OUT + 1))
    e('EXIT 1')
    s = Sim()
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, 'a.cal'), os.path.join(d, 'a.bin')
        open(src, 'w').write('\n'.join(g.out) + '\n')
        cal2.assemble_file(src, out)
        assert s.cmd('load %s 0' % out).startswith('OK')
    s.write(SRC, [cf(x) for x in logits] + tokens)
    assert s.cmd('start 0 0') == 'OK'
    code = parse_exit([l for l in s.go() if l.startswith('EXIT')][0])[1]
    r = s.read(OUT, 2)
    s.close()
    assert code == 1, code
    return r[1]


def main():
    fail = 0
    cases = [
        # (description, logits, tokens, expected token)
        ('tie: lower token in a later lane', [1.0] + [0.0] * 62 + [1.0], [64] + list(range(2, 64)) + [1], 1),
        ('tie: lower token in an earlier lane', [2.0, 2.0] + [0.0] * 62, [5, 9] + list(range(100, 162)), 5),
        ('unique maximum', [0.0] * 40 + [3.0] + [0.0] * 23, list(range(64)), 40),
        ('all equal', [-1.5] * 64, list(range(1000, 936, -1)), 937),
    ]
    for name, logits, tokens, want in cases:
        got = run(logits, tokens)
        ok = got == want
        fail += not ok
        print('%-40s %s %d' % (name, 'PASS' if ok else 'FAIL', got))
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
