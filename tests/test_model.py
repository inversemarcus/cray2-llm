#!/usr/bin/env python3
"""End-to-end check of a built model on the emulated CRAY-2.

    python3 tests/test_model.py --model ternary6

Needs the model built with build.py.  Checks that:
  * token 198 at position 0 selects the token recorded in tests/expected.json
    (from llama.cpp on the same model), in both exact and fast emulation;
  * the exact clock count matches the recorded value (catches timing changes);
  * a position at or beyond the KV-cache length is refused.
Exits non-zero on any failure.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cray2lm.machine import Machine, default_dirs  # noqa: E402

FAIL = []


def check(name, cond, detail=''):
    print('%-52s %s %s' % (name, 'PASS' if cond else 'FAIL', detail))
    if not cond:
        FAIL.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='ternary6', choices=['ternary6', 'ternary8', 'binary'])
    a = ap.parse_args()
    want = json.loads((ROOT / 'tests' / 'expected.json').read_text())[a.model]
    build, image = default_dirs(a.model)
    for fast in (False, True):
        m = Machine(build, image, fast=fast, log=lambda *x: None)
        tok, logit, cp = m.step(198, 0)
        mode = 'fast' if fast else 'exact'
        check('%s: token 198 at position 0 selects %d' % (mode, want['token']), tok == want['token'], str(tok))
        if not fast:
            check('exact: clock periods %d' % want['cp'], cp == want['cp'], str(cp))
            try:
                m.step(198, m.ctx)
                check('position %d refused' % m.ctx, False)
            except ValueError:
                check('position %d refused' % m.ctx, True)
        m.close()
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
