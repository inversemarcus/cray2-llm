#!/usr/bin/env python3
"""Ask a question; the answer is computed by the emulated 1985 CRAY-2.

    python3 chat.py "Which river is the longest in Europe?"
    python3 chat.py --model binary "What is 17 * 23?"
    python3 chat.py --raw "The capital of France is"
    python3 chat.py --exact "..."      # single-threaded reference timing

Tokenization happens on the host (the Foreground Processor's job); every
neural-network operation runs as CRAY-2 instructions in the emulator.  The
reported seconds are emulated CRAY-2 time (4.1 ns clock periods).
"""
import argparse
import codecs
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cray2lm.machine import CLOCK_NS, Machine, default_dirs  # noqa: E402
from cray2lm.tokenizer import Tokenizer  # noqa: E402

TEMPLATE = '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('prompt')
    ap.add_argument('--model', default='ternary6', choices=['ternary6', 'ternary6-long', 'ternary8', 'binary'])
    ap.add_argument('--raw', action='store_true', help='do not apply the chat template')
    ap.add_argument('--max-new-tokens', type=int, default=48)
    ap.add_argument('--exact', action='store_true',
                    help='single-threaded emulation with exact cross-processor bank timing (slower)')
    a = ap.parse_args()
    build, image = default_dirs(a.model)
    if not (build / 'program.bin').exists():
        sys.exit('model not built: python3 build.py %s --gguf ...' % a.model)
    tok = Tokenizer(cache=str(image / 'vocab.json'))
    ids = tok.encode(a.prompt if a.raw else TEMPLATE % a.prompt)
    err = sys.stderr
    m = Machine(build, image, fast=not a.exact, log=lambda s: print('[%s]' % s, file=err))
    if len(ids) + a.max_new_tokens > m.ctx:
        sys.exit('prompt too long for the %d-position KV cache' % m.ctx)
    t0 = time.time()
    cps = []
    for pos, t in enumerate(ids):
        nxt, _, cp = m.step(t, pos)
        cps.append(cp)
        print('\r[reading prompt %d/%d]' % (pos + 1, len(ids)), end='', file=err, flush=True)
    print(file=err)
    gen, pos = [], len(ids)
    utf8 = codecs.getincrementaldecoder('utf-8')(errors='replace')   # tokens can split characters
    while True:
        gen.append(nxt)
        if tok.is_eog(nxt):
            break
        sys.stdout.write(utf8.decode(tok.decode_bytes([nxt])))
        sys.stdout.flush()
        if len(gen) >= a.max_new_tokens:
            break
        nxt, _, cp = m.step(nxt, pos)
        cps.append(cp)
        pos += 1
    sys.stdout.write(utf8.decode(b'', final=True))
    print()
    secs = [c * CLOCK_NS * 1e-9 for c in cps]
    print('[%d forward passes on 4 CRAY-2 processors: %.1f s of CRAY-2 time (%.2f-%.2f s per token)%s; host %.0f s]'
          % (len(cps), sum(secs), min(secs), max(secs), '' if a.exact else ', fast mode', time.time() - t0), file=err)
    m.close()


if __name__ == '__main__':
    main()
