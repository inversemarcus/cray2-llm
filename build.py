#!/usr/bin/env python3
"""Build everything needed to run a model on the emulated CRAY-2.

    python3 build.py ternary6 --gguf Ternary-Bonsai-8B-PQ2_0.gguf    (default model)
    python3 build.py ternary8 --gguf Ternary-Bonsai-8B-PQ2_0.gguf
    python3 build.py binary   --gguf Bonsai-8B-Q1_0.gguf

Steps (each skipped if already done):
  1. images/<binary|ternary>/   Common Memory weight image (+ embedding disk image)
  2. images/.../vocab.json      tokenizer vocabulary read from the GGUF
  3. build/<model>/             generated CAL program, assembled program.bin,
                                RoPE table, layer descriptors
The emulator (external/cray2sim) is built by `make`.
"""
import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cray2lm import gen_binary, gen_ternary, layout  # noqa: E402
from cray2lm.machine import EMU  # noqa: E402,F401  (puts the emulator package on sys.path)
from cray2 import cal2  # noqa: E402
from cray2lm.tokenizer import Tokenizer  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402

MODELS = {'binary': ('binary', None), 'ternary6': ('ternary', 6), 'ternary8': ('ternary', 8),
          'ternary6-long': ('ternary', 6)}   # packed attention cache, 400 positions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', choices=sorted(MODELS))
    ap.add_argument('--gguf', required=True, help='Bonsai-8B-Q1_0.gguf or Ternary-Bonsai-8B-PQ2_0.gguf')
    ap.add_argument('--force', action='store_true', help='rebuild the weight image even if present')
    a = ap.parse_args()
    kind, bits = MODELS[a.model]
    img = ROOT / 'images' / kind
    h = hashlib.sha256()
    with open(a.gguf, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    digest = h.hexdigest()
    if (img / 'layout.json').exists():
        old = json.loads((img / 'layout.json').read_text()).get('source_sha256')
        if old != digest and not a.force:
            sys.exit('images/%s was built from a different GGUF (sha256 %s); use --force to rebuild' % (kind, old))
    if a.force and img.exists():
        shutil.rmtree(img)
    if not (img / 'layout.json').exists():
        print('[1/3] converting weights into a CRAY-2 memory image (a few minutes)')
        layout.build_image(a.gguf, img, kind, log=lambda s: None)
        lay = json.loads((img / 'layout.json').read_text())
        lay['source_sha256'] = digest
        (img / 'layout.json').write_text(json.dumps(lay, indent=1))
    else:
        print('[1/3] weight image present: %s' % img)
    if not (img / 'vocab.json').exists():
        print('[2/3] reading tokenizer vocabulary')
        Tokenizer(a.gguf, cache=str(img / 'vocab.json'))
    out = ROOT / 'build' / a.model
    print('model %s, GGUF sha256 %s' % (a.model, digest))
    print('[3/3] generating and assembling the CAL program -> %s' % out)
    if kind == 'binary':
        gen_binary.generate(img / 'layout.json', out)
    else:
        gen_ternary.generate(img / 'layout.json', out, bits, kv_pack=a.model.endswith('-long'))
    n = cal2.assemble_file(str(out / 'program.cal'), str(out / 'program.bin'))
    print('done: %d words of CRAY-2 code' % n)


if __name__ == '__main__':
    main()
