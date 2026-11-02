"""Build the CRAY-2 Common Memory image (and, for the ternary model, the disk
image holding the token-embedding table) from a Bonsai GGUF file.

Offline storage conversion only: the C converters in tools/ repack weight
bits and fp16 scales into the layouts the CAL kernels expect; no inference
arithmetic is done here.

    binary  (Bonsai-8B Q1_0):          weights.img, layout.json
    ternary (Ternary-Bonsai-8B PQ2_0): weights.img, embd.disk, layout.json
"""
import json
import math
import pathlib
import struct
import subprocess

from .gguf import GGUF

ROOT = pathlib.Path(__file__).resolve().parent.parent
L = 36
MATS = ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down']
MODELS = {
    # model: (GGUF tensor type, Common Memory base of the weights, converter)
    'binary': (41, 4 << 20, 'convert_q1'),
    'ternary': (142, 0x180000, 'convert_pq2'),
}


def cray(x):
    """Cray-2 floating representation of a Python float."""
    if x == 0:
        return 0
    m, e = math.frexp(abs(x))
    return (int(x < 0) << 63) | ((e + 16384) << 48) | int(m * (1 << 48))


def tile_words(model, rows, cols):
    if model == 'binary':
        return (rows + 63) // 64 * (cols // 128) * 160
    return (rows + 63) // 64 * (cols // 128 // 4) * (64 + 1024)


def converter(name, build_dir):
    exe = pathlib.Path(build_dir) / name
    src = ROOT / 'tools' / (name + '.c')
    if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
        subprocess.check_call(['cc', '-O2', '-o', str(exe), str(src)])
    return exe


def build_image(gguf_path, out_dir, model, log=print):
    gtype, base, conv_name = MODELS[model]
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = GGUF(gguf_path)
    T = g.tensors
    assert g.metadata['general.architecture'] == 'qwen3' and g.metadata['qwen3.block_count'] == L
    assert all(t['type'] in (0, gtype) for t in T.values()), 'unexpected tensor types for %s model' % model
    conv = converter(conv_name, out)
    addr = base
    lay = {'weights_base': base, 'model': model, 'source': pathlib.Path(gguf_path).name, 'tensors': {}}
    jobs = []
    order = ['blk.%d.%s.weight' % (l, k) for l in range(L) for k in MATS] + ['output.weight']
    if model == 'binary':
        order.append('token_embd.weight')
    for n in order:
        t = T[n]
        cols, rows = t['shape']
        kind = 'rows' if n == 'token_embd.weight' else 'tiles'
        words = rows * (cols // 128) * 3 if kind == 'rows' else tile_words(model, rows, cols)
        lay['tensors'][n] = dict(addr=addr, rows=rows, cols=cols, kind=kind, words=words)
        jobs.append((n, t['absolute_offset'], rows, cols, addr - base, kind, 'weights.img'))
        addr += words
    norms = sorted((n, t) for n, t in T.items() if t['type'] == 0)
    for n, t in norms:
        lay['tensors'][n] = dict(addr=addr, n=t['elements'], kind='f32')
        addr += t['elements']
    lay['end'] = addr
    if model == 'ternary':
        cols, rows = T['token_embd.weight']['shape']
        lay['embedding_disk'] = dict(file='embd.disk', rows=rows, cols=cols, row_words=cols // 128 * 5)
        jobs.append(('token_embd.weight', T['token_embd.weight']['absolute_offset'], rows, cols, 0, 'rows', 'embd.disk'))
        with open(out / 'embd.disk', 'wb') as f:
            f.truncate(rows * (cols // 128 * 5) * 8)
    with open(out / 'weights.img', 'wb') as f:
        f.truncate((addr - base) * 8)
    for n, off, rows, cols, at, kind, img in jobs:
        r = subprocess.run([str(conv), str(gguf_path), str(off), str(rows), str(cols), str(out / img), str(at), kind],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit('%s: %s' % (n, r.stderr))
        log('%s %s' % (n, r.stdout.strip()))
    with open(gguf_path, 'rb') as src, open(out / 'weights.img', 'r+b') as f:
        for n, t in norms:
            src.seek(t['absolute_offset'])
            v = struct.unpack('<%df' % t['elements'], src.read(4 * t['elements']))
            f.seek((lay['tensors'][n]['addr'] - base) * 8)
            f.write(struct.pack('<%dQ' % len(v), *(cray(x) for x in v)))
    lay['config'] = {k: v for k, v in g.metadata.items() if k.startswith('qwen3.')}
    (out / 'layout.json').write_text(json.dumps(lay, indent=1))
    log('image ends at word %d (%.2f of 256 Mwords)' % (addr, addr / (1 << 20)))
    return out
