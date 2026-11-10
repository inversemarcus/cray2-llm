#!/usr/bin/env python3
"""Record a teacher model's next-token predictions on the training chunks.

    python3 distill/teacher_topk.py TEACHER_MLX_DIR CHUNKS_DIR OUT_DIR [--k 32]

For every chunk (513 tokens) the teacher reads the first 512 and its top-k
next-token log-probabilities at each position are saved:
OUT_DIR/topk-NNN.npz with ids (uint32 [n,512,k]) and logp (float16 [n,512,k]).
Teacher and student must share the tokenizer (checked by the caller).
"""
import argparse
import pathlib
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('teacher')
    ap.add_argument('chunks')
    ap.add_argument('out')
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--batch', type=int, default=2)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load(a.teacher)
    for f in sorted(pathlib.Path(a.chunks).glob('chunks-*.npy')):
        dst = out / f.name.replace('chunks-', 'topk-').replace('.npy', '.npz')
        if dst.exists():
            continue
        rows = np.load(f)
        ids_all = np.zeros((len(rows), rows.shape[1] - 1, a.k), np.uint32)
        lp_all = np.zeros((len(rows), rows.shape[1] - 1, a.k), np.float16)
        t0 = time.time()
        for i in range(0, len(rows), a.batch):
            x = mx.array(rows[i:i + a.batch, :-1].astype(np.int32))
            logits = model(x).astype(mx.float32)
            logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            top = mx.argpartition(-logp, a.k - 1, axis=-1)[..., :a.k]
            vals = mx.take_along_axis(logp, top, axis=-1)
            mx.eval(top, vals)
            ids_all[i:i + a.batch] = np.array(top)
            lp_all[i:i + a.batch] = np.array(vals).astype(np.float16)
            if (i // a.batch) % 100 == 99:
                done = (i + a.batch) * (rows.shape[1] - 1)
                print('  %s %d/%d rows, %.0f tok/s' % (f.name, i + a.batch, len(rows), done / (time.time() - t0)),
                      flush=True)
        np.savez(dst, ids=ids_all, logp=lp_all)
        n = len(rows) * (rows.shape[1] - 1)
        print('%s: %d tokens, %.0f tok/s' % (dst.name, n, n / (time.time() - t0)), flush=True)


if __name__ == '__main__':
    main()
