#!/usr/bin/env python3
"""Train a low-rank correction on frozen Ternary-Bonsai-8B to match a teacher.

    python3 distill/train.py STUDENT_MLX_DIR CHUNKS_DIR TOPK_DIR OUT_DIR
        [--layers 12-23] [--rank 16] [--tokens 5000000] [--lr 2e-4]

The ternary weights stay frozen.  Each MLP projection (gate/up/down) of the
chosen layers gets y = W x + s B(A x) (mlx-lm LoRALinear; B starts at zero, so
training starts from the unchanged model).  Loss: KL divergence from the
teacher's top-k next-token distribution (renormalized over those k) to the
student's, plus a small cross-entropy on the actual next token.  The
correction is saved as adapters.safetensors (A, B per matrix) and kept
separate from the ternary weights, as on the CRAY-2.
"""
import argparse
import json
import pathlib
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear


def batches(chunks, topk, rng):
    files = sorted(pathlib.Path(chunks).glob('chunks-*.npy'))
    while True:
        for f in rng.permutation(len(files)):
            t = pathlib.Path(topk) / files[f].name.replace('chunks-', 'topk-').replace('.npy', '.npz')
            if not t.exists():
                continue
            rows, z = np.load(files[f]), np.load(t)
            for i in rng.permutation(len(rows)):
                yield rows[i], z['ids'][i], z['logp'][i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('student')
    ap.add_argument('chunks')
    ap.add_argument('topk')
    ap.add_argument('out')
    ap.add_argument('--layers', default='12-23')
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--scale', type=float, default=2.0)
    ap.add_argument('--tokens', type=int, default=5_000_000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--ce', type=float, default=0.1)
    ap.add_argument('--save-every', type=int, default=500_000)
    ap.add_argument('--warmup', type=int, default=30, help='optimizer steps of linear learning-rate warm-up')
    ap.add_argument('--clip', type=float, default=1.0, help='gradient norm clip')
    ap.add_argument('--gap-weight', type=float, default=1.0,
                    help='extra weight on knowledge gaps: positions where the teacher puts >0.5 on its top token '
                         'and the student puts <0.5 on it (1.0 = off)')
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(vars(a), indent=1))
    mx.random.seed(1985)
    model, _ = load(a.student)
    model.freeze()
    lo, hi = (0, len(model.model.layers) - 1) if a.layers == 'all' else map(int, a.layers.split('-'))
    for i in range(lo, hi + 1):
        mlp = model.model.layers[i].mlp
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            setattr(mlp, name, LoRALinear.from_base(getattr(mlp, name), r=a.rank, scale=a.scale, dropout=0.0))
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print('trainable parameters: %d' % n_train, flush=True)

    def loss_fn(model, x, y, tid, tlp):
        logits = model(x).astype(mx.float32)
        slogp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        pt = mx.softmax(tlp.astype(mx.float32), axis=-1)            # teacher over its top-k
        s_at = mx.take_along_axis(slogp, tid, axis=-1)
        s_at = s_at - mx.logsumexp(s_at, axis=-1, keepdims=True)     # student over the same k
        kl_pos = mx.sum(pt * (mx.log(pt + 1e-9) - s_at), axis=-1)
        if a.gap_weight != 1.0:
            # teacher's top token is the first of its top-k by probability
            top = mx.argmax(tlp, axis=-1)[..., None]
            t_top = mx.take_along_axis(pt, top, axis=-1)[..., 0]
            s_top = mx.exp(mx.take_along_axis(slogp, mx.take_along_axis(tid, top, axis=-1), axis=-1)[..., 0])
            gap = mx.stop_gradient(((t_top > 0.5) & (s_top < 0.5)).astype(mx.float32))
            w = 1.0 + (a.gap_weight - 1.0) * gap
            kl = mx.sum(w * kl_pos) / mx.sum(w)
        else:
            kl = kl_pos.mean()
        ce = -mx.take_along_axis(slogp, y[..., None], axis=-1).mean()
        return kl + a.ce * ce, (kl, ce)

    total = max(1, a.tokens // 512 // a.accum)
    sched = optim.join_schedules([optim.linear_schedule(a.lr / 100, a.lr, a.warmup),
                                  optim.cosine_decay(a.lr, max(1, total - a.warmup))], [a.warmup])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.0)
    vg = nn.value_and_grad(model, lambda m, *b: loss_fn(m, *b)[0])
    rng = np.random.default_rng(1985)
    it = batches(a.chunks, a.topk, rng)
    seen, step, t0, acc, logs = 0, 0, time.time(), None, []
    next_save = a.save_every
    while seen < a.tokens:
        grads_sum, lsum = None, 0.0
        for _ in range(a.accum):
            row, tid, tlp = next(it)
            x = mx.array(row[None, :-1].astype(np.int32))
            y = mx.array(row[None, 1:].astype(np.int32))
            loss, g = vg(model, x, y, mx.array(tid[None].astype(np.int32)), mx.array(tlp[None]))
            grads_sum = g if grads_sum is None else tree_map(lambda u, v: u + v, grads_sum, g)
            mx.eval(grads_sum, loss)     # free each pass's graph before the next (lazy evaluation)
            lsum += loss.item()
            seen += x.shape[1]
        grads = tree_map(lambda u: u / a.accum, grads_sum)
        grads, gnorm = optim.clip_grad_norm(grads, a.clip)
        if not np.isfinite(lsum):
            raise SystemExit('loss is not finite at step %d; stopping' % step)
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state)
        step += 1
        if step % 10 == 0:
            rate = seen / (time.time() - t0)
            print('step %d tokens %d loss %.4f %.0f tok/s' % (step, seen, lsum / a.accum, rate), flush=True)
            logs.append(dict(step=step, tokens=seen, loss=lsum / a.accum))
        if seen >= next_save or seen >= a.tokens:
            mx.save_safetensors(str(out / 'adapters.safetensors'), dict(tree_flatten(model.trainable_parameters())))
            (out / 'log.json').write_text(json.dumps(logs))
            next_save += a.save_every
    print('done', seen)


if __name__ == '__main__':
    main()
