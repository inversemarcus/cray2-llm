#!/usr/bin/env python3
"""Diagnostic: for trivia questions a teacher answers correctly and Bonsai
does not, how much probability does Bonsai give the correct answer?

    python3 distill/diag_gap.py MODEL_MLX TEACHER_10SHOT.jsonl STUDENT_10SHOT.jsonl [--adapters DIR]

Scores the gold answer (the dataset's canonical answer) under the 10-shot
prompt by teacher forcing, and reports per-token average log-probability and
the rank of the first answer token, split into gap questions (teacher right,
student wrong), both-right and both-wrong.
"""
import argparse
import json
import pathlib
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'distill'))
from eval_mlx import attach  # noqa: E402

SETS = ROOT / 'build' / 'eval'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model'); ap.add_argument('teacher'); ap.add_argument('student')
    ap.add_argument('--adapters')
    a = ap.parse_args()
    model, tok = load(a.model)
    if a.adapters:
        attach(model, a.adapters)
    q = {json.loads(l)['id']: json.loads(l) for l in open(SETS / 'trivia.jsonl')}
    t = {json.loads(l)['id']: json.loads(l)['em'] for l in open(a.teacher)}
    s = {json.loads(l)['id']: json.loads(l)['em'] for l in open(a.student)}
    shots = json.loads((SETS / 'trivia_shots.json').read_text())
    prefix = ''.join('Q: %s\nA: %s\n\n' % (x['question'], x['answer']) for x in shots)
    groups = {'gap (teacher right, student wrong)': [], 'both right': [], 'both wrong': []}
    for i, r in q.items():
        g = ('gap (teacher right, student wrong)' if t[i] and not s[i] else 'both right' if t[i] and s[i]
             else 'both wrong' if not t[i] and not s[i] else None)
        if g is None:
            continue
        p = tok.encode(prefix + 'Q: %s\nA:' % r['question'], add_special_tokens=False)
        ans = tok.encode(' ' + r['answer'], add_special_tokens=False)
        logits = model(mx.array([p + ans]))[0].astype(mx.float32)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        lp = [logp[len(p) - 1 + k, ans[k]].item() for k in range(len(ans))]
        first_rank = int((logp[len(p) - 1] > logp[len(p) - 1, ans[0]]).sum().item())
        groups[g].append((float(np.mean(lp)), lp[0], first_rank))
    for g, v in groups.items():
        if not v:
            continue
        m = np.array(v)
        print('%-36s n=%3d  mean token logp %.2f  first-token p %.3f  median first-token rank %d  rank<=5: %.0f%%'
              % (g, len(v), m[:, 0].mean(), np.exp(m[:, 1]).mean(), np.median(m[:, 2]), 100 * (m[:, 2] <= 4).mean()))


if __name__ == '__main__':
    main()
