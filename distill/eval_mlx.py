#!/usr/bin/env python3
"""Evaluate an MLX model, optionally with a trained correction, on the eval sets.

    python3 distill/eval_mlx.py MODEL_MLX_DIR OUT_DIR [--adapters DIR] [--trivia-set trivia_dev] [--limit N]

Same prompts, greedy decoding, stop rules and scoring as eval/run_native.py:
trivia chat (thinking disabled) and 10-shot, MMLU in the standard "Answer:"
format.  With --adapters, the correction trained by distill/train.py is
attached before evaluation.
"""
import argparse
import json
import pathlib
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.tuner.lora import LoRALinear

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'eval'))
from run_native import normalize_answer  # noqa: E402

SETS = ROOT / 'build' / 'eval'


def attach(model, adapters):
    cfg = json.loads((pathlib.Path(adapters) / 'config.json').read_text())
    lo, hi = (0, len(model.model.layers) - 1) if cfg['layers'] == 'all' else map(int, cfg['layers'].split('-'))
    for i in range(lo, hi + 1):
        mlp = model.model.layers[i].mlp
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            setattr(mlp, name, LoRALinear.from_base(getattr(mlp, name), r=cfg['rank'], scale=cfg['scale'], dropout=0.0))
    model.load_weights(str(pathlib.Path(adapters) / 'adapters.safetensors'), strict=False)


def greedy(model, tok, prompt, n=32):
    ids = tok.encode(prompt, add_special_tokens=False)
    out = []
    for (t, _), _ in zip(generate_step(mx.array(ids), model, max_tokens=n, sampler=lambda x: mx.argmax(x, -1)),
                         range(n)):
        t = int(t)
        if t in tok.eos_token_ids:
            break
        out.append(t)
        if '\n' in tok.decode(out):
            break
    return tok.decode(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('out')
    ap.add_argument('--adapters')
    ap.add_argument('--trivia-set', default='trivia_dev')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--skip-mmlu', action='store_true')
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load(a.model)
    if a.adapters:
        attach(model, a.adapters)
    trivia = [json.loads(l) for l in open(SETS / (a.trivia_set + '.jsonl'))]
    shots = json.loads((SETS / 'trivia_shots.json').read_text())
    mmlu = [json.loads(l) for l in open(SETS / 'mmlu.jsonl')]
    if a.limit:
        trivia, mmlu = trivia[:a.limit], mmlu[:a.limit]
    res = {}
    prefix = ''.join('Q: %s\nA: %s\n\n' % (x['question'], x['answer']) for x in shots)
    for proto in ('trivia_chat', 'trivia_10shot'):
        k = 0
        with open(out / (proto + '.jsonl'), 'w') as f:
            for q in trivia:
                if proto == 'trivia_chat':
                    p = tok.apply_chat_template([{'role': 'user', 'content': 'Answer the trivia question with only the '
                                                  'answer. Do not explain.\n\n' + q['question']}],
                                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
                else:
                    p = prefix + 'Q: %s\nA:' % q['question']
                pred = greedy(model, tok, p).strip().split('\n')[0].strip()
                em = normalize_answer(pred) in set(q['normalized_aliases'])
                k += em
                f.write(json.dumps(dict(id=q['id'], answer=q['answer'], prediction=pred, em=em), ensure_ascii=False)
                        + '\n')
        res[proto] = [k, len(trivia)]
        print(proto, res[proto], flush=True)
    if not a.skip_mmlu:
        letters = {L: tok.encode(' ' + L, add_special_tokens=False)[-1] for L in 'ABCD'}
        k = 0
        for q in mmlu:
            p = ('The following are multiple choice questions (with answers) about %s.\n\n' % q['subject'].replace('_', ' ')
                 + q['question'] + '\n' + ''.join('%s. %s\n' % (L, c) for L, c in zip('ABCD', q['choices']))
                 + 'Answer:')
            logits = model(mx.array([tok.encode(p, add_special_tokens=False)]))[0, -1]
            pick = max('ABCD', key=lambda L: logits[letters[L]].item())
            k += pick == q['answer']
        res['mmlu_answer'] = [k, len(mmlu)]
        print('mmlu', res['mmlu_answer'], flush=True)
    (out / 'summary.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
