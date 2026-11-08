#!/usr/bin/env python3
"""Native (CPU/GPU, not CRAY-2) quality pilot for a GGUF model via llama-server.

    python3 eval/run_native.py --server llama-server --model M.gguf --out build/eval/results/NAME

Needs the sets from eval/build_sets.py.  Each model is prompted through its own
chat template (llama-server --jinja, thinking disabled where the template
supports it), with greedy decoding:

  trivia_chat   "Answer the trivia question with only the answer. Do not
                explain." + question; first line of the reply
  trivia_10shot plain "Q: ...\\nA: ..." pairs with the 10 fixed examples; first line
  mmlu          "Select the correct answer. Reply with exactly one letter ..."
                scored by the highest-ranked of A-D as the first output token
                (the GPT-3 paper's method compares answer likelihoods; this is
                the single-token equivalent), and by the letter the reply starts with

TriviaQA answers use the official normalization and exact match against the
dataset's normalized aliases.
"""
import argparse
import json
import pathlib
import re
import string
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETS = ROOT / 'build' / 'eval'


def normalize_answer(s):
    """TriviaQA official normalization (triviaqa/utils/dataset_utils.py)."""
    def remove_articles(t):
        return re.sub(r'\b(a|an|the)\b', ' ', t)

    def white_space_fix(t):
        return ' '.join(t.split())

    def handle_punc(t):
        exclude = set(string.punctuation + ''.join([u'‘', u'’', u'´', u'`']))
        return ''.join(ch if ch not in exclude else ' ' for ch in t)

    def replace_underscore(t):
        return t.replace('_', ' ')
    return white_space_fix(remove_articles(handle_punc(replace_underscore(s.lower())))).strip()


class Server:
    def __init__(self, exe, model, port, extra=()):
        self.url = 'http://127.0.0.1:%d' % port
        self.p = subprocess.Popen([exe, '-m', model, '--port', str(port), '-c', '4096', '--jinja', '-np', '1',
                                   '--temp', '0', *extra], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(600):
            try:
                if json.load(urllib.request.urlopen(self.url + '/health', timeout=2)).get('status') == 'ok':
                    break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError('server did not start')

    def post(self, path, body):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        return json.load(urllib.request.urlopen(req, timeout=600))

    def chat_prompt(self, user):
        r = self.post('/apply-template', {'messages': [{'role': 'user', 'content': user}],
                                          'chat_template_kwargs': {'enable_thinking': False}})
        return r['prompt']

    def complete(self, prompt, n, probs=0):
        r = self.post('/completion', {'prompt': prompt, 'n_predict': n, 'temperature': 0, 'top_k': 1,
                                      'stop': ['\n'] if not probs else [], 'n_probs': probs, 'cache_prompt': True})
        return r

    def close(self):
        self.p.terminate()
        self.p.wait()


def first_token_letter(r):
    probs = r.get('completion_probabilities') or []
    if not probs:
        return None
    top = probs[0].get('top_logprobs') or probs[0].get('probs') or []
    best = None
    for c in top:
        tok = (c.get('token') or c.get('tok_str') or '').strip().strip('*(').upper()
        p = c.get('logprob', c.get('prob'))
        if len(tok) == 1 and tok in 'ABCD' and (best is None or p > best[1]):
            best = (tok, p)
    return best[0] if best else None


def mmlu_answer(s, mmlu, out):
    """Standard MMLU zero-shot format (Hendrycks et al.): plain prompt ending in
    "Answer:", scored by the most probable of " A", " B", " C", " D" as the next token."""
    k = 0
    with open(out / 'mmlu_answer.jsonl', 'w') as f:
        for q in mmlu:
            p = ('The following are multiple choice questions (with answers) about %s.\n\n' % q['subject'].replace('_', ' ')
                 + q['question'] + '\n' + ''.join('%s. %s\n' % (L, c) for L, c in zip('ABCD', q['choices'])) + 'Answer:')
            r = s.complete(p, 1, probs=200)
            top = r['completion_probabilities'][0].get('top_logprobs') or []
            lp = {}
            for c in top:
                t = c.get('token', '').strip()
                if t in ('A', 'B', 'C', 'D') and t not in lp:
                    lp[t] = c['logprob']
            pick = max(lp, key=lp.get) if lp else None
            k += pick == q['answer']
            f.write(json.dumps(dict(id=q['id'], answer=q['answer'], pick=pick, found=sorted(lp))) + '\n')
    return k


def strip_bos(prompt):
    for b in ('<|begin_of_text|>', '<s>'):
        if prompt.startswith(b):
            return prompt[len(b):]
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='llama-server')
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--port', type=int, default=8091)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--extra', default='', help='extra llama-server arguments')
    ap.add_argument('--only', default='', help='run only this part: mmlu_answer')
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    trivia = [json.loads(l) for l in open(SETS / 'trivia.jsonl')]
    shots = json.loads((SETS / 'trivia_shots.json').read_text())
    mmlu = [json.loads(l) for l in open(SETS / 'mmlu.jsonl')]
    if a.limit:
        trivia, mmlu = trivia[:a.limit], mmlu[:a.limit]
    s = Server(a.server, a.model, a.port, a.extra.split())
    res = {}
    try:
        if a.only == 'mmlu_answer':
            res = json.loads((out / 'summary.json').read_text()) if (out / 'summary.json').exists() else {}
            res['mmlu_answer'] = [mmlu_answer(s, mmlu, out), len(mmlu)]
            (out / 'summary.json').write_text(json.dumps(res, indent=1))
            print(json.dumps(res))
            return
        # MMLU
        k_rank = k_start = 0
        with open(out / 'mmlu.jsonl', 'w') as f:
            for q in mmlu:
                user = ('Select the correct answer. Reply with exactly one letter: A, B, C, or D. Do not explain.\n\n'
                        + q['question'] + '\n' + '\n'.join('%s. %s' % (L, c) for L, c in zip('ABCD', q['choices'])))
                r = s.complete(strip_bos(s.chat_prompt(user)), 8, probs=50)
                text = r['content']
                t = text.strip().lstrip('*( ').upper()
                start = t[0] if t[:1] in ('A', 'B', 'C', 'D') and (len(t) == 1 or not t[1].isalpha()) else None
                rank = first_token_letter(r) or start
                k_rank += rank == q['answer']
                k_start += start == q['answer']
                f.write(json.dumps(dict(id=q['id'], answer=q['answer'], rank=rank, start=start, text=text)) + '\n')
        res['mmlu_rank'] = [k_rank, len(mmlu)]
        res['mmlu_start'] = [k_start, len(mmlu)]
        res['mmlu_answer'] = [mmlu_answer(s, mmlu, out), len(mmlu)]
        # TriviaQA
        prefix = ''.join('Q: %s\nA: %s\n\n' % (x['question'], x['answer']) for x in shots)
        for proto in ('trivia_chat', 'trivia_10shot'):
            k = 0
            with open(out / (proto + '.jsonl'), 'w') as f:
                for q in trivia:
                    if proto == 'trivia_chat':
                        p = strip_bos(s.chat_prompt('Answer the trivia question with only the answer. Do not explain.'
                                                    '\n\n' + q['question']))
                    else:
                        p = prefix + 'Q: %s\nA:' % q['question']
                    text = s.complete(p, 32)['content']
                    pred = text.strip().split('\n')[0].strip()
                    em = normalize_answer(pred) in set(q['normalized_aliases'])
                    k += em
                    f.write(json.dumps(dict(id=q['id'], answer=q['answer'], prediction=pred, em=em),
                                       ensure_ascii=False) + '\n')
            res[proto] = [k, len(trivia)]
    finally:
        s.close()
    res['model'] = a.model
    (out / 'summary.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == '__main__':
    main()
