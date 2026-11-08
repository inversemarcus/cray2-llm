#!/usr/bin/env python3
"""Build the frozen evaluation sets used in docs/RESULTS.md.

    uv run --with pyarrow python3 eval/build_sets.py

Downloads the source data files at pinned revisions into build/eval/src/,
selects with a fixed seed, and writes build/eval/{trivia,mmlu}.jsonl and
trivia_shots.json.  The selected IDs are recorded in eval/sets/*.ids.json and
checked on every rebuild.  Question text is not stored in this repository.

  * TriviaQA (mandarjoshi/trivia_qa) unfiltered.nocontext validation: 200
    questions, plus 10 training questions as the fixed few-shot examples.
  * MMLU (cais/mmlu) test: 5 questions from each of the 57 subjects (285).
"""
import json
import pathlib
import random
import urllib.request

import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent.parent
SETS = ROOT / 'eval' / 'sets'
OUT = ROOT / 'build' / 'eval'
SEED = 1985
SRC = {
    'trivia_val': ('mandarjoshi/trivia_qa', '0f7faf33a3908546c6fd5b73a660e0f8ff173c2f',
                   'unfiltered.nocontext/validation-00000-of-00001.parquet'),
    'trivia_train': ('mandarjoshi/trivia_qa', '0f7faf33a3908546c6fd5b73a660e0f8ff173c2f',
                     'unfiltered.nocontext/train-00000-of-00001.parquet'),
    'mmlu_test': ('cais/mmlu', 'c30699e8356da336a370243923dbaf21066bb9fe', 'all/test-00000-of-00001.parquet'),
}


def table(key):
    repo, rev, path = SRC[key]
    f = OUT / 'src' / (key + '.parquet')
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://huggingface.co/datasets/%s/resolve/%s/%s' % (repo, rev, path)
        urllib.request.urlretrieve(url, f)
    return pq.read_table(f).to_pylist()


def main():
    SETS.mkdir(parents=True, exist_ok=True)
    val, train, mmlu = table('trivia_val'), table('trivia_train'), table('mmlu_test')
    rnd = random.Random(SEED)
    q_idx = sorted(rnd.sample(range(len(val)), 200))
    s_idx = rnd.sample(range(len(train)), 10)
    by_subject = {}
    for i, r in enumerate(mmlu):
        by_subject.setdefault(r['subject'], []).append(i)
    assert len(by_subject) == 57
    m_idx = {s: sorted(rnd.sample(range(len(v)), 5)) for s, v in sorted(by_subject.items())}

    ids = dict(trivia=[val[i]['question_id'] for i in q_idx], shots=[train[i]['question_id'] for i in s_idx],
               mmlu=['%s:%d' % (s, k) for s, ks in m_idx.items() for k in ks],
               sources={k: dict(repo=v[0], revision=v[1], file=v[2]) for k, v in SRC.items()}, seed=SEED)
    f = SETS / 'ids.json'
    if f.exists():
        old = json.loads(f.read_text())
        assert old == ids, 'selection differs from the recorded eval/sets/ids.json'
    else:
        f.write_text(json.dumps(ids, indent=1))

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'trivia.jsonl', 'w') as o:
        for i in q_idx:
            r = val[i]
            o.write(json.dumps(dict(id=r['question_id'], question=r['question'], answer=r['answer']['value'],
                                    normalized_aliases=r['answer']['normalized_aliases']), ensure_ascii=False) + '\n')
    shots = [dict(id=train[i]['question_id'], question=train[i]['question'], answer=train[i]['answer']['value'])
             for i in s_idx]
    (OUT / 'trivia_shots.json').write_text(json.dumps(shots, ensure_ascii=False, indent=1))
    with open(OUT / 'mmlu.jsonl', 'w') as o:
        for s, ks in m_idx.items():
            for k in ks:
                r = mmlu[by_subject[s][k]]
                o.write(json.dumps(dict(id='%s:%d' % (s, k), subject=s, question=r['question'], choices=r['choices'],
                                        answer='ABCD'[r['answer']]), ensure_ascii=False) + '\n')
    print('trivia 200 (+10 shots), mmlu %d -> %s' % (sum(len(v) for v in m_idx.values()), OUT))


if __name__ == '__main__':
    main()
