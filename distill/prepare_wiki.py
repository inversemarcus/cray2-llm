#!/usr/bin/env python3
"""Cut Wikipedia into fixed-length training chunks for distillation.

    uv run --with pyarrow --with tokenizers python3 distill/prepare_wiki.py WIKI_DIR TOKENIZER.json OUT_DIR [tokens]

WIKI_DIR holds the wikimedia/wikipedia 20231101.en parquet files.  Articles
are drawn with probability proportional to their length (longer articles are
mostly the notable ones), and each contributes its opening text, where an
article's key facts are.  Writes OUT_DIR/chunks-NNN.npy (uint32, [n, 513]):
513 tokens per row, so a model sees 512 positions with a next-token target.
"""
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

SEQ = 513
SEED = 1985
PER_ARTICLE = 2          # at most this many chunks from the start of an article


def main():
    wiki, tokf, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
    want = int(sys.argv[4]) if len(sys.argv) > 4 else 20_000_000
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(tokf)
    files = sorted(wiki.glob('train-*.parquet'))
    assert len(files) == 41, len(files)
    # pass 1: article lengths only
    where, lens = [], []
    for fi, f in enumerate(files):
        t = pq.read_table(f, columns=['text'])
        for ri, n in enumerate(np.array([len(x) for x in t.column('text').to_pylist()])):
            if n >= 1500:
                where.append((fi, ri)); lens.append(min(n, 60000))
        del t
        print(f.name, len(where), flush=True)
    rnd = np.random.default_rng(SEED)
    p = np.array(lens, dtype=np.float64); p /= p.sum()
    need = 2 * (want // (SEQ - 1) + 1)
    order = rnd.choice(len(where), size=min(len(where), need), replace=False, p=p)
    # pass 2: fetch the chosen articles, keeping the sampled order
    chosen = {}
    by_file = {}
    for k, i in enumerate(order):
        fi, ri = where[i]
        by_file.setdefault(fi, []).append((ri, k))
    for fi, lst in sorted(by_file.items()):
        t = pq.read_table(files[fi], columns=['title', 'text'])
        ti, tx = t.column('title'), t.column('text')
        for ri, k in lst:
            chosen[k] = (ti[ri].as_py(), tx[ri].as_py()[:12000])
        del t
    texts = [chosen[k] for k in range(len(order))]
    rows, shard, total = [], 0, 0
    for title, text in texts:
        ids = tok.encode(title + '\n\n' + text).ids
        for c in range(PER_ARTICLE):
            seg = ids[c * (SEQ - 1): c * (SEQ - 1) + SEQ]
            if len(seg) < SEQ:
                break
            rows.append(seg); total += SEQ - 1
        if len(rows) >= 4096:
            np.save(out / ('chunks-%03d.npy' % shard), np.array(rows, dtype=np.uint32)); shard += 1; rows = []
            print('shard', shard, 'tokens', total, flush=True)
        if total >= want:
            break
    if rows:
        np.save(out / ('chunks-%03d.npy' % shard), np.array(rows, dtype=np.uint32))
    print('done: %d tokens' % total)


if __name__ == '__main__':
    main()
