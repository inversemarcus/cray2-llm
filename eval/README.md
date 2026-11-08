# Reproducible quality pilot

Native runs (llama.cpp on a CPU/GPU, not the CRAY-2) on a fixed sample:
200 TriviaQA questions (unfiltered, validation) and 285 MMLU test questions
(5 per subject), selected with a fixed seed from pinned dataset revisions.

```sh
uv run --with pyarrow python3 eval/build_sets.py        # downloads and builds build/eval/*.jsonl
python3 eval/run_native.py --server PATH/llama-server --model M.gguf --out build/eval/results/NAME
```

`eval/sets/ids.json` records the selected question IDs and source revisions;
the builder checks them on every run. Question text is not stored here.
`eval/results/` holds the summaries quoted in `docs/RESULTS.md`.

Bonsai models need the PrismML llama.cpp fork (PrismML-Eng/llama.cpp at
a153697) for their Q1_0 and PQ2_0 tensor types; other GGUF files work with
mainline llama.cpp.
