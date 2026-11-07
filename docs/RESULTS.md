# Results

All CRAY-2 times are emulated clock periods × 4.1 ns from the exact
(single-threaded) emulation, at context position 0 unless stated.

## Speed

| Model | CRAY-2 s/token | Notes |
|---|---|---|
| Ternary-Bonsai-8B, 6-bit activations (default) | **4.10** | includes an assumed 25 ms disk read for the embedding row |
| Ternary-Bonsai-8B, 8-bit (native) activations | 5.01 | |
| Bonsai-8B (1-bit), 6-bit activations | 2.21 | 2.30 s at position 127, 2.59 s at 511, 2.98 s at 1023 |

Timing sensitivity, binary model at position 0:
- **Optimistic:** vector logical/integer startup 10 CP, unit gap 0: 2.09 s.
- **Pessimistic:** all vector startups 30 CP, unit gap 4, taken branch 8 CP: 2.32 s.

Where the time goes: about 98% is the bit-plane matrix kernels, at the same
cost per weight for every matrix including the output head. Norms,
quantization and attention take under 2% at short context.

Why not faster: the CRAY-2 has no chaining and eight vector registers, and a
fully pipelined bit-plane step needs about nine. With the masks staged through
Local Memory, the logical and integer units run at about 70% of their limit.

## Fidelity (emulated CRAY-2 vs llama.cpp on a CPU, individual prompts)

These were measured during development. The native captures and comparison
outputs are not in the repository; `run.py --ref` with a capture from
`tools/native_capture.cpp` repeats the per-layer comparison, and
`tests/test_model.py` checks the selected token for each model.

| Check | Result |
|---|---|
| Ternary 6-bit, token 198 | Hidden state bit-identical to the native 6-bit reference through layer 20; final layer 1.4% rel. L2 (cos 0.99991); same selected token |
| Ternary 8-bit, token 198 | Layers 0–5 bit-identical; final layer 0.44% rel. L2 (cos 0.99999); same token |
| Ternary 8-bit, generation | TriviaQA "longest river in Europe" prompt (36 tokens): generated tokens identical to native ("Volga River" + end of turn), 41 passes |
| Binary, generation | TriviaQA prompt (36 tokens): generated tokens identical to native ("Antoine Lavoisier" + end of turn), 43 passes |
| Activation codes | All 4,096 quantized codes and 32 scales of a full activation vector match native exactly |

We attribute the remaining drift in late layers to arithmetic differences,
not demonstrated in detail: the CRAY-2 accumulates in 48-bit floating point
in its own order, native in float32. It does not change
the tokens in the checks above.

## Capability (native runs, small pilot samples)

Same frozen questions, prompts and scorers for every model. The 6-bit variant
is the `ternary6` configuration. These are historical measurements: the pilot
harness, question lists and outputs are not included in this repository, so
the table cannot be reproduced from it alone.

| Test | Binary 8B | Ternary 8B (8-bit) | Ternary 8B (6-bit) | GPT-3 175B (published) |
|---|---|---|---|---|
| TriviaQA, raw prompt, 0-shot (64) | 7 | 14 | 13 | |
| TriviaQA, chat, 0-shot (64) | 21 (33%) | 29 (45%) | 28 (44%) | 64.3% zero-shot |
| TriviaQA, 10-shot (64) | 27 (42%) | 30 (47%) | **31 (48%)** | 71.2% few-shot (up to 64 examples) |
| MMLU (57), letter scoring | 39 (68%) | 37 (65%) | **37 (65%)** | 43.9% (few-shot) |
| MMLU (57), strict: reply must be the letter alone | 34 | 29 | 28 | |
| PIQA (32) | 25 | 26 (81%) | 26 (81%) | 81.0% |
| ARC-Challenge (32) | 15 | 15 (47%) | 15 (47%) | 51.4% |

Vendor-published scores for the stock 1-bit Bonsai-8B (EvalScope, not our
6-bit configuration, and not run on the emulator) are MMLU-Redux 65.7, GSM8K
88, HumanEval+ 73.8 and IFEval 79.8. HumanEval itself was introduced in the
Codex paper (Chen et al. 2021, arXiv:2107.03374), which reports GPT-3 solving
0% of its problems.

Reading these honestly:
- **MMLU is above GPT-3 and PIQA/ARC are at its level** in our small native
  pilots of the same 6-bit arithmetic. The default model's 65% on MMLU has a
  95% interval of 52–76%, above GPT-3's 43.9%, and is in line with the
  vendor's MMLU-Redux 65.7 for the stock model. **Math and code** are far
  above GPT-3 in the vendor's numbers for the stock model, which we have not
  reproduced.
- **Memorized facts are clearly below.** With 10-shot prompting the default
  model gets 48% against GPT-3's 71%, and the upper end of the 95% interval is
  60%. Our hypothesis, not a demonstrated cause, is limited weight capacity:
  knowledge capacity scales with stored parameters, and GPT-3 has 175 billion.
- **The 10-shot prompts were run natively only.** They are 320–358 tokens,
  longer than the ternary configuration's 200-position context on the
  emulated machine.
- **Small samples.** 32–64 questions per test, which are pilots rather than
  full benchmark runs. Protocols and question samples differ from the GPT-3
  paper.
- **MMLU scoring.** The GPT-3 paper scored multiple choice by comparing the
  model's likelihood of each answer, so formatting cannot cost points. Our
  matching "letter scoring" takes whichever of A–D the model ranks highest
  as its first output token. Reading the letter at the start of the reply,
  such as "C. True, False", gives nearly the same counts (39, 36, 36). The strict row accepts
  only a reply that is the letter alone, which rejected even "D."; we show
  it for completeness.
