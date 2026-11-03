# Third-party material

This repository's own code is under the MIT license in `LICENSE`. It contains
or refers to the following material from other projects.

| Item | Origin | License |
|---|---|---|
| `cray2lm/tokenizer.py`, `_bytes_to_unicode` | Byte-to-Unicode table from OpenAI's GPT-2 encoder (`encoder.py`), reimplemented | MIT, Copyright (c) 2019 OpenAI |
| `tools/q8k_6bit.patch` | Diff against `ggml/src/ggml-quants.c` in llama.cpp; the context lines are llama.cpp code | MIT, Copyright (c) 2023-2024 The ggml authors |
| `tools/llama_tokenizer.cpp`, `tools/native_capture.cpp` | Written for this project; they link against llama.cpp and use its public API | llama.cpp: MIT, Copyright (c) 2023-2024 The ggml authors |
| `external/cray2sim` | The emulator, a separate repository | MIT |

Not included and not redistributed:

- **Model weights.** Bonsai-8B and Ternary-Bonsai-8B are published by
  PrismML. `build.py` reads a GGUF file that you download yourself. Their
  license terms apply to the weights and to the images built from them.
- **llama.cpp.** The reference tools are built against PrismML-Eng/llama.cpp
  at commit a153697, which supports the Q1_0 and PQ2_0 tensor types.
- **Evaluation data.** TriviaQA, MMLU, PIQA and ARC-Challenge questions used
  for the pilot results in `docs/RESULTS.md` are not included.
