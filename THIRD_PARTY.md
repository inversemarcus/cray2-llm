# Third-party material

This repository's own code is under the MIT license in `LICENSE`. It contains
or refers to the following material from other projects. Their notices are
reproduced in full below.

| Item | Origin | License |
|---|---|---|
| `cray2lm/tokenizer.py`, `_bytes_to_unicode` | Byte-to-Unicode table from OpenAI's GPT-2 `src/encoder.py` (openai/gpt-2 at 9b63575), reimplemented | Modified MIT License, Copyright (c) 2019 OpenAI |
| `tools/q8k_6bit.patch` | Diff against `ggml/src/ggml-quants.c` in PrismML-Eng/llama.cpp at a153697; the context lines are llama.cpp code | MIT, Copyright (c) 2023-2026 The ggml authors |
| `tools/llama_tokenizer.cpp`, `tools/native_capture.cpp` | Written for this project; they link against llama.cpp (same revision) and use its public API | llama.cpp: MIT, as above |
| `external/cray2sim` | The emulator, a separate repository | MIT |

Not included and not redistributed:

- **Model weights.** Bonsai-8B and Ternary-Bonsai-8B are published by
  PrismML under Apache-2.0. `build.py` reads a GGUF file that you download
  yourself. Their license terms apply to the weights and to the images built
  from them.
- **llama.cpp.** The reference tools are built against PrismML-Eng/llama.cpp
  at commit a15369706017f177cae8aa40b6b8592259a48806, which supports the Q1_0
  and PQ2_0 tensor types.
- **Evaluation data.** TriviaQA, MMLU, PIQA and ARC-Challenge questions used
  for the pilot results in `docs/RESULTS.md` are not included.

## openai/gpt-2 (LICENSE at 9b63575)

```
Modified MIT License

Software Copyright (c) 2019 OpenAI

We don’t claim ownership of the content you create with GPT-2, so it is yours to do with as you please.
We only ask that you use GPT-2 responsibly and clearly indicate your content was created using GPT-2.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
The above copyright notice and this permission notice need not be included
with content created by the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
```

## PrismML-Eng/llama.cpp (LICENSE at a153697)

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
