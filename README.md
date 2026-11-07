# cray2-llm: an 8B language model on an emulated 1985 CRAY-2

This runs a modern 8-billion-parameter language model on a four-processor 1985
CRAY-2, as CRAY-2 machine code, inside an instruction-level emulator with a
clock-period timing model. The question it answers: *could the fastest computer
of 1985 have run a model with GPT-3-class abilities at a usable speed?*

```
$ python3 chat.py "Who painted the Mona Lisa?"
The Mona Lisa was painted by **Leonardo da Vinci**, an Italian Renaissance artist. ...
[37 forward passes on 4 CRAY-2 processors: 150.8 s of CRAY-2 time (4.06-4.09 s per token)]
```

(`chat.py` uses the fast multithreaded emulation, whose clock reads about 1%
low for this model. Quoted timings below are from the exact emulation.)

**Short answer:** partly. Ternary-Bonsai-8B runs in the machine's 2 GiB at
**about 4.1 s per token** of emulated CRAY-2 time. Its capability evidence is
mixed:
- **Small native pilots** use the same 6-bit arithmetic (32–64 questions per
  test). On those, it is roughly at GPT-3 (175B)'s published level on MMLU,
  PIQA and ARC-Challenge.
- **Vendor scores** are published for the stock model. On math and code they
  are far above anything reported for GPT-3.
- **Memorized trivia is clearly behind:** 48% vs GPT-3's 71% few-shot.

None of this is a full benchmark run, and the emulated machine itself was only
checked on individual prompts. See [docs/RESULTS.md](docs/RESULTS.md).

## What is here

| Path | Contents |
|---|---|
| `external/cray2sim/` | Git submodule: **[cray2sim](https://github.com/inversemarcus/cray2sim)**, the CRAY-2 emulator (four processors, every 1985 instruction, CRAY-2 floating point, clock-period timing model), the CAL Version 2 assembler and the cited instruction-set data |
| `cray2lm/gen_*.py` | Generators that write the whole model as CAL: binary/ternary bit-plane matrix kernels, RMSNorm, activation quantization, RoPE, attention, SwiGLU, output head and argmax, split across four processors with semaphore barriers. |
| `cray2lm/sched.py` | Instruction scheduler for the matrix kernel. It only reorders instructions and renames registers; it never changes the arithmetic. |
| `cray2lm/layout.py`, `tools/convert_*.c` | Convert a GGUF model into a CRAY-2 memory image (and a disk image for the embedding table). |
| `cray2lm/tokenizer.py`, `cray2lm/gguf.py` | Pure-Python GGUF reader and Qwen tokenizer, checked identical to llama.cpp on 731 strings: the 417 built into `tests/test_tokenizer.py` plus our evaluation prompts. |
| `cray2lm/machine.py` | The host side, playing the role of the CRAY-2 Foreground Processor: loads memory, feeds tokens, reads results. |
| `chat.py`, `run.py`, `build.py` | Ask questions; timing probes and verification; build steps. |
| `tests/test_argmax.py` | The generated token-selection code, run with tied logits |
| `tests/test_model.py` | End-to-end check of a built model: selected token, exact clock count, context limit |
| `tests/test_tokenizer.py` | Tokenizer comparison against llama.cpp |
| `tools/native_capture.cpp`, `tools/q8k_6bit.patch` | For reproducing the fidelity checks against llama.cpp. |

Everything is C and the Python standard library; there are no packages to
install.

## Quick start

You need a C compiler, Python 3.9+, about 2.5 GB of RAM and about 5 GB of disk.

```sh
git clone --recurse-submodules https://github.com/inversemarcus/cray2-llm.git
cd cray2-llm
make && make test                                   # build and self-test the emulator

# Download the model from PrismML (Apache-2.0):
#   https://huggingface.co/prism-ml/Ternary-Bonsai-8B-gguf  ->  Ternary-Bonsai-8B-PQ2_0.gguf
mkdir -p models && mv ~/Downloads/Ternary-Bonsai-8B-PQ2_0.gguf models/

python3 build.py ternary6 --gguf models/Ternary-Bonsai-8B-PQ2_0.gguf
python3 chat.py "Which river is the longest in Europe?"
python3 tests/test_model.py --model ternary6        # end-to-end check (about a minute)
```

`build.py` converts the weights once (under a minute), reads the tokenizer
vocabulary, then generates and assembles the CAL program. `chat.py` runs in
**fast mode** by default: one host thread per CRAY-2 processor. In our tests
fast mode gave bit-identical results. Its CRAY-2 clock reads low, by about 1% for the
ternary models and 6% for the 1-bit model,
because cross-processor memory-bank conflicts are only seen at
2,000-clock-period boundaries. `--exact` (and `run.py` by default) uses the single-threaded
reference emulation behind every timing quoted here.

Host speed for the default model: on an Apple M4 Pro one token takes about
2.4 s in fast mode and 10 s in exact mode, against the 1985 machine's 4.1 s. A
2013 Xeon (E5-1650 v2) takes about 20 s and 90 s.

### Models

| `--model` | Weights | Activations | Context | CRAY-2 s/token |
|---|---|---|---|---|
| `ternary6` (default) | Ternary-Bonsai-8B, {-1,0,+1} | 6-bit, blocks of 256 | 200 | **4.10** |
| `ternary8` | Ternary-Bonsai-8B | 8-bit (native Q8_K) | 200 | 5.01 |
| `binary` | Bonsai-8B (1-bit), [prism-ml/Bonsai-8B-gguf](https://huggingface.co/prism-ml/Bonsai-8B-gguf) | 6-bit, groups of 128 | 1024 | 2.21 |

Context is limited by memory. The ternary weights take 241 of the 256 Mwords
(2 GiB), and the attention cache gets the rest. The ternary embedding table
(194 MB) lives on disk, and the Foreground Processor reads one row per token,
as CRAY-2 disk I/O.

## How faithful is it?

- **The emulator** follows the 1985 manual instruction by instruction. No
  CRAY-2 survives in working order to check it against, so its fidelity is to
  the documentation. The reciprocal and square-root approximation tables were
  never published; they are modelled, and the model only uses them followed by
  the documented refinement step.
- **Timing** combines documented mechanisms with sourced figures: 4.1 ns
  clock, 2-clock issue, no chaining, 57-clock bank busy, phased quadrant
  access, 22-clock floating-point vector startup. Some values are explicit
  assumptions: logical/integer vector startup, unit turnaround, branch cost,
  and the 25 ms disk read. Sensitivity runs move per-token time by about ±5%.
  See the emulator's
  [docs/TIMING.md](external/cray2sim/docs/TIMING.md) and this repository's
  [docs/MEASUREMENT.md](docs/MEASUREMENT.md).
- **Numerics** are checked against llama.cpp.
  - The ternary model's RMSNorm and activation quantizer reproduce IEEE-F32
    rounding with CRAY-2 integer and logical operations, so every quantized
    activation code matches native.
  - The 6-bit ternary model is bit-identical to its native reference through
    layer 20 and selects the same tokens.
  - An 8-bit generation test produced exactly the native answer, token for
    token.
- **Era authenticity.** Only resources a 1985 CRAY-2 had are used: 2 GiB of
  Common Memory, Local Memory, the documented instruction set, and disk I/O
  through the Foreground Processor. The code is generated CAL rather than
  1985 CFT2 Fortran.

## How it works

[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) explains the kernels:
- bit-plane popcounts for 1-bit and ternary weights;
- exact integer accumulation through exponent-tagged floating adds;
- IEEE-F32 rounding emulated on a non-IEEE machine;
- Local Memory double-buffering;
- the four-processor split.

## License

MIT, see [LICENSE](LICENSE). The model weights are not included: download them
from PrismML under their licenses. The CRAY-2 manuals are Cray Research
documents; cray2sim cites them and does not redistribute them. Third-party
material is listed in [THIRD_PARTY.md](THIRD_PARTY.md).
