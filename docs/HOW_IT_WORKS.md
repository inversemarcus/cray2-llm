# How an 8B model fits a 1985 CRAY-2

## The machine

A 1985 CRAY-2 has four Background Processors with a 4.1 ns clock. Each has:
- eight 64-element vector registers;
- eight 64-bit scalar and eight 32-bit address registers;
- 16 Kwords of Local Memory;
- one functional unit of each kind: vector logical, vector integer (including
  population count and shifts), floating add, floating multiply.

The processors share 256 Mwords (2 GiB) of Common Memory in 128 banks. A
Foreground Processor runs the disks and I/O. Unlike the CRAY-1 and X-MP, the
CRAY-2 has **no chaining**: a vector result can't be used until the whole
vector is written. Each processor reaches each memory quadrant only on its own
phase, every fourth clock.

## Matrix-vector products with popcounts

Bonsai-8B stores each weight as one bit (sign) per weight with an fp16 scale
per group of 128. Ternary-Bonsai stores {-1, 0, +1} plus a scale.

Activations are quantized to small integers q (6 or 8 bits) and split into
**bit planes**: for each bit p, a 64-bit word holds bit p of 64 consecutive
activations. For 64 weight rows held in a vector register:

    Σ_i w_i q_i  =  Σ_p  c_p · popcount(W & P_p)

Here c_p = 2^p, and the top (two's-complement sign) plane has weight -2^p. One
vector AND (logical unit) plus one vector population count (integer unit)
covers 64 rows × 64 weights × one bit plane.

- **Binary weights** (bit = 1 means +1): Σ w q = 2·S − Σ q, with S = Σ_{w=+1} q.
- **Ternary weights** use two masks per 64 columns, POS (w = +1) and NNEG
  (w ≠ −1). Then Σ w q = S_POS + S_NNEG − Σ q.

## Exact integer sums in a floating-point adder

Each popcount is turned into an exact floating-point number by OR-ing it with
an **exponent tag**, (040060 + p) << 48. The result is an unnormalized Cray
float whose value is popcount × 2^p.

Adding tagged values to a running sum on the floating add unit is exact if the
planes are added **most significant first**. The running sum is then always a
multiple of 2^p, so nothing falls below the addend's least-significant bit
during alignment.

This spreads the work across three units: AND on the logical unit, popcount on
the integer unit, tagging alternately on logical and integer, and accumulation
on the floating adder. Per-group scales are applied once at the end of each
128-column group, on the multiply unit.

## Living with no chaining and eight vector registers

`cray2lm/sched.py` is a list scheduler. It replays the emulator's timing rules
(in-order issue, unit reservation, register write/read reservation) to choose
the instruction order and physical registers for the loop body. For the
ternary model:
- **Masks through Local Memory.** The four weight-mask vectors of a group are
  double-buffered through Local Memory: copied from Common Memory for the
  *next* group while the current group computes, then reloaded just before
  each AND. That frees vector registers for pipelining.
- **Scales decoded once per tile.** The 16-bit packed scales are decoded into
  a Local Memory table once per 64-row tile, not once per group.

The loop body is 89 words, so it stays inside the 128-word instruction buffer.

## Reproducing IEEE float32 on a non-IEEE machine

CRAY-2 floats have a 48-bit coefficient and no IEEE rounding. The native
reference rounds to float32 at specific points: RMSNorm steps, x·(1/scale),
round-half-to-even. Near a rounding boundary, a 1-ulp difference flips an
activation code.

The quantizer and RMSNorm therefore reproduce float32 rounding explicitly:
- round to 24 significant bits with vector integer and logical operations on
  the coefficient;
- round-half-even integer conversion through FIX/FLT and a tie test.

With that, every activation code matches native exactly.

## Four processors

- **Matrices split by rows.** Each processor does a quarter of the 64-row
  tiles.
- **Attention split by heads.** Each processor owns two KV heads, so its eight
  query heads need no cross-processor data.
- **Quantization split by group.** Each processor quantizes its own share of
  activation groups into a shared plane table.
- **Barriers.** Synchronization uses the documented semaphore (test-and-set
  jump, 004/005) and a counter in Common Memory: seven barriers per layer.

## Memory map (ternary model)

| Region | Mwords |
|---|---|
| Code, mailboxes, activations, plane tables, logits, RoPE table | 1.5 |
| Matrix weights: two masks per 64 columns, 16-bit scales packed four per word | 239.9 |
| KV cache: K stored transposed with an odd stride to spread banks, 200 positions | 14.8 |
| Embedding table | (on disk; one 160-word row read per token) |

## Host side

The emulator speaks a line protocol on stdin/stdout:
- `load`, `w`, `r`: memory access;
- `start`, `go`, `resume`: run control;
- `stats`: counters.

The Python host (`cray2lm/machine.py`) acts as the Foreground Processor and
never does model arithmetic. Tokenization is pure Python
(`cray2lm/tokenizer.py`).
