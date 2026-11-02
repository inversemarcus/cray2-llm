# How per-token CRAY-2 time is measured

The emulator's timing model (clock-period rules, sources and assumptions) is
documented in [cray2sim's TIMING.md](../external/cray2sim/docs/TIMING.md).
This page covers what this repository adds on top of it.

## Per-token time

The time is CPU0's clock from the moment it is restarted with a token until it
exits with the selected next token. That covers:
- the embedding lookup;
- 36 transformer layers;
- all barriers;
- the 151,669-row output head;
- the four-processor argmax combine.

All four processors run concurrently in the emulator. Quoted figures come from
the exact single-threaded emulation (`run.py`, or `chat.py --exact`) without
`--ref`: the per-layer stops used for verification shift the clock slightly.

## Disk read for the ternary model's embedding row

The ternary configuration reads one 160-word embedding row per token from
disk, through the Foreground Processor. The driver adds an **assumed 25 ms**
of latency per token: seek plus rotational delay for a single record.
HR-2000 §5.3 documents the disk system (36 units, synchronous groups at
525 Mbit/s) but not seek time. The transfer itself takes microseconds.

## Sensitivity

`run.py --timing name=value,...` passes overrides to the emulator. On the
binary model at position 0:
- **Optimistic:** vector logical/integer startup 10 CP, unit gap 0: 2.18 s.
- **Pessimistic:** all vector startups 30 CP, unit gap 4, taken branch 8 CP:
  2.39 s.

The nominal figure is 2.25 s.
