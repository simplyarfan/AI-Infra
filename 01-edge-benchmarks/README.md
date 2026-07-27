# Edge inference benchmarks (EdgeSpecBench)

This is the main build. It measures how small language models actually perform on
consumer Apple Silicon, and how that performance holds up under sustained use.

## Why

Almost every discussion of inference performance is about data center GPUs. But
the WISP paper Dr Rabab shared is about running models on edge devices, and edge
hardware behaves differently: far less memory bandwidth, and a hard thermal limit.
I wanted real numbers from hardware I own rather than quoting benchmarks from
blog posts.

## What it measures

For each model:
- **Time to first token**: how long before the answer starts. Dominated by prefill.
- **Decode throughput** (tokens per second): steady state generation speed.
- **Prefill throughput** (tokens per second): how fast the prompt is ingested.
- **Peak memory** (GB): what actually fits in unified memory.

And separately, a **sustained load test**: generate continuously for several
minutes and record throughput over time, to see whether the device throttles as
it heats up.

## How to run

```
pip install mlx-lm matplotlib pandas

python bench_mlx.py          # per model benchmark, writes results/mlx_results.csv
python bench_sustained.py    # sustained load test, writes results/sustained_*.csv
python plot.py               # makes the charts in results/
```

Models used are 4-bit quantized community builds that fit in 16 GB of unified
memory (Qwen2.5 at 0.5B, 1.5B, 3B). The prompt and generation length are fixed
across models so the comparison is fair.

## What I expected to see, and why

- Smaller models generate faster but the relationship is not linear, because
  decode is limited by memory bandwidth (how many bytes of weights get read per
  token), not just parameter count.
- Lower precision (4-bit here) helps decode speed for the same reason: fewer bytes
  moved per token.
- Under sustained load on a device that cannot shed heat well, throughput should
  fall over the first minutes and settle at a lower plateau. On a plugged in
  laptop with good cooling the drop may be small, which is itself a valid result.

Results and charts are in results/ once the scripts are run. See the top level
notes/hpc-notes.md for why these numbers are really a memory bandwidth story.

## Cross device comparison

Where I also run the same model family on an iPhone (using the PocketPal app,
which runs GGUF models on device), those numbers are recorded here too. One honest
caveat: the Mac uses MLX format and the phone app uses GGUF, with different
quantization and sampling, so cross app numbers are not exactly comparable. I fix
the model family and prompt and report the method rather than pretending the
setups are identical.

## Honest limits

This is single device, small model, entry tier hardware (16 GB). It is a real
measurement of the edge, not a claim about production serving. The value is in
seeing the memory bandwidth and thermal effects directly and connecting them to
the systems ideas in the notes.
