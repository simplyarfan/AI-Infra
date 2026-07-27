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

## Cross device comparison: Mac vs iPhone

I ran a model on my iPhone as well, using the PocketPal app (which runs GGUF
models fully on device via llama.cpp), to get a real second tier of edge hardware
rather than only benchmarking the laptop.

Model: Llama-3.2-1B-Instruct, 4-bit (Q4_0), 773 MB, running offline on the phone.

iPhone results:
- Decode throughput: about 77 tokens per second when the phone was cool.
- Time to first token: roughly 210 to 262 ms.

For context, my Mac got about 96 tokens per second on the larger 1.5B model. So
for tiny models the phone lands in the same rough range as the laptop, which is a
real finding on its own: modern phone silicon is surprisingly capable at small
model inference. The gap would widen quickly as models get bigger, because the
phone has far less memory bandwidth, and decode is memory bandwidth bound.

One honest caveat on comparability: the Mac uses the MLX format and the phone app
uses GGUF, with different quantization details and sampling, so the cross device
numbers are not an exact apples to apples race. I fix the model family and the
prompt and report the method rather than pretending the setups are identical.

## The thermal result (the interesting part)

This is the observation that motivates my research idea (see
notes/research-idea.md). I ran the phone continuously with back to back prompts
for a few minutes and watched the decode throughput:

- Start (cool phone): 77.15, then 77.39, then 77.93 tokens per second. Steady.
- As it heated up: 73.37 tokens per second.
- Further in: 69.14 tokens per second.

That is roughly a 10 percent drop, and I could physically feel it happening: the
phone got noticeably hot in my hand over the run, and the slowdown tracked the
heat. This is thermal throttling, measured directly on my own device. The Mac, by
contrast, held flat over five minutes (see the sustained chart) because it has
active cooling. Same test, two devices, opposite behaviour, and the difference is
entirely about the ability to shed heat.

A 1B model on a phone was only heavy enough to produce about a 10 percent drop in
a few minutes. A larger model or a longer session would almost certainly throttle
harder. The documented numbers in the literature show much steeper drops on phones
under heavier sustained load, so my measurement is a mild, honest version of a
real and larger effect.

I also hit the model's context window limit during the long run (the app warned it
was out of room and offered to grow the context). That is the KV cache filling up
in practice: as the conversation grows, the cache grows with it until it runs out
of allocated space. So in one session I ran into two real edge constraints at
once, the thermal wall and the memory limit.

Screenshots of the phone runs are in results/.

## Honest limits

This is single device, small model, entry tier hardware (16 GB). It is a real
measurement of the edge, not a claim about production serving. The value is in
seeing the memory bandwidth and thermal effects directly and connecting them to
the systems ideas in the notes.
