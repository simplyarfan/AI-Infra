# Week 2 notes

Second week, again organised by session rather than by day. This week was more
building than reading. Notes kept locally and pushed together.

## Session: HPC framing

Before building, I wrote down the HPC lens properly in hpc-notes.md, because it is
the thread that ties the whole thing to Dr Rabab's field. The key ideas: compute
bound versus memory bound, the roofline model, and why decode sits under the
memory bandwidth roof. This gave the benchmarks a point beyond just collecting
numbers, which is to show the bandwidth wall directly.

## Session: building the MLX benchmark

Decided to use Apple's MLX because vLLM and SGLang do not run on a Mac, and MLX is
the fast native path on Apple Silicon and uses the unified memory well. Installed
mlx-lm. Wrote bench_mlx.py to measure time to first token, decode throughput,
prefill throughput, and peak memory, with a warmup and several measured runs
averaged. Chose small 4-bit models (Qwen2.5 at 0.5B, 1.5B, 3B) that fit in 16 GB.

What I was checking: whether decode speed scales with model size the way memory
bandwidth predicts, rather than linearly with parameter count.

Tool: MLX and mlx-lm (https://github.com/ml-explore/mlx).

## Session: the sustained load and thermal test

Wrote bench_sustained.py to hammer the device for several minutes and record
throughput over time. This is the part I care about most, because sustained edge
inference runs into a thermal wall, and that connects to my research idea. The
plan was to run it on the Mac and, where possible, compare against the phone.

Read a 2026 measurement paper on mobile inference under sustained load that
documents a large throughput drop on an iPhone as it heats and throttles, which is
what motivated running my own version.

## Session: the prefix caching demonstration

Built prefix_cache_demo.py to show SGLang's core idea locally. It sends a batch of
questions that either share one long prefix (warm) or each get a unique prefix
(cold), using a local model through Ollama, and compares time to first token. The
point is to make prefix caching concrete and prove I understand when it helps and
when it does not. Wrote the project README explaining RadixAttention and how it
differs from vLLM's block hashing.

## Session: writing the research idea

Pulled the thread from week 1 together into research-idea.md: thermal aware
speculative decoding. The gap is that adaptive drafting methods react to content
or network, not to the device's thermal state, even though I can measure that
throttling directly. Found the closest existing work (PELM, which tunes power and
frequency with speculative decoding) and wrote down how my angle differs, so I do
not overclaim originality.

## Session: pulling the repo together

Wrote the top level README and the project READMEs, organised the notes, and put
the charts with the results. Made sure every file says honestly what was built
versus what was only read about.

## Where I ended

A repo that has real benchmarks on my own hardware, a working demonstration of the
prefix caching principle, a clear map of the landscape, and an original research
direction that came out of my own measurements. Next step I want to take is the
WISP reproduction: log the draft model's output statistics against acceptance and
train a small predictor.
