# AI-Infra

A working lab where I explore AI inference infrastructure: how large language
models are served, benchmarked, and optimised, with a focus on the edge and on
the hardware limits that decide performance.

I started this after meeting Dr Rabab and going through the DeepLearning.AI course
on efficient LLM inference with vLLM. The aim is to move from just reading about
this area to actually measuring things on real hardware and writing down what I
understand, honestly, including the limits of what I have done so far.

## What is in here

- **00-course** — my notes from the vLLM inference course, plus the certificate.
- **01-edge-benchmarks** — the main build. I benchmark small models on Apple
  Silicon with MLX, measure time to first token, decode and prefill throughput,
  and memory, and I run a sustained load test to see the device throttle as it
  heats up. This is edge inference measured on hardware I own.
- **02-prefix-caching** — a local demonstration of the idea behind SGLang's
  RadixAttention: when many requests share a long prefix, reusing that work makes
  answers come back faster. Shown with a local model.
- **notes** — my working understanding of the landscape: the engines (vLLM,
  SGLang), the cluster layer (llm-d), benchmarking tools (GuideLLM, llama-bench),
  the compiler layer (MLIR), how HPC ideas apply, a summary of the WISP paper,
  speculative decoding, and a research idea of my own.
- **weekly_logs** — notes from my working sessions across the two weeks.
- **screenshots** — course completion, the benchmark runs, and other supporting images.

## The through line

Two facts drive most of this area. Generation happens in two phases: prefill
reads the prompt and is compute bound, decode writes one token at a time and is
memory bandwidth bound. And the model keeps a running memory called the KV cache
that most optimisations are really about managing. My benchmarks measure the
consequences of these facts directly, and my notes trace how the bigger systems
(SGLang, llm-d, the WISP paper) are all different answers to the same underlying
problem.

## How this connects to high performance computing

The questions here are HPC questions: is a workload compute bound or memory bound,
where does bandwidth run out, and how far can lower precision push performance
before accuracy suffers. My sustained load test even runs into a power and thermal
wall, which is the same kind of limit that constrains large machines, just at a
phone's scale. See notes/hpc-notes.md.

## Honest status

This is early, hands on work built in an intense stretch, not a finished research
result. Where I have only read about something rather than run it (llm-d at
cluster scale, MLIR, SGLang on a GPU), I say so and keep it to a level I can
actually explain. The next step I most want to do is reproduce the core finding of
the WISP paper: that a draft model's own output statistics predict whether a
larger model will accept its tokens.
