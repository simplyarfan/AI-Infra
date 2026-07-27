# vLLM inference course notes

Course: Fast and Efficient LLM Inference with vLLM (DeepLearning.AI, short course).
Completed. Certificate in this folder.

These are my notes in my own words. The scanned handwritten version is also in
this folder.

## LLM optimization fundamentals

The reason inference is hard is the two phase nature of generation. Prefill reads
the whole prompt in one parallel pass and is limited by compute. Decode produces
one token at a time, each depending on the last, and is limited by memory
bandwidth because every token reads the weights and the growing cache back out of
memory. Almost every optimization targets one of these two phases.

The KV cache is the running memory of everything processed so far, kept so the
model does not recompute past tokens for every new one. It saves huge amounts of
work but takes memory, and managing it well is most of the battle.

## Optimizing a model with a compressor (quantization)

Quantization stores the model's numbers using fewer bits (for example 4-bit
instead of 16-bit). Smaller numbers mean fewer bytes moved per token, which speeds
up the memory bound decode phase and shrinks memory use so larger models fit. The
cost is a small accuracy loss, usually acceptable when done carefully. This is a
practical, everyday lever.

## Serving LLMs efficiently with vLLM

Two ideas make vLLM fast:
- PagedAttention: the KV cache is stored in small fixed size blocks handed out on
  demand, like operating system memory pages, so almost no memory is wasted on
  padding and many more requests fit at once.
- Continuous batching: instead of waiting for a whole group of requests to finish
  together, the moment one finishes a waiting request takes its slot, so the GPU
  is never idle waiting for the slowest one.

The practical knobs I noted: how much memory to give the cache pool, the maximum
tokens per batch, the maximum context length, and how many GPUs to split a model
across.

## Measuring what matters: benchmarking

The metrics that matter are time to first token, decode tokens per second, prefill
tokens per second, throughput under concurrency, and peak memory. Good method
means warming up before measuring, running several times and reporting spread,
using unique prompts so prefix caching does not silently inflate results, and
recording exact settings. I carried this straight into my own edge benchmarks.

## Putting it together

The takeaways I am carrying forward: think in terms of the two phases, know that
decode is a memory bandwidth problem, treat the KV cache as the central resource,
and always benchmark honestly rather than trusting a single run.
