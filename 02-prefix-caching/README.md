# Prefix caching: SGLang's core idea, shown locally

SGLang's headline feature is RadixAttention, which is automatic prefix caching.
This small project explains what that is and demonstrates the underlying effect on
a local machine, without a GPU and without SGLang itself.

## The idea

When you run a language model, processing the prompt (prefill) produces a key value
cache for every token. Normally that work is thrown away after the request. But in
the real world many requests share the same opening: a long system prompt, a fixed
set of few shot examples, a document that gets asked about repeatedly, or the
growing history in a multi turn chat.

Prefix caching keeps the cached work for that shared opening and reuses it. SGLang
does this automatically by storing the cache in a radix tree (a trie where an edge
can stand for a run of tokens). A new request is matched against the tree for the
longest shared prefix, and only the new part gets fresh work. It adds cache aware
scheduling (group requests that share prefixes) and evicts least recently used
branches first so shared parents survive.

The payoff is large on workloads with heavy prefix overlap and roughly nothing on
fully unique prompts, which makes sense: there is only a benefit when there is
shared work to reuse.

## How this differs from vLLM

vLLM also caches prefixes, but by hashing fixed size blocks and reusing them when
the hashes line up. SGLang's radix tree matches at the token level to any depth,
which handles branching and growing histories more naturally. Both are solving the
same problem; the difference is the matching granularity.

## The local demonstration

I cannot run SGLang on a Mac (it targets NVIDIA GPUs), but I can show the same
principle. Using a local model served by Ollama, which does prompt caching, I send
a batch of questions two ways:

- Warm: every question uses the exact same long shared prefix, so after the first
  one the prefix work can be reused.
- Cold: every question gets a slightly different long prefix, so there is nothing
  to reuse.

Then I compare the time to first token. If prefix reuse is doing what SGLang's
design promises, the warm case should reach the first token faster.

## How to run

```
brew install ollama
ollama serve                 # in one terminal
ollama pull qwen2.5:1.5b     # in another
pip install requests

python prefix_cache_demo.py  # writes results/prefix_cache_results.csv
```

## Honest limits

This shows the principle, not SGLang's actual radix tree implementation, and a
local single stream setup will not show the full gains that appear under real
concurrent load on a server. It is meant to make the concept concrete and to prove
I understand why prefix caching helps and when it does not.
