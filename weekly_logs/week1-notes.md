# Week 1 notes

These are notes from my working sessions in the first week. I did not work every
day (some days went to work and family), and I studied more than I built early on,
so this is organised by session rather than one entry per day. I kept my notes
locally and pushed everything to GitHub together rather than committing each day.

## Session: getting oriented

Started the DeepLearning.AI course on efficient LLM inference with vLLM. Watched
the fundamentals and the first serving section. The thing that clicked was the two
phase model: prefill reads the whole prompt at once and is compute bound, decode
writes one token at a time and is memory bandwidth bound. Wrote the first version
of my course notes.

Also read up on what vLLM actually does differently, so the course had context:
PagedAttention for memory and continuous batching for keeping the GPU busy.

Papers and links from this session:
- Course: https://www.deeplearning.ai/short-courses/ (Fast and Efficient LLM
  Inference with vLLM)
- vLLM PagedAttention paper (Kwon et al., 2023): https://arxiv.org/abs/2309.06180

## Session: the paper Dr Rabab sent

Read the WISP paper properly. Took me a couple of passes. The core that I actually
understood and could explain: it splits speculative decoding across an edge device
(small draft model) and a cloud server (big target model), and it names two
specific wastes, wasted drafting time and verification interference, then fixes
each. Wrote wisp-summary.md in my own words. The part I flagged as reproducible at
small scale is their observation that the draft model's own output statistics
predict whether the target will accept a token.

Papers and links:
- WISP (Li et al., 2026). Read sections 1, 2.4, 3, and the evaluation.
- Speculative decoding origin (Leviathan et al., 2023):
  https://arxiv.org/abs/2211.17192

## Session: reading around speculative decoding

Went wider than the one paper to understand the area. Found that choosing how many
tokens to draft is its own active subfield: DISCO, AdaEDL, SpecDec++, GammaTune,
EAGLE and AdaEAGLE, SVIP. Noticed they all decide based on content difficulty or
network delay. Started speculative-decoding.md. This is also where the seed of my
own idea came from: none of them react to the device physically heating up.

## Session: mapping the rest of the landscape

Wanted to understand SGLang and llm-d well enough to talk about them, not just
name them. SGLang's RadixAttention is automatic prefix caching using a radix tree.
llm-d is a Kubernetes layer on top of engines like vLLM that splits prefill and
decode across the cluster and routes by cache. Also read the Google post Dr Rabab
sent about llm-d and disaggregated serving. Wrote landscape.md. Kept MLIR to a one
paragraph understanding on purpose.

Links:
- SGLang RadixAttention writeup (LMSYS blog).
- llm-d project site and the Google Cloud post on scaling inference.

## Where I ended week 1

Finished the course. Solid on the core concepts and the WISP paper. Had a clear
plan for the build in week 2: benchmark small models on my own Apple Silicon,
demonstrate the prefix caching principle locally, and write up an original idea.
