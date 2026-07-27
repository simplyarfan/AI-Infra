# Speculative decoding notes

## The core algorithm

Speculative decoding speeds up generation without changing the output. It uses two
models:

- A small, fast draft model proposes several tokens ahead.
- A big, slow target model checks all of those proposed tokens in a single
  forward pass.

The trick is that checking many tokens at once is cheap (one parallel pass),
while generating them one at a time is what is normally slow. Accepted tokens are
kept; at the first token the target disagrees with, the target supplies the
correct token and the cycle restarts. A rejection sampling rule makes the final
output mathematically identical to what the target would have produced alone, so
it is lossless. The original idea is from Leviathan et al. and Chen et al., 2023.

## Terms I want straight

- Draft model and target model: the small guesser and the big verifier.
- Acceptance rate: the fraction of drafted tokens the target keeps. Higher means
  more speedup. Bigger draft models tend to have higher acceptance because they
  agree with the target more often.
- Speculation length (often called gamma or K): how many tokens to draft before
  verifying. Too short means too many round trips. Too long means more wasted
  drafting when a rejection comes early. Choosing this well is the whole subfield
  below.
- Goodput: verified, committed tokens per second delivered to the user, not
  counting rejected drafts. A more honest measure than raw throughput.

## The adaptive draft length family (what I found reading around)

Choosing the speculation length dynamically instead of fixing it is an active
area. The ones I came across:

- DISCO (2024): trains a small classifier to set the draft length each step.
- AdaEDL (2024): training free, stops drafting early using an entropy based bound
  on the acceptance probability.
- SpecDec++ (2024): frames stopping as a decision process with a trained head that
  predicts acceptance and stops past a threshold.
- GammaTune (2025): training free, adjusts the length from recent acceptance
  rates.
- EAGLE and AdaEAGLE (2024): AdaEAGLE uses a small predictor to choose draft
  structure, lossless speedup around 1.6x.
- SVIP (2024): difficulty aware length from the draft model's entropy.

## The edge and cloud split branch

This is the branch the WISP paper lives in: put the draft model on the edge device
and the target model in the cloud. Related work here includes SLED (the paper WISP
improves on), DSD, and several others working on splitting draft and verify across
a network. The common theme is that they adapt to content difficulty or to network
delay.

## The gap I care about

Across all of these, the thing that decides the draft length is either the content
(entropy, confidence, acceptance history) or the network. None of them react to
the physical state of the edge device as it heats up and throttles, even though I
can measure that throttling directly on a phone. That is the opening my research
idea note describes.
