# WISP paper, in my own words

Paper: WISP, Waste and Interference Suppressed Distributed Speculative LLM Serving
at the Edge via Dynamic Drafting and SLO Aware Batching. Li et al., 2026.

This is the paper Dr Rabab sent me, so I wanted to be able to explain it plainly
rather than just recognise the title.

## What problem it solves

Most LLM inference runs entirely on data center GPUs while the phones and laptops
that send the requests sit idle. That is a waste of capable hardware, and it piles
all the cost on the cloud. WISP splits the work using speculative decoding: a
small draft model runs on the edge device and guesses the next few tokens, and the
big target model on the server checks those guesses. Because verifying a batch of
guesses is one fast parallel pass, while generating them one by one is slow, this
saves server work and frees the GPU for verification instead of generation.

## The two bottlenecks it names

Wasted Drafting Time. The edge device drafts a fixed number of tokens each round.
But the server only accepts tokens up to the first one it disagrees with, and
throws away the rest. Any drafting effort spent on those thrown away tokens was
wasted. The bigger the fixed draft window, the more can be wasted.

Verification Interference. On the server, verification requests of different sizes
get batched together for efficiency. But a batch finishes only when its slowest
member finishes, so a few long requests hold up all the short ones sharing the
batch. This is head of line blocking, and it wrecks the latency promises for the
short requests.

## How it fixes them

For wasted drafting, WISP adds a small, cheap predictor (a little MLP) that runs on
the edge and watches the draft model's own output statistics for each token:
confidence, entropy, the margin between the top two choices, and the spread of the
logits. From those it predicts when the server is about to reject, and tells the
device to stop drafting and send for verification right then, instead of blindly
drafting a fixed number. Training this predictor to have a low false alarm rate
provably lowers the wasted drafting.

For verification interference, WISP makes the server's batching deadline aware. It
treats building each batch as a knapsack style packing problem: urgent requests
whose deadlines are close get admitted first by earliest deadline, then remaining
room is filled by a value score, all while a latency estimator checks the batch
will still finish in time.

## Roughly what the results were

Compared to running everything centrally, WISP supported up to about twice as many
devices under the same latency targets, and up to around four times as many
compared to the earlier SLED system it builds on. The predictor on its own raised
how many drafted tokens got accepted by 34 to 54 percent, and lifted useful output
(goodput) by 20 to 30 percent. The output stays identical to what the big model
would have produced on its own, so none of this costs quality.

## What I want to reproduce

The part that is doable at small scale by one person is their core observation
(their section 3.3): that the draft model's own output statistics predict whether
the target will accept a token. If I run a small draft model against a bigger
target, log those four statistics per token along with whether it was accepted, I
should be able to see the same relationship and even train a small predictor. That
is the natural flagship follow up to the benchmarks in this repo.

## The idea this sparked

Every adaptive drafting method I found, including this one, decides when to stop
based on the content (entropy, confidence, acceptance history) or on network delay.
None of them react to the edge device getting physically hot and slowing down,
even though sustained generation on a phone throttles hard. That gap is what my
research idea note is about.
