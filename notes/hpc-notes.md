# HPC notes and how they connect to inference

High performance computing is the backbone of this whole area. I will not pretend
I have run jobs on a supercomputer, but the way HPC people think about hardware
is exactly the way you have to think to make inference fast, so these are the
ideas I want to have straight.

## Compute bound versus memory bound

The first question HPC asks about any piece of work: is it limited by how fast the
chip can do math, or by how fast it can move data in and out of memory?

- Prefill (reading the prompt) is compute bound. You are pushing a lot of tokens
  through big matrix multiplies at once, and the math units are the bottleneck.
- Decode (generating one token at a time) is memory bound. For each single token
  you have to read the whole set of model weights and the KV cache out of memory,
  and you do very little math with them before needing the next read. The memory
  bandwidth runs out long before the math units are busy.

This is why decode speed on a device tracks memory bandwidth more than raw
compute, and it is the single most useful lens I picked up. It also explains why
a phone, which has far less memory bandwidth than a data center GPU, is so much
slower at generation even when its raw compute looks respectable on paper.

## The roofline model

The roofline model is the classic HPC picture. You plot performance against
arithmetic intensity (how much math you do per byte you move). There are two
ceilings: a flat one set by peak compute, and a sloped one set by memory
bandwidth. Low intensity work sits under the sloped part, meaning it is bandwidth
limited and you cannot reach peak compute no matter what. Token by token decode
is a low intensity workload, so it lives under the bandwidth roof. Knowing this
tells you that for decode, the wins come from moving less data (smaller weights
through quantization, smarter cache use) rather than from a faster math unit.

## Mixed precision and quantization

Models store numbers at some precision. Using fewer bits per number (going from
16 bit down to 8 or 4 bit) means every weight is smaller, so you move fewer bytes
per token. Because decode is bandwidth bound, this often gives close to a
proportional speedup, and it shrinks memory use so bigger models fit. The cost is
a small accuracy loss, usually a percent or two if done well. Mixed precision is
the general version of the idea: keep high precision only where the accuracy
actually needs it, use low precision everywhere you can get away with it. This is
central to Dr Rabab's own line of work, so I want to be fluent in why it helps and
what it costs.

## Why this matters for the edge work

My edge benchmarks are a small HPC study in disguise. When I measure decode speed
against model size and quantization level, I am really measuring the bandwidth
roof of the device. When I measure sustained generation and watch the speed fall
as the device heats up, I am watching a power and thermal wall, which is the same
kind of wall that limits how densely you can pack and cool a supercomputer, just
at a phone's scale. Framing it this way is how a small project on consumer
hardware connects honestly to serious infrastructure questions.

## Terms I want to be able to use correctly

Compute bound, memory bound, memory bandwidth, arithmetic intensity, roofline,
peak FLOPS, mixed precision, quantization, and the general habit of asking "where
is the bottleneck" before trying to optimize anything.
