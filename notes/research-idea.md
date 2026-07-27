# Research idea: thermal aware speculative decoding

## The one line

Adapt how much a phone drafts (the speculation length), or when it hands work to
the cloud, based on the device's real thermal state, because sustained generation
on a phone throttles hard and none of the current methods react to that.

## Where it comes from

Two things line up:

1. In my own edge benchmarks I measured generation speed on my iPhone dropping
   from about 77 tokens per second when cool to about 69 over a few minutes of
   sustained use, roughly a 10 percent fall, and I could feel the phone getting
   hot in my hand as it happened. The Mac, with active cooling, stayed flat over
   the same kind of test. This is a real, repeatable effect I saw myself, not a
   guess. A bigger model or a longer run would throttle harder.

2. When I read around adaptive speculative decoding, every method decides the
   draft length from the content (entropy, confidence, how many tokens got
   accepted recently) or from network delay. I could not find one that uses the
   device's thermal state as the signal.

So there is a clean gap. When the phone is cool it can afford to draft more
aggressively. When it is throttling, drafting hard is wasteful because the draft
model itself has slowed down, and it might be better to draft less or lean more on
the cloud. The device already exposes its thermal state to apps, so the signal is
available for free.

## What is closest to it (so I do not overclaim)

The nearest work I found is PELM, which tunes power and speed on edge devices using
speculative decoding together with frequency scaling under thermal conditions. It
is about power and latency, and it uses a learned power governor. It is close, so I
would cite it and be clear about the difference: my angle is to make the thermal
state the explicit input that controls the draft length or the edge to cloud
split, and to validate it with real on device measurements. There is also work that
is genuinely thermal aware but controls clock frequency rather than speculation.
I would not claim total originality in writing until I have read those fully, but
the specific framing looks open.

## How I would test it (rough)

- Measure the throttling curve on a real device (this repo already starts that).
- Show that draft speed and acceptance behaviour change as the device heats.
- Prototype a simple rule that lowers the draft length as thermal state worsens,
  and compare useful output against a fixed draft length over a long session.

This is a proposal, not a finished result. I am putting it here as an example of a
direction I came up with myself from my own measurements, which is the kind of
thinking I want to bring to the group.
