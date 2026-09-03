# Cross device benchmark table

Same kind of model across every device I can get my hands on, to see how decode
speed and thermal behaviour change with the hardware. Device specs are kept
separately in HARDWARE_APPENDIX.md so this file stays focused on results.

## Two different protocols, kept clearly separate

I am running two genuinely different test methods here, and I want to be
upfront that they are not directly interchangeable. Mixing them up would make
the comparison look more standardised than it actually is.

**Protocol A, the matched mobile protocol (iPhone and iPad):** the exact same
model, Llama-3.2-1B-Instruct 4 bit Q4_0 GGUF, run through the PocketPal app.
The exact same four prompts, in the exact same order, in a single growing
conversation: write a paragraph on how LLMs generate text, then explain the KV
cache, then explain prefill versus decode, then more detail on why decode is
slower. Cold reading is the first response. On both devices, the conversation
organically hit PocketPal's context room limit right after the fourth response,
so both tests are the same length by construction, not because I cut one short.
This pair is a clean, apples to apples comparison.

**Protocol B, the Mac protocol (MacBook only, so far):** a Python script using
MLX, running Qwen2.5 models rather than Llama, with different prompts, and a
different generation method. Instead of a growing multi turn conversation, the
sustained test regenerates a fixed number of fresh tokens from the same
starting prompt, over and over, for a genuine 300 plus seconds of wall clock
time. Because it never accumulates a growing chat history, it never hits a
context wall the way the phone and tablet do. This is a different way of
stressing the hardware, not a lesser or greater one, and it should be read as a
separate reference point rather than the same test run on a third device.

I want a matched Mac result eventually, meaning the same model, same app style
growing conversation, same four prompts, so all three devices sit on one
directly comparable row. That is not done yet.

## Results

| Device | Chip | Protocol | Cold decode (tok/s) | TTFT (ms) | Later decode (tok/s) | Change | Felt warm |
|---|---|---|---|---|---|---|---|
| iPhone 17 Pro | A19 Pro | A, matched mobile | 77.2 | 209 to 262 | 69.1 | about -10 percent | yes, clearly, whole device |
| iPad Pro 11" (4th gen) | A12Z Bionic | A, matched mobile | 97.05 | 192 | 94.90 | about -2 percent | yes, mild, centred on the back near the Apple logo |
| MacBook Pro (M4) | Apple M4 | B, Mac only, different model and method | 54.5 (Qwen 3B, MLX) | 190 | 54.4 | none | no |
| iPhone 15 | A16 Bionic | not yet run | | | | | |
| Windows PC | RTX 5060 Ti 16GB | not yet run | | | | | |

Raw numbers for the matched mobile protocol, in the order the prompts were sent,
identical on both devices:

iPhone 17 Pro:
1. Write a detailed paragraph explaining how LLMs generate text: 77.15 tok/s
2. Now explain what a KV cache is and why it matters: 77.93 tok/s
3. Continue and explain the difference between prefill and decode: 73.37 tok/s
4. Write more detail about why decode is slower than prefill: 69.14 tok/s, then hit the context room warning

iPad Pro 11" (4th gen):
1. Write a detailed paragraph explaining how LLMs generate text: 97.05 tok/s, 192ms TTFT
2. Now explain what a KV cache is and why it matters: 94.68 tok/s, 308ms TTFT
3. Continue and explain the difference between prefill and decode: 94.11 tok/s, 67ms TTFT
4. Write more detail about why decode is slower than prefill: 94.90 tok/s, 72ms TTFT, then hit the context room warning

## Observations so far

The iPhone and the iPad ran the identical protocol, same model, same four
prompts, same order, and both were stopped by the same context room limit right
after the fourth exchange. Given that, the comparison between them is fair. The
iPhone dropped about 10 percent over those four exchanges and was clearly hot to
the touch across the whole body by the end. The iPad dropped only about 2
percent, which is close to normal run to run noise, and the warmth I felt was
milder and localised to the back panel near the Apple logo. That is a real
difference in how the two devices handle the same short burst of sustained
work, with the larger iPad body likely giving it more thermal mass to absorb
heat before it needs to visibly slow down, though four exchanges is a short
window and a longer matched run would make this more conclusive.

The MacBook number is a separate data point under a different protocol, not
part of this direct comparison. It held completely flat over a genuine five
minutes of continuous fresh generation, which fits with it having active
cooling, but because the model, prompt, and generation method all differ from
the mobile test, I am not reading this as "the Mac beats the phone and tablet",
only as "the Mac does not show throttling under its own sustained test."

## Open questions this table should still answer

Running a matched Protocol A style test on the Mac, meaning the same model
family, same growing conversation style, same four prompts, so all three
devices can sit in one directly comparable row instead of two protocols side by
side.

Whether a longer matched mobile run, with the context room increased so it does
not stop after four exchanges, changes the iPad's roughly flat result. Four
exchanges may simply not be enough time for its larger body to start showing
heat the way the phone did.

Does a newer phone throttle less than an older one, or does it just run hotter
because it is faster. The iPhone 15 and 17 Pro comparison, run under the same
Protocol A, should show whether newer chips genuinely help or just move the
problem.

How does a desktop GPU behave under sustained pressure, and whether it is worth
designing a Protocol C for GPUs specifically, since a desktop GPU workload looks
very different from a phone app conversation. The DGX Spark writeup describes
serious hardware thermally shutting down under continuous inference, with the
fix being to cap the clock speed for almost no throughput loss, because the
workload was memory bandwidth bound rather than compute bound. Same reasoning
as my mobile results, much bigger scale.

## A real answer on the iPhone generation question

Apple's own technical materials for the iPhone 17 Pro describe a new vapor
chamber cooling system, laser welded into the aluminium unibody, that moves heat
away from the A19 Pro chip more effectively than previous designs. Apple's
stated claim is that this delivers up to 40 percent better sustained
performance compared to the previous generation. That gives a real, citable
engineering reason why my 17 Pro result, about a 10 percent drop over four
exchanges, is so much milder than the iPhone 16 Pro result reported in the
paper in my reading notes, about a 40 percent drop over their own longer
protocol. Testing the iPhone 15 under my own matched Protocol A
would help separate how much of that gap is the newer cooling design and how
much is simply a difference between my test and theirs.
