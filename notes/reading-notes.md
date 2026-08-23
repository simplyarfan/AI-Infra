Reading notes

Paper by paper notes on things I have read for the thermal and edge inference
side of the project. Each entry covers what the paper actually did, what they
found, gaps I noticed, and how it connects to my own results.

---

Paper 1: LLM Inference at the Edge, Mobile, NPU, and GPU Performance
Efficiency Trade offs Under Sustained Load

arXiv 2603.23640, Tummalapalli, Arayakandy, Pal, and Kundan, 2026.

What the paper is about

The core question is simple: when you actually use an AI model continuously
on real hardware instead of testing it once, what breaks first. Most existing
benchmarks only measure a single quick run, so they miss what happens after a
device has been working for a while and starts heating up. This paper is
built specifically to catch that.

The hardware and setup

They tested four devices, chosen to represent very different points in the
edge hardware world:

- A Raspberry Pi 5 paired with a Hailo 10H NPU, a dedicated small AI chip
  connected over PCIe, running at under 5 watts.
- A Samsung Galaxy S24 Ultra, representing a typical high end Android phone.
- An iPhone 16 Pro, representing a typical high end Apple phone.
- A laptop with an NVIDIA RTX 4050 GPU, running on battery power, representing
  a portable but far more powerful device than a phone.

The model was fixed across all four: Qwen 2.5 1.5B, quantised to 4 bit, the
same size class as models I have been testing. The prompt was also fixed at
258 tokens, and they ran it 20 times in a row on each device, back to back,
with greedy decoding so the output length would be consistent and comparable.
The first run on each device was excluded from the main results because it
reflects the model loading into memory rather than steady running performance,
which is a sensible thing to control for.

They measured throughput (tokens per second), latency, power draw, and
temperature, all logged across the 20 runs so they could see the shape of the
slowdown over time, not just a before and after number.

What they found, device by device

iPhone 16 Pro: the sharpest drop of any device. It lost close to 40 percent
of its peak speed within about the first three runs, then settled into what
they call a hot state plateau, holding steady around 23.7 tokens per second
for the rest of the test. So most of the damage happens fast, in the first
few minutes, and then the phone finds a new, much slower, sustainable speed.

Samsung Galaxy S24 Ultra: a slower, steadier decline, roughly 15 percent
over the full 20 runs, but it then hit a hard limit where the Android system
itself force capped the GPU clock speed to protect the chip, flattening the
speed at a low, fixed floor regardless of what the hardware could otherwise
do.

RTX 4050 laptop GPU: did not show the same kind of heat driven throttling.
Instead its limiting factor was the battery's power ceiling. Running on
battery, the system stabilised power draw around 34 watts and throughput
around 132 tokens per second, meaning the bottleneck here was how much power
the battery could supply, not heat build up in the chip.

Raspberry Pi with the Hailo NPU: by far the most stable. Almost no
degradation across all 20 runs, low and steady temperature throughout. The
paper's conclusion is that dedicated small AI chips, built specifically for
this kind of steady workload, handle sustained inference far better than
general purpose mobile chips that have to do everything else on the phone as
well.

The core idea across the whole paper

For phones specifically, the thing that actually limits real world
performance is not the chip's raw processing power, it is how well the device
can manage and get rid of heat. Two phones with similar peak specs can behave
completely differently under sustained load depending on their cooling and
their thermal software policies. This is exactly the distinction between peak
performance and sustained performance that keeps coming up in my own work.

Gaps and things they did not track

A few things stood out as missing or worth questioning:

They report power draw in watts from direct sensors, which is precise, but I
did not see them tracking the phone's actual battery percentage over the
course of the test. Battery level itself can affect thermal and power
behaviour on a real phone, a nearly full battery and a half drained one can
manage heat and sustained draw differently, and charging state changes this
further. Since I am flagging this as a gap rather than something I confirmed
by reading their full methods section in detail, I want to note it honestly
as an assumption based on what I could access, not a certainty.

They also ran a fixed, short 258 token prompt repeated identically 20 times.
That is good for controlled comparison, but it does not tell us what happens
over a longer session, ten or twenty minutes of varied, real conversation,
which is closer to how people would actually use these devices. My own test
ran continuous varied prompts for about five minutes, which is a different
and in some ways more realistic condition, even though it is less tightly
controlled than theirs.

They tested one model size class, 1.5B, on one prompt length. It is not clear
from what I can see whether a larger model would throttle faster since it
generates more heat per token, or slower because it is already so slow that
the chip has more idle time between tokens to cool down. That is an open
question their setup cannot answer with a single model size.

Finally, like most work in this space, they do not propose changing the
model's behaviour in response to the thermal state they measured so carefully.
They characterise the problem in detail but stop there. That is exactly the
gap my own research idea sits in.

How this connects to my own results

This is the most useful comparison I have found so far. My iPhone 17 Pro
result showed about a 10 percent drop over four identical exchanges in a
growing conversation, 77 tokens per second cold, down to about 69. Their
iPhone 16 Pro, one generation older chip, showed close to a 40 percent drop,
down to a 23.7 tokens per second floor, over their own longer 20 run protocol.

That is a big difference, and I want to be careful about what I claim from
it. There are two honest explanations, and probably both play a part:

The newer A series chip in the 17 Pro may genuinely manage heat better than
the 16 Pro's chip. This is not just a guess: Apple's own technical materials
for the 17 Pro describe a new laser welded vapor chamber cooling system, with
Apple's own stated claim of up to 40 percent better sustained performance
versus the previous generation, which is a real, citable engineering reason
for at least part of the gap.

The tests are not identical. They used Qwen 2.5 1.5B with a fixed repeated
258 token prompt over 20 controlled runs. I used Llama 3.2 1B in a shorter,
growing four message conversation that stopped once the app's context room
filled up. Different model, different prompt pattern, different test length.
Some of the gap could come from the test setup rather than the hardware
alone.

The right next step, rather than assuming the newer phone is simply better, is
to run my own test under conditions closer to theirs, same model size class,
same kind of fixed repeated prompt, similar run count, so the comparison is
fair. That would help separate how much of the gap is the new cooling design
and how much is just a difference in method.

---

Idea for what to do next based on this paper

Adjust my own benchmark protocol to move closer to theirs where it makes
sense: fixed prompt length repeated a set number of times, excluding the
first run as a warm up, so my numbers are easier to compare against published
work. Keep my own growing conversation test as well, since it represents a
different and also realistic usage pattern, and note clearly which protocol
produced which number.

Track battery percentage alongside performance on my own phone tests, since
that is a gap in their work I can actually fill cheaply on devices I already
have.

Once I have the iPhone 15, iPhone 15 Pro, and PC data using a more rigorous
protocol, build a proper comparison table that references this paper's
numbers directly, so the repo shows not just my own measurements but how they
sit next to published research.

---

Paper 2: EnerInfer, Energy-Aware On-Device LLM Inference

arXiv 2606.23001, Zou, Liu, Sun, Mascherin, Roy, Liu, Peng, Jia, and Chen,
submitted 22 June 2026, revised 24 June 2026.

What the paper is about

Most systems that run LLMs on a device assume that faster is always better,
and push the chip as hard as it can go. This paper pushes back on that
assumption. Their central finding is that on-device LLM inference usually has
what they call configuration slack: you can turn the NPU and memory down to a
slightly lower frequency, lose only a small amount of decoding speed, often an
amount a user would not notice, and get a real gain in energy efficiency and a
real reduction in heat in return. Most devices are running faster than they
actually need to for a good experience, and paying for that in heat and
battery.

The two hard problems they identify

They name two specific reasons this is difficult to act on in practice, which
they label C1 and C2.

C1: how fast a given energy setting runs, and how much power it uses, varies a
lot depending on the exact model, the inference engine, and the platform, and
this is impractical to solve by testing every model in advance, since Hugging
Face alone hosts over a million models. On top of that, most real phones do
not expose hardware sensors that measure how much power the NPU specifically
is drawing, so you often cannot even directly measure the thing you are
trying to optimise.

C2: there is no consistent ranking of which frequency setting is most energy
efficient across different models and platforms. What works best for one
model can be a poor choice for another, so there is no single safe default.

How they solve it

Rather than trying to measure power directly with sensors that mostly do not
exist on commercial phones, or profiling every model by hand, EnerInfer trains
a machine learning model to predict how fast and how power hungry an unseen
model will be at a given frequency setting, based on the structure of the
model itself, rather than needing to have run that exact model before. This
lets it pick a frequency setting that is energy efficient while still meeting
a target decoding speed, for models it has never directly measured. It also
includes a runtime thermal predictor that watches the device's temperature
and turns a thermal-aware controller on or off depending on conditions.

The important distinction from my own idea

This is the closest existing work to what I am proposing, so it is worth
being precise about the difference rather than just noting that it exists.

EnerInfer works one level below where my idea sits. It decides hardware
configuration: which NPU and memory frequency to run at. It does this
proactively, choosing a slightly lower frequency from the start because the
speed is not needed, before any overheating happens. This is different from
the throttling I measured myself, which is reactive: the chip runs at full
speed, gets hot, and only then gets forced to slow down by the system's own
protection mechanism.

My idea sits at the algorithm level, not the hardware configuration level. It
is about changing how much a speculative decoding system drafts ahead, or
whether it leans more on the NPU versus the GPU, based on live thermal
conditions, not about changing the chip's clock speed directly. EnerInfer
never mentions speculative decoding, and its thermal controller decides
whether to apply a frequency change, not how many tokens to draft. So the two
ideas are complementary rather than the same thing: a device could use
EnerInfer style frequency management and a thermal-aware speculative decoding
scheme at the same time, one managing the hardware setting, the other
managing the software behaviour running on top of it.

Gaps and things worth noting

The paper is about energy and thermal comfort broadly, and does not touch
speculative decoding at all, which is exactly the gap my idea sits in. It also
assumes a device capable enough to run an ML based predictor alongside the LLM
itself, a reasonable assumption for a modern phone but worth keeping in mind.
It is also very recent, submitted in June 2026, so it represents the current
edge of this specific research thread rather than settled practice.

---

References

- Paper 1: Tummalapalli et al., "LLM Inference at the Edge: Mobile, NPU, and
  GPU Performance Efficiency Trade-offs Under Sustained Load", arXiv:2603.23640.
  https://arxiv.org/abs/2603.23640
- Paper 2: Zou et al., "EnerInfer: Energy-Aware On-Device LLM Inference",
  arXiv:2606.23001. https://arxiv.org/abs/2606.23001
- Apple iPhone 17 Pro vapor chamber cooling: Apple's iPhone 17 Pro technical
  specifications and product pages.

See notes/research-idea.md for the fuller reference list covering NPU versus
GPU efficiency, the fixed shape problem on Apple hardware, and the Qualcomm
Hexagon NPU backend, gathered while developing the idea these two papers feed
into.

