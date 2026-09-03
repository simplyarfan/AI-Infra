# Research idea: thermal aware and NPU aware mobile LLM inference

## The starting point

Phones get hot when they run AI for a while, and when they get hot they slow
themselves down to cool off. I measured this myself. On the iPhone 17 Pro,
decode speed dropped from about 77 tokens per second to about 69 over four
identical exchanges, roughly a ten percent drop, and the phone was clearly hot
to the touch. On the iPad Pro, under the exact same test, the drop was only
about two percent, and the warmth was milder and localised to the back panel.
Both devices were running the same model, same prompts, same app.

## Two separate but connected problems

Reading around this, I found the issue is really two related problems, not
one.

**Problem 1: the chip throttles once it gets too hot.** This is the direct
thing I measured. The technical name for the mechanism is DVFS, dynamic
voltage and frequency scaling, where the system deliberately lowers the clock
speed and voltage to reduce heat once a limit is reached. Every adaptive
method I found in the literature reacts to content difficulty or to network
delay, not to the device's thermal state directly, even though the thermal
state is measurable and freely available to any app.

**Problem 2: most phones run AI on the wrong chip in the first place.** Every
modern phone actually has a dedicated AI chip built in, called an NPU, neural
processing unit. It exists specifically to do this kind of math efficiently.
But almost none of the common ways people actually run an LLM on their phone
today use it. Instead they default to the GPU, which is a worse fit for two
reasons: the GPU is also busy rendering the phone's screen and everything
else on it, so LLM work has to compete for it, and the GPU uses noticeably
more power to do the same job. One study I found measured the NPU running
matrix operations representative of LLM inference at roughly four times
better efficiency than the GPU, at less than half the power draw. Using the
wrong chip does not just waste efficiency, it directly creates more of the
heat that causes problem 1.

## Is this only an Apple problem, or true across phones generally

I checked this specifically rather than assume. The honest answer is that it
affects mobile devices broadly, but the reason differs by chip maker, so it is
worth separating clearly rather than saying "phones are bad at this."

On Apple hardware (iPhone, iPad), the mainstream tools I have actually used
in this project, MLX on the Mac and llama.cpp inside the PocketPal app on
iOS, both explicitly only support CPU and GPU, not the Apple Neural Engine.
Apple's own framework that can reach the Neural Engine, Core ML, requires the
model's input and output sizes to be fixed ahead of time, which does not fit
how LLMs generate text naturally, since you do not know how many tokens you
will produce in advance. I originally treated this as an unsolved wall. It is
not: a real, working open source project called CoreML-LLM has engineered
around it using chunked decoding, with measured numbers on an iPhone 17 Pro,
my own phone model, showing large gains over the GPU path (see below). So the
accurate statement is that the mainstream, most commonly used tools do not
reach the Neural Engine, but a working solution exists outside that
mainstream, and it is not yet part of the widely adopted ecosystem.

On Qualcomm Snapdragon hardware, which powers most Android flagship phones
including many Samsung Galaxy models, the situation is considerably further
along. llama.cpp has an official backend for Qualcomm's NPU, called Hexagon,
and it genuinely works. Reported numbers show a 3 billion parameter model
running entirely on the Hexagon NPU at around 10 tokens per second, described
by the developers involved as fast enough to feel fluid and interactive, not
just a proof of concept.

So the accurate way to describe this is not "Apple has a problem and Android
does not." It is that on both platforms, most people running an LLM on their
phone today are, by default, using the GPU rather than the NPU. On Apple
devices that is because the tooling for general LLMs on the NPU does not
really exist yet. On Qualcomm devices the tooling exists and works, but it
is not what apps default to, so most real world usage still ends up on the
GPU there too. Either way, the GPU stays the common path in practice, and
that is what is generating avoidable heat.

One thing I have not verified: Samsung also ships some phones with its own
Exynos chip instead of Qualcomm Snapdragon depending on the region and model,
and I do not yet know the NPU support situation for Exynos specifically. That
is worth checking before making claims about Samsung devices generally rather
than about Qualcomm powered Android phones specifically.

## What frameworks currently exist, and where they stand today

- **llama.cpp**, the most widely used local inference engine, and what the
  PocketPal app on my iPhone and iPad is built on. CPU and GPU only on Apple
  devices. On Qualcomm Snapdragon devices it has an official Hexagon NPU
  backend that works, alongside CPU and GPU options, selectable by the
  developer or user.
- **MLX**, Apple's own framework, used for my Mac benchmarks. CPU and GPU
  only, no Apple Neural Engine support at the time of writing.
- **Core ML**, Apple's framework that can reach the Apple Neural Engine. I
  originally wrote this off as blocked by the fixed input and output shape
  requirement, which is real, but I was wrong to treat it as an unsolved
  wall. See CoreML-LLM below, which shows it has been engineered around.

- **CoreML-LLM**, an actively developed open source project that gets LLMs
  running properly on the Apple Neural Engine, working around the fixed
  shape problem using chunked decoding and Apple's Core ML stateful model
  API for KV cache reuse across turns. This is a direct correction to what I
  first wrote in this file: the fixed shape wall is not unsolved, it has
  been engineered around, at least for smaller models. Published numbers on
  an iPhone 17 Pro, the same device I benchmarked myself, show a 2B model
  going from 7.5 to 24 tokens per second and from 1.7 GB down to 256 MB of
  memory by moving from a GPU style approach to this Neural Engine approach,
  and a 32x reduction in second turn response time from KV cache reuse
  across turns, which is the same underlying idea as my own prefix caching
  demo in 02-prefix-caching, just implemented through the Neural Engine
  instead of a server side cache. Caveat: this works well specifically for
  smaller models, roughly 2B and under, which happens to be the exact size
  class I have been testing on my own devices. General guidance for 2026
  still points to the GPU as the right choice for larger, 7B plus
  interactive models. This is also a community project, not yet as widely
  adopted as MLX or llama.cpp, so it is a real, working solution rather than
  a mainstream default.

- Recent research systems specifically trying to close this gap: one paper
  from April 2026 built a runtime to get a certain style of LLM (mixture of
  experts models) working efficiently on the Apple Neural Engine, carefully
  working around the fixed shape constraint. Another piece of work explored
  running the prefill phase on the NPU and the decode phase on the GPU within
  a single device, splitting the two phases across the two chips based on
  which one suits each better.

That last idea is worth sitting with for a moment, because it is the same
splitting idea behind WISP and behind llm-d, just applied inside one phone
instead of across a phone and a cloud server, or across a whole cluster.
Prefill and decode keep turning up as the natural place to split work,
whatever the scale.

## The challenges that actually need solving

Pulling this together, there are a small number of real, specific problems
standing between where things are now and a phone that runs LLMs efficiently
on its NPU without throttling:

1. **The fixed shape problem on Apple hardware, now partially solved.** LLM
   generation is variable length by nature and NPUs want fixed shapes. I
   originally treated this as unsolved in this file. It is not: CoreML-LLM
   demonstrates a real, working solution using chunked decoding and Apple's
   stateful model API, with measured numbers on my own phone model. The open
   part of this challenge now is whether the same approach scales to larger
   models, and whether it becomes mainstream tooling rather than staying a
   community project outside the main MLX and llama.cpp ecosystem.

2. **NPU support existing but not being the default.** On Qualcomm devices the
   capability is there, but ordinary apps and users are not automatically
   getting the benefit of it, because it has to be explicitly selected rather
   than happening automatically.

3. **No thermal awareness in either case.** Even where NPU support exists, I
   have not found anything that dynamically decides which chip to use, or how
   much to draft, based on the device's actual thermal state in the moment.
   Everything I have found reacts to content or to a fixed configuration
   choice made ahead of time, not to what the hardware is doing right now.

4. **Almost no work connecting this to speculative decoding specifically.**
   Speculative decoding, the technique behind WISP, needs a small draft model
   that runs fast and cheap. That is exactly the kind of workload an NPU is
   well suited to. I have not found work that specifically puts the small
   draft model on the NPU while keeping the larger, more flexible verification
   step on the GPU or in the cloud.

## The idea, stated properly

There are two levels to this, a smaller one I can actually investigate now,
and a bigger one it points toward.

**The immediate idea:** design speculative decoding for phones so that the
small draft model is specifically placed on the NPU rather than the GPU,
since it is a smaller, more fixed shaped model that is a better match for the
NPU's constraints than the larger verifier would be, and since running it
there generates less heat in the first place. Then, on top of that, make the
speculation length itself adapt to the device's live thermal state, so that
even the NPU side backs off if the device is still running hot, rather than
always drafting at a fixed rate regardless of conditions.

**The bigger picture this sits inside:** the same "which piece of work goes on
which piece of hardware" question shows up at every scale in this whole area.
WISP splits draft and verify across a phone and a cloud server. llm-d splits
prefill and decode across machines in a cluster. The NPU and GPU split inside
a single phone is the same question again, just at the smallest scale. A
phone that is thermally aware and NPU aware would be applying that same idea
consistently, all the way down.

## Honest state of this idea

This is still a proposal, not a result. I have real measured evidence for the
thermal side (my own iPhone and iPad numbers), and real, current literature
support for the NPU efficiency side, including a working, measured solution
to the fixed shape problem I originally thought was unsolved. What I do not
yet have is anything built that actually puts a draft model on the NPU
myself, or that adapts drafting to live thermal readings. A concrete next
step this opens up: installing CoreML-LLM on my own iPhone 17 Pro and
directly comparing its Neural Engine numbers against my existing GPU numbers
from PocketPal, on the same device, same size class of model. That would turn
the NPU efficiency claim from something I cite from other papers into
something I measured myself, the same way I did for thermal throttling. The
other realistic next step is to look at whether the Qualcomm Hexagon backend
in llama.cpp is usable enough right now to try running a small draft model on
an NPU on Android hardware as well, to see the same comparison on the other
platform.

Attempted next step, blocked for a real reason: I tried to install and run
CoreML-LLM's example app on my iPhone 17 Pro through Xcode. The build tooling
itself would not launch at all, and after ruling out signing, developer
directory configuration, and Launch Services registration as causes, the
real reason turned out to be that my Mac is running a macOS 27 beta, and the
current public Xcode from the App Store does not support beta operating
systems, only the latest stable release. Getting a working setup would
require the matching beta build of Xcode from developer.apple.com under an
Apple Developer account, which is a reasonable thing to do at a calmer point
but not something to rush through. Recording this here because it is a real,
specific environment constraint, not a dead end, and it is worth knowing
exactly why before trying again.

## Live examples that tie this together

A running collection of real, current, commercial signals that this exact
problem, splitting AI work sensibly across hardware and managing heat, is
taken seriously by industry, not only by the research papers cited below.
Adding to this as I come across more.

### Siri AI in iOS 27

While waiting on device access for other tests, I looked into why my own
iPhone barely warms up when I use the new Siri AI in the iOS 27 beta, given
everything above about GPU based LLM apps throttling. The answer turns out to
have two parts, and both connect directly to this project.

First, and this was a genuine surprise: a large part of what Siri AI does is
not computed on the phone at all. Apple built Siri AI as a three tier system.
Simple requests are handled by a small model on the device. Moderate requests
are sent to Apple's own Private Cloud Compute servers. The heaviest reasoning
is handled by a custom Google Gemini model, reported at roughly 1.2 trillion
parameters, running on Google Cloud, under a paid partnership between Apple
and Google announced in January 2026. So unlike my own benchmarks, where the
whole conversation runs start to finish on the phone's own chip, a meaningful
share of what Siri AI does is offloaded elsewhere by design. That is a
commercial, very large scale version of exactly the edge and cloud split idea
behind WISP, just with three tiers instead of two, and it is a big part of why
the phone does not have to generate as much heat: much of the hardest work
never runs on it in the first place.

Second, for the part that does run on the phone, the most capable on-device
tier is limited to devices with at least 12 GB of RAM, which currently means
the iPhone 17 Pro and iPhone Air, and my own hardware appendix already lists
the 17 Pro at exactly 12 GB. That on-device tier is Apple's own Foundation
Model, and while I have not found a source stating outright that this
specific model runs on the Apple Neural Engine, it fits everything else I
found this session: Apple's own on-device model line uses small parameter
counts with aggressive quantization and KV cache sharing, which is the same
family of technique CoreML-LLM uses to run community models on the ANE. The
reasonable, though not fully confirmed, read is that Apple's own first party
model already does for itself what CoreML-LLM is doing unofficially for
third party models: keep the on-device portion of the work on the Neural
Engine rather than the GPU. I am flagging the difference between what is
directly sourced here (the three tier architecture, the Google partnership,
the 12 GB requirement) and what is a well supported inference rather than a
confirmed fact (that the on-device tier specifically uses the ANE).

Either way, Siri AI is a real, live, large scale example sitting on my own
phone of the exact two ideas this whole file is about: splitting work between
edge and cloud, and keeping the on-device portion off the GPU.

Sources: Apple Newsroom, "Apple introduces Siri AI, a profoundly more
capable and personal assistant", June 2026,
https://www.apple.com/newsroom/2026/06/apple-introduces-siri-ai-a-profoundly-more-capable-and-personal-assistant/.
MacRumors, "Which iPhones Support Every iOS 27 Feature?", August 2026,
https://www.macrumors.com/2026/08/17/which-iphones-support-every-ios-27-feature/,
for the three tier architecture, the Google Gemini partnership, and the 12 GB
on-device model requirement.

### HUMAIN's AI PC, built with Qualcomm

Came across a LinkedIn teaser from HUMAIN, a Saudi AI company, showing a
laptop called Horizon Ultra, built with Qualcomm over about a year, and
described as bringing CPU, GPU, and NPU together to run AI models locally.
This is directly relevant, and worth being precise about what is genuinely
new versus what is standard platform behaviour being presented as news.

Confirmed the chip: Qualcomm Snapdragon X2 Elite, launched in 2026. Qualcomm's
own product materials for this chip state directly that it spreads AI
workloads across the CPU, GPU, and NPU to optimise power consumption, so the
three way split is a feature of the chip platform itself, not something
HUMAIN engineered. What HUMAIN is actually building on top of it, per their
own post, is a new operating system aimed at agentic AI, described as
rethought around intent rather than apps, with the fuller reveal saved for a
conference called LEAP, a major annual tech event in Riyadh. The post itself
says the hardware is not the part they are most excited about, which is a
fairly direct admission that the headline chip story is not the novel part.

The genuinely relevant technical detail: the X2 Elite's NPU is rated at 80
TOPS, up from 45 TOPS on the previous generation, and its internal scheduler
dynamically adjusts voltage and frequency, DVFS again, specifically to
balance AI performance against the device's thermal envelope. This is the
same DVFS mechanism from my own measurements and from EnerInfer, now clearly
named as a first class design goal in a shipping 2026 laptop chip, not just
something I inferred from watching my own phone throttle. The CPU side also
gained new instructions, called SME, specifically to accelerate AI and HPC
style workloads, meaning even the fallback chip in this three way split is
now purpose built for this kind of work, not a general purpose afterthought.

One honest note of scepticism worth keeping: tech press coverage around the
X2 Elite launch pointed out that some of Qualcomm's longest standing laptop
partners were not publicly endorsing this platform at launch, only newer
partners including HUMAIN, Asus, and HP. Worth taking the announcement
seriously as a real signal that CPU, GPU, NPU allocation and thermal
management is now a marketed, competitive feature of commercial hardware, but
without assuming every claim in a launch post is fully proven yet.

Sources: Tareq Amin (HUMAIN), LinkedIn post, August 2026. Qualcomm Snapdragon
X2 Elite product materials, for the CPU, GPU, NPU workload spreading claim,
the 80 TOPS NPU figure, the DVFS based thermal scheduler, and the SME CPU
instructions.

### DFlash, sighted via a real production result

Covered in full in notes/speculative-decoding.md, but noting here too since
it is the same kind of signal: a GenAI team lead at a Saudi bank posted real
production numbers, 159 tokens per second and 0.150 second time to first
token on a 27B model on a single Hopper GPU using FP8, achieved through DFlash
style speculative decoding. Another live, current, Saudi hosted data point
that the techniques this project is built around are being used for real,
measured production work right now, not only studied in papers.

## References and sources

Papers and technical sources behind the claims above, so I can go back to any
of them for more depth or check a number.

**On why NPUs suit LLM inference better than GPUs:**

- Xu, D. et al., "Fast On-device LLM Inference with NPUs", arXiv:2407.05858.
  https://arxiv.org/pdf/2407.05858
  Explains directly why mobile GPUs are a worse fit: they are shared with
  graphics rendering, while NPUs are dedicated and mostly idle, with greater
  computational capacity and energy efficiency for this specific workload.

- "Benchmarking Edge AI Platforms for High-Performance ML Inference",
  arXiv:2409.14803. https://arxiv.org/pdf/2409.14803
  Source of the roughly four times efficiency figure: NPU showed close to 4x
  better performance than GPU for LLM style matrix vector multiplication, at
  under half the peak power draw (35W versus 75W in their test setup).

**On the fixed shape problem and Apple's Neural Engine specifically, and a
real working solution:**

- john-rocky, CoreML-LLM, GitHub repository.
  https://github.com/john-rocky/CoreML-LLM
  An actively developed open source project (developer Daisuke Majima)
  running LLMs on the Apple Neural Engine via chunked decoding and Core ML's
  stateful model API for KV cache reuse. Source of the measured iPhone 17 Pro
  numbers cited above: 24 tokens per second at 256 MB for a 2B model on the
  Neural Engine, versus 7.5 tokens per second at 1.7 GB on an earlier GPU
  style build, and a 32x reduction in second turn latency from cross turn KV
  cache reuse.

- BrightCoding blog, "Stop Wasting GPU Cycles! CoreML-LLM Unlocks ANE for
  Insane On-Device Speed".
  https://www.blog.brightcoding.dev/2026/05/23/stop-wasting-gpu-cycles-coreml-llm-unlocks-ane-for-insane-on-device-speed
  Write up of the project above, describing the ANE first, GPU free design
  and the practical case for moving LLM inference off the GPU entirely.

- "Orion: Characterizing and Programming Apple's Neural Engine for LLM
  Training and Inference", arXiv:2603.06728. https://arxiv.org/pdf/2603.06728
  Academic paper specifically studying how to program the Apple Neural
  Engine for LLMs, further evidence this is an active, real research and
  engineering area rather than a closed door.

- "Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs"
  (NPUMoE), arXiv:2604.18788. https://arxiv.org/pdf/2604.18788
  April 2026 paper building a runtime specifically to get mixture of experts
  LLMs running on the Apple Neural Engine, working around its fixed shape and
  static computation constraints.

- SqueezeBits Tech blog, "Disaggregated Inference on Apple Silicon: NPU
  prefill and GPU decode".
  https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176
  States plainly that MLX lacks Apple Neural Engine support and that Core ML
  requires fixed input shapes for ANE compatibility, which is why they split
  prefill onto the NPU and decode onto the GPU rather than using the NPU for
  everything.

- ggml-org/llama.cpp, GitHub Discussion #336, "Neural Engine Support".
  https://github.com/ggml-org/llama.cpp/discussions/336
  The maintainers' own explanation for why they have not added Apple Neural
  Engine support: on iPhone the NPU is a small part of the chip's silicon
  area, roughly 2 to 3 GPU cores' worth, and they did not see it as worth
  building for LLM inference specifically.

**On NITRO and Intel's laptop NPU, the fixed shape cost in practice:**

- "NITRO: LLM Inference on Intel Laptop NPUs", arXiv:2412.11053.
  https://arxiv.org/pdf/2412.11053
  Documents the concrete cost of the fixed shape requirement: because the
  decode stage cannot use a dynamic shape, it wastes compute running the
  worst case size on every iteration. Found the NPU faster than CPU by up to
  1.8x on medium sized models, but still behind the GPU on raw speed at the
  time of writing, with weight compression tooling not yet helping NPU
  performance the way it helps CPU and GPU.

**On Qualcomm Snapdragon and the Hexagon NPU, the more mature side:**

- llama.cpp official docs, Snapdragon backend.
  https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md
  Confirms llama.cpp supports three backends on Snapdragon devices directly:
  CPU, Adreno GPU via OpenCL, and Hexagon NPU, selectable per run.

- Grapeup blog, "Running LLMs on-device with Qualcomm Snapdragon 8 Elite".
  https://grapeup.com/blog/running-llms-on-device-with-qualcomm-snapdragon-8-elite
  Source of the real world numbers: Llama 3.2 3B at about 10 tokens per
  second and Llama 3.1 8B at about 5 tokens per second, running entirely on
  the Hexagon NPU, described as fluid and interactive rather than a demo.

- ggml-org/llama.cpp, GitHub Discussion #8273, "Performance of llama.cpp on
  Snapdragon X Elite/Plus". https://github.com/ggml-org/llama.cpp/discussions/8273
  Community benchmarking showing close to 4x improvement of NPU FastRPC
  execution over CPU FP32 for large matrix multiplication representative of
  LLM inference patterns, plus detail on dequantization overhead as a current
  bottleneck on the NPU path specifically.

**Connecting back to material already in this repo:**

- My own measurements: 01-edge-benchmarks/DEVICE_TABLE.md, iPhone 17 Pro and
  iPad Pro 11 inch 4th generation results.
- notes/wisp-summary.md, for the draft and verify split this idea extends.
- notes/landscape.md, for how llm-d's prefill and decode split at cluster
  scale is the same shape of idea applied here at chip scale inside one
  phone.

