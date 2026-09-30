Idea: AutoOptAgent and the thermal gap

This note works through Dr Alomairy's own patent architecture, AutoOptAgent, and lays out precisely where a thermal awareness extension would fit inside it, using the research done throughout this project, plus a directly relevant paper she pointed me to that supports and sharpens this direction. Based on a single presentation slide I was shown, not the full patent filing, which is not publicly available yet since patent applications are normally kept confidential for up to 18 months after filing. Everything here should be treated as a reading of that slide, to be checked against the actual filing when possible.

What AutoOptAgent is

Working title: AI-driven agentic optimizer that decides how to optimize models automatically at runtime. Filed as a patent in both the US and Saudi Arabia. It is a general purpose, multi-agent system: not built for one specific model, but a framework that can automatically find a good runtime configuration for any model, on any hardware, by searching and learning from feedback rather than a person hand tuning settings.

The architecture, agent by agent

Agent 1, Optimization Objectives and Constraints. Defines what the system is optimizing for. Objectives named on the slide: latency, throughput, memory, energy. Constraints named: accuracy, cost, memory limits, latency deadlines. It also incorporates hardware and system context: CPU, GPU, accelerator, memory hierarchy, cache.

Agent 2, Baseline. Runs the model with its default or prior configuration and records baseline performance metrics. Parameters may include batch size, precision, replica count, and similar settings.

Agent 3, Evaluation. Checks the baseline against the objectives and constraints from Agent 1, and detects system state: underutilization, overload, instability, or constraint violation. If everything already meets the objectives, the configuration is accepted as is.

Decision point, is optimization needed. If no, go straight to Agent 8. If yes, enter the loop at Agent 4.

Agent 4, Optimization and Reasoning, with three sub-agents cycling together. This is the core search loop.
- 4a, Math Reasoning and Configuration Prediction. Treats the system as a black box function y = f(x). Uses Bayesian optimization, reinforcement learning, or heuristic and hybrid methods to predict the next configuration x(t+1) worth trying.
- 4b, Transformation. Actually applies the predicted configuration. Example actions named on the slide: quantization, pruning, batch size adjustment, scheduling, scaling replicas.
- 4c, Observability and Reasoning, the feedback agent. Monitors execution after the change, collects runtime metrics, generates (x(t), y(t)) pairs, and decides whether further optimization is required.

These three repeat, 4a predicts, 4b applies, 4c observes and feeds back into 4a, until Agent 4 decides the result is good enough. Every (x, y) pair generated is stored in a Configuration-Performance Dataset, which is reused both to guide future predictions inside the loop and to seed Agent 1 the next time the whole process runs, so the system gets smarter with use.

Agent 8, Deployment and Monitoring. Deploys the chosen configuration, monitors long-term performance, and can reuse a past configuration or trigger a fresh round of optimization if conditions change later.

The existing worked example: LLM Inference KV Cache Optimization

Attached to the Transformation Agent (4b) as a concrete use case, the slide lists adaptive actions specific to LLM serving: change KV cache precision (FP16 to INT8 or FP8), adjust cache allocation size based on active sequence lengths, select cache compression or quantization policies, modify batching behaviour, change token scheduling or request grouping, adapt memory layout or placement of KV cache buffers.

This overlaps directly with material already in this repo: KV cache and PagedAttention in notes/landscape.md, continuous batching, and the precision and quantization ideas in notes/hpc-notes.md.

Supporting evidence the general approach works: ASAP (Google and Meta, 2026)

Dr Alomairy pointed me to a paper called ASAP, an Agentic Solution to Auto-optimize Performance of Large-Scale LLM Training, arXiv 2511.03844, from authors at Google and Meta. Reading it closely, it is structurally the same kind of system as AutoOptAgent, applied to a different stage of the ML lifecycle.

ASAP has four pieces: a Coordinator Agent that orchestrates the workflow, an Analyzer Agent that diagnoses bottlenecks from profiling data and roofline analysis, a Proposal Agent that generates optimized configurations using a knowledge base and a database of past optimizations, and a persistent Sharding Memory that logs everything tried.

Mapped onto AutoOptAgent, this lines up closely: ASAP's Analyzer plays the role of Agent 3 and Agent 4c together, its Proposal Agent plays the role of Agent 4a, its Coordinator plays the role of Agent 4 overall, and its Sharding Memory plays the role of the Configuration-Performance Dataset.

The results are a strong, credible validation of this whole style of system: tested on three real training workloads on TPU pods (compute bound, memory bound, and communication bound), the agent's proposed configuration matched what human performance engineers independently arrived at in all three cases, producing up to 28 percent reduction in training step time on its own, and up to 2.58 times throughput improvement when combined with further human tuning. This is not a small scale demo, it is Google and Meta authors reporting results that match expert humans on real production-scale training jobs.

The precise gap this paper leaves open

Reading closely for what ASAP does not do, three things stand out.

It optimizes training, not inference. AutoOptAgent's own worked example is about inference (the KV cache use case), a different point in the model lifecycle than what ASAP addresses.

It runs entirely on TPU data center hardware. Thermal state is not mentioned anywhere in the paper, not once, in any form. This makes sense given data centers have dedicated cooling infrastructure that a phone does not have, but it means the thermal dimension is simply absent from this line of work, not solved and not considered.

Most directly relevant: in their own Limitations and Future Work section, the ASAP authors write that they intend to expand the agent's scope to, in their words, holistically optimize the trade-offs between on-device computation and network communication. They are naming edge or on-device extension as their own explicitly unsolved future direction.

So ASAP is a real, published, credible demonstration that the agent driven auto-tuning approach behind AutoOptAgent works and gets serious, expert-matching results at scale, while leaving completely untouched exactly the three things this project has spent the most time on: thermal awareness, on-device and edge application, and the choice between accelerator types like NPU versus GPU. That is not a coincidence to note in passing, it is the precise, citable gap this project's research fills.

The gap in AutoOptAgent itself, independent of ASAP

Reading the architecture closely, thermal state does not appear anywhere in her own slide either.

Agent 1's objectives are latency, throughput, memory, energy. Energy is adjacent to heat but is not the same measurement: a configuration can be energy efficient in total joules used while still concentrating that energy into a hotspot that triggers throttling on a small device, which is exactly the kind of thing I measured on my own iPhone.

Agent 4c's job is to monitor runtime metrics and generate feedback pairs, but nothing on the slide names temperature or thermal state as one of those metrics.

Agent 4b's action list, quantization, pruning, batch size, scheduling, scaling, does not include anything like switching which chip runs the workload, or backing off drafting or generation rate specifically because the device is running hot.

This is the gap Dr Alomairy asked me to work on, and it lines up closely with almost everything researched in this project so far, and now also with the gap the ASAP authors themselves flagged as open.

The proposed extension, mapped onto her own agents

Extend Agent 1's objectives and constraints. Add a thermal objective or constraint alongside latency, throughput, memory, and energy, something like avoid throttling or stay under a thermal budget. This is directly supported by my own measurements: about a 10 percent throughput drop on the iPhone 17 Pro and about 2 percent on the iPad Pro under the same test, recorded in 01-edge-benchmarks/DEVICE_TABLE.md.

Extend Agent 4c's observability to include thermal state. iOS and Android both expose a thermal state signal to apps, so this is a real, available input, not something that needs new hardware access. Agent 4c would add this as one more quantity tracked alongside whatever performance metrics it already collects, and it becomes part of the (x, y) pairs stored for future predictions, the same role ASAP's Analyzer plays for training jobs, just with a signal ASAP never uses.

Extend Agent 4b's transformation actions with thermal responses. Three specific candidate actions, all grounded in research already in this repo:
- Switch which chip runs inference, NPU instead of GPU, when thermal pressure is rising. Supported by the roughly four times efficiency difference between NPU and GPU for this kind of workload, and by the CoreML-LLM numbers on my own iPhone 17 Pro model, see notes/research-idea.md.
- Adapt speculative decoding draft length to live thermal state, drafting less aggressively as the device heats up rather than at a fixed rate regardless of conditions. This is my own original proposal, detailed in notes/research-idea.md and notes/speculative-decoding.md.
- Proactively lower NPU or memory frequency before overheating happens, rather than only reacting after the fact. This is EnerInfer's approach, see notes/reading-notes.md, and it operates one level below my own idea: EnerInfer changes hardware configuration, my idea changes algorithm behaviour, and Agent 4b could reasonably do both.

A second worked example, sibling to the existing KV cache one: Thermal-Aware Mobile Inference. Using the same format as her own diagram: monitor device thermal state continuously as a new Agent 4c signal, if thermal pressure rises shift the draft model from GPU to NPU, reduce speculative decoding window size proportionally to thermal severity, if sustained consider offloading more of the workload to the cloud, the same edge and cloud split idea behind WISP now triggered specifically by heat rather than by network conditions or content difficulty alone, and log the outcome back into the Configuration-Performance Dataset so future predictions account for this device's specific thermal behaviour over time.

How this connects to everything else in this project

notes/research-idea.md holds the full thermal and NPU research this extension draws on, including the live commercial examples, Siri AI's edge and cloud split, and HUMAIN and Qualcomm's DVFS-aware NPU scheduler. notes/wisp-summary.md is the source for the edge and cloud split idea extended here into the cloud offload branch of the worked example. notes/hpc-notes.md is the source for why energy and thermal are related but distinct, and why decode is memory bandwidth bound in the first place. 01-edge-benchmarks/DEVICE_TABLE.md holds the actual measurements backing the Agent 1 objective proposal. 02-prefix-caching demonstrates the KV cache reuse principle already present in her existing use case box, from the other direction.

Honest scope of this note

This is my own reading of one presentation slide plus one paper she pointed me to, built to show precisely where my research this project fits into her existing, filed architecture, not a claim about what her actual patent document says or does not say. The mapping onto Agents 1, 4b, and 4c is my proposal for how to extend the framework, not something confirmed as her own plan. The ASAP comparison is my own reading of a public paper, not a claim that she is unaware of it or that it is the only relevant prior work. Worth checking this reading against the real filing, or against her own description of what she wants next, before treating it as settled.

Reference: Ding, Chen, Zhang, and Zhou, ASAP, an Agentic Solution to Auto-optimize Performance of Large-Scale LLM Training, arXiv 2511.03844. https://arxiv.org/pdf/2511.03844
