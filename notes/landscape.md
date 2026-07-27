# The inference landscape (my working map)

This is my attempt to lay out how the main tools and ideas in AI inference fit
together, so I understand where each one sits rather than treating them as a
list of names. I built this up while reading and going through the vLLM course.

## The one problem underneath all of it

Running a trained model to answer a user is called inference. Doing it for many
users at once, fast and cheap, without wasting the hardware, is the whole game.
Two facts drive almost every design decision:

- Generating text happens in two phases. Prefill reads the whole prompt at once
  and is compute heavy. Decode writes the answer one token at a time and is
  limited by memory bandwidth, not raw compute, because every new token has to
  read the model weights and the growing cache back out of memory.
- The model keeps a running memory of everything it has processed so it does not
  redo that work for every token. This is the KV cache. Most of the clever work
  in serving is really about managing that cache well.

## The engines (run one model on a GPU efficiently)

**vLLM.** The most widely used engine. Two ideas made it fast. PagedAttention
stores the KV cache in small fixed size blocks handed out on demand, the same
way an operating system hands out memory pages, so very little memory is wasted
and many more users fit on one GPU. Continuous batching works at the token level:
the moment one request finishes, a waiting request takes its place, so the GPU
never sits idle waiting for the slowest one in a group.

**SGLang.** Another engine, a direct alternative to vLLM. Its headline feature is
RadixAttention, which is automatic prefix caching. It keeps the KV cache in a
radix tree (a trie where each edge can stand for a whole run of tokens). When a
new request shares an opening with something already computed, SGLang finds the
longest matching prefix in the tree and reuses that work instead of recomputing
it. It pairs this with cache aware scheduling (it orders the queue to group
requests that share prefixes) and an LRU eviction policy that throws away the
least recently used leaves first, which keeps shared parents alive.

Where it shines: workloads where many requests share text. A long shared system
prompt, few shot examples repeated across calls, multi turn chat where the
history keeps growing, RAG where the same document is fed in again and again. On
those, the public numbers show large throughput gains over recomputing from
scratch. On fully unique prompts there is basically nothing to reuse, so the
benefit disappears. This is exactly the principle I demonstrate in project 02.

The difference from vLLM's own prefix caching is mostly matching granularity.
vLLM hashes fixed size blocks and reuses them when the hashes line up. SGLang's
tree does token level longest prefix matching at any depth, which handles
branching and growing histories more naturally.

## The cluster layer (run many engines across many machines)

**llm-d.** Not an engine. It is a Kubernetes native layer that sits on top of
engines like vLLM and coordinates a whole cluster. It became a CNCF Sandbox
project in March 2026, backed by Red Hat, Google, IBM, NVIDIA, AMD and others.

Its three main ideas:
- Disaggregated prefill and decode. Because prefill is compute bound and decode
  is bandwidth bound, they fight each other on a single machine. llm-d runs them
  as separate pools that scale independently, and ships the KV cache from the
  prefill side to the decode side.
- Cache aware routing. Instead of sending each request to a random machine, it
  routes to whichever instance already holds the relevant prefix in cache, so
  the hit rate goes up.
- Multi tier KV cache. Prefixes can live in GPU memory, spill to CPU memory, or
  go to storage, so more can be kept around.

The public write ups report meaningful throughput gains and time to first token
reductions from cache aware routing and from splitting prefill and decode. The
often quoted "40 percent throughput, 30 percent cost" style numbers mostly come
from Google's gateway and KV offload posts sitting next to llm-d rather than one
single clean llm-d benchmark, so I am careful about how I attribute that.

This is the same prefill and decode split that shows up in the WISP paper, just
at cluster scale instead of edge and cloud scale.

## Benchmarking

**GuideLLM** (Red Hat / Neural Magic) is the standard open tool for load testing
an inference server that speaks the OpenAI API. You point it at a server, pick a
request rate or a sweep, and it reports throughput and latency under load. It is
a server side tool, so it fits the vLLM and SGLang world more than on device.

**llama-bench** ships with llama.cpp and is the go to for local measurements. It
reports prompt processing speed and token generation speed and is good for before
and after comparisons when you change a setting.

The metrics that actually matter, on any hardware:
- Time to first token. Dominates how responsive it feels.
- Decode tokens per second. Steady state generation speed.
- Prefill tokens per second. How fast prompts get ingested, matters for long
  context and RAG.
- Peak memory. Decides what fits.
- Energy per token. Matters a lot on battery powered devices.

Good practice I am following in my own benchmarks: warm up runs before measuring,
several runs with mean and spread, unique prompts so prefix caching does not
secretly inflate the numbers, and logging exact model, quantization and settings.

## The compiler layer

**MLIR** (Multi Level Intermediate Representation, part of LLVM) is the compiler
infrastructure used to build the compilers that turn a model into fast code for
many different chips. Its core idea is dialects: the same program can be written
at several levels of abstraction (high level tensor ops down to hardware specific
ops down to machine code) and progressively lowered from one level to the next.
It is the plumbing under things like OpenXLA and StableHLO and IREE, which is the
path people use to deploy models onto edge and mobile hardware. I keep this at a
one paragraph understanding on purpose; it is a deep compiler topic and I would
rather be honest about the depth than pretend more.

## Where HPC ties in

See hpc-notes.md. The short version: the questions high performance computing has
always asked (is this compute bound or memory bound, what is the roofline, where
does the bandwidth run out, how do we use lower precision without losing accuracy)
are exactly the questions that decide inference performance. My edge benchmarks
answer some of these directly on the hardware I own.
