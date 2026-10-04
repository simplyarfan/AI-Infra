# Thermal-aware AutoOptAgent for edge devices

The AutoOptAgent lifecycle, instantiated for a new use case: keeping an edge
device under its thermal budget while a model runs. On Dr. Rabab's slide the use
case box is "LLM Inference KV Cache Optimization". This is the sibling use case,
"thermal-aware mobile inference", built with the same agents and the same loop
(see `notes/autoopt-thermal-extension.md` for the reasoning behind it).

## Status, stated plainly

- **Validated:** the lifecycle and its control logic, on a simulated device that I
  wrote. 148 tests pass. A mutation check shows the tests catch 18 of 18 injected
  bugs, none of them by crashing.
- **Not validated:** anything on real hardware. The simulator's presets and
  scenarios are illustrative and not fitted to any real phone, tablet or GPU. The
  knob priors (how much a quantization level cools or speeds up) and the quality
  numbers are placeholders. The NVIDIA backend is tested against fake
  `nvidia-smi` output only.
- **Source of the architecture:** my reading of one slide. The patent is not
  public, so the mapping may be wrong in places.

## The lifecycle, as code

```
 1 Objectives ---> 2 Baseline ---> 3 Evaluation ---> gate: is optimization needed?
 (thermal budget,  (starts from a    (states: stable,        |
  quality floor)    prior config)     underutilization,      +-- no  --> 8 Deploy / monitor (hold)
        ^                             overload, instability,  |
        |                             constraint violation)   +-- yes --> 4 Optimization and reasoning
        |                                                                  |  coordinates, manages explore/exploit
        |                                                                  v
        |                                                      4a predict --> 4b apply --> 4c observe
        |                                                      (surrogate +    (knobs)      (records, thermal
        |                                                       thermal model)               model, drift error)
        |                                                          ^                              |
        |                                                          +------------------------------+
        |                                                                                         v
        +---------------- Configuration-Performance Dataset <--------------------------- (x_t, y_t) pairs
                          (trains 4a, supplies the prior config for 2, reused by 8)
```

| Slide box | Module | What it does here |
|---|---|---|
| 1 Objectives and constraints | `objectives.py` | Thermal budget, quality floor, and the objective function (`tok/s / reference + quality_weight * quality`) |
| 2 Baseline | `evaluation.py` (`BaselineAgent`, `known_good_config`) | Runs the starting configuration, which is the best known one from the dataset if there is one |
| 3 Evaluation + gate | `evaluation.py` (`EvaluationAgent`) | Forecasts temperature, names the state, and decides whether optimization is needed |
| 4 Optimization and reasoning | `optimizer.py` | Turns the verdict into a mode, calls 4a and 4b, manages explore versus exploit, vets any advisor |
| 4a Configuration prediction | `predictor.py` over `surrogate.py` | Bayesian model of power and speed per configuration, thermal forecast, ranks candidates by the objective |
| 4b Transformation | `transformer.py` (`Transformer`) | Applies knob changes. Lowering is immediate. Raising waits a dwell time scaled by switch cost |
| 4c Observability and feedback | `observer.py` | Records every (configuration, outcome) pair, identifies the thermal model online, measures forecast error |
| 8 Deployment and monitoring | `transformer.py` (`DeploymentMonitor`) | Detects drift (conditions changed, so re-optimize) and reuses the known good configuration at startup |
| Configuration-Performance Dataset | `dataset.py` | jsonl records. Trains 4a, supplies the prior configuration to Agent 2, reused across sessions |
| the loop | `runtime.py` (`ThermalAgent.tick`) | One pass around the diagram. Each `Decision` carries the stages it went through |

See the path as running code: `python scripts/run_lifecycle_demo.py`. One tick that
needs action prints as
`4c observe -> 8 monitor -> 3 evaluate (overload) -> gate: optimization needed -> 4 plan (back_off) -> 4a predict -> 4b apply`,
and a stable tick as `... -> 3 evaluate (stable) -> gate: not needed -> 8 deploy and hold`.

## What the agent can change (4b's actions)

| Knob | Kind | Switch cost | Needs |
|---|---|---|---|
| `token_pacing` | platform | cheap | any app, including iOS |
| `quantization` | model | medium | several model files on the device |
| `kv_precision` | model | medium | runtime support for a quantized KV cache |
| `power_cap` | platform | cheap | a platform API such as `nvidia-smi -pl`, not available to iOS apps |

Knobs are ordered from coolest to the user's preferred setting, each with a prior
on power, speed and quality relative to the top level. Quantization is the
interesting one: fewer bits always speeds decode up (fewer bytes read per token),
but whether the watts fall is an empirical question about each device. The
prior assumes some cooling, the simulator has a scenario where there is none, and
the agent has to cope with both.

## How 4a decides

1. A Bayesian log-linear surrogate predicts power and tokens per second for any
   configuration. It starts at the knob priors and updates on every unthrottled
   observation, with slow forgetting. What it learns about one level carries over
   to configurations it has not tried.
2. A first-order thermal model, identified online by least squares, turns a
   predicted power into a forecast temperature over a planning horizon.
3. Under thermal pressure it may jump several levels: among configurations that
   forecast under the limit and meet the quality floor, take the highest value of
   the objective function. This is the expensive mistake to avoid.
4. With headroom it explores: one knob, one level, only if the predicted steady
   state is under the limit and the objective improves. After a back off, returning
   to that power level is blocked for a while, and the block doubles each time an
   exploration fails (Agent 4's explore/exploit rule).
5. It never proposes a raise that 4b would hold, so the configuration that was
   vetted is the one that gets applied.

An optional **advisor** (for example an LLM on a slower timescale, or off the
device) can suggest a configuration to Agent 4. Nothing it says is applied
without passing the same rules: no raising power while overheating, only
single-step exploration, thermal forecast and quality floor respected. The
default is no advisor, and the tests use scripted advisors including a reckless
one. No LLM is wired in.

## Run it

```
pip install pytest matplotlib
python -m pytest                        # 148 tests, a couple of seconds
python scripts/mutation_check.py        # injects 18 bugs, expects the suite to fail on each
python scripts/run_sim_demo.py          # tables, CSVs and charts in results/
python scripts/run_lifecycle_demo.py    # the stage path of each decision
```

## What the tests show

| File | Tests | Claim it supports |
|---|---|---|
| `test_thermal_model.py` | 10 | The online fit recovers the thermal parameters, tolerates noise and irregular sampling, and refuses to guess when the data cannot identify them |
| `test_simulator.py` | 11 | The simulated device obeys the physics it claims: closed-form steady state, reactive throttling with hysteresis, decode memory bound |
| `test_surrogate_and_knobs.py` | 14 | 4a's model equals the priors before data, learns that a knob does not cool when the data says so, carries that to unvisited configurations, forgets when the device changes. The knob priors are internally consistent |
| `test_components.py` | 28 | Each agent in isolation: objectives, dataset, knobs, evaluation states, dwell rules, drift monitor |
| `test_lifecycle.py` | 27 | The slide stage by stage: Agent 3 states and gate, Agent 2 prior configuration, Agent 4 modes, failed-exploration penalty and advisor vetting, every tick takes a legal path, Agent 8 drift and reuse, warm start |
| `test_agent_scenarios.py` | 21 | Power cap device end to end: avoids throttling, beats unmanaged full power, matches a hand-tuned cap, does not thrash, does not hurt a cool device, recovers from wrong priors, adapts to an ambient jump, works from coarse states alone |
| `test_model_level_scenarios.py` | 16 | Phone with model-level knobs end to end, in both quantization scenarios: no throttling, close to the best fixed configuration chosen with hindsight, quality floor never violated, works with pacing alone, works from coarse states |
| `test_backends.py` | 21 | `nvidia-smi` parsing, throttle-reason bits, power-limit command and its error handling, all against fake output |
| `tests/mutations.py` | 18 mutations | The suite is not vacuous: each injected bug is caught |

## Results (simulated)

**Phone with model-level knobs** (900 s, no power cap, pacing + quantization + KV precision):

| Scenario | Policy | Mean tok/s | Peak C | Throttled s | Mean quality | Changes |
|---|---|---|---|---|---|---|
| quantization does not cool | fixed top | 61.5 | 55.3 | 405 | 1.000 | 0 |
| quantization does not cool | best fixed, chosen with hindsight | 41.7 | 46.6 | 0 | 0.970 | 0 |
| quantization does not cool | thermal agent | 44.2 | 47.7 | 0 | 0.962 | 4 |
| quantization also cools | fixed top | 61.5 | 55.3 | 405 | 1.000 | 0 |
| quantization also cools | best fixed, chosen with hindsight | 80.6 | 50.3 | 0 | 0.915 | 0 |
| quantization also cools | thermal agent | 74.5 | 51.8 | 0 | 0.946 | 3 |

**Device with a power cap** (600 s, the first version of this folder):

| Preset | Policy | Mean tok/s | Peak C | Throttled s | J per token |
|---|---|---|---|---|---|
| phone_severe | fixed full power | 62.1 | 55.3 | 261 | 0.059 |
| phone_severe | fixed 50% cap | 66.7 | 51.0 | 0 | 0.049 |
| phone_severe | thermal agent | 66.8 | 51.0 | 0 | 0.049 |
| tablet_cool | fixed full power | 97.0 | 42.4 | 0 | 0.062 |
| tablet_cool | fixed 50% cap | 84.0 | 34.4 | 0 | 0.039 |
| tablet_cool | thermal agent | 96.2 | 42.3 | 0 | 0.060 |

Charts: `results/sim_phone_model_knobs_traces.png`, `results/sim_phone_severe_traces.png`.
CSVs: `results/sim_model_level_comparison.csv`, `results/sim_comparison.csv`.

Honest reading:

- **The agent finds the right configuration without being told.** In the flat
  scenario it ends on exactly the configuration the hindsight oracle picks. In the
  cooling scenario its objective value is within about 1 percent of the oracle's.
  It does not beat a configuration chosen with full knowledge of the device.
- **Staying under the budget is not free.** With only linear levers (pacing) and a
  quantization that does not cool, the unmanaged phone averages more tokens per
  second (61.5 against 44.2) because the OS governor throttles hard but runs flat
  out in between. The agent buys a device that stays about 8 C cooler, never
  throttles and has steady latency. Whether that trade is wanted is a product
  decision, set by the thermal budget in Agent 1.
- **It beats the governor only if a model knob cools efficiently.** In the cooling
  scenario the agent is about 21 percent faster than unmanaged and stays in
  budget. On a device with a DVFS power cap (the first table) it is also ahead,
  because power falls much faster than speed. Which case a real phone is in is the
  thing to measure.
- **Learning has a visible cost.** In the first minute or two the agent sometimes
  gives up more quality than it needs, because it acts on priors that turn out to
  be wrong (the dip in the quality panel of the chart).
- **The surrogate is only partly identified.** A controller that moves one knob at
  a time and sits on one configuration for a long time cannot separate the effects
  of knobs that always changed together. The closed loop corrects for this, and
  the agent still settles correctly, but the learned per-knob effects should not
  be read as measurements.

## What is not built

- **Real hardware.** Nothing here has run on a phone, a tablet or a GPU. Next:
  `01-edge-benchmarks/gpu_thermal_sweep.py`, then `scripts/run_nvidia_agent.py`.
- **Actuators for real devices beyond the NVIDIA power limit.** Token pacing,
  quantization swap and KV precision exist as simulated knobs. The Ollama and
  llama.cpp hooks that would drive them are not written.
- **The other actions in the extension note:** switching the workload between GPU
  and NPU, adapting speculative decoding draft length, and offloading to the
  cloud. The knob structure supports them (a chip switch is a high switch cost
  knob), but they have no priors, no simulator physics and no tests.
- **Fitting the simulator to the measured phone and tablet traces** in
  `01-edge-benchmarks`. Until then the scenarios are scenarios.
- **A learned 4a policy** (Bayesian optimization over a larger space, or RL). The
  surrogate is Bayesian but the decision rule is a model-predictive heuristic.
- **Quality measurement.** The agent cannot grade an answer. Quality numbers are
  inputs supplied by whoever knows the model, ideally from an offline evaluation.
- **No LLM advisor** is connected, only the interface and the safety vetting.

## Run it on the RTX 5060 Ti (pending)

Needs the NVIDIA driver, Ollama running with the model pulled, and an
administrator terminal, because the agent changes the GPU power limit. Only the
power cap knob is available on this path.

```
ollama pull qwen2.5:7b

# 1. Measure the real power-versus-speed tradeoff
cd 01-edge-benchmarks
python gpu_thermal_sweep.py

# 2. Run the agent live, with the knob built from the measured sweep
cd ../03-thermal-runtime-agent
python scripts/run_nvidia_agent.py --model qwen2.5:7b --duration 600 --max-temp 70 \
    --sweep-requests ../01-edge-benchmarks/results/gpu_thermal_requests.csv \
    --sweep-telemetry ../01-edge-benchmarks/results/gpu_thermal_telemetry.csv
```

The live script always restores the default power limit when it exits. It has not
run on hardware, so expect to fix something on the first run. A 5060 Ti with a
good cooler may never get hot enough to throttle. If so, that is a real result,
and a lower `--max-temp` will still exercise the control loop.
