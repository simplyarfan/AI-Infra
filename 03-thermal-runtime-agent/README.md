# Runtime thermal agent for edge LLM inference

A controller that watches an inference workload at runtime and changes the device
configuration (here, the power cap) so the device stays under a thermal budget
while giving up as little throughput as possible. It is written as the thermal
extension to Dr. Rabab's AutoOptAgent, following the structure of ASAP
(arXiv 2511.03844).

## Status, stated plainly

- **Validated:** the control logic, on a simulated device that I wrote. 91 tests pass.
  A mutation check shows the tests catch 9 of 9 deliberately injected bugs.
- **Not validated:** anything on real hardware. The simulator presets are
  illustrative and are not fitted to any real phone or tablet. The NVIDIA backend
  is tested against fake `nvidia-smi` output only. It has never driven a real card.
- **Pending:** the RTX 5060 Ti sweep and live run (instructions below). Until
  those are done, no claim here is about a real device.

The AutoOptAgent mapping below is my reading of one slide. The patent is not
public, so the mapping may be wrong in places.

## The idea in one paragraph

On a phone, decode speed falls after a few minutes because the chip heats up and
the OS throttles it. The throttle is blunt and late. The agent instead learns a
small thermal model of the device online, forecasts where the temperature is
heading for each candidate configuration, and backs off early to the fastest
configuration that is predicted to stay under budget. It only steps back up one
level at a time, and only when the model says the higher level is safe at steady
state. Decode is memory bandwidth bound, so cutting power costs far less speed
than it saves heat. That is why this works at all.

## How it maps to AutoOptAgent

| AutoOptAgent role (my reading) | Module | What it does here |
|---|---|---|
| Agent 1, objectives | `objectives.py` | Latency and throughput goals plus a thermal budget (the part the slide did not have) |
| Agent 2, baseline | `evaluation.py` (`BaselineAgent`) | Measures unthrottled throughput as the reference |
| Agent 3, evaluation | `evaluation.py` (`EvaluationAgent`) | Classifies the state: violation, thermal pressure, headroom, ok |
| Agent 4a, config prediction | `predictor.py` | Picks the fastest configuration whose forecast stays under the limit |
| Agent 4b, transformation | `transformer.py` (`Transformer`) | Applies the change. Lowering is immediate, raising waits a dwell time scaled by switch cost |
| Agent 4c, observability | `observer.py` | Records readings, refits the thermal model online, tracks forecast error |
| Configuration-Performance Dataset | `dataset.py` | Per-configuration throughput, power, J per token, saved as jsonl |
| Agent 8, deployment and monitoring | `transformer.py` (`DeploymentMonitor`) | Detects model drift and triggers re-optimisation |
| The loop | `runtime.py` | `ThermalAgent.tick()`: observe, record, evaluate, propose, apply, log |

Supporting code: `thermal_model.py` (first-order RC model and its least-squares
fit), `knobs.py` (configuration knobs and the DVFS prior), `sim.py` (the simulated
device), `harness.py` (policy comparison), `backends/` (NVIDIA and Ollama).

## Run it

```
pip install pytest matplotlib
python -m pytest                      # 91 tests, under a second
python scripts/mutation_check.py      # injects 9 bugs, expects the suite to fail on each
python scripts/run_sim_demo.py        # writes results/sim_comparison.csv and the trace PNG
```

## What the tests show

| File | Claim it supports |
|---|---|
| `test_thermal_model.py` | The online fit recovers the thermal parameters, tolerates noise and irregular sampling, and refuses to guess when the data cannot identify them |
| `test_simulator.py` | The simulated device behaves like the physics it claims: closed-form steady state, reactive throttling with hysteresis, decode memory bound |
| `test_components.py` | Each agent role in isolation: objectives, dataset, knobs, evaluation states, dwell rules, drift monitor |
| `test_agent_scenarios.py` | End to end: the agent avoids hardware throttling, beats unmanaged full power, matches a hand-tuned static cap, does not thrash, does not hurt a cool device, recovers from wrong priors, adapts to an ambient jump, works from coarse thermal states alone, and is deterministic |
| `test_backends.py` | `nvidia-smi` parsing, throttle-reason bits, the power-limit command and its error handling, all against fake output |
| `tests/mutations.py` + `scripts/mutation_check.py` | The suite is not vacuous: 9 of 9 injected bugs are caught |

## Simulator results (600 s per run)

Simulated, not measured. The presets are my own parameters.

| Preset | Policy | Mean tok/s | Tail tok/s | Peak C | Throttled s | J per token | Level changes |
|---|---|---|---|---|---|---|---|
| phone_severe | fixed full power | 62.1 | 60.4 | 55.3 | 261 | 0.059 | 0 |
| phone_severe | fixed 50 percent | 66.7 | 66.7 | 51.0 | 0 | 0.049 | 0 |
| phone_severe | thermal agent | 66.8 | 66.7 | 51.0 | 0 | 0.049 | 2 |
| tablet_cool | fixed full power | 97.0 | 97.0 | 42.4 | 0 | 0.062 | 0 |
| tablet_cool | fixed 50 percent | 84.0 | 84.0 | 34.4 | 0 | 0.039 | 0 |
| tablet_cool | thermal agent | 96.2 | 97.0 | 42.3 | 0 | 0.060 | 4 |

Honest reading:

- On the hot device the agent beats unmanaged full power: no throttling, higher
  sustained throughput, less energy per token.
- It only matches a hand-tuned 50 percent cap. It does not beat it. The value is
  that it finds that cap by itself, without being told the device is hot.
- On the cool device a fixed 50 percent cap loses about 13 percent throughput.
  The agent stays near full power and gives up about 0.8 percent.

`results/sim_phone_severe_traces.png` shows the three policies over time. The
agent and the fixed 50 percent line overlap closely once the agent has backed off.

## Design choices worth questioning

- **Back off fast, climb slowly.** Backing off may jump several levels. Stepping up
  is one level on one knob, and only when the model-predicted steady state is
  under the limit. A short block follows any back-off so the agent does not retry
  the level that just failed.
- **Dwell scales with switch cost.** Cheap knobs can change often, expensive ones
  (reloading a model, say) are held longer.
- **Coarse-state fallback.** Some platforms expose only a few thermal levels, not a
  temperature. The agent can run from those alone. My recollection is that iOS
  works this way (nominal, fair, serious, critical). I have not verified that.
- **Hardware throttling is only the hardware bits.** The agent's own software power
  cap is not counted as throttling. A test covers this because it is an easy bug.

## Run it on the RTX 5060 Ti (pending)

Needs the NVIDIA driver, Ollama running with the model pulled, and an
administrator terminal, because the agent changes the GPU power limit.

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

## Not done yet

- Real-hardware numbers (GPU first, then iPhone and iPad through a coarse-state backend).
- Fitting the simulator presets to the measured phone and tablet traces from `01-edge-benchmarks`.
- A backend for actual phone knobs (quantization level, context length, draft length).
- Comparing the Bayesian or RL predictor the slide describes against this model-based one.
