"""Run the thermal agent live on an NVIDIA GPU with an Ollama workload.

NOT YET RUN ON REAL HARDWARE. The pieces it uses are unit tested against fake
nvidia-smi output, but this script has never driven a real card. Expect to fix
something on the first run.

Requirements: NVIDIA driver, Ollama running with the model pulled, and an
administrator terminal (the agent changes the GPU power limit).

Usage (from 03-thermal-runtime-agent):
    python scripts/run_nvidia_agent.py --model qwen2.5:7b --duration 600 --max-temp 70

Optionally build the knob from a measured sweep instead of the DVFS prior:
    python scripts/run_nvidia_agent.py --sweep-requests ../01-edge-benchmarks/results/gpu_thermal_requests.csv \
        --sweep-telemetry ../01-edge-benchmarks/results/gpu_thermal_telemetry.csv

The agent always restores the default power limit when it exits.
"""
import argparse
import csv
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from thermal_agent import (  # noqa: E402
    AgentConfig, Objectives, ThermalAgent, ThermalParams, knob_from_sweep,
)
from thermal_agent.backends.nvidia import (  # noqa: E402
    NvidiaPowerLimitActuator, NvidiaTelemetry, build_power_knob, default_runner,
)
from thermal_agent.backends.ollama import BackgroundWorkload, ollama_generate  # noqa: E402

PROMPT = ("Write a long, detailed technical explanation of how a GPU executes a matrix "
          "multiplication, covering the memory hierarchy, tiling, and tensor cores. "
          "Keep going with more and more detail.")


def power_limits():
    rc, out, err = default_runner(["--query-gpu=power.default_limit,power.min_limit,power.max_limit",
                                   "--format=csv,noheader,nounits"])
    if rc != 0:
        raise SystemExit("could not read power limits: " + err)
    d, lo, hi = [float(x) for x in out.splitlines()[0].split(",")]
    return d, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--max-temp", type=float, default=70.0)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--horizon", type=float, default=120.0)
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--r-th", type=float, default=0.3, help="prior thermal resistance, C per W")
    ap.add_argument("--tau", type=float, default=40.0, help="prior thermal time constant, s")
    ap.add_argument("--sweep-requests")
    ap.add_argument("--sweep-telemetry")
    args = ap.parse_args()

    default_w, min_w, max_w = power_limits()
    if args.sweep_requests and args.sweep_telemetry:
        knob = knob_from_sweep(args.sweep_requests, args.sweep_telemetry, name="power_limit")
        watts = [int(re.search(r"\d+", l.label).group()) for l in knob.levels]
        idle = 0.0
    else:
        knob, watts = build_power_knob(default_w, min_w, max_w, args.levels)
        idle = 0.0
    print("power levels (W): %s" % watts)

    workload = BackgroundWorkload(lambda: ollama_generate(args.model, PROMPT))
    workload.start()
    t0 = time.time()
    while workload.tok_per_s() == 0.0 and time.time() - t0 < 120:
        if workload.error:
            raise SystemExit("workload failed: %r" % (workload.error,))
        time.sleep(0.5)

    telemetry = NvidiaTelemetry(workload.tok_per_s)
    actuator = NvidiaPowerLimitActuator(knob, watts)
    cfg = AgentConfig(
        objectives=Objectives(max_temp_c=args.max_temp, temp_margin_c=args.margin, horizon_s=args.horizon),
        prior=ThermalParams(t_amb=30.0, r_th=args.r_th, tau_s=args.tau),
        idle_power_w=idle,
    )
    agent = ThermalAgent(telemetry, [actuator], cfg)
    out = os.path.join(ROOT, "results")
    os.makedirs(out, exist_ok=True)
    try:
        end = time.time() + args.duration
        while time.time() < end:
            d = agent.tick()
            o = telemetry.read()
            print("t=%5.0fs temp=%4.1fC power=%5.1fW tok/s=%5.1f level=%s status=%s %s" % (
                d.t_s, o.temp_c or 0.0, o.power_w, o.tok_per_s, d.after, d.status, d.note))
            time.sleep(args.interval)
    finally:
        workload.stop()
        try:
            default_runner(["-pl", str(int(round(default_w)))])
            print("power limit restored to %d W" % int(round(default_w)))
        except Exception as e:
            print("could not restore the power limit, do it manually: %r" % (e,))
        agent.dataset.save(os.path.join(out, "live_gpu_dataset.jsonl"))
        with open(os.path.join(out, "live_gpu_decisions.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "status", "applied", "before", "after", "forecast_c", "note"])
            for d in agent.decisions:
                w.writerow([round(d.t_s, 1), d.status, int(d.applied), d.before, d.after,
                            None if d.forecast_c is None else round(d.forecast_c, 1), d.note])
        print("wrote results/live_gpu_dataset.jsonl and results/live_gpu_decisions.csv")


if __name__ == "__main__":
    main()
