"""
gpu_thermal_sweep.py

Runtime thermal telemetry and power limit sweep for an NVIDIA GPU, using a local
Ollama model as the workload.

NOT YET RUN ON REAL HARDWARE. Syntax checked only. Expect to fix something on the
first run, and do not read any result as final until it has been run.

What it does, per configuration (a power limit as a fraction of the card's default):
  - sets the GPU power limit (needs an administrator terminal on Windows)
  - generates continuously for SECONDS_PER_CONFIG seconds
  - in the background, samples nvidia-smi once a second: temperature, SM clock,
    memory clock, power draw, utilisation, and the throttle reason bitmask
  - records tokens per second for every generation request
  - waits for the card to cool down before the next configuration

Outputs, in results/:
  gpu_thermal_telemetry.csv   one row per nvidia-smi sample
  gpu_thermal_requests.csv    one row per generation request

The question this answers: how much decode speed do we give up per watt and per
degree? If decode is memory bandwidth bound, a lower power limit should cost very
little speed while cutting power and heat. That measured tradeoff is the raw
material for a runtime controller.

Setup:
    Ollama installed and running, and the model pulled:  ollama pull qwen2.5:7b
    NVIDIA driver installed (nvidia-smi on PATH)
    Run from an administrator terminal so that nvidia-smi -pl is allowed.

Usage:
    python gpu_thermal_sweep.py
"""

import csv
import json
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b"

PROMPT = (
    "Write a long, detailed technical explanation of how a GPU executes a matrix "
    "multiplication, covering the memory hierarchy, tiling, and tensor cores. "
    "Keep going with more and more detail."
)
NUM_PREDICT = 400

POWER_FRACTIONS = [1.0, 0.85, 0.70, 0.55]   # of the card's default power limit
SECONDS_PER_CONFIG = 180
SAMPLE_INTERVAL_S = 1.0
COOLDOWN_TARGET_C = 50
COOLDOWN_MAX_S = 300

RESULTS_DIR = Path(__file__).parent / "results"

FIELDS_NEW = "temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,clocks_event_reasons.active"
FIELDS_OLD = "temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,clocks_throttle_reasons.active"


def smi(args):
    r = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def pick_fields():
    """Newer drivers renamed the throttle reasons field, so try both."""
    for f in (FIELDS_NEW, FIELDS_OLD):
        rc, out, _ = smi(["--query-gpu=" + f, "--format=csv,noheader,nounits"])
        if rc == 0 and out:
            return f
    raise RuntimeError("nvidia-smi query failed. Is the NVIDIA driver installed and nvidia-smi on PATH?")


def parse_float(s):
    try:
        return float(s)
    except ValueError:
        return None


def sample(fields):
    rc, out, _ = smi(["--query-gpu=" + fields, "--format=csv,noheader,nounits"])
    if rc != 0 or not out:
        return None
    p = [x.strip() for x in out.splitlines()[0].split(",")]
    return {
        "temp_c": parse_float(p[0]),
        "sm_mhz": parse_float(p[1]),
        "mem_mhz": parse_float(p[2]),
        "power_w": parse_float(p[3]),
        "util_pct": parse_float(p[4]),
        "throttle_reasons": p[5],
    }


def get_power_limits():
    rc, out, err = smi(["--query-gpu=power.default_limit,power.min_limit,power.max_limit",
                        "--format=csv,noheader,nounits"])
    if rc != 0:
        raise RuntimeError("could not read power limits: " + err)
    d, lo, hi = [float(x) for x in out.splitlines()[0].split(",")]
    return d, lo, hi


def set_power_limit(watts):
    rc, out, err = smi(["-pl", str(int(round(watts)))])
    return rc == 0, (out or err)


def generate():
    body = json.dumps({
        "model": MODEL, "prompt": PROMPT, "stream": False, "keep_alive": "30m",
        "options": {"num_predict": NUM_PREDICT, "temperature": 0},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    tokens = d.get("eval_count", 0)
    secs = d.get("eval_duration", 0) / 1e9
    return tokens, (tokens / secs if secs > 0 else 0.0)


class Sampler(threading.Thread):
    def __init__(self, fields, label, t0):
        super().__init__(daemon=True)
        self.fields, self.label, self.t0 = fields, label, t0
        self.rows = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            s = sample(self.fields)
            if s:
                s["t_s"] = round(time.time() - self.t0, 1)
                s["config"] = self.label
                self.rows.append(s)
            self._halt.wait(SAMPLE_INTERVAL_S)

    def stop(self):
        self._halt.set()


def cooldown(fields):
    start = time.time()
    while time.time() - start < COOLDOWN_MAX_S:
        s = sample(fields)
        if s and s["temp_c"] is not None and s["temp_c"] <= COOLDOWN_TARGET_C:
            return
        time.sleep(5)


def run_config(label, fields, telemetry, requests_out):
    t0 = time.time()
    sampler = Sampler(fields, label, t0)
    sampler.start()
    while time.time() - t0 < SECONDS_PER_CONFIG:
        tokens, tps = generate()
        t = round(time.time() - t0, 1)
        requests_out.append({"config": label, "t_s": t, "tokens": tokens, "tok_per_s": round(tps, 2)})
        print("  " + label + "  t=" + str(t) + "s  " + str(round(tps, 1)) + " tok/s")
    sampler.stop()
    sampler.join()
    telemetry.extend(sampler.rows)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def summarise(labels, telemetry, requests_out):
    print("\nSummary")
    print("config        first3 tok/s  last3 tok/s  mean temp C  mean power W  J per token")
    for label in labels:
        reqs = [r["tok_per_s"] for r in requests_out if r["config"] == label]
        tele = [t for t in telemetry if t["config"] == label]
        first, last, avg = mean(reqs[:3]), mean(reqs[-3:]), mean(reqs)
        temp, power = mean([t["temp_c"] for t in tele]), mean([t["power_w"] for t in tele])
        jpt = power / avg if avg > 0 else 0.0
        print("%-12s  %12.1f  %11.1f  %11.1f  %12.1f  %11.2f" % (label, first, last, temp, power, jpt))


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    fields = pick_fields()
    default_w, min_w, max_w = get_power_limits()
    print("Power limit: default %.0f W, min %.0f W, max %.0f W" % (default_w, min_w, max_w))

    print("Warming up the model ...")
    generate()

    telemetry, requests_out, labels = [], [], []
    try:
        for frac in POWER_FRACTIONS:
            watts = max(min_w, min(max_w, default_w * frac))
            label = "%dW" % int(round(watts))
            ok, msg = set_power_limit(watts)
            if not ok:
                print("Could not set %s (%s). Run from an administrator terminal. Skipping." % (label, msg))
                continue
            print("\nConfig " + label + " (" + str(int(frac * 100)) + " percent of default)")
            labels.append(label)
            run_config(label, fields, telemetry, requests_out)
            cooldown(fields)
    finally:
        set_power_limit(default_w)
        print("\nPower limit restored to default.")

    write_csv(RESULTS_DIR / "gpu_thermal_telemetry.csv", telemetry,
              ["config", "t_s", "temp_c", "sm_mhz", "mem_mhz", "power_w", "util_pct", "throttle_reasons"])
    write_csv(RESULTS_DIR / "gpu_thermal_requests.csv", requests_out,
              ["config", "t_s", "tokens", "tok_per_s"])
    print("Wrote results/gpu_thermal_telemetry.csv and results/gpu_thermal_requests.csv")
    summarise(labels, telemetry, requests_out)


if __name__ == "__main__":
    main()
