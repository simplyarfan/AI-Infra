"""Run the policy comparison on the simulator and write the results.

Usage:  python scripts/run_sim_demo.py

Writes results/sim_comparison.csv and, if matplotlib is installed,
results/sim_phone_severe_traces.png.

This is a simulation. The presets are illustrative, not fitted to a real device.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from thermal_agent import run_comparison  # noqa: E402
from thermal_agent.harness import write_comparison_csv  # noqa: E402

DURATION_S = 600.0


def print_table(preset, res):
    print("\n" + preset)
    print("%-18s %9s %9s %8s %10s %10s %9s %8s" % (
        "policy", "mean tok/s", "tail tok/s", "peak C", "throttled s", "over budget s", "J/token", "changes"))
    for policy, (_rows, m) in res.items():
        print("%-18s %9.1f %9.1f %8.1f %10.0f %10.0f %9.3f %8d" % (
            policy, m.mean_tps, m.tail_tps, m.peak_temp_c, m.time_throttled_s,
            m.time_over_budget_s, m.j_per_token, m.level_changes))


def plot(res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, skipping the plot")
        return
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for policy, (rows, _m) in res.items():
        t = [r["t_s"] for r in rows]
        axes[0].plot(t, [r["temp_c"] for r in rows], label=policy)
        axes[1].plot(t, [r["tok_per_s"] for r in rows], label=policy)
        axes[2].plot(t, [r["level"] for r in rows], label=policy)
    axes[0].axhline(53.0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("temperature (C)")
    axes[0].set_title("Simulated phone under sustained load (illustrative preset)")
    axes[1].set_ylabel("tokens per second")
    axes[2].set_ylabel("power cap level")
    axes[2].set_xlabel("time (s)")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote " + path)


def main():
    out = os.path.join(ROOT, "results")
    os.makedirs(out, exist_ok=True)
    by_preset = {p: run_comparison(p, DURATION_S) for p in ("phone_severe", "tablet_cool")}
    for p, res in by_preset.items():
        print_table(p, res)
    csv_path = os.path.join(out, "sim_comparison.csv")
    write_comparison_csv(csv_path, by_preset)
    print("\nwrote " + csv_path)
    plot(by_preset["phone_severe"], os.path.join(out, "sim_phone_severe_traces.png"))


if __name__ == "__main__":
    main()
