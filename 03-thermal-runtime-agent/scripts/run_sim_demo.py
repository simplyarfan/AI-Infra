"""Run the policy comparisons on the simulator and write the results.

Usage:  python scripts/run_sim_demo.py

Writes into results/:
  sim_comparison.csv                  power cap device, two presets
  sim_model_level_comparison.csv      phone with token pacing, quantization, KV precision
  sim_phone_severe_traces.png         power cap device, temperature and speed over time
  sim_phone_model_knobs_traces.png    model level knobs, both scenarios

This is a simulation. The presets and scenarios are illustrative, not fitted to
any real device.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from thermal_agent import Objectives, objective_value, run_comparison, run_model_level_comparison  # noqa: E402
from thermal_agent.harness import write_comparison_csv  # noqa: E402

DURATION_S = 600.0
MODEL_DURATION_S = 900.0
OBJ = Objectives()
REF_TPS = 77.0   # simulated phone base throughput at the top configuration

# Validated categorical slots 1 to 3 (blue, orange, aqua). Line style is a second
# encoding because aqua has low contrast on white.
STYLE = {
    "fixed_full_power": ("#2a78d6", "-", 1.3),
    "fixed_top": ("#2a78d6", "-", 1.3),
    "fixed_50_percent": ("#eb6834", "--", 1.3),
    "best_fixed_with_hindsight": ("#eb6834", "--", 1.3),
    "thermal_agent": ("#1baf7a", "-", 2.2),
}


def print_table(title, res, with_quality=False):
    print("\n" + title)
    cols = "%-27s %9s %10s %7s %10s %13s %8s %8s" % (
        "policy", "mean tok/s", "tail tok/s", "peak C", "throttled s", "over budget s", "J/token", "changes")
    print(cols + ("  quality  objective" if with_quality else ""))
    for policy, (_rows, m) in res.items():
        line = "%-27s %9.1f %10.1f %7.1f %10.0f %13.0f %8.3f %8d" % (
            policy, m.mean_tps, m.tail_tps, m.peak_temp_c, m.time_throttled_s,
            m.time_over_budget_s, m.j_per_token, m.level_changes)
        if with_quality:
            line += "   %.3f   %.3f" % (m.mean_quality, objective_value(m, REF_TPS, OBJ))
        print(line)
    if with_quality:
        print("(objective = mean tok/s / %.0f + %.1f * quality. Compare only policies with 0 throttled s.)"
              % (REF_TPS, OBJ.quality_weight))


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        print("matplotlib not available, skipping the plots")
        return None


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.7)
    ax.set_axisbelow(True)


def _end_label(ax, t, y, text, color):
    ax.annotate(text, (t[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color="#52514e")


def plot_power_cap(res, path):
    plt = _plt()
    if plt is None:
        return
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for policy, (rows, _m) in res.items():
        color, ls, lw = STYLE[policy]
        t = [r["t_s"] for r in rows]
        for ax, key in zip(axes, ("temp_c", "tok_per_s", "level")):
            ax.plot(t, [r[key] for r in rows], color=color, linestyle=ls, linewidth=lw, label=policy)
    for ax in axes:
        _style_axes(ax)
    axes[0].axhline(OBJ.max_temp_c, color="#8a8984", linestyle=":", linewidth=0.9)
    axes[0].text(2, OBJ.max_temp_c + 0.4, "thermal budget", fontsize=8, color="#52514e")
    axes[0].set_ylabel("temperature (C)")
    axes[0].set_title("Simulated device with a power cap, sustained load (illustrative preset)", fontsize=10)
    axes[1].set_ylabel("tokens per second")
    axes[2].set_ylabel("power cap level")
    axes[2].set_xlabel("time (s)")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.text(0.01, 0.005, "The agent and the fixed 50 percent line overlap once the agent has backed off.",
             fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    print("wrote " + path)


def plot_model_knobs(by_scenario, path):
    plt = _plt()
    if plt is None:
        return
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    titles = {"phone_quant_flat": "quantization does not cool (watts unchanged)",
              "phone_quant_cools": "quantization also cools (watts fall)"}
    for col, (scenario, res) in enumerate(by_scenario.items()):
        for policy, (rows, _m) in res.items():
            color, ls, lw = STYLE[policy]
            t = [r["t_s"] for r in rows]
            for row, key in enumerate(("temp_c", "tok_per_s", "quality")):
                axes[row][col].plot(t, [r[key] for r in rows], color=color, linestyle=ls,
                                    linewidth=lw, label=policy)
        axes[0][col].axhline(OBJ.max_temp_c, color="#8a8984", linestyle=":", linewidth=0.9)
        axes[0][col].set_title(titles[scenario], fontsize=10)
        axes[2][col].set_xlabel("time (s)")
        for row in range(3):
            _style_axes(axes[row][col])
    axes[0][0].set_ylabel("temperature (C)")
    axes[1][0].set_ylabel("tokens per second")
    axes[2][0].set_ylabel("quality (relative)")
    axes[0][0].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Simulated phone, model level knobs only (illustrative scenarios)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    print("wrote " + path)


def main():
    out = os.path.join(ROOT, "results")
    os.makedirs(out, exist_ok=True)

    by_preset = {p: run_comparison(p, DURATION_S) for p in ("phone_severe", "tablet_cool")}
    for p, res in by_preset.items():
        print_table("power cap device, preset " + p, res)
    write_comparison_csv(os.path.join(out, "sim_comparison.csv"), by_preset)

    by_scenario = {s: run_model_level_comparison(s, MODEL_DURATION_S, objectives=OBJ)
                   for s in ("phone_quant_flat", "phone_quant_cools")}
    for s, res in by_scenario.items():
        print_table("model level knobs, scenario " + s, res, with_quality=True)
    write_comparison_csv(os.path.join(out, "sim_model_level_comparison.csv"), by_scenario)

    plot_power_cap(by_preset["phone_severe"], os.path.join(out, "sim_phone_severe_traces.png"))
    plot_model_knobs(by_scenario, os.path.join(out, "sim_phone_model_knobs_traces.png"))
    print("\nwrote the CSVs in " + out)


if __name__ == "__main__":
    main()
