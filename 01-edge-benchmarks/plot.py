"""
plot.py

Makes charts from the CSVs produced by bench_mlx.py and bench_sustained.py.

Setup:
    pip install matplotlib pandas

Usage:
    python plot.py

Produces:
    results/decode_by_model.png       decode tokens per second per model
    results/ttft_by_model.png         time to first token per model
    results/sustained_throttle.png    throughput over time under sustained load
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"


def short_name(model):
    return model.split("/")[-1].replace("-Instruct-4bit", "")


def plot_decode_by_model():
    path = RESULTS_DIR / "mlx_results.csv"
    if not path.exists():
        print("no mlx_results.csv yet, run bench_mlx.py first")
        return
    df = pd.read_csv(path)
    df["name"] = df["model"].apply(short_name)

    plt.figure(figsize=(7, 4))
    plt.bar(df["name"], df["decode_tps_mean"],
            yerr=df.get("decode_tps_std"), capsize=4, color="#4c72b0")
    plt.ylabel("decode tokens per second")
    plt.title("Decode throughput by model (Apple Silicon, MLX, 4-bit)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out = RESULTS_DIR / "decode_by_model.png"
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_ttft_by_model():
    path = RESULTS_DIR / "mlx_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df["name"] = df["model"].apply(short_name)

    plt.figure(figsize=(7, 4))
    plt.bar(df["name"], df["ttft_s_mean"], color="#dd8452")
    plt.ylabel("time to first token (seconds)")
    plt.title("Time to first token by model (Apple Silicon, MLX, 4-bit)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out = RESULTS_DIR / "ttft_by_model.png"
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_sustained():
    files = list(RESULTS_DIR.glob("sustained_*.csv"))
    if not files:
        print("no sustained_*.csv yet, run bench_sustained.py first")
        return
    plt.figure(figsize=(7, 4))
    for path in files:
        df = pd.read_csv(path)
        label = path.stem.replace("sustained_", "")
        plt.plot(df["elapsed_s"], df["decode_tps"], marker="o", markersize=3, label=label)
    plt.xlabel("elapsed time (seconds)")
    plt.ylabel("decode tokens per second")
    plt.title("Throughput under sustained load (thermal behaviour)")
    plt.legend()
    plt.tight_layout()
    out = RESULTS_DIR / "sustained_throttle.png"
    plt.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(exist_ok=True)
    plot_decode_by_model()
    plot_ttft_by_model()
    plot_sustained()
