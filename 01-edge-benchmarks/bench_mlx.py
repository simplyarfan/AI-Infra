"""
bench_mlx.py

Benchmarks local LLM inference on Apple Silicon using MLX.
Measures, per model:
  - time to first token (seconds)
  - decode throughput (tokens per second, steady state generation)
  - prefill throughput (prompt tokens processed per second)
  - peak memory used (GB)

Run several prompts, discard a warmup, and report mean and spread.

Setup:
    pip install mlx-lm

Usage:
    python bench_mlx.py

Notes:
  - Uses the streaming generator so we can time the first token separately.
  - Keep the prompt and generation length fixed across models so the numbers
    are comparable.
"""

import time
import csv
import statistics
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

# Models to benchmark. These are 4-bit quantized community builds that fit in
# 16 GB of unified memory. Swap or trim this list to match what you pulled.
MODELS = [
    "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
]

PROMPT = (
    "Explain what a KV cache is in large language model inference, "
    "and why decoding one token at a time is limited by memory bandwidth. "
    "Answer in a few clear sentences."
)

MAX_TOKENS = 200        # tokens to generate per run
WARMUP_RUNS = 1         # discarded, lets weights and caches settle
MEASURED_RUNS = 3       # averaged
RESULTS_DIR = Path(__file__).parent / "results"


def bench_one_run(model, tokenizer, prompt, max_tokens):
    """Run a single generation and return timing metrics."""
    prompt_tokens = tokenizer.encode(prompt)
    n_prompt = len(prompt_tokens)

    start = time.perf_counter()
    first_token_time = None
    n_generated = 0

    # stream_generate yields one generation step at a time
    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        n_generated += 1

    end = time.perf_counter()

    ttft = first_token_time - start                       # time to first token
    decode_time = end - first_token_time                  # time spent generating
    decode_tps = (n_generated - 1) / decode_time if decode_time > 0 else 0
    # prefill speed: prompt tokens divided by the time before the first token
    prefill_tps = n_prompt / ttft if ttft > 0 else 0

    peak_gb = mx.get_peak_memory() / 1e9

    return {
        "prompt_tokens": n_prompt,
        "generated_tokens": n_generated,
        "ttft_s": round(ttft, 4),
        "decode_tps": round(decode_tps, 2),
        "prefill_tps": round(prefill_tps, 2),
        "peak_mem_gb": round(peak_gb, 2),
    }


def bench_model(model_name):
    print(f"\nLoading {model_name} ...")
    model, tokenizer = load(model_name)

    # warmup
    for _ in range(WARMUP_RUNS):
        bench_one_run(model, tokenizer, PROMPT, MAX_TOKENS)

    runs = []
    for i in range(MEASURED_RUNS):
        mx.reset_peak_memory()
        r = bench_one_run(model, tokenizer, PROMPT, MAX_TOKENS)
        runs.append(r)
        print(
            f"  run {i+1}: ttft {r['ttft_s']}s, "
            f"decode {r['decode_tps']} tok/s, "
            f"prefill {r['prefill_tps']} tok/s, "
            f"peak {r['peak_mem_gb']} GB"
        )

    # average the measured runs
    def avg(key):
        return round(statistics.mean(r[key] for r in runs), 2)

    def spread(key):
        if len(runs) < 2:
            return 0.0
        return round(statistics.stdev(r[key] for r in runs), 2)

    summary = {
        "model": model_name,
        "prompt_tokens": runs[0]["prompt_tokens"],
        "generated_tokens": runs[0]["generated_tokens"],
        "ttft_s_mean": avg("ttft_s"),
        "ttft_s_std": spread("ttft_s"),
        "decode_tps_mean": avg("decode_tps"),
        "decode_tps_std": spread("decode_tps"),
        "prefill_tps_mean": avg("prefill_tps"),
        "peak_mem_gb": avg("peak_mem_gb"),
    }
    return summary


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    out_csv = RESULTS_DIR / "mlx_results.csv"

    all_rows = []
    for model_name in MODELS:
        try:
            all_rows.append(bench_model(model_name))
        except Exception as e:
            print(f"  skipped {model_name}: {e}")

    if not all_rows:
        print("No results produced.")
        return

    fields = list(all_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {out_csv}")
    print("\nSummary:")
    for r in all_rows:
        short = r["model"].split("/")[-1]
        print(f"  {short:40s} decode {r['decode_tps_mean']:6.1f} tok/s   "
              f"ttft {r['ttft_s_mean']:.3f}s   peak {r['peak_mem_gb']:.2f} GB")


if __name__ == "__main__":
    main()
