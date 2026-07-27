"""
bench_sustained.py

Runs generation continuously for a set number of minutes and records the
decode throughput of each successive chunk. The point is to see whether the
device slows down over time as it heats up (thermal throttling).

On a laptop plugged in you may see little or no drop. On a phone, or a laptop
on battery or under thermal pressure, you should see the tokens per second fall
and then settle at a lower plateau. Either result is a real finding worth
writing down.

Setup:
    pip install mlx-lm

Usage:
    python bench_sustained.py

Output:
    results/sustained_<model>.csv with one row per chunk:
      elapsed_s, chunk_index, decode_tps
"""

import time
import csv
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

# One model is enough for this test. Pick a mid size one so it works the chip.
MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"

PROMPT = (
    "Write a long, detailed explanation of how large language models generate "
    "text, covering tokens, attention, the KV cache, and why generation is "
    "slower than reading the prompt. Keep going with more detail."
)

TOTAL_MINUTES = 5          # how long to hammer the device
CHUNK_TOKENS = 100         # measure throughput every this many generated tokens
RESULTS_DIR = Path(__file__).parent / "results"


def run_chunk(model, tokenizer, prompt, n_tokens):
    """Generate n_tokens and return decode tokens per second for this chunk."""
    start = time.perf_counter()
    first = None
    count = 0
    for _ in stream_generate(model, tokenizer, prompt, max_tokens=n_tokens):
        if first is None:
            first = time.perf_counter()
        count += 1
    end = time.perf_counter()
    decode_time = end - (first if first else start)
    tps = (count - 1) / decode_time if decode_time > 0 else 0
    return round(tps, 2)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    short = MODEL.split("/")[-1]
    out_csv = RESULTS_DIR / f"sustained_{short}.csv"

    print(f"Loading {MODEL} ...")
    model, tokenizer = load(MODEL)

    print(f"Running sustained load for {TOTAL_MINUTES} minutes. "
          f"Watch the tokens per second column.")

    rows = []
    run_start = time.perf_counter()
    deadline = run_start + TOTAL_MINUTES * 60
    chunk_index = 0

    while time.perf_counter() < deadline:
        tps = run_chunk(model, tokenizer, PROMPT, CHUNK_TOKENS)
        elapsed = round(time.perf_counter() - run_start, 1)
        rows.append({
            "elapsed_s": elapsed,
            "chunk_index": chunk_index,
            "decode_tps": tps,
        })
        print(f"  t={elapsed:6.1f}s   chunk {chunk_index:3d}   {tps:6.1f} tok/s")
        chunk_index += 1

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["elapsed_s", "chunk_index", "decode_tps"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {out_csv}")
    if rows:
        first_tps = rows[0]["decode_tps"]
        last_tps = rows[-1]["decode_tps"]
        if first_tps > 0:
            drop = round((first_tps - last_tps) / first_tps * 100, 1)
            print(f"Start {first_tps} tok/s, end {last_tps} tok/s, "
                  f"change {drop} percent over {TOTAL_MINUTES} minutes.")


if __name__ == "__main__":
    main()
