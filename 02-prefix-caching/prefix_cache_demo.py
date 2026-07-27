"""
prefix_cache_demo.py

Demonstrates the idea behind SGLang's RadixAttention (prefix caching) on a local
machine, without needing a GPU or SGLang itself.

The principle: when many requests share the same long opening text, an engine
that caches the work done on that shared prefix can answer much faster than one
that reprocesses it every time. SGLang automates this with a radix tree. Here we
show the same effect using a local model served by Ollama, which does prompt
caching, by sending many questions that all share one long prefix and comparing
the time to first token when the prefix is reused versus when it is not.

Setup:
    brew install ollama
    ollama serve                 # in one terminal
    ollama pull qwen2.5:1.5b     # in another

    pip install requests

Usage:
    python prefix_cache_demo.py

What it does:
  - Builds one long shared prefix (a big system prompt / context block).
  - "Cold" case: each request uses a slightly different long prefix, so there is
    nothing to reuse.
  - "Warm" case: every request uses the exact same shared prefix, so after the
    first one the prefix work can be reused.
  - Measures time to first token in both cases and compares.
"""

import time
import csv
import requests
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:1.5b"
RESULTS_DIR = Path(__file__).parent / "results"

# A long shared prefix. In a real system this would be a long system prompt,
# a set of few-shot examples, or a retrieved document reused across requests.
SHARED_PREFIX = (
    "You are a careful technical assistant specialised in computer systems. "
    "You always answer precisely and briefly. Here is background you must use: "
    + ("Large language model inference has two phases, prefill and decode. "
       "Prefill reads the whole prompt at once and is compute bound. Decode "
       "generates one token at a time and is memory bandwidth bound because "
       "each token must read the model weights and the key value cache from "
       "memory. ") * 12   # repeated to make the prefix genuinely long
)

QUESTIONS = [
    "Which phase is compute bound?",
    "Which phase is memory bandwidth bound?",
    "Why is decode slower than prefill?",
    "What has to be read from memory for each token?",
    "In one word, what does prefill read?",
    "Name the two phases.",
    "Is the KV cache used during decode?",
    "What limits decode speed?",
]


def time_to_first_token(prompt):
    """Send a streaming request and measure seconds until the first token."""
    start = time.perf_counter()
    first = None
    with requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": True},
        stream=True,
    ) as resp:
        for line in resp.iter_lines():
            if line:
                if first is None:
                    first = time.perf_counter()
                # we only care about the first token timing here
                break
    if first is None:
        return None
    return first - start


def warm_up():
    """One throwaway call so the model is loaded into memory."""
    try:
        requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": "hi", "stream": False}, timeout=120)
    except Exception as e:
        print(f"warmup failed, is ollama running? {e}")
        raise


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    warm_up()

    rows = []

    # Warm case: same long prefix every time, so prefix work can be reused.
    print("Warm case (shared prefix reused):")
    for i, q in enumerate(QUESTIONS):
        prompt = SHARED_PREFIX + "\n\nQuestion: " + q + "\nAnswer:"
        ttft = time_to_first_token(prompt)
        print(f"  q{i}: ttft {ttft:.3f}s")
        rows.append({"case": "warm_shared_prefix", "index": i, "ttft_s": round(ttft, 4)})

    # Cold case: change the prefix each time so there is nothing to reuse.
    print("\nCold case (prefix differs every time):")
    for i, q in enumerate(QUESTIONS):
        unique = f"Session {i} unique marker {time.time()}. "
        prompt = unique + SHARED_PREFIX + "\n\nQuestion: " + q + "\nAnswer:"
        ttft = time_to_first_token(prompt)
        print(f"  q{i}: ttft {ttft:.3f}s")
        rows.append({"case": "cold_unique_prefix", "index": i, "ttft_s": round(ttft, 4)})

    out_csv = RESULTS_DIR / "prefix_cache_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case", "index", "ttft_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_csv}")

    # quick comparison, skipping the first warm request (it fills the cache)
    warm = [r["ttft_s"] for r in rows if r["case"] == "warm_shared_prefix"][1:]
    cold = [r["ttft_s"] for r in rows if r["case"] == "cold_unique_prefix"][1:]
    if warm and cold:
        warm_avg = sum(warm) / len(warm)
        cold_avg = sum(cold) / len(cold)
        print(f"\nAverage time to first token (excluding first of each):")
        print(f"  warm, shared prefix: {warm_avg:.3f}s")
        print(f"  cold, unique prefix: {cold_avg:.3f}s")
        if warm_avg > 0:
            print(f"  shared prefix was {cold_avg / warm_avg:.2f}x faster to first token")


if __name__ == "__main__":
    main()
