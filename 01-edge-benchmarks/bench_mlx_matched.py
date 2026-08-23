"""
bench_mlx_matched.py

A Mac benchmark that matches Protocol A from DEVICE_TABLE.md, the same growing
multi turn conversation test used on the iPhone and iPad through PocketPal.

Unlike bench_sustained.py, which regenerates fresh text from the same starting
prompt every time, this script builds one real conversation: each new prompt is
sent after the model's previous answer, so the model sees the whole growing
history, exactly like a real chat app. This is the same idea behind the prefix
caching demo in 02-prefix-caching, reused here to make a fair multi device
comparison.

Model: Llama-3.2-1B-Instruct, 4 bit, chosen to match the model used on the
phone and tablet as closely as MLX allows. The exact quantization format still
differs (MLX vs GGUF), which is worth keeping in mind when comparing numbers.

The four prompts are the exact same ones, in the exact same order, sent on the
iPhone and iPad.

Setup:
    pip install mlx-lm

Usage:
    python bench_mlx_matched.py

Output:
    results/mlx_matched_results.csv, one row per turn.
"""

import time
import csv
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate

try:
    from mlx_lm.models.cache import make_prompt_cache
    HAVE_CACHE = True
except ImportError:
    HAVE_CACHE = False

MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"

# the exact same four prompts, in the exact same order, used on the iPhone and
# iPad through PocketPal
PROMPTS = [
    "Write a detailed paragraph explaining how large language models generate text.",
    "Now explain what a KV cache is and why it matters.",
    "Continue and explain the difference between prefill and decode.",
    "Write more detail about why decode is slower than prefill.",
]

MAX_TOKENS = 300
RESULTS_DIR = Path(__file__).parent / "results"


def run_turn(model, tokenizer, prompt_cache, conversation, user_text, max_tokens):
    """Send one more turn in a growing conversation, return timing and text."""
    conversation.append({"role": "user", "content": user_text})
    prompt = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    kwargs = {"max_tokens": max_tokens}
    if HAVE_CACHE:
        kwargs["prompt_cache"] = prompt_cache

    start = time.perf_counter()
    first_token_time = None
    n_generated = 0
    full_response = ""

    for response in stream_generate(model, tokenizer, prompt, **kwargs):
        if first_token_time is None:
            first_token_time = time.perf_counter()
        n_generated += 1
        full_response += response.text

    end = time.perf_counter()
    conversation.append({"role": "assistant", "content": full_response})

    ttft = (first_token_time - start) if first_token_time else 0
    decode_time = (end - first_token_time) if first_token_time else 0
    decode_tps = (n_generated - 1) / decode_time if decode_time > 0 else 0

    return {
        "ttft_s": round(ttft, 4),
        "decode_tps": round(decode_tps, 2),
        "generated_tokens": n_generated,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Loading {MODEL} ...")
    model, tokenizer = load(MODEL)

    prompt_cache = make_prompt_cache(model) if HAVE_CACHE else None
    conversation = []
    rows = []

    for i, user_text in enumerate(PROMPTS):
        print(f"\nTurn {i + 1}: {user_text}")
        result = run_turn(model, tokenizer, prompt_cache, conversation, user_text, MAX_TOKENS)
        print(
            f"  ttft {result['ttft_s']}s, decode {result['decode_tps']} tok/s, "
            f"generated {result['generated_tokens']} tokens"
        )
        rows.append({
            "turn": i + 1,
            "prompt": user_text,
            "ttft_s": result["ttft_s"],
            "decode_tps": result["decode_tps"],
            "generated_tokens": result["generated_tokens"],
        })

    out_csv = RESULTS_DIR / "mlx_matched_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["turn", "prompt", "ttft_s", "decode_tps", "generated_tokens"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {out_csv}")

    first_tps = rows[0]["decode_tps"]
    last_tps = rows[-1]["decode_tps"]
    if first_tps > 0:
        change = round((last_tps - first_tps) / first_tps * 100, 1)
        print(
            f"\nTurn 1: {first_tps} tok/s -> Turn {len(rows)}: {last_tps} tok/s, "
            f"change {change} percent"
        )


if __name__ == "__main__":
    main()
