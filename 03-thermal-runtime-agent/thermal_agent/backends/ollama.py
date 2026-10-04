"""Ollama workload: keeps generating in the background and reports recent speed."""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from collections import deque
from typing import Callable, Deque, Optional, Tuple

OLLAMA_URL = "http://localhost:11434/api/generate"


def ollama_generate(model: str, prompt: str, num_predict: int = 256, url: str = OLLAMA_URL) -> Tuple[int, float]:
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
        "options": {"num_predict": num_predict, "temperature": 0},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    tokens = d.get("eval_count", 0)
    secs = d.get("eval_duration", 0) / 1e9
    return tokens, (tokens / secs if secs > 0 else 0.0)


class BackgroundWorkload:
    """Runs generate_fn() in a loop on a thread. generate_fn returns
    (tokens, tokens_per_second). tok_per_s() is the mean of the latest results."""

    def __init__(self, generate_fn: Callable[[], Tuple[int, float]], keep: int = 3) -> None:
        self._gen = generate_fn
        self._recent: Deque[float] = deque(maxlen=keep)
        self._halt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None

    def _loop(self) -> None:
        while not self._halt.is_set():
            try:
                _tokens, tps = self._gen()
                self._recent.append(tps)
            except BaseException as e:  # keep the thread from dying silently
                self.error = e
                return

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._halt.set()
        if self._thread:
            self._thread.join(timeout=5)

    def tok_per_s(self) -> float:
        vals = list(self._recent)
        return sum(vals) / len(vals) if vals else 0.0
