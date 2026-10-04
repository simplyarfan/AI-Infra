"""Configuration-Performance Dataset.

Stores every (configuration, context) -> (throughput, power, temperature) pair
the agent observes, as in the dataset box of the AutoOptAgent diagram. It is
used to correct priors at runtime and can be saved and reloaded so a later
session (or a cloud side learner) starts from what was already measured.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import Config, config_key


@dataclass
class Record:
    t_s: float
    config: Dict[str, int]
    tok_per_s: float
    power_w: float
    temp_c: Optional[float]
    throttled: bool
    context: Dict[str, float] = field(default_factory=dict)


@dataclass
class ConfigStats:
    n: int
    mean_tps: float
    mean_power: float
    mean_temp: Optional[float]
    throttled_fraction: float

    @property
    def j_per_token(self) -> float:
        return self.mean_power / self.mean_tps if self.mean_tps > 0 else float("inf")


class ConfigPerfDataset:
    def __init__(self) -> None:
        self.records: List[Record] = []
        # (config key, throttled) -> [n, sum_tps, sum_power, sum_temp, n_temp]
        self._agg: Dict[Tuple[Tuple[Tuple[str, int], ...], bool], List[float]] = {}

    def add(self, rec: Record) -> None:
        self.records.append(rec)
        k = (config_key(rec.config), bool(rec.throttled))
        a = self._agg.setdefault(k, [0, 0.0, 0.0, 0.0, 0])
        a[0] += 1
        a[1] += rec.tok_per_s
        a[2] += rec.power_w
        if rec.temp_c is not None:
            a[3] += rec.temp_c
            a[4] += 1

    def stats(self, config: Config, include_throttled: bool = False, min_n: int = 1) -> Optional[ConfigStats]:
        key = config_key(config)
        free = self._agg.get((key, False))
        thr = self._agg.get((key, True))
        parts = [p for p in ([free] + ([thr] if include_throttled else [])) if p]
        n = int(sum(p[0] for p in parts))
        if n < min_n or n == 0:
            return None
        n_temp = sum(p[4] for p in parts)
        total_n = (free[0] if free else 0) + (thr[0] if thr else 0)
        return ConfigStats(
            n=n,
            mean_tps=sum(p[1] for p in parts) / n,
            mean_power=sum(p[2] for p in parts) / n,
            mean_temp=(sum(p[3] for p in parts) / n_temp) if n_temp else None,
            throttled_fraction=(thr[0] / total_n) if (thr and total_n) else 0.0,
        )

    def configs(self) -> List[Dict[str, int]]:
        seen = {}
        for (key, _), _v in self._agg.items():
            seen[key] = dict(key)
        return list(seen.values())

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            for r in self.records:
                f.write(json.dumps(asdict(r)) + "\n")

    @classmethod
    def load(cls, path: str) -> "ConfigPerfDataset":
        ds = cls()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ds.add(Record(**json.loads(line)))
        return ds
