"""Small, dependency-free statistics helpers. Instances (not trials) are the unit of resampling."""
from __future__ import annotations

import itertools
import math
import random


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cluster_boot_ci(groups: dict, B: int = 2000, seed: int = 0, alpha: float = 0.05):
    """groups: instance_id -> (sum, count). Pooled mean, percentile CI over instance resamples."""
    ids = list(groups)
    if not ids:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        s = c = 0
        for _ in ids:
            a, b = groups[ids[rng.randrange(len(ids))]]
            s += a
            c += b
        vals.append(s / c if c else float("nan"))
    vals.sort()
    return (vals[int(alpha / 2 * B)], vals[min(B - 1, int((1 - alpha / 2) * B))])


def paired_boot_ci(diffs, B: int = 2000, seed: int = 0, alpha: float = 0.05):
    if not diffs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(diffs)
    vals = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return (vals[int(alpha / 2 * B)], vals[min(B - 1, int((1 - alpha / 2) * B))])


def sign_flip_p(diffs, B: int = 10000, seed: int = 0):
    """Two-sided paired sign-flip permutation test on the mean difference (exact when n <= 14)."""
    n = len(diffs)
    if n == 0:
        return float("nan")
    obs = abs(sum(diffs))
    if n <= 14:
        cnt = sum(1 for s in itertools.product((1, -1), repeat=n)
                  if abs(sum(si * d for si, d in zip(s, diffs))) >= obs - 1e-12)
        return cnt / 2 ** n
    rng = random.Random(seed)
    cnt = sum(1 for _ in range(B) if abs(sum(d if rng.random() < .5 else -d for d in diffs)) >= obs - 1e-12)
    return (cnt + 1) / (B + 1)
