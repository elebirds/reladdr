"""以独立场景族为重采样单位估计比率或均值区间，避免把同图多问当独立样本。"""

import random


def clustered_interval(totals: dict[str, tuple[float, int]], *, replicates: int, seed: int) -> dict:
    values = [
        (float(numerator), denominator)
        for numerator, denominator in totals.values()
        if denominator > 0
    ]
    if not values:
        return {"estimate": None, "ci95": None, "n_families": 0, "n": 0}
    numerator = sum(value for value, _ in values)
    denominator = sum(count for _, count in values)
    result = {
        "estimate": numerator / denominator,
        "ci95": None,
        "n_families": len(values),
        "n": denominator,
    }
    if len(values) < 2:
        return result
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        selected = [values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(value for value, _ in selected) / sum(n for _, n in selected))
    samples.sort()
    result["ci95"] = [
        samples[int((len(samples) - 1) * 0.025)],
        samples[int((len(samples) - 1) * 0.975)],
    ]
    return result
