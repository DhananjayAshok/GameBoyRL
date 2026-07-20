"""
Small statistics helpers for the debug reports.

Deliberately dependency-light (stdlib ``math`` only) so the debug package stays importable
without scipy, and deliberately explicit about uncertainty: success rates in this project
are measured on 13-50 tasks, where a raw percentage difference is almost never meaningful
on its own.
"""

import math

Z_95 = 1.959963984540054


def wilson(successes: int, total: int, z: float = Z_95) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because it stays inside [0, 1] and behaves
    sensibly at the small n and extreme p this project routinely produces.

    :return: ``(low, high)`` as proportions. ``(0.0, 0.0)`` when *total* is 0.
    """
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denominator = 1 + z ** 2 / total
    centre = p + z ** 2 / (2 * total)
    spread = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2))
    return max(0.0, (centre - spread) / denominator), min(1.0, (centre + spread) / denominator)


def wilson_str(successes: int, total: int, as_percent: bool = True) -> str:
    """``"11.4% [4.5, 26.0] (n=35)"`` — the standard way rates are quoted in these reports."""
    if total <= 0:
        return "n/a (n=0)"
    low, high = wilson(successes, total)
    scale = 100 if as_percent else 1
    suffix = "%" if as_percent else ""
    return (f"{successes / total * scale:.1f}{suffix} "
            f"[{low * scale:.1f}, {high * scale:.1f}] (n={total})")


def mcnemar_exact(b: int, c: int) -> float:
    """
    Two-sided exact McNemar p-value for paired binary outcomes.

    *b* and *c* are the discordant counts (one model right where the other is wrong).
    Concordant pairs carry no information about a difference and are excluded by
    construction. Exact binomial rather than the chi-square approximation because the
    discordant counts here are typically single digits.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def gini(values) -> float:
    """Gini coefficient of a non-negative sequence. 0 = perfectly even, →1 = concentrated."""
    array = sorted(float(v) for v in values)
    n = len(array)
    total = sum(array)
    if n == 0 or total == 0:
        return 0.0
    weighted = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(array))
    return weighted / (n * total)
