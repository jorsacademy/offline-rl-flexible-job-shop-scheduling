from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedComparison:
    mean_difference: float
    ci_low: float
    ci_high: float
    win_rate: float
    median_difference: float
    n: int


def paired_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    n_bootstrap: int = 4000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> PairedComparison:
    """Compare lower-is-better paired KPI values using a deterministic bootstrap."""
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("candidate and reference must be equal-length one-dimensional arrays")
    if len(candidate) == 0:
        raise ValueError("at least one paired observation is required")

    differences = candidate - reference
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
    boot_means = differences[indices].mean(axis=1)
    alpha = 1.0 - confidence
    low, high = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return PairedComparison(
        mean_difference=float(np.mean(differences)),
        ci_low=float(low),
        ci_high=float(high),
        win_rate=float(np.mean(candidate < reference)),
        median_difference=float(np.median(differences)),
        n=len(candidate),
    )


def probability_of_superiority(candidate: np.ndarray, reference: np.ndarray) -> float:
    """Paired probability that the lower-is-better candidate beats the reference."""
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape:
        raise ValueError("paired arrays must have the same shape")
    wins = np.sum(candidate < reference)
    ties = np.sum(candidate == reference)
    return float((wins + 0.5 * ties) / len(candidate))
