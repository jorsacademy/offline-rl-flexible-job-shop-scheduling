from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cpsat_expert import solve_cpsat_expert
from .features import action_features
from .generator import random_instance
from .model import FJSPEnv


@dataclass(frozen=True)
class DemonstrationDataset:
    features: np.ndarray
    labels: np.ndarray
    instance_seeds: tuple[int, ...]
    expert_weighted_tardiness: tuple[float, ...]


def build_cpsat_dataset(
    seeds: list[int],
    n_jobs: int = 6,
    n_machines: int = 4,
    operations_per_job: int = 3,
    time_limit_seconds: float = 1.0,
) -> DemonstrationDataset:
    """Create candidate-ranking data from CP-SAT expert schedules.

    At every decision state all feasible actions become rows. The expert-selected
    action receives label 1 and all alternatives label 0. This avoids pretending
    that a state-action pair absent from the log is necessarily good.
    """
    rows: list[np.ndarray] = []
    labels: list[int] = []
    expert_wtt: list[float] = []

    for seed in seeds:
        jobs, machine_count = random_instance(
            seed,
            n_jobs=n_jobs,
            n_machines=n_machines,
            operations_per_job=operations_per_job,
        )
        decisions = solve_cpsat_expert(
            jobs, machine_count, time_limit_seconds=time_limit_seconds, seed=seed
        )
        env = FJSPEnv(jobs, machine_count)
        for decision in decisions:
            chosen = (decision.job_id, decision.machine_id)
            feasible = env.feasible_actions()
            if chosen not in feasible:
                raise RuntimeError("expert sequence is inconsistent with environment precedence")
            for candidate in feasible:
                rows.append(action_features(env, candidate))
                labels.append(int(candidate == chosen))
            env.step(chosen)
        expert_wtt.append(float(env.metrics()["weighted_tardiness"]))

    return DemonstrationDataset(
        features=np.vstack(rows),
        labels=np.asarray(labels, dtype=float),
        instance_seeds=tuple(seeds),
        expert_weighted_tardiness=tuple(expert_wtt),
    )
