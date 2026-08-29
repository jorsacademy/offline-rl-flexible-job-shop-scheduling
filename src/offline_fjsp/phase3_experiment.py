from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .behavior_cloning import LinearBehaviorCloning
from .cpsat_expert import solve_cpsat_expert
from .cql import LinearCQL
from .dataset import build_cpsat_dataset
from .generator import random_instance
from .iql import LinearIQL
from .model import FJSPEnv
from .offline_dataset import build_mixed_offline_dataset
from .policies import earliest_due_date, minimum_slack, rollout, shortest_processing_time


@dataclass(frozen=True)
class BenchmarkRow:
    split: str
    seed: int
    policy: str
    weighted_tardiness: float
    makespan: float


def _evaluate_policy(seed: int, name: str, policy, split: str) -> BenchmarkRow:
    jobs, n_machines = random_instance(seed)
    env = FJSPEnv(jobs, n_machines)
    _, metrics = rollout(env, policy)
    return BenchmarkRow(
        split=split,
        seed=seed,
        policy=name,
        weighted_tardiness=float(metrics["weighted_tardiness"]),
        makespan=float(metrics["makespan"]),
    )


def _evaluate_cpsat(seed: int, split: str) -> BenchmarkRow:
    jobs, n_machines = random_instance(seed)
    decisions = solve_cpsat_expert(jobs, n_machines, time_limit_seconds=1.0, seed=seed)
    env = FJSPEnv(jobs, n_machines)
    for decision in decisions:
        env.step((decision.job_id, decision.machine_id))
    metrics = env.metrics()
    return BenchmarkRow(
        split=split,
        seed=seed,
        policy="cpsat",
        weighted_tardiness=float(metrics["weighted_tardiness"]),
        makespan=float(metrics["makespan"]),
    )


def run_phase3_benchmark(
    train_seeds: list[int] | None = None,
    validation_seeds: list[int] | None = None,
    ood_seeds: list[int] | None = None,
) -> list[BenchmarkRow]:
    train_seeds = train_seeds or list(range(10, 18))
    validation_seeds = validation_seeds or list(range(100, 104))
    ood_seeds = ood_seeds or list(range(200, 204))

    expert_dataset = build_cpsat_dataset(train_seeds)
    bc = LinearBehaviorCloning().fit(expert_dataset.features, expert_dataset.labels)
    mixed = build_mixed_offline_dataset(train_seeds)
    cql = LinearCQL(epochs=160).fit(mixed)
    iql = LinearIQL(epochs=180).fit(mixed)

    policies = {
        "spt": shortest_processing_time,
        "edd": earliest_due_date,
        "minimum_slack": minimum_slack,
        "bc": bc.act,
        "cql": cql.act,
        "iql": iql.act,
    }
    rows: list[BenchmarkRow] = []
    for split, seeds in (("validation", validation_seeds), ("ood", ood_seeds)):
        for seed in seeds:
            for name, policy in policies.items():
                rows.append(_evaluate_policy(seed, name, policy, split))
            rows.append(_evaluate_cpsat(seed, split))
    return rows


def summarize(rows: list[BenchmarkRow]) -> list[dict[str, float | str]]:
    output = []
    groups = sorted({(row.split, row.policy) for row in rows})
    for split, policy in groups:
        selected = [row for row in rows if row.split == split and row.policy == policy]
        output.append(
            {
                "split": split,
                "policy": policy,
                "mean_weighted_tardiness": float(
                    np.mean([row.weighted_tardiness for row in selected])
                ),
                "mean_makespan": float(np.mean([row.makespan for row in selected])),
            }
        )
    return output


def main():
    rows = run_phase3_benchmark()
    print("split,policy,mean_weighted_tardiness,mean_makespan")
    for row in summarize(rows):
        print(
            f"{row['split']},{row['policy']},"
            f"{row['mean_weighted_tardiness']:.3f},{row['mean_makespan']:.3f}"
        )


if __name__ == "__main__":
    main()
