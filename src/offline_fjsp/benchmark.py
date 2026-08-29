from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .behavior_cloning import LinearBehaviorCloning
from .cpsat_expert import solve_cpsat_expert
from .dataset import build_cpsat_dataset
from .generator import random_instance
from .model import FJSPEnv
from .policies import earliest_due_date, minimum_slack, rollout, shortest_processing_time


@dataclass(frozen=True)
class BenchmarkRow:
    seed: int
    controller: str
    makespan: float
    weighted_tardiness: float


def _expert_rollout(jobs, n_machines, seed: int, time_limit_seconds: float):
    decisions = solve_cpsat_expert(
        jobs, n_machines, time_limit_seconds=time_limit_seconds, seed=seed
    )
    env = FJSPEnv(jobs, n_machines)
    for decision in decisions:
        env.step((decision.job_id, decision.machine_id))
    return env.metrics()


def run_benchmark(
    train_seeds: list[int],
    test_seeds: list[int],
    n_jobs: int = 5,
    n_machines: int = 3,
    operations_per_job: int = 3,
    time_limit_seconds: float = 0.5,
) -> list[BenchmarkRow]:
    dataset = build_cpsat_dataset(
        train_seeds,
        n_jobs=n_jobs,
        n_machines=n_machines,
        operations_per_job=operations_per_job,
        time_limit_seconds=time_limit_seconds,
    )
    bc = LinearBehaviorCloning().fit(dataset.features, dataset.labels)
    controllers = {
        "spt": shortest_processing_time,
        "edd": earliest_due_date,
        "minimum_slack": minimum_slack,
        "behavior_cloning": bc.act,
    }

    rows: list[BenchmarkRow] = []
    for seed in test_seeds:
        jobs, machine_count = random_instance(
            seed,
            n_jobs=n_jobs,
            n_machines=n_machines,
            operations_per_job=operations_per_job,
        )
        for name, policy in controllers.items():
            _, metrics = rollout(FJSPEnv(jobs, machine_count), policy)
            rows.append(
                BenchmarkRow(seed, name, metrics["makespan"], metrics["weighted_tardiness"])
            )
        metrics = _expert_rollout(jobs, machine_count, seed, time_limit_seconds)
        rows.append(
            BenchmarkRow(seed, "cpsat_expert", metrics["makespan"], metrics["weighted_tardiness"])
        )
    return rows


def summarize(rows: list[BenchmarkRow]) -> dict[str, dict[str, float]]:
    controllers = sorted({row.controller for row in rows})
    return {
        controller: {
            "mean_makespan": mean(row.makespan for row in rows if row.controller == controller),
            "mean_weighted_tardiness": mean(
                row.weighted_tardiness for row in rows if row.controller == controller
            ),
        }
        for controller in controllers
    }
