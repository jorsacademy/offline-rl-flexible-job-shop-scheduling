from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .behavior_cloning import LinearBehaviorCloning
from .dataset import build_cpsat_dataset
from .generator import random_instance
from .model import FJSPEnv
from .neural_offline import NeuralCQL, NeuralIQL
from .offline_dataset import build_mixed_offline_dataset
from .policies import minimum_slack, rollout


@dataclass(frozen=True)
class AblationRow:
    dataset_name: str
    split: str
    seed: int
    policy: str
    weighted_tardiness: float
    makespan: float


DATASET_MIXES: dict[str, tuple[str, ...]] = {
    "expert_only": ("cpsat",),
    "expert_strong": ("cpsat", "minimum_slack"),
    "balanced": ("cpsat", "minimum_slack", "edd", "spt"),
    "corrupted": ("cpsat", "minimum_slack", "edd", "spt", "random"),
}


def _evaluate(seed: int, policy, dataset_name: str, split: str, policy_name: str) -> AblationRow:
    jobs, n_machines = random_instance(seed)
    env = FJSPEnv(jobs, n_machines)
    _, metrics = rollout(env, policy)
    return AblationRow(
        dataset_name=dataset_name,
        split=split,
        seed=seed,
        policy=policy_name,
        weighted_tardiness=float(metrics["weighted_tardiness"]),
        makespan=float(metrics["makespan"]),
    )


def run_phase4_benchmark(
    train_seeds: list[int] | None = None,
    validation_seeds: list[int] | None = None,
    ood_seeds: list[int] | None = None,
    neural_epochs: int = 12,
) -> list[AblationRow]:
    train_seeds = train_seeds or [10, 11, 12, 13]
    validation_seeds = validation_seeds or [100, 101]
    ood_seeds = ood_seeds or [200, 201]

    expert = build_cpsat_dataset(train_seeds)
    bc = LinearBehaviorCloning().fit(expert.features, expert.labels)
    rows: list[AblationRow] = []

    for dataset_name, policies in DATASET_MIXES.items():
        dataset = build_mixed_offline_dataset(train_seeds, behavior_policies=policies)
        cql = NeuralCQL(hidden_dim=24, seed=0).fit(dataset, epochs=neural_epochs)
        iql = NeuralIQL(hidden_dim=24, seed=0).fit(dataset, epochs=neural_epochs)
        controllers = {
            "minimum_slack": minimum_slack,
            "bc_expert_only": bc.act,
            "neural_cql": cql.act,
            "neural_iql": iql.act,
        }
        for split, seeds in (("validation", validation_seeds), ("ood", ood_seeds)):
            for seed in seeds:
                for policy_name, policy in controllers.items():
                    rows.append(_evaluate(seed, policy, dataset_name, split, policy_name))
    return rows


def summarize(rows: list[AblationRow]) -> list[dict[str, float | str]]:
    output: list[dict[str, float | str]] = []
    groups = sorted({(row.dataset_name, row.split, row.policy) for row in rows})
    for dataset_name, split, policy in groups:
        selected = [
            row
            for row in rows
            if row.dataset_name == dataset_name and row.split == split and row.policy == policy
        ]
        output.append(
            {
                "dataset": dataset_name,
                "split": split,
                "policy": policy,
                "mean_weighted_tardiness": float(
                    np.mean([row.weighted_tardiness for row in selected])
                ),
                "mean_makespan": float(np.mean([row.makespan for row in selected])),
            }
        )
    return output


def main() -> None:
    rows = run_phase4_benchmark()
    print("dataset,split,policy,mean_weighted_tardiness,mean_makespan")
    for row in summarize(rows):
        print(
            f"{row['dataset']},{row['split']},{row['policy']},"
            f"{row['mean_weighted_tardiness']:.3f},{row['mean_makespan']:.3f}"
        )


if __name__ == "__main__":
    main()
