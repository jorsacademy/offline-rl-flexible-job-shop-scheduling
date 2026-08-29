from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .behavior_cloning import LinearBehaviorCloning
from .cpsat_expert import solve_cpsat_expert
from .dataset import build_cpsat_dataset
from .generator import random_instance
from .model import FJSPEnv
from .neural_offline import NeuralCQL, NeuralIQL
from .offline_dataset import build_mixed_offline_dataset
from .policies import minimum_slack, rollout
from .statistics import paired_bootstrap


@dataclass(frozen=True)
class FinalRow:
    scenario: str
    instance_seed: int
    model_seed: int
    policy: str
    weighted_tardiness: float
    makespan: float
    latency_ms: float
    feasible: bool


def _instance(seed: int, scenario: str):
    if scenario == "nominal":
        return random_instance(seed)
    if scenario == "ood_scale":
        return random_instance(seed, n_jobs=8, n_machines=5, operations_per_job=4, eligibility=2)
    if scenario == "ood_flexibility":
        return random_instance(seed, n_jobs=6, n_machines=4, operations_per_job=3, eligibility=1)
    raise ValueError(f"unknown scenario: {scenario}")


def _evaluate_policy(seed: int, scenario: str, model_seed: int, name: str, policy) -> FinalRow:
    jobs, n_machines = _instance(seed, scenario)
    env = FJSPEnv(jobs, n_machines)
    start = perf_counter()
    try:
        _, metrics = rollout(env, policy)
        feasible = len(env.completion) == len(jobs)
    except Exception:
        metrics = {"weighted_tardiness": float("inf"), "makespan": float("inf")}
        feasible = False
    elapsed = (perf_counter() - start) * 1000.0
    return FinalRow(
        scenario,
        seed,
        model_seed,
        name,
        float(metrics["weighted_tardiness"]),
        float(metrics["makespan"]),
        elapsed,
        feasible,
    )


def _evaluate_cpsat(seed: int, scenario: str) -> FinalRow:
    jobs, n_machines = _instance(seed, scenario)
    start = perf_counter()
    decisions = solve_cpsat_expert(jobs, n_machines, time_limit_seconds=2.0, seed=seed)
    env = FJSPEnv(jobs, n_machines)
    for decision in decisions:
        env.step((decision.job_id, decision.machine_id))
    elapsed = (perf_counter() - start) * 1000.0
    metrics = env.metrics()
    return FinalRow(
        scenario,
        seed,
        -1,
        "cpsat",
        float(metrics["weighted_tardiness"]),
        float(metrics["makespan"]),
        elapsed,
        len(env.completion) == len(jobs),
    )


def run_final_campaign(
    *,
    train_seeds: list[int] | None = None,
    model_seeds: list[int] | None = None,
    nominal_final: list[int] | None = None,
    ood_scale: list[int] | None = None,
    ood_flexibility: list[int] | None = None,
    neural_epochs: int = 30,
) -> list[FinalRow]:
    """Run the frozen final benchmark. Do not tune on the final seed blocks."""
    train_seeds = train_seeds or list(range(10, 20))
    model_seeds = model_seeds or [0, 1, 2]
    nominal_final = nominal_final or list(range(1000, 1020))
    ood_scale = ood_scale or list(range(1100, 1110))
    ood_flexibility = ood_flexibility or list(range(1200, 1210))

    expert = build_cpsat_dataset(train_seeds)
    bc = LinearBehaviorCloning().fit(expert.features, expert.labels)
    dataset = build_mixed_offline_dataset(
        train_seeds,
        behavior_policies=("cpsat", "minimum_slack", "edd", "spt"),
    )

    learned = []
    for model_seed in model_seeds:
        learned.append((model_seed, "neural_cql", NeuralCQL(hidden_dim=32, seed=model_seed).fit(dataset, epochs=neural_epochs).act))
        learned.append((model_seed, "neural_iql", NeuralIQL(hidden_dim=32, seed=model_seed).fit(dataset, epochs=neural_epochs).act))

    rows: list[FinalRow] = []
    blocks = (("nominal", nominal_final), ("ood_scale", ood_scale), ("ood_flexibility", ood_flexibility))
    for scenario, seeds in blocks:
        for seed in seeds:
            rows.append(_evaluate_policy(seed, scenario, -1, "minimum_slack", minimum_slack))
            rows.append(_evaluate_policy(seed, scenario, -1, "bc", bc.act))
            for model_seed, name, policy in learned:
                rows.append(_evaluate_policy(seed, scenario, model_seed, name, policy))
            rows.append(_evaluate_cpsat(seed, scenario))
    return rows


def aggregate(rows: list[FinalRow]) -> list[dict[str, float | str]]:
    output = []
    for scenario, policy in sorted({(row.scenario, row.policy) for row in rows}):
        selected = [row for row in rows if row.scenario == scenario and row.policy == policy]
        output.append({
            "scenario": scenario,
            "policy": policy,
            "mean_weighted_tardiness": float(np.mean([r.weighted_tardiness for r in selected])),
            "mean_makespan": float(np.mean([r.makespan for r in selected])),
            "mean_latency_ms": float(np.mean([r.latency_ms for r in selected])),
            "feasibility_rate": float(np.mean([r.feasible for r in selected])),
        })
    return output


def paired_against_minimum_slack(rows: list[FinalRow]) -> list[dict[str, float | str]]:
    output = []
    for scenario in sorted({row.scenario for row in rows}):
        reference_rows = [r for r in rows if r.scenario == scenario and r.policy == "minimum_slack"]
        reference = {r.instance_seed: r.weighted_tardiness for r in reference_rows}
        for policy in sorted({r.policy for r in rows if r.scenario == scenario} - {"minimum_slack"}):
            candidate_rows = [r for r in rows if r.scenario == scenario and r.policy == policy]
            per_instance = {}
            for seed in reference:
                values = [r.weighted_tardiness for r in candidate_rows if r.instance_seed == seed]
                if values:
                    per_instance[seed] = float(np.mean(values))
            common = sorted(set(reference) & set(per_instance))
            comparison = paired_bootstrap(
                np.asarray([per_instance[s] for s in common]),
                np.asarray([reference[s] for s in common]),
                n_bootstrap=2000,
            )
            output.append({
                "scenario": scenario,
                "policy": policy,
                "mean_difference": comparison.mean_difference,
                "ci_low": comparison.ci_low,
                "ci_high": comparison.ci_high,
                "win_rate": comparison.win_rate,
                "sign_test_pvalue": comparison.sign_test_pvalue,
                "n": float(comparison.n),
            })
    return output


def main() -> None:
    rows = run_final_campaign()
    print("scenario,policy,mean_wtt,mean_makespan,mean_latency_ms,feasibility_rate")
    for row in aggregate(rows):
        print(
            f"{row['scenario']},{row['policy']},{row['mean_weighted_tardiness']:.3f},"
            f"{row['mean_makespan']:.3f},{row['mean_latency_ms']:.3f},"
            f"{row['feasibility_rate']:.3f}"
        )
    print("scenario,policy,mean_diff_vs_min_slack,ci_low,ci_high,win_rate,sign_p,n")
    for row in paired_against_minimum_slack(rows):
        print(
            f"{row['scenario']},{row['policy']},{row['mean_difference']:.3f},"
            f"{row['ci_low']:.3f},{row['ci_high']:.3f},{row['win_rate']:.3f},"
            f"{row['sign_test_pvalue']:.4f},{int(row['n'])}"
        )


if __name__ == "__main__":
    main()
