from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .cpsat_expert import solve_cpsat_expert
from .features import action_features
from .generator import random_instance
from .model import FJSPEnv
from .policies import earliest_due_date, minimum_slack, shortest_processing_time

Policy = Callable[[FJSPEnv], tuple[int, int]]


@dataclass(frozen=True)
class OfflineTransition:
    action_features: np.ndarray
    next_action_features: np.ndarray
    reward: float
    done: bool
    behavior_policy: str
    seed: int


@dataclass(frozen=True)
class OfflineTransitionDataset:
    transitions: tuple[OfflineTransition, ...]
    seeds: tuple[int, ...]
    policy_counts: dict[str, int]


def _expert_policy(jobs, n_machines: int, seed: int) -> Policy:
    decisions = solve_cpsat_expert(jobs, n_machines, time_limit_seconds=1.0, seed=seed)
    sequence = iter((decision.job_id, decision.machine_id) for decision in decisions)

    def policy(env: FJSPEnv) -> tuple[int, int]:
        action = next(sequence)
        if action not in env.feasible_actions():
            raise RuntimeError("CP-SAT demonstration is inconsistent with the simulator")
        return action

    return policy


def _rollout_transitions(
    jobs,
    n_machines: int,
    seed: int,
    policy_name: str,
    policy: Policy,
) -> list[OfflineTransition]:
    env = FJSPEnv(jobs, n_machines)
    staged: list[tuple[np.ndarray, np.ndarray, bool]] = []
    done = False
    while not done:
        action = policy(env)
        phi = action_features(env, action)
        _, _, done, _ = env.step(action)
        next_features = (
            np.vstack([action_features(env, candidate) for candidate in env.feasible_actions()])
            if not done
            else np.empty((0, phi.shape[0]), dtype=float)
        )
        staged.append((phi, next_features, done))

    metrics = env.metrics()
    terminal_utility = -float(metrics["weighted_tardiness"] + 0.02 * metrics["makespan"])
    transitions = []
    for phi, next_features, terminal in staged:
        transitions.append(
            OfflineTransition(
                action_features=phi,
                next_action_features=next_features,
                reward=terminal_utility if terminal else 0.0,
                done=terminal,
                behavior_policy=policy_name,
                seed=seed,
            )
        )
    return transitions


def build_mixed_offline_dataset(
    seeds: list[int],
    n_jobs: int = 6,
    n_machines: int = 4,
    operations_per_job: int = 3,
    include_expert: bool = True,
) -> OfflineTransitionDataset:
    """Build expert + heuristic logged data with explicit provenance.

    The dataset contains trajectories from CP-SAT and transparent dispatching
    policies. Terminal utility is based on weighted tardiness with a small
    makespan tie-breaker, while intermediate rewards are zero.
    """
    transitions: list[OfflineTransition] = []
    counts: dict[str, int] = {}
    heuristics: dict[str, Policy] = {
        "minimum_slack": minimum_slack,
        "edd": earliest_due_date,
        "spt": shortest_processing_time,
    }

    for seed in seeds:
        jobs, machine_count = random_instance(
            seed,
            n_jobs=n_jobs,
            n_machines=n_machines,
            operations_per_job=operations_per_job,
        )
        policies = dict(heuristics)
        if include_expert:
            policies["cpsat"] = _expert_policy(jobs, machine_count, seed)
        for name, policy in policies.items():
            episode = _rollout_transitions(jobs, machine_count, seed, name, policy)
            transitions.extend(episode)
            counts[name] = counts.get(name, 0) + len(episode)

    return OfflineTransitionDataset(tuple(transitions), tuple(seeds), counts)
