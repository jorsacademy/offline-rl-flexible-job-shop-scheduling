from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
    candidate_action_features: np.ndarray
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


def _random_policy(seed: int) -> Policy:
    rng = np.random.default_rng(seed)

    def policy(env: FJSPEnv) -> tuple[int, int]:
        actions = env.feasible_actions()
        return actions[int(rng.integers(0, len(actions)))]

    return policy


def _rollout_transitions(
    jobs,
    n_machines: int,
    seed: int,
    policy_name: str,
    policy: Policy,
) -> list[OfflineTransition]:
    env = FJSPEnv(jobs, n_machines)
    staged: list[tuple[np.ndarray, np.ndarray, np.ndarray, bool]] = []
    done = False
    while not done:
        feasible = env.feasible_actions()
        candidate_features = np.vstack([action_features(env, candidate) for candidate in feasible])
        action = policy(env)
        phi = action_features(env, action)
        _, _, done, _ = env.step(action)
        next_features = (
            np.vstack([action_features(env, candidate) for candidate in env.feasible_actions()])
            if not done
            else np.empty((0, phi.shape[0]), dtype=float)
        )
        staged.append((phi, candidate_features, next_features, done))

    metrics = env.metrics()
    terminal_utility = -float(metrics["weighted_tardiness"] + 0.02 * metrics["makespan"])
    return [
        OfflineTransition(
            action_features=phi,
            candidate_action_features=candidates,
            next_action_features=next_features,
            reward=terminal_utility if terminal else 0.0,
            done=terminal,
            behavior_policy=policy_name,
            seed=seed,
        )
        for phi, candidates, next_features, terminal in staged
    ]


def build_mixed_offline_dataset(
    seeds: list[int],
    n_jobs: int = 6,
    n_machines: int = 4,
    operations_per_job: int = 3,
    behavior_policies: tuple[str, ...] = ("cpsat", "minimum_slack", "edd", "spt"),
) -> OfflineTransitionDataset:
    """Build logged trajectories with explicit quality/provenance controls.

    Supported behavior policies are ``cpsat``, ``minimum_slack``, ``edd``, ``spt``
    and ``random``. Dataset-quality ablations are created by changing their mixture,
    not by relabeling trajectories after collection.
    """
    allowed = {"cpsat", "minimum_slack", "edd", "spt", "random"}
    unknown = set(behavior_policies) - allowed
    if unknown:
        raise ValueError(f"unknown behavior policies: {sorted(unknown)}")

    transitions: list[OfflineTransition] = []
    counts: dict[str, int] = {}
    for seed in seeds:
        jobs, machine_count = random_instance(
            seed,
            n_jobs=n_jobs,
            n_machines=n_machines,
            operations_per_job=operations_per_job,
        )
        policies: dict[str, Policy] = {
            "minimum_slack": minimum_slack,
            "edd": earliest_due_date,
            "spt": shortest_processing_time,
            "random": _random_policy(seed + 100_000),
        }
        if "cpsat" in behavior_policies:
            policies["cpsat"] = _expert_policy(jobs, machine_count, seed)

        for name in behavior_policies:
            policy = policies[name]
            episode = _rollout_transitions(jobs, machine_count, seed, name, policy)
            transitions.extend(episode)
            counts[name] = counts.get(name, 0) + len(episode)

    return OfflineTransitionDataset(tuple(transitions), tuple(seeds), counts)
