from __future__ import annotations

import numpy as np

from .features import action_features
from .model import FJSPEnv
from .offline_dataset import OfflineTransitionDataset


class LinearCQL:
    """Small fitted conservative Q-learning baseline over action features.

    Bellman targets use next-state feasible actions. The conservative term uses
    the full feasible action set at the *current* logged decision state, which
    prevents accidental support leakage from the next state.
    """

    def __init__(
        self,
        gamma: float = 0.98,
        learning_rate: float = 0.01,
        conservative_weight: float = 0.05,
        l2: float = 1e-4,
        epochs: int = 400,
        seed: int = 0,
    ):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.conservative_weight = conservative_weight
        self.l2 = l2
        self.epochs = epochs
        self.rng = np.random.default_rng(seed)
        self.weights_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("model is not fit")
        return (x - self.mean_) / self.scale_

    def fit(self, dataset: OfflineTransitionDataset) -> LinearCQL:
        x = np.vstack([transition.action_features for transition in dataset.transitions])
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-9] = 1.0
        x = self._normalize(x)
        weights = np.zeros(x.shape[1], dtype=float)

        for _ in range(self.epochs):
            grad = np.zeros_like(weights)
            for row, transition in zip(x, dataset.transitions):
                if transition.done or len(transition.next_action_features) == 0:
                    target = transition.reward
                else:
                    next_x = self._normalize(transition.next_action_features)
                    target = transition.reward + self.gamma * float(np.max(next_x @ weights))
                error = float(row @ weights - target)
                grad += 2.0 * error * row

                candidates = self._normalize(transition.candidate_action_features)
                q_values = candidates @ weights
                probabilities = np.exp(q_values - np.max(q_values))
                probabilities /= probabilities.sum()
                grad += self.conservative_weight * (probabilities @ candidates - row)

            grad = grad / max(len(dataset.transitions), 1) + self.l2 * weights
            weights -= self.learning_rate * grad

        self.weights_ = weights
        return self

    def score(self, features: np.ndarray) -> float:
        if self.weights_ is None:
            raise RuntimeError("model is not fit")
        return float(self._normalize(features) @ self.weights_)

    def act(self, env: FJSPEnv) -> tuple[int, int]:
        actions = env.feasible_actions()
        return max(
            actions,
            key=lambda action: (
                self.score(action_features(env, action)),
                -action[0],
                -action[1],
            ),
        )
