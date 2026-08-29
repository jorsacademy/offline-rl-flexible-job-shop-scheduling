from __future__ import annotations

import numpy as np

from .features import action_features
from .model import FJSPEnv
from .offline_dataset import OfflineTransitionDataset


class LinearIQL:
    """Auditable linear implicit Q-learning baseline for logged scheduling data."""

    def __init__(
        self,
        gamma: float = 0.98,
        expectile: float = 0.7,
        learning_rate: float = 0.01,
        l2: float = 1e-4,
        epochs: int = 500,
    ):
        if not 0.5 <= expectile < 1.0:
            raise ValueError("expectile must be in [0.5, 1.0)")
        self.gamma = gamma
        self.expectile = expectile
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.q_weights_: np.ndarray | None = None
        self.v_weights_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("model is not fit")
        return (x - self.mean_) / self.scale_

    def fit(self, dataset: OfflineTransitionDataset) -> LinearIQL:
        x = np.vstack([transition.action_features for transition in dataset.transitions])
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-9] = 1.0
        x = self._normalize(x)
        q_weights = np.zeros(x.shape[1], dtype=float)
        v_weights = np.zeros(x.shape[1], dtype=float)

        for _ in range(self.epochs):
            q_grad = np.zeros_like(q_weights)
            v_grad = np.zeros_like(v_weights)
            for row, transition in zip(x, dataset.transitions):
                q_value = float(row @ q_weights)
                v_value = float(row @ v_weights)
                residual = q_value - v_value
                weight = self.expectile if residual >= 0 else 1.0 - self.expectile
                v_grad += -2.0 * weight * residual * row

                if transition.done or len(transition.next_action_features) == 0:
                    target = transition.reward
                else:
                    next_x = self._normalize(transition.next_action_features)
                    next_v = float(np.max(next_x @ v_weights))
                    target = transition.reward + self.gamma * next_v
                q_error = q_value - target
                q_grad += 2.0 * q_error * row

            scale = max(len(dataset.transitions), 1)
            q_grad = q_grad / scale + self.l2 * q_weights
            v_grad = v_grad / scale + self.l2 * v_weights
            q_weights -= self.learning_rate * q_grad
            v_weights -= self.learning_rate * v_grad

        self.q_weights_ = q_weights
        self.v_weights_ = v_weights
        return self

    def score(self, features: np.ndarray) -> float:
        if self.q_weights_ is None:
            raise RuntimeError("model is not fit")
        return float(self._normalize(features) @ self.q_weights_)

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
