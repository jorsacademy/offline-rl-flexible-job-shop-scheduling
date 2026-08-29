from __future__ import annotations

import numpy as np

from .features import action_features
from .model import FJSPEnv


class LinearBehaviorCloning:
    """Ridge-regularized candidate ranker learned from expert demonstrations."""

    def __init__(self, ridge: float = 1e-3):
        self.ridge = ridge
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> LinearBehaviorCloning:
        self.mean_ = features.mean(axis=0)
        self.scale_ = features.std(axis=0)
        self.scale_[self.scale_ < 1e-9] = 1.0
        x = (features - self.mean_) / self.scale_
        gram = x.T @ x + self.ridge * np.eye(x.shape[1])
        self.coef_ = np.linalg.solve(gram, x.T @ labels)
        return self

    def score(self, features: np.ndarray) -> float:
        if self.coef_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("policy must be fit before scoring")
        x = (features - self.mean_) / self.scale_
        return float(x @ self.coef_)

    def act(self, env: FJSPEnv) -> tuple[int, int]:
        actions = env.feasible_actions()
        if not actions:
            raise RuntimeError("no feasible actions")
        return max(
            actions,
            key=lambda action: (
                self.score(action_features(env, action)),
                -action[0],
                -action[1],
            ),
        )
