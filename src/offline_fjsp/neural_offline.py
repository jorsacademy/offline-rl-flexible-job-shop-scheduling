from __future__ import annotations

import numpy as np

from .features import action_features
from .model import FJSPEnv
from .offline_dataset import OfflineTransitionDataset


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install the neural extra with: pip install -e '.[neural]'") from exc
    return torch, nn


class _NeuralScorer:
    def __init__(self, hidden_dim: int = 32, seed: int = 0):
        torch, nn = _torch()
        torch.manual_seed(seed)
        self.torch = torch
        self.model = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _fit_normalizer(self, dataset: OfflineTransitionDataset) -> None:
        x = np.vstack([transition.action_features for transition in dataset.transitions])
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-9] = 1.0

    def _tensor(self, x: np.ndarray):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("model is not fit")
        normalized = (np.asarray(x, dtype=np.float32) - self.mean_) / self.scale_
        return self.torch.as_tensor(normalized, dtype=self.torch.float32)

    def score_many(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        with self.torch.no_grad():
            return self.model(self._tensor(features)).squeeze(-1).cpu().numpy()

    def act(self, env: FJSPEnv) -> tuple[int, int]:
        actions = env.feasible_actions()
        features = np.vstack([action_features(env, action) for action in actions])
        scores = self.score_many(features)
        best = max(range(len(actions)), key=lambda idx: (float(scores[idx]), -actions[idx][0], -actions[idx][1]))
        return actions[best]


class NeuralCQL(_NeuralScorer):
    """Masked discrete CQL over variable-size feasible action sets."""

    def fit(
        self,
        dataset: OfflineTransitionDataset,
        epochs: int = 40,
        learning_rate: float = 2e-3,
        gamma: float = 0.98,
        alpha: float = 0.2,
    ) -> NeuralCQL:
        torch = self.torch
        self._fit_normalizer(dataset)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for _ in range(epochs):
            for transition in dataset.transitions:
                q_data = self.model(self._tensor(transition.action_features)).squeeze()
                with torch.no_grad():
                    if transition.done or len(transition.next_action_features) == 0:
                        target = torch.tensor(transition.reward, dtype=torch.float32)
                    else:
                        next_q = self.model(self._tensor(transition.next_action_features)).squeeze(-1)
                        target = torch.tensor(transition.reward, dtype=torch.float32) + gamma * next_q.max()
                bellman = (q_data - target).pow(2)
                current_q = self.model(self._tensor(transition.candidate_action_features)).squeeze(-1)
                conservative = torch.logsumexp(current_q, dim=0) - q_data
                loss = bellman + alpha * conservative
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self


class NeuralIQL(_NeuralScorer):
    """Compact neural IQL-style scorer with expectile value fitting."""

    def __init__(self, hidden_dim: int = 32, seed: int = 0):
        super().__init__(hidden_dim=hidden_dim, seed=seed)
        torch, nn = _torch()
        torch.manual_seed(seed + 1)
        self.value_model = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def fit(
        self,
        dataset: OfflineTransitionDataset,
        epochs: int = 40,
        learning_rate: float = 2e-3,
        gamma: float = 0.98,
        expectile: float = 0.7,
    ) -> NeuralIQL:
        torch = self.torch
        self._fit_normalizer(dataset)
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.value_model.parameters()),
            lr=learning_rate,
        )
        for _ in range(epochs):
            for transition in dataset.transitions:
                phi = self._tensor(transition.action_features)
                q = self.model(phi).squeeze()
                v = self.value_model(phi).squeeze()
                diff = q.detach() - v
                weight = torch.where(diff > 0, expectile, 1.0 - expectile)
                value_loss = weight * diff.pow(2)

                with torch.no_grad():
                    if transition.done or len(transition.next_action_features) == 0:
                        target = torch.tensor(transition.reward, dtype=torch.float32)
                    else:
                        next_features = self._tensor(transition.next_action_features)
                        next_v = self.value_model(next_features).squeeze(-1).max()
                        target = torch.tensor(transition.reward, dtype=torch.float32) + gamma * next_v
                q_loss = (q - target).pow(2)
                loss = q_loss + value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self
