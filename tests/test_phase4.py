import numpy as np
import pytest

from offline_fjsp.offline_dataset import build_mixed_offline_dataset
from offline_fjsp.phase4_experiment import DATASET_MIXES


def test_dataset_quality_mixes_have_expected_provenance():
    for name, policies in DATASET_MIXES.items():
        dataset = build_mixed_offline_dataset(
            [10], n_jobs=3, operations_per_job=2, behavior_policies=policies
        )
        assert set(dataset.policy_counts) == set(policies), name
        assert all(len(t.candidate_action_features) >= 1 for t in dataset.transitions)
        assert all(
            t.action_features.shape[0] == t.candidate_action_features.shape[1]
            for t in dataset.transitions
        )


def test_random_corruption_is_explicit_not_silent():
    clean = build_mixed_offline_dataset([11], behavior_policies=("cpsat",))
    corrupted = build_mixed_offline_dataset([11], behavior_policies=("cpsat", "random"))
    assert "random" not in clean.policy_counts
    assert corrupted.policy_counts["random"] > 0
    assert len(corrupted.transitions) > len(clean.transitions)


def test_neural_cql_and_iql_smoke():
    pytest.importorskip("torch")
    from offline_fjsp.neural_offline import NeuralCQL, NeuralIQL

    dataset = build_mixed_offline_dataset(
        [12], n_jobs=3, operations_per_job=2, behavior_policies=("cpsat", "random")
    )
    cql = NeuralCQL(hidden_dim=8, seed=1).fit(dataset, epochs=2)
    iql = NeuralIQL(hidden_dim=8, seed=1).fit(dataset, epochs=2)
    probe = dataset.transitions[0].candidate_action_features
    assert np.isfinite(cql.score_many(probe)).all()
    assert np.isfinite(iql.score_many(probe)).all()
