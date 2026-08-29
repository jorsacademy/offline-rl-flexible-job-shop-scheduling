import numpy as np

from offline_fjsp.cql import LinearCQL
from offline_fjsp.iql import LinearIQL
from offline_fjsp.offline_dataset import build_mixed_offline_dataset
from offline_fjsp.phase3_experiment import run_phase3_benchmark, summarize


def test_mixed_dataset_has_explicit_policy_provenance():
    dataset = build_mixed_offline_dataset([10], n_jobs=4, operations_per_job=2)
    assert set(dataset.policy_counts) == {"minimum_slack", "edd", "spt", "cpsat"}
    assert all(count > 0 for count in dataset.policy_counts.values())
    assert len(dataset.transitions) == sum(dataset.policy_counts.values())


def test_cql_and_iql_fit_finite_parameters():
    dataset = build_mixed_offline_dataset([11], n_jobs=4, operations_per_job=2)
    cql = LinearCQL(epochs=10).fit(dataset)
    iql = LinearIQL(epochs=10).fit(dataset)
    assert cql.weights_ is not None and np.isfinite(cql.weights_).all()
    assert iql.q_weights_ is not None and np.isfinite(iql.q_weights_).all()
    assert iql.v_weights_ is not None and np.isfinite(iql.v_weights_).all()


def test_phase3_benchmark_keeps_splits_separate():
    rows = run_phase3_benchmark(
        train_seeds=[12], validation_seeds=[101], ood_seeds=[201]
    )
    summary = summarize(rows)
    assert {row["split"] for row in summary} == {"validation", "ood"}
    assert {row["policy"] for row in summary} == {
        "spt",
        "edd",
        "minimum_slack",
        "bc",
        "cql",
        "iql",
        "cpsat",
    }
