import numpy as np
import pytest

from offline_fjsp.final_evaluation import (
    aggregate,
    paired_against_minimum_slack,
    run_final_campaign,
)
from offline_fjsp.statistics import exact_sign_test, paired_bootstrap


def test_paired_bootstrap_detects_consistent_improvement():
    candidate = np.array([1.0, 2.0, 3.0, 4.0])
    reference = np.array([2.0, 3.0, 4.0, 5.0])
    result = paired_bootstrap(candidate, reference, n_bootstrap=200, seed=1)
    assert result.mean_difference == -1.0
    assert result.ci_high < 0.0
    assert result.win_rate == 1.0


def test_exact_sign_test_is_symmetric():
    a = np.array([1.0, 2.0, 5.0, 4.0])
    b = np.array([2.0, 3.0, 4.0, 5.0])
    assert exact_sign_test(a, b) == exact_sign_test(b, a)


def test_final_campaign_smoke_keeps_frozen_blocks_and_feasibility():
    pytest.importorskip("torch")
    rows = run_final_campaign(
        train_seeds=[10],
        model_seeds=[0],
        nominal_final=[1000],
        ood_scale=[1100],
        ood_flexibility=[1200],
        neural_epochs=1,
    )
    assert {row.scenario for row in rows} == {"nominal", "ood_scale", "ood_flexibility"}
    assert {row.policy for row in rows} == {
        "minimum_slack",
        "bc",
        "neural_cql",
        "neural_iql",
        "cpsat",
    }
    assert all(row.feasible for row in rows)
    assert all(np.isfinite(row.weighted_tardiness) for row in rows)
    summary = aggregate(rows)
    comparisons = paired_against_minimum_slack(rows)
    assert summary
    assert comparisons
