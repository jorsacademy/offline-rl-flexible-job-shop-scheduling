from offline_fjsp.behavior_cloning import LinearBehaviorCloning
from offline_fjsp.cpsat_expert import solve_cpsat_expert
from offline_fjsp.dataset import build_cpsat_dataset
from offline_fjsp.generator import random_instance
from offline_fjsp.model import FJSPEnv
from offline_fjsp.policies import rollout


def test_random_instance_is_reproducible():
    jobs_a, machines_a = random_instance(42)
    jobs_b, machines_b = random_instance(42)
    assert machines_a == machines_b
    assert jobs_a == jobs_b


def test_cpsat_expert_schedules_every_operation_once():
    jobs, n_machines = random_instance(7, n_jobs=4, n_machines=3, operations_per_job=2)
    decisions = solve_cpsat_expert(jobs, n_machines, time_limit_seconds=0.5, seed=7)
    expected = sum(len(job.operations) for job in jobs)
    assert len(decisions) == expected
    assert len({(d.job_id, d.operation_index) for d in decisions}) == expected


def test_behavior_cloning_trains_and_rolls_out_on_unseen_instance():
    dataset = build_cpsat_dataset(
        [10, 11], n_jobs=4, n_machines=3, operations_per_job=2, time_limit_seconds=0.4
    )
    assert dataset.features.shape[0] == dataset.labels.shape[0]
    assert dataset.features.shape[1] == 9
    assert dataset.labels.sum() > 0

    policy = LinearBehaviorCloning().fit(dataset.features, dataset.labels)
    jobs, n_machines = random_instance(99, n_jobs=4, n_machines=3, operations_per_job=2)
    transitions, metrics = rollout(FJSPEnv(jobs, n_machines), policy.act)
    assert len(transitions) == 8
    assert metrics["makespan"] > 0
