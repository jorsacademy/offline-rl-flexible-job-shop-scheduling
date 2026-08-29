from offline_fjsp.model import FJSPEnv, toy_instance
from offline_fjsp.offline_q import ConservativeOfflineQ
from offline_fjsp.policies import rollout, shortest_processing_time


def test_rollout_finishes_all_operations():
    jobs, n_machines = toy_instance()
    env = FJSPEnv(jobs, n_machines)
    transitions, metrics = rollout(env, shortest_processing_time)
    assert len(transitions) == sum(len(j.operations) for j in jobs)
    assert metrics["makespan"] > 0
    assert len(env.completion) == len(jobs)


def test_infeasible_action_rejected():
    jobs, n_machines = toy_instance()
    env = FJSPEnv(jobs, n_machines)
    try:
        env.step((0, 99))
    except ValueError:
        pass
    else:
        raise AssertionError("infeasible action should raise ValueError")


def test_offline_q_can_replay_logged_support():
    jobs, n_machines = toy_instance()
    env = FJSPEnv(jobs, n_machines)
    transitions, _ = rollout(env, shortest_processing_time)
    learner = ConservativeOfflineQ().fit(transitions, epochs=10)
    env.reset()
    action = learner.act(env)
    assert action in env.feasible_actions()
