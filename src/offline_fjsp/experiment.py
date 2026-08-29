from .model import FJSPEnv, toy_instance
from .offline_q import ConservativeOfflineQ
from .policies import earliest_due_date, minimum_slack, rollout, shortest_processing_time


def main():
    jobs, n_machines = toy_instance()
    policies = {
        "spt": shortest_processing_time,
        "edd": earliest_due_date,
        "minimum_slack": minimum_slack,
    }
    logged = []
    print("baseline,makespan,weighted_tardiness")
    for name, policy in policies.items():
        env = FJSPEnv(jobs, n_machines)
        transitions, metrics = rollout(env, policy)
        logged.extend(transitions)
        print(f"{name},{metrics['makespan']},{metrics['weighted_tardiness']:.3f}")

    learner = ConservativeOfflineQ().fit(logged, epochs=200)
    env = FJSPEnv(jobs, n_machines)
    _, metrics = rollout(env, learner.act)
    print(f"offline_q,{metrics['makespan']},{metrics['weighted_tardiness']:.3f}")


if __name__ == "__main__":
    main()
