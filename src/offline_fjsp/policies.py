def shortest_processing_time(env):
    actions = env.feasible_actions()
    return min(
        actions,
        key=lambda a: env.jobs[a[0]].operations[env.next_op[a[0]]].processing_times[a[1]],
    )


def earliest_due_date(env):
    return min(env.feasible_actions(), key=lambda a: (env.jobs[a[0]].due_date, a[0], a[1]))


def minimum_slack(env):
    def slack(action):
        job_id, _ = action
        job = env.jobs[job_id]
        remaining = 0
        for op in job.operations[env.next_op[job_id]:]:
            remaining += min(op.processing_times.values())
        return job.due_date - env.job_ready[job_id] - remaining

    return min(env.feasible_actions(), key=lambda a: (slack(a), a[0], a[1]))


def rollout(env, policy):
    env.reset()
    transitions = []
    done = False
    while not done:
        state = env.state()
        action = policy(env)
        next_state, reward, done, info = env.step(action)
        transitions.append((state, action, reward, next_state, done, info))
    return transitions, env.metrics()
