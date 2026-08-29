from dataclasses import dataclass


@dataclass(frozen=True)
class Operation:
    job: int
    index: int
    processing_times: dict[int, int]


@dataclass(frozen=True)
class Job:
    job_id: int
    due_date: int
    weight: float
    operations: tuple[Operation, ...]


class FJSPEnv:
    """Small deterministic event-driven FJSP environment.

    An action is (job_id, machine_id) for the next unscheduled operation of a job.
    The simulator schedules that operation at the earliest precedence- and machine-feasible time.
    """

    def __init__(self, jobs: list[Job], n_machines: int):
        self.jobs = {j.job_id: j for j in jobs}
        self.n_machines = n_machines
        self.reset()

    def reset(self):
        self.next_op = {j: 0 for j in self.jobs}
        self.job_ready = {j: 0 for j in self.jobs}
        self.machine_ready = {m: 0 for m in range(self.n_machines)}
        self.completion = {}
        self.history = []
        return self.state()

    def feasible_actions(self):
        actions = []
        for job_id, idx in self.next_op.items():
            job = self.jobs[job_id]
            if idx < len(job.operations):
                op = job.operations[idx]
                actions.extend((job_id, m) for m in op.processing_times)
        return sorted(actions)

    def state(self):
        return (
            tuple(self.next_op[j] for j in sorted(self.jobs)),
            tuple(self.job_ready[j] for j in sorted(self.jobs)),
            tuple(self.machine_ready[m] for m in range(self.n_machines)),
        )

    def step(self, action):
        job_id, machine = action
        if action not in self.feasible_actions():
            raise ValueError(f"infeasible action: {action}")
        idx = self.next_op[job_id]
        op = self.jobs[job_id].operations[idx]
        start = max(self.job_ready[job_id], self.machine_ready[machine])
        finish = start + op.processing_times[machine]
        self.job_ready[job_id] = finish
        self.machine_ready[machine] = finish
        self.next_op[job_id] += 1
        self.history.append((job_id, idx, machine, start, finish))
        if self.next_op[job_id] == len(self.jobs[job_id].operations):
            self.completion[job_id] = finish
        done = not self.feasible_actions()
        reward = -float(finish - start)
        return self.state(), reward, done, {"start": start, "finish": finish}

    def metrics(self):
        makespan = max(self.machine_ready.values(), default=0)
        weighted_tardiness = 0.0
        for job_id, completion in self.completion.items():
            job = self.jobs[job_id]
            weighted_tardiness += job.weight * max(0, completion - job.due_date)
        return {"makespan": makespan, "weighted_tardiness": weighted_tardiness}


def toy_instance():
    jobs = [
        Job(
            0,
            9,
            2.0,
            (
                Operation(0, 0, {0: 3, 1: 5}),
                Operation(0, 1, {1: 2, 2: 4}),
            ),
        ),
        Job(
            1,
            10,
            1.0,
            (
                Operation(1, 0, {0: 4, 2: 3}),
                Operation(1, 1, {1: 4, 2: 2}),
            ),
        ),
        Job(
            2,
            12,
            1.5,
            (
                Operation(2, 0, {1: 3, 2: 5}),
                Operation(2, 1, {0: 2, 2: 3}),
            ),
        ),
    ]
    return jobs, 3
