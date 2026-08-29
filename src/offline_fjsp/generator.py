from __future__ import annotations

from numpy.random import default_rng

from .model import Job, Operation


def random_instance(
    seed: int,
    n_jobs: int = 6,
    n_machines: int = 4,
    operations_per_job: int = 3,
    eligibility: int = 2,
) -> tuple[list[Job], int]:
    """Generate a deterministic synthetic FJSP instance from a seed.

    Each operation is eligible on ``eligibility`` machines with integer processing
    times. Due dates are tied to the job's minimum-work content so that tardiness is
    meaningful without making every instance trivially early or late.
    """
    if not 1 <= eligibility <= n_machines:
        raise ValueError("eligibility must be between 1 and n_machines")

    rng = default_rng(seed)
    jobs: list[Job] = []
    for job_id in range(n_jobs):
        operations: list[Operation] = []
        min_work = 0
        for op_idx in range(operations_per_job):
            machines = sorted(rng.choice(n_machines, size=eligibility, replace=False).tolist())
            processing = {int(m): int(rng.integers(2, 12)) for m in machines}
            min_work += min(processing.values())
            operations.append(Operation(job_id, op_idx, processing))
        due_date = round(min_work * float(rng.uniform(1.6, 2.6)))
        weight = float(rng.choice([1.0, 1.5, 2.0, 3.0]))
        jobs.append(Job(job_id, due_date, weight, tuple(operations)))
    return jobs, n_machines
