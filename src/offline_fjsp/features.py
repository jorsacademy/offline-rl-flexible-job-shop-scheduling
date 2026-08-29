from __future__ import annotations

import numpy as np

from .model import FJSPEnv


def action_features(env: FJSPEnv, action: tuple[int, int]) -> np.ndarray:
    """Return scale-friendly operational features for a feasible dispatch action."""
    job_id, machine = action
    job = env.jobs[job_id]
    op_idx = env.next_op[job_id]
    op = job.operations[op_idx]
    processing = op.processing_times[machine]
    remaining_min_work = sum(
        min(future.processing_times.values()) for future in job.operations[op_idx:]
    )
    earliest_start = max(env.job_ready[job_id], env.machine_ready[machine])
    slack = job.due_date - earliest_start - remaining_min_work
    progress = op_idx / max(1, len(job.operations) - 1)
    machine_load = env.machine_ready[machine]
    return np.asarray(
        [
            1.0,
            float(processing),
            float(job.due_date),
            float(job.weight),
            float(earliest_start),
            float(machine_load),
            float(remaining_min_work),
            float(slack),
            float(progress),
        ],
        dtype=float,
    )
