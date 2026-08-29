from __future__ import annotations

from dataclasses import dataclass

from ortools.sat.python import cp_model

from .model import Job


@dataclass(frozen=True)
class ExpertDecision:
    job_id: int
    operation_index: int
    machine_id: int
    start: int
    end: int


def solve_cpsat_expert(
    jobs: list[Job],
    n_machines: int,
    time_limit_seconds: float = 2.0,
    seed: int = 0,
) -> list[ExpertDecision]:
    """Solve a static FJSP with weighted tardiness + small makespan tie-breaker."""
    model = cp_model.CpModel()
    horizon = sum(max(op.processing_times.values()) for job in jobs for op in job.operations)

    starts: dict[tuple[int, int], cp_model.IntVar] = {}
    ends: dict[tuple[int, int], cp_model.IntVar] = {}
    machine_presence: dict[tuple[int, int, int], cp_model.BoolVar] = {}
    machine_intervals: dict[int, list[cp_model.IntervalVar]] = {
        machine: [] for machine in range(n_machines)
    }

    for job in jobs:
        for op in job.operations:
            key = (job.job_id, op.index)
            starts[key] = model.new_int_var(0, horizon, f"start_{job.job_id}_{op.index}")
            ends[key] = model.new_int_var(0, horizon, f"end_{job.job_id}_{op.index}")
            presences = []
            for machine, duration in op.processing_times.items():
                present = model.new_bool_var(f"present_{job.job_id}_{op.index}_{machine}")
                interval = model.new_optional_interval_var(
                    starts[key],
                    duration,
                    ends[key],
                    present,
                    f"interval_{job.job_id}_{op.index}_{machine}",
                )
                machine_presence[(job.job_id, op.index, machine)] = present
                machine_intervals[machine].append(interval)
                presences.append(present)
            model.add_exactly_one(presences)

        for previous, current in zip(job.operations, job.operations[1:]):
            model.add(starts[(job.job_id, current.index)] >= ends[(job.job_id, previous.index)])

    for intervals in machine_intervals.values():
        model.add_no_overlap(intervals)

    tardiness_terms = []
    completion_vars = []
    for job in jobs:
        completion = ends[(job.job_id, job.operations[-1].index)]
        completion_vars.append(completion)
        tardiness = model.new_int_var(0, horizon, f"tardiness_{job.job_id}")
        model.add(tardiness >= completion - job.due_date)
        model.add(tardiness >= 0)
        weight = round(job.weight * 10)
        tardiness_terms.append(weight * tardiness)

    makespan = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(makespan, completion_vars)
    model.minimize(sum(tardiness_terms) * 100 + makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT did not find a feasible FJSP schedule")

    decisions: list[ExpertDecision] = []
    for job in jobs:
        for op in job.operations:
            machine = next(
                m
                for m in op.processing_times
                if solver.value(machine_presence[(job.job_id, op.index, m)])
            )
            decisions.append(
                ExpertDecision(
                    job.job_id,
                    op.index,
                    machine,
                    solver.value(starts[(job.job_id, op.index)]),
                    solver.value(ends[(job.job_id, op.index)]),
                )
            )
    return sorted(decisions, key=lambda d: (d.start, d.end, d.job_id, d.operation_index))
