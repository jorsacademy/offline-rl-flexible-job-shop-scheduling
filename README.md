# Offline RL for Flexible Job Shop Scheduling

Research-oriented Industrial Engineering / Operations Research project studying whether policies learned **offline from OR-generated scheduling decisions** can approach strong optimization and dispatching baselines without online exploration.

## Research question

Can a learned dispatch policy imitate strong CP-SAT decisions well enough to preserve scheduling quality on previously unseen Flexible Job Shop Scheduling Problem (FJSP) instances, and how does that trade off against solver latency and transparent dispatching rules?

## Current status

**Phase 2 implemented: OR expert demonstrations + held-out behavior-cloning benchmark.**

The repository now contains a reproducible synthetic FJSP generator, deterministic event-driven simulator, three dispatching baselines, a real OR-Tools CP-SAT expert, candidate-level demonstration logging, a generalizable linear behavior-cloning ranker, a conservative tabular offline-Q reference, held-out train/test instance blocks, tests and GitHub Actions CI.

The project does not claim that offline RL is already superior. CQL/IQL and the larger frozen validation/final campaigns remain future research stages.

## Implemented decision stack

1. **Transparent dispatching** — shortest processing time, earliest due date and minimum slack.
2. **OR expert** — CP-SAT with alternative-machine assignment, precedence, no-overlap constraints and a weighted-tardiness objective with makespan tie-breaking.
3. **Offline demonstrations** — every expert decision state logs all feasible candidate actions; the expert-selected candidate is explicitly labeled.
4. **Behavior cloning** — ridge-regularized linear candidate ranker trained only on demonstration features.
5. **Offline-Q reference** — conservative tabular learner retained for small-state diagnostics, not treated as the main generalization result.

## Why candidate-level logging matters

A common offline-RL mistake is to treat actions absent from a trajectory as if their value were known. This project instead records the entire feasible candidate set at each expert decision epoch. The behavior-cloning baseline therefore learns a ranking problem from observed expert choices while preserving the distinction between demonstrated and merely feasible actions.

## FJSP model

Each job contains multiple precedence-constrained operations. Every operation has a subset of eligible machines and machine-dependent processing times. A scheduling action selects `(job_id, machine_id)` for the next operation of a job. The simulator executes the assignment at the earliest precedence- and machine-feasible start time.

Primary KPI: **weighted tardiness**.

Secondary KPIs currently include makespan; decision and solver latency are part of the next benchmark expansion.

## Reproducible experiment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m offline_fjsp.experiment
```

The smoke benchmark uses separate instance blocks:

- training demonstrations: seeds `100-102`;
- held-out test: seeds `200-202`.

These small blocks keep CI fast. Larger research campaigns should define independent development, validation and final-test blocks before tuning begins.

## Repository map

```text
src/offline_fjsp/
  model.py              # deterministic FJSP simulator
  generator.py          # seeded synthetic instance generator
  policies.py           # SPT, EDD, minimum-slack baselines
  cpsat_expert.py       # OR-Tools CP-SAT expert scheduler
  features.py           # candidate action representation
  dataset.py            # CP-SAT demonstration dataset builder
  behavior_cloning.py   # generalizable candidate-ranking baseline
  offline_q.py          # conservative tabular offline-Q reference
  benchmark.py          # held-out controller comparison
  experiment.py         # reproducible CLI smoke benchmark

tests/
  test_core.py
  test_research_pipeline.py
configs/
  experiment.json
.github/workflows/
  ci.yml
```

## Scientific evaluation contract

- strong non-learning baselines precede learning claims;
- instance seeds are partitioned before evaluation;
- expert decisions and candidate support are logged explicitly;
- feasibility is mandatory;
- model selection must not use test seeds;
- optimization quality and computation cost must eventually be reported together;
- negative learning results are retained rather than tuned away.

## Next research stages

### Phase 3 — dataset quality study
Generate CP-SAT expert-only, mixed heuristic/expert, deliberately noisy and low-quality datasets. Quantify how policy quality degrades with demonstration quality and coverage.

### Phase 4 — modern offline RL
Add neural CQL and IQL with masked discrete candidate actions. Compare them against behavior cloning rather than assuming value learning is necessary.

### Phase 5 — frozen validation and final test
Introduce development, validation and one-time final blocks; multi-seed training; bootstrap confidence intervals; paired inference; latency and reliability accounting.

### Phase 6 — OOD stress campaign
Increase job count and utilization, tighten due dates, alter eligibility density and shift processing-time distributions. Measure generalization separately from nominal performance.

## Success criterion

A learned policy earns promotion only if it is feasible on every final instance and delivers a defensible decision-quality/latency trade-off against strong dispatching rules and CP-SAT. If CP-SAT or a heuristic remains superior, that result is the operational recommendation.

## License

MIT
