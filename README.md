# Offline RL for Flexible Job Shop Scheduling

Research-oriented Industrial Engineering / Operations Research project studying whether policies learned **offline from OR-generated scheduling decisions** can approach strong optimization and dispatching baselines without online exploration.

## Research question

Can offline policies trained from a mixture of CP-SAT expert and heuristic trajectories generalize to unseen FJSP instances, and when do value-based methods add anything beyond behavior cloning?

## Current status

**Phase 3 implemented: mixed-quality offline dataset + BC/CQL/IQL validation and OOD benchmark.**

The repository contains a seeded synthetic FJSP generator, deterministic event-driven simulator, dispatching baselines, a real OR-Tools CP-SAT expert, candidate-level expert demonstrations, mixed-quality logged trajectories with policy provenance, linear behavior cloning, auditable fitted CQL and expectile-IQL baselines, held-out validation/OOD blocks, tests and GitHub Actions CI.

The CQL and IQL implementations are deliberately linear rather than presented as production-scale neural agents. Their role is to make the offline-RL objectives inspectable and to establish a reproducible research baseline before any neural extension.

## Implemented decision stack

1. **Transparent dispatching** — shortest processing time, earliest due date and minimum slack.
2. **OR expert** — CP-SAT with alternative-machine assignment, precedence, no-overlap constraints and weighted-tardiness minimization with makespan tie-breaking.
3. **Expert demonstrations** — every CP-SAT decision state logs all feasible candidate actions.
4. **Mixed-quality offline data** — CP-SAT, minimum slack, EDD and SPT trajectories are retained with explicit behavior-policy provenance.
5. **Behavior cloning** — ridge-regularized candidate ranker trained on expert candidate labels.
6. **Linear CQL** — fitted conservative Q-learning over action features with a conservative penalty on feasible alternatives.
7. **Linear IQL** — fitted Q/value learning with asymmetric expectile regression.
8. **Tabular offline-Q reference** — retained only for small-state diagnostics.

## Offline reward contract

The simulator's native step reward is intentionally not repurposed for the modern offline-RL benchmark. Instead, logged trajectories use sparse terminal utility

`-(weighted_tardiness + 0.02 * makespan)`

with zero intermediate reward. This keeps the learning objective aligned with the scheduling KPI and avoids silently optimizing processing time instead of tardiness.

## Reproducible experiments

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m offline_fjsp.experiment
python -m offline_fjsp.phase3_experiment
```

The Phase-3 runner uses independent seed blocks for training, validation and OOD evaluation. Model fitting only touches the training block. CP-SAT remains a benchmark controller on held-out instances.

## Repository map

```text
src/offline_fjsp/
  model.py              # deterministic FJSP simulator
  generator.py          # seeded synthetic instance generator
  policies.py           # SPT, EDD, minimum-slack baselines
  cpsat_expert.py       # OR-Tools CP-SAT expert scheduler
  features.py           # candidate action representation
  dataset.py            # candidate-labeled CP-SAT demonstrations
  offline_dataset.py    # mixed expert/heuristic transition dataset
  behavior_cloning.py   # expert imitation baseline
  cql.py                # auditable fitted linear CQL
  iql.py                # auditable fitted linear IQL
  offline_q.py          # conservative tabular diagnostic baseline
  phase3_experiment.py  # BC/CQL/IQL/heuristic/CP-SAT benchmark
  experiment.py         # Phase-2 smoke benchmark
tests/
  test_core.py
  test_research_pipeline.py
  test_phase3.py
configs/
  experiment.json
.github/workflows/
  ci.yml
```

## Scientific evaluation contract

- strong non-learning baselines precede learning claims;
- dataset provenance is explicit;
- train, validation and OOD instance blocks are separated before model fitting;
- feasibility is mandatory;
- model selection must not use final/OOD seeds;
- negative results are retained;
- CQL/IQL are not considered superior merely because they are more complex;
- decision quality and solve/inference latency must eventually be reported together.

## Next research stages

### Phase 4 — dataset-quality ablation
Construct expert-only, expert-heavy, balanced, heuristic-heavy and deliberately corrupted datasets. Measure policy degradation against data quality and action-support coverage.

### Phase 5 — neural masked offline RL
Add PyTorch masked discrete CQL/IQL only after the linear baselines are frozen. Compare neural models against BC and linear value learning, not only against weak heuristics.

### Phase 6 — frozen final evaluation
Add a one-time final seed block, multi-seed training, bootstrap confidence intervals, paired tests, effect sizes, latency and reliability accounting.

### Phase 7 — OOD stress campaign
Increase job count and utilization, tighten due dates, alter machine eligibility density and shift processing-time distributions. Generalization must be reported separately from nominal performance.

## Success criterion

A learned policy earns promotion only if it is feasible on every final instance and produces a defensible quality/latency trade-off against CP-SAT and strong dispatching rules. If behavior cloning or a heuristic remains better than CQL/IQL, that result is retained as the operational recommendation.

## License

MIT
