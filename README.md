# Offline RL for Flexible Job Shop Scheduling

Research-oriented Industrial Engineering / Operations Research project studying whether policies learned **offline from OR-generated scheduling decisions** can approach strong optimization and dispatching baselines without online exploration.

## Research question

How sensitive are offline scheduling policies to demonstration quality, and do masked neural CQL/IQL models add value beyond behavior cloning, linear offline RL and strong dispatching heuristics?

## Current status

**Phase 4 implemented: dataset-quality ablation + masked neural CQL/IQL.**

The repository contains a seeded FJSP generator, deterministic simulator, dispatching baselines, an OR-Tools CP-SAT expert, candidate-level demonstrations, mixed-quality transition datasets with explicit behavior-policy provenance, behavior cloning, linear CQL/IQL, PyTorch neural CQL/IQL, validation/OOD splits, tests and CI.

The project does not assume that neural offline RL should win. Added complexity earns promotion only through held-out scheduling quality, feasibility and eventually latency/reliability evidence.

## Decision stack

1. transparent SPT, EDD and minimum-slack dispatching;
2. CP-SAT expert scheduling;
3. expert candidate-ranking demonstrations;
4. mixed offline trajectories from CP-SAT, heuristics and explicit random corruption;
5. linear behavior cloning;
6. auditable linear CQL and IQL;
7. masked neural CQL and neural IQL over variable-size feasible action sets.

## Offline data contract

Every logged transition stores:

- selected action features;
- **all feasible current-action features**;
- all feasible next-action features;
- terminal reward;
- done flag;
- behavior-policy name;
- instance seed.

This matters because the CQL conservative penalty must be evaluated on the current feasible action set, not accidentally on next-state actions.

Trajectory utility is

`-(weighted_tardiness + 0.02 * makespan)`

with zero intermediate reward.

## Dataset-quality ablation

Phase 4 freezes four dataset families:

- `expert_only`: CP-SAT only;
- `expert_strong`: CP-SAT + minimum slack;
- `balanced`: CP-SAT + minimum slack + EDD + SPT;
- `corrupted`: balanced data + an explicitly random behavior policy.

Random trajectories are never hidden inside an unlabeled mixture. Provenance remains machine-readable so performance degradation can be attributed to the data regime.

## Neural masked offline RL

`NeuralCQL` and `NeuralIQL` are compact PyTorch MLP baselines. Candidate sets have variable size, so policies score only currently feasible actions rather than learning over a fixed padded action catalog.

Neural support is an optional dependency:

```bash
pip install -e '.[dev,neural]'
```

The normal OR/linear test matrix stays lightweight; a dedicated Python 3.11 CI job installs PyTorch and runs the neural smoke benchmark.

## Reproducible experiments

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m offline_fjsp.experiment
python -m offline_fjsp.phase3_experiment

# Neural Phase 4
pip install -e '.[dev,neural]'
python -m offline_fjsp.phase4_experiment
```

## Repository map

```text
src/offline_fjsp/
  model.py
  generator.py
  policies.py
  cpsat_expert.py
  features.py
  dataset.py
  offline_dataset.py      # current/next feasible support + provenance
  behavior_cloning.py
  cql.py                  # linear CQL
  iql.py                  # linear IQL
  neural_offline.py       # masked PyTorch CQL/IQL
  phase3_experiment.py
  phase4_experiment.py    # dataset-quality neural ablation
  experiment.py
tests/
  test_core.py
  test_research_pipeline.py
  test_phase3.py
  test_phase4.py
configs/
  experiment.json
.github/workflows/
  ci.yml
```

## Scientific evaluation contract

- strong OR and heuristic baselines precede learning claims;
- behavior-policy provenance is explicit;
- current feasible action support is logged;
- train, validation and OOD seeds remain separate;
- feasibility is mandatory;
- model selection must not use final/OOD seeds;
- corrupted data is deliberate and labeled;
- neural models are compared against BC and linear value learning, not only weak baselines;
- negative results are retained.

## Next research stages

### Phase 5 — frozen final evaluation
Add a one-time final seed block, multiple training seeds, paired comparisons, bootstrap confidence intervals, effect sizes and inference/solver latency.

### Phase 6 — stronger OOD stress campaign
Vary job count, utilization, due-date tightness, eligibility density and processing-time distributions rather than using seed shift alone.

### Phase 7 — neural ablations
Study hidden width, CQL penalty, IQL expectile, reward design and action-feature subsets using validation data only.

## Success criterion

A learned controller is promoted only if it is feasible on every final instance and offers a defensible quality/latency trade-off against CP-SAT, behavior cloning and strong dispatching rules. If a simpler method remains superior, that is the operational recommendation.

## License

MIT
