# Final Research Report — Offline RL for Flexible Job Shop Scheduling

## Scope

This repository evaluates whether policies trained offline from OR-generated and heuristic scheduling trajectories can provide a defensible quality/latency trade-off for Flexible Job Shop Scheduling (FJSP).

The project is deliberately benchmark-first. A learning method is not promoted because it is more sophisticated; it must earn its place against CP-SAT and strong dispatching rules on frozen unseen instances.

## Decision problem

Each job contains precedence-constrained operations. Every operation is eligible on a subset of machines with machine-dependent processing times. At each decision epoch, the controller chooses a `(job_id, machine_id)` pair for the next unscheduled operation of a job.

Primary objective:

- weighted tardiness.

Secondary metrics:

- makespan;
- episode-level controller latency;
- feasibility rate.

## Controllers

The final campaign compares:

1. Minimum Slack dispatching;
2. linear behavior cloning trained on CP-SAT demonstrations;
3. neural masked CQL trained from mixed offline trajectories;
4. neural IQL trained from the same offline support;
5. CP-SAT expert scheduling.

Linear CQL/IQL remain in the repository as auditable diagnostic baselines but the frozen final campaign focuses on the strongest learned variants plus strong non-learning references.

## Data and training contract

Training instances are generated only from seeds `10-19`.

Offline data includes:

- CP-SAT expert trajectories;
- Minimum Slack trajectories;
- Earliest Due Date trajectories;
- Shortest Processing Time trajectories.

Every transition retains:

- the selected action feature vector;
- the full feasible action set at the current state;
- the feasible action set at the next state;
- terminal utility;
- behavior-policy provenance;
- instance seed.

The offline reward is sparse terminal utility:

`-(weighted_tardiness + 0.02 * makespan)`.

## Frozen evaluation blocks

The final blocks are fixed in `configs/final_evaluation.json` and must not be used for hyperparameter tuning.

### Nominal final

Seeds `1000-1019` with the training instance structure.

### OOD scale

Seeds `1100-1109` with 8 jobs, 5 machines and 4 operations per job.

### OOD flexibility

Seeds `1200-1209` with reduced machine eligibility, creating a structurally different feasible-action distribution.

## Multi-seed learning

Neural models are trained independently with model seeds `0`, `1`, and `2`.

For paired statistical inference, learned-model results are first averaged across model seeds for each evaluation instance. This prevents treating training-seed replicates as independent scheduling instances.

## Statistical protocol

All lower-is-better weighted-tardiness comparisons are paired by instance seed against Minimum Slack.

Reported quantities:

- mean paired difference;
- paired 95% bootstrap confidence interval;
- median paired difference;
- win rate;
- exact two-sided sign-test p-value;
- sample size.

The benchmark does not interpret `p < 0.05` as sufficient evidence of operational superiority. Effect magnitude, latency and feasibility are considered jointly.

## Latency protocol

Episode latency is measured with `time.perf_counter()` around the complete controller rollout.

For CP-SAT, reported latency includes optimization plus replay of the resulting decisions through the common simulator. For learned and heuristic controllers, latency includes repeated action selection and simulation.

The numbers therefore represent end-to-end controller cost for the benchmark environment, not isolated neural-network forward-pass latency.

## Reliability requirement

A controller is not eligible for promotion unless feasibility rate is 1.0 on every frozen evaluation block.

Any exception, infeasible action, incomplete schedule or non-finite KPI is recorded as failure rather than silently dropped.

## Reproduction

Install the neural research dependencies:

```bash
pip install -e '.[dev,neural]'
```

Run all tests:

```bash
pytest -q
```

Run the frozen campaign:

```bash
python -m offline_fjsp.final_evaluation
```

The command prints aggregate KPI/latency results followed by paired inference against Minimum Slack.

## Interpretation rules

A learned controller earns a positive recommendation only when all of the following hold:

1. feasibility is 100% on nominal and OOD final blocks;
2. its weighted-tardiness difference against Minimum Slack is operationally meaningful;
3. the paired confidence interval supports the claimed direction;
4. performance does not collapse under either OOD block;
5. latency remains appropriate for dispatching use;
6. the conclusion remains stable across independent model seeds.

If CP-SAT remains clearly better and its solve latency is acceptable, CP-SAT is the recommended controller.

If Minimum Slack remains competitive with substantially lower latency, the heuristic is the recommended controller.

If behavior cloning matches value-based offline RL, the simpler imitation model is preferred.

Negative or null results are valid final outcomes.

## Reproducibility and auditability

The repository includes:

- deterministic instance seeds;
- deterministic model seeds;
- explicit train/final separation;
- strong OR and heuristic baselines;
- dataset provenance;
- unit tests;
- Python 3.10-3.12 CI for the base stack;
- dedicated neural CI smoke coverage;
- statistical inference code;
- latency and feasibility accounting.

## Final status

The research codebase is considered feature-complete for the stated research question. Further work should be treated as a new study—e.g. graph neural policies, industrial benchmark datasets, richer disturbances or live digital-twin integration—rather than silently expanding the scope of this repository.
