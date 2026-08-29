# Experimental Protocol

## Objective

Evaluate whether policies learned from offline scheduling demonstrations can deliver a useful quality/latency trade-off on FJSP instances not used for training or model selection.

## Data boundaries

The small CI experiment uses training seeds 100-102 and held-out smoke-test seeds 200-202. These blocks exist to validate the research pipeline, not to support a publication claim.

Before large experiments, freeze three disjoint seed ranges:

- development: environment debugging, feature design and implementation;
- validation: algorithm and hyperparameter selection;
- final test: opened once after controller selection is frozen.

Final-test instances must never be used for feature engineering, reward design, hyperparameter tuning or early stopping.

## Controller hierarchy

Every learning result must be compared with:

1. SPT;
2. EDD;
3. minimum slack;
4. CP-SAT expert/reference;
5. behavior cloning;
6. any additional offline-RL method under study.

CQL or IQL should not be promoted merely because they beat another neural method. They must improve the operational decision-quality/latency frontier relative to behavior cloning and strong non-learning baselines.

## Demonstration provenance

Each dataset version should record:

- generator version and seed range;
- instance dimensions and eligibility density;
- expert controller and solver version;
- CP-SAT time budget and solver seed;
- behavior-policy mixture if heuristic/noisy trajectories are included;
- number of decision states and candidate actions;
- expert KPI distribution.

## Dataset-quality experiments

Construct at least four offline datasets with identical instance distributions:

- expert-only CP-SAT demonstrations;
- mixed CP-SAT + strong heuristic demonstrations;
- mixed-quality demonstrations including weak heuristics;
- corrupted demonstrations with controlled action noise.

Report policy performance as a function of both dataset size and dataset quality.

## Primary and secondary outcomes

Primary outcome: weighted tardiness.

Secondary outcomes:

- makespan;
- mean flow time when added to the simulator;
- feasibility rate;
- per-decision inference latency;
- CP-SAT solve latency;
- probability that a learned controller beats the strongest fixed heuristic.

## Statistical reporting

For the final campaign use paired common instances across all controllers. Report:

- mean and median KPI;
- paired bootstrap confidence interval for controller differences;
- effect size;
- probability of superiority;
- failure/fallback counts;
- decision latency distribution.

A single best seed must never stand in for an algorithm-level result.

## OOD campaign

After nominal controller selection, test without retuning on shifts such as:

- more jobs;
- tighter due dates;
- higher utilization;
- lower machine eligibility density;
- processing-time scale shift;
- machine-specific slowdown.

Nominal and OOD conclusions must be reported separately.

## Promotion rule

A learned controller is portfolio/research-grade only when it:

1. remains feasible on every final instance;
2. is evaluated on frozen unseen instances;
3. is compared against strong OR and heuristic baselines;
4. has multi-seed evidence when stochastic training is introduced;
5. offers a defensible quality/latency trade-off.

If a heuristic or CP-SAT remains superior, that negative learning result is retained as the final operational recommendation.
