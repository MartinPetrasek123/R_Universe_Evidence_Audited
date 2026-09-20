# VSTF Full-Validation Protocol

This file defines what would count as full validation of the Valid State
Transition Framework beyond the component-level audits in the present package.

## Current status

The current evidence package establishes reproducibility and component-level
computability:

- synthetic HMM benchmark and stress tests are reproducible from code;
- static real-data acceptance and pending decisions are computed on five public datasets;
- temporal event segmentation, dwell, first-exit retention, and cost-normalized VSTE are computed on a public household-power time series;
- a descriptive treated-versus-PSID comparison is computed on the Lalonde/MatchIt data.
- additional external component checks are computed on UCI Banknote Authentication, UCI Spambase, UCI Appliances Energy Prediction, and Project STAR and are included in the one-command reproduction pipeline.

This is not full clinical, biological, or hardware validation.

## Required elements for full validation

Each domain-specific validation study must lock the following before outcome
inspection:

1. Scientific claim and claim level: detection, retained outcome, cost-normalized efficiency, prediction utility, intervention response, or causal benefit.
2. State representation and measurement model.
3. State regions and admissible target transitions.
4. Candidate-event segmentation rule.
5. Dwell, first-exit, retention, and censoring rules.
6. Domain-validity predicate and reference-standard matching rule.
7. Full denominator and cost boundary.
8. Comparator baselines: raw activity, first-passage/crossing counts, ordinary yield, standard prediction models, or domain-specific accepted metrics.
9. Primary endpoint and secondary endpoints.
10. Missing-data, censoring, and intercurrent-event rules.
11. Uncertainty method, sensitivity grid, and stopping/failure criteria.
12. Reproducibility package with code, checksums, and immutable release identifier.

## Minimum validation designs

### Static decision tasks

Use external datasets or temporal holdout datasets. Report raw accuracy,
accepted accuracy, coverage, pending fraction, calibration, decision utility,
and comparison with abstention/rejection baselines.

### Time-series transition tasks

Use time-indexed data with observed follow-up. Report candidate events, dwell
passes, first exits, retained events, rejected events, censoring, sampling-rate
sensitivity, and missing-data sensitivity.

### Cost-normalized tasks

Use a complete denominator ledger. Report accepted retained events per unit
energy, cost, time, dose, or risk exposure, and compare against ordinary yield
and event rate.

### Intervention or causal tasks

Use a locked causal estimand. Randomized studies are preferred. Observational
studies require pre-specified covariates, positivity checks, matching or
weighting, balance diagnostics, standard-error method, and sensitivity to
unmeasured confounding.

## Release rule

A VSTF domain claim should be marked "fully validated" only if the manuscript,
source code, generated outputs, data provenance, and checksums are all tied to
one immutable release or DOI and the full validation protocol above is satisfied
for that domain.

The current working package is reproducible and component-audited, but a journal
submission should still be frozen as a versioned GitHub release, Zenodo DOI, or
equivalent immutable archive. The release should state the code/data license
explicitly.

The current package includes explicit reuse terms in `LICENSE.md` and a
GitHub Actions workflow that reruns the public reproducibility pipeline. These
engineering controls support auditability; they do not convert component-level
evidence into full domain validation.
