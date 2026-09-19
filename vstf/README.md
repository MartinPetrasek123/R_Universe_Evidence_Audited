# Valid State Transition Framework

This repository contains the manuscript and reproducibility package for:

**Valid State Transition Framework: A Pre-Specified Event-Level Adjudication Framework for Stabilized Outcomes**

## Scope

The package supports a methodological framework paper. The synthetic HMM benchmark is a controlled proof-of-use, not a biological, clinical, or hardware validation study. The real-data files are component audits: they show that the acceptance layer, pending status, temporal retention, first-exit rejection, cost-normalized retained-event denominator, and treated-versus-comparison distinction can be computed on public datasets.

The real-data audits do not prove clinical utility, causal treatment benefit, or deployment readiness.

## One-command reproduction

Run from this directory:

```bash
python3 reproduce_all.py
```

The workflow regenerates all synthetic CSV/PDF outputs, downloads or reuses public real datasets, regenerates all real-data ledgers, validates the sensitivity table values used by the manuscript, and writes `EVIDENCE_MANIFEST.sha256`.

## Core files

- `main.tex` - LaTeX manuscript.
- `reproduce_all.py` - single evidence-package regeneration workflow.
- `requirements.txt` - pinned minimal Python requirements.
- `EVIDENCE_MANIFEST.sha256` - checksums for manuscript, scripts, generated CSV/PDF outputs, and requirements.
- `FULL_VALIDATION_PROTOCOL.md` - domain-level requirements for any future claim of full validation.

## Synthetic benchmark

- `vstf_hmm_benchmark.py` - deterministic hidden-Markov simulation benchmark with multi-seed robustness, event-level baselines, matched/reference-validity error rates, operation-based cost accounting, sensitivity sweep, sampling-resolution stress, controlled ablations, random-parameter stress, correlated-domain stress, HSMM stress, model-misspecification stress, many-system rank-discordance stress, and figure generation.
- `vstf_hmm_benchmark_summary.csv` - aggregate benchmark results.
- `vstf_hmm_benchmark_monte_carlo.csv` - multi-seed robustness summary.
- `vstf_hmm_benchmark_batch_level.csv` - all 30 HMM batch-level results for both benchmark systems.
- `vstf_hmm_benchmark_sensitivity.csv` - 30-batch-per-cell sensitivity sweep over observation noise and retention window.
- `vstf_hmm_sampling_resolution.csv` - sampling-grid stress test.
- `vstf_hmm_benchmark_ablations.csv` - controlled A/B ablation summary.
- `vstf_hmm_random_parameter_stress.csv` - random-parameter A/B stress test.
- `vstf_hmm_scenario_stress.csv` - correlated-domain and HSMM scenario stress tests.
- `vstf_model_misspecification_stress.csv` - model-misspecification and reference-error stress tests.
- `vstf_many_system_benchmark.csv` - 30-system synthetic benchmark summary.
- `vstf_many_system_discordance.csv` - pairwise rank-discordance summary.
- `vstf_hmm_event_sample.csv` - machine-readable event-level audit sample.
- `vstf_event_cascade.pdf` - generated event-cascade workflow diagram.
- `vstf_hmm_benchmark.pdf` - generated benchmark figure.
- `vstf_many_system_rank_displacement.pdf` - generated many-system retained-rank versus VSTE-rank diagnostic.

## Public real-data component audits

- `vstf_real_datasets_validation.py` - locked multi-dataset real-data acceptance audit.
- `vstf_real_datasets_summary.csv` - aggregate real-data classification audit.
- `vstf_real_datasets_folds.csv` - fold-level real-data classification audit.
- `vstf_real_component_validation.py` - temporal, cost-normalized, and treated-versus-comparison component audit.
- `vstf_real_component_validation_summary.csv` - component-level real-data summary.
- `vstf_household_power_events.csv` - event-level household-power time-series ledger.
- `vstf_lalonde_treatment_effect.csv` - descriptive Lalonde/MatchIt treated-versus-PSID comparison ledger.
- `vstf_full_real_calculation_ledger.csv` - unified real-calculation ledger.

Public source datasets:

- WDBC: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
- Ionosphere: https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data
- Sonar: https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data
- Cleveland Heart Disease: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
- Haberman Survival: https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data
- UCI household electric power consumption: https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip
- Lalonde/MatchIt: https://vincentarelbundock.github.io/Rdatasets/csv/MatchIt/lalonde.csv

## Interpretation guardrails

- The current HMM sensitivity table reports raw-to-VSTE reversal probabilities exactly as regenerated from `vstf_hmm_benchmark_sensitivity.csv`: 0.00 for all cells at sigma factors 0.75 and 1.00, and 0.4333, 0.5333, 0.8333 at sigma factor 1.25 for retention windows 4, 6, and 10.
- The Lalonde/MatchIt calculation is descriptive. The 429 comparison observations are PSID comparison cases, not randomized NSW controls. The unadjusted difference must not be interpreted as a causal treatment effect without a separate locked matching, weighting, balance, and sensitivity analysis.
- Static classification datasets audit the acceptance/pending layer. They do not validate temporal retention or cost-normalized VSTE; those components are audited separately with the household-power time series.
