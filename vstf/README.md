# Valid State Transition Framework

This repository contains the manuscript and reproducibility materials for:

**Valid State Transition Framework: A Pre-Specified Event-Level Adjudication Framework for Stabilized Outcomes**

## Files

- `main.tex` - LaTeX manuscript.
- `vstf_hmm_benchmark.py` - deterministic hidden-Markov simulation benchmark with multi-seed robustness, event-level baselines, latent-ground-truth error rates, sensitivity sweep, controlled ablations, random-parameter stress, correlated-domain stress, HSMM stress, and figure generation.
- `vstf_hmm_benchmark_summary.csv` - aggregate benchmark results.
- `vstf_hmm_benchmark_monte_carlo.csv` - multi-seed robustness summary.
- `vstf_hmm_benchmark_sensitivity.csv` - sensitivity sweep over observation noise and retention window.
- `vstf_hmm_benchmark_ablations.csv` - controlled A/B ablation summary.
- `vstf_hmm_random_parameter_stress.csv` - random-parameter A/B stress test over transition, noise, domain-validity, and energy-ledger ranges.
- `vstf_hmm_scenario_stress.csv` - correlated-domain and HSMM scenario stress tests.
- `vstf_hmm_event_sample.csv` - machine-readable event-level audit sample showing segmentation, latent truth, and operational decisions.
- `vstf_event_cascade.pdf` - generated event-cascade and claim-governance workflow diagram.
- `vstf_hmm_benchmark.pdf` - generated benchmark figure referenced by the manuscript.
- `REPRODUCIBILITY.md` - Python/library versions, seed policy, reproduction command, and SHA-256 output hashes.
- `requirements.txt` - minimal Python package requirements.

## Reproducibility

Run:

```bash
python3 vstf_hmm_benchmark.py
```

The script uses a fixed master seed and regenerates the CSV summaries, event-level audit sample, reproducibility manifest, workflow figure, and benchmark figure.

The benchmark is a synthetic methodological demonstration. It is not a biological, clinical, or hardware validation study.
