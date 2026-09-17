# Valid State Transition Framework

This repository contains the manuscript and reproducibility materials for:

**Valid State Transition Framework: An Operational Standard for Stabilized Outcomes in Physical, Biological, and Computational Systems**

## Files

- `main.tex` - LaTeX manuscript.
- `vstf_hmm_benchmark.py` - deterministic hidden-Markov simulation benchmark with multi-seed robustness, event-level baselines, latent-ground-truth error rates, sensitivity sweep, and controlled ablations.
- `vstf_hmm_benchmark_summary.csv` - aggregate benchmark results.
- `vstf_hmm_benchmark_monte_carlo.csv` - multi-seed robustness summary.
- `vstf_hmm_benchmark_sensitivity.csv` - sensitivity sweep over observation noise and retention window.
- `vstf_hmm_benchmark_ablations.csv` - controlled A/B ablation summary.
- `vstf_hmm_benchmark.pdf` - generated benchmark figure referenced by the manuscript.

## Reproducibility

Run:

```bash
python3 vstf_hmm_benchmark.py
```

The script uses a fixed master seed and regenerates the CSV summaries and PDF figure.

The benchmark is a synthetic methodological demonstration. It is not a biological, clinical, or hardware validation study.
