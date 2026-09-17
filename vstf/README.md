# Valid State Transition Framework

This repository contains the manuscript and reproducibility materials for:

**Valid State Transition Framework: An Operational Standard for Stabilized Outcomes in Physical, Biological, and Computational Systems**

## Files

- `main.tex` - LaTeX manuscript.
- `vstf_hmm_benchmark.py` - deterministic hidden-Markov simulation benchmark.
- `vstf_hmm_benchmark_summary.csv` - aggregate benchmark results.
- `vstf_hmm_benchmark.pdf` - generated benchmark figure referenced by the manuscript.

## Reproducibility

Run:

```bash
python3 vstf_hmm_benchmark.py
```

The script uses a fixed seed and regenerates both the CSV summary and PDF figure.

The benchmark is a synthetic methodological demonstration. It is not a biological, clinical, or hardware validation study.
