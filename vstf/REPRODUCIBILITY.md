# VSTF HMM Benchmark Reproducibility Manifest

Python: 3.9.6
Platform: macOS-26.6.2-arm64-arm-64bit
NumPy: 2.0.2
ReportLab: 5.0.0
Master seed: 20260917
Trajectories per batch: 10000
Monte Carlo batches: 30
Sensitivity batches per cell: 3
Ablation batches per cell: 8
Event-sample rows: 40

Reproduction command:

```bash
python3 vstf_hmm_benchmark.py
```

SHA-256 outputs:

- `vstf_hmm_benchmark_summary.csv`: `1507b8938c6a76d2f633e6ebb1e2e2cca2042f52563c20d92b37d3055879f4f7`
- `vstf_hmm_benchmark_monte_carlo.csv`: `b22692b8c872d0bc0cd7adab5765da77a57dd7a780486e024322783b4ab835c7`
- `vstf_hmm_benchmark_sensitivity.csv`: `8e99d8586fec9e4b6e100b65d16935fa05676ff274b7df60b0dc0dc0acf5e763`
- `vstf_hmm_benchmark_ablations.csv`: `8d584bc46d859d68c1fac365ab337917b8f41fed7e7323c803deaf02e0346ec3`
- `vstf_hmm_random_parameter_stress.csv`: `ff490183eeb75473c722a2ff7973857182e12714a7cb5225d45d6da96f65fa38`
- `vstf_hmm_scenario_stress.csv`: `8ba355bf7dcb94462041b85a880bb9a95037f65efd919b5ebf5768f833f165e0`
- `vstf_hmm_event_sample.csv`: `db88a8c971d834cb304758dfe6cccb57331298db82d052d0b5036c8626782b46`
- `vstf_event_cascade.pdf`: `064d323e05c5b5e6c07acc0edc28c9c97e58da8f6d5f717ca51f56100b4ccae8`
- `vstf_hmm_benchmark.pdf`: `3a08c8c754ebf4bea6c33f7543c6afb2f7ef82aad3b032b403b924caec2cdaf0`
