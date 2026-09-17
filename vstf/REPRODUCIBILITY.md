# VSTF HMM Benchmark Reproducibility Manifest

Python: 3.9.6
Platform: macOS-26.6.2-arm64-arm-64bit
NumPy: 2.0.2
ReportLab: 5.0.0
Master seed: 20260917
Trajectories per batch: 10000
Monte Carlo batches: 100
Sensitivity batches per cell: 5
Ablation batches per cell: 20
Event-sample rows: 40

Reproduction command:

```bash
python3 vstf_hmm_benchmark.py
```

SHA-256 outputs:

- `vstf_hmm_benchmark_summary.csv`: `b0b61997991db87e03c7e0aa88f71334e3506402afb3df0489d393ae85c850ea`
- `vstf_hmm_benchmark_monte_carlo.csv`: `58826553336ddcbd7416ebd1134f44fa1262ca59761f9641ae3a39b2e51fc47f`
- `vstf_hmm_benchmark_sensitivity.csv`: `7d5aab18d6af272dec5c786a0c33de7865e9dd964088160e066afabf564f41e9`
- `vstf_hmm_benchmark_ablations.csv`: `dff05ceb8710db48ea5a035233dc51ac6bbd5b0987d04ef7abef097231cb1aed`
- `vstf_hmm_random_parameter_stress.csv`: `c209bab446592e101b673a1c3feef19bba715238329c081fba0a7fefad9678f8`
- `vstf_hmm_event_sample.csv`: `db88a8c971d834cb304758dfe6cccb57331298db82d052d0b5036c8626782b46`
- `vstf_hmm_benchmark.pdf`: `c2f4fd6f1fa91ba80b2f6b604d755bdbc0a71529093059bd17bda71dcf6999ae`
