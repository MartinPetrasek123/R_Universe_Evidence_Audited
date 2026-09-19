# VSTF HMM Benchmark Reproducibility Manifest

Python: 3.12.14
Platform: macOS-26.6.2-arm64-arm-64bit
NumPy: 2.3.5
ReportLab: 4.4.9
Master seed: 20260917
Trajectories per batch: 10000
Monte Carlo batches: 30
Sensitivity batches per cell: 30
Ablation batches per cell: 8
Sampling-resolution batches per cell: 2
Many-system benchmark: 30 systems x 5 seeds x 1500 trajectories per seed
Many-system bootstrap resamples: 500
Event-sample rows: 40

Reproduction command:

```bash
python3 vstf_hmm_benchmark.py
```

SHA-256 outputs:

- `vstf_hmm_benchmark_summary.csv`: `cb957ebf8da90ea14c1c0c842caedb53049946d443b5d48c8a7b8662594b7e83`
- `vstf_hmm_benchmark_monte_carlo.csv`: `99fc93b5ba5315c947fbd4c0908f1ac96c86dd48da8b3772e7a53b781d58eb93`
- `vstf_hmm_benchmark_batch_level.csv`: `241960a39308e5e0e569a30f16a586d98da33cdc9f163ea33e90b8dcff11e497`
- `vstf_hmm_benchmark_sensitivity.csv`: `76ad58b9718db651602662aaeb0d0ea23bfa1812507e6443e60419fd03d0d223`
- `vstf_hmm_sampling_resolution.csv`: `09feede75006fe9300bfc7ac21499c970926282ef80d8a6ee36ba129c72973ad`
- `vstf_hmm_benchmark_ablations.csv`: `1bf2c156b6d752398ce68f3cf2dda7657eb4475764c8bab738012507c1f3fb00`
- `vstf_hmm_random_parameter_stress.csv`: `dc15749362f5f047bf72130b91e42e66ed421364c245b53bba0c8c080d855d92`
- `vstf_hmm_scenario_stress.csv`: `7e617ba4b0b3971634d18570ce1f9abd0b6adf8b259addbd3b7528e3ee7b8be1`
- `vstf_model_misspecification_stress.csv`: `38626e0b6adcce412d2bb4fafd708a605bb187a09834da751012146167f5ba07`
- `vstf_many_system_benchmark.csv`: `08d0eea2ddae2f18df831ae3e7e28d4e271f4785d943e063bd1d74d546f3fb62`
- `vstf_many_system_discordance.csv`: `68b25c926ebecf4dce7844e30690c9317cbc8b3fc8cb54e29380c0eb3e4dabfc`
- `vstf_hmm_event_sample.csv`: `72f521de724e219550becef9342355bcb4c3cb93cca7ff6682a98e3354f717cb`
- `vstf_event_cascade.pdf`: `2128793c72ffd8561bc4b07cd4f6731c53ea04cd6543fcef53ce6a46112e258f`
- `vstf_hmm_benchmark.pdf`: `e0640c18e787b00054683cbe195c783b8ad3fc95b09804a2ab07f016dbbc1651`
- `vstf_many_system_rank_displacement.pdf`: `e291ef845a7be0856f5b8f709ac4bed8b9d211a5d2e4fc48426319e409f4bd52`
