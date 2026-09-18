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
Sampling-resolution batches per cell: 2
Many-system benchmark: 30 systems x 5 seeds x 1500 trajectories per seed
Many-system bootstrap resamples: 500
Event-sample rows: 40

Reproduction command:

```bash
python3 vstf_hmm_benchmark.py
```

SHA-256 outputs:

- `vstf_hmm_benchmark_summary.csv`: `aaef61896a7598c680e30997816060d2bbe8d4d18518c38b551be54a7d94eed1`
- `vstf_hmm_benchmark_monte_carlo.csv`: `99fc93b5ba5315c947fbd4c0908f1ac96c86dd48da8b3772e7a53b781d58eb93`
- `vstf_hmm_benchmark_sensitivity.csv`: `632544804bc86e26280ee58e8c3d7ab1eabc844ab52f133173d57773da4d35b6`
- `vstf_hmm_sampling_resolution.csv`: `09feede75006fe9300bfc7ac21499c970926282ef80d8a6ee36ba129c72973ad`
- `vstf_hmm_benchmark_ablations.csv`: `1bf2c156b6d752398ce68f3cf2dda7657eb4475764c8bab738012507c1f3fb00`
- `vstf_hmm_random_parameter_stress.csv`: `dc15749362f5f047bf72130b91e42e66ed421364c245b53bba0c8c080d855d92`
- `vstf_hmm_scenario_stress.csv`: `7e617ba4b0b3971634d18570ce1f9abd0b6adf8b259addbd3b7528e3ee7b8be1`
- `vstf_many_system_benchmark.csv`: `08d0eea2ddae2f18df831ae3e7e28d4e271f4785d943e063bd1d74d546f3fb62`
- `vstf_many_system_discordance.csv`: `37f19767ab165fe565f4d1f649869789c78cc60c46c5bd263c9f862fb73ef130`
- `vstf_hmm_event_sample.csv`: `68b25c926ebecf4dce7844e30690c9317cbc8b3fc8cb54e29380c0eb3e4dabfc`
- `vstf_event_cascade.pdf`: `f48aab39df6cb7a1ff9d57377cee16e2c0aa0c80f2fd544205670d8f2b91dc56`
- `vstf_hmm_benchmark.pdf`: `a6b8370b7956b79a25c441397888169314cbbf25995b7508328674d0f88ac8a2`
- `vstf_many_system_rank_displacement.pdf`: `ec7c21c0b1d31dcc412fc26d90970aa0f0d34ad20381869e5488cd41524cd7a6`
