#!/usr/bin/env python3
"""Hidden-Markov benchmark for the Valid State Transition Framework.

The benchmark is a reproducible methodological stress test. It compares two
synthetic systems with similar raw activity efficiency but different retention,
domain validity, and full-boundary cost. It repeats the experiment over
independent seeds, computes latent-ground-truth error rates, and runs a small
sensitivity sweep plus controlled ablations so that the ranking reversal is not
tied to one realization or one simultaneously changed parameter bundle.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import csv
import hashlib
import math
import platform
import sys
from statistics import mean, pstdev

import numpy as np
import reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


MASTER_SEED = 20260917
N_TRAJ = 10_000
N_BATCHES = 30
T = 72
THETA_CROSS = 0.72
POST_COMMIT = 0.65
POST_VALIDATE = 0.80
VALIDATION_LAG = 2
TAU_RET = 6
POST_RET = 0.55
HYST_LOW = 0.35
SENS_SIGMA_FACTORS = (0.75, 1.0, 1.25)
SENS_TAU_RET = (4, 6, 10)
N_SENS_BATCHES = 3
N_ABLATION_BATCHES = 8
N_EVENT_SAMPLE_ROWS = 40
N_RANDOM_SYSTEM_PAIRS = 8
N_RANDOM_PAIR_BATCHES = 2
N_SCENARIO_BATCHES = 3
N_SAMPLING_BATCHES = 2
SAMPLING_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
N_MANY_SYSTEMS = 30
N_MANY_SEEDS = 5
N_MANY_TRAJ = 1_500
N_BOOTSTRAP_SYSTEMS = 500


@dataclass(frozen=True)
class SystemSpec:
    name: str
    short: str
    p01: float
    p10: float
    sigma: float
    domain_valid_prob: float
    energy_base: float
    energy_per_crossing: float
    energy_per_committed: float
    energy_per_validation: float


SYSTEM_A = SystemSpec(
    name="System A: high activity, fragile retention",
    short="A",
    p01=0.125,
    p10=0.225,
    sigma=0.44,
    domain_valid_prob=0.72,
    energy_base=92.0,
    energy_per_crossing=1.15,
    energy_per_committed=2.30,
    energy_per_validation=3.20,
)

SYSTEM_B = SystemSpec(
    name="System B: moderate activity, retained outcomes",
    short="B",
    p01=0.024,
    p10=0.048,
    sigma=0.27,
    domain_valid_prob=0.88,
    energy_base=88.0,
    energy_per_crossing=0.95,
    energy_per_committed=1.45,
    energy_per_validation=2.10,
)

SYSTEMS = (SYSTEM_A, SYSTEM_B)


def simulate_hidden(spec: SystemSpec, rng: np.random.Generator, n_traj: int = N_TRAJ) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((n_traj, T), dtype=np.int8)
    for t in range(1, T):
        prev = x[:, t - 1]
        u = rng.random(n_traj)
        x[:, t] = np.where(prev == 0, (u < spec.p01).astype(np.int8), (u >= spec.p10).astype(np.int8))
    y = rng.normal(loc=x.astype(float), scale=spec.sigma)
    return x, y


def rescale_transition_probability(prob: float, factor: float) -> float:
    return float(1.0 - (1.0 - prob) ** factor)


def simulate_hidden_scaled(spec: SystemSpec, rng: np.random.Generator, sampling_factor: float, n_traj: int = N_TRAJ) -> tuple[np.ndarray, np.ndarray]:
    n_steps = max(8, int(round(T / sampling_factor)))
    p01 = rescale_transition_probability(spec.p01, sampling_factor)
    p10 = rescale_transition_probability(spec.p10, sampling_factor)
    x = np.zeros((n_traj, n_steps), dtype=np.int8)
    for t in range(1, n_steps):
        prev = x[:, t - 1]
        u = rng.random(n_traj)
        x[:, t] = np.where(prev == 0, (u < p01).astype(np.int8), (u >= p10).astype(np.int8))
    y = rng.normal(loc=x.astype(float), scale=spec.sigma)
    return x, y


def simulate_hidden_hsmm(spec: SystemSpec, rng: np.random.Generator, n_traj: int = N_TRAJ) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((n_traj, T), dtype=np.int8)
    mean_on = max(2.0, 1.0 / max(spec.p10, 1e-6))
    mean_off = max(2.0, 1.0 / max(spec.p01, 1e-6))
    for i in range(n_traj):
        state = 1 if rng.random() < 0.06 else 0
        t = 0
        while t < T:
            mean_dwell = mean_on if state else mean_off
            dwell = int(max(1, rng.gamma(shape=2.4, scale=mean_dwell / 2.4)))
            x[i, t : min(T, t + dwell)] = state
            t += dwell
            state = 1 - state
    y = rng.normal(loc=x.astype(float), scale=spec.sigma)
    return x, y


def filter_posterior(spec: SystemSpec, y: np.ndarray) -> np.ndarray:
    """Stable two-state forward filter with Gaussian emissions."""
    p = np.full(y.shape[0], 0.06, dtype=float)
    post = np.zeros_like(y, dtype=float)
    for t in range(y.shape[1]):
        if t > 0:
            p = (1.0 - p) * spec.p01 + p * (1.0 - spec.p10)
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        obs = y[:, t]
        logit_prior = np.log(p / (1.0 - p))
        log_likelihood_ratio = -0.5 * ((obs - 1.0) / spec.sigma) ** 2 + 0.5 * (obs / spec.sigma) ** 2
        z = np.clip(logit_prior + log_likelihood_ratio, -40.0, 40.0)
        p = 1.0 / (1.0 + np.exp(-z))
        post[:, t] = p
    return post


def add_derived(row: dict[str, float]) -> dict[str, float]:
    row = dict(row)
    row["raw_per_energy"] = row["raw_activity"] / row["energy"]
    row["crossings_per_energy"] = row["crossings"] / row["energy"]
    row["committed_per_energy"] = row["committed"] / row["energy"]
    row["retained_per_energy"] = row["retained"] / row["energy"]
    row["vste"] = row["valid_vst"] / row["energy"]
    row["candidate_yield"] = row["valid_vst"] / max(row["crossings"], 1.0)
    row["commit_yield"] = row["valid_vst"] / max(row["committed"], 1.0)
    row["retention_yield"] = row["retained"] / max(row["committed"], 1.0)
    row["validity_yield"] = row["valid_vst"] / max(row["retained"], 1.0)
    row["downstream_rejection_fraction"] = row["false_success"] / max(row["committed"], 1.0)
    row["downstream_rejection_resolved_fraction"] = row["false_success"] / max(row["false_success"] + row["valid_vst"], 1.0)
    row["true_positive"] = row["valid_vst"] - row["false_acceptance"]
    row["precision_ppv"] = row["true_positive"] / max(row["valid_vst"], 1.0)
    row["sensitivity_tpr"] = row["true_positive"] / max(row["latent_valid"], 1.0)
    row["false_discovery_fraction"] = row["false_acceptance"] / max(row["valid_vst"], 1.0)
    row["far"] = row["false_acceptance"] / max(row["resolved"] - row["latent_valid"], 1.0)
    row["frr"] = row["false_rejection"] / max(row["latent_valid"], 1.0)
    row["pending_fraction"] = row["pending"] / max(row["committed"], 1.0)
    return row


def domain_draws(
    spec: SystemSpec,
    rng: np.random.Generator,
    latent_retained: bool,
    posterior_retained: bool,
    posterior_mean: float,
) -> tuple[bool, bool]:
    if posterior_retained:
        p_obs = spec.domain_valid_prob
    else:
        p_obs = 0.35 * spec.domain_valid_prob
    domain_truth = rng.random() <= spec.domain_valid_prob
    domain_observed = rng.random() <= p_obs
    return domain_truth, domain_observed


def correlated_domain_draws(
    spec: SystemSpec,
    rng: np.random.Generator,
    latent_retained: bool,
    posterior_retained: bool,
    posterior_mean: float,
) -> tuple[bool, bool]:
    latent_term = 1.25 if latent_retained else -1.25
    observed_term = 2.0 * (posterior_mean - 0.55)
    base = math.log(spec.domain_valid_prob / (1.0 - spec.domain_valid_prob))
    p_truth = 1.0 / (1.0 + math.exp(-(base + latent_term)))
    p_obs = 1.0 / (1.0 + math.exp(-(base + observed_term + (0.65 if posterior_retained else -0.65))))
    return rng.random() <= p_truth, rng.random() <= p_obs


def summarize_events(
    spec: SystemSpec,
    rng: np.random.Generator,
    tau_ret: int = TAU_RET,
    validation_lag: int = VALIDATION_LAG,
    dynamics: str = "hmm",
    domain_mode: str = "independent",
    sampling_factor: float = 1.0,
    n_traj: int = N_TRAJ,
) -> dict[str, float]:
    if sampling_factor != 1.0 and dynamics == "hmm":
        x, y = simulate_hidden_scaled(spec, rng, sampling_factor, n_traj=n_traj)
    elif dynamics == "hsmm":
        x, y = simulate_hidden_hsmm(spec, rng, n_traj=n_traj)
    else:
        x, y = simulate_hidden(spec, rng, n_traj=n_traj)
    post = filter_posterior(spec, y)
    n_t = x.shape[1]
    totals = {
        "raw_activity": 0.0,
        "crossings": 0.0,
        "committed": 0.0,
        "retained": 0.0,
        "valid_vst": 0.0,
        "false_success": 0.0,
        "false_acceptance": 0.0,
        "false_rejection": 0.0,
        "latent_valid": 0.0,
        "pending": 0.0,
        "resolved": 0.0,
        "energy": 0.0,
    }

    for i in range(n_traj):
        yi = y[i]
        xi = x[i]
        pi = post[i]
        totals["raw_activity"] += float(np.sum(yi >= THETA_CROSS))

        armed = True
        crossings: list[int] = []
        for t, obs in enumerate(yi):
            if armed and obs >= THETA_CROSS:
                crossings.append(t)
                armed = False
            elif not armed and obs <= HYST_LOW:
                armed = True
        totals["crossings"] += len(crossings)

        committed = 0
        resolved_attempts = 0
        for t in crossings:
            if pi[t] < POST_COMMIT:
                continue
            committed += 1
            totals["committed"] += 1
            v = t + validation_lag
            end = v + tau_ret
            if end >= n_t:
                totals["pending"] += 1
                continue

            totals["resolved"] += 1
            resolved_attempts += 1
            posterior_retained = pi[v] >= POST_VALIDATE and float(np.min(pi[v : end + 1])) >= POST_RET
            latent_retained = xi[v] == 1 and bool(np.all(xi[v : end + 1] == 1))
            posterior_mean = float(np.mean(pi[v : end + 1]))
            if domain_mode == "correlated":
                domain_truth, domain_observed = correlated_domain_draws(spec, rng, latent_retained, posterior_retained, posterior_mean)
            else:
                domain_truth, domain_observed = domain_draws(spec, rng, latent_retained, posterior_retained, posterior_mean)
            ground_truth = latent_retained and domain_truth
            accepted = posterior_retained and domain_observed

            if ground_truth:
                totals["latent_valid"] += 1
            if posterior_retained:
                totals["retained"] += 1
            if accepted:
                totals["valid_vst"] += 1
            else:
                totals["false_success"] += 1
            if accepted and not ground_truth:
                totals["false_acceptance"] += 1
            if (not accepted) and ground_truth:
                totals["false_rejection"] += 1

        totals["energy"] += (
            spec.energy_base
            + spec.energy_per_crossing * len(crossings)
            + spec.energy_per_committed * committed
            + spec.energy_per_validation * resolved_attempts
        )

    return add_derived(totals)


def collect_event_sample(spec: SystemSpec, rng: np.random.Generator, max_rows: int) -> list[dict[str, float | int | str]]:
    x, y = simulate_hidden(spec, rng)
    post = filter_posterior(spec, y)
    rows: list[dict[str, float | int | str]] = []

    for i in range(N_TRAJ):
        yi = y[i]
        xi = x[i]
        pi = post[i]
        armed = True
        crossings: list[int] = []
        for t, obs in enumerate(yi):
            if armed and obs >= THETA_CROSS:
                crossings.append(t)
                armed = False
            elif not armed and obs <= HYST_LOW:
                armed = True

        event_id = 0
        for t in crossings:
            if pi[t] < POST_COMMIT:
                continue
            v = t + VALIDATION_LAG
            end = v + TAU_RET
            event_id += 1
            if end >= T:
                rows.append(
                    {
                        "system": spec.short,
                        "trajectory_id": i,
                        "event_id": event_id,
                        "t_cross": t,
                        "t_validate": v,
                        "t_end": end,
                        "y_cross": yi[t],
                        "posterior_commit": pi[t],
                        "posterior_validate": "",
                        "min_posterior_retention": "",
                        "latent_retained": "",
                        "posterior_retained": "",
                        "domain_truth": "",
                        "domain_observed": "",
                        "reference_valid_z": "",
                        "operational_decision_a": "",
                        "outcome": "pending",
                    }
                )
                continue

            posterior_retained = pi[v] >= POST_VALIDATE and float(np.min(pi[v : end + 1])) >= POST_RET
            latent_retained = xi[v] == 1 and bool(np.all(xi[v : end + 1] == 1))
            domain_truth = rng.random() <= spec.domain_valid_prob
            domain_observed = rng.random() <= spec.domain_valid_prob
            latent_truth = latent_retained and domain_truth
            accepted = posterior_retained and domain_observed
            if accepted and latent_truth:
                outcome = "true_positive"
            elif accepted:
                outcome = "false_acceptance"
            elif latent_truth:
                outcome = "false_rejection"
            else:
                outcome = "true_negative_or_downstream_rejection"
            rows.append(
                {
                    "system": spec.short,
                    "trajectory_id": i,
                    "event_id": event_id,
                    "t_cross": t,
                    "t_validate": v,
                    "t_end": end,
                    "y_cross": yi[t],
                    "posterior_commit": pi[t],
                    "posterior_validate": pi[v],
                    "min_posterior_retention": float(np.min(pi[v : end + 1])),
                    "latent_retained": int(latent_retained),
                    "posterior_retained": int(posterior_retained),
                    "domain_truth": int(domain_truth),
                    "domain_observed": int(domain_observed),
                    "reference_valid_z": int(latent_truth),
                    "operational_decision_a": int(accepted),
                    "outcome": outcome,
                }
            )
            if len(rows) >= max_rows:
                return rows
    return rows


def run_batch(seed: int, tau_ret: int = TAU_RET, sigma_factor: float = 1.0) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, float]] = {}
    for spec in SYSTEMS:
        adjusted = replace(spec, sigma=spec.sigma * sigma_factor)
        out[spec.short] = summarize_events(adjusted, rng, tau_ret=tau_ret)
    return out


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    m = mean(values)
    if len(values) < 2:
        return m, m, m
    se = pstdev(values) / math.sqrt(len(values))
    return m, m - 1.96 * se, m + 1.96 * se


def monte_carlo() -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    batches = [run_batch(MASTER_SEED + 1009 * i) for i in range(N_BATCHES)]
    aggregate: dict[str, dict[str, float]] = {}
    derived = {
        "raw_per_energy",
        "crossings_per_energy",
        "committed_per_energy",
        "retained_per_energy",
        "vste",
        "candidate_yield",
        "commit_yield",
        "retention_yield",
        "validity_yield",
        "downstream_rejection_fraction",
        "downstream_rejection_resolved_fraction",
        "true_positive",
        "precision_ppv",
        "sensitivity_tpr",
        "false_discovery_fraction",
        "far",
        "frr",
        "pending_fraction",
    }
    for short in ("A", "B"):
        totals: dict[str, float] = {}
        for key in batches[0][short]:
            if key not in derived:
                totals[key] = sum(batch[short][key] for batch in batches)
        aggregate[short] = add_derived(totals)

    raw_a = [b["A"]["raw_per_energy"] for b in batches]
    raw_b = [b["B"]["raw_per_energy"] for b in batches]
    vste_a = [b["A"]["vste"] for b in batches]
    vste_b = [b["B"]["vste"] for b in batches]
    reversal = [ra > rb and vb > va for ra, rb, va, vb in zip(raw_a, raw_b, vste_a, vste_b)]
    crossing_reversal = [b["A"]["crossings_per_energy"] > b["B"]["crossings_per_energy"] and b["B"]["vste"] > b["A"]["vste"] for b in batches]
    committed_reversal = [b["A"]["committed_per_energy"] > b["B"]["committed_per_energy"] and b["B"]["vste"] > b["A"]["vste"] for b in batches]
    retained_reversal = [b["A"]["retained_per_energy"] > b["B"]["retained_per_energy"] and b["B"]["vste"] > b["A"]["vste"] for b in batches]
    vste_a_ci = mean_ci(vste_a)
    vste_b_ci = mean_ci(vste_b)
    stats = {
        "n_batches": float(N_BATCHES),
        "n_traj_per_batch": float(N_TRAJ),
        "ranking_reversal_probability": sum(reversal) / len(reversal),
        "crossing_to_vste_reversal_probability": sum(crossing_reversal) / len(crossing_reversal),
        "committed_to_vste_reversal_probability": sum(committed_reversal) / len(committed_reversal),
        "retained_to_vste_reversal_probability": sum(retained_reversal) / len(retained_reversal),
        "mean_vste_ratio_b_over_a": mean(vb / va for va, vb in zip(vste_a, vste_b)),
        "mean_raw_ratio_a_over_b": mean(ra / rb for ra, rb in zip(raw_a, raw_b)),
        "vste_a_mean": vste_a_ci[0],
        "vste_a_ci_low": vste_a_ci[1],
        "vste_a_ci_high": vste_a_ci[2],
        "vste_b_mean": vste_b_ci[0],
        "vste_b_ci_low": vste_b_ci[1],
        "vste_b_ci_high": vste_b_ci[2],
    }
    return aggregate, stats


def ablation_systems(kind: str) -> tuple[SystemSpec, SystemSpec]:
    if kind == "combined":
        return SYSTEM_A, SYSTEM_B
    common = SYSTEM_A
    if kind == "dynamics_only":
        return common, replace(common, p01=SYSTEM_B.p01, p10=SYSTEM_B.p10, name="System B: dynamics-only", short="B")
    if kind == "noise_only":
        return common, replace(common, sigma=SYSTEM_B.sigma, name="System B: noise-only", short="B")
    if kind == "domain_only":
        return common, replace(common, domain_valid_prob=SYSTEM_B.domain_valid_prob, name="System B: domain-only", short="B")
    if kind == "cost_only":
        return common, replace(
            common,
            energy_base=SYSTEM_B.energy_base,
            energy_per_crossing=SYSTEM_B.energy_per_crossing,
            energy_per_committed=SYSTEM_B.energy_per_committed,
            energy_per_validation=SYSTEM_B.energy_per_validation,
            name="System B: cost-only",
            short="B",
        )
    if kind == "domain_cost_only":
        return common, replace(
            common,
            domain_valid_prob=SYSTEM_B.domain_valid_prob,
            energy_base=SYSTEM_B.energy_base,
            energy_per_crossing=SYSTEM_B.energy_per_crossing,
            energy_per_committed=SYSTEM_B.energy_per_committed,
            energy_per_validation=SYSTEM_B.energy_per_validation,
            name="System B: domain+cost-only",
            short="B",
        )
    if kind == "retention_challenge":
        return common, replace(
            common,
            p01=0.105,
            p10=0.255,
            sigma=SYSTEM_A.sigma,
            domain_valid_prob=0.98,
            energy_base=58.0,
            energy_per_crossing=0.52,
            energy_per_committed=0.72,
            energy_per_validation=0.80,
            name="System B: retention-challenge",
            short="B",
        )
    if kind == "equal_domain":
        return SYSTEM_A, replace(SYSTEM_B, domain_valid_prob=SYSTEM_A.domain_valid_prob, name="System B: equal-domain", short="B")
    raise ValueError(kind)


def run_pair(seed: int, spec_a: SystemSpec, spec_b: SystemSpec) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    return {"A": summarize_events(spec_a, rng), "B": summarize_events(spec_b, rng)}


def run_pair_scenario(
    seed: int,
    spec_a: SystemSpec,
    spec_b: SystemSpec,
    dynamics: str,
    domain_mode: str,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    return {
        "A": summarize_events(spec_a, rng, dynamics=dynamics, domain_mode=domain_mode),
        "B": summarize_events(spec_b, rng, dynamics=dynamics, domain_mode=domain_mode),
    }


def ablations() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for kind in ["dynamics_only", "noise_only", "domain_only", "cost_only", "domain_cost_only", "retention_challenge", "equal_domain", "combined"]:
        spec_a, spec_b = ablation_systems(kind)
        reversals = []
        crossing_reversals = []
        retained_reversals = []
        ratios = []
        for i in range(N_ABLATION_BATCHES):
            batch = run_pair(MASTER_SEED + 90_000 + 1009 * i, spec_a, spec_b)
            reversals.append(batch["A"]["raw_per_energy"] > batch["B"]["raw_per_energy"] and batch["B"]["vste"] > batch["A"]["vste"])
            crossing_reversals.append(batch["A"]["crossings_per_energy"] > batch["B"]["crossings_per_energy"] and batch["B"]["vste"] > batch["A"]["vste"])
            retained_reversals.append(batch["A"]["retained_per_energy"] > batch["B"]["retained_per_energy"] and batch["B"]["vste"] > batch["A"]["vste"])
            ratios.append(batch["B"]["vste"] / max(batch["A"]["vste"], 1e-12))
        rows.append(
            {
                "ablation": kind,
                "n_batches": float(N_ABLATION_BATCHES),
                "raw_to_vste_reversal_probability": sum(reversals) / len(reversals),
                "crossing_to_vste_reversal_probability": sum(crossing_reversals) / len(crossing_reversals),
                "retained_to_vste_reversal_probability": sum(retained_reversals) / len(retained_reversals),
                "mean_vste_ratio_b_over_a": mean(ratios),
            }
        )
    return rows


def sensitivity() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for sigma_factor in SENS_SIGMA_FACTORS:
        for tau_ret in SENS_TAU_RET:
            wins = 0
            for i in range(N_SENS_BATCHES):
                batch = run_batch(MASTER_SEED + 50_000 + i * 1009, tau_ret=tau_ret, sigma_factor=sigma_factor)
                wins += int(batch["A"]["raw_per_energy"] > batch["B"]["raw_per_energy"] and batch["B"]["vste"] > batch["A"]["vste"])
            rows.append({"sigma_factor": sigma_factor, "tau_ret": float(tau_ret), "reversal_probability": wins / float(N_SENS_BATCHES)})
    return rows


def sampling_resolution_stress() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    baseline: dict[str, float] | None = None
    for factor in SAMPLING_FACTORS:
        tau_steps = max(1, int(round(TAU_RET / factor)))
        lag_steps = max(1, int(round(VALIDATION_LAG / factor)))
        raw_reversals = []
        crossing_reversals = []
        retained_reversals = []
        ratios = []
        n_vst_a = []
        n_vst_b = []
        ppv_a = []
        ppv_b = []
        tpr_a = []
        tpr_b = []
        for i in range(N_SAMPLING_BATCHES):
            rng = np.random.default_rng(MASTER_SEED + 400_000 + int(factor * 1000) + i * 1009)
            batch = {
                "A": summarize_events(SYSTEM_A, rng, tau_ret=tau_steps, validation_lag=lag_steps, sampling_factor=factor),
                "B": summarize_events(SYSTEM_B, rng, tau_ret=tau_steps, validation_lag=lag_steps, sampling_factor=factor),
            }
            raw_reversals.append(batch["A"]["raw_per_energy"] > batch["B"]["raw_per_energy"] and batch["B"]["vste"] > batch["A"]["vste"])
            crossing_reversals.append(batch["A"]["crossings_per_energy"] > batch["B"]["crossings_per_energy"] and batch["B"]["vste"] > batch["A"]["vste"])
            retained_reversals.append(batch["A"]["retained_per_energy"] > batch["B"]["retained_per_energy"] and batch["B"]["vste"] > batch["A"]["vste"])
            ratios.append(batch["B"]["vste"] / max(batch["A"]["vste"], 1e-12))
            n_vst_a.append(batch["A"]["valid_vst"])
            n_vst_b.append(batch["B"]["valid_vst"])
            ppv_a.append(batch["A"]["precision_ppv"])
            ppv_b.append(batch["B"]["precision_ppv"])
            tpr_a.append(batch["A"]["sensitivity_tpr"])
            tpr_b.append(batch["B"]["sensitivity_tpr"])
        row = {
            "sampling_factor_relative_to_dt0": factor,
            "time_steps": float(max(8, int(round(T / factor)))),
            "tau_ret_steps": float(tau_steps),
            "validation_lag_steps": float(lag_steps),
            "raw_to_vste_reversal_probability": sum(raw_reversals) / len(raw_reversals),
            "crossing_to_vste_reversal_probability": sum(crossing_reversals) / len(crossing_reversals),
            "retained_to_vste_reversal_probability": sum(retained_reversals) / len(retained_reversals),
            "mean_vste_ratio_b_over_a": mean(ratios),
            "mean_valid_vst_a": mean(n_vst_a),
            "mean_valid_vst_b": mean(n_vst_b),
            "relative_valid_vst_a_vs_dt0": 1.0,
            "relative_valid_vst_b_vs_dt0": 1.0,
            "mean_ppv_a": mean(ppv_a),
            "mean_ppv_b": mean(ppv_b),
            "mean_tpr_a": mean(tpr_a),
            "mean_tpr_b": mean(tpr_b),
        }
        if factor == 1.0:
            baseline = row
        rows.append(row)
    if baseline is not None:
        for row in rows:
            row["relative_valid_vst_a_vs_dt0"] = float(row["mean_valid_vst_a"]) / max(float(baseline["mean_valid_vst_a"]), 1e-12)
            row["relative_valid_vst_b_vs_dt0"] = float(row["mean_valid_vst_b"]) / max(float(baseline["mean_valid_vst_b"]), 1e-12)
    return rows


def draw_random_spec(name: str, short: str, rng: np.random.Generator) -> SystemSpec:
    return SystemSpec(
        name=name,
        short=short,
        p01=float(rng.uniform(0.01, 0.15)),
        p10=float(rng.uniform(0.03, 0.25)),
        sigma=float(rng.uniform(0.15, 0.55)),
        domain_valid_prob=float(rng.uniform(0.60, 0.95)),
        energy_base=float(rng.uniform(80.0, 105.0)),
        energy_per_crossing=float(rng.uniform(0.75, 1.40)),
        energy_per_committed=float(rng.uniform(1.10, 2.60)),
        energy_per_validation=float(rng.uniform(1.60, 3.40)),
    )


def random_parameter_stress() -> list[dict[str, float | int]]:
    rng = np.random.default_rng(MASTER_SEED + 123_456)
    rows: list[dict[str, float | int]] = []
    for pair_id in range(N_RANDOM_SYSTEM_PAIRS):
        spec_a = draw_random_spec(f"Random pair {pair_id} A", "A", rng)
        spec_b = draw_random_spec(f"Random pair {pair_id} B", "B", rng)
        raw_discordant = 0
        crossing_discordant = 0
        retained_discordant = 0
        ratios = []
        for batch_id in range(N_RANDOM_PAIR_BATCHES):
            batch = run_pair(MASTER_SEED + 200_000 + pair_id * 10_000 + batch_id * 1009, spec_a, spec_b)
            raw_order = math.copysign(1.0, batch["A"]["raw_per_energy"] - batch["B"]["raw_per_energy"])
            crossing_order = math.copysign(1.0, batch["A"]["crossings_per_energy"] - batch["B"]["crossings_per_energy"])
            retained_order = math.copysign(1.0, batch["A"]["retained_per_energy"] - batch["B"]["retained_per_energy"])
            vste_order = math.copysign(1.0, batch["A"]["vste"] - batch["B"]["vste"])
            raw_discordant += int(raw_order != vste_order)
            crossing_discordant += int(crossing_order != vste_order)
            retained_discordant += int(retained_order != vste_order)
            ratios.append(batch["B"]["vste"] / max(batch["A"]["vste"], 1e-12))
        rows.append(
            {
                "pair_id": pair_id,
                "n_batches": N_RANDOM_PAIR_BATCHES,
                "raw_vste_discordance": raw_discordant / N_RANDOM_PAIR_BATCHES,
                "crossing_vste_discordance": crossing_discordant / N_RANDOM_PAIR_BATCHES,
                "retained_vste_discordance": retained_discordant / N_RANDOM_PAIR_BATCHES,
                "mean_vste_ratio_b_over_a": mean(ratios),
                "a_p01": spec_a.p01,
                "a_p10": spec_a.p10,
                "a_sigma": spec_a.sigma,
                "a_domain_valid_prob": spec_a.domain_valid_prob,
                "b_p01": spec_b.p01,
                "b_p10": spec_b.p10,
                "b_sigma": spec_b.sigma,
                "b_domain_valid_prob": spec_b.domain_valid_prob,
            }
        )
    return rows


def stratified_values(rng: np.random.Generator, n: int, low: float, high: float) -> np.ndarray:
    values = (np.arange(n, dtype=float) + rng.random(n)) / n
    rng.shuffle(values)
    return low + values * (high - low)


def many_system_specs() -> list[SystemSpec]:
    rng = np.random.default_rng(MASTER_SEED + 456_789)
    p01 = stratified_values(rng, N_MANY_SYSTEMS, 0.02, 0.13)
    p10 = stratified_values(rng, N_MANY_SYSTEMS, 0.04, 0.23)
    sigma = stratified_values(rng, N_MANY_SYSTEMS, 0.25, 0.45)
    domain = stratified_values(rng, N_MANY_SYSTEMS, 0.70, 0.90)
    crossing_cost = stratified_values(rng, N_MANY_SYSTEMS, 0.85, 1.25)
    committed_cost = stratified_values(rng, N_MANY_SYSTEMS, 1.25, 2.45)
    validation_cost = stratified_values(rng, N_MANY_SYSTEMS, 1.80, 3.30)
    base_energy = stratified_values(rng, N_MANY_SYSTEMS, 84.0, 100.0)
    specs = []
    for i in range(N_MANY_SYSTEMS):
        specs.append(
            SystemSpec(
                name=f"Many-system {i + 1:02d}",
                short=f"S{i + 1:02d}",
                p01=float(p01[i]),
                p10=float(p10[i]),
                sigma=float(sigma[i]),
                domain_valid_prob=float(domain[i]),
                energy_base=float(base_energy[i]),
                energy_per_crossing=float(crossing_cost[i]),
                energy_per_committed=float(committed_cost[i]),
                energy_per_validation=float(validation_cost[i]),
            )
        )
    return specs


def average_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    raw_keys = [
        "raw_activity",
        "crossings",
        "committed",
        "retained",
        "valid_vst",
        "false_success",
        "false_acceptance",
        "false_rejection",
        "latent_valid",
        "pending",
        "resolved",
        "energy",
    ]
    out = {key: mean([row[key] for row in rows]) for key in raw_keys}
    return add_derived(out)


def rank_desc(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for idx in order[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def pairwise_discordance(rows: list[dict[str, float | str]], baseline_key: str) -> float:
    discordant = 0
    comparable = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            baseline_delta = float(rows[i][baseline_key]) - float(rows[j][baseline_key])
            vste_delta = float(rows[i]["vste"]) - float(rows[j]["vste"])
            if baseline_delta == 0 or vste_delta == 0:
                continue
            comparable += 1
            discordant += int(math.copysign(1.0, baseline_delta) != math.copysign(1.0, vste_delta))
    return discordant / max(comparable, 1)


def bootstrap_discordance(rows: list[dict[str, float | str]], baseline_key: str, rng: np.random.Generator) -> tuple[float, float, float]:
    observed = pairwise_discordance(rows, baseline_key)
    boots = []
    n = len(rows)
    for _ in range(N_BOOTSTRAP_SYSTEMS):
        sample = [rows[int(idx)] for idx in rng.integers(0, n, size=n)]
        boots.append(pairwise_discordance(sample, baseline_key))
    boots.sort()
    lo = boots[int(0.025 * (len(boots) - 1))]
    hi = boots[int(0.975 * (len(boots) - 1))]
    return observed, lo, hi


def many_system_benchmark() -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    specs = many_system_specs()
    rows: list[dict[str, float | str]] = []
    for system_id, spec in enumerate(specs, start=1):
        seed_rows = []
        for seed_id in range(N_MANY_SEEDS):
            rng = np.random.default_rng(MASTER_SEED + 500_000 + system_id * 10_000 + seed_id * 1009)
            seed_rows.append(summarize_events(spec, rng, n_traj=N_MANY_TRAJ))
        avg = average_rows(seed_rows)
        row: dict[str, float | str] = {
            "system_id": system_id,
            "system": spec.short,
            "p01": spec.p01,
            "p10": spec.p10,
            "sigma": spec.sigma,
            "domain_valid_prob": spec.domain_valid_prob,
            "energy_base": spec.energy_base,
            "energy_per_crossing": spec.energy_per_crossing,
            "energy_per_committed": spec.energy_per_committed,
            "energy_per_validation": spec.energy_per_validation,
            "n_seeds": N_MANY_SEEDS,
            "n_traj_per_seed": N_MANY_TRAJ,
        }
        row.update(avg)
        rows.append(row)

    retained_ranks = rank_desc([float(row["retained_per_energy"]) for row in rows])
    vste_ranks = rank_desc([float(row["vste"]) for row in rows])
    for row, r_ret, r_vste in zip(rows, retained_ranks, vste_ranks):
        row["rank_retained_per_energy"] = r_ret
        row["rank_vste"] = r_vste
        row["rank_shift_retained_to_vste"] = r_vste - r_ret

    rng = np.random.default_rng(MASTER_SEED + 654_321)
    discordance_rows = []
    for label, key in [
        ("raw_per_energy", "raw_per_energy"),
        ("crossings_per_energy", "crossings_per_energy"),
        ("committed_per_energy", "committed_per_energy"),
        ("retained_per_energy", "retained_per_energy"),
        ("retention_yield", "retention_yield"),
        ("validity_yield", "validity_yield"),
    ]:
        observed, lo, hi = bootstrap_discordance(rows, key, rng)
        discordance_rows.append(
            {
                "baseline": label,
                "metric_key": key,
                "n_systems": N_MANY_SYSTEMS,
                "n_bootstrap_system_resamples": N_BOOTSTRAP_SYSTEMS,
                "pairwise_discordance_vs_vste": observed,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
            }
        )
    return rows, discordance_rows


def scenario_stress() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    scenarios = [
        ("hmm_independent_domain", "hmm", "independent"),
        ("hmm_correlated_domain", "hmm", "correlated"),
        ("hsmm_independent_domain", "hsmm", "independent"),
        ("hsmm_correlated_domain", "hsmm", "correlated"),
    ]
    for name, dynamics, domain_mode in scenarios:
        raw_reversals = []
        crossing_reversals = []
        retained_reversals = []
        ratios = []
        ppv_a = []
        ppv_b = []
        for i in range(N_SCENARIO_BATCHES):
            batch = run_pair_scenario(MASTER_SEED + 300_000 + i * 1009, SYSTEM_A, SYSTEM_B, dynamics, domain_mode)
            raw_reversals.append(batch["A"]["raw_per_energy"] > batch["B"]["raw_per_energy"] and batch["B"]["vste"] > batch["A"]["vste"])
            crossing_reversals.append(batch["A"]["crossings_per_energy"] > batch["B"]["crossings_per_energy"] and batch["B"]["vste"] > batch["A"]["vste"])
            retained_reversals.append(batch["A"]["retained_per_energy"] > batch["B"]["retained_per_energy"] and batch["B"]["vste"] > batch["A"]["vste"])
            ratios.append(batch["B"]["vste"] / max(batch["A"]["vste"], 1e-12))
            ppv_a.append(batch["A"]["precision_ppv"])
            ppv_b.append(batch["B"]["precision_ppv"])
        rows.append(
            {
                "scenario": name,
                "n_batches": float(N_SCENARIO_BATCHES),
                "raw_to_vste_reversal_probability": sum(raw_reversals) / len(raw_reversals),
                "crossing_to_vste_reversal_probability": sum(crossing_reversals) / len(crossing_reversals),
                "retained_to_vste_reversal_probability": sum(retained_reversals) / len(retained_reversals),
                "mean_vste_ratio_b_over_a": mean(ratios),
                "mean_ppv_a": mean(ppv_a),
                "mean_ppv_b": mean(ppv_b),
            }
        )
    return rows


def write_summary_csv(path: Path, aggregate: dict[str, dict[str, float]]) -> None:
    fields = [
        "system",
        "raw_activity",
        "crossings",
        "committed",
        "retained",
        "valid_vst",
        "false_success",
        "false_acceptance",
        "false_rejection",
        "latent_valid",
        "pending",
        "resolved",
        "energy",
        "raw_per_energy",
        "crossings_per_energy",
        "committed_per_energy",
        "retained_per_energy",
        "vste",
        "candidate_yield",
        "commit_yield",
        "retention_yield",
        "validity_yield",
        "downstream_rejection_fraction",
        "downstream_rejection_resolved_fraction",
        "true_positive",
        "precision_ppv",
        "sensitivity_tpr",
        "false_discovery_fraction",
        "far",
        "frr",
        "pending_fraction",
    ]
    names = {"A": SYSTEM_A.name, "B": SYSTEM_B.name}
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for short, row in aggregate.items():
            writer.writerow({"system": names[short], **{k: row[k] for k in fields if k != "system"}})


def write_metric_csv(path: Path, stats: dict[str, float]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in stats.items():
            writer.writerow({"metric": key, "value": value})


def write_sensitivity_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sigma_factor", "tau_ret", "reversal_probability"])
        writer.writeheader()
        writer.writerows(rows)


def write_sampling_resolution_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sampling_factor_relative_to_dt0",
                "time_steps",
                "tau_ret_steps",
                "validation_lag_steps",
                "raw_to_vste_reversal_probability",
                "crossing_to_vste_reversal_probability",
                "retained_to_vste_reversal_probability",
                "mean_vste_ratio_b_over_a",
                "mean_valid_vst_a",
                "mean_valid_vst_b",
                "relative_valid_vst_a_vs_dt0",
                "relative_valid_vst_b_vs_dt0",
                "mean_ppv_a",
                "mean_ppv_b",
                "mean_tpr_a",
                "mean_tpr_b",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_ablation_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ablation",
                "n_batches",
                "raw_to_vste_reversal_probability",
                "crossing_to_vste_reversal_probability",
                "retained_to_vste_reversal_probability",
                "mean_vste_ratio_b_over_a",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_random_parameter_stress_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "n_batches",
                "raw_vste_discordance",
                "crossing_vste_discordance",
                "retained_vste_discordance",
                "mean_vste_ratio_b_over_a",
                "a_p01",
                "a_p10",
                "a_sigma",
                "a_domain_valid_prob",
                "b_p01",
                "b_p10",
                "b_sigma",
                "b_domain_valid_prob",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_scenario_stress_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "n_batches",
                "raw_to_vste_reversal_probability",
                "crossing_to_vste_reversal_probability",
                "retained_to_vste_reversal_probability",
                "mean_vste_ratio_b_over_a",
                "mean_ppv_a",
                "mean_ppv_b",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_many_system_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fields = [
        "system_id",
        "system",
        "p01",
        "p10",
        "sigma",
        "domain_valid_prob",
        "energy_base",
        "energy_per_crossing",
        "energy_per_committed",
        "energy_per_validation",
        "n_seeds",
        "n_traj_per_seed",
        "raw_activity",
        "crossings",
        "committed",
        "retained",
        "valid_vst",
        "false_success",
        "false_acceptance",
        "false_rejection",
        "latent_valid",
        "pending",
        "resolved",
        "energy",
        "raw_per_energy",
        "crossings_per_energy",
        "committed_per_energy",
        "retained_per_energy",
        "vste",
        "candidate_yield",
        "commit_yield",
        "retention_yield",
        "validity_yield",
        "downstream_rejection_fraction",
        "downstream_rejection_resolved_fraction",
        "true_positive",
        "precision_ppv",
        "sensitivity_tpr",
        "false_discovery_fraction",
        "far",
        "frr",
        "pending_fraction",
        "rank_retained_per_energy",
        "rank_vste",
        "rank_shift_retained_to_vste",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_many_system_discordance_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fields = [
        "baseline",
        "metric_key",
        "n_systems",
        "n_bootstrap_system_resamples",
        "pairwise_discordance_vs_vste",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_event_sample_csv(path: Path) -> None:
    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(MASTER_SEED + 777)
    for spec in SYSTEMS:
        rows.extend(collect_event_sample(spec, rng, N_EVENT_SAMPLE_ROWS // 2))
    fields = [
        "system",
        "trajectory_id",
        "event_id",
        "t_cross",
        "t_validate",
        "t_end",
        "y_cross",
        "posterior_commit",
        "posterior_validate",
        "min_posterior_retention",
        "latent_retained",
        "posterior_retained",
        "domain_truth",
        "domain_observed",
        "reference_valid_z",
        "operational_decision_a",
        "outcome",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_reproducibility_manifest(path: Path, output_files: list[Path]) -> None:
    lines = [
        "# VSTF HMM Benchmark Reproducibility Manifest",
        "",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
        f"NumPy: {np.__version__}",
        f"ReportLab: {reportlab.Version}",
        f"Master seed: {MASTER_SEED}",
        f"Trajectories per batch: {N_TRAJ}",
        f"Monte Carlo batches: {N_BATCHES}",
        f"Sensitivity batches per cell: {N_SENS_BATCHES}",
        f"Ablation batches per cell: {N_ABLATION_BATCHES}",
        f"Sampling-resolution batches per cell: {N_SAMPLING_BATCHES}",
        f"Many-system benchmark: {N_MANY_SYSTEMS} systems x {N_MANY_SEEDS} seeds x {N_MANY_TRAJ} trajectories per seed",
        f"Many-system bootstrap resamples: {N_BOOTSTRAP_SYSTEMS}",
        f"Event-sample rows: {N_EVENT_SAMPLE_ROWS}",
        "",
        "Reproduction command:",
        "",
        "```bash",
        "python3 vstf_hmm_benchmark.py",
        "```",
        "",
        "SHA-256 outputs:",
        "",
    ]
    for file_path in output_files:
        lines.append(f"- `{file_path.name}`: `{sha256(file_path)}`")
    path.write_text("\n".join(lines) + "\n")


def write_event_cascade_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 15)
    c.drawString(42, height - 48, "VSTF Event-Cascade and Claim-Governance Layer")
    c.setFont("Helvetica", 8)
    c.drawString(42, height - 64, "Operational candidates and reference events are matched before error rates are claimed.")

    steps = [
        ("Latent trajectory", "X_t and reference standard"),
        ("Reference events", "E_ref = {e*_r}"),
        ("Observed trajectory", "Y_t from measurement process"),
        ("Candidate events", "E_P = {ehat_k}"),
        ("Event matching", "locked one-to-one rule m"),
        ("Adjudication", "Q_dist,ij * L_k * R_k * V_dom,k"),
        ("Error accounting", "TP / FP / FN after matching"),
        ("Claim gate", "N_VST^op and complete denominator"),
    ]
    x0 = 46
    y0 = height - 132
    box_w = 120
    box_h = 44
    gap_x = 22
    gap_y = 88
    colors_fill = [
        "#dbeafe",
        "#ccfbf1",
        "#fef3c7",
        "#fef3c7",
        "#fef3c7",
        "#fef3c7",
        "#dcfce7",
        "#ede9fe",
    ]
    positions = []
    for idx, (title, body) in enumerate(steps):
        row = idx // 4
        col = idx % 4
        x = x0 + col * (box_w + gap_x)
        y = y0 - row * gap_y
        positions.append((x, y))
        c.setFillColor(colors.HexColor(colors_fill[idx]))
        c.roundRect(x, y, box_w, box_h, 6, fill=1, stroke=0)
        c.setStrokeColor(colors.HexColor("#111827"))
        c.roundRect(x, y, box_w, box_h, 6, fill=0, stroke=1)
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawCentredString(x + box_w / 2, y + 27, title)
        c.setFont("Helvetica", 6.6)
        c.drawCentredString(x + box_w / 2, y + 13, body)
    c.setStrokeColor(colors.HexColor("#374151"))
    for idx in range(len(steps) - 1):
        x, y = positions[idx]
        nx, ny = positions[idx + 1]
        if idx == 3:
            c.line(x + box_w / 2, y - 7, nx + box_w / 2, ny + box_h + 7)
        else:
            c.line(x + box_w + 3, y + box_h / 2, nx - 3, ny + box_h / 2)

    c.setFont("Helvetica-Bold", 10)
    c.drawString(52, 328, "Rejected or unresolved observations remain auditable")
    c.setFont("Helvetica", 7.4)
    rejected = [
        "No Q_dist: proxy signal only; do not count as accepted transition.",
        "No localization: candidate event remains unresolved or rejected.",
        "No retention: acute response, not stabilized outcome.",
        "No V_dom: generic state movement, not domain-valid success.",
        "No complete denominator: may be state-valid but cannot support VSTE/cost-yield claims.",
    ]
    for i, line in enumerate(rejected):
        c.drawString(66, 308 - i * 14, f"- {line}")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(52, 214, "Locked numerator, matching, and denominator")
    c.setFont("Helvetica", 8)
    c.drawString(66, 194, "A_k is evaluated on detected candidates; reference validity belongs to matched reference events.")
    c.drawString(66, 178, "Unmatched accepted candidates are false acceptances; unmatched valid references are false rejections.")
    c.drawString(66, 162, "Cost-normalized claims additionally require Q_B^acct and a declared boundary B.")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(52, 122, "Reviewer-facing interpretation")
    c.setFont("Helvetica", 8)
    c.drawString(66, 102, "VSTF is not a new transition detector. It is a pre-specified adjudication and claim-governance layer")
    c.drawString(66, 88, "placed after or alongside transition detection, before a detected event enters an outcome numerator.")
    c.setFont("Helvetica", 7)
    c.drawString(42, 28, "Generated by vstf_hmm_benchmark.py; synthetic methodology diagram, not empirical validation.")
    c.showPage()
    c.save()


def draw_bar(c: canvas.Canvas, x: float, y: float, width: float, height: float, frac: float, color) -> None:
    c.setFillColor(colors.whitesmoke)
    c.rect(x, y, width, height, fill=1, stroke=0)
    c.setFillColor(color)
    c.rect(x, y, width * max(0.0, min(1.0, frac)), height, fill=1, stroke=0)
    c.setStrokeColor(colors.black)
    c.rect(x, y, width, height, fill=0, stroke=1)


def write_pdf(path: Path, aggregate: dict[str, dict[str, float]], stats: dict[str, float], sens: list[dict[str, float]]) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    _, height = letter
    c.setFont("Helvetica-Bold", 15)
    c.drawString(42, height - 46, "VSTF HMM Benchmark: retained outcomes change rankings")
    c.setFont("Helvetica", 8)
    c.drawString(42, height - 61, f"{N_BATCHES} batches x {N_TRAJ:,} trajectories per system; master seed={MASTER_SEED}")

    metrics = ["raw_activity", "crossings", "committed", "retained", "valid_vst"]
    labels = ["Raw", "Crossing", "Committed", "Retained", "Valid VST"]
    palette = [colors.HexColor("#4f83c4"), colors.HexColor("#43a49a"), colors.HexColor("#e0a23b"), colors.HexColor("#c45b65"), colors.HexColor("#6d5cae")]
    max_raw = max(s["raw_activity"] for s in aggregate.values())
    y0 = height - 110
    c.setFont("Helvetica-Bold", 10)
    c.drawString(42, y0 + 24, "A. Nested attrition")
    for idx, short in enumerate(("A", "B")):
        summary = aggregate[short]
        y = y0 - idx * 124
        c.setFont("Helvetica-Bold", 8)
        c.drawString(42, y + 8, SYSTEM_A.name if short == "A" else SYSTEM_B.name)
        for j, (m, lab) in enumerate(zip(metrics, labels)):
            yy = y - 14 - j * 17
            draw_bar(c, 128, yy, 230, 9, summary[m] / max_raw, palette[j])
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 7.5)
            c.drawString(42, yy + 1, lab)
            c.drawRightString(410, yy + 1, f"{summary[m]:,.0f}")

    y = 292
    c.setFont("Helvetica-Bold", 10)
    c.drawString(42, y, "B. Efficiency and latent-ground-truth errors")
    max_raw_eff = max(s["raw_per_energy"] for s in aggregate.values())
    max_vste = max(s["vste"] for s in aggregate.values())
    for idx, short in enumerate(("A", "B")):
        summary = aggregate[short]
        yy = y - 36 - idx * 62
        c.setFont("Helvetica-Bold", 8)
        c.drawString(42, yy + 27, "System A" if short == "A" else "System B")
        c.setFont("Helvetica", 7.5)
        c.drawString(58, yy + 10, "Raw / energy")
        draw_bar(c, 145, yy + 8, 155, 9, summary["raw_per_energy"] / max_raw_eff, colors.HexColor("#4f83c4"))
        c.drawRightString(350, yy + 9, f"{summary['raw_per_energy']:.4f}")
        c.drawString(58, yy - 8, "Valid VST / energy")
        draw_bar(c, 145, yy - 10, 155, 9, summary["vste"] / max_vste, colors.HexColor("#6d5cae"))
        c.drawRightString(350, yy - 9, f"{summary['vste']:.4f}")
        c.drawString(375, yy + 4, f"PPV={summary['precision_ppv']:.1%}, TPR={summary['sensitivity_tpr']:.1%}")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(42, 146, "C. Multi-seed robustness")
    c.setFont("Helvetica", 8)
    lines = [
        f"Raw-sample to VSTE reversal: {stats['ranking_reversal_probability']:.1%}",
        f"Crossing-event to VSTE reversal: {stats['crossing_to_vste_reversal_probability']:.1%}",
        f"Mean B/A VSTE ratio: {stats['mean_vste_ratio_b_over_a']:.2f}",
        f"VSTE CIs are Monte Carlo intervals.",
    ]
    for i, line in enumerate(lines):
        c.drawString(58, 128 - i * 12, line)

    c.setFont("Helvetica-Bold", 10)
    c.drawString(326, 146, "D. Sensitivity")
    c.setFont("Helvetica", 7.2)
    c.drawString(326, 130, "sigma factor / tau -> reversal probability")
    for i, row in enumerate(sens[:9]):
        c.drawString(326, 116 - i * 10, f"{row['sigma_factor']:.2f} / {int(row['tau_ret'])}: {row['reversal_probability']:.2f}")

    c.setFont("Helvetica", 7.5)
    c.drawString(42, 24, "Synthetic benchmark only: not physical, clinical, or biological validation.")
    c.showPage()
    c.save()


def write_many_system_rank_pdf(path: Path, rows: list[dict[str, float | str]], discordance_rows: list[dict[str, float | str]]) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 15)
    c.drawString(42, height - 48, "Many-System VSTF Rank Displacement")
    c.setFont("Helvetica", 8)
    c.drawString(42, height - 64, "Synthetic systems ranked by retained-event efficiency versus VSTE.")

    plot_x = 80
    plot_y = 150
    plot_w = 380
    plot_h = 380
    n = max(1, len(rows))
    c.setStrokeColor(colors.HexColor("#111827"))
    c.rect(plot_x, plot_y, plot_w, plot_h, fill=0, stroke=1)
    c.setStrokeColor(colors.HexColor("#9ca3af"))
    c.setDash(3, 3)
    c.line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y)
    c.setDash()
    c.setFont("Helvetica", 7)
    for tick in [1, 10, 20, 30]:
        if tick > n:
            continue
        x = plot_x + (tick - 1) / max(n - 1, 1) * plot_w
        y = plot_y + plot_h - (tick - 1) / max(n - 1, 1) * plot_h
        c.setStrokeColor(colors.HexColor("#e5e7eb"))
        c.line(x, plot_y, x, plot_y + plot_h)
        c.line(plot_x, y, plot_x + plot_w, y)
        c.setFillColor(colors.black)
        c.drawCentredString(x, plot_y - 14, str(tick))
        c.drawRightString(plot_x - 8, y - 3, str(tick))

    c.setFillColor(colors.HexColor("#2563eb"))
    for row in rows:
        rx = float(row["rank_retained_per_energy"])
        ry = float(row["rank_vste"])
        x = plot_x + (rx - 1) / max(n - 1, 1) * plot_w
        y = plot_y + plot_h - (ry - 1) / max(n - 1, 1) * plot_h
        c.circle(x, y, 3.2, fill=1, stroke=0)

    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(colors.black)
    c.drawCentredString(plot_x + plot_w / 2, plot_y - 36, "Rank by retained-event efficiency")
    c.saveState()
    c.translate(plot_x - 45, plot_y + plot_h / 2)
    c.rotate(90)
    c.drawCentredString(0, 0, "Rank by VSTE")
    c.restoreState()

    c.setFont("Helvetica-Bold", 10)
    c.drawString(485, height - 132, "Pairwise discordance vs VSTE")
    c.setFont("Helvetica", 7.2)
    for i, row in enumerate(discordance_rows):
        y = height - 152 - i * 18
        c.drawString(
            485,
            y,
            f"{row['baseline']}: {float(row['pairwise_discordance_vs_vste']):.2f} "
            f"[{float(row['bootstrap_ci_low']):.2f}, {float(row['bootstrap_ci_high']):.2f}]",
        )
    c.setFont("Helvetica", 7)
    c.drawString(42, 28, f"{N_MANY_SYSTEMS} systems x {N_MANY_SEEDS} seeds x {N_MANY_TRAJ} trajectories; bootstrap resamples systems.")
    c.showPage()
    c.save()


def main() -> None:
    out_dir = Path("/Users/mpetr/Desktop")
    aggregate, stats = monte_carlo()
    sens = sensitivity()
    abl = ablations()
    random_stress = random_parameter_stress()
    scenarios = scenario_stress()
    sampling_rows = sampling_resolution_stress()
    many_rows, many_discordance = many_system_benchmark()
    write_summary_csv(out_dir / "vstf_hmm_benchmark_summary.csv", aggregate)
    write_metric_csv(out_dir / "vstf_hmm_benchmark_monte_carlo.csv", stats)
    write_sensitivity_csv(out_dir / "vstf_hmm_benchmark_sensitivity.csv", sens)
    write_sampling_resolution_csv(out_dir / "vstf_hmm_sampling_resolution.csv", sampling_rows)
    write_ablation_csv(out_dir / "vstf_hmm_benchmark_ablations.csv", abl)
    write_random_parameter_stress_csv(out_dir / "vstf_hmm_random_parameter_stress.csv", random_stress)
    write_scenario_stress_csv(out_dir / "vstf_hmm_scenario_stress.csv", scenarios)
    write_many_system_csv(out_dir / "vstf_many_system_benchmark.csv", many_rows)
    write_many_system_discordance_csv(out_dir / "vstf_many_system_discordance.csv", many_discordance)
    output_files = [
        out_dir / "vstf_hmm_benchmark_summary.csv",
        out_dir / "vstf_hmm_benchmark_monte_carlo.csv",
        out_dir / "vstf_hmm_benchmark_sensitivity.csv",
        out_dir / "vstf_hmm_sampling_resolution.csv",
        out_dir / "vstf_hmm_benchmark_ablations.csv",
        out_dir / "vstf_hmm_random_parameter_stress.csv",
        out_dir / "vstf_hmm_scenario_stress.csv",
        out_dir / "vstf_many_system_benchmark.csv",
        out_dir / "vstf_many_system_discordance.csv",
        out_dir / "vstf_hmm_event_sample.csv",
        out_dir / "vstf_event_cascade.pdf",
        out_dir / "vstf_hmm_benchmark.pdf",
        out_dir / "vstf_many_system_rank_displacement.pdf",
    ]
    write_event_sample_csv(output_files[9])
    write_event_cascade_pdf(output_files[10])
    write_pdf(output_files[11], aggregate, stats, sens)
    write_many_system_rank_pdf(output_files[12], many_rows, many_discordance)
    write_reproducibility_manifest(out_dir / "REPRODUCIBILITY.md", output_files)

    print("Monte Carlo benchmark")
    for short in ("A", "B"):
        row = aggregate[short]
        print("System A" if short == "A" else "System B")
        for key in [
            "raw_activity",
            "crossings",
            "committed",
            "retained",
            "valid_vst",
            "false_success",
            "false_acceptance",
            "false_rejection",
            "true_positive",
            "latent_valid",
            "pending",
            "energy",
            "raw_per_energy",
            "crossings_per_energy",
            "committed_per_energy",
            "retained_per_energy",
            "vste",
            "candidate_yield",
            "precision_ppv",
            "sensitivity_tpr",
            "false_discovery_fraction",
        ]:
            print(f"  {key}: {row[key]:.6g}")
    for key, value in stats.items():
        print(f"{key}: {value:.6g}")
    for row in abl:
        print(
            f"ablation {row['ablation']}: raw_to_vste={row['raw_to_vste_reversal_probability']:.3g}, "
            f"crossing_to_vste={row['crossing_to_vste_reversal_probability']:.3g}, "
            f"retained_to_vste={row['retained_to_vste_reversal_probability']:.3g}, "
            f"ratio={row['mean_vste_ratio_b_over_a']:.3g}"
        )
    print(
        "random parameter stress: "
        f"{len(random_stress)} pairs x {N_RANDOM_PAIR_BATCHES} batches, "
        f"mean raw/VSTE discordance={mean(float(r['raw_vste_discordance']) for r in random_stress):.3g}"
    )
    for row in scenarios:
        print(
            f"scenario {row['scenario']}: raw_to_vste={row['raw_to_vste_reversal_probability']:.3g}, "
            f"crossing_to_vste={row['crossing_to_vste_reversal_probability']:.3g}, "
            f"retained_to_vste={row['retained_to_vste_reversal_probability']:.3g}, "
            f"ratio={row['mean_vste_ratio_b_over_a']:.3g}"
        )
    for row in many_discordance:
        print(
            f"many-system {row['baseline']}: discordance={row['pairwise_discordance_vs_vste']:.3g} "
            f"[{row['bootstrap_ci_low']:.3g}, {row['bootstrap_ci_high']:.3g}]"
        )
    for row in sampling_rows:
        print(
            f"sampling factor {row['sampling_factor_relative_to_dt0']}: "
            f"raw_to_vste={row['raw_to_vste_reversal_probability']:.3g}, "
            f"rel_valid_A={row['relative_valid_vst_a_vs_dt0']:.3g}, "
            f"rel_valid_B={row['relative_valid_vst_b_vs_dt0']:.3g}"
        )


if __name__ == "__main__":
    main()
