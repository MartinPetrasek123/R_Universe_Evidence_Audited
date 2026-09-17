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
import math
from statistics import mean, pstdev

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


MASTER_SEED = 20260917
N_TRAJ = 10_000
N_BATCHES = 100
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
N_SENS_BATCHES = 5
N_ABLATION_BATCHES = 20


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
    energy_per_valid: float


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
    energy_per_valid=3.20,
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
    energy_per_valid=2.10,
)

SYSTEMS = (SYSTEM_A, SYSTEM_B)


def simulate_hidden(spec: SystemSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((N_TRAJ, T), dtype=np.int8)
    for t in range(1, T):
        prev = x[:, t - 1]
        u = rng.random(N_TRAJ)
        x[:, t] = np.where(prev == 0, (u < spec.p01).astype(np.int8), (u >= spec.p10).astype(np.int8))
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


def summarize_events(spec: SystemSpec, rng: np.random.Generator, tau_ret: int = TAU_RET) -> dict[str, float]:
    x, y = simulate_hidden(spec, rng)
    post = filter_posterior(spec, y)
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

    for i in range(N_TRAJ):
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
        valid = 0
        for t in crossings:
            if pi[t] < POST_COMMIT:
                continue
            committed += 1
            totals["committed"] += 1
            v = t + VALIDATION_LAG
            end = v + tau_ret
            if end >= T:
                totals["pending"] += 1
                continue

            totals["resolved"] += 1
            posterior_retained = pi[v] >= POST_VALIDATE and float(np.min(pi[v : end + 1])) >= POST_RET
            latent_retained = xi[v] == 1 and bool(np.all(xi[v : end + 1] == 1))
            domain_truth = rng.random() <= spec.domain_valid_prob
            domain_observed = rng.random() <= spec.domain_valid_prob
            ground_truth = latent_retained and domain_truth
            accepted = posterior_retained and domain_observed

            if ground_truth:
                totals["latent_valid"] += 1
            if posterior_retained:
                totals["retained"] += 1
            if accepted:
                valid += 1
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
            + spec.energy_per_valid * valid
        )

    return add_derived(totals)


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
            energy_per_valid=SYSTEM_B.energy_per_valid,
            name="System B: cost-only",
            short="B",
        )
    if kind == "equal_domain":
        return SYSTEM_A, replace(SYSTEM_B, domain_valid_prob=SYSTEM_A.domain_valid_prob, name="System B: equal-domain", short="B")
    raise ValueError(kind)


def run_pair(seed: int, spec_a: SystemSpec, spec_b: SystemSpec) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    return {"A": summarize_events(spec_a, rng), "B": summarize_events(spec_b, rng)}


def ablations() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for kind in ["dynamics_only", "noise_only", "domain_only", "cost_only", "equal_domain", "combined"]:
        spec_a, spec_b = ablation_systems(kind)
        reversals = []
        crossing_reversals = []
        ratios = []
        for i in range(N_ABLATION_BATCHES):
            batch = run_pair(MASTER_SEED + 90_000 + 1009 * i, spec_a, spec_b)
            reversals.append(batch["A"]["raw_per_energy"] > batch["B"]["raw_per_energy"] and batch["B"]["vste"] > batch["A"]["vste"])
            crossing_reversals.append(batch["A"]["crossings_per_energy"] > batch["B"]["crossings_per_energy"] and batch["B"]["vste"] > batch["A"]["vste"])
            ratios.append(batch["B"]["vste"] / max(batch["A"]["vste"], 1e-12))
        rows.append(
            {
                "ablation": kind,
                "n_batches": float(N_ABLATION_BATCHES),
                "raw_to_vste_reversal_probability": sum(reversals) / len(reversals),
                "crossing_to_vste_reversal_probability": sum(crossing_reversals) / len(crossing_reversals),
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


def write_ablation_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "ablation",
                "n_batches",
                "raw_to_vste_reversal_probability",
                "crossing_to_vste_reversal_probability",
                "mean_vste_ratio_b_over_a",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


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


def main() -> None:
    out_dir = Path("/Users/mpetr/Desktop")
    aggregate, stats = monte_carlo()
    sens = sensitivity()
    abl = ablations()
    write_summary_csv(out_dir / "vstf_hmm_benchmark_summary.csv", aggregate)
    write_metric_csv(out_dir / "vstf_hmm_benchmark_monte_carlo.csv", stats)
    write_sensitivity_csv(out_dir / "vstf_hmm_benchmark_sensitivity.csv", sens)
    write_ablation_csv(out_dir / "vstf_hmm_benchmark_ablations.csv", abl)
    write_pdf(out_dir / "vstf_hmm_benchmark.pdf", aggregate, stats, sens)

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
            f"ratio={row['mean_vste_ratio_b_over_a']:.3g}"
        )


if __name__ == "__main__":
    main()
