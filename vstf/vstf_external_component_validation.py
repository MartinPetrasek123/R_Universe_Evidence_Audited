#!/usr/bin/env python3
"""Additional public-data component checks for the VSTF manuscript.

This script regenerates the external component-check table from public raw
datasets. It is intentionally limited to component-level evidence: two static
acceptance/pending checks, one temporal retention/cost check, and one
intervention-contrast check. It is not an end-to-end validation dataset for the
whole framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import math
import urllib.request

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
MASTER_SEED = 20260919
N_FOLDS = 5
P_ACCEPT_HIGH = 0.98
P_ACCEPT_LOW = 0.02
SHRINKAGE = 0.10
COV_EPS = 1e-6

OUT_SUMMARY = OUT_DIR / "vstf_external_component_validation_summary.csv"
OUT_FOLDS = OUT_DIR / "vstf_external_component_validation_folds.csv"
OUT_APPLIANCES = OUT_DIR / "vstf_external_appliances_events.csv"
OUT_STAR = OUT_DIR / "vstf_external_star_contrast.csv"


@dataclass(frozen=True)
class StaticDatasetSpec:
    key: str
    name: str
    url: str
    filename: str
    loader: str


STATIC_DATASETS = [
    StaticDatasetSpec(
        key="banknote",
        name="UCI Banknote Authentication",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
        filename="data_banknote_authentication.txt",
        loader="banknote",
    ),
    StaticDatasetSpec(
        key="spambase",
        name="UCI Spambase",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
        filename="spambase.data",
        loader="spambase",
    ),
]

APPLIANCES_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
APPLIANCES_CSV = OUT_DIR / "energydata_complete.csv"
STAR_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/AER/STAR.csv"
STAR_CSV = OUT_DIR / "STAR.csv"


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.write_bytes(urllib.request.urlopen(url, timeout=120).read())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_file(url: str, filename: str) -> Path:
    path = OUT_DIR / filename
    download(url, path)
    return path


def load_static_dataset(spec: StaticDatasetSpec) -> tuple[np.ndarray, np.ndarray, str]:
    path = ensure_file(spec.url, spec.filename)
    if spec.loader == "banknote":
        df = pd.read_csv(path, header=None)
        y = df.iloc[:, -1].to_numpy(dtype=float)
        x = df.iloc[:, :-1].to_numpy(dtype=float)
    elif spec.loader == "spambase":
        df = pd.read_csv(path, header=None)
        y = df.iloc[:, -1].to_numpy(dtype=float)
        x = df.iloc[:, :-1].to_numpy(dtype=float)
    else:
        raise ValueError(f"unknown static loader: {spec.loader}")
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[finite], y[finite], sha256(path)


def stratified_folds(y: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    fold_parts = [[] for _ in range(N_FOLDS)]
    for cls in [0.0, 1.0]:
        idx = np.flatnonzero(y == cls)
        shuffled = rng.permutation(idx)
        for fold_id, part in enumerate(np.array_split(shuffled, N_FOLDS)):
            fold_parts[fold_id].extend(part.tolist())
    return [np.array(sorted(part), dtype=int) for part in fold_parts]


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)
    sigma[sigma == 0.0] = 1.0
    return (train_x - mu) / sigma, (test_x - mu) / sigma


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_reference_model(train_x: np.ndarray, train_y: np.ndarray) -> dict[str, np.ndarray | float]:
    class0 = train_x[train_y == 0.0]
    class1 = train_x[train_y == 1.0]
    mean0 = class0.mean(axis=0)
    mean1 = class1.mean(axis=0)
    centered = np.vstack([class0 - mean0, class1 - mean1])
    pooled = np.einsum("ni,nj->ij", centered, centered) / max(train_x.shape[0] - 2, 1)
    diagonal_target = np.trace(pooled) / pooled.shape[0]
    covariance = (
        (1.0 - SHRINKAGE) * pooled
        + SHRINKAGE * diagonal_target * np.eye(pooled.shape[0])
        + COV_EPS * np.eye(pooled.shape[0])
    )
    prior1 = float(class1.shape[0] / train_x.shape[0])
    prior0 = 1.0 - prior1
    return {
        "mean0": mean0,
        "mean1": mean1,
        "precision": np.linalg.inv(covariance),
        "log_prior0": math.log(prior0),
        "log_prior1": math.log(prior1),
    }


def mahalanobis_squared(x: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    diff = x - mean
    return np.einsum("ni,ij,nj->n", diff, precision, diff)


def predict_prob(x: np.ndarray, model: dict[str, np.ndarray | float]) -> np.ndarray:
    d0 = mahalanobis_squared(x, model["mean0"], model["precision"])
    d1 = mahalanobis_squared(x, model["mean1"], model["precision"])
    log_odds = -0.5 * d1 + 0.5 * d0 + float(model["log_prior1"]) - float(model["log_prior0"])
    return sigmoid(log_odds)


def run_static_dataset(spec: StaticDatasetSpec) -> tuple[dict[str, object], list[dict[str, object]]]:
    x, y, data_hash = load_static_dataset(spec)
    folds = stratified_folds(y, MASTER_SEED + sum(ord(c) for c in spec.key))
    all_idx = np.arange(y.size)
    fold_rows: list[dict[str, object]] = []
    for fold_id, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        train_x, test_x = standardize(x[train_idx], x[test_idx])
        prob = predict_prob(test_x, fit_reference_model(train_x, y[train_idx]))
        raw_pred = (prob >= 0.5).astype(float)
        accepted = (prob >= P_ACCEPT_HIGH) | (prob <= P_ACCEPT_LOW)
        accepted_pred = (prob[accepted] >= 0.5).astype(float)
        accepted_y = y[test_idx][accepted]
        true_accept = int(np.sum(accepted_pred == accepted_y))
        false_accept = int(np.sum(accepted_pred != accepted_y))
        fold_rows.append(
            {
                "dataset": spec.key,
                "fold": fold_id,
                "n_test": int(test_idx.size),
                "raw_accuracy": float(np.mean(raw_pred == y[test_idx])),
                "accepted_decisions": int(np.sum(accepted)),
                "pending_decisions": int(np.sum(~accepted)),
                "accepted_true_reference": true_accept,
                "accepted_false_reference": false_accept,
                "accepted_reference_accuracy": true_accept / max(true_accept + false_accept, 1),
                "accepted_coverage": int(np.sum(accepted)) / test_idx.size,
                "pending_fraction": int(np.sum(~accepted)) / test_idx.size,
            }
        )

    total_n = sum(int(row["n_test"]) for row in fold_rows)
    accepted_total = sum(int(row["accepted_decisions"]) for row in fold_rows)
    pending_total = sum(int(row["pending_decisions"]) for row in fold_rows)
    true_total = sum(int(row["accepted_true_reference"]) for row in fold_rows)
    false_total = sum(int(row["accepted_false_reference"]) for row in fold_rows)
    summary = {
        "component": "acceptance_pending_layer",
        "dataset": spec.key,
        "dataset_name": spec.name,
        "source": spec.url,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_folds": N_FOLDS,
        "model": f"training-fold-standardized shrinkage LDA, shrinkage={SHRINKAGE:.2f}",
        "acceptance_rule": f"accepted if p<= {P_ACCEPT_LOW:.2f} or p>= {P_ACCEPT_HIGH:.2f}; otherwise pending",
        "raw_accuracy_mean_over_folds": float(np.mean([float(row["raw_accuracy"]) for row in fold_rows])),
        "accepted_reference_accuracy_over_accepted": true_total / max(true_total + false_total, 1),
        "accepted_coverage": accepted_total / total_n,
        "pending_fraction": pending_total / total_n,
        "accepted_true_reference": true_total,
        "accepted_false_reference": false_total,
        "accepted_total": accepted_total,
        "pending_total": pending_total,
        "data_sha256": data_hash,
        "status": "real_calculated_external_component",
    }
    return summary, fold_rows


def audit_appliances() -> tuple[dict[str, object], list[dict[str, object]]]:
    download(APPLIANCES_URL, APPLIANCES_CSV)
    df = pd.read_csv(APPLIANCES_CSV)
    appliances = df["Appliances"].to_numpy(dtype=float)
    timestamps = pd.to_datetime(df["date"]).astype(str).to_numpy()
    q50 = float(np.quantile(appliances, 0.50))
    q75 = float(np.quantile(appliances, 0.75))
    dwell_steps = 3
    retention_steps_primary = 6
    retention_steps_strict = 12
    refractory_steps = 3
    total_energy_kwh = float(np.sum(appliances) / 1000.0)

    events: list[dict[str, object]] = []
    last_event_idx = -10**9
    for i in range(1, len(appliances) - retention_steps_strict):
        if i - last_event_idx < refractory_steps:
            continue
        candidate = appliances[i - 1] <= q50 and appliances[i] >= q75
        if not candidate:
            continue
        dwell_ok = bool(np.all(appliances[i : i + dwell_steps] >= q75))
        if not dwell_ok:
            continue
        retained_60 = bool(np.all(appliances[i : i + retention_steps_primary] >= q50))
        retained_120 = bool(np.all(appliances[i : i + retention_steps_strict] >= q50))
        events.append(
            {
                "event_id": len(events) + 1,
                "timestamp": timestamps[i],
                "start_index": i,
                "q50_wh": q50,
                "q75_wh": q75,
                "dwell_minutes": dwell_steps * 10,
                "retention_primary_minutes": retention_steps_primary * 10,
                "retention_strict_minutes": retention_steps_strict * 10,
                "retained_60min": int(retained_60),
                "retained_120min": int(retained_120),
                "event_energy_60min_kwh": float(np.sum(appliances[i : i + retention_steps_primary]) / 1000.0),
                "event_energy_120min_kwh": float(np.sum(appliances[i : i + retention_steps_strict]) / 1000.0),
            }
        )
        last_event_idx = i

    retained_60 = sum(int(event["retained_60min"]) for event in events)
    retained_120 = sum(int(event["retained_120min"]) for event in events)
    summary = {
        "component": "temporal_retention_and_cost_denominator",
        "dataset": "appliances",
        "dataset_name": "UCI Appliances Energy Prediction",
        "source": APPLIANCES_URL,
        "n_samples": int(len(df)),
        "sample_interval_minutes": 10,
        "q50_wh": q50,
        "q75_wh": q75,
        "candidate_rule": "previous Appliances <= q50 and current Appliances >= q75",
        "dwell_rule": "Appliances >= q75 for 30 minutes",
        "retention_primary_rule": "after dwell, Appliances never exits below q50 for 60 minutes",
        "retention_strict_rule": "after dwell, Appliances never exits below q50 for 120 minutes",
        "candidate_after_dwell_events": len(events),
        "retained_60min_events": retained_60,
        "retained_120min_events": retained_120,
        "total_appliance_energy_kwh": total_energy_kwh,
        "retained_60min_yield_events_per_kwh": retained_60 / total_energy_kwh,
        "retained_120min_yield_events_per_kwh": retained_120 / total_energy_kwh,
        "data_sha256": sha256(APPLIANCES_CSV),
        "status": "real_calculated_external_component",
    }
    return summary, events


def audit_star() -> tuple[dict[str, object], dict[str, object]]:
    download(STAR_URL, STAR_CSV)
    df = pd.read_csv(STAR_CSV)
    sub = df[
        df["stark"].isin(["small", "regular"])
        & df["readk"].notna()
        & df["mathk"].notna()
    ].copy()
    small = sub[sub["stark"] == "small"]
    regular = sub[sub["stark"] == "regular"]

    def contrast(outcome: str) -> dict[str, float]:
        small_y = small[outcome].to_numpy(dtype=float)
        regular_y = regular[outcome].to_numpy(dtype=float)
        diff = float(np.mean(small_y) - np.mean(regular_y))
        se = math.sqrt(float(np.var(small_y, ddof=1)) / len(small_y) + float(np.var(regular_y, ddof=1)) / len(regular_y))
        return {
            "small_mean": float(np.mean(small_y)),
            "regular_mean": float(np.mean(regular_y)),
            "difference": diff,
            "standard_error": se,
            "ci95_low": diff - 1.96 * se,
            "ci95_high": diff + 1.96 * se,
        }

    read = contrast("readk")
    mathk = contrast("mathk")
    summary = {
        "component": "intervention_contrast",
        "dataset": "project_star",
        "dataset_name": "Project STAR kindergarten class-size dataset",
        "source": STAR_URL,
        "n_complete_small_regular": int(len(sub)),
        "small_class_n": int(len(small)),
        "regular_class_n": int(len(regular)),
        "assignment_variable": "stark",
        "reading_small_mean": read["small_mean"],
        "reading_regular_mean": read["regular_mean"],
        "reading_difference": read["difference"],
        "reading_ci95_low": read["ci95_low"],
        "reading_ci95_high": read["ci95_high"],
        "math_small_mean": mathk["small_mean"],
        "math_regular_mean": mathk["regular_mean"],
        "math_difference": mathk["difference"],
        "math_ci95_low": mathk["ci95_low"],
        "math_ci95_high": mathk["ci95_high"],
        "interpretation": "unadjusted small-versus-regular kindergarten contrast; not a full causal estimand analysis",
        "data_sha256": sha256(STAR_CSV),
        "status": "real_calculated_external_component",
    }
    star_row = dict(summary)
    return summary, star_row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for spec in STATIC_DATASETS:
        summary, folds = run_static_dataset(spec)
        summary_rows.append(summary)
        fold_rows.extend(folds)

    appliances_summary, appliance_events = audit_appliances()
    summary_rows.append(appliances_summary)

    star_summary, star_row = audit_star()
    summary_rows.append(star_summary)

    all_summary_keys = sorted({key for row in summary_rows for key in row})
    with OUT_SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_summary_keys)
        writer.writeheader()
        writer.writerows(summary_rows)
    write_csv(OUT_FOLDS, fold_rows)
    write_csv(OUT_APPLIANCES, appliance_events)
    write_csv(OUT_STAR, [star_row])

    for row in summary_rows:
        print(row["dataset"], row["component"], row["status"])
    print(f"wrote: {OUT_SUMMARY}")
    print(f"wrote: {OUT_FOLDS}")
    print(f"wrote: {OUT_APPLIANCES}")
    print(f"wrote: {OUT_STAR}")


if __name__ == "__main__":
    main()
