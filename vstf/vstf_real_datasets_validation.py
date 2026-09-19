#!/usr/bin/env python3
"""Multi-dataset real-data verification audit for the VSTF manuscript.

The script downloads public real datasets, runs the same locked train/test
protocol on each dataset, and reports raw classification accuracy versus the
accuracy and coverage of a conservative VSTF-style acceptance layer.
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

OUT_SUMMARY = OUT_DIR / "vstf_real_datasets_summary.csv"
OUT_FOLDS = OUT_DIR / "vstf_real_datasets_folds.csv"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    url: str
    filename: str
    loader: str


DATASETS = [
    DatasetSpec(
        key="wdbc",
        name="Wisconsin Diagnostic Breast Cancer",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        filename="wdbc.data",
        loader="wdbc",
    ),
    DatasetSpec(
        key="ionosphere",
        name="Johns Hopkins Ionosphere",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
        filename="ionosphere.data",
        loader="ionosphere",
    ),
    DatasetSpec(
        key="sonar",
        name="Connectionist Bench Sonar",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
        filename="sonar.all-data",
        loader="sonar",
    ),
    DatasetSpec(
        key="cleveland",
        name="Cleveland Heart Disease",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
        filename="processed.cleveland.data",
        loader="cleveland",
    ),
    DatasetSpec(
        key="haberman",
        name="Haberman Survival",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data",
        filename="haberman.data",
        loader="haberman",
    ),
]


def ensure_file(spec: DatasetSpec) -> Path:
    path = OUT_DIR / spec.filename
    if path.exists() and path.stat().st_size > 0:
        return path
    path.write_bytes(urllib.request.urlopen(spec.url, timeout=30).read())
    return path


def load_dataset(spec: DatasetSpec) -> tuple[np.ndarray, np.ndarray]:
    path = ensure_file(spec)
    if spec.loader == "wdbc":
        df = pd.read_csv(path, header=None)
        y = (df.iloc[:, 1].to_numpy() == "M").astype(float)
        x = df.iloc[:, 2:].to_numpy(dtype=float)
    elif spec.loader == "ionosphere":
        df = pd.read_csv(path, header=None)
        y = (df.iloc[:, -1].to_numpy() == "g").astype(float)
        x = df.iloc[:, :-1].to_numpy(dtype=float)
    elif spec.loader == "sonar":
        df = pd.read_csv(path, header=None)
        y = (df.iloc[:, -1].to_numpy() == "M").astype(float)
        x = df.iloc[:, :-1].to_numpy(dtype=float)
    elif spec.loader == "cleveland":
        df = pd.read_csv(path, header=None, na_values="?").dropna(axis=0)
        y = (df.iloc[:, -1].to_numpy(dtype=float) > 0.0).astype(float)
        x = df.iloc[:, :-1].to_numpy(dtype=float)
    elif spec.loader == "haberman":
        df = pd.read_csv(path, header=None)
        y = (df.iloc[:, -1].to_numpy(dtype=float) == 2.0).astype(float)
        x = df.iloc[:, :-1].to_numpy(dtype=float)
    else:
        raise ValueError(f"unknown loader: {spec.loader}")

    finite = np.isfinite(x).all(axis=1) & np.isfinite(y)
    return x[finite], y[finite]


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


def fold_metrics(dataset_key: str, fold_id: int, prob: np.ndarray, y: np.ndarray) -> dict[str, float | int | str]:
    raw_pred = (prob >= 0.5).astype(float)
    accepted = (prob >= P_ACCEPT_HIGH) | (prob <= P_ACCEPT_LOW)
    accepted_pred = (prob[accepted] >= 0.5).astype(float)
    accepted_y = y[accepted]
    accepted_correct = accepted_pred == accepted_y
    true_accept = int(np.sum(accepted_correct))
    false_accept = int(np.sum(~accepted_correct))
    pending = int(np.sum(~accepted))
    return {
        "dataset": dataset_key,
        "fold": fold_id,
        "n_test": int(y.size),
        "raw_accuracy": float(np.mean(raw_pred == y)),
        "accepted_decisions": int(np.sum(accepted)),
        "pending_decisions": pending,
        "accepted_true_reference": true_accept,
        "accepted_false_reference": false_accept,
        "accepted_reference_accuracy": true_accept / max(true_accept + false_accept, 1),
        "accepted_coverage": int(np.sum(accepted)) / y.size,
        "pending_fraction": pending / y.size,
    }


def calibration_metrics(prob: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    pred = (prob >= 0.5).astype(float)
    confidence = np.maximum(prob, 1.0 - prob)
    correct = (pred == y).astype(float)
    order = np.argsort(-confidence)
    sorted_errors = 1.0 - correct[order]
    cumulative_risk = np.cumsum(sorted_errors) / np.arange(1, y.size + 1)

    ece = 0.0
    for lo in np.linspace(0.0, 1.0, n_bins, endpoint=False):
        hi = lo + 1.0 / n_bins
        if hi >= 1.0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(correct[mask]))
        bin_conf = float(np.mean(confidence[mask]))
        ece += float(np.mean(mask)) * abs(bin_acc - bin_conf)

    def risk_at_coverage(coverage: float) -> float:
        k = max(1, int(math.ceil(coverage * y.size)))
        return float(cumulative_risk[k - 1])

    return {
        "brier_score": float(np.mean((prob - y) ** 2)),
        "ece_10bin": ece,
        "aurc_confidence": float(np.mean(cumulative_risk)),
        "risk_at_50pct_coverage": risk_at_coverage(0.50),
        "risk_at_80pct_coverage": risk_at_coverage(0.80),
    }


def run_dataset(spec: DatasetSpec) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    x, y = load_dataset(spec)
    folds = stratified_folds(y, MASTER_SEED + sum(ord(c) for c in spec.key))
    rows = []
    all_prob = np.empty(y.size, dtype=float)
    all_idx = np.arange(y.size)
    for fold_id, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        train_x, test_x = standardize(x[train_idx], x[test_idx])
        model = fit_reference_model(train_x, y[train_idx])
        prob = predict_prob(test_x, model)
        all_prob[test_idx] = prob
        rows.append(fold_metrics(spec.key, fold_id, prob, y[test_idx]))

    total_test = sum(int(r["n_test"]) for r in rows)
    total_accepted = sum(int(r["accepted_decisions"]) for r in rows)
    total_pending = sum(int(r["pending_decisions"]) for r in rows)
    total_true = sum(int(r["accepted_true_reference"]) for r in rows)
    total_false = sum(int(r["accepted_false_reference"]) for r in rows)
    path = ensure_file(spec)
    summary = {
        "dataset": spec.key,
        "dataset_name": spec.name,
        "source": spec.url,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "positive_class_count": int(np.sum(y == 1.0)),
        "negative_class_count": int(np.sum(y == 0.0)),
        "n_folds": N_FOLDS,
        "model": f"training-fold-standardized shrinkage LDA, shrinkage={SHRINKAGE:.2f}",
        "acceptance_rule": f"accepted if p<= {P_ACCEPT_LOW:.2f} or p>= {P_ACCEPT_HIGH:.2f}; otherwise pending",
        "raw_accuracy_mean_over_folds": float(np.mean([float(r["raw_accuracy"]) for r in rows])),
        "accepted_reference_accuracy_over_accepted": total_true / max(total_true + total_false, 1),
        "accepted_coverage": total_accepted / total_test,
        "pending_fraction": total_pending / total_test,
        "accepted_true_reference": total_true,
        "accepted_false_reference": total_false,
        "accepted_total": total_accepted,
        "pending_total": total_pending,
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    summary.update(calibration_metrics(all_prob, y))
    return summary, rows


def main() -> None:
    summaries = []
    fold_rows = []
    for spec in DATASETS:
        summary, rows = run_dataset(spec)
        summaries.append(summary)
        fold_rows.extend(rows)

    with OUT_SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    with OUT_FOLDS.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)

    for row in summaries:
        print(
            f"{row['dataset']}: n={row['n_samples']}, raw={row['raw_accuracy_mean_over_folds']:.4f}, "
            f"accepted={row['accepted_reference_accuracy_over_accepted']:.4f}, "
            f"coverage={row['accepted_coverage']:.4f}, false={row['accepted_false_reference']}, "
            f"pending={row['pending_total']}"
        )


if __name__ == "__main__":
    main()
