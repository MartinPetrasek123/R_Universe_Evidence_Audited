#!/usr/bin/env python3
"""Real-data verification audit for the VSTF manuscript.

This script uses the Wisconsin Diagnostic Breast Cancer data from the UCI
Machine Learning Repository. It is not a clinical validation study. It is a
locked, reproducible real-data audit showing how a VSTF-style acceptance layer
separates raw model activity from accepted, reference-checked decisions.
"""

from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import math
import urllib.request

import numpy as np
import pandas as pd


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
DATA_PATH = Path("/Users/mpetr/Desktop/wdbc.data")
OUT_SUMMARY = Path("/Users/mpetr/Desktop/vstf_real_data_wdbc_summary.csv")
OUT_FOLD = Path("/Users/mpetr/Desktop/vstf_real_data_wdbc_folds.csv")
MASTER_SEED = 20260919
N_FOLDS = 5
P_ACCEPT_HIGH = 0.98
P_ACCEPT_LOW = 0.02
SHRINKAGE = 0.10
COV_EPS = 1e-6


def ensure_data() -> None:
    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 100_000:
        return
    DATA_PATH.write_bytes(urllib.request.urlopen(DATA_URL, timeout=30).read())


def load_wdbc() -> tuple[np.ndarray, np.ndarray]:
    ensure_data()
    df = pd.read_csv(DATA_PATH, header=None)
    y = (df.iloc[:, 1].to_numpy() == "M").astype(float)
    x = df.iloc[:, 2:].to_numpy(dtype=float)
    return x, y


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
    pooled = np.einsum("ni,nj->ij", centered, centered) / (train_x.shape[0] - 2)
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


def _mahalanobis_squared(x: np.ndarray, mean: np.ndarray, precision: np.ndarray) -> np.ndarray:
    diff = x - mean
    return np.einsum("ni,ij,nj->n", diff, precision, diff)


def predict_prob(x: np.ndarray, model: dict[str, np.ndarray | float]) -> np.ndarray:
    d0 = _mahalanobis_squared(x, model["mean0"], model["precision"])
    d1 = _mahalanobis_squared(x, model["mean1"], model["precision"])
    log_odds = -0.5 * d1 + 0.5 * d0 + float(model["log_prior1"]) - float(model["log_prior0"])
    return sigmoid(log_odds)


def metrics_for_fold(fold_id: int, prob: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    raw_pred = (prob >= 0.5).astype(float)
    accepted = (prob >= P_ACCEPT_HIGH) | (prob <= P_ACCEPT_LOW)
    accepted_pred = (prob[accepted] >= 0.5).astype(float)
    accepted_y = y[accepted]
    raw_correct = raw_pred == y
    accepted_correct = accepted_pred == accepted_y
    false_accept = int(np.sum(~accepted_correct))
    true_accept = int(np.sum(accepted_correct))
    pending = int(np.sum(~accepted))
    return {
        "fold": fold_id,
        "n_test": int(y.size),
        "raw_decisions": int(y.size),
        "raw_accuracy": float(np.mean(raw_correct)),
        "accepted_decisions": int(np.sum(accepted)),
        "pending_decisions": pending,
        "accepted_coverage": float(np.mean(accepted)),
        "accepted_reference_accuracy": float(np.mean(accepted_correct)) if accepted_y.size else math.nan,
        "accepted_true_reference": true_accept,
        "accepted_false_reference": false_accept,
        "false_acceptance_rate_among_accepted": false_accept / max(true_accept + false_accept, 1),
        "pending_fraction": pending / y.size,
    }


def main() -> None:
    x, y = load_wdbc()
    rng = np.random.default_rng(MASTER_SEED)
    order = rng.permutation(x.shape[0])
    folds = np.array_split(order, N_FOLDS)
    rows = []
    for fold_id, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(order, test_idx, assume_unique=True)
        train_x, test_x = standardize(x[train_idx], x[test_idx])
        model = fit_reference_model(train_x, y[train_idx])
        prob = predict_prob(test_x, model)
        rows.append(metrics_for_fold(fold_id, prob, y[test_idx]))

    with OUT_FOLD.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_raw = sum(r["raw_decisions"] for r in rows)
    total_acc = sum(r["accepted_decisions"] for r in rows)
    total_pending = sum(r["pending_decisions"] for r in rows)
    total_true = sum(r["accepted_true_reference"] for r in rows)
    total_false = sum(r["accepted_false_reference"] for r in rows)
    summary = {
        "dataset": "Wisconsin Diagnostic Breast Cancer",
        "source": DATA_URL,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_folds": N_FOLDS,
        "model": f"training-fold-standardized shrinkage LDA, shrinkage={SHRINKAGE:.2f}",
        "acceptance_rule": f"accepted if p<= {P_ACCEPT_LOW:.2f} or p>= {P_ACCEPT_HIGH:.2f}; otherwise pending",
        "raw_accuracy_mean_over_folds": float(np.mean([r["raw_accuracy"] for r in rows])),
        "accepted_reference_accuracy_over_accepted": total_true / max(total_true + total_false, 1),
        "accepted_coverage": total_acc / total_raw,
        "pending_fraction": total_pending / total_raw,
        "accepted_false_reference": int(total_false),
        "accepted_true_reference": int(total_true),
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
    }
    with OUT_SUMMARY.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": value})

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
