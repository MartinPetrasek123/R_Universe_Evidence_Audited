#!/usr/bin/env python3
"""Regenerate the VSTF evidence package.

This script is the single entry point for the reproducibility package. It
regenerates the synthetic benchmark, the public real-data acceptance audits,
the temporal/cost/treatment-comparison component audits, and a checksum
manifest for all generated outputs.
"""

from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parent

SYNTHETIC_OUTPUTS = [
    "vstf_hmm_benchmark_summary.csv",
    "vstf_hmm_benchmark_monte_carlo.csv",
    "vstf_hmm_benchmark_batch_level.csv",
    "vstf_hmm_benchmark_sensitivity.csv",
    "vstf_hmm_sampling_resolution.csv",
    "vstf_hmm_benchmark_ablations.csv",
    "vstf_hmm_random_parameter_stress.csv",
    "vstf_hmm_scenario_stress.csv",
    "vstf_model_misspecification_stress.csv",
    "vstf_many_system_benchmark.csv",
    "vstf_many_system_discordance.csv",
    "vstf_hmm_event_sample.csv",
    "vstf_event_cascade.pdf",
    "vstf_hmm_benchmark.pdf",
    "vstf_many_system_rank_displacement.pdf",
]

REAL_DATA_OUTPUTS = [
    "vstf_real_datasets_summary.csv",
    "vstf_real_datasets_folds.csv",
    "vstf_real_component_validation_summary.csv",
    "vstf_household_power_events.csv",
    "vstf_lalonde_treatment_effect.csv",
    "vstf_external_component_validation_summary.csv",
    "vstf_external_component_validation_folds.csv",
    "vstf_external_appliances_events.csv",
    "vstf_external_star_contrast.csv",
    "vstf_full_real_calculation_ledger.csv",
]


def run(script: str) -> None:
    print(f"running {script}")
    subprocess.run([sys.executable, script], cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_outputs(names: list[str]) -> None:
    missing = [name for name in names if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing expected outputs: {missing}")


def validate_sensitivity_table() -> None:
    path = ROOT / "vstf_hmm_benchmark_sensitivity.csv"
    rows = list(csv.DictReader(path.open()))
    expected = {
        ("0.75", "4.0"): "0.0",
        ("0.75", "6.0"): "0.0",
        ("0.75", "10.0"): "0.0",
        ("1.0", "4.0"): "0.0",
        ("1.0", "6.0"): "0.0",
        ("1.0", "10.0"): "0.0",
        ("1.25", "4.0"): "0.43333333333333335",
        ("1.25", "6.0"): "0.5333333333333333",
        ("1.25", "10.0"): "0.8333333333333334",
    }
    observed = {
        (row["sigma_factor"], row["tau_ret"]): row["reversal_probability"]
        for row in rows
    }
    if observed != expected:
        raise AssertionError(f"sensitivity CSV does not match manuscript table: {observed}")


def validate_manuscript_numbers() -> None:
    """Fail if manuscript literals drift away from generated CSV outputs."""
    manuscript = (ROOT / "main.tex").read_text()

    hmm_rows = list(csv.DictReader((ROOT / "vstf_hmm_benchmark_summary.csv").open()))
    system_a, system_b = hmm_rows[0], hmm_rows[1]
    expected_literals = [
        f"{int(float(system_a['raw_activity'])):,}",
        f"{int(float(system_a['crossings'])):,}",
        f"{int(float(system_a['committed'])):,}",
        f"{int(float(system_b['raw_activity'])):,}",
        f"{int(float(system_b['crossings'])):,}",
        f"{int(float(system_b['committed'])):,}",
        f"{int(float(system_a['valid_vst'])):,}",
        f"{int(float(system_b['valid_vst'])):,}",
    ]

    external_rows = {
        row["dataset"]: row
        for row in csv.DictReader((ROOT / "vstf_external_component_validation_summary.csv").open())
    }
    banknote = external_rows["banknote"]
    spambase = external_rows["spambase"]
    appliances = external_rows["appliances"]
    star = external_rows["project_star"]
    expected_literals.extend(
        [
            f"{float(banknote['raw_accuracy_mean_over_folds']):.4f}",
            f"{int(float(banknote['accepted_true_reference']))}/{int(float(banknote['accepted_total']))}",
            f"{float(banknote['accepted_coverage']):.4f}",
            f"{float(banknote['pending_fraction']):.4f}",
            f"{float(spambase['raw_accuracy_mean_over_folds']):.4f}",
            f"{int(float(spambase['accepted_true_reference']))}/{int(float(spambase['accepted_total']))}",
            f"{float(spambase['accepted_reference_accuracy_over_accepted']):.4f}",
            f"{float(spambase['accepted_coverage']):.4f}",
            f"{int(float(appliances['candidate_after_dwell_events']))}",
            f"{int(float(appliances['retained_60min_events']))}/{int(float(appliances['candidate_after_dwell_events']))}",
            f"{int(float(appliances['retained_120min_events']))}/{int(float(appliances['candidate_after_dwell_events']))}",
            f"{float(appliances['retained_120min_yield_events_per_kwh']):.4f}",
            f"{float(star['reading_difference']):.4f}",
            f"{float(star['math_difference']):.4f}",
        ]
    )
    missing = [literal for literal in expected_literals if literal not in manuscript]
    if missing:
        raise AssertionError(f"manuscript is missing generated values: {missing}")


def write_full_real_ledger() -> None:
    rows: list[dict[str, str]] = []
    with (ROOT / "vstf_real_datasets_summary.csv").open() as f:
        for row in csv.DictReader(f):
            dataset = row["dataset"]
            name = row["dataset_name"]
            source = row["source"]
            n = row["n_samples"]
            accepted = row["accepted_total"]
            pending = row["pending_total"]
            true = row["accepted_true_reference"]
            false = row["accepted_false_reference"]
            accepted_total = int(true) + int(false)
            rows.extend(
                [
                    {
                        "component": "raw_decision_accuracy",
                        "dataset": dataset,
                        "dataset_name": name,
                        "source": source,
                        "formula": "mean(fold raw accuracies)",
                        "numerator": "see fold CSV",
                        "denominator": "5 folds",
                        "value": row["raw_accuracy_mean_over_folds"],
                        "status": "real_calculated",
                    },
                    {
                        "component": "accepted_decision_count",
                        "dataset": dataset,
                        "dataset_name": name,
                        "source": source,
                        "formula": "sum(accepted decisions over folds)",
                        "numerator": accepted,
                        "denominator": n,
                        "value": accepted,
                        "status": "real_calculated",
                    },
                    {
                        "component": "pending_decision_count",
                        "dataset": dataset,
                        "dataset_name": name,
                        "source": source,
                        "formula": "sum(pending decisions over folds)",
                        "numerator": pending,
                        "denominator": n,
                        "value": pending,
                        "status": "real_calculated",
                    },
                    {
                        "component": "accepted_reference_accuracy",
                        "dataset": dataset,
                        "dataset_name": name,
                        "source": source,
                        "formula": "accepted_true/(accepted_true+accepted_false)",
                        "numerator": true,
                        "denominator": str(accepted_total),
                        "value": row["accepted_reference_accuracy_over_accepted"],
                        "status": "real_calculated"
                        if accepted_total
                        else "real_calculated_zero_accepted_negative_result",
                    },
                    {
                        "component": "accepted_coverage",
                        "dataset": dataset,
                        "dataset_name": name,
                        "source": source,
                        "formula": "accepted_total/n_samples",
                        "numerator": accepted,
                        "denominator": n,
                        "value": row["accepted_coverage"],
                        "status": "real_calculated",
                    },
                    {
                        "component": "pending_fraction",
                        "dataset": dataset,
                        "dataset_name": name,
                        "source": source,
                        "formula": "pending_total/n_samples",
                        "numerator": pending,
                        "denominator": n,
                        "value": row["pending_fraction"],
                        "status": "real_calculated",
                    },
                    {
                        "component": "brier_score",
                        "dataset": dataset,
                        "dataset_name": name,
                        "source": source,
                        "formula": "mean((p-y)^2) over held-out fold predictions",
                        "numerator": "see prediction probabilities in script",
                        "denominator": n,
                        "value": row["brier_score"],
                        "status": "real_calculated",
                    },
                    {
                        "component": "ece_10bin",
                        "dataset": dataset,
                        "dataset_name": name,
                        "source": source,
                        "formula": "10-bin expected calibration error over held-out fold predictions",
                        "numerator": "weighted absolute calibration gaps",
                        "denominator": n,
                        "value": row["ece_10bin"],
                        "status": "real_calculated",
                    },
                    {
                        "component": "aurc_confidence",
                        "dataset": dataset,
                        "dataset_name": name,
                        "source": source,
                        "formula": "mean selective risk while sweeping accepted coverage by confidence",
                        "numerator": "cumulative held-out errors ordered by confidence",
                        "denominator": n,
                        "value": row["aurc_confidence"],
                        "status": "real_calculated",
                    },
                ]
            )

    with (ROOT / "vstf_real_component_validation_summary.csv").open() as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "component": row["component"],
                    "dataset": row["dataset"],
                    "dataset_name": row["dataset"],
                    "source": row["source"],
                    "formula": row["formula"],
                    "numerator": row["numerator"],
                    "denominator": row["denominator"],
                    "value": row["value"],
                    "status": row["status"],
                }
            )

    with (ROOT / "vstf_external_component_validation_summary.csv").open() as f:
        for row in csv.DictReader(f):
            component = row["component"]
            dataset = row["dataset"]
            source = row["source"]
            dataset_name = row["dataset_name"]
            if dataset in {"banknote", "spambase"}:
                rows.extend(
                    [
                        {
                            "component": f"external_{component}_raw_accuracy",
                            "dataset": dataset,
                            "dataset_name": dataset_name,
                            "source": source,
                            "formula": "mean(fold raw accuracies)",
                            "numerator": "see external fold CSV",
                            "denominator": "5 folds",
                            "value": row["raw_accuracy_mean_over_folds"],
                            "status": row["status"],
                        },
                        {
                            "component": f"external_{component}_accepted_accuracy",
                            "dataset": dataset,
                            "dataset_name": dataset_name,
                            "source": source,
                            "formula": "accepted_true/(accepted_true+accepted_false)",
                            "numerator": row["accepted_true_reference"],
                            "denominator": row["accepted_total"],
                            "value": row["accepted_reference_accuracy_over_accepted"],
                            "status": row["status"],
                        },
                        {
                            "component": f"external_{component}_coverage",
                            "dataset": dataset,
                            "dataset_name": dataset_name,
                            "source": source,
                            "formula": "accepted_total/n_samples",
                            "numerator": row["accepted_total"],
                            "denominator": row["n_samples"],
                            "value": row["accepted_coverage"],
                            "status": row["status"],
                        },
                    ]
                )
            elif dataset == "appliances":
                rows.extend(
                    [
                        {
                            "component": "external_appliances_temporal_retention_60min",
                            "dataset": dataset,
                            "dataset_name": dataset_name,
                            "source": source,
                            "formula": row["retention_primary_rule"],
                            "numerator": row["retained_60min_events"],
                            "denominator": row["candidate_after_dwell_events"],
                            "value": row["retained_60min_yield_events_per_kwh"],
                            "status": row["status"],
                        },
                        {
                            "component": "external_appliances_temporal_retention_120min",
                            "dataset": dataset,
                            "dataset_name": dataset_name,
                            "source": source,
                            "formula": row["retention_strict_rule"],
                            "numerator": row["retained_120min_events"],
                            "denominator": row["candidate_after_dwell_events"],
                            "value": row["retained_120min_yield_events_per_kwh"],
                            "status": row["status"],
                        },
                    ]
                )
            elif dataset == "project_star":
                rows.extend(
                    [
                        {
                            "component": "external_project_star_reading_difference",
                            "dataset": dataset,
                            "dataset_name": dataset_name,
                            "source": source,
                            "formula": "mean(readk | small class) - mean(readk | regular class)",
                            "numerator": row["reading_difference"],
                            "denominator": "test-score scale",
                            "value": row["reading_difference"],
                            "status": row["status"],
                        },
                        {
                            "component": "external_project_star_math_difference",
                            "dataset": dataset,
                            "dataset_name": dataset_name,
                            "source": source,
                            "formula": "mean(mathk | small class) - mean(mathk | regular class)",
                            "numerator": row["math_difference"],
                            "denominator": "test-score scale",
                            "value": row["math_difference"],
                            "status": row["status"],
                        },
                    ]
                )

    fields = [
        "component",
        "dataset",
        "dataset_name",
        "source",
        "formula",
        "numerator",
        "denominator",
        "value",
        "status",
    ]
    with (ROOT / "vstf_full_real_calculation_ledger.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest() -> None:
    outputs = SYNTHETIC_OUTPUTS + REAL_DATA_OUTPUTS + [
        "main.tex",
        "vstf_hmm_benchmark.py",
        "vstf_real_datasets_validation.py",
        "vstf_real_component_validation.py",
        "vstf_external_component_validation.py",
        "reproduce_all.py",
        "README.md",
        "REPRODUCIBILITY.md",
        "FULL_VALIDATION_PROTOCOL.md",
        "requirements.txt",
    ]
    with (ROOT / "EVIDENCE_MANIFEST.sha256").open("w") as f:
        f.write(f"Python: {sys.version.split()[0]}\n")
        f.write(f"Platform: {platform.platform()}\n")
        for name in outputs:
            path = ROOT / name
            if path.exists():
                f.write(f"{sha256(path)}  {name}\n")


def main() -> None:
    run("vstf_hmm_benchmark.py")
    require_outputs(SYNTHETIC_OUTPUTS)
    validate_sensitivity_table()

    run("vstf_real_datasets_validation.py")
    run("vstf_real_component_validation.py")
    run("vstf_external_component_validation.py")
    write_full_real_ledger()
    require_outputs(REAL_DATA_OUTPUTS)
    validate_manuscript_numbers()

    write_manifest()
    print("all evidence outputs regenerated successfully")


if __name__ == "__main__":
    main()
