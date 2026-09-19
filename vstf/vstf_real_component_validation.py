#!/usr/bin/env python3
"""Real-data component validation ledger for VSTF.

This script computes VSTF components that cannot be measured from static
classification rows alone:

1. Temporal event segmentation, dwell, first-exit/retention, and
   cost-normalized VSTE on a real household power time series.
2. Intervention/treatment effect on a real treatment dataset.

The script writes machine-readable component ledgers. It does not edit the
manuscript.
"""

from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import io
import math
import urllib.request
import zipfile

import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent

POWER_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
POWER_ZIP = OUT_DIR / "household_power_consumption.zip"
POWER_TXT = OUT_DIR / "household_power_consumption.txt"

LALONDE_URL = "https://vincentarelbundock.github.io/Rdatasets/csv/MatchIt/lalonde.csv"
LALONDE_CSV = OUT_DIR / "lalonde.csv"

OUT_COMPONENTS = OUT_DIR / "vstf_real_component_validation_summary.csv"
OUT_POWER_EVENTS = OUT_DIR / "vstf_household_power_events.csv"
OUT_LALONDE = OUT_DIR / "vstf_lalonde_treatment_effect.csv"


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.write_bytes(urllib.request.urlopen(url, timeout=120).read())


def ensure_power_data() -> None:
    download(POWER_URL, POWER_ZIP)
    if POWER_TXT.exists() and POWER_TXT.stat().st_size > 0:
        return
    with zipfile.ZipFile(POWER_ZIP) as zf:
        name = [n for n in zf.namelist() if n.endswith(".txt")][0]
        POWER_TXT.write_bytes(zf.read(name))


def ensure_lalonde_data() -> None:
    download(LALONDE_URL, LALONDE_CSV)


def load_power() -> pd.DataFrame:
    ensure_power_data()
    cols = ["Date", "Time", "Global_active_power"]
    df = pd.read_csv(
        POWER_TXT,
        sep=";",
        usecols=cols,
        na_values="?",
        low_memory=False,
    )
    dt = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    power_df = pd.DataFrame(
        {
            "timestamp": dt,
            "power_kw": pd.to_numeric(df["Global_active_power"], errors="coerce"),
        }
    )
    full_index = pd.date_range(power_df["timestamp"].iloc[0], power_df["timestamp"].iloc[-1], freq="min")
    power_df = power_df.set_index("timestamp").reindex(full_index)
    power_df.index.name = "timestamp"
    power_df["is_missing"] = power_df["power_kw"].isna()
    return power_df.reset_index()


def audit_power_transitions(df: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, object]]:
    power = df["power_kw"].to_numpy(dtype=float)
    missing = df["is_missing"].to_numpy(dtype=bool)
    observed_power = power[~missing]
    q25 = float(np.quantile(observed_power, 0.25))
    q75 = float(np.quantile(observed_power, 0.75))
    low_threshold = q25
    high_threshold = q75
    dwell_minutes = 5
    retention_minutes = 30
    refractory_minutes = 5
    sample_hours = 1.0 / 60.0
    total_observed_energy_kwh = float(np.nansum(power) * sample_hours)
    observed_median_kw = float(np.median(observed_power))
    missing_samples = int(np.sum(missing))
    estimated_missing_energy_kwh = float(missing_samples * observed_median_kw * sample_hours)
    estimated_total_energy_kwh = total_observed_energy_kwh + estimated_missing_energy_kwh

    events: list[dict[str, object]] = []
    last_event_idx = -10**9
    for i in range(1, len(power) - retention_minutes):
        if i - last_event_idx < refractory_minutes:
            continue
        if missing[i - 1] or missing[i]:
            continue
        candidate = power[i - 1] <= low_threshold and power[i] >= high_threshold
        if not candidate:
            continue
        dwell_missing = bool(np.any(missing[i : i + dwell_minutes]))
        segment_missing = bool(np.any(missing[i : i + retention_minutes]))
        n_missing_window = int(np.sum(missing[i : i + retention_minutes]))
        segment = power[i : i + retention_minutes]
        segment_observed = segment[~missing[i : i + retention_minutes]]
        if dwell_missing or segment_missing:
            status = "pending_missing_window"
            retained = 0
            first_exit = None
        else:
            dwell_ok = bool(np.all(power[i : i + dwell_minutes] >= high_threshold))
            if not dwell_ok:
                status = "rejected_dwell_failure"
                retained = 0
                first_exit = None
            else:
                retained_bool = bool(np.all(segment >= low_threshold))
                retained = int(retained_bool)
                first_exit = next((j for j, value in enumerate(segment) if value < low_threshold), None)
                status = "accepted_retained" if retained_bool else "rejected_first_exit"

        observed_event_energy = float(np.nansum(segment) * sample_hours)
        estimated_event_energy = observed_event_energy + float(n_missing_window * observed_median_kw * sample_hours)
        peak_power = float(np.nanmax(segment_observed)) if segment_observed.size else float("nan")
        mean_power = float(np.nanmean(segment_observed)) if segment_observed.size else float("nan")
        segment = power[i : i + retention_minutes]
        events.append(
            {
                "event_id": len(events) + 1,
                "timestamp": str(df["timestamp"].iloc[i]),
                "start_index": i,
                "status": status,
                "low_threshold_kw": low_threshold,
                "high_threshold_kw": high_threshold,
                "dwell_minutes": dwell_minutes,
                "retention_minutes": retention_minutes,
                "retained": int(retained),
                "first_exit_minute": "" if first_exit is None else int(first_exit),
                "missing_in_dwell": int(dwell_missing),
                "missing_in_retention": int(segment_missing),
                "n_missing_in_window": n_missing_window,
                "observed_event_energy_kwh": observed_event_energy,
                "estimated_event_energy_kwh": estimated_event_energy,
                "peak_power_kw": peak_power,
                "mean_power_kw": mean_power,
            }
        )
        last_event_idx = i

    accepted = [e for e in events if e["status"] == "accepted_retained"]
    rejected_first_exit = [e for e in events if e["status"] == "rejected_first_exit"]
    rejected_dwell = [e for e in events if e["status"] == "rejected_dwell_failure"]
    pending_missing = [e for e in events if e["status"] == "pending_missing_window"]
    retained_energy = sum(float(e["observed_event_energy_kwh"]) for e in accepted)
    resolved_retention_denominator = len(accepted) + len(rejected_first_exit)
    summary = {
        "dataset": "UCI Individual household electric power consumption",
        "source": POWER_URL,
        "n_samples_full_minute_grid": int(len(df)),
        "n_samples_observed": int(np.sum(~missing)),
        "n_samples_missing": missing_samples,
        "missing_fraction": missing_samples / len(df),
        "time_span_start": str(df["timestamp"].iloc[0]),
        "time_span_end": str(df["timestamp"].iloc[-1]),
        "sample_interval_minutes": 1,
        "low_threshold_kw_q25": low_threshold,
        "high_threshold_kw_q75": high_threshold,
        "candidate_rule": "low-to-high crossing: previous power <= q25 and current power >= q75",
        "dwell_rule": f"power >= q75 for {dwell_minutes} consecutive minutes",
        "retention_rule": f"after dwell, power never exits below q25 during {retention_minutes} minutes",
        "raw_candidate_events": len(events),
        "candidate_after_dwell_events": resolved_retention_denominator,
        "accepted_retained_events": len(accepted),
        "rejected_first_exit_events": len(rejected_first_exit),
        "rejected_dwell_failure_events": len(rejected_dwell),
        "pending_missing_window_events": len(pending_missing),
        "retention_fraction": len(accepted) / max(resolved_retention_denominator, 1),
        "total_observed_energy_kwh": total_observed_energy_kwh,
        "estimated_missing_energy_kwh": estimated_missing_energy_kwh,
        "estimated_total_energy_kwh": estimated_total_energy_kwh,
        "denominator_complete": int(missing_samples == 0),
        "accepted_event_energy_kwh": retained_energy,
        "cost_normalized_vste_events_per_observed_kwh": len(accepted) / total_observed_energy_kwh,
        "cost_normalized_vste_events_per_estimated_kwh": len(accepted) / estimated_total_energy_kwh,
        "accepted_event_energy_share_observed": retained_energy / total_observed_energy_kwh,
        "data_sha256": hashlib.sha256(POWER_TXT.read_bytes()).hexdigest(),
    }
    return events, summary


def load_lalonde() -> pd.DataFrame:
    ensure_lalonde_data()
    return pd.read_csv(LALONDE_CSV)


def audit_lalonde(df: pd.DataFrame) -> dict[str, object]:
    treated = df[df["treat"] == 1].copy()
    control = df[df["treat"] == 0].copy()
    y_t = treated["re78"].to_numpy(dtype=float)
    y_c = control["re78"].to_numpy(dtype=float)
    effect = float(np.mean(y_t) - np.mean(y_c))

    # Conservative acceptance layer: count individual positive earnings transitions
    # from 1975 to 1978, then compare treatment and control fractions.
    treated_transition = (treated["re78"].to_numpy(dtype=float) > treated["re75"].to_numpy(dtype=float)).astype(int)
    control_transition = (control["re78"].to_numpy(dtype=float) > control["re75"].to_numpy(dtype=float)).astype(int)
    p_t = float(np.mean(treated_transition))
    p_c = float(np.mean(control_transition))
    risk_difference = p_t - p_c

    se = math.sqrt(np.var(y_t, ddof=1) / len(y_t) + np.var(y_c, ddof=1) / len(y_c))
    ci_low = effect - 1.96 * se
    ci_high = effect + 1.96 * se
    return {
        "dataset": "Lalonde/MatchIt NSW treated versus PSID comparison dataset",
        "source": LALONDE_URL,
        "n_samples": int(len(df)),
        "treated_n": int(len(treated)),
        "psid_comparison_n": int(len(control)),
        "outcome": "real earnings in 1978 (re78)",
        "treated_mean_re78": float(np.mean(y_t)),
        "psid_comparison_mean_re78": float(np.mean(y_c)),
        "difference_in_means_re78": effect,
        "standard_error_difference": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "treated_positive_transition_fraction_re78_gt_re75": p_t,
        "psid_comparison_positive_transition_fraction_re78_gt_re75": p_c,
        "positive_transition_fraction_difference": risk_difference,
        "interpretation": "descriptive treated-versus-PSID comparison; not an unadjusted causal estimate",
        "data_sha256": hashlib.sha256(LALONDE_CSV.read_bytes()).hexdigest(),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    power_df = load_power()
    power_events, power_summary = audit_power_transitions(power_df)
    write_csv(OUT_POWER_EVENTS, power_events)

    lalonde_df = load_lalonde()
    lalonde_summary = audit_lalonde(lalonde_df)
    write_csv(OUT_LALONDE, [lalonde_summary])

    component_rows = [
        {
            "component": "temporal_event_segmentation",
            "dataset": power_summary["dataset"],
            "source": power_summary["source"],
            "formula": power_summary["candidate_rule"],
            "numerator": power_summary["raw_candidate_events"],
            "denominator": power_summary["n_samples_full_minute_grid"],
            "value": power_summary["raw_candidate_events"],
            "status": "real_calculated",
        },
        {
            "component": "dwell_rule",
            "dataset": power_summary["dataset"],
            "source": power_summary["source"],
            "formula": power_summary["dwell_rule"],
            "numerator": power_summary["candidate_after_dwell_events"],
            "denominator": power_summary["raw_candidate_events"],
            "value": power_summary["candidate_after_dwell_events"],
            "status": "real_calculated",
        },
        {
            "component": "first_exit_retention",
            "dataset": power_summary["dataset"],
            "source": power_summary["source"],
            "formula": "accepted_retained_events / candidate_after_dwell_events",
            "numerator": power_summary["accepted_retained_events"],
            "denominator": power_summary["candidate_after_dwell_events"],
            "value": power_summary["retention_fraction"],
            "status": "real_calculated",
        },
        {
            "component": "cost_normalized_vste",
            "dataset": power_summary["dataset"],
            "source": power_summary["source"],
            "formula": "accepted_retained_events / total_observed_energy_kwh",
            "numerator": power_summary["accepted_retained_events"],
            "denominator": power_summary["total_observed_energy_kwh"],
            "value": power_summary["cost_normalized_vste_events_per_observed_kwh"],
            "status": "real_calculated_observed_denominator_incomplete",
        },
        {
            "component": "missingness_and_denominator_completeness",
            "dataset": power_summary["dataset"],
            "source": power_summary["source"],
            "formula": "missing one-minute samples / full one-minute grid",
            "numerator": power_summary["n_samples_missing"],
            "denominator": power_summary["n_samples_full_minute_grid"],
            "value": power_summary["missing_fraction"],
            "status": "real_calculated_denominator_flagged",
        },
        {
            "component": "cost_normalized_vste_missing_adjusted_sensitivity",
            "dataset": power_summary["dataset"],
            "source": power_summary["source"],
            "formula": "accepted_retained_events / (observed_energy + missing_minutes * observed_median_power / 60)",
            "numerator": power_summary["accepted_retained_events"],
            "denominator": power_summary["estimated_total_energy_kwh"],
            "value": power_summary["cost_normalized_vste_events_per_estimated_kwh"],
            "status": "real_calculated_sensitivity_denominator_estimated",
        },
        {
            "component": "descriptive_treated_vs_psid_mean_outcome_difference",
            "dataset": lalonde_summary["dataset"],
            "source": lalonde_summary["source"],
            "formula": "mean(re78 | treated) - mean(re78 | PSID comparison)",
            "numerator": lalonde_summary["difference_in_means_re78"],
            "denominator": "USD outcome scale",
            "value": lalonde_summary["difference_in_means_re78"],
            "status": "real_calculated",
        },
        {
            "component": "descriptive_treated_vs_psid_transition_fraction_difference",
            "dataset": lalonde_summary["dataset"],
            "source": lalonde_summary["source"],
            "formula": "Pr(re78 > re75 | treated) - Pr(re78 > re75 | PSID comparison)",
            "numerator": lalonde_summary["treated_positive_transition_fraction_re78_gt_re75"],
            "denominator": lalonde_summary["psid_comparison_positive_transition_fraction_re78_gt_re75"],
            "value": lalonde_summary["positive_transition_fraction_difference"],
            "status": "real_calculated",
        },
    ]
    write_csv(OUT_COMPONENTS, component_rows)

    print("POWER")
    for key, value in power_summary.items():
        print(f"{key}: {value}")
    print("LALONDE")
    for key, value in lalonde_summary.items():
        print(f"{key}: {value}")
    print(f"wrote: {OUT_COMPONENTS}")
    print(f"wrote: {OUT_POWER_EVENTS}")
    print(f"wrote: {OUT_LALONDE}")


if __name__ == "__main__":
    main()
