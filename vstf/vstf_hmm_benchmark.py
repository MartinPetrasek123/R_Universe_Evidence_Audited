#!/usr/bin/env python3
"""Hidden-Markov benchmark for the Valid State Transition Framework.

The benchmark is intentionally simple and reproducible.  It compares two
synthetic systems with similar raw threshold activity but different retention,
domain validity, and full-boundary cost.  The point is not biological or
hardware realism; it is a worked demonstration that the VSTF acceptance layer
can change ranking relative to raw activity, crossings, and first-passage
counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import random
from statistics import mean

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


SEED = 20260917
N_TRAJ = 10_000
T = 72
THETA_CROSS = 0.72
POST_COMMIT = 0.65
POST_VALIDATE = 0.80
VALIDATION_LAG = 2
TAU_RET = 6
POST_RET = 0.55
HYST_LOW = 0.35


@dataclass(frozen=True)
class SystemSpec:
    name: str
    p01: float
    p10: float
    sigma: float
    domain_valid_prob: float
    energy_base: float
    energy_per_crossing: float
    energy_per_committed: float
    energy_per_valid: float


SYSTEMS = [
    SystemSpec(
        name="System A: high activity, fragile retention",
        p01=0.125,
        p10=0.225,
        sigma=0.44,
        domain_valid_prob=0.72,
        energy_base=92.0,
        energy_per_crossing=1.15,
        energy_per_committed=2.30,
        energy_per_valid=3.20,
    ),
    SystemSpec(
        name="System B: moderate activity, retained outcomes",
        p01=0.024,
        p10=0.048,
        sigma=0.27,
        domain_valid_prob=0.88,
        energy_base=88.0,
        energy_per_crossing=0.95,
        energy_per_committed=1.45,
        energy_per_valid=2.10,
    ),
]


def simulate_hidden(spec: SystemSpec, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros(T, dtype=int)
    for t in range(1, T):
        if x[t - 1] == 0:
            x[t] = 1 if rng.random() < spec.p01 else 0
        else:
            x[t] = 0 if rng.random() < spec.p10 else 1
    y = rng.normal(loc=x.astype(float), scale=spec.sigma)
    return x, y


def filter_posterior(spec: SystemSpec, y: np.ndarray) -> np.ndarray:
    """Forward filtering for a two-state HMM with Gaussian emissions."""
    trans = np.array([[1 - spec.p01, spec.p01], [spec.p10, 1 - spec.p10]])
    prior = np.array([0.94, 0.06], dtype=float)
    post = np.zeros(T, dtype=float)
    alpha = prior.copy()
    for t, obs in enumerate(y):
        if t > 0:
            alpha = alpha @ trans
        like0 = math.exp(-0.5 * ((obs - 0.0) / spec.sigma) ** 2)
        like1 = math.exp(-0.5 * ((obs - 1.0) / spec.sigma) ** 2)
        alpha *= np.array([like0, like1])
        total = alpha.sum()
        if total <= 0:
            alpha = prior.copy()
        else:
            alpha /= total
        post[t] = alpha[1]
    return post


def count_raw_crossings(y: np.ndarray) -> int:
    return int(np.sum(y >= THETA_CROSS))


def crossing_events(y: np.ndarray) -> list[int]:
    events: list[int] = []
    armed = True
    for t, obs in enumerate(y):
        if armed and obs >= THETA_CROSS:
            events.append(t)
            armed = False
        elif not armed and obs <= HYST_LOW:
            armed = True
    return events


def evaluate_trajectory(spec: SystemSpec, rng: np.random.Generator) -> dict[str, int | float]:
    x, y = simulate_hidden(spec, rng)
    post = filter_posterior(spec, y)
    crossings = crossing_events(y)

    committed: list[int] = []
    retained: list[int] = []
    valid: list[int] = []
    pending = 0
    false_success = 0

    for t in crossings:
        if post[t] < POST_COMMIT:
            continue
        committed.append(t)
        v = t + VALIDATION_LAG
        if v >= T:
            pending += 1
            continue
        if post[v] < POST_VALIDATE:
            false_success += 1
            continue
        end = v + TAU_RET
        if end >= T:
            pending += 1
            continue
        if float(np.min(post[v:end + 1])) < POST_RET:
            false_success += 1
            continue
        retained.append(t)
        if rng.random() <= spec.domain_valid_prob:
            valid.append(t)
        else:
            false_success += 1

    raw = count_raw_crossings(y)
    energy = (
        spec.energy_base
        + spec.energy_per_crossing * len(crossings)
        + spec.energy_per_committed * len(committed)
        + spec.energy_per_valid * len(valid)
    )
    return {
        "raw_activity": raw,
        "crossings": len(crossings),
        "committed": len(committed),
        "retained": len(retained),
        "valid_vst": len(valid),
        "false_success": false_success,
        "pending": pending,
        "energy": energy,
    }


def summarize(results: list[dict[str, int | float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in results[0]:
        out[key] = float(sum(float(r[key]) for r in results))
    out["raw_per_energy"] = out["raw_activity"] / out["energy"]
    out["crossings_per_energy"] = out["crossings"] / out["energy"]
    out["vste"] = out["valid_vst"] / out["energy"]
    out["retention_yield"] = out["retained"] / max(out["committed"], 1.0)
    out["validity_yield"] = out["valid_vst"] / max(out["retained"], 1.0)
    out["false_success_fraction"] = out["false_success"] / max(out["committed"], 1.0)
    return out


def write_csv(path: Path, summaries: dict[str, dict[str, float]]) -> None:
    fields = [
        "system",
        "raw_activity",
        "crossings",
        "committed",
        "retained",
        "valid_vst",
        "false_success",
        "pending",
        "energy",
        "raw_per_energy",
        "crossings_per_energy",
        "vste",
        "retention_yield",
        "validity_yield",
        "false_success_fraction",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, row in summaries.items():
            writer.writerow({"system": name, **{k: row[k] for k in fields if k != "system"}})


def draw_bar(c: canvas.Canvas, x: float, y: float, width: float, height: float, frac: float, color) -> None:
    c.setFillColor(colors.whitesmoke)
    c.rect(x, y, width, height, fill=1, stroke=0)
    c.setFillColor(color)
    c.rect(x, y, width * max(0.0, min(1.0, frac)), height, fill=1, stroke=0)
    c.setStrokeColor(colors.black)
    c.rect(x, y, width, height, fill=0, stroke=1)


def write_pdf(path: Path, summaries: dict[str, dict[str, float]]) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    W, H = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(46, H - 48, "VSTF HMM Benchmark: activity is not outcome")
    c.setFont("Helvetica", 9)
    c.drawString(46, H - 64, f"{N_TRAJ:,} trajectories per system; seed={SEED}; validation lag={VALIDATION_LAG}; retention window={TAU_RET}")

    metrics = ["raw_activity", "crossings", "committed", "retained", "valid_vst"]
    labels = ["Raw", "Crossing", "Committed", "Retained", "Valid VST"]
    colors_ = [colors.HexColor("#5b8fd1"), colors.HexColor("#64b6ac"), colors.HexColor("#f0b35a"), colors.HexColor("#cb6f6f"), colors.HexColor("#6d5cae")]

    max_raw = max(s["raw_activity"] for s in summaries.values())
    y0 = H - 118
    c.setFont("Helvetica-Bold", 11)
    c.drawString(46, y0 + 30, "Nested event attrition")
    c.setFont("Helvetica", 8)
    for idx, (sys_name, summary) in enumerate(summaries.items()):
        y = y0 - idx * 150
        c.setFont("Helvetica-Bold", 9)
        c.drawString(46, y + 10, sys_name)
        for j, (m, lab) in enumerate(zip(metrics, labels)):
            yy = y - 18 - j * 20
            frac = summary[m] / max_raw
            draw_bar(c, 145, yy, 260, 11, frac, colors_[j])
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 8)
            c.drawString(46, yy + 2, lab)
            c.drawRightString(455, yy + 2, f"{summary[m]:,.0f}")

    y = 292
    c.setFont("Helvetica-Bold", 11)
    c.drawString(46, y, "Efficiency ranking")
    c.setFont("Helvetica", 8)
    c.drawString(46, y - 14, "Raw activity per energy and valid VST per energy are normalized to the best system.")
    max_raw_eff = max(s["raw_per_energy"] for s in summaries.values())
    max_vste = max(s["vste"] for s in summaries.values())
    for idx, (sys_name, summary) in enumerate(summaries.items()):
        yy = y - 46 - idx * 72
        c.setFont("Helvetica-Bold", 9)
        c.drawString(46, yy + 30, sys_name)
        c.setFont("Helvetica", 8)
        c.drawString(62, yy + 11, "Raw / energy")
        draw_bar(c, 150, yy + 8, 180, 10, summary["raw_per_energy"] / max_raw_eff, colors.HexColor("#5b8fd1"))
        c.drawRightString(385, yy + 10, f"{summary['raw_per_energy']:.3f}")
        c.drawString(62, yy - 9, "Valid VST / energy")
        draw_bar(c, 150, yy - 12, 180, 10, summary["vste"] / max_vste, colors.HexColor("#6d5cae"))
        c.drawRightString(385, yy - 10, f"{summary['vste']:.3f}")
        c.drawString(410, yy + 2, f"false-success fraction={summary['false_success_fraction']:.2%}")

    c.setFont("Helvetica-Bold", 10)
    c.drawString(46, 92, "Interpretation")
    c.setFont("Helvetica", 8)
    txt = c.beginText(46, 78)
    lines = [
        "System A produces more raw activity and crossings, but many events fail validation or retention and its cost boundary is higher.",
        "System B ranks lower on raw activity, but higher on retained valid VST yield per supplied energy.",
        "This is the incremental VSTF claim in miniature: the acceptance layer changes the scientific ranking.",
    ]
    for line in lines:
        txt.textLine(line)
    c.drawText(txt)
    c.showPage()
    c.save()


def main() -> None:
    rng = np.random.default_rng(SEED)
    summaries: dict[str, dict[str, float]] = {}
    for spec in SYSTEMS:
        results = [evaluate_trajectory(spec, rng) for _ in range(N_TRAJ)]
        summaries[spec.name] = summarize(results)

    out_dir = Path("/Users/mpetr/Desktop")
    write_csv(out_dir / "vstf_hmm_benchmark_summary.csv", summaries)
    write_pdf(out_dir / "vstf_hmm_benchmark.pdf", summaries)

    for name, row in summaries.items():
        print(name)
        for key in [
            "raw_activity",
            "crossings",
            "committed",
            "retained",
            "valid_vst",
            "false_success",
            "pending",
            "energy",
            "raw_per_energy",
            "vste",
            "retention_yield",
            "false_success_fraction",
        ]:
            print(f"  {key}: {row[key]:.6g}")


if __name__ == "__main__":
    main()
