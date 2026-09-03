# R-Universe Evidence-Audited Manuscript

This repository contains the evidence-audited manuscript package for:

**Relational Capacity Dynamics, Relational Foliation Gravity, and RFG-R**

Canonical repository URL:

<https://github.com/MartinPetrasek123/R_Universe_Evidence_Audited>

The current manuscript source includes the fully converged production
Bayesian-evidence comparison between the covariant KGB R-Universe realization
and a local LambdaCDM reference posterior.

## Main Files

- `main.tex` - manuscript source with the evidence section and data-availability
  statement updated.
- `R_Universe_evidence_audited_20260830.pdf` - previous compiled manuscript
  snapshot. It predates the final converged evidence update; regenerate it from
  the authoritative `main.tex` before the next release.
- `references.bib`, `graf.tex`, `download*.png` - source bibliography and
  supporting manuscript assets.
- `EVIDENCE_MANIFEST.md` - SHA-256 manifest for the manuscript, PDF, figures,
  and evidence artifacts.

## Evidence Summary

The completed dynamic nested-sampling evidence records report:

| Model | logZ | logZ error | -2 logZ | Calls | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Covariant KGB R-Universe | -1250.3771730462715 | 0.4241603584574464 | 2500.754346092543 | 157132 | converged |
| Local LambdaCDM reference | -1250.9013480259352 | 0.41356920558342064 | 2501.8026960518705 | 151268 | converged |

With the convention that larger `logZ` is favored,

`Delta logZ = logZ_KGB - logZ_LCDM = +0.5241749796637123`.

The quadrature-combined numerical uncertainty is `0.5924115946651036`, and the
Bayes factor is `B_KGB,LambdaCDM = exp(Delta logZ) = 1.6890647606059777`.
Thus the point estimate weakly favors KGB under the declared priors and probe
set, but the preference is not decisive at the achieved numerical precision.

## Evidence Artifacts

KGB:

- `evidence/kgb/kgb_dynesty_materialized_production.json`
- `evidence/kgb/kgb_dynesty_materialized_production.log`
- `evidence/kgb/kgb_dynesty_materialized_production.pkl`

LCDM:

- `evidence/lcdm/lcdm_dynesty_materialized_production.json`
- `evidence/lcdm/lcdm_dynesty_materialized_production.log`
- `evidence/lcdm/lcdm_dynesty_materialized_production.pkl`

Comparison:

- `evidence/evidence_comparison.json`

Archive-transfer audit:

- `evidence/archive-transfer/archive_lcdm_chain_points_t7_batches.log`

The Planck likelihood distribution and other licensed upstream observational
likelihood packages are not redistributed here.

## Reproduction Notes

The JSON files are the quickest way to verify the reported evidence numbers.
The `.pkl` checkpoint files preserve the converged dynesty states used for the
reported evidence summaries. The `.log` files preserve the complete production
and checkpoint-continuation records.

Direct artifact links:

- KGB evidence directory:
  <https://github.com/MartinPetrasek123/R_Universe_Evidence_Audited/tree/main/evidence/kgb>
- LCDM evidence directory:
  <https://github.com/MartinPetrasek123/R_Universe_Evidence_Audited/tree/main/evidence/lcdm>
- Machine-checkable manifest:
  <https://github.com/MartinPetrasek123/R_Universe_Evidence_Audited/blob/main/EVIDENCE_MANIFEST.sha256>
- Human-readable evidence manifest:
  <https://github.com/MartinPetrasek123/R_Universe_Evidence_Audited/blob/main/EVIDENCE_MANIFEST.md>

Run this from the repository root to verify file integrity:

```sh
shasum -a 256 -c EVIDENCE_MANIFEST.sha256
```
