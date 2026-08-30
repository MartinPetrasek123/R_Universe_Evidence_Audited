# R-Universe Evidence-Audited Manuscript

This repository contains the evidence-audited manuscript package for:

**Relational Capacity Dynamics, Relational Foliation Gravity, and RFG-R**

Canonical repository URL:

<https://github.com/MartinPetrasek123/R_Universe_Evidence_Audited>

The current manuscript includes the completed production Bayesian-evidence
comparison between the covariant KGB R-Universe realization and a local
LambdaCDM reference posterior.

## Main Files

- `main.tex` - manuscript source with the evidence section and data-availability
  statement updated.
- `R_Universe_evidence_audited_20260830.pdf` - compiled PDF corresponding to
  the updated manuscript.
- `references.bib`, `graf.tex`, `download*.png` - source bibliography and
  supporting manuscript assets.
- `EVIDENCE_MANIFEST.md` - SHA-256 manifest for the manuscript, PDF, figures,
  and evidence artifacts.

## Evidence Summary

The completed dynamic nested-sampling evidence records report:

| Model | logZ | logZ error | -2 logZ | Calls | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Covariant KGB R-Universe | -1335.599690469821 | 2.8042019114790784 | 2671.199380939642 | 20268 | completed |
| Local LambdaCDM reference | -1291.3199169376812 | 2.813517742757359 | 2582.6398338753625 | 20264 | completed |

With the convention that larger `logZ` is favored,

`Delta logZ = logZ_KGB - logZ_LCDM = -44.27977353213983`.

Thus this specific production marginal-likelihood comparison favors the local
LambdaCDM reference under the declared priors and probe set. The manuscript
therefore presents the KGB R-Universe realization as a physical candidate with a
completed posterior and evidence audit, not as an empirical evidence winner over
LambdaCDM.

## Evidence Artifacts

KGB:

- `evidence/kgb/kgb_dynesty_materialized_production.json`
- `evidence/kgb/kgb_dynesty_materialized_production.log`
- `evidence/kgb/kgb_dynesty_materialized_production.pkl`

LCDM:

- `evidence/lcdm/lcdm_dynesty_materialized_production.json`
- `evidence/lcdm/lcdm_dynesty_materialized_production.log`
- `evidence/lcdm/lcdm_dynesty_materialized_production.pkl`

Archive-transfer audit:

- `evidence/archive-transfer/archive_lcdm_chain_points_t7_batches.log`

The Planck likelihood distribution and other licensed upstream observational
likelihood packages are not redistributed here.

## Reproduction Notes

The JSON files are the quickest way to verify the reported evidence numbers.
The `.pkl` checkpoint files preserve the completed dynesty state used for the
reported evidence summaries. The `.log` files preserve the production run
records.

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
