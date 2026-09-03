# Evidence Manifest

Generated for the final converged evidence update to the R-Universe manuscript
package on 2026-09-03.

The manuscript source and numerical evidence artifacts were refreshed after
both dynamic nested-sampling calculations crossed the declared
`dlogz <= 0.3` threshold. The bundled PDF is the previous rendered snapshot and
must be regenerated from `main.tex` for the next release.

Canonical repository URL:

<https://github.com/MartinPetrasek123/R_Universe_Evidence_Audited>

The production evidence values included in the manuscript are:

- KGB: `logZ = -1250.3771730462715`, `logZerr = 0.4241603584574464`,
  `ncall = 157132`, `achieved_dlogz = 0.298987164967541`, `status = converged`
- Local LambdaCDM: `logZ = -1250.9013480259352`,
  `logZerr = 0.41356920558342064`, `ncall = 151268`,
  `achieved_dlogz = 0.29912794572777185`, `status = converged`
- `Delta logZ = logZ_KGB - logZ_LCDM = +0.5241749796637123`
- Quadrature-combined numerical uncertainty: `0.5924115946651036`
- `B_KGB,LambdaCDM = exp(Delta logZ) = 1.6890647606059777`

Larger `logZ` is favored, so the point estimate weakly favors KGB for the
declared priors and probe set. The difference is smaller than the combined
numerical uncertainty and is therefore not a decisive model preference.

The directly reported records are accompanied by
`evidence/evidence_comparison.json`, which stores the derived comparison in a
machine-readable form.

See `EVIDENCE_MANIFEST.sha256` for machine-checkable SHA-256 checksums.
