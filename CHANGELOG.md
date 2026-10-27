# Changelog

## 1.0.0rc1 (revised manuscript)

The repository now contains one implementation of the method of the revised paper, ported from the finalized
revision code without changing any numerical algorithm or convention (`tests/test_revision_equivalence.py`).

### Analysis (as in the revised paper)
- OpenFAST data processed at the output interval of the files (0.05 s); every load case is a separate segment, so
  that no delay window spans two cases.
- Identification: Hankel-DMD in the correlation domain (damping) and the response domain, SSI-COV and FDD on the
  same data; medians over d x r; per-OLC tracking; mode-shape check against the ElastoDyn kinematics; comparison
  with the OpenFAST linearisation.
- Virtual sensing: time-blocked leave-one-OLC-out validation with cross-fitted hyper-parameters; Hankel-DMD
  projection (Tikhonov) and its Kalman form; baselines gappy POD, static least squares and Wiener filter; forecast
  horizon; robustness tests.
- Experimental verification on the FOCAL Campaign IV basin tests (Section 5) with saved outputs.

### Verified OpenFAST linear reference
- Repository metadata and regeneration settings now reflect the verified Table 3 linearisation: 8 m/s at 100 s,
  nearly stationary rotor (0.1801 rpm), MAP++, SubDyn enabled, still water, `DT = 0.0125 s`, `DT_Out = 0.05 s`,
  and 36 linearisation positions. No analysis algorithm or reported numerical result was changed.
- The OpenFAST SSI-COV wording is aligned with the implemented selection (median over orders 20--80, no
  stabilisation criterion); the FOCAL stabilised SSI-COV procedure is unchanged.

### Repository
- New package layout `src/eftwin` (`hankel`, `dmd`, `estimators`, `model`, `validation`, `baselines/`, `io`,
  `preprocess`, `modal`, `plotting`, `studies/`); configurations `configs/openfast.yaml`, `configs/focal.yaml`;
  reproduction scripts in `scripts/openfast`, `scripts/focal`, `scripts/make_tables.py`, `scripts/make_figures.py`.
- Tests against the revision code, mathematical identities, configurations and table rebuilds; CI workflow.
- Documentation: `docs/method.md`, `docs/reproduce.md`, `docs/open_items.md`, `data/README.md`,
  `legacy/README.md`, `openfast/README.md`, `results/README.md`.
- Saved FOCAL outputs (`results/focal`) and the tables and figures of the revised paper (`results/reference`).

### Removed from the main branch (available in the tag `v0.2-original-submission`)
- The original-submission modules and scripts (`src/eftwin/*` of v0.2, `scripts/analysis`, `scripts/openfast`),
  their configurations, documentation, notebook, sample data and smoke test, `requirements.txt` (dependencies are
  in `pyproject.toml`), `assets/digital_twin_demo.mp4` (video of the original version) and
  `assets/figure_4_virtual_sensing.png` (schematic of the original estimator; `docs/open_items.md`).
- The original-submission scripts themselves remain in `legacy/` for traceability.

## 0.2 (original submission)
Tagged `v0.2-original-submission`.
