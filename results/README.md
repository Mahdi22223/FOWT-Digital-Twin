# Results

| Folder | Content | Tracked |
|---|---|---|
| `reference/tables`, `reference/figures` | Tables and figures of the revised paper, as submitted (for comparison with regenerated outputs) | yes |
| `focal/` | Outputs of the FOCAL analysis from the revision code (CSV, NPZ): Tables 6-7, Fig. 14 and the numbers of Section 5 | yes |
| `openfast/` | Outputs of `scripts/openfast/*` (CSV, NPZ); empty until the OpenFAST study is rerun (`docs/open_items.md`, item 1) | small files |
| `*/intermediate/` | Per-fold records and time series (pickle/NPZ, up to about 70 MB) | no |
| `tables/`, `figures/` | Written by `scripts/make_tables.py` and `scripts/make_figures.py` | no |

`docs/reproduce.md` lists which script writes which file and where each table and figure of the paper comes from.

## `focal/`

| File | Content | Paper |
|---|---|---|
| `focal_sysid_grid.csv` | correlation-domain Hankel-DMD per (d, r, mode), pooled and per run | Table 6 |
| `focal_sysid_summary.csv` | FDD, SSI-COV, Hankel-DMD (median, IQR, run range), MAC/MPC; platform roll and pitch | Table 6, Section 5.2 |
| `focal_resp_dmd_grid.csv`, `focal_resp_dmd_summary.csv` | response-domain Hankel-DMD | Table 6 |
| `focal_rotor_check.csv` | poles in the lowest and highest rotor-speed tercile | Section 5.2 |
| `focal_shapes.npz` | de-standardised shapes of the first tower modes and 3P (DMD, SSI, FDD) | Section 5.2 |
| `wideband_<DIR>.npz` | first singular value 0.2-11 Hz (all blocks and rotor-speed terciles) | Fig. 14 |
| `wide_poles_<DIR>.npz` | wide-band poles of Hankel-DMD and SSI-COV | Section 5.2 |
| `focal_vs_hyper.csv`, `focal_vs_per_run.csv`, `focal_vs_summary.csv` | cross-fitted hyper-parameters and R^2/NRMSE of the virtual sensing | Table 7 |
| `focal_coherence_bound.csv` | coherence bound and spectral composition | Table 7, Section 5.3 |
