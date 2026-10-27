# Reproducing the paper

All commands are run from the repository root after `pip install -e .`. Settings are read from
`configs/openfast.yaml` and `configs/focal.yaml`; every value there is the one used for the reported results.
Outputs go to `results/openfast`, `results/focal`, `results/tables` and `results/figures`; the tables and figures
of the revised paper are in `results/reference` for comparison. Table and figure numbers follow the response
letter; the LaTeX labels are given because the numbering may change in the final layout.

## What runs without the raw data

| Item | Command | Inputs (in the repository) |
|---|---|---|
| Table 6 (`tab:focal_sysid`) | `python scripts/make_tables.py focal_sysid` | `results/focal/focal_sysid_summary.csv`, `focal_resp_dmd_summary.csv` |
| Table 7 (`tab:focal_vs`) | `python scripts/make_tables.py focal_vs` | `results/focal/focal_vs_summary.csv`, `focal_coherence_bound.csv` |
| Fig. 14 (`fig:focal_id`) | `python scripts/make_figures.py fig14` | `results/focal/wideband_*.npz`, `focal_sysid_summary.csv` |
| Tests, quickstart, online cost | `python -m pytest`, `python examples/quickstart.py`, `python scripts/supplementary/online_cost.py` | synthetic data |

Tables 6-7 rebuild byte-identically (checked by `tests/test_configs_and_tables.py`) and Fig. 14 pixel-identically
(matplotlib 3.11) from the saved outputs. Everything else needs the data described in `data/README.md`.

## OpenFAST study (Sections 3-4): `bash scripts/run_openfast.sh`

Data: `Case_1.csv` ... `Case_12.csv` (`paths.csv_dir` or `EFTWIN_OPENFAST_DIR`) and `case_1.1.lin`
(`paths.lin_file` or `EFTWIN_LIN_FILE`). The `.lin` file is the authoritative linear reference: `t = 100 s`,
8 m/s steady inflow, still water, MAP++, SubDyn enabled (`CompSub = 1`), rotor speed 0.1801 rpm, `DT = 0.0125 s`,
`DT_Out = 0.05 s`, with 36 linearisation positions defined by the deck. The historical InflowWind filename contains
`Steady12mps`, but the generated `.lin`/`.out` confirm 8 m/s. The run order and outputs (in `results/openfast/`):

| Step | Script | Outputs | Paper |
|---|---|---|---|
| 1 | `openfast/identify.py` | `sysid_sensitivity.csv`, `corr_dmd_sensitivity.csv`, `corr_dmd_summary.csv`, `table3_revised.csv`, `sysid_per_olc.csv`, `modeshapes.npz` | Table 3, Fig. A1, Section 4.1-4.2 |
| 2 | `openfast/shape_check.py` | `shape_check.csv` | Table A1 (`tab:shape_check`) |
| 3 | `openfast/linear_reference.py` | `lin_modes.csv` | Table 3, column Linear |
| 4 | `openfast/virtual_sensing.py FA`, `... SS` | `intermediate/blocked_records_<DIR>.pkl`, Gram caches in `cache/openfast` | Sections 4.4-4.8 |
| 5 | `openfast/summarise.py` | `blocked_hyper.csv`, `blocked_selected_olc.csv`, `blocked_summary_random.csv`, `blocked_cases_per_sensor.csv`, `blocked_layouts.csv`, `blocked_forecast.csv`, `blocked_L1L2_per_channel.csv` | Table 5, A2, A3; Figs. 11-12 |
| 6 | `openfast/robustness.py FA`, `... SS` | `blocked_robust_<DIR>.csv` | Table A4 (`tab:vs_robust`) |
| 7 | `openfast/excerpts.py` | `vs_excerpts.npy` | Figs. 8 and 13 |
| 8 | `make_tables.py vs_results l1_channels robust shape_check` | `results/tables/*.tex` | Tables 5, A1-A4 |
| 9 | `make_figures.py openfast` | `results/figures/fig5_*` ... `fig13_*`, `figA_*` | Figs. 5-13, A1 |

The virtual-sensing step (4) dominates the run time. Steps 5 and 8 only read saved outputs, so the tables can be
rebuilt without repeating the analysis; steps 6, 7 and 9 also read the data (Gram sums are cached).

**Table 3** (`tab:sys_id_comparison`) is assembled from three files: FDD, SSI-COV, response-domain Hankel-DMD
(median [IQR]) and the DMD-SSI MACs from `table3_revised.csv`; the correlation-domain Hankel-DMD column from
`corr_dmd_summary.csv`; the linear column and the DMD-linear MACs (`mac_acc_dmd`, `mac_mom_dmd`) from
`lin_modes.csv`.

Table 4 (`tab:vs_scenarios`) lists the layouts of `virtual_sensing.layouts` and `virtual_sensing.scenarios`.
Figs. 5, 6, 7, 9 and 10 are computed from the data inside `make_figures.py`; Figs. 8 and 11-13 and A1 only read saved outputs.

## FOCAL Campaign IV study (Section 5): `bash scripts/run_focal.sh`

Data: the four `FOCAL_DLC12_*.csv` files (`paths.csv_dir` or `EFTWIN_FOCAL_DIR`). Outputs in `results/focal/`:

| Step | Script | Outputs | Paper |
|---|---|---|---|
| 1 | `focal/identify.py` | `focal_sysid_grid.csv`, `focal_sysid_summary.csv`, `focal_shapes.npz` | Table 6 |
| 2 | `focal/response_dmd.py` | `focal_resp_dmd_grid.csv`, `focal_resp_dmd_summary.csv` | Table 6, column resp. |
| 3 | `focal/rotor_check.py` | `focal_rotor_check.csv` | Section 5.2 |
| 4 | `focal/wideband.py all` | `wideband_<DIR>.npz`, `wide_poles_<DIR>.npz` (+ printed candidate and first-mode tables) | Fig. 14, Section 5.2 |
| 5 | `focal/sensor_kinematics.py constrained` (and `free`) | printed | Section 5.1 |
| 6 | `focal/virtual_sensing.py FA`, `... SS` | `intermediate/focal_vs_records_<DIR>.pkl` | Section 5.3 |
| 7 | `focal/summarise.py` | `focal_vs_hyper.csv`, `focal_vs_per_run.csv`, `focal_vs_summary.csv` | Table 7 |
| 8 | `focal/coherence_bound.py` | `focal_coherence_bound.csv` | Table 7, bound |
| 9 | `focal/series.py` | `intermediate/vs_series_<DIR>.npz` | Figs. 15-16 |
| 10 | `make_tables.py focal_sysid focal_vs`, `make_figures.py focal` | `results/tables`, `results/figures` | Tables 6-7, Figs. 14-16 |

The saved files in `results/focal/` are the outputs of the revision code that produced Section 5.

## Supplementary scripts

| Script | Purpose |
|---|---|
| `supplementary/legacy_dt_check.py` | time step of the case files and the frequencies the original-submission pipeline reports for them |
| `supplementary/lyapunov_surrogate.py` | surrogate-data test of the Lyapunov-exponent estimate (response to Reviewer 4; `pip install -e .[supplementary]`) |
| `supplementary/online_cost.py` | online cost per sample of the projection and the Kalman form (Section 2.4.3) |
| `simulation/openfast_cases.py` | regenerate the OpenFAST input files and export `Case_<n>.csv` (see `openfast/README.md`) |

## Equivalence with the revision code

The scripts reproduce the finalized revision code (the code behind the reported numbers). The library functions are
checked against outputs of that code (`tests/test_revision_equivalence.py`). In addition, both complete pipelines
(all steps above, both studies) were run side by side with the revision scripts on synthetic data in the format of
the OpenFAST and FOCAL files (shorter records, set in the configurations; algorithms unchanged): all 59 outputs were
identical (CSV values, NPZ arrays and per-fold records exactly, LaTeX tables byte for byte, figures pixel for pixel),
as were the printed results of the scripts without file output. Conventions that differ between the two studies are documented in `docs/method.md`.
