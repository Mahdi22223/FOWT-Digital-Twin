# Data description

The Hankel-DMD analysis pipeline expects twelve OpenFAST-derived CSV
files, named sequentially as `Case_1.csv`, `Case_2.csv`, and so on
through `Case_12.csv`. Each case is the time-series export from a
single OpenFAST simulation of the NREL 5 MW OC3 Hywind spar floating
wind turbine under a specific wind-wave loading scenario. The full
list of scenarios is defined in `configs/openfast_cases.yaml` and
sweeps significant wave height, peak period, and wave direction across
the operational envelope described in the accompanying paper.

## Required columns

Every case file must contain a time column, eighteen tower acceleration
channels, and eighteen tower bending moment channels. The time column
is named `Time_[s]`. The fore-aft acceleration channels are
`TwHt1ALxt_[m/s^2]` through `TwHt9ALxt_[m/s^2]`, the side-to-side
acceleration channels are `TwHt1ALyt_[m/s^2]` through
`TwHt9ALyt_[m/s^2]`, the side-to-side bending moments (used for fore-
aft analysis under standard sign conventions) are `TwHt1MLyt_[kN-m]`
through `TwHt9MLyt_[kN-m]`, and the fore-aft bending moments (used for
side-to-side analysis) are `TwHt1MLxt_[kN-m]` through
`TwHt9MLxt_[kN-m]`.

Some OpenFAST exports produce duplicate column names for
`TwHt1MLxt_[kN-m]` and `TwHt1MLyt_[kN-m]` because the underlying
request list places the channel into both ElastoDyn and SubDyn output
streams. The cleaned loader in `src/eftwin/data_io.py:resolve_columns`
handles this case robustly by performing substring matching rather
than relying on column-name uniqueness, and prefers exact matches when
they are available. As a result, the pipeline does not require any
manual cleanup of the OpenFAST exports.

## Directional convention

The fore-aft analysis pairs the nine acceleration channels along the
local-x axis (`TwHtNALxt`) with the nine bending moments about the
local-y axis (`TwHtNMLyt`). Mechanically, this corresponds to
translations in fore-aft producing bending about the side-to-side
axis. Conversely, the side-to-side analysis pairs `TwHtNALyt` with
`TwHtNMLxt`. Both directions are handled symmetrically by passing the
corresponding column lists to `run_hankel_dmd`; the helper
`analysis_columns` in `src/eftwin/constants.py` returns the correct
lists given a direction keyword.

## Sampling rate

The sampling rate of the exported `Case_<N>.csv` files is determined
by whatever value of OpenFAST's `DT_Out` was used when the simulations
were run. The shipped configuration `configs/openfast_cases.yaml`
sets `DT_Out` equal to the simulation `DT` so that the exported files
match the simulation rate, but if a user runs OpenFAST with a different
`DT_Out`, the resulting CSV files will have whatever spacing OpenFAST
wrote, and this repository does not assume one specific value.

The analysis pipeline does have a configured time step `dt` (default
`0.0125` s in the YAML files under `configs/`) that the band-pass
filter and the Hankel-DMD eigenvalue mapping rely on for correctness.
To prevent silent inconsistencies between the configured `dt` and the
spacing actually present in the files, the loader in
`src/eftwin/data_io.py:check_dt_consistency` reads the median spacing
of the `Time_[s]` column on every load and raises a `DTMismatchError`
if it disagrees with the configured value by more than 0.0001 s. If
this error fires, you have two valid responses. You can re-export the
cases with `DT_Out = 0.0125` so that the exports match the analysis
configuration, or you can set the `dt` field in your analysis YAML
files to whatever value the loader reports as observed in your CSVs.
The safer option is the first, because it keeps the exports aligned
with the filter design that has been used throughout the paper.

The Butterworth band-pass filter applied during loading is fourth-
order with cutoffs at 0.25 Hz and 5.0 Hz; these cutoffs are designed
relative to the sampling frequency `1 / dt`, so changing `dt` without
also rechecking the cutoffs is not advised.

## Why the full data is not committed

The complete twelve-case dataset is approximately 500 MB once stored
in the CSV format expected by the pipeline. GitHub's hard limit on
individual file size is 100 MB and the recommended repository size is
well under 1 GB, so committing the full data directly is not
appropriate. This repository takes the simplest possible position on
the question: the full data is not committed in any form, the
`.gitignore` excludes `data/full/` from version control, and the
repository does not configure Git LFS for this content. Users have
two ways to obtain the dataset. They can run the OpenFAST pipeline
locally as documented in `docs/openfast_pipeline.md`, in which case
the resulting CSVs land under `data/full/` and stay there outside the
Git history. Or they can obtain a published dataset from a separate
archive (Zenodo, Figshare, or institutional storage) and place the
files under `data/full/` themselves. The author intends to deposit a
DOI-bearing copy on Zenodo after the arXiv preprint stabilises; if
that has happened by the time you read this, the DOI will be
documented in the project README.

## Wind input regeneration

The OpenFAST configuration uses turbulent wind input files generated
by TurbSim, the IEC-Kaimal turbulence generator distributed alongside
OpenFAST. The original twelve-case workflow used a turbulent wind
binary that is approximately 70 MB and is therefore not in scope for
this repository regardless of how it is regenerated. Users who want
to regenerate it should run TurbSim against the input deck
`90m_12mps_twr.inp` from the NREL 5 MW baseline distribution that
they have placed locally under `openfast/baseline/5MW_Baseline/Wind/`
following the instructions in `openfast/baseline/README.md`. The
typical invocation is `turbsim 90m_12mps_twr.inp`, which produces a
`.bts` binary in the same directory. Rename or symlink it to whatever
filename the OpenFAST `InflowWind` input deck expects, or update the
input deck to match the filename you produced. Steady-wind input
files included in the NREL baseline distribution can be used for runs
that do not require a turbulent inflow, including the linearisation
workflow.

## Synthetic sample data

A small synthetic CSV is shipped under
`data/sample/sample_case_small.csv` for smoke-testing the data
loaders, the band-pass filter, and the Hankel-DMD fit. The synthetic
data is a deliberately simple multi-tone signal sampled at
`dt = 0.0125` s and does not reproduce the complex aero-hydro-servo-
elastic response of the floating turbine. Do not interpret any modal
frequency, mode shape, or virtual sensing accuracy obtained on the
synthetic data as a paper-relevant result; the synthetic data exists
only to verify that the pipeline runs end-to-end without error and
that the loader's `dt` consistency check accepts a correctly-spaced
file.
