# Code inventory

This document maps every original research script to its cleaned
counterpart in the reusable package. The intent is to provide a clear
trail of how the methodology has been organised, so that a reviewer or
new user can locate any specific algorithm in the codebase quickly.

The cleaned modules preserve the original numerical workflow, the
parameter defaults, and the algorithmic sequence of the original
scripts. The original scripts are retained under `legacy/` for
traceability rather than being treated as the canonical source for
downstream development. Two of the original scripts had syntax issues
that prevented them from being imported as Python modules without
modification; in those two specific cases the cleaned `legacy/` copies
contain a small comment-only patch that wraps the broken lines so the
files now parse, with the original broken text preserved verbatim
inside the comment. The exact patches are described in the relevant
entries below.

## OpenFAST automation package

The original OpenFAST automation lived in five Python files at the
root of the `OpenFast Codes` archive. They are preserved unchanged
under `legacy/openfast_original/` (with the single comment-only patch
to `Linearize_ParametricInputs.py` noted below) and their cleaned
wrappers live in `src/eftwin/openfast_pipeline.py`, called from
`scripts/openfast/`.

The primary nonlinear case-generation script is `Sensor_Tower.py`,
which defines the twelve wind-wave load cases, the tower sensor
output list, and the OpenFAST batch execution. Its parameter-
generation logic is reproduced in `build_nonlinear_params` in
`openfast_pipeline.py`, and the cleaned runner is
`scripts/openfast/generate_nonlinear_cases.py`.

The linearisation-case script is `Linearize_ParametricInputs.py`. The
original contained a non-Python placeholder line in its post-
processing section
(`avg_results = postpro.averagePostPro(outFiles, *** Moda shapes for tower ***)`)
that caused a `SyntaxError` on import. This is one of the two
documented patches: the broken line and the three subsequent lines
that depended on it have been commented out in the `legacy/` copy
while the original broken text is preserved verbatim inside the
comment. No other line of the file has been modified, so the
case-generation logic and the OpenFAST parameter values are exactly
what the original author wrote. The cleaned linearisation runner is
`scripts/openfast/generate_linear_cases.py`, which uses
`build_linear_params` in `openfast_pipeline.py`.

`Data_for_DmD.py` performs the bridge from OpenFAST `.outb` files to
the DMD-ready `Case_*.csv` files. Its core extraction logic is
reproduced in `extract_for_dmd` and `export_dmd_csv_batch` in
`openfast_pipeline.py`, called via `scripts/openfast/export_dmd_csv.py`.

`Modal_OutPut_Final_02.py` reads OpenFAST `.lin` linearisation files,
extracts tower and blade mode shapes, and writes them to
`Extracted_Mode_Shapes.xlsx`. Because this script depends heavily on
specific `pyFAST` and `openfast_toolbox` installation details, the
cleaned scaffold does not refactor its body; instead, the wrapper
`scripts/openfast/extract_linear_modes.py` imports the legacy
script's `analyze_modes` function and exposes it through a proper
command-line interface that accepts the input `.lin` file path and the
desired output Excel path as arguments. This replaces the legacy
script's hardcoded `outputs_lin_No_SubD/parametric/case_1.1.lin` path
with proper CLI handling.

`Data_ML.py` is a pyDatView-generated quick plotting utility that the
original author used during exploratory analysis. It is preserved as
legacy support but has no cleaned counterpart since the same
visualisations are available through the analysis pipeline plotters.

## Hankel-DMD analysis package

The original Hankel-DMD analysis package contained ten Python files.
They are preserved unchanged under `legacy/hankel_dmd_original/` (with
the single comment-only patch to `Missing_Sensor_Generalization_2 2.py`
noted below) and their cleaned counterparts are distributed across the
`src/eftwin/` package.

### Identification: Filter_Hankel_03.py

The primary identification script is `Filter_Hankel_03.py`. Its core
fit routine — load CSVs, validate `dt`, apply zero-phase Butterworth
band-pass filter, scale, fit `HankelDMD(svd_rank=24, tlsq_rank=0,
exact=True, opt=False, d=60)` — is reproduced in
`src/eftwin/hankel_dmd.py:run_hankel_dmd`. The eigenvalue stability
plot, SVD rank-stability diagnostic, physical mode-shape extraction,
and dynamics plot are split into `src/eftwin/modal_analysis.py` and
`src/eftwin/plotting.py`. The cleaned runner is
`scripts/analysis/run_hankel_dmd.py`.

### Virtual sensing (in-sample): Virtual_sensing 2.py

`Virtual_sensing 2.py` implements the in-sample variant of the rolling-
horizon virtual sensing algorithm: identify which Hankel rows
correspond to available sensors, compute a sparse pseudoinverse, and
use the Vandermonde-based mode-amplitude calibration to reconstruct
the missing channels from a sliding window of the available channels.
This is reproduced in `src/eftwin/virtual_sensing.py:run_virtual_sensing`
and called from `scripts/analysis/run_virtual_sensing.py`.

### Missing/failed sensor reconstruction: Missing_Sensor_Generalization_2 2.py

`Missing_Sensor_Generalization_2 2.py` implements the cross-case
missing-sensor reconstruction experiment, which is the central
practical contribution of the paper. Tower sensor channels are masked
from a set of test cases that the Hankel-DMD model has not seen during
training, and the model reconstructs them using only the remaining
channels. The reconstruction accuracy reported per masked sensor is
the result that supports the paper's claim that the equation-free
digital twin can recover failed or never-installed sensors at
operational sampling rates with R squared above 0.99 across unseen
wave-load conditions. The methodology comprises three stages:
training a Hankel-DMD model with `svd_rank = 34` on the first eight of
twelve cases, masking the user-selected sensor indices from the
remaining four cases, and rolling the trained model forward through
the test data to reconstruct the masked channels.

This script's functionality is mapped to the cleaned codebase as
follows. The training stage lives in
`src/eftwin/virtual_sensing.py:train_generalization_model`. The
prediction stage lives in `run_generalization_prediction`. A unified
orchestrator that calls both stages in sequence and returns a single
result dictionary is `run_missing_sensor_generalization`. The
runnable command-line script is
`scripts/analysis/run_missing_sensor_generalization.py`. Its YAML
configuration is `configs/missing_sensor_generalization.yaml`. The
tutorial notebook section that walks through the experiment
interactively is in `notebooks/01_hankel_dmd_tutorial.ipynb`. The
methodology is described in detail in `docs/methodology.md` under the
section "Virtual sensing of failed and missing tower channels".

This is the second documented patched file. The original ended with a
markdown code-fence (triple backticks) followed by a numbered "How to
use this" section. Markdown is valid documentation but invalid Python,
and so the original would not parse as a module without modification.
The `legacy/` copy comments out the offending lines and preserves the
original markdown content inside the comment for provenance. No other
line of the file has been modified.

### SSI-COV cross-validation: SSICOV_2.py

`SSICOV_2.py` implements covariance-driven Stochastic Subspace
Identification at a fixed model order. The cleaned counterpart is the
`SSICOV` class in `src/eftwin/ssi_cov.py` and the `run_ssi_validation`
driver. The runner is `scripts/analysis/run_ssi_cov.py`.

### SSI stabilisation: SSI_COV_Stablizing.py

`SSI_COV_Stablizing.py` builds the stabilisation diagram by sweeping
the SSI model order and tracking which poles persist. The cleaned
counterpart is the `SSIMultiOrder` class plus `run_stabilization` in
`src/eftwin/ssi_cov.py`. The runner is
`scripts/analysis/run_ssi_stabilization.py`.

### Noise-floor analysis: Noise_Floor_Analysis.py

`Noise_Floor_Analysis.py` constructs the explicit Hankel block matrix,
performs randomized SVD, and applies the Gavish-Donoho threshold to
estimate the optimal rank. This is reproduced in
`src/eftwin/noise_lyapunov.py:hankel_singular_values`.

### Probabilistic mode-shape sweep: Probablistic_Mode_shape 2.py

`Probablistic_Mode_shape 2.py` implements the Hankel-delay sensitivity
sweep (`d` from 10 to 90) that produces the probabilistic mode shape
with 95 % confidence bands. The original used `svd_rank = 35` for this
sweep specifically, which differs from the identification rank (24)
and the missing-sensor generalisation rank (34); this distinction is
preserved in the cleaned module. The cleaned counterpart is
`src/eftwin/probabilistic_modes.py:run_probabilistic_mode_sweep`,
called from `scripts/analysis/run_probabilistic_modes.py`.

### MAC validation: MAC_Final.py

`MAC_Final.py` implements both the standard complex MAC and the real-
projected MAC used for validating identified mode shapes against the
OpenFAST linearisation modes. The numerical formulas are preserved in
`src/eftwin/modal_analysis.py:complex_mac` and `real_projected_mac`.
The original script depended on workspace variables (`df_SSI_FA`,
`df_SSI_SS`, `df_fa_modes`, `df_ss_modes`) that exist only after
running other scripts interactively, so the cleaned scaffold provides
`scripts/analysis/run_mac_validation.py`, which loads the persisted
artefacts produced by `run_hankel_dmd.py` and `run_ssi_cov.py`
together with the `Extracted_Mode_Shapes.xlsx` produced by
`Modal_OutPut_Final_02.py`.

### OpenFAST mode-shape plotter: OpenFast_Mode_Shapes.py

`OpenFast_Mode_Shapes.py` is a plotting utility that reads
`Extracted_Mode_Shapes.xlsx` and visualises the OpenFAST linear mode
shapes. It can be run directly from `legacy/`; its plotting layout is
also represented inside `run_mac_validation.py` and the tutorial
notebook.

### Lyapunov exponent: Lyapunov_Exponent_Estimation.py

`Lyapunov_Exponent_Estimation.py` implements the Rosenstein algorithm
for estimating the largest Lyapunov exponent and the predictability
horizon from a phase-space embedding of the top-tower acceleration.
This is reproduced in `src/eftwin/noise_lyapunov.py:embed_series` and
`calculate_lyapunov`.

## Integration notes

The refactored package preserves the original equations and parameter
defaults. Specifically: the time step is `dt = 0.0125 s`
(corresponding to a sampling frequency of 80 Hz); the band-pass filter
is fourth-order Butterworth with cutoffs at 0.25 Hz and 5.0 Hz; the
Hankel delay is `d = 60`; the Hankel-DMD rank is 24 for modal
identification, 34 for the missing-sensor generalisation experiment,
and 35 for the probabilistic mode sweep; SSI-COV uses 60 block rows,
model order 40 for fixed-order identification, and orders 2 to 60 for
stabilisation; downsampling for SSI-COV is by a factor of 4; the
Lyapunov embedding dimension is 10 with time delay 10. All of these
values are encoded in the YAML files under `configs/` and were
verified against the originals.

Configuration values previously hardcoded in scripts have been moved
into `configs/*.yaml` so that users do not have to edit Python files
to change analysis parameters. Large simulation outputs, full CSV
cases, NREL distribution baseline files, OpenFAST executables and
controller libraries, and turbulent wind binaries are excluded from
version control through `.gitignore`. The repository deliberately
does not use Git LFS because no in-scope artefact is large enough to
warrant it; the policy is that anything large enough for LFS is also
something the repository is not the right place for.
