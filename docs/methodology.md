# Methodology

This document describes the algorithms that the cleaned package
implements and explains where each algorithm lives in the codebase.
Its purpose is to make it straightforward for a reader to confirm that
the refactor preserves the original numerical workflow, parameter
defaults, and algorithmic sequence. The original research scripts
under `legacy/` remain available for traceability; this document
explains what each cleaned module corresponds to and why each
parameter takes the value it does.

## Signal preprocessing

Each tower channel is filtered with a fourth-order Butterworth band-pass
filter applied via `scipy.signal.filtfilt` for zero phase distortion.
The band is 0.25 Hz to 5.0 Hz, which captures the platform-rigid-body
modes and the tower-flexible modes while rejecting both the very-low-
frequency drift below the wave-energy band and the high-frequency
measurement noise above the structural response. The implementation
is in `src/eftwin/preprocessing.py:apply_zero_phase_filter` and is
identical to the helper of the same name in every original script.

## Sampling-rate validation

Before any filtering or fitting takes place, the data loader in
`src/eftwin/data_io.py` validates the sampling rate of every loaded
case against the configured `dt`. The check computes the median
difference between consecutive `Time_[s]` values in the file and
raises `DTMismatchError` if it disagrees with the configured value by
more than 0.0001 seconds. The reason this check exists is that the
filter is designed for a specific Nyquist frequency, the Hankel-DMD
eigenvalue mapping uses `dt` to recover continuous-time frequencies,
and the rolling-horizon virtual sensing computes its prediction window
in samples derived from `dt`. If the underlying CSVs were exported at
OpenFAST's `DT_Out` interval (which defaults to 0.05 s in the case
templates) instead of at the simulation `DT` interval (0.0125 s for
this study), every downstream calculation is silently scaled by an
incorrect factor of four. The validation catches that mistake at the
data-loading step rather than letting it propagate into the modal
identification.

## Hankel-DMD identification

The identification stacks nine acceleration channels with nine bending
moment channels along the rows of a state matrix `X`, with time
running along the columns. The state matrix is standardised with
`sklearn.preprocessing.StandardScaler` fitted on the transposed matrix
so that each channel has zero mean and unit variance. The Hankel-DMD
model is then fitted using the `pydmd.HankelDMD` class with delay
`d = 60` and exact DMD with `svd_rank = 24`, `tlsq_rank = 0`, and
`opt = False`. These hyperparameters match the original
`Filter_Hankel_03.py` exactly and are exposed as configuration fields
in `configs/hankel_dmd.yaml`.

The discrete-time eigenvalues of the Koopman approximation produced by
the fit are mapped to continuous-time frequencies and damping ratios
using the standard formula `omega = log(eigs) / dt`, with frequency
`f = |Im(omega)| / (2 * pi)` and damping ratio
`zeta = -Re(omega) / |omega|`. The modal extraction code lives in
`src/eftwin/modal_analysis.py:modal_parameters`.

A physical mode shape is recovered from the first eighteen entries of
each DMD mode (corresponding to one snapshot block). The first nine
entries are acceleration components and the next nine are bending-
moment components. Acceleration components are converted to
displacement components using the frequency-domain integration
relation `disp = -acc / omega**2`. The implementation is in
`acceleration_mode_to_displacement`.

## Virtual sensing of failed and missing tower channels

Virtual sensing is the central practical contribution of the paper and
the experimental result that demonstrates the equation-free digital
twin's value as a structural-monitoring tool. The methodology comes in
two flavours, both implemented in `src/eftwin/virtual_sensing.py`.

The in-sample variant, `run_virtual_sensing`, simulates the failure of
designated sensor channels on a dataset that the Hankel-DMD model has
already seen during identification, and demonstrates that the
remaining channels carry enough information to reconstruct the failed
ones. This is the methodology of the original `Virtual_sensing 2.py`
script. The cross-case variant,
`run_missing_sensor_generalization`, is the more demanding experiment:
the model is fitted on the first eight cases of the twelve-case wave-
load sweep, designated channels are masked from the four unseen test
cases, and the model reconstructs them under operating conditions it
has never trained on. This second experiment is the methodology of the
original `Missing_Sensor_Generalization_2 2.py` script and is the
result that supports the paper's headline claim that the digital twin
can recover failed sensors at operational sampling rates with R squared
above 0.99 across unseen wave-load conditions. It is what justifies
calling the framework a digital twin rather than a system identification
toolkit.

The two variants share the same underlying algebra. Given a fitted
Hankel-DMD model with mode matrix `Phi_full` (with `d * 18` rows
corresponding to `d` time lags of the eighteen-channel state vector),
the routine identifies the rows that correspond to the still-available
channels by computing `available_indices = setdiff1d(0..17, missing_sensors)`
and concatenating the available row indices across all `d` lag blocks.
Slicing `Phi_full` to those rows gives a sparse mode matrix `Phi_sparse`
that maps DMD amplitudes onto only the observable parts of the
state-snapshot history. Pseudo-inverting `Phi_sparse` gives a linear
operator that recovers the DMD amplitudes from a snapshot history of
the available channels alone. The reconstructed amplitudes are then
multiplied through the full mode matrix and the Vandermonde matrix of
DMD eigenvalues to predict the entire eighteen-channel state forward
in time. The result is unscaled by inverting the StandardScaler
transform applied during identification, and the predicted values for
the masked channel indices are compared against the (withheld) ground
truth using R squared, RMSE, and NRMSE.

The cross-case variant uses `svd_rank = 34` rather than 24 because the
generalisation task requires capturing modal variability across cases
that the model has not seen during training. The original script
documented this rank choice as the result of a trade-off between
representational richness and overfitting; lower ranks underfit the
inter-case variability and higher ranks overfit individual training
cases.

The runner script for the cross-case variant is
`scripts/analysis/run_missing_sensor_generalization.py` and its
configuration is `configs/missing_sensor_generalization.yaml`. The
runner exposes the masked-sensor indices, the train/test split, the
rolling-update horizon, the Hankel hyperparameters, and the transient-
trim length as YAML fields so that variations on the experiment can be
run without editing Python source code.

## SSI-COV cross-validation

The covariance-driven Stochastic Subspace Identification routine is
used as an independent baseline against which the Hankel-DMD
identification is compared. The implementation builds a block Toeplitz
matrix from sample covariances `R_k` of the multi-channel signal at
lags from 1 to `2 * i` (with `i = 60` block rows), takes its singular
value decomposition, truncates to the chosen model order, computes the
observability matrix `O_i = U_1 * sqrt(S_1)`, recovers the discrete-
time state-transition matrix `A` as the pseudoinverse of `O_i[:-rows]`
times `O_i[rows:]`, and extracts the system matrix `C = O_i[:rows]`.
Eigendecomposition of `A` yields modal frequencies, damping ratios,
and mode shapes via `shapes = C * Psi`. The two implementations are
`SSICOV` (single-order) and `SSIMultiOrder` (multi-order, used by the
stabilization diagram), both in `src/eftwin/ssi_cov.py` and matching
the originals in `SSICOV_2.py` and `SSI_COV_Stablizing.py`.

The stabilization diagram tracks pole persistence across consecutive
model orders. A pole is classified as `stable` if its frequency,
damping, and mode shape under MAC all match a pole in the previous
order within the tolerances `tol_freq = 0.01`, `tol_damp = 0.05`, and
`tol_mac = 0.98`. Looser categories (`stable_freq_mac`,
`stable_freq`, `new`) are also tracked and visualised with
progressively smaller marker sizes.

## Noise floor and rank truncation

A diagnostic Hankel block matrix is constructed manually (without
invoking `pydmd`) from the standardised state matrix downsampled by a
factor of five. Randomized SVD is applied via
`sklearn.utils.extmath.randomized_svd`, and the Gavish-Donoho threshold
(`median(S) * 2.858` for square matrices, adjusted by aspect ratio for
rectangular ones) is used as a heuristic for the optimal rank. The
implementation lives in
`src/eftwin/noise_lyapunov.py:hankel_singular_values` and matches
`Noise_Floor_Analysis.py`.

## Lyapunov exponent

The largest Lyapunov exponent of the top-tower acceleration is
estimated using the Rosenstein algorithm. A delay-coordinate embedding
is constructed with embedding dimension 10 and time delay 10 samples.
For each point in the embedded trajectory, the nearest neighbour in
phase space is identified using a KD-tree, with a temporal-separation
constraint of 100 samples to exclude trivially close neighbours. The
mean log-distance between trajectory pairs is then tracked as a
function of time, and the largest Lyapunov exponent is recovered as
the slope of the resulting curve in its linear region (the first
second). The predictability horizon is reported as
`T_lambda = 1 / lambda_max`. The implementation is in `embed_series`
and `calculate_lyapunov` and matches `Lyapunov_Exponent_Estimation.py`.

## Probabilistic mode-shape sensitivity

The Hankel-delay parameter `d` is swept across the values `[10, 20,
30, 35, 40, 45, 50, 55, 57, 63, 65, 70, 75, 80, 90]`. At each value
of `d`, a Hankel-DMD model is fitted with `svd_rank = 35` (chosen
slightly above the modal-identification rank because this analysis
prizes shape stability over rank parsimony). For each fitted model,
the mode whose frequency lies closest to a target value (0.54 Hz for
fore-aft, 0.52 Hz for side-to-side) within a tolerance of 0.1 Hz is
selected. Its acceleration mode-shape component is converted to
displacement, rotated so that the entry of largest modulus becomes
real and positive, and scaled so that this entry has unit modulus.
The real part of the resulting normalised shape is collected for all
`d` values inside a "convergence zone" of `[45, 75]`. The mean shape
and the 95 % Gaussian confidence interval `mean +/- 1.96 * std` are
then computed across the convergence zone and reported as the
probabilistic mode shape. This is implemented in
`src/eftwin/probabilistic_modes.py:run_probabilistic_mode_sweep` and
matches `Probablistic_Mode_shape 2.py` numerically.

## MAC validation

Two Modal Assurance Criterion variants are used. The complex MAC for
two mode-shape vectors `v1` and `v2` is
`|v1^H v2|^2 / ((v1^H v1) * (v2^H v2))`. The real-projected MAC,
which is designed for comparing complex DMD modes against magnitude-
only OpenFAST linear modes, first rotates the DMD mode so that its
largest entry is real and positive, then takes only the real part,
and finally compares that real vector to the absolute value of the
OpenFAST mode using the standard real MAC formula. Both variants are
implemented in `src/eftwin/modal_analysis.py` and match `MAC_Final.py`.
The orchestration script `scripts/analysis/run_mac_validation.py`
reads the mode-shape CSVs produced by the DMD and SSI runners
together with the OpenFAST `Extracted_Mode_Shapes.xlsx`, identifies
the best matching column for each user-specified target frequency,
computes both MAC variants for each match, and writes a heatmap of
the real-projected MAC matrix.

## OpenFAST simulation pipeline

The twelve nonlinear cases are generated by varying significant wave
height `Hs` over `{1, 2, 3, 4}` metres, peak period `Tp` over a
case-specific list, and wave direction over `{0, 45, 90}` degrees.
The rotor speed is held at 12 rpm and the blade pitch at 0 degrees
for all cases. Each case is simulated for 3600 seconds at a
0.0125-second time step, with output written every 0.0125 seconds (the
templates set `DT_Out = DT` so that the exported CSVs are at the same
rate the analysis pipeline expects). The full case list is in
`configs/openfast_cases.yaml`. Tower output gauges are placed at
nodes `[1, 3, 5, 7, 10, 13, 15, 17, 20]` of the structural mesh,
giving the nine gauge stations from which the analysis state matrix
is constructed.

The linearisation case uses a single operating point at 12 rpm and
17.1 degrees blade pitch (corresponding to the rated-wind operating
condition), simulated for 100 seconds. The OpenFAST `.lin` outputs
are then processed by `legacy/openfast_original/Modal_OutPut_Final_02.py`,
which extracts the state-transition matrix `A` and the output matrix
`C`, performs an eigendecomposition, identifies tower and blade state
indices via name matching, computes the modal participation in tower
and blade subspaces, projects the eigenvectors through `C` to obtain
physical mode shapes at the gauge stations, and writes the results
to `Extracted_Mode_Shapes.xlsx`. The cleaned wrapper
`scripts/openfast/extract_linear_modes.py` exposes that extraction
through a proper command-line interface that accepts arbitrary input
and output paths.
