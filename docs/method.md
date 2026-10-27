# Method and implementation

Where each step of the paper lives in the code, and the numerical conventions of the reported results. Equation
labels refer to the revised manuscript (`eq:tikhonov`, `eq:nowcast`, `eq:koopman_ss`, `eq:QR`, `eq:kf`,
Algorithm `alg:vs`); numbers are not used because they change with the final layout.

## 1. Data and preprocessing

| Step | OpenFAST (Sections 3-4) | FOCAL Campaign IV (Section 5) | Code |
|---|---|---|---|
| Records | 12 OLCs, `Case_<n>.csv`, written at the output interval `DT_Out = 0.05 s` | 4 runs (W02 with E11, E13, E14, E15), full scale (Froude 1:70), dt = 0.005 sqrt(70) s | `io.OpenFASTData`, `io.FocalData` |
| Channels per direction | 18: accelerations 0-8 and bending moments 9-17, tower base to top | 4: nacelle acc., tower-base (platform) acc., tower-top moment, tower-base moment | `io.FA_ACC` ... `io.FOCAL_CH` |
| Window | first 300 s (6000 samples) discarded | 600-9000 s | `configs/*.yaml` |
| Band-pass | 4th-order Butterworth, 0.25-5 Hz, `filtfilt` (b, a), applied after trimming | same filter on the window; platform modes: low-pass + decimation by 8, SOS band-pass 0.01-0.2 Hz | `preprocess.bandpass`, `bandpass_sos`, `lowpass_decimate` |
| Scaling for identification | z-score with mean and std of all OLCs (`StandardScaler`) | division by the std of all runs | `preprocess.standardise_segments` |
| Scaling for virtual sensing | mean and std (ddof 0) of the training fold only | same | `hankel.fold_stats` |

The time step is read from the files and checked against the configuration: the analysis uses the output
interval 0.05 s, not the solver step 0.0125 s (see `legacy/README.md`).

The independent OpenFAST linear reference used for Table 3 is the operating point at `t = 100 s`: 8 m/s steady
inflow, still water, MAP++ mooring, SubDyn enabled (`CompSub = 1`), and a nearly stationary rotor at 0.1801 rpm.
The linearisation deck uses `DT = 0.0125 s`, `DT_Out = 0.05 s` and defines 36 linearisation positions. Its historical
InflowWind filename contains `Steady12mps`, but the generated `.lin` and `.out` report 8 m/s and are authoritative.

## 2. Hankel-DMD (Algorithm `alg:vs`, offline phase)

* **Delay vector**: lag-major, oldest sample first, `h_k = [y_k; y_{k+1}; ...; y_{k+d-1}]` (length pd).
* **Gram route**: the Hankel matrices are never formed. `G = H1 H1^T` and `C = H2 H1^T` are accumulated chunk by
  chunk and segment by segment, so that no column spans two OLCs, runs or time halves
  (`hankel.delay_gram` for one segment with column sums for centring, `hankel.gram_over_segments` for several).
  With `G = U S^2 U^T`: `A_tilde = U_r^T C U_r S_r^-2`, `A_tilde W = W Lambda`, `Phi = C U_r S_r^-2 W`
  (`dmd.dmd_from_gram`). This is algebraically identical to the SVD route of Algorithm `alg:vs`
  (`tests/test_core.py::test_gram_route_equals_svd_route_exact_dmd`).
* **Modal parameters**: `f = |Im ln mu| / (2 pi dt)`, `zeta = -Re(lambda)/|lambda|`, `lambda = ln(mu)/dt` (`dmd.modal_params`).
* **Pole selection**: in every frequency band the pole with `Im mu > 0` and the largest modal energy
  `E_j = (Phi^+ G Phi^+H)_jj ||phi_j||^2` (`dmd.modal_energy`, `dmd.pick`). Medians and quartiles are taken over
  the grid d x r (`configs/*.yaml: identification`).
* **Correlation domain** (stochastic Koopman form, used for damping): output correlation functions
  `R_k = E[y_{t+k} y_t^T]`, k = 1..K; every column of `R_k` is a free-decay-like trajectory and the trajectories
  are the segments of the Hankel-DMD (`dmd.correlation_sum`, `dmd.correlation`, `dmd.correlation_trajectories`).
* **High-level API**: `model.identify_modes`, `model.fit_hankel_dmd`.

## 3. Virtual sensing (online phase)

| Paper | Code |
|---|---|
| Sparse mode matrix `Phi_sparse`: rows of the q active channels over all d delays | `hankel.delay_rows`, `estimators.sparse_svd` |
| Tikhonov estimate `b_k` (`eq:tikhonov`), filter factors `sigma_i/(sigma_i^2 + alpha^2 sigma_max^2)` | `estimators.tikhonov_filter`, `project_windows`, `modal_amplitudes` |
| Nowcast `Re(Phi_L b_k)` and forecast `Re(Phi_L Lambda^h b_k)` (`eq:nowcast`), `Phi_L` = most recent delay block | `estimators.projection_estimate`, `projection_forecast` |
| Causal windows (no sample after k); start-up windows padded with zeros in the API | `estimators.delay_windows` |
| Kalman form: modal state-space model (`eq:koopman_ss`), `Q` and `R` from `Sigma_0`, `Sigma_1` (`eq:QR`) | `estimators.kalman_model` |
| Steady-state gain from the DARE and recursion (`eq:kf`) | `estimators.kalman_gain`, `kalman_states` |
| Classes for users | `model.ProjectionEstimator`, `model.KalmanEstimator` |

Implementation details of the Kalman form that affect the numbers: eigenvalues with `|lambda| > 0.9995` are moved
radially to 0.9995; `Q` is symmetrised and its eigenvalues floored at `1e-10 max eig(Q)`; `R_aa` is regularised by
`1e-6 tr(R)/p`; the recursion is evaluated through the eigen-decomposition of `F = (I - K C_a) Lambda` with
first-order filters, falling back to the direct recursion when `cond(V_F) > 1e8`.

## 4. Baselines

* Virtual sensing (`baselines/vs.py`), all from the same training covariance: gappy POD (one model for all
  layouts, `k = min(q, 5)` POD modes of the instantaneous covariance), static least squares per layout (ridge
  `1e-9 tr(G_aa)`), finite-window Wiener filter = delay ridge regression on the same d-sample window per layout
  (ridge `rho tr(G_AA)/n`, `rho` cross-fitted).
* Operational modal analysis (`baselines/oma.py`): SSI-COV (block Toeplitz `T[a, b] = R_{i+a-b}`) and FDD
  (first singular vector of the CSD matrix).

## 5. Validation (`validation.py`)

Time-blocked leave-one-out: each record is split into two halves; fold (test record c, half h) trains on half
1-h of all other records (`blocked_folds`), so that neither the test record nor its time interval is seen. Per
hidden channel only sufficient statistics are stored (`sufficient_stats`), so that R^2 and NRMSE of any
aggregate are exact (`combine`). Hyper-parameters are cross-fitted between two groups of records (`crossfit`):
OLC 1-6 vs 7-12 (per number of active sensors q) and E11/E13 vs E14/E15 (per layout). The coherence bound of
Table 7 is `coherence_bound`.

## 6. Conventions of the reported results

The two studies were run with slightly different conventions. Each is kept as it was used, behind a named option,
so that every reported number is reproduced; none is changed for uniformity. Changing one would change the
corresponding numbers and requires a rerun on the raw data (see `docs/open_items.md`).

| Item | OpenFAST (Table 3, Figs. 5-7, A1) | FOCAL (Table 6, Fig. 14) |
|---|---|---|
| Shape block returned by `pick` | first delay block (`shape_rows=slice(0, p)`) | most recent block (`slice(-p, None)`), and `positive_damping=True` |
| Pooled correlation functions | `correlation_sum`: sum over OLCs divided by the total number of samples | mean of the per-run `correlation` (each divided by its N) |
| SSI-COV selection | `ssi_median_over_orders`: in-band pole with 0 < zeta < 0.2 and the largest shape norm per order, medians over orders 20-80, no stabilisation | `ssi_stabilised`: df < 1 %, dzeta < 5 %, MAC > 0.95 between consecutive orders, zeta < 0.3 |
| FDD | 100-s Welch segments; first singular vector at every frequency, peak of the first singular value in the band | 200-s segments; first singular value in the band, vector at the peak (`fdd_peak`) |
| Reference model for shapes and MACs | d = 60, r = 24 (response domain) | d = 86, r = min(40, max r) (correlation domain) |
| Correlation lags | 400 (20 s) | 20 s |

Further conventions: Gram sums are accumulated in chunks of 20 000 columns (15 000 in the delay-depth sweep of
Fig. 5, which also uses Gram sums not divided by n; both only change the floating-point summation order). The
first-mode parameters drawn in Fig. 11 (`figures.fig11_first_mode` in `configs/openfast.yaml`) and the rotor 1P
frequency of the FOCAL coherence bound (`coherence_bound.rotor_1p_hz`) are configuration values, as in the
revision code. The noise of robustness test T3 is drawn from `default_rng(11)` at the start of each run, one run
per direction.

## 7. Verification

`tests/test_revision_equivalence.py` recomputes, for synthetic inputs, every quantity stored in
`tests/data/revision_golden.npz` by the finalized revision scripts (identification in both domains, SSI-COV with
both selection rules, FDD, the five virtual-sensing estimators, forecasts, Kalman form). `tests/test_core.py`
checks the identities the implementation relies on (Gram sums vs explicit Hankel products, Gram vs SVD route,
Tikhonov filter vs regularised least squares, eigen vs direct Kalman recursion, Wiener weights vs ridge regression
on explicit windows, exact aggregation of the sufficient statistics, causality of the API estimators).
