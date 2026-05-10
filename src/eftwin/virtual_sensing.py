"""Virtual sensing and missing-sensor generalisation with Hankel-DMD modes.

This module implements two related virtual-sensing workflows that together
form the digital twin's sensor-recovery capability described in the paper:

1. ``run_virtual_sensing`` performs rolling-horizon reconstruction of
   designated hidden sensor channels using the Hankel-DMD modes that were
   identified on the same dataset. It is the in-sample variant: training
   and testing happen on the same case set. This corresponds to the
   methodology in the original ``Virtual_sensing 2.py`` script.

2. ``train_generalization_model`` plus ``run_generalization_prediction``
   (or the convenience orchestrator ``run_missing_sensor_generalization``)
   implements the cross-case generalisation experiment: a Hankel-DMD model
   is fitted on the first N cases and then used to reconstruct hidden
   sensors on the remaining unseen cases. This is the missing/failed
   sensor experiment from the original
   ``Missing_Sensor_Generalization_2 2.py`` script and is the result that
   demonstrates the equation-free digital twin's ability to recover real
   structural signals on operating conditions it has not seen during
   training.

The two workflows share the same underlying algebra: identify which rows
of the full Hankel mode matrix correspond to available sensors, pseudo-
invert the resulting sparse mode matrix, calibrate mode amplitudes from
the available channels in a sliding observation window, and reconstruct
the full state through the Vandermonde matrix of the DMD eigenvalues.
The differences between the two routines are confined to data handling
(in-sample versus train/test split) and to the default DMD rank (24 for
modal identification, 34 for generalisation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from pydmd import HankelDMD

from .data_io import load_direction_matrix


def run_virtual_sensing(analysis_result, missing_sensors=(0, 8), update_every=1.0, duration=300.0, dt=0.0125, hankel_d=None):
    """Recover hidden sensors by calibrating modal amplitudes from available Hankel rows.

    This follows the algebra in the original `Virtual_sensing 2.py` script.
    The ``hankel_d`` argument allows callers to pass the Hankel delay
    explicitly. When ``None`` (the default), the function attempts to
    read ``_d`` from the fitted pydmd model and falls back to 60 if it
    cannot. Passing ``hankel_d`` explicitly is recommended because pydmd
    does not formally guarantee the ``_d`` private attribute across
    versions.
    """
    if hasattr(analysis_result, "as_legacy_dict"):
        analysis_result = analysis_result.as_legacy_dict()
    hdmd = analysis_result["model"]
    scaler = analysis_result["scaler"]
    X_true_all = analysis_result["X_raw"]
    if hankel_d is None:
        hankel_d = getattr(hdmd, "_d", getattr(hdmd, "d", 60))

    Phi_full = hdmd.modes
    Phi_phys_full = Phi_full[0:18, :]
    eigs = hdmd.eigs

    all_indices = np.arange(18)
    available_indices = np.setdiff1d(all_indices, np.asarray(missing_sensors))
    valid_hankel_rows = []
    for lag in range(hankel_d):
        valid_hankel_rows.extend(available_indices + lag * 18)
    valid_hankel_rows = np.asarray(valid_hankel_rows)

    Phi_sparse = Phi_full[valid_hankel_rows, :]
    Phi_sparse_pinv = np.linalg.pinv(Phi_sparse)

    update_steps = max(1, int(update_every / dt))
    total_steps = min(int(duration / dt), X_true_all.shape[1] - hankel_d)
    X_virtual_pred = np.zeros((18, total_steps))

    current_step = 0
    while current_step < total_steps:
        end_step = min(current_step + update_steps, total_steps)
        window_len = end_step - current_step
        X_hist_full = X_true_all[:, current_step: current_step + hankel_d]
        X_hist_scaled = scaler.transform(X_hist_full.T).T
        x_hankel_full = X_hist_scaled.flatten(order="F")
        x_hankel_sparse = x_hankel_full[valid_hankel_rows]
        b_new = np.dot(Phi_sparse_pinv, x_hankel_sparse)
        vandermonde = np.vander(eigs, window_len, increasing=True)
        dynamics_window = b_new[:, None] * vandermonde
        X_pred_scaled = np.dot(Phi_phys_full, dynamics_window).real
        X_pred_phys = scaler.inverse_transform(X_pred_scaled.T).T
        X_virtual_pred[:, current_step:end_step] = X_pred_phys
        current_step = end_step

    metrics = []
    for idx in missing_sensors:
        truth = X_true_all[idx, :total_steps]
        pred = X_virtual_pred[idx, :total_steps]
        rmse = float(np.sqrt(mean_squared_error(truth, pred)))
        r2 = float(r2_score(truth, pred))
        range_val = float(np.max(truth) - np.min(truth))
        nrmse = float((rmse / range_val) * 100.0) if range_val != 0 else np.nan
        metrics.append({"sensor_index": int(idx), "R2": r2, "RMSE": rmse, "NRMSE_percent": nrmse})

    return {"pred": X_virtual_pred, "truth": X_true_all[:, :total_steps], "metrics": pd.DataFrame(metrics), "update_s": update_every, "dt": dt}


def train_generalization_model(case_files, acc_cols, mom_cols, *, train_case_count=8, dt=0.0125, low_cutoff=0.25, high_cutoff=5.0, filter_order=4, hankel_d=60, svd_rank=34, opt=False, trim_steps=1000):
    """Train on the first cases and prepare unseen test cases, matching the original generalization script.

    The ``trim_steps`` argument removes that many samples from the leading
    and trailing edges of each case before filtering, to discard the
    transient response from OpenFAST's start-up. The original script
    used 1000 samples (12.5 s at dt = 0.0125 s) which is appropriate
    for the paper's 3600 s runs but would trim away the entire signal
    on shorter datasets. Reduce it for smoke tests on small synthetic
    samples; keep the default for paper reproduction.
    """
    train_files = case_files[:train_case_count]
    test_files = case_files[train_case_count:]
    fs = 1.0 / dt
    X_train = load_direction_matrix(train_files, acc_cols, mom_cols, low_cutoff=low_cutoff, high_cutoff=high_cutoff, fs=fs, filter_order=filter_order, trim_steps=trim_steps)
    scaler = StandardScaler()
    scaler.fit(X_train.T)
    X_train_scaled = scaler.transform(X_train.T).T
    hdmd = HankelDMD(svd_rank=svd_rank, tlsq_rank=0, exact=True, opt=opt, d=hankel_d)
    hdmd.fit(X_train_scaled)
    X_test_raw = load_direction_matrix(test_files, acc_cols, mom_cols, low_cutoff=low_cutoff, high_cutoff=high_cutoff, fs=fs, filter_order=filter_order, trim_steps=trim_steps)
    X_test_scaled = scaler.transform(X_test_raw.T).T
    return {"model": hdmd, "scaler": scaler, "X_test_raw": X_test_raw, "X_test_scaled": X_test_scaled, "train_files": train_files, "test_files": test_files}


def run_generalization_prediction(model_data, missing_sensors=(0, 8), update_every_seconds=1.0, duration=300.0, dt=0.0125, hankel_d=None):
    """Prediction loop for unseen cases, matching `Missing_Sensor_Generalization_2 2.py`.

    See the docstring of ``run_virtual_sensing`` for an explanation of the
    ``hankel_d`` argument. Passing it explicitly is the safer pattern.
    """
    hdmd = model_data["model"]
    scaler = model_data["scaler"]
    X_test_raw = model_data["X_test_raw"]
    X_test_scaled = model_data["X_test_scaled"]
    if hankel_d is None:
        hankel_d = getattr(hdmd, "_d", getattr(hdmd, "d", 60))
    Phi_full = hdmd.modes
    Phi_phys = Phi_full[0:18, :]
    eigs = hdmd.eigs

    all_indices = np.arange(18)
    avail_indices = np.setdiff1d(all_indices, np.asarray(missing_sensors))
    valid_hankel_rows = []
    for lag in range(hankel_d):
        valid_hankel_rows.extend(avail_indices + lag * 18)
    valid_hankel_rows = np.asarray(valid_hankel_rows)

    Phi_sparse_pinv = np.linalg.pinv(Phi_full[valid_hankel_rows, :])
    total_steps = min(int(duration / dt), X_test_scaled.shape[1] - hankel_d)
    update_steps = max(1, int(update_every_seconds / dt))
    X_virtual_pred = np.zeros((18, total_steps))

    current_step = 0
    while current_step < total_steps:
        end_step = min(current_step + update_steps, total_steps)
        pred_len = end_step - current_step
        X_hist_full = X_test_scaled[:, current_step: current_step + hankel_d]
        x_hankel_full = X_hist_full.flatten(order="F")
        b_new = np.dot(Phi_sparse_pinv, x_hankel_full[valid_hankel_rows])
        V = np.vander(eigs, pred_len, increasing=True)
        X_pred_scaled = np.dot(Phi_phys, b_new[:, None] * V).real
        X_virtual_pred[:, current_step:end_step] = scaler.inverse_transform(X_pred_scaled.T).T
        current_step = end_step

    metrics = []
    for idx in missing_sensors:
        truth = X_test_raw[idx, :total_steps]
        pred = X_virtual_pred[idx, :total_steps]
        rmse = float(np.sqrt(mean_squared_error(truth, pred)))
        r2 = float(r2_score(truth, pred))
        rng = float(np.max(truth) - np.min(truth))
        metrics.append({"sensor_index": int(idx), "R2": r2, "RMSE": rmse, "NRMSE_percent": (rmse / rng) * 100 if rng else np.nan})
    return {"pred": X_virtual_pred, "truth": X_test_raw[:, :total_steps], "metrics": pd.DataFrame(metrics), "update_s": update_every_seconds, "dt": dt}


def run_missing_sensor_generalization(
    case_files,
    acc_cols,
    mom_cols,
    *,
    missing_sensors=(0, 8),
    train_case_count=8,
    update_every_seconds=1.0,
    duration=300.0,
    dt=0.0125,
    low_cutoff=0.25,
    high_cutoff=5.0,
    filter_order=4,
    hankel_d=60,
    svd_rank=34,
    opt=False,
    trim_steps=1000,
):
    """Train-on-N / test-on-(M-N) missing-sensor generalisation experiment.

    This is the methodology of the original ``Missing_Sensor_Generalization_2 2.py``
    script: a single high-level function that fits a Hankel-DMD model on the
    first ``train_case_count`` cases and uses it to reconstruct hidden tower
    sensors on the remaining unseen cases. It is the experiment that
    demonstrates the equation-free digital twin's ability to recover failed
    or never-installed sensors on operating conditions the model has not
    seen during training, which is the central virtual-sensing capability
    reported in the paper.

    Parameters mirror the originals exactly: rank 34 for the Hankel-DMD
    fit (intentionally higher than the modal-identification rank 24
    because the generalisation task needs additional modal richness), the
    same band-pass filter window (0.25-5 Hz, 4th-order Butterworth), the
    same Hankel delay (60 samples), and the same default missing-sensor
    set (lowest and highest acceleration channels, indices 0 and 8 of the
    18-channel state vector).

    Returns a dictionary with keys:
        ``model_data``  -- the trained model bundle from ``train_generalization_model``;
        ``prediction``  -- the reconstruction result from ``run_generalization_prediction``,
                            including the predicted and ground-truth time series and the
                            per-sensor accuracy metrics (R2, RMSE, NRMSE).
    """
    model_data = train_generalization_model(
        case_files,
        acc_cols,
        mom_cols,
        train_case_count=train_case_count,
        dt=dt,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        filter_order=filter_order,
        hankel_d=hankel_d,
        svd_rank=svd_rank,
        opt=opt,
        trim_steps=trim_steps,
    )
    prediction = run_generalization_prediction(
        model_data,
        missing_sensors=missing_sensors,
        update_every_seconds=update_every_seconds,
        duration=duration,
        dt=dt,
        hankel_d=hankel_d,
    )
    return {"model_data": model_data, "prediction": prediction}
