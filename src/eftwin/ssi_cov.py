"""SSI-COV comparison and stabilization utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import svd, pinv

from .data_io import load_direction_matrix
from .modal_analysis import complex_mac


class SSICOV:
    def __init__(self, dt: float, block_rows: int = 60):
        self.dt = dt
        self.i = block_rows

    def fit(self, data: np.ndarray, order: int):
        n_sensors, n_samples = data.shape
        R_matrices = []
        for k in range(1, 2 * self.i + 1):
            Y_future = data[:, k:]
            Y_past = data[:, :-k]
            R_k = (Y_future @ Y_past.T) / (n_samples - k)
            R_matrices.append(R_k)
        T1 = np.vstack([np.hstack(R_matrices[r: r + self.i]) for r in range(self.i)])
        U, S, _ = svd(T1)
        U1 = U[:, :order]
        S1 = np.diag(S[:order])
        Oi = U1 @ np.sqrt(S1)
        rows_per_block = n_sensors
        A = pinv(Oi[:-rows_per_block, :]) @ Oi[rows_per_block:, :]
        C = Oi[:rows_per_block, :]
        mu, Psi = np.linalg.eig(A)
        lambda_c = np.log(mu) / self.dt
        freqs = np.abs(lambda_c) / (2 * np.pi)
        damps = -lambda_c.real / np.abs(lambda_c)
        mode_shapes = C @ Psi
        return freqs, damps, mode_shapes


class SSIMultiOrder:
    def __init__(self, dt: float, block_rows: int = 60):
        self.dt = dt
        self.i = block_rows
        self.U = None
        self.S = None
        self.n_sensors = 0

    def prepare_svd(self, data: np.ndarray):
        self.n_sensors, n_samples = data.shape
        R_matrices = []
        for k in range(1, 2 * self.i + 1):
            Y_future = data[:, k:]
            Y_past = data[:, :-k]
            R_k = (Y_future @ Y_past.T) / (n_samples - k)
            R_matrices.append(R_k)
        T1 = np.vstack([np.hstack(R_matrices[r: r + self.i]) for r in range(self.i)])
        self.U, self.S, _ = svd(T1, full_matrices=False)

    def solve_for_order(self, order: int):
        U1 = self.U[:, :order]
        S1 = np.diag(self.S[:order])
        Oi = U1 @ np.sqrt(S1)
        rows_per_block = self.n_sensors
        A = pinv(Oi[:-rows_per_block, :]) @ Oi[rows_per_block:, :]
        C = Oi[:rows_per_block, :]
        mu, Psi = np.linalg.eig(A)
        lambda_c = np.log(mu) / self.dt
        freqs = np.abs(lambda_c) / (2 * np.pi)
        damps = -lambda_c.real / np.abs(lambda_c)
        mode_shapes = C @ Psi
        return freqs, damps, mode_shapes


def run_ssi_validation(case_files, acc_cols, mom_cols, *, dt=0.0125, low_cutoff=0.25, high_cutoff=5.0, filter_order=4, block_rows=60, model_order=40, downsample_factor=4, min_frequency_hz=0.3, max_frequency_hz=5.0, max_damping_ratio=0.2):
    X = load_direction_matrix(case_files, acc_cols, mom_cols, low_cutoff=low_cutoff, high_cutoff=high_cutoff, fs=1.0/dt, filter_order=filter_order)
    X_sub = X[:, ::downsample_factor]
    ssi = SSICOV(dt=dt * downsample_factor, block_rows=block_rows)
    freqs, damps, modes = ssi.fit(X_sub, order=model_order)
    valid = []
    for idx, f in enumerate(freqs):
        d = damps[idx]
        if min_frequency_hz < f < max_frequency_hz and 0.0 < d < max_damping_ratio:
            valid.append({"Freq": f, "Damp": d, "Mode": modes[:, idx]})
    valid.sort(key=lambda x: x["Freq"])
    clean = []
    seen = []
    for res in valid:
        if not any(abs(res["Freq"] - x) < 0.05 for x in seen):
            clean.append(res)
            seen.append(res["Freq"])
    mode_dict = {f"Freq_{res['Freq']:.3f}Hz": res["Mode"][9:18] for res in clean}
    stats = pd.DataFrame([{"Freq_Hz": r["Freq"], "Damping": r["Damp"]} for r in clean])
    return pd.DataFrame(mode_dict), stats, clean


def run_stabilization(case_files, acc_cols, mom_cols, *, dt=0.0125, low_cutoff=0.25, high_cutoff=5.0, filter_order=4, block_rows=60, downsample_factor=4, min_order=2, max_order=60, step_order=2, tol_freq=0.01, tol_damp=0.05, tol_mac=0.98):
    X = load_direction_matrix(case_files, acc_cols, mom_cols, low_cutoff=low_cutoff, high_cutoff=high_cutoff, fs=1.0/dt, filter_order=filter_order)
    X_sub = X[:, ::downsample_factor]
    ssi = SSIMultiOrder(dt=dt * downsample_factor, block_rows=block_rows)
    ssi.prepare_svd(X_sub)
    all_poles = []
    prev_modes = []
    for order in range(min_order, max_order + 1, step_order):
        freqs, damps, shapes = ssi.solve_for_order(order)
        current_modes = []
        for k in range(len(freqs)):
            f = freqs[k]
            d = damps[k]
            v = shapes[:, k]
            if f < 0.25 or f > 5.0 or d < 0 or d > 0.3:
                continue
            stable_freq = stable_damp = stable_mac = False
            if prev_modes:
                best_match = min(prev_modes, key=lambda pm: abs(f - pm["f"]))
                if abs(f - best_match["f"]) / best_match["f"] < tol_freq:
                    stable_freq = True
                if abs(d - best_match["d"]) / (best_match["d"] + 1e-6) < tol_damp:
                    stable_damp = True
                if complex_mac(v, best_match["v"]) > tol_mac:
                    stable_mac = True
            if stable_freq and stable_damp and stable_mac:
                status = "stable"
            elif stable_freq and stable_mac:
                status = "stable_freq_mac"
            elif stable_freq:
                status = "stable_freq"
            else:
                status = "new"
            pole = {"order": order, "f": f, "d": d, "v": v, "status": status}
            all_poles.append(pole)
            current_modes.append(pole)
        prev_modes = current_modes
    return all_poles
