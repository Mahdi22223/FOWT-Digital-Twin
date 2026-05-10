"""Probabilistic mode-shape estimation via Hankel-delay sensitivity.

This module is a faithful translation of the methodology in the original
``Probablistic_Mode_shape 2.py`` script. The numerical procedure is unchanged:

1. For each Hankel delay ``d`` in a sweep range, fit a Hankel-DMD model.
2. For each fitted model, find the mode whose discrete-time eigenvalue maps
   to a frequency closest to a target frequency (within a tolerance).
3. Convert the mode's acceleration component to a displacement shape using
   the standard ``-a / omega**2`` relation, rotate it so the largest entry
   is real and positive, and scale it so its largest entry has magnitude 1.
4. Collect the normalized real-valued shapes from the user-defined
   "convergence zone" of ``d`` values (in the original script, ``d in [45, 75]``).
5. Compute the mean shape and the 95% confidence interval using a Gaussian
   approximation: ``CI = mean +/- 1.96 * std``.

The original script combined this loop with plotting in a single file. Here
the loop is exposed as a function returning a results dictionary, and a
companion plotting helper lives in :mod:`eftwin.plotting`. This separation is
software organization only; the numerical formulas are identical.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd

from .data_io import load_direction_matrix
from .hankel_dmd import run_hankel_dmd


def extract_mode_data(hdmd, target_freq: float, dt: float, search_tol: float = 0.1) -> dict | None:
    """Return frequency, damping, and a normalized real shape for the closest mode to target_freq.

    Reproduces the per-d extraction logic from ``Probablistic_Mode_shape 2.py``.
    Returns ``None`` if no mode lies within ``search_tol`` Hz of ``target_freq``.
    """
    eigs = hdmd.eigs
    if eigs is None:
        return None

    # Continuous-time mapping (matches modal_analysis.modal_parameters)
    omega = np.log(eigs) / dt
    freqs_hz = np.abs(omega.imag) / (2.0 * np.pi)
    damping = -omega.real / np.abs(omega)

    valid = np.where(
        (freqs_hz > target_freq - search_tol) & (freqs_hz < target_freq + search_tol)
    )[0]
    if len(valid) == 0:
        return None

    best_local = np.argmin(np.abs(freqs_hz[valid] - target_freq))
    idx = valid[best_local]

    # Same physical-row slice as the original: first 18 rows (9 acc + 9 mom)
    phi_full = hdmd.modes[:, idx]
    phi_phys = phi_full[0:18]
    acc_complex = phi_phys[0:9]

    # Acceleration -> displacement using -a/omega^2
    w = 2.0 * np.pi * freqs_hz[idx]
    if w == 0:
        w = 1e-6
    disp_complex = -acc_complex / (w**2)

    # Rotate so the entry with maximum modulus becomes real and positive,
    # then scale so its modulus is 1. This is exactly the original normalization.
    max_idx = np.argmax(np.abs(disp_complex))
    phase_shift = np.angle(disp_complex[max_idx])
    disp_norm = disp_complex * np.exp(-1j * phase_shift)
    denom = np.abs(disp_norm[max_idx])
    if denom != 0:
        disp_norm = disp_norm / denom

    return {
        "Freq": float(freqs_hz[idx]),
        "Damping": float(damping[idx]),
        "Shape": disp_norm.real,  # real part of the rotated normalized complex shape
    }


def run_probabilistic_mode_sweep(
    case_files: list,
    acc_cols: list[str],
    mom_cols: list[str],
    *,
    target_freq: float,
    direction_name: str = "Fore-Aft",
    d_range: Iterable[int] = (10, 20, 30, 35, 40, 45, 50, 55, 57, 63, 65, 70, 75, 80, 90),
    convergence_d_min: int = 45,
    convergence_d_max: int = 75,
    dt: float = 0.0125,
    low_cutoff: float = 0.25,
    high_cutoff: float = 5.0,
    filter_order: int = 4,
    svd_rank: int = 35,
    search_tol: float = 0.1,
) -> dict:
    """Sweep Hankel delay ``d`` and return frequency, damping, and shape statistics.

    Parameters mirror the original ``Probablistic_Mode_shape 2.py`` defaults:
    ``svd_rank=35`` (note this differs from identification rank 24 and from
    virtual-sensing rank 34 — the original script used 35 for this sweep),
    target frequencies are typically 0.54 Hz for fore-aft and 0.52 Hz for
    side-to-side, and the convergence zone is ``d in [45, 75]``.
    """
    sensitivity_rows: list[dict] = []
    convergence_shapes: list[np.ndarray] = []

    for d in d_range:
        result = run_hankel_dmd(
            case_files,
            acc_cols,
            mom_cols,
            direction_name=direction_name,
            dt=dt,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            filter_order=filter_order,
            hankel_d=int(d),
            svd_rank=svd_rank,
        )
        mode_info = extract_mode_data(result.model, target_freq, dt, search_tol=search_tol)
        if mode_info is None:
            continue
        sensitivity_rows.append({
            "d": int(d),
            "Freq": mode_info["Freq"],
            "Damping": mode_info["Damping"],
        })
        if convergence_d_min <= int(d) <= convergence_d_max:
            convergence_shapes.append(mode_info["Shape"])

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    if convergence_shapes:
        S = np.array(convergence_shapes)
        mean_shape = S.mean(axis=0)
        std_shape = S.std(axis=0)
        ci_upper = mean_shape + 1.96 * std_shape
        ci_lower = mean_shape - 1.96 * std_shape
    else:
        S = np.empty((0, 9))
        mean_shape = np.zeros(9)
        std_shape = np.zeros(9)
        ci_upper = np.zeros(9)
        ci_lower = np.zeros(9)

    return {
        "sensitivity": sensitivity_df,
        "shapes": S,
        "mean_shape": mean_shape,
        "std_shape": std_shape,
        "ci_upper": ci_upper,
        "ci_lower": ci_lower,
        "target_freq": target_freq,
        "convergence_window": (convergence_d_min, convergence_d_max),
        "direction_name": direction_name,
    }
