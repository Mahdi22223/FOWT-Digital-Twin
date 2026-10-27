"""Signal preprocessing: zero-phase band-pass filters and standardisation.

The filters are applied to whole records before identification or estimation (zero-phase, i.e. not causal).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt
from sklearn.preprocessing import StandardScaler


def bandpass(x: np.ndarray, low: float, high: float, fs_design: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass in transfer-function form (``filtfilt``), axis 0 = time.

    Used for all OpenFAST analyses and for the FOCAL analysis band (0.25-5 Hz).
    """
    nyq = 0.5 * fs_design
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, x, axis=0)


def bandpass_sos(x: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass in second-order sections (``sosfiltfilt``), axis 0 = time.

    Used for the FOCAL wide-band analyses (0.2-11 Hz and 0.25-10 Hz) and the platform band.
    """
    sos = butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype="band", output="sos")
    return sosfiltfilt(sos, x, axis=0)


def lowpass_decimate(x: np.ndarray, fs: float, cutoff: float, factor: int, order: int = 8) -> np.ndarray:
    """Zero-phase low-pass (second-order sections) followed by down-sampling by ``factor``."""
    return sosfiltfilt(butter(order, cutoff / (0.5 * fs), btype="low", output="sos"), x, axis=0)[::factor]


def standardise_segments(segments: dict) -> tuple[dict, StandardScaler]:
    """z-score all segments with the mean and standard deviation (ddof = 0) of their union.

    Returns the scaled segments (same keys) and the fitted scaler (``scaler.scale_`` de-standardises shapes).
    """
    scaler = StandardScaler().fit(np.vstack(list(segments.values())))
    return {k: scaler.transform(s) for k, s in segments.items()}, scaler
