"""Signal preprocessing utilities."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def apply_zero_phase_filter(data: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """Apply the same Butterworth zero-phase band-pass filter used in the original scripts."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data, axis=0)
