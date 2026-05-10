"""Modal-parameter and mode-shape post-processing."""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


def modal_parameters(eigs: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map discrete DMD eigenvalues to continuous-time frequency and damping."""
    omega = np.log(eigs) / dt
    freqs_hz = np.abs(omega.imag) / (2.0 * np.pi)
    damping = -omega.real / np.abs(omega)
    return omega, freqs_hz, damping


def extract_physical_modes(analysis_result, dt: float, low_f: float = 0.3, high_f: float = 5.0, dedup_tol: float = 0.1):
    """Extract physical moment mode shapes and modal statistics from a Hankel-DMD result."""
    hdmd = analysis_result.model if hasattr(analysis_result, "model") else analysis_result["model"]
    eigs = hdmd.eigs
    _, freqs_hz, damping = modal_parameters(eigs, dt)
    valid_idx = np.where((freqs_hz > low_f) & (freqs_hz < high_f))[0]
    sorted_idx = valid_idx[np.argsort(freqs_hz[valid_idx])]

    unique_freqs = []
    extracted_modes = {}
    stats = []
    for idx in sorted_idx:
        f = freqs_hz[idx]
        if any(abs(f - u) < dedup_tol for u in unique_freqs):
            continue
        unique_freqs.append(f)
        phi_phys = hdmd.modes[:, idx][0:18]
        extracted_modes[f"Freq_{f:.3f}Hz_Mode{idx}"] = phi_phys[9:18]
        stats.append({"Freq_Hz": f, "Damping": damping[idx], "Index": int(idx)})
    return pd.DataFrame(extracted_modes), pd.DataFrame(stats)


def acceleration_mode_to_displacement(acc_complex: np.ndarray, freq_hz: float) -> np.ndarray:
    """Convert acceleration mode shape to displacement mode shape using -a/omega^2."""
    w = 2.0 * np.pi * freq_hz
    if w == 0:
        w = 1e-6
    return -acc_complex / (w**2)


def normalize_complex_shape(shape: np.ndarray) -> np.ndarray:
    """Rotate and normalize a complex mode shape using its largest entry."""
    shape = np.asarray(shape)
    max_idx = np.argmax(np.abs(shape))
    phase = np.angle(shape[max_idx])
    rotated = shape * np.exp(-1j * phase)
    denom = np.abs(rotated[max_idx])
    return rotated / denom if denom != 0 else rotated


def parse_complex(value) -> complex:
    if isinstance(value, (int, float, complex)):
        return complex(value)
    s = str(value).replace("(", "").replace(")", "").replace("i", "j")
    try:
        return complex(s)
    except Exception:
        return 0j


def extract_freq_from_name(name: str) -> float:
    match = re.search(r"(\d+\.\d+)Hz", str(name))
    return float(match.group(1)) if match else -1.0


def complex_mac(v1, v2) -> float:
    v1 = np.asarray(v1).flatten()
    v2 = np.asarray(v2).flatten()
    if len(v1) != len(v2):
        return 0.0
    num = np.abs(np.vdot(v1, v2)) ** 2
    den = np.vdot(v1, v1).real * np.vdot(v2, v2).real
    return float(num / den) if den != 0 else 0.0


def real_projected_mac(v_dmd, v_of) -> float:
    """MAC variant used in the original validation script for complex DMD modes."""
    v_dmd = np.asarray(v_dmd).flatten()
    v_of = np.asarray(v_of).flatten()
    idx_max = np.argmax(np.abs(v_dmd))
    angle = np.angle(v_dmd[idx_max])
    v_dmd_rot = (v_dmd * np.exp(-1j * angle)).real
    v_of_shape = np.abs(v_of)
    num = np.dot(v_dmd_rot, v_of_shape) ** 2
    den = np.dot(v_dmd_rot, v_dmd_rot) * np.dot(v_of_shape, v_of_shape)
    return float(num / den) if den != 0 else 0.0
