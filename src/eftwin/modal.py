"""Mode-shape comparison: modal assurance criterion, modal phase collinearity and normalisation."""
from __future__ import annotations

import numpy as np


def mac(a: np.ndarray, b: np.ndarray) -> float:
    """Modal assurance criterion |a^H b|^2 / ((a^H a)(b^H b)) for complex shapes."""
    return float(np.abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real))


def mpc(v: np.ndarray) -> float:
    """Modal phase collinearity ((l1 - l2) / (l1 + l2))^2 of the 2x2 covariance of [Re v, Im v]."""
    re, im = v.real, v.imag
    l1, l2 = np.linalg.eigvalsh(np.array([[re @ re, re @ im], [re @ im, im @ im]]))[::-1]
    return float(((l1 - l2) / (l1 + l2)) ** 2)


def real_shape(v: np.ndarray) -> np.ndarray:
    """Rotate a complex shape so that its largest entry is real; real part scaled to unit maximum modulus."""
    v = v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))]))
    return v.real / np.abs(v.real).max()


def displacement_shape(acc: np.ndarray, freq_hz: float) -> np.ndarray:
    """Displacement shape -a / omega^2 of an acceleration shape, rotated and scaled to unit amplitude at its
    largest entry (real part returned), as plotted in Fig. 5."""
    disp = -acc / (2 * np.pi * freq_hz) ** 2
    j = np.argmax(np.abs(disp))
    disp = disp * np.exp(-1j * np.angle(disp[j])) / np.abs(disp[j])
    return disp.real
