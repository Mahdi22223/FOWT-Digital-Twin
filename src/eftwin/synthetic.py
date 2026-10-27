"""Small synthetic tower-like response for examples and tests (not a physical model).

Modal coordinates are second-order resonators driven by white noise; the outputs are fixed mode shapes times the
modal coordinates plus a little sensor noise, so that frequencies, damping ratios and shapes are known exactly.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def resonator(n: int, f: float, zeta: float, dt: float, rng) -> np.ndarray:
    """Unit-variance response of a discrete second-order resonator to white noise."""
    w = 2 * np.pi * f
    r = np.exp(-zeta * w * dt)
    th = w * dt * np.sqrt(1 - zeta ** 2)
    y = lfilter([1.0], [1.0, -2 * r * np.cos(th), r ** 2], rng.standard_normal(n))
    return y / y.std()


def tower_response(n: int = 6000, dt: float = 0.05, modes=((0.5, 0.03), (0.6, 0.012), (3.7, 0.015)),
                   p: int = 6, noise: float = 1e-3, seed: int = 0):
    """Synthetic multi-channel response (n, p) with the given (frequency Hz, damping ratio) modes.

    Returns ``(Y, shapes)``; shapes (p, n_modes) are smooth functions of a normalised height.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0.1, 1.0, p)
    shapes = np.column_stack([np.cos((k + 0.5) * np.pi * (1 - x)) + 0.2 * x for k in range(len(modes))])
    q = np.column_stack([resonator(n, f, z, dt, rng) * (1.0 / (k + 1)) for k, (f, z) in enumerate(modes)])
    Y = q @ shapes.T
    Y += noise * Y.std(0) * rng.standard_normal(Y.shape)
    return Y, shapes
