"""Baseline virtual-sensing estimators, computed from the same training covariance as Hankel-DMD.

* Gappy POD: static, one model for all layouts; min(q, 5) POD modes of the instantaneous covariance.
* Static least squares: best instantaneous linear map from the active to the hidden channels, per layout.
* Wiener filter: delay ridge regression on the same d-sample window, per layout (finite-window Wiener filter).
"""
from __future__ import annotations

import numpy as np


def pod_basis(G0: np.ndarray) -> np.ndarray:
    """POD modes of the instantaneous covariance G0 (most recent delay block), decreasing energy."""
    evp, Upod = np.linalg.eigh(G0)
    return Upod[:, np.argsort(evp)[::-1]]


def gappy_pod(Upod: np.ndarray, active, hidden, Xa: np.ndarray, k: int) -> np.ndarray:
    """Hidden channels from the least-squares fit of k POD modes to the active channels ``Xa`` (q, N)."""
    return (Upod[:, :k] @ np.linalg.lstsq(Upod[active][:, :k], Xa, rcond=None)[0])[hidden]


def static_ls_weights(G0: np.ndarray, active, hidden, reg: float = 1e-9) -> np.ndarray:
    """Weights W (q x h) of the static least-squares estimate y_h = W^T y_a (ridge 1e-9 tr(G_aa))."""
    Ga = G0[np.ix_(active, active)]
    return np.linalg.solve(Ga + reg * np.trace(Ga) * np.eye(len(active)), G0[np.ix_(active, hidden)])


def wiener_blocks(G: np.ndarray, rows, hidden_rows):
    """Covariance blocks (window x window, window x hidden) of the delay ridge regression."""
    return G[np.ix_(rows, rows)], G[np.ix_(rows, hidden_rows)]


def wiener_weights(GAA: np.ndarray, GAh: np.ndarray, rho: float) -> np.ndarray:
    """Delay ridge regression (finite-window Wiener filter), ridge rho tr(G_AA)/n_rows."""
    return np.linalg.solve(GAA + rho * np.trace(GAA) / len(GAA) * np.eye(len(GAA)), GAh)
