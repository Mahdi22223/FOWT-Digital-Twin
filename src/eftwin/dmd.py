"""Exact Hankel-DMD from delay Gram sums, modal parameters, modal energy and pole selection.

With G = H1 H1^T = U S^2 U^T and C = H2 H1^T (``eftwin.hankel``):
    A_tilde = U_r^T C U_r S_r^-2,   A_tilde W = W Lambda,   Phi = C U_r S_r^-2 W   (exact DMD modes)
This is algebraically identical to the SVD route of Algorithm 1 (U, Sigma, V from H1).

Identification is applied either to the response (response domain) or to the output correlation functions
R_k = E[y_{t+k} y_t^T], k >= 1 (correlation domain, the stochastic-Koopman form used for the damping).
"""
from __future__ import annotations

import numpy as np


def modal_params(eigs: np.ndarray, dt: float):
    """Frequency (Hz), damping ratio and continuous-time eigenvalue of discrete eigenvalues ``eigs``."""
    lam = np.log(eigs.astype(complex)) / dt
    f = np.abs(lam.imag) / (2 * np.pi)
    zeta = -lam.real / np.abs(lam)
    return f, zeta, lam


def sorted_eigh(G: np.ndarray):
    """Eigen-pairs of a symmetric Gram matrix in decreasing order of the eigenvalues."""
    ev, U = np.linalg.eigh(G)
    o = np.argsort(ev)[::-1]
    return ev[o], U[:, o]


def dmd_from_gram(G: np.ndarray, C: np.ndarray, r: int, eig=None):
    """Rank-r exact Hankel-DMD. Returns the discrete eigenvalues ``mu`` (r) and modes ``Phi`` (pd x r).

    ``eig``: precomputed ``sorted_eigh(G)`` (the result does not depend on whether it is reused).
    """
    ev, U = sorted_eigh(G) if eig is None else eig
    ev, U = ev[:r], U[:, :r]
    mu, W = np.linalg.eig((U.T @ C @ U) / ev[None, :])
    Phi = (C @ U / ev[None, :]) @ W
    return mu, Phi


def modal_energy(Phi: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Relative modal energy E_j = (Phi^+ G Phi^+H)_jj * ||phi_j||^2, used to pick one pole per band."""
    Pp = np.linalg.pinv(Phi)
    return np.real(np.einsum("ij,jk,ik->i", Pp, G, Pp.conj())) * np.sum(np.abs(Phi) ** 2, axis=0)


def dmd_poles(G: np.ndarray, C: np.ndarray, r: int, dt: float, eig=None):
    """Poles of the rank-r model: ``(f, zeta, Phi, E, mu)`` with full modes Phi (pd x r) and modal energy E."""
    mu, Phi = dmd_from_gram(G, C, r, eig)
    E = modal_energy(Phi, G)
    f, z, _ = modal_params(mu, dt)
    return f, z, Phi, E, mu


def pick(f, z, Phi, E, mu, band, shape_rows=slice(None), positive_damping: bool = False):
    """Pole of largest modal energy with Im(mu) > 0 and f in ``band``; returns ``(f, zeta, shape)`` or None.

    Conventions of the reported results:
    * OpenFAST (Table 3, Figs. 5-7): ``shape_rows = slice(0, p)`` (first delay block), no damping condition;
    * FOCAL (Table 6): ``shape_rows = slice(-p, None)`` (most recent delay block), ``positive_damping=True``.
    MAC and MPC are insensitive to this choice for modes with the delay (shift) structure.
    """
    m = (mu.imag > 0) & (f > band[0]) & (f < band[1])
    if positive_damping:
        m = m & (z > 0)
    if not m.any():
        return None
    j = np.where(m)[0][np.argmax(E[m])]
    return f[j], z[j], Phi[shape_rows, j]


def correlation_sum(segments, K: int) -> np.ndarray:
    """Output correlation functions R_k (k = 0..K) pooled over segments: sum_s sum_t y_{t+k} y_t^T / sum_s N_s.

    ``segments``: arrays (N_i, p). Convention of the OpenFAST study (correlation-domain DMD and SSI-COV).
    """
    p = segments[0].shape[1]
    R = np.zeros((K + 1, p, p)); n = 0
    for s in segments:
        N = len(s)
        for k in range(K + 1):
            R[k] += s[k:].T @ s[:N - k]
        n += N
    return R / n


def correlation(s: np.ndarray, K: int) -> np.ndarray:
    """Output correlation functions of one segment, R_k = sum_t y_{t+k} y_t^T / N (k = 0..K).

    Convention of the FOCAL study, where the pooled estimate is the mean of the per-run functions.
    """
    N = len(s)
    return np.array([s[k:].T @ s[:N - k] for k in range(K + 1)]) / N


def correlation_trajectories(R: np.ndarray) -> list[np.ndarray]:
    """Columns of R_k (k >= 1) as free-decay-like trajectories, one (p, K) array per reference channel."""
    return [R[1:, :, j].T for j in range(R.shape[1])]
