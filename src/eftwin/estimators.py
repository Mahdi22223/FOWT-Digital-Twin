"""The two realizations of the Hankel-DMD virtual sensor (paper Sections 2.4.2-2.4.3).

Projection (deterministic estimator of the framework). At sample k the window of the q active channels,
y_obs = S [y_{k-d+1}; ...; y_k] (qd), gives the modal amplitudes by Tikhonov-regularised inversion of the
sparse mode matrix Phi_sparse = U_s Sigma_s V_s^H,
    b_k = V_s diag(sigma_i / (sigma_i^2 + alpha^2 sigma_max^2)) U_s^H y_obs,
and the hidden channels follow from the most recent delay block, y_k = Re(Phi_L b_k); the open-loop
forecast is y_{k+h|k} = Re(Phi_L Lambda^h b_k).

Kalman form (stochastic extension). The same (Lambda, Phi) as a state-space model in modal coordinates,
x_{k+1} = Lambda x_k + w_k, y_k = Phi_L x_k + v_k, with Q and R from the training covariances, and the
steady-state Kalman filter x_k = (I - K C_a) Lambda x_{k-1} + K y_k. Implementation details of the reported
results: eigenvalues with |lambda| > 0.9995 are moved radially to 0.9995, Q is symmetrised and its
eigenvalues floored at 1e-10 max(eig Q), and R_aa is regularised by 1e-6 tr(R)/p.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.signal import lfilter

from .dmd import dmd_from_gram


# ------------------------------------------------------------------ delay windows
def delay_windows(X: np.ndarray, active, d: int, start: int = 0, pad: bool = False) -> np.ndarray:
    """Delay windows of the active channels, one row per estimate (lag-major, oldest first).

    X: standardised record (p, N). ``pad=False``: windows ending at samples start..N-1 (start >= d-1);
    ``pad=True``: one window per sample, the first d-1 windows completed with zeros.
    """
    a = np.asarray(active)
    q = len(a)
    if pad:
        Xp = np.concatenate([np.zeros((X.shape[0], d - 1)), X], axis=1)
        idx = np.arange(d)[None, :] + np.arange(X.shape[1])[:, None]
        return Xp[a][:, idx].transpose(1, 2, 0).reshape(-1, q * d)
    idx = np.arange(d)[None, :] + np.arange(X.shape[1] - d + 1)[:, None]
    return X[a][:, idx[start - (d - 1):]].transpose(1, 2, 0).reshape(-1, q * d)


# ------------------------------------------------------------------ projection form
def sparse_svd(Phi: np.ndarray, rows: np.ndarray):
    """SVD of the sparse mode matrix Phi_sparse = Phi[rows] (thin)."""
    return np.linalg.svd(Phi[rows], full_matrices=False)


def tikhonov_filter(s: np.ndarray, alpha: float) -> np.ndarray:
    """Filter factors sigma_i / (sigma_i^2 + (alpha sigma_max)^2)."""
    return s / (s ** 2 + (alpha * s[0]) ** 2)


def project_windows(W: np.ndarray, Uu: np.ndarray) -> np.ndarray:
    """W conj(U_s), evaluated with real matrix products (rows of W: delay windows)."""
    return W @ Uu.real - 1j * (W @ Uu.imag)


def projection_estimate(T1: np.ndarray, filt: np.ndarray, Vh: np.ndarray, PhiL_rows: np.ndarray) -> np.ndarray:
    """Re(Phi_L[rows] b_k) for all windows; returns (len(rows), n_windows)."""
    return ((T1 * filt[None, :]) @ (Vh.conj() @ PhiL_rows.T)).real.T


def modal_amplitudes(T1: np.ndarray, filt: np.ndarray, Vh: np.ndarray) -> np.ndarray:
    """Modal amplitudes b_k (one row per window)."""
    return (T1 * filt[None, :]) @ Vh.conj()


def projection_forecast(B: np.ndarray, lam: np.ndarray, PhiL_rows: np.ndarray, lead: int) -> np.ndarray:
    """Open-loop forecast Re(Phi_L Lambda^h b_k), h = ``lead`` samples, for windows 0..n-1-lead."""
    return ((B[:-lead] * lam[None, :] ** lead) @ PhiL_rows.T).real.T


# ------------------------------------------------------------------ Kalman form
def kalman_model(st: dict, r: int, p: int, clip: float = 0.9995, q_floor: float = 1e-10):
    """Modal state-space model (lambda, Phi_L, Q, R) of rank r from the fold statistics ``st`` (``fold_stats``)."""
    G, C = st["G"], st["C"]
    lam, Phi = dmd_from_gram(G, C, r, eig=(st["ev"], st["U"]))
    big = np.abs(lam) > clip
    lam[big] = lam[big] / np.abs(lam[big]) * clip
    Psi = np.linalg.pinv(Phi)
    Sxx = Psi @ G @ Psi.conj().T
    Sx1x = Psi @ C @ Psi.conj().T
    L = np.diag(lam)
    Q = Sxx - L @ Sx1x.conj().T - Sx1x @ L.conj().T + L @ Sxx @ L.conj().T
    Q = 0.5 * (Q + Q.conj().T)
    wq, vq = np.linalg.eigh(Q)
    Q = (vq * np.clip(wq, q_floor * wq.max(), None)) @ vq.conj().T
    d = G.shape[0] // p
    last = slice((d - 1) * p, d * p)
    M = np.zeros((p, p * d)); M[:, last] = np.eye(p)
    E = M - Phi[last] @ Psi
    Rm = (E @ G @ E.conj().T).real
    Rm = 0.5 * (Rm + Rm.T)
    return lam, Phi[last], Q, Rm


def kalman_gain(lam, PhiL, Q, Rm, active, r_reg: float = 1e-6):
    """Steady-state gain K (DARE) and closed-loop matrix F = (I - K C_a) Lambda for the active channels."""
    Ca = PhiL[active]
    Ra = Rm[np.ix_(active, active)] + r_reg * np.trace(Rm) / Rm.shape[0] * np.eye(len(active))
    A = np.diag(lam)
    Pp = solve_discrete_are(A.conj().T, Ca.conj().T, Q, Ra)
    K = Pp @ Ca.conj().T @ np.linalg.inv(Ca @ Pp @ Ca.conj().T + Ra)
    return K, (np.eye(len(lam)) - K @ Ca) @ A


def kalman_states_loop(K, F, Yobs):
    """Filtered modal states x_k = F x_{k-1} + K y_k by direct recursion (Yobs: (q, N))."""
    Ky = K @ Yobs
    x = np.zeros(F.shape[0], complex)
    X = np.empty((F.shape[0], Yobs.shape[1]), complex)
    for k in range(Yobs.shape[1]):
        x = F @ x + Ky[:, k]
        X[:, k] = x
    return X


def kalman_states(K, F, Yobs, cond_max: float = 1e8):
    """Filtered modal states via the eigen-decomposition of F (first-order recursions); falls back to the
    direct recursion when the eigenvector matrix is ill-conditioned."""
    ph, VF = np.linalg.eig(F)
    if np.linalg.cond(VF) > cond_max:
        return kalman_states_loop(K, F, Yobs)
    U = np.linalg.solve(VF, K @ Yobs)
    Z = np.empty_like(U)
    for i in range(len(ph)):
        Z[i] = lfilter([1.0], [1.0, -ph[i]], U[i])
    return VF @ Z
