"""Reference methods for operational modal analysis: covariance-driven SSI (SSI-COV) and FDD.

SSI-COV: block Toeplitz matrix of output correlations, T[a, b] = R_{i+a-b} (a, b = 0..i-1), its SVD, and for
every model order n the system matrix A = O_up^+ O_down of the observability matrix O = U_n S_n^(1/2).

Two pole-selection rules are used in the paper and both are kept:
* ``ssi_median_over_orders``: OpenFAST study (Table 3). For every order, the in-band pole (Im mu > 0,
  0 < zeta < 0.2) with the largest shape norm; frequency and damping are medians over the orders 20-80.
  No stabilisation criterion is applied.
* ``ssi_stabilised``: FOCAL study (Table 6). A pole is stable if a pole of the previous order lies within
  1 % in frequency and 5 % in damping with MAC > 0.95; per order the stable in-band pole with the largest
  shape norm is kept, and medians are taken over those orders.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import csd

from ..dmd import modal_params
from ..modal import mac


def ssi_poles(R: np.ndarray, p: int, i: int, orders, dt: float) -> list:
    """Poles for every model order: list of ``(order, f, zeta, shapes, mu)``.

    R: output correlation functions R_k (k = 0..>=2i-1), shape (K+1, p, p); i: number of block rows.
    """
    T = np.zeros((p * i, p * i))
    for a in range(i):
        for b in range(i):
            T[a * p:(a + 1) * p, b * p:(b + 1) * p] = R[i + a - b]
    Ut, St, _ = np.linalg.svd(T)
    out = []
    for nord in orders:
        Ob = Ut[:, :nord] * np.sqrt(St[:nord])[None, :]
        A = np.linalg.pinv(Ob[:-p]) @ Ob[p:]
        mu, Psi = np.linalg.eig(A)
        f, z, _ = modal_params(mu, dt)
        out.append((nord, f, z, Ob[:p] @ Psi, mu))
    return out


def ssi_median_over_orders(poles: list, band, max_damping: float = 0.2):
    """OpenFAST selection (Table 3): ``(f_median, zeta_median, shape, zeta_IQR)`` or None."""
    fs, zs, shapes = [], [], []
    for nord, f, z, shp, mu in poles:
        m = (mu.imag > 0) & (f > band[0]) & (f < band[1]) & (z > 0) & (z < max_damping)
        if m.any():
            j = np.where(m)[0][np.argmax(np.linalg.norm(shp[:, m], axis=0))]
            fs.append(f[j]); zs.append(z[j]); shapes.append(shp[:, j])
    if not fs:
        return None
    k = np.argsort(fs)[len(fs) // 2]
    return np.median(fs), np.median(zs), shapes[k], np.percentile(zs, [25, 75])


def _candidates(f, z, shp, mu, max_damping, f_range=None):
    return [(f[j], z[j], shp[:, j]) for j in range(len(mu))
            if mu[j].imag > 0 and 0 < z[j] < max_damping and (f_range is None or f_range[0] < f[j] < f_range[1])]


def _is_stable(pole, previous, df, dz, mac_min):
    return any(abs(pole[0] - q[0]) / q[0] < df and abs(pole[1] - q[1]) / q[1] < dz and mac(pole[2], q[2]) > mac_min
               for q in previous)


def ssi_stabilised(poles: list, band, df: float = 0.01, dz: float = 0.05, mac_min: float = 0.95,
                   max_damping: float = 0.3):
    """FOCAL selection (Table 6): ``(f_median, zeta_median, shape, number_of_stable_orders)`` or None."""
    prev, keep = None, []
    for nord, f, z, shp, mu in poles:
        cur = _candidates(f, z, shp, mu, max_damping)
        if prev is not None:
            stab = [p for p in cur if band[0] < p[0] < band[1] and _is_stable(p, prev, df, dz, mac_min)]
            if stab:
                keep.append(max(stab, key=lambda p: np.linalg.norm(p[2])))
        prev = cur
    if not keep:
        return None
    fs = np.array([k[0] for k in keep]); zs = np.array([k[1] for k in keep])
    k = np.argsort(fs)[len(fs) // 2]
    return np.median(fs), np.median(zs), keep[k][2], len(keep)


def ssi_stability_diagram(poles: list, f_range, df: float = 0.01, dz: float = 0.05, mac_min: float = 0.95,
                          max_damping: float = 0.3) -> list:
    """All poles in ``f_range`` from the second order on: ``(order, f, zeta, stable, shape)``."""
    out, prev = [], None
    for nord, f, z, shp, mu in poles:
        cur = _candidates(f, z, shp, mu, max_damping, f_range)
        if prev is not None:
            for p in cur:
                out.append((nord, p[0], p[1], _is_stable(p, prev, df, dz, mac_min), p[2]))
        prev = cur
    return out


# ------------------------------------------------------------------ FDD
def csd_sum(segments, fs: float, nperseg: int):
    """Cross-spectral density matrices summed over segments (each (N_i, p)): ``(f, S[f, p, p])``."""
    S = None
    for s in segments:
        f, Sij = csd(s[:, :, None], s[:, None, :], fs=fs, nperseg=nperseg, axis=0)
        S = Sij if S is None else S + Sij
    return f, S


def first_singular_values(S: np.ndarray) -> np.ndarray:
    """First singular value of the CSD matrix at every frequency (values only)."""
    return np.array([np.linalg.svd(S[i], compute_uv=False)[0] for i in range(S.shape[0])])


def first_singular(S: np.ndarray):
    """First singular vector and value of the CSD matrix at every frequency."""
    U1, s1 = [], []
    for k in range(S.shape[0]):
        u, sv, _ = np.linalg.svd(S[k])
        U1.append(u[:, 0]); s1.append(sv[0])
    return np.array(U1), np.array(s1)


def fdd_peak(segments, band, fs: float, seg_s: float):
    """Frequency and first singular vector at the peak of the first singular value within ``band``
    (FOCAL convention: CSD of ``seg_s``-second Welch segments summed over runs)."""
    f, S = csd_sum(segments, fs, int(seg_s * fs))
    m = np.where((f > band[0]) & (f < band[1]))[0]
    sv = np.array([np.linalg.svd(S[k], compute_uv=False)[0] for k in m])
    k = m[np.argmax(sv)]
    return f[k], np.linalg.svd(S[k])[0][:, 0]
