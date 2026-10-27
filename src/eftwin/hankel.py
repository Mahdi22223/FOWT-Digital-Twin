"""Delay embedding through Gram sums.

For a segment y_1..y_N (p channels) and delay depth d, the Hankel column h_k = [y_k; y_{k+1}; ...; y_{k+d-1}]
(lag-major, oldest first) is never stored. Only H1 H1^T and H2 H1^T (H1 = [h_0..h_{m-2}], H2 = [h_1..h_{m-1}])
are accumulated, chunk by chunk, so that no Hankel column spans two segments (load cases, runs or time
halves). Exact Hankel-DMD only needs these two matrices (``eftwin.dmd``).
"""
from __future__ import annotations

import numpy as np


def delay_gram(Y: np.ndarray, d: int, chunk: int = 20000):
    """Raw delay Gram sums of one segment ``Y`` (N, p).

    Returns ``(G, C, s1, s2, n)``: G = H1 H1^T, C = H2 H1^T, s1 and s2 the column sums of H1 and H2, n the
    number of column pairs. The chunk length only sets the order of floating-point summation
    (20000 everywhere, 15000 for the delay-depth sweep of Fig. 5).
    """
    N, P = Y.shape
    m = N - d + 1
    G = np.zeros((P * d, P * d)); C = np.zeros_like(G)
    s1 = np.zeros(P * d); s2 = np.zeros(P * d); n = 0
    for st in range(0, m - 1, chunk):
        sp = min(st + chunk, m - 1)
        L = sp - st + 1
        H = np.empty((P * d, L))
        for lag in range(d):
            H[lag * P:(lag + 1) * P] = Y[st + lag: st + lag + L].T
        G += H[:, :-1] @ H[:, :-1].T
        C += H[:, 1:] @ H[:, :-1].T
        s1 += H[:, :-1].sum(1); s2 += H[:, 1:].sum(1); n += L - 1
    return G, C, s1, s2, n


def gram_over_segments(segments, d: int, chunk: int = 20000):
    """H1 H1^T and H2 H1^T accumulated over several segments in one running sum.

    ``segments``: arrays of shape (p, N_i). This is the accumulation used for the identification (all OLCs,
    output correlation functions, FOCAL runs); it is algebraically identical to PyDMD's
    ``HankelDMD(exact=True, opt=False, tlsq_rank=0)`` applied segment-wise.
    """
    G = C = None
    for Y in segments:
        p, N = Y.shape
        m = N - d + 1
        if m < 2:
            continue
        if G is None:
            G = np.zeros((p * d, p * d)); C = np.zeros((p * d, p * d))
        start = 0
        while start < m - 1:
            stop = min(start + chunk, m - 1)
            L = stop - start + 1
            Hc = np.empty((p * d, L))
            for lag in range(d):
                Hc[lag * p:(lag + 1) * p, :] = Y[:, start + lag: start + lag + L]
            H1 = Hc[:, :-1]
            H2 = Hc[:, 1:]
            G += H1 @ H1.T
            C += H2 @ H1.T
            start = stop
    return G, C


def fold_stats(gram_list, raw_list, divide_by_n: bool = True) -> dict:
    """Centred and scaled delay covariances of a training set, from per-segment raw Gram sums.

    The channels are standardised with the mean and standard deviation (ddof = 0) of the training segments
    ``raw_list`` (each (N_i, p)); ``gram_list`` holds the matching ``delay_gram`` tuples. Returns
    ``dict(G, C, ev, U, mu, sd)`` with G = Sigma_0, C = Sigma_1 per column pair (``divide_by_n=False``: sums,
    as in the Fig. 5 sweep) and the eigen-pairs of G sorted in decreasing order.
    """
    X = np.vstack(raw_list)
    mu, sd = X.mean(0), X.std(0)
    G = sum(g[0] for g in gram_list); C = sum(g[1] for g in gram_list)
    s1 = sum(g[2] for g in gram_list); s2 = sum(g[3] for g in gram_list); n = sum(g[4] for g in gram_list)
    d = G.shape[0] // len(mu)
    mut, Dg = np.tile(mu, d), 1 / np.tile(sd, d)
    G = Dg[:, None] * (G - np.outer(mut, s1) - np.outer(s1, mut) + n * np.outer(mut, mut)) * Dg[None, :]
    C = Dg[:, None] * (C - np.outer(mut, s1) - np.outer(s2, mut) + n * np.outer(mut, mut)) * Dg[None, :]
    if divide_by_n:
        G = G / n
        C = C / n
    ev, U = np.linalg.eigh(G)
    o = np.argsort(ev)[::-1]
    return dict(G=G, C=C, ev=ev[o], U=U[:, o], mu=mu, sd=sd)


def delay_rows(active, p: int, d: int) -> np.ndarray:
    """Rows of the delay vector that belong to the active channels, for all d delays (lag-major)."""
    return np.concatenate([np.asarray(active) + lag * p for lag in range(d)])


def window_index(n: int, d: int) -> np.ndarray:
    """Sample indices of the delay windows of a record of length n: row j = window ending at sample j+d-1."""
    return np.arange(d)[None, :] + np.arange(n - d + 1)[:, None]
