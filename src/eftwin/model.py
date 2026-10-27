"""High-level API: identify modes, fit one Hankel-DMD model, reconstruct hidden channels for any sensor layout.

    f, zeta, E, mu, Phi = identify_modes(segments, d=60, r=24, dt=0.05)    # correlation domain (damping)
    model = fit_hankel_dmd(segments, d=60, r=70)          # segments: list of (N_i, p) band-passed records
    y_hat = ProjectionEstimator(model, active=[8], alpha=0.01).estimate(Y)   # (N, p), causal
    y_hat = KalmanEstimator(model, active=[8]).estimate(Y)

All functions call the routines of ``eftwin.dmd``, ``eftwin.hankel`` and ``eftwin.estimators`` that produced
the reported results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dmd import correlation_sum, correlation_trajectories, dmd_from_gram, dmd_poles, modal_params
from .estimators import (delay_windows, kalman_gain, kalman_model, kalman_states, project_windows,
                         projection_estimate, sparse_svd, tikhonov_filter)
from .hankel import delay_gram, delay_rows, fold_stats, gram_over_segments
from .preprocess import standardise_segments


@dataclass
class HankelDMDModel:
    """Exact Hankel-DMD model identified from standardised training segments.

    Attributes: delay ``d``, rank ``r``, number of channels ``p``, fold statistics ``stats`` (G, C, ev, U,
    mu, sd), discrete eigenvalues ``lam`` (r) and modes ``Phi`` (pd x r).
    """
    d: int
    r: int
    p: int
    stats: dict
    lam: np.ndarray
    Phi: np.ndarray

    @property
    def Phi_L(self) -> np.ndarray:
        """Modes restricted to the most recent delay (p x r)."""
        return self.Phi[(self.d - 1) * self.p:self.d * self.p]

    def modal_parameters(self, dt: float):
        """Frequencies (Hz) and damping ratios of all r eigenvalues."""
        f, z, _ = modal_params(self.lam, dt)
        return f, z

    def standardise(self, Y: np.ndarray) -> np.ndarray:
        """(N, p) physical -> (p, N) standardised with the training mean and standard deviation."""
        return ((Y - self.stats["mu"]) / self.stats["sd"]).T

    def destandardise(self, Z: np.ndarray) -> np.ndarray:
        """(p, N) standardised -> (N, p) physical."""
        return self.stats["mu"] + self.stats["sd"] * Z.T


def identify_modes(segments, d: int, r: int, dt: float, lags: int | None = 400):
    """Operational modal identification (Section 4.1): z-score all segments jointly, then Hankel-DMD of the
    output correlation functions R_1..R_lags (``lags=None``: of the response itself).

    Returns ``(f, zeta, E, mu, Phi)`` of all r eigenvalues (frequency in Hz, damping ratio, modal energy,
    discrete eigenvalue, modes in the standardised space).
    """
    scaled, _ = standardise_segments({k: np.asarray(s) for k, s in enumerate(segments)})
    seg = list(scaled.values())
    trajectories = correlation_trajectories(correlation_sum(seg, lags)) if lags else [s.T for s in seg]
    G, C = gram_over_segments(trajectories, d)
    f, z, Phi, E, mu = dmd_poles(G, C, r, dt)
    return f, z, E, mu, Phi


def fit_hankel_dmd(segments, d: int, r: int) -> HankelDMDModel:
    """Identify the model from training segments (each (N_i, p); no delay window spans two segments)."""
    grams = [delay_gram(np.asarray(s), d) for s in segments]
    st = fold_stats(grams, segments)
    lam, Phi = dmd_from_gram(st["G"], st["C"], r, eig=(st["ev"], st["U"]))
    return HankelDMDModel(d=d, r=r, p=segments[0].shape[1], stats=st, lam=lam, Phi=Phi)


class ProjectionEstimator:
    """Tikhonov projection (Eq. tikhonov): one model for every sensor layout; only ``alpha`` is tuned."""

    def __init__(self, model: HankelDMDModel, active, alpha: float):
        self.model, self.active, self.alpha = model, np.asarray(active), alpha
        self.Uu, self.s, self.Vh = sparse_svd(model.Phi, delay_rows(self.active, model.p, model.d))
        self.filt = tikhonov_filter(self.s, alpha)

    @property
    def condition_number(self) -> float:
        """kappa = sigma_max / sigma_min of the sparse mode matrix."""
        return float(self.s[0] / self.s[-1])

    def estimate(self, Y: np.ndarray) -> np.ndarray:
        """Causal estimate of all p channels (N, p) from the active columns of ``Y`` (other columns ignored).
        The first d-1 samples use zero-padded windows (start-up)."""
        Z = self.model.standardise(Y)
        W = delay_windows(Z, self.active, self.model.d, pad=True)
        est = projection_estimate(project_windows(W, self.Uu), self.filt, self.Vh, self.model.Phi_L)
        return self.model.destandardise(est)


class KalmanEstimator:
    """Kalman form of the same model (stochastic extension); the rank is the only tuned parameter."""

    def __init__(self, model: HankelDMDModel, active):
        self.model, self.active = model, np.asarray(active)
        self.lam, self.PhiL, self.Q, self.R = kalman_model(model.stats, model.r, model.p)
        self.K, self.F = kalman_gain(self.lam, self.PhiL, self.Q, self.R, self.active)

    def estimate(self, Y: np.ndarray) -> np.ndarray:
        """Filtered estimate of all p channels (N, p) from the active columns of ``Y``."""
        Z = self.model.standardise(Y)
        X = kalman_states(self.K, self.F, Z[self.active])
        return self.model.destandardise((self.PhiL @ X).real)
