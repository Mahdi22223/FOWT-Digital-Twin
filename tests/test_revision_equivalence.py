"""The package reproduces the finalized revision code.

``tests/data/revision_golden.npz`` holds outputs of the original revision scripts (the code that produced the
reported results) for small synthetic inputs; every quantity is recomputed here with ``eftwin`` and compared.
Tolerances only absorb BLAS/LAPACK differences between machines (on the reference machine the agreement is exact).
"""
from pathlib import Path

import numpy as np
import pytest

from eftwin.baselines import oma
from eftwin.baselines.vs import gappy_pod, pod_basis, static_ls_weights, wiener_blocks, wiener_weights
from eftwin.dmd import correlation, correlation_sum, correlation_trajectories, dmd_from_gram, dmd_poles, pick
from eftwin.estimators import (delay_windows, kalman_gain, kalman_model, kalman_states, modal_amplitudes,
                               project_windows, projection_estimate, projection_forecast, sparse_svd, tikhonov_filter)
from eftwin.hankel import delay_gram, delay_rows, fold_stats, gram_over_segments
from eftwin.modal import mac
from eftwin.preprocess import standardise_segments
from eftwin.synthetic import tower_response

GOLD = np.load(Path(__file__).parent / "data" / "revision_golden.npz")
DT, P, D = 0.05, 6, 20
MODES = ((0.5, 0.02), (1.6, 0.02), (3.7, 0.015))
BANDS = {"m1": (0.3, 0.8), "m2": (1.2, 2.0), "m3": (3.0, 4.5)}


@pytest.fixture(scope="module")
def segs():
    return [tower_response(3000, DT, MODES, p=P, seed=s)[0] for s in (11, 12, 13)]


@pytest.fixture(scope="module")
def scaled(segs):
    z, _ = standardise_segments({i: s for i, s in enumerate(segs)})
    return list(z.values())


@pytest.fixture(scope="module")
def corr_gram(scaled):
    return gram_over_segments(correlation_trajectories(correlation_sum(scaled, 100)), D)


def test_identification_response_domain(scaled):
    G, C = gram_over_segments([s.T for s in scaled], D)
    for r in (6, 10):
        poles = dmd_poles(G, C, r, DT)
        for b, band in BANDS.items():
            f, z, shape = pick(*poles, band, slice(0, P))
            np.testing.assert_allclose([f, z], GOLD[f"resp_r{r}_{b}"], rtol=1e-8)
            assert mac(shape, GOLD[f"resp_r{r}_{b}_shape"]) > 1 - 1e-9


def test_identification_correlation_domain(corr_gram):
    poles = dmd_poles(*corr_gram, 8, DT)
    for b, band in BANDS.items():
        f, z, _ = pick(*poles, band, slice(0, P))
        np.testing.assert_allclose([f, z], GOLD[f"corr_r8_{b}"], rtol=1e-8)
        f, z, _ = pick(*poles, band, slice(-P, None), positive_damping=True)
        np.testing.assert_allclose([f, z], GOLD[f"focalpick_{b}"], rtol=1e-8)


def test_ssi_both_selection_rules(scaled):
    poles = oma.ssi_poles(correlation_sum(scaled, 40), P, 20, range(10, 32, 2), DT)
    Rm = np.mean([correlation(s, 60) for s in scaled], axis=0)
    poles_focal = oma.ssi_poles(Rm, P, 20, range(10, 32, 2), DT)
    for b, band in BANDS.items():
        f, z, _, iqr = oma.ssi_median_over_orders(poles, band)
        np.testing.assert_allclose([f, z, *iqr], GOLD[f"ssi_median_{b}"], rtol=1e-8)
        st = oma.ssi_stabilised(poles_focal, band)
        np.testing.assert_allclose([st[0], st[1], st[3]] if st else [np.nan] * 3, GOLD[f"ssi_stab_{b}"], rtol=1e-8)
        np.testing.assert_allclose(oma.fdd_peak(scaled, band, 1 / DT, 50)[0], GOLD[f"fdd_{b}"][0], rtol=1e-12)


def test_virtual_sensing_estimators(segs):
    train, test = segs[:2], segs[2]
    st = fold_stats([delay_gram(s, D) for s in train], train)
    G, C = st["G"], st["C"]
    LAST, S0, K_ = slice((D - 1) * P, D * P), 60, 300
    Xt = ((test - st["mu"]) / st["sd"]).T
    a = np.array([5]); hid = np.setdiff1d(np.arange(P), a)
    W = delay_windows(Xt, a, D, start=S0)
    rows = delay_rows(a, P, D)
    G0 = G[LAST, LAST]
    scale = np.abs(Xt).max()

    def close(name, x):
        np.testing.assert_allclose(x[:, :K_], GOLD[name], rtol=0, atol=1e-7 * scale, err_msg=name)
    close("gappyPOD", gappy_pod(pod_basis(G0), a, hid, Xt[a, S0:], 1))
    close("OLS", (Xt[a, S0:].T @ static_ls_weights(G0, a, hid)).T)
    close("FIR", (W @ wiener_weights(*wiener_blocks(G, rows, (D - 1) * P + hid), 1e-4)).T)
    lam, Phi = dmd_from_gram(G, C, 8, eig=(st["ev"], st["U"]))
    Uu, s, Vh = sparse_svd(Phi, rows)
    T1, filt = project_windows(W, Uu), tikhonov_filter(s, 0.01)
    close("DMD", projection_estimate(T1, filt, Vh, Phi[LAST][hid]))
    close("DMD_lead5", projection_forecast(modal_amplitudes(T1, filt, Vh), lam, Phi[LAST][hid], 5))
    np.testing.assert_allclose(s[0] / s[-1], GOLD["cond"][0], rtol=1e-6)
    lamk, PhiL, Q, R = kalman_model(st, 8, P)
    K, F = kalman_gain(lamk, PhiL, Q, R, a)
    X = kalman_states(K, F, Xt[a])
    close("KF", (PhiL[hid] @ X[:, S0:]).real)
    close("KF_lead5", (PhiL[hid] @ (lamk[:, None] ** 5 * X[:, S0:Xt.shape[1] - 5])).real)
