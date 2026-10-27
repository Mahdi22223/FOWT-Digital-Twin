"""Mathematical properties of the core routines (synthetic data, seconds)."""
import numpy as np

from eftwin import KalmanEstimator, ProjectionEstimator, fit_hankel_dmd, identify_modes
from eftwin.baselines.vs import wiener_blocks, wiener_weights
from eftwin.dmd import dmd_from_gram, modal_params
from eftwin.estimators import kalman_gain, kalman_model, kalman_states, kalman_states_loop, sparse_svd, tikhonov_filter
from eftwin.hankel import delay_gram, delay_rows, fold_stats, gram_over_segments
from eftwin.modal import mac
from eftwin.synthetic import tower_response
from eftwin.validation import blocked_folds, combine, r2_channels, sufficient_stats

DT = 0.05
MODES = ((0.5, 0.02), (1.6, 0.02), (3.7, 0.015))


def hankel(Y, d):
    """Explicit Hankel matrix of a (N, p) record, columns h_k = [y_k; ...; y_{k+d-1}]."""
    N, p = Y.shape
    return np.vstack([Y[lag:N - d + 1 + lag].T for lag in range(d)])


def test_gram_sums_equal_explicit_hankel_products():
    segs = [tower_response(700, DT, MODES, p=3, seed=s)[0] for s in (1, 2)]
    d = 7
    G = sum(hankel(s, d)[:, :-1] @ hankel(s, d)[:, :-1].T for s in segs)
    C = sum(hankel(s, d)[:, 1:] @ hankel(s, d)[:, :-1].T for s in segs)
    Gs, Cs = gram_over_segments([s.T for s in segs], d, chunk=100)          # several chunks per segment
    np.testing.assert_allclose(Gs, G, rtol=1e-12); np.testing.assert_allclose(Cs, C, rtol=1e-12)
    g = [delay_gram(s, d, chunk=64) for s in segs]
    np.testing.assert_allclose(g[0][0] + g[1][0], G, rtol=1e-12)
    np.testing.assert_allclose(g[0][2], hankel(segs[0], d)[:, :-1].sum(1), rtol=1e-12)


def test_gram_route_equals_svd_route_exact_dmd():
    Y = tower_response(2000, DT, MODES, p=4, seed=3)[0]
    d, r = 12, 8
    H = hankel(Y, d); H1, H2 = H[:, :-1], H[:, 1:]
    U, S, Vh = np.linalg.svd(H1, full_matrices=False)
    U, S, V = U[:, :r], S[:r], Vh[:r].T
    mu_svd, W = np.linalg.eig(U.T @ H2 @ V / S)
    Phi_svd = H2 @ V / S @ W
    mu, Phi = dmd_from_gram(H1 @ H1.T, H2 @ H1.T, r)
    o1, o2 = np.argsort(np.angle(mu)), np.argsort(np.angle(mu_svd))
    np.testing.assert_allclose(mu[o1], mu_svd[o2], rtol=1e-8, atol=1e-10)
    for i, j in zip(o1, o2):
        assert mac(Phi[:, i], Phi_svd[:, j]) > 1 - 1e-8


def test_correlation_domain_recovers_modes():
    segs = [tower_response(6000, DT, MODES, seed=s)[0] for s in (1, 2, 3)]
    f, z, E, mu, _ = identify_modes(segs, d=60, r=10, dt=DT, lags=400)
    for (f0, z0), (lo, hi) in zip(MODES, ((0.3, 0.8), (1.2, 2.0), (3.0, 4.5))):
        m = np.where((mu.imag > 0) & (f > lo) & (f < hi))[0]
        j = m[np.argmax(E[m])]
        assert abs(f[j] - f0) / f0 < 0.01
        assert abs(z[j] - z0) < 0.006


def test_modal_params_of_known_eigenvalue():
    lam = (-0.03 + 1j) * 2 * np.pi * 0.5 / np.sqrt(1 + 0.03 ** 2)
    f, z, _ = modal_params(np.exp(np.array([lam]) * DT), DT)
    np.testing.assert_allclose([f[0], z[0]], [0.5 / np.sqrt(1 + 0.03 ** 2), 0.03 / np.sqrt(1 + 0.03 ** 2)], rtol=1e-10)


def test_tikhonov_filter_equals_regularised_least_squares():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 8)) + 1j * rng.standard_normal((40, 8)); y = rng.standard_normal(40)
    Uu, s, Vh = np.linalg.svd(A, full_matrices=False)
    alpha = 0.05
    b = Vh.conj().T @ (tikhonov_filter(s, alpha) * (Uu.conj().T @ y))
    lam = alpha * s[0]
    b_ls = np.linalg.lstsq(np.vstack([A, lam * np.eye(8)]), np.concatenate([y, np.zeros(8)]), rcond=None)[0]
    np.testing.assert_allclose(b, b_ls, rtol=1e-10, atol=1e-12)


def test_kalman_eigen_recursion_equals_direct_recursion():
    segs = [tower_response(2000, DT, MODES, seed=s)[0] for s in (1, 2)]
    st = fold_stats([delay_gram(s, 20) for s in segs], segs)
    lam, PhiL, Q, R = kalman_model(st, 8, 6)
    K, F = kalman_gain(lam, PhiL, Q, R, [5])
    Z = ((segs[0] - st["mu"]) / st["sd"]).T[[5]]
    np.testing.assert_allclose(kalman_states(K, F, Z), kalman_states_loop(K, F, Z), rtol=1e-9, atol=1e-10)
    assert np.all(np.abs(lam) <= 0.9995 + 1e-12)


def test_wiener_weights_equal_ridge_regression_on_windows():
    segs = [tower_response(1500, DT, MODES, p=3, seed=s)[0] for s in (4, 5)]
    d, a, h = 10, np.array([2]), np.array([0, 1])
    st = fold_stats([delay_gram(s, d) for s in segs], segs)
    rows = delay_rows(a, 3, d)
    W = wiener_weights(*wiener_blocks(st["G"], rows, (d - 1) * 3 + h), 1e-3)
    H = np.hstack([hankel((s - st["mu"]) / st["sd"], d)[:, :-1] for s in segs])
    A, B = H[rows].T, H[(d - 1) * 3 + h].T
    n = H.shape[1]
    GAA = A.T @ A / n
    W_ls = np.linalg.solve(GAA + 1e-3 * np.trace(GAA) / len(rows) * np.eye(len(rows)), A.T @ B / n)
    np.testing.assert_allclose(W, W_ls, rtol=1e-8, atol=1e-10)


def test_api_estimators_reconstruct_hidden_channels():
    train = [tower_response(6000, DT, MODES, seed=s)[0] for s in (1, 2, 3)]
    test = tower_response(6000, DT, MODES, seed=4)[0]
    model = fit_hankel_dmd(train, d=60, r=10)
    hidden = [0, 1, 2, 3, 4]
    for est in (ProjectionEstimator(model, [5], alpha=0.01), KalmanEstimator(model, [5])):
        y = est.estimate(test)
        assert y.shape == test.shape
        assert r2_channels(test[60:, hidden].T, y[60:, hidden].T).mean() > 0.75
    assert ProjectionEstimator(model, [5], alpha=0.01).condition_number > 1
    # the projection is causal: changing future samples does not change past estimates
    t2 = test.copy(); t2[3000:] += 1.0
    e1 = ProjectionEstimator(model, [5], alpha=0.01).estimate(test)
    e2 = ProjectionEstimator(model, [5], alpha=0.01).estimate(t2)
    np.testing.assert_array_equal(e1[:3000], e2[:3000])
    _, s, _ = sparse_svd(model.Phi, delay_rows([5], 6, 60))
    assert s[0] / s[-1] == ProjectionEstimator(model, [5], alpha=0.01).condition_number


def test_blocked_folds_never_train_on_the_test_record_or_interval():
    folds = list(blocked_folds([1, 2, 3, 4]))
    assert len(folds) == 8
    for test, h, train in folds:
        assert all(c != test and hh == 1 - h for c, hh in train) and len(train) == 3


def test_combined_statistics_equal_direct_r2():
    rng = np.random.default_rng(1)
    T = rng.standard_normal((3, 400)); E = T + 0.3 * rng.standard_normal((3, 400))
    import pandas as pd
    halves = pd.DataFrame([sufficient_stats(T[:, :200], E[:, :200]), sufficient_stats(T[:, 200:], E[:, 200:])])
    r2, _ = combine(halves)
    np.testing.assert_allclose(r2, r2_channels(T, E), rtol=1e-12)
