"""Noise-floor and Lyapunov-horizon utilities."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.extmath import randomized_svd

from .data_io import load_direction_matrix


def hankel_singular_values(case_files, acc_cols, mom_cols, *, dt=0.0125, low_cutoff=0.25, high_cutoff=5.0, filter_order=4, hankel_d=60, downsample_factor=5, n_components=200, random_state=100):
    X = load_direction_matrix(case_files, acc_cols, mom_cols, low_cutoff=low_cutoff, high_cutoff=high_cutoff, fs=1.0/dt, filter_order=filter_order)
    X_scaled = StandardScaler().fit_transform(X.T).T
    X_sub = X_scaled[:, ::downsample_factor]
    n_features, n_time = X_sub.shape
    H_rows = n_features * hankel_d
    H_cols = n_time - hankel_d + 1
    H = np.zeros((H_rows, H_cols))
    for i in range(hankel_d):
        H[i*n_features:(i+1)*n_features, :] = X_sub[:, i:i+H_cols]
    _, S, _ = randomized_svd(H, n_components=n_components, random_state=random_state)
    cutoff = np.median(S) * 2.858
    optimal_rank = int(np.sum(S > cutoff))
    return S, optimal_rank


def embed_series(series: np.ndarray, m: int, d: int) -> np.ndarray:
    N = len(series)
    Y = np.zeros((N - (m - 1) * d, m))
    for i in range(m):
        Y[:, i] = series[i*d:i*d + Y.shape[0]]
    return Y


def calculate_lyapunov(data_series, *, dt=0.0125, embedding_dim=10, time_delay=10, max_t_steps=200, min_separation=100):
    X_emb = embed_series(np.asarray(data_series), m=embedding_dim, d=time_delay)
    n_points = X_emb.shape[0]
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(X_emb)
    _, indices = nbrs.kneighbors(X_emb)
    nearest_idx = indices[:, 1]
    valid_pairs = [(i, j) for i, j in enumerate(nearest_idx) if abs(i - j) > min_separation]
    divergence = np.zeros(max_t_steps)
    for k in range(max_t_steps):
        vals = []
        for i, j in valid_pairs:
            if i + k < n_points and j + k < n_points:
                dist = np.linalg.norm(X_emb[i+k] - X_emb[j+k])
                if dist > 1e-10:
                    vals.append(np.log(dist))
        divergence[k] = np.mean(vals) if vals else np.nan
    t_axis = np.arange(max_t_steps) * dt
    fit_region = int(1.0 / dt)
    coeffs = np.polyfit(t_axis[:fit_region], divergence[:fit_region], 1)
    lambda_max = float(coeffs[0])
    lyapunov_time = float(1.0 / lambda_max) if lambda_max > 0 else 999.0
    return {"t": t_axis, "divergence": divergence, "lambda": lambda_max, "lyapunov_time": lyapunov_time, "fit_coeffs": coeffs}
