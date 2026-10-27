"""Surrogate-data test of the largest-Lyapunov-exponent estimate (response to Reviewer 4, comment 4).

Rosenstein's estimator (nolds.lyap_r) is applied to the nacelle fore-aft acceleration (0.25-5 Hz, 20 Hz) and to
phase-randomised surrogates with the same power spectrum (linear Gaussian processes, no deterministic chaos).
If the data's estimate lies within the surrogate distribution, the positive exponent does not indicate chaos.
Requires: pip install nolds
Usage: python scripts/supplementary/lyapunov_surrogate.py [--config configs/openfast.yaml]
"""
import argparse

import nolds
import numpy as np

from eftwin.config import load_config
from eftwin.studies.openfast import OpenFASTStudy


def main(cfg, start=12000, length=6000):
    S_ = OpenFASTStudy(cfg)
    FS = S_.fs
    rng = np.random.default_rng(3)
    N0, N = start, length                    # 300 s segment starting 600 s after the transient

    def surrogate(x):
        X = np.fft.rfft(x - x.mean())
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi, len(X))); ph[0] = 1; ph[-1] = 1
        return np.fft.irfft(np.abs(X) * ph, len(x))
    kw = dict(emb_dim=10, lag=2, min_tsep=40, trajectory_len=40, fit="poly")
    for c in (2, 7, 11):
        x = S_.data.filtered(c, "FA")[:, 8][N0:N0 + N]
        lam = nolds.lyap_r(x, **kw) * FS
        sur = [nolds.lyap_r(surrogate(x), **kw) * FS for _ in range(10)]
        print(f"OLC {c}: lambda_max data = {lam:.3f} 1/s | surrogates: mean {np.mean(sur):.3f}, "
              f"range {np.min(sur):.3f}-{np.max(sur):.3f} 1/s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/openfast.yaml")
    ap.add_argument("--start", type=int, default=12000, help="first sample after the transient (default 600 s)")
    ap.add_argument("--length", type=int, default=6000, help="number of samples (default 300 s)")
    a = ap.parse_args()
    main(load_config(a.config), a.start, a.length)
