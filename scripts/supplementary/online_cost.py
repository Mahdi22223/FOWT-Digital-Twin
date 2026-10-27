"""Online cost per sample of the two virtual-sensor realizations (Section 2.4.3; response to Reviewer 4).

Projection: one precomputed (h x qd) matrix-vector product per sample. Kalman form: x <- F x + K y (r x r and
r x q products) and y_h = Re(Phi_L,h x). Timed with random matrices of the given dimensions (machine dependent).
Usage: python scripts/supplementary/online_cost.py [--q 1] [--d 60] [--p 18] [--ranks 50 120 250]
"""
import argparse
import time

import numpy as np


def per_call(fn, n=20000):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main(q, d, p, ranks):
    rng = np.random.default_rng(0)
    h = p - q
    for r in ranks:
        M = rng.standard_normal((h, q * d))                       # Re(Phi_L,h V diag(f) U^H) precomputed
        w = rng.standard_normal(q * d)
        F = (rng.standard_normal((r, r)) + 1j * rng.standard_normal((r, r))) / (2 * np.sqrt(r))   # stable
        K = rng.standard_normal((r, q)) + 1j * rng.standard_normal((r, q))
        C = rng.standard_normal((h, r)) + 1j * rng.standard_normal((h, r))
        x = np.zeros(r, complex); y = rng.standard_normal(q)

        def kalman():
            nonlocal x
            x = F @ x + K @ y
            return (C @ x).real
        print(f"r = {r:4d}: projection {1e6 * per_call(lambda: M @ w):6.2f} us/sample, "
              f"Kalman form {1e6 * per_call(kalman):6.2f} us/sample (q = {q}, d = {d}, p = {p})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--q", type=int, default=1)
    ap.add_argument("--d", type=int, default=60)
    ap.add_argument("--p", type=int, default=18)
    ap.add_argument("--ranks", type=int, nargs="+", default=[50, 120, 250])
    a = ap.parse_args()
    main(a.q, a.d, a.p, a.ranks)
