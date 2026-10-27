"""Pre-declared robustness tests under the time-blocked protocol (Table vs_robust, Section 4.7).

Random layouts with q <= 3 and layout L2; hyper-parameters fixed to the cross-fitted selections of the main
study (results/openfast/blocked_hyper.csv), no re-tuning.
  T1  600 s of a single OLC (first 600 s of the first half of OLC 5) -> second halves of the other 11 OLCs
  T2  sea-state shift: first halves of OLC 1-6 (Hs 1-2 m) -> second halves of OLC 7-12 (Hs 3-4 m), and reverse
  T3  5 % white noise (of each channel's std) on all channels in training and test; for every OLC c:
      first halves of the other 11 OLCs -> second half of c (noisy inputs, scored against noise-free truth)
Output: results/openfast/blocked_robust_<DIR>.csv. Run once per direction (the noise generator is seeded per run).
Usage: python scripts/openfast/robustness.py FA   (then SS) [--config configs/openfast.yaml]
"""
import argparse

import numpy as np
import pandas as pd

from eftwin.baselines.vs import static_ls_weights, wiener_blocks, wiener_weights
from eftwin.config import load_config
from eftwin.dmd import dmd_from_gram
from eftwin.estimators import (delay_windows, kalman_gain, kalman_model, kalman_states, project_windows,
                               projection_estimate, sparse_svd, tikhonov_filter)
from eftwin.hankel import delay_gram, delay_rows, fold_stats
from eftwin.studies.openfast import P, OpenFASTStudy
from eftwin.validation import r2_channels


def main(cfg, direction):
    S_ = OpenFASTStudy(cfg)
    vc, rc = cfg["virtual_sensing"], cfg["robustness"]
    D, HALF, SCORE0 = vc["delay"], vc["half"], vc["score_start"]
    LAST = slice((D - 1) * P, D * P)
    hyp = pd.read_csv(S_.results / "blocked_hyper.csv")
    masks = [m for m in S_.random_layouts() if len(m) <= rc["max_q"]] + [S_.layout(n) for n in rc["extra_layouts"]]
    rng = np.random.default_rng(rc["noise_seed"])
    group_b = set(vc["crossfit_group_b"])

    def hp(q, method, test):
        s = hyp[(hyp.direction == direction) & (hyp.q == q) & (hyp.method == method) & (hyp.group == int(test in group_b))]
        r, a, rho = s.hp.iloc[0].split("|")
        return (None if r == "nan" else int(float(r))), (None if a == "nan" else float(a)), \
               (None if rho == "nan" else float(rho))

    def fit(segs):
        return fold_stats([delay_gram(s, D) for s in segs], segs)

    def evaluate(st, tests, label):
        """tests: list of (test OLC, observed (N, P), truth (N, P))."""
        out = []
        G, C = st["G"], st["C"]
        cache_dmd, cache_kf = {}, {}
        for avail in masks:
            a = np.array(avail); hid = np.setdiff1d(np.arange(P), a); q = len(a)
            vrows = delay_rows(a, P, D)
            G0 = G[LAST, LAST]
            for test, Xo, Xt in tests:
                Zo = ((Xo - st["mu"]) / st["sd"]).T; Zt = ((Xt - st["mu"]) / st["sd"]).T
                T = Zt[hid, SCORE0:]
                Wobs = delay_windows(Zo, a, D, start=SCORE0)
                est = {}
                est["OLS"] = (Zo[a, SCORE0:].T @ static_ls_weights(G0, a, hid)).T
                _, _, rho = hp(q, "FIR", test)
                GAA, GAh = wiener_blocks(G, vrows, (D - 1) * P + hid)
                est["FIR"] = (Wobs @ wiener_weights(GAA, GAh, rho)).T
                r, alpha, _ = hp(q, "DMD", test)
                if r not in cache_dmd:
                    cache_dmd[r] = dmd_from_gram(G, C, r, eig=(st["ev"], st["U"]))[1]
                Phi = cache_dmd[r]
                Uu, s, Vh = sparse_svd(Phi, vrows)
                est["DMD"] = projection_estimate(project_windows(Wobs, Uu), tikhonov_filter(s, alpha), Vh, Phi[LAST][hid])
                rk, _, _ = hp(q, "KF", test)
                if rk not in cache_kf:
                    cache_kf[rk] = kalman_model(st, rk, P)
                lamk, PhiL, Q, Rm = cache_kf[rk]
                K, F = kalman_gain(lamk, PhiL, Q, Rm, a)
                est["KF"] = (PhiL[hid] @ kalman_states(K, F, Zo[a])[:, SCORE0:]).real
                for m, e in est.items():
                    out.append(dict(test_label=label, direction=direction, q=q, mask=" ".join(map(str, a)), method=m,
                                    test=test, R2=r2_channels(T, e).mean()))
        return out

    raw = S_.raw(direction)
    cases = list(S_.cases)
    H0 = {c: raw[c][:HALF] for c in cases}; H1 = {c: raw[c][HALF:2 * HALF] for c in cases}
    t1c = rc["t1_case"]
    lo, hi = rc["t2_groups"]
    rows = []
    rows += evaluate(fit([H0[t1c][:rc["t1_samples"]]]), [(c, H1[c], H1[c]) for c in cases if c != t1c], "T1: 600 s of one OLC")
    rows += evaluate(fit([H0[c] for c in lo]), [(c, H1[c], H1[c]) for c in hi], "T2: Hs 1-2 m -> Hs 3-4 m")
    rows += evaluate(fit([H0[c] for c in hi]), [(c, H1[c], H1[c]) for c in lo], "T2: Hs 3-4 m -> Hs 1-2 m")
    noisy = {c: raw[c] + rc["noise_level"] * raw[c].std(0) * rng.standard_normal(raw[c].shape) for c in cases}
    for c in cases:
        st = fit([noisy[k][:HALF] for k in cases if k != c])
        rows += evaluate(st, [(c, noisy[c][HALF:2 * HALF], H1[c])], "T3: 5% sensor noise")
    print(f"{direction} done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(S_.results / f"blocked_robust_{direction}.csv", index=False)
    print(df.groupby(["test_label", "direction", "q", "method"]).R2.median().unstack("method").round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("direction", choices=("FA", "SS"))
    ap.add_argument("--config", default="configs/openfast.yaml")
    a = ap.parse_args()
    main(load_config(a.config), a.direction)
