"""Time-series excerpts for Figs. 8 and 13 (time-blocked protocol, cross-fitted hyper-parameters): second half of
the test OLC, model trained on the first halves of the other 11 OLCs; static least squares, Hankel-DMD
projection and Kalman form.
Inputs: blocked_hyper.csv (summarise.py), cached half-record Gram sums (virtual_sensing.py).
Output: results/openfast/vs_excerpts.npy
Usage: python scripts/openfast/excerpts.py [--config configs/openfast.yaml]
"""
import argparse

import numpy as np
import pandas as pd

from eftwin.baselines.vs import static_ls_weights
from eftwin.config import load_config
from eftwin.dmd import dmd_from_gram
from eftwin.estimators import (delay_windows, kalman_gain, kalman_model, kalman_states, project_windows,
                               projection_estimate, sparse_svd, tikhonov_filter)
from eftwin.hankel import delay_rows, fold_stats
from eftwin.studies.openfast import P, OpenFASTStudy


def fold_model(S_, d, test, h, raw):
    """Fold statistics of the model that reconstructs half h of OLC ``test`` (other half of the other OLCs)."""
    HALF = S_.cfg["virtual_sensing"]["half"]
    gr = [S_.half_gram(d, c, 1 - h, raw) for c in S_.cases if c != test]
    return fold_stats(gr, [raw[c][(1 - h) * HALF:(2 - h) * HALF] for c in S_.cases if c != test])


def hyper(S_, hyp, d, q, method, test):
    s = hyp[(hyp.direction == d) & (hyp.q == q) & (hyp.method == method)
            & (hyp.group == int(test in S_.cfg["virtual_sensing"]["crossfit_group_b"]))].hp.iloc[0]
    r, a, rho = s.split("|")
    return (None if r == "nan" else int(float(r))), (None if a == "nan" else float(a))


def estimates(S_, hyp, st, Xt, a, hid, d, test):
    vc = S_.cfg["virtual_sensing"]
    D, SCORE0 = vc["delay"], vc["score_start"]
    LAST = slice((D - 1) * P, D * P)
    G, C = st["G"], st["C"]
    vrows = delay_rows(a, P, D)
    Wobs = delay_windows(Xt, a, D, start=SCORE0)
    out = {}
    G0 = G[LAST, LAST]
    out["Static OLS"] = (Xt[a, SCORE0:].T @ static_ls_weights(G0, a, hid)).T
    r, alpha = hyper(S_, hyp, d, len(a), "DMD", test)
    lam, Phi = dmd_from_gram(G, C, r, eig=(st["ev"], st["U"]))
    Uu, s, Vh = sparse_svd(Phi, vrows)
    out["Hankel-DMD (projection)"] = projection_estimate(project_windows(Wobs, Uu), tikhonov_filter(s, alpha), Vh,
                                                         Phi[LAST][hid])
    rk, _ = hyper(S_, hyp, d, len(a), "KF", test)
    lamk, PhiL, Q, Rm = kalman_model(st, rk, P)
    K, F = kalman_gain(lamk, PhiL, Q, Rm, a)
    Xk = kalman_states(K, F, Xt[a])
    out["Hankel-DMD (Koopman-Kalman)"] = (PhiL[hid] @ Xk[:, SCORE0:]).real
    return out


def main(cfg):
    S_ = OpenFASTStudy(cfg)
    vc, ec = cfg["virtual_sensing"], cfg["excerpts"]
    HALF, SCORE0 = vc["half"], vc["score_start"]
    hyp = pd.read_csv(S_.results / "blocked_hyper.csv")
    X0, XL = ec["start"], ec["length"]
    EXC = {}
    d = "FA"
    raw = S_.raw(d)
    for job in ec["jobs"]:
        name, test = job["name"], job["test"]
        a = np.array(S_.layout(job["layout"])); hid = np.setdiff1d(np.arange(P), a)
        st = fold_model(S_, d, test, 1, raw)
        Xt = ((raw[test][HALF:2 * HALF] - st["mu"]) / st["sd"]).T
        est = estimates(S_, hyp, st, Xt, a, hid, d, test)
        T = Xt[hid, SCORE0:]
        EXC[f"{d}|{name}|{test}"] = dict(hid=hid, mu=st["mu"][hid], sd=st["sd"][hid],
                                        truth=T[:, X0:X0 + XL].astype(np.float32),
                                        **{m: e[:, X0:X0 + XL].astype(np.float32) for m, e in est.items()})
        print(name, "done", flush=True)
    np.save(S_.results / "vs_excerpts.npy", EXC, allow_pickle=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/openfast.yaml")
    main(load_config(ap.parse_args().config))
