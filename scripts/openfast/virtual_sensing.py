"""Virtual-sensing study, time-blocked leave-one-OLC-out (Sections 4.4-4.8).

The 12 OLCs share one turbulent wind field and differ in sea state. Every record (after the 300 s transient) is
split into two halves of 1650 s; fold = (test OLC c, half h), trained on half 1-h of the 11 other OLCs, so that
neither the test OLC nor the test time interval is seen in training (24 folds per direction). The first 60 s
of every test half are not scored.

Estimators (all from the same training covariance):
  gappyPOD  baseline, static, one model for all layouts, k = min(q, 5)
  OLS       baseline, static least squares, per layout
  FIR       baseline, delay ridge regression = finite-window Wiener filter, per layout (rho cross-fitted)
  DMD       Hankel-DMD projection (Tikhonov), one model for all layouts ((r, alpha) cross-fitted)
  KF        Hankel-DMD Kalman form, one model for all layouts (r cross-fitted)
Forecasts (DMD and KF) for layouts L1, L2 and Case B/C at the configured leads.
Per hidden channel the sufficient statistics are stored (see eftwin.validation).

Output: results/openfast/intermediate/blocked_records_<DIR>.pkl (list of dicts; not version-controlled)
Usage: python scripts/openfast/virtual_sensing.py FA   (then SS) [--config configs/openfast.yaml]
"""
import argparse
import pickle
import time

import numpy as np

from eftwin.baselines.vs import gappy_pod, pod_basis, static_ls_weights, wiener_blocks, wiener_weights
from eftwin.config import ensure_dir, load_config
from eftwin.dmd import dmd_from_gram
from eftwin.estimators import (delay_windows, kalman_gain, kalman_model, kalman_states, modal_amplitudes,
                               project_windows, projection_estimate, projection_forecast, sparse_svd, tikhonov_filter)
from eftwin.hankel import delay_rows, fold_stats
from eftwin.studies.openfast import P, OpenFASTStudy
from eftwin.validation import blocked_folds, sufficient_stats as stats


def main(cfg, directions):
    S_ = OpenFASTStudy(cfg)
    vc = cfg["virtual_sensing"]
    D, HALF, SCORE0 = vc["delay"], vc["half"], vc["score_start"]
    LAST = slice((D - 1) * P, D * P)
    layouts, forecast = S_.vs_layouts(), S_.forecast_layouts()
    out_dir = ensure_dir(S_.results / "intermediate")
    out = out_dir / f"blocked_records_{'_'.join(directions)}.pkl"
    records, t0 = [], time.time()
    for direction in directions:
        raw = S_.raw(direction)
        hg = {(c, h): S_.half_gram(direction, c, h, raw) for c in S_.cases for h in (0, 1)}
        for test, h, train in blocked_folds(list(S_.cases)):
            segs = [raw[c][hh * HALF:(hh + 1) * HALF] for c, hh in train]
            st = fold_stats([hg[k] for k in train], segs)
            G, C = st["G"], st["C"]
            Xt = ((raw[test][h * HALF:(h + 1) * HALF] - st["mu"]) / st["sd"]).T
            N = Xt.shape[1]
            G0 = G[LAST, LAST]
            Upod = pod_basis(G0)
            dmd = {}
            for r in vc["projection_ranks"]:
                lam, Phi = dmd_from_gram(G, C, r, eig=(st["ev"], st["U"]))
                dmd[r] = (Phi, lam)
            kfm = {r: kalman_model(st, r, P) for r in vc["kalman_ranks"]}
            base = dict(direction=direction, test=test, half=h)
            for avail in layouts:
                a = np.array(avail); hid = np.setdiff1d(np.arange(P), a); q = len(a)
                key = dict(base, mask=" ".join(map(str, a)), q=q)
                T = Xt[hid, SCORE0:]
                Wobs = delay_windows(Xt, a, D, start=SCORE0)
                vrows = delay_rows(a, P, D)
                # baselines
                k = min(q, vc["gappy_pod_max_modes"])
                records.append(dict(key, method="gappyPOD", **stats(T, gappy_pod(Upod, a, hid, Xt[a, SCORE0:], k))))
                Wols = static_ls_weights(G0, a, hid)
                records.append(dict(key, method="OLS", **stats(T, (Xt[a, SCORE0:].T @ Wols).T)))
                GAA, GAh = wiener_blocks(G, vrows, (D - 1) * P + hid)
                for rho in vc["wiener_rhos"]:
                    Wr = wiener_weights(GAA, GAh, rho)
                    records.append(dict(key, method="FIR", rho=rho, **stats(T, (Wobs @ Wr).T)))
                # Hankel-DMD projection: windows projected once per rank, then filtered per alpha
                for r, (Phi, lam) in dmd.items():
                    Uu, s, Vh = sparse_svd(Phi, vrows)
                    T1 = project_windows(Wobs, Uu)
                    for alpha in vc["alphas"]:
                        f = tikhonov_filter(s, alpha)
                        est = projection_estimate(T1, f, Vh, Phi[LAST][hid])
                        records.append(dict(key, method="DMD", r=r, alpha=alpha, sigma_min=s[-1], cond=s[0] / s[-1],
                                            **stats(T, est)))
                        if avail in forecast:
                            Bf = modal_amplitudes(T1, f, Vh)
                            for L in vc["forecast_leads"]:
                                eh = projection_forecast(Bf, lam, Phi[LAST][hid], L)
                                records.append(dict(key, method="DMD", r=r, alpha=alpha, lead=L, **stats(T[:, L:], eh)))
                # Hankel-DMD Kalman form
                for r, (lamk, PhiL, Q, Rm) in kfm.items():
                    K, F = kalman_gain(lamk, PhiL, Q, Rm, a)
                    X = kalman_states(K, F, Xt[a])
                    est = (PhiL[hid] @ X[:, SCORE0:]).real
                    records.append(dict(key, method="KF", r=r, **stats(T, est)))
                    if avail in forecast:
                        for L in vc["forecast_leads"]:
                            eh = (PhiL[hid] @ (lamk[:, None] ** L * X[:, SCORE0:N - L])).real
                            records.append(dict(key, method="KF", r=r, lead=L, **stats(T[:, L:], eh)))
            print(f"{direction}: OLC {test} half {h} done ({time.time() - t0:.0f} s)", flush=True)
            with open(out, "wb") as fh:
                pickle.dump(records, fh)
    with open(out, "wb") as fh:
        pickle.dump(records, fh)
    print("done", len(records), "->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directions", nargs="*", default=["FA", "SS"])
    ap.add_argument("--config", default="configs/openfast.yaml")
    a = ap.parse_args()
    main(load_config(a.config), a.directions)
