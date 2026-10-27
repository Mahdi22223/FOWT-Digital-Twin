"""Virtual sensing on the FOCAL Campaign IV tower data: time-blocked leave-one-run-out (Section 5.3, Table 7).

The four runs share the commanded wind (W02) and differ in the wave realisation (E11, E13, E14, E15). Each run
(600-9000 s, 0.25-5 Hz) is split into two halves of 4200 s; fold = (test run, half h), trained on half 1-h of the
three other runs (8 folds per direction). The first 60 s of each test half are not scored. Estimators and
formulas are those of the OpenFAST study, with P = 4 channels and d = 86 samples (3.6 s = 1.5 first-mode
periods). The projection grid includes the extension alpha <= 1 and r = 160, 250; the Kalman ranks go to 250.
Output: results/focal/intermediate/focal_vs_records_<DIR>.pkl
Usage: python scripts/focal/virtual_sensing.py FA   (then SS) [--config configs/focal.yaml]
"""
import argparse
import pickle
import time

import numpy as np

from eftwin.baselines.vs import gappy_pod, pod_basis, static_ls_weights, wiener_blocks, wiener_weights
from eftwin.config import ensure_dir, load_config
from eftwin.dmd import dmd_from_gram
from eftwin.estimators import (delay_windows, kalman_gain, kalman_model, kalman_states, project_windows,
                               projection_estimate, sparse_svd, tikhonov_filter)
from eftwin.hankel import delay_gram, delay_rows, fold_stats
from eftwin.studies.focal import P, FocalStudy
from eftwin.validation import blocked_folds, sufficient_stats as stats


def projection_grid(vc):
    """{rank: [alphas]} of the projection form, in rank order."""
    grid = {}
    for block in vc["projection_grid"]:
        for r in block["ranks"]:
            grid.setdefault(r, [])
            grid[r] += [a for a in block["alphas"] if a not in grid[r]]
    return dict(sorted(grid.items()))


def main(cfg, directions):
    S_ = FocalStudy(cfg)
    vc = cfg["virtual_sensing"]
    D = vc["delay"]
    LAST = slice((D - 1) * P, D * P)
    SCORE0 = int(round(vc["score_start_s"] / S_.dt))
    grid = projection_grid(vc)
    layouts = S_.layouts()
    out_dir = ensure_dir(S_.results / "intermediate")
    records, t0 = [], time.time()
    for direction in directions:
        raw = {run: S_.channels(run, direction) for run in S_.runs}
        HALF = min(len(v) for v in raw.values()) // 2
        grams = {(run, h): delay_gram(raw[run][h * HALF:(h + 1) * HALF], D) for run in S_.runs for h in (0, 1)}
        for test, h, train in blocked_folds(S_.runs):
            st = fold_stats([grams[k] for k in train], [raw[r][hh * HALF:(hh + 1) * HALF] for r, hh in train])
            G, C = st["G"], st["C"]
            Xt = ((raw[test][h * HALF:(h + 1) * HALF] - st["mu"]) / st["sd"]).T
            G0 = G[LAST, LAST]
            Upod = pod_basis(G0)
            dmd = {r: dmd_from_gram(G, C, r, eig=(st["ev"], st["U"]))[1] for r in grid}
            kfm = {r: kalman_model(st, r, P) for r in vc["kalman_ranks"]}
            for name, avail in layouts.items():
                a = np.array(avail); hid = np.setdiff1d(np.arange(P), a); q = len(a)
                key = dict(direction=direction, test=S_.run_label[test], half=h, layout=name,
                           mask=" ".join(map(str, a)), hidden=" ".join(map(str, hid)), q=q)
                T = Xt[hid, SCORE0:]
                Wobs = delay_windows(Xt, a, D, start=SCORE0)
                vrows = delay_rows(a, P, D)
                k = min(q, vc["gappy_pod_max_modes"])
                records.append(dict(key, method="gappyPOD", **stats(T, gappy_pod(Upod, a, hid, Xt[a, SCORE0:], k))))
                records.append(dict(key, method="OLS", **stats(T, (Xt[a, SCORE0:].T @ static_ls_weights(G0, a, hid)).T)))
                GAA, GAh = wiener_blocks(G, vrows, (D - 1) * P + hid)
                for rho in vc["wiener_rhos"]:
                    records.append(dict(key, method="FIR", rho=rho, **stats(T, (Wobs @ wiener_weights(GAA, GAh, rho)).T)))
                for r, Phi in dmd.items():
                    Uu, s, Vh = sparse_svd(Phi, vrows)
                    T1 = project_windows(Wobs, Uu)
                    for alpha in grid[r]:
                        est = projection_estimate(T1, tikhonov_filter(s, alpha), Vh, Phi[LAST][hid])
                        records.append(dict(key, method="DMD", r=r, alpha=alpha, cond=s[0] / s[-1], **stats(T, est)))
                for r, (lamk, PhiL, Q, Rm) in kfm.items():
                    K, F = kalman_gain(lamk, PhiL, Q, Rm, list(a))
                    est = (PhiL[hid] @ kalman_states(K, F, Xt[a])[:, SCORE0:]).real
                    records.append(dict(key, method="KF", r=r, **stats(T, est)))
            print(f"{direction} {S_.run_label[test]} half {h} done ({time.time() - t0:.0f} s)", flush=True)
    out = out_dir / f"focal_vs_records_{'_'.join(directions)}.pkl"
    with open(out, "wb") as fh:
        pickle.dump(records, fh)
    print("done", len(records), "->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("directions", nargs="*", default=["FA", "SS"])
    ap.add_argument("--config", default="configs/focal.yaml")
    a = ap.parse_args()
    main(load_config(a.config), a.directions)
