"""Reconstructions on the unseen half used for Figs. 15-16 (run E11, second half; model from the first half of the
other runs): all five estimators at their cross-fitted hyper-parameters, layouts N and N+P (nacelle; nacelle +
platform accelerometer). Delay windows are completed with zeros before the first d-1 samples.
Input: results/focal/focal_vs_hyper.csv. Output: results/focal/intermediate/vs_series_<DIR>.npz (about 35 MB each)
Usage: python scripts/focal/series.py [--config configs/focal.yaml]
"""
import argparse

import numpy as np
import pandas as pd

from eftwin.baselines.vs import gappy_pod, pod_basis, static_ls_weights, wiener_blocks, wiener_weights
from eftwin.config import ensure_dir, load_config
from eftwin.dmd import dmd_from_gram
from eftwin.estimators import (delay_windows, kalman_gain, kalman_model, kalman_states, project_windows,
                               projection_estimate, sparse_svd, tikhonov_filter)
from eftwin.hankel import delay_gram, delay_rows, fold_stats
from eftwin.studies.focal import P, FocalStudy


def main(cfg):
    S_ = FocalStudy(cfg)
    vc = cfg["virtual_sensing"]
    D = vc["delay"]
    LAST = slice((D - 1) * P, D * P)
    hyp = pd.read_csv(S_.results / "focal_vs_hyper.csv")
    test, h = vc["series"]["run"], vc["series"]["half"]
    grp = int(S_.run_label[test] in vc["crossfit_group_b"])      # E11: group 0 -> settings selected on group 1
    layouts = S_.layouts()
    out_dir = ensure_dir(S_.results / "intermediate")
    for d in ("FA", "SS"):
        raw = {r: S_.channels(r, d) for r in S_.runs}
        HALF = min(len(v) for v in raw.values()) // 2
        train = [(r, 1 - h) for r in S_.runs if r != test]
        st = fold_stats([delay_gram(raw[r][hh * HALF:(hh + 1) * HALF], D) for r, hh in train],
                        [raw[r][hh * HALF:(hh + 1) * HALF] for r, hh in train])
        Xt = ((raw[test][h * HALF:(h + 1) * HALF] - st["mu"]) / st["sd"]).T
        N = Xt.shape[1]
        out = dict(mu=st["mu"], sd=st["sd"], truth=Xt, omega=S_.aux(test, "genSpeed")[h * HALF:(h + 1) * HALF])
        G, C = st["G"], st["C"]
        for lay in vc["series"]["layouts"]:
            a = np.array(layouts[lay]); hid = np.setdiff1d(np.arange(P), a); q = len(a)
            g = hyp[(hyp.direction == d) & (hyp.layout == lay) & (hyp.group == grp)].set_index("method").hp
            G0 = G[LAST, LAST]
            k = min(q, vc["gappy_pod_max_modes"])
            est = {"gappyPOD": gappy_pod(pod_basis(G0), a, hid, Xt[a], k)}
            est["OLS"] = (Xt[a].T @ static_ls_weights(G0, a, hid)).T
            Wobs = delay_windows(Xt, a, D, pad=True)
            vrows = delay_rows(a, P, D)
            rho = float(g["FIR"].split("|")[2])
            est["FIR"] = (Wobs @ wiener_weights(*wiener_blocks(G, vrows, (D - 1) * P + hid), rho)).T
            r, alpha = int(float(g["DMD"].split("|")[0])), float(g["DMD"].split("|")[1])
            Phi = dmd_from_gram(G, C, r, eig=(st["ev"], st["U"]))[1]
            Uu, s, Vh = sparse_svd(Phi, vrows)
            est["DMD"] = projection_estimate(project_windows(Wobs, Uu), tikhonov_filter(s, alpha), Vh, Phi[LAST][hid])
            rk = int(float(g["KF"].split("|")[0]))
            lamk, PhiL, Q, Rm = kalman_model(st, rk, P)
            K, F = kalman_gain(lamk, PhiL, Q, Rm, list(a))
            est["KF"] = (PhiL[hid] @ kalman_states(K, F, Xt[a])).real
            for m_, e in est.items():
                full = np.full((P, N), np.nan); full[hid] = e
                out[f"{lay}|{m_}"] = full
            print(d, lay, "r_DMD", r, "alpha", alpha, "r_KF", rk, "rho", rho, flush=True)
        np.savez(out_dir / f"vs_series_{d}.npz", **out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/focal.yaml")
    main(load_config(ap.parse_args().config))
