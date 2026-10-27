"""Operational modal identification of the 12 OLCs (Table 3, Figs. 5 and A1; Section 4.1-4.3).

Hankel-DMD in the response domain over d x r, in the correlation domain over d x r, SSI-COV and FDD on all OLCs
(each OLC a separate segment), and per-OLC tracking.

Outputs (results/openfast/):
  sysid_sensitivity.csv    response-domain Hankel-DMD f, zeta per (d, r, mode)
  corr_dmd_sensitivity.csv correlation-domain Hankel-DMD f, zeta per (d, r, mode)
  corr_dmd_summary.csv     medians and quartiles of corr_dmd_sensitivity (column "Hankel-DMD (corr.)" of Table 3)
  table3_revised.csv       FDD, SSI-COV, response-domain medians/IQR, cross-method MACs, MPC, per-OLC spread
  sysid_per_olc.csv        per-OLC Hankel-DMD (d = 60, r = 24) and SSI-COV estimates
  modeshapes.npz           de-standardised shapes (DMD, SSI, FDD) and their frequencies
Usage: python scripts/openfast/identify.py [--config configs/openfast.yaml]
"""
import argparse

import numpy as np
import pandas as pd

from eftwin.baselines.oma import csd_sum, first_singular, ssi_median_over_orders, ssi_poles
from eftwin.config import as_range, load_config
from eftwin.dmd import correlation_sum, correlation_trajectories, dmd_poles, pick, sorted_eigh
from eftwin.hankel import gram_over_segments
from eftwin.modal import mac, mpc
from eftwin.studies.openfast import P, OpenFASTStudy


def main(cfg):
    S_ = OpenFASTStudy(cfg)
    ic = cfg["identification"]
    DT, FS = S_.dt, S_.fs
    BANDS = {d: {m: tuple(b) for m, b in v.items()} for d, v in ic["bands"].items()}
    first = slice(0, P)                                   # shapes from the first delay block
    d_ref, r_ref = ic["reference"]["delay"], ic["reference"]["rank"]
    ssi_cfg = ic["ssi"]
    i_ssi = ssi_cfg["block_rows"]
    sens_rows, corr_rows, olc_rows, table_rows, shapes_store = [], [], [], [], {}
    for direction in ("FA", "SS"):
        segs, scaler = S_.scaled_segments(direction)
        sc = scaler.scale_
        seglist = [s.T for s in segs.values()]
        # ---------------- response domain: sensitivity over d and r (all OLCs) ----------------
        Gd = {}
        for d in ic["delays"]:
            G, C = gram_over_segments(seglist, d)
            eig = sorted_eigh(G)
            Gd[d] = (G, C, eig)
            for r in as_range(ic["ranks"]):
                f, z, Phi, E, mu = dmd_poles(G, C, r, DT, eig)
                for mode, band in BANDS[direction].items():
                    pk = pick(f, z, Phi, E, mu, band, first)
                    if pk:
                        sens_rows.append(dict(direction=direction, d=d, r=r, mode=mode, f=pk[0], zeta=pk[1]))
            print(f"{direction} d={d} done", flush=True)
        sens = pd.DataFrame(sens_rows)
        # ---------------- correlation domain (stochastic-Koopman form) ----------------
        trajs = correlation_trajectories(correlation_sum(list(segs.values()), ic["correlation_lags"]))
        for d in ic["delays"]:
            G, C = gram_over_segments(trajs, d)
            eig = sorted_eigh(G)
            for r in as_range(ic["ranks"]):
                f, z, Phi, E, mu = dmd_poles(G, C, r, DT, eig)
                for mode, band in BANDS[direction].items():
                    pk = pick(f, z, Phi, E, mu, band, first)
                    if pk:
                        corr_rows.append(dict(direction=direction, d=d, r=r, mode=mode, f=pk[0], zeta=pk[1]))
        # ---------------- SSI-COV (all OLCs) ----------------
        ssi_all = ssi_poles(correlation_sum(list(segs.values()), 2 * i_ssi), P, i_ssi, as_range(ssi_cfg["orders"]), DT)
        # ---------------- FDD (all OLCs): first singular vector of the CSD matrix ----------------
        nper = int(ic["fdd_segment_s"] * FS)
        _, S = csd_sum(list(segs.values()), FS, nper)
        ffdd = np.fft.rfftfreq(nper, DT)
        U1, s1 = first_singular(S)
        # ---------------- consensus per mode ----------------
        G60, C60, eig60 = Gd[d_ref]
        f, z, Phi, E, mu = dmd_poles(G60, C60, r_ref, DT, eig60)
        for mode, band in BANDS[direction].items():
            sub = sens[(sens.direction == direction) & (sens["mode"] == mode)]
            f_d, z_d = sub.f.median(), sub.zeta.median()
            f_iqr = sub.f.quantile([.25, .75]).values; z_iqr = sub.zeta.quantile([.25, .75]).values
            pk = pick(f, z, Phi, E, mu, band, first)
            ss = ssi_median_over_orders(ssi_all, band, ssi_cfg["max_damping"])
            kb = np.where((ffdd > band[0]) & (ffdd < band[1]))[0]
            kpk = kb[np.argmax(s1[kb])]
            v_fdd = U1[kpk] * sc
            row = dict(direction=direction, mode=mode,
                       DMD_f_median=f_d, DMD_f_IQR=f"{f_iqr[0]:.3f}-{f_iqr[1]:.3f}",
                       DMD_zeta_median_pct=100 * z_d, DMD_zeta_IQR_pct=f"{100*z_iqr[0]:.2f}-{100*z_iqr[1]:.2f}",
                       SSI_f=ss[0] if ss else np.nan, SSI_zeta_pct=100 * ss[1] if ss else np.nan,
                       SSI_zeta_IQR_pct=f"{100*ss[3][0]:.2f}-{100*ss[3][1]:.2f}" if ss else "",
                       FDD_f=ffdd[kpk])
            if pk and ss:
                vd, vs = pk[2] * sc, ss[2] * sc
                row.update(MAC_DMD_SSI_acc=mac(vd[:9], vs[:9]), MAC_DMD_SSI_mom=mac(vd[9:], vs[9:]),
                           MAC_DMD_FDD_acc=mac(vd[:9], v_fdd[:9]), MAC_DMD_FDD_mom=mac(vd[9:], v_fdd[9:]),
                           MAC_SSI_FDD_acc=mac(vs[:9], v_fdd[:9]),
                           MPC_DMD_acc=mpc(vd[:9]), MPC_SSI_acc=mpc(vs[:9]))
                shapes_store[f"{direction}_{mode}_DMD"] = vd
                shapes_store[f"{direction}_{mode}_SSI"] = vs
                shapes_store[f"{direction}_{mode}_FDD"] = v_fdd
                shapes_store[f"{direction}_{mode}_f"] = np.array([pk[0], ss[0], ffdd[kpk]])
            table_rows.append(row)
        # ---------------- per-OLC identification ----------------
        for c, s in segs.items():
            G, C = gram_over_segments([s.T], d_ref)
            fo, zo, Phio, Eo, muo = dmd_poles(G, C, r_ref, DT)
            ssc = ssi_poles(correlation_sum([s], 2 * i_ssi), P, i_ssi, as_range(ssi_cfg["per_olc_orders"]), DT)
            for mode, band in BANDS[direction].items():
                pk = pick(fo, zo, Phio, Eo, muo, band, first)
                ss = ssi_median_over_orders(ssc, band, ssi_cfg["max_damping"])
                olc_rows.append(dict(direction=direction, olc=c, mode=mode,
                                     DMD_f=pk[0] if pk else np.nan, DMD_zeta=pk[1] if pk else np.nan,
                                     SSI_f=ss[0] if ss else np.nan, SSI_zeta=ss[1] if ss else np.nan,
                                     MAC_olc_vs_global_DMD=mac(pk[2] * sc, shapes_store.get(f"{direction}_{mode}_DMD", pk[2] * sc)) if pk else np.nan))
        print(f"{direction} per-OLC done", flush=True)

    out = S_.results
    pd.DataFrame(sens_rows).to_csv(out / "sysid_sensitivity.csv", index=False)
    cor = pd.DataFrame(corr_rows)
    cor.to_csv(out / "corr_dmd_sensitivity.csv", index=False)
    summ = cor.groupby(["direction", "mode"]).agg(f_med=("f", "median"), f_q25=("f", lambda x: x.quantile(.25)),
                                                  f_q75=("f", lambda x: x.quantile(.75)),
                                                  z_med=("zeta", lambda x: 100 * x.median()),
                                                  z_q25=("zeta", lambda x: 100 * x.quantile(.25)),
                                                  z_q75=("zeta", lambda x: 100 * x.quantile(.75)), n=("f", "count"))
    summ.to_csv(out / "corr_dmd_summary.csv")
    olc = pd.DataFrame(olc_rows); olc.to_csv(out / "sysid_per_olc.csv", index=False)
    tab = pd.DataFrame(table_rows)
    agg = olc.groupby(["direction", "mode"]).agg(DMD_f_olc_mean=("DMD_f", "mean"), DMD_f_olc_std=("DMD_f", "std"),
                                                  DMD_z_olc_mean=("DMD_zeta", "mean"), DMD_z_olc_std=("DMD_zeta", "std"),
                                                  SSI_f_olc_mean=("SSI_f", "mean"), SSI_f_olc_std=("SSI_f", "std"),
                                                  SSI_z_olc_mean=("SSI_zeta", "mean"), SSI_z_olc_std=("SSI_zeta", "std"),
                                                  MAC_olc_min=("MAC_olc_vs_global_DMD", "min")).reset_index()
    tab = tab.merge(agg, on=["direction", "mode"])
    tab.to_csv(out / "table3_revised.csv", index=False)
    np.savez(out / "modeshapes.npz", **shapes_store)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(tab.round(4).T.to_string())
    print(summ.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/openfast.yaml")
    main(load_config(ap.parse_args().config))
