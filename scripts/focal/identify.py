"""Operational modal identification of the FOCAL Campaign IV tower (full scale): protocol of Table 3 (Table 6).

Tower channels per direction: nacelle acc., tower-base acc., tower-top moment, tower-base moment.
Band 0.25-5 Hz, stationary window 600-9000 s, each run a separate segment.
Delay windows scaled to the same number of first-mode periods as in the OpenFAST study (0.75-2.2 periods):
  corr-DMD : d in {43, 86, 129} samples x r in 10..50, pole of largest modal energy per band
  SSI-COV  : i = 86 block rows, orders 20..80, stabilisation (df < 1 %, dzeta < 5 %, MAC > 0.95 between orders)
  FDD      : first singular value of the CSD matrix (200-s segments)
Platform roll/pitch: same tower channels, 0.01-0.2 Hz, decimated by 8, delays 15/30/45 s.
Outputs (results/focal/): focal_sysid_grid.csv, focal_sysid_summary.csv, focal_shapes.npz
Usage: python scripts/focal/identify.py [--config configs/focal.yaml]
"""
import argparse

import numpy as np
import pandas as pd

from eftwin.baselines.oma import fdd_peak, ssi_poles, ssi_stabilised
from eftwin.config import as_range, load_config
from eftwin.dmd import correlation, correlation_trajectories, dmd_poles, pick
from eftwin.hankel import gram_over_segments
from eftwin.modal import mac, mpc
from eftwin.preprocess import bandpass_sos, lowpass_decimate
from eftwin.studies.focal import P, FocalStudy

LAST = slice(-P, None)          # FOCAL convention: shapes from the most recent delay block


def identify(Rlist, dt, bands, d_grid, r_grid, i_ssi, s_list, fs_, label, fdd_seg, ref_rank,
             ssi_orders=range(20, 82, 2)):
    """Correlation-domain Hankel-DMD over (d, r), and at (d_ref, r_ref) the reference DMD pole, the stabilised
    SSI-COV estimate and the FDD peak. Rlist: correlation functions (one per run; their mean is used)."""
    Rm = np.mean(Rlist, axis=0)
    rows, best = [], {}
    trajs = correlation_trajectories(Rm)
    Gd = {}
    for d in d_grid:
        G, C = gram_over_segments(trajs, d)
        Gd[d] = (G, C)
        for r in r_grid:
            f, z, Phi, E, mu = dmd_poles(G, C, r, dt)
            for mode, band in bands.items():
                pk = pick(f, z, Phi, E, mu, band, LAST, positive_damping=True)
                if pk:
                    rows.append(dict(set=label, d=d, r=r, mode=mode, f=pk[0], zeta=pk[1]))
    dref, rref = d_grid[1], min(ref_rank, max(r_grid))
    f, z, Phi, E, mu = dmd_poles(*Gd[dref], rref, dt)
    poles = ssi_poles(Rm, P, i_ssi, ssi_orders, dt)
    for mode, band in bands.items():
        pk = pick(f, z, Phi, E, mu, band, LAST, positive_damping=True)
        ss = ssi_stabilised(poles, band)
        best[mode] = dict(pk=pk, ss=ss, fdd=fdd_peak(s_list, band, fs_, fdd_seg))
    return rows, best


def summary_row(Gdf, direction, mode, best, per_run, runs, run_label, sd, with_runs_dir=True):
    sel = (Gdf["mode"] == mode) & (Gdf.set == "pooled")
    if with_runs_dir:
        sel &= (Gdf.direction == direction)
    sub = Gdf[sel]
    pk, ss, (ff, uf) = best[mode]["pk"], best[mode]["ss"], best[mode]["fdd"]

    def runsel(r):
        m = (Gdf["mode"] == mode) & (Gdf.set == run_label[r])
        return Gdf[(m & (Gdf.direction == direction)) if with_runs_dir else m]
    run_f = [runsel(r).f.median() for r in runs]
    run_z = [runsel(r).zeta.median() for r in runs]
    run_fs = [per_run[r][mode]["ss"][0] if per_run[r][mode]["ss"] else np.nan for r in runs]
    run_zs = [per_run[r][mode]["ss"][1] if per_run[r][mode]["ss"] else np.nan for r in runs]
    row = dict(direction=direction, mode=mode, FDD_f=ff,
               SSI_f=ss[0] if ss else np.nan, SSI_zeta=100 * ss[1] if ss else np.nan, SSI_nstab=ss[3] if ss else 0,
               DMD_f=sub.f.median(), DMD_f_q25=sub.f.quantile(.25), DMD_f_q75=sub.f.quantile(.75),
               DMD_zeta=100 * sub.zeta.median(), DMD_z_q25=100 * sub.zeta.quantile(.25),
               DMD_z_q75=100 * sub.zeta.quantile(.75),
               DMD_f_runs_min=min(run_f), DMD_f_runs_max=max(run_f),
               DMD_z_runs_min=100 * min(run_z), DMD_z_runs_max=100 * max(run_z),
               SSI_f_runs_min=np.nanmin(run_fs), SSI_f_runs_max=np.nanmax(run_fs),
               SSI_z_runs_min=100 * np.nanmin(run_zs), SSI_z_runs_max=100 * np.nanmax(run_zs))
    return row, pk, ss, uf


def main(cfg):
    S_ = FocalStudy(cfg)
    ic = cfg["identification"]
    DT, FS = S_.dt, S_.fs
    RUNS, RUN_LABEL = S_.runs, S_.run_label
    D_GRID, R_GRID, I_SSI = tuple(ic["delays"]), as_range(ic["ranks"]), ic["ssi_block_rows"]
    ORDERS = as_range(ic["ssi_orders"])
    grid, summary, shapes = [], [], {}
    K = int(ic["correlation_s"] / DT)
    for direction in ("FA", "SS"):
        bands = {m: tuple(b) for m, b in ic["bands"][direction].items()}
        segs = {run: S_.channels(run, direction) for run in RUNS}
        sd = np.std(np.vstack(list(segs.values())), axis=0)         # common scaling (as StandardScaler)
        segs = {k: v / sd for k, v in segs.items()}
        Rs = {run: correlation(s, max(K, 2 * I_SSI + 1)) for run, s in segs.items()}
        per_run = {}
        for run in RUNS:
            rows, best = identify([Rs[run]], DT, bands, D_GRID, R_GRID, I_SSI, [segs[run]], FS, RUN_LABEL[run],
                                  ic["fdd_segment_s"], ic["reference_rank"], ORDERS)
            for r_ in rows:
                r_["direction"] = direction
            grid += rows; per_run[run] = best
        rows, best = identify(list(Rs.values()), DT, bands, D_GRID, R_GRID, I_SSI, list(segs.values()), FS, "pooled",
                              ic["fdd_segment_s"], ic["reference_rank"], ORDERS)
        for r_ in rows:
            r_["direction"] = direction
        grid += rows
        Gdf = pd.DataFrame(grid)
        for mode in bands:
            row, pk, ss, uf = summary_row(Gdf, direction, mode, best, per_run, RUNS, RUN_LABEL, sd)
            if pk and ss:
                vd, vs, vf = pk[2] * sd, ss[2] * sd, uf * sd
                row.update(MAC_DMD_SSI=mac(vd, vs), MAC_DMD_FDD=mac(vd, vf), MPC_DMD=mpc(vd), MPC_SSI=mpc(vs),
                           MAC_runs_min=min(mac(vd, per_run[r][mode]["pk"][2] * sd) for r in RUNS if per_run[r][mode]["pk"]))
                shapes[f"{direction}|{mode}|DMD"] = vd; shapes[f"{direction}|{mode}|SSI"] = vs
                shapes[f"{direction}|{mode}|FDD"] = vf
            summary.append(row)
        print(direction, "done", flush=True)

    # ---- platform roll (SS channels) and pitch (FA channels), low band ----
    pc = ic["platform"]
    q = pc["decimate"]; dtq = DT * q; fsq = 1 / dtq
    for dirn, bdict in (("SS", pc["bands"]["SS"]), ("FA", pc["bands"]["FA"])):
        pname, pband = next(iter(bdict.items()))
        lows = []
        for run in RUNS:
            seg = S_.data.segment(run, dirn)
            seg = lowpass_decimate(seg, FS, pc["lowpass_hz"], q)
            seg = bandpass_sos(seg, pc["band"][0], pc["band"][1], fsq)
            lows.append(seg)
        sdl = np.std(np.vstack(lows), axis=0); lows = [s_ / sdl for s_ in lows]
        Kl = int(pc["correlation_s"] / dtq)
        Rl = {RUNS[k]: correlation(s_, Kl) for k, s_ in enumerate(lows)}
        band = {pname: tuple(pband)}
        dl, il = tuple(int(s / dtq) for s in pc["delays_s"]), int(pc["ssi_block_rows_s"] / dtq)
        r_low = as_range(pc["ranks"])
        low_runs = {}
        for k, run in enumerate(RUNS):
            rows, best = identify([Rl[run]], dtq, band, dl, r_low, il, [lows[k]], fsq, RUN_LABEL[run],
                                  pc["fdd_segment_s"], ic["reference_rank"], ORDERS)
            for r_ in rows:
                r_["direction"] = dirn
            grid += rows; low_runs[run] = best
        rows, best = identify(list(Rl.values()), dtq, band, dl, r_low, il, lows, fsq, "pooled",
                              pc["fdd_segment_s"], ic["reference_rank"], ORDERS)
        for r_ in rows:
            r_["direction"] = dirn
        grid += rows
        Gdf = pd.DataFrame(grid)
        row, pk, ss, uf = summary_row(Gdf, dirn, pname, best, low_runs, RUNS, RUN_LABEL, sdl, with_runs_dir=False)
        if pk and ss:
            vd, vs, vf = pk[2] * sdl, ss[2] * sdl, uf * sdl
            row.update(MAC_DMD_SSI=mac(vd, vs), MAC_DMD_FDD=mac(vd, vf), MPC_DMD=mpc(vd), MPC_SSI=mpc(vs))
        summary.append(row)

    out = S_.results
    pd.DataFrame(grid).to_csv(out / "focal_sysid_grid.csv", index=False)
    Sdf = pd.DataFrame(summary); Sdf.to_csv(out / "focal_sysid_summary.csv", index=False)
    np.savez(out / "focal_shapes.npz", **shapes)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(Sdf.round(4).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/focal.yaml")
    main(load_config(ap.parse_args().config))
