"""Post-processing of the time-blocked virtual-sensing study (virtual_sensing.py).

Hyper-parameters are cross-fitted by OLC group: folds whose test OLC is in 1-6 use the setting selected on the
folds with test OLC in 7-12 (both time halves) and vice versa; selection criterion = mean R^2 over those folds
and over all layouts with the same number of active sensors q (forecasts: per layout, mean over leads).
Accuracy per held-out OLC is computed from both time halves.

Outputs (results/openfast/): blocked_hyper.csv, blocked_selected_olc.csv, blocked_summary_random.csv,
blocked_cases_per_sensor.csv, blocked_layouts.csv, blocked_forecast.csv, blocked_L1L2_per_channel.csv
Usage: python scripts/openfast/summarise.py [--config configs/openfast.yaml]
"""
import argparse
import glob
import pickle

import numpy as np
import pandas as pd

from eftwin.config import load_config
from eftwin.studies.openfast import P, OpenFASTStudy
from eftwin.validation import combine, crossfit, records_frame


def load_records(S_):
    recs = []
    for f in sorted(glob.glob(str(S_.results / "intermediate" / "blocked_records_*.pkl"))):
        with open(f, "rb") as fh:
            recs += pickle.load(fh)
    return recs


def per_olc(g):
    """Combine the two halves of each (test OLC, mask, method): per-channel R^2 and NRMSE over the full record."""
    out = []
    for (d, test, mask, m), gg in g.groupby(["direction", "test", "mask", "method"]):
        r2, nr = combine(gg)
        out.append(dict(direction=d, test=test, mask=mask, q=gg.q.iloc[0], method=m, R2=r2.mean(), R2min=r2.min(),
                        R2_ch=r2, NRMSE_ch=nr, cond=gg["cond"].median() if "cond" in gg else np.nan,
                        hp=";".join(sorted(set(gg.hp)))))
    return pd.DataFrame(out)


def main(cfg):
    S_ = OpenFASTStudy(cfg)
    R = S_.results
    vc = cfg["virtual_sensing"]
    df = records_frame(load_records(S_), S_.group_of)
    if "lead" not in df:
        df["lead"] = np.nan
    df["lead"] = df["lead"].fillna(0).astype(int)

    # ---------------- cross-fitted selection (nowcast) ----------------
    now = df[df.lead == 0]
    sel, hyper = crossfit(now, ["direction", "q", "method"])
    hyper.to_csv(R / "blocked_hyper.csv", index=False)
    olc = per_olc(sel)
    olc.drop(columns=["R2_ch", "NRMSE_ch"]).to_csv(R / "blocked_selected_olc.csv", index=False)

    key = S_.mask_key
    CASE_A, CASE_BC, L2 = key(S_.layout("case_a")), key(S_.layout("case_bc")), key(S_.layout("l2"))
    EXTRA = {CASE_A, CASE_BC, L2}
    rnd = olc[~olc["mask"].isin(EXTRA)]
    summ = rnd.groupby(["direction", "q", "method"]).R2.agg(
        median="median", p25=lambda x: x.quantile(.25), p75=lambda x: x.quantile(.75), n="count").reset_index()
    summ.to_csv(R / "blocked_summary_random.csv", index=False)

    # ---------------- paper scenarios, per sensor ----------------
    rows = []
    for sc in vc["scenarios"]:
        mask, test = key(S_.layout(sc["layout"])), sc["test"]
        g = olc[(olc["mask"] == mask) & (olc.test == test)]
        hid = np.setdiff1d(np.arange(P), np.array(mask.split(), int))
        for _, rr in g.iterrows():
            for i, s in enumerate(hid):
                rows.append(dict(direction=rr.direction, scenario=sc["name"], test=test, q=rr.q, sensor=int(s),
                                 method=rr.method, R2=rr.R2_ch[i], NRMSE=rr.NRMSE_ch[i], cond=rr.cond, hp=rr.hp))
    cases = pd.DataFrame(rows); cases.to_csv(R / "blocked_cases_per_sensor.csv", index=False)

    # ---------------- practical layouts ----------------
    rows = []
    for pl in vc["practical_layouts"]:
        mask = key(S_.layout(pl["layout"]))
        g = olc[olc["mask"] == mask]
        hid = list(np.setdiff1d(np.arange(P), np.array(mask.split(), int)))
        for _, rr in g.iterrows():
            rows.append(dict(direction=rr.direction, scenario=pl["name"], test=rr.test, q=rr.q, method=rr.method,
                             R2_mean=rr.R2, R2_min=rr.R2min, R2_base_moment=rr.R2_ch[hid.index(9)] if 9 in hid else np.nan,
                             R2_mid_moment=rr.R2_ch[hid.index(13)], NRMSE_mean=np.mean(rr.NRMSE_ch), cond=rr.cond))
    lay = pd.DataFrame(rows); lay.to_csv(R / "blocked_layouts.csv", index=False)

    # ---------------- forecasts: cross-fitted on mean skill over leads ----------------
    fsel, _ = crossfit(df[df.lead > 0], ["direction", "mask", "method"])
    frows = []
    for (d, mask, m, L, test), g in fsel.groupby(["direction", "mask", "method", "lead", "test"]):
        n = g.n.sum(); sy = np.sum(np.vstack(g.sy), 0); syy = np.sum(np.vstack(g.syy), 0); sse = np.sum(np.vstack(g.sse), 0)
        frows.append(dict(direction=d, mask=mask, q=g.q.iloc[0], method=m, lead_s=L / S_.fs, test=test,
                          R2=np.mean(1 - sse / (syy - sy ** 2 / n))))
    fo = pd.DataFrame(frows); fo.to_csv(R / "blocked_forecast.csv", index=False)

    # ---------------- per-channel accuracy of layouts L1 and L2 (Table A2) ----------------
    hyp = pd.read_csv(R / "blocked_hyper.csv")
    l12 = df[(df["mask"].isin([key(S_.layout("l1")), L2])) & (df.lead == 0)]
    rows = []
    for (d, mask, m), g in l12.groupby(["direction", "mask", "method"]):
        q = len(mask.split())
        for test, gg in g.groupby("test"):
            grp = int(test in vc["crossfit_group_b"])
            hp = hyp[(hyp.direction == d) & (hyp.q == q) & (hyp.method == m) & (hyp.group == grp)].hp.iloc[0]
            gs = gg[gg.hp == hp]
            n = gs.n.sum(); sy = np.sum(np.vstack(gs.sy), 0); syy = np.sum(np.vstack(gs.syy), 0); sse = np.sum(np.vstack(gs.sse), 0)
            r2 = 1 - sse / (syy - sy ** 2 / n)
            hid = np.setdiff1d(np.arange(P), np.array(mask.split(), int))
            for i, s in enumerate(hid):
                rows.append(dict(direction=d, mask=mask, method=m, test=test, sensor=int(s), R2=r2[i]))
    pd.DataFrame(rows).to_csv(R / "blocked_L1L2_per_channel.csv", index=False)

    pd.set_option("display.width", 220)
    print("== random layouts: median R2 per held-out OLC ==")
    print(summ.pivot_table(index=["direction", "q"], columns="method", values="median").round(3).to_string())
    print("== selected hyper-parameters ==")
    print(hyper.pivot_table(index=["direction", "q", "group"], columns="method", values="hp", aggfunc="first").to_string())
    print("== Cases A-C: mean [min] over hidden sensors ==")
    print(cases.groupby(["direction", "scenario", "method"]).R2.agg(["mean", "min"]).round(4).unstack("method").to_string())
    print("== layouts (median over OLCs) ==")
    print(lay.groupby(["direction", "scenario", "method"])[["R2_mean", "R2_min", "R2_base_moment", "R2_mid_moment"]]
          .median().round(3).to_string())
    print("== forecast skill (median over OLCs) ==")
    print(fo.groupby(["direction", "q", "method", "lead_s"]).R2.median().unstack("lead_s").round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/openfast.yaml")
    main(load_config(ap.parse_args().config))
