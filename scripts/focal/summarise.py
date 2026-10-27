"""Post-processing of the FOCAL virtual-sensing study (same rules as scripts/openfast/summarise.py, per layout).

Cross-fitting: folds whose test run is E11/E13 use the hyper-parameters selected (mean R^2 over folds and hidden
channels) on the folds of E14/E15, and vice versa. Accuracy per held-out run is computed from both time halves.
Outputs (results/focal/): focal_vs_hyper.csv, focal_vs_per_run.csv, focal_vs_summary.csv
Usage: python scripts/focal/summarise.py [--config configs/focal.yaml]
"""
import argparse
import glob
import pickle

import numpy as np
import pandas as pd

from eftwin.config import load_config
from eftwin.studies.focal import FocalStudy
from eftwin.validation import combine, crossfit, records_frame


def main(cfg):
    S_ = FocalStudy(cfg)
    R = S_.results
    CHN = S_.channel_label
    recs = []
    for f in sorted(glob.glob(str(R / "intermediate" / "focal_vs_records_*.pkl"))):
        with open(f, "rb") as fh:
            recs += pickle.load(fh)
    df = records_frame(recs, S_.group_of)
    sel, hyper = crossfit(df, ["direction", "layout", "method"])
    hyper.to_csv(R / "focal_vs_hyper.csv", index=False)

    rows = []
    for (d, test, lay, m), g in sel.groupby(["direction", "test", "layout", "method"]):
        r2, nr = combine(g)
        hid = [int(x) for x in g.hidden.iloc[0].split()]
        for i, ch in enumerate(hid):
            rows.append(dict(direction=d, test=test, layout=lay, mask=g["mask"].iloc[0], q=g.q.iloc[0], method=m,
                             channel=ch, channel_name=CHN[ch], R2=r2[i], NRMSE=nr[i],
                             cond=g["cond"].median() if "cond" in g and g["cond"].notna().any() else np.nan,
                             hp=";".join(sorted(set(g.hp)))))
    per = pd.DataFrame(rows); per.to_csv(R / "focal_vs_per_run.csv", index=False)
    summ = per.groupby(["direction", "layout", "q", "channel", "channel_name", "method"]).R2.agg(
        median="median", rmin="min", rmax="max").reset_index()
    summ.to_csv(R / "focal_vs_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.pivot_table(index=["layout", "channel_name"], columns=["direction", "method"], values="median").round(3).to_string())
    print(hyper.pivot_table(index=["direction", "layout", "group"], columns="method", values="hp", aggfunc="first").to_string())
    print(per[per.method == "DMD"].groupby(["direction", "layout"]).cond.median().apply(lambda x: f"{x:.2e}").to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/focal.yaml")
    main(load_config(ap.parse_args().config))
