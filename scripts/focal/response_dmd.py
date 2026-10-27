"""Response-domain Hankel-DMD on the FOCAL tower data (each run a separate segment), same grid as identify.py
(column "Hankel-DMD (resp.)" of Table 6).
Outputs (results/focal/): focal_resp_dmd_grid.csv, focal_resp_dmd_summary.csv
Usage: python scripts/focal/response_dmd.py [--config configs/focal.yaml]
"""
import argparse

import numpy as np
import pandas as pd

from eftwin.config import as_range, load_config
from eftwin.dmd import dmd_poles, pick
from eftwin.hankel import gram_over_segments
from eftwin.studies.focal import P, FocalStudy


def main(cfg):
    S_ = FocalStudy(cfg)
    ic = cfg["identification"]
    rows = []
    for direction in ("FA", "SS"):
        bands = {m: tuple(b) for m, b in ic["bands"][direction].items()}
        segs = [S_.channels(r, direction) for r in S_.runs]
        sd = np.std(np.vstack(segs), axis=0)
        segs = [(s / sd).T for s in segs]
        for d in ic["delays"]:
            G, C = gram_over_segments(segs, d)
            for r in as_range(ic["ranks"]):
                f, z, Phi, E, mu = dmd_poles(G, C, r, S_.dt)
                for mode, band in bands.items():
                    pk = pick(f, z, Phi, E, mu, band, slice(-P, None), positive_damping=True)
                    if pk:
                        rows.append(dict(direction=direction, d=d, r=r, mode=mode, f=pk[0], zeta=pk[1]))
            print(direction, d, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(S_.results / "focal_resp_dmd_grid.csv", index=False)
    s = df.groupby(["direction", "mode"]).agg(f=("f", "median"), f_q25=("f", lambda x: x.quantile(.25)),
                                              f_q75=("f", lambda x: x.quantile(.75)),
                                              z=("zeta", lambda x: 100 * x.median()),
                                              z_q25=("zeta", lambda x: 100 * x.quantile(.25)),
                                              z_q75=("zeta", lambda x: 100 * x.quantile(.75)), n=("f", "count"))
    s.to_csv(S_.results / "focal_resp_dmd_summary.csv")
    print(s.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/focal.yaml")
    main(load_config(ap.parse_args().config))
