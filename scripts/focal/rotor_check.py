"""Rotor-speed conditioning (Section 5.2): correlation-domain Hankel-DMD on 200-s blocks of the lowest and highest
rotor-speed tercile. A rotor harmonic must follow the rotor speed (f_3P = 3 Omega / 2 pi); a structural pole must not.
Output: results/focal/focal_rotor_check.csv
Usage: python scripts/focal/rotor_check.py [--config configs/focal.yaml]
"""
import argparse

import numpy as np
import pandas as pd

from eftwin.config import as_range, load_config
from eftwin.dmd import correlation, correlation_trajectories, dmd_poles, pick
from eftwin.hankel import gram_over_segments
from eftwin.studies.focal import P, FocalStudy


def main(cfg):
    S_ = FocalStudy(cfg)
    ic = cfg["identification"]
    DT = S_.dt
    nb, K = int(ic["rotor_check"]["block_s"] / DT), int(ic["correlation_s"] / DT)
    band3p = tuple(ic["rotor_check"]["harmonic_band_3P"])
    rows = []
    for direction in ("FA", "SS"):
        bands = {m: tuple(b) for m, b in ic["bands"][direction].items()}
        blocks, om = [], []
        segs = {run: S_.channels(run, direction) for run in S_.runs}
        sd = np.std(np.vstack(list(segs.values())), axis=0)
        for run in S_.runs:
            s = segs[run] / sd; w = S_.aux(run, "genSpeed")
            for b in range(0, len(s) - nb + 1, nb):
                blocks.append(correlation(s[b:b + nb], K)); om.append(w[b:b + nb].mean())
        om = np.array(om); lo, hi = om <= np.percentile(om, 100 / 3), om >= np.percentile(om, 200 / 3)
        for name, m in (("low", lo), ("high", hi)):
            Rm = np.mean(np.array(blocks)[m], axis=0)
            trajs = correlation_trajectories(Rm)
            for d in ic["delays"]:
                G, C = gram_over_segments(trajs, d)
                for r in as_range(ic["ranks"]):
                    f, z, Phi, E, mu = dmd_poles(G, C, r, DT)
                    for mode, band in bands.items():
                        band = band3p if mode == "3P" else band
                        pk = pick(f, z, Phi, E, mu, band, slice(-P, None), positive_damping=True)
                        if pk:
                            rows.append(dict(direction=direction, group=name, omega=om[m].mean(), mode=mode, f=pk[0], zeta=pk[1]))
    df = pd.DataFrame(rows)
    s = df.groupby(["direction", "mode", "group"]).agg(omega=("omega", "first"), f=("f", "median"),
                                                       z=("zeta", lambda x: 100 * x.median())).reset_index()
    s["f_3P_expected"] = 3 * s.omega / (2 * np.pi)
    print(s.round(4).to_string())
    for (dirn, mode), g in s.groupby(["direction", "mode"]):
        lo_, hi_ = g[g.group == "low"].iloc[0], g[g.group == "high"].iloc[0]
        print(f"{dirn} {mode}: rotor speed +{100*(hi_.omega/lo_.omega-1):.1f}%  ->  f change {100*(hi_.f/lo_.f-1):+.2f}%")
    s.to_csv(S_.results / "focal_rotor_check.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/focal.yaml")
    main(load_config(ap.parse_args().config))
