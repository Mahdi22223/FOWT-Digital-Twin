"""Coherence bound of the FOCAL virtual sensing (Table 7, Section 5.3).

Upper bound of the R^2 achievable by any linear time-invariant estimator (two-sided Wiener filter) of a hidden
channel from a set of active channels: R2_max = int gamma^2_{h|a}(f) S_hh(f) df / int S_hh(f) df over 0.25-5 Hz,
with the multiple coherence gamma^2_{h|a} = S_ha S_aa^-1 S_ah / S_hh estimated from the measured data (CSD averaged
over the four runs, 100-s Welch segments). Also the share of each channel's in-band variance near the rotor
harmonics, near the first tower mode and above 2 Hz.
Output: results/focal/focal_coherence_bound.csv
Usage: python scripts/focal/coherence_bound.py [--config configs/focal.yaml]
"""
import argparse

import numpy as np
import pandas as pd

from eftwin.baselines.oma import csd_sum
from eftwin.config import load_config
from eftwin.studies.focal import P, FocalStudy
from eftwin.validation import coherence_bound


def main(cfg):
    S_ = FocalStudy(cfg)
    cc = cfg["coherence_bound"]
    CHN = S_.channel_label
    rows = []
    for d in ("FA", "SS"):
        f, S = csd_sum([(lambda x: x / x.std(0))(S_.channels(r, d)) for r in S_.runs], S_.fs, int(cc["segment_s"] * S_.fs))
        S /= len(S_.runs)
        m = (f >= cc["band"][0]) & (f <= cc["band"][1])
        Sm = S[m]
        for name, a in S_.layouts().items():
            a = list(a); hid = [i for i in range(P) if i not in a]
            for h in hid:
                rows.append(dict(direction=d, layout=name, channel=h, channel_name=CHN[h], R2_bound=coherence_bound(Sm, a, h)))
        # spectral composition: share of in-band variance near 3P..12P, near the tower mode and above 2 Hz
        fm = f[m]
        hw = cc["harmonic_halfwidth_hz"]
        for h in range(P):
            Shh = Sm[:, h, h].real
            tot = Shh.sum()
            f1 = cc["tower_hz"][d]
            harm = np.zeros_like(fm, bool)
            for n in (3, 6, 9, 12):
                harm |= np.abs(fm - n * cc["rotor_1p_hz"]) < hw
            tower = np.abs(fm - f1) < hw
            hi = fm > cc["high_band_hz"]
            rows.append(dict(direction=d, layout="composition", channel=h, channel_name=CHN[h],
                             share_harmonics=Shh[harm].sum() / tot, share_tower=Shh[tower & ~harm].sum() / tot,
                             share_above_2Hz=Shh[hi].sum() / tot))
    df = pd.DataFrame(rows)
    df.to_csv(S_.results / "focal_coherence_bound.csv", index=False)
    pd.set_option("display.width", 200)
    print(df[df.layout != "composition"].pivot_table(index=["layout", "channel_name"], columns="direction", values="R2_bound").round(3).to_string())
    print(df[df.layout == "composition"][["direction", "channel_name", "share_harmonics", "share_tower", "share_above_2Hz"]].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/focal.yaml")
    main(load_config(ap.parse_args().config))
