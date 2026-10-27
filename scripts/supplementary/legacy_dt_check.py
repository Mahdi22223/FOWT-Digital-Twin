"""Sampling interval of the OpenFAST case files, and the frequencies that the original-submission pipeline
(legacy/hankel_dmd_original/Filter_Hankel_03.py) reports when it is applied to them.

1) time-column spacing and record length of every case;
2) spectral peaks of the tower-top accelerations at the true sampling rate;
3) the legacy identification exactly as in Filter_Hankel_03.py (dt assumed 0.0125 s, band-pass designed for
   80 Hz, all 12 cases concatenated in lexicographic file order, z-score, Hankel d = 60, rank 24, exact DMD),
   printed next to the corresponding true frequencies (legacy / 4).
Usage: python scripts/supplementary/legacy_dt_check.py [--config configs/openfast.yaml]
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, welch
from sklearn.preprocessing import StandardScaler

from eftwin.config import load_config
from eftwin.dmd import dmd_from_gram, modal_params
from eftwin.hankel import gram_over_segments
from eftwin.io import direction_slice
from eftwin.preprocess import bandpass
from eftwin.studies.openfast import OpenFASTStudy


def main(cfg):
    S_ = OpenFASTStudy(cfg)
    cases = S_.cases
    print("case  N      dt_median[s]  duration[s]")
    data = {}
    for c in cases:
        t, x = S_.data.load(c)
        data[c] = x
        print(f"{c:>4}  {len(t):>6} {np.median(np.diff(t)):.5f}      {t[-1]-t[0]:.1f}")
    fs_true = 1.0 / np.median(np.diff(S_.data.load(1)[0]))
    print(f"\nTrue sampling rate: {fs_true:.2f} Hz")

    names = {8: "FA top accel (TwHt9ALxt)", 26: "SS top accel (TwHt9ALyt)"}
    for c in (1, 7, 12):
        x = data[c][int(300 * fs_true):]  # drop first 300 s
        print(f"\nCase {c}  Hs={cases[c][0]} Tp={cases[c][1]} dir={cases[c][2]}  (wave peak 1/Tp={1/cases[c][1]:.3f} Hz)")
        for col, nm in names.items():
            f, Pxx = welch(x[:, col], fs=fs_true, nperseg=int(200 * fs_true))
            band = (f > 0.03) & (f < 4.0)
            pk, _ = find_peaks(np.log10(Pxx[band]), prominence=0.5)
            fb, Pb = f[band][pk], Pxx[band][pk]
            top = np.argsort(Pb)[::-1][:8]
            print(f"  {nm:26s} peaks [Hz]: " + ", ".join(f"{v:.3f}" for v in np.sort(fb[top])))

    files = sorted(p.name for p in Path(S_.data.csv_dir).glob("Case_*.csv"))
    order = [int(f.split("_")[1].split(".")[0]) for f in files]
    print(f"\nLegacy (lexicographic) case order: {order}")
    for direction in ("FA", "SS"):
        sl = direction_slice(direction)
        segs = [bandpass(data[c][:, sl], 0.25, 5.0, fs_design=80.0) for c in order]  # legacy: filter designed for fs=80
        X = np.vstack(segs)                      # concatenated as in the legacy code
        Xs = StandardScaler().fit_transform(X)   # z-score over all data
        G, C = gram_over_segments([Xs.T], 60)   # one concatenated record (legacy)
        eigs, _ = dmd_from_gram(G, C, 24)
        f_legacy, z, _ = modal_params(eigs, dt=0.0125)
        keep = (eigs.imag > 0)
        idx = np.argsort(f_legacy[keep])
        print(f"\n{direction}: legacy-reported freq [Hz] | true freq [Hz] (=legacy/4) | damping [%]")
        for fl, zz in zip(f_legacy[keep][idx], z[keep][idx]):
            print(f"   {fl:6.3f}   |   {fl/4:6.3f}   |   {100*zz:6.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/openfast.yaml")
    main(load_config(ap.parse_args().config))
