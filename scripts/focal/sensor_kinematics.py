"""What does the FOCAL tower-base accelerometer measure? Test against rigid-platform kinematics (Section 5.1).

For a body-fixed accelerometer at height z above the motion reference point (small rotations)
    a_X = xdd + z thdd - g th,   a_Y = ydd - z phdd + g ph      (th: platform pitch, ph: roll)
  free         unconstrained least squares in 0.03-0.25 Hz (lever arm z and gravity coefficient fitted)
  constrained  translation coefficient 1 and gravity coefficient g fixed; lever arm z, R^2 and the coherence of
               each accelerometer with the rigid-body prediction (wave band and 0.39-0.45 Hz)
Derivatives are taken in the frequency domain.
Usage: python scripts/focal/sensor_kinematics.py {constrained,free} [--config configs/focal.yaml]
"""
import argparse

import numpy as np
from scipy.signal import coherence

from eftwin.config import load_config
from eftwin.studies.focal import FocalStudy

G = 9.81


def band(x, dt, order=0, lo=0.03, hi=0.25):
    X = np.fft.rfft(x - x.mean()); f = np.fft.rfftfreq(len(x), dt)
    W = ((f >= lo) & (f <= hi)).astype(float)
    return np.fft.irfft(X * W * (2j * np.pi * f) ** order, n=len(x))


def stationary(S_, run):
    """All channels of one run over the stationary window, and a name -> column map."""
    cols, x = S_.data.load(run)
    return x[S_.data.window()], {k: i for i, k in enumerate(cols)}


def free(S_):
    rows = []
    for run in S_.runs:
        seg, c = stationary(S_, run)
        th = np.deg2rad(seg[:, c["Pitch"]]); ph = np.deg2rad(seg[:, c["Roll"]])
        for dirn, acc_names, trans, rot in (("X", ("towerBotAccX", "accelNacelleAx"), "Surge", th),
                                             ("Y", ("towerBotAccY", "accelNacelleAy"), "Sway", ph)):
            tdd = band(seg[:, c[trans]], S_.dt, 2); rdd = band(rot, S_.dt, 2); r0 = band(rot, S_.dt, 0)
            A = np.column_stack([tdd, rdd, r0])
            for an in acc_names:
                y = band(seg[:, c[an]], S_.dt, 0)
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                r2 = 1 - ((y - A @ coef) ** 2).sum() / (y ** 2).sum()
                rows.append((run[-7:], an, coef[0], coef[1], coef[2], r2))
    print(f"{'run':8s} {'channel':15s} {'c_trans':>8s} {'z (m)':>8s} {'c_g (m/s2/rad)':>15s} {'R2':>6s}")
    for r in rows:
        print(f"{r[0]:8s} {r[1]:15s} {r[2]:8.3f} {r[3]:8.2f} {r[4]:15.3f} {r[5]:6.3f}")


def constrained(S_):
    res = {}
    for run in S_.runs:
        seg, c = stationary(S_, run)
        th = np.deg2rad(seg[:, c["Pitch"]]); ph = np.deg2rad(seg[:, c["Roll"]])
        for an, trans, rot, s_t, s_g in (("towerBotAccX", "Surge", th, 1, -1), ("accelNacelleAx", "Surge", th, 1, -1),
                                         ("towerBotAccY", "Sway", ph, -1, -1), ("accelNacelleAy", "Sway", ph, -1, -1)):
            y = band(seg[:, c[an]], S_.dt); tdd = band(seg[:, c[trans]], S_.dt, 2); rdd = band(rot, S_.dt, 2)
            r0 = band(rot, S_.dt)
            resid = y - s_t * tdd - s_g * G * r0
            z = (rdd @ resid) / (rdd @ rdd)                       # lever arm (sign absorbs the axis convention)
            pred = s_t * tdd + z * rdd + s_g * G * r0
            r2 = 1 - ((y - pred) ** 2).sum() / (y ** 2).sum()
            tdd_w = band(seg[:, c[trans]], S_.dt, 2, 0.02, 1.0); rdd_w = band(rot, S_.dt, 2, 0.02, 1.0)
            r0_w = band(rot, S_.dt, 0, 0.02, 1.0)
            pred_w = s_t * tdd_w + z * rdd_w + s_g * G * r0_w
            f, C = coherence(seg[:, c[an]], pred_w, fs=S_.fs, nperseg=int(200 * S_.fs))

            def cmean(lo, hi):
                m = (f > lo) & (f < hi)
                return C[m].mean()
            res.setdefault(an, []).append((z, r2, cmean(0.05, 0.2), cmean(0.39, 0.45)))
    for an, v in res.items():
        v = np.array(v)
        print(f"{an:15s} lever arm z = {v[:,0].mean():7.1f} m (runs {v[:,0].min():.1f}..{v[:,0].max():.1f}); "
              f"R2 (0.03-0.25 Hz) = {v[:,1].mean():.3f}; coherence with rigid-body prediction: wave band {v[:,2].mean():.2f}, "
              f"0.39-0.45 Hz {v[:,3].mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("constrained", "free"), nargs="?", default="constrained")
    ap.add_argument("--config", default="configs/focal.yaml")
    a = ap.parse_args()
    S_ = FocalStudy(load_config(a.config))
    (constrained if a.what == "constrained" else free)(S_)
