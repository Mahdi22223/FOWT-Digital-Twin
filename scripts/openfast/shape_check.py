"""Physical check of the identified tower mode shapes against the ElastoDyn kinematics (Table shape_check).

In ElastoDyn the fore-aft tower acceleration at height z is a linear combination of the platform translation
(constant in z), the platform rotation (linear in z) and the two assumed tower modes phi_1(z), phi_2(z). An
identified acceleration shape must therefore lie in span{1, z, phi_1, phi_2}. Least-squares fits with the
assumed modes only (fixed-base kinematics) and with the platform motion added give the share of rigid-body
motion in each identified mode.
Input: results/openfast/modeshapes.npz (identify.py). Output: results/openfast/shape_check.csv
Usage: python scripts/openfast/shape_check.py [--config configs/openfast.yaml] [--heights equidistant]
"""
import argparse

import numpy as np
import pandas as pd

from eftwin.config import load_config
from eftwin.io import gauge_heights
from eftwin.modal import real_shape
from eftwin.studies.openfast import OpenFASTStudy


def phi(c, x):
    return sum(ci * x ** (i + 2) for i, ci in enumerate(c))


def main(cfg, heights="nodes"):
    S_ = OpenFASTStudy(cfg)
    sc = cfg["shape_check"]
    BASE, TOP, NODES = sc["tower_base"], sc["tower_top"], sc["tower_nodes"]
    if heights == "equidistant":
        z = np.linspace(BASE, TOP, 9)
    else:
        z = gauge_heights(sc["gauge_nodes"], BASE, TOP, NODES)
    x = (z - BASE) / (TOP - BASE)
    shapes = np.load(S_.results / "modeshapes.npz")
    COEF = sc["assumed_modes"]
    print(f"gauge heights [m]: {np.round(z, 1)}")
    for d in ("FA", "SS"):
        B = np.column_stack([np.ones_like(z), z - BASE, phi(COEF[d][0], x), phi(COEF[d][1], x)])
        for m in ("1st", "2nd"):
            for meth in ("DMD", "SSI"):
                a = real_shape(shapes[f"{d}_{m}_{meth}"][:9])
                coef, *_ = np.linalg.lstsq(B, a, rcond=None)
                fit = B @ coef
                r2 = 1 - ((a - fit) ** 2).sum() / ((a - a.mean()) ** 2).sum()
                mac_el = (a @ B[:, 2 + (m == "2nd")]) ** 2 / ((a @ a) * (B[:, 2 + (m == "2nd")] @ B[:, 2 + (m == "2nd")]))
                rb = B[:, :2] @ coef[:2]; el = B[:, 2:] @ coef[2:]
                share = (rb ** 2).sum() / ((rb ** 2).sum() + (el ** 2).sum())
                sign = np.sign(a)
                node = None
                k = np.where(np.diff(sign) != 0)[0]
                if len(k):
                    k = k[0]; node = z[k] - a[k] * (z[k + 1] - z[k]) / (a[k + 1] - a[k])
                print(f"{d} {m} {meth}: fit R2 = {r2:.4f} | MAC(shape, assumed mode {m}) = {mac_el:.3f} | "
                      f"rigid-body share = {100 * share:.0f}% | coefs (surge, pitch/m, phi1, phi2) = {np.round(coef, 3)}"
                      + (f" | node at {node:.1f} m" if node is not None else ""))

    # nested fits: fixed-base assumed modes vs. + platform motion (values of the LaTeX table)
    rows = []
    for d in ("FA", "SS"):
        B = np.column_stack([np.ones_like(z), z - BASE, phi(COEF[d][0], x), phi(COEF[d][1], x)])
        for m in ("1st", "2nd"):
            row = dict(direction=d, mode=m)
            for meth in ("DMD", "SSI"):
                a = real_shape(shapes[f"{d}_{m}_{meth}"][:9])

                def r2fit(Bm):
                    c, *_ = np.linalg.lstsq(Bm, a, rcond=None)
                    return 1 - ((a - Bm @ c) ** 2).sum() / ((a - a.mean()) ** 2).sum(), c
                r2_fixed, _ = r2fit(B[:, 2:])
                r2_full, coef = r2fit(B)
                j = 2 + (m == "2nd")
                mac = (a @ B[:, j]) ** 2 / ((a @ a) * (B[:, j] @ B[:, j]))
                rb = B[:, :2] @ coef[:2]; el = B[:, 2:] @ coef[2:]
                share = 100 * (rb ** 2).sum() / ((rb ** 2).sum() + (el ** 2).sum())
                row.update({f"{meth}_R2_tower": r2_fixed, f"{meth}_R2_platform": r2_full, f"{meth}_MAC": mac,
                            f"{meth}_platform_pct": share})
            rows.append(row)
    pd.DataFrame(rows).to_csv(S_.results / "shape_check.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/openfast.yaml")
    ap.add_argument("--heights", choices=("nodes", "equidistant"), default="nodes",
                    help="gauge heights: ElastoDyn nodes of the config (default) or equidistant 10-87.6 m")
    a = ap.parse_args()
    main(load_config(a.config), a.heights)
