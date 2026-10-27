"""Eigen-analysis of the OpenFAST linearisation (case_1.1.lin) and comparison with the identified tower modes
(column "Linear (OpenFAST)" of Table 3; Section 4.2).

Mode shapes in the output space are C v (rows of the tower-gauge outputs). Every identified mode is paired with
the linear mode below 6 Hz whose tower-acceleration energy is dominated (> 50 %) by the same direction and whose
shape agrees best (MAC of the acceleration and moment components, small frequency penalty).
Inputs: the .lin file (paths.lin_file) and results/openfast/modeshapes.npz. Output: results/openfast/lin_modes.csv
Usage: python scripts/openfast/linear_reference.py [--config configs/openfast.yaml]
"""
import argparse
import re

import numpy as np
import pandas as pd

from eftwin.config import load_config
from eftwin.modal import mac
from eftwin.studies.openfast import OpenFASTStudy


def read_lin(path):
    txt = open(path).read().splitlines()

    def block(name):
        i = next(k for k, s in enumerate(txt) if s.startswith(f"{name}:"))
        nr, nc = map(int, re.findall(r"(\d+) x (\d+)", txt[i])[0])
        return np.array([[float(v) for v in txt[i + 1 + r].split()] for r in range(nr)])

    def descriptions(header, n):
        i = next(k for k, s in enumerate(txt) if s.startswith(header))
        return [s[s.find("ED ") if "ED " in s else 60:].strip() for s in txt[i + 3:i + 3 + n]]

    A = block("A"); C = block("C")
    return A, C, descriptions("Order of continuous states", A.shape[0]), descriptions("Order of outputs", C.shape[0])


def main(cfg):
    S_ = OpenFASTStudy(cfg)
    A, C, sd, od = read_lin(cfg["paths"]["lin_file"])
    ns = A.shape[0]
    lam, V = np.linalg.eig(A)
    f = np.abs(lam.imag) / (2 * np.pi); z = -lam.real / np.abs(lam)
    nd = ns // 2
    rows = []
    for j in np.argsort(f):
        if lam[j].imag <= 0:
            continue
        part = np.abs(V[:nd, j]) / np.abs(V[:nd, j]).max()
        dom = [sd[k].split("(")[0].replace("ED ", "").strip() for k in np.argsort(-part)[:2]]
        rows.append((f[j], 100 * z[j], dom, j))
    print("Linearised modes (f < 6 Hz):")
    for fr, zz, dom, j in rows:
        if fr < 6:
            print(f"  {fr:7.3f} Hz  zeta {zz:6.2f} %   {dom[0]} | {dom[1]}")

    def out_idx(pattern):
        return [k for k, s in enumerate(od) if re.search(rf"\b{pattern},", s)]

    acc_fa = [out_idx(rf"TwHt{g}ALxt")[0] for g in range(1, 10)]
    acc_ss = [out_idx(rf"TwHt{g}ALyt")[0] for g in range(1, 10)]
    mom_fa = [out_idx(rf"TwHt{g}MLyt")[0] for g in range(1, 10)]
    mom_ss = [out_idx(rf"TwHt{g}MLxt")[0] for g in range(1, 10)]
    shapes = np.load(S_.results / "modeshapes.npz")
    res = []
    for d, acc, mom in (("FA", acc_fa, mom_fa), ("SS", acc_ss, mom_ss)):
        for m in ("1st", "2nd"):
            ident = shapes[f"{d}_{m}_SSI"]; idd = shapes[f"{d}_{m}_DMD"]
            cands = []
            for fr, zz, dom, j in rows:
                if fr > 6 or fr < 0.1:
                    continue
                y = C @ V[:, j]
                efa = np.sum(np.abs(y[acc_fa]) ** 2); ess = np.sum(np.abs(y[acc_ss]) ** 2)
                share = efa / (efa + ess) if d == "FA" else ess / (efa + ess)
                if share > 0.5:
                    score = 0.5 * (mac(y[acc], ident[:9]) + mac(y[mom], ident[9:])) - 0.01 * abs(fr - shapes[f'{d}_{m}_f'][1])
                    cands.append((score, fr, zz, j, y, share))
            cands.sort(key=lambda t: -t[0])
            _, fr, zz, j, y, share = cands[0]
            r = dict(direction=d, mode=m, f=fr, zeta_pct=zz, share=share, mac_acc=mac(y[acc], ident[:9]),
                     mac_mom=mac(y[mom], ident[9:]), mac_acc_dmd=mac(y[acc], idd[:9]), mac_mom_dmd=mac(y[mom], idd[9:]))
            res.append(r)
            print(f"{d} {m}: tracked linear mode f = {fr:.3f} Hz, zeta = {zz:.2f} %, direction share {share:.2f} | "
                  f"MAC(lin, SSI) acc {r['mac_acc']:.3f} mom {r['mac_mom']:.3f} | MAC(lin, DMD) acc {r['mac_acc_dmd']:.3f} "
                  f"mom {r['mac_mom_dmd']:.3f}")
    # the modes listed in the original Table 3 as linear references
    for fr0 in (0.477, 1.542, 2.003):
        j = min(rows, key=lambda t: abs(t[0] - fr0))[3]
        y = C @ V[:, j]
        print(f"linear mode {fr0} Hz: MAC with identified 2nd FA (SSI) acc {mac(y[acc_fa], shapes['FA_2nd_SSI'][:9]):.3f}, "
              f"with 1st FA acc {mac(y[acc_fa], shapes['FA_1st_SSI'][:9]):.3f}; tower-top FA acc / blade-1 root flap moment ratio "
              f"{abs(y[acc_fa[-1]]):.2e} / {abs(y[out_idx('RootMyc1')[0]]):.2e}")
    pd.DataFrame(res).to_csv(S_.results / "lin_modes.csv", index=False)

    print("\nLinear modes below 6 Hz: tower-acceleration energy split and MAC with the identified shapes")
    print("   f [Hz]  zeta[%]  FA share  MAC(1st FA) MAC(2nd FA) MAC(1st SS) MAC(2nd SS)  TT acc |y|")
    for fr, zz, dom, j in rows:
        if fr > 6 or fr < 0.1:
            continue
        y = C @ V[:, j]
        efa = np.sum(np.abs(y[acc_fa]) ** 2); ess = np.sum(np.abs(y[acc_ss]) ** 2)
        sh = efa / (efa + ess)
        print(f"  {fr:7.3f}  {zz:6.2f}   {sh:6.2f}     " +
              "  ".join(f"{mac(y[acc_fa if dd == 'FA' else acc_ss], shapes[f'{dd}_{mm}_SSI'][:9]):9.3f}"
                        for dd, mm in (("FA", "1st"), ("FA", "2nd"), ("SS", "1st"), ("SS", "2nd"))) +
              f"   {np.sqrt(efa + ess):.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/openfast.yaml")
    main(load_config(ap.parse_args().config))
