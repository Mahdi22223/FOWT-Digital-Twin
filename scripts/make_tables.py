"""LaTeX tables of the paper from the CSV outputs (results/openfast, results/focal) -> results/tables/.

  vs_results    Table 5 (table5_revised.tex) and Table A3 (table5_per_sensor.tex)
  l1_channels   Table A2 (table_vs_l1_channels.tex)
  robust        Table A4 (table_vs_robust.tex)
  shape_check   Table A1 (table_shape_check.tex)
  focal_sysid   Table 6 (table_focal_sysid.tex)
  focal_vs      Table 7 (table_focal_vs.tex)
Table 3 combines table3_revised.csv, corr_dmd_summary.csv and lin_modes.csv (see docs/reproduce.md).
Usage: python scripts/make_tables.py [all|<name> ...] [--openfast configs/openfast.yaml] [--focal configs/focal.yaml]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eftwin.config import ensure_dir, load_config
from eftwin.studies.openfast import SENSOR_NAME


def sci(v):
    e = int(np.floor(np.log10(v))); m = v / 10 ** e
    return f"${m:.0f}\\times10^{{{e}}}$"


def f3(x):
    return f"{x:.3f}" if x > -10 else "$<\\!-10$"


def fmt2(v):
    if v < -1:
        return r"$<\!-1$"
    return f"${v:.2f}$" if v < 0 else f"{v:.2f}"


def table5(R, L):
    ps = pd.read_csv(R / "blocked_cases_per_sensor.csv").rename(columns={"cond": "kappa"})
    ly = pd.read_csv(R / "blocked_layouts.csv").rename(columns={"cond": "kappa"})
    M = ["gappyPOD", "OLS", "FIR", "DMD", "KF"]
    SHORT = ["Gappy POD", "Static OLS", "Wiener", "Projection", "Kalman form"]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Virtual-sensing accuracy (mean $R^2$ over the hidden channels, minimum in brackets) for the",
             r"scenarios of Table~\ref{tab:vs_scenarios} and two common sparse layouts. Time-blocked leave-one-OLC-out:",
             r"neither the test OLC nor the test time interval is used for training or for selecting hyper-parameters.",
             r"Cases A--C: the OLC of",
             r"Table~\ref{tab:vs_scenarios}; layouts L1 (nacelle accelerometer) and L2 (nacelle accelerometer and",
             r"tower-base moment): median over the 12 held-out OLCs. $\kappa$: condition number of",
             r"$\boldsymbol{\Phi}_{sparse}$ at the selected rank.}",
             r"\label{tab:vs_results}", r"\footnotesize", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{ll c c ccccc}", r"\toprule",
             r" & & & & \multicolumn{3}{c}{\textit{Baselines}} & \multicolumn{2}{c}{\textit{Hankel-DMD}}\\",
             r"\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
             r"Dir. & Scenario & $q$ & $\kappa$ & " + " & ".join(rf"\textbf{{{s}}}" for s in SHORT) + r"\\",
             r"\midrule"]
    for d in ("FA", "SS"):
        first = True
        for sc in ("Case A", "Case B", "Case C"):
            g = ps[(ps.direction == d) & (ps.scenario == sc)]
            q = int(g.q.iloc[0]); kap = g[g.method == "DMD"].kappa.iloc[0]
            cells = []
            for m in M:
                gm = g[g.method == m]
                cells.append(f"{f3(gm.R2.mean())} [{f3(gm.R2.min())}]")
            lines.append(f"{d if first else ''} & {sc} & {q} & {sci(kap)} & " + " & ".join(cells) + r"\\")
            first = False
        for sc, lab in (("L1 nacelle acc.", "L1"), ("L2 nacelle acc. + base moment", "L2")):
            g = ly[(ly.direction == d) & (ly.scenario == sc)]
            q = int(g.q.iloc[0]); kap = g[g.method == "DMD"].kappa.median()
            cells = []
            for m in M:
                gm = g[g.method == m]
                cells.append(f"{f3(gm.R2_mean.median())} [{f3(gm.R2_min.median())}]")
            lines.append(f" & {lab} & {q} & {sci(kap)} & " + " & ".join(cells) + r"\\")
        lines.append(r"\midrule" if d == "FA" else r"\bottomrule")
    lines += [r"\end{tabular}}", r"\end{table}"]
    (L / "table5_revised.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Per-sensor reconstruction accuracy for Cases A--C ($R^2$; NRMSE in \% for the two Hankel-DMD",
             r"estimators, Eq.~\eqref{eq:metrics}).}", r"\label{tab:vs_per_sensor}", r"\footnotesize",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lll ccccc cc}", r"\toprule",
             r" & & & \multicolumn{3}{c}{\textit{Baselines}} & \multicolumn{2}{c}{\textit{Hankel-DMD}} & & \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
             r"Dir. & Scenario & Hidden sensor & " + " & ".join(SHORT) + r" & NRMSE proj. & NRMSE Kalman\\", r"\midrule"]
    for d in ("FA", "SS"):
        for sc in ("Case A", "Case B", "Case C"):
            g = ps[(ps.direction == d) & (ps.scenario == sc)]
            for k, s in enumerate(sorted(g.sensor.unique())):
                gs = g[g.sensor == s].set_index("method")
                cells = [f3(gs.loc[m, "R2"]) for m in M]
                n1 = gs.loc["DMD", "NRMSE"]; n2 = gs.loc["KF", "NRMSE"]
                lines.append(f"{d if (sc == 'Case A' and k == 0) else ''} & {sc if k == 0 else ''} & "
                             f"{s} ({SENSOR_NAME[s]}) & " + " & ".join(cells) + f" & {n1:.2f} & {n2:.2f}" + r"\\")
        lines.append(r"\midrule" if d == "FA" else r"\bottomrule")
    lines += [r"\end{tabular}}", r"\end{table}"]
    (L / "table5_per_sensor.tex").write_text("\n".join(lines) + "\n")


def l1_channels(R, L):
    t = pd.read_csv(R / "blocked_L1L2_per_channel.csv")
    t = t[t["mask"].astype(str) == "8"]
    SEN = [0, 2, 3, 4, 5, 6, 7, 9, 13, 16, 17]
    GROUPS = [(r"\textit{Baselines}", [("gappyPOD", "Gappy POD"), ("OLS", "Static least squares"),
                                       ("FIR", "Wiener filter (per layout)")]),
              (r"\textit{Hankel-DMD}", [("DMD", "Projection"), ("KF", "Kalman form (stochastic extension)")])]
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Reconstruction from the nacelle accelerometer alone (layout L1): $R^2$ of selected hidden channels",
             r"(median over the 12 held-out OLCs, time-blocked protocol). Sensors 0--7: accelerations, 9--17: bending",
             r"moments, numbered from the tower base; the node of the first coupled mode lies between sensors 4 and 5.}",
             r"\label{tab:vs_l1_channels}", r"\footnotesize", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{ll " + "c" * len(SEN) + "}", r"\toprule",
             r"Dir. & Estimator & " + " & ".join(str(s) for s in SEN) + r"\\", r"\midrule"]
    for d in ("FA", "SS"):
        first = True
        for glabel, M in GROUPS:
            lines.append((d if first else "") + " & " + glabel + " & " * len(SEN) + r"\\")
            first = False
            for m, lab in M:
                g = t[(t.direction == d) & (t.method == m)].groupby("sensor").R2.median()
                lines.append(" & \\quad " + lab + " & " + " & ".join(fmt2(g[s_]) for s_ in SEN) + r"\\")
        lines.append(r"\midrule" if d == "FA" else r"\bottomrule")
    lines += [r"\end{tabular}}", r"\end{table}"]
    (L / "table_vs_l1_channels.tex").write_text("\n".join(lines) + "\n")


def robust(R, L):
    rb = pd.concat([pd.read_csv(R / f"blocked_robust_{d}.csv") for d in ("FA", "SS")])
    rb = rb[rb["mask"] != "8 9"]
    main = pd.read_csv(R / "blocked_summary_random.csv")
    M = ["OLS", "FIR", "DMD", "KF"]
    TESTS = ["Main study", "T1: 600 s of one OLC", "T2: Hs 1-2 m -> Hs 3-4 m", "T2: Hs 3-4 m -> Hs 1-2 m", "T3: 5% sensor noise"]
    LAB = {"Main study": "Main study (Fig.~\\ref{fig:vs_sensors})", "T1: 600 s of one OLC": "T1: 600 s of one OLC",
           "T2: Hs 1-2 m -> Hs 3-4 m": "T2: $H_s$ 1--2 m $\\rightarrow$ 3--4 m",
           "T2: Hs 3-4 m -> Hs 1-2 m": "T2: $H_s$ 3--4 m $\\rightarrow$ 1--2 m", "T3: 5% sensor noise": "T3: 5\\% sensor noise"}
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Pre-declared robustness tests (time-blocked; hyper-parameters fixed to the values selected in the",
             r"main study). Median $R^2$ of the hidden channels over the random layouts with $q$ active sensors and the",
             r"test OLCs. Baselines: S, static least squares; W, Wiener filter. Hankel-DMD: P, projection; K, Kalman form.}",
             r"\label{tab:vs_robust}", r"\footnotesize", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{ll cccc cccc cccc}", r"\toprule",
             r" & & \multicolumn{4}{c}{$q=1$} & \multicolumn{4}{c}{$q=2$} & \multicolumn{4}{c}{$q=3$}\\",
             r"Test & Dir. & S & W & P & K & S & W & P & K & S & W & P & K\\", r"\midrule"]
    out = []
    for t in TESTS:
        for d in ("FA", "SS"):
            cells = []
            for q in (1, 2, 3):
                for m in M:
                    if t == "Main study":
                        v = main[(main.direction == d) & (main.q == q) & (main.method == m)]["median"].iloc[0]
                    else:
                        v = rb[(rb.test_label == t) & (rb.direction == d) & (rb.q == q) & (rb.method == m)].R2.median()
                    cells.append(f"{v:.2f}")
                    out.append(dict(test=t, direction=d, q=q, method=m, R2=v))
            lines.append(f"{LAB[t] if d == 'FA' else ''} & {d} & " + " & ".join(cells) + r"\\")
        lines.append(r"\midrule" if t != TESTS[-1] else r"\bottomrule")
    lines += [r"\end{tabular}}", r"\end{table}"]
    (L / "table_vs_robust.tex").write_text("\n".join(lines) + "\n")
    pd.DataFrame(out).to_csv(R / "blocked_robust_summary.csv", index=False)


def shape_check(R, L, nodes):
    t = pd.read_csv(R / "shape_check.csv")
    rows = []
    for _, r in t.iterrows():
        cells = []
        for meth in ("DMD", "SSI"):
            cells += [f"{r[f'{meth}_R2_tower']:.2f}", f"{r[f'{meth}_R2_platform']:.3f}", f"{r[f'{meth}_MAC']:.2f}",
                      f"{r[f'{meth}_platform_pct']:.0f}"]
        rows.append(f"{'Fore-aft' if r.direction == 'FA' else 'Side-to-side'} {r['mode']} & " + " & ".join(cells) + r"\\")
    node_txt = ", ".join(str(n) for n in nodes[:-1]) + f" and {nodes[-1]}"
    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\caption{Check of the identified acceleration mode shapes against the kinematics of the simulation model.",
           r"$R^2_{tower}$: least-squares fit with the two assumed tower modes of ElastoDyn only (fixed-base kinematics);",
           r"$R^2_{+pl}$: fit with platform translation and rotation added; MAC with the assumed tower mode of the same",
           rf"order; share of platform rigid-body motion in the fitted shape. Gauge heights: ElastoDyn tower nodes {node_txt} of 20 (11.9--85.7~m).}}",
           r"\label{tab:shape_check}", r"\footnotesize",
           r"\begin{tabular}{l cccc cccc}", r"\toprule",
           r" & \multicolumn{4}{c}{\textbf{Hankel-DMD}} & \multicolumn{4}{c}{\textbf{SSI-COV}}\\",
           r"Mode & $R^2_{tower}$ & $R^2_{+pl}$ & MAC & platform (\%) & $R^2_{tower}$ & $R^2_{+pl}$ & MAC & platform (\%)\\",
           r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (L / "table_shape_check.tex").write_text("\n".join(tex) + "\n")


def focal_sysid(R, L):
    S = pd.read_csv(R / "focal_sysid_summary.csv").set_index(["direction", "mode"])
    Rr = pd.read_csv(R / "focal_resp_dmd_summary.csv").set_index(["direction", "mode"])

    def tower(d):
        r, q = S.loc[(d, "Tower 1st")], Rr.loc[(d, "Tower 1st")]
        ref = {"FA": "0.409$^{a}$", "SS": "0.425$^{a}$"}[d]
        return (f"1st {d} tower & {r.FDD_f:.3f} & {r.SSI_f:.3f} & {r.SSI_zeta:.1f} & "
                f"{r.DMD_f:.3f} [{r.DMD_f_q25:.3f}--{r.DMD_f_q75:.3f}] & {r.DMD_zeta:.1f} [{r.DMD_z_q25:.1f}--{r.DMD_z_q75:.1f}] & "
                f"{q.f:.3f} [{q.f_q25:.3f}--{q.f_q75:.3f}] & {q.z:.1f} [{q.z_q25:.1f}--{q.z_q75:.1f}] & {ref} & "
                f"{r.MAC_DMD_SSI:.2f}" + r"\\")

    a, b = S.loc[("FA", "3P")], S.loc[("SS", "3P")]
    qa, qb = Rr.loc[("FA", "3P")], Rr.loc[("SS", "3P")]
    roll, pitch = S.loc[("SS", "Platform roll")], S.loc[("FA", "Platform pitch")]
    lines = [r"""\begin{table}[htbp]
\centering
\caption{Modal identification from the FOCAL Campaign~IV basin tests (full scale; four runs with the same wind input and
different wave realisations; four channels per direction; 0.25--5~Hz, platform modes 0.01--0.2~Hz) and comparison
with published measurements on the same model. Hankel-DMD: medians [inter-quartile range] over $d\in\{43,86,129\}$
and $r\in[10,50]$, the same delay windows in units of the first-mode period as in Table~\ref{tab:sys_id_comparison},
applied to the output correlation functions (corr.) or to the response data (resp.). SSI-COV: medians over stabilised
orders ($i=86$). FDD: peak of the first singular value. MAC between the de-standardised Hankel-DMD (corr.) and SSI-COV
shapes.}
\label{tab:focal_sysid}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{l c cc cc cc c c}
\toprule
\textbf{Mode} & \textbf{FDD} & \multicolumn{2}{c}{\textbf{SSI-COV}} & \multicolumn{2}{c}{\textbf{Hankel-DMD (corr.)}} &
\multicolumn{2}{c}{\textbf{Hankel-DMD (resp.)}} & \textbf{Published} & \textbf{MAC}\\
 & $f$ (Hz) & $f$ (Hz) & $\zeta$ (\%) & $f$ (Hz) & $\zeta$ (\%) & $f$ (Hz) & $\zeta$ (\%) & $f$ (Hz) & DMD--SSI\\
\midrule""", tower("FA"), tower("SS"),
             r"2nd FA / SS tower & \multicolumn{9}{c}{not identifiable from the available channels (Section~\ref{sec:focal_sysid})}\\",
             r"\midrule",
             (f"3P (FA / SS) & {a.FDD_f:.3f} / {b.FDD_f:.3f} & {a.SSI_f:.3f} / {b.SSI_f:.3f} & {a.SSI_zeta:.1f} / {b.SSI_zeta:.1f} & "
              f"{a.DMD_f:.3f} / {b.DMD_f:.3f} & {a.DMD_zeta:.1f} / {b.DMD_zeta:.1f} & {qa.f:.3f} / {qb.f:.3f} & "
              f"{qa.z:.1f} / {qb.z:.1f} & 0.366$^{{b}}$ & {a.MAC_DMD_SSI:.2f} / {b.MAC_DMD_SSI:.2f}" + r"\\"),
             (f"Platform roll & {roll.FDD_f:.4f} & {roll.SSI_f:.4f} & {roll.SSI_zeta:.1f} & {roll.DMD_f:.4f} [{roll.DMD_f_q25:.4f}--"
              f"{roll.DMD_f_q75:.4f}] & {roll.DMD_zeta:.1f} [{roll.DMD_z_q25:.1f}--{roll.DMD_z_q75:.1f}] & -- & -- & "
              f"0.0322--0.0323$^{{c}}$ & {roll.MAC_DMD_SSI:.2f}" + r"\\"),
             (f"Platform pitch & {pitch.FDD_f:.4f} & {pitch.SSI_f:.4f} & {pitch.SSI_zeta:.1f} & {pitch.DMD_f:.4f} [{pitch.DMD_f_q25:.4f}--"
              f"{pitch.DMD_f_q75:.4f}] & {pitch.DMD_zeta:.1f} [{pitch.DMD_z_q25:.1f}--{pitch.DMD_z_q75:.1f}] & -- & -- & "
              f"0.0318--0.0325$^{{c}}$ & {pitch.MAC_DMD_SSI:.2f}" + r"\\"),
             r"""\bottomrule
\end{tabular}}
\par\smallskip\raggedright\scriptsize
$^{a}$ Hammer tests of the floating system \cite{Fowler2023ISOPE}. $^{b}$ Three times the mean rotor frequency,
$3\bar{\Omega}/2\pi$. $^{c}$ Free-decay tests, periods of 31.0--31.1~s (roll) and 30.8--31.4~s (pitch)
\cite{Fowler2023ISOPE,Fowler2023FOCAL}; identified periods: roll 31.1~s, pitch 35.3--35.6~s.
\end{table}"""]
    (L / "table_focal_sysid.tex").write_text("\n".join(lines) + "\n")


def focal_vs(R, L):
    S = pd.read_csv(R / "focal_vs_summary.csv")
    B = pd.read_csv(R / "focal_coherence_bound.csv")
    COLS = [("N", 2), ("N", 3), ("N+Bacc", 2), ("N+Bacc", 3), ("loo0", 0), ("loo1", 1), ("N+Bacc+Bmom", 2), ("loo3", 3)]
    GROUPS = [(r"\textit{Baselines}", [("gappyPOD", "Gappy POD"), ("OLS", "Static least squares"),
                                       ("FIR", "Wiener filter (per layout)")]),
              (r"\textit{Hankel-DMD}", [("DMD", "Projection"), ("KF", "Kalman form (stochastic extension)")])]

    def val(d, lay, ch, m):
        g = S[(S.direction == d) & (S.layout == lay) & (S.channel == ch) & (S.method == m)]
        return "--" if not len(g) else fmt2(g["median"].iloc[0])

    def bound(d, lay, ch):
        g = B[(B.direction == d) & (B.layout == lay) & (B.channel == ch)]
        return f"{g['R2_bound'].iloc[0]:.2f}" if len(g) else "--"

    lines = [r"""\begin{table}[htbp]
\centering
\caption{Virtual sensing on the FOCAL Campaign~IV basin tests: $R^2$ of the hidden channels against the measured
signals (median over the four held-out runs; time-blocked leave-one-run-out, hyper-parameters cross-fitted between run
pairs; 0.25--5~Hz). Active sensors: N, nacelle (tower-top) accelerometer; P, platform accelerometer at the tower base.
Hidden channels: a$_N$, a$_P$, nacelle and platform acceleration; M$_T$, M$_B$, tower-top and tower-base bending moment.
Coherence bound: fraction of the variance of the hidden channel that is linearly coherent with the active channels
(multiple coherence of the measured data), an upper limit for any linear estimator.}
\label{tab:focal_vs}
\footnotesize
\resizebox{\textwidth}{!}{%
\begin{tabular}{ll cc cc cccc}
\toprule
 & & \multicolumn{2}{c}{N} & \multicolumn{2}{c}{N + P} & \multicolumn{4}{c}{Three of four sensors}\\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-10}
Dir. & Estimator & M$_T$ & M$_B$ & M$_T$ & M$_B$ & a$_N$ & a$_P$ & M$_T$ & M$_B$\\
\midrule"""]
    for d in ("FA", "SS"):
        first = True
        for glabel, meths in GROUPS:
            lines.append((d if first else "") + " & " + glabel + r" & & & & & & & & \\")
            first = False
            for m, name in meths:
                lines.append(" & \\quad " + name + " & " + " & ".join(val(d, lay, ch, m) for lay, ch in COLS) + r"\\")
        lines.append(r" & \textit{Coherence bound} & " + " & ".join(r"\textit{" + bound(d, lay, ch) + "}" for lay, ch in COLS) + r"\\")
        lines.append(r"\midrule" if d == "FA" else r"\bottomrule")
    lines.append(r"""\end{tabular}}
\end{table}""")
    (L / "table_focal_vs.tex").write_text("\n".join(lines) + "\n")


def main(names, of_cfg, fc_cfg, out_dir):
    L = ensure_dir(out_dir)
    R = Path(of_cfg["paths"]["results_dir"]); RF = Path(fc_cfg["paths"]["results_dir"])
    jobs = {"vs_results": lambda: table5(R, L), "l1_channels": lambda: l1_channels(R, L), "robust": lambda: robust(R, L),
            "shape_check": lambda: shape_check(R, L, of_cfg["shape_check"]["gauge_nodes"]),
            "focal_sysid": lambda: focal_sysid(RF, L), "focal_vs": lambda: focal_vs(RF, L)}
    for name, job in jobs.items():
        if "all" in names or name in names:
            try:
                job()
                print("written:", name)
            except FileNotFoundError as e:
                print(f"skipped {name}: missing input ({e}); run the corresponding analysis first")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", default=["all"])
    ap.add_argument("--openfast", default="configs/openfast.yaml")
    ap.add_argument("--focal", default="configs/focal.yaml")
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    main(a.names, load_config(a.openfast), load_config(a.focal), a.out)
