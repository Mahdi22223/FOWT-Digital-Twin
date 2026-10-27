"""Paper figures in the original plotting style -> results/figures/.

  fig5    first tower-mode shapes vs Hankel delay depth (raw data)
  fig6_7  modal dynamics and eigenvalue map, response-domain model of all 12 OLCs (raw data)
  fig8    time-frequency comparison, Cases A-C (results/openfast/vs_excerpts.npy)
  fig9    long-duration reconstruction, Case B layout, OLC 7 (raw data + blocked_hyper.csv)
  fig10   singular values of the Hankel matrix (raw data)
  fig11   forecast horizon (blocked_forecast.csv)
  fig12   accuracy vs number of sensors (blocked_summary_random.csv)
  fig13   reconstruction from the nacelle accelerometer (vs_excerpts.npy)
  figA    sensitivity to the embedding parameters (sysid_sensitivity.csv, corr_dmd_sensitivity.csv)
  fig14   FOCAL identification (results/focal/wideband_*.npz, focal_sysid_summary.csv)
  fig15   FOCAL time-frequency reconstruction (results/focal/intermediate/vs_series_*.npz)
  fig16   FOCAL reconstruction on an unseen run (vs_series_*.npz)
Usage: python scripts/make_figures.py [all|openfast|focal|<name> ...] [--openfast ...] [--focal ...]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.signal import spectrogram  # noqa: E402

from eftwin.config import as_range, ensure_dir, load_config  # noqa: E402
from eftwin.dmd import dmd_from_gram, modal_energy  # noqa: E402
from eftwin.estimators import (delay_windows, project_windows, projection_estimate, sparse_svd,  # noqa: E402
                               tikhonov_filter)
from eftwin.hankel import delay_gram, delay_rows, fold_stats  # noqa: E402
from eftwin.io import DIRECTION_NAME as NAME, gauge_heights  # noqa: E402
from eftwin.modal import displacement_shape, mac  # noqa: E402
from eftwin.plotting import sans, serif, use_paper_style  # noqa: E402
from eftwin.studies.focal import FocalStudy  # noqa: E402
from eftwin.studies.openfast import P, OpenFASTStudy  # noqa: E402

use_paper_style()


# ======================================================================= OpenFAST study
class OF:
    def __init__(self, cfg, out):
        self.S = OpenFASTStudy(cfg)
        self.cfg, self.out = cfg, out
        self.R, self.CACHE = self.S.results, self.S.cache
        vc = cfg["virtual_sensing"]
        self.D, self.HALF, self.SCORE0 = vc["delay"], vc["half"], vc["score_start"]
        self.LAST = slice((self.D - 1) * P, self.D * P)
        self.DT, self.FS, self.TRIM = self.S.dt, self.S.fs, cfg["data"]["trim"]
        sc = cfg["shape_check"]
        self.TOWER_H = gauge_heights(sc["gauge_nodes"], sc["tower_base"], sc["tower_top"], sc["tower_nodes"])

    def model_all(self, d, r=24):
        """Response-domain exact Hankel-DMD (d = 60) on all 12 OLCs from the per-case Gram sums."""
        raw = self.S.raw(d)
        st = fold_stats([self.S.case_gram(d, c, raw) for c in self.S.cases], [raw[c] for c in self.S.cases])
        lam, Phi = dmd_from_gram(st["G"], st["C"], r, eig=(st["ev"], st["U"]))
        return st, lam, Phi, raw

    # ------------------------------------------------------------------ Fig. 5
    def fig5(self):
        """First-mode shapes versus Hankel delay depth d = 30, 40, ..., 90 (the range of the identification
        sensitivity of Table 3), response-domain exact Hankel-DMD on all 12 OLCs, rank 35. The Gram sums of every
        depth are the leading blocks of those of d = 90. The band is the range over d (not a confidence interval)."""
        c5 = self.cfg["figures"]["fig5"]
        D_RANGE = as_range(c5["delays"]); RANK = c5["rank"]; TARGET = c5["target_hz"]; DT = self.DT
        cases = list(self.S.cases)
        for d in ("FA", "SS"):
            raw = self.S.raw(d)
            f90 = self.CACHE / f"gram90_{d}.npz"
            if f90.exists():
                z = np.load(f90); grs = [(z[f"G{c}"], z[f"C{c}"], z[f"s1{c}"], z[f"s2{c}"], int(z[f"n{c}"])) for c in cases]
            else:
                grs = [delay_gram(raw[c], max(D_RANGE), chunk=c5["chunk"]) for c in cases]
                ensure_dir(f90.parent)
                np.savez(f90, **{f"{k}{c}": v for c, g in zip(cases, grs) for k, v in zip(("G", "C", "s1", "s2", "n"), g)})
            shapes = []
            for dd in D_RANGE:
                k = P * dd
                sub = [(g[0][:k, :k], g[1][:k, :k], g[2][:k], g[3][:k], g[4]) for g in grs]
                X = np.vstack([raw[c] for c in cases]); mu, sd = X.mean(0), X.std(0)
                G = sum(g[0] for g in sub); C = sum(g[1] for g in sub); s1 = sum(g[2] for g in sub)
                s2 = sum(g[3] for g in sub); n = sum(g[4] for g in sub)
                mut, Dg = np.tile(mu, dd), 1 / np.tile(sd, dd)
                G = Dg[:, None] * (G - np.outer(mut, s1) - np.outer(s1, mut) + n * np.outer(mut, mut)) * Dg[None, :]
                C = Dg[:, None] * (C - np.outer(mut, s1) - np.outer(s2, mut) + n * np.outer(mut, mut)) * Dg[None, :]
                ev, U = np.linalg.eigh(G); o = np.argsort(ev)[::-1][:RANK]; ev, U = ev[o], U[:, o]
                lam, W = np.linalg.eig((U.T @ C @ U) / ev[None, :]); Phi = (C @ U / ev[None, :]) @ W
                fr = np.abs(np.log(lam).imag) / (DT * 2 * np.pi)
                ok = np.where((fr > TARGET[d] - c5["search_hz"]) & (fr < TARGET[d] + c5["search_hz"]))[0]
                if not len(ok):
                    continue
                i = ok[np.argmin(np.abs(fr[ok] - TARGET[d]))]
                shapes.append(displacement_shape(Phi[:9, i] * sd[:9], fr[i]))      # de-standardised acceleration
            S = np.array(shapes); mean = S.mean(0)
            maxdev = np.abs(S - mean).max()
            mac_min = min(mac(S[i], S[j]) for i in range(len(S)) for j in range(i + 1, len(S)))
            fig, ax = plt.subplots(figsize=(3.4, 4.2))
            for k, s in enumerate(S):
                ax.plot(s, self.TOWER_H, color="gray", alpha=0.3, linewidth=0.8, label="Individual d" if k == 0 else None)
            ax.fill_betweenx(self.TOWER_H, S.min(0), S.max(0), color="blue", alpha=0.2, label="Range over d")
            ax.plot(mean, self.TOWER_H, "b-o", linewidth=2, markersize=4, label="Mean over d")
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(f"{NAME[d]} 1st Mode Shape\n(Delay-Depth Sensitivity, d = {D_RANGE[0]}–{D_RANGE[-1]})",
                         fontsize=9)
            ax.set_xlabel("Normalized Displacement"); ax.set_ylabel("Tower Height (m)")
            ax.legend(loc="upper left", fontsize=7); ax.grid(True, alpha=0.5)
            ax.text(0.97, 0.04, f"max deviation from mean: {maxdev:.2f}\nMAC between any two shapes: ≥ {np.floor(mac_min * 1000) / 1000:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.9))
            fig.tight_layout(); fig.savefig(self.out / f"fig5_{d}_probabilistic_mode_shape.png", dpi=300); plt.close(fig)
            print(d, "fig 5: d =", D_RANGE, "| shapes", len(shapes), f"| max deviation {maxdev:.4f}, min pairwise MAC {mac_min:.5f}, "
                  f"max range {(S.max(0) - S.min(0)).max():.4f}", flush=True)

    # ------------------------------------------------------------------ Figs. 6 and 7
    def fig6_7(self):
        DT, D = self.DT, self.D
        for d in ("FA", "SS"):
            st, lam, Phi, raw = self.model_all(d, self.cfg["figures"]["fig6_7"]["rank"])
            f = np.abs(np.log(lam).imag) / (DT * 2 * np.pi)
            Y1 = ((raw[1] - st["mu"]) / st["sd"])
            b = np.linalg.lstsq(Phi, np.concatenate([Y1[k] for k in range(D)]), rcond=None)[0]
            E = modal_energy(Phi, st["G"])
            sel = []
            for lo, hi in ((0.45, 0.56), (3.40, 3.95)):                       # same selection as Section 4 (modal energy)
                m = np.where((f > lo) & (f < hi) & (lam.imag > 0))[0]
                sel.append(m[np.argmax(E[m])])
            time = np.arange(int(30 / DT)) * DT
            fig, ax = plt.subplots(figsize=(7.0, 2.9))
            for idx in sel[::-1]:
                dyn = (b[idx] * lam[idx] ** np.arange(len(time))).real
                ax.plot(time, dyn, label=f"Mode {idx} ({f[idx]:.2f} Hz)", linewidth=1.5)
            ax.set_title(f"{NAME[d]}: Mode Dynamics", fontweight="bold")
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right"); ax.grid(True, alpha=0.3); ax.set_xlim(0, 30)
            fig.tight_layout(); fig.savefig(self.out / f"fig6_{d}_mode_dynamics.png", dpi=300); plt.close(fig)
            hl = ((f > 0.4) & (f < 0.6)) | ((f > 3.4) & (f < 3.95))
            fig, ax = plt.subplots(figsize=(3.6, 3.6))
            th = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(th), np.sin(th), "k--", linewidth=1.2, alpha=0.6, label="Stability Limit")
            ax.scatter(lam[~hl].real, lam[~hl].imag, s=22, c="dodgerblue", alpha=0.6, label="Other Modes",
                       edgecolors="k", linewidth=0.5)
            ax.scatter(lam[hl].real, lam[hl].imag, s=70, c="red", marker="*", label="Tower Candidate Modes(1st/2nd)",
                       zorder=10, edgecolors="k", linewidth=0.5)
            for lo, hi, txt in ((0.4, 0.6, "1st Mode\nCandidate"), (3.4, 3.95, "2nd Mode\nCandidate")):
                m = hl & (f > lo) & (f < hi) & (lam.imag > 0)
                if m.any():
                    e = lam[m][np.argmax(lam[m].real)]
                    ax.annotate(txt, xy=(e.real, e.imag), xytext=(e.real * 0.62, e.imag * 0.62 + (0.12 if lo < 1 else -0.1)),
                                fontsize=7, weight="bold", color="darkred", ha="center", va="center",
                                arrowprops=dict(arrowstyle="-|>", color="gray", lw=0.8))
            ax.set_title(f"{NAME[d]}: Stability Map (Eigenvalues)", fontweight="bold")
            ax.set_xlabel("Real Part"); ax.set_ylabel("Imaginary Part")
            ax.grid(True, linestyle=":", alpha=0.6); ax.legend(loc="lower left", fontsize=6.5)
            ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
            fig.tight_layout(); fig.savefig(self.out / f"fig7_{d}_stability_map.png", dpi=300); plt.close(fig)
            print(d, "fig 6/7:", [(int(i), round(f[i], 3)) for i in sel], flush=True)

    # ------------------------------------------------------------------ Fig. 8
    def fig8(self):
        serif()
        FS = self.FS
        EXC = np.load(self.R / "vs_excerpts.npy", allow_pickle=True).item()
        panels = [("FA|Case A|5", 11, "(a) Case A: Mid-Span Moment (Sensor 11)"),
                  ("FA|Case A|5", 5, "(b) Case A: Mid-Span Accel (Sensor 5)"),
                  ("FA|Case B|7", 17, "(c) Case B: Nacelle Moment (OLC 7)"),
                  ("FA|Case C|2", 17, "(d) Case C: Nacelle Moment (OLC 2)")]
        fig, axes = plt.subplots(4, 2, figsize=(7.0, 8.6))
        for row, (key, s, title) in enumerate(panels):
            e = EXC[key]; i = list(e["hid"]).index(s)
            for col, (k, lab) in enumerate((("truth", "Actual (Hidden)"), ("Hankel-DMD (projection)", "Virtual"))):
                x = e[k][i, :1200] * e["sd"][i]
                fq, tt, S = spectrogram(x, fs=FS, nperseg=128, noverlap=120)
                ax = axes[row, col]
                im = ax.pcolormesh(tt, fq, 10 * np.log10(S + 1e-12), shading="gouraud", cmap="viridis",
                                   vmin=10 * np.log10(S.max()) - 60, vmax=10 * np.log10(S.max()))
                ax.set_ylim(0, 5); ax.set_title(lab, fontsize=8)
                ax.set_ylabel("Frequency [Hz]", fontsize=7); ax.set_xlabel("Time [s]", fontsize=7)
                fig.colorbar(im, ax=ax, label="Power (dB)")
            axes[row, 0].annotate(title, xy=(0, 1.22), xycoords="axes fraction", fontsize=8.5, fontweight="bold")
        fig.tight_layout(); fig.savefig(self.out / "fig8_vs_spectrograms.png", dpi=300); plt.close(fig)
        sans()

    # ------------------------------------------------------------------ Fig. 9
    def fig9(self):
        serif()
        D, HALF, SCORE0, LAST, FS, TRIM = self.D, self.HALF, self.SCORE0, self.LAST, self.FS, self.TRIM
        d, test = "FA", 7
        hyp = pd.read_csv(self.R / "blocked_hyper.csv")
        raw = self.S.raw(d)
        hid = np.array([0, 8, 9, 17]); a = np.setdiff1d(np.arange(P), hid); i8 = 1
        cases = list(self.S.cases)
        tru, rec = [], []
        for h in (0, 1):
            gr = [self.S.half_gram(d, c, 1 - h, raw) for c in cases if c != test]
            st = fold_stats(gr, [raw[c][(1 - h) * HALF:(2 - h) * HALF] for c in cases if c != test])
            Xt = ((raw[test][h * HALF:(h + 1) * HALF] - st["mu"]) / st["sd"]).T
            # projection form (deterministic estimator) with the cross-fitted (r, alpha) of this layout
            hp_ = hyp[(hyp.direction == d) & (hyp.q == 14) & (hyp.method == "DMD") & (hyp.group == 1)].hp.iloc[0].split("|")
            r_p, alpha = int(float(hp_[0])), float(hp_[1])
            lam_p, Phi = dmd_from_gram(st["G"], st["C"], r_p, eig=(st["ev"], st["U"]))
            Uu, sv, Vh = sparse_svd(Phi, delay_rows(a, P, D))
            Wobs = delay_windows(Xt, a, D, pad=True)
            est = projection_estimate(project_windows(Wobs, Uu), tikhonov_filter(sv, alpha), Vh, Phi[LAST][hid])[i8]
            tru.append(raw[test][h * HALF:(h + 1) * HALF, 8]); rec.append(st["mu"][8] + st["sd"][8] * est)
        tru, rec = np.concatenate(tru), np.concatenate(rec)
        t = TRIM / FS + np.arange(len(tru)) / FS
        fig = plt.figure(figsize=(7.0, 7.4))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15])
        ax = fig.add_subplot(gs[0, :])
        ax.plot(t, tru, color="gray", lw=0.5, label="Actual (Hidden)")
        ax.plot(t, rec, color="tab:blue", lw=0.4, ls=":", label="Virtual")
        ax.add_patch(Rectangle((1000, ax.get_ylim()[0]), 20, np.ptp(ax.get_ylim()), fill=False, ec="red", lw=1.2))
        ax.text(1030, ax.get_ylim()[1] * 0.85, "Zoom Region", color="red", fontsize=7)
        ax.set_xlim(t[0], t[-1]); ax.set_xlabel("Time [s]"); ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.3)
        ax = fig.add_subplot(gs[1, :])
        m = (t >= 1000) & (t <= 1020)
        ax.plot(t[m], tru[m], color="tab:blue", lw=1.2, alpha=0.5, label="Actual")
        ax.plot(t[m], rec[m], color="tab:blue", lw=1.2, ls="--", label="Virtual")
        ax.set_title("Detail View: High-Fidelity Tracking (1000 s - 1020 s)", fontweight="bold")
        ax.set_xlim(1000, 1020); ax.set_xlabel("Time [s]"); ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=7); ax.grid(True, alpha=0.3)
        for col, (x, lab) in enumerate(((tru, "Actual (Hidden)"), (rec, "Virtual"))):
            ax = fig.add_subplot(gs[2, col])
            fq, tt, S = spectrogram(x, fs=FS, nperseg=512, noverlap=384)
            im = ax.pcolormesh(tt + t[0], fq, 10 * np.log10(S + 1e-12), shading="auto", cmap="viridis",
                               vmin=10 * np.log10(S.max()) - 60, vmax=10 * np.log10(S.max()))
            ax.set_ylim(0, 5); ax.set_title(lab); ax.set_xlabel("Time [s]"); ax.set_ylabel("Frequency [Hz]")
            fig.colorbar(im, ax=ax, label="Power (dB)")
        fig.tight_layout(); fig.savefig(self.out / "fig9_long_term.png", dpi=300); plt.close(fig)
        mm = np.ones(len(tru), bool); mm[:SCORE0] = False; mm[HALF:HALF + SCORE0] = False
        print("fig 9 R2:", 1 - ((tru[mm] - rec[mm]) ** 2).sum() / ((tru[mm] - tru[mm].mean()) ** 2).sum(), flush=True)
        sans()

    # ------------------------------------------------------------------ Fig. 10
    def fig10(self):
        serif()
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for d, mk in (("FA", ".-"), ):
            st, lam, Phi, raw = self.model_all(d)
            n = sum(self.S.case_gram(d, c, raw)[4] for c in self.S.cases)
            sv = np.sqrt(np.clip(st["ev"], 0, None) * n)[:200]
            ax.semilogy(np.arange(1, 201), sv, "k.-", markersize=2, linewidth=0.8, label="Singular Values")
        ax.axvspan(10, 50, color="blue", alpha=0.2, label=r"Zone I: System Identification [$r\in(10, 50)$]")
        ax.axvspan(50, 120, color="green", alpha=0.2, label=r"Zone II: Virtual Sensing [$r\in(50, 120)$]")
        ax.set_title("SVD Scree Plot (d=60)"); ax.set_xlabel("Rank (k)"); ax.set_ylabel("Energy (Singular Value)")
        ax.legend(loc="upper right", fontsize=7); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); fig.savefig(self.out / "fig10_scree.png", dpi=300); plt.close(fig)
        sans()

    # ------------------------------------------------------------------ Fig. 11 (replaces the Lyapunov figure)
    def fig11(self):
        fo = pd.read_csv(self.R / "blocked_forecast.csv")
        MODE1 = {k: tuple(v) for k, v in self.cfg["figures"]["fig11_first_mode"].items()}
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
        for ax, d in zip(axes, ("FA", "SS")):
            g14 = fo[(fo.direction == d) & (fo.q == 14) & (fo.method == "KF")].groupby("lead_s").R2.median()
            g1 = fo[(fo.direction == d) & (fo.q == 1) & (fo.method == "KF")].groupby("lead_s").R2.median()
            f1, z1 = MODE1[d]; rate = 2 * z1 * 2 * np.pi * f1
            hh = np.linspace(0, 4.2, 200); hreq = np.log(1 / 0.9) / rate
            ax.plot(g14.index, g14.values, "k-o", markersize=3, linewidth=1.5, label="Forecast Skill (Case B/C sensors)")
            ax.plot(g1.index, g1.values, color="gray", marker="s", markersize=3, linewidth=1.2,
                    label="Forecast Skill (nacelle accelerometer)")
            ax.plot(hh, np.exp(-rate * hh), "r--", linewidth=1.2, label=r"$e^{-2\zeta_1\omega_1 h}$ (Table 3)")
            ax.axvline(hreq, color="b", linestyle=":", linewidth=1.2, label=f"Horizon for $R^2$=0.9 ({hreq:.1f}s)")
            ax.set_title(f"Forecast Horizon ({NAME[d]})", fontweight="bold")
            ax.set_xlabel("Lead Time h (s)"); ax.set_xlim(0, 4.2); ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3); ax.legend(loc="lower left", fontsize=6.5)
        axes[0].set_ylabel("$R^2$ (Hidden Channels)")
        fig.tight_layout(); fig.savefig(self.out / "fig11_forecast_horizon.png", dpi=300); plt.close(fig)

    # ------------------------------------------------------------------ Fig. 12
    def fig12(self):
        summ = pd.read_csv(self.R / "blocked_summary_random.csv")
        LAB = {"gappyPOD": "Baseline: Gappy POD", "OLS": "Baseline: Static Least Squares",
               "FIR": "Baseline: Wiener Filter (per layout)",
               "DMD": "Hankel-DMD: Projection", "KF": "Hankel-DMD: Kalman Form"}
        MK = {"gappyPOD": "v", "OLS": "s", "FIR": "^", "DMD": "o", "KF": "*"}
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), sharey=True)
        for ax, d in zip(axes, ("FA", "SS")):
            for m in ("gappyPOD", "OLS", "FIR", "DMD", "KF"):
                s = summ[(summ.direction == d) & (summ.method == m)].sort_values("q")
                base = m in ("gappyPOD", "OLS", "FIR")
                ax.plot(s.q, s["median"].clip(lower=-0.05), marker=MK[m], markersize=5 if m != "KF" else 8,
                        linewidth=1.2 if base else 1.8, linestyle="--" if base else "-",
                        markerfacecolor="white" if base else None, label=LAB[m])
            ax.axvline(5, color="k", linestyle=":", linewidth=1.0)
            ax.set_title(f"{NAME[d]}", fontweight="bold")
            ax.set_xlabel("Number of Active Sensors q"); ax.set_xticks([1, 2, 3, 4, 6, 9, 14])
            ax.set_ylim(0.45, 1.02); ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Median $R^2$ (Hidden Channels)")
        axes[1].legend(loc="lower right", fontsize=7)
        fig.tight_layout(); fig.savefig(self.out / "fig12_accuracy_vs_sensors.png", dpi=300); plt.close(fig)

    # ------------------------------------------------------------------ Fig. 13
    def fig13(self):
        FS = self.FS
        EXC = np.load(self.R / "vs_excerpts.npy", allow_pickle=True).item()
        e = EXC["FA|L1 nacelle acc.|7"]
        rows = [(9, "Tower-Base Moment (Sensor 9)", 1e-3, "Moment (MN m)", 30),
                (4, "Mid-Tower Acceleration (Sensor 4)", 1.0, "Accel. (m/s$^2$)", 10),
                (17, "Nacelle Moment (Sensor 17)", 1e-3, "Moment (MN m)", 30)]
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.2))
        for ax, (s, title, sc, unit, dur) in zip(axes, rows):
            i = list(e["hid"]).index(s); n = int(dur * FS); t = np.arange(n) / FS
            phys = lambda z: (e["mu"][i] + e["sd"][i] * z) * sc  # noqa: E731
            tr = e["truth"][i]
            r2 = {k: 1 - ((tr - e[k][i]) ** 2).sum() / ((tr - tr.mean()) ** 2).sum()
                  for k in ("Static OLS", "Hankel-DMD (projection)", "Hankel-DMD (Koopman-Kalman)")}
            ax.plot(t, phys(tr[:n]), "k-", alpha=0.5, linewidth=1.5, label="Actual (Hidden)")
            ax.plot(t, phys(e["Static OLS"][i, :n]), "b:", linewidth=1.0, label=f"Static Least Squares ($R^2$={r2['Static OLS']:.2f})")
            ax.plot(t, phys(e["Hankel-DMD (projection)"][i, :n]), "r--", linewidth=1.2,
                    label=f"Hankel-DMD, Projection ($R^2$={r2['Hankel-DMD (projection)']:.2f})")
            ax.plot(t, phys(e["Hankel-DMD (Koopman-Kalman)"][i, :n]), color="tab:green", linestyle="-.", linewidth=1.1,
                    label=f"Hankel-DMD, Kalman Form ($R^2$={r2['Hankel-DMD (Koopman-Kalman)']:.2f})")
            ax.set_title(f"Nacelle Accelerometer Only: {title}", fontweight="bold")
            ax.set_ylabel(unit); ax.set_xlim(0, dur); ax.grid(True, alpha=0.3)
            lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.45 * (hi - lo)); ax.legend(loc="upper right", fontsize=6.3, ncol=2)
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout(); fig.savefig(self.out / "fig13_sparse_layout_reconstruction.png", dpi=300); plt.close(fig)

    # ------------------------------------------------------------------ Fig. A1
    def figA(self):
        raw = pd.read_csv(self.R / "sysid_sensitivity.csv"); cor = pd.read_csv(self.R / "corr_dmd_sensitivity.csv")
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
        for i, d in enumerate(("FA", "SS")):
            for j, (df, lab) in enumerate(((raw, "Response Domain"), (cor, "Correlation Domain"))):
                g = df[(df.direction == d) & (df["mode"] == "1st") & (df.d == 60)].sort_values("r")
                ax1 = axes[i, j]
                ax1.plot(g.r, g.f, color="tab:blue", marker="o", markersize=3, label="Freq")
                ax1.set_ylabel("Freq (Hz)", color="tab:blue", fontweight="bold"); ax1.tick_params(axis="y", labelcolor="tab:blue")
                ax1.set_ylim(0.48, 0.56); ax1.grid(True, alpha=0.3)
                ax2 = ax1.twinx()
                ax2.plot(g.r, 100 * g.zeta, color="tab:red", marker="s", markersize=3, linestyle="--", label="Damping")
                ax2.set_ylabel("Damping Ratio (%)", color="tab:red", fontweight="bold"); ax2.tick_params(axis="y", labelcolor="tab:red")
                ax2.set_ylim(0, 4.5)
                ax1.set_title(f"{NAME[d]} (1st Mode): {lab}, d=60", fontsize=9)
                if i == 1:
                    ax1.set_xlabel("Truncation Rank r", fontweight="bold")
        fig.tight_layout(); fig.savefig(self.out / "figA_sensitivity_domains.png", dpi=300); plt.close(fig)


# ======================================================================= FOCAL study
class FC:
    def __init__(self, cfg, out):
        self.S = FocalStudy(cfg)
        self.cfg, self.out = cfg, out
        self.RES = self.S.results
        self.DT, self.FS = self.S.dt, self.S.fs
        fc = cfg["figures"]
        self.HAMMER = fc["hammer_hz"]
        self.F1P = fc["rotor_3p_hz"] / 3.0               # mean rotor frequency (Hz): 3*Omega_bar/(2*pi) = 0.366 Hz

    def fig14(self):
        """(a) fore-aft, (b) side-to-side: measured wide-band spectrum, rotor harmonics 3P-12P at n*Omega_bar/2pi, first
        tower mode (Hankel-DMD pole) and, in an inset, the published hammer-test value, FDD, SSI-COV and Hankel-DMD."""
        c14 = self.cfg["figures"]["fig14"]
        ZOOM = tuple(c14["zoom_hz"]); F1P = self.F1P; HAMMER = self.HAMMER
        summ = pd.read_csv(self.RES / "focal_sysid_summary.csv")
        W, H, LM, RM, PH = 7.0, 5.9, 0.85, 0.15, 2.1               # figure, margins and panel height (inches)
        xlim, ylim = tuple(c14["xlim"]), tuple(c14["ylim"])
        lx = lambda x: np.log10(x / xlim[0]) / np.log10(xlim[1] / xlim[0])  # noqa: E731
        ly = lambda y: np.log10(y / ylim[0]) / np.log10(ylim[1] / ylim[0])  # noqa: E731
        fig = plt.figure(figsize=(W, H))
        for k, d in enumerate(("FA", "SS")):
            ax = fig.add_axes([LM / W, (H - 0.40 - PH if k == 0 else 0.55) / H, (W - LM - RM) / W, PH / H])
            row = summ[(summ.direction == d) & (summ["mode"] == "Tower 1st")].iloc[0]
            z = np.load(self.RES / f"wideband_{d}.npz"); f = z["f"]; A = np.sqrt(z["sv"])
            m = (f >= xlim[0]) & (f <= xlim[1])
            yv = lambda x: np.interp(x, f[m], A[m])  # noqa: E731
            ax.loglog(f[m], A[m], "k-", linewidth=0.8, zorder=3)
            ax.axvspan(*ZOOM, color="red", alpha=0.10, linewidth=0, zorder=0)
            for n in (3, 6, 9, 12):
                ax.axvline(n * F1P, ymax=0.90, color="dodgerblue", linestyle=":", linewidth=1.2, zorder=2)
                ax.text(n * F1P, 0.915, f"{n}P", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="royalblue")
            ax.plot([row.DMD_f], [yv(row.DMD_f)], "r*", markersize=13, markeredgecolor="k", markeredgewidth=0.5, zorder=6)
            ax.annotate("1st tower\nmode", (row.DMD_f, yv(row.DMD_f)), xytext=(0.47, 0.85), textcoords="data",
                        ha="left", va="center", fontsize=8, fontweight="bold", color="darkred",
                        arrowprops=dict(arrowstyle="-", color="darkred", linewidth=0.8, shrinkA=2, shrinkB=7))
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            ax.set_xticks([0.3, 0.5, 1, 2, 5, 10])
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_title(f"({'ab'[k]}) {NAME[d]}", fontweight="bold")
            ax.set_xlabel("Frequency (Hz, full scale)"); ax.set_ylabel("Normalised Spectral Amplitude")
            ax.grid(True, which="major", linestyle=":", alpha=0.6)
            x0, x1, y0, y1 = lx(1.75), lx(6.6), ly(0.20), ly(1.6)
            ins = ax.inset_axes([x0, y0, x1 - x0, y1 - y0])
            xl = ZOOM[0] - (ZOOM[1] - ZOOM[0]) * 0.58 / 0.42                            # left 58% of the inset: labels
            ins.set_xlim(xl, ZOOM[1]); ins.set_ylim(-0.55, 3.55)
            ins.axhspan(2.5, 3.55, color="0.93", linewidth=0); ins.axhspan(0.5, 1.5, color="0.93", linewidth=0)
            ins.axvline(HAMMER[d], color="0.45", linestyle="--", linewidth=0.9, zorder=2)
            ins.plot([HAMMER[d]], [3], "D", color="0.55", markeredgecolor="k", markeredgewidth=0.5, markersize=5, zorder=4)
            ins.plot([row.FDD_f], [2], "v", color="k", markerfacecolor="white", markersize=6.5, zorder=4)
            ins.plot([row.SSI_f], [1], "s", color="k", markerfacecolor="white", markersize=5.5, zorder=4)
            ins.plot([row.DMD_f], [0], "r*", markersize=11, markeredgecolor="k", markeredgewidth=0.5, zorder=4)
            for lab, yy in (("Hammer test (published)", 3), ("FDD", 2), ("SSI-COV", 1), ("Hankel-DMD", 0)):
                ins.text(xl + 0.0015, yy, lab, ha="left", va="center", fontsize=7.5)
            ins.set_xticks([0.40, 0.42, 0.44]); ins.set_xticklabels(["0.40", "0.42", "0.44"]); ins.set_yticks([])
            ins.tick_params(axis="x", labelsize=7.5, length=2.5, pad=2)
            ins.grid(True, axis="x", linestyle=":", alpha=0.6)
            ins.set_title("1st tower mode: frequency (Hz)", fontsize=8, fontweight="bold", color="darkred", pad=3)
            for s in ins.spines.values():
                s.set_edgecolor("darkred"); s.set_linewidth(0.9)
        fig.savefig(self.out / "fig14_focal_identification.png", dpi=300); plt.close(fig)

    def fig15(self):
        """Spectrograms (serif, viridis, dB, as Fig. 8) of the measured (hidden) tower-base moment and its
        reconstructions from the nacelle accelerometer alone (projection, Kalman form), run E11, second half;
        tracks of 3P, 6P and 9P from the measured rotor speed (dotted) and the first tower mode (dashed)."""
        c15 = self.cfg["figures"]["fig15"]
        t0, t_span, win, fmax = c15["t0"], c15["span_s"], c15["window_s"], c15["fmax"]
        DT, FS = self.DT, self.FS
        plt.rcParams.update({"font.family": "serif"})
        summ = pd.read_csv(self.RES / "focal_sysid_summary.csv")
        cols = [("truth", "Measured (Hidden)"), ("N|DMD", "Virtual: Projection"), ("N|KF", "Virtual: Kalman Form")]
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.4))
        fig.subplots_adjust(left=0.08, right=0.89, top=0.89, bottom=0.08, hspace=0.62, wspace=0.07)
        for row, d in enumerate(("FA", "SS")):
            z = np.load(self.RES / "intermediate" / f"vs_series_{d}.npz")
            i0_, n = int(t0 / DT), int(t_span / DT); nper = int(win / DT)
            om = z["omega"][i0_:i0_ + n]; omw = np.convolve(om, np.ones(nper) / nper, mode="same")
            tt = t0 + np.arange(n) * DT
            ft = summ[(summ.direction == d) & (summ["mode"] == "Tower 1st")].DMD_f.iloc[0]
            vmax = None
            for col, (key, lab) in enumerate(cols):
                x = z["truth"][3] if key == "truth" else z[key][3]
                x = (z["mu"][3] + z["sd"][3] * x[i0_:i0_ + n]) * 1e-6          # MN m
                fq, ts, S = spectrogram(x, fs=FS, nperseg=nper, noverlap=int(0.8 * nper))
                m = fq <= fmax; Pdb = 10 * np.log10(S[m] + 1e-12)
                if vmax is None:
                    vmax = Pdb.max()
                ax = axes[row, col]
                im = ax.pcolormesh(ts + t0, fq[m], Pdb, shading="gouraud", cmap="viridis", vmin=vmax - 40, vmax=vmax)
                sl = slice(nper // 2, -nper // 2, 20)
                for k, nh in enumerate((3, 6, 9)):
                    ax.plot(tt[sl], nh * omw[sl] / (2 * np.pi), color="w", linestyle=":", linewidth=0.7, alpha=0.75,
                            label=r"Rotor Harmonics $n\Omega(t)/2\pi$" if k == 0 else None)
                ax.axhline(ft, color="r", linestyle="--", linewidth=0.8, label=f"1st Tower Mode ({ft:.3f} Hz)")
                ax.set_ylim(0.25, fq[m][-1]); ax.set_xlim(ts[0] + t0, ts[-1] + t0)
                ax.set_title(lab, fontsize=7.5)
                ax.set_xlabel("Time [s]", fontsize=7); ax.tick_params(labelsize=7)
                if col == 0:
                    ax.set_ylabel("Frequency [Hz]", fontsize=7)
                else:
                    ax.set_yticklabels([])
                if col == 2:
                    cax = fig.add_axes([0.905, ax.get_position().y0, 0.015, ax.get_position().height])
                    fig.colorbar(im, cax=cax, label="Power (dB)")
                if row == 0 and col == 0:
                    ax.legend(loc="upper right", fontsize=5.5, framealpha=0.85)
            axes[row, 0].annotate(f"({'ab'[row]}) {NAME[d]}: Tower-Base Bending Moment, Hankel-DMD from the Nacelle "
                                  f"Accelerometer Only", xy=(0, 1.2), xycoords="axes fraction", fontsize=8.5, fontweight="bold")
        fig.savefig(self.out / "fig15_focal_time_frequency.png", dpi=300); plt.close(fig)
        plt.rcParams.update({"font.family": "sans-serif"})

    def fig16(self):
        """Unseen-data reconstruction of the tower-base bending moment (style of Fig. 13): measured, static LS
        (baseline), Hankel-DMD projection and Kalman form; run E11, second half."""
        c16 = self.cfg["figures"]["fig16"]
        panels, t_start, dur = [tuple(p) for p in c16["panels"]], c16["t_start"], c16["duration_s"]
        DT = self.DT
        LAYNAME = {"N": "Nacelle Accelerometer Only", "N+Bacc": "Nacelle and Platform Accelerometers"}
        S0 = int(round(60 / DT))
        fig, axes = plt.subplots(len(panels), 1, figsize=(7.0, 2.07 * len(panels)))
        for ax, (d, lay) in zip(np.atleast_1d(axes), panels):
            z = np.load(self.RES / "intermediate" / f"vs_series_{d}.npz")
            tr = z["truth"][3]
            phys = lambda e: (z["mu"][3] + z["sd"][3] * e) * 1e-6  # noqa: E731
            r2 = lambda e: 1 - ((tr[S0:] - e[S0:]) ** 2).sum() / ((tr[S0:] - tr[S0:].mean()) ** 2).sum()  # noqa: E731
            n0, n = int(t_start / DT), int(dur / DT); t = np.arange(n) * DT + t_start
            e_ls, e_p, e_k = z[f"{lay}|OLS"][3], z[f"{lay}|DMD"][3], z[f"{lay}|KF"][3]
            ax.plot(t, phys(tr[n0:n0 + n]), "k-", alpha=0.5, linewidth=1.5, label="Measured (Hidden)")
            ax.plot(t, phys(e_ls[n0:n0 + n]), "b:", linewidth=1.0, label=f"Static Least Squares ($R^2$={r2(e_ls):.2f})")
            ax.plot(t, phys(e_p[n0:n0 + n]), "r--", linewidth=1.2, label=f"Hankel-DMD, Projection ($R^2$={r2(e_p):.2f})")
            ax.plot(t, phys(e_k[n0:n0 + n]), color="tab:green", linestyle="-.", linewidth=1.1,
                    label=f"Hankel-DMD, Kalman Form ($R^2$={r2(e_k):.2f})")
            ax.set_title(f"{LAYNAME[lay]}: {NAME[d]} Tower-Base Moment", fontweight="bold")
            ax.set_ylabel("Moment (MN m)"); ax.set_xlim(t[0], t[-1]); ax.grid(True, alpha=0.3)
            lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 0.45 * (hi - lo))
            ax.legend(loc="upper right", fontsize=6.3, ncol=2)
            print(d, lay, f"R2 LS {r2(e_ls):.3f} proj {r2(e_p):.3f} KF {r2(e_k):.3f}", flush=True)
        np.atleast_1d(axes)[-1].set_xlabel("Time in Test Record (s, full scale; run E11, second half, unseen)")
        fig.tight_layout(); fig.savefig(self.out / "fig16_focal_virtual_sensing.png", dpi=300); plt.close(fig)


OPENFAST = ["fig5", "fig6_7", "fig8", "fig9", "fig10", "fig11", "fig12", "fig13", "figA"]
FOCAL = ["fig14", "fig15", "fig16"]


def main(names, of_cfg, fc_cfg, out_dir):
    out = ensure_dir(out_dir)
    want = set(names)
    if "all" in want:
        want |= set(OPENFAST + FOCAL)
    if "openfast" in want:
        want |= set(OPENFAST)
    if "focal" in want:
        want |= set(FOCAL)
    of = OF(of_cfg, out) if want & set(OPENFAST) else None
    fc = FC(fc_cfg, out) if want & set(FOCAL) else None
    for name in OPENFAST + FOCAL:
        if name in want:
            obj = of if name in OPENFAST else fc
            try:
                getattr(obj, name)()
                print("written:", name, flush=True)
            except FileNotFoundError as e:
                print(f"skipped {name}: missing input ({e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", default=["all"])
    ap.add_argument("--openfast", default="configs/openfast.yaml")
    ap.add_argument("--focal", default="configs/focal.yaml")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    main(a.names, load_config(a.openfast), load_config(a.focal), a.out)
