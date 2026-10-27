"""Wide-band analyses of the FOCAL tower channels (Section 5.2, Fig. 14; second tower modes).

  scan        0.2-11 Hz spectral scan: first singular value of the CSD matrix (all blocks and lowest/highest
              rotor-speed tercile); peaks that keep their frequency between terciles are structural candidates
              -> results/focal/wideband_<DIR>.npz (Fig. 14)
  poles       0.25-10 Hz: correlation-domain Hankel-DMD (pooled runs, d = 86, r = 10..80) and SSI-COV
              (i = 86, orders 20..120, stability flags) -> results/focal/wide_poles_<DIR>.npz
  candidates  candidate modes above 1 Hz per run and per rotor-speed tercile (frequency shift, scatter, MAC)
  first-mode  all poles in 0.33-0.50 Hz per rotor-speed tercile (frequency, damping, modal energy, recurrence)
Usage: python scripts/focal/wideband.py {scan,poles,candidates,first-mode,all} [--config configs/focal.yaml]
"""
import argparse

import numpy as np
from scipy.signal import find_peaks

from eftwin.baselines.oma import csd_sum, first_singular_values, ssi_poles, ssi_stability_diagram
from eftwin.config import as_range, load_config
from eftwin.dmd import correlation, dmd_from_gram, dmd_poles, modal_energy, modal_params
from eftwin.hankel import gram_over_segments
from eftwin.modal import mac
from eftwin.preprocess import bandpass_sos
from eftwin.studies.focal import P, FocalStudy


def chans(S_, run, d, lo, hi):
    """Mean-removed tower channels band-passed with second-order sections (zero phase)."""
    return bandpass_sos(S_.data.segment(run, d), lo, hi, S_.fs)


def dmd_all(G, C, r, dt, f_range):
    """All poles of the rank-r model in f_range: f, zeta, last-block shapes, modal energy (relative)."""
    mu, Phi = dmd_from_gram(G, C, r)
    E = modal_energy(Phi, G)
    f, z, _ = modal_params(mu, dt)
    keep = (mu.imag > 0) & (f > f_range[0]) & (f < f_range[1])
    return f[keep], z[keep], Phi[-P:, keep], E[keep] / E[keep].max()


def blocks_by_rotor_speed(S_, segs, nb, transform=None):
    blocks, om, runid = [], [], []
    for r in S_.runs:
        w = S_.aux(r, "genSpeed"); s = segs[r]
        for b in range(0, len(s) - nb + 1, nb):
            blk = s[b:b + nb]
            blocks.append(blk if transform is None else transform(blk)); om.append(w[b:b + nb].mean())
            runid.append(S_.run_label[r])
    return blocks, np.array(om), np.array(runid)


def scan(S_, cfg):
    wc = cfg["wideband"]
    nb = int(wc["block_s"] / S_.dt)
    lo_b, hi_b = wc["scan_band"]
    for d in ("FA", "SS"):
        X = {r: chans(S_, r, d, lo_b, hi_b) for r in S_.runs}
        sd = np.std(np.vstack(list(X.values())), axis=0)
        blocks, om, _ = blocks_by_rotor_speed(S_, {r: X[r] / sd for r in S_.runs}, nb)
        lo, hi = om <= np.percentile(om, 100 / 3), om >= np.percentile(om, 200 / 3)
        out = {}
        for name, m in (("all", np.ones(len(om), bool)), ("low", lo), ("high", hi)):
            f, S = csd_sum([blocks[k] for k in np.where(m)[0]], S_.fs, int(100 * S_.fs))
            sv = first_singular_values(S)
            out[name] = (f, sv / sv[(f > 0.25)].max())
        f, sv = out["all"]
        band = (f > 0.25) & (f < 11)
        pk, prop = find_peaks(np.log10(sv[band]), prominence=0.15)
        f1p_lo, f1p_hi = om[lo].mean() / (2 * np.pi), om[hi].mean() / (2 * np.pi)
        print(f"== {d}: 1P low/high = {f1p_lo:.4f}/{f1p_hi:.4f} Hz")
        for k in pk:
            fk = f[band][k]
            locs = []
            for name in ("low", "high"):
                ff, ss = out[name]; mm = (ff > fk * 0.94) & (ff < fk * 1.06)
                locs.append(ff[mm][np.argmax(ss[mm])])
            shift = 100 * (locs[1] / locs[0] - 1)
            print(f"  peak {fk:6.3f} Hz  level {sv[band][k]:.2e}  prom {prop['prominences'][list(pk).index(k)]:.2f}  "
                  f"low/high {locs[0]:.3f}/{locs[1]:.3f} (shift {shift:+.1f}%)  ~{fk/((f1p_lo+f1p_hi)/2):.2f}P")
        np.savez(S_.results / f"wideband_{d}.npz", f=f, sv=sv, f_lo=out["low"][0], sv_lo=out["low"][1],
                 sv_hi=out["high"][1], f1p=(f1p_lo, f1p_hi))


def poles(S_, cfg):
    wc = cfg["wideband"]
    LO, HI = wc["id_band"]
    D = wc["id_delay"]
    K = int(cfg["identification"]["correlation_s"] / S_.dt)
    for d in ("FA", "SS"):
        segs = {r: chans(S_, r, d, LO, HI) for r in S_.runs}
        sd = np.std(np.vstack(list(segs.values())), axis=0); segs = {k: v / sd for k, v in segs.items()}
        Rs = {r: correlation(s, max(K, 2 * D + 1)) for r, s in segs.items()}
        Rm = np.mean(list(Rs.values()), axis=0)
        G, C = gram_over_segments([Rm[1:K + 1, :, j].T for j in range(P)], D)
        dmd_rows = []
        for r in as_range(wc["id_ranks"]):
            f, z, Phi, E = dmd_all(G, C, r, S_.dt, (LO, HI))
            for k in range(len(f)):
                dmd_rows.append((r, f[k], z[k], E[k]))
        ssi_rows = ssi_stability_diagram(ssi_poles(Rm, P, D, as_range(wc["id_ssi_orders"]), S_.dt), (LO, HI))
        np.savez(S_.results / f"wide_poles_{d}.npz", dmd=np.array(dmd_rows),
                 ssi=np.array([(o, f, z, s) for o, f, z, s, _ in ssi_rows]), sd=sd)
        print(f"== {d}: DMD poles {len(dmd_rows)}, SSI poles {len(ssi_rows)} ({sum(r[3] for r in ssi_rows)} stable)", flush=True)
        st = np.array([(f, z) for o, f, z, s, _ in ssi_rows if s and f > 1.0])
        if len(st):
            st = st[np.argsort(st[:, 0])]
            clusters, cur = [], [st[0]]
            for p in st[1:]:
                if p[0] / cur[-1][0] - 1 < 0.02:
                    cur.append(p)
                else:
                    clusters.append(np.array(cur)); cur = [p]
            clusters.append(np.array(cur))
            for c in clusters:
                if len(c) >= 6:
                    print(f"   SSI stable cluster: f = {np.median(c[:,0]):.3f} Hz, zeta = {100*np.median(c[:,1]):.1f} %, n = {len(c)}")


def candidates(S_, cfg):
    wc = cfg["wideband"]
    LO, HI = wc["id_band"]
    D = wc["id_delay"]
    nb, K = int(wc["block_s"] / S_.dt), int(cfg["identification"]["correlation_s"] / S_.dt)
    for d in ("FA", "SS"):
        segs = {r: chans(S_, r, d, LO, HI) for r in S_.runs}
        sd = np.std(np.vstack(list(segs.values())), axis=0); segs = {k: v / sd for k, v in segs.items()}
        blocks, om, runid = blocks_by_rotor_speed(S_, segs, nb, lambda b: correlation(b, K))
        blocks = np.array(blocks)
        groups = {"low": om <= np.percentile(om, 100 / 3), "high": om >= np.percentile(om, 200 / 3)}
        for lab in ("E11", "E13", "E14", "E15"):
            groups[lab] = runid == lab
        models = {}
        for g, m in groups.items():
            Rm = blocks[m].mean(0)
            models[g] = gram_over_segments([Rm[1:, :, j].T for j in range(P)], D)
        print(f"== {d}")
        for lo, hi, name in wc["candidates"][d]:
            res = {}
            for g, (G, C) in models.items():
                fs_, zs_, shp = [], [], []
                for r in range(20, 82, 4):
                    f, z, Phi, E = dmd_all(G, C, r, S_.dt, (LO, HI))
                    m = (f > lo) & (f < hi) & (z > 0) & (z < 0.3)
                    if m.any():
                        j = np.where(m)[0][np.argmax(E[m])]
                        fs_.append(f[j]); zs_.append(z[j]); shp.append(Phi[:, j] * sd)
                if fs_:
                    k = np.argsort(fs_)[len(fs_) // 2]
                    res[g] = (np.median(fs_), 100 * np.median(zs_), shp[k], len(fs_))
            if "low" in res and "high" in res:
                shift = 100 * (res["high"][0] / res["low"][0] - 1)
                runs = [res[r][0] for r in ("E11", "E13", "E14", "E15") if r in res]
                macs = [mac(res["low"][2], res[r][2]) for r in ("high", "E11", "E13", "E14", "E15") if r in res]
                print(f"  {name:12s}: f low/high = {res['low'][0]:.3f}/{res['high'][0]:.3f} Hz (shift {shift:+.2f}%, rotor +4.3%), "
                      f"runs {min(runs):.3f}-{max(runs):.3f}, zeta {res['low'][1]:.1f}/{res['high'][1]:.1f} %, MAC min {min(macs):.2f}, "
                      f"n={res['low'][3]}/{res['high'][3]}")
                v = np.real(res["low"][2] * np.exp(-1j * np.angle(res["low"][2][np.argmax(np.abs(res['low'][2]))])))
                print(f"      shape (a_N, a_P, M_T, M_B; de-standardised, normalised): {np.round(v / np.max(np.abs(v)), 2)}")
            else:
                print(f"  {name:12s}: not found in both terciles ({list(res.keys())})")


def first_mode(S_, cfg):
    D = cfg["wideband"]["id_delay"]
    nb, K = int(cfg["wideband"]["block_s"] / S_.dt), int(cfg["identification"]["correlation_s"] / S_.dt)
    for d in ("FA", "SS"):
        segs = {r: S_.channels(r, d) for r in S_.runs}
        sd = np.std(np.vstack(list(segs.values())), axis=0)
        blocks, om, _ = blocks_by_rotor_speed(S_, {r: segs[r] / sd for r in S_.runs}, nb, lambda b: correlation(b, K))
        blocks = np.array(blocks)
        for g, m in (("all", np.ones(len(om), bool)), ("low", om <= np.percentile(om, 100/3)), ("high", om >= np.percentile(om, 200/3))):
            Rm = blocks[m].mean(0)
            G, C = gram_over_segments([Rm[1:, :, j].T for j in range(P)], D)
            pts = []
            for r in range(16, 62, 2):
                f, z, Phi, E, mu = dmd_poles(G, C, r, S_.dt)
                En = E / E[(mu.imag > 0)].max()
                mm = (mu.imag > 0) & (f > 0.33) & (f < 0.50) & (z > 0)
                for j in np.where(mm)[0]:
                    pts.append((f[j], z[j], En[j]))
            pts = np.array(pts); pts = pts[np.argsort(pts[:, 0])]
            cl, cur = [], [pts[0]]
            for p in pts[1:]:
                if p[0] / cur[-1][0] - 1 < 0.015:
                    cur.append(p)
                else:
                    cl.append(np.array(cur)); cur = [p]
            cl.append(np.array(cur))
            desc = [f"{np.median(c[:,0]):.3f}Hz/z{100*np.median(c[:,1]):.1f}%/E{np.median(c[:,2]):.3f}/n{len(c)}" for c in cl if len(c) >= 8]
            print(f"{d} {g:4s} (1P={om[m].mean()/2/np.pi:.4f}, 3P={3*om[m].mean()/2/np.pi:.3f}): " + "  ".join(desc))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("scan", "poles", "candidates", "first-mode", "all"))
    ap.add_argument("--config", default="configs/focal.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    S_ = FocalStudy(cfg)
    for name, fn in (("scan", scan), ("poles", poles), ("candidates", candidates), ("first-mode", first_mode)):
        if a.what in (name, "all"):
            fn(S_, cfg)
