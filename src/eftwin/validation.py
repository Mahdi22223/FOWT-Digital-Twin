"""Validation protocol: time-blocked folds, sufficient statistics, cross-fitted hyper-parameters, metrics.

Time-blocked leave-one-out (Section 4.4): every record is split into two halves; a fold (test record c,
half h) is trained on half 1-h of all other records, so that neither the test record nor its time interval is
seen in training. Per hidden channel only sufficient statistics are stored, so that any aggregate (per half,
per record over both halves, per sensor) can be formed exactly afterwards. Hyper-parameters are cross-fitted:
folds of one group of records use the setting selected on the other group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def blocked_folds(records, halves=(0, 1)):
    """Folds ``(test, h, train)`` with ``train = [(c, 1 - h) for c != test]``."""
    for test in records:
        for h in halves:
            yield test, h, [(c, 1 - h) for c in records if c != test]


def sufficient_stats(T: np.ndarray, E: np.ndarray) -> dict:
    """Per-channel statistics of truth T and estimate E (channels x samples)."""
    return dict(n=T.shape[1], sy=T.sum(1), syy=(T ** 2).sum(1), sse=((T - E) ** 2).sum(1),
                ymax=T.max(1), ymin=T.min(1))


def r2_channels(T: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Coefficient of determination of every channel (rows)."""
    return 1 - ((T - E) ** 2).sum(1) / ((T - T.mean(1, keepdims=True)) ** 2).sum(1)


def records_frame(records: list, group_of) -> pd.DataFrame:
    """Records (list of dicts) -> DataFrame with hyper-parameter key ``hp`` ("r|alpha|rho"), cross-fitting
    ``group`` (``group_of`` applied to the test-record column) and mean R^2 of the fold ``R2_fold``."""
    df = pd.DataFrame(records)
    for c in ("r", "alpha", "rho"):
        if c not in df:
            df[c] = np.nan
    df["hp"] = [f"{r}|{a}|{rho}" for r, a, rho in zip(df.r, df.alpha, df.rho)]
    df["group"] = group_of(df.test).astype(int)
    df["R2_fold"] = [np.mean(1 - s / (yy - y ** 2 / n)) for s, yy, y, n in zip(df.sse, df.syy, df.sy, df.n)]
    return df


def crossfit(df: pd.DataFrame, keys: list):
    """Cross-fitted selection: within every ``keys`` group, folds of group g use the ``hp`` with the largest
    mean ``R2_fold`` over the folds of the other group. Returns ``(selected rows, hyper-parameter table)``."""
    parts, hyper = [], []
    for vals, g in df.groupby(keys):
        for grp in (0, 1):
            best = g[g.group != grp].groupby("hp").R2_fold.mean().idxmax()
            parts.append(g[(g.group == grp) & (g.hp == best)])
            hyper.append(dict(zip(keys, vals), group=grp, hp=best))
    return pd.concat(parts), pd.DataFrame(hyper)


def combine(g: pd.DataFrame):
    """Per-channel R^2 and NRMSE (% of range) over all rows of ``g`` (e.g. both time halves of a record)."""
    n = g.n.sum(); sy = np.sum(np.vstack(g.sy), 0); syy = np.sum(np.vstack(g.syy), 0)
    sse = np.sum(np.vstack(g.sse), 0)
    ymax = np.max(np.vstack(g.ymax), 0); ymin = np.min(np.vstack(g.ymin), 0)
    r2 = 1 - sse / (syy - sy ** 2 / n)
    nrmse = 100 * np.sqrt(sse / n) / (ymax - ymin)
    return r2, nrmse


def coherence_bound(S: np.ndarray, active, hidden: int) -> float:
    """Fraction of the variance of channel ``hidden`` that is linearly coherent with the ``active`` channels,
    int gamma^2_{h|a} S_hh df / int S_hh df, from CSD matrices S[f, p, p] over the band of interest. No linear
    estimator, causal or not, can exceed it."""
    num = 0.0; den = 0.0
    for k in range(S.shape[0]):
        Saa = S[k][np.ix_(active, active)]; Sah = S[k][np.ix_(active, [hidden])]; Shh = S[k][hidden, hidden].real
        num += float(np.real((Sah.conj().T @ np.linalg.solve(Saa, Sah))[0, 0]))
        den += Shh
    return num / den
