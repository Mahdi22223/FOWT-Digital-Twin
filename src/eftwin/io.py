"""Data access: OpenFAST case files (Sections 3-4) and FOCAL Campaign IV basin-test runs (Section 5).

OpenFAST: ``Case_<n>.csv`` files exported from the ``.outb`` outputs at the native output interval
(``DT_Out = 0.05 s``), with a ``Time_[s]`` column and the tower-gauge channels ``TwHt<i>AL{x,y}t`` and
``TwHt<i>ML{x,y}t``. Fore-aft (FA) = accelerations ``ALxt`` with moments ``MLyt``; side-to-side (SS) =
``ALyt`` with ``MLxt``. Within a direction the 18 channels are ordered as in the paper: sensors 0-8 are
accelerations and 9-17 bending moments, from the tower base to the top.

FOCAL: the four Campaign IV runs (same wind input W02, waves E11, E13, E14, E15), converted to full scale
(Froude 1:70); the CSV files have a units row below the header. Model-scale sampling at 200 Hz gives
dt = 0.005*sqrt(70) s at full scale.

Arrays read from CSV are cached as ``.npz`` files (cache directories are not version-controlled).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from .preprocess import bandpass

# ------------------------------------------------------------------ OpenFAST channels
FA_ACC = [f"TwHt{i}ALxt_[m/s^2]" for i in range(1, 10)]
FA_MOM = [f"TwHt{i}MLyt_[kN-m]" for i in range(1, 10)]
SS_ACC = [f"TwHt{i}ALyt_[m/s^2]" for i in range(1, 10)]
SS_MOM = [f"TwHt{i}MLxt_[kN-m]" for i in range(1, 10)]
ALL_COLS = FA_ACC + FA_MOM + SS_ACC + SS_MOM  # 36 channels; FA = columns 0-17, SS = 18-35
DIRECTIONS = ("FA", "SS")
DIRECTION_NAME = {"FA": "Fore-Aft", "SS": "Side-to-Side"}

# ElastoDyn tower gauges (TwrGagNd) of the simulation model, 20 tower nodes between 10 m and 87.6 m
GAUGE_NODES = (1, 3, 5, 7, 10, 13, 15, 17, 20)


def gauge_heights(nodes=GAUGE_NODES, base: float = 10.0, top: float = 87.6, n_nodes: int = 20) -> np.ndarray:
    """Heights of ElastoDyn tower nodes: base + (node - 1/2) (top - base) / n_nodes (11.9-85.7 m)."""
    return base + (np.asarray(nodes) - 0.5) * (top - base) / n_nodes


def direction_slice(direction: str) -> slice:
    """Columns of one direction in the 36-channel array."""
    return slice(0, 18) if direction == "FA" else slice(18, 36)


class OpenFASTData:
    """The twelve OpenFAST operational load cases (OLCs).

    Parameters
    ----------
    csv_dir : folder with ``Case_1.csv`` ... ``Case_12.csv``
    cache_dir : folder for ``case_<n>.npz`` caches
    dt : output interval of the files (0.05 s); checked against the ``Time_[s]`` column
    trim : samples discarded at the start of every record (6000 = 300 s)
    band, order : zero-phase Butterworth band-pass (``filtfilt``) applied after trimming
    """

    def __init__(self, csv_dir, cache_dir, dt: float = 0.05, trim: int = 6000, band=(0.25, 5.0), order: int = 4,
                 dt_tolerance: float = 1e-4):
        self.csv_dir, self.cache_dir = Path(csv_dir), Path(cache_dir)
        self.dt, self.trim, self.band, self.order, self.dt_tolerance = dt, trim, tuple(band), order, dt_tolerance

    @property
    def fs(self) -> float:
        return 1.0 / self.dt

    def load(self, case_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(time, data[N, 36])`` of one case, columns ordered as ``ALL_COLS``.

        Duplicated column names (OpenFAST writes ``TwHt1MLxt``/``TwHt1MLyt`` twice) resolve to the first
        occurrence, as pandas keeps it under the exact name.
        """
        cache = self.cache_dir / f"case_{case_id}.npz"
        if cache.exists():
            z = np.load(cache)
            return z["t"], z["x"]
        df = pd.read_csv(self.csv_dir / f"Case_{case_id}.csv", engine="c")
        t = df["Time_[s]"].to_numpy(dtype=np.float64)
        x = df[ALL_COLS].to_numpy(dtype=np.float64)
        dt_file = float(np.median(np.diff(t)))
        if abs(dt_file - self.dt) > self.dt_tolerance:
            raise ValueError(f"Case_{case_id}.csv: time step {dt_file:.6f} s, configured dt = {self.dt} s. The analysis "
                             "uses the OpenFAST output interval (DT_Out = 0.05 s).")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache, t=t, x=x)
        return t, x

    def filtered(self, case_id: int, direction: str) -> np.ndarray:
        """Trimmed, band-passed record ``(N, 18)`` of one direction (filter applied after trimming)."""
        return bandpass(self.load(case_id)[1][self.trim:, direction_slice(direction)], self.band[0], self.band[1],
                        fs_design=self.fs, order=self.order)


# ------------------------------------------------------------------ FOCAL Campaign IV
FOCAL_DT = 0.005 * np.sqrt(70.0)
FOCAL_FS = 1.0 / FOCAL_DT
FOCAL_RUNS = ["FOCAL_DLC12_Seed1_3", "FOCAL_DLC12_Seed3_1", "FOCAL_DLC12_Seed4_1", "FOCAL_DLC12_Seed5_1"]
FOCAL_RUN_LABEL = {"FOCAL_DLC12_Seed1_3": "E11", "FOCAL_DLC12_Seed3_1": "E13",
                   "FOCAL_DLC12_Seed4_1": "E14", "FOCAL_DLC12_Seed5_1": "E15"}
FOCAL_CH = {"FA": ["accelNacelleAx", "towerBotAccX", "towerTopMy", "towerBotMy"],
            "SS": ["accelNacelleAy", "towerBotAccY", "towerTopMx", "towerBotMx"]}
FOCAL_CH_LABEL = {0: "Nacelle acc.", 1: "Tower-base acc.", 2: "Tower-top moment", 3: "Tower-base moment"}


class FocalData:
    """FOCAL Campaign IV runs at full scale.

    Parameters
    ----------
    csv_dir : folder with the four ``FOCAL_DLC12_*.csv`` files
    cache_dir : folder for ``<run>.npz`` caches
    t0, t1 : stationary window (full-scale seconds), identical for all runs (600-9000 s)
    band : default analysis band (0.25-5 Hz, as in the OpenFAST study)
    """

    def __init__(self, csv_dir, cache_dir, t0: float = 600.0, t1: float = 9000.0, band=(0.25, 5.0)):
        self.csv_dir, self.cache_dir = Path(csv_dir), Path(cache_dir)
        self.t0, self.t1, self.band = t0, t1, tuple(band)
        self.dt, self.fs = FOCAL_DT, FOCAL_FS

    def load(self, run: str) -> tuple[list, np.ndarray]:
        """Column names and full-scale data of one run (the CSV's second row holds the units)."""
        f = self.cache_dir / f"{run}.npz"
        if not f.exists():
            df = pd.read_csv(self.csv_dir / f"{run}.csv", skiprows=[1])
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(f, cols=np.array(df.columns, dtype=str), x=df.values.astype(float), t_raw=df["Time"].values)
        z = np.load(f)
        return list(z["cols"]), z["x"]

    def window(self, t0=None, t1=None) -> slice:
        """Sample range of the window ``[t0, t1)`` in full-scale seconds (default: the stationary window)."""
        return slice(int((self.t0 if t0 is None else t0) / self.dt), int((self.t1 if t1 is None else t1) / self.dt))

    def segment(self, run: str, direction: str, t0=None, t1=None) -> np.ndarray:
        """Stationary window of the four tower channels of one direction, mean removed (no filter)."""
        cols, x = self.load(run)
        seg = x[self.window(t0, t1), :][:, [cols.index(c) for c in FOCAL_CH[direction]]]
        return seg - seg.mean(0)

    def channels(self, run: str, direction: str, band="default", t0=None, t1=None) -> np.ndarray:
        """Mean-removed, band-passed tower channels (4th-order Butterworth, ``filtfilt``); ``band=None``: unfiltered."""
        seg = self.segment(run, direction, t0, t1)
        band = self.band if band == "default" else band
        if band is None:
            return seg
        b, a = butter(4, [band[0] / (0.5 * self.fs), band[1] / (0.5 * self.fs)], btype="band")
        return filtfilt(b, a, seg, axis=0)

    def aux(self, run: str, name: str, t0=None, t1=None) -> np.ndarray:
        """Any other channel (e.g. ``genSpeed`` in rad/s, ``Pitch`` and ``Roll`` in deg) over the same window."""
        cols, x = self.load(run)
        return x[self.window(t0, t1), cols.index(name)]
