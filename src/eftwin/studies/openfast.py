"""OpenFAST study (Sections 3-4): configuration, data and cached delay Gram sums shared by the scripts."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..config import as_range, ensure_dir
from ..hankel import delay_gram
from ..io import OpenFASTData
from ..preprocess import standardise_segments

P = 18  # channels per direction (9 accelerations + 9 bending moments)
SENSOR_NAME = {0: "Base acc.", 1: "Acc. 2", 2: "Lower-tower acc.", 3: "Lower-tower acc.", 4: "Acc. 5",
               5: "Mid-tower acc.", 6: "Acc. 7", 7: "Sub-nacelle acc.", 8: "Nacelle acc.", 9: "Base moment",
               10: "Moment 11", 11: "Lower-mid moment", 12: "Mid-tower moment", 13: "Moment 14",
               14: "Upper-tower moment", 15: "Moment 16", 16: "Sub-nacelle moment", 17: "Nacelle moment"}


class OpenFASTStudy:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        d, p = cfg["data"], cfg["paths"]
        self.data = OpenFASTData(p["csv_dir"], p["cache_dir"], dt=d["dt"], trim=d["trim"], band=d["band"],
                                 order=d["filter_order"])
        self.cases = {int(k): tuple(v) for k, v in d["cases"].items()}
        self.cache = Path(p["cache_dir"])
        self.results = ensure_dir(p["results_dir"])
        self.dt, self.fs = d["dt"], 1.0 / d["dt"]

    # ---------------------------------------------------------------- data
    def raw(self, direction: str) -> dict:
        """Trimmed, band-passed records of all OLCs, {olc: (N, 18)}."""
        return {c: self.data.filtered(c, direction) for c in self.cases}

    def scaled_segments(self, direction: str):
        """Records z-scored with the statistics of all OLCs (identification), and the scaler."""
        return standardise_segments(self.raw(direction))

    # ---------------------------------------------------------------- cached Gram sums
    def _gram(self, path: Path, Y: np.ndarray, d: int):
        if path.exists():
            z = np.load(path)
            return (z["G"], z["C"], z["s1"], z["s2"], int(z["n"]))
        g = delay_gram(Y, d)
        ensure_dir(path.parent)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, G=g[0], C=g[1], s1=g[2], s2=g[3], n=g[4])
        os.replace(tmp, path)
        return g

    def half_gram(self, direction: str, c: int, h: int, raw: dict):
        """Delay Gram sums of time half h of OLC c (virtual-sensing folds)."""
        half, d = self.cfg["virtual_sensing"]["half"], self.cfg["virtual_sensing"]["delay"]
        return self._gram(self.cache / f"gram_{direction}_{c}_h{h}.npz", raw[c][h * half:(h + 1) * half], d)

    def case_gram(self, direction: str, c: int, raw: dict):
        """Delay Gram sums of the whole record of OLC c (Figs. 6, 7 and 10)."""
        return self._gram(self.cache / f"gram_{direction}_{c}.npz", raw[c], self.cfg["virtual_sensing"]["delay"])

    # ---------------------------------------------------------------- sensor layouts
    def layout(self, name: str) -> tuple:
        return tuple(int(i) for i in self.cfg["virtual_sensing"]["layouts"][name])

    def random_layouts(self) -> list:
        return [tuple(int(i) for i in m) for m in self.cfg["virtual_sensing"]["layouts"]["random"]]

    def vs_layouts(self) -> list:
        """Layouts of the main study in the order of evaluation: random, Case A, Case B/C, L2 (L1 is random #1)."""
        out = self.random_layouts()
        for name in ("case_a", "case_bc", "l2", "l1"):
            if self.layout(name) not in out:
                out.append(self.layout(name))
        return out

    def forecast_layouts(self) -> set:
        return {self.layout(n) for n in self.cfg["virtual_sensing"]["forecast_layouts"]}

    @staticmethod
    def mask_key(layout) -> str:
        return " ".join(map(str, np.asarray(layout)))

    def ranks(self, key: str) -> list:
        return as_range(self.cfg["identification"][key])

    def group_of(self, tests):
        """Cross-fitting group of the test OLC (0: OLC 1-6, 1: OLC 7-12)."""
        return tests.isin(self.cfg["virtual_sensing"]["crossfit_group_b"])
