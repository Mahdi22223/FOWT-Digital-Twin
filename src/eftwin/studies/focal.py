"""FOCAL Campaign IV study (Section 5): configuration and data shared by the scripts."""
from __future__ import annotations

from pathlib import Path

from ..config import as_range, ensure_dir
from ..io import FOCAL_CH_LABEL, FOCAL_RUN_LABEL, FOCAL_RUNS, FocalData

P = 4  # channels per direction: 0 nacelle acc., 1 tower-base (platform) acc., 2 tower-top moment, 3 tower-base moment


class FocalStudy:
    runs = FOCAL_RUNS
    run_label = FOCAL_RUN_LABEL
    channel_label = FOCAL_CH_LABEL

    def __init__(self, cfg: dict):
        self.cfg = cfg
        d, p = cfg["data"], cfg["paths"]
        self.data = FocalData(p["csv_dir"], p["cache_dir"], t0=d["t0"], t1=d["t1"], band=d["band"])
        self.dt, self.fs = self.data.dt, self.data.fs
        self.results = ensure_dir(p["results_dir"])
        self.cache = Path(p["cache_dir"])

    def channels(self, run, direction, **kw):
        return self.data.channels(run, direction, **kw)

    def aux(self, run, name, **kw):
        return self.data.aux(run, name, **kw)

    def ranks(self, spec) -> list:
        return as_range(spec)

    def layouts(self) -> dict:
        return {k: tuple(v) for k, v in self.cfg["virtual_sensing"]["layouts"].items()}

    def group_of(self, tests):
        """Cross-fitting group of the test run (0: E11/E13, 1: E14/E15)."""
        return tests.isin(self.cfg["virtual_sensing"]["crossfit_group_b"])
