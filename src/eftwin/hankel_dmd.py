"""Hankel-DMD identification routines."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from pydmd import HankelDMD

from .constants import FS_DEFAULT, LOW_CUTOFF_HZ_DEFAULT, HIGH_CUTOFF_HZ_DEFAULT, FILTER_ORDER_DEFAULT
from .data_io import load_direction_matrix


@dataclass
class DMDAnalysisResult:
    model: HankelDMD
    scaler: StandardScaler
    X_raw: np.ndarray
    X_scaled: np.ndarray
    name: str
    case_files: list[Path]

    def as_legacy_dict(self) -> dict:
        """Return the dictionary shape expected by the original virtual-sensing script."""
        return {"model": self.model, "scaler": self.scaler, "X_raw": self.X_raw, "X_scaled": self.X_scaled, "name": self.name}


def run_hankel_dmd(
    case_files: list[str | Path],
    acc_cols: list[str],
    mom_cols: list[str],
    *,
    direction_name: str,
    dt: float = 0.0125,
    low_cutoff: float = LOW_CUTOFF_HZ_DEFAULT,
    high_cutoff: float = HIGH_CUTOFF_HZ_DEFAULT,
    filter_order: int = FILTER_ORDER_DEFAULT,
    hankel_d: int = 60,
    svd_rank: int = 24,
    tlsq_rank: int = 0,
    exact: bool = True,
    opt: bool = False,
) -> DMDAnalysisResult:
    """Run the same load/filter/scale/Hankel-DMD workflow used in the original scripts."""
    fs = 1.0 / dt
    X = load_direction_matrix(
        case_files,
        acc_cols,
        mom_cols,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        fs=fs,
        filter_order=filter_order,
    )
    scaler = StandardScaler()
    scaler.fit(X.T)
    X_scaled = scaler.transform(X.T).T

    hdmd = HankelDMD(svd_rank=svd_rank, tlsq_rank=tlsq_rank, exact=exact, opt=opt, d=hankel_d)
    hdmd.fit(X_scaled)

    return DMDAnalysisResult(
        model=hdmd,
        scaler=scaler,
        X_raw=X,
        X_scaled=X_scaled,
        name=direction_name,
        case_files=[Path(p) for p in case_files],
    )
