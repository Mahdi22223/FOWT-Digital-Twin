"""Data loading, OpenFAST channel resolution, and time-step validation.

This module is the entry point through which every analysis routine sees
the raw OpenFAST simulation data. It is intentionally defensive: it sorts
case files numerically rather than lexicographically (so that ``Case_2``
appears before ``Case_10``), it tolerates the duplicate-column situation
that OpenFAST sometimes produces for ``TwHt1MLxt`` and ``TwHt1MLyt``, and
it validates the assumed sampling rate against the actual ``Time_[s]``
column of every loaded case.

The time-step validation deserves an explicit explanation because it
guards against a silent and consequential failure mode. The Hankel-DMD
pipeline assumes ``dt = 0.0125 s`` (corresponding to a sampling frequency
of 80 Hz). The Butterworth band-pass filter is designed against that
sampling frequency, the Hankel-DMD eigenvalue-to-frequency mapping uses
that ``dt``, and the rolling-horizon virtual sensing computes its
prediction window in samples derived from ``dt``. OpenFAST writes its
output to disk at the ``DT_Out`` interval, which defaults to ``0.05 s``
(20 Hz) in the templates shipped with this repository. If a user runs
the pipeline against case files that were exported at the OpenFAST
output rate rather than at the simulation rate, every downstream
calculation is silently wrong: the filter assumes the wrong Nyquist
frequency, the modal frequencies come out scaled by a factor of 4, and
the virtual-sensing reconstruction does not converge. This module
therefore checks the median spacing of the ``Time_[s]`` column on every
case load and raises ``DTMismatchError`` if it disagrees with the
configured ``dt`` by more than a configurable tolerance.
"""
from __future__ import annotations

from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd

from .preprocessing import apply_zero_phase_filter


# -----------------------------------------------------------------------------
# Custom exception so the calling code can distinguish a dt mismatch from
# any other failure mode (for example, missing columns or unreadable file).
# -----------------------------------------------------------------------------
class DTMismatchError(ValueError):
    """Raised when the median Time_[s] spacing disagrees with the configured dt."""


# -----------------------------------------------------------------------------
# Default tolerance: the time column is written by OpenFAST as a 32-bit
# float, so the actual spacing can drift from the nominal value at the
# 1e-7 to 1e-5 level. Anything tighter than this can produce false
# positives on legitimate data; anything looser can hide real DT_Out
# vs DT mismatches at the percent level.
# -----------------------------------------------------------------------------
DT_TOLERANCE_DEFAULT = 1e-4


def discover_case_files(data_dir: str | Path, pattern: str = "Case_*.csv") -> list[Path]:
    """Return Case_*.csv files sorted numerically by their case index.

    Lexicographic sorting would place ``Case_10.csv`` immediately after
    ``Case_1.csv`` and before ``Case_2.csv``, scrambling the train/test
    split used by the missing-sensor generalisation experiment. Sorting
    by the integer in the filename stem keeps the cases in their
    physical order across the wave-load sweep.
    """
    data_dir = Path(data_dir)
    files = list(data_dir.glob(pattern))

    def key(p: Path):
        m = re.search(r"(\d+)", p.stem)
        return int(m.group(1)) if m else p.name

    return sorted(files, key=key)


def read_case_csv(path: str | Path) -> pd.DataFrame:
    """Read an OpenFAST-exported CSV, tolerating delimiter variations.

    pyFAST and OpenFAST write CSVs with either commas or whitespace as
    the delimiter depending on the version and the export pathway. The
    ``sep=None`` plus ``engine='python'`` combination lets pandas sniff
    the delimiter from the file's first line.
    """
    return pd.read_csv(path, sep=None, engine="python")


def find_time_column(df: pd.DataFrame) -> str | None:
    """Locate the time column in an OpenFAST-exported DataFrame.

    OpenFAST names it ``Time_[s]`` by convention but some pyDatView and
    third-party export paths produce variants such as ``Time``,
    ``time_s``, or ``Time [s]`` (note the space). We accept any column
    whose normalised name starts with ``time``.
    """
    for col in df.columns:
        normalised = str(col).strip().lower().replace("[", "").replace("]", "").replace("_", "")
        if normalised.startswith("times") or normalised == "time" or normalised == "times":
            return col
    return None


def check_dt_consistency(
    df: pd.DataFrame,
    expected_dt: float,
    *,
    tolerance: float = DT_TOLERANCE_DEFAULT,
    file_label: str = "<unknown>",
    on_mismatch: str = "raise",
) -> float:
    """Verify the median Time_[s] spacing matches the configured dt.

    Returns the actual median spacing observed in the file. The
    ``on_mismatch`` argument controls behaviour when the observed
    spacing disagrees with ``expected_dt`` by more than ``tolerance``:
    ``"raise"`` raises ``DTMismatchError`` (the safe default), ``"warn"``
    emits a ``UserWarning`` and returns, and ``"ignore"`` silently
    returns the observed spacing for the caller to handle.
    """
    time_col = find_time_column(df)
    if time_col is None:
        warnings.warn(
            f"{file_label}: no recognisable Time column found; "
            "skipping dt consistency check.",
            UserWarning,
            stacklevel=2,
        )
        return float("nan")

    times = df[time_col].values
    if times.size < 2:
        return float("nan")

    diffs = np.diff(times)
    observed_dt = float(np.median(diffs))
    if abs(observed_dt - expected_dt) > tolerance:
        message = (
            f"{file_label}: median Time spacing {observed_dt:.6f} s does not match "
            f"the configured dt {expected_dt:.6f} s (tolerance {tolerance:g}). "
            "This usually means the case CSV was exported at OpenFAST's DT_Out "
            "rate rather than at the simulation DT rate. Re-export the case "
            "with DT_Out = DT, or update the dt field in your YAML configuration "
            f"to match the observed value ({observed_dt:.6f} s)."
        )
        if on_mismatch == "raise":
            raise DTMismatchError(message)
        if on_mismatch == "warn":
            warnings.warn(message, UserWarning, stacklevel=2)
    return observed_dt


def resolve_columns(df: pd.DataFrame, requested: list[str]) -> list[str]:
    """Map canonical OpenFAST channel names to actual DataFrame columns.

    The OpenFAST exporter sometimes produces duplicate column names for
    ``TwHt1MLxt_[kN-m]`` and ``TwHt1MLyt_[kN-m]`` because the channel
    appears in both ElastoDyn and SubDyn output streams. Pandas
    auto-suffixes the duplicates with ``.1``, ``.2``, etc. This
    resolver handles both the unit-suffix variation (``_[kN-m]``) and
    the duplicate-name suffixes by performing substring matching while
    preferring exact matches when they exist, and never reusing a
    column once it has been claimed by an earlier requested name.
    """
    resolved: list[str] = []
    used: set[str] = set()
    for target in requested:
        stem = target.split("_[")[0]
        candidates = [c for c in df.columns if c not in used and (c == target or stem in str(c))]
        if not candidates:
            raise KeyError(f"Required channel not found: {target}")
        # Prefer exact match; otherwise first substring match in file order.
        exact = [c for c in candidates if c == target]
        chosen = exact[0] if exact else candidates[0]
        resolved.append(chosen)
        used.add(chosen)
    return resolved


def load_direction_matrix(
    case_files: list[str | Path],
    acc_cols: list[str],
    mom_cols: list[str],
    *,
    low_cutoff: float,
    high_cutoff: float,
    fs: float,
    filter_order: int = 4,
    trim_steps: int = 0,
    dt_check: str = "raise",
    dt_tolerance: float = DT_TOLERANCE_DEFAULT,
) -> np.ndarray:
    """Load, validate, filter, and stack 9 acc + 9 mom channels per case.

    The returned matrix has shape ``(18, total_time)`` with the row
    layout ``[9 accelerations | 9 bending moments]``. The construction
    matches the original ``Filter_Hankel_03.py`` pipeline exactly:
    discover columns, apply the band-pass filter case by case, optionally
    trim the leading and trailing transient samples, and ``hstack`` the
    per-case blocks before stacking them along the time axis.

    The new ``dt_check`` argument routes each case through
    ``check_dt_consistency`` before filtering. ``"raise"`` (the default)
    aborts with ``DTMismatchError`` if any case has the wrong sampling
    interval; ``"warn"`` emits a ``UserWarning`` and proceeds anyway,
    which is useful when knowingly running the pipeline at a
    non-standard sampling rate; ``"ignore"`` skips the check entirely
    for users who have already verified their data and want maximum
    speed during repeated reruns.
    """
    expected_dt = 1.0 / fs
    snapshots = []
    for file in case_files:
        df = read_case_csv(file)
        check_dt_consistency(
            df,
            expected_dt=expected_dt,
            tolerance=dt_tolerance,
            file_label=str(file),
            on_mismatch=dt_check,
        )
        acc_actual = resolve_columns(df, acc_cols)
        mom_actual = resolve_columns(df, mom_cols)
        acc = df[acc_actual].values
        mom = df[mom_actual].values
        if trim_steps:
            acc = acc[trim_steps:-trim_steps]
            mom = mom[trim_steps:-trim_steps]
        acc_clean = apply_zero_phase_filter(acc, low_cutoff, high_cutoff, fs, filter_order)
        mom_clean = apply_zero_phase_filter(mom, low_cutoff, high_cutoff, fs, filter_order)
        snapshots.append(np.hstack((acc_clean, mom_clean)))
    return np.vstack(snapshots).T
