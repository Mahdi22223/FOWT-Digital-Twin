"""Shared constants and OpenFAST channel definitions."""
from __future__ import annotations

import numpy as np

DT_DEFAULT = 0.0125
FS_DEFAULT = 1.0 / DT_DEFAULT
LOW_CUTOFF_HZ_DEFAULT = 0.25
HIGH_CUTOFF_HZ_DEFAULT = 5.0
FILTER_ORDER_DEFAULT = 4
HANKEL_DELAY_DEFAULT = 60
TOWER_HEIGHTS_9 = np.linspace(10, 87.6, 20)[[0, 2, 4, 6, 9, 12, 14, 16, 19]]


def tower_acceleration_columns(direction: str) -> list[str]:
    """Return the 9 OpenFAST tower acceleration channel names for a direction."""
    if direction in {"fore_aft", "fa", "x"}:
        return [f"TwHt{i}ALxt_[m/s^2]" for i in range(1, 10)]
    if direction in {"side_to_side", "ss", "y"}:
        return [f"TwHt{i}ALyt_[m/s^2]" for i in range(1, 10)]
    raise ValueError(f"Unknown direction: {direction}")


def tower_moment_columns(direction: str) -> list[str]:
    """Return the 9 OpenFAST tower bending-moment channels used by the paper workflow.

    Fore-aft displacement is paired with local-y bending moment MLyt.
    Side-to-side displacement is paired with local-x bending moment MLxt.
    """
    if direction in {"fore_aft", "fa", "x"}:
        return [f"TwHt{i}MLyt_[kN-m]" for i in range(1, 10)]
    if direction in {"side_to_side", "ss", "y"}:
        return [f"TwHt{i}MLxt_[kN-m]" for i in range(1, 10)]
    raise ValueError(f"Unknown direction: {direction}")


def analysis_columns(direction: str) -> tuple[list[str], list[str]]:
    return tower_acceleration_columns(direction), tower_moment_columns(direction)
