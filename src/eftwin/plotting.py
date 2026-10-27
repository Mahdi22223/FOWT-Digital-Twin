"""Plotting style of the paper figures (original style, fonts sized for print: 8-10 pt)."""
from __future__ import annotations

import matplotlib.pyplot as plt

BASE = {"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "sans-serif"}


def use_paper_style():
    plt.rcParams.update(BASE)


def serif():
    plt.rcParams.update({"font.family": "serif"})


def sans():
    plt.rcParams.update({"font.family": "sans-serif"})
