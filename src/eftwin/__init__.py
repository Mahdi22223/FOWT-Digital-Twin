"""eftwin: equation-free Hankel-DMD (Koopman) models for operational modal analysis and virtual sensing.

The package is the reference implementation of the revised paper. Every routine reproduces the numerical
convention used for the reported results; where the OpenFAST and FOCAL studies used different conventions
(for example the SSI-COV pole selection), both are kept as explicitly named options.

Modules
    io           OpenFAST case files and FOCAL Campaign IV runs
    preprocess   band-pass filtering and standardisation
    hankel       delay-embedded Gram sums and fold statistics (Hankel matrices are never formed)
    dmd          exact Hankel-DMD from Gram sums, modal parameters, modal energy, pole picking
    modal        MAC, MPC and mode-shape normalisation
    estimators   the two realizations of the virtual sensor: Tikhonov projection and Kalman form
    model        high-level API (fit a model, reconstruct hidden channels)
    validation   time-blocked folds, sufficient statistics, cross-fitting, accuracy metrics
    baselines    comparison methods: SSI-COV and FDD (identification); gappy POD, static least squares,
                 Wiener filter (virtual sensing)
    plotting     plotting style of the paper figures
"""
__version__ = "1.0.0rc1"

from .model import (HankelDMDModel, KalmanEstimator, ProjectionEstimator, fit_hankel_dmd,  # noqa: E402,F401
                    identify_modes)

__all__ = ["HankelDMDModel", "fit_hankel_dmd", "identify_modes", "ProjectionEstimator", "KalmanEstimator",
           "__version__"]
