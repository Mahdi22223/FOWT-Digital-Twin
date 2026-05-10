"""Equation-free digital twin utilities.

This package contains the cleaned, reusable implementation of the methodology
in the arXiv paper *Equation-Free Digital Twins for Nonlinear Structural
Dynamics* (arXiv:2605.00950). The numerical procedures match the original
research scripts that are preserved in ``legacy/``.

Modules:
    constants            -- Default time step, filter cutoffs, tower heights,
                            and OpenFAST channel-name helpers.
    config               -- YAML loading and directory helpers.
    preprocessing        -- Butterworth zero-phase band-pass filter.
    data_io              -- Robust loading of OpenFAST-exported Case_*.csv files,
                            including handling of duplicate column names.
    hankel_dmd           -- Hankel-DMD identification (matches the original
                            ``Filter_Hankel_03.py`` numerical core).
    modal_analysis       -- Frequency, damping, mode-shape extraction, MAC
                            functions (complex MAC and real-projected MAC).
    virtual_sensing      -- Rolling-horizon virtual sensing and the
                            train/test generalization workflow.
    ssi_cov              -- SSI-COV identification and stabilization diagram.
    noise_lyapunov       -- Hankel singular-value rank check and Rosenstein
                            Lyapunov-exponent / prediction-horizon estimation.
    probabilistic_modes  -- Hankel-delay sensitivity sweep producing mean
                            mode shape plus 95% confidence band (matches
                            ``Probablistic_Mode_shape 2.py``).
    plotting             -- Matplotlib/Seaborn plotting helpers separated
                            from the numerical algorithms.
    openfast_pipeline    -- pyFAST/openfast-toolbox helpers for case
                            generation, execution, and CSV export.
"""

__version__ = "0.2.0"

__all__ = [
    "constants",
    "config",
    "preprocessing",
    "data_io",
    "hankel_dmd",
    "modal_analysis",
    "virtual_sensing",
    "ssi_cov",
    "noise_lyapunov",
    "probabilistic_modes",
    "plotting",
    "openfast_pipeline",
]
