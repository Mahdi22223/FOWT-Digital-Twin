# Original-submission scripts (reference only)

These are the scripts behind the results of the original submission, kept unchanged so that those results remain
traceable (the complete repository state is the tag `v0.2-original-submission`). They are not maintained and are
not used by the package or the reproduction scripts. Use `src/eftwin` and `scripts/` for new work.

Two processing choices distinguish them from the revised analysis:

* **Time step.** The scripts use `dt = 0.0125 s`, the OpenFAST solver step, whereas the case files are written at
  the output interval `DT_Out = 0.05 s`. The time axis of these scripts is therefore four times shorter than that of
  the files: frequencies are four times higher, and the band-pass filter (designed for 80 Hz sampling) acts on
  0.0625-1.25 Hz of the 20-Hz data. The SSI-COV scripts decimate by 4 and use `4 dt`, which gives the same factor.
* **Concatenation.** All twelve cases are joined into one record before the Hankel matrix is built, so that some delay
  windows span two cases.

The revised analysis reads the time step from the files (0.05 s) and keeps every case as a separate segment (no
delay window spans two cases). `scripts/supplementary/legacy_dt_check.py` applies the identification of
`Filter_Hankel_03.py` to the case files and prints its frequencies next to the frequencies at the true time step.

| Original script | Revised counterpart |
|---|---|
| `hankel_dmd_original/Filter_Hankel_03.py` | `scripts/openfast/identify.py` (Table 3) |
| `SSICOV_2.py`, `SSI_COV_Stablizing.py`, `MAC_Final.py` | `eftwin.baselines.oma`, MACs in `identify.py` |
| `Probablistic_Mode_shape 2.py` | `make_figures.py fig5` (delay-depth sensitivity, no confidence bands) |
| `Virtual_sensing 2.py`, `Missing_Sensor_Generalization_2 2.py` | `scripts/openfast/virtual_sensing.py` (time-blocked protocol) |
| `Noise_Floor_Analysis.py` | `make_figures.py fig10` (singular values; rank choice, Section 4.7) |
| `Lyapunov_Exponent_Estimation.py` | forecast horizon (`make_figures.py fig11`, Section 4.8); `supplementary/lyapunov_surrogate.py` |
| `hankel_dmd_original/OpenFast_Mode_Shapes.py`, `openfast_original/Modal_OutPut_Final_02.py` | `scripts/openfast/linear_reference.py` |
| `openfast_original/Sensor_Tower.py`, `Linearize_ParametricInputs.py`, `Data_for_DmD.py`, `Data_ML.py` | `scripts/simulation/openfast_cases.py` |

`Linearize_ParametricInputs.py` contains one commented-out line (a non-Python placeholder in the original that
prevented parsing); nothing else has been edited.
