# Data

The data are not stored in this repository. Put them in the folders below (default paths of the configurations) or
point the environment variables to them. Files read from CSV are cached as `.npz` in `cache/` on first use.

## OpenFAST study (Sections 3-4): `data/openfast/`

| File | Content |
|---|---|
| `Case_1.csv` ... `Case_12.csv` | One time-domain simulation per operational load case (Table `tab:olc`; `data.cases` in `configs/openfast.yaml`), 3600 s written at the output interval `DT_Out = 0.05 s`. Columns used: `Time_[s]`, `TwHt<i>ALxt_[m/s^2]`, `TwHt<i>ALyt_[m/s^2]`, `TwHt<i>MLxt_[kN-m]`, `TwHt<i>MLyt_[kN-m]` for the nine tower gauges i = 1..9 (base to top). |
| `case_1.1.lin` | OpenFAST linearisation (reference column of Table 3): t = 100 s, 8 m/s steady inflow, still water, MAP++, SubDyn enabled, rotor 0.1801 rpm. |

Override the location with `EFTWIN_OPENFAST_DIR` and `EFTWIN_LIN_FILE`. The time step of every file is checked
against `data.dt` (0.05 s). The files can be regenerated with `scripts/simulation/openfast_cases.py` (see
`openfast/README.md`), but the turbulent wind field and the exact input deck of the reported runs are needed to
reproduce them sample by sample. They are not public at present (about 0.5 GB); see `docs/open_items.md`.

## FOCAL Campaign IV basin tests (Section 5): `data/focal/`

Source: FOCAL Campaign IV "Integrated System Control: Turbine + Hull", NREL/PNNL, Wind Data Hub,
[OSTI 1972267](https://www.osti.gov/biblio/1972267); runs `m02d01t01.ac01` with wind `w02` and waves `e11`, `e13`,
`e14`, `e15`.

| File | Run label |
|---|---|
| `FOCAL_DLC12_Seed1_3.csv` | E11 |
| `FOCAL_DLC12_Seed3_1.csv` | E13 |
| `FOCAL_DLC12_Seed4_1.csv` | E14 |
| `FOCAL_DLC12_Seed5_1.csv` | E15 |

Format: full-scale values (Froude scale 1:70; model sampling 200 Hz, i.e. dt = 0.005 sqrt(70) s at full scale), a
header row with the channel names and a second row with the units. Channels used: `Time`, `accelNacelleAx`,
`accelNacelleAy`, `towerBotAccX`, `towerBotAccY`, `towerTopMx`, `towerTopMy`, `towerBotMx`, `towerBotMy` (analysis),
`genSpeed` (rotor-speed check, Figs. 14-15), `Surge`, `Sway`, `Pitch`, `Roll` (sensor kinematics, Section 5.1).
Override the location with `EFTWIN_FOCAL_DIR`.

The outputs of the FOCAL analysis are saved in `results/focal`, so Tables 6-7 and Fig. 14 can be rebuilt without
these files (`docs/reproduce.md`).
