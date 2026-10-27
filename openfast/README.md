# OpenFAST input templates (data generation, auxiliary)

The analysis starts from the exported case files (`data/README.md`); OpenFAST is only needed to regenerate them.

```
openfast/
  templates/nonlinear/   main.fst and module files of the 12 time-domain load cases (FEAMooring)
  templates/linear/      main.fst and module files of the linearisation (MAP++)
  baseline/README.md     where to obtain the NREL 5-MW baseline files, TurbSim wind and the OpenFAST executable
```

`scripts/simulation/openfast_cases.py` writes the input files with pyFAST/openfast_toolbox from the templates and
`configs/openfast_cases.yaml` (wave cases of Table `tab:olc`, tower gauges at nodes 1, 3, 5, 7, 10, 13, 15, 17, 20,
`DT = 0.0125 s`, `DT_Out = 0.05 s`, 3600 s), optionally runs OpenFAST (`--run`), and exports `Case_<n>.csv`:

```bash
pip install -e .[simulation]
python scripts/simulation/openfast_cases.py nonlinear --run     # edit openfast_exe in configs/openfast_cases.yaml first
python scripts/simulation/openfast_cases.py export
python scripts/simulation/openfast_cases.py linear --run
```

The analysis expects the output interval 0.05 s (`DT_Out`); `eftwin.io.OpenFASTData` checks the time column.

The verified Table 3 linear reference is the OpenFAST linearisation at `t = 100 s` with 8 m/s steady inflow,
still water, MAP++ mooring, `CompSub = 1` (SubDyn), `DT = 0.0125 s`, `DT_Out = 0.05 s`, and 36 linearisation
positions. At the linearisation instant the rotor is nearly stationary (`0.1801 rpm`). The 12 rpm and 17.1 deg
values in `linear_rotor` are initial conditions used before the steady-state trim and must not be interpreted as
the linearisation operating point.

The historical source deck points to an InflowWind file named `Steady12mps.dat`; that filename is misleading for
the reported reference. The generated `.lin` and associated `.out` are authoritative and show 8 m/s. The case
generator therefore explicitly sets `HWindSpeed = 8.0` for the linear workflow. The turbulent wind file of the
nonlinear reported runs is not part of the repository (`baseline/README.md`), so regenerated nonlinear cases match
the reported ones statistically, not sample by sample, unless the same wind field is used.
