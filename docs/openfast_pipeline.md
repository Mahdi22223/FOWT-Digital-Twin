# OpenFAST pipeline (auxiliary)

The OpenFAST workflow is the upstream half of the data-generation
pipeline that produced the simulation dataset analysed in the paper.
This documentation explains how to run that pipeline locally if you
want to regenerate the dataset from scratch. It is auxiliary to the
main contribution of this repository, which is the Hankel-DMD analysis
methodology under `src/eftwin/`. Users who already have the
`Case_<N>.csv` files (from a published dataset, an institutional
archive, or a colleague who has already run the simulations) can skip
this entire document and proceed directly to the analysis pipeline.

## What this repository does and does not provide

This repository provides the customised case-generation templates
(`openfast/templates/nonlinear/` and `openfast/templates/linear/`),
the Python wrappers that drive OpenFAST through the `pyFAST` or
`openfast_toolbox` libraries (`scripts/openfast/`), and the
configuration file (`configs/openfast_cases.yaml`) that defines the
twelve wave-load cases. These are the parts of the pipeline that
encode this particular study's parametric choices and that this
repository owns.

This repository does not provide the OpenFAST executable, the dynamic-
link controller libraries, the NREL 5 MW baseline input files, or the
turbulent wind-field binaries. Those are external dependencies that
users obtain separately from their authoritative sources. Bundling
them here would duplicate upstream distributions, risk drifting out of
sync, inflate the repository, and conflate this paper's contribution
with NREL's reference data. The dependencies are documented at the
points where they are needed, with pointers to the authoritative
sources.

## External prerequisites

You will need a working OpenFAST installation. OpenFAST binaries for
Windows, Linux, and macOS are available from the OpenFAST GitHub
releases page at https://github.com/OpenFAST/openfast/releases. The
version used for the paper's runs is the OpenFAST 3.x line; later
versions are likely compatible but have not been independently tested
against this codebase. Note the path to the executable on your local
machine, because you will provide that path to the case-generation
scripts via the YAML configuration.

You will need the NREL 5 MW reference turbine baseline files, which
are part of the OpenFAST regression-test repository at
https://github.com/OpenFAST/r-test under the path
`glue-codes/openfast/5MW_Baseline`. Copy that directory into
`openfast/baseline/5MW_Baseline/` inside this repository. The
`.gitignore` excludes that path from version control so the files you
add locally will not be committed accidentally. The detailed file-
layout instructions are in `openfast/baseline/README.md`.

You will need either `pyFAST` or `openfast_toolbox` installed in your
Python environment. Both are commented out in `requirements.txt`
because they depend on the specific OpenFAST distribution available
on your system; install whichever one is appropriate for your
OpenFAST version using pip.

## Step 1: Generate nonlinear cases

The nonlinear case-generation logic was originally implemented in
`Sensor_Tower.py`, which is preserved unchanged under
`legacy/openfast_original/Sensor_Tower.py`. The cleaned counterpart is
in `src/eftwin/openfast_pipeline.py:build_nonlinear_params`, called
from `scripts/openfast/generate_nonlinear_cases.py`. The runner reads
its parameters from `configs/openfast_cases.yaml`, which documents the
twelve wave-condition tuples (significant wave height, peak period,
wave direction), the rotor speed, the blade pitch, the total simulation
time, the time step, and the tower-gauge node selection.

Before running anything, edit `configs/openfast_cases.yaml` to point
the `openfast_exe` field at your local OpenFAST executable. To
generate the OpenFAST input files for all twelve cases without running
the simulations yet:

```
python scripts/openfast/generate_nonlinear_cases.py --config configs/openfast_cases.yaml
```

To generate the input files and immediately execute the simulations in
parallel using the executable specified in the configuration file, add
the `--run` flag:

```
python scripts/openfast/generate_nonlinear_cases.py --config configs/openfast_cases.yaml --run
```

The runner uses `pyFAST.case_generation.case_gen.templateReplace` to
clone the template directory, performs the parametric substitutions on
the copied input files, and uses `pyFAST.case_generation.runner.run_fastfiles`
to execute them in parallel using the number of cores specified by the
`n_cores` field (default 4). The output binaries are written to the
directory specified by `nonlinear_output_dir`, which defaults to
`outputs/parametric/`. That directory is excluded from version control
so the simulation outputs will not be committed.

## Step 2: Generate the linearisation case

The linearisation workflow was originally implemented in
`Linearize_ParametricInputs.py`, which is preserved under
`legacy/openfast_original/`. That original file contained a non-Python
placeholder line in its post-processing section that prevented it from
being imported as a Python module. The cleaned scaffold patches this
issue by commenting out the broken line and the three subsequent
post-processing lines that depended on it, while keeping the original
text inside the comment for traceability. No other line of the
original script has been modified, so the case-generation logic and
the OpenFAST parameter values remain as the original author wrote them.

The cleaned counterpart for case generation is `build_linear_params` in
`openfast_pipeline.py`, called from
`scripts/openfast/generate_linear_cases.py`. Run it the same way as
the nonlinear runner, with `--run` to execute the linearisation
simulation after the input files are generated:

```
python scripts/openfast/generate_linear_cases.py --config configs/openfast_cases.yaml --run
```

The post-processing of the resulting `.lin` files is handled by the
legacy script `Modal_OutPut_Final_02.py`, which the cleaned scaffold
does not refactor because its dependence on specific `pyFAST` and
`openfast_toolbox` API details is easier to maintain in one place
rather than spreading across the package. The cleaned wrapper
`scripts/openfast/extract_linear_modes.py` exposes the legacy
extraction through a proper command-line interface that accepts the
input `.lin` file and the desired output Excel path:

```
python scripts/openfast/extract_linear_modes.py \
    --lin-file outputs_lin/parametric/case_1.1.lin \
    --output results/tables/Extracted_Mode_Shapes.xlsx
```

The extractor imports the legacy script's `analyze_modes` function and
calls it directly with the user-supplied path, so the eigendecomposition
of the OpenFAST state matrix, the tower and blade state-index
identification, the projection of eigenvectors through the output
matrix to physical mode shapes, and the two-sheet Excel export all
happen exactly as in the legacy script.

## Step 3: Export the analysis CSVs

After the nonlinear simulations have produced their `.outb` binaries
under `outputs/parametric/`, the analysis CSVs are produced by
extracting the relevant time and channel columns and writing them in
the order expected by the Hankel-DMD loader. This functionality was
originally in `Data_for_DmD.py` and is reproduced in `extract_for_dmd`
and `export_dmd_csv_batch` in `openfast_pipeline.py`. The runner is:

```
python scripts/openfast/export_dmd_csv.py --input-dir outputs/parametric --output-dir data/full --n-cases 12
```

The export retains the time column, the eighteen acceleration channels
and the eighteen bending-moment channels listed in
`docs/data_description.md`, and writes them to `data/full/Case_1.csv`
through `Case_12.csv`. The analysis pipeline can be run as soon as
this step completes.

A subtle but important point about this step concerns the time-step
choice. OpenFAST writes its output to disk at the `DT_Out` interval. The
Hankel-DMD analysis pipeline assumes a sampling interval of
`dt = 0.0125 s`, which is the internal simulation time step `DT` used
in the paper workflow. Some original OpenFAST examples and local runs
may use a coarser output interval such as `DT_Out = 0.05 s`; if so, the
exported CSVs will be at the wrong sampling rate for the default
analysis configuration, and the data loader in `src/eftwin/data_io.py`
will refuse them with a `DTMismatchError`. The fix is either to set
`DT_Out = DT = 0.0125 s` in the OpenFAST input deck before running the
simulations, or to update the `dt` field in the analysis YAML
configurations to whatever your CSV files actually contain. The case
templates and `configs/openfast_cases.yaml` in this repository use
`DT_Out = DT = 0.0125 s` to avoid this trap.

## Wind input regeneration

The original twelve-case workflow used a turbulent wind binary
(`short_turb12mps.bts`) generated by TurbSim, the IEC-Kaimal turbulence
generator distributed alongside OpenFAST. This file is approximately
70 MB and is excluded from version control. Users who want to
regenerate it should run TurbSim against the input deck `90m_12mps_twr.inp`
from NREL's `5MW_Baseline/Wind/` directory:

```
turbsim 90m_12mps_twr.inp
```

The resulting `90m_12mps_twr.bts` should be renamed to
`short_turb12mps.bts`, or the wind-file references inside the OpenFAST
`InflowWind` input files should be updated to point at whichever
filename you produce. Alternatively, the steady-wind input files
`NRELOffshrBsline5MW_InflowWind_Steady8mps.dat` and
`NRELOffshrBsline5MW_InflowWind_Steady12mps.dat` from the NREL
baseline can be used for runs that do not require a turbulent inflow,
including the linearisation workflow.
