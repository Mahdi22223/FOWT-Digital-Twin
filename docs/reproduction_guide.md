# Reproduction guide

This document is a step-by-step walkthrough for reproducing the analysis
results of the paper. It assumes you already have, or are willing to
generate, the twelve-case OpenFAST simulation dataset; if you also need
to regenerate the simulation data from scratch, follow
`docs/openfast_pipeline.md` first and then return here. The analysis
pipeline does not depend on a specific OpenFAST version or installation,
because it operates entirely on the exported `Case_<N>.csv` files.

## Environment setup

Create the conda environment defined in `environment.yml` and install
the package itself in editable mode so that the cleaned modules are
importable from anywhere on the system. The editable install also
gives the runner scripts under `scripts/` access to the package
through the regular Python import system rather than requiring local
path manipulation. The OpenFAST automation libraries (`pyFAST` or
`openfast_toolbox`) are not required for the analysis pipeline, so do
not install them unless you also intend to regenerate simulation data.

```
conda env create -f environment.yml
conda activate equation-free-dt
pip install -e .
```

## Data placement

Place the twelve full case CSV files under `data/full/`. The expected
naming convention is `Case_<N>.csv` where `N` runs from 1 to 12, and
the file content should follow the column specification documented in
`docs/data_description.md`. The data loader inside
`src/eftwin/data_io.py` validates each loaded case against the
configured `dt` and raises a `DTMismatchError` if the actual time
spacing differs by more than 0.0001 seconds. If you hit that error,
either re-export your cases at the configured rate or update the `dt`
field in the analysis YAML files to match what your CSVs actually
contain; both responses are valid and the appropriate choice depends
on whether you want to keep the band-pass filter design unchanged
(which requires the configured rate) or accept a different design
matched to your data rate.

If you also intend to run the MAC validation against the OpenFAST
linearisation mode shapes, you will additionally need an
`Extracted_Mode_Shapes.xlsx` file produced from a `.lin` output of
the linearisation case. To produce it, run

```
python scripts/openfast/extract_linear_modes.py \
    --lin-file path/to/case_1.1.lin \
    --output results/tables/Extracted_Mode_Shapes.xlsx
```

with `--lin-file` pointing at the actual linearisation output file
produced by your OpenFAST run. The default destination puts the file
where the MAC validation runner looks for it.

## Analysis pipeline

The analysis is split into orchestration scripts that should be run
in the order shown below. Each script reads its parameters from a
YAML configuration file under `configs/` and writes its outputs to
`results/figures/` and `results/tables/`. The scripts are largely
independent in that you can re-run any single step without re-running
the others, with the exception that the MAC validation script
consumes the CSV outputs of the Hankel-DMD and SSI-COV runners.

The first step is the Hankel-DMD identification, which produces the
fundamental modal frequencies, damping ratios, and complex mode
shapes for both the fore-aft and side-to-side directions. The second
step is the in-sample virtual sensing reconstruction, which simulates
the failure of two tower sensors and recovers their time series from
the remaining channels on the same dataset that was used for
identification. The third step is the missing-sensor generalisation
experiment, which is the central virtual-sensing contribution of the
paper: the Hankel-DMD model is fitted on the first eight cases of the
wave-load sweep, designated tower channels are masked from the four
unseen test cases, and the model reconstructs them across operational
conditions it has not been trained on. The fourth step is the SSI-
COV cross-validation, which provides an independent modal
identification using a classical subspace method and writes the mode-
shape CSVs that the MAC validation reads. The fifth step is the SSI
stabilisation diagram, which traces pole persistence across model
orders. The sixth step is the Hankel-delay sensitivity sweep, which
produces the probabilistic mode shapes with 95 percent confidence
bands. The seventh step is the MAC validation, which compares the
Hankel-DMD and SSI-COV mode shapes against the OpenFAST linearisation
shapes.

```
python scripts/analysis/run_hankel_dmd.py --config configs/hankel_dmd.yaml
python scripts/analysis/run_virtual_sensing.py --config configs/virtual_sensing.yaml --direction fore_aft
python scripts/analysis/run_virtual_sensing.py --config configs/virtual_sensing.yaml --direction side_to_side
python scripts/analysis/run_missing_sensor_generalization.py --config configs/missing_sensor_generalization.yaml --direction fore_aft
python scripts/analysis/run_missing_sensor_generalization.py --config configs/missing_sensor_generalization.yaml --direction side_to_side
python scripts/analysis/run_ssi_cov.py --config configs/ssi_cov.yaml --direction fore_aft
python scripts/analysis/run_ssi_cov.py --config configs/ssi_cov.yaml --direction side_to_side
python scripts/analysis/run_ssi_stabilization.py --config configs/ssi_cov.yaml --direction fore_aft
python scripts/analysis/run_ssi_stabilization.py --config configs/ssi_cov.yaml --direction side_to_side
python scripts/analysis/run_probabilistic_modes.py --config configs/probabilistic_modes.yaml
python scripts/analysis/run_mac_validation.py --config configs/mac_validation.yaml
```

## Parameters preserved from the originals

The values in the table below match the corresponding hardcoded
constants in the original research scripts and are encoded as YAML
fields under `configs/`. They are reproduced here so that anyone
reading this document does not have to open the YAML files to learn
which numbers the analysis depends on. The Hankel-DMD rank differs
across workflows: identification uses rank 24 because the goal is to
extract a small number of physically interpretable modes; the
missing-sensor generalisation experiment uses rank 34 because the
additional rank is required to reconstruct masked channels accurately
across cases the model has not seen during training; the
probabilistic mode-shape sweep uses rank 35 because the goal there is
to study mode-shape stability across Hankel delays rather than rank
parsimony.

| Quantity                                       | Value           |
|------------------------------------------------|----------------:|
| Time step `dt` (analysis configuration)        | 0.0125 s        |
| Sampling frequency assumed by filter           | 80 Hz           |
| Band-pass filter cutoffs                       | 0.25 - 5.0 Hz   |
| Filter order                                   | 4               |
| Hankel delay `d`                               | 60              |
| Hankel-DMD rank for identification             | 24              |
| Hankel-DMD rank for missing-sensor experiment  | 34              |
| Hankel-DMD rank for probabilistic sweep        | 35              |
| Missing-sensor train/test split                | 8 train / 4 test|
| Missing-sensor masked indices                  | [0, 8]          |
| Missing-sensor rolling-update horizon          | 1.0 s           |
| Missing-sensor transient trim per case         | 1000 samples    |
| SSI-COV block rows                             | 60              |
| SSI-COV model order (fixed)                    | 40              |
| SSI-COV stabilisation order range              | 2 - 60 step 2   |
| SSI-COV downsampling factor                    | 4               |
| Stabilisation tolerance: frequency             | 0.01 (relative) |
| Stabilisation tolerance: damping               | 0.05 (relative) |
| Stabilisation tolerance: MAC                   | 0.98            |
| Lyapunov embedding dimension                   | 10              |
| Lyapunov time delay                            | 10 samples      |
| Lyapunov fit region                            | first 1.0 s     |

The "Time step `dt` (analysis configuration)" entry is the value
encoded in the YAML configuration files. The actual `dt` present in
your `Case_<N>.csv` files depends on how OpenFAST was run, and the
loader's consistency check is what bridges the two.

## Outputs

The analysis pipeline produces three categories of artefacts. Figures
are written under `results/figures/` and include the eigenvalue
stability maps, the mode-shape plots, the virtual-sensing
reconstruction time series for both the in-sample and the missing-
sensor experiments, the SSI stabilisation diagrams, the delay-
sensitivity twin-axis plots, the probabilistic mode-shape plots with
95 percent confidence bands, and the MAC heatmaps. Tabular results
are written under `results/tables/` and include the modal frequency
and damping CSVs, the mode-shape CSVs, the virtual-sensing
reconstruction metrics (R squared, RMSE, NRMSE), the missing-sensor
generalisation metrics, the delay-sensitivity sweep table, and the
MAC summary tables. The `results/models/` directory is reserved for
serialised identified models if a user chooses to save them. The
`.gitkeep` files in each subdirectory ensure that the empty
directories survive a clean checkout.

## Smoke test

If you want to verify that the pipeline runs end-to-end without
placing the full dataset, copy the synthetic sample into `data/full/`
and run the identification step:

```
mkdir -p data/full
cp data/sample/sample_case_small.csv data/full/Case_1.csv
python scripts/analysis/run_hankel_dmd.py --config configs/hankel_dmd.yaml
```

The synthetic data has only one case, so cross-case workflows
including the missing-sensor generalisation runner and the MAC
validation will not produce paper-relevant results from it. The
smoke test is intended only to verify that the loaders, filter, fit,
and plotting code execute without error. The Jupyter notebook at
`notebooks/01_hankel_dmd_tutorial.ipynb` provides an interactive
version of the same end-to-end check that walks through identification,
mode extraction, virtual sensing, and SSI-COV in sequence.
