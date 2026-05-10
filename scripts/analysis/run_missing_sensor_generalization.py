"""Runner for the missing/failed sensor generalisation experiment.

This is the orchestration script for the cross-case missing-sensor
reconstruction workflow that appears as the closing experiment in the
arXiv paper. It corresponds directly to the original
``Missing_Sensor_Generalization_2 2.py`` script, which the cleaned codebase
has split into reusable functions inside ``src/eftwin/virtual_sensing.py``.

The experiment proceeds in three stages. First, a Hankel-DMD model is
fitted on the first ``train_case_count`` cases (default 8 of the 12 wave
load cases). Second, designated tower sensors are masked from the test
cases, and the trained model is used to reconstruct their time series
through the rolling-horizon pseudoinverse algorithm. Third, the
reconstruction accuracy is reported as R squared, RMSE, and NRMSE per
masked sensor, and the recovered traces are plotted alongside ground
truth for visual comparison. None of the numerical work happens in this
file; it is a thin orchestrator that reads YAML configuration, locates
data, calls ``run_missing_sensor_generalization``, and writes outputs.

Typical use:

    python scripts/analysis/run_missing_sensor_generalization.py \
        --config configs/missing_sensor_generalization.yaml \
        --direction fore_aft

Override the direction with ``--direction side_to_side`` to run the same
experiment on the side-to-side state vector. The YAML file controls
every other parameter, including the masked-sensor indices, the Hankel
hyperparameters, the rolling-update horizon, and the train/test split.
"""
from pathlib import Path
import argparse

from eftwin.config import load_yaml, ensure_dir
from eftwin.constants import analysis_columns
from eftwin.data_io import discover_case_files
from eftwin.virtual_sensing import run_missing_sensor_generalization
from eftwin.plotting import plot_virtual_sensing_result


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the missing/failed sensor generalisation experiment."
    )
    parser.add_argument(
        "--config",
        default="configs/missing_sensor_generalization.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--direction",
        default="fore_aft",
        choices=["fore_aft", "side_to_side"],
        help="Which tower direction to analyse.",
    )
    args = parser.parse_args()

    # Read configuration. Defaults baked into the YAML match the originals
    # exactly: rank 34, Hankel delay 60, 8/4 train-test split, sensors 0
    # and 8 masked, 1.0 s rolling update, 50 s prediction window for the
    # quick comparison runs (the paper's full run extends to 300 s).
    cfg = load_yaml(args.config)
    case_files = discover_case_files(
        cfg["data_dir"], cfg.get("case_pattern", "Case_*.csv")
    )
    if not case_files:
        raise FileNotFoundError(
            f"No case files found in {cfg['data_dir']}. "
            "Place the full Case_1.csv ... Case_12.csv files there before running."
        )

    train_count = cfg.get("train_case_count", 8)
    if len(case_files) <= train_count:
        raise ValueError(
            f"Need at least train_case_count + 1 cases for a generalisation "
            f"experiment. Found {len(case_files)}, need >{train_count}."
        )

    acc_cols, mom_cols = analysis_columns(args.direction)

    # The YAML "filter" subkey wraps the band-pass parameters and the
    # "hankel" subkey wraps the Hankel-DMD hyperparameters. We unpack
    # them with .get() defaults that match the original script so that
    # users with minimal YAML still get the paper's exact pipeline.
    filt = cfg.get("filter", {})
    hankel = cfg.get("hankel", {})

    result = run_missing_sensor_generalization(
        case_files,
        acc_cols,
        mom_cols,
        missing_sensors=tuple(cfg.get("missing_sensors", [0, 8])),
        train_case_count=train_count,
        update_every_seconds=cfg.get("update_every_seconds", 1.0),
        duration=cfg.get("duration_seconds", 50.0),
        dt=cfg.get("dt", 0.0125),
        low_cutoff=filt.get("low_cutoff_hz", 0.25),
        high_cutoff=filt.get("high_cutoff_hz", 5.0),
        filter_order=filt.get("order", 4),
        hankel_d=hankel.get("delay_d", 60),
        svd_rank=hankel.get("svd_rank", 34),
        opt=hankel.get("opt", False),
        trim_steps=cfg.get("trim_steps", 1000),
    )

    prediction = result["prediction"]
    metrics_df = prediction["metrics"]

    # Persist tabular outputs first so they are available even if the
    # plotting backend is non-interactive or fails to render.
    tab_dir = ensure_dir(cfg.get("outputs", {}).get("tables_dir", "results/tables"))
    fig_dir = ensure_dir(cfg.get("outputs", {}).get("figures_dir", "results/figures"))
    metrics_path = tab_dir / f"{args.direction}_missing_sensor_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Wrote per-sensor accuracy metrics to {metrics_path}")
    print(metrics_df.to_string(index=False))

    # One comparison plot per masked sensor: ground truth in solid black,
    # reconstruction in dashed red. The plot duration matches the
    # prediction duration so that update boundaries are not artificially
    # cropped from the figure.
    for idx in cfg.get("missing_sensors", [0, 8]):
        plot_path = fig_dir / f"{args.direction}_missing_sensor_{idx}.png"
        plot_virtual_sensing_result(
            prediction,
            sensor_index=idx,
            plot_duration=cfg.get("duration_seconds", 50.0),
            save_path=plot_path,
        )
        print(f"Wrote reconstruction plot for sensor {idx} to {plot_path}")


if __name__ == "__main__":
    main()
