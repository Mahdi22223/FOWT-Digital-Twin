"""End-to-end smoke tests for the eftwin package.

These tests are intentionally lightweight. Their job is to confirm that
the package imports cleanly, that the data loader handles the synthetic
sample CSV, and that the Hankel-DMD identification, virtual sensing, and
SSI-COV pipelines run to completion on that sample without raising. They
do not assert numerical equivalence to the paper results, because the
synthetic sample data is not the paper's full dataset; that level of
verification belongs to the paper's reproduction guide.

Running the tests:
    pytest -q tests/

The tests automatically copy the synthetic sample into a temporary
``data/full/`` directory inside ``tmp_path`` so that no state leaks across
the test session and the user's real ``data/full/`` directory (if any)
is not touched.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

# Allow running pytest from the repository root without first installing
# the package in editable mode. This keeps the smoke test usable in CI
# environments that bypass installation.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from eftwin.constants import analysis_columns, TOWER_HEIGHTS_9
from eftwin.data_io import (
    DTMismatchError,
    check_dt_consistency,
    discover_case_files,
    load_direction_matrix,
    read_case_csv,
    resolve_columns,
)
from eftwin.hankel_dmd import run_hankel_dmd
from eftwin.modal_analysis import (
    complex_mac,
    extract_physical_modes,
    modal_parameters,
    real_projected_mac,
)
from eftwin.virtual_sensing import run_missing_sensor_generalization


def _bootstrap_sample(tmp_path: Path) -> list[Path]:
    """Copy the synthetic sample into a temporary data/full directory.

    Several Hankel-DMD parameters of interest (``hankel_d * svd_rank``)
    require more than one case to be statistically meaningful, but the
    synthetic sample contains only one. That is fine for a smoke test
    because we only need the pipeline to *execute*, not produce
    paper-relevant numerics. We therefore duplicate the same sample twice
    so that ``np.vstack`` has multiple snapshot blocks to glue together.
    """
    sample = ROOT / "data" / "sample" / "sample_case_small.csv"
    full_dir = tmp_path / "data" / "full"
    full_dir.mkdir(parents=True)
    shutil.copy(sample, full_dir / "Case_1.csv")
    shutil.copy(sample, full_dir / "Case_2.csv")
    return discover_case_files(full_dir)


def test_package_metadata():
    import eftwin
    # Version is the only metadata we contractually expose; check that it
    # parses as the expected three-part semver string.
    parts = eftwin.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_constants_helpers():
    # The directional helpers are the public-facing API for OpenFAST
    # column names and are used by every runner script. They must return
    # exactly nine acceleration channels and nine moment channels for
    # both directions, with the ALxt+MLyt and ALyt+MLxt pairings the
    # paper uses.
    acc_fa, mom_fa = analysis_columns("fore_aft")
    acc_ss, mom_ss = analysis_columns("side_to_side")
    assert len(acc_fa) == 9 and len(mom_fa) == 9
    assert len(acc_ss) == 9 and len(mom_ss) == 9
    assert all("ALxt" in c for c in acc_fa)
    assert all("MLyt" in c for c in mom_fa)
    assert all("ALyt" in c for c in acc_ss)
    assert all("MLxt" in c for c in mom_ss)
    assert TOWER_HEIGHTS_9.shape == (9,)


def test_data_loader_resolves_columns(tmp_path):
    case_files = _bootstrap_sample(tmp_path)
    df = read_case_csv(case_files[0])
    acc_cols, mom_cols = analysis_columns("fore_aft")
    # The robust resolver must find every requested column even when the
    # CSV was written by OpenFAST with the unit suffix attached.
    resolved_acc = resolve_columns(df, acc_cols)
    resolved_mom = resolve_columns(df, mom_cols)
    assert len(resolved_acc) == 9
    assert len(resolved_mom) == 9


def test_hankel_dmd_pipeline_runs(tmp_path):
    case_files = _bootstrap_sample(tmp_path)
    acc_cols, mom_cols = analysis_columns("fore_aft")
    # Use a smaller delay than the paper default because the synthetic
    # sample has only ~800 time samples per case. The point of this test
    # is only that the pipeline executes without error; numerical values
    # are not asserted because the synthetic data is not paper-relevant.
    result = run_hankel_dmd(
        case_files,
        acc_cols,
        mom_cols,
        direction_name="Fore-Aft",
        dt=0.0125,
        hankel_d=10,
        svd_rank=6,
    )
    # Exactly 18 physical channels (9 acc + 9 mom) come out of the load
    # step regardless of the underlying data.
    assert result.X_raw.shape[0] == 18
    # The DMD model should have at least one mode after fitting.
    assert result.model.modes.shape[1] > 0


def test_modal_parameters_and_extraction(tmp_path):
    case_files = _bootstrap_sample(tmp_path)
    acc_cols, mom_cols = analysis_columns("fore_aft")
    result = run_hankel_dmd(
        case_files, acc_cols, mom_cols, direction_name="Fore-Aft",
        dt=0.0125, hankel_d=10, svd_rank=6,
    )
    omega, freqs, damping = modal_parameters(result.model.eigs, dt=0.0125)
    # Frequencies and damping arrays must be the same length as the
    # eigenvalue array.
    assert freqs.shape == result.model.eigs.shape
    assert damping.shape == result.model.eigs.shape

    modes_df, stats_df = extract_physical_modes(result, dt=0.0125, low_f=0.0, high_f=20.0)
    # The extraction may legitimately return zero rows on a single-tone
    # synthetic signal; the test only verifies the call does not raise.
    assert hasattr(modes_df, "shape")
    assert hasattr(stats_df, "shape")


def test_mac_self_match_is_unity():
    import numpy as np
    v = np.array([1 + 0j, 2 + 1j, 3 - 2j, 0.5 + 0.5j])
    # A mode shape compared with itself must score MAC = 1 exactly.
    # This is a property of the MAC formula and a useful regression
    # guard against future refactors of complex_mac.
    assert abs(complex_mac(v, v) - 1.0) < 1e-12

    # For real_projected_mac the relationship is more subtle: the function
    # rotates v_dmd so that its largest-magnitude entry is real and
    # positive, then takes only the real part. For a vector that is
    # ALREADY real and positive, the rotation is the identity and the
    # comparison against its own absolute value collapses to a real
    # MAC of 1. We use this as the cleanest regression check.
    v_real = np.array([1.0, 2.0, 3.0, 0.5])
    assert abs(real_projected_mac(v_real, v_real) - 1.0) < 1e-12

    # As a second regression check, an orthogonal pair of real vectors
    # must score 0. This catches accidental swaps of numerator/denominator.
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(real_projected_mac(a, b)) < 1e-12


def test_dt_consistency_check_accepts_correct_spacing(tmp_path):
    # The synthetic sample is exported at exactly dt = 0.0125 s, which is
    # the value the entire pipeline assumes. The check must not raise on
    # legitimate data.
    case_files = _bootstrap_sample(tmp_path)
    df = read_case_csv(case_files[0])
    observed = check_dt_consistency(df, expected_dt=0.0125, on_mismatch="raise",
                                    file_label=str(case_files[0]))
    assert abs(observed - 0.0125) < 1e-6


def test_dt_consistency_check_catches_mismatch(tmp_path):
    # The defining failure mode this check guards against is a user
    # exporting at OpenFAST DT_Out (typically 0.05 s) instead of at the
    # simulation DT (0.0125 s). Multiplying the time column by 4
    # simulates exactly that.
    case_files = _bootstrap_sample(tmp_path)
    df = read_case_csv(case_files[0])
    df_bad = df.copy()
    df_bad["Time_[s]"] = df_bad["Time_[s]"] * 4   # 0.0125 -> 0.05
    with pytest.raises(DTMismatchError):
        check_dt_consistency(df_bad, expected_dt=0.0125, on_mismatch="raise",
                             file_label="spoofed-mismatch")


def test_load_direction_matrix_routes_dt_check(tmp_path):
    # Verify that load_direction_matrix actually invokes the dt check by
    # giving it a deliberately mismatched file and confirming the error
    # propagates up through the loader.
    case_files = _bootstrap_sample(tmp_path)
    df = read_case_csv(case_files[0])
    df["Time_[s]"] = df["Time_[s]"] * 4   # break dt
    spoofed_path = tmp_path / "data" / "full" / "Case_3.csv"
    df.to_csv(spoofed_path, index=False)

    acc_cols, mom_cols = analysis_columns("fore_aft")
    with pytest.raises(DTMismatchError):
        load_direction_matrix(
            [spoofed_path], acc_cols, mom_cols,
            low_cutoff=0.25, high_cutoff=5.0, fs=80.0, filter_order=4,
            dt_check="raise",
        )


def test_missing_sensor_generalization_runs(tmp_path):
    # The orchestrator function should execute end-to-end against a
    # multi-case dataset. We bootstrap the sample three times so that
    # train_case_count=2 leaves one test case. The point of this test is
    # only that the train -> mask -> reconstruct -> evaluate pipeline
    # executes without error and produces a metrics DataFrame of the
    # expected shape; we do not assert numerical accuracy because the
    # synthetic data is not paper-relevant.
    sample = ROOT / "data" / "sample" / "sample_case_small.csv"
    full_dir = tmp_path / "data" / "full"
    full_dir.mkdir(parents=True)
    for i in range(1, 4):
        shutil.copy(sample, full_dir / f"Case_{i}.csv")
    case_files = discover_case_files(full_dir)

    acc_cols, mom_cols = analysis_columns("fore_aft")
    result = run_missing_sensor_generalization(
        case_files, acc_cols, mom_cols,
        missing_sensors=(0, 8),
        train_case_count=2,
        update_every_seconds=1.0,
        duration=5.0,
        dt=0.0125,
        hankel_d=10,        # smaller than the paper default to fit the synthetic sample
        svd_rank=6,
        trim_steps=10,      # synthetic sample is only 800 rows, so 1000-sample trim would empty it
    )
    metrics = result["prediction"]["metrics"]
    # Two masked sensors -> two metric rows, each with R2/RMSE/NRMSE_percent.
    assert len(metrics) == 2
    assert {"sensor_index", "R2", "RMSE", "NRMSE_percent"}.issubset(metrics.columns)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
