"""Configurations reproduce the settings of the reported results; the FOCAL tables rebuild from the saved CSVs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from eftwin.config import as_range, load_config

ROOT = Path(__file__).resolve().parents[1]
OF = load_config(ROOT / "configs" / "openfast.yaml")
FC = load_config(ROOT / "configs" / "focal.yaml")


def test_random_layouts_match_the_seeded_draw():
    rng = np.random.default_rng(1)
    draws = [[int(i) for i in np.sort(rng.choice(18, q, replace=False))] for q in (1, 2, 3, 4, 6, 9, 14) for _ in range(6)]
    assert OF["virtual_sensing"]["layouts"]["random"] == draws
    assert OF["virtual_sensing"]["layouts"]["l1"] == draws[0] == [8]


def test_openfast_settings_of_the_paper():
    assert OF["data"]["dt"] == 0.05 and OF["data"]["trim"] == 6000 and OF["data"]["band"] == [0.25, 5.0]
    vc = OF["virtual_sensing"]
    assert (vc["delay"], vc["half"], vc["score_start"]) == (60, 33000, 1200)
    assert vc["projection_ranks"] == [24, 34, 50, 70, 90, 120] and vc["alphas"] == [0.001, 0.003, 0.01, 0.03]
    assert vc["kalman_ranks"] == [34, 50, 70, 90, 120]
    assert as_range(OF["identification"]["ranks"]) == list(range(10, 52, 2))
    case_a = sorted(set(range(18)) - {2, 3, 5, 7, 11, 12, 14, 16})
    assert vc["layouts"]["case_a"] == case_a
    assert vc["layouts"]["case_bc"] == sorted(set(range(18)) - {0, 8, 9, 17})


def test_simulation_config_writes_the_analysed_output_interval():
    cases = load_config(ROOT / "configs" / "openfast_cases.yaml")
    assert cases["base"]["DT_Out"] == OF["data"]["dt"] == 0.05
    assert cases["linear_base"]["TMax"] == 100
    assert cases["linear_base"]["DT"] == 0.0125 and cases["linear_base"]["DT_Out"] == 0.05
    assert cases["linear_base"]["CompSub"] == 1 and cases["linear_base"]["CompMooring"] == 1
    assert cases["linear_base"]["NLinTimes"] == 36
    assert cases["linear_inflow"]["HWindSpeed"] == 8.0 and cases["linear_hydro"]["WaveMod"] == 0
    ref = cases["linear_reference"]
    assert ref["time_s"] == 100 and ref["wind_speed_mps"] == 8.0 and ref["rotor_speed_rpm"] == 0.1801
    assert ref["subdyn_enabled"] and ref["still_water"] and ref["n_linearisation_positions"] == 36


def _make_tables():
    spec = importlib.util.spec_from_file_location("make_tables", ROOT / "scripts" / "make_tables.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name, fn", [("table_focal_sysid.tex", "focal_sysid"), ("table_focal_vs.tex", "focal_vs")])
def test_focal_tables_rebuild_from_saved_results(tmp_path, name, fn):
    getattr(_make_tables(), fn)(ROOT / "results" / "focal", tmp_path)
    assert (tmp_path / name).read_text() == (ROOT / "results" / "reference" / "tables" / name).read_text()
