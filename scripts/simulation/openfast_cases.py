"""Regenerate the OpenFAST data (auxiliary; needs OpenFAST v3.2.1, the NREL 5-MW baseline files and pyFAST or
openfast_toolbox). See openfast/README.md for the templates and the open items.

  nonlinear   12 time-domain cases of the wave-load sweep (DT = 0.0125 s, outputs at DT_Out = 0.05 s, 3600 s)
  linear      the linearisation case
  export      .outb -> Case_<n>.csv with the time column and the tower-gauge channels used by the analysis
Usage: python scripts/simulation/openfast_cases.py {nonlinear,linear,export} [--config configs/openfast_cases.yaml] [--run]
"""
import argparse
import os
from pathlib import Path

from eftwin.config import load_config


def tower_sensor_outlist() -> list:
    sensors = []
    for n in range(1, 10):
        sensors.extend([f"TwHt{n}TDxt", f"TwHt{n}TDyt", f"TwHt{n}ALxt", f"TwHt{n}ALyt", f"TwHt{n}FLxt", f"TwHt{n}FLyt",
                        f"TwHt{n}FLzt", f"TwHt{n}MLxt", f"TwHt{n}MLyt", f"TwHt{n}MLzt"])
    return sensors


def nonlinear_params(cfg: dict) -> list:
    base = cfg["base"]
    rpm, pitch = cfg["rotor"]["rpm"], cfg["rotor"]["pitch_deg"]
    params = []
    for case in cfg["wave_cases"]:
        p = dict(base)
        p["EDFile|RotSpeed"] = rpm
        for b in (1, 2, 3):
            p[f"EDFile|BlPitch({b})"] = pitch
        p["EDFile|NTwGages"] = 9
        p["EDFile|TwrGagNd"] = cfg["tower_gauge_nodes"]
        p["EDFile|OutList"] = tower_sensor_outlist() + ["RotSpeed", "BldPitch1", "QD_TFA1", "QD_TSS1"]
        p["InflowFile|OutList"] = ["Wind1VelX"]
        p["ServoFile|OutList"] = ["GenPwr", "GenTq"]
        p["HydroFile|RdtnDT"] = base["DT"]
        p["HydroFile|WaveHs"] = case["Hs"]
        p["HydroFile|WaveTp"] = case["Tp"]
        p["HydroFile|WaveDir"] = case["WaveDir"]
        p["HydroFile|OutList"] = ["HydroFxi", "HydroFyi", "Wave1Elev"]
        p["__name__"] = f"case_{case['case']}"
        params.append(p)
    return params


def linear_params(cfg: dict) -> list:
    p = dict(cfg["linear_base"])
    p["InflowFile|HWindSpeed"] = cfg["linear_inflow"]["HWindSpeed"]
    p["HydroFile|WaveMod"] = cfg["linear_hydro"]["WaveMod"]
    p["EDFile|RotSpeed"] = cfg["linear_rotor"]["rpm"]
    for b in (1, 2, 3):
        p[f"EDFile|BlPitch({b})"] = cfg["linear_rotor"]["pitch_deg"]
    p["EDFile|NTwGages"] = 9
    p["EDFile|TwrGagNd"] = cfg["tower_gauge_nodes"]
    p["__name__"] = "case_1"
    return [p]


def generate(params, template_dir, output_dir, main_file, openfast_exe=None, run=False, n_cores=4):
    try:
        import pyFAST.case_generation.case_gen as case_gen
        import pyFAST.case_generation.runner as runner
    except ImportError:
        import openfast_toolbox.case_generation.case_gen as case_gen
        import openfast_toolbox.case_generation.runner as runner

    files = case_gen.templateReplace(params, str(template_dir), outputDir=str(output_dir), removeRefSubFiles=True,
                                     main_file=main_file, oneSimPerDir=False)
    if openfast_exe:
        runner.writeBatch(os.path.join(str(output_dir), "_RUN_ALL.bat"), files, fastExe=openfast_exe)
        if run:
            runner.run_fastfiles(files, fastExe=openfast_exe, parallel=True, showOutputs=True, nCores=n_cores)
    return files


def export(input_dir, output_dir, n_cases=12):
    """Write Case_<n>.csv (Time and the tower-gauge accelerations and bending moments) from case_<n>.outb."""
    try:
        from pyFAST.input_output import FASTOutputFile
    except ImportError:
        from openfast_toolbox.io import FASTOutputFile
    targets = ["Time"] + [f"TwHt{i}AL{a}t" for i in range(1, 10) for a in "xy"] + \
              [f"TwHt{i}ML{a}t" for i in range(1, 10) for a in "xy"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i in range(1, n_cases + 1):
        df = FASTOutputFile(str(Path(input_dir) / f"case_{i}.outb")).toDataFrame()
        cols = [next(c for c in df.columns if t in c) for t in targets]
        df[cols].to_csv(Path(output_dir) / f"Case_{i}.csv", index=False)
        print("written", Path(output_dir) / f"Case_{i}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("nonlinear", "linear", "export"))
    ap.add_argument("--config", default="configs/openfast_cases.yaml")
    ap.add_argument("--run", action="store_true", help="run OpenFAST after writing the input files")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.what == "export":
        export(cfg["nonlinear_output_dir"], cfg["csv_output_dir"], len(cfg["wave_cases"]))
    else:
        lin = a.what == "linear"
        files = generate(linear_params(cfg) if lin else nonlinear_params(cfg),
                         cfg["linear_template_dir" if lin else "nonlinear_template_dir"],
                         cfg["linear_output_dir" if lin else "nonlinear_output_dir"], cfg.get("main_file", "main.fst"),
                         cfg.get("openfast_exe"), run=a.run, n_cores=cfg.get("n_cores", 4))
        print("\n".join(map(str, files)))
