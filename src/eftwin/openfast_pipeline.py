"""OpenFAST case-generation and post-processing helpers.

The functions here wrap the user's original pyFAST automation logic while moving paths and case definitions into configuration files.
"""
from __future__ import annotations

from pathlib import Path
import os
import pandas as pd


def tower_sensor_outlist() -> list[str]:
    sensors = []
    for n in range(1, 10):
        sensors.extend([
            f"TwHt{n}TDxt", f"TwHt{n}TDyt",
            f"TwHt{n}ALxt", f"TwHt{n}ALyt",
            f"TwHt{n}FLxt", f"TwHt{n}FLyt", f"TwHt{n}FLzt",
            f"TwHt{n}MLxt", f"TwHt{n}MLyt", f"TwHt{n}MLzt",
        ])
    return sensors


def build_nonlinear_params(config: dict) -> list[dict]:
    base = config.get('base', {'TMax': 3600, 'DT': 0.0125, 'DT_Out': 0.05})
    rpm = config.get('rotor', {}).get('rpm', 12)
    pitch = config.get('rotor', {}).get('pitch_deg', 0)
    params = []
    ed_outlist = tower_sensor_outlist() + ["RotSpeed", "BldPitch1", "QD_TFA1", "QD_TSS1"]
    servo_vars = ["GenPwr", "GenTq"]
    inflow_vars = ["Wind1VelX"]
    hydro_outlist = ["HydroFxi", "HydroFyi", "Wave1Elev"]
    for case in config['wave_cases']:
        p = dict(base)
        p['EDFile|RotSpeed'] = rpm
        p['EDFile|BlPitch(1)'] = pitch
        p['EDFile|BlPitch(2)'] = pitch
        p['EDFile|BlPitch(3)'] = pitch
        p['EDFile|NTwGages'] = 9
        p['EDFile|TwrGagNd'] = [1, 3, 5, 7, 10, 13, 15, 17, 20]
        p['EDFile|OutList'] = ed_outlist
        p['InflowFile|OutList'] = inflow_vars
        p['ServoFile|OutList'] = servo_vars
        p['HydroFile|RdtnDT'] = base['DT']
        p['HydroFile|WaveHs'] = case['Hs']
        p['HydroFile|WaveTp'] = case['Tp']
        p['HydroFile|WaveDir'] = case['WaveDir']
        p['HydroFile|OutList'] = hydro_outlist
        p['__name__'] = f"case_{case['case']}"
        params.append(p)
    return params


def build_linear_params(config: dict) -> list[dict]:
    base = {'TMax': 100, 'DT': 0.0125, 'DT_Out': 0.05}
    base.update(config.get('linear_base', {}))
    rpm = config.get('linear_rotor', {}).get('rpm', 12)
    pitch = config.get('linear_rotor', {}).get('pitch_deg', 17.1)
    p = dict(base)
    p['EDFile|RotSpeed'] = rpm
    p['EDFile|BlPitch(1)'] = pitch
    p['EDFile|BlPitch(2)'] = pitch
    p['EDFile|BlPitch(3)'] = pitch
    p['EDFile|NTwGages'] = 9
    p['EDFile|TwrGagNd'] = [1, 3, 5, 7, 10, 13, 15, 17, 20]
    p['__name__'] = 'case_1'
    return [p]


def generate_cases(params: list[dict], template_dir: str | Path, output_dir: str | Path, main_file: str, openfast_exe: str | None = None, run: bool = False, n_cores: int = 4):
    import pyFAST.case_generation.case_gen as case_gen
    import pyFAST.case_generation.runner as runner

    fast_files = case_gen.templateReplace(params, str(template_dir), outputDir=str(output_dir), removeRefSubFiles=True, main_file=main_file, oneSimPerDir=False)
    if openfast_exe:
        runner.writeBatch(os.path.join(str(output_dir), '_RUN_ALL.bat'), fast_files, fastExe=openfast_exe)
        if run:
            runner.run_fastfiles(fast_files, fastExe=openfast_exe, parallel=True, showOutputs=True, nCores=n_cores)
    return fast_files


def extract_for_dmd(output_file_path: str | Path, csv_output_path: str | Path):
    try:
        from pyFAST.input_output import FASTOutputFile
    except ImportError:
        from openfast_toolbox.io import FASTOutputFile

    df = FASTOutputFile(str(output_file_path)).toDataFrame()
    targets = ['Time']
    for i in range(1, 10):
        targets += [f"TwHt{i}ALxt", f"TwHt{i}ALyt"]
    for i in range(1, 10):
        targets += [f"TwHt{i}MLxt", f"TwHt{i}MLyt"]

    final_cols = []
    for target in targets:
        for actual in df.columns:
            if target in actual:
                final_cols.append(actual)
                break
    out = df[final_cols]
    csv_output_path = Path(csv_output_path)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_output_path, index=False)
    return out


def export_dmd_csv_batch(input_dir: str | Path, output_dir: str | Path, n_cases: int = 12):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    outputs = []
    for i in range(1, n_cases + 1):
        src = input_dir / f"case_{i}.outb"
        dst = output_dir / f"Case_{i}.csv"
        outputs.append(extract_for_dmd(src, dst))
    return outputs
