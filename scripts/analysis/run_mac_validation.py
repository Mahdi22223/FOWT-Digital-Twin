"""End-to-end MAC validation script.

The original ``MAC_Final.py`` depends on workspace variables (``df_SSI_FA``,
``df_SSI_SS``, ``df_fa_modes``, ``df_ss_modes``) that exist only after running
the other identification scripts interactively. This driver reads the
persisted CSV/Excel artefacts produced by ``run_hankel_dmd.py`` and
``run_ssi_cov.py``, plus the OpenFAST linear modes exported by the legacy
``Modal_OutPut_Final_02.py``, and runs the full MAC comparison without any
methodological change.

Inputs (defaults; override via configs/mac_validation.yaml):
    results/tables/fore_aft_dmd_modes.csv         (from run_hankel_dmd.py)
    results/tables/side_to_side_dmd_modes.csv     (from run_hankel_dmd.py)
    results/tables/fore_aft_ssi_modes.csv         (from run_ssi_cov.py)
    results/tables/side_to_side_ssi_modes.csv     (from run_ssi_cov.py)
    Extracted_Mode_Shapes.xlsx                    (from Modal_OutPut_Final_02.py)
"""
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd

from eftwin.config import load_yaml, ensure_dir
from eftwin.modal_analysis import (
    parse_complex,
    extract_freq_from_name,
    complex_mac,
    real_projected_mac,
)
from eftwin.plotting import plot_mac_heatmap


def load_complex_csv(path: Path) -> pd.DataFrame:
    """Load a CSV of complex-valued mode shapes (one mode per column)."""
    df = pd.read_csv(path, index_col=0)
    return df.applymap(parse_complex)


def load_openfast_modes(xlsx_path: Path, sheet: str = 'Tower_Modes') -> pd.DataFrame:
    """Load the OpenFAST linearization mode shapes exported by Modal_OutPut_Final_02.py."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, index_col=0)
    except Exception:
        df = pd.read_excel(xlsx_path, index_col=0)
    try:
        df = df.map(parse_complex)
    except AttributeError:
        df = df.applymap(parse_complex)
    return df


def best_match(df: pd.DataFrame, target_freq: float, tolerance: float) -> str | None:
    best, best_diff = None, tolerance
    for col in df.columns:
        f = extract_freq_from_name(col)
        if f < 0:
            continue
        diff = abs(f - target_freq)
        if diff < best_diff:
            best, best_diff = col, diff
    return best


def analyze_direction(
    direction: str,
    df_of: pd.DataFrame,
    df_id: pd.DataFrame,
    targets: dict,
    tolerance: float,
    method_label: str,
    fig_dir: Path,
) -> pd.DataFrame:
    """Run the MAC comparison for one direction. Same algebra as the original."""
    print(f"\n{'='*40} {direction} VALIDATION (method: {method_label}) {'='*40}")
    rows = []
    sel_of, sel_id = [], []
    for name, freqs in targets.items():
        of_col = best_match(df_of, freqs['OF'], tolerance)
        id_col = best_match(df_id, freqs[method_label], tolerance)
        if of_col and id_col:
            if of_col not in sel_of:
                sel_of.append(of_col)
            if id_col not in sel_id:
                sel_id.append(id_col)
            v_of = df_of[of_col].values
            v_id = df_id[id_col].values
            f_diff = abs(extract_freq_from_name(of_col) - extract_freq_from_name(id_col))
            mac_c = complex_mac(v_id, v_of)
            mac_r = real_projected_mac(v_id, v_of)
            rows.append({
                'target': name,
                'of_column': of_col,
                f'{method_label}_column': id_col,
                'freq_diff_hz': f_diff,
                'complex_mac': mac_c,
                'real_projected_mac': mac_r,
            })
            print(f"  {name:<20} OF={of_col[:30]} | {method_label}={id_col[:30]} | df={f_diff:.3f} Hz | MACc={mac_c:.4f} | MACr={mac_r:.4f}")
        else:
            rows.append({'target': name, 'of_column': of_col, f'{method_label}_column': id_col,
                         'freq_diff_hz': None, 'complex_mac': None, 'real_projected_mac': None})
            print(f"  {name:<20} NO MATCH (of_col={of_col}, id_col={id_col})")

    summary = pd.DataFrame(rows)

    if sel_of and sel_id:
        mac_mat = pd.DataFrame(index=sel_of, columns=sel_id, dtype=float)
        for r in sel_of:
            for c in sel_id:
                mac_mat.loc[r, c] = real_projected_mac(df_id[c].values, df_of[r].values)
        labels_y = [f"OF {extract_freq_from_name(x):.3f} Hz" for x in mac_mat.index]
        labels_x = [f"{method_label} {extract_freq_from_name(x):.3f} Hz" for x in mac_mat.columns]
        mac_mat_plot = mac_mat.copy()
        mac_mat_plot.index = labels_y
        mac_mat_plot.columns = labels_x
        plot_mac_heatmap(
            mac_mat_plot,
            title=f"{direction} Mode Shape Validation ({method_label})",
            save_path=fig_dir / f"{direction.lower().replace('-', '_')}_mac_{method_label.lower()}.png",
        )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/mac_validation.yaml')
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    fig_dir = ensure_dir(cfg.get('outputs', {}).get('figures_dir', 'results/figures'))
    tab_dir = ensure_dir(cfg.get('outputs', {}).get('tables_dir', 'results/tables'))
    tolerance = cfg.get('tolerance_hz', 0.9)

    df_of = load_openfast_modes(Path(cfg['openfast_modes_xlsx']),
                                sheet=cfg.get('openfast_sheet', 'Tower_Modes'))

    fa_targets = cfg['fa_targets']
    ss_targets = cfg['ss_targets']

    # DMD branch
    if cfg.get('include_dmd', True):
        df_fa = load_complex_csv(Path(cfg['fa_dmd_csv']))
        df_ss = load_complex_csv(Path(cfg['ss_dmd_csv']))
        s_fa = analyze_direction('Fore-Aft', df_of, df_fa, fa_targets, tolerance, 'DMD', fig_dir)
        s_ss = analyze_direction('Side-to-Side', df_of, df_ss, ss_targets, tolerance, 'DMD', fig_dir)
        s_fa.to_csv(tab_dir / 'fore_aft_mac_dmd.csv', index=False)
        s_ss.to_csv(tab_dir / 'side_to_side_mac_dmd.csv', index=False)

    # SSI branch
    if cfg.get('include_ssi', True):
        df_fa = load_complex_csv(Path(cfg['fa_ssi_csv']))
        df_ss = load_complex_csv(Path(cfg['ss_ssi_csv']))
        s_fa = analyze_direction('Fore-Aft', df_of, df_fa, fa_targets, tolerance, 'SSI', fig_dir)
        s_ss = analyze_direction('Side-to-Side', df_of, df_ss, ss_targets, tolerance, 'SSI', fig_dir)
        s_fa.to_csv(tab_dir / 'fore_aft_mac_ssi.csv', index=False)
        s_ss.to_csv(tab_dir / 'side_to_side_mac_ssi.csv', index=False)


if __name__ == '__main__':
    main()
