from pathlib import Path
import argparse
import pandas as pd

from eftwin.config import load_yaml, ensure_dir
from eftwin.constants import analysis_columns
from eftwin.data_io import discover_case_files
from eftwin.hankel_dmd import run_hankel_dmd
from eftwin.modal_analysis import extract_physical_modes
from eftwin.plotting import plot_eigs_custom, plot_mode_shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/hankel_dmd.yaml')
    parser.add_argument('--data-dir', default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    data_dir = Path(args.data_dir or cfg['data_dir'])
    case_files = discover_case_files(data_dir, cfg.get('case_pattern', 'Case_*.csv'))
    if not case_files:
        raise FileNotFoundError(f'No case files found in {data_dir}')
    dt = cfg.get('dt', 0.0125)
    filt = cfg.get('filter', {})
    hankel = cfg.get('hankel', {})
    fig_dir = ensure_dir(cfg.get('outputs', {}).get('figures_dir', 'results/figures'))
    tab_dir = ensure_dir(cfg.get('outputs', {}).get('tables_dir', 'results/tables'))

    for direction in cfg.get('directions', ['fore_aft', 'side_to_side']):
        acc_cols, mom_cols = analysis_columns(direction)
        name = 'Fore-Aft' if direction == 'fore_aft' else 'Side-to-Side'
        result = run_hankel_dmd(
            case_files, acc_cols, mom_cols, direction_name=name, dt=dt,
            low_cutoff=filt.get('low_cutoff_hz', 0.25), high_cutoff=filt.get('high_cutoff_hz', 5.0), filter_order=filt.get('order', 4),
            hankel_d=hankel.get('delay_d', 60), svd_rank=hankel.get('svd_rank', 24), tlsq_rank=hankel.get('tlsq_rank', 0), exact=hankel.get('exact', True), opt=hankel.get('opt', False)
        )
        low_f, high_f = cfg.get('frequency_window_hz', [0.3, 3.0])
        modes, stats = extract_physical_modes(result, dt=dt, low_f=low_f, high_f=high_f)
        modes.to_csv(tab_dir / f'{direction}_dmd_modes.csv')
        stats.to_csv(tab_dir / f'{direction}_dmd_stats.csv', index=False)
        plot_eigs_custom(result, dt=dt, save_path=fig_dir / f'{direction}_eigenvalues.png')
        plot_mode_shapes(result, dt=dt, low_f=low_f, high_f=high_f, save_path=fig_dir / f'{direction}_mode_shapes.png')
        print(f'Completed {name}.')

if __name__ == '__main__':
    main()
