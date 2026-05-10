from pathlib import Path
import argparse

from eftwin.config import load_yaml, ensure_dir
from eftwin.constants import analysis_columns
from eftwin.data_io import discover_case_files
from eftwin.hankel_dmd import run_hankel_dmd
from eftwin.virtual_sensing import run_virtual_sensing
from eftwin.plotting import plot_virtual_sensing_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/virtual_sensing.yaml')
    parser.add_argument('--direction', default='fore_aft', choices=['fore_aft', 'side_to_side'])
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    case_files = discover_case_files(cfg['data_dir'], cfg.get('case_pattern', 'Case_*.csv'))
    dt = cfg.get('dt', 0.0125)
    filt = cfg.get('filter', {})
    hankel = cfg.get('hankel', {})
    acc_cols, mom_cols = analysis_columns(args.direction)
    name = 'Fore-Aft' if args.direction == 'fore_aft' else 'Side-to-Side'
    result = run_hankel_dmd(
        case_files, acc_cols, mom_cols, direction_name=name, dt=dt,
        low_cutoff=filt.get('low_cutoff_hz', 0.25), high_cutoff=filt.get('high_cutoff_hz', 5.0), filter_order=filt.get('order', 4),
        hankel_d=hankel.get('delay_d', 60), svd_rank=hankel.get('svd_rank', 34), tlsq_rank=hankel.get('tlsq_rank', 0), exact=hankel.get('exact', True), opt=hankel.get('opt', False)
    )
    vs = run_virtual_sensing(result, missing_sensors=cfg.get('missing_sensors', [0, 8]), update_every=cfg.get('update_every_seconds', 1.0), duration=cfg.get('duration_seconds', 50), dt=dt)
    tab_dir = ensure_dir(cfg.get('outputs', {}).get('tables_dir', 'results/tables'))
    fig_dir = ensure_dir(cfg.get('outputs', {}).get('figures_dir', 'results/figures'))
    vs['metrics'].to_csv(tab_dir / f'{args.direction}_virtual_sensing_metrics.csv', index=False)
    print(vs['metrics'])
    for idx in cfg.get('missing_sensors', [0, 8]):
        plot_virtual_sensing_result(vs, idx, plot_duration=cfg.get('duration_seconds', 50), save_path=fig_dir / f'{args.direction}_virtual_sensor_{idx}.png')

if __name__ == '__main__':
    main()
