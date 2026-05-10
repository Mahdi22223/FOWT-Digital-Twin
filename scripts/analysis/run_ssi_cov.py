from pathlib import Path
import argparse

from eftwin.config import load_yaml, ensure_dir
from eftwin.constants import analysis_columns
from eftwin.data_io import discover_case_files
from eftwin.ssi_cov import run_ssi_validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/ssi_cov.yaml')
    parser.add_argument('--direction', default='fore_aft', choices=['fore_aft','side_to_side'])
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    files = discover_case_files(cfg['data_dir'], cfg.get('case_pattern','Case_*.csv'))
    filt = cfg.get('filter', {})
    ssi = cfg.get('ssi_cov', {})
    acc_cols, mom_cols = analysis_columns(args.direction)
    modes, stats, _ = run_ssi_validation(
        files, acc_cols, mom_cols, dt=cfg.get('dt',0.0125), low_cutoff=filt.get('low_cutoff_hz',0.25), high_cutoff=filt.get('high_cutoff_hz',5.0), filter_order=filt.get('order',4),
        block_rows=ssi.get('block_rows',60), model_order=ssi.get('model_order',40), downsample_factor=ssi.get('downsample_factor',4), min_frequency_hz=ssi.get('min_frequency_hz',0.3), max_frequency_hz=ssi.get('max_frequency_hz',5.0), max_damping_ratio=ssi.get('max_damping_ratio',0.2))
    out = ensure_dir(cfg.get('outputs',{}).get('tables_dir','results/tables'))
    modes.to_csv(out / f'{args.direction}_ssi_modes.csv')
    stats.to_csv(out / f'{args.direction}_ssi_stats.csv', index=False)
    print(stats)

if __name__ == '__main__':
    main()
