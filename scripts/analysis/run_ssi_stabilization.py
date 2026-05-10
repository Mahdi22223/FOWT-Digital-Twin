import argparse
from eftwin.config import load_yaml, ensure_dir
from eftwin.constants import analysis_columns
from eftwin.data_io import discover_case_files
from eftwin.ssi_cov import run_stabilization
from eftwin.plotting import plot_stabilization_diagram


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/ssi_cov.yaml')
    p.add_argument('--direction', default='fore_aft', choices=['fore_aft','side_to_side'])
    args = p.parse_args()
    cfg = load_yaml(args.config)
    files = discover_case_files(cfg['data_dir'], cfg.get('case_pattern','Case_*.csv'))
    filt = cfg.get('filter', {})
    ssi = cfg.get('ssi_cov', {})
    stab = cfg.get('stabilization', {})
    acc_cols, mom_cols = analysis_columns(args.direction)
    poles = run_stabilization(files, acc_cols, mom_cols, dt=cfg.get('dt',0.0125), low_cutoff=filt.get('low_cutoff_hz',0.25), high_cutoff=filt.get('high_cutoff_hz',5.0), filter_order=filt.get('order',4), block_rows=ssi.get('block_rows',60), downsample_factor=ssi.get('downsample_factor',4), min_order=stab.get('min_order',2), max_order=stab.get('max_order',60), step_order=stab.get('step_order',2), tol_freq=stab.get('tol_freq',0.01), tol_damp=stab.get('tol_damp',0.05), tol_mac=stab.get('tol_mac',0.98))
    fig_dir = ensure_dir(cfg.get('outputs',{}).get('figures_dir','results/figures'))
    plot_stabilization_diagram(poles, title=f'SSI Stabilization: {args.direction}', save_path=fig_dir / f'{args.direction}_ssi_stabilization.png')

if __name__ == '__main__':
    main()
