"""Run the Hankel-delay probabilistic mode-shape sweep for both directions.

Reproduces the workflow of the original ``Probablistic_Mode_shape 2.py``:
sweep ``d``, find the mode closest to a target frequency at each d, collect
the normalized real-valued shapes from the convergence zone, and compute
mean +/- 95% confidence intervals across that zone.
"""
from pathlib import Path
import argparse

from eftwin.config import load_yaml, ensure_dir
from eftwin.constants import analysis_columns, TOWER_HEIGHTS_9
from eftwin.data_io import discover_case_files
from eftwin.probabilistic_modes import run_probabilistic_mode_sweep
from eftwin.plotting import plot_delay_sensitivity, plot_probabilistic_mode_shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/probabilistic_modes.yaml')
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    case_files = discover_case_files(cfg['data_dir'], cfg.get('case_pattern', 'Case_*.csv'))
    if not case_files:
        raise FileNotFoundError(f"No case files in {cfg['data_dir']}")

    fig_dir = ensure_dir(cfg.get('outputs', {}).get('figures_dir', 'results/figures'))
    tab_dir = ensure_dir(cfg.get('outputs', {}).get('tables_dir', 'results/tables'))

    for direction_cfg in cfg['directions']:
        direction = direction_cfg['name']
        target_freq = direction_cfg['target_freq']
        display_name = 'Fore-Aft' if direction == 'fore_aft' else 'Side-to-Side'
        acc_cols, mom_cols = analysis_columns(direction)

        result = run_probabilistic_mode_sweep(
            case_files,
            acc_cols,
            mom_cols,
            target_freq=target_freq,
            direction_name=display_name,
            d_range=cfg.get('d_range', [10, 20, 30, 35, 40, 45, 50, 55, 57, 63, 65, 70, 75, 80, 90]),
            convergence_d_min=cfg.get('convergence_d_min', 45),
            convergence_d_max=cfg.get('convergence_d_max', 75),
            dt=cfg.get('dt', 0.0125),
            low_cutoff=cfg.get('filter', {}).get('low_cutoff_hz', 0.25),
            high_cutoff=cfg.get('filter', {}).get('high_cutoff_hz', 5.0),
            filter_order=cfg.get('filter', {}).get('order', 4),
            svd_rank=cfg.get('svd_rank', 35),
            search_tol=cfg.get('search_tol', 0.1),
        )

        result['sensitivity'].to_csv(tab_dir / f'{direction}_delay_sensitivity.csv', index=False)
        plot_delay_sensitivity(
            result['sensitivity'],
            title=f"{display_name} (Target {target_freq:.2f} Hz): Parameter Convergence",
            convergence_window=result['convergence_window'],
            selected_d=cfg.get('selected_d', 60),
            save_path=fig_dir / f'{direction}_delay_sensitivity.png',
        )
        plot_probabilistic_mode_shape(
            result,
            tower_heights=TOWER_HEIGHTS_9,
            save_path=fig_dir / f'{direction}_probabilistic_mode_shape.png',
        )
        print(f"Completed probabilistic sweep for {display_name}: target {target_freq} Hz, "
              f"{len(result['sensitivity'])} valid d-points, "
              f"{result['shapes'].shape[0]} shapes in convergence zone.")


if __name__ == '__main__':
    main()
