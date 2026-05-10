import argparse
from eftwin.config import load_yaml
from eftwin.openfast_pipeline import build_nonlinear_params, generate_cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/openfast_cases.yaml')
    p.add_argument('--run', action='store_true', help='Run OpenFAST after generating cases.')
    args = p.parse_args()
    cfg = load_yaml(args.config)
    params = build_nonlinear_params(cfg)
    files = generate_cases(params, cfg['nonlinear_template_dir'], cfg['nonlinear_output_dir'], cfg.get('main_file','main.fst'), cfg.get('openfast_exe'), run=args.run, n_cores=cfg.get('n_cores',4))
    print('Generated files:')
    for f in files:
        print(f)

if __name__ == '__main__':
    main()
