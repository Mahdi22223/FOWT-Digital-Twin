import argparse
from eftwin.config import load_yaml
from eftwin.openfast_pipeline import build_linear_params, generate_cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/openfast_cases.yaml')
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    cfg = load_yaml(args.config)
    params = build_linear_params(cfg)
    files = generate_cases(params, cfg['linear_template_dir'], cfg['linear_output_dir'], cfg.get('main_file','main.fst'), cfg.get('openfast_exe'), run=args.run, n_cores=cfg.get('n_cores',4))
    print('Generated linearization files:')
    for f in files:
        print(f)

if __name__ == '__main__':
    main()
