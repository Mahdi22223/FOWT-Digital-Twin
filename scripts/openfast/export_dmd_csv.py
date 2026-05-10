import argparse
from eftwin.openfast_pipeline import export_dmd_csv_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', default='outputs/parametric')
    p.add_argument('--output-dir', default='data/full')
    p.add_argument('--n-cases', type=int, default=12)
    args = p.parse_args()
    export_dmd_csv_batch(args.input_dir, args.output_dir, args.n_cases)
    print(f'Exported {args.n_cases} Case_*.csv files to {args.output_dir}')

if __name__ == '__main__':
    main()
