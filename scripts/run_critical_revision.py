from pathlib import Path
import argparse
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage')
    args=parser.parse_args()
    if args.stage in ['baseline','identifiability','uncertainty']:
        from coherencygraph_das.critical_revision import main
        main(args.stage)
    elif args.stage=='raw':
        from coherencygraph_das.critical_raw import run_raw_audit
        run_raw_audit()
    elif args.stage in ['train','sensitivity','internal','missingness','downstream','projection','summary','figures','release','verify']:
        from coherencygraph_das.critical_experiments import main
        main(args.stage)
    else:
        raise SystemExit(f'Unknown stage {args.stage}')
