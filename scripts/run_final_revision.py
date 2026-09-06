from pathlib import Path
import os, sys
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
if __name__ == '__main__':
    from coherencygraph_das.final_revision import main
    main(sys.argv[1])
