"""Resumable Amendment-09 runner; prior experiment directories are read-only."""
from pathlib import Path
import os
import sys
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
if __name__ == '__main__':
    from coherencygraph_das import submission_revision
    getattr(submission_revision, sys.argv[1])()
