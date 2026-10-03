"""Independent direct-array check of the fixed-window development extraction."""
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.signal.windows import dpss

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from coherencygraph_das.causal_validation import sha256
OUT=ROOT/'revisions/20260922_unseen_validation'
table=pd.read_csv(OUT/'development_records.csv',dtype={'event_id':str})
row=table.iloc[0]
cache=OUT/f'development/cache/{row.event_id}_{row.route.replace("-","")}.npz'
with h5py.File(row.raw_path,'r') as h:
    raw=h['Acquisition/Raw[0]/RawData']
    # cutoff20 context6..20, first block200..1199, local684..715
    context=np.asarray(raw[150:500,200:1200],dtype=float)
    target=np.asarray(raw[525:700,684:716],dtype=float)

def transform(data):
    clean=detrend(data,axis=0,type='linear')
    taper=dpss(len(data),2.5,3,sym=False)
    return np.fft.rfft(taper[:,:,None]*clean[None],axis=1),np.fft.rfftfreq(len(data),1/25)

fft,f=transform(context)
x=fft[:,(f>=1)&(f<2)].reshape(-1,1000)
cross=np.mean(x[:,:-3]*x[:,3:].conj(),axis=0)
power=np.mean(abs(x)**2,axis=0)
g=np.mean(cross/np.sqrt(power[:-3]*power[3:]))
fft,f=transform(target)
x=fft[:,(f>=1)&(f<2)].reshape(-1,32)
matrix=np.zeros((32,32),complex)
for snapshot in x:matrix+=np.outer(snapshot,snapshot.conj())
matrix/=len(x)
matrix+=.001*np.trace(matrix).real/32*np.eye(32)
with np.load(cache) as z:
    got=z['target'][0,0,1]
    feature=z['features'][0,0]
    gamma=feature[8]+1j*feature[40]
    matrix_error=float(np.max(abs(got-matrix))/max(np.max(abs(matrix)),1e-30))
    gamma_error=float(abs(gamma-g))
    assert z['features'].shape==(4,8,135)
    assert matrix_error<1e-6 and gamma_error<1e-6
    raw_hash=str(z['raw_sha256'])
record=dict(event_id=row.event_id,route=row.route,raw_sha256=raw_hash,
            cache_sha256=sha256(cache),direct_matrix_relative_max_error=matrix_error,
            direct_feature_complex_error=gamma_error,passed=True,
            implementation='Independent raw-array FFT, explicit outer-product loop and pair normalisation')
(OUT/'development/direct_array_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
