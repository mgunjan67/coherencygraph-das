"""Re-evaluate matched checkpoints on CPU without reusing cached predictions."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
from coherencygraph_das.critical_revision import bundle,MODELS,OUT,components
from coherencygraph_das.critical_experiments import MatchedModel,predict_model
_,ds,roles,_=bundle();records=[]
for kind in ['full_mlp_direct','full_mlp_psd','state_direct','state_psd']:
    ensemble=[]
    for seed in [19,43,71]:
        saved=torch.load(MODELS/f'matched_{kind}_seed{seed}.pt',map_location='cpu',weights_only=True)
        model=MatchedModel(137,kind);model.load_state_dict(saved['state']);model.eval()
        pred,_=predict_model(model,ds.features,saved['mean'].numpy(),saved['std'].numpy());ensemble.append(pred)
        old=np.load(MODELS/f'matched_{kind}_seed{seed}_all_predictions.npy');difference=float(np.max(abs(pred-old)))
        records.append(dict(model=kind,seed=seed,max_prediction_difference=difference,tolerance=2e-5,passed=difference<2e-5))
    old=np.load(MODELS/f'matched_{kind}_all_predictions.npy');difference=float(np.max(abs(np.mean(ensemble,0)-old)))
    records.append(dict(model=kind,seed='ensemble',max_prediction_difference=difference,tolerance=2e-5,passed=difference<2e-5))
for kind in ['block_ridge','full_ridge','residual_ridge']:
    s=np.load(MODELS/f'matched_{kind}_parameters.npz');x=(ds.features-s['mean'])/s['std']
    x=x.reshape(-1,137) if kind=='block_ridge' else x.reshape(len(x),-1)
    prediction=(x@s['coef'].T+s['intercept']).reshape(ds.targets.shape)
    if kind=='residual_ridge':prediction+=np.stack([components(np.load(p)['context_gamma']) for p in ds.paths])
    difference=float(np.max(abs(prediction-np.load(MODELS/f'matched_{kind}_all_predictions.npy'))))
    records.append(dict(model=kind,seed='classical',max_prediction_difference=difference,tolerance=2e-5,passed=difference<2e-5))
(OUT/'checkpoint_regeneration.json').write_text(json.dumps(records,indent=2))
assert all(r['passed'] for r in records)
print('CPU checkpoint regeneration:',len(records),'checks; maximum difference',max(r['max_prediction_difference'] for r in records))
