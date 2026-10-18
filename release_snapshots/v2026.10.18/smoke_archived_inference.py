"""One cached statistical-feature inference check; no waveforms or retraining."""
from pathlib import Path
import argparse,hashlib,json,platform,sys,time
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();root=a.package_root.resolve();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    receipt=out/'INFERENCE_SMOKE_RECEIPT.json'
    if receipt.exists():raise FileExistsError('Preserve existing inference evidence')
    sys.path.insert(0,str(root/'src'))
    import torch
    from coherencygraph_das.causal_validation_model import CausalSpectralModel
    from coherencygraph_das.causal_validation import Q
    torch.set_num_threads(2);started=time.perf_counter()
    with np.load(root/'verification/inference_fixture.npz') as data:
        features=np.array(data['features']);expected=np.array(data['expected_kernel'])
    assert features.shape==(1,8,135) and expected.shape==(1,8,4,32,32)
    modes=np.asarray(Q);difference=np.arange(32)[None,:]-np.arange(32)[:,None]
    basis=np.exp(1j*difference[...,None]*modes)
    outputs=[];checks=[]
    for seed in (19,43,71):
        path=root/f'revisions/20261003_scientific_revision/models/common_localinput_seed{seed}.pt'
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=CausalSpectralModel(saved['lags'].numpy()).eval();model.load_state_dict(saved['state'],strict=True)
        mean,std=saved['mean'].numpy(),saved['std'].numpy()
        assert np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all()
        for flag in (0,1):
            x=features.copy();x[...,-2]=flag
            with torch.no_grad(): _,prob=model(torch.tensor((x-mean)/std,dtype=torch.float32))
            probabilities=prob.numpy()
            assert np.isfinite(probabilities).all() and probabilities.min()>=0
            np.testing.assert_allclose(probabilities.sum(-1),1,rtol=0,atol=3e-7)
            probabilities=np.maximum(probabilities.astype('float64'),0)
            probabilities/=probabilities.sum(-1,keepdims=True)
            kernel=np.einsum('nxbq,ijq->nxbij',probabilities,basis)
            outputs.append((kernel+kernel.conj().swapaxes(-1,-2))/2)
        checks.append(dict(seed=seed,checkpoint=path.relative_to(root).as_posix(),sha256=sha(path),
            normalization_shape=list(mean.shape),state_parameters=sum(t.numel() for t in model.parameters())))
    actual=np.mean(outputs,axis=0)
    difference=float(np.max(np.abs(actual-expected)))
    if not np.isfinite(difference) or difference>2e-6:raise AssertionError(f'Archived inference changed: {difference}')
    hermitian=float(np.max(abs(actual-actual.conj().swapaxes(-1,-2))))
    diagonal=float(np.max(abs(np.diagonal(actual,axis1=-2,axis2=-1)-1)))
    eigen=float(np.linalg.eigvalsh(actual).min())
    assert hermitian<1e-10 and diagonal<3e-7 and eigen>-1e-7
    report=dict(status='PASS_ARCHIVED_THREE_SEED_TWO_ROUTE_ENSEMBLE_SMOKE',features_shape=list(features.shape),
        output_shape=list(actual.shape),fixture_sha256=sha(root/'verification/inference_fixture.npz'),
        checkpoint_checks=checks,maximum_absolute_kernel_error=difference,tolerance=2e-6,
        maximum_hermitian_error=hermitian,maximum_diagonal_error=diagonal,minimum_eigenvalue=eigen,
        elapsed_seconds=time.perf_counter()-started,python=sys.version,platform=platform.platform(),
        numpy=np.__version__,torch=torch.__version__,waveforms_read=False,training=False,
        new_evaluation=False,independent_human_replication=False,
        limitation='One saved statistical feature case from exposed development; not complete training or raw extraction regeneration.')
    receipt.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
