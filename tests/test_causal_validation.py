from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from coherencygraph_das.causal_validation import (
    CUTOFFS, FS, LOCAL, SPARSE, DENSE, Q, sample_slice, window_features,
    upper_kernel, candidate_order, hypocentral_distances, source_components,
    storage_guard, target_matrices,
)
from coherencygraph_das.processor_audit import moments, correlation, remove_loading
from coherencygraph_das.spectral import multitaper_fourier, lag_coherency, band_snapshots


def test_no_context_target_overlap_and_exact_reference():
    for cutoff in CUTOFFS:
        context=sample_slice(cutoff-14,cutoff,2200)
        target=sample_slice(cutoff+1,cutoff+8,2200)
        assert context.stop <= target.start-25
        assert context.stop == int(cutoff*FS)
        assert target.stop-target.start == 175
    assert sample_slice(.5,5.5,2200) == slice(13,138)


@pytest.mark.parametrize('start,stop,n', [(-1,10,500),(10,90,2200),(1,1.1,500)])
def test_incomplete_windows_fail(start,stop,n):
    with pytest.raises(ValueError):sample_slice(start,stop,n)


def test_features_are_past_only_and_match_direct_fft():
    rng=np.random.default_rng(1)
    raw=rng.normal(size=(900,1000))
    context=sample_slice(6,20,len(raw));reference=sample_slice(.5,5.5,len(raw))
    first,_,_=window_features(raw[context].copy(),raw[reference].copy(),0,'TERRA',23.93)
    raw[500:]=rng.normal(size=raw[500:].shape)*1e6
    second,_,_=window_features(raw[context].copy(),raw[reference].copy(),0,'TERRA',23.93)
    np.testing.assert_array_equal(first,second)
    fft,f=multitaper_fourier(raw[context].copy(),25,2.5,3)
    g=lag_coherency(band_snapshots(fft,f,(.5,1)),[3,5,8,13,21,34,55,89])
    np.testing.assert_allclose(first[:8],g.real)
    np.testing.assert_allclose(first[32:40],g.imag)
    assert first.shape==(135,)


def test_target_support_and_complex_orientation():
    index=180;p=np.zeros(257);p[index]=1
    kernel=upper_kernel(p)
    assert np.linalg.eigvalsh(kernel).min()>-1e-12
    np.testing.assert_allclose(moments(kernel,DENSE),np.exp(1j*DENSE*Q[index]),atol=1e-12)
    np.testing.assert_allclose(moments(kernel,SPARSE),moments(kernel,DENSE)[SPARSE-1])


def test_target_halves_use_exact_local_channels():
    rng=np.random.default_rng(2);raw=rng.normal(size=(175,1000))
    full,one,two=target_matrices(raw)
    assert full.shape==one.shape==two.shape==(4,32,32)
    for start,stop,matrix in [(0,87,one),(87,175,two)]:
        fft,f=multitaper_fourier(raw[start:stop,LOCAL],25,2.5,3)
        x=band_snapshots(fft,f,(1,2));direct=x.T@x.conj()/len(x)
        np.testing.assert_allclose(remove_loading(matrix[1]),direct,rtol=1e-6,atol=1e-6)


def test_source_components_are_transitive():
    frame=pd.DataFrame(dict(latitude_deg=[0,0,0,10],longitude_deg=[0,.1,.2,0],depth_km=[0]*4))
    labels=source_components(frame,15)
    assert labels[0]==labels[1]==labels[2] and labels[3]!=labels[0]


def test_candidate_order_exclusion_buffer_and_shuffle_invariance():
    root=Path(__file__).resolve().parents[1]
    catalog=pd.read_parquet(root/'revisions/20260922_unseen_validation/availability/live_catalog.parquet')
    old=pd.read_parquet(root/'data/provenance/paired_event_manifest.parquet')
    _,a=candidate_order(catalog,old)
    _,b=candidate_order(catalog.sample(frac=1,random_state=3),old.sample(frac=1,random_state=4))
    assert a.event_id.tolist()==b.event_id.tolist()
    assert len(a)>=30 and not set(a.event_id.astype(str)) & set(old.event_id.astype(str))
    assert hypocentral_distances(a,old).min()>=25
    distance=hypocentral_distances(a,a)+np.eye(len(a))*1e9
    assert distance.min()>=25


def test_storage_guard(monkeypatch,tmp_path):
    import shutil
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=110*1024**3))
    storage_guard(tmp_path,1*1024**3,0)
    with pytest.raises(RuntimeError):storage_guard(tmp_path,1*1024**3,8*1024**3)
    with pytest.raises(RuntimeError):storage_guard(tmp_path,11*1024**3,0,maximum=20*1024**3)


def test_recurrence_simplex_and_shapes():
    import torch
    from coherencygraph_das.causal_validation_model import CausalSpectralModel
    torch.set_num_threads(1)
    for lags in [SPARSE,DENSE]:
        m=CausalSpectralModel(lags).eval()
        with torch.no_grad():y,p=m(torch.zeros(2,8,135))
        assert y.shape==(2,8,4,len(lags),2)
        np.testing.assert_allclose(p.sum(-1).numpy(),1,atol=1e-6)
        assert (p>=0).all()


def runner_module():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/run_causal_validation.py'
    spec=importlib.util.spec_from_file_location('causal_validation_runner',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_complete_event_metric_not_window_weighting():
    module=runner_module()
    y=np.zeros((6,1,1,1,2));p=y.copy();p[0]=1
    assert module.event_mse(p,y,np.array(['a','b','b','b','b','b']))==1.0


def test_scaler_never_uses_validation_features():
    module=runner_module();rng=np.random.default_rng(6)
    x=rng.normal(size=(6,8,135)).astype('float32')
    _,mean1,std1=module.scale(x,np.arange(3))
    x[3:]+=1e6
    _,mean2,std2=module.scale(x,np.arange(3))
    np.testing.assert_array_equal(mean1,mean2)
    np.testing.assert_array_equal(std1,std2)
