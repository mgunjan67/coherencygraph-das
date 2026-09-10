"""Regression guards for the bounded projection/processing closeout."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("closeout_matched_processing",ROOT/"scripts/closeout_matched_processing.py")
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)


def test_processing_simplex_and_first_order_acceptance():
    A=np.eye(3);p=np.array([.2,.3,.5]);z=p.copy()
    assert module.acceptance(p,A,z)["accepted"]
    assert not module.acceptance(p*.99,A,z)["accepted"]
    assert not module.acceptance(np.array([-.1,.4,.7]),A,z)["accepted"]
    assert not module.acceptance(np.array([1.,0.,0.]),A,z)["accepted"]


def test_seed_average_is_within_earthquake_not_independent_replicates():
    rows=[]
    for event in ("e1","e2"):
        for seed,base in ((19,1.),(43,2.),(71,6.)):
            for route in ("TERRA","KKFL-S"):
                for block in range(8):
                    for band in range(4):
                        for endpoint,offset in (("full",0.),("half0",2.),("half1",4.)):
                            rows.append(dict(event_id=event,seed=seed,route=route,block=block,band=band,
                                support="local_dense",model="recurrence",endpoint=endpoint,ratio_db=base+offset))
    per_seed,event=module.aggregate(pd.DataFrame(rows))
    assert len(per_seed)==12
    assert len(event)==4
    np.testing.assert_allclose(event.loc[event.evaluation=="full","ratio_db"],3.)
    np.testing.assert_allclose(event.loc[event.evaluation=="split_target","ratio_db"],6.)


def test_paired_mean_and_size_weighted_component_deletion_identity():
    ids=[str(i) for i in range(6)]
    a=np.array([0.,1.,2.,20.,30.,50.]);b=np.array([2.,1.,3.,5.,7.,9.])
    event=pd.DataFrame([dict(evaluation="full",event_id=e,method=m,ratio_db=v)
                        for m,vs in (("A",a),("B",b)) for e,v in zip(ids,vs)])
    mapping=dict(zip(ids,["g0","g0","g0","g1","g1","g2"]))
    comp,deletion,checks=module.comparison_frames(event,mapping)
    c=comp.loc[(comp.method=="A")&(comp.reference=="B")].iloc[0]
    assert c.events==6 and c.groups==3
    assert abs(c["mean"]-(a.mean()-b.mean()))<1e-12
    d=deletion.loc[(deletion.method=="A")&(deletion.reference=="B")]
    assert abs(np.average(d["mean"],weights=d.retained_events)-c["mean"])<1e-12
    assert abs(d["mean"].mean()-c["mean"])>.01  # Unequal-size groups require the stated weights.
    assert checks.filter(like="minus").abs().max().max()<1e-12


def test_saved_processing_projections_must_all_pass():
    path=ROOT/"reports/submission_closeout/processing/projection_acceptance.csv"
    if not path.exists():
        import pytest
        pytest.skip("Closeout projection output not included/generated in this checkout")
    frame=pd.read_csv(path)
    assert len(frame)==9216
    assert frame.after_accepted.all()
    assert (frame.after_first_order_gap<1e-8).all()
    assert (frame.after_simplex_sum_error<=1e-10).all()
    assert (frame.after_minimum_mass>=-1e-12).all()
    assert (frame.after_squared_objective<=frame.before_squared_objective+1e-12).all()


def test_saved_seed_summaries_reproduce_event_scores():
    directory=ROOT/"reports/submission_closeout/processing"
    if not (directory/"processing_event_scores.csv").exists():
        import pytest
        pytest.skip("Closeout processing output not included/generated in this checkout")
    seed=pd.read_csv(directory/"processing_event_seed_scores.csv",dtype={"event_id":str})
    event=pd.read_csv(directory/"processing_event_scores.csv",dtype={"event_id":str})
    keys=["evaluation","event_id","method"]
    reconstructed=seed.groupby(keys).ratio_db.mean().reset_index()
    check=event.merge(reconstructed,on=keys,suffixes=("_saved","_reconstructed"),validate="one_to_one")
    assert len(seed)==960 and len(check)==576
    np.testing.assert_allclose(check.ratio_db_saved,check.ratio_db_reconstructed,rtol=0,atol=1e-12)
    recurrence=seed.loc[seed.method.str.endswith(" / recurrence")]
    assert recurrence.groupby(keys).seed.apply(lambda s:set(s)=={19,43,71}).all()
