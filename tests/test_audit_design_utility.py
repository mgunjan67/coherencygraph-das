from pathlib import Path
import importlib.util,itertools,json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('design_utility',ROOT/'scripts/evaluate_audit_design_utility.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_geometry_rules_and_numeric_lexicographic_tie():
    candidates=list(itertools.combinations([1,2,3,5,8,13,21,34,55,89],8))
    assert len(candidates)==len(set(candidates))==45
    assert all(len(c)==8 and not set(c)&{4,10} for c in candidates)
    assert all(module.definitions(c)['H0_direct_count']==0 for c in candidates)
    for field in ['H1_mean_nearest_distance','H2_max_nearest_distance']:
        assert min(candidates,key=lambda c:(module.definitions(c)[field],c))==(1,2,3,5,8,13,21,34)
    assert module.definitions(candidates[0])==dict(H0_direct_count=0,H1_mean_nearest_distance=1.5,H2_max_nearest_distance=2,H3_max_lag=34)

def test_equal_route_then_earthquake_aggregation():
    frame=pd.DataFrame([('a','TERRA',0.),('a','TERRA',2.),('a','KKFL-S',3.),('b','TERRA',10.),('b','KKFL-S',10.)],columns=['event_id','route','width'])
    assert module.event_average(frame,'width')==6.

def test_frozen_specification_before_evaluation():
    path=ROOT/'reports/audit_design_utility/frozen_receipt.json'
    if not path.exists():return
    f=json.loads(path.read_text())
    assert module.sha(ROOT/'configs/audit_design_utility.json')==f['specification_sha256']
    assert f['candidate_count']==45 and f['evaluation_lags']==[4,10]
    assert f['heuristic_selected']=={'H1':'D01','H2':'D01'}
    assert not f['retrospective_ranking_calculated']
