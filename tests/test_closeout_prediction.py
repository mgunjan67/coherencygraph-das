"""Independent prediction-ensemble aggregation checks without model imports."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("closeout_prediction_aggregation",ROOT/"scripts/closeout_prediction_aggregation.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_predictions_are_averaged_before_nonlinear_score():
    target=np.ones((2,3),dtype=np.complex64)
    predictions=np.stack([target-1,target,target+1])
    ensemble=module.score(predictions.mean(0),target)["nrmse"]
    mean_scores=np.mean([module.score(p,target)["nrmse"] for p in predictions])
    assert ensemble==0
    assert abs(mean_scores-2/3)<1e-12


def test_independent_component_bootstrap_uses_event_counts():
    values=np.array([1.,3.,5.,7.]);labels=np.array(["a","a","a","b"])
    out=module.independent_bootstrap(values,labels)
    assert out[0]==values.mean()
    assert out[1]<=out[0]<=out[2]
    assert out[0]!=(values[:3].mean()+values[3])/2


def test_saved_ensemble_arrays_and_all_named_comparisons_verified():
    directory=ROOT/"reports/submission_closeout/prediction"
    if not (directory/"verification.json").exists():
        pytest.skip("Closeout prediction verification not included/generated")
    data=json.loads((directory/"verification.json").read_text())
    assert data["status"]=="PASS"
    assert data["seed_mean_arrays_bitwise_equal"]==4
    assert data["route_model_scores_reproduced"]==1152
    assert data["distinct_ensemble_vs_mean_seed_scores"]==192
    assert data["named_ridge_comparisons_reproduced"]==8
    assert data["maximum_case_score_error"]<1e-12
    assert data["maximum_comparison_error"]<1e-12
    summary=pd.read_csv(directory/"ensemble_summary.csv")
    assert summary.seed_prediction_mean_bitwise_equal.all()
    assert summary.all_route_scores_distinguish_ensemble_from_mean_seed_score.all()
