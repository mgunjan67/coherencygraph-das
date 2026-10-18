import numpy as np
import pytest

from derivative_lcmv import derivative_frequency_weights
from recorded_component_metrics import evaluate_recorded_component


def fixture():
    rng = np.random.default_rng(20261018)
    n,c = 175,32
    f = np.fft.rfftfreq(n,1/25)
    active = (f>=.5)&(f<8)
    wr = np.zeros((len(f),c),complex)
    wr[active] = 1/c
    component = rng.normal(size=(n,1))@np.linspace(.8,1.2,c)[None,:]
    background = rng.normal(size=(n,c))
    return background,component,wr


def test_reference_has_zero_error_and_preserves_separate_powers():
    background,component,wr = fixture()
    result = evaluate_recorded_component(background,component,wr,wr)
    assert result["recorded_component_relative_mse"] == 0
    assert result["component_power_relative_reference_db"] == 0
    np.testing.assert_allclose(result["recorded_component_output_power"],result["conventional_reference_output_power"])
    # Exactlinear decomposition, never SNR inferred from the mixture.
    combined = np.fft.irfft(np.sum(np.fft.rfft(background+component,axis=0)*wr.conj(),axis=1),n=175)
    separated = np.fft.irfft(np.sum(np.fft.rfft(background,axis=0)*wr.conj(),axis=1),n=175)
    separated += np.fft.irfft(np.sum(np.fft.rfft(component,axis=0)*wr.conj(),axis=1),n=175)
    np.testing.assert_allclose(combined,separated,atol=1e-14)


def test_component_scale_does_not_change_gain_or_relative_error():
    background,component,wr = fixture()
    w = derivative_frequency_weights(np.repeat(np.eye(32)[None],4,axis=0),175,np.zeros(32))
    first = evaluate_recorded_component(background,component,w,wr)
    for scale in (.03, 4.0, -2.0):
        second = evaluate_recorded_component(background,scale*component,w,wr)
        for name in ("recorded_component_relative_mse","component_ratio_gain_db",
                     "reference_to_input_power_ratio","component_power_relative_reference_db"):
            np.testing.assert_allclose(first[name],second[name],atol=1e-12)


def test_twofold_candidate_output_has_unit_relative_error():
    background,component,wr = fixture()
    result = evaluate_recorded_component(background,component,2*wr,wr)
    np.testing.assert_allclose(result["recorded_component_relative_mse"],1,atol=1e-14)
    np.testing.assert_allclose(result["component_power_relative_reference_db"],10*np.log10(4))
    # Bothseparateoutputs scale equally, so gain does not improve by scalar amplification.
    base = evaluate_recorded_component(background,component,wr,wr)
    np.testing.assert_allclose(result["component_ratio_gain_db"],base["component_ratio_gain_db"],atol=1e-14)


def test_nearly_cancelled_reference_is_kept_with_explicit_ratio():
    background,component,wr = fixture()
    q = np.linspace(-1,1,32)
    component = component[:,0,None]*q[None,:]
    component[:,0] += 1e-6*component[:,1]
    result = evaluate_recorded_component(background,component,wr,wr)
    assert 0 < result["reference_to_input_power_ratio"] < 1e-10
    assert np.isfinite(list(result.values())).all()


@pytest.mark.parametrize("kind",["zero_reference","nonfinite","wrong_shape","out_of_band","zero_background"])
def test_invalid_or_undefined_cases_fail_explicitly(kind):
    background,component,wr = fixture()
    w = wr.copy()
    if kind == "zero_reference": wr[:] = 0
    if kind == "nonfinite": component[0,0] = np.nan
    if kind == "wrong_shape": component = component[:-1]
    if kind == "out_of_band": w[0] = 1
    if kind == "zero_background": background[:] = 0
    with pytest.raises(ValueError):
        evaluate_recorded_component(background,component,w,wr)
