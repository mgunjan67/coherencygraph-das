"""Mathematical correctness tests only, not new field results."""
import numpy as np
import pytest

from derivative_lcmv import (lcmv_weights, centred_coordinate, derivative_constraints,
                             derivative_frequency_weights, affine_delay_response_error_bound)


def hpd(channels=32, seed=20261018):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(channels, channels)) + 1j*rng.normal(size=(channels, channels))
    return z @ z.conj().T / channels + .3*np.eye(channels)


@pytest.mark.parametrize("seed", [19,43,71])
def test_single_constraint_equals_closed_form_mvdr(seed):
    r = hpd(seed=seed)
    a = np.exp(-2j*np.pi*2.0*.004*(np.arange(32)-15.5))
    loaded = r + .001*np.trace(r).real/32*np.eye(32)
    z = np.linalg.solve(loaded, a)
    expected = z/np.vdot(a,z).real
    np.testing.assert_allclose(lcmv_weights(r,a[:,None],[1]), expected, atol=2e-14, rtol=2e-13)


def test_complex_gains_constraint_convention():
    c = np.column_stack((np.ones(8),np.arange(8))).astype(complex)
    g = np.array([1+.5j, -.2j])
    w = lcmv_weights(hpd(8),c,g)
    np.testing.assert_allclose(c.conj().T@w,g,atol=1e-13)
    np.testing.assert_allclose(w.conj()@c,g.conj(),atol=1e-13)


def test_exact_linear_amplitude_and_complex_tangent_preservation():
    a = np.exp(-2j*np.pi*4*.004*(np.arange(32)-15.5))
    q = centred_coordinate(32)
    c,g = derivative_constraints(a,q)
    w = lcmv_weights(hpd(),c,g)
    for alpha in (-.2, .2, .9, .3j, -.6+.8j):
        np.testing.assert_allclose(np.vdot(w,a*(1+alpha*q)),1,atol=2e-14)


def test_delay_first_derivative_zero_quadratic_error_and_bound():
    q = centred_coordinate(32)
    f = 4.0
    a = np.exp(-2j*np.pi*f*.004*(np.arange(32)-15.5))
    c,g = derivative_constraints(a,q)
    w = lcmv_weights(hpd(),c,g)
    np.testing.assert_allclose(np.vdot(w,(-2j*np.pi*f*q)*a),0,atol=2e-14)
    errors = []
    for epsilon in (.001,.0005,.00025):
        error = abs(np.vdot(w,a*np.exp(-2j*np.pi*f*epsilon*q))-1)
        assert error <= affine_delay_response_error_bound(w,a,q,f,epsilon)+1e-14
        errors.append(error)
    assert 3.9 < errors[0]/errors[1] < 4.1
    assert 3.9 < errors[1]/errors[2] < 4.1


def test_kkt_and_feasible_perturbations_do_not_reduce_loaded_power():
    r = hpd()
    q = centred_coordinate(32)
    c,g = derivative_constraints(np.ones(32),q)
    w = lcmv_weights(r,c,g)
    loaded = r + .001*np.trace(r).real/32*np.eye(32)
    projector = np.eye(32) - c@np.linalg.solve(c.conj().T@c,c.conj().T)
    np.testing.assert_allclose(projector@loaded@w,0,atol=1e-13)
    rng = np.random.default_rng(123)
    optimum = np.vdot(w,loaded@w).real
    for _ in range(12):
        delta = .1*projector@(rng.normal(size=32)+1j*rng.normal(size=32))
        np.testing.assert_allclose(c.conj().T@(w+delta),g,atol=1e-13)
        assert np.vdot(w+delta,loaded@(w+delta)).real >= optimum-1e-13


def test_constraint_basis_and_covariance_scale_invariance():
    r = hpd()
    c,g = derivative_constraints(np.ones(32),centred_coordinate(32))
    t = np.array([[1+.2j,.4j],[.3,2-.1j]])
    w = lcmv_weights(r,c,g)
    np.testing.assert_allclose(lcmv_weights(r,c@t,t.conj().T@g),w,atol=2e-14)
    np.testing.assert_allclose(lcmv_weights(1e6*r,c,g),w,atol=2e-14)


def test_white_covariance_returns_conventional_response():
    a = np.exp(-2j*np.pi*2*.004*(np.arange(32)-15.5))
    c,g = derivative_constraints(a,centred_coordinate(32))
    np.testing.assert_allclose(lcmv_weights(np.eye(32),c,g),a/32,atol=1e-14)


def test_rank_deficient_psd_covariance_allowed_only_with_positive_load():
    c,g = derivative_constraints(np.ones(32),centred_coordinate(32))
    r = np.ones((32,32))
    w = lcmv_weights(r,c,g)
    np.testing.assert_allclose(c.conj().T@w,g,atol=1e-12)
    with pytest.raises(ValueError):
        lcmv_weights(r,c,g,loading=0)


@pytest.mark.parametrize("kind",["duplicate","nan","nonhermitian","indefinite","negative_load","empty_gain"])
def test_invalid_inputs_fail_without_silent_fallback(kind):
    r = hpd(8)
    c,g = derivative_constraints(np.ones(8),centred_coordinate(8))
    loading = .001
    if kind == "duplicate": c[:,1] = c[:,0]
    if kind == "nan": r[0,0] = np.nan
    if kind == "nonhermitian": r[0,1] += 1j
    if kind == "indefinite": r = np.diag([-1]+[2]*7)
    if kind == "negative_load": loading = -.1
    if kind == "empty_gain": g = []
    with pytest.raises(ValueError):
        lcmv_weights(r,c,g,loading=loading)


def test_frequency_support_constraint_and_waveform_preservation():
    samples = 175
    frequency = np.fft.rfftfreq(samples,1/25)
    delays = .004*(np.arange(32)-15.5)
    q = centred_coordinate(32)
    matrices = np.stack([hpd(seed=i) for i in range(4)])
    weights = derivative_frequency_weights(matrices,samples,delays)
    active = (frequency>=.5)&(frequency<8)
    assert np.all(weights[~active] == 0)
    responses = np.exp(-2j*np.pi*frequency[:,None]*delays)
    np.testing.assert_allclose(np.sum(weights[active].conj()*responses[active],axis=1),1,atol=2e-13)
    np.testing.assert_allclose(np.sum(weights[active].conj()*responses[active]*q,axis=1),0,atol=2e-13)
    time = np.arange(samples)/25-3.5
    z = (np.pi*2*time)**2
    spectrum = np.fft.rfft((1-2*z)*np.exp(-z))
    spectrum[~active] = 0
    component = spectrum[:,None]*responses*(1+.2*q)
    output = np.fft.irfft(np.sum(weights.conj()*component,axis=1),n=samples)
    reference = np.fft.irfft(spectrum,n=samples)
    np.testing.assert_allclose(output,reference,atol=2e-13,rtol=2e-12)


@pytest.mark.parametrize("coordinate",[np.ones(32),np.zeros(31),np.r_[np.nan,np.arange(31)]])
def test_bad_coordinate_rejected(coordinate):
    with pytest.raises(ValueError):
        derivative_frequency_weights(np.repeat(np.eye(32)[None],4,axis=0),175,np.zeros(32),coordinate)


def test_missing_band_bins_rejected():
    with pytest.raises(ValueError):
        derivative_frequency_weights(np.repeat(np.eye(32)[None],4,axis=0),16,np.zeros(32))
