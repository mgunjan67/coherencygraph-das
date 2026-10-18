"""Known ADDED recorded-wavefield metrics, not inferred clean-earthquake truth.

No labels or arrival accuracy are inferred. A template may contain unseparated
natural background, and a target background may contain natural earthquakes.
The reference is an explicitly supplied fixed conventional filtered template.
"""
from __future__ import annotations
import numpy as np


def _filter(data, weights):
    return np.fft.irfft(np.sum(np.fft.rfft(data,axis=0)*weights.conj(),axis=1),n=len(data))


def evaluate_recorded_component(background, recorded_component, weights,
                                reference_weights, sample_rate=25.0):
    """Evaluate separately retained components through identical linear support.

    All powers use the common0.5<=f<8Hz passband. Both the candidate and reference
    weights must vanish outside it. Exactzero reference/background/output powers
    stop explicitly; nearly cancelling conventional references are NOT excluded.
    Their reference-to-input power ratio is returned as an interpretation audit.
    """
    background = np.asarray(background,dtype=float)
    component = np.asarray(recorded_component,dtype=float)
    w = np.asarray(weights,dtype=complex)
    wr = np.asarray(reference_weights,dtype=complex)
    if (background.ndim != 2 or len(background)<16 or background.shape[1]<2
            or component.shape != background.shape
            or w.shape != (len(background)//2+1,background.shape[1]) or wr.shape != w.shape
            or not all(np.isfinite(x).all() for x in (background,component,w,wr))
            or not np.isfinite(sample_rate) or sample_rate<=0):
        raise ValueError("Aligned finite time-channel components/frequency weights required")
    frequencies = np.fft.rfftfreq(len(background),1/sample_rate)
    active = (frequencies>=.5)&(frequencies<8)
    if np.any(w[~active] != 0) or np.any(wr[~active] != 0):
        raise ValueError("Both filters must have identical declared passband support")
    spectra_s = np.fft.rfft(component,axis=0)
    spectra_n = np.fft.rfft(background,axis=0)
    spectra_s[~active] = 0
    spectra_n[~active] = 0
    band_s = np.fft.irfft(spectra_s,n=len(background),axis=0)
    band_n = np.fft.irfft(spectra_n,n=len(background),axis=0)
    out_s = _filter(component,w)
    out_n = _filter(background,w)
    reference = _filter(component,wr)
    ps,pn,pref,pin_s,pin_n = map(float,(np.mean(out_s**2),np.mean(out_n**2),
        np.mean(reference**2),np.mean(band_s**2),np.mean(band_n**2)))
    if min(ps,pn,pref,pin_s,pin_n)<=0:
        raise ValueError("Zero component/reference/background power; record failure, do not silently drop")
    mse = float(np.mean((out_s-reference)**2))
    result = dict(recorded_component_output_power=ps,background_output_power=pn,
        conventional_reference_output_power=pref,recorded_component_input_power=pin_s,
        background_input_power=pin_n,recorded_component_relative_mse=mse/pref,
        output_error_power=mse,reference_to_input_power_ratio=pref/pin_s,
        component_input_ratio_db=float(10*np.log10(pin_s/pin_n)),
        component_output_ratio_db=float(10*np.log10(ps/pn)),
        component_ratio_gain_db=float(10*np.log10(ps/pn)-10*np.log10(pin_s/pin_n)),
        component_power_relative_reference_db=float(10*np.log10(ps/pref)))
    if not np.isfinite(list(result.values())).all():
        raise ValueError("Nonfinite known-recorded-component metric")
    return result
