"""Small end-to-end synthetic timing ONLY; no recorded array/model is opened."""
import os
for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key] = "2"
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import psutil

from derivative_lcmv import derivative_frequency_weights
from recorded_component_metrics import evaluate_recorded_component

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT/"revisions/20261010_forecast_processing"))
from frequency_filter import frequency_weights, ricker_wavelet, inject_component, evaluate_components


def main():
    rng = np.random.default_rng(20261018)
    context = np.empty((8,4,32,32),complex)
    for aperture in range(8):
        z = rng.normal(size=(4,32,32))+1j*rng.normal(size=(4,32,32))
        context[aperture] = z@z.conj().swapaxes(-1,-2)/32+.3*np.eye(32)
    backgrounds = rng.normal(size=(8,175,32))
    templates = rng.normal(size=(2,8,175,32))
    coordinate = np.arange(32)-15.5
    transplant_tic = time.perf_counter()
    transplant_rows = 0
    for source in range(2):
        for aperture in range(8):
            ctx = context[aperture]
            delays = (.001+.001*source)*coordinate
            # Synthetic proxies preserve ONLY the six-method code paths and
            # 32-channel matrix dimensions; these are NOT neural forecasts.
            matrices = {name:ctx.copy() for name in ("neural_proxy","ridge_proxy","context025","context075","conventional")}
            reference = frequency_weights(ctx,175,delays,conventional=True)
            weights = {name:frequency_weights(matrix,175,delays,conventional=name=="conventional")
                       for name,matrix in matrices.items()}
            weights["derivative_lcmv"] = derivative_frequency_weights(ctx,175,delays)
            for w in weights.values():
                evaluate_recorded_component(backgrounds[aperture],templates[source,aperture],w,reference)
                transplant_rows += 1
    transplant_seconds = time.perf_counter()-transplant_tic
    ricker_tic = time.perf_counter()
    ricker_rows = 0
    for aperture in range(8):
        for slope in (-.004,0,.004):
            weights = derivative_frequency_weights(context[aperture],175,slope*coordinate)
            for hz in (1,2,4):
                wavelet = ricker_wavelet(central_frequency=hz)
                for delta,gradient in ((0,0),(-.001,0),(.001,0),(0,-.2),(0,.2)):
                    component = inject_component(wavelet,(slope+delta)*coordinate,1+gradient*coordinate/15.5)
                    evaluate_components(backgrounds[aperture],component,wavelet,weights)
                    ricker_rows += 1
    ricker_seconds = time.perf_counter()-ricker_tic
    result = dict(status="SYNTHETIC_PIPELINE_TIMING_NOT_FIELD_RESULTS",
        no_recorded_payload_opened=True,no_models_loaded=True,thread_environment=2,
        python=sys.version,numpy=np.__version__,worker_rss_at_finish_bytes=psutil.Process().memory_info().rss,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__),HERE/"derivative_lcmv.py",HERE/"recorded_component_metrics.py")},
        transplant=dict(synthetic_source_templates=2,apertures=8,methods=6,
            rows=transplant_rows,seconds=transplant_seconds,
            full_8hour_30cutoff_rows=23040,linear_compute_seconds=240*transplant_seconds),
        ricker_new_comparator=dict(apertures=8,slopes=3,wavelets=3,conditions=5,
            rows=ricker_rows,seconds=ricker_seconds,
            full_8hour_30cutoff_rows=86400,linear_compute_seconds=240*ricker_seconds),
        excluded_costs=["field archive hashing/decompression","source template acquisition/extraction",
            "table/plot generation","independent verification","hardware contention"],
        scaling_warning="One synthetic timed group; linear arithmetic scenarios, not validated field throughput or a delivery promise")
    (HERE/"synthetic_pipeline_resource_receipt.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
