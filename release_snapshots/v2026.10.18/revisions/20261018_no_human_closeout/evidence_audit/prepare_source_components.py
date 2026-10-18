"""Prepare metadata-only source seal; decode only under a hash-bound parent gate.

No neural model is imported or evaluated here. The optional extraction mode is
not authorized by preparation. It requires the parent experiment protocol to
explicitly permit source sample decoding and the caller to supply its SHA256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OPTION = ROOT / "revisions/20261012_option_a_real_events/implementation"
SESSION = ROOT / "revisions/20261010_research_session"
GEOMETRY = ROOT / "revisions/20261004_detection_feasibility/sources/kefalonia_metadata/das-kefalonia-v2/Cable_geometry/das_channels.csv"
ARCHIVE = ROOT / "revisions/20261004_detection_feasibility/sources/das-kefalonia-v2.zip"
EXPECTED_ARCHIVE = "7f38c3c65c21c66402e973c599c723e02d1fcdb1c1b0fcf47a84c4307b099674"
for directory in (OPTION, SESSION):
    sys.path.insert(0, str(directory))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def derive_steering(eligibility):
    from obspy import UTCDateTime
    from obspy.geodetics.base import gps2dist_azimuth, locations2degrees
    from obspy.taup import TauPyModel
    import obspy.taup
    from geometry_scan import load_aperture_geometry

    geometry = load_aperture_geometry(GEOMETRY)
    model = TauPyModel(model="ak135")
    arrays, rows = [], []
    for source in eligibility["sources"]:
        event = source["origin_catalog"]
        latitude, longitude, depth = float(event["latitude"]), float(event["longitude"]), float(event["depth"])
        origin = UTCDateTime(event["origin"])
        local_slow, local_rows = [], []
        for aperture, (lat, lon) in enumerate(geometry["origins_latitude_longitude_degrees"]):
            distance = locations2degrees(latitude, longitude, lat, lon)
            arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=distance, phase_list=["S", "s"])
            if not arrivals:
                raise RuntimeError("Missing fixed AK135 S/s branch")
            arrival = min(arrivals, key=lambda v: v.time)
            _, bearing, _ = gps2dist_azimuth(lat, lon, latitude, longitude)
            propagation = (bearing + 180) % 360
            km_per_degree = np.pi * model.model.radius_of_planet / 180
            p = float(arrival.ray_param_sec_degree) / km_per_degree
            epsilon = 1e-5
            times = []
            for delta in (distance - epsilon, distance + epsilon):
                branches = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=delta, phase_list=[arrival.name])
                if not branches:
                    raise RuntimeError("Cannot check fixed ray-parameter derivative")
                times.append(min(v.time for v in branches))
            derivative = (times[1] - times[0]) / (2 * epsilon * km_per_degree)
            if not np.isfinite(p) or p < 0 or not np.isclose(p, derivative, rtol=.01, atol=.001):
                raise RuntimeError("AK135 metadata derivative check failed")
            vector = p * np.asarray([np.sin(np.deg2rad(propagation)), np.cos(np.deg2rad(propagation))])
            local_slow.append(vector)
            local_rows.append(dict(aperture=aperture, aperture_origin_latitude=float(lat), aperture_origin_longitude=float(lon),
                phase=arrival.name, travel_time_seconds=float(arrival.time), predicted_arrival_utc=str(origin + arrival.time),
                distance_degrees=float(distance), horizontal_slowness_s_per_km=p,
                finite_difference_slowness_s_per_km=float(derivative), propagation_bearing_degrees=float(propagation),
                slowness_east_north_s_per_km=vector.tolist()))
        delays = np.einsum("ad,acd->ac", np.asarray(local_slow), geometry["coordinates_enu_km"])
        arrays.append(delays)
        rows.append(dict(event_id=event["event_id"], origin_utc=str(origin), depth_km=depth, latitude=latitude, longitude=longitude,
            apertures=local_rows, maximum_absolute_delay_seconds=float(abs(delays).max())))
    return np.asarray(arrays), rows, geometry, Path(obspy.taup.__file__).parent / "data/ak135.npz"


def prepare():
    from obspy import UTCDateTime

    if (HERE / "source_spec.json").exists() or (HERE / "source_seal.json").exists():
        raise RuntimeError("Do not overwrite a pre-outcome source design")
    eligibility = json.loads((HERE / "source_eligibility.json").read_text(encoding="utf-8"))
    headers = json.loads((HERE / "hdf_header_audit.json").read_text(encoding="utf-8"))
    if sha(ARCHIVE) != EXPECTED_ARCHIVE:
        raise RuntimeError("Pinned archive changed")
    delays, rows, geometry, model_file = derive_steering(eligibility)
    assert delays.shape == (2, 8, 32) and np.isfinite(delays).all()
    np.savez(HERE / "source_catalog_steering.npz", delays_seconds=delays,
        coordinates_enu_km=geometry["coordinates_enu_km"], aperture_recorded_ids=geometry["aperture_recorded_ids"],
        aperture_starts=geometry["aperture_starts"], recorded_roi_ids=geometry["roi_recorded_ids"])
    by_member = {r["member"]: r for r in headers["members"]}
    sources = []
    for source in eligibility["sources"]:
        member = by_member[source["member"]]
        first = source["proposed_crop_native_start"]
        stop = source["proposed_crop_native_stop_exclusive"]
        anchor = UTCDateTime(source["source_packet_start_utc"]).ns
        crop_ns = anchor + first * 4_000_000
        assert UTCDateTime(source["proposed_crop_start_utc"]).ns == crop_ns
        sources.append(dict(event_id=source["event_id"], member=source["member"], member_bytes=member["file_bytes"],
            zip_crc32=member["zip_crc32"], packet_anchor_ns=anchor,
            native_start=first, native_stop_exclusive=stop, crop_start_ns=crop_ns,
            crop_stop_exclusive_ns=crop_ns + 7_000_000_000,
            source_origin=source["origin_catalog"], detection=source["detection_catalog"],
            source_shape=[7500, 7750], selected_native_shape=[1750, 381], processed_shape=[175, 381],
            source_crop_utc_first=str(UTCDateTime(ns=crop_ns)),
            source_analysis_utc_first=str(UTCDateTime(ns=crop_ns)),
            source_analysis_utc_last=str(UTCDateTime(ns=crop_ns + 174 * 40_000_000)),
            source_clock_handling="Crop-local 25-Hz grid anchored at the selected native first sample; relative component grid is transplanted, not asserted to match the source packet's global decimation phase."))
    binding = [Path(__file__), HERE / "source_eligibility.json", HERE / "hdf_header_audit.json", GEOMETRY,
               OPTION / "geometry_scan.py", SESSION / "phase_resampling.py", model_file]
    spec = dict(schema="fixed-recorded-component-source-v1", status="SEALED_METADATA_ONLY_NO_SAMPLE_READ",
        source_exposure="Previously cached public examples; earlier author exposure uncertain; exploratory, not untouched",
        archive=str(ARCHIVE), archive_sha256=EXPECTED_ARCHIVE, sources=sources,
        roi_recorded_ids=geometry["roi_recorded_ids"].tolist(), aperture_starts=geometry["aperture_starts"].tolist(),
        crop_rule="first native sample at or after detection minus 2 s, exactly 1750 samples",
        preprocessing="phase_resample(native_crop,1,10,axis=0) with existing offset removal and padtype=line, then scipy.signal.detrend(type=linear,axis=0)",
        units="source int32 optical-phase count units; no strain calibration; method-shared amplitude scaling belongs to parent protocol",
        source_model="AK135 earliest S/s at catalog source depth and aperture center, existing WGS84 horizontal ENU",
        steering=rows, steering_limits="Assumed horizontal response, elevation excluded; modeled S arrival/slowness is not observed phase truth and is not refit.",
        source_hashes={str(p): sha(p) for p in binding},
        sample_qc="exact header native anchor; exact requested timestamps under documented ns/us time-coordinate match; GPS bits0/1 set, compromised bit1 clear; no int32 rails or constant retained channel; no substitution",
        expected_processed_relative_time_seconds=(np.arange(175) / 25).tolist(),
        resampling_support="only the selected 7-s component; existing segment-local line extrapolation; offline/acausal. No neighboring samples read or onset latency claim.",
        extraction_caps=dict(seconds=300, new_scratch_bytes=300_000_000, member_bytes=100_000_000),
        parent_gate_required=True, assessment_96h_disabled=True, frozen_48h_unchanged=True,
        automatic_pick_csv_access=False, waveform_samples_read_in_preparation=False)
    save(HERE / "source_spec.json", spec)
    save(HERE / "source_seal.json", dict(source_spec_sha256=sha(HERE / "source_spec.json"),
        source_runner_sha256=sha(__file__), steering_sha256=sha(HERE / "source_catalog_steering.npz"),
        sealed_before_sample_reads=True))
    print(json.dumps(dict(status="SEALED_METADATA_ONLY", sources=[r["event_id"] for r in sources], delay_shape=list(delays.shape))), flush=True)


def extract(parent_protocol, parent_sha256):
    """Guarded API for the root-owned, externally watched experiment, never auto-run."""
    import zipfile
    import h5py
    from obspy import UTCDateTime
    from scipy.signal import detrend
    from phase_resampling import phase_resample

    protocol_path = Path(parent_protocol).resolve()
    allowed_root = ROOT / "revisions/20261018_no_human_closeout"
    if not protocol_path.is_relative_to(allowed_root) or sha(protocol_path) != parent_sha256:
        raise PermissionError("Parent protocol must be exact hash-bound local closeout file")
    parent = json.loads(protocol_path.read_text(encoding="utf-8"))
    if parent.get("allow_source_sample_decode") is not True or parent.get("assessment_96h_disabled") is not True:
        raise PermissionError("Parent protocol has not unlocked only the bounded source sample read")
    seal = json.loads((HERE / "source_seal.json").read_text())
    if sha(HERE / "source_spec.json") != seal["source_spec_sha256"] or sha(__file__) != seal["source_runner_sha256"]:
        raise RuntimeError("Source specification/implementation changed since seal")
    if sha(HERE / "source_catalog_steering.npz") != seal["steering_sha256"]:
        raise RuntimeError("Catalog steering changed")
    spec = json.loads((HERE / "source_spec.json").read_text())
    for filename, expected in spec["source_hashes"].items():
        if sha(filename) != expected:
            raise RuntimeError("Bound source dependency changed")
    if sha(ARCHIVE) != spec["archive_sha256"]:
        raise RuntimeError("Source archive changed")
    destination = HERE / "decoded_sources"
    if destination.exists():
        raise RuntimeError("Source extraction already exists; no replacement or unlogged rerun")
    destination.mkdir()
    started = time.perf_counter()
    receipt = dict(status="RUNNING", parent_protocol_sha256=parent_sha256,
        source_spec_sha256=seal["source_spec_sha256"], sources=[], waveform_samples_read=True,
        model_calls=0, assessment_96h_disabled=True)

    def guard():
        if time.perf_counter() - started > spec["extraction_caps"]["seconds"]:
            raise RuntimeError("Source extraction time cap exceeded")
        used = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
        if used > spec["extraction_caps"]["new_scratch_bytes"]:
            raise RuntimeError("Source extraction scratch cap exceeded")

    try:
        with zipfile.ZipFile(ARCHIVE) as z:
            for source in spec["sources"]:
                guard()
                info = z.getinfo(source["member"])
                if info.file_size != source["member_bytes"] or f"{info.CRC:08x}" != source["zip_crc32"]:
                    raise RuntimeError("Sealed ZIP member mismatch")
                target = destination / Path(source["member"]).name
                digest, crc, count = hashlib.sha256(), 0, 0
                with z.open(info) as incoming, target.open("xb") as outgoing:
                    while chunk := incoming.read(1 << 20):
                        count += len(chunk)
                        if count > spec["extraction_caps"]["member_bytes"]:
                            raise RuntimeError("HDF member cap exceeded")
                        digest.update(chunk)
                        crc = zlib.crc32(chunk, crc)
                        outgoing.write(chunk)
                        guard()
                if count != source["member_bytes"] or f"{crc:08x}" != source["zip_crc32"]:
                    raise RuntimeError("Incomplete/corrupt full HDF extraction")
                a, b = source["native_start"], source["native_stop_exclusive"]
                with h5py.File(target, "r") as h:
                    group = h["Acquisition/Raw[0]"]
                    raw = group["RawData"]
                    if list(raw.shape) != source["source_shape"] or str(raw.dtype) != "int32":
                        raise RuntimeError("Source shape/dtype mismatch")
                    if UTCDateTime(text(raw.attrs["PartStartTime"])).ns != source["packet_anchor_ns"]:
                        raise RuntimeError("Source fractional native anchor changed")
                    if float(group.attrs["OutputDataRate"]) != 250 or int(group.attrs["StartLocusIndex"]) != 0:
                        raise RuntimeError("Source rate/channel origin mismatch")
                    if text(group.attrs["RawDataUnit"]) != "rad * 2PI/2^16":
                        raise RuntimeError("Source unit mismatch")
                    stamps = np.asarray(group["RawDataTime"][a:b], dtype=np.int64)
                    expected_ns = source["crop_start_ns"] + np.arange(1750, dtype=np.int64) * 4_000_000
                    # RawDataTime has no unit attribute in the pinned HDF schema.
                    # Choose only an exact header/grid match, never a fitted offset.
                    scales = [scale for scale in (1, 1000) if np.array_equal(stamps * scale, expected_ns)]
                    if len(scales) != 1:
                        raise RuntimeError("Source sample timestamps do not match exact native UTC grid")
                    gps = np.asarray(group["Custom/GpsStatus"][a:b])
                    compromised = np.asarray(group["Custom/GpBits"][a:b])
                    gps_bad = int(np.count_nonzero((gps & 3) != 3))
                    compromised_bad = int(np.count_nonzero((compromised & 2) != 0))
                    if gps_bad or compromised_bad:
                        raise RuntimeError("Source acquisition quality flags in requested support")
                    native = np.asarray(raw[a:b, 4000:5901:5])
                if native.shape != (1750, 381) or native.dtype != np.int32:
                    raise RuntimeError("Exact native component crop required")
                rails = int(np.count_nonzero((native == np.iinfo(np.int32).min) | (native == np.iinfo(np.int32).max)))
                constants = int(np.count_nonzero(np.ptp(native.astype(np.int64), axis=0) == 0))
                if rails or constants:
                    raise RuntimeError("Rail hit or constant retained source channel")
                component = detrend(phase_resample(native, 1, 10, axis=0), axis=0, type="linear")
                if component.shape != (175, 381) or not np.isfinite(component).all():
                    raise RuntimeError("Processed component shape/finite check failed")
                grid_ns = source["crop_start_ns"] + np.arange(175, dtype=np.int64) * 40_000_000
                output = destination / (source["event_id"] + "_source_component.npz")
                np.savez_compressed(output, native_component=native, component=component,
                    source_analysis_utc_ns=grid_ns, relative_time_seconds=np.arange(175) / 25,
                    roi_recorded_ids=np.asarray(spec["roi_recorded_ids"], dtype=np.int64))
                receipt["sources"].append(dict(event_id=source["event_id"], member=source["member"],
                    full_member_sha256=digest.hexdigest(), full_member_crc32=f"{crc:08x}", full_member_bytes=count,
                    native_crop_sha256=hashlib.sha256(native.tobytes(order="C")).hexdigest(),
                    component_sha256=hashlib.sha256(component.tobytes(order="C")).hexdigest(),
                    output=str(output), output_sha256=sha(output), native_shape=list(native.shape), processed_shape=list(component.shape),
                    native_first_utc=str(UTCDateTime(ns=int(expected_ns[0]))),
                    native_last_utc=str(UTCDateTime(ns=int(expected_ns[-1]))),
                    analysis_first_utc=str(UTCDateTime(ns=int(grid_ns[0]))),
                    analysis_last_utc=str(UTCDateTime(ns=int(grid_ns[-1]))),
                    raw_time_coordinate_scale_to_ns=scales[0], exact_native_grid_check=True,
                    exact_analysis_grid_check=True, gps_bad=gps_bad, compromised_bad=compromised_bad,
                    rails=rails, constant_channels=constants))
                save(destination / "source_extraction_receipt.json", receipt)
                guard()
        receipt["status"] = "COMPLETE_BOUNDED_SOURCE_EXTRACTION_ONLY"
        return receipt
    except Exception as exc:
        receipt.update(status="FAILED_PRESERVE_PARTIAL_NO_SUBSTITUTION", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["seconds"] = time.perf_counter() - started
        receipt["scratch_bytes"] = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
        save(destination / "source_extraction_receipt.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "extract"])
    parser.add_argument("--parent-protocol")
    parser.add_argument("--parent-sha256")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        if not args.parent_protocol or not args.parent_sha256:
            parser.error("extract requires exact parent protocol path and SHA256")
        print(json.dumps(extract(args.parent_protocol, args.parent_sha256), indent=2), flush=True)
