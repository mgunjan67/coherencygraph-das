"""Metadata-only event eligibility, source/background overlap and byte arithmetic."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def instant(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    headers_path = OUT / "hdf_header_audit.json"
    headers = json.loads(headers_path.read_text(encoding="utf-8"))
    archive = ROOT / headers["archive"]
    ledger_path = ROOT / "revisions/20261012_option_a_real_events/codex_review/DATA_EXPOSURE_LEDGER.csv"
    with ledger_path.open(encoding="utf-8-sig", newline="") as f:
        ledger = list(csv.DictReader(f))
    backgrounds = [r for r in ledger if r["role"] == "development"]
    assert len(backgrounds) == 10
    event_origins = headers["catalog_rows_with_origin_or_detection_in_example_members"]["Catalog_with_absolute_earthquake_locations.csv"]
    event_detections = headers["catalog_rows_with_origin_or_detection_in_example_members"]["list_of_detections.csv"]
    by_id = {r["ID"]: r for r in event_detections}
    with zipfile.ZipFile(archive) as z:
        geometry_content = z.read("das-kefalonia-v2/Cable_geometry/das_channels.csv")
        geometry = list(csv.DictReader(io.StringIO(geometry_content.decode())))
    ids = {int(r["channel_number_recorded"]) for r in geometry}
    roi = list(range(4000, 5901, 5))
    assert len(roi) == 381 and set(roi).issubset(ids)

    sources = []
    for origin in event_origins:
        det = by_id[origin["event_id"]]
        assert det["classification"] == "T"
        members = [m for m in headers["members"] if m["member"] in origin["overlapping_members"]]
        assert len(members) == 1
        member = members[0]
        attrs = member["datasets"]["Acquisition/Raw[0]/RawData"]["attrs"]
        start = instant(attrs["PartStartTime"])
        stop = instant(attrs["PartEndTime"]) + timedelta(seconds=0.004)
        detection = instant(det["Detection_Time"])
        requested_start = detection - timedelta(seconds=2)
        requested_stop = detection + timedelta(seconds=5)
        begin_index = math.ceil((requested_start - start).total_seconds() * 250 - 1e-8)
        # Exactly 1750 samples; no source/outcome adjustment to make a desired pattern.
        end_index = begin_index + 1750
        assert 0 <= begin_index < end_index <= 7500
        crop_start = start + timedelta(seconds=begin_index / 250)
        crop_stop = start + timedelta(seconds=end_index / 250)
        overlaps = []
        for bg in backgrounds:
            a, b = instant(bg["acquired_start_utc"]), instant(bg["acquired_end_utc"])
            if max(start, a) < min(stop, b):
                overlaps.append(bg["data_id"])
        assert not overlaps
        other_detections = [
            r for r in event_detections
            if r["ID"] != origin["event_id"]
            and crop_start <= instant(r["Detection_Time"]) < crop_stop
        ]
        assert not other_detections
        row_chunks = set(range(begin_index // 537, (end_index - 1) // 537 + 1))
        col_chunks = {c // 122 for c in roi}
        chunk_bytes = sum(min(537, 7500 - r * 537) * min(122, 7750 - c * 122) * 4 for r in row_chunks for c in col_chunks)
        sources.append({
            "event_id": origin["event_id"], "origin_catalog": origin,
            "detection_catalog": det,
            "member": member["member"], "source_packet_start_utc": start.isoformat(),
            "source_packet_stop_exclusive_utc": stop.isoformat(),
            "proposed_crop_rule_not_executed": "first native sample at or after detection minus 2 s; exactly 1750 native samples (7 s)",
            "proposed_crop_native_start": begin_index, "proposed_crop_native_stop_exclusive": end_index,
            "proposed_crop_start_utc": crop_start.isoformat(), "proposed_crop_stop_exclusive_utc": crop_stop.isoformat(),
            "source_packet_temporal_overlaps_with_any_of_10_development_background_caches": overlaps,
            "other_catalog_detections_inside_proposed_crop": other_detections,
            "proposed_crop_roi_logical_int32_bytes": 1750 * 381 * 4,
            "hdf_internal_chunks_intersecting_crop_roi": len(row_chunks) * len(col_chunks),
            "hdf_internal_intersecting_chunk_uncompressed_int32_bytes": chunk_bytes,
            "source_exposure_label": "EXPLORATORY_SOURCE_PROVIDED_EXAMPLE; prior author exposure unknown; no untouched claim",
        })
    result = {
        "status": "METADATA_ELIGIBLE_FOR_FIXED_RECORDED_COMPONENT_CASE_STUDY_NOT_SAMPLE_QC",
        "archive_sha256": digest(archive),
        "hdf_header_audit_sha256": digest(headers_path),
        "ledger_sha256": digest(ledger_path),
        "geometry_csv_sha256": hashlib.sha256(geometry_content).hexdigest(),
        "recorded_roi_ids": roi,
        "geometry_roi_missing_ids": sorted(set(roi) - ids),
        "source_count": 2, "source_calendar_days": 1,
        "background_development_caches": [{k: r[k] for k in ["data_id", "acquired_start_utc", "acquired_end_utc", "local_path", "local_bytes", "untouched_status"]} for r in backgrounds],
        "sources": sources,
        "volumes": {
            "archive_compressed_bytes": archive.stat().st_size,
            "four_hdf_member_files_bytes": sum(m["file_bytes"] for m in headers["members"]),
            "four_hdf_zip_compressed_member_bytes": sum(m["zip_compressed_bytes"] for m in headers["members"]),
            "four_hdf_full_rawdata_logical_int32_bytes": 4 * 7500 * 7750 * 4,
            "four_hdf_fulltime_roi_logical_int32_bytes": 4 * 7500 * 381 * 4,
            "two_proposed_7s_roi_crops_logical_int32_bytes": 2 * 1750 * 381 * 4,
            "two_proposed_7s_roi_25hz_float64_bytes": 2 * 175 * 381 * 8,
            "two_proposed_7s_eight_aperture_float64_bytes": 2 * 175 * 8 * 32 * 8,
            "two_crop_hdf_internal_chunk_uncompressed_bytes": sum(s["hdf_internal_intersecting_chunk_uncompressed_int32_bytes"] for s in sources),
        },
        "no_waveform_samples_decoded": True,
        "no_model_calls": True,
        "no_calibration_or_96h_assessment_access": True,
        "runtime_prediction": "Not measured. Byte arithmetic is not an acquisition/execution ETA.",
    }
    (OUT / "source_eligibility.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"source_count": 2, "source_days": 1, "no_overlap": all(not s['source_packet_temporal_overlaps_with_any_of_10_development_background_caches'] for s in sources), "volumes": result['volumes']}, indent=2))


if __name__ == "__main__":
    main()
