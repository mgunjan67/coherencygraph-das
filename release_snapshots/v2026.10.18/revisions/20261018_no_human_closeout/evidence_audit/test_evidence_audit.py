"""Finite tests of the source metadata and no-sample-read boundary."""
import io
import json
import hashlib
import unittest
from pathlib import Path

import numpy as np

from audit_zip_headers import CAP, MetadataOnlyFile

OUT = Path(__file__).resolve().parent


class EvidenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.headers = json.loads((OUT / "hdf_header_audit.json").read_text())
        self.eligibility = json.loads((OUT / "source_eligibility.json").read_text())

    def test_returned_read_guard(self):
        limited = MetadataOnlyFile(io.BytesIO(b"metadata"))
        self.assertEqual(limited.read(4), b"meta")
        with self.assertRaises(RuntimeError):
            limited.read(-1)
        with self.assertRaises(RuntimeError):
            limited.read(CAP)

    def test_all_headers_metadata_only(self):
        self.assertEqual(self.headers["dataset_sample_reads"], 0)
        self.assertEqual(len(self.headers["members"]), 4)
        for m in self.headers["members"]:
            self.assertLess(m["hdf_header_bytes_returned"], CAP)
            d = m["datasets"]["Acquisition/Raw[0]/RawData"]
            self.assertEqual(d["shape"], [7500, 7750])
            self.assertEqual(d["dtype"], "int32")
            self.assertEqual(m["groups"]["Acquisition/Raw[0]"]["attrs"]["OutputDataRate"], 250)
            self.assertEqual(m["groups"]["Acquisition/Raw[0]"]["attrs"]["StartLocusIndex"], 0)
            self.assertIn("2024-08-02", m["member"])

    def test_source_coverage_and_background_disjointness(self):
        self.assertEqual(self.eligibility["source_count"], 2)
        self.assertEqual(self.eligibility["source_calendar_days"], 1)
        self.assertEqual(len(self.eligibility["background_development_caches"]), 10)
        for s in self.eligibility["sources"]:
            self.assertEqual(s["proposed_crop_native_stop_exclusive"] - s["proposed_crop_native_start"], 1750)
            self.assertFalse(s["source_packet_temporal_overlaps_with_any_of_10_development_background_caches"])
            self.assertFalse(s["other_catalog_detections_inside_proposed_crop"])
            self.assertEqual(s["detection_catalog"]["classification"], "T")
        self.assertFalse(self.eligibility["geometry_roi_missing_ids"])

    def test_resource_arithmetic_and_exposure(self):
        self.assertEqual(self.eligibility["volumes"]["two_proposed_7s_roi_crops_logical_int32_bytes"], 5_334_000)
        self.assertTrue(self.eligibility["no_waveform_samples_decoded"])
        self.assertTrue(self.eligibility["no_model_calls"])
        self.assertTrue(self.eligibility["no_calibration_or_96h_assessment_access"])
        self.assertEqual(self.eligibility["archive_sha256"], "7f38c3c65c21c66402e973c599c723e02d1fcdb1c1b0fcf47a84c4307b099674")

    def test_source_metadata_seal_and_steering(self):
        spec = json.loads((OUT / "source_spec.json").read_text())
        seal = json.loads((OUT / "source_seal.json").read_text())
        self.assertEqual(hashlib.sha256((OUT / "source_spec.json").read_bytes()).hexdigest(), seal["source_spec_sha256"])
        with np.load(OUT / "source_catalog_steering.npz") as geometry:
            self.assertEqual(geometry["delays_seconds"].shape, (2, 8, 32))
            self.assertTrue(np.isfinite(geometry["delays_seconds"]).all())
            np.testing.assert_allclose(geometry["delays_seconds"].mean(-1), 0, atol=1e-15)
        for source in spec["steering"]:
            self.assertEqual(len(source["apertures"]), 8)
            for aperture in source["apertures"]:
                self.assertTrue(np.isclose(aperture["horizontal_slowness_s_per_km"], aperture["finite_difference_slowness_s_per_km"], rtol=.01, atol=.001))
        self.assertFalse(spec["waveform_samples_read_in_preparation"])

    def test_extract_rejects_unapproved_outside_protocol(self):
        import prepare_source_components
        before = (OUT / "decoded_sources").exists()
        with self.assertRaises(PermissionError):
            prepare_source_components.extract(OUT.parents[2] / "README.md", "invalid")
        self.assertEqual((OUT / "decoded_sources").exists(), before)


if __name__ == "__main__":
    unittest.main()
