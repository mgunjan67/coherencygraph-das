"""Read ZIP directory/source metadata and HDF headers, never dataset samples.

ZipExtFile.seek may internally decompress/skip container bytes. The explicit
limit applies to bytes returned to HDF5; no HDF waveform dataset is evaluated,
no member is extracted, and the ZIP is never unpacked wholesale.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "revisions/20261004_detection_feasibility/sources/das-kefalonia-v2.zip"
PREFIX = "das-kefalonia-v2/"
CAP = 5_000_000


class MetadataOnlyFile(io.RawIOBase):
    def __init__(self, stream):
        self.stream = stream
        self.bytes_returned = 0
        self.read_calls = 0
        self.seek_calls = 0
        self.ranges = []

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.stream.tell()

    def seek(self, offset, whence=io.SEEK_SET):
        self.seek_calls += 1
        return self.stream.seek(offset, whence)

    def read(self, size=-1):
        if size < 0 or self.bytes_returned + size > CAP:
            raise RuntimeError("Metadata-only returned-read cap exceeded")
        start = self.tell()
        result = self.stream.read(size)
        self.bytes_returned += len(result)
        self.read_calls += 1
        self.ranges.append([start, len(result)])
        return result

    def readinto(self, target):
        result = self.read(len(target))
        target[: len(result)] = result
        return len(result)


def normalize(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size > 1000:
            return {"attribute_shape": list(value.shape), "not_expanded": True}
        return normalize(value.tolist())
    if isinstance(value, np.generic):
        return normalize(value.item())
    if isinstance(value, (list, tuple)):
        return [normalize(x) for x in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def utc(value):
    return datetime.fromisoformat(value.rstrip("Z") + "+00:00")


def main():
    started = time.perf_counter()
    result = {
        "status": "METADATA_ONLY_NO_SAMPLE_DECODING",
        "date": "2026-10-18",
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "metadata_returned_byte_cap_per_hdf": CAP,
        "zip_seek_caveat": "ZIP deflate skip/decompression is possible; no HDF dataset samples read and no member extracted.",
        "dataset_sample_reads": 0,
        "members": [],
        "metadata_sources": {},
    }
    with zipfile.ZipFile(ARCHIVE) as z:
        header_names = sorted(n for n in z.namelist() if n.endswith(".h5"))
        assert len(header_names) == 4
        assert all("2024-08-02" in n for n in header_names)
        for name in header_names:
            info = z.getinfo(name)
            local_started = time.perf_counter()
            row = {
                "member": name,
                "file_bytes": info.file_size,
                "zip_compressed_bytes": info.compress_size,
                "zip_crc32": f"{info.CRC:08x}",
                "datasets": {},
                "groups": {},
            }
            with z.open(info) as member:
                limited = MetadataOnlyFile(member)
                with h5py.File(limited, "r") as h:
                    row["root_attrs"] = {k: normalize(v) for k, v in h.attrs.items()}

                    def visit(path, obj):
                        attrs = {k: normalize(v) for k, v in obj.attrs.items()}
                        if isinstance(obj, h5py.Dataset):
                            row["datasets"][path] = {
                                "shape": list(obj.shape),
                                "dtype": str(obj.dtype),
                                "chunks": normalize(obj.chunks),
                                "compression": obj.compression,
                                "compression_opts": normalize(obj.compression_opts),
                                "logical_bytes": int(obj.size * obj.dtype.itemsize),
                                "attrs": attrs,
                            }
                        else:
                            row["groups"][path] = {"attrs": attrs}
                    h.visititems(visit)
                row["hdf_header_bytes_returned"] = limited.bytes_returned
                row["hdf_header_read_calls"] = limited.read_calls
                row["hdf_header_seek_calls"] = limited.seek_calls
                row["hdf_header_ranges"] = limited.ranges
            row["seconds"] = time.perf_counter() - local_started
            result["members"].append(row)
            print(name, row["hdf_header_bytes_returned"], row["seconds"], flush=True)

        for name in ["README.md", "Codes/das_class.py", "Codes/main_plot_events.py", "Codes/utils_filetime.py"]:
            content = z.read(PREFIX + name)
            result["metadata_sources"][name] = {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        catalogs = {}
        for name in ["Catalog_with_absolute_earthquake_locations.csv", "Enhanced_earthquake_catalog.csv", "list_of_detections.csv"]:
            content = z.read(PREFIX + "Catalogs/" + name)
            catalogs[name] = list(csv.DictReader(io.StringIO(content.decode())))
            result["metadata_sources"][name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}

        spans = []
        for m in result["members"]:
            dataset = m["datasets"]["Acquisition/Raw[0]/RawData"]
            a = dataset["attrs"]
            spans.append((utc(a["PartStartTime"]), utc(a["PartEndTime"]), m["member"]))
        overlap_rows = {}
        timefields = {
            "Catalog_with_absolute_earthquake_locations.csv": "origin",
            "Enhanced_earthquake_catalog.csv": "detection_time",
            "list_of_detections.csv": "Detection_Time",
        }
        for name, rows in catalogs.items():
            matched = []
            for row in rows:
                timestamp = utc(row[timefields[name]])
                members = [member for start, end, member in spans if start <= timestamp <= end]
                if members:
                    matched.append({**row, "overlapping_members": members})
            overlap_rows[name] = matched
        result["catalog_rows_with_origin_or_detection_in_example_members"] = overlap_rows

    result["seconds"] = time.perf_counter() - started
    result["returned_hdf_metadata_bytes_total"] = sum(m["hdf_header_bytes_returned"] for m in result["members"])
    (OUT / "hdf_header_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"seconds": result["seconds"], "metadata_bytes": result["returned_hdf_metadata_bytes_total"], "overlap_counts": {k:len(v) for k,v in overlap_rows.items()}}, indent=2))


if __name__ == "__main__":
    main()
