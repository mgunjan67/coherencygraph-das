"""Data-free safety and provenance checks for manifest-driven reconstruction."""
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reproduce_closeout_assets',ROOT/'scripts/reproduce_closeout_assets.py')
repro=importlib.util.module_from_spec(spec);spec.loader.exec_module(repro)

def test_archive_paths_stay_inside_destination(tmp_path):
    assert repro.safe_destination(tmp_path,'data/example.npz')==tmp_path/'data/example.npz'
    for name in ['../outside.txt','/absolute.txt','C:/absolute.txt','data/../../outside','data/..\\../outside']:
        with pytest.raises(ValueError):repro.safe_destination(tmp_path,name)

def test_manifest_pins_source_and_exact_overlay_order():
    cfg=json.loads((ROOT/'configs/closeout_reproduction_manifest.json').read_text())
    assert cfg['source_commit']=='83ded01acc754c5e38aeef521f67bd546e99450a'
    assert [a['id'] for a in cfg['assets']]==cfg['overlay_order']==['base','reporting_correction','audit_utility']
    assert all(len(a['sha256'])==64 for a in cfg['assets'])

def test_manifest_contains_no_training_or_publication_commands():
    cfg=json.loads((ROOT/'configs/closeout_reproduction_manifest.json').read_text())
    assert all(not set(c)&{'train','select_design','push','publish','raw','validate_raw_local'} for c in cfg['commands'])
    assert any(c[:2]==['scripts/run_submission_revision.py','regenerate'] for c in cfg['commands'])
