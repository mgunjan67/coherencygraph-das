import pytest
import execution_guard


def test_field_scoring_requires_external_watchdog(monkeypatch):
    monkeypatch.delenv("CLOSEOUT_WATCHDOG_PID",raising=False)
    with pytest.raises(PermissionError):execution_guard.require_owned_watchdog()


def test_windows_redirector_one_intermediate_parent_is_allowed(monkeypatch):
    class Process:
        def parents(self):
            return [type("P",(),{"pid":12})(),type("P",(),{"pid":34})()]
    monkeypatch.setenv("CLOSEOUT_WATCHDOG_PID","34")
    monkeypatch.setattr(execution_guard.psutil,"Process",Process)
    execution_guard.require_owned_watchdog()


def test_unrelated_parent_is_rejected(monkeypatch):
    class Process:
        def parents(self):return [type("P",(),{"pid":12})()]
    monkeypatch.setenv("CLOSEOUT_WATCHDOG_PID","34")
    monkeypatch.setattr(execution_guard.psutil,"Process",Process)
    with pytest.raises(PermissionError):execution_guard.require_owned_watchdog()
