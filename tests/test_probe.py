"""Tests for the probe module — fully offline via monkeypatch."""

import io

import pandas as pd
import pytest

from vitaldb_audit import probe


class FakeRaw:
    def __init__(self, payload):
        self._payload = payload

    def read(self, n, decode_content=True):
        return self._payload[:n]


class FakeResponse:
    def __init__(self, payload):
        self.raw = FakeRaw(payload)

    def raise_for_status(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _numeric_payload():
    """Simulate Solar8000/HR at 0.5 Hz (dt = 2.0 s)."""
    lines = ["Time,Solar8000/HR"]
    for i in range(200):
        lines.append(f"{i * 2.0:.1f},{70 + (i % 10)}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _waveform_payload():
    """Simulate SNUADC/ECG_II at 500 Hz (dt = 0.002 s)."""
    lines = ["Time,SNUADC/ECG_II"]
    for i in range(2000):
        lines.append(f"{i * 0.002:.4f},{0.5 + (i % 100) * 0.01}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_measures_half_hertz_for_solar8000_numeric(monkeypatch):
    monkeypatch.setattr(
        probe.requests, "get", lambda *a, **k: FakeResponse(_numeric_payload())
    )
    result = probe.measure_track_rate("tid1", "Solar8000/HR", max_bytes=100_000)
    assert result["observed_sampling_hz"] == pytest.approx(0.5, abs=1e-6)
    assert result["median_dt_s"] == pytest.approx(2.0, abs=1e-6)
    assert result["observed_track_kind"] == "numeric"
    assert result["sampling_rate_source"] == "empirical_probe"
    assert result["error"] is None


def test_measures_500_hertz_for_snuadc_waveform(monkeypatch):
    monkeypatch.setattr(
        probe.requests, "get", lambda *a, **k: FakeResponse(_waveform_payload())
    )
    result = probe.measure_track_rate("tid2", "SNUADC/ECG_II", max_bytes=100_000)
    assert result["observed_sampling_hz"] == pytest.approx(500.0, abs=1e-3)
    assert result["observed_track_kind"] == "waveform"
    assert result["sampling_rate_source"] == "empirical_probe"


def test_truncated_final_line_is_discarded(monkeypatch):
    payload = b"Time,Solar8000/HR\n1.818,88\n3.817,87\n5.818,8"
    monkeypatch.setattr(
        probe.requests, "get", lambda *a, **k: FakeResponse(payload)
    )
    result = probe.measure_track_rate("tid3", "Solar8000/HR", max_bytes=100_000)
    assert result["n_rows_read"] == 2
    assert result["observed_sampling_hz"] == pytest.approx(0.5, abs=0.01)


def test_error_is_captured_not_raised(monkeypatch):
    def boom(*args, **kwargs):
        raise probe.requests.RequestException("connection reset")

    monkeypatch.setattr(probe.requests, "get", boom)
    result = probe.measure_track_rate("tid4", "Solar8000/HR", max_bytes=1000)
    assert result["observed_sampling_hz"] is None
    assert "connection reset" in result["error"]
    assert result["sampling_rate_source"] == "empirical_probe"


def test_probe_cases_covers_requested_case_track_pairs(monkeypatch, trks_df):
    monkeypatch.setattr(
        probe.requests, "get", lambda *a, **k: FakeResponse(_numeric_payload())
    )
    out = probe.probe_cases(trks_df, [1, 2], ["Solar8000/HR", "Solar8000/BT"])
    assert len(out) == 4
    assert set(out["caseid"]) == {1, 2}
    assert all(out["sampling_rate_source"] == "empirical_probe")


def test_probe_cases_includes_caseid_column(monkeypatch, trks_df):
    monkeypatch.setattr(
        probe.requests, "get", lambda *a, **k: FakeResponse(_numeric_payload())
    )
    out = probe.probe_cases(trks_df, [1], ["Solar8000/HR"])
    assert "caseid" in out.columns
    assert out.iloc[0]["caseid"] == 1
