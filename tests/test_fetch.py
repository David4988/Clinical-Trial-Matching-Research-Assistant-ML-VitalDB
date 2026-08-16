"""Tests for the fetch module — fully offline via monkeypatch."""

import json
from pathlib import Path

import pandas as pd
import pytest

from vitaldb_audit import fetch


def test_read_raw_csv_strips_the_utf8_bom(tmp_path):
    path = tmp_path / "cases.csv"
    path.write_bytes("\ufeffcaseid,age\n1,77\n".encode("utf-8-sig"))
    df = fetch.read_raw_csv(path)
    assert list(df.columns) == ["caseid", "age"]
    assert df["caseid"].tolist() == [1]


def test_download_endpoint_writes_bytes_verbatim(tmp_path, monkeypatch):
    payload = b"caseid,tname,tid\n1,Solar8000/HR,abc\n"

    class FakeResponse:
        content = payload
        status_code = 200

        def raise_for_status(self):
            return None

    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: FakeResponse())
    dest = tmp_path / "trks.csv"
    record = fetch.download_endpoint("trks", "https://example/trks", dest)
    assert dest.read_bytes() == payload
    assert record["cached"] is False
    assert record["n_bytes"] == len(payload)
    assert len(record["sha256"]) == 64


def test_download_endpoint_uses_cache_when_file_exists(tmp_path, monkeypatch):
    dest = tmp_path / "trks.csv"
    dest.write_bytes(b"caseid,tname,tid\n1,Solar8000/HR,abc\n")
    # get should NOT be called
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call")))
    record = fetch.download_endpoint("trks", "https://example/trks", dest)
    assert record["cached"] is True


def test_download_endpoint_refetches_when_forced(tmp_path, monkeypatch):
    dest = tmp_path / "trks.csv"
    dest.write_bytes(b"stale")

    class FakeResponse:
        content = b"caseid,tname,tid\n1,Solar8000/HR,abc\n"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: FakeResponse())
    record = fetch.download_endpoint(
        "trks", "https://example/trks", dest, force_refresh=True
    )
    assert record["cached"] is False
    assert dest.read_bytes() != b"stale"


def test_download_endpoint_raises_clear_error_on_empty_payload(tmp_path, monkeypatch):
    class FakeResponse:
        content = b""

        def raise_for_status(self):
            return None

    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: FakeResponse())
    with pytest.raises(fetch.FetchError, match="empty payload"):
        fetch.download_endpoint("trks", "https://example/trks", tmp_path / "trks.csv")
