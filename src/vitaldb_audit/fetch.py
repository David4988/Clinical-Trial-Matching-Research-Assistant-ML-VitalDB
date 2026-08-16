"""Retrieval and verbatim persistence of VitalDB metadata.

Only the three documented metadata endpoints are touched here.  No signal data
is downloaded by this module under any circumstances.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from vitaldb_audit import config

logger = logging.getLogger("vitaldb_audit.fetch")

REQUEST_TIMEOUT = 300


class FetchError(RuntimeError):
    """Raised when a metadata download fails or returns bad data."""


@dataclass(frozen=True)
class RawTables:
    """The three raw metadata tables, parsed into DataFrames."""
    cases: pd.DataFrame
    trks: pd.DataFrame
    labs: pd.DataFrame


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read a CSV saved verbatim from the API.

    Uses encoding ``utf-8-sig`` to strip the BOM that /cases carries on its
    first column header.
    """
    return pd.read_csv(path, encoding="utf-8-sig")


def download_endpoint(
    name: str,
    url: str,
    dest: Path,
    force_refresh: bool = False,
) -> dict:
    """Download a single metadata endpoint and persist its raw bytes.

    Returns a manifest record with the file's sha256, size, and cache status.
    """
    cached = False

    if dest.exists() and not force_refresh:
        payload = dest.read_bytes()
        logger.info("using cached %s (%d bytes) at %s", name, len(payload), dest)
        cached = True
    else:
        logger.info("downloading %s from %s", name, url)
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FetchError(f"failed to download {name} from {url}: {exc}") from exc

        payload = response.content
        if not payload:
            raise FetchError(f"{name} returned an empty payload from {url}")

        dest.write_bytes(payload)
        logger.info("saved %s (%d bytes) to %s", name, len(payload), dest)
        cached = False

    return {
        "name": name,
        "url": url,
        "path": str(dest),
        "n_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "cached": cached,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": config.DATASET_VERSION,
    }


def fetch_metadata(force_refresh: bool = False) -> tuple[RawTables, list[dict]]:
    """Retrieve cases, trks and labs; persist raw bytes and a manifest.

    Returns the parsed tables alongside the manifest records.
    """
    config.ensure_dirs()

    records = []
    frames = {}
    for name, url in config.ENDPOINTS.items():
        dest = config.RAW_DIR / f"{name}.csv"
        record = download_endpoint(name, url, dest, force_refresh=force_refresh)
        try:
            frame = read_raw_csv(dest)
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            raise FetchError(f"failed to parse {name} from {dest}: {exc}") from exc

        record["n_rows"], record["n_cols"] = frame.shape
        logger.info("%s parsed to shape %s", name, frame.shape)
        records.append(record)
        frames[name] = frame

    manifest_path = config.RAW_DIR / "fetch_manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("wrote fetch manifest to %s", manifest_path)

    return RawTables(cases=frames["cases"], trks=frames["trks"], labs=frames["labs"]), records
