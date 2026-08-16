"""Opt-in empirical sampling-rate probe for selected cases.

IMPORTANT TERMINOLOGY:
    The /trks endpoint contains NO sampling-rate metadata.  Every measurement
    produced by this module is an *empirical observation* from actual returned
    samples, NOT a metadata-provided "native rate".

    Fields use ``observed_sampling_hz`` and ``sampling_rate_source = "empirical_probe"``
    to make this origin explicit.

Uses streaming requests with ``decode_content=True`` so gzip-encoded waveform
tracks inflate correctly.  A plain HTTP Range request cannot be used: a partial
gzip stream is not independently decodable.
"""

import io
import logging

import pandas as pd
import requests

from vitaldb_audit import config

logger = logging.getLogger("vitaldb_audit.probe")

REQUEST_TIMEOUT = 120

# Tracks with an observed rate above this threshold are classified as waveforms.
# This is an empirical observation, not a metadata-wide guarantee.
WAVEFORM_HZ_THRESHOLD = 10.0


def read_track_head(tid: str, max_bytes: int = config.PROBE_MAX_BYTES) -> str:
    """Return the first ``max_bytes`` decoded bytes of a track CSV.

    Uses a streaming read with ``decode_content=True`` so gzip-encoded waveform
    tracks inflate correctly.  A plain HTTP Range request cannot be used: a
    partial gzip stream is not independently decodable.
    """
    url = f"{config.API_URL}/{tid}"
    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        raw = response.raw.read(max_bytes, decode_content=True)
    return raw.decode("utf-8", errors="replace")


def measure_track_rate(
    tid: str,
    tname: str,
    max_bytes: int = config.PROBE_MAX_BYTES,
) -> dict:
    """Derive a track's observed sampling rate from its first samples.

    Returns a record with ``observed_sampling_hz`` set to None and ``error``
    populated if anything goes wrong; probing one bad track must not abort the
    audit.

    All rate measurements are labelled ``sampling_rate_source: "empirical_probe"``
    to distinguish them from any hypothetical metadata-provided values (which
    do not exist in VitalDB's current API).
    """
    record = {
        "tname": tname,
        "tid": tid,
        "n_rows_read": 0,
        "median_dt_s": None,
        "observed_sampling_hz": None,
        "observed_track_kind": None,
        "n_bytes_read": 0,
        "sampling_rate_source": "empirical_probe",
        "error": None,
    }

    try:
        text = read_track_head(tid, max_bytes)
    except requests.RequestException as exc:
        record["error"] = f"request failed: {exc}"
        logger.warning("probe failed for %s (%s): %s", tname, tid, exc)
        return record

    record["n_bytes_read"] = len(text)

    # The prefix almost certainly ends mid-row; drop the partial last line.
    cutoff = text.rfind("\n")
    if cutoff <= 0:
        record["error"] = "response too short to contain a complete row"
        return record

    try:
        frame = pd.read_csv(io.StringIO(text[:cutoff + 1]))
    except Exception as exc:
        record["error"] = f"CSV parse error: {exc}"
        return record

    if "Time" not in frame.columns:
        record["error"] = f"no 'Time' column in response; got {list(frame.columns)}"
        return record

    record["n_rows_read"] = len(frame)
    deltas = frame["Time"].diff().dropna()
    deltas = deltas[deltas > 0]

    if deltas.empty:
        record["error"] = "no positive time deltas found"
        return record

    median_dt = float(deltas.median())
    observed_hz = 1.0 / median_dt
    record["median_dt_s"] = round(median_dt, 6)
    record["observed_sampling_hz"] = round(observed_hz, 2)
    record["observed_track_kind"] = (
        "waveform" if observed_hz >= WAVEFORM_HZ_THRESHOLD else "numeric"
    )
    logger.info(
        "%s: %.2f Hz (median dt %.6f s, %d rows) [empirical probe]",
        tname, observed_hz, median_dt, len(frame),
    )
    return record


def probe_cases(
    trks: pd.DataFrame,
    caseids: list[int],
    track_names: list[str],
) -> pd.DataFrame:
    """Measure observed sampling rates for the given tracks in the given cases only.

    Results are labelled with sampling_rate_source="empirical_probe" to make
    clear that these are not metadata-provided values.
    """
    wanted = trks[
        trks["caseid"].isin(caseids) & trks["tname"].isin(track_names)
    ].sort_values(["caseid", "tname"])

    logger.info(
        "probing %d case/track pairs across %d cases (max %d bytes each)",
        len(wanted), len(caseids), config.PROBE_MAX_BYTES,
    )

    results = []
    for _, row in wanted.iterrows():
        record = measure_track_rate(row["tid"], row["tname"])
        record["caseid"] = int(row["caseid"])
        results.append(record)

    frame = pd.DataFrame(results)
    if not frame.empty:
        # Reorder columns for clarity
        col_order = [
            "caseid", "tname", "tid", "n_rows_read", "n_bytes_read",
            "median_dt_s", "observed_sampling_hz", "observed_track_kind",
            "sampling_rate_source", "error",
        ]
        frame = frame[[c for c in col_order if c in frame.columns]]

    n_ok = len(frame[frame["error"].isna()]) if not frame.empty else 0
    n_err = len(frame) - n_ok
    logger.info("probe complete: %d successful, %d errors", n_ok, n_err)
    return frame
