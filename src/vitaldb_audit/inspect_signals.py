"""Selected-case signal inspection utility (inspection stage only).

Scope and deliberate non-goals
------------------------------
This module is a *read-and-look* tool for a handful of hand-picked cases.  It
is intentionally NOT wired into the reconnaissance pipeline
(fetch -> profile -> classify -> select -> probe) and NOT wired into the main
application.  It performs no anomaly detection and no modelling.

Two data-handling rules are enforced here and stated in every emitted record:

    NO INTERPOLATION   Missing samples are never filled.  Gaps are measured
                       and reported as gaps.  The only NaNs this module ever
                       *inserts* are plot-level line breaks (see
                       ``_break_across_gaps``), which exist precisely so that
                       matplotlib does NOT draw a straight line across a gap
                       and imply data that was never recorded.

    NO RESAMPLING      Samples are used on their original irregular time base.
                       The sampling interval is *measured*, never imposed.

Both rules are echoed into ``TrackSummary.interpolation`` / ``.resampling`` so
that a downstream reader of the JSON cannot mistake the provenance.

Terminology follows :mod:`vitaldb_audit.probe`: every rate here is an
*empirical observation* from returned samples.  The /trks endpoint carries no
sampling-rate metadata.
"""

import io
import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path

import pandas as pd
import requests

from vitaldb_audit import config

logger = logging.getLogger("vitaldb_audit.inspect_signals")

REQUEST_TIMEOUT = 300

# ── Inspection parameters (all explicit; none inherited from the pipeline) ────

# The four core numeric signals under inspection.  All are Solar8000 numeric
# tracks observed at ~0.5 Hz, so a whole track is ~80 KB / ~8k rows.  That is
# small enough to load in full, which is why this module does not page or
# truncate: "a manageable amount" here is genuinely the entire track.
CORE_SIGNALS = [
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
    "Solar8000/ART_MBP",
    "Solar8000/RR_CO2",
]

# A refusal threshold, not a truncation strategy.  If a track exceeds this we
# skip it loudly rather than silently inspecting a biased prefix.  Waveform
# tracks (500 Hz) would trip this; the four core signals never do.
MAX_TRACK_BYTES = 20_000_000

# A forward time step larger than this multiple of the observed median step is
# classified as a gap.  3x is permissive enough to tolerate ordinary jitter in
# the ~2 s Solar8000 cadence while still catching real dropouts.
GAP_FACTOR = 3.0

SIGNAL_CACHE_DIR = config.RAW_DIR / "signals"
PLOT_DIR = config.RESULTS_DIR / "plots"

# Physiologically plausible display bands.  Used ONLY to draw a shaded
# reference band on plots and to raise a human-readable flag.  Nothing is
# filtered, clipped or corrected on the basis of these numbers.
PLAUSIBLE_RANGES = {
    "Solar8000/HR": (20.0, 200.0),
    "Solar8000/PLETH_SPO2": (70.0, 100.0),
    "Solar8000/ART_MBP": (30.0, 150.0),
    "Solar8000/RR_CO2": (4.0, 40.0),
}


class SignalLoadError(RuntimeError):
    """Raised when a track cannot be retrieved or parsed."""


@dataclass
class TrackSummary:
    """Everything observed about one (case, track) pair.

    ``available`` is False when the case simply does not carry the track; that
    is a metadata fact, not an error, and is reported rather than imputed.
    """

    caseid: int
    tname: str
    available: bool
    tid: str | None = None
    error: str | None = None

    # Volume
    n_rows: int = 0
    n_values_present: int = 0
    n_values_missing: int = 0
    pct_values_missing: float = 0.0

    # Time base (seconds relative to case start)
    first_time_s: float | None = None
    last_time_s: float | None = None
    observed_duration_min: float | None = None
    case_duration_min: float | None = None

    # Observed sampling interval
    median_dt_s: float | None = None
    min_dt_s: float | None = None
    max_dt_s: float | None = None
    observed_sampling_hz: float | None = None
    sampling_rate_source: str = "empirical_observation"

    # Gaps
    n_gaps: int = 0
    total_gap_s: float = 0.0
    largest_gap_s: float = 0.0
    pct_time_in_gaps: float = 0.0

    # Value statistics (computed on present values only; never imputed)
    value_min: float | None = None
    value_max: float | None = None
    value_median: float | None = None

    # Coverage of the expected timeline (casestart -> caseend)
    span_coverage_pct: float | None = None
    sample_coverage_pct: float | None = None

    # Provenance guarantees
    interpolation: str = "none"
    resampling: str = "none"

    # Human-readable flags
    flags: list[str] = field(default_factory=list)


# ── Loading ──────────────────────────────────────────────────────────────────


def _cache_path(caseid: int, tname: str) -> Path:
    """Local cache path for one track, mirroring the API payload verbatim."""
    safe = tname.replace("/", "__")
    return SIGNAL_CACHE_DIR / f"case_{caseid}" / f"{safe}.csv"


def load_track(
    tid: str,
    caseid: int,
    tname: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download (or reuse) one full track and parse it.

    The payload is cached verbatim on disk so that re-running the inspection
    is offline and byte-identical.  Waveform-sized payloads are refused rather
    than truncated, so a summary is never computed from a biased prefix.
    """
    dest = _cache_path(caseid, tname)

    if dest.exists() and not force_refresh:
        payload = dest.read_bytes()
        logger.info("using cached %s for case %d (%d bytes)", tname, caseid, len(payload))
    else:
        url = f"{config.API_URL}/{tid}"
        logger.info("downloading %s for case %d from %s", tname, caseid, url)
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SignalLoadError(f"request failed for {tname} (case {caseid}): {exc}") from exc

        payload = response.content
        if not payload:
            raise SignalLoadError(f"empty payload for {tname} (case {caseid})")
        if len(payload) > MAX_TRACK_BYTES:
            raise SignalLoadError(
                f"{tname} (case {caseid}) is {len(payload)} bytes, above the "
                f"{MAX_TRACK_BYTES}-byte inspection ceiling; refusing to inspect "
                f"a truncated prefix"
            )

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        logger.info("cached %s for case %d to %s", tname, caseid, dest)

    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise SignalLoadError(f"CSV parse error for {tname} (case {caseid}): {exc}") from exc

    if "Time" not in frame.columns:
        raise SignalLoadError(
            f"no 'Time' column for {tname} (case {caseid}); got {list(frame.columns)}"
        )
    if tname not in frame.columns:
        raise SignalLoadError(
            f"no '{tname}' column for case {caseid}; got {list(frame.columns)}"
        )

    return frame[["Time", tname]]


# ── Summarising ──────────────────────────────────────────────────────────────


def summarize_track(
    frame: pd.DataFrame,
    caseid: int,
    tname: str,
    tid: str,
    case_duration_min: float,
) -> TrackSummary:
    """Measure one track. Nothing is filled, smoothed, resampled or clipped."""
    summary = TrackSummary(
        caseid=caseid, tname=tname, available=True, tid=tid,
        case_duration_min=round(case_duration_min, 2),
    )

    times = pd.to_numeric(frame["Time"], errors="coerce")
    values = pd.to_numeric(frame[tname], errors="coerce")

    summary.n_rows = int(len(frame))
    if summary.n_rows == 0:
        summary.flags.append("track returned zero rows")
        return summary

    # Missingness is measured at two distinct levels, which are NOT the same
    # thing: a row that exists but carries no value, versus a stretch of time
    # where no row exists at all (a gap, measured below).
    summary.n_values_present = int(values.notna().sum())
    summary.n_values_missing = int(values.isna().sum())
    summary.pct_values_missing = round(
        summary.n_values_missing / summary.n_rows * 100, 2
    )

    valid_time = times.notna()
    if not valid_time.any():
        summary.flags.append("no parseable timestamps")
        return summary

    t = times[valid_time].to_numpy()
    summary.first_time_s = round(float(t[0]), 3)
    summary.last_time_s = round(float(t[-1]), 3)
    observed_span_s = float(t[-1] - t[0])
    summary.observed_duration_min = round(observed_span_s / 60.0, 2)

    deltas = pd.Series(t).diff().dropna()
    positive = deltas[deltas > 0]
    if positive.empty:
        summary.flags.append("no positive time deltas; cannot measure interval")
        return summary

    median_dt = float(positive.median())
    summary.median_dt_s = round(median_dt, 6)
    summary.min_dt_s = round(float(positive.min()), 6)
    summary.max_dt_s = round(float(positive.max()), 6)
    summary.observed_sampling_hz = round(1.0 / median_dt, 4) if median_dt > 0 else None

    # Gaps: forward steps materially larger than the observed cadence.
    gap_threshold = GAP_FACTOR * median_dt
    gaps = positive[positive > gap_threshold]
    summary.n_gaps = int(len(gaps))
    # Only the excess beyond one nominal step is genuinely "missing" time.
    summary.total_gap_s = round(float((gaps - median_dt).sum()), 3) if summary.n_gaps else 0.0
    summary.largest_gap_s = round(float(gaps.max()), 3) if summary.n_gaps else 0.0
    summary.pct_time_in_gaps = (
        round(summary.total_gap_s / observed_span_s * 100, 2)
        if observed_span_s > 0 else 0.0
    )

    present = values.dropna()
    if not present.empty:
        summary.value_min = round(float(present.min()), 3)
        summary.value_max = round(float(present.max()), 3)
        summary.value_median = round(float(present.median()), 3)

    # Coverage of the expected timeline, reported two ways because they answer
    # different questions and can disagree sharply:
    #   span    - does the track reach from case start to case end?
    #   sample  - are there as many samples as the cadence implies there should be?
    # A track can span the whole case yet be full of holes, or be dense yet
    # cover only the first half.
    case_duration_s = case_duration_min * 60.0
    if case_duration_s > 0:
        summary.span_coverage_pct = round(observed_span_s / case_duration_s * 100, 2)
        expected_samples = case_duration_s / median_dt
        if expected_samples > 0:
            summary.sample_coverage_pct = round(
                summary.n_values_present / expected_samples * 100, 2
            )

    _add_flags(summary)
    return summary


def _add_flags(summary: TrackSummary) -> None:
    """Attach human-readable observations. Diagnostic only; nothing is altered."""
    lo, hi = PLAUSIBLE_RANGES.get(summary.tname, (None, None))
    if lo is not None:
        if summary.value_min is not None and summary.value_min < lo:
            summary.flags.append(
                f"minimum {summary.value_min} below plausible {lo} "
                f"(likely sensor dropout/artifact, retained unmodified)"
            )
        if summary.value_max is not None and summary.value_max > hi:
            summary.flags.append(
                f"maximum {summary.value_max} above plausible {hi} "
                f"(likely artifact, retained unmodified)"
            )
    if summary.value_min == 0.0:
        summary.flags.append("contains exact zeros (sensor-disconnect sentinel in VitalDB)")
    if summary.span_coverage_pct is not None and summary.span_coverage_pct < 80.0:
        summary.flags.append(
            f"spans only {summary.span_coverage_pct}% of the case timeline"
        )
    if summary.sample_coverage_pct is not None and summary.sample_coverage_pct < 80.0:
        summary.flags.append(
            f"sample density {summary.sample_coverage_pct}% of expected"
        )
    if summary.n_gaps > 0:
        summary.flags.append(
            f"{summary.n_gaps} gap(s), largest {summary.largest_gap_s}s"
        )


# ── Plotting ─────────────────────────────────────────────────────────────────


def _break_across_gaps(
    frame: pd.DataFrame, tname: str, median_dt: float | None
) -> pd.DataFrame:
    """Insert NaN rows at gap boundaries so plotted lines break instead of
    spanning a dropout.

    This is the inverse of interpolation: it prevents matplotlib from drawing a
    straight segment across missing time and thereby implying samples that were
    never recorded.  The returned frame is for plotting only and is never used
    for statistics.
    """
    if median_dt is None or median_dt <= 0 or len(frame) < 2:
        return frame

    times = pd.to_numeric(frame["Time"], errors="coerce")
    gap_after = times.diff().shift(-1) > GAP_FACTOR * median_dt
    if not gap_after.any():
        return frame

    breaks = pd.DataFrame({
        "Time": times[gap_after] + median_dt / 2.0,
        tname: pd.NA,
    })
    combined = pd.concat([frame, breaks], ignore_index=True)
    return combined.sort_values("Time", kind="stable").reset_index(drop=True)


def plot_track(
    frame: pd.DataFrame, summary: TrackSummary, out_dir: Path
) -> Path:
    """One plot for one signal in one case."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_frame = _break_across_gaps(frame, summary.tname, summary.median_dt_s)
    minutes = pd.to_numeric(plot_frame["Time"], errors="coerce") / 60.0
    values = pd.to_numeric(plot_frame[summary.tname], errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(minutes, values, linewidth=0.8, color="#1f77b4")

    lo, hi = PLAUSIBLE_RANGES.get(summary.tname, (None, None))
    if lo is not None:
        ax.axhspan(lo, hi, color="#2ca02c", alpha=0.07,
                   label=f"plausible band {lo:g}-{hi:g}")

    if summary.case_duration_min:
        ax.axvline(summary.case_duration_min, color="#d62728", linestyle="--",
                   linewidth=1.0, label=f"case end ({summary.case_duration_min:g} min)")

    ax.set_title(
        f"Case {summary.caseid} — {summary.tname}\n"
        f"{summary.n_rows} rows @ ~{summary.observed_sampling_hz} Hz "
        f"(median dt {summary.median_dt_s}s) | "
        f"span {summary.span_coverage_pct}% / density {summary.sample_coverage_pct}% "
        f"| {summary.n_gaps} gap(s) | raw, no interpolation, no resampling",
        fontsize=9,
    )
    ax.set_xlabel("Time from case start (minutes)")
    ax.set_ylabel(summary.tname.split("/")[-1])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"case_{summary.caseid}__{summary.tname.replace('/', '__')}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_case_synchronized(
    caseid: int,
    frames: dict[str, pd.DataFrame],
    summaries: dict[str, TrackSummary],
    signals: list[str],
    case_duration_min: float,
    out_dir: Path,
) -> Path:
    """One stacked, shared-x plot showing all four signals on a common clock.

    Unavailable signals keep their row so the panel geometry is identical
    across cases and an absence is visible rather than silently collapsed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        len(signals), 1, figsize=(13, 2.4 * len(signals)), sharex=True
    )
    if len(signals) == 1:
        axes = [axes]

    for ax, tname in zip(axes, signals):
        summary = summaries.get(tname)
        frame = frames.get(tname)

        if summary is None or not summary.available or frame is None:
            ax.text(0.5, 0.5, f"{tname} — NOT RECORDED FOR THIS CASE",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#d62728", weight="bold")
            ax.set_ylabel(tname.split("/")[-1])
            ax.set_yticks([])
            ax.grid(alpha=0.3)
            continue

        plot_frame = _break_across_gaps(frame, tname, summary.median_dt_s)
        minutes = pd.to_numeric(plot_frame["Time"], errors="coerce") / 60.0
        values = pd.to_numeric(plot_frame[tname], errors="coerce")
        ax.plot(minutes, values, linewidth=0.8, color="#1f77b4")

        lo, hi = PLAUSIBLE_RANGES.get(tname, (None, None))
        if lo is not None:
            ax.axhspan(lo, hi, color="#2ca02c", alpha=0.07)

        ax.axvline(case_duration_min, color="#d62728", linestyle="--", linewidth=1.0)
        ax.set_ylabel(tname.split("/")[-1])
        ax.grid(alpha=0.3)
        ax.text(
            0.005, 0.93,
            f"n={summary.n_rows} | ~{summary.observed_sampling_hz}Hz | "
            f"span {summary.span_coverage_pct}% | density {summary.sample_coverage_pct}% | "
            f"gaps {summary.n_gaps}",
            transform=ax.transAxes, fontsize=7, va="top", color="#444444",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85),
        )

    axes[-1].set_xlabel("Time from case start (minutes)")
    axes[0].set_title(
        f"Case {caseid} — synchronized core panel "
        f"(case duration {case_duration_min:g} min; dashed line = case end)\n"
        f"Common clock, original irregular time base — no interpolation, no resampling; "
        f"line breaks are real gaps",
        fontsize=10,
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"case_{caseid}__synchronized_panel.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ── Orchestration ────────────────────────────────────────────────────────────


def inspect_case(
    caseid: int,
    cases: pd.DataFrame,
    trks: pd.DataFrame,
    signals: list[str] = None,
    force_refresh: bool = False,
) -> dict:
    """Load, measure and plot every core signal for one case."""
    signals = list(signals or CORE_SIGNALS)

    case_row = cases[cases["caseid"] == caseid]
    if case_row.empty:
        raise SignalLoadError(f"caseid {caseid} not present in the cases table")
    row = case_row.iloc[0]
    case_duration_min = float((row["caseend"] - row["casestart"]) / 60.0)

    available = (
        trks[trks["caseid"] == caseid]
        .set_index("tname")["tid"]
        .to_dict()
    )

    out_dir = PLOT_DIR / f"case_{caseid}"
    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, TrackSummary] = {}
    plot_paths: dict[str, str] = {}

    for tname in signals:
        tid = available.get(tname)
        if tid is None:
            logger.warning("case %d does not carry %s", caseid, tname)
            summaries[tname] = TrackSummary(
                caseid=caseid, tname=tname, available=False,
                case_duration_min=round(case_duration_min, 2),
                error="track not present in /trks for this case",
                flags=["track absent from this case (metadata fact, not a load failure)"],
            )
            continue

        try:
            frame = load_track(tid, caseid, tname, force_refresh=force_refresh)
        except SignalLoadError as exc:
            logger.warning("failed to load %s for case %d: %s", tname, caseid, exc)
            summaries[tname] = TrackSummary(
                caseid=caseid, tname=tname, available=False, tid=tid,
                case_duration_min=round(case_duration_min, 2), error=str(exc),
            )
            continue

        summary = summarize_track(frame, caseid, tname, tid, case_duration_min)
        frames[tname] = frame
        summaries[tname] = summary
        plot_paths[tname] = str(plot_track(frame, summary, out_dir))

    sync_path = plot_case_synchronized(
        caseid, frames, summaries, signals, case_duration_min, out_dir
    )

    return {
        "caseid": caseid,
        "case_duration_min": round(case_duration_min, 2),
        "signals": {k: asdict(v) for k, v in summaries.items()},
        "plots": plot_paths,
        "synchronized_plot": str(sync_path),
    }


def inspect_cases(
    caseids: list[int],
    cases: pd.DataFrame,
    trks: pd.DataFrame,
    signals: list[str] = None,
    force_refresh: bool = False,
) -> dict:
    """Inspect several cases and persist one combined JSON record."""
    reports = [
        inspect_case(cid, cases, trks, signals=signals, force_refresh=force_refresh)
        for cid in caseids
    ]
    bundle = {
        "dataset_version": config.DATASET_VERSION,
        "signals_inspected": list(signals or CORE_SIGNALS),
        "gap_factor": GAP_FACTOR,
        "interpolation": "none",
        "resampling": "none",
        "load_strategy": "full track download (numeric tracks only, ~80KB each)",
        "cases": reports,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.RESULTS_DIR / "signal_inspection.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    logger.info("wrote inspection bundle to %s", out)
    return bundle
