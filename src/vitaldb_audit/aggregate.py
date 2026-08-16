"""Fixed-width monitoring-window aggregation for selected cases.

Purpose
-------
Answer one question: does collapsing ~0.5 Hz numeric signals into fixed
monitoring windows preserve enough physiological information while making the
data easier to work with?  This module produces the aggregation and the
evidence; it draws no conclusions and detects nothing.

Deliberate non-goals: no anomaly detection, no modelling, no feature
engineering beyond the window statistics below, no waveform (500 Hz) tracks.
This module is separate from the reconnaissance pipeline
(fetch -> profile -> classify -> select -> probe) and imports from it only to
reuse the cached signal loader.

Data-handling rules, enforced and echoed into every emitted record
------------------------------------------------------------------
NO INTERPOLATION / NO FILL
    Windows are summarised from whatever observations genuinely fall inside
    them.  Nothing is interpolated, forward-filled, backward-filled or
    fabricated.  A window with 40% of its expected observations reports
    coverage_percent = 40 and computes its statistics from that 40%.

EMPTY WINDOWS SURVIVE
    A window with zero observations is retained with null statistics,
    observation_count = 0, coverage_percent = 0 and data_unavailable = True.
    It is never silently dropped, because dropping it would make a monitoring
    outage look like continuous data.

EXACT ZEROS ARE OBSERVATIONS
    RR_CO2 = 0 is a real recorded value and is included in every statistic.
    It is counted separately in ``rr_co2_zero_count`` so the decision of
    whether it is apnoea or a disconnected sensor stays open and reversible.

EXPECTED COUNTS ARE DERIVED, NOT GUESSED
    expected_observation_count = window_seconds / EXPECTED_INTERVAL_S.  It is
    never inferred from how many rows happen to be present, which would make
    coverage trivially 100% by construction.
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from vitaldb_audit import config, inspect_signals

logger = logging.getLogger("vitaldb_audit.aggregate")

# ── Window configuration ─────────────────────────────────────────────────────

# Default monitoring window.  Every function takes window_minutes as an
# argument; this constant is only the default, never an assumption baked into
# the arithmetic.
WINDOW_MINUTES = 10.0

# The live probe empirically observed ~0.5 Hz on these Solar8000 numeric
# tracks, i.e. one observation per ~2 s.  Expected observation counts are
# derived from this interval.
EXPECTED_INTERVAL_S = 2.0

# Reused verbatim from the inspection stage so gap detection stays consistent
# across checkpoints: a forward step above 3x the observed median dt is a gap.
GAP_FACTOR = inspect_signals.GAP_FACTOR

# ── Signal configuration ─────────────────────────────────────────────────────

# Per-signal statistic sets.
#
# SpO2 std CAVEAT: SpO2 is a ceiling-saturated variable that sits at 100 for
# most of a case, so its dispersion is partly a dropout artifact rather than
# physiology.  The inspection stage therefore originally omitted it.  It is
# emitted now because the v1 feature schema consumes it, but a model that finds
# spo2_std highly informative should be checked against spo2_coverage_pct
# before that finding is believed.
SIGNAL_SPECS: dict[str, dict] = {
    "Solar8000/HR": {
        "short": "hr",
        "stats": ("mean", "min", "max", "std"),
        "required": True,
    },
    "Solar8000/PLETH_SPO2": {
        "short": "spo2",
        "stats": ("mean", "min", "max", "std"),
        "required": True,
    },
    "Solar8000/RR_CO2": {
        "short": "rr_co2",
        "stats": ("mean", "min", "max", "std"),
        "required": True,
    },
    "Solar8000/ART_MBP": {
        "short": "art_mbp",
        "stats": ("mean", "min", "max", "std"),
        "required": False,
    },
}

CORE_SIGNALS = [t for t, s in SIGNAL_SPECS.items() if s["required"]]
OPTIONAL_SIGNALS = [t for t, s in SIGNAL_SPECS.items() if not s["required"]]
ALL_SIGNALS = CORE_SIGNALS + OPTIONAL_SIGNALS

AGGREGATED_DIR = config.RESULTS_DIR / "aggregated"
PLOT_DIR = config.RESULTS_DIR / "plots" / "aggregation"

# Coverage at or below this fraction marks a window as low-coverage in the
# report.  A reporting threshold only: nothing is filtered on it.
LOW_COVERAGE_FRACTION = 0.5


# ── Window construction ──────────────────────────────────────────────────────


def build_window_edges(
    total_seconds: float, window_minutes: float = WINDOW_MINUTES
) -> list[tuple[float, float]]:
    """Non-overlapping [start, end) windows tiling ``total_seconds``.

    Windows are anchored at t=0 (case start), not at the first observation, so
    that every signal in a case shares one window grid and windows are
    comparable across cases.  The final window is truncated to the case end
    rather than extended, so its expected observation count is scaled to its
    real width instead of over-stating what was possible to record.
    """
    if window_minutes <= 0:
        raise ValueError("window_minutes must be positive")
    if total_seconds <= 0:
        return []

    width = window_minutes * 60.0
    n = int(math.ceil(total_seconds / width))
    edges = []
    for i in range(n):
        start = i * width
        end = min((i + 1) * width, total_seconds)
        edges.append((start, end))
    return edges


def _stat_or_none(values: pd.Series, stat: str) -> float | None:
    """Compute one statistic, or None when it is genuinely undefined.

    Standard deviation of a single observation is undefined (not zero), and is
    reported as None so that a one-sample window is never mistaken for a
    perfectly stable one.
    """
    if values.empty:
        return None
    if stat == "std":
        if len(values) < 2:
            return None
        return round(float(values.std(ddof=1)), 4)
    value = getattr(values, stat)()
    if pd.isna(value):
        return None
    return round(float(value), 4)


def summarize_window(
    times: np.ndarray,
    values: np.ndarray,
    window_start: float,
    window_end: float,
    stats: tuple[str, ...],
    median_dt_s: float | None,
    short: str,
) -> dict:
    """Summarise one signal inside one window. No filling of any kind."""
    width_s = window_end - window_start
    expected = width_s / EXPECTED_INTERVAL_S

    record = {
        f"{short}_observation_count": 0,
        f"{short}_expected_observation_count": round(expected, 2),
        f"{short}_coverage_fraction": 0.0,
        f"{short}_coverage_percent": 0.0,
        f"{short}_first_observation_time": None,
        f"{short}_last_observation_time": None,
        f"{short}_largest_gap_seconds": 0.0,
        f"{short}_total_gap_seconds": 0.0,
        f"{short}_data_unavailable": True,
    }
    for stat in stats:
        record[f"{short}_{stat}"] = None
    if short == "rr_co2":
        record["rr_co2_zero_count"] = 0

    in_window = (times >= window_start) & (times < window_end)
    if not in_window.any():
        # Retained, not dropped: an outage must stay visible downstream.
        return record

    w_times = times[in_window]
    w_values = pd.Series(values[in_window]).astype(float)
    present = w_values.dropna()

    record[f"{short}_observation_count"] = int(len(present))
    record[f"{short}_first_observation_time"] = round(float(w_times[0]), 3)
    record[f"{short}_last_observation_time"] = round(float(w_times[-1]), 3)

    fraction = (len(present) / expected) if expected > 0 else 0.0
    record[f"{short}_coverage_fraction"] = round(fraction, 4)
    record[f"{short}_coverage_percent"] = round(fraction * 100, 2)
    record[f"{short}_data_unavailable"] = len(present) == 0

    # Gaps inside the window, using the whole-track median dt so the threshold
    # does not drift window to window.
    if median_dt_s and len(w_times) >= 2:
        deltas = np.diff(w_times)
        deltas = deltas[deltas > 0]
        gaps = deltas[deltas > GAP_FACTOR * median_dt_s]
        if gaps.size:
            record[f"{short}_largest_gap_seconds"] = round(float(gaps.max()), 3)
            # Only the excess beyond one nominal step is genuinely missing time.
            record[f"{short}_total_gap_seconds"] = round(
                float((gaps - median_dt_s).sum()), 3
            )

    if short == "rr_co2":
        # Exact zeros are retained as valid observations and also counted.
        record["rr_co2_zero_count"] = int((present == 0).sum())

    for stat in stats:
        record[f"{short}_{stat}"] = _stat_or_none(present, stat)

    return record


def aggregate_case(
    caseid: int,
    cases: pd.DataFrame,
    trks: pd.DataFrame,
    window_minutes: float = WINDOW_MINUTES,
    signals: list[str] = None,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate every configured signal for one case onto a shared window grid.

    Returns the per-window frame and a metadata dict describing what was found.
    Signals absent from the case are reported as unavailable rather than
    silently skipped.
    """
    signals = list(signals or ALL_SIGNALS)

    case_row = cases[cases["caseid"] == caseid]
    if case_row.empty:
        raise ValueError(f"caseid {caseid} not present in the cases table")
    row = case_row.iloc[0]
    case_duration_s = float(row["caseend"] - row["casestart"])

    available_tids = trks[trks["caseid"] == caseid].set_index("tname")["tid"].to_dict()

    loaded: dict[str, dict] = {}
    signal_meta: dict[str, dict] = {}

    for tname in signals:
        spec = SIGNAL_SPECS[tname]
        tid = available_tids.get(tname)
        if tid is None:
            signal_meta[tname] = {
                "available": False,
                "required": spec["required"],
                "reason": "track not present in /trks for this case",
            }
            logger.info("case %d: %s unavailable", caseid, tname)
            continue

        frame = inspect_signals.load_track(tid, caseid, tname)
        times = pd.to_numeric(frame["Time"], errors="coerce")
        values = pd.to_numeric(frame[tname], errors="coerce")
        valid = times.notna()
        t = times[valid].to_numpy(dtype=float)
        v = values[valid].to_numpy(dtype=float)

        deltas = pd.Series(t).diff().dropna()
        positive = deltas[deltas > 0]
        median_dt = float(positive.median()) if not positive.empty else None

        loaded[tname] = {"times": t, "values": v, "median_dt": median_dt}
        signal_meta[tname] = {
            "available": True,
            "required": spec["required"],
            "n_raw_rows": int(len(t)),
            "median_dt_s": round(median_dt, 6) if median_dt else None,
            "observed_sampling_hz": round(1.0 / median_dt, 4) if median_dt else None,
        }

    # The grid spans the case; if a track somehow runs past caseend, extend so
    # no observation is silently discarded.
    max_observed = max(
        (float(d["times"][-1]) for d in loaded.values() if len(d["times"])),
        default=0.0,
    )
    total_seconds = max(case_duration_s, max_observed)
    edges = build_window_edges(total_seconds, window_minutes)

    rows = []
    for index, (start, end) in enumerate(edges):
        record = {
            "caseid": caseid,
            "window_index": index,
            "window_start_s": round(start, 3),
            "window_end_s": round(end, 3),
            "window_start_min": round(start / 60.0, 4),
            "window_end_min": round(end / 60.0, 4),
            "window_width_min": round((end - start) / 60.0, 4),
        }
        for tname in signals:
            spec = SIGNAL_SPECS[tname]
            short = spec["short"]
            data = loaded.get(tname)
            if data is None:
                # Signal absent from the case entirely: emit the same columns
                # so every case CSV has an identical schema.
                width_s = end - start
                record[f"{short}_observation_count"] = 0
                record[f"{short}_expected_observation_count"] = round(
                    width_s / EXPECTED_INTERVAL_S, 2
                )
                record[f"{short}_coverage_fraction"] = 0.0
                record[f"{short}_coverage_percent"] = 0.0
                record[f"{short}_first_observation_time"] = None
                record[f"{short}_last_observation_time"] = None
                record[f"{short}_largest_gap_seconds"] = 0.0
                record[f"{short}_total_gap_seconds"] = 0.0
                record[f"{short}_data_unavailable"] = True
                for stat in spec["stats"]:
                    record[f"{short}_{stat}"] = None
                if short == "rr_co2":
                    record["rr_co2_zero_count"] = 0
                continue

            record.update(summarize_window(
                data["times"], data["values"], start, end,
                spec["stats"], data["median_dt"], short,
            ))

        # Window-level verdict across the CORE signals only; an absent optional
        # signal must not make an otherwise well-monitored window look empty.
        core_shorts = [
            SIGNAL_SPECS[t]["short"] for t in signals if SIGNAL_SPECS[t]["required"]
        ]
        record["window_data_unavailable"] = all(
            record[f"{s}_data_unavailable"] for s in core_shorts
        )
        record["core_signals_present"] = sum(
            0 if record[f"{s}_data_unavailable"] else 1 for s in core_shorts
        )
        rows.append(record)

    frame = pd.DataFrame(rows)
    meta = {
        "caseid": caseid,
        "case_duration_s": round(case_duration_s, 3),
        "case_duration_min": round(case_duration_s / 60.0, 2),
        "window_minutes": window_minutes,
        "expected_interval_s": EXPECTED_INTERVAL_S,
        "gap_factor": GAP_FACTOR,
        "n_windows": len(frame),
        "signals": signal_meta,
        "interpolation": "none",
        "fill": "none",
        "resampling": "windowed aggregation only; no value resampling",
    }
    logger.info(
        "case %d: %d windows of %.1f min from %.1f min of case timeline",
        caseid, len(frame), window_minutes, case_duration_s / 60.0,
    )
    return frame, meta


# ── Plotting ─────────────────────────────────────────────────────────────────


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _masked_series(frame: pd.DataFrame, short: str, column: str):
    """Window centres and values, with empty windows forced to NaN.

    Forcing NaN is what makes matplotlib break the line at an outage instead of
    drawing a segment across it.  This is the opposite of filling: it refuses
    to imply measurements that were never recorded.
    """
    centres = (frame["window_start_min"] + frame["window_end_min"]) / 2.0
    values = pd.to_numeric(frame[column], errors="coerce").copy()
    values[frame[f"{short}_data_unavailable"].astype(bool)] = np.nan
    return centres, values


def plot_raw_signals(
    caseid: int, loaded: dict, case_duration_min: float, out_dir: Path
) -> Path:
    """Raw observations at their native cadence, for visual comparison."""
    plt = _plt()
    names = [t for t in ALL_SIGNALS if t in loaded]
    fig, axes = plt.subplots(len(names), 1, figsize=(13, 2.3 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]

    for ax, tname in zip(axes, names):
        data = loaded[tname]
        t = data["times"] / 60.0
        v = data["values"].astype(float).copy()
        median_dt = data["median_dt"]
        # Break the line at real gaps rather than spanning them.
        if median_dt:
            step = np.diff(data["times"])
            gap_idx = np.where(step > GAP_FACTOR * median_dt)[0]
            v = v.copy()
            t = t.copy()
            for i in reversed(gap_idx):
                t = np.insert(t, i + 1, t[i] + (median_dt / 2.0) / 60.0)
                v = np.insert(v, i + 1, np.nan)
        ax.plot(t, v, linewidth=0.7, color="#1f77b4")
        ax.set_ylabel(tname.split("/")[-1], fontsize=9)
        ax.grid(alpha=0.3)
        ax.axvline(case_duration_min, color="#d62728", linestyle="--", linewidth=1.0)

    axes[0].set_title(
        f"Case {caseid} — RAW observations (~0.5 Hz, native irregular cadence)\n"
        f"No interpolation; line breaks are real gaps",
        fontsize=10,
    )
    axes[-1].set_xlabel("Time from case start (minutes)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"case_{caseid}__raw_signals.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_aggregated_trajectory(
    caseid: int,
    frame: pd.DataFrame,
    window_minutes: float,
    case_duration_min: float,
    signal_meta: dict,
    out_dir: Path,
) -> Path:
    """Window means with min-max envelopes; empty windows stay visibly empty."""
    plt = _plt()
    names = [t for t in ALL_SIGNALS if signal_meta.get(t, {}).get("available")]
    fig, axes = plt.subplots(len(names), 1, figsize=(13, 2.5 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]

    for ax, tname in zip(axes, names):
        short = SIGNAL_SPECS[tname]["short"]
        centres, means = _masked_series(frame, short, f"{short}_mean")
        _, lows = _masked_series(frame, short, f"{short}_min")
        _, highs = _masked_series(frame, short, f"{short}_max")

        ax.fill_between(centres, lows, highs, color="#1f77b4", alpha=0.18,
                        label="window min-max")
        ax.plot(centres, means, marker="o", markersize=4, linewidth=1.6,
                color="#1f77b4", label="window mean")

        # Shade windows with no data so absence reads as absence, not as a flat line.
        empty = frame[frame[f"{short}_data_unavailable"].astype(bool)]
        for _, r in empty.iterrows():
            ax.axvspan(r["window_start_min"], r["window_end_min"],
                       color="#d62728", alpha=0.13)
        # Shade low-coverage windows in a lighter tone.
        low = frame[
            (~frame[f"{short}_data_unavailable"].astype(bool))
            & (frame[f"{short}_coverage_fraction"] < LOW_COVERAGE_FRACTION)
        ]
        for _, r in low.iterrows():
            ax.axvspan(r["window_start_min"], r["window_end_min"],
                       color="#ff7f0e", alpha=0.13)

        ax.set_ylabel(tname.split("/")[-1], fontsize=9)
        ax.grid(alpha=0.3)
        ax.axvline(case_duration_min, color="#d62728", linestyle="--", linewidth=1.0)
        ax.legend(fontsize=7, loc="lower left")

    axes[0].set_title(
        f"Case {caseid} — {window_minutes:g}-minute aggregated trajectory\n"
        f"Red band = window with NO data (line intentionally broken)  |  "
        f"Orange band = coverage < {LOW_COVERAGE_FRACTION:.0%}  |  no interpolation, no fill",
        fontsize=10,
    )
    axes[-1].set_xlabel("Time from case start (minutes)")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"case_{caseid}__aggregated_{window_minutes:g}min.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_coverage(
    caseid: int,
    frame: pd.DataFrame,
    window_minutes: float,
    signal_meta: dict,
    out_dir: Path,
) -> Path:
    """Per-window coverage heatmap: which windows carry usable data."""
    plt = _plt()
    names = [t for t in ALL_SIGNALS if t in signal_meta]
    shorts = [SIGNAL_SPECS[t]["short"] for t in names]

    matrix = np.array([
        pd.to_numeric(frame[f"{s}_coverage_percent"], errors="coerce").to_numpy()
        for s in shorts
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(max(9, 0.42 * len(frame) + 4), 1.0 * len(names) + 2.6))
    mesh = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100,
                     interpolation="nearest")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([
        n.split("/")[-1] + ("" if signal_meta[n]["available"] else "  (ABSENT)")
        for n in names
    ], fontsize=9)
    ax.set_xticks(range(len(frame)))
    ax.set_xticklabels([str(i) for i in frame["window_index"]], fontsize=7)
    ax.set_xlabel(f"Window index ({window_minutes:g}-minute windows)")

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            val = matrix[r, c]
            ax.text(c, r, f"{val:.0f}", ha="center", va="center", fontsize=6,
                    color="black" if 25 < val < 85 else "white")

    fig.colorbar(mesh, ax=ax, label="coverage %", fraction=0.025)
    ax.set_title(
        f"Case {caseid} — window coverage "
        f"(observations / expected at {EXPECTED_INTERVAL_S:g}s interval)",
        fontsize=10,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"case_{caseid}__coverage_{window_minutes:g}min.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ── Orchestration ────────────────────────────────────────────────────────────


def run_experiment(
    caseids: list[int],
    cases: pd.DataFrame,
    trks: pd.DataFrame,
    window_minutes: float = WINDOW_MINUTES,
    signals: list[str] = None,
) -> dict:
    """Aggregate every case, write CSVs, plots and a combined JSON summary."""
    signals = list(signals or ALL_SIGNALS)
    AGGREGATED_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    label = f"{window_minutes:g}min"
    case_reports = []

    for caseid in caseids:
        frame, meta = aggregate_case(
            caseid, cases, trks, window_minutes=window_minutes, signals=signals
        )

        csv_path = AGGREGATED_DIR / f"case_{caseid}_{label}.csv"
        frame.to_csv(csv_path, index=False)

        # Reload raw series once more for the raw plot (cached, no network).
        loaded = {}
        tids = trks[trks["caseid"] == caseid].set_index("tname")["tid"].to_dict()
        for tname in signals:
            if not meta["signals"].get(tname, {}).get("available"):
                continue
            raw = inspect_signals.load_track(tids[tname], caseid, tname)
            t = pd.to_numeric(raw["Time"], errors="coerce")
            v = pd.to_numeric(raw[tname], errors="coerce")
            ok = t.notna()
            loaded[tname] = {
                "times": t[ok].to_numpy(dtype=float),
                "values": v[ok].to_numpy(dtype=float),
                "median_dt": meta["signals"][tname]["median_dt_s"],
            }

        plots = {
            "raw": str(plot_raw_signals(
                caseid, loaded, meta["case_duration_min"], PLOT_DIR)),
            "aggregated": str(plot_aggregated_trajectory(
                caseid, frame, window_minutes, meta["case_duration_min"],
                meta["signals"], PLOT_DIR)),
            "coverage": str(plot_coverage(
                caseid, frame, window_minutes, meta["signals"], PLOT_DIR)),
        }

        meta["csv"] = str(csv_path)
        meta["plots"] = plots
        meta["stats"] = _case_stats(frame, signals)
        case_reports.append(meta)

    bundle = {
        "dataset_version": config.DATASET_VERSION,
        "window_minutes": window_minutes,
        "expected_interval_s": EXPECTED_INTERVAL_S,
        "gap_factor": GAP_FACTOR,
        "core_signals": [t for t in signals if SIGNAL_SPECS[t]["required"]],
        "optional_signals": [t for t in signals if not SIGNAL_SPECS[t]["required"]],
        "interpolation": "none",
        "fill": "none",
        "low_coverage_threshold": LOW_COVERAGE_FRACTION,
        "cases": case_reports,
    }
    out = config.RESULTS_DIR / "aggregation_summary.json"
    out.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    logger.info("wrote aggregation summary to %s", out)
    return bundle


def _case_stats(frame: pd.DataFrame, signals: list[str]) -> dict:
    """Per-case roll-up used by the checkpoint report."""
    stats = {
        "n_windows": int(len(frame)),
        "n_windows_all_core_missing": int(frame["window_data_unavailable"].sum()),
        "per_signal": {},
    }
    for tname in signals:
        short = SIGNAL_SPECS[tname]["short"]
        cov = pd.to_numeric(frame[f"{short}_coverage_fraction"], errors="coerce")
        unavailable = frame[f"{short}_data_unavailable"].astype(bool)
        entry = {
            "n_windows_empty": int(unavailable.sum()),
            "n_windows_low_coverage": int(
                ((~unavailable) & (cov < LOW_COVERAGE_FRACTION)).sum()
            ),
            "coverage_mean_pct": round(float(cov.mean() * 100), 2),
            "coverage_median_pct": round(float(cov.median() * 100), 2),
            "coverage_min_pct": round(float(cov.min() * 100), 2),
            "coverage_max_pct": round(float(cov.max() * 100), 2),
            "largest_gap_seconds": round(
                float(pd.to_numeric(
                    frame[f"{short}_largest_gap_seconds"], errors="coerce").max()), 3
            ),
            "total_gap_seconds": round(
                float(pd.to_numeric(
                    frame[f"{short}_total_gap_seconds"], errors="coerce").sum()), 3
            ),
        }
        if short == "rr_co2":
            entry["rr_co2_zero_count"] = int(
                pd.to_numeric(frame["rr_co2_zero_count"], errors="coerce").sum()
            )
        stats["per_signal"][tname] = entry
    return stats
