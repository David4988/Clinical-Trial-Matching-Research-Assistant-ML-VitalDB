"""Feature Schema v1 — the compact, interpretable table the model consumes.

Input   : 5-minute aggregated windows produced by ``aggregate.py``.
Output  : one row per (caseid, window_index), 28 columns.
Non-goal: this module scores nothing.  No anomaly detection, no model, no
          thresholds beyond the single coverage bar inherited from the
          preprocessing contract.

Rules carried through from the preprocessing contract, enforced here
--------------------------------------------------------------------
NO INTERPOLATION / NO FILL
    A missing statistic stays NULL.  It is never replaced by 0, by a sentinel,
    by the previous window's value, or by anything else.

UNUSABLE WINDOWS SURVIVE
    Every aggregated window becomes exactly one feature row, including windows
    that fail the coverage bar.  Filtering is a training-time decision made
    downstream, not silently applied here.  Dropping them would make a
    monitoring outage look like continuous data.

NO FUTURE INFORMATION
    Row k is a function of windows 0..k only.  Verified directly by
    ``verify_no_future_information``, which rebuilds each row from a truncated
    history and requires an identical result.

NO CROSS-CASE REFERENCES
    Deltas and run-counters reset at every case boundary.  Verified by
    ``verify_no_cross_case_leakage``.

DELTAS SKIP UNUSABLE WINDOWS, THEY DO NOT SPAN THEM SILENTLY
    ``{signal}_delta`` compares against the most recent earlier window in which
    THAT signal was usable, which is not necessarily window k-1.
"""

import logging

import pandas as pd

from vitaldb_audit import config

logger = logging.getLogger("vitaldb_audit.features")

# ── Schema configuration ─────────────────────────────────────────────────────

# The coverage bar from the preprocessing decision.  A window at or above this
# fraction is usable for that signal.  Fixed at 70%; not re-litigated here.
USABLE_COVERAGE_FRACTION = 0.70

# Feature-table prefix -> aggregation-table prefix.  The aggregation stage
# names the respiratory track after its source (rr_co2); the feature table uses
# the shorter physiological name.
CORE_SIGNALS: dict[str, str] = {
    "hr": "hr",
    "spo2": "spo2",
    "rr": "rr_co2",
}

# Current-state statistics carried per core signal.
STATS = ("mean", "std", "min", "max")

WINDOW_MINUTES = 5.0
AGGREGATED_DIR = config.RESULTS_DIR / "aggregated"
FEATURES_DIR = config.RESULTS_DIR / "features"


def feature_columns() -> list[str]:
    """The v1 column order, built from the config so it cannot drift."""
    cols = ["caseid", "window_index", "window_start_s", "window_end_s"]
    for feat in CORE_SIGNALS:
        cols += [f"{feat}_{stat}" for stat in STATS]
    cols += [f"{feat}_delta" for feat in CORE_SIGNALS]
    cols += [f"{feat}_coverage_pct" for feat in CORE_SIGNALS]
    cols += [f"{feat}_usable" for feat in CORE_SIGNALS]
    cols += ["n_core_usable", "window_usable", "consecutive_usable_windows"]
    return cols


FEATURE_COLUMNS = feature_columns()


# ── Row-level helpers ────────────────────────────────────────────────────────


def _delta_from_previous_usable(means, usable) -> pd.array:
    """Change in the window mean since this signal was last usable.

    NULL when the current window is unusable for this signal (we refuse to
    compare against a statistic we already declared untrustworthy) and when no
    earlier usable window exists in the case.  The reference is the most recent
    earlier USABLE window, so an intervening outage is stepped over rather than
    treated as a zero change.
    """
    deltas: list = []
    last_usable_mean = pd.NA

    for mean, is_usable in zip(means, usable):
        usable_now = pd.notna(is_usable) and bool(is_usable) and pd.notna(mean)
        if not usable_now:
            deltas.append(pd.NA)
            continue
        if pd.isna(last_usable_mean):
            deltas.append(pd.NA)          # first usable window of the case
        else:
            deltas.append(round(float(mean) - float(last_usable_mean), 4))
        last_usable_mean = mean

    return pd.array(deltas, dtype="Float64")


def _consecutive_run(flags) -> pd.array:
    """Length of the run of True values ending at (and including) each row.

    Resets to 0 on any False.  Because this is only ever called per case, it
    cannot carry a run across a case boundary.
    """
    run = 0
    out: list[int] = []
    for flag in flags:
        run = run + 1 if (pd.notna(flag) and bool(flag)) else 0
        out.append(run)
    return pd.array(out, dtype="Int64")


# ── Table construction ───────────────────────────────────────────────────────


def build_case_features(agg: pd.DataFrame) -> pd.DataFrame:
    """Build the v1 feature rows for ONE case's aggregated windows.

    Accepting exactly one case is deliberate: it makes cross-case leakage a
    structural impossibility rather than something a sort order has to prevent.
    """
    required = {"caseid", "window_index", "window_start_s", "window_end_s"}
    missing = required - set(agg.columns)
    if missing:
        raise ValueError(f"aggregated frame is missing columns: {sorted(missing)}")

    if agg.empty:
        return pd.DataFrame({c: pd.array([], dtype="Float64") for c in FEATURE_COLUMNS})

    caseids = pd.unique(agg["caseid"])
    if len(caseids) != 1:
        raise ValueError(
            f"build_case_features expects exactly one case, got {list(caseids)}"
        )

    # Sorting by window_index is what makes "previous" mean "earlier in time".
    agg = agg.sort_values("window_index").reset_index(drop=True)

    out = pd.DataFrame({
        "caseid": agg["caseid"].astype("Int64"),
        "window_index": agg["window_index"].astype("Int64"),
        "window_start_s": pd.to_numeric(agg["window_start_s"]).astype("Float64"),
        "window_end_s": pd.to_numeric(agg["window_end_s"]).astype("Float64"),
    })

    for feat, src in CORE_SIGNALS.items():
        for stat in STATS:
            col = f"{src}_{stat}"
            if col not in agg.columns:
                # Fail loudly.  Emitting a silently all-null column would let a
                # missing upstream statistic reach the model as "no data".
                raise ValueError(
                    f"aggregated frame has no column {col!r}; "
                    f"re-run aggregation so {feat}_{stat} can be built"
                )
            out[f"{feat}_{stat}"] = pd.to_numeric(
                agg[col], errors="coerce").astype("Float64")

        coverage = pd.to_numeric(
            agg[f"{src}_coverage_fraction"], errors="coerce")
        out[f"{feat}_coverage_pct"] = (coverage * 100).round(2).astype("Float64")
        # A usable signal needs both the coverage bar AND an actual mean; the
        # second clause guards against a window that somehow counts observations
        # it cannot summarise.
        out[f"{feat}_usable"] = (
            (coverage >= USABLE_COVERAGE_FRACTION) & out[f"{feat}_mean"].notna()
        ).fillna(False).astype("boolean")

    for feat in CORE_SIGNALS:
        out[f"{feat}_delta"] = _delta_from_previous_usable(
            out[f"{feat}_mean"], out[f"{feat}_usable"]
        )

    usable_flags = [out[f"{feat}_usable"].fillna(False) for feat in CORE_SIGNALS]
    out["n_core_usable"] = sum(f.astype(int) for f in usable_flags).astype("Int64")
    out["window_usable"] = (
        out["n_core_usable"] == len(CORE_SIGNALS)
    ).astype("boolean")
    out["consecutive_usable_windows"] = _consecutive_run(out["window_usable"])

    return out[FEATURE_COLUMNS]


def build_feature_table(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Build and stack the feature rows for several cases."""
    if not frames:
        return pd.DataFrame({c: pd.array([], dtype="Float64") for c in FEATURE_COLUMNS})
    built = [build_case_features(f) for f in frames]
    table = pd.concat(built, ignore_index=True)
    return table.sort_values(["caseid", "window_index"]).reset_index(drop=True)


# ── Loading ──────────────────────────────────────────────────────────────────


def load_aggregated_frames(
    window_minutes: float = WINDOW_MINUTES,
    directory=None,
) -> list[pd.DataFrame]:
    """Read every aggregated CSV for the given window size, one frame per case."""
    directory = directory or AGGREGATED_DIR
    label = f"{window_minutes:g}min"
    paths = sorted(
        directory.glob(f"case_*_{label}.csv"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not paths:
        raise FileNotFoundError(
            f"no aggregated {label} CSVs in {directory}; run the aggregation stage first"
        )
    frames = []
    for path in paths:
        frames.append(pd.read_csv(path))
        logger.info("loaded %s", path.name)
    return frames


# ── Verification ─────────────────────────────────────────────────────────────


def verify_no_future_information(frames: list[pd.DataFrame]) -> dict:
    """Rebuild each row from a truncated history and require an identical result.

    If any feature peeked at a later window, the row computed from the full case
    would differ from the row computed from windows 0..k.  This is a direct
    check on the artifact, not an argument about the code.
    """
    checked = 0
    for frame in frames:
        full = build_case_features(frame)
        n = len(full)
        # Check every window for short cases, a spread for longer ones.
        indices = range(n) if n <= 40 else sorted(
            {0, 1, n // 4, n // 3, n // 2, (2 * n) // 3, n - 2, n - 1} - {-1}
        )
        for k in indices:
            if k < 0 or k >= n:
                continue
            truncated = build_case_features(frame.iloc[: k + 1])
            expected = full.iloc[[k]].reset_index(drop=True)
            actual = truncated.iloc[[k]].reset_index(drop=True)
            if not expected.equals(actual):
                differing = [
                    c for c in FEATURE_COLUMNS
                    if not expected[c].equals(actual[c])
                ]
                raise AssertionError(
                    f"future information detected in case "
                    f"{frame['caseid'].iloc[0]} window {k}: columns {differing}"
                )
            checked += 1
    return {"rows_rebuilt_from_truncated_history": checked, "passed": True}


def verify_no_cross_case_leakage(frames: list[pd.DataFrame]) -> dict:
    """Require each case's rows to be identical alone and in the combined table."""
    combined = build_feature_table(frames)
    checked = 0
    for frame in frames:
        caseid = int(frame["caseid"].iloc[0])
        alone = build_case_features(frame).reset_index(drop=True)
        in_table = combined[combined["caseid"] == caseid].reset_index(drop=True)
        if not alone.equals(in_table):
            differing = [
                c for c in FEATURE_COLUMNS if not alone[c].equals(in_table[c])
            ]
            raise AssertionError(
                f"cross-case leakage detected for case {caseid}: columns {differing}"
            )
        checked += 1

    # A case's first row can have no predecessor to reference.
    firsts = combined.sort_values("window_index").groupby("caseid").head(1)
    for feat in CORE_SIGNALS:
        leaked = firsts[firsts[f"{feat}_delta"].notna()]
        if not leaked.empty:
            raise AssertionError(
                f"{feat}_delta is non-null on the first window of case(s) "
                f"{leaked['caseid'].tolist()}"
            )
    return {"cases_checked": checked, "passed": True}


def null_report(table: pd.DataFrame) -> pd.DataFrame:
    """Per-column null counts, so absence is visible rather than assumed away."""
    nulls = table.isna().sum()
    return pd.DataFrame({
        "column": nulls.index,
        "nulls": nulls.to_numpy(),
        "null_pct": (nulls.to_numpy() / max(len(table), 1) * 100).round(1),
        "dtype": [str(table[c].dtype) for c in nulls.index],
    }).reset_index(drop=True)
