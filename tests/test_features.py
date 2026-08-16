"""Tests for the compact v1 feature table.

All offline: synthetic aggregated frames with hand-computed expected values.
The emphasis is on the four properties that would silently corrupt a monitoring
model if they broke — no fill, no future information, no cross-case references,
and deltas that step over unusable windows instead of spanning them.
"""

import pandas as pd
import pytest

from vitaldb_audit import features


# ── Helpers ──────────────────────────────────────────────────────────────────


def _window(caseid, index, hr=70.0, spo2=98.0, rr=12.0,
            hr_cov=1.0, spo2_cov=1.0, rr_cov=1.0):
    """One aggregated window with the columns the feature builder reads."""
    record = {
        "caseid": caseid,
        "window_index": index,
        "window_start_s": index * 300.0,
        "window_end_s": (index + 1) * 300.0,
    }
    values = {"hr": (hr, hr_cov), "spo2": (spo2, spo2_cov), "rr_co2": (rr, rr_cov)}
    for short, (mean, cov) in values.items():
        record[f"{short}_mean"] = mean
        record[f"{short}_std"] = None if mean is None else 1.5
        record[f"{short}_min"] = None if mean is None else mean - 5
        record[f"{short}_max"] = None if mean is None else mean + 5
        record[f"{short}_coverage_fraction"] = cov
    return record


def _frame(records):
    return pd.DataFrame(records)


# ── Schema ───────────────────────────────────────────────────────────────────


def test_schema_is_the_compact_28_column_v1_set():
    cols = features.FEATURE_COLUMNS
    assert len(cols) == 28
    assert cols[:4] == ["caseid", "window_index", "window_start_s", "window_end_s"]
    for feat in ("hr", "spo2", "rr"):
        for stat in ("mean", "std", "min", "max"):
            assert f"{feat}_{stat}" in cols
        assert f"{feat}_delta" in cols
        assert f"{feat}_coverage_pct" in cols
        assert f"{feat}_usable" in cols
    assert "n_core_usable" in cols
    assert "window_usable" in cols
    assert "consecutive_usable_windows" in cols


def test_no_cross_signal_or_trend_columns_leaked_into_v1():
    """The deferred feature families must not appear; v1 stays compact."""
    cols = set(features.FEATURE_COLUMNS)
    for banned in ("cross_", "trend", "run_", "_pct_change", "mbp"):
        assert not any(banned in c for c in cols), f"{banned} leaked into v1"
    # ART_MBP stays out of the model-facing table entirely (it is optional and
    # its availability is a property of the case, not of the patient's state).
    assert not any(c.startswith("art") for c in cols)


def test_column_order_is_stable():
    frame = _frame([_window(1, 0), _window(1, 1)])
    out = features.build_case_features(frame)
    assert list(out.columns) == features.FEATURE_COLUMNS


def test_missing_upstream_statistic_raises_instead_of_emitting_nulls():
    frame = _frame([_window(1, 0)]).drop(columns=["spo2_std"])
    with pytest.raises(ValueError, match="spo2_std"):
        features.build_case_features(frame)


def test_multiple_cases_in_one_frame_is_refused():
    frame = _frame([_window(1, 0), _window(2, 0)])
    with pytest.raises(ValueError, match="exactly one case"):
        features.build_case_features(frame)


# ── Usability and coverage ───────────────────────────────────────────────────


@pytest.mark.parametrize("coverage,expected", [
    (1.00, True), (0.85, True), (0.70, True),   # the bar is inclusive
    (0.699, False), (0.41, False), (0.0, False),
])
def test_usable_flag_applies_the_70_percent_bar(coverage, expected):
    frame = _frame([_window(1, 0, hr_cov=coverage)])
    out = features.build_case_features(frame)
    assert bool(out["hr_usable"].iloc[0]) is expected


def test_coverage_is_reported_as_a_percentage():
    frame = _frame([_window(1, 0, hr_cov=0.9467)])
    out = features.build_case_features(frame)
    assert out["hr_coverage_pct"].iloc[0] == pytest.approx(94.67)


def test_window_usable_requires_all_three_core_signals():
    frame = _frame([
        _window(1, 0),                       # all three usable
        _window(1, 1, rr_cov=0.41),          # RR below the bar
        _window(1, 2, rr_cov=0.41, spo2_cov=0.2),
    ])
    out = features.build_case_features(frame)
    assert out["n_core_usable"].tolist() == [3, 2, 1]
    assert out["window_usable"].tolist() == [True, False, False]


def test_unusable_window_is_kept_not_dropped():
    """An outage must stay visible as a row, or it looks like continuous data."""
    frame = _frame([_window(1, 0), _window(1, 1, hr_cov=0.0, spo2_cov=0.0, rr_cov=0.0)])
    out = features.build_case_features(frame)
    assert len(out) == 2
    assert out["window_index"].tolist() == [0, 1]
    assert bool(out["window_usable"].iloc[1]) is False


# ── Deltas ───────────────────────────────────────────────────────────────────


def test_first_usable_window_has_null_delta():
    frame = _frame([_window(1, 0, hr=70.0)])
    out = features.build_case_features(frame)
    assert pd.isna(out["hr_delta"].iloc[0])


def test_delta_is_difference_of_consecutive_window_means():
    frame = _frame([_window(1, 0, hr=72.4), _window(1, 1, hr=79.8)])
    out = features.build_case_features(frame)
    assert pd.isna(out["hr_delta"].iloc[0])
    assert out["hr_delta"].iloc[1] == pytest.approx(7.4)


def test_delta_references_previous_USABLE_window_not_previous_window():
    """HR drops out in window 1; window 2 must compare back to window 0."""
    frame = _frame([
        _window(1, 0, hr=72.4),
        _window(1, 1, hr=95.0, hr_cov=0.41),   # unusable: must not be a reference
        _window(1, 2, hr=76.4),
    ])
    out = features.build_case_features(frame)
    assert pd.isna(out["hr_delta"].iloc[1])
    assert out["hr_delta"].iloc[2] == pytest.approx(4.0)   # 76.4 - 72.4, not - 95.0


def test_delta_is_null_when_the_current_window_is_unusable():
    frame = _frame([_window(1, 0, hr=70.0), _window(1, 1, hr=90.0, hr_cov=0.3)])
    out = features.build_case_features(frame)
    assert pd.isna(out["hr_delta"].iloc[1])


def test_deltas_are_independent_per_signal():
    """RR dropping out must not null HR's or SpO2's delta."""
    frame = _frame([
        _window(1, 0, hr=70.0, spo2=98.0, rr=12.0),
        _window(1, 1, hr=78.0, spo2=96.0, rr=15.0, rr_cov=0.41),
    ])
    out = features.build_case_features(frame)
    assert out["hr_delta"].iloc[1] == pytest.approx(8.0)
    assert out["spo2_delta"].iloc[1] == pytest.approx(-2.0)
    assert pd.isna(out["rr_delta"].iloc[1])


def test_delta_is_never_zero_filled_for_missing_data():
    """The failure mode this guards: NULL quietly becoming 'no change'."""
    frame = _frame([_window(1, 0, hr_cov=0.0), _window(1, 1, hr_cov=0.0)])
    out = features.build_case_features(frame)
    assert out["hr_delta"].isna().all()
    assert not (out["hr_delta"] == 0).any()


# ── Temporal counter ─────────────────────────────────────────────────────────


def test_consecutive_usable_windows_counts_and_resets():
    frame = _frame([
        _window(1, 0),                 # usable  -> 1
        _window(1, 1),                 # usable  -> 2
        _window(1, 2, rr_cov=0.4),     # not     -> 0
        _window(1, 3),                 # usable  -> 1
        _window(1, 4),                 # usable  -> 2
    ])
    out = features.build_case_features(frame)
    assert out["consecutive_usable_windows"].tolist() == [1, 2, 0, 1, 2]


def test_consecutive_counter_starts_at_zero_for_an_unusable_first_window():
    frame = _frame([_window(1, 0, hr_cov=0.1), _window(1, 1)])
    out = features.build_case_features(frame)
    assert out["consecutive_usable_windows"].tolist() == [0, 1]


# ── Leakage properties ───────────────────────────────────────────────────────


def test_no_future_information_on_a_synthetic_case():
    frame = _frame([
        _window(1, 0, hr=70.0),
        _window(1, 1, hr=78.0, rr_cov=0.4),
        _window(1, 2, hr=85.0),
        _window(1, 3, hr=91.0, spo2_cov=0.5),
        _window(1, 4, hr=88.0),
    ])
    report = features.verify_no_future_information([frame])
    assert report["passed"]
    assert report["rows_rebuilt_from_truncated_history"] == 5


def test_truncating_the_case_does_not_change_earlier_rows():
    frame = _frame([_window(1, i, hr=70.0 + 3 * i) for i in range(6)])
    full = features.build_case_features(frame)
    partial = features.build_case_features(frame.iloc[:3])
    pd.testing.assert_frame_equal(full.iloc[:3], partial)


def test_no_cross_case_leakage_between_stacked_cases():
    case_a = _frame([_window(1, i, hr=70.0 + i) for i in range(4)])
    case_b = _frame([_window(2, i, hr=120.0 + i) for i in range(4)])
    report = features.verify_no_cross_case_leakage([case_a, case_b])
    assert report["passed"]

    table = features.build_feature_table([case_a, case_b])
    first_b = table[(table["caseid"] == 2) & (table["window_index"] == 0)]
    # Case B opens at 120 while case A ends at 73; a leaked reference would
    # show up as a large positive delta rather than a null.
    assert pd.isna(first_b["hr_delta"].iloc[0])
    assert first_b["consecutive_usable_windows"].iloc[0] == 1


def test_case_ordering_in_the_input_does_not_change_the_output():
    case_a = _frame([_window(1, i, hr=70.0 + i) for i in range(3)])
    case_b = _frame([_window(2, i, hr=120.0 + i) for i in range(3)])
    forward = features.build_feature_table([case_a, case_b])
    reverse = features.build_feature_table([case_b, case_a])
    pd.testing.assert_frame_equal(forward, reverse)


def test_shuffled_window_order_is_sorted_before_deltas_are_computed():
    frame = _frame([_window(1, i, hr=70.0 + 5 * i) for i in range(4)])
    shuffled = frame.iloc[[3, 0, 2, 1]].reset_index(drop=True)
    out = features.build_case_features(shuffled)
    assert out["window_index"].tolist() == [0, 1, 2, 3]
    assert out["hr_delta"].tolist()[1:] == [5.0, 5.0, 5.0]


# ── Empty input ──────────────────────────────────────────────────────────────


def test_empty_frame_yields_empty_table_with_the_full_schema():
    out = features.build_feature_table([])
    assert len(out) == 0
    assert list(out.columns) == features.FEATURE_COLUMNS
