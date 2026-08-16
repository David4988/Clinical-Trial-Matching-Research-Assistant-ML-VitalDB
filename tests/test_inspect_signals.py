"""Tests for the selected-case signal inspection utility.

These exercise the measurement logic on synthetic tracks with known,
hand-computed properties.  No network access.
"""

import pandas as pd
import pytest

from vitaldb_audit import inspect_signals


def _track(times, values, tname="Solar8000/HR"):
    return pd.DataFrame({"Time": times, tname: values})


@pytest.fixture
def clean_track():
    """120 samples at exactly 2 s, no gaps, no NaNs: 0..238 s."""
    times = [i * 2.0 for i in range(120)]
    return _track(times, [70 + (i % 5) for i in range(120)])


def test_clean_track_measures_interval_and_rate(clean_track):
    s = inspect_signals.summarize_track(
        clean_track, caseid=1, tname="Solar8000/HR", tid="t", case_duration_min=4.0
    )
    assert s.n_rows == 120
    assert s.median_dt_s == 2.0
    assert s.observed_sampling_hz == 0.5
    assert s.n_gaps == 0
    assert s.first_time_s == 0.0
    assert s.last_time_s == 238.0
    assert s.observed_duration_min == pytest.approx(3.97, abs=0.01)


def test_gap_is_detected_and_measured():
    """A 20 s hole in a 2 s cadence is one gap, excess 18 s."""
    times = [0.0, 2.0, 4.0, 24.0, 26.0, 28.0]
    s = inspect_signals.summarize_track(
        _track(times, [70] * 6), caseid=1, tname="Solar8000/HR",
        tid="t", case_duration_min=1.0,
    )
    assert s.n_gaps == 1
    assert s.largest_gap_s == 20.0
    assert s.total_gap_s == 18.0


def test_nan_values_are_counted_not_filled():
    """NaN values are reported, never imputed, and excluded from statistics."""
    times = [0.0, 2.0, 4.0, 6.0]
    s = inspect_signals.summarize_track(
        _track(times, [70, None, 90, None]), caseid=1, tname="Solar8000/HR",
        tid="t", case_duration_min=1.0,
    )
    assert s.n_rows == 4
    assert s.n_values_missing == 2
    assert s.n_values_present == 2
    assert s.pct_values_missing == 50.0
    # Statistics ignore the holes rather than filling them.
    assert s.value_min == 70.0
    assert s.value_max == 90.0
    assert s.interpolation == "none"
    assert s.resampling == "none"


def test_nan_values_are_distinct_from_timestamp_gaps():
    """A dense track full of NaNs has missingness but zero gaps."""
    times = [i * 2.0 for i in range(10)]
    s = inspect_signals.summarize_track(
        _track(times, [None] * 5 + [70] * 5), caseid=1, tname="Solar8000/HR",
        tid="t", case_duration_min=1.0,
    )
    assert s.n_values_missing == 5
    assert s.n_gaps == 0


def test_span_and_density_coverage_differ():
    """A sparse track can span the case yet miss most of its samples."""
    # 4 samples spanning the full 60 s case, cadence 2 s => 30 expected.
    times = [0.0, 2.0, 58.0, 60.0]
    s = inspect_signals.summarize_track(
        _track(times, [70] * 4), caseid=1, tname="Solar8000/HR",
        tid="t", case_duration_min=1.0,
    )
    assert s.span_coverage_pct == 100.0
    assert s.sample_coverage_pct == pytest.approx(13.33, abs=0.01)


def test_values_are_never_clipped():
    """Out-of-band values survive unmodified and are flagged, not corrected."""
    times = [0.0, 2.0, 4.0]
    s = inspect_signals.summarize_track(
        _track(times, [-61, 71, 289], tname="Solar8000/ART_MBP"),
        caseid=1, tname="Solar8000/ART_MBP", tid="t", case_duration_min=1.0,
    )
    assert s.value_min == -61.0
    assert s.value_max == 289.0
    assert any("below plausible" in f for f in s.flags)
    assert any("above plausible" in f for f in s.flags)


def test_break_across_gaps_inserts_only_plot_breaks():
    """Gap breaking adds NaN rows for rendering and never alters real samples."""
    frame = _track([0.0, 2.0, 24.0, 26.0], [70, 71, 72, 73])
    broken = inspect_signals._break_across_gaps(frame, "Solar8000/HR", 2.0)
    assert len(broken) == len(frame) + 1
    # Every original observation survives untouched.
    kept = broken.dropna(subset=["Solar8000/HR"])["Solar8000/HR"].tolist()
    assert kept == [70, 71, 72, 73]
    # The inserted row is a NaN placed inside the gap.
    inserted = broken[broken["Solar8000/HR"].isna()]
    assert len(inserted) == 1
    assert 2.0 < float(inserted["Time"].iloc[0]) < 24.0


def test_break_across_gaps_noop_when_dense():
    frame = _track([0.0, 2.0, 4.0], [70, 71, 72])
    assert len(inspect_signals._break_across_gaps(frame, "Solar8000/HR", 2.0)) == 3


def test_empty_track_is_reported_not_crashed():
    s = inspect_signals.summarize_track(
        _track([], []), caseid=1, tname="Solar8000/HR", tid="t", case_duration_min=1.0,
    )
    assert s.n_rows == 0
    assert s.value_median is None
    assert any("zero rows" in f for f in s.flags)
