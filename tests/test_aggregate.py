"""Tests for monitoring-window aggregation.

All tests are offline: they build synthetic tracks with hand-computed expected
values and call the aggregation functions directly.  No network, no cached
files required.
"""

import math

import numpy as np
import pandas as pd
import pytest

from vitaldb_audit import aggregate


# ── Helpers ──────────────────────────────────────────────────────────────────


def _regular(start, stop, step=2.0, value=70.0):
    """Evenly spaced observations on [start, stop)."""
    times = np.arange(start, stop, step, dtype=float)
    values = np.full(len(times), float(value))
    return times, values


def _summarize(times, values, start, end, stats=("mean", "min", "max", "std"),
               median_dt=2.0, short="hr"):
    return aggregate.summarize_window(
        np.asarray(times, dtype=float), np.asarray(values, dtype=float),
        start, end, stats, median_dt, short,
    )


# ── Window boundaries ────────────────────────────────────────────────────────


def test_window_edges_are_non_overlapping_and_contiguous():
    edges = aggregate.build_window_edges(1800.0, window_minutes=10.0)
    assert edges == [(0.0, 600.0), (600.0, 1200.0), (1200.0, 1800.0)]
    for (_, prev_end), (next_start, _) in zip(edges, edges[1:]):
        assert prev_end == next_start


def test_final_window_is_truncated_not_extended():
    """A 25-minute case yields three windows, the last only 5 minutes wide."""
    edges = aggregate.build_window_edges(1500.0, window_minutes=10.0)
    assert len(edges) == 3
    assert edges[-1] == (1200.0, 1500.0)


def test_window_edges_anchor_at_case_start():
    """Grid starts at t=0 regardless of when observations begin."""
    edges = aggregate.build_window_edges(600.0, window_minutes=10.0)
    assert edges[0][0] == 0.0


@pytest.mark.parametrize("minutes,expected_n", [(5.0, 6), (10.0, 3), (30.0, 1)])
def test_window_size_is_configurable(minutes, expected_n):
    """5 / 10 / 30 minute windows over the same 30-minute span."""
    edges = aggregate.build_window_edges(1800.0, window_minutes=minutes)
    assert len(edges) == expected_n
    assert edges[0][1] == minutes * 60.0


def test_zero_and_negative_window_minutes_rejected():
    with pytest.raises(ValueError):
        aggregate.build_window_edges(600.0, window_minutes=0)
    with pytest.raises(ValueError):
        aggregate.build_window_edges(600.0, window_minutes=-5)


def test_boundary_is_half_open():
    """An observation exactly on the upper edge belongs to the NEXT window."""
    times = np.array([599.0, 600.0], dtype=float)
    values = np.array([70.0, 90.0], dtype=float)
    first = _summarize(times, values, 0.0, 600.0)
    second = _summarize(times, values, 600.0, 1200.0)
    assert first["hr_observation_count"] == 1
    assert first["hr_max"] == 70.0
    assert second["hr_observation_count"] == 1
    assert second["hr_max"] == 90.0


# ── Statistics ───────────────────────────────────────────────────────────────


def test_mean_min_max_are_correct():
    times = np.array([0.0, 2.0, 4.0, 6.0], dtype=float)
    values = np.array([60.0, 70.0, 80.0, 90.0], dtype=float)
    rec = _summarize(times, values, 0.0, 600.0)
    assert rec["hr_mean"] == 75.0
    assert rec["hr_min"] == 60.0
    assert rec["hr_max"] == 90.0


def test_standard_deviation_is_sample_std():
    """ddof=1: std of [2,4,4,4,5,5,7,9] is 2.138..., not the population 2.0."""
    times = np.arange(0.0, 16.0, 2.0)
    values = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
    rec = _summarize(times, values, 0.0, 600.0)
    assert rec["hr_std"] == pytest.approx(2.1381, abs=1e-4)


def test_std_of_single_observation_is_none_not_zero():
    """One sample has undefined dispersion; reporting 0.0 would imply stability."""
    rec = _summarize([0.0], [70.0], 0.0, 600.0)
    assert rec["hr_observation_count"] == 1
    assert rec["hr_std"] is None
    assert rec["hr_mean"] == 70.0


def test_summarize_emits_only_the_requested_stats():
    """The stat set is driven by the caller, never hard-coded per signal."""
    rec = _summarize([0.0, 2.0], [99.0, 100.0], 0.0, 600.0,
                     stats=("mean", "min", "max"), short="spo2")
    assert "spo2_std" not in rec
    assert rec["spo2_mean"] == 99.5


def test_spo2_spec_now_carries_std_for_the_v1_feature_schema():
    """SpO2 std is emitted (see the CAVEAT on SIGNAL_SPECS) because the
    compact v1 feature table consumes it."""
    assert "std" in aggregate.SIGNAL_SPECS["Solar8000/PLETH_SPO2"]["stats"]
    rec = _summarize([0.0, 2.0], [99.0, 100.0], 0.0, 600.0,
                     stats=aggregate.SIGNAL_SPECS["Solar8000/PLETH_SPO2"]["stats"],
                     short="spo2")
    assert rec["spo2_std"] == pytest.approx(0.7071, abs=1e-4)


# ── Observation counts and coverage ──────────────────────────────────────────


def test_expected_count_derived_from_interval_not_row_count():
    """A 10-min window expects 300 observations at 2 s, regardless of rows present."""
    rec = _summarize([0.0, 2.0], [70.0, 71.0], 0.0, 600.0)
    assert rec["hr_expected_observation_count"] == 300.0
    assert rec["hr_observation_count"] == 2


def test_coverage_fraction_and_percent():
    """120 of 300 expected observations is exactly 40%."""
    times, values = _regular(0.0, 240.0)
    assert len(times) == 120
    rec = _summarize(times, values, 0.0, 600.0)
    assert rec["hr_observation_count"] == 120
    assert rec["hr_coverage_fraction"] == 0.4
    assert rec["hr_coverage_percent"] == 40.0
    # The available 40% is retained for statistics, not discarded.
    assert rec["hr_mean"] == 70.0
    assert rec["hr_data_unavailable"] is False


def test_full_coverage_is_one_hundred_percent():
    times, values = _regular(0.0, 600.0)
    rec = _summarize(times, values, 0.0, 600.0)
    assert rec["hr_observation_count"] == 300
    assert rec["hr_coverage_percent"] == 100.0


def test_truncated_final_window_scales_expected_count():
    """A 5-minute tail window expects 150, not 300."""
    rec = _summarize([1200.0], [70.0], 1200.0, 1500.0)
    assert rec["hr_expected_observation_count"] == 150.0


def test_first_and_last_observation_times_recorded():
    times, values = _regular(100.0, 300.0)
    rec = _summarize(times, values, 0.0, 600.0)
    assert rec["hr_first_observation_time"] == 100.0
    assert rec["hr_last_observation_time"] == 298.0


# ── Empty windows ────────────────────────────────────────────────────────────


def test_zero_observation_window_is_retained_with_nulls():
    """An empty window survives with null stats and is marked unavailable."""
    times, values = _regular(0.0, 100.0)
    rec = _summarize(times, values, 600.0, 1200.0)
    assert rec["hr_observation_count"] == 0
    assert rec["hr_coverage_percent"] == 0.0
    assert rec["hr_coverage_fraction"] == 0.0
    assert rec["hr_data_unavailable"] is True
    for stat in ("mean", "min", "max", "std"):
        assert rec[f"hr_{stat}"] is None
    assert rec["hr_first_observation_time"] is None
    assert rec["hr_last_observation_time"] is None


def test_empty_windows_are_not_dropped_from_the_frame():
    """A case with a mid-case outage keeps the outage window in the output."""
    cases = pd.DataFrame([{"caseid": 1, "casestart": 0, "caseend": 1800}])
    trks = pd.DataFrame([{"caseid": 1, "tname": "Solar8000/HR", "tid": "x"}])
    # Observations in windows 0 and 2 only; window 1 is a total outage.
    t = np.concatenate([np.arange(0, 600, 2.0), np.arange(1200, 1800, 2.0)])
    frame = pd.DataFrame({"Time": t, "Solar8000/HR": np.full(len(t), 70.0)})

    result, meta = _aggregate_with_stub(cases, trks, {"Solar8000/HR": frame})
    assert len(result) == 3
    assert bool(result.loc[1, "hr_data_unavailable"]) is True
    assert int(result.loc[1, "hr_observation_count"]) == 0
    assert bool(result.loc[1, "window_data_unavailable"]) is True
    # The surrounding windows are unaffected.
    assert int(result.loc[0, "hr_observation_count"]) == 300
    assert int(result.loc[2, "hr_observation_count"]) == 300


# ── Gaps ─────────────────────────────────────────────────────────────────────


def test_gap_detected_and_measured_within_window():
    """A 20 s hole at 2 s cadence: largest gap 20 s, excess 18 s."""
    times = np.array([0.0, 2.0, 4.0, 24.0, 26.0], dtype=float)
    values = np.full(5, 70.0)
    rec = _summarize(times, values, 0.0, 600.0, median_dt=2.0)
    assert rec["hr_largest_gap_seconds"] == 20.0
    assert rec["hr_total_gap_seconds"] == 18.0


def test_no_gap_reported_for_regular_sampling():
    times, values = _regular(0.0, 600.0)
    rec = _summarize(times, values, 0.0, 600.0)
    assert rec["hr_largest_gap_seconds"] == 0.0
    assert rec["hr_total_gap_seconds"] == 0.0


def test_gaps_are_not_bridged_observation_count_stays_low():
    """Gap handling measures the hole; it never adds observations back."""
    times = np.array([0.0, 2.0, 400.0, 402.0], dtype=float)
    rec = _summarize(times, np.full(4, 70.0), 0.0, 600.0)
    assert rec["hr_observation_count"] == 4
    assert rec["hr_largest_gap_seconds"] == 398.0


# ── No interpolation / fill ──────────────────────────────────────────────────


def test_nan_values_are_excluded_not_filled():
    """NaN observations lower the count and never inherit a neighbour's value."""
    times = np.array([0.0, 2.0, 4.0, 6.0], dtype=float)
    values = np.array([60.0, np.nan, np.nan, 90.0], dtype=float)
    rec = _summarize(times, values, 0.0, 600.0)
    assert rec["hr_observation_count"] == 2
    assert rec["hr_mean"] == 75.0
    assert rec["hr_min"] == 60.0
    assert rec["hr_max"] == 90.0


def test_sparse_window_stats_use_only_real_observations():
    """40% coverage must not be inflated toward a filled 100%."""
    times = np.array([0.0, 2.0, 4.0], dtype=float)
    values = np.array([10.0, 20.0, 30.0], dtype=float)
    rec = _summarize(times, values, 0.0, 600.0)
    assert rec["hr_mean"] == 20.0
    assert rec["hr_observation_count"] == 3
    assert rec["hr_coverage_percent"] == 1.0


def test_empty_window_does_not_borrow_from_neighbours():
    cases = pd.DataFrame([{"caseid": 1, "casestart": 0, "caseend": 1800}])
    trks = pd.DataFrame([{"caseid": 1, "tname": "Solar8000/HR", "tid": "x"}])
    t = np.concatenate([np.arange(0, 600, 2.0), np.arange(1200, 1800, 2.0)])
    v = np.concatenate([np.full(300, 60.0), np.full(300, 90.0)])
    frame = pd.DataFrame({"Time": t, "Solar8000/HR": v})

    result, _ = _aggregate_with_stub(cases, trks, {"Solar8000/HR": frame})
    assert result.loc[0, "hr_mean"] == 60.0
    assert result.loc[2, "hr_mean"] == 90.0
    # The outage window inherits neither 60 nor 90 nor their average.
    assert pd.isna(result.loc[1, "hr_mean"])


# ── RR_CO2 exact zeros ───────────────────────────────────────────────────────


def test_rr_co2_zeros_counted_and_retained_as_observations():
    times = np.array([0.0, 2.0, 4.0, 6.0], dtype=float)
    values = np.array([0.0, 12.0, 0.0, 16.0], dtype=float)
    rec = _summarize(times, values, 0.0, 600.0, short="rr_co2")
    assert rec["rr_co2_zero_count"] == 2
    # Zeros count as observations and participate in the statistics.
    assert rec["rr_co2_observation_count"] == 4
    assert rec["rr_co2_min"] == 0.0
    assert rec["rr_co2_mean"] == 7.0


def test_rr_co2_all_zero_window_is_available_not_missing():
    """An all-zero window has real observations; it is not an outage."""
    times, values = _regular(0.0, 600.0, value=0.0)
    rec = _summarize(times, values, 0.0, 600.0, short="rr_co2")
    assert rec["rr_co2_data_unavailable"] is False
    assert rec["rr_co2_observation_count"] == 300
    assert rec["rr_co2_zero_count"] == 300
    assert rec["rr_co2_coverage_percent"] == 100.0


def test_zero_count_absent_for_non_rr_signals():
    rec = _summarize([0.0], [70.0], 0.0, 600.0, short="hr")
    assert "rr_co2_zero_count" not in rec


# ── Optional ART_MBP ─────────────────────────────────────────────────────────


def test_absent_optional_signal_yields_schema_with_nulls():
    """ART_MBP missing from /trks still produces its columns, all null/zero."""
    cases = pd.DataFrame([{"caseid": 1, "casestart": 0, "caseend": 600}])
    trks = pd.DataFrame([{"caseid": 1, "tname": "Solar8000/HR", "tid": "x"}])
    t = np.arange(0, 600, 2.0)
    frame = pd.DataFrame({"Time": t, "Solar8000/HR": np.full(len(t), 70.0)})

    result, meta = _aggregate_with_stub(cases, trks, {"Solar8000/HR": frame})
    assert meta["signals"]["Solar8000/ART_MBP"]["available"] is False
    assert bool(result.loc[0, "art_mbp_data_unavailable"]) is True
    assert int(result.loc[0, "art_mbp_observation_count"]) == 0
    assert pd.isna(result.loc[0, "art_mbp_mean"])
    # Expected count is still stated, so coverage stays interpretable.
    assert result.loc[0, "art_mbp_expected_observation_count"] == 300.0


def test_absent_optional_signal_does_not_mark_window_unavailable():
    """A window well covered by the core panel is not an outage just because
    the optional track is missing."""
    cases = pd.DataFrame([{"caseid": 1, "casestart": 0, "caseend": 600}])
    trks = pd.DataFrame([
        {"caseid": 1, "tname": n, "tid": "x"}
        for n in ("Solar8000/HR", "Solar8000/PLETH_SPO2", "Solar8000/RR_CO2")
    ])
    t = np.arange(0, 600, 2.0)
    frames = {
        n: pd.DataFrame({"Time": t, n: np.full(len(t), 70.0)})
        for n in ("Solar8000/HR", "Solar8000/PLETH_SPO2", "Solar8000/RR_CO2")
    }
    result, _ = _aggregate_with_stub(cases, trks, frames)
    assert bool(result.loc[0, "window_data_unavailable"]) is False
    assert int(result.loc[0, "core_signals_present"]) == 3


def test_present_optional_signal_is_aggregated():
    times = np.array([0.0, 2.0, 4.0], dtype=float)
    values = np.array([60.0, 70.0, 80.0], dtype=float)
    rec = _summarize(times, values, 0.0, 600.0, short="art_mbp")
    assert rec["art_mbp_mean"] == 70.0
    assert rec["art_mbp_data_unavailable"] is False


# ── End-to-end shape ─────────────────────────────────────────────────────────


def test_configurable_window_changes_row_count_end_to_end():
    cases = pd.DataFrame([{"caseid": 1, "casestart": 0, "caseend": 1800}])
    trks = pd.DataFrame([{"caseid": 1, "tname": "Solar8000/HR", "tid": "x"}])
    t = np.arange(0, 1800, 2.0)
    frames = {"Solar8000/HR": pd.DataFrame({"Time": t, "Solar8000/HR": np.full(len(t), 70.0)})}

    for minutes, expected in [(5.0, 6), (10.0, 3), (30.0, 1)]:
        result, meta = _aggregate_with_stub(cases, trks, frames, window_minutes=minutes)
        assert len(result) == expected
        assert meta["window_minutes"] == minutes


def test_metadata_declares_no_interpolation():
    cases = pd.DataFrame([{"caseid": 1, "casestart": 0, "caseend": 600}])
    trks = pd.DataFrame([{"caseid": 1, "tname": "Solar8000/HR", "tid": "x"}])
    t = np.arange(0, 600, 2.0)
    frames = {"Solar8000/HR": pd.DataFrame({"Time": t, "Solar8000/HR": np.full(len(t), 70.0)})}
    _, meta = _aggregate_with_stub(cases, trks, frames)
    assert meta["interpolation"] == "none"
    assert meta["fill"] == "none"
    assert meta["expected_interval_s"] == 2.0


# ── Stub plumbing ────────────────────────────────────────────────────────────


def _aggregate_with_stub(cases, trks, frames, window_minutes=10.0, monkeypatch=None):
    """Run aggregate_case with load_track stubbed to serve in-memory frames.

    Keeps every test offline and independent of the on-disk signal cache.
    """
    original = aggregate.inspect_signals.load_track

    def fake_load_track(tid, caseid, tname, force_refresh=False):
        return frames[tname]

    aggregate.inspect_signals.load_track = fake_load_track
    try:
        return aggregate.aggregate_case(
            int(cases.iloc[0]["caseid"]), cases, trks, window_minutes=window_minutes
        )
    finally:
        aggregate.inspect_signals.load_track = original
