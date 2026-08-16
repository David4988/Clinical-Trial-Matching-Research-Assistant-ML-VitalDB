"""Tests for the evidence-preparation layer.

The evidence object is what a downstream LLM will read, so the failure that
matters most is a number that looks authoritative and is wrong: a previous mean
taken from the wrong window, a null quietly becoming a zero, or a numpy scalar
that forces a custom encoder and hides type errors.
"""

import json

import numpy as np
import pandas as pd
import pytest

from vitaldb_audit import evidence


# ── Helpers ──────────────────────────────────────────────────────────────────


def _row(caseid, index, hr=70.0, spo2=99.0, rr=12.0, std=1.5,
         usable=True, coverage=100.0, deltas=None):
    record = {
        "caseid": caseid,
        "window_index": index,
        "window_start_s": index * 300.0,
        "window_end_s": (index + 1) * 300.0,
        "n_core_usable": 3 if usable else 1,
        "window_usable": usable,
        "consecutive_usable_windows": index + 1 if usable else 0,
    }
    deltas = deltas or {}
    for short, mean in (("hr", hr), ("spo2", spo2), ("rr", rr)):
        record[f"{short}_mean"] = mean
        record[f"{short}_std"] = std
        record[f"{short}_min"] = mean - 4
        record[f"{short}_max"] = mean + 4
        record[f"{short}_delta"] = deltas.get(short, np.nan)
        record[f"{short}_coverage_pct"] = coverage
        record[f"{short}_usable"] = usable
    return record


def _result(caseid, index, score=0.05, rank=1, label=1):
    return {
        "caseid": caseid,
        "window_index": index,
        "window_start_s": index * 300.0,
        "window_end_s": (index + 1) * 300.0,
        "anomaly_score": score,
        "anomaly_rank": rank,
        "anomaly_label": label,
    }


# ── Previous usable window ───────────────────────────────────────────────────


def test_previous_usable_mean_is_the_immediately_previous_window():
    table = pd.DataFrame([_row(1, 0, hr=70.0), _row(1, 1, hr=78.0)])
    reference = evidence.previous_usable_window(table, 1, 1, "hr")
    assert reference["mean"] == 70.0
    assert reference["window_index"] == 0
    assert reference["windows_back"] == 1


def test_previous_usable_mean_skips_unusable_windows():
    """The reference must be the previous USABLE window, not the previous one."""
    table = pd.DataFrame([
        _row(1, 0, hr=70.0),
        _row(1, 1, hr=140.0, usable=False),   # must not be used as reference
        _row(1, 2, hr=76.0),
    ])
    reference = evidence.previous_usable_window(table, 1, 2, "hr")
    assert reference["mean"] == 70.0
    assert reference["window_index"] == 0
    assert reference["windows_back"] == 2


def test_previous_usable_mean_is_null_when_none_exists():
    table = pd.DataFrame([_row(1, 0, hr=70.0)])
    reference = evidence.previous_usable_window(table, 1, 0, "hr")
    assert reference["mean"] is None
    assert reference["window_index"] is None
    assert reference["windows_back"] is None


def test_previous_usable_mean_never_crosses_a_case_boundary():
    table = pd.DataFrame([
        _row(1, 0, hr=70.0), _row(1, 1, hr=72.0),
        _row(2, 0, hr=130.0), _row(2, 1, hr=132.0),
    ])
    reference = evidence.previous_usable_window(table, 2, 0, "hr")
    assert reference["mean"] is None, "case 2 must not reference case 1"


def test_previous_mean_is_consistent_with_the_recorded_delta():
    """current_mean - previous_usable_mean must equal the stored delta.

    If these ever disagree, the evidence object and the feature table are
    describing different things.
    """
    table = pd.DataFrame([
        _row(1, 0, hr=70.0),
        _row(1, 1, hr=78.4, deltas={"hr": 8.4}),
    ])
    entry = evidence.build_window_evidence(table, pd.Series(_result(1, 1)))
    hr = entry["signals"]["hr"]
    assert hr["current_mean"] - hr["previous_usable_mean"] == pytest.approx(hr["delta"])


# ── Change flags ─────────────────────────────────────────────────────────────


def test_change_flag_true_when_movement_exceeds_own_variability():
    table = pd.DataFrame([
        _row(1, 0, hr=70.0, std=1.0),
        _row(1, 1, hr=78.0, std=1.0, deltas={"hr": 8.0}),
    ])
    basis = evidence.change_basis(table, table.iloc[1], "hr")
    assert basis["pooled_std"] == pytest.approx(1.0)
    assert basis["trend_score"] == pytest.approx(8.0)
    assert basis["changed"] is True


def test_change_flag_false_for_movement_inside_own_variability():
    table = pd.DataFrame([
        _row(1, 0, hr=70.0, std=5.0),
        _row(1, 1, hr=72.0, std=5.0, deltas={"hr": 2.0}),
    ])
    basis = evidence.change_basis(table, table.iloc[1], "hr")
    assert basis["trend_score"] == pytest.approx(0.4)
    assert basis["changed"] is False


def test_change_flag_is_null_not_false_when_no_reference_exists():
    """Unknown must never collapse into 'did not change'."""
    table = pd.DataFrame([_row(1, 0, hr=70.0)])
    basis = evidence.change_basis(table, table.iloc[0], "hr")
    assert basis["changed"] is None
    assert basis["trend_score"] is None
    assert "no usable reference" in basis["reason"]


def test_change_basis_is_emitted_for_audit():
    table = pd.DataFrame([
        _row(1, 0, hr=70.0, std=2.0),
        _row(1, 1, hr=80.0, std=2.0, deltas={"hr": 10.0}),
    ])
    entry = evidence.build_window_evidence(table, pd.Series(_result(1, 1)))
    basis = entry["observations"]["change_basis"]["hr"]
    for key in ("delta", "pooled_std", "trend_score", "threshold_k", "changed"):
        assert key in basis
    # The flag must be reproducible from the emitted numbers alone.
    assert (abs(basis["trend_score"]) > basis["threshold_k"]) is basis["changed"]


def test_k_is_configurable():
    table = pd.DataFrame([
        _row(1, 0, hr=70.0, std=2.0),
        _row(1, 1, hr=73.0, std=2.0, deltas={"hr": 3.0}),
    ])
    assert evidence.change_basis(table, table.iloc[1], "hr", k=1.0)["changed"] is True
    assert evidence.change_basis(table, table.iloc[1], "hr", k=2.0)["changed"] is False


def test_multiple_signals_changed_requires_at_least_two():
    table = pd.DataFrame([
        _row(1, 0, hr=70.0, spo2=99.0, rr=12.0, std=1.0),
        _row(1, 1, hr=80.0, spo2=99.2, rr=12.1, std=1.0,
             deltas={"hr": 10.0, "spo2": 0.2, "rr": 0.1}),
    ])
    entry = evidence.build_window_evidence(table, pd.Series(_result(1, 1)))
    observations = entry["observations"]
    assert observations["hr_changed"] is True
    assert observations["spo2_changed"] is False
    assert observations["rr_changed"] is False
    assert observations["n_signals_changed"] == 1
    assert observations["multiple_signals_changed"] is False


def test_multiple_signals_changed_true_when_two_move():
    table = pd.DataFrame([
        _row(1, 0, hr=70.0, spo2=99.0, rr=12.0, std=1.0),
        _row(1, 1, hr=80.0, spo2=95.0, rr=12.1, std=1.0,
             deltas={"hr": 10.0, "spo2": -4.0, "rr": 0.1}),
    ])
    observations = evidence.build_window_evidence(
        table, pd.Series(_result(1, 1)))["observations"]
    assert observations["multiple_signals_changed"] is True
    assert observations["n_signals_changed"] == 2


def test_direction_matches_the_sign_of_delta():
    table = pd.DataFrame([
        _row(1, 0, hr=70.0, spo2=99.0, rr=12.0, std=1.0),
        _row(1, 1, hr=80.0, spo2=95.0, rr=12.0, std=1.0,
             deltas={"hr": 10.0, "spo2": -4.0, "rr": 0.0}),
    ])
    observations = evidence.build_window_evidence(
        table, pd.Series(_result(1, 1)))["observations"]
    assert observations["hr_direction"] == "increase"
    assert observations["spo2_direction"] == "decrease"
    assert observations["rr_direction"] == "no change"


# ── Dispersion flags ─────────────────────────────────────────────────────────


def _case_with_stds(stds, caseid=1):
    """A case whose windows have the given per-window std for all signals."""
    return pd.DataFrame([
        _row(caseid, i, std=s, deltas={"hr": 0.1, "spo2": 0.1, "rr": 0.1})
        for i, s in enumerate(stds)
    ])


def test_dispersion_flag_true_for_a_window_far_above_the_case_fence():
    # Reference stds 1.0-2.0 -> Q3 = 2.0, IQR = 1.0, fence = 3.5
    table = _case_with_stds([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 9.0])
    basis = evidence.dispersion_basis(table, table.iloc[6], "hr")
    assert basis["threshold"] == pytest.approx(3.5)
    assert basis["unusual"] is True
    assert basis["degenerate_reference"] is False


def test_dispersion_flag_false_for_a_typical_window():
    table = _case_with_stds([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    basis = evidence.dispersion_basis(table, table.iloc[6], "hr")
    assert basis["unusual"] is False


def test_dispersion_reference_excludes_the_window_being_judged():
    """An extreme window must not inflate the bar it is measured against."""
    table = _case_with_stds([1.0, 1.0, 1.0, 1.0, 1.0, 40.0])
    basis = evidence.dispersion_basis(table, table.iloc[5], "hr")
    assert basis["reference_windows"] == 5
    assert basis["reference_median"] == pytest.approx(1.0)
    assert basis["unusual"] is True


def test_dispersion_flag_handles_a_flat_signal_without_dividing_by_zero():
    """SpO2 std is exactly zero across whole cases in the real data."""
    table = _case_with_stds([0.0, 0.0, 0.0, 0.0, 0.0, 2.5])
    basis = evidence.dispersion_basis(table, table.iloc[5], "spo2")
    assert basis["threshold"] == pytest.approx(0.0)
    assert basis["degenerate_reference"] is True
    assert basis["unusual"] is True
    assert "flat" in basis["reason"] or "zero dispersion" in basis["reason"]


def test_flat_window_in_a_flat_case_is_not_flagged():
    table = _case_with_stds([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    basis = evidence.dispersion_basis(table, table.iloc[5], "spo2")
    assert basis["unusual"] is False
    assert basis["degenerate_reference"] is True


def test_dispersion_flag_is_null_when_the_case_is_too_short():
    table = _case_with_stds([1.0, 5.0])
    basis = evidence.dispersion_basis(table, table.iloc[1], "hr")
    assert basis["unusual"] is None
    assert "reference windows" in basis["reason"]


def test_dispersion_flag_is_null_when_std_is_missing():
    table = _case_with_stds([1.0, 1.0, 1.0, 1.0])
    table.loc[3, "hr_std"] = np.nan
    basis = evidence.dispersion_basis(table, table.iloc[3], "hr")
    assert basis["unusual"] is None


def test_dispersion_reference_never_crosses_a_case_boundary():
    quiet = _case_with_stds([1.0] * 6, caseid=1)
    noisy = _case_with_stds([8.0] * 6, caseid=2)
    table = pd.concat([quiet, noisy], ignore_index=True)
    basis = evidence.dispersion_basis(table, noisy.iloc[5], "hr")
    assert basis["reference_windows"] == 5
    assert basis["reference_median"] == pytest.approx(8.0)
    # Normal for its own case, even though it dwarfs case 1.
    assert basis["unusual"] is False


def test_dispersion_reference_ignores_unusable_windows():
    table = _case_with_stds([1.0, 1.0, 1.0, 1.0, 5.0])
    table.loc[4, "hr_usable"] = False
    table.loc[4, "window_usable"] = False
    basis = evidence.dispersion_basis(table, table.iloc[3], "hr")
    assert basis["reference_windows"] == 3


def test_dispersion_is_one_sided_low_spread_is_not_unusual():
    table = _case_with_stds([9.0, 9.0, 9.0, 9.0, 0.0])
    basis = evidence.dispersion_basis(table, table.iloc[4], "hr")
    assert basis["unusual"] is False, "unusually SMALL spread must not flag"


def test_dispersion_k_is_configurable():
    table = _case_with_stds([1.0, 1.0, 2.0, 2.0, 4.0])
    loose = evidence.dispersion_basis(table, table.iloc[4], "hr", k=5.0)
    tight = evidence.dispersion_basis(table, table.iloc[4], "hr", k=0.0)
    assert tight["threshold"] <= loose["threshold"]
    assert tight["unusual"] is True
    assert loose["unusual"] is False


def test_dispersion_basis_is_reproducible_from_its_own_numbers():
    table = _case_with_stds([1.0, 1.0, 1.0, 2.0, 2.0, 9.0])
    basis = evidence.dispersion_basis(table, table.iloc[5], "hr")
    assert (basis["current_std"] > basis["threshold"]) is basis["unusual"]
    assert basis["threshold"] == pytest.approx(
        basis["reference_q3"] + basis["threshold_k"] * basis["reference_iqr"]
    )


def test_three_dispersion_fields_are_present_on_every_object():
    table = _case_with_stds([1.0, 1.0, 1.0, 1.0, 8.0])
    results = pd.DataFrame([_result(1, 4, rank=1, label=1)])
    document = evidence.build_document(table, results)
    observations = document["evidence"][0]["observations"]
    for field in ("hr_dispersion_unusual", "spo2_dispersion_unusual",
                  "rr_dispersion_unusual"):
        assert field in observations
        assert observations[field] in (True, False, None)


def test_dispersion_flags_do_not_disturb_the_existing_fields():
    """Every field from the previous evidence schema must still be present."""
    table = _case_with_stds([1.0, 1.0, 1.0, 1.0, 8.0])
    results = pd.DataFrame([_result(1, 4, rank=1, label=1)])
    entry = evidence.build_document(table, results)["evidence"][0]

    for key in ("case_id", "window_index", "time_range", "anomaly_score",
                "anomaly_rank", "signals", "n_core_signals_usable",
                "window_usable", "observations"):
        assert key in entry
    for signal in ("hr", "spo2", "rr"):
        for field in ("current_mean", "previous_usable_mean", "delta", "std",
                      "min", "max", "coverage_pct"):
            assert field in entry["signals"][signal]
    for flag in ("hr_changed", "spo2_changed", "rr_changed",
                 "multiple_signals_changed", "n_signals_changed",
                 "hr_direction", "change_basis"):
        assert flag in entry["observations"]


def test_raw_std_min_max_are_preserved_alongside_the_new_flags():
    table = _case_with_stds([1.0, 1.0, 1.0, 1.0, 8.0])
    results = pd.DataFrame([_result(1, 4, rank=1, label=1)])
    hr = evidence.build_document(table, results)["evidence"][0]["signals"]["hr"]
    assert hr["std"] == pytest.approx(8.0)
    assert hr["min"] is not None and hr["max"] is not None


def test_document_declares_the_dispersion_rule_as_non_clinical_and_non_causal():
    table = _case_with_stds([1.0, 1.0, 1.0, 1.0, 8.0])
    results = pd.DataFrame([_result(1, 4, rank=1, label=1)])
    rule = evidence.build_document(table, results)["dispersion_rule"]
    assert rule["not_a_clinical_threshold"] is True
    assert rule["causal"] is False
    assert rule["one_sided"] is True


# ── Object shape ─────────────────────────────────────────────────────────────


def test_evidence_object_has_every_requested_field():
    table = pd.DataFrame([
        _row(1, 0), _row(1, 1, deltas={"hr": 5.0, "spo2": -1.0, "rr": 2.0})])
    entry = evidence.build_window_evidence(table, pd.Series(_result(1, 1)))

    for key in ("case_id", "window_index", "time_range", "anomaly_score",
                "anomaly_rank", "signals", "n_core_signals_usable",
                "window_usable", "observations"):
        assert key in entry

    for signal in ("hr", "spo2", "rr"):
        data = entry["signals"][signal]
        for field in ("current_mean", "previous_usable_mean", "delta", "std",
                      "min", "max", "coverage_pct"):
            assert field in data, f"{signal}.{field} missing"

    for flag in ("hr_changed", "spo2_changed", "rr_changed",
                 "multiple_signals_changed"):
        assert flag in entry["observations"]


def test_time_range_is_present_and_coherent():
    table = pd.DataFrame([_row(1, 0), _row(1, 62)])
    entry = evidence.build_window_evidence(table, pd.Series(_result(1, 62)))
    time_range = entry["time_range"]
    assert time_range["start_min"] == pytest.approx(310.0)
    assert time_range["end_min"] == pytest.approx(315.0)
    assert time_range["duration_min"] == pytest.approx(5.0)
    assert "min" in time_range["label"]


def test_coverage_is_reported_even_though_model_b_ignores_it():
    table = pd.DataFrame([_row(1, 0, coverage=93.5), _row(1, 1, coverage=93.5)])
    entry = evidence.build_window_evidence(table, pd.Series(_result(1, 1)))
    for signal in ("hr", "spo2", "rr"):
        assert entry["signals"][signal]["coverage_pct"] == pytest.approx(93.5)


def test_missing_window_raises():
    table = pd.DataFrame([_row(1, 0)])
    with pytest.raises(ValueError, match="not in the feature table"):
        evidence.build_window_evidence(table, pd.Series(_result(1, 99)))


# ── Document ─────────────────────────────────────────────────────────────────


def _small_document():
    table = pd.DataFrame([_row(1, i, hr=70.0 + i, deltas={"hr": 1.0, "spo2": 0.0,
                                                          "rr": 0.0})
                          for i in range(5)])
    results = pd.DataFrame([
        _result(1, 3, score=0.09, rank=1, label=1),
        _result(1, 4, score=0.04, rank=2, label=1),
        _result(1, 1, score=-0.10, rank=3, label=0),
    ])
    return table, results, evidence.build_document(table, results)


def test_document_covers_exactly_the_flagged_windows():
    _, _, document = _small_document()
    assert document["counts"]["windows_flagged"] == 2
    assert [e["window_index"] for e in document["evidence"]] == [3, 4]


def test_evidence_is_ordered_most_unusual_first():
    _, _, document = _small_document()
    ranks = [e["anomaly_rank"] for e in document["evidence"]]
    assert ranks == sorted(ranks)


def test_identifier_fields_are_integers_not_floats():
    """pandas iterrows() collapses a mixed-dtype row to float; rank 1 must not
    reach the consumer as 1.0."""
    _, _, document = _small_document()
    for entry in document["evidence"]:
        for field in ("case_id", "window_index", "anomaly_rank", "anomaly_label",
                      "n_core_signals_usable"):
            assert isinstance(entry[field], int), f"{field} is {type(entry[field])}"
            assert not isinstance(entry[field], bool)


def test_document_serialises_without_a_custom_encoder():
    """numpy scalars would force a default= handler and leak types downstream."""
    _, _, document = _small_document()
    text = json.dumps(document)          # no default= on purpose
    reloaded = json.loads(text)
    assert reloaded["schema_version"] == evidence.SCHEMA_VERSION


def test_document_carries_the_non_diagnostic_framing():
    _, _, document = _small_document()
    interpretation = document["interpretation"].lower()
    assert "not an adverse event" in interpretation
    assert "statistically unusual" in interpretation
    assert document["change_rule"]["not_a_clinical_threshold"] is True


def test_document_contains_no_prompt_or_generated_text():
    """This stage prepares evidence only; no LLM plumbing yet.

    The scan covers the payload. The `purpose` field is excluded because its
    job is to state that there is no prompt here, which would otherwise trip a
    naive substring check.
    """
    _, _, document = _small_document()
    payload = {k: v for k, v in document.items() if k != "purpose"}
    text = json.dumps(payload).lower()
    for banned in ("you are a", "prompt", "system message", "assistant",
                   "diagnosis", "diagnose", "treatment", "prescribe",
                   "recommend", "should be"):
        assert banned not in text, f"{banned!r} leaked into the evidence payload"

    assert "no prompt" in document["purpose"].lower()


def test_report_renders_the_top_windows():
    _, _, document = _small_document()
    report = evidence.render_report(document, n=2)
    assert "case 1" in report
    assert "window 3" in report
    for signal in ("hr", "spo2", "rr"):
        assert signal in report
    assert "not a diagnosis" in report.lower()


def test_report_marks_a_multi_window_reference_gap():
    table = pd.DataFrame([
        _row(1, 0, hr=70.0),
        _row(1, 1, hr=100.0, usable=False),
        _row(1, 2, hr=76.0, deltas={"hr": 6.0, "spo2": 0.0, "rr": 0.0}),
    ])
    results = pd.DataFrame([_result(1, 2, rank=1, label=1)])
    document = evidence.build_document(table, results)
    assert document["evidence"][0]["signals"]["hr"]["windows_since_reference"] == 2
    assert "not 5 minutes" in evidence.render_report(document)
