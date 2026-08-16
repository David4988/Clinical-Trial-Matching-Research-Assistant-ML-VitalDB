"""Tests for the deterministic XAI explanation layer.

Tests verify:
- correct output schema
- signal narrative generation for all 4 cases
- headline priority tiers
- driver ranking and driver_type classification
- supporting detail (degenerate refs, coverage, reference gaps)
- data quality notes
- disclaimer presence
- no forbidden clinical language
- no claim that rules reconstruct Isolation Forest internals
"""

import copy

import pytest

from vitaldb_audit.explain import (
    DISCLAIMER,
    SIGNALS,
    SIGNAL_LABELS,
    explain_all,
    explain_window,
)


# ── Fixture helpers ──────────────────────────────────────────────────────────
# Minimal evidence entries that exercise every code path.  These are synthetic
# but structurally identical to the real evidence_cases.json objects.


def _make_signal(mean=80.0, prev_mean=80.0, delta=0.0, std=1.0,
                 vmin=78.0, vmax=82.0, cov=100.0, usable=True,
                 unit="bpm", prev_idx=10, gap=1):
    return {
        "unit": unit,
        "current_mean": mean,
        "previous_usable_mean": prev_mean,
        "previous_usable_window_index": prev_idx,
        "windows_since_reference": gap,
        "delta": delta,
        "std": std,
        "min": vmin,
        "max": vmax,
        "coverage_pct": cov,
        "usable": usable,
    }


def _make_change_basis(delta=0.0, pooled_std=1.0, trend_score=0.0,
                       changed=False, reason="stable"):
    return {
        "delta": delta,
        "pooled_std": pooled_std,
        "trend_score": trend_score,
        "threshold_k": 1.0,
        "changed": changed,
        "reason": reason,
    }


def _make_dispersion_basis(current_std=1.0, ref_windows=20, median=1.0,
                           q3=1.5, iqr=0.5, threshold=2.25,
                           degenerate=False, unusual=False,
                           reason="within-window spread is within the case's normal range"):
    return {
        "current_std": current_std,
        "reference_windows": ref_windows,
        "reference_median": median,
        "reference_q3": q3,
        "reference_iqr": iqr,
        "threshold": threshold,
        "threshold_k": 1.5,
        "degenerate_reference": degenerate,
        "unusual": unusual,
        "reason": reason,
    }


def _make_entry(
    case_id=1, window_index=10, rank=1, score=0.1,
    hr_sig=None, spo2_sig=None, rr_sig=None,
    hr_changed=False, spo2_changed=False, rr_changed=False,
    hr_direction="no change", spo2_direction="no change", rr_direction="no change",
    hr_disp_unusual=False, spo2_disp_unusual=False, rr_disp_unusual=False,
    hr_change_basis=None, spo2_change_basis=None, rr_change_basis=None,
    hr_disp_basis=None, spo2_disp_basis=None, rr_disp_basis=None,
):
    """Build a minimal evidence entry for testing."""
    n_changed = sum([hr_changed, spo2_changed, rr_changed])
    n_unusual = sum([hr_disp_unusual, spo2_disp_unusual, rr_disp_unusual])

    return {
        "case_id": case_id,
        "window_index": window_index,
        "time_range": {
            "start_s": window_index * 300.0,
            "end_s": (window_index + 1) * 300.0,
            "start_min": window_index * 5.0,
            "end_min": (window_index + 1) * 5.0,
            "duration_min": 5.0,
            "label": f"{window_index * 5.0}-{(window_index + 1) * 5.0} min from case start",
        },
        "anomaly_score": score,
        "anomaly_rank": rank,
        "anomaly_label": 1,
        "signals": {
            "hr": hr_sig or _make_signal(unit="bpm"),
            "spo2": spo2_sig or _make_signal(unit="%"),
            "rr": rr_sig or _make_signal(unit="breaths/min"),
        },
        "n_core_signals_usable": 3,
        "window_usable": True,
        "consecutive_usable_windows": 10,
        "observations": {
            "hr_changed": hr_changed,
            "spo2_changed": spo2_changed,
            "rr_changed": rr_changed,
            "multiple_signals_changed": n_changed >= 2,
            "n_signals_changed": n_changed,
            "hr_direction": hr_direction,
            "spo2_direction": spo2_direction,
            "rr_direction": rr_direction,
            "hr_dispersion_unusual": hr_disp_unusual,
            "spo2_dispersion_unusual": spo2_disp_unusual,
            "rr_dispersion_unusual": rr_disp_unusual,
            "n_signals_dispersion_unusual": n_unusual,
            "change_basis": {
                "hr": hr_change_basis or _make_change_basis(),
                "spo2": spo2_change_basis or _make_change_basis(),
                "rr": rr_change_basis or _make_change_basis(),
            },
            "dispersion_basis": {
                "hr": hr_disp_basis or _make_dispersion_basis(),
                "spo2": spo2_disp_basis or _make_dispersion_basis(),
                "rr": rr_disp_basis or _make_dispersion_basis(),
            },
        },
    }


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def dispersion_dominant():
    """Case 4 window 62 pattern: RR and SpO2 dispersion unusual, HR changed."""
    return _make_entry(
        case_id=4, window_index=62, rank=1, score=0.2389,
        hr_sig=_make_signal(mean=99.69, prev_mean=101.46, delta=-1.77,
                            std=0.67, vmin=98, vmax=101, unit="bpm"),
        spo2_sig=_make_signal(mean=99.29, prev_mean=99.85, delta=-0.57,
                              std=3.11, vmin=81, vmax=100, unit="%"),
        rr_sig=_make_signal(mean=16.75, prev_mean=11.42, delta=5.33,
                            std=8.94, vmin=5, vmax=33, unit="breaths/min"),
        hr_changed=True, hr_direction="decrease",
        spo2_direction="decrease", rr_direction="increase",
        spo2_disp_unusual=True, rr_disp_unusual=True,
        hr_change_basis=_make_change_basis(delta=-1.77, pooled_std=1.24,
                                           trend_score=-1.43, changed=True),
        spo2_change_basis=_make_change_basis(delta=-0.57, pooled_std=2.23,
                                             trend_score=-0.25, changed=False),
        rr_change_basis=_make_change_basis(delta=5.33, pooled_std=7.27,
                                           trend_score=0.73, changed=False),
        spo2_disp_basis=_make_dispersion_basis(current_std=3.11, threshold=1.00,
                                               unusual=True),
        rr_disp_basis=_make_dispersion_basis(current_std=8.94, threshold=2.57,
                                             unusual=True),
    )


@pytest.fixture
def change_dominant():
    """Case 8 window 2 pattern: SpO2 and RR changed, no unusual dispersion."""
    return _make_entry(
        case_id=8, window_index=2, rank=10, score=0.0474,
        hr_sig=_make_signal(mean=54.61, prev_mean=54.71, delta=-0.10,
                            std=2.15, vmin=52, vmax=62, unit="bpm"),
        spo2_sig=_make_signal(mean=98.27, prev_mean=99.80, delta=-1.53,
                              std=0.58, vmin=97, vmax=99, unit="%"),
        rr_sig=_make_signal(mean=10.00, prev_mean=13.44, delta=-3.44,
                            std=0.00, vmin=10, vmax=10, unit="breaths/min"),
        spo2_changed=True, rr_changed=True,
        spo2_direction="decrease", rr_direction="decrease",
        hr_direction="decrease",
        spo2_change_basis=_make_change_basis(delta=-1.53, pooled_std=0.49,
                                             trend_score=-3.09, changed=True),
        rr_change_basis=_make_change_basis(delta=-3.44, pooled_std=2.71,
                                           trend_score=-1.27, changed=True),
    )


@pytest.fixture
def mixed_case():
    """RR dispersion unusual + HR changed — instability + mean shift on
    different signals."""
    return _make_entry(
        case_id=4, window_index=60, rank=8, score=0.0707,
        hr_sig=_make_signal(mean=103.41, prev_mean=84.29, delta=19.12,
                            std=4.15, vmin=96, vmax=149, unit="bpm"),
        rr_sig=_make_signal(mean=15.05, prev_mean=15.68, delta=-0.63,
                            std=4.24, vmin=6, vmax=21, unit="breaths/min"),
        hr_changed=True, hr_direction="increase",
        rr_disp_unusual=True,
        hr_change_basis=_make_change_basis(delta=19.12, pooled_std=3.22,
                                           trend_score=5.93, changed=True),
        rr_disp_basis=_make_dispersion_basis(current_std=4.24, threshold=2.57,
                                             unusual=True),
    )


@pytest.fixture
def silent_case():
    """No signal changed, no dispersion unusual — the fallback headline."""
    return _make_entry(
        case_id=99, window_index=50, rank=14, score=0.01,
    )


@pytest.fixture
def degenerate_ref():
    """SpO2 dispersion unusual with degenerate reference (signal flat
    everywhere else)."""
    return _make_entry(
        case_id=4, window_index=70, rank=5, score=0.05,
        spo2_disp_unusual=True,
        spo2_disp_basis=_make_dispersion_basis(
            current_std=0.5, ref_windows=60, median=0.0, q3=0.0, iqr=0.0,
            threshold=0.0, degenerate=True, unusual=True,
            reason="flat elsewhere",
        ),
    )


@pytest.fixture
def low_coverage():
    """HR has partial coverage."""
    return _make_entry(
        case_id=3, window_index=20, rank=12, score=0.02,
        hr_sig=_make_signal(mean=75.0, prev_mean=76.0, delta=-1.0,
                            std=2.0, vmin=70, vmax=80, cov=82.5, unit="bpm"),
    )


@pytest.fixture
def reference_gap():
    """RR reference window is 3 windows back."""
    return _make_entry(
        case_id=5, window_index=15, rank=11, score=0.03,
        rr_sig=_make_signal(mean=14.0, prev_mean=12.0, delta=2.0,
                            std=1.0, vmin=12, vmax=16, gap=3,
                            unit="breaths/min"),
    )


# ── Required output schema ───────────────────────────────────────────────────

REQUIRED_FIELDS = [
    "case_id", "window_index", "anomaly_rank", "anomaly_score",
    "time_label", "headline", "signal_narratives", "primary_drivers",
    "driver_type", "supporting_detail", "data_quality_notes",
    "interpretation",
]


class TestOutputSchema:
    """Every explanation must have the complete field set."""

    def test_all_fields_present(self, dispersion_dominant):
        result = explain_window(dispersion_dominant)
        for field in REQUIRED_FIELDS:
            assert field in result, f"missing field: {field}"

    def test_passthrough_fields(self, dispersion_dominant):
        result = explain_window(dispersion_dominant)
        assert result["case_id"] == 4
        assert result["window_index"] == 62
        assert result["anomaly_rank"] == 1
        assert result["anomaly_score"] == 0.2389

    def test_signal_narratives_keys(self, dispersion_dominant):
        result = explain_window(dispersion_dominant)
        for signal in SIGNALS:
            assert signal in result["signal_narratives"]

    def test_primary_drivers_is_list(self, dispersion_dominant):
        result = explain_window(dispersion_dominant)
        assert isinstance(result["primary_drivers"], list)

    def test_driver_type_valid(self, dispersion_dominant, change_dominant,
                                mixed_case, silent_case):
        valid = {"instability", "mean_shift", "mixed", "undetermined"}
        for entry in [dispersion_dominant, change_dominant, mixed_case, silent_case]:
            result = explain_window(entry)
            assert result["driver_type"] in valid

    def test_supporting_detail_is_list(self, dispersion_dominant):
        result = explain_window(dispersion_dominant)
        assert isinstance(result["supporting_detail"], list)

    def test_data_quality_notes_is_list(self, dispersion_dominant):
        result = explain_window(dispersion_dominant)
        assert isinstance(result["data_quality_notes"], list)


# ── Disclaimer ───────────────────────────────────────────────────────────────


class TestDisclaimer:
    """The NOT-A-DIAGNOSIS disclaimer must appear in every explanation."""

    def test_disclaimer_present(self, dispersion_dominant):
        result = explain_window(dispersion_dominant)
        assert result["interpretation"] == DISCLAIMER

    def test_disclaimer_consistent_across_all(self, dispersion_dominant,
                                               change_dominant, silent_case):
        for entry in [dispersion_dominant, change_dominant, silent_case]:
            result = explain_window(entry)
            assert result["interpretation"] == DISCLAIMER

    def test_disclaimer_content(self):
        assert "clinical diagnosis" in DISCLAIMER.lower()
        assert "adverse" in DISCLAIMER.lower()


# ── Signal narratives ────────────────────────────────────────────────────────


class TestSignalNarratives:
    """Test the four narrative cases for per-signal descriptions."""

    def test_dispersion_and_changed(self, dispersion_dominant):
        """Case 1: HR changed + if it were also dispersion unusual."""
        entry = copy.deepcopy(dispersion_dominant)
        entry["observations"]["hr_dispersion_unusual"] = True
        entry["observations"]["dispersion_basis"]["hr"]["unusual"] = True
        entry["observations"]["dispersion_basis"]["hr"]["threshold"] = 5.25

        result = explain_window(entry)
        narrative = result["signal_narratives"]["hr"]
        assert "unusually large within-window variability" in narrative
        assert "shifted" in narrative.lower() or "mean" in narrative.lower()

    def test_dispersion_only(self, dispersion_dominant):
        """Case 2: RR dispersion unusual, not changed."""
        result = explain_window(dispersion_dominant)
        narrative = result["signal_narratives"]["rr"]
        assert "unusually large within-window variability" in narrative
        assert "normal variation" in narrative or "mean" in narrative.lower()

    def test_changed_only(self, dispersion_dominant):
        """Case 3: HR changed, not dispersion unusual."""
        result = explain_window(dispersion_dominant)
        narrative = result["signal_narratives"]["hr"]
        assert "decreased" in narrative.lower() or "shifted" in narrative.lower()
        assert "trend score" in narrative.lower()

    def test_neither(self, silent_case):
        """Case 4: stable, typical spread."""
        result = explain_window(silent_case)
        for signal in SIGNALS:
            narrative = result["signal_narratives"][signal]
            assert "stable" in narrative.lower()

    def test_unusable_signal(self):
        """Signal marked not usable."""
        entry = _make_entry()
        entry["signals"]["hr"]["usable"] = False
        entry["signals"]["hr"]["current_mean"] = None
        result = explain_window(entry)
        assert "not usable" in result["signal_narratives"]["hr"].lower()


# ── Headline priority ────────────────────────────────────────────────────────


class TestHeadlinePriority:
    """Headlines follow the defined priority order."""

    def test_priority_1_multiple_dispersion(self, dispersion_dominant):
        """Priority 1: multiple dispersion-unusual signals."""
        result = explain_window(dispersion_dominant)
        assert "instability" in result["headline"].lower()
        assert "The model flagged this window" in result["headline"]

    def test_priority_2_dispersion_plus_change(self, mixed_case):
        """Priority 2: single dispersion unusual + change in another."""
        result = explain_window(mixed_case)
        assert "instability" in result["headline"].lower()
        assert "mean shift" in result["headline"].lower()
        assert "The model flagged this window" in result["headline"]

    def test_priority_3_single_dispersion(self):
        """Priority 3: single dispersion unusual only."""
        entry = _make_entry(rr_disp_unusual=True,
                            rr_disp_basis=_make_dispersion_basis(unusual=True))
        result = explain_window(entry)
        assert "variability" in result["headline"].lower()
        assert "The model flagged this window" in result["headline"]

    def test_priority_4_multiple_changed(self, change_dominant):
        """Priority 4: multiple changed, no unusual dispersion."""
        result = explain_window(change_dominant)
        assert "concurrent mean shifts" in result["headline"].lower()
        assert "The model flagged this window" in result["headline"]

    def test_priority_5_single_changed(self):
        """Priority 5: single signal changed."""
        entry = _make_entry(
            hr_changed=True, hr_direction="increase",
            hr_change_basis=_make_change_basis(delta=5.0, trend_score=2.5,
                                               changed=True),
        )
        result = explain_window(entry)
        assert "HR" in result["headline"]
        assert "shifted" in result["headline"].lower()
        assert "The model flagged this window" in result["headline"]

    def test_priority_6_fallback(self, silent_case):
        """Priority 6: no single-signal explanation."""
        result = explain_window(silent_case)
        assert "no single-signal rule" in result["headline"].lower()
        assert "The model flagged this window" in result["headline"]


# ── Driver ranking and type ──────────────────────────────────────────────────


class TestDriverRanking:
    """Primary drivers are ranked by evidence salience."""

    def test_dispersion_dominant_drivers(self, dispersion_dominant):
        result = explain_window(dispersion_dominant)
        drivers = result["primary_drivers"]
        # RR and SpO2 have dispersion unusual (weight 2 each)
        # HR has changed (weight 1)
        # Both dispersion signals should come before HR
        assert len(drivers) == 3
        # The dispersion signals (RR, SpO2) should be first two
        assert set(drivers[:2]) == {"RR", "SpO2"}
        assert drivers[2] == "HR"

    def test_change_dominant_drivers(self, change_dominant):
        result = explain_window(change_dominant)
        drivers = result["primary_drivers"]
        assert set(drivers) == {"SpO2", "RR"}
        # SpO2 has higher trend_score magnitude (3.09 vs 1.27)
        assert drivers[0] == "SpO2"

    def test_silent_case_no_drivers(self, silent_case):
        result = explain_window(silent_case)
        assert result["primary_drivers"] == []

    def test_driver_type_instability(self):
        """Pure dispersion → instability."""
        entry = _make_entry(rr_disp_unusual=True,
                            rr_disp_basis=_make_dispersion_basis(unusual=True))
        result = explain_window(entry)
        assert result["driver_type"] == "instability"

    def test_driver_type_mean_shift(self, change_dominant):
        result = explain_window(change_dominant)
        assert result["driver_type"] == "mean_shift"

    def test_driver_type_mixed(self, mixed_case):
        result = explain_window(mixed_case)
        assert result["driver_type"] == "mixed"

    def test_driver_type_undetermined(self, silent_case):
        result = explain_window(silent_case)
        assert result["driver_type"] == "undetermined"


# ── Supporting detail ────────────────────────────────────────────────────────


class TestSupportingDetail:
    """Supporting detail sentences for special evidence patterns."""

    def test_degenerate_reference(self, degenerate_ref):
        result = explain_window(degenerate_ref)
        notes = result["supporting_detail"]
        assert any("flat" in n.lower() for n in notes), \
            f"expected degenerate-reference note, got: {notes}"

    def test_coverage_note(self, low_coverage):
        result = explain_window(low_coverage)
        notes = result["supporting_detail"]
        assert any("coverage" in n.lower() and "82.5" in n for n in notes), \
            f"expected coverage note, got: {notes}"

    def test_reference_gap(self, reference_gap):
        result = explain_window(reference_gap)
        notes = result["supporting_detail"]
        assert any("3 windows" in n for n in notes), \
            f"expected reference gap note, got: {notes}"

    def test_no_spurious_notes(self, silent_case):
        """A clean, normal window should have no supporting detail."""
        result = explain_window(silent_case)
        assert result["supporting_detail"] == []


# ── Data quality notes ───────────────────────────────────────────────────────


class TestDataQualityNotes:
    """Data quality characterisation per signal."""

    def test_sufficient_coverage_no_note(self, dispersion_dominant):
        """100% coverage → no data quality note for that signal."""
        result = explain_window(dispersion_dominant)
        # All signals have 100% coverage, no gaps — no notes needed
        assert result["data_quality_notes"] == []

    def test_partial_coverage_noted(self, low_coverage):
        result = explain_window(low_coverage)
        notes = result["data_quality_notes"]
        assert any("partial coverage" in n.lower() for n in notes), \
            f"expected partial coverage note, got: {notes}"

    def test_unusable_signal_noted(self):
        entry = _make_entry()
        entry["signals"]["hr"]["usable"] = False
        entry["signals"]["hr"]["coverage_pct"] = None
        result = explain_window(entry)
        notes = result["data_quality_notes"]
        assert any("unusable" in n.lower() for n in notes), \
            f"expected unusable note, got: {notes}"

    def test_reference_gap_noted(self, reference_gap):
        result = explain_window(reference_gap)
        notes = result["data_quality_notes"]
        assert any("reference gap" in n.lower() for n in notes), \
            f"expected reference gap data quality note, got: {notes}"


# ── Forbidden language ───────────────────────────────────────────────────────


FORBIDDEN_TERMS = [
    "adverse event",
    "clinical deterioration",
    "respiratory distress",
    "cardiac arrest",
    "patient experienced",
    "clinically validated",
    "the model flagged this window because",
    "the isolation forest detected",
    "the model determined",
    "the model identified the cause",
]


class TestForbiddenLanguage:
    """No clinical or causal-claim language in explanations."""

    @pytest.fixture(params=["dispersion_dominant", "change_dominant",
                            "mixed_case", "silent_case"])
    def any_explanation(self, request):
        entry = request.getfixturevalue(request.param)
        return explain_window(entry)

    def test_no_forbidden_terms(self, any_explanation):
        """None of the forbidden clinical or causal terms appear."""
        full_text = str(any_explanation).lower()
        for term in FORBIDDEN_TERMS:
            assert term not in full_text, \
                f"found forbidden term '{term}' in explanation"

    def test_does_not_claim_internal_reasoning(self, any_explanation):
        """The explainer must not claim its rules ARE the model's reasoning."""
        full_text = str(any_explanation).lower()
        # These phrases would imply the rules reconstruct IF internals
        causal_claims = [
            "because the isolation forest",
            "the model's internal",
            "the model decided",
            "the model's reasoning",
            "the forest split",
        ]
        for claim in causal_claims:
            assert claim not in full_text, \
                f"found causal claim '{claim}' in explanation"

    def test_headline_uses_evidence_framing(self, any_explanation):
        """Headlines that mention the model use 'evidence shows' framing."""
        headline = any_explanation["headline"]
        if "The model flagged" in headline:
            assert ("evidence shows" in headline.lower()
                    or "no single-signal rule" in headline.lower()), \
                f"headline lacks evidence-framing: {headline}"


# ── explain_all ──────────────────────────────────────────────────────────────


class TestExplainAll:
    """explain_all processes a list and preserves order."""

    def test_processes_list(self, dispersion_dominant, change_dominant):
        results = explain_all([dispersion_dominant, change_dominant])
        assert len(results) == 2
        assert results[0]["anomaly_rank"] == 1
        assert results[1]["anomaly_rank"] == 10

    def test_empty_list(self):
        assert explain_all([]) == []

    def test_each_has_all_fields(self, dispersion_dominant, silent_case):
        results = explain_all([dispersion_dominant, silent_case])
        for result in results:
            for field in REQUIRED_FIELDS:
                assert field in result


# ── Real evidence patterns ───────────────────────────────────────────────────


class TestRealPatterns:
    """Verify specific real-world evidence patterns produce sensible output."""

    def test_case4_w62_rr_instability_prominent(self, dispersion_dominant):
        """The strongest anomaly is driven by RR instability, not HR change."""
        result = explain_window(dispersion_dominant)
        # RR has the highest dispersion ratio (8.94 / 2.57 ≈ 3.48)
        # It should be the most prominent driver
        assert "RR" in result["headline"]
        # RR narrative should mention the range
        assert "5" in result["signal_narratives"]["rr"]
        assert "33" in result["signal_narratives"]["rr"]

    def test_case8_w2_concurrent_shifts(self, change_dominant):
        """Case 8 window 2 should mention concurrent mean shifts."""
        result = explain_window(change_dominant)
        assert "concurrent" in result["headline"].lower()
        assert "SpO2" in result["headline"]
        assert "RR" in result["headline"]
