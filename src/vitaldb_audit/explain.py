"""Deterministic rule-based XAI layer: evidence dict → structured explanation.

This module is a TRANSLATION LAYER, not a model.  It reads the structured
evidence objects produced by ``evidence.py`` and emits plain-English sentences
that describe what the signals did.  It does not reconstruct the Isolation
Forest's internal decision path, it does not assign causes, and it does not
produce clinical language.

The wording is always:

    "The model flagged this window.  The supplied evidence shows …"

Never:

    "The model flagged this window BECAUSE …"

This distinction matters: the rule-based layer surfaces the most salient
evidence for a human reviewer, but the Isolation Forest is a multivariate
scorer whose internal splits are not directly readable.  These rules are an
evidence-based summary, not a faithful reconstruction.

Architecture
------------
One public function:

    explain_window(evidence_entry: dict) -> dict

Pure function.  No file I/O, no side effects, no model calls.
"""

# ── Constants ────────────────────────────────────────────────────────────────

SIGNALS = ("hr", "spo2", "rr")

SIGNAL_LABELS = {
    "hr": "HR",
    "spo2": "SpO2",
    "rr": "RR",
}

SIGNAL_UNITS = {
    "hr": "bpm",
    "spo2": "%",
    "rr": "breaths/min",
}

DISCLAIMER = (
    "This is a statistical description of unusual monitoring data, "
    "not a clinical diagnosis or adverse-event determination."
)

# Coverage thresholds — these are DATA-QUALITY descriptors, not clinical.
_PARTIAL_COVERAGE_PCT = 95.0   # below this → note the gap
_LOW_COVERAGE_PCT = 70.0       # below this → explicit caveat

# Priority weights for driver ranking.
_DISPERSION_WEIGHT = 2
_CHANGE_WEIGHT = 1


# ── Helpers ──────────────────────────────────────────────────────────────────


def _fmt(value, spec=".2f", fallback="n/a"):
    """Format a numeric value, returning *fallback* for None."""
    if value is None:
        return fallback
    return format(value, spec)


def _abs_or_zero(value):
    """abs() that treats None as 0."""
    if value is None:
        return 0.0
    return abs(value)


def _safe_div(numerator, denominator):
    """numerator / denominator, returning 0 when either is None or denom is 0."""
    if numerator is None or denominator is None or denominator == 0:
        return 0.0
    return numerator / denominator


# ── Per-signal narrative ─────────────────────────────────────────────────────


def _signal_narrative(signal: str, evidence_entry: dict) -> str:
    """One descriptive sentence about what this signal did in this window."""
    label = SIGNAL_LABELS[signal]
    unit = SIGNAL_UNITS[signal]
    sig = evidence_entry["signals"][signal]
    obs = evidence_entry["observations"]
    disp = obs["dispersion_basis"][signal]
    chg = obs["change_basis"][signal]

    dispersion_unusual = obs.get(f"{signal}_dispersion_unusual")
    changed = obs.get(f"{signal}_changed")

    mean = sig["current_mean"]
    std = sig["std"]
    vmin = sig["min"]
    vmax = sig["max"]
    delta = sig["delta"]
    direction = obs.get(f"{signal}_direction")
    trend_score = chg.get("trend_score")
    threshold = disp.get("threshold")
    prev_mean = sig["previous_usable_mean"]

    # Case 1: dispersion unusual AND changed
    if dispersion_unusual and changed:
        dir_word = direction or "shifted"
        if dir_word == "increase":
            dir_phrase = f"shifted up by {_fmt(_abs_or_zero(delta))}"
        elif dir_word == "decrease":
            dir_phrase = f"shifted down by {_fmt(_abs_or_zero(delta))}"
        else:
            dir_phrase = f"shifted by {_fmt(_abs_or_zero(delta))}"

        parts = [
            f"{label} showed unusually large within-window variability "
            f"(std {_fmt(std)}, range {_fmt(vmin, '.0f')}–{_fmt(vmax, '.0f')} {unit}, "
            f"case threshold {_fmt(threshold)})"
        ]
        parts.append(
            f"and its mean {dir_phrase} {unit} from the reference window"
        )
        if trend_score is not None:
            parts[-1] += f" (trend score {_fmt(trend_score, '+.2f')} sd)"
        return " ".join(parts) + "."

    # Case 2: dispersion unusual only
    if dispersion_unusual and not changed:
        qualifier = (
            "while its mean remained within normal variation"
            if changed is not None
            else "while its mean change could not be assessed"
        )
        return (
            f"{label} showed unusually large within-window variability "
            f"(std {_fmt(std)}, range {_fmt(vmin, '.0f')}–{_fmt(vmax, '.0f')} {unit}, "
            f"case threshold {_fmt(threshold)}), {qualifier}."
        )

    # Case 3: changed only (no unusual dispersion)
    if changed and not dispersion_unusual:
        dir_word = direction or "shifted"
        if dir_word == "increase":
            verb = "increased"
        elif dir_word == "decrease":
            verb = "decreased"
        else:
            verb = "shifted"

        sentence = (
            f"{label} mean {verb} by {_fmt(_abs_or_zero(delta))} {unit} "
            f"from the reference window"
        )
        if prev_mean is not None:
            sentence += f" ({_fmt(prev_mean)} → {_fmt(mean)})"
        if trend_score is not None:
            sentence += f", trend score {_fmt(trend_score, '+.2f')} sd"
        sentence += "."
        return sentence

    # Case 4: neither
    if sig.get("usable") is False or mean is None:
        return f"{label} was not usable in this window."

    return f"{label} was stable and showed typical within-window spread."


# ── Primary drivers ──────────────────────────────────────────────────────────


def _rank_drivers(evidence_entry: dict) -> list[dict]:
    """Rank signals by explanatory salience.  Returns a sorted list of dicts.

    This is a heuristic ranking of the EVIDENCE, not a reconstruction of the
    Isolation Forest's internal feature importances.
    """
    obs = evidence_entry["observations"]
    drivers = []

    for signal in SIGNALS:
        dispersion_unusual = obs.get(f"{signal}_dispersion_unusual") or False
        changed = obs.get(f"{signal}_changed") or False

        priority = (
            (_DISPERSION_WEIGHT if dispersion_unusual else 0)
            + (_CHANGE_WEIGHT if changed else 0)
        )
        if priority == 0:
            continue

        # Magnitude tie-break
        disp = obs["dispersion_basis"][signal]
        chg = obs["change_basis"][signal]
        magnitude = 0.0
        if dispersion_unusual and disp.get("threshold") is not None:
            magnitude = max(magnitude, _safe_div(disp["current_std"],
                                                  disp["threshold"]))
        if changed:
            magnitude = max(magnitude, _abs_or_zero(chg.get("trend_score")))

        drivers.append({
            "signal": signal,
            "label": SIGNAL_LABELS[signal],
            "priority": priority,
            "magnitude": magnitude,
            "dispersion_unusual": dispersion_unusual,
            "changed": changed,
        })

    # Sort: highest priority first, then highest magnitude
    drivers.sort(key=lambda d: (d["priority"], d["magnitude"]), reverse=True)
    return drivers


def _driver_type(drivers: list[dict]) -> str:
    """Classify the dominant evidence pattern."""
    if not drivers:
        return "undetermined"

    has_dispersion = any(d["dispersion_unusual"] for d in drivers)
    has_change = any(d["changed"] for d in drivers)

    if has_dispersion and has_change:
        # Check if it's the SAME signal or different signals
        disp_signals = {d["signal"] for d in drivers if d["dispersion_unusual"]}
        chg_signals = {d["signal"] for d in drivers if d["changed"]}
        # If there's any overlap or both present, it's mixed
        return "mixed"
    if has_dispersion:
        return "instability"
    if has_change:
        return "mean_shift"
    return "undetermined"


# ── Headline ─────────────────────────────────────────────────────────────────


def _headline(evidence_entry: dict, drivers: list[dict]) -> str:
    """One-sentence headline summarising what the evidence shows."""
    obs = evidence_entry["observations"]

    disp_signals = [d for d in drivers if d["dispersion_unusual"]]
    chg_signals = [d for d in drivers if d["changed"]]
    # Signals that changed but do NOT also have unusual dispersion
    chg_only = [d for d in chg_signals
                if not d["dispersion_unusual"]]

    n_disp = len(disp_signals)
    n_chg_only = len(chg_only)

    # Priority 1: multiple dispersion-unusual signals
    if n_disp >= 2:
        names = " and ".join(d["label"] for d in disp_signals)
        strongest = disp_signals[0]["label"]  # already sorted by magnitude
        return (
            f"The model flagged this window. The supplied evidence shows "
            f"within-window instability in {names}, "
            f"most prominently {strongest}."
        )

    # Priority 2: single dispersion-unusual + change in another signal
    if n_disp == 1 and n_chg_only >= 1:
        disp_name = disp_signals[0]["label"]
        chg_name = chg_only[0]["label"]
        return (
            f"The model flagged this window. The supplied evidence shows "
            f"instability in {disp_name} alongside a mean shift in {chg_name}."
        )

    # Priority 3: single dispersion-unusual only
    if n_disp == 1:
        disp_name = disp_signals[0]["label"]
        return (
            f"The model flagged this window. The supplied evidence shows "
            f"{disp_name} had unusually large within-window variability."
        )

    # Priority 4: multiple changed, no unusual dispersion
    if n_chg_only >= 2:
        names = " and ".join(d["label"] for d in chg_only)
        return (
            f"The model flagged this window. The supplied evidence shows "
            f"concurrent mean shifts in {names}."
        )

    # Priority 5: single changed
    if n_chg_only == 1:
        d = chg_only[0]
        direction = obs.get(f"{d['signal']}_direction", "shifted")
        if direction == "increase":
            dir_word = "upward"
        elif direction == "decrease":
            dir_word = "downward"
        else:
            dir_word = ""
        shift_phrase = f"{d['label']} mean shifted {dir_word}".strip()
        return (
            f"The model flagged this window. The supplied evidence shows "
            f"{shift_phrase}."
        )

    # Priority 6: no single-signal explanation
    return (
        "The model flagged this window, but no single-signal rule "
        "explains the result."
    )


# ── Supporting detail ────────────────────────────────────────────────────────


def _supporting_detail(evidence_entry: dict) -> list[str]:
    """Context sentences drawn from the evidence object."""
    details = []
    obs = evidence_entry["observations"]

    # Degenerate dispersion references
    for signal in SIGNALS:
        disp = obs["dispersion_basis"][signal]
        if disp.get("degenerate_reference") is True and disp.get("unusual") is True:
            label = SIGNAL_LABELS[signal]
            details.append(
                f"{label} dispersion is compared against a case where this "
                f"signal is flat in nearly every other window, so any non-zero "
                f"spread is flagged as atypical."
            )

    # Coverage caveats
    for signal in SIGNALS:
        sig = evidence_entry["signals"][signal]
        cov = sig.get("coverage_pct")
        label = SIGNAL_LABELS[signal]
        if cov is not None and cov < _PARTIAL_COVERAGE_PCT:
            if cov < _LOW_COVERAGE_PCT:
                details.append(
                    f"{label} coverage was {cov:.1f}% in this window "
                    f"(substantial missing data)."
                )
            else:
                details.append(
                    f"{label} coverage was {cov:.1f}% in this window "
                    f"(some samples missing)."
                )

    # Reference-window gaps
    for signal in SIGNALS:
        sig = evidence_entry["signals"][signal]
        gap = sig.get("windows_since_reference")
        label = SIGNAL_LABELS[signal]
        if gap is not None and gap > 1:
            details.append(
                f"{label} compares back {gap} windows "
                f"(a gap in usable monitoring, not {gap * 5} continuous minutes)."
            )

    # Unusable signals
    for signal in SIGNALS:
        sig = evidence_entry["signals"][signal]
        label = SIGNAL_LABELS[signal]
        if sig.get("usable") is False:
            details.append(
                f"{label} was not usable in this window; its values should "
                f"be interpreted with caution."
            )

    return details


# ── Data quality notes ───────────────────────────────────────────────────────


def _data_quality_notes(evidence_entry: dict) -> list[str]:
    """Explicit data-quality characterisation per signal."""
    notes = []

    for signal in SIGNALS:
        sig = evidence_entry["signals"][signal]
        label = SIGNAL_LABELS[signal]
        cov = sig.get("coverage_pct")
        usable = sig.get("usable")
        gap = sig.get("windows_since_reference")

        if usable is False or cov is None:
            notes.append(f"{label}: unusable in this window.")
            continue

        parts = []
        if cov is not None:
            if cov >= _PARTIAL_COVERAGE_PCT:
                pass  # sufficient coverage is the default — no note needed
            elif cov >= _LOW_COVERAGE_PCT:
                parts.append(f"partial coverage ({cov:.1f}%)")
            else:
                parts.append(f"low coverage ({cov:.1f}%)")

        if gap is not None and gap > 1:
            parts.append(f"reference gap of {gap} windows")

        if parts:
            notes.append(f"{label}: {'; '.join(parts)}.")

    return notes


# ── Public API ───────────────────────────────────────────────────────────────


def explain_window(evidence_entry: dict) -> dict:
    """Convert one evidence object into a structured explanation.

    Pure function: evidence dict in, explanation dict out.

    The explanation describes what the evidence shows.  It does NOT claim to
    reconstruct the Isolation Forest's internal decision path, and it does NOT
    produce clinical interpretations.
    """
    drivers = _rank_drivers(evidence_entry)

    signal_narratives = {
        signal: _signal_narrative(signal, evidence_entry)
        for signal in SIGNALS
    }

    primary_driver_labels = [d["label"] for d in drivers]

    return {
        "case_id": evidence_entry["case_id"],
        "window_index": evidence_entry["window_index"],
        "anomaly_rank": evidence_entry["anomaly_rank"],
        "anomaly_score": evidence_entry["anomaly_score"],
        "time_label": evidence_entry["time_range"]["label"],
        "headline": _headline(evidence_entry, drivers),
        "signal_narratives": signal_narratives,
        "primary_drivers": primary_driver_labels,
        "driver_type": _driver_type(drivers),
        "supporting_detail": _supporting_detail(evidence_entry),
        "data_quality_notes": _data_quality_notes(evidence_entry),
        "interpretation": DISCLAIMER,
    }


def explain_all(evidence_list: list[dict]) -> list[dict]:
    """Run explain_window on every evidence entry, preserving rank order."""
    return [explain_window(entry) for entry in evidence_list]
