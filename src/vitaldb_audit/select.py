"""Deterministic selection of candidate cases for deeper inspection.

A case qualifies only if it carries the complete 8-track panel AND falls
within the experimental duration band.  The duration band (60-600 min) is
a hackathon case-selection heuristic, NOT a VitalDB-defined rule.

Selection is fully deterministic — same inputs always yield the same case IDs.
Ranking: full-panel + in-duration-band → most physiological categories →
most tracks → lowest caseid as tie-break.
"""

import logging

import pandas as pd

from vitaldb_audit import config, signals

logger = logging.getLogger("vitaldb_audit.select")


class NoCandidatesError(RuntimeError):
    """Raised when no case satisfies the required panel and duration band."""


def build_case_track_sets(trks: pd.DataFrame) -> dict[int, set[str]]:
    """Map each caseid to the set of REAL track names it carries."""
    grouped = trks.groupby("caseid")["tname"].apply(set)
    return {int(caseid): names for caseid, names in grouped.items()}


def score_cases(
    case_tracks: dict[int, set[str]],
    cases: pd.DataFrame,
) -> pd.DataFrame:
    """Score every case on panel completeness, signal breadth and duration."""
    panel = set(config.REQUIRED_PANEL)
    durations = (
        ((cases["caseend"] - cases["casestart"]) / 60.0)
        .set_axis(cases["caseid"])
        .to_dict()
    )

    rows = []
    for caseid, names in sorted(case_tracks.items()):
        categories = {
            category
            for category in (signals.classify_track(n) for n in names)
            if category is not None
        }
        minutes = durations.get(caseid)
        in_band = (
            minutes is not None
            and not pd.isna(minutes)
            and config.MIN_CASE_MINUTES <= minutes <= config.MAX_CASE_MINUTES
        )
        has_panel = panel.issubset(names)
        rows.append({
            "caseid": caseid,
            "n_tracks": len(names),
            "n_physio_categories": len(categories),
            "has_full_panel": has_panel,
            "case_minutes": round(float(minutes), 2) if minutes is not None and not pd.isna(minutes) else None,
            "in_duration_band": bool(in_band),
            "qualifies": bool(has_panel and in_band),
        })

    frame = pd.DataFrame(rows, columns=[
        "caseid", "n_tracks", "n_physio_categories", "has_full_panel",
        "case_minutes", "in_duration_band", "qualifies",
    ])
    logger.info(
        "scored %d cases; %d carry the full panel; %d qualify overall",
        len(frame), int(frame["has_full_panel"].sum()), int(frame["qualifies"].sum()),
    )
    return frame


def select_candidates(scored: pd.DataFrame, n: int = config.N_CANDIDATE_CASES) -> list[int]:
    """Pick the top-n candidates from the scored table.

    Only cases that qualify (full panel + within duration band) are considered.
    """
    qualifying = scored[scored["qualifies"]]
    if qualifying.empty:
        raise NoCandidatesError(
            f"no case carries the full required panel ({config.REQUIRED_PANEL}) within "
            f"{config.MIN_CASE_MINUTES}-{config.MAX_CASE_MINUTES} minutes"
        )

    ranked = qualifying.sort_values(
        ["n_physio_categories", "n_tracks", "caseid"],
        ascending=[False, False, True],
    )
    selected = [int(c) for c in ranked["caseid"].head(n)]

    if len(selected) < n:
        logger.warning(
            "only %d qualifying cases available, requested %d", len(selected), n
        )
    logger.info("selected candidate cases: %s", selected)
    return selected
