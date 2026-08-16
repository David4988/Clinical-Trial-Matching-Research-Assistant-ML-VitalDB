"""Classification of REAL VitalDB track names into physiological families.

This module never invents or renames a track.  It attaches a ``category`` label
to names that already exist in the track inventory, and reports configured
suffixes that matched nothing so a stale map cannot pass silently.
"""

import logging

import pandas as pd

from vitaldb_audit import config

logger = logging.getLogger("vitaldb_audit.signals")

# Inverted lookup: signal suffix -> category.  Built once at import.
_SUFFIX_TO_CATEGORY = {
    suffix: category
    for category, suffixes in config.SIGNAL_SUFFIXES.items()
    for suffix in suffixes
}


def classify_track(tname: str) -> str | None:
    """Return the physiological category for a real track name, or None.

    VitalDB names are "DEVICE/SIGNAL"; classification keys off the SIGNAL
    part.  Returns None for tracks that are not physiological measurements
    (drug pumps, ventilator settings, device quality indices).
    """
    suffix = tname.rsplit("/", 1)[-1]
    return _SUFFIX_TO_CATEGORY.get(suffix)


def build_physiological_candidates(inventory: pd.DataFrame) -> pd.DataFrame:
    """Filter the track inventory down to classified physiological signals.

    Adds ``category`` and ``is_patient_signal``.  The latter is False for tracks
    from devices that measure equipment rather than the patient (FMS fluid
    warmer temperatures), which would otherwise be mistaken for body temp.
    """
    cats = inventory["tname"].apply(classify_track)
    frame = inventory[cats.notna()].copy()
    frame["category"] = cats[cats.notna()].values

    frame["is_patient_signal"] = ~frame["device"].isin(
        config.NON_PATIENT_TEMPERATURE_DEVICES
    )

    logger.info(
        "physiological candidates: %d tracks across %d categories",
        len(frame), frame["category"].nunique(),
    )
    return frame


def unmatched_suffixes(inventory: pd.DataFrame) -> dict[str, list[str]]:
    """Configured suffixes that match no track in the inventory.

    A non-empty result means SIGNAL_SUFFIXES has drifted from the dataset and
    the audit should say so rather than quietly under-reporting.
    """
    present = set(inventory["tname"].str.rsplit("/", n=1).str[-1])
    missing = {}
    for category, suffixes in config.SIGNAL_SUFFIXES.items():
        absent = sorted(suffixes - present)
        if absent:
            missing[category] = absent
    if missing:
        logger.warning("configured suffixes with no matching track: %s", missing)
    return missing


def category_coverage(candidates: pd.DataFrame) -> pd.DataFrame:
    """Per-category summary: how many tracks, and the best-covered one."""
    patient_only = candidates[candidates["is_patient_signal"]]
    rows = []
    for category, group in patient_only.groupby("category"):
        best = group.sort_values(
            ["n_cases", "tname"], ascending=[False, True]
        ).iloc[0]
        rows.append({
            "category": category,
            "n_tracks": len(group),
            "best_tname": best["tname"],
            "best_n_cases": int(best["n_cases"]),
            "best_pct_cases": float(best["pct_cases"]),
            "all_tnames": ", ".join(sorted(group["tname"])),
        })
    frame = pd.DataFrame(rows)
    logger.info("category coverage: %d categories", len(frame))
    return frame
