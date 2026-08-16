"""Fixtures mirroring the REAL VitalDB schema exactly, for fully offline tests."""

import pandas as pd
import pytest

# The 74 real columns of GET /cases, in order.
CASES_COLUMNS = [
    "caseid", "subjectid", "casestart", "caseend", "anestart", "aneend",
    "opstart", "opend", "adm", "dis", "icu_days", "death_inhosp", "age", "sex",
    "height", "weight", "bmi", "asa", "emop", "department", "optype", "dx",
    "opname", "approach", "position", "ane_type", "preop_htn", "preop_dm",
    "preop_ecg", "preop_pft", "preop_hb", "preop_plt", "preop_pt",
    "preop_aptt", "preop_na", "preop_k", "preop_gluc", "preop_alb",
    "preop_ast", "preop_alt", "preop_bun", "preop_cr", "preop_ph",
    "preop_hco3", "preop_be", "preop_pao2", "preop_paco2", "preop_sao2",
    "cormack", "airway", "tubesize", "dltubesize", "lmasize", "iv1", "iv2",
    "aline1", "aline2", "cline1", "cline2", "intraop_ebl", "intraop_uo",
    "intraop_rbc", "intraop_ffp", "intraop_crystalloid", "intraop_colloid",
    "intraop_ppf", "intraop_mdz", "intraop_ftn", "intraop_rocu",
    "intraop_vecu", "intraop_eph", "intraop_phe", "intraop_epi", "intraop_ca",
]

FULL_PANEL = [
    "Solar8000/HR", "Solar8000/PLETH_SPO2", "Solar8000/ART_MBP",
    "Solar8000/RR_CO2", "Solar8000/BT", "SNUADC/ECG_II", "SNUADC/ART",
    "SNUADC/PLETH",
]


@pytest.fixture
def cases_df():
    """Five synthetic cases with the real 74-column schema.

    caseid 1-3 run 100/200/300 minutes; caseid 4 is too short (30 min) and
    caseid 5 too long (700 min) for the duration band.
    """
    durations_sec = [6000, 12000, 18000, 1800, 42000]
    rows = []
    for i, dur in enumerate(durations_sec, start=1):
        row = dict.fromkeys(CASES_COLUMNS, None)
        row.update(
            caseid=i, subjectid=1000 + i, casestart=0, caseend=dur,
            anestart=-500, aneend=dur - 500, opstart=600, opend=dur - 600,
            age=50.0 + i, sex="M" if i % 2 else "F", asa=2.0,
            department="General surgery", optype="Colorectal",
            ane_type="General", bmi=24.0 + i,
        )
        rows.append(row)
    df = pd.DataFrame(rows, columns=CASES_COLUMNS)
    # cline2 is ~99% missing in the real data; keep it fully missing here.
    df["cline2"] = None
    return df


@pytest.fixture
def trks_df():
    """Track listing mirroring the real /trks schema (caseid, tname, tid).

    Cases 1-5 have the full 8-track panel; case 6 has only 2 tracks.
    Cases 1-2 also carry extra tracks.
    """
    rows = []

    def add(caseid, names):
        for n in names:
            rows.append({
                "caseid": caseid,
                "tname": n,
                "tid": f"tid{caseid}{n}".replace("/", ""),
            })

    add(1, FULL_PANEL + ["Solar8000/NIBP_MBP", "SNUADC/ECG_V5"])
    add(2, FULL_PANEL + ["Solar8000/NIBP_MBP"])
    add(3, FULL_PANEL)
    add(4, FULL_PANEL)
    add(5, FULL_PANEL)
    add(6, ["Solar8000/HR", "Solar8000/PLETH_SPO2"])
    return pd.DataFrame(rows, columns=["caseid", "tname", "tid"])


@pytest.fixture
def labs_df():
    """Lab results with the real 4-column schema."""
    return pd.DataFrame(
        [
            {"caseid": 1, "dt": 594470, "name": "alb", "result": 2.9},
            {"caseid": 1, "dt": 399575, "name": "alb", "result": 3.2},
            {"caseid": 1, "dt": 12614, "name": "hb", "result": 14.1},
            {"caseid": 2, "dt": 5000, "name": "hb", "result": 13.0},
        ],
        columns=["caseid", "dt", "name", "result"],
    )
