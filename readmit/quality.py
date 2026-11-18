"""Stage 6 — Quality assurance.

Automated checks organised by the Kahn et al. (2016) harmonized data-quality framework:
  Conformance (value / relational / computational), Completeness, Plausibility
  × context Verification (internal expectations).

Outputs
  tables/dq_checks.csv     one row per check (category, severity, passed, observed, expected)
  tables/dq_scorecard.csv  one row per raw column: completeness, validity, fitness verdict
A failed check with severity "error" stops the pipeline.
"""
from __future__ import annotations

import pandas as pd

from .clean import INELIGIBLE_DISCHARGE, MED_COLS, AGE_MID
from .config import Config, EXPECTED, DATA_FILE
from .storage import TABLES
from .utils import LOG, load_parquet, read_json

ALLOWED = {
    "A1Cresult": {"None", "Norm", ">7", ">8"},
    "max_glu_serum": {"None", "Norm", ">200", ">300"},
    "readmitted": {"NO", "<30", ">30"},
    "gender": {"Female", "Male", "Unknown/Invalid"},
    "change": {"No", "Ch"},
    "diabetesMed": {"No", "Yes"},
    "age": set(AGE_MID),
    "race": {"Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"},
    **{m: {"No", "Steady", "Up", "Down"} for m in MED_COLS},
}
RANGES = {  # plausibility ranges (inclusive)
    "time_in_hospital": (1, 14), "num_lab_procedures": (0, 200), "num_procedures": (0, 10),
    "num_medications": (0, 100), "number_outpatient": (0, 100), "number_emergency": (0, 100),
    "number_inpatient": (0, 30), "number_diagnoses": (1, 20),
}
UNUSABLE_COMPLETENESS = 0.50   # < 50% complete -> not fit for modelling as-is


class Checks:
    def __init__(self):
        self.rows = []

    def add(self, check_id, category, subcategory, description, passed, observed, expected, severity="error"):
        self.rows.append({"check_id": check_id, "category": category, "subcategory": subcategory,
                          "description": description, "severity": severity, "passed": bool(passed),
                          "observed": observed, "expected": expected})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def raw_checks(ck: Checks, raw: pd.DataFrame, cfg: Config, prov: dict) -> None:
    full = not cfg.smoke
    # --- Conformance: value
    for col, allowed in ALLOWED.items():
        vals = set(raw[col].dropna().astype(str).unique())
        bad = sorted(vals - allowed)
        ck.add(f"CV-{col}", "Conformance", "Value", f"{col} values in allowed set", not bad,
               ",".join(bad) or "ok", "subset of allowed")
    icd_bad = 0
    for k in (1, 2, 3):
        s = raw[f"diag_{k}"].dropna().astype(str)
        icd_bad += int((~s.str.match(r"^(\d{1,3}(\.\d{1,2})?|V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?)$")).sum())
    ck.add("CV-icd9", "Conformance", "Value", "diag_1..3 match an ICD-9 pattern", icd_bad == 0, icd_bad, 0, "warn")
    # --- Conformance: relational
    ck.add("CR-encounter-pk", "Conformance", "Relational", "encounter_id unique",
           raw["encounter_id"].is_unique, int(raw["encounter_id"].duplicated().sum()), 0)
    # --- Verification of the extract against the published spec (full run only)
    if full:
        f = prov["files"][DATA_FILE]
        ck.add("CC-shape", "Conformance", "Computational", "raw shape equals UCI spec",
               f["shape_matches_uci"], f"{f['rows']}x{f['cols']}",
               f"{EXPECTED[DATA_FILE]['rows']}x{EXPECTED[DATA_FILE]['cols']}")
        n_none = int((raw["A1Cresult"] == "None").sum())
        ck.add("CC-a1c-none", "Conformance", "Computational",
               "A1Cresult 'None' kept as a value (not parsed to NaN)", n_none == 84_748, n_none, 84_748)
        n_inel = int(raw["discharge_disposition_id"].isin(INELIGIBLE_DISCHARGE).sum())
        ck.add("CC-ineligible", "Conformance", "Computational", "death/hospice discharges count",
               n_inel == 2_423, n_inel, 2_423, "warn")
    else:
        ck.add("CC-a1c-none", "Conformance", "Computational", "A1Cresult has no NaN (None kept as value)",
               raw["A1Cresult"].notna().all(), int(raw["A1Cresult"].isna().sum()), 0)
    # --- Plausibility: atemporal ranges
    for col, (lo, hi) in RANGES.items():
        out = int(((raw[col] < lo) | (raw[col] > hi)).sum())
        ck.add(f"PA-{col}", "Plausibility", "Atemporal", f"{col} within [{lo},{hi}]", out == 0, out, 0, "warn")
    # --- Completeness
    ck.add("CO-outcome", "Completeness", "—", "readmitted fully observed", raw["readmitted"].notna().all(),
           int(raw["readmitted"].isna().sum()), 0)


def cohort_checks(ck: Checks, allc: pd.DataFrame, first: pd.DataFrame) -> None:
    ck.add("CC-no-ineligible", "Conformance", "Computational", "no death/hospice discharges remain",
           not allc["discharge_disposition_id"].isin(INELIGIBLE_DISCHARGE).any(),
           int(allc["discharge_disposition_id"].isin(INELIGIBLE_DISCHARGE).sum()), 0)
    ck.add("CR-first-unique", "Conformance", "Relational", "first-encounter cohort has one row per patient",
           first["patient_nbr"].is_unique, int(first["patient_nbr"].duplicated().sum()), 0)
    ck.add("CC-outcome-binary", "Conformance", "Computational", "readmit30 is 0/1",
           set(allc["readmit30"].unique()) <= {0, 1}, sorted(allc["readmit30"].unique().tolist()), "[0, 1]")
    ck.add("CC-exposure-binary", "Conformance", "Computational", "hba1c_tested is 0/1 with both levels present",
           set(allc["hba1c_tested"].unique()) == {0, 1}, sorted(allc["hba1c_tested"].unique().tolist()), "[0, 1]")
    unk = int((allc["diag1_grp"] == "Invalid").sum())
    ck.add("CV-diag1-group", "Conformance", "Value", "every primary diagnosis maps to a group", unk == 0, unk, 0, "warn")


def warehouse_checks(ck: Checks, cfg: Config) -> None:
    p = cfg.warehouse / "parquet"
    t = {name: load_parquet(p / f"{name}.parquet") for name in TABLES}
    f = t["fact_encounter"]
    for fk, dim in (("admission_type_id", "dim_admission_type"),
                    ("discharge_disposition_id", "dim_discharge_disposition"),
                    ("admission_source_id", "dim_admission_source")):
        orphans = int((~f[fk].isin(t[dim]["id"])).sum())
        ck.add(f"CR-fk-{fk}", "Conformance", "Relational", f"fact.{fk} → {dim}.id", orphans == 0, orphans, 0)
    orphans = int((~f["patient_nbr"].isin(t["dim_patient"]["patient_nbr"])).sum())
    ck.add("CR-fk-patient", "Conformance", "Relational", "fact.patient_nbr → dim_patient", orphans == 0, orphans, 0)
    codes = set(t["dim_diagnosis"]["icd9_code"].astype(str))
    orphans = int((~f["diag_1"].dropna().astype(str).isin(codes)).sum())
    ck.add("CR-fk-diag", "Conformance", "Relational", "fact.diag_1 → dim_diagnosis", orphans == 0, orphans, 0)
    bad_fmt = int((~t["dim_diagnosis"]["is_valid_format"]).sum())
    ck.add("CV-dim-diag-format", "Conformance", "Value", "normalised ICD-9 codes match pattern",
           bad_fmt == 0, bad_fmt, 0, "warn")
    orphans = int((~t["bridge_encounter_medication"]["encounter_id"].isin(f["encounter_id"])).sum())
    ck.add("CR-fk-bridge", "Conformance", "Relational", "bridge.encounter_id → fact", orphans == 0, orphans, 0)
    dp = t["dim_patient"]
    ck.add("PT-race-consistent", "Plausibility", "Temporal",
           "race constant across a patient's encounters", not dp["race_inconsistent"].any(),
           int(dp["race_inconsistent"].sum()), 0, "warn")
    ck.add("PT-gender-consistent", "Plausibility", "Temporal",
           "gender constant across a patient's encounters", not dp["gender_inconsistent"].any(),
           int(dp["gender_inconsistent"].sum()), 0, "warn")


def scorecard(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in raw.columns:
        s = raw[c]
        completeness = s.notna().mean()
        if c in ALLOWED:
            validity = s.dropna().astype(str).isin(ALLOWED[c]).mean() if s.notna().any() else float("nan")
        elif c in RANGES:
            lo, hi = RANGES[c]
            validity = s.between(lo, hi).mean()
        elif c.startswith("diag_"):
            validity = s.dropna().astype(str).str.match(r"^(\d{1,3}(\.\d{1,2})?|V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?)$").mean()
        else:
            validity = float("nan")
        not_measured = float((s == "None").mean()) if c in ("A1Cresult", "max_glu_serum") else 0.0
        if completeness < UNUSABLE_COMPLETENESS:
            verdict = "Not fit (exclude / indicator only)"
        elif completeness < 0.95:
            verdict = "Fit with missing-data handling"
        else:
            verdict = "Fit"
        rows.append({"column": c, "completeness": round(completeness, 4),
                     "validity": None if pd.isna(validity) else round(float(validity), 4),
                     "not_measured_share": round(not_measured, 4), "verdict": verdict})
    return pd.DataFrame(rows)


def run(cfg: Config) -> pd.DataFrame:
    raw = load_parquet(cfg.interim / "raw_encounters.parquet")
    allc = load_parquet(cfg.interim / "cohort_all.parquet")
    first = load_parquet(cfg.interim / "cohort_first.parquet")
    prov = read_json(cfg.logs / "provenance.json")
    ck = Checks()
    raw_checks(ck, raw, cfg, prov)
    cohort_checks(ck, allc, first)
    warehouse_checks(ck, cfg)
    res = ck.frame()
    res.to_csv(cfg.tables / "dq_checks.csv", index=False)
    scorecard(raw).to_csv(cfg.tables / "dq_scorecard.csv", index=False)
    n_fail = int((~res["passed"]).sum())
    errors = res[(~res["passed"]) & (res["severity"] == "error")]
    LOG.info("DQ checks: %d run, %d failed (%d errors, %d warnings)", len(res), n_fail, len(errors), n_fail - len(errors))
    for _, r in res[~res["passed"]].iterrows():
        LOG.warning("  DQ %-5s %-22s %s | observed=%s", r["severity"].upper(), r["check_id"], r["description"], r["observed"])
    if len(errors):
        raise RuntimeError(f"{len(errors)} data-quality error(s) — see {cfg.tables / 'dq_checks.csv'}")
    return res
