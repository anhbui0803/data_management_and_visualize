"""Stage 3 — Cleaning and feature derivation.

Every exclusion is a numbered rule logged with row/patient counts (-> cohort flow,
RECORD item 6.3 / 12.2). Two analytic cohorts are produced:

  * all-encounter cohort  (unit = encounter; repeated patients -> correlated data)
  * first-encounter cohort (unit = patient; first encounter_id per patient, as in
    Strack et al., 2014). ASSUMPTION: encounter_id increases with time; the extract
    has no dates, so this cannot be verified and is stated as a limitation.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import Config
from .utils import LOG, load_parquet, save_parquet

# ---- discharge dispositions after which readmission is impossible ----------
INELIGIBLE_DISCHARGE = {11, 13, 14, 19, 20, 21}  # expired / hospice

MED_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
    "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

# ---- ICD-9 primary-diagnosis groups (Strack et al., 2014, Table 2) ---------
def icd9_group(code) -> str:
    if code is None or (isinstance(code, float) and np.isnan(code)) or pd.isna(code):
        return "Missing"
    s = str(code).strip()
    if not s:
        return "Missing"
    if s[0] in "VvEe":
        return "Other"
    try:
        x = float(s)
    except ValueError:
        return "Invalid"
    if int(x) == 250:
        return "Diabetes"
    if 390 <= x <= 459 or int(x) == 785:
        return "Circulatory"
    if 460 <= x <= 519 or int(x) == 786:
        return "Respiratory"
    if 520 <= x <= 579 or int(x) == 787:
        return "Digestive"
    if 800 <= x <= 999:
        return "Injury"
    if 710 <= x <= 739:
        return "Musculoskeletal"
    if 580 <= x <= 629 or int(x) == 788:
        return "Genitourinary"
    if 140 <= x <= 239:
        return "Neoplasms"
    return "Other"


ICD9_CHAPTERS = [  # finer chapter label for dim_diagnosis
    (1, 139, "Infectious & parasitic"), (140, 239, "Neoplasms"), (240, 279, "Endocrine, nutritional, metabolic"),
    (280, 289, "Blood"), (290, 319, "Mental disorders"), (320, 389, "Nervous system & sense organs"),
    (390, 459, "Circulatory"), (460, 519, "Respiratory"), (520, 579, "Digestive"), (580, 629, "Genitourinary"),
    (630, 679, "Pregnancy & childbirth"), (680, 709, "Skin"), (710, 739, "Musculoskeletal"),
    (740, 759, "Congenital anomalies"), (760, 779, "Perinatal"), (780, 799, "Symptoms & ill-defined"),
    (800, 999, "Injury & poisoning"),
]
ICD9_PATTERN = re.compile(r"^(\d{3}(\.\d{1,2})?|V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?)$")


def icd9_chapter(code) -> str:
    if pd.isna(code):
        return "Missing"
    s = str(code).strip()
    if s[:1] == "V":
        return "Supplementary (V codes)"
    if s[:1] == "E":
        return "External causes (E codes)"
    try:
        x = float(s)
    except ValueError:
        return "Invalid"
    for lo, hi, name in ICD9_CHAPTERS:
        if lo <= x < hi + 1:
            return name
    return "Invalid"


def normalise_icd9(code):
    """Codes appear without leading zeros (e.g. '8' for 008). Pad numeric part to 3 digits."""
    if pd.isna(code):
        return pd.NA
    s = str(code).strip()
    if s[:1] in "VE":
        return s
    head, _, tail = s.partition(".")
    if head.isdigit():
        head = head.zfill(3)
    return head + ("." + tail if tail else "")


# ---- grouped admission / discharge categories ------------------------------
ADMISSION_SOURCE_GROUP = {7: "Emergency room", 1: "Referral", 2: "Referral", 3: "Referral",
                          4: "Transfer", 5: "Transfer", 6: "Transfer", 10: "Transfer", 18: "Transfer",
                          22: "Transfer", 25: "Transfer", 26: "Transfer",
                          9: "Missing", 15: "Missing", 17: "Missing", 20: "Missing", 21: "Missing"}
DISCHARGE_GROUP = {1: "Home", 6: "Home with health service", 8: "Home with health service",
                   2: "Transfer / facility", 3: "Transfer / facility", 4: "Transfer / facility",
                   5: "Transfer / facility", 15: "Transfer / facility", 22: "Transfer / facility",
                   23: "Transfer / facility", 24: "Transfer / facility", 27: "Transfer / facility",
                   28: "Transfer / facility", 29: "Transfer / facility", 30: "Transfer / facility",
                   7: "Left AMA", 18: "Missing", 25: "Missing", 26: "Missing"}
ADMISSION_TYPE_GROUP = {1: "Emergency", 2: "Urgent", 3: "Elective", 7: "Emergency",
                        4: "Other", 5: "Missing", 6: "Missing", 8: "Missing"}

AGE_MID = {f"[{a}-{a + 10})": a + 5 for a in range(0, 100, 10)}


def derive_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = df.copy()
    d["readmit30"] = (d["readmitted"] == "<30").astype(int)
    d["readmit_any"] = (d["readmitted"] != "NO").astype(int)

    # exposure: HbA1c measured during the encounter ("None" = not measured)
    d["hba1c_tested"] = (d["A1Cresult"] != "None").astype(int)
    high = d["A1Cresult"].isin(cfg.high_a1c_values)
    changed = d["change"] == "Ch"
    d["a1c_group"] = np.select(
        [d["hba1c_tested"] == 0, high & changed, high & ~changed],
        ["Not tested", "High, med changed", "High, med not changed"],
        default="Normal / not high",
    )
    d["glu_tested"] = (d["max_glu_serum"] != "None").astype(int)

    # demographics
    d["age_mid"] = d["age"].map(AGE_MID)
    d["age_group"] = pd.cut(d["age_mid"], [0, 30, 60, 100], labels=["<30", "30-60", ">60"], right=False).astype(str)
    d["race_f"] = d["race"].fillna("Missing").astype(str)

    # admission / discharge
    d["adm_source_grp"] = d["admission_source_id"].map(ADMISSION_SOURCE_GROUP).fillna("Other")
    d["discharge_grp"] = d["discharge_disposition_id"].map(DISCHARGE_GROUP).fillna("Other")
    d["adm_type_grp"] = d["admission_type_id"].map(ADMISSION_TYPE_GROUP).fillna("Other")

    # diagnoses
    for k in (1, 2, 3):
        d[f"diag_{k}"] = d[f"diag_{k}"].map(normalise_icd9).astype("string")
        d[f"diag{k}_grp"] = d[f"diag_{k}"].map(icd9_group)

    # specialty: top-k + Other + Missing
    spec = d["medical_specialty"].fillna("Missing").astype(str)
    top = spec[spec != "Missing"].value_counts().head(cfg.specialty_top_k).index
    d["specialty_grp"] = np.where(spec.isin(top) | (spec == "Missing"), spec, "Other")
    d["payer_grp"] = d["payer_code"].fillna("Missing").astype(str)

    # medication activity
    meds = d[MED_COLS]
    d["n_meds_changed"] = meds.isin(["Up", "Down"]).sum(axis=1)
    d["n_meds_active"] = (meds != "No").sum(axis=1)
    d["insulin_status"] = d["insulin"]
    d["med_change"] = changed.astype(int)
    d["diabetes_med"] = (d["diabetesMed"] == "Yes").astype(int)

    # utilisation, capped to tame extreme tails for regression (raw kept)
    for c in ("number_outpatient", "number_emergency", "number_inpatient"):
        d[f"{c}_c"] = d[c].clip(upper=d[c].quantile(0.995))
    d["prior_visits"] = d["number_outpatient"] + d["number_emergency"] + d["number_inpatient"]

    # missingness indicators (for the missing-data analysis)
    for c in ("race", "payer_code", "medical_specialty"):
        d[f"miss_{c}"] = d[c].isna().astype(int)
    return d


def run(cfg: Config) -> dict:
    raw = load_parquet(cfg.interim / "raw_encounters.parquet")
    flow = []

    def log_step(step, rule, before: pd.DataFrame, after: pd.DataFrame):
        flow.append({"step": step, "rule": rule, "rows_before": len(before), "rows_removed": len(before) - len(after),
                     "rows_after": len(after), "patients_after": int(after["patient_nbr"].nunique())})
        LOG.info("  %s %-55s -%6s → %s", step, rule, f"{len(before) - len(after):,}", f"{len(after):,}")

    flow.append({"step": "R0", "rule": "Raw extract", "rows_before": len(raw), "rows_removed": 0,
                 "rows_after": len(raw), "patients_after": int(raw["patient_nbr"].nunique())})

    d1 = raw[raw["gender"] != "Unknown/Invalid"]
    log_step("R1", "Remove gender = Unknown/Invalid", raw, d1)

    d2 = d1[~d1["discharge_disposition_id"].isin(INELIGIBLE_DISCHARGE)]
    log_step("R2", "Remove discharges to death or hospice (cannot be readmitted)", d1, d2)

    d3 = d2.drop_duplicates(subset="encounter_id")
    log_step("R3", "Remove duplicate encounter_id", d2, d3)

    allc = derive_features(d3, cfg).sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
    allc["encounter_seq"] = allc.groupby("patient_nbr").cumcount() + 1
    allc["n_encounters_patient"] = allc.groupby("patient_nbr")["encounter_id"].transform("size")

    first = allc[allc["encounter_seq"] == 1].reset_index(drop=True)
    log_step("R4", "First encounter per patient (first-encounter cohort)", allc, first)

    flow_df = pd.DataFrame(flow)
    flow_df.to_csv(cfg.tables / "cohort_flow.csv", index=False)
    save_parquet(allc, cfg.interim / "cohort_all.parquet")
    save_parquet(first, cfg.interim / "cohort_first.parquet")
    return {"all_rows": len(allc), "first_rows": len(first)}
