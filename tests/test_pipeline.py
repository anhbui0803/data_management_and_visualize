"""Pipeline validation tests.

  pytest -q tests/                      # unit tests + smoke run (needs the raw files)
  READMIT_DATA_DIR=... pytest -q        # point at the raw folder if not in ../data/raw

Unit tests run without data. The smoke test runs the full pipeline on a patient sample
and asserts that every stage succeeds and every expected artifact exists.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readmit.clean import icd9_group, normalise_icd9, derive_features, INELIGIBLE_DISCHARGE  # noqa: E402
from readmit.config import Config, find_raw_dir  # noqa: E402
from readmit.missing import rubin_pool  # noqa: E402
from readmit.models import collapse_rare, e_value, expit, marginal_rd_from_logit  # noqa: E402
from readmit.risk import cluster_bootstrap_auc  # noqa: E402
from readmit.stats_tests import repeat_patient_profile, within_patient_correlation  # noqa: E402


# ----------------------------------------------------------------- unit tests
@pytest.mark.parametrize("code,group", [
    ("250.83", "Diabetes"), ("250", "Diabetes"), ("428", "Circulatory"), ("785", "Circulatory"),
    ("486", "Respiratory"), ("786", "Respiratory"), ("562", "Digestive"), ("820", "Injury"),
    ("715", "Musculoskeletal"), ("599", "Genitourinary"), ("788", "Genitourinary"), ("174", "Neoplasms"),
    ("V45", "Other"), ("E909", "Other"), ("008", "Other"), (None, "Missing"), (np.nan, "Missing"),
])
def test_icd9_group(code, group):
    assert icd9_group(code) == group


def test_normalise_icd9():
    assert normalise_icd9("8") == "008"
    assert normalise_icd9("38.0") == "038.0"
    assert normalise_icd9("V57") == "V57"
    assert pd.isna(normalise_icd9(pd.NA))


def test_none_is_not_missing():
    """'None' in A1Cresult means 'not measured' and must give hba1c_tested = 0, never NaN."""
    df = pd.DataFrame({
        "readmitted": ["<30", "NO", ">30"], "A1Cresult": ["None", ">8", "Norm"], "change": ["No", "Ch", "No"],
        "max_glu_serum": ["None", "None", ">200"], "age": ["[70-80)", "[50-60)", "[20-30)"],
        "race": pd.array(["Caucasian", pd.NA, "Asian"], dtype="string"), "admission_source_id": [7, 1, 4],
        "discharge_disposition_id": [1, 6, 3], "admission_type_id": [1, 3, 2],
        "diag_1": pd.array(["250.8", "428", pd.NA], dtype="string"), "diag_2": pd.array(["8", pd.NA, "V45"], dtype="string"),
        "diag_3": pd.array([pd.NA, "786", "820"], dtype="string"),
        "medical_specialty": pd.array([pd.NA, "Cardiology", "Surgery"], dtype="string"),
        "payer_code": pd.array([pd.NA, "MC", "BC"], dtype="string"), "diabetesMed": ["Yes", "Yes", "No"],
        "number_outpatient": [0, 1, 2], "number_emergency": [0, 0, 1], "number_inpatient": [1, 0, 3],
        **{m: ["No", "Up", "Steady"] for m in ["metformin", "insulin"]},
    })
    from readmit import clean
    df_full = df.copy()
    for m in clean.MED_COLS:
        if m not in df_full:
            df_full[m] = "No"
    cfg = Config(raw_dir=Path("."), out_dir=Path("."))
    d = derive_features(df_full, cfg)
    assert d["hba1c_tested"].tolist() == [0, 1, 1]
    assert d["a1c_group"].tolist() == ["Not tested", "High, med changed", "Normal / not high"]
    assert d["readmit30"].tolist() == [1, 0, 0]
    assert d["diag2_grp"].tolist() == ["Other", "Missing", "Other"]


def test_ineligible_codes():
    assert INELIGIBLE_DISCHARGE == {11, 13, 14, 19, 20, 21}


def test_collapse_rare_keeps_reference_and_removes_sparse():
    df = pd.DataFrame({"x": ["A"] * 200 + ["B"] * 3 + ["C"] * 100, "readmit30": [0, 1] * 100 + [0] * 3 + [1, 0] * 50})
    out = collapse_rare(df, cols=["x"], min_n=50, min_events=5)
    assert "B" not in set(out["x"])
    assert {"A", "C"} <= set(out["x"])


def test_rubin_pool_identical_estimates():
    r = rubin_pool([0.1, 0.1, 0.1], [0.01, 0.01, 0.01])
    assert np.isclose(r["coef"], 0.1) and np.isclose(r["se"], 0.1) and r["fmi"] == 0


def test_e_value_null_ci_is_one():
    assert e_value(0.95, 0.8, 1.1, 0.1)["e_value_ci"] == 1.0
    assert e_value(2.0, 1.5, 2.5, 0.1)["e_value_point"] == pytest.approx(2 + np.sqrt(2))


def test_marginal_rd_matches_simulated_truth():
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.normal(size=n)
    t = rng.binomial(1, 0.4, size=n)
    p = expit(-2 + 0.5 * x - 0.4 * t)
    df = pd.DataFrame({"readmit30": rng.binomial(1, p), "hba1c_tested": t, "x": x})
    truth = (expit(-2 + 0.5 * x - 0.4) - expit(-2 + 0.5 * x)).mean()
    r = marginal_rd_from_logit("readmit30 ~ hba1c_tested + x", df)
    assert r["rd_low"] < truth < r["rd_high"]
    assert r["rd"] == pytest.approx(r["risk_tested"] - r["risk_untested"])


def test_cluster_bootstrap_auc_brackets_point_estimate():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.3, 3000)
    p = np.clip(0.3 + 0.2 * (y - 0.3) + rng.normal(0, 0.2, 3000), 0, 1)
    g = np.repeat(np.arange(1500), 2)
    lo, hi = cluster_bootstrap_auc(y, p, g, n_boot=200)
    assert lo < roc_auc_score(y, p) < hi


def test_repeat_patient_tables():
    allc = pd.DataFrame({"encounter_id": range(6), "patient_nbr": [1, 1, 1, 2, 3, 3],
                         "encounter_seq": [1, 2, 3, 1, 1, 2], "n_encounters_patient": [3, 3, 3, 1, 2, 2],
                         "readmit30": [1, 1, 0, 0, 0, 1], "hba1c_tested": [0, 1, 0, 1, 0, 0]})
    prof = repeat_patient_profile(allc)
    single = prof[(prof["grouping"] == "patient_type") & (prof["level"] == "Single-stay patient")].iloc[0]
    assert single["encounters"] == 1 and single["patients"] == 1
    assert prof.loc[prof["grouping"] == "sequence", "encounters"].sum() == 6
    corr = within_patient_correlation(allc).iloc[0]
    assert corr["consecutive_pairs"] == 3 and corr["patients_with_pairs"] == 2


# ----------------------------------------------------------------- smoke test
def _raw_dir_or_skip():
    try:
        return find_raw_dir()
    except FileNotFoundError:
        pytest.skip("raw data not found — set READMIT_DATA_DIR")


EXPECTED_ARTIFACTS = [
    "logs/provenance.json", "logs/run_manifest.json", "tables/cohort_flow.csv", "tables/profile_columns.csv",
    "tables/dq_checks.csv", "tables/dq_scorecard.csv", "tables/readmit_by_segment.csv",
    "tables/tests_categorical.csv", "tables/tests_numeric.csv", "tables/power_analysis.csv",
    "tables/model_exposure_summary.csv", "tables/model_hba1c_by_diag.csv", "tables/ipw_balance.csv",
    "tables/missing_strategy_compare.csv", "tables/risk_metrics.csv", "tables/risk_caseload.csv",
    "models/risk_logit.joblib", "models/risk_hgb.joblib", "dashboard.html",
    "tables/model_risk_differences.csv", "tables/repeat_patients.csv", "tables/within_patient_corr.csv",
    "tables/data_dictionary.csv", "warehouse/er_diagram.dot",
]


def test_smoke_pipeline(tmp_path):
    raw = _raw_dir_or_skip()
    import run_pipeline
    out = tmp_path / "smoke"
    rc = run_pipeline.main(["--smoke", "--smoke-patients", "4000", "--data-dir", str(raw), "--out-dir", str(out)])
    assert rc == 0
    for rel in EXPECTED_ARTIFACTS:
        assert (out / rel).exists(), f"missing artifact {rel}"
    dq = pd.read_csv(out / "tables/dq_checks.csv")
    assert not ((~dq["passed"]) & (dq["severity"] == "error")).any()
    flow = pd.read_csv(out / "tables/cohort_flow.csv")
    assert (flow["rows_after"].diff().dropna() <= 0).all(), "cohort can only shrink"
    first = pd.read_parquet(out / "interim/cohort_first.parquet")
    assert first["patient_nbr"].is_unique
