"""Stage 7a — Descriptive epidemiology, hypothesis tests, sample size / power.

Tests use the FIRST-ENCOUNTER cohort (one row per patient) so that the independence
assumption of chi-square / Mann-Whitney holds. Multiple testing: Benjamini-Hochberg FDR.

Outputs
  tables/readmit_by_segment.csv   rate + Wilson 95% CI per category level
  tables/tests_categorical.csv    chi-square, df, p, BH-adjusted p, Cramér's V
  tables/tests_numeric.csv        Mann-Whitney U, p, BH p, medians, rank-biserial r
  tables/power_analysis.csv       power to detect absolute differences; n for 80% power
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_confint, proportion_effectsize

from .config import Config
from .utils import load_parquet

CATEGORICAL = ["age_group", "gender", "race_f", "adm_type_grp", "adm_source_grp", "discharge_grp",
               "diag1_grp", "specialty_grp", "hba1c_tested", "a1c_group", "med_change", "diabetes_med",
               "insulin_status", "glu_tested"]
NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses", "age_mid"]


def segment_rates(df: pd.DataFrame, cols=CATEGORICAL, outcome="readmit30") -> pd.DataFrame:
    rows = []
    for c in cols:
        g = df.groupby(c, observed=True)[outcome].agg(["sum", "count"]).reset_index()
        for _, r in g.iterrows():
            lo, hi = proportion_confint(r["sum"], r["count"], method="wilson")
            rows.append({"variable": c, "level": str(r[c]), "n": int(r["count"]), "events": int(r["sum"]),
                         "rate": r["sum"] / r["count"], "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def chi_square_tests(df: pd.DataFrame, cols=CATEGORICAL, outcome="readmit30") -> pd.DataFrame:
    rows = []
    for c in cols:
        tab = pd.crosstab(df[c], df[outcome])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        chi2, p, dof, exp = stats.chi2_contingency(tab)
        n = tab.values.sum()
        v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
        rows.append({"variable": c, "chi2": chi2, "df": dof, "p": p, "cramers_v": v,
                     "min_expected": float(exp.min()), "n": int(n)})
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    out["significant_fdr05"] = out["p_bh"] < 0.05
    return out


def mann_whitney_tests(df: pd.DataFrame, cols=NUMERIC, outcome="readmit30") -> pd.DataFrame:
    rows = []
    for c in cols:
        a = df.loc[df[outcome] == 1, c].dropna()
        b = df.loc[df[outcome] == 0, c].dropna()
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        r_rb = 1 - 2 * u / (len(a) * len(b))      # rank-biserial correlation
        rows.append({"variable": c, "U": u, "p": p, "median_readmit": a.median(), "median_not": b.median(),
                     "mean_readmit": a.mean(), "mean_not": b.mean(), "rank_biserial_r": -r_rb,
                     "n_readmit": len(a), "n_not": len(b)})
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    out["significant_fdr05"] = out["p_bh"] < 0.05
    return out


def power_analysis(df: pd.DataFrame, exposure="hba1c_tested", outcome="readmit30",
                   diffs=(0.005, 0.01, 0.015, 0.02, 0.03), alpha=0.05) -> pd.DataFrame:
    """Two-sample test of proportions (normal approximation, Cohen's h).
    Baseline p0 = observed outcome rate among the unexposed."""
    n1 = int((df[exposure] == 1).sum())
    n0 = int((df[exposure] == 0).sum())
    p0 = df.loc[df[exposure] == 0, outcome].mean()
    ratio = n1 / n0
    pw = NormalIndPower()
    rows = []
    for d in diffs:
        h = abs(proportion_effectsize(p0, p0 - d))
        power = pw.power(effect_size=h, nobs1=n0, alpha=alpha, ratio=ratio)
        n_needed = pw.solve_power(effect_size=h, power=0.80, alpha=alpha, ratio=ratio)
        rows.append({"abs_difference": d, "baseline_rate": p0, "cohens_h": h, "n_unexposed": n0,
                     "n_exposed": n1, "power_at_observed_n": power,
                     "n_unexposed_for_80pct": int(np.ceil(n_needed)),
                     "n_exposed_for_80pct": int(np.ceil(n_needed * ratio))})
    return pd.DataFrame(rows)


def power_curve(df: pd.DataFrame, exposure="hba1c_tested", outcome="readmit30", alpha=0.05) -> pd.DataFrame:
    p0 = df.loc[df[exposure] == 0, outcome].mean()
    n0 = int((df[exposure] == 0).sum())
    n1 = int((df[exposure] == 1).sum())
    pw = NormalIndPower()
    grid = np.round(np.arange(0.0025, 0.0401, 0.0025), 4)
    rows = []
    for frac in (0.25, 0.5, 1.0):        # power at 25/50/100% of the available sample
        for d in grid:
            h = abs(proportion_effectsize(p0, p0 - d))
            rows.append({"sample_fraction": frac, "abs_difference": d,
                         "power": pw.power(effect_size=h, nobs1=n0 * frac, alpha=alpha, ratio=n1 / n0)})
    return pd.DataFrame(rows)


def repeat_patient_profile(allc: pd.DataFrame) -> pd.DataFrame:
    """Readmission and HbA1c testing by patient type and encounter sequence (all-encounter cohort).
    Supports the estimand discussion: who contributes the extra encounters, and how they differ."""
    d = allc.copy()
    d["patient_type"] = np.where(d["n_encounters_patient"] > 1, "Repeat patient (2+ stays)", "Single-stay patient")
    d["sequence"] = np.where(d["encounter_seq"] >= 3, "3rd+ stay", np.where(d["encounter_seq"] == 2, "2nd stay", "1st stay"))
    rows = []
    for var in ("patient_type", "sequence"):
        g = d.groupby(var).agg(encounters=("encounter_id", "size"), patients=("patient_nbr", "nunique"),
                               readmit30_rate=("readmit30", "mean"), hba1c_tested_rate=("hba1c_tested", "mean"))
        for level, r in g.iterrows():
            rows.append({"grouping": var, "level": level, **r.to_dict(),
                         "share_of_encounters": r["encounters"] / len(d)})
    return pd.DataFrame(rows)


def within_patient_correlation(allc: pd.DataFrame) -> pd.DataFrame:
    """Phi correlation of readmit30 between consecutive encounters of the same patient."""
    d = allc.sort_values(["patient_nbr", "encounter_seq"])
    nxt = d.groupby("patient_nbr")["readmit30"].shift(-1)
    pairs = d.assign(nxt=nxt).dropna(subset=["nxt"])
    phi = float(np.corrcoef(pairs["readmit30"], pairs["nxt"])[0, 1]) if len(pairs) > 2 else float("nan")
    return pd.DataFrame([{"consecutive_pairs": len(pairs), "patients_with_pairs": int(pairs["patient_nbr"].nunique()),
                          "phi_readmit30_consecutive": phi}])


def run(cfg: Config) -> dict:
    first = load_parquet(cfg.interim / "cohort_first.parquet")
    allc = load_parquet(cfg.interim / "cohort_all.parquet")
    seg = segment_rates(first)
    seg.insert(0, "cohort", "first_encounter")
    seg_all = segment_rates(allc)
    seg_all.insert(0, "cohort", "all_encounters")
    pd.concat([seg, seg_all]).to_csv(cfg.tables / "readmit_by_segment.csv", index=False)
    chi_square_tests(first).to_csv(cfg.tables / "tests_categorical.csv", index=False)
    mann_whitney_tests(first).to_csv(cfg.tables / "tests_numeric.csv", index=False)
    power_analysis(first).to_csv(cfg.tables / "power_analysis.csv", index=False)
    power_curve(first).to_csv(cfg.tables / "power_curve.csv", index=False)
    # headline KPIs (both cohorts) for the dashboard tiles
    kpi = pd.DataFrame([
        {"cohort": name, "encounters": len(d), "patients": int(d["patient_nbr"].nunique()),
         "readmit30_rate": d["readmit30"].mean(), "hba1c_tested_rate": d["hba1c_tested"].mean(),
         "repeat_patient_share": float((d.groupby("patient_nbr").size() > 1).mean())}
        for name, d in (("all_encounters", allc), ("first_encounter", first))
    ])
    kpi.to_csv(cfg.tables / "kpi.csv", index=False)
    repeat_patient_profile(allc).to_csv(cfg.tables / "repeat_patients.csv", index=False)
    within_patient_correlation(allc).to_csv(cfg.tables / "within_patient_corr.csv", index=False)
    return {"segments": len(seg)}
