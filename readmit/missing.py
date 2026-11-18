"""Stage 7c — Missing-data analysis (first-encounter cohort).

Compares the exposure OR from model M1 under three strategies:
  CC        complete-case: drop patients with missing race or medical specialty
  INDICATOR missing coded as its own category (the default used in models.py)
  MICE      multiple imputation by chained equations for race, medical specialty and payer
            (multinomial logistic imputation models with random draws; the outcome and the
            exposure are included as predictors, as recommended by Sterne et al., 2009);
            estimates pooled with Rubin's rules.

`weight` (≈97% missing) is never imputed — it is excluded as "not fit for use".

Outputs
  tables/missing_patterns.csv        counts of joint missingness patterns
  tables/missing_vs_outcome.csv      readmission rate by missingness indicator (MAR plausibility)
  tables/missing_strategy_compare.csv exposure OR under CC / INDICATOR / MICE
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

from .config import Config
from .models import collapse_rare, covariate_formula, fit_glm
from .utils import LOG, load_parquet

TARGETS = {"race_f": "race", "specialty_grp": "medical_specialty", "payer_grp": "payer_code"}
AUX_CAT = ["age_group", "gender", "adm_source_grp", "discharge_grp", "diag1_grp", "adm_type_grp"]
AUX_NUM = ["readmit30", "hba1c_tested", "time_in_hospital", "number_inpatient_c", "number_emergency_c",
           "number_outpatient_c", "number_diagnoses", "num_medications", "num_lab_procedures", "age_mid"]
PATTERN_COLS = ["weight", "medical_specialty", "payer_code", "race", "diag_2", "diag_3"]


def missing_patterns(df: pd.DataFrame) -> pd.DataFrame:
    m = df[PATTERN_COLS].isna().astype(int)
    m["A1C_not_measured"] = (df["A1Cresult"] == "None").astype(int)
    pat = m.value_counts().reset_index(name="n")
    pat["share"] = pat["n"] / len(df)
    return pat


def missing_vs_outcome(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("race", "payer_code", "medical_specialty"):
        flag = df[col].isna()
        tab = pd.crosstab(flag, df["readmit30"])
        chi2, p, *_ = stats.chi2_contingency(tab) if tab.shape == (2, 2) else (np.nan, np.nan)
        rows.append({"variable": col, "missing_share": flag.mean(),
                     "readmit_rate_missing": df.loc[flag, "readmit30"].mean(),
                     "readmit_rate_observed": df.loc[~flag, "readmit30"].mean(), "chi2_p": p})
    return pd.DataFrame(rows)


def _design(df: pd.DataFrame, cols_cat: list[str]) -> np.ndarray:
    num = df[AUX_NUM].astype(float)
    num = (num - num.mean()) / num.std(ddof=0).replace(0, 1)
    dummies = pd.get_dummies(df[cols_cat].astype(str), drop_first=True, dtype=float)
    return np.hstack([num.values, dummies.values])


def mice_impute(df: pd.DataFrame, m: int, iters: int, seed: int, max_fit_rows: int = 30_000) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    miss = {t: df[src].isna().values for t, src in TARGETS.items()}
    base = df.copy()
    # target columns without the "Missing" level; NaN where missing
    for t in TARGETS:
        base[t] = base[t].astype(str).where(~miss[t], np.nan)
    completed = []
    for k in range(m):
        d = base.copy()
        for t in TARGETS:                     # initialise from observed marginal distribution
            obs = d.loc[~miss[t], t]
            d.loc[miss[t], t] = rng.choice(obs.values, size=int(miss[t].sum()), replace=True)
        for _ in range(iters):
            for t in TARGETS:
                if not miss[t].any():
                    continue
                others = [o for o in TARGETS if o != t]
                X = _design(d, AUX_CAT + others)
                y = d[t].values
                obs_idx = np.where(~miss[t])[0]
                if len(obs_idx) > max_fit_rows:
                    obs_idx = rng.choice(obs_idx, max_fit_rows, replace=False)
                clf = LogisticRegression(max_iter=300, C=1.0)
                clf.fit(X[obs_idx], y[obs_idx])
                proba = clf.predict_proba(X[miss[t]])
                cum = proba.cumsum(axis=1)
                u = rng.random((proba.shape[0], 1))
                draws = clf.classes_[(u > cum).sum(axis=1).clip(max=len(clf.classes_) - 1)]
                d.loc[miss[t], t] = draws
        completed.append(d)
        LOG.info("  MICE imputation %d/%d done", k + 1, m)
    return completed


def rubin_pool(estimates: list[float], variances: list[float]) -> dict:
    m = len(estimates)
    q = np.mean(estimates)
    u = np.mean(variances)
    b = np.var(estimates, ddof=1) if m > 1 else 0.0
    if b < 1e-12 * u:          # numerically identical imputations
        b = 0.0
    t = u + (1 + 1 / m) * b
    df = (m - 1) * (1 + u / ((1 + 1 / m) * b)) ** 2 if b > 0 else np.inf
    half = stats.t.ppf(0.975, df) * np.sqrt(t) if np.isfinite(df) else 1.96 * np.sqrt(t)
    fmi = ((1 + 1 / m) * b) / t if t > 0 else 0.0
    return {"coef": q, "se": np.sqrt(t), "or": np.exp(q), "or_low": np.exp(q - half), "or_high": np.exp(q + half),
            "df": df, "fmi": fmi}


def _exposure_row(res, label, n) -> dict:
    ci = res.conf_int().loc["hba1c_tested"]
    return {"strategy": label, "n": n, "coef": res.params["hba1c_tested"], "se": res.bse["hba1c_tested"],
            "or": np.exp(res.params["hba1c_tested"]), "or_low": np.exp(ci.iloc[0]), "or_high": np.exp(ci.iloc[1])}


def run(cfg: Config) -> pd.DataFrame:
    first = load_parquet(cfg.interim / "cohort_first.parquet")
    missing_patterns(first).to_csv(cfg.tables / "missing_patterns.csv", index=False)
    missing_vs_outcome(first).to_csv(cfg.tables / "missing_vs_outcome.csv", index=False)
    rows = []

    ind_d = collapse_rare(first)
    ind = fit_glm(f"readmit30 ~ hba1c_tested + {covariate_formula(ind_d)}", ind_d)
    rows.append(_exposure_row(ind, "INDICATOR (missing = category)", len(first)))

    cc = collapse_rare(first[(first["miss_race"] == 0) & (first["miss_medical_specialty"] == 0)])
    cc_res = fit_glm(f"readmit30 ~ hba1c_tested + {covariate_formula(cc)}", cc)
    rows.append(_exposure_row(cc_res, "CC (complete cases)", len(cc)))

    imps = mice_impute(first, cfg.mice_m, cfg.mice_iter, cfg.seed)
    ests, vars_ = [], []
    for d in imps:
        d = collapse_rare(d)
        r = fit_glm(f"readmit30 ~ hba1c_tested + {covariate_formula(d)}", d)
        ests.append(r.params["hba1c_tested"])
        vars_.append(r.bse["hba1c_tested"] ** 2)
    pooled = rubin_pool(ests, vars_)
    rows.append({"strategy": f"MICE (m={cfg.mice_m}, Rubin's rules)", "n": len(first), **pooled})

    out = pd.DataFrame(rows)
    out.to_csv(cfg.tables / "missing_strategy_compare.csv", index=False)
    return out
