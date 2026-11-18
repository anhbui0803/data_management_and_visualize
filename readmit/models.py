"""Stage 7b — Regression models for the HbA1c-testing / readmission association.

Exposure: hba1c_tested (1 = HbA1c measured during the encounter). Outcome: readmit30.
All models adjust for the same case-mix covariates (COVARIATES below).

  M1  logistic, first-encounter cohort (patient-level; replicates Strack et al.'s design)
  M1i M1 + hba1c_tested × primary-diagnosis interaction -> OR per diagnosis group
  M1g M1 with the 4-level Strack HbA1c × medication-change grouping
  M2  logistic, ALL encounters, naive (ignores repeated patients -> SEs too small)
  M3  GEE logistic, ALL encounters, independence working correlation + patient-clustered
      sandwich SEs (fast; valid under any within-patient correlation)
  M3x GEE exchangeable working correlation  [optional: --gee-exchangeable; slow]
  M4  random-intercept logistic (variational Bayes), ALL encounters  [optional: --mixed]
  M5  inverse-probability-of-treatment weighting (stabilised, trimmed), first-encounter
      cohort, robust (sandwich) SEs + covariate balance table

Association language only — the design cannot establish causation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

from .config import Config
from .utils import LOG, load_parquet, save_parquet

REFS = {"age_group": ">60", "race_f": "Caucasian", "adm_source_grp": "Emergency room", "discharge_grp": "Home",
        "diag1_grp": "Circulatory", "specialty_grp": "Missing", "gender": "Female",
        "a1c_group": "Not tested"}
CATEGORICAL_COVS = ["age_group", "gender", "race_f", "adm_source_grp", "discharge_grp", "diag1_grp", "specialty_grp"]
NUMERIC_COVS = ["time_in_hospital", "number_inpatient_c", "number_emergency_c", "number_outpatient_c",
                "number_diagnoses", "num_medications"]


def collapse_rare(df: pd.DataFrame, cols=CATEGORICAL_COVS, min_n: int = 50, min_events: int = 5,
                  outcome: str = "readmit30") -> pd.DataFrame:
    """Merge sparse category levels into 'Other' to avoid (quasi-)separation.
    A level is sparse if it has < min_n rows or < min_events outcome events.
    Reference levels (REFS) are never merged. Applied identically in every model."""
    d = df.copy()
    for c in cols:
        s = d[c].astype(str)
        g = d.groupby(s)[outcome].agg(["size", "sum"])
        sparse = g[(g["size"] < min_n) | (g["sum"] < min_events)].index.difference([REFS.get(c)])
        s = s.where(~s.isin(sparse), "Other") if len(sparse) else s
        # if the pooled 'Other' level is itself still sparse, fold it into the most frequent level
        g2 = d.assign(_s=s).groupby("_s")[outcome].agg(["size", "sum"])
        if "Other" in g2.index and (g2.loc["Other", "size"] < min_n or g2.loc["Other", "sum"] < min_events):
            s = s.where(s != "Other", g2["size"].drop("Other").idxmax())
        d[c] = s
    return d


def cat(var: str, df: pd.DataFrame) -> str:
    ref = REFS.get(var)
    if ref is not None and ref in set(df[var].astype(str)):
        return f"C({var}, Treatment('{ref}'))"
    return f"C({var})"


def covariate_formula(df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> str:
    terms = [cat(v, df) for v in CATEGORICAL_COVS if v not in exclude] + [v for v in NUMERIC_COVS if v not in exclude]
    return " + ".join(terms)


def tidy(res, model: str, n: int, n_clusters: int | None = None, robust: bool = False) -> pd.DataFrame:
    ci = res.conf_int()
    out = pd.DataFrame({"model": model, "term": res.params.index, "coef": res.params.values,
                        "se": res.bse.values, "p": res.pvalues.values,
                        "or": np.exp(res.params.values), "or_low": np.exp(ci.iloc[:, 0].values),
                        "or_high": np.exp(ci.iloc[:, 1].values)})
    out["n"] = n
    out["n_clusters"] = n_clusters
    out["robust_se"] = robust
    return out


def e_value(or_point: float, or_low: float, or_high: float, outcome_rate: float) -> dict:
    """VanderWeele & Ding (2017). OR≈RR when outcome < 15%; otherwise sqrt(OR) approximation."""
    def to_rr(o):
        return o if outcome_rate < 0.15 else np.sqrt(o)

    def ev(rr):
        rr = 1 / rr if rr < 1 else rr
        return rr + np.sqrt(rr * (rr - 1))

    rr, lo, hi = to_rr(or_point), to_rr(or_low), to_rr(or_high)
    if lo <= 1 <= hi:
        ev_ci = 1.0
    else:
        ev_ci = ev(hi if rr < 1 else lo)
    return {"rr_approx": rr, "e_value_point": ev(rr), "e_value_ci": ev_ci}


def expit(x):
    return 1 / (1 + np.exp(-x))


def marginal_rd_from_logit(formula: str, df: pd.DataFrame) -> dict:
    """Average marginal risk difference of hba1c_tested (g-computation with a delta-method SE)
    from a covariate-adjusted logistic model — the standardized (marginal) effect."""
    res = smf.logit(formula, data=df).fit(disp=0, maxiter=200)
    # exposure enters as a single main-effect column (no interactions in this formula), so setting
    # that column to 1 / 0 in the fitted design matrix gives the counterfactual designs.
    j = list(res.model.exog_names).index("hba1c_tested")
    X1, X0 = res.model.exog.copy(), res.model.exog.copy()
    X1[:, j], X0[:, j] = 1.0, 0.0
    b = res.params.values
    p1, p0 = expit(X1 @ b), expit(X0 @ b)
    grad = (X1 * (p1 * (1 - p1))[:, None] - X0 * (p0 * (1 - p0))[:, None]).mean(axis=0)
    rd = float(p1.mean() - p0.mean())
    se = float(np.sqrt(grad @ res.cov_params().values @ grad))
    return {"risk_tested": float(p1.mean()), "risk_untested": float(p0.mean()), "rd": rd, "rd_low": rd - 1.96 * se,
            "rd_high": rd + 1.96 * se, "rd_se": se}


def marginal_rd_from_ipw(res) -> dict:
    """Risk difference from the weighted exposure-only logistic model (delta method, robust covariance)."""
    b0, b1 = res.params["Intercept"], res.params["hba1c_tested"]
    p1, p0 = expit(b0 + b1), expit(b0)
    grad = np.array([p1 * (1 - p1) - p0 * (1 - p0), p1 * (1 - p1)])
    cov = res.cov_params().loc[["Intercept", "hba1c_tested"], ["Intercept", "hba1c_tested"]].values
    se = float(np.sqrt(grad @ cov @ grad))
    rd = float(p1 - p0)
    return {"risk_tested": float(p1), "risk_untested": float(p0), "rd": rd, "rd_low": rd - 1.96 * se,
            "rd_high": rd + 1.96 * se, "rd_se": se}


# --------------------------------------------------------------------------- models
def fit_glm(formula: str, df: pd.DataFrame, **kw):
    return smf.glm(formula, data=df, family=sm.families.Binomial(), **kw).fit()


def interaction_ors(res, df: pd.DataFrame) -> pd.DataFrame:
    """OR of hba1c_tested within each primary-diagnosis group from the interaction model."""
    names = list(res.params.index)
    rows = []
    for g in sorted(df["diag1_grp"].astype(str).unique()):
        c = np.zeros(len(names))
        c[names.index("hba1c_tested")] = 1
        inter = [n for n in names if n.startswith("hba1c_tested:") and f"[T.{g}]" in n]
        for n in inter:
            c[names.index(n)] = 1
        t = res.t_test(c)
        est = float(np.squeeze(t.effect))
        lo, hi = np.squeeze(t.conf_int())
        rows.append({"diag_group": g, "or": np.exp(est), "or_low": np.exp(lo), "or_high": np.exp(hi),
                     "p": float(np.squeeze(t.pvalue)), "n": int((df["diag1_grp"] == g).sum())})
    return pd.DataFrame(rows)


def ipw(df: pd.DataFrame, cfg: Config):
    covf = covariate_formula(df)
    ps_res = fit_glm(f"hba1c_tested ~ {covf}", df)
    ps = ps_res.predict(df).clip(*np.quantile(ps_res.predict(df), cfg.ipw_trim))
    t = df["hba1c_tested"].values
    pt = t.mean()
    w = np.where(t == 1, pt / ps, (1 - pt) / (1 - ps))
    out_res = smf.glm("readmit30 ~ hba1c_tested", data=df, family=sm.families.Binomial(),
                      var_weights=w).fit(cov_type="HC0")
    # balance diagnostics: standardised mean differences before/after weighting
    X = patsy.dmatrix(covf, df, return_type="dataframe").drop(columns="Intercept")
    rows = []
    for c in X.columns:
        x = X[c].values
        x1, x0 = x[t == 1], x[t == 0]
        w1, w0 = w[t == 1], w[t == 0]
        sd = np.sqrt((x1.var() + x0.var()) / 2) or 1.0
        smd_before = (x1.mean() - x0.mean()) / sd
        smd_after = (np.average(x1, weights=w1) - np.average(x0, weights=w0)) / sd
        rows.append({"covariate": c, "smd_before": smd_before, "smd_after": smd_after})
    bal = pd.DataFrame(rows)
    diag = {"ps_min": float(ps.min()), "ps_max": float(ps.max()), "weight_max": float(w.max()),
            "weight_mean": float(w.mean()), "effective_n": float(w.sum() ** 2 / (w ** 2).sum()),
            "max_abs_smd_after": float(bal["smd_after"].abs().max())}
    return out_res, bal, diag, pd.DataFrame({"encounter_id": df["encounter_id"], "ps": ps, "weight": w})


def run(cfg: Config) -> dict:
    first = collapse_rare(load_parquet(cfg.interim / "cohort_first.parquet"))
    allc = collapse_rare(load_parquet(cfg.interim / "cohort_all.parquet"))
    coefs, summary = [], []

    def record(res, name, d, n_clusters=None, robust=False):
        tb = tidy(res, name, len(d), n_clusters, robust)
        coefs.append(tb)
        row = tb[tb["term"] == "hba1c_tested"].iloc[0].to_dict()
        summary.append(row)
        LOG.info("  %-38s OR(hba1c_tested)=%.3f [%.3f, %.3f]", name, row["or"], row["or_low"], row["or_high"])

    f_first = f"readmit30 ~ hba1c_tested + {covariate_formula(first)}"
    f_all = f"readmit30 ~ hba1c_tested + {covariate_formula(allc)}"

    m1 = fit_glm(f_first, first)
    record(m1, "M1 logistic – first encounter", first)

    m1i = fit_glm(f"readmit30 ~ hba1c_tested * {cat('diag1_grp', first)} + "
                  f"{covariate_formula(first, exclude=('diag1_grp',))}", first)
    coefs.append(tidy(m1i, "M1i interaction – first encounter", len(first)))
    inter = interaction_ors(m1i, first)
    from statsmodels.stats.multitest import multipletests
    inter["p_holm"] = multipletests(inter["p"], method="holm")[1]
    inter["p_bh"] = multipletests(inter["p"], method="fdr_bh")[1]
    inter.to_csv(cfg.tables / "model_hba1c_by_diag.csv", index=False)
    # likelihood-ratio test for the interaction block
    lr = 2 * (m1i.llf - m1.llf)
    df_diff = int(m1i.df_model - m1.df_model)
    from scipy import stats
    pd.DataFrame([{"lr_stat": lr, "df": df_diff, "p": stats.chi2.sf(lr, df_diff)}]).to_csv(
        cfg.tables / "model_interaction_lrt.csv", index=False)

    m1g = fit_glm(f"readmit30 ~ {cat('a1c_group', first)} + {covariate_formula(first)}", first)
    coefs.append(tidy(m1g, "M1g Strack 4-level HbA1c group", len(first)))

    m2 = fit_glm(f_all, allc)
    record(m2, "M2 logistic – all encounters (naive)", allc, allc["patient_nbr"].nunique())

    # M3: GEE with independence working correlation == logistic with patient-clustered sandwich SEs.
    # Fast on the full extract and gives valid inference under any within-patient correlation.
    m3 = smf.glm(f_all, data=allc, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": allc["patient_nbr"].values})
    record(m3, "M3 GEE (independence, cluster-robust) – all", allc, allc["patient_nbr"].nunique(), robust=True)
    gee_corr, gee_converged = np.nan, True
    if cfg.gee_exchangeable:
        # M3x: exchangeable working correlation (statsmodels loops over clusters in Python -> slow on 70k patients)
        import warnings
        from statsmodels.tools.sm_exceptions import IterationLimitWarning
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always", IterationLimitWarning)
            m3x = smf.gee(f_all, groups="patient_nbr", data=allc, family=sm.families.Binomial(),
                          cov_struct=sm.cov_struct.Exchangeable()).fit(maxiter=cfg.gee_maxiter, start_params=m3.params)
        gee_converged = not any(issubclass(w.category, IterationLimitWarning) for w in wlist)
        if not gee_converged:
            LOG.warning("  exchangeable GEE hit the iteration limit — treat M3x as provisional")
        record(m3x, "M3x GEE exchangeable – all encounters", allc, allc["patient_nbr"].nunique(), robust=True)
        gee_corr = float(m3x.model.cov_struct.dep_params)

    if cfg.run_mixed:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        mm = BinomialBayesMixedGLM.from_formula(f_all, {"patient": "0 + C(patient_nbr)"}, allc).fit_vb()
        idx = list(mm.model.exog_names).index("hba1c_tested")
        est, sd = mm.fe_mean[idx], mm.fe_sd[idx]
        summary.append({"model": "M4 random-intercept (VB) – all encounters", "term": "hba1c_tested",
                        "coef": est, "se": sd, "p": np.nan, "or": np.exp(est),
                        "or_low": np.exp(est - 1.96 * sd), "or_high": np.exp(est + 1.96 * sd),
                        "n": len(allc), "n_clusters": allc["patient_nbr"].nunique(), "robust_se": False})

    m5, bal, diag, weights = ipw(first, cfg)
    record(m5, "M5 IPW (stabilised) – first encounter", first, robust=True)
    bal.to_csv(cfg.tables / "ipw_balance.csv", index=False)
    pd.DataFrame([diag]).to_csv(cfg.tables / "ipw_diagnostics.csv", index=False)
    save_parquet(weights, cfg.interim / "ipw_weights.parquet")

    rd_rows = [
        {"model": "M1 logistic – first encounter (g-computation)", "estimand": "marginal risk difference, patient level",
         **marginal_rd_from_logit(f_first, first)},
        {"model": "M3 logistic – all encounters (g-computation, independence)", "estimand": "marginal risk difference, encounter level",
         **marginal_rd_from_logit(f_all, allc)},
        {"model": "M5 IPW (stabilised) – first encounter", "estimand": "marginal risk difference, patient level",
         **marginal_rd_from_ipw(m5)},
    ]
    pd.DataFrame(rd_rows).to_csv(cfg.tables / "model_risk_differences.csv", index=False)

    summ = pd.DataFrame(summary)
    rate = first["readmit30"].mean()
    ev = summ.apply(lambda r: pd.Series(e_value(r["or"], r["or_low"], r["or_high"], rate)), axis=1)
    summ = pd.concat([summ, ev], axis=1)
    summ.to_csv(cfg.tables / "model_exposure_summary.csv", index=False)
    pd.concat(coefs).to_csv(cfg.tables / "model_coefficients.csv", index=False)
    pd.DataFrame([{"gee_exchangeable_corr": gee_corr, "gee_converged": gee_converged}]).to_csv(cfg.tables / "model_gee_corr.csv", index=False)
    return {"models": summ["model"].tolist()}
