"""Stage 7d — Risk stratification model (the "training" step).

Predicts readmit30 at discharge from information available in the encounter record.
Train/test split is GROUPED BY PATIENT, so no patient appears in both sets (prevents
leakage through repeated admissions).

Models
  logit  one-hot + standardised logistic regression (interpretable baseline)
  hgb    HistGradientBoostingClassifier with native categorical support

Outputs
  tables/risk_metrics.csv        AUC, average precision, Brier score on the hold-out set
  tables/risk_calibration.csv    observed vs predicted rate per risk decile
  tables/risk_caseload.csv       share of readmissions captured when following the top-k% riskiest
  models/risk_<name>.joblib      fitted pipelines
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import Config
from .utils import LOG, load_parquet

CAT_FEATURES = ["age_group", "gender", "race_f", "adm_type_grp", "adm_source_grp", "discharge_grp",
                "diag1_grp", "diag2_grp", "diag3_grp", "specialty_grp", "payer_grp", "a1c_group",
                "insulin_status", "max_glu_serum"]
NUM_FEATURES = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
                "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
                "n_meds_changed", "n_meds_active", "med_change", "diabetes_med", "age_mid"]
CASELOAD_K = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def build_models(seed: int) -> dict[str, Pipeline]:
    logit = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES)])),
        ("clf", LogisticRegression(max_iter=2000, C=0.5)),
    ])
    hgb = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-1), CAT_FEATURES),
            ("num", "passthrough", NUM_FEATURES)])),
        ("clf", HistGradientBoostingClassifier(
            categorical_features=list(range(len(CAT_FEATURES))), learning_rate=0.05, max_iter=400,
            max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            random_state=seed)),
    ])
    return {"logit": logit, "hgb": hgb}


def calibration_table(y, p, bins=10) -> pd.DataFrame:
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "decile": q + 1})
    return df.groupby("decile").agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean")).reset_index()


def caseload_table(y, p, ks=CASELOAD_K) -> pd.DataFrame:
    order = np.argsort(-p)
    y_sorted = np.asarray(y)[order]
    total = y_sorted.sum()
    rows = []
    N = len(y_sorted)
    for k in ks:
        n = int(np.ceil(k * N))
        caught = y_sorted[:n].sum()
        rows.append({"top_k_share": k, "patients_followed": n, "readmissions_caught": int(caught),
                     "capture_rate": caught / total if total else np.nan, "ppv": caught / n if n else np.nan,
                     "risk_threshold": float(np.sort(p)[::-1][n - 1]) if n else np.nan,
                     # workload view for a quality manager: per 1,000 discharges
                     "followed_per_1000": 1000 * n / N, "caught_per_1000": 1000 * caught / N,
                     "readmissions_per_1000": 1000 * total / N})
    return pd.DataFrame(rows)


def cluster_bootstrap_auc(y, p, groups, n_boot: int = 300, seed: int = 42) -> tuple[float, float]:
    """95% percentile CI for AUC, resampling patients (not rows) from the hold-out set."""
    rng = np.random.default_rng(seed)
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    idx_by_group = np.split(np.argsort(inv), np.cumsum(np.bincount(inv))[:-1])
    aucs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_group[i] for i in pick])
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def run(cfg: Config) -> pd.DataFrame:
    df = load_parquet(cfg.interim / "cohort_all.parquet")
    df["max_glu_serum"] = df["max_glu_serum"].astype(str)
    X = df[CAT_FEATURES + NUM_FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype(str)
    y = df["readmit30"].values
    split = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    tr, te = next(split.split(X, y, groups=df["patient_nbr"]))
    assert set(df["patient_nbr"].iloc[tr]).isdisjoint(df["patient_nbr"].iloc[te]), "patient leakage"
    LOG.info("  risk split: train %s / test %s encounters (grouped by patient)", f"{len(tr):,}", f"{len(te):,}")

    metrics, calib, case = [], [], []
    for name, model in build_models(cfg.seed).items():
        model.fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[te])[:, 1]
        lo, hi = cluster_bootstrap_auc(y[te], p, df["patient_nbr"].values[te], seed=cfg.seed)
        m = {"model": name, "auc": roc_auc_score(y[te], p), "auc_low": lo, "auc_high": hi,
             "avg_precision": average_precision_score(y[te], p),
             "brier": brier_score_loss(y[te], p), "base_rate_test": float(y[te].mean()),
             "n_train": len(tr), "n_test": len(te)}
        metrics.append(m)
        LOG.info("  %-6s AUC=%.3f  AP=%.3f  Brier=%.4f", name, m["auc"], m["avg_precision"], m["brier"])
        calib.append(calibration_table(y[te], p).assign(model=name))
        case.append(caseload_table(y[te], p).assign(model=name))
        joblib.dump(model, cfg.models_dir / f"risk_{name}.joblib")
    out = pd.DataFrame(metrics)
    out.to_csv(cfg.tables / "risk_metrics.csv", index=False)
    pd.concat(calib).to_csv(cfg.tables / "risk_calibration.csv", index=False)
    pd.concat(case).to_csv(cfg.tables / "risk_caseload.csv", index=False)
    return out
