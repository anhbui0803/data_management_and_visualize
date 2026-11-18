"""Stage 2 — Profiling of the raw extract (before any cleaning).

Produces tables/profile_columns.csv: one row per column with type, cardinality,
true-missing share, coded-missing share and top values.
"""
from __future__ import annotations

import pandas as pd

from .config import Config
from .utils import load_parquet

# Values that mean "unknown" without being NaN, per column family.
CODED_MISSING = {
    "A1Cresult": {"None"},          # not measured (a value, but reported here for transparency)
    "max_glu_serum": {"None"},
    "gender": {"Unknown/Invalid"},
}
ID_COLS_CODED = {  # ids whose lookup description is a coded-missing label
    "admission_type_id": {5, 6, 8},
    "discharge_disposition_id": {18, 25, 26},
    "admission_source_id": {9, 15, 17, 20, 21},
}


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        coded = 0
        if c in CODED_MISSING:
            coded = int(s.isin(CODED_MISSING[c]).sum())
        elif c in ID_COLS_CODED:
            coded = int(s.isin(ID_COLS_CODED[c]).sum())
        vc = s.value_counts(dropna=True).head(3)
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "n_unique": int(s.nunique(dropna=True)),
            "missing_n": int(s.isna().sum()),
            "missing_pct": round(100 * s.isna().mean(), 2),
            "coded_missing_n": coded,
            "coded_missing_pct": round(100 * coded / n, 2) if n else 0.0,
            "top_values": "; ".join(f"{k}={v}" for k, v in vc.items()),
        })
    return pd.DataFrame(rows)


def run(cfg: Config) -> pd.DataFrame:
    df = load_parquet(cfg.interim / "raw_encounters.parquet")
    prof = profile(df)
    prof.to_csv(cfg.tables / "profile_columns.csv", index=False)
    # repeated-patient structure (the "correlated data" property)
    per_pat = df.groupby("patient_nbr").size()
    pd.DataFrame({
        "encounters_per_patient": per_pat.value_counts().sort_index().index,
        "n_patients": per_pat.value_counts().sort_index().values,
    }).to_csv(cfg.tables / "profile_encounters_per_patient.csv", index=False)
    return prof
