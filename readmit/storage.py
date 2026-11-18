"""Stage 5 — Storage: load the star schema into DuckDB (fallback: SQLite) and
create analytical views used by the dashboard.

Both engines are file-based, so the warehouse runs identically on a laptop and on Kaggle.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import Config
from .utils import LOG, load_parquet

TABLES = ["fact_encounter", "dim_patient", "dim_admission_type", "dim_discharge_disposition",
          "dim_admission_source", "dim_diagnosis", "dim_age", "bridge_encounter_medication"]

VIEWS = {
    # readmission KPI by primary-diagnosis group and HbA1c testing
    "v_readmit_by_diag_hba1c": """
        SELECT d.strack_group AS diag_group, f.hba1c_tested,
               COUNT(*) AS encounters, SUM(f.readmit30) AS readmissions,
               AVG(f.readmit30) AS readmit_rate
        FROM fact_encounter f
        LEFT JOIN dim_diagnosis d ON f.diag_1 = d.icd9_code
        GROUP BY 1, 2""",
    # readmission by discharge group (lookup join -> integration in action)
    "v_readmit_by_discharge": """
        SELECT dd."group" AS discharge_group, COUNT(*) AS encounters,
               AVG(f.readmit30) AS readmit_rate
        FROM fact_encounter f
        JOIN dim_discharge_disposition dd ON f.discharge_disposition_id = dd.id
        GROUP BY 1""",
    # medication activity
    "v_med_activity": """
        SELECT b.drug, b.status, COUNT(*) AS encounters
        FROM bridge_encounter_medication b GROUP BY 1, 2""",
}


def _connect(cfg: Config):
    try:
        import duckdb  # noqa: F401
        path = cfg.warehouse / "diabetes.duckdb"
        if path.exists():
            path.unlink()
        return "duckdb", duckdb.connect(str(path)), path
    except ImportError:
        path = cfg.warehouse / "diabetes.sqlite"
        if path.exists():
            path.unlink()
        LOG.warning("duckdb not installed — falling back to SQLite (%s)", path.name)
        return "sqlite", sqlite3.connect(path), path


def _to_storable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if str(out[c].dtype) in ("string", "category"):
            out[c] = out[c].astype(object).where(out[c].notna(), None)
    return out


def run(cfg: Config) -> dict:
    engine, con, path = _connect(cfg)
    counts = {}
    for t in TABLES:
        df = _to_storable(load_parquet(cfg.warehouse / "parquet" / f"{t}.parquet"))
        if engine == "duckdb":
            con.register("tmp_df", df)
            con.execute(f'CREATE TABLE "{t}" AS SELECT * FROM tmp_df')
            con.unregister("tmp_df")
        else:
            df.to_sql(t, con, index=False)
        counts[t] = len(df)
    for name, sql in VIEWS.items():
        con.execute(f'CREATE VIEW "{name}" AS {sql}')
    # round-trip check: what was stored equals what was built
    stored = {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in TABLES}
    con.close()
    mismatch = {t: (counts[t], stored[t]) for t in TABLES if counts[t] != stored[t]}
    if mismatch:
        raise RuntimeError(f"Row-count mismatch after load: {mismatch}")
    LOG.info("Warehouse (%s) written: %s", engine, path)
    return {"engine": engine, "path": str(path), "row_counts": stored}


def query(cfg: Config, sql: str) -> pd.DataFrame:
    """Convenience for notebooks: run SQL against whichever warehouse exists."""
    duck = cfg.warehouse / "diabetes.duckdb"
    if duck.exists():
        import duckdb
        with duckdb.connect(str(duck), read_only=True) as con:
            return con.execute(sql).df()
    with sqlite3.connect(cfg.warehouse / "diabetes.sqlite") as con:
        return pd.read_sql_query(sql, con)
