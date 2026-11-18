"""Stage 1 — Collection / ingestion.

Reads the two raw files with EXPLICIT missing-value rules and records provenance.

Missing-value semantics (the most important data-management decision here):
  * "?"        -> true missing (NaN)          e.g. race, payer_code, medical_specialty, weight, diag_*
  * "None"     -> a VALUE meaning "test not performed" (A1Cresult, max_glu_serum)
                  pandas' default would silently turn this into NaN, flipping the exposure.
  * "NULL", "Not Mapped", "Not Available", "Unknown/Invalid" in the ID lookup tables
                -> coded-missing categories, flagged in the dimension tables (integrate.py)
"""
from __future__ import annotations

import io
from datetime import datetime, timezone

import pandas as pd

from .config import EXPECTED, DATA_FILE, MAPPING_FILE, Config
from .utils import LOG, sha256_file, write_json, save_parquet, environment_fingerprint

TRUE_MISSING_TOKENS = ["?"]

STRING_COLS = ["payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3", "race", "weight"]


def load_encounters(cfg: Config) -> pd.DataFrame:
    path = cfg.data_path()
    df = pd.read_csv(
        path,
        na_values=TRUE_MISSING_TOKENS,
        keep_default_na=False,        # keeps "None" as a literal string
        dtype={c: "string" for c in STRING_COLS},
        low_memory=False,
    )
    return df


def split_mapping(cfg: Config) -> dict[str, pd.DataFrame]:
    """IDs_mapping.csv holds three lookup tables separated by a ',' line.
    Returns {"admission_type": df, "discharge_disposition": df, "admission_source": df}."""
    text = cfg.mapping_path().read_text(encoding="utf-8-sig")
    blocks, current = [], []
    for line in text.splitlines():
        if line.strip() in {",", ""}:
            if current:
                blocks.append(current)
                current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)
    out: dict[str, pd.DataFrame] = {}
    for b in blocks:
        df = pd.read_csv(io.StringIO("\n".join(b)), keep_default_na=False)
        key_col = df.columns[0]
        name = key_col.replace("_id", "")
        df = df.rename(columns={key_col: "id"})
        df["id"] = df["id"].astype(int)
        df["description"] = df["description"].astype(str).str.strip()
        out[name] = df
    expected = {"admission_type", "discharge_disposition", "admission_source"}
    if set(out) != expected:
        raise ValueError(f"Unexpected mapping blocks: {sorted(out)}")
    return out


def smoke_sample(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Sample whole patients (not rows) so that repeated encounters stay together."""
    pats = df["patient_nbr"].drop_duplicates().sample(
        n=min(cfg.smoke_patients, df["patient_nbr"].nunique()), random_state=cfg.seed
    )
    return df[df["patient_nbr"].isin(pats)].reset_index(drop=True)


def run(cfg: Config) -> dict:
    df = load_encounters(cfg)
    maps = split_mapping(cfg)

    prov = {"ingested_at_utc": datetime.now(timezone.utc).isoformat(), "environment": cfg.environment,
            "fingerprint": environment_fingerprint(), "files": {}}
    for fname, path in ((DATA_FILE, cfg.data_path()), (MAPPING_FILE, cfg.mapping_path())):
        digest = sha256_file(path)
        exp = EXPECTED[fname]
        entry = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest,
                 "sha256_matches_uci": digest == exp["sha256"]}
        if not entry["sha256_matches_uci"]:
            LOG.warning("%s SHA-256 differs from the UCI reference (%s…). Content checks below still apply.",
                        fname, exp["sha256"][:12])
        prov["files"][fname] = entry
    prov["files"][DATA_FILE].update({"rows": len(df), "cols": df.shape[1],
                                     "shape_matches_uci": (len(df), df.shape[1]) == (EXPECTED[DATA_FILE]["rows"],
                                                                                    EXPECTED[DATA_FILE]["cols"])})
    LOG.info("Raw encounters: %s rows × %s cols; patients: %s", f"{len(df):,}", df.shape[1],
             f"{df['patient_nbr'].nunique():,}")

    if cfg.smoke:
        df = smoke_sample(df, cfg)
        prov["smoke_sample"] = {"patients": int(df["patient_nbr"].nunique()), "rows": len(df)}
        LOG.info("SMOKE mode: sampled %s patients / %s encounters", f"{df['patient_nbr'].nunique():,}", f"{len(df):,}")

    save_parquet(df, cfg.interim / "raw_encounters.parquet")
    for name, m in maps.items():
        save_parquet(m, cfg.interim / f"map_{name}.parquet")
    write_json(prov, cfg.logs / "provenance.json")
    return prov
