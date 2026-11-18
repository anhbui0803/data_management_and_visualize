"""Stage 4 — Integration into a star schema.

  fact_encounter              grain: one row per eligible encounter
  dim_patient                 patient_nbr, race, gender (first-seen values + consistency flags)
  dim_admission_type          from IDs_mapping.csv block 1 (+ group, coded-missing flag)
  dim_discharge_disposition   from IDs_mapping.csv block 2
  dim_admission_source        from IDs_mapping.csv block 3
  dim_diagnosis               every ICD-9 code seen in diag_1..3 -> Strack group + ICD-9 chapter
  dim_age                     age band -> midpoint, 3-level group
  bridge_encounter_medication long format: encounter × drug × status (only drugs not "No")
"""
from __future__ import annotations

import pandas as pd

from .clean import (ADMISSION_SOURCE_GROUP, ADMISSION_TYPE_GROUP, DISCHARGE_GROUP, MED_COLS, AGE_MID,
                    icd9_chapter, icd9_group, ICD9_PATTERN)
from .config import Config
from .utils import load_parquet, save_parquet

CODED_MISSING_LABELS = {"NULL", "Not Mapped", "Not Available", "Unknown/Invalid"}

FACT_COLS = [
    "encounter_id", "patient_nbr", "encounter_seq", "n_encounters_patient", "age",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "diag_1", "diag_2", "diag_3", "medical_specialty", "payer_code",
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "A1Cresult", "max_glu_serum", "change", "diabetesMed", "readmitted",
    # derived
    "readmit30", "hba1c_tested", "a1c_group", "glu_tested", "n_meds_changed", "n_meds_active",
    "specialty_grp", "payer_grp", "prior_visits",
]


def build_dim_lookup(m: pd.DataFrame, group_map: dict) -> pd.DataFrame:
    d = m.copy()
    d["group"] = d["id"].map(group_map).fillna("Other")
    d["is_coded_missing"] = d["description"].isin(CODED_MISSING_LABELS)
    return d


def build_dim_patient(allc: pd.DataFrame) -> pd.DataFrame:
    g = allc.groupby("patient_nbr")
    dim = g.agg(race=("race", "first"), gender=("gender", "first"), n_encounters=("encounter_id", "size")).reset_index()
    dim["race_values"] = g["race"].nunique(dropna=True).values
    dim["gender_values"] = g["gender"].nunique(dropna=True).values
    dim["race_inconsistent"] = dim["race_values"] > 1
    dim["gender_inconsistent"] = dim["gender_values"] > 1
    return dim.drop(columns=["race_values", "gender_values"])


def build_dim_diagnosis(allc: pd.DataFrame) -> pd.DataFrame:
    codes = pd.unique(pd.concat([allc["diag_1"], allc["diag_2"], allc["diag_3"]]).dropna())
    dim = pd.DataFrame({"icd9_code": pd.Series(codes, dtype="string")})
    dim["strack_group"] = dim["icd9_code"].map(icd9_group)
    dim["icd9_chapter"] = dim["icd9_code"].map(icd9_chapter)
    dim["is_valid_format"] = dim["icd9_code"].map(lambda s: bool(ICD9_PATTERN.match(str(s))))
    return dim.sort_values("icd9_code").reset_index(drop=True)


def build_dim_age() -> pd.DataFrame:
    rows = [{"age": k, "age_mid": v, "age_group": "<30" if v < 30 else ("30-60" if v < 60 else ">60")}
            for k, v in AGE_MID.items()]
    return pd.DataFrame(rows)


def build_bridge_meds(allc: pd.DataFrame) -> pd.DataFrame:
    long = allc[["encounter_id"] + MED_COLS].melt(id_vars="encounter_id", var_name="drug", value_name="status")
    return long[long["status"] != "No"].reset_index(drop=True)


# ---------------------------------------------------------------- schema documentation
KEYS = {  # table -> (primary key, {fk column: referenced table.column})
    "fact_encounter": ("encounter_id", {"patient_nbr": "dim_patient.patient_nbr", "age": "dim_age.age",
                                        "admission_type_id": "dim_admission_type.id",
                                        "discharge_disposition_id": "dim_discharge_disposition.id",
                                        "admission_source_id": "dim_admission_source.id",
                                        "diag_1": "dim_diagnosis.icd9_code", "diag_2": "dim_diagnosis.icd9_code",
                                        "diag_3": "dim_diagnosis.icd9_code"}),
    "dim_patient": ("patient_nbr", {}),
    "dim_admission_type": ("id", {}),
    "dim_discharge_disposition": ("id", {}),
    "dim_admission_source": ("id", {}),
    "dim_diagnosis": ("icd9_code", {}),
    "dim_age": ("age", {}),
    "bridge_encounter_medication": ("encounter_id, drug", {"encounter_id": "fact_encounter.encounter_id"}),
}

DESCRIPTIONS = {
    "encounter_id": "Unique hospital stay (source key)", "patient_nbr": "Patient identifier (source key)",
    "encounter_seq": "Order of this stay among the patient's stays (by encounter_id)",
    "n_encounters_patient": "Number of stays the patient has in the extract",
    "age": "10-year age band as released", "admission_type_id": "Admission type code (IDs_mapping)",
    "discharge_disposition_id": "Discharge disposition code (IDs_mapping)",
    "admission_source_id": "Admission source code (IDs_mapping)",
    "diag_1": "Primary diagnosis, ICD-9-CM", "diag_2": "Secondary diagnosis, ICD-9-CM",
    "diag_3": "Additional secondary diagnosis, ICD-9-CM", "medical_specialty": "Admitting physician specialty",
    "payer_code": "Payer", "time_in_hospital": "Length of stay, days (1-14)",
    "num_lab_procedures": "Lab tests during stay", "num_procedures": "Non-lab procedures during stay",
    "num_medications": "Distinct generic drugs during stay", "number_outpatient": "Outpatient visits, prior year",
    "number_emergency": "Emergency visits, prior year", "number_inpatient": "Inpatient stays, prior year",
    "number_diagnoses": "Diagnoses entered", "A1Cresult": "HbA1c result during stay (None = not measured)",
    "max_glu_serum": "Max glucose serum test result (None = not measured)",
    "change": "Any diabetes medication changed (Ch/No)", "diabetesMed": "Any diabetes medication prescribed",
    "readmitted": "Readmission as released (<30, >30, NO)", "readmit30": "Outcome: readmitted within 30 days (1/0)",
    "hba1c_tested": "Exposure: HbA1c measured during the stay (1/0)",
    "a1c_group": "Strack et al. HbA1c group (not measured / normal / high-changed / high-unchanged)",
    "glu_tested": "Glucose serum test done (1/0)", "n_meds_changed": "Diabetes drugs with dose up/down",
    "n_meds_active": "Diabetes drugs prescribed", "specialty_grp": "Specialty, top-k + Other + Missing",
    "payer_grp": "Payer, grouped", "prior_visits": "Sum of outpatient, emergency and inpatient visits, prior year",
    "race": "Race (first recorded)", "gender": "Gender (first recorded)", "n_encounters": "Stays in the extract",
    "race_inconsistent": "Patient has more than one race value", "gender_inconsistent": "Patient has more than one gender value",
    "id": "Source code", "description": "Source label", "group": "Analysis grouping",
    "is_coded_missing": "Label means missing (NULL / Not Mapped / ...)", "icd9_code": "ICD-9-CM code",
    "strack_group": "Diagnosis group (Strack et al. 2014)", "icd9_chapter": "ICD-9-CM chapter",
    "is_valid_format": "Code matches ICD-9-CM pattern", "age_mid": "Band midpoint, years",
    "age_group": "Age group used in analysis", "drug": "Diabetes drug name", "status": "Up / Down / Steady",
}


def data_dictionary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, t in tables.items():
        pk, fks = KEYS.get(name, ("", {}))
        pk_cols = [c.strip() for c in pk.split(",") if c.strip()]
        for c in t.columns:
            role = "PK" if c in pk_cols else ""
            if c in fks:
                role = (role + " FK→" + fks[c]).strip()
            rows.append({"table": name, "column": c, "dtype": str(t[c].dtype), "key": role,
                         "null_share": float(t[c].isna().mean()) if len(t) else float("nan"),
                         "n_distinct": int(t[c].nunique(dropna=True)), "description": DESCRIPTIONS.get(c, "")})
    return pd.DataFrame(rows)


def er_dot(tables: dict[str, pd.DataFrame]) -> str:
    """Graphviz DOT for the star schema (fact table in the centre, key columns only + row counts)."""
    lines = ['digraph star {', '  graph [rankdir=LR, fontname="Helvetica", nodesep=0.35, ranksep=1.1];',
             '  node [shape=plaintext, fontname="Helvetica", fontsize=10];', '  edge [color="#8a8984", arrowhead=crow, arrowtail=tee, dir=both];']
    for name, t in tables.items():
        pk, fks = KEYS.get(name, ("", {}))
        head = "#2a78d6" if name.startswith("fact") else ("#1baf7a" if name.startswith("bridge") else "#52514d")
        cols = [c.strip() for c in pk.split(",") if c.strip()] + [c for c in fks if c not in pk]
        if name.startswith("fact"):
            cols += ["readmit30", "hba1c_tested", "… (%d more)" % (len(t.columns) - len(cols) - 2)]
        rows = "".join(f'<tr><td align="left" port="{c.split()[0] if c[0] != "…" else "more"}">{c}{" (PK)" if c in pk else ""}</td></tr>' for c in cols)
        lines.append(f'  {name} [label=<<table border="0" cellborder="1" cellspacing="0" cellpadding="4">'
                     f'<tr><td bgcolor="{head}"><font color="white"><b>{name}</b></font></td></tr>'
                     f'<tr><td><i>{len(t):,} rows</i></td></tr>{rows}</table>>];')
    for name, (pk, fks) in KEYS.items():
        for col, ref in fks.items():
            rt = ref.split(".")[0]
            lines.append(f'  {rt}:{ref.split(".")[1]}:e -> {name}:{col}:w;')
    lines.append("}")
    return "\n".join(lines)


def write_schema_docs(cfg: Config, tables: dict[str, pd.DataFrame]) -> None:
    import shutil
    import subprocess
    data_dictionary(tables).to_csv(cfg.tables / "data_dictionary.csv", index=False)
    dot_path = cfg.warehouse / "er_diagram.dot"
    dot_path.write_text(er_dot(tables), encoding="utf-8")
    if shutil.which("dot"):  # optional: render if Graphviz is installed
        for fmt in ("png", "svg"):
            subprocess.run(["dot", f"-T{fmt}", "-Gdpi=160", str(dot_path), "-o", str(dot_path.with_suffix("." + fmt))],
                           check=False)


def run(cfg: Config) -> dict[str, pd.DataFrame]:
    allc = load_parquet(cfg.interim / "cohort_all.parquet")
    tables = {
        "fact_encounter": allc[FACT_COLS].copy(),
        "dim_patient": build_dim_patient(allc),
        "dim_admission_type": build_dim_lookup(load_parquet(cfg.interim / "map_admission_type.parquet"),
                                               ADMISSION_TYPE_GROUP),
        "dim_discharge_disposition": build_dim_lookup(
            load_parquet(cfg.interim / "map_discharge_disposition.parquet"), DISCHARGE_GROUP),
        "dim_admission_source": build_dim_lookup(load_parquet(cfg.interim / "map_admission_source.parquet"),
                                                 ADMISSION_SOURCE_GROUP),
        "dim_diagnosis": build_dim_diagnosis(allc),
        "dim_age": build_dim_age(),
        "bridge_encounter_medication": build_bridge_meds(allc),
    }
    for name, t in tables.items():
        save_parquet(t, cfg.warehouse / "parquet" / f"{name}.parquet")
    write_schema_docs(cfg, tables)
    return tables
