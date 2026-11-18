# readmit — diabetic 30-day readmission pipeline

Data-management, statistical-analysis and visualization pipeline for the course project
*Data Management & Visualization — Topic 1 (Statistics application in medicine)*,
using the UCI **Diabetes 130-US Hospitals (1999–2008)** dataset (Strack et al., 2014).

Runs unchanged **locally** (Windows / macOS / Linux) and on **Kaggle**.

## Folder layout

```
project1_clinical/
├── data/raw/                 diabetic_data.csv, IDs_mapping.csv   (input, never modified)
├── docs/                     Stage 1 research documents
├── code/                     ← this folder
│   ├── run_pipeline.py       CLI entry point (all stages)
│   ├── readmit/              the package (one module per stage)
│   ├── tests/                pytest unit tests + smoke test
│   ├── notebooks/kaggle_runner.ipynb
│   └── requirements.txt
└── outputs/                  created by the pipeline (outputs_smoke/ for --smoke)
```

## Quick start — local

```bash
cd project1_clinical/code
python -m venv .venv && .venv\Scripts\activate        # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python run_pipeline.py --smoke     # ~1 min: validates every stage on 8,000 sampled patients
python run_pipeline.py             # full run → ../outputs/
pytest -q tests                    # unit tests + smoke test
python export_figures.py --out-dir ../outputs   # static PNGs for the report (needs: pip install kaleido)
```

Open `../outputs/dashboard.html` in a browser.

## Quick start — Kaggle

1. Create a Kaggle dataset from this `code/` folder (e.g. `readmit-code`).
2. New notebook → *Add Input*: your code dataset **and** a copy of the data
   (e.g. `brandao/diabetes`, or upload the two CSVs as your own dataset).
3. Upload `notebooks/kaggle_runner.ipynb` (*File → Import notebook*) and *Run All*.
   Outputs land in `/kaggle/working/readmit_outputs` plus a zip for download.

Paths are detected automatically (`readmit/config.py`). Override with
`--data-dir/--out-dir` or env vars `READMIT_DATA_DIR` / `READMIT_OUT_DIR`.

## Pipeline stages

| # | Stage | Module | Main outputs |
|---|---|---|---|
| 1 | Ingest (collection) | `ingest.py` | `interim/raw_encounters.parquet`, `logs/provenance.json` (SHA-256, shape, versions) |
| 2 | Profile | `profiling.py` | `tables/profile_columns.csv` |
| 3 | Clean + derive | `clean.py` | `tables/cohort_flow.csv`, `interim/cohort_all.parquet`, `interim/cohort_first.parquet` |
| 4 | Integrate (star schema) | `integrate.py` | `warehouse/parquet/*.parquet` (fact + 6 dims + bridge) |
| 5 | Store | `storage.py` | `warehouse/diabetes.duckdb` (SQLite fallback) + 3 analytical views |
| 6 | Quality assurance | `quality.py` | `tables/dq_checks.csv`, `tables/dq_scorecard.csv` — stops on any error-level failure |
| 7a | Tests & power | `stats_tests.py` | segment rates, χ²/Mann-Whitney + BH-FDR, power tables |
| 7b | Regression | `models.py` | logistic (first encounter), interaction by diagnosis, naive vs **GEE** with patient-clustered SEs (all encounters; exchangeable optional), optional random-intercept, **IPW** + balance, E-values |
| 7c | Missing data | `missing.py` | complete-case vs indicator vs **MICE** (Rubin's rules) |
| 7d | Risk model ("training") | `risk.py` | logistic + gradient boosting, patient-grouped split, calibration, caseload curve, `models/*.joblib` |
| 8 | Visualization | `viz.py` | `dashboard.html` + `figures/V*.html`, each with user · question · decision |

Run a subset with `--stages models,missing,viz` (later stages reload saved artifacts).

## Key data-management decisions

* **`"None"` ≠ missing.** In `A1Cresult` / `max_glu_serum`, `"None"` means *test not performed*;
  the CSV is read with `keep_default_na=False` and only `"?"` is treated as missing.
  QA check `CC-a1c-none` asserts 84,748 such values in the full extract.
* **Ineligible discharges** (death / hospice: ids 11, 13, 14, 19, 20, 21) are removed — they cannot be readmitted.
* **Two cohorts:** all encounters (correlated, modelled with GEE) and first encounter per patient
  (independent, used for tests / IPW / MICE). *Assumption:* `encounter_id` order reflects time (no dates in the extract).
* **Sparse categories** (< 50 rows or < 5 events) are pooled before modelling to avoid separation (`models.collapse_rare`).
* **`weight`** (≈97% missing) is reported in the DQ scorecard as not fit for use and never imputed.

## Reproducibility

Every run writes `logs/run_manifest.json` (arguments, config, package versions, stage status, timings)
and `logs/provenance.json` (input hashes). Fix `--seed` for identical sampling / splits / imputations.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--smoke` | off | patient-sample validation run (`--smoke-patients`, default 8,000); MICE reduced to m=2 |
| `--mice-m` / `--mice-iter` | 5 / 5 | imputations / chained sweeps |
| `--mixed` | off | also fit the random-intercept logistic model (variational Bayes; slow) |
| `--gee-exchangeable` | off | also fit exchangeable-correlation GEE (slow on the full extract; default M3 is independence GEE with patient-clustered SEs) |
| `--seed` | 42 | random seed |
