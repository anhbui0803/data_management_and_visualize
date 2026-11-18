"""Run configuration and environment detection (local machine vs Kaggle).

Path resolution order for the raw data folder:
  1. --data-dir argument
  2. READMIT_DATA_DIR environment variable
  3. any folder under /kaggle/input that contains diabetic_data.csv (Kaggle)
  4. <project>/data/raw, ./data/raw, current directory

Output folder:
  1. --out-dir argument
  2. READMIT_OUT_DIR environment variable
  3. /kaggle/working/readmit_outputs on Kaggle
  4. <project>/outputs locally
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# code/readmit/config.py -> parents[0]=readmit, [1]=code, [2]=project1_clinical
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

DATA_FILE = "diabetic_data.csv"
MAPPING_FILE = "IDs_mapping.csv"

# Reference fingerprints of the original UCI files (dataset id 296).
# A mismatch is reported as a warning, not an error: Kaggle mirrors are
# sometimes re-saved with different line endings.
EXPECTED = {
    DATA_FILE: {
        "rows": 101_766,
        "cols": 50,
        "sha256": "da21942c25a10a51a8842907ab01a8530b06232616b17d317441d592a20c4359",
    },
    MAPPING_FILE: {
        "sha256": "d75eb9283fa1f81c6c096a53870f26eb7f9b1b3c2d52bc97b718b534ee9d8f91",
    },
}


def is_kaggle() -> bool:
    return KAGGLE_INPUT.exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def _find_case_insensitive(folder: Path, name: str) -> Path | None:
    if not folder.is_dir():
        return None
    target = name.lower()
    for p in folder.iterdir():
        if p.is_file() and p.name.lower() == target:
            return p
    return None


def find_raw_dir(explicit: str | os.PathLike | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("READMIT_DATA_DIR"):
        candidates.append(Path(os.environ["READMIT_DATA_DIR"]))
    if KAGGLE_INPUT.exists():
        candidates += sorted({p.parent for p in KAGGLE_INPUT.rglob("*") if p.name.lower() == DATA_FILE})
    candidates += [PROJECT_ROOT / "data" / "raw", Path.cwd() / "data" / "raw", Path.cwd()]
    for c in candidates:
        if _find_case_insensitive(c, DATA_FILE) and _find_case_insensitive(c, MAPPING_FILE):
            return c.resolve()
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find {DATA_FILE} + {MAPPING_FILE}. Tried:\n  {tried}\n"
        "Pass --data-dir or set READMIT_DATA_DIR."
    )


def default_out_dir(explicit: str | os.PathLike | None = None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    if os.environ.get("READMIT_OUT_DIR"):
        return Path(os.environ["READMIT_OUT_DIR"]).resolve()
    if KAGGLE_WORKING.exists():
        return KAGGLE_WORKING / "readmit_outputs"
    return PROJECT_ROOT / "outputs"


@dataclass
class Config:
    raw_dir: Path
    out_dir: Path
    seed: int = 42
    smoke: bool = False             # small patient sample for a fast end-to-end check
    smoke_patients: int = 8_000
    mice_m: int = 5                 # number of imputations
    mice_iter: int = 5              # chained-equation sweeps per imputation
    run_mixed: bool = False         # random-intercept logistic (slow; optional)
    gee_exchangeable: bool = False  # exchangeable-correlation GEE (slow on full data; optional)
    gee_maxiter: int = 30
    ipw_trim: tuple[float, float] = (0.01, 0.99)
    high_a1c_values: tuple[str, ...] = (">8",)   # Strack et al. "high" HbA1c; see clean.py
    specialty_top_k: int = 10
    test_size: float = 0.25         # risk-model hold-out share (grouped by patient)
    environment: str = field(default_factory=lambda: "kaggle" if is_kaggle() else "local")

    # derived folders ---------------------------------------------------------
    @property
    def interim(self) -> Path:
        return self.out_dir / "interim"

    @property
    def warehouse(self) -> Path:
        return self.out_dir / "warehouse"

    @property
    def tables(self) -> Path:
        return self.out_dir / "tables"

    @property
    def figures(self) -> Path:
        return self.out_dir / "figures"

    @property
    def models_dir(self) -> Path:
        return self.out_dir / "models"

    @property
    def logs(self) -> Path:
        return self.out_dir / "logs"

    def make_dirs(self) -> None:
        for d in (self.interim, self.warehouse, self.tables, self.figures, self.models_dir, self.logs):
            d.mkdir(parents=True, exist_ok=True)

    def data_path(self) -> Path:
        return _find_case_insensitive(self.raw_dir, DATA_FILE)  # type: ignore[return-value]

    def mapping_path(self) -> Path:
        return _find_case_insensitive(self.raw_dir, MAPPING_FILE)  # type: ignore[return-value]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["raw_dir"] = str(self.raw_dir)
        d["out_dir"] = str(self.out_dir)
        return d
