"""Small helpers: logging, hashing, JSON, timing, artifact IO."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("readmit")


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_dir / "pipeline.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    LOG.addHandler(sh)
    LOG.addHandler(fh)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def timed(label: str, sink: dict | None = None):
    t0 = time.perf_counter()
    LOG.info("▶ %s", label)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        LOG.info("✔ %s (%.1fs)", label, dt)
        if sink is not None:
            sink[label] = round(dt, 2)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run the earlier pipeline stage first.")
    return pd.read_parquet(path)


def environment_fingerprint() -> dict:
    pkgs = {}
    for name in ("pandas", "numpy", "scipy", "statsmodels", "sklearn", "plotly", "duckdb", "pyarrow"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001
            pkgs[name] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pkgs}
