#!/usr/bin/env python
"""End-to-end pipeline runner — works the same locally and on Kaggle.

Examples
  python run_pipeline.py --smoke                     # ~1-2 min validation on 8,000 sampled patients
  python run_pipeline.py                             # full run, all stages
  python run_pipeline.py --stages models,risk,viz    # re-run later stages from saved artifacts
  python run_pipeline.py --data-dir D:/data/raw --out-dir D:/runs/full --mice-m 10 --mixed

Stages (in order): ingest, profile, clean, integrate, store, qa, tests, models, missing, risk, viz
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from readmit import __version__  # noqa: E402
from readmit import (clean, ingest, integrate, missing, models, profiling, quality, risk,  # noqa: E402
                     stats_tests, storage, viz)
from readmit.config import Config, default_out_dir, find_raw_dir  # noqa: E402
from readmit.utils import LOG, environment_fingerprint, setup_logging, timed, write_json  # noqa: E402

STAGES = {
    "ingest": ingest.run, "profile": profiling.run, "clean": clean.run, "integrate": integrate.run,
    "store": storage.run, "qa": quality.run, "tests": stats_tests.run, "models": models.run,
    "missing": missing.run, "risk": risk.run, "viz": viz.run,
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", help="folder with diabetic_data.csv and IDs_mapping.csv")
    p.add_argument("--out-dir", help="output folder (default: <project>/outputs or /kaggle/working/readmit_outputs)")
    p.add_argument("--stages", default="all", help="comma list of stages or 'all'")
    p.add_argument("--smoke", action="store_true", help="fast validation run on a patient sample")
    p.add_argument("--smoke-patients", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mice-m", type=int, default=5)
    p.add_argument("--mice-iter", type=int, default=5)
    p.add_argument("--mixed", action="store_true", help="also fit the random-intercept model (slow)")
    p.add_argument("--gee-exchangeable", action="store_true", help="also fit exchangeable-correlation GEE (slow)")
    # Jupyter/Kaggle pass extra '-f kernel.json' args: ignore unknowns
    args, _ = p.parse_known_args(argv)
    return args


def main(argv=None) -> int:
    a = parse_args(argv)
    raw_dir = find_raw_dir(a.data_dir)
    out_dir = default_out_dir(a.out_dir)
    if a.smoke and not a.out_dir:
        out_dir = out_dir.parent / (out_dir.name + "_smoke")
    cfg = Config(raw_dir=raw_dir, out_dir=out_dir, seed=a.seed, smoke=a.smoke, smoke_patients=a.smoke_patients,
                 mice_m=2 if a.smoke and a.mice_m == 5 else a.mice_m, mice_iter=2 if a.smoke and a.mice_iter == 5 else a.mice_iter,
                 run_mixed=a.mixed, gee_exchangeable=a.gee_exchangeable)
    cfg.make_dirs()
    setup_logging(cfg.logs)
    stages = list(STAGES) if a.stages == "all" else [s.strip() for s in a.stages.split(",")]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        LOG.error("Unknown stage(s): %s. Valid: %s", unknown, ", ".join(STAGES))
        return 2
    LOG.info("readmit %s | env=%s | smoke=%s | raw=%s | out=%s", __version__, cfg.environment, cfg.smoke,
             cfg.raw_dir, cfg.out_dir)

    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "package_version": __version__,
                "argv": sys.argv[1:], "config": cfg.to_dict(), "environment": environment_fingerprint(),
                "stages": {}, "timings_s": {}}
    status = 0
    t0 = time.perf_counter()
    for s in stages:
        try:
            with timed(s, manifest["timings_s"]):
                STAGES[s](cfg)
            manifest["stages"][s] = "ok"
        except Exception as e:  # noqa: BLE001
            manifest["stages"][s] = f"FAILED: {e}"
            LOG.error("Stage '%s' failed: %s\n%s", s, e, traceback.format_exc())
            status = 1
            break
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["total_s"] = round(time.perf_counter() - t0, 1)
    write_json(manifest, cfg.logs / "run_manifest.json")
    LOG.info("%s in %.1fs — manifest: %s", "DONE" if status == 0 else "STOPPED", manifest["total_s"],
             cfg.logs / "run_manifest.json")
    return status


if __name__ == "__main__":
    sys.exit(main())
