#!/usr/bin/env python
"""Export the dashboard figures as static PNG files for the written report.

  python export_figures.py --out-dir ../outputs_kaggle            # reads tables/, writes figures_png/
  python export_figures.py --out-dir ../outputs --width 1100 --scale 2

Requires `kaleido` (pip install kaleido) and a Chrome/Chromium install for kaleido >= 1.0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from readmit.config import Config  # noqa: E402
from readmit.viz import STATIC_OVERRIDES, build_figures  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, help="a pipeline output folder (contains tables/ and interim/)")
    p.add_argument("--width", type=int, default=1100)
    p.add_argument("--scale", type=float, default=2.0)
    a = p.parse_args(argv)
    out = Path(a.out_dir).resolve()
    cfg = Config(raw_dir=out, out_dir=out)
    dest = out / "figures_png"
    dest.mkdir(parents=True, exist_ok=True)
    for f in build_figures(cfg):
        if f.fig is None:
            print(f"skip {f.key} (source table missing)")
            continue
        path = dest / f"{f.key}.png"
        f.fig.write_image(path, width=a.width, scale=a.scale)
        print(f"wrote {path}")
        if f.key in STATIC_OVERRIDES:  # print-friendly version (no dropdown / filtered + labelled)
            sf = STATIC_OVERRIDES[f.key](cfg)
            if sf is not None:
                sp = dest / f"{f.key}_static.png"
                sf.write_image(sp, width=a.width, scale=a.scale)
                print(f"wrote {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
