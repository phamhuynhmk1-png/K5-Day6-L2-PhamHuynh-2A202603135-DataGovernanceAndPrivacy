"""Freeze T02 thresholds from the *dev* split + our manual dev labels (never eval, never gt/).

    .venv/Scripts/python -m tools.calibrate_t02            # prints the YAML block for configs/lab.yaml
    python -m tools.calibrate_t02 --out reports/t02_calibration.json

Protocol (also in reports/metric_contract.md):
  * labels: configs/t02_dev_manual_labels.csv, hand-labelled by looking at dev images only;
  * reference luminance per `light` = median mean luminance of *clean* dev images of that light;
  * per defect family: threshold = log-midpoint between the closest clean and closest defect dev image,
    scale = half of that log gap (src/t02_qc.py::calibrate_family);
  * baseline: a single global Laplacian threshold maximising F1 on dev (blur-or-not is all it can say,
    so every manual defect counts as positive).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.t02_qc import DEFECT_TYPES, FAMILY_FEATURE, add_dark_ratio, best_f1_threshold, calibrate_family, extract_features

FIXTURE = Path("fixtures/t02_raw_qc")
LABELS = Path("configs/t02_dev_manual_labels.csv")


def calibrate(fixture: Path = FIXTURE, labels_path: Path = LABELS) -> dict:
    images = pd.read_csv(fixture / "images.csv")
    dev = images[images["split"] == "dev"]
    labels = pd.read_csv(labels_path)[["image_id", "manual_defect_type", "is_defect"]]
    if not set(labels.image_id) <= set(dev.image_id):
        raise ValueError("manual labels must only cover dev images")
    feats = extract_features(dev, fixture).merge(labels, on="image_id", how="inner")

    clean = feats[feats.is_defect == 0]
    ref = {k: round(float(v), 2) for k, v in clean.groupby("light")["mean_luma"].median().items()}
    feats = add_dark_ratio(feats, ref)

    thresholds = {}
    for fam in DEFECT_TYPES:
        col, low_bad = FAMILY_FEATURE[fam]
        sub = feats[(feats.manual_defect_type == fam) | (feats.is_defect == 0)]
        thresholds[fam] = calibrate_family(sub[col], sub.manual_defect_type == fam, low_bad)

    base = best_f1_threshold(feats["lap_var"], feats["is_defect"] == 1, low_is_bad=True)
    return {"ref_luma_by_light": ref, "thresholds": thresholds, "baseline_lap_threshold": base,
            "n_dev_labelled": int(len(feats))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cal = calibrate()
    if a.out:
        Path(a.out).write_text(json.dumps(cal, indent=2), encoding="utf-8")
    params = {
        "ref_luma_by_light": cal["ref_luma_by_light"],
        "thresholds": {f: {"threshold": v["threshold"], "scale": v["scale"]} for f, v in cal["thresholds"].items()},
        "baseline_lap_threshold": cal["baseline_lap_threshold"]["threshold"],
    }
    print(json.dumps(cal, indent=2))
    print("\n# --- paste under `params:` in configs/lab.yaml ---")
    print(yaml.safe_dump(params, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
