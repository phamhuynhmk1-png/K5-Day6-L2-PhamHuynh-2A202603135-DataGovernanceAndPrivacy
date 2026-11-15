"""Internal check on the DEV split with our manual labels (the official gt/ is instructor-only).

    python -m tools.dev_eval_t02 out/baseline/ranked_review.csv out/poc/ranked_review.csv

Uses the repo's own evaluator (src/metrics/tracks.py::eval_t02) unchanged, only pointed at
configs/t02_dev_manual_labels.csv and split=dev. Thresholds were fitted on these same images, so the
numbers are optimistic: they show the wiring and the ranking behaviour, not generalisation.
"""
from __future__ import annotations

import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.metrics.tracks import eval_t02

LABELS = Path("configs/t02_dev_manual_labels.csv")


def dev_eval(pred_path: str | Path, budget: int = 20) -> pd.DataFrame:
    lab = pd.read_csv(LABELS)
    gt = lab.rename(columns={"manual_defect_type": "defect_type"}).assign(split="dev")
    with tempfile.TemporaryDirectory() as td:
        gpath = Path(td) / "dev_labels.csv"
        gt.to_csv(gpath, index=False)
        res = eval_t02(gpath, Path(pred_path), {"budget": budget, "split": "dev"})
    rows = [{"metric": m.name, **{k: v for k, v in asdict(m).items() if k != "name"}} for m in res.metrics]
    return pd.DataFrame(rows), res.slices


def main(argv=None) -> int:
    paths = (argv if argv is not None else sys.argv[1:]) or ["out/baseline/ranked_review.csv", "out/poc/ranked_review.csv"]
    for p in paths:
        m, s = dev_eval(p)
        keep = [c for c in ["metric", "value", "numerator", "denominator", "slice"] if c in m.columns]
        print(f"\n== {p} (DEV, manual labels, B=20)")
        print(m[keep].to_string(index=False))
        print(s.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
