"""T02 baseline required by the handout: ONE global Laplacian-variance threshold for every image.

    python -m src.main --config configs/lab.yaml --track t02 --pipeline src.baseline_t02 \
        --output out/baseline --run-name baseline --offline --seed 42

score = log(threshold / lap_var): low sharpness = reviewed first, no camera/light normalisation, no other
feature. The threshold (params.baseline_lap_threshold) maximises F1 on dev manual labels
(tools/calibrate_t02.py). Same load / queue / review / decision code as the PoC, so only the scoring differs.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import pipeline as poc
from .common.context import RunContext
from .common.decision import write_decision
from .common.image_ops import laplacian_variance, load_rgb
from .metrics.tracks import get_track


def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    thr = float(poc._need_param(ctx, "baseline_lap_threshold"))
    imgs = data["images"].reset_index(drop=True)
    lap = np.array([laplacian_variance(load_rgb(ctx.input_dir / p)) for p in imgs.path])
    score = np.round(math.log(thr) - np.log(np.maximum(lap, 1e-6)), 6)
    flagged = score > 0
    out = imgs.assign(lap_var=lap, score=score,
                      pred_defect_type=np.where(flagged, "blur", "none"),
                      pred_action=np.where(flagged, "RECAPTURE", "KEEP"),
                      reason=[f"lap_var={v:.1f} {'<' if f else '>='} {thr:.1f}" for v, f in zip(lap, flagged)])
    cols = ["image_id", "score", "pred_defect_type", "pred_action", "reason", "camera_id", "light", "split", "path", "lap_var"]
    return out[cols].sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = poc.load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = poc.rank(ctx, scored)
    final = poc.review(ctx, scored, queue)
    final[spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(poc.decide(ctx, final, queue), ctx.out("decision.json"), ctx.out("decision_log.csv"))
