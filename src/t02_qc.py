"""T02 Raw QC Triage: image-quality features + per-defect scoring.

One feature per defect family, each chosen so that the failure modes listed in docs/tracks/T02.md
do not trigger it:

| defect      | feature                                   | why it survives the traps                         |
|-------------|-------------------------------------------|---------------------------------------------------|
| blur        | Laplacian variance (whole image)          | threshold sits far below cam_B "soft optics"       |
| dark        | mean luminance / reference for its `light`| a night image is compared to night, not to day     |
| glare       | area of the largest connected blob > 235  | local: catches a burnt plate in a sharp image      |
| compression | 8x8 JPEG block-boundary gradient ratio    | independent of brightness and of overall sharpness |

Each feature becomes a signed *margin* in log space: 0 at the frozen threshold, +1 at the closest
dev defect, -1 at the closest dev clean image (see tools/calibrate_t02.py). The final review score
is the max margin over the four families, so one strong defect is enough to be reviewed first and
the families stay comparable without learning weights on 56 dev images.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .common.image_ops import laplacian_variance, load_rgb, to_gray

DEFECT_TYPES = ("blur", "dark", "glare", "compression")
GLARE_LEVEL = 235          # gray level treated as burnt-out
BLOCK = 8                  # JPEG block size
EPS = 1e-6


def blockiness(gray: np.ndarray, block: int = BLOCK) -> float:
    """Mean |gradient| across 8x8 block boundaries / mean |gradient| elsewhere (≈1 for clean images)."""
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    edge_x = dx[:, block - 1::block]
    edge_y = dy[block - 1::block, :]
    inner_x = np.delete(dx, np.s_[block - 1::block], axis=1)
    inner_y = np.delete(dy, np.s_[block - 1::block], axis=0)
    num = edge_x.mean() + edge_y.mean()
    den = inner_x.mean() + inner_y.mean()
    return float(num / (den + EPS))


def largest_bright_blob(gray: np.ndarray, level: int = GLARE_LEVEL) -> int:
    """Pixel area of the largest 4-connected region brighter than `level`."""
    labels, n = ndi.label(gray > level)
    if n == 0:
        return 0
    return int(np.bincount(labels.ravel())[1:].max())


def image_features(img: np.ndarray) -> dict:
    g = to_gray(img)
    return {
        "lap_var": laplacian_variance(img),
        "mean_luma": float(g.mean()),
        "glare_blob_px": largest_bright_blob(g),
        "blockiness": blockiness(g),
    }


def extract_features(images: pd.DataFrame, image_root: str | Path) -> pd.DataFrame:
    """One row per image: the metadata columns + raw features."""
    root = Path(image_root)
    rows = [image_features(load_rgb(root / p)) for p in images["path"]]
    return pd.concat([images.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def add_dark_ratio(feats: pd.DataFrame, ref_luma_by_light: dict) -> pd.DataFrame:
    """mean luminance relative to the reference of the same `light` slice (day/dusk/night)."""
    out = feats.copy()
    missing = sorted(set(out["light"]) - set(ref_luma_by_light))
    if missing:
        raise ValueError(f"no reference luminance for light={missing}; add it to params.ref_luma_by_light")
    out["dark_ratio"] = out["mean_luma"] / out["light"].map(ref_luma_by_light).astype(float)
    return out


# feature used for each family, and whether *low* values mean "defect"
FAMILY_FEATURE = {
    "blur": ("lap_var", True),
    "dark": ("dark_ratio", True),
    "glare": ("glare_blob_px", False),
    "compression": ("blockiness", False),
}


def margin(values: pd.Series, threshold: float, scale: float, low_is_bad: bool) -> pd.Series:
    """Signed log-margin: >0 means beyond the threshold (defect side), in units of `scale`."""
    v = np.log(values.astype(float).clip(lower=EPS) + (1.0 if not low_is_bad else 0.0))
    t = math.log(threshold + (1.0 if not low_is_bad else 0.0))
    m = (t - v) if low_is_bad else (v - t)
    return m / max(scale, EPS)


def score_images(feats: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Add `m_<family>` margins, `score` (= max margin), `pred_defect_type`, `pred_action`, `reason`.

    thresholds = {family: {"threshold": float, "scale": float}} (frozen in configs/lab.yaml).
    """
    out = feats.copy()
    cols = []
    for fam in DEFECT_TYPES:
        feat, low_bad = FAMILY_FEATURE[fam]
        th = thresholds[fam]
        out[f"m_{fam}"] = margin(out[feat], float(th["threshold"]), float(th["scale"]), low_bad)
        cols.append(f"m_{fam}")
    m = out[cols]
    out["score"] = m.max(axis=1).round(6)
    top = m.idxmax(axis=1).str[2:]
    flagged = out["score"] > 0
    out["pred_defect_type"] = np.where(flagged, top, "none")
    out["pred_action"] = np.where(flagged, "RECAPTURE", "KEEP")
    out["reason"] = [
        _reason(r, fams=DEFECT_TYPES) for r in out.itertuples(index=False)
    ]
    return out


def _reason(r, fams) -> str:
    parts = [f"{f}={getattr(r, 'm_' + f):+.2f}" for f in fams if getattr(r, "m_" + f) > 0]
    return ("flags: " + ", ".join(parts)) if parts else "no family beyond threshold"


def best_f1_threshold(values: pd.Series, is_defect: pd.Series, low_is_bad: bool = True) -> dict:
    """Threshold (between two consecutive sorted values) that maximises F1 on dev. Used by the baseline,
    whose single feature cannot separate the classes, so a gap-midpoint would be meaningless."""
    v = values.astype(float).to_numpy()
    y = is_defect.astype(bool).to_numpy()
    s = np.sort(np.unique(v))
    cands = (s[:-1] + s[1:]) / 2 if len(s) > 1 else s
    best = {"threshold": float(cands[0]), "f1": -1.0}
    for t in cands:
        pred = v < t if low_is_bad else v > t
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best["f1"]:
            best = {"threshold": round(float(t), 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}
    return best


def calibrate_family(values: pd.Series, is_defect: pd.Series, low_is_bad: bool) -> dict:
    """Threshold = log-midpoint between the closest clean and the closest defect; scale = half gap.

    If the classes overlap on dev, the gap is negative: we still return the midpoint but flag it,
    so the report can say the family is not separable on dev.
    """
    shift = 0.0 if low_is_bad else 1.0
    lv = np.log(values.astype(float).clip(lower=EPS) + shift)
    pos, neg = lv[is_defect.astype(bool)], lv[~is_defect.astype(bool)]
    if pos.empty or neg.empty:
        raise ValueError("calibration needs at least one defect and one clean dev image per family")
    if low_is_bad:
        worst_def, worst_clean = pos.max(), neg.min()
        gap = worst_clean - worst_def
    else:
        worst_def, worst_clean = pos.min(), neg.max()
        gap = worst_def - worst_clean
    mid = (worst_def + worst_clean) / 2
    return {
        "threshold": round(float(math.exp(mid) - shift), 4),
        "scale": round(float(max(abs(gap) / 2, 0.05)), 4),
        "separable_on_dev": bool(gap > 0),
        "n_defect": int(len(pos)),
        "n_clean": int(len(neg)),
    }
