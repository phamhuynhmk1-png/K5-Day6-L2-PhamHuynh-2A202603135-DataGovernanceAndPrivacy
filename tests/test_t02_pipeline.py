"""Group tests for the T02 PoC: success paths, failure modes from the handout, human-review edge cases."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common import corruptions
from src.common.image_ops import load_rgb
from src.common.io import load_config
from src.main import main
from src.t02_qc import add_dark_ratio, blockiness, image_features, score_images

FIX = Path("fixtures/t02_raw_qc")
PARAMS = load_config("configs/lab.yaml")["params"]


def _score_one(img: np.ndarray, light: str = "day") -> pd.Series:
    f = pd.DataFrame([image_features(img) | {"light": light}])
    return score_images(add_dark_ratio(f, PARAMS["ref_luma_by_light"]), PARAMS["thresholds"]).iloc[0]


# ------------------------------------------------------------------ features (success)
@pytest.mark.parametrize("name,expected", [("blur", "blur"), ("dark", "dark"), ("glare", "glare"), ("jpeg", "compression")])
def test_injected_corruption_is_flagged_with_right_type(name, expected):
    clean = load_rgb(FIX / "images/img_02_0.jpg")                 # sharp cam_A day image (dev)
    bad = corruptions.apply(name, clean, severity=2, seed=42)
    r = _score_one(bad)
    assert r.pred_action == "RECAPTURE" and r.pred_defect_type == expected, r.reason


def test_clean_day_image_is_kept():
    assert _score_one(load_rgb(FIX / "images/img_02_0.jpg")).pred_action == "KEEP"


# ------------------------------------------------------------------ failure modes from docs/tracks/T02.md
def test_clean_night_image_is_not_flagged_dark():
    """'Ảnh ban đêm hữu ích bị loại hết': night is compared to the night reference, not to day."""
    r = _score_one(load_rgb(FIX / "images/img_18_1.jpg"), light="night")
    assert r.pred_action == "KEEP", r.reason
    assert _score_one(load_rgb(FIX / "images/img_18_1.jpg"), light="day").pred_defect_type == "dark"


def test_soft_cam_b_optics_not_flagged_blur():
    """'cam_B quang học mềm bị báo nhầm': clean cam_B images have low Laplacian but stay above the blur threshold."""
    for iid, light in [("img_00_0", "day"), ("img_21_0", "day"), ("img_12_0", "dusk")]:
        r = _score_one(load_rgb(FIX / f"images/{iid}.jpg"), light=light)
        assert r.m_blur < 0, (iid, r.reason)


def test_local_glare_in_sharp_image_is_caught():
    """'Ảnh nét nhưng vùng biển số cháy sáng': a small glare spot in an otherwise sharp image."""
    clean = load_rgb(FIX / "images/img_02_0.jpg")
    bad = corruptions.apply("glare", clean, severity=1, seed=42, radius_frac=0.12)
    r = _score_one(bad)
    assert r.pred_defect_type == "glare", r.reason


def test_blockiness_is_about_one_for_smooth_noise():
    g = np.random.default_rng(42).normal(128, 20, (180, 320)).astype(np.float32)
    assert 0.9 < blockiness(g) < 1.1


# ------------------------------------------------------------------ end-to-end + idempotency
def _run(out: Path, *extra: str) -> int:
    return main(["--config", "configs/lab.yaml", "--track", "t02", "--output", str(out), "--offline", "--seed", "42", *extra])


def test_e2e_runs_twice_with_identical_outputs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert _run(a) == 0 and _run(b) == 0
    ra, rb = pd.read_csv(a / "ranked_review.csv"), pd.read_csv(b / "ranked_review.csv")
    pd.testing.assert_frame_equal(ra, rb)
    assert {"image_id", "score", "pred_defect_type", "pred_action"} <= set(ra.columns)
    assert len(ra) == 96 and ra.image_id.is_unique
    q = pd.read_csv(a / "review_queue.csv")
    assert len(q) == 20 and (a / "contact_sheet.html").exists()
    d = json.loads((a / "decision.json").read_text(encoding="utf-8"))
    assert d["canonical"] == "HOLD" and d["reason_code"] == "PENDING_HUMAN_REVIEW"


def test_baseline_runs_on_same_pool(tmp_path):
    assert _run(tmp_path, "--pipeline", "src.baseline_t02") == 0
    q = pd.read_csv(tmp_path / "review_queue.csv")
    assert len(q) == 20


# ------------------------------------------------------------------ human review
def _reviewed_queue(out: Path) -> pd.DataFrame:
    assert _run(out) == 0
    return pd.read_csv(out / "review_queue.csv", dtype=str).fillna("")


def test_review_without_reviewer_name_fails_loudly(tmp_path):
    q = _reviewed_queue(tmp_path)
    q.loc[0, "reviewer_decision"] = "approve"                       # no reviewer -> must not pass silently
    q.to_csv(tmp_path / "review_queue_reviewed.csv", index=False)
    assert _run(tmp_path) == 2


def test_review_overrides_are_applied_and_audited(tmp_path):
    q = _reviewed_queue(tmp_path)
    q["reviewer"], q["reviewer_decision"] = "test-reviewer", "approve"
    q.loc[0, ["reviewer_decision", "review_note"]] = ["reject", "false alarm in test"]
    q.loc[1, ["reviewer_decision", "override_value"]] = ["edit", "blur"]
    q.to_csv(tmp_path / "review_queue_reviewed.csv", index=False)
    assert _run(tmp_path) == 0
    r = pd.read_csv(tmp_path / "ranked_review.csv").set_index("image_id")
    assert r.loc[q.item_id[0], "final_action"] == "KEEP" and r.loc[q.item_id[0], "review_status"] == "rejected"
    assert r.loc[q.item_id[1], "final_defect_type"] == "blur"
    events = [json.loads(l) for l in (tmp_path / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()]
    rej = [e for e in events if e["action"] == "review:reject"]
    assert rej and rej[0]["actor"] == "test-reviewer" and rej[0]["after"]["final_action"] == "KEEP"
    d = json.loads((tmp_path / "decision.json").read_text(encoding="utf-8"))
    assert d["canonical"] in {"GO", "REWORK"} and d["owner"]


def test_bad_override_value_is_rejected(tmp_path):
    q = _reviewed_queue(tmp_path)
    q.loc[0, ["reviewer", "reviewer_decision", "override_value"]] = ["test-reviewer", "edit", "fog"]
    q.to_csv(tmp_path / "review_queue_reviewed.csv", index=False)
    assert _run(tmp_path) == 2


def test_missing_frozen_thresholds_fail_clearly(tmp_path):
    cfg = load_config("configs/lab.yaml")
    cfg["params"].pop("thresholds")
    p = tmp_path / "lab.yaml"
    import yaml
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (tmp_path / "tracks").mkdir()
    (tmp_path / "tracks" / "t02.yaml").write_text(Path("configs/tracks/t02.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    rc = main(["--config", str(p), "--track", "t02", "--output", str(tmp_path / "o"), "--offline"])
    assert rc == 2
