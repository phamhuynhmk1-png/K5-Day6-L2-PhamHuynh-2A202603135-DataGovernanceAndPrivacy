"""T02 Raw QC Triage — PoC pipeline (owned by the group).

    python -m src.main --config configs/lab.yaml --track t02 --output out/poc --run-name poc --offline --seed 42

01 load      images.csv + provenance check (manifest sha256)
02 process   4 defect families (blur / dark / glare / compression) -> signed margins -> score = max margin
03 rank      top-B review queue on the release pool (eval split) + offline HTML contact sheet
04 review    reviewer decisions from out/.../review_queue_reviewed.csv -> overrides + audit log
05 decide    release decision GO / HOLD / REWORK with reason_code + owner

Thresholds are frozen in configs/lab.yaml (produced by tools/calibrate_t02.py on the dev split only).
The pipeline never reads gt/ and has no randomness (ctx.rng unused), so two runs give identical files.
"""
from __future__ import annotations

import time

import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track
from .t02_qc import DEFECT_TYPES, add_dark_ratio, extract_features, score_images

ALLOWED_OVERRIDE = set(DEFECT_TYPES) | {"none"}
FEATURE_COLS = ["lap_var", "mean_luma", "dark_ratio", "glare_blob_px", "blockiness"] + [f"m_{f}" for f in DEFECT_TYPES]


def _need_param(ctx: RunContext, key: str):
    v = ctx.param(key)
    if v is None:
        raise ValueError(f"params.{key} missing in config: run `python -m tools.calibrate_t02` and freeze it in configs/lab.yaml")
    return v


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict:
    if ctx.track != "t02":
        raise NotImplementedError(f"src.pipeline implements T02 only (got {ctx.track})")
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")
    images = pd.read_csv(ctx.input_dir / "images.csv")
    need = {"image_id", "path", "camera_id", "light", "split"}
    if not need <= set(images.columns):
        raise ValueError(f"images.csv missing columns {sorted(need - set(images.columns))}")
    if images.image_id.duplicated().any():
        raise ValueError("images.csv has duplicate image_id")
    missing = [p for p in images.path if not (ctx.input_dir / p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images listed in images.csv do not exist, first: {missing[:3]}")
    return {"images": images}


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    t0 = time.perf_counter()
    feats = extract_features(data["images"], ctx.input_dir)
    feats = add_dark_ratio(feats, _need_param(ctx, "ref_luma_by_light"))
    scored = score_images(feats, _need_param(ctx, "thresholds"))
    ms = (time.perf_counter() - t0) * 1000 / max(len(scored), 1)
    ctx.stats["ms_per_image"] = round(ms, 2)
    ctx.stats["n_flagged_auto"] = int((scored.pred_action == "RECAPTURE").sum())
    cols = ["image_id", "score", "pred_defect_type", "pred_action", "reason", "camera_id", "light", "split", "path"]
    return scored[cols + FEATURE_COLS].sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    budget = int(ctx.param("budget", 20))
    split = ctx.param("queue_split")
    pool = scored[scored.split == split] if split else scored
    pool = pool.assign(slice=pool.camera_id + "/" + pool.light,
                       suggested=pool.pred_action + ":" + pool.pred_defect_type)
    q = make_review_queue(pool, id_col="image_id", score_col="score", budget=budget, reason_col="reason",
                          action_col="suggested", slice_col="slice", evidence_col="path")
    write_review_queue(q, ctx.out("review_queue.csv"))
    render_contact_sheet(q, ctx.out("contact_sheet.html"), image_root=ctx.input_dir,
                         title=f"T02 Raw QC — top-{budget} review queue ({split or 'all'})")
    ctx.audit.log(actor="pipeline", action="review_queue_created", item_id=str(ctx.out("review_queue.csv")),
                  after={"budget": budget, "pool": split or "all", "n_pool": int(len(pool)),
                         "items": q.item_id.tolist()})
    return q


# ---------------------------------------------------------------- 04 review
def apply_review(scored: pd.DataFrame, reviewed: pd.DataFrame, audit=None) -> pd.DataFrame:
    """Reviewer decision -> final label (raw `score` is never changed, so the ranking stays auditable).

    approve  : keep machine label           reject : false alarm -> KEEP / none
    edit     : override_value = defect type (blur|dark|glare|compression|none)
    escalate : keep label, mark for a second opinion (blocks GO)          skip : nothing
    """
    out = scored.copy()
    out["final_defect_type"] = out["pred_defect_type"]
    out["final_action"] = out["pred_action"]
    out["review_status"] = "unreviewed"
    out["reviewer"] = ""
    ids = set(out.image_id)
    for r in reviewed.itertuples(index=False):
        if r.item_id not in ids:
            raise ValueError(f"reviewed item {r.item_id} is not in this run's predictions")
        m = out.image_id == r.item_id
        before = out.loc[m, ["final_defect_type", "final_action"]].iloc[0].to_dict()
        dec = r.reviewer_decision
        if dec == "reject":
            out.loc[m, ["final_defect_type", "final_action"]] = ["none", "KEEP"]
        elif dec == "edit":
            val = str(r.override_value).strip().lower()
            if val not in ALLOWED_OVERRIDE:
                raise ValueError(f"{r.item_id}: override_value {r.override_value!r} not in {sorted(ALLOWED_OVERRIDE)}")
            out.loc[m, ["final_defect_type", "final_action"]] = [val, "KEEP" if val == "none" else "RECAPTURE"]
        status = {"approve": "approved", "reject": "rejected", "edit": "edited", "escalate": "escalated", "skip": "skipped"}[dec]
        out.loc[m, "review_status"] = status
        out.loc[m, "reviewer"] = r.reviewer
        if audit:
            after = out.loc[m, ["final_defect_type", "final_action"]].iloc[0].to_dict()
            audit.log(actor=r.reviewer, action=f"review:{dec}", item_id=r.item_id, before=before, after=after,
                      reason=r.review_note)
    return out


def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if not reviewed_path.exists():
        ctx.stats["n_reviewed"] = 0
        return apply_review(scored, pd.DataFrame(columns=["item_id"]))
    reviewed = load_review_decisions(reviewed_path)
    ctx.stats["n_reviewed"] = int(len(reviewed))
    return apply_review(scored, reviewed, ctx.audit)


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame, queue: pd.DataFrame) -> Decision:
    owner = ctx.config.get("owner") or "TODO-owner"
    split = ctx.param("queue_split")
    pool = final[final.split == split] if split else final
    in_queue = final[final.image_id.isin(queue.item_id)]
    n_rec = int((pool.final_action == "RECAPTURE").sum())
    rate = n_rec / max(len(pool), 1)
    snapshot = {
        "pool": split or "all", "n_pool": int(len(pool)), "n_recapture": n_rec, "recapture_rate": round(rate, 4),
        "by_type": pool[pool.final_action == "RECAPTURE"].final_defect_type.value_counts().to_dict(),
        "queue_reviewed": int((in_queue.review_status != "unreviewed").sum()), "queue_size": int(len(queue)),
    }
    evidence = [str(ctx.out("ranked_review.csv")), str(ctx.out("review_queue.csv")), str(ctx.out("audit_log.jsonl"))]
    kw = dict(owner=owner, scope=f"t02:{snapshot['pool']}", evidence=evidence, metrics=snapshot)
    if (in_queue.review_status == "unreviewed").any():
        return Decision("HOLD", "PENDING_HUMAN_REVIEW",
                        f"{snapshot['queue_size'] - snapshot['queue_reviewed']} of {len(queue)} queued images not reviewed yet", **kw)
    if (final.review_status == "escalated").any():
        return Decision("HOLD", "ESCALATED_BY_REVIEWER", "a reviewer asked for a second opinion on at least one image", **kw)
    max_rate = float(ctx.param("max_recapture_rate", 0.25))
    if rate > max_rate:
        return Decision("REWORK", "RECAPTURE_RATE_ABOVE_POLICY",
                        f"{n_rec}/{len(pool)} images ({rate:.0%}) need recapture > policy {max_rate:.0%}; recapture before release", **kw)
    return Decision("GO", "QC_PASS_WITH_RECAPTURE_LIST",
                    f"{n_rec}/{len(pool)} images ({rate:.0%}) flagged for recapture, within policy {max_rate:.0%}", **kw)


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    final[spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]].to_csv(ctx.out(spec.pred_file), index=False)
    decision = decide(ctx, final, queue)
    write_decision(decision, ctx.out("decision.json"), ctx.out("decision_log.csv"))
    ctx.audit.log(actor="pipeline", action="decision", item_id=decision.scope, after=decision.canonical,
                  reason=decision.reason_code)
