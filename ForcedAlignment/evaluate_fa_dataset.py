"""
evaluate_fa_dataset.py
----------------------
Evaluate ForcedAlignment Task 2 prediction CSVs against per-clip EAF ground
truth tiers.

The primary matching strategy is positional and IoU-only:
    pred[i] <-> gt[i] by time order inside each clip.

This avoids text-match leakage because several GT tiers were derived from the
same gloss strings used as alignment input.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


FA_DIR = Path(__file__).resolve().parent
DEFAULT_EAF_DIR = FA_DIR / "elan_forced_alignment"
DEFAULT_PRED_DIR = FA_DIR / "output" / "predictions"
DEFAULT_OUT_DIR = FA_DIR / "output" / "evaluation"

FPS_DEFAULT = 25


@dataclass(frozen=True)
class EvalConfig:
    key: str
    label: str
    gt_tier: str
    pred_csv: str
    drop_gt_sil: bool = False


CONFIGS: dict[str, EvalConfig] = {
    "1": EvalConfig(
        key="1",
        label="config1_CC_Aligned_pred",
        gt_tier="CC_Aligned",
        pred_csv="config1_CC_Aligned_pred.csv",
        drop_gt_sil=True,
    ),
    "2": EvalConfig(
        key="2",
        label="config2_CC_Aligned_silmodel_pred",
        gt_tier="CC_Aligned",
        pred_csv="config2_CC_Aligned_silmodel_pred.csv",
    ),
    "3": EvalConfig(
        key="3",
        label="config3_Gloss_Labeling_pred",
        gt_tier="Gloss_Labeling",
        pred_csv="config3_Gloss_Labeling_pred.csv",
    ),
    "4": EvalConfig(
        key="4",
        label="config4_Gloss_Labeling1_pred",
        gt_tier="Gloss_Labeling1",
        pred_csv="config4_Gloss_Labeling1_pred.csv",
    ),
    "5": EvalConfig(
        key="5",
        label="config5_Gloss_Labeling2_pred",
        gt_tier="Gloss_Labeling2",
        pred_csv="config5_Gloss_Labeling2_pred.csv",
    ),
}

SIL_TOKENS = {"sil", "sil1", "sil2"}


def numeric_sort_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def normalize_token(text: str) -> str:
    return " ".join(text.split()).lower()


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def load_tier_entries(eaf_path: Path, tier_id: str, drop_sil: bool = False) -> list[dict]:
    tree = ET.parse(eaf_path)
    root = tree.getroot()
    ts_map = {
        ts.get("TIME_SLOT_ID"): float(ts.get("TIME_VALUE", 0)) / 1000.0
        for ts in root.findall(".//TIME_SLOT")
    }

    target = None
    for tier in root.findall("TIER"):
        if tier.get("TIER_ID") == tier_id:
            target = tier
            break
    if target is None:
        raise ValueError(f"Tier {tier_id!r} not found in {eaf_path}")

    entries: list[dict] = []
    for ann in target.findall("ANNOTATION"):
        elem = next(iter(ann), None)
        if elem is None or elem.tag != "ALIGNABLE_ANNOTATION":
            continue
        t1 = ts_map.get(elem.get("TIME_SLOT_REF1"))
        t2 = ts_map.get(elem.get("TIME_SLOT_REF2"))
        val = elem.find("ANNOTATION_VALUE")
        text = (val.text or "").strip() if val is not None else ""
        if t1 is None or t2 is None or not text:
            continue
        if drop_sil and normalize_token(text) in SIL_TOKENS:
            continue
        if t2 < t1:
            t1, t2 = t2, t1
        entries.append({"start": t1, "end": t2, "token": text})

    entries.sort(key=lambda x: (x["start"], x["end"], x["token"]))
    return entries


def load_predictions(pred_csv: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with open(pred_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_id = str(row["clip_id"])
            out.setdefault(clip_id, []).append({
                "start": float(row["start_s"]),
                "end": float(row["end_s"]),
                "token": row["token"],
                "fallback": row.get("fallback", ""),
                "token_idx": int(row.get("token_idx") or 0),
            })
    for rows in out.values():
        rows.sort(key=lambda x: (x["token_idx"], x["start"], x["end"]))
    return out


def frames_for(entries: list[dict], max_time: float, fps: int) -> list[int]:
    total_frames = max(1, round(max_time * fps))
    frames = [-1] * total_frames
    for idx, item in enumerate(entries):
        s = max(0, min(round(item["start"] * fps), total_frames))
        e = max(0, min(round(item["end"] * fps), total_frames))
        if s < e:
            frames[s:e] = [idx] * (e - s)
    return frames


def evaluate_config(
    cfg: EvalConfig,
    eaf_dir: Path,
    pred_dir: Path,
    out_dir: Path,
    threshold: float,
    fps: int,
) -> dict:
    pred_csv = pred_dir / cfg.pred_csv
    if not pred_csv.exists():
        raise FileNotFoundError(pred_csv)
    predictions_by_clip = load_predictions(pred_csv)

    out_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = out_dir / f"eval_config{cfg.key}.csv"

    detail_rows: list[dict] = []
    all_ious: list[float] = []
    all_offsets_start: list[float] = []
    all_offsets_end: list[float] = []
    text_matches = 0
    paired_count = 0
    hits_50 = 0
    hits_30 = 0
    any_overlap = 0
    total_pred = 0
    total_gt = 0
    fallback_count = 0
    total_frames = 0
    frame_correct = 0

    clip_ids = sorted(predictions_by_clip.keys(), key=lambda x: int(x) if x.isdigit() else x)
    for clip_id in clip_ids:
        eaf_path = eaf_dir / f"{clip_id}.eaf"
        if not eaf_path.exists():
            raise FileNotFoundError(eaf_path)
        gt = load_tier_entries(eaf_path, cfg.gt_tier, drop_sil=cfg.drop_gt_sil)
        pred = predictions_by_clip.get(clip_id, [])

        total_pred += len(pred)
        total_gt += len(gt)
        fallback_count += sum(1 for p in pred if p.get("fallback"))

        n_pair = min(len(pred), len(gt))
        for i in range(n_pair):
            p = pred[i]
            g = gt[i]
            io = interval_iou(p["start"], p["end"], g["start"], g["end"])
            paired_count += 1
            all_ious.append(io)
            all_offsets_start.append(p["start"] - g["start"])
            all_offsets_end.append(p["end"] - g["end"])
            if io >= threshold:
                hits_50 += 1
            if io >= 0.3:
                hits_30 += 1
            if io > 0:
                any_overlap += 1
            if normalize_token(p["token"]) == normalize_token(g["token"]):
                text_matches += 1
            detail_rows.append({
                "clip_id": clip_id,
                "idx": i,
                "pred_token": p["token"],
                "pred_start_s": round(p["start"], 3),
                "pred_end_s": round(p["end"], 3),
                "gt_token": g["token"],
                "gt_start_s": round(g["start"], 3),
                "gt_end_s": round(g["end"], 3),
                "iou": round(io, 4),
                "match_iou_0_5": io >= threshold,
                "start_offset_s": round(p["start"] - g["start"], 3),
                "end_offset_s": round(p["end"] - g["end"], 3),
                "text_match_reference_only": normalize_token(p["token"]) == normalize_token(g["token"]),
                "fallback": p.get("fallback", ""),
            })

        for i in range(n_pair, len(pred)):
            p = pred[i]
            detail_rows.append({
                "clip_id": clip_id,
                "idx": i,
                "pred_token": p["token"],
                "pred_start_s": round(p["start"], 3),
                "pred_end_s": round(p["end"], 3),
                "gt_token": "",
                "gt_start_s": "",
                "gt_end_s": "",
                "iou": 0.0,
                "match_iou_0_5": False,
                "start_offset_s": "",
                "end_offset_s": "",
                "text_match_reference_only": False,
                "fallback": p.get("fallback", ""),
            })

        for i in range(n_pair, len(gt)):
            g = gt[i]
            detail_rows.append({
                "clip_id": clip_id,
                "idx": i,
                "pred_token": "",
                "pred_start_s": "",
                "pred_end_s": "",
                "gt_token": g["token"],
                "gt_start_s": round(g["start"], 3),
                "gt_end_s": round(g["end"], 3),
                "iou": 0.0,
                "match_iou_0_5": False,
                "start_offset_s": "",
                "end_offset_s": "",
                "text_match_reference_only": False,
                "fallback": "",
            })

        max_time = max(
            [x["end"] for x in gt] + [x["end"] for x in pred] + [0.001],
        )
        pred_frames = frames_for(pred, max_time, fps)
        gt_frames = frames_for(gt, max_time, fps)
        n = min(len(pred_frames), len(gt_frames))
        total_frames += n
        frame_correct += sum(1 for a, b in zip(pred_frames[:n], gt_frames[:n]) if a == b)

    precision = hits_50 / total_pred if total_pred else 0.0
    recall = hits_50 / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = hits_50 / total_gt if total_gt else 0.0
    mean_iou = statistics.mean(all_ious) if all_ious else 0.0
    median_iou = statistics.median(all_ious) if all_ious else 0.0
    mean_abs_start = statistics.mean(abs(x) for x in all_offsets_start) if all_offsets_start else 0.0
    mean_abs_end = statistics.mean(abs(x) for x in all_offsets_end) if all_offsets_end else 0.0
    frame_acc = frame_correct / total_frames if total_frames else 0.0

    fieldnames = [
        "clip_id", "idx",
        "pred_token", "pred_start_s", "pred_end_s",
        "gt_token", "gt_start_s", "gt_end_s",
        "iou", "match_iou_0_5",
        "start_offset_s", "end_offset_s",
        "text_match_reference_only", "fallback",
    ]
    with open(detail_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    return {
        "config": cfg.key,
        "label": cfg.label,
        "gt_tier": cfg.gt_tier,
        "pred_csv": str(pred_csv),
        "clips": len(clip_ids),
        "total_pred": total_pred,
        "total_gt": total_gt,
        "paired": paired_count,
        "precision_iou_0_5": precision,
        "recall_iou_0_5": recall,
        "f1_iou_0_5": f1,
        "accuracy_iou_0_5": accuracy,
        "mean_iou": mean_iou,
        "median_iou": median_iou,
        "pct_iou_ge_0_5_over_gt": hits_50 / total_gt if total_gt else 0.0,
        "pct_iou_ge_0_3_over_gt": hits_30 / total_gt if total_gt else 0.0,
        "pct_any_overlap_over_gt": any_overlap / total_gt if total_gt else 0.0,
        "pct_zero_overlap_paired": 1.0 - (any_overlap / paired_count if paired_count else 0.0),
        "mean_abs_start_offset_s": mean_abs_start,
        "mean_abs_end_offset_s": mean_abs_end,
        "frame_accuracy": frame_acc,
        "fallback_predictions": fallback_count,
        "text_match_reference_only": text_matches / paired_count if paired_count else 0.0,
        "detail_csv": str(detail_csv),
    }


def parse_configs(spec: str) -> list[EvalConfig]:
    if spec.lower() == "all":
        keys = ["1", "2", "3", "4", "5"]
    else:
        keys = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [k for k in keys if k not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown configs: {', '.join(unknown)}")
    return [CONFIGS[k] for k in keys]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ForcedAlignment Task 2 predictions.")
    parser.add_argument("--eaf-dir", type=Path, default=DEFAULT_EAF_DIR,
                        help="Directory of ground-truth EAF files (default: ForcedAlignment/elan_forced_alignment/)")
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR,
                        help="Directory of prediction CSVs from run_forced_alignment.py (default: output/predictions/)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Where to write evaluation CSVs (default: output/evaluation/)")
    parser.add_argument("--configs", default="all",
                        help="Configs to evaluate: 'all' or comma list (e.g. '1,3') matching prediction CSV keys")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="IoU threshold for Precision/Recall/F1 'match' (default: 0.5)")
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT,
                        help="Frame rate for frame-accuracy metric (default: 25)")
    args = parser.parse_args()

    rows = [
        evaluate_config(cfg, args.eaf_dir, args.pred_dir, args.out_dir, args.threshold, args.fps)
        for cfg in parse_configs(args.configs)
    ]

    summary_csv = args.out_dir / "evaluation_summary.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config", "label", "gt_tier", "clips", "total_pred", "total_gt", "paired",
        "precision_iou_0_5", "recall_iou_0_5", "f1_iou_0_5", "accuracy_iou_0_5",
        "mean_iou", "median_iou",
        "pct_iou_ge_0_5_over_gt", "pct_iou_ge_0_3_over_gt", "pct_any_overlap_over_gt",
        "pct_zero_overlap_paired",
        "mean_abs_start_offset_s", "mean_abs_end_offset_s", "frame_accuracy",
        "fallback_predictions", "text_match_reference_only",
        "pred_csv", "detail_csv",
    ]
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nForcedAlignment Task 2 evaluation")
    print("=" * 96)
    print(f"{'cfg':>3} {'clips':>5} {'pred':>6} {'gt':>6} {'P@.5':>7} {'R@.5':>7} {'F1@.5':>7} {'mIoU':>7} {'FrmAcc':>7}")
    print("-" * 96)
    for r in rows:
        print(
            f"{r['config']:>3} {r['clips']:>5} {r['total_pred']:>6} {r['total_gt']:>6} "
            f"{r['precision_iou_0_5']*100:>6.1f}% {r['recall_iou_0_5']*100:>6.1f}% "
            f"{r['f1_iou_0_5']*100:>6.1f}% {r['mean_iou']:>7.4f} "
            f"{r['frame_accuracy']*100:>6.1f}%"
        )
    print("=" * 96)
    print(f"[OK] Wrote summary -> {summary_csv}")


if __name__ == "__main__":
    main()
