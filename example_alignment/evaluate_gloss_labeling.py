"""
evaluate_gloss_labeling.py
--------------------------
Compare predicted gloss labels (Task 2 output) against the manual
"Gloss Labeling" tier in the original EAF (852 ground-truth entries).

Metric strategy (per prediction):
  * Find the GT entry with largest temporal IoU.
  * Record IoU, signed start/end offset (s), text-match flag.

Aggregates:
  * mean / median IoU
  * % IoU >= 0.5
  * % IoU >= 0.3
  * % temporal overlap > 0
  * mean / median signed start offset, end offset
  * exact-text-match accuracy on temporally-overlapping pairs (pred ↔ best GT)

Outputs:
  - example_alignment/evaluation_gloss_labeling.csv  (per-row)
  - stdout summary table

Re-uses: evaluate_all.load_cc_aligned pattern (same EAF parser).
"""
from __future__ import annotations

import argparse
import csv
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFAULT_EAF      = HERE / "การเปรียบเทียบและเรียงลำดับ (11.07 นาที).eaf"
DEFAULT_PRED_CSV = HERE / "gloss_labels_pred.csv"
DEFAULT_OUT_CSV  = HERE / "evaluation_gloss_labeling.csv"


def load_gloss_labeling_tier(eaf_path: Path, tier_id: str = "Gloss Labeling") -> list[dict]:
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    time_slots: dict[str, float] = {}
    for ts in root.find("TIME_ORDER").findall("TIME_SLOT"):
        time_slots[ts.get("TIME_SLOT_ID")] = float(ts.get("TIME_VALUE", 0)) / 1000.0

    target = None
    for tier in root.findall("TIER"):
        if tier.get("TIER_ID") == tier_id:
            target = tier
            break
    if target is None:
        raise ValueError(f"Tier '{tier_id}' not found")

    out: list[dict] = []
    # `Gloss Labeling` is a default-lt tier whose annotations might be REF_ANNOTATION
    # OR ALIGNABLE_ANNOTATION. Walk both.
    for ann in target.findall("ANNOTATION"):
        elem = next(iter(ann), None)
        if elem is None:
            continue
        if elem.tag == "ALIGNABLE_ANNOTATION":
            t1 = time_slots.get(elem.get("TIME_SLOT_REF1"))
            t2 = time_slots.get(elem.get("TIME_SLOT_REF2"))
        elif elem.tag == "REF_ANNOTATION":
            # REF_ANNOTATION inherits from a parent ALIGNABLE_ANNOTATION; we'd
            # need a second pass. Skip if encountered.
            continue
        else:
            continue
        val = elem.find("ANNOTATION_VALUE")
        text = (val.text or "").strip() if val is not None else ""
        if t1 is None or t2 is None or not text:
            continue
        if t2 < t1:
            t1, t2 = t2, t1
        out.append({"start": t1, "end": t2, "text": text})

    out.sort(key=lambda x: x["start"])
    return out


def normalize(t: str) -> str:
    return " ".join(t.split()).lower()


def iou(a_s: float, a_e: float, b_s: float, b_e: float) -> float:
    inter = max(0.0, min(a_e, b_e) - max(a_s, b_s))
    union = max(a_e, b_e) - min(a_s, b_s)
    if union <= 0:
        return 0.0
    return inter / union


def overlap_seconds(a_s: float, a_e: float, b_s: float, b_e: float) -> float:
    return max(0.0, min(a_e, b_e) - max(a_s, b_s))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eaf",      type=Path, default=DEFAULT_EAF)
    p.add_argument("--pred-csv", type=Path, default=DEFAULT_PRED_CSV)
    p.add_argument("--out-csv",  type=Path, default=DEFAULT_OUT_CSV)
    args = p.parse_args()

    print(f"Loading Gloss Labeling tier from {args.eaf.name} ...")
    gt = load_gloss_labeling_tier(args.eaf)
    print(f"  {len(gt)} ground-truth labels")

    print(f"Loading predictions from {args.pred_csv.name} ...")
    preds: list[dict] = []
    with open(args.pred_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds.append({
                "start": float(row["start_s"]),
                "end":   float(row["end_s"]),
                "token": row["gloss_token"],
                "fallback": row.get("fallback", ""),
            })
    print(f"  {len(preds)} predictions")

    # Per-row IoU comparison
    rows: list[dict] = []
    for pr in preds:
        best_iou = 0.0
        best_gt = None
        for g in gt:
            io = iou(pr["start"], pr["end"], g["start"], g["end"])
            if io > best_iou:
                best_iou = io
                best_gt = g
        # If no temporal overlap at all, best_gt is None.
        if best_gt is None:
            rows.append({
                "pred_token": pr["token"],
                "pred_start": pr["start"],
                "pred_end":   pr["end"],
                "gt_token":   "",
                "gt_start":   "",
                "gt_end":     "",
                "iou":        0.0,
                "start_off_s": "",
                "end_off_s":   "",
                "text_match": False,
                "fallback":   pr["fallback"],
            })
            continue
        rows.append({
            "pred_token": pr["token"],
            "pred_start": round(pr["start"], 3),
            "pred_end":   round(pr["end"],   3),
            "gt_token":   best_gt["text"],
            "gt_start":   round(best_gt["start"], 3),
            "gt_end":     round(best_gt["end"],   3),
            "iou":        round(best_iou, 4),
            "start_off_s": round(pr["start"] - best_gt["start"], 3),
            "end_off_s":   round(pr["end"]   - best_gt["end"],   3),
            "text_match": normalize(pr["token"]) == normalize(best_gt["text"]),
            "fallback":   pr["fallback"],
        })

    # Aggregates
    ious = [r["iou"] for r in rows]
    n = len(ious) or 1
    mean_iou   = statistics.mean(ious) if ious else 0.0
    median_iou = statistics.median(ious) if ious else 0.0
    pct_iou_50 = sum(1 for i in ious if i >= 0.5) / n * 100
    pct_iou_30 = sum(1 for i in ious if i >= 0.3) / n * 100
    pct_overlap_pos = sum(1 for r in rows if r["iou"] > 0) / n * 100

    overlapping = [r for r in rows if r["iou"] > 0 and r["start_off_s"] != ""]
    if overlapping:
        starts = [r["start_off_s"] for r in overlapping]
        ends   = [r["end_off_s"]   for r in overlapping]
        mean_start = statistics.mean(starts)
        mean_end   = statistics.mean(ends)
        med_start  = statistics.median(starts)
        med_end    = statistics.median(ends)
    else:
        mean_start = mean_end = med_start = med_end = 0.0

    text_matches = sum(1 for r in overlapping if r["text_match"])
    text_match_pct = text_matches / max(len(overlapping), 1) * 100

    fallback_count = sum(1 for r in rows if r["fallback"])

    # Print summary
    print()
    print("=" * 80)
    print("Task 2 — Gloss Labeling evaluation")
    print("=" * 80)
    print(f"  predictions:         {len(rows)}")
    print(f"  GT labels:           {len(gt)}")
    print(f"  fallback_uniform:    {fallback_count}")
    print()
    print(f"  mean IoU:            {mean_iou:.4f}")
    print(f"  median IoU:          {median_iou:.4f}")
    print(f"  % IoU >= 0.5:        {pct_iou_50:.1f}%")
    print(f"  % IoU >= 0.3:        {pct_iou_30:.1f}%")
    print(f"  % any temporal overlap: {pct_overlap_pos:.1f}%")
    print()
    print(f"  mean start offset:   {mean_start:+.3f}s   (median: {med_start:+.3f}s)")
    print(f"  mean end offset:     {mean_end:+.3f}s   (median: {med_end:+.3f}s)")
    print(f"  exact text match (overlapping pairs): {text_matches}/{len(overlapping)} = {text_match_pct:.1f}%")
    print("=" * 80)

    # Write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pred_token", "pred_start", "pred_end",
        "gt_token", "gt_start", "gt_end",
        "iou", "start_off_s", "end_off_s",
        "text_match", "fallback",
    ]
    with open(args.out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Wrote {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
