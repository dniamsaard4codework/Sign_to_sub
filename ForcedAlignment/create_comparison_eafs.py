"""
create_comparison_eafs.py
-------------------------
Create ELAN files for visual ground-truth vs prediction review.

Each output EAF starts from the corrected original EAF so the source
annotations and media link are preserved. It then appends, for each selected
configuration, three comparison tiers:

    cfgN_GT_*    - the ground-truth intervals used by evaluation
    cfgN_PRED_*  - the predicted intervals from output/predictions
    cfgN_EVAL_*  - per-token IoU / offset labels spanning the compared region
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
DEFAULT_OUT_DIR = FA_DIR / "output" / "comparison_eafs"

IOU_THRESHOLD_DEFAULT = 0.5
SIL_TOKENS = {"sil", "sil1", "sil2"}
COMPARISON_LT = "fa-comparison-lt"


@dataclass(frozen=True)
class ComparisonConfig:
    key: str
    label: str
    gt_tier: str
    pred_csv: str
    drop_gt_sil: bool = False

    @property
    def short_label(self) -> str:
        prefix = f"config{self.key}_"
        suffix = self.label
        if suffix.startswith(prefix):
            suffix = suffix[len(prefix):]
        if suffix.endswith("_pred"):
            suffix = suffix[:-len("_pred")]
        return suffix

    @property
    def gt_compare_tier(self) -> str:
        tier = f"cfg{self.key}_GT_{self.gt_tier}"
        if self.drop_gt_sil:
            tier += "_nosil"
        return tier

    @property
    def pred_compare_tier(self) -> str:
        return f"cfg{self.key}_PRED_{self.short_label}"

    @property
    def eval_compare_tier(self) -> str:
        return f"cfg{self.key}_EVAL_{self.short_label}"


CONFIGS: dict[str, ComparisonConfig] = {
    "1": ComparisonConfig(
        key="1",
        label="config1_CC_Aligned_pred",
        gt_tier="CC_Aligned",
        pred_csv="config1_CC_Aligned_pred.csv",
        drop_gt_sil=True,
    ),
    "2": ComparisonConfig(
        key="2",
        label="config2_CC_Aligned_silmodel_pred",
        gt_tier="CC_Aligned",
        pred_csv="config2_CC_Aligned_silmodel_pred.csv",
    ),
    "3": ComparisonConfig(
        key="3",
        label="config3_Gloss_Labeling_pred",
        gt_tier="Gloss_Labeling",
        pred_csv="config3_Gloss_Labeling_pred.csv",
    ),
    "4": ComparisonConfig(
        key="4",
        label="config4_Gloss_Labeling1_pred",
        gt_tier="Gloss_Labeling1",
        pred_csv="config4_Gloss_Labeling1_pred.csv",
    ),
    "5": ComparisonConfig(
        key="5",
        label="config5_Gloss_Labeling2_pred",
        gt_tier="Gloss_Labeling2",
        pred_csv="config5_Gloss_Labeling2_pred.csv",
    ),
}


def numeric_sort_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def clip_sort_key(clip_id: str) -> tuple[int, int | str]:
    return (0, int(clip_id)) if clip_id.isdigit() else (1, clip_id)


def parse_configs(spec: str) -> list[ComparisonConfig]:
    if spec.lower() == "all":
        keys = ["1", "2", "3", "4", "5"]
    else:
        keys = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [k for k in keys if k not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown configs: {', '.join(unknown)}")
    return [CONFIGS[k] for k in keys]


def parse_id_spec(spec: str | None) -> set[str] | None:
    if not spec:
        return None
    out: set[str] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            for value in range(int(start), int(end) + 1):
                out.add(str(value))
        else:
            out.add(str(int(chunk)) if chunk.isdigit() else chunk)
    return out or None


def normalize_token(text: str) -> str:
    return " ".join(text.split()).lower()


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def load_predictions(pred_csv: Path) -> dict[str, list[dict]]:
    if not pred_csv.exists():
        raise FileNotFoundError(pred_csv)
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


def load_time_slots(root: ET.Element) -> dict[str, float]:
    return {
        ts.get("TIME_SLOT_ID"): float(ts.get("TIME_VALUE", 0)) / 1000.0
        for ts in root.findall(".//TIME_SLOT")
        if ts.get("TIME_SLOT_ID") is not None
    }


def load_tier_entries(root: ET.Element, tier_id: str, drop_sil: bool = False) -> list[dict]:
    ts_map = load_time_slots(root)
    target = None
    for tier in root.findall("TIER"):
        if tier.get("TIER_ID") == tier_id:
            target = tier
            break
    if target is None:
        raise ValueError(f"Tier {tier_id!r} not found")

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


def next_time_slot_index(root: ET.Element) -> int:
    max_idx = 0
    for ts in root.findall(".//TIME_SLOT"):
        ts_id = ts.get("TIME_SLOT_ID", "")
        digits = "".join(ch for ch in ts_id if ch.isdigit())
        if digits:
            max_idx = max(max_idx, int(digits))
    return max_idx + 1


def next_annotation_index(root: ET.Element) -> int:
    max_idx = 0
    for elem in root.findall(".//ALIGNABLE_ANNOTATION"):
        ann_id = elem.get("ANNOTATION_ID", "")
        digits = "".join(ch for ch in ann_id if ch.isdigit())
        if digits:
            max_idx = max(max_idx, int(digits))
    return max_idx + 1


def ensure_comparison_linguistic_type(root: ET.Element) -> None:
    for lt in root.findall("LINGUISTIC_TYPE"):
        if lt.get("LINGUISTIC_TYPE_ID") == COMPARISON_LT:
            return

    new_lt = ET.Element("LINGUISTIC_TYPE", {
        "GRAPHIC_REFERENCES": "false",
        "LINGUISTIC_TYPE_ID": COMPARISON_LT,
        "TIME_ALIGNABLE": "true",
    })

    children = list(root)
    last_lt_idx = None
    for idx, child in enumerate(children):
        if child.tag == "LINGUISTIC_TYPE":
            last_lt_idx = idx
    if last_lt_idx is None:
        root.append(new_lt)
    else:
        root.insert(last_lt_idx + 1, new_lt)


def comparison_tier_ids(configs: list[ComparisonConfig]) -> set[str]:
    ids: set[str] = set()
    for cfg in configs:
        ids.update({
            cfg.gt_compare_tier,
            cfg.pred_compare_tier,
            cfg.eval_compare_tier,
        })
    return ids


def remove_existing_comparison_tiers(root: ET.Element, configs: list[ComparisonConfig]) -> None:
    tier_ids = comparison_tier_ids(configs)
    for tier in list(root.findall("TIER")):
        if tier.get("TIER_ID") in tier_ids:
            root.remove(tier)


def append_time_slot(time_order: ET.Element, ts_id: str, seconds: float) -> None:
    time_order.append(ET.Element("TIME_SLOT", {
        "TIME_SLOT_ID": ts_id,
        "TIME_VALUE": str(max(0, int(round(seconds * 1000)))),
    }))


def make_tier(
    tier_id: str,
    entries: list[dict],
    time_order: ET.Element,
    state: dict[str, int],
) -> ET.Element:
    tier = ET.Element("TIER", {
        "LINGUISTIC_TYPE_REF": COMPARISON_LT,
        "TIER_ID": tier_id,
    })
    for entry in entries:
        start = float(entry["start"])
        end = float(entry["end"])
        if end < start:
            start, end = end, start
        if end == start:
            end += 0.001

        ts1 = f"fa_cmp_ts{state['ts_idx']}"
        state["ts_idx"] += 1
        ts2 = f"fa_cmp_ts{state['ts_idx']}"
        state["ts_idx"] += 1
        append_time_slot(time_order, ts1, start)
        append_time_slot(time_order, ts2, end)

        ann = ET.SubElement(tier, "ANNOTATION")
        aa = ET.SubElement(ann, "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": f"fa_cmp_a{state['ann_idx']}",
            "TIME_SLOT_REF1": ts1,
            "TIME_SLOT_REF2": ts2,
        })
        state["ann_idx"] += 1
        ET.SubElement(aa, "ANNOTATION_VALUE").text = str(entry.get("token", ""))
    return tier


def evaluation_entries(
    gt: list[dict],
    pred: list[dict],
    threshold: float,
) -> tuple[list[dict], dict]:
    entries: list[dict] = []
    ious: list[float] = []
    hits = 0
    paired = min(len(gt), len(pred))

    for idx in range(paired):
        g = gt[idx]
        p = pred[idx]
        iou = interval_iou(p["start"], p["end"], g["start"], g["end"])
        ious.append(iou)
        if iou >= threshold:
            hits += 1
        start_offset = p["start"] - g["start"]
        end_offset = p["end"] - g["end"]
        status = "HIT" if iou >= threshold else "MISS"
        text_status = "TEXT_OK" if normalize_token(p["token"]) == normalize_token(g["token"]) else "TEXT_DIFF"
        fallback = f" fallback={p['fallback']}" if p.get("fallback") else ""
        entries.append({
            "start": min(g["start"], p["start"]),
            "end": max(g["end"], p["end"]),
            "token": (
                f"idx={idx} {status} iou={iou:.3f} {text_status} "
                f"pred={p['token']} gt={g['token']} "
                f"ds={start_offset:+.3f}s de={end_offset:+.3f}s{fallback}"
            ),
        })

    for idx in range(paired, len(pred)):
        p = pred[idx]
        entries.append({
            "start": p["start"],
            "end": p["end"],
            "token": f"idx={idx} UNPAIRED_PRED pred={p['token']} fallback={p.get('fallback', '')}".rstrip(),
        })

    for idx in range(paired, len(gt)):
        g = gt[idx]
        entries.append({
            "start": g["start"],
            "end": g["end"],
            "token": f"idx={idx} UNPAIRED_GT gt={g['token']}",
        })

    summary = {
        "gt_count": len(gt),
        "pred_count": len(pred),
        "paired": paired,
        "hit_iou_0_5": hits,
        "mean_iou": statistics.mean(ious) if ious else 0.0,
    }
    return entries, summary


def insert_comparison_tiers(root: ET.Element, tiers: list[ET.Element]) -> None:
    children = list(root)
    insert_at = len(children)
    for idx, child in enumerate(children):
        if child.tag == "LINGUISTIC_TYPE":
            insert_at = idx
            break
    for offset, tier in enumerate(tiers):
        root.insert(insert_at + offset, tier)


def write_comparison_eaf(
    eaf_path: Path,
    out_path: Path,
    configs: list[ComparisonConfig],
    predictions_by_config: dict[str, dict[str, list[dict]]],
    threshold: float,
) -> list[dict]:
    tree = ET.parse(eaf_path)
    root = tree.getroot()
    time_order = root.find("TIME_ORDER")
    if time_order is None:
        raise ValueError(f"TIME_ORDER missing in {eaf_path}")

    ensure_comparison_linguistic_type(root)
    remove_existing_comparison_tiers(root, configs)

    state = {
        "ts_idx": next_time_slot_index(root),
        "ann_idx": next_annotation_index(root),
    }
    new_tiers: list[ET.Element] = []
    rows: list[dict] = []

    for cfg in configs:
        gt = load_tier_entries(root, cfg.gt_tier, drop_sil=cfg.drop_gt_sil)
        pred = predictions_by_config[cfg.key].get(eaf_path.stem, [])
        eval_rows, summary = evaluation_entries(gt, pred, threshold)

        new_tiers.extend([
            make_tier(cfg.gt_compare_tier, gt, time_order, state),
            make_tier(cfg.pred_compare_tier, pred, time_order, state),
            make_tier(cfg.eval_compare_tier, eval_rows, time_order, state),
        ])
        rows.append({
            "clip_id": eaf_path.stem,
            "config": cfg.key,
            "gt_tier": cfg.gt_tier,
            "gt_compare_tier": cfg.gt_compare_tier,
            "pred_compare_tier": cfg.pred_compare_tier,
            "eval_compare_tier": cfg.eval_compare_tier,
            **summary,
        })

    insert_comparison_tiers(root, new_tiers)
    try:
        ET.indent(tree, space="    ")
    except AttributeError:
        pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create GT-vs-prediction comparison EAFs.")
    parser.add_argument("--eaf-dir", type=Path, default=DEFAULT_EAF_DIR)
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--configs", default="all")
    parser.add_argument("--only-ids", default=None)
    parser.add_argument("--threshold", type=float, default=IOU_THRESHOLD_DEFAULT)
    args = parser.parse_args()

    configs = parse_configs(args.configs)
    selected_ids = parse_id_spec(args.only_ids)

    predictions_by_config = {
        cfg.key: load_predictions(args.pred_dir / cfg.pred_csv)
        for cfg in configs
    }

    eaf_paths = sorted(args.eaf_dir.glob("*.eaf"), key=numeric_sort_key)
    if selected_ids is not None:
        eaf_paths = [p for p in eaf_paths if p.stem in selected_ids]
    if not eaf_paths:
        raise FileNotFoundError(f"No EAF files found in {args.eaf_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict] = []
    for eaf_path in eaf_paths:
        out_path = args.out_dir / eaf_path.name
        index_rows.extend(write_comparison_eaf(
            eaf_path=eaf_path,
            out_path=out_path,
            configs=configs,
            predictions_by_config=predictions_by_config,
            threshold=args.threshold,
        ))

    index_rows.sort(key=lambda x: (clip_sort_key(x["clip_id"]), int(x["config"])))
    index_csv = args.out_dir / "comparison_index.csv"
    fieldnames = [
        "clip_id", "config", "gt_tier", "gt_compare_tier",
        "pred_compare_tier", "eval_compare_tier",
        "gt_count", "pred_count", "paired", "hit_iou_0_5", "mean_iou",
    ]
    with open(index_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"[OK] Wrote {len(eaf_paths)} comparison EAFs -> {args.out_dir}")
    print(f"[OK] Wrote index -> {index_csv}")


if __name__ == "__main__":
    main()
