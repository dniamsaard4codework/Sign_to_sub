"""
make_task2_comparison_eaf.py
----------------------------
Build a focused Task 2 comparison EAF — only the tiers needed to inspect
gloss-token alignment side-by-side in ELAN.

Tiers added (preserves originals from Test.eaf as the base):
  Gloss                              <- input sentences (from Test.eaf)
  Gloss Labeling                     <- ground truth (from Test.eaf)
  GLOSS_LABEL_PRED__Gloss            <- whole-video baseline (--tier Gloss)
  GLOSS_LABEL_PRED__Gloss_Input      <- whole-video alt    (--tier Gloss_Input)
  GLOSS_LABEL_PRED__per_sentence     <- per-sentence pipeline (Progress_16052026)

Sources (VTT files):
  ablation/gloss_labels_pred__Gloss.vtt
  ablation/gloss_labels_pred__Gloss_Input.vtt
  ablation_per_sentence/gloss_labels_pred__per_sentence.vtt

Output:
  example_alignment/Test_task2_comparison.eaf

Usage:
  python example_alignment\\make_task2_comparison_eaf.py
  python example_alignment\\make_task2_comparison_eaf.py --overwrite
  python example_alignment\\make_task2_comparison_eaf.py --output custom.eaf

Reuses the parser / writer helpers in add_vtt_tiers_to_eaf.py to keep one
authoritative implementation of VTT-> tier injection.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from add_vtt_tiers_to_eaf import (  # type: ignore
    add_vtt_as_tier,
    ensure_subtitle_lt,
    existing_ann_ids,
    existing_ts_ids,
    load_vtt,
)

# Defaults
SOURCE_EAF = HERE / "Test.eaf"
TARGET_EAF = HERE / "Test_task2_comparison.eaf"

# Tiers (order = ELAN display order; first tier shown topmost)
TIER_VTTS: list[tuple[str, Path]] = [
    ("GLOSS_LABEL_PRED__Gloss",
     HERE / "ablation" / "gloss_labels_pred__Gloss.vtt"),
    ("GLOSS_LABEL_PRED__Gloss_Input",
     HERE / "ablation" / "gloss_labels_pred__Gloss_Input.vtt"),
    ("GLOSS_LABEL_PRED__per_sentence",
     HERE / "ablation_per_sentence" / "gloss_labels_pred__per_sentence.vtt"),
]

# Original tiers we want to KEEP in the comparison EAF; everything else is dropped
KEEP_ORIGINAL_TIERS = {"Gloss", "Gloss Labeling"}


def strip_unwanted_tiers(root: ET.Element, keep: set[str]) -> None:
    """Remove TIER elements whose TIER_ID is not in `keep`."""
    removed = 0
    for tier in list(root.findall("TIER")):
        tid = tier.get("TIER_ID")
        if tid not in keep:
            root.remove(tier)
            removed += 1
    if removed:
        print(f"  [-] removed {removed} unrelated original tiers (kept {sorted(keep)})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a focused Task 2 comparison EAF (Gloss + GT + 3 prediction tiers).",
    )
    ap.add_argument("--source", type=Path, default=SOURCE_EAF,
                    help="Base EAF to copy (default: Test.eaf)")
    ap.add_argument("--output", type=Path, default=TARGET_EAF,
                    help="Output EAF path")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-add tier even if it already exists in the source.")
    ap.add_argument("--keep-all-originals", action="store_true",
                    help="Keep all original tiers from the source EAF (do not strip).")
    args = ap.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source EAF not found: {args.source}")

    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

    shutil.copy2(args.source, args.output)
    print(f"Copied {args.source.name} -> {args.output.name}")

    tree = ET.parse(args.output)
    root = tree.getroot()
    time_order = root.find("TIME_ORDER")
    if time_order is None:
        raise SystemExit("TIME_ORDER missing from EAF")

    if not args.keep_all_originals:
        strip_unwanted_tiers(root, KEEP_ORIGINAL_TIERS)

    ensure_subtitle_lt(root)

    used_ts = existing_ts_ids(root)
    used_ann = existing_ann_ids(root)
    existing_tier_ids = {t.get("TIER_ID") for t in root.findall("TIER")}

    n_added = 0
    n_skip = 0
    for tier_id, vtt_path in TIER_VTTS:
        if not vtt_path.exists():
            print(f"  [skip] not found: {vtt_path.relative_to(HERE.parent)}")
            n_skip += 1
            continue

        if tier_id in existing_tier_ids:
            if not args.overwrite:
                print(f"  [skip] tier '{tier_id}' already exists (use --overwrite)")
                continue
            for old in root.findall("TIER"):
                if old.get("TIER_ID") == tier_id:
                    root.remove(old)
                    existing_tier_ids.discard(tier_id)
                    print(f"  [-] removed old tier '{tier_id}'")
                    break

        cues = load_vtt(vtt_path)
        n = add_vtt_as_tier(root, time_order, tier_id, cues, used_ts, used_ann)
        existing_tier_ids.add(tier_id)
        n_added += 1
        print(f"  [+] tier '{tier_id}' ({n} cues) <- {vtt_path.relative_to(HERE.parent)}")

    tree.write(str(args.output), encoding="utf-8", xml_declaration=True)

    print()
    print(f"[OK] wrote {args.output}")
    print(f"     added {n_added} prediction tier(s), skipped {n_skip}")
    print("     open in ELAN to view tiers in this order:")
    print("       Gloss (input)")
    print("       Gloss Labeling (GT)")
    print("       GLOSS_LABEL_PRED__Gloss            <- whole-video best")
    print("       GLOSS_LABEL_PRED__Gloss_Input      <- whole-video alt")
    print("       GLOSS_LABEL_PRED__per_sentence     <- per-sentence pipeline")


if __name__ == "__main__":
    main()
