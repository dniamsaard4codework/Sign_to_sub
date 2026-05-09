"""
add_best_to_eaf.py
------------------
Build a comparison EAF with only the BEST-CASE tiers for Task 1 and Task 2.

  Task 1 best : C_MULTI (Multilingual + Gloss text)
                - SUBTITLE_C_MULTI            <- aligned_output_multi_gloss/04.vtt
                - SUBTITLE_C_MULTI_no_overlap <- aligned_output_multi_gloss/04_no_overlap.vtt
  Task 2      : GLOSS_LABEL_PRED              <- gloss_labels_pred.vtt

Source : Test.eaf  (CC, CC_Input, CC_Aligned, Gloss, Gloss_Input, Gloss Labeling)
Output : Test_best.eaf

Usage:
    python add_best_to_eaf.py
    python add_best_to_eaf.py --overwrite
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from add_vtt_tiers_to_eaf import (
    ensure_subtitle_lt,
    existing_ann_ids,
    existing_ts_ids,
    add_vtt_as_tier,
    load_vtt,
)

BASE = Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment")
SOURCE_EAF = BASE / "Test.eaf"
TARGET_EAF = BASE / "Test_best.eaf"

VTT_TIERS = [
    ("SUBTITLE_C_MULTI",            BASE / "aligned_output_multi_gloss" / "04.vtt"),
    ("SUBTITLE_C_MULTI_no_overlap", BASE / "aligned_output_multi_gloss" / "04_no_overlap.vtt"),
    ("GLOSS_LABEL_PRED",            BASE / "gloss_labels_pred.vtt"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Test_best.eaf with best-case Task 1 (C_MULTI) and Task 2 tiers."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_EAF)
    parser.add_argument("--output", type=Path, default=TARGET_EAF)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-add tier even if it already exists.")
    args = parser.parse_args()

    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

    shutil.copy2(args.source, args.output)
    print(f"Copied {args.source.name} -> {args.output.name}\n")

    tree = ET.parse(args.output)
    root = tree.getroot()

    tgt_time_order = root.find("TIME_ORDER")
    if tgt_time_order is None:
        raise ValueError("TIME_ORDER not found in EAF")

    ensure_subtitle_lt(root)

    used_ts = existing_ts_ids(root)
    used_ann = existing_ann_ids(root)
    existing_tier_ids = {t.get("TIER_ID") for t in root.findall("TIER")}

    for tier_id, vtt_path in VTT_TIERS:
        if not vtt_path.exists():
            print(f"  [skip] missing file: {vtt_path}")
            continue

        if tier_id in existing_tier_ids:
            if not args.overwrite:
                print(f"  [skip] tier '{tier_id}' already exists (use --overwrite)")
                continue
            for old in root.findall("TIER"):
                if old.get("TIER_ID") == tier_id:
                    root.remove(old)
                    existing_tier_ids.discard(tier_id)
                    print(f"  [rm]   removed old tier '{tier_id}'")
                    break

        cues = load_vtt(vtt_path)
        n = add_vtt_as_tier(root, tgt_time_order, tier_id, cues, used_ts, used_ann)
        existing_tier_ids.add(tier_id)
        print(f"  [+]    added tier '{tier_id}' ({n} cues) <- {vtt_path.name}")

    tree.write(str(args.output), encoding="utf-8", xml_declaration=True)
    print(f"\n[OK] saved -> {args.output}")
    print("     ELAN tiers in this file:")
    print("       Original  : CC, CC_Input, CC_Aligned, Gloss, Gloss_Input, Gloss Labeling")
    print("       Task 1    : SUBTITLE_C_MULTI, SUBTITLE_C_MULTI_no_overlap")
    print("       Task 2    : GLOSS_LABEL_PRED")


if __name__ == "__main__":
    main()
