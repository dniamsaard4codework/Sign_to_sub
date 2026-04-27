"""
add_vtt_tiers_to_eaf.py
-----------------------
อ่าน VTT จากผล experiment หลาย ๆ อัน แล้วเพิ่มเป็น tier แยกใน 04_updated.eaf
เพื่อให้เปิด ELAN แล้วเปรียบเทียบ alignment ทุก experiment ได้ในไฟล์เดียว

tier ที่จะเพิ่ม:
  SUBTITLE_B2           <- aligned_output_with_embedding_tuned/04.vtt
  SUBTITLE_B_MULTI      <- aligned_output_multi_b2/04.vtt
  SUBTITLE_C_MULTI      <- aligned_output_multi_gloss/04.vtt  (best mean offset)
  SUBTITLE_C_MULTI_word <- aligned_output_multi_gloss_word/04.vtt
  SUBTITLE_D_ASL        <- aligned_output_asl_b2/04.vtt
  SUBTITLE_D_ASL_gloss  <- aligned_output_asl_gloss/04.vtt
  SUBTITLE_D_ASL_word   <- aligned_output_asl_gloss_word/04.vtt

วิธีใช้:
    python add_vtt_tiers_to_eaf.py
    python add_vtt_tiers_to_eaf.py --eaf path/to/04_updated.eaf --overwrite
"""
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path

BASE = Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment")
# Base for comparison: original EAF (has CC, CC_Aligned, Gloss, Gloss Labeling)
SOURCE_EAF  = BASE / "การเปรียบเทียบและเรียงลำดับ (11.07 นาที).eaf"
TARGET_EAF  = BASE / "การเปรียบเทียบและเรียงลำดับ (11.07 นาที)_comparison.eaf"

# (tier_id_in_eaf, vtt_path)  — ลำดับจะแสดงในไทม์ไลน์ ELAN ตามนี้
# Layer 1: pre-overlap-fix experiment outputs
# Layer 2: post-overlap-fix variants (NEW — added 2026-04-26)
# Layer 3: Task 2 Gloss Labeling prediction (NEW — added 2026-04-26)
VTT_TIERS = [
    # ── Layer 1: pre-overlap experiment outputs ────────────────────────────
    ("SUBTITLE_B2",                     BASE / "aligned_output_with_embedding_tuned" / "04.vtt"),
    ("SUBTITLE_B_MULTI",                BASE / "aligned_output_multi_b2"             / "04.vtt"),
    ("SUBTITLE_C_MULTI",                BASE / "aligned_output_multi_gloss"          / "04.vtt"),
    ("SUBTITLE_C_MULTI_word",           BASE / "aligned_output_multi_gloss_word"     / "04.vtt"),
    ("SUBTITLE_D_ASL",                  BASE / "aligned_output_asl_b2"               / "04.vtt"),
    ("SUBTITLE_D_ASL_gloss",            BASE / "aligned_output_asl_gloss"            / "04.vtt"),
    ("SUBTITLE_D_ASL_word",             BASE / "aligned_output_asl_gloss_word"       / "04.vtt"),
    # ── Layer 2: post-overlap-fix variants ─────────────────────────────────
    ("SUBTITLE_B2_no_overlap",          BASE / "aligned_output_with_embedding_tuned" / "04_no_overlap.vtt"),
    ("SUBTITLE_B_MULTI_no_overlap",     BASE / "aligned_output_multi_b2"             / "04_no_overlap.vtt"),
    ("SUBTITLE_C_MULTI_no_overlap",     BASE / "aligned_output_multi_gloss"          / "04_no_overlap.vtt"),
    ("SUBTITLE_C_MULTI_word_no_overlap", BASE / "aligned_output_multi_gloss_word"    / "04_no_overlap.vtt"),
    ("SUBTITLE_D_ASL_no_overlap",       BASE / "aligned_output_asl_b2"               / "04_no_overlap.vtt"),
    ("SUBTITLE_D_ASL_gloss_no_overlap", BASE / "aligned_output_asl_gloss"            / "04_no_overlap.vtt"),
    ("SUBTITLE_D_ASL_word_no_overlap",  BASE / "aligned_output_asl_gloss_word"       / "04_no_overlap.vtt"),
    # ── Layer 3: Task 2 Gloss Labeling prediction ──────────────────────────
    ("GLOSS_LABEL_PRED",                BASE / "gloss_labels_pred.vtt"),
]


# ── helpers ──────────────────────────────────────────────────────────────────

def vtt_to_ms(ts: str) -> int:
    h, m, s = ts.strip().split(":")
    return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)


def load_vtt(path: Path) -> list[tuple[int, int, str]]:
    """Return list of (start_ms, end_ms, text)."""
    cues = []
    content = path.read_text(encoding="utf-8")
    for block in content.strip().split("\n\n")[1:]:
        lines = block.strip().split("\n")
        arrow = next((l for l in lines if "-->" in l), None)
        if not arrow:
            continue
        t1s, t2s = arrow.split("-->")
        text_lines = [l for l in lines if "-->" not in l and not l.strip().isdigit()]
        cues.append((vtt_to_ms(t1s), vtt_to_ms(t2s), " ".join(text_lines).strip()))
    return cues


def existing_ts_ids(tgt_root: ET.Element) -> set[str]:
    return {ts.get("TIME_SLOT_ID") for ts in tgt_root.findall(".//TIME_SLOT")}


def existing_ann_ids(tgt_root: ET.Element) -> set[str]:
    ids = set()
    for tag in ("ALIGNABLE_ANNOTATION", "REF_ANNOTATION"):
        for ann in tgt_root.findall(f".//{tag}"):
            if ann.get("ANNOTATION_ID"):
                ids.add(ann.get("ANNOTATION_ID"))
    return ids


def ensure_subtitle_lt(tgt_root: ET.Element) -> None:
    """Add a basic LINGUISTIC_TYPE for subtitle tiers if not present."""
    lt_ids = {lt.get("LINGUISTIC_TYPE_ID") for lt in tgt_root.findall("LINGUISTIC_TYPE")}
    if "subtitle-lt" not in lt_ids:
        lt = ET.Element("LINGUISTIC_TYPE", {
            "GRAPHIC_REFERENCES": "false",
            "LINGUISTIC_TYPE_ID": "subtitle-lt",
            "TIME_ALIGNABLE": "true",
        })
        tgt_root.append(lt)


def add_vtt_as_tier(
    tgt_root: ET.Element,
    tgt_time_order: ET.Element,
    tier_id: str,
    cues: list[tuple[int, int, str]],
    used_ts_ids: set[str],
    used_ann_ids: set[str],
) -> int:
    """Add cues as a new TIER. Returns number of annotations added."""
    new_tier = ET.Element("TIER", {
        "LINGUISTIC_TYPE_REF": "subtitle-lt",
        "TIER_ID": tier_id,
    })

    ann_count = 0
    for i, (t1, t2, text) in enumerate(cues):
        # create two time slots
        ts1_id = f"{tier_id}_ts{i*2}"
        ts2_id = f"{tier_id}_ts{i*2+1}"
        # deduplicate just in case
        for ts_id, val in [(ts1_id, t1), (ts2_id, t2)]:
            while ts_id in used_ts_ids:
                ts_id += "_x"
            used_ts_ids.add(ts_id)
            tgt_time_order.append(ET.Element("TIME_SLOT", {
                "TIME_SLOT_ID": ts_id,
                "TIME_VALUE": str(val),
            }))
        # use the last assigned ids
        all_ts = list(tgt_time_order.findall("TIME_SLOT"))
        ts1_real = all_ts[-2].get("TIME_SLOT_ID")
        ts2_real = all_ts[-1].get("TIME_SLOT_ID")

        ann_id = f"{tier_id}_a{i}"
        while ann_id in used_ann_ids:
            ann_id += "_x"
        used_ann_ids.add(ann_id)

        ann_wrapper = ET.SubElement(new_tier, "ANNOTATION")
        aa = ET.SubElement(ann_wrapper, "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": ann_id,
            "TIME_SLOT_REF1": ts1_real,
            "TIME_SLOT_REF2": ts2_real,
        })
        val_el = ET.SubElement(aa, "ANNOTATION_VALUE")
        val_el.text = text
        ann_count += 1

    tgt_root.append(new_tier)
    return ann_count


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add aligned VTT experiment outputs as comparison tiers into a copy of the original EAF."
    )
    parser.add_argument("--source", type=Path, default=SOURCE_EAF,
                        help="Source EAF to copy as base (original, not modified).")
    parser.add_argument("--output", type=Path, default=TARGET_EAF,
                        help="Output EAF path (copy of source + experiment tiers).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-add tier even if it already exists (removes old first).")
    args = parser.parse_args()

    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

    # Work on a fresh copy of the source so the original is never modified
    import shutil
    shutil.copy2(args.source, args.output)
    print(f"Copied {args.source.name} -> {args.output.name}\n")

    tree = ET.parse(args.output)
    root = tree.getroot()
    args.eaf = args.output

    tgt_time_order = root.find("TIME_ORDER")
    if tgt_time_order is None:
        raise ValueError("TIME_ORDER not found in EAF")

    ensure_subtitle_lt(root)

    used_ts  = existing_ts_ids(root)
    used_ann = existing_ann_ids(root)
    existing_tier_ids = {t.get("TIER_ID") for t in root.findall("TIER")}

    for tier_id, vtt_path in VTT_TIERS:
        if not vtt_path.exists():
            print(f"  [skip] ไม่พบไฟล์: {vtt_path.name}")
            continue

        if tier_id in existing_tier_ids:
            if not args.overwrite:
                print(f"  [skip] tier '{tier_id}' มีอยู่แล้ว (ใช้ --overwrite เพื่อแทนที่)")
                continue
            # remove old tier
            for old in root.findall("TIER"):
                if old.get("TIER_ID") == tier_id:
                    root.remove(old)
                    existing_tier_ids.discard(tier_id)
                    print(f"  [rm]   ลบ tier เก่า '{tier_id}'")
                    break

        cues = load_vtt(vtt_path)
        n = add_vtt_as_tier(root, tgt_time_order, tier_id, cues, used_ts, used_ann)
        existing_tier_ids.add(tier_id)
        print(f"  [+]   เพิ่ม tier '{tier_id}' ({n} cues) <- {vtt_path.name}")

    tree.write(str(args.output), encoding="utf-8", xml_declaration=True)
    print(f"\n[OK] บันทึกแล้ว -> {args.output}")
    print("     เปิด ELAN แล้วดู tier:")
    print("       CC / CC_Aligned / Gloss / Gloss Labeling  <- tier ต้นฉบับ")
    print("       SUBTITLE_B2 / SUBTITLE_B_MULTI / SUBTITLE_C_MULTI / ... <- experiment outputs")


if __name__ == "__main__":
    main()
