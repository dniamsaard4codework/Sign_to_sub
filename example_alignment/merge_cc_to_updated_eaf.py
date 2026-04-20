"""
merge_cc_to_updated_eaf.py
--------------------------
คัดลอก tier CC (และ CC_Aligned, Gloss ถ้ามี) จากไฟล์ EAF ต้นฉบับ
แล้วเพิ่มเข้าไปใน 04_updated.eaf เพื่อให้เปรียบเทียบใน ELAN ได้

วิธีใช้:
    python merge_cc_to_updated_eaf.py

ผลลัพธ์:
    segmentation_output/E4s-1_30_50/04_updated.eaf  (แก้ไข in-place)
"""

import xml.etree.ElementTree as ET
import os
import argparse
from pathlib import Path

# --- พาธไฟล์ ---
SOURCE_EAF = Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\การเปรียบเทียบและเรียงลำดับ (11.07 นาที).eaf")
TARGET_EAF = Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\segmentation_output\E4s-1_30_50\04_updated.eaf")

# tier ที่ต้องการคัดลอก (ตามลำดับ)
TIERS_TO_COPY = ["CC", "CC_Aligned", "Gloss", "Gloss Labeling"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge tiers from source EAF into target updated EAF and sync media link."
    )
    parser.add_argument(
        "--source-eaf",
        type=Path,
        default=SOURCE_EAF,
        help="Path to source/original EAF (contains CC tiers).",
    )
    parser.add_argument(
        "--target-eaf",
        type=Path,
        default=TARGET_EAF,
        help="Path to target updated EAF to modify in-place.",
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=TIERS_TO_COPY,
        help="Tier IDs to copy in order.",
    )
    return parser.parse_args()


def _existing_annotation_ids(root: ET.Element) -> set[str]:
    ids: set[str] = set()
    for ann in root.findall(".//ALIGNABLE_ANNOTATION"):
        ann_id = ann.get("ANNOTATION_ID")
        if ann_id:
            ids.add(ann_id)
    for ann in root.findall(".//REF_ANNOTATION"):
        ann_id = ann.get("ANNOTATION_ID")
        if ann_id:
            ids.add(ann_id)
    return ids


def _next_unique(base_id: str, used_ids: set[str]) -> str:
    if base_id not in used_ids:
        used_ids.add(base_id)
        return base_id
    i = 1
    while f"{base_id}_{i}" in used_ids:
        i += 1
    new_id = f"{base_id}_{i}"
    used_ids.add(new_id)
    return new_id


def _sync_media_descriptor(source_path: Path, src_root: ET.Element, target_path: Path, tgt_root: ET.Element) -> bool:
    src_header = src_root.find("HEADER")
    tgt_header = tgt_root.find("HEADER")
    if src_header is None or tgt_header is None:
        return False

    src_md = src_header.find("MEDIA_DESCRIPTOR")
    if src_md is None:
        return False

    src_media_url = src_md.get("MEDIA_URL", "")
    src_rel_media_url = src_md.get("RELATIVE_MEDIA_URL", "")
    mime_type = src_md.get("MIME_TYPE", "video/mp4")

    video_path: Path | None = None

    if src_media_url:
        cand = Path(src_media_url)
        if not cand.is_absolute():
            cand = source_path.parent / cand
        if cand.exists():
            video_path = cand.resolve()

    if video_path is None and src_rel_media_url:
        rel = src_rel_media_url.replace("./", "")
        cand = source_path.parent / rel
        if cand.exists():
            video_path = cand.resolve()

    if video_path is None:
        candidate_names = ["04.mp4", f"{source_path.stem}.mp4"]
        for name in candidate_names:
            cand = source_path.parent / name
            if cand.exists():
                video_path = cand.resolve()
                break

    if video_path is None:
        return False

    tgt_md = tgt_header.find("MEDIA_DESCRIPTOR")
    if tgt_md is None:
        tgt_md = ET.SubElement(tgt_header, "MEDIA_DESCRIPTOR")

    rel_to_target = Path(os.path.relpath(str(video_path), str(target_path.parent.resolve())))
    relative_path = Path("./") / rel_to_target.as_posix()
    tgt_md.set("MEDIA_URL", str(video_path))
    tgt_md.set("RELATIVE_MEDIA_URL", str(relative_path))
    tgt_md.set("MIME_TYPE", mime_type)
    return True


def merge_tiers(source_path: Path, target_path: Path, tier_ids: list[str]) -> None:
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

    # --- อ่านไฟล์ต้นทาง ---
    src_tree = ET.parse(source_path)
    src_root = src_tree.getroot()

    src_time_slots: dict[str, str] = {}
    src_time_order = src_root.find("TIME_ORDER")
    if src_time_order is not None:
        for ts in src_time_order.findall("TIME_SLOT"):
            src_time_slots[ts.get("TIME_SLOT_ID")] = ts.get("TIME_VALUE")

    # --- อ่านไฟล์ปลายทาง ---
    tgt_tree = ET.parse(target_path)
    tgt_root = tgt_tree.getroot()

    tgt_time_order = tgt_root.find("TIME_ORDER")
    if tgt_time_order is None:
        raise ValueError("ไม่พบ TIME_ORDER ใน target EAF")

    media_synced = _sync_media_descriptor(source_path, src_root, target_path, tgt_root)
    if media_synced:
        print("  [+] sync media link สำหรับ ELAN เรียบร้อย")
    else:
        print("  [skip] media link: ไม่พบข้อมูล/ไฟล์วิดีโอที่ใช้ได้จาก source")

    existing_ts_ids = {ts.get("TIME_SLOT_ID") for ts in tgt_time_order.findall("TIME_SLOT")}
    existing_ann_ids = _existing_annotation_ids(tgt_root)

    # --- ตรวจสอบ LINGUISTIC_TYPE ที่ต้องการ ---
    existing_lt_ids = {lt.get("LINGUISTIC_TYPE_ID") for lt in tgt_root.findall("LINGUISTIC_TYPE")}

    # เพิ่ม linguistic type ที่ยังไม่มีจากต้นฉบับ
    for lt in src_root.findall("LINGUISTIC_TYPE"):
        lt_id = lt.get("LINGUISTIC_TYPE_ID")
        if lt_id not in existing_lt_ids:
            tgt_root.append(lt)
            existing_lt_ids.add(lt_id)
            print(f"  [+] เพิ่ม LINGUISTIC_TYPE: {lt_id}")

    # --- ตรวจสอบว่า tier ใดมีอยู่แล้วใน target ---
    existing_tier_ids = {t.get("TIER_ID") for t in tgt_root.findall("TIER")}

    added_tiers = 0

    for tier_id in tier_ids:
        if tier_id in existing_tier_ids:
            print(f"  [skip] tier '{tier_id}' มีอยู่แล้ว")
            continue

        # หา tier ใน source
        src_tier = None
        for t in src_root.findall("TIER"):
            if t.get("TIER_ID") == tier_id:
                src_tier = t
                break

        if src_tier is None:
            print(f"  [skip] ไม่พบ tier '{tier_id}' ในต้นฉบับ")
            continue

        # --- รวบรวม TIME_SLOT_IDs ที่ tier นี้ใช้ ---
        used_ts_ids: set[str] = set()
        for ann in src_tier.findall(".//ALIGNABLE_ANNOTATION"):
            ref1 = ann.get("TIME_SLOT_REF1")
            ref2 = ann.get("TIME_SLOT_REF2")
            if ref1:
                used_ts_ids.add(ref1)
            if ref2:
                used_ts_ids.add(ref2)

        # --- เพิ่ม TIME_SLOT ที่ขาดอยู่ใน target (ใส่ prefix "cc_" เพื่อหลีกเลี่ยง ID ชน) ---
        prefix = f"cc_{tier_id.replace(' ', '_')}_"
        ts_id_map: dict[str, str] = {}  # old_id -> new_id

        for ts_id in used_ts_ids:
            new_ts_id = prefix + ts_id
            if new_ts_id in existing_ts_ids:
                i = 1
                while f"{new_ts_id}_{i}" in existing_ts_ids:
                    i += 1
                new_ts_id = f"{new_ts_id}_{i}"
            ts_id_map[ts_id] = new_ts_id
            existing_ts_ids.add(new_ts_id)
            ts_value = src_time_slots.get(ts_id, "0")
            new_ts = ET.Element("TIME_SLOT", {
                "TIME_SLOT_ID": new_ts_id,
                "TIME_VALUE": ts_value
            })
            tgt_time_order.append(new_ts)

        # --- สร้าง tier ใหม่พร้อม annotation ที่แก้ TIME_SLOT_REF แล้ว ---
        new_tier = ET.Element("TIER", {
            "TIER_ID": src_tier.get("TIER_ID"),
            "LINGUISTIC_TYPE_REF": src_tier.get("LINGUISTIC_TYPE_REF", "default-lt"),
        })
        # คัดลอก attribute อื่น ๆ (เช่น PARTICIPANT, ANNOTATOR)
        for attr in ("PARTICIPANT", "ANNOTATOR", "DEFAULT_LOCALE", "PARENT_REF"):
            val = src_tier.get(attr)
            if val:
                new_tier.set(attr, val)

        ann_count = 0
        ann_id_map: dict[str, str] = {}

        for annotation in src_tier.findall("ANNOTATION"):
            aa = annotation.find("ALIGNABLE_ANNOTATION")
            ra = annotation.find("REF_ANNOTATION")
            if aa is None and ra is None:
                continue

            src_ann = aa if aa is not None else ra
            old_ann_id = src_ann.get("ANNOTATION_ID", f"cc_ann_{ann_count}")
            base_ann_id = f"cc_{tier_id.replace(' ', '_')}_{old_ann_id}"
            ann_id_map[old_ann_id] = _next_unique(base_ann_id, existing_ann_ids)

        for annotation in src_tier.findall("ANNOTATION"):
            aa = annotation.find("ALIGNABLE_ANNOTATION")
            ra = annotation.find("REF_ANNOTATION")
            if aa is None and ra is None:
                continue

            src_ann = aa if aa is not None else ra
            old_ann_id = src_ann.get("ANNOTATION_ID", f"cc_ann_{ann_count}")

            text_el = src_ann.find("ANNOTATION_VALUE")
            text = (text_el.text or "") if text_el is not None else ""

            new_ann = ET.SubElement(new_tier, "ANNOTATION")
            if aa is not None:
                old_ref1 = aa.get("TIME_SLOT_REF1")
                old_ref2 = aa.get("TIME_SLOT_REF2")
                new_aa = ET.SubElement(new_ann, "ALIGNABLE_ANNOTATION", {
                    "ANNOTATION_ID": ann_id_map[old_ann_id],
                    "TIME_SLOT_REF1": ts_id_map.get(old_ref1, old_ref1),
                    "TIME_SLOT_REF2": ts_id_map.get(old_ref2, old_ref2),
                })
                new_val = ET.SubElement(new_aa, "ANNOTATION_VALUE")
                new_val.text = text
            else:
                old_ann_ref = ra.get("ANNOTATION_REF")
                old_prev_ref = ra.get("PREVIOUS_ANNOTATION")
                attrs = {
                    "ANNOTATION_ID": ann_id_map[old_ann_id],
                    "ANNOTATION_REF": ann_id_map.get(old_ann_ref, old_ann_ref),
                }
                if old_prev_ref:
                    attrs["PREVIOUS_ANNOTATION"] = ann_id_map.get(old_prev_ref, old_prev_ref)
                new_ra = ET.SubElement(new_ann, "REF_ANNOTATION", attrs)
                new_val = ET.SubElement(new_ra, "ANNOTATION_VALUE")
                new_val.text = text
            ann_count += 1

        tgt_root.append(new_tier)
        existing_tier_ids.add(tier_id)
        print(f"  [+] เพิ่ม tier '{tier_id}' ({ann_count} annotations)")
        added_tiers += 1

    if added_tiers == 0 and not media_synced:
        print("ไม่มี tier ใดถูกเพิ่ม")
        return

    # --- บันทึกไฟล์ ---
    tgt_tree.write(
        str(target_path),
        encoding="utf-8",
        xml_declaration=True,
    )
    print(f"\n[OK] บันทึกแล้ว -> {target_path}")


if __name__ == "__main__":
    args = _parse_args()
    print(f"Source : {args.source_eaf.name}")
    print(f"Target : {args.target_eaf}")
    print(f"Tiers  : {args.tiers}\n")
    merge_tiers(args.source_eaf, args.target_eaf, args.tiers)
