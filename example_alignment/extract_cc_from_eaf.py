"""
extract_cc_from_eaf.py
----------------------
สคริปต์สำหรับแยก CC tier (คำบรรยาย) จากไฟล์ EAF แล้วบันทึกเป็น WebVTT (.vtt)

วิธีใช้:
    python extract_cc_from_eaf.py <input.eaf> [output.vtt] [--tier CC]

ตัวอย่าง:
    python extract_cc_from_eaf.py "การเปรียบเทียบและเรียงลำดับ (11.07 นาที).eaf" 04.vtt
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def seconds_to_vtt_timestamp(seconds: float) -> str:
    """แปลงวินาที (float) เป็น VTT timestamp รูปแบบ HH:MM:SS.mmm"""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def extract_cc_to_vtt(eaf_path: str, output_path: str, tier_id: str = "CC") -> None:
    """
    อ่าน tier ที่ระบุจากไฟล์ EAF แล้วบันทึกเป็น .vtt

    Parameters
    ----------
    eaf_path  : พาธไฟล์ .eaf
    output_path: พาธไฟล์ .vtt ที่จะบันทึก
    tier_id   : ชื่อ tier ใน ELAN (ค่าเริ่มต้น: "CC")
    """
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    # 1. สร้าง dict ของ TIME_SLOT_ID -> เวลา (วินาที)
    time_order = root.find("TIME_ORDER")
    if time_order is None:
        raise ValueError("ไม่พบ TIME_ORDER ในไฟล์ EAF")

    time_slots: dict[str, float] = {}
    for ts in time_order.findall("TIME_SLOT"):
        ts_id = ts.get("TIME_SLOT_ID")
        ts_val = ts.get("TIME_VALUE")
        if ts_id and ts_val is not None:
            time_slots[ts_id] = float(ts_val) / 1000.0  # ms -> วินาที

    # 2. ค้นหา tier ที่ต้องการ
    target_tier = None
    for tier in root.findall("TIER"):
        if tier.get("TIER_ID") == tier_id:
            target_tier = tier
            break

    if target_tier is None:
        available = [t.get("TIER_ID") for t in root.findall("TIER")]
        raise ValueError(
            f"ไม่พบ tier '{tier_id}' ในไฟล์ EAF\n"
            f"Tier ที่มีอยู่: {available}"
        )

    # 3. แยก annotation ออกมาพร้อมเวลา
    cues: list[dict] = []
    for annotation in target_tier.findall("ANNOTATION"):
        elem = next(iter(annotation), None)
        if elem is None:
            continue
        ts1 = elem.get("TIME_SLOT_REF1")
        ts2 = elem.get("TIME_SLOT_REF2")
        text_elem = elem.find("ANNOTATION_VALUE")
        text = (text_elem.text or "").strip() if text_elem is not None else ""

        start = time_slots.get(ts1)
        end = time_slots.get(ts2)
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start

        cues.append({"start": start, "end": end, "text": text})

    # เรียงตามเวลาเริ่ม
    cues.sort(key=lambda c: c["start"])

    # 4. เขียนไฟล์ VTT
    lines = ["WEBVTT", ""]
    for i, cue in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(
            f"{seconds_to_vtt_timestamp(cue['start'])} --> "
            f"{seconds_to_vtt_timestamp(cue['end'])}"
        )
        lines.append(cue["text"])
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Extracted {len(cues)} cues from tier '{tier_id}'")
    print(f"[OK] Saved VTT to: {output_path}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    eaf_file = args[0]
    out_file = args[1] if len(args) > 1 else str(Path(eaf_file).stem) + ".vtt"

    # ดึง --tier ถ้ามี
    tier = "CC"
    if "--tier" in args:
        idx = args.index("--tier")
        tier = args[idx + 1]

    extract_cc_to_vtt(eaf_file, out_file, tier_id=tier)
