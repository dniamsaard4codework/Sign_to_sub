"""
make_gloss_input_tier.py
------------------------
สร้าง tier ใหม่ชื่อ "Gloss_Input" ใน Test.eaf
โดยใช้ timestamp จาก CC_Input และ text จาก Gloss
(จับคู่ด้วย maximum overlap ระหว่างสองกลุ่ม annotation)

วิธีใช้:
    python make_gloss_input_tier.py
    python make_gloss_input_tier.py --eaf path/to/file.eaf --out path/to/out.eaf
"""
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path

BASE = Path(__file__).parent
DEFAULT_EAF = BASE / "Test.eaf"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eaf", type=Path, default=DEFAULT_EAF)
    p.add_argument("--out", type=Path, default=None,
                   help="Output path. Defaults to overwriting --eaf.")
    p.add_argument("--overwrite-tier", action="store_true",
                   help="Remove existing Gloss_Input tier before adding.")
    return p.parse_args()


def _ts_map(root: ET.Element) -> dict[str, int]:
    return {ts.get("TIME_SLOT_ID"): int(ts.get("TIME_VALUE"))
            for ts in root.findall(".//TIME_SLOT")}


def _get_alignable_annotations(tier: ET.Element, ts_map: dict[str, int]):
    """Return list of (start_ms, end_ms, annotation_value)."""
    result = []
    for ann in tier.findall("ANNOTATION"):
        a = ann.find("ALIGNABLE_ANNOTATION")
        if a is None:
            continue
        s = ts_map.get(a.get("TIME_SLOT_REF1"), 0)
        e = ts_map.get(a.get("TIME_SLOT_REF2"), 0)
        v = (a.find("ANNOTATION_VALUE").text or "").strip()
        result.append((s, e, v))
    return result


def _overlap(s1, e1, s2, e2) -> int:
    return max(0, min(e1, e2) - max(s1, s2))


def _best_gloss_for_cc(cc_start, cc_end, gloss_anns):
    """Return gloss text with maximum overlap with [cc_start, cc_end]."""
    best_text = ""
    best_ov = -1
    for gs, ge, gv in gloss_anns:
        ov = _overlap(cc_start, cc_end, gs, ge)
        if ov > best_ov:
            best_ov = ov
            best_text = gv
    return best_text


def _max_ann_num(root: ET.Element) -> int:
    nums = []
    for tag in ("ALIGNABLE_ANNOTATION", "REF_ANNOTATION"):
        for a in root.findall(f".//{tag}"):
            aid = a.get("ANNOTATION_ID", "")
            if aid.startswith("a") and aid[1:].isdigit():
                nums.append(int(aid[1:]))
    return max(nums) if nums else 0


def _add_gloss_input_tier(root: ET.Element, ts_map: dict[str, int]) -> None:
    # Remove existing Gloss_Input tier if present
    for t in root.findall(".//TIER[@TIER_ID='Gloss_Input']"):
        root.remove(t)

    cc_tier = root.find(".//TIER[@TIER_ID='CC_Input']")
    gloss_tier = root.find(".//TIER[@TIER_ID='Gloss']")
    if cc_tier is None:
        raise ValueError("CC_Input tier not found")
    if gloss_tier is None:
        raise ValueError("Gloss tier not found")

    gloss_anns = _get_alignable_annotations(gloss_tier, ts_map)

    # Build new tier element
    new_tier = ET.Element("TIER", {
        "LINGUISTIC_TYPE_REF": "imported-sub",
        "TIER_ID": "Gloss_Input",
    })

    next_id = _max_ann_num(root) + 1

    for cc_ann in cc_tier.findall("ANNOTATION"):
        a = cc_ann.find("ALIGNABLE_ANNOTATION")
        if a is None:
            continue
        ts1 = a.get("TIME_SLOT_REF1")
        ts2 = a.get("TIME_SLOT_REF2")
        cc_start = ts_map.get(ts1, 0)
        cc_end = ts_map.get(ts2, 0)

        gloss_text = _best_gloss_for_cc(cc_start, cc_end, gloss_anns)

        ann_elem = ET.SubElement(new_tier, "ANNOTATION")
        align_ann = ET.SubElement(ann_elem, "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": f"a{next_id}",
            "TIME_SLOT_REF1": ts1,
            "TIME_SLOT_REF2": ts2,
        })
        val = ET.SubElement(align_ann, "ANNOTATION_VALUE")
        val.text = gloss_text
        next_id += 1

    # Insert Gloss_Input tier after Gloss tier
    children = list(root)
    gloss_idx = next((i for i, c in enumerate(children) if c.tag == "TIER" and c.get("TIER_ID") == "Gloss"), None)
    if gloss_idx is not None:
        root.insert(gloss_idx + 1, new_tier)
    else:
        root.append(new_tier)

    # Update lastUsedAnnotationId in HEADER
    header = root.find("HEADER")
    if header is not None:
        for prop in header.findall("PROPERTY"):
            if prop.get("NAME") == "lastUsedAnnotationId":
                prop.text = str(next_id - 1)


def main():
    args = _parse_args()
    eaf_path = args.eaf
    out_path = args.out if args.out else eaf_path

    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")

    tree = ET.parse(eaf_path)
    root = tree.getroot()

    ts_map = _ts_map(root)
    _add_gloss_input_tier(root, ts_map)

    # Preserve XML declaration and indentation
    ET.indent(tree, space="    ")
    tree.write(out_path, encoding="UTF-8", xml_declaration=True)
    print(f"Saved -> {out_path}")

    # Verify
    tree2 = ET.parse(out_path)
    root2 = tree2.getroot()
    gi_tier = root2.find(".//TIER[@TIER_ID='Gloss_Input']")
    count = len(gi_tier.findall("ANNOTATION")) if gi_tier is not None else 0
    print(f"Gloss_Input tier has {count} annotations")


if __name__ == "__main__":
    main()
