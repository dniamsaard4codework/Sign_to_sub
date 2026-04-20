# make_gloss_cc_vtt.py
"""
สร้าง VTT ที่มี:
  - timestamp จาก CC (speech time) — เพื่อให้ SEA ยังต้อง align เอง
  - text จาก Gloss tier — เพื่อให้ embedding ตรงกับมือมากขึ้น
สำหรับ CC entry ที่ไม่ overlap กับ Gloss ใดเลย ให้ใช้ CC text ต้นฉบับแทน
"""
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE_EAF = Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\การเปรียบเทียบและเรียงลำดับ (11.07 นาที).eaf")
CC_VTT     = Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\subtitles\04.vtt")
OUT_VTT    = Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\subtitles_gloss_cc_time\04.vtt")


def ms_to_vtt(v: int) -> str:
    h, r = divmod(v, 3_600_000)
    m, r = divmod(r, 60_000)
    s, f = divmod(r, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{f:03d}"


def vtt_to_ms(ts: str) -> int:
    h, m, s = ts.strip().split(":")
    return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)


def overlap(a1, a2, b1, b2) -> int:
    return max(0, min(a2, b2) - max(a1, b1))


def main():
    tree = ET.parse(SOURCE_EAF)
    root = tree.getroot()
    ts_map = {
        ts.get("TIME_SLOT_ID"): int(ts.get("TIME_VALUE", 0))
        for ts in root.findall(".//TIME_SLOT")
    }
    gloss_entries = []
    for tier in root.findall("TIER"):
        if tier.get("TIER_ID") == "Gloss":
            for ann in tier.findall(".//ALIGNABLE_ANNOTATION"):
                t1 = ts_map.get(ann.get("TIME_SLOT_REF1", ""), 0)
                t2 = ts_map.get(ann.get("TIME_SLOT_REF2", ""), 0)
                val = ann.find("ANNOTATION_VALUE")
                text = (val.text or "").strip() if val is not None else ""
                if text:
                    gloss_entries.append((t1, t2, text))
    gloss_entries.sort(key=lambda x: x[0])

    cc_cues = []
    with open(CC_VTT, encoding="utf-8") as f:
        content = f.read()
    for block in content.strip().split("\n\n")[1:]:
        lines = block.strip().split("\n")
        arrow_line = next((l for l in lines if "-->" in l), None)
        if arrow_line is None:
            continue
        t1s, t2s = arrow_line.split("-->")
        t1, t2 = vtt_to_ms(t1s), vtt_to_ms(t2s)
        text_lines = [l for l in lines if "-->" not in l and not l.strip().isdigit()]
        cc_cues.append((t1, t2, " ".join(text_lines).strip()))

    OUT_VTT.parent.mkdir(parents=True, exist_ok=True)
    mapped = no_overlap = 0
    with open(OUT_VTT, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, (cc_t1, cc_t2, cc_text) in enumerate(cc_cues, 1):
            best = max(gloss_entries, key=lambda g: overlap(cc_t1, cc_t2, g[0], g[1]))
            if overlap(cc_t1, cc_t2, best[0], best[1]) > 0:
                use_text = best[2]
                mapped += 1
            else:
                use_text = cc_text
                no_overlap += 1
            f.write(f"{i}\n{ms_to_vtt(cc_t1)} --> {ms_to_vtt(cc_t2)}\n{use_text}\n\n")

    print(f"[OK] {len(cc_cues)} cues -> {OUT_VTT}")
    print(f"     Gloss text: {mapped} cues | fallback CC: {no_overlap} cues")


if __name__ == "__main__":
    main()
