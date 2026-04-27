"""
plot_alignment.py
-----------------
Render a multi-lane timeline visualization for the first WINDOW_SECS seconds
of the demo video, comparing:
  * CC          - raw closed captions (timed by audio)
  * CC_Aligned  - manual ground-truth alignment to signs
  * C_MULTI     - best Task 1 alignment (Multilingual + Gloss text)
  * GLOSS_LABEL_PRED - new Task 2 token-level prediction

Output:
  example_alignment/figures/timeline_first_2min.png
"""
from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent

EAF_PATH    = HERE / "การเปรียบเทียบและเรียงลำดับ (11.07 นาที).eaf"
C_MULTI_VTT = HERE / "aligned_output_multi_gloss" / "04_no_overlap.vtt"
PRED_CSV    = HERE / "gloss_labels_pred.csv"
OUT_PNG     = HERE / "figures" / "timeline_first_2min.png"

WINDOW_SECS = 120.0


def load_tier(eaf: Path, tier_id: str) -> list[tuple[float, float, str]]:
    tree = ET.parse(eaf)
    root = tree.getroot()
    ts_map = {ts.get("TIME_SLOT_ID"): float(ts.get("TIME_VALUE", 0)) / 1000.0
              for ts in root.find("TIME_ORDER").findall("TIME_SLOT")}
    out: list[tuple[float, float, str]] = []
    for tier in root.findall("TIER"):
        if tier.get("TIER_ID") != tier_id:
            continue
        for ann in tier.findall("ANNOTATION"):
            elem = next(iter(ann), None)
            if elem is None or elem.tag != "ALIGNABLE_ANNOTATION":
                continue
            t1 = ts_map.get(elem.get("TIME_SLOT_REF1"))
            t2 = ts_map.get(elem.get("TIME_SLOT_REF2"))
            val = elem.find("ANNOTATION_VALUE")
            txt = (val.text or "").strip() if val is not None else ""
            if t1 is None or t2 is None:
                continue
            out.append((min(t1, t2), max(t1, t2), txt))
    return out


def load_vtt_intervals(vtt: Path) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    content = vtt.read_text(encoding="utf-8")
    for block in content.strip().split("\n\n")[1:]:
        lines = block.strip().split("\n")
        arrow = next((l for l in lines if "-->" in l), None)
        if not arrow:
            continue
        t1s, t2s = arrow.split("-->")

        def to_s(t: str) -> float:
            h, m, s = t.strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)

        text_lines = [l for l in lines if "-->" not in l and not l.strip().isdigit()]
        out.append((to_s(t1s), to_s(t2s), " ".join(text_lines).strip()))
    return out


def load_pred_csv(csv_path: Path) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out.append((float(row["start_s"]), float(row["end_s"]), row["gloss_token"]))
    return out


def filter_window(intervals, t_max):
    return [(s, e, t) for (s, e, t) in intervals if s < t_max]


def draw_lane(ax, intervals, y, color, label):
    for (s, e, _txt) in intervals:
        rect = Rectangle((s, y - 0.35), max(e - s, 0.05), 0.7,
                         facecolor=color, edgecolor="black", alpha=0.75, linewidth=0.4)
        ax.add_patch(rect)


def main() -> None:
    cc        = filter_window(load_tier(EAF_PATH, "CC"),         WINDOW_SECS)
    cc_align  = filter_window(load_tier(EAF_PATH, "CC_Aligned"), WINDOW_SECS)
    c_multi   = filter_window(load_vtt_intervals(C_MULTI_VTT),   WINDOW_SECS)
    gloss_pr  = filter_window(load_pred_csv(PRED_CSV),           WINDOW_SECS)

    fig, ax = plt.subplots(figsize=(16, 4.5))
    lanes = [
        ("CC (raw)",          cc,       "#9e9e9e", 4),
        ("CC_Aligned (GT)",   cc_align, "#1976d2", 3),
        ("C_MULTI (pred)",    c_multi,  "#43a047", 2),
        ("GLOSS_LABEL_PRED",  gloss_pr, "#f57c00", 1),
    ]
    for name, ivs, color, y in lanes:
        draw_lane(ax, ivs, y, color, name)

    ax.set_xlim(0, WINDOW_SECS)
    ax.set_ylim(0.4, 4.6)
    ax.set_yticks([y for _, _, _, y in lanes])
    ax.set_yticklabels([f"{name}\n(n={len(ivs)})" for name, ivs, _, _ in lanes])
    ax.set_xlabel("Time (seconds)")
    ax.set_title(f"Alignment timeline — first {int(WINDOW_SECS)} seconds")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    print(f"[OK] Saved -> {OUT_PNG}")
    print(f"     Lanes: " + ", ".join(f"{name}({len(ivs)})" for name, ivs, _, _ in lanes))


if __name__ == "__main__":
    main()
