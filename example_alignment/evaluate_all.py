"""
evaluate_all.py
Compares multiple aligned VTT outputs against CC_Aligned ground truth
extracted from 04_updated.eaf.

Metrics (same as README_EVAL.md):
  - mean / median start offset (signed, seconds)
  - stdev of offset
  - % within ±1s / ±2s / ±3s
  - overlap rate (how many consecutive cue pairs overlap)
"""
import xml.etree.ElementTree as ET
import statistics
from pathlib import Path

EAF_PATH = Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\การเปรียบเทียบและเรียงลำดับ (11.07 นาที).eaf")

CC_VTT = Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\subtitles\04.vtt")

# (vtt_path, use_cc_text_for_matching)
# use_cc_text_for_matching=True  -> cue text is Gloss; look up CC text by index for matching
EXPERIMENTS = {
    "B2   (BSL, CC text, tuned)":
        (Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\aligned_output_with_embedding_tuned\04.vtt"), False),
    "B_MULTI (multilingual, CC text)":
        (Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\aligned_output_multi_b2\04.vtt"), False),
    "C_MULTI  (multilingual, Gloss text)":
        (Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\aligned_output_multi_gloss\04.vtt"), True),
    "C_MULTI_word (multilingual, Gloss, word-level)":
        (Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\aligned_output_multi_gloss_word\04.vtt"), True),
    "D_ASL        (ASL, CC text)":
        (Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\aligned_output_asl_b2\04.vtt"), False),
    "D_ASL_gloss  (ASL, Gloss text)":
        (Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\aligned_output_asl_gloss\04.vtt"), True),
    "D_ASL_word   (ASL, Gloss, word-level)":
        (Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\aligned_output_asl_gloss_word\04.vtt"), True),
}


# ── helpers ─────────────────────────────────────────────────────────────────

def vtt_to_ms(ts: str) -> int:
    h, m, s = ts.strip().split(":")
    return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)


def load_cc_aligned(eaf_path: Path):
    """Return sorted list of (start_ms, end_ms, text) from CC_Aligned tier."""
    tree = ET.parse(eaf_path)
    root = tree.getroot()
    ts_map = {
        ts.get("TIME_SLOT_ID"): int(ts.get("TIME_VALUE", 0))
        for ts in root.findall(".//TIME_SLOT")
    }
    entries = []
    for tier in root.findall("TIER"):
        if tier.get("TIER_ID") == "CC_Aligned":
            for ann in tier.findall(".//ALIGNABLE_ANNOTATION"):
                t1 = ts_map.get(ann.get("TIME_SLOT_REF1", ""), 0)
                t2 = ts_map.get(ann.get("TIME_SLOT_REF2", ""), 0)
                val = ann.find("ANNOTATION_VALUE")
                text = (val.text or "").strip() if val is not None else ""
                entries.append((t1, t2, text))
    entries.sort(key=lambda x: x[0])
    return entries


def load_vtt(path: Path):
    """Return list of (start_ms, end_ms, text) from a VTT file."""
    cues = []
    with open(path, encoding="utf-8") as f:
        content = f.read()
    for block in content.strip().split("\n\n")[1:]:
        lines = block.strip().split("\n")
        arrow = next((l for l in lines if "-->" in l), None)
        if arrow is None:
            continue
        t1s, t2s = arrow.split("-->")
        t1, t2 = vtt_to_ms(t1s), vtt_to_ms(t2s)
        text_lines = [l for l in lines if "-->" not in l and not l.strip().isdigit()]
        text = " ".join(text_lines).strip()
        cues.append((t1, t2, text))
    return cues


def normalize_text(t: str) -> str:
    """Simple normalization for text matching."""
    return " ".join(t.split()).lower()


def match_cues(pred_cues, gt_cues, cc_cues_for_lookup=None):
    """
    Match predicted cues to ground-truth by text lookup.
    If cc_cues_for_lookup is given, the predicted cues have Gloss text;
    look up the corresponding CC text by index to do the matching.
    """
    gt_by_text = {}
    for (t1, t2, text) in gt_cues:
        key = normalize_text(text)
        if key not in gt_by_text:
            gt_by_text[key] = (t1, t2)

    matched = []
    for idx, (p_t1, p_t2, p_text) in enumerate(pred_cues):
        if cc_cues_for_lookup is not None and idx < len(cc_cues_for_lookup):
            lookup_text = cc_cues_for_lookup[idx][2]  # use original CC text for matching
        else:
            lookup_text = p_text
        key = normalize_text(lookup_text)
        if key in gt_by_text:
            g_t1, g_t2 = gt_by_text[key]
            matched.append((p_t1 / 1000.0, g_t1 / 1000.0))
    return matched


def overlap_pairs(cues):
    """Count consecutive overlapping pairs."""
    count = 0
    for i in range(len(cues) - 1):
        if cues[i][1] > cues[i + 1][0]:
            count += 1
    return count


# ── main ─────────────────────────────────────────────────────────────────────

def evaluate(pred_cues, gt_cues, cc_cues_for_lookup=None):
    matched = match_cues(pred_cues, gt_cues, cc_cues_for_lookup=cc_cues_for_lookup)
    if not matched:
        return None

    offsets = [p - g for p, g in matched]
    n = len(offsets)

    mean_off = statistics.mean(offsets)
    median_off = statistics.median(offsets)
    stdev_off = statistics.pstdev(offsets)

    w1 = sum(1 for o in offsets if abs(o) <= 1.0) / n * 100
    w2 = sum(1 for o in offsets if abs(o) <= 2.0) / n * 100
    w3 = sum(1 for o in offsets if abs(o) <= 3.0) / n * 100

    n_overlap = overlap_pairs(pred_cues)
    total_pairs = max(len(pred_cues) - 1, 1)
    overlap_pct = n_overlap / total_pairs * 100

    return {
        "matched": n,
        "total_pred": len(pred_cues),
        "mean_off": mean_off,
        "median_off": median_off,
        "stdev": stdev_off,
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "overlap_pct": overlap_pct,
    }


def main():
    gt = load_cc_aligned(EAF_PATH)
    cc_cues = load_vtt(CC_VTT)
    print(f"CC_Aligned ground truth: {len(gt)} entries")
    print(f"CC subtitles (for Gloss lookup): {len(cc_cues)} cues\n")

    rows = []
    for label, (vtt_path, use_cc_lookup) in EXPERIMENTS.items():
        if not vtt_path.exists():
            print(f"[SKIP] {label} -- file not found: {vtt_path}")
            continue
        pred = load_vtt(vtt_path)
        cc_lookup = cc_cues if use_cc_lookup else None
        r = evaluate(pred, gt, cc_cues_for_lookup=cc_lookup)
        if r is None:
            print(f"[WARN] {label} -- no text matches found")
            continue
        rows.append((label, r))

    # Print comparison table
    print("=" * 100)
    print(f"{'Experiment':<44} {'Match':>6} {'Mean':>7} {'Median':>8} {'Stdev':>6} {'+-1s':>5} {'+-2s':>5} {'+-3s':>5} {'Overlap':>8}")
    print("-" * 100)
    for label, r in rows:
        print(
            f"{label:<44} "
            f"{r['matched']:>5}/{r['total_pred']:<4} "
            f"{r['mean_off']:>+7.2f}s "
            f"{r['median_off']:>+7.2f}s "
            f"{r['stdev']:>6.2f}s "
            f"{r['w1']:>4.0f}% "
            f"{r['w2']:>4.0f}% "
            f"{r['w3']:>4.0f}% "
            f"{r['overlap_pct']:>6.1f}%"
        )
    print("=" * 100)

    # Dataset alignment verification
    print("\n--- Dataset alignment verification ---")
    print(f"CC_Aligned entries : {len(gt)}")
    print("Checking that CC_Aligned has 0% overlap (expected: ground truth is overlap-free)...")
    n_gt_overlap = overlap_pairs(gt)
    print(f"CC_Aligned overlapping pairs: {n_gt_overlap}/{len(gt)-1} ({n_gt_overlap/(len(gt)-1)*100:.1f}%)")
    if n_gt_overlap == 0:
        print("  OK - CC_Aligned is overlap-free (confirmed as clean ground truth)")
    else:
        print("  WARNING - CC_Aligned has overlaps (check ground truth integrity)")


if __name__ == "__main__":
    main()
