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

FPS_DEFAULT = 25  # frame rate used for frame-level accuracy and F1 metrics

EAF_PATH = Path(r"C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\Test.eaf")

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


# ── SEA-style frame-level helpers ────────────────────────────────────────────

def _get_labels_start_end_time(frame_wise_labels, bg_class):
    """Find (labels, starts, ends) of each contiguous non-background segment."""
    labels, starts, ends = [], [], []
    if not frame_wise_labels:
        return labels, starts, ends
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(1, len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if last_label not in bg_class:
                ends.append(i)
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(len(frame_wise_labels))
    return labels, starts, ends


def _f_score(recognized, ground_truth, overlap, bg_class):
    """Segment-level precision/recall/F1 at an IoU overlap threshold (SEA-style).
    Returns (tp, fp, fn) as floats.
    """
    p_label, p_start, p_end = _get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = _get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0
    hits = [0] * len(y_label)

    for j in range(len(p_label)):
        best_iou = 0.0
        best_k = -1
        for k in range(len(y_label)):
            if p_label[j] != y_label[k]:
                continue
            intersection = min(p_end[j], y_end[k]) - max(p_start[j], y_start[k])
            union = max(p_end[j], y_end[k]) - min(p_start[j], y_start[k])
            iou = max(0.0, intersection) / max(union, 1)
            if iou > best_iou:
                best_iou = iou
                best_k = k
        if best_k >= 0 and best_iou >= overlap and not hits[best_k]:
            tp += 1
            hits[best_k] = 1
        else:
            fp += 1

    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def subs2frames(cues_ms, max_time_ms, fps, background_label=-1):
    """Convert (start_ms, end_ms, ...) cues into a frame-level label sequence.
    Each cue gets its index as label; gaps/silence get background_label (-1).
    """
    total_frames = round(fps * max_time_ms / 1000.0)
    if total_frames == 0:
        return []
    frames = [background_label] * total_frames
    for idx, (start_ms, end_ms, *_) in enumerate(cues_ms):
        s = max(0, min(round(fps * start_ms / 1000.0), total_frames))
        e = max(0, min(round(fps * end_ms   / 1000.0), total_frames))
        if s < e:
            frames[s:e] = [idx] * (e - s)
    return frames


def match_cues(pred_cues, gt_cues, cc_cues_for_lookup=None):
    """
    Index-based matching: CC_Input[i] <-> CC_Aligned[i] by position.
    Both have 119 entries so all pairs are evaluated (no text-lookup filtering).
    cc_cues_for_lookup is accepted for backward-compat but ignored.

    อัปเดต 2026-05-04: เปลี่ยนจาก text-lookup (69/172 matched) เป็น
    index-based (119/119 matched) เนื่องจาก CC_Input และ CC_Aligned
    มีจำนวน cue เท่ากัน (119) และ aligned ตาม index เดียวกัน
    """
    matched = []
    n = min(len(pred_cues), len(gt_cues))
    for i in range(n):
        p_t1 = pred_cues[i][0] / 1000.0
        g_t1 = gt_cues[i][0]  / 1000.0
        p_t2 = pred_cues[i][1] / 1000.0
        g_t2 = gt_cues[i][1]  / 1000.0
        matched.append((p_t1, g_t1, p_t2, g_t2))
    return matched


def overlap_pairs(cues):
    """Count consecutive overlapping pairs."""
    count = 0
    for i in range(len(cues) - 1):
        if cues[i][1] > cues[i + 1][0]:
            count += 1
    return count


# ── main ─────────────────────────────────────────────────────────────────────

def evaluate(pred_cues, gt_cues, cc_cues_for_lookup=None, fps=FPS_DEFAULT):
    matched = match_cues(pred_cues, gt_cues, cc_cues_for_lookup=cc_cues_for_lookup)
    if not matched:
        return None

    offsets     = [p_s - g_s for p_s, g_s, _p_e, _g_e in matched]
    end_offsets = [p_e - g_e for _p_s, _g_s, p_e, g_e in matched]
    n = len(offsets)

    mean_off        = statistics.mean(offsets)
    median_off      = statistics.median(offsets)
    stdev_off       = statistics.pstdev(offsets)
    mean_off_abs    = statistics.mean(abs(o) for o in offsets)
    median_off_abs  = statistics.median(abs(o) for o in offsets)

    mean_end_off        = statistics.mean(end_offsets)
    median_end_off      = statistics.median(end_offsets)
    mean_end_off_abs    = statistics.mean(abs(o) for o in end_offsets)
    median_end_off_abs  = statistics.median(abs(o) for o in end_offsets)

    w1 = sum(1 for o in offsets if abs(o) <= 1.0) / n * 100
    w2 = sum(1 for o in offsets if abs(o) <= 2.0) / n * 100
    w3 = sum(1 for o in offsets if abs(o) <= 3.0) / n * 100

    n_overlap = overlap_pairs(pred_cues)
    total_pairs = max(len(pred_cues) - 1, 1)
    overlap_pct = n_overlap / total_pairs * 100

    # ── SEA-style frame-level accuracy + F1@0.10/0.25/0.50 ──────────────────
    max_time_ms = max((c[1] for c in gt_cues), default=0) + 10_000
    pred_frames = subs2frames(pred_cues, max_time_ms, fps)
    gt_frames   = subs2frames(gt_cues,  max_time_ms, fps)
    min_len = min(len(pred_frames), len(gt_frames))
    pred_frames, gt_frames = pred_frames[:min_len], gt_frames[:min_len]
    frame_acc = (
        sum(1 for p, g in zip(pred_frames, gt_frames) if p == g) / min_len * 100
        if min_len else 0.0
    )

    f1_10, f1_25, f1_50 = 0.0, 0.0, 0.0
    for thr, attr in [(0.10, 'f1_10'), (0.25, 'f1_25'), (0.50, 'f1_50')]:
        tp, fp, fn = _f_score(pred_frames, gt_frames, thr, bg_class=[-1])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        val  = 2 * prec * rec / (prec + rec) * 100 if (prec + rec) > 0 else 0.0
        if attr == 'f1_10':   f1_10 = val
        elif attr == 'f1_25': f1_25 = val
        else:                 f1_50 = val

    return {
        # ── existing keys (unchanged — evaluate_all_to_csv.py depends on these) ──
        "matched":    n,
        "total_pred": len(pred_cues),
        "mean_off":   mean_off,
        "median_off": median_off,
        "stdev":      stdev_off,
        "w1":         w1,
        "w2":         w2,
        "w3":         w3,
        "overlap_pct": overlap_pct,
        # ── new: absolute start offset ──────────────────────────────────────
        "mean_off_abs":   mean_off_abs,
        "median_off_abs": median_off_abs,
        # ── new: end-time offset (signed + abs) ─────────────────────────────
        "mean_end_off":       mean_end_off,
        "median_end_off":     median_end_off,
        "mean_end_off_abs":   mean_end_off_abs,
        "median_end_off_abs": median_end_off_abs,
        # ── new: SEA-style frame-level + F1 ────────────────────────────────
        "frame_acc": frame_acc,
        "f1_10":     f1_10,
        "f1_25":     f1_25,
        "f1_50":     f1_50,
    }


def main():
    gt = load_cc_aligned(EAF_PATH)
    cc_cues = load_vtt(CC_VTT)
    print(f"CC_Aligned ground truth: {len(gt)} entries")
    print(f"CC_Input subtitles (index-based matching): {len(cc_cues)} cues\n")

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

    # ── Table 1: start-offset metrics (original) ────────────────────────────
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

    # ── Table 2: SEA-style metrics (frame accuracy + F1 + abs/end offset) ───
    print()
    print("SEA-style metrics (frame-level accuracy, F1@IoU, end-time offset)")
    print("=" * 110)
    print(f"{'Experiment':<44} {'FrameAcc':>9} {'F1@.10':>7} {'F1@.25':>7} {'F1@.50':>7} "
          f"{'|start|':>8} {'|end|':>7} {'EndMean':>8} {'EndMed':>7}")
    print("-" * 110)
    for label, r in rows:
        print(
            f"{label:<44} "
            f"{r['frame_acc']:>8.2f}% "
            f"{r['f1_10']:>6.2f}% "
            f"{r['f1_25']:>6.2f}% "
            f"{r['f1_50']:>6.2f}% "
            f"{r['mean_off_abs']:>7.2f}s "
            f"{r['mean_end_off_abs']:>6.2f}s "
            f"{r['mean_end_off']:>+7.2f}s "
            f"{r['median_end_off']:>+6.2f}s"
        )
    print("=" * 110)

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
