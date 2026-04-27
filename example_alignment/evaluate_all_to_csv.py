"""
evaluate_all_to_csv.py
----------------------
1) Run fix_overlap_vtt logic on every aligned_output_*/04.vtt to create
   a sibling 04_no_overlap.vtt (skips files that already exist & are newer).
2) Re-evaluate ALL experiments (pre- and post-overlap) against CC_Aligned.
3) Write evaluation_task1_results.csv (UTF-8 with BOM) so Excel/ELAN tools
   render Thai correctly.
4) Print a side-by-side before/after-overlap-fix table to stdout.

Reuses (no monkey-patching):
  - evaluate_all.EXPERIMENTS, load_cc_aligned, load_vtt, evaluate, overlap_pairs
  - fix_overlap_vtt.vtt_to_ms, ms_to_vtt
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from evaluate_all import (
    EXPERIMENTS,
    EAF_PATH,
    CC_VTT,
    load_cc_aligned,
    load_vtt,
    evaluate,
)
from fix_overlap_vtt import vtt_to_ms, ms_to_vtt

OUT_CSV = HERE / "evaluation_task1_results.csv"


def make_no_overlap(src: Path) -> tuple[Path, int, int]:
    """
    Produce a sibling <name>_no_overlap.vtt next to src by clamping each cue's
    end to the next cue's start. Returns (out_path, before_overlap, total_pairs).
    Skips work (but still reports counts) if dst exists and is newer than src.
    """
    dst = src.with_name(src.stem + "_no_overlap.vtt")

    content = src.read_text(encoding="utf-8")
    cues = []
    for block in content.strip().split("\n\n")[1:]:
        lines = block.strip().split("\n")
        arrow = next((l for l in lines if "-->" in l), None)
        if not arrow:
            continue
        t1s, t2s = arrow.split("-->")
        text_lines = [l for l in lines if "-->" not in l and not l.strip().isdigit()]
        cues.append([vtt_to_ms(t1s), vtt_to_ms(t2s), " ".join(text_lines).strip()])

    before = sum(1 for i in range(len(cues) - 1) if cues[i][1] > cues[i + 1][0])
    total_pairs = max(len(cues) - 1, 1)

    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst, before, total_pairs

    for i in range(len(cues) - 1):
        if cues[i][1] > cues[i + 1][0]:
            cues[i][1] = cues[i + 1][0]

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, (t1, t2, txt) in enumerate(cues, 1):
            f.write(f"{i}\n{ms_to_vtt(t1)} --> {ms_to_vtt(t2)}\n{txt}\n\n")

    return dst, before, total_pairs


def main() -> None:
    gt = load_cc_aligned(EAF_PATH)
    cc_cues = load_vtt(CC_VTT)
    print(f"CC_Aligned ground truth: {len(gt)} entries")
    print(f"CC subtitles (Gloss lookup): {len(cc_cues)} cues\n")

    rows: list[dict] = []

    print("--- Generating *_no_overlap.vtt for every experiment ---")
    no_overlap_paths: dict[str, Path] = {}
    for label, (vtt_path, _) in EXPERIMENTS.items():
        if not vtt_path.exists():
            print(f"  [skip] {label}: source missing -> {vtt_path}")
            continue
        dst, before, total = make_no_overlap(vtt_path)
        no_overlap_paths[label] = dst
        print(f"  [+] {label}: {before}/{total} overlapping pairs -> {dst.name}")
    print()

    print("--- Evaluating pre- and post-overlap variants ---")
    for label, (vtt_path, use_cc_lookup) in EXPERIMENTS.items():
        if not vtt_path.exists():
            continue
        for variant_label, variant_path, post_overlap in [
            (label, vtt_path, False),
            (label + "  [no_overlap]", no_overlap_paths.get(label), True),
        ]:
            if variant_path is None or not variant_path.exists():
                continue
            pred = load_vtt(variant_path)
            cc_lookup = cc_cues if use_cc_lookup else None
            r = evaluate(pred, gt, cc_cues_for_lookup=cc_lookup)
            if r is None:
                print(f"  [warn] {variant_label}: no text matches")
                continue
            rows.append({
                "experiment": variant_label,
                "vtt_path": str(variant_path),
                "post_overlap": post_overlap,
                "matched": r["matched"],
                "total_pred": r["total_pred"],
                "mean_off_s": round(r["mean_off"], 4),
                "median_off_s": round(r["median_off"], 4),
                "stdev_s": round(r["stdev"], 4),
                "w1_pct": round(r["w1"], 2),
                "w2_pct": round(r["w2"], 2),
                "w3_pct": round(r["w3"], 2),
                "overlap_pct": round(r["overlap_pct"], 2),
            })
    print()

    # Sanity: post_overlap rows must have overlap_pct == 0
    for row in rows:
        if row["post_overlap"] and row["overlap_pct"] != 0.0:
            print(f"  [warn] {row['experiment']}: post-overlap row has overlap_pct={row['overlap_pct']}")

    # Side-by-side summary
    print("=" * 110)
    header = (
        f"{'Experiment':<46} {'Variant':<10} {'Mean':>8} "
        f"{'+-1s':>6} {'+-2s':>6} {'+-3s':>6} {'Overlap':>9}"
    )
    print(header)
    print("-" * 110)
    by_label: dict[str, dict[bool, dict]] = {}
    for r in rows:
        base = r["experiment"].replace("  [no_overlap]", "")
        by_label.setdefault(base, {})[r["post_overlap"]] = r
    for base, variants in by_label.items():
        for is_post, tag in [(False, "before"), (True, " after")]:
            r = variants.get(is_post)
            if not r:
                continue
            print(
                f"{base:<46} {tag:<10} "
                f"{r['mean_off_s']:>+7.2f}s "
                f"{r['w1_pct']:>5.0f}% "
                f"{r['w2_pct']:>5.0f}% "
                f"{r['w3_pct']:>5.0f}% "
                f"{r['overlap_pct']:>7.1f}%"
            )
    print("=" * 110)

    # Write CSV
    fieldnames = [
        "experiment", "vtt_path", "post_overlap",
        "matched", "total_pred",
        "mean_off_s", "median_off_s", "stdev_s",
        "w1_pct", "w2_pct", "w3_pct", "overlap_pct",
    ]
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Wrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
