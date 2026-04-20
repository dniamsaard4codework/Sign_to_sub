"""
fix_overlap_vtt.py
Remove overlaps from a VTT file by clamping each cue's end time
to the next cue's start time.

Usage:
    python fix_overlap_vtt.py --input <input.vtt> --output <output.vtt>
"""
import argparse
from pathlib import Path


def vtt_to_ms(ts: str) -> int:
    h, m, s = ts.strip().split(":")
    return int((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)


def ms_to_vtt(v: int) -> str:
    h, r = divmod(v, 3_600_000)
    m, r = divmod(r, 60_000)
    s, f = divmod(r, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}.{f:03d}"


def main():
    parser = argparse.ArgumentParser(description="Remove VTT subtitle overlaps by clamping end times.")
    parser.add_argument("--input",  required=True, help="Input VTT file path")
    parser.add_argument("--output", required=True, help="Output VTT file path")
    args = parser.parse_args()

    content = Path(args.input).read_text(encoding="utf-8")
    cues = []
    for block in content.strip().split("\n\n")[1:]:
        lines = block.strip().split("\n")
        arrow = next((l for l in lines if "-->" in l), None)
        if not arrow:
            continue
        t1s, t2s = arrow.split("-->")
        text_lines = [l for l in lines if "-->" not in l and not l.strip().isdigit()]
        cues.append([vtt_to_ms(t1s), vtt_to_ms(t2s), " ".join(text_lines).strip()])

    before_overlap = sum(1 for i in range(len(cues) - 1) if cues[i][1] > cues[i + 1][0])

    fixed = 0
    for i in range(len(cues) - 1):
        if cues[i][1] > cues[i + 1][0]:
            cues[i][1] = cues[i + 1][0]
            fixed += 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, (t1, t2, txt) in enumerate(cues, 1):
            f.write(f"{i}\n{ms_to_vtt(t1)} --> {ms_to_vtt(t2)}\n{txt}\n\n")

    total_pairs = max(len(cues) - 1, 1)
    print(f"[OK] {len(cues)} cues processed")
    print(f"     Before: {before_overlap}/{total_pairs} overlapping pairs ({before_overlap/total_pairs*100:.1f}%)")
    print(f"     After:  0/{total_pairs} overlapping pairs (0.0%)")
    print(f"     Fixed {fixed} cue end-times -> {out}")


if __name__ == "__main__":
    main()
