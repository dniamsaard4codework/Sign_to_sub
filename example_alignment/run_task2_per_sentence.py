"""
run_task2_per_sentence.py
-------------------------
Task 2 per-sentence experiment: crop 04.mp4 into one clip per Gloss-tier
sentence, run the full pipeline (pose -> SEA segmentation -> SignCLIP sign
embedding -> per-sentence DP) on each clip, then aggregate the 119
per-sentence outputs into one CSV/VTT/EAF and evaluate against
"Gloss Labeling".

Batch-processing strategy (v1)
==============================
- Phase 1 (parallel-friendly): crop 119 clips into ablation_per_sentence/clips/
  using bundled ffmpeg (imageio_ffmpeg). Frame-accurate re-encode (libx264).
- Phase 2: videos_to_poses --directory clips/   -> 119 .pose files at once.
- Phase 3: SEA/segmentation.py with a video_ids.txt listing all 119 clip ids.
  Outputs to clips/seg/E4s-1_30_50/<clip_id>.eaf .
- Phase 4: extract_episode_features.py --mode=segmentation with the same
  video_ids file. Loads SignCLIP ONCE, emits 119 .npy files into emb/.
- Phase 5 (in-process): for each sentence, load its .eaf + .npy, call the
  shared align_one_sentence() helper with the per-clip sign embedding pool.
  Shift predicted times by +clip_start to recover original video time,
  accumulate.
- Phase 6: write combined CSV/VTT/EAF in ablation_per_sentence/.

Usage
=====
  python example_alignment\\run_task2_per_sentence.py
  # debug: python ... --only-idx 0,1,2 --skip-eval
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from align_gloss_labels import (  # type: ignore
    align_one_sentence,
    embed_tokens_cached,
    inject_eaf_tier,
    load_gloss_sentences,
    load_sign_segments,
    write_predictions_vtt,
)


# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_VIDEO   = HERE / "04.mp4"
DEFAULT_EAF     = HERE / "Test.eaf"
DEFAULT_OUT_DIR = HERE / "ablation_per_sentence"
DEFAULT_BASE_EAF = HERE / "segmentation_output" / "E4s-1_30_50" / "04.eaf"

CLIP_PREFIX = "clip_"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_ffmpeg() -> str:
    """Return absolute path to a usable ffmpeg binary."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as ex:
        raise RuntimeError(
            "ffmpeg not found on PATH and imageio_ffmpeg is unavailable. "
            "pip install imageio_ffmpeg."
        ) from ex


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def probe_duration_seconds(ffmpeg_exe: str, video: Path) -> float:
    """Parse 'Duration: HH:MM:SS.xx' from ffmpeg -i stderr."""
    result = subprocess.run(
        [ffmpeg_exe, "-hide_banner", "-i", str(video)],
        capture_output=True,
        text=True,
    )
    text = (result.stderr or "") + (result.stdout or "")
    m = _DUR_RE.search(text)
    if not m:
        raise RuntimeError(f"Could not parse duration from ffmpeg output for {video}")
    h, mn, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mn * 60 + s


def crop_clip(ffmpeg_exe: str, video: Path, start: float, end: float, out_path: Path,
              quiet: bool = True) -> None:
    """Re-encode crop [start, end) into out_path (libx264, no audio)."""
    duration = max(0.0, end - start)
    cmd = [
        ffmpeg_exe,
        "-y",
        "-hide_banner",
        "-loglevel", "error" if quiet else "info",
        "-ss", f"{start:.3f}",
        "-i", str(video),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",
        str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"ffmpeg crop failed for {out_path.name} "
            f"(start={start:.3f}, dur={duration:.3f}):\n{res.stderr}"
        )


def run_subprocess(cmd: list[str], cwd: Path | None = None, env_extras: dict[str, str] | None = None,
                   stream: bool = True) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    import os
    env = os.environ.copy()
    if env_extras:
        env.update(env_extras)
    if stream:
        res = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    else:
        res = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env,
                             capture_output=True, text=True)
        if res.stdout:
            print(res.stdout)
        if res.stderr:
            print(res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed (rc={res.returncode}): {' '.join(str(c) for c in cmd)}")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Phases
# ──────────────────────────────────────────────────────────────────────────────

def phase_resolve_boundaries(eaf: Path, tier: str, video: Path, ffmpeg_exe: str
                             ) -> tuple[list[dict], float]:
    """Load sentences and pin clip boundaries: clip i = [s[i].start, s[i+1].start)."""
    sentences = load_gloss_sentences(eaf, tier_id=tier)
    print(f"[phase 1] {len(sentences)} sentences loaded from tier '{tier}'.")
    duration = probe_duration_seconds(ffmpeg_exe, video)
    print(f"[phase 1] video duration = {duration:.3f} s")
    for i, sent in enumerate(sentences):
        start = sent["start"]
        end = sentences[i + 1]["start"] if (i + 1) < len(sentences) else duration
        if end < start:
            end = sent["end"]
        sent["clip_start"] = start
        sent["clip_end"] = end
        sent["clip_id"] = f"{CLIP_PREFIX}{i:03d}"
    return sentences, duration


def phase_crop_clips(sentences: list[dict], video: Path, clips_dir: Path,
                     ffmpeg_exe: str, overwrite: bool) -> None:
    clips_dir.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = 0
    t0 = time.time()
    for sent in sentences:
        out_mp4 = clips_dir / f"{sent['clip_id']}.mp4"
        if out_mp4.exists() and not overwrite:
            n_skip += 1
            continue
        crop_clip(ffmpeg_exe, video, sent["clip_start"], sent["clip_end"], out_mp4)
        n_done += 1
    print(f"[phase 2] cropped={n_done}, skipped_existing={n_skip}, "
          f"elapsed={time.time() - t0:.1f}s")


def phase_extract_poses(clips_dir: Path, overwrite: bool) -> None:
    """Run videos_to_poses on the whole clips/ directory."""
    print(f"[phase 3] videos_to_poses --directory {clips_dir}")
    cmd = [
        "videos_to_poses",
        "--format", "mediapipe",
        "--directory", str(clips_dir),
        "--additional-config",
        "model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true",
    ]
    t0 = time.time()
    run_subprocess(cmd)
    print(f"[phase 3] poses done in {time.time() - t0:.1f}s")


def phase_run_segmentation(clips_dir: Path, sign_b: int, sign_o: int) -> Path:
    """Run SEA/segmentation.py on every clip id. Returns the seg subdir."""
    video_ids = sorted(p.stem for p in clips_dir.glob(f"{CLIP_PREFIX}*.pose"))
    if not video_ids:
        raise RuntimeError(f"No .pose files in {clips_dir}")
    ids_file = clips_dir / "video_ids.txt"
    ids_file.write_text("\n".join(video_ids) + "\n", encoding="utf-8")

    save_dir = clips_dir / "seg"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase 4] SEA segmentation on {len(video_ids)} clips")
    cmd = [
        sys.executable,
        str(ROOT / "SEA" / "segmentation.py"),
        "--video_ids", str(ids_file),
        "--pose_dir", str(clips_dir),
        "--video_dir", str(clips_dir),
        "--save_dir", str(save_dir),
        "--sign-b-threshold", str(sign_b),
        "--sign-o-threshold", str(sign_o),
        "--num_workers", "1",
    ]
    t0 = time.time()
    run_subprocess(cmd)
    seg_subdir = save_dir / f"E4s-1_{sign_b}_{sign_o}"
    print(f"[phase 4] segmentation done in {time.time() - t0:.1f}s -> {seg_subdir}")
    return seg_subdir


def phase_extract_embeddings(clips_dir: Path, seg_subdir: Path,
                             emb_dir: Path, model_name: str, language_tag: str) -> None:
    """Run extract_episode_features.py in segmentation mode for all clips."""
    ids_file = clips_dir / "video_ids.txt"
    if not ids_file.exists():
        raise RuntimeError(f"{ids_file} missing; segmentation phase must run first")
    emb_dir.mkdir(parents=True, exist_ok=True)
    mmpt_dir = ROOT / "fairseq_signclip" / "examples" / "MMPT"
    script = mmpt_dir / "scripts_bsl" / "extract_episode_features.py"
    print(f"[phase 5] SignCLIP segmentation-mode embedding ({model_name})")
    cmd = [
        sys.executable,
        str(script),
        "--video_ids", str(ids_file),
        "--mode", "segmentation",
        "--model_name", model_name,
        "--language_tag", language_tag,
        "--pose_dir", str(clips_dir),
        "--segmentation_dir", str(seg_subdir),
        "--save_dir", str(emb_dir),
    ]
    t0 = time.time()
    run_subprocess(cmd, cwd=mmpt_dir)
    print(f"[phase 5] embeddings done in {time.time() - t0:.1f}s")


def phase_align_per_sentence(sentences: list[dict], seg_subdir: Path, emb_dir: Path,
                             token_cache: Path, model_name: str, language_tag: str,
                             gap_penalty: float, coverage_penalty: float,
                             window_pad: float) -> tuple[list[dict], list[dict]]:
    """Per-sentence DP. Returns (global_predictions, boundary_rows)."""
    # Pre-embed every token across all sentences (one cache, one SignCLIP load)
    flat_tokens: list[str] = []
    for s in sentences:
        flat_tokens.extend(s["tokens"])
    print(f"[phase 6] embedding {len(flat_tokens)} tokens ...")
    token_embs = embed_tokens_cached(flat_tokens, model_name, language_tag, token_cache)
    print(f"[phase 6] token_embs.shape = {token_embs.shape}")

    boundary_rows: list[dict] = []
    global_predictions: list[dict] = []
    flat_idx = 0
    fallback_count = 0

    for s_idx, sent in enumerate(sentences):
        T = len(sent["tokens"])
        clip_id = sent["clip_id"]
        clip_start = sent["clip_start"]
        clip_end = sent["clip_end"]
        clip_dur = clip_end - clip_start

        eaf_path = seg_subdir / f"{clip_id}.eaf"
        emb_path = emb_dir / f"{clip_id}.npy"
        status = "ok"
        n_segs = 0

        try:
            segs = load_sign_segments(eaf_path)
            n_segs = len(segs)
        except Exception as ex:
            print(f"  [warn] {clip_id}: load_sign_segments failed: {ex}")
            segs = []
            status = "seg_load_fail"

        try:
            sign_embs = np.load(emb_path).astype(np.float32) if emb_path.exists() else None
        except Exception as ex:
            print(f"  [warn] {clip_id}: load embedding failed: {ex}")
            sign_embs = None
            status = "emb_load_fail"

        if sign_embs is not None and n_segs != sign_embs.shape[0]:
            print(f"  [warn] {clip_id}: segs={n_segs} != embs={sign_embs.shape[0]}; "
                  f"using min length")
            limit = min(n_segs, sign_embs.shape[0])
            segs = segs[:limit]
            sign_embs = sign_embs[:limit]
            n_segs = limit
            status = "len_mismatch_truncated" if status == "ok" else status

        local_sent = {
            "start": 0.0,
            "end":   clip_dur,
            "text":  sent["text"],
            "tokens": sent["tokens"],
        }
        sent_token_embs = token_embs[flat_idx:flat_idx + T]
        flat_idx += T

        if n_segs == 0 or sign_embs is None:
            preds_local: list[dict] = []
            used_fallback = True
            step = clip_dur / max(T, 1)
            for t_idx, tok in enumerate(sent["tokens"]):
                preds_local.append({
                    "start": t_idx * step,
                    "end":   (t_idx + 1) * step,
                    "token": tok,
                    "sentence_idx": s_idx,
                    "token_idx":    t_idx,
                    "score": float("nan"),
                    "fallback": "uniform_no_segs",
                })
            if status == "ok":
                status = "no_segments"
        else:
            seg_starts_arr = np.array([sg["start"] for sg in segs], dtype=np.float64)
            seg_ends_arr   = np.array([sg["end"]   for sg in segs], dtype=np.float64)
            seg_mids_arr   = np.array([sg["mid"]   for sg in segs], dtype=np.float64)
            preds_local, used_fallback = align_one_sentence(
                sent=local_sent,
                s_idx=s_idx,
                sent_token_embs=sent_token_embs,
                seg_starts_arr=seg_starts_arr,
                seg_ends_arr=seg_ends_arr,
                seg_mids_arr=seg_mids_arr,
                sign_embs=sign_embs,
                gap_penalty=gap_penalty,
                coverage_penalty=coverage_penalty,
                window_pad=window_pad,
            )

        if used_fallback:
            fallback_count += 1

        # Shift predictions back to original video time and accumulate.
        for p_local in preds_local:
            global_predictions.append({
                "start": p_local["start"] + clip_start,
                "end":   p_local["end"]   + clip_start,
                "token": p_local["token"],
                "sentence_idx": p_local["sentence_idx"],
                "token_idx":    p_local["token_idx"],
                "score": p_local["score"],
                "fallback": p_local["fallback"],
            })

        boundary_rows.append({
            "sentence_idx": s_idx,
            "clip_id": clip_id,
            "clip_start_s": round(clip_start, 3),
            "clip_end_s":   round(clip_end, 3),
            "clip_dur_s":   round(clip_dur, 3),
            "n_tokens": T,
            "n_segs": n_segs,
            "fallback": "uniform" if used_fallback else "",
            "status": status,
        })

    print(f"[phase 6] done. fallback sentences: {fallback_count} / {len(sentences)}")
    return global_predictions, boundary_rows


def write_combined_outputs(predictions: list[dict], boundary_rows: list[dict],
                           out_dir: Path, base_eaf: Path) -> dict:
    """Write combined CSV/VTT/EAF + boundaries.csv. Return paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gloss_labels_pred__per_sentence.csv"
    vtt_path = out_dir / "gloss_labels_pred__per_sentence.vtt"
    eaf_path = out_dir / "04_gloss_pred__per_sentence.eaf"
    bnd_path = out_dir / "boundaries.csv"

    sorted_preds = sorted(predictions, key=lambda x: (x["start"], x["end"]))

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "start_s", "end_s", "gloss_token",
            "sentence_idx", "token_idx", "score", "fallback",
        ])
        writer.writeheader()
        for p in sorted_preds:
            writer.writerow({
                "start_s": round(p["start"], 3),
                "end_s":   round(p["end"],   3),
                "gloss_token": p["token"],
                "sentence_idx": p["sentence_idx"],
                "token_idx":    p["token_idx"],
                "score": "" if (p["score"] != p["score"]) else round(p["score"], 4),
                "fallback": p["fallback"],
            })
    write_predictions_vtt(sorted_preds, vtt_path)
    inject_eaf_tier(base_eaf, eaf_path, sorted_preds, tier_id="GLOSS_LABEL_PRED")

    with open(bnd_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "sentence_idx", "clip_id", "clip_start_s", "clip_end_s",
            "clip_dur_s", "n_tokens", "n_segs", "fallback", "status",
        ])
        writer.writeheader()
        for row in boundary_rows:
            writer.writerow(row)

    return {"csv": csv_path, "vtt": vtt_path, "eaf": eaf_path, "boundaries": bnd_path}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_only_idx(spec: str | None, max_idx: int) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            for i in range(int(a), int(b) + 1):
                out.add(i)
        else:
            out.add(int(chunk))
    if not out:
        return None
    return {i for i in out if 0 <= i <= max_idx}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-sentence Task 2: crop video -> full pipeline per clip -> aggregate.",
    )
    ap.add_argument("--video",     type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--eaf",       type=Path, default=DEFAULT_EAF)
    ap.add_argument("--tier",      default="Gloss")
    ap.add_argument("--out-dir",   type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--base-eaf",  type=Path, default=DEFAULT_BASE_EAF,
                    help="Whole-video segmentation EAF used as base for the combined output EAF.")
    ap.add_argument("--sign-b",    type=int, default=30)
    ap.add_argument("--sign-o",    type=int, default=50)
    ap.add_argument("--model-name",   default="multilingual")
    ap.add_argument("--language-tag", default="<en> <bfi>")
    ap.add_argument("--gap-penalty",      type=float, default=2.0)
    ap.add_argument("--coverage-penalty", type=float, default=0.5)
    ap.add_argument("--window-pad",       type=float, default=0.5)
    ap.add_argument("--only-idx", default=None,
                    help="Comma-separated sentence indices or ranges (e.g. '0,1,5-7') to process.")
    ap.add_argument("--overwrite-clips", action="store_true",
                    help="Re-crop clips even if mp4 already exists.")
    ap.add_argument("--skip-crop", action="store_true",
                    help="Skip phase 2 (assume clips already cropped).")
    ap.add_argument("--skip-pose", action="store_true",
                    help="Skip phase 3 (assume .pose files exist).")
    ap.add_argument("--skip-seg",  action="store_true",
                    help="Skip phase 4 (assume seg EAFs exist).")
    ap.add_argument("--skip-emb",  action="store_true",
                    help="Skip phase 5 (assume .npy files exist).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = args.out_dir / "clips"
    emb_dir   = args.out_dir / "emb"
    token_cache = args.out_dir / "token_cache.npz"

    ffmpeg_exe = get_ffmpeg()
    print(f"[init] ffmpeg = {ffmpeg_exe}")

    # Phase 1: boundaries
    all_sentences, vid_dur = phase_resolve_boundaries(args.eaf, args.tier, args.video, ffmpeg_exe)
    keep = parse_only_idx(args.only_idx, len(all_sentences) - 1)
    if keep is None:
        sentences = all_sentences
    else:
        sentences = [s for i, s in enumerate(all_sentences) if i in keep]
        print(f"[init] --only-idx -> processing {len(sentences)} of {len(all_sentences)} sentences")

    # Phase 2: crop
    if args.skip_crop:
        print("[phase 2] SKIPPED")
    else:
        phase_crop_clips(sentences, args.video, clips_dir, ffmpeg_exe, args.overwrite_clips)

    # Phase 3: pose
    if args.skip_pose:
        print("[phase 3] SKIPPED")
    else:
        phase_extract_poses(clips_dir, overwrite=False)

    # Phase 4: segmentation
    seg_subdir = clips_dir / "seg" / f"E4s-1_{args.sign_b}_{args.sign_o}"
    if args.skip_seg:
        print(f"[phase 4] SKIPPED (expecting {seg_subdir})")
    else:
        seg_subdir = phase_run_segmentation(clips_dir, args.sign_b, args.sign_o)

    # Phase 5: embeddings
    if args.skip_emb:
        print("[phase 5] SKIPPED")
    else:
        phase_extract_embeddings(clips_dir, seg_subdir, emb_dir,
                                 args.model_name, args.language_tag)

    # Phase 6: align per sentence (in-process)
    global_predictions, boundary_rows = phase_align_per_sentence(
        sentences=sentences,
        seg_subdir=seg_subdir,
        emb_dir=emb_dir,
        token_cache=token_cache,
        model_name=args.model_name,
        language_tag=args.language_tag,
        gap_penalty=args.gap_penalty,
        coverage_penalty=args.coverage_penalty,
        window_pad=args.window_pad,
    )

    # Phase 7: combine
    paths = write_combined_outputs(global_predictions, boundary_rows, args.out_dir, args.base_eaf)
    print(f"[OK] CSV         -> {paths['csv']}")
    print(f"[OK] VTT         -> {paths['vtt']}")
    print(f"[OK] EAF         -> {paths['eaf']}")
    print(f"[OK] boundaries  -> {paths['boundaries']}")
    print(f"[done] {len(global_predictions)} predictions across {len(sentences)} sentences.")


if __name__ == "__main__":
    main()
