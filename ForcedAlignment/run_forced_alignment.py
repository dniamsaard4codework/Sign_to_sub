"""
run_forced_alignment.py
-----------------------
Dataset orchestrator for Task 2 on the ForcedAlignment corpus.

Phases:
  1. Build manifest and video_ids.txt.
  2. Stage videos and run videos_to_poses.
  3. Run SEA segmentation.
  4. Run SignCLIP segment embeddings.
  5. Run in-process DP alignment for configs 1..5.
  6. Inject prediction tiers and export CSV/VTT.
  7. Evaluate predictions with evaluate_fa_dataset.py.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FA_DIR = Path(__file__).resolve().parent
ROOT = FA_DIR.parent
DEFAULT_EAF_DIR = FA_DIR / "elan_forced_alignment"
DEFAULT_OUT_DIR = FA_DIR / "output"

sys.path.insert(0, str(ROOT / "example_alignment"))
from align_gloss_labels import (  # type: ignore
    align_one_sentence,
    embed_tokens_cached,
    load_sign_segments,
    seconds_to_vtt,
)


@dataclass(frozen=True)
class RunConfig:
    key: str
    input_tier: str
    gt_tier: str
    output_tier: str
    csv_name: str
    token_mode: str


CONFIGS: dict[str, RunConfig] = {
    "1": RunConfig(
        key="1",
        input_tier="CC",
        gt_tier="CC_Aligned",
        output_tier="CC_Aligned_pred",
        csv_name="config1_CC_Aligned_pred.csv",
        token_mode="whitespace_no_sil",
    ),
    "2": RunConfig(
        key="2",
        input_tier="CC",
        gt_tier="CC_Aligned",
        output_tier="CC_Aligned_silmodel_pred",
        csv_name="config2_CC_Aligned_silmodel_pred.csv",
        token_mode="whitespace_with_sil",
    ),
    "3": RunConfig(
        key="3",
        input_tier="Gloss",
        gt_tier="Gloss_Labeling",
        output_tier="Gloss_Labeling_pred",
        csv_name="config3_Gloss_Labeling_pred.csv",
        token_mode="pipe_drop_empty",
    ),
    "4": RunConfig(
        key="4",
        input_tier="Gloss1",
        gt_tier="Gloss_Labeling1",
        output_tier="Gloss_Labeling1_pred",
        csv_name="config4_Gloss_Labeling1_pred.csv",
        token_mode="pipe_drop_empty",
    ),
    "5": RunConfig(
        key="5",
        input_tier="Gloss2",
        gt_tier="Gloss_Labeling2",
        output_tier="Gloss_Labeling2_pred",
        csv_name="config5_Gloss_Labeling2_pred.csv",
        token_mode="pipe_drop_empty",
    ),
}


def numeric_sort_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def parse_configs(spec: str) -> list[RunConfig]:
    if spec.lower() == "all":
        keys = ["1", "2", "3", "4", "5"]
    else:
        keys = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [k for k in keys if k not in CONFIGS]
    if unknown:
        raise ValueError(f"Unknown configs: {', '.join(unknown)}")
    return [CONFIGS[k] for k in keys]


def parse_id_spec(spec: str | None) -> set[str] | None:
    if not spec:
        return None
    out: set[str] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            for i in range(int(a), int(b) + 1):
                out.add(str(i))
        else:
            out.add(str(int(chunk)) if chunk.isdigit() else chunk)
    return out or None


def default_video_root() -> Path:
    candidates: list[tuple[int, Path]] = []
    for child in FA_DIR.iterdir():
        if not child.is_dir() or child.name in {"elan_forced_alignment", "output"}:
            continue
        count = sum(1 for _ in child.rglob("*.mp4"))
        if count:
            candidates.append((count, child))
    if not candidates:
        raise FileNotFoundError("Could not find a video root under ForcedAlignment/")
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def build_video_index(video_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for video in video_root.rglob("*.mp4"):
        if video.stem in index:
            duplicates.setdefault(video.stem, [index[video.stem]]).append(video)
        else:
            index[video.stem] = video.resolve()
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:10])
        raise RuntimeError(f"Duplicate MP4 stems under {video_root}: {sample}")
    return index


def load_time_slots(root: ET.Element) -> dict[str, float]:
    return {
        ts.get("TIME_SLOT_ID"): float(ts.get("TIME_VALUE", 0)) / 1000.0
        for ts in root.findall(".//TIME_SLOT")
    }


def load_tier_entries(eaf_path: Path, tier_id: str) -> list[dict]:
    tree = ET.parse(eaf_path)
    root = tree.getroot()
    ts_map = load_time_slots(root)
    target = None
    for tier in root.findall("TIER"):
        if tier.get("TIER_ID") == tier_id:
            target = tier
            break
    if target is None:
        raise ValueError(f"Tier {tier_id!r} not found in {eaf_path}")

    out: list[dict] = []
    for ann in target.findall("ANNOTATION"):
        elem = next(iter(ann), None)
        if elem is None or elem.tag != "ALIGNABLE_ANNOTATION":
            continue
        t1 = ts_map.get(elem.get("TIME_SLOT_REF1"))
        t2 = ts_map.get(elem.get("TIME_SLOT_REF2"))
        val = elem.find("ANNOTATION_VALUE")
        text = (val.text or "").strip() if val is not None else ""
        if t1 is None or t2 is None or not text:
            continue
        if t2 < t1:
            t1, t2 = t2, t1
        out.append({"start": t1, "end": t2, "text": text})
    out.sort(key=lambda x: (x["start"], x["end"], x["text"]))
    return out


def tokens_for_config(text: str, cfg: RunConfig) -> list[str]:
    if cfg.token_mode == "whitespace_no_sil":
        return [t for t in text.split() if t]
    if cfg.token_mode == "whitespace_with_sil":
        return ["sil", *[t for t in text.split() if t], "sil"]
    if cfg.token_mode == "pipe_drop_empty":
        return [t.strip() for t in text.split("|") if t.strip()]
    raise ValueError(f"Unsupported token mode: {cfg.token_mode}")


def clip_sentence_for_config(eaf_path: Path, cfg: RunConfig) -> dict:
    entries = load_tier_entries(eaf_path, cfg.input_tier)
    if not entries:
        raise ValueError(f"No entries in tier {cfg.input_tier!r}: {eaf_path}")
    start = min(e["start"] for e in entries)
    end = max(e["end"] for e in entries)
    text = " ".join(e["text"] for e in entries)
    tokens = tokens_for_config(text, cfg)
    if not tokens:
        raise ValueError(f"No tokens for config {cfg.key} in {eaf_path}")
    return {"start": start, "end": end, "text": text, "tokens": tokens}


def build_manifest(eaf_dir: Path, video_root: Path, out_dir: Path, only_ids: set[str] | None) -> list[dict]:
    eaf_files = sorted(eaf_dir.glob("*.eaf"), key=numeric_sort_key)
    if only_ids is not None:
        eaf_files = [p for p in eaf_files if p.stem in only_ids]
    if not eaf_files:
        raise FileNotFoundError("No EAF files selected")

    video_index = build_video_index(video_root)
    rows: list[dict] = []
    missing: list[str] = []
    for eaf in eaf_files:
        video = video_index.get(eaf.stem)
        if video is None:
            missing.append(eaf.name)
            continue
        rows.append({
            "clip_id": eaf.stem,
            "eaf_path": str(eaf.resolve()),
            "video_path": str(video),
            "video_parent": str(video.parent),
        })
    if missing:
        raise FileNotFoundError(f"Missing videos for EAFs: {', '.join(missing[:20])}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = out_dir / "manifest.csv"
    with open(manifest_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "eaf_path", "video_path", "video_parent"])
        writer.writeheader()
        writer.writerows(rows)

    video_ids = out_dir / "video_ids.txt"
    video_ids.write_text("\n".join(row["clip_id"] for row in rows) + "\n", encoding="utf-8")
    print(f"[phase 1] manifest: {len(rows)} clips -> {manifest_csv}")
    return rows


def verify_required_tiers(rows: list[dict], configs: list[RunConfig]) -> None:
    required = sorted({c.input_tier for c in configs} | {c.gt_tier for c in configs})
    counts = {tier: 0 for tier in required}
    for row in rows:
        root = ET.parse(row["eaf_path"]).getroot()
        present = {tier.get("TIER_ID") for tier in root.findall("TIER")}
        for tier in required:
            if tier in present:
                counts[tier] += 1
    print("[preflight] tier presence")
    for tier in required:
        print(f"  {tier:<18} {counts[tier]} / {len(rows)}")
    missing = [tier for tier in required if counts[tier] != len(rows)]
    if missing:
        raise RuntimeError(f"Missing required tiers: {', '.join(missing)}")


def command_env() -> dict[str, str]:
    env = os.environ.copy()
    scripts_dir = ROOT / "venv" / "Scripts"
    if scripts_dir.exists():
        env["PATH"] = str(scripts_dir) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=command_env())
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(str(c) for c in cmd)}")


def videos_to_poses_exe() -> str:
    found = shutil.which("videos_to_poses")
    if found:
        return found
    exe = ROOT / "venv" / "Scripts" / "videos_to_poses.exe"
    if exe.exists():
        return str(exe)
    return "videos_to_poses"


def stage_videos(rows: list[dict], video_work_dir: Path, overwrite: bool) -> None:
    video_work_dir.mkdir(parents=True, exist_ok=True)
    linked = copied = skipped = 0
    for row in rows:
        src = Path(row["video_path"])
        dst = video_work_dir / f"{row['clip_id']}.mp4"
        if dst.exists() and not overwrite:
            skipped += 1
            continue
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
            linked += 1
        except OSError:
            shutil.copy2(src, dst)
            copied += 1
    print(f"[phase 2] staged videos: hardlinked={linked}, copied={copied}, skipped={skipped}")


def phase_pose(rows: list[dict], out_dir: Path, overwrite: bool, num_workers: int) -> None:
    video_work_dir = out_dir / "video_work"
    pose_dir = out_dir / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)

    missing_rows = [
        row for row in rows
        if overwrite or not (pose_dir / f"{row['clip_id']}.pose").exists()
    ]
    if not missing_rows:
        print("[phase 2] all selected pose files already exist")
        return

    stage_videos(missing_rows, video_work_dir, overwrite=overwrite)
    cmd = [
        videos_to_poses_exe(),
        "--format", "mediapipe",
        "--directory", str(video_work_dir),
        "--additional-config", "model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true",
        "--num-workers", str(num_workers),
    ]
    run_cmd(cmd)

    copied = 0
    for row in missing_rows:
        src_pose = video_work_dir / f"{row['clip_id']}.pose"
        dst_pose = pose_dir / f"{row['clip_id']}.pose"
        if not src_pose.exists():
            raise FileNotFoundError(src_pose)
        shutil.copy2(src_pose, dst_pose)
        copied += 1
    print(f"[phase 2] poses ready: copied {copied} -> {pose_dir}")


def phase_segmentation(rows: list[dict], out_dir: Path, sign_b: int, sign_o: int, overwrite: bool) -> Path:
    pose_dir = out_dir / "poses"
    video_work_dir = out_dir / "video_work"
    save_dir = out_dir / "seg"
    seg_subdir = save_dir / f"E4s-1_{sign_b}_{sign_o}"
    save_dir.mkdir(parents=True, exist_ok=True)

    missing = [row["clip_id"] for row in rows if overwrite or not (seg_subdir / f"{row['clip_id']}.eaf").exists()]
    if not missing:
        print(f"[phase 3] all selected segmentation EAFs already exist -> {seg_subdir}")
        return seg_subdir

    ids_file = out_dir / "video_ids_seg.txt"
    ids_file.write_text("\n".join(missing) + "\n", encoding="utf-8")
    cmd = [
        sys.executable,
        str(ROOT / "SEA" / "segmentation.py"),
        "--video_ids", str(ids_file),
        "--pose_dir", str(pose_dir),
        "--video_dir", str(video_work_dir),
        "--save_dir", str(save_dir),
        "--sign-b-threshold", str(sign_b),
        "--sign-o-threshold", str(sign_o),
        "--num_workers", "1",
    ]
    if overwrite:
        cmd.append("--overwrite")
    run_cmd(cmd)
    print(f"[phase 3] segmentation ready -> {seg_subdir}")
    return seg_subdir


def phase_embeddings(
    rows: list[dict],
    out_dir: Path,
    seg_subdir: Path,
    model_name: str,
    language_tag: str,
    overwrite: bool,
) -> Path:
    emb_dir = out_dir / "emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    missing = [row["clip_id"] for row in rows if overwrite or not (emb_dir / f"{row['clip_id']}.npy").exists()]
    if not missing:
        print(f"[phase 4] all selected embeddings already exist -> {emb_dir}")
        return emb_dir

    ids_file = out_dir / "video_ids_emb.txt"
    ids_file.write_text("\n".join(missing) + "\n", encoding="utf-8")
    mmpt_dir = ROOT / "fairseq_signclip" / "examples" / "MMPT"
    script = mmpt_dir / "scripts_bsl" / "extract_episode_features.py"
    cmd = [
        sys.executable,
        str(script),
        "--video_ids", str(ids_file),
        "--pose_dir", str(out_dir / "poses"),
        "--save_dir", str(emb_dir),
        "--mode", "segmentation",
        "--segmentation_dir", str(seg_subdir),
        "--model_name", model_name,
        "--language_tag", language_tag,
    ]
    if overwrite:
        cmd.append("--overwrite")
    run_cmd(cmd, cwd=mmpt_dir)
    print(f"[phase 4] embeddings ready -> {emb_dir}")
    return emb_dir


def next_annotation_index(root: ET.Element) -> int:
    max_idx = 0
    for elem in root.findall(".//ALIGNABLE_ANNOTATION"):
        ann_id = elem.get("ANNOTATION_ID", "")
        digits = "".join(ch for ch in ann_id if ch.isdigit())
        if digits:
            max_idx = max(max_idx, int(digits))
    return max_idx + 1


def next_time_slot_index(root: ET.Element) -> int:
    max_idx = 0
    for ts in root.findall(".//TIME_SLOT"):
        ts_id = ts.get("TIME_SLOT_ID", "")
        digits = "".join(ch for ch in ts_id if ch.isdigit())
        if digits:
            max_idx = max(max_idx, int(digits))
    return max_idx + 1


def inject_prediction_tiers(base_eaf: Path, out_eaf: Path, tiers: dict[str, list[dict]]) -> None:
    out_eaf.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(base_eaf)
    root = tree.getroot()
    time_order = root.find("TIME_ORDER")
    if time_order is None:
        raise ValueError(f"TIME_ORDER missing in {base_eaf}")

    for tier_id in tiers:
        for old in list(root.findall("TIER")):
            if old.get("TIER_ID") == tier_id:
                root.remove(old)

    lt_ids = {lt.get("LINGUISTIC_TYPE_ID") for lt in root.findall("LINGUISTIC_TYPE")}
    if "subtitle-lt" not in lt_ids:
        root.append(ET.Element("LINGUISTIC_TYPE", {
            "GRAPHIC_REFERENCES": "false",
            "LINGUISTIC_TYPE_ID": "subtitle-lt",
            "TIME_ALIGNABLE": "true",
        }))

    ts_idx = next_time_slot_index(root)
    ann_idx = next_annotation_index(root)
    for tier_id, preds in tiers.items():
        tier = ET.Element("TIER", {
            "LINGUISTIC_TYPE_REF": "subtitle-lt",
            "TIER_ID": tier_id,
        })
        for pred in preds:
            ts1 = f"{tier_id}_ts{ts_idx}"
            ts_idx += 1
            ts2 = f"{tier_id}_ts{ts_idx}"
            ts_idx += 1
            time_order.append(ET.Element("TIME_SLOT", {
                "TIME_SLOT_ID": ts1,
                "TIME_VALUE": str(int(round(pred["start"] * 1000))),
            }))
            time_order.append(ET.Element("TIME_SLOT", {
                "TIME_SLOT_ID": ts2,
                "TIME_VALUE": str(int(round(pred["end"] * 1000))),
            }))
            ann = ET.SubElement(tier, "ANNOTATION")
            aa = ET.SubElement(ann, "ALIGNABLE_ANNOTATION", {
                "ANNOTATION_ID": f"{tier_id}_a{ann_idx}",
                "TIME_SLOT_REF1": ts1,
                "TIME_SLOT_REF2": ts2,
            })
            ann_idx += 1
            ET.SubElement(aa, "ANNOTATION_VALUE").text = pred["token"]
        root.append(tier)
    tree.write(out_eaf, encoding="utf-8", xml_declaration=True)


def write_vtt(predictions: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, pred in enumerate(predictions, 1):
            f.write(
                f"{i}\n"
                f"{seconds_to_vtt(pred['start'])} --> {seconds_to_vtt(pred['end'])}\n"
                f"{pred['token']}\n\n"
            )


def align_config(
    rows: list[dict],
    cfg: RunConfig,
    out_dir: Path,
    seg_subdir: Path,
    emb_dir: Path,
    model_name: str,
    language_tag: str,
    gap_penalty: float,
    coverage_penalty: float,
    window_pad: float,
) -> tuple[list[dict], dict[str, list[dict]]]:
    sentences: list[dict] = []
    flat_tokens: list[str] = []
    for row in rows:
        sent = clip_sentence_for_config(Path(row["eaf_path"]), cfg)
        sent["clip_id"] = row["clip_id"]
        sentences.append(sent)
        flat_tokens.extend(sent["tokens"])

    cache = out_dir / "token_cache" / f"{model_name}_{language_tag.replace(' ', '_').replace('<', '').replace('>', '')}.npz"
    print(f"[phase 5] config {cfg.key}: embedding {len(flat_tokens)} tokens")
    token_embs = embed_tokens_cached(flat_tokens, model_name, language_tag, cache)

    csv_rows: list[dict] = []
    tiers_by_clip: dict[str, list[dict]] = {}
    flat_idx = 0
    fallback_clips = 0

    for sent in sentences:
        clip_id = sent["clip_id"]
        token_count = len(sent["tokens"])
        sent_token_embs = token_embs[flat_idx:flat_idx + token_count]
        flat_idx += token_count

        seg_eaf = seg_subdir / f"{clip_id}.eaf"
        emb_path = emb_dir / f"{clip_id}.npy"
        status = "ok"

        try:
            segs = load_sign_segments(seg_eaf)
        except Exception as ex:
            print(f"  [warn] {clip_id}: cannot load SIGN tier: {ex}")
            segs = []
            status = "seg_load_fail"

        sign_embs = None
        if emb_path.exists():
            try:
                sign_embs = np.load(emb_path).astype(np.float32)
            except Exception as ex:
                print(f"  [warn] {clip_id}: cannot load embeddings: {ex}")
                status = "emb_load_fail"
        else:
            status = "emb_missing"

        if sign_embs is not None and len(segs) != sign_embs.shape[0]:
            limit = min(len(segs), sign_embs.shape[0])
            print(f"  [warn] {clip_id}: segs={len(segs)} embs={sign_embs.shape[0]}, truncating to {limit}")
            segs = segs[:limit]
            sign_embs = sign_embs[:limit]
            status = "len_mismatch_truncated"

        if not segs or sign_embs is None:
            duration = max(0.001, sent["end"] - sent["start"])
            step = duration / max(token_count, 1)
            preds = [
                {
                    "start": sent["start"] + i * step,
                    "end": sent["start"] + (i + 1) * step,
                    "token": token,
                    "sentence_idx": 0,
                    "token_idx": i,
                    "score": float("nan"),
                    "fallback": f"uniform_{status}",
                }
                for i, token in enumerate(sent["tokens"])
            ]
            used_fallback = True
        else:
            seg_starts = np.array([s["start"] for s in segs], dtype=np.float64)
            seg_ends = np.array([s["end"] for s in segs], dtype=np.float64)
            seg_mids = np.array([s["mid"] for s in segs], dtype=np.float64)
            preds, used_fallback = align_one_sentence(
                sent=sent,
                s_idx=0,
                sent_token_embs=sent_token_embs,
                seg_starts_arr=seg_starts,
                seg_ends_arr=seg_ends,
                seg_mids_arr=seg_mids,
                sign_embs=sign_embs,
                gap_penalty=gap_penalty,
                coverage_penalty=coverage_penalty,
                window_pad=window_pad,
            )

        if used_fallback:
            fallback_clips += 1

        sorted_preds = sorted(preds, key=lambda x: (x["token_idx"], x["start"], x["end"]))
        tiers_by_clip[clip_id] = sorted_preds
        for pred in sorted_preds:
            csv_rows.append({
                "clip_id": clip_id,
                "start_s": round(pred["start"], 3),
                "end_s": round(pred["end"], 3),
                "token": pred["token"],
                "source_tier": cfg.input_tier,
                "gt_tier": cfg.gt_tier,
                "sentence_idx": pred["sentence_idx"],
                "token_idx": pred["token_idx"],
                "score": "" if pred["score"] != pred["score"] else round(pred["score"], 4),
                "fallback": pred["fallback"],
            })

    print(f"[phase 5] config {cfg.key}: predictions={len(csv_rows)}, fallback_clips={fallback_clips}")
    return csv_rows, tiers_by_clip


def phase_align(
    rows: list[dict],
    configs: list[RunConfig],
    out_dir: Path,
    seg_subdir: Path,
    emb_dir: Path,
    model_name: str,
    language_tag: str,
    gap_penalty: float,
    coverage_penalty: float,
    window_pad: float,
) -> None:
    pred_dir = out_dir / "predictions"
    vtt_dir = out_dir / "prediction_vtt"
    eaf_dir = out_dir / "predicted_eafs"
    pred_dir.mkdir(parents=True, exist_ok=True)
    per_clip_tiers: dict[str, dict[str, list[dict]]] = {row["clip_id"]: {} for row in rows}

    for cfg in configs:
        csv_rows, tiers_by_clip = align_config(
            rows=rows,
            cfg=cfg,
            out_dir=out_dir,
            seg_subdir=seg_subdir,
            emb_dir=emb_dir,
            model_name=model_name,
            language_tag=language_tag,
            gap_penalty=gap_penalty,
            coverage_penalty=coverage_penalty,
            window_pad=window_pad,
        )
        csv_path = pred_dir / cfg.csv_name
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "clip_id", "start_s", "end_s", "token", "source_tier", "gt_tier",
                "sentence_idx", "token_idx", "score", "fallback",
            ])
            writer.writeheader()
            writer.writerows(csv_rows)
        write_vtt([
            {
                "start": row["start_s"],
                "end": row["end_s"],
                "token": f"{row['clip_id']}:{row['token']}",
            }
            for row in csv_rows
        ], vtt_dir / cfg.csv_name.replace(".csv", ".vtt"))
        for clip_id, preds in tiers_by_clip.items():
            per_clip_tiers[clip_id][cfg.output_tier] = preds
        print(f"[phase 5] wrote {csv_path}")

    for row in rows:
        clip_id = row["clip_id"]
        base_eaf = seg_subdir / f"{clip_id}.eaf"
        if not base_eaf.exists():
            base_eaf = Path(row["eaf_path"])
        inject_prediction_tiers(base_eaf, eaf_dir / f"{clip_id}.eaf", per_clip_tiers[clip_id])
    print(f"[phase 6] injected prediction EAFs -> {eaf_dir}")


def phase_evaluate(out_dir: Path, eaf_dir: Path, configs: list[RunConfig]) -> None:
    cfg_spec = ",".join(cfg.key for cfg in configs)
    cmd = [
        sys.executable,
        str(FA_DIR / "evaluate_fa_dataset.py"),
        "--eaf-dir", str(eaf_dir),
        "--pred-dir", str(out_dir / "predictions"),
        "--out-dir", str(out_dir / "evaluation"),
        "--configs", cfg_spec,
    ]
    run_cmd(cmd)


def selected_rows_with_existing_manifest(out_dir: Path, eaf_dir: Path, video_root: Path, only_ids: set[str] | None) -> list[dict]:
    return build_manifest(eaf_dir, video_root, out_dir, only_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 2 on the ForcedAlignment dataset.")
    parser.add_argument("--eaf-dir", type=Path, default=DEFAULT_EAF_DIR)
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--configs", default="all")
    parser.add_argument("--only-ids", default=None, help="Comma/range list, e.g. 1,500,1132 or 1-10")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--skip-seg", action="store_true")
    parser.add_argument("--skip-emb", action="store_true")
    parser.add_argument("--skip-align", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pose-num-workers", type=int, default=1)
    parser.add_argument("--sign-b", type=int, default=30)
    parser.add_argument("--sign-o", type=int, default=50)
    parser.add_argument("--model-name", default="multilingual")
    parser.add_argument("--language-tag", default="<en> <bfi>")
    parser.add_argument("--gap-penalty", type=float, default=2.0)
    parser.add_argument("--coverage-penalty", type=float, default=0.5)
    parser.add_argument("--window-pad", type=float, default=0.5)
    args = parser.parse_args()

    t0 = time.time()
    configs = parse_configs(args.configs)
    video_root = args.video_root or default_video_root()
    only_ids = parse_id_spec(args.only_ids)

    print(f"[init] eaf_dir    = {args.eaf_dir}")
    print(f"[init] video_root = {video_root}")
    print(f"[init] out_dir    = {args.out_dir}")
    print(f"[init] configs    = {','.join(cfg.key for cfg in configs)}")

    rows = selected_rows_with_existing_manifest(args.out_dir, args.eaf_dir, video_root, only_ids)
    verify_required_tiers(rows, configs)
    if args.preflight_only:
        print("[done] preflight-only")
        return

    if args.skip_pose:
        print("[phase 2] SKIPPED pose")
    else:
        phase_pose(rows, args.out_dir, overwrite=args.overwrite, num_workers=args.pose_num_workers)

    seg_subdir = args.out_dir / "seg" / f"E4s-1_{args.sign_b}_{args.sign_o}"
    if args.skip_seg:
        print(f"[phase 3] SKIPPED segmentation, expecting {seg_subdir}")
    else:
        seg_subdir = phase_segmentation(rows, args.out_dir, args.sign_b, args.sign_o, overwrite=args.overwrite)

    emb_dir = args.out_dir / "emb"
    if args.skip_emb:
        print(f"[phase 4] SKIPPED embeddings, expecting {emb_dir}")
    else:
        emb_dir = phase_embeddings(
            rows,
            args.out_dir,
            seg_subdir,
            model_name=args.model_name,
            language_tag=args.language_tag,
            overwrite=args.overwrite,
        )

    if args.skip_align:
        print("[phase 5/6] SKIPPED alignment/export")
    else:
        phase_align(
            rows=rows,
            configs=configs,
            out_dir=args.out_dir,
            seg_subdir=seg_subdir,
            emb_dir=emb_dir,
            model_name=args.model_name,
            language_tag=args.language_tag,
            gap_penalty=args.gap_penalty,
            coverage_penalty=args.coverage_penalty,
            window_pad=args.window_pad,
        )

    if args.skip_eval:
        print("[phase 7] SKIPPED evaluation")
    else:
        phase_evaluate(args.out_dir, args.eaf_dir, configs)

    print(f"[done] elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
