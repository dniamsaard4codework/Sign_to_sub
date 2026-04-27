"""
align_gloss_labels.py
---------------------
Task 2 prototype: align individual gloss tokens to individual SIGN segments
within each Gloss-tier sentence.

Approach
========
For each Gloss sentence (start_s, end_s, "tok1 tok2 tok3 ..."):
  1. Tokenize on whitespace (annotators already split tokens with spaces).
  2. Restrict candidate SIGN segments to those whose midpoint is in [start_s, end_s]
     (with a configurable padding fallback if the window contains 0 segments).
  3. Embed each token via SignCLIP multilingual text encoder.
  4. Build T x K similarity matrix (cosine -> row softmax).
  5. Run a small monotonic DP per sentence to assign each token to a contiguous
     range of segments. (T~7, K~30 -> O(T*K^2) is trivial.)

Outputs
=======
  - example_alignment/gloss_labels_pred.csv   (UTF-8-sig)
  - example_alignment/gloss_labels_pred.vtt   (companion VTT for ELAN visibility)
  - example_alignment/segmentation_output/E4s-1_30_50/04_gloss_pred.eaf
        new tier "GLOSS_LABEL_PRED" injected into a copy of 04.eaf
  - example_alignment/subtitle_embedding/sign_clip_multi_gloss_tokens/04.npz
        cached token embeddings keyed by (token, language_tag)

Reused
======
  - SEA/utils.py: get_sign_segments_from_eaf
  - example_alignment/extract_cc_from_eaf.py: tier reading pattern
  - example_alignment/add_vtt_tiers_to_eaf.py: EAF tier injection pattern
  - fairseq_signclip/.../extract_episode_features.py: load_model, embed_text
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "SEA"))
sys.path.insert(0, str(ROOT / "fairseq_signclip" / "examples" / "MMPT" / "scripts_bsl"))

# Defaults
DEFAULT_EAF       = HERE / "การเปรียบเทียบและเรียงลำดับ (11.07 นาที).eaf"
DEFAULT_SEG_EAF   = HERE / "segmentation_output" / "E4s-1_30_50" / "04.eaf"
DEFAULT_SIGN_EMB  = HERE / "segmentation_embedding" / "sign_clip_multi" / "04.npy"
DEFAULT_OUT_CSV   = HERE / "gloss_labels_pred.csv"
DEFAULT_OUT_VTT   = HERE / "gloss_labels_pred.vtt"
DEFAULT_OUT_EAF   = HERE / "segmentation_output" / "E4s-1_30_50" / "04_gloss_pred.eaf"
DEFAULT_CACHE     = HERE / "subtitle_embedding" / "sign_clip_multi_gloss_tokens" / "04.npz"


# ── Gloss tier reader (Thai-aware) ───────────────────────────────────────────

def load_gloss_sentences(eaf_path: Path, tier_id: str = "Gloss") -> list[dict]:
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    time_slots: dict[str, float] = {}
    for ts in root.find("TIME_ORDER").findall("TIME_SLOT"):
        ts_id = ts.get("TIME_SLOT_ID")
        ts_val = ts.get("TIME_VALUE")
        if ts_id and ts_val is not None:
            time_slots[ts_id] = float(ts_val) / 1000.0

    target = None
    for tier in root.findall("TIER"):
        if tier.get("TIER_ID") == tier_id:
            target = tier
            break
    if target is None:
        raise ValueError(f"Tier '{tier_id}' not found in {eaf_path}")

    sentences: list[dict] = []
    for ann in target.findall("ANNOTATION"):
        elem = next(iter(ann), None)
        if elem is None:
            continue
        ts1 = elem.get("TIME_SLOT_REF1")
        ts2 = elem.get("TIME_SLOT_REF2")
        val = elem.find("ANNOTATION_VALUE")
        text = (val.text or "").strip() if val is not None else ""
        s, e = time_slots.get(ts1), time_slots.get(ts2)
        if s is None or e is None:
            continue
        if e < s:
            s, e = e, s
        if not text:
            continue
        tokens = text.split()
        if not tokens:
            continue
        sentences.append({"start": s, "end": e, "text": text, "tokens": tokens})

    sentences.sort(key=lambda x: x["start"])
    return sentences


# ── SIGN tier reader (no SEA import needed; matches utils.get_sign_segments_from_eaf) ──

def load_sign_segments(seg_eaf: Path, tier_id: str = "SIGN") -> list[dict]:
    tree = ET.parse(seg_eaf)
    root = tree.getroot()

    time_slots: dict[str, int] = {}
    for ts in root.find("TIME_ORDER").findall("TIME_SLOT"):
        time_slots[ts.get("TIME_SLOT_ID")] = int(ts.get("TIME_VALUE", 0))

    target = None
    for tier in root.findall("TIER"):
        if tier.get("TIER_ID") == tier_id:
            target = tier
            break
    if target is None:
        raise ValueError(f"Tier '{tier_id}' not found in {seg_eaf}")

    segs: list[dict] = []
    for ann in target.findall("ANNOTATION"):
        elem = next(iter(ann), None)
        if elem is None:
            continue
        t1 = time_slots.get(elem.get("TIME_SLOT_REF1"))
        t2 = time_slots.get(elem.get("TIME_SLOT_REF2"))
        val = elem.find("ANNOTATION_VALUE")
        txt = (val.text or "").strip() if val is not None else ""
        if t1 is None or t2 is None:
            continue
        if t2 < t1:
            t1, t2 = t2, t1
        segs.append({
            "start_ms": t1,
            "end_ms":   t2,
            "start":    t1 / 1000.0,
            "end":      t2 / 1000.0,
            "mid":      (t1 + t2) / 2000.0,
            "text":     txt,
        })
    segs.sort(key=lambda s: s["start_ms"])
    return segs


# ── Token embedding (cached) ─────────────────────────────────────────────────

def embed_tokens_cached(
    all_tokens: list[str],
    model_name: str,
    language_tag: str,
    cache_path: Path,
) -> np.ndarray:
    """Return (N, 768) token embeddings; cache by (token, language_tag) key."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        npz = np.load(cache_path, allow_pickle=False)
        keys = npz["keys"]
        vecs = npz["vecs"]
        for k, v in zip(keys, vecs):
            cache[str(k)] = v
        print(f"  [cache] loaded {len(cache)} cached token embeddings from {cache_path.name}")

    def make_key(tok: str) -> str:
        return f"{language_tag}||{tok}"

    missing = []
    for tok in all_tokens:
        if make_key(tok) not in cache:
            if tok not in missing:
                missing.append(tok)

    if missing:
        import os
        print(f"  [embed] {len(missing)} new tokens to embed via SignCLIP {model_name}")
        # SignCLIP yaml configs reference checkpoints by RELATIVE paths
        # (e.g. 'runs/retri_v1_1/baseline_temporal/checkpoint_best.pt'),
        # so the cwd must be the MMPT root while loading and embedding.
        mmpt_root = ROOT / "fairseq_signclip" / "examples" / "MMPT"
        old_cwd = os.getcwd()
        os.chdir(mmpt_root)
        try:
            from extract_episode_features import load_model, embed_text  # type: ignore
            import extract_episode_features as _eef  # type: ignore
            if model_name not in _eef.models:
                load_model(model_name)
            for tok in missing:
                text_prompt = f"{language_tag} {tok}"
                emb = embed_text(text_prompt, model_name=model_name)  # (1, 768)
                cache[make_key(tok)] = emb[0].astype(np.float32)
        finally:
            os.chdir(old_cwd)
        keys_arr = np.array(list(cache.keys()))
        vecs_arr = np.stack(list(cache.values()), axis=0).astype(np.float32)
        np.savez(cache_path, keys=keys_arr, vecs=vecs_arr)
        print(f"  [cache] saved {len(cache)} token embeddings -> {cache_path.name}")

    out = np.stack([cache[make_key(tok)] for tok in all_tokens], axis=0)
    return out


# ── DP within one sentence ──────────────────────────────────────────────────

INF = 1e18


def monotonic_token_dp(
    sim: np.ndarray,                # (T, K) row-normalised similarity
    seg_starts: np.ndarray,         # (K,) seconds
    seg_ends: np.ndarray,           # (K,) seconds
    sentence_dur: float,
    gap_penalty: float,
    coverage_penalty: float,
) -> list[tuple[int, int]]:
    """Return list of (k_start, k_end) inclusive ranges, length T."""
    T, K = sim.shape
    if K < T:
        return []  # caller will fall back

    target_dur = sentence_dur / max(T, 1)

    # Pre-compute cumulative similarity along K for fast range sums per token.
    # cum_sim[t, j] = sum sim[t, 0..j-1]
    cum_sim = np.zeros((T, K + 1), dtype=np.float64)
    cum_sim[:, 1:] = np.cumsum(sim, axis=1)

    dp = np.full((T + 1, K + 1), INF, dtype=np.float64)
    prev = np.full((T + 1, K + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0

    for t in range(1, T + 1):
        # Token (t-1) ends at segment index (j-1), having started at index (k-1).
        # Need k >= t (each prior token at least one segment), j >= k+1, j <= K.
        j_max = K - (T - t)
        for j in range(t, j_max + 1):
            best = INF
            best_k = -1
            for k in range(t, j + 1):
                if dp[t - 1, k - 1] >= INF:
                    continue
                # Negative similarity term
                neg_sim = -(cum_sim[t - 1, j] - cum_sim[t - 1, k - 1])
                # Inter-segment gap inside [k-1 .. j-1]
                gap = 0.0
                if j - 1 > k - 1:
                    gap = float(np.sum(seg_starts[k:j] - seg_ends[k - 1:j - 1]).clip(min=0))
                # Coverage: deviation of group duration from target
                grp_dur = seg_ends[j - 1] - seg_starts[k - 1]
                cov = abs(grp_dur - target_dur)
                cost = dp[t - 1, k - 1] + neg_sim + gap_penalty * gap + coverage_penalty * cov
                if cost < best:
                    best = cost
                    best_k = k
            dp[t, j] = best
            prev[t, j] = best_k

    # Final: choose j* in [T..K] minimising dp[T, j]
    j_star = int(np.argmin(dp[T, T:K + 1])) + T
    if dp[T, j_star] >= INF:
        return []

    # Backtrack
    ranges: list[tuple[int, int]] = []
    j = j_star
    for t in range(T, 0, -1):
        k = int(prev[t, j])
        ranges.append((k - 1, j - 1))
        j = k - 1
    ranges.reverse()
    return ranges


# ── EAF / VTT writers ────────────────────────────────────────────────────────

def seconds_to_vtt(s: float) -> str:
    if s < 0:
        s = 0
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def write_predictions_vtt(predictions: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, p in enumerate(predictions, 1):
            f.write(f"{i}\n{seconds_to_vtt(p['start'])} --> {seconds_to_vtt(p['end'])}\n{p['token']}\n\n")


def inject_eaf_tier(
    base_eaf: Path,
    out_eaf: Path,
    predictions: list[dict],
    tier_id: str = "GLOSS_LABEL_PRED",
) -> None:
    out_eaf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_eaf, out_eaf)

    tree = ET.parse(out_eaf)
    root = tree.getroot()
    time_order = root.find("TIME_ORDER")
    if time_order is None:
        raise ValueError("TIME_ORDER missing")

    # Drop existing tier with same id
    for old in root.findall("TIER"):
        if old.get("TIER_ID") == tier_id:
            root.remove(old)

    # Ensure linguistic-type
    lt_ids = {lt.get("LINGUISTIC_TYPE_ID") for lt in root.findall("LINGUISTIC_TYPE")}
    if "subtitle-lt" not in lt_ids:
        root.append(ET.Element("LINGUISTIC_TYPE", {
            "GRAPHIC_REFERENCES": "false",
            "LINGUISTIC_TYPE_ID": "subtitle-lt",
            "TIME_ALIGNABLE": "true",
        }))

    new_tier = ET.Element("TIER", {
        "LINGUISTIC_TYPE_REF": "subtitle-lt",
        "TIER_ID": tier_id,
    })
    for i, p in enumerate(predictions):
        ts1_id = f"{tier_id}_ts{i*2}"
        ts2_id = f"{tier_id}_ts{i*2+1}"
        time_order.append(ET.Element("TIME_SLOT", {"TIME_SLOT_ID": ts1_id, "TIME_VALUE": str(int(p["start"] * 1000))}))
        time_order.append(ET.Element("TIME_SLOT", {"TIME_SLOT_ID": ts2_id, "TIME_VALUE": str(int(p["end"] * 1000))}))
        ann_w = ET.SubElement(new_tier, "ANNOTATION")
        aa = ET.SubElement(ann_w, "ALIGNABLE_ANNOTATION", {
            "ANNOTATION_ID": f"{tier_id}_a{i}",
            "TIME_SLOT_REF1": ts1_id,
            "TIME_SLOT_REF2": ts2_id,
        })
        ET.SubElement(aa, "ANNOTATION_VALUE").text = p["token"]
    root.append(new_tier)
    tree.write(str(out_eaf), encoding="utf-8", xml_declaration=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Task 2 prototype: per-Gloss-sentence token-level alignment.")
    p.add_argument("--eaf",          type=Path, default=DEFAULT_EAF)
    p.add_argument("--seg-eaf",      type=Path, default=DEFAULT_SEG_EAF)
    p.add_argument("--sign-emb",     type=Path, default=DEFAULT_SIGN_EMB)
    p.add_argument("--out-csv",      type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--out-vtt",      type=Path, default=DEFAULT_OUT_VTT)
    p.add_argument("--out-eaf",      type=Path, default=DEFAULT_OUT_EAF)
    p.add_argument("--cache",        type=Path, default=DEFAULT_CACHE)
    p.add_argument("--model-name",   default="multilingual")
    p.add_argument("--language-tag", default="<en> <bfi>")
    p.add_argument("--gap-penalty",      type=float, default=2.0)
    p.add_argument("--coverage-penalty", type=float, default=0.5)
    p.add_argument("--window-pad",       type=float, default=0.5,
                   help="Seconds added to sentence window if no candidate segments fall inside")
    p.add_argument("--merge-gap-ms",     type=float, default=0.0,
                   help="If >0, merge consecutive SIGN segments separated by less than this many ms")
    args = p.parse_args()

    print(f"Loading Gloss sentences from {args.eaf.name} ...")
    sentences = load_gloss_sentences(args.eaf)
    n_tokens = sum(len(s["tokens"]) for s in sentences)
    print(f"  {len(sentences)} sentences, {n_tokens} total tokens")

    print(f"Loading SIGN segments from {args.seg_eaf.name} ...")
    segs = load_sign_segments(args.seg_eaf)
    print(f"  {len(segs)} sign segments")

    if args.merge_gap_ms > 0:
        merged: list[dict] = []
        for s in segs:
            if merged and (s["start_ms"] - merged[-1]["end_ms"]) < args.merge_gap_ms:
                merged[-1]["end_ms"] = s["end_ms"]
                merged[-1]["end"] = s["end"]
                merged[-1]["mid"] = (merged[-1]["start_ms"] + merged[-1]["end_ms"]) / 2000.0
            else:
                merged.append(dict(s))
        print(f"  merged with gap < {args.merge_gap_ms} ms -> {len(merged)} segments")
        segs = merged

    print(f"Loading sign embeddings from {args.sign_emb.name} ...")
    sign_embs = np.load(args.sign_emb).astype(np.float32)
    print(f"  shape = {sign_embs.shape}")
    if sign_embs.shape[0] != len(segs) and args.merge_gap_ms == 0:
        print(f"  [warn] sign_embs rows ({sign_embs.shape[0]}) != #segments ({len(segs)})")
    if args.merge_gap_ms > 0:
        print("  [warn] merge-gap-ms > 0: sign_embs no longer 1:1 with segments; falling back to per-segment max-pool would be ideal. Disabling merge for safety.")
        # Reload unmerged segs
        segs = load_sign_segments(args.seg_eaf)

    # Embed all unique tokens in one pass (cached)
    flat_tokens: list[str] = []
    for s in sentences:
        flat_tokens.extend(s["tokens"])
    print(f"Embedding {len(flat_tokens)} tokens (model={args.model_name}, lang_tag={args.language_tag!r}) ...")
    token_embs = embed_tokens_cached(flat_tokens, args.model_name, args.language_tag, args.cache)
    print(f"  token_embs.shape = {token_embs.shape}")

    # Per-sentence DP
    seg_starts_arr = np.array([s["start"] for s in segs], dtype=np.float64)
    seg_ends_arr   = np.array([s["end"]   for s in segs], dtype=np.float64)
    seg_mids_arr   = np.array([s["mid"]   for s in segs], dtype=np.float64)

    predictions: list[dict] = []
    fallback_count = 0
    n_processed = 0
    flat_idx = 0

    for s_idx, sent in enumerate(sentences):
        T = len(sent["tokens"])
        s, e = sent["start"], sent["end"]

        cand_mask = (seg_mids_arr >= s) & (seg_mids_arr <= e)
        if cand_mask.sum() < T:
            cand_mask = (seg_mids_arr >= s - args.window_pad) & (seg_mids_arr <= e + args.window_pad)
        cand_idx = np.where(cand_mask)[0]
        K = len(cand_idx)

        sent_token_embs = token_embs[flat_idx:flat_idx + T]

        if K >= T and K > 0:
            cand_sign_embs = sign_embs[cand_idx]
            # Cosine
            tn = sent_token_embs / (np.linalg.norm(sent_token_embs, axis=1, keepdims=True) + 1e-12)
            cn = cand_sign_embs  / (np.linalg.norm(cand_sign_embs,  axis=1, keepdims=True) + 1e-12)
            sim = tn @ cn.T  # (T, K)
            # Row-softmax with mild temperature for numerical stability
            sim = sim - sim.max(axis=1, keepdims=True)
            sim = np.exp(sim) / (np.exp(sim).sum(axis=1, keepdims=True) + 1e-12)

            ranges = monotonic_token_dp(
                sim,
                seg_starts_arr[cand_idx],
                seg_ends_arr[cand_idx],
                sentence_dur=e - s,
                gap_penalty=args.gap_penalty,
                coverage_penalty=args.coverage_penalty,
            )
        else:
            ranges = []

        if not ranges:
            # Fallback: split [s, e] uniformly across T tokens
            fallback_count += 1
            step = (e - s) / max(T, 1)
            for t_idx, tok in enumerate(sent["tokens"]):
                ts = s + t_idx * step
                te = s + (t_idx + 1) * step
                predictions.append({
                    "start": ts,
                    "end":   te,
                    "token": tok,
                    "sentence_idx": s_idx,
                    "token_idx":    t_idx,
                    "score": float("nan"),
                    "fallback": "uniform",
                })
        else:
            for t_idx, ((k0, k1), tok) in enumerate(zip(ranges, sent["tokens"])):
                gi0, gi1 = cand_idx[k0], cand_idx[k1]
                tok_score = float(np.sum(sim[t_idx, k0:k1 + 1]))
                predictions.append({
                    "start": float(seg_starts_arr[gi0]),
                    "end":   float(seg_ends_arr[gi1]),
                    "token": tok,
                    "sentence_idx": s_idx,
                    "token_idx":    t_idx,
                    "score": tok_score,
                    "fallback": "",
                })

        flat_idx += T
        n_processed += 1

    print(f"\nProcessed {n_processed} sentences.")
    print(f"  fallback_uniform sentences: {fallback_count}")
    print(f"  predicted labels: {len(predictions)}")

    # Write CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "start_s", "end_s", "gloss_token",
            "sentence_idx", "token_idx", "score", "fallback",
        ])
        writer.writeheader()
        for p_ in predictions:
            writer.writerow({
                "start_s": round(p_["start"], 3),
                "end_s":   round(p_["end"],   3),
                "gloss_token": p_["token"],
                "sentence_idx": p_["sentence_idx"],
                "token_idx":    p_["token_idx"],
                "score": "" if (p_["score"] != p_["score"]) else round(p_["score"], 4),  # NaN -> ''
                "fallback": p_["fallback"],
            })
    print(f"[OK] CSV  -> {args.out_csv}")

    # Companion VTT (sorted by start; ELAN-friendly text-only)
    sorted_preds = sorted(predictions, key=lambda x: (x["start"], x["end"]))
    write_predictions_vtt(sorted_preds, args.out_vtt)
    print(f"[OK] VTT  -> {args.out_vtt}")

    # EAF tier injection
    inject_eaf_tier(args.seg_eaf, args.out_eaf, sorted_preds, tier_id="GLOSS_LABEL_PRED")
    print(f"[OK] EAF  -> {args.out_eaf}")


if __name__ == "__main__":
    main()
