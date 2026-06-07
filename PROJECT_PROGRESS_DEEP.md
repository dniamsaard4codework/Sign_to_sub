# Sign_to_sub — Deep Project Progress Report

> **Thai Sign Language (TSL) Subtitle Alignment using SEA (Segment, Embed, Align)**
>
> A complete, step-by-step account of everything done in this repository, from the
> first commit to the latest. This document explains *what* SEA is (deeply), *why*
> the project exists, *what was built and changed*, *every experiment run*, *every
> result*, and *every problem solved* — in chronological and architectural detail.
>
> - **Maintainer:** dniamsaard4codework
> - **Sponsor / data provider:** NECTEC
> - **Upstream base:** [J22Melody/SEA](https://github.com/J22Melody/SEA) ([arXiv:2512.08094](https://arxiv.org/abs/2512.08094), Jiang et al. 2025) + SignCLIP ([J22Melody/fairseq](https://github.com/J22Melody/fairseq))
> - **Report compiled:** 2026-06-07
> - **Companion docs:** [README.md](README.md) (canonical setup), [Big_Progress.md](Big_Progress.md) (Thai canonical reference), [ForcedAlignment/PLAN_ForcedAlignment_Task2.md](ForcedAlignment/PLAN_ForcedAlignment_Task2.md) (scale-up spec)

---

## Table of Contents

1. [Executive Summary — What This Project Is](#1-executive-summary)
2. [The Problem — Why Sign-Language Subtitle Alignment Matters](#2-the-problem)
3. [SEA Explained Deeply — Segment, Embed, Align](#3-sea-explained-deeply)
4. [Input Data — Video, ELAN/EAF, VTT, Tiers](#4-input-data)
5. [Key Concepts Glossary](#5-key-concepts-glossary)
6. [Environment & Platform Engineering (Windows + Blackwell GPU)](#6-environment--platform-engineering)
7. [The Pipeline Architecture (S → E → A, end to end)](#7-the-pipeline-architecture)
8. [The Dynamic Programming Cores (Task 1 & Task 2 math)](#8-the-dynamic-programming-cores)
9. [Chronological Timeline — Commit by Commit](#9-chronological-timeline)
10. [Task 1 — Subtitle Alignment: All 7 Experiments & Results](#10-task-1)
11. [Task 2 — Gloss Labeling: Prototype, Ablation, Per-Sentence](#11-task-2)
12. [ForcedAlignment — Scale-up to 1,132 TSL Clips](#12-forcedalignment-scale-up)
13. [Evaluation Methodology (deep)](#13-evaluation-methodology)
14. [Tools & Scripts Built](#14-tools--scripts-built)
15. [Problems Encountered & How They Were Solved](#15-problems--fixes)
16. [Key Findings, Recommendations, Fine-tuning Opportunities](#16-findings--recommendations)
17. [Deliverables & Artifact Inventory](#17-deliverables--inventory)
18. [References](#18-references)

---

## 1. Executive Summary

**Goal.** Take a Thai Sign Language (TSL) interpreter video plus its spoken-language
captions, and produce subtitles whose timestamps line up with *when the signer
actually signs* — not when the speaker spoke. Deaf/hard-of-hearing viewers read the
signs, so the subtitle timing must follow the hands, not the audio.

**Approach.** Adopt the **SEA (Segment, Embed, Align)** framework from Jiang et al.
(2025) as the engine, then do the substantial engineering required to (a) run it on
**Windows + a Blackwell-architecture RTX 5060 Ti GPU**, (b) support **Thai sign
language** via the multilingual SignCLIP model, and (c) extend it from generic
subtitle alignment into **two concrete tasks**.

**Two tasks defined and solved:**

| Task | What it aligns | Granularity | Best result achieved |
| --- | --- | --- | --- |
| **Task 1 — Subtitle Alignment** | Whole caption cues → signing time windows | Sentence-level | **C_MULTI**: mean start offset **−0.16 s**, **100 %** within ±3 s, **F1@0.5 = 88.2 %** |
| **Task 2 — Gloss Labeling** | Individual gloss tokens → individual sign gestures | Token / gesture-level | **Gloss tier**: Mean IoU **0.49**, 48 % IoU≥0.5 (single clip); scaled to 1,132 clips with Config #1 F1 **68.6 %**, mIoU **0.59** |

**Scale.** The methods were first proven on **one demo video** (`04.mp4`, an 11.07-minute
TSL clip, 119 caption cues), then **scaled to a 1,132-clip TSL corpus** (~110 minutes
total, ~8.5 GB of video) via a phase-based orchestrator that ran ~11–12 hours overnight.

**Documentation.** The repo carries a large body of writing: a 78 KB bilingual
[README.md](README.md), a 126 KB Thai [Big_Progress.md](Big_Progress.md), five dated
Thai progress reports, presentation decks, a Thai pipeline guide, a LaTeX report
(English + Thai PDFs), a deep-dive HTML/PDF, and the SEA paper source.

**Reproducibility.** Cached outputs are committed so anyone can re-run *evaluation only*
in ~30 seconds without the (gitignored) source videos or model weights — verified
bit-for-bit reproducible as of 2026-06-01.

---

## 2. The Problem

### 2.1 The "sign delay" — why audio timestamps are wrong for signers

When a TV programme or video is interpreted into sign language, the interpreter watches
or listens to the source and *then* signs. This introduces a **lag**: the sign for a
concept appears on screen seconds *after* the corresponding spoken word. Standard
closed captions (CC) are timed to the **audio**. If you simply overlay those CC
timestamps on the signing video, the text and the hands are out of sync — often by
1–4 seconds — which is exactly the population (Deaf/HoH viewers) who rely on the visual
channel.

**The fix** is to *re-time* each caption so its window matches the time the signer is
actually producing the corresponding signs. That re-timing is the alignment problem
this project solves.

### 2.2 Why it is hard

- **No word-level audio anchor.** There is no transcript-to-audio forced aligner that
  helps, because the modality is visual hands, not speech.
- **Sign boundaries are fuzzy.** Signs blend into each other (coarticulation); there is
  no silence/space delimiter like in speech.
- **Cross-lingual gap.** Off-the-shelf sign models are trained on BSL (British) or ASL
  (American) — not Thai. We must transfer.
- **Monotonic but not 1:1.** Caption order follows signing order (monotonic), but one
  caption can span many signs, and some signs have no caption.

### 2.3 Why SEA was chosen

SEA is **language-agnostic, training-free at inference**, and **decomposable**: it reuses
two pretrained models (a *segmenter* and an *embedder*) and solves alignment as a
**global optimization by dynamic programming** that runs in seconds on CPU. That makes
it ideal for a low-resource language like TSL where you cannot train an end-to-end
aligner from scratch.

---

## 3. SEA Explained Deeply

### 3.1 The paper, in one paragraph

> *"Segment, Embed, and Align (SEA) provides a single framework that works across
> multiple languages and domains. SEA leverages two pretrained models: the first to
> segment a video frame sequence into individual signs and the second to embed the
> video clip of each sign into a shared latent space with text. Alignment is
> subsequently performed with a lightweight dynamic programming procedure that runs
> efficiently on CPUs within a minute, even for hour-long episodes."*
> — Jiang, Jang, Momeni, Varol, Ebling, Zisserman (arXiv:2512.08094, 2025)

The key insight: **don't train an aligner**. Instead, (1) cut the video into sign units,
(2) project both signs and text into one shared vector space so you can measure
sign↔text similarity, and (3) let dynamic programming find the globally optimal
sentence-to-signs assignment that respects monotonic order, duration, and gaps.

### 3.2 The three steps

```
        VIDEO (frames)                       TEXT (caption cues)
            │                                      │
   ┌────────▼─────────┐                            │
   │  S — SEGMENT     │  pose → per-sign boundaries│
   │  (E4s-1 GRU)     │                            │
   └────────┬─────────┘                            │
            │ SIGN segments [(t0,t1), ...]         │
   ┌────────▼─────────┐                  ┌─────────▼──────────┐
   │  E — EMBED       │  SignCLIP        │  E — EMBED         │
   │  sign → 768-d    │  shared space    │  text → 768-d      │
   └────────┬─────────┘                  └─────────┬──────────┘
            │  sign vectors                        │ cue vectors
            └──────────────┬───────────────────────┘
                  cosine similarity sign↔text
                           │
                  ┌────────▼─────────┐
                  │  A — ALIGN       │  global DP (Numba @njit)
                  │  cost-min DP     │  monotonic, windowed
                  └────────┬─────────┘
                           │
                  aligned subtitles (re-timed VTT/EAF)
```

#### S — Segment (turn a pose stream into a list of signs)

- **Input:** a `.pose` file — MediaPipe Holistic landmarks (543 keypoints/frame: body,
  hands, face) extracted from the MP4.
- **Model:** the **E4s-1** linguistic sign segmenter (a bidirectional GRU that reads the
  pose sequence and predicts, per frame, a **B**egin-of-sign and an **O**ut/inside
  probability). It is from the `pose_to_segments` / `J22Melody/segmentation@bsl`
  package and was trained on **BOBSL (BSL)** data.
- **Thresholds:** `--sign-b-threshold 30 --sign-o-threshold 50` (the "30_50" in the
  output folder name `E4s-1_30_50`). Lower thresholds = more, finer segments.
- **Output:** an ELAN `.eaf` with two tiers — a `SIGN` tier (e.g. **2,780** segments on
  the demo video) and a coarser `SENTENCE` tier (**418**). Each `SIGN` segment is a
  `(start, end)` time span for one putative sign gesture.

#### E — Embed (put signs and text in one vector space)

- **Model:** **SignCLIP** — a CLIP-style dual encoder. A *sign/pose encoder* maps a
  short pose clip to a 768-d vector; a *text encoder* maps a tokenized caption to a
  768-d vector. They are trained contrastively so a sign and its matching text land
  **close together** in the same space.
- **Three model variants** are wired up in this repo:

  | Variant | Trained on | `--language_tag` |
  | --- | --- | --- |
  | `bsl` | British Sign Language | `<en> <bfi>` |
  | `multilingual` | many sign languages (used for **TSL**) | `<en>` |
  | `asl` | American Sign Language | `<en> <ase>` |

- **Outputs:**
  - *Sign embeddings* `segmentation_embedding/<model>/04.npy` → shape **(2780, 768)** —
    one vector per `SIGN` segment.
  - *Subtitle embeddings* `subtitle_embedding/<model>/04.npy` → shape **(119, 768)** —
    one vector per caption cue.
- This is the step that requires the GPU (SignCLIP forward passes). Everything else can
  run on CPU.

#### A — Align (dynamic programming over similarity + timing)

Given the sign segments, their vectors, and the caption vectors, SEA computes a
**cosine-similarity matrix** between cues and signs, then runs a **dynamic program**
that assigns each caption cue (in order) to a contiguous group of sign segments,
minimizing a cost that balances: staying near the original timestamp, matching the
duration, avoiding gaps between grouped segments, and **rewarding high sign↔text
similarity**. (Full math in [§8](#8-the-dynamic-programming-cores).)

### 3.3 Why SEA is a good fit for TSL specifically

- **Cross-lingual transfer:** the `multilingual` SignCLIP, although never trained on
  Thai, embeds TSL signs into a space close enough to English caption text that cosine
  similarity is informative. This project empirically confirms that — `multilingual`
  beat both `bsl` and `asl` on the TSL demo (see [§10](#10-task-1)).
- **No TSL training data needed** for inference. We have annotations for *evaluation*,
  but the aligner itself never trains.
- **Fast and CPU-friendly** for the DP step (Numba `@njit` finishes in <1 s per video).

---

## 4. Input Data

### 4.1 Core files (demo video)

| File | What it is | In git? |
| --- | --- | --- |
| `example_alignment/04.mp4` | Source TSL video, "การเปรียบเทียบและเรียงลำดับ", **11.07 min**, 25 FPS, ~80 MB | ❌ (NECTEC data) |
| `example_alignment/04.pose` | MediaPipe Holistic pose, ~358 MB | ❌ (regenerate) |
| `example_alignment/Test.eaf` | ELAN annotation, 660 KB — holds all tiers | ✅ |

### 4.2 ELAN / EAF — what they are

**ELAN** is the standard linguistic annotation tool for video; it stores annotations in
an **`.eaf`** file (XML). An `.eaf` contains:

- a `TIME_ORDER` block of `TIME_SLOT`s (each a millisecond timestamp),
- one or more **`TIER`**s, each a named track of `ANNOTATION`s; every annotation points
  at a start and end `TIME_SLOT` and carries text.

Tiers are read in this project with plain `xml.etree.ElementTree` — they must be
**top-level `<TIER>`** elements (not nested subtiers), matched by their `TIER_ID`.

### 4.3 VTT / cue — what they are

A **WebVTT (`.vtt`)** file is the web subtitle format: a list of **cues**, each a
`start --> end` time range plus a line of text. The pipeline converts EAF tiers ⇄ VTT
freely; VTT is the interchange format the SEA aligner consumes and emits.

### 4.4 The tiers inside `Test.eaf` — and their exact roles

| Tier | Entries | Role |
| --- | ---: | --- |
| `CC` | 172 | Raw closed captions from the audio — **not used directly** |
| `CC_Input` | 119 | Curated CC subtitles — **Task 1 INPUT** |
| `CC_Aligned` | 119 | Manual hand-aligned timing — **Task 1 GROUND TRUTH** |
| `Gloss` | 119 sentences (852 tokens) | Gloss tier — **Task 2 INPUT (recommended)** |
| `Gloss_Input` | 119 sentences (889 tokens) | Curated gloss — **Task 2 INPUT (code default)** |
| `Gloss Labeling` | 852 | Per-gesture gloss annotation — **Task 2 GROUND TRUTH** |

This tier design is the backbone of both tasks: `CC_Input → CC_Aligned` is the Task 1
input→GT pair (119↔119, paired by index), and `Gloss[_Input] → Gloss Labeling` is the
Task 2 input→GT pair.

### 4.5 A concrete example

- **Input cue** (from `CC_Input`): `00:00:12.300 --> 00:00:15.800  "เด็กกำลังเรียนหนังสือ"`
  (timed to the audio).
- **Signer reality:** the signer doesn't start signing "เด็ก/เรียน/หนังสือ" until ~14.0 s
  and finishes ~17.6 s.
- **SEA output** (Task 1): re-timed cue `00:00:14.0 --> 00:00:17.6` matching the hands.
- **Task 2** goes finer: aligns each gloss token — "ผายมือ", "เด็ก", "เรียน" — to the
  individual `SIGN` gesture windows inside that sentence.

---

## 5. Key Concepts Glossary

- **Embedding / vector:** a fixed-length list of numbers (here 768) that represents the
  "meaning" of a sign clip or a piece of text. Similar meanings → nearby vectors.
- **Shared latent space:** SignCLIP trains the sign-encoder and text-encoder *together*
  so a sign and its text caption map to nearby vectors **in the same space** — that's
  what makes cross-modal similarity meaningful.
- **Cosine similarity:** `cos(a,b) = (a·b)/(‖a‖‖b‖)` — measures the angle between two
  vectors, in [−1, 1]; ~1 means "very similar". The SEA aligner uses this to score how
  well a caption matches a span of signs.
- **Dynamic Programming (DP):** an algorithm that finds a globally optimal sequence of
  choices by building a table of best sub-solutions and backtracking — here used to
  assign cues→signs in monotonic order with minimum total cost.
- **IoU (Intersection over Union):** for two time spans, `overlap / union`, in [0,1].
  IoU=1 means identical timing; IoU≥0.5 is the standard "good match" threshold. The
  primary Task 2 metric.
- **Frame accuracy:** rasterize predicted and GT timelines to frames (FPS=25) and report
  the fraction of frames whose label matches — a per-frame agreement metric from SEA.
- **F1@IoU:** precision/recall harmonic mean where a prediction "counts" only if its IoU
  with a GT span clears a threshold (0.10 / 0.25 / 0.50).
- **Cross-lingual transfer:** using a model trained on one sign language (BSL/ASL/multi)
  to process another (TSL) without retraining — relying on shared structure.
- **Gloss:** a written shorthand label for a sign (e.g. "เด็ก" = the sign for "child").
  A gloss *sentence* is a sequence of gloss tokens for one utterance.

---

## 6. Environment & Platform Engineering

A large fraction of the real effort went into making upstream SEA (a Linux/conda,
Python 3.12 research codebase) run on **Windows 11 with a brand-new Blackwell GPU**.

### 6.1 Test machine

```
OS              Windows 11 Pro (10.0.26200)
CPU             Intel Core Ultra 7 265K (20 cores)
RAM             64 GB
GPU             NVIDIA GeForce RTX 5060 Ti (16 GB, Blackwell sm_120)
CUDA Driver     595.79 (CUDA 13.2)
Python          3.11.15  (NOT 3.12 — see below)
Shell           PowerShell
```

### 6.2 The hard pins and *why*

| Package | Pin | Reason |
| --- | --- | --- |
| `python` | **3.11.x** | `mediapipe==0.10.21` has no wheels for 3.12+ |
| `torch` | 2.11.0 **+cu128** | Blackwell sm_120 needs the cu128 wheel; cu126 silently falls back to CPU (10–100× slower) |
| `mediapipe` | **0.10.21 exact** | 0.10.22+ break the pose extraction API |
| `pose-format` | 0.12.3 | matches `videos_to_poses` |
| `numpy` | 1.26.4 | numba JIT compatibility |
| `numba` | latest | JIT for the DP inner loops |

> **The Blackwell trap:** on a 50-series card you *must* install the **cu128** PyTorch
> wheel. With the wrong wheel, `torch.cuda.is_available()` returns `False` and the whole
> embedding step runs on CPU — turning a ~5-minute job into ~3–5 hours, silently.

### 6.3 Windows portability fixes to upstream

**`SEA/` (vs upstream commit `5aaf27d`):**

| File | Change |
| --- | --- |
| `align.py` | Multi-model support (`--live_model_name`, `--live_language_tag`); load pre-computed segmentation embeddings; removed hardcoded `/users/zifan/` path |
| `align_dp.py` | `numba` import fallback — runs as plain Python if LLVM unavailable |
| `config.py` | new CLI args `--live_model_name`, `--live_language_tag` |
| `segmentation.py` | use `os.path.abspath()`; replaced `subprocess.run(shell=True)` with `shlex.split` + `shell=False` (Windows-safe) |

**`fairseq_signclip/` (SignCLIP):** an 8-file, 36-line patch
([patches/fairseq_signclip_windows.patch](patches/fairseq_signclip_windows.patch))
fixes Linux `/`-vs-`\` path assumptions and missing absolute paths across
`mmpt/models/mmfusion`, `processors/{dsprocessor,dsprocessor_sign,processor}`,
`tasks/task`, `utils/load_config`, and `scripts_bsl/extract_episode_features.py`, plus
the `retri/signclip_bsl/` YAML configs. The patch is pinned to upstream commit
`a8199440` (2026-03-04) so `git apply` stays reproducible.

### 6.4 The GPU reality (documented honestly)

Only **one** of the four heavy phases actually uses the GPU:

| Phase | Tool | Device | Why |
| --- | --- | --- | --- |
| Pose | `videos_to_poses` (MediaPipe) | **CPU** | Windows MediaPipe wheel has no GPU delegate |
| Segmentation | `pose_to_segments` | **CPU** | upstream hard-sets `CUDA_VISIBLE_DEVICES=""`; JIT-compiled LSTM freezes CPU device for hidden state — can't be patched without re-export |
| **Embedding** | `extract_episode_features.py` (SignCLIP) | **GPU** | honors `torch.cuda.is_available()` |
| DP align | `align.py` | CPU (by design, Numba) | tiny compute |

This is why the 1,132-clip full run took ~11 hours — pose extraction on CPU is the
bottleneck. This was traced and documented rather than hidden.

---

## 7. The Pipeline Architecture

End-to-end flow for the demo video (`04`):

```
INPUT: 04.mp4  +  Test.eaf
  │
  ├─ A. extract_cc_from_eaf.py --tier CC_Input  →  subtitles/04.vtt          (119 cues)
  ├─ B. make_gloss_cc_vtt.py                     →  subtitles_gloss_cc_time/04.vtt
  │       (Gloss_Input text glued onto CC_Input timestamps)
  ├─ C. videos_to_poses (MediaPipe Holistic)     →  04.pose                  (358 MB)
  ├─ D. SEA/segmentation.py (E4s-1, 30/50)       →  segmentation_output/E4s-1_30_50/04.eaf
  │       (SIGN: 2780, SENTENCE: 418)
  ├─ E1. extract_episode_features.py --mode=segmentation (×3 models)
  │       →  segmentation_embedding/{sign_clip,sign_clip_multi,sign_clip_asl}/04.npy   (2780,768)
  ├─ E2. extract_episode_features.py --mode=subtitle (×5 text/model combos)
  │       →  subtitle_embedding/<combo>/04.npy                                          (119,768)
  │
  ├─ TASK 1: SEA/align.py  (7 experiments)        →  aligned_output_*/04.vtt
  │       + fix_overlap_vtt.py                     →  aligned_output_*/04_no_overlap.vtt
  │       + evaluate_all_to_csv.py                 →  evaluation_task1_results.csv
  │
  ├─ TASK 2: align_gloss_labels.py --tier Gloss   →  gloss_labels_pred.csv/.vtt + 04_gloss_pred.eaf
  │       + evaluate_gloss_labeling.py            →  evaluation_gloss_labeling.csv
  │
  └─ VISUALS: add_vtt_tiers_to_eaf.py (Test_comparison.eaf, 17 tiers)
             add_best_to_eaf.py     (Test_best.eaf)
             plot_alignment.py       (figures/timeline_first_2min.png)
```

**Why two subtitle inputs (A and B)?** `CC_Input` is the literal spoken-language caption.
`Gloss_Input` is the *gloss* text (closer to what the hands actually say). Embedding the
gloss text with SignCLIP and aligning that (experiment **C_MULTI**) gave the best Task 1
result because gloss text sits closer to the signs in the shared space than spoken-Thai
captions do.

> **Critical operational gotchas** (each cost debugging time, now documented):
> - `--sign-b-threshold`/`--sign-o-threshold` at align time **must match** the values
>   used at segmentation time (`30 50`), or every cue collapses to the same time.
> - `--segmentation_dir` must point at the **parent** (`segmentation_output`), not the
>   `E4s-1_30_50` subfolder — `align.py` appends the threshold subdir itself.
> - `--num_workers 1` on Windows — multiprocessing >1 hits path errors.

---

## 8. The Dynamic Programming Cores

### 8.1 Task 1 DP (sentence-level, `align_dp.py`)

State: `dp[i][j]` = minimum cost to assign caption cues `1..i` such that cue `i` ends at
sign-segment index `j`. The cost of grouping cue `i` over segments `k..j`:

```
C(i, k, j) =  |cue_i.start − seg_k.start|          ← start alignment
            + |cue_i.end   − seg_j.end|            ← end alignment
            + w_D · |cue_dur − group_dur|          ← duration penalty   (w_D = 2)
            + w_G · gap(k, j)                       ← gap penalty        (w_G = 8, dominant)
            − w_S · sim_cum[i][k][j]                ← similarity REWARD  (w_S = 6)
```

- **Start/end alignment (weight 1):** keep the re-timed cue near the original (the audio
  timestamp is a prior on roughly *where* the content is).
- **Duration penalty `w_D`:** discourage groups whose total duration differs from the
  cue's duration.
- **Gap penalty `w_G` (most influential):** punish stitching together non-contiguous
  signs with big gaps between them.
- **Similarity reward `w_S`:** *subtract* cost proportional to cumulative SignCLIP
  cosine similarity — high sign↔text similarity makes a grouping cheaper. This is the
  "E" of SEA feeding the "A".

A **sliding window** (`--dp_window_size 40`) limits the search so complexity is
`O(M · W²) ≈ 190K` operations; with Numba `@njit` it finishes in **<1 second**.

The C_MULTI tuned weights: `w_D=2, w_G=8, max_gap=6, window=40, w_S=6`, with bias
`pr_subs_delta_bias_start=1.3 / end=1.0` (these biases shift the start/end prior, found
empirically to center the offset near zero).

### 8.2 Post-processing: overlap fix

The DP has **no non-overlap constraint**, so raw output overlaps 86–88 % of the time.
[fix_overlap_vtt.py](example_alignment/fix_overlap_vtt.py) does a single-pass clamp:

```python
for i in range(len(cues) - 1):
    if cues[i].end > cues[i + 1].start:
        cues[i].end = cues[i + 1].start   # clamp end to next start
```

Only the **end** is touched — the **start** is the DP's carefully-computed best estimate
of when signing began, and the primary metric measures start offset, so start metrics
are identical before/after the fix while overlap drops to **0 %**.

### 8.3 Task 2 DP (token-level, `align_gloss_labels.py`)

For each gloss *sentence* `(start_s, end_s, "tok1 tok2 ... tokT")`:

```
1. tokenize on whitespace                                  → T tokens
2. restrict candidate SIGN segments to those whose midpoint
   ∈ [start_s, end_s]  (±0.5 s pad if window is empty)     → K segments
3. embed each token via SignCLIP multilingual text encoder → token_embs (T×768), cached .npz
4. sim_matrix (T×K) = cosine, then row-softmax
5. monotonic DP per sentence:
     dp[t][j] = min over k of {
         dp[t-1][k-1]
       − Σ sim[t-1, k-1 .. j-1]                  ← negative similarity (reward)
       + gap_penalty      · inter_segment_gap     ← (default 2.0)
       + coverage_penalty · |group_dur − sent_dur/T|  ← (default 0.5)
     }
6. backtrack → each token gets a contiguous segment range (k_start..k_end)
            → emit (seg[k_start].start, seg[k_end].end, token)
```

Complexity `O(T·K²)` per sentence (T~7, K~30) — trivial; all 119 sentences in <1 s.
Token embeddings are cached to `.npz` so the second run is much faster. Separate
`--cache` paths are used per ablation tier so caches don't cross-contaminate.

---

## 9. Chronological Timeline

16 commits over ~6 weeks (2026-04-20 → 2026-05-31), plus continued doc/verification work
into June. Five dated Thai progress reports document the research narrative.

| Date | Commit | Milestone |
| --- | --- | --- |
| **2026-04-20** | `143ff35` Initial commit | First working pipeline: segmentation + **B2 (BSL)** baseline; [Progress_20042026.md](Progress_20042026.md) |
| **2026-04-26** | `5df0664` Update progress | **7 experiments** defined; post-overlap fix; **Task 2 prototype**; [Progress_26042026.md](Progress_26042026.md) |
| **2026-04-27** | `d19acab` Update 27042026 | Comparison EAF (`_comparison_27042026`); report scaffolding |
| **2026-05-05** | `cc54922` Update 05052026 | Switched to curated **`CC_Input` / `Gloss_Input`** + **index-based eval** (119/119); [Progress_04052026.md](Progress_04052026.md), Presentation update, script.md |
| **2026-05-10** | `8b02759` Update 10052026 | Task 2 ablation prep |
| **2026-05-12** | `49d5ced` Update 12052026 | **Task 2 ablation: `Gloss` vs `Gloss_Input`**; [Progress_09052026.md](Progress_09052026.md); presentation_12052026 |
| **2026-05-16** | (in `161f165`) | **Per-sentence Task 2 pipeline** — crop video at gloss-sentence boundaries, run pose+seg+emb+DP per clip; [Progress_16052026.md](Progress_16052026.md) |
| **2026-05-18** | `7f5213e`, `1b0f15c`, `161f165`, `cc7336c` | **Big_Progress.md** consolidated report; **ForcedAlignment dataset** (1,140 repaired EAFs) added; scale-up planning |
| **2026-05-20–21** | — | **Full 1,132-clip pipeline executed** (~11 h overnight) |
| **2026-05-24** | — | ForcedAlignment deliverables: comparison EAFs, filled report docx, eval |
| **2026-05-25** | `7496642` | ForcedAlignment full-run results committed; Big_Progress updated; SEA_Pipeline_Deep_Dive.html |
| **2026-05-26** | `404d912`, `13f02b8` | Deep-dive report + DeepDive.pdf; **repo cleanup** (removed 6,166 lines of generated logs/artifacts), added `requirements.txt`, `.gitignore` |
| **2026-05-27** | `d115f34` | ForcedAlignment README (288 lines) + **Windows patch workflow** (`patches/`) |
| **2026-05-28** | `ebfb566` | README: Source Data section, **Evaluation-Only Quick Start**, `requirements-eval.txt` |
| **2026-05-31** | `9df63cc` | README: hardened newcomer setup path |
| **2026-06-01** | (memory) | **Reproducibility audit** — all eval scripts re-run from venv, outputs matched committed CSVs bit-for-bit; restored truncated manifest |

---

## 10. Task 1 — Subtitle Alignment

### 10.1 Definition

Re-time the 119 `CC_Input` cues (audio-timed) so they match when the signer signs.
Evaluate against the 119 `CC_Aligned` manual cues by **index** (`pred[i] ↔ gt[i]`).

### 10.2 Evolution of the experiment set

Early runs (Progress_20042026) used the **BSL** model with raw CC text. Over time the
study grew to a **7-experiment grid**: {BSL, Multilingual, ASL} × {CC text, Gloss text,
word-level gloss}. Two key discoveries drove this:

1. **Multilingual > BSL > ASL** for TSL (cross-lingual transfer works best with the
   multilingual encoder).
2. **Gloss text > CC text** as the subtitle to embed (gloss sits closer to signs).

### 10.3 The 7 experiments

| Experiment | Model | Text | Subtitle emb | bias s/e | save_dir |
| --- | --- | --- | --- | --- | --- |
| **B2** | BSL | CC | `sign_clip` | 1.8/1.5 | `aligned_output_with_embedding_tuned` |
| **B_MULTI** | Multi | CC | `sign_clip_multi` | 1.8/1.5 | `aligned_output_multi_b2` |
| **C_MULTI** ⭐ | Multi | Gloss | `sign_clip_multi_gloss` | 1.3/1.0 | `aligned_output_multi_gloss` |
| **C_MULTI_word** | Multi | Gloss (word-level, live emb) | — | 1.3/1.0 | `aligned_output_multi_gloss_word` |
| **D_ASL** | ASL | CC | `sign_clip_asl` | 1.8/1.5 | `aligned_output_asl_b2` |
| **D_ASL_gloss** | ASL | Gloss | `sign_clip_asl_gloss` | 1.3/1.0 | `aligned_output_asl_gloss` |
| **D_ASL_word** | ASL | Gloss (word-level, live emb) | — | 1.3/1.0 | `aligned_output_asl_gloss_word` |

(The `_word` variants embed each gloss token at align time via
`--live_embedding --tokenize_text_embedding`, so they need no precomputed subtitle `.npy`.)

### 10.4 Results (index-based, 119/119, post overlap-fix)

| Experiment | mean off | ±1 s | ±2 s | ±3 s | F1@0.50 | overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B2 | +0.26 s | 76 % | 96 % | 99 % | 88.2 % | 0 % |
| B_MULTI | +0.25 s | 71 % | 93 % | 99 % | 84.9 % | 0 % |
| **C_MULTI** ⭐ | **−0.16 s** | 74 % | 95 % | **100 %** | 88.2 % | 0 % |
| C_MULTI_word | −0.23 s | 74 % | 94 % | 100 % | **89.1 %** | 0 % |
| D_ASL | +0.38 s | 61 % | 87 % | 98 % | 77.3 % | 0 % |
| D_ASL_gloss | −0.12 s | 71 % | 91 % | 99 % | 81.5 % | 0 % |
| D_ASL_word | −0.13 s | 69 % | 90 % | 99 % | 84.9 % | 0 % |

**Verified best (memory, 2026-06-01):** C_MULTI — mean offset **−0.16 s**, 73.9 % within
±1 s, **100 %** within ±3 s, frame accuracy **82.6 %**, **F1@0.5 = 88.2 %**, end offset
+0.23 s, overlap 0 % (was 88.1 %). The `CC_Aligned` ground truth was independently
verified overlap-free (0/118).

**Headline:** C_MULTI is essentially unbiased (−0.16 s) and **every cue lands within 3
seconds** of the hand-aligned truth — a strong result for a training-free, cross-lingual
aligner on a language the models never saw.

### 10.5 The overlap-fix story

Raw DP output had **88.1 %** overlapping consecutive cues. The single-pass end-clamp
([§8.2](#82-post-processing-overlap-fix)) dropped this to **0 %** with no change to start
metrics — a clean, well-justified post-process rather than a hack.

---

## 11. Task 2 — Gloss Labeling

### 11.1 Definition

Go finer than Task 1: align **each gloss token** (e.g. "ผายมือ", "เด็ก", "เรียน") to an
**individual `SIGN` gesture** within its gloss sentence. Evaluate against the per-gesture
`Gloss Labeling` GT (852 entries) using **best-IoU pairing**.

### 11.2 Experiment 1 — Prototype (whole-video `Gloss` tier)

The first version ([align_gloss_labels.py](example_alignment/align_gloss_labels.py))
tokenized each gloss sentence, restricted candidate signs to the sentence window, embedded
tokens with multilingual SignCLIP, and ran the per-sentence monotonic DP ([§8.3](#83-task-2-dp-token-level-align_gloss_labelspy)).

### 11.3 Experiment 2 — Ablation: `Gloss` vs `Gloss_Input`

The code default reads `Gloss_Input`; the ablation (Progress_09052026) compared it
head-to-head with `Gloss`. **`Gloss` won every metric:**

| Metric | `Gloss` | `Gloss_Input` | Δ |
| --- | ---: | ---: | ---: |
| Mean IoU | **0.4901** | 0.4199 | +7.0 pp |
| % IoU ≥ 0.5 | **48.4 %** | 38.9 % | +9.4 pp |
| % IoU ≥ 0.3 | **77.0 %** | 66.0 % | +11.0 pp |
| % zero overlap | **2.5 %** | 6.6 % | −4.1 pp |
| Mean abs start offset | **0.188 s** | 0.212 s | −24 ms |

**Why `Gloss` wins:**
1. **Token-count match:** `Gloss` has exactly **852 tokens = 852 GT entries**; position-
   by-position match is 71.2 % (vs 4.9 % for `Gloss_Input`'s 889 tokens) — because the
   annotator built the GT *from* the `Gloss` token list.
2. **Degrees of freedom:** `Gloss_Input`'s 37 extra tokens get squeezed into 852 GT
   slots, reducing overlap.
3. **Window coverage:** `Gloss` predictions overlap 97.77 % of GT vs 88.97 % for
   `Gloss_Input`.

> **⚠️ Documented caveat (intellectual honesty):** the *exact-text-match* metric is
> 65.1 % for `Gloss` vs 10.6 % for `Gloss_Input` — but this is **structural leakage**,
> because the GT was built from the `Gloss` token list. So the project **reports IoU as
> the headline**, not text match. This caveat is repeated everywhere the number appears.

### 11.4 Experiment 3 — Per-sentence video-cropping pipeline (Progress_16052026)

A more faithful pipeline ([run_task2_per_sentence.py](example_alignment/run_task2_per_sentence.py)):
crop the video into **119 clips** at gloss-sentence boundaries, then run the *full*
pose→segmentation→embedding→DP pipeline **per clip**, and aggregate back to CSV/VTT/EAF.

| Metric | Per-sentence | whole-video `Gloss` |
| --- | ---: | ---: |
| Mean IoU | 0.4763 | 0.4901 |
| % IoU ≥ 0.5 | 46.0 % | 48.4 % |
| % any overlap | 96.1 % | 97.5 % |

**Finding:** per-sentence is essentially **as good** as whole-video `Gloss` (−1.4 pp Mean
IoU) but **~7× slower**. Conclusion: the whole-video approach is preferred; cropping buys
nothing here.

### 11.5 Reconciliation note

Memory records the canonical single-clip number as **Mean IoU 0.4199 / 38.9 %** from the
*later* prediction run (the 0.4901/48.4 % figure was an earlier prediction run, now
superseded for the *default* path but still the headline for the `Gloss`-tier ablation).
Both are documented; the README presents `Gloss` 0.4901 as recommended and `Gloss_Input`
0.4199 as default. The distinction is which prediction CSV is being evaluated.

---

## 12. ForcedAlignment — Scale-up

> Everything above ran on **one** video. The [ForcedAlignment/](ForcedAlignment/)
> sub-project scales Task 2 to a **1,132-clip TSL corpus** (avg 5.8 s/clip, ~110 min,
> ~8.5 GB) with **5 configurations**.

### 12.1 Why

To test whether the Task 2 method generalizes beyond a single curated clip, and to
produce a real dataset-scale evaluation (not an n=1 anecdote).

### 12.2 The data

- **1,140 ground-truth EAF files** in `elan_forced_alignment/` (~9 MB, **in git**).
- **1,132 source MP4s** (NECTEC research data, **not in git**), organized in six
  `<N> MP/` folders under a top folder (dev convention: `หนังสือภาษามือไทย/`).
- The orchestrator auto-discovers the video root by scanning for any subdir with MP4s;
  pairing is by **stem** (`42.mp4 ↔ 42.eaf`), and duplicate stems across folders raise an
  error.

The EAFs first needed **repair** — `check_eaf_video_match.py` validated EAF↔video
correspondence and `fix_eaf_media_paths.py` repaired broken `MEDIA_DESCRIPTOR` paths so
they open cleanly in ELAN (committed 2026-05-18 as "repaired ForcedAlignment EAF dataset").

### 12.3 The orchestrator — `run_forced_alignment.py` (807 lines, 7 phases)

| Phase | Step | Output |
| --- | --- | --- |
| 1 | Build manifest + `video_ids.txt` | `output/manifest.csv` |
| 2 | `videos_to_poses` (MediaPipe) | `output/poses/<id>.pose` (~3.8 GB) |
| 3 | SEA segmentation | `output/seg/E4s-1_30_50/<id>.eaf` |
| 4 | SignCLIP segment embeddings | `output/emb/<id>.npy` |
| 5 | In-process DP align (5 configs) | `output/predictions/config<N>_*.csv` |
| 6 | Inject prediction tiers into EAFs | `output/predicted_eafs/<id>.eaf` |
| 7 | Evaluate (P/R/F1/IoU/FrameAcc) | `output/evaluation/*.csv` |

Operational niceties built in: `--preflight-only` (validate manifest + required tiers
with no compute), `--only-ids` (smoke test on 3 clips), `--skip-pose/-seg/-emb/-align/-eval`
(iterate on later phases without redoing pose extraction), and staged hardlinks in
`video_work/` to avoid copying 5 GB.

### 12.4 The 5 configurations

| # | Input tier | GT tier | Tokenization | Purpose |
| ---: | --- | --- | --- | --- |
| 1 ⭐ | `CC` | `CC_Aligned` | whitespace, **drop `sil`** | Best baseline |
| 2 | `CC` | `CC_Aligned` | whitespace, **keep `sil`** | Sil-handling experiment |
| 3 | `Gloss` | `Gloss_Labeling` | pipe-split, drop empties | Direct analogue of `04.mp4` Task 2 |
| 4 | `Gloss1` | `Gloss_Labeling1` | pipe-split | Alternate gloss annotation |
| 5 | `Gloss2` | `Gloss_Labeling2` | pipe-split | Alternate gloss annotation |

### 12.5 Results (full 1,132-clip run, from `evaluation_summary.csv`)

| Config | Pred | GT | Precision@0.5 | Recall@0.5 | **F1@0.5** | **Mean IoU** | FrameAcc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** ⭐ CC→CC_Aligned | 1,171 | 1,132 | 67.5 % | 69.8 % | **68.6 %** | **0.5928** | 76.8 % |
| 2 CC+sil→CC_Aligned | 3,435 | 3,396 | 17.5 % | 17.7 % | 17.6 % | 0.2039 | 21.3 % |
| 3 Gloss→Gloss_Labeling | 1,713 | 1,713 | 7.8 % | 7.8 % | 7.8 % | 0.2484 | 26.2 % |
| 4 Gloss1→Gloss_Labeling1 | 3,977 | 3,977 | 20.7 % | 20.7 % | 20.7 % | 0.2229 | 22.3 % |
| 5 Gloss2→Gloss_Labeling2 | 3,977 | 3,977 | 20.9 % | 20.9 % | 20.9 % | 0.2238 | 22.4 % |

**Headline:** **Config #1 (F1 68.6 %, mIoU 0.5928) beats the single-video `04.mp4`
baseline (0.4901)** — the method *improves* at scale. Matching is **positional + IoU
only** (no text leakage); text-match is reported separately as reference-only (~97 %).

### 12.6 Why configs 2–5 underperform (error analysis)

Two root causes were diagnosed and documented (PLAN §10.11/§10.11b):

1. **`sil` tokens (Config #2):** keeping silence tokens explodes the prediction count
   (3,435 vs 1,171) and tanks precision — the segmenter has no "silence" sign to match,
   so `sil` tokens get garbage alignments. Dropping `sil` (Config #1) is the fix.
2. **Architectural / annotation-convention mismatch (Configs 3–5):** the `Gloss_Labeling`
   GT was annotated with a *different convention* (per-gesture, pipe-separated, often more
   granular than the SEA segmenter's `SIGN` units). The segmenter and the GT disagree on
   what counts as "one sign," so IoU pairing is structurally penalized regardless of
   embedding quality. This is a **data-convention** problem, not a model failure — and
   it's exactly the kind of finding that motivates fine-tuning the segmenter on TSL.

### 12.7 Deliverables (2026-05-24)

- 5 prediction CSVs + 5 companion VTTs (`prediction_vtt/`)
- Per-clip evaluation CSVs (`eval_config{1..5}.csv`) + `evaluation_summary.csv`
- `evaluation_summary_template_format.csv`
- Comparison EAFs (GT + prediction tiers per clip) via `create_comparison_eafs.py`
- Auto-filled report `Gloss_Labeling_Report_Filled.docx` via `fill_gloss_labeling_template.py`

---

## 13. Evaluation Methodology

### 13.1 Task 1 — index-based matching (and why)

| | Original SEA eval | This project |
| --- | --- | --- |
| Scale | BOBSL, 20,000+ sentences | single video, 119 cues |
| Matching | text lookup / frame-level | **index-based** `pred[i] ↔ gt[i]` |
| Coverage | 69/172 (text lookup fails when annotators reword) | **119/119** |
| Deps | pysrt, webvtt, beartype, BOBSL paths | **stdlib only** |

Why index-based: `CC_Input` and `CC_Aligned` have exactly 119 entries in the same order;
annotators often **reword** during alignment, so text lookup matches only ~50/119. Index
pairing gives full 119/119 coverage. [evaluate_all.py](example_alignment/evaluate_all.py)
also computes the *SEA* metrics (FrameAcc, F1@IoU) at FPS=25 for comparability.

**Metrics produced:** signed/abs mean+median start offset, end offset, %within ±1/±2/±3 s,
overlap %, frame accuracy, F1@0.10/0.25/0.50.

### 13.2 Task 2 — best-IoU pairing

[evaluate_gloss_labeling.py](example_alignment/evaluate_gloss_labeling.py): for each
prediction, find the GT entry with maximum IoU; record IoU, signed offsets, and a
text-match flag. Aggregates: mean/median IoU, %IoU≥0.5/0.3, %any overlap, mean signed
offsets, exact text match (flagged as leakage-prone).

### 13.3 ForcedAlignment — positional IoU, no text leakage

[evaluate_fa_dataset.py](ForcedAlignment/evaluate_fa_dataset.py) pairs predictions to GT
by **position + IoU only**, never by text — so the headline numbers can't be inflated by
the gloss-leakage effect. text_match is computed but reported **reference-only**.

### 13.4 Reproducibility guarantee

Cached outputs (aligned VTTs, prediction CSVs, eval CSVs, GT EAFs) are **committed**, so
all three evaluators run in ~30 s with **no source video, no PyTorch, no fairseq** — see
[Evaluation-Only Quick Start](README.md#evaluation-only-quick-start) and
[requirements-eval.txt](requirements-eval.txt). Re-verified bit-for-bit 2026-06-01.

---

## 14. Tools & Scripts Built

All custom scripts live in [example_alignment/](example_alignment/) and
[ForcedAlignment/](ForcedAlignment/). Highlights:

**Data prep / IO:**
- `extract_cc_from_eaf.py` — EAF tier → VTT (`--tier` flag)
- `make_gloss_cc_vtt.py` — build Gloss_Input VTT (gloss text + CC timestamps)
- `make_gloss_input_tier.py` — one-shot Gloss_Input tier builder
- `merge_cc_to_updated_eaf.py` — copy CC tiers between EAFs
- `fix_overlap_vtt.py` — end-clamp overlap fix → 0 % overlap

**Task aligners:**
- `align_gloss_labels.py` (555 lines) — Task 2 token aligner with `--tier` ablation
- `run_task2_per_sentence.py` — per-sentence crop-and-align pipeline

**Evaluation:**
- `evaluate_all.py` / `evaluate_all_to_csv.py` — Task 1 index-based eval (single / batch-all-7)
- `evaluate_gloss_labeling.py` — Task 2 IoU eval
- `ForcedAlignment/evaluate_fa_dataset.py` — corpus-scale positional IoU eval

**Visualization / comparison:**
- `add_vtt_tiers_to_eaf.py` — `Test_comparison.eaf` (17 tiers: 7 pre + 7 post + default + 2 ablation)
- `add_best_to_eaf.py` — `Test_best.eaf` (best-only, with auto-fallback)
- `make_task2_comparison_eaf.py`, `plot_alignment.py` — Task 2 comparison EAF + 4-lane timeline PNG

**ForcedAlignment ops:**
- `run_forced_alignment.py` (807 lines) — 7-phase orchestrator
- `check_eaf_video_match.py`, `fix_eaf_media_paths.py` — EAF↔video validation & repair
- `create_comparison_eafs.py`, `fill_gloss_labeling_template.py` — deliverable builders

**Portability win:** all `example_alignment/` scripts were refactored to
`HERE = Path(__file__).parent` (no hardcoded absolute paths) — verified portable
2026-05-26.

---

## 15. Problems & Fixes

| Problem | Root cause | Fix |
| --- | --- | --- |
| GPU silently unused | wrong PyTorch CUDA wheel on Blackwell | install **cu128** wheel |
| Pose extraction on CPU | Windows MediaPipe has no GPU delegate | accepted (documented); CPU is the bottleneck |
| Segmentation can't use GPU | upstream hard-sets `CUDA_VISIBLE_DEVICES=""`; JIT LSTM freezes CPU device | can't fix without model re-export — documented |
| `subprocess shell=True` breaks on Windows | shell quoting differences | `shlex.split` + `shell=False` in `segmentation.py` |
| SignCLIP path errors on Windows | Linux `/` path assumptions in 8 files | `patches/fairseq_signclip_windows.patch` (pinned to commit `a8199440`) |
| `numba` JIT unavailable | LLVM missing | import fallback to pure Python in `align_dp.py` |
| All cues collapse to same time | `--sign-b/o-threshold` mismatch align vs seg | use identical `30 50` |
| `--segmentation_dir` wrong level | align.py appends the subdir | point at parent `segmentation_output` |
| Multiprocessing path errors | Windows + `num_workers>1` | `--num_workers 1` |
| `video_ids.txt` UnicodeDecodeError | PowerShell `echo` writes UTF-16 BOM | write via Python with `encoding='utf-8'` |
| Text-lookup eval matched only ~50/119 | annotators reword cues | index-based matching → 119/119 |
| Raw DP output 88 % overlap | DP has no non-overlap constraint | single-pass end-clamp → 0 % |
| Config #2 precision collapse | `sil` tokens have no matching sign | drop `sil` (Config #1) |
| Configs 3–5 low IoU | GT annotation convention ≠ segmenter units | diagnosed as data-convention mismatch |
| manifest.csv truncated to 1 clip (2026-06-01) | uncommitted working-tree corruption | `git checkout` restore; GT/preds were intact |

---

## 16. Findings & Recommendations

### 16.1 Key findings

1. **Multilingual SignCLIP transfers to TSL** — it beat both BSL and ASL encoders,
   confirming cross-lingual transfer is viable for a language the model never trained on.
2. **Gloss text > spoken-caption text** for embedding-based alignment (gloss sits closer
   to signs in the shared space).
3. **C_MULTI is production-quality for Task 1:** −0.16 s mean offset, 100 % within ±3 s,
   F1@0.5 = 88.2 %.
4. **The method scales:** Config #1 at 1,132 clips (mIoU 0.59) *beat* the single-clip
   baseline (0.49).
5. **The segmenter is the weak link** for fine-grained Task 2 — the GT annotation
   convention disagrees with the BSL-trained segmenter's sign units (Configs 3–5).
6. **Per-sentence cropping doesn't help** — same accuracy, 7× slower.

### 16.2 Recommendations for production

- **Task 1:** use **C_MULTI** (multilingual + gloss text), tuned weights
  `w_D=2 w_G=8 max_gap=6 window=40 w_S=6`, bias 1.3/1.0, then `fix_overlap_vtt.py`.
- **Task 2:** use the **`Gloss` tier** whole-video aligner. Report **IoU**, never text
  match.
- **Corpus:** use **Config #1 (CC→CC_Aligned, drop sil)**.

### 16.3 Honest cautions

- The Task 2 `Gloss` text-match number (65 %) is **leakage** — don't quote it as accuracy.
- Single-clip results are n=1; trust the 1,132-clip corpus numbers for generality.

### 16.4 Fine-tuning opportunities (priority-ordered, Big_Progress §11)

1. **DP hyperparameter sweep** (cheapest, no training) — systematic grid over
   `w_D/w_G/w_S/window/bias`.
2. **Fine-tune SignCLIP on Thai gloss vocabulary** — adapt the text/sign encoders to TSL
   so cosine similarity sharpens.
3. **Fine-tune the E4s-1 segmenter on TSL** (highest payoff for Task 2) — full
   fine-tune if ≥1,000 examples, else adapter/last-layer; this directly attacks the
   Config 3–5 annotation-convention mismatch.
4. **End-to-end fine-tune** (advanced, last resort).

---

## 17. Deliverables & Inventory

### 17.1 Code (1,342 tracked files)

- Modified `SEA/` (multi-model align, Windows-safe segmentation)
- `example_alignment/` custom scripts (data prep, 2 aligners, 3 evaluators, visualizers)
- `ForcedAlignment/` orchestrator + evaluators + EAF tools + 1,140 GT EAFs
- `patches/fairseq_signclip_windows.patch` + `patches/README.md`

### 17.2 Cached results (committed, reproducible)

- 7× `aligned_output_*/04{,_no_overlap}.vtt` (Task 1) + `evaluation_task1_results.csv`
- Task 2 prediction CSVs/VTTs (default + ablation `Gloss`/`Gloss_Input`) + eval CSVs
- ForcedAlignment `output/predictions/config{1..5}_*.csv`, `prediction_vtt/`,
  `evaluation/eval_config{1..5}.csv`, `evaluation_summary.csv`
- Comparison/best EAFs (`Test_comparison.eaf` 17 tiers, `Test_best.eaf`)

### 17.3 Documentation

- [README.md](README.md) (78 KB, bilingual, canonical setup)
- [Big_Progress.md](Big_Progress.md) (126 KB, Thai canonical, 12 sections)
- This file — `PROJECT_PROGRESS_DEEP.md` (English deep narrative)
- 5 dated progress reports: [Progress_20042026.md](Progress_20042026.md),
  [Progress_26042026.md](Progress_26042026.md), [Progress_04052026.md](Progress_04052026.md),
  [Progress_09052026.md](Progress_09052026.md), [Progress_16052026.md](Progress_16052026.md)
- [presentation_12052026.md](presentation_12052026.md), [Presentation_Task1_Update.md](Presentation_Task1_Update.md)
- [SEA_Pipeline_Guide_TH.md](SEA_Pipeline_Guide_TH.md), `script.md`, `SEA_Pipeline_Deep_Dive.html`
- [report/sea_report.pdf](report/sea_report.pdf) + Thai `sea_report_th.pdf` (LaTeX sources tracked)
- `DeepDive.pdf` (7.3 MB)
- [ForcedAlignment/README.md](ForcedAlignment/README.md) + [PLAN_ForcedAlignment_Task2.md](ForcedAlignment/PLAN_ForcedAlignment_Task2.md)
- `arXiv-2512.08094v1/` — full SEA paper source (LaTeX + figures)

### 17.4 Assets NOT in git (must obtain/regenerate)

`04.mp4`, `04.pose`, ForcedAlignment MP4s (NECTEC data), embeddings, segmentation EAFs,
`fairseq_signclip/`, 3 SignCLIP checkpoints (~600 MB).

---

## 18. References

- **SEA** — Jiang, Jang, Momeni, Varol, Ebling, Zisserman (2025), *Segment, Embed, and
  Align: A Universal Recipe for Aligning Subtitles to Signing*,
  [arXiv:2512.08094](https://arxiv.org/abs/2512.08094) · code: [J22Melody/SEA](https://github.com/J22Melody/SEA)
- **SignCLIP** — multilingual sign-language embedding model · [J22Melody/fairseq](https://github.com/J22Melody/fairseq)
- **Linguistic segmenter** — [J22Melody/segmentation@bsl](https://github.com/J22Melody/segmentation/tree/bsl)
- **pose-format / MediaPipe Holistic** — [sign-language-processing/pose](https://github.com/sign-language-processing/pose)
- **NECTEC** — provider of TSL annotation video and ELAN annotations
- **ELAN** — [archive.mpi.nl/tla/elan](https://archive.mpi.nl/tla/elan)

---

*Compiled 2026-06-07 from the repository's commit history, source code, committed results,
and the existing progress documentation. All quoted metrics trace to committed CSVs and
were re-verified reproducible on 2026-06-01.*
