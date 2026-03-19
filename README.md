# SEA: Segment, Embed, and Align

**A universal recipe for aligning subtitles to sign language video content.**

> **Paper:** *Segment, Embed, and Align: A Universal Recipe for Aligning Subtitles to Signing*
> Zifan Jiang, Youngjoon Jang, Liliane Momeni, Gül Varol, Sarah Ebling, Andrew Zisserman (2025)
> [arXiv:2512.08094](https://arxiv.org/abs/2512.08094)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Concepts Explained](#key-concepts-explained)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Input Format](#input-format)
5. [Output Format](#output-format)
6. [Repository Structure](#repository-structure)
7. [Environment Setup (uv)](#environment-setup-uv)
8. [Running the Example Notebook](#running-the-example-notebook)
9. [Results](#results)
10. [Running the Full SEA Pipeline](#running-the-full-sea-pipeline)
11. [Benchmark Performance](#benchmark-performance)
12. [Citation](#citation)

---

## Project Overview

When a sign language video is captioned, subtitles are typically derived from the **spoken audio track** — either via a human interpreter or automatic speech recognition. This means every subtitle timestamp reflects when words were *spoken*, not when they were *signed*.

Signing lags behind speech by a variable delay (commonly 2–5 seconds) due to interpretation lag and the slower articulation rate of sign language. The result is subtitles that appear **seconds before or after** the signing they describe.

**SEA corrects this** by treating alignment as a global optimisation problem. Given a sequence of subtitle cues and a sequence of sign-language segments, it finds the monotone mapping that minimises total temporal mismatch — using Dynamic Programming, optionally guided by semantic similarity from vision-language embeddings (SignCLIP).

---

## Key Concepts Explained

### What is "Alignment"?

**Alignment** is the operation of replacing a subtitle cue's `(start, end)` timestamps with corrected values that match when the corresponding content is actually being signed on screen.

- The **text content** of each subtitle is never modified.
- Only the **start** and **end** times are corrected.
- Alignment is solved **globally** via Dynamic Programming across all cues at once — the algorithm considers the full sequence of sign segments and finds the monotone assignment of cues to sign groups that minimises the total cost function shown below.

**Cost function for aligning subtitle cue *i* to sign group [k, j):**

```
Cost(i, k→j) = |cue.start − sign[k].start|          (start-time mismatch)
             + |cue.end   − sign[j].end  |            (end-time mismatch)
             + α × |cue.duration − group.duration|    (duration penalty, α = duration_penalty_weight)
             + β × Σ gaps_between_signs(k..j)         (gap penalty, β = gap_penalty_weight)
             − γ × semantic_similarity(cue_i, signs)  (similarity reward, γ = similarity_weight)
```

The total cost is summed over all M cues, and DP finds the globally optimal solution in O(M × window_size²) time.

---

### What is a "Gloss Label"?

A **gloss** is a written token representing a single sign gesture, transcribed using a spoken-language word (or short phrase) that approximates the sign's meaning.

| Gloss label | Meaning |
| --- | --- |
| `สวัสดี` | "Hello" (signing the greeting) |
| `(ผายมือ)` | Open-hand wave gesture |
| `เด็ก` | "Child" (sign for the concept of child) |
| `(ลักษณนามรถยนต์)` | Classifier for vehicles |
| `(ชี้นิ้วชี้)` | Index-finger pointing gesture |

**The Gloss Labeling tier** provides **sub-sentence granularity** — each annotation corresponds to exactly one sign gesture with a precise start/end timestamp. This gives the DP aligner fine-grained temporal anchors, analogous to what `segmentation.py` produces automatically (the `SIGN` tier) for BOBSL and How2Sign.

In the full SEA pipeline, gloss-level segmentation is produced automatically by MediaPipe Holistic pose estimation + a sign-language segmentation model. In manually-annotated corpora like our example dataset, the same information is stored as a human-annotated `Gloss Labeling` tier.

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          SEA 4-STAGE PIPELINE                                │
│                                                                              │
│  ┌──────────┐     ┌──────────────────┐     ┌──────────────────┐             │
│  │  Input   │     │   STAGE 0        │     │   STAGE 1        │             │
│  │  Video   │────▶│  Pose Estimation │────▶│  Sign            │             │
│  │  .mp4    │     │  (MediaPipe)     │     │  Segmentation    │             │
│  └──────────┘     │                  │     │  (pose→.eaf)     │             │
│                   │  .mp4 → .pose    │     │  SIGN tier       │             │
│                   └──────────────────┘     └────────┬─────────┘             │
│                                                     │                        │
│  ┌──────────┐     ┌──────────────────┐              │                        │
│  │Subtitles │     │   STAGE 2        │              │                        │
│  │ .vtt/srt │────▶│  Embedding       │──────────────┤                        │
│  └──────────┘     │  (SignCLIP)      │  .npy        │                        │
│                   │  OPTIONAL        │              │                        │
│                   └──────────────────┘              ▼                        │
│                                            ┌──────────────────┐             │
│  ┌──────────┐                              │   STAGE 3        │             │
│  │Subtitles │─────────────────────────────▶│  DP Alignment    │             │
│  │ .vtt/srt │                              │  (align_dp.py)   │             │
│  └──────────┘                              └────────┬─────────┘             │
│                                                     │                        │
│                                           ┌─────────┴────────┐              │
│                                           │  OUTPUT           │              │
│                                           │  *_updated.eaf    │              │
│                                           │  aligned .vtt     │              │
│                                           └──────────────────┘              │
└──────────────────────────────────────────────────────────────────────────────┘
```

| Stage | Script | Input | Output | Notes |
| --- | --- | --- | --- | --- |
| **0 — Pose** | `videos_to_poses` (external) | `*.mp4` | `*.pose` | MediaPipe Holistic keypoints |
| **1 — Segment** | `SEA/segmentation.py` | `*.pose` | `*.eaf` (SIGN tier) | Sign boundary detection |
| **2 — Embed** | SignCLIP (external) | `*.pose`, `*.vtt` | `*.npy` | *Optional*; improves F1@0.50 by +6% |
| **3 — Align** | `SEA/align.py` | `*.eaf`, `*.vtt`, `*.npy` | `*_updated.eaf`, aligned `*.vtt` | Core DP alignment |

---

## Input Format

### Video (`*.mp4`)
Standard MP4 video of a sign language recording. Any resolution or frame rate; the pipeline normalises internally. The example dataset uses `04.mp4` — an 11-minute Thai Sign Language educational video.

### ELAN Annotation File (`*.eaf`)
ELAN files are XML documents storing multiple time-aligned annotation layers over the same video. All timestamps are in **milliseconds**. The two-level structure uses `TIME_SLOT` references:

```xml
<TIME_ORDER>
  <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="34030"/>  <!-- 34.030 seconds -->
  <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="36210"/>  <!-- 36.210 seconds -->
</TIME_ORDER>

<TIER TIER_ID="CC_Aligned">
  <ANNOTATION>
    <ALIGNABLE_ANNOTATION ANNOTATION_ID="a1"
                          TIME_SLOT_REF1="ts1"
                          TIME_SLOT_REF2="ts2">
      <ANNOTATION_VALUE>สวัสดีค่ะนักเรียนทุกคน</ANNOTATION_VALUE>
    </ALIGNABLE_ANNOTATION>
  </ANNOTATION>
</TIER>
```

**Tier roles in the SEA pipeline:**

| Tier | Content | Granularity | SEA role |
| --- | --- | --- | --- |
| `CC` | Raw Thai text transcription from audio | Sentence | Not used directly |
| `CC_Aligned` | Refined sentence-level captions with corrected timestamps | Sentence | **Input subtitle cues** for DP alignment |
| `Gloss` | Full sign-language gloss sequence per sentence | Sentence | Reference/documentation |
| `Gloss Labeling` | One annotation per sign gesture, precisely timestamped | Sub-sentence | **Sign-segment anchors** (= `SIGN` tier equivalent) |
| `SIGN` | Auto-generated sign segments from `segmentation.py` | Sub-sentence | Standard SEA pipeline sign anchors |
| `SIGN_MERGED` | Sign segments merged from multiple sources | Sub-sentence | Output (Stage 3, if multiple sign sources) |
| `SUBTITLE_SHIFTED` | DP-aligned subtitle timestamps | Sentence | **Output** of Stage 3 |

### Subtitle File (`*.vtt` or `*.srt`)
WebVTT or SubRip format. Each cue: `start --> end` + text:
```
WEBVTT

00:00:34.030 --> 00:00:36.210
(คุณครูจิรชพรรณ) สวัสดีค่ะนักเรียนทุกคน
```

### Embedding Files (`*.npy`) — Stage 2 only

NumPy arrays of shape `(N, D)` where `D = 512` (SignCLIP dimension). One file per video, per modality (signs / subtitles).

---

## Output Format

### Updated ELAN File (`*_updated.eaf`)

The original EAF with new tiers appended:

- **`SUBTITLE_SHIFTED`** — DP-aligned subtitle cues with corrected `(start, end)` timestamps, same text
- **`SIGN_MERGED`** — merged sign segments (only if multiple sign sources were used)

Open in [ELAN](https://archive.mpi.nl/tla/elan) to visually inspect alignment quality across all tiers simultaneously.

### Aligned Subtitle File (`*.vtt`)
Standard WebVTT file with corrected timestamps. Drop-in replacement for the original `.vtt` — compatible with all video players (browser, VLC, FFmpeg, mpv, etc.).

---

## Repository Structure

```
Sign_to_sub/
├── pyproject.toml              ← uv project manifest (all dependencies declared here)
├── .python-version             ← Python 3.12 pin (read automatically by uv)
├── .gitignore
├── README.md                   ← This file
├── requirements.txt            ← Pinned snapshot from `uv pip freeze`
│
├── assets/                     ← Generated visualisation outputs (by running the notebook)
│   ├── alignment_visualization.png    ← 3-track timeline comparison chart
│   └── evaluation_metrics.png         ← F1 scores + timing offset bar charts
│
├── SEA/                        ← Core SEA source code (unchanged from original repo)
│   ├── align.py                ← Main orchestrator: inference / dev / training modes
│   ├── align_dp.py             ← DP alignment algorithm (numba @njit inner loop)
│   ├── align_dp_dtw.py         ← Alternative DTW-based alignment
│   ├── align_similarity.py     ← Similarity matrix: SentenceTransformer / SignCLIP
│   ├── align_dp_visualization.py
│   ├── config.py               ← CLI argument parser (all hyperparameters)
│   ├── segmentation.py         ← Pose → ELAN sign segmentation wrapper
│   ├── utils.py                ← I/O utilities: EAF, VTT, SRT parsing + writing
│   ├── environment.yml         ← Conda env spec (kept as reference)
│   ├── README.md               ← SEA-specific CLI documentation
│   ├── data/                   ← Video ID lists: BOBSL, How2Sign, MITENAND, YouTube
│   ├── misc/
│   │   ├── evaluate_sub_alignment.py  ← Frame acc., F1@IoU, timing offset metrics
│   │   └── postprocessing_remove_intersections.py
│   ├── figures/                ← Paper figure generation scripts
│   ├── How2Sign/               ← How2Sign-specific utilities
│   └── mitenand/               ← MITENAND-specific utilities
│
├── data/
│   └── example_alignment/      ← Thai Sign Language demo dataset
│       ├── 04.mp4              ← 11-minute TSL educational video (159 MB)
│       ├── *.eaf               ← ELAN annotation: CC, CC_Aligned, Gloss, Gloss Labeling tiers
│       └── *.pfsx              ← ELAN display preferences
│
└── notebooks/
    └── example_alignment_usage.ipynb  ← End-to-end alignment walkthrough
```

---

## Environment Setup (uv)

This project uses [`uv`](https://docs.astral.sh/uv/) — a fast, modern Python package manager.

### 1. Install uv

```bash
# macOS / Linux (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip (any platform)
pip install uv
```

### 2. Create the virtual environment and install all dependencies

```bash
# From the repository root:
uv sync
```

`uv sync` reads `pyproject.toml`, creates `.venv/` at the repository root, and installs all packages. It also fetches `sign-language-segmentation` directly from its Git repository (`github.com/J22Melody/segmentation@bsl`) — a network connection is required on first install.

**CUDA note:** `pyproject.toml` directs `uv` to the [PyTorch CUDA 12.8 wheel index](https://download.pytorch.org/whl/cu128), so `uv sync` automatically installs `torch==2.7.0+cu128`. No separate CUDA install step is needed. This wheel supports NVIDIA GPUs from Maxwell (GTX 900) through Blackwell (RTX 5000 series, sm_120), including the RTX 5060 Ti used during development. Ensure your NVIDIA driver is ≥ 527.41 (Windows) or ≥ 525.60.13 (Linux).

### 3. Activate the environment

```bash
# Windows (Command Prompt / PowerShell)
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Verify the installation

```bash
python -c "import torch, numba, pympi, align_dp; print('All imports OK')"
```

### 5. (Optional) Generate a pinned requirements file

```bash
uv pip freeze > requirements.txt
```

---

## Running the Example Notebook

```bash
# Activate the environment first (see above), then:
cd notebooks
jupyter notebook example_alignment_usage.ipynb
```

The notebook `example_alignment_usage.ipynb` runs the **complete 4-stage SEA pipeline** on the Thai Sign Language example in `data/example_alignment/`, starting from the raw `.mp4` video and producing aligned `.vtt` subtitles. The `.vtt` input is extracted from the `.eaf` file's `CC_Aligned` tier (in production, this would come from ASR). Every code cell is preceded by a detailed markdown explanation.

**Notebook sections:**

| Section | Content |
| --- | --- |
| 1 | Environment & path setup — venv instructions, `sys.path`, CUDA/GPU detection |
| 2 | Preprocessing — extract `.vtt` from EAF `CC_Aligned` tier; extract `Gloss Labeling` as GT |
| 3 | Stage 0: Pose Estimation — `videos_to_poses` (MediaPipe Holistic), `.mp4` → `.pose` |
| 4 | Stage 1: Sign Segmentation — `pose_to_segments`, `.pose` → `.eaf` with SIGN tier |
| 5 | Stage 2: Embedding (optional) — SignCLIP explanation, skipped in demo |
| 6 | Stage 3: DP Alignment — cost function, Numba JIT, hyperparameters, output writing |
| 7 | Evaluation — auto-segmented vs human Gloss Labeling comparison, F1@IoU metrics chart |
| 8 | Timeline visualisation — 4-track comparison chart (auto SIGN / human Gloss / original / aligned) |
| 9 | Summary and next steps |

**Outputs generated by the notebook:**

```text
data/example_alignment/pipeline_output/
├── poses/04.pose                              ← Stage 0
├── subtitles/04.vtt                           ← Input VTT (from CC_Aligned)
├── ground_truth/04.vtt                        ← Ground truth for evaluation
├── segmentation/E4s-1_30_70/04.eaf            ← Stage 1
├── segmentation/E4s-1_30_70/04_updated.eaf    ← Stage 3 (with SUBTITLE_SHIFTED tier)
└── aligned/04.vtt                             ← Stage 3 (aligned subtitles)

assets/
├── alignment_visualization.png    ← 4-track timeline comparison chart
└── evaluation_metrics.png         ← Quantitative metrics chart
```

---

## Results

The figures below are generated automatically when the notebook is run (`uv run jupyter nbconvert --to notebook --execute notebooks/example_alignment_usage.ipynb`).

### Alignment Timeline

The four-track chart shows a 60-second window of the video with auto-segmented signs, human-annotated signs, original subtitles, and DP-aligned subtitles:

![Alignment Timeline](assets/alignment_visualization.png)

> **How to read:** Steel blue bars (Track 1) are auto-detected sign segments from Stage 1. Coral bars (Track 2) are human-annotated Gloss Labeling (ground truth). Tomato bars (Track 3) are the original CC\_Aligned subtitle timestamps (speech-timed input). Sea green bars (Track 4) are the DP-aligned output — note how they shift to align with sign boundaries.

### Evaluation Metrics

**Panel A** shows F1 scores and frame-level accuracy comparing auto-segmented pipeline output vs alignment with human Gloss Labeling (upper bound).
**Panel B** shows mean and median absolute start/end offsets in seconds — lower bars mean timestamps closer to ground truth.

![Evaluation Metrics](assets/evaluation_metrics.png)

> **Results:** The notebook compares the full auto pipeline (Stage 0 → 1 → 3) against using human-annotated sign segments. The gap between auto-segmented and human Gloss Labeling shows the impact of segmentation quality on alignment accuracy.

---

## Running the Full SEA Pipeline

See [`SEA/README.md`](SEA/README.md) for complete CLI documentation. Minimal example on the BOBSL validation set:

```bash
# Stage 0: extract poses from video (requires pose-format package)
videos_to_poses \
  --num-workers 4 --format mediapipe \
  --additional-config="model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true" \
  --directory ~/BOBSL/videos/

# Stage 1: segment signs from poses
python SEA/segmentation.py \
  --sign-b-threshold 30 --sign-o-threshold 50 \
  --num_workers 4 \
  --video_ids SEA/data/bobsl_align_val.txt \
  --pose_dir ~/BOBSL/poses/ \
  --save_dir ~/BOBSL/segmentation/

# Stage 3a: align without embeddings (Segment + Align)
python SEA/align.py \
  --overwrite --mode=inference \
  --video_ids SEA/data/bobsl_align_val.txt \
  --num_workers 4 \
  --dp_duration_penalty_weight 1 \
  --dp_gap_penalty_weight 5 \
  --dp_max_gap 10 \
  --dp_window_size 50 \
  --pr_subs_delta_bias_start 2.6 \
  --pr_subs_delta_bias_end 2.1 \
  --similarity_measure none \
  --segmentation_dir ~/BOBSL/segmentation/ \
  --save_dir ~/BOBSL/aligned_subtitles/

# Stage 3b: align with SignCLIP embeddings (Segment + Embed + Align)
python SEA/align.py \
  --overwrite --mode=inference \
  --similarity_measure sign_clip_embedding \
  --similarity_weight 10 \
  --segmentation_embedding_dir ~/BOBSL/segmentation_embedding/ \
  --subtitle_embedding_dir ~/BOBSL/subtitle_embedding/ \
  [... same flags as above ...]
```

### Operating Modes

| Mode | Purpose | When to use |
| --- | --- | --- |
| `inference` | Align subtitles on new / test data | Production use, new datasets |
| `dev` | Evaluate on train/val/test splits | Measuring alignment quality |
| `training` | Random hyperparameter search | Tuning for a new language/corpus |

---

## Benchmark Performance

Results on **BOBSL validation set** (British Sign Language, 32 videos, 1,973 subtitle cues):

| Method | Frame Acc. | F1@0.10 | F1@0.25 | F1@0.50 | Mean Start Δ | Mean End Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Segment + Align | 80.68% | 83.07% | 79.32% | 66.24% | −0.50s | −1.04s |
| **Segment + Embed + Align** | **82.52%** | **86.37%** | **82.92%** | **72.23%** | **−0.36s** | **−0.91s** |

Adding SignCLIP embeddings improves F1@0.50 by **+6 percentage points** and reduces timing errors by ~30%. The semantic similarity signal steers the DP aligner towards sign groups whose visual content matches the subtitle text, beyond what pure timing can achieve.

---

## Citation

```bibtex
@article{jiang2025segment,
  title   = {Segment, Embed, and Align: A Universal Recipe for Aligning Subtitles to Signing},
  author  = {Jiang, Zifan and Jang, Youngjoon and Momeni, Liliane and Varol, G{\"u}l
             and Ebling, Sarah and Zisserman, Andrew},
  journal = {arXiv preprint arXiv:2512.08094},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.08094}
}
```
