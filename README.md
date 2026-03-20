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
4. [Stage-by-Stage Technical Details](#stage-by-stage-technical-details)
5. [Input Format](#input-format)
6. [Output Format](#output-format)
7. [Repository Structure](#repository-structure)
8. [Environment Setup (uv)](#environment-setup-uv)
9. [Running the Example Notebook](#running-the-example-notebook)
10. [Results](#results)
11. [Evaluation Metrics Explained](#evaluation-metrics-explained)
12. [Running the Full SEA Pipeline](#running-the-full-sea-pipeline)
13. [Benchmark Performance](#benchmark-performance)
14. [Known Limitations and Common Issues](#known-limitations-and-common-issues)
15. [Citation](#citation)

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

```text
Cost(i, k→j) = |cue.start − sign[k].start|          (start-time mismatch)
             + |cue.end   − sign[j].end  |            (end-time mismatch)
             + α × |cue.duration − group.duration|    (duration penalty, α = duration_penalty_weight)
             + β × Σ gaps_between_signs(k..j)         (gap penalty, β = gap_penalty_weight)
             − γ × semantic_similarity(cue_i, signs)  (similarity reward, γ = similarity_weight)
```

The total cost is summed over all M cues, and DP finds the globally optimal solution in O(M × window_size²) time.

**Monotonicity constraint:** subtitle cue *i* must be assigned to sign segments that appear after those assigned to cue *i−1*. This preserves subtitle ordering.

**Grouping:** each subtitle cue can be matched to a *group* of consecutive sign segments [k, j). The group's effective start is `sign[k].start` and its effective end is `sign[j].end`. The cost penalises both timing mismatch and gaps between the signs within the group.

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

### What is a "Sign Segment"?

A **sign segment** is any time interval during which the signer is actively producing a recognisable sign gesture. Segments are represented as dicts:

```python
{
    "start": 33.56,    # segment start in seconds (float)
    "end":   34.61,    # segment end in seconds (float)
    "mid":   34.085,   # midpoint = (start + end) / 2 — used by DP window selection
    "text":  "สวัสดี" # gloss label (empty string for auto-segmented SIGN tier)
}
```

The `'mid'` key is **required** by `dp_align_subtitles_to_signs()` and must be present in all sign segment dicts.

---

## Pipeline Architecture

```text
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

## Stage-by-Stage Technical Details

### Stage 0 — Pose Estimation (`videos_to_poses`)

**Tool:** [`pose-format`](https://github.com/sign-language-processing/pose) package, which wraps **MediaPipe Holistic**.

**What MediaPipe Holistic extracts (per frame):**

| Component | Landmarks | Dimensions | Notes |
| --- | --- | --- | --- |
| Body (BlazePose) | 33 | x, y, z | World-space body skeleton |
| Face mesh | 468 | x, y, z | Full face with iris landmarks |
| Left hand | 21 | x, y, z | Finger joints + wrist |
| Right hand | 21 | x, y, z | Finger joints + wrist |
| **Total** | **543** | **3** | Stored as `(frames, 1, 543, 3)` array |

> **Note on shape:** In practice, the pose file shape may vary (e.g. `(frames, 1, 586, 3)`) depending on the MediaPipe version and which landmark groups are enabled. The first axis is the frame count, the second is the person count (always 1 for single-person videos), the third is the landmark count, and the fourth is (x, y, z) or (x, y, z, visibility).

**Video cropping (right half):**
The notebook crops each frame to the **right half** (`x = width/2 … width`) before pose estimation. This is because the example Thai Sign Language video uses a dual-panel format where the signer appears on the right half. Cropping eliminates the left panel (text/graphics), which would otherwise add noise to MediaPipe's person detector.

**Config flags used:**

```bash
--additional-config=model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true
```

- `model_complexity=2` — highest-accuracy BlazePose model (slower but more precise landmarks)
- `smooth_landmarks=false` — disables temporal smoothing across frames; this preserves frame-to-frame jitter which the segmentation model uses to detect sign boundaries
- `refine_face_landmarks=true` — enables the attention-mesh model for more precise iris and lip landmarks

**Output file format (`.pose`):**

Binary format from the `pose-format` library. Contains:

- A fixed-size header with FPS, shape, and component metadata
- A NumPy-compatible float16 array of shape `(frames, persons, landmarks, dims)`
- A confidence mask (same shape) for occluded/undetected landmarks

**Performance (11-minute video):**

| Platform | Approximate time |
| --- | --- |
| macOS CPU | 5–15 min |
| Linux/Windows GPU (CUDA) | 1–2 min |

The `.pose` file is cached — the cell skips re-running if the file already exists. Delete `poses/04.pose` to force recomputation.

---

### Stage 1 — Sign Segmentation (`pose_to_segments`)

**Tool:** [`sign-language-segmentation`](https://github.com/J22Melody/segmentation/tree/bsl) package (`bsl` branch), CLI command `pose_to_segments`.

**What the model does:**
A Transformer-based binary classifier runs over the pose feature sequence and assigns each frame one of three labels:

| Label | Meaning |
| --- | --- |
| `SIGN` | Frame is in the middle of an active sign |
| `SIGN-B` | Frame is at the beginning/onset of a new sign |
| `SIGN-O` | Frame is at rest / not signing (transition or pause) |

**Thresholds:**

- `--sign-b-threshold 30` — confidence threshold (0–100) to call a frame `SIGN-B`. Lower values detect more sign onsets (higher recall, more short segments)
- `--sign-o-threshold 70` — confidence threshold to call a frame `SIGN-O`. Higher values treat more frames as non-signing (fewer false sign detections)

The model produces the `SIGN` tier in an ELAN `.eaf` file, where each annotation is a contiguous run of `SIGN` / `SIGN-B` frames grouped into one sign-boundary interval.

**Why auto-segmentation over-generates:**
On this 11-minute video, auto-segmentation produces ~2,800 segments vs ~852 in the human Gloss Labeling. This is normal and expected: the model detects fine-grained movement transitions (including head movements, body shifts, and co-articulation) that a human annotator would label as part of a larger sign or as a non-signing gesture. The DP aligner handles this over-generation by grouping multiple consecutive segments per subtitle cue.

**Model file:**
`model_E4s-1.pth` — a pre-trained model checkpoint shipped with the `sign-language-segmentation` package. The `--no-pose-link` flag tells the tool not to embed the pose file path inside the output `.eaf`.

---

### Stage 2 — Embedding (`align_similarity.py`)

**Function:** `compute_similarity_matrix(cues, sign_segments, similarity_measure, ...)`

Returns an `(M, N)` NumPy float32 array where `M = len(cues)` and `N = len(sign_segments)`. Entry `[i, j]` represents how semantically similar subtitle cue *i* is to sign segment *j*.

**Embedding modes:**

#### `"text_embedding"` — SentenceTransformer

Model: `all-MiniLM-L6-v2` (384-dimensional embeddings, ~80 MB download on first use).

1. Each sign segment's `.text` gloss label is encoded to a 384-dim L2-normalised vector
2. Each subtitle cue's `.text` is encoded similarly
3. Similarity is computed as the **dot product** (equivalent to cosine similarity on L2-normalised vectors)

Because this mode encodes text-to-text, it requires sign segments to have non-empty text labels — i.e., it only works with the human `Gloss Labeling` tier, not the auto-segmented SIGN tier.

Normalization (applied row-wise by default): softmax with temperature τ=10. This turns each row of the (M, N) matrix into a probability distribution over sign segments, preventing a single dominant sign from capturing all cues.

#### `"sign_clip_embedding"` — SignCLIP

Model: [SignCLIP](https://aclanthology.org/2024.emnlp-main.518/) — a CLIP-style vision-language model jointly trained on sign video clips and subtitle text.

- Sign embeddings come from **video/pose features** (not text), so no gloss labels are needed
- Subtitle embeddings are text embeddings from SignCLIP's language encoder
- Both are 512-dimensional

This is the **recommended mode for production** use. It is fully automated (no human annotation needed) and achieves +6% F1@0.50 improvement over temporal-only alignment on BOBSL.

#### `"none"` — No embeddings

`sim_matrix = None`, `similarity_weight = 0.0`. Pure temporal alignment.

**Normalization methods available:**

- `softmax` (default): temperature-scaled softmax along rows. Keeps relative ordering, prevents saturation.
- `z-score + sigmoid`: z-score normalize each row, then apply sigmoid. Maps to (0, 1).
- `sinkhorn`: iterative row/column normalization producing a doubly stochastic matrix. Useful when each subtitle should match exactly one sign group.

---

### Stage 3 — DP Alignment (`align_dp.py`)

**Function:** `dp_align_subtitles_to_signs(cues, sign_segments, gt_cues, duration_penalty_weight, gap_penalty_weight, window_size, max_gap, similarity_weight, sim_matrix)`

This function modifies `cues` **in place** — each cue's `'start'` and `'end'` are replaced with the aligned values.

#### The DP formulation

Let:

- `M` = number of subtitle cues
- `N` = number of sign segments
- `dp[i][j]` = minimum total alignment cost when cue `i` is aligned to sign segment `j` as the last segment of its group

The recurrence is:

```text
dp[i][j] = min over k in [candidate_min, j-1]:
    dp[i-1][k] + cost(cue_i, signs[k+1 .. j])
```

where `cost(cue_i, signs[k+1..j])` is the full cost function:

```text
cost = |cue.start  − signs[k+1].start| +    ← start-time mismatch
       |cue.end    − signs[j].end     | +    ← end-time mismatch
       α × |cue.duration − group.dur | +     ← duration penalty
       β × gap_cost[k+1][j]          −       ← gap penalty (sum of gaps within group)
       γ × sim_cumsum[i][j] − sim_cumsum[i][k]  ← similarity reward (if enabled)
```

**`gap_cost` precomputation:** Rather than recomputing the sum of inter-sign gaps at every DP step, `compute_gap_cost()` builds an `(N+1, N+1)` cumulative sum matrix once. `gap_cost[i][j] = Σ max(0, sign[p].start − sign[p−1].end)` for `p = i+1..j`. This reduces the inner loop from O(N) gap summations to O(1) array lookups.

**`candidate_min` / `candidate_max` (sliding window):**
For each cue `i`, only sign segments in a window `[candidate_min[i], candidate_max[i]]` are considered. The window is centred on the sign segment whose `.mid` is closest to the cue's `.mid`, with radius `window_size`. This reduces the O(M × N²) DP to O(M × window_size²).

#### Numba JIT compilation

The inner DP loop (`dp_inner_loop`) is decorated with `@njit` (Numba ahead-of-time compilation). On the **first call**, Numba compiles the function to native machine code (10–45 seconds one-time cost). Subsequent calls are instant. The compiled function:

- Operates entirely on NumPy arrays (no Python objects inside the loop)
- Runs without the GIL, enabling future parallelisation
- Handles the `sim_matrix` slice, softmax normalisation, and cumulative sum all within native code

**Tip:** the JIT warmup cell (Section 6) calls `dp_align_subtitles_to_signs()` with tiny dummy data to trigger compilation before the real alignment run.

#### Parameter guide

| Parameter | Recommended | Effect |
| --- | --- | --- |
| `duration_penalty_weight` (α) | 1.0 | Higher → aligner more strongly prefers cue/group durations to match. Good when subtitle durations track sign durations. |
| `gap_penalty_weight` (β) | 5.0 | Higher → aligner avoids grouping signs with large gaps. Prevents one subtitle spanning a long pause. |
| `window_size` | 50 | Candidate window radius. Larger = broader search, slower. 50 is safe for videos with consistent signing pace. |
| `max_gap` | 10.0 s | Post-processing step: if two consecutive aligned cues have a gap > max_gap in sign coverage, the later cue's start is clipped. |
| `similarity_weight` (γ) | 10.0 (with embeddings), 0.0 (none) | Higher → semantic match dominates timing. Should only be > 0 when `sim_matrix` is not None. |

#### Output

After the DP solves the global optimum, each cue gets:

- `cue['start']` = `sign_group[0].start`
- `cue['end']` = `sign_group[-1].end`

These replace the original speech-timed timestamps. The text is unchanged.

---

## Input Format

### Video (`*.mp4`)

Standard MP4 video of a sign language recording. Any resolution or frame rate; the pipeline normalises internally. The example dataset uses `04.mp4` — an 11-minute Thai Sign Language educational video recorded at 60 fps, resolution 1920×1080 (dual-panel; right half contains the signer).

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

**Parsing in Python:**
`SEA/utils.py → get_sign_segments_from_eaf()` parses the `SIGN` tier. The notebook's `eaf_tier_to_cues()` is a more general version that can parse any named tier. Both convert `TIME_VALUE` (milliseconds integer) to seconds (float) by dividing by 1000.

**Tier roles in the SEA pipeline:**

| Tier | Content | Granularity | SEA role |
| --- | --- | --- | --- |
| `CC` | Full transcript (172 cues, 0.0 s – 663.6 s) — includes two bracketed music cues `[เสียงดนตรี]` | Sentence | **Input subtitle cues** for DP alignment |
| `CC_Aligned` | Manually refined signed-portion subset (119 cues, 33.6 s – 640.8 s) | Sentence | Not used — cue count (119) differs from GT (172) |
| `Gloss` | Full sign-language gloss sequence per sentence | Sentence | Reference/documentation |
| `Gloss Labeling` | One annotation per sign gesture, precisely timestamped | Sub-sentence | **Sign-segment anchors** (= `SIGN` tier equivalent) |
| `SIGN` | Auto-generated sign segments from `segmentation.py` | Sub-sentence | Standard SEA pipeline sign anchors |
| `SIGN_MERGED` | Sign segments merged from multiple sources | Sub-sentence | Output (Stage 3, if multiple sign sources) |
| `SUBTITLE_SHIFTED` | DP-aligned subtitle timestamps | Sentence | **Output** of Stage 3 |

### Subtitle File (`*.vtt` or `*.srt`)

WebVTT or SubRip format. Each cue: `start --> end` + text:

```text
WEBVTT

00:00:34.030 --> 00:00:36.210
(คุณครูจิรชพรรณ) สวัสดีค่ะนักเรียนทุกคน
```

**Internal representation:** the pipeline converts VTT timestamps to seconds using `HH×3600 + MM×60 + SS.mmm`. Each cue is a Python dict:

```python
{"start": 34.030, "end": 36.210, "mid": 35.12, "text": "สวัสดีค่ะนักเรียนทุกคน"}
```

The `'mid'` key (midpoint) is used by the DP aligner's sliding-window candidate selection.

### Embedding Files (`*.npy`) — Stage 2 only

NumPy arrays of shape `(N, D)` where `D = 512` (SignCLIP dimension). One file per video, per modality:

- `subtitle_embedding/04.npy` — shape `(M, 512)`, one row per subtitle cue
- `segmentation_embedding/04.npy` — shape `(N, 512)`, one row per sign segment

---

## Output Format

### Updated ELAN File (`*_updated.eaf`)

The original EAF with new tiers appended by `write_updated_eaf()` in `SEA/utils.py`:

- **`SUBTITLE_SHIFTED`** — DP-aligned subtitle cues with corrected `(start, end)` timestamps, same text as input
- **`SIGN_MERGED`** — merged sign segments (only if `signs` argument is passed to `write_updated_eaf`)

New `TIME_SLOT` entries are added to the `TIME_ORDER` block with IDs of the form:

```text
SUBTITLE_TS_{video_id}_{i}_1  ← start of cue i
SUBTITLE_TS_{video_id}_{i}_2  ← end of cue i
```

Open in [ELAN](https://archive.mpi.nl/tla/elan) to visually inspect alignment quality across all tiers simultaneously.

### Aligned Subtitle File (`*.vtt`)

Standard WebVTT file with corrected timestamps, generated by `reconstruct_vtt()` in `SEA/utils.py`. Drop-in replacement for the original `.vtt` — compatible with all video players (browser, VLC, FFmpeg, mpv, etc.).

Format guarantees:

- Header line: `WEBVTT`
- One blank line after header
- Each cue: timing line (`HH:MM:SS.mmm --> HH:MM:SS.mmm`) + text line + blank line
- Timestamps are formatted to millisecond precision

---

## Repository Structure

```text
Sign_to_sub/
├── pyproject.toml              ← uv project manifest (all dependencies declared here)
├── .python-version             ← Python 3.12 pin (read automatically by uv)
├── .gitignore
├── README.md                   ← This file
├── requirements.txt            ← Pinned snapshot from `uv pip freeze`
│
├── assets/                     ← Generated visualisation outputs (by running the notebook)
│   ├── alignment_visualization.png    ← 4-track timeline comparison chart
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
│       ├── 04.mp4              ← 11-minute TSL educational video (159 MB, 60 fps, 1920×1080)
│       ├── *.eaf               ← ELAN annotation: CC, CC_Aligned, Gloss, Gloss Labeling tiers
│       ├── *.pfsx              ← ELAN display preferences (colours, tier order, etc.)
│       └── aligned_output.vtt  ← External reference alignment (172 cues — different subtitle source)
│
└── notebooks/
    └── example_alignment_usage.ipynb  ← End-to-end alignment walkthrough
```

**Important path note:** `SEA/` stays at the repository root (not moved into `src/`) because `SEA/segmentation.py` and other scripts use `__file__`-relative paths internally. Moving `SEA/` would break those imports. The notebook adds both `SEA/` and `SEA/misc/` to `sys.path` explicitly.

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

The notebook `example_alignment_usage.ipynb` runs the **complete 4-stage SEA pipeline** on the Thai Sign Language example in `data/example_alignment/`, starting from the raw `.mp4` video and producing aligned `.vtt` subtitles. The `.vtt` input is extracted from the `.eaf` file's `CC` tier (172 cues, 0.0 s – 663.6 s; in production this would come from ASR). Every code cell is preceded by a detailed markdown explanation.

**Notebook sections:**

| Section | Content |
| --- | --- |
| 1 | Environment & path setup — venv instructions, `sys.path`, CUDA/GPU detection |
| 2 | Preprocessing — extract `.vtt` from EAF `CC` tier (172 cues, 0.0–663.6 s); load `aligned_output.vtt` as GT; verify cue-count match; extract `Gloss Labeling` as GT sign segments |
| 3 | Stage 0: Pose Estimation — right-half crop, `videos_to_poses` (MediaPipe Holistic), `.mp4` → `.pose` |
| 4 | Stage 1: Sign Segmentation — `pose_to_segments`, `.pose` → `.eaf` with SIGN tier |
| 5 | Stage 2: Embedding (optional) — `text_embedding` / `sign_clip_embedding` / `none` |
| 6 | Stage 3: DP Alignment — cost function, Numba JIT warmup, hyperparameters, output writing |
| 7 | Evaluation — frame-level accuracy, F1@IoU, three-way comparison, GT mismatch explanation |
| 8 | Timeline visualisation — 4-track comparison chart |
| 9 | Summary and next steps |

**Outputs generated by the notebook:**

```text
data/example_alignment/pipeline_output/
├── poses/04.pose                              ← Stage 0
├── subtitles/04.vtt                           ← Input VTT (from CC tier, 172 cues, 0–663 s)
├── ground_truth/04.vtt                        ← GT for evaluation (aligned_output.vtt, 172 cues)
├── segmentation/E4s-1_30_70/04.eaf            ← Stage 1 (SIGN tier, 2,803 segments)
├── segmentation/E4s-1_30_70/04_updated.eaf    ← Stage 3 (SUBTITLE_SHIFTED + SIGN_MERGED tiers added)
└── aligned/04.vtt                             ← Stage 3 (aligned subtitles, 172 cues)

assets/
├── alignment_visualization.png    ← 4-track timeline comparison chart
└── evaluation_metrics.png         ← Quantitative metrics chart
```

---

## Results

The figures below are generated automatically when the notebook is run. Input: `CC` tier (172 cues), GT: `aligned_output.vtt` (172 cues), mode: `text_embedding`.

### Demo Results (Thai Sign Language, `text_embedding` mode)

Three-way comparison from the notebook. GT = `aligned_output.vtt` (172 cues), input = CC tier (172 cues), 170 evaluation pairs after filtering bracketed music cues.

| Method | Frame Acc. | F1@0.10 | F1@0.25 | F1@0.50 | Mean \|Δstart\| | Median \|Δstart\| |
| --- | --- | --- | --- | --- | --- | --- |
| **Pipeline (text\_embedding, GT signs)** | 78.53% | **98.82%** | **97.65%** | **89.41%** | 0.45 s | 0.36 s |
| Temporal-only (GT signs) | 78.77% | 98.82% | 97.65% | 89.41% | 0.45 s | 0.37 s |
| Auto-sign temporal | 77.94% | 98.24% | 95.88% | 87.65% | 0.50 s | 0.34 s |

Key observations:

- **F1@0.10 = 98.82%** — nearly all 170 subtitle cues land within 10% temporal overlap of their GT boundary; the DP aligner places cues almost perfectly at the right sign onset
- **F1@0.50 = 89.41%** — 89% of cues have ≥ 50% overlap with GT, meaning tight half-second precision on more than 150 of 170 cues
- **Mean |Δstart| = 0.45 s** — on average, aligned start times are within half a second of the independently-produced GT
- **Frame-level accuracy = 78.53%** — slightly below the BOBSL benchmark (~80%) because the `model_E4s-1.pth` segmentation model was trained on British Sign Language and produces slightly coarser sign-boundary estimates on Thai SL
- **text\_embedding vs temporal-only**: essentially identical on this dataset — the semantic signal (Thai subtitle text ↔ Thai gloss labels) adds marginal benefit because the temporal priors are already very strong with 852 well-spaced GT sign segments
- **Auto-sign temporal** (fully automated, no human annotation) scores 2.3 pp lower on F1@0.50 (87.65% vs 89.41%) — the cost of using ~2,800 noisy auto-segments instead of 852 human-annotated signs. This gap would narrow with a Thai-specific segmentation model

### Alignment Timeline

The four-track chart shows a 60-second window of the video (seconds 33–93) with auto-segmented signs, human-annotated signs, original subtitles, and DP-aligned subtitles:

![Alignment Timeline](assets/alignment_visualization.png)

> **How to read:** Steel blue bars (Track 1, "Auto SIGN") are auto-detected sign segments from Stage 1 — note the high density (~2,803 over 11 min). Coral bars (Track 2, "Human Gloss Labeling") are the human-annotated reference at 852 segments. Tomato bars (Track 3, "Original CC") are the original CC tier subtitle timestamps — speech-timed, some appearing before the signing starts. Sea green bars (Track 4, "DP-Aligned Output") shift to snap to detected sign boundaries. The first cue `[เสียงดนตรี]` (music annotation, Δstart ≈ +33.5 s) is pushed to the first sign onset — correct behaviour. When Stage 3 aligns well, Track 4 bars overlap with Track 2 coral bars.

### Evaluation Metrics

**Panel A** shows F1 scores and frame-level accuracy for three variants: `text_embedding` pipeline, temporal-only (same GT signs), and auto-sign temporal.
**Panel B** shows mean and median absolute start/end offsets in seconds — all bars are under 0.5 s, confirming sub-second accuracy.

![Evaluation Metrics](assets/evaluation_metrics.png)

> **Note on the demo results:** Input is `CC` tier (172 cues) and GT is `aligned_output.vtt` (172 cues from the same source) — cue counts match, so all metrics are valid. The slight frame-accuracy gap vs BOBSL (~80%) reflects the BSL-trained segmentation model's coarser boundaries on Thai SL. F1@0.50 = 89.41% means 89% of subtitle cues land within ≥ 50% temporal overlap of the independently-produced GT.

---

## Evaluation Metrics Explained

### Frame-level accuracy

Both the prediction VTT and the GT VTT are converted to **per-frame label sequences** at the video's FPS. Each frame is labeled with:

- The **subtitle cue index** (0, 1, 2, …) if that frame falls within any cue's `[start, end)` window
- **−1** ("background") if no subtitle is active

`Frame-level accuracy = (frames where pred_label == gt_label) / total_frames × 100`

### F1 @ IoU threshold

For threshold τ ∈ {0.10, 0.25, 0.50}:

1. Scan the frame sequence for contiguous runs of the same non-background label → "segments"
2. For each predicted segment with label *i*: find any GT segment with label *i* where `IoU = overlap / union ≥ τ`
3. Count: TP (matched), FP (unmatched pred), FN (unmatched GT)
4. `F1 = 2·TP / (2·TP + FP + FN)`

F1@0.10 is lenient (10% overlap required); F1@0.50 is strict (50% overlap required). Because matching is by cue index, two segments with different indices cannot match even if they overlap temporally — this means **a valid GT must have the same cue count as the prediction**.

### Offset metrics (Δstart, Δend)

```text
Δstart_i = gt_cue[i].start − pred_cue[i].start   (positive = GT is later = pred is early)
```

These are computed by **index matching** — cue[i] in prediction is compared to cue[i] in GT. This is only meaningful when:

- GT and prediction have the **same number of cues**
- Cue[i] in prediction and GT[i] in GT describe the **same subtitle event**

When cue counts differ (as in the demo), cue[i] in pred and cue[i] in GT are different sentences → the offsets are random and meaningless.

### This dataset's evaluation setup

The included `aligned_output.vtt` (172 cues) was produced from the same CC tier used as pipeline input — cue counts match, so **all metrics are valid out of the box**. `STRICT_INDEPENDENT_GT = True` (the default) enforces this by refusing to run evaluation if the GT file is missing.

Offset metrics are index-matched: pred cue *i* is compared to gt cue *i*. This is valid when both sequences are derived from the same source transcript with identical ordering. The two bracketed music cues `[เสียงดนตรี]` (intro 0–32 s and outro 650–663 s) are automatically excluded by the evaluator, leaving **170 evaluation pairs**.

To adapt this pipeline to a new dataset: provide `aligned_output.vtt` with the same number of cues as your input subtitle tier, with timestamps representing the ground-truth signing-aligned boundaries.

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

### BOBSL validation set (SEA code repo, 32 videos, 1,973 subtitle cues)

Results using the BOBSL-finetuned SignCLIP (`sign_clip_embedding` mode), independent human-annotated GT:

| Method | Frame Acc. | F1@0.10 | F1@0.25 | F1@0.50 | Mean Start Δ | Mean End Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Segment + Align (temporal only) | 80.68% | 83.07% | 79.32% | 66.24% | −0.50s | −1.04s |
| **Segment + Embed + Align (SignCLIP)** | **82.52%** | **86.37%** | **82.92%** | **72.23%** | **−0.36s** | **−0.91s** |

Adding SignCLIP embeddings improves F1@0.50 by **+6 percentage points** and reduces timing errors by ~30%. The semantic similarity signal steers the DP aligner towards sign groups whose visual content matches the subtitle text, beyond what pure timing can achieve.

### Paper test-set results (Table 2, arXiv:2512.08094)

The paper reports F1@0.50 on held-out test sets using independent human-annotated GT:

| Dataset | Method | F1@0.50 |
| --- | --- | --- |
| BOBSL (British SL) | SEA multilingual | 50.68% |
| BOBSL (British SL) | SEA finetuned BSL | 54.50% |
| How2Sign (American SL) | SEA finetuned ASL | 39.57% |
| WMT-SLT (Swiss German SL) | SEA finetuned DSGS | 77.69% |
| SwissSLi (Swiss SL) | SEA multilingual | 85.57% |

> **Note on demo vs benchmark numbers:** The demo Thai SL results (F1@0.50 ≈ 87–89%) are **not comparable** to the BOBSL/How2Sign test-set numbers above. The demo GT (`aligned_output.vtt`) is a reference alignment — not independent human sign-annotation — so the task is substantially easier. To reproduce paper-level results, run with `sign_clip_embedding` on a dataset with truly independent human-annotated GT, and tune parameters on a training split.

### Paper hyperparameter defaults (Table 5, arXiv:2512.08094)

| Dataset | b-thresh | o-thresh | w\_dur | w\_gap | w\_sim | window | max\_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BOBSL | 30 | 50 | 1 | 5 | 10 | 50 | 10 s |
| How2Sign | 40 | 50 | 5 | 0.8 | 10 | 50 | 8 s |
| WMT-SLT | 20 | 30 | 0.5 | 5 | 5 | 50 | 6 s |
| SwissSLi | 20 | 30 | 0.5 | 5 | 1 | 50 | 6 s |

This demo uses **BOBSL DP defaults** (`w_dur=1, w_gap=5, w_sim=10, window=50, max_gap=10 s`) exactly. The segmentation uses `SIGN_O=70` (vs BOBSL default 50) because the BSL-trained model over-segments this Thai SL video at lower thresholds.

---

## Known Limitations and Common Issues

### Windows: symlink creation fails

On Windows, creating symbolic links requires either **Developer Mode** (Settings → For Developers → Developer Mode → ON) or elevated (Administrator) privilege. The notebook handles this gracefully: if `os.symlink()` raises `[WinError 1314]`, it falls back to `shutil.copy2()` which copies the cropped video file instead. This uses extra disk space but is functionally identical.

### CUDA: RTX 5000 series (Blackwell)

The RTX 5060 Ti and other sm_120 (Blackwell) GPUs require PyTorch ≥ 2.7.0 with the CUDA 12.8 wheel index. Older PyTorch versions will fall back to CPU silently. The `pyproject.toml` already points to the correct wheel index; simply run `uv sync` and you'll get the right version automatically.

### Numba JIT: first-run compilation delay

The first call to `dp_align_subtitles_to_signs()` triggers Numba's AOT compilation of the `dp_inner_loop` function. This takes 10–45 seconds depending on CPU speed. The compiled bytecode is cached in `__pycache__/` and reused on subsequent runs. The warmup cell in Section 6 of the notebook is designed to front-load this cost before the real alignment run.

### Evaluation: choosing the right input subtitle tier

The `.eaf` file for the demo dataset has two subtitle tiers: `CC` (172 cues, full transcript) and `CC_Aligned` (119 cues, signed portion only). The GT file `aligned_output.vtt` was produced from the `CC` tier — so the pipeline must also use `CC` as input. Using `CC_Aligned` (119 cues) against a 172-cue GT causes a cue-count mismatch where offset metrics become meaningless (~93 s random differences) and F1 scores collapse to near zero.

The notebook's cell 4 uses `eaf_tier_to_cues(EAF_PATH, "CC")` and includes a runtime cue-count check with a clear `UserWarning` if counts diverge. When adapting to a new dataset, always ensure your input subtitle tier and `aligned_output.vtt` have the same number of cues.

### EAF files with Thai (non-ASCII) filenames

Python's `xml.etree.ElementTree` and `glob.glob()` handle Unicode filenames correctly on all platforms as long as the locale is UTF-8. On Windows with non-UTF-8 system locale, ensure your terminal is set to UTF-8: `chcp 65001` in Command Prompt, or use Windows Terminal.

### MediaPipe version compatibility

MediaPipe 0.10.x changed the internal landmark format. `pose-format` 0.x wraps MediaPipe's `Holistic` solution API which was deprecated in MediaPipe ≥ 0.10.15. If you encounter `AttributeError: module 'mediapipe.solutions' has no attribute 'holistic'`, pin `mediapipe==0.10.14` in `pyproject.toml`.

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
