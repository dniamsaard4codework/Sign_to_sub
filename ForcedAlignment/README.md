# ForcedAlignment — Task 2 Scale-up to 1,132 TSL Clips

Scale Task 2 (Gloss-token-to-SIGN-segment alignment) from the single demo video
(`example_alignment/04.mp4`, 119 cues) to a TSL corpus of **1,132 short clips**
(avg 5.8 s, ~110 minutes total) with 5 input/GT-tier configurations.

> **Full spec, evaluation criteria, and error analysis:**
> [PLAN_ForcedAlignment_Task2.md](PLAN_ForcedAlignment_Task2.md) (1,100-line Thai reference).
> **Top-level project overview:** [../README.md](../README.md).

---

## Prerequisites

Same as the main pipeline — finish the [main README setup](../README.md#full-setup-from-clone)
through **Step 7 (Verify install)** first. This sub-project reuses:

- The activated `venv/` with PyTorch + `mediapipe==0.10.21` + `pose-format`
- `fairseq_signclip/` cloned with SignCLIP checkpoints in place
- `videos_to_poses` and `pose_to_segments` on PATH (installed via `requirements.txt`)

GPU is auto-detected — see main README §"GPU usage note".

---

## Source data layout

The 1,140 EAF ground-truth files are in the repo (`elan_forced_alignment/`, ~9 MB total). The matching 1,132 source MP4s are **NECTEC-provided research data and are not publicly distributed** — contact the project maintainer for access. Without the MP4s you cannot reproduce the present-dataset run; if you have your own TSL recordings, see [Bringing your own dataset](#bringing-your-own-dataset) below.

Once you have the MP4s, arrange them like this:

```text
ForcedAlignment/
├── elan_forced_alignment/        ← 1,140 EAF ground-truth files (in repo, 9 MB)
│   ├── 1.eaf
│   ├── 2.eaf
│   └── ...
├── <any-folder-name>/            ← Source MP4s (NOT in repo, ~8.5 GB)
│   ├── 1 MP/  →  1.mp4 ... 185.mp4         (185 clips)
│   ├── 2 MP/  →  186.mp4 ... 371.mp4       (186 clips)
│   ├── 3 MP/  →  372.mp4 ... 560.mp4       (189 clips)
│   ├── 4 MP/  →  561.mp4 ... 754.mp4       (194 clips)
│   ├── 5 MP/  →  755.mp4 ... 925.mp4       (171 clips)
│   └── 6 MP/  →  926.mp4 ... 1132.mp4      (207 clips)
└── output/                        ← Generated artifacts (gitignored, ~10 GB)
```

`run_forced_alignment.py` auto-discovers the video root by scanning
`ForcedAlignment/<subdir>/**/*.mp4` for any subdir that contains MP4s (skipping
`elan_forced_alignment/` and `output/`). The convention used in development is a
folder named `หนังสือภาษามือไทย/` ("Thai Sign Language Book") containing the six
`<N> MP/` subfolders. You can name the top folder anything — only **MP4 stems
must match EAF stems** (e.g. `42.mp4` ↔ `42.eaf`).

Pass `--video-root <path>` to override auto-discovery.

---

## Quickstart

### 1. Smoke test (3 clips, ~2–3 minutes)

Confirms every phase works end-to-end before committing to an overnight run:

```powershell
venv\Scripts\activate
python ForcedAlignment\run_forced_alignment.py --only-ids 1,500,1132 --configs all
```

Expected: 3 poses, 3 segmentation EAFs, 2–3 embeddings (1 may have no SIGN
segments — handled by uniform fallback), 5 prediction CSVs, 1 evaluation
summary. See [PLAN §14.2](PLAN_ForcedAlignment_Task2.md#142-smoke-test-results)
for expected metric ranges.

### 2. Preflight check (no compute)

Validates manifest + required EAF tiers without running any heavy work:

```powershell
python ForcedAlignment\run_forced_alignment.py --preflight-only
```

### 3. Full run (1,132 clips, ~12 hours)

```powershell
python ForcedAlignment\run_forced_alignment.py --configs all
```

GPU strongly recommended. Pose extraction (Phase 2) is the bottleneck.

### 4. Re-evaluate without re-running compute

If predictions already exist in `output/predictions/`, just rerun the evaluator:

```powershell
python ForcedAlignment\evaluate_fa_dataset.py --configs all
```

---

## Pipeline phases (`run_forced_alignment.py`)

| Phase | Step | Output |
| --- | --- | --- |
| 1 | Build manifest + `video_ids.txt` | `output/manifest.csv` |
| 2 | `videos_to_poses` (MediaPipe) | `output/poses/<id>.pose` |
| 3 | SEA segmentation | `output/seg/E4s-1_30_50/<id>.eaf` |
| 4 | SignCLIP segment embeddings | `output/emb/<id>.npy` |
| 5 | In-process DP align (configs 1–5) | `output/predictions/config<N>_*.csv` |
| 6 | Inject prediction tiers into EAFs | `output/predicted_eafs/<id>.eaf` |
| 7 | Evaluate (Precision/Recall/F1/IoU) | `output/evaluation/*.csv` |

Skip flags: `--skip-pose`, `--skip-seg`, `--skip-emb`, `--skip-align`, `--skip-eval`
(useful for iterating on later phases without re-running pose extraction).

---

## Configurations

| # | Input tier | GT tier | Tokenization | Notes |
| ---: | --- | --- | --- | --- |
| 1 | `CC` | `CC_Aligned` | whitespace, drop `sil` | Best baseline (F1 68.6 %, mIoU 0.5928) |
| 2 | `CC` | `CC_Aligned` | whitespace, keep `sil` | Sil-handling experiment |
| 3 | `Gloss` | `Gloss_Labeling` | pipe-separated, drop empties | Direct analogue of `04.mp4` baseline |
| 4 | `Gloss1` | `Gloss_Labeling1` | pipe-separated | Alternate gloss annotation |
| 5 | `Gloss2` | `Gloss_Labeling2` | pipe-separated | Alternate gloss annotation |

Run a subset with `--configs 1,3` (comma-separated keys).

---

## Outputs

All under `output/` (gitignored, ~10 GB after full run):

```text
output/
├── manifest.csv                          ← clip_id ↔ eaf_path ↔ video_path
├── video_ids.txt
├── poses/<id>.pose                       ← 1,132 files, ~3.8 GB
├── video_work/<id>.mp4                   ← staged hardlinks, ~5.1 GB
├── seg/E4s-1_30_50/<id>.eaf              ← 1,132 SIGN-tier EAFs, ~7 MB
├── emb/<id>.npy                          ← (K, 768) per clip, ~29 MB total
├── predictions/config<N>_*.csv           ← 5 CSVs, per-token predictions
├── prediction_vtt/config<N>_*.vtt        ← Same as CSV in WebVTT for ELAN preview
├── predicted_eafs/<id>.eaf               ← GT EAF + 5 prediction tiers per clip
├── comparison_eafs/<id>.eaf              ← (optional) created by create_comparison_eafs.py
└── evaluation/
    ├── evaluation_summary.csv            ← 5 rows: config × {P, R, F1, mIoU, FrameAcc}
    └── per_clip_<config>.csv             ← per-clip breakdown
```

---

## Other scripts in this folder

| Script | Purpose |
| --- | --- |
| [check_eaf_video_match.py](check_eaf_video_match.py) | Validate EAF ↔ video stem match and `MEDIA_DESCRIPTOR` URLs |
| [fix_eaf_media_paths.py](fix_eaf_media_paths.py) | Repair broken `MEDIA_DESCRIPTOR` paths so EAFs open cleanly in ELAN GUI |
| [create_comparison_eafs.py](create_comparison_eafs.py) | Build per-clip GT + prediction comparison EAFs |
| [fill_gloss_labeling_template.py](fill_gloss_labeling_template.py) | Auto-fill `Gloss_Labeling_Template.docx` with final metrics |

---

## Results (full 1,132-clip run)

Source: [output/evaluation/evaluation_summary.csv](output/evaluation/evaluation_summary.csv).

| Config | Pred | GT | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 | Mean IoU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** ⭐ CC → CC_Aligned | 1,171 | 1,132 | 67.5 % | 69.8 % | **68.6 %** | **0.5928** |
| 2 CC (with sil) → CC_Aligned | 3,435 | 3,396 | 17.5 % | 17.7 % | 17.6 % | 0.2039 |
| 3 Gloss → Gloss_Labeling | 1,713 | 1,713 | 7.8 % | 7.8 % | 7.8 % | 0.2484 |
| 4 Gloss1 → Gloss_Labeling1 | 3,977 | 3,977 | 20.7 % | 20.7 % | 20.7 % | 0.2229 |
| 5 Gloss2 → Gloss_Labeling2 | 3,977 | 3,977 | 20.9 % | 20.9 % | 20.9 % | 0.2238 |

Config #1 beats the single-video `04.mp4` baseline (mIoU 0.4901). Configs 2–5
underperform — see the Config #3 regression error analysis (sil token handling)
and the architectural-mismatch analysis (GT annotation convention) at the
bottom of [PLAN_ForcedAlignment_Task2.md](PLAN_ForcedAlignment_Task2.md).

---

## Bringing Your Own Dataset

If you have your own TSL (or any sign language) corpus with ELAN annotations, you can run this pipeline on it. You need three things: **MP4 files + matching EAF files + the right tiers inside each EAF**.

### Required EAF tiers per config

Each config reads a specific input tier and evaluates against a specific GT tier. Your EAFs need only the tiers for the configs you want to run.

| Config | Input tier | GT tier | Notes on tokenization |
| ---: | --- | --- | --- |
| 1 | `CC` | `CC_Aligned` | whitespace split, drop `sil` tokens |
| 2 | `CC` | `CC_Aligned` | whitespace split, **keep** `sil` tokens |
| 3 | `Gloss` | `Gloss_Labeling` | pipe-separated tokens, drop empty |
| 4 | `Gloss1` | `Gloss_Labeling1` | pipe-separated tokens, drop empty |
| 5 | `Gloss2` | `Gloss_Labeling2` | pipe-separated tokens, drop empty |

Total: up to 8 distinct tier names across all 5 configs. The preflight check (`--preflight-only`) reports which tiers are present in your EAFs and refuses to run if a required tier is missing.

### Running a subset of configs

If your EAFs only have some of the tiers, run only the configs you have data for:

```powershell
# I only have CC + CC_Aligned tiers
python ForcedAlignment\run_forced_alignment.py --configs 1

# I have CC + CC_Aligned + Gloss + Gloss_Labeling
python ForcedAlignment\run_forced_alignment.py --configs 1,3

# Run any comma-separated subset of {1,2,3,4,5}
python ForcedAlignment\run_forced_alignment.py --configs 3,4,5
```

### File layout rules (enforced by the orchestrator)

1. **EAF stem = MP4 stem.** `myclip.mp4` must match `myclip.eaf`. Stems can be any string (numeric like `42` or descriptive like `interview_oct03`) — the orchestrator uses `Path.stem` to pair them.
2. **Stems must be globally unique across all subfolders under `--video-root`.** [run_forced_alignment.py:148-155](run_forced_alignment.py#L148-L155) walks the tree with `rglob('*.mp4')` and raises `RuntimeError: Duplicate MP4 stems` if any name collides.
3. **EAFs go in `--eaf-dir`** (default `ForcedAlignment/elan_forced_alignment/`). All EAFs flat in one directory.
4. **MP4s go anywhere under `--video-root`** — flat, nested in subfolders, mixed depths all work because of `rglob`.

### Step-by-step — setting up your own dataset

```powershell
# 1. Place your EAFs (flat, one per clip)
mkdir ForcedAlignment\my_eafs
Copy-Item path\to\your\*.eaf ForcedAlignment\my_eafs\

# 2. Place your MP4s anywhere (organize however you like)
mkdir ForcedAlignment\my_videos
Copy-Item -Recurse path\to\your\mp4s\* ForcedAlignment\my_videos\

# 3. Preflight check — verifies stems match and required tiers are present
venv\Scripts\activate
python ForcedAlignment\run_forced_alignment.py `
  --eaf-dir ForcedAlignment\my_eafs `
  --video-root ForcedAlignment\my_videos `
  --configs 1 `
  --preflight-only

# 4. Smoke test on first 3 clips
python ForcedAlignment\run_forced_alignment.py `
  --eaf-dir ForcedAlignment\my_eafs `
  --video-root ForcedAlignment\my_videos `
  --configs 1 `
  --only-ids <stem1>,<stem2>,<stem3>

# 5. Full run
python ForcedAlignment\run_forced_alignment.py `
  --eaf-dir ForcedAlignment\my_eafs `
  --video-root ForcedAlignment\my_videos `
  --configs 1
```

### MP4 format requirements

- **Codec:** H.264 in `.mp4` container (tested). MediaPipe also accepts `.avi`, `.mov`, `.mkv` per [`pose-format`](https://github.com/sign-language-processing/pose) but the orchestrator only stages `.mp4`.
- **FPS:** any (tested at 25 FPS). MediaPipe Holistic processes frames independently; FPS only matters for time-to-frame conversion (`segmentation.py` derives FPS from the pose file).
- **Resolution:** any (MediaPipe normalizes internally). Larger frames → slower pose extraction.
- **Duration:** any. Tested on 3.5–10.3 s clips. The DP aligner handles arbitrary clip length.
- **Audio:** ignored (pose-only pipeline).

### EAF format requirements

- **ELAN .eaf XML** — any version readable by `xml.etree.ElementTree` (most ELAN 5.x / 6.x exports).
- **Tier structure:** required tiers must be **top-level `<TIER>`** elements (not subtiers). Scripts use `root.findall("TIER")` and match on `TIER_ID` attribute.
- **Annotation content:** UTF-8 (Thai and other non-ASCII characters OK).
- **Time format:** standard `TIME_ORDER` + `TIME_SLOT` with `TIME_VALUE` in milliseconds.
- **Token separator inside annotation text** depends on tier:
  - `CC` / `CC_Aligned` tiers → whitespace-split (`"hello world"` → 2 tokens)
  - `Gloss*` / `Gloss_Labeling*` tiers → pipe-split (`"hello|world|"` → 2 tokens, trailing empty dropped)
  - See [run_forced_alignment.py:55-95](run_forced_alignment.py#L55-L95) `CONFIGS` for exact behavior

### Tweaking other parameters

| Flag | Default | When to change |
| --- | --- | --- |
| `--model-name` | `multilingual` | Use `bsl` or `asl` if your language is closer to those — see main README [§Language tag selection](../README.md#language-tag-selection) |
| `--language-tag` | `"<en> <bfi>"` | Match your model. For TSL use multilingual + `"<en>"` |
| `--sign-b`, `--sign-o` | 30, 50 | SEA segmenter thresholds — tune if segmentation misses signs (lower = more segments) |
| `--gap-penalty` | 2.0 | Increase if DP groups too many non-contiguous segments |
| `--coverage-penalty` | 0.5 | Increase to force more uniform per-token durations |
| `--window-pad` | 0.5 | Increase (seconds) if your sentences often have signs slightly outside the annotation boundary |

For tier-name customization (e.g., your data uses `subtitle` instead of `CC`), edit the `CONFIGS` dict at [run_forced_alignment.py:55](run_forced_alignment.py#L55) — each `RunConfig` is a tuple of `(input_tier, gt_tier, output_tier, csv_name, token_mode)`.
