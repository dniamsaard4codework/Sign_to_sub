# Sign_to_sub — Thai Sign Language (TSL) Subtitle Alignment

โปรเจกต์สำหรับการจัดเรียง (align) คำบรรยาย (subtitle) ให้ตรงกับภาษามือไทย (TSL) ในวิดีโอ
โดยใช้ระบบ **SEA (Segment, Embed, and Align)** เป็นฐาน

> **Quick navigation:**
> - [Full Setup Guide (clone → first run)](#full-setup-guide-from-git-clone-to-first-run)
> - [Running the Pipeline (step-by-step)](#running-the-pipeline)
> - [Adapting for Multiple Videos](#adapting-for-multiple-videos)
> - [Evaluation Methodology](#evaluation-methodology)

## Upstream References

โปรเจกต์นี้สร้างขึ้นบนงานวิจัยต้นฉบับ:

| Component | Original Repository | Paper |
| --- | --- | --- |
| **SEA** | [J22Melody/SEA](https://github.com/J22Melody/SEA) | [arXiv:2512.08094](https://arxiv.org/abs/2512.08094) — Jiang et al. 2025 |
| **SignCLIP (fairseq)** | [J22Melody/fairseq](https://github.com/J22Melody/fairseq) (fork of [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)) | SignCLIP models for sign language embeddings |

---

## Hardware Requirements

| Component | Minimum | Tested on |
| --- | --- | --- |
| GPU VRAM | 8 GB | RTX 5060 Ti 16 GB |
| RAM | 16 GB | 64 GB |
| Storage (free) | 20 GB | — |
| CUDA | 11.8+ | 13.2 (driver 595.79) |
| GPU Architecture | Maxwell+ | Blackwell sm_120 |

> **RTX 40xx / 50xx (Ada/Blackwell):** ต้องใช้ PyTorch **cu128** เท่านั้น — cu126 จะ fallback ไป CPU โดยอัตโนมัติ (ช้า 10–100×)

> **ไม่มี GPU:** ระบบยังทำงานได้ แต่ขั้นตอน embedding (Steps 8–9) จะช้ามาก (~3–5 ชั่วโมง แทน ~5 นาที)

---

## Changes from Original SEA

ไฟล์ที่แก้ไขจาก [J22Melody/SEA](https://github.com/J22Melody/SEA) (commit `5aaf27d`):

| File | Changes |
| --- | --- |
| `SEA/align.py` | Added multi-model support (`live_model_name`, `live_language_tag`), pre-computed segmentation embedding loading, dynamic path resolution (removed hardcoded `/users/zifan/` path) |
| `SEA/align_dp.py` | Added `numba` import fallback — runs as plain Python if numba/LLVM unavailable |
| `SEA/config.py` | Added `--live_model_name` and `--live_language_tag` CLI arguments |
| `SEA/segmentation.py` | Fixed video path to use `os.path.abspath()`, changed `subprocess.run` to use `shlex.split()` instead of `shell=True` |

## Changes from Original fairseq_signclip

ไฟล์ที่แก้ไขจาก [J22Melody/fairseq](https://github.com/J22Melody/fairseq):

| File | Changes |
| --- | --- |
| `examples/MMPT/mmpt/models/mmfusion.py` | Path compatibility fixes |
| `examples/MMPT/mmpt/processors/dsprocessor.py` | Minor fix |
| `examples/MMPT/mmpt/processors/dsprocessor_sign.py` | Model loading fixes |
| `examples/MMPT/mmpt/processors/processor.py` | Path fixes |
| `examples/MMPT/mmpt/tasks/task.py` | Task loading fixes |
| `examples/MMPT/mmpt/utils/load_config.py` | Config path fixes |
| `examples/MMPT/scripts_bsl/extract_episode_features.py` | Path fixes |
| `retri/signclip_bsl/bobsl_islr_finetune_long_context.yaml` | Config path fixes |

## Project Structure

```
Sign_to_sub/
├── SEA/                            ← Modified SEA system
│   ├── align.py                    ← Main pipeline entry point
│   ├── align_dp.py                 ← DP alignment algorithm (@numba.njit)
│   ├── align_similarity.py         ← Similarity matrix computation
│   ├── config.py                   ← CLI arguments
│   ├── segmentation.py             ← Sign detection from pose
│   └── misc/
│       └── evaluate_sub_alignment.py  ← Original SEA evaluation (BOBSL-scale)
├── fairseq_signclip/               ← SignCLIP (clone separately — not in repo)
│   └── examples/MMPT/
│       ├── scripts_bsl/
│       │   └── extract_episode_features.py  ← Embedding extractor
│       └── runs/                   ← Model checkpoints (download separately)
│           ├── retri_bsl/          ← BSL model
│           ├── retri_v1_1/         ← Multilingual model
│           └── retri_asl/          ← ASL model
├── example_alignment/              ← Experiment data & scripts (TSL video 04)
│   ├── 04.mp4                      ← Source video
│   ├── 04.pose                     ← MediaPipe skeleton (358 MB)
│   ├── video_ids.txt               ← List of video IDs to process
│   ├── Test.eaf                    ← ELAN annotation (CC, CC_Input, Gloss_Input, …)
│   ├── extract_cc_from_eaf.py      ← EAF tier → VTT
│   ├── make_gloss_cc_vtt.py        ← Build Gloss_Input subtitle VTT
│   ├── fix_overlap_vtt.py          ← Post-process: clamp overlapping cue ends
│   ├── evaluate_all.py             ← Custom index-based evaluation
│   ├── evaluate_all_to_csv.py      ← Batch: overlap fix + eval → CSV
│   ├── align_gloss_labels.py       ← Task 2: token-level gloss alignment
│   ├── evaluate_gloss_labeling.py  ← Task 2: IoU evaluation
│   ├── add_vtt_tiers_to_eaf.py     ← Add all VTT results into comparison EAF
│   ├── plot_alignment.py           ← Timeline visualisation
│   ├── subtitles/                  ← CC_Input VTT (119 cues) — subtitle input
│   ├── subtitles_gloss_cc_time/    ← Gloss_Input text + CC timestamps
│   ├── segmentation_output/        ← Sign segments from segmentation.py
│   ├── segmentation_embedding/     ← Sign segment embeddings (.npy)
│   ├── subtitle_embedding/         ← Subtitle embeddings (.npy)
│   └── aligned_output_*/           ← Alignment results (7 experiments)
├── report/                         ← LaTeX report
├── Progress_*.md                   ← Progress notes (Thai)
└── README.md                       ← This file
```

---

## Full Setup Guide (from git clone to first run)

### Step 0 — Prerequisites (install once on this machine)

**Software you need before starting:**

| Software | Where to get | Notes |
| --- | --- | --- |
| Git | https://git-scm.com | |
| Python 3.11.x | https://astral.sh/uv or python.org | 3.12+ may break some deps |
| NVIDIA GPU Driver | NVIDIA website | For CUDA embedding |
| ELAN (optional) | https://archive.mpi.nl/tla/elan | For viewing EAF results |
| ffmpeg (optional) | https://ffmpeg.org | For video cropping only |

---

### Step 1 — Clone this Repository

```powershell
git clone https://github.com/dniamsaard4codework/Sign_to_sub.git
cd Sign_to_sub
```

This gives you:
- `SEA/` — modified alignment code
- `example_alignment/` — scripts and pre-computed data (`.pose`, `.npy`, `.eaf`)
- Everything **except** `fairseq_signclip/` which is large and cloned separately

---

### Step 2 — Create Python Virtual Environment

```powershell
# Use Python 3.11 specifically
python3.11 -m venv venv

# Activate (do this every time you open a new terminal)
venv\Scripts\activate
```

After activation you should see `(venv)` at the start of your prompt.

---

### Step 3 — Install Python Dependencies

```powershell
# Core processing
pip install pysrt webvtt-py lxml numpy==1.26.4 pympi-ling

# Pose and segmentation
pip install "mediapipe==0.10.21" pose-format
pip install "git+https://github.com/J22Melody/segmentation@bsl"

# Utilities
pip install beartype numba tqdm scikit-learn tabulate matplotlib sentence-transformers
```

> **mediapipe must be exactly 0.10.21** — 0.10.22+ changes the API and breaks pose extraction.

---

### Step 4 — Install PyTorch with CUDA

```powershell
# For RTX 30xx / 40xx / 50xx (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# For older GPUs (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  NVIDIA GeForce RTX ...
```

If you see `False`, your GPU driver or CUDA version does not match the PyTorch build.

---

### Step 5 — Clone and Install SignCLIP (fairseq_signclip)

```powershell
# From the Sign_to_sub root folder:
git clone https://github.com/J22Melody/fairseq.git fairseq_signclip
cd fairseq_signclip
pip install -e .
cd examples\MMPT
pip install -e .
cd ..\..\..\   # back to Sign_to_sub root
```

> **Patches already applied** in this repo's `fairseq_signclip/` once you clone. See the
> [Changes from Original fairseq_signclip](#changes-from-original-fairseq_signclip) table above
> for what was modified.

---

### Step 6 — Download SignCLIP Model Weights

Model checkpoints are **not** included in the repository (too large). Download manually:

```powershell
pip install gdown
cd fairseq_signclip\examples\MMPT

# Download all checkpoints into runs/
gdown --folder "https://drive.google.com/drive/folders/10q7FxPlicrfwZn7_FgtNqKFDiAJi6CTc?usp=sharing" -O .\runs
```

Then place each checkpoint in the path its YAML config expects:

```powershell
# BSL model
mkdir runs\retri_bsl\bobsl_islr_finetune
copy runs\bobsl_finetune_checkpoint_best.pt runs\retri_bsl\bobsl_islr_finetune\checkpoint_best.pt

# Multilingual model
mkdir runs\retri_v1_1\baseline_temporal
copy runs\baseline_temporal_checkpoint_best.pt runs\retri_v1_1\baseline_temporal\checkpoint_best.pt

# ASL model
mkdir runs\retri_asl\asl_finetune
copy runs\asl_finetune_checkpoint_best.pt runs\retri_asl\asl_finetune\checkpoint_best.pt
```

Final structure:

```
fairseq_signclip\examples\MMPT\runs\
├── retri_bsl\bobsl_islr_finetune\checkpoint_best.pt       ← BSL
├── retri_v1_1\baseline_temporal\checkpoint_best.pt        ← Multilingual
└── retri_asl\asl_finetune\checkpoint_best.pt              ← ASL
```

---

## Running the Pipeline

> All commands below assume:
> - Working directory: `C:\path\to\Sign_to_sub`
> - `venv\Scripts\activate` has been run
> - Video ID is `04` (video file `example_alignment\04.mp4`)

---

### Step 7 — Prepare video_ids.txt

> **Windows gotcha:** Do NOT use `echo 04 > video_ids.txt` — PowerShell writes UTF-16 BOM which Python cannot read. Use Python instead:

```powershell
python -c "open('example_alignment\\video_ids.txt','w',encoding='utf-8').write('04\n')"
```

For multiple videos, add one ID per line:

```
04
05
06
```

---

### Step 8 — Extract Subtitle VTT from EAF

EAF annotation file must contain a tier named **`CC_Input`** (119 entries aligned to CC_Aligned):

```powershell
cd example_alignment
python extract_cc_from_eaf.py Test.eaf subtitles\04.vtt --tier CC_Input
cd ..
```

Output: `example_alignment\subtitles\04.vtt` (119 cues, speech-timed)

For Gloss text experiments, also build the Gloss_Input VTT (timestamp from CC_Input, text from Gloss_Input tier):

```powershell
cd example_alignment
python make_gloss_cc_vtt.py
cd ..
```

Output: `example_alignment\subtitles_gloss_cc_time\04.vtt` (119 cues, 0 fallbacks)

---

### Step 9 — Pose Estimation (videos → .pose)

> Skip if `04.pose` already exists (358 MB file in `example_alignment\`)

```powershell
cd example_alignment
videos_to_poses --format mediapipe --directory . `
  --additional-config="model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true"
cd ..
```

Runtime: ~15 min on CPU for 11-minute video. Use `model_complexity=1` to speed up at some accuracy cost.

---

### Step 10 — Sign Segmentation

```powershell
cd SEA
python segmentation.py `
  --sign-b-threshold 30 --sign-o-threshold 50 `
  --num_workers 1 `
  --video_ids ..\example_alignment\video_ids.txt `
  --pose_dir ..\example_alignment `
  --save_dir ..\example_alignment\segmentation_output `
  --video_dir ..\example_alignment
cd ..
```

Output: `example_alignment\segmentation_output\E4s-1_30_50\04.eaf` (SIGN tier: ~2780 segments)

> **`--num_workers 1` is required on Windows** — multiprocessing with more workers causes path errors.

> **`--sign-b-threshold` and `--sign-o-threshold` must stay consistent** across all subsequent steps (segmentation embedding, alignment). Changing them requires re-running everything from this step.

---

### Step 11 — Extract Embeddings

Run once per model. All commands run from `fairseq_signclip\examples\MMPT\`.

```powershell
cd fairseq_signclip\examples\MMPT
```

#### 11a — Segmentation Embeddings (sign video → .npy)

These are **shared** across all subtitle experiments for the same model.

```powershell
# Multilingual model (used by B_MULTI, C_MULTI, C_MULTI_word)
python scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=segmentation --model_name multilingual `
  --pose_dir ..\..\..\example_alignment `
  --segmentation_dir ..\..\..\example_alignment\segmentation_output\E4s-1_30_50 `
  --save_dir ..\..\..\example_alignment\segmentation_embedding\sign_clip_multi

# ASL model (used by D_ASL, D_ASL_gloss, D_ASL_word)
python scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=segmentation --model_name asl `
  --pose_dir ..\..\..\example_alignment `
  --segmentation_dir ..\..\..\example_alignment\segmentation_output\E4s-1_30_50 `
  --save_dir ..\..\..\example_alignment\segmentation_embedding\sign_clip_asl

# BSL model (used by B2)
python scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=segmentation --model_name bsl --language_tag "<en> <bfi>" `
  --pose_dir ..\..\..\example_alignment `
  --segmentation_dir ..\..\..\example_alignment\segmentation_output\E4s-1_30_50 `
  --save_dir ..\..\..\example_alignment\segmentation_embedding\sign_clip
```

#### 11b — Subtitle Embeddings (text → .npy)

One per (model, subtitle text) combination:

```powershell
# Multilingual + CC text  (B_MULTI)
python scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=subtitle --model_name multilingual --language_tag "<en>" --batch_size=1024 `
  --subtitle_dir ..\..\..\example_alignment\subtitles `
  --save_dir ..\..\..\example_alignment\subtitle_embedding\sign_clip_multi

# Multilingual + Gloss text  (C_MULTI)
python scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=subtitle --model_name multilingual --language_tag "<en>" --batch_size=1024 `
  --subtitle_dir ..\..\..\example_alignment\subtitles_gloss_cc_time `
  --save_dir ..\..\..\example_alignment\subtitle_embedding\sign_clip_multi_gloss

# ASL + CC text  (D_ASL)
python scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=subtitle --model_name asl --language_tag "<en> <ase>" --batch_size=1024 `
  --subtitle_dir ..\..\..\example_alignment\subtitles `
  --save_dir ..\..\..\example_alignment\subtitle_embedding\sign_clip_asl

# ASL + Gloss text  (D_ASL_gloss)
python scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=subtitle --model_name asl --language_tag "<en> <ase>" --batch_size=1024 `
  --subtitle_dir ..\..\..\example_alignment\subtitles_gloss_cc_time `
  --save_dir ..\..\..\example_alignment\subtitle_embedding\sign_clip_asl_gloss

# BSL + CC text  (B2)
python scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=subtitle --model_name bsl --language_tag "<en> <bfi>" --batch_size=1024 `
  --subtitle_dir ..\..\..\example_alignment\subtitles `
  --save_dir ..\..\..\example_alignment\subtitle_embedding\sign_clip
```

> `C_MULTI_word` and `D_ASL_word` use `--live_embedding --tokenize_text_embedding` at alignment time — **no subtitle .npy needed** for those.

```powershell
cd ..\..\..   # back to Sign_to_sub root
```

---

### Step 12 — DP Alignment (per experiment)

Run from `SEA\`:

```powershell
cd SEA
```

#### C_MULTI ⭐ (Best configuration — Multilingual + Gloss text)

```powershell
python align.py --overwrite --mode=inference `
  --video_ids ..\example_alignment\video_ids.txt --num_workers 1 `
  --sign-b-threshold 30 --sign-o-threshold 50 `
  --dp_duration_penalty_weight 2 --dp_gap_penalty_weight 8 `
  --dp_max_gap 6 --dp_window_size 40 `
  --pr_subs_delta_bias_start 1.3 --pr_subs_delta_bias_end 1.0 `
  --similarity_measure sign_clip_embedding --similarity_weight 6 `
  --pr_sub_path ..\example_alignment\subtitles_gloss_cc_time `
  --segmentation_dir ..\example_alignment\segmentation_output `
  --subtitle_embedding_dir ..\example_alignment\subtitle_embedding\sign_clip_multi_gloss `
  --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip_multi `
  --save_dir ..\example_alignment\aligned_output_multi_gloss
```

#### C_MULTI_word (Multilingual + Gloss + word-level, live embedding)

```powershell
python align.py --overwrite --mode=inference `
  --video_ids ..\example_alignment\video_ids.txt --num_workers 1 `
  --sign-b-threshold 30 --sign-o-threshold 50 `
  --dp_duration_penalty_weight 2 --dp_gap_penalty_weight 8 `
  --dp_max_gap 6 --dp_window_size 40 `
  --pr_subs_delta_bias_start 1.3 --pr_subs_delta_bias_end 1.0 `
  --similarity_measure sign_clip_embedding --similarity_weight 6 `
  --live_embedding --tokenize_text_embedding --live_model_name multilingual `
  --pr_sub_path ..\example_alignment\subtitles_gloss_cc_time `
  --segmentation_dir ..\example_alignment\segmentation_output `
  --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip_multi `
  --save_dir ..\example_alignment\aligned_output_multi_gloss_word
```

#### B_MULTI (Multilingual + CC text)

```powershell
python align.py --overwrite --mode=inference `
  --video_ids ..\example_alignment\video_ids.txt --num_workers 1 `
  --sign-b-threshold 30 --sign-o-threshold 50 `
  --dp_duration_penalty_weight 2 --dp_gap_penalty_weight 8 `
  --dp_max_gap 6 --dp_window_size 40 `
  --pr_subs_delta_bias_start 1.8 --pr_subs_delta_bias_end 1.5 `
  --similarity_measure sign_clip_embedding --similarity_weight 6 `
  --pr_sub_path ..\example_alignment\subtitles `
  --segmentation_dir ..\example_alignment\segmentation_output `
  --subtitle_embedding_dir ..\example_alignment\subtitle_embedding\sign_clip_multi `
  --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip_multi `
  --save_dir ..\example_alignment\aligned_output_multi_b2
```

#### B2 (BSL + CC text, baseline)

```powershell
python align.py --overwrite --mode=inference `
  --video_ids ..\example_alignment\video_ids.txt --num_workers 1 `
  --sign-b-threshold 30 --sign-o-threshold 50 `
  --dp_duration_penalty_weight 2 --dp_gap_penalty_weight 8 `
  --dp_max_gap 6 --dp_window_size 40 `
  --pr_subs_delta_bias_start 1.8 --pr_subs_delta_bias_end 1.5 `
  --similarity_measure sign_clip_embedding --similarity_weight 6 `
  --pr_sub_path ..\example_alignment\subtitles `
  --segmentation_dir ..\example_alignment\segmentation_output `
  --subtitle_embedding_dir ..\example_alignment\subtitle_embedding\sign_clip `
  --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip `
  --save_dir ..\example_alignment\aligned_output_with_embedding_tuned
```

#### D_ASL, D_ASL_gloss, D_ASL_word

Same pattern — swap `sign_clip_multi*` → `sign_clip_asl*` and `bias 1.8/1.5`. See `example_alignment/README_TH.md` §7.7 for exact commands.

```powershell
cd ..   # back to Sign_to_sub root
```

---

### Step 13 — Overlap Fix + Evaluation

Runs overlap fix on all experiments, then evaluates against ground truth in one command:

```powershell
python example_alignment\evaluate_all_to_csv.py
```

Outputs:
- `example_alignment\aligned_output_*/04_no_overlap.vtt` — overlap-free VTTs
- `example_alignment\evaluation_task1_results.csv` — metrics for all 7 experiments (before + after fix)

Expected result for **C_MULTI after fix**: mean offset ≈ −0.16s, ±1s ≈ 74%, ±3s = 100%, overlap = 0%

---

### Step 14 — Task 2: Gloss Labeling Alignment (optional)

```powershell
# Align gloss tokens to sign segments
python example_alignment\align_gloss_labels.py

# Evaluate IoU against Gloss Labeling ground truth
python example_alignment\evaluate_gloss_labeling.py
```

Outputs:
- `example_alignment\gloss_labels_pred.csv` (889 rows)
- `example_alignment\gloss_labels_pred.vtt`
- `example_alignment\evaluation_gloss_labeling.csv`

Expected: Mean IoU ≈ 0.42, 93% temporal overlap, 0 fallbacks

---

### Step 15 — Build Comparison EAF + Timeline Plot

```powershell
# Add all 7 experiment VTTs as tiers in Test_comparison.eaf
python example_alignment\add_vtt_tiers_to_eaf.py --overwrite

# Generate timeline PNG (first 2 minutes)
python example_alignment\plot_alignment.py
```

Outputs:
- `example_alignment\Test_comparison.eaf` — ELAN file with 15 tiers (open in ELAN to compare)
- `example_alignment\figures\timeline_first_2min.png`

---

## Adapting for Multiple Videos

The pipeline is designed for a single video (`04`) out of the box. To run on multiple videos:

### 1. video_ids.txt — list all video IDs

```
04
05
06
```

All pipeline steps read this file and process each ID in a loop.

### 2. File naming convention

Every script expects files named `<video_id>.<ext>` in the same folder:

| File | Location |
| --- | --- |
| `<id>.mp4` | `example_alignment\` |
| `<id>.pose` | `example_alignment\` (output of videos_to_poses) |
| `<id>.vtt` | `example_alignment\subtitles\` (CC_Input) |
| `<id>.vtt` | `example_alignment\subtitles_gloss_cc_time\` (Gloss_Input) |
| `<id>.npy` | `example_alignment\segmentation_embedding\*\` |
| `<id>.npy` | `example_alignment\subtitle_embedding\*\` |

### 3. Scripts that need changes for multi-video evaluation

The following scripts in `example_alignment/` have **hardcoded single-video paths** that must be updated:

| Script | Hardcoded reference | What to change |
| --- | --- | --- |
| `evaluate_all.py` | `EAF_PATH`, `CC_VTT` | Change to accept `--video_id` argument or loop over `video_ids.txt` |
| `evaluate_all_to_csv.py` | Iterates `EXPERIMENTS` dict which has hardcoded VTT paths | Change `EXPERIMENTS` to be built dynamically per video ID |
| `add_vtt_tiers_to_eaf.py` | Source `Test.eaf` and output `Test_comparison.eaf` paths | Accept `--video_id` argument |
| `plot_alignment.py` | Reads `Test.eaf` directly | Accept `--video_id` argument |
| `align_gloss_labels.py` | Source EAF path | Accept `--video_id` argument |
| `evaluate_gloss_labeling.py` | Source EAF path | Accept `--video_id` argument |

### 4. `make_gloss_cc_vtt.py` — per-video EAF source

Currently reads from a single EAF. For multi-video, wrap in a loop:

```python
for video_id in video_ids:
    eaf_path = f"path/to/{video_id}.eaf"
    out_path = f"subtitles_gloss_cc_time/{video_id}.vtt"
    # ... process
```

### 5. Evaluation for multiple videos

For proper multi-video evaluation (matching original SEA metrics), you would use `SEA/misc/evaluate_sub_alignment.py` with:
- Ground truth aligned subtitle folder
- Predicted VTT folder
- `video_ids.txt` pointing to all test videos

See `SEA/scripts/evaluate.sh` for the original BOBSL evaluation call pattern.

---

## Evaluation Methodology

> **This project uses a custom evaluation, not the original SEA evaluation.**

| | Original SEA (`SEA/misc/evaluate_sub_alignment.py`) | This project (`example_alignment/evaluate_all.py`) |
| --- | --- | --- |
| **Scale** | BOBSL dataset — 20,000+ sentences | Single video — 119 cues |
| **Matching** | Text lookup or frame-level | **Index-based** (pred[i] ↔ gt[i]) |
| **Metrics** | frame accuracy, F1@0.1/0.25/0.5, abs mean/median start+end | signed mean/median offset, ±1s/±2s/±3s, overlap% |
| **End time** | Measured | Not measured |
| **Dependencies** | pysrt, webvtt, beartype, BOBSL path structure | stdlib only |

**Why we use index-based matching instead of text-lookup:**

`CC_Input` and `CC_Aligned` each have exactly 119 entries and are aligned by index (entry 0 ↔ entry 0, etc.). Text-lookup fails on ~50/119 entries because annotators modified the text when creating `CC_Aligned` (merged sentences, changed wording). Index-based matching captures all 119 pairs without text dependency.

**Why we don't use the original SEA frame-level metrics:**

The original evaluation requires frame-rate and label sequences reconstructed from subtitle timestamps, and is designed for datasets of thousands of videos. For a single-video evaluation, start-offset metrics (mean offset, ±Ns coverage) are more interpretable and comparable to the numbers reported in the SEA paper.

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `UnicodeDecodeError` on video_ids.txt | PowerShell `echo` writes UTF-16 | Use `python -c "open(...).write('04\n')"` |
| `torch.cuda.is_available()` returns False | Wrong PyTorch CUDA build | Reinstall with `--index-url .../cu128` |
| `mediapipe` import error | Wrong version | `pip install mediapipe==0.10.21` |
| `numba` JIT error on first run | LLVM not found | `pip install numba`; or falls back to pure Python |
| `FileNotFoundError: checkpoint_best.pt` | Wrong weights path | Check `runs/retri_*/*/checkpoint_best.pt` structure |
| Alignment produces all cues at same time | `--sign-b-threshold` mismatch with segmentation | Must use same threshold values as segmentation step |
| `evaluate_all.py` matched = 0 | EAF path wrong or wrong tier name | Check `EAF_PATH` in script and tier name `CC_Aligned` |
| `subprocess` error in segmentation.py on Windows | `shlex.quote` + `shell=True` | Already fixed in this repo (uses `shlex.split` + `shell=False`) |

---

## Quick Start (if all setup is done)

If environment, weights, and `.pose`/`.eaf` files are already in place:

```powershell
venv\Scripts\activate

# 1. Build subtitle inputs (if not done)
python example_alignment\extract_cc_from_eaf.py example_alignment\Test.eaf example_alignment\subtitles\04.vtt --tier CC_Input
python example_alignment\make_gloss_cc_vtt.py

# 2. Run best alignment (C_MULTI)
cd SEA
python align.py --overwrite --mode=inference --video_ids ..\example_alignment\video_ids.txt --num_workers 1 --sign-b-threshold 30 --sign-o-threshold 50 --dp_duration_penalty_weight 2 --dp_gap_penalty_weight 8 --dp_max_gap 6 --dp_window_size 40 --pr_subs_delta_bias_start 1.3 --pr_subs_delta_bias_end 1.0 --similarity_measure sign_clip_embedding --similarity_weight 6 --pr_sub_path ..\example_alignment\subtitles_gloss_cc_time --segmentation_dir ..\example_alignment\segmentation_output --subtitle_embedding_dir ..\example_alignment\subtitle_embedding\sign_clip_multi_gloss --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip_multi --save_dir ..\example_alignment\aligned_output_multi_gloss
cd ..

# 3. Overlap fix + evaluate
python example_alignment\evaluate_all_to_csv.py

# 4. Build comparison EAF
python example_alignment\add_vtt_tiers_to_eaf.py --overwrite
python example_alignment\plot_alignment.py
```

---

## License

SEA original code is under the license from [J22Melody/SEA](https://github.com/J22Melody/SEA).
Custom scripts in `example_alignment/` are part of this project.

