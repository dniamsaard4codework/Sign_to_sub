# Sign_to_sub — Thai Sign Language (TSL) Subtitle Alignment

โปรเจกต์สำหรับการจัดเรียง (align) คำบรรยาย (subtitle) ให้ตรงกับช่วงเวลาที่ผู้แปล
ภาษามือไทย (TSL) แสดงท่าทางในวิดีโอ — ใช้ระบบ **SEA (Segment, Embed, and Align)**
เป็นฐาน + ปรับให้รันบน Windows + GPU Blackwell + รองรับ TSL ด้วย SignCLIP
ทั้ง 3 โมเดล (BSL / Multilingual / ASL)

> **Quick navigation**
>
> - [Project Status & Latest Results](#project-status--latest-results)
> - [Hardware & Software Requirements](#hardware--software-requirements)
> - [Full Setup from Clone](#full-setup-from-clone)
> - [Pipeline Overview](#pipeline-overview)
> - [Task 1 — Subtitle Alignment](#task-1--subtitle-alignment)
> - [Task 2 — Gloss Labeling](#task-2--gloss-labeling)
> - [Ablation Study (Task 2: Gloss vs Gloss_Input)](#ablation-study-task-2-gloss-vs-gloss_input)
> - [Adapting for Multiple Videos](#adapting-for-multiple-videos)
> - [Troubleshooting](#troubleshooting)

---

## Upstream References

| Component | Original Repository | Paper |
| --- | --- | --- |
| **SEA** | [J22Melody/SEA](https://github.com/J22Melody/SEA) | [arXiv:2512.08094](https://arxiv.org/abs/2512.08094) — Jiang et al. 2025 |
| **SignCLIP (fairseq)** | [J22Melody/fairseq](https://github.com/J22Melody/fairseq) (fork of [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)) | SignCLIP models for sign-language embeddings |

---

## Project Status & Latest Results

โปรเจกต์ครอบคลุม **2 งานหลัก** บนวิดีโอตัวอย่าง 1 คลิป
"การเปรียบเทียบและเรียงลำดับ" (11.07 นาที) และมี progress reports 4 ฉบับ:

| รายงาน | วันที่ | เนื้อหาหลัก |
| --- | --- | --- |
| [Progress_20042026.md](Progress_20042026.md) | 20 เม.ย. 2569 | Initial pipeline — segmentation + B2 (BSL) baseline |
| [Progress_26042026.md](Progress_26042026.md) | 26 เม.ย. 2569 | 7 experiments + post-overlap fix + Task 2 prototype |
| [Progress_04052026.md](Progress_04052026.md) | 4 พ.ค. 2569 | เปลี่ยนเป็น `CC_Input` / `Gloss_Input` curated input + index-based eval |
| [Progress_09052026.md](Progress_09052026.md) | 9 พ.ค. 2569 | **Task 2 ablation: `Gloss` vs `Gloss_Input`** |

### Task 1 — Subtitle Alignment (current best run)

**Best experiment: C_MULTI ⭐** (Multilingual + Gloss text)

| Metric | C_MULTI (after overlap fix) |
| --- | --- |
| Total cues | 119 / 119 (index-based eval) |
| Mean start offset | **−0.16 s** |
| % within ±1 s | 73.9 % |
| % within ±2 s | 95.0 % |
| % within ±3 s | **100 %** |
| Frame accuracy (FPS=25) | 82.6 % |
| F1 @ 0.50 IoU | 88.2 % |
| Mean end offset | +0.23 s (post-fix) |
| Overlap rate | 0 % (was 88.1 %) |

ดู [evaluation_task1_results.csv](example_alignment/evaluation_task1_results.csv) สำหรับ
ตัวเลขทั้ง 7 experiments (BSL / Multilingual / ASL × CC text / Gloss text / word-level)

### Task 2 — Gloss Labeling (current best run)

**Recommended input: `--tier Gloss`** (ดู [Ablation Study](#ablation-study-task-2-gloss-vs-gloss_input))

| Metric | Tier `Gloss` (recommended) | Tier `Gloss_Input` (default in code) |
| --- | --- | --- |
| Predictions | 852 | 889 |
| Mean IoU | **0.4901** | 0.4199 |
| % IoU ≥ 0.5 | **48.4 %** | 38.9 % |
| % IoU ≥ 0.3 | **77.0 %** | 66.0 % |
| % any temporal overlap | **97.5 %** | 93.4 % |
| Fallback uniform sentences | 0 / 119 | 0 / 119 |

> ⚠️ **Caveat:** ตัวเลข exact-text-match ของ `Gloss` (65 %) มี structural
> leakage เพราะ GT `Gloss Labeling` ถูก build จาก `Gloss` token list —
> รายงาน metric เป็น **IoU เป็นหลัก**, ไม่ใช่ text match ดู
> [Progress_09052026.md](Progress_09052026.md) §6 — Ground-truth leakage check

---

## Hardware & Software Requirements

### Hardware

| Component | Minimum | Tested on |
| --- | --- | --- |
| GPU VRAM | 8 GB | RTX 5060 Ti 16 GB |
| RAM | 16 GB | 64 GB |
| Free disk space | 20 GB | — |
| CUDA driver | 11.8+ | 13.2 (driver 595.79) |
| GPU architecture | Maxwell or newer | Blackwell sm_120 |

> **RTX 40xx / 50xx (Ada / Blackwell):** ใช้ PyTorch wheel **cu128**
> เท่านั้น — cu126 จะ fall back ไป CPU เงียบๆ (ช้าลง 10–100×)
>
> **ไม่มี GPU:** Pipeline ยังทำงานได้ แต่ขั้น embedding ใช้เวลา
> ~3–5 ชั่วโมง แทน ~5 นาที

### Software (install ก่อน)

| Software | Source | Notes |
| --- | --- | --- |
| Git | <https://git-scm.com> | |
| Python **3.11.x** | <https://python.org> / uv | 3.12+ break บาง deps |
| NVIDIA GPU driver | <https://www.nvidia.com/Download/index.aspx> | สำหรับ CUDA |
| ELAN (optional) | <https://archive.mpi.nl/tla/elan> | ดู EAF results |
| ffmpeg (optional) | <https://ffmpeg.org> | crop video อย่างเดียว |

### Test environment specs

```text
OS              Windows 11 Pro
CPU             Intel Core Ultra 7 265K (20 cores)
RAM             64 GB
GPU             NVIDIA GeForce RTX 5060 Ti (17.1 GB VRAM, Blackwell sm_120)
CUDA Driver     595.79 (CUDA 13.2)
Python          3.11.15
Shell           PowerShell 5.1
```

### Pinned Python dependencies

| Package | Version | Why pinned |
| --- | --- | --- |
| `torch` | 2.11.0+cu128 | Blackwell-compatible only |
| `mediapipe` | **0.10.21 (exact)** | 0.10.22+ break pose API |
| `pose-format` | 0.12.3 | matches `videos_to_poses` |
| `numpy` | 1.26.4 | numba JIT compatibility |
| `numba` | latest | JIT for DP inner loops |
| `transformers` | (auto via SignCLIP) | tokenizer for SignCLIP |
| `pysrt`, `webvtt-py` | latest | VTT IO |
| `lxml`, `pympi-ling` | latest | EAF IO |
| `scikit-learn`, `tabulate` | latest | metrics |
| `matplotlib` | 3.10.8 | plot_alignment.py |

---

## Changes from Upstream

### `SEA/` (vs [J22Melody/SEA](https://github.com/J22Melody/SEA) commit `5aaf27d`)

| File | Changes |
| --- | --- |
| [SEA/align.py](SEA/align.py) | Multi-model support (`--live_model_name`, `--live_language_tag`), pre-computed segmentation embedding loading, removed hardcoded `/users/zifan/` path |
| [SEA/align_dp.py](SEA/align_dp.py) | `numba` import fallback — runs as plain Python if LLVM unavailable |
| [SEA/config.py](SEA/config.py) | New CLI args `--live_model_name`, `--live_language_tag` |
| [SEA/segmentation.py](SEA/segmentation.py) | Use `os.path.abspath()`, replaced `subprocess.run(shell=True)` with `shlex.split` |

### `fairseq_signclip/` (vs [J22Melody/fairseq](https://github.com/J22Melody/fairseq))

Path-fix patches in `examples/MMPT/mmpt/{models/mmfusion,processors/dsprocessor,
processors/dsprocessor_sign,processors/processor,tasks/task,utils/load_config}.py`
และ `examples/MMPT/scripts_bsl/extract_episode_features.py` รวมทั้ง YAML
configs ใน `retri/signclip_bsl/`

### Custom scripts (this repo)

ทั้งหมดอยู่ใน [example_alignment/](example_alignment/) — ดูตารางใน
[Pipeline Overview](#pipeline-overview)

---

## Project Structure

```text
Sign_to_sub/
├── SEA/                                    ← Modified SEA system
│   ├── align.py                            ← Main alignment entry
│   ├── align_dp.py                         ← DP algorithm (@numba.njit)
│   ├── align_similarity.py                 ← Similarity computation
│   ├── config.py                           ← CLI args
│   ├── segmentation.py                     ← Sign detection from pose
│   └── misc/evaluate_sub_alignment.py      ← Original SEA eval (BOBSL-scale)
│
├── fairseq_signclip/                       ← SignCLIP — clone separately, NOT in repo
│   └── examples/MMPT/
│       ├── scripts_bsl/extract_episode_features.py   ← Embedding extractor
│       └── runs/                                     ← Model checkpoints (download)
│           ├── retri_bsl/bobsl_islr_finetune/checkpoint_best.pt
│           ├── retri_v1_1/baseline_temporal/checkpoint_best.pt
│           └── retri_asl/asl_finetune/checkpoint_best.pt
│
├── example_alignment/                      ← Experiment data + scripts (TSL video 04)
│   │
│   │  ── Source files (in repo)
│   ├── 04.mp4                              ← Source video
│   ├── 04.pose                             ← MediaPipe pose (358 MB)
│   ├── Test.eaf, Test.pfsx                 ← ELAN annotation (CC, CC_Input, CC_Aligned, Gloss, Gloss_Input, Gloss Labeling)
│   ├── video_ids.txt                       ← `04` (one line)
│   │
│   │  ── Pipeline scripts
│   ├── extract_cc_from_eaf.py              ← EAF tier → VTT
│   ├── make_gloss_cc_vtt.py                ← Build Gloss_Input subtitle VTT
│   ├── make_gloss_input_tier.py            ← Build Gloss_Input tier in EAF (one-shot)
│   ├── merge_cc_to_updated_eaf.py          ← Copy CC tiers into existing EAF
│   ├── fix_overlap_vtt.py                  ← Clamp overlapping cue ends → 0% overlap
│   ├── align_gloss_labels.py               ← Task 2: token-level gloss alignment (with --tier flag)
│   ├── add_vtt_tiers_to_eaf.py             ← Build comparison EAF (15 tiers)
│   ├── add_best_to_eaf.py                  ← Build best-only EAF (Test_best.eaf)
│   ├── plot_alignment.py                   ← Timeline visualization
│   │
│   │  ── Evaluation scripts
│   ├── evaluate_all.py                     ← Task 1 index-based eval (single experiment)
│   ├── evaluate_all_to_csv.py              ← Task 1 batch: overlap fix + eval all 7
│   ├── evaluate_gloss_labeling.py          ← Task 2 IoU eval
│   │
│   │  ── Pre-computed inputs (in repo)
│   ├── subtitles/04.vtt                    ← CC_Input VTT (119 cues)
│   ├── subtitles_gloss_cc_time/04.vtt      ← Gloss_Input + CC_Input timestamps
│   ├── segmentation_output/E4s-1_30_50/    ← SIGN tier (2780 segments)
│   │
│   │  ── Embeddings (regenerable; 6 dirs)
│   ├── segmentation_embedding/{sign_clip,sign_clip_multi,sign_clip_asl}/04.npy
│   ├── subtitle_embedding/{sign_clip,sign_clip_multi,sign_clip_asl,
│   │                       sign_clip_multi_gloss,sign_clip_asl_gloss,
│   │                       sign_clip_multi_gloss_tokens}/04.npy
│   │
│   │  ── Alignment outputs (regenerable; 7 dirs × 2 variants)
│   ├── aligned_output_with_embedding_tuned/{04.vtt,04_no_overlap.vtt}    ← B2
│   ├── aligned_output_multi_b2/{04.vtt,04_no_overlap.vtt}                ← B_MULTI
│   ├── aligned_output_multi_gloss/{04.vtt,04_no_overlap.vtt}             ← C_MULTI ⭐
│   ├── aligned_output_multi_gloss_word/{04.vtt,04_no_overlap.vtt}        ← C_MULTI_word
│   ├── aligned_output_asl_b2/{04.vtt,04_no_overlap.vtt}                  ← D_ASL
│   ├── aligned_output_asl_gloss/{04.vtt,04_no_overlap.vtt}               ← D_ASL_gloss
│   ├── aligned_output_asl_gloss_word/{04.vtt,04_no_overlap.vtt}          ← D_ASL_word
│   │
│   │  ── Task 2 outputs (default = Gloss_Input)
│   ├── gloss_labels_pred.csv               ← 889 predictions
│   ├── gloss_labels_pred.vtt
│   │
│   │  ── Ablation outputs (Progress_09052026)
│   └── ablation/
│       ├── gloss_labels_pred__Gloss.{csv,vtt}            ← 852 preds (recommended)
│       ├── gloss_labels_pred__Gloss_Input.{csv,vtt}      ← 889 preds
│       ├── 04_gloss_pred__{Gloss,Gloss_Input}.eaf
│       ├── evaluation_gloss_labeling__{Gloss,Gloss_Input}.csv
│       └── diagnostics.json
│
├── Test_comparison.eaf                     ← All 15 tiers (built by add_vtt_tiers_to_eaf.py)
├── Test_best.eaf                           ← Best-only EAF (built by add_best_to_eaf.py)
│
├── Progress_*.md                           ← 4 progress reports (Thai)
├── README.md                               ← This file
├── arXiv-2512.08094v1/                     ← SEA paper (PDF)
├── report/                                 ← LaTeX project report
└── venv/                                   ← Python virtualenv (created by you)
```

---

## Full Setup from Clone

> **Goal:** ทำตามทุกขั้นตอน → ได้ environment ที่รัน pipeline ได้ครบ
> ทั้ง Task 1, Task 2, ablation

### Step 0 — Verify prerequisites

```powershell
git --version            # Git 2.30+
python --version         # 3.11.x
nvidia-smi               # GPU driver visible
```

### Step 1 — Clone this repository

```powershell
git clone https://github.com/dniamsaard4codework/Sign_to_sub.git
cd Sign_to_sub
```

ในที่นี้ assume working directory คือ `C:\path\to\Sign_to_sub` ทุกคำสั่ง

### Step 2 — Create & activate virtual environment

```powershell
python3.11 -m venv venv
venv\Scripts\activate
# ทุก terminal ใหม่ต้องรัน activate ก่อนเสมอ
```

ควรเห็น `(venv)` หน้า prompt

### Step 3 — Install Python dependencies

```powershell
# Core
pip install pysrt webvtt-py lxml numpy==1.26.4 pympi-ling

# Pose / segmentation
pip install "mediapipe==0.10.21" pose-format
pip install "git+https://github.com/J22Melody/segmentation@bsl"

# Utilities
pip install beartype numba tqdm scikit-learn tabulate matplotlib sentence-transformers
```

> **mediapipe ต้องเป็น 0.10.21 เป๊ะ** — 0.10.22+ ทำให้ pose extraction พัง

### Step 4 — Install PyTorch with CUDA

```powershell
# RTX 30xx / 40xx / 50xx (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# GPU เก่า (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

ตรวจ GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expected: True  NVIDIA GeForce RTX ...
```

ถ้าได้ `False` → driver / CUDA build ไม่ตรง → reinstall ด้วย URL ที่ถูก

### Step 5 — Clone & install SignCLIP (fairseq_signclip)

```powershell
git clone https://github.com/J22Melody/fairseq.git fairseq_signclip
cd fairseq_signclip
pip install -e .
cd examples\MMPT
pip install -e .
cd ..\..\..
```

> **Patch files:** `fairseq_signclip` ในเครื่องคุณยังเป็น vanilla upstream
> ต้อง replace ตามตาราง [Changes from Upstream](#changes-from-upstream)
> โดยตรง — patches ที่ใช้ใน repo นี้อยู่ใน `fairseq_signclip/` ของ git
> tree ที่ผ่าน clone มา (ใน `Sign_to_sub` repo ไม่มีโฟลเดอร์ vendoring
> `fairseq_signclip` แยก) ถ้าใครต้องการ patch ที่ผ่านแล้วจริง ๆ ให้
> diff กับ commit ปัจจุบัน

### Step 6 — Download SignCLIP model checkpoints

```powershell
pip install gdown
cd fairseq_signclip\examples\MMPT
gdown --folder "https://drive.google.com/drive/folders/10q7FxPlicrfwZn7_FgtNqKFDiAJi6CTc?usp=sharing" -O .\runs

# จัดวางตาม path ที่ YAML config อ้างอิง
mkdir runs\retri_bsl\bobsl_islr_finetune
copy runs\bobsl_finetune_checkpoint_best.pt runs\retri_bsl\bobsl_islr_finetune\checkpoint_best.pt

mkdir runs\retri_v1_1\baseline_temporal
copy runs\baseline_temporal_checkpoint_best.pt runs\retri_v1_1\baseline_temporal\checkpoint_best.pt

mkdir runs\retri_asl\asl_finetune
copy runs\asl_finetune_checkpoint_best.pt runs\retri_asl\asl_finetune\checkpoint_best.pt
cd ..\..\..
```

โครงสร้างปลายทาง:

```text
fairseq_signclip\examples\MMPT\runs\
├── retri_bsl\bobsl_islr_finetune\checkpoint_best.pt       ← BSL
├── retri_v1_1\baseline_temporal\checkpoint_best.pt        ← Multilingual
└── retri_asl\asl_finetune\checkpoint_best.pt              ← ASL
```

### Step 7 — Verify install

```powershell
# 1. Python deps
python -c "import torch, mediapipe, pose_format, numpy, pympi, lxml; print('OK')"

# 2. SignCLIP imports
cd fairseq_signclip\examples\MMPT
python -c "from extract_episode_features import load_model, embed_text; print('OK')"
cd ..\..\..

# 3. Test data exists
python -c "import pathlib; p=pathlib.Path('example_alignment/Test.eaf'); print(p.exists(), p.stat().st_size)"
# expected: True 660439

# 4. Pose file exists (already in repo, 358 MB)
python -c "import pathlib; p=pathlib.Path('example_alignment/04.pose'); print(p.exists())"
```

ถ้าทุกอันได้ `OK` หรือ `True` → setup เรียบร้อย

---

## Pipeline Overview

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                              INPUT (in repo)                               │
│   04.mp4 + Test.eaf (CC_Input, CC_Aligned, Gloss, Gloss_Input,             │
│                       Gloss Labeling)                                      │
└──────────┬─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────┐
│ Step A: Extract CC_Input → VTT     │  extract_cc_from_eaf.py --tier CC_Input
│  → subtitles/04.vtt (119 cues)     │
├────────────────────────────────────┤
│ Step B: Make Gloss_Input VTT       │  make_gloss_cc_vtt.py
│  → subtitles_gloss_cc_time/04.vtt  │  (Gloss_Input text + CC_Input timestamps)
├────────────────────────────────────┤
│ Step C: Pose Estimation            │  videos_to_poses
│  → 04.pose (358 MB)                │  [in repo — skip if exists]
├────────────────────────────────────┤
│ Step D: Sign Segmentation          │  SEA/segmentation.py
│  → segmentation_output/.../04.eaf  │  (SIGN: 2780, SENTENCE: 418)
└──────────┬─────────────────────────┘
           │
   ┌───────┴────────┐
   ▼                ▼
┌───────────────┐ ┌──────────────────────────────────────┐
│ Step E1       │ │ Step E2: SignCLIP subtitle embed     │
│ Sign embed    │ │  - sign_clip       (BSL)  CC_Input   │
│ extract_      │ │  - sign_clip_multi (Multi) CC_Input  │
│ episode_      │ │  - sign_clip_asl   (ASL)  CC_Input   │
│ features.py   │ │  - sign_clip_multi_gloss  Gloss_Input│
│ → *.npy       │ │  - sign_clip_asl_gloss    Gloss_Input│
│ (2780 × 768)  │ │ → subtitle_embedding/*/04.npy        │
└──────┬────────┘ │   (119, 768) each                    │
       │          └────────────┬─────────────────────────┘
       └─────────┬─────────────┘
                 ▼
┌─────────────────────────────────────────────┐
│  Task 1: DP Alignment + Overlap Fix         │  SEA/align.py + fix_overlap_vtt.py
│  - 7 experiments × 2 variants               │  → aligned_output_*/04{,_no_overlap}.vtt
│  - Index-based eval (119/119)               │  → evaluation_task1_results.csv
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Task 2: Gloss Token Alignment              │  align_gloss_labels.py --tier {Gloss,Gloss_Input}
│  - per-sentence monotonic DP                │  → gloss_labels_pred.csv
│  - IoU vs Gloss Labeling tier               │  → evaluation_gloss_labeling.csv
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  Comparison + Visualization                 │  add_vtt_tiers_to_eaf.py
│                                             │  add_best_to_eaf.py
│                                             │  plot_alignment.py
│                                             │  → Test_comparison.eaf, Test_best.eaf,
│                                             │    figures/timeline_first_2min.png
└─────────────────────────────────────────────┘
```

### Tiers ใน Test.eaf — quick reference

| Tier | Entries | บทบาท |
| --- | --- | --- |
| `CC` | 172 | คำบรรยายดิบจากเสียงพูด — **ไม่ใช้** |
| `CC_Input` | 119 | curated CC subtitle — **input ของ Task 1** |
| `CC_Aligned` | 119 | manual alignment — **GT ของ Task 1** |
| `Gloss` | 119 (852 tokens) | gloss tier เดิม — **input ทางเลือกที่ดีกว่าของ Task 2** |
| `Gloss_Input` | 119 (889 tokens) | curated gloss — **input default ของ Task 2** |
| `Gloss Labeling` | 852 | per-sign-gesture annotation — **GT ของ Task 2** |

---

## Task 1 — Subtitle Alignment

### Task 1 — What it does

> ปรับ timestamp ของ `CC_Input` (119 cues, จับเวลาจากเสียงพูด) ให้ตรงกับ
> ช่วงเวลาที่ผู้แปลแสดงท่ามือจริง — ผลที่ได้คือ VTT ที่ใช้ในวิดีโอผู้พิการ
> ทางการได้ยินได้ตรงกับท่ามือ

### Task 1 — Input / Output

| | |
| --- | --- |
| Input video | `example_alignment/04.mp4` |
| Input subtitle | `example_alignment/Test.eaf` → tier `CC_Input` (119 cues) |
| GT for evaluation | tier `CC_Aligned` (119 entries, index-based pairing) |
| Output | `example_alignment/aligned_output_*/04{,_no_overlap}.vtt` |

### How to run — full Task 1 (7 experiments)

```powershell
venv\Scripts\activate

# 1. Extract subtitle inputs
python example_alignment\extract_cc_from_eaf.py example_alignment\Test.eaf example_alignment\subtitles\04.vtt --tier CC_Input
python example_alignment\make_gloss_cc_vtt.py
# Verify: subtitles\04.vtt = 119 cues; subtitles_gloss_cc_time\04.vtt = 119 cues, 0 fallback

# 2. Pose estimation (skip if 04.pose มีอยู่แล้ว)
cd example_alignment
videos_to_poses --format mediapipe --directory . `
  --additional-config="model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true"
cd ..

# 3. Sign segmentation
cd SEA
python segmentation.py `
  --sign-b-threshold 30 --sign-o-threshold 50 `
  --num_workers 1 `
  --video_ids ..\example_alignment\video_ids.txt `
  --pose_dir ..\example_alignment `
  --save_dir ..\example_alignment\segmentation_output `
  --video_dir ..\example_alignment
cd ..

# 4. Embeddings (run all 3 sign embeddings + 5 subtitle embeddings — see below)

# 5. Alignment (7 experiments — see below)

# 6. Overlap fix + evaluation (one command)
python example_alignment\evaluate_all_to_csv.py
```

#### Step 4 detail — Embeddings

```powershell
cd fairseq_signclip\examples\MMPT

# Sign embeddings (3 — used by all experiments)
python scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=segmentation --model_name multilingual `
  --pose_dir ..\..\..\example_alignment `
  --segmentation_dir ..\..\..\example_alignment\segmentation_output\E4s-1_30_50 `
  --save_dir ..\..\..\example_alignment\segmentation_embedding\sign_clip_multi

python scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=segmentation --model_name asl `
  --pose_dir ..\..\..\example_alignment `
  --segmentation_dir ..\..\..\example_alignment\segmentation_output\E4s-1_30_50 `
  --save_dir ..\..\..\example_alignment\segmentation_embedding\sign_clip_asl

python scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=segmentation --model_name bsl --language_tag "<en> <bfi>" `
  --pose_dir ..\..\..\example_alignment `
  --segmentation_dir ..\..\..\example_alignment\segmentation_output\E4s-1_30_50 `
  --save_dir ..\..\..\example_alignment\segmentation_embedding\sign_clip

# Subtitle embeddings (5 — one per experiment that uses precomputed text emb)
python scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=subtitle --model_name bsl --language_tag "<en> <bfi>" --batch_size=1024 `
  --subtitle_dir ..\..\..\example_alignment\subtitles `
  --save_dir ..\..\..\example_alignment\subtitle_embedding\sign_clip
# repeat for: multilingual+CC, multilingual+Gloss, asl+CC, asl+Gloss

cd ..\..\..
```

#### Step 5 detail — 7 alignment experiments

Common flags ทุก experiment:

- `--sign-b-threshold 30 --sign-o-threshold 50` (ต้องตรงกับค่าที่ segmentation ใช้)
- `--dp_duration_penalty_weight 2 --dp_gap_penalty_weight 8 --dp_max_gap 6 --dp_window_size 40`
- `--similarity_measure sign_clip_embedding --similarity_weight 6`
- `--video_ids ..\example_alignment\video_ids.txt --num_workers 1`

##### C_MULTI ⭐ (Best — Multilingual + Gloss text)

```powershell
cd SEA
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
cd ..
```

##### Other 6 experiments — parameter table

| Experiment | Subtitle path | Subtitle emb dir | Sign emb dir | bias start/end | extra flags | save_dir |
| --- | --- | --- | --- | --- | --- | --- |
| **B2** (BSL, CC text) | `subtitles` | `sign_clip` | `sign_clip` | 1.8 / 1.5 | — | `aligned_output_with_embedding_tuned` |
| **B_MULTI** (Multi, CC text) | `subtitles` | `sign_clip_multi` | `sign_clip_multi` | 1.8 / 1.5 | — | `aligned_output_multi_b2` |
| **C_MULTI** ⭐ (Multi, Gloss text) | `subtitles_gloss_cc_time` | `sign_clip_multi_gloss` | `sign_clip_multi` | 1.3 / 1.0 | — | `aligned_output_multi_gloss` |
| **C_MULTI_word** (Multi, Gloss, word-level) | `subtitles_gloss_cc_time` | _(omit)_ | `sign_clip_multi` | 1.3 / 1.0 | `--live_embedding --tokenize_text_embedding --live_model_name multilingual` | `aligned_output_multi_gloss_word` |
| **D_ASL** (ASL, CC text) | `subtitles` | `sign_clip_asl` | `sign_clip_asl` | 1.8 / 1.5 | — | `aligned_output_asl_b2` |
| **D_ASL_gloss** (ASL, Gloss text) | `subtitles_gloss_cc_time` | `sign_clip_asl_gloss` | `sign_clip_asl` | 1.3 / 1.0 | — | `aligned_output_asl_gloss` |
| **D_ASL_word** (ASL, Gloss, word-level) | `subtitles_gloss_cc_time` | _(omit)_ | `sign_clip_asl` | 1.3 / 1.0 | `--live_embedding --tokenize_text_embedding --live_model_name asl --live_language_tag "<en> <ase>"` | `aligned_output_asl_gloss_word` |

> ⚠️ **`--segmentation_dir` ต้องชี้ที่ parent dir** เช่น `segmentation_output`
> ไม่ใช่ `segmentation_output\E4s-1_30_50` — `align.py` ต่อ subdirectory
> เองจาก threshold parameters
>
> ⚠️ **`--num_workers 1` บน Windows** — multiprocessing > 1 จะ path-error
>
> ⚠️ **C_MULTI_word / D_ASL_word ไม่ต้องการ subtitle .npy** — embedding
> ทำที่ alignment time ผ่าน `--live_embedding --tokenize_text_embedding`

### How Task 1 works internally

#### Step D: Sign segmentation (E4s-1)

อ่าน [Progress_04052026.md §8](Progress_04052026.md) สำหรับรายละเอียด

- Input: `04.pose` (MediaPipe holistic, 543 landmarks)
- Output: `segmentation_output/E4s-1_30_50/04.eaf` มี tier `SIGN` (2,780) + `SENTENCE` (418)
- Algorithm: bidirectional GRU + **--sign-b-threshold 30, --sign-o-threshold 50**

#### Step E: SignCLIP embeddings

- **Sign embeddings** (`segmentation_embedding/*/04.npy`): `(2780, 768)` — vector ต่อ SIGN segment
- **Subtitle embeddings** (`subtitle_embedding/*/04.npy`): `(119, 768)` — vector ต่อ subtitle cue
- Multi-model: BSL / Multilingual / ASL — แต่ละโมเดลใช้ language_tag ต่างกัน:

| Model | language_tag |
| --- | --- |
| BSL | `<en> <bfi>` |
| Multilingual | `<en>` (subtitle) / not needed at sign extraction |
| ASL | `<en> <ase>` |

#### Step F: DP Alignment

State: `dp[i][j]` = ต้นทุนต่ำสุดที่ assign cues 1..i โดย cue i จบที่ segment index j

Cost function:

$$C(i, k, j) = \underbrace{|\text{cue}_i.\text{start} - \text{seg}_k.\text{start}|}_{\text{start align}} + |\text{cue}_i.\text{end} - \text{seg}_j.\text{end}| + w_D|\text{cue\_dur} - \text{group\_dur}| + w_G \cdot \text{gap}(k,j) - w_S \cdot \text{sim\_cum}[i][k][j]$$

| Term | Default weight | Description |
| --- | --- | --- |
| Start / end alignment | 1 | ดึง cue ให้ใกล้ cue เดิม |
| Duration penalty $w_D$ | 2 | ลงโทษถ้า group_dur ≠ cue_dur |
| Gap penalty $w_G$ | 8 | ลงโทษ gap ระหว่าง segments — สำคัญที่สุด |
| Similarity reward $w_S$ | 6 | ลด cost เมื่อ SignCLIP similarity สูง |

ใช้ **sliding window** size 40 (`--dp_window_size 40`) เพื่อจำกัด search
space → complexity $O(M \cdot W^2) \approx 190K$ operations,
และ Numba `@njit` → finishes ในไม่ถึง 1 วินาที

#### Step G: Overlap fix (post-processing)

DP **ไม่มี non-overlap constraint** → output มี overlap 86–88%
[fix_overlap_vtt.py](example_alignment/fix_overlap_vtt.py) แก้ด้วย
single-pass clamp:

```python
for i in range(len(cues) - 1):
    if cues[i].end > cues[i + 1].start:
        cues[i].end = cues[i + 1].start   # clamp
```

> **ทำไมแตะแค่ end ไม่แตะ start?** Start คือสิ่งที่ DP คำนวณมาอย่าง
> ระมัดระวัง — เป็น "best estimate" ว่าผู้แปลเริ่มท่ามือเมื่อไร
> Metric หลักวัด start offset เท่านั้น → ผลก่อน-หลัง overlap fix
> เหมือนกันใน start metrics

### How to evaluate Task 1

```powershell
python example_alignment\evaluate_all_to_csv.py
```

ทำสามอย่างใน command เดียว:

1. Generate `04_no_overlap.vtt` ทุก experiment (clamp end times)
2. Evaluate ทั้ง pre/post overlap variants ด้วย **index-based matching**
3. Write [evaluation_task1_results.csv](example_alignment/evaluation_task1_results.csv) (14 rows)

#### Why index-based matching (vs text lookup)

| | Text lookup | Index-based (this project) |
| --- | --- | --- |
| Match | predicted text → GT entry | `pred[i] ↔ gt[i]` |
| Coverage on Test.eaf | 69 / 172 | **119 / 119** |
| ปัญหา | annotators เปลี่ยน wording ทำให้ lookup fail | ต้องการ pred / GT จำนวนเท่ากัน |

#### Metrics produced

| Metric | คำอธิบาย |
| --- | --- |
| `mean_off_s`, `median_off_s` | start offset (signed, sec) |
| `mean_off_abs`, `median_off_abs` | start offset (absolute) |
| `mean_end_off`, `mean_end_off_abs` | end offset |
| `w1_pct` / `w2_pct` / `w3_pct` | % cues within ±1 / ±2 / ±3 sec |
| `overlap_pct` | % consecutive pairs ที่ overlap |
| `frame_acc` | frame accuracy (FPS=25, label-wise) |
| `f1_10` / `f1_25` / `f1_50` | F1 @ IoU ≥ 0.10 / 0.25 / 0.50 |

#### Expected results (สำหรับ video 04 ปัจจุบัน)

ดู [evaluation_task1_results.csv](example_alignment/evaluation_task1_results.csv) — สรุป:

| Experiment | mean off | ±1s | ±2s | ±3s | F1@0.50 | overlap (post-fix) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B2 | +0.26s | 76% | 96% | 99% | 88.2% | 0% |
| B_MULTI | +0.25s | 71% | 93% | 99% | 84.9% | 0% |
| **C_MULTI** ⭐ | **−0.16s** | 74% | 95% | **100%** | 88.2% | 0% |
| C_MULTI_word | −0.23s | 74% | 94% | 100% | **89.1%** | 0% |
| D_ASL | +0.38s | 61% | 87% | 98% | 77.3% | 0% |
| D_ASL_gloss | −0.12s | 71% | 91% | 99% | 81.5% | 0% |
| D_ASL_word | −0.13s | 69% | 90% | 99% | 84.9% | 0% |

---

## Task 2 — Gloss Labeling

### Task 2 — What it does

> Task 1 สร้าง subtitle ระดับ "ประโยค" — Task 2 ลงลึกกว่า: align
> **gloss token แต่ละตัว** (เช่น "ผายมือ", "เด็ก", "เรียน") กับ
> **SIGN segment ระดับท่ามือเดี่ยว** ภายใน sentence ของ gloss tier
> ผลที่ได้คือ annotation ระดับท่าทางที่ใกล้กับ `Gloss Labeling` GT

### Task 2 — Input / Output

| | |
| --- | --- |
| Input EAF | `example_alignment/Test.eaf` |
| Input tier | **`Gloss_Input`** (default) หรือ **`Gloss`** (recommended — ดู ablation) |
| Sign segments | `segmentation_output/E4s-1_30_50/04.eaf` (SIGN tier, 2780) |
| Sign embeddings | `segmentation_embedding/sign_clip_multi/04.npy` (2780, 768) |
| GT for evaluation | tier `Gloss Labeling` (852 entries, per-gesture) |
| Output | `gloss_labels_pred.{csv,vtt}` + `04_gloss_pred.eaf` (tier `GLOSS_LABEL_PRED`) |

### How to run — Task 2

#### Default (uses `Gloss_Input` tier — backwards compatible with Progress_04052026)

```powershell
venv\Scripts\activate
python example_alignment\align_gloss_labels.py
python example_alignment\evaluate_gloss_labeling.py
```

#### Recommended (uses `Gloss` tier — ดู ablation results)

```powershell
python example_alignment\align_gloss_labels.py `
  --tier Gloss `
  --out-csv example_alignment\ablation\gloss_labels_pred__Gloss.csv `
  --out-vtt example_alignment\ablation\gloss_labels_pred__Gloss.vtt `
  --out-eaf example_alignment\ablation\04_gloss_pred__Gloss.eaf `
  --cache   example_alignment\subtitle_embedding\sign_clip_multi_gloss_tokens\04__Gloss.npz

python example_alignment\evaluate_gloss_labeling.py `
  --pred-csv example_alignment\ablation\gloss_labels_pred__Gloss.csv `
  --out-csv  example_alignment\ablation\evaluation_gloss_labeling__Gloss.csv
```

### How Task 2 works internally

```text
Gloss sentence (start_s, end_s, "ผายมือ เด็ก เรียน")
    │
    ├─► tokenize on whitespace → ["ผายมือ", "เด็ก", "เรียน"]   (T tokens)
    │
    ├─► restrict candidate SIGN segments
    │     mid in [start_s, end_s]   (±0.5s pad if empty)        (K segments)
    │
    ├─► embed each token via SignCLIP multilingual text encoder
    │     → token_embs (T × 768), cached to .npz
    │
    ├─► sim_matrix (T × K) = cosine + row softmax
    │
    ├─► monotonic DP per sentence:
    │     dp[t][j] = min over k of {
    │         dp[t-1][k-1]
    │       + (-Σ sim[t-1, k-1..j-1])              ← negative similarity
    │       + gap_penalty   × inter_segment_gap_total
    │       + coverage_pen. × |group_dur − sentence_dur/T|
    │     }
    │
    └─► backtrack → assign each token to (k_start, k_end) range
          → emit (seg[k_start].start, seg[k_end].end, token)
```

**Complexity:** $O(T \cdot K^2)$ per sentence — T~7, K~30 → finishes ทั้ง 119 sentences ในไม่ถึง 1 วินาที

### Task 2 default parameters

| Flag | Default | ผลกระทบ |
| --- | --- | --- |
| `--tier` | `Gloss_Input` | source tier ใน EAF |
| `--model-name` | `multilingual` | SignCLIP variant |
| `--language-tag` | `<en> <bfi>` | language tag |
| `--gap-penalty` | 2.0 | ลด group ที่มี gap |
| `--coverage-penalty` | 0.5 | บังคับ duration เฉลี่ย |
| `--window-pad` | 0.5 | seconds เผื่อ window ขยาย |

### How to evaluate Task 2

[evaluate_gloss_labeling.py](example_alignment/evaluate_gloss_labeling.py)
ใช้ **best-IoU pairing**: สำหรับ prediction แต่ละตัว หา GT entry ที่
overlap สูงสุดด้วย IoU แล้วบันทึก IoU + signed offset + text-match flag

#### Aggregates ที่คำนวณ

| Metric | คำอธิบาย |
| --- | --- |
| Mean / Median IoU | average overlap quality |
| % IoU ≥ 0.5 / 0.3 | predictions ที่อยู่ในระดับใช้ได้ / ใช้พอได้ |
| % any temporal overlap | predictions ที่แตะ GT |
| Mean signed start/end offset | timing bias |
| Exact text match (overlapping pairs) | string equality (มี leakage component — ดูคำเตือน) |

#### Expected results

| Tier | Predictions | Mean IoU | %IoU≥0.5 | %any overlap |
| --- | ---: | ---: | ---: | ---: |
| `Gloss_Input` (default) | 889 | 0.4199 | 38.9% | 93.4% |
| `Gloss` (recommended) | 852 | **0.4901** | **48.4%** | **97.5%** |

---

## Ablation Study (Task 2: Gloss vs Gloss_Input)

> **Full report:** [Progress_09052026.md](Progress_09052026.md)
> **All output files:** [example_alignment/ablation/](example_alignment/ablation/)

### TL;DR

`Gloss` tier ชนะ `Gloss_Input` tier **ทุก metric** ใน Task 2 alignment

| Metric | `Gloss` | `Gloss_Input` | Δ |
| --- | ---: | ---: | ---: |
| Mean IoU | **0.4901** | 0.4199 | +7.0 pp |
| % IoU ≥ 0.5 | **48.4%** | 38.9% | +9.4 pp |
| % IoU ≥ 0.3 | **77.0%** | 66.0% | +11.0 pp |
| % zero overlap | **2.5%** | 6.6% | −4.1 pp |
| Mean abs start offset | **0.188 s** | 0.212 s | −24 ms |

### How to reproduce the ablation

```powershell
venv\Scripts\activate

# Variant A: --tier Gloss
python example_alignment\align_gloss_labels.py `
  --tier Gloss `
  --out-csv example_alignment\ablation\gloss_labels_pred__Gloss.csv `
  --out-vtt example_alignment\ablation\gloss_labels_pred__Gloss.vtt `
  --out-eaf example_alignment\ablation\04_gloss_pred__Gloss.eaf `
  --cache   example_alignment\subtitle_embedding\sign_clip_multi_gloss_tokens\04__Gloss.npz

python example_alignment\evaluate_gloss_labeling.py `
  --pred-csv example_alignment\ablation\gloss_labels_pred__Gloss.csv `
  --out-csv  example_alignment\ablation\evaluation_gloss_labeling__Gloss.csv

# Variant B: --tier Gloss_Input
python example_alignment\align_gloss_labels.py `
  --tier Gloss_Input `
  --out-csv example_alignment\ablation\gloss_labels_pred__Gloss_Input.csv `
  --out-vtt example_alignment\ablation\gloss_labels_pred__Gloss_Input.vtt `
  --out-eaf example_alignment\ablation\04_gloss_pred__Gloss_Input.eaf `
  --cache   example_alignment\subtitle_embedding\sign_clip_multi_gloss_tokens\04__Gloss_Input.npz

python example_alignment\evaluate_gloss_labeling.py `
  --pred-csv example_alignment\ablation\gloss_labels_pred__Gloss_Input.csv `
  --out-csv  example_alignment\ablation\evaluation_gloss_labeling__Gloss_Input.csv
```

### Why does Gloss win?

1. **Token boundary alignment** — `Gloss` มี 852 tokens ตรงกับ GT 852
   entries เป๊ะ และ position-by-position match 71.2% (vs 4.9% ของ
   `Gloss_Input` ที่มี 889 tokens) — เพราะ annotator ใช้ `Gloss` เป็น
   base ในการสร้าง GT
2. **Token count = degrees of freedom** — DP ของ `Gloss` ทำงานในขนาดที่
   "ออกแบบมาตรง" กับ GT, ของ `Gloss_Input` มี 37 tokens ส่วนเกินที่บีบ
   เข้าไปใน 852 entries ทำให้ overlap น้อยลง
3. **Sentence window coverage** — `Gloss` ครอบ GT ได้ดีกว่า: 97.77%
   ของ GT มี Gloss prediction overlap, vs 88.97% ของ Gloss_Input

### Caveat: text-match metric leakage

| Metric | `Gloss` | `Gloss_Input` |
| --- | ---: | ---: |
| Exact text match | 65.10% | 10.60% |

`Gloss` token list ถูกใช้ในการ build GT → exact-text-match ของ Gloss
**ไม่ใช่ model accuracy บริสุทธิ์** — รายงาน metric หลักเป็น **IoU**
ไม่ใช่ text match. ดู [Progress_09052026.md §6](Progress_09052026.md)

---

## Build Comparison & Visualization Outputs

### Test_comparison.eaf (15 experiment tiers + originals)

```powershell
python example_alignment\add_vtt_tiers_to_eaf.py --overwrite
```

Output: `example_alignment/Test_comparison.eaf` — เปิดใน ELAN ดู

- Original tiers (จาก `Test.eaf`): CC, CC_Input, CC_Aligned, Gloss, Gloss_Input, Gloss Labeling
- 7 pre-overlap experiment tiers: SUBTITLE_B2, SUBTITLE_B_MULTI, ...
- 7 post-overlap variants: ..._no_overlap
- Task 2 prediction: GLOSS_LABEL_PRED

### Test_best.eaf (best-only EAF)

```powershell
python example_alignment\add_best_to_eaf.py --overwrite
```

Output: `example_alignment/Test_best.eaf` — minimal EAF ที่มีแค่

- Original tiers
- SUBTITLE_C_MULTI + SUBTITLE_C_MULTI_no_overlap (Task 1 best)
- GLOSS_LABEL_PRED (Task 2)

### Timeline visualization

```powershell
python example_alignment\plot_alignment.py
```

Output: `example_alignment/figures/timeline_first_2min.png` — 4-lane
timeline (CC / CC_Aligned / C_MULTI / GLOSS_LABEL_PRED) สำหรับ 0-120 วินาทีแรก

---

## Adapting for Multiple Videos

Pipeline ปัจจุบัน design มาสำหรับ video เดียว (`04`) — สำหรับหลาย videos:

### 1. video_ids.txt — list ทุก IDs

```text
04
05
06
```

ทุก step ของ pipeline หลัก (`extract_cc_from_eaf` → `videos_to_poses`
→ `segmentation` → `extract_episode_features` → `align`) อ่าน file นี้
และ loop ผ่านทุก ID

### 2. File naming convention

ทุก script ต้องการ `<video_id>.<ext>` ใน folder ที่กำหนด:

| File | Location |
| --- | --- |
| `<id>.mp4` | `example_alignment/` |
| `<id>.pose` | `example_alignment/` |
| `<id>.vtt` | `example_alignment/subtitles/` |
| `<id>.vtt` | `example_alignment/subtitles_gloss_cc_time/` |
| `<id>.npy` | `example_alignment/segmentation_embedding/*/` |
| `<id>.npy` | `example_alignment/subtitle_embedding/*/` |

### 3. Scripts ที่ต้องแก้ — hardcoded paths

ตอนนี้ scripts เหล่านี้ใช้ hardcoded absolute paths สำหรับ video `04`
ก่อน production ใช้ multi-video ต้อง refactor:

| Script | Hardcoded reference | What to change |
| --- | --- | --- |
| `evaluate_all.py` | `EAF_PATH`, `CC_VTT`, `EXPERIMENTS` | Accept `--video_id` arg, build EXPERIMENTS dynamically |
| `evaluate_all_to_csv.py` | imports from `evaluate_all` | follows from above |
| `add_vtt_tiers_to_eaf.py` | `SOURCE_EAF`, `TARGET_EAF`, `BASE` | Accept `--video_id` arg |
| `add_best_to_eaf.py` | `SOURCE_EAF`, `TARGET_EAF`, `BASE`, `VTT_TIERS` | Accept `--video_id` arg |
| `plot_alignment.py` | reads `Test.eaf` directly | Accept `--video_id` arg |
| `align_gloss_labels.py` | `DEFAULT_EAF`, `DEFAULT_SEG_EAF`, `DEFAULT_SIGN_EMB` | already accepts `--eaf`, `--seg-eaf`, `--sign-emb` flags ✓ |
| `evaluate_gloss_labeling.py` | `DEFAULT_EAF` | Accept `--video_id` arg |
| `make_gloss_cc_vtt.py` | reads `Test.eaf` directly | Loop over `video_ids.txt` |

### 4. Multi-video evaluation (BOBSL-scale)

สำหรับการรายงาน metric เหมือน SEA paper ต้องใช้
[SEA/misc/evaluate_sub_alignment.py](SEA/misc/evaluate_sub_alignment.py) —
รับ folder ของ aligned VTT + folder ของ GT VTT + video_ids.txt — ดู
`SEA/scripts/evaluate.sh` (BOBSL pattern)

---

## Quick Start (everything already set up)

หลัง setup เสร็จและไฟล์ทั้งหมดใน `example_alignment/` พร้อม:

```powershell
venv\Scripts\activate

# (Optional) Re-extract subtitle inputs
python example_alignment\extract_cc_from_eaf.py example_alignment\Test.eaf example_alignment\subtitles\04.vtt --tier CC_Input
python example_alignment\make_gloss_cc_vtt.py

# Task 1: best alignment (C_MULTI)
cd SEA
python align.py --overwrite --mode=inference --video_ids ..\example_alignment\video_ids.txt --num_workers 1 --sign-b-threshold 30 --sign-o-threshold 50 --dp_duration_penalty_weight 2 --dp_gap_penalty_weight 8 --dp_max_gap 6 --dp_window_size 40 --pr_subs_delta_bias_start 1.3 --pr_subs_delta_bias_end 1.0 --similarity_measure sign_clip_embedding --similarity_weight 6 --pr_sub_path ..\example_alignment\subtitles_gloss_cc_time --segmentation_dir ..\example_alignment\segmentation_output --subtitle_embedding_dir ..\example_alignment\subtitle_embedding\sign_clip_multi_gloss --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip_multi --save_dir ..\example_alignment\aligned_output_multi_gloss
cd ..

# Task 1: overlap fix + evaluate all 7 experiments
python example_alignment\evaluate_all_to_csv.py

# Task 2: recommended (Gloss tier)
python example_alignment\align_gloss_labels.py --tier Gloss `
  --out-csv example_alignment\ablation\gloss_labels_pred__Gloss.csv `
  --out-vtt example_alignment\ablation\gloss_labels_pred__Gloss.vtt `
  --out-eaf example_alignment\ablation\04_gloss_pred__Gloss.eaf `
  --cache   example_alignment\subtitle_embedding\sign_clip_multi_gloss_tokens\04__Gloss.npz
python example_alignment\evaluate_gloss_labeling.py `
  --pred-csv example_alignment\ablation\gloss_labels_pred__Gloss.csv `
  --out-csv  example_alignment\ablation\evaluation_gloss_labeling__Gloss.csv

# Build comparison + best EAFs
python example_alignment\add_vtt_tiers_to_eaf.py --overwrite
python example_alignment\add_best_to_eaf.py --overwrite

# Plot timeline
python example_alignment\plot_alignment.py
```

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `UnicodeDecodeError` on `video_ids.txt` | PowerShell `echo` writes UTF-16 BOM | `python -c "open('example_alignment/video_ids.txt','w',encoding='utf-8').write('04\n')"` |
| `torch.cuda.is_available()` = False | Wrong PyTorch CUDA build | `pip install ... --index-url https://download.pytorch.org/whl/cu128` |
| `mediapipe` import error | Wrong version | `pip install mediapipe==0.10.21` |
| `numba` JIT error | LLVM not found | `pip install numba` หรือ fallback เป็น pure Python (ช้าลง) |
| `FileNotFoundError: checkpoint_best.pt` | Wrong weights path | ดู [Step 6](#step-6--download-signclip-model-checkpoints) |
| Alignment produces all cues at same time | `--sign-b-threshold` mismatch กับ segmentation | ใช้ค่าเดียวกัน (`30 50`) ใน segmentation + alignment |
| `evaluate_all.py` matched = 0 | EAF path ผิด หรือ tier name ผิด | ตรวจ `EAF_PATH` ใน script + tier name `CC_Aligned` |
| `subprocess` error ใน segmentation บน Windows | `shlex.quote` + `shell=True` | แก้แล้วใน repo นี้ (uses `shlex.split` + `shell=False`) |
| `align_gloss_labels.py` ทำงานช้าครั้งแรก | Token cache เปล่า | ครั้งที่ 2+ จะเร็วกว่ามาก (โหลด `.npz`) |
| `align_gloss_labels.py` cache ปะปนกัน | ใช้ default cache สำหรับ tier ต่างกัน | ใช้ `--cache` แยก path ตามการ ablate |

---

## Evaluation Methodology Summary

| | Original SEA `SEA/misc/evaluate_sub_alignment.py` | This project `example_alignment/evaluate_all.py` |
| --- | --- | --- |
| **Scale** | BOBSL — 20,000+ sentences | Single video — 119 cues |
| **Matching** | Text lookup or frame-level | **Index-based** (`pred[i] ↔ gt[i]`) |
| **Metrics** | frame acc, F1@0.1/0.25/0.5, mean abs start+end | signed/abs mean offset, ±1s/±2s/±3s, overlap%, plus same SEA metrics |
| **Dependencies** | pysrt, webvtt, beartype, BOBSL paths | stdlib only |

> **เหตุผลที่ต้อง index-based matching:** `CC_Input` กับ `CC_Aligned`
> มี 119 entries เท่ากันและ map ตาม index ตรงๆ — แต่ annotators มักแก้
> wording ตอน build `CC_Aligned` ทำให้ text lookup match ได้แค่ ~50/119
> entries
>
> **ทำไมไม่ใช้ original SEA evaluation:** Original ต้องการ frame-rate
> และ label sequences ที่ reconstruct จาก subtitle timestamps ของ
> หลาย videos — ใช้กับ single video ได้ไม่ดี — แต่ตอนนี้
> [evaluate_all.py](example_alignment/evaluate_all.py) คำนวณ
> SEA metrics ด้วย (FrameAcc, F1@IoU) ที่ FPS=25

---

## License

- SEA original code: license จาก [J22Melody/SEA](https://github.com/J22Melody/SEA)
- SignCLIP: license จาก [J22Melody/fairseq](https://github.com/J22Melody/fairseq)
- Custom scripts ใน `example_alignment/`: เป็นส่วนหนึ่งของโปรเจกต์นี้

---

## Acknowledgements

- **SEA** — Jiang et al. (2025) "Segment, Embed, and Align" — [arXiv:2512.08094](https://arxiv.org/abs/2512.08094)
- **SignCLIP** — multilingual sign-language embedding model
- **NECTEC** — provider of TSL annotation video and ELAN annotations
- **MediaPipe** — pose estimation
