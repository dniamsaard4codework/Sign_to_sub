# Sign_to_sub — Thai Sign Language (TSL) Subtitle Alignment

โปรเจกต์สำหรับการจัดเรียง (align) คำบรรยาย (subtitle) ให้ตรงกับภาษามือไทย (TSL) ในวิดีโอ
โดยใช้ระบบ **SEA (Segment, Embed, and Align)** เป็นฐาน

## Upstream References

โปรเจกต์นี้สร้างขึ้นบนงานวิจัยต้นฉบับ:

| Component | Original Repository | Paper |
| --- | --- | --- |
| **SEA** | [J22Melody/SEA](https://github.com/J22Melody/SEA) | [arXiv:2512.08094](https://arxiv.org/abs/2512.08094) — Jiang et al. 2025 |
| **SignCLIP (fairseq)** | [J22Melody/fairseq](https://github.com/J22Melody/fairseq) (fork of [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)) | SignCLIP models for sign language embeddings |

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
├── SEA/                        ← Modified SEA system (upstream: J22Melody/SEA)
│   ├── align.py                ← Main alignment pipeline
│   ├── align_dp.py             ← Dynamic Programming alignment
│   ├── align_similarity.py     ← Similarity computation
│   ├── config.py               ← CLI argument configuration
│   ├── segmentation.py         ← Sign segmentation
│   └── ...
├── example_alignment/          ← TSL experiment scripts & data
│   ├── evaluate_all.py         ← Evaluation across all experiments
│   ├── add_vtt_tiers_to_eaf.py ← Add VTT results to ELAN
│   ├── extract_cc_from_eaf.py  ← Extract CC from EAF
│   ├── fix_overlap_vtt.py      ← Post-process overlap removal
│   ├── make_gloss_cc_vtt.py    ← Create Gloss-CC hybrid VTT
│   ├── merge_cc_to_updated_eaf.py
│   ├── README_TH.md            ← Detailed Thai documentation
│   └── ...
├── assets/                     ← Images
├── Progress_20042026.md        ← Progress report (Thai)
├── .gitignore
└── README.md                   ← This file
```

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/dniamsaard4codework/Sign_to_sub.git
cd Sign_to_sub
```

### 2. Install fairseq_signclip (separately — not included due to size)

```bash
git clone https://github.com/J22Melody/fairseq.git fairseq_signclip
cd fairseq_signclip
pip install -e .
cd ..
```

> **Note:** ดู Changes from Original fairseq_signclip ด้านบน สำหรับ patch ที่ต้องทำ
> หรือดูรายละเอียดใน `example_alignment/README_TH.md`

### 3. Python environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install mediapipe==0.10.21 pose-format webvtt-py tqdm numpy
```

### 4. Download SignCLIP model checkpoints

ดาวน์โหลด checkpoint ไว้ใน `fairseq_signclip/examples/MMPT/runs/retri/signclip_*/`
ดูรายละเอียดใน `example_alignment/README_TH.md`

## Quick Start

ดูรายละเอียดการใช้งานทั้งหมดใน [`example_alignment/README_TH.md`](example_alignment/README_TH.md)

## License

SEA original code is under the license from [J22Melody/SEA](https://github.com/J22Melody/SEA).
Custom scripts in `example_alignment/` are part of this project.
