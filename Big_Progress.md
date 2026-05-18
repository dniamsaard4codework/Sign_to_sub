# Big Progress — SEA Project Consolidated Report

> **SEA** (Segment, Embed, and Align) — ระบบจัดตำแหน่งคำบรรยาย / annotation ภาษามือไทย (TSL)
> บน pipeline ที่พัฒนาต่อจากงานวิจัย [arXiv:2512.08094](https://arxiv.org/abs/2512.08094) — Jiang et al., 2025
>
> เอกสารนี้รวมทุก progress report (20 เม.ย. → 16 พ.ค. 2569) ไว้ในที่เดียว
> โฟกัสที่ **pipeline, เหตุผลของแต่ละ experiment, และผลลัพธ์** ไม่แบ่งตามวัน

---

## สารบัญ

1. [ภาพรวมโครงการและเป้าหมาย](#1-ภาพรวมโครงการและเป้าหมาย)
2. [ข้อมูลนำเข้า](#2-ข้อมูลนำเข้า)
3. [Environment Setup](#3-environment-setup)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Task 1 — Subtitle Alignment: Experiments และผลลัพธ์](#5-task-1--subtitle-alignment-experiments-และผลลัพธ์)
6. [Task 2 — Gloss Labeling: Experiments และผลลัพธ์](#6-task-2--gloss-labeling-experiments-และผลลัพธ์)
7. [เครื่องมือเสริมที่พัฒนาขึ้น](#7-เครื่องมือเสริมที่พัฒนาขึ้น)
8. [ปัญหาที่พบและวิธีแก้ไข](#8-ปัญหาที่พบและวิธีแก้ไข)
9. [ผลสรุปและ Recommendation](#9-ผลสรุปและ-recommendation)
10. [ForcedAlignment Dataset — Task 2 Scale-up](#10-forcedalignment-dataset--task-2-scale-up-แผน-17-พค-2569)
11. [อ้างอิง](#11-อ้างอิง)

---

## 1. ภาพรวมโครงการและเป้าหมาย

นำระบบ **SEA** มาประยุกต์กับ **ภาษามือไทย (TSL)** เพื่อแก้ 2 งานหลัก:

### Task 1 — CC Subtitle Alignment

> **เป้าหมาย:** เลื่อน timestamp ของคำบรรยาย (CC_Input) ซึ่ง sync กับเสียงพูด ให้ไปตรงกับจังหวะที่ผู้แปลภาษามือแสดงท่าทาง (CC_Aligned)

**ทำไมถึงเป็นปัญหา:** ผู้แปลภาษามือต้องรับฟังเสียงก่อนแล้วจึงแปล ทำให้ท่ามือช้ากว่าเสียงพูดเสมอ
**วิธีที่ SEA แก้:** Pose estimation → Segmentation ตรวจจับช่วงเวลาท่ามือ → DP alignment จับคู่ subtitle กับ sign segment

**ผลดีที่สุดปัจจุบัน (C_MULTI, index-based evaluation, 119/119 cues):**

| Metric | ค่า |
| --- | ---: |
| Mean start offset | −0.16 s |
| % within ±1 s | 73.9% |
| % within ±2 s | 95.0% |
| % within ±3 s | **100%** |
| Frame accuracy | 82.6% |
| F1 @ 0.50 IoU | 88.2% |
| Overlap rate (after fix) | **0%** |

### Task 2 — Gloss Labeling

> **เป้าหมาย:** แยก gloss ระดับประโยค ("สวัสดี ผายมือ เด็ก เรียน") ลงไปเป็น annotation รายท่ามือแต่ละท่า (852 entries)

**ทำไมถึงเป็นปัญหา:** การทำ Gloss Labeling ด้วยมือใช้เวลามาก ต้องการระบบอัตโนมัติ
**วิธีที่ SEA แก้:** Token-level SignCLIP embedding + per-sentence monotonic DP บน Gloss sentence window

**ผลดีที่สุดปัจจุบัน (Gloss whole-video, 852 predictions):**

| Metric | ค่า |
| --- | ---: |
| Mean IoU | 0.4901 |
| % IoU ≥ 0.5 | 48.4% |
| % IoU ≥ 0.3 | 77.0% |
| % any temporal overlap | 97.5% |
| Fallback uniform | 0 / 119 sentences |

---

## 2. ข้อมูลนำเข้า

วิดีโอตัวอย่าง: **"การเปรียบเทียบและเรียงลำดับ"** (11.07 นาที, 1920×1080, 60fps)

| ไฟล์ | คำอธิบาย | ขนาด/จำนวน |
| --- | --- | --- |
| `04.mp4` | วิดีโอต้นฉบับ | ~80 MB |
| `04.pose` | Skeleton pose จาก MediaPipe Holistic | 358 MB |
| `Test.eaf` | ELAN annotation (input หลัก) | — |

### Tiers ใน Test.eaf

| Tier | จำนวน | บทบาท |
| --- | ---: | --- |
| **CC** | 172 | คำบรรยายดิบ (timestamp จากเสียงพูด) — ไม่ใช้แล้ว |
| **CC_Input** | 119 | คำบรรยาย curated — **input Task 1** |
| **CC_Aligned** | 119 | align ด้วยมือโดยนักวิจัย — **ground truth Task 1** |
| **Gloss** | 119 | Gloss tier เดิม — **input Task 2 (ดีที่สุด)** |
| **Gloss_Input** | 119 | Gloss tier ที่ curated — **input Task 2 (ด้อยกว่า)** |
| **Gloss Labeling** | 852 | annotation รายท่ามือแต่ละท่า — **ground truth Task 2** |

> **วิวัฒนาการ input:** เริ่มต้นใช้ `CC` (172 cues) แล้วเปลี่ยนเป็น `CC_Input` (119 cues) เมื่อมี `Test.eaf` ที่ curated แล้ว
> **วิวัฒนาการ evaluation Task 1:** เริ่มต้นใช้ text-lookup (match ได้แค่ 69/172 = 58%) → เปลี่ยนเป็น index-based (119/119 = 100%)

---

## 3. Environment Setup

| ส่วนประกอบ | รายละเอียด |
| --- | --- |
| OS | Windows 11 Pro |
| CPU | Intel Core Ultra 7 265K (20 cores) |
| RAM | 64 GB |
| GPU | NVIDIA GeForce RTX 5060 Ti — Blackwell (sm_120), 17.1 GB VRAM |
| CUDA Driver | 595.79 (CUDA 13.2) |
| Python | 3.11.15 |
| PyTorch | 2.11.0+**cu128** (ต้อง cu128 สำหรับ Blackwell) |

### Dependencies หลัก

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA
C:\Users\dniam\.local\bin\python3.11.exe -m venv venv
venv\Scripts\activate

pip install pysrt webvtt-py lxml numpy pympi-ling
pip install "mediapipe==0.10.21" pose-format   # ต้องล็อก 0.10.21 เท่านั้น
pip install beartype numba tqdm scikit-learn tabulate matplotlib imageio_ffmpeg
pip install "git+https://github.com/J22Melody/segmentation@bsl"
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

### SignCLIP (ติดตั้งครั้งเดียว)

```powershell
git clone https://github.com/J22Melody/fairseq.git fairseq_signclip
cd fairseq_signclip && pip install -e .
cd examples\MMPT && pip install -e .
# ดาวน์โหลด model weights
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/10q7FxPlicrfwZn7_FgtNqKFDiAJi6CTc?usp=sharing" -O .\runs
```

| Model variant | Checkpoint | Train language |
| --- | --- | --- |
| `bsl` | `bobsl_finetune_checkpoint_best.pt` | British Sign Language |
| `multilingual` | `baseline_temporal_checkpoint_best.pt` | หลายภาษา |
| `asl` | `asl_finetune_checkpoint_best.pt` | American Sign Language |

---

## 4. Pipeline Architecture

```text
Test.eaf (CC_Input / Gloss / Gloss_Input)
    │
    ├─► Step 1: Extract CC_Input → VTT          extract_cc_from_eaf.py --tier CC_Input
    │   subtitles/04.vtt (119 cues)
    │
    ├─► Step 1b: Make Gloss VTT                 make_gloss_cc_vtt.py
    │   subtitles_gloss_cc_time/04.vtt (119 cues, Gloss text + CC timestamp)
    │
    ├─► Step 2: Pose Estimation                 videos_to_poses --format mediapipe
    │   04.pose (358 MB, 543 landmarks × 60fps)
    │
    ├─► Step 3: Segmentation                    SEA/segmentation.py --b 30 --o 50
    │   segmentation_output/E4s-1_30_50/04.eaf
    │   → SIGN: 2,780 segments / SENTENCE: 418 segments
    │
    ├─┬─► Step 4a: Sign Embeddings              extract_episode_features.py --feature_type sign
    │ │   segmentation_embedding/{bsl|multi|asl}/04.npy  (2780 × 768)
    │ │
    │ └─► Step 4b: Subtitle Embeddings          extract_episode_features.py --feature_type subtitle
    │     subtitle_embedding/{model}_{text}/04.npy  (119 × 768 each)
    │
    ├─► Step 5: DP Alignment                    SEA/align.py  [7 experiments]
    │   aligned_output_*/04.vtt
    │
    ├─► Step 6: Post-processing                 fix_overlap_vtt.py (clamp end time)
    │   aligned_output_*/04_no_overlap.vtt  → overlap 88% → 0%
    │
    ├─► Step 7: Task 2 — Gloss Labeling         align_gloss_labels.py
    │   gloss_labels_pred.csv (852 rows) + 04_gloss_pred.eaf
    │
    └─► Step 8: Evaluation                      evaluate_all_to_csv.py / evaluate_gloss_labeling.py
        evaluation_task1_results.csv / evaluation_gloss_labeling.csv
```

### DP Cost Function (align.py)

```text
cost(cue_i, group[k:j]) =
    |cue_start − group_start|
  + |cue_end   − group_end|
  + dp_duration_penalty_weight × |cue_dur − group_dur|
  + dp_gap_penalty_weight       × total_gap_in_group
  + similarity_weight           × (−similarity_total)
```

### Tunable Parameters (ค่าที่ใช้ใน C_MULTI — experiment ดีที่สุด)

| Flag | C_MULTI value | ผลกระทบ |
| --- | --- | --- |
| `--pr_subs_delta_bias_start` | 1.3 s | Pre-shift subtitle ก่อน align (ชดเชย delay ท่ามือ) |
| `--pr_subs_delta_bias_end` | 1.0 s | — |
| `--dp_window_size` | 40 | Search window ของ DP |
| `--dp_max_gap` | 6 | จำกัด sign segments ต่อ group |
| `--dp_duration_penalty_weight` | 2 | บังคับ duration match |
| `--dp_gap_penalty_weight` | 8 | ลงโทษ group ที่มีช่องว่าง |
| `--similarity_weight` | 6 | บาลานซ์ semantic vs timing |
| `--sign_clip_model` | `multilingual` | SignCLIP variant |
| `--pr_sub_path` | `subtitles_gloss_cc_time` | ใช้ Gloss text เป็น embedding input |

---

## 5. Task 1 — Subtitle Alignment: Experiments และผลลัพธ์

### 5.1 วิวัฒนาการของ Experiments

#### Experiment A — Baseline (no embedding)

**เหตุผล:** รัน DP alignment อิงจาก timing เพียงอย่างเดียวเพื่อเป็น baseline ก่อนเพิ่ม embedding
**ผล:** Mean start shift ~2.49 s — สูงมาก แสดงว่า timing alone ไม่พอ

---

#### Experiment B1 → B2 — BSL model + CC text (tuning)

**เหตุผล B1:** เพิ่ม semantic embedding ด้วย BSL SignCLIP เพื่อดูว่า embedding ช่วยได้ไหม
**เหตุผล B2:** B1 ยังมี mean shift สูง (~2.49 s) → tune parameters (ลด bias, เพิ่ม duration/gap penalty)
**ผล B2:** Mean +1.02 s, 74% ±1 s — ดีขึ้น แต่ยังห่าง ground truth

---

#### Experiment B_MULTI — Multilingual model + CC text

**เหตุผล:** BSL model train บน BSL-specific data เท่านั้น → Multilingual น่าจะ generalize ไป TSL ได้ดีกว่า
**ผล:** Mean +0.91 s, 78% ±1 s — ดีกว่า BSL ชัดเจน ยืนยันว่า Multilingual เหมาะกว่าสำหรับ TSL

---

#### Experiment C_MULTI — Multilingual + Gloss text ⭐ ดีที่สุด

**เหตุผล:** CC text คือคำบรรยายเสียงพูด (ภาษาไทยพูด) ซึ่งความหมาย semantic ต่างจากท่ามือ
ขณะที่ Gloss text คือรหัสท่ามือโดยตรง ("สวัสดี ผายมือ เด็ก") → embedding ควรใกล้กับ sign embedding มากกว่า
**ผล (old evaluation, text-lookup, 69/172):** Mean +0.49 s, 80% ±1 s, 99% ±3 s
**ผล (new evaluation, index-based, 119/119):** Mean **−0.16 s**, 73.9% ±1 s, 95.0% ±2 s, **100% ±3 s**

---

#### Experiment C_MULTI_word — Word-level similarity

**เหตุผล:** ทดสอบว่าการ embed ทีละคำแล้ว pool (แทน embed ทั้ง cue เป็น 1 vector) จะช่วยได้ไหม
เพราะ Gloss text มีหลายคำ และแต่ละคำ map กับท่ามือแต่ละท่า
**ผล:** Mean +0.51 s, 77% ±1 s — **ไม่ช่วย** เทียบ C_MULTI pooled ผลใกล้เคียงกันมาก
**บทเรียน:** Sentence-level pooling ดีพอแล้ว word-level เพิ่ม complexity โดยไม่ได้ผล

---

#### Experiments D_ASL, D_ASL_gloss, D_ASL_word — ASL model

**เหตุผล:** ทดสอบ ASL model ซึ่งเป็น language-specific อีกตัว เพื่อเปรียบเทียบกับ Multilingual
**ผล:**

| Experiment | Mean offset | ±1 s |
| --- | ---: | ---: |
| D_ASL (ASL + CC text) | +1.25 s | 59% |
| D_ASL_gloss (ASL + Gloss text) | +0.77 s | 64% |
| D_ASL_word (ASL + Gloss + word) | +0.78 s | 67% |

**บทเรียน:** ASL model ด้อยกว่า Multilingual ทุกกรณี เพราะ ASL train บน ASL-specific data → generalize มา TSL ไม่ดี

---

### 5.2 ตารางสรุปทุก Experiment (index-based evaluation, 119/119 cues)

หมายเหตุ: ตาราง index-based จาก Progress_04052026 — metric ทุกตัวครอบคลุม 119/119 cues

| Experiment | Model | Text | Mean | ±1 s | ±2 s | ±3 s | Overlap (before/after) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| **C_MULTI** ⭐ | Multilingual | Gloss | **−0.16 s** | **73.9%** | 95.0% | **100%** | 88% / **0%** |
| B_MULTI | Multilingual | CC | +0.91 s | 78% | 97% | 99% | 88% / **0%** |
| B2 | BSL | CC | +1.02 s | 74% | 96% | 97% | 88% / **0%** |
| C_MULTI_word | Multilingual | Gloss (word) | +0.51 s | 77% | 96% | 99% | 88% / **0%** |
| D_ASL_word | ASL | Gloss (word) | +0.78 s | 67% | 93% | 96% | 88% / **0%** |
| D_ASL_gloss | ASL | Gloss | +0.77 s | 64% | 91% | 97% | 88% / **0%** |
| D_ASL | ASL | CC | +1.25 s | 59% | 81% | 96% | 89% / **0%** |

> ⚠️ **Overlap fix:** `fix_overlap_vtt.py` clamp end time เท่านั้น ไม่แตะ start time → metric ±Ns และ mean offset ไม่เปลี่ยนหลัง fix แต่ overlap ลดเหลือ 0% ทุก experiment — "safe post-processing"

### 5.3 วิวัฒนาการของ Evaluation Method

| รุ่น | Method | Coverage | ผล C_MULTI |
| --- | --- | --- | --- |
| รุ่นแรก (CC 172 cues) | Text-lookup (text ต้องตรงตัวอักษร) | **69/172 matched (58%)** | Mean +0.49 s, 80% ±1 s |
| รุ่นใหม่ (CC_Input 119 cues) | Index-based (pred[i] ↔ gt[i]) | **119/119 matched (100%)** | Mean −0.16 s, 73.9% ±1 s |

**ทำไมเปลี่ยน:** Text-lookup พลาด 49 entries ใน CC_Aligned ที่ผู้ annotate แก้ข้อความจาก CC ดิบ (รวมประโยค, แก้คำ) → ใช้ CC_Input + index-based แทน

---

## 6. Task 2 — Gloss Labeling: Experiments และผลลัพธ์

### 6.1 DP สำหรับ Task 2 (per-sentence monotonic DP)

```text
Gloss sentence (start_s, end_s, "ผายมือ เด็ก เรียน")
        │
        ├─► tokenize → ["ผายมือ", "เด็ก", "เรียน"]  (T tokens)
        ├─► restrict SIGN candidates: mid ∈ [start_s, end_s] ±0.5s pad
        ├─► embed each token via SignCLIP multilingual text encoder → (T × 768)
        ├─► sim_matrix (T × K) = cosine + row softmax
        └─► monotonic DP per sentence:
              dp[t][j] = min over k of {
                  dp[t-1][k-1]
                + (−Σ sim[t-1, k-1..j-1])
                + gap_penalty × inter-segment gap total
                + coverage_penalty × |group_dur − sentence_dur/T|
              }
              → backtrack → (seg[k_start].start, seg[k_end].end, token)
```

**Default parameters:** `gap_penalty=2.0`, `coverage_penalty=0.5`, `window_pad=0.5s`, `model=multilingual`, `language_tag=<en> <bfi>`

---

### 6.2 Experiment 1 — Prototype: Whole-video `Gloss` tier

**เหตุผล:** ทดสอบว่า per-sentence DP + SignCLIP multilingual text encoder สามารถ align Gloss tokens กับ SIGN segments ได้ไหม
โดย embed token ทีละคำ และ constraint search space ด้วย Gloss sentence boundary

**ความกังวลก่อนทดลอง:** SignCLIP multilingual อาจไม่เข้าใจ Thai gloss tokens ระดับคำ → IoU ต่ำ

**ผลจริง:**

| Metric | ค่า |
| --- | --- |
| Predictions | 852 (= GT count) |
| **Mean IoU** | **0.4901** |
| Median IoU | 0.4861 |
| % IoU ≥ 0.5 | **48.4%** |
| % IoU ≥ 0.3 | **77.0%** |
| % any temporal overlap | 97.5% |
| Mean start offset | −0.034 s |
| Exact text match (overlapping pairs) | 65.1% |
| Fallback uniform | **0 / 119** |

> 🎯 **Sanity bar คือ IoU ≥ 0.30 (chance ≈ 1/7 = 0.14)** — ผลที่ได้ 0.49 เกินเกณฑ์มาก
> แสดงว่า Multilingual encoder เข้าใจ Thai Gloss tokens ได้ดีกว่าที่คาดไว้
> สาเหตุ: tokens ส่วนใหญ่เป็น parenthesized notation ซ้ำๆ (`(ผายมือ)`) + DP ใช้ temporal constraint → search space เล็ก

**Token caching:** มี unique tokens เพียง 192 ตัวจาก 852 occurrences
→ run ครั้งแรก ~3 นาที (embed 192 tokens), ครั้งถัดไป ~5 วินาที (cache hit)

---

### 6.3 Experiment 2 — Ablation: `Gloss` vs `Gloss_Input`

**เหตุผล:** `Test.eaf` มี `Gloss_Input` tier ที่ curated ไว้แล้ว (เช่นเดียวกับ CC_Input สำหรับ Task 1)
ทดสอบว่า curated tier ให้ผลดีกว่า original `Gloss` tier ในงาน Task 2 ไหม

| Metric | `Gloss` (original) | `Gloss_Input` (curated) | Δ |
| --- | ---: | ---: | ---: |
| **Predictions** | **852** | 889 | +37 |
| **Mean IoU** | **0.4901** | 0.4199 | **+7.0 pp** |
| **% IoU ≥ 0.5** | **48.4%** | 38.9% | **+9.4 pp** |
| **% IoU ≥ 0.3** | **77.0%** | 66.0% | **+11.0 pp** |
| % any temporal overlap | **97.5%** | 93.4% | +4.1 pp |
| Mean abs start offset | **0.188 s** | 0.212 s | −24 ms |
| Exact text match | **65.1%** | 10.6%* | +54.5 pp |

**\*** text match ของ `Gloss` มี structural advantage (leakage) — ใช้ IoU metric เป็นหลักแทน

**ทำไม `Gloss` ชนะ:**

1. **Token count ตรงกับ GT:** `Gloss` มี 852 tokens = 852 GT entries → DP มี degree of freedom ถูกต้อง
   ขณะที่ `Gloss_Input` มี 889 tokens (+37) จาก re-tokenization → token count ไม่ตรง GT
2. **Sentence window กว้างกว่า:** Gloss total duration 560.55 s vs Gloss_Input 541.25 s (+19 s)
   → `Gloss_Input` ทิ้ง GT 11.03% ที่ตกขอบ window แม้หลัง 0.5 s padding
3. **Structural alignment:** นักวิจัยใช้ `Gloss` เป็น base ในการสร้าง `Gloss Labeling` GT → token boundary ใกล้ GT โดยธรรมชาติ

**Recommendation:** ใช้ `--tier Gloss` เป็น default สำหรับ Task 2

---

### 6.4 Experiment 3 — Per-sentence Video Cropping Pipeline

**เหตุผล/สมมติฐาน:**
จาก ablation รอบก่อน พบว่า sentence window filter ไม่ครอบ GT ทุก entry (`Gloss_Input` ตก GT 11%)
**สมมติฐาน:** ถ้า crop วิดีโอเป็น 119 clip ตาม Gloss sentence boundaries แล้วรัน pose + segmentation ใหม่ต่อ clip —
SIGN segments จะเป็น "sentence-local" → candidate pool สะอาดกว่า → alignment ดีขึ้น

**Workflow:**

```text
04.mp4 ──crop (libx264)──► 119 × clip_NNN.mp4
                                │
                    videos_to_poses → 119 × clip_NNN.pose
                                │
              SEA/segmentation.py (batch, E4s-1_30_50)
                                │
    extract_episode_features.py (multilingual, batch)
                                │
              per-sentence DP (reuse align_one_sentence())
                                │
          shift times +clip_start → recover global video time
                                │
              combined CSV / VTT / EAF → evaluate vs GT
```

**ผล:**

| Metric | Whole-video `Gloss` (baseline) | Per-sentence | Δ |
| --- | ---: | ---: | ---: |
| #predictions | 852 | 852 | 0 |
| **Mean IoU** | **0.4901** | 0.4763 | **−0.0138** |
| % IoU ≥ 0.5 | **48.4%** | 46.0% | −2.35 pp |
| % IoU ≥ 0.3 | 77.0% | 76.9% | ≈ tie |
| % any temporal overlap | **97.5%** | 96.1% | −1.41 pp |
| % zero overlap | **2.5%** | 3.9% | worse |
| Exact text match | **65.1%** | 60.0% | −5.1 pp |
| **Runtime** | **~10 min** | **~69 min** | **~7×** ช้ากว่า |
| Fallback | 0 / 119 | 0 / 119 | tie |

**สมมติฐานไม่เป็นจริง** — ผล worse กว่า baseline ทุก metric ที่สำคัญ

**Per-sentence breakdown ตาม token count (เข้าใจ root cause):**

| Token bucket | # sentences | Per-sent IoU | Gloss base IoU | Δ |
| --- | ---: | ---: | ---: | ---: |
| 1–3 tokens | 13 | 0.4675 | **0.6268** | **−0.1593** (เสียหนัก) |
| 4–6 tokens | 42 | 0.4930 | 0.5175 | −0.0245 |
| 7–9 tokens | 39 | 0.4643 | 0.4722 | −0.0079 |
| 10–12 tokens | 17 | 0.4608 | 0.4750 | −0.0142 |
| **13+ tokens** | 8 | **0.5052** | 0.4906 | **+0.0146** (per-sentence ชนะ) |

**Root cause ของ regression:**

1. **Short clip (< 3 s) ขาด temporal context:**
   - MediaPipe Holistic ต้องการ context หลายเฟรมเพื่อ smooth landmarks — clip สั้นทำให้ smoothing ไม่สมบูรณ์
   - SEA segmentation (GRU-based) hidden state เริ่มที่ zero ทุก clip → sign แรกของ clip detect ช้ากว่า whole-video pipeline ที่ GRU state warm จาก frame ก่อน
   - Frame-accurate libx264 re-encode shift timestamp เล็กน้อย (±1 frame at 25fps = ±40 ms)

2. **ประโยคยาว (13+ tokens) ได้กำไรเล็กน้อย (+1.46 pp):** window กว้างอยู่แล้ว, noise จาก cross-sentence context ลดลง — แต่ subset แคบเกินไปสำหรับ production

**Runtime breakdown:**

| Phase | เวลา | หมายเหตุ |
| --- | ---: | --- |
| Crop 119 clips (libx264) | 44.7 s | — |
| Pose extraction (MediaPipe × 119) | **3,696.6 s (61.6 min)** | **bottleneck** (~31 s/clip) |
| SEA segmentation (batch) | 232.9 s | model load 1× |
| SignCLIP embedding (batch) | 179.6 s | model load 1× |
| DP + combine | ~10 s | — |
| **รวม** | **~69 min** | vs ~10 min whole-video |

---

### 6.5 ตารางสรุป Task 2 ทุก Experiment

| Experiment | Input tier | Predictions | Mean IoU | % ≥ 0.5 | % ≥ 0.3 | % any overlap | Runtime |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **Whole-video Gloss** ⭐ | `Gloss` (852 tokens) | 852 | **0.4901** | **48.4%** | **77.0%** | **97.5%** | ~10 min |
| Per-sentence | `Gloss` (852 tokens) | 852 | 0.4763 | 46.0% | 76.9% | 96.1% | ~69 min |
| Whole-video Gloss_Input | `Gloss_Input` (889 tokens) | 889 | 0.4199 | 38.9% | 66.0% | 93.4% | ~10 min |

**Conclusion:** Whole-video Gloss baseline คือ best + cheapest option

---

## 7. เครื่องมือเสริมที่พัฒนาขึ้น

| Script | หน้าที่ | Output |
| --- | --- | --- |
| `extract_cc_from_eaf.py` | แยก tier ที่ระบุ (CC_Input, Gloss ฯ) จาก EAF → WebVTT | `subtitles/04.vtt` |
| `make_gloss_cc_vtt.py` | ผสม Gloss text กับ CC_Input timestamp → VTT สำหรับ embedding | `subtitles_gloss_cc_time/04.vtt` |
| `merge_cc_to_updated_eaf.py` | คัดลอก tiers จาก source EAF → target EAF | `04_updated.eaf` |
| `fix_overlap_vtt.py` | clamp end time ไม่ให้เกิน start ของ cue ถัดไป (overlap → 0%) | `04_no_overlap.vtt` |
| `evaluate_all_to_csv.py` | รัน fix_overlap + re-evaluate ทุก experiment ในครั้งเดียว | `evaluation_task1_results.csv` |
| `evaluate_all.py` | เปรียบเทียบ VTT กับ CC_Aligned GT (print stdout) | — |
| `add_vtt_tiers_to_eaf.py` | เพิ่มผลทุก experiment + ablation เป็น tiers ใน comparison EAF | `Test_comparison.eaf` |
| `add_best_to_eaf.py` | สร้าง EAF "best run only" (Task 1 best + Task 2 Gloss ablation) | `Test_best.eaf` |
| `align_gloss_labels.py` | Task 2: token-level DP alignment + cache token embeddings | `gloss_labels_pred.csv`, `.vtt`, `.eaf` |
| `evaluate_gloss_labeling.py` | Task 2 evaluator: IoU per prediction vs Gloss Labeling GT | `evaluation_gloss_labeling.csv` |
| `run_task2_per_sentence.py` | Orchestrator: crop → pose → seg → emb → DP per 119 clips | `ablation_per_sentence/` |
| `plot_alignment.py` | Matplotlib timeline 4-lane (CC / CC_Aligned / C_MULTI / GLOSS_LABEL_PRED) | `figures/timeline_first_2min.png` |

---

## 8. ปัญหาที่พบและวิธีแก้ไข

| # | ปัญหา | สาเหตุ | วิธีแก้ |
| --- | --- | --- | --- |
| 1 | mediapipe ใช้ไม่ได้ | version 0.10.22+ เปลี่ยน API | ล็อก `mediapipe==0.10.21` |
| 2 | `UnicodeDecodeError` ใน `video_ids.txt` | `echo 04 > file` บน Windows สร้าง UTF-16 BOM | เขียนด้วย Python: `open(...,'w',encoding='utf-8').write('04\n')` |
| 3 | `FileNotFoundError` ใน segmentation | `shlex.quote()` ใส่ single quote ที่ cmd.exe ไม่รู้จัก | `segmentation.py` line 71: `subprocess.run(shlex.split(cmd), shell=False)` |
| 4 | `unrecognized argument: --subtitle_dir` | argument ชื่อผิด | ใช้ `--pr_sub_path` แทน |
| 5 | `ModuleNotFoundError: numba/beartype` | dependencies ไม่ได้ติดตั้ง | `pip install numba beartype tqdm scikit-learn tabulate matplotlib` |
| 6 | PyTorch ไม่ detect GPU (RTX 5060 Ti) | Blackwell (sm_120) ต้องการ CUDA 12.8+ | `pip install --force-reinstall torch --index-url .../cu128` |
| 7 | Overlap ~88% ในผล alignment | 172 CC cues แต่ sign slots มีแค่ ~119 | `fix_overlap_vtt.py` clamp end time → overlap 0% |
| 8 | `MMPTModel` หา checkpoint ไม่เจอใน `align_gloss_labels.py` | ใช้ relative path เทียบกับ cwd | `os.chdir(fairseq_signclip/examples/MMPT)` ก่อน `load_model()` แล้ว `chdir` กลับ |
| 9 | `evaluate_all.py` match ได้แค่ 69/172 | text-lookup พลาด entry ที่ผู้ annotate แก้ข้อความ | เปลี่ยนเป็น index-based evaluation ด้วย `CC_Input` (119 cues) |
| 10 | `gt_by_text` เก็บแค่ first occurrence | dict guard `if key not in gt_by_text` | known limitation — กระทบ 1 duplicate entry เท่านั้น |

---

## 9. ผลสรุปและ Recommendation

### 9.1 Key Findings

| # | ข้อค้นพบ | หลักฐาน |
| --- | --- | --- |
| 1 | **SEA cross-lingual transfer ใช้ได้กับ TSL** โดยไม่ต้อง retrain | Task 1 C_MULTI 100% within ±3 s |
| 2 | **Multilingual > BSL-specific > ASL** สำหรับ TSL | B_MULTI +0.91 s vs D_ASL +1.25 s |
| 3 | **Gloss text > CC text** เป็น embedding input | C_MULTI −0.16 s vs B_MULTI +0.91 s |
| 4 | **Word-level similarity ไม่ช่วย** สำหรับ TSL | C_MULTI_word ≈ C_MULTI |
| 5 | **Token count ต้องตรงกับ GT** สำหรับ Task 2 | Gloss 852=852 (IoU 0.49) vs Gloss_Input 889≠852 (IoU 0.42) |
| 6 | **Per-sentence cropping ไม่ช่วย** ยกเว้นประโยคยาว (13+ tokens) | IoU 0.4763 vs baseline 0.4901, 7× ช้ากว่า |
| 7 | **Overlap fix ปลอดภัย** — ไม่กระทบ start-based metrics | Mean offset / ±Ns เหมือนเดิมหลัง fix |
| 8 | **Index-based evaluation จำเป็น** สำหรับ full coverage | text-lookup 69/172 (58%) → index 119/119 (100%) |
| 9 | **Token caching คุ้มมาก** — 192 unique tokens / 852 occurrences | run 1: 3 min, run 2+: 5 s |

### 9.2 Recommendation สำหรับ Production

**Task 1:** ใช้ `C_MULTI` configuration (Multilingual + Gloss text, CC_Input 119 cues) + `fix_overlap_vtt.py`
**Task 2:** ใช้ `align_gloss_labels.py --tier Gloss` แบบ whole-video — ดีที่สุด, เร็วที่สุด

### 9.3 Caution เรื่อง Metric Reporting

| Metric | ข้อควรระวัง |
| --- | --- |
| Task 1 "80% within ±1s" | เป็นตัวเลขจาก text-lookup (69/172) — ไม่ใช่ครบทุก cue |
| Task 1 "73.9% within ±1s" | index-based (119/119) — ตัวเลขที่ถูกต้อง |
| Task 2 text match 65.1% | มี structural leakage (annotator ใช้ Gloss tier เป็น base สร้าง GT) — รายงานด้วย IoU แทน |
| Task 2 IoU evaluation | ใช้ greedy matching (non-exclusive) → เป็น upper-bound estimate; Hungarian matching เข้มกว่า |

### 9.4 แนวทางพัฒนาต่อ

**Task 1:**

- ทดสอบกับวิดีโอชุดใหม่ (ยืนยัน generalization)
- Crop วิดีโอเฉพาะส่วนผู้แปล (ffmpeg) → ลด pose noise
- วิเคราะห์ outlier cues (offset > ±5 s) — stdev ~5.5 s สูงมาก

**Task 2:**

- Parameter sweep: `gap_penalty`, `coverage_penalty`
- ทดสอบ language tag อื่น (`<en> <ase>`, `<en>`, no-tag)
- ทดสอบ embedding model: multilingual vs ASL (ยังไม่ได้ทำใน Task 2)
- Pre-merge SIGN segments ก่อน DP (ลด over-segmentation)
- ทดสอบบน video ชุดอื่น

---

## 10. ForcedAlignment Dataset — Task 2 Scale-up (แผน 17 พ.ค. 2569)

> ขยาย Task 2 จากวิดีโอตัวอย่างเดียว (`04.mp4`) ไปยัง **1,132 คลิป** ใน ForcedAlignment dataset
> เพื่อวัดผลในระดับ dataset จริงและรายงานใน `Gloss_Labeling_Template.docx`
> แผนเต็มอยู่ใน [ForcedAlignment/PLAN_ForcedAlignment_Task2.md](ForcedAlignment/PLAN_ForcedAlignment_Task2.md)

### 10.1 ทำไมถึงทำ (Motivation)

งานทั้งหมดใน §5–6 ทดสอบบน `04.mp4` เพียงคลิปเดียว (11 นาที) ซึ่งไม่เพียงพอสำหรับยืนยัน generalization
ForcedAlignment dataset มี 1,132 คลิปสั้น (avg 5.8 s) พร้อม ground truth หลายรูปแบบ — ทั้ง CC-level และ Gloss-level
เป็นโอกาสวัดประสิทธิภาพจริงในระดับ dataset พร้อมรายงาน Precision / Recall / F1 / Accuracy

### 10.2 ข้อมูล Dataset

| รายการ | ค่า |
| --- | --- |
| จำนวน EAF / MP4 | **1,132 ไฟล์** (match ครบทุกไฟล์ ✓) |
| Clip duration (avg / median / min / max) | 5.8 s / 5.8 s / 3.5 s / 10.3 s |
| รวมเวลาทั้งหมด | **~110 นาที (1.82 ชั่วโมง)** |
| วิดีโอกระจายใน | 6 subfolder (เล่ม 1–6) |

**Tiers ใน EAF แต่ละไฟล์:**

| Tier | บทบาท | รูปแบบ |
| --- | --- | --- |
| `CC` | คำบรรยาย (ทั้งประโยค) | ข้อความเดียว |
| `CC_Aligned` | GT สำหรับ CC | `sil \| word \| sil` |
| `Gloss` | Gloss ทั้งประโยค | `คำ1\|คำ2\|` |
| `Gloss1` | Gloss + sil token | `sil\|คำ1\|คำ2\|sil\|` |
| `Gloss2` | Gloss + sil มีหมายเลข | `sil1\|คำ1\|คำ2\|sil2\|` |
| `Gloss_Labeling` | GT Gloss (ไม่มี sil) | word-level |
| `Gloss_Labeling1` | GT Gloss1 (มี sil) | sil \| word \| sil |
| `Gloss_Labeling2` | GT Gloss2 (มี sil1/sil2) | sil1 \| word \| sil2 |

**ตัวอย่าง (clip "สวัสดี"):**

```text
CC              : "สวัสดี"
CC_Aligned      : sil [0–1766ms] | สวัสดี [1766–3533ms] | sil [3533–5716ms]
Gloss_Labeling  : สวัสดี
Gloss_Labeling1 : sil | สวัสดี | sil
```

### 10.3 แผนการทดลอง 6 ชุด

| # | Input Tier | Ground Truth Tier | เหตุผล |
| --- | --- | --- | --- |
| 1 | `CC` (whitespace tokenize) | `CC_Aligned` | Baseline — align CC text กับ sign segment (ไม่มี sil) |
| 2 | `CC` (tokenize + sil) | `CC_Aligned` | ทดสอบว่าการ model sil frames ช่วย CC alignment ไหม |
| 3 | `Gloss` (pipe-delimited) | `Gloss_Labeling` | เทียบตรงกับ `Gloss` whole-video baseline (04.mp4) |
| 4 | `Gloss1` (มี sil token) | `Gloss_Labeling1` | ทดสอบ explicit sil modeling สำหรับ Gloss |
| 5 | `Gloss2` (มี sil1/sil2) | `Gloss_Labeling2` | ทดสอบ numbered sil — แยก sil เริ่มต้น/สิ้นสุดได้ไหม |
| 6 | `Gloss` (per-sentence mode) | `Gloss_Labeling` | เทียบกับ per-sentence pipeline ของ 04.mp4 (ผลแย่กว่า ~1.4 pp) |

**Metrics ที่รายงาน:** Precision / Recall / F1 / Accuracy ที่ IoU ≥ 0.5 + Mean IoU + % zero overlap

### 10.4 Pipeline ใหม่ vs Pipeline เดิม

| ด้าน | 04.mp4 (เดิม) | ForcedAlignment (ใหม่) |
| --- | --- | --- |
| Input | 1 ไฟล์ 11 นาที | 1,132 ไฟล์ avg 5.8 s |
| Pose | 1 × `04.pose` (358 MB) | 1,132 × `N.pose` (~3.3 GB รวม) |
| EAF | tiers รวมใน `Test.eaf` เดียว | EAF แยกต่อคลิป |
| Video location | `example_alignment/04.mp4` | กระจายใน 6 subfolder |
| Configs | ablation (Gloss vs Gloss_Input) | 6 configs พร้อมกัน |

**Script ใหม่ที่ต้องสร้าง:**

- `ForcedAlignment/run_forced_alignment.py` — Orchestrator 7 phases (scan → pose → seg → emb → DP × 6 → export → evaluate)
- `ForcedAlignment/evaluate_fa_dataset.py` — compute Precision/Recall/F1/Accuracy per config

**Script ที่ reuse จาก pipeline เดิม:**

- `align_one_sentence()` จาก `example_alignment/align_gloss_labels.py` (refactor ทำไปแล้วใน Progress_16052026 ✓)
- `evaluate_gloss_labeling.py` — ปรับรองรับ multi-clip

### 10.5 ประมาณ Computation Time (RTX 5060 Ti)

| Phase | เวลา | หมายเหตุ |
| --- | ---: | --- |
| Scan EAF + manifest | < 1 min | — |
| **Pose extraction × 1,132** | **~9.7 ชั่วโมง** | ⚠ Bottleneck (~31 s/clip init overhead) |
| SEA segmentation | ~30 min | model load 1× |
| SignCLIP embedding | ~27 min | model load 1× |
| DP alignment × 6 configs | ~60 min | in-process, GPU-light |
| Export + Evaluate | < 15 min | — |
| **รวม** | **~11–12 ชั่วโมง** | รัน overnight |

> **Optimistic (batch pose ดี):** ~5–6 ชั่วโมง
> **Pessimistic (per-clip subprocess overhead):** ~13–14 ชั่วโมง

### 10.6 ความเสี่ยงหลัก

| ความเสี่ยง | ระดับ | แนวทางรับมือ |
| --- | --- | --- |
| Pose extraction ช้า (~31 s/clip) | สูง | ตรวจว่า `videos_to_poses` batch per-directory ได้ไหม → model init 1× per subfolder |
| คลิปสั้น (3.5–4 s) — MediaPipe/GRU ขาด context | กลาง | Pad clip ±0.5–1 s ก่อน pose แล้ว crop output กลับ |
| EAF บางไฟล์ (1000+) มี RELATIVE_MEDIA_URL ว่างเปล่า | กลาง | Stem-based lookup แทน literal path (implement แล้วใน `check_eaf_video_match.py` ✓) |
| CC tier ไม่มี `\|` delimiter (Exp 1–2) | ต่ำ | Whitespace tokenize (CC มักเป็น 1 คำหรือวลีสั้น) |
| Disk space ~3.5 GB สำหรับ .pose + .npy | ต่ำ | ตรวจก่อนรัน |

### 10.7 คำถามที่ยังเปิดอยู่ (Open Questions)

| # | คำถาม | ผลกระทบ |
| --- | --- | --- |
| Q1 | CC1 vs CC2 tokenization strategy ต่างกันอย่างไรชัดๆ? | กำหนด Exp 1 vs 2 implementation |
| Q2 | `CC_Aligned2` tier มีอยู่ใน EAF ไหม? (ไม่พบใน scan) | อาจต้องสร้างเอง หรือ Exp 2 ใช้ `CC_Aligned` เหมือน Exp 1 |
| Q3 | Exp 6 (per-sentence) บนคลิปที่เป็น 1 ประโยคอยู่แล้ว — ต่างจาก Exp 3 ตรงไหน? | อาจเป็น padding/window strategy ต่างกัน |
| Q4 | DP parameters — ใช้ default (gap=2.0, coverage=0.5) หรือ tune ใหม่? | กระทบ IoU โดยตรง |

### 10.8 Timeline และสถานะ

```text
วันที่ 1 (เช้า–บ่าย)  : เขียน run_forced_alignment.py + evaluate_fa_dataset.py
วันที่ 1 (บ่าย)       : ทดสอบ debug mode 3 clips
วันที่ 1 (ค่ำ–คืน)    : รัน full pipeline overnight (~12 ชั่วโมง)
วันที่ 2 (เช้า)       : ตรวจผล + error logs
วันที่ 2 (เช้า–บ่าย)  : วิเคราะห์ผล + กรอก Gloss_Labeling_Template.docx + เขียน Progress note
```

**สถานะ (17 พ.ค. 2569):** กำลังวางแผน — ยังไม่เริ่ม computation

---

## 11. อ้างอิง

- **SEA Paper:** [arXiv:2512.08094](https://arxiv.org/abs/2512.08094) — Jiang, Z., Jang, Y., Momeni, L., Varol, G., Ebling, S., & Zisserman, A. (2025)
- **SignCLIP:** [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.518/)
- **Linguistic Segmentation (E4s-1):** [EMNLP 2023 Findings](https://aclanthology.org/2023.findings-emnlp.846/)
- **pose-format:** [sign-language-processing/pose](https://github.com/sign-language-processing/pose)
- **ELAN:** [archive.mpi.nl/tla/elan](https://archive.mpi.nl/tla/elan)
- **SEA Repository:** [J22Melody/SEA](https://github.com/J22Melody/SEA)
- **SignCLIP Repository:** [J22Melody/fairseq](https://github.com/J22Melody/fairseq)
