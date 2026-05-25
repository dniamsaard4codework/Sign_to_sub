# แผนการดำเนินงาน — Forced Alignment Task 2 บน ForcedAlignment Dataset

> **วันที่จัดทำ:** 17 พฤษภาคม 2569
> **อัปเดตล่าสุด:** 24 พฤษภาคม 2569 (Config #3 error analysis resolved; docx report finalized)
> **ผู้รับผิดชอบ:** dniamsaard4codework
> **สถานะ:** ✅ Done — full run + evaluation + error analysis + final report all completed

---

## 1. บริบทและแรงจูงใจ

### 1.1 ที่มา

โปรเจกต์ SEA (Sign Embedding Alignment) ได้ทดสอบ Task 2 (Forced Alignment ภาษามือไทย) บนวิดีโอ `04.mp4` เพียงไฟล์เดียวใน `example_alignment/` มาตลอด — ทั้งแบบ whole-video pipeline (`Progress_09052026`) และแบบ per-sentence cropping (`Progress_16052026`)

ขั้นตอนถัดไปคือ **ขยาย Task 2 ไปยัง ForcedAlignment Dataset** ซึ่งเป็นคลังวิดีโอภาษามือไทยขนาดใหญ่ที่มีการ annotate ไว้ล่วงหน้าอย่างละเอียด เพื่อวัดผลการทำงานของ pipeline ในระดับ dataset จริง ไม่ใช่แค่วิดีโอตัวอย่างเดียว

### 1.2 เป้าหมาย

1. รัน forced alignment pipeline ของ SEA บน **1,132 คลิปวิดีโอ** ใน ForcedAlignment dataset
2. ทดสอบ **5 การกำหนดค่าการทดลอง** (เดิม 6 — ตัด Exp 6 ออกหลัง verified ว่า per-sentence ≡ Exp 3 บน dataset นี้)
3. ประเมินผลด้วย Precision / Recall / F1-Score / Accuracy + Mean IoU
4. กรอกรายงานใน `Gloss_Labeling_Template.docx` และเขียน Progress note

### 1.3 ความสัมพันธ์กับงานที่ผ่านมา

| งาน | ผล Mean IoU | Note |
|---|---:|---|
| Task 2 — whole-video `Gloss` (`Progress_09052026`) | **0.4901** | best baseline ปัจจุบัน |
| Task 2 — per-sentence (`Progress_16052026`) | 0.4763 | แพ้ baseline ~1.4 pp |
| **Task 2 — ForcedAlignment (งานนี้)** | TBD | Dataset ใหม่, 5 configs |

---

## 2. ข้อมูล ForcedAlignment Dataset

### 2.1 ภาพรวม Dataset

| รายการ | ค่า |
|---|---|
| จำนวน EAF files | **1,132 ไฟล์** (`elan_forced_alignment/`) |
| จำนวนวิดีโอ MP4 | **1,132 ไฟล์** กระจาย 6 subfolder |
| ระยะเวลาคลิป (เฉลี่ย / มัธยฐาน) | **5.8 s / 5.8 s** |
| ระยะเวลาคลิป (min / max) | 3.5 s / 10.3 s |
| รวมเวลาวิดีโอทั้งหมด | **109.5 นาที ≈ 1.82 ชั่วโมง** |

#### EAF ↔ Video Match Verification

##### ✅ Stem-based lookup (functional pipeline test)

รัน `check_eaf_video_match.py` (stem-based lookup) บน 1,132 EAFs:

| ผลลัพธ์ | จำนวน |
| --- | ---: |
| OK (EAF stem == video stem, video file exists) | **1,132 / 1,132** ✅ |
| No media descriptor | 0 |
| Stem mismatch (EAF name ≠ video name) | 0 |
| Video not found in tree | 0 |

**สถานะ:** Pipeline ทำงานได้ครบ 1,132 clips โดย match ด้วย numeric stem (1.mp4 ↔ 1.eaf, etc.)

##### ✅ Deep audit ของ EAF MEDIA_DESCRIPTOR paths (18 พ.ค. 2569, repaired)

ตรวจ `RELATIVE_MEDIA_URL` และ `MEDIA_URL` ทุก EAF หลังรัน `fix_eaf_media_paths.py` ว่าสามารถ resolve เป็นไฟล์วิดีโอที่ถูกต้องจริงไหม:

| RELATIVE_MEDIA_URL status | จำนวน |
| --- | ---: |
| ✅ resolves to existing video on disk, stem matches | **1,132 / 1,132** ✅ |
| ❌ ชี้ไปยัง path ที่ไม่มีอยู่ | 0 |
| ❌ field ว่างเปล่า | 0 |
| ⚠️ resolves แต่ stem mismatch | 0 |

| MEDIA_URL (absolute) status | จำนวน |
| --- | ---: |
| ✅ `file:///C:/...` local Windows path | **1,132 / 1,132** ✅ |
| ❌ `file://192.168.1.18/...` network/SMB | 0 |
| ❌ empty / unknown scheme | 0 |

**สถานะหลัง repair:**

- EAF ทุกไฟล์เปิดวิดีโอถูกต้องด้วย path local บนเครื่องนี้
- path ถูกเขียนเป็น literal/plain local path (ไม่ใช้ percent-encoded path) เพื่อให้ ELAN อ่านง่ายขึ้น
- mapping folder ถูกต้องตามช่วงไฟล์:

| Folder | Clip range | Count |
|---|---:|---:|
| 1 MP | 1–185 | 185 |
| 2 MP | 186–371 | 186 |
| 3 MP | 372–560 | 189 |
| 4 MP | 561–754 | 194 |
| 5 MP | 755–925 | 171 |
| 6 MP | 926–1132 | 207 |

**Historical root cause ก่อน repair:**

EAFs ในชุดข้อมูลถูกสร้างจาก **หลาย annotator/เครื่อง** ที่ใช้ encoding และ mount point ต่างกัน ทำให้ก่อนซ่อมมี `RELATIVE_MEDIA_URL` ใช้ได้เพียง 2/1,132, มี broken path 890 ไฟล์, field ว่าง 240 ไฟล์, และ `MEDIA_URL` แบบ network host `file://192.168.1.18/...` 301 ไฟล์

**ผลกระทบ:**

- ✅ Pipeline ของเราใช้ **stem-based lookup** อยู่แล้ว → ไม่กระทบการรัน
- ✅ EAFs ใน workspace ปัจจุบันควรเปิดใน ELAN GUI ได้โดยตรง ไม่ต้อง re-link media ด้วยมือ
- ✅ `check_eaf_video_match.py` Part B เป็น regression check ก่อนรันงานหรือก่อนส่งไฟล์ให้ annotator
- ⚠️ ถ้าดึง raw archive ใหม่หรือ restore EAF ชุดเก่า ต้องรัน `fix_eaf_media_paths.py` อีกครั้งก่อนใช้ ELAN GUI

**คำสั่งที่ใช้:** `venv\Scripts\python.exe ForcedAlignment\fix_eaf_media_paths.py` แล้ว verify ด้วย `venv\Scripts\python.exe ForcedAlignment\check_eaf_video_match.py`

### 2.2 ที่อยู่ไฟล์วิดีโอ

วิดีโอกระจายอยู่ใน **6 subfolder** ตาม "เล่ม":

```
ForcedAlignment/
├── elan_forced_alignment/           # 1,132 × N.eaf  (annotations)
└── หนังสือ ภาษามือไทย/
    ├── หนังสือ ภาษามือไทยเล่ม 1 MP/  # 185 MP4
    ├── หนังสือ ภาษามือไทยเล่ม 2 MP/  # 186 MP4
    ├── หนังสือ ภาษามือไทยเล่ม 3 MP/  # 189 MP4
    ├── หนังสือ ภาษามือไทยเล่ม 4 MP/  # 194 MP4
    ├── หนังสือ ภาษามือไทยเล่ม 5 MP/  # 171 MP4
    └── หนังสือ ภาษามือไทยเล่ม 6 MP/  # 207 MP4
```

> **หมายเหตุด้านการอ่านไฟล์:** EAF media paths ใน workspace ปัจจุบันถูก repair แล้วด้วย
> `fix_eaf_media_paths.py` และ verify แล้วว่า `RELATIVE_MEDIA_URL` / `MEDIA_URL`
> resolve ได้ครบ 1,132/1,132 ไฟล์ ถ้า restore dataset จาก raw archive ให้รัน repair script นี้ซ้ำ
> ก่อนเปิดใน ELAN GUI

### 2.3 โครงสร้าง EAF Tiers

EAF แต่ละไฟล์มีทั้งหมด **8 tiers** ที่ verified ว่ามีข้อมูลครบ **1,132 / 1,132** ทุก tier (scan วันที่ 18 พ.ค. 2569):

| Tier ID | บทบาท | รูปแบบ | จำนวน annotation / คลิป |
|---|---|---|---:|
| `CC` | คำบรรยายต้นฉบับ (ทั้งประโยค) | ข้อความเดียว | 1.0 |
| `CC_Aligned` | Ground Truth สำหรับ CC (word-level, มี sil) | sil \| word \| sil | 3.0 |
| `Gloss` | Gloss tier (ทั้งประโยค คั่นด้วย `\|`) | `คำ1\|คำ2\|` | 1.0 |
| `Gloss1` | Gloss รวม sil token (คั่นด้วย `\|`) | `sil\|คำ1\|คำ2\|sil\|` | 1.0 |
| `Gloss2` | Gloss รวม sil ที่มีหมายเลข (คั่นด้วย `\|`) | `sil1\|คำ1\|คำ2\|sil2\|` | 1.0 |
| `Gloss_Labeling` | Ground Truth สำหรับ Gloss (ไม่มี sil) | word-level annotation | 1.5 |
| `Gloss_Labeling1` | Ground Truth สำหรับ Gloss1 (มี sil) | sil \| word \| sil | 3.5 |
| `Gloss_Labeling2` | Ground Truth สำหรับ Gloss2 (มี sil1/sil2) | sil1 \| word \| sil2 | 3.5 |

**สรุปตัวเลข Annotation ทั้ง Dataset:**

| Tier | รวมทั้ง dataset |
|---|---:|
| CC | 1,132 |
| CC_Aligned | 3,396 |
| Gloss_Labeling | 1,713 |
| Gloss_Labeling1 | 3,977 |
| Gloss_Labeling2 | 3,977 |

**ตัวอย่างข้อมูล (clip 1 — "สวัสดี"):**

```
CC              : "สวัสดี"
CC_Aligned      : sil [0–1766ms] | สวัสดี [1766–3533ms] | sil [3533–5716ms]
Gloss           : "สวัสดี|"
Gloss1          : "sil|สวัสดี|sil|"
Gloss2          : "sil1|สวัสดี|sil2|"
Gloss_Labeling  : สวัสดี
Gloss_Labeling1 : sil | สวัสดี | sil
Gloss_Labeling2 : sil1 | สวัสดี | sil2
```

---

## 3. แผนการทดลอง 5 ชุด (ปรับลดจาก 6 หลัง verified tier inventory)

### 3.1 Tier Inventory ที่ verified แล้ว (18 พ.ค. 2569)

จาก tier-presence scan บน 1,132 EAFs:

| Tier ที่ระบุไว้เดิมในแผน | สถานะจริง | ผลต่อแผน |
| --- | --- | --- |
| `CC`, `CC_Aligned` | ✅ 1,132/1,132 | ใช้ได้ |
| `Gloss`, `Gloss1`, `Gloss2` | ✅ 1,132/1,132 | ใช้ได้ |
| `Gloss_Labeling`, `Gloss_Labeling1`, `Gloss_Labeling2` | ✅ 1,132/1,132 | ใช้ได้ |
| `CC_Aligned2` (เคยตั้งสมมุติฐาน) | ❌ **0/1,132 — ไม่มี** | ต้อง redesign Exp 2 |
| `CC1`, `CC2` | ❌ 0/1,132 — ไม่มี | ทั้งสอง Exp 1 และ 2 ใช้ tier `CC` แล้ว tokenize ต่างกัน |

**ข้อสังเกตสำคัญ:** `CC_Aligned` มี structure `sil | word | sil` (3 entries/clip avg) — เหมือน `Gloss_Labeling1` แต่ใช้คำของ CC ดังนั้น Exp 1 vs Exp 2 แยกกันด้วย **strategy การประเมิน** ไม่ใช่ GT tier ที่ต่างกัน

### 3.2 ตารางการทดลองหลังปรับ

| # | Input Tier | Tokenization | Ground Truth | Output Tier (ที่จะสร้าง) | ลักษณะ |
|---|---|---|---|---|---|
| **1** | `CC` | whitespace (no sil) | `CC_Aligned` (กรอง sil ออก) → คำเดียวต่อคลิป | `CC_Aligned_pred` | word-only baseline |
| **2** | `CC` | whitespace + เพิ่ม `sil` ที่หัว/ท้าย | `CC_Aligned` (ใช้ตามจริง: sil \| word \| sil) | `CC_Aligned_silmodel_pred` | sil-modeling variant |
| **3** | `Gloss` (`\|`-split) | drop empty tokens | `Gloss_Labeling` | `Gloss_Labeling_pred` | ⭐ baseline (เทียบกับ 04.mp4 ablation) |
| **4** | `Gloss1` (`\|`-split รวม sil) | keep all incl. sil | `Gloss_Labeling1` | `Gloss_Labeling1_pred` | sil-token modeling |
| **5** | `Gloss2` (`\|`-split รวม sil1/sil2) | keep all incl. sil1/sil2 | `Gloss_Labeling2` | `Gloss_Labeling2_pred` | numbered-sil modeling |

> **Note 1:** เดิมแผนมี 6 configs โดย Exp 6 = "per-sentence variant ของ Exp 3" แต่เนื่องจาก **ทุกคลิปคือ 1 ประโยคอยู่แล้ว** (avg 5.8s = sentence-level by design) per-sentence mode จึงเทียบเท่า Exp 3 ทุกอย่าง — **ตัด Exp 6 ออกเพื่อประหยัด compute ~1.5 ชม.** (`Progress_16052026` ได้ทดสอบ per-sentence บน 04.mp4 ไปแล้ว ผลแย่กว่า baseline 1.4 pp)
>
> **Note 2:** ถ้าต้องการรักษา Exp 6 ใน docx template ให้รายงานผล Exp 6 = Exp 3 พร้อมหมายเหตุ "structurally equivalent on this dataset"

### 3.3 ข้อสังเกตเกี่ยวกับ Input Format

- **Exp 1–2:** `CC` tier เป็นข้อความเต็มประโยค 1 entry ไม่มี `|` delimiter → tokenize ด้วย `text.split()` (whitespace) ก่อนส่ง aligner คำส่วนใหญ่เป็นภาษาไทย "1 คำ" หรือ "วลีสั้น" (clip "สวัสดี" → `["สวัสดี"]`)
- **Exp 3–5:** `Gloss`, `Gloss1`, `Gloss2` มี `|` เป็น delimiter อยู่แล้ว → split แล้ว filter `""` ออก (ระวัง trailing `|`)
- **sil token handling (Exp 2/4/5):** sil ถูก embed ด้วย SignCLIP เหมือน token อื่น แต่ DP cost function ต้องเพิ่ม "low-similarity expected" prior ไม่งั้น sil prediction จะแย่
  - แนวทาง: ใช้ template `"<en> <bfi> [PAUSE]"` แทน `"<en> <bfi> sil"` ตอน embed sil tokens หรือ skip embedding sil ทั้งหมดและใช้ uniform fill ระหว่าง word boundaries

### 3.4 Metrics ที่ต้องรายงาน

สำหรับแต่ละ Experiment คำนวณ **per-clip** แล้ว aggregate:

| Metric | นิยาม | Threshold |
|---|---|---|
| **Precision** | count(pred ที่มี gt ทับซ้อน IoU ≥ τ) / count(total pred) | τ = 0.5 |
| **Recall** | count(gt ที่มี pred ทับซ้อน IoU ≥ τ) / count(total gt) | τ = 0.5 |
| **F1-Score** | 2 × P × R / (P + R) | — |
| **Accuracy** | count(matched pairs ที่ IoU ≥ τ) / count(total gt) — positional, IoU-only ไม่ใช้ text-match | τ = 0.5 |
| **Mean IoU** | average IoU ของทุก matched pair | — |
| **Frame Accuracy** | per-frame label agreement (เหมือน SEA paper, FPS=25) | — |

> **⚠️ ไม่ใช้ text-match metric สำหรับ Accuracy** — ตามคำเตือนใน `Big_Progress.md` §9.3 และ `Progress_09052026` §6: Gloss tokens ถูกใช้ในการสร้าง GT → text-match มี **structural leakage** ทำให้ตัวเลขดูสูงเกินจริง
>
> ใช้ **positional IoU matching** เป็น primary metric แทน — เทียบ pred[i] ↔ gt[i] ตาม token order ภายใน clip
>
> **Multi-threshold reporting:** รายงาน Mean IoU, %IoU≥0.5, %IoU≥0.3, %any-overlap ครบทั้ง 4 ค่าตามรูปแบบ Task 2 ablation บน 04.mp4 (`Progress_09052026`) เพื่อ direct comparison

---

## 4. สถาปัตยกรรม Pipeline

### 4.1 Diagram ภาพรวม

```
ForcedAlignment/elan_forced_alignment/*.eaf
        │  (อ่าน tier Gloss/CC + timestamps)
        ▼
[PHASE 1] สร้าง video_ids.txt สำหรับ 1,132 คลิป
        │
        ▼
[PHASE 2] Pose Extraction — MediaPipe Holistic
        │  videos_to_poses --directory <video_subfolder>
        │  ผลลัพธ์: 1,132 × N.pose  (เก็บใน output/poses/)
        │  ⚠ BOTTLENECK: ~9–10 ชั่วโมง
        ▼
[PHASE 3] SEA Sign Segmentation — E4s-1 (sign-b=30, sign-o=50)
        │  SEA/segmentation.py --video-ids video_ids.txt
        │  ผลลัพธ์: 1,132 × N.eaf  (SIGN tier, เก็บใน output/seg/)
        │  ~30 นาที
        ▼
[PHASE 4] SignCLIP Embeddings — multilingual
        │  extract_episode_features.py --mode=segmentation
        │  ผลลัพธ์: 1,132 × N.npy  (K_i × 768, เก็บใน output/emb/)
        │  ~27 นาที
        ▼
[PHASE 5] DP Alignment × 5 Experiment Configs
        │  align_gloss_labels.py (หรือ orchestrator ใหม่)
        │  สำหรับแต่ละ clip × แต่ละ config → predicted tier
        │  ~1 ชั่วโมง รวมทุก config (in-process, NumPy + Numba JIT)
        ▼
[PHASE 6] รวม Predictions กลับเข้า EAF + export CSV
        │
        ▼
[PHASE 7] Evaluate vs Ground Truth
        │  evaluate_gloss_labeling.py (ปรับให้รองรับ multi-config)
        │
        ▼
รายงานผล → กรอก Gloss_Labeling_Template.docx
```

### 4.2 ความแตกต่างจาก Pipeline เดิม (04.mp4)

| ด้าน | Pipeline เดิม (04.mp4) | Pipeline ใหม่ (ForcedAlignment) |
|---|---|---|
| Input video | 1 ไฟล์ยาว 11 นาที | 1,132 ไฟล์สั้น (avg 5.8s) |
| Pose file | 1 × `04.pose` (358 MB) | 1,132 × `N.pose` |
| EAF structure | tier อยู่ใน Test.eaf เดียว | tier อยู่ใน EAF แยกต่อ clip |
| Video location | `example_alignment/04.mp4` | กระจายใน 6 subfolder |
| Output | 1 CSV + 1 EAF | 1,132 per-clip EAF + aggregated CSV |
| Configs | 1 (ทดสอบแบบ ablation) | 5 configs พร้อมกัน |

---

## 5. แผนการเขียนโค้ด (Implementation Plan)

### 5.1 Script ที่ต้องสร้างใหม่

#### `ForcedAlignment/run_forced_alignment.py` — Orchestrator หลัก

```
รับ arguments:
  --eaf-dir       ForcedAlignment/elan_forced_alignment/
  --video-root    ForcedAlignment/หนังสือ ภาษามือไทย/
  --out-dir       ForcedAlignment/output/
  --configs       1,2,3,4,5  (หรือ all)
  --only-idx      0,1,2  (debug subset)
  --skip-pose     (ถ้ามี .pose แล้ว)
  --skip-seg      (ถ้ามี seg แล้ว)
  --skip-emb      (ถ้ามี emb แล้ว)

Phases:
  Phase 1: scan EAF dir → สร้าง video_ids.txt + clip_manifest.csv
           (clip_id, video_path, eaf_path, duration)
  Phase 2: videos_to_poses  (batch ทุก clip ใน 1 subfolder ต่อครั้ง)
  Phase 3: SEA segmentation  (batch ทุก clip)
  Phase 4: SignCLIP embedding  (batch ทุก clip)
  Phase 5: DP alignment  (per-clip, per-config, in-process)
  Phase 6: inject predicted tier กลับเข้า EAF + write CSV
  Phase 7: evaluate + print summary
```

#### `ForcedAlignment/evaluate_fa_dataset.py` — Evaluator สำหรับ multi-clip

```
อ่าน prediction CSV + ground truth จาก EAF
คำนวณ per-clip และ aggregate metrics สำหรับแต่ละ config
output: evaluation_summary.csv + console table
```

### 5.2 Script ที่ต้อง Refactor

| Script | การเปลี่ยนแปลง |
|---|---|
| `example_alignment/align_gloss_labels.py` | ตรวจสอบว่า `align_one_sentence()` helper พร้อม reuse ได้ (ทำไปแล้วใน Progress_16052026 ✓) |
| `example_alignment/run_task2_per_sentence.py` | ดึง logic video-finding ออกมาเป็น utility function ใช้ร่วมกัน |

### 5.3 Utility ที่มีแล้ว (ไม่ต้องเขียนใหม่)

| Script | บทบาท |
|---|---|
| `ForcedAlignment/check_eaf_video_match.py` ✓ | **Part A** stem-based match check + **Part B** deep MEDIA_DESCRIPTOR audit (upgraded 18 พ.ค.) |
| `ForcedAlignment/fix_eaf_media_paths.py` ✓ | rewrite `MEDIA_URL` + `RELATIVE_MEDIA_URL` ให้ชี้ local MP4 ถูกต้องครบ 1,132 ไฟล์ |
| `example_alignment/align_gloss_labels.py` ✓ | `align_one_sentence()`, `load_sign_segments()`, `embed_tokens_cached()` |
| `example_alignment/evaluate_gloss_labeling.py` ✓ | evaluate IoU per-prediction |

### 5.4 ELAN GUI Path Repair Helper

#### `ForcedAlignment/fix_eaf_media_paths.py` — [Implemented, ใช้ซ้ำได้]

```text
รับ input  : elan_forced_alignment/*.eaf + stem → local video path mapping
output     : rewrite MEDIA_URL และ RELATIVE_MEDIA_URL ใน EAF เดิมให้ชี้ local MP4 ถูกต้อง

ใช้สำหรับ : ทำให้ EAFs เปิดใน ELAN GUI ได้บนเครื่องนี้โดยไม่ต้อง re-link media
            และใช้ซ้ำได้ถ้า restore raw EAF archive กลับมา

หมายเหตุ   : pipeline หลักยังควรใช้ stem-based lookup เพราะ robust กับ raw/unrepaired archive
```

> Current status: รันแล้วบน workspace นี้ และ verify แล้วว่า media descriptors สะอาดครบ 1,132/1,132

### 5.5 ลำดับขั้นตอนการเขียนโค้ด (Coding Checklist)

- [ ] **Step 1:** สร้าง `run_forced_alignment.py` — Phase 1 (scan EAF + สร้าง manifest)
- [ ] **Step 2:** Phase 2 — pose extraction loop (แบ่งตาม subfolder, batch per subfolder)
- [ ] **Step 3:** Phase 3–4 — segmentation + embedding (batch ทุก clip พร้อมกัน)
- [ ] **Step 4:** Phase 5 — DP alignment สำหรับแต่ละ config (reuse `align_one_sentence`)
- [ ] **Step 5:** Phase 6 — inject prediction tier กลับเข้า EAF + export CSV
- [ ] **Step 6:** สร้าง `evaluate_fa_dataset.py` — compute Precision/Recall/F1/Accuracy
- [ ] **Step 7:** ทดสอบ debug mode (`--only-idx 0,1,2`) บน 3 clip ก่อน
- [ ] **Step 8:** รัน full pipeline (1,132 clips) — run overnight
- [ ] **Step 9:** รวบรวมผลและกรอก `Gloss_Labeling_Template.docx`

---

## 6. การประมาณเวลา (Time Budget)

### 6.1 Computation Time (RTX 5060 Ti)

| Phase | อัตราอ้างอิง | 1,132 clips × 5 configs | หมายเหตุ |
|---|---|---|---|
| Phase 1 — Scan EAF + manifest | < 1 นาที | **< 1 min** | (pre-verified ✓) |
| Phase 2 — Pose extraction | ~31 s/clip (per-clip subprocess) | **~9.7 ชั่วโมง** | ⚠ BOTTLENECK — model init overhead |
| Phase 3 — SEA segmentation | model 1× + ~1.6 s/clip | **~30 นาที** | batch ทั้ง 1,132 clips |
| Phase 4 — SignCLIP embedding | model 1× + ~1.4 s/clip | **~27 นาที** | batch ทั้ง 1,132 clips |
| Phase 5 — DP alignment × 5 configs | ~0.6 s/clip/config (vectorized, in-process) | **~60 นาที** | DP-only งานเบา ใช้ Numba JIT |
| Phase 6 — Inject + CSV | < 5 min | **< 5 min** | per-clip EAF update |
| Phase 7 — Evaluate 5 configs | < 10 min | **< 10 min** | |
| **รวม Computation (expected)** | | **~12 ชั่วโมง** | รัน overnight |

> **Phase 5 calculation:**
>
> - Per-clip per-config DP: ~0.6 s (vectorized, NumPy + Numba JIT, in-process — no subprocess overhead)
> - Total: 1,132 × 5 × 0.6 s = **3,396 s ≈ 57 นาที** ✓
> - หากไม่ vectorize (≈10 s/clip/config from subprocess overhead) จะเพิ่มเป็น ~16 ชม. → **ต้อง keep in-process**
>
> **Optimistic scenario (pose batch directory mode ใช้ได้):** ~5–6 ชั่วโมง
> **Pessimistic scenario (per-clip subprocess overhead ใน Phase 2):** ~13–14 ชั่วโมง

### 6.2 Implementation Time (Active Work)

| งาน | ประมาณเวลา |
|---|---|
| เขียน `run_forced_alignment.py` (Phase 1–6) | 6–8 ชั่วโมง |
| เขียน `evaluate_fa_dataset.py` | 2–3 ชั่วโมง |
| Debug + test บน 3 clips | 1–2 ชั่วโมง |
| รัน full pipeline + ตรวจผล | 1–2 ชั่วโมง (active) |
| เขียน Progress note + กรอก docx | 3–4 ชั่วโมง |
| **รวม Active Work** | **~13–19 ชั่วโมง (~2–2.5 วัน)** |

### 6.3 Timeline สรุป

```
วันที่ 1 (เช้า–บ่าย)  : เขียนโค้ด orchestrator + evaluator (Step 1–6)
วันที่ 1 (บ่าย)       : ทดสอบ debug mode 3 clips (Step 7)
วันที่ 1 (ค่ำ–คืน)    : รัน full pipeline overnight (Step 8) ← ~12 ชั่วโมง
วันที่ 2 (เช้า)       : ตรวจผล computation + ตรวจ error logs
วันที่ 2 (เช้า–บ่าย)  : วิเคราะห์ผล + กรอก docx + เขียน Progress note (Step 9)
```

---

## 7. โครงสร้าง Output Files ที่คาดหวัง

```
ForcedAlignment/
├── output/
│   ├── manifest.csv                              # clip_id, video_path, eaf_path, duration
│   ├── video_ids.txt                             # 1,132 บรรทัด
│   ├── poses/
│   │   └── N.pose                               # 1,132 ไฟล์
│   ├── seg/E4s-1_30_50/
│   │   └── N.eaf                                # 1,132 ไฟล์ (SIGN tier)
│   ├── emb/
│   │   └── N.npy                                # 1,132 ไฟล์ (K_i × 768)
│   ├── predictions/
│   │   ├── config1_CC_Aligned_pred.csv             # 1,132 rows  (CC, no sil)
│   │   ├── config2_CC_Aligned_silmodel_pred.csv    # 1,132 rows  (CC, with sil)
│   │   ├── config3_Gloss_Labeling_pred.csv         # 1,132 rows  ⭐ baseline
│   │   ├── config4_Gloss_Labeling1_pred.csv        # 1,132 rows  (sil token)
│   │   └── config5_Gloss_Labeling2_pred.csv        # 1,132 rows  (sil1/sil2)
│   └── evaluation/
│       ├── eval_config1.csv                        # per-row IoU + match
│       ├── eval_config2.csv
│       ├── eval_config3.csv
│       ├── eval_config4.csv
│       ├── eval_config5.csv
│       └── evaluation_summary.csv                 # 5 rows × {P, R, F1, Acc, Mean IoU}
├── PLAN_ForcedAlignment_Task2.md                # ไฟล์นี้
├── check_eaf_video_match.py                     # ✓ Part A (stem) + Part B (deep) audit
├── fix_eaf_media_paths.py                       # ✓ rewrite EAF media links to local MP4 paths
├── run_forced_alignment.py                      # [TODO] orchestrator
├── evaluate_fa_dataset.py                       # [TODO] evaluator
└── Gloss_Labeling_Template.docx                 # กรอกผลท้ายสุด
```

---

## 8. ความเสี่ยงและข้อควรระวัง

### 8.1 ความเสี่ยงสูง — Pose Extraction Bottleneck

**ปัญหา:** MediaPipe Holistic init overhead ทุก clip ทำให้ pose extraction ช้ามาก (~31 s/clip)
สำหรับ 1,132 clips = ~9.7 ชั่วโมง

**แนวทางลด:**
- ถ้า `videos_to_poses` รองรับ batch directory input → ส่ง videos ทีละ subfolder (6 รอบ แทน 1,132 รอบ)
- ตรวจสอบว่า `videos_to_poses` init model ครั้งเดียวต่อ directory หรือต่อ file
- ถ้า model init ทำได้ครั้งเดียว: เวลาอาจลดเหลือ 5–6 ชั่วโมง

### 8.2 ความเสี่ยงกลาง — คลิปสั้นมาก (< 4s)

**ปัญหา:** คลิปที่สั้นกว่า 4 วินาที (min = 3.5s) อาจทำให้:
- MediaPipe ขาด temporal context → sign boundary ไม่แม่น
- SEA GRU cold-start → detect sign แรกของคลิปช้า
- ผลจาก `Progress_16052026` §7: bucket 1–3 tokens เสียหาย −15.93 pp

**แนวทาง:** Pad คลิปด้วย 0.5–1s ก่อนส่ง pose extraction แล้ว crop output กลับ

### 8.3 ความเสี่ยงควบคุมแล้ว — EAF MEDIA_DESCRIPTOR Path Corruption

**สถานะปัจจุบัน (18 พ.ค. 2569):**

- ✅ `RELATIVE_MEDIA_URL` resolve ถูกต้อง **1,132 / 1,132**
- ✅ `MEDIA_URL` เป็น local `file:///C:/...` path **1,132 / 1,132**
- ✅ ไม่มี empty path, broken path, network host, หรือ percent-encoded path เหลืออยู่

**สาเหตุเดิมก่อน repair:** Dataset ถูก annotate โดยหลายคนใน LAN ภายในที่ใช้ encoding และ mount point ต่างกัน — กลับมา archive รวมในเครื่องเดียวภายหลัง ทำให้ raw EAF archive เคยมี broken/empty/network paths จำนวนมาก

**ผลกระทบต่อ Pipeline:**

| ส่วน | ผลกระทบ |
| --- | --- |
| Stem-based lookup (เราใช้) | ✅ ไม่กระทบ — match ด้วย numeric stem 1132/1132 OK |
| ELAN GUI เปิดไฟล์ | ✅ workspace ปัจจุบันเปิดได้โดยตรงหลัง repair |
| Scripts ที่ใช้ ELAN's path API โดยตรง | ✅ ใช้ได้บน repaired EAFs; ยังเสี่ยงถ้าใช้ raw/unrepaired archive |
| ELAN Force Alignment plugin (ถ้าใช้) | ✅ ใช้ repaired EAFs ได้; raw archive ต้อง repair ก่อน |

**แนวทางที่ใช้:**

1. ✅ **Pipeline ใช้ stem-based video lookup เท่านั้น** ผ่าน `check_eaf_video_match.py` style:

   ```python
   video_index = {p.stem: p for p in FA_DIR.rglob("*.mp4")}
   video_path = video_index[eaf_path.stem]   # ไม่อ่าน MEDIA_DESCRIPTOR
   ```

2. ✅ **ใช้ `fix_eaf_media_paths.py` เป็น repair step** ถ้า restore raw EAF archive หรือถ้าต้องส่งไฟล์ให้เปิดใน ELAN GUI
3. ⚠️ **ระวัง `pympi.Elan.Eaf.get_linked_files()` กับ raw archive** — raw EAF อาจคืน path mojibake/network ที่ใช้ไม่ได้ แต่ repaired EAFs ใน workspace ปัจจุบัน clean แล้ว
4. ✅ **`check_eaf_video_match.py` ตอนนี้รายงาน BOTH Parts:** Part A (stem-based, pipeline test) และ Part B (deep MEDIA_DESCRIPTOR audit) → รัน script เดียวเห็นทั้ง 2 มิติ exit code: 0 = ทั้งสองสะอาด, 2 = stem OK แต่ paths broken, 1 = pipeline blocker

### 8.4 ความเสี่ยงต่ำ — Experiment 1 / 2 (CC tier)

**ปัญหา:** `CC` tier เป็นข้อความเต็มประโยคไม่มี `|` delimiter ขณะที่ aligner คาดหวัง
token list

**แนวทาง:** Tokenize ด้วย whitespace ก่อนส่ง aligner (CC มักเป็น Thai word ตัวเดียวหรือวลีสั้น)
ตัวอย่าง: "สวัสดี" → ["สวัสดี"], "ดีใจ" → ["ดีใจ"]

### 8.5 ความเสี่ยงต่ำ — Disk Space

| รายการ | ประมาณ |
|---|---|
| 1,132 × .pose files | ~1,132 × 3 MB avg = **~3.3 GB** |
| 1,132 × .npy embeddings | ~1,132 × 0.1 MB avg = **~113 MB** |
| 1,132 × seg EAF | ~1,132 × 50 KB avg = **~55 MB** |
| **รวม** | **~3.5 GB** |

ตรวจให้แน่ใจว่ามี disk space เพียงพอก่อนรัน

---

## 9. เกณฑ์การประเมิน (Evaluation Criteria)

### 9.1 นิยาม "Match" สำหรับ Precision/Recall/F1

ใช้ **positional IoU matching** (สอดคล้องกับ Task 2 ablation บน 04.mp4 ใน `Progress_09052026`):

```text
สำหรับแต่ละ clip ใช้ index-based pairing:
  pred[i] ↔ gt[i]  (เรียง token ตาม time order)

แต่ละ pair → คำนวณ:
  IoU(pred[i], gt[i]) = intersection_ms / union_ms

threshold ที่แนะนำ: τ = 0.5  (เหมือนกับ ablation 04.mp4)
```

นิยามที่ใช้ (IoU-only, **ไม่ใช้ text-match**):

```text
Precision = count(pred ที่มี gt ทับซ้อน IoU ≥ τ) / count(total pred)
Recall    = count(gt ที่มี pred ทับซ้อน IoU ≥ τ) / count(total gt)
F1        = 2 × Precision × Recall / (Precision + Recall)
Accuracy  = count(matched pairs ที่ IoU ≥ τ) / count(total gt)
            (positional, IoU-only)
```

> **⚠️ ทำไมไม่ใช้ text-match ใน Accuracy:** `Big_Progress.md` §9.3 และ `Progress_09052026` §6 ระบุว่า Gloss tokens ถูกใช้ในการ build GT (`Gloss_Labeling*` ใช้ Gloss tier เป็น base) → **text-match มี structural leakage** ทำให้ Accuracy ดูสูงเกินจริง 50+ pp ใช้ IoU-only จึงเป็น metric บริสุทธิ์

### 9.2 Metrics เพิ่มเติมที่ควรรายงาน (นอกเหนือ docx template)

- Mean IoU (เทียบกับ ablation 04.mp4 ได้)
- % IoU ≥ 0.5 (exact match threshold)
- % zero overlap
- Mean abs start / end offset
- Breakdown ตามจำนวน token ต่อคลิป (1-2 tokens, 3-4 tokens, ≥5 tokens)

---

## 10. สิ่งที่ต้องทำก่อนเริ่มเขียนโค้ด (Pre-requisites)

- [x] **ตรวจสอบ EAF ↔ video match (stem-based)** — 1,132/1,132 OK (verified 2026-05-18 ด้วย `check_eaf_video_match.py` Part A)
- [x] **Repair + deep audit ของ EAF MEDIA_DESCRIPTOR paths** — 1,132/1,132 clean หลังรัน `fix_eaf_media_paths.py` และ verify ด้วย `check_eaf_video_match.py` Part B (ดู §8.3)
- [x] **วิเคราะห์โครงสร้าง EAF tiers** — 8 tiers × 1,132 EAFs OK (CC, CC_Aligned, Gloss, Gloss1, Gloss2, Gloss_Labeling, Gloss_Labeling1, Gloss_Labeling2)
- [x] **Resolve Q2 (CC_Aligned2)** — ไม่มีจริง → ปรับ Exp 2 ให้ใช้ CC_Aligned (เดียวกับ Exp 1) แต่ tokenize CC แบบรวม sil
- [x] **Resolve Q3 (Exp 6 redundancy)** — ตัด Exp 6 ออก (ทุกคลิป = 1 sentence อยู่แล้ว → เทียบเท่า Exp 3) เหลือ 5 configs
- [x] **ยืนยัน threshold IoU** — ใช้ τ = 0.5 (เหมือน 04.mp4 ablation)
- [x] **ยืนยันนิยาม "match" สำหรับ CC tier** — whitespace tokenize ผ่าน `text.split()`
- [x] **ประมาณ computation time** — ~12 ชั่วโมง expected (5–14 ชม. ขึ้นกับ pose batch mode)
- [x] **ตรวจสอบ disk space** — C: free ~479 GB (checked 2026-05-20) เพียงพอสำหรับ output ~4 GB+
- [x] **ตรวจสอบ `videos_to_poses` batch directory mode** — ✅ full overnight run ใช้ staged directory mode สำเร็จ 1,132/1,132 clips (2026-05-21)
- [x] **ทดสอบ in-process DP** บน 3 clips ก่อน — passed end-to-end smoke test (clips 1, 500, 1132; configs 1–5; 2026-05-20)

### 10.1 Pre-implementation Smoke Test

ก่อน implement orchestrator เต็มรูปแบบ ให้รัน end-to-end pipeline บน **3 clips** เพื่อ verify ทุก phase ทำงานต่อเนื่อง:

```text
1. Pick 3 clips: clip 1 (สั้นสุด), clip ~500 (กลาง), clip 1132 (เลขสูง)
2. Run Phase 2–7 sequentially บน 3 clips
3. Check outputs:
   - output/poses/{1,N,1132}.pose       (3 ไฟล์)
   - output/seg/E4s-1_30_50/{...}.eaf   (3 ไฟล์ มี SIGN tier)
   - output/emb/{...}.npy               (3 ไฟล์ shape (K_i, 768))
   - output/predictions/config{1..5}_*.csv  (5 ไฟล์ × 3 rows)
   - output/evaluation/evaluation_summary.csv  (5 rows)
4. Sanity check: Mean IoU ของ Exp 3 (Gloss → Gloss_Labeling) ควรใกล้ 0.4–0.6 เหมือน 04.mp4
5. ถ้า OK → run full 1,132 overnight
```

---

## 11. คำถามที่ resolved แล้ว (Resolved Questions)

| # | คำถาม | คำตอบที่ตัดสินใจ (18 พ.ค. 2569) |
|---|---|---|
| ~~Q1~~ ✅ | CC1 และ CC2 ต่างกันอย่างไร | ทั้งสองไม่มีอยู่จริงใน EAF (verified) — Exp 1 และ 2 ใช้ tier `CC` แล้ว tokenize ต่างกัน (no-sil vs with-sil) |
| ~~Q2~~ ✅ | CC_Aligned2 คืออะไร? มีใน EAF ไหม | **ไม่มี** (0/1132 EAFs) — Exp 2 ใช้ `CC_Aligned` เดียวกับ Exp 1 (ซึ่งมี structure `sil \| word \| sil` อยู่แล้ว) |
| ~~Q3~~ ✅ | Exp 6 (per-sentence) ต่างจาก Exp 3 อย่างไร | **ไม่ต่าง** — ทุกคลิปเป็น 1 sentence อยู่แล้ว → ตัด Exp 6 ออก เหลือ 5 configs |
| Q4 | DP parameters — ใช้ default (`gap=2.0, coverage=0.5`) หรือ tune ใหม่ | **ใช้ default ก่อน** สำหรับ baseline run — ถ้าผลแย่กว่า 04.mp4 baseline (Mean IoU 0.49) มากกว่า 10 pp ให้ sweep parameters หลังจากนั้น |
| Q5 | จะรัน Experiment 1–5 แบบ parallel หรือ sequential | **Sequential ใน-process** (single Python session) — Phase 2 (pose) ใช้ GPU แต่ Phase 3–7 บน CPU/RAM ส่วนใหญ่ การ batch sequential ดีกว่าเพื่อ minimize model-load overhead |

> **สถานะ:** ✅ คำถามวางแผนทั้งหมดได้รับการ resolve แล้ว แผนพร้อม implement

---

## 12. การเชื่อมโยงกับไฟล์อื่นในโปรเจกต์

| ไฟล์ | บทบาท |
|---|---|
| [../Big_Progress.md](../Big_Progress.md) §10 | Master reference สำหรับโปรเจกต์ — section 10 สรุปแผน ForcedAlignment |
| [../Progress_09052026.md](../Progress_09052026.md) | Task 2 Gloss vs Gloss_Input ablation บน 04.mp4 (Mean IoU 0.49 baseline) |
| [../Progress_16052026.md](../Progress_16052026.md) | Task 2 per-sentence บน 04.mp4 (per-sentence baseline) |
| [../example_alignment/align_gloss_labels.py](../example_alignment/align_gloss_labels.py) | `align_one_sentence()` — core DP aligner (reuse) |
| [../example_alignment/run_task2_per_sentence.py](../example_alignment/run_task2_per_sentence.py) | ตัวอย่าง orchestrator architecture |
| [../example_alignment/evaluate_gloss_labeling.py](../example_alignment/evaluate_gloss_labeling.py) | IoU evaluator (ปรับให้รองรับ multi-config) |
| [check_eaf_video_match.py](check_eaf_video_match.py) | ตรวจ EAF ↔ video (✅ 1132/1132 OK) |
| [fix_eaf_media_paths.py](fix_eaf_media_paths.py) | repair MEDIA_DESCRIPTOR ให้เปิด EAF ใน ELAN GUI ได้โดยตรง |
| [run_forced_alignment.py](run_forced_alignment.py) | ✅ Orchestrator 7 phases สำหรับ ForcedAlignment dataset |
| [evaluate_fa_dataset.py](evaluate_fa_dataset.py) | ✅ Dataset-level evaluator สำหรับ configs 1–5 |
| `Gloss_Labeling_Template.docx` | template รายงาน (กรอกผลสุดท้าย) |

---

## 13. TL;DR — สรุปสำหรับอ่านเร็ว

| รายการ | ค่า |
|---|---|
| Dataset | 1,132 คลิป × 5.8s avg, รวม 110 นาที |
| EAF ↔ video match (stem) | ✅ **1,132 / 1,132 OK** (verified 18 พ.ค. 2569) |
| EAF MEDIA_DESCRIPTOR paths | ✅ **1,132 / 1,132 clean** — repaired with local paths, ELAN GUI-ready (ดู §8.3) |
| Tiers (verified) | 8 tiers × 1,132 EAFs ✅ — CC, CC_Aligned, Gloss, Gloss1, Gloss2, Gloss_Labeling, Gloss_Labeling1, Gloss_Labeling2 |
| จำนวน Experiments | **5 configs** (ลดจาก 6 — ตัด Exp 6 ที่ redundant กับ Exp 3) |
| Metrics (IoU-only, no text-match leakage) | Precision / Recall / F1 / Accuracy / Mean IoU (τ = 0.5) |
| Computation | ~12 ชั่วโมง expected (5–14 ชม. range) → รัน overnight |
| Implementation | ✅ scripts created: `run_forced_alignment.py`, `evaluate_fa_dataset.py` |
| Timeline รวม | **~3 วัน** (วันที่ 1: โค้ด+smoke test, คืนที่ 1: compute, วันที่ 2: วิเคราะห์+รายงาน) |
| Blocker ใหญ่สุด | ✅ ไม่มี — pose extraction, full run, error analysis เสร็จครบ |
| Open questions | ✅ **resolved ทั้งหมด** (Q1–Q3 — verified จาก data; Q4–Q5 — default settings + sequential in-process; Config #3 regression — annotation convention mismatch ดู §14.6) |
| สถานะ | ✅ **เสร็จสมบูรณ์ — full run + evaluation + error analysis + final docx report** |
| ผลหลักของรายงาน | Config #1 (CC → CC_Aligned): F1 68.6%, mIoU 0.5928 (ดีกว่า 04.mp4 baseline 0.4901) |

---

## 14. Execution Status — 20 พฤษภาคม 2569

### 14.1 สิ่งที่ทำแล้ว

- ✅ รัน `fix_eaf_media_paths.py` ซ้ำและ verify ด้วย `check_eaf_video_match.py`
  - EAF ↔ video stem lookup: **1,132 / 1,132 OK**
  - `RELATIVE_MEDIA_URL`: **1,132 / 1,132 clean**
  - `MEDIA_URL`: **1,132 / 1,132 local file path**
- ✅ สร้าง `ForcedAlignment/run_forced_alignment.py`
  - scan manifest / stage videos / pose / segmentation / embeddings / DP configs 1–5 / inject EAF tiers / evaluate
- ✅ สร้าง `ForcedAlignment/evaluate_fa_dataset.py`
  - positional IoU-only Precision / Recall / F1 / Accuracy / Mean IoU / frame accuracy
- ✅ Smoke test end-to-end บน clips **1, 500, 1132** สำเร็จ
  - output poses: 3/3
  - output segmentation EAFs: 3/3
  - output embeddings: 2/3 (`1.eaf` had no SIGN segments, handled by uniform fallback)
  - output prediction CSVs: configs 1–5
  - output evaluation summary: 5 rows

### 14.2 Smoke Test Results

Command:

```powershell
venv\Scripts\python.exe ForcedAlignment\run_forced_alignment.py --only-ids 1,500,1132 --configs all
```

Summary:

| Config | Pred | GT | P@0.5 | R@0.5 | F1@0.5 | Mean IoU | Frame Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | 66.7% | 66.7% | 66.7% | 0.6024 | 64.1% |
| 2 | 9 | 9 | 33.3% | 33.3% | 33.3% | 0.3669 | 38.3% |
| 3 | 5 | 5 | 20.0% | 20.0% | 20.0% | 0.3417 | 48.7% |
| 4 | 11 | 11 | 27.3% | 27.3% | 27.3% | 0.2848 | 35.3% |
| 5 | 11 | 11 | 36.4% | 36.4% | 36.4% | 0.3425 | 39.8% |

### 14.3 Full Run Completed

Full dataset command:

```powershell
venv\Scripts\python.exe ForcedAlignment\run_forced_alignment.py --configs all
```

Run summary:

| Item | Value |
|---|---|
| Start | 20 พ.ค. 2569 ~13:26 |
| Finish | 21 พ.ค. 2569 ~01:30 |
| Runtime | 43,395.3 s ≈ 12.05 hours |
| stdout log | `ForcedAlignment/output/logs/full_run_20260520_132647.out.log` |
| stderr/progress log | `ForcedAlignment/output/logs/full_run_20260520_132647.err.log` |
| Poses | 1,132 / 1,132 |
| Segmentation EAFs | 1,132 / 1,132 |
| Embeddings | 1,075 / 1,132 (`57` clips had no valid SIGN embeddings and used fallback) |
| Predicted EAFs | 1,132 / 1,132 |
| GT-vs-prediction comparison EAFs | 1,132 / 1,132 (`ForcedAlignment/output/comparison_eafs`) |
| Prediction CSVs | 5 / 5 |
| Evaluation CSVs | 6 (`eval_config1..5.csv` + `evaluation_summary.csv`) |

If smoke outputs should be reused and only missing clips should be processed, run without `--overwrite`.
If a clean rerun is needed, add `--overwrite`.

### 14.4 Full Run Results

`ForcedAlignment/output/evaluation/evaluation_summary.csv`:

| Config | Pred | GT | P@0.5 | R@0.5 | F1@0.5 | Mean IoU | Frame Acc | Fallback preds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,171 | 1,132 | 67.5% | 69.8% | 68.6% | 0.5928 | 76.8% | 57 |
| 2 | 3,435 | 3,396 | 17.5% | 17.7% | 17.6% | 0.2039 | 21.3% | 427 |
| 3 | 1,713 | 1,713 | 7.8% | 7.8% | 7.8% | 0.2484 | 26.2% | 73 |
| 4 | 3,977 | 3,977 | 20.7% | 20.7% | 20.7% | 0.2229 | 22.3% | 495 |
| 5 | 3,977 | 3,977 | 20.9% | 20.9% | 20.9% | 0.2238 | 22.4% | 495 |

**Important note:** Config 1 (`CC` → `CC_Aligned`, no sil) is the strongest result on this full dataset. Config 3 (`Gloss` → `Gloss_Labeling`) is much lower than the 04.mp4 baseline (0.2484 vs 0.4901 Mean IoU). ✅ **Error analysis completed (24 พ.ค. 2569)** — root cause identified, see §14.6 below.

### 14.6 Error Analysis — Config #3 Mean IoU regression (resolved 24 พ.ค. 2569)

**Finding:** Config #3's low Mean IoU is **not aligner failure** — it's an annotation-convention mismatch between the ForcedAlignment dataset and the 04.mp4 reference.

**Quantitative evidence (computed on `eval_config3.csv`):**

| Metric | Config 1 | Config 3 | Implication |
|---|---:|---:|---|
| Mean pred duration | 1.70s | 1.16s | Aligner predicts tight intervals around the actual sign |
| Mean GT duration | 1.91s | **3.84s** | GT in Gloss_Labeling is 2x wider than CC_Aligned-without-sil |
| pred/GT duration ratio | 0.89 | **0.30** | Config 3 pred is only 30% of GT width — IoU mechanically capped |
| Text match (positional) | 97.1% | **100%** | Token identity & order both correct in Config 3 |
| pct any overlap over GT | 96.4% | **97.3%** | Aligner does land inside the right GT interval |
| pred fully contained in GT | n/a | **72.7%** | Majority of preds lie entirely within the correct GT span |
| mean fraction of pred inside GT | n/a | **88.7%** | Almost the whole pred sits inside the GT interval |

**Root cause:**

- `Gloss_Labeling` annotates each Gloss word to fill contiguous time spans across the clip (no silence gaps between words; each word ~3.8s on a 5.8s clip)
- `CC_Aligned` (with `sil` dropped) leaves only the active-sign portion (~1.9s)
- The DP aligner predicts tight intervals matching where the sign actually occurs (~1.0–1.7s)
- IoU = intersection/union — when GT is 3x wider than pred, IoU is bounded above by pred/GT ratio (~0.30) even when pred is perfectly placed inside GT

**Why 04.mp4 scored 0.49 on the same config:** The 04.mp4 `Gloss_Labeling` was annotated with tighter per-word boundaries (closer to active-sign timing), so pred/GT widths matched and IoU could go higher. The methodology is identical — only the annotation convention differs.

**Implications for the report:**

1. **Headline result = Config #1** (F1 68.6%, mIoU 0.5928) — directly comparable to 04.mp4 baseline (0.4901) and slightly **better** ✓
2. **Do NOT** compare Config #3 mIoU 0.2484 against 04.mp4 mIoU 0.4901 as a regression — they measure different annotation conventions
3. For Gloss-tier comparison, prefer `pct_any_overlap_over_gt` (97.3%) or `frame_accuracy` (26.2%) — both are less sensitive to GT width
4. Config #4-5 (sil-modeling) achieve mIoU ~0.22 — slightly *lower* than #3 because adding sil tokens further subdivides GT, but Recall@0.5 jumps to 20.7% (vs 7.8% for #3) because the sub-divided GT widths come closer to pred widths

**Status:** ✅ Config #3 regression resolved as annotation-convention artifact, not aligner regression. Report `Gloss_Labeling_Report_Filled.docx` updated 24 พ.ค. 2569 with this analysis.

### 14.5 ELAN Comparison EAFs

Generated with:

```powershell
venv\Scripts\python.exe ForcedAlignment\create_comparison_eafs.py --configs all
```

Output:

| Artifact | Value |
|---|---|
| Comparison EAF folder | `ForcedAlignment/output/comparison_eafs` |
| Comparison EAF count | 1,132 |
| Index CSV | `ForcedAlignment/output/comparison_eafs/comparison_index.csv` |
| Tier guide | `ForcedAlignment/output/comparison_eafs/README_tiers.md` |
| Added tiers per clip | `cfgN_GT_*`, `cfgN_PRED_*`, `cfgN_EVAL_*` for configs 1-5 |

Notes:

- The comparison files are generated from `ForcedAlignment/elan_forced_alignment`, so the original ground-truth tiers and corrected media paths are preserved.
- `cfg1_GT_CC_Aligned_nosil` drops `sil/sil1/sil2`, matching the evaluation rule for config 1.
- `cfgN_EVAL_*` tiers contain per-token IoU, hit/miss at IoU 0.5, text check, start/end offsets, and fallback marker where applicable.

Tier explanations:

| Tier | Meaning | Used by |
|---|---|---|
| `CC` | Source caption/text sentence for the clip. Usually one annotation covering the full sentence. | Input for config 1 and 2 |
| `CC_Aligned` | Ground-truth time alignment for `CC`; can include `sil`. | GT for config 1 and 2 |
| `Gloss` | Source gloss sequence, normally pipe-separated. | Input for config 3 |
| `Gloss1` | Source gloss sequence with generic `sil` tokens. | Input for config 4 |
| `Gloss2` | Source gloss sequence with `sil1` and `sil2` tokens. | Input for config 5 |
| `Gloss_Labeling` | Ground-truth time alignment for `Gloss`. | GT for config 3 |
| `Gloss_Labeling1` | Ground-truth time alignment for `Gloss1`, including generic `sil`. | GT for config 4 |
| `Gloss_Labeling2` | Ground-truth time alignment for `Gloss2`, including `sil1` and `sil2`. | GT for config 5 |
| `cfgN_GT_*` | Ground-truth intervals copied into a config-specific review tier. | ELAN comparison |
| `cfgN_PRED_*` | Predicted intervals from the forced-alignment pipeline. | ELAN comparison |
| `cfgN_EVAL_*` | Per-token comparison labels: index, HIT/MISS at IoU 0.5, IoU value, text status, predicted token, GT token, start/end offsets, and fallback marker. | ELAN comparison |

Config-to-tier map:

| Config | Input | GT tier | Added GT tier | Added prediction tier | Added evaluation tier |
|---:|---|---|---|---|---|
| 1 | `CC` | `CC_Aligned` without `sil/sil1/sil2` | `cfg1_GT_CC_Aligned_nosil` | `cfg1_PRED_CC_Aligned` | `cfg1_EVAL_CC_Aligned` |
| 2 | `CC` plus modeled silence | `CC_Aligned` including `sil` | `cfg2_GT_CC_Aligned` | `cfg2_PRED_CC_Aligned_silmodel` | `cfg2_EVAL_CC_Aligned_silmodel` |
| 3 | `Gloss` | `Gloss_Labeling` | `cfg3_GT_Gloss_Labeling` | `cfg3_PRED_Gloss_Labeling` | `cfg3_EVAL_Gloss_Labeling` |
| 4 | `Gloss1` | `Gloss_Labeling1` | `cfg4_GT_Gloss_Labeling1` | `cfg4_PRED_Gloss_Labeling1` | `cfg4_EVAL_Gloss_Labeling1` |
| 5 | `Gloss2` | `Gloss_Labeling2` | `cfg5_GT_Gloss_Labeling2` | `cfg5_PRED_Gloss_Labeling2` | `cfg5_EVAL_Gloss_Labeling2` |

---

## 15. DP Alignment — กลไกการทำงานเชิงลึก

หัวใจของ Phase 5 คือ **Monotonic Token-to-Segment DP** ที่จับคู่ token (จาก Gloss/CC) กับ SIGN segments (จาก SEA E4s-1) บนแกนเวลา code อยู่ที่ [example_alignment/align_gloss_labels.py:212](../example_alignment/align_gloss_labels.py#L212) (`monotonic_token_dp`) และ [example_alignment/align_gloss_labels.py:280](../example_alignment/align_gloss_labels.py#L280) (`align_one_sentence`)

### 15.1 ภาพรวม — Inputs / Outputs

```text
INPUTS
─────────────────────────────────────────────────────────────
tokens         : list[str]      — N คำของประโยค (เช่น ["สวัสดี"] หรือ ["สบาย", "ดี"])
token_embs     : (T, 768)       — SignCLIP text embedding ของแต่ละ token
sign_embs      : (K, 768)       — SignCLIP pose embedding ของ K segments
seg_starts/ends: (K,) seconds   — เวลาเริ่ม-จบของแต่ละ SIGN segment
seg_mids       : (K,)           — เวลากึ่งกลางของแต่ละ segment (ใช้ filter window)
sent.start/end : float          — ช่วงเวลาของประโยคในวิดีโอ

OUTPUTS
─────────────────────────────────────────────────────────────
ranges         : list[(k_start, k_end)] ยาว T
                 — แต่ละ token i ได้ช่วง segment ตั้งแต่ k_start[i] ถึง k_end[i]
predictions    : list[dict] ยาว T
                 — { start, end, token, token_idx, score, fallback }
```

จุดสำคัญ: aligner ไม่ได้จับ "1 token = 1 segment" — มันจับ **"1 token = ช่วงต่อเนื่องของ segments [k_start..k_end]"** (consecutive run) เพื่อรองรับกรณีที่ผู้ส่งสัญลักษณ์เปลี่ยน handshape กลางคำหรือ SEA ตัด segment ละเอียดเกิน

### 15.2 Pre-processing — Candidate window + Cosine similarity

ก่อนเข้า DP, `align_one_sentence` ทำ 3 ขั้นตอน:

#### 15.2.1 คัดกรอง candidate segments ตาม temporal window

```python
cand_mask = (seg_mids >= sent.start) & (seg_mids <= sent.end)
if cand_mask.sum() < T:
    cand_mask = (seg_mids >= sent.start - window_pad) & ...
cand_idx = np.where(cand_mask)[0]   # K candidate segments
```

- คัดเฉพาะ segments ที่ midpoint อยู่ในช่วงเวลาของประโยค
- ถ้า candidate น้อยกว่า token count (`K < T`) → ขยายหน้าต่างด้วย `window_pad` (default 0.5s)
- ถ้ายังไม่พอ → **fallback** (uniform split, ดู §15.6)

#### 15.2.2 คำนวณ cosine similarity matrix `(T, K)`

```python
tn = token_embs / ||token_embs||      # row-normalise → unit vectors
cn = sign_embs[cand_idx] / ||...||
sim = tn @ cn.T                       # shape (T, K), แต่ละแถว = 1 token, แต่ละหลัก = 1 segment
```

ค่า `sim[i, j]` = ความใกล้ใน embedding space ระหว่าง token i กับ SIGN segment j

#### 15.2.3 Per-token softmax normalisation (สำคัญ!)

```python
sim = sim - sim.max(axis=1, keepdims=True)        # numerical stability
sim = exp(sim) / sum(exp(sim), axis=1)            # softmax along K
```

ทำไมต้อง softmax: ค่า cosine สูงสุดของ token แต่ละตัวอาจไม่เท่ากัน (เช่น token หายากได้ค่าสูงสุดแค่ 0.2 ขณะที่ token ทั่วไปได้ 0.6) ถ้าใช้ raw cosine ตรง ๆ DP จะลำเอียงไปจับ token ทั่วไป **หลัง softmax ทุกแถวรวม = 1** → token เปรียบเทียบกันได้แฟร์

### 15.3 DP Recurrence — กลไก scoring แบบละเอียด

State: `dp[t, j]` = ต้นทุนต่ำสุดของการ assign token แรก ๆ `t` ตัวให้ครอบครอง segments ตั้งแต่ index 0 ถึง j-1

Transition: token ที่ t กิน segments `[k-1 .. j-1]` (inclusive) → คำนวณจาก dp[t-1, k-1]

```text
dp[t, j] = min over k in [t .. j] of:
             dp[t-1, k-1]                                  # ต้นทุนสะสมก่อนหน้า
           + (-Σ sim[t-1, k-1 .. j-1])                     # similarity reward (-)
           + gap_penalty   × inter_segment_gap(k-1, j-1)   # gap penalty
           + coverage_penalty × |group_dur - target_dur|   # coverage penalty
```

#### 3 องค์ประกอบของ cost

**(A) Negative similarity term** — รางวัลของการจับคู่ที่ similarity สูง

```python
neg_sim = -(cum_sim[t-1, j] - cum_sim[t-1, k-1])
        = -Σ sim[t-1, k-1 .. j-1]
```

ใช้ cumulative sum (`cum_sim`) ทำให้คำนวณ range sum ได้ O(1) แทน O(K) → speed ทั้ง DP O(T·K²) แทน O(T·K³)

**(B) Gap penalty** — ห้ามจับ segments ที่ห่างไกลกันมา assign ให้ token เดียว

```python
gap = Σ (seg_starts[k+1..j-1] - seg_ends[k..j-2]).clip(min=0)
       # ผลรวมของช่องว่างระหว่าง consecutive segments ในกลุ่ม
cost += gap_penalty × gap
```

ตัวอย่าง: ถ้าจับ segment 3 (เวลา 1.0–1.2s) ร่วมกับ segment 5 (เวลา 3.0–3.5s) → gap = 1.8s → cost เพิ่ม 2.0 × 1.8 = 3.6 (default `gap_penalty=2.0`)

**(C) Coverage penalty** — token ไม่ควรกินช่วงเวลายาว/สั้นเกินกว่าเฉลี่ย

```python
target_dur = sentence_dur / T          # ความยาวเฉลี่ยที่ token "ควร" กิน
grp_dur    = seg_ends[j-1] - seg_starts[k-1]
cov        = |grp_dur - target_dur|
cost += coverage_penalty × cov
```

ตัวอย่าง: ประโยค 5s, T=2 → target = 2.5s/token ถ้า DP อยาก assign segments [0.0–4.5s] ให้ token 1 ตัวเดียว (grp_dur=4.5s) → cov=2.0 → cost เพิ่ม 0.5 × 2.0 = 1.0 (default `coverage_penalty=0.5`)

**สรุป tradeoff 3 ทาง:**

- **similarity** อยาก grab segment ที่ embedding ตรงกัน
- **gap** ป้องกันการจับ segments ที่กระจัดกระจาย
- **coverage** ป้องกัน token เดียวกินยาวเกินไป (หรือสั้นเกินไป) จากค่ากลาง

### 15.4 Monotonicity Constraint — เหตุผลที่ DP นี้ correct

จาก loop structure:

```python
for t in range(1, T+1):
    j_max = K - (T - t)                  # เหลือ segments ให้ tokens ถัดไปเสมอ
    for j in range(t, j_max + 1):        # j >= t: ต้องมี ≥t segments ใช้ไปแล้ว
        for k in range(t, j + 1):        # k <= j: range ต้องไม่ว่าง
            ...
```

- **`j >= t`** → token แรก ๆ t ตัว ต้องครอบครอง segments อย่างน้อย t อัน
- **`j_max = K - (T-t)`** → ต้องเหลือ segments อย่างน้อย (T-t) อันให้ tokens ถัดไป
- **`k <= j`** → range ของ token ปัจจุบันต้องไม่ว่าง (k=j คือกิน 1 segment เท่านั้น)

ผลลัพธ์: **ranges ของ token i กับ token i+1 ไม่ overlap และเรียงตามเวลาเสมอ** — สอดคล้องกับธรรมชาติของภาษามือที่ token สื่อต่อกันเป็นลำดับเวลา

### 15.5 Backtracking — กู้ ranges ออกจาก dp/prev

หลังเติม `dp` เต็ม table ขั้นตอนสุดท้าย:

```python
# 1) เลือก j* ใน [T..K] ที่ทำให้ dp[T, j*] ต่ำสุด → token สุดท้ายจบที่ segment j*-1
j_star = argmin(dp[T, T..K]) + T

# 2) Backtrack ผ่าน prev[t, j] เพื่อกู้ k ของแต่ละ token
ranges = []
j = j_star
for t in range(T, 0, -1):
    k = prev[t, j]
    ranges.append((k-1, j-1))          # token t-1 กิน segments [k-1 .. j-1]
    j = k - 1                          # token ก่อนหน้าจบที่ k-2
ranges.reverse()
```

`prev[t, j]` เก็บ `k*` ที่ทำให้ `dp[t, j]` ต่ำสุดในขั้นตอน forward DP ใช้ trace กลับว่า token i ตัวที่ t กิน range ไหน

จาก `ranges`, predictions ของ token i = ช่วงเวลา **[seg_starts[k_start], seg_ends[k_end]]** — ทำให้ start/end ของ prediction เป็นเวลาจริงของ SIGN segments ไม่ใช่เวลาเฉลี่ย

### 15.6 Fallback Path — เมื่อ DP ทำไม่ได้

DP ตกเป็น fallback ในเงื่อนไขเหล่านี้:

1. `K < T` แม้ขยาย window แล้ว → segments ไม่พอจับคู่ทุก token
2. `sign_embs` หาย / load ไม่ได้
3. SIGN tier ของคลิปว่าง (เกิดกับ 57 clips ในรอบนี้)
4. `dp[T, j*] = ∞` → ไม่มี path ที่ valid (เกือบไม่เคยเกิดถ้าผ่าน condition 1)

Fallback strategy: **Uniform split**

```python
step = sentence_dur / T
for t in 0..T-1:
    pred.start = sent.start + t * step
    pred.end   = sent.start + (t+1) * step
    pred.fallback = "uniform_<reason>"        # e.g., "uniform_emb_missing"
```

แบ่งช่วงเวลาประโยคออกเป็น T ส่วนเท่ากันให้ token เรียงไปตามลำดับ ไม่ optimal แต่ดีกว่าไม่ปล่อย empty prediction

### 15.7 ตัวอย่างเชิงตัวเลข — Clip 9 ("สบายดี")

```text
Input
─────
sent.tokens   = ["สบาย", "ดี"]              T = 2
sent.start/end = 0.0 / 4.0 s              sentence_dur = 4.0s
target_dur    = 4.0 / 2 = 2.0 s/token

SIGN segments (after window filter) — สมมุติ K = 4
─────────────────────────────────────────────────
seg 0 : 1.50–1.65 s  (similar to "สบาย": sim=0.6, "ดี": sim=0.1)
seg 1 : 1.70–1.90 s  ("สบาย": 0.5,        "ดี": 0.2)
seg 2 : 2.20–2.50 s  ("สบาย": 0.1,        "ดี": 0.4)
seg 3 : 2.60–2.80 s  ("สบาย": 0.05,       "ดี": 0.5)

After per-token softmax (T=2, K=4)
sim[0] = [0.41, 0.37, 0.12, 0.10]    ← "สบาย" prefers seg 0–1
sim[1] = [0.16, 0.20, 0.30, 0.34]    ← "ดี" prefers seg 2–3

DP fills table dp[2, 4] then picks j* minimising dp[2, 2..4]
Best path (สมมุติ): "สบาย" → seg [0..1], "ดี" → seg [2..3]
                    → cost = -(0.41+0.37) - (0.30+0.34)
                            + 2.0 × gap(1→2: 0.30s) + 0.5 × |cov| ≈ -0.82 ...

Backtrack ranges = [(0,1), (2,3)]

Predictions:
─────────────
สบาย : start = seg_starts[0] = 1.50, end = seg_ends[1] = 1.90  → (1.50, 1.90)
ดี   : start = seg_starts[2] = 2.20, end = seg_ends[3] = 2.80  → (2.20, 2.80)
```

เทียบจริงใน `eval_config3.csv` clip 9:

- pred "สบาย" = [1.583, 1.616], pred "ดี" = [1.716, 2.65] (น้อย segment กว่าตัวอย่าง แต่ pattern เดียวกัน)
- GT "สบาย" = [0.0, 2.216], GT "ดี" = [2.216, 4.0] (annotation ยืดเต็มประโยค → IoU ต่ำ ตามที่อธิบายใน §14.6)

### 15.8 Complexity & Performance

| ส่วน | Complexity | หมายเหตุ |
|---|---|---|
| Similarity matrix | O(T · K · D) | D = 768 (SignCLIP) |
| Cumulative similarity | O(T · K) | ทำครั้งเดียวต่อประโยค |
| DP forward | O(T · K²) | inner loop over k, j |
| Backtrack | O(T) | linear |
| **รวมต่อประโยค (T=2, K=5)** | **~50 ops** | < 1 ms ต่อ clip |
| **รวมทั้ง dataset (1,132 clips × 5 configs)** | | ~57 นาที (จริง: Phase 5 ~60 นาที ตาม §6.1) |

vector ops ทั้งหมดเป็น NumPy in-process — ไม่มี subprocess overhead, ไม่ต้อง JIT compile

### 15.9 Hyperparameters — สรุปและพฤติกรรม

| Param | Default | ผลถ้าเพิ่ม | ผลถ้าลด |
|---|---:|---|---|
| `gap_penalty` | 2.0 | อยากให้ token กิน segments ติดกัน → range สั้นลง | ยอมรับ scattered segments → range กระจาย |
| `coverage_penalty` | 0.5 | บังคับ range ใกล้ target_dur → ปรับแต่ง IoU บน narrow GT | ยอมให้ range สั้น/ยาวต่างกันมาก |
| `window_pad` | 0.5s | candidate มากขึ้น → DP มีตัวเลือก แต่ noise เพิ่ม | เสี่ยง fallback ถ้าไม่มี segments ใกล้ |

ใน Q4 ของ §11 ตัดสินใจใช้ default ทั้งหมดสำหรับ baseline run ผลลัพธ์: Config #1 mIoU 0.5928 (เทียบ 04.mp4 ที่ 0.4901) ดีกว่า → **default parameters ใช้งานได้ดีกับ ForcedAlignment dataset โดยไม่ต้อง tune**

### 15.10 จุดอ่อนเชิงทฤษฎี (Limitations)

1. **Hard monotonicity** — ถ้าผู้ส่งสัญลักษณ์สลับลำดับคำ DP จะ assign ผิด (เกิดยากในชุดข้อมูลนี้เพราะคลิปสั้นและเป็นประโยคเดียว)
2. **Per-token softmax ทำให้ similarity comparison ระหว่าง tokens เปรียบเทียบไม่ตรง** — ดีในแง่ fairness แต่เสีย information เรื่อง absolute similarity
3. **Fixed `target_dur = sentence_dur / T`** — สมมุติทุก token ยาวเท่ากัน ผิดเมื่อบาง token เป็น "sil" หรือ stretch word เช่น "ขอบ----คุณ" (Config 4/5 ได้รับผลกระทบเพราะ sil tokens สั้นกว่าค่าเฉลี่ยมาก)
4. **ไม่มี likelihood score per-segment-per-token แบบ HMM** — sim เป็น cosine ของ embedding เฉพาะ ไม่ใช่ posterior ของ HMM → ไม่ robust ต่อ embedding noise
5. **Fallback ค่อนข้าง crude** — uniform split ไม่ดู pose features ของ clip เลย ถ้าจะปรับปรุงในรอบหน้า แนะนำใช้ pose-based heuristic (เช่น hand motion peaks) แทน

---
