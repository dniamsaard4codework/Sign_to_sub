# แผนการดำเนินงาน — Forced Alignment Task 2 บน ForcedAlignment Dataset

> **วันที่จัดทำ:** 17 พฤษภาคม 2569
> **อัปเดตล่าสุด:** 18 พฤษภาคม 2569 (resolved all open questions, repaired EAF media paths, verified all preconditions)
> **ผู้รับผิดชอบ:** dniamsaard4codework
> **สถานะ:** ✅ แผนสมบูรณ์พร้อม implement — preconditions ทุกข้อ verified (EAF↔video 1132/1132 OK, MEDIA_DESCRIPTOR paths 1132/1132 clean, 8 tiers/file × 1132 files OK)

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
- [ ] **ตรวจสอบ disk space** — ต้องการ ~4 GB free สำหรับ output (3.5 GB poses + .npy + seg EAF + predictions)
- [ ] **ตรวจสอบ `videos_to_poses` batch directory mode** — ทดลองส่งทั้ง subfolder (185–207 clips) ใน 1 call ก่อน → ถ้า model init ครั้งเดียว Phase 2 ลดเหลือ ~3 ชม.
- [ ] **ทดสอบ in-process DP** บน 3 clips ก่อน — verify ~0.6 s/clip/config (Phase 5 calculation)

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
| Implementation | ~2 วัน (เขียนโค้ด 13–19 ชั่วโมง active) |
| Timeline รวม | **~3 วัน** (วันที่ 1: โค้ด+smoke test, คืนที่ 1: compute, วันที่ 2: วิเคราะห์+รายงาน) |
| Blocker ใหญ่สุด | Pose extraction ช้า — ทดสอบ `videos_to_poses` directory mode ก่อนรัน full pipeline |
| Open questions | ✅ **resolved ทั้งหมด** (Q1–Q3 — verified จาก data; Q4–Q5 — default settings + sequential in-process) |
| สถานะ | ✅ **แผนสมบูรณ์ พร้อม implement** |
