# Big Progress — SEA Project Consolidated Report

> **SEA** (Segment, Embed, and Align) — ระบบจัดตำแหน่งคำบรรยาย / annotation ภาษามือไทย (TSL)
> บน pipeline ที่พัฒนาต่อจากงานวิจัย [arXiv:2512.08094](https://arxiv.org/abs/2512.08094) — Jiang et al., 2025
>
> เอกสารนี้รวมทุก progress report (20 เม.ย. → 24 พ.ค. 2569) ไว้ในที่เดียว
> โฟกัสที่ **pipeline, เหตุผลของแต่ละ experiment, และผลลัพธ์** ไม่แบ่งตามวัน
> รวม ForcedAlignment scale-up (§10) + DP alignment deep-dive (§10.12) + final report deliverables (§10.13)

---

## สารบัญ

1. [ภาพรวมโครงการและเป้าหมาย](#1-ภาพรวมโครงการและเป้าหมาย)
2. [ข้อมูลนำเข้า](#2-ข้อมูลนำเข้า)
   - 2.6 [Key Concepts — อ่านก่อนเข้า Pipeline](#26-key-concepts--อ่านก่อนเข้า-pipeline)
3. [Environment Setup](#3-environment-setup)
4. [Pipeline Architecture](#4-pipeline-architecture)
   - 4a. [S — Segment: ตรวจจับท่ามือแต่ละท่า](#4a-s--segment-ตรวจจับท่ามือแต่ละท่า-seasegmentationpy)
   - 4b. [E — Embed: แปลงท่ามือและข้อความเป็น Vector](#4b-e--embed-แปลงท่ามือและข้อความเป็น-vector-signclip)
   - 4c. [A — Align: จับคู่ Subtitle กับ Sign Segment (DP)](#4c-a--align-จับคู่-subtitle-กับ-sign-segment-dp)
   - 4d. [Pipeline Reference — คำสั่งทุกขั้นตอน](#4d-pipeline-reference--คำสั่งทุกขั้นตอน-copyable-commands)
   - 4e. [Evaluation Metrics — อ่านก่อนดูผลลัพธ์](#4e-evaluation-metrics--อ่านก่อนดูผลลัพธ์)
5. [Task 1 — Subtitle Alignment: Experiments และผลลัพธ์](#5-task-1--subtitle-alignment-experiments-และผลลัพธ์)
6. [Task 2 — Gloss Labeling: Experiments และผลลัพธ์](#6-task-2--gloss-labeling-experiments-และผลลัพธ์)
7. [เครื่องมือเสริมที่พัฒนาขึ้น](#7-เครื่องมือเสริมที่พัฒนาขึ้น)
8. [ปัญหาที่พบและวิธีแก้ไข](#8-ปัญหาที่พบและวิธีแก้ไข)
9. [ผลสรุปและ Recommendation](#9-ผลสรุปและ-recommendation)
10. [ForcedAlignment Dataset — Task 2 Scale-up](#10-forcedalignment-dataset--task-2-scale-up-เริ่มวางแผน-17-พค-เสร็จสมบูรณ์-24-พค-2569)
11. [Fine-tuning Opportunities — จะทำให้ดีขึ้นได้อย่างไร](#11-fine-tuning-opportunities--จะทำให้ดีขึ้นได้อย่างไร)
12. [อ้างอิง](#12-อ้างอิง)

---

## 1. ภาพรวมโครงการและเป้าหมาย

### 1.1 บริบท — ทำไมโครงการนี้ถึงมีอยู่

**ภาษามือไทย (Thai Sign Language — TSL)** เป็นภาษาหลักของผู้พิการทางการได้ยินในประเทศไทย วิดีโอการสอนที่ใช้ผู้แปลภาษามือมักมีคำบรรยาย (subtitle) แนบมาด้วย แต่ subtitle เหล่านี้ถูกสร้างจาก **เสียงพูดของผู้บรรยาย** ไม่ใช่จากท่ามือของผู้แปล

ปัญหาคือ: **ผู้แปลภาษามือต้องได้ยินเสียงก่อน แล้วจึงค่อยแปลเป็นท่ามือ** — จึงมีความล่าช้าตามธรรมชาติ 0.5–2 วินาที ทำให้ subtitle ที่ sync กับเสียงพูดนั้น **ไม่ตรงกับจังหวะที่ผู้แปลมือแสดงท่าจริง**

#### ภาพแสดง Sign Delay Problem

```text
เสียงพูด:   ══[  "เด็กเรียน"  ]════════════════════════════════►  เวลา
               35.0s   36.5s

ท่ามือ:     ══════════════════════[ "เด็กเรียน" ]════════════════►  เวลา
                                   36.2s  37.5s
                                   ↑
                              ล่าช้า ~1.2s

CC_Input (timestamp จากเสียง):  start=35.0s, end=36.5s   ← ผิด
CC_Aligned (ที่ต้องการ):         start=36.2s, end=37.5s   ← ถูก
```

นักวิจัยต้องนั่ง align timestamp เหล่านี้ด้วยมือทีละ cue — ซึ่งสำหรับวิดีโอ 11 นาทีที่มี 119 cues ใช้เวลาหลายชั่วโมง และถ้ามีหลายร้อยวิดีโอในชุดข้อมูล ForcedAlignment (1,132 คลิป) ก็เป็นไปไม่ได้เลย

**งานนี้จึงสร้างระบบอัตโนมัติ** เพื่อทำสิ่งที่นักวิจัยทำด้วยมือให้แทน

---

### 1.2 SEA คืออะไร

**SEA** ย่อมาจาก **S**egment → **E**mbed → **A**lign ซึ่งเป็น 3 ขั้นตอนหลักของระบบ:

```text
┌─────────────────────────────────────────────────────────────────────┐
│  S — Segment                                                        │
│  ดู pose ของผู้แปลมือ → ตรวจจับว่าช่วงเวลาไหน "กำลังแสดงท่ามือ"      │
│  Output: 2,780 sign segments (แต่ละ segment = 1 ท่ามือ)             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  E — Embed                                                          │
│  แปลงทั้งท่ามือ (video) และข้อความ (text) ให้เป็น vector 768 มิติ    │
│  ใช้โมเดล SignCLIP เพื่อให้ "ท่ามือ เด็ก" กับ text "เด็ก"           │
│  อยู่ใกล้กันใน embedding space                                       │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│  A — Align                                                          │
│  ใช้ Dynamic Programming (DP) จับคู่ subtitle แต่ละ cue              │
│  กับกลุ่ม sign segments ที่ "ดูเหมือน" กันมากที่สุดในเชิง semantic    │
│  Output: timestamp ใหม่สำหรับแต่ละ subtitle cue                     │
└─────────────────────────────────────────────────────────────────────┘
```

SEA ถูกพัฒนาโดย Jiang et al. (EMNLP 2024, [arXiv:2512.08094](https://arxiv.org/abs/2512.08094)) สำหรับ **British Sign Language (BSL)** เป็นหลัก งานนี้นำมา **cross-lingual transfer** กับ TSL — คือใช้โมเดลที่ train ไว้บน BSL/Multilingual data โดยตรง โดยไม่ต้อง retrain ใหม่ แล้วทดสอบว่า generalize ข้ามภาษามือได้หรือเปล่า

---

### 1.3 สองงานหลัก (Two Main Tasks)

#### Task 1 — Subtitle Alignment (ระดับประโยค)

**เป้าหมาย:** รับ subtitle ที่ timestamp sync กับเสียง → คืน subtitle ที่ timestamp ตรงกับท่ามือ

```text
INPUT  (CC_Input):   "เด็กเรียน"  start=35.0s  end=36.5s
                     "ครูสอน"     start=37.0s  end=38.2s
                     ... (119 cues ทั้งหมด)

OUTPUT (CC_Aligned): "เด็กเรียน"  start=36.2s  end=37.5s  ← ขยับ +1.2s
                     "ครูสอน"     start=38.1s  end=39.4s  ← ขยับ +1.1s
                     ...
```

Ground truth (CC_Aligned) คือ timestamp ที่นักวิจัยทำด้วยมือ — ใช้ประเมินว่าระบบอัตโนมัติใกล้เคียงแค่ไหน

**ผลดีที่สุด (experiment C_MULTI, 119/119 cues):**

| Metric | ความหมาย | ค่า |
| --- | --- | ---: |
| Mean start offset | ค่าเฉลี่ย start ที่ขยับไปจาก GT (ลบ = ขยับก่อน GT) | −0.16 s |
| % within ±1 s | สัดส่วน cues ที่ start อยู่ห่างจาก GT ≤ 1 วินาที | 73.9% |
| % within ±2 s | ห่างจาก GT ≤ 2 วินาที | 95.0% |
| % within ±3 s | ห่างจาก GT ≤ 3 วินาที | **100%** |
| F1 @ 0.50 IoU | คะแนน F1 เมื่อนับ prediction ที่ overlap GT ≥ 50% ว่า "ถูก" | 88.2% |
| Overlap rate | สัดส่วน subtitle cues ที่ทับซ้อนกัน (หลัง fix) | **0%** |

#### Task 2 — Gloss Labeling (ระดับท่ามือเดี่ยว)

**เป้าหมาย:** รับ Gloss ระดับประโยค → แยกเป็น annotation รายท่ามือแต่ละท่า

> **Gloss คืออะไร?** Gloss คือระบบ notation สำหรับภาษามือ — เขียนแต่ละท่ามือด้วยคำศัพท์สั้นๆ เช่น "สวัสดี ผายมือ เด็ก เรียน" แทนที่จะเขียนเป็นประโยคภาษาไทยพูดปกติ เป็น "รหัสท่ามือ" ที่ผู้เชี่ยวชาญสร้างขึ้น

```text
INPUT (Gloss tier, sentence level):
  ประโยค 5:  start=34.0s  end=38.5s  text="ผายมือ เด็ก เรียน"
                                           (3 tokens = 3 ท่ามือ)

OUTPUT (Gloss Labeling, sign-level):
  ท่า 1:  start=34.1s  end=34.9s  label="ผายมือ"
  ท่า 2:  start=35.3s  end=36.0s  label="เด็ก"
  ท่า 3:  start=36.4s  end=37.2s  label="เรียน"
```

Ground truth (Gloss Labeling tier) มี 852 entries ที่นักวิจัยทำด้วยมือ

**ผลดีที่สุด (Gloss whole-video, 852 predictions):**

| Metric | ความหมาย | ค่า |
| --- | --- | ---: |
| Mean IoU | ความทับซ้อนเฉลี่ยระหว่าง prediction กับ GT (0=ไม่ทับ, 1=ตรงทุก ms) | 0.4901 |
| % IoU ≥ 0.5 | สัดส่วนที่ทับซ้อนกัน ≥ 50% | 48.4% |
| % IoU ≥ 0.3 | สัดส่วนที่ทับซ้อนกัน ≥ 30% (sanity bar ≈ 1/7 = 14%) | 77.0% |
| % any overlap | สัดส่วนที่ทับซ้อน GT อย่างน้อยบางส่วน | 97.5% |
| Fallback uniform | ประโยคที่ระบบล้มเหลวจนต้อง fallback เป็น uniform split | 0 / 119 |

---

## 2. ข้อมูลนำเข้า

### 2.1 ไฟล์หลัก

วิดีโอตัวอย่าง: **"การเปรียบเทียบและเรียงลำดับ"** (11.07 นาที, 1920×1080, 60fps)

| ไฟล์ | คำอธิบาย | ขนาด |
| --- | --- | --- |
| `04.mp4` | วิดีโอต้นฉบับ | ~80 MB |
| `04.pose` | Skeleton pose จาก MediaPipe Holistic (543 landmarks × ทุกเฟรม) | 358 MB |
| `Test.eaf` | ELAN annotation file — ที่เก็บ ground truth และ input ทั้งหมด | — |

### 2.2 ELAN / EAF คืออะไร

**ELAN** (EUDICO Linguistic Annotator) คือโปรแกรม annotation วิดีโอที่นักภาษาศาสตร์ใช้กันทั่วโลก ไฟล์ที่ ELAN สร้างมีนามสกุล `.eaf` (ELAN Annotation Format) ซึ่งเป็น XML ที่เก็บ annotation หลายชั้น (เรียกว่า **tier**) พร้อมกับวิดีโอเดียวกัน

**Tier** คือ "ชั้น annotation" หนึ่งชั้น แต่ละ tier มีชื่อและเก็บ annotation ประเภทเดียวกัน เช่น tier `CC_Input` เก็บ subtitle cues, tier `Gloss` เก็บ gloss tokens

```text
Test.eaf (ไฟล์ XML เดียว)
  ├── Tier: CC              ← 172 subtitle entries (timestamp จากเสียง, ดิบ)
  ├── Tier: CC_Input        ← 119 subtitle entries (curated) — INPUT Task 1
  ├── Tier: CC_Aligned      ← 119 entries ที่นักวิจัย align ด้วยมือ — GT Task 1
  ├── Tier: Gloss           ← 119 gloss sentences — INPUT Task 2 (ดีที่สุด)
  ├── Tier: Gloss_Input     ← 119 gloss sentences (curated) — INPUT Task 2 (ด้อยกว่า)
  └── Tier: Gloss Labeling  ← 852 sign-level entries — GT Task 2
```

แต่ละ annotation entry ใน tier มีรูปแบบ: `(start_time, end_time, text)`

### 2.3 VTT / Cue คืออะไร

**WebVTT** (Web Video Text Tracks) คือ format ของ subtitle file ที่ใช้บนเว็บ ไฟล์ `.vtt` เก็บ "cues" — แต่ละ cue มี timestamp และ text:

```text
WEBVTT

00:00:35.000 --> 00:00:36.500
เด็กเรียน

00:00:37.000 --> 00:00:38.200
ครูสอน
```

ใน pipeline นี้ EAF tier จะถูก export เป็น VTT ก่อน เพื่อให้ SignCLIP และ align.py อ่านได้

### 2.4 Tiers ทั้งหมดและบทบาท

| Tier | จำนวน | บทบาท | หมายเหตุ |
| --- | ---: | --- | --- |
| **CC** | 172 | คำบรรยายดิบ (timestamp จากเสียงพูด) | ไม่ใช้เป็น input อีกแล้ว |
| **CC_Input** | 119 | คำบรรยาย curated | **input Task 1** |
| **CC_Aligned** | 119 | align ด้วยมือโดยนักวิจัย | **ground truth Task 1** |
| **Gloss** | 119 | Gloss tier ดั้งเดิม (852 tokens) | **input Task 2 (ดีที่สุด)** — token count ตรง GT |
| **Gloss_Input** | 119 | Gloss tier curated (889 tokens) | **input Task 2 (ด้อยกว่า)** — +37 tokens ส่วนเกิน |
| **Gloss Labeling** | 852 | annotation รายท่ามือแต่ละท่า | **ground truth Task 2** |

#### ทำไม CC มี 172 cue แต่ CC_Input มีแค่ 119

`CC` tier คือ subtitle ดิบที่ได้จากระบบ Speech-to-Text ซึ่งมักแยก/รวมประโยคผิดพลาด นักวิจัยทำการ curate ใหม่ได้ `CC_Input` (119 cues) ที่สะอาดกว่า การเปลี่ยนมาใช้ CC_Input ทำให้ evaluation ครอบคลุม **119/119 cues (100%)** แทนที่จะเป็น 69/172 (58%) แบบเดิม

#### ทำไม Gloss มี 852 tokens แต่ Gloss_Input มี 889

Gloss tier เดิมมี 852 tokens ซึ่งตรงกับ Gloss Labeling GT **พอดี** (1:1) ขณะที่ Gloss_Input ที่ curated ใหม่มี 889 tokens — มี 37 tokens ส่วนเกินเนื่องจาก re-tokenization ที่แตกต่างกัน ทำให้ alignment ผิดพลาดได้ง่ายกว่า

### 2.5 ตัวอย่าง Concrete: Input และ Expected Output

```text
ตัวอย่าง cue เดียวตลอด pipeline:

── INPUT ──────────────────────────────────────────────────────────────
CC_Input cue #42:
  start = 4:23.150  (263.15s)
  end   = 4:25.200  (265.20s)
  text  = "การเรียงลำดับจากมากไปน้อย"

── SEGMENT (S) ──────────────────────────────────────────────────────
SIGN segments ใน window รอบๆ 263–265s (ตัวอย่าง):
  seg_780: 263.90s – 264.35s
  seg_781: 264.50s – 264.85s
  seg_782: 265.10s – 265.60s
  seg_783: 265.75s – 266.20s

── EMBED (E) ────────────────────────────────────────────────────────
text "การเรียงลำดับจากมากไปน้อย"  → vector [0.12, −0.34, ..., 0.87]  (768 dims)
seg_780 pose data                 → vector [0.08, −0.31, ..., 0.91]  (768 dims)
cosine similarity = 0.94  ← สูง = "ท่ามือนี้ match กับข้อความ"

── ALIGN (A, DP) ────────────────────────────────────────────────────
DP เลือก group = {seg_780, seg_781, seg_782}  (similarity สูงสุด)
  → start_new = seg_780.start = 263.90s
  → end_new   = seg_782.end   = 265.60s

── OUTPUT ───────────────────────────────────────────────────────────
CC_Aligned prediction:
  start = 4:23.900  (263.90s)   ← ขยับ +0.75s จาก input
  end   = 4:25.600  (265.60s)
  text  = "การเรียงลำดับจากมากไปน้อย"
```

---

## 2.6 Key Concepts — อ่านก่อนเข้า Pipeline

> ถ้าไม่คุ้นกับ AI/ML concepts เหล่านี้ อ่านส่วนนี้ก่อน จะทำให้เข้าใจ Pipeline ได้ดีขึ้นมาก

### Embedding / Vector Space

**Embedding** คือการแปลงข้อมูล (เช่น รูปภาพ หรือ ข้อความ) ให้เป็น **list ของตัวเลข (vector)** ซึ่งแทนความหมายของข้อมูลนั้น ในโปรเจกต์นี้:

```text
Sign embedding:  ท่ามือ "เด็ก"  →  [0.12, -0.34, 0.71, ..., 0.45]  (768 ตัวเลข)
Text embedding:  คำว่า "เด็ก"   →  [0.11, -0.33, 0.70, ..., 0.44]  (768 ตัวเลข)

สังเกต: ตัวเลขใกล้กันมาก → "อยู่ใกล้กันใน embedding space"
```

เราใช้ embedding space นี้เพื่อวัดว่า "ท่ามือนี้มีความหมายตรงกับข้อความนั้นมากแค่ไหน"

### Cosine Similarity

วิธีวัด "ความใกล้" ระหว่าง 2 vectors ใน embedding space ค่าอยู่ระหว่าง -1 ถึง 1:

- **1.0** = เหมือนกันเป๊ะ
- **0.9+** = ความหมายใกล้เคียงกันมาก
- **~0.0** = ไม่มีความสัมพันธ์กัน
- **-1.0** = ตรงข้ามกัน

```text
cosine_sim(ท่ามือ "เด็ก", text "เด็ก")     = 0.94  ← สูง (match)
cosine_sim(ท่ามือ "เด็ก", text "ครูสอน")   = 0.21  ← ต่ำ (ไม่ match)
```

### Dynamic Programming (DP)

**Dynamic Programming** คือเทคนิคการแก้ปัญหา optimization ที่แบ่งปัญหาใหญ่ออกเป็นปัญหาย่อย แล้วเก็บคำตอบย่อยไว้เพื่อนำมาใช้ซ้ำ

ในโปรเจกต์นี้: DP แก้ปัญหา "จับคู่ 119 subtitle กับ 2,780 sign segments อย่างไรให้ cost รวมต่ำสุด โดย mapping ต้องเรียงตามลำดับเวลา"

```text
แทนที่จะลองทุก combination (2780^119 = มากเกินไป)
DP ทำแบบ bottom-up: เริ่มจาก cue 1 → cue 2 → ... → cue 119
แต่ละ cue จำค่า "best assignment ที่เป็นไปได้ทั้งหมด" ไว้
→ ใช้เวลาเพียง O(M × W²) ≈ 190,000 operations
```

### IoU (Intersection over Union)

วัดว่า 2 ช่วงเวลาทับซ้อนกันแค่ไหน ใช้สำหรับ Task 2 evaluation:

```text
Prediction:   [===34.1s===34.9s===]
Ground Truth: [=====34.0s====34.8s=====]

Intersection (ส่วนที่ทับกัน) = 34.1s → 34.8s = 0.7s
Union (ส่วนรวม) = 34.0s → 34.9s = 0.9s

IoU = 0.7 / 0.9 = 0.78  ← ทับซ้อน 78% = ดี
```

ถ้า IoU = 1.0 → prediction ตรงกับ GT เป๊ะ
ถ้า IoU = 0.5 → ทับซ้อน 50% → ยอมรับได้
ถ้า IoU = 0 → ไม่ทับซ้อนเลย → ผิดทั้งหมด

### Cross-lingual Transfer

SEA ถูก train บน BSL (British Sign Language) แต่เราใช้กับ TSL (Thai Sign Language) โดยไม่ retrain — นี่เรียกว่า **cross-lingual transfer** หรือ **zero-shot transfer**

ได้ผลได้เพราะ:

1. SignCLIP multilingual model train บน dataset หลายภาษามือรวมกัน → มี "ความเข้าใจ" sign language ทั่วไป
2. ท่ามือพื้นฐานบางท่า (เช่น pointing, waving) คล้ายกันข้ามภาษา
3. DP alignment ใช้ทั้ง timing และ semantic similarity → ไม่ได้ขึ้นกับภาษาเดียว

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
cd <path\to\Sign_to_sub>
python3.11 -m venv venv
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

### 4a. S — Segment: ตรวจจับท่ามือแต่ละท่า (SEA/segmentation.py)

#### ขั้นตอนก่อน Segmentation — Pose Estimation

**เครื่องมือ:** `videos_to_poses` (จาก `pose-format`) เรียก **MediaPipe Holistic**

| Parameter | ค่า | ความหมาย |
| --- | --- | --- |
| `--format` | `mediapipe` | ใช้ Google MediaPipe Holistic |
| `model_complexity` | 2 | ความแม่นยำสูงสุด |
| `smooth_landmarks` | false | ไม่ smooth — ต้องการ boundary ที่คมชัด |
| `refine_face_landmarks` | true | ตรวจจับใบหน้าละเอียด |

**Output:** `04.pose` — 358 MB (binary)
Landmark ทั้งหมด **543 จุดต่อเฟรม** (มือ 21×2, ใบหน้า 468, ลำตัว 33) ที่ 60 fps

#### หลักการทำงานของ E4s-1 Segmenter

โมเดล **E4s-1** (EMNLP 2023 Findings) train บน BOBSL dataset (BSL) — ใช้เป็น binary classifier ต่อเฟรม:

```text
04.pose (543 landmarks × ทุกเฟรม)
        │
        ▼
  E4s-1 GRU model — sliding window forward pass
        │
        ▼  probability per frame: "กำลังแสดงท่ามือ" vs "หยุดพัก/เปลี่ยนท่า"
        │
  Threshold 2 ค่า:
    sign-b-threshold = 30  →  boundary candidate (จุดเปลี่ยนท่ามือ)
    sign-o-threshold = 50  →  onset candidate (จุดเริ่มท่ามือใหม่)
        │
        ▼
  Non-max suppression → final annotation
        │
  Output EAF:
    SIGN tier     : 2,780 segments  (แต่ละ segment = 1 ท่ามือ)
    SENTENCE tier : 418 segments    (กลุ่มท่ามือที่เป็นประโยค)
```

**ผลกระทบของ threshold:**

| threshold | ค่าสูง | ค่าต่ำ |
| --- | --- | --- |
| `sign-b-threshold` | segment น้อยลง (merge ท่าเล็กเข้ากัน) | segment ถี่ขึ้น |
| `sign-o-threshold` | จับเฉพาะท่าชัดเจน | จับท่ามือเบาๆ ด้วย |

> ⚠️ **Windows fix:** `segmentation.py` line 71 เปลี่ยนจาก `subprocess.run(cmd, shell=True)` เป็น `subprocess.run(shlex.split(cmd), shell=False)` เพราะ `shlex.quote()` ใส่ single quote ที่ `cmd.exe` ไม่รู้จัก

---

### 4b. E — Embed: แปลงท่ามือและข้อความเป็น Vector (SignCLIP)

#### SignCLIP คืออะไร

**SignCLIP** (EMNLP 2024) เป็นโมเดล multimodal ที่ project ทั้ง sign language และ text ลงใน embedding space **768 มิติ** เดียวกัน โดย:

- รับ **pose data** ของท่ามือ → **sign embedding** (768-dim)
- รับ **text** → **text embedding** (768-dim)
- ท่ามือและข้อความที่มีความหมายเดียวกัน → อยู่ **ใกล้กัน** ใน embedding space (cosine similarity สูง)

มี 3 variant ที่ train บน dataset ต่างกัน:

| Model | Checkpoint | Train data | Suitability สำหรับ TSL |
| --- | --- | --- | --- |
| `bsl` | `bobsl_finetune_checkpoint_best.pt` | BSL (British SL) | ปานกลาง |
| `multilingual` | `baseline_temporal_checkpoint_best.pt` | หลายภาษามือ | **ดีที่สุด** ✓ |
| `asl` | `asl_finetune_checkpoint_best.pt` | ASL (American SL) | ด้อยกว่า |

#### โหมด Sign Embedding (`--feature_type sign`)

`extract_episode_features.py` ทำงานดังนี้:

```text
Input: 04.pose + segmentation EAF (SIGN tier, 2,780 segments)
        │
  For each SIGN segment [start, end]:
    ├─► slice pose frames → normalize ตาม shoulder reference
    ├─► zero-pad / trim → 32 frames คงที่
    ├─► forward pass SignCLIP visual encoder
    └─► ดึง pooled output → 768-dim vector
        │
Output: (2780, 768) numpy array
  segmentation_embedding/{bsl|multi|asl}/04.npy
```

#### โหมด Subtitle/Text Embedding (`--feature_type subtitle`)

```text
Input: subtitles/04.vtt หรือ subtitles_gloss_cc_time/04.vtt (119 cues)
        │
  For each cue text:
    ├─► prepend language tag: "<en> <bfi>" + cue_text
    ├─► tokenize → token IDs
    ├─► forward pass SignCLIP text encoder
    └─► ดึง pooled_text output → 768-dim vector
        │
Output: (119, 768) numpy array
  subtitle_embedding/{model}_{text_variant}/04.npy
```

**Embeddings ทั้งหมดที่สร้าง:**

| ไฟล์ | Model | Text/Sign source | Shape |
| --- | --- | --- | --- |
| `segmentation_embedding/sign_clip/04.npy` | BSL | Sign (pose) | (2780, 768) |
| `segmentation_embedding/sign_clip_multi/04.npy` | Multilingual | Sign (pose) | (2780, 768) |
| `segmentation_embedding/sign_clip_asl/04.npy` | ASL | Sign (pose) | (2780, 768) |
| `subtitle_embedding/sign_clip_multi/04.npy` | Multilingual | CC text | (119, 768) |
| `subtitle_embedding/sign_clip_multi_gloss/04.npy` | Multilingual | Gloss text | (119, 768) |
| `subtitle_embedding/sign_clip_asl/04.npy` | ASL | CC text | (119, 768) |
| `subtitle_embedding/sign_clip_asl_gloss/04.npy` | ASL | Gloss text | (119, 768) |
| `subtitle_embedding/sign_clip_multi_gloss_tokens/04.npz` | Multilingual | Gloss tokens (word-level, cached) | (192 unique, 768) |

#### Similarity Matrix

```text
subtitle_embedding : (M × 768)   ← M cues
sign_embedding     : (N × 768)   ← N sign segments (2,780)

Step 1: sim_matrix = dot(subtitle_emb, sign_emb.T)   → (M × N)
         sim_matrix[i][j] = cosine similarity ระหว่าง cue_i กับ sign_j

Step 2: softmax normalization (row-wise)
         → แต่ละแถว (แต่ละ cue) รวมเป็น 1

Step 3: cumulative sum ตามแถว
         sim_cumsum[i][j] = Σ sim_matrix[i][0..j-1]
         → ใช้คำนวณ similarity_total ของ group[k:j] ได้เร็ว:
           similarity_total = sim_cumsum[i][j] - sim_cumsum[i][k]
```

#### ทำไม Softmax ไม่ใช่ Raw Cosine

Raw cosine similarity ของ SignCLIP vectors มีค่าเฉลี่ย ~0.2–0.4 ทั่วทั้ง matrix — ต่างกันน้อยมากระหว่างคู่ที่ "ดี" กับ "ไม่ดี" ทำให้ similarity term ไม่มีพลังชี้นำ DP

`softmax_normalize` (normalize แต่ละ row ด้วย softmax) แปลงค่าให้ **ค่าสูงสุดในแต่ละ row ถูก amplify** และค่าต่ำ ๆ ถูกกด → DP จะชัดเจนมากขึ้นว่า segment ไหนเหมาะกับ cue ไหน

Sinkhorn normalization (option อื่น) ทำ joint normalization ทั้ง row และ column — เหมาะกว่าเมื่อต้องการ doubly-stochastic assignment แต่ซับซ้อนกว่าและช้ากว่า softmax

---

### 4c. A — Align: จับคู่ Subtitle กับ Sign Segment (DP)

`align.py` เรียก 5 ขั้นตอนย่อยตามลำดับ:

#### ทำไมปัญหานี้ยาก — Problem Formulation

ก่อนอธิบาย DP ต้องเข้าใจก่อนว่าเราแก้ปัญหาอะไร:

- **Input:** 119 cues จาก CC_Input แต่ละ cue มี timestamp จากเสียงพูด *(เช่น cue 5: "เด็กเรียน" เริ่ม 35.0s จบ 36.5s)*
- **เป้าหมาย:** เลื่อน timestamp ของแต่ละ cue ให้ตรงกับช่วงเวลาที่ผู้แปลภาษามือแสดงท่าจริง *(เช่น ท่ามือ "เด็กเรียน" อยู่ที่ 36.0s–37.2s)*
- **สิ่งที่มีให้:** sign segments 2,780 รายการ detect จาก pose estimation + E4s-1 segmenter

ความยากคือ cues กับ sign segments **ไม่ตรงกัน 1:1** เพราะ:

1. **Sign delay:** ผู้พูดพูด "เด็กเรียน" ก่อน แต่ผู้แปลมือมักแสดงท่า **หลัง** เสียงพูดประมาณ 0.5–2 วินาที
2. **Non-1:1 mapping:** sign segments ถูก detect ต่อเนื่อง 2,780 ท่า แต่ cue 1 ตรงกับ segment กลุ่มหนึ่ง ขณะที่ cue 2 ตรงกับ segment กลุ่มอื่น
3. **Variable group size:** cue สั้นอาจครอบคลุม 3 segments, cue ยาวอาจครอบคลุม 20 segments

งานนี้จึงเป็น **monotonic partition problem**: หา partition ของ segments 2,780 ตัวเป็น 119 กลุ่มตามลำดับ โดยแต่ละกลุ่มถูกกำหนดให้กับ cue หนึ่ง และ minimize total cost รวม

#### ขั้นที่ 1 — Pre-shift Subtitles (Bias Correction)

ก่อน alignment ระบบเลื่อน timestamp ของทุก subtitle ไปข้างหน้า เพราะท่ามือมาช้ากว่าเสียงพูดเสมอ:

```text
cue.start = cue.start + delta_bias_start   (C_MULTI: +1.3 s)
cue.end   = cue.end   + delta_bias_end     (C_MULTI: +1.0 s)
```

**วิธีหาค่า bias:** รัน experiment แรกด้วย bias=0 → ดู median start offset จากผล → นำค่านั้นเป็น bias ใน run ถัดไป (iterative refinement)

#### ขั้นที่ 2 — Candidate Window Selection

สำหรับแต่ละ cue ระบบเลือก **W sign segments** ที่ midpoint ใกล้ cue midpoint ที่สุด:

```text
cue_mid  = (cue.start + cue.end) / 2
sign_mid = (sign.start + sign.end) / 2

เลือก W segments ที่ |sign_mid - cue_mid| น้อยที่สุด
W = dp_window_size  (C_MULTI: 40)
```

W มากเกินไป → ช้า, อาจจับคู่ผิด — W น้อยเกินไป → พลาด segment ที่ถูกต้อง

**Complexity analysis:**

DP แบบ naive: $O(M \cdot N^2) = 119 \times 2780^2 \approx$ **2.3 พันล้าน operations** — ช้าเกินไป

Sliding window ลดเหลือ $O(M \cdot W^2) = 119 \times 40^2 \approx$ **190K operations** — เร็วมาก และ Numba `@njit` เพิ่มความเร็วอีก 10–50×

```python
cue_mid = (cue['start'] + cue['end']) / 2
cand = np.argsort(np.abs(sign_mids - cue_mid))[:window_size]   # argsort ตาม distance
candidate_min = int(np.min(cand))
candidate_max = int(np.max(cand))
# DP จะ loop เฉพาะ k, j ∈ [candidate_min, candidate_max]
```

#### ขั้นที่ 3 — Cost Matrix Computation

สำหรับทุกคู่ (cue_i, sign_group[k:j]) คำนวณ cost ด้วยสูตร:

```text
cost(cue_i, group[k:j]) =
    |cue_start − group_start|
  + |cue_end   − group_end|
  + dp_duration_penalty_weight × |cue_dur − group_dur|
  + dp_gap_penalty_weight       × total_gap_in_group
  + similarity_weight           × (−similarity_total)
```

| Weight | Flag | C_MULTI | ผลกระทบ |
| --- | --- | --- | --- |
| `W_dur` | `--dp_duration_penalty_weight` | 2 | บังคับ duration match |
| `W_gap` | `--dp_gap_penalty_weight` | 8 | ลงโทษ group ที่มีช่องว่าง |
| `W_sim` | `--similarity_weight` | 6 | ยิ่งคล้าย embedding → cost ต่ำ |
| `max_gap` | `--dp_max_gap` | 6 | จำกัดจำนวน sign segments สูงสุดต่อ group |

อธิบายแต่ละ term ของ cost function:

##### Term 1 & 2 — Start/End Alignment (ไม่มี weight)

$$|\text{cue\_start} - \text{group\_start}| + |\text{cue\_end} - \text{group\_end}|$$

ลงโทษถ้า timestamp ของ cue ไม่ตรงกับขอบของ group — เป็น "gravity" ที่ดึง alignment ให้ใกล้ timestamp เดิม ทำให้ DP ไม่ shift cue ออกไปไกลโดยไม่มีเหตุผล ไม่มี weight เพราะต้องการให้ term นี้สมดุลกับ timing โดยธรรมชาติ

##### Term 3 — Duration Penalty ($w_D = 2$)

$$w_D \cdot |\text{cue\_duration} - \text{group\_duration}|$$

ถ้า cue ยาว 3 วินาทีแต่ถูก map ไปยัง group ที่ยาวแค่ 0.5 วินาที → ถูกลงโทษสูง บังคับให้ group duration สัมพันธ์กับ cue duration

##### Term 4 — Gap Penalty ($w_G = 8$, term สำคัญที่สุด)

$$w_G \cdot \text{gap}(k, j) = w_G \cdot \sum_{p=k}^{j-1} \max\bigl(0,\ \text{seg}_{p+1}\text{.start} - \text{seg}_p\text{.end}\bigr)$$

ถ้า group มี "รู" ขนาดใหญ่ระหว่าง segments (ผู้แปลหยุดมือแล้วเริ่มใหม่) → ถูกลงโทษหนัก DP นิยม group ที่ segments เรียงต่อเนื่อง `gap_cost` precompute เป็น prefix cumsum → หา gap(k,j) ได้ $O(1)$ ขณะ DP ทำงาน

##### Term 5 — Similarity Reward ($w_S = 6$)

$$w_S \cdot \bigl(-\text{sim\_cum}[i][k][j]\bigr) = -w_S \cdot \sum_{s=k}^{j-1} \text{sim}[i][s]$$

เครื่องหมายลบ → **similarity สูง = cost ต่ำ** → DP เลือก group ที่ sign segments "ตรงกับ" text ของ cue มากที่สุด `sim_cumsum` precompute เป็น prefix sum → คำนวณ cumulative similarity ของช่วง [k,j) ได้ $O(1)$: `sim_cumsum[i,j] - sim_cumsum[i,k]`

#### DP Formal Formulation (Optimization Problem)

นิยาม: $M = 119$ cues, $N = 2780$ sign segments — หา **partition** ของ segments เป็น $M$ กลุ่มตามลำดับ

| | |
| --- | --- |
| **State** | $\text{dp}[i][j]$ = cost ต่ำสุดในการ assign cues $1\ldots i$ โดย cue $i$ จบที่ segment index $j$ |
| **Boundary** | $\text{dp}[0][0] = 0$, ส่วนอื่น $= +\infty$ |
| **Transition** | $\text{dp}[i][j] = \min_{k \in [i-1 \ldots j]} \bigl( \text{dp}[i-1][k] + C(i, k, j) \bigr)$ |
| **Answer** | $j^* = \arg\min_j \text{dp}[M][j]$, จากนั้น backtrack ผ่าน $\text{prev}[i][j]$ |

DP state เก็บ **segment index** ไม่ใช่ time — นี่คือเหตุผลที่ overlap เกิดขึ้นหลัง DP (อธิบายด้านล่าง)

#### ขั้นที่ 4 — DP Optimization (Numba JIT)

```text
State: dp[i][j] = min total cost ที่ assign cue 1..i โดย cue i ถูก assign ถึง sign segment j

Recurrence (i: cue index, j: sign segment index):
  dp[i][j] = min over k ∈ [i-1, j) of {
      dp[i-1][k] + cost(cue_i, group[k+1:j])
  }

Boundary: dp[0][0] = 0
Final:    j* = argmin(dp[M, :])

Complexity: O(M × W² × N)  ←  ไม่กี่วินาทีที่ M=119, N=2780, W=40
```

Inner loop compile ด้วย **Numba @njit** เพื่อความเร็ว

#### ขั้นที่ 5 — Backtracking และ Output

```text
1. j* = argmin(dp[M, :])   ← sign segment สุดท้ายที่ถูก assign
2. Backtrack ผ่าน prev[][] → คืน boundary ของแต่ละ cue
3. อัปเดต cue.start = group.start, cue.end = group.end
4. เขียน VTT + EAF (tier SUBTITLE_SHIFTED)
```

#### Subgroup Refinement (Post-DP Pass)

หลัง DP คืน boundary $[k_i, j_i]$ สำหรับแต่ละ cue — group นั้นอาจมี **resting segments** ที่ผู้แปลหยุดพักระหว่างกลาง ซึ่งไม่ควรรวมอยู่ใน timestamp

Post-processing ทำ **subgroup refinement**:

1. ตัด group ออกเป็น contiguous subgroups โดย segments ที่ gap > `--dp_max_gap` (6 วินาที) จะถูกตัดแยก
2. คำนวณ `cost_for_subgroup()` ของแต่ละ subgroup (ใช้ cost function เดียวกัน)
3. เลือก subgroup ที่ cost ต่ำสุด → ใช้ start ของ segment แรกและ end ของ segment สุดท้ายของ subgroup นั้นเป็น timestamp ใหม่

ผลคือ cue timestamp ที่แม่นยำขึ้น — ไม่ถูกยืดออกโดย resting segments ที่ไม่เกี่ยวข้อง

#### Call Chain ใน Code

```text
process_video()                           [align.py]
  ├── get_subtitle_cues()                 [utils.py]       → 119 cues
  ├── get_sign_segments_from_eaf()        [utils.py]       → 2,780 segs
  ├── shift_cues(delta_start, delta_end)  [utils.py]       → pre-shift
  ├── compute_similarity_matrix()         [align_similarity.py]
  │     ├── load embeddings (.npy)
  │     ├── cosine normalize
  │     └── softmax_normalize (row-wise)
  ├── dp_align_subtitles_to_signs()       [align_dp.py]
  │     ├── compute_alignment_cost()      [align_dp.py]    ← cost formula
  │     ├── @njit Numba inner loop
  │     └── backtrack via prev[][]
  └── write_updated_eaf()                 [utils.py]
```

#### Post-processing — Overlap Fix

DP บางครั้ง assign cues ให้ overlap กัน (เพราะ 172 CC cues → 119 sign slots)
`fix_overlap_vtt.py` แก้ด้วยการ clamp end time:

```python
for i in range(len(cues) - 1):
    if cues[i].end > cues[i+1].start:
        cues[i].end = cues[i+1].start    # clamp end เท่านั้น
```

ผล: overlap ~88% → **0%** โดยไม่กระทบ start time → metric ±Ns และ mean offset ไม่เปลี่ยน

#### ทำไม Output จึงมี Overlap สูง ~86–88%

DP ออกแบบมาเพื่อหา **start time ที่แม่นยำ** ของแต่ละ cue — ไม่มี constraint ว่า cue ห้าม overlap กัน เหตุผล 3 ข้อ:

##### เหตุผลที่ 1 — DP state ไม่เก็บ end time

State $\text{dp}[i][j]$ = cost ที่ cue $i$ จบที่ segment **index $j$** (ไม่ใช่เวลา $t$) → DP ไม่รู้ว่า end time ของ cue $i-1$ อยู่ที่เท่าไร จึงไม่มีทางรู้ว่า start time ของ cue $i$ จะทับ cue $i-1$ หรือเปล่า

##### เหตุผลที่ 2 — Subgroup refinement อาจดัน end time กลับ

ขั้นตอน subgroup refinement เลือก subgroup ที่ cost ต่ำสุด ซึ่งบางครั้ง subgroup ที่ดีของ cue $i$ จะจบก่อน start time ของ cue $i+1$ → cue $i$ มี end time > start time ของ cue $i+1$ โดยไม่ได้ตั้งใจ

ตัวอย่าง: DP assign cue 5 → segments 120–128 → **end = 65.2s**, cue 6 → segments 130–140 → **start = 64.8s** → overlap 0.4s — DP ไม่ได้ผิด มันแค่ minimize cost ของแต่ละ cue โดยอิสระ

##### เหตุผลที่ 3 — TSL มี resting periods มากกว่า BSL

SEA ถูกออกแบบสำหรับ BSL ที่ segments เรียงค่อนข้างแน่น → overlap น้อยในต้นฉบับ แต่ TSL มี resting periods มากกว่า → gap ระหว่าง groups ใหญ่กว่า → โอกาส overlap สูงขึ้น

การบังคับ $\text{end}_i \leq \text{start}_{i+1}$ ใน DP ต้องเพิ่ม dimension ของ state หรือทำ 2-pass ซึ่งซับซ้อนและช้ากว่า แนวทางของ SEA คือแยก overlap fix เป็น post-processing เพราะ (1) clamp end ไม่กระทบ start → ไม่กระทบ metric หลัก, (2) ทำได้ใน $O(N)$ pass เดียว, (3) overlap ส่วนใหญ่ < 0.5s ไม่มีผลต่อ mean offset

---

### 4d. Pipeline Reference — คำสั่งทุกขั้นตอน (Copyable Commands)

> สำหรับการรัน pipeline ตั้งแต่ต้นจนจบ — ทุกขั้นตอนมีคำสั่งจริง, พารามิเตอร์ที่ใช้, output ที่ได้ และวิธีตรวจสอบ

```powershell
$repo = (Resolve-Path .).Path        # ปรับให้ชี้ไปที่ Sign_to_sub root ของคุณ
$ea   = "$repo\example_alignment"
cd $repo
venv\Scripts\activate
```

#### Step 1 — Extract CC_Input จาก EAF

| | |
| --- | --- |
| **Purpose** | แยก `CC_Input` tier (119 cues) จาก `Test.eaf` เป็น VTT |
| **Input** | `$ea\Test.eaf` |
| **Command** | `python example_alignment\extract_cc_from_eaf.py "$ea\Test.eaf" $ea\subtitles\04.vtt --tier CC_Input` |
| **Output** | `$ea\subtitles\04.vtt` (119 cues) |
| **Verify** | stdout: `[OK] Extracted 119 cues` |

#### Step 1b — Make Gloss VTT

| | |
| --- | --- |
| **Purpose** | สร้าง VTT ที่มี Gloss text แต่ใช้ timestamp ของ CC_Input |
| **Command** | `python example_alignment\make_gloss_cc_vtt.py` |
| **Output** | `$ea\subtitles_gloss_cc_time\04.vtt` (119 cues, 0 fallback) |
| **Verify** | stdout: `119 cues \| fallback CC: 0 cues` |

#### Step 2 — Pose Estimation

| | |
| --- | --- |
| **Purpose** | ดึง skeleton landmarks 543 จุดต่อเฟรมจากวิดีโอ |
| **Command** | `videos_to_poses --format mediapipe --model_complexity 2 --refine_face_landmarks --no-smooth_landmarks --directory $ea` |
| **Output** | `$ea\04.pose` (~358 MB binary) |
| **Runtime** | ~15 นาที CPU สำหรับวิดีโอ 11 นาที |

#### Step 3 — Segmentation (E4s-1)

| | |
| --- | --- |
| **Purpose** | ตัดท่ามือออกเป็น sign segments ด้วย E4s-1 GRU |
| **Command** | `python SEA\segmentation.py --pose_dir $ea --segmentation_dir $ea\segmentation_output --video_ids $ea\video_ids.txt --sign-b-threshold 30 --sign-o-threshold 50 --num_workers 1` |
| **Output** | `$ea\segmentation_output\E4s-1_30_50\04.eaf` (SIGN: 2,780 / SENTENCE: 418) |

> ⚠️ **Critical:** ต้องส่ง `--sign-b-threshold 30 --sign-o-threshold 50` เสมอ — ค่า default ใน `config.py` คือ 70 ซึ่งให้ segmentation ที่ต่างออกไป
>
> ⚠️ **Critical:** `--segmentation_dir` ต้องชี้ไปที่ **parent dir** (`segmentation_output`) ไม่ใช่ `segmentation_output\E4s-1_30_50` — `align.py` จะต่อ subdirectory เองจาก threshold parameters

#### Step 4a — Sign Embeddings

```powershell
cd fairseq_signclip\examples\MMPT

python scripts_bsl\extract_episode_features.py `
  --pose_dir ..\..\..\..\example_alignment `
  --segmentation_dir ..\..\..\..\example_alignment\segmentation_output\E4s-1_30_50 `
  --feature_dir ..\..\..\..\example_alignment\segmentation_embedding\sign_clip_multi `
  --video_ids ..\..\..\..\example_alignment\video_ids.txt `
  --model_name multilingual --feature_type sign
```

ทำซ้ำกับ `--model_name bsl` และ `--model_name asl` (เปลี่ยน `--feature_dir` ให้ตรงกับ model)

#### Step 4b — Subtitle Embeddings (ต้อง regenerate เมื่อ input text เปลี่ยน)

```powershell
cd fairseq_signclip\examples\MMPT

:: Multilingual + Gloss text (สำหรับ C_MULTI — experiment ดีที่สุด)
python scripts_bsl\extract_episode_features.py `
  --feature_type subtitle `
  --subtitle_dir ..\..\..\..\example_alignment\subtitles_gloss_cc_time `
  --feature_dir  ..\..\..\..\example_alignment\subtitle_embedding\sign_clip_multi_gloss `
  --video_ids    ..\..\..\..\example_alignment\video_ids.txt `
  --model_name   multilingual

:: Multilingual + CC text (สำหรับ B_MULTI)
python scripts_bsl\extract_episode_features.py `
  --feature_type subtitle `
  --subtitle_dir ..\..\..\..\example_alignment\subtitles `
  --feature_dir  ..\..\..\..\example_alignment\subtitle_embedding\sign_clip_multi `
  --video_ids    ..\..\..\..\example_alignment\video_ids.txt `
  --model_name   multilingual
```

Expected output: shape `(119, 768)` ทุกไฟล์

#### Step 5 — Alignment (C_MULTI — Best Experiment)

```powershell
python SEA\align.py `
  --video_ids "$ea\video_ids.txt" `
  --pr_sub_path "$ea\subtitles_gloss_cc_time" `
  --save_dir "$ea\aligned_output_multi_gloss" `
  --segmentation_dir "$ea\segmentation_output" `
  --sign-b-threshold 30 --sign-o-threshold 50 `
  --segmentation_embedding_dir "$ea\segmentation_embedding\sign_clip_multi" `
  --subtitle_embedding_dir "$ea\subtitle_embedding\sign_clip_multi_gloss" `
  --similarity_measure sign_clip_embedding --live_model_name multilingual `
  --similarity_weight 6 --dp_duration_penalty_weight 2 --dp_gap_penalty_weight 8 `
  --dp_max_gap 6 --dp_window_size 40 `
  --pr_subs_delta_bias_start 1.3 --pr_subs_delta_bias_end 1.0 --overwrite
```

สำหรับ word-level experiments (C_MULTI_word) ใช้ `--live_embedding --tokenize_text_embedding` แทน `--subtitle_embedding_dir`

#### Step 6 — Post-processing + Evaluation Task 1

```powershell
python example_alignment\evaluate_all_to_csv.py
```

Output: `aligned_output_*/04_no_overlap.vtt` (7 files) + `evaluation_task1_results.csv`

#### Step 7 — Task 2 Gloss Labeling

```powershell
python example_alignment\align_gloss_labels.py      # alignment (รัน ~3 min ครั้งแรก, ~5s ครั้งถัดไป)
python example_alignment\evaluate_gloss_labeling.py  # evaluation
```

#### Step 8 — Build Comparison EAF + Visualization

```powershell
python example_alignment\add_vtt_tiers_to_eaf.py --overwrite
# Output: Test_comparison.eaf (15 experiment tiers + original tiers)

python example_alignment\plot_alignment.py
# Output: figures\timeline_first_2min.png
```

---

## 4e. Evaluation Metrics — อ่านก่อนดูผลลัพธ์

> ส่วนนี้อธิบายความหมายของ metric แต่ละตัวที่ปรากฏในตาราง Section 5–6 เพื่อให้เข้าใจว่า "ผลดี" หรือ "ผลแย่" แค่ไหน

### Task 1 Metrics — Subtitle Alignment

**วิธีวัด:** เปรียบเทียบ prediction ทีละ cue กับ Ground Truth ที่นักวิจัยทำด้วยมือ (`CC_Aligned` tier) โดยใช้ index-based matching: pred[i] ↔ gt[i] ครอบคลุม 119/119 cues

---

#### Mean Start Offset (วินาที)

```text
start_offset[i] = pred_start[i] − gt_start[i]

ตัวอย่าง:
  pred_start = 36.2s,  gt_start = 36.0s  →  offset = +0.2s  (predict ช้าไปนิดหน่อย)
  pred_start = 35.9s,  gt_start = 36.0s  →  offset = −0.1s  (predict เร็วกว่า GT)

mean_start_offset = เฉลี่ยของ offset ทั้ง 119 cues
```

| ค่า | ความหมาย |
| ---: | --- |
| `0.00 s` | perfect — predict ตรงกับ GT เป๊ะโดยเฉลี่ย |
| `−0.16 s` | (C_MULTI) predict เร็วกว่า GT เล็กน้อย 160ms — ดีมาก |
| `+1.25 s` | (D_ASL) predict ช้ากว่า GT 1.25 วินาที — ยอมรับได้แต่ไม่ดี |
| ค่าลบ | ระบบ "รีบ" ขึ้น timestamp ก่อน GT บอก |
| ค่าบวก | ระบบ "ช้า" ขึ้น timestamp หลัง GT บอก |

> ⚠️ **ข้อควรระวัง:** mean offset คือค่าเฉลี่ย — อาจมี outlier ที่ offset ±5 วินาทีแต่ถูกชดเชยด้วย cues อื่นที่ดีมาก ดู distribution ด้วย ±Ns coverage

---

#### % within ±N seconds (coverage metrics)

```text
within_N[i] = 1  ถ้า  |pred_start[i] − gt_start[i]| ≤ N วินาที
            = 0  ถ้า  เกิน N วินาที

% within ±N s = (จำนวน cues ที่ offset ≤ N s) / 119 × 100
```

| Threshold | ความหมายในทางปฏิบัติ |
| --- | --- |
| **±1 s** | ดูด้วยตา: subtitle ปรากฏก่อน/หลังท่ามือ ≤ 1 วินาที — ผู้ชมรู้สึกว่า sync |
| **±2 s** | ยอมรับได้: เห็นว่า offset มีอยู่แต่ยังอ่านตามได้ |
| **±3 s** | สังเกตเห็นชัด แต่ยังเดาได้ว่าตรงกับท่าไหน |
| **> ±3 s** | misaligned อย่างชัดเจน |

ผล C_MULTI: 73.9% / 95.0% / **100%** — หมายความว่า ทุก cue อยู่ภายใน 3 วินาทีจาก GT ทั้งหมด

---

#### F1 @ IoU threshold (Task 1)

```text
"นับว่า cue ถูก" ถ้า IoU(pred_interval, gt_interval) ≥ threshold

Precision = TP / (TP + FP)   ← ใน predictions ที่ระบบให้มา มีกี่ % ที่ถูกต้อง
Recall    = TP / (TP + FN)   ← ใน GT ทั้งหมด มีกี่ % ที่ระบบหาเจอ
F1        = 2 × P × R / (P + R)
```

F1 @ 0.50 IoU = 88.2% หมายความว่า 88% ของ predictions มีความทับซ้อนกับ GT ≥ 50%

---

#### Overlap Rate (หลัง fix_overlap_vtt.py)

```text
overlap_rate = จำนวน cues ที่ end[i] > start[i+1] / 119 × 100
```

ค่า 0% หลัง fix หมายความว่าไม่มี subtitle ทับซ้อนกันเลย — ระบบ safe สำหรับ production

---

### Task 2 Metrics — Gloss Labeling

**วิธีวัด:** เปรียบเทียบ 852 predictions (pred) กับ 852 GT entries แบบ greedy matching — จับคู่แต่ละ pred กับ GT entry ที่ตำแหน่งเดียวกัน (index-based ตาม token order ต่อ sentence)

---

#### Mean IoU (ค่าหลักของ Task 2)

```text
IoU(pred, gt) = overlap(pred, gt) / union(pred, gt)

ตัวอย่าง:
  pred:  34.1s → 34.9s   (0.8s)
  GT:    34.0s → 34.8s   (0.8s)
  intersection: 34.1s → 34.8s = 0.7s
  union:        34.0s → 34.9s = 0.9s
  IoU = 0.7 / 0.9 = 0.78

mean_IoU = เฉลี่ยของ IoU ทั้ง 852 predictions
```

| ค่า | ความหมาย |
| ---: | --- |
| 1.0 | ตรงกับ GT เป๊ะทุก ms |
| **0.49** | (ผลที่ได้) ทับซ้อน ~49% โดยเฉลี่ย — เกิน sanity bar มาก |
| 0.14 | **Sanity bar** = chance level (1/7 ≈ สุ่มเดาจาก 7 segments ต่อ token) |
| 0.0 | ไม่ทับซ้อน GT เลย |

---

#### % IoU ≥ 0.5 และ % IoU ≥ 0.3

```text
% IoU ≥ 0.5 = สัดส่วน predictions ที่ทับซ้อน GT อย่างน้อยครึ่งหนึ่ง
% IoU ≥ 0.3 = สัดส่วนที่ทับซ้อน GT อย่างน้อย 30%
```

ใช้ threshold 2 ค่าเพราะ Task 2 ยากมาก (alignment ระดับ ms) — 0.3 เป็น "ยอมรับได้" 0.5 เป็น "ดี"

ผล: 48.4% ของ predictions ทับซ้อน GT ≥ 50%, และ 77% ทับซ้อน ≥ 30%

---

#### % any temporal overlap

```text
= สัดส่วน predictions ที่ IoU > 0 (ทับซ้อน GT บ้างไม่ว่าน้อยแค่ไหน)
```

ผล 97.5% หมายความว่า 97.5% ของ predictions อยู่ "ในพื้นที่ถูกต้อง" แม้จะยังไม่ตรงพอดี

---

#### Fallback uniform

```text
= จำนวนประโยคที่ DP ล้มเหลวจนต้อง fallback เป็น uniform split
  (แบ่ง sentence duration เท่าๆ กันทุก token)
```

ผล 0 / 119 หมายความว่าทุก sentence ถูก align โดย DP ปกติ ไม่มีการ fallback แม้แต่ครั้งเดียว

---

### สรุปรวด — อ่านตารางอย่างไร

```text
ตาราง Task 1 (Section 5.2):
  → ดู "Mean" ก่อน: ใกล้ 0 = ดี, ติดลบเล็กน้อย = ระบบเร็วกว่า GT นิดหน่อย (OK)
  → ดู "±1 s" ต่อ: ≥ 70% = ใช้ได้, ≥ 80% = ดี
  → ดู "±3 s": 100% หมายถึงทุก cue ยังอยู่ในพื้นที่ที่ยอมรับได้

ตาราง Task 2 (Section 6.5):
  → ดู "Mean IoU": > 0.3 = ผ่าน sanity bar, ~0.49 = ดีสำหรับ zero-shot cross-lingual
  → ดู "% ≥ 0.5": มากกว่า 48% → เกือบครึ่งหนึ่งของ predictions ถูกต้องระดับ 50%+ overlap
  → ดู "Fallback": 0 = ระบบ stable ทั้งหมด
```

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

## 10. ForcedAlignment Dataset — Task 2 Scale-up (เริ่มวางแผน 17 พ.ค., เสร็จสมบูรณ์ 24 พ.ค. 2569)

> ขยาย Task 2 จากวิดีโอตัวอย่างเดียว (`04.mp4`) ไปยัง **1,132 คลิป** ใน ForcedAlignment dataset
> เพื่อวัดผลในระดับ dataset จริงและรายงานใน `Gloss_Labeling_Template.docx`
> แผนเต็ม + DP deep-dive อยู่ใน [ForcedAlignment/PLAN_ForcedAlignment_Task2.md](ForcedAlignment/PLAN_ForcedAlignment_Task2.md)
>
> **สถานะ (24 พ.ค. 2569):** ✅ **เสร็จสมบูรณ์** — full run + evaluation + error analysis + final docx report
> **ผลหลัก:** Config #1 (CC → CC_Aligned, no sil) — F1@0.5 = 68.6%, Mean IoU = 0.5928 (ดีกว่า 04.mp4 baseline 0.4901)

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

### 10.3 แผนการทดลอง 5 ชุด (revised 18 พ.ค.)

| # | Input Tier | Ground Truth Tier | เหตุผล |
| --- | --- | --- | --- |
| 1 | `CC` (whitespace tokenize, no sil) | `CC_Aligned` (กรอง sil ออก) | Baseline — align CC text ไม่มี sil |
| 2 | `CC` (whitespace + เพิ่ม sil head/tail) | `CC_Aligned` (ใช้ตามจริง: sil\|word\|sil) | ทดสอบว่าการ model sil frames ช่วยไหม |
| 3 | `Gloss` (pipe-delimited) | `Gloss_Labeling` | ⭐ baseline — เทียบกับ Gloss whole-video (04.mp4) |
| 4 | `Gloss1` (มี sil token) | `Gloss_Labeling1` | ทดสอบ explicit sil modeling สำหรับ Gloss |
| 5 | `Gloss2` (มี sil1/sil2) | `Gloss_Labeling2` | ทดสอบ numbered sil — แยก sil เริ่มต้น/สิ้นสุดได้ไหม |

> **Note:** เดิมแผนมี 6 configs โดย Exp 6 = "Gloss per-sentence" — ตัดออกหลัง verified ว่าทุกคลิปคือ 1 sentence อยู่แล้ว → เทียบเท่า Exp 3 ดู `ForcedAlignment/PLAN_ForcedAlignment_Task2.md` §3
>
> **Note 2:** Q2 (CC_Aligned2) — verified ว่า**ไม่มี**ใน EAF ใดเลย → Exp 1 และ 2 ใช้ `CC_Aligned` เดียวกัน ต่างกันที่ tokenization+evaluation strategy

**Metrics ที่รายงาน:** Precision / Recall / F1 / Accuracy ที่ IoU ≥ 0.5 + Mean IoU + % zero overlap

### 10.4 Pipeline ใหม่ vs Pipeline เดิม

| ด้าน | 04.mp4 (เดิม) | ForcedAlignment (ใหม่) |
| --- | --- | --- |
| Input | 1 ไฟล์ 11 นาที | 1,132 ไฟล์ avg 5.8 s |
| Pose | 1 × `04.pose` (358 MB) | 1,132 × `N.pose` (~3.3 GB รวม) |
| EAF | tiers รวมใน `Test.eaf` เดียว | EAF แยกต่อคลิป |
| Video location | `example_alignment/04.mp4` | กระจายใน 6 subfolder |
| Configs | ablation (Gloss vs Gloss_Input) | 5 configs พร้อมกัน |

**Script ใหม่ที่ต้องสร้าง:**

- `ForcedAlignment/run_forced_alignment.py` — Orchestrator 7 phases (scan → pose → seg → emb → DP × 5 → export → evaluate)
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
| DP alignment × 5 configs | ~60 min | in-process, NumPy + Numba JIT |
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

### 10.7 คำถามที่ resolved แล้ว (อัปเดต 18 พ.ค. 2569)

| # | คำถาม | คำตอบ |
| --- | --- | --- |
| ~~Q1~~ ✅ | CC1 vs CC2 tokenization | CC1/CC2 ไม่มีจริงใน EAF — ใช้ tier `CC` แล้ว tokenize 2 แบบ (no-sil vs with-sil) |
| ~~Q2~~ ✅ | `CC_Aligned2` มีไหม | **ไม่มี** (0/1132 EAFs verified) — Exp 2 ใช้ `CC_Aligned` เดียวกับ Exp 1 |
| ~~Q3~~ ✅ | Exp 6 per-sentence ต่างจาก Exp 3 | **ไม่ต่าง** (ทุกคลิป = 1 sentence) → **ตัด Exp 6 ออก** เหลือ 5 configs |
| Q4 | DP parameters — default หรือ tune | ใช้ default ก่อน — sweep เฉพาะถ้า Mean IoU แย่กว่า 04.mp4 baseline > 10 pp |

ดูแผนเต็ม + verification log ที่ [ForcedAlignment/PLAN_ForcedAlignment_Task2.md](ForcedAlignment/PLAN_ForcedAlignment_Task2.md) §11

### 10.8 Timeline และสถานะ

```text
วันที่ 1 (เช้า–บ่าย)  : เขียน run_forced_alignment.py + evaluate_fa_dataset.py
วันที่ 1 (บ่าย)       : ทดสอบ debug mode 3 clips
วันที่ 1 (ค่ำ–คืน)    : รัน full pipeline overnight (~12 ชั่วโมง)
วันที่ 2 (เช้า)       : ตรวจผล + error logs
วันที่ 2 (เช้า–บ่าย)  : วิเคราะห์ผล + กรอก Gloss_Labeling_Template.docx + เขียน Progress note
```

**สถานะ:**

- 17 พ.ค. 2569 — วางแผนเสร็จ
- 18 พ.ค. 2569 — verify dataset (EAF/video match, tier inventory, MEDIA_DESCRIPTOR repair)
- 20 พ.ค. 2569 — smoke test 3 clips ผ่าน + เริ่ม full run overnight
- 21 พ.ค. 2569 — full run จบ (43,395 s ≈ 12.05 ชม.) + เริ่ม evaluation
- 24 พ.ค. 2569 — error analysis Config #3 + finalize docx report ✅ **เสร็จ**

### 10.9 ผลการ Execute Full Pipeline (20–21 พ.ค. 2569)

คำสั่งที่ใช้รัน:

```powershell
venv\Scripts\python.exe ForcedAlignment\run_forced_alignment.py --configs all
```

**สรุป run:**

| Item | Value |
| --- | --- |
| Start | 20 พ.ค. 2569 ~13:26 |
| Finish | 21 พ.ค. 2569 ~01:30 |
| Total runtime | 43,395.3 s ≈ **12.05 ชั่วโมง** (ตรงกับ estimate 11–12 ชม.) |
| Poses ที่สกัดได้ | 1,132 / 1,132 |
| Segmentation EAFs | 1,132 / 1,132 |
| Embeddings | **1,075 / 1,132** (57 clips ไม่มี valid SIGN segments → fallback) |
| Predicted EAFs | 1,132 / 1,132 |
| GT-vs-prediction comparison EAFs | 1,132 / 1,132 (`ForcedAlignment/output/comparison_eafs`) |
| Prediction CSVs | 5 / 5 (configs 1–5) |
| Evaluation CSVs | 6 (`eval_config1..5.csv` + `evaluation_summary.csv`) |
| Logs | `output/logs/full_run_20260520_132647.{out,err}.log` |

**Disk usage จริง:**

| Artifact | Size |
| --- | --- |
| `output/poses/*.pose` | ~3.3 GB |
| `output/emb/*.npy` | ~113 MB |
| `output/seg/E4s-1_30_50/*.eaf` | ~55 MB |
| Predictions + EAFs + comparison + logs | ~150 MB |
| **รวม** | **~3.6 GB** |

### 10.10 ผลลัพธ์ทั้ง 5 Configs (full dataset)

จาก `ForcedAlignment/output/evaluation/evaluation_summary.csv`:

| Config | GT Tier | Input | Pred | GT | P@0.5 | R@0.5 | F1@0.5 | Accuracy | **Mean IoU** | Frame Acc | Fallback rows |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | CC_Aligned (no sil) | CC | 1,171 | 1,132 | **67.5%** | **69.8%** | **68.6%** | **69.8%** | **0.5928** | **76.8%** | 57 |
| 2 | CC_Aligned | CC + sil | 3,435 | 3,396 | 17.5% | 17.7% | 17.6% | 17.7% | 0.2039 | 21.3% | 427 |
| 3 | Gloss_Labeling | Gloss | 1,713 | 1,713 | 7.8% | 7.8% | 7.8% | 7.8% | 0.2484 | 26.2% | 73 |
| 4 | Gloss_Labeling1 | Gloss1 | 3,977 | 3,977 | 20.7% | 20.7% | 20.7% | 20.7% | 0.2229 | 22.3% | 495 |
| 5 | Gloss_Labeling2 | Gloss2 | 3,977 | 3,977 | 20.9% | 20.9% | 20.9% | 20.9% | 0.2238 | 22.4% | 495 |

**ข้อสังเกต:**

- **Config #1 ชนะทุก metric หลัก** — F1 68.6% และ Mean IoU 0.5928 **ดีกว่า 04.mp4 baseline (0.4901)** → SignCLIP + DP generalize ได้ดีไปยัง dataset ใหม่
- Config #2 (เติม sil หัว/ท้าย) ผลแย่ลงมาก (mIoU 0.20) — sil token ใน SignCLIP เป็น noise สำหรับ DP ไม่ใช่ structural signal
- Config #3 mIoU 0.2484 ดูตกจาก 04.mp4 baseline (0.4901) → **ไม่ใช่ aligner regression** ดู §10.11
- Configs #4–5 (sil-modeling สำหรับ Gloss) ใกล้เคียงกัน mIoU ~0.22 — Recall ดีกว่า #3 มากเพราะ token count ใน GT มากขึ้น
- Fallback rows ของ configs 2/4/5 ใหญ่กว่า config 1 เพราะ 57 clips × (CC tokens + sil) หรือ 57 × (Gloss tokens + sil)

### 10.11 Error Analysis — Config #3 Mean IoU Regression (resolved 24 พ.ค.)

**สังเกตุ:** Config #3 (Gloss → Gloss_Labeling) ได้ Mean IoU = 0.2484 บน ForcedAlignment แต่ config เดียวกันได้ 0.4901 บน 04.mp4 → ตก ~25 pp

**ทำการ deep analysis บน `eval_config3.csv`:**

| Metric | Config 1 | Config 3 | Implication |
| --- | ---: | ---: | --- |
| Mean **pred** duration | 1.70 s | 1.16 s | Aligner ทำนายช่วงสั้น (ตามสัญลักษณ์จริง) |
| Mean **GT** duration | 1.91 s | **3.84 s** | GT Gloss_Labeling กว้างกว่า 2 เท่า |
| Pred/GT duration ratio | 0.89 | **0.30** | IoU ถูก cap ที่ ~0.30 ทันทีจากกลศาสตร์ |
| Text match (positional) | 97.1% | **100%** | ทุก token ลำดับ + ระบุตัวถูกต้อง |
| % any overlap over GT | 96.4% | **97.3%** | Aligner ลงในช่วง GT ที่ถูกต้อง |
| Pred fully contained in GT | n/a | **72.7%** | 3 ใน 4 ของ preds อยู่ภายใน GT span ทั้งหมด |
| Mean fraction of pred inside GT | n/a | **88.7%** | เกือบทั้ง pred อยู่ใน GT |

**Root Cause (annotation convention mismatch):**

- `Gloss_Labeling` ใน ForcedAlignment annotate ด้วย convention ที่ **ยืดทุก word ให้ครอบคลุมช่วงเวลายาวต่อเนื่อง** (avg 3.84 s/word บนคลิป 5.8 s) — ไม่มี silence gap ระหว่างคำ
- 04.mp4 ใช้ convention ที่กระชับกว่า (per-word boundary ใกล้สัญลักษณ์จริง) → pred/GT widths matched → IoU สูง
- IoU = intersection / union — เมื่อ GT กว้างกว่า pred 3 เท่า IoU มี upper bound ที่ ~0.30 แม้ aligner ลงในตำแหน่งถูกต้องเป๊ะ

**ข้อสรุป:** Config #3 mIoU 0.2484 vs 04.mp4 mIoU 0.4901 ไม่ใช่ aligner regression — เป็น artifact ของ annotation convention ที่ต่างกันระหว่าง 2 dataset

**คำแนะนำสำหรับการตีความ:**

1. ใช้ **Config #1** เป็นผลหลักของรายงาน (F1 68.6%, mIoU 0.5928 — เทียบได้กับ 04.mp4 0.4901 → **ดีกว่า**)
2. ห้ามเทียบ Config #3 mIoU โดยตรงกับ 04.mp4 ใน publication
3. สำหรับ Gloss-tier configs ใช้ `% any-overlap over GT` (97.3% สำหรับ #3) หรือ `frame_accuracy` (26.2%) เป็น metric เปรียบเทียบ — ทั้งคู่ไม่อ่อนไหวต่อความกว้างของ GT
4. Config #4–5 ได้ Recall@0.5 ดีกว่า #3 ถึง 2.6 เท่า เพราะ sil tokens แบ่ง GT ออกเป็นช่วงย่อย → ความกว้าง GT ใกล้ pred มากขึ้น (ไม่ใช่เพราะ sil modeling ทำงานดี — ดู §10.11b)

### 10.11b Architectural Mismatch — ทำไม sil-bearing configs ผลตก (resolved 25 พ.ค. 2569)

**Finding:** Configs ที่ใส่ sil tokens ใน input (#2, #4, #5) ผลตกหนัก **ไม่ใช่เพราะ DP ทำงานผิด** แต่เป็นเพราะ **architectural mismatch ระหว่าง SEA E4s-1 segmenter กับการ model silence**

**Evidence — distinct fallback clips ต่อ config:**

| Config | Tokens/clip | Distinct fallback clips | เพิ่มจาก base | สาเหตุ |
| ---: | ---: | ---: | ---: | --- |
| 1 (CC, no sil) | 1.0 | 57 | base | embedding หาย |
| 2 (CC + sil) | 3.0 | **142** | **+85** | K < T (SEA ไม่มี sil segment) |
| 3 (Gloss, no sil) | 1.5 | 60 | +3 | embedding หาย |
| 4 (Gloss + sil) | 3.5 | **154** | **+97** | K < T |
| 5 (Gloss + numbered sil) | 3.5 | **154** | **+97** | K < T |

**Root cause:**

SEA E4s-1 ถูก train บน binary classification "frame นี้ทำท่าอยู่หรือไม่?" → output เป็น SIGN segments **เฉพาะตอน active signing** ไม่สร้าง segment สำหรับช่วงพัก/sil (เป็น design intent ไม่ใช่ bug)

ปัญหาเกิดเมื่อ input config มี sil tokens:

- ถ้า K (segments) < T (tokens) → DP มี segments ไม่พอ → fallback uniform ทันที
- ถ้า K ≥ T → DP ถูกบังคับให้ assign sil ให้ word-segment ที่ผิด → text match ตก + IoU ตก

**ผลกระทบเชิงตัวเลข:** 7.5–8.6% ของ dataset (85–97 clips ต่อ config) ตกหล่นเป็น uniform split ทันทีเพราะ K < T

**Per-pair analysis แยก DP vs Fallback (สำคัญ):**

| Config | DP pairs | DP mean IoU | DP hit@0.5 | Fallback pairs | Fallback mean IoU | Fallback hit@0.5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 (CC, no sil) | 1,075 | **0.608** | **73.5%** | 57 | 0.302 | 0.0% |
| 2 (CC + sil) | 2,970 | 0.127 | 7.0% | 426 | 0.738 | **92.5%** |
| 3 (Gloss, no sil) | 1,640 | 0.217 | 3.7% | 73 | 0.962 | **100%** |
| 4 (Gloss + sil) | 3,482 | 0.154 | 11.1% | 495 | 0.709 | **88.3%** |
| 5 (Gloss + num sil) | 3,482 | 0.154 | 11.0% | 495 | 0.711 | **88.3%** |

**Key insights ที่สำคัญกว่า K<T เพียงอย่างเดียว:**

1. **DP ทำงานดีเฉพาะ Config 1** (hit 73.5%) — เมื่อมี sil tokens ใน input DP ทำได้แย่มาก (Config 2: 7%, Config 4: 11.1%) เพราะ per-token softmax บังคับให้ DP จับ segment แม้ sil token ไม่มี real signal
2. **Fallback uniform บังเอิญตรงโครงสร้าง GT** ใน sil-bearing configs — Config 2 fallback hit 92.5%, Config 4 fallback hit 88.3% เพราะทั้ง uniform split และ GT (sil+word+sil) แบ่ง clip เท่า ๆ กัน
3. **53% ของ hits ของ Config 4** มาจาก fallback uniform (437/825) ไม่ใช่ DP → F1 ของ Config 4 ที่ดูสูงกว่า Config 3 มาจาก fallback ไม่ใช่ aligner ทำงานดีขึ้น

**ทำไม Config 4/5 F1 ดูดีกว่า #3 ทั้งที่มี sil?** ไม่ใช่ sil modeling ดี และไม่ใช่ GT แคบลง (pred/GT ratio Config 4 = 0.37 ≈ Config 3 = 0.30) — เป็นเพราะ **fallback uniform บังเอิญตรง GT structure** ใน clips ที่มีหลาย sil tokens

**ทำไม Config 5 (numbered sil) ≈ Config 4?** 91.7% ของ predictions identical กับ Config 4 (3,645 / 3,977 pairs) — Numbered sil เพิ่มความเฉพาะเจาะจงของ token embedding แต่ไม่มี structure ใน SIGN tier ให้ DP ใช้ (SEA ไม่มี sil segments)

**Implications สำหรับงานต่อไป (corrected):**

1. **Bottleneck คือ DP + sil tokens, ไม่ใช่ segmenter เพียงอย่างเดียว** — แม้คลิปที่ K ≥ T ก็ตาม DP ก็ยังสับสนเมื่อมี sil tokens (DP hit rate 7-11% ใน sil configs vs 73.5% ใน Config 1)
2. **ทางแก้ที่เหมาะสมที่สุด (priority ใหม่):**
   - (A) **Post-processing** ใช้ Config #1 หรือ #3 ทำ word alignment ก่อน แล้วเติม sil intervals ระหว่าง word predictions ในขั้นตอนแยก (ไม่ผ่าน DP) — ง่ายและตรง root cause
   - (B) **แก้ DP cost function** ให้ sil tokens skip embedding หรือใช้ uniform similarity (ไม่บังคับ DP เลือก word-segment ผิด) — แก้ตรงจุดที่ DP สับสน
   - (C) Synthesize "sil segments" ระหว่าง SIGN segments ก่อนเข้า DP — แก้ K<T แต่ไม่แก้ DP confusion
   - (D) Fine-tune SEA E4s-1 ให้ output multi-class (sign/sil/transition) — ใหญ่ที่สุด
3. **อย่ารายงาน Config 2/4/5 เป็น "sil modeling แย่"** — รายงานเป็น "DP-with-sil limitation" + ระบุว่า F1 ที่ดูสูงกว่า DP-only baseline มาจาก fallback uniform บังเอิญตรง GT structure

ดูรายละเอียดเต็มที่ [ForcedAlignment/PLAN_ForcedAlignment_Task2.md §14.7](ForcedAlignment/PLAN_ForcedAlignment_Task2.md)

### 10.12 DP Alignment — กลไกการทำงานเชิงลึก (สรุปสั้น)

หัวใจของ Phase 5 คือ **Monotonic Token-to-Segment DP** ใน [example_alignment/align_gloss_labels.py:212](example_alignment/align_gloss_labels.py#L212) เนื้อหาเชิงลึก 10 หัวข้อย่อยอยู่ใน [PLAN_ForcedAlignment_Task2.md §15](ForcedAlignment/PLAN_ForcedAlignment_Task2.md) สรุปประเด็นสำคัญ:

**Inputs / Outputs:**

```text
token_embs : (T, 768)   SignCLIP text embedding ต่อ token
sign_embs  : (K, 768)   SignCLIP pose embedding ต่อ SIGN segment
seg_starts/ends : (K,)  เวลาเริ่ม-จบของ segment ทุกอัน
→ output: ranges = [(k_start_i, k_end_i)] ยาว T
   → token i กิน segments [k_start_i .. k_end_i] เป็น consecutive run
```

**Pre-processing 3 ขั้น:**

1. กรอง candidates ตาม `seg_mids ∈ [sent.start − pad, sent.end + pad]` (`window_pad = 0.5 s`)
2. คำนวณ cosine similarity matrix `(T, K)` หลัง row-normalise embeddings ทั้ง 2 ฝั่ง
3. **Per-token softmax** — ทำให้ token แต่ละตัวแข่งบน scale เดียวกัน (token หายากไม่เสียเปรียบ token ทั่วไป)

**DP Recurrence — 3 องค์ประกอบของ cost:**

```text
dp[t, j] = min over k in [t .. j] of:
             dp[t-1, k-1]                                  # ต้นทุนสะสม
           + (-Σ sim[t-1, k-1 .. j-1])                     # similarity reward (- = ยิ่งคล้ายยิ่งดี)
           + gap_penalty × inter_segment_gap(k-1, j-1)     # gap penalty (ห้ามจับ segments ที่ห่างกัน)
           + coverage_penalty × |grp_dur - target_dur|     # coverage penalty (token ไม่ควรกินยาว/สั้นเกิน)
```

- `gap_penalty = 2.0`, `coverage_penalty = 0.5`, `target_dur = sentence_dur / T`
- Cumulative similarity ทำให้คำนวณ range sum O(1) → DP รวม O(T·K²)

**Monotonicity Constraint:**

- `j ≥ t`, `j_max = K − (T − t)`, `k ≤ j` → guarantee ranges ของ tokens **ไม่ overlap** และ **เรียงตามเวลา**

**Backtracking:**

- เลือก `j* = argmin dp[T, T..K]`
- ตามทาง `prev[t, j]` ย้อนกลับเพื่อกู้ `(k_start_i, k_end_i)` ของทุก token
- Prediction's start/end = เวลาจริงของ `seg_starts[k_start]` และ `seg_ends[k_end]` (ไม่ใช่ค่าเฉลี่ย)

**Fallback Path (uniform split):**

ตกเป็น fallback เมื่อ: `K < T` หลังขยาย window, `sign_embs` หาย, SIGN tier ว่าง, หรือ `dp[T, j*] = ∞`
→ แบ่งช่วงเวลาประโยคเท่า ๆ กันให้ T tokens (กรณีนี้เกิดกับ 57/1,132 clips)

**Performance:**

- T = 1–5, K = 3–8 ในทาง practice → DP per clip < 1 ms
- รวม 1,132 × 5 configs ≈ 60 นาที (ตรงกับ estimate Phase 5 ใน §10.5)

**Limitations เชิงทฤษฎี (5 จุด):**

1. **Hard monotonicity** — ผู้ส่งสัญลักษณ์สลับลำดับคำจะ assign ผิด
2. **Per-token softmax** ทำให้เปรียบเทียบ absolute similarity ระหว่าง tokens ไม่ได้
3. **Fixed target_dur** = sentence_dur / T สมมุติทุก token ยาวเท่ากัน → กระทบ Config 4–5 ที่ sil tokens สั้นกว่าค่ากลางมาก
4. **ไม่มี per-segment likelihood แบบ HMM** — sim เป็น cosine ของ embedding ไม่ใช่ posterior
5. **Fallback แบบ crude** — uniform split ไม่ดู pose ของ clip ปรับปรุงได้ด้วย hand-motion peaks

ดูสูตรเต็ม, ตัวอย่าง step-by-step ของ clip 9 ("สบาย ดี"), ตาราง hyperparameter sensitivity, complexity analysis และ code reference ใน [PLAN_ForcedAlignment_Task2.md §15](ForcedAlignment/PLAN_ForcedAlignment_Task2.md)

### 10.13 Deliverables (เสร็จ 24 พ.ค. 2569)

**Code (สร้างใหม่):**

- `ForcedAlignment/run_forced_alignment.py` (792 บรรทัด) — orchestrator 7 phases
- `ForcedAlignment/evaluate_fa_dataset.py` (393 บรรทัด) — positional IoU-only evaluator
- `ForcedAlignment/create_comparison_eafs.py` (489 บรรทัด) — สร้าง ELAN comparison EAFs
- `ForcedAlignment/fix_eaf_media_paths.py` (117 บรรทัด) — ซ่อม MEDIA_DESCRIPTOR ให้ ELAN GUI เปิดได้
- `ForcedAlignment/check_eaf_video_match.py` (247 บรรทัด) — verify EAF ↔ video stem + deep MEDIA_DESCRIPTOR audit
- `ForcedAlignment/fill_gloss_labeling_template.py` — กรอก docx report จาก evaluation summary

**Data Artifacts:**

- `ForcedAlignment/output/poses/{1..1132}.pose` — 1,132 pose files
- `ForcedAlignment/output/seg/E4s-1_30_50/{...}.eaf` — 1,132 SEA segmentation EAFs
- `ForcedAlignment/output/emb/{...}.npy` — 1,075 SignCLIP embeddings (57 fallback)
- `ForcedAlignment/output/predictions/config{1..5}_*.csv` — 5 prediction CSVs
- `ForcedAlignment/output/predicted_eafs/{...}.eaf` — 1,132 EAFs with all 5 prediction tiers
- `ForcedAlignment/output/comparison_eafs/{...}.eaf` — 1,132 EAFs with `cfgN_GT_*`, `cfgN_PRED_*`, `cfgN_EVAL_*` tiers + `comparison_index.csv` + `README_tiers.md`
- `ForcedAlignment/output/evaluation/eval_config{1..5}.csv` + `evaluation_summary.csv`

**Documentation:**

- `ForcedAlignment/PLAN_ForcedAlignment_Task2.md` (1,040+ บรรทัด) — แผน + verification + execution status + DP deep-dive (§15)
- `ForcedAlignment/Gloss_Labeling_Report_Filled.docx` — final report พร้อมส่ง:
  - §1 Overview (model, dataset, method, token delimiter)
  - §2 ผลแต่ละ config 1–6 (#6 marked `*` เป็น equivalent กับ #3)
  - §3 ตารางสรุป Precision/Recall/F1/Accuracy
  - §4 Analysis & Conclusion — best config = #1, Config #3 root cause explained, recommendations 4 ข้อ
  - Footer note: IoU-only positional matching, no text-leakage

**Open Items (ไม่บล็อก deliverable):**

- ทดสอบ DP hyperparameter sweep ถ้าต้องการ Config 4–5 ดีขึ้น (sweep `gap_penalty`, `coverage_penalty`, `target_dur` policy)
- ปรับ SEA segmentation threshold หรือใช้ pose-based fallback เพื่อลด 57 fallback clips
- ถ้าจะ publish ผลเชิงเปรียบเทียบกับ 04.mp4 ต้องระวังเรื่อง annotation convention (§10.11 ข้อ 2)

---

## 11. Fine-tuning Opportunities — จะทำให้ดีขึ้นได้อย่างไร

> ระบบปัจจุบัน (C_MULTI) เป็น **zero-shot cross-lingual transfer** — ใช้โมเดลที่ train บน BSL/Multilingual โดยตรงกับ TSL โดยไม่ปรับอะไรเลย ส่วนนี้วิเคราะห์ว่าถ้าลงทุน fine-tune ส่วนใดสักส่วน จะคาดหวังผลดีขึ้นแค่ไหนและต้องใช้ข้อมูลอะไร

---

### 11.1 ภาพรวม — จุดอ่อนหลักของระบบปัจจุบัน

ก่อนจะ fine-tune ต้องรู้ว่าระบบพังตรงไหน:

| จุดอ่อน | สาเหตุ | ผลกระทบ |
| --- | --- | --- |
| Segmenter (E4s-1) train บน BSL | movement pattern ของ TSL ต่างจาก BSL | บาง sign segments miss หรือแตกผิดตำแหน่ง |
| SignCLIP ไม่เคย "เห็น" Thai Gloss text | vocabulary TSL ไม่อยู่ใน training data | text embedding ของ Gloss tokens มีคุณภาพต่ำกว่าที่ควร |
| DP bias ตั้งแบบ iterative guess | ไม่มีตัวอย่าง TSL จำนวนมากพอ | bias 1.3s/1.0s อาจไม่เหมาะกับ video อื่น |
| stdev start offset สูง (~5.5 s) | outlier cues ที่ segmenter miss | ผลดีโดยเฉลี่ยแต่ unstable รายตัว |

---

### 11.2 Fine-tune E4s-1 Segmenter บน TSL

**E4s-1** คือ GRU binary classifier ที่ตรวจจับ sign boundary — train บน BOBSL (BSL) เท่านั้น

#### ทำไมถึงช่วย

TSL มีลักษณะท่ามือที่แตกต่างจาก BSL:

- **resting position** ต่างกัน — ผู้แปล TSL มักวางมือต่ำกว่าและมีช่วงพักนานกว่า
- **movement velocity** ต่างกัน — ท่าบางท่าในภาษาไทยเร็วหรือช้ากว่า BSL
- **hand dominance pattern** — ผู้แปลบางคนใช้มือซ้าย/ขวาต่างกับ corpus ที่ train

เมื่อ segmenter ตัด boundary ผิด → DP ได้ segment ที่ผิด → alignment offset สูง

#### ข้อมูลที่ต้องการ

```text
input:  pose sequences (543 landmarks × frames) จากวิดีโอ TSL
label:  binary per-frame: 1 = กำลังแสดงท่ามือ, 0 = หยุดพัก/เปลี่ยนท่า

แหล่งข้อมูล:
  - ForcedAlignment dataset (1,132 clips, avg 5.8 s) + Gloss Labeling GT
    → GT ให้ boundary ของแต่ละ sign → แปลงเป็น per-frame label ได้
  - Gloss Labeling tier ของ 04.mp4 (852 entries) — เล็กไปสำหรับ fine-tune
  - ยิ่งมีคลิปมากยิ่งดี เป้าหมาย: ≥ 500 คลิปสำหรับ fine-tune + 200 สำหรับ validate
```

#### วิธี fine-tune

```python
# E4s-1 architecture: GRU encoder → linear binary output
# โมเดลเดิม: train บน BOBSL (BSL data)

# Fine-tune approach 1 — Full fine-tune (ถ้ามีข้อมูลมาก ≥ 1,000 examples)
model.load_pretrained("E4s-1_bsl.pt")
model.train()  # unfreeze ทุก layer

# Fine-tune approach 2 — Adapter / last-layer fine-tune (ถ้าข้อมูลน้อย ~500 examples)
model.load_pretrained("E4s-1_bsl.pt")
for param in model.gru.parameters():
    param.requires_grad = False  # freeze GRU
model.classifier.train()  # เฉพาะ final linear layer
```

**ผลที่คาดหวัง:** segment count ถูกต้องขึ้น → DP มี candidates ที่ดีขึ้น → stdev start offset ลด

---

### 11.3 Fine-tune SignCLIP บน Thai Gloss Vocabulary

**SignCLIP** คือ multimodal encoder — ฝั่ง text มี tokenizer + transformer ที่ train บน multilingual corpus แต่ **Thai Gloss tokens** (`ผายมือ`, `เรียงลำดับ`, ฯลฯ) ไม่ใช่คำภาษาทั่วไป → embedding อาจอ่อนแอ

#### ทำไม SignCLIP Fine-tune ถึงช่วย

ผลที่ได้แล้วว่า Gloss text > CC text สำหรับ embedding input บ่งชี้ว่า **semantic signal ของ Gloss tokens มีประโยชน์** แต่ยังไม่ถึงศักยภาพเต็มที่ เพราะ tokenizer ของ SignCLIP ไม่เคยเห็น Thai Gloss vocabulary

```text
ตัวอย่าง:
  คำว่า "ผายมือ" อาจถูก tokenize เป็น ["ผา", "##ยมือ"] (subword)
  → embedding ไม่ตรงกับท่ามือจริง

หลัง fine-tune บน (Thai Gloss, TSL pose) pairs:
  "ผายมือ" → token embedding ที่ใกล้กับ pose ของท่า "ผายมือ" จริงๆ
```

#### ข้อมูลที่ต้องการสำหรับ SignCLIP

```text
input pairs: (Gloss token, pose segment) คู่ที่ verified match กัน
  ← สร้างได้จาก: Gloss Labeling GT + SIGN segments จาก segmenter
     เช่น  ("ผายมือ", seg_780.pose_data)  → positive pair
           ("ผายมือ", seg_999.pose_data)  → negative pair

เป้าหมาย: ≥ 5,000 positive pairs (852 unique entries × data augmentation)
  - ForcedAlignment dataset ให้ขยาย scale ได้มาก (~6,000 entries จาก 1,132 clips)
```

#### วิธี fine-tune SignCLIP

```text
Loss: InfoNCE (contrastive) — เหมือน CLIP training
  minimize  −log [ exp(sim(text_i, sign_i) / τ)
                 / Σ_j exp(sim(text_i, sign_j) / τ) ]

Option A: Fine-tune text encoder only (เร็วกว่า, data น้อยกว่า)
  → ปรับ text embedding ให้ใกล้ TSL sign embeddings
  → sign encoder ยังคงเหมือนเดิม

Option B: Fine-tune ทั้งสองฝั่ง (ดีกว่า, ต้องการ data มากกว่า)
  → ทั้ง text และ sign encoder ปรับพร้อมกัน
```

**ผลที่คาดหวัง:** similarity matrix มีค่า separation ชัดขึ้น → DP ตัดสินใจได้ดีขึ้น → ±1s coverage จาก 73.9% อาจขึ้นถึง 80–85%

---

### 11.4 ปรับ DP Hyperparameters อย่างเป็นระบบ (Hyperparameter Search)

C_MULTI ใช้ค่า weights ที่ได้จาก manual tuning บน `04.mp4` เพียงวิดีโอเดียว ซึ่งอาจ overfit กับ video นั้น

#### Parameters ที่ควร sweep

| Parameter | C_MULTI ปัจจุบัน | Range ที่ควรลอง | ผลกระทบ |
| --- | ---: | --- | --- |
| `dp_gap_penalty_weight` | 8 | 4–16 | ลงโทษ group ที่มี gap — สำคัญที่สุด |
| `dp_duration_penalty_weight` | 2 | 1–6 | บังคับ duration match |
| `similarity_weight` | 6 | 3–12 | น้ำหนัก semantic vs timing |
| `dp_window_size` | 40 | 20–80 | search window ของ DP |
| `pr_subs_delta_bias_start` | 1.3 s | 0.5–2.5 s | sign delay compensation |
| `dp_max_gap` | 6 | 3–10 | จำกัด gap ภายใน group |

#### วิธีทำ

```text
ต้องการ validation set หลายวิดีโอ (ไม่ใช่ 04.mp4 เพียงวิดีโอเดียว):
  - อย่างน้อย 5–10 วิดีโอ TSL พร้อม CC_Aligned GT ที่ทำด้วยมือ
  - Grid search หรือ Optuna (Bayesian optimization) บน validation set
  - ค่า optimal อาจต่างจาก C_MULTI ปัจจุบัน

คาดว่า:
  - gap_penalty สูงขึ้น (>8) → ลด outlier cues ที่ DP ตัดสินใจผิด
  - bias ปรับตามลักษณะของผู้แปลแต่ละคน (per-interpreter bias)
```

**ผลที่คาดหวัง:** ลด stdev start offset (~5.5 s → ~3–4 s) โดยไม่ต้อง retrain โมเดลใด

---

### 11.5 Fine-tune แบบ End-to-End (ขั้นสูง)

หากมีข้อมูล TSL เพียงพอ สามารถ fine-tune ทั้ง pipeline พร้อมกัน:

```text
ระดับ 1 (ง่ายสุด): ปรับ DP weights เท่านั้น           → ใช้ข้อมูล ~5 วิดีโอ
ระดับ 2:           fine-tune SignCLIP text encoder   → ใช้ข้อมูล ~1,000 pairs
ระดับ 3:           fine-tune E4s-1 segmenter         → ใช้ข้อมูล ~500 annotated clips
ระดับ 4 (ยากสุด):  fine-tune SignCLIP ทั้ง 2 encoder  → ใช้ข้อมูล ~5,000 pairs
```

สำหรับ **zero-additional-annotation** path — ForcedAlignment dataset (1,132 clips) มี Gloss Labeling GT ซึ่งเพียงพอสำหรับระดับ 2–3 โดยไม่ต้องทำ annotation ใหม่

---

### 11.6 สรุป Priority

| Priority | Fine-tune อะไร | ความยาก | ข้อมูลที่ต้องการ | คาดว่าช่วยได้ |
| ---: | --- | --- | --- | --- |
| 1 | DP hyperparameter sweep | ง่าย | 5–10 วิดีโอ TSL + GT | ลด stdev, เสถียรขึ้น |
| 2 | SignCLIP text encoder (TSL Gloss) | ปานกลาง | ~1,000 (Gloss, pose) pairs | ±1s coverage +5–10 pp |
| 3 | E4s-1 segmenter (TSL) | ปานกลาง | ~500 annotated clips | ลด segment boundary error |
| 4 | Per-interpreter bias calibration | ง่าย | 1 วิดีโอต่อผู้แปล | ลด systematic offset |
| 5 | Full SignCLIP fine-tune (ทั้ง 2 encoder) | ยาก | ~5,000 pairs | ผลดีที่สุดระยะยาว |

> **ข้อสรุป:** เริ่มจาก Priority 1 (DP sweep) ก่อน — ไม่ต้อง train ใหม่, เห็นผลเร็ว, ช่วย quantify ว่า parameter sensitivity อยู่ที่ใด แล้วค่อยลงทุน Priority 2–3 หลังมี ForcedAlignment results เป็น validation base

---

## 12. อ้างอิง

- **SEA Paper:** [arXiv:2512.08094](https://arxiv.org/abs/2512.08094) — Jiang, Z., Jang, Y., Momeni, L., Varol, G., Ebling, S., & Zisserman, A. (2025)
- **SignCLIP:** [EMNLP 2024](https://aclanthology.org/2024.emnlp-main.518/)
- **Linguistic Segmentation (E4s-1):** [EMNLP 2023 Findings](https://aclanthology.org/2023.findings-emnlp.846/)
- **pose-format:** [sign-language-processing/pose](https://github.com/sign-language-processing/pose)
- **ELAN:** [archive.mpi.nl/tla/elan](https://archive.mpi.nl/tla/elan)
- **SEA Repository:** [J22Melody/SEA](https://github.com/J22Melody/SEA)
- **SignCLIP Repository:** [J22Melody/fairseq](https://github.com/J22Melody/fairseq)
