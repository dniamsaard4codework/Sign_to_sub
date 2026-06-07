# Sign_to_sub — รายงานความคืบหน้าโครงการเชิงลึก

> **การจัดเรียงคำบรรยายภาษามือไทย (TSL) ด้วย SEA (Segment, Embed, Align)**
>
> บันทึกแบบละเอียดทีละขั้นตอนของทุกสิ่งที่ทำในโปรเจกต์นี้ ตั้งแต่ commit แรกจนถึงล่าสุด
> เอกสารนี้อธิบายว่า SEA คืออะไร (เชิงลึก), ทำไมโปรเจกต์นี้ถึงมีอยู่, สร้างและแก้อะไรไปบ้าง,
> ทดลองอะไรไปบ้างทุกการทดลอง, ผลลัพธ์ทุกตัว, และแก้ปัญหาอะไรไปบ้าง —
> เรียงตามลำดับเวลาและสถาปัตยกรรมอย่างละเอียด
>
> - **ผู้ดูแล:** dniamsaard4codework
> - **ผู้สนับสนุน / เจ้าของข้อมูล:** NECTEC
> - **ฐานต้นทาง (upstream):** [J22Melody/SEA](https://github.com/J22Melody/SEA) ([arXiv:2512.08094](https://arxiv.org/abs/2512.08094), Jiang et al. 2025) + SignCLIP ([J22Melody/fairseq](https://github.com/J22Melody/fairseq))
> - **เรียบเรียง:** 7 มิ.ย. 2569
> - **เอกสารคู่กัน:** [README.md](README.md) (คู่มือ setup หลัก), [Big_Progress.md](Big_Progress.md) (เอกสารอ้างอิงหลักภาษาไทย), [PROJECT_PROGRESS_DEEP.md](PROJECT_PROGRESS_DEEP.md) (ฉบับภาษาอังกฤษ), [ForcedAlignment/PLAN_ForcedAlignment_Task2.md](ForcedAlignment/PLAN_ForcedAlignment_Task2.md)

---

## สารบัญ

1. [บทสรุปผู้บริหาร — โปรเจกต์นี้คืออะไร](#1-บทสรุปผู้บริหาร)
2. [ปัญหา — ทำไมการจัดเรียงคำบรรยายภาษามือถึงสำคัญ](#2-ปัญหา)
3. [อธิบาย SEA เชิงลึก — Segment, Embed, Align](#3-อธิบาย-sea-เชิงลึก)
4. [ข้อมูลนำเข้า — วิดีโอ, ELAN/EAF, VTT, Tiers](#4-ข้อมูลนำเข้า)
5. [คำศัพท์และแนวคิดสำคัญ](#5-คำศัพท์และแนวคิดสำคัญ)
6. [การปรับ Environment และ Platform (Windows + Blackwell GPU)](#6-การปรับ-environment-และ-platform)
7. [สถาปัตยกรรม Pipeline (S → E → A ตั้งแต่ต้นจนจบ)](#7-สถาปัตยกรรม-pipeline)
8. [แกน Dynamic Programming (คณิตศาสตร์ของ Task 1 และ Task 2)](#8-แกน-dynamic-programming)
9. [ลำดับเวลา — ทีละ Commit](#9-ลำดับเวลา)
10. [Task 1 — Subtitle Alignment: 7 การทดลองและผลลัพธ์](#10-task-1)
11. [Task 2 — Gloss Labeling: Prototype, Ablation, Per-Sentence](#11-task-2)
12. [ForcedAlignment — ขยายสเกลเป็น 1,132 คลิป](#12-forcedalignment-ขยายสเกล)
13. [ระเบียบวิธีการประเมินผล (เชิงลึก)](#13-ระเบียบวิธีการประเมินผล)
14. [เครื่องมือและ Script ที่สร้างขึ้น](#14-เครื่องมือและ-script-ที่สร้างขึ้น)
15. [ปัญหาที่เจอและวิธีแก้](#15-ปัญหาและวิธีแก้)
16. [ข้อค้นพบสำคัญ, คำแนะนำ, แนวทางพัฒนาต่อ](#16-ข้อค้นพบและคำแนะนำ)
17. [สิ่งส่งมอบและรายการ Artifact](#17-สิ่งส่งมอบและรายการ)
18. [อ้างอิง](#18-อ้างอิง)

---

## 1. บทสรุปผู้บริหาร

**เป้าหมาย** รับวิดีโอล่ามภาษามือไทย (TSL) พร้อมคำบรรยายภาษาพูด แล้วสร้างคำบรรยายที่
timestamp ตรงกับ *ช่วงที่ล่ามแสดงท่ามือจริง* — ไม่ใช่ตอนที่ผู้พูดพูด เพราะผู้ชมที่หูหนวก/
บกพร่องทางการได้ยินอ่านจากท่ามือ ดังนั้นเวลาของคำบรรยายต้องตามมือ ไม่ใช่ตามเสียง

**แนวทาง** ใช้เฟรมเวิร์ก **SEA (Segment, Embed, Align)** ของ Jiang et al. (2025) เป็นเครื่องยนต์
แล้วทำงานวิศวกรรมจำนวนมากเพื่อ (ก) รันบน **Windows + GPU RTX 5060 Ti สถาปัตยกรรม Blackwell**,
(ข) รองรับ **ภาษามือไทย** ผ่านโมเดล SignCLIP แบบ multilingual, และ (ค) ขยายจากการจัดเรียง
คำบรรยายทั่วไปเป็น **สองงานหลักที่ชัดเจน**

**สองงานที่นิยามและแก้ได้:**

| งาน | จัดเรียงอะไร | ระดับ | ผลลัพธ์ดีที่สุด |
| --- | --- | --- | --- |
| **Task 1 — Subtitle Alignment** | คำบรรยายทั้งประโยค → ช่วงเวลาแสดงท่ามือ | ระดับประโยค | **C_MULTI**: mean start offset **−0.16 วิ**, **100%** อยู่ใน ±3 วิ, **F1@0.5 = 88.2%** |
| **Task 2 — Gloss Labeling** | gloss token แต่ละตัว → ท่ามือเดี่ยวแต่ละท่า | ระดับ token/ท่าทาง | **tier Gloss**: Mean IoU **0.49**, 48% IoU≥0.5 (คลิปเดียว); ขยายเป็น 1,132 คลิปได้ Config #1 F1 **68.6%**, mIoU **0.59** |

**สเกล** พิสูจน์วิธีบน **วิดีโอตัวอย่าง 1 คลิป** (`04.mp4`, คลิป TSL ยาว 11.07 นาที, 119 cue) ก่อน
แล้ว **ขยายเป็นคลัง TSL 1,132 คลิป** (~110 นาที, วิดีโอ ~8.5 GB) ผ่าน orchestrator แบบ phase
ที่รัน ~11–12 ชั่วโมงข้ามคืน

**เอกสาร** repo มีงานเขียนจำนวนมาก: [README.md](README.md) สองภาษา 78 KB,
[Big_Progress.md](Big_Progress.md) ภาษาไทย 126 KB, progress report ภาษาไทย 5 ฉบับตามวันที่,
สไลด์นำเสนอ, คู่มือ pipeline ภาษาไทย, รายงาน LaTeX (PDF อังกฤษ+ไทย), deep-dive HTML/PDF
และ source ของ paper SEA

**Reproducibility** มีการ commit cached output ไว้ ทำให้ใครก็รัน *เฉพาะ evaluation* ได้ใน ~30
วินาที โดยไม่ต้องมีวิดีโอต้นทาง (ที่ gitignore ไว้) หรือ model weights — ตรวจสอบแล้วว่า reproduce
ได้ตรง bit-for-bit ณ 1 มิ.ย. 2569

---

## 2. ปัญหา

### 2.1 "Sign delay" — ทำไม timestamp จากเสียงถึงผิดสำหรับล่ามภาษามือ

เมื่อรายการหรือวิดีโอถูกแปลเป็นภาษามือ ล่ามจะดู/ฟังต้นฉบับ *ก่อน* แล้วค่อยแสดงท่ามือ ทำให้เกิด
**ความหน่วง (lag)**: ท่ามือของแนวคิดหนึ่งจะปรากฏบนจอ *หลัง* คำพูดที่ตรงกันไปหลายวินาที คำบรรยาย
ปิด (CC) มาตรฐานจับเวลาตาม **เสียง** ถ้านำ timestamp ของ CC มาวางบนวิดีโอภาษามือตรงๆ ข้อความ
กับมือจะไม่ตรงกัน — มักคลาดเคลื่อน 1–4 วินาที — ซึ่งกระทบกลุ่มคนที่พึ่งช่องทางภาพ (หูหนวก/บกพร่อง
การได้ยิน) โดยตรง

**วิธีแก้** คือ *ปรับเวลาใหม่* ให้แต่ละคำบรรยายตรงกับช่วงที่ล่ามแสดงท่ามือของเนื้อหานั้นจริง — การ
ปรับเวลานี้คือปัญหาที่โปรเจกต์นี้แก้

### 2.2 ทำไมถึงยาก

- **ไม่มีจุดยึดระดับคำจากเสียง** — ไม่มี forced aligner ระหว่าง transcript กับเสียงมาช่วย เพราะ
  modality เป็นมือที่มองเห็น ไม่ใช่เสียงพูด
- **ขอบเขตของท่ามือคลุมเครือ** — ท่ามือต่อเนื่องกลืนกัน (coarticulation) ไม่มีตัวคั่นแบบช่องว่างใน
  ภาษาพูด
- **ช่องว่างข้ามภาษา** — โมเดลภาษามือสำเร็จรูป train บน BSL (อังกฤษ) หรือ ASL (อเมริกัน) ไม่ใช่ไทย
  จึงต้อง transfer
- **เรียงลำดับเดียวกันแต่ไม่ 1:1** — ลำดับคำบรรยายตามลำดับท่ามือ (monotonic) แต่หนึ่งคำบรรยาย
  อาจครอบหลายท่ามือ และบางท่ามือไม่มีคำบรรยาย

### 2.3 ทำไมเลือก SEA

SEA **ไม่ผูกกับภาษา**, **ไม่ต้อง train ตอน inference** และ **แยกส่วนได้**: ใช้โมเดล pretrain สองตัว
(*segmenter* และ *embedder*) แล้วแก้การจัดเรียงเป็น **global optimization ด้วย dynamic programming**
ที่รันเสร็จในไม่กี่วินาทีบน CPU เหมาะกับภาษาทรัพยากรน้อยอย่าง TSL ที่ train aligner แบบ end-to-end
จากศูนย์ไม่ได้

---

## 3. อธิบาย SEA เชิงลึก

### 3.1 ตัว paper สรุปย่อหน้าเดียว

> *"SEA (Segment, Embed, and Align) เป็นเฟรมเวิร์กเดียวที่ทำงานข้ามหลายภาษาและโดเมน ใช้โมเดล
> pretrain สองตัว: ตัวแรก segment ลำดับเฟรมวิดีโอเป็นท่ามือเดี่ยวๆ ตัวที่สอง embed คลิปวิดีโอของแต่ละ
> ท่ามือเข้าสู่ latent space ร่วมกับข้อความ จากนั้น alignment ทำด้วย dynamic programming ที่เบาและรัน
> บน CPU ได้เร็วในไม่ถึงนาที แม้กับ episode ยาวเป็นชั่วโมง"*
> — Jiang, Jang, Momeni, Varol, Ebling, Zisserman (arXiv:2512.08094, 2025)

แก่นคือ **ไม่ train aligner** แต่ (1) ตัดวิดีโอเป็นหน่วยท่ามือ, (2) ฉายทั้งท่ามือและข้อความเข้าสู่
vector space เดียวกันเพื่อวัดความคล้ายระหว่างท่ามือ↔ข้อความ และ (3) ให้ DP หา assignment ระหว่าง
ประโยค-กับ-ท่ามือที่ดีที่สุดทั้งภาพ โดยเคารพลำดับ monotonic, ระยะเวลา และช่องว่าง

### 3.2 สามขั้นตอน

```
        วิดีโอ (เฟรม)                          ข้อความ (cue คำบรรยาย)
            │                                      │
   ┌────────▼─────────┐                            │
   │  S — SEGMENT     │  pose → ขอบเขตแต่ละท่ามือ  │
   │  (E4s-1 GRU)     │                            │
   └────────┬─────────┘                            │
            │ SIGN segments [(t0,t1), ...]         │
   ┌────────▼─────────┐                  ┌─────────▼──────────┐
   │  E — EMBED       │  SignCLIP        │  E — EMBED         │
   │  ท่ามือ → 768-d  │  space ร่วม      │  ข้อความ → 768-d   │
   └────────┬─────────┘                  └─────────┬──────────┘
            │  vector ท่ามือ                       │ vector cue
            └──────────────┬───────────────────────┘
                  cosine similarity ท่ามือ↔ข้อความ
                           │
                  ┌────────▼─────────┐
                  │  A — ALIGN       │  global DP (Numba @njit)
                  │  ลด cost รวม     │  monotonic, มี window
                  └────────┬─────────┘
                           │
                  คำบรรยายที่จัดเรียงแล้ว (VTT/EAF ที่ปรับเวลาใหม่)
```

#### S — Segment (เปลี่ยนสตรีม pose เป็นรายการท่ามือ)

- **อินพุต:** ไฟล์ `.pose` — landmark MediaPipe Holistic (543 จุด/เฟรม: ลำตัว มือ ใบหน้า) สกัด
  จาก MP4
- **โมเดล:** segmenter ภาษาศาสตร์ **E4s-1** (bidirectional GRU อ่านลำดับ pose แล้วทำนายต่อเฟรม
  ว่าเป็นจุด **B**egin ของท่ามือ และความน่าจะเป็น **O**ut/อยู่ในท่ามือ) มาจาก package
  `pose_to_segments` / `J22Melody/segmentation@bsl` และ train บนข้อมูล **BOBSL (BSL)**
- **Threshold:** `--sign-b-threshold 30 --sign-o-threshold 50` (ที่มาของ "30_50" ในชื่อโฟลเดอร์
  `E4s-1_30_50`) ค่ายิ่งต่ำ = segment ยิ่งมากและละเอียด
- **เอาต์พุต:** `.eaf` ของ ELAN ที่มีสอง tier — tier `SIGN` (เช่น **2,780** segment บนวิดีโอตัวอย่าง)
  และ tier ที่หยาบกว่า `SENTENCE` (**418**) แต่ละ `SIGN` segment คือช่วงเวลา `(start, end)` ของท่ามือ
  หนึ่งท่า

#### E — Embed (วางท่ามือและข้อความใน vector space เดียวกัน)

- **โมเดล:** **SignCLIP** — dual encoder สไตล์ CLIP *sign/pose encoder* แปลงคลิป pose สั้นๆ เป็น
  vector 768 มิติ; *text encoder* แปลงคำบรรยายที่ tokenize แล้วเป็น vector 768 มิติ ทั้งคู่ train แบบ
  contrastive ให้ท่ามือและข้อความที่ตรงกัน **อยู่ใกล้กัน** ใน space เดียวกัน
- **โมเดล 3 ตัว** ที่ต่อไว้ใน repo นี้:

  | ตัว | train บน | `--language_tag` |
  | --- | --- | --- |
  | `bsl` | ภาษามืออังกฤษ | `<en> <bfi>` |
  | `multilingual` | หลายภาษามือ (ใช้กับ **TSL**) | `<en>` |
  | `asl` | ภาษามืออเมริกัน | `<en> <ase>` |

- **เอาต์พุต:**
  - *Sign embedding* `segmentation_embedding/<model>/04.npy` → shape **(2780, 768)** — หนึ่ง
    vector ต่อ `SIGN` segment
  - *Subtitle embedding* `subtitle_embedding/<model>/04.npy` → shape **(119, 768)** — หนึ่ง
    vector ต่อ cue คำบรรยาย
- ขั้นนี้คือขั้นที่ต้องใช้ GPU (forward pass ของ SignCLIP) ที่เหลือรันบน CPU ได้

#### A — Align (dynamic programming บน similarity + เวลา)

เมื่อมี sign segments, vector ของมัน และ vector คำบรรยาย SEA คำนวณ **เมทริกซ์ cosine similarity**
ระหว่าง cue กับ sign แล้วรัน **dynamic program** ที่ assign cue แต่ละตัว (ตามลำดับ) ให้กับกลุ่ม sign
segment ที่ติดกัน โดยลด cost ที่สมดุลระหว่าง: อยู่ใกล้ timestamp เดิม, ระยะเวลาตรงกัน, เลี่ยงช่องว่าง
ระหว่าง segment ที่จัดกลุ่ม และ **ให้รางวัลเมื่อ similarity ท่ามือ↔ข้อความสูง** (คณิตศาสตร์เต็มใน
[§8](#8-แกน-dynamic-programming))

### 3.3 ทำไม SEA เหมาะกับ TSL โดยเฉพาะ

- **Cross-lingual transfer:** `multilingual` SignCLIP แม้ไม่เคย train บนภาษาไทย ก็ embed ท่ามือ
  TSL เข้าสู่ space ที่ใกล้พอกับข้อความอังกฤษ ทำให้ cosine similarity มีความหมาย — โปรเจกต์นี้ยืนยัน
  เชิงประจักษ์: `multilingual` ชนะทั้ง `bsl` และ `asl` บน TSL ตัวอย่าง (ดู [§10](#10-task-1))
- **ไม่ต้องมีข้อมูล train TSL** สำหรับ inference เรามี annotation ไว้ *ประเมินผล* แต่ตัว aligner ไม่
  train เลย
- **เร็วและเป็นมิตรกับ CPU** ในขั้น DP (Numba `@njit` เสร็จใน <1 วิ/วิดีโอ)

---

## 4. ข้อมูลนำเข้า

### 4.1 ไฟล์หลัก (วิดีโอตัวอย่าง)

| ไฟล์ | คืออะไร | อยู่ใน git? |
| --- | --- | --- |
| `example_alignment/04.mp4` | วิดีโอ TSL ต้นทาง "การเปรียบเทียบและเรียงลำดับ" **11.07 นาที**, 25 FPS, ~80 MB | ❌ (ข้อมูล NECTEC) |
| `example_alignment/04.pose` | pose MediaPipe Holistic, ~358 MB | ❌ (สร้างใหม่ได้) |
| `example_alignment/Test.eaf` | annotation ELAN, 660 KB — เก็บทุก tier | ✅ |

### 4.2 ELAN / EAF — คืออะไร

**ELAN** เป็นเครื่องมือมาตรฐานในการ annotate วิดีโอเชิงภาษาศาสตร์ เก็บ annotation ในไฟล์
**`.eaf`** (XML) ซึ่งประกอบด้วย:

- บล็อก `TIME_ORDER` ของ `TIME_SLOT` (แต่ละตัวคือ timestamp หน่วยมิลลิวินาที),
- หนึ่งหรือหลาย **`TIER`** แต่ละ tier เป็น track ของ `ANNOTATION` ที่มีชื่อ; แต่ละ annotation ชี้
  ไปยัง `TIME_SLOT` เริ่ม-จบ และมีข้อความ

tier ในโปรเจกต์นี้อ่านด้วย `xml.etree.ElementTree` ธรรมดา — ต้องเป็น **`<TIER>` ระดับบนสุด** (ไม่ใช่
subtier ซ้อน) จับคู่ด้วย `TIER_ID`

### 4.3 VTT / cue — คืออะไร

**WebVTT (`.vtt`)** เป็นรูปแบบ subtitle บนเว็บ: รายการ **cue** แต่ละตัวเป็นช่วงเวลา `start --> end`
พร้อมข้อความหนึ่งบรรทัด pipeline แปลง EAF tier ⇄ VTT ได้อิสระ; VTT เป็นรูปแบบกลางที่ aligner ของ
SEA รับเข้าและส่งออก

### 4.4 tier ใน `Test.eaf` — และบทบาทที่แน่นอน

| Tier | จำนวน | บทบาท |
| --- | ---: | --- |
| `CC` | 172 | คำบรรยายปิดดิบจากเสียง — **ไม่ใช้ตรงๆ** |
| `CC_Input` | 119 | CC ที่คัดแล้ว — **INPUT ของ Task 1** |
| `CC_Aligned` | 119 | จัดเวลาด้วยมือ — **GROUND TRUTH ของ Task 1** |
| `Gloss` | 119 ประโยค (852 token) | tier gloss — **INPUT ของ Task 2 (แนะนำ)** |
| `Gloss_Input` | 119 ประโยค (889 token) | gloss ที่คัดแล้ว — **INPUT ของ Task 2 (default ในโค้ด)** |
| `Gloss Labeling` | 852 | annotation gloss ระดับท่าทาง — **GROUND TRUTH ของ Task 2** |

โครงสร้าง tier นี้เป็นแกนของทั้งสองงาน: `CC_Input → CC_Aligned` คือคู่ input→GT ของ Task 1
(119↔119 จับคู่ตาม index) และ `Gloss[_Input] → Gloss Labeling` คือคู่ input→GT ของ Task 2

### 4.5 ตัวอย่างเป็นรูปธรรม

- **cue อินพุต** (จาก `CC_Input`): `00:00:12.300 --> 00:00:15.800  "เด็กกำลังเรียนหนังสือ"`
  (จับเวลาตามเสียง)
- **ความจริงของล่าม:** ล่ามยังไม่เริ่มแสดงท่า "เด็ก/เรียน/หนังสือ" จนถึง ~14.0 วิ และจบ ~17.6 วิ
- **เอาต์พุต SEA** (Task 1): cue ปรับเวลาใหม่ `00:00:14.0 --> 00:00:17.6` ตรงกับมือ
- **Task 2** ลงลึกกว่า: จัดเรียง gloss token แต่ละตัว — "ผายมือ", "เด็ก", "เรียน" — กับช่วง `SIGN`
  ท่าเดี่ยวภายในประโยคนั้น

---

## 5. คำศัพท์และแนวคิดสำคัญ

- **Embedding / vector:** รายการตัวเลขความยาวคงที่ (ที่นี่ 768) แทน "ความหมาย" ของคลิปท่ามือหรือ
  ข้อความ ความหมายคล้ายกัน → vector ใกล้กัน
- **Shared latent space:** SignCLIP train sign-encoder และ text-encoder *พร้อมกัน* ให้ท่ามือและ
  ข้อความคำบรรยายของมัน map ไปยัง vector ใกล้กัน **ใน space เดียวกัน** — นี่คือสิ่งที่ทำให้ similarity
  ข้าม modality มีความหมาย
- **Cosine similarity:** `cos(a,b) = (a·b)/(‖a‖‖b‖)` — วัดมุมระหว่าง vector สองตัว อยู่ใน [−1, 1];
  ~1 = "คล้ายมาก" aligner ใช้ค่านี้ให้คะแนนว่าคำบรรยายตรงกับช่วงท่ามือดีแค่ไหน
- **Dynamic Programming (DP):** อัลกอริทึมที่หา sequence ของการตัดสินใจที่ดีที่สุดทั้งภาพ โดยสร้าง
  ตารางของผลย่อยที่ดีที่สุดแล้ว backtrack — ที่นี่ใช้ assign cue→sign ตามลำดับ monotonic ด้วย cost
  รวมต่ำสุด
- **IoU (Intersection over Union):** สำหรับสองช่วงเวลา = `overlap / union` ใน [0,1] IoU=1 หมายถึง
  เวลาตรงกันเป๊ะ; IoU≥0.5 คือเกณฑ์ "match ดี" มาตรฐาน เป็น metric หลักของ Task 2
- **Frame accuracy:** แปลง timeline ที่ทำนายกับ GT เป็นเฟรม (FPS=25) แล้วรายงานสัดส่วนเฟรมที่ป้าย
  ตรงกัน — metric ระดับเฟรมจาก SEA
- **F1@IoU:** ค่าเฉลี่ยฮาร์มอนิกของ precision/recall โดย prediction จะ "นับ" ก็ต่อเมื่อ IoU กับ GT
  เกินเกณฑ์ (0.10 / 0.25 / 0.50)
- **Cross-lingual transfer:** ใช้โมเดลที่ train บนภาษามือหนึ่ง (BSL/ASL/multi) มาประมวลผลอีกภาษา
  (TSL) โดยไม่ retrain — อาศัยโครงสร้างที่ร่วมกัน
- **Gloss:** ป้ายเขียนย่อของท่ามือ (เช่น "เด็ก" = ท่ามือของ "เด็ก") gloss *ประโยค* คือลำดับ gloss
  token ของหนึ่งคำพูด

---

## 6. การปรับ Environment และ Platform

ความพยายามจริงส่วนใหญ่ทุ่มไปกับการทำให้ SEA ต้นทาง (codebase งานวิจัยแบบ Linux/conda, Python
3.12) รันบน **Windows 11 กับ GPU Blackwell ใหม่เอี่ยม**

### 6.1 เครื่องทดสอบ

```
OS              Windows 11 Pro (10.0.26200)
CPU             Intel Core Ultra 7 265K (20 cores)
RAM             64 GB
GPU             NVIDIA GeForce RTX 5060 Ti (16 GB, Blackwell sm_120)
CUDA Driver     595.79 (CUDA 13.2)
Python          3.11.15  (ไม่ใช่ 3.12 — ดูเหตุผลด้านล่าง)
Shell           PowerShell
```

### 6.2 การ pin เวอร์ชันที่สำคัญและ *เหตุผล*

| Package | Pin | เหตุผล |
| --- | --- | --- |
| `python` | **3.11.x** | `mediapipe==0.10.21` ไม่มี wheel สำหรับ 3.12+ |
| `torch` | 2.11.0 **+cu128** | Blackwell sm_120 ต้องใช้ wheel cu128; cu126 จะ fall back ไป CPU เงียบๆ (ช้าลง 10–100×) |
| `mediapipe` | **0.10.21 เป๊ะ** | 0.10.22+ ทำให้ API pose extraction พัง |
| `pose-format` | 0.12.3 | ตรงกับ `videos_to_poses` |
| `numpy` | 1.26.4 | เข้ากับ numba JIT |
| `numba` | ล่าสุด | JIT สำหรับ DP inner loops |

> **กับดัก Blackwell:** การ์ดซีรีส์ 50 *ต้อง* ลง wheel **cu128** ถ้าลงผิด wheel
> `torch.cuda.is_available()` จะคืน `False` แล้วขั้น embedding ทั้งหมดจะรันบน CPU — เปลี่ยนงาน ~5
> นาทีเป็น ~3–5 ชั่วโมง โดยเงียบๆ

### 6.3 การแก้ portability บน Windows ต่อ upstream

**`SEA/` (เทียบ upstream commit `5aaf27d`):**

| ไฟล์ | การเปลี่ยน |
| --- | --- |
| `align.py` | รองรับหลายโมเดล (`--live_model_name`, `--live_language_tag`); โหลด segmentation embedding ที่คำนวณไว้ล่วงหน้า; ลบ path hardcode `/users/zifan/` |
| `align_dp.py` | fallback การ import `numba` — รันเป็น Python ธรรมดาถ้าไม่มี LLVM |
| `config.py` | เพิ่ม CLI arg `--live_model_name`, `--live_language_tag` |
| `segmentation.py` | ใช้ `os.path.abspath()`; เปลี่ยน `subprocess.run(shell=True)` เป็น `shlex.split` + `shell=False` (ปลอดภัยบน Windows) |

**`fairseq_signclip/` (SignCLIP):** patch 8 ไฟล์ 36 บรรทัด
([patches/fairseq_signclip_windows.patch](patches/fairseq_signclip_windows.patch)) แก้สมมติฐาน path
แบบ Linux (`/` vs `\`) และ absolute path ที่ขาดใน `mmpt/models/mmfusion`,
`processors/{dsprocessor,dsprocessor_sign,processor}`, `tasks/task`, `utils/load_config`, และ
`scripts_bsl/extract_episode_features.py` รวมถึง YAML config ใน `retri/signclip_bsl/` patch ถูก pin
ไว้ที่ upstream commit `a8199440` (4 มี.ค. 2569) ให้ `git apply` reproduce ได้

### 6.4 ความจริงเรื่อง GPU (บันทึกอย่างตรงไปตรงมา)

มีเพียง **1** ใน 4 phase หนักที่ใช้ GPU จริง:

| Phase | เครื่องมือ | อุปกรณ์ | เหตุผล |
| --- | --- | --- | --- |
| Pose | `videos_to_poses` (MediaPipe) | **CPU** | wheel MediaPipe บน Windows ไม่มี GPU delegate |
| Segmentation | `pose_to_segments` | **CPU** | upstream บังคับ `CUDA_VISIBLE_DEVICES=""`; LSTM ที่ JIT freeze CPU device ไว้สำหรับ hidden state — patch ไม่ได้ถ้าไม่ re-export |
| **Embedding** | `extract_episode_features.py` (SignCLIP) | **GPU** | honor `torch.cuda.is_available()` |
| DP align | `align.py` | CPU (โดยตั้งใจ, Numba) | compute เล็ก |

นี่คือเหตุผลที่ full run 1,132 คลิปใช้ ~11 ชั่วโมง — pose extraction บน CPU เป็นคอขวด เรื่องนี้ถูก
สืบหาและบันทึกไว้ ไม่ปิดบัง

---

## 7. สถาปัตยกรรม Pipeline

flow ตั้งแต่ต้นจนจบของวิดีโอตัวอย่าง (`04`):

```
INPUT: 04.mp4  +  Test.eaf
  │
  ├─ A. extract_cc_from_eaf.py --tier CC_Input  →  subtitles/04.vtt          (119 cue)
  ├─ B. make_gloss_cc_vtt.py                     →  subtitles_gloss_cc_time/04.vtt
  │       (ข้อความ Gloss_Input แปะบน timestamp ของ CC_Input)
  ├─ C. videos_to_poses (MediaPipe Holistic)     →  04.pose                  (358 MB)
  ├─ D. SEA/segmentation.py (E4s-1, 30/50)       →  segmentation_output/E4s-1_30_50/04.eaf
  │       (SIGN: 2780, SENTENCE: 418)
  ├─ E1. extract_episode_features.py --mode=segmentation (×3 โมเดล)
  │       →  segmentation_embedding/{sign_clip,sign_clip_multi,sign_clip_asl}/04.npy   (2780,768)
  ├─ E2. extract_episode_features.py --mode=subtitle (×5 combo ข้อความ/โมเดล)
  │       →  subtitle_embedding/<combo>/04.npy                                          (119,768)
  │
  ├─ TASK 1: SEA/align.py  (7 การทดลอง)           →  aligned_output_*/04.vtt
  │       + fix_overlap_vtt.py                     →  aligned_output_*/04_no_overlap.vtt
  │       + evaluate_all_to_csv.py                 →  evaluation_task1_results.csv
  │
  ├─ TASK 2: align_gloss_labels.py --tier Gloss   →  gloss_labels_pred.csv/.vtt + 04_gloss_pred.eaf
  │       + evaluate_gloss_labeling.py            →  evaluation_gloss_labeling.csv
  │
  └─ VISUALS: add_vtt_tiers_to_eaf.py (Test_comparison.eaf, 17 tier)
             add_best_to_eaf.py     (Test_best.eaf)
             plot_alignment.py       (figures/timeline_first_2min.png)
```

**ทำไมมี subtitle input สองตัว (A และ B)?** `CC_Input` เป็นคำบรรยายภาษาพูดตรงๆ ส่วน `Gloss_Input`
เป็นข้อความ *gloss* (ใกล้กับสิ่งที่มือพูดมากกว่า) การ embed ข้อความ gloss ด้วย SignCLIP แล้วจัดเรียง
(การทดลอง **C_MULTI**) ให้ผล Task 1 ดีที่สุด เพราะข้อความ gloss อยู่ใกล้ท่ามือใน space ร่วมมากกว่า
คำบรรยายภาษาไทยพูด

> **จุดพลาดเชิงปฏิบัติที่สำคัญ** (แต่ละข้อเคยเสียเวลา debug, บันทึกไว้แล้ว):
> - `--sign-b-threshold`/`--sign-o-threshold` ตอน align **ต้องตรง** กับค่าตอน segmentation (`30 50`)
>   ไม่งั้นทุก cue จะยุบไปที่เวลาเดียวกัน
> - `--segmentation_dir` ต้องชี้ที่ **parent** (`segmentation_output`) ไม่ใช่ subfolder `E4s-1_30_50`
>   — `align.py` ต่อ subdir ของ threshold เอง
> - `--num_workers 1` บน Windows — multiprocessing >1 จะเจอ path error

---

## 8. แกน Dynamic Programming

### 8.1 DP ของ Task 1 (ระดับประโยค, `align_dp.py`)

State: `dp[i][j]` = cost ต่ำสุดในการ assign cue `1..i` โดย cue `i` จบที่ sign-segment index `j`
cost ของการจัดกลุ่ม cue `i` ครอบ segment `k..j`:

```
C(i, k, j) =  |cue_i.start − seg_k.start|          ← จัดเรียงจุดเริ่ม
            + |cue_i.end   − seg_j.end|            ← จัดเรียงจุดจบ
            + w_D · |cue_dur − group_dur|          ← penalty ระยะเวลา   (w_D = 2)
            + w_G · gap(k, j)                       ← penalty ช่องว่าง   (w_G = 8, สำคัญสุด)
            − w_S · sim_cum[i][k][j]                ← รางวัล similarity  (w_S = 6)
```

- **จัดเรียงเริ่ม/จบ (น้ำหนัก 1):** ดึง cue ที่ปรับเวลาแล้วให้ใกล้ของเดิม (timestamp จากเสียงเป็น
  prior คร่าวๆ ว่าเนื้อหาอยู่ตรงไหน)
- **penalty ระยะเวลา `w_D`:** ลงโทษกลุ่มที่ระยะเวลารวมต่างจากของ cue
- **penalty ช่องว่าง `w_G` (มีอิทธิพลสุด):** ลงโทษการเย็บ sign ที่ไม่ติดกันด้วยช่องว่างใหญ่
- **รางวัล similarity `w_S`:** *ลบ* cost ตามสัดส่วน cosine similarity ของ SignCLIP สะสม — similarity
  ท่ามือ↔ข้อความสูงทำให้การจัดกลุ่มถูกลง นี่คือ "E" ของ SEA ป้อนให้ "A"

ใช้ **sliding window** (`--dp_window_size 40`) จำกัด search ทำให้ complexity เป็น `O(M · W²) ≈ 190K`
operation; ด้วย Numba `@njit` เสร็จใน **<1 วินาที**

น้ำหนัก tuned ของ C_MULTI: `w_D=2, w_G=8, max_gap=6, window=40, w_S=6` พร้อม bias
`pr_subs_delta_bias_start=1.3 / end=1.0` (bias เลื่อน prior เริ่ม/จบ พบจากการทดลองว่าทำให้ offset
อยู่ใกล้ศูนย์)

### 8.2 Post-processing: แก้ overlap

DP **ไม่มี constraint ห้าม overlap** เอาต์พุตดิบจึง overlap 86–88% ของเวลา
[fix_overlap_vtt.py](example_alignment/fix_overlap_vtt.py) clamp รอบเดียว:

```python
for i in range(len(cues) - 1):
    if cues[i].end > cues[i + 1].start:
        cues[i].end = cues[i + 1].start   # clamp จุดจบให้เท่ากับจุดเริ่มถัดไป
```

แตะแค่ **จุดจบ** เท่านั้น — **จุดเริ่ม** เป็น best estimate ที่ DP คำนวณมาอย่างระวังว่าท่ามือเริ่ม
เมื่อไร และ metric หลักวัด start offset ดังนั้น start metric เหมือนกันก่อน/หลังแก้ ขณะที่ overlap ลดเป็น
**0%**

### 8.3 DP ของ Task 2 (ระดับ token, `align_gloss_labels.py`)

สำหรับ gloss *ประโยค* `(start_s, end_s, "tok1 tok2 ... tokT")`:

```
1. tokenize ด้วยช่องว่าง                                  → T token
2. จำกัด SIGN segment ที่เป็น candidate เฉพาะที่ midpoint
   ∈ [start_s, end_s]  (pad ±0.5 วิ ถ้า window ว่าง)       → K segment
3. embed แต่ละ token ด้วย SignCLIP multilingual text encoder → token_embs (T×768), cache .npz
4. sim_matrix (T×K) = cosine, แล้ว row-softmax
5. monotonic DP ต่อประโยค:
     dp[t][j] = min over k of {
         dp[t-1][k-1]
       − Σ sim[t-1, k-1 .. j-1]                  ← similarity เชิงลบ (รางวัล)
       + gap_penalty      · inter_segment_gap     ← (default 2.0)
       + coverage_penalty · |group_dur − sent_dur/T|  ← (default 0.5)
     }
6. backtrack → แต่ละ token ได้ช่วง segment ที่ติดกัน (k_start..k_end)
            → emit (seg[k_start].start, seg[k_end].end, token)
```

Complexity `O(T·K²)` ต่อประโยค (T~7, K~30) — เล็กมาก; ทั้ง 119 ประโยคใน <1 วิ token embedding ถูก
cache เป็น `.npz` รอบสองจึงเร็วขึ้นมาก ใช้ path `--cache` แยกต่อ ablation tier เพื่อไม่ให้ cache ปนกัน

---

## 9. ลำดับเวลา

16 commit ในเวลา ~6 สัปดาห์ (20 เม.ย. 2569 → 31 พ.ค. 2569) บวกงาน doc/verify ต่อเนื่องถึง
มิ.ย. progress report ภาษาไทย 5 ฉบับบันทึกเส้นเรื่องวิจัย

| วันที่ | Commit | หมุดหมาย |
| --- | --- | --- |
| **20 เม.ย. 69** | `143ff35` Initial commit | pipeline แรกที่รันได้: segmentation + baseline **B2 (BSL)**; [Progress_20042026.md](Progress_20042026.md) |
| **26 เม.ย. 69** | `5df0664` Update progress | นิยาม **7 การทดลอง**; แก้ overlap; **Task 2 prototype**; [Progress_26042026.md](Progress_26042026.md) |
| **27 เม.ย. 69** | `d19acab` Update 27042026 | Comparison EAF (`_comparison_27042026`); โครงรายงาน |
| **5 พ.ค. 69** | `cc54922` Update 05052026 | เปลี่ยนเป็น **`CC_Input` / `Gloss_Input`** ที่คัดแล้ว + **index-based eval** (119/119); [Progress_04052026.md](Progress_04052026.md), Presentation update, script.md |
| **10 พ.ค. 69** | `8b02759` Update 10052026 | เตรียม ablation ของ Task 2 |
| **12 พ.ค. 69** | `49d5ced` Update 12052026 | **Task 2 ablation: `Gloss` vs `Gloss_Input`**; [Progress_09052026.md](Progress_09052026.md); presentation_12052026 |
| **16 พ.ค. 69** | (ใน `161f165`) | **Per-sentence Task 2 pipeline** — crop วิดีโอที่ขอบประโยค gloss แล้วรัน pose+seg+emb+DP ต่อ clip; [Progress_16052026.md](Progress_16052026.md) |
| **18 พ.ค. 69** | `7f5213e`, `1b0f15c`, `161f165`, `cc7336c` | รายงานรวม **Big_Progress.md**; เพิ่ม **ForcedAlignment dataset** (1,140 EAF ที่ซ่อมแล้ว); วางแผนขยายสเกล |
| **20–21 พ.ค. 69** | — | **รัน full pipeline 1,132 คลิป** (~11 ชม. ข้ามคืน) |
| **24 พ.ค. 69** | — | สิ่งส่งมอบ ForcedAlignment: comparison EAF, รายงาน docx ที่เติมแล้ว, eval |
| **25 พ.ค. 69** | `7496642` | commit ผล full-run ForcedAlignment; อัปเดต Big_Progress; SEA_Pipeline_Deep_Dive.html |
| **26 พ.ค. 69** | `404d912`, `13f02b8` | รายงาน deep-dive + DeepDive.pdf; **ทำความสะอาด repo** (ลบ log/artifact ที่ gen 6,166 บรรทัด), เพิ่ม `requirements.txt`, `.gitignore` |
| **27 พ.ค. 69** | `d115f34` | ForcedAlignment README (288 บรรทัด) + **workflow patch ของ Windows** (`patches/`) |
| **28 พ.ค. 69** | `ebfb566` | README: ส่วน Source Data, **Evaluation-Only Quick Start**, `requirements-eval.txt` |
| **31 พ.ค. 69** | `9df63cc` | README: ทำให้เส้นทาง setup สำหรับมือใหม่แข็งแรงขึ้น |
| **1 มิ.ย. 69** | (memory) | **ตรวจ reproducibility** — รัน eval script ทุกตัวจาก venv ใหม่, output ตรง CSV ที่ commit ไว้ bit-for-bit; กู้ manifest ที่ถูกตัด |

---

## 10. Task 1 — Subtitle Alignment

### 10.1 นิยาม

ปรับเวลา cue `CC_Input` 119 ตัว (จับเวลาตามเสียง) ให้ตรงกับช่วงที่ล่ามแสดงท่ามือ ประเมินกับ cue
`CC_Aligned` 119 ตัวที่จัดด้วยมือ โดยจับคู่ตาม **index** (`pred[i] ↔ gt[i]`)

### 10.2 วิวัฒนาการของชุดการทดลอง

รุ่นแรก (Progress_20042026) ใช้โมเดล **BSL** กับข้อความ CC ดิบ เมื่อเวลาผ่านไปการศึกษาขยายเป็น
**กริด 7 การทดลอง**: {BSL, Multilingual, ASL} × {ข้อความ CC, ข้อความ Gloss, gloss ระดับคำ} ข้อค้นพบ
สำคัญสองข้อขับเคลื่อนเรื่องนี้:

1. **Multilingual > BSL > ASL** สำหรับ TSL (cross-lingual transfer ทำงานดีที่สุดกับ encoder multilingual)
2. **ข้อความ Gloss > ข้อความ CC** ในการเป็น subtitle ที่ embed (gloss อยู่ใกล้ท่ามือกว่า)

### 10.3 7 การทดลอง

| การทดลอง | โมเดล | ข้อความ | Subtitle emb | bias เริ่ม/จบ | save_dir |
| --- | --- | --- | --- | --- | --- |
| **B2** | BSL | CC | `sign_clip` | 1.8/1.5 | `aligned_output_with_embedding_tuned` |
| **B_MULTI** | Multi | CC | `sign_clip_multi` | 1.8/1.5 | `aligned_output_multi_b2` |
| **C_MULTI** ⭐ | Multi | Gloss | `sign_clip_multi_gloss` | 1.3/1.0 | `aligned_output_multi_gloss` |
| **C_MULTI_word** | Multi | Gloss (ระดับคำ, live emb) | — | 1.3/1.0 | `aligned_output_multi_gloss_word` |
| **D_ASL** | ASL | CC | `sign_clip_asl` | 1.8/1.5 | `aligned_output_asl_b2` |
| **D_ASL_gloss** | ASL | Gloss | `sign_clip_asl_gloss` | 1.3/1.0 | `aligned_output_asl_gloss` |
| **D_ASL_word** | ASL | Gloss (ระดับคำ, live emb) | — | 1.3/1.0 | `aligned_output_asl_gloss_word` |

(variant `_word` embed gloss token แต่ละตัวตอน align ผ่าน `--live_embedding --tokenize_text_embedding`
จึงไม่ต้องมี subtitle `.npy` ที่คำนวณไว้ล่วงหน้า)

### 10.4 ผลลัพธ์ (index-based, 119/119, หลังแก้ overlap)

| การทดลอง | mean off | ±1 วิ | ±2 วิ | ±3 วิ | F1@0.50 | overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B2 | +0.26 วิ | 76% | 96% | 99% | 88.2% | 0% |
| B_MULTI | +0.25 วิ | 71% | 93% | 99% | 84.9% | 0% |
| **C_MULTI** ⭐ | **−0.16 วิ** | 74% | 95% | **100%** | 88.2% | 0% |
| C_MULTI_word | −0.23 วิ | 74% | 94% | 100% | **89.1%** | 0% |
| D_ASL | +0.38 วิ | 61% | 87% | 98% | 77.3% | 0% |
| D_ASL_gloss | −0.12 วิ | 71% | 91% | 99% | 81.5% | 0% |
| D_ASL_word | −0.13 วิ | 69% | 90% | 99% | 84.9% | 0% |

**ที่ดีที่สุดที่ยืนยันแล้ว (memory, 1 มิ.ย. 69):** C_MULTI — mean offset **−0.16 วิ**, 73.9% อยู่ใน
±1 วิ, **100%** อยู่ใน ±3 วิ, frame accuracy **82.6%**, **F1@0.5 = 88.2%**, end offset +0.23 วิ,
overlap 0% (เดิม 88.1%) ground truth `CC_Aligned` ตรวจอิสระแล้วว่าไม่มี overlap (0/118)

**สรุปเด่น:** C_MULTI แทบไม่มี bias (−0.16 วิ) และ **ทุก cue อยู่ภายใน 3 วินาที** จากความจริงที่จัด
ด้วยมือ — ผลที่ดีมากสำหรับ aligner ที่ไม่ต้อง train และข้ามภาษากับภาษาที่โมเดลไม่เคยเห็น

### 10.5 เรื่องของการแก้ overlap

เอาต์พุต DP ดิบ overlap cue ติดกัน **88.1%** การ clamp จุดจบรอบเดียว ([§8.2](#82-post-processing-แก้-overlap))
ลดเหลือ **0%** โดยไม่เปลี่ยน start metric — เป็น post-process ที่สะอาดและมีเหตุผล ไม่ใช่ hack

---

## 11. Task 2 — Gloss Labeling

### 11.1 นิยาม

ลงลึกกว่า Task 1: จัดเรียง **gloss token แต่ละตัว** (เช่น "ผายมือ", "เด็ก", "เรียน") กับ **ท่ามือ
เดี่ยว `SIGN`** ภายในประโยค gloss ของมัน ประเมินกับ GT `Gloss Labeling` ระดับท่าทาง (852 entry)
ด้วย **best-IoU pairing**

### 11.2 การทดลอง 1 — Prototype (tier `Gloss` ทั้งวิดีโอ)

รุ่นแรก ([align_gloss_labels.py](example_alignment/align_gloss_labels.py)) tokenize แต่ละ gloss
ประโยค, จำกัด candidate sign ภายใน window ของประโยค, embed token ด้วย multilingual SignCLIP แล้ว
รัน monotonic DP ต่อประโยค ([§8.3](#83-dp-ของ-task-2-ระดับ-token-align_gloss_labelspy))

### 11.3 การทดลอง 2 — Ablation: `Gloss` vs `Gloss_Input`

default ในโค้ดอ่าน `Gloss_Input`; ablation (Progress_09052026) เทียบกับ `Gloss` ตรงๆ **`Gloss` ชนะ
ทุก metric:**

| Metric | `Gloss` | `Gloss_Input` | Δ |
| --- | ---: | ---: | ---: |
| Mean IoU | **0.4901** | 0.4199 | +7.0 pp |
| % IoU ≥ 0.5 | **48.4%** | 38.9% | +9.4 pp |
| % IoU ≥ 0.3 | **77.0%** | 66.0% | +11.0 pp |
| % zero overlap | **2.5%** | 6.6% | −4.1 pp |
| Mean abs start offset | **0.188 วิ** | 0.212 วิ | −24 ms |

**ทำไม `Gloss` ชนะ:**
1. **จำนวน token ตรงกัน:** `Gloss` มี **852 token = 852 GT entry** เป๊ะ; match ตามตำแหน่ง 71.2%
   (เทียบ 4.9% ของ 889 token ใน `Gloss_Input`) — เพราะ annotator สร้าง GT *จาก* รายการ token `Gloss`
2. **องศาอิสระ:** token เกิน 37 ตัวของ `Gloss_Input` ถูกบีบเข้า 852 slot ของ GT ทำให้ overlap ลดลง
3. **การครอบ window:** prediction ของ `Gloss` overlap 97.77% ของ GT เทียบ 88.97% ของ `Gloss_Input`

> **⚠️ ข้อควรระวังที่บันทึกไว้ (ความซื่อตรงเชิงวิชาการ):** metric *exact-text-match* เป็น 65.1%
> สำหรับ `Gloss` เทียบ 10.6% สำหรับ `Gloss_Input` — แต่นี่คือ **leakage เชิงโครงสร้าง** เพราะ GT ถูก
> สร้างจากรายการ token `Gloss` ดังนั้นโปรเจกต์ **รายงาน IoU เป็นตัวหลัก** ไม่ใช่ text match ข้อระวังนี้
> ถูกย้ำทุกที่ที่ตัวเลขนี้ปรากฏ

### 11.4 การทดลอง 3 — Pipeline crop วิดีโอต่อประโยค (Progress_16052026)

pipeline ที่ซื่อตรงกว่า ([run_task2_per_sentence.py](example_alignment/run_task2_per_sentence.py)):
crop วิดีโอเป็น **119 คลิป** ที่ขอบประโยค gloss แล้วรัน pipeline เต็ม pose→segmentation→embedding→DP
*ต่อคลิป* แล้ว aggregate กลับเป็น CSV/VTT/EAF

| Metric | Per-sentence | `Gloss` ทั้งวิดีโอ |
| --- | ---: | ---: |
| Mean IoU | 0.4763 | 0.4901 |
| % IoU ≥ 0.5 | 46.0% | 48.4% |
| % any overlap | 96.1% | 97.5% |

**ข้อค้นพบ:** per-sentence **ดีพอๆ กัน** กับ `Gloss` ทั้งวิดีโอ (−1.4 pp Mean IoU) แต่ **ช้ากว่า ~7×**
สรุป: ใช้แบบทั้งวิดีโอดีกว่า การ crop ไม่ได้ประโยชน์เพิ่ม

### 11.5 หมายเหตุการกระทบยอด

memory บันทึกตัวเลข canonical ของคลิปเดียวเป็น **Mean IoU 0.4199 / 38.9%** จาก prediction run
*ภายหลัง* (ตัวเลข 0.4901/48.4% มาจาก prediction run ก่อนหน้า ปัจจุบันถูกแทนสำหรับ path *default* แต่
ยังเป็นตัวหลักของ ablation tier `Gloss`) ทั้งสองถูกบันทึก; README นำเสนอ `Gloss` 0.4901 เป็นที่แนะนำ
และ `Gloss_Input` 0.4199 เป็น default ความต่างคือกำลังประเมิน prediction CSV ไหน

---

## 12. ForcedAlignment — ขยายสเกล

> ทุกอย่างข้างบนรันบนวิดีโอ **เดียว** sub-project [ForcedAlignment/](ForcedAlignment/) ขยาย Task 2
> เป็นคลัง TSL **1,132 คลิป** (เฉลี่ย 5.8 วิ/คลิป, ~110 นาที, ~8.5 GB) ด้วย **5 configuration**

### 12.1 ทำไม

เพื่อทดสอบว่าวิธี Task 2 generalize ได้เกินคลิปเดียวที่คัดมาหรือไม่ และเพื่อสร้างผลประเมินระดับ dataset
จริง (ไม่ใช่กรณี n=1)

### 12.2 ข้อมูล

- **ไฟล์ ground-truth EAF 1,140 ไฟล์** ใน `elan_forced_alignment/` (~9 MB, **อยู่ใน git**)
- **MP4 ต้นทาง 1,132 ไฟล์** (ข้อมูลวิจัย NECTEC, **ไม่อยู่ใน git**) จัดในหกโฟลเดอร์ `<N> MP/` ใต้
  โฟลเดอร์บนสุด (convention พัฒนา: `หนังสือภาษามือไทย/`)
- orchestrator ค้นหา video root อัตโนมัติโดย scan subdir ที่มี MP4; จับคู่ด้วย **stem** (`42.mp4 ↔
  42.eaf`) และ stem ซ้ำข้ามโฟลเดอร์จะ error

EAF ต้อง **ซ่อม** ก่อน — `check_eaf_video_match.py` ตรวจความตรง EAF↔video และ
`fix_eaf_media_paths.py` ซ่อม path `MEDIA_DESCRIPTOR` ที่พังให้เปิดใน ELAN ได้สะอาด (commit 18 พ.ค.
69 ในชื่อ "repaired ForcedAlignment EAF dataset")

### 12.3 Orchestrator — `run_forced_alignment.py` (807 บรรทัด, 7 phase)

| Phase | ขั้น | เอาต์พุต |
| --- | --- | --- |
| 1 | สร้าง manifest + `video_ids.txt` | `output/manifest.csv` |
| 2 | `videos_to_poses` (MediaPipe) | `output/poses/<id>.pose` (~3.8 GB) |
| 3 | SEA segmentation | `output/seg/E4s-1_30_50/<id>.eaf` |
| 4 | SignCLIP segment embedding | `output/emb/<id>.npy` |
| 5 | DP align ในกระบวนการ (5 config) | `output/predictions/config<N>_*.csv` |
| 6 | ฉีด tier prediction เข้า EAF | `output/predicted_eafs/<id>.eaf` |
| 7 | ประเมิน (P/R/F1/IoU/FrameAcc) | `output/evaluation/*.csv` |

ฟีเจอร์เชิงปฏิบัติที่ใส่ไว้: `--preflight-only` (ตรวจ manifest + tier ที่ต้องมี โดยไม่ compute),
`--only-ids` (smoke test 3 คลิป), `--skip-pose/-seg/-emb/-align/-eval` (วน iterate phase หลังโดยไม่
รัน pose ใหม่) และ hardlink ใน `video_work/` เพื่อเลี่ยงคัดลอก 5 GB

### 12.4 5 configuration

| # | Input tier | GT tier | Tokenization | จุดประสงค์ |
| ---: | --- | --- | --- | --- |
| 1 ⭐ | `CC` | `CC_Aligned` | ช่องว่าง, **ตัด `sil`** | baseline ดีที่สุด |
| 2 | `CC` | `CC_Aligned` | ช่องว่าง, **เก็บ `sil`** | ทดลองจัดการ sil |
| 3 | `Gloss` | `Gloss_Labeling` | คั่น pipe, ตัดว่าง | analogue ตรงของ Task 2 บน `04.mp4` |
| 4 | `Gloss1` | `Gloss_Labeling1` | คั่น pipe | annotation gloss ทางเลือก |
| 5 | `Gloss2` | `Gloss_Labeling2` | คั่น pipe | annotation gloss ทางเลือก |

### 12.5 ผลลัพธ์ (full run 1,132 คลิป, จาก `evaluation_summary.csv`)

| Config | Pred | GT | Precision@0.5 | Recall@0.5 | **F1@0.5** | **Mean IoU** | FrameAcc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** ⭐ CC→CC_Aligned | 1,171 | 1,132 | 67.5% | 69.8% | **68.6%** | **0.5928** | 76.8% |
| 2 CC+sil→CC_Aligned | 3,435 | 3,396 | 17.5% | 17.7% | 17.6% | 0.2039 | 21.3% |
| 3 Gloss→Gloss_Labeling | 1,713 | 1,713 | 7.8% | 7.8% | 7.8% | 0.2484 | 26.2% |
| 4 Gloss1→Gloss_Labeling1 | 3,977 | 3,977 | 20.7% | 20.7% | 20.7% | 0.2229 | 22.3% |
| 5 Gloss2→Gloss_Labeling2 | 3,977 | 3,977 | 20.9% | 20.9% | 20.9% | 0.2238 | 22.4% |

**สรุปเด่น:** **Config #1 (F1 68.6%, mIoU 0.5928) ชนะ baseline คลิปเดียว `04.mp4` (0.4901)** — วิธีนี้
*ดีขึ้น* เมื่อขยายสเกล การจับคู่ใช้ **ตำแหน่ง + IoU เท่านั้น** (ไม่มี text leakage); text-match รายงาน
แยกเป็น reference-only (~97%)

### 12.6 ทำไม config 2–5 ผลตก (error analysis)

วิเคราะห์และบันทึกสองสาเหตุหลัก (PLAN §10.11/§10.11b):

1. **token `sil` (Config #2):** การเก็บ token เงียบทำให้จำนวน prediction ระเบิด (3,435 เทียบ 1,171)
   และ precision ตก — segmenter ไม่มี "ท่ามือเงียบ" ให้ match token `sil` จึงได้ alignment ขยะ การตัด
   `sil` (Config #1) คือวิธีแก้
2. **ความไม่ตรงเชิงสถาปัตยกรรม/convention การ annotate (Config 3–5):** GT `Gloss_Labeling` ถูก
   annotate ด้วย convention *ต่างกัน* (ระดับท่าทาง, คั่น pipe, มักละเอียดกว่าหน่วย `SIGN` ของ segmenter
   SEA) segmenter กับ GT จึงไม่ตรงกันว่าอะไรคือ "ท่ามือหนึ่งท่า" ทำให้การจับคู่ IoU ถูกลงโทษเชิง
   โครงสร้างไม่ว่า embedding จะดีแค่ไหน นี่คือปัญหา **convention ของข้อมูล** ไม่ใช่โมเดลพัง — และเป็น
   ข้อค้นพบที่จูงใจให้ fine-tune segmenter บน TSL พอดี

### 12.7 สิ่งส่งมอบ (24 พ.ค. 69)

- prediction CSV 5 ไฟล์ + VTT คู่กัน 5 ไฟล์ (`prediction_vtt/`)
- CSV ประเมินต่อคลิป (`eval_config{1..5}.csv`) + `evaluation_summary.csv`
- `evaluation_summary_template_format.csv`
- Comparison EAF (tier GT + prediction ต่อคลิป) ผ่าน `create_comparison_eafs.py`
- รายงานเติมอัตโนมัติ `Gloss_Labeling_Report_Filled.docx` ผ่าน `fill_gloss_labeling_template.py`

---

## 13. ระเบียบวิธีการประเมินผล

### 13.1 Task 1 — index-based matching (และเหตุผล)

| | eval เดิมของ SEA | โปรเจกต์นี้ |
| --- | --- | --- |
| สเกล | BOBSL, 20,000+ ประโยค | วิดีโอเดียว, 119 cue |
| Matching | text lookup / ระดับเฟรม | **index-based** `pred[i] ↔ gt[i]` |
| Coverage | 69/172 (text lookup fail เมื่อ annotator แก้คำ) | **119/119** |
| Dependency | pysrt, webvtt, beartype, path BOBSL | **stdlib เท่านั้น** |

เหตุผลที่ใช้ index-based: `CC_Input` กับ `CC_Aligned` มี 119 entry เท่ากันตามลำดับ; annotator มักแก้
**ถ้อยคำ** ระหว่าง align ทำให้ text lookup match แค่ ~50/119 การจับคู่ตาม index ให้ coverage เต็ม
119/119 [evaluate_all.py](example_alignment/evaluate_all.py) ยังคำนวณ metric ของ *SEA* (FrameAcc,
F1@IoU) ที่ FPS=25 เพื่อเทียบกันได้

**Metric ที่ผลิต:** signed/abs mean+median start offset, end offset, %ภายใน ±1/±2/±3 วิ, overlap %,
frame accuracy, F1@0.10/0.25/0.50

### 13.2 Task 2 — best-IoU pairing

[evaluate_gloss_labeling.py](example_alignment/evaluate_gloss_labeling.py): สำหรับแต่ละ prediction
หา GT entry ที่ IoU สูงสุด; บันทึก IoU, signed offset, และ flag text-match aggregate: mean/median
IoU, %IoU≥0.5/0.3, %any overlap, mean signed offset, exact text match (flag ว่ามีโอกาส leak)

### 13.3 ForcedAlignment — positional IoU, ไม่มี text leakage

[evaluate_fa_dataset.py](ForcedAlignment/evaluate_fa_dataset.py) จับคู่ prediction กับ GT ด้วย
**ตำแหน่ง + IoU เท่านั้น** ไม่เคยใช้ข้อความ — ตัวเลขหลักจึงไม่ถูกปั่นด้วย gloss-leakage text_match
คำนวณแต่รายงาน **reference-only**

### 13.4 การรับประกัน reproducibility

cached output (aligned VTT, prediction CSV, eval CSV, GT EAF) ถูก **commit** ไว้ ดังนั้น evaluator
ทั้งสามรันใน ~30 วิ โดย **ไม่ต้องมีวิดีโอต้นทาง ไม่ต้องมี PyTorch ไม่ต้องมี fairseq** — ดู
[Evaluation-Only Quick Start](README.md#evaluation-only-quick-start) และ
[requirements-eval.txt](requirements-eval.txt) ตรวจซ้ำ bit-for-bit แล้ว 1 มิ.ย. 69

---

## 14. เครื่องมือและ Script ที่สร้างขึ้น

script ที่เขียนเองอยู่ใน [example_alignment/](example_alignment/) และ
[ForcedAlignment/](ForcedAlignment/) ตัวเด่น:

**เตรียมข้อมูล / IO:**
- `extract_cc_from_eaf.py` — EAF tier → VTT (flag `--tier`)
- `make_gloss_cc_vtt.py` — สร้าง Gloss_Input VTT (ข้อความ gloss + timestamp CC)
- `make_gloss_input_tier.py` — สร้าง tier Gloss_Input ครั้งเดียว
- `merge_cc_to_updated_eaf.py` — คัดลอก tier CC ระหว่าง EAF
- `fix_overlap_vtt.py` — แก้ overlap แบบ clamp จุดจบ → overlap 0%

**aligner ของงาน:**
- `align_gloss_labels.py` (555 บรรทัด) — token aligner ของ Task 2 พร้อม ablation `--tier`
- `run_task2_per_sentence.py` — pipeline crop-and-align ต่อประโยค

**ประเมินผล:**
- `evaluate_all.py` / `evaluate_all_to_csv.py` — Task 1 index-based eval (เดี่ยว / batch ทั้ง 7)
- `evaluate_gloss_labeling.py` — Task 2 IoU eval
- `ForcedAlignment/evaluate_fa_dataset.py` — positional IoU eval ระดับคลัง

**Visualization / comparison:**
- `add_vtt_tiers_to_eaf.py` — `Test_comparison.eaf` (17 tier: 7 pre + 7 post + default + 2 ablation)
- `add_best_to_eaf.py` — `Test_best.eaf` (เฉพาะที่ดีที่สุด, มี auto-fallback)
- `make_task2_comparison_eaf.py`, `plot_alignment.py` — Task 2 comparison EAF + timeline PNG 4 เลน

**Ops ของ ForcedAlignment:**
- `run_forced_alignment.py` (807 บรรทัด) — orchestrator 7 phase
- `check_eaf_video_match.py`, `fix_eaf_media_paths.py` — ตรวจ/ซ่อม EAF↔video
- `create_comparison_eafs.py`, `fill_gloss_labeling_template.py` — สร้างสิ่งส่งมอบ

**จุดชนะด้าน portability:** script ใน `example_alignment/` ทั้งหมดถูก refactor เป็น
`HERE = Path(__file__).parent` (ไม่มี absolute path hardcode) — ตรวจ portable แล้ว 26 พ.ค. 69

---

## 15. ปัญหาและวิธีแก้

| ปัญหา | ต้นเหตุ | วิธีแก้ |
| --- | --- | --- |
| GPU ไม่ถูกใช้เงียบๆ | wheel PyTorch CUDA ผิดบน Blackwell | ลง wheel **cu128** |
| Pose extraction รันบน CPU | MediaPipe บน Windows ไม่มี GPU delegate | ยอมรับ (บันทึกไว้); CPU เป็นคอขวด |
| Segmentation ใช้ GPU ไม่ได้ | upstream บังคับ `CUDA_VISIBLE_DEVICES=""`; JIT LSTM freeze CPU device | แก้ไม่ได้ถ้าไม่ re-export โมเดล — บันทึกไว้ |
| `subprocess shell=True` พังบน Windows | shell quoting ต่างกัน | `shlex.split` + `shell=False` ใน `segmentation.py` |
| SignCLIP path error บน Windows | สมมติฐาน path แบบ Linux `/` ใน 8 ไฟล์ | `patches/fairseq_signclip_windows.patch` (pin ที่ commit `a8199440`) |
| `numba` JIT ใช้ไม่ได้ | ไม่มี LLVM | fallback import เป็น Python ธรรมดาใน `align_dp.py` |
| ทุก cue ยุบไปเวลาเดียวกัน | `--sign-b/o-threshold` ไม่ตรงระหว่าง align กับ seg | ใช้ `30 50` เหมือนกัน |
| `--segmentation_dir` ผิดระดับ | align.py ต่อ subdir เอง | ชี้ที่ parent `segmentation_output` |
| Multiprocessing path error | Windows + `num_workers>1` | `--num_workers 1` |
| `video_ids.txt` UnicodeDecodeError | PowerShell `echo` เขียน UTF-16 BOM | เขียนผ่าน Python ด้วย `encoding='utf-8'` |
| text-lookup eval match แค่ ~50/119 | annotator แก้คำ cue | ใช้ index-based matching → 119/119 |
| เอาต์พุต DP ดิบ overlap 88% | DP ไม่มี constraint ห้าม overlap | clamp จุดจบรอบเดียว → 0% |
| Config #2 precision ตก | token `sil` ไม่มีท่ามือ match | ตัด `sil` (Config #1) |
| Config 3–5 IoU ต่ำ | convention annotate ของ GT ≠ หน่วยของ segmenter | วินิจฉัยว่าเป็น data-convention mismatch |
| manifest.csv ถูกตัดเหลือ 1 คลิป (1 มิ.ย. 69) | working-tree เสียที่ยังไม่ commit | กู้ด้วย `git checkout`; GT/pred ยังครบ |

---

## 16. ข้อค้นพบและคำแนะนำ

### 16.1 ข้อค้นพบสำคัญ

1. **Multilingual SignCLIP transfer ไป TSL ได้** — ชนะทั้ง BSL และ ASL ยืนยันว่า cross-lingual
   transfer ใช้ได้กับภาษาที่โมเดลไม่เคย train
2. **ข้อความ Gloss > ข้อความคำบรรยายภาษาพูด** สำหรับการ align ด้วย embedding (gloss อยู่ใกล้ท่ามือ
   ใน space ร่วม)
3. **C_MULTI คุณภาพระดับ production สำหรับ Task 1:** mean offset −0.16 วิ, 100% ภายใน ±3 วิ,
   F1@0.5 = 88.2%
4. **วิธีนี้ขยายสเกลได้:** Config #1 ที่ 1,132 คลิป (mIoU 0.59) *ชนะ* baseline คลิปเดียว (0.49)
5. **segmenter เป็นจุดอ่อน** ของ Task 2 ละเอียด — convention annotate ของ GT ไม่ตรงกับหน่วยท่ามือของ
   segmenter ที่ train บน BSL (Config 3–5)
6. **การ crop ต่อประโยคไม่ช่วย** — แม่นเท่ากันแต่ช้ากว่า 7×

### 16.2 คำแนะนำสำหรับ production

- **Task 1:** ใช้ **C_MULTI** (multilingual + ข้อความ gloss), น้ำหนัก tuned `w_D=2 w_G=8 max_gap=6
  window=40 w_S=6`, bias 1.3/1.0 แล้วตามด้วย `fix_overlap_vtt.py`
- **Task 2:** ใช้ aligner ทั้งวิดีโอบน **tier `Gloss`** รายงาน **IoU** ไม่ใช่ text match
- **คลัง:** ใช้ **Config #1 (CC→CC_Aligned, ตัด sil)**

### 16.3 ข้อควรระวังเชิงซื่อตรง

- ตัวเลข text-match ของ `Gloss` ใน Task 2 (65%) เป็น **leakage** — อย่าอ้างเป็น accuracy
- ผลคลิปเดียวคือ n=1; เชื่อตัวเลขคลัง 1,132 คลิปสำหรับการ generalize

### 16.4 แนวทางพัฒนาต่อ (เรียงตามลำดับความสำคัญ, Big_Progress §11)

1. **กวาด DP hyperparameter** (ถูกสุด, ไม่ train) — grid อย่างเป็นระบบบน `w_D/w_G/w_S/window/bias`
2. **Fine-tune SignCLIP บนคลังคำ gloss ไทย** — ปรับ encoder ข้อความ/ท่ามือให้เข้ากับ TSL ให้ cosine
   similarity คมขึ้น
3. **Fine-tune segmenter E4s-1 บน TSL** (ผลตอบแทนสูงสุดสำหรับ Task 2) — full fine-tune ถ้ามี
   ≥1,000 ตัวอย่าง ไม่งั้นใช้ adapter/last-layer; โจมตี mismatch convention ของ Config 3–5 โดยตรง
4. **Fine-tune end-to-end** (ขั้นสูง, ทางเลือกสุดท้าย)

---

## 17. สิ่งส่งมอบและรายการ

### 17.1 โค้ด (ไฟล์ที่ track 1,342 ไฟล์)

- `SEA/` ที่แก้แล้ว (align หลายโมเดล, segmentation ปลอดภัยบน Windows)
- script เองใน `example_alignment/` (เตรียมข้อมูล, aligner 2 ตัว, evaluator 3 ตัว, visualizer)
- orchestrator + evaluator + เครื่องมือ EAF + GT EAF 1,140 ไฟล์ ใน `ForcedAlignment/`
- `patches/fairseq_signclip_windows.patch` + `patches/README.md`

### 17.2 ผลลัพธ์ที่ cache (commit, reproduce ได้)

- 7× `aligned_output_*/04{,_no_overlap}.vtt` (Task 1) + `evaluation_task1_results.csv`
- prediction CSV/VTT ของ Task 2 (default + ablation `Gloss`/`Gloss_Input`) + eval CSV
- ForcedAlignment `output/predictions/config{1..5}_*.csv`, `prediction_vtt/`,
  `evaluation/eval_config{1..5}.csv`, `evaluation_summary.csv`
- Comparison/best EAF (`Test_comparison.eaf` 17 tier, `Test_best.eaf`)

### 17.3 เอกสาร

- [README.md](README.md) (78 KB, สองภาษา, คู่มือ setup หลัก)
- [Big_Progress.md](Big_Progress.md) (126 KB, เอกสารหลักภาษาไทย, 12 ส่วน)
- ไฟล์นี้ — `PROJECT_PROGRESS_DEEP_TH.md` (เรื่องเล่าเชิงลึกภาษาไทย) + ฉบับอังกฤษ
  [PROJECT_PROGRESS_DEEP.md](PROJECT_PROGRESS_DEEP.md)
- progress report ตามวันที่ 5 ฉบับ: [Progress_20042026.md](Progress_20042026.md),
  [Progress_26042026.md](Progress_26042026.md), [Progress_04052026.md](Progress_04052026.md),
  [Progress_09052026.md](Progress_09052026.md), [Progress_16052026.md](Progress_16052026.md)
- [presentation_12052026.md](presentation_12052026.md), [Presentation_Task1_Update.md](Presentation_Task1_Update.md)
- [SEA_Pipeline_Guide_TH.md](SEA_Pipeline_Guide_TH.md), `script.md`, `SEA_Pipeline_Deep_Dive.html`
- [report/sea_report.pdf](report/sea_report.pdf) + `sea_report_th.pdf` ภาษาไทย (มี source LaTeX)
- `DeepDive.pdf` (7.3 MB)
- [ForcedAlignment/README.md](ForcedAlignment/README.md) + [PLAN_ForcedAlignment_Task2.md](ForcedAlignment/PLAN_ForcedAlignment_Task2.md)
- `arXiv-2512.08094v1/` — source เต็มของ paper SEA (LaTeX + figure)

### 17.4 Asset ที่ไม่อยู่ใน git (ต้องขอ/สร้างใหม่)

`04.mp4`, `04.pose`, MP4 ของ ForcedAlignment (ข้อมูล NECTEC), embedding, segmentation EAF,
`fairseq_signclip/`, SignCLIP checkpoint 3 ไฟล์ (~600 MB)

---

## 18. อ้างอิง

- **SEA** — Jiang, Jang, Momeni, Varol, Ebling, Zisserman (2025), *Segment, Embed, and Align: A
  Universal Recipe for Aligning Subtitles to Signing*, [arXiv:2512.08094](https://arxiv.org/abs/2512.08094)
  · โค้ด: [J22Melody/SEA](https://github.com/J22Melody/SEA)
- **SignCLIP** — โมเดล embedding ภาษามือแบบ multilingual · [J22Melody/fairseq](https://github.com/J22Melody/fairseq)
- **Linguistic segmenter** — [J22Melody/segmentation@bsl](https://github.com/J22Melody/segmentation/tree/bsl)
- **pose-format / MediaPipe Holistic** — [sign-language-processing/pose](https://github.com/sign-language-processing/pose)
- **NECTEC** — ผู้ให้วิดีโอ annotation TSL และ ELAN annotation
- **ELAN** — [archive.mpi.nl/tla/elan](https://archive.mpi.nl/tla/elan)

---

*เรียบเรียง 7 มิ.ย. 2569 จาก commit history, source code, ผลลัพธ์ที่ commit ไว้ และเอกสารความคืบหน้า
เดิมของ repo ทุก metric ที่อ้างอิงโยงถึง CSV ที่ commit ไว้ และตรวจสอบว่า reproduce ได้แล้วเมื่อ
1 มิ.ย. 2569*
