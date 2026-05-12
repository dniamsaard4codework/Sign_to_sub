# คู่มือเข้าใจ SEA Pipeline ฉบับสมบูรณ์

> **เป้าหมายของเอกสารนี้:** ทำให้ผู้อ่านที่ไม่เคยรู้จัก SEA สามารถเข้าใจ
> โครงการทั้งหมดได้อย่างละเอียดถึงรากของ design decisions
>
> **ครอบคลุม:**
>
> - **Task 1** — Subtitle Alignment (จัดเวลาคำบรรยายให้ตรงกับท่ามือ)
> - **Task 2** — Gloss Labeling (จัด token-level annotation รายท่ามือ)
> - **Section พิเศษ:** ทำไม Overlap ถึงเกิดและแก้อย่างไร

---

## สารบัญ

1. [ปัญหาที่เราพยายามแก้](#1-ปัญหาที่เราพยายามแก้)
2. [แนวคิดหลักของ SEA — Segment, Embed, Align](#2-แนวคิดหลักของ-sea--segment-embed-align)
3. [ข้อมูลนำเข้า — รู้จัก EAF tiers](#3-ข้อมูลนำเข้า--รู้จัก-eaf-tiers)
4. [Task 1 — Subtitle Alignment ระดับประโยค](#4-task-1--subtitle-alignment-ระดับประโยค)
   - 4.1 Pose Estimation
   - 4.2 Sign Segmentation
   - 4.3 SignCLIP Embeddings
   - 4.4 **DP Alignment** (หัวใจของระบบ)
   - 4.5 **ทำไม Overlap ถึงเกิด** (พิเศษ)
   - 4.6 Overlap Fix
   - 4.7 Evaluation
5. [Task 2 — Gloss Labeling ระดับท่ามือ](#5-task-2--gloss-labeling-ระดับท่ามือ)
   - 5.1 Task 2 เพิ่มอะไรจาก Task 1?
   - 5.2 Per-sentence Monotonic DP
   - 5.3 Token Embedding Cache
   - 5.4 Evaluation ด้วย IoU
   - 5.5 Ablation: Gloss vs Gloss_Input
6. [Design Decisions รวบยอด](#6-design-decisions-รวบยอด)
7. [สรุปภาพรวม — Mental Model](#7-สรุปภาพรวม--mental-model)

---

## 1. ปัญหาที่เราพยายามแก้

### 1.1 คำบรรยายของวิดีโอภาษามือมาช้ากว่าท่ามือเสมอ

ในวิดีโอที่มีล่ามภาษามือ (TSL = Thai Sign Language) **คำบรรยาย** (subtitle, CC)
ที่แสดงด้านล่างของจอ **มาจากเสียงพูด** ของผู้บรรยายต้นฉบับ ขณะที่ผู้แปลภาษามือ
**จะแสดงท่ามือช้ากว่าเสียง ~1–2 วินาที** (เพราะต้องฟังเสียง → ตีความ → แสดงท่า)

ผลลัพธ์: **คำบรรยายและท่ามือไม่ตรงเวลากัน** — ผู้พิการทางการได้ยินที่อ่าน
ทั้งคำบรรยายและดูท่ามือพร้อมกันจะสับสนเพราะข้อมูลทั้งสองช่องไม่ sync

```text
เวลา (sec):    0    1    2    3    4    5    6    7    8    9   10
เสียง:        [─── "การเปรียบเทียบและเรียงลำดับ" ───]
CC text:      [──── timestamp จากเสียง ─────]                     ← มาเร็ว
ท่ามือจริง:                  [─── ผายมือ เปรียบเทียบ ───]       ← มาช้า ~2s
```

### 1.2 เป้าหมายของโครงการ

**Task 1 — Subtitle Alignment:** ปรับเวลาของ CC text ให้ไปตรงกับท่ามือจริง

```text
ก่อน align:    [── CC (start=1s) ──]                              ← ตามเสียง
หลัง align:                          [── CC (start=3s) ──]         ← ตามท่ามือ
ท่ามือจริง:                            [── (start=3s) ──]
```

**Task 2 — Gloss Labeling:** ลงลึกถึงระดับ "token เดียวต่อท่ามือเดี่ยว"
(เพราะหนึ่งประโยคมีหลายท่ามือ แต่ละท่าใช้เวลาต่างกัน) — เพื่อสร้าง
annotation ละเอียดสำหรับการ research ภาษามือต่อ

### 1.3 ทำไม Cross-lingual จาก BSL → TSL?

โมเดล SEA และ SignCLIP เดิม train บน **BSL (British Sign Language)** —
มี dataset ใหญ่ (BOBSL, 20,000+ clips) ส่วน TSL **มีข้อมูล annotated น้อย**
จึงต้องลอง "ยืม" โมเดลจากภาษาอื่นมาใช้กับ TSL โดยไม่ retrain

> **สมมติฐานหลัก:** ท่ามือของภาษามือต่างๆ แม้จะไม่เหมือนกัน แต่ **โครงสร้าง
> เวลาของการเคลื่อนไหวมือ** (เริ่ม → เคลื่อนไหว → จบ) ใกล้เคียงกัน — โมเดล
> ที่เรียนรู้ pattern เวลาจาก BSL จึงน่าจะ generalize มา segment ท่ามือ TSL
> ได้

---

## 2. แนวคิดหลักของ SEA — Segment, Embed, Align

SEA = **S**egment + **E**mbed + **A**lign — สามขั้นตอนหลักของ pipeline

### 2.1 ภาพรวม

```text
                    ┌──────────────┐
   วิดีโอ ─────────►│  S — Segment │──► sign_segments[]
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
   sign + text ────►│  E — Embed   │──► (sign_vec, text_vec)
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
   M cues × N segs ►│  A — Align   │──► aligned_cues[]
                    └──────────────┘
```

### 2.2 S — Segment: หาขอบเขตของท่ามือเดี่ยว

**ปัญหา:** วิดีโอ 11 นาที = 39,600 เฟรม — ต้องตัดเป็นช่วงๆ ก่อน align
เพราะการ align ทุก cue กับทุกเฟรมต้นทุนสูงเกินไป

**วิธี:** ใช้ pose information (skeleton landmarks) มาทำนายว่าเฟรมไหนเป็น
"ขอบ" ของท่ามือ (boundary) — แล้วตัดเป็น **SIGN segments** ~2,780 อัน
สำหรับวิดีโอ 11 นาที (เฉลี่ย ~250 ms ต่อท่ามือ)

### 2.3 E — Embed: ทำให้ sign กับ text อยู่ใน semantic space เดียวกัน

**ปัญหา:** sign segment คือ "sequence ของ pose landmarks" — แต่ cue คือ
"ข้อความ" ภาษาไทย → ทั้งคู่อยู่คนละ domain เลย → เทียบกันตรงๆ ไม่ได้

**วิธี:** ใช้ **SignCLIP** (เลียนแบบ CLIP จาก image+text) ที่ฝึก contrastive
learning ให้ **sign vector ของท่ามือ "เรียน" ใกล้กับ text vector ของคำว่า
"เรียน"** ในพื้นที่ vector 768 มิติเดียวกัน

ผลลัพธ์: คำนวณ **cosine similarity** ระหว่าง sign_vec กับ text_vec ได้
ทันที — ค่าสูง = ใกล้ความหมาย, ค่าต่ำ = ไม่เกี่ยว

### 2.4 A — Align: จับคู่ cue กับ sign segments ด้วย DP

**ปัญหา:** มี 119 cues และ 2,780 sign segments → ต้องเลือกว่า cue ไหน
ผูกกับ segments ไหนบ้าง (และไม่ทับซ้อนกัน)

**วิธี:** Dynamic Programming หา **partition** ที่ดีที่สุด — minimize
"ต้นทุนรวม" ของการจับคู่ (รายละเอียดใน §4.4)

---

## 3. ข้อมูลนำเข้า — รู้จัก EAF tiers

### 3.1 EAF คืออะไร

EAF = **E**LAN **A**nnotation **F**ormat — ไฟล์ XML ที่ใช้กับโปรแกรม ELAN
(เครื่องมือ annotate ภาษามือ/ภาษาธรรมชาติ) แต่ละ EAF มี **tier** หลายชั้น
ที่บอกข้อมูลคนละแบบในเวลาเดียวกัน

### 3.2 Tiers ใน Test.eaf (input หลักของโครงการ)

| Tier | จำนวน entries | ความหมาย | บทบาท |
| --- | ---: | --- | --- |
| `CC` | 172 | คำบรรยายดิบจากเสียงพูด — ไม่ได้ใช้ | (deprecated) |
| `CC_Input` | 119 | คำบรรยายที่ curated แล้ว, timestamp ตามเสียง | **Input ของ Task 1** |
| `CC_Aligned` | 119 | คำบรรยายที่นักวิจัย align ด้วยมือ, timestamp ตามท่ามือ | **GT ของ Task 1** |
| `Gloss` | 119 (852 tokens) | gloss tier เดิม, 1 sentence = หลาย tokens | **Input ทางเลือกที่ดีที่สุดของ Task 2** |
| `Gloss_Input` | 119 (889 tokens) | gloss tier ที่ curated ใหม่ | **Input default ของ Task 2** |
| `Gloss Labeling` | 852 | annotation ละเอียด 1 entry = 1 ท่ามือ | **GT ของ Task 2** |

### 3.3 ความสัมพันธ์ระหว่าง tiers

```text
ระดับประโยค (sentence-level):
  CC_Input    [── "การเปรียบเทียบ" ──]   ← timestamp จากเสียง
              └──┐ pred[i] ↔ gt[i] (index 0..118)
                 ▼
  CC_Aligned                  [── "การเปรียบเทียบ" ──]   ← timestamp จากท่ามือ

ระดับ token (sub-sentence):
  Gloss       [── "ผายมือ การเปรียบเทียบ ตัวเลข" ──]   ← 3 tokens / 1 sentence
                │
                ├─► flatten + per-sentence DP
                ▼
  Gloss Labeling [ผายมือ][การเปรียบเทียบ][ตัวเลข]   ← 3 entries (sub-cues)
```

> 💡 **Key insight:** Gloss tier มี 119 sentences × ~7 tokens = 852 tokens
> ซึ่งตรงกับ 852 entries ใน Gloss Labeling GT เป๊ะ — เพราะ annotator
> สร้าง GT โดย **split sentences ของ Gloss เป็น tokens เดี่ยว** แล้ว
> กำหนดเวลาของแต่ละ token ด้วยมือ

---

## 4. Task 1 — Subtitle Alignment ระดับประโยค

### 4.1 Step 1: Pose Estimation

**Input:** `04.mp4` (1920×1080, 60 fps, 11 นาที = ~39,600 frames)
**Output:** `04.pose` (binary, 358 MB)

#### 4.1.1 ทำไมต้องใช้ pose ไม่ใช่ raw video?

**Raw video มีข้อมูลเยอะเกินไป** — สี, แสง, พื้นหลัง, เสื้อผ้าผู้แปล
สิ่งเหล่านี้ **ไม่เกี่ยวข้องกับการทำนายขอบเขตของท่ามือ** และจะทำให้
โมเดลเรียนรู้ลำบาก

**Pose สรุปเฉพาะข้อมูลสำคัญ:** ตำแหน่ง 543 landmarks ใน 3D space:

- **มือซ้าย 21 จุด** + **มือขวา 21 จุด** = 42 จุด (สำคัญที่สุด)
- **ใบหน้า 468 จุด** (สำหรับ expression, mouthing)
- **ลำตัว 33 จุด** (สำหรับ posture, body shifts)

→ จาก image 1920×1080×3 (~6.2 ล้าน values/frame) ลดเหลือ 543×3 = 1,629
values/frame → **เบาขึ้น 3,800 เท่า**

#### 4.1.2 MediaPipe Holistic — ตัวที่ใช้

ใช้ `videos_to_poses --format mediapipe` จาก library `pose-format`
ภายในเรียก **MediaPipe Holistic** ที่ทำงาน 3 รวมในตัวเดียว:

1. Hand detection + landmark estimation (มือซ้าย/ขวา)
2. Face mesh (ใบหน้า)
3. Pose detection (ลำตัว)

**Parameters สำคัญ:**

```text
--model_complexity 2          ← ความละเอียดสูงสุด (0=fast, 2=accurate)
--refine_face_landmarks       ← เพิ่ม face mesh detail
--no-smooth_landmarks         ← ไม่ทำ temporal smoothing (เก็บ raw)
```

**Runtime:** ~15 นาที CPU สำหรับวิดีโอ 11 นาที (60 fps) — ทำครั้งเดียว
แล้วใช้ผลซ้ำได้

### 4.2 Step 2: Sign Segmentation

**Input:** `04.pose`
**Output:** `segmentation_output/E4s-1_30_50/04.eaf` มี 2 tiers:

- `SIGN` — 2,780 segments (ท่ามือแต่ละท่า)
- `SENTENCE` — 418 segments (กลุ่มของท่ามือที่ต่อเนื่อง)

#### 4.2.1 ปัญหา: เฟรมไหนคือ "ขอบ" ของท่ามือ?

ในวิดีโอภาษามือ ผู้แปลทำท่ามือต่อเนื่องเป็น sequence — เราต้องรู้ว่า
**เฟรมไหนเป็นจุดเริ่ม/จบของท่ามือเดี่ยว** เพื่อแยกออกเป็น segments

**ปัจจัยที่บอก boundary:**

- ความเร็วของมือ (เร็ว = กลางท่า, ช้า = ขอบ)
- ระยะห่างของมือทั้งสอง
- การหยุดนิ่ง / เปลี่ยนทิศ

#### 4.2.2 Algorithm: E4s-1 (BiGRU)

ใช้โมเดล **Bidirectional GRU** ที่อ่าน pose sequence ทั้งสองทาง (forward
+ backward) → ทำนายความน่าจะเป็นว่าเฟรมนั้นเป็น boundary หรือไม่

```text
pose features → BiGRU → P(boundary | frame) ใน [0, 1]
```

#### 4.2.3 Thresholds สำคัญ

```text
--sign-b-threshold 30   ← boundary probability threshold (0–100)
                          ค่าต่ำ → segments เยอะ, ค่าสูง → segments น้อย
--sign-o-threshold 50   ← overlap merging threshold
                          ใช้ตอน merge adjacent boundaries
```

> ⚠️ **Critical:** ค่าทั้งสอง **ต้องคงที่ระหว่าง segmentation, embedding,
> alignment** — เพราะ pipeline ส่วนหลังคำนวณ path ของไฟล์ output จาก
> threshold (`E4s-1_30_50/` คือ `E4s-1` algorithm + b=30 + o=50)
> ถ้าเปลี่ยน → ทุกขั้นตอนต้องทำใหม่

**ผลของ video 04 ที่ b=30, o=50:**

- SIGN segments: 2,780 (เฉลี่ย ~239 ms/segment)
- SENTENCE segments: 418

#### 4.2.4 ทำไมต้องมี SENTENCE tier ด้วย?

SIGN = ท่ามือเดี่ยว (~250 ms) แต่บางครั้ง pipeline ต้องการ "กลุ่ม
ของท่ามือที่ต่อเนื่อง" (เช่น สำหรับการ visualize หรือสำหรับ refinement
ใน Task 2) — `SENTENCE` คือกลุ่ม SIGN ที่ติดกัน gap ไม่เกิน threshold

### 4.3 Step 3: SignCLIP Embeddings

**เป้าหมาย:** แปลง sign + text ให้เป็น vector 768 มิติในพื้นที่เดียวกัน

#### 4.3.1 SignCLIP คืออะไร

**CLIP** (Contrastive Language-Image Pre-training, OpenAI) คือโมเดลที่
ฝึกให้ image vector กับ text vector ที่ "ตรงกัน" อยู่ใกล้กันใน vector
space → ทำให้คำนวณ similarity ระหว่าง image กับ text ได้

**SignCLIP** ขยายแนวคิดเดียวกันมาสู่ภาษามือ:

```text
ภาพ "หมา"  ─► image_encoder ─┐
                              ├─► contrastive: ใกล้กัน
text "dog" ─► text_encoder ──┘

  เปลี่ยนเป็น

sign "หมา" ─► sign_encoder ──┐
                              ├─► contrastive: ใกล้กัน
text "dog" ─► text_encoder ──┘
```

#### 4.3.2 ทำไมต้องมี 3 โมเดล (BSL / Multilingual / ASL)?

| โมเดล | Training data | Language tag | ที่มา |
| --- | --- | --- | --- |
| `bsl` | BOBSL — BBC British Sign Language broadcasts | `<en> <bfi>` | UK Sign Language |
| `multilingual` | หลายภาษา รวม BSL + ASL + อื่นๆ | `<en>` | Generalist |
| `asl` | ASL Citizen + How2Sign | `<en> <ase>` | American Sign Language |

→ Task 1 ทดลองทั้ง 3 โมเดลเพื่อหาว่าตัวไหน **transfer มา TSL ได้ดีที่สุด**
ผลคือ `multilingual` ดีสุด (เพราะมี exposure หลายภาษา)

#### 4.3.3 ทำไมต้องมี 2 ประเภทของ embedding?

**(a) Sign embedding** — สำหรับแต่ละ SIGN segment (2,780 vectors, 768 มิติ)

```text
sign_encoder(pose_frames[start..end]) → (768,)
```

**(b) Subtitle embedding** — สำหรับแต่ละ CC cue (119 vectors, 768 มิติ)

```text
text_encoder(language_tag + " " + cue_text) → (768,)
```

ทั้งคู่อยู่ใน 768-dim space เดียวกัน → คำนวณ `sign_vec @ text_vec.T` ได้

#### 4.3.4 ทำไม Task 1 มี 7 experiments?

ลองทุก combination ของ:

- **Subtitle source:** CC_Input / Gloss_Input
- **Model:** BSL / Multilingual / ASL
- **Embedding mode:** precomputed (sentence-level) / live + tokenize (word-level)

```text
Experiment | Subtitle text | Model        | Mode
-----------|---------------|--------------|------------------
B2         | CC_Input      | BSL          | precomputed
B_MULTI    | CC_Input      | Multilingual | precomputed
C_MULTI ⭐ | Gloss_Input   | Multilingual | precomputed   ← winner
C_MULTI_w  | Gloss_Input   | Multilingual | live+tokenize
D_ASL      | CC_Input      | ASL          | precomputed
D_ASL_g    | Gloss_Input   | ASL          | precomputed
D_ASL_w    | Gloss_Input   | ASL          | live+tokenize
```

**ทำไม C_MULTI ดีที่สุด:**

1. `Gloss_Input` text **ตรงกับ content ของ sign มากกว่า** CC_Input
   (CC คือเสียงพูด ภาษาไทยธรรมชาติ; Gloss คือ notation ของผู้แปล
   ที่ใกล้กับท่ามือจริง)
2. `Multilingual` model **transfer cross-lingual ดีกว่า** model ที่ fine-tune
   เฉพาะภาษาเดียว

### 4.4 Step 4: DP Alignment — หัวใจของระบบ

> **หัวใจของทั้งโครงการ** — ส่วนนี้ยาวเพราะต้องอธิบายให้เข้าใจจริงๆ

#### 4.4.1 ปัญหาที่ DP ต้องแก้

```text
Input:  M = 119 cues (มี text, start, end จาก CC_Input)
        N = 2,780 sign segments (มี start, end จาก segmentation)
        similarity matrix S ขนาด M × N (จาก SignCLIP embeddings)

Goal:   จัดสรร 2,780 segments ให้กับ 119 cues
        - แต่ละ cue ผูกกับ "กลุ่มของ segments ที่ต่อเนื่อง"
        - กลุ่มต้องเรียงตามเวลา (cue 1 → cue 2 → ... → cue 119)
        - กลุ่มไม่ทับซ้อนกัน (segment แต่ละตัวถูก assign cue เดียว)

Output: 119 (start_time, end_time) ใหม่
```

#### 4.4.2 ตัวอย่างเชิงรูปธรรม

สมมติเราจัด 10 segments ให้ 3 cues:

```text
segments:      s1 s2 s3 s4 s5 s6 s7 s8 s9 s10
                ↓
partition:    [s1 s2] [s3 s4 s5 s6] [s7 s8 s9 s10]
                ↓        ↓              ↓
cue 1 ────────  ✓                                   start=s1.start, end=s2.end
cue 2 ──────────────────  ✓                         start=s3.start, end=s6.end
cue 3 ──────────────────────────────────  ✓         start=s7.start, end=s10.end
```

**กี่ partitions เป็นไปได้?** สำหรับ M = 119, N = 2,780 — เป็น **combinatorial
explosion** → ลองทุก partition ไม่ได้ → ต้องใช้ DP

#### 4.4.3 DP State

**Definition:**

$$
\text{dp}[i][j] = \text{ต้นทุนต่ำสุดในการจัด cues } 1{..}i \text{ โดย cue } i \text{ ใช้ segments ตั้งแต่ index บางตัวจนถึง } j
$$

อ่านอีกแบบ: "ต้นทุนต่ำสุดของการ align cues 1 ถึง *i* โดย cue *i* **ปิดท้ายที่ segment *j***"

**ทำไม "ปิดท้ายที่ j" สำคัญ?** เพราะ cue *i+1* ต้องเริ่มที่ segment *j+1*
ขึ้นไป → state ต้องรู้ว่า cue ก่อนหน้าจบที่ไหน เพื่อ transition ต่อได้

#### 4.4.4 DP Recurrence (สูตรเปลี่ยน state)

สำหรับ cue *i* ที่เริ่มที่ segment *k* และจบที่ segment *j* (โดย $k \leq j$):

$$
\text{dp}[i][j] = \min_{k \in [i-1,\, j]} \Bigl( \text{dp}[i-1][k-1] + C(i, k, j) \Bigr)
$$

**ตีความ:**

- `dp[i-1][k-1]` = ต้นทุนของการจัด cues `1..i-1` โดย cue `i-1` จบที่ `k-1`
- `C(i, k, j)` = ต้นทุนของการให้ cue `i` ครอบ segments `k..j`
- เราหาค่า `k` ที่ทำให้ผลรวมต่ำสุด

**Boundary:** `dp[0][0] = 0`, ส่วนอื่น = ∞

**Final answer:**

$$
j^* = \arg\min_j \text{dp}[M][j]
$$

จากนั้น **backtrack** ผ่าน `prev[i][j]` (ที่เก็บ k* ที่เลือกในแต่ละ step)
เพื่อคืน partition ของแต่ละ cue → ได้ `(start, end)` ใหม่

#### 4.4.5 Cost Function — เจาะลึกแต่ละ Term

$$
C(i, k, j) = T_1 + T_2 + w_D \cdot T_3 + w_G \cdot T_4 - w_S \cdot T_5
$$

| Term | สูตร | Weight | บทบาท |
| :---: | --- | :---: | --- |
| $T_1$ | $\lvert \text{cue}_i.\text{start} - \text{seg}_k.\text{start} \rvert$ | 1 | "Gravity": ดึง start ของ cue ไม่ให้หลุดจากเวลาเดิม |
| $T_2$ | $\lvert \text{cue}_i.\text{end} - \text{seg}_j.\text{end} \rvert$ | 1 | Gravity ที่ end |
| $T_3$ | $\lvert \text{cue\_dur} - \text{group\_dur} \rvert$ | $w_D=2$ | บังคับให้กลุ่ม segments ยาวพอๆ กับ cue เดิม |
| $T_4$ | $\sum \max(0, \text{seg}_{p+1}.\text{start} - \text{seg}_p.\text{end})$ | $w_G=8$ | **สำคัญสุด:** ลงโทษ "รู" ระหว่าง segments |
| $T_5$ | $\sum_s \text{sim}[i][s]$ | $w_S=6$ | Reward (ลบในสูตร): similarity สูง ⇒ cost ต่ำ |

#### 4.4.6 ทำไมแต่ละ Term ถึงสำคัญ — อธิบายเชิงรูปธรรม

**$T_1, T_2$ — Gravity terms:**

หากไม่มี gravity DP อาจ shift cue ไปไกลมากเพื่อเพิ่ม similarity (เช่น
ย้าย cue จากตำแหน่ง 5s ไป 50s เพราะ similarity สูงกว่า) → ไม่สมเหตุสมผล
เพราะคำบรรยายไม่ควรห่างจากเสียงพูดไปหลักสิบวินาที

Gravity term ทำให้ DP **ไม่ย้ายไกลโดยไม่มีเหตุผล** — ย้ายได้แต่ต้อง
"จ่าย" ค่า gravity

**$T_3$ — Duration penalty:**

ถ้า cue ยาว 5s แต่ DP จับคู่กับกลุ่ม segments ยาว 0.5s → ผิดธรรมชาติ
(คนแสดงท่ามือไม่เร็วขนาดนั้น) → ลงโทษ

$w_D = 2$ คือ "ตึง" ระดับกลาง (ไม่บังคับเป๊ะ แต่ไม่ปล่อยฟรี)

**$T_4$ — Gap penalty (สำคัญที่สุด):**

ระหว่าง segments ที่อยู่ใน group เดียวกัน บางครั้งมี **gap** (ผู้แปล
หยุดมือชั่วครู่) — gap ใหญ่ = น่าจะไม่ใช่ท่ามือต่อเนื่อง → ไม่ควรอยู่
group เดียวกัน

$$
\text{gap}(k, j) = \sum_{p=k}^{j-1} \max(0, \text{seg}_{p+1}.\text{start} - \text{seg}_p.\text{end})
$$

$w_G = 8$ (สูงที่สุดในทุก term) เพราะ gap ใหญ่ส่งสัญญาณชัดเจนว่า DP
รวม segments ผิดกลุ่ม

**$T_5$ — Similarity reward:**

ถ้า text ของ cue **ใกล้ semantic** ของ sign segments → similarity สูง
→ cost ลด → DP ชอบกลุ่มนี้

$w_S = 6$ คือ balance ระหว่าง "เชื่อ SignCLIP" กับ "เชื่อ timing"

#### 4.4.7 Trade-off: Gravity vs Similarity

นี่คือ trade-off สำคัญที่สุด:

```text
                 cue เดิมที่เวลา 5s
                       │
                       ▼
T1/T2 อยากให้ start ≈ 5s
   (ตามเสียงพูด)
                                            T5 อยากให้ start ≈ 7s
                                          (ตามท่ามือจริง — similarity สูง)
                       │                          │
                       └─────────┬────────────────┘
                                 ▼
                     DP เลือกค่ากลาง — เช่น 6.5s
                     (ขึ้นกับ weight ratio w_S = 6 vs T1/T2 = 1)
```

→ ผลคือ **cue ค่อยๆ ขยับไปทางท่ามือจริง** ตาม weight — ไม่ใช่ snap ไป
100% เลย

#### 4.4.8 Sliding Window — ลด complexity

**ปัญหา complexity:** Naive DP มี complexity $O(M \cdot N^2)$:

$$
119 \times 2780^2 \approx 9.2 \times 10^8 \text{ operations}
$$

→ ช้าเกินไป (หลายนาที per video)

**วิธีแก้:** จำกัด search space ของแต่ละ cue ให้อยู่ใน **window ของ
segments ที่ใกล้ midpoint ของ cue ที่สุด**:

```python
W = 40  # --dp_window_size
for each cue_i:
    cue_mid = (cue_i.start + cue_i.end) / 2
    cand    = argsort(|seg_mids - cue_mid|)[:W]
    cand_min_i, cand_max_i = min(cand), max(cand)
    # ใน DP, จำกัด k, j ∈ [cand_min_i, cand_max_i]
```

**Complexity ใหม่:** $O(M \cdot W^2) = 119 \times 40^2 \approx 1.9 \times 10^5$
→ **เร็วขึ้น 5,000 เท่า**

**ทำไม W = 40 พอ?**

- แต่ละ cue กินเวลา ~3–5 วินาที (เฉลี่ย)
- แต่ละ SIGN segment ~250 ms
- ดังนั้น 1 cue ≈ 12–20 segments → window 40 ครอบคลุมเหลือเฟือ

#### 4.4.9 Precomputations

หลายสูตรใน cost function ใช้ **sum ของช่วง** ซึ่งถ้าคำนวณซ้ำในทุก loop
จะช้า → precompute เป็น **cumulative sum** ก่อน เพื่อให้ลูคในระหว่าง DP
ใช้ $O(1)$:

| Precompute | สิ่งที่เก็บ | ใช้ใน |
| --- | --- | --- |
| `gap_cost[k][j]` | cumulative gap sum ระหว่าง segments k..j | $T_4$ |
| `sim_cumsum[i][j]` | prefix sum ตาม column ของ similarity matrix | $T_5$ — $\text{sim}[i][k..j-1] = \text{sim\_cumsum}[i][j] - \text{sim\_cumsum}[i][k]$ |
| `softmax_normalize(sim)` | normalize row-wise ของ similarity | $T_5$ — ลด noise |

#### 4.4.10 ทำไมต้อง Softmax Similarity?

Raw SignCLIP cosine ของ TSL กระจุกอยู่ที่ **~0.2–0.4 ทั่วทั้ง matrix** —
ค่าระหว่างคู่ที่ "ดี" กับ "ไม่ดี" ต่างกันแค่ ~0.1

→ $T_5$ จะอ่อนเกินไป (signal น้อย, noise มาก)

**Softmax-row normalize** ดันค่าสูงสุดในแต่ละ row ให้ใกล้ 1 และค่าที่เหลือ
ใกล้ 0 → DP **เห็นชัดเจน** ว่า segment ไหนเหมาะกับ cue ไหน

#### 4.4.11 Subgroup Refinement (Post-DP)

หลัง DP คืน partition ของแต่ละ cue — แต่บางครั้งกลุ่มของ cue *i* มี
**resting segments** ปนอยู่ (ผู้แปลหยุดมือระหว่างท่า) ที่ไม่ควรอยู่ใน
timestamp สุดท้าย

**Algorithm:**

1. ตัด group ออกเป็น **contiguous subgroups** โดย segments ที่ gap > `--dp_max_gap` (=6s) จะตัดแยก
2. คำนวณ `cost_for_subgroup()` ของแต่ละ subgroup (ใช้สูตรเดียวกับ DP)
3. เลือก subgroup ที่ cost ต่ำสุด → ใช้ start ของ segment แรก + end ของ segment สุดท้ายของ subgroup นั้น

**ตัวอย่าง:**

```text
cue i original group:  [s120 s121 ... s132 .. (4s gap) .. s145 ... s148]
                                              ▲
                                              gap > 6s? ไม่ → ไม่ตัด
                                              gap > 6s? ใช่ → ตัด
                                              ▼
subgroups:             [s120 s121 ... s132]    [s145 ... s148]
                       cost = 5.2              cost = 12.8
                       ✓ ใช้ subgroup นี้
                       
final cue i timestamp = (s120.start, s132.end)
```

> ⚠️ **Side-effect ที่สำคัญ:** subgroup refinement บางครั้งดัน end ของ
> cue *i* ไปข้างหน้าจนทับ start ของ cue *i+1* → **ทำให้เกิด overlap**
> (อ่านต่อใน §4.5)

### 4.5 ทำไม Overlap ถึงเกิด (สำคัญ — section พิเศษ)

> **ก่อนทำ overlap fix: 86–88% ของ adjacent cue pairs มี overlap**
> ใช่แล้ว — เกือบทุกคู่ติดกัน

นี่ไม่ใช่ **bug** แต่เป็น **design consequence** ของ DP — เข้าใจสาเหตุก่อน
จะแก้ได้ตรงจุด

#### 4.5.1 สาเหตุที่ 1 — DP State ไม่เก็บ End Time ของ Cue ก่อนหน้า

**ทบทวน DP state:**

$$
\text{dp}[i][j] = \text{ต้นทุนต่ำสุด, cue } i \text{ จบที่ segment index } j
$$

สังเกตว่า state เก็บ **segment index *j*** — ไม่ใช่ **เวลา *t***

ดังนั้น เมื่อ DP จะ assign cue *i+1*:

- รู้ว่า cue *i* จบที่ segment *j*
- แต่ **ไม่รู้** ว่า "เวลาจริง" ของ end คือเท่าไร (เพราะ subgroup refinement
  จะแก้ end time หลัง backtrack)
- → ไม่มี constraint ว่า start ของ cue *i+1* ต้อง ≥ end ของ cue *i*

#### 4.5.2 สาเหตุที่ 2 — Subgroup Refinement ดัน End ไปข้างหน้า

จาก §4.4.11 — subgroup refinement เลือก subgroup ที่ cost ต่ำสุดของแต่ละ
cue **อย่างอิสระ** → end time ของ cue *i* อาจจบช้ากว่า start time ของ
cue *i+1*

**ตัวอย่างเชิงตัวเลข:**

```text
DP backtrack ให้:
  cue 5  → group segments 120-135 → subgroup ที่ดีสุด 120-128
                                   → end = 65.2s
  cue 6  → group segments 129-145 → subgroup ที่ดีสุด 130-140
                                   → start = 64.8s

ผล: cue 5 end (65.2s)  >  cue 6 start (64.8s)
    → overlap = 0.4s
```

→ ไม่มีใครผิด DP หา timestamp ที่ minimize cost ของแต่ละ cue **เป็นอิสระ**
หลัง backtrack

#### 4.5.3 สาเหตุที่ 3 — Design ของ SEA สำหรับ BSL ไม่ใช่ TSL

SEA ออกแบบมาสำหรับ **BSL** ที่:

- มี sign density สูง (มือทำท่าต่อเนื่อง)
- Segments เรียงต่อกันแน่น (gap น้อย)
- → overlap ที่เกิดขึ้น **น้อยมาก** ตามธรรมชาติ

แต่ **TSL** มี:

- Resting periods มากกว่า (หยุดมือเพื่อ pause)
- Gap ระหว่าง groups ใหญ่กว่า
- → โอกาส overlap สูงขึ้น

#### 4.5.4 ทำไม DP ไม่ใส่ Non-overlap Constraint ตั้งแต่แรก?

**สมมติเราต้องการบังคับว่า** $\text{end}_i \leq \text{start}_{i+1}$:

- ต้องเพิ่ม **dimension ของ state** → จาก `dp[i][j]` เป็น `dp[i][j][t_end]`
- หรือทำ **2-pass DP** ที่ pass แรกหา candidate, pass สองบังคับ ordering
- → ซับซ้อนขึ้นมาก + ช้าขึ้น

**SEA เลือกอีกแนวทาง:** แยก overlap fix เป็น **post-processing step** ที่
แก้ใน $O(N)$ pass เดียว — เพราะ:

1. การ clamp end time **ไม่กระทบ start time** → metric หลัก (mean offset)
   ไม่เปลี่ยน
2. ทำได้ใน $O(N)$ — เร็วมาก
3. ไม่ทำลาย alignment คุณภาพ — overlap ส่วนใหญ่น้อยกว่า 0.5s

### 4.6 Step 5: Overlap Fix

```python
for i in range(len(cues) - 1):
    if cues[i].end > cues[i + 1].start:
        cues[i].end = cues[i + 1].start   # clamp
```

**O(N) single-pass clamp** — วน loop หนึ่งครั้งจากต้นถึงท้าย

#### 4.6.1 ทำไมแตะแค่ End ไม่แตะ Start?

นี่คือ **design choice ที่ Tao Important:**

| | Start time | End time |
| --- | --- | --- |
| ที่มา | DP คำนวณมา **อย่างระมัดระวัง** | "ขอบหลัง" — ยืดหยุ่นกว่า |
| ความหมาย | "best estimate" ว่าผู้แปลเริ่มท่ามือเมื่อไร | กำหนดเพื่อให้ subtitle แสดงนานพอ |
| Metric หลัก | วัด start offset | ไม่ค่อยวัด |
| → | **ห้ามแตะ** | **คลามป์ได้** |

#### 4.6.2 ยืนยันจากผลจริง

| Metric | Before fix | After fix | เปลี่ยน? |
| --- | --- | --- | :---: |
| Mean offset | −0.16s | −0.16s | ❌ ไม่ |
| ±1s | 73.9% | 73.9% | ❌ ไม่ |
| ±2s | 95.0% | 95.0% | ❌ ไม่ |
| Overlap | 88.1% | **0.0%** | ✅ ใช่ |

→ Overlap fix ปลอดภัย 100% สำหรับ start metrics

### 4.7 Step 6: Evaluation

#### 4.7.1 ทำไมไม่ใช้ Original SEA Evaluation?

**Original SEA evaluation:**

- ออกแบบสำหรับ BOBSL scale (20,000+ sentences)
- ใช้ **text lookup** เพื่อ match predicted text → GT text
- คำนวณ frame-level accuracy + F1@IoU บนหลาย videos

**ปัญหากับโปรเจกต์เรา:**

- มีแค่ 1 video → frame-level metrics ไม่ stable
- `CC_Input` กับ `CC_Aligned` มี text **ไม่ตรงกัน** (annotator แก้ wording
  เวลา align) → text lookup match ได้แค่ ~50/119 entries

#### 4.7.2 Index-based Matching (วิธีของเรา)

สังเกตว่า **`CC_Input` และ `CC_Aligned` มี 119 entries เท่ากันและ
เรียงตามลำดับเดียวกัน** — `pred[i]` คือคำเดียวกับ `gt[i]` แน่ๆ
(แม้ text ต่างกันเล็กน้อย)

→ ใช้ **index-based pairing**:

```python
for i in range(119):
    pred_cue = predictions[i]
    gt_cue   = ground_truth[i]
    offset[i] = pred_cue.start - gt_cue.start
```

→ Match ครบ **119/119** ทุก experiment

#### 4.7.3 Metrics ที่ใช้

| Metric | คำอธิบาย | ทำไมสำคัญ |
| --- | --- | --- |
| Mean offset (signed) | average ของ `pred.start - gt.start` | บอก systematic bias |
| Median offset | median เดียวกัน | ไม่ sensitive ต่อ outlier |
| Stdev offset | standard deviation | บอกความนิ่ง |
| % within ±1s / ±2s / ±3s | % cues ที่อยู่ใน tolerance | บอก practical accuracy |
| Overlap % | % consecutive pairs ที่ overlap | บอก quality หลัง fix |
| Frame accuracy | label-wise accuracy ที่ FPS=25 | comparable กับ SEA paper |
| F1 @ IoU ≥ 0.1/0.25/0.5 | precision/recall ที่ threshold ต่างๆ | comparable กับ SEA paper |

#### 4.7.4 ผลของ Champion Run (C_MULTI หลัง Overlap Fix)

```text
Mean offset:    −0.16 s     ← เกือบศูนย์
±1s coverage:   73.9%        ← 3/4 ของ cues
±2s coverage:   95.0%
±3s coverage:   100.0%       ← ทุก cue
Overlap:        0.0%
Frame acc:      82.6%
F1 @ 0.50:      88.2%
```

---

## 5. Task 2 — Gloss Labeling ระดับท่ามือ

### 5.1 Task 2 เพิ่มอะไรจาก Task 1?

#### 5.1.1 ความต่างของ Granularity

```text
Timeline ──────────────────────────────────────────────►

Task 1: [── "การเปรียบเทียบและการเรียงลำดับ" ─────]    ← 1 cue / 1 sentence
        (1 entry = 1 ประโยค, ~4-5s ต่อ entry)

Task 2: [ผายมือ][เปรียบ][เรียงลำดับ][ตัวเลข][....]    ← 1 entry / 1 sign gesture
        (1 entry = 1 ท่ามือ, ~250ms ต่อ entry)
```

| | **Task 1** | **Task 2** |
| --- | --- | --- |
| Input | `CC_Input` (119 sentences) | `Gloss` หรือ `Gloss_Input` (119 sentences with tokens) |
| Output | sentence-aligned VTT | token-aligned VTT |
| #Entries | 119 | 852 (Gloss) or 889 (Gloss_Input) |
| GT | `CC_Aligned` (119) | `Gloss Labeling` (852) |
| DP scope | **Global** (1 ใหญ่: 119 × 2780) | **Per-sentence** (119 ใบเล็ก: T × K) |

#### 5.1.2 ประโยชน์ของ Task 2

1. **Annotation ละเอียดสำหรับ linguistic research** — รู้ว่าท่ามือ
   เฉพาะคำไหนเริ่ม-จบเมื่อไร
2. **Sign-level retrieval** — search ท่ามือเดี่ยวข้ามวิดีโอได้
3. **Training data สำหรับ sign-level models** — สร้าง dataset ระดับท่ามือ
   จาก video ที่มี gloss tier เท่านั้น (ไม่ต้อง annotate รายท่ามือเอง)

### 5.2 Per-sentence Monotonic DP

#### 5.2.1 ทำไม Per-sentence ไม่ใช่ Global?

**Task 1** ใช้ global DP กับทั้ง video (119 × 2780) เพราะต้องการ ordering
ทั่วทั้งวิดีโอ

**Task 2** ทำงานใน **ขอบเขตของแต่ละ Gloss sentence** เท่านั้น:

```text
Gloss sentence_5 (start=15.2s, end=18.7s, "ผายมือ เปรียบเทียบ เรียง")
                                                 │
                                                 ├─► tokens: T=3
                                                 │
candidate SIGN segments มีอยู่:
  mid ∈ [15.2, 18.7]  →  segments [s48, s49, s50, s51, s52, s53]  → K=6
                                                 │
                                                 ▼
                              Per-sentence DP: 3 tokens × 6 segments
```

→ Task 2 รัน DP **119 ครั้ง** (1 ครั้ง per Gloss sentence) แต่ละครั้ง
ขนาดเล็กมาก (T~7, K~30)

#### 5.2.2 DP Recurrence

$$
\text{dp}[t][j] = \min_{k \in [t,\, j]} \Bigl(
\text{dp}[t-1][k-1]
\underbrace{- \textstyle\sum_{p=k-1}^{j-1} \text{sim}[t-1][p]}_{\text{negative similarity}}
+ w_{\text{gap}} \cdot \text{gap}_{\text{total}}(k, j)
+ w_{\text{cov}} \cdot |\text{group\_dur} - \tfrac{\text{sentence\_dur}}{T}|
\Bigr)
$$

#### 5.2.3 Cost Terms ของ Task 2

| Term | Default weight | บทบาท |
| --- | :---: | --- |
| Negative similarity | (implicit 1) | reward สำหรับ semantic match |
| Gap penalty | `--gap-penalty` 2.0 | ลด group ที่มี gap ใหญ่ |
| Coverage penalty | `--coverage-penalty` 0.5 | บังคับ duration ของแต่ละ token ≈ sentence_dur/T |
| `--window-pad` | 0.5 s | ขยาย window ถ้า K < T |

> **เปรียบเทียบกับ Task 1:** Task 2 **ไม่มี $T_1, T_2$ gravity term** —
> เพราะ token แต่ละตัวไม่มี "start ที่ควรอยู่" เป็น scope ของ sentence
> เท่านั้น

#### 5.2.4 ทำไม "Monotonic" DP?

**Monotonic** = ขอบของ token *t* ต้องอยู่หลังขอบของ token *t-1*

```text
Gloss sentence: [ผายมือ][เปรียบ][เรียง]
                  ↓        ↓       ↓
                k=0..1   k=2..3   k=4..5
                  
✓ valid:   [0-1][2-3][4-5]
✗ invalid: [0-1][2-3][1-2]  ← เพราะ "เรียง" overlaps กับ "เปรียบ"
```

DP บังคับ $k_t > j_{t-1}$ ใน transition → ensures contiguous + non-overlapping
ranges

#### 5.2.5 Complexity

$O(T \cdot K^2)$ per sentence — T~7, K~30 → ~6,300 ops per sentence
→ 119 sentences เสร็จในไม่ถึง 1 วินาที

### 5.3 Token Embedding Cache

#### 5.3.1 ปัญหา: Token เดิมๆ ถูก embed หลายครั้ง

```text
Gloss tier มี 852 tokens แต่หลายตัวซ้ำกัน:
  "ผายมือ" ปรากฏ ~15 ครั้ง
  "การ" ปรากฏ ~30 ครั้ง
  "(สำหรับ)" ปรากฏ ~8 ครั้ง
  → unique tokens จริงๆ มีแค่ ~192 ตัว
```

→ ถ้าเรียก SignCLIP encoder 852 ครั้ง = **เปลือง 4.4 เท่า**

#### 5.3.2 วิธีแก้: NPZ Cache

```python
cache_key = f"{language_tag}||{token}"
# ตัวอย่าง: "<en> <bfi>||ผายมือ"

if cache_key not in cache:
    cache[cache_key] = embed(token, language_tag)  # call SignCLIP

token_emb = cache[cache_key]  # O(1) lookup
```

Cache เก็บใน `.npz` file → reuse ระหว่าง runs:

- `04__Gloss.npz` — 192 unique tokens
- `04__Gloss_Input.npz` — 185 unique tokens

#### 5.3.3 ทำไม Key Includes Language Tag?

ถ้าใช้แค่ token เป็น key → ถ้าเปลี่ยน `--language-tag` (เช่นจาก `<en> <bfi>`
เป็น `<en> <ase>`) cache จะคืน vector ผิด

→ ใช้ `f"{language_tag}||{token}"` เป็น key เพื่อ isolation

### 5.4 Evaluation ด้วย IoU

#### 5.4.1 ทำไมต้องใช้ IoU ไม่ใช่ Index-based?

Task 1 ใช้ index-based เพราะ pred และ GT มีจำนวนเท่ากัน (119 = 119)

Task 2 มี:

- `--tier Gloss` → 852 predictions vs 852 GT entries
- `--tier Gloss_Input` → 889 predictions vs 852 GT entries

ในกรณี Gloss_Input มี **prediction ส่วนเกิน 37** → ไม่สามารถ pair 1:1 ตรงๆ
ได้ → ต้องใช้ **best-IoU pairing**

#### 5.4.2 Best-IoU Pairing

สำหรับแต่ละ prediction หา GT entry ที่ overlap สูงสุดด้วย IoU:

$$
\text{IoU}(p, g) = \frac{\max(0, \min(p_e, g_e) - \max(p_s, g_s))}{\max(p_e, g_e) - \min(p_s, g_s)}
$$

```python
for pred in predictions:
    best_iou, best_gt = 0, None
    for gt in ground_truths:
        if overlap(pred, gt) > 0:
            io = iou(pred, gt)
            if io > best_iou:
                best_iou, best_gt = io, gt
    record(pred, best_gt, best_iou)
```

#### 5.4.3 Metrics

| Metric | คำอธิบาย |
| --- | --- |
| Mean / Median IoU | average overlap quality |
| % IoU ≥ 0.5 | % predictions ที่ overlap GT มากกว่าครึ่ง |
| % IoU ≥ 0.3 | tolerant threshold |
| % any overlap | % predictions ที่แตะ GT |
| Mean signed start / end offset | timing bias |
| Exact text match | string equality (มี caveat — ดู §5.5) |

### 5.5 Ablation: Gloss vs Gloss_Input

#### 5.5.1 ตัวเลขสรุป

| Metric | **Gloss** | **Gloss_Input** | Δ |
| --- | ---: | ---: | ---: |
| #Predictions | 852 | 889 | −37 |
| Mean IoU | **0.4901** | 0.4199 | +7.0 pp |
| % IoU ≥ 0.5 | **48.4%** | 38.9% | +9.4 pp |
| % IoU ≥ 0.3 | **77.0%** | 66.0% | +11.0 pp |
| % zero overlap | 2.5% | 6.6% | −4.2 pp |
| Mean abs start offset | 0.188s | 0.212s | −24 ms |
| Exact text match | 65.1% | 10.6% | +54.5 pp \* |

\* exact text match มี **leakage component** — ดู §5.5.3

#### 5.5.2 ทำไม Gloss ชนะ

1. **Token count = GT count** — Gloss มี 852 tokens เท่ากับ 852 GT entries
   → DP มี degrees of freedom **ตรงกับขนาดที่ออกแบบ**
2. **Sentence window coverage** — Gloss ครอบ GT entries ได้ดีกว่า (97.77%
   vs 88.97%) → ทิ้ง GT น้อยกว่า
3. **Token boundary alignment** — Gloss tokens **71.2% positional match กับ
   GT** vs Gloss_Input 4.9% → annotator น่าจะใช้ Gloss เป็น base ในการ
   build GT

#### 5.5.3 Caveat: Ground-truth Leakage

> ก่อนเชื่อ "Gloss ดีกว่า 100%" — ต้องตรวจ data leakage

| Comparison | Token positional match กับ GT | % |
| --- | --- | --- |
| Gloss tokens vs GT texts | 607 / 852 | **71.2%** |
| Gloss_Input tokens vs GT texts | 42 / 852 | 4.9% |

→ `Gloss` token list มี **71.2% positional match กับ GT** = annotator
น่าจะใช้ Gloss เป็นจุดตั้งต้นในการ build GT

→ **65% exact-text-match ของ Gloss ส่วนหนึ่งเกิดจาก structural alignment**
ไม่ใช่ความสามารถของโมเดล

**Metric ที่ "fair":**

| Metric | กระทบจาก leakage? |
| --- | --- |
| Exact text match | **ใช่ — กระทบหนัก** |
| Mean IoU | กระทบเล็กน้อย |
| % IoU ≥ 0.3 / 0.5 | **กระทบน้อย** |
| Mean abs offset | **ไม่กระทบ** |

→ รายงาน metric หลักเป็น **IoU + offset**, ไม่ใช่ text match

#### 5.5.4 Recommendation

**Default ของ Task 2:** เปลี่ยนเป็น `--tier Gloss` (ปัจจุบัน default ยัง
เป็น `Gloss_Input` เพื่อ backward-compat)

```powershell
python example_alignment\align_gloss_labels.py --tier Gloss
```

---

## 6. Design Decisions รวบยอด

ตารางสรุป **ทำไมถึงเลือกอย่างนี้** ของ design decisions ที่สำคัญ:

| Decision | เลือกอะไร | ทำไม |
| --- | --- | --- |
| Pose vs raw video | pose (MediaPipe Holistic) | ลด data ~3800x, focus เฉพาะข้อมูลที่เกี่ยวข้อง |
| Segmentation algorithm | E4s-1 BiGRU | BSL-proven, transfer ดีมา TSL |
| Embedding model | Multilingual | Cross-lingual transfer ดีสุดสำหรับ TSL |
| Subtitle text input | Gloss_Input | ตรงกับ sign content มากกว่า CC text |
| DP cost: gravity ratio | $w_S = 6$, $T_1, T_2 = 1$ | balance: เชื่อ similarity 6 เท่าของ timing |
| DP cost: gap penalty | $w_G = 8$ (สูงสุด) | gap คือสัญญาณชัดที่สุดของ wrong group |
| Sliding window size | $W = 40$ | ครอบ 5,000× ของ naive แต่ยังเหลือ buffer |
| Similarity normalization | Softmax-row | raw cosine variance ต่ำเกินไป |
| Overlap handling | Post-processing clamp | คง start metric ที่สำคัญที่สุด + เร็ว O(N) |
| Evaluation method | Index-based (Task 1) | match ทั้ง 119/119 (text-lookup match แค่ 69) |
| Task 2 DP scope | Per-sentence | T~7, K~30 → เล็กมาก, รัน 119 ครั้งเร็วกว่า global |
| Task 2 default tier | Gloss_Input → Gloss (recommended) | ablation พิสูจน์ Gloss ดีกว่าทุก metric |
| Token embedding cache | NPZ file with `lang_tag||token` key | reuse + safe across language tags |

---

## 7. สรุปภาพรวม — Mental Model

### 7.1 ภาพรวม 1 ประโยค

> SEA Pipeline แปลง **video + EAF** เป็น **aligned subtitles + per-gesture
> annotations** ผ่าน 4 ขั้นตอน: Pose → Segment → Embed → Align (+ overlap fix)

### 7.2 ภาพรวมเป็นรูป

```text
                    ┌───────────────────────────────────────────────┐
                    │  Input: 04.mp4 (11 นาที, 60fps)               │
                    │         Test.eaf (CC_Input, Gloss, …)         │
                    └────────────┬──────────────────────────────────┘
                                 │
                                 ▼
                    ┌───────────────────────────────────────────────┐
                    │  Pose Estimation (MediaPipe Holistic)         │
                    │  → 04.pose (543 landmarks × 39,600 frames)    │
                    └────────────┬──────────────────────────────────┘
                                 │
                                 ▼
                    ┌───────────────────────────────────────────────┐
                    │  Sign Segmentation (E4s-1 BiGRU)              │
                    │  → 2780 SIGN segments + 418 SENTENCE segments │
                    └────────────┬──────────────────────────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
        ┌──────────────────┐         ┌──────────────────┐
        │ Sign embeddings  │         │ Text embeddings  │
        │ (2780 × 768)     │         │ (119 × 768)      │
        │ - BSL            │         │ - CC_Input       │
        │ - Multilingual   │         │ - Gloss_Input    │
        │ - ASL            │         │ - per token (T2) │
        └────────┬─────────┘         └────────┬─────────┘
                 │                            │
                 └────────────┬───────────────┘
                              ▼
                ┌──────────────────────────────┐
                │ TASK 1: Sentence-level DP    │
                │ - cost: gravity+gap+sim+dur  │
                │ - sliding window W=40         │
                │ - subgroup refinement         │
                │ - overlap fix (post)          │
                │ → aligned_output_*/04.vtt     │
                └──────────────────────────────┘
                              ▼
                ┌──────────────────────────────┐
                │ TASK 2: Per-sentence DP      │
                │ - per-token monotonic align  │
                │ - cost: sim+gap+coverage     │
                │ - cached token embeddings    │
                │ → gloss_labels_pred.{csv,vtt}│
                └──────────────────────────────┘
                              ▼
                ┌──────────────────────────────┐
                │ Evaluation                   │
                │ - Task 1: index-based + IoU  │
                │ - Task 2: best-IoU pairing   │
                │ → evaluation_*.csv           │
                └──────────────────────────────┘
                              ▼
                ┌──────────────────────────────┐
                │ EAF builders                 │
                │ - Test_comparison.eaf (23 t.)│
                │ - Test_best.eaf (9 tiers)    │
                └──────────────────────────────┘
```

### 7.3 จุดที่ต้องจำให้ได้

1. **SEA = Segment + Embed + Align** — เป็น 3 ขั้นตอน design ที่แยกชัดเจน
2. **DP เป็นหัวใจ** — global DP สำหรับ Task 1, per-sentence DP สำหรับ Task 2
3. **Cost function มี 5 terms** ใน Task 1 (gravity×2, duration, gap, similarity)
4. **Sliding window W=40** ลด complexity 5,000 เท่า — Numba JIT ลดอีก 10–50×
5. **Overlap คือ design quirk** ของ DP — ไม่ใช่ bug — แก้ด้วย post-process clamp
6. **Overlap fix แตะแค่ end ไม่แตะ start** — เพราะ start คือสิ่งที่ DP คิดมาดี
7. **C_MULTI ⭐** (Multilingual + Gloss text) คือ best run ของ Task 1
8. **`--tier Gloss`** (ไม่ใช่ default `Gloss_Input`) คือ best ของ Task 2
9. **Index-based evaluation** สำหรับ Task 1 (match ครบ 119/119)
10. **Best-IoU pairing** สำหรับ Task 2 (เพราะ #pred อาจ ≠ #GT)

### 7.4 ถ้าจะอธิบาย SEA ให้เพื่อน 30 วินาที

> "เรา take video ภาษามือมา detect ท่ามือเดี่ยวด้วย pose + BiGRU → 2780
> ท่า embed ทั้ง sign กับ text ด้วย SignCLIP ให้อยู่ space เดียวกัน →
> DP จัดคู่ 119 ประโยคให้กับกลุ่ม segments ที่ minimize gravity + gap +
> similarity-cost → ได้ subtitle ที่ตรงกับท่ามือ mean offset −0.16s
> ส่วน Task 2 ทำคล้ายๆ แต่ลงลึกระดับ token ของ Gloss tier"

---

## ภาคผนวก — เอกสารอื่นที่ควรอ่านต่อ

| เอกสาร | เนื้อหา |
| --- | --- |
| [README.md](README.md) | Setup instructions, full commands per task |
| [Progress_20042026.md](Progress_20042026.md) | Initial pipeline + B2 baseline |
| [Progress_26042026.md](Progress_26042026.md) | 7 experiments + post-overlap |
| [Progress_04052026.md](Progress_04052026.md) | CC_Input / Gloss_Input + index-based eval |
| [Progress_09052026.md](Progress_09052026.md) | **Task 2 ablation: Gloss vs Gloss_Input** |
| [presentation_12052026.md](presentation_12052026.md) | Presentation deck (Part 1–3 slides) |
| [arXiv-2512.08094v1/](arXiv-2512.08094v1/) | SEA paper (PDF) — Jiang et al. 2025 |

---

**เขียนเมื่อ:** 12 พฤษภาคม 2569
**ใช้สำหรับ:** ทำให้ผู้อ่านใหม่เข้าใจ SEA project ทั้งหมดในเอกสารเดียว
