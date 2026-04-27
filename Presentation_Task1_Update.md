# SEA for Thai Sign Language — Progress Update
### Next Presentation Brief
> **อัปเดตล่าสุด:** 26 เมษายน 2569 | วิดีโอทดสอบ: "การเปรียบเทียบและเรียงลำดับ" (11 นาที)

---

## Slide 1 — Project Overview

**SEA = Segment → Embed → Align**

> นำระบบ SEA (arXiv:2512.08094, Oxford/ETH 2025) มาทดสอบกับภาษามือไทย (TSL) แบบ cross-lingual

**2 งานหลักที่กำลังทำ:**

| งาน | เป้าหมาย | สถานะ |
|---|---|---|
| **Task 1** | จัดเวลา CC subtitle ให้ตรงกับช่วงที่ผู้แปลแสดงท่ามือ | 🔄 กำลังปรับปรุง |
| **Task 2** | แยก gloss ระดับประโยคออกเป็น annotation รายท่ามือ | 🔄 Prototype รันได้แล้ว |

**ข้อมูลที่มี:**
- `04.mp4` — วิดีโอ 11.07 นาที (1920×1080, 60fps)
- EAF annotation 4 tier: **CC** (172), **CC_Aligned** (119), **Gloss** (119), **Gloss Labeling** (852)
- `CC_Aligned` = ground truth ที่นักวิจัย annotate ด้วยมือ (ใช้ evaluate Task 1)

---

## Slide 2 — Full Pipeline (ภาพรวม)

```
┌─────────────────────────────────────────────────────────────────────┐
│  INPUT: 04.mp4  +  EAF (CC, CC_Aligned, Gloss, Gloss Labeling)     │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  Step 1: Extract CC    │  extract_cc_from_eaf.py
          │  EAF → WebVTT          │  → subtitles/04.vtt
          │  172 cues              │    (timestamp จากเสียงพูด)
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  Step 2: Pose          │  videos_to_poses
          │  Estimation            │  MediaPipe Holistic
          │  → 04.pose (358 MB)    │  543 landmarks × ทุกเฟรม
          └────────────┬───────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  Step 3: Segmentation  │  segmentation.py (E4s-1 model)
          │  ตรวจจับท่ามือจาก pose  │  --b 30  --o 50
          │  SIGN:  2,780 segments │  → segmentation_output/04.eaf
          │  SENT:    418 segments │
          └─────────┬──────┬───────┘
                    │      │
          ┌─────────┘      └──────────────────────┐
          ▼                                       ▼
┌──────────────────────┐             ┌────────────────────────────┐
│  Step 4a             │             │  Step 4b: SignCLIP Embed    │
│  Sign Embedding      │             │  Text Embedding             │
│  pose → 768-dim vec  │             │  subtitle text → 768-dim   │
│  × 3 models          │             │  × 3 models × 2 text types  │
│  → seg_embedding/    │             │  → sub_embedding/           │
│    *.npy (2780×768)  │             │    *.npy (172×768)          │
└──────────┬───────────┘             └──────────────┬─────────────┘
           │                                        │
           └───────────────┬────────────────────────┘
                           ▼
          ┌────────────────────────────────────────┐
          │  Step 5: DP Alignment                  │  align.py
          │  DP + embedding similarity             │
          │  เชื่อมต่อ subtitle → sign segment      │
          │  → aligned_output_*/04.vtt             │
          │                                        │
          │  7 experiments (B2, B_MULTI, C_MULTI,  │
          │  C_MULTI_word, D_ASL, D_ASL_gloss,     │
          │  D_ASL_word)                           │
          └────────────────┬───────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────────────┐
          │  Step 6: Post-processing               │  fix_overlap_vtt.py
          │  Overlap fix (clamp end time)          │
          │  overlap ~88% → 0%                     │
          │  → 04_no_overlap.vtt (×7)              │
          └────────────────┬───────────────────────┘
                           │
                           ▼
          ┌────────────────────────────────────────┐
          │  Step 7: Evaluation                    │  evaluate_all_to_csv.py
          │  เทียบกับ CC_Aligned ground truth      │
          │  วัด offset, ±1s/±2s/±3s, overlap     │
          │  → evaluation_task1_results.csv        │
          └────────────────────────────────────────┘
```

**คำอธิบายแต่ละขั้นตอน (สรุปสั้น):**

| Step | ทำอะไร | ทำไม |
|---|---|---|
| **Step 1** Extract CC | แยก tier CC ออกจากไฟล์ EAF (XML) แปลงเป็น WebVTT | SEA รับ input เป็น VTT เท่านั้น EAF เป็นรูปแบบ ELAN annotation ที่ต้องแปลงก่อน |
| **Step 2** Pose Estimation | รันวิดีโอผ่าน MediaPipe Holistic → ได้ตำแหน่งข้อต่อ 543 จุดต่อเฟรม | ระบบไม่ได้อ่าน pixels โดยตรง แต่อ่าน "การเคลื่อนไหวของมือ" ผ่านพิกัด landmark — ลดขนาดข้อมูลและ noise |
| **Step 3** Segmentation | โมเดล E4s-1 ทำนายทุกเฟรมว่า "กำลังแสดงท่ามือ" หรือ "พัก" แล้วรวมเป็น segment | ต้องรู้ว่าแต่ละ "ท่า" อยู่ที่ช่วงเวลาไหน ก่อนจะ align subtitle เข้าไป |
| **Step 4a** Sign Embedding | ส่ง pose ของแต่ละ sign segment เข้า SignCLIP → ได้ vector 768 มิติ | แปลงท่ามือเป็นตัวเลขที่เปรียบเทียบกับ text ได้ |
| **Step 4b** Text Embedding | ส่ง text ของแต่ละ subtitle cue เข้า SignCLIP text encoder → ได้ vector 768 มิติ | แปลง subtitle เป็นตัวเลขในพื้นที่เดียวกับท่ามือ เพื่อวัดความคล้ายกัน |
| **Step 5** DP Alignment | Dynamic Programming หาการจับคู่ subtitle↔sign segment ที่มี total cost ต่ำสุด (timing + embedding similarity) | หัวใจของระบบ — กำหนดว่า subtitle แต่ละตัวควรปรากฏที่เวลาไหนในวิดีโอ |
| **Step 6** Overlap Fix | clamp end time ของแต่ละ cue ไม่ให้เกิน start time ของ cue ถัดไป | DP ไม่ได้ห้าม overlap โดยตรง post-processing ขั้นนี้ทำให้ subtitle ไม่ซ้อนทับกัน |
| **Step 7** Evaluation | เทียบ predicted timestamp กับ ground truth (CC_Aligned) วัด mean offset และ % ที่ error < 1s | ตรวจสอบว่า alignment ดีแค่ไหนโดยใช้ manual annotation เป็นตัวอ้างอิง |

**โมเดล SignCLIP ที่ทดสอบ:**

| ชื่อ | Checkpoint | ภาษาที่ train |
|---|---|---|
| `bsl` | bobsl_finetune_checkpoint_best.pt | British SL |
| `multilingual` | baseline_temporal_checkpoint_best.pt | หลายภาษา |
| `asl` | asl_finetune_checkpoint_best.pt | American SL |

---

## Slide 2a — Step 1: Extract CC จาก EAF → VTT

**ปัญหาตั้งต้น:** EAF เป็น XML ซับซ้อน ระบบ SEA รับ WebVTT เท่านั้น

```
EAF (XML)                                          WebVTT
─────────────────────────────────────────────────────────────────
<TIER TIER_ID="CC">                      WEBVTT
  <ANNOTATION>
    <ALIGNABLE_ANNOTATION               1
       TIME_SLOT_REF1="ts40"            00:00:00.040 --> 00:00:31.890
       TIME_SLOT_REF2="ts41">           [เสียงดนตรี]
      <ANNOTATION_VALUE>
        [เสียงดนตรี]                    2
      </ANNOTATION_VALUE>               00:00:31.930 --> 00:00:35.600
    </ALIGNABLE_ANNOTATION>             สวัสดีครับ นักเรียน ...
  </ANNOTATION>
...
```

**กระบวนการ:**
1. อ่าน XML → สร้าง map `TIME_SLOT_ID → milliseconds`
2. ค้นหา tier ชื่อ `CC` → ดึง annotation ทุกตัว
3. แปลง ms → `HH:MM:SS.mmm` → เขียน VTT

**ผลลัพธ์:** `subtitles/04.vtt` — 172 cues พร้อม timestamp ที่ sync กับ**เสียงพูด** (ยังไม่ sync กับมือ)

> ⚠️ timestamp ใน CC sync กับ**เสียงครู** ไม่ใช่มือผู้แปล — นี่คือสิ่งที่ทั้ง pipeline พยายามแก้

---

## Slide 2b — Step 2: Pose Estimation (MediaPipe Holistic)

**เป้าหมาย:** แปลงวิดีโอ (pixels) → ตำแหน่งข้อต่อร่างกาย (coordinates) ที่ระบบ AI อ่านได้

```
ทุกเฟรมของวิดีโอ (60fps × 11 นาที = ~39,600 เฟรม)

   Frame t                                  .pose file
   ┌──────────────────────┐                ┌───────────────────────┐
   │   [ภาพวิดีโอ]         │   MediaPipe   │  frame 0:             │
   │                      │  ──────────►  │    pose[0..32]  (33)  │ ← ลำตัว
   │   ●─●  ●─●           │   Holistic    │    face[0..467] (468) │ ← ใบหน้า
   │   │    │             │               │    lh[0..20]   (21)  │ ← มือซ้าย
   │   ●    ●             │               │    rh[0..20]   (21)  │ ← มือขวา
   │                      │               │  frame 1: ...         │
   └──────────────────────┘               └───────────────────────┘
                                              รวม 543 landmarks/เฟรม
                                              แต่ละ landmark = (x, y, z, visibility)
```

**จุดสำคัญ:**
- มือ (42 จุด) คือส่วนที่ segmenter และ SignCLIP ให้ความสนใจมากที่สุด
- ใบหน้า (468 จุด) รวมอยู่ด้วยเพราะ facial expression เป็นส่วนหนึ่งของภาษามือ
- ไฟล์ `.pose` ขนาด 358 MB เพราะเก็บทุก landmark × ทุกเฟรม ในรูปแบบ binary float

> ภาพในใจ: แทนที่จะเก็บ pixels แสน pixel, เก็บแค่ **543 จุดพิกัด × 39,600 เฟรม** — ลดข้อมูลลงมากแต่ยังจับการเคลื่อนไหวได้

---

## Slide 2c — Step 3: Segmentation (E4s-1 — ตรวจจับท่ามือ)

**เป้าหมาย:** หาว่า "ช่วงเวลาไหนในวิดีโอที่ผู้แปลกำลังแสดงท่ามือ" และตัดออกมาเป็น segment

```
Timeline (ภาพในใจ):

  เวลา (วินาที)  0        10        20        30        40
                 │────────────────────────────────────────│
  ความเคลื่อนไหว  ░░░▓▓▓▓░░▓▓▓▓▓▓░░░▓▓▓░░░░░▓▓▓▓▓░░▓▓▓░░
  ของมือ         (░ = พัก/เคลื่อนน้อย, ▓ = กำลังแสดงท่า)

  SIGN segments  ╠══╣  ╠════╣  ╠══╣       ╠═══╣  ╠══╣
  (output)       [s1]  [s2]    [s3]        [s4]   [s5]

  SENTENCE       ╠══════════════╣          ╠══════════╣
  (output)            [sent1]                  [sent2]
```

**กระบวนการ:**
1. โมเดล E4s-1 อ่าน `.pose` → ทำนายทุกเฟรมว่า "กำลังแสดงท่ามือ (1)" หรือ "พัก (0)"
2. `--sign-b-threshold 30` — ถ้า probability > 30% = จุดเปลี่ยนท่า (boundary)
3. `--sign-o-threshold 50` — ถ้า probability > 50% = เริ่มท่าใหม่ (onset)
4. รวม consecutive frames ที่ผ่าน threshold → 1 SIGN segment

**ผลลัพธ์:**
- **2,780 SIGN segments** — แต่ละตัวแทนท่ามือ 1 ท่า (ความยาวเฉลี่ย ~0.2s)
- **418 SENTENCE segments** — กลุ่มท่ามือที่เป็นประโยค (ความยาวเฉลี่ย ~1.5s)

> ภาพในใจ: เหมือนมี "นักอ่านภาษามือ AI" ที่นั่งดูวิดีโอแล้วกดหยุดทุกครั้งที่เห็นว่า "ท่านี้จบแล้ว ท่าใหม่เริ่มแล้ว"

---

## Slide 2d — Step 4: SignCLIP Embedding (แปลงทุกอย่างเป็นตัวเลข)

**เป้าหมาย:** แปลงทั้ง "ท่ามือ" (จาก video) และ "ข้อความ subtitle" (จาก text) ให้อยู่ใน **พื้นที่เดียวกัน** เพื่อเปรียบเทียบกันได้

```
SignCLIP — แนวคิดหลัก (คล้าย CLIP ของ OpenAI แต่สำหรับภาษามือ)

  ท่ามือ (pose)           ╔═══════════════╗         Sign
  ────────────────────►   ║  Sign Encoder ║  ─────►  Vector
  (skeleton 543 pts)      ╚═══════════════╝         [0.2, -0.5, 0.8, ...]
                                                        768 มิติ
                                                            ↕ cosine similarity
  ข้อความ subtitle         ╔═══════════════╗         Text
  ────────────────────►   ║  Text Encoder ║  ─────►  Vector
  "สวัสดี เด็ก เรียน"     ╚═══════════════╝         [0.1, -0.6, 0.9, ...]
                                                        768 มิติ

  ถ้าข้อความ "ตรงกับ" ท่ามือ → vectors จะ "ใกล้กัน" ใน 768-dim space
```

**สิ่งที่สร้างได้:**

```
Sign embeddings (จาก pose + segmentation):
  segmentation_embedding/sign_clip/04.npy         → shape (2780, 768)  [BSL model]
  segmentation_embedding/sign_clip_multi/04.npy   → shape (2780, 768)  [Multilingual]
  segmentation_embedding/sign_clip_asl/04.npy     → shape (2780, 768)  [ASL model]

Text embeddings (จาก subtitle text):
  subtitle_embedding/sign_clip_multi/04.npy       → shape (172, 768)   [CC text]
  subtitle_embedding/sign_clip_multi_gloss/04.npy → shape (172, 768)   [Gloss text]
```

**ทำไม Gloss text ดีกว่า CC text:**
```
CC text:   "นักเรียน ต้องเรียนรู้ การเปรียบเทียบ"   ← ภาษาไทยพูด (สังเคราะห์จากเสียงครู)
Gloss text: "นักเรียน เรียน เปรียบเทียบ"             ← ภาษามือไทย (คำต่อคำที่มือแสดง)
                                                         ↑ ใกล้เคียงกับท่ามือที่แสดงจริงกว่า
```

---

## Slide 2e — Step 5: DP Alignment (หัวใจของระบบ)

**เป้าหมาย:** หาว่า subtitle cue แต่ละตัว ควร "map" ไปยัง sign segment ไหน

```
ภาพในใจ — ปัญหาที่ต้องแก้:

  Subtitle timeline (CC, 172 cues — sync กับเสียงพูด):
  │──[s1]──[s2]────[s3]──[s4]──[s5]──...──[s172]──│

  Sign segment timeline (2780 segments — จากท่ามือจริง):
  │─[g1]─[g2][g3]──[g4]─[g5][g6][g7]──...──[g2780]─│

  งาน: จับคู่ s1→g?, s2→g?, ... s172→g? แบบ monotonic (ต้องเรียงลำดับ)
```

**3 ขั้นตอนหลักของ DP:**

```
① Pre-shift subtitles (bias correction)
   ─────────────────────────────────────────────────────
   cue.start += 1.3s    (C_MULTI)
   cue.end   += 1.0s
   เหตุผล: ผู้แปลแสดงท่ามือ "หลัง" เสียงครูพูดเสมอ ~1-2 วินาที
   ต้องเลื่อน cue ไปข้างหน้าก่อนถึงจะ match ได้

② Cost matrix computation
   ─────────────────────────────────────────────────────
   สำหรับทุกคู่ (cue_i, sign_group[k:j]):

   cost = |start_diff| + |end_diff|
        + 2 × |duration_diff|         ← W_dur = 2
        + 8 × total_gap               ← W_gap = 8
        + 6 × (−similarity)           ← W_sim = 6

   ยิ่ง timing ตรง + ยิ่ง embedding คล้าย = cost ต่ำ = คู่ที่ดี

③ Dynamic Programming (Numba @njit)
   ─────────────────────────────────────────────────────
   dp[i][j] = min cost ที่ assign cue 1..i ให้ sign segments 1..j

   dp[i][j] = min over k { dp[i-1][k] + cost(cue_i, group[k:j]) }

   Constraint: monotonic — cue ที่ i+1 ต้อง assign segment หลัง cue ที่ i
               (เพราะ subtitle ต้องเรียงตามเวลา)
```

```
ภาพในใจ — DP matrix:

              sign segments →  g1  g2  g3  g4  g5  g6 ... g2780
  subtitle  s1               [ ∞  1.2  2.5  3.1  ∞   ∞  ...  ∞  ]
  cues    ↓ s2               [ ∞   ∞   0.8  1.4  2.2  ∞  ...  ∞  ]
            s3               [ ∞   ∞    ∞   0.3  0.9  1.8 ...  ∞  ]
            ...

  แต่ละ cell = min cost ที่ assign cue 1..i ให้ sign 1..j
  backtrack จาก cell ล่างขวา → หาเส้นทาง assign ที่ดีที่สุด
```

**ผลลัพธ์:** `aligned_output_*/04.vtt` — timestamp ของแต่ละ cue ถูกแทนที่ด้วย timestamp ของ sign segment ที่ match

---

## Slide 2f — Step 6: Overlap Fix

**ปัญหา:** หลัง DP alignment cues หลายตัวมี timestamp ซ้อนทับกัน

```
ก่อน fix:                              หลัง fix:
─────────────────────────────────────  ─────────────────────────────────────
cue 1: ████████████████                cue 1: █████████████
cue 2:       ██████████████████        cue 2:              ██████████████████
              ↑ overlap ตรงนี้
```

**สาเหตุ:** CC มี 172 cues แต่ sign segments ที่ "เหมาะ" มีแค่ ~119 slots — บาง cues จึงถูก assign ให้ segment เดียวกันหรือ segment ที่ติดกันมาก DP ไม่ได้ห้าม overlap โดยตรง

**วิธีแก้ (clamp end time):**
```python
for i in range(len(cues) - 1):
    if cues[i].end > cues[i+1].start:
        cues[i].end = cues[i+1].start   # ตัด end ตรงที่ cue ถัดไปเริ่ม
```

**ผล:** overlap ~88% → 0% โดย start time ไม่ถูกแตะ → timing metrics ไม่เปลี่ยน

---

## Slide 2g — Step 7: Evaluation

**เป้าหมาย:** วัดว่า alignment ดีแค่ไหนเทียบกับ ground truth ที่นักวิจัย annotate ด้วยมือ

```
Ground truth (CC_Aligned, 119 entries):
  cue: "สวัสดี"   GT timestamp: 35.00s → 36.50s

Prediction (C_MULTI output):
  cue: "สวัสดี"   Pred timestamp: 35.20s → 36.80s

  start offset = 35.20 − 35.00 = +0.20s   ← ดีมาก (< 1s)
  end offset   = 36.80 − 36.50 = +0.30s   ← ดีมาก
```

**Metrics ที่วัด:**

```
mean offset  = ค่าเฉลี่ยของ (pred_start − gt_start) ทุก cue ที่ match
               → บอกว่า alignment shift ไปทิศไหนโดยรวม

% within ±1s = จำนวน cues ที่ |pred_start − gt_start| < 1s
               → metric หลักที่บอกว่า "ถูกต้องพอใช้งานได้"

overlap %    = สัดส่วน cues ที่ timestamp ซ้อนทับกับ cue ถัดไป
               → ต้องการให้เป็น 0% หลัง fix
```

**การ match cue:** ใช้ text lookup — หา cue ใน prediction ที่มีข้อความตรงกับ GT
```
GT text "สวัสดี" → หา cue ใน VTT output ที่มี text = "สวัสดี" → วัด offset
ถ้าหาไม่เจอ (49 entries ถูก annotator แก้ข้อความ) → ข้ามไป → วัดได้แค่ 69/119 GT entries
```

---

## Slide 3 — DP Alignment คืออะไร (Core Algorithm)

DP Alignment เป็นหัวใจของระบบ — จับคู่ subtitle แต่ละ cue กับ sign segment ที่ "เหมาะสมที่สุด"

**สูตร cost function:**

```
cost(cue_i, group[k:j]) =
    | cue_start − group_start |          ← start offset
  + | cue_end   − group_end   |          ← end offset
  + W_dur × | cue_dur − group_dur |      ← duration mismatch
  + W_gap × total_gap_in_group           ← gap ระหว่าง segments
  + W_sim × (−similarity_total)          ← ยิ่งคล้ายกัน cost ยิ่งต่ำ
```

**พารามิเตอร์ที่ใช้ใน C_MULTI (best run):**

| พารามิเตอร์ | ค่า | ผลกระทบ |
|---|---|---|
| `similarity_weight` | 6 | น้ำหนักของ embedding similarity |
| `dp_duration_penalty_weight` | 2 | บังคับ duration ให้ใกล้เคียงกัน |
| `dp_gap_penalty_weight` | 8 | ลงโทษ group ที่มีช่องว่างสูง |
| `dp_max_gap` | 6 | จำกัด segments ต่อ group สูงสุด |
| `dp_window_size` | 40 | จำนวน candidate segments ต่อ cue |
| `pr_subs_delta_bias_start` | 1.3 | เลื่อน start time ไปข้างหน้า |
| `pr_subs_delta_bias_end` | 1.0 | เลื่อน end time ไปข้างหน้า |

> **Pre-shift:** cue timestamps ถูกเลื่อนไปข้างหน้า (~1-2 วินาที) ก่อน DP เพราะผู้แปลภาษามือแสดงท่า**หลัง**เสียงพูดเสมอ

---

## Slide 4 — Task 1: อะไรที่อัปเดตตั้งแต่ครั้งก่อน (Apr 20 → Apr 26)

### สิ่งที่มีใน Apr 20

- รัน 7 experiments ครบแล้ว → ได้ผล alignment VTT ทั้ง 7 ชุด
- ทำ overlap fix **เฉพาะบางส่วน** (C_MULTI และ D_ASL_gloss)
- comparison EAF มี **7 tiers** (pre-overlap เท่านั้น)
- ยังไม่มี evaluation CSV (ผลอยู่ใน stdout เท่านั้น)

### สิ่งที่อัปเดตใน Apr 26

```
Apr 20 → Apr 26
─────────────────────────────────────────────────────────────────
Overlap fix    : เฉพาะ 2 experiments    →  ครบทั้ง 7 experiments
                                             (14 VTT files, before/after)

Evaluation     : stdout เท่านั้น         →  evaluation_task1_results.csv
                                             14 rows (7 experiments × 2 variants)

Comparison EAF :  7 tiers (pre-overlap)  →  15 tiers
                                             (7 pre + 7 post + GLOSS_LABEL_PRED)

Visualization  :  ไม่มี                  →  figures/timeline_first_2min.png
                                             (4 lanes: CC / CC_Aligned / C_MULTI / GLOSS_PRED)

Evaluation     :  ไม่มี scope note       →  เพิ่ม note ชัดเจนว่า metrics วัดบน
scope                                       69/119 GT entries ที่ match ได้เท่านั้น
```

---

## Slide 5 — Task 1: ผลทั้ง 7 Experiments

> วัดเทียบกับ **CC_Aligned ground truth** (119 entries)
> Evaluator จับคู่ด้วย text lookup → match ได้ 69/119 entries
> *(49 entries ที่ annotator แก้ข้อความจาก CC ต้นฉบับ ยัง match ด้วย text ตรงๆ ไม่ได้)*

| Experiment | Text Input | Model | Mean offset | ±1s | ±2s | ±3s | Overlap |
|---|---|---|---|---|---|---|---|
| B2 | CC | BSL | +1.02s | 74% | 96% | 97% | → **0%** |
| B_MULTI | CC | Multi | +0.91s | 78% | 97% | 99% | → **0%** |
| **C_MULTI** ⭐ | **Gloss** | **Multi** | **+0.49s** | **80%** | **96%** | **99%** | → **0%** |
| C_MULTI_word | Gloss (word) | Multi | +0.51s | 77% | 96% | 99% | → **0%** |
| D_ASL | CC | ASL | +1.25s | 59% | 81% | 96% | → **0%** |
| D_ASL_gloss | Gloss | ASL | +0.77s | 64% | 91% | 97% | → **0%** |
| D_ASL_word | Gloss (word) | ASL | +0.78s | 67% | 93% | 96% | → **0%** |

*(ทุก experiment หลัง overlap fix → overlap = 0.0%)*

**สิ่งที่สังเกตได้:**
- `multilingual` > `bsl` > `asl` สำหรับ TSL
- Gloss text > CC text เพราะ gloss ใกล้เคียงกับท่ามือที่แสดง
- C_MULTI มี mean offset ต่ำสุดและ ±1s สูงสุด → champion run ปัจจุบัน
- stdev ~5.5s **สม่ำเสมอในทุก experiment** (ไม่ขึ้นกับโมเดลหรือ text source) → น่าจะเป็นปัญหาโครงสร้างข้อมูล ไม่ใช่ปัญหาของโมเดล — ยังต้องวิเคราะห์ outlier cues

---

## Slide 6 — Task 1: Post-processing (Overlap Fix)

**ปัญหา:** alignment ดิบมี overlap ~88% — subtitle cues หลายตัวถูก assign timestamp ที่ซ้อนทับกัน

**สาเหตุ:** CC มี 172 cues แต่ sign segments รองรับได้แค่ ~119 slots จริง DP จึงบางครั้งให้ timestamp เหลื่อมกัน

**วิธีแก้ — `fix_overlap_vtt.py`:**
```python
for i in range(len(cues) - 1):
    if cues[i].end > cues[i+1].start:
        cues[i].end = cues[i+1].start   # clamp end time เท่านั้น
```

**ผล:**
- overlap **~88% → 0.0%** ทุก experiment
- start time ไม่ถูกแตะ → metric ที่วัดจาก start (mean offset, ±1s/2s/3s) **ไม่เปลี่ยน**
- เหมาะสำหรับ subtitle ที่ต้องการแสดงต่อเนื่องโดยไม่ overlap

---

## Slide 7 — Task 1: สิ่งที่ยังต้องทำต่อ

```
สถานะปัจจุบัน: รัน 7 experiments บน 1 วิดีโอ, เลือก C_MULTI เป็น best run
```

| ลำดับ | งานที่ต้องทำ | เหตุผล |
|---|---|---|
| 1 | **วิเคราะห์ outlier cues** (offset > ±5s) | stdev ~5.5s สูงมาก — ต้องดูว่า cues ไหนที่ align ผิด และทำไม |
| 2 | **ขยาย evaluator ให้ครอบคลุม 49 GT entries ที่หายไป** | ปัจจุบันวัดแค่ 69/119 entries — ต้องทดสอบ fuzzy match หรือ index-based |
| 3 | **ทดสอบกับวิดีโอเพิ่ม (5-10 คลิป)** | ผลปัจจุบันมาจาก 1 วิดีโอเท่านั้น ยังไม่สามารถ generalize ได้ |
| 4 | **Parameter sweep** | ลองค่า gap_penalty, similarity_weight ต่างๆ อาจดีกว่า C_MULTI ปัจจุบัน |
| 5 | **Crop ROI ผู้แปล** | ลด noise ใน pose estimation จากพื้นหลังและผู้บรรยาย |
| 6 | **แก้ `gt_by_text` ใน `evaluate_all.py`** | บรรทัด 97–101: `if key not in gt_by_text: gt_by_text[key] = ...` → เก็บแค่ entry แรกของ text ที่ซ้ำ — drop entry ที่สองทิ้งโดยไม่แจ้ง กระทบ 1 cue ในชุดข้อมูลนี้ |

---

## Slide 8 — Evaluation Scope (สิ่งสำคัญที่ต้องระบุเสมอ)

> ⚠️ ทุกครั้งที่อ้างตัวเลข Task 1 ต้องระบุ scope ให้ชัด

**ตัวเลขที่รายงาน เช่น "80% within ±1s" หมายความว่า:**
- 80% ของ **69 cues ที่ evaluator จับคู่ได้** (= 55/69 cues)
- ไม่ใช่ 80% ของ 172 cues ทั้งหมด
- ไม่ใช่ 80% ของ 119 GT entries ทั้งหมด

**เหตุผลที่ match ได้แค่ 69/119:**
- CC_Aligned (GT) สร้างโดยนักวิจัยที่มักแก้ไขข้อความระหว่าง annotate
- 49 entries มีข้อความต่างจาก CC ต้นฉบับ → text lookup ไม่เจอ
- นี่เป็น **limitation ของ evaluator** ไม่ใช่ alignment เอง

| ตัวเลข | ความหมายที่ถูกต้อง |
|---|---|
| 69/172 matched | evaluator หาคู่ได้ 69 cues (40% ของ output ทั้งหมด) |
| 80% within ±1s | 80% ของ 69 cues ที่ match ได้อยู่ใน ±1s |
| 99% within ±3s | 99% ของ 69 cues ที่ match ได้อยู่ใน ±3s |

---

## Slide 8b — ⚠️ สิ่งที่ต้องแก้ในโค้ด (Known Issues)

> รายการนี้ไม่ใช่ bug ที่ทำให้ระบบพัง แต่เป็น **ความไม่ถูกต้องที่ต้องแก้ก่อน report อย่างเป็นทางการ**

### Issue 1 — Task 1: `gt_by_text` ใน `evaluate_all.py` เก็บแค่ entry แรก

```python
# บรรทัด 97–101 (ปัจจุบัน — ต้องแก้)
if key not in gt_by_text:
    gt_by_text[key] = (t1, t2)   # ← entry ที่สองของ text เดิมถูก drop ทิ้งโดยไม่แจ้ง
```

- กระทบ: 1 cue ใน CC_Aligned มี text ซ้ำกัน → entry ที่สองหายไปจาก evaluation โดยอัตโนมัติ
- ผลกระทบต่อตัวเลขปัจจุบัน: เล็กน้อย (1/69) แต่ต้องแก้เพื่อความถูกต้อง
- วิธีแก้: เปลี่ยน `gt_by_text` จาก `dict[str → single entry]` เป็น `dict[str → list[entry]]` หรือใช้ index-based matching

### Issue 2 — Task 2: Mean IoU = 0.49 เป็น upper-bound estimate

- `evaluate_gloss_labeling.py` ใช้ **greedy matching** (non-exclusive)
- แต่ละ prediction จับคู่กับ GT entry ที่มี IoU สูงสุด **โดยไม่ตรวจว่า GT นั้นถูก claim ไปแล้วหรือยัง**
- ผล: prediction หลายตัวอาจ claim GT entry เดียวกัน → IoU **สูงกว่าความเป็นจริง**
- วิธีแก้ที่ถูกต้อง: ใช้ **Hungarian matching** (one-to-one assignment)
- ตัวเลขที่ควร report: *"Mean IoU = 0.49 (upper-bound; non-exclusive greedy matching)"*

### Issue 3 — Task 1: stdev ~5.5s ยังไม่ได้วิเคราะห์

- stdev ~5.5s เท่ากันในทุก 7 experiments ไม่ว่าจะใช้โมเดลหรือ text source ใด
- แสดงว่าปัญหาน่าจะมาจาก **outlier cues บางตัว** ที่ align ผิดมาก ไม่ใช่จากโมเดล
- ยังไม่ได้ระบุว่า cues ไหนเป็น outlier และทำไมถึง offset ≥ 5s
- จำเป็นต้องวิเคราะห์ก่อน claim ว่า C_MULTI "ดี" โดยไม่มี caveat เรื่อง outlier

---

## Slide 9 — ผลลัพธ์ที่จับต้องได้ (Artifacts)

**ไฟล์ที่สร้างได้จาก Task 1 pipeline จนถึงตอนนี้:**

```
example_alignment/
├── aligned_output_multi_gloss/
│   ├── 04.vtt                 ← C_MULTI alignment (pre-fix)
│   └── 04_no_overlap.vtt      ← C_MULTI alignment (post-fix) ← ใช้งานจริง
│
├── aligned_output_*/04_no_overlap.vtt   ← ทั้ง 7 experiments
│
├── evaluation_task1_results.csv         ← metrics ทุก experiment (14 rows)
│
├── การเปรียบเทียบฯ_comparison.eaf      ← ELAN file 15 tiers:
│     CC, CC_Aligned, Gloss, Gloss Labeling   (original 4)
│     SUBTITLE_*                              (7 pre-overlap)
│     SUBTITLE_*_no_overlap                   (7 post-overlap)
│     GLOSS_LABEL_PRED                        (Task 2)
│
└── figures/timeline_first_2min.png      ← visualization 4 lanes (0-120s)
```

---

## Slide 10 — สรุปและขั้นตอนถัดไป

**ความคืบหน้า Task 1 จนถึงตอนนี้:**
- Pipeline ครบทุกขั้นตอน — segment → embed → align → post-process → evaluate
- ทดสอบ 7 experiments เพื่อเปรียบเทียบโมเดลและ text input
- C_MULTI (Multilingual + Gloss text) ให้ผลดีที่สุดในชุดทดสอบปัจจุบัน
- Post-processing ลด overlap จาก ~88% เป็น 0% โดยไม่กระทบ timing metrics
- Evaluation ถูก archive เป็น CSV พร้อม scope note ที่ชัดเจน

**สิ่งที่กำลังทำต่อ:**
- วิเคราะห์ outlier cues เพื่อเข้าใจว่า stdev ~5.5s มาจากไหน
- ขยาย evaluation coverage จาก 69/119 → ครอบคลุม GT ทั้งหมด
- ทดสอบบนวิดีโอเพิ่มเพื่อตรวจสอบว่า C_MULTI generalize ได้จริง

**ข้อค้นพบสำคัญจากการทดสอบ:**
- Multilingual SignCLIP encoder generalize มา TSL ได้ดีกว่าที่คาด แม้ไม่มี TSL training data
- Gloss text (ภาษามือ) ให้ embedding ที่ match กับ sign segment ได้ดีกว่า CC text (ภาษาไทยพูด)
- DP alignment สามารถทำงาน cross-lingual ได้โดยไม่ต้อง fine-tune — แต่ยังต้องการ parameter tuning

---

*อ้างอิง: [arXiv:2512.08094](https://arxiv.org/abs/2512.08094) — Jiang et al., 2025 | ข้อมูลครบถ้วนใน [Progress_26042026.md](Progress_26042026.md)*
