# SEA for Thai Sign Language — Progress Update
### Next Presentation Brief
> **อัปเดตล่าสุด:** 5 พฤษภาคม 2569 | วิดีโอทดสอบ: "การเปรียบเทียบและเรียงลำดับ" (11 นาที)

---

## Slide 1 — Project Overview

**SEA = Segment → Embed → Align**

> นำระบบ SEA (arXiv:2512.08094, Oxford/ETH 2025) มาทดสอบกับภาษามือไทย (TSL) แบบ cross-lingual

**2 งานหลักที่กำลังทำ:**

| งาน | เป้าหมาย | สถานะ |
|---|---|---|
| **Task 1** | จัดเวลา CC subtitle ให้ตรงกับช่วงที่ผู้แปลแสดงท่ามือ | ทดสอบ 7 experiments บน 1 วิดีโอ — C_MULTI: mean −0.16s, 100% ±3s, F1@0.50=88.24% |
| **Task 2** | แยก gloss ระดับประโยคออกเป็น annotation รายท่ามือ | Prototype: 889 predictions, Mean IoU 0.4199, 93.4% temporal overlap |

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

**เป้าหมาย:** หาว่า subtitle cue แต่ละตัว ควร "map" ไปยัง sign segment ไหน — โดยรักษา **ลำดับ** ของ cue ไว้เสมอ

```
ภาพในใจ — ปัญหาที่ต้องแก้:

  Subtitle timeline (CC, 172 cues — sync กับเสียงพูด):
  │──[s1]──[s2]────[s3]──[s4]──[s5]──...──[s172]──│

  Sign segment timeline (2,780 segments — จากท่ามือจริง):
  │─[g1]─[g2][g3]──[g4]─[g5][g6][g7]──...──[g2780]─│

  งาน: จับคู่ s1→g?, s2→g?, ... s172→g? แบบ monotonic
  ข้อสังเกต: subtitle 1 ตัว อาจ map กับ sign หลายตัว ("group")
             เพราะ 1 ประโยคมักประกอบด้วยหลายท่ามือ
```

**3 เฟสหลักของ DP:**

```
┌──────────────────────────────────────────────────────────────────────┐
│  เฟส 1: Pre-shift Subtitles (bias correction)                        │
│  ────────────────────────────────────────────────────────────────    │
│                                                                      │
│  cue.start += 1.3s    cue.end += 1.0s    (ค่าของ C_MULTI)            │
│                                                                      │
│  เหตุผล: ผู้แปลแสดงท่า "หลัง" เสียงครูพูด ~1-2 วินาทีเสมอ           │
│                                                                      │
│  เสียงครู:  ──────● "นักเรียน"──────────────────                     │
│                  t=35.0s                                             │
│  มือผู้แปล: ─────────────────────●▓▓▓▓▓●──                          │
│                                 t≈36.3s  ← เริ่มท่ามือ              │
│                                                                      │
│  ถ้าไม่ pre-shift: DP จะหา sign segment ที่ t=35.0s (ผิด)           │
│  หลัง pre-shift:  DP จะหา sign segment ที่ t≈36.3s ✓               │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  เฟส 2: Dynamic Programming (Numba @njit)                            │
│  ────────────────────────────────────────────────────────────────    │
│                                                                      │
│  dp[i][j] = cost ต่ำสุดในการ assign cue 1..i                        │
│             โดยใช้ sign segments ถึง index j                         │
│                                                                      │
│  recurrence:                                                         │
│    dp[i][j] = min over k < j {                                      │
│                dp[i-1][k] + cost(cue_i, signs[k+1..j])              │
│              }                                                       │
│              → cue i ใช้ signs[k+1..j] (อาจเป็น 1 หรือหลาย signs)  │
│                                                                      │
│  Monotonic constraint:                                               │
│    ถ้า cue i ใช้ signs ถึง index j                                   │
│    cue i+1 ต้องใช้ signs ที่ index > j เท่านั้น                     │
│    (subtitle ต้องเรียงตามเวลา — ห้ามย้อนกลับ)                       │
│                                                                      │
│  Window optimization:                                                │
│    ไม่ต้อง loop k จาก 0..N ทั้งหมด — ใช้แค่ window_size=40         │
│    signs ที่ใกล้ cue.mid ที่สุด → ลด O(M×N²) → O(M×W²)              │
│                                                                      │
│  cost(cue_i, signs[k+1..j]):                                        │
│    |cue_start − sign_starts[k+1]|    ← start timing                 │
│  + |cue_end   − sign_ends[j]    |    ← end timing                   │
│  + W_dur=2 × |dur_cue − dur_group|  ← duration match                │
│  + W_gap=8 × Σ gaps in group        ← ลงโทษ group ที่มีช่องว่าง    │
│  + W_sim=6 × (−Σ similarity[i, k..j]) ← embedding (ยิ่งคล้ายยิ่งดี) │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  เฟส 3: Backtrack + Subgroup Refinement                              │
│  ────────────────────────────────────────────────────────────────    │
│                                                                      │
│  Backtrack: ย้อนกลับจาก dp[M][N] ผ่าน prev[][] matrix               │
│    → boundaries[i] = sign index ที่ cue i สิ้นสุด                   │
│    → cue i ใช้ signs[boundaries[i-1] : boundaries[i]]               │
│                                                                      │
│  Subgroup Refinement (ต่อ cue, independent):                         │
│    ภายใน range ที่ DP กำหนด ทดลองทุก contiguous sub-slice           │
│    → เลือก sub-slice ที่มี cost ต่ำสุด (max_gap=6 ระหว่าง signs)    │
│    → cue.start = subgroup[0].start                                  │
│       cue.end   = subgroup[-1].end                                  │
│                                                                      │
│  ⚠️  Subgroup refinement ทำแบบ independent ต่อ cue                   │
│      → cue i .end อาจ > cue i+1 .start → เกิด overlap               │
│      (root cause อธิบายใน Slide 3a และ Slide 6)                     │
└──────────────────────────────────────────────────────────────────────┘
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

**Metrics ที่วัด (evaluate_all.py — 10 metrics + overlap, ครบทุก 119 cues ต่อ experiment):**

```
── Start-time offset (primary metrics) ────────────────────────────────────────
mean_off       = mean (pred_start − gt_start)  ทุก 119 cues
                 → positive = align ช้ากว่า GT, negative = เร็วกว่า GT, 0 = สมบูรณ์
                   ตัวอย่าง C_MULTI: −0.16s (align เร็วกว่า GT เล็กน้อย)
median_off     = median offset  (robust ต่อ outlier)
stdev_off      = standard deviation ของ offset  (~0.85–1.2s ต่อ experiment)
                 → stdev สูงกว่า 1.5s = มี outlier cues ที่ align ผิดมาก
w1 = ±1s       = % cues ที่ |pred_start − gt_start| ≤ 1s  → primary usability metric
                   B2: 76%, C_MULTI: 74%, D_ASL: 61%
w2 = ±2s       = % cues ที่ |pred_start − gt_start| ≤ 2s
w3 = ±3s       = % cues ที่ |pred_start − gt_start| ≤ 3s  → C_MULTI = 100%

── End-time offset ─────────────────────────────────────────────────────────────
mean_end_off   = mean (pred_end − gt_end)  ทุก 119 cues
                 → ทุก experiment end time ยาวกว่า GT ประมาณ +0.84–1.46s
                   DP ไม่ optimize end time โดยตรง → end มักยาวกว่า GT เสมอ
mean_end_off_abs = mean |pred_end − gt_end|  → magnitude ไม่แยกทิศทาง

── SEA-style frame-level metrics (@25fps) ──────────────────────────────────────
frame_acc      = % frames ที่ label ถูก (background vs sign class)
                 → แปลง 119 cue timestamps → frame-level array ด้วย subs2frames()
                   เทียบ pred vs GT ทีละ frame
                   C_MULTI: 82.63%, C_MULTI_word: 82.74% (highest)
f1_10          = F1 score ที่ IoU threshold 0.10  (segment-level, ผ่อนปรน)
                 → ทุก experiment ≥ 97% แสดงว่า segment placement ถูกต้องสูงมาก
f1_25          = F1 score ที่ IoU threshold 0.25
f1_50          = F1 score ที่ IoU threshold 0.50  (เข้มงวด — overlap > 50% ถึงนับ)
                 → C_MULTI_word สูงสุด: 89.08%, D_ASL ต่ำสุด: 77.31%
                 → ported จาก SEA evaluate_sub_alignment.py

── Overlap ─────────────────────────────────────────────────────────────────────
overlap_pct    = สัดส่วน consecutive cue pairs ที่ end[i] > start[i+1]
                 → ก่อน fix_overlap_vtt.py: ~86–89% ทุก experiment
                 → หลัง fix: 0.0% ทุก experiment (clamp end time)
```

**การ match cue (ปัจจุบัน — index-based):**
```
pred[0] ↔ gt[0], pred[1] ↔ gt[1], ..., pred[118] ↔ gt[118]
→ match ครบ 119/119 ทุก experiment โดยไม่ต้องค้นหาข้อความ
```

> 📌 เดิม (ก่อน May 4): ใช้ text lookup → match ได้แค่ 69/119 GT entries เพราะ annotator แก้ข้อความ 49 entries

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

### ตัวอย่างเล็กๆ: 3 cues, 6 sign segments

```
Setup หลัง pre-shift (+1.3s / +1.0s):

  Cue 1 "นักเรียน"    pre-shifted: 36.3s → 36.8s
  Cue 2 "เรียน"       pre-shifted: 38.3s → 38.6s
  Cue 3 "เปรียบเทียบ" pre-shifted: 40.3s → 41.0s

  Sign segments:
    g1: 35.9–36.1s   g2: 36.2–36.6s   g3: 36.8–37.2s
    g4: 38.2–38.7s   g5: 38.9–39.3s   g6: 40.2–41.1s
```

```
DP matrix (dp[i][j] = min total cost):

      j=0  j=1  j=2  j=3   j=4   j=5   j=6
i=0  [0.0   ∞    ∞    ∞     ∞     ∞     ∞  ]  ← ยังไม่ assign
i=1  [ ∞   2.4  0.9  1.3    ∞     ∞     ∞  ]  ← assign cue 1
i=2  [ ∞    ∞    ∞   2.8   1.1   1.8    ∞  ]  ← assign cue 1+2
i=3  [ ∞    ∞    ∞    ∞    4.0   3.2   1.4 ]  ← assign cue 1+2+3

  Best path backtrack จาก dp[3][6] = 1.4:
    prev[3][6] = 4  →  cue 3 ใช้ signs[4:6] = {g5, g6}
    prev[2][4] = 3  →  cue 2 ใช้ signs[3:4] = {g4}
    prev[1][3] = 0  →  cue 1 ใช้ signs[0:3] = {g1, g2, g3}
```

```
ผลลัพธ์ alignment:

  ก่อน alignment (เสียงพูด):
  ──[cue1:35.0-35.8]────[cue2:37.0-37.6]────[cue3:39.0-40.0]──

  หลัง alignment (ท่ามือ):
  ──[cue1:35.9-37.2]──[cue2:38.2-38.7]──[cue3:38.9-41.1]──
        g1+g2+g3              g4               g5+g6

  ⚠️ cue2 end (38.7) > cue3 start (38.9) → ไม่ overlap ในตัวอย่างนี้
     แต่ในข้อมูลจริง (2,780 signs, 172 cues) เกิด overlap ~88%
     เพราะ subgroup refinement ทำแบบ independent (ดู Slide 3a)
```

---

## Slide 3a — Subgroup Refinement: ทำไม Overlap ถึงเกิด (Root Cause)

**ขั้นตอน Subgroup Refinement คืออะไร:**

หลัง DP backtrack แต่ละ cue ได้ index range ของ sign segments เช่น cue 3 ได้ signs[80:95]
ในกรณีนี้ DP คิดว่า cue 3 "ควร match" กับ sign cluster บางส่วนใน range นั้น แต่ไม่รู้ว่าส่วนไหนดีที่สุด

Subgroup Refinement จึง:
1. Loop ผ่าน signs[80:95] ทั้งหมด
2. ทดลองทุก contiguous sub-slice (แต่ละ "กลุ่มติดกัน" ที่ไม่มี gap > max_gap)
3. เลือก sub-slice ที่มี cost ต่ำสุด
4. set `cue.start = subslice[0].start`, `cue.end = subslice[-1].end`

**ทำไมถึงเกิด overlap:**

```
DP boundaries (monotonic by index):
  cue 2 → signs[40:70]    (index range)
  cue 3 → signs[70:95]    (index range)

Subgroup Refinement (independent per cue):
  cue 2: best subgroup = signs[55:62]  → end = signs[61].end  = 87.4s
  cue 3: best subgroup = signs[70:76]  → start = signs[70].start = 87.1s

  → cue 2 end (87.4s) > cue 3 start (87.1s) → OVERLAP 0.3s ⚠️
```

```
ภาพ timeline:

  cue 2:   ────────────────────[══════]──────────────────────
                               55  62  (best subgroup)
                                      end = 87.4s

  cue 3:   ─────────────────────────────[════════]───────────
                                        70  76
                              start = 87.1s

  overlap:                          ████
                               87.1–87.4s
```

**ทำไม subgroup refinement ต้องเป็น independent:**
- DP จัดสรร index range → แต่ range นั้นอาจมี signs หลายกลุ่มที่ต่างกัน
- การเลือก best sub-slice ต้องอาศัย embedding similarity ของ cue นั้นๆ
- ถ้าทำ joint optimization ระหว่าง cues จะต้องรัน DP อีกรอบ → ซับซ้อนขึ้นมาก
- **แก้ง่ายกว่าด้วย post-processing clamp** (ดู Slide 6)

**สรุป root cause chain:**
```
DP รับประกัน: index[cue i] < index[cue i+1]   ✓ (monotonic by index)
DP ไม่รับประกัน: end_time[cue i] < start_time[cue i+1]  ✗
                  ↑ เพราะ subgroup refinement เลือก timestamp อิสระ
→ แก้ด้วย fix_overlap_vtt.py: clamp end[i] = min(end[i], start[i+1])
```

---

## Slide 4 — Task 1: อะไรที่อัปเดตตั้งแต่ครั้งก่อน (Apr 20 → Apr 26 → May 5)

### สิ่งที่มีใน Apr 20

- รัน 7 experiments ครบแล้ว → ได้ผล alignment VTT ทั้ง 7 ชุด
- ทำ overlap fix **เฉพาะบางส่วน** (C_MULTI และ D_ASL_gloss)
- comparison EAF มี **7 tiers** (pre-overlap เท่านั้น)
- ยังไม่มี evaluation CSV (ผลอยู่ใน stdout เท่านั้น)

### สิ่งที่อัปเดตใน Apr 26

**1. Overlap fix — ครบทั้ง 7 experiments**
- เดิม (Apr 20): fix เฉพาะ C_MULTI และ D_ASL_gloss (2 ไฟล์)
- อัปเดต: fix ครบทุก experiment → `04_no_overlap.vtt` ทั้ง 7 ชุด (14 VTT files รวม before/after)
- ผล: overlap 87–89% → **0.0% ทุก experiment** โดยไม่กระทบ start-time metrics

**2. Evaluation output — บันทึกเป็น CSV**
- เดิม: ผล evaluation อยู่ใน stdout เท่านั้น ดูย้อนหลังไม่ได้
- อัปเดต: `evaluate_all_to_csv.py` เขียน `evaluation_task1_results.csv` (14 rows = 7 × 2 variants)
- CSV Apr 26 มี **9 columns**: `experiment, variant, match_count, mean_off, median_off, w1, w2, w3, overlap_pct`

**3. Comparison EAF — ขยายจาก 7 → 19 tiers**
- เดิม: 7 tiers (pre-overlap alignment เท่านั้น)
- อัปเดต: `add_vtt_tiers_to_eaf.py` เพิ่ม post-overlap tiers + GLOSS_LABEL_PRED
- รวม: 4 original (CC, CC_Aligned, Gloss, Gloss Labeling) + 7 pre + 7 post + 1 GLOSS_LABEL_PRED = **19 tiers**

**4. Task 2 Prototype — เพิ่มใหม่ทั้งหมดใน Apr 26**
- `align_gloss_labels.py` — per-sentence monotonic DP (T~7 tokens × K~30 signs)
- `evaluate_gloss_labeling.py` — greedy IoU matching vs 852 GT entries
- ผล Task 2 (Apr 26, ใช้ Gloss tier เก่า): **Mean IoU = 0.49**, 48.4% ≥ 0.5, 65.1% text match, 97.5% temporal overlap
- ⚠️ ตัวเลข Apr 26 ใช้ Gloss tier เดิม (timestamp จาก manual annotation) — ต่างจาก May 4-5 ที่ใช้ Gloss_Input

**5. Evaluation scope note — ระบุ bias ชัดเจน**
- เพิ่ม note ว่า metrics วัดบน **69/119 GT entries** เท่านั้น (text lookup ที่ match ได้)
- 50 entries ที่ annotator แก้ข้อความ + duplicate text keys → ถูก exclude → ตัวเลข ±1s ใน Apr 26 เป็น biased

**Task 1 ผล Apr 26 (text lookup, 69/119 matched, post-overlap):**

| Experiment | Mean offset | ±1s | ±2s | ±3s | Overlap |
|---|---|---|---|---|---|
| B2 | +1.02s | 74% | 96% | 97% | 0.0% |
| B_MULTI | +0.91s | 78% | 97% | 99% | 0.0% |
| **C_MULTI** ⭐ | **+0.49s** | **80%** | 96% | 99% | 0.0% |
| C_MULTI_word | +0.51s | 77% | 96% | 99% | 0.0% |
| D_ASL | +1.25s | 59% | 81% | 96% | 0.0% |
| D_ASL_gloss | +0.77s | 64% | 91% | 97% | 0.0% |
| D_ASL_word | +0.78s | 67% | 93% | 96% | 0.0% |

> stdev ≈ 5.5s ทุก experiment (สูงผิดปกติ — เกิดจาก biased selection + duplicate key collision ใน gt_by_text dict)

---

### สิ่งที่อัปเดตใน May 4–5 — เปลี่ยนแปลงเชิงลึก

#### 1. Evaluation Method: Text Lookup → Index-based (ใหญ่ที่สุด)

**ปัญหาของ text lookup (Apr 26):**
```python
# เดิม: สร้าง dict จาก GT text
gt_by_text = {}
for ann in gt_cues:
    key = normalize(ann.text)
    if key not in gt_by_text:       # ← DROP entry ที่ text ซ้ำ
        gt_by_text[key] = (t1, t2)

for pred_cue in pred_cues:          # ← ต้อง text match ตรงทุก char
    key = normalize(pred_cue.text)
    if key in gt_by_text: matched.append(...)
```

สาเหตุที่ match ได้แค่ 69/119:
- **49 entries** ใน CC_Aligned: annotator แก้ข้อความหลัง annotation → text ไม่ตรง pred VTT
- **หลาย entries** มี text ซ้ำกัน (เช่น คำว่า "ดังนั้น" ปรากฏหลายครั้ง) → entry ที่สองและสามถูก drop
- ผล: 50 entries ที่ offset สูงบางส่วนถูก exclude → stdev เสีย (biased ~5.5s)

**วิธีแก้ index-based (May 4):**
```python
# ใหม่: match ตาม index โดยตรง — gt_by_text dict ถูกลบออกทั้งหมด
def match_cues(pred_cues, gt_cues):
    # pred VTT output มี 119 cues ↔ GT มี 119 entries → index ตรงกัน
    return [(p.start, g.start, p.end, g.end)
            for p, g in zip(pred_cues, gt_cues)]
```

ทำได้เพราะ tier เปลี่ยนจาก CC (172 cues) → CC_Input (119 cues) → DP output 119 cues ↔ GT 119 entries พอดี

ผล: match ครบ **119/119 cues ทุก experiment** (100%)

---

#### 2. Tier Input Changes: CC (172) → CC_Input (119), Gloss → Gloss_Input

**Apr 26 tier:**
- CC experiments (B2, B_MULTI, D_ASL): ใช้ `subtitles/04.vtt` = CC tier = **172 cues** (รวม cues ที่ไม่มีท่ามือ)
- Gloss experiments (C_MULTI, C_MULTI_word, D_ASL_gloss, D_ASL_word): ใช้ `subtitles_gloss_cc_time/04.vtt` = **172 timestamps** จาก CC, 170 ใช้ Gloss text
- DP output → **172 cues** → pred[i] ≠ gt[i] เพราะ GT มี 119 entries → index-based ทำไม่ได้

**May 4-5 tier:**
```
CC tier (172) ──► ตัด 53 cues ที่ไม่มีท่ามือออก ──► CC_Input tier (119 cues)
                  (maximum-overlap matching กับ CC_Aligned GT)

Gloss tier (119, manual annotation timestamp)
    + CC_Input tier (119, เสียงพูด timestamp)
    ──► make_gloss_input_tier.py (maximum-overlap matching) ──► Gloss_Input tier (119)
        timestamp: จาก CC_Input
        text: จาก Gloss tier
    → DP input 119 cues ↔ GT 119 entries → pred[i] ↔ gt[i] ✓
```

Script ใหม่ `make_gloss_input_tier.py`:
```python
for cc_cue in cc_input_cues:       # 119 cues
    best_gloss = max(gloss_anns,
                     key=lambda g: overlap(cc_cue, g))
    result.text  = best_gloss.text if overlap > 0 else cc_cue.text
    result.start = cc_cue.start    # timestamp จาก CC_Input
    result.end   = cc_cue.end
# Output: Gloss_Input tier, 119 entries, 0 fallback
```

---

#### 3. ผลกระทบต่อตัวเลขทุก Experiment — เปรียบเทียบ Apr 26 vs May 4-5

| Experiment | Apr 26 mean (69) | May 4-5 mean (119) | Apr 26 ±1s | May 4-5 ±1s | Apr 26 ±3s | May 4-5 ±3s | stdev (May) |
|---|---|---|---|---|---|---|---|
| B2 | +1.02s | **+0.26s** | 74% | 76% | 97% | 99% | 0.85s |
| B_MULTI | +0.91s | **+0.25s** | 78% | 71% | 99% | 99% | 0.96s |
| **C_MULTI** ⭐ | +0.49s | **−0.16s** | 80% | 74% | 99% | **100%** | 0.96s |
| C_MULTI_word | +0.51s | **−0.23s** | 77% | 74% | 99% | **100%** | 0.96s |
| D_ASL | +1.25s | **+0.38s** | 59% | 61% | 96% | 98% | 1.20s |
| D_ASL_gloss | +0.77s | **−0.12s** | 64% | 71% | 97% | 99% | 1.07s |
| D_ASL_word | +0.78s | **−0.13s** | 67% | 69% | 96% | 99% | 1.08s |

> Mean offset ลดลงทุก experiment เพราะ 50 cues ที่เพิ่มขึ้น (index-based) มี negative offset
> C_MULTI/Gloss experiments align ก่อน GT เล็กน้อย (bias_start=1.3s ดึงให้เร็วกว่าเสียงพูด)
> stdev ลดจาก ~5.5s → 0.85–1.2s เพราะไม่มี biased selection อีกต่อไป

---

#### 4. New Metrics (May 4-5) — ported จาก SEA evaluate_sub_alignment.py

**Apr 26 CSV: 9 columns**
```
experiment | variant | match_count | mean_off | median_off | w1 | w2 | w3 | overlap_pct
```

**May 4-5 CSV: 21 columns**
```
experiment | variant | match_count
| mean_off | median_off | stdev_off | mean_off_abs | median_off_abs
| w1 | w2 | w3
| mean_end_off | median_end_off | mean_end_off_abs
| frame_acc | f1_10 | f1_25 | f1_50
| overlap_pct
```

ฟังก์ชันใหม่ใน `evaluate_all.py`:
```python
def subs2frames(cues, fps=25, total_frames=None):
    """VTT cues → frame-level label array (@25fps)"""
    # แต่ละ cue → frames[start_frame:end_frame] = sign_class_id

def _get_labels_start_end_time(frame_wise_labels, bg_class=0):
    """Extract non-background segment boundaries (port จาก SEA)"""

def _f_score(recognized, ground_truth, overlap_threshold, bg_class=0):
    """F1 at IoU threshold — tp = segment pairs ที่ IoU ≥ threshold"""
    # precision = tp / pred_count
    # recall    = tp / gt_count
    # F1 = 2*p*r / (p+r)
```

**End-time metrics (May 4-5, ทุก experiment):**

| Experiment | mean_end_off | \|end\| abs |
|---|---|---|
| B2 | +1.33s | 1.39s |
| B_MULTI | +1.31s | 1.41s |
| **C_MULTI** ⭐ | **+0.91s** | **1.16s** |
| C_MULTI_word | +0.84s | 1.10s |
| D_ASL | +1.46s | 1.58s |
| D_ASL_gloss | +0.98s | 1.22s |
| D_ASL_word | +0.95s | 1.20s |

> ทุก experiment end time ยาวกว่า GT ประมาณ +0.84–1.46s
> สาเหตุ: DP cost function ไม่มี term minimize end-time error โดยตรง — ต้องการ end-time optimization (ดู Slide 7)

---

#### 5. Task 2 — Gloss Tier (Apr 26) vs Gloss_Input Tier (May 4-5)

| Metric | Apr 26 (Gloss tier) | May 4-5 (Gloss_Input) | เหตุผลที่เปลี่ยน |
|---|---|---|---|
| Predictions | 852 | **889** | tokenization ต่างกัน |
| GT labels | 852 | 852 | เหมือนเดิม |
| Mean IoU | **0.49** | **0.4199** | timestamp offset สูงขึ้น (CC_Input ≠ manual Gloss timestamp) |
| % IoU ≥ 0.5 | 48.4% | 38.9% | ↓ เพราะ window offset จาก GT มากขึ้น |
| % IoU ≥ 0.3 | 77.0% | 66.0% | ↓ |
| Text match | 65.1% | 10.6% | notation ต่างกัน (เช่น "ผายมือ" vs "ผายมือ_1") |
| Temporal overlap | 97.5% | 93.4% | ↓ เล็กน้อย |

> ทำไมเปลี่ยน tier: Gloss_Input ใช้ timestamp เดียวกับ Task 1 pipeline (CC_Input)
> → สอดคล้องกันทั้งระบบ แม้ IoU จะต่ำกว่า เพราะ GT Gloss Labeling annotate ตาม Gloss tier เดิม

---

#### 6. Output Files สรุป — Apr 26 vs May 4-5

| ไฟล์ / Script | Apr 26 | May 4-5 |
|---|---|---|
| `evaluation_task1_results.csv` | 14 rows, 9 cols | 14 rows, **21 cols** |
| `aligned_output_*/04_no_overlap.vtt` | 7 files | 7 files (re-generated with CC_Input) |
| `gloss_labels_pred.csv` | 852 rows (Gloss tier) | **889 rows** (Gloss_Input tier) |
| CC_Input tier | ใช้ CC (172 cues) | **CC_Input (119 cues)** |
| Gloss_Input tier | ใช้ Gloss timestamp | **Gloss_Input (CC_Input timestamp + Gloss text)** |
| `make_gloss_input_tier.py` | ไม่มี | **มี (script ใหม่)** |
| `evaluate_all.py` ฟังก์ชัน | match_cues (text), 9 metrics | match_cues (index), **21 metrics** |
| `Progress_04052026.md` | ไม่มี | **มี** |

---

## Slide 5 — Task 1: ผลทั้ง 7 Experiments

> วัดเทียบกับ **CC_Aligned ground truth** (119 entries)
> 🆕 Evaluator จับคู่ด้วย **index-based** → match ครบ **119/119 cues** ทุก experiment
> *(อัปเดต 4 พ.ค. 2569: เปลี่ยนจาก text lookup 69/119 → index-based 119/119)*

**Start-offset metrics (primary):**

| Experiment | Text Input | Model | Mean offset | ±1s | ±2s | ±3s | Overlap |
|---|---|---|---|---|---|---|---|
| B2 | CC | BSL | +0.26s | 76% | 96% | 99% | → **0%** |
| B_MULTI | CC | Multi | +0.25s | 71% | 93% | 99% | → **0%** |
| **C_MULTI** ⭐ | **Gloss** | **Multi** | **−0.16s** | **74%** | 95% | **100%** | → **0%** |
| C_MULTI_word | Gloss (word) | Multi | −0.23s | 74% | 94% | 100% | → **0%** |
| D_ASL | CC | ASL | +0.38s | 61% | 87% | 98% | → **0%** |
| D_ASL_gloss | Gloss | ASL | −0.12s | 71% | 91% | 99% | → **0%** |
| D_ASL_word | Gloss (word) | ASL | −0.13s | 69% | 90% | 99% | → **0%** |

**SEA-style frame-level metrics:**

| Experiment | FrameAcc | F1@.10 | F1@.25 | F1@.50 | \|end\| abs | End mean |
|---|---|---|---|---|---|---|
| B2 | 82.81% | 100.00% | 100.00% | 88.24% | 1.39s | +1.33s |
| B_MULTI | 81.54% | 99.16% | 99.16% | 84.87% | 1.41s | +1.31s |
| **C_MULTI** ⭐ | 82.63% | 99.16% | 99.16% | 88.24% | **1.16s** | **+0.91s** |
| C_MULTI_word | **82.74%** | 99.16% | 99.16% | **89.08%** | **1.10s** | **+0.84s** |
| D_ASL | 77.31% | 97.48% | 92.44% | 77.31% | 1.58s | +1.46s |
| D_ASL_gloss | 81.43% | 99.16% | 98.32% | 81.51% | 1.22s | +0.98s |
| D_ASL_word | 81.08% | 99.16% | 98.32% | 84.87% | 1.20s | +0.95s |

*(ทุก experiment หลัง overlap fix → overlap = 0.0%)*

**สิ่งที่สังเกตได้:**
- `multilingual` > `bsl` > `asl` สำหรับ TSL ทั้ง start-offset และ F1
- Gloss text > CC text เพราะ gloss ใกล้เคียงกับท่ามือที่แสดง
- **C_MULTI** มี mean start offset ต่ำ (−0.16s ใกล้ศูนย์), ±3s = 100%, F1@0.50 = 88.24% → champion run
- End time ยาวกว่า GT ประมาณ 0.9–1.5s ทุก experiment (เพราะ DP ไม่ optimize end time โดยตรง)
- F1@0.10 ≥ 97% ทุก experiment → segment-level placement ถูกต้องสูงมาก

---

## Slide 6 — Task 1: Post-processing (Overlap Fix)

**ปัญหา:** หลัง DP + subgroup refinement, cues หลายตัวมี timestamp ซ้อนทับกัน

**Root cause (จาก Slide 3a):**
- DP รับประกันแค่ **monotonic index** (sign range ของ cue i+1 มาหลัง cue i เสมอ)
- Subgroup refinement เลือก best sub-slice ใน range ของตัวเอง **โดยอิสระ**
- sub-slice ของ cue i อาจเลือก sign ที่ end time ล้ำเข้า territory ของ cue i+1

```
ก่อน fix (ตัวอย่าง timestamps):
  cue 5: ─────[████████████████████]────────────────
                          end = 87.4s
  cue 6: ─────────────[████████████████████]────────
                start = 87.1s
                       ↑ overlap 0.3s ⚠️

หลัง fix (clamp end):
  cue 5: ─────[█████████████]──────────────────────
                         end = 87.1s (clamped)
  cue 6: ─────────────[████████████████████]────────
                start = 87.1s
                       ↑ ชิดพอดี ไม่ overlap ✓
```

**ทำไม overlap สูงถึง ~88% ในข้อมูลจริง:**
- Sign segments หนาแน่นมาก (2,780 segments / 660 วินาที ≈ 4.2 segs/s)
- DP window = 40 signs → candidate range กว้าง → subgroup มีอิสระเลือกสูง
- CC มี 172 cues แต่ ground truth slot มีแค่ ~119 → บาง cues แชร์ sign cluster → เหลื่อม

**วิธีแก้ — `fix_overlap_vtt.py` (clamp end time):**
```python
for i in range(len(cues) - 1):
    if cues[i].end > cues[i+1].start:
        cues[i].end = cues[i+1].start   # clamp end ณ จุดที่ cue ถัดไปเริ่ม
```

**ผลลัพธ์ quantitative (ทุก 7 experiments):**

| Experiment | Overlap ก่อน fix | Overlap หลัง fix | Overlapping pairs |
|---|---|---|---|
| B2 | ~86% | **0.0%** | 102/118 pairs fixed |
| B_MULTI | ~86% | **0.0%** | 102/118 |
| C_MULTI | ~88% | **0.0%** | 104/118 |
| C_MULTI_word | ~88% | **0.0%** | 104/118 |
| D_ASL | ~86% | **0.0%** | 102/118 |
| D_ASL_gloss | ~88% | **0.0%** | 104/118 |
| D_ASL_word | ~88% | **0.0%** | 104/118 |

> ✅ **start time ไม่ถูกแตะ** → mean offset, ±1s/±2s/±3s **ไม่เปลี่ยน** หลัง fix

---

## Slide 7 — Task 1: สิ่งที่ยังต้องทำต่อ

```
สถานะปัจจุบัน: รัน 7 experiments บน 1 วิดีโอ, เลือก C_MULTI เป็น best run
อัปเดต 4 พ.ค. 2569: เพิ่ม index-based eval (119/119) + SEA-style frame metrics
```

| ลำดับ | งานที่ต้องทำ | เหตุผล | สถานะ |
|---|---|---|---|
| 1 | **วิเคราะห์ outlier cues** (offset > ±3s) | stdev ~0.96s — ต้องดูว่า cues ไหนที่ align ผิด และทำไม | ยังไม่ได้ทำ |
| 2 | ~~ขยาย evaluator ให้ครอบคลุม 49 GT entries ที่หายไป~~ | อัปเดต (May 4): เปลี่ยน text lookup → index-based — match 119/119 cues ทุก experiment, stdev ลดจาก ~5.5s → 0.85–1.2s | อัปเดตแล้ว |
| 3 | ~~แก้ `gt_by_text` drop duplicate entries~~ | อัปเดต (May 4): `gt_by_text` dict ถูกลบออกทั้งหมด เปลี่ยนเป็น index-based: `pred[i] ↔ gt[i]` โดยตรง | อัปเดตแล้ว |
| 4 | **ทดสอบกับวิดีโอเพิ่ม (5-10 คลิป)** | ผลปัจจุบันมาจาก 1 วิดีโอเท่านั้น ยังไม่สามารถ generalize ได้ | ยังไม่ได้ทำ |
| 5 | **Parameter sweep** | ลองค่า gap_penalty, similarity_weight ต่างๆ อาจดีกว่า C_MULTI ปัจจุบัน | ยังไม่ได้ทำ |
| 6 | **End-time optimization** | End time ยาวกว่า GT ~0.9–1.5s ทุก experiment — ลอง penalty บน end duration | ยังไม่ได้ทำ |
| 7 | **Crop ROI ผู้แปล** | ลด noise ใน pose estimation จากพื้นหลังและผู้บรรยาย | ยังไม่ได้ทำ |
| 8 | **Task 2: Hungarian matching** | เปลี่ยน greedy matching → one-to-one Hungarian → IoU จะไม่ใช่ upper-bound อีกต่อไป | ยังไม่ได้ทำ |

---

## Slide 8 — Evaluation Scope (สิ่งสำคัญที่ต้องระบุเสมอ)

> **อัปเดต 4 พ.ค. 2569:** เปลี่ยนจาก text lookup → **index-based evaluation**
> ผลลัพธ์: match ครบ **119/119 cues** ทุก experiment (เดิม 69/119)

**Evaluator เดิม (text lookup — ก่อน May 4):**
- หา cue ใน VTT output ที่มีข้อความตรงกับ GT ด้วย exact text match
- ปัญหา: CC_Aligned (GT) มี 49 entries ที่ annotator แก้ข้อความ → match ไม่ได้
- ผล: วัดได้แค่ 69/119 GT entries (58%)

**Evaluator ใหม่ (index-based — May 4 เป็นต้นไป):**
- `pred[i]` จับคู่กับ `gt[i]` ตาม index ตรงๆ (cue ที่ 1 ↔ GT entry ที่ 1)
- ทำได้เพราะ VTT output มี 119 cues ที่ตรงกับ 119 GT entries พอดี
- ผล: วัดครบ **119/119 cues** ทุก experiment

| ตัวเลข | ความหมายที่ถูกต้อง (May 5) |
|---|---|
| 119/119 matched | **ครบทุก cue** — pred[i] ↔ gt[i] |
| 74% within ±1s | 74% ของ **119 cues** อยู่ใน ±1s |
| 100% within ±3s | 100% ของ **119 cues** อยู่ใน ±3s |
| FrameAcc 82.63% | 82.63% ของ frames ใน 119 cues ถูก label ถูก |

---

## Slide 8b — Known Issues: สถานะปัจจุบัน

> อัปเดต 5 พ.ค. 2569 — สรุปสถานะ issues ทั้งหมด

### Issue 1 — Task 1: `gt_by_text` ใน `evaluate_all.py` เก็บแค่ entry แรก (อัปเดต: เปลี่ยนเป็น index-based)

```python
# เดิม (ก่อน May 4):
if key not in gt_by_text:
    gt_by_text[key] = (t1, t2)   # ← entry ที่สองของ text เดิมถูก drop
```

- **สถานะ:** อัปเดตแล้ว — เปลี่ยนมาใช้ **index-based evaluation** ทั้งหมด
- `gt_by_text` dict ถูกลบออกจาก `evaluate_all.py` ทั้งหมด
- `match_cues()` ปัจจุบัน: `pred_cues[i] ↔ gt_cues[i]` — ไม่ต้องค้นหาด้วย text
- ผล: 119/119 matched (เดิม 69/172)

### Issue 2 — Task 2: Mean IoU เป็น upper-bound estimate (greedy matching) ⚠️ ยังอยู่

- `evaluate_gloss_labeling.py` ใช้ **greedy matching** (non-exclusive)
- แต่ละ prediction จับคู่กับ GT ที่มี IoU สูงสุด โดยไม่ตรวจ exclusive claim
- ตัวเลขจริงจาก run ล่าสุด (5 พ.ค. 2569): **Mean IoU = 0.4199**
- เดิมรายงาน 0.49 (จาก Gloss tier เก่า) — ลดลงเพราะ Gloss_Input ทำ tokenization ต่างออกไป
- วิธีแก้ที่ถูกต้อง: ใช้ **Hungarian matching** (one-to-one assignment)
- ตัวเลขที่ควร report: *"Mean IoU = 0.4199 (greedy upper-bound)"*

### Issue 3 — Task 1: stdev สูงผิดปกติ (สาเหตุได้รับการอธิบาย — ต้องการ investigation ต่อ)

- **เดิม** (text-lookup, 69/172 matched): stdev ≈ **5.5s** — สูงมากผิดปกติ
- **ปัจจุบัน** (index-based, 119/119 matched): stdev = **0.85–1.2s** ต่อ experiment

| Experiment | stdev (index-based) |
|---|---|
| B2 | 0.85s |
| B_MULTI | 0.96s |
| C_MULTI ⭐ | 0.96s |
| C_MULTI_word | 0.96s |
| D_ASL | 1.20s |
| D_ASL_gloss | 1.07s |
| D_ASL_word | 1.08s |

- สาเหตุที่ stdev เดิมสูง: text-lookup เลือก 69 entries ที่ "อยู่ใกล้" GT ต้นฉบับ (bias toward low-offset entries) → 50 entries ที่ offset สูงถูก exclude ออกไป — ทำให้ stdev ต่ำเทียม → แต่ตอนเพิ่ม index-based entries บางตัวที่มี offset สูงขึ้นมาด้วย
- stdev ~1s ปัจจุบัน: สมเหตุสมผล — คือ alignment ส่วนใหญ่ดี มีบาง outlier cues ที่ align ไม่ตรง

---

## Slide 9a — Task 2: Gloss-to-Gloss-Label คืออะไร

**เป้าหมาย Task 2:**

ใน Task 1 เราจัดเวลา subtitle ทั้งประโยค (1 cue = 1 ประโยค ~7.5 คำ) ให้ตรงกับช่วงที่มือแสดง

Task 2 ต้องการ **ระดับละเอียดขึ้น**: แบ่ง 1 ประโยคออกเป็นรายท่ามือแต่ละท่า และหาว่า **ท่ามือไหนตรงกับคำไหน**

```
Task 1 (ระดับประโยค):
  "นักเรียน ต้องเรียน เปรียบเทียบ"   → CC_Aligned 36.3s–40.1s
  ↑ ทั้งประโยคได้ timestamp เดียว

Task 2 (ต้องการ):
  "นักเรียน"    → 36.3s–37.0s   ← ท่ามือเดียว
  "ต้องเรียน"   → 37.3s–38.2s   ← ท่ามือสองท่า
  "เปรียบเทียบ" → 38.5s–40.1s   ← ท่ามือสองท่า
```

**Data ที่มี (EAF):**

| Tier | จำนวน | คืออะไร |
|---|---|---|
| `CC_Input` | 119 cues | subtitle ระดับประโยค (timestamp จากเสียง) |
| `Gloss_Input` | 119 cues | gloss ระดับประโยค (ข้อความจาก Gloss tier, timestamp จาก CC_Input) |
| `Gloss Labeling` | **852 entries** | ground truth รายท่า — annotated ด้วยมือโดยนักวิจัย |

**งาน:** สร้าง predictions 889 entries โดยใช้แค่ `Gloss_Input` + `SIGN segments` + SignCLIP

---

## Slide 9b — Task 2: Pipeline (4 ขั้นตอน)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Test.eaf → tier "Gloss_Input" (119 sentences × ~7.5 tokens avg)    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────────────┐
                    │                                       │
                    ▼                                       ▼
       ┌────────────────────────┐          ┌────────────────────────────────┐
       │  Step 1: Tokenize      │          │  Step 1b: Restrict candidates  │
       │                        │          │                                │
       │  "นักเรียน เรียน       │          │  SIGN segments จาก 04.eaf     │
       │   เปรียบเทียบ"         │          │  → กรองเอาเฉพาะ segments ที่   │
       │  → ["นักเรียน",        │          │    mid ∈ [sentence_start,      │
       │     "เรียน",           │          │           sentence_end]        │
       │     "เปรียบเทียบ"]     │          │  → K candidate segments        │
       │  T = 3 tokens          │          │    (ปกติ K ≈ 15–40)            │
       └──────────┬─────────────┘          └──────────────┬─────────────────┘
                  │                                       │
                  ▼                                       ▼
       ┌────────────────────────┐          ┌────────────────────────────────┐
       │  Step 2: Token Embed   │          │  Step 2b: Sign Embed           │
       │                        │          │                                │
       │  SignCLIP multilingual  │          │  precomputed .npy              │
       │  text encoder          │          │  (from Task 1 pipeline)        │
       │  "<en> <bfi> นักเรียน" │          │  → sign_embs (K × 768)         │
       │  → token_embs (T×768)  │          │                                │
       │  cached to .npz        │          │                                │
       └──────────┬─────────────┘          └──────────────┬─────────────────┘
                  │                                       │
                  └───────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │  Step 3: Similarity Matrix        │
                    │                                   │
                    │  sim (T × K) = cosine similarity  │
                    │  + row softmax normalization      │
                    │                                   │
                    │  ตัวอย่าง (T=3, K=5):             │
                    │      g1   g2   g3   g4   g5       │
                    │  t1 [0.8  0.6  0.2  0.1  0.1]     │
                    │  t2 [0.1  0.2  0.7  0.8  0.3]     │
                    │  t3 [0.1  0.1  0.2  0.3  0.9]     │
                    └──────────────┬────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │  Step 4: Monotonic DP             │
                    │  (per sentence, T~7 K~30)         │
                    │                                   │
                    │  assign each token to             │
                    │  contiguous range of segments     │
                    │  with min cost                    │
                    │                                   │
                    │  → 889 predictions total          │
                    └──────────────┬────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────────┐
                    │  OUTPUT: gloss_labels_pred.csv    │
                    │  + gloss_labels_pred.vtt          │
                    │  + GLOSS_LABEL_PRED tier in EAF  │
                    └───────────────────────────────────┘
```

---

## Slide 9c — Task 2: Gloss_Input Tier สร้างยังไง

**ปัญหาตั้งต้น:** `Gloss` tier มี timestamp จาก manual annotation (ไม่สัมพันธ์กับ CC_Input)
เราต้องการ tier ที่มี **timestamp จาก CC_Input** แต่ **text จาก Gloss**

**ทำไมต้องสร้าง Gloss_Input:**

```
CC_Input tier (timestamp จากเสียง, text ภาษาไทยพูด):
  35.0s–36.8s  "นักเรียน ต้องเรียนรู้ การเปรียบเทียบ"

Gloss tier (timestamp จาก manual annotation, text ภาษามือ):
  36.3s–40.1s  "นักเรียน เรียน เปรียบเทียบ"

Gloss_Input tier (ผสม CC_Input timestamp + Gloss text):
  35.0s–36.8s  "นักเรียน เรียน เปรียบเทียบ"
                ↑ timestamp ยังเป็นเสียง แต่ text เป็นภาษามือ
                  → DP ใน Task 2 จะ align ออกมาเป็น timestamp จริงของมือ
```

**Scripts ที่ใช้:**

```
make_gloss_input_tier.py
  จับคู่ CC_Input ↔ Gloss ด้วย maximum overlap
  → เพิ่ม tier "Gloss_Input" เข้า Test.eaf
  → 119 entries, 0 fallback

make_gloss_cc_vtt.py
  สร้าง VTT ที่มี timestamp จาก CC_Input + text จาก Gloss_Input
  → subtitles_gloss_cc_time/04.vtt
  → ใช้เป็น input ของ C_MULTI / D_ASL_gloss experiments ใน Task 1
```

**Matching method (maximum overlap):**
```python
for cc_cue in cc_cues:
    best_gloss = max(gloss_anns,
                     key=lambda g: overlap(cc_cue.start, cc_cue.end, g.start, g.end))
    if overlap(...) > 0:
        use_text = best_gloss.text
    else:
        use_text = cc_cue.text   # fallback (0 cases ในชุดข้อมูลนี้)
```

---

## Slide 9d — Task 2: DP Algorithm (per sentence)

DP ของ Task 2 มีขนาดเล็กกว่า Task 1 มาก — ทำงานแยกต่อประโยค (T~7, K~30)

```
State:
  dp[t][j] = min cost ที่ assign tokens 1..t โดย token t สิ้นสุดที่ segment j

Init:
  dp[0][0] = 0,  ทุกอื่น = +∞

Transition (1 ≤ t ≤ T, t ≤ j ≤ K):
  dp[t][j] = min over k ∈ [t .. j] of:
  {
      dp[t-1][k-1]
    + (−cumsum_sim[t−1, k−1..j−1])         ← negative similarity (ยิ่งคล้ายยิ่งดี)
    + gap_penalty=2.0 × inter-segment gaps  ← ลงโทษ group ที่มีช่องว่าง
    + coverage_penalty=0.5 × |group_dur − sentence_dur/T|  ← ให้แต่ละ token ได้ duration เฉลี่ย
  }

Final:
  j* = argmin dp[T][T..K]   ← token สุดท้ายสิ้นสุดที่ segment ไหนก็ได้
  backtrack → ranges[(k*-1, j-1)] สำหรับ t = T → 1
```

**ตัวอย่าง (T=3 tokens, K=5 signs):**

```
              j=1  j=2  j=3  j=4  j=5
 t=1 (token1) [0.8  1.4   ∞    ∞    ∞  ]  ← sign 1 หรือ signs[1:2]
 t=2 (token2) [ ∞   ∞   2.1  1.3  2.0  ]  ← signs[2..4]
 t=3 (token3) [ ∞   ∞    ∞   3.5  2.8  ]  ← signs[3..5]

  Best path: j*=5
  prev[3][5] = 4  → token3 = signs[4:5]
  prev[2][4] = 3  → token2 = signs[3:4]
  prev[1][3] = 1  → token1 = signs[1:3]  (group 2 signs)
```

**ข้อแตกต่างจาก Task 1 DP:**

| | Task 1 | Task 2 |
|---|---|---|
| Scope | วิดีโอทั้งหมด (M=172, N=2780) | ต่อประโยค (T~7, K~30) |
| Constraint | monotonic across all sentences | monotonic ภายในประโยคเดียว |
| Cost | timing + similarity + gap | similarity + gap + duration coverage |
| Pre-shift | ✅ bias correction 1.3s | ✗ (sentence window กำหนด scope แล้ว) |
| Complexity | O(M×W²) ≈ O(172×40²) | O(T×K²) ≈ O(7×30²) — fast |

---

## Slide 9e — Task 2: ผลการทดลอง

**Prototype รันได้ — Mean IoU = 0.4199 (greedy upper-bound, ยังต้องการ Hungarian matching)**

| Metric | ค่า | หมายเหตุ |
|---|---|---|
| Predictions | **889** | 119 sentences × ~7.5 tokens/sentence |
| GT labels | 852 | Gloss Labeling tier |
| fallback_uniform | **0** | ✅ DP หา solution ได้ทุกประโยค |
| **Mean IoU** | **0.4199** | greedy non-exclusive matching |
| Median IoU | 0.4150 | |
| **% IoU ≥ 0.5** | **38.9%** | ~1/3 ของ predictions overlap > 50% |
| **% IoU ≥ 0.3** | **66.0%** | 2/3 overlap ระดับพอใช้ |
| **% temporal overlap** | **93.4%** | ✅ เกือบทุก pred มี overlap กับ GT บางตัว |
| Mean start offset | −0.002s | ≈ 0 — ดีมาก |
| Mean end offset | −0.093s | เกือบ 0 — ดีมาก |
| Exact text match | 10.6% (88/830) | notation ต่างกันระหว่าง tiers |

**ทำไม Exact text match ต่ำ (10.6%):**
```
Gloss_Input token: "ผายมือ"     ← notation ของผู้แปล (functional description)
Gloss Labeling GT: "ผายมือ_1"   ← annotator ใช้ suffix index

Gloss_Input token: "เปรียบเทียบ"  ← รวมเป็นคำเดียว
Gloss Labeling GT: "เปรียบ" + "เทียบ"  ← แยกสองท่า

→ Text match ต่ำเป็นเรื่องปกติ — IoU เป็น metric ที่สำคัญกว่า
```

**สิ่งที่สังเกตได้:**
- 93.4% มี temporal overlap → ระบบ "หาที่ถูกต้อง" ได้เกือบทุกครั้ง
- Mean IoU 0.42 ยังต่ำกว่า 0.5 — ชี้ว่า boundary ยังไม่แม่นยำ
- Coverage penalty (0.5) ช่วยให้ duration เฉลี่ยต่อ token สมเหตุสมผล
- ไม่มี fallback → DP มี solution ทุกประโยค (รวมประโยคที่ T > K ซึ่งไม่เกิดขึ้นในชุดนี้)

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

**ความคืบหน้า Task 1:**
- Pipeline ครบทุกขั้นตอน — segment → embed → align → post-process → evaluate
- ทดสอบ 7 experiments เพื่อเปรียบเทียบโมเดลและ text input
- **C_MULTI (Multilingual + Gloss text) ดีที่สุด**: mean offset −0.16s, 100% ±3s, stdev 0.96s
- Post-processing ลด overlap จาก ~88% เป็น 0% โดยไม่กระทบ timing metrics
- Evaluation: index-based 119/119 matched, archive เป็น CSV

**ความคืบหน้า Task 2 (Prototype):**
- DP ระดับ token ภายในประโยค: T~7 tokens, K~30 candidates per sentence
- 889 predictions, 0 fallbacks, Mean IoU 0.4199, 93.4% temporal overlap
- ใช้ SignCLIP multilingual text encoder พร้อม language tag `<en> <bfi>`
- ยังต้องการ: Hungarian matching evaluation (ปัจจุบัน greedy upper-bound)

**สิ่งที่กำลังทำต่อ:**
- ทดสอบ Task 1 บนวิดีโอเพิ่ม (5-10 คลิป) เพื่อตรวจสอบว่า C_MULTI generalize ได้จริง
- Task 2: เปลี่ยน evaluate ด้วย Hungarian matching (one-to-one)
- Task 2: ปรับ gap_penalty / coverage_penalty เพื่อเพิ่ม IoU ≥ 0.5
- End-time optimization: Task 1 end time ยาวกว่า GT ~0.9–1.5s ทุก experiment

**ข้อค้นพบสำคัญ:**
- Multilingual SignCLIP generalize มา TSL ได้ดี แม้ไม่มี TSL training data
- Gloss text (ภาษามือ) ให้ embedding ที่ match กับ sign segment ได้ดีกว่า CC text (ภาษาไทยพูด)
- DP alignment ทำงาน cross-lingual ได้โดยไม่ต้อง fine-tune — แต่ยังต้องการ parameter tuning

---

*อ้างอิง: [arXiv:2512.08094](https://arxiv.org/abs/2512.08094) — Jiang et al., 2025 | ข้อมูลครบถ้วนใน [Progress_26042026.md](Progress_26042026.md), [Progress_04052026.md](Progress_04052026.md)*
