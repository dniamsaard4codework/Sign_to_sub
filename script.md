# สคริปต์นำเสนอ — SEA for Thai Sign Language (Task 1 Update)
> อัปเดตล่าสุด: 5 พฤษภาคม 2569

---

## Slide 1 — ภาพรวมโครงการ

- เรานำระบบ **SEA** (Segment → Embed → Align) จากงานวิจัย Oxford/ETH 2025 มาทดสอบกับ **ภาษามือไทย (TSL)** แบบ cross-lingual
- SEA เดิม train กับภาษามือยุโรป — เราต้องการดูว่ามัน generalize มาที่ TSL ได้แค่ไหน

**งานหลัก 2 งาน:**
- **Task 1** — จัดเวลา CC subtitle ให้ตรงกับช่วงที่ผู้แปลแสดงท่ามือ
  - ทดสอบ 7 experiments บน 1 วิดีโอ
  - ผลดีที่สุด (C_MULTI): mean offset −0.16s, 100% อยู่ใน ±3s, F1@0.50 = 88.24%
- **Task 2** — แยก gloss ระดับประโยคออกเป็น annotation รายท่ามือ
  - Prototype: 889 predictions, Mean IoU 0.4199, 93.4% temporal overlap

**ข้อมูลที่ใช้:**
- วิดีโอ `04.mp4` — 11.07 นาที, 1920×1080, 60fps
- EAF annotation 4 tier: CC (172 entries), CC_Aligned (119), Gloss (119), Gloss Labeling (852)
- `CC_Aligned` คือ ground truth ที่นักวิจัย annotate ด้วยมือ — ใช้ evaluate Task 1

---

## Slide 2 — Full Pipeline (ภาพรวม 7 ขั้นตอน)

- **Input:** `04.mp4` + EAF (CC, CC_Aligned, Gloss, Gloss Labeling)
- Pipeline มีทั้งหมด 7 ขั้น:

| ขั้น | ทำอะไร | ผลลัพธ์ |
|---|---|---|
| Step 1 | แปลง EAF → WebVTT (extract tier CC) | `subtitles/04.vtt` — 172 cues |
| Step 2 | Pose estimation (MediaPipe Holistic) | `04.pose` — 543 landmarks/frame |
| Step 3 | Segmentation ตรวจจับท่ามือ (E4s-1) | 2,780 SIGN segments, 418 SENT segments |
| Step 4a | Sign embedding (SignCLIP) | `.npy` shape (2780×768) |
| Step 4b | Text embedding (SignCLIP text encoder) | `.npy` shape (172×768) |
| Step 5 | DP Alignment — จับคู่ subtitle ↔ sign segment | `aligned_output_*/04.vtt` × 7 experiments |
| Step 6 | Overlap fix (clamp end time) | `04_no_overlap.vtt` × 7 |
| Step 7 | Evaluation เทียบกับ CC_Aligned | `evaluation_task1_results.csv` |

- **โมเดล SignCLIP ที่ทดสอบ:** BSL (British SL), Multilingual, ASL (American SL)

---

## Slide 2a — Step 1: Extract CC จาก EAF → VTT

- ปัญหาตั้งต้น: EAF เป็น XML รูปแบบ ELAN — ระบบ SEA รับแค่ WebVTT
- Script `extract_cc_from_eaf.py` ทำงาน 3 ขั้น:
  1. อ่าน XML → สร้าง map `TIME_SLOT_ID → milliseconds`
  2. ค้นหา tier ชื่อ `CC` → ดึง annotation ทุกตัว
  3. แปลง ms → `HH:MM:SS.mmm` → เขียน VTT
- ผลลัพธ์: `subtitles/04.vtt` — 172 cues
- **⚠️ จุดสำคัญ:** timestamp ใน CC sync กับ**เสียงครู** ไม่ใช่มือผู้แปล — นี่คือสิ่งที่ pipeline ทั้งหมดพยายามแก้

---

## Slide 2b — Step 2: Pose Estimation (MediaPipe Holistic)

- เป้าหมาย: แปลงวิดีโอ (pixels) → พิกัดข้อต่อร่างกาย ที่ AI อ่านได้
- MediaPipe Holistic จับ **543 landmarks ต่อเฟรม**:
  - ลำตัว: 33 จุด
  - ใบหน้า: 468 จุด (facial expression เป็นส่วนหนึ่งของภาษามือ)
  - มือซ้าย + ขวา: 21+21 = 42 จุด — ส่วนที่ signmenter ให้ความสนใจมากที่สุด
- วิดีโอ 60fps × 11 นาที ≈ 39,600 เฟรม
- ไฟล์ `.pose` ขนาด 358 MB (binary float — เก็บทุก landmark × ทุกเฟรม)
- **ภาพในใจ:** แทนที่จะเก็บ pixels แสน pixel เก็บแค่ 543 จุดพิกัด — ลดข้อมูลลงมากแต่ยังจับการเคลื่อนไหวได้

---

## Slide 2c — Step 3: Segmentation (E4s-1)

- เป้าหมาย: หาว่า "ช่วงเวลาไหนในวิดีโอที่ผู้แปลกำลังแสดงท่ามือ"
- โมเดล E4s-1 อ่าน `.pose` → ทำนายทุกเฟรมว่า "กำลังแสดงท่ามือ" หรือ "พัก"
  - `--sign-b-threshold 30` — ถ้า probability > 30% = จุดเปลี่ยนท่า (boundary)
  - `--sign-o-threshold 50` — ถ้า probability > 50% = เริ่มท่าใหม่ (onset)
  - รวม consecutive frames ที่ผ่าน threshold → 1 SIGN segment
- ผลลัพธ์:
  - **2,780 SIGN segments** — แต่ละตัวแทนท่ามือ 1 ท่า (เฉลี่ย ~0.2s)
  - **418 SENTENCE segments** — กลุ่มท่ามือที่เป็นประโยค (เฉลี่ย ~1.5s)
- **ภาพในใจ:** เหมือน AI นั่งดูวิดีโอแล้วกดหยุดทุกครั้งที่เห็น "ท่านี้จบ ท่าใหม่เริ่ม"

---

## Slide 2d — Step 4: SignCLIP Embedding

- เป้าหมาย: แปลงทั้ง "ท่ามือ" และ "ข้อความ subtitle" ให้อยู่ใน**พื้นที่เดียวกัน** เพื่อเปรียบเทียบได้
- แนวคิดคล้าย CLIP ของ OpenAI แต่สำหรับภาษามือ:
  - ท่ามือ (pose) → Sign Encoder → vector 768 มิติ
  - ข้อความ subtitle → Text Encoder → vector 768 มิติ
  - ถ้า text "ตรงกับ" ท่ามือ → vectors จะ "ใกล้กัน" ใน 768-dim space
- Sign embeddings: `.npy` shape (2780, 768) × 3 models
- Text embeddings: `.npy` shape (172, 768) × ทั้ง CC text และ Gloss text
- **ทำไม Gloss text ดีกว่า CC text:**
  - CC text: "นักเรียน ต้องเรียนรู้ การเปรียบเทียบ" ← ภาษาไทยพูด
  - Gloss text: "นักเรียน เรียน เปรียบเทียบ" ← ภาษามือไทย คำต่อคำ — ใกล้ท่ามือที่แสดงจริงกว่า

---

## Slide 2e — Step 5: DP Alignment (หัวใจของระบบ)

- เป้าหมาย: หาว่า subtitle cue แต่ละตัวควร map ไปยัง sign segment ไหน โดยรักษา**ลำดับ**ไว้เสมอ

**3 เฟสหลัก:**

- **เฟส 1 — Pre-shift Subtitles (bias correction)**
  - เลื่อน cue.start +1.3s, cue.end +1.0s ก่อนเข้า DP
  - เหตุผล: ผู้แปลแสดงท่า "หลัง" เสียงครูพูดเสมอ ~1-2 วินาที
  - ถ้าไม่ pre-shift DP จะหา sign segment ที่เวลาผิด
  - **📌 ที่มาของค่า 1.3/1.0:** ไม่ใช่ค่าทางทฤษฎี — ได้จาก iterative refinement:
    1. รัน experiment ครั้งแรกด้วย bias ใหญ่ (2.6/2.1s — ค่า BOBSL ดั้งเดิมจาก paper)
    2. ดู **median start offset** ของผลลัพธ์ ถ้า predicted เร็วกว่า GT x วินาที → bias มากเกินไป x วินาที
    3. ลด bias ลงตามค่า median แล้วรันใหม่ → ทำซ้ำจนกว่า median offset ≈ 0
  - **ทำไม Gloss experiments ใช้ค่าน้อยกว่า:** Gloss tier เดิม annotate ด้วยมือ ทำให้ timestamp ใกล้ท่ามือจริงอยู่แล้ว → ช่องว่างกับมือแคบกว่า CC → bias น้อยกว่า
  - **ทำไม start (1.3) ≠ end (1.0):** ผู้แปลมักเริ่มท่าล่าช้ากว่าเสียงมากกว่าที่จะ *จบท่า* ล่าช้า

  | Experiment group | Bias start | Bias end | เหตุผล |
  |---|---|---|---|
  | B2, B_MULTI, D_ASL | 1.8s | 1.5s | ใช้ CC timestamp (เสียงล้วน) |
  | C_MULTI, D_ASL_gloss, D_ASL_word | **1.3s** | **1.0s** | ใช้ Gloss_Input (timestamp ใกล้มือกว่า) |
  | BOBSL (paper default) | 2.6s | 2.1s | interpreter delay สูงกว่า (BSL) |

- **เฟส 2 — Dynamic Programming (Numba @njit)**
  - `dp[i][j]` = cost ต่ำสุดในการ assign cue 1..i โดยใช้ sign segments ถึง index j
  - Monotonic constraint: cue i+1 ต้องใช้ signs ที่ index > j ของ cue i เสมอ
  - Window optimization: ใช้ window_size=40 ลด complexity จาก O(M×N²) → O(M×W²)
  - Cost function = start timing + end timing + duration mismatch + gap penalty + embedding similarity

- **เฟส 3 — Backtrack + Subgroup Refinement**
  - Backtrack จาก dp[M][N] ผ่าน prev matrix → ได้ sign range ของแต่ละ cue
  - Subgroup Refinement: ใน range ที่ DP กำหนด ลอง sub-slice ทุกชุด เลือกที่ cost ต่ำสุด
  - ⚠️ Subgroup refinement เป็น independent ต่อ cue → เกิด overlap (แก้ใน Step 6)

---

## Slide 2f — Step 6: Overlap Fix

- ปัญหา: หลัง DP + subgroup refinement มี cues ซ้อนทับกัน ~88%
- สาเหตุ: subgroup refinement เลือก timestamp อิสระ → end[i] อาจ > start[i+1]
- วิธีแก้: `fix_overlap_vtt.py` — clamp end time
  - ถ้า `end[i] > start[i+1]` → ตัด `end[i] = start[i+1]`
- ผล: overlap 87–89% → **0.0%** ทุก 7 experiments
- **✅ start time ไม่ถูกแตะ** → timing metrics ไม่เปลี่ยนหลัง fix

---

## Slide 2g — Step 7: Evaluation

- เทียบ predicted timestamp กับ CC_Aligned ground truth (119 entries)
- **Metrics หลักที่วัด (21 columns):**
  - Start-time offset: mean, median, stdev, |abs|
  - Window metrics: % cues ที่ |offset| ≤ 1s / 2s / 3s
  - End-time offset: mean, |abs|
  - Frame-level: frame accuracy, F1@IoU 0.10 / 0.25 / 0.50
  - Overlap percentage
- **การ match cue (index-based, May 4 เป็นต้นไป):**
  - `pred[0] ↔ gt[0]`, `pred[1] ↔ gt[1]`, ..., `pred[118] ↔ gt[118]`
  - match ครบ **119/119 ทุก experiment**
  - เดิม (ก่อน May 4): text lookup → match ได้แค่ 69/119 เพราะ annotator แก้ข้อความ 49 entries

---

## Slide 3 — DP Alignment: Core Algorithm

- DP จับคู่ subtitle แต่ละ cue กับ sign segment "กลุ่ม" ที่ minimize cost รวม
- **สูตร cost:**
  - `|cue_start − group_start|` — start timing offset
  - `|cue_end − group_end|` — end timing offset
  - `W_dur × |cue_dur − group_dur|` — duration mismatch
  - `W_gap × total_gap_in_group` — ลงโทษ group ที่มีช่องว่าง
  - `W_sim × (−similarity_total)` — ยิ่งคล้ายกัน cost ยิ่งต่ำ

**พารามิเตอร์ C_MULTI (best run):**

| พารามิเตอร์ | ค่า |
|---|---|
| similarity_weight | 6 |
| dp_duration_penalty_weight | 2 |
| dp_gap_penalty_weight | 8 |
| dp_max_gap | 6 |
| dp_window_size | 40 |
| pr_subs_delta_bias_start | 1.3s |
| pr_subs_delta_bias_end | 1.0s |

---

## Slide 3a — Subgroup Refinement: ทำไม Overlap ถึงเกิด

- หลัง DP backtrack แต่ละ cue ได้ index range ของ sign segments (monotonic by index)
- Subgroup Refinement เลือก best sub-slice ในแต่ละ range **โดยอิสระ** ต่อ cue
- ปัญหา:
  - cue 2 best subgroup → end = 87.4s
  - cue 3 best subgroup → start = 87.1s
  - **cue 2 end > cue 3 start → overlap 0.3s**
- สาเหตุที่ต้องเป็น independent: ต้องอาศัย embedding similarity เฉพาะของแต่ละ cue
- แก้ง่ายกว่าด้วย post-processing clamp (Step 6)
- **Root cause chain:**
  - DP รับประกัน: index[cue i] < index[cue i+1] ✓
  - DP ไม่รับประกัน: end_time[cue i] < start_time[cue i+1] ✗

---

## Slide 4 — สิ่งที่อัปเดตตั้งแต่ครั้งก่อน (Apr 20 → Apr 26 → May 5)

### Apr 20 (ก่อนหน้า)

- รัน 7 experiments ครบแล้ว — ได้ VTT alignment ทั้ง 7 ชุด
- ทำ overlap fix เฉพาะ C_MULTI และ D_ASL_gloss (2 ไฟล์เท่านั้น)
- Comparison EAF มี 7 tiers (pre-overlap เท่านั้น)
- ยังไม่มี evaluation CSV — ผลอยู่ใน stdout เท่านั้น
- ⚠️ ใช้ text lookup → match ได้แค่ 69/119 GT entries

### Apr 26

- Overlap fix ครบทั้ง 7 experiments (เดิม fix แค่ 2 ไฟล์) → 0% ทุก experiment
- บันทึก evaluation เป็น CSV (`evaluation_task1_results.csv`)
- ขยาย comparison EAF จาก 7 → 19 tiers (original 4 + pre/post overlap × 7 + GLOSS_LABEL_PRED)
- Task 2 prototype เพิ่มใหม่: `align_gloss_labels.py`, `evaluate_gloss_labeling.py`
- ⚠️ Apr 26 วัดแค่ 69/119 cues (text lookup biased)

### May 4–5

**1. Evaluation: text lookup → index-based (เปลี่ยนใหญ่ที่สุด)**
- เดิม: match ด้วย exact text → 69/119 (49 entries annotator แก้ข้อความ + duplicate text keys)
- ใหม่: `pred[i] ↔ gt[i]` โดยตรง → **119/119 ทุก experiment**
- stdev ลดจาก ~5.5s → 0.85–1.2s (ไม่ biased อีกต่อไป)

**2. Input tier เปลี่ยน: CC (172) → CC_Input (119), Gloss → Gloss_Input**
- ตัด 53 cues ที่ไม่มีท่ามือออก → CC_Input 119 cues = ตรงกับ GT 119 entries พอดี
- Gloss_Input = timestamp จาก CC_Input + text จาก Gloss tier (script ใหม่ `make_gloss_input_tier.py`)

**3. Metrics เพิ่มจาก 9 → 21 columns**
- เพิ่ม stdev, end-time metrics, frame accuracy, F1@0.10/0.25/0.50
- port มาจาก SEA evaluate_sub_alignment.py

**4. Task 2: ใช้ Gloss_Input tier แทน Gloss tier เดิม**
- IoU ลดลง (0.49 → 0.4199) เพราะ timestamp offset ต่างจาก manual annotation
- แต่สอดคล้องกับ Task 1 pipeline ทั้งระบบ

---

## Slide 5 — Task 1: ผลทั้ง 7 Experiments

> Match ครบ **119/119 cues** ทุก experiment (index-based, อัปเดต 4 พ.ค. 2569)

**Start-offset metrics:**

| Experiment | Text Input | Model | Mean offset | ±1s | ±3s | Overlap |
|---|---|---|---|---|---|---|
| B2 | CC | BSL | +0.26s | 76% | 99% | 0% |
| B_MULTI | CC | Multi | +0.25s | 71% | 99% | 0% |
| **C_MULTI ⭐** | **Gloss** | **Multi** | **−0.16s** | **74%** | **100%** | **0%** |
| C_MULTI_word | Gloss (word) | Multi | −0.23s | 74% | 100% | 0% |
| D_ASL | CC | ASL | +0.38s | 61% | 98% | 0% |
| D_ASL_gloss | Gloss | ASL | −0.12s | 71% | 99% | 0% |
| D_ASL_word | Gloss (word) | ASL | −0.13s | 69% | 99% | 0% |

**Frame-level metrics:**

| Experiment | FrameAcc | F1@.50 | End mean |
|---|---|---|---|
| B2 | 82.81% | 88.24% | +1.33s |
| **C_MULTI ⭐** | 82.63% | 88.24% | **+0.91s** |
| C_MULTI_word | **82.74%** | **89.08%** | **+0.84s** |
| D_ASL | 77.31% | 77.31% | +1.46s |

**สิ่งที่สังเกตได้:**
- `multilingual` > `bsl` > `asl` สำหรับ TSL ทั้ง start-offset และ F1
- Gloss text > CC text — gloss ใกล้ท่ามือที่แสดงจริงกว่า
- **C_MULTI** คือ champion run: mean offset −0.16s (ใกล้ศูนย์มาก), ±3s = 100%
- End time ยาวกว่า GT ~0.9–1.5s ทุก experiment (DP ไม่ optimize end time โดยตรง)
- F1@0.10 ≥ 97% ทุก experiment → segment-level placement ถูกต้องสูงมาก

---

## Slide 6 — Task 1: Post-processing (Overlap Fix ครบ 7 experiments)

- ก่อน fix: overlap ~86–88% ทุก experiment
- หลัง fix: **0.0% ทุก experiment**
- จำนวน pairs ที่ fix: ~102–104 จาก 118 pairs ต่อ experiment
- Start time ไม่ถูกแตะ → mean offset และ window metrics ไม่เปลี่ยน
- ทำไม overlap สูงมาก:
  - Sign segments หนาแน่น (2,780 segs / 660 วินาที ≈ 4.2 segs/s)
  - CC มี 172 cues แต่ GT slot มีแค่ ~119 → บาง cues แชร์ sign cluster → เหลื่อม
  - DP window กว้าง (40 signs) → subgroup มีอิสระเลือกสูง

---

## Slide 7 — Task 1: สิ่งที่ยังต้องทำต่อ

| ลำดับ | งาน | สถานะ |
|---|---|---|
| 1 | **วิเคราะห์ outlier cues** (offset > ±3s) | ยังไม่ได้ทำ |
| 2 | ~~เปลี่ยน text lookup → index-based~~ | ✅ อัปเดตแล้ว (May 4) |
| 3 | **ทดสอบกับวิดีโอเพิ่ม (5-10 คลิป)** | ยังไม่ได้ทำ |
| 4 | **Parameter sweep** (gap_penalty, similarity_weight ต่างๆ) | ยังไม่ได้ทำ |
| 5 | **End-time optimization** | ยังไม่ได้ทำ |
| 6 | **Crop ROI ผู้แปล** (ลด noise ใน pose estimation) | ยังไม่ได้ทำ |
| 7 | **Task 2: Hungarian matching** (เปลี่ยนจาก greedy → one-to-one) | ยังไม่ได้ทำ |

---

## Slide 8 — Evaluation Scope

- **Evaluator ใหม่ (index-based, May 4 เป็นต้นไป):**
  - `pred[i] ↔ gt[i]` ตาม index ตรงๆ
  - match ครบ **119/119 cues** ทุก experiment
  - ทำได้เพราะ CC_Input (119 cues) ↔ GT (119 entries) พอดีกัน

- **Evaluator เดิม (text lookup, ก่อน May 4):**
  - match ด้วย exact text → match ได้แค่ 69/119 (58%)
  - 49 entries ใน GT annotator แก้ข้อความ + duplicate keys → ถูก exclude
  - stdev ≈ 5.5s (biased selection)

**ความหมายของตัวเลขปัจจุบัน (C_MULTI):**
- 74% within ±1s → 74% ของ **119 cues** อยู่ใน ±1s
- 100% within ±3s → **ทุก cue** อยู่ใน ±3s
- FrameAcc 82.63% → 82.63% ของ frames ใน 119 cues ถูก label ถูก

---

## Slide 8b — Known Issues (สถานะปัจจุบัน)

- **Issue 1 — `gt_by_text` duplicate drop:** ✅ แก้แล้ว — เปลี่ยนเป็น index-based ทั้งหมด

- **Issue 2 — Task 2: Mean IoU เป็น greedy upper-bound:** ⚠️ ยังอยู่
  - `evaluate_gloss_labeling.py` ใช้ greedy matching (non-exclusive)
  - ตัวเลข: **Mean IoU = 0.4199** (greedy upper-bound)
  - วิธีแก้ที่ถูกต้อง: Hungarian matching (one-to-one assignment)
  - ควร report ว่า "Mean IoU = 0.4199 (greedy upper-bound)"

- **Issue 3 — stdev:** ✅ อธิบายสาเหตุแล้ว
  - เดิม (text-lookup): stdev ≈ 5.5s — biased selection
  - ปัจจุบัน (index-based): stdev = 0.85–1.2s ต่อ experiment — สมเหตุสมผล

---

## Slide 9a — Task 2: Gloss-to-Gloss-Label คืออะไร

- Task 1 จัดเวลา subtitle ระดับประโยค (1 cue = 1 ประโยค ~7.5 คำ)
- Task 2 ต้องการ**ระดับละเอียดขึ้น**: แบ่ง 1 ประโยคเป็นรายท่ามือ และหาว่าท่ามือไหนตรงกับคำไหน

**ตัวอย่าง:**
- Task 1: "นักเรียน ต้องเรียน เปรียบเทียบ" → CC_Aligned 36.3s–40.1s (ทั้งประโยค)
- Task 2: "นักเรียน" → 36.3–37.0s | "ต้องเรียน" → 37.3–38.2s | "เปรียบเทียบ" → 38.5–40.1s

**Data ที่มี:**
- `CC_Input` — 119 cues (timestamp จากเสียง)
- `Gloss_Input` — 119 cues (text ภาษามือ, timestamp จาก CC_Input)
- `Gloss Labeling` — **852 entries** (ground truth รายท่า — annotated ด้วยมือ)

---

## Slide 9b — Task 2: Pipeline (4 ขั้นตอน)

- **Step 1** — Tokenize ประโยคจาก Gloss_Input (เฉลี่ย ~7.5 tokens/ประโยค)
- **Step 1b** — กรอง SIGN segments ให้เหลือเฉพาะที่ mid ∈ [sentence_start, sentence_end]
- **Step 2** — Token Embedding (SignCLIP multilingual text encoder) → token_embs (T×768)
- **Step 2b** — Sign Embedding จาก precomputed .npy → sign_embs (K×768)
- **Step 3** — Similarity Matrix (T×K) + row softmax normalization
- **Step 4** — Monotonic DP per sentence (T~7, K~30)
  - assign each token to contiguous range of segments with min cost
  - Cost = similarity + gap penalty + duration coverage
- **Output:** `gloss_labels_pred.csv`, `gloss_labels_pred.vtt`, tier `GLOSS_LABEL_PRED` ใน EAF

---

## Slide 9c — Task 2: Gloss_Input Tier สร้างยังไง

- `Gloss` tier มี timestamp จาก manual annotation — ไม่สัมพันธ์กับ CC_Input
- ต้องการ tier ที่มี **timestamp จาก CC_Input** + **text จาก Gloss**
- `make_gloss_input_tier.py` — จับคู่ CC_Input ↔ Gloss ด้วย maximum overlap:
  - สำหรับแต่ละ CC_Input cue → ค้นหา Gloss annotation ที่ overlap มากที่สุด
  - ใช้ text ของ Gloss นั้น + timestamp ของ CC_Input
  - 0 fallback (match ครบ 119/119)
- ผล: Gloss_Input 119 entries — timestamp สอดคล้องกับ Task 1 pipeline ทั้งระบบ

---

## Slide 9d — Task 2: DP Algorithm (per sentence)

- DP Task 2 ขนาดเล็กกว่า Task 1 มาก: T~7, K~30 (ต่อประโยค)
- State: `dp[t][j]` = min cost ที่ assign tokens 1..t โดย token t สิ้นสุดที่ segment j
- Cost = negative similarity + gap penalty + duration coverage penalty
- Backtrack → ranges ของแต่ละ token

**ข้อแตกต่างจาก Task 1:**

| | Task 1 | Task 2 |
|---|---|---|
| Scope | วิดีโอทั้งหมด (M=172, N=2780) | ต่อประโยค (T~7, K~30) |
| Pre-shift | ✅ bias +1.3s | ✗ (sentence window กำหนด scope แล้ว) |
| Complexity | O(M×W²) ≈ O(172×40²) | O(T×K²) ≈ O(7×30²) — เร็วมาก |

---

## Slide 9e — Task 2: ผลการทดลอง

- ทำงานได้ครบ: fallback_uniform = **0** (DP หา solution ครบทุกประโยค)

| Metric | ค่า |
|---|---|
| Predictions | **889** |
| GT labels | 852 |
| **Mean IoU** | **0.4199** (greedy upper-bound) |
| % IoU ≥ 0.5 | 38.9% |
| % IoU ≥ 0.3 | 66.0% |
| % temporal overlap | **93.4%** |
| Mean start offset | −0.002s ≈ 0 |
| Mean end offset | −0.093s ≈ 0 |
| Exact text match | 10.6% |

- **ทำไม text match ต่ำ (10.6%):** notation ต่างกัน 2 แบบ:
  - suffix index: Gloss_Input ใช้ "ผายมือ" vs GT ใช้ "ผายมือ_1"
  - tokenization ต่างกัน: Gloss_Input รวมเป็น "เปรียบเทียบ" (1 token) vs GT แยกเป็น "เปรียบ" + "เทียบ" (2 entries)
  - **📌 ดังนั้น text match ไม่ใช่ metric ที่เชื่อถือได้ — IoU สำคัญกว่า**
- **ข้อสังเกตสำคัญ:**
  - 93.4% temporal overlap → ระบบ "หาที่ถูกต้อง" ได้เกือบทุกครั้ง แต่ boundary ยังไม่แม่น
  - Mean IoU 0.42 < 0.5 → ชี้ว่า boundary ยังต้องปรับปรุง
  - Mean start/end offset ≈ 0 → prediction ไม่ shift ซ้าย/ขวาจาก GT อย่างเป็นระบบ
  - Coverage penalty (0.5) ช่วยให้ duration เฉลี่ยต่อ token สมเหตุสมผล
  - DP fallback = 0 → มี solution ทุกประโยค (รวมประโยคที่ T > K ซึ่งไม่เกิดขึ้นในชุดนี้)
  - **⚠️ ตัวเลข Mean IoU = 0.4199 เป็น greedy upper-bound** ต้องระบุเสมอเมื่อ report
- **SignCLIP language tag ที่ใช้ใน Task 2:** `<en> <bfi>` (English + British Sign Language code) — เพราะ multilingual model ใช้ language tag นำหน้า text

---

## Slide 9 — Output Artifacts (ไฟล์ผลลัพธ์ที่มีอยู่)

```
example_alignment/
├── aligned_output_multi_gloss/
│   ├── 04.vtt                 ← C_MULTI alignment (pre-fix)
│   └── 04_no_overlap.vtt      ← C_MULTI alignment (post-fix) ← ใช้งานจริง
│
├── aligned_output_*/04_no_overlap.vtt   ← ทั้ง 7 experiments
│
├── evaluation_task1_results.csv         ← metrics ทุก experiment (14 rows × 21 cols)
│
├── การเปรียบเทียบฯ_comparison.eaf      ← ELAN file รวม:
│     CC, CC_Aligned, Gloss, Gloss Labeling   (4 original tiers)
│     SUBTITLE_*                              (7 pre-overlap tiers)
│     SUBTITLE_*_no_overlap                   (7 post-overlap tiers)
│     GLOSS_LABEL_PRED                        (Task 2 predictions)
│
├── gloss_labels_pred.csv                ← Task 2: 889 predictions
├── gloss_labels_pred.vtt                ← Task 2: VTT format
└── figures/timeline_first_2min.png      ← visualization 4 lanes (0–120s)
```

**📌 ไฟล์ที่ใช้ดูผลจริงใน ELAN:** `การเปรียบเทียบฯ_comparison.eaf` — เปิดได้ด้วย ELAN และดู 19 tiers เคียงกัน

---

## Slide 10 — สรุปและขั้นตอนถัดไป

**ความคืบหน้า Task 1:**
- Pipeline ครบทุกขั้นตอน — segment → embed → align → post-process → evaluate
- ทดสอบ 7 experiments เปรียบเทียบโมเดล (BSL/Multi/ASL) × text input (CC/Gloss)
- **C_MULTI (Multilingual + Gloss text) ดีที่สุด:** mean offset −0.16s, 100% ±3s, F1@0.50 = 88.24%
- Post-processing: overlap ~88% → 0% โดยไม่กระทบ timing metrics
- Evaluation: index-based 119/119 matched, archive เป็น CSV 21 columns

**ความคืบหน้า Task 2 (Prototype):**
- DP ระดับ token ภายในประโยค: T~7 tokens, K~30 candidates per sentence
- 889 predictions, 0 fallbacks, Mean IoU 0.4199 (greedy upper-bound), 93.4% temporal overlap
- ใช้ SignCLIP multilingual text encoder พร้อม language tag `<en> <bfi>`
- ยังต้องการ: Hungarian matching evaluation (one-to-one assignment)

**ข้อค้นพบสำคัญ:**
- **Multilingual SignCLIP generalize มา TSL ได้** แม้ไม่มี TSL training data
- **Gloss text ให้ embedding ที่ match sign segment ได้ดีกว่า CC text** เพราะ notation ตรงกับท่ามือที่แสดงจริง
- **DP alignment ทำงาน cross-lingual ได้โดยไม่ต้อง fine-tune** — แต่ยังต้องการ parameter tuning
- **Bias tuning สำคัญมาก** — การปรับ bias ตาม median offset จริงช่วยเพิ่มความแม่นยำอย่างมีนัยสำคัญ
- **End time ยาวกว่า GT เสมอ (~0.9–1.5s)** เพราะ cost function ไม่มี term สำหรับ end-time โดยตรง

**สิ่งที่กำลังทำต่อ:**
- ทดสอบ Task 1 บนวิดีโอเพิ่ม (5–10 คลิป) — ผลปัจจุบันมาจาก 1 วิดีโอ ยัง generalize ไม่ได้
- Task 2: Hungarian matching (one-to-one) เพื่อได้ค่า IoU ที่ไม่ใช่ upper-bound
- Task 2: ปรับ gap_penalty / coverage_penalty เพื่อเพิ่ม % IoU ≥ 0.5
- End-time optimization: เพิ่ม term ใน cost function สำหรับ end-time error
- Crop ROI ผู้แปล: ลด noise ใน pose estimation จากพื้นหลังและผู้บรรยาย

*อ้างอิง: arXiv:2512.08094 — Jiang et al., 2025 (Oxford/ETH)*

---

## สรุป (Quick Reference)

| งาน | สถานะ | ตัวเลขสำคัญ |
|---|---|---|
| **Task 1** | ✅ 7 experiments, 119/119 match | C_MULTI: mean −0.16s, ±3s = 100%, F1@0.50 = 88.24% |
| **Task 2** | ⚠️ Prototype, greedy upper-bound | Mean IoU 0.4199, 93.4% temporal overlap |
| Overlap fix | ✅ ครบทั้ง 7 experiments | 87–89% → 0% |
| Index-based eval | ✅ May 4 | 119/119 match, stdev 0.85–1.2s |
| Hungarian matching (Task 2) | 🔲 ยังไม่ได้ทำ | — |
| ทดสอบวิดีโอเพิ่ม | 🔲 ยังไม่ได้ทำ | — |
