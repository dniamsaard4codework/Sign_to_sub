# Presentation — 12 พฤษภาคม 2569

> สรุปจาก [Progress_09052026.md](Progress_09052026.md)
> โครงการ SEA — Thai Sign Language Subtitle Alignment

---

## 📑 Slide Scope

> **สร้าง slides เฉพาะ Part 1–3 เท่านั้น**
> Part 4 เป็นข้อเสนอ forward-looking สำหรับ progress update รอบหน้า
> (ไม่ใช่เนื้อหา presentation)

| Part | Section | นำไปทำ slides? |
| :---: | --- | :---: |
| **0** | Project Overview — Context, Input Data, S/E Steps | ✅ ใช่ |
| **1** | Task 1 — Pipeline Recap (DP + Overlap) + Experiments | ✅ ใช่ |
| **2** | Task 2 — สิ่งที่เพิ่มจาก Task 1 | ✅ ใช่ |
| **3** | ผลการทดลอง — Gloss vs Gloss_Input | ✅ ใช่ |
| 4 | What's Next — สิ่งที่ควรเพิ่มในรอบหน้า | ❌ ไม่ — สำหรับ progress doc |

---

## สารบัญ

0. [PART 0 — Project Overview & Pipeline Foundations](#part-0--project-overview--pipeline-foundations)
   - Slide 0.1 — Context: ทำไมต้องทำ SEA สำหรับ TSL
   - Slide 0.2 — Input Data: Test.eaf และ Tiers
   - Slide 0.3 — S: Segment (MediaPipe Holistic + E4s-1 GRU)
   - Slide 0.4 — E: Embed (SignCLIP + Similarity Matrix)
1. [PART 1 — Task 1 Pipeline Recap](#part-1--task-1-pipeline-recap)
   - Slide 1.1 — ภาพรวม pipeline
   - Slide 1.2 — DP alignment ทำงานอย่างไร
   - Slide 1.3 — ทำไม Overlap ถึงเกิด (และวิธีแก้)
   - Slide 1.4 — 7 Experiments: วิวัฒนาการและเหตุผล
   - Slide 1.5 — Full Results Table
   - Slide 1.6 — Evaluation Method: Text-lookup → Index-based
2. [PART 2 — Task 2 สิ่งที่เพิ่มจาก Task 1](#part-2--task-2-สิ่งที่เพิ่มจาก-task-1)
   - Slide 2.1 — Granularity: sentence → sign-gesture
   - Slide 2.2 — Per-sentence Monotonic DP
   - Slide 2.3 — ส่วนประกอบใหม่ที่ Task 2 เพิ่ม
3. [PART 3 — Gloss vs Gloss_Input Results](#part-3--gloss-vs-gloss_input-results)
   - Slide 3.1 — Setup และตัวเลขสรุป
   - Slide 3.2 — IoU bucket distribution
   - Slide 3.3 — ทำไม Gloss ชนะ
   - Slide 3.4 — Caveat: Ground-truth leakage
   - Slide 3.5 — Recommendation
4. [Closing Slide — Recap (Part 0–3 wrap-up)](#closing-slide--recap-part-03-wrap-up)
5. [Appendix — What's Next (progress doc only)](#appendix--whats-next-progress-doc-only) — *not slides*

---

═══════════════════════════════════════════════════════════════════════

## PART 0 — Project Overview & Pipeline Foundations

> 📊 **Slides for this part: 0.1, 0.2, 0.3, 0.4**
> เป้าหมาย: ให้ผู้อ่านที่ไม่เคยเห็น SEA เข้าใจ context, ข้อมูล, และ 2 ขั้นตอนแรก (Segment + Embed) ก่อนเข้า DP

═══════════════════════════════════════════════════════════════════════

───────────────────────────────────────────────────────────────────────

### Slide 0.1 — Context: ทำไมต้องทำ SEA สำหรับ TSL

#### SEA คืออะไร

**SEA** = **S**egment → **E**mbed → **A**lign — pipeline สำหรับ align คำบรรยายกับท่ามือในวิดีโอภาษามือ
พัฒนาโดย Jiang et al. (EMNLP 2024, [arXiv:2512.08094](https://arxiv.org/abs/2512.08094)) สำหรับ BSL (British Sign Language)
**งานนี้นำ SEA มา cross-lingual transfer กับ TSL (Thai Sign Language) โดยไม่ต้อง retrain โมเดล**

#### ทำไมถึงจำเป็น — Sign Delay Problem

```text
Speaker (เสียงพูด):  "เด็ก  เรียน"
                      │      │
                      ▼      ▼
Timeline:  ══════════[35.0s][36.5s]═══════════════
                                    ↑ speech timestamp

Interpreter (ท่ามือ):              "เด็ก  เรียน"
                                    │      │
                                    ▼      ▼
Timeline:  ════════════════════════[36.0s][37.2s]══
                                    ↑ sign timestamp
                                    (ช้ากว่าเสียง ~0.5–2s)
```

ผู้แปลต้องรับฟังก่อนแล้วจึงแปล → คำบรรยาย (timestamp จากเสียง) **ไม่ตรงกับจังหวะท่ามือจริงเสมอ**

#### 2 Tasks ที่แก้ปัญหา

| Task | Input → Output | Ground Truth | เป้าหมาย |
| --- | --- | --- | --- |
| **Task 1** | CC_Input (119 cues, speech timing) → CC_Aligned (119 cues, sign timing) | CC_Aligned (manual, นักวิจัย) | เลื่อน subtitle timestamp ให้ตรงกับท่ามือ |
| **Task 2** | Gloss (119 sentences with tokens) → Gloss Labeling (852 sign-level entries) | Gloss Labeling (manual, นักวิจัย) | ลงลึกจาก sentence → ท่ามือเดี่ยว (sub-sentence) |

---

───────────────────────────────────────────────────────────────────────

### Slide 0.2 — Input Data: Test.eaf และ Tiers

#### วิดีโอที่ใช้ทดสอบ

| ไฟล์ | คำอธิบาย | ขนาด |
| --- | --- | --- |
| `04.mp4` | วิดีโอ "การเปรียบเทียบและเรียงลำดับ" — 11.07 นาที, 1920×1080, 60fps | ~80 MB |
| `04.pose` | Skeleton landmarks จาก MediaPipe Holistic (543 จุด × ทุกเฟรม) | **358 MB** |
| `Test.eaf` | ELAN annotation file — source ของ tiers ทั้งหมด | — |

#### Tiers ใน Test.eaf

```text
┌─────────────────┬────────┬──────────────────────────────────────────────────┐
│ Tier            │ Count  │ บทบาท                                            │
├─────────────────┼────────┼──────────────────────────────────────────────────┤
│ CC              │   172  │ คำบรรยายดิบ (timestamp จากเสียง) — ไม่ใช้แล้ว    │
│ CC_Input        │   119  │ คำบรรยาย curated → INPUT Task 1                  │
│ CC_Aligned      │   119  │ ที่นักวิจัย align ด้วยมือ → GROUND TRUTH Task 1  │
│ Gloss           │   119  │ Gloss tier ดั้งเดิม → INPUT Task 2 (ดีที่สุด)   │
│ Gloss_Input     │   119  │ Gloss tier curated → INPUT Task 2 (ด้อยกว่า)     │
│ Gloss Labeling  │   852  │ annotation รายท่ามือ → GROUND TRUTH Task 2       │
└─────────────────┴────────┴──────────────────────────────────────────────────┘
```

> **วิวัฒนาการ:** เริ่มต้นใช้ CC (172 cues) → เปลี่ยนเป็น CC_Input (119 cues) เมื่อมี Test.eaf curated
> **ทำไม Gloss ≠ Gloss_Input:** Gloss มี 852 tokens = 852 GT entries (1:1 ตรงกัน); Gloss_Input มี 889 tokens (+37 ส่วนเกิน)

---

───────────────────────────────────────────────────────────────────────

### Slide 0.3 — S: Segment ตรวจจับท่ามือแต่ละท่า

#### Step 1 — Pose Estimation (MediaPipe Holistic)

```text
04.mp4  ─►  videos_to_poses  ─►  04.pose (358 MB)
            │
            ├ --format mediapipe
            ├ --model_complexity 2      (ความแม่นยำสูงสุด)
            ├ --refine_face_landmarks   (ใบหน้าละเอียด)
            └ --no-smooth_landmarks     (ต้องการ boundary คมชัด)

Output: 543 landmarks ต่อเฟรม × 60fps
        └─ มือ: 21×2 = 42 จุด
           ใบหน้า: 468 จุด
           ลำตัว: 33 จุด
```

#### Step 2 — Segmentation (E4s-1 GRU)

โมเดล **E4s-1** (EMNLP 2023) — GRU-based binary classifier ต่อเฟรม train บน BOBSL dataset (BSL):

```text
04.pose  ─►  E4s-1 GRU  ─►  probability per frame: [sign] vs [rest/transition]
                │
         Threshold 2 ค่า:
           sign-b-threshold = 30  →  boundary candidate (จุดเปลี่ยนท่า)
           sign-o-threshold = 50  →  onset candidate (จุดเริ่มท่าใหม่)
                │
         Non-max suppression
                │
         Output EAF:
           SIGN tier     : 2,780 segments  (แต่ละ segment = 1 ท่ามือ)
           SENTENCE tier : 418 segments    (กลุ่มประโยค)
```

| threshold | ค่าสูง | ค่าต่ำ |
| --- | --- | --- |
| `sign-b-threshold` | ตัด segment น้อยลง | segment ถี่ขึ้น |
| `sign-o-threshold` | จับเฉพาะท่าชัดเจน | จับท่าเบาๆ ด้วย |

> ⚠️ **Critical warnings:**
> (1) ต้องส่ง `--sign-b-threshold 30 --sign-o-threshold 50` เสมอ — default ใน config.py คือ 70 (ผิด)
> (2) `--segmentation_dir` ชี้ไป **parent dir** (`segmentation_output`) ไม่ใช่ `E4s-1_30_50`

---

───────────────────────────────────────────────────────────────────────

### Slide 0.4 — E: Embed แปลงท่ามือและข้อความเป็น Vector

#### SignCLIP คืออะไร

**SignCLIP** (EMNLP 2024) — โมเดล multimodal ที่ project ทั้ง sign pose และ text ลงใน **768-dim embedding space เดียวกัน**

- ท่ามือ "เด็ก" และ text "เด็ก" → vector ที่ **ใกล้กัน** ใน space นี้ (cosine similarity สูง)

| Variant | Checkpoint | Train language | Suitability สำหรับ TSL |
| --- | --- | --- | --- |
| `bsl` | `bobsl_finetune_checkpoint_best.pt` | British Sign Language | ปานกลาง |
| `multilingual` | `baseline_temporal_checkpoint_best.pt` | หลายภาษามือ | **ดีที่สุด ✓** |
| `asl` | `asl_finetune_checkpoint_best.pt` | American Sign Language | ด้อยกว่า |

> **ทำไม multilingual ดีที่สุดสำหรับ TSL:** train บน dataset หลายภาษา → generalize มา TSL ได้ดีกว่า BSL-only หรือ ASL-only

#### Sign Embedding (`extract_episode_features.py --feature_type sign`)

```text
For each SIGN segment [start, end]:
  1. slice pose frames จาก 04.pose
  2. normalize landmark ตาม shoulder reference
  3. zero-pad / trim → 32 frames คงที่
  4. forward pass SignCLIP visual encoder
  5. pooled output → 768-dim vector

Output: (2780, 768) numpy array
```

#### Text/Subtitle Embedding (`--feature_type subtitle`)

```text
For each cue text:
  1. prepend language tag: "<en> <bfi>" + cue_text
  2. tokenize → token IDs
  3. forward pass SignCLIP text encoder
  4. pooled output → 768-dim vector

Output: (119, 768) numpy array
```

#### Similarity Matrix — สะพานระหว่าง Text กับ Sign

```text
subtitle_emb  (119 × 768)   × sign_emb.T  (768 × 2780)
         ↓
sim_matrix  (119 × 2780)   ← sim[i][j] = "ความเหมาะสม" ของ cue_i กับ sign_j
         ↓
softmax normalization (row-wise)   ← amplify peaks, suppress noise
         ↓
prefix cumsum (sim_cumsum)         ← คำนวณ similarity ของ range [k,j) ได้ O(1)
```

> **ทำไม softmax ไม่ใช่ raw cosine:** Raw cosine ของ SignCLIP กระจุกแค่ ~0.2–0.4 ทั่ว matrix (ต่างกันน้อย) — softmax amplify ค่าสูงสุดของแต่ละ row → DP เห็น signal ที่ชัดขึ้นว่า segment ไหนเหมาะกับ cue ไหน

---

═══════════════════════════════════════════════════════════════════════

## PART 1 — Task 1 Pipeline Recap

> 📊 **Slides for this part: 1.1, 1.2, 1.3**
> เป้าหมาย: recap pipeline เดิม + เน้น DP กับ overlap

═══════════════════════════════════════════════════════════════════════

### Slide 1.1 — ภาพรวม Pipeline

```text
04.mp4 ─► [Pose]─► 04.pose ─► [Segment]─► SIGN segs (2780)
                                                │
04.eaf ─► CC_Input (119) ──────────────────┐    │
        │  + Gloss_Input (119, 889 tok) ──┐│    │
        ▼                                  ▼▼   ▼
   SignCLIP embed (text)         SignCLIP embed (video segs)
                │                            │
                └────────────┬───────────────┘
                             ▼
                ┌──────────────────────────┐
                │  Step F: DP Alignment    │  ← เป้าหมาย: align 119 cues
                │   (sentence-level)       │     กับ sign segments
                └──────────┬───────────────┘
                           ▼
                ┌──────────────────────────┐
                │  Step G: Overlap fix     │  ← clamp ends → 0% overlap
                │   (post-processing)      │
                └──────────┬───────────────┘
                           ▼
                  04_no_overlap.vtt
```

**Best run: C_MULTI ⭐** (Multilingual + Gloss text):

- Mean offset = **−0.16 s**
- ±1 s = 74 %, ±2 s = 95 %, **±3 s = 100 %**
- Overlap = 0 % (จากเดิม 88 %)
- F1 @ 0.50 IoU = 88.2 %

---

───────────────────────────────────────────────────────────────────────

### Slide 1.2 — DP Alignment ทำงานอย่างไร (Recap)

#### Problem Setup

```text
Input:  M cues (CC_Input)            = 119  ← เวลาจากเสียงพูด
        N sign segments (SIGN tier)  = 2780 ← จากท่ามือจริง

Goal:   หา partition ของ N segments เป็น M contiguous groups เรียงตามลำดับ
        แต่ละ group i ผูกกับ cue i — ให้ total cost ต่ำสุด
```

**ตัวอย่าง:**

```text
cues:      [ c1 ][ c2 ][ c3 ] ... [ c119 ]
segments:  [ s1 s2 s3 | s4 s5 | s6 s7 s8 s9 | ... | s2780 ]
              group 1    grp2     group 3
              (← c1)    (← c2)    (← c3)
```

---

#### DP State & Recurrence

**State:**

$$
\text{dp}[i][j] = \text{ต้นทุนต่ำสุดที่ assign cues } 1{..}i \text{ โดย cue } i \text{ จบที่ segment } j
$$

**Boundary:** $\text{dp}[0][0] = 0$, ส่วนอื่น $= +\infty$

**Transition:** สำหรับ cue *i* ที่ "เริ่มที่ segment *k*" และ "จบที่ segment *j*" (โดย $k \leq j$):

$$
\text{dp}[i][j] = \min_{k \,\in\, [i-1,\, j]} \Bigl( \text{dp}[i-1][k-1] + C(i, k, j) \Bigr)
$$

พร้อมเก็บ `prev[i][j] = k*` เพื่อ backtrack หลังเสร็จ

**Final answer:**

$$
j^* = \arg\min_j \text{dp}[M][j]
$$

จากนั้น backtrack ผ่าน `prev[i][j]` จาก $i=M$ ลงไป $i=1$ เพื่อคืน boundary
ของแต่ละ cue → timestamp ใหม่ของ cue *i* = `(seg[k_i].start, seg[j_i].end)`

---

#### Cost Function — Term-by-term

$$
C(i, k, j) = \underbrace{|\text{cue}_i.\text{start} - \text{seg}_k.\text{start}|}_{T_1}
+ \underbrace{|\text{cue}_i.\text{end} - \text{seg}_j.\text{end}|}_{T_2}
+ \underbrace{w_D \cdot |\text{cue\_dur} - \text{group\_dur}|}_{T_3}
+ \underbrace{w_G \cdot \text{gap}(k, j)}_{T_4}
- \underbrace{w_S \cdot \text{sim\_cum}[i][k][j]}_{T_5}
$$

| Term | Weight | สูตร / ความหมาย |
| :---: | :---: | --- |
| **T₁** Start align | 1 | $\lvert \text{cue}_i.\text{start} - \text{seg}_k.\text{start} \rvert$ — "gravity" ดึง start ของ cue ไม่ให้หลุดจากเวลาเดิม |
| **T₂** End align | 1 | $\lvert \text{cue}_i.\text{end} - \text{seg}_j.\text{end} \rvert$ — gravity เดียวกันที่ end |
| **T₃** Duration | $w_D = 2$ | ลงโทษถ้า group_dur ≠ cue_dur — ป้องกัน group ที่สั้น/ยาวเกินไป |
| **T₄** **Gap penalty** | $w_G = 8$ | $\sum_{p=k}^{j-1} \max(0, \text{seg}_{p+1}.\text{start} - \text{seg}_p.\text{end})$ — **term ที่สำคัญที่สุด** ป้องกันการเลือก group ที่มี "รู" ใหญ่ระหว่าง segments |
| **T₅** **Similarity** | $w_S = 6$ | $-\sum_{s=k}^{j-1} \text{sim}[i][s]$ — เครื่องหมายลบ = similarity สูง ⇒ cost ต่ำ ⇒ DP ชอบ |

> 💡 **Trade-off:** T₁/T₂ ดึงให้ตรงเวลาเดิม vs T₅ ดึงให้ตรง content (sign)
> — ตัวอย่าง: ถ้าผู้แปลแสดงท่ามือช้ากว่าเสียง 1 วินาที, T₁ อยากให้ start
> อยู่ที่เวลาเสียง แต่ T₅ ดันไปทาง sign segment ที่ similarity สูงกว่า
> ผลคือ DP ทำให้ **เลื่อนไปกลางๆ** ตาม penalty weights

---

#### Precomputations (ทำก่อน DP loop)

| Precompute | Complexity | ใช้ใน |
| --- | --- | --- |
| `gap_cost[k][j]` = cumulative gap sum | $O(N)$ | T₄ ใน $O(1)$ |
| `sim_cumsum[i][j]` = prefix sum ตาม column ของ sim | $O(M \cdot N)$ | T₅: $\text{sim}[i][k..j-1] = \text{sim\_cumsum}[i][j] - \text{sim\_cumsum}[i][k]$ ใน $O(1)$ |
| `softmax_normalize(sim)` | $O(M \cdot N)$ | T₅: amplify max-per-row, suppress noise |

> **ทำไม Softmax ไม่ใช่ raw cosine?**
> Raw SignCLIP cosine ของ TSL กระจุกแค่ ~0.2–0.4 ทั่ว matrix — ต่างกันน้อย
> ระหว่างคู่ที่ "ดี" กับ "ไม่ดี" → T₅ จะอ่อนเกินไป
> Softmax-by-row ดันค่าสูงสุดให้ใกล้ 1 และค่าที่เหลือใกล้ 0 → DP เห็นชัดว่า
> segment ไหนเหมาะกับ cue ไหน

---

#### Sliding Window — ลด complexity

**ปัญหา:** Naive DP มี complexity $O(M \cdot N^2) = 119 \times 2780^2 \approx 9.2 \times 10^8$ ops → ช้าเกินไป

**แก้:** จำกัด search space ของแต่ละ cue ให้อยู่ภายใน window ของ segments
ที่ใกล้ midpoint ของ cue ที่สุด:

```python
W = 40                                  # --dp_window_size
for each cue_i:
    cue_mid = (cue_i.start + cue_i.end) / 2
    cand    = argsort(|seg_mids - cue_mid|)[:W]
    cand_min_i, cand_max_i = min(cand), max(cand)
    # ใน DP, จำกัด k, j ∈ [cand_min_i, cand_max_i]
```

**Complexity ใหม่:** $O(M \cdot W^2) = 119 \times 40^2 \approx 1.9 \times 10^5$
ops → **เร็วขึ้น 5000× ใน big-O**

**Numba `@njit`** บน inner DP loop อีก 10–50× → **เสร็จในไม่ถึง 1 วินาที**

---

#### Subgroup Refinement (post-DP)

หลัง DP คืน boundary ของแต่ละ cue — แต่บางครั้ง group นั้นมี **resting
segments** (ผู้แปลหยุดมือระหว่าง) ที่ไม่ควรอยู่ใน final timestamp

**Algorithm:**

1. ตัด group ออกเป็น contiguous subgroups โดย segments ที่ gap $>$ `--dp_max_gap` (= 6 s) จะตัดแยก
2. คำนวณ `cost_for_subgroup()` ของแต่ละ subgroup (ใช้สูตรเดียวกับ DP)
3. เลือก subgroup ที่ cost ต่ำสุด → ใช้ start ของ segment แรก และ end ของ segment สุดท้ายของ subgroup นั้น

> ⚠️ **Side-effect:** subgroup refinement บางครั้งดัน end ของ cue *i*
> ไปข้างหน้า → ทับ start ของ cue *i+1* → **เกิด overlap** (เป็นที่มาของ
> 86–88 % overlap ที่ Slide 1.3 จะอธิบาย)

---

#### Summary Box

```text
┌─────────────────────────────────────────────────────────────────────┐
│  Input    : 119 cues + 2780 sign segments + 119×2780 similarity     │
│  Output   : 119 (start, end) timestamps + backtrack pointers        │
│                                                                      │
│  Algorithm: 2D DP with 5-term cost function                         │
│             + sliding window (W=40)                                  │
│             + precompute (gap, sim_cumsum, softmax)                  │
│             + subgroup refinement                                    │
│                                                                      │
│  Runtime  : < 1 second (Numba JIT)                                  │
│  Output   : aligned VTT — 119 cues, mean offset ~ −0.16 s (best)    │
└─────────────────────────────────────────────────────────────────────┘
```

---

───────────────────────────────────────────────────────────────────────

### Slide 1.3 — ทำไม Overlap ถึงเกิด (และวิธีแก้)

> **ก่อน fix: 86–88 % ของ cue pairs ติดกัน overlap กัน**
> เกิดจาก design ของ DP — ไม่ใช่ bug

#### สาเหตุ 1 — DP state ไม่เก็บ end time ของ cue ก่อนหน้า

State เก็บแค่ว่า cue *i* **จบที่ segment index** *j* — ไม่ใช่ "จบที่เวลา *t*"
→ DP **ไม่รู้** ว่า start time ของ cue *i+1* จะทับ end ของ cue *i* หรือเปล่า

#### สาเหตุ 2 — Subgroup refinement (post-processing) อาจดัน end กลับ

ขั้นตอน refinement เลือก subgroup ที่ cost ต่ำสุด — บางครั้งทำให้ cue *i*
end ช้ากว่า cue *i+1* start ที่ DP assign ไว้

#### สาเหตุ 3 — Design choice ของ SEA ดั้งเดิม

SEA ออกแบบสำหรับ BSL ที่ sign density สูงและ segments ติดกันแน่น → overlap
น้อย แต่ TSL มี resting periods เยอะ → gap ระหว่าง groups ใหญ่ → overlap สูง

#### วิธีแก้ — Single-pass clamp (Step G)

```python
for i in range(len(cues) - 1):
    if cues[i].end > cues[i + 1].start:
        cues[i].end = cues[i + 1].start   # clamp
```

**ทำไมแตะแค่ end ไม่แตะ start?**

- **Start** = สิ่งที่ DP คำนวณมาอย่างระมัดระวัง — "best estimate" ว่าผู้แปล
  เริ่มท่ามือเมื่อไร → ห้ามแตะ
- **End** = ขอบหลังที่ flexible — แค่ต้องการให้ subtitle แสดงนานพอ
- **Metric หลัก** (mean offset, ±1/2/3s) วัด start เท่านั้น → ผลก่อน-หลัง
  fix เหมือนกันทุกอย่าง ยกเว้น `overlap_pct` → 0 %

> ✅ **Verified:** Mean offset / coverage ของ 7 experiments **เท่ากันเป๊ะ**
> ก่อนและหลัง overlap fix — แสดงว่า fix ปลอดภัย 100 %

---

───────────────────────────────────────────────────────────────────────

### Slide 1.4 — 7 Experiments: วิวัฒนาการและเหตุผล

> แต่ละ experiment เพิ่ม/เปลี่ยนอะไรจาก experiment ก่อนหน้า และทำไม

#### Experiment A — Baseline (timing only)

**เหตุผล:** รัน DP อิงเวลาเพียงอย่างเดียว (ไม่ใช้ embedding) เพื่อสร้าง baseline
**ผล:** Mean start shift ~2.49 s — สูงมาก แสดงว่า timing alone ไม่พอ

#### Experiment B1 — BSL + CC text

**เหตุผล:** เพิ่ม semantic embedding ด้วย BSL SignCLIP เพื่อดูว่า embedding ช่วยลด offset ได้ไหม
**ผล:** Mean ~2.49 s — ยังสูง เพราะ parameters ยังไม่ได้ tune

#### Experiment B2 — BSL + CC text + tuned params

**เหตุผล:** B1 ยังแย่ → tune bias, duration/gap penalty
**ผล:** Mean **+1.02 s**, ±1s = 74 % — ดีขึ้น แต่ยังห่าง GT

#### Experiment B_MULTI — Multilingual + CC text

**เหตุผล:** BSL train บน BSL-specific data → Multilingual น่าจะ generalize มา TSL ได้ดีกว่า
**ผล:** Mean **+0.91 s**, ±1s = 78 % — ดีกว่า BSL อย่างชัดเจน ✓ ยืนยัน Multilingual เหมาะกว่าสำหรับ TSL

#### Experiment C_MULTI ⭐ — Multilingual + Gloss text

**เหตุผล:** CC text คือคำพูด (ภาษาไทยพูด) ซึ่งต่างจากท่ามือ; Gloss text คือ sign notation ("สวัสดี ผายมือ เด็ก") → embedding ควรใกล้ sign embedding มากกว่า
**ผล:** Mean **−0.16 s**, ±1s = 73.9 %, ±3s = **100 %** — **ดีที่สุด ⭐**

#### Experiment C_MULTI_word — word-level similarity

**เหตุผล:** ทดสอบ embed ทีละคำแล้ว pool (แทน embed ทั้ง cue) เพราะ Gloss text มีหลายคำ
**ผล:** Mean +0.51 s, ±1s = 77 % — **ไม่ช่วย** เทียบ C_MULTI → sentence pooling ดีพอแล้ว

#### Experiments D_ASL, D_ASL_gloss, D_ASL_word — ASL model

**เหตุผล:** ทดสอบ ASL model (language-specific อีกตัว) เพื่อเปรียบเทียบกับ Multilingual
**ผล:** ทุก variant แย่กว่า Multilingual — ASL-specific ไม่ generalize มา TSL ดีเท่า Multilingual

---

───────────────────────────────────────────────────────────────────────

### Slide 1.5 — Full Results Table (index-based evaluation, 119/119 cues)

| Experiment | Model | Text input | Mean offset | ±1 s | ±2 s | ±3 s | Overlap → fix |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| **C_MULTI ⭐** | Multilingual | Gloss | **−0.16 s** | **73.9 %** | 95.0 % | **100 %** | 88 % → **0 %** |
| C_MULTI_word | Multilingual | Gloss (word) | +0.51 s | 77 % | 96 % | 99 % | 88 % → **0 %** |
| B_MULTI | Multilingual | CC text | +0.91 s | 78 % | 97 % | 99 % | 88 % → **0 %** |
| B2 | BSL | CC text | +1.02 s | 74 % | 96 % | 97 % | 88 % → **0 %** |
| D_ASL_word | ASL | Gloss (word) | +0.78 s | 67 % | 93 % | 96 % | 88 % → **0 %** |
| D_ASL_gloss | ASL | Gloss | +0.77 s | 64 % | 91 % | 97 % | 88 % → **0 %** |
| D_ASL | ASL | CC text | +1.25 s | 59 % | 81 % | 96 % | 89 % → **0 %** |

**สรุป pattern จากตาราง:**

```text
Multilingual > BSL > ASL              (model ranking สำหรับ TSL)
Gloss text   > CC text                (text input: sign notation ดีกว่าคำพูด)
Sentence-level pooling > word-level   (word-level ไม่ช่วย — complexity เพิ่มโดยไม่ได้ผล)
```

> 💡 **F1 @ 0.50 IoU ของ C_MULTI = 88.2 %** (Frame accuracy = 82.6 %)

---

───────────────────────────────────────────────────────────────────────

### Slide 1.6 — Evaluation Method: Text-lookup → Index-based

> **การเปลี่ยน evaluation method เป็นสิ่งสำคัญที่สุดของ Progress_04052026**
> (ผลทุกตัวใน Slide 1.5 ใช้ index-based — ต้องเข้าใจความแตกต่าง)

#### Method 1 — Text-lookup (รุ่นเดิม)

```text
For each pred_cue:
  ค้นหา GT entry ที่ text ตรงกัน (exact string match)
  ถ้าไม่เจอ → ไม่นับ (missed)

ปัญหา: ผู้ annotate แก้ข้อความใน CC_Aligned บางส่วน
        (รวมประโยค, แก้คำ) → text ไม่ตรงแม้เป็น cue เดียวกัน
ผลลัพธ์: matched เพียง 69/172 (58%) → ผลที่ได้ไม่ครอบคลุม
```

#### Method 2 — Index-based (รุ่นใหม่)

```text
pred_cues[i]  ↔  gt_cues[i]   (จับคู่ตาม index โดยตรง)

เงื่อนไข: ใช้ CC_Input (119 cues) แทน CC (172 cues)
          → pred และ GT มีจำนวน cue เท่ากัน → match ได้ 119/119 = 100%
```

| | Text-lookup | Index-based |
| --- | ---: | ---: |
| Coverage | **69/172 (58%)** | **119/119 (100%)** |
| C_MULTI mean offset | +0.49 s | **−0.16 s** |
| C_MULTI ±1 s | 80 % | 73.9 % |
| C_MULTI ±3 s | 99 % | **100 %** |

> ⚠️ **ตัวเลขเก่าที่อาจเห็นในเอกสารก่อน Progress_04052026** (เช่น "80% within ±1s")
> มาจาก text-lookup บน 69/172 cues — **ไม่ใช้อ้างอิงอีกต่อไป**
> ใช้ index-based 119/119 เป็น metric มาตรฐาน

---

═══════════════════════════════════════════════════════════════════════

## PART 2 — Task 2 สิ่งที่เพิ่มจาก Task 1

> 📊 **Slides for this part: 2.1, 2.2, 2.3**
> เป้าหมาย: แสดงว่า Task 2 ลงลึกอย่างไรจาก sentence-level → sign-gesture-level

═══════════════════════════════════════════════════════════════════════

### Slide 2.1 — Granularity: sentence → sign-gesture

```text
Timeline ─────────────────────────────────────────────►
   │
Task 1:  [── "การเปรียบเทียบและการเรียงลำดับ" ──────]   ← 1 cue = 1 sentence
         (start: 34.0s, end: 36.2s, 119 cues total)
   │
Task 2:  [ผายมือ][เด็ก][เรียน][..][..][..][..][..]      ← 1 entry = 1 sign gesture
         (852 GT entries total — sub-sentence granularity)
```

| | **Task 1** | **Task 2** |
| --- | --- | --- |
| Input | `CC_Input` (119 sentence-level cues) | `Gloss` หรือ `Gloss_Input` tier (119 sentences with tokens) |
| Output | sentence-aligned VTT | token-aligned VTT (รายท่ามือ) |
| GT | `CC_Aligned` (119 entries) | `Gloss Labeling` (852 entries) |
| Granularity | ประโยค | ท่ามือเดี่ยว |
| DP scope | global (119 cues × 2780 segments) | per-sentence (T tokens × K segments) |

---

───────────────────────────────────────────────────────────────────────

### Slide 2.2 — Per-sentence Monotonic DP

Task 2 รัน **DP ใหม่ในแต่ละประโยค** (ไม่ใช่ DP global):

```text
สำหรับแต่ละ Gloss sentence (start_s, end_s, "tok1 tok2 ... tokT"):
  ┌──────────────────────────────────────────────────────┐
  │ 1. Tokenize on whitespace      → T tokens            │
  │                                                       │
  │ 2. Restrict candidate SIGN segments to:              │
  │     mid ∈ [start_s, end_s]   (±0.5s pad if K < T)    │
  │                                  → K segments         │
  │                                                       │
  │ 3. Embed each token via SignCLIP multilingual        │
  │     text encoder                  → (T, 768)         │
  │                                                       │
  │ 4. Build (T × K) similarity matrix                   │
  │     cosine → row-softmax                              │
  │                                                       │
  │ 5. Run monotonic DP to split [start_s, end_s] into   │
  │     T contiguous segment ranges                       │
  │                                                       │
  │ 6. Emit (seg[k0].start, seg[k1].end, token)          │
  └──────────────────────────────────────────────────────┘
```

**DP recurrence:**

$$
\text{dp}[t][j] = \min_{k \in [t, j]} \bigl(
\text{dp}[t-1][k-1]
\underbrace{- \textstyle\sum_{p=k-1}^{j-1} \text{sim}[t-1][p]}_{\text{neg similarity}}
+ w_{\text{gap}} \cdot \text{gap}_{\text{total}}(k, j)
+ w_{\text{cov}} \cdot |\text{group\_dur} - \tfrac{\text{sentence\_dur}}{T}|
\bigr)
$$

| Parameter | Default | บทบาท |
| --- | ---: | --- |
| `--gap-penalty` | 2.0 | ลด group ที่มี gap |
| `--coverage-penalty` | 0.5 | บังคับ duration แต่ละ token ≈ ค่าเฉลี่ย |
| `--window-pad` | 0.5 s | ขยาย window ถ้า K < T |

**Complexity:** $O(T \cdot K^2)$ per sentence — T~7, K~30 → 119 sentences เสร็จในไม่ถึง 1 วินาที

---

───────────────────────────────────────────────────────────────────────

### Slide 2.3 — ส่วนประกอบใหม่ที่ Task 2 เพิ่ม

| ส่วน | Task 1 มีอยู่แล้ว? | Task 2 เพิ่มอะไร |
| --- | --- | --- |
| Pose, segmentation, sign embeddings | ✅ ใช้ของเดิม | reuse `04.pose`, `segmentation_output/`, `segmentation_embedding/sign_clip_multi` |
| Text embedding | sentence-level (119 × 768) | **token-level** (852 / 889 × 768) ผ่าน per-token SignCLIP query |
| Token embedding cache | — | **`.npz` cache** (`subtitle_embedding/sign_clip_multi_gloss_tokens/`) — key by `language_tag\|\|token` |
| DP scope | global, 119 × 2780 | **per-sentence**, T × K (T~7, K~30) — รัน 119 ครั้ง |
| DP cost terms | start/end align + duration + gap + similarity | **negative similarity + gap + coverage** (ไม่มี start/end anchor term) |
| Fallback | — | **uniform-split** ถ้า candidate window มี segments น้อยกว่า T → 0/119 sentences ใช้ fallback |
| Evaluation | index-based (pred[i] ↔ gt[i]) | **best-IoU pairing** (เพราะ #pred อาจ ≠ #GT) |
| Metric | offset, ±Ns coverage, frame-acc | **IoU** distribution, %IoU≥0.5/0.3, text-match |

> 🔑 **Key insight:** Task 2 ไม่ได้แทนที่ Task 1 — มันลงลึกใน *ภายใน* แต่ละ
> ประโยคที่ Task 1 ให้มา (ทางอ้อมผ่าน `Gloss_Input`'s sentence boundaries)
> → สามารถมี output ทั้ง 2 ระดับใน EAF เดียวกันได้

---

═══════════════════════════════════════════════════════════════════════

## PART 3 — Gloss vs Gloss_Input Results

> 📊 **Slides for this part: 3.1, 3.2, 3.3, 3.4, 3.5**
> เป้าหมาย: รายงานผล ablation + อธิบายว่าทำไมเลือก Gloss + caveat

═══════════════════════════════════════════════════════════════════════

### Slide 3.1 — Setup และตัวเลขสรุป

#### Setup

| สิ่งที่คงที่ทั้ง 2 รัน | ค่า |
| --- | --- |
| Sign segmentation | `segmentation_output/E4s-1_30_50/04.eaf` (SIGN 2,780) |
| Sign embeddings | `sign_clip_multi/04.npy` (2780, 768) |
| Embedding model | `multilingual` (`<en> <bfi>`) |
| DP penalties | `gap=2.0`, `coverage=0.5` |
| Window pad | 0.5 s |
| Ground truth | `Gloss Labeling` tier (852 entries) |

| สิ่งที่ต่างกัน | **Gloss** | **Gloss_Input** |
| --- | --- | --- |
| #Sentences | 119 | 119 |
| **#Tokens** | **852** | **889 (+37)** |
| Mean tokens/sentence | 7.16 | 7.47 |
| Total duration | 560.55 s | 541.25 s |

#### ผลลัพธ์ (per-prediction view)

| Metric | **Gloss** | **Gloss_Input** | Δ (Gloss − G_Input) | ผู้ชนะ |
| --- | ---: | ---: | ---: | :---: |
| #Predictions | 852 | 889 | −37 | — |
| **Mean IoU** | **0.4901** | 0.4199 | **+0.07** | Gloss |
| Median IoU | 0.4861 | 0.4150 | +0.07 | Gloss |
| **% IoU ≥ 0.5** | **48.4 %** | 38.9 % | **+9.5 pp** | Gloss |
| **% IoU ≥ 0.3** | **77.0 %** | 66.0 % | **+11.0 pp** | Gloss |
| % any temporal overlap | 97.5 % | 93.4 % | +4.2 pp | Gloss |
| % zero overlap | 2.5 % | 6.6 % | −4.2 pp | Gloss |
| Mean abs start offset | **0.188 s** | 0.212 s | −24 ms | Gloss |
| Exact text match | **65.1 %** | 10.6 % | +54.5 pp | Gloss * |
| Fallback uniform sents | 0 / 119 | 0 / 119 | — | tie |

\* text-match มี leakage component — ดู §3.3

───────────────────────────────────────────────────────────────────────

### Slide 3.2 — IoU bucket distribution

| Bucket | Gloss % | Gloss_Input % |
| --- | ---: | ---: |
| [0.0] no overlap | 2.5 % | **6.6 %** |
| (0.0, 0.1) | 3.8 % | 7.0 % |
| [0.1, 0.3) | 16.8 % | 20.4 % |
| [0.3, 0.5) | 28.6 % | 27.1 % |
| **[0.5, 0.7)** | **25.6 %** | 22.2 % |
| **[0.7, 0.9)** | **19.4 %** | 14.4 % |
| **[0.9, 1.0]** | **3.4 %** | 2.4 % |

> 📊 **Gloss ดันมวลของ predictions ไปฝั่งคุณภาพสูง** — กลุ่ม IoU ≥ 0.5
> รวม 48.4 % (vs 38.9 %) ส่วน Gloss_Input มี predictions ที่ "เกือบไม่
> overlap" (zero + (0, 0.1)) รวม 13.6 % vs 6.3 % ของ Gloss

---

───────────────────────────────────────────────────────────────────────

### Slide 3.3 — ทำไม Gloss ชนะ

#### เหตุผลที่ 1 — Token count = Degrees of freedom

```text
Gloss tier:         852 tokens  ──┐
                                  ├──► 1:1 ผูกกับ GT entries ตามธรรมชาติ
Gloss Labeling GT:  852 entries ──┘

Gloss_Input tier:   889 tokens  ──┐
                                  ├──► มี 37 tokens ส่วนเกินที่ต้องบีบใส่
Gloss Labeling GT:  852 entries ──┘     852 GT entries → ตำแหน่งผิดเพี้ยน
```

#### เหตุผลที่ 2 — Sentence window coverage

Gloss มี total annotation duration ใหญ่กว่า (560 s vs 541 s) → window
ของแต่ละประโยคครอบ GT ได้กว้างกว่า:

| | Gloss | Gloss_Input |
| --- | ---: | ---: |
| % GT covered (any overlap) | **97.77 %** | 88.97 % |
| % GT missed entirely | 2.23 % | **11.03 %** |

> ⚠️ **Gloss_Input ทิ้ง GT entries 11 %** ที่ไม่มี prediction overlap แม้
> หลัง 0.5 s padding — ปัญหาของ window ที่แคบเกินไป

#### เหตุผลที่ 3 — Mean IoU ดีกว่าในทุก sentence-length bucket

| Token bucket | Gloss mean IoU | Gloss_Input mean IoU |
| --- | ---: | ---: |
| 1–3 | **0.627** | 0.375 |
| 4–6 | **0.518** | 0.436 |
| 7–9 | **0.472** | 0.388 |
| 10–12 | **0.475** | 0.429 |
| 13+ | **0.491** | 0.484 |

> ประโยคยาว (13+ tokens) สอง tier เกือบเท่ากัน — เพราะ search space ใหญ่
> จน tokenization difference ส่งผลน้อยลง
> ประโยคสั้น (1–3) ต่างกันมากที่สุด — DP เลือกผิดได้ง่ายเมื่อ K small

---

───────────────────────────────────────────────────────────────────────

### Slide 3.4 — Caveat: Ground-truth Leakage

> ก่อนรีบเชื่อ "Gloss ชนะทุกอย่าง" — ต้องตรวจสอบว่า exact-text-match
> 65 % ไม่ใช่ data leakage

#### Token position-match (flatten + zip with GT)

| Source | #tokens | match positional กับ GT | % |
| --- | ---: | ---: | ---: |
| Gloss | 852 | 607 / 852 | **71.2 %** |
| Gloss_Input | 889 | 42 / 852 | 4.9 % |

#### ตีความ

1. **`Gloss` token list มี 71.2 % ตรงกับ GT positional** → annotator น่าจะใช้
   `Gloss` เป็นจุดตั้งต้นในการ build GT (split sentences ตาม token เดิม)
2. **`Gloss_Input` มีแค่ 4.9 %** เพราะ 37 ขอบ token ที่ต่างกัน ทำให้
   ทุก position หลังจุดต่างแรกเลื่อนตำแหน่ง
3. ดังนั้น **65 % exact-text-match ของ Gloss = structural alignment**,
   **ไม่ใช่** ความสามารถของโมเดล

#### Metric ที่ "fair" จริง

| Metric | กระทบจาก leakage? |
| --- | --- |
| Exact text match | **ใช่ — กระทบหนัก** |
| Mean IoU | กระทบเล็กน้อยทางอ้อม |
| % IoU ≥ 0.3 / 0.5 | **กระทบน้อย** |
| Mean abs start/end offset | **ไม่กระทบ** |

> ✅ **ใช้ IoU + offset เป็น metric หลัก** ในการเปรียบเทียบ
> ❌ **อย่ารายงาน 65 % text-match** เป็น "model accuracy" โดยไม่ disclose

---

───────────────────────────────────────────────────────────────────────

### Slide 3.5 — Recommendation

| สิ่ง | คำแนะนำ |
| --- | --- |
| **Default tier ของ Task 2** | **เปลี่ยนเป็น `--tier Gloss`** (ปัจจุบัน default ยังเป็น Gloss_Input เพื่อ backward-compat) |
| ถ้าจะใช้ Gloss_Input | ปรับ tokenization ให้สอดคล้องกับ GT (ลด 37 token ส่วนเกิน) + ขยาย window pad |
| รายงาน metric ภายนอก | ใช้ **% IoU ≥ 0.3** หรือ **mean abs start offset** เป็นหลัก |
| paper / report | ระบุชัดเจน: "Gloss tokens were used as substrate for GT construction" |

---

═══════════════════════════════════════════════════════════════════════

## Closing Slide — Recap (Part 0–3 wrap-up)

> 📊 **Slide สุดท้ายของ presentation — รวบยอด 4 parts**

1. **SEA cross-lingual transfer ใช้ได้กับ TSL** โดยไม่ต้อง retrain — pipeline S→E→A ทำงาน end-to-end

2. **Pipeline Task 1 + Task 2 ทำงานครบ** บน video `04`
   - Task 1: 7 experiments, **C_MULTI ⭐** ดีสุด (mean offset −0.16 s, ±3s = 100 %)
   - Task 2: prototype 0 fallback ทั้ง 119 sentences, Mean IoU = 0.4901

3. **Key findings:**
   - Multilingual > BSL > ASL สำหรับ TSL
   - Gloss text > CC text เป็น embedding input
   - Overlap คือ design quirk ของ DP → แก้ด้วย single-pass clamp (ปลอดภัย 100%)
   - Word-level similarity ไม่ช่วยสำหรับ TSL

4. **Ablation: `Gloss` ชนะ `Gloss_Input` ทุก IoU metric** (+9.5 pp ที่ threshold 0.5)
   — แต่ text-match advantage มี leakage component
   → **Recommendation: ใช้ `--tier Gloss` เป็น default**

═══════════════════════════════════════════════════════════════════════

## Appendix — What's Next (progress doc only)

> ⚠️ **ส่วนนี้ไม่ใช่ slides** — เป็นข้อเสนอ forward-looking สำหรับ
> progress update รอบหน้า (Progress_19052026.md หรือใกล้เคียง)
>
> Slides เฉพาะ PART 1–3 + Closing Slide เท่านั้น

### 4.1 ขยาย Dataset — รันบนวิดีโออื่น

**ปัญหา:** ทุกตัวเลขปัจจุบันมาจาก video `04` คลิปเดียว (11.07 นาที,
"การเปรียบเทียบและการเรียงลำดับ") → ผลอาจ **over-fit ต่อ content
ของวิดีโอนี้**

**Action items:**

- หา/annotate วิดีโอ TSL อีก 2–3 คลิป ที่มี `CC_Input`, `CC_Aligned`,
  `Gloss`, `Gloss Labeling` ครบ
- รัน pipeline ที่ใส่ multi-video loop ตาม
  [README §"Adapting for Multiple Videos"](README.md#adapting-for-multiple-videos)
- รายงาน metric แบบ **mean across videos** + 95 % CI

**ทำไมสำคัญ:** SEA paper รายงานบน 20,000+ videos — เราอ้างผลบน 1 video
ไม่ได้ตอน publish

---

### 4.2 Cross-validation ของ Hyperparameters

**ปัญหา:** Penalty weights ปัจจุบัน (`gap=8`, `sim=6`, `bias_start=1.3`)
เลือกด้วย **manual tuning บน video `04`** → อาจ overfit

**Action items:**

- สร้าง grid: `dp_gap ∈ {4, 6, 8, 10}`, `sim_weight ∈ {4, 6, 8}`,
  `bias_start ∈ {1.0, 1.3, 1.8, 2.0}`
- รันบน train video + เลือก best บน validation video
- รายงานทั้ง best-on-val และ best-on-test (ตรวจ overfitting)

---

### 4.3 Language-tag Ablation สำหรับ Task 2

**ปัญหา:** ทุก Task 2 run ใช้ `<en> <bfi>` (English BSL prompt) แม้
input เป็น Thai gloss → suboptimal

**Action items (ลองตามลำดับ):**

1. `<en> <bfi>` (current) — baseline
2. `<en> <ase>` (English ASL)
3. `<th>` ถ้า SignCLIP รองรับ (ดู `runs/retri_v1_1/baseline_temporal` config)
4. ไม่มี language tag เลย — เป็น control

วัด mean IoU + %IoU≥0.5 ของแต่ละ tag

---

### 4.4 Embedding Model Ablation สำหรับ Task 2

**ปัญหา:** Task 2 ใช้ `multilingual` only แต่ Task 1 พบว่า `multi+Gloss`
ดีกว่า `BSL` หรือ `ASL` — Task 2 ควร ablate เหมือนกัน

**Action items:**

- รัน Task 2 ด้วย `--model-name bsl` และ `--model-name asl`
- เปรียบเทียบ mean IoU, %IoU≥0.5, fallback count
- ตาราง 3×2 (model × tier) → 6 cells

---

### 4.5 ลด %GT-missed ของ Gloss_Input

**ปัญหา:** Gloss_Input ทิ้ง GT entries 11 % → window pad 0.5 s ไม่พอ

**Action items:**

- ทดลอง `--window-pad` ∈ {0.5, 1.0, 1.5, 2.0, 3.0} s
- วัด trade-off ระหว่าง:
  - % GT covered (อยากให้สูง)
  - Mean IoU (อาจลดเพราะ window ใหญ่ → wrong picks)
  - Fallback count (ไม่ควรเพิ่ม)

ถ้า pad ใหญ่ไม่พอ → ลอง **sliding-window strategy** ที่ขยาย adaptive
ตาม sentence_dur / T

---

### 4.6 End-time Accuracy (Task 1)

**ปัญหา:** Mean end offset ของ C_MULTI = **+0.91 s** (ก่อน overlap fix)
→ subtitles ค้างนานกว่าจริง ~1 วินาที

**Action items (เลือกอย่างใดอย่างหนึ่ง):**

1. เพิ่ม end-alignment term ใน DP cost function (currently weight = 1)
2. Post-processing แยก: หา last sign segment per cue ที่ similarity ยังสูง
3. Train end-time regressor บน CC_Aligned

> 💡 หมายเหตุ: overlap fix แก้ end เฉพาะตอน next start ใกล้ → ไม่ได้
> แก้ structural end-bias

---

### 4.7 Confidence Calibration (Score ↔ IoU)

**ปัญหา:** DP scores (sum of softmax similarity) ตอนนี้ **ไม่ correlate
กับ IoU** อย่างมีประโยชน์ → ไม่สามารถใช้เป็น "confidence" ในการ filter
output

**Action items:**

- คำนวณ correlation (Pearson, Spearman) ของ score vs IoU
- ลอง alternative confidence signals:
  - Row-max softmax (most-likely-segment confidence)
  - Entropy ของ row softmax
  - Margin (max − 2nd max)
- เลือกตัวที่ correlate ดีที่สุด → ใช้สำหรับ filtering ใน production

---

### 4.8 Human Qualitative Evaluation

**ปัญหา:** IoU ≥ 0.5 ไม่ได้แปลว่า "subtitle ดู right" — อาจ start ตรง
แต่ end เพี้ยน หรือคุณภาพ subjective ที่ metric จับไม่ได้

**Action items:**

- ให้ผู้รู้ภาษามือ 2–3 คน ดู `04_no_overlap.vtt` ทับวิดีโอ
- rate 1–5 ใน 50 cues สุ่ม
- คำนวณ inter-rater agreement (Cohen's κ)
- เทียบกับ IoU — ดูว่า IoU 0.5 ↔ rating กี่คะแนน

---

### 4.9 Tokenization Ablation (Task 2)

**ปัญหา:** Ablation ปัจจุบันชี้ว่า tokenization สำคัญที่สุด — แต่ยัง
ไม่ได้ทดลอง tokenization strategies แบบอื่น

**Action items:**

- ทดลอง tokenizers ใน Gloss_Input:
  1. whitespace (current)
  2. PyThaiNLP word-segment
  3. Character-level
  4. Hybrid: คงคำในวงเล็บ `(...)` เป็น token เดียว
- รัน Task 2 ด้วยแต่ละ tokenization → mean IoU comparison

---

### 4.10 Pipeline / Engineering

| Item | Priority |
| --- | --- |
| Refactor hardcoded paths → CLI args (`evaluate_all.py`, `add_*_to_eaf.py`) | High — ต้องการสำหรับ multi-video |
| Add `pytest` smoke tests สำหรับแต่ละ step | Medium — ป้องกัน regression |
| Pin `requirements.txt` ออกมาเป็น lockfile | Medium — reproducibility |
| Commit precomputed `04.pose` กับ `.npy` ผ่าน Git LFS (ตอนนี้ใส่ใน git ปกติ) | Medium — repo size |
| สร้าง `Makefile` / `tasks.json` รวบ commands ยาวๆ | Low — quality of life |

---

<!-- moved to closing slide before appendix -->
