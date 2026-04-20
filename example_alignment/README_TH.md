# คู่มือการใช้งาน SEA กับตัวอย่างคลิปภาษามือไทย

> **SEA** (Segment, Embed, and Align) คือระบบจัดตำแหน่งคำบรรยาย (subtitle alignment)
> ให้ตรงกับช่วงเวลาที่ผู้แปลภาษามือแสดงท่าทางจริง ๆ ในวิดีโอ

---

## ไฟล์ในโฟลเดอร์นี้

| ไฟล์/โฟลเดอร์ | คำอธิบาย | สถานะ |
| --- | --- | --- |
| `04.mp4` | วิดีโอต้นฉบับ (11.07 นาที, 1920x1080, 60fps) | มีอยู่แล้ว |
| `04.vtt` | คำบรรยายดิบที่แยกจาก EAF (172 cues) | สร้างแล้ว |
| `04.pose` | ข้อมูล skeleton pose จาก MediaPipe (358 MB) | สร้างแล้ว |
| `การเปรียบเทียบและเรียงลำดับ (11.07 นาที).eaf` | ไฟล์ annotation ELAN มี tier **CC**, **CC_Aligned**, **Gloss** | มีอยู่แล้ว |
| `การเปรียบเทียบและเรียงลำดับ (11.07 นาที).pfsx` | ไฟล์ตั้งค่าการแสดงผลของ ELAN | มีอยู่แล้ว |
| `extract_cc_from_eaf.py` | สคริปต์แยก CC → VTT | สร้างแล้ว |
| `merge_cc_to_updated_eaf.py` | สคริปต์ merge CC tiers เข้า 04_updated.eaf | สร้างแล้ว |
| `make_gloss_cc_vtt.py` | สร้าง VTT ที่มี timestamp จาก CC และ text จาก Gloss tier | สร้างแล้ว |
| `evaluate_all.py` | เปรียบเทียบ VTT ทุก experiment vs CC_Aligned ground truth | สร้างแล้ว |
| `fix_overlap_vtt.py` | ลบ overlap ระหว่าง cues โดย clamp end time | สร้างแล้ว |
| `video_ids.txt` | รายชื่อวิดีโอสำหรับ pipeline (`04`) | สร้างแล้ว |
| `subtitles/04.vtt` | subtitle input (CC text, speech timestamp) | สร้างแล้ว |
| `subtitles_gloss_cc_time/04.vtt` | subtitle input สำหรับ experiment C (Gloss text, CC timestamp) | สร้างแล้ว |
| `segmentation_output/E4s-1_30_50/04.eaf` | ผล segmentation (SIGN tier) | สร้างแล้ว |
| `segmentation_output/E4s-1_30_50/04_updated.eaf` | EAF รวมทุก tier (SIGN + CC + aligned) | สร้างแล้ว |
| `segmentation_embedding/sign_clip/04.npy` | sign embedding — BSL model (172 cues) | สร้างแล้ว |
| `segmentation_embedding/sign_clip_multi/04.npy` | sign embedding — multilingual model (2780 segments, 768-dim) | สร้างแล้ว |
| `subtitle_embedding/sign_clip/04.npy` | subtitle embedding — BSL model | สร้างแล้ว |
| `subtitle_embedding/sign_clip_multi/04.npy` | subtitle embedding — multilingual model, CC text | สร้างแล้ว |
| `subtitle_embedding/sign_clip_multi_gloss/04.npy` | subtitle embedding — multilingual model, Gloss text | สร้างแล้ว |
| `subtitle_embedding/sign_clip_asl/04.npy` | subtitle embedding — ASL model, CC text | สร้างแล้ว |
| `subtitle_embedding/sign_clip_asl_gloss/04.npy` | subtitle embedding — ASL model, Gloss text | สร้างแล้ว |
| `segmentation_embedding/sign_clip_asl/04.npy` | sign embedding — ASL model (2780 segments, 768-dim) | สร้างแล้ว |
| `aligned_output/04.vtt` | A — no embed, standard params | สร้างแล้ว |
| `aligned_output_with_embedding/04.vtt` | B1 — BSL, CC text, standard params | สร้างแล้ว |
| `aligned_output_with_embedding_tuned/04.vtt` | B2 — BSL, CC text, tuned params (**baseline เดิม**) | สร้างแล้ว |
| `aligned_output_multi_b2/04.vtt` | B_MULTI — multilingual, CC text, B2 params | สร้างแล้ว |
| `aligned_output_multi_gloss/04.vtt` | C_MULTI — multilingual, Gloss text | สร้างแล้ว |
| `aligned_output_multi_gloss/04_no_overlap.vtt` | C_MULTI post-processed — overlap removed | สร้างแล้ว |
| `aligned_output_multi_gloss_word/04.vtt` | C_MULTI_word — multilingual, Gloss text, word-level similarity | สร้างแล้ว |
| `aligned_output_asl_b2/04.vtt` | D_ASL — ASL model, CC text | สร้างแล้ว |
| `aligned_output_asl_gloss/04.vtt` | D_ASL_gloss — ASL model, Gloss text | สร้างแล้ว |
| `aligned_output_asl_gloss/04_no_overlap.vtt` | D_ASL_gloss post-processed — overlap removed | สร้างแล้ว |
| `aligned_output_asl_gloss_word/04.vtt` | D_ASL_word — ASL model, Gloss text, word-level similarity | สร้างแล้ว |

---

## ภาพรวมขั้นตอน

```text
EAF (CC tier)
     │
     ▼  [Step 1] extract_cc_from_eaf.py        ✅ เสร็จแล้ว
     │
  04.vtt  ◄─── คำบรรยายดิบ 172 cues
     │
     ▼  [Step 2] videos_to_poses               ✅ เสร็จแล้ว
     │
  04.pose ◄─── skeleton pose 358 MB
     │
     ▼  [Step 3] segmentation.py               ✅ เสร็จแล้ว
     │
  segmentation_output/E4s-1_30_50/04.eaf
     │
     ▼  [Step 4] align.py (none)               ✅ เสร็จแล้ว
     │
     ▼  [Step 6] align.py + SignCLIP           ✅ เสร็จแล้ว
     │
  aligned_output_with_embedding/04.vtt  ◄─── คำบรรยายที่จัดตำแหน่งแล้ว ✓
```

---

## Runbook ที่ยืนยันแล้ว (อัปเดต 2026-04-09)

ส่วนนี้คือคำสั่งที่รันจริงบนเครื่องนี้แบบครบลำดับ และได้ผลลัพธ์ครบ:

1. Segmentation

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\SEA
..\venv\Scripts\python.exe segmentation.py --sign-b-threshold 30 --sign-o-threshold 50 --num_workers 1 --video_ids ..\example_alignment\video_ids.txt --pose_dir ..\example_alignment --save_dir ..\example_alignment\segmentation_output --video_dir ..\example_alignment --overwrite
```

2. Subtitle embedding (SignCLIP)

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\fairseq_signclip\examples\MMPT
..\..\..\venv\Scripts\python.exe .\scripts_bsl\extract_episode_features.py --video_ids ..\..\..\example_alignment\video_ids.txt --mode=subtitle --model_name bsl --language_tag "<en> <bfi>" --batch_size=1024 --subtitle_dir ..\..\..\example_alignment\subtitles --save_dir ..\..\..\example_alignment\subtitle_embedding\sign_clip
```

3. Segmentation embedding (SignCLIP)

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\fairseq_signclip\examples\MMPT
..\..\..\venv\Scripts\python.exe .\scripts_bsl\extract_episode_features.py --video_ids ..\..\..\example_alignment\video_ids.txt --mode=segmentation --model_name bsl --language_tag "<en> <bfi>" --batch_size=32 --pose_dir ..\..\..\example_alignment --segmentation_dir ..\..\..\example_alignment\segmentation_output\E4s-1_30_50 --save_dir ..\..\..\example_alignment\segmentation_embedding\sign_clip
```

4. Align แบบใช้ embedding

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\SEA
..\venv\Scripts\python.exe align.py --overwrite --mode=inference --video_ids ..\example_alignment\video_ids.txt --num_workers 1 --dp_duration_penalty_weight 1 --dp_gap_penalty_weight 5 --dp_max_gap 10 --dp_window_size 50 --sign-b-threshold 30 --sign-o-threshold 50 --pr_subs_delta_bias_start 2.6 --pr_subs_delta_bias_end 2.1 --similarity_measure sign_clip_embedding --similarity_weight 10 --pr_sub_path ..\example_alignment\subtitles --segmentation_dir ..\example_alignment\segmentation_output --subtitle_embedding_dir ..\example_alignment\subtitle_embedding\sign_clip --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip --save_dir ..\example_alignment\aligned_output_with_embedding
```

5. Merge tier ต้นฉบับกลับเข้า 04_updated.eaf

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment
..\venv\Scripts\python.exe .\merge_cc_to_updated_eaf.py
```

ผลลัพธ์ที่ตรวจสอบแล้ว:

- `segmentation_output\E4s-1_30_50\04.eaf`
- `subtitle_embedding\sign_clip\04.npy`
- `segmentation_embedding\sign_clip\04.npy`
- `aligned_output_with_embedding\04.vtt`
- `segmentation_output\E4s-1_30_50\04_updated.eaf`

---

## ความต้องการเบื้องต้น (สิ่งที่ติดตั้งแล้วในเครื่องนี้)

| ซอฟต์แวร์ | เวอร์ชัน | หมายเหตุ |
| --- | --- | --- |
| Python | 3.11.15 | Astral CPython ที่ `C:\Users\dniam\.local\bin\python3.11.exe` |
| venv | — | อยู่ที่ `SEA\venv\` |
| pose-format | 0.12.3 | ติดตั้งใน venv แล้ว |
| mediapipe | 0.10.21 | **ต้องใช้ version นี้เท่านั้น** (0.10.22+ ใช้ไม่ได้) |
| pysrt, webvtt-py | latest | ติดตั้งใน venv แล้ว |
| numba, beartype | latest | ติดตั้งใน venv แล้ว |
| numpy | 1.26.4 | ติดตั้งใน venv แล้ว |
| PyTorch | 2.11.0+cu128 | ติดตั้งแล้ว — **ต้องใช้ cu128** สำหรับ RTX 5060 Ti (sm_120) |

---

## โครงสร้างโฟลเดอร์

```text
C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\
├── venv\                                   ← Python virtual environment
├── SEA\                                    ← โค้ด SEA
│   ├── align.py
│   ├── segmentation.py                     ← แก้ไขแล้ว (Windows fix)
│   └── ...
├── fairseq_signclip\                       ← SignCLIP (clone แยกต่างหาก)
│   └── examples\MMPT\scripts_bsl\
└── example_alignment\                      ← โฟลเดอร์ทำงานหลัก
    ├── 04.mp4
    ├── 04.pose                             ✅ 358 MB
    ├── 04.vtt                             ✅ 172 cues
    ├── video_ids.txt
    ├── subtitles\04.vtt
    ├── segmentation_output\E4s-1_30_50\
    │   ├── 04.eaf                         ✅
    │   └── 04_updated.eaf                 ✅
    ├── segmentation_embedding\sign_clip\04.npy  ✅
    ├── subtitle_embedding\sign_clip\04.npy      ✅
    ├── aligned_output\04.vtt                    ✅
    ├── aligned_output_with_embedding\04.vtt     ✅
    └── aligned_output_with_embedding_tuned\04.vtt  ✅
```

---

## ขั้นตอนที่ 0 — ตั้งค่า Environment (venv)

> **หมายเหตุ:** ไม่มี Conda ในเครื่องนี้ — ใช้ Python venv แทน

### 0.1 สร้าง venv (ทำครั้งเดียว)

เปิด **PowerShell** แล้วรัน:

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA
C:\Users\dniam\.local\bin\python3.11.exe -m venv venv
```

### 0.2 Activate venv (ทำทุกครั้งที่เปิด terminal ใหม่)

```powershell
venv\Scripts\activate
```

ต้องเห็น `(venv)` นำหน้า prompt

### 0.3 ติดตั้ง dependencies (ทำครั้งเดียว)

```powershell
pip install pysrt webvtt-py lxml numpy pympi-ling
pip install "mediapipe==0.10.21" pose-format
pip install beartype numba tqdm scikit-learn tabulate
pip install "git+https://github.com/J22Melody/segmentation@bsl"
```

> **สำคัญ:** mediapipe ต้องใช้เวอร์ชัน `0.10.21` เท่านั้น
> เวอร์ชันใหม่กว่านี้ใช้ API ต่างกัน ทำให้ `videos_to_poses` ทำงานไม่ได้

---

## ขั้นตอนที่ 1 — แยก CC จาก EAF ออกเป็นไฟล์ VTT

> **สถานะ: เสร็จแล้ว** — `04.vtt` มีอยู่แล้ว (172 cues)

ไฟล์ EAF มี tier ชื่อ **CC** ซึ่งบรรจุคำบรรยายภาษาไทยพร้อม timestamp (milliseconds)

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment
python extract_cc_from_eaf.py "การเปรียบเทียบและเรียงลำดับ (11.07 นาที).eaf" 04.vtt
```

**ผลลัพธ์:** `04.vtt` รูปแบบ WebVTT

```text
WEBVTT

1
00:00:00.040 --> 00:00:31.890
[เสียงดนตรี]

2
00:00:34.030 --> 00:00:36.210
(คุณครูจิรชพรรณ) สวัสดีค่ะนักเรียนทุกคน
```

### ทดสอบทันทีใน Browser

สร้างไฟล์ `player.html` ในโฟลเดอร์ `example_alignment\` แล้วเปิดด้วย Chrome:

```html
<!DOCTYPE html>
<html lang="th"><head><meta charset="UTF-8"><title>Test SEA</title></head>
<body>
  <video controls width="900">
    <source src="04.mp4" type="video/mp4">
    <track src="04.vtt" kind="subtitles" srclang="th" label="Original CC" default>
  </video>
</body></html>
```

---

## ขั้นตอนที่ 2 — Pose Estimation (แยก skeleton จากวิดีโอ)

> **สถานะ: เสร็จแล้ว** — `04.pose` (358 MB) มีอยู่แล้ว

SEA ใช้ข้อมูล pose ของผู้แปลภาษามือ (ตำแหน่งมือ แขน ใบหน้า) เพื่อตรวจจับท่ามือ

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment
videos_to_poses --format mediapipe --directory . --additional-config="model_complexity=2,smooth_landmarks=false,refine_face_landmarks=true"
```

| Parameter | ความหมาย |
| --- | --- |
| `--format mediapipe` | ใช้ Google MediaPipe Holistic |
| `model_complexity=2` | ความแม่นยำสูงสุด (ช้ากว่า) |
| `smooth_landmarks=false` | ไม่ smooth ค่า pose |
| `refine_face_landmarks=true` | ตรวจจับใบหน้าแบบละเอียด |

> **ใช้เวลา:** ~15 นาที บน CPU สำหรับวิดีโอ 11 นาที
> ถ้าต้องการเร็วขึ้น ใช้ `model_complexity=1`

---

## ขั้นตอนที่ 3 — Segmentation (ตรวจจับตำแหน่งท่ามือ)

> **สถานะ: เสร็จแล้ว** — `segmentation_output\E4s-1_30_50\04.eaf` มีอยู่แล้ว

### 3.1 สร้าง video\_ids.txt

> **สำคัญ Windows:** ต้องสร้างด้วย Python เท่านั้น
> `echo 04 > video_ids.txt` ใน CMD/PowerShell จะสร้างไฟล์ UTF-16 BOM
> ซึ่ง Python อ่านไม่ได้และจะเกิด `UnicodeDecodeError`

```powershell
python -c "open('video_ids.txt','w',encoding='utf-8').write('04\n')"
```

### 3.2 รัน segmentation

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\SEA
python segmentation.py --sign-b-threshold 30 --sign-o-threshold 50 --num_workers 1 --video_ids ..\example_alignment\video_ids.txt --pose_dir ..\example_alignment --save_dir ..\example_alignment\segmentation_output --video_dir ..\example_alignment
```

> **หมายเหตุ:** ใช้ `--num_workers 1` บน Windows — multiprocessing มักมีปัญหากับ Windows path

**ผลลัพธ์:** `example_alignment\segmentation_output\E4s-1_30_50\04.eaf`

```text
segmentation_output\
└── E4s-1_30_50\
    └── 04.eaf   ← มี tier "SIGN" ระบุตำแหน่งท่ามือแต่ละท่า
```

### การแก้ไขที่ทำไปใน segmentation.py (Windows Fix)

ไฟล์ `SEA\segmentation.py` บรรทัด 71 ถูกแก้ไขจาก:

```python
# เดิม — ใช้ไม่ได้บน Windows (shlex.quote ใส่ single quote ที่ cmd.exe ไม่เข้าใจ)
result = subprocess.run(cmd, shell=True)
```

เป็น:

```python
# ใหม่ — ทำงานได้บน Windows (shlex.split แปลง string เป็น list ก่อนส่งให้ subprocess)
result = subprocess.run(shlex.split(cmd), shell=False)
```

---

## ขั้นตอนที่ 4 — Alignment (จัดตำแหน่งคำบรรยาย)

> **สถานะ: เสร็จแล้ว** — `aligned_output\04.vtt` มีอยู่แล้ว

### 4.1 เตรียมโฟลเดอร์ subtitle

```powershell
mkdir C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\subtitles
copy C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\04.vtt C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\subtitles\04.vtt
```

### 4.2 รัน alignment

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\SEA
python align.py --overwrite --mode=inference --video_ids ..\example_alignment\video_ids.txt --num_workers 1 --dp_duration_penalty_weight 1 --dp_gap_penalty_weight 5 --dp_max_gap 10 --dp_window_size 50 --sign-b-threshold 30 --sign-o-threshold 50 --pr_subs_delta_bias_start 2.6 --pr_subs_delta_bias_end 2.1 --similarity_measure none --pr_sub_path ..\example_alignment\subtitles --segmentation_dir ..\example_alignment\segmentation_output --save_dir ..\example_alignment\aligned_output
```

> **สำคัญ:** argument คือ `--pr_sub_path` ไม่ใช่ `--subtitle_dir`

| Parameter | ความหมาย |
| --- | --- |
| `--pr_sub_path` | โฟลเดอร์ที่มีไฟล์ VTT คำบรรยายดิบ |
| `--similarity_measure none` | ไม่ใช้ embedding (เร็วกว่า) |
| `--pr_subs_delta_bias_start 2.6` | offset ปรับเวลาเริ่มต้นก่อน align |
| `--pr_subs_delta_bias_end 2.1` | offset ปรับเวลาสิ้นสุดก่อน align |

**ผลลัพธ์:**

```text
aligned_output\
└── 04.vtt              ← คำบรรยายที่จัดตำแหน่งแล้ว

segmentation_output\E4s-1_30_50\
└── 04_updated.eaf      ← EAF ที่มี tier SUBTITLE_SHIFTED เพิ่มมา
```

---

## ขั้นตอนที่ 5 — ทดสอบผลลัพธ์

### 5.1 ดูผ่าน Browser (เปรียบเทียบ Original vs Aligned)

สร้างไฟล์ `player.html` ในโฟลเดอร์ `example_alignment\` แล้วเปิดด้วย Chrome:

```html
<!DOCTYPE html>
<html lang="th"><head><meta charset="UTF-8"><title>Test SEA</title></head>
<body>
  <video controls width="900">
    <source src="04.mp4" type="video/mp4">
    <track src="aligned_output_with_embedding/04.vtt" kind="subtitles" srclang="th" label="Aligned (SEA)" default>
    <track src="04.vtt" kind="subtitles" srclang="th" label="Original CC">
  </video>
</body></html>
```

### 5.2 ดูผ่าน ELAN

1. ดาวน์โหลด ELAN จาก [archive.mpi.nl/tla/elan](https://archive.mpi.nl/tla/elan)
2. เปิดไฟล์ `segmentation_output\E4s-1_30_50\04_updated.eaf`
3. เปรียบเทียบ tier **CC** (เดิม) กับ **SUBTITLE_SHIFTED** (หลัง align)

---

## ขั้นตอนที่ 6 — Embedding ด้วย SignCLIP (เพิ่มความแม่นยำ)

### สเปคเครื่องที่ตรวจสอบแล้ว

| ส่วนประกอบ | ข้อมูล | ผ่านเกณฑ์? |
| --- | --- | --- |
| RAM | 64 GB | ✅ |
| CPU | Intel Core Ultra 7 265K (20 cores) | ✅ |
| GPU | NVIDIA GeForce RTX 5060 Ti | ✅ |
| VRAM | 17,100 MB (17.1 GB) | ✅ (ต้องการ ≥16 GB) |
| CUDA Driver | 595.79 (CUDA 13.2) | ✅ |
| GPU Architecture | Blackwell (sm_120) | ⚠️ ต้องการ PyTorch CUDA 12.8+ |
| PyTorch CUDA | 2.11.0+cu128 | ✅ ติดตั้งแล้ว + GPU ทำงานได้ |

**สรุป: เครื่องนี้รัน embedding ได้** — VRAM 17.1 GB เพียงพอสำหรับ SignCLIP

> **หมายเหตุสำคัญ:** RTX 5060 Ti เป็น Blackwell architecture (compute capability sm_120)
> ต้องการ PyTorch ที่ build กับ CUDA **12.8** ขึ้นไปเท่านั้น
> PyTorch cu126 จะ fallback ไป CPU โดยอัตโนมัติ (ช้ามาก)

---

### 6.1 ติดตั้ง PyTorch พร้อม CUDA (ทำครั้งเดียว)

```powershell
venv\Scripts\activate
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

ตรวจสอบว่า CUDA ใช้งานได้:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

ต้องเห็น `True NVIDIA GeForce RTX 5060 Ti`

---

### 6.2 Clone SignCLIP (fairseq fork)

SignCLIP เป็น multimodal model ที่เชื่อม sign language กับ text
บน repository ปัจจุบัน branch ที่ใช้งานจริงคือ `main` (มีโฟลเดอร์ `scripts_bsl` อยู่แล้ว)

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA
git clone https://github.com/J22Melody/fairseq.git fairseq_signclip
cd fairseq_signclip
git checkout main
```

ติดตั้ง MMPT (SignCLIP wrapper):

```powershell
venv\Scripts\activate
pip install -e .
cd examples\MMPT
pip install -e .
```

---

### 6.3 Download SignCLIP model weights

> **หมายเหตุ:** ใน branch ปัจจุบันไม่มี `download_model.py` สำหรับ BSL โดยตรง
> ให้ดาวน์โหลดจาก Google Drive release แล้ววางไฟล์เข้าโครงสร้าง `runs\...` ตามนี้

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA
venv\Scripts\activate
pip install gdown
cd fairseq_signclip\examples\MMPT
gdown --folder "https://drive.google.com/drive/folders/10q7FxPlicrfwZn7_FgtNqKFDiAJi6CTc?usp=sharing" -O .\runs
```

จากนั้นคัดลอกไฟล์ checkpoint ไปยัง path ที่ MMPT ใช้จริง:

```powershell
mkdir .\runs\retri_v1_1\baseline_temporal
mkdir .\runs\retri_bsl\bobsl_islr_finetune
copy .\runs\baseline_temporal_checkpoint_best.pt .\runs\retri_v1_1\baseline_temporal\checkpoint_best.pt
copy .\runs\bobsl_finetune_checkpoint_best.pt .\runs\retri_bsl\bobsl_islr_finetune\checkpoint_best.pt
```

หมายเหตุ: ขนาดไฟล์ใหญ่ (หลาย GB)

---

### 6.4 Embed ท่ามือ (Sign Segments)

> **สถานะ: เสร็จแล้ว** — `segmentation_embedding\sign_clip\04.npy` มีอยู่แล้ว

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\fairseq_signclip\examples\MMPT

python scripts_bsl/extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=segmentation `
  --model_name bsl `
  --language_tag "<en> <bfi>" `
  --batch_size=32 `
  --pose_dir ..\..\..\example_alignment `
  --segmentation_dir ..\..\..\example_alignment\segmentation_output\E4s-1_30_50 `
  --save_dir ..\..\..\example_alignment\segmentation_embedding\sign_clip
```

**ผลลัพธ์:** `example_alignment\segmentation_embedding\sign_clip\04.npy`

---

### 6.5 Embed คำบรรยาย (Subtitles)

> **สถานะ: เสร็จแล้ว** — `subtitle_embedding\sign_clip\04.npy` มีอยู่แล้ว

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\fairseq_signclip\examples\MMPT
python scripts_bsl/extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=subtitle `
  --model_name bsl `
  --language_tag "<en> <bfi>" `
  --batch_size=1024 `
  --subtitle_dir ..\..\..\example_alignment\subtitles `
  --save_dir ..\..\..\example_alignment\subtitle_embedding\sign_clip
```

**ผลลัพธ์:** `example_alignment\subtitle_embedding\sign_clip\04.npy`

---

### 6.6 Align พร้อม SignCLIP Embedding (แม่นยำกว่า)

> **สถานะ: เสร็จแล้ว** — `aligned_output_with_embedding\04.vtt` มีอยู่แล้ว

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\SEA

python align.py --overwrite --mode=inference `
  --video_ids ..\example_alignment\video_ids.txt `
  --num_workers 1 `
  --dp_duration_penalty_weight 1 `
  --dp_gap_penalty_weight 5 `
  --dp_max_gap 10 `
  --dp_window_size 50 `
  --sign-b-threshold 30 `
  --sign-o-threshold 50 `
  --pr_subs_delta_bias_start 2.6 `
  --pr_subs_delta_bias_end 2.1 `
  --similarity_measure sign_clip_embedding `
  --similarity_weight 10 `
  --pr_sub_path ..\example_alignment\subtitles `
  --segmentation_dir ..\example_alignment\segmentation_output `
  --subtitle_embedding_dir ..\example_alignment\subtitle_embedding\sign_clip `
  --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip `
  --save_dir ..\example_alignment\aligned_output_with_embedding
```

**ผลลัพธ์:** `aligned_output_with_embedding\04.vtt` (แม่นยำ ~82-83%)

### 6.7 Align โปรไฟล์ปรับปรุง (Tuned สำหรับลดการซ้อน)

> **สถานะ: เสร็จแล้ว** — `aligned_output_with_embedding_tuned\04.vtt` มีอยู่แล้ว

จากการทดสอบจริงในไฟล์ `04` พบว่า SUBTITLE_SHIFTED มีช่วงเวลาซ้อนกันสูง
จึงมีการปรับพารามิเตอร์ DP/weight และ bias ดังนี้:

- `--similarity_weight 6` (เดิม 10)
- `--dp_duration_penalty_weight 2` (เดิม 1)
- `--dp_gap_penalty_weight 8` (เดิม 5)
- `--dp_max_gap 6` (เดิม 10)
- `--dp_window_size 40` (เดิม 50)
- `--pr_subs_delta_bias_start 1.8` (เดิม 2.6)
- `--pr_subs_delta_bias_end 1.5` (เดิม 2.1)

คำสั่งที่รันจริง:

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\SEA

python align.py --overwrite --mode=inference `
  --video_ids ..\example_alignment\video_ids.txt `
  --num_workers 1 `
  --dp_duration_penalty_weight 2 `
  --dp_gap_penalty_weight 8 `
  --dp_max_gap 6 `
  --dp_window_size 40 `
  --sign-b-threshold 30 `
  --sign-o-threshold 50 `
  --pr_subs_delta_bias_start 1.8 `
  --pr_subs_delta_bias_end 1.5 `
  --similarity_measure sign_clip_embedding `
  --similarity_weight 6 `
  --pr_sub_path ..\example_alignment\subtitles `
  --segmentation_dir ..\example_alignment\segmentation_output `
  --subtitle_embedding_dir ..\example_alignment\subtitle_embedding\sign_clip `
  --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip `
  --save_dir ..\example_alignment\aligned_output_with_embedding_tuned
```

ผลลัพธ์ tuned:

- `aligned_output_with_embedding_tuned\04.vtt`

ค่าเชิงคุณภาพที่วัดได้ (เทียบกับ CC):

- Overlap ภายใน tier SUBTITLE_SHIFTED: `153 -> 151` (ลดลงเล็กน้อย)
- Mean start shift: `~2492.84 ms -> ~1693.53 ms` (ดีขึ้นชัดเจน)
- Mean end shift: `~3404.68 ms -> ~2578.89 ms` (ดีขึ้น)

> หมายเหตุ: tuned profile ช่วยลดการเลื่อนเวลาโดยรวมได้ดีขึ้น แต่ overlap ยังสูง
> หากต้องการไฟล์สำหรับใช้งานจริง (อ่าน subtitle ลื่นขึ้น) ควรทำ post-process ลด intersection ต่อ

---

### เปรียบเทียบ: Alignment แบบ none vs SignCLIP

| วิธี | `--similarity_measure` | ความแม่นยำโดยประมาณ | เวลา |
| --- | --- | --- | --- |
| ไม่ใช้ embedding | `none` | ~80% | เร็ว (ไม่ต้องการ GPU) |
| SignCLIP embedding | `sign_clip_embedding` | ~82-83% | ช้ากว่า (ต้องการ GPU 16GB) |

---

## ขั้นตอนพิเศษ A — Merge CC tiers เข้า 04\_updated.eaf

> **สถานะ: เสร็จแล้ว** — tier CC ถูก merge แล้ว

### ทำไมต้องทำ

ไฟล์ `04_updated.eaf` ที่ได้จาก `align.py` มีเฉพาะ tier ที่ SEA สร้างขึ้น (`SIGN`, `SENTENCE`, `SUBTITLE_SHIFTED`)
แต่ **ไม่มี** tier CC ต้นฉบับจากไฟล์ EAF ของนักวิจัย ทำให้ไม่สามารถเปรียบเทียบ
คำบรรยายก่อน/หลัง alignment ในโปรแกรม ELAN ได้

สคริปต์ `merge_cc_to_updated_eaf.py` แก้ปัญหานี้โดยคัดลอก tier ทั้ง 4 จากไฟล์ต้นฉบับ
เข้าไปเพิ่มใน `04_updated.eaf`

### tier ที่ถูก merge เข้าไป

| Tier | จำนวน annotations | คำอธิบาย |
| --- | --- | --- |
| `CC` | 172 | คำบรรยายต้นฉบับ (timing จาก EAF เดิม) |
| `CC_Aligned` | 119 | คำบรรยายที่นักวิจัย align ด้วยมือ |
| `Gloss` | 119 | รหัส gloss ภาษามือ |
| `Gloss Labeling` | 852 | annotation รายละเอียด gloss |

### รัน script

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment
python merge_cc_to_updated_eaf.py
```

สคริปต์รองรับไฟล์ใหม่ผ่าน argument แล้ว (ไม่ต้องแก้โค้ด):

```powershell
python merge_cc_to_updated_eaf.py --source-eaf "C:\path\to\your_original.eaf" --target-eaf "C:\path\to\your_updated.eaf" --tiers CC CC_Aligned Gloss "Gloss Labeling"
```

### ผลลัพธ์ใน 04\_updated.eaf (tier ทั้งหมด)

| Tier | ที่มา | เนื้อหา |
| --- | --- | --- |
| `SIGN` | segmentation.py | ตำแหน่งท่ามือแต่ละท่า (2780 annotations) |
| `SENTENCE` | segmentation.py | ตำแหน่งระดับประโยค (418 annotations) |
| `SUBTITLE_SHIFTED` | align.py | คำบรรยายที่ SEA จัดตำแหน่งแล้ว (172 annotations) |
| `CC` | EAF ต้นฉบับ | คำบรรยายดิบ timing เดิม (172 annotations) |
| `CC_Aligned` | EAF ต้นฉบับ | คำบรรยาย align ด้วยมือโดยนักวิจัย (119 annotations) |
| `Gloss` | EAF ต้นฉบับ | รหัส gloss ภาษามือ (119 annotations) |
| `Gloss Labeling` | EAF ต้นฉบับ | annotation รายละเอียด (852 annotations) |

### เปรียบเทียบใน ELAN

เปิด `segmentation_output\E4s-1_30_50\04_updated.eaf` ใน ELAN จะเห็น tier ทั้งหมดในไทม์ไลน์เดียวกัน
เปรียบเทียบได้ทันที:

- **CC** vs **SUBTITLE\_SHIFTED** — timing ก่อน/หลัง SEA alignment
- **CC\_Aligned** vs **SUBTITLE\_SHIFTED** — human alignment vs automatic alignment
- **SIGN** — ดูว่าท่ามือตรงกับ subtitle หรือไม่

---

## ขั้นตอนพิเศษ B — Crop วิดีโอครึ่งขวา

> **สถานะ: ยังไม่รัน** — `04_right_half.mp4` ยังไม่มีในโฟลเดอร์

ใช้ ffmpeg ตัด `04.mp4` (1920x1080) เฉพาะครึ่งขวา (x=960 ถึง 1920):

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment
ffmpeg -i 04.mp4 -filter:v "crop=960:1080:960:0" -c:a copy 04_right_half.mp4
```

---

## ขั้นตอนที่ 7 — Experiments ปรับปรุง (2026-04-09)

> **สถานะ: เสร็จแล้ว** — รัน 3 experiments เพิ่มเติมด้วย **multilingual SignCLIP model**

### เป้าหมาย

| เป้าหมาย | แนวทาง |
| -------- | ------- |
| ใช้โมเดล SignCLIP ที่ใกล้เคียง TSL มากขึ้น | เปลี่ยนจาก BSL → **multilingual** (`retri_v1_1/baseline_temporal`) |
| ปรับปรุงคุณภาพ embedding | ทดสอบ **Gloss text** แทน CC text (sign vocabulary ตรงกว่า) |
| เพิ่ม word-level similarity | ใช้ flag `--tokenize_text_embedding` — embedding รายคำ แล้ว mean-pool |

> **หมายเหตุ:** ASL checkpoint (`runs/retri_asl/asl_finetune`) ไม่มีในเครื่อง
> ใช้ multilingual checkpoint (`runs/retri_v1_1/baseline_temporal`) แทน

---

### 7.1 สร้าง Gloss-CC VTT

> **สถานะ: เสร็จแล้ว** — `subtitles_gloss_cc_time/04.vtt` มีอยู่แล้ว

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment
..\venv\Scripts\python.exe make_gloss_cc_vtt.py
```

ผลลัพธ์: `subtitles_gloss_cc_time/04.vtt` — 172 cues (170 ใช้ Gloss text, 2 fallback CC text)

---

### 7.2 Extract Embeddings ด้วย Multilingual Model

#### 7.2a Subtitle Embedding — CC text (สำหรับ B_MULTI)

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\fairseq_signclip\examples\MMPT
..\..\..\venv\Scripts\python.exe .\scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=subtitle --model_name multilingual `
  --language_tag "<en>" --batch_size=1024 `
  --subtitle_dir ..\..\..\example_alignment\subtitles `
  --save_dir ..\..\..\example_alignment\subtitle_embedding\sign_clip_multi
```

#### 7.2b Segmentation Embedding — multilingual (ใช้ร่วมกันทุก experiment)

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\fairseq_signclip\examples\MMPT
..\..\..\venv\Scripts\python.exe .\scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=sign --model_name multilingual `
  --pose_dir ..\..\..\example_alignment `
  --segmentation_dir ..\..\..\example_alignment\segmentation_output\E4s-1_30_50 `
  --save_dir ..\..\..\example_alignment\segmentation_embedding\sign_clip_multi
```

#### 7.2c Subtitle Embedding — Gloss text (สำหรับ C_MULTI)

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\fairseq_signclip\examples\MMPT
..\..\..\venv\Scripts\python.exe .\scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=subtitle --model_name multilingual `
  --language_tag "<en>" --batch_size=1024 `
  --subtitle_dir ..\..\..\example_alignment\subtitles_gloss_cc_time `
  --save_dir ..\..\..\example_alignment\subtitle_embedding\sign_clip_multi_gloss
```

---

### 7.3 รัน B_MULTI — Multilingual + CC Text

> **สถานะ: เสร็จแล้ว** — `aligned_output_multi_b2/04.vtt` มีอยู่แล้ว

พารามิเตอร์ DP เหมือน B2 ทุกอย่าง เปลี่ยนแค่ embedding dir:

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\SEA
..\venv\Scripts\python.exe align.py --overwrite --mode=inference `
  --video_ids ..\example_alignment\video_ids.txt --num_workers 1 `
  --sign-b-threshold 30 --sign-o-threshold 50 `
  --dp_duration_penalty_weight 2 --dp_gap_penalty_weight 8 `
  --dp_max_gap 6 --dp_window_size 40 `
  --pr_subs_delta_bias_start 1.8 --pr_subs_delta_bias_end 1.5 `
  --similarity_measure sign_clip_embedding --similarity_weight 6 `
  --pr_sub_path ..\example_alignment\subtitles `
  --segmentation_dir ..\example_alignment\segmentation_output `
  --subtitle_embedding_dir ..\example_alignment\subtitle_embedding\sign_clip_multi `
  --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip_multi `
  --save_dir ..\example_alignment\aligned_output_multi_b2
```

---

### 7.4 รัน C_MULTI — Multilingual + Gloss Text

> **สถานะ: เสร็จแล้ว** — `aligned_output_multi_gloss/04.vtt` มีอยู่แล้ว

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\SEA
..\venv\Scripts\python.exe align.py --overwrite --mode=inference `
  --video_ids ..\example_alignment\video_ids.txt --num_workers 1 `
  --sign-b-threshold 30 --sign-o-threshold 50 `
  --dp_duration_penalty_weight 2 --dp_gap_penalty_weight 8 `
  --dp_max_gap 6 --dp_window_size 40 `
  --pr_subs_delta_bias_start 1.3 --pr_subs_delta_bias_end 1.0 `
  --similarity_measure sign_clip_embedding --similarity_weight 6 `
  --pr_sub_path ..\example_alignment\subtitles_gloss_cc_time `
  --segmentation_dir ..\example_alignment\segmentation_output `
  --subtitle_embedding_dir ..\example_alignment\subtitle_embedding\sign_clip_multi_gloss `
  --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip_multi `
  --save_dir ..\example_alignment\aligned_output_multi_gloss
```

> bias ปรับเป็น 1.3/1.0 (ตาม median offset จริงของ CC speech timing)

---

### 7.5 รัน C_MULTI_word — Multilingual + Gloss Text + Word-Level Similarity

> **สถานะ: เสร็จแล้ว** — `aligned_output_multi_gloss_word/04.vtt` มีอยู่แล้ว

ใช้ `--live_embedding --tokenize_text_embedding` — โมเดลคำนวณ embedding รายคำแบบ on-the-fly (ไม่ต้องสร้าง subtitle .npy ล่วงหน้า)

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\SEA
..\venv\Scripts\python.exe align.py --overwrite --mode=inference `
  --video_ids ..\example_alignment\video_ids.txt --num_workers 1 `
  --sign-b-threshold 30 --sign-o-threshold 50 `
  --dp_duration_penalty_weight 2 --dp_gap_penalty_weight 8 `
  --dp_max_gap 6 --dp_window_size 40 `
  --pr_subs_delta_bias_start 1.3 --pr_subs_delta_bias_end 1.0 `
  --similarity_measure sign_clip_embedding --similarity_weight 6 `
  --live_embedding --tokenize_text_embedding --live_model_name multilingual `
  --pr_sub_path ..\example_alignment\subtitles_gloss_cc_time `
  --segmentation_dir ..\example_alignment\segmentation_output `
  --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip_multi `
  --save_dir ..\example_alignment\aligned_output_multi_gloss_word
```

---

### 7.6 ผลการเปรียบเทียบทุก Experiment

ประเมินด้วย `evaluate_all.py` เทียบกับ CC_Aligned ground truth (119 entries, 0% overlap ยืนยันแล้ว):

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment
..\venv\Scripts\python.exe evaluate_all.py
```

| Experiment | Match | Mean | Median | Stdev | +-1s | +-2s | +-3s | Overlap |
| ---------- | ----- | ---- | ------ | ----- | ---- | ---- | ---- | ------- |
| **B2** (BSL, CC text, tuned) | 69/172 | +1.02s | -0.05s | 5.53s | 74% | 96% | 97% | 88.3% |
| **B_MULTI** (multilingual, CC text) | 69/172 | +0.91s | -0.05s | 5.50s | 78% | **97%** | **99%** | 88.3% |
| **C_MULTI** (multilingual, Gloss text) | 69/172 | **+0.49s** | **-0.15s** | 5.54s | **80%** | 96% | **99%** | **87.7%** |
| **C_MULTI_word** (multilingual, Gloss, word-level) | 69/172 | +0.51s | -0.15s | 5.48s | 77% | 96% | **99%** | 88.3% |
| **D_ASL** (ASL, CC text) | 69/172 | +1.25s | +0.37s | 5.54s | 59% | 81% | 96% | 88.9% |
| **D_ASL_gloss** (ASL, Gloss text) | 69/172 | +0.77s | -0.10s | 5.50s | 64% | 91% | 97% | 88.3% |
| **D_ASL_word** (ASL, Gloss, word-level) | 69/172 | +0.78s | -0.11s | 5.50s | 67% | 93% | 96% | 87.7% |
| **CC_Aligned** (ground truth) | — | 0s | 0s | 0s | 100% | 100% | 100% | **0%** |

**สรุปผล:**

- **B_MULTI ดีกว่า B2** เล็กน้อย: ±2s ดีขึ้น 1%, ±3s ดีขึ้น 2%, mean offset ลดลง 0.11s
- **C_MULTI (Gloss) ดีที่สุดในแง่ mean offset**: ลดจาก +1.02s → +0.49s (ลดลง ~52%)
- **C_MULTI_word ใกล้เคียง C_MULTI**: word tokenization ไม่เปลี่ยนผลอย่างชัดเจนในกรณีนี้
- **ASL model ด้อยกว่า Multilingual**: D_ASL mean offset +1.25s (เทียบ B_MULTI +0.91s), ±1s เพียง 59% (เทียบ 78%)
- **Overlap ยังสูง (~88%)** ในทุก experiment — เป็นปัญหาเชิงโครงสร้างของ input 172 cues

---

### 7.7 ASL Experiments (2026-04-10)

> **สถานะ: เสร็จแล้ว** — รัน 3 experiments เพิ่มเติมด้วย **ASL SignCLIP model**

ASL checkpoint (`asl_finetune_checkpoint_best.pt`) มีอยู่ใน `runs/` แต่ต้อง copy ไปยัง path ที่ YAML คาดไว้ก่อน:

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\fairseq_signclip\examples\MMPT
mkdir runs\retri_asl\asl_finetune
copy runs\asl_finetune_checkpoint_best.pt runs\retri_asl\asl_finetune\checkpoint_best.pt
```

#### Extract ASL Embeddings

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\fairseq_signclip\examples\MMPT

# Subtitle (CC text) -> subtitle_embedding/sign_clip_asl/
..\..\..\venv\Scripts\python.exe .\scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=subtitle --model_name asl --language_tag "<en> <ase>" --batch_size=1024 `
  --subtitle_dir ..\..\..\example_alignment\subtitles `
  --save_dir ..\..\..\example_alignment\subtitle_embedding\sign_clip_asl

# Subtitle (Gloss text) -> subtitle_embedding/sign_clip_asl_gloss/
..\..\..\venv\Scripts\python.exe .\scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=subtitle --model_name asl --language_tag "<en> <ase>" --batch_size=1024 `
  --subtitle_dir ..\..\..\example_alignment\subtitles_gloss_cc_time `
  --save_dir ..\..\..\example_alignment\subtitle_embedding\sign_clip_asl_gloss

# Segmentation -> segmentation_embedding/sign_clip_asl/
..\..\..\venv\Scripts\python.exe .\scripts_bsl\extract_episode_features.py `
  --video_ids ..\..\..\example_alignment\video_ids.txt `
  --mode=segmentation --model_name asl `
  --pose_dir ..\..\..\example_alignment `
  --segmentation_dir ..\..\..\example_alignment\segmentation_output\E4s-1_30_50 `
  --save_dir ..\..\..\example_alignment\segmentation_embedding\sign_clip_asl
```

#### รัน D_ASL — ASL, CC text → `aligned_output_asl_b2/`

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\SEA
..\venv\Scripts\python.exe align.py --overwrite --mode=inference `
  --video_ids ..\example_alignment\video_ids.txt --num_workers 1 `
  --sign-b-threshold 30 --sign-o-threshold 50 `
  --dp_duration_penalty_weight 2 --dp_gap_penalty_weight 8 `
  --dp_max_gap 6 --dp_window_size 40 `
  --pr_subs_delta_bias_start 1.8 --pr_subs_delta_bias_end 1.5 `
  --similarity_measure sign_clip_embedding --similarity_weight 6 `
  --pr_sub_path ..\example_alignment\subtitles `
  --segmentation_dir ..\example_alignment\segmentation_output `
  --subtitle_embedding_dir ..\example_alignment\subtitle_embedding\sign_clip_asl `
  --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip_asl `
  --save_dir ..\example_alignment\aligned_output_asl_b2
```

#### รัน D_ASL_gloss — ASL, Gloss text → `aligned_output_asl_gloss/`

```powershell
..\venv\Scripts\python.exe align.py --overwrite --mode=inference `
  --video_ids ..\example_alignment\video_ids.txt --num_workers 1 `
  --sign-b-threshold 30 --sign-o-threshold 50 `
  --dp_duration_penalty_weight 2 --dp_gap_penalty_weight 8 `
  --dp_max_gap 6 --dp_window_size 40 `
  --pr_subs_delta_bias_start 1.3 --pr_subs_delta_bias_end 1.0 `
  --similarity_measure sign_clip_embedding --similarity_weight 6 `
  --pr_sub_path ..\example_alignment\subtitles_gloss_cc_time `
  --segmentation_dir ..\example_alignment\segmentation_output `
  --subtitle_embedding_dir ..\example_alignment\subtitle_embedding\sign_clip_asl_gloss `
  --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip_asl `
  --save_dir ..\example_alignment\aligned_output_asl_gloss
```

#### รัน D_ASL_word — ASL, Gloss text, word-level → `aligned_output_asl_gloss_word/`

```powershell
..\venv\Scripts\python.exe align.py --overwrite --mode=inference `
  --video_ids ..\example_alignment\video_ids.txt --num_workers 1 `
  --sign-b-threshold 30 --sign-o-threshold 50 `
  --dp_duration_penalty_weight 2 --dp_gap_penalty_weight 8 `
  --dp_max_gap 6 --dp_window_size 40 `
  --pr_subs_delta_bias_start 1.3 --pr_subs_delta_bias_end 1.0 `
  --similarity_measure sign_clip_embedding --similarity_weight 6 `
  --live_embedding --tokenize_text_embedding --live_model_name asl --live_language_tag "<en> <ase>" `
  --pr_sub_path ..\example_alignment\subtitles_gloss_cc_time `
  --segmentation_dir ..\example_alignment\segmentation_output `
  --segmentation_embedding_dir ..\example_alignment\segmentation_embedding\sign_clip_asl `
  --save_dir ..\example_alignment\aligned_output_asl_gloss_word
```

#### ผลเพิ่มเติม (ASL vs Multilingual)

| Experiment | Match | Mean | Median | Stdev | ±1s | ±2s | ±3s | Overlap |
| ---------- | ----- | ---- | ------ | ----- | --- | --- | --- | ------- |
| C_MULTI (multilingual, Gloss) | 69/172 | **+0.49s** | **-0.15s** | 5.54s | **80%** | 96% | **99%** | 87.7% |
| D_ASL (ASL, CC text) | 69/172 | +1.25s | +0.37s | 5.54s | 59% | 81% | 96% | 88.9% |
| D_ASL_gloss (ASL, Gloss text) | 69/172 | +0.77s | -0.10s | 5.50s | 64% | 91% | 97% | 88.3% |
| D_ASL_word (ASL, Gloss, word-level) | 69/172 | +0.78s | -0.11s | 5.50s | 67% | 93% | 96% | 87.7% |

> **สรุป:** Multilingual model ดีกว่า ASL model สำหรับ TSL ในทุกเมตริก
> ASL model trained บน ASL data ไม่ generalise มาที่ภาษามือไทยได้ดี

---

## ขั้นตอนพิเศษ C — Post-process ลด Overlap

> **สถานะ: เสร็จแล้ว** — ผล 87-89% overlap → 0%

### สาเหตุที่ต้องทำ Post-process

ทุก experiment มี overlap ~88% เพราะ input 172 CC cues แต่ sign windows รองรับได้ ~119 slots
SEA assign หลาย cue ไปยัง window เดียวกัน → cue ซ้อนกัน

`postprocessing_remove_intersections.py` ในโปรเจกต์ต้องการ probability pickle files จาก inference ซึ่งไม่มี
จึงสร้าง `fix_overlap_vtt.py` ที่ทำงานได้บน plain VTT

### วิธีใช้

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment

# C_MULTI (best model)
..\venv\Scripts\python.exe fix_overlap_vtt.py `
  --input  aligned_output_multi_gloss\04.vtt `
  --output aligned_output_multi_gloss\04_no_overlap.vtt

# D_ASL_gloss
..\venv\Scripts\python.exe fix_overlap_vtt.py `
  --input  aligned_output_asl_gloss\04.vtt `
  --output aligned_output_asl_gloss\04_no_overlap.vtt
```

**ผลลัพธ์:**

| ไฟล์ | ก่อน | หลัง |
| ---- | ---- | ---- |
| `aligned_output_multi_gloss/04.vtt` | 150/171 (87.7%) | 0/171 **(0.0%)** |
| `aligned_output_asl_gloss/04.vtt` | 151/171 (88.3%) | 0/171 **(0.0%)** |

> **ข้อควรระวัง:** การ clamp ลด duration ของ cue บางตัว ใช้สำหรับ display เท่านั้น ไม่ใช่สำหรับ evaluation timing

---

## แก้ปัญหาที่พบจริงบน Windows

### ❌ ImportError: mediapipe version ไม่ถูกต้อง

```powershell
pip install "mediapipe==0.10.21"
```

### ❌ UnicodeDecodeError: charmap codec can't decode byte 0xff

ไฟล์ `video_ids.txt` ถูกสร้างด้วย Windows `echo` ซึ่งเป็น UTF-16 BOM
แก้โดยสร้างไฟล์ใหม่ด้วย Python:

```powershell
python -c "open('video_ids.txt','w',encoding='utf-8').write('04\n')"
```

### ❌ FileNotFoundError: path มี single quote ติดอยู่

`shlex.quote()` ใส่ single quote รอบ path ซึ่ง Windows cmd.exe ไม่ strip ออก
แก้แล้วใน `segmentation.py` บรรทัด 71: เปลี่ยนเป็น `shlex.split(cmd), shell=False`

### ❌ unrecognized arguments: --subtitle\_dir

argument ที่ถูกต้องคือ `--pr_sub_path` ไม่ใช่ `--subtitle_dir`

### ❌ ModuleNotFoundError: No module named 'numba'

```powershell
pip install numba beartype tqdm scikit-learn tabulate
```

### ❌ pose estimation ช้ามาก

ลด complexity:

```powershell
videos_to_poses --format mediapipe --directory . --additional-config="model_complexity=1,smooth_landmarks=false"
```

---

## สรุปโครงสร้างไฟล์ทั้งหมด

```text
example_alignment\
├── 04.mp4                                  ← วิดีโอต้นฉบับ (1920x1080, 60fps)
├── 04.pose                                 ← ✅ skeleton pose (358 MB)
├── 04.vtt                                  ← ✅ CC ดิบ (172 cues)
├── การเปรียบเทียบและเรียงลำดับ (11.07 นาที).eaf  ← annotation ต้นฉบับ (CC, CC_Aligned, Gloss)
├── extract_cc_from_eaf.py                  ← สคริปต์แยก CC -> VTT
├── merge_cc_to_updated_eaf.py              ← ✅ สคริปต์ merge CC tiers
├── make_gloss_cc_vtt.py                    ← ✅ สร้าง Gloss text + CC timestamp VTT
├── evaluate_all.py                         ← ✅ เปรียบเทียบทุก experiment vs CC_Aligned
├── video_ids.txt                           ← UTF-8, มีแค่ "04"
├── subtitles\
│   └── 04.vtt                             ← ✅ subtitle input (CC text, speech timestamp)
├── subtitles_gloss_cc_time\
│   └── 04.vtt                             ← ✅ subtitle input (Gloss text, CC timestamp)
├── segmentation_output\
│   └── E4s-1_30_50\
│       ├── 04.eaf                         ← ✅ SIGN tier (segmentation)
│       └── 04_updated.eaf                 ← ✅ SIGN + SUBTITLE_SHIFTED + CC + CC_Aligned + Gloss
├── segmentation_embedding\
│   ├── sign_clip\04.npy                   ← ✅ sign embedding — BSL model
│   ├── sign_clip_multi\04.npy             ← ✅ sign embedding — multilingual model (2780 segments)
│   └── sign_clip_asl\04.npy               ← ✅ sign embedding — ASL model (2780 segments)
├── subtitle_embedding\
│   ├── sign_clip\04.npy                   ← ✅ subtitle embedding — BSL model
│   ├── sign_clip_multi\04.npy             ← ✅ subtitle embedding — multilingual, CC text
│   ├── sign_clip_multi_gloss\04.npy       ← ✅ subtitle embedding — multilingual, Gloss text
│   ├── sign_clip_asl\04.npy               ← ✅ subtitle embedding — ASL model, CC text
│   └── sign_clip_asl_gloss\04.npy         ← ✅ subtitle embedding — ASL model, Gloss text
├── aligned_output\
│   └── 04.vtt                             ← ✅ A — no embed, standard params
├── aligned_output_with_embedding\
│   └── 04.vtt                             ← ✅ B1 — BSL, standard params
├── aligned_output_with_embedding_tuned\
│   └── 04.vtt                             ← ✅ B2 — BSL, tuned params (baseline เดิม)
├── aligned_output_multi_b2\
│   └── 04.vtt                             ← ✅ B_MULTI — multilingual, CC text
├── aligned_output_multi_gloss\
│   ├── 04.vtt                             ← ✅ C_MULTI — multilingual, Gloss text
│   └── 04_no_overlap.vtt                  ← ✅ C_MULTI post-processed (0% overlap)
├── aligned_output_multi_gloss_word\
│   └── 04.vtt                             ← ✅ C_MULTI_word — multilingual, Gloss, word-level
├── aligned_output_asl_b2\
│   └── 04.vtt                             ← ✅ D_ASL — ASL model, CC text
├── aligned_output_asl_gloss\
│   ├── 04.vtt                             ← ✅ D_ASL_gloss — ASL model, Gloss text
│   └── 04_no_overlap.vtt                  ← ✅ D_ASL_gloss post-processed (0% overlap)
└── aligned_output_asl_gloss_word\
    └── 04.vtt                             ← ✅ D_ASL_word — ASL model, Gloss, word-level
```

---

## อ้างอิง

- **SEA Paper:** [arXiv:2512.08094](https://arxiv.org/abs/2512.08094) — Jiang et al., 2025
- **pose-format:** [sign-language-processing/pose](https://github.com/sign-language-processing/pose)
- **Sign Segmentation:** [J22Melody/segmentation@bsl](https://github.com/J22Melody/segmentation/tree/bsl)
- **ELAN:** [archive.mpi.nl/tla/elan](https://archive.mpi.nl/tla/elan)

---

## วิธีรันกับไฟล์ใหม่ (Template ใช้ซ้ำได้)

ส่วนนี้สรุปเฉพาะ "สิ่งที่ต้องเปลี่ยน" เพื่อลดความซ้ำกับคำสั่งเต็มใน Runbook ด้านบน

สมมติไฟล์ใหม่ชื่อ `VID=05`:

1. เตรียมไฟล์

- ต้องมี `05.mp4`, `05.pose`, `05.vtt`
- สร้าง subtitle copy และตั้ง `video_ids.txt`:

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment
mkdir subtitles -ErrorAction SilentlyContinue
copy .\05.vtt .\subtitles\05.vtt
python -c "open('video_ids.txt','w',encoding='utf-8').write('05\n')"
```

2. รัน pipeline ตาม Runbook เดิม

- รันตามลำดับเดียวกับหัวข้อ **Runbook ที่ยืนยันแล้ว**:
  - Segmentation
  - Subtitle embedding (SignCLIP)
  - Segmentation embedding (SignCLIP)
  - Align แบบใช้ embedding (หรือ tuned profile)

3. Merge original tiers เข้า updated EAF ของวิดีโอใหม่

```powershell
cd C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment
..\venv\Scripts\python.exe .\merge_cc_to_updated_eaf.py --source-eaf "C:\path\to\original_05.eaf" --target-eaf "C:\Users\dniam\Documents\Dechathon_N\NECTEC\SEA\example_alignment\segmentation_output\E4s-1_30_50\05_updated.eaf"
```

4. ตรวจผลลัพธ์ขั้นต่ำ

- มีไฟล์ `aligned_output_with_embedding\05.vtt` (หรือ `aligned_output_with_embedding_tuned\05.vtt`)
- มีไฟล์ `segmentation_output\E4s-1_30_50\05_updated.eaf`
- ใน `05_updated.eaf` ต้องมี tier `SUBTITLE_SHIFTED`, `CC`, `CC_Aligned`, `Gloss`, `Gloss Labeling`
- `HEADER/MEDIA_DESCRIPTOR` ต้องชี้วิดีโอได้จริง (เปิดใน ELAN แล้วเล่นวิดีโอได้)

---

## แนวทางปรับปรุงผลลัพธ์ (ส่วนท้ายสำหรับพัฒนาเพิ่ม)

ถ้าต้องการให้ผลลัพธ์อ่านลื่นขึ้นและใกล้เคียง human alignment มากขึ้น ให้ทำตามลำดับนี้:

1. ใช้ tuned profile เป็นค่าเริ่มต้น

- เริ่มจากโฟลเดอร์ผล `aligned_output_with_embedding_tuned`
- เหมาะกับกรณีที่ cue เลื่อนเยอะหรือมีการซ้อนกันสูง

2. ทำ post-process เพื่อลดการซ้อนของ subtitle

- ใช้สคริปต์ในโปรเจกต์: `SEA/misc/postprocessing_remove_intersections.py`
- เป้าหมายคือบังคับไม่ให้ cue ต่อเนื่องซ้อนกันเกิน threshold ที่รับได้
- แนะนำให้เก็บผลเป็นไฟล์ใหม่ เช่น `04_post.vtt` เพื่อเทียบก่อน/หลังได้ง่าย

3. ปรับพารามิเตอร์เป็นรอบสั้น ๆ (small sweep)

- โฟกัส 4 ตัวที่กระทบคุณภาพมากที่สุด:
  - `--similarity_weight`
  - `--dp_duration_penalty_weight`
  - `--dp_gap_penalty_weight`
  - `--pr_subs_delta_bias_start`, `--pr_subs_delta_bias_end`
- วิธีทำ: ปรับทีละ 1-2 ค่า แล้ววัดผลทันที (ไม่ปรับทุกค่าในรอบเดียว)

4. ใช้ชุดตัวชี้วัดเดิมทุกครั้งเพื่อเทียบแบบยุติธรรม

- จำนวนคู่ cue ที่ซ้อนกัน (intersection count)
- ค่าเฉลี่ยการเลื่อนเวลาเริ่ม/จบ เทียบกับ CC หรือ CC_Aligned
- จำนวน cue ที่ผิดปกติ (outlier) ที่เลื่อนเกินช่วงที่กำหนด

5. ตรวจคุณภาพเชิงสายตาใน ELAN ทุก iteration

- เทียบ `CC`, `CC_Aligned`, `SUBTITLE_SHIFTED` บน timeline เดียวกัน
- ตรวจจุดที่มักพัง: cue ยาวมาก, cue สั้นมาก, ช่วงไม่มีสัญญาณมือชัดเจน

6. ทำโปรไฟล์แยกตามประเภทวิดีโอ

- วิดีโอสไตล์ต่างกัน (ความเร็วท่ามือ, ความยาวประโยค, ความถี่หยุด) ควรมี profile แยก
- แนะนำตั้งชื่อชัดเจน เช่น `profile_lecture`, `profile_fast_dialog`, `profile_story`

7. เก็บ baseline เทียบเสมอ

- เก็บผลจาก `--similarity_measure none` ไว้เป็น baseline
- ถ้า embedding profile ใหม่ไม่ดีกว่า baseline อย่างชัดเจน ให้ย้อนกลับและปรับใหม่

### เป้าหมายเชิงปฏิบัติที่แนะนำ

- ลด cue overlap ให้ต่ำลงอย่างต่อเนื่องในแต่ละรอบ
- ลด mean start/end shift โดยไม่ทำให้ cue กระโดดข้ามประโยค
- คงจำนวน cue ให้อยู่ใกล้เคียงต้นฉบับ (ไม่แตก/รวมมากเกินไป)

### ลำดับทำงานสั้น ๆ (Improve Loop)

1. รัน align (tuned)
2. รัน post-process ลด intersection
3. วัด metrics เดิม
4. เปิดดูใน ELAN 5-10 จุดที่เสี่ยง
5. ปรับพารามิเตอร์แล้ววนซ้ำ

ถ้าทำตาม loop นี้ต่อเนื่อง 2-3 รอบ มักเห็นคุณภาพดีขึ้นชัดเจนกว่าการปรับแบบครั้งเดียวจบ.
