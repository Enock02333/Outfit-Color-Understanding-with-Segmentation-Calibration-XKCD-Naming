# Outfit Color Understanding with Segmentation, Calibration & XKCD Naming

<p align="center">
  <img src="results/images/Screenshot%202025-08-06%20005216.png" alt="Pipeline Architecture" width="600"/>
</p>

End-to-end pipeline for **detecting/segmenting fashion items**, **calibrating colors**, and **naming dominant colors** using **KMeans** mapped to human-friendly **XKCD colors**.  
Supports **image**, **video**, **webcam/USB/DSLR (Canon)** inputs and **device-wise threshold verification**.  
Built around **Ultralytics YOLO (Seg)** + **Segment Anything (SAM) data prep**.

---

## 1) Features

- **Data preparation with SAM** for polygon masks  
- **YOLO-Seg training** on prepared masks  
- **Color calibration** with a simple **3×3 correction matrix**  
- **Dominant color extraction** from masks via **KMeans** (RGB)  
- **XKCD color naming** (≈950 names) for interpretability  
- **Batch inference** on images/videos + **live webcam/Canon** demos  
- **Device threshold testing** (CCTV / Camera / Phone)  
- **Dataset-level plots** (color distributions, co-occurrence with classes)  

---

## 2) Environment Setup

> Tested on Python 3.10/3.11, Windows 11

```bash
# Create & activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install ultralytics==8.2.103
pip install opencv-python==4.10.0.84
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib==3.8.4 seaborn==0.13.2 scikit-learn==1.4.2 pandas==2.2.2 numpy==1.26.4
```

> GUI note: If cv2.imshow errors occur, uninstall opencv-python-headless and ensure opencv-python is installed

---

## 3) Repository Structure

```
.
├─ configs/
│  ├─ train.yaml                 # YOLO-Seg data spec
│  └─ thresholds.yaml            # device-wise thresholds (optional)
├─ data/
│  ├─ raw/                       # raw images/videos (Kaggle/DeepFashion2/custom)
│  ├─ sam/                       # SAM masks & polygons
│  └─ yolo/                      # YOLO-Seg formatted dataset
├─ models/
│  ├─ best.pt                    # trained YOLO-Seg weights
│  └─ checkpoints/               # intermediate checkpoints
├─ scripts/
│  ├─ 00_sam_prep.py             # SAM-assisted mask generation
│  ├─ 01_split_to_yolo.py        # convert SAM → YOLO-Seg format
│  ├─ 02_train_yoloseg.py        # train segmentation model
│  ├─ 03_infer_images.py         # batch images (seg + KMeans + XKCD)
│  ├─ 04_infer_video.py          # video inference & export
│  ├─ 05_live_cam.py             # webcam/USB/Canon capture inference
│  ├─ 06_color_kmeans_xkcd.py    # color naming utilities
│  ├─ 07_dataset_plots.py        # global plots from inference CSV
│  ├─ 08_device_verification.py  # threshold sweeps (CCTV/Camera/Phone)
│  └─ 09_calibration_fit.py      # estimate 3×3 color correction matrix
├─ results/
│  ├─ images/                    # annotated images
│  ├─ videos/                    # processed videos
│  ├─ plots/                     # analytics plots
│  └─ reports/                   # CSVs, verification logs
├─ README.md
└─ requirements.txt
```

---

## 4) Datasets

* **Training (SAM → segmentation):**
  **Colorful Fashion Dataset for Object Detection** (used during SAM prep and as base for segmentation)
  [https://www.kaggle.com/datasets/nguyngiabol/colorful-fashion-dataset-for-object-detection](https://www.kaggle.com/datasets/nguyngiabol/colorful-fashion-dataset-for-object-detection)

* **Testing & verification:**

  * **DeepFashion2** (subset) – [https://github.com/switchablenorms/DeepFashion2](https://github.com/switchablenorms/DeepFashion2)
  * **Custom NPUST**: fashion show images + CCTV cropped frames (privacy protected)
  * **Real-world**: CCTV / Canon R50 / iPhone 15 Pro Max videos (privacy protected; use your own clips to reproduce)

> The repo provides scripts to run on **your own data** following the same pipeline.

---

## 5) Color Calibration (optional, but recommended)

We estimate a **3×3 linear color correction matrix** $M$ that maps sensor RGB to a reference RGB:

$$
\mathbf{c}_{\text{calib}} = M \, \mathbf{c}_{\text{raw}}
$$

Given measured chart colors $I_i \in \mathbb{R}^3$ and reference colors $R_i \in \mathbb{R}^3$, solve:

$$
\min_{M} \sum_i \lVert M I_i - R_i \rVert_2^2
\quad\Rightarrow\quad
M = R\,I^\top \,(I\,I^\top)^{-1}
$$

* Fit with: `scripts/09_calibration_fit.py` (click-on-patches workflow or CSV of correspondences).
* Save as: `results/calibration_matrix.npy`.
* Apply at inference via `--calib results/calibration_matrix.npy`.

---

## 6) Training: SAM → YOLO-Seg

**(a) Mask Prep with SAM**

```bash
python scripts/00_sam_prep.py \
  --images data/raw/train_images \
  --out data/sam \
  --points_per_side 32
```

**(b) Convert SAM → YOLO-Seg**

```bash
python scripts/01_split_to_yolo.py \
  --sam_dir data/sam \
  --yolo_dir data/yolo \
  --split 0.8 0.1 0.1
```

**(c) Train YOLO-Seg**

```bash
python scripts/02_train_yoloseg.py \
  --data configs/train.yaml \
  --model yolo11s-seg.pt \
  --epochs 100 \
  --imgsz 640 \
  --project models/checkpoints \
  --name exp_seg
```

> Final weights expected at `models/best.pt`.

---

## 7) Inference (Images / Video)

**Batch Images (segmentation + KMeans + XKCD naming)**

```bash
python scripts/03_infer_images.py \
  --images data/raw/test_images \
  --weights models/best.pt \
  --out results/images \
  --conf 0.5 --iou 0.5 \
  --kcolors 3 \
  --calib results/calibration_matrix.npy
```

**Single/Batch Video**

```bash
python scripts/04_infer_video.py \
  --video data/raw/test_video.mp4 \
  --weights models/best.pt \
  --out results/videos/out.mp4 \
  --conf 0.5 --iou 0.5 \
  --kcolors 3 \
  --fps_out 15 \
  --calib results/calibration_matrix.npy
```

> Annotated frames show **class + XKCD color** (e.g., *“blue dress”*). If color confidence is too low, label may show `???` per our safety logic to **avoid mislabeling**.

---

## 8) Live Demos (Webcam / Canon)

**Webcam (internal)**

```bash
python scripts/05_live_cam.py --source 0 --weights models/best.pt --conf 0.5 --iou 0.5
```

**Canon R50 (via capture card / USB video class)**

```bash
python scripts/05_live_cam.py --source 1 --weights models/best.pt --conf 0.5 --iou 0.5
```

> If GUI errors on Windows: `pip uninstall opencv-python-headless` then `pip install opencv-python`.

---

## 9) XKCD Color Naming

* We map KMeans centroids to the **nearest XKCD color** (≈950 names) using Euclidean distance in RGB.
* The mapping list is embedded/loaded in `scripts/06_color_kmeans_xkcd.py` (JSON or dict).
* Output examples: `blue dress`, `charcoal jacket`, `beige pants`.
* If calibrated matrix is provided (`--calib`), we apply it to **masked pixels** before clustering to reduce lighting bias.

---

## 10) Device Threshold Verification

Run sweeps across **person-detection confidence** and **color-confidence**:

```bash
python scripts/08_device_verification.py \
  --inputs "data/raw/cctv/*.mp4" "data/raw/camera/*.mp4" "data/raw/phone/*.mp4" \
  --weights models/best.pt \
  --grid_det 0.30 0.40 0.50 0.70 0.85 1.00 \
  --grid_color 0.10 0.40 0.50 1.00 \
  --out results/reports/verification.csv
```

Generates device-wise CSV with precision-like signals for **ID stability**, **color stability**, and **missed detections** at each threshold pair.

---

## 11) Results & Observations (Brief)

* **CCTV**: Camera4 & Camera16 worst; Camera14 borderline; Camera1 acceptable. Very low color discernibility in some Camera14 clips → color labeled `???` (by design).
* **Canon R50**: Exceptional; stable across thresholds; colors correctly named; far subjects may still get `???` if mask area too small.
* **iPhone 15 Pro Max**: Very good; at low thresholds, similar hues (cement/grey/silver) may interchange with motion/exposure changes; stabilizes at moderate thresholds.
* **Best practice thresholds** (empirical from tests):

  * Person detection **0.70–0.85** → robust tracking without duplicate IDs from motion.
  * Color detection **0.40–0.50** → stable naming, minimal swapping; `0.10` too permissive on phone/CCTV.

---

## 12) Dataset-Level Plots

Create global plots from consolidated inference CSV (image/video runs):

```bash
python scripts/07_dataset_plots.py \
  --reports results/reports/inference.csv \
  --out results/plots
```

Generates:

* **Top-10 dominant colors** (bar)
* **Color proportion** (donut)
* **Horizontal color strip** (palette)
* **Outfit class distribution** (bar)
* **Class–color co-occurrence** (heatmap)

---

## 13) Performance Comparison Tables

### Legend

* **✅** strong/consistent **△** borderline/variable **❌** poor/failed **—** not applicable

### A) Device-wise Threshold Sensitivity (Summary)

| Device                     | Person Conf (Best) | Color Conf (Best) | Low Thresholds (0.10/0.30)         | High Thresholds (1.00)                | Notes                                                   |
| -------------------------- | -----------------: | ----------------: | ---------------------------------- | ------------------------------------- | ------------------------------------------------------- |
| CCTV (mixed: Cam1/4/14/16) |          0.70–0.85 |         0.40–0.50 | △ (dup IDs, color swaps)           | ❌ (often no detections/colors)        | Cam4/16 weakest; Cam14 borderline; Cam1 acceptable.     |
| Canon R50                  |          0.70–0.85 |         0.40–0.50 | ✅ (minor swaps only at 0.10)       | △ (over-prunes distant subjects)      | Best overall stability & color fidelity.                |
| iPhone 15 Pro Max          |          0.70–0.85 |         0.40–0.50 | △ (cement/grey/silver interchange) | △/❌ (missed detections in some clips) | Good daylight, sensitive to exposure at low color conf. |

### B) CCTV Camera Breakdown (Qualitative)

| Camera | Clarity | Color Discernibility | Detection Reliability | Comment                                            |
| ------ | ------- | -------------------- | --------------------- | -------------------------------------------------- |
| Cam1   | ✅       | △                    | ✅                     | Acceptable baseline.                               |
| Cam4   | ❌       | ❌                    | △                     | Weakest; heavy noise/blur.                         |
| Cam14  | △       | ❌                    | △                     | Some clips acceptable; many too ambiguous → `???`. |
| Cam16  | ❌       | ❌                    | △                     | Similar to Cam4; unstable.                         |

### C) Lighting Robustness (Day vs Night)

| Device            | Daytime Detection | Daytime Color | Night Detection | Night Color | Notes                                                                               |
| ----------------- | ----------------- | ------------- | --------------- | ----------- | ----------------------------------------------------------------------------------- |
| CCTV              | ✅                 | △             | △               | △           | Night lights brighten backgrounds; calibration reduces noise but ambiguity remains. |
| Canon R50         | ✅                 | ✅             | ✅               | △/✅         | Strong sensor; slight hue drift under mixed light.                                  |
| iPhone 15 Pro Max | ✅                 | ✅             | △/✅             | △           | Good overall; prefers moderate thresholds at night.                                 |

### D) Final Consolidated View

| Device            | Overall Detection | Overall Color Naming | Recommended Thresholds (Det / Color) |
| ----------------- | ----------------- | -------------------- | ------------------------------------ |
| CCTV              | △                 | △                    | 0.70–0.85 / 0.40–0.50                |
| Canon R50         | ✅                 | ✅                    | 0.70–0.85 / 0.40–0.50                |
| iPhone 15 Pro Max | ✅                 | ✅                    | 0.70–0.85 / 0.40–0.50                |

**Implications:** For **uncontrolled CCTV**, prioritize **higher detection thresholds** and **moderate color thresholds**; expect `???` for far/small masks. For **cameras/phones**, the recommended bands above provide a good balance between **stability** and **coverage**.

---

## 14) Troubleshooting

* **No GUI / imshow error (Windows):**
  `pip uninstall opencv-python-headless` → `pip install opencv-python`
* **ModuleNotFoundError: cv2**
  Ensure VS Code uses the **same interpreter** where you installed deps (`Python: Select Interpreter`).
* **No detections on video:**
  Lower `--conf` or increase `--imgsz`; verify correct `--source` index for camera.
* **Color looks wrong:**
  Fit/apply **calibration**; increase `--kcolors`; ensure masks are correct (segmentation).
* **Rapid color swapping:**
  Use **color conf 0.40–0.50**; apply calibration; filter tiny masks.

---

## 15) Citation

```bibtex
@misc{outfitcolor2025,
  title = {Outfit Color Understanding with Segmentation, Calibration \& XKCD Naming},
  author = {Enock Isack},
  year = {2025},
  howpublished = {\url{https://github.com/Enock02333/Outfit-Color-Understanding-with-Segmentation-Calibration-XKCD-Naming}}
}
```

---

## 16) Acknowledgements

* Ultralytics YOLO-Seg, Meta AI SAM, XKCD color survey contributors.
* Datasets: Kaggle Colorful Fashion, DeepFashion2, and private NPUST collections (used for evaluation only).

```

::contentReference[oaicite:0]{index=0}
```

