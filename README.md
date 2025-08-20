# Outfit Color Understanding with Segmentation, Calibration & XKCD Naming

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
│  ├─ train.yaml                 # YOLO training config
│  └─ thresholds.yaml            # Device-specific thresholds
├─ data/
│  ├─ raw/                       # Raw datasets
│  ├─ sam/                       # SAM-prepared masks
│  └─ yolo/                      # YOLO-formatted data
├─ models/
│  ├─ best.pt                    # Trained YOLO-seg weights
│  └─ checkpoints/               # Training checkpoints
├─ notebooks/                    # Exploratory analysis
├─ scripts/                      # Processing scripts
│  ├─ 00_sam_prep.py             # SAM mask generation
│  ├─ 01_split_to_yolo.py        # SAM → YOLO conversion
│  ├─ 02_train_yoloseg.py        # Model training
│  ├─ 03_infer_images.py         # Image inference
│  ├─ 04_infer_video.py          # Video processing
│  ├─ 05_live_cam.py             # Live camera demo
│  ├─ 06_color_kmeans_xkcd.py    # Color extraction & naming
│  ├─ 07_dataset_plots.py        # Dataset analysis
│  ├─ 08_device_verification.py  # Threshold testing
│  └─ 09_calibration_fit.py      # Color calibration
├─ results/                      # Output directory
│  ├─ images/                    # Annotated images
│  ├─ videos/                    # Processed videos
│  ├─ plots/                     # Visualizations
│  └─ reports/                   # Analysis reports
├─ README.md
└─ requirements.txt
```

---

## 4) Datasets

**Primary Training Data:**  
[Colorful Fashion Dataset for Object Detection](https://www.kaggle.com/datasets/nguyngiabol/colorful-fashion-dataset-for-object-detection)

**Additional Test Data:**  
- [DeepFashion2](https://github.com/switchablenorms/DeepFashion2) (subset)
- Custom NPUST fashion show images
- CCTV cropped frames (private)

**Supported Classes:**  
`sunglass`, `hat`, `jacket`, `shirt`, `pants`, `shorts`, `skirt`, `dress`, `bag`, `shoe`

---

## 5) Color Calibration

Optional but recommended color calibration using a 3×3 correction matrix:

```
c_calib = M × c_raw
```

Where M is estimated from reference color charts:

```python
# Calibration application
calibrated = (image @ M.T).clip(0, 255).astype(np.uint8)
```

> Generated via `scripts/09_calibration_fit.py` → `results/calibration_matrix.npy`

---

## 6) Training Pipeline

### 6.1 SAM Mask Preparation
```bash
python scripts/00_sam_prep.py --images data/raw/train_images --out data/sam
```

### 6.2 YOLO Format Conversion
```bash
python scripts/01_split_to_yolo.py --sam_dir data/sam --yolo_dir data/yolo
```

### 6.3 Model Training
```bash
python scripts/02_train_yoloseg.py \
  --data configs/train.yaml \
  --model yolo11s-seg.pt \
  --epochs 100 \
  --imgsz 640
```

> Best weights saved to `models/best.pt`

---

## 7) Inference

### 7.1 Image Processing
```bash
python scripts/03_infer_images.py \
  --images data/raw/test_images \
  --weights models/best.pt \
  --out results/images \
  --kcolors 3
```

### 7.2 Video Processing
```bash
python scripts/04_infer_video.py \
  --video data/raw/test_video.mp4 \
  --weights models/best.pt \
  --out results/videos/out.mp4 \
  --fps_out 15
```

---

## 8) Live Demos

### 8.1 Webcam
```bash
python scripts/05_live_cam.py \
  --source 0 \
  --weights models/best.pt \
  --conf_det 0.75 \
  --conf_color 0.45
```

### 8.2 Canon DSLR
```bash
python scripts/05_live_cam.py \
  --source 1 \
  --weights models/best.pt \
  --conf_det 0.8 \
  --conf_color 0.45
```

> Optimal thresholds: Detection 0.70-0.85, Color 0.40-0.50

---

## 9) Color Extraction & Naming

RGB → XKCD color mapping (≈950 named colors):

```python
def dominant_colors(masked_bgr, k=3, M=None):
    rgb = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2RGB)
    if M: rgb = (rgb @ M.T).clip(0,255).astype(np.uint8)
    pts = rgb.reshape(-1,3)
    pts = pts[~np.all(pts==0, axis=1)]
    kmeans = KMeans(n_clusters=k, n_init=10).fit(pts)
    centers = np.rint(kmeans.cluster_centers_).astype(np.uint8)
    names = [nearest_xkcd(c) for c in centers]
    return list(zip(centers, names))
```

---

## 10) Device Verification

Threshold optimization for different capture devices:

```bash
python scripts/08_device_verification.py \
  --inputs "data/raw/cctv/*.mp4" \
  --weights models/best.pt \
  --grid_det 0.30 0.40 0.50 0.70 0.85 1.00 \
  --grid_color 0.10 0.40 0.50 1.00
```

> Results saved to `results/reports/verification.csv`

---

## 11) Dataset Analysis

Generate comprehensive visualizations:

```bash
python scripts/07_dataset_plots.py \
  --reports results/reports/inference.csv \
  --out results/plots
```

**Output includes:**  
- Color distribution charts
- Top-10 color bars
- Class-color heatmaps
- Outfit class distributions

---

## 12) Quick Start

```bash
# 1. Environment setup
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 2. Data preparation (optional)
python scripts/00_sam_prep.py --images data/raw/train_images --out data/sam
python scripts/01_split_to_yolo.py --sam_dir data/sam --yolo_dir data/yolo

# 3. Model training
python scripts/02_train_yoloseg.py --data configs/train.yaml --epochs 100

# 4. Inference
python scripts/03_infer_images.py --images data/raw/test_images --weights models/best.pt

# 5. Live demo
python scripts/05_live_cam.py --source 0 --weights models/best.pt
```

---

## 13) Troubleshooting

**Common issues:**
- GUI errors: Ensure `opencv-python` (not headless) is installed
- Import errors: Pin `ultralytics==8.2.103`
- No detections: Adjust confidence thresholds (try --conf 0.3)
- Color inaccuracies: Verify calibration matrix and avoid auto HDR

---

## 14) Citation

```bibtex
@misc{outfitcolor2025,
  title = {Outfit Color Understanding with Segmentation, Calibration & XKCD Naming},
  author = {Your Name},
  year = {2025},
  howpublished = {\url{https://github.com/Enock02333/Outfit-Color-Understanding-with-Segmentation-Calibration-XKCD-Naming}}
}
```

> Please respect dataset licenses and privacy constraints for captured data

---

## 15) License

- **Code**: MIT License
- **Trained Weights**: Check institutional policies
- **Datasets**: Respect original licenses (Kaggle, DeepFashion2)
- **Captured Data**: Do not publish private CCTV/phone/camera clips
