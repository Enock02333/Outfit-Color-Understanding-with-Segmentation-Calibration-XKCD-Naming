"""
Run device-wise threshold verification on videos or image folders.
Uses thresholds.yaml profiles and records detection/color stability stats.
"""

import argparse, time, yaml, numpy as np, pandas as pd
from pathlib import Path
import cv2
from ultralytics import YOLO

def load_profile(th_yaml, profile):
    cfg = yaml.safe_load(Path(th_yaml).read_text())
    prof = cfg.get("profiles", {}).get(profile, cfg.get("default", {}))
    return {
        "person_conf": float(prof.get("person_conf", 0.5)),
        "person_iou": float(prof.get("person_iou", 0.5)),
        "color_conf": float(prof.get("color_conf", 0.4)),
        "imgsz": int(prof.get("imgsz", 640))
    }

def analyze_frame(model, frame, pconf, piou, imgsz):
    res = model.predict(source=frame, conf=pconf, iou=piou, imgsz=imgsz, verbose=False)[0]
    n = 0 if res.boxes is None else len(res.boxes)
    return n, res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--inputs", type=str, required=True, help="file or folder")
    ap.add_argument("--thresholds", type=str, default="thresholds.yaml")
    ap.add_argument("--profile", type=str, default="cctv", choices=["cctv","camera","phone"])
    ap.add_argument("--max_frames", type=int, default=500)
    ap.add_argument("--out_csv", type=str, default="results/device_verification.csv")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True, parents=True)
    prof = load_profile(args.thresholds, args.profile)
    model = YOLO(args.weights)

    paths = []
    p = Path(args.inputs)
    if p.is_dir():
        paths = sorted([q for q in p.iterdir() if q.suffix.lower() in [".mp4",".mov",".avi",".mkv"]])
    else:
        paths = [p]

    rows = []
    for vid in paths:
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print("Cannot open", vid)
            continue
        cnt = 0; det_counts = []; t0 = time.time()
        while True and cnt < args.max_frames:
            ok, frame = cap.read()
            if not ok: break
            n, _ = analyze_frame(model, frame, prof["person_conf"], prof["person_iou"], prof["imgsz"])
            det_counts.append(n)
            cnt += 1
        cap.release()
        fps = cnt / max(1e-6, (time.time() - t0))
        rows.append({
            "video": vid.name,
            "profile": args.profile,
            "frames": cnt,
            "mean_detections": float(np.mean(det_counts)) if det_counts else 0.0,
            "std_detections": float(np.std(det_counts)) if det_counts else 0.0,
            "fps": fps
        })

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)

if __name__ == "__main__":
    main()
