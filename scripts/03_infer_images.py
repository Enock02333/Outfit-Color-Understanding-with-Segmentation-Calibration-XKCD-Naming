"""
Batch infer on images; saves overlays and raw predictions.
"""

import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--images", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/images")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    img_paths = [p for p in Path(args.images).glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]]
    print(f"Found {len(img_paths)} images")

    for p in img_paths:
        res = model.predict(source=str(p), conf=args.conf, iou=args.iou, imgsz=args.imgsz, save=False, verbose=False)[0]
        im = res.plot()  # overlay masks/boxes/labels
        cv2.imwrite(str(out_dir / f"{p.stem}_pred.png"), im)

    print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()
