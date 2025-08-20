"""
Train YOLO-Seg using Ultralytics.
"""

import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="train.yaml")
    ap.add_argument("--model", type=str, default="yolov8s-seg.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default=None)  # e.g., "0"
    ap.add_argument("--project", type=str, default="runs/segment")
    ap.add_argument("--name", type=str, default="train")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        verbose=True,
        task="segment",
    )

if __name__ == "__main__":
    main()
