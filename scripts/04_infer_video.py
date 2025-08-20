"""
Run segmentation on a video and save annotated output.
"""

import argparse, cv2
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/videos/out.mp4")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): 
        raise RuntimeError("Cannot open video")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(args.out, fourcc, fps, (w,h))

    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.predict(source=frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        vis = res.plot()
        out.write(vis)

    cap.release(); out.release()
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
