"""
Live webcam/USB/Canon (as UVC) demo with segmentation overlay.
- Use --src 0 for default webcam, or pass RTSP/HTTP if your device exposes it.
"""

import argparse, cv2
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--src", type=str, default="0")  # index or URL
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.weights)
    src = int(args.src) if args.src.isdigit() else args.src
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): 
        raise RuntimeError("Cannot open camera")

    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.predict(source=frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        vis = res.plot()
        cv2.imshow("YOLO-Seg Live", vis)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
