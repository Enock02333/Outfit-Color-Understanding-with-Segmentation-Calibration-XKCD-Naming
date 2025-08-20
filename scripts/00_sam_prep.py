"""
Run SAM (Segment Anything) to auto-generate polygon masks and export YOLO-Seg labels.
- Use this to bootstrap masks, then manually fix class ids if needed.
- Requires: pip install segment-anything opencv-python numpy
"""

import argparse, os, json
from pathlib import Path
import numpy as np
import cv2

# Optional SAM imports (lazy)
def _load_sam(checkpoint, model_type="vit_h"):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=2000,
    )
    return mask_generator

def polygon_to_yolo_seg(poly_xy, img_w, img_h):
    # YOLO needs normalized x,y flattened
    return [coord for (x,y) in poly_xy for coord in (x/img_w, y/img_h)]

def save_yolo_label(txt_path, segments, cls_id):
    with open(txt_path, "w") as f:
        for seg in segments:
            row = [str(cls_id)] + [f"{v:.6f}" for v in seg]
            f.write(" ".join(row) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, required=True, help="Folder with images")
    ap.add_argument("--out", type=str, required=True, help="Output YOLO labels dir")
    ap.add_argument("--sam_ckpt", type=str, required=True, help="Path to SAM .pth")
    ap.add_argument("--model_type", type=str, default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--cls", type=int, default=7, help="Default class id to assign (e.g., dress=7)")
    args = ap.parse_args()

    img_dir = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_gen = _load_sam(args.sam_ckpt, args.model_type)

    img_paths = [p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]]
    print(f"Found {len(img_paths)} images")

    for p in img_paths:
        img = cv2.imread(str(p))
        if img is None: 
            continue
        h,w = img.shape[:2]
        masks = mask_gen.generate(img)
        segments = []
        for m in masks:
            # prefer the largest polygons (first one)
            if "points" in m["segmentation"]:
                poly = m["segmentation"]["points"][0]
            else:
                # raster mask -> contour
                msk = m["segmentation"].astype(np.uint8)
                cnts,_ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts: 
                    continue
                cnt = max(cnts, key=cv2.contourArea).squeeze(1)
                poly = cnt.tolist()

            if len(poly) < 3:
                continue

            seg = polygon_to_yolo_seg(poly, w, h)
            if len(seg) >= 6:  # at least 3 points
                segments.append(seg)

        if segments:
            save_yolo_label(out_dir / f"{p.stem}.txt", segments, args.cls)

if __name__ == "__main__":
    main()
