"""
Extract dominant colors per detected mask using KMeans, map to XKCD names,
optionally apply a 3x3 calibration matrix, and write a CSV summary.
"""

import argparse, json
from pathlib import Path
import numpy as np
import cv2, pandas as pd
from sklearn.cluster import KMeans
from ultralytics import YOLO

def load_xkcd_colors(path_json):
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    # returns (N,3) float RGB and list of names
    rgbs = []
    names = []
    for e in data:
        h = e["hex"].lstrip("#")
        rgb = tuple(int(h[i:i+2], 16) for i in (0,2,4))
        rgbs.append(rgb); names.append(e["name"])
    return np.array(rgbs, dtype=np.float32), names

def apply_calibration(rgb, M):
    # rgb: (N,3) uint8; M: (3,3)
    r = rgb.astype(np.float32)
    out = r @ M.T
    return np.clip(out, 0, 255).astype(np.uint8)

def dominant_colors_kmeans(pixels, k=3, n_init=10):
    if len(pixels) < k: k = max(1, len(pixels))
    km = KMeans(n_clusters=k, n_init=n_init, random_state=42)
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_.astype(np.float32)
    counts = np.bincount(labels, minlength=k)
    order = np.argsort(-counts)
    return centers[order], counts[order]

def nearest_xkcd(rgb, xkcd_rgbs):
    # rgb: (3,), xkcd_rgbs: (N,3)
    d = np.linalg.norm(xkcd_rgbs - rgb[None,:], axis=1)
    j = int(np.argmin(d))
    return j, float(d[j])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--images", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="results/colors_summary.csv")
    ap.add_argument("--xkcd_json", type=str, default="data/xkcd_colors_950.json")
    ap.add_argument("--calib_npy", type=str, default="results/calibration_matrix.npy")
    ap.add_argument("--person_conf", type=float, default=0.75)
    ap.add_argument("--color_conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True, parents=True)

    xkcd_rgbs, xkcd_names = load_xkcd_colors(args.xkcd_json)
    if Path(args.calib_npy).exists():
        M = np.load(args.calib_npy)  # (3,3)
    else:
        M = np.eye(3, dtype=np.float32)

    model = YOLO(args.weights)
    rows = []
    img_paths = [p for p in Path(args.images).glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]]
    print(f"Found {len(img_paths)} images")

    for p in img_paths:
        image = cv2.imread(str(p))
        if image is None: continue
        res = model.predict(source=image, conf=args.person_conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        if res.masks is None:
            continue

        for mask_t, cls_t, conf_t in zip(res.masks.data, res.boxes.cls, res.boxes.conf):
            if float(conf_t) < args.color_conf:
                continue
            cls_id = int(cls_t)
            bin_mask = (mask_t.cpu().numpy() * 255).astype(np.uint8)
            bin_mask = cv2.resize(bin_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            masked = cv2.bitwise_and(image, image, mask=bin_mask)
            pixels = masked[bin_mask > 0].reshape(-1, 3)
            if pixels.size == 0: 
                continue

            # apply calibration
            pixels = apply_calibration(pixels, M)
            centers, counts = dominant_colors_kmeans(pixels, k=args.topk)
            for rank, (c, cnt) in enumerate(zip(centers, counts), start=1):
                j, dist = nearest_xkcd(c, xkcd_rgbs)
                rows.append({
                    "image": p.name,
                    "class_id": cls_id,
                    "rank": rank,
                    "rgb_r": float(c[0]), "rgb_g": float(c[1]), "rgb_b": float(c[2]),
                    "xkcd_name": xkcd_names[j],
                    "xkcd_dist": dist,
                    "pixel_count": int(cnt)
                })

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("Saved:", args.out_csv)

if __name__ == "__main__":
    main()
