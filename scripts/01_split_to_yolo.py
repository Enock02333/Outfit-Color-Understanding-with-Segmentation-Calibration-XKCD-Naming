"""
Split a folder of images+labels into YOLO-Seg structure:
data/yolo_seg/
  images/{train,val,test}
  labels/{train,val,test}
Assumes labels are in a parallel folder or same name .txt per image.
"""

import argparse, random, shutil
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, required=True)
    ap.add_argument("--labels", type=str, required=True)
    ap.add_argument("--out", type=str, default="data/yolo_seg")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    img_dir, lbl_dir = Path(args.images), Path(args.labels)
    out_root = Path(args.out)

    for split in ["train","val","test"]:
        (out_root / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (out_root / f"labels/{split}").mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(args.train * n)
    n_val = int(args.val * n)
    splits = {
        "train": imgs[:n_train],
        "val": imgs[n_train:n_train+n_val],
        "test": imgs[n_train+n_val:],
    }

    for split, paths in splits.items():
        for p in paths:
            lbl = lbl_dir / f"{p.stem}.txt"
            if not lbl.exists(): 
                continue
            shutil.copy2(p, out_root / f"images/{split}/{p.name}")
            shutil.copy2(lbl, out_root / f"labels/{split}/{lbl.name}")

    print("Done:", {k: len(v) for k,v in splits.items()})

if __name__ == "__main__":
    main()
