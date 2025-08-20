"""
Create dataset-level plots:
- Top XKCD colors (bar)
- Class distribution (bar)
- Class-color co-occurrence heatmap
"""

import argparse, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="results/colors_summary.csv")
    ap.add_argument("--class_names", nargs="+", default=['sunglass','hat','jacket','shirt','pants','shorts','skirt','dress','bag','shoe'])
    ap.add_argument("--outdir", type=str, default="results/plots")
    ap.add_argument("--topk", type=int, default=15)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    # 1) Top XKCD colors
    top = (df.groupby("xkcd_name")["pixel_count"].sum().sort_values(ascending=False).head(args.topk))
    plt.figure(figsize=(10,5))
    top.plot(kind="bar")
    plt.title(f"Top {args.topk} Dominant XKCD Colors")
    plt.ylabel("Aggregated pixel count")
    plt.tight_layout(); plt.savefig(out / "top_xkcd_colors.png"); plt.close()

    # 2) Class distribution (by instances)
    cls_counts = df.groupby("class_id")["image"].nunique().reindex(range(len(args.class_names)), fill_value=0)
    plt.figure(figsize=(10,5))
    plt.bar([args.class_names[i] for i in cls_counts.index], cls_counts.values)
    plt.xticks(rotation=45); plt.ylabel("Images with detections")
    plt.title("Outfit Class Coverage")
    plt.tight_layout(); plt.savefig(out / "class_distribution.png"); plt.close()

    # 3) Heatmap class vs top colors
    top_names = list(top.index)
    mat = np.zeros((len(args.class_names), len(top_names)), dtype=int)
    for i in range(len(args.class_names)):
        subset = df[df["class_id"] == i]
        counts = subset.groupby("xkcd_name")["pixel_count"].sum()
        for j, cname in enumerate(top_names):
            mat[i, j] = int(counts.get(cname, 0))
    plt.figure(figsize=(12,6))
    sns.heatmap(mat, annot=False, xticklabels=top_names, yticklabels=args.class_names, cmap="YlGnBu")
    plt.xticks(rotation=45, ha="right")
    plt.title("Class vs Dominant XKCD Colors (pixel-weighted)")
    plt.tight_layout(); plt.savefig(out / "class_color_heatmap.png"); plt.close()

    print("Saved plots to", out)

if __name__ == "__main__":
    main()
