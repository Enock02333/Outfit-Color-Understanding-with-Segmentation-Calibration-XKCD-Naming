"""
Fit a 3x3 color calibration matrix M that maps measured RGB -> reference RGB.
Usage:
  python 09_calibration_fit.py --measured csv_measured.csv --reference csv_reference.csv --out results/calibration_matrix.npy
Where each CSV has columns: R,G,B (0-255), rows correspond (same number/order).
"""

import argparse, numpy as np, pandas as pd
from pathlib import Path

def solve_color_matrix(X, Y):
    """
    Solve M in Y ≈ X @ M^T using least squares (with bias augmentation optional).
    X: (N,3) measured, Y: (N,3) reference
    """
    # Optionally include bias term (comment/uncomment next two lines)
    # X_aug = np.hstack([X, np.ones((X.shape[0],1))])  # (N,4)
    # M = np.linalg.lstsq(X_aug, Y, rcond=None)[0].T   # (3,4) -> last column is bias
    M = np.linalg.lstsq(X, Y, rcond=None)[0].T         # (3,3)
    return M

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured", type=str, required=True)
    ap.add_argument("--reference", type=str, required=True)
    ap.add_argument("--out", type=str, default="results/calibration_matrix.npy")
    args = ap.parse_args()

    X = pd.read_csv(args.measured)[["R","G","B"]].to_numpy(dtype=np.float32)
    Y = pd.read_csv(args.reference)[["R","G","B"]].to_numpy(dtype=np.float32)

    if len(X) != len(Y):
        raise ValueError("Measured and reference must have same number of rows")

    M = solve_color_matrix(X, Y)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, M.astype(np.float32))
    print("Saved calibration matrix to:", args.out)
    print("Matrix:\n", M)

if __name__ == "__main__":
    main()
