#!/usr/bin/env python3
"""
Assignment 2, Section 7: analysis of the feature extractor.

Runs segment2feature (defined in
inl2_to_students/garment_recognition_project/assignment2.py) over the whole
short1 dataset, reproduces the benchmark's two-dimensional projection as a
saved figure, shows example segments with their feature vectors, and measures
how separable the classes actually are with leave-one-out nearest neighbour.

Outputs: assets/garment_projection.png, assets/garment_examples.png
"""

import glob
import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

THISDIR = os.path.dirname(os.path.realpath(__file__))
PROJECT = os.path.join(THISDIR, "inl2_to_students", "garment_recognition_project")
DATADIR = os.path.join(PROJECT, "datasets", "short1")
ASSETS = os.path.join(THISDIR, "assets")
sys.path.insert(0, PROJECT)

from assignment2 import segment2feature, FEATURE_NAMES     # noqa: E402

# the label characters are Fashion-MNIST class indices, not digits
CLASS = {"0": "T-shirt", "1": "Trouser", "2": "Pullover", "3": "Dress",
         "4": "Coat", "5": "Sandal", "6": "Shirt", "7": "Sneaker",
         "8": "Bag", "9": "Boot"}
# the two groups of garments that share a silhouette
UPPER, FOOT = set("0246"), set("579")


def load_all():
    """Feature vectors and labels for every ground-truth segment."""
    X, Y, segs = [], [], []
    for sn, gn in zip(sorted(glob.glob(os.path.join(DATADIR, "*.npy"))),
                      sorted(glob.glob(os.path.join(DATADIR, "*.txt")))):
        S, gt = np.load(sn), open(gn).read().strip()
        for i, ch in enumerate(gt):
            X.append(segment2feature(S[i]).ravel())
            Y.append(ch)
            segs.append(S[i])
    return np.array(X), np.array(Y), segs


def leave_one_out(X, Y):
    """1-NN accuracy on standardised features, and the confusion list."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    pred = []
    for i in range(len(Z)):
        d = np.linalg.norm(Z - Z[i], axis=1)
        d[i] = np.inf
        pred.append(Y[d.argmin()])
    return np.array(pred)


def projection_figure(X, Y):
    """The benchmark's 2-D projection: the first two left singular vectors."""
    # the benchmark works with a features-by-samples matrix
    A = X.T
    U, _, _ = np.linalg.svd(A)
    P = U[:, :2].T @ A
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for c in sorted(CLASS):
        m = Y == c
        if not m.any():
            continue
        col = "C0" if c in UPPER else "C1" if c in FOOT else "C2"
        ax.scatter(P[0, m], P[1, m], c=col, s=10, alpha=.5)
        for x, y in zip(P[0, m], P[1, m]):
            ax.text(x, y, c, color=col, fontsize=9, ha="center", va="center")
    ax.set_xlabel("first singular direction")
    ax.set_ylabel("second singular direction")
    ax.set_title("2-D projection of the 10-D feature vectors\n"
                 "blue = upper body (0,2,4,6)   orange = footwear (5,7,9)   "
                 "green = trouser (1), dress (3), bag (8)", fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "garment_projection.png"), dpi=130)


def examples_figure(X, Y, segs):
    """Two garments of the same class and two of different classes."""
    want = [("6", 0), ("6", 1), ("1", 0), ("5", 0)]
    picks = []
    for c, k in want:
        idx = np.where(Y == c)[0]
        picks.append(idx[k])
    fig, ax = plt.subplots(2, 4, figsize=(12, 5),
                           gridspec_kw={"height_ratios": [1.1, 1]})
    for j, i in enumerate(picks):
        s = segs[i]
        ys, xs = np.nonzero(s)
        ax[0, j].imshow(s[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
                        cmap="gray")
        ax[0, j].set_title(f"class {Y[i]} ({CLASS[Y[i]]})", fontsize=10)
        ax[0, j].axis("off")
        ax[1, j].barh(range(len(FEATURE_NAMES)), X[i], color="C0")
        ax[1, j].set_yticks(range(len(FEATURE_NAMES)))
        ax[1, j].set_yticklabels(FEATURE_NAMES if j == 0 else [], fontsize=8)
        ax[1, j].set_xlim(-0.7, 1.05)
        ax[1, j].axvline(0, color="k", lw=.6)
        ax[1, j].invert_yaxis()
        ax[1, j].grid(alpha=.3, axis="x")
    fig.suptitle("Two shirts from different images (left) and two other "
                 "classes (right), with their feature vectors", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "garment_examples.png"), dpi=130)


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    X, Y, segs = load_all()
    print(f"{X.shape[0]} segments, {X.shape[1]} features\n")

    print(f"{'feature':12s} {'min':>6s} {'max':>6s}   class means 0..9")
    for j, nm in enumerate(FEATURE_NAMES):
        mu = [X[Y == c, j].mean() if (Y == c).any() else np.nan
              for c in sorted(CLASS)]
        print(f"{nm:12s} {X[:, j].min():6.2f} {X[:, j].max():6.2f}   "
              + " ".join(f"{m:5.2f}" for m in mu))

    pred = leave_one_out(X, Y)
    print(f"\nleave-one-out 1-NN: {(pred == Y).sum()}/{len(Y)} "
          f"= {100 * (pred == Y).mean():.0f}% correct")

    print("\nper-class recall:")
    for c in sorted(CLASS):
        m = Y == c
        if m.any():
            print(f"  {c} {CLASS[c]:9s} {(pred[m] == c).sum()}/{m.sum()}")

    print("\nmost common confusions:")
    cc = Counter((t, p) for t, p in zip(Y, pred) if t != p)
    for (t, p), n in cc.most_common(6):
        print(f"  {CLASS[t]:9s} -> {CLASS[p]:9s}  x{n}")

    for nm, grp in (("upper body", UPPER), ("footwear", FOOT)):
        m = np.isin(Y, list(grp))
        inside = sum(1 for t, p in zip(Y[m], pred[m]) if p in grp and p != t)
        print(f"\n{nm}: {(pred[m] == Y[m]).sum()}/{m.sum()} correct, "
              f"{inside} of the {m.sum() - (pred[m] == Y[m]).sum()} errors "
              f"stay inside the group")

    projection_figure(X, Y)
    examples_figure(X, Y, segs)
    print("\nWrote assets/garment_projection.png and assets/garment_examples.png")
