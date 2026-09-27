#!/usr/bin/env python3
"""
Assignment 2, Section 5: line classification in a 4x4 binary image.

The correct scene is all white except one fully black column.  Every pixel is
observed wrongly with probability epsilon, in both directions, so a candidate
scene disagreeing with the observation in k of the 16 pixels has likelihood
epsilon^k (1-epsilon)^(16-k).  The prior is 0.3 for the outer columns 1 and 4
and 0.2 for the inner columns 2 and 3.

Output: assets/line_classification.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np

THISDIR = os.path.dirname(os.path.realpath(__file__))
ASSETS = os.path.join(THISDIR, "assets")

N = 4
EPS = 0.2
PRIOR = {1: 0.3, 2: 0.2, 3: 0.2, 4: 0.3}

# observed image: 1 = black, 0 = white.  Black at (1,1), (2,2), (3,3), (4,2).
OBS = np.zeros((N, N), dtype=int)
for r, c in [(1, 1), (2, 2), (3, 3), (4, 2)]:
    OBS[r - 1, c - 1] = 1


def scene(col):
    """The noise-free scene with the black line in the given column."""
    s = np.zeros((N, N), dtype=int)
    s[:, col - 1] = 1
    return s


def mismatches(col):
    return int((scene(col) != OBS).sum())


def posteriors(eps=EPS):
    num = {c: PRIOR[c] * eps ** mismatches(c) * (1 - eps) ** (N * N - mismatches(c))
           for c in PRIOR}
    px = sum(num.values())
    return {c: num[c] / px for c in PRIOR}, num, px


def figure():
    os.makedirs(ASSETS, exist_ok=True)
    post, _, _ = posteriors()
    fig, ax = plt.subplots(1, 5, figsize=(14, 3.2))
    ax[0].imshow(1 - OBS, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("observed image $x$")
    for a, c in zip(ax[1:], (1, 2, 3, 4)):
        s = scene(c)
        a.imshow(1 - s, cmap="gray", vmin=0, vmax=1)
        bad = np.argwhere(s != OBS)
        a.plot(bad[:, 1], bad[:, 0], "x", color="red", ms=11, mew=2.2)
        a.set_title(f"line in column {c}\n$k={mismatches(c)}$, "
                    f"$P={post[c]*100:.2f}\\%$")
    for a in ax:
        a.set_xticks(range(N)); a.set_yticks(range(N))
        a.set_xticklabels(range(1, N + 1)); a.set_yticklabels(range(1, N + 1))
        a.grid(which="major", visible=False)
        a.set_xticks(np.arange(-.5, N, 1), minor=True)
        a.set_yticks(np.arange(-.5, N, 1), minor=True)
        a.grid(which="minor", color="k", lw=1)
        a.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "line_classification.png"), dpi=130)
    print("\nWrote assets/line_classification.png")


if __name__ == "__main__":
    print("observed image (1 = black):\n", OBS, "\n")
    post, num, px = posteriors()
    print(" col  prior  mismatches k   P(x|c)=e^k(1-e)^(16-k)   P(x|c)P(c)      posterior")
    for c in PRIOR:
        lik = EPS ** mismatches(c) * (1 - EPS) ** (N * N - mismatches(c))
        print(f"  {c}    {PRIOR[c]:.1f}        {mismatches(c)}          "
              f"{lik:.6e}          {num[c]:.6e}     {post[c]*100:6.2f}%")
    print(f"\n P(x) = {px:.6e}")
    print(f" MAP = column {max(post, key=lambda c: post[c])}")

    print("\n columns 1 and 3 have the same k, so the evidence cannot separate them:")
    print(f"   posterior ratio 1:3 = {post[1]/post[3]:.4f}"
          f"   prior ratio = {PRIOR[1]/PRIOR[3]:.4f}   (identical)")
    print("\n column 2 against column 1:")
    print(f"   likelihood ratio = {(1-EPS)**2/EPS**2:.1f}  (2 fewer mismatches)")
    print(f"   prior ratio      = {PRIOR[2]/PRIOR[1]:.4f}")
    print(f"   posterior ratio  = {post[2]/post[1]:.4f}"
          f"   check: {(1-EPS)**2/EPS**2 * PRIOR[2]/PRIOR[1]:.4f}")
    figure()
