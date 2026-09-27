#!/usr/bin/env python3
"""
Assignment 2, Section 1: Filtering.

The assignment shows one original image and five convolved versions A-E, and
asks which of six candidate filters f1...f6 produced each one.  The matching
can be argued from the structure of the filters alone, but we also verify it
numerically here: we extracted the six images from the assignment PDF (with
`pdfimages`, see initial/filter_images/), convolve the original with every
candidate filter, and correlate each result with every candidate output.

Output: assets/filtering_verification.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

THISDIR = os.path.dirname(os.path.realpath(__file__))
IMDIR = os.path.join(THISDIR, "initial", "filter_images")
ASSETS = os.path.join(THISDIR, "assets")

# The six candidate filters, exactly as given in the assignment.
FILTERS = {
    "f1": np.array([[1, -1, -1, 1]] * 3) / 6,
    "f2": np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]]) / 12,
    "f3": np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]) / 3,
    "f4": np.array([[-1, -1, -1], [-1, 17, -1], [-1, -1, -1]]) / 9,
    "f5": np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float),
    "f6": -np.ones((3, 3)) / 9,
}

# The matching we argue for in the report.
MATCH = {"A": "f6", "B": "f2", "C": "f1", "D": "f5", "E": "f3"}


def load_gray(name):
    """Read a PNG from the assignment figure and return it as a float gray image."""
    a = plt.imread(os.path.join(IMDIR, name))
    if a.ndim == 3:            # drop a possible alpha channel and average RGB
        a = a[..., :3].mean(-1)
    return a.astype(float)


def zero_mean_unit_std(a):
    """Normalise an image so that correlation ignores brightness/contrast.

    This matters because the assignment images are displayed with automatic
    grey-scale scaling, so only the *pattern* is comparable, not the absolute
    pixel values.
    """
    a = a - a.mean()
    s = a.std()
    return a / s if s > 1e-12 else a


def correlation_table(original, outputs):
    """Correlate original*f against every output image, for every filter f."""
    table = {}
    for fname, f in FILTERS.items():
        conv = convolve2d(original, f, mode="same", boundary="symm")
        row = {}
        for key, out in outputs.items():
            # crop a border off: convolution near the image edge depends on the
            # (unknown) boundary handling the assignment used
            a = zero_mean_unit_std(conv[20:-20, 20:-20]).ravel()
            b = zero_mean_unit_std(out[20:-20, 20:-20]).ravel()
            row[key] = float(np.corrcoef(a, b)[0, 1])
        table[fname] = row
    return table


def f1_vs_f3_demo():
    """Show the two tests that separate f1 from f3, on a synthetic letter E.

    f1 and f3 both have coefficient sum 0, so both produce a mid-grey image
    with only structure on it, and at small size they look alike.  They are
    separated by (1) which edges survive -- f1 is separable with [1,1,1] in y,
    an average, so it is exactly blind to horizontal edges -- and (2) whether
    the response is signed -- f3 is antisymmetric, hence a first derivative,
    so every feature gets a bright side and a dark side (relief), while f1 is
    symmetric, hence a second derivative, giving doubled outlines instead.
    """
    # numeric evidence for test 1: a purely horizontal edge
    horiz = np.zeros((7, 7))
    horiz[4:, :] = 60
    print("Test 1 -- response to a horizontal edge:")
    print("  max |u * f1| =", np.abs(convolve2d(horiz, FILTERS["f1"], "same", "symm")).max())
    print("  max |u * f3| =", np.abs(convolve2d(horiz, FILTERS["f3"], "same", "symm")).max())

    # numeric evidence for test 2: a one-pixel-wide bright vertical line
    line = np.zeros((5, 9))
    line[:, 4] = 60
    print("Test 2 -- response to a thin bright vertical line (one row):")
    for fname in ("f1", "f3"):
        row = convolve2d(line, FILTERS[fname], "same", "symm")[2]
        print(f"  u * {fname}: {np.round(row, 1)}")

    # a synthetic letter E: one vertical stroke and three horizontal bars
    E = np.zeros((40, 32))
    E[4:36, 5:10] = 1     # vertical stroke
    E[4:9, 5:26] = 1      # top bar
    E[18:23, 5:22] = 1    # middle bar
    E[31:36, 5:26] = 1    # bottom bar

    fig, ax = plt.subplots(1, 3, figsize=(10, 4.2))
    ax[0].imshow(E, cmap="gray")
    ax[0].set_title("input: letter E")
    titles = {"f1": "$*\\,f_1$: verticals only",
              "f3": "$*\\,f_3$: relief, all strokes"}
    for a, fname in zip(ax[1:], ("f1", "f3")):
        v = convolve2d(E, FILTERS[fname], mode="same", boundary="symm")
        m = np.abs(v).max()
        a.imshow(v, cmap="gray", vmin=-m, vmax=m)   # symmetric scaling: 0 -> grey
        a.set_title(titles[fname])
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "filtering_f1_vs_f3.png"), dpi=130)
    print("Wrote assets/filtering_f1_vs_f3.png")


def main():
    original = load_gray("original.png")
    outputs = {k: load_gray(f"{k}.png") for k in "ABCDE"}

    table = correlation_table(original, outputs)
    print("Correlation between (original * f) and each output image:\n")
    print("     " + "".join(f"{k:>9s}" for k in outputs))
    for fname, row in table.items():
        print(f"{fname:5s}" + "".join(f"{row[k]:9.3f}" for k in outputs))
    print("\nBest filter for each output image:")
    for k in outputs:
        best = max(FILTERS, key=lambda f: table[f][k])
        print(f"  {k}: {best}  (corr {table[best][k]:.3f})")

    # Figure: assignment output on top, our own convolution below.
    fig, ax = plt.subplots(2, 5, figsize=(14, 6))
    for j, k in enumerate("ABCDE"):
        fname = MATCH[k]
        ax[0, j].imshow(outputs[k], cmap="gray")
        ax[0, j].set_title(f"assignment: {k}")
        conv = convolve2d(original, FILTERS[fname], mode="same", boundary="symm")
        ax[1, j].imshow(conv, cmap="gray")
        ax[1, j].set_title(f"original $*$ ${fname[0]}_{fname[1]}$")
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    os.makedirs(ASSETS, exist_ok=True)
    fig.savefig(os.path.join(ASSETS, "filtering_verification.png"), dpi=110)
    print("\nWrote assets/filtering_verification.png")

    print()
    f1_vs_f3_demo()


if __name__ == "__main__":
    main()
