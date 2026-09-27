#!/usr/bin/env python3
"""
Assignment 2, Section 6: character classification of a 5x3 binary image.

Three templates ('B', '0', '8') with priors 0.3, 0.45, 0.25.  The noise is
ASYMMETRIC: a truly white pixel is measured as black with probability 0.35,
while a truly black pixel is measured as white with probability 0.2.  Because
the two error rates differ, the likelihood no longer depends only on the
number of mismatches; the two kinds of mismatch must be counted separately.

Output: assets/character_classification.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np

THISDIR = os.path.dirname(os.path.realpath(__file__))
ASSETS = os.path.join(THISDIR, "assets")

# 1 = black, 0 = white
TEMPLATES = {
    "B": np.array([[1, 1, 0], [1, 0, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0]]),
    "0": np.array([[0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0]]),
    "8": np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]]),
}
PRIOR = {"B": 0.30, "0": 0.45, "8": 0.25}
OBS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])

# P(observed | true).  Rows: true white, true black.  Cols: obs white, obs black.
P_OBS_GIVEN_TRUE = np.array([[0.65, 0.35],     # truly white
                             [0.20, 0.80]])    # truly black


def counts(name):
    """The four pixel categories: (white->white, white->black,
    black->white, black->black)."""
    t = TEMPLATES[name]
    return (int(((t == 0) & (OBS == 0)).sum()),
            int(((t == 0) & (OBS == 1)).sum()),
            int(((t == 1) & (OBS == 0)).sum()),
            int(((t == 1) & (OBS == 1)).sum()))


def likelihood(name):
    a, b, c, d = counts(name)
    return 0.65 ** a * 0.35 ** b * 0.20 ** c * 0.80 ** d


def posteriors():
    num = {n: likelihood(n) * PRIOR[n] for n in TEMPLATES}
    px = sum(num.values())
    return {n: num[n] / px for n in TEMPLATES}, num, px


def figure():
    os.makedirs(ASSETS, exist_ok=True)
    post, _, _ = posteriors()
    fig, ax = plt.subplots(1, 4, figsize=(9, 3.4))
    ax[0].imshow(1 - OBS, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("observed $x$")
    for a, n in zip(ax[1:], TEMPLATES):
        t = TEMPLATES[n]
        a.imshow(1 - t, cmap="gray", vmin=0, vmax=1)
        # white template pixel seen as black: costs 0.35
        wb = np.argwhere((t == 0) & (OBS == 1))
        # black template pixel seen as white: costs 0.20
        bw = np.argwhere((t == 1) & (OBS == 0))
        a.plot(wb[:, 1], wb[:, 0], "o", mfc="none", mec="red", ms=13, mew=2.2)
        a.plot(bw[:, 1], bw[:, 0], "x", color="red", ms=13, mew=2.2)
        a.set_title(f"'{n}':  $P={post[n]*100:.2f}\\%$")
    for a in ax:
        a.set_xticks(np.arange(-.5, 3, 1), minor=True)
        a.set_yticks(np.arange(-.5, 5, 1), minor=True)
        a.grid(which="minor", color="C7", lw=1)
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "character_classification.png"), dpi=130)
    print("\nWrote assets/character_classification.png")


if __name__ == "__main__":
    print("observed x (1 = black):")
    for r in OBS:
        print("   " + " ".join("#" if v else "." for v in r))
    post, num, px = posteriors()
    print("\n            w->w  w->b  b->w  b->b")
    print("  class  a(.65) b(.35) c(.20) d(.80)   P(x|w)        prior   P(x|w)P(w)     posterior")
    for n in TEMPLATES:
        a, b, c, d = counts(n)
        print(f"   '{n}'     {a:2d}    {b:2d}    {c:2d}    {d:2d}    "
              f"{likelihood(n):.6e}  {PRIOR[n]:.2f}   {num[n]:.6e}   {post[n]*100:6.2f}%")
    print(f"\n  P(x) = {px:.6e}")
    print(f"  MAP = '{max(post, key=lambda n: post[n])}'")

    print("\n  mismatch counts (b+c) for comparison:")
    for n in TEMPLATES:
        a, b, c, d = counts(n)
        print(f"   '{n}': total mismatches {b+c}  (white->black {b}, black->white {c})")
    figure()
