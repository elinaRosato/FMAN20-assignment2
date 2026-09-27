#!/usr/bin/env python3
"""
Assignment 2, Section 3: classification with nearest neighbour and with a
Gaussian (MAP) model.

3.1  Nearest neighbour: the first four measurements of each class are training
     data, the last three are test data.
3.2  Gaussian MAP: each class is modelled as N(m_i, sigma_i^2) with equal
     priors, so the MAP rule reduces to picking the largest likelihood.  No
     measurements were used for training, so all seven per class are scored.

Output: assets/classification.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

THISDIR = os.path.dirname(os.path.realpath(__file__))
ASSETS = os.path.join(THISDIR, "assets")

DATA = {1: [0.4003, 0.3985, 0.3998, 0.3997, 0.4015, 0.3995, 0.3991],
        2: [0.2554, 0.3139, 0.2627, 0.3802, 0.3247, 0.3360, 0.2974],
        3: [0.5632, 0.7687, 0.0524, 0.7586, 0.4443, 0.5505, 0.6469]}
M = {1: 0.40, 2: 0.32, 3: 0.55}      # class means
S = {1: 0.01, 2: 0.05, 3: 0.20}      # class standard deviations
CLASSES = (1, 2, 3)


def likelihood(x, k):
    """p(x | class k) for the Gaussian model."""
    return norm.pdf(x, M[k], S[k])


def map_predict(x):
    """MAP class.  Priors are equal, so the prior cancels and we compare
    likelihoods directly."""
    return max(CLASSES, key=lambda k: likelihood(x, k))


def nn_predict(x, train):
    """Nearest-neighbour class: the label of the closest training sample."""
    return min(train, key=lambda t: abs(x - t[0]))[1]


def nearest_neighbour():
    train = [(x, k) for k in CLASSES for x in DATA[k][:4]]
    test = [(x, k) for k in CLASSES for x in DATA[k][4:]]
    print(f"--- 3.1 nearest neighbour ({len(train)} train, {len(test)} test) ---")
    correct = 0
    for x, true in test:
        dist, xt, pk = min((abs(x - xt), xt, kt) for xt, kt in train)
        pred = pk
        correct += pred == true
        flag = "" if pred == true else "   <-- error"
        print(f"  x={x:.4f} true={true}  nearest {xt:.4f} (class {pk}) at "
              f"distance {dist:.4f}  -> {pred}{flag}")
    print(f"  {correct} of {len(test)} correct\n")
    return correct, len(test)


def gaussian_map():
    print("--- 3.2 Gaussian MAP (all 7 per class) ---")
    print("  peak heights 1/(sigma*sqrt(2pi)): "
          + ", ".join(f"class {k}: {1/(S[k]*np.sqrt(2*np.pi)):.2f}" for k in CLASSES))
    correct = total = 0
    for k in CLASSES:
        for x in DATA[k]:
            L = [likelihood(x, j) for j in CLASSES]
            pred = map_predict(x)
            total += 1
            correct += pred == k
            flag = "" if pred == k else "   <-- error"
            print(f"  x={x:.4f} true={k}  p={L[0]:8.4f} {L[1]:8.4f} {L[2]:8.4f}"
                  f"  -> {pred}{flag}")
    print(f"  {correct} of {total} correct\n")
    return correct, total


def boundaries():
    """Locate the decision boundaries by scanning the MAP label."""
    xs = np.linspace(0, 1, 200001)
    lab = np.array([map_predict(x) for x in xs])
    bs = [( (xs[i] + xs[i+1]) / 2, lab[i], lab[i+1])
          for i in np.where(np.diff(lab) != 0)[0]]
    print("--- decision boundaries ---")
    for b, a, c in bs:
        print(f"  x = {b:.4f}: class {a} -> class {c}")
    return [b for b, _, _ in bs]


def figure(bs):
    os.makedirs(ASSETS, exist_ok=True)
    x = np.linspace(0, 0.9, 4001)
    fig, ax = plt.subplots(figsize=(9, 4.2))

    # shade the MAP decision regions
    edges = [0] + list(bs) + [0.9]
    shade = {1: "#cfe3f7", 2: "#d7f0d7", 3: "#fbe0cf"}
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = map_predict((lo + hi) / 2)
        ax.axvspan(lo, hi, color=shade[k], zorder=0)
        ax.text((lo + hi) / 2, 1.2e2, f"decide {k}", ha="center", fontsize=8)

    for k, c in zip(CLASSES, ("C0", "C2", "C1")):
        ax.semilogy(x, likelihood(x, k), color=c, lw=1.8,
                    label=f"$p(x\\mid\\omega_{k})$, $m={M[k]}$, $\\sigma={S[k]}$")
        ax.plot(DATA[k], [10 ** (-2.55 + 0.30 * k)] * 7, "o", color=c, ms=6,
                mec="k", mew=.4, clip_on=False)
    for xe in (0.3802, 0.4443):
        ax.axvline(xe, color="k", ls=":", lw=1)
    ax.annotate("0.3802\n(class 2, MAP says 1)", xy=(0.3802, 4.5e-3),
                xytext=(0.10, 3e-4), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=.8))
    ax.annotate("0.4443\n(class 3, NN says 1)", xy=(0.4443, 8.9e-3),
                xytext=(0.55, 2.5e-4), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=.8))

    ax.set_ylim(1e-4, 6e2)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("$x$")
    ax.set_ylabel("likelihood (log scale)")
    ax.legend(loc="center right", fontsize=8, framealpha=.95)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "classification.png"), dpi=130)
    print("\nWrote assets/classification.png")


if __name__ == "__main__":
    for k in CLASSES:
        v = np.array(DATA[k])
        print(f"class {k}: mean {v.mean():.4f}  sample sd {v.std(ddof=1):.4f}"
              f"   (model: m={M[k]}, sigma={S[k]})")
    print()
    nearest_neighbour()
    gaussian_map()
    figure(boundaries())
