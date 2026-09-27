#!/usr/bin/env python3
"""
Assignment 2, Section 4: image classification of a 2x2 binary scene.

Three scenes are possible, with given a priori probabilities.  One of them is
observed through a noisy channel that flips each pixel independently with
probability epsilon.  Given the observed image we compute the posterior of
each scene and the MAP estimate, for epsilon = 10% and 50%.

Output: assets/image_classification.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from fractions import Fraction

THISDIR = os.path.dirname(os.path.realpath(__file__))
ASSETS = os.path.join(THISDIR, "assets")

# scenes flattened in the order top-left, top-right, bottom-left, bottom-right
SCENES = {1: (0, 1, 1, 0), 2: (1, 0, 0, 1), 3: (1, 1, 1, 0)}
PRIOR = {1: Fraction(1, 4), 2: Fraction(1, 2), 3: Fraction(1, 4)}
OBSERVED = (0, 1, 1, 1)


def mismatches(k):
    """Number of pixels where scene k disagrees with the observed image."""
    return sum(a != b for a, b in zip(SCENES[k], OBSERVED))


def posteriors(eps):
    """Exact posteriors P(Y=k | x) for a given noise level."""
    e = Fraction(eps)
    num = {k: PRIOR[k] * e ** mismatches(k) * (1 - e) ** (4 - mismatches(k))
           for k in SCENES}
    px = sum(num.values())                       # P(x), the normalising sum
    return {k: num[k] / px for k in SCENES}, num, px


def crossover():
    """Noise level at which the MAP answer switches from scene 1 to scene 2.

    Scene 1 needs 1 flip and scene 2 needs 3, so the two are equally good when
        (1/4) e (1-e)^3 = (1/2) e^3 (1-e)   <=>   (1-e)^2 = 2 e^2,
    giving 1-e = sqrt(2) e and therefore e = 1/(1+sqrt(2)) = sqrt(2)-1.
    """
    return np.sqrt(2) - 1


def figure():
    os.makedirs(ASSETS, exist_ok=True)
    e = np.linspace(1e-4, 0.5, 2001)
    k = {j: mismatches(j) for j in SCENES}
    num = {j: float(PRIOR[j]) * e ** k[j] * (1 - e) ** (4 - k[j]) for j in SCENES}
    tot = sum(num.values())

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for j, c in zip(SCENES, ("C0", "C2", "C1")):
        ax.plot(e, num[j] / tot, color=c, lw=2,
                label=f"scene {j} ($k={k[j]}$ mismatches, prior {PRIOR[j]})")
    xc = crossover()
    ax.axvline(xc, color="k", ls="--", lw=1)
    ax.annotate(rf"$\epsilon=\sqrt{{2}}-1\approx{xc:.3f}$" "\nMAP switches 1 to 2",
                xy=(xc, .55), xytext=(0.18, .72), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=.8))
    for xv in (0.1, 0.5):
        ax.axvline(xv, color="grey", ls=":", lw=1)
    ax.set_xlabel(r"noise level $\epsilon$")
    ax.set_ylabel(r"posterior $P(Y=k \mid x)$")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "image_classification.png"), dpi=130)
    print("Wrote assets/image_classification.png")


if __name__ == "__main__":
    print("observed image x =", OBSERVED)
    for j in SCENES:
        print(f"  scene {j} = {SCENES[j]}  prior {PRIOR[j]}  "
              f"mismatches k = {mismatches(j)}")
    for eps in ("1/10", "1/2"):
        post, num, px = posteriors(eps)
        print(f"\n--- epsilon = {eps} ---")
        for j in SCENES:
            print(f"  scene {j}: P(x|Y={j})P(Y={j}) = {str(num[j]):>12s}"
                  f" = {float(num[j]):.6f}   posterior {float(post[j])*100:6.2f}%"
                  f"  = {post[j]}")
        print(f"  P(x) = {px} = {float(px):.6f}")
        print(f"  MAP = scene {max(post, key=lambda j: post[j])}")
    print(f"\ncrossover at epsilon = {crossover():.6f}")
    figure()
