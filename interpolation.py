#!/usr/bin/env python3
"""
Assignment 2, Section 2: Interpolation.

The signal is f = [1 2 2 4 7 6 5] sampled at i = 1,...,7.  We build

  (a) the linear interpolant F_lin,
  (b) the kernel g that reproduces it through F_g(x) = sum_i g(x-i) f(i),
  (c) the interpolant F_g for the piecewise cubic g given in the assignment,

and check continuity and differentiability analytically.

Outputs: assets/interp_linear.png, assets/interp_kernel.png, assets/interp_Fg.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np

THISDIR = os.path.dirname(os.path.realpath(__file__))
ASSETS = os.path.join(THISDIR, "assets")

F = np.array([1, 2, 2, 4, 7, 6, 5], dtype=float)   # f(1) ... f(7)
I = np.arange(1, 8)


# ----------------------------------------------------------------- kernels
def g_tri(x):
    """Triangle (hat) kernel: the g that makes F_g the linear interpolant."""
    s = np.abs(np.asarray(x, float))
    return np.where(s <= 1, 1 - s, 0.0)


def g_cub(x):
    """The piecewise kernel given in part (c).

    On 1 < |x| <= 2 the cubic |x|^3 - 5|x|^2 + 8|x| - 4 factors as
    (|x|-1)(|x|-2)^2, which is the form that makes the joins easy to read:
    the simple root at |x|=1 makes g vanish there, and the double root at
    |x|=2 makes both g and g' vanish there.
    """
    s = np.abs(np.asarray(x, float))
    return np.where(s <= 1, 6 / 7 * (1 - s ** 2),
           np.where(s <= 2, -6 / 7 * (s - 1) * (s - 2) ** 2, 0.0))


def interpolate(x, kernel):
    """F_g(x) = sum_{i=1}^{7} g(x - i) f(i)."""
    x = np.asarray(x, float)
    return sum(kernel(x - i) * F[i - 1] for i in I)


# ----------------------------------------------------------------- analysis
def report_numbers():
    print("(a) slopes of the linear interpolant on each interval:")
    d = np.diff(F)
    for i, s in enumerate(d):
        print(f"      [{i+1},{i+2}]: {s:+.0f}")
    print("    interior nodes, left slope vs right slope:")
    for i in range(1, 6):
        l, r = d[i - 1], d[i]
        tag = "differentiable (slopes agree)" if l == r else "kink"
        print(f"      x={i+1}: {l:+.0f} vs {r:+.0f}   -> {tag}")

    print("\n(c) the kernel g:")
    print(f"      g(0)  = {g_cub(0):.6f}   (= 6/7, not 1)")
    print(f"      g(1)  = {g_cub(1):.6f},  g(2) = {g_cub(2):.6f}")
    # one-sided derivatives at the joins, from the two pieces
    print("      g'(1-) = -12/7 = %.6f   from  d/ds [6/7 (1-s^2)]" % (-12 / 7))
    print("      g'(1+) =  -6/7 = %.6f   from  d/ds [-6/7 (s-1)(s-2)^2]" % (-6 / 7))
    print("      -> jump of 6/7, so g is NOT differentiable at |x| = 1")
    print("      g'(2-) =  0,  g'(2+) = 0  -> g IS differentiable at |x| = 2")

    print("\n    values of F_g at the sample points:")
    for i in I:
        print(f"      F_g({i}) = {interpolate(i, g_cub):.4f}   f({i}) = {F[i-1]:.0f}"
              f"   ratio {interpolate(i, g_cub)/F[i-1]:.4f}")

    print("\n    jump in F_g' at each interior integer, (6/7)[f(m-1)+f(m+1)]:")
    h = 1e-6
    for m in range(2, 7):
        pred = 6 / 7 * (F[m - 2] + F[m])
        num = ((interpolate(m + h, g_cub) - interpolate(m, g_cub)) / h
               - (interpolate(m, g_cub) - interpolate(m - h, g_cub)) / h)
        print(f"      m={m}: predicted {pred:7.4f}   numerical {num:7.4f}")


# ----------------------------------------------------------------- figures
def figures():
    os.makedirs(ASSETS, exist_ok=True)
    x = np.linspace(1, 7, 2001)

    # (a) the linear interpolant
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(I, F, "-", color="C0", lw=2, label=r"$F_{lin}(x)$")
    ax.plot(I, F, "o", color="C0", ms=7, label="samples $f(i)$")
    ax.plot(6, F[5], "s", color="C3", ms=11, mfc="none", mew=2,
            label="differentiable node")
    ax.set_xlabel("$x$"); ax.set_ylabel("$F_{lin}(x)$")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "interp_linear.png"), dpi=130)

    # (b) the triangle kernel
    t = np.linspace(-3, 3, 3001)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(t, g_tri(t), color="C0", lw=2)
    ax.axhline(0, color="k", lw=.8)
    ax.grid(alpha=.3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$g(x)$")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "interp_kernel_tri.png"), dpi=130)

    # (c) the given piecewise cubic kernel
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(t, g_cub(t), color="C1", lw=2)
    ax.plot([-1, 1], [0, 0], "o", color="C3", ms=6)
    ax.annotate("kink at $|x|=1$", xy=(1, 0), xytext=(1.3, .45),
                arrowprops=dict(arrowstyle="->", color="C3"), color="C3")
    ax.annotate("negative lobe", xy=(1.5, g_cub(1.5)), xytext=(1.6, -.34),
                arrowprops=dict(arrowstyle="->", color="C2"), color="C2")
    ax.axhline(0, color="k", lw=.8)
    ax.grid(alpha=.3)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$g(x)$")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "interp_kernel_cub.png"), dpi=130)

    # (c) the two interpolants together
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(I, F, "--", color="C0", lw=1.6, label=r"$F_{lin}(x)$")
    ax.plot(x, interpolate(x, g_cub), color="C1", lw=2, label=r"$F_g(x)$")
    ax.plot(I, F, "o", color="C0", ms=7, label="samples $f(i)$")
    ax.plot(I, interpolate(I, g_cub), "x", color="C1", ms=9, mew=2,
            label=r"$F_g(i) = \frac{6}{7} f(i)$")
    ax.set_xlabel("$x$"); ax.set_ylabel("value")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "interp_Fg.png"), dpi=130)
    print("\nWrote the three figures to assets/")


if __name__ == "__main__":
    report_numbers()
    figures()
