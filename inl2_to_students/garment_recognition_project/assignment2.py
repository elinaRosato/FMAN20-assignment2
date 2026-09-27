#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Analysis, Assignment 2, Section 7: feature extraction.

im2segment        carried over from Assignment 1 (brightness threshold plus
                  8-connected components).
segment2feature   the new part: turns one binary garment segment into a
                  10-element feature vector.

All features are computed from the segment after cropping it to its bounding
box, or from central moments, so none of them depends on where the garment sits
in the frame: the same garment shifted in x or y gives exactly the same vector.
Every feature is scaled to roughly [0,1] (the tilt to [-1,1]) so that no single
one dominates a distance computation.
"""

import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from benchmarking.benchmark_assignment2 import benchmark_assignment2


# --------------------------------------------------------------- segmentation
def find_connected_components(mask):
    """Label the 8-connected components of the True pixels in a binary mask.

    Scans column by column so components come out ordered left to right,
    which is the order the benchmark expects.
    """
    n, m = mask.shape
    labels = np.zeros(mask.shape, dtype=int)
    current_label = 0
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1)]
    for j in range(m):
        for i in range(n):
            if mask[i, j] and labels[i, j] == 0:
                current_label += 1
                queue = deque([(i, j)])
                labels[i, j] = current_label
                while queue:
                    ci, cj = queue.popleft()
                    for di, dj in offsets:
                        ni, nj = ci + di, cj + dj
                        if (0 <= ni < n and 0 <= nj < m
                                and mask[ni, nj] and labels[ni, nj] == 0):
                            labels[ni, nj] = current_label
                            queue.append((ni, nj))
    return labels, current_label


def im2segment(im, background_threshold=20, min_area=15):
    """Segment garments laid out on a dark background (from Assignment 1)."""
    labels, num = find_connected_components(im > background_threshold)
    return [labels == k for k in range(1, num + 1)
            if np.sum(labels == k) >= min_area]


# ------------------------------------------------------------------- features
FEATURE_NAMES = ["aspect", "extent", "spread_x", "spread_y", "tilt",
                 "holes", "compactness", "v_centre", "width_25", "width_75"]


def segment2feature(Si):
    """Return a 10x1 feature vector describing the shape of one segment.

    The features, in order:
      0 aspect      h/(h+w) of the bounding box.  Tall narrow garments
                    (trousers, dresses) score high, footwear scores low.
      1 extent      area / bounding-box area.  Solid garments (pullover) fill
                    their box; trousers leave a notch between the legs and
                    sandals are mostly gaps, so both score low.
      2 spread_x    standard deviation of the pixels in x, divided by
                    sqrt(area).  A normalised second central moment.
      3 spread_y    the same in y.  Together 2 and 3 describe elongation
                    independently of size.
      4 tilt        normalised mixed moment mu11/sqrt(mu20*mu02), in [-1,1].
                    Zero for a shape symmetric about a vertical axis, clearly
                    negative for footwear, which lies diagonally in the frame.
      5 holes       number of enclosed background regions, capped at 2 and
                    divided by 2.  A bag with a handle encloses a hole; almost
                    nothing else does.
      6 compactness 4*pi*area/perimeter^2, which is 1 for a disc and small for
                    a shape with a long ragged boundary such as a sandal.
      7 v_centre    height of the centre of mass within the bounding box, 0 at
                    the top and 1 at the bottom.  Separates top-heavy garments
                    from bottom-heavy ones.
      8 width_25    width of the silhouette one quarter of the way down,
                    relative to the widest row.
      9 width_75    the same three quarters of the way down.  Together 8 and 9
                    sketch the outline: a shirt is wide at the shoulders, a
                    dress widens towards the hem, a bag is narrow at the handle.

    All of them are computed either from the cropped bounding box or from
    central moments, so all are translation invariant.
    """
    S = (np.asarray(Si) > 0).astype(np.uint8)
    ys, xs = np.nonzero(S)
    if ys.size == 0:                       # empty segment: return zeros
        return np.zeros((len(FEATURE_NAMES), 1))

    # crop to the bounding box -- this is what makes the features independent
    # of where the garment sits in the image
    B = S[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = B.shape
    area = float(B.sum())

    aspect = h / (h + w)
    extent = area / (h * w)

    # central moments: measured relative to the centre of mass, so again
    # independent of position
    yy, xx = np.nonzero(B)
    cy, cx = yy.mean(), xx.mean()
    dy, dx = yy - cy, xx - cx
    mu20 = (dx ** 2).sum() / area
    mu02 = (dy ** 2).sum() / area
    mu11 = (dx * dy).sum() / area
    scale = np.sqrt(area)                  # divide by sqrt(area) to normalise
    spread_x = np.sqrt(mu20) / scale
    spread_y = np.sqrt(mu02) / scale
    tilt = mu11 / np.sqrt(mu20 * mu02 + 1e-9)

    # holes: label the background of the padded shape; one region is the
    # outside, any others are enclosed holes
    padded = np.pad(B, 1)
    _, n_background = ndimage.label(1 - padded)
    holes = min(n_background - 1, 2) / 2

    # perimeter: foreground pixels that touch the background
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    eroded = ndimage.binary_erosion(padded, cross)
    perimeter = float((padded & ~eroded).sum())
    compactness = 4 * np.pi * area / perimeter ** 2 if perimeter else 0.0

    v_centre = cy / (h - 1) if h > 1 else 0.5

    row_widths = B.sum(1).astype(float)
    widest = row_widths.max()
    width_25 = row_widths[int(0.25 * (h - 1))] / widest
    width_75 = row_widths[int(0.75 * (h - 1))] / widest

    return np.array([[aspect], [extent], [spread_x], [spread_y], [tilt],
                     [holes], [compactness], [v_centre],
                     [width_25], [width_75]], dtype=float)


if __name__ == "__main__":
    thisdir = os.path.dirname(os.path.realpath(__file__))
    datadir = os.path.join(thisdir, 'datasets', 'short1')

    im = plt.imread(os.path.join(datadir, 'im1.jpg'))
    S = im2segment(im)
    f = segment2feature(S[2])
    print("feature vector of one segment:")
    for name, value in zip(FEATURE_NAMES, f.ravel()):
        print(f"  {name:12s} {value: .4f}")

    debug = True
    allX, allY = benchmark_assignment2(segment2feature, datadir, debug)
    print("\nallX", allX.shape, " allY", allY)
