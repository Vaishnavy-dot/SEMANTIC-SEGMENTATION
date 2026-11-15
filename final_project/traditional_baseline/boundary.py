"""Exact-grid class-aware boundary F1/F2; no tuned matching tolerance.

An inner boundary pixel has a differently labelled valid 4-neighbour.
Image exterior and ignored pixels do not create boundaries. Matches require
the same pixel coordinate and class. Counts pool tiles before computing each
class score; macro scores average defined classes. This is NOT tolerant BF.
"""
import json
from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ['impervious_surface', 'building', 'low_vegetation', 'tree', 'car', 'clutter']
PALETTE = [(255,255,255), (0,0,255), (0,255,255), (0,255,0), (255,255,0), (255,0,0)]


def boundaries(labels, valid):
    edge = np.zeros(labels.shape, dtype=bool)
    different = (labels[1:] != labels[:-1]) & valid[1:] & valid[:-1]
    edge[1:] |= different
    edge[:-1] |= different
    different = (labels[:,1:] != labels[:,:-1]) & valid[:,1:] & valid[:,:-1]
    edge[:,1:] |= different
    edge[:,:-1] |= different
    return edge


def counts(pred, truth):
    valid = (pred != 255) & (truth != 255)
    pb, gb = boundaries(pred, valid), boundaries(truth, valid)
    matched = pb & gb & (pred == truth)
    return np.stack([np.bincount(a, minlength=6) for a in
                     (pred[pb], truth[gb], pred[matched])])


def self_test():
    a = np.array([[0,0,1,1],[0,0,1,1]], dtype=np.uint8)
    c = counts(a,a)
    assert np.array_equal(c[0],c[1]) and np.array_equal(c[0],c[2])
    assert c[0].tolist() == [2,2,0,0,0,0]
    b = np.array([[0,0,0,1],[0,0,0,1]], dtype=np.uint8)
    assert counts(a,b)[2].sum() == 0  # Shifted outlines do not exactly match.
    assert counts(np.zeros_like(a),np.zeros_like(a)).sum() == 0
    a[:,2:] = 255
    assert counts(a,a).sum() == 0  # Ignore interfaces are excluded.

