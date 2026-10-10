"""
Whether a turned reference drawing is one the head's keys can be fitted to.

The image model is asked to turn the head and change nothing else. When it
also redraws what should stay (a longer fringe, a bigger head, the outfit),
keys fitted to that drawing make the head grow its fringe or change size as
it turns, and no fit can follow it. A turned drawing is judged against the
front one on what a turn of about 30 degrees keeps, measured on their
decompositions and pictures. Thresholds calibrated on 20 reference sets
(2026-10-10): good drawings stay within them; the known bad ones do not.

- fringe: the front hair's area, turned over front. A turn shows the fringe
  foreshortened or as it was (0.75-1.18 seen); a redrawn, longer fringe is
  bigger (1.38-1.53 on three bad drawings).
- head: crown to chin, turned over front. Raising the head shortens it
  (0.83-0.92), lowering it keeps it (0.96-1.10); a redrawn head does not
  (1.18 on one).
- body: the share of the outfit, uncovered by hair or hands in either drawing,
  that changed. Only a body drawn anew passes the bound (0.02-0.20 seen).

    turn_judge.py <front.psd> <plus.psd> <minus.psd> <up.psd> <down.psd>
                  <front.png> <plus.png> <minus.png> <up.png> <down.png>
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageFilter
from psd_tools import PSDImage

SIDES = ('plus', 'minus', 'up', 'down')
FRINGE_MAX = 1.25
HEAD_RANGE = (0.80, 1.12)
BODY_MAX = 0.25
# A pixel this far (sum of channel differences) from the front picture's has changed.
CHANGED = 60

GARMENT = ('topwear', 'bottomwear', 'legwear', 'footwear')
# What moves with the head or the hands, over the outfit.
MOVING = ('front hair', 'back hair', 'handwear', 'headwear', 'earwear', 'face', 'ears', 'neck', 'neckwear')
HEAD = ('front hair', 'back hair', 'face')


class Layers:
    """A decomposition's layers by family (the name before '-N'), as masks on its canvas."""

    def __init__(self, path):
        psd = PSDImage.open(path)
        self.W, self.H = psd.size
        self.masks = {}
        for layer in psd.descendants():
            if layer.is_group():
                continue
            alpha = np.asarray(layer.topil().convert('RGBA'))[..., 3] > 127
            if not alpha.any():
                continue
            family = layer.name.split('-')[0].strip()
            mask = self.masks.setdefault(family, np.zeros((self.H, self.W), bool))
            y0, x0 = max(0, layer.top), max(0, layer.left)
            y1, x1 = min(self.H, layer.top + alpha.shape[0]), min(self.W, layer.left + alpha.shape[1])
            mask[y0:y1, x0:x1] |= alpha[y0 - layer.top:y1 - layer.top, x0 - layer.left:x1 - layer.left]

    def area(self, family):
        mask = self.masks.get(family)
        return int(mask.sum()) if mask is not None else 0

    def union(self, families):
        out = np.zeros((self.H, self.W), bool)
        for family in families:
            if family in self.masks:
                out |= self.masks[family]
        return out

    def head_height(self):
        """Crown (top of hair or face) to chin (bottom of face), or None without a face."""
        face = self.masks.get('face')
        if face is None:
            return None
        head = self.union(HEAD)
        return int(np.nonzero(face.any(1))[0].max() - np.nonzero(head.any(1))[0].min())


def _on_picture(mask, size):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0


def body_changed(front, turned, front_picture, turned_picture):
    """Share of the front's outfit, uncovered by what moves in either drawing, that changed; None when none is seen."""
    a = np.asarray(front_picture.convert('RGB')).astype(np.int16)
    b = np.asarray(turned_picture.convert('RGB')).astype(np.int16)
    if a.shape != b.shape:
        return None
    size = (a.shape[1], a.shape[0])
    keep = _on_picture(front.union(GARMENT), size) & ~_on_picture(front.union(MOVING) | turned.union(MOVING), size)
    # Not the outfit's edge, where a turn's small shifts land.
    keep = np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(9))) > 0
    if keep.sum() < 1000:
        return None
    return float((np.abs(b - a).sum(-1) > CHANGED)[keep].mean())


def judge_side(side, front, turned, front_picture=None, turned_picture=None):
    """One turned drawing's measures and the faults they show (empty: fit it as drawn)."""
    fringe = turned.area('front hair') / front.area('front hair') if front.area('front hair') else None
    hf, ht = front.head_height(), turned.head_height()
    head = ht / hf if hf and ht else None
    body = body_changed(front, turned, front_picture, turned_picture) if front_picture and turned_picture else None
    faults = []
    if fringe is not None and fringe > FRINGE_MAX:
        faults.append('fringe')
    if head is not None and not HEAD_RANGE[0] <= head <= HEAD_RANGE[1]:
        faults.append('head')
    if body is not None and body > BODY_MAX:
        faults.append('body')
    # How far past the bounds, to choose between two drawings of one side.
    badness = (max(0.0, (fringe or 0) - FRINGE_MAX) / FRINGE_MAX
               + (max(0.0, HEAD_RANGE[0] - head, head - HEAD_RANGE[1]) if head is not None else 0.0)
               + max(0.0, (body or 0) - BODY_MAX))
    return dict(fringe=fringe, head=head, body=body, faults=faults, badness=round(badness, 4))


def judge(front_psd, turned_psds, front_png=None, turned_pngs=None):
    """{side: judge_side(...)} for the turned decompositions given (by side)."""
    front = Layers(front_psd)
    front_picture = Image.open(front_png) if front_png and os.path.exists(front_png) else None
    out = {}
    for side, path in turned_psds.items():
        path_png = (turned_pngs or {}).get(side)
        picture = Image.open(path_png) if path_png and os.path.exists(path_png) else None
        out[side] = judge_side(side, front, Layers(path), front_picture, picture)
    return out


if __name__ == '__main__':
    psds, pngs = sys.argv[1:6], sys.argv[6:11]
    print(json.dumps(judge(psds[0], dict(zip(SIDES, psds[1:])), pngs[0] if pngs else None,
                           dict(zip(SIDES, pngs[1:])) if pngs else None), indent=1))
