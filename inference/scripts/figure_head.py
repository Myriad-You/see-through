"""
A standing figure's head, decomposed again as a bust is framed.

Decomposed whole, a full figure's head is a couple of hundred pixels tall: the
body pass finds its hair, neck and choker at that size, and the head pass's
parts are drawn back at it too. A choker comes out a blurred band, an earring
darkened, clips sunk under the bangs. Cut out at a bust's framing and
decomposed again, the head comes out as a bust's does; its parts go back into
the whole figure's canvas, scaled down, and the body's parts stay the whole
decomposition's.

    python figure_head.py image.png full.psd head.psd out.psd    (merge only;
    the head crop to decompose is head_crop(full.psd, image))
"""
import sys

import cv2
import numpy as np
from PIL import Image

from turn_keyforms import Decomposition, save_psd

# What the head decomposition gives: See-through's layer names, by the part
# before any '-l'/'-r'/'-N' suffix.
HEAD_PARTS = ('face', 'eyewhite', 'irides', 'eyelash', 'eyebrow', 'eyewear', 'nose', 'mouth',
              'ears', 'earwear', 'headwear', 'front hair', 'back hair', 'neck', 'neckwear')
# A bust's framing (contract: close full head through lower chest, 3:4): the
# crown at the top, the chin at BUST_CHIN of the height (0.47-0.52 on six
# generated busts), the face's middle at the middle.
BUST_CHIN = 0.49
BUST_TOP = 0.015
BUST_ASPECT = 0.75
# Inside the crop the head's parts are the head decomposition's; they fade to
# the whole decomposition's over this many canvas pixels at its edge (long
# hair running past the crop).
FEATHER = 6
# Not worth a second decomposition: a head this big already (canvas px, crown
# to chin) is a bust's.
HEAD_ENOUGH = 520


def is_head_part(name):
    return name.split('-')[0] in HEAD_PARTS


def placement(src_hw, canvas_hw):
    """Where an image of src_hw sits on a canvas (scale, x, y), as the decomposition fits it (cv.fit_placement)."""
    sh, sw = src_hw
    ch, cw = canvas_hw
    scale = min(cw / sw, ch / sh)
    w = min(cw, int(np.floor(sw * scale + 0.5)))
    h = min(ch, int(np.floor(sh * scale + 0.5)))
    return scale, (cw - w) // 2, (ch - h) // 2


def head_crop(full, image_hw):
    """
    The bust-framed box (x0, y0, x1, y1) in image pixels around the head of a
    whole decomposition, or None without a face or with a head big enough.
    """
    face = full.layers.get('face')
    if face is None or not (face[..., 3] > 0.5).any():
        return None
    head = np.zeros((full.H, full.W), bool)
    for name in full.order:
        if name.split('-')[0] in ('front hair', 'back hair', 'headwear', 'face'):
            head |= full.layers[name][..., 3] > 0.5
    face = face[..., 3] > 0.5
    ys, xs = np.nonzero(face)
    crown = np.nonzero(head.any(1))[0].min()
    chin = ys.max()
    if chin - crown >= HEAD_ENOUGH:
        return None
    scale, ox, oy = placement(image_hw, (full.H, full.W))
    # Canvas to image pixels.
    crown, chin = (crown - oy) / scale, (chin - oy) / scale
    cx = ((xs.min() + xs.max()) / 2 - ox) / scale
    height = (chin - crown) / (BUST_CHIN - BUST_TOP)
    top = crown - BUST_TOP * height
    width = height * BUST_ASPECT
    x0, y0 = int(round(cx - width / 2)), int(round(top))
    # Whole pixels, 3:4 exactly; past the image's edge is filled white (crop_image).
    w = int(round(width))
    return [x0, y0, x0 + w, y0 + int(round(w / BUST_ASPECT))]


# The size the head is drawn and decomposed at: a bust's (contract generationPixels).
BUST_PIXELS = (1152, 1536)


def crop_image(image, box, size=BUST_PIXELS):
    """The box of the image, white past its edges, scaled to `size` (w, h)."""
    x0, y0, x1, y1 = box
    out = Image.new('RGB', (x1 - x0, y1 - y0), (255, 255, 255))
    part = image.convert('RGB').crop((max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)))
    out.paste(part, (max(0, -x0), max(0, -y0)))
    return out.resize(size, Image.LANCZOS)


def _scaled(layer, factor, out_hw, offset):
    """A canvas layer scaled by `factor` and moved by `offset` onto out_hw, premultiplied so edges do not darken."""
    h, w = layer.shape[:2]
    pre = layer.copy()
    pre[..., :3] *= pre[..., 3:4]
    size = (max(1, int(round(w * factor))), max(1, int(round(h * factor))))
    small = cv2.resize(pre, size, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
    out = np.zeros(out_hw + (4,), np.float32)
    dx, dy = int(round(offset[0])), int(round(offset[1]))
    y0, x0 = max(0, dy), max(0, dx)
    y1, x1 = min(out_hw[0], dy + small.shape[0]), min(out_hw[1], dx + small.shape[1])
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = small[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    return out


def _unpremultiply(pre):
    out = pre.copy()
    a = out[..., 3:4]
    out[..., :3] = np.where(a > 1e-4, out[..., :3] / np.maximum(a, 1e-4), 0)
    return np.clip(out, 0, 1)


def head_to_full(full, head, box, image_hw, head_hw=BUST_PIXELS[::-1]):
    """
    The head canvas onto the whole canvas: p_full = p_head * factor + offset.
    box: the head crop in image pixels; head_hw: the size (h, w) the crop was
    decomposed at.
    """
    x0, y0, x1, y1 = box
    fs, fx, fy = placement(image_hw, (full.H, full.W))
    hs, hx, hy = placement(head_hw, (head.H, head.W))
    # Head canvas to crop input pixels, to image pixels, to whole canvas.
    to_image = (x1 - x0) / head_hw[1]
    factor = fs * to_image / hs
    return factor, (fx + fs * (x0 - hx * to_image / hs), fy + fs * (y0 - hy * to_image / hs))


def merge(full, head, box, image_hw, head_hw=BUST_PIXELS[::-1]):
    """The whole decomposition with its head parts the head decomposition's (box, head_hw: head_to_full)."""
    x0, y0, x1, y1 = box
    fs, fx, fy = placement(image_hw, (full.H, full.W))
    factor, offset = head_to_full(full, head, box, image_hw, head_hw)
    # Where the crop lies on the whole canvas, and how much of each pixel the head decomposition owns there.
    rx0, ry0 = fx + fs * x0, fy + fs * y0
    rx1, ry1 = fx + fs * x1, fy + fs * y1
    yy, xx = np.mgrid[0:full.H, 0:full.W].astype(np.float32)
    ih, iw = image_hw
    # A side of the crop at the image's own edge is no seam: nothing lies past it.
    far = np.float32(1e6)
    sides = [xx - rx0 if x0 > 0 else far + 0 * xx, rx1 - 1 - xx if x1 < iw else far + 0 * xx,
             yy - ry0 if y0 > 0 else far + 0 * yy, ry1 - 1 - yy if y1 < ih else far + 0 * yy]
    inside = np.minimum.reduce(sides)
    # Outside the crop (past an image edge too) the head decomposition has nothing.
    inside = np.where((xx >= rx0 - 0.5) & (xx <= rx1 - 0.5) & (yy >= ry0 - 0.5) & (yy <= ry1 - 0.5), inside, -1)
    weight = np.clip(inside / FEATHER, 0, 1)[..., None]

    taken = [name for name in head.order if is_head_part(name)]
    out = Decomposition.__new__(Decomposition)
    out.psd, out.W, out.H = None, full.W, full.H
    out.layers = dict(full.layers)
    # A part the head decomposition draws in other layers (the front hair cut
    # into locks) keeps the whole decomposition's drawing of it only outside the crop.
    bases = {name.split('-')[0] for name in taken}
    for name in full.order:
        if name.split('-')[0] in bases and name not in taken:
            outside = full.layers[name].copy()
            outside[..., 3:4] *= 1 - weight
            out.layers[name] = outside
    for name in taken:
        mine = _scaled(head.layers[name], factor, (full.H, full.W), offset) * weight
        if name in full.layers:
            theirs = full.layers[name].copy()
            theirs[..., :3] *= theirs[..., 3:4]
            mine = mine + theirs * (1 - weight)
        out.layers[name] = _unpremultiply(mine)
    out.order = merged_order(full.order, head.order, set(taken))
    return out


def place_keys(keys, full, factor, offset):
    """Turn keys fitted on the head canvas, onto the whole canvas."""
    out = dict(keys, canvas=[full.W, full.H])
    placed = {}
    for family, sides in keys['keyforms'].items():
        placed[family] = {}
        for side, lattice in sides.items():
            x0, y0, x1, y1 = lattice['box']
            moved = dict(lattice, box=[x0 * factor + offset[0], y0 * factor + offset[1],
                                       x1 * factor + offset[0], y1 * factor + offset[1]],
                         back=[v * factor for v in lattice['back']])
            placed[family][side] = moved
    out['keyforms'] = placed
    out['figure'] = dict(factor=factor, offset=list(offset))
    return out


def merged_order(full_order, head_order, taken):
    """
    The whole decomposition's order, with the head parts it has put in its
    head parts' places in the head decomposition's order among themselves,
    and those only the head decomposition has put after their predecessor there.
    """
    slots = [i for i, name in enumerate(full_order) if name in taken]
    ordered = [name for name in head_order if name in taken and name in full_order]
    out = list(full_order)
    for i, name in zip(slots, ordered):
        out[i] = name
    for name in head_order:
        if name not in taken or name in out:
            continue
        before = [n for n in head_order[:head_order.index(name)] if n in out]
        out.insert(out.index(before[-1]) + 1 if before else (slots[0] if slots else len(out)), name)
    return out


if __name__ == '__main__':
    image_path, full_path, head_path, out_path = sys.argv[1:5]
    image = Image.open(image_path).convert('RGB')
    full = Decomposition(full_path)
    box = [int(v) for v in sys.argv[5].split(',')] if len(sys.argv) > 5 else head_crop(full, (image.height, image.width))
    print('box', box)
    merged = merge(full, Decomposition(head_path), box, (image.height, image.width))
    save_psd(merged, out_path)
    print('order', merged.order)
