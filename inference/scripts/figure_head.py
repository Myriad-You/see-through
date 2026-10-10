"""
A standing figure's head, keyed as a bust.

Decomposed whole, a full figure's head is a couple of hundred pixels tall, too
small to fit turn keys well. head_crop frames the head as a bust is framed;
that crop is drawn turned, decomposed and keyed as a bust (turn_keyforms), and
place_keys puts the keys back onto the figure's canvas, where they move the
figure's own head parts. The figure keeps its own decomposition at rest:
merging the head decomposition's parts in instead was tried and lost what
only the figure's had (a pendant), and gained nothing at rest.

    python figure_head.py image.png figure.psd keyforms.json out.json   (box from head_crop)
"""
import re
import sys

import cv2
import numpy as np
from PIL import Image


# A bust's framing (contract: close full head through lower chest, 3:4): the
# crown at the top, the chin at BUST_CHIN of the height (0.47-0.52 on six
# generated busts), the face's middle at the middle.
BUST_CHIN = 0.49
BUST_TOP = 0.015
BUST_ASPECT = 0.75
# Not worth a second decomposition: a head this big already (canvas px, crown
# to chin) is a bust's.
HEAD_ENOUGH = 520


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


# The head's decomposition and the figure's split the hair differently: the
# figure's often gives the locks falling beside the face to the back hair,
# under the face, where the head's (whose keys move them) has them in the
# front hair. Turned, the face then slides over them. And the head's front hair
# is cut into locks (turn_keyforms, hair_locks), each with a key fitted well;
# a whole front hair's single key cannot follow locks the turned drawings
# redraw (IoU ~0.75 against ~0.95). So the figure takes its hair from the head,
# placed on its canvas: the head's locks and back hair (baked where turns
# uncover it) in place of its own front and back hair. All else stays the
# figure's own (what only it has, like a pendant, is kept).
FRONT_LOCK = re.compile(r'front hair-\d+$')


def take_hair(figure, head, factor, offset):
    """
    Replaces the figure's front hair (whole, or locks taken before) and back
    hair with the head's front hair locks and back hair placed on the figure's
    canvas (p_figure = p_head * factor + offset). Returns the names taken, or []
    when the head has no locks or back hair (the figure keeps its own hair).
    """
    locks = [name for name in head.order if FRONT_LOCK.match(name)]
    if not locks or 'back hair' not in head.layers:
        return []
    front = [name for name in figure.order if name == 'front hair' or FRONT_LOCK.match(name)]
    if not front:
        return []
    place = np.float32([[factor, 0, offset[0]], [0, factor, offset[1]]])

    def placed(name):
        layer = head.layers[name]
        # Premultiplied while resampled, so edges keep their colour.
        rgb = layer[..., :3] * layer[..., 3:4]
        out = cv2.warpAffine(np.concatenate([rgb, layer[..., 3:4]], 2), place, (figure.W, figure.H),
                             flags=cv2.INTER_AREA if factor < 1 else cv2.INTER_LINEAR)
        alpha = np.clip(out[..., 3:4], 0, 1)
        out[..., :3] = np.where(alpha > 1e-4, out[..., :3] / np.maximum(alpha, 1e-4), 0)
        out[..., 3:4] = alpha
        return np.clip(out, 0, 1).astype(np.float32)

    at = figure.order.index(front[0])
    for name in front:
        figure.order.remove(name)
        del figure.layers[name]
    figure.order[at:at] = locks
    for name in locks:
        figure.layers[name] = placed(name)
    if 'back hair' not in figure.layers:
        figure.order.insert(0, 'back hair')
    figure.layers['back hair'] = placed('back hair')
    return locks + ['back hair']


class _Canvas:
    """A decomposition's canvas size, all head_to_full and place_keys need of it."""
    def __init__(self, W, H):
        self.W, self.H = W, H


def figure_keys(keys, figure_hw, box, image_hw, head_hw=BUST_PIXELS[::-1]):
    """The head's keys (canvas in keys['canvas']) on a figure canvas of figure_hw (h, w)."""
    figure = _Canvas(figure_hw[1], figure_hw[0])
    head = _Canvas(*keys['canvas'])
    factor, offset = head_to_full(figure, head, box, image_hw, head_hw)
    return place_keys(keys, figure, factor, offset)


if __name__ == '__main__':
    import json
    from turn_keyforms import Decomposition
    image_path, figure_path, keys_path, out_path = sys.argv[1:5]
    image = Image.open(image_path)
    figure = Decomposition(figure_path)
    box = head_crop(figure, (image.height, image.width))
    print('box', box)
    with open(keys_path) as f:
        keys = figure_keys(json.load(f), (figure.H, figure.W), box, (image.height, image.width))
    with open(out_path, 'w') as f:
        json.dump(keys, f)
