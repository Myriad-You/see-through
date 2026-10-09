"""
A standing figure's body turn, keyed as its head's turn is (turn_keyforms).

The figure is drawn turned about 20 degrees each way, its feet where they
stood, and both drawings are decomposed on the canvas its own decomposition
was made on (same framing). Each body part of the figure's decomposition is
fitted onto the same part of each turned one, then refined on the turned
picture: the head as one piece, which only carries it (its own turn has the
head's keys); each thigh, shin and shoe apart, since a turn moves the near leg
and the far one differently, and a leg's top with the hips and its foot where
it stands. The legs are cut at the knee the figure's own decomposition puts
halfway from where the leg's drawing starts to the ankle (the shoe's top),
the same row in all three, which are lined up.

Both turned drawings also share what the generator drew differently from the
figure whichever way it turned (a neck a little to one side, a waist a little
to the other). That is not the turn, and keyed it pulls neck and collar apart
as the figure turns: each part keeps only what moves opposite ways in the two
turns, half their difference, as +key and -key.

Keys are lattices as turn_keyforms makes them ('plus' toward image right),
placed from the canvas onto the figure's picture, `image_hw`, whose
decomposition the figure's is. Myriad's sides are the picture's: See-through's
"-r" layers are on its left.

Run as a process of its own (it sets turn_keyforms' part tables):

    python body_turn.py figure.psd right.psd left.psd right.png left.png H W out.json
"""
import json
import sys
import time

import cv2
import numpy as np

import turn_keyforms as tk
from figure_head import placement

HEAD = ['front hair', 'face', 'nose', 'mouth', 'eyebrow-r', 'eyebrow-l', 'irides-l', 'irides-r',
        'eyewhite-r', 'eyewhite-l', 'eyelash-l', 'eyelash-r', 'ears-r', 'ears-l', 'ears', 'headwear', 'earwear']
# The back hair is left out of the head: turned decompositions lump it with the body.
BODY = {
    'head': HEAD,
    'neck': ['neck'],
    'topwear': ['topwear'],
    'bottomwear': ['bottomwear'],
    'thigh:L': ['thigh-L'],
    'thigh:R': ['thigh-R'],
    'shin:L': ['shin-L'],
    'shin:R': ['shin-R'],
    'footwear:L': ['footwear-L'],
    'footwear:R': ['footwear-R'],
    'arm:L': ['handwear-r'],
    'arm:R': ['handwear-l'],
}
# A fine lattice follows the turned drawing's redrawn folds and prints and
# smears them (stretch 0.2-2.4 at 13 points a side); 7 holds the drawing.
GRID = 7
HEAD_GRID = 9
# Points of the forward field a symmetric key is measured at, canvas px apart.
STRIDE = 4


def split_limbs(dec):
    """Each leg and shoe on its own (the picture's sides), cut at the legs' midline."""
    legs = dec.layers.get('legwear')
    if legs is None or not (legs[..., 3] > 0.5).any():
        return dec
    mid = int(round(np.nonzero(legs[..., 3] > 0.5)[1].mean()))
    for name in ('legwear', 'footwear'):
        if name not in dec.layers:
            continue
        layer = dec.layers.pop(name)
        at = dec.order.index(name)
        left, right = layer.copy(), layer.copy()
        left[:, mid:] = 0
        right[:, :mid] = 0
        dec.layers[f'{name}-L'], dec.layers[f'{name}-R'] = left, right
        dec.order[at:at + 1] = [f'{name}-L', f'{name}-R']
    return dec


def knees_of(dec):
    """Each leg's knee row: halfway from the top of its drawing to its ankle (its shoe's top, else its bottom)."""
    knees = {}
    for side in ('L', 'R'):
        leg = dec.layers.get(f'legwear-{side}')
        if leg is None or not (leg[..., 3] > 0.5).any():
            continue
        rows = np.nonzero((leg[..., 3] > 0.5).any(1))[0]
        shoe = dec.layers.get(f'footwear-{side}')
        shoe_rows = np.nonzero((shoe[..., 3] > 0.5).any(1))[0] if shoe is not None else []
        ankle = shoe_rows.min() if len(shoe_rows) else rows.max()
        knees[side] = int(round((rows.min() + ankle) / 2))
    return knees


def split_knees(dec, knees):
    """Each leg as its thigh and its shin, cut at `knees`."""
    for side, knee in knees.items():
        name = f'legwear-{side}'
        if name not in dec.layers:
            continue
        layer = dec.layers.pop(name)
        at = dec.order.index(name)
        thigh, shin = layer.copy(), layer.copy()
        thigh[knee:] = 0
        shin[:knee] = 0
        dec.layers[f'thigh-{side}'], dec.layers[f'shin-{side}'] = thigh, shin
        dec.order[at:at + 1] = [f'thigh-{side}', f'shin-{side}']
    return dec


def use_body_tables():
    tk.FAMILIES.clear()
    tk.FAMILIES.update(BODY)
    tk.ACCESSORIES[:] = []
    tk.GRID.clear()
    tk.GRID.update({family: GRID for family in BODY})
    tk.GRID['head'] = HEAD_GRID
    tk.COARSE_GRID = GRID


def fit_side(figure, turned, picture, log):
    keys = {}
    for family, names in BODY.items():
        use = [n for n in names if n in turned.layers and n in figure.layers]
        if not use:
            continue
        fit = tk.fit_family(figure, turned, use, family, log)
        if fit:
            keys[family] = {'side': fit}
    tk.refine_on_picture(figure, keys, 'side', picture, [], log)
    return {family: key['side'] for family, key in keys.items()}


def forward_field(key, X, Y):
    """Forward offset (turned - rest) at rest points: t with t + back(t) = q."""
    x0, y0, x1, y1 = key['box']
    g = key['grid']
    back = np.asarray(key['back'], np.float32).reshape(-1, 2)
    xs, ys = np.linspace(x0, x1, g), np.linspace(y0, y1, g)
    tx, ty = X.copy(), Y.copy()
    for _ in range(40):
        bx, by = tk.sample(back, xs, ys, tx, ty)
        tx, ty = X - bx, Y - by
    return tx - X, ty - Y


def opposite_keys(right, left):
    """+key and -key of what moves opposite ways in the two turns, and what both share (the drift dropped)."""
    grid = right['grid']
    x0, y0 = min(right['box'][0], left['box'][0]), min(right['box'][1], left['box'][1])
    x1, y1 = max(right['box'][2], left['box'][2]), max(right['box'][3], left['box'][3])
    X, Y = np.meshgrid(np.arange(x0, x1 + STRIDE, STRIDE, dtype=np.float32),
                       np.arange(y0, y1 + STRIDE, STRIDE, dtype=np.float32))
    rx, ry = forward_field(right, X.ravel(), Y.ravel())
    lx, ly = forward_field(left, X.ravel(), Y.ravel())
    ax = ((rx - lx) / 2).reshape(X.shape).astype(np.float32)
    ay = ((ry - ly) / 2).reshape(X.shape).astype(np.float32)
    drift = (float(np.median((rx + lx) / 2)), float(np.median((ry + ly) / 2)))
    pad = float(np.abs(np.stack([ax, ay])).max()) + 2

    def lattice(sign):
        # Backward at the turned lattice's points t: q - t, where q + sign * A(q) = t.
        TX, TY = np.meshgrid(np.linspace(x0 - pad, x1 + pad, grid).astype(np.float32),
                             np.linspace(y0 - pad, y1 + pad, grid).astype(np.float32))
        qx, qy = TX.copy(), TY.copy()
        for _ in range(40):
            mx, my = (qx - x0) / STRIDE, (qy - y0) / STRIDE
            fx = cv2.remap(ax, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fy = cv2.remap(ay, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            qx, qy = TX - sign * fx, TY - sign * fy
        back = np.stack([qx - TX, qy - TY], -1).reshape(-1)
        return dict(box=[float(x0 - pad), float(y0 - pad), float(x1 + pad), float(y1 + pad)], grid=grid,
                    back=[float(v) for v in back])

    return lattice(+1), lattice(-1), drift


def placed(lattice, scale, ox, oy):
    """A canvas lattice in the picture's pixels."""
    x0, y0, x1, y1 = lattice['box']
    return dict(lattice, box=[(x0 - ox) / scale, (y0 - oy) / scale, (x1 - ox) / scale, (y1 - oy) / scale],
                back=[v / scale for v in lattice['back']])


def body_turn(figure_path, right_path, left_path, right_png, left_png, image_hw, log=print):
    """
    {canvas: [W, H] of the picture, keyforms: {family: {plus, minus}}, and what was found:
    drift: {family: [dx, dy]} dropped, knees: {side: y} cut at}, in the picture's pixels.
    """
    use_body_tables()
    figure = split_limbs(tk.Decomposition(figure_path))
    knees = knees_of(figure)
    split_knees(figure, knees)
    fits = {}
    for side, psd, png in (('right', right_path, right_png), ('left', left_path, left_png)):
        turned = split_knees(split_limbs(tk.Decomposition(psd)), knees)
        if (turned.W, turned.H) != (figure.W, figure.H):
            raise ValueError(f'{side}: canvas {turned.W}x{turned.H} is not the figure canvas {figure.W}x{figure.H}')
        picture = tk.placed_illustration(png, figure.W, figure.H)[0]
        fits[side] = fit_side(figure, turned, picture, lambda line, side=side: log(f'{side:5s} {line}'))
    scale, ox, oy = placement(image_hw, (figure.H, figure.W))
    keyforms, drift = {}, {}
    for family in BODY:
        if family not in fits['right'] or family not in fits['left']:
            continue
        plus, minus, shared = opposite_keys(fits['right'][family], fits['left'][family])
        keyforms[family] = {'plus': placed(plus, scale, ox, oy), 'minus': placed(minus, scale, ox, oy)}
        drift[family] = [round(shared[0] / scale, 1), round(shared[1] / scale, 1)]
        log(f'{family:11s} drift dropped ({drift[family][0]:+.1f}, {drift[family][1]:+.1f}) px')
    knees = {side: round((y - oy) / scale, 1) for side, y in knees.items()}
    return dict(canvas=[image_hw[1], image_hw[0]], keyforms=keyforms, drift=drift, knees=knees)


if __name__ == '__main__':
    figure_psd, right_psd, left_psd, right_png, left_png, height, width, out = sys.argv[1:9]
    t0 = time.time()
    result = body_turn(figure_psd, right_psd, left_psd, right_png, left_png, (int(height), int(width)),
                       log=lambda line: print(line, flush=True))
    with open(out, 'w') as f:
        json.dump(result, f)
    print(f'body turn keys in {time.time() - t0:.0f}s', flush=True)
