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

Each drawing must turn the way it was asked to: a generator asked for the
other way often draws the same turn again (mirroring the figure for it does
not help), and then what moves opposite ways is only what the two drawings
got differently. The trunk's front says which way a drawing turned
(`turned`, picture px, + toward image right). A drawing that turned the wrong
way is not used: the other's fit, mirrored about the trunk, keys its turn
(`mirrored`), each limb's from the other limb's. A body turns about alike
either way; what was drawn differently on one side is lost to the other. One
drawing alone does not tell its turn from where the generator moved the whole
figure, but the feet were to stay: how far the shoes went is taken off first.
How far each part goes across is read off the pictures, not the fit: a
lattice fitted to a big garment follows its outline, which a 20 degree turn
barely moves, more than its prints and seams; the same pair fitted twice
keyed the top -4 and +17 px, the skirt +145 and +38, while the drawing moved
them +80 and +55. The optical flow from the figure to each drawing, its median
over each part (`flow`), sets each key's travel; the fit keeps its shape.
The trunk and what hangs over it (top, skirt, neck, sleeves) lie on one
another, and fitted apart they part at every edge where they meet: they go
as one, each row of the canvas across by the trunk's median travel at that
height, nothing else, on one lattice over all of them (TRUNK_GRID rows):
the same travel sampled on lattices of their own boxes would part them by a
few px where the travel bends. What a turn does across a row (it narrows the
figure) is the same either way and drops out of what moves opposite ways anyway.
A part keyed to go against the trunk's turn (a small, hidden neck fitted to
something else) is not keyed by its own fit: the neck and arms, which hang
from the torso, take the top's key; a leg goes unkeyed.

Keys are lattices as turn_keyforms makes them ('plus' toward image right),
placed from the canvas onto the figure's picture, `image_hw`, whose
decomposition the figure's is. Myriad's sides are the picture's: See-through's
"-r" layers are on its left.

The same fit keys a weight shift (mode 'weight'): the figure drawn with its
weight on the leg on the right of the picture ('plus') and on the left. There
the legs are not mirror images of each other: the free leg bends, the one under
the weight stays straight, so each thigh, shin and shoe keeps its own fit in
each drawing; the rest (hips, skirt, torso, arms, the head's carry) keeps what
moves opposite ways, as in a turn.

Run as a process of its own (it sets turn_keyforms' part tables):

    python body_turn.py figure.psd right.psd left.psd right.png left.png H W out.json [turn|weight]
"""
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor

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
# Which way a turned drawing went is read from these, whose fronts turn plainly.
TRUNK = ('topwear', 'bottomwear')
# A drawing whose trunk went less than this (picture px) its own way drew the wrong turn.
TURNED_AT_LEAST = 10.0
# The optical flow is measured at this width, where a turn is a few dozen px.
FLOW_WIDTH = 420
# The trunk's layers, whose travel goes by height so that top and skirt meet at the waist.
TRUNK_ROWS = ('topwear', 'bottomwear', 'neck')
# What goes as the trunk does, row by row: the trunk and the sleeves lying over its sides.
AS_TRUNK = ('topwear', 'bottomwear', 'neck', 'arm:L', 'arm:R')
# The trunk's one lattice: only rows move, so a fine one holds the drawing.
TRUNK_GRID = 13
# The trunk's travel by height is smoothed over this share of its height either way.
TRUNK_SMOOTH = 0.04
# Where a part is sampled to measure its key's travel, canvas px apart.
PART_STRIDE = 8
# What hangs from the torso, and takes its key when its own went astray.
ON_TORSO = ('neck', 'arm:L', 'arm:R')
# What a mirror makes of each part: a limb becomes the other one.
MIRRORED = {'thigh:L': 'thigh:R', 'thigh:R': 'thigh:L', 'shin:L': 'shin:R', 'shin:R': 'shin:L',
            'footwear:L': 'footwear:R', 'footwear:R': 'footwear:L', 'arm:L': 'arm:R', 'arm:R': 'arm:L'}
# In a weight shift these are one-sided, each drawing's own fit (see above).
ONE_SIDED = {'thigh:L', 'thigh:R', 'shin:L', 'shin:R', 'footwear:L', 'footwear:R'}


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


def part_flows(figure, picture):
    """
    Each body part's median travel across (canvas px) from the figure to a turned
    picture (RGB 0..1, on its canvas), and the trunk's by height: 'rows', the
    median of each canvas row of the trunk, smoothed (NaN off the trunk).
    """
    front = figure.family(figure.order)
    k = FLOW_WIDTH / figure.W

    def gray(rgb):
        small = cv2.resize((np.clip(rgb, 0, 1) * 255).astype(np.uint8), None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    f0 = gray(front[..., :3] * front[..., 3:4] + (1 - front[..., 3:4]))
    f1 = gray(picture)
    flow = cv2.calcOpticalFlowFarneback(f0, f1, None, 0.5, 5, 25, 5, 7, 1.5, 0)[..., 0] / k
    out = {}
    for family, names in BODY.items():
        use = [n for n in names if n in figure.layers]
        if not use:
            continue
        alpha = sum(figure.layers[n][..., 3] for n in use)
        mask = cv2.resize(alpha, (f0.shape[1], f0.shape[0]), interpolation=cv2.INTER_AREA) > 0.5
        if mask.sum() >= 20:
            out[family] = float(np.median(flow[mask]))
    trunk = sum(figure.layers[n][..., 3] for n in TRUNK_ROWS if n in figure.layers)
    mask = cv2.resize(np.asarray(trunk, np.float32), (f0.shape[1], f0.shape[0]), interpolation=cv2.INTER_AREA) > 0.5
    rows = np.array([np.median(flow[r][mask[r]]) if mask[r].sum() >= 3 else np.nan for r in range(len(mask))])
    known = np.nonzero(~np.isnan(rows))[0]
    if len(known):
        half = max(1, int(TRUNK_SMOOTH * len(known)))
        smooth = np.array([np.median(rows[known[max(0, i - half):i + half + 1]]) for i in range(len(known))])
        rows[known] = smooth
    out['rows'] = np.interp(np.arange(figure.H), np.arange(len(rows)) / k, rows, left=np.nan, right=np.nan).tolist()
    return out


def part_points(figure, family):
    """Points on a part of the figure, canvas px: (xs, ys)."""
    use = [n for n in BODY[family] if n in figure.layers]
    alpha = sum(figure.layers[n][..., 3] for n in use) if use else np.zeros((figure.H, figure.W), np.float32)
    ys, xs = np.nonzero(alpha[::PART_STRIDE, ::PART_STRIDE] > 0.5)
    return (xs * PART_STRIDE).astype(np.float32), (ys * PART_STRIDE).astype(np.float32)


def part_way(key, points):
    """How far a key carries the part at `points` across, canvas px (its median)."""
    dx, _ = forward_field(key, *points)
    return float(np.median(dx))


def fit_turned(figure_path, psd, png):
    """One turned drawing fitted, in a process of its own: (keys, each part's flow, log lines)."""
    use_body_tables()
    figure = split_limbs(tk.Decomposition(figure_path))
    knees = knees_of(figure)
    split_knees(figure, knees)
    turned = split_knees(split_limbs(tk.Decomposition(psd)), knees)
    if (turned.W, turned.H) != (figure.W, figure.H):
        raise ValueError(f'canvas {turned.W}x{turned.H} is not the figure canvas {figure.W}x{figure.H}')
    picture = tk.placed_illustration(png, figure.W, figure.H)[0]
    lines = []
    return fit_side(figure, turned, picture, lines.append), part_flows(figure, picture), lines


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


def key_way(key):
    """How far a key carries its part across, canvas px (+ toward image right)."""
    x0, y0, x1, y1 = key['box']
    X, Y = np.meshgrid(np.linspace(x0, x1, 9, dtype=np.float32), np.linspace(y0, y1, 9, dtype=np.float32))
    dx, _ = forward_field(key, X.ravel(), Y.ravel())
    return float(np.median(dx))


def sole_way(key):
    """How far a shoe's key carries its sole across, canvas px: its bottom rows, which stay put as it turns."""
    x0, y0, x1, y1 = key['box']
    X, Y = np.meshgrid(np.linspace(x0, x1, 9, dtype=np.float32), np.linspace(y1 - 0.15 * (y1 - y0), y1, 3, dtype=np.float32))
    dx, _ = forward_field(key, X.ravel(), Y.ravel())
    return float(np.median(dx))


def turned_way(fit, families=TRUNK):
    """How far a turned drawing carried these parts (the trunk's front), canvas px, or None."""
    moves = [key_way(fit[family]) for family in families if fit.get(family)]
    return float(np.mean(moves)) if moves else None


def shifted_key(key, dx):
    """A lattice key whose turned part lies dx further across."""
    x0, y0, x1, y1 = key['box']
    back = np.asarray(key['back'], np.float32).reshape(-1, 2).copy()
    back[:, 0] -= dx
    return dict(key, box=[float(x0 + dx), float(y0), float(x1 + dx), float(y1)],
                back=[float(v) for v in back.reshape(-1)])


def trunk_axis(figure):
    """The figure's middle, canvas x: halfway across its trunk."""
    alpha = sum(figure.layers[n][..., 3] for n in TRUNK if n in figure.layers)
    xs = np.nonzero((np.asarray(alpha) > 0.5).any(0))[0]
    return float(xs[0] + xs[-1] + 1) / 2 if len(xs) else figure.W / 2


def mirrored_key(key, axis):
    """A lattice key reflected about x = axis: what its part does turned the other way."""
    x0, y0, x1, y1 = key['box']
    g = key['grid']
    back = np.asarray(key['back'], np.float32).reshape(g, g, 2)[:, ::-1].copy()
    back[..., 0] *= -1
    return dict(key, box=[float(2 * axis - x1), float(y0), float(2 * axis - x0), float(y1)],
                back=[float(v) for v in back.reshape(-1)])


def mirrored_fit(fit, axis):
    """A turned drawing's fit reflected: every part's key as the other turn's, limbs swapped."""
    return {MIRRORED.get(family, family): mirrored_key(key, axis) for family, key in fit.items()}


def opposite_keys(right, left, across=None):
    """
    +key and -key of what moves opposite ways in the two turns, and what both
    share (the drift dropped). `across(ax, ay, X, Y)` may set the +key's
    field (canvas px) at the points X, Y: (ax, ay).
    """
    grid = right['grid']
    x0, y0 = min(right['box'][0], left['box'][0]), min(right['box'][1], left['box'][1])
    x1, y1 = max(right['box'][2], left['box'][2]), max(right['box'][3], left['box'][3])
    X, Y = np.meshgrid(np.arange(x0, x1 + STRIDE, STRIDE, dtype=np.float32),
                       np.arange(y0, y1 + STRIDE, STRIDE, dtype=np.float32))
    rx, ry = forward_field(right, X.ravel(), Y.ravel())
    lx, ly = forward_field(left, X.ravel(), Y.ravel())
    ax = ((rx - lx) / 2).reshape(X.shape).astype(np.float32)
    ay = ((ry - ly) / 2).reshape(X.shape).astype(np.float32)
    if across is not None:
        ax, ay = (np.asarray(v, np.float32) for v in across(ax, ay, X, Y))
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


def body_turn(figure_path, right_path, left_path, right_png, left_png, image_hw, log=print, mode='turn'):
    """
    {canvas: [W, H] of the picture, keyforms: {family: {plus, minus}}, and what was found:
    drift: {family: [dx, dy]} dropped, knees: {side: y} cut at, and for a turn turned:
    {side: dx} which way each drawing turned, mirrored: the side keyed from the other's
    mirror or None}, in the picture's pixels. Neither drawing turned its own way: ValueError.
    """
    use_body_tables()
    figure = split_limbs(tk.Decomposition(figure_path))
    # The two turns fit apart, a process each, started afresh (spawn): forked
    # with OpenCV's threads running, a child can hang.
    fits, flows = {}, {}
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = {side: pool.submit(fit_turned, figure_path, psd, png)
                for side, psd, png in (('right', right_path, right_png), ('left', left_path, left_png))}
        for side, job in jobs.items():
            fits[side], flows[side], lines = job.result()
            for line in lines:
                log(f'{side:5s} {line}')
    return keyed(fits, figure, image_hw, log, mode, flows)


def keyed(fits, figure, image_hw, log=print, mode='turn', flows=None):
    """
    body_turn's result from the two drawings' fits ({side: {family: key}}) on the
    figure's canvas, and each part's flow in them ({side: {family: dx}}; without
    them a turn's travel is the fit's own).
    """
    knees = knees_of(figure)
    split_knees(figure, knees)
    scale, ox, oy = placement(image_hw, (figure.H, figure.W))
    fits = dict(fits)
    turned, mirrored = {}, None
    feet = ('footwear:L', 'footwear:R')

    def flowed(side, families):
        moves = [flows[side][f] for f in families if f in flows[side]]
        return float(np.mean(moves)) if moves else None

    if mode == 'turn':
        for side in ('right', 'left'):
            if flows:
                way = flowed(side, TRUNK)
                way = None if way is None else way - (flowed(side, feet) or 0.0)
            else:
                way = turned_way(fits[side])
            turned[side] = None if way is None else round(way / scale, 1)
            log(f'{side:5s} turned {turned[side]} px')
        right_ok = (turned['right'] or 0) >= TURNED_AT_LEAST
        left_ok = (turned['left'] or 0) <= -TURNED_AT_LEAST
        if right_ok != left_ok:
            mirrored = 'left' if right_ok else 'right'
            good = 'right' if right_ok else 'left'
            moved = np.mean([sole_way(fits[good][f]) for f in feet if f in fits[good]] or [0.0])
            fits[good] = {family: shifted_key(key, -moved) for family, key in fits[good].items()}
            fits[mirrored] = mirrored_fit(fits[good], trunk_axis(figure))
            log(f'{mirrored} drew the wrong turn; keyed as the {good} turn mirrored '
                f'(the figure moved {moved / scale:+.1f} px, taken off)')
        elif not right_ok:
            raise ValueError(f'neither drawing turned its own way: {turned}')
    # Each part's travel in the turn to the right, from the pictures; the trunk's by height.
    travel, profile = {}, None
    if mode == 'turn' and flows:
        if mirrored:
            good = 'left' if mirrored == 'right' else 'right'
            sign = 1 if good == 'right' else -1
            still = flowed(good, feet) or 0.0
            for family in BODY:
                pair = [f for f in {family, MIRRORED.get(family, family)} if f in flows[good]]
                # A mirrored turn is the same either way: a limb goes as both limbs do.
                if pair:
                    travel[family] = sign * (np.mean([flows[good][f] for f in pair]) - still)
            profile = sign * (np.asarray(flows[good]['rows']) - still)
        else:
            travel = {f: (flows['right'][f] - flows['left'][f]) / 2 for f in BODY if f in flows['right'] and f in flows['left']}
            profile = (np.asarray(flows['right']['rows']) - np.asarray(flows['left']['rows'])) / 2
        known = np.nonzero(~np.isnan(profile))[0]
        profile = np.interp(np.arange(len(profile)), known, profile[known]) if len(known) else None

    def as_trunk(ax, ay, X, Y):
        """Every row across by the trunk's travel at its height, and no more."""
        yi = np.clip(Y.astype(int), 0, figure.H - 1)
        return profile[yi], np.zeros_like(ay)

    shared = None
    if mode == 'turn' and profile is not None:
        boxes = [fits[side][f]['box'] for side in ('right', 'left') for f in AS_TRUNK if f in fits[side]]
        if boxes:
            box = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
            still = dict(box=box, grid=TRUNK_GRID, back=[0.0] * (TRUNK_GRID * TRUNK_GRID * 2))
            shared = opposite_keys(still, still, as_trunk)[:2]
    keyforms, drift, plus_way = {}, {}, {}
    for family in BODY:
        if family not in fits['right'] or family not in fits['left']:
            continue
        if mode == 'weight' and family in ONE_SIDED:
            plus, minus = fits['right'][family], fits['left'][family]
        else:
            trunk_row = family in AS_TRUNK and profile is not None
            across = as_trunk if trunk_row else None
            plus, minus, common = opposite_keys(fits['right'][family], fits['left'][family], across)
            drift[family] = [round(common[0] / scale, 1), round(common[1] / scale, 1)]
            log(f'{family:11s} drift dropped ({drift[family][0]:+.1f}, {drift[family][1]:+.1f}) px')
            if trunk_row and shared:
                plus, minus = shared
            if trunk_row:
                points = part_points(figure, family)
                log(f'{family:11s} travels {part_way(plus, points) / scale:+.1f} px as the trunk is drawn, by height')
            elif family in travel:
                points = part_points(figure, family)
                if len(points[0]):
                    fitted = part_way(plus, points)
                    plus = shifted_key(plus, travel[family] - fitted)
                    minus = shifted_key(minus, -travel[family] - part_way(minus, points))
                    log(f'{family:11s} travels {travel[family] / scale:+.1f} px as drawn (fitted {fitted / scale:+.1f})')
        keyforms[family] = {'plus': placed(plus, scale, ox, oy), 'minus': placed(minus, scale, ox, oy)}
        plus_way[family] = key_way(plus)
    if mode == 'turn' and (trunk := [plus_way[f] for f in TRUNK if f in plus_way]):
        # Keyed against the trunk only by a fit gone astray.
        way = np.sign(np.mean(trunk))
        for family, moves in plus_way.items():
            # A part whose travel was read off the pictures goes as drawn.
            measured = family in travel or (family in AS_TRUNK and profile is not None)
            if family not in TRUNK and not measured and moves * way < -TURNED_AT_LEAST * scale:
                if family in ON_TORSO and 'topwear' in keyforms:
                    keyforms[family] = keyforms['topwear']
                    log(f'{family:11s} keyed against the turn ({moves / scale:+.1f} px); takes the top\'s key')
                else:
                    del keyforms[family]
                    log(f'{family:11s} keyed against the turn ({moves / scale:+.1f} px); left unkeyed')
    knees = {side: round((y - oy) / scale, 1) for side, y in knees.items()}
    result = dict(canvas=[image_hw[1], image_hw[0]], keyforms=keyforms, drift=drift, knees=knees)
    if mode == 'turn':
        result.update(turned=turned, mirrored=mirrored)
    return result

if __name__ == '__main__':
    figure_psd, right_psd, left_psd, right_png, left_png, height, width, out = sys.argv[1:9]
    mode = sys.argv[9] if len(sys.argv) > 9 else 'turn'
    if mode not in ('turn', 'weight'):
        sys.exit(f'mode must be turn or weight, not {mode}')
    t0 = time.time()
    result = body_turn(figure_psd, right_psd, left_psd, right_png, left_png, (int(height), int(width)),
                       log=lambda line: print(line, flush=True), mode=mode)
    with open(out, 'w') as f:
        json.dump(result, f)
    print(f'body turn keys in {time.time() - t0:.0f}s', flush=True)
