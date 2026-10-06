"""Keyed head turns from turned references.

A Live2D rigger keys each part of the head at Angle X / Angle Y ±30: the
shape the part takes when the head has turned that far. Here those shapes
are measured. The front portrait and four turned drawings of it (the head
turned toward image right and left, raised and lowered, about 30 degrees,
the body unchanged and lined up) are decomposed alike; each part of the
front decomposition is fitted onto the same part of each turned one.

A key is a backward lattice over the part's bounds in the turned picture:
for each lattice point, how far back to where it was at rest. Its runtime
(Myriad's turnKeyforms) inverts it per vertex.

What a turn uncovers is taken from the turned drawings themselves: for each
hidden pixel of the back hair, front hair, ears and neck, the key says where
it lands; where the turned decomposition shows that same part on top there,
its pixel is baked into the front layer. The face under the hair stays as it
is (skin only, the runtime's to clean).

Rigid accessories (hair clips, earrings) are matched on the illustrations
themselves when they are given: the front picture and the four turned ones
that were decomposed. A decomposition can drop a small piece altogether (an
earring the front decomposition missed, which Myriad's import recovers from
the illustration); such pieces are found where the front picture has art
and no layer does, and are keyed with the rest.

  turn_keyforms.py <front.psd> <right.psd> <left.psd> <up.psd> <down.psd> <out.json> <out.psd>
                   [<front.png> <right.png> <left.png> <up.png> <down.png>]
"""
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage

SIDES = ('plus', 'minus', 'up', 'down')

# Myriad part families by See-through layer name. A See-through "-r" part is
# the character's right, drawn on the image's left: Myriad's side "L".
FAMILIES = {
    'face': ['face'],
    'eye:L': ['eyewhite-r', 'irides-r', 'eyelash-r'],
    'eye:R': ['eyewhite-l', 'irides-l', 'eyelash-l'],
    'brow:L': ['eyebrow-r'],
    'brow:R': ['eyebrow-l'],
    'nose': ['nose'],
    'mouth': ['mouth'],
    'ears': ['ears-r', 'ears-l', 'ears'],
    'neck': ['neck'],
    'neckwear': ['neckwear'],
    'back-hair': ['back hair'],
    'front-hair': ['front hair'],
}
# Finer keys where the drawing holds, coarser where the turned drawing redraws.
GRID = {'face': 17, 'eye:L': 13, 'eye:R': 13, 'ears': 13, 'neck': 13, 'neckwear': 13, 'back-hair': 17}
COARSE_GRID = 9
FRONT_HAIR_GRID = 5
ACCESSORIES = ['headwear', 'earwear']
BAKED = {'back hair': 'back-hair', 'front hair': 'front-hair', 'ears-r': 'ears', 'ears-l': 'ears', 'ears': 'ears', 'neck': 'neck'}


# ---------------------------------------------------------------- layers

class Decomposition:
    def __init__(self, path):
        self.psd = PSDImage.open(path)
        self.W, self.H = self.psd.size
        self.order = []
        self.layers = {}
        for layer in self.psd.descendants():
            if layer.is_group():
                continue
            canvas = np.zeros((self.H, self.W, 4), np.float32)
            image = np.asarray(layer.topil().convert('RGBA')).astype(np.float32) / 255
            y0, x0 = max(0, layer.top), max(0, layer.left)
            y1 = min(self.H, layer.top + image.shape[0]); x1 = min(self.W, layer.left + image.shape[1])
            canvas[y0:y1, x0:x1] = image[y0 - layer.top:y1 - layer.top, x0 - layer.left:x1 - layer.left]
            self.order.append(layer.name)
            self.layers[layer.name] = canvas

    def family(self, names):
        out = np.zeros((self.H, self.W, 4), np.float32)
        for name in self.order:
            if name in names:
                a = self.layers[name][..., 3:4]
                out[..., :3] = out[..., :3] * (1 - a) + self.layers[name][..., :3] * a
                out[..., 3:4] = a + out[..., 3:4] * (1 - a)
        return out

    def composite(self):
        out = np.ones((self.H, self.W, 3), np.float32)
        for name in self.order:
            a = self.layers[name][..., 3:4]
            out = out * (1 - a) + self.layers[name][..., :3] * a
        return out

    def owner(self):
        owner = np.full((self.H, self.W), '', dtype=object)
        for name in self.order:
            owner[self.layers[name][..., 3] > 0.5] = name
        return owner


# ---------------------------------------------------------------- lattices

def bilinear_weights(px, py, xs, ys):
    n = len(xs)
    gx = np.clip((px - xs[0]) / (xs[-1] - xs[0]) * (n - 1), 0, n - 1 - 1e-6)
    gy = np.clip((py - ys[0]) / (ys[-1] - ys[0]) * (n - 1), 0, n - 1 - 1e-6)
    i = np.floor(gx).astype(int); j = np.floor(gy).astype(int)
    tx = gx - i; ty = gy - j
    idx = np.stack([j * n + i, j * n + i + 1, (j + 1) * n + i, (j + 1) * n + i + 1], 1)
    w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], 1)
    return idx, w


def sample(offsets, xs, ys, px, py):
    idx, w = bilinear_weights(px, py, xs, ys)
    return (offsets[idx, 0] * w).sum(1), (offsets[idx, 1] * w).sum(1)


def warp_forward(img, offsets, xs, ys):
    """Each pixel x lands at x + forward(x); resampled by inverting the lattice."""
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sx, sy = xx.ravel().copy(), yy.ravel().copy()
    for _ in range(12):
        ox, oy = sample(offsets, xs, ys, sx, sy)
        sx = xx.ravel() - ox; sy = yy.ravel() - oy
    return remap(img, sx.reshape(H, W), sy.reshape(H, W))


def remap(img, sx, sy):
    pm = np.concatenate([img[..., :3] * img[..., 3:4], img[..., 3:4]], 2).astype(np.float32)
    out = cv2.remap(pm, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = out[..., 3:4]
    return np.concatenate([np.where(a > 1e-4, out[..., :3] / np.maximum(a, 1e-4), 0), a], 2)


def metrics(P, Q):
    pa, qa = P[..., 3] > 0.5, Q[..., 3] > 0.5
    iou = (pa & qa).sum() / max(1, (pa | qa).sum())
    ep = cv2.Canny(pa.astype(np.uint8) * 255, 50, 150) > 0
    eq = cv2.Canny(qa.astype(np.uint8) * 255, 50, 150) > 0
    if not ep.any() or not eq.any():
        return dict(iou=float(iou), outline=float('inf'), outline95=float('inf'))
    dq = cv2.distanceTransform((~eq).astype(np.uint8), cv2.DIST_L2, 5)
    dp = cv2.distanceTransform((~ep).astype(np.uint8), cv2.DIST_L2, 5)
    return dict(iou=round(float(iou), 4), outline=round(float(0.5 * (dq[ep].mean() + dp[eq].mean())), 3),
                outline95=round(float(max(np.percentile(dq[ep], 95), np.percentile(dp[eq], 95))), 2))


def score(m):
    return m['outline'] + 0.1 * m['outline95']


# ---------------------------------------------------------------- fitting

def coarse_forward(A, B, grid, search_steps=()):
    """Forward lattice: centre and extent onto the reference's, dense flow for
    the rest, least squares with smoothness; optionally a coordinate search."""
    h, w = A.shape[:2]
    xs = np.linspace(0, w - 1, grid); ys = np.linspace(0, h - 1, grid)

    def frame(img):
        a = img[..., 3]
        yy, xx = np.nonzero(a > 0.5)
        m = a[yy, xx]
        c = np.array([(xx * m).sum() / m.sum(), (yy * m).sum() / m.sum()])
        ext = np.array([np.percentile(xx, 97) - np.percentile(xx, 3), np.percentile(yy, 97) - np.percentile(yy, 3)])
        return c, np.maximum(ext, 1)

    cA, eA = frame(A); cB, eB = frame(B)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    init = (cB + (pts - cA) * (eB / eA) - pts).astype(np.float32)
    A1 = warp_forward(A, init, xs, ys)

    def gray(img):
        rgb = img[..., :3] * img[..., 3:4] + 0.5 * (1 - img[..., 3:4])
        return cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    dis.setFinestScale(0); dis.setPatchSize(12); dis.setPatchStride(3)
    dis.setGradientDescentIterations(30); dis.setVariationalRefinementIterations(20)
    fwd = dis.calc(gray(A1), gray(B), None)
    bwd = dis.calc(gray(B), gray(A1), None)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    back = cv2.remap(bwd, xx + fwd[..., 0], yy + fwd[..., 1], cv2.INTER_LINEAR)
    sel = (A1[..., 3] > 0.05) & (np.linalg.norm(fwd + back, axis=2) < 2.0)
    py, px = np.nonzero(sel)
    if len(px) < 50:
        py, px = np.nonzero(A1[..., 3] > 0.05)
    step = max(1, len(px) // 40000)
    px, py = px[::step], py[::step]
    qx, qy = px.astype(np.float32), py.astype(np.float32)
    for _ in range(10):
        ox, oy = sample(init, xs, ys, qx, qy)
        qx, qy = px - ox, py - oy
    ox, oy = sample(init, xs, ys, qx, qy)
    target = fwd[py, px] + np.stack([ox, oy], 1)
    idx, wts = bilinear_weights(qx, qy, xs, ys)
    n = grid * grid
    M = np.zeros((len(px), n), np.float32)
    np.put_along_axis(M, idx, wts, 1)
    M = np.vstack([M, laplacian(grid) * 0.5 * np.sqrt(len(px) / n)])
    offsets = np.zeros((n, 2), np.float32)
    for c in range(2):
        rhs = np.concatenate([target[:, c], np.zeros(M.shape[0] - len(px), np.float32)])
        offsets[:, c] = np.linalg.lstsq(M, rhs, rcond=None)[0]

    def cost(off):
        # Rendered through the inverse lattice: the forward warp itself is slow.
        P = full_backward(A, to_backward(off, xs, ys, grid * 2 - 1, w, h))
        pa, qa = P[..., 3], B[..., 3]
        region = (pa > 0.02) | (qa > 0.02)
        return (np.abs(pa - qa)[region].mean() * 4 +
                (np.abs(P[..., :3] - B[..., :3]).mean(2) * np.minimum(pa, qa))[region].mean())

    if search_steps:
        best = cost(offsets)
        for size in search_steps:
            for _ in range(2):
                improved = False
                for k in range(n):
                    for c in range(2):
                        for sign in (1, -1):
                            trial = offsets.copy(); trial[k, c] += sign * size
                            e = cost(trial)
                            if e < best - 1e-6:
                                best, offsets, improved = e, trial, True
                                break
                if not improved:
                    break
    return offsets, xs, ys


def laplacian(grid):
    n = grid * grid
    rows = []
    for j in range(grid):
        for i in range(grid):
            for di, dj in ((1, 0), (0, 1)):
                if i + di < grid and j + dj < grid:
                    r = np.zeros(n, np.float32); r[j * grid + i] = -1; r[(j + dj) * grid + i + di] = 1
                    rows.append(r)
    return np.array(rows)


def to_backward(forward, fxs, fys, grid, w, h):
    """The backward lattice of a forward one: at each turned lattice point t, s - t with s + f(s) = t."""
    xs = np.linspace(0, w - 1, grid); ys = np.linspace(0, h - 1, grid)
    gx, gy = np.meshgrid(xs, ys)
    tx, ty = gx.ravel().astype(np.float32), gy.ravel().astype(np.float32)
    sx, sy = tx.copy(), ty.copy()
    for _ in range(30):
        ox, oy = sample(forward, fxs, fys, sx, sy)
        sx, sy = tx - ox, ty - oy
    return np.stack([sx - tx, sy - ty], 1).reshape(grid, grid, 2).astype(np.float32)


_GRID_MAPS = {}


def _grid_maps(h, w, grid, region):
    """Per pixel of the region, its place on the lattice (corner-aligned), cached."""
    key = (h, w, grid, region)
    if key not in _GRID_MAPS:
        xa, ya, xb, yb = region
        cw, ch = (w - 1) / (grid - 1), (h - 1) / (grid - 1)
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        _GRID_MAPS[key] = (xx, yy, np.clip(xx / cw, 0, grid - 1).astype(np.float32),
                           np.clip(yy / ch, 0, grid - 1).astype(np.float32))
        if len(_GRID_MAPS) > 64:
            _GRID_MAPS.pop(next(iter(_GRID_MAPS)))
    return _GRID_MAPS[key]


def render_backward(A, back, region, cw=None, ch=None):
    h, w = A.shape[:2]
    xx, yy, gx, gy = _grid_maps(h, w, back.shape[0], tuple(int(v) for v in region))
    o = cv2.remap(np.ascontiguousarray(back, np.float32), gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return remap(A, xx + o[..., 0], yy + o[..., 1])


def refine_backward(A, B, back, smooth=0.0005, steps=(8.0, 4.0, 2.0, 1.0, 0.5), rounds=4):
    """Coordinate descent on the lattice, many points at once: points three
    apart touch cells that do not overlap, so each of the nine classes moves
    together, judged on the cells around each point from one render."""
    grid = back.shape[0]
    h, w = A.shape[:2]
    cw, ch = (w - 1) / (grid - 1), (h - 1) / (grid - 1)
    Q = B
    # The pixel box of the cells around each lattice point.
    x0 = np.clip(((np.arange(grid) - 1) * cw).astype(int), 0, w)
    x1 = np.clip(((np.arange(grid) + 1) * cw + 1).astype(int), 0, w)
    y0 = np.clip(((np.arange(grid) - 1) * ch).astype(int), 0, h)
    y1 = np.clip(((np.arange(grid) + 1) * ch + 1).astype(int), 0, h)

    def region_costs(lattice):
        P = render_backward(A, lattice, (0, 0, w, h), cw, ch)
        pa, qa = P[..., 3], Q[..., 3]
        E = np.abs(pa - qa) * 4 + np.abs(P[..., :3] - Q[..., :3]).mean(2) * np.minimum(pa, qa)
        S = np.zeros((h + 1, w + 1), np.float64)
        S[1:, 1:] = E.cumsum(0).cumsum(1)
        return (S[y1][:, x1] - S[y0][:, x1] - S[y1][:, x0] + S[y0][:, x0])

    def smoothness(lattice):
        padded = np.pad(lattice, ((1, 1), (1, 1), (0, 0)), mode='edge')
        mean = (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) / 4
        return smooth * ((lattice - mean) ** 2).sum(2) * cw * ch

    # Only points with something to fit nearby take part.
    live = region_costs(back) > 0
    alpha = np.maximum(A[..., 3], Q[..., 3])
    S = np.zeros((h + 1, w + 1)); S[1:, 1:] = (alpha > 0.02).cumsum(0).cumsum(1)
    live |= (S[y1][:, x1] - S[y0][:, x1] - S[y1][:, x0] + S[y0][:, x0]) > 0
    classes = [(ci, cj) for cj in range(3) for ci in range(3)]
    for step in steps:
        for _ in range(rounds):
            moved = 0
            for ci, cj in classes:
                mask = np.zeros((grid, grid), bool)
                mask[cj::3, ci::3] = True
                mask &= live
                if not mask.any():
                    continue
                base = region_costs(back) + smoothness(back)
                best = base.copy()
                best_move = np.zeros((grid, grid, 2), np.float32)
                for c in range(2):
                    for sign in (1, -1):
                        trial = back.copy()
                        trial[..., c][mask] += sign * step
                        cost = region_costs(trial) + smoothness(trial)
                        better = mask & (cost < best - 1e-6)
                        best[better] = cost[better]
                        best_move[better] = 0
                        best_move[..., c][better] = sign * step
                changed = mask & (best < base - 1e-6)
                if changed.any():
                    back[changed] += best_move[changed]
                    moved += int(changed.sum())
            if moved == 0:
                break
    return back


def full_backward(A, back):
    grid = back.shape[0]
    h, w = A.shape[:2]
    return render_backward(A, back, (0, 0, w, h), (w - 1) / (grid - 1), (h - 1) / (grid - 1))


def fit_family(front, turned, names, family, log):
    A = front.family(names)
    B = turned.family(names)
    if (A[..., 3] > 0.5).sum() < 150 or (B[..., 3] > 0.5).sum() < 150:
        return None
    ys_, xs_ = np.nonzero((A[..., 3] > 0.05) | (B[..., 3] > 0.05))
    pad = 24
    x0, y0 = max(0, xs_.min() - pad), max(0, ys_.min() - pad)
    x1, y1 = min(front.W - 1, xs_.max() + pad), min(front.H - 1, ys_.max() + pad)
    A = np.ascontiguousarray(A[y0:y1 + 1, x0:x1 + 1]); B = np.ascontiguousarray(B[y0:y1 + 1, x0:x1 + 1])
    h, w = A.shape[:2]
    t0 = time.time()
    if family == 'front-hair':
        # The turned drawing redraws the locks: a low-order warp searched whole.
        forward, fxs, fys = coarse_forward(A, B, FRONT_HAIR_GRID, search_steps=(6.0, 3.0, 1.5))
        back = to_backward(forward, fxs, fys, COARSE_GRID, w, h)
        chosen = metrics(full_backward(A, back), B)
    else:
        forward, fxs, fys = coarse_forward(A, B, COARSE_GRID)
        grid = GRID.get(family, COARSE_GRID)
        seed = to_backward(forward, fxs, fys, grid, w, h)
        seed_m = metrics(full_backward(A, seed), B)
        back = refine_backward(A, B, seed.copy())
        chosen = metrics(full_backward(A, back), B)
        if score(seed_m) < score(chosen):
            back, chosen = seed, seed_m
    log(f'{family:11s} {chosen} ({time.time() - t0:.0f}s)')
    return dict(box=[float(x0), float(y0), float(x1), float(y1)], grid=int(back.shape[0]),
                back=[float(v) for v in back.reshape(-1)], fit=chosen)


# ---------------------------------------------------------------- accessories

# The importer's own thresholds (Myriad psdReconciliation): art is this far
# (RGB distance) from a flat background; a layer showing art this far from the
# illustration's holds different art; pieces smaller than this are fringe.
BACKGROUND_DISTANCE = 28
MISMATCH_DISTANCE = 98
MIN_MISSING_AREA = 400


def placed_illustration(path, W, H):
    """
    An illustration where its decomposition put it: scaled by one factor to fit
    the canvas and centred (See-through's fit_pad_resize). Returns RGB in 0..1
    over white, and where it has art: its alpha, or else its distance from the
    background colour read off its border.
    """
    img = np.asarray(Image.open(path).convert('RGBA')).astype(np.float32) / 255
    sh, sw = img.shape[:2]
    scale = min(W / sw, H / sh)
    w, h = min(W, int(np.floor(sw * scale + 0.5))), min(H, int(np.floor(sh * scale + 0.5)))
    x, y = (W - w) // 2, (H - h) // 2
    interpolation = cv2.INTER_AREA if w * h < sw * sh else cv2.INTER_LINEAR
    img = cv2.resize(img, (w, h), interpolation=interpolation) if (w, h) != (sw, sh) else img
    rgb = np.ones((H, W, 3), np.float32)
    art = np.zeros((H, W), bool)
    a = img[..., 3:4]
    rgb[y:y + h, x:x + w] = img[..., :3] * a + (1 - a)
    if (a < 0.5).mean() > 0.02:
        art[y:y + h, x:x + w] = a[..., 0] > 0.3
    else:
        border = np.concatenate([img[0, :, :3], img[-1, :, :3], img[:, 0, :3], img[:, -1, :3]])
        background = np.median(border, 0)
        art[y:y + h, x:x + w] = np.linalg.norm(img[..., :3] - background, axis=-1) * 255 > BACKGROUND_DISTANCE
    return rgb, art


def missing_pieces(front, rgb, art):
    """
    Pieces of the head the front picture has and no front layer does: art no
    layer covers, or covers with other art (an earring the decomposition drew
    as hair). Myriad's import lifts these from the illustration as they are.
    """
    alpha = np.zeros((front.H, front.W), np.float32)
    for name in front.order:
        a = front.layers[name][..., 3]
        alpha = a + alpha * (1 - a)
    covered = cv2.dilate((alpha > 0.3).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    other = np.linalg.norm(front.composite() - rgb, axis=-1) * 255 > MISMATCH_DISTANCE
    miss = cv2.morphologyEx((art & (~covered | other)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    head = np.zeros((front.H, front.W), bool)
    for name in front.order:
        if name.split('-')[0] in ('face', 'front hair', 'back hair', 'ears', 'headwear', 'earwear'):
            head |= front.layers[name][..., 3] > 0.3
    if not head.any() or not miss.any():
        return []
    ys, xs = np.nonzero(head)
    zone = (xs.min() - 40, ys.min() - 40, xs.max() + 40, ys.max() + 40)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(cv2.dilate(miss, np.ones((9, 9), np.uint8)))
    pieces = []
    for k in range(1, n):
        cx, cy = cents[k]
        if not (zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]):
            continue
        mask = (lab == k) & (miss > 0)
        if mask.sum() >= MIN_MISSING_AREA:
            pieces.append(mask)
    return pieces


def accessory_pieces(front):
    pieces = []
    for name in front.order:
        if name.split('-')[0] not in ACCESSORIES:
            continue
        a = (front.layers[name][..., 3] > 0.3).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(a, np.ones((9, 9), np.uint8)))
        if n < 2:
            continue
        big = stats[1:, 4].max()
        for k in range(1, n):
            if stats[k, 4] >= big * 0.05:
                pieces.append(((lab == k) & (a > 0), name))
    return pieces


def match_piece(mask, front_rgb, turned_rgb, side, face_cx, earring):
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    tmpl = front_rgb[y0:y1, x0:x1].astype(np.float32)
    m = mask[y0:y1, x0:x1].astype(np.float32)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if side in ('plus', 'minus'):
        far = (cx > face_cx) == (side == 'plus')
        squeeze = (0.85, 1.05) if earring else ((0.6, 1.0) if far else (0.9, 1.15))
        reach = (260, 120)
    else:
        squeeze, reach = (0.9, 1.1), (30, 90)
    H, W = turned_rgb.shape[:2]
    best = None
    for s in np.linspace(0.85, 1.1, 6):
        for q in np.linspace(squeeze[0], squeeze[1], 10):
            tw, th = max(4, int((x1 - x0) * s * q)), max(4, int((y1 - y0) * s))
            t = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
            mk = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
            sx0, sy0 = int(max(0, cx - tw / 2 - reach[0])), int(max(0, cy - th / 2 - reach[1]))
            sx1, sy1 = int(min(W, cx + tw / 2 + reach[0])), int(min(H, cy + th / 2 + reach[1]))
            region = turned_rgb[sy0:sy1, sx0:sx1].astype(np.float32)
            if region.shape[0] < th or region.shape[1] < tw:
                continue
            r = cv2.matchTemplate(region, t, cv2.TM_SQDIFF, mask=np.repeat(mk[..., None], 3, 2)) / max(1.0, mk.sum())
            _, _, mn, _ = cv2.minMaxLoc(r)
            if best is None or r[mn[1], mn[0]] < best[0]:
                best = (r[mn[1], mn[0]], s, q, sx0 + mn[0] + tw / 2, sy0 + mn[1] + th / 2)
    _, s, q, tx, ty = best
    return dict(source=(cx, cy), target=(tx, ty), scale=float(s), squeeze=float(q),
                box=(int(x0), int(y0), int(x1), int(y1)))


def accessory_lattice(found, grid=25):
    """Backward lattice over where the pieces went: each turned point comes from its piece, undone."""
    tos = []
    for g in found:
        bx0, by0, bx1, by1 = g['box']
        hw = (bx1 - bx0) / 2 * g['scale'] * g['squeeze']; hh = (by1 - by0) / 2 * g['scale']
        tos.append((g['target'][0] - hw, g['target'][1] - hh, g['target'][0] + hw, g['target'][1] + hh))
    tos = np.array(tos)
    x0, y0 = tos[:, 0].min() - 20, tos[:, 1].min() - 20
    x1, y1 = tos[:, 2].max() + 20, tos[:, 3].max() + 20
    back = []
    for y in np.linspace(y0, y1, grid):
        for x in np.linspace(x0, x1, grid):
            d = [max(0, b[0] - x, x - b[2]) ** 2 + max(0, b[1] - y, y - b[3]) ** 2 for b in tos]
            g = found[int(np.argmin(d))]
            sx = g['source'][0] + (x - g['target'][0]) / (g['scale'] * g['squeeze'])
            sy = g['source'][1] + (y - g['target'][1]) / g['scale']
            back += [float(sx - x), float(sy - y)]
    return dict(box=[float(x0), float(y0), float(x1), float(y1)], grid=grid, back=back)


# ---------------------------------------------------------------- the picture itself

# The decompositions say which part is which; the turned picture says where it
# is. A decomposition can redraw an outline (a raised chin drawn lower than the
# picture has it), and a key fitted to that inherits it. So the keys are
# rendered together, as the runtime draws them, and compared with the picture
# by dense flow; each part's lattice takes up the flow where that part shows.
PICTURE_ROUNDS = 3
PICTURE_STRIDE = 3
PICTURE_SMOOTH = 0.5
PICTURE_HOLD = 0.05


def family_of(name):
    for family, names in FAMILIES.items():
        if name in names:
            return family
    base = name.split('-')[0]
    return base if base in ACCESSORIES else None


def lattice_maps(key, W, H):
    """Per canvas pixel, the backward offset of a key (its lattice clamped at the box, as the runtime samples it)."""
    x0, y0, x1, y1 = key['box']
    g = key['grid']
    back = np.asarray(key['back'], np.float32).reshape(g, g, 2)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    gx = np.clip((xx - x0) / max(1e-6, x1 - x0) * (g - 1), 0, g - 1).astype(np.float32)
    gy = np.clip((yy - y0) / max(1e-6, y1 - y0) * (g - 1), 0, g - 1).astype(np.float32)
    o = cv2.remap(back, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return xx + o[..., 0], yy + o[..., 1]


def render_keys(layers, keys, side, W, H):
    """The front layers drawn at one key, back to front: RGB over white and which layer is on top."""
    rgb = np.ones((H, W, 3), np.float32)
    top = np.full((H, W), -1, np.int32)
    maps = {}
    for i, (name, family, layer) in enumerate(layers):
        key = keys.get(family, {}).get(side) if family else None
        if key is not None:
            if family not in maps:
                maps[family] = lattice_maps(key, W, H)
            layer = remap(layer, *maps[family])
        a = layer[..., 3:4]
        rgb = rgb * (1 - a) + layer[..., :3] * a
        top[a[..., 0] > 0.5] = i
    return rgb, top


def picture_flow(render, picture):
    """For each picture pixel p, where the same thing is in the render (p + f), and whether both ways agree."""
    gray = lambda img: cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0)
    f = dis.calc(gray(picture), gray(render), None)
    r = dis.calc(gray(render), gray(picture), None)
    H, W = f.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    back = cv2.remap(r, xx + f[..., 0], yy + f[..., 1], cv2.INTER_LINEAR)
    return f, np.linalg.norm(f + back, axis=2) < 1.5


def head_edges(rgb):
    return cv2.Canny(cv2.cvtColor(cv2.GaussianBlur((np.clip(rgb, 0, 1) * 255).astype(np.uint8), (3, 3), 0),
                                  cv2.COLOR_RGB2GRAY), 60, 140) > 0


def edge_distance(a, b, region):
    ea, eb = head_edges(a) & region, head_edges(b) & region
    if ea.sum() < 20 or eb.sum() < 20:
        return 0.0
    da = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 5)[ea]
    db = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 5)[eb]
    return 0.5 * (float(da.mean()) + float(db.mean()))


def refit_lattice(key, px, py, target):
    """Least squares of a lattice through backward offsets measured at turned points, smooth, held near the old one."""
    x0, y0, x1, y1 = key['box']
    g = key['grid']
    n = g * g
    old = np.asarray(key['back'], np.float32).reshape(n, 2)
    xs = np.linspace(x0, x1, g); ys = np.linspace(y0, y1, g)
    idx, wts = bilinear_weights(px, py, xs, ys)
    M = np.zeros((len(px), n), np.float32)
    np.put_along_axis(M, idx, wts, 1)
    scale = np.sqrt(len(px) / n)
    L = laplacian(g)
    A = np.vstack([M, L * PICTURE_SMOOTH * scale, np.eye(n, dtype=np.float32) * PICTURE_HOLD * scale])
    out = np.zeros((n, 2), np.float32)
    for c in range(2):
        rhs = np.concatenate([target[:, c], L @ old[:, c] * PICTURE_SMOOTH * scale, old[:, c] * PICTURE_HOLD * scale])
        out[:, c] = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return dict(key, back=[float(v) for v in out.reshape(-1)])


def refine_on_picture(front, keys, side, picture, extra, log):
    """
    Moves each keyed part toward where the turned picture has it. extra:
    (name, family, layer) pieces drawn above the front layers (recovered art).
    Keys change in place; a part keeps a change only where it draws closer.
    """
    W, H = front.W, front.H
    layers = [(name, family_of(name), front.layers[name]) for name in front.order] + list(extra)
    head = np.zeros((H, W), bool)
    for name, family, layer in layers:
        if family and family in keys:
            head |= layer[..., 3] > 0.3
    ys, xs = np.nonzero(head)
    if not len(xs):
        return None
    region = np.zeros((H, W), bool)
    region[max(0, ys.min() - 60):ys.max() + 60, max(0, xs.min() - 60):xs.max() + 60] = True
    rgb, top = render_keys(layers, keys, side, W, H)
    start = before = edge_distance(rgb, picture, region)
    for _ in range(PICTURE_ROUNDS):
        flow, agree = picture_flow(rgb, picture)
        yy, xx = np.mgrid[0:H:PICTURE_STRIDE, 0:W:PICTURE_STRIDE]
        fx, fy = flow[yy, xx, 0], flow[yy, xx, 1]
        tx = np.clip(np.round(xx + fx).astype(int), 0, W - 1); ty = np.clip(np.round(yy + fy).astype(int), 0, H - 1)
        shown = top[ty, tx]
        ok = agree[yy, xx] & region[yy, xx] & (shown >= 0)
        trial = dict(keys)
        for family in {f for _, f, _ in layers if f and f in keys and side in keys[f]}:
            if family in ACCESSORIES:
                continue
            mine = np.array([i for i, (_, f, _) in enumerate(layers) if f == family])
            sel = ok & np.isin(shown, mine)
            if sel.sum() < 200:
                continue
            key = keys[family][side]
            px, py = xx[sel].astype(np.float32), yy[sel].astype(np.float32)
            qx, qy = px + fx[sel], py + fy[sel]
            x0, y0, x1, y1 = key['box']
            g = key['grid']
            bx, by = sample(np.asarray(key['back'], np.float32).reshape(-1, 2), np.linspace(x0, x1, g), np.linspace(y0, y1, g), qx, qy)
            target = np.stack([fx[sel] + bx, fy[sel] + by], 1)
            inside = (px >= x0) & (px <= x1) & (py >= y0) & (py <= y1)
            if inside.sum() < 200:
                continue
            trial[family] = dict(keys[family], **{side: refit_lattice(key, px[inside], py[inside], target[inside])})
        new_rgb, new_top = render_keys(layers, trial, side, W, H)
        # Each part keeps its change only if its own outline draws closer.
        accepted = []
        for family in [f for f in trial if trial[f] is not keys[f]]:
            mine = np.array([i for i, (_, f, _) in enumerate(layers) if f == family])
            near = cv2.dilate((np.isin(top, mine) | np.isin(new_top, mine)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            old_d = edge_distance(rgb, picture, region & near)
            new_d = edge_distance(new_rgb, picture, region & near)
            if new_d < old_d - 0.02:
                keys[family] = trial[family]
                accepted.append(f'{family} {old_d:.2f}->{new_d:.2f}')
        if not accepted:
            break
        rgb, top = render_keys(layers, keys, side, W, H)
        after = edge_distance(rgb, picture, region)
        log(f'{side:5s} picture: {before:.2f} -> {after:.2f} ({", ".join(accepted)})')
        before = after
    log(f'{side:5s} picture total {start:.2f} -> {before:.2f}')
    return [round(start, 3), round(before, 3)]


# ---------------------------------------------------------------- baking

def turned_position(key, qx, qy):
    x0, y0, x1, y1 = key['box']
    g = key['grid']
    back = np.array(key['back'], np.float32).reshape(g, g, 2)
    xs = np.linspace(x0, x1, g); ys = np.linspace(y0, y1, g)
    offsets = back.reshape(-1, 2)
    tx, ty = qx.astype(np.float32).copy(), qy.astype(np.float32).copy()
    for _ in range(30):
        bx, by = sample(offsets, xs, ys, tx, ty)
        tx, ty = qx - bx, qy - by
    return tx, ty


def bake(front, turned, keys, log):
    """Hidden pixels of the parts a turn uncovers, from the turned drawings where the same part is on top."""
    owners = {side: t.owner() for side, t in turned.items()}
    report = {}
    for li, name in enumerate(front.order):
        family = BAKED.get(name)
        if not family or family not in keys:
            continue
        layer = front.layers[name]
        cover = np.zeros((front.H, front.W), np.float32)
        for above in front.order[li + 1:]:
            cover = np.maximum(cover, front.layers[above][..., 3])
        hidden = (layer[..., 3] > 0.25) & (cover > 0.9)
        qy, qx = np.nonzero(hidden)
        filled = np.zeros((front.H, front.W), bool)
        out = layer.copy()
        taken = {}
        for side, t in turned.items():
            key = keys[family].get(side)
            if key is None:
                continue
            tx, ty = turned_position(key, qx.astype(np.float32), qy.astype(np.float32))
            ix = np.clip(np.round(tx).astype(int), 0, front.W - 1); iy = np.clip(np.round(ty).astype(int), 0, front.H - 1)
            mine = np.isin(owners[side][iy, ix], FAMILIES[family]) & ~filled[qy, qx]
            src = t.family(FAMILIES[family])
            x0 = np.clip(np.floor(tx).astype(int), 0, front.W - 2); y0 = np.clip(np.floor(ty).astype(int), 0, front.H - 2)
            fx = np.clip(tx - x0, 0, 1)[:, None]; fy = np.clip(ty - y0, 0, 1)[:, None]
            vals = (src[y0, x0] * (1 - fx) * (1 - fy) + src[y0, x0 + 1] * fx * (1 - fy) +
                    src[y0 + 1, x0] * (1 - fx) * fy + src[y0 + 1, x0 + 1] * fx * fy)
            use = mine & (vals[:, 3] > 0.5)
            out[qy[use], qx[use], :3] = vals[use, :3]
            filled[qy[use], qx[use]] = True
            taken[side] = int(use.sum())
        weight = cv2.GaussianBlur(filled.astype(np.float32), (7, 7), 0) * filled
        layer[..., :3] = layer[..., :3] * (1 - weight[..., None]) + out[..., :3] * weight[..., None]
        report[name] = dict(hidden=int(hidden.sum()), **taken)
    log(f'baked {report}')
    return report


def save_psd(front, path):
    from PIL import Image
    out = PSDImage.new(mode='RGBA', size=(front.W, front.H), depth=8)
    for name in front.order:
        f = front.layers[name]
        ys, xs = np.nonzero(f[..., 3] > 0)
        if len(ys) == 0:
            continue
        t, b, l, r = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = (np.clip(f[t:b, l:r], 0, 1) * 255 + 0.5).astype(np.uint8)
        out.create_pixel_layer(Image.fromarray(crop, 'RGBA'), name=name, top=int(t), left=int(l), opacity=255)
    out.save(path)


# ---------------------------------------------------------------- entry

def _fit_direction(front_path, turned_path):
    front = Decomposition(front_path)
    turned = Decomposition(turned_path)
    lines = []
    fits = {}
    for family, names in FAMILIES.items():
        if not any(n in front.layers for n in names) or not any(n in turned.layers for n in names):
            continue
        fit = fit_family(front, turned, names, family, lines.append)
        if fit:
            fits[family] = fit
    return fits, lines


def _refine_direction(front_path, keys, side, front_picture, picture_path, recovered):
    front = Decomposition(front_path)
    rgb, _ = placed_illustration(front_picture, front.W, front.H)
    extra = [('recovered', 'earwear', np.concatenate([rgb, m[..., None].astype(np.float32)], 2)) for m in recovered]
    picture, _ = placed_illustration(picture_path, front.W, front.H)
    keys = {family: dict(sides) for family, sides in keys.items()}
    lines = []
    change = refine_on_picture(front, keys, side, picture, extra, lines.append)
    return {family: sides[side] for family, sides in keys.items() if side in sides}, change, lines


def turn_keyforms(front_path, turned_paths, log=print, pictures=None):
    """
    Keys and baked front decomposition. turned_paths: {'plus','minus','up','down'}
    -> PSD; pictures, optional: the same with 'front', the illustrations decomposed.
    """
    front = Decomposition(front_path)
    turned = {side: Decomposition(p) for side, p in turned_paths.items()}
    for side, t in turned.items():
        if (t.W, t.H) != (front.W, front.H):
            raise ValueError(f'{side}: canvas {t.W}x{t.H} is not the front canvas {front.W}x{front.H}')
    keys = {}
    # The four directions fit apart, one process each where the machine has them.
    workers = max(1, min(len(turned_paths), (os.cpu_count() or 1)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {side: pool.submit(_fit_direction, front_path, path) for side, path in turned_paths.items()}
        for side, job in jobs.items():
            fits, lines = job.result()
            for line in lines:
                log(f'{side:5s} {line}')
            for family, fit in fits.items():
                keys.setdefault(family, {})[side] = fit
    # Accessories ride rigidly: every piece of every hair clip and earring,
    # matched on the illustrations when given (they hold what the
    # decompositions re-render or drop).
    face = front.layers.get('face')
    face_cx = float(np.nonzero(face[..., 3] > 0.5)[1].mean()) if face is not None else front.W / 2
    pieces = [(mask, name.startswith('earwear')) for mask, name in accessory_pieces(front)]
    drawn = {}
    recovered = []
    if pictures:
        drawn = {side: placed_illustration(path, front.W, front.H) for side, path in pictures.items()}
        ears = [front.layers[n][..., 3] > 0.3 for n in front.order if n.split('-')[0] == 'ears']
        ears_top = min(np.nonzero(e)[0].min() for e in ears if e.any()) if any(e.any() for e in ears) else front.H
        recovered = missing_pieces(front, *drawn['front'])
        for mask in recovered:
            # Below the top of the ears a recovered piece hangs (an earring); above, it is a clip.
            pieces.append((mask, np.nonzero(mask)[0].mean() > ears_top))
            ys, xs = np.nonzero(mask)
            log(f'recovered piece at ({xs.mean():.0f}, {ys.mean():.0f}), {mask.sum()} px')
    front_rgb = drawn['front'][0] if drawn else front.composite()
    if pieces:
        for side, t in turned.items():
            turned_rgb = drawn[side][0] if side in drawn else t.composite()
            found = [match_piece(mask, front_rgb, turned_rgb, side, face_cx, earring)
                     for mask, earring in pieces]
            lattice = accessory_lattice(found)
            for family in ACCESSORIES:
                keys.setdefault(family, {})[side] = lattice
            log(f'{side:5s} accessories {[(round(f["target"][0]), round(f["target"][1]), round(f["squeeze"], 2)) for f in found]}')
    # A family is keyed only with both turn keys; nod keys only as a pair.
    complete = {}
    for family, sides in keys.items():
        if 'plus' not in sides or 'minus' not in sides:
            continue
        entry = {s: {k: v for k, v in sides[s].items() if k != 'fit'} for s in ('plus', 'minus')}
        if 'up' in sides and 'down' in sides:
            entry['up'] = {k: v for k, v in sides['up'].items() if k != 'fit'}
            entry['down'] = {k: v for k, v in sides['down'].items() if k != 'fit'}
        complete[family] = entry
    report = {f: {s: v.get('fit') for s, v in sides.items() if 'fit' in v} for f, sides in keys.items()}
    baked = bake(front, turned, complete, log)
    result = dict(canvas=[front.W, front.H], keyforms=complete, fit=report, baked=baked)
    if drawn:
        # Drawn together with what they uncover, as the runtime draws them, the
        # keys move to where the pictures have the parts; then what they
        # uncover is taken again for the moved keys.
        with tempfile.TemporaryDirectory() as tmp:
            baked_path = os.path.join(tmp, 'baked.psd')
            save_psd(front, baked_path)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                jobs = {side: pool.submit(_refine_direction, baked_path, complete, side, pictures['front'], pictures[side], recovered)
                        for side in turned if side in pictures}
                result['picture'] = {}
                for side, job in jobs.items():
                    moved, change, lines = job.result()
                    for line in lines:
                        log(line)
                    for family, key in moved.items():
                        complete[family][side] = key
                    result['picture'][side] = change
        result['baked'] = bake(front, turned, complete, log)
    return front, result


if __name__ == '__main__':
    front_path, right, left, up, down, out_json, out_psd = sys.argv[1:8]
    pictures = dict(zip(('front', 'plus', 'minus', 'up', 'down'), sys.argv[8:13])) if len(sys.argv) >= 13 else None
    front, result = turn_keyforms(front_path, dict(plus=right, minus=left, up=up, down=down), pictures=pictures)
    save_psd(front, out_psd)
    json.dump(result, open(out_json, 'w'))
