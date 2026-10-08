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
hidden pixel of the back hair, front hair and ears, the key says where
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
import re
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage

SIDES = ('plus', 'minus', 'up', 'down')


def available_cpus():
    """
    The CPUs this process may use: a container's quota (cgroup cpu.max) and
    affinity, not the host's count (which oversubscribes a 2-CPU Space many
    times over).
    """
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    try:
        quota, period = open('/sys/fs/cgroup/cpu.max').read().split()
        if quota != 'max':
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def _limit_threads(threads):
    cv2.setNumThreads(threads)

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
GRID = {'face': 17, 'eye:L': 13, 'eye:R': 13, 'ears': 13, 'neck': 13, 'neckwear': 13, 'back-hair': 17, 'headwear': 13}
COARSE_GRID = 9
FRONT_HAIR_GRID = 5
ACCESSORIES = ['headwear', 'earwear']
# Headwear as big as the head (headphones, a hood, a hat) is no clip: it turns
# in depth, its near side growing and its far side going behind the head, which
# no rigid move draws. When it covers BIG_HEADWEAR of the face's area and every
# turned decomposition has it at about its size (BIG_HEADWEAR_RATIO), it is
# fitted on them as a part of its own, like the face.
BIG_HEADWEAR = 0.5
BIG_HEADWEAR_RATIO = (0.5, 2.0)
# A turned decomposition can drop most of it (headphones kept at a tenth of
# their size): fitted under BIG_HEADWEAR_IOU, or not there at its size, it
# starts from the head's own move (the face key's affine part, over the
# headwear's box and HEAD_KEY_PAD around it) and is fitted on the pictures.
BIG_HEADWEAR_IOU = 0.8
HEAD_KEY_PAD = 60


def head_key(face_key, box, face, grid=13):
    """
    A backward lattice over `box` moving as the face key does on the whole (its
    affine part), read where the key draws the face (`face`: its front alpha);
    the lattice off the face holds nothing.
    """
    fx0, fy0, fx1, fy1 = face_key['box']
    g = face_key['grid']
    back = np.asarray(face_key['back'], np.float64).reshape(g * g, 2)
    yy, xx = np.meshgrid(np.linspace(fy0, fy1, g), np.linspace(fx0, fx1, g), indexing='ij')
    P = np.stack([xx.reshape(-1), yy.reshape(-1), np.ones(g * g)], 1)
    sx = np.clip(np.round(P[:, 0] + back[:, 0]).astype(int), 0, face.shape[1] - 1)
    sy = np.clip(np.round(P[:, 1] + back[:, 1]).astype(int), 0, face.shape[0] - 1)
    on = face[sy, sx] > 0.5
    if on.sum() >= 6:
        P, back = P[on], back[on]
    M, *_ = np.linalg.lstsq(P, back, rcond=None)
    x0, y0, x1, y1 = box
    yy, xx = np.meshgrid(np.linspace(y0, y1, grid), np.linspace(x0, x1, grid), indexing='ij')
    out = np.stack([xx.reshape(-1), yy.reshape(-1), np.ones(grid * grid)], 1) @ M
    return dict(box=[float(v) for v in box], grid=grid, back=[float(v) for v in out.reshape(-1)])


def big_headwear(dec):
    """The headwear layer's area over the face's when it is as big as BIG_HEADWEAR, else None."""
    if 'headwear' not in dec.layers or 'face' not in dec.layers:
        return None
    face = (dec.layers['face'][..., 3] > 0.5).sum()
    area = (dec.layers['headwear'][..., 3] > 0.5).sum()
    return area if face and area >= BIG_HEADWEAR * face else None
# The neck is not taken from the turned drawings: each is lit its own way, and
# under the chin they patch into bands. The decomposition's own inpainting is
# one smooth cylinder of skin with the chin's shadow, which the neck's key
# carries up with the chin.
BAKED = {'back hair': 'back-hair', 'front hair': 'front-hair', 'ears-r': 'ears', 'ears-l': 'ears', 'ears': 'ears'}
# The front hair as locks (hair_locks.py): layers 'front hair-1'.., families
# 'front-hair:1'.., each fitted against the turned decompositions' whole front hair.
LOCK_LAYER = re.compile(r'front hair-(\d+)$')


def lock_family(name):
    m = LOCK_LAYER.match(name)
    return f'front-hair:{m.group(1)}' if m else None


def turned_names(family):
    """The layers that draw a family in a turned decomposition, which keeps its front hair whole."""
    return ['front hair'] if family.startswith('front-hair:') else FAMILIES[family]


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
    # The head itself; back hair long enough to reach the waist would take in
    # the body (an obi the decomposition left out is not on the head).
    head = np.zeros((front.H, front.W), bool)
    for name in front.order:
        if name.split('-')[0] in ('face', 'front hair', 'ears', 'headwear', 'earwear'):
            head |= front.layers[name][..., 3] > 0.3
    if not head.any() or not miss.any():
        return []
    ys, xs = np.nonzero(head)
    zone = (xs.min() - 40, ys.min() - 40, xs.max() + 40, ys.max() + 40)
    # What hangs from the hair reaches out of that (a bow with its ribbons):
    # it is the head's too when it touches the head or the back hair above the
    # chin and is mostly above the chin (Myriad's import mounts it so).
    face = front.layers.get('face')
    chin = np.nonzero(face[..., 3] > 0.5)[0].max() if face is not None and (face[..., 3] > 0.5).any() else front.H
    held = head.copy()
    if 'back hair' in front.layers:
        held[:chin] |= front.layers['back hair'][:chin, :, 3] > 0.3
    held = cv2.dilate(held.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    n, lab, stats, cents = cv2.connectedComponentsWithStats(cv2.dilate(miss, np.ones((9, 9), np.uint8)))
    pieces = []
    for k in range(1, n):
        cx, cy = cents[k]
        inside = zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]
        if not inside and not (cy < chin and (held & (lab == k)).any()):
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


# A piece rides what holds it (an earring the ear, a clip the hair): it is
# looked for this far around where that host's key carries it (a share of the
# piece's size, at least ACCESSORY_REACH px), and stays there when nothing in
# reach looks like it (mean squared colour error per pixel over this). The
# reach is what keeps a match from the wrong piece (an earring found on the
# other ear matched at 0.05); a right one foreshortened by the turn can match
# at 0.1 (a hair clip on the far side).
ACCESSORY_REACH = 30
ACCESSORY_REACH_SHARE = 0.5
ACCESSORY_MATCH_ERROR = 0.12
# An earring's hook lies within this many px of the ear.
EAR_HOOK_REACH = 15
# A piece of the outfit rides the neck when this share of its rim is the neck.
ACCESSORY_NECK_SHARE = 0.1


# Pieces this close are one accessory (a hairpin and the tassels the
# decomposition dropped, which the import recovers on their own): they move
# as one. Matched apart, each lands a little elsewhere and the lattice
# between them smears the accessory.
ACCESSORY_GROUP_GAP = 12


def group_pieces(front, pieces):
    """
    Pieces (mask, earring) that touch within ACCESSORY_GROUP_GAP, merged, with
    the host of each (accessory_host). Pieces of the outfit stay apart.
    earring: True for a drawn earring, False for a drawn hair ornament, None
    for a recovered piece; a group with a drawn hair ornament is not an
    earring.
    """
    hosts = [accessory_host(front, mask, earring) for mask, earring in pieces]
    worn = [piece for piece, host in zip(pieces, hosts) if host is not None]
    outfit = [piece for piece, host in zip(pieces, hosts) if host is None]
    if len(worn) >= 2:
        k = 2 * (ACCESSORY_GROUP_GAP // 2) + 1
        near = np.zeros(worn[0][0].shape, np.uint8)
        for mask, _ in worn:
            near |= cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8))
        _, lab = cv2.connectedComponents(near)
        groups = {}
        for mask, earring in worn:
            g = int(np.bincount(lab[mask]).argmax())
            groups[g] = (groups[g][0] | mask, joined(groups[g][1], earring)) if g in groups else (mask, earring)
        worn = list(groups.values())
        hosts = [accessory_host(front, mask, earring) for mask, earring in worn] + [None] * len(outfit)
    else:
        hosts = [host for host in hosts if host is not None] + [None] * len(outfit)
    return worn + outfit, hosts


def hangs_from_ear(front, mask):
    """Whether a piece hangs from an ear: the top of it (the hook) is at the ear."""
    ears = np.zeros((front.H, front.W), np.uint8)
    for name in FAMILIES['ears']:
        if name in front.layers:
            ears |= (front.layers[name][..., 3] > 0.3).astype(np.uint8)
    if not ears.any():
        return False
    ys, _ = np.nonzero(mask)
    top = mask.copy()
    top[int(ys.min() + 0.15 * (ys.max() - ys.min() + 1)):] = False
    near = cv2.dilate(ears, np.ones((2 * EAR_HOOK_REACH + 1, 2 * EAR_HOOK_REACH + 1), np.uint8)) > 0
    return (top & near).sum() >= 10


def joined(a, b):
    """The earring flag of two pieces as one: a drawn hair ornament wins, then a drawn earring."""
    if a is False or b is False:
        return False
    return True if a or b else None


def accessory_host(front, mask, earring=None):
    """
    The ear for an earring that hangs from it; else the part
    around a piece the most (its rim's owners), which it rides; the face if
    only the background is around it. When the body is around it the
    most it is part of the outfit: on the neck (a choker's charm) it rides the
    neck, elsewhere (None) it stays put.
    """
    if earring is not False and hangs_from_ear(front, mask):
        return 'ears'
    owner = front.owner()
    ring = (cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~mask
    counts, body = {}, 0
    for name in owner[ring]:
        family = family_of(name) if name else None
        if family and family not in ACCESSORIES:
            counts[family] = counts.get(family, 0) + 1
        elif name and not family and name.split('-')[0] not in ACCESSORIES:
            body += 1
    if body > sum(counts.values()):
        neck = {f: counts.get(f, 0) for f in ('neckwear', 'neck')}
        if sum(neck.values()) < ACCESSORY_NECK_SHARE * ring.sum():
            return None
        # Drawn over the neckwear (a choker's charm the decomposition drew in both), it rides the neckwear.
        if 'neckwear' in front.layers and (front.layers['neckwear'][..., 3][mask] > 0.5).mean() >= 0.3:
            return 'neckwear'
        return max(neck, key=neck.get)
    return max(counts, key=counts.get) if counts else 'face'


# Under a clip the decomposition did not draw (the import recovers it from the
# illustration) the hair it sits on has a hole with the face's skin under it;
# when the clip moves on its own key the hole comes out as a patch of skin. The
# hair is filled in under it from the hair around (inpainting), over at most
# this much of the piece's rim's width.
UNDER_PIECE_RADIUS = 6


# A decomposition can order an accessory under what it sits on (a clip drawn
# under the bun it holds): the picture shows it on top. Myriad's import then
# erases the hair over it, and when the hair and the clip turn by their own
# keys the erased hole comes out. The part of the accessory the picture shows
# on top (its colour within BURIED_MATCH of the picture, the composite past
# MISMATCH_DISTANCE from it) is moved to a layer of its own over the others,
# whole (its holes filled) and coloured as the picture shows it.
BURIED_MATCH = 40


def raise_buried_accessories(front, picture, log):
    """Moves the parts of accessory layers the picture shows on top to a layer over the others."""
    owner = front.owner()
    composite = front.composite()
    for name in list(front.order):
        if name.split('-')[0] not in ACCESSORIES:
            continue
        layer = front.layers[name]
        own = np.linalg.norm(layer[..., :3] - picture, axis=-1) * 255 < BURIED_MATCH
        off = np.linalg.norm(composite - picture, axis=-1) * 255 > MISMATCH_DISTANCE
        buried = (layer[..., 3] > 0.5) & (owner != name) & own & off
        buried = cv2.morphologyEx(buried.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(buried, np.ones((9, 9), np.uint8)))
        keep = np.isin(lab, [k for k in range(1, n) if (buried * (lab == k)).sum() >= MIN_MISSING_AREA])
        if not keep.any():
            continue
        # The whole of the drawing there, its soft edge too, and what it encloses.
        part = keep & (layer[..., 3] > 0)
        _, gaps = cv2.connectedComponents((~part).astype(np.uint8))
        outside = np.unique(np.concatenate([gaps[0], gaps[-1], gaps[:, 0], gaps[:, -1]]))
        part |= ~np.isin(gaps, outside)
        raised = np.zeros_like(layer)
        raised[part, :3] = picture[part]
        raised[part, 3] = np.where(layer[part, 3] > 0, np.maximum(layer[part, 3], 0.0), 1.0)
        inner = cv2.erode(part.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        raised[inner, 3] = 1.0
        layer[part, 3] = 0
        # Numbered, as Myriad reads a further drawing of the same part.
        number = 2
        while f'{name.split("-")[0]}-{number}' in front.layers:
            number += 1
        new = f'{name.split("-")[0]}-{number}'
        front.layers[new] = raised
        front.order.append(new)
        log(f'raised {int(part.sum())} px of {name} over the layers that covered it')


# A decomposition redraws an ornament: a choker's star comes out smaller,
# with a smudge for a facet. Near an ornament (RESTORE_REACH px around it) the
# picture says what the ornament is at rest: where it differs from what the
# other layers show there (by RESTORE_DIFFERENT, or RESTORE_DIFFERENT_DRAWN
# where the decomposition drew the ornament), it is the ornament, taken as
# the picture has it; where it does not, the ornament has none.
RESTORE_LAYERS = ('headwear', 'earwear', 'neckwear')
RESTORE_REACH = 6
RESTORE_DIFFERENT = 40
RESTORE_DIFFERENT_DRAWN = 15
RESTORE_AREA = (0.5, 2.0)
RESTORE_HOLE = 0.2


def restore_ornaments(front, picture, log):
    """Redraws each ornament layer as the picture shows it at rest."""
    report = {}
    for name in list(front.order):
        if name.split('-')[0] not in RESTORE_LAYERS:
            continue
        layer = front.layers[name]
        drawn = layer[..., 3] > 0.3
        if drawn.sum() < 50:
            continue
        k = 2 * RESTORE_REACH + 1
        near = cv2.dilate(drawn.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        under = np.ones((front.H, front.W, 3), np.float32)
        for other in front.order:
            if other != name:
                a = front.layers[other][..., 3:4]
                under = under * (1 - a) + front.layers[other][..., :3] * a
        differs = np.linalg.norm(picture - under, axis=-1) * 255
        # Where the decomposition drew it, slight evidence will do (a pale gold on pale skin).
        art = near & ((differs > RESTORE_DIFFERENT) | (drawn & (differs > RESTORE_DIFFERENT_DRAWN)))
        art = cv2.morphologyEx(art.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        # What the ornament encloses is the ornament (a highlight as pale as the skin under it).
        n, gaps, stats, _ = cv2.connectedComponentsWithStats((~art).astype(np.uint8))
        border = set(np.unique(np.concatenate([gaps[0], gaps[-1], gaps[:, 0], gaps[:, -1]])))
        enclosed = [g for g in range(1, n) if g not in border and stats[g, 4] < RESTORE_HOLE * art.sum()]
        art |= np.isin(gaps, enclosed)
        ratio = art.sum() / drawn.sum()
        if not RESTORE_AREA[0] <= ratio <= RESTORE_AREA[1]:
            continue
        layer[art, :3] = picture[art]
        layer[art, 3] = 1.0
        layer[near & ~art, 3] = 0.0
        report[name] = round(float(ratio), 2)
    log(f'ornaments redrawn from the picture {report}')
    return report


def fill_under_piece(front, mask, family, log):
    """
    Fills the hair a piece riding `family` sits on where it has a hole under
    the piece. Under a clip on the back hair the decomposition paints the
    scalp's flat guess, not a hole: all of it is hidden at rest, so all of it
    is drawn again from the hair around.
    """
    names = [n for n in front.order if (family_of(n) or '').split(':')[0] == family]
    if not names or family not in ('front-hair', 'back-hair'):
        return 0
    # The layer around the piece the most.
    ring = (cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & ~mask
    name = max(names, key=lambda n: (front.layers[n][..., 3][ring] > 0.5).sum())
    layer = front.layers[name]
    hole = mask & (layer[..., 3] < 0.5) if family == 'front-hair' else mask.copy()
    if hole.sum() < 50 or (layer[..., 3][ring] > 0.5).mean() < 0.5:
        return 0
    rgb = (np.clip(layer[..., :3], 0, 1) * 255).astype(np.uint8)
    # Every transparent pixel near the piece is unknown, so only the hair itself is read.
    k = 4 * UNDER_PIECE_RADIUS + 1
    unknown = ((layer[..., 3] < 0.5) | hole) & (cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0)
    filled = cv2.inpaint(rgb, unknown.astype(np.uint8), UNDER_PIECE_RADIUS, cv2.INPAINT_TELEA)
    layer[hole, :3] = filled[hole].astype(np.float32) / 255
    layer[hole, 3] = 1.0
    log(f'filled {hole.sum()} px of {name} under a piece')
    return int(hole.sum())


def match_piece(mask, front_rgb, turned_rgb, side, face_cx, earring, center=None):
    """
    Where a piece went in the turned picture: the best match of its drawing,
    squeezed as the turn foreshortens it, near `center` (where its host's key
    carries it); at `center` itself when nothing near looks like it.
    """
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
    if center is not None:
        r = max(ACCESSORY_REACH, ACCESSORY_REACH_SHARE * max(x1 - x0, y1 - y0))
        reach = (r, r)
        cx, cy = center
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
    source = ((x0 + x1) / 2, (y0 + y1) / 2)
    if best is None or (center is not None and best[0] > ACCESSORY_MATCH_ERROR):
        return dict(source=source, target=(float(cx), float(cy)), scale=1.0, squeeze=1.0,
                    box=(int(x0), int(y0), int(x1), int(y1)), matched=False,
                    error=None if best is None else float(best[0]))
    error, s, q, tx, ty = best
    return dict(source=source, target=(tx, ty), scale=float(s), squeeze=float(q),
                box=(int(x0), int(y0), int(x1), int(y1)), matched=True, error=float(error))


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
    lock = lock_family(name)
    if lock:
        return lock
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


def headwear_error(front, keys, side, key, picture, box):
    """How far the picture's outlines are from the front drawn at `side`'s keys with `key` for the headwear, around `box`."""
    layers = [(name, family_of(name), front.layers[name]) for name in front.order]
    trial = {f: {side: v[side]} for f, v in keys.items() if side in v and f != 'headwear'}
    trial['headwear'] = {side: key}
    rgb, _ = render_keys(layers, trial, side, front.W, front.H)
    region = np.zeros((front.H, front.W), bool)
    x0, y0, x1, y1 = (int(v) for v in box)
    region[y0:y1 + 1, x0:x1 + 1] = True
    return edge_distance(rgb, picture, region)


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


def refine_on_picture(front, keys, side, picture, extra, log, own=()):
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
            if family in ACCESSORIES and family not in own:
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


# ---------------------------------------------------------------- locks

LOCK_GRID = 9


def sub_lattice(key, box, grid=LOCK_GRID):
    """The same backward field, on a lattice over a smaller box."""
    g = key['grid']
    x0, y0, x1, y1 = key['box']
    back = np.asarray(key['back'], np.float32).reshape(-1, 2)
    gx, gy = np.meshgrid(np.linspace(box[0], box[2], grid), np.linspace(box[1], box[3], grid))
    bx, by = sample(back, np.linspace(x0, x1, g), np.linspace(y0, y1, g), gx.ravel(), gy.ravel())
    return dict(box=[float(v) for v in box], grid=grid, back=[float(v) for p in zip(bx, by) for v in p])


def split_front_hair(front, keys, log):
    """
    The front hair as locks (hair_locks.py): one layer per lock in its place in
    the order, and one key per lock, the whole front hair's to begin with. The
    keys are then refined lock by lock on the pictures. Returns the lock count.
    """
    import hair_locks
    if 'front hair' not in front.layers or 'front-hair' not in keys:
        return 0
    L = front.layers['front hair']
    ys, xs = np.nonzero(L[..., 3] > 0.02)
    if not len(ys):
        return 0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    lab, info = hair_locks.split_locks(L[y0:y1, x0:x1, :3], L[y0:y1, x0:x1, 3])
    if info['locks'] < 2:
        return info['locks']
    full = np.zeros(L.shape[:2], np.int32)
    full[y0:y1, x0:x1] = lab
    # The soft rim joins the lock next to it.
    soft = L[..., 3] > 0
    for _ in range(24):
        if not (soft & (full == 0)).any():
            break
        grown = cv2.dilate(full.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        full = np.where((full == 0) & soft, grown, full)
    # Locks nearer the parting lie over the ones to the sides.
    cx = info['crown'][1] + x0
    ids = [k for k in range(1, full.max() + 1) if (full == k).any()]
    ids.sort(key=lambda k: -abs(np.nonzero(full == k)[1].mean() - cx))
    at = front.order.index('front hair')
    del front.layers['front hair']
    front.order.pop(at)
    whole = keys.pop('front-hair')
    for n, k in enumerate(ids, 1):
        name = f'front hair-{n}'
        layer = L.copy()
        layer[..., 3] *= full == k
        front.layers[name] = layer
        front.order.insert(at + n - 1, name)
        ly, lx = np.nonzero(layer[..., 3] > 0.02)
        box = (float(lx.min() - 30), float(ly.min() - 30), float(lx.max() + 30), float(ly.max() + 30))
        keys[lock_family(name)] = {side: sub_lattice(key, box) for side, key in whole.items()}
    log(f'front hair: {len(ids)} locks')
    return len(ids)


LOCK_FEATURES = ('eyewhite', 'irides', 'eyelash', 'eyebrow', 'nose', 'mouth')
LOCK_STEPS = (16.0, 8.0, 4.0, 2.0)
LOCK_COARSE = 3
LOCK_FEATURE_WEIGHT = 4.0
LOCK_SCALE = 0.5
LOCK_MIN_DET = 0.35


def folds(back, box, min_det=LOCK_MIN_DET):
    """Whether a backward lattice squeezes a cell flat or turns it over (the local area ratio below min_det)."""
    g = back.shape[0]
    hx, hy = (box[2] - box[0]) / (g - 1), (box[3] - box[1]) / (g - 1)
    dbx = np.diff(back, axis=1) / hx
    dby = np.diff(back, axis=0) / hy
    det = (1 + dbx[:-1, :, 0]) * (1 + dby[:, :-1, 1]) - dby[:, :-1, 0] * dbx[:-1, :, 1]
    return float(det.min()) < min_det


def _over(rgb, a, src):
    sa = src[..., 3:4]
    return rgb * (1 - sa) + src[..., :3] * sa, src[..., 3] + a * (1 - src[..., 3])


def fit_locks_on_turned(front, turned, keys, side, log):
    """
    Each lock of the front hair moved on its own until the locks together cover
    what the turned decomposition draws as front hair, and nothing it shows on
    top of the face (eyes, brows, nose, mouth). The whole front hair's key is
    one low-order warp of a drawing the turned picture redraws; a lock that
    inherits it can end up across an eye. Each lock's key takes a smooth
    correction (LOCK_COARSE x LOCK_COARSE over its box), searched coordinate
    by coordinate at half size, never folding. Returns the moved keys.
    """
    names = [n for n in front.order if lock_family(n) and lock_family(n) in keys and side in keys[lock_family(n)]]
    if len(names) < 2 or 'front hair' not in turned.layers:
        return {}
    T = turned.layers['front hair']
    owner = turned.owner()
    feature = np.isin(owner, [n for n in turned.order if n.split('-')[0] in LOCK_FEATURES])
    ys, xs = np.nonzero((T[..., 3] > 0.02) | np.any([front.layers[n][..., 3] > 0.02 for n in names], 0))
    cx0, cy0 = max(0, xs.min() - 80), max(0, ys.min() - 80)
    cx1, cy1 = min(front.W, xs.max() + 80), min(front.H, ys.max() + 80)
    small = lambda img: cv2.resize(img, None, fx=LOCK_SCALE, fy=LOCK_SCALE, interpolation=cv2.INTER_AREA)
    Ta = small(T[cy0:cy1, cx0:cx1, 3])
    Trgb = small(T[cy0:cy1, cx0:cx1, :3])
    F = small(feature[cy0:cy1, cx0:cx1].astype(np.float32))
    layers = {n: small(front.layers[n]) for n in names}
    h, w = Ta.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px = cx0 + (xx + 0.5) / LOCK_SCALE - 0.5
    py = cy0 + (yy + 0.5) / LOCK_SCALE - 0.5
    corr = {n: np.zeros((LOCK_COARSE, LOCK_COARSE, 2), np.float32) for n in names}

    def lattice(n):
        key = keys[lock_family(n)][side]
        g = key['grid']
        back = np.asarray(key['back'], np.float32).reshape(g, g, 2)
        return key, back + cv2.resize(corr[n], (g, g), interpolation=cv2.INTER_LINEAR)

    def render(n):
        key, back = lattice(n)
        x0, y0, x1, y1 = key['box']
        g = key['grid']
        gx = np.clip((px - x0) / max(1e-6, x1 - x0) * (g - 1), 0, g - 1).astype(np.float32)
        gy = np.clip((py - y0) / max(1e-6, y1 - y0) * (g - 1), 0, g - 1).astype(np.float32)
        o = cv2.remap(back, gx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return remap(layers[n], (px + o[..., 0] + 0.5) * LOCK_SCALE - 0.5, (py + o[..., 1] + 0.5) * LOCK_SCALE - 0.5)

    def cost(rgb, a):
        both = np.minimum(a, Ta)
        return float((np.abs(a - Ta) * 4 + np.abs(rgb - Trgb).mean(2) * both + a * F * LOCK_FEATURE_WEIGHT).mean())

    rendered = {n: render(n) for n in names}
    rgb, a = np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32)
    for n in names:
        rgb, a = _over(rgb, a, rendered[n])
    best = start = cost(rgb, a)
    for step in LOCK_STEPS:
        for _ in range(2):
            moved = 0
            for k, n in enumerate(names):
                # What lies under and over this lock stays put while it moves.
                below_rgb, below_a = np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32)
                for m in names[:k]:
                    below_rgb, below_a = _over(below_rgb, below_a, rendered[m])
                above_rgb, above_a = np.zeros((h, w, 3), np.float32), np.zeros((h, w), np.float32)
                for m in names[k + 1:]:
                    above_rgb, above_a = _over(above_rgb, above_a, rendered[m])
                for j in range(LOCK_COARSE):
                    for i in range(LOCK_COARSE):
                        for c in range(2):
                            for sign in (1, -1):
                                corr[n][j, i, c] += sign * step
                                key, back = lattice(n)
                                if not folds(back, key['box']):
                                    r = render(n)
                                    mid_rgb, mid_a = _over(below_rgb, below_a, r)
                                    e = cost(mid_rgb * (1 - above_a[..., None]) + above_rgb, above_a + mid_a * (1 - above_a))
                                    if e < best - 1e-7:
                                        best = e
                                        rendered[n] = r
                                        moved += 1
                                        break
                                corr[n][j, i, c] -= sign * step
            if not moved:
                break
    log(f'{side:5s} locks on the turned front hair: {start:.4f} -> {best:.4f}')
    return {lock_family(n): dict(lattice(n)[0], back=[float(v) for v in lattice(n)[1].reshape(-1)]) for n in names}


# A moved lock is kept where the picture agrees: its outline no further off
# than this without covering this much more of the face's features, or the
# features it covered at least halved.
LOCK_KEEP_SLACK = 0.15
LOCK_KEEP_COVER = 200


def keep_where_picture_agrees(front, turned, keys, moved, side, picture, log):
    """
    The turned decomposition says where the front hair is, not which lock is
    which: a lock can end up filling for another. Each moved lock is checked on
    the turned picture itself and kept only if its outline is no further off,
    or if it uncovers a face feature the old key had it across.
    """
    if not moved:
        return moved
    W, H = front.W, front.H
    layers = [(name, family_of(name), front.layers[name]) for name in front.order]
    old_rgb, old_top = render_keys(layers, keys, side, W, H)
    trial = {family: dict(sides, **({side: moved[family]} if family in moved else {})) for family, sides in keys.items()}
    new_rgb, new_top = render_keys(layers, trial, side, W, H)
    owner = turned.owner()
    feature = np.isin(owner, [n for n in turned.order if n.split('-')[0] in LOCK_FEATURES])
    kept = {}
    for i, (name, family, _) in enumerate(layers):
        if family not in moved:
            continue
        near = cv2.dilate(((old_top == i) | (new_top == i)).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        old_d = edge_distance(old_rgb, picture, near)
        new_d = edge_distance(new_rgb, picture, near)
        old_cover = int(((old_top == i) & feature).sum())
        new_cover = int(((new_top == i) & feature).sum())
        uncovers = old_cover > LOCK_KEEP_COVER and new_cover <= old_cover / 2
        covers = new_cover > old_cover + LOCK_KEEP_COVER
        if (new_d <= old_d + LOCK_KEEP_SLACK and not covers) or uncovers:
            kept[family] = moved[family]
        log(f'{side:5s} {family}: outline {old_d:.2f} -> {new_d:.2f}, over the face {old_cover} -> {new_cover} px, '
            f'{"kept" if family in kept else "left"}')
    return kept


def _fit_locks_direction(front_path, keys, side, turned_path, picture_path=None):
    lines = []
    front, turned = Decomposition(front_path), Decomposition(turned_path)
    moved = fit_locks_on_turned(front, turned, keys, side, lines.append)
    if picture_path:
        picture, _ = placed_illustration(picture_path, front.W, front.H)
        moved = keep_where_picture_agrees(front, turned, keys, moved, side, picture, lines.append)
    return moved, lines


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


# A decomposition paints a part's hidden side whole, and some parts lie over
# others they should not cover: the face's crown over back hair the front hair
# does not reach (a band of skin across the top of the head). The picture
# says what is on top: where such a part is on top, the picture is far from it
# (RGB distance over CUT_DIFFERENT) and close to what lies under it (under
# CUT_MATCH, and nearer by CUT_GAIN), the part is cut away.
PAINTED_OVER = ('face', 'neck', 'ears-r', 'ears-l', 'ears', 'topwear')
CUT_DIFFERENT = 35
CUT_MATCH = 30
CUT_GAIN = 20
CUT_MIN_AREA = 30
CUT_RIM = 4


# A decomposition redraws what it separates: a clip's star comes out soft and
# off-colour. What shows of a layer at rest is the picture itself, so its
# interior there (opaque, nothing over it, PICTURE_INSET px in from where
# another layer begins, which keeps the edges' blend) takes the picture's
# colours, where it is the same art redrawn (within PICTURE_MATCH). Other art
# (an earring the decomposition drew as hair) is left for the import to
# recover as its own piece, and so are the pieces found missing (`keep`).
PICTURE_INSET = 2
PICTURE_MATCH = 60


def paint_from_picture(front, picture, log, keep=None):
    """Gives each layer's interior that shows at rest the picture's colours where it is the same art."""
    over = np.zeros((front.H, front.W), np.float32)
    report = {}
    for name in reversed(front.order):
        layer = front.layers[name]
        same = np.linalg.norm(layer[..., :3] - picture, axis=-1) * 255 < PICTURE_MATCH
        shows = (layer[..., 3] > 0.95) & (over < 0.05) & same
        if keep is not None:
            shows &= ~keep
        shows = cv2.erode(shows.astype(np.uint8), np.ones((2 * PICTURE_INSET + 1, 2 * PICTURE_INSET + 1), np.uint8)) > 0
        if shows.any():
            layer[shows, :3] = picture[shows]
            report[name] = int(shows.sum())
        over = np.maximum(over, layer[..., 3])
    log(f'painted from the picture {sum(report.values())} px')
    return report


def cut_garment_under_face(front, log):
    """
    The garment a decomposition guesses under the chin (a collar's edge the
    face hides at rest) comes out when the chin rises, over the neck's faded
    top: no garment reaches up behind the face, so it goes.
    """
    face = front.layers.get('face')
    if face is None or 'topwear' not in front.layers:
        return 0
    li = front.order.index('face')
    if front.order.index('topwear') > li:
        return 0
    hidden = (face[..., 3] > 0.5) & (front.layers['topwear'][..., 3] > 0)
    front.layers['topwear'][..., 3] *= ~hidden
    log(f'cut the garment under the face: {int(hidden.sum())} px')
    return int(hidden.sum())


def cut_by_picture(front, picture, log, lift=False):
    """
    Cuts each of PAINTED_OVER where the picture shows what lies under it instead.
    With `lift`, an ornament under it the picture shows on top is not uncovered
    by a hole that would open as it moves: it goes over the part, which is
    filled from around under it.
    """
    report = {}
    for name in PAINTED_OVER:
        if name not in front.layers:
            continue
        on_top = front.owner() == name
        under = np.ones((front.H, front.W, 3), np.float32)
        below = np.full((front.H, front.W), -1, np.int32)
        for i, other in enumerate(front.order):
            if other != name:
                a = front.layers[other][..., 3:4]
                under = under * (1 - a) + front.layers[other][..., :3] * a
                below[a[..., 0] > 0.5] = i
        now = np.linalg.norm(front.composite() - picture, axis=-1) * 255
        then = np.linalg.norm(under - picture, axis=-1) * 255
        cut = on_top & (now > CUT_DIFFERENT) & (then < CUT_MATCH) & (now - then > CUT_GAIN)
        # Thin strokes too (a guessed edge under the chin): the picture shows what lies under them.
        cut = cv2.morphologyEx(cut.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cut)
        cut = np.isin(lab, [k for k in range(1, n) if stats[k, 4] >= CUT_MIN_AREA])
        if lift and cut.any():
            cut = lift_ornaments(front, name, cut, below, log)
        if cut.any():
            # The cut's rim, the two drawings blended, goes too where the picture is nearer what lies under.
            k = 2 * CUT_RIM + 1
            rim = (cv2.dilate(cut.astype(np.uint8), np.ones((k, k), np.uint8)) > 0) & ~cut & (front.layers[name][..., 3] > 0)
            cut |= rim & (then < now)
            soft = cv2.GaussianBlur(cv2.dilate(cut.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32), (5, 5), 0)
            front.layers[name][..., 3] *= 1 - np.clip(soft, 0, 1)
            report[name] = int(cut.sum())
    log(f'cut by the picture {report}')
    return report


# An ornament the decomposition put under a part (a choker under the chest's
# skin) shows through where the part is cut, and the hole opens as soon as the
# ornament moves on its own key. The picture shows it on top: it goes over the
# part, and the part is filled from around under it (LIFT_RADIUS). Where the
# part covers it more than the picture shows it on top (LIFT_SHOWN), the
# ornament does pass under the part (a pendant under a shirt): neither moves,
# and the part is not cut there.
LIFT_RADIUS = 6
LIFT_SHOWN = 0.5


def lift_ornaments(front, name, cut, below, log):
    """Takes out of `cut` what uncovers an ornament, lifting the ornament over `name` where the picture says so."""
    layer = front.layers[name]
    for i in sorted(set(np.unique(below[cut])) - {-1}):
        other = front.order[i]
        if other.split('-')[0] not in RESTORE_LAYERS or i > front.order.index(name):
            continue
        shown = cut & (below == i)
        cut &= ~shown
        if shown.sum() < CUT_MIN_AREA:
            continue
        overlap = (front.layers[other][..., 3] > 0.5) & (layer[..., 3] > 0.5)
        if shown.sum() < LIFT_SHOWN * overlap.sum():
            log(f'{other} stays under {name}: the picture shows {int(shown.sum())} of {int(overlap.sum())} px on top')
            continue
        hidden = cv2.dilate((overlap | shown).astype(np.uint8), np.ones((2 * CUT_RIM + 1, 2 * CUT_RIM + 1), np.uint8)) > 0
        # The part's own holes under it (where only the ornament was drawn) close too.
        k = 4 * LIFT_RADIUS + 1
        closed = cv2.morphologyEx(layer[..., 3], cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        layer[..., 3] = np.where(hidden, np.maximum(layer[..., 3], closed), layer[..., 3])
        inside = hidden & (layer[..., 3] > 0)
        rgb = (np.clip(layer[..., :3], 0, 1) * 255).astype(np.uint8)
        filled = cv2.inpaint(rgb, inside.astype(np.uint8), LIFT_RADIUS, cv2.INPAINT_TELEA)
        layer[inside, :3] = filled[inside].astype(np.float32) / 255
        cut &= ~hidden
        front.order.remove(other)
        front.order.insert(front.order.index(name) + 1, other)
        log(f'lifted {other} over {name} ({int(shown.sum())} px shown on top), filled {int(inside.sum())} px under it')
    return cut


# A decomposition draws the eyes whole and over the front hair (eyes drawn
# through the bangs, as much art has them). Where the picture shows the hair
# over an eye instead (a lock hanging over it), the lock goes over the eyes:
# it does when over HAIR_OVER_EYES of where it and the eyes overlap the picture
# is nearer the lock's colour than the eyes' (by HAIR_OVER_EYES_MARGIN).
HAIR_OVER_EYES = 0.5
HAIR_OVER_EYES_MARGIN = 10
EYE_PARTS = ('eyewhite', 'irides', 'eyelash', 'eyebrow')


def raise_hair_over_eyes(front, picture, log):
    """Moves the front hair the picture shows over the eyes above them."""
    eyes = [n for n in front.order if n.split('-')[0] in EYE_PARTS]
    locks = [n for n in front.order if n == 'front hair' or LOCK_LAYER.match(n)]
    if not eyes or not locks:
        return []
    eye_rgb = np.ones((front.H, front.W, 3), np.float32)
    eye_a = np.zeros((front.H, front.W), np.float32)
    for n in eyes:
        a = front.layers[n][..., 3:4]
        eye_rgb = eye_rgb * (1 - a) + front.layers[n][..., :3] * a
        eye_a = np.maximum(eye_a, a[..., 0])
    top = max(front.order.index(n) for n in eyes)
    raised = []
    for name in locks:
        if front.order.index(name) > min(front.order.index(n) for n in eyes):
            continue
        lock = front.layers[name]
        overlap = (lock[..., 3] > 0.5) & (eye_a > 0.5)
        if overlap.sum() < MIN_MISSING_AREA:
            continue
        to_lock = np.linalg.norm(lock[..., :3] - picture, axis=-1)[overlap] * 255
        to_eyes = np.linalg.norm(eye_rgb - picture, axis=-1)[overlap] * 255
        shown = float((to_lock + HAIR_OVER_EYES_MARGIN < to_eyes).mean())
        if shown >= HAIR_OVER_EYES:
            raised.append((name, shown, int(overlap.sum())))
    if raised:
        # Over the eyes, in their own order among themselves.
        names = [n for n, _, _ in raised]
        rest = [n for n in front.order if n not in names]
        top = max(rest.index(n) for n in eyes)
        front.order[:] = rest[:top + 1] + names + rest[top + 1:]
    if raised:
        log(f'front hair over the eyes: {[(n, round(s, 2), a) for n, s, a in raised]}')
    return raised


# The face's crown, painted whole, reaches up under the back hair over the
# skull. Front hair that does not reach the back of the head leaves it to come
# out as skin when the head turns (Myriad paints the face under hair as plain
# skin). Above the hairline the face is cut: the back hair is the scalp there.
# The hairline is the highest forehead that shows at rest (patches of skin of
# CROWN_MIN_SKIN px or more, up to CROWN_FOREHEAD eye spans above the eyes;
# 0.25-0.43 on every character so far), less CROWN_MARGIN eye spans for the
# forehead under the bangs. The forehead lies between the eyes' outer corners:
# above the eyes and outside them (CROWN_MARGIN eye spans out) are the
# temples, under the side hair, cut too. Where the back hair has a gap under
# them (the decomposition painted the scalp only behind the face's outline),
# hidden at rest, it is filled from the back hair around. A pixel is cut only
# where no pose would show a hole for it: at rest and at every key, it is
# still covered, or back hair lies where it goes. What shows at rest is never
# cut.
CROWN_FOREHEAD = 0.6
CROWN_MARGIN = 0.1
CROWN_MIN_SKIN = 300
CROWN_FILL_RADIUS = 8


def keyed_alpha(front, names, keys, side):
    """The coverage of the given layers drawn at their keys for `side` (rest when None)."""
    cover = np.zeros((front.H, front.W), np.float32)
    maps = {}
    for name in names:
        a = front.layers[name][..., 3]
        family = family_of(name)
        key = keys.get(family, {}).get(side) if side else None
        if key is not None:
            if family not in maps:
                maps[family] = lattice_maps(key, front.W, front.H)
            a = cv2.remap(a, *maps[family], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # The lattice clamps past its box; the layer itself is not there.
            x0, y0, x1, y1 = (int(round(v)) for v in key['box'])
            inside = np.zeros_like(a)
            inside[max(0, y0):max(0, y1 + 1), max(0, x0):max(0, x1 + 1)] = 1
            a = a * inside
        cover = np.maximum(cover, a)
    return cover


def cut_crown(front, keys, log):
    """Cuts the face above its hairline and at the temples where no pose shows a hole for it."""
    if 'face' not in front.layers:
        return 0
    face = front.layers['face']
    shows = cv2.morphologyEx(((front.owner() == 'face') & (face[..., 3] > 0.5)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    eyes = np.zeros((front.H, front.W), bool)
    for name in front.order:
        if name.split('-')[0] in ('eyewhite', 'irides', 'eyelash'):
            eyes |= front.layers[name][..., 3] > 0.5
    if not eyes.any():
        log('cut the crown: no eyes')
        return 0
    ey, ex = np.nonzero(eyes)
    span = float(ex.max() - ex.min())
    n, lab, stats, _ = cv2.connectedComponentsWithStats(shows)
    tops = [stats[k, 1] for k in range(1, n)
            if stats[k, 4] >= CROWN_MIN_SKIN and stats[k, 1] >= ey.min() - CROWN_FOREHEAD * span]
    if not tops:
        log('cut the crown: no forehead shows')
        return 0
    hairline = int(min(tops) - CROWN_MARGIN * span)
    margin = int(CROWN_MARGIN * span)
    zone = np.zeros((front.H, front.W), bool)
    zone[:max(0, hairline)] = True
    zone[:ey.min(), :max(0, ex.min() - margin)] = True
    zone[:ey.min(), ex.max() + margin:] = True
    cut = zone & (face[..., 3] > 0) & ~(cv2.dilate(shows, np.ones((5, 5), np.uint8)) > 0)
    li = front.order.index('face')
    above = front.order[li + 1:]
    back = [name for name in front.order[:li] if name in FAMILIES['back-hair']]
    if back:
        hair = front.layers[back[0]]
        covered = keyed_alpha(front, above, keys, None) > 0.9
        gap = cut & covered & (hair[..., 3] < 0.5)
        if gap.sum() >= 50:
            rgb = (np.clip(hair[..., :3], 0, 1) * 255).astype(np.uint8)
            unknown = (hair[..., 3] < 0.5).astype(np.uint8)
            filled = cv2.inpaint(rgb, unknown, CROWN_FILL_RADIUS, cv2.INPAINT_TELEA)
            hair[gap, :3] = filled[gap].astype(np.float32) / 255
            hair[gap, 3] = 1.0
            log(f'filled {int(gap.sum())} px of {back[0]} under the crown')
    face_keys = keys.get('face', {})
    for side in [None] + [s for s in SIDES if s in face_keys]:
        qy, qx = np.nonzero(cut)
        if not len(qy):
            break
        if side is None:
            tx, ty = qx, qy
        else:
            tx, ty = turned_position(face_keys[side], qx.astype(np.float32), qy.astype(np.float32))
        ix = np.clip(np.round(tx).astype(int), 0, front.W - 1); iy = np.clip(np.round(ty).astype(int), 0, front.H - 1)
        safe = (keyed_alpha(front, above, keys, side)[iy, ix] > 0.9) | (keyed_alpha(front, back, keys, side)[iy, ix] > 0.9)
        cut[qy[~safe], qx[~safe]] = False
    face[..., 3] *= 1 - cut
    log(f'cut the crown above y {hairline} and the temples: {int(cut.sum())} px')
    return int(cut.sum())


# The light on a part changes as the head turns: a neck the raised chin no
# longer shades is lighter, one the lowered chin shades darker. Each key of
# such a part holds a multiply colour (as Cubism keys one): the mean colour of
# the part where the turned picture shows it, over the mean of the part drawn
# at that key there, relative to the same at rest.
MULTIPLY_FAMILIES = ('neck',)
MULTIPLY_MIN_PIXELS = 400
MULTIPLY_RANGE = (0.6, 1.4)


def key_multiply(front, turned, keys, front_picture, pictures, log):
    """Sets each MULTIPLY_FAMILIES key's 'multiply' colour from the pictures."""
    report = {}
    front_owner = front.owner()
    for family in MULTIPLY_FAMILIES:
        names = [n for n in FAMILIES[family] if n in front.layers]
        if not names or family not in keys:
            continue
        layer = front.family(names)
        shows = cv2.erode(np.isin(front_owner, names).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        # Hidden at rest (a high collar to the chin), the part is as the decomposition drew it.
        at_rest = (front_picture[shows].mean(0) / np.maximum(layer[..., :3][shows].mean(0), 1e-3)
                   if shows.sum() >= MULTIPLY_MIN_PIXELS else np.ones(3))
        for side, t in turned.items():
            key = keys[family].get(side)
            if key is None or side not in pictures:
                continue
            drawn = remap(layer, *lattice_maps(key, front.W, front.H))
            seen = np.isin(t.owner(), names) & (drawn[..., 3] > 0.9)
            seen = cv2.erode(seen.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            if seen.sum() < MULTIPLY_MIN_PIXELS:
                continue
            gain = pictures[side][seen].mean(0) / np.maximum(drawn[..., :3][seen].mean(0), 1e-3) / at_rest
            key['multiply'] = [round(float(v), 3) for v in np.clip(gain, *MULTIPLY_RANGE)]
            report[f'{family} {side}'] = key['multiply']
    log(f'multiply {report}')
    return report


def bake(front, turned, keys, log, pictures=None):
    """
    Hidden pixels of the parts a turn uncovers, from the turned drawings where
    the same part is on top: their pictures' colours when given (side -> RGB)
    and the decomposition's part looks like them there, else the turned
    decompositions'.
    """
    owners = {side: t.owner() for side, t in turned.items()}
    report = {}
    for li, name in enumerate(front.order):
        family = BAKED.get(name) or lock_family(name)
        if not family or family not in keys:
            continue
        layer = front.layers[name]
        cover = np.zeros((front.H, front.W), np.float32)
        for above in front.order[li + 1:]:
            cover = np.maximum(cover, front.layers[above][..., 3])
        hidden = (layer[..., 3] > 0.25) & (cover > 0.9)
        qy, qx = np.nonzero(hidden)
        filled = np.zeros((front.H, front.W), bool)
        revealed = np.zeros(len(qx), bool)
        out = layer.copy()
        taken = {}
        for side, t in turned.items():
            key = keys[family].get(side)
            if key is None:
                continue
            tx, ty = turned_position(key, qx.astype(np.float32), qy.astype(np.float32))
            ix = np.clip(np.round(tx).astype(int), 0, front.W - 1); iy = np.clip(np.round(ty).astype(int), 0, front.H - 1)
            names = turned_names(family)
            mine = np.isin(owners[side][iy, ix], names)
            if family == 'back-hair':
                # Where nothing keyed covers it at this turn, the back hair is what shows of
                # the scalp there: whatever hair the turn has on top.
                names = names + ['front hair']
                exposed = keyed_alpha(front, front.order[li + 1:], keys, side)[iy, ix] < 0.5
                revealed |= exposed
                mine |= exposed & np.isin(owners[side][iy, ix], names)
            mine &= ~filled[qy, qx]
            src = t.family(names)
            if pictures and side in pictures:
                # The picture where the decomposition's part is what it shows (not other art it took in).
                agrees = np.linalg.norm(src[..., :3] - pictures[side], axis=-1) * 255 < BURIED_MATCH
                src = np.concatenate([np.where(agrees[..., None], pictures[side], src[..., :3]), src[..., 3:4]], 2)
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
        if family == 'back-hair':
            # What a turn shows of the scalp that no turned drawing has as hair (a
            # clip or a lock on top there) keeps the decomposition's flat guess, a
            # patch duller than the hair: it is drawn from the hair around instead.
            guess = np.zeros((front.H, front.W), bool)
            guess[qy[revealed], qx[revealed]] = True
            guess &= ~filled
            if guess.sum() >= MIN_MISSING_AREA:
                rgb = (np.clip(layer[..., :3], 0, 1) * 255).astype(np.uint8)
                # Only hair that shows at rest or was taken from a turn is read.
                unknown = ((hidden & ~filled) | (layer[..., 3] < 0.25)).astype(np.uint8)
                drawn = cv2.inpaint(rgb, unknown, UNDER_PIECE_RADIUS, cv2.INPAINT_TELEA)
                layer[guess, :3] = drawn[guess].astype(np.float32) / 255
                taken['drawn'] = int(guess.sum())
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

# A turn can hide an ear behind the hair: the turned decomposition then has one
# ear where the front has two, and fitting both onto the one pulls the hidden
# ear across the head (or off it). The hidden ear is put where the head carries
# it (ear_carrier), and fitted with the one that shows. An ear shows when this share
# of it, so carried, lies near the turned ears.
EAR_SHOWN_SHARE = 0.3
EAR_NEAR = 31
# Turned ears this far from the front's in area (the hidden one put in) are
# not ears (a decomposition that took half the head for them): they ride the
# face's key, or the back hair's (ear_carrier), instead.
EAR_AREA_RATIO = (0.4, 2.5)
# Ears fitted no better than this (IoU) are not the same ears: they ride the head.
EAR_FIT_IOU = 0.8


def ear_carrier(fits, box):
    """
    The key an ear rides when its own cannot be fitted: the face's when the
    ear lies wholly within the face's box, else the back hair's (cat ears on the
    scalp, past where the face's lattice reaches), else none.
    """
    face = fits.get('face')
    if face is not None:
        x0, y0, x1, y1 = face['box']
        if x0 <= box[0] and box[2] <= x1 and y0 <= box[1] and box[3] <= y1:
            return face, 'face'
    if fits.get('back-hair') is not None:
        return fits['back-hair'], 'back hair'
    return (face, 'face') if face is not None else (None, None)


def hidden_ears(front, turned, fits, log):
    """The front's ears the turned decomposition does not show, carried by the head's keys (RGBA), or None."""
    shown = np.zeros((turned.H, turned.W), np.uint8)
    for n in FAMILIES['ears']:
        if n in turned.layers:
            shown |= (turned.layers[n][..., 3] > 0.3).astype(np.uint8)
    near = cv2.dilate(shown, np.ones((EAR_NEAR, EAR_NEAR), np.uint8)) > 0
    ears = front.family(FAMILIES['ears'])
    n, lab = cv2.connectedComponents((ears[..., 3] > 0.3).astype(np.uint8))
    # Only when the turn shows fewer ears than the front.
    pieces = lambda labels, count: sum((labels == k).sum() >= 150 for k in range(1, count))
    shown_n, shown_lab = cv2.connectedComponents(shown)
    if pieces(shown_lab, shown_n) >= pieces(lab, n):
        return None
    out = None
    for k in range(1, n):
        if (lab == k).sum() < 150:
            continue
        ys, xs = np.nonzero(lab == k)
        key, by = ear_carrier(fits, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        if key is None:
            continue
        piece = ears.copy()
        piece[..., 3] *= (lab == k)
        carried = remap(piece, *lattice_maps(key, front.W, front.H))
        m = carried[..., 3] > 0.3
        if not m.any() or (m & near).sum() >= EAR_SHOWN_SHARE * m.sum():
            continue
        log(f'ears       one hidden by the turn ({m.sum()} px), carried by the {by}')
        if out is None:
            out = carried
        else:
            sa = carried[..., 3:4]
            out = np.concatenate([out[..., :3] * (1 - sa) + carried[..., :3] * sa, sa + out[..., 3:4] * (1 - sa)], 2)
    return out


def _fit_direction(front_path, turned_path):
    front = Decomposition(front_path)
    turned = Decomposition(turned_path)
    lines = []
    fits = {}
    # The ears last: when they cannot be fitted they ride the face or the back hair.
    order = [f for f in FAMILIES if f != 'ears'] + ['ears']
    for family in order:
        names = FAMILIES[family]
        if not any(n in front.layers for n in names) or not any(n in turned.layers for n in names):
            continue
        if family == 'ears':
            hidden = hidden_ears(front, turned, fits, lines.append)
            if hidden is not None:
                turned.layers['ears-hidden'] = hidden
                turned.order.insert(0, 'ears-hidden')
                names = names + ['ears-hidden']
            front_ears = front.family(names)[..., 3] > 0.5
            ratio = (turned.family(names)[..., 3] > 0.5).sum() / max(1, front_ears.sum())
            if not EAR_AREA_RATIO[0] <= ratio <= EAR_AREA_RATIO[1]:
                ys, xs = np.nonzero(front_ears)
                key, by = ear_carrier(fits, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)) if len(xs) else (None, None)
                if key is not None:
                    lines.append(f'ears        turned ears are {ratio:.1f}x the front ones: they ride the {by}')
                    fits['ears'] = dict(key, fit=None)
                continue
        fit = fit_family(front, turned, names, family, lines.append)
        if fit and family == 'ears' and fit['fit']['iou'] < EAR_FIT_IOU:
            # The turned ears are other parts (a human ear where a cat ear was dropped).
            ys, xs = np.nonzero(front.family(names)[..., 3] > 0.5)
            key, by = ear_carrier(fits, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            if key is not None:
                lines.append(f'ears        fitted at IoU {fit["fit"]["iou"]:.2f}: they ride the {by}')
                fit = dict(key, fit=None)
        if fit:
            fits[family] = fit
    front_area, turned_area = big_headwear(front), big_headwear(turned)
    if front_area and turned_area and BIG_HEADWEAR_RATIO[0] <= turned_area / front_area <= BIG_HEADWEAR_RATIO[1]:
        fit = fit_family(front, turned, ['headwear'], 'headwear', lines.append)
        if fit and fit['fit']['iou'] >= BIG_HEADWEAR_IOU:
            fits['headwear'] = fit
    return fits, lines


def _refine_direction(front_path, keys, side, front_picture, picture_path, recovered, own=()):
    front = Decomposition(front_path)
    rgb, _ = placed_illustration(front_picture, front.W, front.H)
    extra = [('recovered', 'earwear', np.concatenate([rgb, m[..., None].astype(np.float32)], 2)) for m in recovered]
    picture, _ = placed_illustration(picture_path, front.W, front.H)
    keys = {family: dict(sides) for family, sides in keys.items()}
    lines = []
    change = refine_on_picture(front, keys, side, picture, extra, lines.append, own)
    return {family: sides[side] for family, sides in keys.items() if side in sides}, change, lines


# ---------------------------------------------------------------- checking a decomposition

# A decomposition can take the long hair at the sides of the face into the
# face layer (a pale character whose hair is near her skin): the face then
# runs as wide as the hair below the eyes. A face is about as wide as the eyes
# reach (0.9-1.25 on every character so far, turned or not; the hidden crown
# above the eyes is the face's too and may be wider).
FACE_SPREAD_LIMIT = 1.5
# Or the headwear takes in the outfit and the hair hanging over it: headwear
# below the chin, over the face's area (0.02 at most so far; 1.35 when it did).
HEADWEAR_BELOW_CHIN_LIMIT = 0.5
# Or the ears take in the head: the ears over the face's area (cat ears 0.18;
# 0.94 when they did).
EARS_LIMIT = 0.5


def face_spread(dec):
    """The face layer's widest row below the eyes over the eyes' span; None without a face or eyes."""
    face = dec.layers.get('face')
    eyes = np.zeros((dec.H, dec.W), bool)
    for name in dec.order:
        if name.split('-')[0] in ('eyewhite', 'irides', 'eyelash'):
            eyes |= dec.layers[name][..., 3] > 0.5
    if face is None or not eyes.any():
        return None
    face = face[..., 3] > 0.5
    ey, ex = np.nonzero(eyes)
    span = ex.max() - ex.min()
    widths = [np.ptp(xs) for xs in (np.nonzero(row)[0] for row in face[int(ey.mean()):]) if len(xs)]
    if not widths or span < 8:
        return None
    return float(max(widths) / span)


def part_shares(dec):
    """Headwear below the chin and the ears, each over the face's area; None without a face."""
    face = dec.layers.get('face')
    if face is None or not (face[..., 3] > 0.5).any():
        return None
    face = face[..., 3] > 0.5
    chin = np.nonzero(face)[0].max()
    shares = {'headwear': 0.0, 'ears': 0.0}
    for name in dec.order:
        a = dec.layers[name][..., 3] > 0.5
        if name.split('-')[0] == 'headwear':
            shares['headwear'] += a[chin:].sum() / face.sum()
        elif name in FAMILIES['ears']:
            shares['ears'] += a.sum() / face.sum()
    return shares


def check_decomposition(path):
    """
    Whether a decomposition can be keyed from: its faults, and how far past
    its limits it is at worst (badness, over 1 when faulty), for choosing the
    least faulty of several.
    """
    dec = Decomposition(path)
    spread = face_spread(dec)
    shares = part_shares(dec)
    if spread is None or shares is None:
        return dict(ok=False, faults=['no face or eyes'], badness=None, face_spread=None)
    measures = {
        'face takes in the hair': spread / FACE_SPREAD_LIMIT,
        'headwear takes in the outfit': shares['headwear'] / HEADWEAR_BELOW_CHIN_LIMIT,
        'ears take in the head': shares['ears'] / EARS_LIMIT,
    }
    faults = [fault for fault, over in measures.items() if over > 1]
    return dict(ok=not faults, faults=faults, badness=round(float(max(measures.values())), 3),
                face_spread=round(spread, 3))


def host_report():
    """What the machine offers the fit: the host's CPUs, this process's, the container's quota."""
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    try:
        quota = open('/sys/fs/cgroup/cpu.max').read().strip()
    except OSError:
        quota = None
    return dict(cpu_count=os.cpu_count(), affinity=affinity, cpu_max=quota)


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
    started = time.time()
    seconds = {}
    spreads = {side: face_spread(d) for side, d in [('front', front), *turned.items()]}
    log(f'face spread {({side: None if v is None else round(v, 2) for side, v in spreads.items()})}')
    keys = {}
    # The four directions fit apart, one process each where the machine has them.
    cpus = available_cpus()
    workers = max(1, min(len(turned_paths), cpus))
    cv2.setNumThreads(cpus)
    # Each process takes its share of the CPUs, its OpenCV threads too.
    pool_args = dict(max_workers=workers, initializer=_limit_threads, initargs=(max(1, cpus // workers),))
    log(f'{cpus} CPUs, {workers} processes')
    with ProcessPoolExecutor(**pool_args) as pool:
        jobs = {side: pool.submit(_fit_direction, front_path, path) for side, path in turned_paths.items()}
        for side, job in jobs.items():
            fits, lines = job.result()
            for line in lines:
                log(f'{side:5s} {line}')
            for family, fit in fits.items():
                keys.setdefault(family, {})[side] = fit
    seconds['directions'] = round(time.time() - started)
    # Big headwear is a part of its own: where its decomposition failed it, it starts from the head's move.
    if big_headwear(front) and 'face' in keys:
        ys, xs = np.nonzero(front.layers['headwear'][..., 3] > 0.3)
        box = (max(0, xs.min() - HEAD_KEY_PAD), max(0, ys.min() - HEAD_KEY_PAD),
               min(front.W - 1, xs.max() + HEAD_KEY_PAD), min(front.H - 1, ys.max() + HEAD_KEY_PAD))
        layer = front.layers['headwear']
        face_cx = float(np.nonzero(front.layers['face'][..., 3] > 0.5)[1].mean())
        whole = layer[..., 3] > 0.5
        for side in turned_paths:
            if side in keys.get('headwear', {}) or side not in keys['face']:
                continue
            key = head_key(keys['face'][side], box, front.layers['face'][..., 3])
            chosen, note = key, 'rides the head'
            if pictures and side in pictures:
                # The picture says where it went: the head's move as it is, shifted to the
                # best match near it, or the whole moved rigidly; the nearest drawn wins.
                picture = placed_illustration(pictures[side], front.W, front.H)[0]
                candidates = [(key, 'rides the head')]
                moved = remap(layer, *lattice_maps(key, front.W, front.H))
                mask = moved[..., 3] > 0.5
                if mask.sum() >= MIN_MISSING_AREA:
                    rgb = moved[..., :3] * moved[..., 3:4] + (1 - moved[..., 3:4])
                    ys, xs = np.nonzero(mask)
                    center = ((xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2)
                    found = match_piece(mask, rgb, picture, side, face_cx, False, center)
                    if found['matched']:
                        dx, dy = found['target'][0] - found['source'][0], found['target'][1] - found['source'][1]
                        back = np.asarray(key['back'], np.float32).reshape(-1, 2) - [dx, dy]
                        x0, y0, x1, y1 = key['box']
                        candidates.append((dict(key, box=[x0 + dx, y0 + dy, x1 + dx, y1 + dy],
                                                back=[float(v) for v in back.reshape(-1)]),
                                           f'rides the head, shifted ({dx:.0f}, {dy:.0f})'))
                rigid = match_piece(whole, placed_illustration(pictures['front'], front.W, front.H)[0], picture, side, face_cx, False)
                candidates.append((accessory_lattice([rigid]), 'moves whole'))
                errors = [headwear_error(front, keys, side, k, picture, box) for k, _ in candidates]
                best = int(np.argmin(errors))
                chosen = candidates[best][0]
                note = f'{candidates[best][1]} ({" / ".join(f"{e:.2f}" for e in errors)})'
            keys.setdefault('headwear', {})[side] = chosen
            log(f'{side:5s} headwear    {note}: its decomposition does not hold it')
    own = tuple(f for f in ACCESSORIES if len(keys.get(f, {})) == len(turned_paths))
    if own:
        log(f'fitted as parts of their own: {own}')
    # Accessories ride rigidly: every piece of every hair clip and earring,
    # matched on the illustrations when given (they hold what the
    # decompositions re-render or drop).
    face = front.layers.get('face')
    face_cx = float(np.nonzero(face[..., 3] > 0.5)[1].mean()) if face is not None else front.W / 2
    drawn = {}
    recovered = []
    if pictures:
        drawn = {side: placed_illustration(path, front.W, front.H) for side, path in pictures.items()}
        raise_buried_accessories(front, drawn['front'][0], log)
        restore_ornaments(front, drawn['front'][0], log)
    pieces = [(mask, name.startswith('earwear')) for mask, name in accessory_pieces(front) if name.split('-')[0] not in own]
    clips = []
    if pictures:
        recovered = missing_pieces(front, *drawn['front'])
        for mask in recovered:
            # Whether it is an earring the ear says: it hangs from one or not (accessory_host).
            pieces.append((mask, None))
            ys, xs = np.nonzero(mask)
            log(f'recovered piece at ({xs.mean():.0f}, {ys.mean():.0f}), {mask.sum()} px')
    front_rgb = drawn['front'][0] if drawn else front.composite()
    if pieces:
        pieces, hosts = group_pieces(front, pieces)
        log(f'accessory hosts {hosts}')
        # A clip on the hair (not an earring hanging in front of the back hair: an
        # openwork one shows what is behind it, which a fill would change) has the
        # hair under it filled, once the front hair is cut into locks.
        clips = [(mask, host) for (mask, earring), host in zip(pieces, hosts)
                 if host == 'front-hair' or (host == 'back-hair' and earring is False)]
    if pieces:
        for side, t in turned.items():
            turned_rgb = drawn[side][0] if side in drawn else t.composite()
            found = []
            for (mask, earring), host in zip(pieces, hosts):
                ys, xs = np.nonzero(mask)
                cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
                if host is None:
                    # Part of the outfit: it does not turn with the head, and holds the lattice still around it.
                    found.append(dict(source=(cx, cy), target=(cx, cy), scale=1.0, squeeze=1.0,
                                      box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), matched=False))
                    continue
                key = keys.get(host, {}).get(side)
                if key is not None:
                    tx, ty = turned_position(key, np.array([cx], np.float32), np.array([cy], np.float32))
                    center = (float(tx[0]), float(ty[0]))
                else:
                    center = (cx, cy)
                found.append(match_piece(mask, front_rgb, turned_rgb, side, face_cx, host == 'ears', center))
            lattice = accessory_lattice(found)
            for family in ACCESSORIES:
                if family not in own:
                    keys.setdefault(family, {})[side] = lattice
            log(f'{side:5s} accessories {[(round(f["target"][0]), round(f["target"][1]), round(f["squeeze"], 2), "m" if f["matched"] else "h") for f in found]}')
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
    locks = split_front_hair(front, complete, log)
    if locks >= 2:
        with tempfile.TemporaryDirectory() as tmp:
            split_path = os.path.join(tmp, 'split.psd')
            save_psd(front, split_path)
            with ProcessPoolExecutor(**pool_args) as pool:
                jobs = {side: pool.submit(_fit_locks_direction, split_path, complete, side, path,
                                          pictures.get(side) if pictures else None)
                        for side, path in turned_paths.items()}
                for side, job in jobs.items():
                    moved, lines = job.result()
                    for line in lines:
                        log(line)
                    for family, key in moved.items():
                        complete[family][side] = key
    seconds['locks'] = round(time.time() - started)
    for mask, host in clips:
        fill_under_piece(front, mask, host, log)
    cut_garment_under_face(front, log)
    if drawn:
        cut_by_picture(front, drawn['front'][0], log, lift=True)
        raise_hair_over_eyes(front, drawn['front'][0], log)
        # Not the pieces the import recovers on their own.
        keep = np.zeros((front.H, front.W), np.uint8)
        for mask in recovered:
            keep |= mask.astype(np.uint8)
        paint_from_picture(front, drawn['front'][0], log, cv2.dilate(keep, np.ones((9, 9), np.uint8)) > 0)
    turned_pictures = {side: drawn[side][0] for side in turned if side in drawn} if drawn else None
    if turned_pictures:
        # The turned decompositions paint over alike; their pictures say what is on top there.
        for side, picture in turned_pictures.items():
            cut_by_picture(turned[side], picture, lambda line, side=side: log(f'{side:5s} {line}'))
    cut_crown(front, complete, log)
    baked = bake(front, turned, complete, log, turned_pictures)
    seconds['baked'] = round(time.time() - started)
    # How the time went, to size the work to the machine it runs on.
    result = dict(canvas=[front.W, front.H], keyforms=complete, fit=report, baked=baked, locks=locks,
                  host=dict(host_report(), seconds=seconds))
    if drawn:
        # Drawn together with what they uncover, as the runtime draws them, the
        # keys move to where the pictures have the parts; then what they
        # uncover is taken again for the moved keys.
        with tempfile.TemporaryDirectory() as tmp:
            baked_path = os.path.join(tmp, 'baked.psd')
            save_psd(front, baked_path)
            with ProcessPoolExecutor(**pool_args) as pool:
                jobs = {side: pool.submit(_refine_direction, baked_path, complete, side, pictures['front'], pictures[side], recovered, own)
                        for side in turned if side in pictures}
                result['picture'] = {}
                for side, job in jobs.items():
                    moved, change, lines = job.result()
                    for line in lines:
                        log(line)
                    for family, key in moved.items():
                        complete[family][side] = key
                    result['picture'][side] = change
        result['baked'] = bake(front, turned, complete, log, turned_pictures)
        key_multiply(front, turned, complete, drawn['front'][0], turned_pictures, log)
    seconds['total'] = round(time.time() - started)
    return front, result


if __name__ == '__main__':
    front_path, right, left, up, down, out_json, out_psd = sys.argv[1:8]
    pictures = dict(zip(('front', 'plus', 'minus', 'up', 'down'), sys.argv[8:13])) if len(sys.argv) >= 13 else None
    front, result = turn_keyforms(front_path, dict(plus=right, minus=left, up=up, down=down), pictures=pictures)
    save_psd(front, out_psd)
    json.dump(result, open(out_json, 'w'))
