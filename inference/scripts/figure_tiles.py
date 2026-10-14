"""
A standing figure decomposed in tiles.

A decomposition's canvas holds a figure at one fixed precision: a whole
figure on it is a couple of hundred pixels a head. Cut into three tiles, each
on a canvas of its own (the head and chest framed as a bust, the waist, the
legs), every part is found at a bust's precision. The figure is decomposed
whole as well: that decomposition says which layers there are and in what
order, and each tile gives a layer's pixels where it agrees with it.

A tile can name a part differently from the whole (a legs tile puts the shoes
in with the legs): where the whole has a layer the tile left empty, and one
of the tile's layers covers it and looks like it, that layer is split by the
whole's layers. A tile's layer keeps only what lies on the whole's layers it
was matched to (a floor shadow the legs tile took in is dropped).

    plan(whole, image_hw) -> {'upper': box, 'middle': box, 'lower': box} or None
    crops(image, tiles)   -> the pictures to decompose, and their sizes
    stitch(...)           -> the figure's PSD at the picture's own size
"""
import numpy as np
import cv2
from PIL import Image
from psd_tools import PSDImage
from psd_tools.constants import Compression

from figure_head import BUST_PIXELS, crop_image, head_crop, placement

# The decomposition's canvas, whose shape a tile is given where it can be.
CANVAS_ASPECT = 1088 / 1664
# Tiles overlap by this share of the figure's height, and have this much room around.
OVERLAP = 0.06
ROOM = 0.02
# A tile fades out over this many output pixels at its inner edges.
FEATHER = 32
# A tile's layer is the whole's layer of its name when they cover each other this much, or
# when it covers this much of what of the whole's shows: a neck the bust tile drew on down
# under the collar and not up behind the chin (IoU 0.46 with the whole's). What lies past the
# whole's is cut (REACH).
AGREE_IOU = 0.5
AGREE_COVER = 0.95
# A layer the tile left empty is in another of its layers when that layer covers
# this share of it, in colours this close (mean RGB distance, 0..1).
TAKEN_IN = 0.5
TAKEN_IN_COLOUR = 0.12
# A tile's layer keeps what lies within this many output pixels of the whole's layers it was matched to.
REACH = 8
# Matching is measured at this fraction of the output size.
MATCH_SCALE = 0.5
MIN_PIXELS = 12


def _union_alpha(layers, W, H):
    """The whole decomposition's drawn pixels (canvas), from Decomposition-like layers."""
    out = np.zeros((H, W), bool)
    for layer in layers.values():
        out |= layer[..., 3] > 0.3
    return out


def plan(whole, image_hw):
    """
    The three tiles (x0, y0, x1, y1 in picture pixels): the head and chest as
    a bust (figure_head.head_crop), then the rest in two, overlapping. None
    when the head is big enough already (not a standing figure's).
    """
    upper = head_crop(whole, image_hw)
    if upper is None:
        return None
    ih, iw = image_hw
    scale, ox, oy = placement(image_hw, (whole.H, whole.W))
    drawn = _union_alpha(whole.layers, whole.W, whole.H)
    ys, xs = np.nonzero(drawn)
    top, bottom = (ys.min() - oy) / scale, (ys.max() + 1 - oy) / scale
    height = bottom - top
    overlap, room = OVERLAP * height, ROOM * height

    def tile(y0, y1):
        rows = drawn[max(0, int(y0 * scale + oy)):max(1, int(y1 * scale + oy))]
        cols = np.nonzero(rows.any(0))[0]
        x0, x1 = ((cols.min() - ox) / scale - room, (cols.max() + 1 - ox) / scale + room) if len(cols) else (0, iw)
        width = max(x1 - x0, (y1 - y0) * CANVAS_ASPECT)
        cx = (x0 + x1) / 2
        return [int(round(cx - width / 2)), int(round(y0)), int(round(cx + width / 2)), int(round(y1))]

    y0 = upper[3] - overlap
    y1 = bottom + room
    middle = (y0 + y1) / 2
    return {'upper': upper, 'middle': tile(y0, middle + overlap / 2), 'lower': tile(middle - overlap / 2, y1)}


def crops(image, tiles):
    """
    Each tile's picture to decompose, and the size (h, w) it is: the upper one
    at a bust's size (it is also the head keyed as a bust), the others as cut,
    white past the picture's edges.
    """
    out = {}
    for name, box in tiles.items():
        if name == 'upper':
            out[name] = crop_image(image, box)
            continue
        x0, y0, x1, y1 = box
        picture = Image.new('RGB', (x1 - x0, y1 - y0), (255, 255, 255))
        part = image.convert('RGB').crop((max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)))
        picture.paste(part, (max(0, -x0), max(0, -y0)))
        out[name] = picture
    return out


# ---------------------------------------------------------------- placing

def load(path):
    """[(name, top, left, rgba float crop)] in order, and the canvas (w, h)."""
    psd = PSDImage.open(path)
    layers = []
    for layer in psd.descendants():
        if layer.is_group():
            continue
        rgba = np.asarray(layer.topil().convert('RGBA')).astype(np.float32) / 255
        layers.append((layer.name, layer.top, layer.left, rgba))
    return layers, psd.size


class Source:
    """A decomposition and where its canvas lies on the output: p_out = p * k + o."""

    def __init__(self, path, box, input_hw, name):
        self.name = name
        self.layers, (self.W, self.H) = load(path)
        x0, y0, x1, y1 = box
        s, tx, ty = placement(input_hw, (self.H, self.W))
        to_image = (x1 - x0) / input_hw[1]
        self.k = to_image / s
        self.o = (x0 - tx * to_image / s, y0 - ty * to_image / s)
        self.box = box
        self.by_name = {name: (top, left, rgba) for name, top, left, rgba in self.layers}
        self._placed = {}

    def placed(self, name, scale=1.0):
        """(top, left, premultiplied rgba) of a layer on the output (times `scale`), or None."""
        if (name, scale) not in self._placed:
            self._placed[name, scale] = self._place(name, scale)
        return self._placed[name, scale]

    def _place(self, name, scale):
        if name not in self.by_name:
            return None
        top, left, rgba = self.by_name[name]
        if not (rgba[..., 3] > 0.02).any():
            return None
        pre = rgba.copy()
        pre[..., :3] *= pre[..., 3:4]
        k = self.k * scale
        h, w = rgba.shape[:2]
        size = (max(1, int(round(w * k))), max(1, int(round(h * k))))
        big = cv2.resize(pre, size, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        return (int(round((top * self.k + self.o[1]) * scale)), int(round((left * self.k + self.o[0]) * scale)),
                np.clip(big, 0, 1))


def paste(box, piece, channels=4):
    """A placed piece (top, left, array) into zeros over box (y0, x0, y1, x1)."""
    y0, x0, y1, x1 = box
    out = np.zeros((y1 - y0, x1 - x0, channels), np.float32)
    if piece is None:
        return out
    top, left, a = piece
    ya, xa = max(top, y0), max(left, x0)
    yb, xb = min(top + a.shape[0], y1), min(left + a.shape[1], x1)
    if yb > ya and xb > xa:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = a[ya - top:yb - top, xa - left:xb - left, :channels]
    return out


def tile_weight(tile, box, out_hw):
    """Over box (y0, x0, y1, x1): 1 inside the tile, fading over FEATHER at its inner edges (none at the picture's)."""
    y0, x0, y1, x1 = box
    H, W = out_hw
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    bx0, by0, bx1, by1 = tile.box
    far = np.float32(1e6)
    d = np.minimum.reduce([xx - bx0 if bx0 > 0 else far + 0 * xx, bx1 - xx if bx1 < W else far + 0 * xx,
                           yy - by0 if by0 > 0 else far + 0 * yy, by1 - yy if by1 < H else far + 0 * yy])
    inside = (xx >= bx0) & (xx < bx1) & (yy >= by0) & (yy < by1)
    return np.where(inside, np.clip(d / FEATHER, 0, 1), 0)[..., None]


# ---------------------------------------------------------------- matching

def match_tile(whole, tile, order, out_hw, log):
    """
    For one tile, which of its layers draws which of the whole's: {whole name:
    [(tile name, owned)]}, owned None for the tile's layer whole, else the
    whole's names whose share of a split tile layer goes to it.
    """
    s = MATCH_SCALE
    H, W = int(out_hw[0] * s), int(out_hw[1] * s)
    bx0, by0, bx1, by1 = (int(v * s) for v in tile.box)
    box = (max(0, by0), max(0, bx0), min(H, by1), min(W, bx1))
    core = tile_weight_small(tile, box, s, out_hw) > 0.99
    alpha = lambda piece: paste(box, piece)[..., 3]
    rgb = lambda piece: paste(box, piece)[..., :3]
    whole_a = {name: alpha(whole.placed(name, s)) for name in order}
    tile_a = {name: alpha(tile.placed(name, s)) for name, *_ in tile.layers}
    whole_rgb = {name: rgb(whole.placed(name, s)) for name in order}
    tile_rgb = {name: rgb(tile.placed(name, s)) for name, *_ in tile.layers}
    # Which of the whole's layers shows on top, per pixel.
    shows = np.full((box[2] - box[0], box[3] - box[1]), -1)
    for index, name in enumerate(order):
        shows = np.where(whole_a[name] > 0.5, index, shows)
    out = {}
    own = {}
    for index, name in enumerate(order):
        a, b = tile_a.get(name), whole_a[name]
        if a is None:
            continue
        # Thin strokes (a brow) drawn a pixel apart still agree.
        grow = lambda m: cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        a_in, b_in = grow(a > 0.5), grow(b > 0.5)
        union = (a_in | b_in) & core
        if ((a > 0.5) & core).sum() < MIN_PIXELS and ((b > 0.5) & core).sum() < MIN_PIXELS:
            continue
        iou = (a_in & b_in & core).sum() / max(1, union.sum())
        seen = (b > 0.5) & (shows == index) & core
        cover = (a_in & seen).sum() / max(1, seen.sum())
        if iou >= AGREE_IOU or cover >= AGREE_COVER:
            own[name] = name
            if iou < AGREE_IOU:
                log(f'{name:14s} {tile.name}: IoU {iou:.2f} with the whole, but covers {cover:.2f} of it')
        else:
            log(f'{name:14s} {tile.name}: IoU {iou:.2f} with the whole, covers {cover:.2f} of it')
    # The whole's layers this tile left empty, taken in by another of its layers.
    taken = {}
    for name in order:
        if name in own:
            continue
        b = (whole_a[name] > 0.5) & core
        if b.sum() < MIN_PIXELS:
            continue
        best = None
        for tname, a in tile_a.items():
            if tname not in own:
                continue
            share = ((a > 0.5) & b).sum() / b.sum()
            if share < TAKEN_IN:
                continue
            region = (a > 0.5) & b
            colour = np.abs(tile_rgb[tname][region] / np.maximum(a[region][:, None], 1e-3)
                            - whole_rgb[name][region] / np.maximum(whole_a[name][region][:, None], 1e-3)).mean()
            if colour < TAKEN_IN_COLOUR and (best is None or share > best[1]):
                best = (tname, share, colour)
        if best:
            taken.setdefault(best[0], []).append(name)
            log(f'{name:14s} {tile.name}: taken in by its {best[0]} ({best[1]:.2f}, colour {best[2]:.3f}), split out')
    for name, tname in own.items():
        out.setdefault(name, []).append((tname, None if tname not in taken else [name] + taken[tname]))
    for tname, names in taken.items():
        for name in names:
            out.setdefault(name, []).append((tname, [own[tname]] + taken[tname]))
    return out


def tile_weight_small(tile, box, s, out_hw):
    class Scaled:
        pass
    scaled = Scaled()
    scaled.box = [v * s for v in tile.box]
    return tile_weight(scaled, box, (out_hw[0] * s, out_hw[1] * s))[..., 0]


# ---------------------------------------------------------------- stitching

# Colour distance past which a stitched pixel shows the picture wrong (as utils/gaps.py).
WRONG_DISTANCE = 98
# The whole stands where it is this much closer to the picture (colour distance).
BETTER_BY = 30
# Wrong places smaller than this many output pixels are a strand's edge, not a dropped part.
WRONG_MIN_AREA = 1500


def _composite(pieces, out_hw):
    """The premultiplied pieces (box, rgba) drawn back to front: straight colour and alpha."""
    H, W = out_hw
    color = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)
    for box, rgba in pieces:
        y0, x0, y1, x1 = box
        a = rgba[..., 3]
        color[y0:y1, x0:x1] = rgba[..., :3] + color[y0:y1, x0:x1] * (1 - a[..., None])
        alpha[y0:y1, x0:x1] = a + alpha[y0:y1, x0:x1] * (1 - a)
    return color / np.maximum(alpha, 1e-6)[..., None], alpha


def _whole_better(stitched, picture, out_hw):
    """Where the stitched layers show the picture (RGB 0..255 at out_hw) wrong and the whole's right, in places of some size."""
    target = picture[..., :3].astype(np.float32) / 255
    tc, ta = _composite([(box, pre) for _, box, pre, _, _ in stitched], out_hw)
    wc, wa = _composite([(box, base) for _, box, _, base, _ in stitched], out_hw)
    whole_error = np.linalg.norm(wc - target, axis=-1)
    tiles_error = np.linalg.norm(tc - target, axis=-1)
    whole_right = (wa > 0.5) & (whole_error <= WRONG_DISTANCE / 255)
    # Wrong by much more than the whole (grey hair through a navy skirt is within 98 of it), or
    # only part drawn where the whole is not (a skirt one tile drew and the next did not, faded).
    tiles_wrong = (ta < np.minimum(wa, 0.9)) | (tiles_error > whole_error + BETTER_BY / 255)
    wrong = cv2.morphologyEx((whole_right & tiles_wrong).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(wrong, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= WRONG_MIN_AREA
    return cv2.dilate(keep[labels].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def stitch(whole, tiles, out_hw, path, log=print, picture=None):
    """
    Writes the figure's PSD at out_hw (the picture's size): the whole's layers
    and order, each layer's pixels from the tiles that drew it, faded into one
    another and into the whole's where no tile did, and the whole's where the
    tiles together show the picture (RGB at out_hw, when given) wrong and the
    whole shows it right.
    """
    H, W = out_hw
    order = [name for name, *_ in whole.layers]
    matches = {tile.name: match_tile(whole, tile, order, out_hw, log) for tile in tiles}
    psd = PSDImage.new(mode='RGBA', size=(W, H), depth=8)
    kernel = np.ones((2 * REACH + 1, 2 * REACH + 1), np.uint8)
    stitched = []
    # How much of each pixel neither the whole's layers nor the stitched ones cover.
    whole_alpha = np.ones((H, W), np.float32)
    tiles_alpha = np.ones((H, W), np.float32)
    for name in order:
        base_piece = whole.placed(name)
        pieces = [base_piece]
        drawn = []
        for tile in tiles:
            for tname, owners in matches[tile.name].get(name, []):
                piece = tile.placed(tname)
                if piece is not None:
                    pieces.append(piece)
                    drawn.append((tile, tname, owners, piece))
        boxes = [(p[0], p[1], p[0] + p[2].shape[0], p[1] + p[2].shape[1]) for p in pieces if p is not None]
        if not boxes:
            continue
        box = (max(0, min(b[0] for b in boxes)), max(0, min(b[1] for b in boxes)),
               min(H, max(b[2] for b in boxes)), min(W, max(b[3] for b in boxes)))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        base = paste(box, base_piece)
        num = np.zeros_like(base)
        den = np.zeros(base.shape[:2] + (1,), np.float32)
        used = []
        for tile, tname, owners, piece in drawn:
            mine = paste(box, piece)
            # Within reach of the whole's layers this one stands for; split by them when it stands for several.
            names = owners or [name]
            stack = np.stack([paste(box, whole.placed(n))[..., 3] for n in names])
            keep = cv2.dilate((stack.max(0) > 0.3).astype(np.uint8), kernel) > 0
            under = None
            if owners:
                # Each pixel to the topmost of them drawn there, else the one most drawn.
                topmost = np.full(stack.shape[1:], -1)
                for i in sorted(range(len(names)), key=lambda i: order.index(names[i])):
                    topmost = np.where(stack[i] > 0.5, i, topmost)
                topmost = np.where(topmost < 0, stack.argmax(0), topmost)
                # Where a layer drawn over this one took the pixel, this one lies under it and the
                # tile drew none of it there: the whole's own pixels stand (a sleeve goes on under
                # the cardigan that hides it, so a turn opens no gap between them).
                # Short of their outline by REACH, so no edge of it shows past the layer over it; up to
                # the part of this layer the tile drew, so the two meet without a seam.
                above = np.array([order.index(n) > order.index(name) for n in names])
                inside = cv2.erode((stack.max(0) > 0.5).astype(np.uint8), kernel) > 0
                under = keep & above[topmost] & inside
                keep = keep & (topmost == names.index(name))
            mine *= keep[..., None]
            w = tile_weight(tile, box, out_hw)
            if under is not None:
                w = w * ~under[..., None]
            num += mine * w
            den += w
            used.append(f'{tile.name}{"" if tname == name else "(" + tname + ")"}')
        cover = np.minimum(den, 1)
        tiled = np.where(den > 0, num / np.maximum(den, 1e-6), 0)
        pre = tiled * cover + base * (1 - cover)
        stitched.append((name, box, pre, base, used))
        whole_alpha[box[0]:box[2], box[1]:box[3]] *= 1 - base[..., 3]
        tiles_alpha[box[0]:box[2], box[1]:box[3]] *= 1 - pre[..., 3]
    # Where the tiles together show the picture wrong and the whole shows it right (a tile named a
    # skirt's panel for the top it carries on, the top was kept only near the whole's top, and the
    # back hair the tile drew behind it showed), every layer is the whole's there.
    holes = _whole_better(stitched, picture, out_hw) if picture is not None else \
        (whole_alpha < 0.5) & (tiles_alpha > 0.5)
    if holes.any():
        log(f'the whole stands where the tiles show the picture wrong: {int(holes.sum())} px')
    for name, box, pre, base, used in stitched:
        hole = holes[box[0]:box[2], box[1]:box[3]]
        if (hole & ((base[..., 3] > 0.004) | (pre[..., 3] > 0.004))).any():
            pre = np.where(hole[..., None], base, pre)
            used = used + ['the whole where the tiles show it wrong']
        rgba = pre.copy()
        rgba[..., :3] = np.where(rgba[..., 3:4] > 1e-4, rgba[..., :3] / np.maximum(rgba[..., 3:4], 1e-4), 0)
        ys, xs = np.nonzero(rgba[..., 3] > 0.004)
        if not len(ys):
            continue
        t0, b0, l0, r0 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = (np.clip(rgba[t0:b0, l0:r0], 0, 1) * 255 + 0.5).astype(np.uint8)
        psd.create_pixel_layer(Image.fromarray(crop, 'RGBA'), name=name, top=int(box[0] + t0), left=int(box[1] + l0),
                               opacity=255, compression=Compression.RLE)
        log(f'{name:14s} from {", ".join(used) or "the whole"}')
    psd.save(path)


def stitch_files(image_hw, whole_path, tile_paths, tiles, path, log=print, picture=None):
    """stitch from files: tile_paths and tiles keyed 'upper', 'middle', 'lower' (crops' sizes as crops() made them)."""
    H, W = image_hw
    whole = Source(whole_path, [0, 0, W, H], image_hw, 'whole')
    sources = []
    for name in ('upper', 'middle', 'lower'):
        x0, y0, x1, y1 = tiles[name]
        input_hw = BUST_PIXELS[::-1] if name == 'upper' else (y1 - y0, x1 - x0)
        sources.append(Source(tile_paths[name], tiles[name], input_hw, name))
    stitch(whole, sources, image_hw, path, log, picture)


if __name__ == '__main__':
    import json
    import sys
    image_path, whole_path, upper, middle, lower, tiles_path, out_path = sys.argv[1:8]
    image = Image.open(image_path)
    with open(tiles_path) as f:
        tiles = json.load(f)
    stitch_files((image.height, image.width), whole_path, dict(upper=upper, middle=middle, lower=lower), tiles, out_path)
