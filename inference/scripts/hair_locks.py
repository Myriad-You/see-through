"""Hair into locks, cut the way a Live2D rigger cuts a front-hair drawing.

A lock runs from the crown to its own tips; locks part where the drawing
parts them. Each pixel follows the strands down (the drawn strand direction,
or radially from the crown where the drawing is flat) to the tip it ends at,
which gives root-to-tip strips. Tips with a deep gap of background between
them end different locks; neighbours with only the shallow V of one lock's
forked ends stay one. Inside a lock, strips stay apart only where a crease
(a drawn line or a shaded valley along the strands) parts them nearly all the
way down, and only if each side is a lock of its own: wide enough, reaching
the root, not a sliver along the silhouette. A thin tuft above the crown (an
ahoge) is a lock of its own.

Every length is tuned at WORK_WIDTH pixels of hair width; the layer is
scaled there and the labels scaled back.

  split_locks(rgb, alpha) -> labels (0 outside, 1..n locks, by mean x; the
  tuft, if any, last), info
"""
import cv2
import numpy as np

WORK_WIDTH = 600
# A gap between two tips this deep (against max(GAP_SCALE of the hair's
# height, 1.5 x their spacing)) ends one lock and starts the next.
GAP_SCALE = 0.12
DEEP_GAP = 0.6
# Inside a group of tips: a crease this strong (share of the 95th percentile)
# over this share of the border, away from the crown, keeps two strips apart.
CREASE_SHARE = 0.35
CREASE_COVER = 0.5
CROWN_ZONE = 0.3
# A lock is at least this wide and this large (shares of the hair), and
# reaches up into the top ROOT_ZONE of the hair below the crown.
MIN_WIDTH = 0.035
MIN_AREA = 0.015
ROOT_ZONE = 0.3
MIN_BORDER = 60


def split_locks(rgb, alpha):
    """Labels at the input size and what the cut was made from (crown, tips at the input scale)."""
    xs = np.nonzero((alpha > 0.3).any(0))[0]
    if len(xs) < 2:
        return np.zeros(alpha.shape, np.int32), dict(locks=0)
    scale = WORK_WIDTH / max(1, xs.max() - xs.min())
    H0, W0 = alpha.shape
    size = (max(1, round(W0 * scale)), max(1, round(H0 * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    lab, info = _split(np.clip(cv2.resize(rgb, size, interpolation=interp), 0, 1),
                       np.clip(cv2.resize(alpha, size, interpolation=interp), 0, 1))
    lab = cv2.resize(lab.astype(np.uint16), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    lab[alpha <= 0.3] = 0
    info['crown'] = (int(info['crown'][0] / scale), int(info['crown'][1] / scale))
    info['tips'] = [(int(y / scale), int(x / scale)) for y, x in info['tips']]
    return lab, info


def _split(rgb, alpha):
    mask = alpha > 0.3
    H, W = mask.shape
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    crown = find_crown(mask)
    tuft = find_tuft(mask, crown)
    hair = mask & ~tuft
    vx, vy = strand_field(gray, hair, crown)
    tips = find_tips(hair, crown)
    if not tips:
        lab = hair.astype(np.int32)
        if tuft.any():
            lab[tuft] = 2
        return lab, dict(crown=crown, tips=[], locks=int(lab.max()))
    fine = follow_strands(hair, tips, vx, vy)
    fine = np.where(hair, cv2.medianBlur(fine.astype(np.uint8), 5).astype(np.int32), 0)
    fine[hair & (fine == 0)] = 1
    groups = tip_groups(hair, tips, crown)

    span = max(1, H - crown[0])
    yy, xx = np.mgrid[0:H, 0:W]
    far = np.hypot(yy - crown[0], xx - crown[1]) > CROWN_ZONE * span
    interior = cv2.distanceTransform(hair.astype(np.uint8), cv2.DIST_L2, 5) > 5
    crease = valley(gray, vx, vy, interior)
    strong = CREASE_SHARE * float(np.percentile(crease[interior], 95)) if interior.any() else np.inf
    hx = np.nonzero(hair.any(0))[0]
    min_width = max(6.0, MIN_WIDTH * (hx.max() - hx.min()))
    min_area = MIN_AREA * hair.sum()
    edge = hair & ~(cv2.erode(hair.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)

    def width(m):
        d = cv2.distanceTransform(np.pad(m.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
        return 2 * float(np.percentile(d[m], 90)) if m.any() else 0.0

    def reaches_root(m):
        return m.any() and yy[m].min() < crown[0] + ROOT_ZONE * span

    def rim(m):
        # A sliver along the silhouette: much of its outline is the hair's own edge, and it is thin.
        ring = m & ~(cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        return ring.any() and (ring & edge).sum() / ring.sum() > 0.35 and width(m) < 2.5 * min_width

    # Within a group of tips, strips merge unless a crease parts them; strips
    # that meet only near the crown, where every strand meets, give no evidence.
    parent = {l: l for l in np.unique(fine) if l > 0}

    def find(l):
        while parent[l] != l:
            parent[l] = parent[parent[l]]
            l = parent[l]
        return l

    for (u, v), (n, creases) in borders(fine, crease, far).items():
        if groups[u - 1] != groups[v - 1] or n < MIN_BORDER or len(creases) < MIN_BORDER // 2:
            continue
        if (creases > strong).mean() < CREASE_COVER:
            parent[find(u)] = find(v)
    lab = np.where(fine > 0, np.vectorize(lambda l: find(l) if l > 0 else 0)(fine), 0)

    # A part kept apart inside its group must be a lock of its own; else it rejoins the group.
    group_of = lambda l: groups[int(np.unique(fine[lab == l])[0]) - 1]
    for l in [l for l in np.unique(lab) if l > 0]:
        m = lab == l
        if m.sum() < min_area or width(m) < min_width or not reaches_root(m) or rim(m):
            mates = [k for k in np.unique(lab) if k > 0 and k != l and group_of(k) == group_of(l)]
            if mates:
                lab[m] = max(mates, key=lambda k: (lab == k).sum())

    # What is still too small or thin for a lock joins the neighbour it shares most border with.
    while True:
        ids = [l for l in np.unique(lab) if l > 0]
        small = sorted([l for l in ids if (lab == l).sum() < min_area or width(lab == l) < min_width],
                       key=lambda l: (lab == l).sum())
        pair = None
        if len(ids) > 1:
            touching = borders(lab)
            for l in small:
                cands = [(n, k) for k, (n, _) in touching.items() if l in k]
                if cands:
                    pair = max(cands)[1]
                    break
        if pair is None:
            break
        u, v = pair
        keep, drop = (u, v) if (lab == u).sum() >= (lab == v).sum() else (v, u)
        lab[lab == drop] = keep

    out = np.zeros_like(lab)
    ids = sorted([l for l in np.unique(lab) if l > 0], key=lambda l: xx[lab == l].mean())
    for k, l in enumerate(ids, 1):
        out[lab == l] = k
    if tuft.any():
        out[tuft] = len(ids) + 1
    return out, dict(crown=crown, tips=tips, locks=int(out.max()))


# ---------------------------------------------------------------- the drawing

def find_crown(mask):
    """Where the hair first spreads wide, below any thin tuft on top."""
    widths = mask.sum(1)
    top = int(np.argmax(widths > 0.25 * widths.max()))
    row = np.nonzero(mask[top])[0]
    return top, int(row.mean())


def find_tuft(mask, crown):
    """A tuft standing above the head (an ahoge): the protrusion holding the topmost point, with its stem."""
    ys, xs = np.nonzero(mask)
    width = xs.max() - xs.min()
    r = max(5, int(0.07 * width))
    dome = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    none = np.zeros_like(mask)
    if not dome.any():
        return none
    dome_top = int(np.nonzero(dome.any(1))[0].min())
    top = int(ys.min())
    if dome_top - top < 0.04 * (ys.max() - top):
        return none
    above = mask & ~dome
    above[dome_top:] = False
    _, lab = cv2.connectedComponents(above.astype(np.uint8))
    k = lab[top, xs[ys == top][0]]
    if k == 0:
        return none
    tuft = lab == k
    # Down from its tip the tuft keeps its own width; where a row grows well past it, that is the head.
    rows = [int(tuft[y].sum()) for y in range(top, dome_top)]
    typical = float(np.median([w for w in rows[: max(1, len(rows) // 2)] if w > 0] or [1]))
    for i, w in enumerate(rows):
        if i > len(rows) // 3 and w > 2.5 * typical:
            tuft[top + i:] = False
            break
    # Its stem, down to the crown: the thin part joined to it there.
    rs = max(3, int(0.03 * width))
    opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rs + 1, 2 * rs + 1))) > 0
    stem = (mask & ~opened) | tuft
    stem[crown[0]:] = False
    _, lab = cv2.connectedComponents(stem.astype(np.uint8))
    ids = np.unique(lab[tuft])
    tuft = np.isin(lab, ids[ids > 0])
    return tuft if tuft.sum() >= 0.003 * mask.sum() else none


def strand_field(gray, mask, crown):
    """Unit strand direction per pixel, pointing away from the crown; radial where the drawing is flat."""
    blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    # Only the drawing inside the hair: the silhouette's own edge says nothing of the strands.
    w = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    jxx, jxy, jyy = (cv2.GaussianBlur(v * w, (0, 0), 7) for v in (gx * gx, gx * gy, gy * gy))
    # Strands run across the gradient: the direction of least change.
    theta = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
    H, W = gray.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rx, ry = xx - crown[1], yy - crown[0]
    rn = np.hypot(rx, ry) + 1e-6
    tx, ty = np.cos(theta), np.sin(theta)
    s = np.sign(tx * rx + ty * ry)
    s[s == 0] = 1
    c = np.clip((coherence - 0.15) / 0.35, 0, 1)
    vx = c * tx * s + (1 - c) * rx / rn
    vy = c * ty * s + (1 - c) * ry / rn
    n = np.hypot(vx, vy) + 1e-6
    return (vx / n).astype(np.float32), (vy / n).astype(np.float32)


def valley(gray, vx, vy, mask, scales=(1.5, 3, 6)):
    """Dark valleys along the strands, thin drawn lines and broad shaded creases alike: curvature across the strands."""
    nx, ny = -vy, vx
    out = np.zeros_like(gray)
    g = np.where(mask, gray, cv2.GaussianBlur(gray, (0, 0), 8))
    for s in scales:
        b = cv2.GaussianBlur(g, (0, 0), s)
        ixx = cv2.Sobel(b, cv2.CV_32F, 2, 0, ksize=3)
        iyy = cv2.Sobel(b, cv2.CV_32F, 0, 2, ksize=3)
        ixy = cv2.Sobel(b, cv2.CV_32F, 1, 1, ksize=3)
        inn = nx * nx * ixx + 2 * nx * ny * ixy + ny * ny * iyy
        out = np.maximum(out, s * s * np.maximum(inn, 0) / 4)
    return np.where(mask, out, 0)


def find_tips(mask, crown):
    """Narrow ends on the outline, below the cap; far ones first, one per 28 px."""
    H = mask.shape[0]
    m8 = mask.astype(np.uint8)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)).astype(np.float32)
    fill = cv2.filter2D(m8.astype(np.float32), -1, disc / disc.sum())
    edge = m8 - cv2.erode(m8, np.ones((3, 3), np.uint8)) > 0
    score = np.where(edge, 1 - fill, 0)
    peak = cv2.dilate(score, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    ty, tx = np.nonzero((score >= peak - 1e-6) & (score > 0.62))
    keep = ty > crown[0] + 0.2 * (H - crown[0])
    ty, tx = ty[keep], tx[keep]
    kept = []
    for i in np.argsort(-np.hypot(ty - crown[0], tx - crown[1])):
        p = (int(ty[i]), int(tx[i]))
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > 28 ** 2 for q in kept):
            kept.append(p)
    return kept


def bilinear(img, x, y, outside=0.0):
    H, W = img.shape
    ok = (x >= 0) & (y >= 0) & (x <= W - 1) & (y <= H - 1)
    xc = np.clip(x, 0, W - 1.001)
    yc = np.clip(y, 0, H - 1.001)
    x0 = xc.astype(np.int32)
    y0 = yc.astype(np.int32)
    fx, fy = xc - x0, yc - y0
    v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) +
         img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return np.where(ok, v, outside)


def follow_strands(mask, tips, vx, vy, step=1.5, iters=600):
    """Each pixel follows the strands away from the crown until it leaves the hair; it belongs to the tip it ends at."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    px, py = xs.astype(np.float32), ys.astype(np.float32)
    live = np.ones(len(px), bool)
    m = mask.astype(np.float32)
    for _ in range(iters):
        if not live.any():
            break
        idx = np.nonzero(live)[0]
        lx, ly = px[idx], py[idx]
        nx = lx + step * bilinear(vx, lx, ly)
        ny = ly + step * bilinear(vy, lx, ly)
        inside = bilinear(m, nx, ny) > 0.5
        px[idx[inside]] = nx[inside]
        py[idx[inside]] = ny[inside]
        live[idx[~inside]] = False
    T = np.array(tips, np.float32)
    lab = np.zeros((H, W), np.int32)
    lab[ys, xs] = np.argmin((py[:, None] - T[None, :, 0]) ** 2 + (px[:, None] - T[None, :, 1]) ** 2, 1) + 1
    return lab


def tip_groups(mask, tips, crown):
    """
    Tips in order round the outline: neighbours with only the shallow V of one
    lock's forked ends between them are one group, a deep gap starts the next,
    and so does going over the top of the head.
    """
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=len)[:, 0, :].astype(np.float32)
    T = np.array([(x, y) for y, x in tips], np.float32)
    at = np.array([int(np.argmin(((c - t) ** 2).sum(1))) for t in T])
    H = mask.shape[0]
    ref = GAP_SCALE * (H - crown[0])
    group = np.zeros(len(tips), np.int32)
    g = 0
    order = np.argsort(at)
    for i, j in zip(order, np.roll(order, -1)):
        group[i] = g
        a, b = at[i], at[j]
        arc = c[a:b + 1] if a <= b else np.concatenate([c[a:], c[:b + 1]])
        p, q = T[i], T[j]
        d = q - p
        ln = float(np.hypot(*d)) + 1e-6
        depth = float(np.abs(((arc[:, 0] - p[0]) * d[1] - (arc[:, 1] - p[1]) * d[0]) / ln).max()) if len(arc) else 0.0
        over_top = (arc[:, 1] < crown[0] + 0.15 * (H - crown[0])).any()
        if over_top or depth / max(ref, 1.5 * ln) >= DEEP_GAP:
            g += 1
    return group


def borders(lab, values=None, keep=None):
    """Per pair of touching labels: border length and, if given, the values along it (where keep holds)."""
    H, W = lab.shape
    if values is not None:
        # A border the strands draw sits within a pixel or two of the drawn crease.
        values = cv2.dilate(values, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    out = {}
    for dy, dx in ((0, 1), (1, 0)):
        a = lab[: H - dy, : W - dx]
        b = lab[dy:, dx:]
        sel = (a != b) & (a > 0) & (b > 0)
        keys = np.minimum(a[sel], b[sel]) * 100000 + np.maximum(a[sel], b[sel])
        if values is not None:
            v = np.maximum(values[: H - dy, : W - dx][sel], values[dy:, dx:][sel])
            k = keep[: H - dy, : W - dx][sel] if keep is not None else np.ones(len(v), bool)
        for key in np.unique(keys):
            m = keys == key
            entry = out.setdefault((int(key // 100000), int(key % 100000)), [0, []])
            entry[0] += int(m.sum())
            if values is not None:
                entry[1].append(v[m & k])
    return {k: (n, np.concatenate(vs) if vs else np.zeros(0, np.float32)) for k, (n, vs) in out.items()}
