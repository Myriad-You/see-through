'''
A part the decomposition leaves out altogether. The hakama under a kimono came
back with its bottomwear empty (2026-10-13), and nothing downstream noticed: a
part only drops out when it is empty. The model cannot be made to see it
(asked for the bottomwear alone on the hakama's own box it drew nothing), so
what it misses is taken from the source.

Once the parts have their depth, they are drawn back to front as the PSD
will have them and compared with the source: where the figure shows
something the drawing does not (nothing, or a part behind the missing one,
as the back hair behind a skirt), the drawing has a gap. A garment-sized gap
is given to the part its place says, from the source's own pixels, and drawn
in behind what hangs in front of it.
'''
import cv2
import numpy as np

# Garment-sized: smaller gaps are accessories and specks (the bottom of a
# birdcage earring, 0.5% of a figure), which Myriad's importer lifts from the
# source up to this same share.
GAP_MIN_SHARE = 0.02
# Alpha (0..255) that counts as drawn.
DRAWN = 40
# Colour distance from the source's backdrop that counts as the figure.
BACKDROP_DISTANCE = 40
# Colour distance past which the drawing shows something else than the source (Myriad's importer's too).
MISMATCH_DISTANCE = 98
# Strands across a gap up to this wide do not split it.
STRAND = 13
# Smaller pieces of a garment have no outline worth completing.
HIDDEN_MIN_PIECE = 400
# How far past a gap the garment takes over what shows of the parts behind it.
REACH = 21


# The source's edge is a backdrop only when this share of it lies near a
# smooth one (a flat colour, a gradient, a vignette; Myriad's upload trim asks
# the same share of a flat one); a scene is not one, and nothing tells the
# figure from it.
FLAT_EDGE_SHARE = 0.6


# The backdrop is fitted again to the edge's pixels this close to the last fit:
# a backdrop's are a few apart, light clothes where a bust is cut ten and more.
BACKDROP_FIT = 15


def backdrop_model(region: np.ndarray):
    '''
    The backdrop's colour at each pixel of the source's region (RGBA): a
    quadratic in x and y fitted to its edge, so a gradient or a vignette is
    backdrop wherever it shows, between locks too; a flat backdrop is its
    colour. Fitted to the edge's pixels near its median colour, then again to
    those close to the last fit (not light clothes where a bust is cut).
    Returns the model and the share of the edge it fits; None for too little
    edge to fit.
    '''
    h, w = region.shape[:2]
    ys = np.concatenate([np.zeros(w), np.full(w, h - 1), np.arange(h), np.arange(h)])
    xs = np.concatenate([np.arange(w), np.arange(w), np.zeros(h), np.full(h, w - 1)])
    ring = region[ys.astype(int), xs.astype(int), :3].astype(np.float64)

    def terms(x, y):
        x, y = x / max(w - 1, 1), y / max(h - 1, 1)
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], -1)

    basis = terms(xs, ys)
    keep = np.linalg.norm(ring - np.median(ring, axis=0), axis=-1) <= BACKDROP_DISTANCE
    for _ in range(3):
        if keep.sum() < 6 * 4:
            return None, 0.0
        coef, *_ = np.linalg.lstsq(basis[keep], ring[keep], rcond=None)
        residual = np.linalg.norm(ring - np.einsum('nk,kc->nc', basis, coef), axis=-1)
        keep = residual <= BACKDROP_FIT
    fit = residual <= BACKDROP_DISTANCE
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float64)
    return np.einsum('hwk,kc->hwc', terms(gx, gy), coef).astype(np.float32), float(fit.mean())


def figure_mask(page: np.ndarray, placement):
    '''
    The figure on the fitted canvas (RGBA): the source's alpha where its
    backdrop is transparent, else what is not its backdrop (backdrop_model).
    None when the edge is not a backdrop: less than FLAT_EDGE_SHARE of it
    fits a smooth one (a scene).
    '''
    _, x, y, w, h = placement
    region = page[y: y + h, x: x + w]
    ring = np.concatenate([region[0], region[-1], region[:, 0], region[:, -1]])
    figure = page[..., 3] > 128
    if (ring[:, 3] < 128).mean() > 0.5:
        return figure
    model, share = backdrop_model(region)
    if model is None or share < FLAT_EDGE_SHARE:
        return None
    distance = np.linalg.norm(region[..., :3].astype(np.float32) - model, axis=-1)
    figure[y: y + h, x: x + w] &= distance > BACKDROP_DISTANCE
    return cv2.morphologyEx(figure.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def open_backdrop(page: np.ndarray, placement):
    '''
    The backdrop that joins the source's edge: pixels of its colour
    (backdrop_model) walked in from the edge, 4-connected, and the canvas
    past the source. White drawn inside an outline (a maid's frill on white)
    is not reached; the gaps between the frill's scallops are. None without a
    backdrop.
    '''
    _, x, y, w, h = placement
    model, share = backdrop_model(page[y: y + h, x: x + w])
    if model is None or share < FLAT_EDGE_SHARE:
        return None
    near = np.linalg.norm(page[y: y + h, x: x + w, :3].astype(np.float32) - model, axis=-1) <= BACKDROP_DISTANCE
    _, labels = cv2.connectedComponents(near.astype(np.uint8), connectivity=4)
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    joined = np.unique(edge[edge > 0])
    out = page[..., 3] <= 128
    out[y: y + h, x: x + w] |= np.isin(labels, joined)
    return out


def composite(images):
    '''The parts (canvas RGBA, back to front) drawn together: colour, alpha, and the index of the part on top (-1).'''
    h, w = images[0].shape[:2]
    color = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    top = np.full((h, w), -1, np.int32)
    for index, image in enumerate(images):
        a = image[..., 3:4].astype(np.float32) / 255
        color = image[..., :3].astype(np.float32) * a + color * (1 - a)
        alpha = a + alpha * (1 - a)
        top[image[..., 3] > 128] = index
    return color / np.maximum(alpha, 1e-6), alpha[..., 0], top


def unexplained(figure: np.ndarray, page: np.ndarray, images) -> np.ndarray:
    '''Where the drawing (images, back to front) misses the figure: nothing drawn, or something else than the source on top.'''
    color, alpha, _ = composite(images)
    distance = np.linalg.norm(color - page[..., :3].astype(np.float32), axis=-1)
    return figure & ((alpha < 0.5) | (distance > MISMATCH_DISTANCE))


# A gap that parts draw as the source has it, under what shows there, for
# more than this share is buried rather than left out: a cardigan's front
# panel drawn in the top, under the skirt drawn over it. Myriad's importer
# reveals buried art, cutting only what covers it; given to a part here it
# would go with that part.
BURIED_SHARE = 0.5


def drawn_right(page: np.ndarray, images) -> np.ndarray:
    '''Where some part (canvas RGBA) draws what the source shows, whether or not it shows there.'''
    out = np.zeros(page.shape[:2], bool)
    for image in images:
        distance = np.linalg.norm(image[..., :3].astype(np.float32) - page[..., :3].astype(np.float32), axis=-1)
        out |= (image[..., 3] > 128) & (distance <= MISMATCH_DISTANCE)
    return out


def find_gaps(figure: np.ndarray, page: np.ndarray, images, head_xyxy=None):
    '''
    The drawing's gaps (unexplained), largest first, each a boolean mask of
    at least GAP_MIN_SHARE of the figure, not centred in the head's box
    (x1, y1, x2, y2), and not buried (BURIED_SHARE); strands across a gap
    do not split it.
    '''
    gap = cv2.morphologyEx(unexplained(figure, page, images).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    gap = (cv2.morphologyEx(gap, cv2.MORPH_CLOSE, np.ones((STRAND, STRAND), np.uint8)) > 0) & figure
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(gap.astype(np.uint8), connectivity=8)
    floor = GAP_MIN_SHARE * figure.sum()
    right = None
    found = []
    for k in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
        if stats[k, cv2.CC_STAT_AREA] < floor:
            break
        cx, cy = centroids[k]
        if head_xyxy is not None and head_xyxy[0] <= cx < head_xyxy[2] and head_xyxy[1] <= cy < head_xyxy[3]:
            continue
        if right is None:
            right = drawn_right(page, images)
        mask = labels == k
        if right[mask].mean() > BURIED_SHARE:
            continue
        found.append(mask)
    return found


def place_tag(gap: np.ndarray, parts: dict) -> str:
    '''
    The part a gap is of, by its place: above the chin headwear; below the
    lower garment the legs'; to one side of the body's middle (the face's,
    else the top's), not across it, from the shoulders down and above the
    lower garment, an arm's (handwear-r on the picture's left, as See-through
    names them); below the top's middle the lower garment's, as is what
    lies across the lower garment where there is no top (a figure's lower
    tile); else an object's.
    '''
    def drawn(tag, axis):
        if tag not in parts:
            return np.zeros(0, int)
        return np.nonzero(parts[tag][..., 3].max(axis=axis) > DRAWN)[0]

    ys, xs = np.nonzero(gap)
    cy = ys.mean()
    face_rows = drawn('face', 1)
    if len(face_rows) and cy < face_rows.max():
        return 'headwear'
    lower, top = drawn('bottomwear', 1), drawn('topwear', 1)
    # Legs start at the hem; what runs on down from well above it is the rest of the skirt.
    if len(lower) and cy > lower.max() and lower.max() - ys.min() <= 0.1 * (ys.max() - ys.min()):
        return 'legwear'
    across = drawn('face', 0) if len(face_rows) else drawn('topwear', 0)
    if len(across):
        middle = across.mean()
        aside = xs.max() < middle or xs.min() > middle
        if aside and (not len(top) or ys.max() > top.min()) and (not len(lower) or ys.min() < lower.min()):
            return 'handwear-r' if xs.mean() < middle else 'handwear-l'
    if len(top) and cy >= (top.min() + top.max()) / 2:
        return 'bottomwear'
    # A figure's lower tile has the skirt and the legs but no top: across the lower garment it is the lower garment.
    if not len(top) and len(lower) and ys.min() <= lower.max() and ys.max() >= lower.min():
        return 'bottomwear'
    return 'objects'


# A gap is the rest of a worn part it touches along this share of its rim at
# least, where the source's colours carry on across the seam (mean distance,
# as Myriad's importer matches them) and the part's own colours are the gap's.
SEAM_SHARE = 0.1
SEAM_DISTANCE = 70
SEAM_REACH = 3
WORN = ('topwear', 'bottomwear', 'legwear', 'handwear', 'footwear', 'headwear')


def carried_on(gap: np.ndarray, page: np.ndarray, names, images, top: np.ndarray):
    '''
    The worn part (its name) the gap is the rest of, or None: a skirt drawn
    only in part. A sleeve that took a skirt's edge in touches it there in
    the skirt's colours, but along little of its rim, and is a sleeve's
    colour itself.
    '''
    band = np.ones((2 * SEAM_REACH + 1,) * 2, np.uint8)
    rim = (cv2.dilate(gap.astype(np.uint8), band) > 0) & ~gap & (top >= 0)
    if not rim.any():
        return None
    inner = (gap & ~(cv2.erode(gap.astype(np.uint8), band) > 0)).astype(np.float32)
    colour = page[..., :3].astype(np.float32)
    weight = cv2.GaussianBlur(inner, (0, 0), SEAM_REACH)
    local = cv2.GaussianBlur(colour * inner[..., None], (0, 0), SEAM_REACH) / np.maximum(weight, 1e-6)[..., None]
    seam = np.linalg.norm(colour - local, axis=-1)
    own = np.median(colour[gap], axis=0)
    best, most = None, SEAM_SHARE * rim.sum()
    for index, name in enumerate(names):
        if kind(name) not in WORN:
            continue
        touch = rim & (top == index) & (weight > 1e-3)
        if touch.sum() < most or seam[touch].mean() > SEAM_DISTANCE:
            continue
        shows = (top == index) & (images[index][..., 3] > 128)
        if np.linalg.norm(np.median(colour[shows], axis=0) - own) > SEAM_DISTANCE:
            continue
        best, most = name, touch.sum()
    return best


def fill_from_source(part: np.ndarray, page: np.ndarray, gap: np.ndarray, figure: np.ndarray) -> np.ndarray:
    '''part (RGBA) with the source's own pixels over the gap, the holes inside it and its rim, within the figure.'''
    whole = gap.astype(np.uint8)
    contours, _ = cv2.findContours(whole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(whole, contours, -1, 1, thickness=cv2.FILLED)
    area = (cv2.dilate(whole, np.ones((5, 5), np.uint8)) > 0) & figure
    over = area & (page[..., 3] > part[..., 3])
    out = part.copy()
    out[over] = page[over]
    return out


# What hangs in front of a part where it was hidden at rest, whatever depth
# the parts were given (a kimono bust had its sleeves behind its top): the
# sleeves over a skirt, locks over anything, braids over the arms. Parts
# behind it (back hair behind a skirt, legs under it) are left as they show.
IN_FRONT_OF = {
    'bottomwear': ('handwear', 'front hair', 'objects'),
    'handwear': ('front hair', 'back hair', 'objects'),
    'legwear': ('handwear', 'front hair', 'bottomwear', 'objects'),
    'objects': ('handwear', 'front hair'),
}


def kind(name: str) -> str:
    '''A part's tag without its side: handwear-l is handwear.'''
    return name[:-2] if name.endswith(('-l', '-r')) else name


# A pixel is of the garment's colours when it is as close to the garment's
# colours close by as this share of the gap's own pixels are.
GARMENT_LIKE_SHARE = 0.9


def garment_colours(gap: np.ndarray, page: np.ndarray) -> np.ndarray:
    '''Where the source shows the colours of the garment the gap is of, close by (within about REACH).'''
    weight = gap.astype(np.float32)
    total = cv2.GaussianBlur(page[..., :3].astype(np.float32) * weight[..., None], (0, 0), REACH)
    local = total / np.maximum(cv2.GaussianBlur(weight, (0, 0), REACH), 1e-6)[..., None]
    distance = np.linalg.norm(page[..., :3].astype(np.float32) - local, axis=-1)
    return distance <= np.percentile(distance[gap], GARMENT_LIKE_SHARE * 100)


def shown_over(images, indices, page, like=None) -> np.ndarray:
    '''
    Where the parts (indices) draw what the source shows, with their soft
    rims: a lock's strands over a skirt, outlines and all. Not the garment's
    to take from the source. With like (a mask of the garment's colours),
    only where the source is not of those colours: a part behind it may
    have drawn some of the garment.
    '''
    core = np.zeros(page.shape[:2], bool)
    rim = np.zeros(page.shape[:2], bool)
    for i in indices:
        image = images[i]
        distance = np.linalg.norm(image[..., :3].astype(np.float32) - page[..., :3].astype(np.float32), axis=-1)
        core |= (image[..., 3] > DRAWN) & (distance <= MISMATCH_DISTANCE)
        rim |= image[..., 3] > 10
    if like is not None:
        core &= ~like
        rim &= ~like
    return (cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & rim


def taken_beside(gap: np.ndarray, figure: np.ndarray, behind: np.ndarray, like: np.ndarray) -> np.ndarray:
    '''
    Beside the gap (within twice REACH, joined to it), where a part behind
    the garment shows its colours: the garment's, drawn into that part (a
    sleeve's layer took the hakama's lit edge beside it).
    '''
    near = cv2.dilate(gap.astype(np.uint8), np.ones((4 * REACH + 1, 4 * REACH + 1), np.uint8)) > 0
    beside = near & ~gap & figure & behind & like
    # Past the gap's outline, not the thin lines it runs on into (the top's dark outline past the band's ends).
    outline = np.zeros(gap.shape, np.uint8)
    contours, _ = cv2.findContours(gap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.fillPoly(outline, [cv2.convexHull(np.concatenate(contours))], 1)
    solid = cv2.morphologyEx((gap | beside).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    beside &= solid | (outline > 0)
    _, labels = cv2.connectedComponents((gap | beside).astype(np.uint8), connectivity=8)
    joined = np.unique(labels[gap])
    return beside & np.isin(labels, joined[joined > 0])


# How far past a part in front of a garment the hidden side looks for what shows beyond it.
ACROSS = 64
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def hidden_side(garment: np.ndarray, in_front: np.ndarray) -> np.ndarray:
    '''
    Where a garment lifted from the source lies hidden at rest: inside the
    outline of its pieces (the convex hull of those up to ACROSS apart, as a
    braid across a sleeve leaves it), where a part in front
    of it shows (in_front, a mask), and where, looking past that part every
    way, the garment shows beyond it at least as often as anything else does.
    A sleeve over a skirt's edge has the skirt above, below and inside it; a
    lock over the sleeve beside a skirt's waist has the sleeve either side.
    '''
    shown = garment[..., 3] > 128
    hull = np.zeros(shown.shape, np.uint8)
    # Pieces up to ACROSS apart are one (a sleeve a braid hangs across); each group's outline is its hull.
    joined = cv2.dilate(shown.astype(np.uint8), np.ones((ACROSS + 1, ACROSS + 1), np.uint8))
    count, groups = cv2.connectedComponents(joined, connectivity=8)
    for k in range(1, count):
        piece = shown & (groups == k)
        if piece.sum() < HIDDEN_MIN_PIECE:
            continue
        contours, _ = cv2.findContours(piece.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(hull, [cv2.convexHull(np.concatenate(contours))], 1)
    candidates = (hull > 0) & in_front & ~shown
    if not candidates.any():
        return candidates
    h, w = shown.shape
    ys, xs = np.nonzero(candidates)
    y0, y1 = max(0, ys.min() - ACROSS), min(h, ys.max() + ACROSS + 1)
    x0, x1 = max(0, xs.min() - ACROSS), min(w, xs.max() + ACROSS + 1)
    # Beyond the parts in front: 1 the garment, 2 anything else (nothing too); 0 still in front.
    seen = np.where(shown, 1, np.where(in_front, 0, 2))[y0: y1, x0: x1].astype(np.uint8)
    sh, sw = seen.shape
    padded = np.pad(seen, ACROSS, constant_values=2)
    garment_votes = np.zeros((sh, sw), np.uint8)
    other_votes = np.zeros((sh, sw), np.uint8)
    for dx, dy in DIRECTIONS:
        hit = np.zeros((sh, sw), np.uint8)
        for k in range(1, ACROSS + 1):
            look = padded[ACROSS + k * dy: ACROSS + k * dy + sh, ACROSS + k * dx: ACROSS + k * dx + sw]
            hit = np.where(hit == 0, look, hit)
        garment_votes += hit == 1
        other_votes += hit == 2
    # Rays a pixel apart pass different pixels on the diagonals: the votes, smoothed over a few
    # pixels, decide, or a ragged edge leaves the hidden side dithered.
    votes = (garment_votes + other_votes).astype(np.float32)
    voted = (votes > 0).astype(np.float32)
    share = np.where(votes > 0, garment_votes / np.maximum(votes, 1), 0).astype(np.float32)
    share = cv2.GaussianBlur(share * voted, (0, 0), 3) / np.maximum(cv2.GaussianBlur(voted, (0, 0), 3), 1e-6)
    out = np.zeros(shown.shape, bool)
    out[y0: y1, x0: x1] = candidates[y0: y1, x0: x1] & (share >= 0.5)
    return out


def complete_hidden(garment: np.ndarray, hidden: np.ndarray, inpaint) -> np.ndarray:
    '''
    garment (RGBA) with its hidden side drawn in by inpaint(rgb, mask255)
    from what of it shows alone: only the hidden side is asked for, with the
    rest around it given the colour of the garment nearest it, so neither
    the parts in front of it nor the backdrop colour the fill, and the
    inpainter is not left to make up a whole picture around a sleeve.
    '''
    if not hidden.any():
        return garment
    shown = garment[..., 3] > 128
    ys, xs = np.nonzero(hidden | shown)
    h, w = hidden.shape
    y0, y1 = max(0, ys.min() - 16), min(h, ys.max() + 17)
    x0, x1 = max(0, xs.min() - 16), min(w, xs.max() + 17)
    crop = shown[y0: y1, x0: x1]
    rgb = np.ascontiguousarray(garment[y0: y1, x0: x1, :3])
    _, labels = cv2.distanceTransformWithLabels((~crop).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    sy, sx = np.nonzero(crop)
    nearest = np.zeros((labels.max() + 1, 3), np.uint8)
    nearest[labels[sy, sx]] = rgb[sy, sx]
    rgb[~crop] = nearest[labels[~crop]]
    region = hidden[y0: y1, x0: x1]
    filled = inpaint(rgb, np.where(region, 255, 0).astype(np.uint8))
    out = garment.copy()
    out[y0: y1, x0: x1][region, :3] = filled[region]
    out[y0: y1, x0: x1][region, 3] = 255
    return out


# A part that hangs in front of a garment is brought in front of it when the
# drawing at rest gets worse from the source in no more than this share of the figure.
MOVE_TOLERANCE = 0.002


def worse(images, page, figure, before_error) -> int:
    '''How many figure pixels the drawing (images) shows further from the source than before_error allowed.'''
    color, alpha, _ = composite(images)
    error = np.linalg.norm(color - page[..., :3].astype(np.float32), axis=-1) * (alpha >= 0.5) + 1e3 * (alpha < 0.5)
    return int((figure & (error > MISMATCH_DISTANCE) & (before_error <= MISMATCH_DISTANCE)).sum())


def palette_colours(samples: np.ndarray, page: np.ndarray, clusters: int = 8) -> np.ndarray:
    '''Where the page is of the samples' colours (RGB rows): as close to one of their clusters as nine in ten of them are.'''
    data = samples.astype(np.float32)
    k = min(clusters, len(data))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centres = cv2.kmeans(data, k, None, criteria, 2, cv2.KMEANS_PP_CENTERS)
    def nearest(colours):
        return np.min(np.linalg.norm(colours[:, None, :] - centres[None, :, :], axis=-1), axis=1)
    limit = np.percentile(nearest(data), GARMENT_LIKE_SHARE * 100)
    h, w = page.shape[:2]
    return (nearest(page[..., :3].reshape(-1, 3).astype(np.float32)) <= limit).reshape(h, w)


# The parts the head pass draws, on its own box of the source.
HEAD_PASS_TAGS = ('headwear', 'face', 'irides', 'eyebrow', 'eyewhite', 'eyelash', 'eyewear', 'ears', 'earwear', 'nose', 'mouth')
# A head part drawn along this many pixels of an edge of that box is cut there.
CUT_RUN = 12
# How far past its box, as a share of the box's height, a cut part goes on.
CUT_REACH = 0.3


def extend_cut(names, images, page, placement, box, log=print):
    '''
    The head pass draws its parts on its own box of the source, and a part
    reaching past it is cut straight across there (a maid's frill above the
    head, the tails of the bows beside it): it goes on past the box from the
    source's own pixels, joined to it where it was cut, as far as CUT_REACH
    of the box, where they are the figure (not the backdrop open to the
    source's edge: a frill's white inside its outline counts), no other part
    draws them as the source has them (hair showing through), and they are
    of its colours there. In place; returns the names of the parts it grew.
    '''
    backdrop = open_backdrop(page, placement)
    if backdrop is None:
        return []
    h, w = page.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 * CUT_RUN or y2 - y1 < 2 * CUT_RUN:
        return []
    inside = np.zeros((h, w), bool)
    inside[y1: y2, x1: x2] = True
    reach = int(CUT_REACH * (y2 - y1))
    near_box = np.zeros((h, w), bool)
    near_box[max(0, y1 - reach): min(h, y2 + reach), max(0, x1 - reach): min(w, x2 + reach)] = True
    grown = []
    for index, name in enumerate(names):
        if kind(name) not in HEAD_PASS_TAGS:
            continue
        drawn = images[index][..., 3] > 128
        # Where it is drawn along each edge, the pixel just past it seeds what goes on beyond.
        seeds = np.zeros((h, w), bool)
        for line, beyond in (((slice(y1, y1 + 1), slice(x1, x2)), (y1 - 1, None)),
                             ((slice(y2 - 1, y2), slice(x1, x2)), (y2, None)),
                             ((slice(y1, y2), slice(x1, x1 + 1)), (None, x1 - 1)),
                             ((slice(y1, y2), slice(x2 - 1, x2)), (None, x2))):
            along = drawn[line]
            if along.sum() < CUT_RUN:
                continue
            by, bx = beyond
            if by is not None and 0 <= by < h:
                seeds[by, x1: x2] |= along[0]
            if bx is not None and 0 <= bx < w:
                seeds[y1: y2, bx] |= along[:, 0]
        if not seeds.any():
            continue
        others = [images[i] for i in range(len(images)) if i != index]
        # Of its colours: as close to the colours it has close to the cut as nine in ten of them are.
        band = drawn & inside & ~(cv2.erode(inside.astype(np.uint8), np.ones((2 * REACH + 1,) * 2, np.uint8)) > 0)
        if band.sum() < CUT_RUN:
            continue
        like = palette_colours(page[band][:, :3], page)
        candidate = near_box & ~inside & ~drawn & ~backdrop & ~drawn_right(page, others) & like & (page[..., 3] > 128)
        _, labels = cv2.connectedComponents((candidate | seeds).astype(np.uint8), connectivity=8)
        joined = np.unique(labels[seeds & candidate])
        add = candidate & np.isin(labels, joined[joined > 0])
        if not add.any():
            continue
        images[index] = images[index].copy()
        images[index][add] = page[add]
        grown.append(name)
        log(f'[gaps] {name} was cut at the head pass\'s box: {int(add.sum())} px past it from the source')
    return grown


def cover(names, images, page, placement, head_xyxy=None, inpaint=None, log=print):
    '''
    The parts (names, canvas RGBA images, back to front) with the figure's
    garment-sized gaps given to the part their place says. Returns the parts'
    new order and images (names, images, back to front) and what changed, or
    None when nothing did. A new part goes just in front of the frontmost one
    it covers; a part that hangs in front of it (IN_FRONT_OF) but was put
    behind is brought in front of it when the source allows, and the garment
    is drawn in behind what hangs in front of it.
    '''
    figure = figure_mask(page, placement)
    if figure is None:
        log('[gaps] the source has no flat backdrop to tell the figure from: not looked for')
        return None
    found = find_gaps(figure, page, images, head_xyxy)
    if not found:
        return None
    names, images = list(names), list(images)
    raw = unexplained(figure, page, images)
    _, _, top = composite(images)
    changes = []
    for gap in found:
        tag = carried_on(gap, page, names, images, top) or place_tag(gap, dict(zip(names, images)))
        hangs = IN_FRONT_OF.get(kind(tag), ())
        ahead = [i for i, name in enumerate(names) if kind(name) in hangs]
        # What it covers: what shows where the drawing was wrong (not the strands bridged across
        # it), but for what hangs in front of it, whose redrawn strands may stray over it.
        covered = top[gap & raw & (top >= 0) & ~np.isin(top, ahead)]
        front = int(covered.max()) + 1 if covered.size else len(names)
        if tag in names:
            at = names.index(tag)
            where = 'grown'
        else:
            at = front
            where = f'new, in front of {names[at - 1]}' if 0 < at <= len(names) else 'new'
            names.insert(at, tag)
            images.insert(at, np.zeros_like(images[0]))
        # Close by, what shows of a part behind it is its too where the source shows it: a top
        # drawn on down over a hakama's band left square notches the gap's cleaning cut off.
        reach = (cv2.morphologyEx(gap.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((REACH, REACH), np.uint8)) > 0) & (top < at)
        like = garment_colours(gap, page)
        # What the parts in front of it draw where it shows stays theirs (a lock's strands, outlines
        # and all): drawn in behind them below, not lifted with them. So does what a part behind it
        # that hangs over it draws of its own colours.
        over = shown_over(images, range(at + 1, len(names)), page) | shown_over(
            images, [i for i in range(at) if kind(names[i]) in hangs], page, like)
        beside = taken_beside(gap, figure, top < at, like)
        images[at] = fill_from_source(images[at], page, gap | reach | beside, figure & ~over)
        # A part left all but empty keeps a haze where it was meant to be (an arm drawn at alpha 1..9):
        # off its edge it would show once what hangs over it swings away.
        part_alpha = images[at][..., 3]
        haze = (part_alpha <= 128) & ~(cv2.dilate((part_alpha > 128).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
        images[at][haze] = 0
        color, alpha, _ = composite(images)
        error = np.linalg.norm(color - page[..., :3].astype(np.float32), axis=-1) * (alpha >= 0.5) + 1e3 * (alpha < 0.5)
        # A part grown where it was put behind what covers it (headwear behind the back hair it sits
        # on) comes in front of that, when the gap still mostly shows wrong and the source allows.
        if where == 'grown' and front > at + 1 and (error[gap] > MISMATCH_DISTANCE).mean() > 0.5:
            trial_names = names[:at] + names[at + 1:]
            trial_images = images[:at] + images[at + 1:]
            trial_names.insert(front - 1, tag)
            trial_images.insert(front - 1, images[at])
            if worse(trial_images, page, figure, error) <= MOVE_TOLERANCE * figure.sum():
                where = f'grown, brought in front of {trial_names[front - 2]}'
                names, images = trial_names, trial_images
                at = front - 1
                color, alpha, _ = composite(images)
                error = np.linalg.norm(color - page[..., :3].astype(np.float32), axis=-1) * (alpha >= 0.5) + 1e3 * (alpha < 0.5)
        # What hangs in front of it but was put behind comes in front, if the source allows.
        moved = []
        for name in [n for n in names[:at] if kind(n) in hangs]:
            i = names.index(name)
            trial_names = names[:i] + names[i + 1:]
            trial_images = images[:i] + images[i + 1:]
            g = trial_names.index(tag)
            trial_names.insert(g + 1, name)
            trial_images.insert(g + 1, images[i])
            if worse(trial_images, page, figure, error) <= MOVE_TOLERANCE * figure.sum():
                names, images = trial_names, trial_images
                moved.append(name)
        at = names.index(tag)
        _, _, top = composite(images)
        in_front = np.zeros(figure.shape, bool)
        for i in range(at + 1, len(names)):
            if kind(names[i]) in hangs:
                in_front |= images[i][..., 3] > DRAWN
        behind = hidden_side(images[at], in_front)
        hidden = 0
        if inpaint is not None and behind.any():
            try:
                images[at] = complete_hidden(images[at], behind, inpaint)
                hidden = int(behind.sum())
            except Exception as error:  # the inpainter is a nicety; the garment stands without it
                log(f'[gaps] hidden side not drawn: {error!r}')
        changes.append((tag, where, moved))
        log(f'[gaps] {int(gap.sum())} px of the figure the drawing missed -> {tag} ({where})'
            f'{"; brought in front of it: " + ", ".join(moved) if moved else ""}; {hidden} px drawn in behind what hangs in front')
    return names, images, changes


def cover_parts(part_dict_list, page, placement, inpaint=None, log=print, head_box=None):
    '''
    cover on the parts as further_extr has them before the PSD (each a dict of
    tag, img cropped to xyxy, its depth (0..255) and depth_median, the PSD
    drawing them by depth_median, far first), in place, after the head parts
    cut at the head pass's box (head_box, canvas x1, y1, x2, y2) are carried
    on past it (extend_cut): a new part is added, and the parts given a new
    place get depth_medians between their new neighbours'; the others keep
    theirs. Returns what changed (cover's).
    '''
    h, w = page.shape[:2]
    order = sorted(part_dict_list, key=lambda part: part['depth_median'], reverse=True)

    def canvas(part):
        out = np.zeros((h, w, 4), np.uint8)
        x1, y1, x2, y2 = part['xyxy']
        out[y1: y2, x1: x2] = part['img']
        return out

    names, images = [part['tag'] for part in order], [canvas(part) for part in order]
    grown = extend_cut(names, images, page, placement, head_box, log) if head_box is not None else []
    face = next((part for part in order if part['tag'] == 'face'), None)
    covered = cover(names, images, page, placement, tuple(face['xyxy']) if face else None, inpaint, log)
    if covered is None and not grown:
        return []
    names, images, changes = covered if covered is not None else (names, images, [])
    by_tag = {part['tag']: part for part in part_dict_list}
    # A part carried on past the head pass's box keeps its depth; its new pixels take its median.
    for tag in set(grown) - {tag for tag, _, _ in changes}:
        part = by_tag[tag]
        image = images[names.index(tag)]
        x, y, bw, bh = cv2.boundingRect(cv2.findNonZero((image[..., 3] > 10).astype(np.uint8)))
        depth = np.full((h, w), round(float(part['depth_median']) * 255), np.uint8)
        x1, y1, x2, y2 = part['xyxy']
        depth[y1: y2, x1: x2] = part['depth']
        part.update(img=image[y: y + bh, x: x + bw], xyxy=[x, y, x + bw, y + bh], depth=depth[y: y + bh, x: x + bw])
    if not changes:
        return []
    placed = {tag for tag, _, _ in changes} | {name for _, _, moved in changes for name in moved}
    for tag in placed - by_tag.keys():
        part = {'tag': tag, 'depth': np.zeros((0, 0), np.uint8), 'xyxy': [0, 0, 0, 0]}
        part_dict_list.append(part)
        by_tag[tag] = part
    # Each run of placed parts shares out the depth between the parts behind and in front of it.
    medians = [None if name in placed else float(by_tag[name]['depth_median']) for name in names]
    index = 0
    while index < len(names):
        if medians[index] is not None:
            index += 1
            continue
        end = index
        while end < len(names) and medians[end] is None:
            end += 1
        behind = medians[index - 1] if index > 0 else 1.0
        ahead = medians[end] if end < len(names) else min(0.0, behind - 1e-3)
        for k in range(index, end):
            medians[k] = behind - (behind - ahead) * (k - index + 1) / (end - index + 1)
        index = end
    if any(medians[k - 1] <= medians[k] or (k + 1 < len(names) and medians[k] <= medians[k + 1])
           for k in range(len(names)) if names[k] in placed and k > 0):
        # Placed between parts tied in depth: the order is kept by rank instead.
        medians = list(np.linspace(1.0, 0.0, len(names)))
    for name, median in zip(names, medians):
        by_tag[name]['depth_median'] = median
    for name in placed:
        part = by_tag[name]
        image = images[names.index(name)]
        x, y, bw, bh = cv2.boundingRect(cv2.findNonZero((image[..., 3] > 10).astype(np.uint8)))
        depth = np.full((h, w), round(part['depth_median'] * 255), np.uint8)
        x1, y1, x2, y2 = part['xyxy']
        if part['depth'].size:
            depth[y1: y2, x1: x2] = part['depth']
        part.update(img=image[y: y + bh, x: x + bw], xyxy=[x, y, x + bw, y + bh], depth=depth[y: y + bh, x: x + bw])
    return changes
