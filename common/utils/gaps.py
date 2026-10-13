'''
A part the body pass leaves out altogether. The hakama under a kimono came
back with its bottomwear empty (2026-10-13), and nothing downstream noticed:
a part only drops out when it is empty. What the source shows of the figure
that no part has is a gap. A gap below the head is run once more on its own
box, scaled up as the head is, and the body tag that draws most of it there
takes what it drew of it, hidden parts included. A gap no tag takes is the
source's own pixels, given to the part its place says.
'''
import cv2
import numpy as np

# Garment-sized: smaller gaps are accessories and specks (the bottom of a
# birdcage earring, 0.5% of a figure), which a body pass would not draw and
# Myriad's importer lifts from the source up to this same share.
GAP_MIN_SHARE = 0.02
# The share of a gap a tag must draw in the gap's own pass to take it.
GAP_TAKE = 0.3
# Alpha (0..255) that counts as drawn.
DRAWN = 40
# Colour distance from the source's backdrop that counts as the figure.
BACKDROP_DISTANCE = 40
# Colour distance within which a part draws what the source shows.
MATCH_DISTANCE = 70
# Strands across a gap up to this wide do not split it.
STRAND = 13
# The tags a gap may go to: the body's parts but the head and its hair, which have passes of their own.
GAP_TAGS = ('neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects')


def figure_mask(page: np.ndarray, placement) -> np.ndarray:
    '''
    The figure on the fitted canvas (RGBA): the source's alpha where its
    backdrop is transparent, else what is not its backdrop, the colour along
    the source's edge.
    '''
    _, x, y, w, h = placement
    region = page[y: y + h, x: x + w]
    ring = np.concatenate([region[0], region[-1], region[:, 0], region[:, -1]])
    figure = page[..., 3] > 128
    if (ring[:, 3] < 128).mean() > 0.5:
        return figure
    backdrop = np.median(ring[:, :3].astype(np.float32), axis=0)
    distance = np.linalg.norm(page[..., :3].astype(np.float32) - backdrop, axis=-1)
    figure &= distance > BACKDROP_DISTANCE
    return cv2.morphologyEx(figure.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0


def find_gaps(figure: np.ndarray, page: np.ndarray, parts, head_xyxy=None):
    '''
    The figure's gaps no part explains, largest first, each a boolean mask, of
    at least GAP_MIN_SHARE of the figure and not centred in the head's box
    (canvas x1, y1, x2, y2). A part explains a pixel it draws in the source's
    colour there: one drawn behind a missing part (the back hair the
    decomposition paints behind a skirt) does not. Strands across a gap do
    not split it. page: the fitted source; parts: the parts' RGBA on the canvas.
    '''
    source = page[..., :3].astype(np.float32)
    explained = np.zeros(figure.shape, bool)
    for part in parts:
        drawn = part[..., 3] > DRAWN
        explained |= drawn & (np.linalg.norm(part[..., :3].astype(np.float32) - source, axis=-1) <= MATCH_DISTANCE)
    # The parts' anti-aliased rims fall short of the source's outlines.
    explained = cv2.dilate(explained.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    gap = cv2.morphologyEx((figure & ~explained).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    gap = (cv2.morphologyEx(gap, cv2.MORPH_CLOSE, np.ones((STRAND, STRAND), np.uint8)) > 0) & figure
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(gap.astype(np.uint8), connectivity=8)
    floor = GAP_MIN_SHARE * figure.sum()
    found = []
    for k in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1:
        if stats[k, cv2.CC_STAT_AREA] < floor:
            break
        cx, cy = centroids[k]
        if head_xyxy is not None and head_xyxy[0] <= cx < head_xyxy[2] and head_xyxy[1] <= cy < head_xyxy[3]:
            continue
        found.append(labels == k)
    return found


def best_tag(gap: np.ndarray, drawn: dict):
    '''Of {tag: RGBA on the canvas}, the tag that draws the largest share of the gap, and that share.'''
    area = max(1, int(gap.sum()))
    best, share = None, 0.0
    for tag, picture in drawn.items():
        s = float((gap & (picture[..., 3] > 128)).sum()) / area
        if s > share:
            best, share = tag, s
    return best, share


def take(part: np.ndarray, drawn: np.ndarray, gap: np.ndarray) -> np.ndarray:
    '''
    part (RGBA) with what drawn draws of the gap: its pieces that reach into
    the gap, whole (the parts the gap's pass found hidden too), where they
    are more opaque than part is. Elsewhere part is as it was.
    '''
    pieces = drawn[..., 3] > DRAWN
    n, labels = cv2.connectedComponents(pieces.astype(np.uint8), connectivity=8)
    reach = np.unique(labels[gap & pieces])
    keep = np.isin(labels, reach[reach > 0])
    over = keep & (drawn[..., 3] > part[..., 3])
    out = part.copy()
    out[over] = drawn[over]
    return out


def place_tag(gap: np.ndarray, parts: dict) -> str:
    '''
    The part a gap no pass explains is of, by its place: below the lower
    garment the legs', below the top's middle the lower garment's, else an
    object's.
    '''
    def rows(tag):
        if tag not in parts:
            return np.zeros(0, int)
        return np.nonzero(parts[tag][..., 3].max(axis=1) > DRAWN)[0]

    cy = np.nonzero(gap)[0].mean()
    lower, top = rows('bottomwear'), rows('topwear')
    if len(lower) and cy > lower.max():
        return 'legwear'
    if len(top) and cy >= (top.min() + top.max()) / 2:
        return 'bottomwear'
    return 'objects'


def fill_from_source(part: np.ndarray, page: np.ndarray, gap: np.ndarray, figure: np.ndarray) -> np.ndarray:
    '''
    part (RGBA) with the source's own pixels over the gap, the holes inside it
    (where another part's hidden side happens to share its colours, as a
    sleeve's print does a hakama's) and its rim, within the figure.
    '''
    whole = gap.astype(np.uint8)
    contours, _ = cv2.findContours(whole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(whole, contours, -1, 1, thickness=cv2.FILLED)
    area = (cv2.dilate(whole, np.ones((5, 5), np.uint8)) > 0) & figure
    over = area & (page[..., 3] > part[..., 3])
    out = part.copy()
    out[over] = page[over]
    return out
