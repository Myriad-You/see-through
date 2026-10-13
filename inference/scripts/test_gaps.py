'''
gaps.py on made-up pictures, and on kept decompositions when their paths are
given: python test_gaps.py [front.psd master.png expected-gaps ...]
'''
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'common'))

import numpy as np

from utils import gaps

H, W = 200, 120
BACKDROP = (250, 250, 250, 255)
PLACEMENT = (1.0, 0, 0, W, H)


def picture(boxes):
    '''An RGBA canvas with each (x, y, w, h, rgba) painted on a clear one.'''
    out = np.zeros((H, W, 4), np.uint8)
    for x, y, w, h, rgba in boxes:
        out[y: y + h, x: x + w] = rgba
    return out


def source(boxes):
    page = np.zeros((H, W, 4), np.uint8)
    page[...] = BACKDROP
    for x, y, w, h, rgba in boxes:
        page[y: y + h, x: x + w] = rgba
    return page


HEAD = (40, 10, 40, 40, (236, 200, 180, 255))
TOP = (30, 50, 60, 60, (60, 110, 70, 255))
SKIRT = (30, 110, 60, 50, (120, 60, 160, 255))


def test_the_figure_is_what_is_not_the_backdrop():
    page = source([HEAD, TOP])
    figure = gaps.figure_mask(page, PLACEMENT)
    assert figure[30, 60] and figure[80, 60] and not figure[190, 5]
    clear = page.copy()
    clear[..., 3] = np.where(figure, 255, 0)
    assert (gaps.figure_mask(clear, PLACEMENT) == figure).all(), 'a transparent backdrop is read from alpha'


def test_a_dropped_skirt_is_the_one_gap_and_goes_below_the_top():
    page = source([HEAD, TOP, SKIRT])
    figure = gaps.figure_mask(page, PLACEMENT)
    parts = {'head': picture([HEAD]), 'topwear': picture([TOP])}
    found = gaps.find_gaps(figure, page, parts.values(), head_xyxy=(40, 10, 80, 50))
    assert len(found) == 1
    assert found[0][130, 60] and not found[0][80, 60]
    assert gaps.place_tag(found[0], parts) == 'bottomwear'
    filled = gaps.fill_from_source(np.zeros((H, W, 4), np.uint8), page, found[0], figure)
    assert tuple(filled[130, 60]) == SKIRT[4] and filled[190, 5, 3] == 0, 'the skirt, not the backdrop around it'


def test_a_part_drawn_behind_the_missing_one_does_not_hide_the_gap():
    page = source([HEAD, TOP, SKIRT])
    figure = gaps.figure_mask(page, PLACEMENT)
    hair = (20, 40, 80, 130, (200, 220, 250, 255))
    parts = [picture([HEAD]), picture([TOP]), picture([hair])]
    found = gaps.find_gaps(figure, page, parts, head_xyxy=(40, 10, 80, 50))
    assert len(found) == 1 and found[0][130, 60]


def test_nothing_is_a_gap_when_every_part_was_drawn():
    page = source([HEAD, TOP, SKIRT])
    figure = gaps.figure_mask(page, PLACEMENT)
    parts = [picture([HEAD]), picture([TOP]), picture([SKIRT])]
    assert gaps.find_gaps(figure, page, parts) == []


def test_a_gap_in_the_head_box_is_the_head_pass_s():
    page = source([HEAD, TOP, SKIRT])
    figure = gaps.figure_mask(page, PLACEMENT)
    parts = [picture([TOP]), picture([SKIRT])]
    assert gaps.find_gaps(figure, page, parts, head_xyxy=(40, 10, 80, 50)) == []
    assert len(gaps.find_gaps(figure, page, parts)) == 1


def test_the_tag_that_draws_the_gap_takes_it_whole_hidden_parts_too():
    gap = picture([SKIRT])[..., 3] > 0
    gap[110:120] = False  # under the top's hem at rest
    drawn = {
        'bottomwear': picture([SKIRT, (30, 100, 60, 10, (120, 60, 160, 255))]),
        'legwear': picture([(40, 150, 40, 10, (236, 200, 180, 255))]),
        'objects': picture([(0, 0, 10, 10, (0, 0, 0, 255))]),
    }
    tag, share = gaps.best_tag(gap, drawn)
    assert tag == 'bottomwear' and share > 0.9
    taken = gaps.take(np.zeros((H, W, 4), np.uint8), drawn['bottomwear'], gap)
    assert taken[105, 60, 3] == 255, 'the part hidden under the top comes with it'
    assert taken[5, 5, 3] == 0


def test_below_the_lower_garment_a_gap_is_the_legs():
    gap = picture([(40, 170, 40, 20, (1, 1, 1, 255))])[..., 3] > 0
    parts = {'topwear': picture([TOP]), 'bottomwear': picture([(30, 110, 60, 40, (120, 60, 160, 255))])}
    assert gaps.place_tag(gap, parts) == 'legwear'
    assert gaps.place_tag(gap, {'topwear': picture([TOP])}) == 'bottomwear'
    assert gaps.place_tag(picture([(0, 0, 5, 5, (1, 1, 1, 255))])[..., 3] > 0, {'topwear': picture([TOP])}) == 'objects'


def test_a_lifted_skirt_is_drawn_in_behind_the_sleeve_over_it_not_behind_the_hair_behind_it():
    import cv2
    skirt = (30, 110, 60, 50, (120, 60, 160, 255))
    # The sleeve hangs over the skirt's side above its hem, which shows below it.
    sleeve = (30, 120, 12, 30, (240, 200, 230, 255))
    hair = (80, 120, 10, 40, (200, 220, 250, 255))
    page = source([HEAD, TOP, skirt, sleeve, hair])
    # Lifted from the source: the skirt but where the sleeve and the hair show.
    garment = picture([skirt])
    garment[120:150, 30:42] = 0
    garment[120:160, 80:90] = 0
    parts = {'handwear': picture([sleeve]), 'back hair': picture([hair])}
    behind = gaps.hidden_side(garment, 'bottomwear', page, parts)
    assert behind[140, 35] and not behind[140, 85], 'under the sleeve, not under the back hair'
    done = gaps.complete_hidden(garment, behind, lambda rgb, mask: cv2.inpaint(rgb, mask, 3, cv2.INPAINT_TELEA))
    assert done[140, 35, 3] == 255 and np.linalg.norm(done[140, 35, :3].astype(float) - (120, 60, 160)) < 40
    assert done[140, 85, 3] == 0 and done[5, 5, 3] == 0


def kept(psd_path, master_path, expected):
    '''A kept front decomposition against its master, fitted as the canvas path fits it.'''
    import cv2
    from PIL import Image
    from psd_tools import PSDImage
    psd = PSDImage.open(psd_path)
    ch, cw = psd.height, psd.width
    master = np.array(Image.open(master_path).convert('RGBA'))
    scale = min(cw / master.shape[1], ch / master.shape[0])
    w, h = round(master.shape[1] * scale), round(master.shape[0] * scale)
    x, y = (cw - w) // 2, (ch - h) // 2
    page = np.zeros((ch, cw, 4), np.uint8)
    page[y: y + h, x: x + w] = cv2.resize(master, (w, h), interpolation=cv2.INTER_AREA)
    parts = {}
    for layer in psd.descendants():
        if layer.is_group():
            continue
        canvas = np.zeros((ch, cw, 4), np.uint8)
        image = np.array(layer.topil().convert('RGBA'))
        canvas[layer.top: layer.top + image.shape[0], layer.left: layer.left + image.shape[1]] = image
        parts[layer.name] = canvas
    figure = gaps.figure_mask(page, (scale, x, y, w, h))
    face = parts.get('face')
    head = None
    if face is not None:
        ys, xs = np.nonzero(face[..., 3] > 40)
        head = (xs.min(), ys.min(), xs.max(), ys.max())
    found = gaps.find_gaps(figure, page, parts.values(), head)
    tags = [gaps.place_tag(gap, parts) for gap in found]
    print(f'{osp.basename(osp.dirname(psd_path))}: {len(found)} gaps {[int(g.sum()) for g in found]} -> {tags} (figure {int(figure.sum())} px)')
    assert len(found) == expected, f'expected {expected}'


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for test in tests:
        test()
        print('ok', test.__name__)
    args = sys.argv[1:]
    for i in range(0, len(args), 3):
        kept(args[i], args[i + 1], int(args[i + 2]))
