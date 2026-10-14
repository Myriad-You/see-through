'''
gaps.py on made-up pictures, and on kept decompositions when their paths are
given (their layers back to front, as the PSD has them):
python test_gaps.py [front.psd master.png expected-gaps ...]
'''
import os.path as osp
import sys

sys.path.insert(0, osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'common'))

import cv2
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
HAIR = (20, 40, 80, 130, (200, 220, 250, 255))
HEAD_BOX = (40, 10, 80, 50)


def test_the_figure_is_what_is_not_the_backdrop():
    page = source([HEAD, TOP])
    figure = gaps.figure_mask(page, PLACEMENT)
    assert figure[30, 60] and figure[80, 60] and not figure[190, 5]
    clear = page.copy()
    clear[..., 3] = np.where(figure, 255, 0)
    assert (gaps.figure_mask(clear, PLACEMENT) == figure).all(), 'a transparent backdrop is read from alpha'


def test_a_scene_behind_the_figure_is_not_looked_into():
    page = source([HEAD, TOP, SKIRT])
    yy, xx = np.mgrid[0:H, 0:W]
    scene = np.stack([128 + 100 * np.sin(xx / 3.0), 128 + 100 * np.cos(yy / 5.0), 128 + 100 * np.sin((xx + yy) / 4.0)], -1)
    clear = np.all(page[..., :3] == BACKDROP[:3], axis=-1)
    page[clear, :3] = scene[clear].astype(np.uint8)
    assert gaps.figure_mask(page, PLACEMENT) is None
    lines = []
    assert gaps.cover(['face', 'topwear'], [picture([HEAD]), picture([TOP])], page, PLACEMENT, HEAD_BOX, None, lines.append) is None
    assert 'no flat backdrop' in lines[0]


def test_a_gradient_behind_the_figure_is_backdrop_between_locks_too():
    page = source([HEAD, TOP, SKIRT])
    yy, xx = np.mgrid[0:H, 0:W]
    gradient = np.stack([250 - 90 * yy / H, 250 - 60 * yy / H, 250 - 20 * xx / W], -1).astype(np.uint8)
    clear = np.all(page[..., :3] == BACKDROP[:3], axis=-1)
    page[clear, :3] = gradient[clear]
    page[60:70, 50:60, :3] = gradient[60:70, 50:60]   # a pocket the figure surrounds
    figure = gaps.figure_mask(page, PLACEMENT)
    assert figure is not None and figure[80, 40] and figure[130, 60]
    assert not figure[195, 5] and not figure[195, 115] and not figure[65, 55]


def test_a_dropped_skirt_is_the_one_gap_and_goes_below_the_top():
    page = source([HEAD, TOP, SKIRT])
    figure = gaps.figure_mask(page, PLACEMENT)
    found = gaps.find_gaps(figure, page, [picture([HEAD]), picture([TOP])], HEAD_BOX)
    assert len(found) == 1
    assert found[0][130, 60] and not found[0][80, 60]
    assert gaps.place_tag(found[0], {'topwear': picture([TOP])}) == 'bottomwear'
    filled = gaps.fill_from_source(np.zeros((H, W, 4), np.uint8), page, found[0], figure)
    assert tuple(filled[130, 60]) == SKIRT[4] and filled[190, 5, 3] == 0, 'the skirt, not the backdrop around it'


def test_the_back_hair_drawn_behind_a_dropped_skirt_does_not_hide_it():
    page = source([HAIR, HEAD, TOP, SKIRT])
    figure = gaps.figure_mask(page, PLACEMENT)
    found = gaps.find_gaps(figure, page, [picture([HAIR]), picture([HEAD]), picture([TOP])], HEAD_BOX)
    assert len(found) == 1 and found[0][130, 60]


def test_a_part_drawn_behind_what_covers_it_is_buried_not_a_gap():
    # The skirt's colour drawn in a part behind the top shows nowhere: no gap there.
    page = source([HEAD, TOP, SKIRT])
    figure = gaps.figure_mask(page, PLACEMENT)
    behind = picture([(30, 110, 60, 50, (120, 60, 160, 255))])
    assert gaps.find_gaps(figure, page, [behind, picture([HEAD]), picture([TOP])], HEAD_BOX) == []
    covered = picture([(30, 110, 60, 50, (240, 240, 240, 255))])
    assert gaps.find_gaps(figure, page, [behind, picture([HEAD]), picture([TOP]), covered], HEAD_BOX) == [], \
        'drawn behind what covers it, it is buried, not left out: the importer reveals it'
    assert len(gaps.find_gaps(figure, page, [picture([HEAD]), picture([TOP]), covered], HEAD_BOX)) == 1, \
        'drawn by no part, what covers it shows something else: a gap'


def test_nothing_is_a_gap_when_every_part_was_drawn():
    page = source([HEAD, TOP, SKIRT])
    figure = gaps.figure_mask(page, PLACEMENT)
    assert gaps.find_gaps(figure, page, [picture([HEAD]), picture([TOP]), picture([SKIRT])]) == []


def test_a_gap_in_the_head_box_is_the_head_pass_s():
    page = source([HEAD, TOP, SKIRT])
    figure = gaps.figure_mask(page, PLACEMENT)
    parts = [picture([TOP]), picture([SKIRT])]
    assert gaps.find_gaps(figure, page, parts, HEAD_BOX) == []
    assert len(gaps.find_gaps(figure, page, parts)) == 1


def test_below_the_lower_garment_a_gap_is_the_legs():
    gap = picture([(40, 170, 40, 20, (1, 1, 1, 255))])[..., 3] > 0
    parts = {'topwear': picture([TOP]), 'bottomwear': picture([(30, 110, 60, 40, (120, 60, 160, 255))])}
    assert gaps.place_tag(gap, parts) == 'legwear'
    assert gaps.place_tag(gap, {'topwear': picture([TOP])}) == 'bottomwear'
    assert gaps.place_tag(picture([(0, 0, 5, 5, (1, 1, 1, 255))])[..., 3] > 0, {'topwear': picture([TOP])}) == 'objects'


def test_above_the_chin_a_gap_is_headwear_and_to_one_side_an_arm():
    parts = {'face': picture([HEAD]), 'topwear': picture([TOP])}
    flower = picture([(70, 5, 20, 15, (1, 1, 1, 255))])[..., 3] > 0
    assert gaps.place_tag(flower, parts) == 'headwear'
    left_arm = picture([(20, 60, 12, 50, (1, 1, 1, 255))])[..., 3] > 0
    right_arm = picture([(88, 60, 12, 50, (1, 1, 1, 255))])[..., 3] > 0
    assert gaps.place_tag(left_arm, parts) == 'handwear-r', 'See-through names the picture-left arm -r'
    assert gaps.place_tag(right_arm, parts) == 'handwear-l'
    skirt = picture([SKIRT])[..., 3] > 0
    assert gaps.place_tag(skirt, parts) == 'bottomwear', 'across the middle it is the lower garment'
    # A figure's lower tile: part of the skirt and the shoes, no face, no top.
    hem = {'bottomwear': picture([(30, 0, 60, 60, (120, 60, 160, 255))]), 'footwear': picture([(45, 150, 30, 20, (9, 9, 9, 255))])}
    assert gaps.place_tag(picture([(10, 20, 100, 90, (1, 1, 1, 255))])[..., 3] > 0, hem) == 'bottomwear'
    # Sleeves hanging beside it there are arms.
    assert gaps.place_tag(picture([(5, 0, 20, 70, (1, 1, 1, 255))])[..., 3] > 0, hem) == 'handwear-r'
    assert gaps.place_tag(picture([(95, 0, 20, 70, (1, 1, 1, 255))])[..., 3] > 0, hem) == 'handwear-l'


def test_a_skirt_drawn_in_part_is_grown_with_the_rest():
    page = source([HEAD, TOP, SKIRT])
    half = picture([(30, 110, 30, 50, SKIRT[4])])
    names = ['face', 'topwear', 'bottomwear']
    images = [picture([HEAD]), picture([TOP]), half]
    figure = gaps.figure_mask(page, PLACEMENT)
    found = gaps.find_gaps(figure, page, images, HEAD_BOX)
    assert len(found) == 1
    _, _, top = gaps.composite(images)
    assert gaps.carried_on(found[0], page, names, images, top) == 'bottomwear'
    lines = []
    out_names, out_images, changes = gaps.cover(names, images, page, PLACEMENT, HEAD_BOX, None, lines.append)
    assert changes[0][:2] == ('bottomwear', 'grown') and out_names == names
    assert tuple(out_images[2][130, 75]) == SKIRT[4]


def test_a_head_part_cut_at_the_head_pass_box_goes_on_from_the_source():
    # A white frill outlined in grey, on white, rising from its band above the box (y 40) the head pass drew.
    face = (45, 70, 30, 30, (236, 200, 180, 255))
    page = source([face])
    page[20: 40, 40: 80, :3] = 252
    page[40: 56, 40: 80, :3] = (200, 170, 190)
    page[40: 44, 41: 79, :3] = 252   # the frill's white runs down past the cut
    page[20, 40: 80, :3] = 90         # outlined above and at its sides, down into the box
    page[20: 56, 40, :3] = 90
    page[20: 56, 79, :3] = 90
    drawn = np.zeros((H, W, 4), np.uint8)
    drawn[40: 56, 40: 80] = page[40: 56, 40: 80]
    names, images = ['face', 'headwear'], [picture([face]), drawn]
    lines = []
    grown = gaps.extend_cut(names, images, page, PLACEMENT, (20, 40, 100, 120), lines.append)
    assert grown == ['headwear'], lines
    assert images[1][30, 60, 3] == 255, 'the white inside the outline goes on above the box'
    assert images[1][20, 60, 3] == 255, 'up to its outline'
    assert images[1][10, 60, 3] == 0 and images[1][30, 30, 3] == 0, 'not the backdrop outside it'
    assert gaps.extend_cut(['face'], [picture([face])], page, PLACEMENT, (20, 40, 100, 120), lines.append) == [], \
        'a part inside its box is not cut'


def test_a_sleeve_grown_behind_the_skirt_it_hangs_over_comes_in_front_of_it():
    sleeve_colour = (200, 230, 240, 255)
    # A wide sleeve hanging over the skirt's side, its top drawn, the rest left out.
    page = source([HEAD, TOP, SKIRT, (60, 60, 40, 80, sleeve_colour)])
    drawn_sleeve = picture([(60, 60, 40, 30, sleeve_colour)])
    # The skirt was drawn whole, hidden part and all, and the sleeve put behind it.
    names = ['handwear-l', 'face', 'topwear', 'bottomwear']
    images = [drawn_sleeve, picture([HEAD]), picture([TOP]), picture([SKIRT])]
    out_names, out_images, changes = gaps.cover(names, images, page, PLACEMENT, HEAD_BOX, None, lambda line: None)
    assert changes[0][0] == 'handwear-l' and 'brought in front of bottomwear' in changes[0][1], changes
    assert out_names.index('handwear-l') > out_names.index('bottomwear')
    color, _, _ = gaps.composite(out_images)
    assert np.linalg.norm(color[130, 80] - np.array(sleeve_colour[:3], np.float32)) < 1, 'the sleeve shows over the skirt'


def test_a_lifted_skirt_is_drawn_in_behind_what_hangs_in_front_only():
    garment = picture([SKIRT])
    garment[120:150, 30:42] = 0   # under a sleeve in front, above its hem, which shows
    garment[120:160, 80:90] = 0   # where a lock behind it shows: not its to draw
    in_front = picture([(30, 120, 12, 30, (1, 1, 1, 255))])[..., 3] > 0
    behind = gaps.hidden_side(garment, in_front)
    assert behind[140, 35] and not behind[140, 85]
    done = gaps.complete_hidden(garment, behind, lambda rgb, mask: cv2.inpaint(rgb, mask, 3, cv2.INPAINT_TELEA))
    assert done[140, 35, 3] == 255 and np.linalg.norm(done[140, 35, :3].astype(float) - (120, 60, 160)) < 40
    assert done[140, 85, 3] == 0 and done[5, 5, 3] == 0


def test_a_lock_over_the_sleeve_beside_a_waist_hides_the_sleeve_not_the_skirt():
    # A narrow waist over a wide skirt; the sleeve shows in the corner beside the waist.
    garment = picture([(30, 100, 40, 20, SKIRT[4]), (30, 120, 60, 40, SKIRT[4])])
    lock = picture([(60, 106, 26, 8, (1, 1, 1, 255))])[..., 3] > 0
    garment[lock] = 0
    behind = gaps.hidden_side(garment, lock)
    assert behind[110, 65], 'the waist goes on under the lock'
    assert not behind[110, 80], 'beside it the lock hides the sleeve'


def test_cover_gives_the_skirt_a_new_part_and_brings_the_sleeve_in_front_of_it():
    sleeve = (30, 120, 12, 30, (240, 200, 230, 255))
    # The skirt's band over the top's hem, as a hakama's over a kimono.
    page = source([HAIR, HEAD, TOP, (30, 100, 60, 60, SKIRT[4]), sleeve])
    # As that kimono bust had it: the sleeve put behind the top.
    names = ['back hair', 'handwear-l', 'face', 'topwear']
    images = [picture([HAIR]), picture([sleeve]), picture([HEAD]), picture([TOP])]
    lines = []
    out_names, out_images, changes = gaps.cover(
        names, images, page, PLACEMENT, HEAD_BOX,
        lambda rgb, mask: cv2.inpaint(rgb, mask, 3, cv2.INPAINT_TELEA), lines.append)
    assert changes == [('bottomwear', 'new, in front of topwear', ['handwear-l'])], lines
    assert out_names == ['back hair', 'face', 'topwear', 'bottomwear', 'handwear-l']
    skirt = out_images[3]
    assert tuple(skirt[130, 60]) == SKIRT[4]
    assert skirt[135, 35, 3] == 255, 'drawn in under the sleeve now in front of it'
    color, _, _ = gaps.composite(out_images)
    assert np.linalg.norm(color[135, 35] - np.array(sleeve[4][:3], np.float32)) < 1, 'the sleeve still shows at rest'


def part_dict(tag, image, median):
    '''A part as further_extr has it before the PSD: cropped to its box, with its depth.'''
    x, y, w, h = cv2.boundingRect(cv2.findNonZero((image[..., 3] > 10).astype(np.uint8)))
    return {'tag': tag, 'img': image[y: y + h, x: x + w], 'xyxy': [x, y, x + w, y + h],
            'depth': np.full((h, w), round(median * 255), np.uint8), 'depth_median': median}


def test_cover_parts_gives_the_new_part_and_the_one_brought_before_it_depths_in_that_order():
    sleeve = (30, 120, 12, 30, (240, 200, 230, 255))
    page = source([HAIR, HEAD, TOP, (30, 100, 60, 60, SKIRT[4]), sleeve])
    parts = [part_dict('topwear', picture([TOP]), 0.6), part_dict('back hair', picture([HAIR]), 0.9),
             part_dict('face', picture([HEAD]), 0.5), part_dict('handwear-l', picture([sleeve]), 0.7)]
    changes = gaps.cover_parts(parts, page, PLACEMENT, lambda rgb, mask: cv2.inpaint(rgb, mask, 3, cv2.INPAINT_TELEA),
                               lambda line: None)
    assert changes == [('bottomwear', 'new, in front of topwear', ['handwear-l'])]
    order = sorted(parts, key=lambda part: part['depth_median'], reverse=True)
    assert [part['tag'] for part in order] == ['back hair', 'topwear', 'bottomwear', 'handwear-l', 'face']
    by = {part['tag']: part for part in parts}
    assert by['back hair']['depth_median'] == 0.9 and by['face']['depth_median'] == 0.5, 'the others keep theirs'
    for part in parts:
        x1, y1, x2, y2 = part['xyxy']
        assert part['img'].shape == (y2 - y1, x2 - x1, 4) and part['depth'].shape == (y2 - y1, x2 - x1), part['tag']
    skirt = by['bottomwear']
    assert tuple(skirt['img'][130 - skirt['xyxy'][1], 60 - skirt['xyxy'][0]]) == SKIRT[4]


def test_cover_parts_keeps_the_order_by_rank_between_tied_depths():
    # The skirt's band over the top's hem: it goes in front of the top, tied with the face.
    page = source([HEAD, TOP, (30, 100, 60, 60, SKIRT[4])])
    parts = [part_dict('topwear', picture([TOP]), 0.6), part_dict('face', picture([HEAD]), 0.6)]
    gaps.cover_parts(parts, page, PLACEMENT, None, lambda line: None)
    order = [part['tag'] for part in sorted(parts, key=lambda part: part['depth_median'], reverse=True)]
    assert order == ['topwear', 'bottomwear', 'face'], order


def kept(psd_path, master_path, expected):
    '''A kept front decomposition (back to front) against its master, fitted as the canvas path fits it.'''
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
    names, images = [], []
    for layer in psd.descendants():
        if layer.is_group():
            continue
        canvas = np.zeros((ch, cw, 4), np.uint8)
        image = np.array(layer.topil().convert('RGBA'))
        canvas[layer.top: layer.top + image.shape[0], layer.left: layer.left + image.shape[1]] = image
        names.append(layer.name)
        images.append(canvas)
    face = images[names.index('face')] if 'face' in names else None
    head = None
    if face is not None:
        ys, xs = np.nonzero(face[..., 3] > 40)
        head = (xs.min(), ys.min(), xs.max(), ys.max())
    figure = gaps.figure_mask(page, (scale, x, y, w, h))
    found = gaps.find_gaps(figure, page, images, head)
    parts = dict(zip(names, images))
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
