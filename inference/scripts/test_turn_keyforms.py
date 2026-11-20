'''
turn_keyforms.restore_ornaments on made-up decompositions: python test_turn_keyforms.py
'''
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import numpy as np

import turn_keyforms

W, H = 120, 100
LAVENDER = (0.7, 0.65, 0.8)
PINK = (0.9, 0.4, 0.6)
SKIN = (1.0, 0.85, 0.75)
EYE = (0.5, 0.2, 0.6)
SHIRT = (0.2, 0.3, 0.6)
GOLD = (0.9, 0.7, 0.2)
HAIR = (0.85, 0.8, 0.9)


class Made:
    def __init__(self, layers):
        self.W, self.H = W, H
        self.order = [name for name, _ in layers]
        self.layers = dict(layers)


def layer(boxes):
    out = np.zeros((H, W, 4), np.float32)
    for (x0, y0, x1, y1), rgb in boxes:
        out[y0:y1, x0:x1] = rgb + (1.0,)
    return out


def paint(picture, box, rgb):
    x0, y0, x1, y1 = box
    picture[y0:y1, x0:x1] = rgb


def quiet(_):
    pass


def hat_over_face():
    """
    A hat's brim (pink, x 10..110, y 10..70) with its opening drawn whole and
    flat behind the face (lavender, x 30..90, y 20..70); the decomposition drew
    the face without its eye, which the picture has at x 50..60, y 40..48.
    """
    hat = layer([((10, 10, 110, 70), PINK), ((30, 20, 90, 70), LAVENDER)])
    face = layer([((30, 20, 90, 70), SKIN)])
    picture = np.ones((H, W, 3), np.float32)
    paint(picture, (10, 10, 110, 70), PINK)
    paint(picture, (30, 20, 90, 70), SKIN)
    paint(picture, (50, 40, 60, 48), EYE)
    return Made([('headwear', hat), ('face', face)]), picture


def test_an_eye_the_face_was_drawn_without_is_not_taken_into_the_hat():
    front, picture = hat_over_face()
    turn_keyforms.restore_ornaments(front, picture, quiet)
    hat = front.layers['headwear']
    assert hat[44, 55, 3] == 0, 'the eye is not the hat\'s'
    assert hat[30, 40, 3] == 0, 'the opening hidden under the face goes'
    assert hat[15, 20, 3] == 1 and np.allclose(hat[15, 20, :3], PINK), 'the brim stays'


def test_the_brim_between_the_locks_is_the_hats():
    front, picture = hat_over_face()
    # Locks over the brim's edge with a gap between them that the picture fills with brim.
    hair = layer([((20, 30, 25, 70), HAIR), ((29, 30, 35, 70), HAIR)])
    front.layers['front hair'] = hair
    front.order.append('front hair')
    paint(picture, (20, 30, 25, 70), HAIR)
    paint(picture, (29, 30, 35, 70), HAIR)
    turn_keyforms.restore_ornaments(front, picture, quiet)
    hat = front.layers['headwear']
    assert hat[50, 27, 3] == 1 and np.allclose(hat[50, 27, :3], PINK), 'the brim seen between the locks'
    assert hat[44, 55, 3] == 0


def test_a_necklace_put_under_the_shirt_is_still_as_the_picture_has_it():
    # The decomposition put the necklace under the shirt; the picture has it on top.
    necklace = layer([((40, 50, 80, 56), GOLD)])
    shirt = layer([((20, 40, 100, 100), SHIRT)])
    picture = np.ones((H, W, 3), np.float32)
    paint(picture, (20, 40, 100, 100), SHIRT)
    paint(picture, (40, 50, 80, 56), GOLD)
    front = Made([('neckwear', necklace), ('topwear', shirt)])
    turn_keyforms.restore_ornaments(front, picture, quiet)
    neckwear = front.layers['neckwear']
    assert neckwear[53, 60, 3] == 1 and np.allclose(neckwear[53, 60, :3], GOLD)


# A face (x 60..240, y 40..260) on a larger canvas, its features where they are at rest.
FW, FH = 300, 300
FEATURES = {
    'eyewhite-r': (100, 120, 130, 130),
    'eyewhite-l': (170, 120, 200, 130),
    'nose': (145, 160, 155, 170),
    'mouth': (135, 200, 165, 206),
}


def face_with_features(dx=0, dy=0):
    out = Made([])
    out.W, out.H = FW, FH

    def put(name, box, rgb):
        x0, y0, x1, y1 = box
        canvas = np.zeros((FH, FW, 4), np.float32)
        canvas[y0:y1, x0:x1] = rgb + (1.0,)
        out.order.append(name)
        out.layers[name] = canvas

    put('face', (60, 40, 240, 260), SKIN)
    for name, (x0, y0, x1, y1) in FEATURES.items():
        put(name, (x0 + dx, y0 + dy, x1 + dx, y1 + dy), EYE)
    return out


def shifted(dx, dy, box=(40, 20, 260, 280), grid=13):
    """A lattice moving all it covers by (dx, dy)."""
    return dict(box=[float(v) for v in box], grid=grid, back=[float(v) for _ in range(grid * grid) for v in (-dx, -dy)])


def face_keys(side, dx, dy):
    # The eyes keyed where the turned drawing has them; the face, fitted on its outline alone, not moved.
    return {'face': {side: shifted(0, 0)}, 'eye:L': {side: shifted(dx, dy)}, 'eye:R': {side: shifted(dx, dy)}}


def face_picture(chin):
    """Skin over white with the jaw's line at row `chin`."""
    picture = np.ones((FH, FW, 3), np.float32)
    paint(picture, (60, 40, 240, chin), SKIN)
    paint(picture, (60, chin, 240, chin + 3), EYE)
    return picture


def test_a_raised_face_carries_its_features_and_chin_up():
    front, turned = face_with_features(), face_with_features(dy=-30)
    keys = face_keys('up', 0, -30)
    turn_keyforms.hold_face_to_features(front, turned, keys, 'up', quiet, (face_picture(260), face_picture(230)))
    key = keys['face']['up']
    mx, my = turn_keyforms.turned_position(key, np.float32([150]), np.float32([203]))
    assert abs(mx[0] - 150) < 3 and abs(my[0] - 173) < 6, (mx, my)
    drawn = turn_keyforms.remap(front.layers['face'], *turn_keyforms.lattice_maps(key, FW, FH))[..., 3]
    bottom = np.nonzero(drawn[:, 150] > 0.5)[0].max()
    assert bottom < 245, f'the chin goes up with the mouth: {bottom}'
    top = np.nonzero(drawn[:, 150] > 0.5)[0].min()
    assert abs(top - 40) <= 2, f'the crown holds: {top}'


def test_a_raised_face_whose_chin_stays_keeps_its_jaw():
    # The picture has the jaw where it was: rising with the mouth, it would bare what is behind it.
    front, turned = face_with_features(), face_with_features(dy=-30)
    keys = face_keys('up', 0, -30)
    turn_keyforms.hold_face_to_features(front, turned, keys, 'up', quiet, (face_picture(260), face_picture(260)))
    key = keys['face']['up']
    _, my = turn_keyforms.turned_position(key, np.float32([150]), np.float32([203]))
    assert my[0] < 190, my
    drawn = turn_keyforms.remap(front.layers['face'], *turn_keyforms.lattice_maps(key, FW, FH))[..., 3]
    assert np.nonzero(drawn[:, 150] > 0.5)[0].max() >= 257


def pointed(face, picture, chin, side):
    """The face (and the picture's jaw line) cut to a chin at row `chin`, rising by `side` toward x 60 and 240."""
    for x in range(60, 240):
        y = int(round(chin - side * abs(x - 150) / 90))
        if face is not None:
            face[y:, x] = 0
        picture[y:, x] = 1.0
        picture[y:y + 3, x] = EYE


def test_a_raised_face_takes_the_jaws_line_from_the_picture():
    # A pointed chin seen from below: the jaw's line flattens, the sides rising less than the chin.
    front, turned = face_with_features(), face_with_features(dy=-30)
    seen, there = face_picture(260), face_picture(260)
    pointed(front.layers['face'], seen, 260, 30)
    pointed(None, there, 230, 0)
    keys = face_keys('up', 0, -30)
    turn_keyforms.hold_face_to_features(front, turned, keys, 'up', quiet, (seen, there))
    drawn = turn_keyforms.remap(front.layers['face'], *turn_keyforms.lattice_maps(keys['face']['up'], FW, FH))[..., 3] > 0.5
    for x in (115, 150, 185):
        bottom = np.nonzero(drawn[:, x])[0].max()
        assert abs(bottom - 230) <= 6, f'the jaw at x {x} is where the picture has it: {bottom}'


def test_a_raised_face_takes_the_jaws_line_where_its_chin_shows_none():
    # The turned picture's line is lit away under the mouth: no chin there, the rest of the line still says where the jaw is.
    front, turned = face_with_features(), face_with_features(dy=-30)
    seen, there = face_picture(260), face_picture(260)
    pointed(front.layers['face'], seen, 260, 30)
    pointed(None, there, 230, 0)
    there[230:233, 138:163] = 1.0
    keys = face_keys('up', 0, -30)
    turn_keyforms.hold_face_to_features(front, turned, keys, 'up', quiet, (seen, there))
    drawn = turn_keyforms.remap(front.layers['face'], *turn_keyforms.lattice_maps(keys['face']['up'], FW, FH))[..., 3] > 0.5
    for x in (100, 115, 185, 200):
        bottom = np.nonzero(drawn[:, x])[0].max()
        assert abs(bottom - 230) <= 6, f'the jaw at x {x} is where the picture has it: {bottom}'


def test_a_turned_face_is_left_as_fitted():
    # A turn's silhouette is the decomposition's; the face key stays as fitted.
    front, turned = face_with_features(), face_with_features(dx=20)
    keys = face_keys('plus', 20, 0)
    fitted = list(keys['face']['plus']['back'])
    assert turn_keyforms.hold_face_to_features(front, turned, keys, 'plus', quiet) is None
    assert keys['face']['plus']['back'] == fitted


def test_a_clip_lifted_over_the_hairs_edge_is_not_filled_black_under():
    # A clip hanging past the lock's edge (gold, x 50..75) the decomposition put under
    # the lock (lavender, x 20..60): lifted over it, the lock is filled from itself, not
    # from the transparent black past its edge.
    clip = layer([((50, 40, 75, 60), GOLD)])
    hair = layer([((20, 10, 60, 90), HAIR)])
    front = Made([('headwear', clip), ('front hair-1', hair)])
    cut = np.zeros((H, W), bool)
    cut[40:60, 50:60] = True
    _, below, _ = turn_keyforms.composite_of(front, ['headwear'])
    turn_keyforms.lift_ornaments(front, 'front hair-1', cut, below, quiet)
    assert front.order == ['front hair-1', 'headwear'], front.order
    lock = front.layers['front hair-1']
    drawn = lock[..., 3] > 0.5
    darkest = lock[..., :3][drawn].sum(-1).min()
    assert darkest > 0.9 * sum(HAIR), f'the lock under the clip is darkened: {darkest:.2f}'


def flat_key(dx, dy, g=17):
    return {'box': [0, 0, W - 1, H - 1], 'grid': g, 'back': [float(v) for _ in range(g * g) for v in (-dx, -dy)]}


def tails_beside_face(ties=2):
    """
    A face (x 40..80, y 10..60) with rear hair over the crown and its sides, and a tail hanging
    beside it each side (x 10..35 and 85..110, y 30..95), a ribbon (y 30..40) tied at each top.
    """
    face = layer([((40, 10, 80, 60), SKIN)])
    hair = layer([((10, 5, 110, 40), HAIR), ((10, 30, 35, 95), HAIR), ((85, 30, 110, 95), HAIR)])
    made = [('back hair', hair), ('face', face)]
    ribbons = [((12, 30, 33, 40), GOLD), ((87, 30, 108, 40), GOLD)][:ties]
    if ribbons:
        made.append(('headwear', layer(ribbons)))
    front = Made(made)
    keys = {'face': {s: flat_key(20 if s == 'plus' else -20, 0) for s in ('plus', 'minus')},
            'back-hair': {s: flat_key(0, 0) for s in ('plus', 'minus')},
            'headwear': {s: flat_key(0, 0) for s in ('plus', 'minus')}}
    return front, keys


def test_a_twin_tail_tied_beside_the_face_goes_along_with_the_head():
    front, keys = tails_beside_face()
    turn_keyforms.carry_hanging_hair(front, keys, quiet)
    for side, way in (('plus', 1), ('minus', -1)):
        key = keys['back-hair'][side]
        tx, _ = turn_keyforms.turned_position(key, np.array([22.0, 60.0]), np.array([80.0, 8.0]))
        assert abs(tx[0] - 22 - way * 20 * turn_keyforms.HANGING_CARRY) < 2, f'the tail goes with the head: {tx[0] - 22:.1f}'
        assert abs(tx[1] - 60) < 1, f'the crown stays as fitted: {tx[1] - 60:.1f}'
        rx, _ = turn_keyforms.turned_position(keys['headwear'][side], np.array([22.0]), np.array([34.0]))
        assert abs(rx[0] - 22 - way * 20 * turn_keyforms.HANGING_CARRY) < 3, f'the ribbon goes with its tail: {rx[0] - 22:.1f}'


def test_hair_tied_on_one_side_or_none_stays_as_fitted():
    # One tie alone (a side braid, a bun over long hair) would take that side's hair along and not the other's.
    for ties in (0, 1):
        front, keys = tails_beside_face(ties)
        before = [list(keys['back-hair'][s]['back']) for s in ('plus', 'minus')]
        turn_keyforms.carry_hanging_hair(front, keys, quiet)
        assert [keys['back-hair'][s]['back'] for s in ('plus', 'minus')] == before, ties


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print(f'ok {name}')
