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


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print(f'ok {name}')
