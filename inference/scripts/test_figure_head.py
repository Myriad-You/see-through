'''
figure_head.take_hair on made-up decompositions: python test_figure_head.py
'''
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import numpy as np

import figure_head

# A head canvas the crop fills (figure_head.BUST_PIXELS at 1/32), placed on
# the figure at x 20..56, y 0..48.
HEAD_W, HEAD_H = 36, 48
FACTOR, OFFSET = 1.0, (20.0, 0.0)
FIGURE_W, FIGURE_H = 80, 100
BROWN = (0.5, 0.3, 0.2)
DARK = (0.3, 0.2, 0.1)
LIGHT = (0.8, 0.6, 0.4)


class Made:
    def __init__(self, W, H, layers):
        self.W, self.H = W, H
        self.order = [name for name, _ in layers]
        self.layers = dict(layers)


def layer(W, H, box, rgb):
    out = np.zeros((H, W, 4), np.float32)
    x0, y0, x1, y1 = box
    out[y0:y1, x0:x1] = rgb + (1.0,)
    return out


def figure_and_head():
    figure = Made(FIGURE_W, FIGURE_H, [
        ('back hair', layer(FIGURE_W, FIGURE_H, (10, 10, 70, 90), BROWN)),
        ('face', layer(FIGURE_W, FIGURE_H, (30, 10, 46, 30), LIGHT)),
        ('front hair', layer(FIGURE_W, FIGURE_H, (28, 5, 48, 15), BROWN)),
    ])
    head = Made(HEAD_W, HEAD_H, [
        ('back hair', layer(HEAD_W, HEAD_H, (0, 10, 36, 48), DARK)),
        ('face', layer(HEAD_W, HEAD_H, (10, 10, 26, 30), LIGHT)),
        ('front hair-1', layer(HEAD_W, HEAD_H, (8, 5, 18, 48), DARK)),
        ('front hair-2', layer(HEAD_W, HEAD_H, (18, 5, 28, 15), DARK)),
    ])
    return figure, head


def test_the_long_hair_past_the_heads_crop_is_the_figures_own():
    figure, head = figure_and_head()
    taken = figure_head.take_hair(figure, head, FACTOR, OFFSET)
    assert 'back hair' in taken
    back = figure.layers['back hair']
    assert tuple(back[30, 40, :3]) == DARK, 'within the crop, the head\'s'
    assert tuple(back[30, 12, :3]) == BROWN and back[30, 12, 3] == 1, 'past its side, the figure\'s'
    assert tuple(back[70, 40, :3]) == BROWN and back[70, 40, 3] == 1, 'below it, the figure\'s'
    assert back[5, 40, 3] == 0, 'within the crop where the head drew none, none'


def test_front_hair_past_the_crop_joins_the_lock_it_continues():
    figure, head = figure_and_head()
    # A long lock of the figure's falling at x 30..34 down past the crop's
    # bottom, where the head's first lock (x 28..38) ends.
    figure.layers['front hair'][15:70, 30:34] = BROWN + (1.0,)
    figure_head.take_hair(figure, head, FACTOR, OFFSET)
    assert figure.layers['front hair-1'][60, 32, 3] == 1, 'into the lock above it'
    assert figure.layers['front hair-2'][60, 32, 3] == 0, 'not the topmost'
    assert 'front hair' not in figure.layers


if __name__ == '__main__':
    for name, test in list(globals().items()):
        if name.startswith('test_'):
            test()
            print('ok', name)
