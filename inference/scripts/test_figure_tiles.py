'''
figure_tiles.stitch on made-up decompositions: python test_figure_tiles.py
'''
import os.path as osp
import sys
import tempfile

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import numpy as np
from PIL import Image
from psd_tools import PSDImage

import figure_tiles

H, W = 160, 100
NAVY = (30, 40, 110)
GREY = (150, 150, 150)


def psd(path, layers, size=(W, H)):
    '''A PSD of W x H with each (name, (x, y, w, h), rgb[, (x0, x1) left out]) a layer, back to front.'''
    doc = PSDImage.new(mode='RGBA', size=size, depth=8)
    for name, (x, y, w, h), rgb, *gap in layers:
        image = np.zeros((h, w, 4), np.uint8)
        image[...] = rgb + (255,)
        if gap:
            image[:, gap[0][0] - x: gap[0][1] - x] = 0
        doc.create_pixel_layer(Image.fromarray(image), name=name, top=y, left=x, opacity=255)
    doc.save(path)
    return path


def stitched(path):
    out = PSDImage.open(path)
    canvas = {}
    for layer in out.descendants():
        image = np.zeros((out.height, out.width, 4), np.uint8)
        pixels = np.array(layer.topil().convert('RGBA'))
        image[layer.top: layer.top + pixels.shape[0], layer.left: layer.left + pixels.shape[1]] = pixels
        canvas[layer.name] = image
    return canvas


def test_the_whole_stands_where_the_tile_left_a_panel_out_and_the_hair_behind_showed():
    picture = np.full((H, W, 3), 255, np.uint8)
    picture[20: 150, 10: 90] = GREY    # the hair behind, wider than the skirt
    picture[40: 150, 25: 75] = NAVY    # the skirt
    with tempfile.TemporaryDirectory() as tmp:
        whole = psd(osp.join(tmp, 'whole.psd'), [('back hair', (10, 20, 80, 130), GREY), ('bottomwear', (25, 40, 50, 110), NAVY)])
        # The tile drew the hair behind it all, and the skirt but for its middle panel.
        tile = psd(osp.join(tmp, 'tile.psd'), [('back hair', (10, 20, 80, 130), GREY),
                                               ('bottomwear', (25, 40, 50, 110), NAVY, (42, 58))])
        sources = (figure_tiles.Source(whole, [0, 0, W, H], (H, W), 'whole'),
                   [figure_tiles.Source(tile, [0, 0, W, H], (H, W), 'middle')])
        lines = []
        without = osp.join(tmp, 'without.psd')
        figure_tiles.stitch(*sources, (H, W), without, lines.append)
        assert stitched(without)['bottomwear'][90, 50, 3] == 0, 'without the picture the tile stands'
        out = osp.join(tmp, 'out.psd')
        figure_tiles.stitch(*sources, (H, W), out, lines.append, picture)
        layers = stitched(out)
        assert layers['bottomwear'][90, 50, 3] == 255, lines
        assert tuple(layers['bottomwear'][90, 50, :3]) == NAVY
        assert layers['bottomwear'][90, 30, 3] == 255, 'where the tile drew it, it stays'
        assert any('stands' in line for line in lines)


def test_a_tile_that_shows_the_picture_right_is_kept():
    picture = np.full((H, W, 3), 255, np.uint8)
    picture[40: 150, 25: 75] = NAVY
    with tempfile.TemporaryDirectory() as tmp:
        whole = psd(osp.join(tmp, 'whole.psd'), [('bottomwear', (25, 40, 50, 110), (60, 60, 60))])
        tile = psd(osp.join(tmp, 'tile.psd'), [('bottomwear', (25, 40, 50, 110), NAVY)])
        lines = []
        out = osp.join(tmp, 'out.psd')
        figure_tiles.stitch(figure_tiles.Source(whole, [0, 0, W, H], (H, W), 'whole'),
                            [figure_tiles.Source(tile, [0, 0, W, H], (H, W), 'middle')], (H, W), out, lines.append, picture)
        assert tuple(stitched(out)['bottomwear'][90, 50, :3]) == NAVY
        assert not any('stands' in line for line in lines)


SKIN = (240, 210, 200)


def test_a_tile_neck_drawn_on_under_the_collar_is_the_whole_neck():
    with tempfile.TemporaryDirectory() as tmp:
        # The whole's neck goes up behind the chin and stops at the collar; the tile's goes on down
        # under the collar and not up behind the chin: little IoU, but it covers what of it shows.
        whole = psd(osp.join(tmp, 'whole.psd'), [('neck', (40, 20, 20, 40), SKIN), ('face', (30, 0, 40, 35), SKIN)])
        tile = psd(osp.join(tmp, 'tile.psd'), [('neck', (38, 32, 24, 70), SKIN), ('face', (30, 0, 40, 35), SKIN)])
        aside = psd(osp.join(tmp, 'aside.psd'), [('neck', (70, 40, 20, 60), SKIN), ('face', (30, 0, 40, 35), SKIN)])
        source = figure_tiles.Source(whole, [0, 0, W, H], (H, W), 'whole')
        lines = []
        matches = figure_tiles.match_tile(source, figure_tiles.Source(tile, [0, 0, W, H], (H, W), 'upper'),
                                          ['neck', 'face'], (H, W), lines.append)
        assert matches.get('neck') == [('neck', None)], (matches, lines)
        assert any('covers 1.00' in line for line in lines), lines
        # A tile's neck beside the whole's is another thing of the same name.
        matches = figure_tiles.match_tile(source, figure_tiles.Source(aside, [0, 0, W, H], (H, W), 'upper'),
                                          ['neck', 'face'], (H, W), lines.append)
        assert 'neck' not in matches, matches


DARK = (40, 40, 50)
WHITE = (235, 235, 235)


def test_trousers_the_whole_put_in_the_top_are_a_layer_of_their_own():
    with tempfile.TemporaryDirectory() as tmp:
        size, hw = (160, 320), (320, 160)
        # The whole saw the trousers and shoes as the top, down to the feet.
        whole = psd(osp.join(tmp, 'whole.psd'), [('handwear-r', (20, 60, 20, 120), SKIN),
                                                 ('topwear', (50, 40, 60, 260), NAVY)], size)
        # The waist tile has the trousers, and the arm twice (once as legwear).
        middle = psd(osp.join(tmp, 'middle.psd'), [('handwear-r', (20, 60, 20, 120), SKIN),
                                                   ('legwear', (20, 60, 20, 120), SKIN),
                                                   ('bottomwear', (50, 120, 60, 80), DARK),
                                                   ('topwear', (50, 40, 60, 80), NAVY)], (160, 200))
        # The legs tile calls the trousers the top again; the shoes under them are shoes.
        lower = psd(osp.join(tmp, 'lower.psd'), [('footwear', (50, 120, 60, 20), WHITE),
                                                 ('topwear', (50, 0, 60, 120), DARK)], (160, 160))
        sources = [figure_tiles.Source(middle, [0, 0, 160, 200], (200, 160), 'middle'),
                   figure_tiles.Source(lower, [0, 160, 160, 320], (160, 160), 'lower')]
        lines = []
        out = osp.join(tmp, 'out.psd')
        figure_tiles.stitch(figure_tiles.Source(whole, [0, 0, 160, 320], hw, 'whole'), sources, hw, out, lines.append)
        layers = stitched(out)
        order = [layer.name for layer in PSDImage.open(out).descendants()]
        assert 'legwear' not in layers, lines
        assert order.index('footwear') < order.index('bottomwear') < order.index('topwear'), order
        for y in (130, 250):
            assert layers['bottomwear'][y, 80, 3] == 255, (y, lines)
            assert tuple(layers['bottomwear'][y, 80, :3]) == DARK
            assert layers['topwear'][y, 80, 3] == 0, ('the trousers are out of the top', y)
        assert tuple(layers['topwear'][80, 80]) == NAVY + (255,)
        assert tuple(layers['footwear'][290, 80]) == WHITE + (255,)
        assert any('carries on middle' in line for line in lines), lines


if __name__ == '__main__':
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
            print('ok', name)
