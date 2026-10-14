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


def psd(path, layers):
    '''A PSD of W x H with each (name, (x, y, w, h), rgb[, (x0, x1) left out]) a layer, back to front.'''
    doc = PSDImage.new(mode='RGBA', size=(W, H), depth=8)
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
        image = np.zeros((H, W, 4), np.uint8)
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


if __name__ == '__main__':
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
            print('ok', name)
