"""
A decomposition PSD checked by hand, as a LayerDiff training sample in the
layout syn_data.py writes (dataset_layerdiff.LayerDiffDataset3D reads it):

    <stem>.png               the picture fitted on the PSD's canvas
    <stem>_ann.json          {final_size, tag_info: {tag: {exists, xyxy}}}
    <stem>_<tag>.png         the tag's layers composited, cropped to xyxy
    <stem>_<tag>_depth.png   their depth (0 near), from the depth PSD or the PSD's order
    <stem>_<tag>_known.png   where the target is known: drawn, or empty and not
                             under a part in front. A checked PSD shows what
                             the picture shows; what lies hidden under a part
                             in front was often never drawn, and an empty
                             target there would teach the model not to draw it.
                             The released training does not read this mask.

Sides and locks are merged (handwear-l, front hair-3), and 'head' is made as
syn_data.py makes it: the face and its features, without hair or headwear.

    python psd_to_sample.py decomposition.psd picture.png out_dir stem [depth.psd]
"""
import json
import os.path as osp
import re
import sys

import cv2
import numpy as np
from PIL import Image
from psd_tools import PSDImage

HEAD_PARTS = ('face', 'irides', 'eyebrow', 'eyewhite', 'eyelash', 'eyewear', 'ears', 'nose', 'mouth')


def tag_of(name):
    '''A layer's tag: handwear-l, front hair-3 and eyes-r are handwear, front hair and eyes.'''
    return re.sub(r'-(l|r|\d+)$', '', name.strip())


def canvas_layers(path):
    psd = PSDImage.open(path)
    out = []
    for layer in psd.descendants():
        if layer.is_group():
            continue
        image = np.array(layer.topil().convert('RGBA'))
        canvas = np.zeros((psd.height, psd.width, 4), np.uint8)
        y, x = max(0, layer.top), max(0, layer.left)
        crop = image[y - layer.top:, x - layer.left:]
        h, w = min(crop.shape[0], psd.height - y), min(crop.shape[1], psd.width - x)
        canvas[y:y + h, x:x + w] = crop[:h, :w]
        out.append((layer.name, canvas))
    return out, (psd.height, psd.width)


def fit(picture, hw):
    '''The picture fitted on the canvas as the decomposition fitted it (centred, one scale).'''
    ch, cw = hw
    sh, sw = picture.shape[:2]
    scale = min(cw / sw, ch / sh)
    w, h = min(cw, int(np.floor(sw * scale + 0.5))), min(ch, int(np.floor(sh * scale + 0.5)))
    page = np.full((ch, cw, 3), 255, np.uint8)
    x, y = (cw - w) // 2, (ch - h) // 2
    page[y:y + h, x:x + w] = cv2.resize(picture, (w, h), interpolation=cv2.INTER_AREA)
    return page


def over(images):
    '''Straight-alpha RGBA images composited back to front.'''
    color = np.zeros(images[0].shape[:2] + (3,), np.float32)
    alpha = np.zeros(images[0].shape[:2] + (1,), np.float32)
    for image in images:
        a = image[..., 3:4].astype(np.float32) / 255
        color = image[..., :3].astype(np.float32) * a + color * (1 - a)
        alpha = a + alpha * (1 - a)
    rgb = np.where(alpha > 0, color / np.maximum(alpha, 1e-6), 0)
    return np.concatenate([rgb, alpha * 255], axis=2).round().clip(0, 255).astype(np.uint8)


def convert(psd_path, picture_path, out_dir, stem, depth_path=None):
    layers, hw = canvas_layers(psd_path)
    depth = {}
    if depth_path:
        for name, image in canvas_layers(depth_path)[0]:
            depth[name] = image[..., 0]
    n = len(layers)
    groups = {}
    for index, (name, image) in enumerate(layers):
        tag = tag_of(name)
        d = depth.get(name)
        if d is None:   # back to front: far to near
            d = np.full(hw, round(255 * (1 - index / max(1, n - 1))), np.uint8)
        groups.setdefault(tag, []).append((index, image, d))
    head = [entry for tag in HEAD_PARTS for entry in groups.get(tag, [])]
    if head:
        groups['head'] = sorted(head, key=lambda entry: entry[0])
    # Where some layer in front of a tag draws: what it hides of the tag is not known to be empty.
    drawn = [image[..., 3] > 10 for _, image in layers]
    front_of = np.zeros((n + 1,) + hw, bool)
    for index in range(n - 1, -1, -1):
        front_of[index] = front_of[index + 1] | drawn[index]
    picture = np.array(Image.open(picture_path).convert('RGB'))
    page = fit(picture, hw)
    Image.fromarray(np.concatenate([page, np.full(hw + (1,), 255, np.uint8)], axis=2)).save(osp.join(out_dir, f'{stem}.png'))
    tag_info = {}
    for tag, entries in groups.items():
        image = over([e[1] for e in entries])
        ys, xs = np.nonzero(image[..., 3] > 0)
        if not len(ys):
            tag_info[tag] = {'exists': False}
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        d = np.zeros(hw, np.uint8)
        for _, layer, layer_depth in entries:
            d = np.where(layer[..., 3] > 10, layer_depth, d)
        front = front_of[max(e[0] for e in entries) + 1]
        known = ((image[..., 3] > 0) | ~front).astype(np.uint8) * 255
        Image.fromarray(image[y1:y2, x1:x2]).save(osp.join(out_dir, f'{stem}_{tag}.png'))
        Image.fromarray(d[y1:y2, x1:x2]).save(osp.join(out_dir, f'{stem}_{tag}_depth.png'))
        Image.fromarray(known).save(osp.join(out_dir, f'{stem}_{tag}_known.png'))
        tag_info[tag] = {'exists': True, 'xyxy': [x1, y1, x2, y2]}
    with open(osp.join(out_dir, f'{stem}_ann.json'), 'w') as f:
        json.dump({'final_size': list(hw), 'tag_info': tag_info, 'source': osp.basename(psd_path)}, f)
    return tag_info


if __name__ == '__main__':
    info = convert(*sys.argv[1:6])
    print({tag: value.get('xyxy') for tag, value in info.items()})
