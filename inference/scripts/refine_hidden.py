"""Regenerate the occluded part of one layer with LayerDiff 3D itself.

LayerDiff learned its amodal completions from Live2D drawables, where hidden
regions are often painted as a flat wash. This keeps every visible pixel of
every layer pinned and lets the model redraw only the hidden part of one target
layer, under three conditions:

  repaint  the original page; does the flat wash change with the seed?
  peel     the page with the hidden region showing the target layer itself
  hint     the page with the hidden region showing a LaMa extension of the
           target's visible texture, for the model to redraw as clean art
  given    the page with the hidden region showing a picture made elsewhere
           (hint_path), such as an image model's redraw of the same frame with
           the occluders removed; it must line up with the source image

Writes <save_dir>/<name>/refine/<mode>-<seed>.png (the target layer) and
stats.json with the share of untextured pixels inside the hidden region.

    python inference/scripts/refine_hidden.py --srcp portrait.png --target "back hair"
"""
import argparse
import json
import os
import os.path as osp

import cv2
import numpy as np
import torch
from PIL import Image

from utils import inference_utils
from utils.inference_utils import apply_layerdiff
from utils.torch_utils import seed_everything

BODY_TAGS = ['front hair', 'back hair', 'head', 'neck', 'neckwear', 'topwear', 'handwear',
             'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects']
# Layers drawn over the back hair in a bust portrait.
OCCLUDERS = {'back hair': ['front hair', 'head', 'neck', 'neckwear', 'topwear']}


def load_rgba(path):
    return np.array(Image.open(path).convert('RGBA'))


def bare_share(rgba, region):
    """Share of region pixels whose 9x9 luminance spread is below 2 (an untextured wash)."""
    lum = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = cv2.blur(lum, (9, 9))
    spread = np.sqrt(np.maximum(cv2.blur(lum * lum, (9, 9)) - mean * mean, 0))
    inside = region & (rgba[..., 3] > 128)
    return float((spread[inside] < 2).mean()) if inside.any() else 0.0


def latent_mask(hidden, latent_hw, grow=2):
    """1 = keep: the pixel mask shrunk to latent size, with the hidden region grown a little."""
    h, w = latent_hw
    small = cv2.resize(hidden.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_AREA) > 0
    small = cv2.dilate(small.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)) > 0
    return torch.from_numpy((~small).astype(np.float32))[None]


def _given_on_page(hint_path, srcp, saved, page_hw):
    '''The hint, at the source's size, placed on the page as the source was.'''
    from utils.cv import center_square_pad_resize, fit_pad_resize
    source_hw = Image.open(srcp).size[::-1]
    hint = np.array(Image.open(hint_path).convert('RGBA').resize(source_hw[::-1], Image.LANCZOS))
    canvas_info = osp.join(saved, inference_utils.CANVAS_INFO)
    if osp.exists(canvas_info):
        placed, _ = fit_pad_resize(hint, json.load(open(canvas_info))['canvas'])
    else:
        placed = center_square_pad_resize(hint, page_hw[0])
    return placed


def refine(srcp, save_dir='workspace/layerdiff_output', target='back hair', modes='repaint,peel,hint',
           seeds='1,2', resolution=1280, steps=30, repo_id_layerdiff='layerdifforg/seethroughv0.0.2_layerdiff3d',
           size_condition='trained', hint_path=None):
    """
    Runs every mode and seed; returns the output directory. resolution is a
    square side, or an (h, w) canvas fitted to the figure as in apply_layerdiff;
    size_condition applies to a canvas, as there.
    """
    args = argparse.Namespace(srcp=srcp, save_dir=save_dir, target=target, modes=modes, seeds=seeds,
                              resolution=resolution, steps=steps, repo_id_layerdiff=repo_id_layerdiff)
    name = osp.splitext(osp.basename(args.srcp))[0]
    saved = osp.join(args.save_dir, name)
    canvas_info = osp.join(saved, inference_utils.CANVAS_INFO)
    if osp.exists(canvas_info) and json.load(open(canvas_info)).get('promoted'):
        raise ValueError(f'{saved} was brought to its output scale; refine works on the canvas-size parts')
    run_kwargs = {} if isinstance(args.resolution, int) else {'size_condition': size_condition}
    if not osp.exists(osp.join(saved, f'{args.target}.png')):
        seed_everything(42)
        apply_layerdiff(args.srcp, args.repo_id_layerdiff, save_dir=args.save_dir, seed=42,
                        resolution=args.resolution, num_inference_steps=args.steps, **run_kwargs)
    pipe = inference_utils.layerdiff_pipeline
    if pipe is None:
        apply_layerdiff(args.srcp, args.repo_id_layerdiff, save_dir=args.save_dir, seed=42,
                        resolution=args.resolution, num_inference_steps=1, **run_kwargs)
        pipe = inference_utils.layerdiff_pipeline
    # The same SDXL size condition the parts were made with.
    condition = None
    if not isinstance(args.resolution, int) and size_condition == 'trained':
        condition = (inference_utils.TRAINED_SIZE, inference_utils.TRAINED_SIZE)

    page = load_rgba(osp.join(saved, 'src_img.png'))
    layers = [load_rgba(osp.join(saved, f'{tag}.png')) for tag in BODY_TAGS]
    target_index = BODY_TAGS.index(args.target)
    target = layers[target_index]
    cover = np.zeros(target.shape[:2], np.float32)
    for tag in OCCLUDERS[args.target]:
        cover = np.maximum(cover, layers[BODY_TAGS.index(tag)][..., 3] / 255.0)
    hidden = (target[..., 3] > 8) & (cover > 0.8)

    known = pipe.encode_layers(layers)
    lh, lw = known.shape[-2:]
    masks = torch.ones((len(layers), 1, lh, lw))
    masks[target_index] = latent_mask(hidden, (lh, lw))

    out_dir = osp.join(saved, 'refine')
    os.makedirs(out_dir, exist_ok=True)
    stats = {'hidden_px': int(hidden.sum()), 'original': bare_share(target, hidden)}
    Image.fromarray(target).save(osp.join(out_dir, 'original.png'))

    for mode in args.modes.split(','):
        fullpage = page.copy()
        if mode == 'peel':
            fullpage[hidden, :3] = target[hidden, :3]
        elif mode == 'given':
            if hint_path is None:
                raise ValueError("mode 'given' needs hint_path")
            given = _given_on_page(hint_path, args.srcp, saved, page.shape[:2])
            Image.fromarray(given).save(osp.join(out_dir, 'given.png'))
            fullpage[hidden, :3] = given[hidden, :3]
        elif mode == 'hint':
            from annotators.lama_inpainter import apply_inpaint
            # Fill what is hidden or transparent from the visible texture alone, so
            # nothing bleeds in from the black outside the layer.
            fill = hidden | (target[..., 3] < 128)
            hint = apply_inpaint(np.ascontiguousarray(target[..., :3]), fill.astype(np.uint8) * 255)
            Image.fromarray(hint.astype(np.uint8)).save(osp.join(out_dir, 'hint.png'))
            fullpage[hidden, :3] = hint[hidden]
        for seed in [int(s) for s in args.seeds.split(',')]:
            rng = torch.Generator(device=pipe.unet.device).manual_seed(seed)
            result = pipe(
                strength=1.0, num_inference_steps=args.steps, batch_size=1, generator=rng,
                guidance_scale=1.0, prompt=BODY_TAGS, negative_prompt='', fullpage=fullpage,
                group_index=0, known_latents=known, known_mask=masks, size_condition=condition,
            ).images[target_index]
            Image.fromarray(result).save(osp.join(out_dir, f'{mode}-{seed}.png'))
            stats[f'{mode}-{seed}'] = bare_share(result, hidden)
            print(mode, seed, stats[f'{mode}-{seed}'], flush=True)

    with open(osp.join(out_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    return out_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcp', required=True)
    parser.add_argument('--save_dir', default='workspace/layerdiff_output')
    parser.add_argument('--target', default='back hair')
    parser.add_argument('--modes', default='repaint,peel,hint')
    parser.add_argument('--seeds', default='1,2')
    parser.add_argument('--resolution', type=int, default=1280)
    parser.add_argument('--canvas', default=None, help='WxH canvas fitted to the figure instead of a square, e.g. 1088x1664')
    parser.add_argument('--size_condition', choices=['trained', 'actual'], default='trained')
    parser.add_argument('--hint_path', default=None, help="for mode 'given': a picture aligned with srcp")
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--repo_id_layerdiff', default='layerdifforg/seethroughv0.0.2_layerdiff3d')
    options = vars(parser.parse_args())
    canvas = options.pop('canvas')
    if canvas is not None:
        w, h = (int(v) for v in canvas.lower().split('x'))
        options['resolution'] = (h, w)
    refine(**options)
