'''
LayerDiffDataset3D with, for each frame, where its target is known
(<stem>_<tag>_known.png, written by scripts/psd_to_sample.py), at the
latent's resolution: a hand-checked decomposition shows what the picture
shows, and what lies hidden under a part in front was often never drawn.
A latent cell is known only when all of its pixels are. A tag a sample
does not have is known only outside the figure (<stem>_absent_known.png);
a sample with no such files is known everywhere (a decomposition the
model drew right, replayed as it drew it).

It also drops the empty entries a sample list's trailing newline leaves.
'''
import os.path as osp
import random

import cv2
import numpy as np
import torch
from PIL import Image

from train.dataset_layerdiff import LayerDiffDataset3D, load_blend_parts, xyxy_from_taglist
from utils.cv import center_square_pad_resize, random_pad_img, pad_rgb
from utils.io_utils import json2dict
from utils.torch_utils import img2tensor

LATENT_SCALE = 8


def latent_known(mask: np.ndarray) -> torch.Tensor:
    '''A pixel mask (H, W, 0..255) as (1, H/8, W/8): a cell is known only if all its pixels are.'''
    h, w = mask.shape[:2]
    cells = (mask > 127).reshape(h // LATENT_SCALE, LATENT_SCALE, w // LATENT_SCALE, LATENT_SCALE)
    return torch.from_numpy(cells.all(axis=(1, 3)).astype(np.float32))[None]


class KnownMaskDataset3D(LayerDiffDataset3D):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_list = [s for s in self.sample_list if s.get('img', '').strip()]
        assert self.target_size % LATENT_SCALE == 0

    def get_sample(self, index: int):
        flip_lr = self.random_flip > 0 and random.random() < self.random_flip

        def _aug_size_transform(img, pad_value=0):
            if flip_lr:
                img = img[:, ::-1]
            return center_square_pad_resize(np.ascontiguousarray(img), target_size=self.target_size, pad_value=pad_value)

        sample = self.sample_list[index]
        imgp = sample['img']
        imgname = osp.join(osp.dirname(imgp), osp.splitext(osp.basename(imgp))[0])
        ann = json2dict(imgname + '_ann.json')
        fullpage = np.array(Image.open(imgp).convert('RGBA'))

        tag_group_info = self.mix_tag_groups[self._current_group]
        target_tags = tag_group_info['tags'].copy()
        blended_rst = load_blend_parts(imgp, tag_group_info['tags'], ann=ann, keep_loaded_parts=True)

        crop_tag_list = tag_group_info.get(
            'crop_tag_list', [t for t in target_tags if t not in tag_group_info.get('size_skip_tags', [])])
        tx1, ty1, tx2, ty2 = xyxy_from_taglist(ann, crop_tag_list)

        if random.random() < self.random_pad_prob:
            blended_img = blended_rst['img'][ty1: ty2, tx1: tx2]
            fh, fw = fullpage.shape[:2]
            _, (t, b, l, r) = random_pad_img(blended_img, ty1, fh - ty2 - 1, tx1, fw - tx2 - 1)
            tx1 -= l
            ty1 -= t
            tx2 += r
            ty2 += b
        fullpage = _aug_size_transform(fullpage[ty1: ty2, tx1: tx2].copy())
        fh, fw = fullpage.shape[:2]

        cond_fullpage = np.concatenate([np.full_like(fullpage[..., :1], fill_value=255), fullpage[..., :3]], axis=2)
        cond_fullpage = img2tensor(img=cond_fullpage, normalize=True, mean=self.pixel_mean, std=self.pixel_std, dim_order='chw')

        def _np_transform(img):
            img = _aug_size_transform(img)
            if self.pad_argb:
                img = pad_rgb(img, return_format='argb')
            else:
                img = np.concatenate([img[..., 3:], img[..., :3]], axis=2).astype(np.float32) / 255.
            return img2tensor(img=img, dim_order='chw')

        def _known(path):
            if not osp.exists(path):
                return None
            mask = np.array(Image.open(path).convert('L'))[ty1: ty2, tx1: tx2]
            # Past the picture the target is empty, and that is known.
            return latent_known(_aug_size_transform(mask, pad_value=255))

        empty_tensor = torch.zeros((4, fh, fw), dtype=torch.float32)
        every = torch.ones((1, fh // LATENT_SCALE, fw // LATENT_SCALE), dtype=torch.float32)
        absent = _known(imgname + '_absent_known.png')
        absent = every if absent is None else absent

        tgt_img_list = [empty_tensor] * len(target_tags)
        known_list = [absent] * len(target_tags)
        caption_list = target_tags
        empty_mask = [True] * len(target_tags)
        footwear_exist = False
        for part in blended_rst['part_list']:
            tag_index = target_tags.index(part['tag'])
            if part['tag'] == 'footwear':
                footwear_exist = True
            empty_mask[tag_index] = False
            tgt_img_list[tag_index] = _np_transform(part['img'][ty1: ty2, tx1: tx2])
            known = _known(imgname + f"_{part['tag']}_known.png")
            known_list[tag_index] = every if known is None else known

        if not footwear_exist and 'legwear' in caption_list:
            caption_list[caption_list.index('legwear')] = 'legwear,footwear'

        npad = self.pad_num_frames - len(caption_list)
        if npad > 0:
            caption_list = caption_list + npad * ['']
            tgt_img_list += [empty_tensor] * npad
            known_list += [every] * npad
            empty_mask += [True] * npad

        batch = {
            'img_list': torch.stack(tgt_img_list),
            'known': torch.stack(known_list),
            'cond_fullpage': cond_fullpage,
            'srcp': imgp,
            'caption_list': '\n'.join(caption_list),
            'original_sizes': (self.target_size, self.target_size),
            'crop_top_lefts': (0, 0),
            'empty_mask': torch.tensor(empty_mask),
            'group_index': self._current_group,
        }
        self._current_group_iters += 1
        if self._current_group_iters % self.ngroup_iters == 0:
            self._current_group_iters = 0
            self._current_group = (self._current_group + 1) % self.ngroup
        return batch
