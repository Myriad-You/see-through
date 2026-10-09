"""
Anime super-resolution (Real-ESRGAN x4plus_anime_6B, BSD-3-Clause) for a
standing figure's picture before it is decomposed in tiles (figure_tiles):
enlarged by the network instead of interpolated, every tile is decomposed from
lines and edges, not blur. It only enlarges: nothing is redrawn.

    python upscale.py in.png out.png [height]     (to `height`, default TARGET_HEIGHT)
"""
import os
import sys
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

WEIGHTS_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
WEIGHTS_SHA256 = 'f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da'
SCALE = 4
# A figure is enlarged to this height; one already near it is left as it is.
TARGET_HEIGHT = 4096
NEAR = 1.25
# Tiles keep memory flat on a CPU; overlapped so their seams are cut away.
TILE = 256
TILE_PAD = 16


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_feat=64, num_block=6, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


_model = None


def weights_path():
    """The weights, fetched once into the torch cache and checked."""
    import hashlib
    path = os.path.join(torch.hub.get_dir(), 'checkpoints', os.path.basename(WEIGHTS_URL))
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(WEIGHTS_URL, path + '.part')
        os.replace(path + '.part', path)
    with open(path, 'rb') as f:
        if hashlib.sha256(f.read()).hexdigest() != WEIGHTS_SHA256:
            os.remove(path)
            raise RuntimeError('super-resolution weights do not match')
    return path


def model(path=None):
    global _model
    if _model is None:
        net = RRDBNet()
        state = torch.load(path or weights_path(), map_location='cpu', weights_only=True)
        net.load_state_dict(state.get('params_ema', state), strict=True)
        _model = net.eval()
    return _model


@torch.no_grad()
def upscale(rgb, path=None):
    """An RGB uint8 array four times larger, tile by tile."""
    net = model(path)
    h, w = rgb.shape[:2]
    x = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None]
    out = torch.zeros(1, 3, h * SCALE, w * SCALE)
    for y0 in range(0, h, TILE):
        for x0 in range(0, w, TILE):
            y1, x1 = min(h, y0 + TILE), min(w, x0 + TILE)
            py0, px0 = max(0, y0 - TILE_PAD), max(0, x0 - TILE_PAD)
            py1, px1 = min(h, y1 + TILE_PAD), min(w, x1 + TILE_PAD)
            tile = net(x[..., py0:py1, px0:px1])
            out[..., y0 * SCALE:y1 * SCALE, x0 * SCALE:x1 * SCALE] = tile[
                ..., (y0 - py0) * SCALE:(y1 - py0) * SCALE, (x0 - px0) * SCALE:(x1 - px0) * SCALE]
    return (out[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)


def cpu_quota():
    """The CPUs this container may use (cgroup cpu.max), else the machine's."""
    try:
        limit, period = open('/sys/fs/cgroup/cpu.max').read().split()
        if limit != 'max':
            return max(1, int(int(limit) / int(period)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def upscale_to(image, height=TARGET_HEIGHT, path=None):
    """A PIL image enlarged to `height` (aspect kept), or itself when it is near it already."""
    from PIL import Image
    if image.height * NEAR >= height:
        return image
    big = Image.fromarray(upscale(np.asarray(image.convert('RGB')), path))
    width = round(image.width * height / image.height)
    return big.resize((width, height), Image.LANCZOS)


if __name__ == '__main__':
    from PIL import Image
    import time
    torch.set_num_threads(cpu_quota())
    t0 = time.time()
    image = Image.open(sys.argv[1])
    out = upscale_to(image, int(sys.argv[3]) if len(sys.argv) > 3 else TARGET_HEIGHT)
    out.save(sys.argv[2])
    print(f'{image.width}x{image.height} -> {out.width}x{out.height} in {time.time() - t0:.1f}s on {torch.get_num_threads()} threads')
