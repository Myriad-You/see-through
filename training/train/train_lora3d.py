'''
A LoRA on the released LayerDiff 3D model, on one GPU, from hand-checked
samples (scripts/psd_to_sample.py, read by dataset_known.py).

Differences from train_layerdiff3d.py, each one a way the full script would
have spent a run for nothing on this task:

- Everything is loaded from the model repo the Space infers with (UNet,
  text encoders, VAE and its transparent encoder), so the targets are
  encoded as inference reads them.
- init_weights() is never called: it zeroes the temporal blocks' outputs
  and the group embeddings, which is right only for a 2D UNet converted to
  3D, and would wipe what the released model knows of layers together.
- Only LoRA weights on the attention projections train; the rest is frozen
  in bf16. The adapter alone is saved, and can be dropped.
- The loss counts only latent cells whose target is known (dataset_known);
  hidden parts nobody drew are left to what the model already does.
- One process, no DeepSpeed; gradient checkpointing; resumable from the
  last checkpoint in the output directory, which a Colab session may lose.

    python training/train/train_lora3d.py --config training/configs/lora3d_pilot.yaml
    python training/train/train_lora3d.py --config ... --tiny   # wiring test on CPU, random small model
'''
import argparse
import glob
import json
import math
import os
import os.path as osp
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path[:0] = [osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), p) for p in ('common', 'training', '')]

from train.dataset_known import KnownMaskDataset3D, LATENT_SCALE  # noqa: E402

BODY_TAGS = ['front hair', 'back hair', 'head', 'neck', 'neckwear', 'topwear', 'handwear', 'bottomwear', 'legwear', 'footwear', 'tail', 'wings', 'objects']
HEAD_TAGS = ['headwear', 'face', 'irides', 'eyebrow', 'eyewhite', 'eyelash', 'eyewear', 'ears', 'earwear', 'nose', 'mouth']
LORA_TARGETS = ['to_q', 'to_k', 'to_v', 'to_out.0']


def log(record, path=None):
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    if path:
        with open(path, 'a') as f:
            f.write(line + '\n')


def masked_loss(pred, target, known, snr_weights):
    '''Per-sample MSE over known latent cells (all channels), weighted per sample.'''
    err = (pred.float() - target.float()) ** 2                 # b f c h w
    k = known.to(err.dtype)                                    # b f 1 h w
    per = (err * k).sum(dim=(1, 2, 3, 4)) / (k.sum(dim=(1, 2, 3, 4)) * err.shape[2]).clamp(min=1)
    return (per * snr_weights).mean(), float(k.mean())


class Real:
    '''The released model's parts, loaded as the Space's pipeline loads them.'''

    def __init__(self, repo, device, dtype):
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
        from diffusers import AutoencoderKL
        from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
        from modules.layerdiffuse.vae import TransparentVAE
        self.tokenizers = [CLIPTokenizer.from_pretrained(repo, subfolder=s) for s in ('tokenizer', 'tokenizer_2')]
        self.text_encoders = [
            CLIPTextModel.from_pretrained(repo, subfolder='text_encoder').to(device, dtype),
            CLIPTextModelWithProjection.from_pretrained(repo, subfolder='text_encoder_2').to(device, dtype),
        ]
        self.vae = AutoencoderKL.from_pretrained(repo, subfolder='vae').to(device, torch.float32)
        self.trans_encoder = TransparentVAE.from_pretrained(repo, subfolder='trans_vae').encoder.to(device, torch.float32)
        self.unet = UNetFrameConditionModel.from_pretrained(repo, subfolder='unet', torch_dtype=dtype).to(device)
        for m in [*self.text_encoders, self.vae, self.trans_encoder]:
            m.requires_grad_(False)
        self._prompt_cache = {}

    def encode(self, argb, use_offset):
        from modules.layerdiffuse.vae import vae_encode
        return vae_encode(self.vae, self.trans_encoder, argb, use_offset=use_offset)

    def prompts(self, captions, device, dtype):
        key = tuple(captions)
        if key not in self._prompt_cache:
            from train.train_layerdiff3d import encode_prompt
            enc = encode_prompt(list(captions), self.text_encoders, self.tokenizers, proportion_empty_prompts=0.0)
            self._prompt_cache[key] = (enc['prompt_embeds'].to(device, dtype), enc['pooled_prompt_embeds'].to(device, dtype))
        return self._prompt_cache[key]


class Tiny:
    '''A small random model with the released config's structure, for testing the wiring on a CPU.'''

    def __init__(self, repo_config, device, dtype):
        from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
        config = dict(repo_config)
        for k in [k for k in config if k.startswith('_')]:
            config.pop(k)
        config.update(block_out_channels=[32, 64, 64], attention_head_dim=[2, 4, 4], transformer_layers_per_block=[1, 1, 1],
                      layers_per_block=1, norm_num_groups=16)
        self.unet = UNetFrameConditionModel(**config).to(device, dtype)
        self.cross_dim = config['cross_attention_dim']
        self.pooled_dim = config['projection_class_embeddings_input_dim'] - 6 * config['addition_time_embed_dim']

    def encode(self, argb, use_offset):
        # Four channels at an eighth of the size, as the VAE's latents are.
        return F.avg_pool2d(argb.float() * 2 - 1, LATENT_SCALE)

    def prompts(self, captions, device, dtype):
        g = torch.Generator().manual_seed(hash(tuple(captions)) % (2 ** 31))
        n = len(captions)
        return (torch.randn(n, 77, self.cross_dim, generator=g).to(device, dtype),
                torch.randn(n, self.pooled_dim, generator=g).to(device, dtype))


def add_lora(unet, rank, alpha):
    from peft import LoraConfig
    unet.requires_grad_(False)
    unet.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights='gaussian', target_modules=LORA_TARGETS))
    params = [p for n, p in unet.named_parameters() if 'lora_' in n]
    for p in params:
        p.data = p.data.float()
        p.requires_grad_(True)
    return params


def lora_state(unet):
    from peft import get_peft_model_state_dict
    return {k: v.detach().to('cpu', torch.float32).contiguous() for k, v in get_peft_model_state_dict(unet).items()}


def save_checkpoint(out_dir, step, unet, optimizer, rng_state):
    from safetensors.torch import save_file
    path = osp.join(out_dir, f'checkpoint-{step:06d}')
    os.makedirs(path, exist_ok=True)
    save_file(lora_state(unet), osp.join(path, 'lora.safetensors'))
    torch.save({'optimizer': optimizer.state_dict(), 'step': step, 'rng': rng_state}, osp.join(path, 'state.pt'))
    with open(osp.join(path, 'DONE'), 'w') as f:   # a checkpoint without it was cut off mid-write
        f.write(str(step))
    return path


def latest_checkpoint(out_dir):
    done = sorted(p for p in glob.glob(osp.join(out_dir, 'checkpoint-*')) if osp.exists(osp.join(p, 'DONE')))
    return done[-1] if done else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--max_steps', type=int, default=None)
    cli = parser.parse_args()
    args = OmegaConf.load(cli.config)
    if cli.max_steps is not None:
        args.max_steps = cli.max_steps
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    log_path = osp.join(out_dir, 'log.jsonl')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not cli.tiny else 'cpu')
    dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

    if cli.tiny:
        repo_config = json.load(open(args.tiny_unet_config))
        model = Tiny(repo_config, device, dtype)
    else:
        model = Real(args.repo, device, dtype)
    unet = model.unet
    # Deliberately no unet.init_weights(): see the module's docstring.
    params = add_lora(unet, args.lora_rank, args.lora_alpha)
    n_lora = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in unet.parameters())
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    unet.train()

    from diffusers import DDPMScheduler
    from diffusers.training_utils import compute_snr
    scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear',
                              num_train_timesteps=1000, prediction_type='epsilon')

    groups = [{'tags': BODY_TAGS}, {'tags': HEAD_TAGS}]
    dataset = KnownMaskDataset3D(list(OmegaConf.to_container(args.samples)), target_size=args.target_size,
                                 random_flip=args.random_flip, tag_list=BODY_TAGS + HEAD_TAGS,
                                 random_pad_prob=args.random_pad_prob, mix_tag_groups=groups,
                                 ngroup_iters=args.get('ngroup_iters', 1), pad_argb=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                         persistent_workers=args.num_workers > 0)
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    step = 0
    resume = latest_checkpoint(out_dir)
    if resume:
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(unet, load_file(osp.join(resume, 'lora.safetensors')))
        state = torch.load(osp.join(resume, 'state.pt'), weights_only=False)
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
        random.setstate(state['rng']['python'])
        torch.set_rng_state(state['rng']['torch'])
    log({'event': 'start', 'device': str(device), 'samples': len(dataset), 'lora_params': n_lora, 'all_params': n_all,
         'resumed_from': resume, 'step': step}, log_path)

    def lr_at(s):
        return args.learning_rate * min(1.0, (s + 1) / max(1, args.warmup_steps))

    accum = args.gradient_accumulation_steps
    t_last = time.time()
    running = []
    data_iter = iter(loader)
    while step < args.max_steps:
        for g in optimizer.param_groups:
            g['lr'] = lr_at(step)
        optimizer.zero_grad(set_to_none=True)
        known_share = []
        for _ in range(accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            with torch.no_grad():
                use_offset = random.random() < args.vae_use_offset_prob
                imgs = batch['img_list'].to(device)                      # 1 f 4 H W (argb, 0..1)
                b, f = imgs.shape[:2]
                # vae_encode runs in inference mode: clone, so autograd may use what comes of it.
                latents = model.encode(imgs.flatten(0, 1), use_offset).clone()
                latents = latents.view(b, f, *latents.shape[1:]).to(dtype)
                cond = model.encode(batch['cond_fullpage'].to(device), use_offset).clone().to(dtype)
                captions = batch['caption_list'][0].split('\n')
                prompt_embeds, pooled = model.prompts(captions, device, dtype)
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (b,), device=device)
                noisy = scheduler.add_noise(latents, noise, timesteps).to(dtype)
                size = args.target_size
                time_ids = torch.tensor([[size, size, 0, 0, size, size]], device=device, dtype=dtype).repeat(b * f, 1)
                known = batch['known'].to(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda'):
                pred = unet(
                    torch.cat([noisy, cond[:, None].expand(-1, f, -1, -1, -1)], dim=2),
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={'time_ids': time_ids, 'text_embeds': pooled},
                    return_dict=False,
                    group_index=int(batch['group_index'][0]),
                )[0]
            snr = compute_snr(scheduler, timesteps)
            weights = torch.minimum(snr, torch.full_like(snr, args.snr_gamma)) / snr
            loss, share = masked_loss(pred, noise, known, weights)
            (loss / accum).backward()
            running.append(loss.item())
            known_share.append(share)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm))
        if not math.isfinite(grad_norm):
            log({'event': 'nonfinite_grad', 'step': step}, log_path)
            raise SystemExit('non-finite gradient: stopping rather than training on')
        optimizer.step()
        step += 1
        if step % args.log_every == 0 or step == 1:
            now = time.time()
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == 'cuda' else 0.0
            log({'step': step, 'loss': round(float(np.mean(running)), 5), 'lr': lr_at(step - 1), 'grad_norm': round(grad_norm, 4),
                 'known': round(float(np.mean(known_share)), 3), 'sec_per_step': round((now - t_last) / (args.log_every if step > 1 else 1), 2),
                 'max_mem_gb': round(mem, 2)}, log_path)
            running, t_last = [], now
        if step % args.checkpoint_every == 0 or step == args.max_steps:
            path = save_checkpoint(out_dir, step, unet, optimizer, {'python': random.getstate(), 'torch': torch.get_rng_state()})
            log({'event': 'checkpoint', 'step': step, 'path': path}, log_path)
    log({'event': 'done', 'step': step}, log_path)


if __name__ == '__main__':
    main()
