'''train_lora3d on a tiny random model (CPU): python training/tests/test_lora3d.py <unet_config.json> <sample_list.txt>'''
import copy
import json
import os
import os.path as osp
import subprocess
import sys
import tempfile

import torch

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path[:0] = [osp.join(ROOT, 'common'), osp.join(ROOT, 'training'), ROOT]

from train import train_lora3d as t  # noqa: E402


def tiny(config_path, seed=0):
    torch.manual_seed(seed)
    return t.Tiny(json.load(open(config_path)), torch.device('cpu'), torch.float32)


def test_masked_loss_ignores_unknown_cells():
    pred = torch.zeros(1, 2, 4, 3, 3)
    target = torch.zeros(1, 2, 4, 3, 3)
    known = torch.ones(1, 2, 1, 3, 3)
    known[0, 1, 0, 1, 1] = 0
    base, _ = t.masked_loss(pred, target, known, torch.ones(1))
    pred2 = pred.clone()
    pred2[0, 1, :, 1, 1] = 10.0                 # wrong only where unknown
    same, _ = t.masked_loss(pred2, target, known, torch.ones(1))
    assert same == base == 0
    pred2[0, 0, :, 1, 1] = 1.0                  # wrong where known
    worse, _ = t.masked_loss(pred2, target, known, torch.ones(1))
    assert worse > 0


def run(config, tiny_cfg, max_steps):
    out = subprocess.run([sys.executable, '-I', osp.join(ROOT, 'training/train/train_lora3d.py'), '--config', config,
                          '--tiny', '--max_steps', str(max_steps)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    return [json.loads(line) for line in out.stdout.splitlines() if line.startswith('{')]


def main(config_path, sample_list):
    test_masked_loss_ignores_unknown_cells()
    print('ok masked loss')
    with tempfile.TemporaryDirectory() as tmp:
        cfg = osp.join(tmp, 'tiny.yaml')
        with open(cfg, 'w') as f:
            f.write(f'''repo: unused
tiny_unet_config: {config_path}
output_dir: {tmp}/out
samples:
  - sample_list: {sample_list}
target_size: 256
random_flip: 0.5
random_pad_prob: 0.5
num_workers: 0
seed: 42
lora_rank: 4
lora_alpha: 4
learning_rate: 1.0e-2
warmup_steps: 1
weight_decay: 0.0
gradient_accumulation_steps: 1
max_grad_norm: 1.0
max_steps: 4
gradient_checkpointing: true
snr_gamma: 5.0
vae_use_offset_prob: 0.25
log_every: 1
checkpoint_every: 2
''')
        first = run(cfg, config_path, 4)
        assert first[-1] == {'event': 'done', 'step': 4}
        resumed = run(cfg, config_path, 6)
        start = [r for r in resumed if r.get('event') == 'start'][0]
        assert start['step'] == 4 and start['resumed_from'].endswith('checkpoint-000004'), start
        print('ok resume from', osp.basename(start['resumed_from']))

        # The base weights are the model's own; only the adapter moved, and it reloads to the same outputs.
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        base = tiny(config_path)
        before = {k: v.clone() for k, v in base.unet.state_dict().items()}
        trained = tiny(config_path)
        t.add_lora(trained.unet, 4, 4)
        lora = load_file(osp.join(tmp, 'out', 'checkpoint-000006', 'lora.safetensors'))
        assert any(v.abs().sum() > 0 for k, v in lora.items() if 'lora_B' in k), 'lora_B moved off zero'
        set_peft_model_state_dict(trained.unet, lora)
        for k, v in trained.unet.state_dict().items():
            if 'lora_' in k:
                continue
            key = k.replace('.base_layer', '')
            assert torch.equal(v, before[key]), f'base weight changed: {k}'
        print('ok base frozen;', len(lora), 'adapter tensors')
        torch.manual_seed(1)
        x = torch.randn(1, 13, 8, 32, 32)
        enc, pooled = trained.prompts(['a'] * 13, torch.device('cpu'), torch.float32)
        time_ids = torch.tensor([[256, 256, 0, 0, 256, 256]], dtype=torch.float32).repeat(13, 1)
        kw = dict(encoder_hidden_states=enc, added_cond_kwargs={'time_ids': time_ids, 'text_embeds': pooled}, return_dict=False, group_index=0)
        trained.unet.eval()
        base.unet.eval()
        with torch.no_grad():
            y_lora = trained.unet(x, torch.tensor([500]), **kw)[0]
            y_base = base.unet(x, torch.tensor([500]), **kw)[0]
            trained.unet.disable_adapters()
            y_off = trained.unet(x, torch.tensor([500]), **kw)[0]
        assert not torch.allclose(y_lora, y_base), 'the adapter changes the output'
        assert torch.allclose(y_off, y_base, atol=1e-5), 'with the adapter off it is the base model'
        print('ok adapter reload changes outputs; switched off it is the base model')


if __name__ == '__main__':
    main(*sys.argv[1:3])
