import spaces
import gradio as gr
import json
import os
import sys
import time
import tempfile
import shutil
import torch

_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "common"))

from PIL import Image

REPO_LAYERDIFF = "layerdifforg/seethroughv0.0.2_layerdiff3d"
REPO_DEPTH = "24yearsold/seethroughv0.0.1_marigold"


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------- Preload models to CPU at startup ---------------
_log("Preloading LayerDiff pipeline to CPU...")
from modules.layerdiffuse.diffusers_kdiffusion_sdxl import KDiffusionStableDiffusionXLPipeline
from modules.layerdiffuse.layerdiff3d import UNetFrameConditionModel
from modules.layerdiffuse.vae import TransparentVAE, TransparentVAEDecoder, TransparentVAEEncoder

_trans_vae = TransparentVAE.from_pretrained(REPO_LAYERDIFF, subfolder="trans_vae")
_unet_ld = UNetFrameConditionModel.from_pretrained(REPO_LAYERDIFF, subfolder="unet")
_layerdiff_pipe = KDiffusionStableDiffusionXLPipeline.from_pretrained(
    REPO_LAYERDIFF, trans_vae=_trans_vae, unet=_unet_ld, scheduler=None
)
_log("LayerDiff pipeline loaded to CPU.")

_log("Preloading Marigold pipeline to CPU...")
from modules.marigold import MarigoldDepthPipeline

_unet_mg = UNetFrameConditionModel.from_pretrained(REPO_DEPTH, subfolder="unet")
_marigold_pipe = MarigoldDepthPipeline.from_pretrained(REPO_DEPTH, unet=_unet_mg)
_log("Marigold pipeline loaded to CPU.")

_models_on_gpu = False

from utils.inference_utils import apply_layerdiff, apply_marigold, further_extr, promote_output_scale
from utils.torch_utils import seed_everything
import utils.inference_utils as _inf


def _move_to_gpu():
    global _models_on_gpu
    if _models_on_gpu:
        _log("Models already on GPU, skipping transfer.")
        return

    t0 = time.time()
    _log("Moving LayerDiff to CUDA bf16...")
    _layerdiff_pipe.vae.to(dtype=torch.bfloat16, device="cuda")
    _layerdiff_pipe.trans_vae.to(dtype=torch.bfloat16, device="cuda")
    _layerdiff_pipe.unet.to(dtype=torch.bfloat16, device="cuda")
    _layerdiff_pipe.text_encoder.to(dtype=torch.bfloat16, device="cuda")
    _layerdiff_pipe.text_encoder_2.to(dtype=torch.bfloat16, device="cuda")
    _log(f"LayerDiff on GPU ({time.time() - t0:.1f}s)")

    t0 = time.time()
    _log("Moving Marigold to CUDA bf16...")
    _marigold_pipe.to(device="cuda", dtype=torch.bfloat16)
    _log(f"Marigold on GPU ({time.time() - t0:.1f}s)")

    # Inject into inference_utils globals so apply_* functions skip their own loading
    _inf.layerdiff_pipeline = _layerdiff_pipe
    _inf.marigold_pipeline = _marigold_pipe

    _models_on_gpu = True


_SKIP_TAGS = {"src_img", "src_head", "reconstruction"}

def _collect_layer_gallery(saved_dir):
    """Collect layer PNGs as (image, label) tuples for the gallery."""
    gallery = []
    for f in sorted(os.listdir(saved_dir)):
        if not f.endswith(".png"):
            continue
        tag = f[:-4]
        if tag.endswith("_depth") or tag in _SKIP_TAGS:
            continue
        img = Image.open(os.path.join(saved_dir, f))
        gallery.append((img, tag))
    return gallery


def _parse_canvas(canvas):
    """'WxH' to (h, w), both multiples of 64."""
    try:
        w, h = (int(v) for v in str(canvas).lower().replace(' ', '').split('x'))
    except ValueError:
        raise gr.Error("Canvas must look like 1088x1664 (width x height).")
    if w <= 0 or h <= 0 or w % 64 or h % 64:
        raise gr.Error("Canvas sides must be positive multiples of 64.")
    if w * h > 2048 * 2048:
        raise gr.Error("Canvas is larger than 2048x2048 in pixels.")
    return h, w


def _decompose(image, resolution, seed, tblr_split, output_scale=1, size_condition="trained"):
    """resolution: a square side, as the original demo, or an (h, w) canvas."""
    t_start = time.time()
    if image is None:
        raise gr.Error("Please upload an image.")
    _log(f"Resolution: {resolution}, Seed: {seed}, Image: {image.size}, Output scale: {output_scale}, Size condition: {size_condition}")

    _move_to_gpu()
    seed_everything(seed)

    tmpdir = tempfile.mkdtemp(prefix="seethrough_")
    try:
        input_path = os.path.join(tmpdir, "input.png")
        image.save(input_path)
        canvas_mode = not isinstance(resolution, int)

        t0 = time.time()
        _log("Running LayerDiff...")
        extra = {"output_scale": output_scale, "size_condition": size_condition} if canvas_mode else {}
        apply_layerdiff(
            input_path, REPO_LAYERDIFF,
            save_dir=tmpdir, seed=seed, resolution=resolution, **extra,
        )
        _log(f"LayerDiff done ({time.time() - t0:.1f}s)")

        t0 = time.time()
        _log("Running Marigold depth...")
        # On a canvas the depth model keeps the canvas's shape at the square's pixel budget.
        depth_resolution = resolution if not canvas_mode else TRAINED_DEPTH_BUDGET
        apply_marigold(
            input_path, REPO_DEPTH,
            save_dir=tmpdir, seed=seed, resolution=depth_resolution,
        )
        _log(f"Marigold done ({time.time() - t0:.1f}s)")

        saved = os.path.join(tmpdir, "input")
        promote_output_scale(saved, input_path)

        # Collect gallery before PSD assembly (further_extr may modify files)
        gallery = _collect_layer_gallery(saved)

        t0 = time.time()
        _log("Running PSD assembly...")
        further_extr(saved, rotate=False, save_to_psd=True, tblr_split=tblr_split)
        _log(f"PSD assembly done ({time.time() - t0:.1f}s)")

        psd_path = saved + ".psd"
        if os.path.exists(psd_path):
            output_path = os.path.join(
                tempfile.mkdtemp(prefix="seethrough_out_"), "seethrough_output.psd"
            )
            shutil.copy2(psd_path, output_path)
            _log(f"Total inference time: {time.time() - t_start:.1f}s")
            return output_path, gallery

        raise gr.Error("PSD generation failed — no output file produced.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# The original demo's depth resolution, used as a pixel budget on a canvas.
TRAINED_DEPTH_BUDGET = 1280


@spaces.GPU(duration=120)
def inference(image: Image.Image, resolution: int = 768, seed: int = 42, tblr_split: bool = False):
    """The original demo: the image padded to a square."""
    resolution = max(64, min(resolution, 1280))
    # Snap to nearest multiple of 64 for clean latent dimensions
    resolution = round(resolution / 64) * 64
    return _decompose(image, resolution, seed, tblr_split)


@spaces.GPU(duration=240)
def decompose(image: Image.Image, canvas: str = "1088x1664", seed: int = 42, tblr_split: bool = True,
              output_scale: int = 1, size_condition: str = "trained"):
    """
    The image fitted on a canvas of its own shape (WxH), never stretched. The
    head pass stays a 1280 square; output_scale 2 keeps its precision in a PSD
    twice the canvas size.
    """
    output_scale = int(output_scale)
    if output_scale not in (1, 2):
        raise gr.Error("Output scale must be 1 or 2.")
    if size_condition not in ("trained", "actual"):
        raise gr.Error("Size condition must be 'trained' or 'actual'.")
    return _decompose(image, _parse_canvas(canvas), seed, tblr_split, output_scale, size_condition)


@spaces.GPU(duration=300)
def refine(image: Image.Image, canvas: str = "", target: str = "back hair", modes: str = "repaint,peel,hint",
           seeds: str = "1,2", size_condition: str = "trained", hint: Image.Image = None):
    """
    The hidden-region experiment (refine_hidden.py): the target layer's hidden
    part redrawn with every visible pixel pinned. canvas empty is a 1280 square.
    hint: for mode 'given', a picture lined up with the image (the same frame
    with the occluders removed, say). Returns a zip of the results, their gallery and the untextured shares.
    """
    import refine_hidden

    if image is None:
        raise gr.Error("Please upload an image.")
    if target not in refine_hidden.OCCLUDERS:
        raise gr.Error(f"Target must be one of {sorted(refine_hidden.OCCLUDERS)}.")
    _move_to_gpu()
    resolution = _parse_canvas(canvas) if str(canvas).strip() else 1280
    tmpdir = tempfile.mkdtemp(prefix="seethrough_refine_")
    try:
        input_path = os.path.join(tmpdir, "input.png")
        image.save(input_path)
        hint_path = None
        if hint is not None:
            hint_path = os.path.join(tmpdir, "hint.png")
            hint.save(hint_path)
        elif "given" in modes.split(","):
            raise gr.Error("Mode 'given' needs a hint image.")
        out_dir = refine_hidden.refine(
            input_path, save_dir=tmpdir, target=target, modes=modes, seeds=seeds,
            resolution=resolution, steps=30, repo_id_layerdiff=REPO_LAYERDIFF, size_condition=size_condition,
            hint_path=hint_path,
        )
        archive = shutil.make_archive(
            os.path.join(tempfile.mkdtemp(prefix="seethrough_out_"), "refine"), "zip", out_dir
        )
        gallery = [
            (Image.open(os.path.join(out_dir, f)), f[:-4])
            for f in sorted(os.listdir(out_dir)) if f.endswith(".png")
        ]
        with open(os.path.join(out_dir, "stats.json")) as f:
            stats = json.load(f)
        return archive, gallery, stats
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


with gr.Blocks(title="See-through: Layer Decomposition") as demo:
    gr.Markdown(
        "# See-through: Single-image Layer Decomposition for Anime Characters\n\n"
        '<a href="https://github.com/shitagaki-lab/see-through" target="_blank">GitHub</a> | '
        '<a href="https://arxiv.org/abs/2602.03749" target="_blank">Paper (arXiv:2602.03749)</a>\n\n'
        "Upload an anime character illustration to decompose it into "
        "fully-inpainted semantic layers with depth ordering, "
        "exported as a layered PSD file.\n\n"
        "**Note:** 768 resolution is recommended for ZeroGPU free tier. "
        "Higher resolutions may timeout or exhaust your daily quota. "
        'For best quality, clone the <a href="https://github.com/shitagaki-lab/see-through" target="_blank">full repo</a> '
        "and run `inference_psd.py` locally. "
        'We also have a <a href="https://github.com/jtydhr88/ComfyUI-See-through" target="_blank">ComfyUI Node Extension</a> '
        "and other community extensions — check our repo for more details.\n\n"
        "**Disclaimer:** This demo uses a newer model checkpoint and "
        "may not fully reproduce identical results reported in the paper."
    )
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload image (non-square images will be padded)")
            resolution = gr.Slider(
                minimum=768, maximum=1280, value=768, step=64,
                label="Resolution",
                info="768 recommended for ZeroGPU free tier. Higher resolutions may timeout or use up your daily quota quickly.",
            )
            seed = gr.Slider(minimum=0, maximum=9999, value=42, step=1, label="Seed")
            tblr_split = gr.Checkbox(
                value=False,
                label="Split left/right arms & legs",
                info="Separate left and right limbs into individual layers. Useful if the default output glues them together.",
            )
            run_btn = gr.Button("Run", variant="primary")
        with gr.Column(scale=2):
            psd_output = gr.File(label="Download layered PSD")
            gallery_output = gr.Gallery(label="Separated layers", columns=4, height="auto")

    run_btn.click(
        fn=inference,
        inputs=[input_image, resolution, seed, tblr_split],
        outputs=[psd_output, gallery_output],
    )
    gr.Examples(
        examples=[["common/assets/test_image.png", 768, 42, False]],
        inputs=[input_image, resolution, seed, tblr_split],
    )

    with gr.Tab("Canvas"):
        gr.Markdown(
            "Fits the image on a canvas of its own shape instead of padding it to a square. "
            "Output scale 2 keeps the head pass's precision in a PSD twice the canvas size."
        )
        with gr.Row():
            with gr.Column(scale=1):
                canvas_image = gr.Image(type="pil", label="Image")
                canvas_size = gr.Textbox(value="1088x1664", label="Canvas (width x height, multiples of 64)")
                canvas_seed = gr.Slider(minimum=0, maximum=9999, value=42, step=1, label="Seed")
                canvas_split = gr.Checkbox(value=True, label="Split left/right arms & legs")
                canvas_scale = gr.Radio(choices=[1, 2], value=1, label="Output scale")
                canvas_condition = gr.Radio(choices=["trained", "actual"], value="trained", label="SDXL size condition")
                canvas_btn = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                canvas_psd = gr.File(label="Download layered PSD")
                canvas_gallery = gr.Gallery(label="Separated layers", columns=4, height="auto")
        canvas_btn.click(
            fn=decompose,
            inputs=[canvas_image, canvas_size, canvas_seed, canvas_split, canvas_scale, canvas_condition],
            outputs=[canvas_psd, canvas_gallery],
            api_name="decompose",
        )

    with gr.Tab("Hidden region"):
        gr.Markdown(
            "Redraws only the hidden part of one layer with every visible pixel pinned "
            "(repaint / peel / hint), and reports the untextured share of the hidden region."
        )
        with gr.Row():
            with gr.Column(scale=1):
                refine_image = gr.Image(type="pil", label="Image")
                refine_canvas = gr.Textbox(value="", label="Canvas (empty for a 1280 square)")
                refine_target = gr.Textbox(value="back hair", label="Target layer")
                refine_modes = gr.Textbox(value="repaint,peel,hint", label="Modes")
                refine_seeds = gr.Textbox(value="1,2", label="Seeds")
                refine_condition = gr.Radio(choices=["trained", "actual"], value="trained", label="SDXL size condition")
                refine_hint = gr.Image(type="pil", label="Hint for mode 'given' (lined up with the image)")
                refine_btn = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                refine_zip = gr.File(label="Download results")
                refine_gallery = gr.Gallery(label="Target layer", columns=4, height="auto")
                refine_stats = gr.JSON(label="Untextured share of the hidden region")
        refine_btn.click(
            fn=refine,
            inputs=[refine_image, refine_canvas, refine_target, refine_modes, refine_seeds, refine_condition, refine_hint],
            outputs=[refine_zip, refine_gallery, refine_stats],
            api_name="refine",
        )

if __name__ == "__main__":
    demo.launch()
