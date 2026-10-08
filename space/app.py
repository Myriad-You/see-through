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


def _decompose(image, resolution, seed, tblr_split, output_scale=1, size_condition="trained", hair_pass="canvas",
               body_tags=None, steps=30):
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
        extra = {"output_scale": output_scale, "size_condition": size_condition, "hair_pass": hair_pass,
                 "body_tags": body_tags} if canvas_mode else {}
        apply_layerdiff(
            input_path, REPO_LAYERDIFF,
            save_dir=tmpdir, seed=seed, resolution=resolution, num_inference_steps=steps, **extra,
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


@spaces.GPU(duration=180)
def decompose_lean(image: Image.Image, canvas: str = "1088x1664", seed: int = 42, tblr_split: bool = True,
                   body_tags: str = "front hair,back hair,head,neck,neckwear", steps: int = 30):
    """
    decompose for a picture whose body is another's: only the given body tags
    (the head and what turns with it), at the given number of steps. The head
    pass is whole.
    """
    tags = [t.strip() for t in str(body_tags).split(",") if t.strip()]
    steps = int(steps)
    if "head" not in tags or not 10 <= steps <= 30:
        raise gr.Error("Body tags must include head; steps 10..30.")
    return _decompose(image, _parse_canvas(canvas), seed, tblr_split, 1, "trained", "canvas", tags, steps)


@spaces.GPU(duration=300)
def decompose_hair(image: Image.Image, canvas: str = "1088x1664", seed: int = 42, tblr_split: bool = True,
                   hair_pass: str = "head"):
    """
    decompose with the hair from a pass of its own: the body tags run again on
    the hair's box scaled to the head pass's square, and its front and back
    hair replace the body pass's. 'canvas' is decompose as it is.
    """
    if hair_pass not in ("canvas", "head"):
        raise gr.Error("Hair pass must be 'canvas' or 'head'.")
    return _decompose(image, _parse_canvas(canvas), seed, tblr_split, 1, "trained", hair_pass)


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


def check(psd):
    """
    Whether a decomposition can be keyed from (turn_keyforms.check_decomposition):
    {ok, faults, badness, face_spread}. A caller redoes a faulty one with another seed. CPU only.
    """
    import turn_keyforms

    if psd is None:
        raise gr.Error("A decomposition PSD is needed.")
    return turn_keyforms.check_decomposition(psd if isinstance(psd, str) else psd.name)


def check_turns(front_psd, front_png, right_png, left_png, up_png, down_png):
    """
    Whether the turned drawings can be keyed from the front one
    (turn_keyforms.check_turned_pictures): {residual: {side: px}, bad: [side]}.
    A caller draws a bad one again before decomposing it. CPU only.
    """
    import turn_keyforms

    files = [front_psd, front_png, right_png, left_png, up_png, down_png]
    if any(f is None for f in files):
        raise gr.Error("The front PSD and all five pictures are needed.")
    path = lambda f: f if isinstance(f, str) else f.name
    return turn_keyforms.check_turned_pictures(path(front_psd), dict(
        front=path(front_png), plus=path(right_png), minus=path(left_png), up=path(up_png), down=path(down_png)))


def keyforms(front_psd, right_psd, left_psd, up_psd, down_psd,
             front_png=None, right_png=None, left_png=None, up_png=None, down_png=None):
    """
    Keyed head turns (turn_keyforms.py): each part of the front decomposition
    fitted onto the same part of four decompositions of the head turned toward
    image right and left, raised and lowered (same canvas, body lined up).
    With the five pictures that were decomposed, accessories are matched on
    them and every key is refined until the drawn turn matches its picture.
    Returns the keys as JSON (canvas coordinates) and the front decomposition
    with what the turns uncover baked in from the turned drawings. CPU only.
    """
    import turn_keyforms

    paths = [front_psd, right_psd, left_psd, up_psd, down_psd]
    if any(p is None for p in paths):
        raise gr.Error("The front and all four turned decompositions are needed.")
    path = lambda f: f if isinstance(f, str) else f.name
    pictures = [front_png, right_png, left_png, up_png, down_png]
    if any(p is not None for p in pictures) and any(p is None for p in pictures):
        raise gr.Error("Give all five pictures, or none.")
    pictures = dict(zip(("front", "plus", "minus", "up", "down"), map(path, pictures))) if pictures[0] is not None else None
    t0 = time.time()
    try:
        front, result = turn_keyforms.turn_keyforms(
            path(front_psd),
            dict(plus=path(right_psd), minus=path(left_psd), up=path(up_psd), down=path(down_psd)),
            log=_log,
            pictures=pictures,
        )
    except ValueError as error:
        raise gr.Error(str(error))
    out_dir = tempfile.mkdtemp(prefix="seethrough_keyforms_")
    psd_path = os.path.join(out_dir, "baked.psd")
    turn_keyforms.save_psd(front, psd_path)
    json_path = os.path.join(out_dir, "keyforms.json")
    with open(json_path, "w") as f:
        json.dump(result, f)
    _log(f"Keyforms done ({time.time() - t0:.1f}s)")
    return json_path, psd_path


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
        lean_tags = gr.Textbox(value="front hair,back hair,head,neck,neckwear", label="Body tags (lean)")
        lean_steps = gr.Slider(minimum=10, maximum=30, value=30, step=1, label="Steps (lean)")
        lean_btn = gr.Button("Run lean")
        lean_btn.click(
            fn=decompose_lean,
            inputs=[canvas_image, canvas_size, canvas_seed, canvas_split, lean_tags, lean_steps],
            outputs=[canvas_psd, canvas_gallery],
            api_name="decompose_lean",
        )
        canvas_hair = gr.Radio(choices=["canvas", "head"], value="head", label="Hair pass")
        canvas_hair_btn = gr.Button("Run with hair pass")
        canvas_hair_btn.click(
            fn=decompose_hair,
            inputs=[canvas_image, canvas_size, canvas_seed, canvas_split, canvas_hair],
            outputs=[canvas_psd, canvas_gallery],
            api_name="decompose_hair",
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

    with gr.Tab("Turn keyforms"):
        gr.Markdown(
            "Per-part keys for a head turned ±30° either way and raised or lowered, fitted from four "
            "turned decompositions onto the front one (all on the same canvas), and the front "
            "decomposition with what the turns uncover baked in. The pictures that were decomposed, "
            "if given, refine every key against them. CPU only."
        )
        with gr.Row():
            with gr.Column(scale=1):
                key_front = gr.File(label="Front PSD")
                key_right = gr.File(label="Turned toward image right PSD")
                key_left = gr.File(label="Turned toward image left PSD")
                key_up = gr.File(label="Raised PSD")
                key_down = gr.File(label="Lowered PSD")
                with gr.Accordion("Pictures (optional)", open=False):
                    pic_front = gr.File(label="Front picture")
                    pic_right = gr.File(label="Turned toward image right picture")
                    pic_left = gr.File(label="Turned toward image left picture")
                    pic_up = gr.File(label="Raised picture")
                    pic_down = gr.File(label="Lowered picture")
                key_btn = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                key_json = gr.File(label="Keyforms JSON")
                key_psd = gr.File(label="Baked front PSD")
        key_btn.click(
            fn=keyforms,
            inputs=[key_front, key_right, key_left, key_up, key_down, pic_front, pic_right, pic_left, pic_up, pic_down],
            outputs=[key_json, key_psd],
            api_name="keyforms",
        )
        with gr.Row():
            check_psd = gr.File(label="Decomposition PSD to check")
            check_json = gr.JSON(label="Check")
        check_btn = gr.Button("Check")
        check_btn.click(fn=check, inputs=[check_psd], outputs=[check_json], api_name="check")
        turns_json = gr.JSON(label="Turned pictures")
        turns_btn = gr.Button("Check the turned pictures (front PSD and pictures above)")
        turns_btn.click(
            fn=check_turns,
            inputs=[key_front, pic_front, pic_right, pic_left, pic_up, pic_down],
            outputs=[turns_json],
            api_name="check_turns",
        )

if __name__ == "__main__":
    demo.launch()
