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


def figure_head(full_psd, image):
    """
    Where a standing figure's head is, framed as a bust (figure_head.head_crop):
    {box: [x0, y0, x1, y1] in image pixels, or null when the head is big
    enough already}, and that box of the image at a bust's size, white past its
    edges, to draw turns of and decompose as a bust. CPU only.
    """
    import figure_head
    import turn_keyforms
    from PIL import Image as PILImage

    if full_psd is None or image is None:
        raise gr.Error("The figure's decomposition and its picture are needed.")
    path = lambda f: f if isinstance(f, str) else f.name
    picture = PILImage.open(path(image)).convert("RGB")
    box = figure_head.head_crop(turn_keyforms.Decomposition(path(full_psd)), (picture.height, picture.width))
    if box is None:
        return {"box": None}, None
    out = os.path.join(tempfile.mkdtemp(prefix="seethrough_figure_"), "head.png")
    figure_head.crop_image(picture, box).save(out)
    return {"box": box}, out


def figure_keys(image, figure_psd, head_keyforms, box):
    """
    A standing figure's turn keys: the keys fitted on its head framed as a bust
    (figure_head's crop, decomposed and keyed as a bust), placed on the
    figure's canvas to move the figure's own head parts (figure_head.figure_keys).
    box: "x0,y0,x1,y1" as figure_head gave it. CPU only.
    """
    import figure_head
    from PIL import Image as PILImage
    from psd_tools import PSDImage

    if any(f is None for f in (image, figure_psd, head_keyforms)) or not box:
        raise gr.Error("The picture, the figure's decomposition, the head's keys and the box are needed.")
    path = lambda f: f if isinstance(f, str) else f.name
    try:
        crop = [int(v) for v in str(box).split(",")]
    except ValueError:
        raise gr.Error("The box must look like x0,y0,x1,y1.")
    if len(crop) != 4 or crop[2] <= crop[0] or crop[3] <= crop[1]:
        raise gr.Error("The box must look like x0,y0,x1,y1.")
    picture = PILImage.open(path(image))
    width, height = PSDImage.open(path(figure_psd)).size
    with open(path(head_keyforms)) as f:
        keys = figure_head.figure_keys(json.load(f), (height, width), crop, (picture.height, picture.width))
    json_path = os.path.join(tempfile.mkdtemp(prefix="seethrough_figure_"), "keyforms.json")
    with open(json_path, "w") as f:
        json.dump(keys, f)
    return json_path


def upscale(image):
    """
    A standing figure's picture enlarged to 4096 px tall by anime
    super-resolution (upscale.py), or as it is when it is near that already.
    Run in a process of its own: the network's threads are never started in
    this one, which forks for the GPU. CPU only.
    """
    import subprocess

    if image is None:
        raise gr.Error("A picture is needed.")
    path = image if isinstance(image, str) else image.name
    out = os.path.join(tempfile.mkdtemp(prefix="seethrough_upscale_"), "upscaled.png")
    script = os.path.join(_root, "upscale.py")
    env = dict(os.environ)
    t0 = time.time()
    done = subprocess.run([sys.executable, script, path, out], env=env, capture_output=True, text=True, timeout=1800)
    if done.returncode != 0:
        _log(done.stderr[-2000:])
        raise gr.Error("Super-resolution failed.")
    _log(f"Upscale: {done.stdout.strip()} ({time.time() - t0:.1f}s)")
    return out


def body_keys(figure_psd, right_psd, left_psd, right_png, left_png, image):
    """
    A standing figure's body turn keys (body_turn.py): each body part of the
    figure's decomposition fitted onto decompositions of the figure drawn
    turned toward image right and left (same canvas, feet lined up), refined
    on those two pictures, keeping only what moves opposite ways in the two,
    and placed on `image`, the picture the figure's decomposition was made of.
    Returns the keys as JSON ({canvas, keyforms, drift, knees}). Run in a
    process of its own (it sets the part tables turn_keyforms reads). CPU only.
    """
    import subprocess
    from PIL import Image as PILImage

    files = [figure_psd, right_psd, left_psd, right_png, left_png, image]
    if any(f is None for f in files):
        raise gr.Error("The figure's decomposition, both turned decompositions, both turned pictures and the figure's picture are needed.")
    path = lambda f: f if isinstance(f, str) else f.name
    width, height = PILImage.open(path(image)).size
    out = os.path.join(tempfile.mkdtemp(prefix="seethrough_body_"), "body_keys.json")
    script = os.path.join(_root, "body_turn.py")
    t0 = time.time()
    done = subprocess.run([sys.executable, script, path(figure_psd), path(right_psd), path(left_psd),
                           path(right_png), path(left_png), str(height), str(width), out],
                          capture_output=True, text=True, timeout=1800)
    if done.returncode != 0:
        _log(done.stderr[-2000:])
        raise gr.Error("The body turn could not be keyed.")
    _log(f"Body keys: {done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ''} ({time.time() - t0:.1f}s)")
    return out


def figure_plan(whole_psd, image):
    """
    A standing figure's tiles (figure_tiles.plan): {tiles: {upper, middle,
    lower: [x0, y0, x1, y1]}} in the picture's pixels, or {tiles: null} when
    its head is big enough already; and each tile's picture to decompose (the
    upper one a bust's, also the head to key). CPU only.
    """
    import figure_tiles
    import turn_keyforms
    from PIL import Image as PILImage

    if whole_psd is None or image is None:
        raise gr.Error("The figure's decomposition and its picture are needed.")
    path = lambda f: f if isinstance(f, str) else f.name
    picture = PILImage.open(path(image)).convert("RGB")
    tiles = figure_tiles.plan(turn_keyforms.Decomposition(path(whole_psd)), (picture.height, picture.width))
    if tiles is None:
        return {"tiles": None}, None, None, None
    out_dir = tempfile.mkdtemp(prefix="seethrough_tiles_")
    files = []
    for name, crop in figure_tiles.crops(picture, tiles).items():
        crop_path = os.path.join(out_dir, f"{name}.png")
        crop.save(crop_path)
        files.append(crop_path)
    return {"tiles": tiles}, *files


def figure_stitch(image, whole_psd, upper_psd, middle_psd, lower_psd, tiles):
    """
    The figure's PSD at the picture's size, stitched from its tiles'
    decompositions on its whole one (figure_tiles.stitch). tiles: the JSON
    figure_plan gave. CPU only.
    """
    import figure_tiles
    from PIL import Image as PILImage

    files = (image, whole_psd, upper_psd, middle_psd, lower_psd)
    if any(f is None for f in files) or not tiles:
        raise gr.Error("The picture, the whole decomposition, the three tiles' and the plan are needed.")
    path = lambda f: f if isinstance(f, str) else f.name
    try:
        plan = json.loads(tiles) if isinstance(tiles, str) else tiles
        plan = plan.get("tiles", plan)
        assert all(len(plan[name]) == 4 for name in ("upper", "middle", "lower"))
    except Exception:
        raise gr.Error("The plan must be figure_plan's JSON.")
    picture = PILImage.open(path(image))
    out = os.path.join(tempfile.mkdtemp(prefix="seethrough_stitch_"), "figure.psd")
    t0 = time.time()
    figure_tiles.stitch_files(
        (picture.height, picture.width), path(whole_psd),
        dict(upper=path(upper_psd), middle=path(middle_psd), lower=path(lower_psd)), plan, out, log=_log)
    _log(f"Stitch done ({time.time() - t0:.1f}s)")
    return out


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

    with gr.Tab("Figure head"):
        gr.Markdown(
            "A standing figure's head, framed as a bust to draw turned, decompose and key on its own; "
            "its keys placed back on the figure's canvas. CPU only."
        )
        with gr.Row():
            with gr.Column(scale=1):
                fig_image = gr.File(label="Figure picture")
                fig_psd = gr.File(label="Figure decomposition PSD")
                fig_btn = gr.Button("Find the head")
                fig_keys = gr.File(label="Head keyforms JSON")
                fig_box = gr.Textbox(label="Box (x0,y0,x1,y1)")
                keys_btn = gr.Button("Place the keys", variant="primary")
            with gr.Column(scale=2):
                fig_found = gr.JSON(label="Head box")
                fig_crop = gr.File(label="Head picture")
                fig_out_json = gr.File(label="Keyforms on the figure")
        fig_btn.click(fn=figure_head, inputs=[fig_psd, fig_image], outputs=[fig_found, fig_crop], api_name="figure_head")
        keys_btn.click(
            fn=figure_keys,
            inputs=[fig_image, fig_psd, fig_keys, fig_box],
            outputs=[fig_out_json],
            api_name="figure_keys",
        )

    with gr.Tab("Figure tiles"):
        gr.Markdown(
            "A standing figure decomposed in tiles: its picture enlarged, its whole decomposition "
            "planned into three tiles (head and chest as a bust, waist, legs), each tile decomposed on "
            "its own canvas, and the tiles stitched on the whole. CPU only."
        )
        with gr.Row():
            with gr.Column(scale=1):
                up_in = gr.File(label="Figure picture")
                up_btn = gr.Button("Enlarge")
                plan_psd = gr.File(label="Whole decomposition PSD")
                plan_btn = gr.Button("Plan the tiles")
                st_upper = gr.File(label="Upper tile PSD")
                st_middle = gr.File(label="Middle tile PSD")
                st_lower = gr.File(label="Lower tile PSD")
                st_plan = gr.Textbox(label="Plan JSON")
                st_btn = gr.Button("Stitch", variant="primary")
            with gr.Column(scale=2):
                up_out = gr.File(label="Enlarged picture")
                plan_json = gr.JSON(label="Tiles")
                plan_upper = gr.File(label="Upper tile picture")
                plan_middle = gr.File(label="Middle tile picture")
                plan_lower = gr.File(label="Lower tile picture")
                st_out = gr.File(label="Figure PSD")
        up_btn.click(fn=upscale, inputs=[up_in], outputs=[up_out], api_name="upscale")
        plan_btn.click(fn=figure_plan, inputs=[plan_psd, up_in], outputs=[plan_json, plan_upper, plan_middle, plan_lower],
                       api_name="figure_plan")
        st_btn.click(fn=figure_stitch, inputs=[up_in, plan_psd, st_upper, st_middle, st_lower, st_plan], outputs=[st_out],
                     api_name="figure_stitch")

    with gr.Tab("Body turn"):
        gr.Markdown(
            "A standing figure's body turn keys: its decomposition fitted onto decompositions of it drawn "
            "turned about 20 degrees toward image right and left, feet where they stood. CPU only."
        )
        with gr.Row():
            with gr.Column(scale=1):
                bt_figure = gr.File(label="Figure decomposition PSD")
                bt_image = gr.File(label="Figure picture (that decomposition's)")
                bt_right_psd = gr.File(label="Turned right PSD")
                bt_left_psd = gr.File(label="Turned left PSD")
                bt_right_png = gr.File(label="Turned right picture")
                bt_left_png = gr.File(label="Turned left picture")
                bt_btn = gr.Button("Key the body turn", variant="primary")
            with gr.Column(scale=2):
                bt_out = gr.File(label="Body turn keys JSON")
        bt_btn.click(fn=body_keys, inputs=[bt_figure, bt_right_psd, bt_left_psd, bt_right_png, bt_left_png, bt_image],
                     outputs=[bt_out], api_name="body_keys")

if __name__ == "__main__":
    demo.launch()
