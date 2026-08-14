"""
Depth model loading and inference.

Uses Depth Anything V2 (metric, outdoor checkpoint) via the HuggingFace
transformers pipeline. Runs on CPU by default -- no GPU required, just slower.

First run on a machine needs internet access once, to download the model
weights from HuggingFace (a few hundred MB). After that, weights are cached
locally (~/.cache/huggingface) and no internet is needed unless you clear
that cache. If you want a fully offline .exe (no internet needed even on
first run), see the note in README.md about pre-downloading and bundling
the weights.
"""

import numpy as np
import cv2
from PIL import Image

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

_pipe = None  # lazy-loaded, cached after first call


def load_depth_model(progress_callback=None):
    """Load (and cache) the depth estimation pipeline. Call once at app start."""
    global _pipe
    if _pipe is not None:
        return _pipe

    if progress_callback:
        progress_callback("Loading depth model (first run may take a while)...")

    from transformers import pipeline
    _pipe = pipeline(task="depth-estimation", model=DEPTH_MODEL_ID, device=-1)  # -1 = CPU

    if progress_callback:
        progress_callback("Depth model ready.")

    return _pipe


def run_depth(image_path):
    """Run the depth model on an image file. Returns (rgb_array, depth_array_meters)."""
    pipe = load_depth_model()

    img = Image.open(image_path).convert("RGB")
    out = pipe(img)
    depth = out["predicted_depth"]

    if hasattr(depth, "numpy"):
        depth = depth.squeeze().cpu().numpy()
    else:
        depth = np.array(depth).squeeze()

    if depth.shape[:2] != (img.height, img.width):
        depth = cv2.resize(depth, (img.width, img.height), interpolation=cv2.INTER_LINEAR)

    rgb = np.array(img)
    return rgb, depth.astype(np.float32)


def depth_to_color_image(depth_map, cmap_name="turbo"):
    """Convert a depth map (meters) to an RGB uint8 image for display, using a colormap."""
    import matplotlib

    d = depth_map.astype(np.float32)
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
    cmap = matplotlib.colormaps[cmap_name]
    colored = (cmap(d_norm)[:, :, :3] * 255).astype(np.uint8)
    return colored


def depth_rgb_blend(rgb_img, depth_map, alpha=0.35, cmap_name="turbo"):
    """Blend the RGB image with a depth colormap -- helps spot camouflaged objects/edges
    that don't stand out by color alone (e.g. an animal against dense green foliage)."""
    depth_colored = depth_to_color_image(depth_map, cmap_name)
    blended = cv2.addWeighted(rgb_img, 1 - alpha, depth_colored, alpha, 0)
    return blended


def bias_to_color_image(bias_map, cmap_name="coolwarm"):
    """Colorize a signed correction/bias map (meters), centered at zero, so red/blue
    show where the model over- or under-estimates distance and by how much."""
    import matplotlib

    d = bias_map.astype(np.float32)
    vmax = max(abs(d.min()), abs(d.max()), 1e-6)
    d_norm = (d + vmax) / (2 * vmax)  # maps [-vmax, vmax] -> [0, 1]
    cmap = matplotlib.colormaps[cmap_name]
    colored = (cmap(d_norm)[:, :, :3] * 255).astype(np.uint8)
    return colored


def draw_grid(img, step=50, color=(0, 255, 255), thickness=1):
    """Overlay a pixel-coordinate grid with tick labels -- makes it easier to read off
    approximate (x, y) when placing tie points."""
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), color, thickness, cv2.LINE_AA)
        cv2.putText(out, str(x), (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), color, thickness, cv2.LINE_AA)
        cv2.putText(out, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return out
