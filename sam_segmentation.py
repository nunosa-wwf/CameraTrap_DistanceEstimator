"""
Segment Anything (SAM) point-prompt segmentation.

Click once on the animal, get a mask -- generally cleaner than GrabCut
against cluttered/camouflaged backgrounds (forest foliage, etc).

Trade-off vs GrabCut: needs a one-time ~375MB checkpoint download (cached
locally after that) and is somewhat heavier per-call on CPU (a few seconds
per image for the embedding step, cached per image after the first click).
"""

import os
import urllib.request
import numpy as np

CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "sam_vit_b_01ec64.pth")

_predictor = None
_embedded_image_id = None  # id() of the last rgb array we called set_image() on, to avoid recomputing


def _ensure_checkpoint(progress_callback=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if not os.path.exists(CHECKPOINT_PATH):
        if progress_callback:
            progress_callback("Downloading SAM checkpoint (~375MB, one-time only)...")
        urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)


def load_sam(progress_callback=None):
    global _predictor
    if _predictor is not None:
        return _predictor

    _ensure_checkpoint(progress_callback)

    if progress_callback:
        progress_callback("Loading SAM model...")

    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT_PATH)
    sam.to("cpu")
    _predictor = SamPredictor(sam)
    return _predictor


def segment_point_and_median(rgb_img, depth_map, point_xy, progress_callback=None):
    """point_xy = (x, y) in pixel space. Returns (median_depth, mask) or (None, mask) if empty."""
    global _embedded_image_id
    predictor = load_sam(progress_callback)

    if _embedded_image_id != id(rgb_img):
        if progress_callback:
            progress_callback("Computing image embedding (first click on this image)...")
        predictor.set_image(rgb_img)
        _embedded_image_id = id(rgb_img)

    input_point = np.array([point_xy])
    input_label = np.array([1])  # 1 = foreground click

    masks, scores, _ = predictor.predict(
        point_coords=input_point, point_labels=input_label, multimask_output=True
    )
    best_idx = int(np.argmax(scores))
    mask = masks[best_idx]

    if mask.sum() == 0:
        return None, mask

    median_depth = float(np.median(depth_map[mask]))
    return median_depth, mask


def reset_image_cache():
    """Call when a new target image is loaded, so the next click recomputes the embedding."""
    global _embedded_image_id
    _embedded_image_id = None
