"""
Box-to-mask segmentation using OpenCV's GrabCut -- no extra model weights
or downloads needed, works fully offline, fast enough on CPU.
"""

import numpy as np
import cv2


def segment_and_median(rgb_img, depth_map, coords, iterations=5):
    """coords = (x0, y0, x1, y1) in pixel space. Returns (median_depth, mask)."""
    x0, y0, x1, y1 = coords
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))

    mask = np.zeros(rgb_img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

    box_mask = np.zeros_like(fg_mask)
    box_mask[y0:y1, x0:x1] = 1
    final_mask = fg_mask & box_mask

    if final_mask.sum() == 0:
        region = depth_map[y0:y1, x0:x1]
        return float(np.median(region)), final_mask

    median_depth = float(np.median(depth_map[final_mask == 1]))
    return median_depth, final_mask


def median_in_box(depth_map, coords):
    x0, y0, x1, y1 = coords
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return float(np.median(depth_map[y0:y1, x0:x1]))
