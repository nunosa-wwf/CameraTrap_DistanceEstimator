"""
Tie-point CSV import/export, and bundling all outputs (tables + images) into
a single downloadable zip for a given session.
"""

import os
import csv
import shutil
import tempfile
from datetime import datetime

import numpy as np
import cv2


def load_tie_points_csv(csv_path):
    """Expects columns x, y, true_dist (case-insensitive, any column order).
    Returns a list of tie point dicts, or raises ValueError with a clear message."""
    tie_points = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldmap = {name.strip().lower(): name for name in (reader.fieldnames or [])}
        required = ["x", "y"]
        dist_key = None
        for candidate in ("true_dist", "true_dist_m", "distance", "distance_m", "dist"):
            if candidate in fieldmap:
                dist_key = fieldmap[candidate]
                break
        if dist_key is None or "x" not in fieldmap or "y" not in fieldmap:
            raise ValueError(
                "CSV needs columns 'x', 'y', and a distance column "
                "(one of: true_dist, true_dist_m, distance, distance_m, dist). "
                f"Found columns: {reader.fieldnames}"
            )
        for row in reader:
            try:
                tie_points.append({
                    "x": int(float(row[fieldmap["x"]])),
                    "y": int(float(row[fieldmap["y"]])),
                    "true_dist": float(row[dist_key]),
                })
            except (ValueError, KeyError):
                continue  # skip malformed rows rather than failing the whole import
    if not tie_points:
        raise ValueError("No valid rows found in the CSV.")
    return tie_points


def save_tie_points_csv(tie_points, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "true_dist_m", "pred_dist_m", "residual_m"])
        for tp in tie_points:
            writer.writerow([
                tp.get("x"), tp.get("y"), tp.get("true_dist"),
                tp.get("pred_dist", ""), tp.get("residual", ""),
            ])


def bundle_results(out_zip_path, *, tie_points=None, calibration_summary=None,
                    bg_rgb=None, bg_depth_raw=None, bg_depth_corrected=None,
                    target_rgb=None, target_depth_corrected=None,
                    segmentation_overlay=None, uncertainty_map=None,
                    result_summary=None):
    """Writes everything passed in (any subset -- pass None to skip) into a temp
    folder, then zips it to out_zip_path. Returns out_zip_path."""
    tmp_dir = tempfile.mkdtemp(prefix="depth_calib_export_")

    try:
        if tie_points:
            save_tie_points_csv(tie_points, os.path.join(tmp_dir, "tie_points.csv"))

        if calibration_summary:
            with open(os.path.join(tmp_dir, "calibration_summary.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in calibration_summary.items():
                    writer.writerow([k, v])

        if result_summary:
            with open(os.path.join(tmp_dir, "result_summary.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                for k, v in result_summary.items():
                    writer.writerow([k, v])

        def save_img(arr, name, cmap=None):
            if arr is None:
                return
            path = os.path.join(tmp_dir, name)
            if cmap is not None:
                import matplotlib.pyplot as plt
                plt.imsave(path, arr, cmap=cmap)
            else:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
                cv2.imwrite(path, bgr)

        save_img(bg_rgb, "background_rgb.png")
        save_img(bg_depth_raw, "background_depth_raw.png", cmap="turbo")
        save_img(bg_depth_corrected, "background_depth_corrected.png", cmap="turbo")
        save_img(target_rgb, "target_rgb.png")
        save_img(target_depth_corrected, "target_depth_corrected.png", cmap="turbo")
        save_img(segmentation_overlay, "segmentation_overlay.png")
        save_img(uncertainty_map, "correction_uncertainty.png", cmap="viridis")

        if uncertainty_map is not None:
            np.save(os.path.join(tmp_dir, "correction_uncertainty_raw.npy"), uncertainty_map)

        base, _ = os.path.splitext(out_zip_path)
        shutil.make_archive(base, "zip", tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return out_zip_path


def default_export_path(prefix="depth_calib_export"):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = tempfile.gettempdir()
    return os.path.join(out_dir, f"{prefix}_{run_id}.zip")
