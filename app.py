"""
Local Gradio app for the depth calibration pipeline.

Run with: python app.py
Opens a browser window pointed at http://127.0.0.1:7860 automatically.
"""

import numpy as np
import gradio as gr
import cv2

from depth_model import run_depth, depth_to_color_image, depth_rgb_blend, draw_grid
from calibration import (
    compute_residuals, fit_scalar_correction, apply_scalar_correction,
    build_correction_rbf, fit_regression_kriging, apply_regression_kriging,
    choose_best_method, bootstrap_uncertainty,
)
import sam_segmentation
import io_utils
import llm_report

state = {
    "bg_rgb": None, "bg_depth": None,
    "tie_points": [],
    "show_grid": True, "show_depth_overlay": False,
    "correction_method": None, "correction_params": None,
    "calibration_maes": None,
    "target_rgb": None, "target_depth_corrected": None, "target_depth_raw": None,
    "target_point": None,
    "uncertainty_map": None, "uncertainty_at_point": None,
    "segmentation_overlay": None,
    "final_distance": None,
}


# ---------------- Tab 1: background + tie points ----------------

def render_bg_display():
    if state["bg_rgb"] is None:
        return None

    if state["show_depth_overlay"] and state["bg_depth"] is not None:
        base = depth_rgb_blend(state["bg_rgb"], state["bg_depth"])
    else:
        base = state["bg_rgb"].copy()

    if state["show_grid"]:
        base = draw_grid(base, step=50)

    for tp in state["tie_points"]:
        cv2.circle(base, (tp["x"], tp["y"]), 6, (255, 0, 0), -1)
        cv2.circle(base, (tp["x"], tp["y"]), 6, (255, 255, 255), 2)
        label = f'{tp["true_dist"]}m'
        cv2.putText(base, label, (tp["x"] + 8, tp["y"] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(base, label, (tp["x"] + 8, tp["y"] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return base


def tie_points_table():
    return [[tp["x"], tp["y"], tp["true_dist"]] for tp in state["tie_points"]]


def load_background(image_path, progress=gr.Progress()):
    if image_path is None:
        return None, "Upload a background/calibration image first.", tie_points_table()
    progress(0.2, desc="Running depth model...")
    rgb, depth = run_depth(image_path)
    state["bg_rgb"] = rgb
    state["bg_depth"] = depth
    # keep existing tie points if a CSV was already imported before loading the image
    return render_bg_display(), "Background loaded and depth computed. Click to add tie points, or import a CSV below.", tie_points_table()


def add_tie_point(true_dist, evt: gr.SelectData):
    if state["bg_rgb"] is None:
        return None, "Load a background image first.", tie_points_table()
    x, y = evt.index[0], evt.index[1]
    state["tie_points"].append({"x": int(x), "y": int(y), "true_dist": float(true_dist)})
    return render_bg_display(), "", tie_points_table()


def undo_last_tie_point():
    if state["tie_points"]:
        state["tie_points"].pop()
    return render_bg_display(), tie_points_table()


def import_tie_points_csv(csv_path):
    if csv_path is None:
        return render_bg_display(), "No file selected.", tie_points_table()
    try:
        tps = io_utils.load_tie_points_csv(csv_path)
    except ValueError as e:
        return render_bg_display(), f"Import failed: {e}", tie_points_table()
    state["tie_points"] = tps
    return render_bg_display(), f"Imported {len(tps)} tie points from CSV.", tie_points_table()


def toggle_grid(value):
    state["show_grid"] = value
    return render_bg_display()


def toggle_depth_overlay(value):
    state["show_depth_overlay"] = value
    return render_bg_display()


# ---------------- Tab 2: calibrate ----------------

def run_calibration():
    tps = state["tie_points"]
    if len(tps) < 2:
        return "Need at least 2 tie points to calibrate.", tie_points_table()
    if state["bg_depth"] is None:
        return "Load a background image first.", tie_points_table()

    compute_residuals(tps, state["bg_depth"])
    best_method, maes = choose_best_method(tps, state["bg_depth"])
    state["calibration_maes"] = maes

    if best_method == "scalar" or len(tps) < 4:
        a, b = fit_scalar_correction(tps)
        state["correction_method"] = "scalar"
        state["correction_params"] = (a, b)
    elif best_method == "rbf":
        correction = build_correction_rbf(tps, state["bg_depth"].shape)
        state["correction_method"] = "rbf"
        state["correction_params"] = correction
    else:
        fit = fit_regression_kriging(tps, state["bg_depth"])
        state["correction_method"] = "kriging"
        state["correction_params"] = fit

    mae_lines = "\n".join(
        f"  {k}: {'%.3f m' % v if v is not None else 'n/a (need >=4 points)'}" for k, v in maes.items()
    )
    msg = f"Calibration fit using: {best_method}\n\nLeave-one-out MAE by method:\n{mae_lines}"
    return msg, tie_points_table()


def _apply_current_correction(depth_map):
    method = state["correction_method"]
    if method == "scalar":
        a, b = state["correction_params"]
        return apply_scalar_correction(depth_map, a, b)
    elif method == "rbf":
        return depth_map + state["correction_params"]
    elif method == "kriging":
        correction = apply_regression_kriging(depth_map, state["correction_params"])
        return depth_map + correction
    else:
        return depth_map


# ---------------- Tab 3: target image + SAM point-click distance ----------------

def load_target(image_path):
    if image_path is None:
        return None, "Upload the target (animal) image first."
    if state["correction_method"] is None:
        return None, "Run calibration first (Tab 2)."

    rgb, depth = run_depth(image_path)
    if state["bg_depth"] is not None and depth.shape != state["bg_depth"].shape:
        return None, (f"Warning: target image resolution {depth.shape} differs from calibration "
                       f"image {state['bg_depth'].shape}. Results may be unreliable.")

    corrected = _apply_current_correction(depth)
    state["target_rgb"] = rgb
    state["target_depth_raw"] = depth
    state["target_depth_corrected"] = corrected
    state["target_point"] = None
    state["uncertainty_map"] = None
    state["segmentation_overlay"] = None
    state["final_distance"] = None
    sam_segmentation.reset_image_cache()
    return rgb, "Target loaded. Click once on the animal, then press 'Segment & compute distance'."


def handle_target_click(evt: gr.SelectData):
    if state["target_rgb"] is None:
        return None, "Load a target image first."
    x, y = evt.index[0], evt.index[1]
    state["target_point"] = (x, y)
    img = state["target_rgb"].copy()
    cv2.drawMarker(img, (x, y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    return img, f"Point set at ({x},{y}). Click 'Segment & compute distance' below."


def compute_distance(progress=gr.Progress()):
    if state["target_point"] is None:
        return None, "Click on the animal in the image above first."
    if state["target_depth_corrected"] is None:
        return None, "Load a target image first."

    def cb(msg):
        progress(0.3, desc=msg)

    median_depth, mask = sam_segmentation.segment_point_and_median(
        state["target_rgb"], state["target_depth_corrected"], state["target_point"], progress_callback=cb
    )

    if median_depth is None:
        return state["target_rgb"], "SAM found an empty mask at that point -- try clicking closer to the center of the animal."

    overlay = state["target_rgb"].copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    x, y = state["target_point"]
    cv2.drawMarker(overlay, (x, y), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
    state["segmentation_overlay"] = overlay
    state["final_distance"] = median_depth

    # uncertainty of the correction itself (from calibration), at the clicked point
    progress(0.7, desc="Estimating correction uncertainty...")
    tps = state["tie_points"]
    if len(tps) >= 2 and state["bg_depth"] is not None:
        std_map, _ = bootstrap_uncertainty(tps, state["bg_depth"], state["correction_method"], n_boot=20)
        # std_map is in calibration-image pixel space; only meaningful if same resolution as target
        if std_map.shape == state["target_depth_corrected"].shape:
            state["uncertainty_map"] = std_map
            state["uncertainty_at_point"] = float(std_map[y, x])
        else:
            state["uncertainty_map"] = None
            state["uncertainty_at_point"] = None

    unc_txt = ""
    if state.get("uncertainty_at_point") is not None:
        unc_txt = f" (correction uncertainty here: +/- {state['uncertainty_at_point']:.2f} m, 1 std dev from bootstrap resampling)"

    msg = f"SAM-segmented median distance: {median_depth:.2f} m{unc_txt}"
    return overlay, msg


def export_all_data(location_text, species_text):
    if state["bg_rgb"] is None:
        return None, "Nothing to export yet -- run at least the background/depth steps first."

    calib_summary = {}
    if state["correction_method"]:
        calib_summary["correction_method"] = state["correction_method"]
        if state["correction_method"] == "scalar" and state["correction_params"]:
            a, b = state["correction_params"]
            calib_summary["scalar_a"] = a
            calib_summary["scalar_b"] = b
    if state["calibration_maes"]:
        for k, v in state["calibration_maes"].items():
            calib_summary[f"loo_mae_{k}"] = v if v is not None else "n/a"

    result_summary = {}
    if state["final_distance"] is not None:
        result_summary["final_median_distance_m"] = state["final_distance"]
    if state.get("uncertainty_at_point") is not None:
        result_summary["correction_uncertainty_std_m"] = state["uncertainty_at_point"]
    if state["target_point"]:
        result_summary["clicked_point_x"] = state["target_point"][0]
        result_summary["clicked_point_y"] = state["target_point"][1]
    if location_text:
        result_summary["location"] = location_text
    if species_text:
        result_summary["suspected_species"] = species_text

    bg_depth_corrected = None
    if state["bg_depth"] is not None and state["correction_method"]:
        bg_depth_corrected = _apply_current_correction(state["bg_depth"])

    out_path = io_utils.default_export_path()
    io_utils.bundle_results(
        out_path,
        tie_points=state["tie_points"],
        calibration_summary=calib_summary if calib_summary else None,
        bg_rgb=state["bg_rgb"], bg_depth_raw=state["bg_depth"], bg_depth_corrected=bg_depth_corrected,
        target_rgb=state["target_rgb"], target_depth_corrected=state["target_depth_corrected"],
        segmentation_overlay=state["segmentation_overlay"],
        uncertainty_map=state["uncertainty_map"],
        result_summary=result_summary if result_summary else None,
    )
    return out_path, f"Exported to {out_path}"


# ---------------- Tab 4: optional AI report ----------------

def generate_ai_report(api_key, location_text, species_text):
    context = {
        "tie_point_count": len(state["tie_points"]),
        "calibration_method": state["correction_method"],
        "final_median_distance_m": state["final_distance"],
        "correction_uncertainty_std_m": state.get("uncertainty_at_point"),
        "suspected_species": species_text,
        "location": location_text,
    }
    if state["calibration_maes"]:
        for k, v in state["calibration_maes"].items():
            context[f"loo_mae_{k}"] = v
    return llm_report.generate_report(api_key, context)


# ---------------- UI ----------------

with gr.Blocks(title="Camera-trap depth calibration") as demo:
    gr.Markdown("# Camera-trap depth calibration")
    gr.Markdown("Runs entirely on this computer. No data leaves your machine, except the optional AI report tab (Tab 4), which sends a short text summary -- not images -- to the Anthropic API if you choose to use it.")

    with gr.Tab("1. Background + tie points"):
        with gr.Row():
            bg_input = gr.Image(type="filepath", label="Upload background / calibration image")
            bg_status = gr.Textbox(label="Status", interactive=False)
        bg_load_btn = gr.Button("Run depth model on background")

        with gr.Row():
            grid_toggle = gr.Checkbox(label="Show pixel grid", value=True)
            depth_overlay_toggle = gr.Checkbox(label="Show depth overlay (helps spot camouflaged edges)", value=False)

        bg_display = gr.Image(label="Click on this image to add a tie point", interactive=False)
        with gr.Row():
            dist_input = gr.Number(label="True distance for the point you're about to click (m)", value=1.0)
            undo_btn = gr.Button("Undo last point")

        gr.Markdown("**Or import tie points from a previous session:** CSV with columns `x, y, true_dist` (or `distance`/`dist`).")
        with gr.Row():
            csv_input = gr.File(label="Import tie points CSV", file_types=[".csv"])
        tie_table = gr.Dataframe(headers=["x", "y", "true_dist_m"], label="Tie points")

        bg_load_btn.click(load_background, inputs=[bg_input], outputs=[bg_display, bg_status, tie_table])
        bg_display.select(add_tie_point, inputs=[dist_input], outputs=[bg_display, bg_status, tie_table])
        undo_btn.click(undo_last_tie_point, outputs=[bg_display, tie_table])
        csv_input.change(import_tie_points_csv, inputs=[csv_input], outputs=[bg_display, bg_status, tie_table])
        grid_toggle.change(toggle_grid, inputs=[grid_toggle], outputs=[bg_display])
        depth_overlay_toggle.change(toggle_depth_overlay, inputs=[depth_overlay_toggle], outputs=[bg_display])

    with gr.Tab("2. Calibrate"):
        calib_btn = gr.Button("Fit correction (auto-picks best method via leave-one-out)")
        calib_output = gr.Textbox(label="Calibration result", lines=8, interactive=False)
        calib_table = gr.Dataframe(headers=["x", "y", "true_dist_m"], label="Tie points used")
        calib_btn.click(run_calibration, outputs=[calib_output, calib_table])

    with gr.Tab("3. Target image + distance"):
        with gr.Row():
            target_input = gr.Image(type="filepath", label="Upload target (animal) image")
            target_status = gr.Textbox(label="Status", interactive=False)
        target_load_btn = gr.Button("Run depth + apply calibration")
        target_display = gr.Image(label="Click once on the animal", interactive=False)
        gr.Markdown("First click on any target image downloads the SAM model (~375MB, one-time) and may take a minute.")
        compute_btn = gr.Button("Segment & compute distance")
        result_display = gr.Image(label="Result overlay", interactive=False)
        result_text = gr.Textbox(label="Distance result", lines=4, interactive=False)

        gr.Markdown("---\n**Export everything from this session** (tables, intermediate images, uncertainty map).")
        with gr.Row():
            export_location = gr.Textbox(label="Location (optional, included in export)", placeholder="e.g. Serra da Estrela, Portugal")
            export_species = gr.Textbox(label="Suspected species (optional, included in export)", placeholder="e.g. roe deer")
        export_btn = gr.Button("Download all data")
        export_file = gr.File(label="Export zip")
        export_status = gr.Textbox(label="Export status", interactive=False)

        target_load_btn.click(load_target, inputs=[target_input], outputs=[target_display, target_status])
        target_display.select(handle_target_click, outputs=[target_display, target_status])
        compute_btn.click(compute_distance, outputs=[result_display, result_text])
        export_btn.click(export_all_data, inputs=[export_location, export_species], outputs=[export_file, export_status])

    with gr.Tab("4. AI report (optional)"):
        gr.Markdown(
            "Generates a short written summary of this session's calibration quality and "
            "result plausibility, using your own Anthropic API key. Your key is kept in memory "
            "only for this session -- never saved to disk. This is the only feature in the app "
            "that sends data over the internet (a short text summary, not any images)."
        )
        api_key_input = gr.Textbox(label="Anthropic API key", type="password", placeholder="sk-ant-...")
        with gr.Row():
            report_location = gr.Textbox(label="Location (optional)", placeholder="e.g. Serra da Estrela, Portugal")
            report_species = gr.Textbox(label="Suspected species (optional)", placeholder="e.g. roe deer")
        report_btn = gr.Button("Generate report")
        report_output = gr.Textbox(label="Report", lines=20, interactive=False)

        report_btn.click(generate_ai_report, inputs=[api_key_input, report_location, report_species], outputs=[report_output])

if __name__ == "__main__":
    demo.launch(inbrowser=True)
