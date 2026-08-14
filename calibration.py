"""
Tie-point calibration: turn a handful of (pixel, true_distance) reference
points into a correction that can be applied to a raw depth map.

Three methods, in increasing order of complexity -- and, with only a few
tie points, roughly DEcreasing order of reliability:

  "scalar"  -- a single RANSAC-robust linear fit (true = a * pred + b) applied
               uniformly across the whole image. This is the field-standard
               approach used in most published camera-trap depth calibration
               work (Haucke et al. 2022 and follow-ups). Most robust with few
               tie points; doesn't correct for spatial/radial bias.

  "rbf"     -- thin-plate-spline interpolation of the per-point residual
               across the image. Works with any number of points, but can
               behave oddly far from tie points.

  "kriging" -- regression kriging: fit residual ~ (predicted_depth, radial
               distance from center), then krige the leftover residual.
               Best once you have >= ~10-15 well-spread tie points; falls
               back toward the trend-only fit otherwise.

Always check the leave-one-out MAE (loo_eval) before trusting rbf/kriging
over the simpler scalar fit.
"""

import numpy as np
import cv2
from scipy.interpolate import Rbf
from sklearn.linear_model import LinearRegression, RANSACRegressor

try:
    from pykrige.ok import OrdinaryKriging
    HAVE_PYKRIGE = True
except Exception:
    HAVE_PYKRIGE = False


def patch_median(depth_map, x, y, radius=4):
    h, w = depth_map.shape
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    return float(np.median(depth_map[y0:y1, x0:x1]))


def compute_residuals(tie_points, depth_map):
    """Fill in pred_dist / residual for each tie point, in place. Returns the list."""
    for tp in tie_points:
        tp["pred_dist"] = patch_median(depth_map, tp["x"], tp["y"])
        tp["residual"] = tp["true_dist"] - tp["pred_dist"]
    return tie_points


# ---------- scalar (field-standard) ----------

def fit_scalar_correction(tie_points):
    """RANSAC-robust linear fit: true = a * pred + b. Returns (a, b)."""
    preds = np.array([tp["pred_dist"] for tp in tie_points]).reshape(-1, 1)
    trues = np.array([tp["true_dist"] for tp in tie_points])

    if len(tie_points) >= 4:
        model = RANSACRegressor(LinearRegression(), min_samples=max(2, len(tie_points) // 2))
        model.fit(preds, trues)
        a = float(model.estimator_.coef_[0])
        b = float(model.estimator_.intercept_)
    else:
        model = LinearRegression().fit(preds, trues)
        a = float(model.coef_[0])
        b = float(model.intercept_)
    return a, b


def apply_scalar_correction(depth_map, a, b):
    return a * depth_map + b


# ---------- RBF ----------

def build_correction_rbf(tie_points, shape, value_key="residual", function="thin_plate", smooth=0.0):
    xs = np.array([tp["x"] for tp in tie_points], dtype=float)
    ys = np.array([tp["y"] for tp in tie_points], dtype=float)
    vs = np.array([tp[value_key] for tp in tie_points], dtype=float)

    rbf = Rbf(xs, ys, vs, function=function, smooth=smooth)

    h, w = shape
    step = max(1, min(h, w) // 200)
    gx = np.arange(0, w, step)
    gy = np.arange(0, h, step)
    GX, GY = np.meshgrid(gx, gy)
    grid_vals = rbf(GX, GY)
    correction = cv2.resize(grid_vals.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return correction


# ---------- regression kriging ----------

def fit_regression_kriging(tie_points, depth_map, value_key="residual",
                            variogram_model="linear", grid_step=None):
    """Fits the trend model (residual ~ predicted_depth + radial_distance) and the
    spatial kriged residual, from the CALIBRATION image's tie points.

    Returns a dict: {trend_model, kriged_resid_full, cx, cy, kriging_var_full}.
    Note: kriged_resid_full is purely a function of pixel position (legitimate to
    reuse on any image with the same resolution/camera framing). trend_model is
    NOT baked into a fixed array here -- it must be evaluated against whichever
    image's own depth values you're correcting (see apply_regression_kriging),
    otherwise you end up re-imprinting the calibration image's own depth structure
    onto whatever new image you apply it to.
    """
    h, w = depth_map.shape
    cx, cy = w / 2.0, h / 2.0

    xs = np.array([tp["x"] for tp in tie_points], dtype=np.float64)
    ys = np.array([tp["y"] for tp in tie_points], dtype=np.float64)
    vs = np.array([tp[value_key] for tp in tie_points], dtype=np.float64)
    preds = np.array([tp["pred_dist"] for tp in tie_points], dtype=np.float64)
    radial = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    X = np.column_stack([preds, radial])
    trend_model = LinearRegression().fit(X, vs)
    trend_resid = (vs - trend_model.predict(X)).astype(np.float64)

    if grid_step is None:
        grid_step = max(1, min(h, w) // 200)
    gx = np.arange(0, w, grid_step, dtype=np.float64)
    gy = np.arange(0, h, grid_step, dtype=np.float64)

    kriging_var = None
    if HAVE_PYKRIGE and len(tie_points) >= 4:
        try:
            ok = OrdinaryKriging(xs, ys, trend_resid, variogram_model=variogram_model, verbose=False, enable_plotting=False)
            z_resid, ss = ok.execute("grid", gx, gy)
            z_resid = np.array(z_resid)
            kriging_var = np.array(ss)
        except Exception:
            z_resid = np.zeros((len(gy), len(gx)))
    else:
        z_resid = np.zeros((len(gy), len(gx)))

    kriged_resid_full = cv2.resize(z_resid.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    kriging_var_full = None
    if kriging_var is not None:
        kriging_var_full = cv2.resize(kriging_var.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    return {
        "trend_model": trend_model,
        "kriged_resid_full": kriged_resid_full,
        "cx": cx, "cy": cy,
        "kriging_var_full": kriging_var_full,
    }


def apply_regression_kriging(depth_map, fit):
    """Evaluates the trend against THIS image's own depth values (not frozen from
    calibration), and adds the static spatial kriged residual. Returns the
    correction array to be added to depth_map."""
    h, w = depth_map.shape
    cx, cy = fit["cx"], fit["cy"]
    yy, xx = np.mgrid[0:h, 0:w]
    radial = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    trend_full = fit["trend_model"].predict(
        np.column_stack([depth_map.ravel(), radial.ravel()])
    ).reshape(h, w).astype(np.float32)

    return trend_full + fit["kriged_resid_full"]


# ---------- validation ----------

def loo_eval_scalar(tie_points):
    errs = []
    for i in range(len(tie_points)):
        train = tie_points[:i] + tie_points[i+1:]
        held = tie_points[i]
        a, b = fit_scalar_correction(train)
        pred_final = a * held["pred_dist"] + b
        errs.append(abs(pred_final - held["true_dist"]))
    return float(np.mean(errs)), errs


def loo_eval_surface(tie_points, depth_map, builder_fn, **kwargs):
    errs = []
    for i in range(len(tie_points)):
        train = tie_points[:i] + tie_points[i+1:]
        held = tie_points[i]
        out = builder_fn(train, depth_map, **kwargs) if builder_fn is not build_correction_rbf else \
              builder_fn(train, depth_map.shape, **kwargs)
        correction = out[0] if isinstance(out, tuple) else out
        pred_corr = correction[held["y"], held["x"]]
        pred_final = held["pred_dist"] + pred_corr
        errs.append(abs(pred_final - held["true_dist"]))
    return float(np.mean(errs)), errs


def loo_eval_kriging(tie_points, depth_map):
    """LOO for kriging, evaluated on the calibration image itself (self-consistent --
    trend is evaluated against depth_map's own values, same as apply_regression_kriging
    would do when reusing this fit on a new image later)."""
    errs = []
    for i in range(len(tie_points)):
        train = tie_points[:i] + tie_points[i+1:]
        held = tie_points[i]
        fit = fit_regression_kriging(train, depth_map)
        correction = apply_regression_kriging(depth_map, fit)
        pred_corr = correction[held["y"], held["x"]]
        pred_final = held["pred_dist"] + pred_corr
        errs.append(abs(pred_final - held["true_dist"]))
    return float(np.mean(errs)), errs


def choose_best_method(tie_points, depth_map):
    """Run all applicable methods, LOO-validate, return (best_name, mae_dict)."""
    compute_residuals(tie_points, depth_map)
    maes = {}

    a, b = fit_scalar_correction(tie_points)
    maes["scalar"] = loo_eval_scalar(tie_points)[0] if len(tie_points) >= 4 else None

    if len(tie_points) >= 4:
        maes["rbf"] = loo_eval_surface(tie_points, depth_map, build_correction_rbf)[0]
        maes["kriging"] = loo_eval_kriging(tie_points, depth_map)[0]
    else:
        maes["rbf"] = None
        maes["kriging"] = None

    valid = {k: v for k, v in maes.items() if v is not None}
    best = min(valid, key=valid.get) if valid else "scalar"
    return best, maes


# ---------- uncertainty ----------

def bootstrap_uncertainty(tie_points, depth_map, method, n_boot=25, downsample=4, random_state=0):
    """Estimate uncertainty of the fitted correction by bootstrap-resampling the tie
    points, refitting, and measuring the spread of the resulting correction across
    resamples. Returns (std_map_full_res, mean_map_full_res).

    Works for all three methods. Runs on a downsampled grid for speed, then upsamples
    the resulting std/mean maps back to full resolution (the uncertainty surface is
    smooth, so this loses little).

    With very few tie points (<4), bootstrap resampling can't meaningfully vary the
    fit -- results should be treated as a rough lower bound, not a precise figure.
    """
    rng = np.random.RandomState(random_state)
    h, w = depth_map.shape
    small_depth = depth_map[::downsample, ::downsample]
    n = len(tie_points)

    corrections = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        sample = [tie_points[i] for i in idx]
        # ensure at least 2 distinct points, else skip this bootstrap draw
        if len(set((tp["x"], tp["y"]) for tp in sample)) < 2:
            continue
        try:
            if method == "scalar":
                a, b = fit_scalar_correction(sample)
                corr_small = (a - 1.0) * small_depth + b
            elif method == "rbf":
                corr_small = build_correction_rbf(sample, small_depth.shape)
            else:  # kriging
                fit = fit_regression_kriging(sample, small_depth)
                corr_small = apply_regression_kriging(small_depth, fit)
        except Exception:
            continue
        corrections.append(corr_small)

    if len(corrections) < 3:
        zeros = np.zeros_like(depth_map)
        return zeros, zeros

    stack = np.stack(corrections, axis=0)
    std_small = stack.std(axis=0)
    mean_small = stack.mean(axis=0)

    std_full = cv2.resize(std_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    mean_full = cv2.resize(mean_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return std_full, mean_full
