# Camera-trap depth calibration tool

A local desktop app (Gradio-based, runs in a browser window on your own machine —
nothing is uploaded anywhere) implementing the calibration pipeline: load a
background/calibration image, click tie points with known field-measured
distances, fit a correction, then load a target image with the animal and get
a calibrated median distance (whole-box or GrabCut-segmented).

## Quick start (testing on any machine with Python)

```
pip install -r requirements.txt
python app.py
```

This opens a browser window at `http://127.0.0.1:7860`. Recommended to run this
first, exactly like this, before attempting to package it as a standalone
`.exe` — much easier to debug UI/logic issues here than inside a PyInstaller
build.

The first time you run the depth model, it needs internet access once to
download weights (~1.3 GB) from HuggingFace. After that they're cached in
`~/.cache/huggingface` and no internet is needed.

## How to use it

1. **Tab 1 — Background + tie points**: upload your calibration/background
   image, click "Run depth model". Two toggles help with placing points
   precisely:
   - **Show pixel grid** — overlays coordinate gridlines and tick labels,
     for reading off approximate pixel positions.
   - **Show depth overlay** — blends the depth colormap over the RGB image,
     which helps spot where the depth model sees an edge/discontinuity even
     when the color doesn't change (e.g. an animal against dense green
     foliage that looks uniform in plain RGB).
   Set the true distance in the number box *before* each click on the image
   (it's used for the click that follows). Use "Undo last point" if you
   misclick.
2. **Tab 2 — Calibrate**: click "Fit correction". It automatically compares
   the scalar (field-standard linear), RBF, and regression-kriging methods
   via leave-one-out validation and picks the best one. Read the reported
   MAE numbers — if they're all similar, the simpler scalar method is a safer
   bet than it looks like it should be.
3. **Tab 3 — Target + distance**: upload the image with the animal in it,
   click "Run depth + apply calibration", then click **once** directly on
   the animal. Click "Segment & compute distance" — this runs Segment
   Anything (SAM) from that single point to isolate the animal's body, then
   reports the median calibrated depth within that mask, along with an
   uncertainty estimate (from bootstrap-resampling the tie points and
   refitting — the reported number is the standard deviation of the
   correction at that exact pixel across resamples; treat it as a rough
   lower bound on real-world error, not a precise figure, especially with
   fewer than ~8-10 tie points). The very first click on any target image
   downloads the SAM checkpoint (~375MB, one-time) and computes an image
   embedding (a few seconds on CPU) — after that, re-clicking on the same
   image is fast.

   **Download all data** at the bottom of this tab exports a zip with every
   table (tie points, calibration summary, result summary) and every
   intermediate image (raw/corrected depth maps, segmentation overlay,
   uncertainty map) from the session — useful for record-keeping or
   reviewing a batch of detections later.

4. **Tab 4 — AI report (optional)**: enter your own Anthropic API key (kept
   in memory for the session only, never written to disk) to get a short
   written summary of calibration quality, distance plausibility, and — if
   you fill in a suspected species and location — a general-knowledge sanity
   check of whether that species is plausible there. This is **not** a
   verified range-map lookup, just a reasonableness check; treat it as a
   second opinion, not a determination.

Important: the target image should come from the **same, unmoved camera** as
the calibration image. If the camera was bumped or repositioned between
sessions, recalibrate with a fresh background image instead of reusing an old
correction.

## Packaging as a Windows .exe for a partner

This has to be built ON a Windows machine — PyInstaller produces
platform-specific binaries, it can't cross-compile from another OS.

1. Copy this whole folder to the Windows machine (or the partner's machine).
2. Run `python app.py` there first, and confirm it works normally in the
   browser, before packaging.
3. Double-click `build_windows.bat` (or run it from a command prompt). It
   creates a virtual environment, installs dependencies, and runs
   PyInstaller. This step is slow (torch + transformers are large) — expect
   10-20+ minutes.
4. Find the output in `dist\DepthCalibTool\`. Zip that **entire folder** (not
   just the .exe — this is a `--onedir` build, the exe needs the files next
   to it) and send it to your partner.
5. Your partner unzips it anywhere and double-clicks `DepthCalibTool.exe`. A
   browser window should open automatically. First launch will be slow
   (unpacking + first depth-model download if not pre-cached — see below).

### Optional: fully offline .exe (no internet needed even on first run)

By default the .exe still needs internet the very first time it's run, to
download depth-model weights from HuggingFace. If your partner's machine has
no internet access, pre-download the weights before building:

```
python -c "from depth_model import load_depth_model; load_depth_model()"
```

Then add this line to the PyInstaller command in `build_windows.bat`, right
after `--collect-all matplotlib`:

```
--add-data "%USERPROFILE%\.cache\huggingface;huggingface_cache"
```

This roughly doubles the packaged size (weights are ~1.3 GB) but removes the
internet dependency entirely. Ask if you want help wiring the app to read
from that bundled cache path instead of the default one — it needs one extra
environment-variable line in `app.py` (`HF_HOME`) before the model loads.

## Known issues

- `requirements.txt` installs `segment-anything` directly from GitHub via
  pip's git support — this requires **git** to be installed and on PATH on
  whatever machine runs `pip install -r requirements.txt`. If pip fails on
  that line specifically, install git first (https://git-scm.com/download/win)
  and retry.
- If you see `AttributeError: module 'matplotlib.cm' has no attribute
  'get_cmap'`, that means a newer matplotlib version removed the old API —
  already fixed in `depth_model.py` in this version, but worth knowing if
  you pull in an older copy of this file.

## Known limitations

- CPU-only: works, but expect several seconds per depth inference — fine for
  single images, not built for batch-processing many frames.
- Camera movement between calibration and target sessions isn't detected
  automatically in this version — the earlier notebook has a SIFT/ORB-based
  registration check that could be ported in if that becomes a real problem.
- Tie-point spatial spread still matters — if your field points are
  collinear (all straight out from the camera), the calibration won't
  correct well for off-axis positions. The scalar method sidesteps this by
  not modeling spatial position at all, which is part of why it's the safer
  default with few points.
