#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# eumovie.py - A program to create a movie from Euclid MER stacks (or any still TIFF image)

# MIT License

# Copyright (c) [2026] [Mischa Schirmer]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# eumovie.py — GPU-accelerated zoom/pan/rotate/tilt movie renderer
#              for very large TIFF astronomical images (e.g. 19200×19200).
#
# ── Overview ──────────────────────────────────────────────────────────────────
# Reads a large TIFF image, uploads it once to GPU VRAM, then renders a movie
# by extracting crops at different positions/scales/angles driven by keyframes.
# All geometry (resize, warp, rotate) happens on the GPU. The CPU only handles
# the ffmpeg pipe write, which is overlapped with the next frame's GPU work via
# double-buffering.
#
# Two rendering modes are supported:
#   Flat mode    : standard rectilinear perspective output at any resolution.
#   Fulldome mode: azimuthal equidistant fisheye for planetarium dome projection.
#
# ── Dependencies ──────────────────────────────────────────────────────────────
#   numpy, cupy-cuda13x, tifffile, opencv-python (built with CUDA), scipy
#   ffmpeg (system, with h264_nvenc / hevc_nvenc hardware encoders)
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   Generate keyframes template (first-time setup):
#     eumovie --generate-keyframes
#
#   Flat output (default 1920×1080, h264_nvenc):
#     eumovie --input TILE.tif [--resolution 2k|4k|WxH] [--fps 60] [--cq 15]
#
#   Flat output with zoom modifier (20% more magnification):
#     eumovie --input TILE.tif --zoom 1.2
#
#   Flat output at half speed:
#     eumovie --input TILE.tif --speed 0.5
#
#   Flat output, ProRes 4444 master:
#     eumovie --input TILE.tif --resolution 4k --prores [--bits-per-mb 1000]
#
#   Fulldome 4K (4096×4096, hevc_nvenc):
#     eumovie --input TILE.tif --fulldome 4k [--fps 30] [--cq 15] [--zoom 1.2]
#
#   Fulldome 8K, ProRes 4444 master:
#     eumovie --input TILE.tif --fulldome 8k --prores [--fps 30] [--bits-per-mb 1000]
#
# ── Keyframes ────────────────────────────────────────────────────────────────
# Camera paths are defined in a user-editable keyframes.py. To create a
# template with full documentation, run:
#
#   eumovie --generate-keyframes
#
# This writes keyframes.py to the current directory and to
# ~/.config/eumovie/keyframes.py. eumovie loads ./keyframes.py first,
# falling back to ~/.config/eumovie/keyframes.py.
#
# Keyframe tuple: (t, cx, cy, zoom, angle, tilt, bank)
#   t     : time in seconds
#   cx,cy : centre of view in FITS coordinates (origin lower-left, y upward)
#   zoom  : fraction of the image short axis visible (1.0=fully out)
#   angle : roll  — in-plane rotation, CCW positive (degrees)
#   tilt  : pitch — camera pitches backward; upper edge recedes (degrees)
#   bank  : bank  — camera banks left; right side recedes (degrees)
#
# ── Flat mode — performance notes ─────────────────────────────────────────────
# All GpuMat objects are pre-allocated once at startup and reused via ROI views.
# Per-frame GPU allocation causes VRAM fragmentation and OOM after ~1000 frames.
# Pinned (page-locked) CPU buffers are pre-allocated for fast H2D/D2H transfers:
#   pinned_crop_buf : holds the full-res crop before H2D upload to gm_patch
#   pinned_out_bufs : two output frame buffers for double-buffering
#
# warpPerspective/warpAffine are NOT used in flat mode. Both leak ~30 MB VRAM
# per call despite dst= being pre-allocated (OpenCV CUDA bug, confirmed in
# production). The warp is implemented via cv2.cuda.remap with pre-allocated
# maps; the inverse homography map is computed on the GPU via CuPy each frame.
#
# ── Fulldome mode ─────────────────────────────────────────────────────────────
# Output is a square frame (4096×4096 or 8192×8192) with the azimuthal
# equidistant fisheye projection inscribed as a circle. Centre = zenith,
# edge = horizon (180° total FOV). The fisheye map is computed once per frame
# by a single CuPy ElementwiseKernel (zero intermediate allocations).
# The remap goes directly from the full-resolution source image to the output —
# no crop/resize pipeline. This avoids ~2 GB of VRAM for intermediate buffers.
#
# ── Zoom convention ───────────────────────────────────────────────────────────
# zoom_native = max(out_w/img_w, out_h/img_h) is the zoom value at which one
# source pixel maps to one output pixel. Keyframe zoom values are
# resolution-independent: zoom=0.1 means the same field of view at 1080p or 4K.

import argparse
import os
import subprocess
import sys
import threading
import numpy as np
import cupy as cp
import cv2
import tifffile
from scipy.interpolate import CubicSpline, PchipInterpolator
import importlib.util
import shutil
from importlib.metadata import version as _pkg_version, PackageNotFoundError

def _load_keyframes():
    """
    Locate and load keyframes.py, returning the get_config function.

    Search order (first found wins):
      1. ./keyframes.py                 — current working directory
      2. ~/.config/eumovie/keyframes.py — user's personal tile library

    On first run, if neither exists, the built-in default is copied to
    ~/.config/eumovie/keyframes.py so the user has a starting point to edit.
    """
    config_dir   = os.path.join(os.path.expanduser("~"), ".config", "eumovie")
    config_file  = os.path.join(config_dir, "keyframes.py")
    local_file   = os.path.join(os.getcwd(), "keyframes.py")

    if os.path.exists(local_file):
        keyframes_path = local_file
        print(f"Using local keyframes: {local_file}")
    elif os.path.exists(config_file):
        keyframes_path = config_file
    else:
        print("No keyframes.py found.")
        print("Run:  eumovie --generate-keyframes")
        print("to create a template in the current directory and in ~/.config/eumovie/")
        sys.exit(1)

    spec   = importlib.util.spec_from_file_location("keyframes", keyframes_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config


# ── Fulldome projection kernel ──────────────────────────────────────────────────────────────────────────────
# Single-pass CuPy ElementwiseKernel: azimuthal equidistant fisheye → source coords.
# Replaces ~12 separate trig kernel launches, cutting memory traffic from
# ~3 GB/frame to ~64 MB/frame at 8K. All intermediates stay in registers.
# Fulldome kernel compiled lazily on first use (avoids hanging at import time).
_fulldome_kernel = None
_CUDA_BODY = (
    "    float r_pix  = sqrtf(px*px + py*py);\n"
    "    float r_norm = r_pix / R;\n"
    "    if (r_norm > 1.0f) { map_x = -1.0f; map_y = -1.0f; return; }\n"
    "    float theta = r_norm * 1.5707963f;\n"
    "    float phi   = atan2f(py, px);\n"
    "    float dx =  sinf(theta) * cosf(phi);\n"
    "    float dy = -sinf(theta) * sinf(phi);\n"
    "    float dz =  cosf(theta);\n"
    "    float dx2 =  cr*dx - sr*dy;  float dy2 = sr*dx + cr*dy;  float dz2 = dz;\n"
    "    float dx3 =  dx2;  float dy3 = ct*dy2 - st*dz2;  float dz3 = st*dy2 + ct*dz2;\n"
    "    float dx4 =  cb*dx3 + sb*dz3;  float dy4 = dy3;  float dz4 = -sb*dx3 + cb*dz3;\n"
    "    if (dz4 <= 0.0f) { map_x = -1.0f; map_y = -1.0f; return; }\n"
    "    map_x = cx_img + f_dome * dx4 / dz4;\n"
    "    map_y = cy_img + f_dome * dy4 / dz4;\n"
)


# ── Tunable constants ──────────────────────────────────────────────────────────

# TENSION controls the "gravitational feel" of zoom, angle, tilt, and bank.
# cx and cy always use pure PCHIP regardless of this setting — CubicSpline on
# position causes kinks through hold segments where cx,cy are identical across
# consecutive keyframes.
# 0.0 = pure PCHIP for all parameters: stiff, no overshoot, kink-free.
# 0.5 = recommended: PCHIP for position, 50/50 blend for dynamic params —
#       smooth inertial feel without risk of zoom going negative.
# 1.0 = pure CubicSpline for dynamic params: maximum gravitational feel,
#       zoom may briefly overshoot keyframe values (clamp at 1e-4 catches negatives).
TENSION = 0.1

# CAMERA_FOV sets the virtual vertical field of view of the camera (degrees).
# This determines the focal length f = (canvas_h/2) / tan(FOV/2), which controls
# how strongly tilt and bank angles translate into perspective distortion.
# At tilt = CAMERA_FOV/2 the horizon reaches the frame edge.
# 60° = moderate cinematic feel; 90° = strong wide-angle perspective.
# NOTE: this constant appears in three places — the homography (via f), the crop
# inflation factor, and the max_canvas sizing. All three derive from this single
# constant so changing it here propagates correctly.
CAMERA_FOV = 60.0


def parse_arguments():
    try:
        current_version = _pkg_version("eumovie")
    except PackageNotFoundError:
        current_version = "dev"

    print(f"\n   eumovie v{current_version} (Mischa Schirmer)\n")

    parser = argparse.ArgumentParser(
        description="GPU-accelerated zoom/pan/rotate movie renderer for large TIFF images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── General options ────────────────────────────────────────────────────
    parser.add_argument("--input",   required=False, default=None,
                        help="Input TIFF file (uint8 or uint16 RGB)")
    parser.add_argument("--generate-keyframes", action="store_true",
                        default=False, dest="generate_keyframes",
                        help="Write a template keyframes.py to the current directory "
                             "and to ~/.config/eumovie/, then exit.")
    parser.add_argument("--fps",     type=int, default=60,
                        help="Output frame rate")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="Zoom modifier applied to all keyframe zoom values. "
                             ">1 = more magnification (e.g. 1.2 = 20%% closer), "
                             "<1 = wider field of view. "
                             "Keyframe zoom values are divided by this factor.")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed modifier applied to all keyframe timestamps. "
                             ">1 = faster (e.g. 2.0 = twice as fast), "
                             "<1 = slower (e.g. 0.5 = half speed). "
                             "Keyframe timestamps are divided by this factor.")
    parser.add_argument("--cq",      type=int, default=15,
                        help="H.264/H.265 encoder quality (12=very high, 15=high, 18=medium)")
    parser.add_argument("--threads", type=int, default=min(max(os.cpu_count() - 1, 1), 16),
                        help="FFmpeg encoding threads [min(cpu_count-1, 16)]")

    # ── Output format — mutually exclusive: flat vs fulldome ──────────────
    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument("--resolution", default=None,
                        metavar="RES",
                        help="Flat output resolution. Named presets: 2k (1920×1080), 4k (3840×2160). "
                             "Or specify as WxH, e.g. 2560x1440. Default: 1920x1080.")
    size_group.add_argument("--fulldome", choices=["4k", "8k"], default=None,
                        help="Fulldome azimuthal equidistant fisheye: 4k=4096×4096, 8k=8192×8192")

    # ── Output quality — apply to both flat and fulldome ──────────────────
    parser.add_argument("--prores", action="store_true", default=False,
                        help="Encode as Apple ProRes 4444 .mov (10-bit 4:4:4, CPU). "
                             "Recommended for planetarium delivery and professional editing. "
                             "Works with both flat and fulldome output.")
    parser.add_argument("--bits-per-mb", type=int, default=1000, dest="bits_per_mb",
                        help="ProRes quality in bits per macroblock (only with --prores). "
                             "500=proxy, 1000=broadcast, 2000=high-end master")

    import sys
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # Validate and set defaults
    # Resolve --resolution into args.width / args.height.
    _PRESETS = {"2k": (1920, 1080), "4k": (3840, 2160)}
    if args.resolution is not None:
        res = args.resolution.lower()
        if res in _PRESETS:
            args.width, args.height = _PRESETS[res]
        elif "x" in res:
            try:
                w, h = res.split("x")
                args.width, args.height = int(w), int(h)
            except ValueError:
                parser.error(f"--resolution: cannot parse '{args.resolution}'. "
                             "Use a preset (2k, 4k) or WxH format (e.g. 2560x1440).")
        else:
            parser.error(f"--resolution: unrecognised value '{args.resolution}'. "
                         "Use a preset (2k, 4k) or WxH format (e.g. 2560x1440).")
    elif not args.fulldome:
        args.width, args.height = 1920, 1080   # default: 2K

    # Set fulldome dimensions so args.width/height are always valid integers.
    if args.fulldome == "4k":
        args.width = args.height = 4096
    elif args.fulldome == "8k":
        args.width = args.height = 8192

    if not args.generate_keyframes and args.input is None:
        parser.error("--input is required unless --generate-keyframes is used")
    return args


def build_interpolators(keyframes, tension=TENSION):
    """
    Build interpolators for all six camera parameters.

    cx and cy use pure PCHIP regardless of tension. PCHIP is monotonic and
    never overshoots, which is essential for position: any CubicSpline blend
    causes the camera to drift backward through position-hold segments (where
    consecutive keyframes share the same cx,cy), producing visible direction
    kinks on the way in and out of each hold.

    zoom, angle, tilt, and bank use CubicSpline when tension > 0.
    These parameters are always changing between keyframes (no flat holds), so
    CubicSpline's natural overshoot is safe and desirable — it gives the camera
    motion a sense of inertia and weight that pure PCHIP lacks.

    tension=0.0: all six parameters use PCHIP (stiff but kink-free everywhere).
    tension=1.0: cx/cy still PCHIP; zoom/angle/tilt/bank use pure CubicSpline
                 (maximum gravitational feel, may overshoot zoom slightly).
    tension=0.5: cx/cy PCHIP; others blended 50/50 (recommended default).

    Returns six callables (cx, cy, zoom, angle, tilt, bank), each
    accepting a scalar or numpy array of time values.
    """
    t            = np.array([k[0] for k in keyframes])
    cx           = np.array([k[1] for k in keyframes], dtype=float)
    cy           = np.array([k[2] for k in keyframes], dtype=float)
    zoom         = np.array([k[3] for k in keyframes], dtype=float)
    angle        = np.array([k[4] for k in keyframes], dtype=float)
    tilt         = np.array([k[5] for k in keyframes], dtype=float)
    bank         = np.array([k[6] for k in keyframes], dtype=float)

    def dynamic(param):
        """zoom/angle/tilt/bank: blend PCHIP and CubicSpline by tension."""
        pchip  = PchipInterpolator(t, param)
        if tension <= 0.0:
            return pchip
        spline = CubicSpline(t, param)
        if tension >= 1.0:
            return spline
        def blended(t_eval):
            return (1.0 - tension) * pchip(t_eval) + tension * spline(t_eval)
        return blended

    return (
        PchipInterpolator(t, cx),           # position: always PCHIP
        PchipInterpolator(t, cy),           # position: always PCHIP
        dynamic(zoom),
        dynamic(angle),
        dynamic(tilt),
        dynamic(bank),
    )



def build_fulldome_map(out_size, zoom, cx_img, cy_img, angle_deg, tilt_deg, bank_deg,
                       img_w, img_h, scale,
                       cp_px, cp_py,
                       cp_map_x_full, cp_map_y_full):
    global _fulldome_kernel
    if _fulldome_kernel is None:
        print("Compiling fulldome GPU kernel ...", flush=True)
        _fulldome_kernel = cp.ElementwiseKernel(
            'float32 px, float32 py, '
            'float32 R, float32 f_dome, float32 cx_img, float32 cy_img, '
            'float32 cr, float32 sr, float32 ct, float32 st, float32 cb, float32 sb',
            'float32 map_x, float32 map_y',
            _CUDA_BODY, '_fulldome_proj')
        print("Kernel ready.", flush=True)
    """
    Build the fulldome remap map for one frame using a single GPU kernel.

    Azimuthal equidistant fisheye: centre=zenith, edge=horizon (90°).
    cp_px, cp_py: pre-allocated (out_size*out_size,) float32 arrays holding
    pixel offsets from frame centre (filled once at startup, reused every frame).
    cp_map_x/y_full: zero-copy CuPy views into gm_map_x/y VRAM.
    The kernel writes src coords directly into GpuMat VRAM in one pass.
    """
    R      = cp.float32(out_size // 2)
    scale_ = min(img_w / out_size, img_h / out_size)
    f_dome = cp.float32(R / (zoom * scale_))

    roll_r = np.radians(angle_deg)
    tilt_r = np.radians(-tilt_deg)   # negated: image-y down, our y up
    bank_r = np.radians(bank_deg)
    cr, sr = cp.float32(np.cos(roll_r)), cp.float32(np.sin(roll_r))
    ct, st = cp.float32(np.cos(tilt_r)), cp.float32(np.sin(tilt_r))
    cb, sb = cp.float32(np.cos(bank_r)), cp.float32(np.sin(bank_r))

    N = out_size * out_size
    map_x_flat = cp_map_x_full[:out_size, :out_size].ravel()
    map_y_flat = cp_map_y_full[:out_size, :out_size].ravel()

    _fulldome_kernel(
        cp_px, cp_py,
        R, f_dome, cp.float32(cx_img), cp.float32(cy_img),
        cr, sr, ct, st, cb, sb,
        map_x_flat, map_y_flat)


def render_frame_gpu(gpu_image, pinned_crop_buf, out_pinned_buf,
                     gm_patch, gm_small, gm_warp_out, max_canvas,
                     gm_map_x, gm_map_y,
                     cp_map_x_full, cp_map_y_full,
                     cp_coords, cp_mapped, cp_xs_full, cp_ys_full,
                     cx, cy, zoom, angle_deg, tilt_deg, bank_angle_deg,
                     out_w, out_h, img_h, img_w, stream):
    """
    Render one frame entirely on the GPU and kick off an async D2H download.

    All GpuMat arguments (gm_patch, gm_small, gm_warp_out) are pre-allocated
    by the caller and reused via ROI views — NO per-frame GPU allocation occurs.
    This is critical: per-frame allocation caused VRAM fragmentation and OOM
    after ~1000 frames in testing.

    The download into out_pinned_buf is ASYNCHRONOUS — the caller must call
    stream.waitForCompletion() before reading the result. In the double-buffered
    loop this wait happens in the pipe_write thread, not the main thread.

    Pipeline steps:
      1. Crop at full resolution      — zero-copy CuPy slice of gpu_image
      2. Pad if out of bounds         — CPU pinned memset+blit (safety net only; cx/cy
      #                                  clamping in step 1 prevents this firing normally)
      3. H2D transfer via pinned RAM  — crop only, not the full image
      4. Resize to canvas size        — cv2.cuda.resize into gm_small ROI
      5. Warp (rotate/tilt)           — cv2.cuda.warpAffine or warpPerspective
                                        into pre-allocated gm_warp_out
      6. Centre-crop to output size   — zero-copy GpuMat ROI
      7. Async D2H download           — into page-locked out_pinned_buf

    ── Crop inflation for black-corner-free warping ──────────────────────────
    A naive implementation crops exactly the output region, then rotates/tilts
    it. This causes black corners because the rotated frame samples outside
    the cropped area. The fix is to crop a larger region upfront so the warp
    always has real image data to fill the corners.

    For rotation by angle a on a W×H output rectangle, the bounding box of the
    rotated rectangle has dimensions:
        W' = W*|cos(a)| + H*|sin(a)|
        H' = W*|sin(a)| + H*|cos(a)|
    The per-axis inflation factors are crop_factor_w = W'/W and crop_factor_h = H'/H.
    These are applied independently: crop_w uses crop_factor_w, crop_h uses
    crop_factor_h, and canvas_w/canvas_h likewise. This preserves the correct
    source aspect ratio so the resize step introduces no distortion.
    Using a single max() factor (the old approach) over-inflated one axis,
    making the crop the wrong shape and causing squeezing/stretching during
    rotation, plus frame-to-frame jitter as max() switched between terms.

    For tilt/bank by angle t, the crop inflation factor is
    1 / (1 - sin(t) / sin(FOV/2)), using the larger of tilt and bank.

    Both factors combine multiplicatively per axis. The inflated crop is then
    clamped to the image bounds. cx/cy are also clamped (step 1) so the crop
    window never extends outside the image — the pad branch (step 2) is a
    safety net only.

    ── Canvas vs output size ─────────────────────────────────────────────────
    The crop is resized to canvas_w × canvas_h (= out_w*crop_factor_w ×
    out_h*crop_factor_h), not to out_w × out_h directly. The warp operates on
    this larger canvas. The final output frame is extracted as a centre ROI of
    size out_w × out_h from the warped canvas. This is why the warp never
    produces black corners when the crop has enough surrounding data.
    """
    needs_warp = abs(angle_deg) > 0.01
    needs_tilt = abs(tilt_deg) > 0.01 or abs(bank_angle_deg) > 0.01

    # scale converts from output pixels to source image pixels at zoom=1.
    # Using min() ensures the limiting axis (shorter side) fills the output
    # without distortion regardless of aspect ratio.
    scale = min(img_w / out_w, img_h / out_h)

    # Compute per-axis crop inflation factors (see docstring above for derivation).
    #
    # crop_factor_w and crop_factor_h are kept separate so that crop_w/crop_h
    # and canvas_w/canvas_h always have the correct source aspect ratio.
    # A single shared factor (old: max of both) made the crop the wrong shape,
    # causing aspect-ratio distortion during rotation and frame-to-frame jitter
    # as the max() switched between the two terms at different angles.
    if needs_warp:
        a      = abs(np.radians(angle_deg))
        ca, sa = abs(np.cos(a)), abs(np.sin(a))
        # Bounding box of the rotated W×H rectangle, normalised per axis.
        crop_factor_w = (out_w * ca + out_h * sa) / out_w   # = W'/W
        crop_factor_h = (out_w * sa + out_h * ca) / out_h   # = H'/H
    else:
        crop_factor_w = crop_factor_h = 1.0

    # Tilt is symmetric (shifts both edges equally) so one factor covers both axes.
    max_tilt_angle = max(abs(tilt_deg), abs(bank_angle_deg))
    fov_half_sin   = np.sin(np.radians(CAMERA_FOV / 2.0))
    tilt_factor    = (min(1.0 / (1.0 - np.sin(np.radians(max_tilt_angle)) / fov_half_sin), 4.0)
                      if needs_tilt else 1.0)

    crop_factor_w *= tilt_factor
    crop_factor_h *= tilt_factor

    # Canvas size — computed first so crop can be derived from it.
    # Each axis clamped to [out_w/h, max_canvas] so the centre-crop ROI always
    # fits (canvas < out crashes; canvas > max_canvas writes past the GpuMat).
    # The lower clamp (canvas >= out) is needed when crop_factor < 1, which
    # occurs for the narrow axis near ±90° rotation on a non-square output.
    canvas_w = max(min(int(out_w * crop_factor_w), max_canvas), out_w)
    canvas_h = max(min(int(out_h * crop_factor_h), max_canvas), out_h)

    # Crop size in source image pixels, derived from the canvas dimensions.
    #
    # The resize step maps crop_w → canvas_w and crop_h → canvas_h. For no
    # aspect-ratio distortion the resize ratio must be identical on both axes:
    #   canvas_w / crop_w == canvas_h / crop_h
    # i.e.  crop_w / crop_h == canvas_w / canvas_h.
    #
    # The old approach computed crop_w and crop_h independently using
    # crop_factor_w/h, then clamped canvas independently. When the canvas clamp
    # changed one axis (e.g. forcing canvas_w up to out_w near ±90°), the crop
    # aspect ratio no longer matched the canvas aspect ratio, causing the resize
    # to stretch one axis — visible as 10-20% vertical distortion.
    #
    # Fix: compute the unclamped crop size from zoom*scale (uniform across axes),
    # then scale it to exactly match the canvas aspect ratio. The reference size
    # uses the h-axis (crop_factor_h is always ≥ 1 for landscape outputs, so it
    # is never affected by the lower canvas clamp and is the stable reference).
    # crop_w is then derived so that crop_w/crop_h == canvas_w/canvas_h exactly.
    # Both are then clamped to image bounds and to a minimum of 1.
    crop_h = max(min(int(zoom * out_h * scale * crop_factor_h), img_h), 1)
    crop_w = max(min(int(crop_h * canvas_w / canvas_h), img_w), 1)

    # ── 1. Full-resolution crop — zero-copy CuPy slice ────────────────────────
    # gpu_image[y0c:y1c, x0c:x1c] is a view, not a copy — no GPU memory moved.
    #
    # Do NOT clamp cx/cy to keep the crop in-bounds here. Clamping silently
    # shifts the actual crop centre away from the intended subject whenever the
    # crop window is large and cx/cy is near the image edge. Since the warp
    # rotation pivot is the canvas centre, which maps back to cx/cy in the
    # source image, any displacement of the crop centre also displaces the
    # rotation pivot — producing visible position jumps as the clamp activates
    # and deactivates during a zoom-out (seen at t=2-5.5s and t=62-65.5s in
    # testing with cx=4540 and cx=18205 near the left/right image edges).
    #
    # Instead, allow the crop to extend partially outside the image. The clamped
    # coordinates x0c..x1c clip to the valid region; the pad branch (step 2)
    # zero-fills the out-of-bounds border in CPU pinned memory at no VRAM cost.
    # The only hard constraint: at least 1 pixel of the crop must be in-bounds,
    # which is guaranteed as long as cx/cy themselves are within [0, img_w/h].
    cx = float(np.clip(cx, 0, img_w))
    cy = float(np.clip(cy, 0, img_h))

    x0 = int(round(cx - crop_w / 2))
    y0 = int(round(cy - crop_h / 2))
    x1 = x0 + crop_w
    y1 = y0 + crop_h

    x0c = max(x0, 0);  y0c = max(y0, 0)
    x1c = min(x1, img_w);  y1c = min(y1, img_h)
    patch_gpu = gpu_image[y0c:y1c, x0c:x1c]

    # ── 2. Pad if crop extends outside image bounds ────────────────────────────
    # Triggered whenever cx/cy is near an image edge and the crop window extends
    # outside. This is intentionally allowed (cx/cy are NOT clamped to keep the
    # crop in-bounds) because clamping shifts the rotation pivot away from the
    # subject, producing visible position jumps during zoom-out near edges.
    #
    # We do the padding entirely in CPU pinned memory, then upload in one shot.
    # pinned_crop_buf is pre-allocated at img_h*img_w*3 bytes — always large
    # enough since crop_w/h are clamped to image dims. This avoids any new VRAM
    # allocation (the old cp.pad path created a brand-new CuPy array the size of
    # the padded crop, causing a spike when gpu_image + pad result coexisted).
    #
    # Why not cv2.cuda.copyMakeBorder? src and dst would both be ROIs of gm_patch
    # starting at (0,0), so they alias — OpenCV CUDA does not guarantee correct
    # behaviour for overlapping src/dst, and it produced the interference pattern.
    pl = x0c - x0;  pr = x1 - x1c
    pt = y0c - y0;  pb = y1 - y1c
    if pl or pr or pt or pb:
        # Build the padded frame in pinned RAM: zero-fill the full region,
        # then blit the valid inner patch into the correct offset.
        # np.frombuffer is a zero-copy view into the pre-allocated pinned buffer.
        padded_view = np.frombuffer(pinned_crop_buf, dtype=np.uint8,
                                    count=crop_h * crop_w * 3).reshape(crop_h, crop_w, 3)
        padded_view[:] = 0
        inner_np = cp.asnumpy(patch_gpu)   # D2H: only the clamped inner region
        padded_view[pt:pt + inner_np.shape[0],
                    pl:pl + inner_np.shape[1]] = inner_np
        ch, cw = crop_h, crop_w
        cw = min(cw, img_w);  ch = min(ch, img_h)
        gm_patch_roi = cv2.cuda_GpuMat(gm_patch, (0, 0, cw, ch))
        gm_patch_roi.upload(padded_view)

    # ── 3. Transfer crop to GpuMat via pinned (page-locked) memory ────────────
    # Pinned memory allows the GPU DMA engine to transfer directly without an
    # intermediate OS copy, giving higher H2D bandwidth than pageable RAM.
    # pinned_crop_buf is pre-allocated at max image size to avoid per-frame alloc.
    # np.frombuffer creates a zero-copy view into the pinned buffer — no copy.
    # cp.asnumpy(..., out=pinned_view) writes directly into the pinned buffer.
    # gm_patch is pre-allocated; we use a ROI view to avoid a new GpuMat alloc.
    #
    # SKIPPED when the pad branch (step 2) already handled the upload — in that
    # case gm_patch_roi is already set and pointing into the pre-allocated GpuMat.
    else:
        crop_bytes  = patch_gpu.nbytes
        pinned_view = np.frombuffer(pinned_crop_buf, dtype=np.uint8,
                                    count=crop_bytes).reshape(patch_gpu.shape)
        cp.asnumpy(patch_gpu, out=pinned_view)
        ch, cw = patch_gpu.shape[0], patch_gpu.shape[1]
        cw = min(cw, img_w)   # safety clamp — ROI must not exceed backing GpuMat
        ch = min(ch, img_h)
        gm_patch_roi = cv2.cuda_GpuMat(gm_patch, (0, 0, cw, ch))
        gm_patch_roi.upload(pinned_view)

    # ── 4. Resize crop to canvas size on GPU ──────────────────────────────────
    # canvas_w/h were computed above alongside crop_w/h. The resize ratio is
    # canvas/crop = canvas_h / (zoom*out_h*scale*crop_factor_h) on both axes
    # equally — guaranteed by the derived crop_w = crop_h * canvas_w/canvas_h.
    # INTER_AREA is the correct downsampling filter (avoids aliasing).
    # gm_small is pre-allocated at max_canvas; we write into a ROI sized to
    # the actual canvas this frame.
    gm_small_roi = cv2.cuda_GpuMat(gm_small, (0, 0, canvas_w, canvas_h))
    cv2.cuda.resize(gm_patch_roi, (canvas_w, canvas_h),
                    dst=gm_small_roi,
                    interpolation=cv2.INTER_AREA,
                    stream=stream)

    # Prepare the output array view backed by the pinned output buffer.
    # np.frombuffer is zero-copy — out_arr points directly into pinned RAM.
    out_arr = np.frombuffer(out_pinned_buf, dtype=np.uint8,
                            count=out_w * out_h * 3).reshape(out_h, out_w, 3)

    # Fast path: no warp needed, download the resized frame directly.
    if not (needs_warp or needs_tilt):
        gm_small_roi.download(dst=out_arr, stream=stream)   # async D2H
        return out_arr

    # ── 5. Canvas is the warp surface ─────────────────────────────────────────
    # The inflated canvas (step 4) already contains real image data in the
    # regions that rotation/tilt will pull into the output corners.
    # No copyMakeBorder padding needed — that was the old approach before crop
    # inflation was implemented, and it caused black corners.
    gm_padded  = gm_small_roi
    pd_w, pd_h = gm_padded.size()   # GpuMat.size() returns (width, height)
    centre     = (pd_w / 2.0, pd_h / 2.0)

    # ── 6. Warp on GPU ────────────────────────────────────────────────────────
    if needs_tilt:
        # Physically correct camera perspective: H = K · Ry(bank) · Rx(tilt) · Rz(roll) · K⁻¹
        #
        # Coordinate system (camera frame, right-handed):
        #   z : optical axis, positive pointing down toward image
        #   x : rightward on screen
        #   y : upward on screen (opposite to OpenCV image y — handled by negating tilt)
        #
        # f = (canvas_h/2) / tan(FOV/2): virtual focal length.
        # At tilt = CAMERA_FOV/2 the horizon reaches the frame edge.
        f   = (pd_h / 2.0) / np.tan(np.radians(CAMERA_FOV / 2.0))
        cx0 = pd_w / 2.0
        cy0 = pd_h / 2.0

        K    = np.array([[f,   0,  cx0],
                         [0,   f,  cy0],
                         [0,   0,  1.0]], dtype=np.float64)
        Kinv = np.array([[1/f, 0,  -cx0/f],
                         [0,   1/f,-cy0/f],
                         [0,   0,  1.0  ]], dtype=np.float64)

        # Rx(tilt): negated because image y increases downward, our y increases upward.
        # Positive tilt → camera pitches back → upper screen edge compresses.
        tr = np.radians(-tilt_deg)
        Rx = np.array([[1,  0,           0          ],
                       [0,  np.cos(tr), -np.sin(tr) ],
                       [0,  np.sin(tr),  np.cos(tr) ]], dtype=np.float64)

        # Ry(bank): positive bank → camera banks left → right side compresses.
        br = np.radians(bank_angle_deg)
        Ry = np.array([[ np.cos(br), 0, np.sin(br)],
                       [ 0,          1, 0          ],
                       [-np.sin(br), 0, np.cos(br)]], dtype=np.float64)

        # Rz(roll): embed getRotationMatrix2D as pure 3×3 rotation (strip translation).
        # Applied first so tilt/bank act in the rolled camera frame.
        Mz       = cv2.getRotationMatrix2D((cx0, cy0), angle_deg, 1.0)
        Rz       = np.eye(3, dtype=np.float64)
        Rz[:2,:] = Mz
        Rz[0, 2] = 0.0
        Rz[1, 2] = 0.0

        H = K @ Ry @ Rx @ Rz @ Kinv
    else:
        # Pure roll — embed as a 3×3 homography so we use one code path.
        M    = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
        H    = np.eye(3, dtype=np.float64)
        H[:2, :] = M

    # Build inverse warp map on GPU via CuPy and apply with cv2.cuda.remap.
    # H_inv maps each output pixel → source pixel (perspective divide).
    # cp_map_x/y_full are zero-copy CuPy views into gm_map_x/y VRAM
    # (via UnownedMemory). The matmul writes directly into GpuMat VRAM.
    H_inv    = np.linalg.inv(H)                          # 3×3 CPU, negligible
    H_inv_cp = cp.asarray(H_inv, dtype=cp.float32)       # 9 floats H2D, trivial
    N        = pd_h * pd_w
    # Fill coordinate grid in-place using pre-allocated slabs and broadcasting.
    xv = cp_coords[0, :N].reshape(pd_h, pd_w)           # view, no allocation
    yv = cp_coords[1, :N].reshape(pd_h, pd_w)           # view, no allocation
    xv[:] = cp_xs_full[:pd_w]                            # broadcast 0..pd_w-1
    yv[:] = cp_ys_full[:pd_h, None]                      # broadcast 0..pd_h-1
    cp_coords[2, :N] = 1.0
    cp.matmul(H_inv_cp, cp_coords[:, :N], out=cp_mapped[:, :N])
    w = cp_mapped[2, :N]
    # Write src coords directly into GpuMat VRAM via the CuPy zero-copy view.
    # Note: cp_map_x/y_full may have padded row stride (step/4 columns); we
    # write only the first pd_w columns of each row using a strided slice.
    cp_map_x_full[:pd_h, :pd_w] = (cp_mapped[0, :N] / w).reshape(pd_h, pd_w)
    cp_map_y_full[:pd_h, :pd_w] = (cp_mapped[1, :N] / w).reshape(pd_h, pd_w)

    gm_map_x_roi = cv2.cuda_GpuMat(gm_map_x, (0, 0, pd_w, pd_h))
    gm_map_y_roi = cv2.cuda_GpuMat(gm_map_y, (0, 0, pd_w, pd_h))
    gm_warp_roi  = cv2.cuda_GpuMat(gm_warp_out, (0, 0, pd_w, pd_h))
    cv2.cuda.remap(gm_padded, gm_map_x_roi, gm_map_y_roi,
                   interpolation=cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT,
                   dst=gm_warp_roi,
                   stream=stream)
    gm_warped = gm_warp_roi

    # ── 7. Centre-crop to output size — zero-copy ROI ─────────────────────────
    # The warped canvas is larger than the output. Extract the centre region.
    # GpuMat ROI is a view — no copy, no allocation.
    x0f    = pd_w // 2 - out_w // 2
    y0f    = pd_h // 2 - out_h // 2
    gm_out = cv2.cuda_GpuMat(gm_warped, (x0f, y0f, out_w, out_h))

    # ── 8. Async D2H download into pinned output buffer ───────────────────────
    # dst=out_arr writes directly into the page-locked buffer.
    # This call returns immediately — the transfer runs in the background on
    # `stream`. The caller (pipe_write thread) calls stream.waitForCompletion()
    # before reading out_arr to ensure the transfer is complete.
    gm_out.download(dst=out_arr, stream=stream)
    return out_arr


def _generate_keyframes():
    """
    Copy the default keyframes.py template to:
      - ./keyframes.py        (current directory)
      - ~/.config/eumovie/keyframes.py
    Does not overwrite existing files.
    """
    default = os.path.join(os.path.dirname(__file__), "keyframes_default.py")
    targets = [
        os.path.join(os.getcwd(), "keyframes.py"),
        os.path.join(os.path.expanduser("~"), ".config", "eumovie", "keyframes.py"),
    ]
    for target in targets:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target):
            print(f"  Skipped (already exists): {target}")
        else:
            shutil.copy(default, target)
            print(f"  Created: {target}")
    print("\nEdit keyframes.py to add camera paths for your tiles.")
    print("See the comments at the top of the file for full instructions.")


def main():
    args = parse_arguments()

    if args.generate_keyframes:
        _generate_keyframes()
        sys.exit(0)

    if not os.path.isfile(args.input):
        print(f"Error: input file '{args.input}' not found.")
        sys.exit(1)

    basename = os.path.basename(args.input)
    tile_id  = os.path.splitext(basename)[0]
    suffix   = f"_fulldome{args.fulldome}" if args.fulldome else ""
    ext      = ".mov" if args.prores else ".mp4"
    output   = os.path.join(os.path.dirname(args.input), tile_id + suffix + ext)

    # ── Initialise GPU ─────────────────────────────────────────────────────────
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("No CUDA devices found.")
        dev = cp.cuda.Device(0)
        dev.use()
        mem_total = dev.mem_info[1] / 1024**3
        print(f"Using GPU 0  ({mem_total:.1f} GB VRAM)")
    except Exception as e:
        print(f"CUDA error: {e}")
        sys.exit(1)

    has_cv2_cuda = hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0
    if not has_cv2_cuda:
        print("ERROR: cv2.cuda not available. This script requires OpenCV built with CUDA support.")
        sys.exit(1)
    print("cv2.cuda available — full GPU pipeline active.")

    # ── Load and normalise image ───────────────────────────────────────────────
    print(f"Loading {args.input} ...")
    image_np = tifffile.imread(args.input)

    if image_np.dtype == np.uint16:
        # Stretch uint16 to uint8 using percentile clipping to avoid
        # washed-out images from hot pixels or extreme values.
        print("Converting uint16 -> uint8 ...")
        lo = np.percentile(image_np, 0.1)
        hi = np.percentile(image_np, 99.9)
        image_np = np.clip(image_np.astype(np.float32), lo, hi)
        image_np = ((image_np - lo) / (hi - lo) * 255).astype(np.uint8)
    elif image_np.dtype != np.uint8:
        print(f"Error: unsupported dtype {image_np.dtype}.")
        sys.exit(1)

    img_h, img_w = image_np.shape[:2]

    # zoom_native: the zoom value at which one source pixel maps to one output pixel.
    #
    # Derivation: scale = min(img_w/out_w, img_h/out_h) converts output pixels to
    # source pixels (the limiting/shorter axis). At zoom=zoom_native the crop on the
    # short axis is: zoom_native * out_short * scale source pixels. After resizing
    # that crop back to out_short output pixels, each output pixel covers
    # (zoom_native * scale) source pixels. For 1:1 mapping we need zoom_native * scale = 1,
    # so zoom_native = 1/scale = max(out_w/img_w, out_h/img_h).
    #
    # For square outputs (out_w==out_h, img_w==img_h): max(N/M, N/M) = N/M, same as before.
    # For 1920x1080 on 19200x19200: max(1920/19200, 1080/19200) = 0.1 (was 0.05625 = wrong).
    # The old formula min(out_short)/min(img_short) gave a value 1.78x too small for 16:9,
    # causing zoom_native keyframes to appear 1.78x too zoomed in on rectangular outputs.
    zoom_native = max(args.width / img_w, args.height / img_h)

    get_config = _load_keyframes()
    config    = get_config(tile_id, zoom_native)

    # Account for y-flip in TIFF image coordinates with respect to FITS.
    # Keyframes are in FITS coordinates
    KEYFRAMES = [
        (t / args.speed, cx, img_h - cy, zoom, angle, tilt, bank)
        for (t, cx, cy, zoom, angle, tilt, bank) in config['keyframes']
    ]


    vram_needed = image_np.nbytes / 1024**3
    print(f"Image size: {img_w} x {img_h}  ({vram_needed:.1f} GB VRAM)")
    print(f"Image centre: {img_w // 2}, {img_h // 2}")
    print(f"First keyframe cx,cy: {KEYFRAMES[0][1]}, {KEYFRAMES[0][2]}")

    if vram_needed > mem_total * 0.8:
        print(f"Warning: image needs {vram_needed:.1f} GB, GPU has {mem_total:.1f} GB.")

    # Upload the full image to GPU VRAM once. It stays there for the entire render.
    # All per-frame crops are zero-copy CuPy slices of this array.
    print("Uploading image to GPU ...")
    gpu_image = cp.asarray(image_np)
    del image_np   # free CPU RAM immediately — no longer needed
    print(f"GPU image ready. VRAM used: {gpu_image.nbytes / 1024**3:.2f} GB")

    # GpuMat of the full source image for cv2.cuda.remap in fulldome mode.
    # Only allocated and uploaded in fulldome mode to avoid wasting ~1 GB VRAM
    # in flat mode where it is never needed.
    if args.fulldome:
        gm_full_image = cv2.cuda_GpuMat(img_h, img_w, cv2.CV_8UC3)
        img_np_view   = cp.asnumpy(gpu_image)   # D2H once for the upload
        gm_full_image.upload(img_np_view)
        del img_np_view
        print(f"Fulldome GpuMat uploaded ({img_h*img_w*3/1024**2:.0f} MB)")
    else:
        gm_full_image = None

    # ── Pre-allocate pinned (page-locked) CPU buffers ─────────────────────────
    # Pinned memory is not swappable by the OS, allowing the GPU's DMA engine to
    # transfer directly without kernel involvement — significantly faster H2D/D2H.
    #
    # pinned_crop_buf: holds the full-res crop before H2D upload to gm_patch.
    #   Sized to the full image (worst case: zoom=1.0, crop = entire image).
    #   crop_w/h are clamped to image bounds so this is always sufficient.
    #
    # pinned_out_bufs[2]: two output frame buffers for double-buffering.
    #   While the GPU downloads frame N into slot[N%2], the CPU pipe_write thread
    #   writes frame N-1 from slot[(N-1)%2] to ffmpeg — fully overlapped.
    #   Each buffer holds one output frame (~12MB at 2000x2000, ~6MB at 1920x1080).
    frame_bytes     = args.width * args.height * 3
    pinned_out_bufs = [cp.cuda.alloc_pinned_memory(frame_bytes) for _ in range(2)]

    # ── Helper: wrap a GpuMat's device memory as a CuPy array (zero-copy) ────
    def gpumat_to_cupy(gm, shape):
        mem    = cp.cuda.UnownedMemory(gm.cudaPtr(), gm.step * shape[0], gm)
        memptr = cp.cuda.MemoryPointer(mem, 0)
        return cp.ndarray((shape[0], gm.step // 4), dtype=cp.float32, memptr=memptr)

    if args.fulldome:
        # ── Fulldome allocations ───────────────────────────────────────────────
        # In fulldome mode the remap goes directly from gm_full_image to the
        # output — no crop/resize/canvas pipeline needed. Allocate only the
        # map GpuMats (out_size × out_size) and the remap output GpuMat.
        # This saves ~2-3 GB VRAM compared to allocating flat-mode buffers.
        out_size    = args.width
        gm_patch    = None   # unused in fulldome
        gm_small    = None   # unused in fulldome
        max_canvas  = out_size
        pinned_crop_buf = None   # unused in fulldome
        gm_warp_out = cv2.cuda_GpuMat(out_size, out_size, cv2.CV_8UC3)
        gm_map_x    = cv2.cuda_GpuMat(out_size, out_size, cv2.CV_32FC1)
        gm_map_y    = cv2.cuda_GpuMat(out_size, out_size, cv2.CV_32FC1)
        cp_map_x_full = gpumat_to_cupy(gm_map_x, (out_size, out_size))
        cp_map_y_full = gpumat_to_cupy(gm_map_y, (out_size, out_size))
        # Pre-build pixel offset grid (constant for all frames).
        # cp_px/cp_py are (out_size*out_size,) flat float32 arrays holding
        # x/y offsets from the dome centre. Built once, reused every frame.
        N_max    = out_size * out_size
        R        = out_size // 2
        xs       = cp.arange(out_size, dtype=cp.float32) - R
        ys       = cp.arange(out_size, dtype=cp.float32) - R
        gx, gy   = cp.meshgrid(xs, ys)
        cp_px    = gx.ravel().copy()
        cp_py    = (-gy).ravel().copy()   # negate: image row increases downward
        del xs, ys, gx, gy
        # Flat mode work arrays set to None (unused in fulldome)
        cp_coords  = None
        cp_mapped  = None
        cp_xs_full = None
        cp_ys_full = None
        map_mb     = out_size * out_size * 4 / 1024**2
        print(f"Pre-allocating fulldome GpuMats: "
              f"warp+maps={out_size}x{out_size} ({out_size*out_size*3/1024**2:.0f} MB + "
              f"{map_mb:.0f} MB x2), px/py grids ({N_max*4/1024**2:.0f} MB each)")
    else:
        # ── Flat mode allocations ─────────────────────────────────────────────
        max_crop_bytes  = img_h * img_w * 3
        pinned_crop_buf = cp.cuda.alloc_pinned_memory(max_crop_bytes)
        out_diag   = np.sqrt(args.width**2 + args.height**2)
        max_canvas = int(np.ceil(out_diag / (1.0 - np.sin(np.radians(CAMERA_FOV / 2.0)))))
        gm_patch    = cv2.cuda_GpuMat(img_h, img_w, cv2.CV_8UC3)
        gm_small    = cv2.cuda_GpuMat(max_canvas, max_canvas, cv2.CV_8UC3)
        gm_warp_out = cv2.cuda_GpuMat(max_canvas, max_canvas, cv2.CV_8UC3)
        gm_map_x    = cv2.cuda_GpuMat(max_canvas, max_canvas, cv2.CV_32FC1)
        gm_map_y    = cv2.cuda_GpuMat(max_canvas, max_canvas, cv2.CV_32FC1)
        cp_map_x_full = gpumat_to_cupy(gm_map_x, (max_canvas, max_canvas))
        cp_map_y_full = gpumat_to_cupy(gm_map_y, (max_canvas, max_canvas))
        N_max      = max_canvas * max_canvas
        cp_coords  = cp.empty((3, N_max), dtype=cp.float32)
        cp_mapped  = cp.empty((3, N_max), dtype=cp.float32)
        cp_xs_full = cp.arange(max_canvas, dtype=cp.float32)
        cp_ys_full = cp.arange(max_canvas, dtype=cp.float32)
        map_mb     = max_canvas * max_canvas * 4 / 1024**2
        print(f"Pre-allocating GpuMats: patch={img_w}x{img_h} ({img_h*img_w*3/1024**2:.0f} MB), "
              f"canvas+warp={max_canvas}x{max_canvas} ({max_canvas*max_canvas*3/1024**2:.0f} MB each), "
              f"maps x2 ({map_mb:.0f} MB each)")

    # Flush CuPy's memory pool before the render loop. CuPy caches freed blocks
    # for reuse; flushing ensures a clean slate after startup allocations.
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    # Two CUDA streams — one per double-buffer slot.
    # Each stream tracks the async D2H download for its slot independently,
    # so the pipe_write thread can wait on the correct stream.
    streams = [cv2.cuda_Stream(), cv2.cuda_Stream()]

    # ── Pre-calculate all frame parameters ────────────────────────────────────
    # Instead of calling each interpolator once per frame (6 calls × N frames
    # = O(N) Python overhead), evaluate all interpolators on the full time array
    # in one vectorized C-level call each. For 6480 frames this replaces ~39,000
    # Python function calls with 6 numpy operations.
    cs_cx, cs_cy, cs_zoom, cs_angle, cs_tilt, cs_bank = build_interpolators(KEYFRAMES)
    print(f"Interpolation tension: {TENSION}  (0.0=PCHIP, 1.0=CubicSpline)")
    total_duration = KEYFRAMES[-1][0]
    total_frames   = int(total_duration * args.fps)
    print(f"Rendering {total_frames} frames at {args.fps} fps ({total_duration:.1f}s)")

    times      = np.linspace(0, total_duration, total_frames, endpoint=False)
    all_cx     = cs_cx(times)
    all_cy     = cs_cy(times)
    all_zoom   = np.clip(cs_zoom(times) / args.zoom, 1e-4, 1.0)
    # zoom_native > 1.0 is only possible when output resolution exceeds source image
    # resolution, in which case zoom_native keyframes are silently capped at 1.0.
    # Dividing by args.zoom applies the modifier: >1 zooms in, <1 zooms out.
    all_angle  = cs_angle(times)
    all_tilt   = cs_tilt(times)
    all_bank   = cs_bank(times)

    # ── FFmpeg subprocess ──────────────────────────────────────────────────────
    # Encoder selection:
    #   --prores         : Apple ProRes 4444 (.mov) via prores_ks — any resolution
    #   fulldome, no prores: hevc_nvenc (GPU H.265) if available, else libx264
    #   flat, no prores  : h264_nvenc (GPU H.264)
    #     -bf 0 / -profile main: NLE (e.g. OpenShot) compatibility
    #     -movflags +faststart: moov atom at file start for web streaming
    # Common raw video input arguments (shared by all encoder paths)
    _raw_in = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{args.width}x{args.height}",
        "-pix_fmt", "rgb24",
        "-r", str(args.fps),
        "-i", "pipe:0",
    ]

    if args.prores:
        # Apple ProRes 4444 — lossless-quality master, works for flat and fulldome.
        # 10-bit 4:4:4, no alpha (rgb24 source has no alpha channel).
        try:
            test = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, timeout=10)
            if b"prores_ks" not in test.stdout:
                print("ERROR: prores_ks not found in ffmpeg. Install ffmpeg with ProRes support.")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR checking ffmpeg encoders: {e}")
            sys.exit(1)
        label = f"{'Fulldome' if args.fulldome else 'Flat'} encoder: Apple ProRes 4444 (.mov)"
        print(label)
        ffmpeg_cmd = _raw_in + [
            "-c:v", "prores_ks",
            "-profile:v", "4444",
            "-pix_fmt", "yuv444p10le",
            "-vendor", "apl0",
            "-bits_per_mb", str(args.bits_per_mb),
            output
        ]
    elif args.fulldome:
        # Fulldome without ProRes: prefer hevc_nvenc (GPU H.265), fall back to libx264.
        try:
            test = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, timeout=10)
            nvenc_ok = b"hevc_nvenc" in test.stdout
        except Exception:
            nvenc_ok = False
        if nvenc_ok:
            print("Fulldome encoder: hevc_nvenc (GPU H.265)")
            ffmpeg_cmd = _raw_in + [
                "-c:v", "hevc_nvenc",
                "-preset", "p7", "-tune", "hq",
                "-rc", "vbr", "-cq", str(args.cq),
                "-b:v", "0", "-maxrate", "120M", "-bufsize", "240M",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-tag:v", "hvc1",   # QuickTime/Apple player compatibility
                output
            ]
        else:
            print("Fulldome encoder: libx264 (hevc_nvenc unavailable)")
            ffmpeg_cmd = _raw_in + [
                "-c:v", "libx264",
                "-crf", str(args.cq), "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output
            ]
    else:
        # Flat mode without ProRes: h264_nvenc (GPU H.264).
        # -bf 0 / -profile main: NLE compatibility (e.g. OpenShot).
        print("Flat encoder: h264_nvenc (GPU H.264)")
        ffmpeg_cmd = _raw_in + [
            "-c:v", "h264_nvenc",
            "-preset", "p7", "-tune", "hq",
            "-rc", "vbr", "-cq", str(args.cq),
            "-b:v", "0", "-maxrate", "80M", "-bufsize", "160M",
            "-bf", "0", "-profile:v", "main",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output
        ]

    if args.fulldome:
        print(f"Fulldome mode: {args.fulldome.upper()} ({args.width}x{args.height}), "
              f"azimuthal equidistant fisheye")
    print(f"Encoding to {output} ...", flush=True)
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # ── Double-buffered render loop ────────────────────────────────────────────
    # Overlaps GPU rendering with ffmpeg pipe writes using two buffer slots.
    #
    # The key insight: gm_out.download(dst=out_arr, stream=stream) is ASYNC —
    # it returns immediately while the GPU DMA transfer runs in the background.
    # This lets the main thread immediately start the next frame's GPU work.
    # Meanwhile a worker thread waits for the previous frame's transfer to
    # complete (stream.waitForCompletion()) then writes it to the ffmpeg pipe.
    #
    # Timeline (S=stream, B=buffer slot):
    #   frame 0: GPU→S0→B0 (async)
    #   frame 1: GPU→S1→B1 (async) | thread: S0.wait, write B0→pipe
    #   frame 2: GPU→S0→B0 (async) | thread: S1.wait, write B1→pipe
    #   frame 3: GPU→S1→B1 (async) | thread: S0.wait, write B0→pipe
    #   ...
    #
    # Before reusing slot N, we join() the thread that was writing it to ensure
    # the buffer is safe to overwrite. The join() only blocks when the pipe write
    # is slower than the GPU render — in practice it's nearly always free.

    write_thread = None
    write_error  = [None]  # single list reused across frames to avoid allocation

    def pipe_write(stream, frame_arr, err):
        """
        Worker thread: wait for GPU D2H download to complete, then write to pipe.
        Runs concurrently with the main thread's next-frame GPU work.
        memoryview(frame_arr) avoids a .tobytes() copy — zero-copy pipe write.
        """
        try:
            stream.waitForCompletion()
            proc.stdin.write(memoryview(frame_arr))
        except Exception as e:
            err[0] = e

    try:
        for frame_idx in range(total_frames):
            slot = frame_idx % 2

            # Ensure the previous write to this slot's buffer is complete before
            # we overwrite it with the current frame. join() is usually instant
            # since the pipe write finishes well before the next GPU frame.
            if write_thread is not None and frame_idx >= 2:
                write_thread.join()
                if write_error[0]:
                    raise write_error[0]
                write_thread = None

            # Render frame into slot's pinned output buffer (async D2H at end).
            if args.fulldome:
                # ── Fulldome branch ───────────────────────────────────────
                # Build the fisheye remap map on GPU, then remap directly
                # from the full-res GPU image into the output buffer.
                # No crop/resize step needed: the map addresses the full
                # gpu_image directly, so we remap from the source image.
                out_size = args.width   # square
                build_fulldome_map(
                    out_size,
                    float(all_zoom[frame_idx]),
                    float(all_cx[frame_idx]),
                    float(all_cy[frame_idx]),
                    float(all_angle[frame_idx]),
                    float(all_tilt[frame_idx]),
                    float(all_bank[frame_idx]),
                    img_w, img_h,
                    min(img_w / out_size, img_h / out_size),
                    cp_px, cp_py,
                    cp_map_x_full, cp_map_y_full)
                # Remap directly from full GPU image to output buffer.
                # gm_map_x/y_roi sized to out_size x out_size.
                gm_map_x_roi = cv2.cuda_GpuMat(gm_map_x, (0, 0, out_size, out_size))
                gm_map_y_roi = cv2.cuda_GpuMat(gm_map_y, (0, 0, out_size, out_size))
                # gm_fulldome_out: ROI of gm_warp_out sized to out_size x out_size.
                gm_fd_out = cv2.cuda_GpuMat(gm_warp_out, (0, 0, out_size, out_size))
                cv2.cuda.remap(gm_full_image, gm_map_x_roi, gm_map_y_roi,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               dst=gm_fd_out,
                               stream=streams[slot])
                out_arr = np.frombuffer(pinned_out_bufs[slot], dtype=np.uint8,
                                       count=out_size * out_size * 3).reshape(out_size, out_size, 3)
                gm_fd_out.download(dst=out_arr, stream=streams[slot])
            else:
                # ── Flat (standard) branch ────────────────────────────────
                render_frame_gpu(
                    gpu_image, pinned_crop_buf, pinned_out_bufs[slot],
                    gm_patch, gm_small, gm_warp_out, max_canvas,
                    gm_map_x, gm_map_y,
                    cp_map_x_full, cp_map_y_full,
                    cp_coords, cp_mapped, cp_xs_full, cp_ys_full,
                    float(all_cx[frame_idx]),
                    float(all_cy[frame_idx]),
                    float(all_zoom[frame_idx]),
                    float(all_angle[frame_idx]),
                    float(all_tilt[frame_idx]),
                    float(all_bank[frame_idx]),
                    args.width, args.height, img_h, img_w, streams[slot]
                )

            # Start writing the PREVIOUS slot to ffmpeg in a background thread.
            # This overlaps with the GPU work we just kicked off above.
            if frame_idx > 0:
                prev_slot      = 1 - slot
                write_error[0] = None
                prev_arr       = np.frombuffer(pinned_out_bufs[prev_slot], dtype=np.uint8,
                                               count=frame_bytes).reshape(args.height, args.width, 3)
                write_thread   = threading.Thread(
                    target=pipe_write,
                    args=(streams[prev_slot], prev_arr, write_error),
                    daemon=True
                )
                write_thread.start()

            if frame_idx % args.fps == 0:
                print(f"  t={times[frame_idx]:.1f}s / {total_duration:.1f}s  "
                      f"cx={all_cx[frame_idx]:.0f} cy={all_cy[frame_idx]:.0f} "
                      f"zoom={all_zoom[frame_idx]:.4f} angle={all_angle[frame_idx]:.1f}deg",
                      flush=True)

        # Flush: wait for the last write thread, then write the final frame.
        # The final frame was never written by a thread (there's no "next" frame
        # to overlap with), so we write it synchronously here.
        if write_thread is not None:
            write_thread.join()
            if write_error[0]:
                raise write_error[0]

        last_slot = (total_frames - 1) % 2
        streams[last_slot].waitForCompletion()
        last_arr = np.frombuffer(pinned_out_bufs[last_slot], dtype=np.uint8,
                                 count=frame_bytes).reshape(args.height, args.width, 3)
        proc.stdin.write(memoryview(last_arr))

    finally:
        # Close ffmpeg stdin to signal end of stream, then wait for it to finish.
        proc.stdin.close()
        proc.wait()

        # Explicit cleanup — Python's GC doesn't always promptly release CUDA objects.
        # Deleting in reverse allocation order and flushing CuPy's pools ensures
        # each subsequent render starts with a clean VRAM state. This is important
        # when rendering many tiles in sequence from a shell script.
        del gpu_image
        if gm_full_image is not None:
            del gm_full_image
        if gm_patch is not None:
            del gm_patch
        if gm_small is not None:
            del gm_small
        if pinned_crop_buf is not None:
            del pinned_crop_buf
        del gm_warp_out
        del cp_map_x_full
        del cp_map_y_full
        if cp_coords is not None:  del cp_coords
        if cp_mapped is not None:  del cp_mapped
        if cp_xs_full is not None: del cp_xs_full
        if cp_ys_full is not None: del cp_ys_full
        if args.fulldome:
            del cp_px
            del cp_py
        del gm_map_x
        del gm_map_y
        del pinned_out_bufs
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.cuda.Device(0).synchronize()   # wait for all pending CUDA ops before exit

    print(f"Done. Output: {output}")



if __name__ == "__main__":
    main()

