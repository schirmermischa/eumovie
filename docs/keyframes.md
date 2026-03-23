# Keyframes

Camera paths in `eumovie` are defined by a series of **keyframes** — snapshots
of the camera state at specific points in time. Between keyframes, all parameters
are smoothly interpolated using spline curves.

## Generating the template

```bash
eumovie --generate-keyframes
```

This creates `keyframes.py` in:

- The **current directory** `./keyframes.py` — for per-tile overrides
- `~/.config/eumovie/keyframes.py` — your personal library of all tiles

`eumovie` searches for `keyframes.py` in the current directory first, then
falls back to `~/.config/eumovie/keyframes.py`. This lets you keep a growing
library of all your tiles in one place, while still being able to drop a local
override next to a specific TIFF.

## Keyframe format

Each keyframe is a tuple:

```python
(t, cx, cy, zoom, angle, tilt, bank)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `t` | float | Time in seconds |
| `cx` | float | Horizontal centre of view (FITS pixels) |
| `cy` | float | Vertical centre of view (FITS pixels) |
| `zoom` | float | Fraction of image short-axis visible |
| `angle` | float | Roll — in-plane rotation (degrees) |
| `tilt` | float | Pitch — forward/backward lean (degrees) |
| `bank` | float | Bank — left/right lean (degrees) |

## Image coordinate system

`cx` and `cy` use **FITS pixel coordinates**:

- Origin `(0, 0)` is at the **lower-left** corner of the image
- `cx` increases toward the **right**
- `cy` increases **upward**

This matches the convention used by DS9, FITS headers, and most astronomy tools.
It is the **opposite** of TIFF/screen coordinates (origin at upper-left, y downward).
`eumovie` converts internally — you never need to worry about the flip.

!!! tip "Finding coordinates in DS9"
    Open your TIFF in DS9, set the coordinate readout to **Image**, and hover
    over the feature of interest. The displayed `x, y` values are your `cx, cy`.

## Camera angles

The camera is defined in a right-handed coordinate system:

- **z** — optical axis, pointing down toward the image
- **x** — rightward on screen
- **y** — upward on screen

### Roll (`angle`)

Rotation around the optical axis. Positive = counter-clockwise.
No perspective distortion — the image simply rotates in-plane.

### Tilt (`tilt`)

The camera pitches backward, as if you lean back in your seat.
Positive values cause the **upper screen edge to recede** (compress)
and the **lower edge to approach** (expand) — like leaning backward
over a landscape.

Zero tilt = camera points straight down, no perspective distortion.

### Bank (`bank`)

The camera banks sideways. Positive values cause the **right side
to recede** (compress) and the **left side to approach** (expand) —
like banking left in flight.

Zero bank = camera points straight down, no perspective distortion.

!!! note
    Tilt and bank are always applied relative to the current roll angle.
    If the camera is rolled 45°, tilt still means "pitch along the camera's
    own up-axis", not the screen's vertical.

The strength of the perspective effect is controlled by `CAMERA_FOV` in
`eumovie.py` (default 60°). At `tilt = CAMERA_FOV/2`, the horizon reaches
the frame edge.

## zoom and zoom_native

`zoom` controls how much of the image is visible:

- `zoom = 1.0` — entire image fits in the frame (fully zoomed out)
- `zoom = zoom_native` — one source pixel maps to one output pixel (native resolution)
- `zoom = 0.5` — half the image short-axis is visible

`zoom_native` is passed automatically to each tile's configuration function.
It equals `max(out_w / img_w, out_h / img_h)` and depends on the output
resolution you choose. Keyframe zoom values expressed as multiples of
`zoom_native` are therefore **resolution-independent** — the same keyframe
produces the same field of view at 1080p or 4K.

!!! tip "Quick zoom adjustment"
    Use `--zoom X` on the command line to scale all keyframe zoom values
    without editing `keyframes.py`. `--zoom 1.2` gives 20% more magnification;
    `--zoom 0.8` gives a wider field of view. Keyframe zoom values are divided
    by this factor.

!!! tip "Quick speed adjustment"
    Use `--speed X` to scale all keyframe timestamps without editing `keyframes.py`.
    `--speed 2.0` plays at twice the speed; `--speed 0.5` plays at half speed.
    Keyframe timestamps are divided by this factor.

## Adding a tile

Open `~/.config/eumovie/keyframes.py` and add a function following this pattern:

```python
def _make_MYTILE(zoom_native):
    # Named waypoints for readability (FITS pixel coordinates)
    X0, Y0 = 5000, 8000   # starting position
    X1, Y1 = 4500, 7200   # interesting galaxy cluster
    X2, Y2 = 3800, 6000   # end position

    return {
        'keyframes': [
            # t      cx    cy    zoom              angle  tilt  bank
            ( 0.0,  X0,  Y0,  1.0,                0,     0,    0),  # start zoomed out
            ( 8.0,  X1,  Y1,  zoom_native*0.8,  -20,     0,    0),  # zoom into cluster
            (12.0,  X1,  Y1,  zoom_native*0.6,  -45,     0,    0),  # hold
            (20.0,  X2,  Y2,  zoom_native*0.7,  -60,     5,    0),  # pan away with tilt
        ]
    }

TILE_CONFIGS['MYTILE'] = _make_MYTILE
```

The key `'MYTILE'` must match the filename of your TIFF without the extension.
So `MYTILE.tif` → key `'MYTILE'`.

## Interpolation and TENSION

Camera motion is interpolated using a blend of two spline methods, controlled
by the `TENSION` constant in `eumovie.py` (default 0.1):

- **PCHIP** (monotone, no overshoot) — always used for `cx` and `cy` position
- **CubicSpline** (smooth, slight overshoot) — blended in for `zoom`, `angle`, `tilt`, `bank`

`TENSION = 0.0` gives stiff, kink-free motion. Higher values add a sense of
inertia and weight to the camera's dynamic parameters.

## Helper: circular orbit

The `make_circle()` function generates a smooth circular camera orbit:

```python
from keyframes import make_circle

# Circle centred on (cx, cy) with radius 500 pixels,
# at zoom_native*0.7, starting at t=10s, lasting 8 seconds
pts = make_circle(cx=5000, cy=4000, radius=500,
                  zoom=zoom_native*0.7,
                  start_t=10.0, duration=8.0, n_points=12)
```

Append the returned list to your keyframes list.
