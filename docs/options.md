# Command-line options

```
eumovie [options]
```

## General

| Option | Default | Description |
|--------|---------|-------------|
| `--input FILE` | *(required)* | Input TIFF file. uint8 or uint16 RGB. uint16 is automatically stretched to uint8 using 0.1–99.9 percentile clipping. |
| `--generate-keyframes` | — | Write a `keyframes.py` template to the current directory and to `~/.config/eumovie/`. Does not require `--input`. Skips existing files. |
| `--fps N` | `60` | Output frame rate. Use 30 for fulldome or ProRes; 60 for flat screen output. |
| `--cq N` | `15` | H.264/H.265 encoder quality. Lower = higher quality and larger file. 12 = very high, 15 = high, 18 = medium. Not used with `--prores`. |
| `--threads N` | half of CPU count | FFmpeg encoding threads (flat H.264 mode only). |

## Output resolution

`--resolution` and `--fulldome` are mutually exclusive.

| Option | Description |
|--------|-------------|
| `--resolution RES` | Flat output resolution. Named presets: `2k` (1920×1080), `4k` (3840×2160). Or specify as `WxH`, e.g. `2560x1440`. Default: `1920x1080`. |
| `--fulldome 4k` | Fulldome azimuthal equidistant fisheye at 4096×4096. |
| `--fulldome 8k` | Fulldome azimuthal equidistant fisheye at 8192×8192. |

## Output quality

These options apply to both flat and fulldome modes.

| Option | Default | Description |
|--------|---------|-------------|
| `--prores` | off | Encode as Apple ProRes 4444 (`.mov`). 10-bit 4:4:4 chroma, CPU-encoded. Recommended for planetarium delivery and professional NLE editing. |
| `--bits-per-mb N` | `1000` | ProRes quality in bits per macroblock. Only used with `--prores`. 500 = proxy, 1000 = broadcast, 2000 = high-end master. |

## Encoder selection

`eumovie` selects the ffmpeg encoder automatically:

| Mode | `--prores` | Encoder | Container |
|------|-----------|---------|-----------|
| Flat | no | `h264_nvenc` (GPU H.264) | `.mp4` |
| Flat | yes | `prores_ks` (CPU ProRes 4444) | `.mov` |
| Fulldome | no | `hevc_nvenc` (GPU H.265), fallback to `libx264` | `.mp4` |
| Fulldome | yes | `prores_ks` (CPU ProRes 4444) | `.mov` |

## Examples

```bash
# Default: 1920×1080 at 60fps, H.264
eumovie --input TILE.tif

# 4K flat at 60fps
eumovie --input TILE.tif --resolution 4k

# Custom resolution
eumovie --input TILE.tif --resolution 2560x1440

# 4K flat, ProRes master
eumovie --input TILE.tif --resolution 4k --prores

# Fulldome 4K at 30fps
eumovie --input TILE.tif --fulldome 4k --fps 30

# Fulldome 8K, ProRes delivery master, high quality
eumovie --input TILE.tif --fulldome 8k --fps 30 --prores --bits-per-mb 2000

# Quick draft (lower quality, faster)
eumovie --input TILE.tif --resolution 2k --fps 30 --cq 18
```

## Tunable constants

Two constants in `eumovie.py` can be adjusted to change the rendering behaviour.
Edit the installed file at `$(python3 -m pip show eumovie | grep Location)/eumovie/eumovie.py`.

### TENSION

Controls the "gravitational feel" of camera motion. Range 0.0–1.0.

- `0.0` — pure PCHIP: stiff, no overshoot, kink-free motion
- `0.1` — default: mostly PCHIP with slight inertial feel
- `0.5` — balanced: smooth inertial motion
- `1.0` — pure CubicSpline: maximum gravitational feel, zoom may briefly overshoot

Position (`cx`, `cy`) always uses PCHIP regardless of this setting.

### CAMERA_FOV

Virtual vertical field of view in degrees. Controls the strength of tilt and
bank perspective effects.

- `60°` — default: moderate cinematic feel. At `tilt = 30°` the horizon reaches the frame edge.
- `90°` — stronger wide-angle perspective

Higher FOV = stronger perspective distortion for the same tilt/bank angles.
