# Fulldome mode

`eumovie` can render content for digital planetarium systems using the standard
fulldome format.

## Projection

The output uses **azimuthal equidistant fisheye projection**, which is the
standard master format for digital planetarium systems:

- The frame is a **square image** with a circular content area inscribed within it.
  Corners outside the circle are black.
- The **centre of the circle** corresponds to the zenith (straight up into the dome).
- The **edge of the circle** corresponds to the horizon (90° from zenith).
- The mapping between radius and angle is **linear** — equal angular spacing per
  pixel from centre to edge.
- The full dome diameter covers a **180° field of view**.

This projection is supported natively by all major fulldome playback systems
including Uniview, Digistar, Sky-Skan, and Domeprojection, and is specified
in the IPS (International Planetarium Society) fulldome content standard.

## Resolutions

| Option | Frame size | Recommended fps | Approx. file size (ProRes, 1 min) |
|--------|-----------|-----------------|-----------------------------------|
| `--fulldome 4k` | 4096×4096 | 30 fps | ~7.5 GB |
| `--fulldome 8k` | 8192×8192 | 24–30 fps | ~30 GB |

## Output formats

### H.265 (default)

```bash
eumovie --input TILE.tif --fulldome 4k --fps 30
```

Encodes with `hevc_nvenc` (GPU H.265) if available, otherwise falls back to
`libx264`. Suitable for preview and smaller installations.

### Apple ProRes 4444 (recommended for delivery)

```bash
eumovie --input TILE.tif --fulldome 4k --fps 30 --prores
eumovie --input TILE.tif --fulldome 8k --fps 30 --prores --bits-per-mb 2000
```

- Container: `.mov`
- Codec: `prores_ks`, profile 4444
- Pixel format: `yuv444p10le` — 10-bit, 4:4:4 chroma, no alpha
- CPU-encoded — the GPU handles all rendering; only the encode step uses CPU

ProRes 4444 is the standard delivery format for high-end planetarium systems.
Most installations require it for final projection. Use `--bits-per-mb` to
control quality vs file size:

| `--bits-per-mb` | Quality | ~File size (4K, 1 min) |
|----------------|---------|------------------------|
| 500 | Proxy / preview | ~3.7 GB |
| 1000 | Broadcast (default) | ~7.5 GB |
| 2000 | High-end master | ~15 GB |

## Camera motion in fulldome

All keyframe parameters (`cx`, `cy`, `zoom`, `angle`, `tilt`, `bank`) work
identically in fulldome and flat mode. The camera coordinate definitions are
the same — see [Keyframes](keyframes.md) for details.

In fulldome mode, the camera always points toward the dome zenith. Tilt and
bank produce perspective foreshortening that, when projected on a dome, gives
the audience a sense of the camera leaning or banking over the landscape.

## Technical note for operators

The fulldome pipeline in `eumovie` bypasses the crop/resize stage used in flat
mode. The fisheye projection map is computed directly from the full-resolution
source image using a single GPU kernel (CuPy `ElementwiseKernel`), then applied
with `cv2.cuda.remap`. This avoids 2–3 GB of intermediate VRAM buffers that
would otherwise be needed for large source images.

The fisheye map pixel coordinates are computed using the standard pinhole
camera model with azimuthal equidistant projection:

```
r = sqrt(px² + py²) / R          (normalised radius, 0=zenith, 1=horizon)
theta = r * 90°                   (angle from zenith)
phi = atan2(py, px)               (azimuth)
direction = (sin θ cos φ, sin θ sin φ, cos θ)
```

Camera rotations (roll, tilt, bank) are applied to the direction vector before
projecting back onto the source image plane.
