# Quick start

This guide assumes you have completed [installation](installation.md) and have
a colour TIFF image ready. The image can be uint8 or uint16 RGB; uint16 images
are automatically stretched to uint8 using 0.1–99.9 percentile clipping.

## Step 1 — Generate a keyframes template

```bash
eumovie --generate-keyframes
```

This creates `keyframes.py` in the current directory. Open it and add a camera
path for your tile. See the [Keyframes](keyframes.md) documentation for the
full format description and worked example.

## Step 2 — Render a test movie

Start with a short, low-resolution render to check your camera path:

```bash
eumovie --input MYTILE.tif --resolution 2k --fps 30
```

Output: `MYTILE.mp4` in the same directory as the input file.

The terminal will show progress once per second:

```
Using GPU 0  (11.6 GB VRAM)
cv2.cuda available — full GPU pipeline active.
Loading MYTILE.tif ...
Image size: 19200 x 19200  (1.0 GB VRAM)
...
Rendering 600 frames at 30 fps (20.0s)
Flat encoder: h264_nvenc (GPU H.264)
Encoding to MYTILE.mp4 ...
  t=0.0s / 20.0s  cx=12850 cy=2470 zoom=0.2133 angle=0.0deg
  t=1.0s / 20.0s  cx=12797 cy=2320 zoom=0.2090 angle=-0.8deg
  ...
Done. Output: MYTILE.mp4
```

## Step 3 — Render at full quality

Once the camera path looks right, render at your target resolution:

=== "4K flat"
    ```bash
    eumovie --input MYTILE.tif --resolution 4k --fps 60
    ```

=== "4K flat, ProRes master"
    ```bash
    eumovie --input MYTILE.tif --resolution 4k --fps 60 --prores
    ```

=== "Fulldome 4K"
    ```bash
    eumovie --input MYTILE.tif --fulldome 4k --fps 30
    ```

=== "Fulldome 8K, ProRes master"
    ```bash
    eumovie --input MYTILE.tif --fulldome 8k --fps 30 --prores
    ```

## Output filenames

Output files are placed in the same directory as the input TIFF:

| Mode | Filename |
|------|----------|
| Flat, H.264 | `MYTILE.mp4` |
| Flat, ProRes | `MYTILE.mov` |
| Fulldome 4K, H.265 | `MYTILE_fulldome4k.mp4` |
| Fulldome 8K, ProRes | `MYTILE_fulldome8k.mov` |

## Tips

- Start with `--fps 30` for testing — renders twice as fast as `--fps 60`
- Use `--cq 18` for quick drafts (smaller file, slightly lower quality)
- Use `--zoom 1.2` to zoom in 20% without editing `keyframes.py`
- Use `--speed 0.5` to slow the movie to half speed, `--speed 2.0` to double it
- For very large images (≥19200×19200), check available VRAM before rendering
  at 8K fulldome — see [Technical notes](technical.md)
- If your tile is not found, check that the tile filename (without extension)
  matches a key in your `keyframes.py`
