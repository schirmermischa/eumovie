# eumovie

GPU-accelerated movie renderer for Euclid MER TIFF images.

Renders cinematic zoom/pan/rotate movies from very large (up to 20000×20000 pixel)
astronomical TIFF images produced by the [Euclid space telescope](https://www.euclid-ec.org/).
Supports flat (rectilinear) and fulldome (azimuthal equidistant fisheye) output.

## Requirements

The following must be installed manually — they require a CUDA toolkit and/or
custom builds not available on PyPI:

- **NVIDIA GPU** with CUDA 12+
- **CuPy** — `pip install cupy-cuda12x` (match your CUDA version)
- **OpenCV with CUDA support** — must be built from source
- **ffmpeg** — system install with `h264_nvenc` / `hevc_nvenc` support

Standard dependencies (`numpy`, `scipy`, `tifffile`) are installed automatically by pip.

## Installation

```bash
pip install eumovie
```

## Usage

```bash
# Flat output, default 1920×1080
eumovie --input TILE.tif

# Flat output, 4K
eumovie --input TILE.tif --resolution 4k

# Custom resolution
eumovie --input TILE.tif --resolution 2560x1440

# ProRes 4444 master (flat)
eumovie --input TILE.tif --resolution 4k --prores

# Fulldome 4K (azimuthal equidistant fisheye)
eumovie --input TILE.tif --fulldome 4k --fps 30

# Fulldome 8K, ProRes master
eumovie --input TILE.tif --fulldome 8k --prores --fps 30
```

## Camera keyframes

Camera paths are defined per tile in a user-editable `keyframes.py` file.
A default file can be generated with

```
eumovie --generate-keyframes
```

which places two copies in

```
./keyframes.py
~/.config/eumovie/keyframes.py
```

The file in the current working directory takes precedence. Edit this file to add or modify camera paths for your tiles. Each tile is identified by its filename (without extension), e.g. `TILE102029855.tif` maps to the key `TILE102029855`.

See the comments at the top of `./keyframes.py` for the
keyframe format and camera coordinate system definition.

## Output formats

| Flag | Format | Encoder | Notes |
|------|--------|---------|-------|
| *(default)* | MP4 H.264 | h264_nvenc | GPU, NLE-compatible |
| `--fulldome 4k/8k` | MP4 H.265 | hevc_nvenc | GPU, fulldome standard |
| `--prores` | MOV ProRes 4444 | prores_ks | CPU, 10-bit 4:4:4, delivery master |

## License

MIT — see [LICENSE](LICENSE).
