# eumovie

**GPU-accelerated movie renderer for Euclid MER TIFF images.**

`eumovie` creates cinematic zoom/pan/rotate movies from very large (up to 20000×20000 pixel)
astronomical TIFF images, with scripted camera paths driven by keyframes. It is designed
for imaging data from the [Euclid space telescope](https://www.esa.int/Science_Exploration/Space_Science/Euclid)
but works with any RGB TIFF file.

## Features

- **GPU-accelerated** rendering via CuPy and OpenCV CUDA — full pipeline on the GPU
- **Flat output** — standard rectilinear perspective at any resolution (up to 4K and beyond)
- **Fulldome output** — azimuthal equidistant fisheye projection for planetarium dome
  projection (4K and 8K)
- **Physical camera model** — roll, tilt (pitch), and bank with correct perspective foreshortening
- **ProRes 4444** output for professional editing and planetarium delivery
- **Smooth interpolation** — camera paths use PCHIP/CubicSpline blending for natural motion
- **Double-buffered pipeline** — GPU rendering overlapped with ffmpeg encoding

## Quick links

- [Installation](installation.md)
- [Quick start](quickstart.md)
- [Keyframes reference](keyframes.md)
- [All command-line options](options.md)
- [Fulldome mode](fulldome.md)

## Citation

If you use `eumovie` for a publication or public outreach, please acknowledge:

> Schirmer, M. (2026). *eumovie: GPU-accelerated movie renderer for Euclid MER images.*
> Max Planck Institute for Astronomy, Heidelberg.
> [https://github.com/schirmermischa/eumovie](https://github.com/schirmermischa/eumovie)

## License

MIT — see [LICENSE](https://github.com/schirmermischa/eumovie/blob/main/LICENSE).
