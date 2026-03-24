# Technical notes

## GPU pipeline

`eumovie` offloads as much work as possible to the GPU. The CPU only handles
writing compressed frames to the ffmpeg pipe, which runs concurrently with GPU
rendering via double-buffering.

### Flat mode pipeline (per frame)

1. **Crop** — zero-copy CuPy slice of the full image in VRAM
2. **Pad** — if the crop extends outside image bounds, zero-fill in CPU pinned memory
3. **H2D transfer** — crop uploaded to `cv2.cuda_GpuMat` via pinned memory
4. **Resize** — `cv2.cuda.resize` with `INTER_AREA` downsampling to canvas size
5. **Build warp map** — inverse homography computed on GPU via CuPy `matmul`,
   written directly into pre-allocated `GpuMat` VRAM (zero-copy via `UnownedMemory`)
6. **Remap** — `cv2.cuda.remap` applies the warp with no hidden allocations
7. **Centre crop** — zero-copy `GpuMat` ROI to output size
8. **Async D2H** — asynchronous download into page-locked output buffer

### Fulldome mode pipeline (per frame)

1. **Build fisheye map** — single `CuPy ElementwiseKernel` computes all source
   coordinates in one GPU pass, writing directly into `GpuMat` VRAM
2. **Remap** — `cv2.cuda.remap` from the full-resolution source image
3. **Async D2H** — asynchronous download into page-locked output buffer

No crop or resize step — the fisheye map addresses the source image directly.

### Double-buffering

GPU rendering and ffmpeg pipe writes are fully overlapped using two alternating
output buffers:

```
frame 0: GPU→buffer0 (async)
frame 1: GPU→buffer1 (async) | CPU: write buffer0→pipe
frame 2: GPU→buffer0 (async) | CPU: write buffer1→pipe
...
```

The CPU pipe write runs in a background thread. In practice the GPU is the
bottleneck and the pipe write completes before the next frame is ready.

## VRAM requirements (minimum)

The VRAM usages listed here are the theoretical minimum. In practise,
they may be 2x, approaching 10-11 GB for a 19200×19200 source on our
test system with a NVIDIA RTX A2000 12GB card.


### Flat mode

| Component | Size (19200×19200 source) |
|-----------|--------------------------|
| Source image (CuPy) | 1.03 GB |
| `gm_patch` (GpuMat) | 1.03 GB |
| `gm_small` + `gm_warp_out` | ~180 MB each (2K output) |
| `gm_map_x` + `gm_map_y` | ~50 MB each (2K output) |
| CuPy work arrays | ~50 MB total |
| Pinned output buffers | ~12 MB |
| **Total (2K output)** | **~2.5 GB** |

Canvas size scales with output resolution and maximum rotation angle.

### Fulldome mode

| Component | 4K fulldome | 8K fulldome |
|-----------|-------------|-------------|
| Source image (CuPy) | 1.03 GB | 1.03 GB |
| `gm_full_image` (GpuMat) | 1.03 GB | 1.03 GB |
| `gm_warp_out` | 48 MB | 192 MB |
| `gm_map_x` + `gm_map_y` | 64 MB each | 256 MB each |
| `cp_px` + `cp_py` grids | 64 MB each | 256 MB each |
| Pinned output buffers | 48 MB | 192 MB |
| **Total** | **~2.4 GB** | **~3.5 GB** |

An 8K render of a 19200×19200 source uses approximately **5–6 GB VRAM**
in practice (including CUDA context overhead and CuPy memory pool).

## Known issues

### OpenCV CUDA warpPerspective VRAM leak

`cv2.cuda.warpPerspective` and `cv2.cuda.warpAffine` allocate an internal
scratch buffer on every call, despite `dst=` being pre-allocated. This causes
a steady VRAM leak of ~30 MB/call in production (when large co-resident
allocations are present). The isolated MWE does not reproduce the leak.

`eumovie` works around this by using `cv2.cuda.remap` with pre-allocated maps
instead of `warpPerspective`/`warpAffine`.

### Fulldome kernel compilation delay

The CuPy CUDA kernel for fulldome projection is compiled on first use (lazy
compilation). This causes a pause of a few seconds before the first frame.
Subsequent runs use the cached compiled kernel and start immediately.

## Performance tips

- **Use `--fps 30`** for draft renders — twice as fast as 60fps
- **Use `--cq 18`** for quick previews
- **Large images** — if VRAM is tight, close other GPU applications before rendering
- **8K fulldome** — for source images larger than 10000×10000, ensure at least
  8 GB VRAM free before starting
- **ProRes encoding** — CPU-bound; encoding speed (~11 fps at 4K/30fps) is
  independent of GPU load. The GPU pipeline runs faster than the CPU can encode,
  so the effective render speed is limited by the CPU encoder in ProRes mode.
