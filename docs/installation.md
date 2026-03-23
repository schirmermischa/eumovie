# Installation

## Prerequisites

`eumovie` requires several components that must be installed manually before
installing the package itself, because they depend on your specific CUDA version
and require custom builds.

### 1. NVIDIA GPU and CUDA toolkit

A CUDA-capable NVIDIA GPU is required. CUDA 12 or later is recommended.
Install the CUDA toolkit from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads).

Verify your installation:
```bash
nvidia-smi
nvcc --version
```

### 2. CuPy

CuPy provides GPU-accelerated array operations. Install the version matching
your CUDA toolkit:

```bash
pip install cupy-cuda12x   # for CUDA 12.x
pip install cupy-cuda13x   # for CUDA 13.x
```

See [docs.cupy.dev](https://docs.cupy.dev/en/stable/install.html) for other versions.

### 3. OpenCV with CUDA support

The standard `opencv-python` package from PyPI does **not** include CUDA support.
You must build OpenCV from source with `-DWITH_CUDA=ON`.

A minimal build sequence (adjust paths for your system):

```bash
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir build && cd build
cmake .. \
    -DWITH_CUDA=ON \
    -DCUDA_FAST_MATH=ON \
    -DBUILD_opencv_python3=ON \
    -DOPENCV_PYTHON3_INSTALL_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])") \
    -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
```

Verify CUDA support is available:
```bash
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
# Should print 1 or more
```

### 4. ffmpeg with NVENC support

Install ffmpeg from your system package manager. On Ubuntu:

```bash
sudo apt install ffmpeg
```

Verify hardware encoding is available:
```bash
ffmpeg -encoders 2>/dev/null | grep nvenc
# Should list h264_nvenc, hevc_nvenc
```

## Installing eumovie

Once the prerequisites are in place:

```bash
pip install eumovie
```

This installs `eumovie` as a command-line tool and pulls in the remaining
Python dependencies (`numpy`, `scipy`, `tifffile`) automatically.

## Setting up keyframes

On first use, generate the keyframes template:

```bash
eumovie --generate-keyframes
```

This creates `keyframes.py` in the current directory and in
`~/.config/eumovie/keyframes.py`. Edit either file to add camera paths
for your tiles. See the [Keyframes](keyframes.md) documentation for details.

## Verifying the installation

```bash
eumovie --help
```

Should print the full list of options without errors.
