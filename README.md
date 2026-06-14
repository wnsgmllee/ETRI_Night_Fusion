# ETRI Night RGB-Thermal Image Fusion

This repository provides an inference pipeline for RGB-thermal image fusion under low-light conditions.  
Given a visible RGB image and its paired thermal image, `inference.py` generates a final fused RGB image.

The codebase is based on the original LEFuse repository:

- Reference: [cmhang/LEFuse](https://github.com/cmhang/LEFuse)

---

## 1. Clone This Repository

First, clone this repository:

```bash
git clone https://github.com/wnsgmllee/ETRI_Night_Fusion.git
cd ETRI_Night_Fusion
```

---

## 2. Environment Setup

This code does not require a strict Python or PyTorch version.  
You can use a recent Python version and a recent PyTorch version that matches your CUDA environment.

For example, you may create a conda environment as follows:

```bash
conda create -n etri_fusion python=3.10 -y
conda activate etri_fusion
```

Install PyTorch according to your own CUDA environment.  
For example, install a recent PyTorch version using either `conda` or `pip`.

After installing PyTorch, install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The provided `requirements.txt` contains the additional packages required by the inference code, such as NumPy, OpenCV, and scikit-image.

---

## 3. Configuration

All inference paths are configured in `options.py`.

There are two option classes:

```python
TrainOptions_E
TrainOptions_K
```

- `TrainOptions_E`: configuration for the ETRI dataset
- `TrainOptions_K`: configuration for the KIRO dataset

In each option class, set the following paths according to your own dataset location:

```python
--ckpt_path
--vi_path
--ir_path
--out_path
```

### Path Description

| Argument | Description |
|---|---|
| `ckpt_path` | Path to the pretrained LEFuse checkpoint, for example `L2024.pth` |
| `vi_path` | Absolute path to the visible RGB image folder |
| `ir_path` | Absolute path to the thermal image folder |
| `out_path` | Absolute path to the output folder where fused images will be saved |

For example, in `options.py`, modify:

```python
self.parser.add_argument('--vi_path', type=str, default="/absolute/path/to/rgb/images")
self.parser.add_argument('--ir_path', type=str, default="/absolute/path/to/thermal/images")
self.parser.add_argument('--out_path', type=str, default="/absolute/path/to/save/fused/results")
```

You should set `vi_path`, `ir_path`, and `out_path` to the absolute paths of your own data.

---

## 4. Input Folder Structure

The visible RGB image folder and thermal image folder must contain paired images.

For each visible RGB image in `vi_path`, the corresponding thermal image should exist in `ir_path` with the same base filename.

Example:

```text
vi_path/
├── 1.png
├── 2.png
├── 3.png

ir_path/
├── 1.png
├── 2.png
├── 3.png
```

The code matches images by filename base.  
For example, if the visible image is:

```text
1.png
```

then the corresponding thermal image should also have the same base name:

```text
1.png
```

The supported image extensions are:

```text
.png, .jpg, .jpeg, .bmp, .tif, .tiff
```

If a corresponding thermal image is not found, that image will be skipped and counted as a skipped image in the final report.

---

## 5. Inference

### Run inference on the ETRI dataset

```bash
python inference.py etri
```

This uses `TrainOptions_E` from `options.py`.

### Run inference on the KIRO dataset

```bash
python inference.py kiro
```

This uses `TrainOptions_K` from `options.py`.

---

## 6. Output

The fused images are saved to the `out_path` specified in `options.py`.

For each input image pair, the output fused image is saved as:

```text
{base_name}.png
```

Example:

```text
out_path/
├── 1.png
├── 2.png
├── 3.png
├── final_metrics.txt
```

In addition to fused images, the inference script also saves a text file:

```text
final_metrics.txt
```

This file contains the final quantitative summary of the inference results.

---

## 7. Metrics in `final_metrics.txt`

After inference, the script saves a summary of fusion quality metrics in `final_metrics.txt`.

The main metrics are:

```text
INFO_GAIN_EN: higher is better
VIS_STRUCTURE_SSIM: higher is better
THERMAL_TRANSFER_MI: higher is better
DARK_FLAT_NOISE: lower is better
```

### `INFO_GAIN_EN`

Entropy-based information gain between the original visible image and the fused image.  
A higher value indicates that the fused image contains more information.

### `VIS_STRUCTURE_SSIM`

Structural similarity between the original visible luminance channel and the fused luminance channel.  
A higher value indicates that the fused image better preserves visible structural information.

### `THERMAL_TRANSFER_MI`

Mutual information between the thermal image and the fused image.  
A higher value indicates that more thermal information is reflected in the fused result.

### `DARK_FLAT_NOISE`

Noise level measured in dark and flat image regions.  
A lower value indicates better suppression of low-light noise in smooth dark areas.

---

## 8. Inference Pipeline Summary

The final fusion process consists of the following steps.

---

### 8.1 Read RGB and Thermal Images

The visible RGB image and thermal image are loaded from `vi_path` and `ir_path`.

Both images are converted to tensors and normalized to the range `[0, 1]`.

---

### 8.2 Convert RGB Images to YCrCb

The visible RGB image is converted to YCrCb:

```text
RGB visible image → Y, Cr, Cb
```

The thermal image is also converted to YCrCb, and its Y channel is used for fusion:

```text
Thermal image → thermal Y
```

The visible Y channel contains luminance information, while Cr and Cb contain chrominance information.

---

### 8.3 Dark-Region-Aware Y-Channel Denoising

The visible Y channel is denoised before fusion.

This denoising step is specifically designed to suppress noise that appears in low-light regions.  
Instead of applying uniform denoising to the entire image, the code builds a dark-region mask so that stronger denoising is applied mainly to dark areas.

The dark mask is computed based on the visible Y intensity:

```text
darker pixels → stronger denoising
brighter pixels → weaker denoising
```

The denoising is performed using Non-Local Means filtering.

---

### 8.4 Edge-Aware Protection

To avoid over-smoothing important structures, the denoising mask is further controlled by an edge map.

The code computes an edge map using Sobel filtering.  
Strong edge regions are protected so that structural details and object boundaries are not excessively smoothed.

As a result:

```text
dark flat regions → denoised strongly
edge/detail regions → denoised weakly
```

This is important because nighttime RGB images often contain severe noise in dark smooth areas, while still requiring edge and texture preservation for visually meaningful fusion.

---

### 8.5 Detail Restoration

After denoising, a weak detail restoration step is applied.

The code adds back part of the difference between the original visible Y channel and the denoised Y channel, especially around edge regions.

This helps prevent the denoised visible image from becoming overly smooth.

---

### 8.6 Dark-Region Chroma Smoothing

The Cr and Cb channels from the visible RGB image are also processed.

In dark low-texture regions, chroma noise can produce unnatural color artifacts.  
To reduce this, the code smooths the chroma channels in dark regions while preserving chroma near edge regions.

This helps produce a cleaner final RGB fused image.

---

### 8.7 LEFuse Fusion

The processed visible Y channel and the thermal Y channel are passed to the pretrained LEFuse model:

```text
LEFuse(denoised visible Y, thermal Y) → fused Y
```

The output is a fused luminance channel that combines information from both the visible RGB image and the thermal image.

---

### 8.8 Robust Normalization

The fused Y channel is normalized using robust percentile-based normalization.

The selected normalization mode is:

```text
robust_0p5_99p5
```

This uses the 0.5% and 99.5% percentiles to reduce the effect of extreme outlier values.

---

### 8.9 RGB Restoration

Finally, the fused Y channel is combined with the processed visible chroma channels:

```text
fused Y + processed Cb + processed Cr → final RGB image
```

The final RGB image is clipped to the valid range and saved as a PNG image.

---

## 9. Key Design Motivation

The main motivation of this inference pipeline is to improve RGB-thermal fusion under low-light conditions.

Directly enhancing or fusing noisy low-light RGB images can amplify noise.  
Therefore, this code first suppresses low-light noise in the RGB luminance channel, especially in dark flat regions, before applying LEFuse.

The denoising step is intentionally designed to be:

```text
dark-region-aware
edge-aware
detail-preserving
chroma-noise-aware
```

This allows the final fused image to preserve useful visible structures while reducing noise artifacts commonly observed in nighttime RGB images.

---

## 10. Reference

This project is based on the LEFuse codebase:

```text
https://github.com/cmhang/LEFuse
```

If you use this repository, please also refer to the original LEFuse implementation.
