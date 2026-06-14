import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from skimage.io import imsave
from skimage.metrics import structural_similarity as ssim

from options import TrainOptions_E
from img_utils import RGB2YCrCb, YCbCr2RGB, image_read_cv2
from net import LEFuse


# ==================================================
# Fixed final parameters
# ==================================================
# Selected trial:
# h_04_dt_0p35_mp_1p50_ep_2p00_dr_0p15_norm_robust_0p5_99p5_chroma_dark_smooth_cs_0p35_ck_05_cep_2p00

NLM_H = 4

DARK_THRESHOLD = 0.35
DARK_MASK_POWER = 1.5

EDGE_POWER = 2.0
DETAIL_RESTORE_STRENGTH = 0.15

NORMALIZE_MODE = "robust_0p5_99p5"

CHROMA_MODE = "dark_smooth"
CHROMA_STRENGTH = 0.35
CHROMA_BLUR_KERNEL = 5
CHROMA_EDGE_POWER = 2.0


# ==================================================
# Fixed NLM parameters
# ==================================================
NLM_TEMPLATE_WINDOW = 7
NLM_SEARCH_WINDOW = 21


# ==================================================
# Edge-aware options
# ==================================================
USE_IR_EDGE_FOR_MASK = False

EDGE_NORMALIZE_QUANTILE = 0.95
EDGE_BLUR_KERNEL = 5


# ==================================================
# Metric settings
# ==================================================
BASELINE_NORMALIZE_MODE = "robust_0p5_99p5"

EVAL_DARK_THRESHOLD = 0.35
EVAL_EDGE_THRESHOLD = 0.20
EVAL_HIGH_PASS_KERNEL = 5
MI_BINS = 64


# ==================================================
# Other settings
# ==================================================
VALID_EXTS = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']


def find_image(folder, base_name):
    for ext in VALID_EXTS:
        path = os.path.join(folder, base_name + ext)
        if os.path.exists(path):
            return path
    return None


def sync_cuda_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def safe_nanmean(values):
    vals = np.array(values, dtype=np.float64)
    if vals.size == 0:
        return None
    if np.all(np.isnan(vals)):
        return None
    return float(np.nanmean(vals))


def format_float(x):
    if x is None:
        return "None"
    try:
        if np.isnan(x):
            return "None"
    except TypeError:
        pass
    return f"{float(x):.6f}"


def safe_minmax_normalize(x, eps=1e-8):
    x_min = torch.min(x)
    x_max = torch.max(x)
    return (x - x_min) / (x_max - x_min + eps)


def robust_normalize(x, low_q=0.005, high_q=0.995, eps=1e-8):
    flat = x.flatten().float()

    lo = torch.quantile(flat, low_q)
    hi = torch.quantile(flat, high_q)

    denom = hi - lo
    if float(torch.abs(denom).detach().cpu()) < eps:
        return safe_minmax_normalize(x, eps=eps)

    x = (x - lo) / (denom + eps)
    return torch.clamp(x, 0.0, 1.0)


def normalize_fused_y(out, normalize_mode):
    if normalize_mode == "minmax":
        return safe_minmax_normalize(out)

    if normalize_mode == "robust_0p5_99p5":
        return robust_normalize(out, low_q=0.005, high_q=0.995)

    if normalize_mode == "robust_1_99":
        return robust_normalize(out, low_q=0.01, high_q=0.99)

    raise ValueError(f"Unknown normalize_mode: {normalize_mode}")


def box_filter_2d(x, kernel_size=5):
    if kernel_size is None or kernel_size <= 1:
        return x

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    pad = kernel_size // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    x_smooth = F.avg_pool2d(x_pad, kernel_size=kernel_size, stride=1, padding=0)

    return x_smooth


def sobel_edge_map(
    x,
    normalize_quantile=EDGE_NORMALIZE_QUANTILE,
    blur_kernel=EDGE_BLUR_KERNEL,
    eps=1e-8,
):
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected x shape [B, 1, H, W], but got {tuple(x.shape)}")

    device = x.device
    dtype = x.dtype

    sobel_x = torch.tensor(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)

    edge = torch.sqrt(gx * gx + gy * gy + eps)

    q = torch.quantile(edge.flatten().float(), normalize_quantile)
    if float(torch.abs(q).detach().cpu()) < eps:
        return torch.zeros_like(edge)

    edge = torch.clamp(edge / (q + eps), 0.0, 1.0)

    if blur_kernel is not None and blur_kernel > 1:
        if blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be odd.")

        pad = blur_kernel // 2
        edge = F.avg_pool2d(edge, kernel_size=blur_kernel, stride=1, padding=pad)
        edge = torch.clamp(edge, 0.0, 1.0)

    return edge


def build_dark_mask(
    vi_y,
    dark_threshold=DARK_THRESHOLD,
    mask_power=DARK_MASK_POWER,
):
    dark_mask = torch.clamp(
        (dark_threshold - vi_y) / (dark_threshold + 1e-8),
        min=0.0,
        max=1.0,
    )

    if mask_power != 1.0:
        dark_mask = dark_mask.pow(mask_power)

    return dark_mask


def dark_region_edge_aware_nlm_denoise_y(
    vi_y,
    ir_y=None,
    h=NLM_H,
    template_window_size=NLM_TEMPLATE_WINDOW,
    search_window_size=NLM_SEARCH_WINDOW,
    dark_threshold=DARK_THRESHOLD,
    mask_power=DARK_MASK_POWER,
    edge_power=EDGE_POWER,
    detail_restore_strength=DETAIL_RESTORE_STRENGTH,
    use_ir_edge_for_mask=USE_IR_EDGE_FOR_MASK,
):
    if vi_y.ndim != 4 or vi_y.shape[0] != 1 or vi_y.shape[1] != 1:
        raise ValueError(
            f"Expected vi_y shape [1, 1, H, W], but got {tuple(vi_y.shape)}"
        )

    device = vi_y.device
    dtype = vi_y.dtype

    # --------------------------------------------------
    # 1. Y tensor -> uint8 numpy grayscale
    # --------------------------------------------------
    y_raw = torch.clamp(vi_y.detach(), 0.0, 1.0)

    y_np = (
        y_raw.squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    y_uint8 = np.clip(y_np * 255.0, 0, 255).round().astype(np.uint8)

    # --------------------------------------------------
    # 2. NLM denoising
    # --------------------------------------------------
    y_denoised_uint8 = cv2.fastNlMeansDenoising(
        y_uint8,
        None,
        h=float(h),
        templateWindowSize=template_window_size,
        searchWindowSize=search_window_size,
    )

    y_denoised = (
        torch.from_numpy(y_denoised_uint8.astype(np.float32) / 255.0)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device, dtype=dtype)
    )

    # --------------------------------------------------
    # 3. Dark mask
    # --------------------------------------------------
    dark_mask = build_dark_mask(
        vi_y=vi_y,
        dark_threshold=dark_threshold,
        mask_power=mask_power,
    )

    # --------------------------------------------------
    # 4. Edge mask
    # --------------------------------------------------
    vi_edge = sobel_edge_map(vi_y)

    if use_ir_edge_for_mask and ir_y is not None:
        ir_edge = sobel_edge_map(ir_y)
        edge_mask = torch.maximum(vi_edge, ir_edge)
    else:
        edge_mask = vi_edge

    # --------------------------------------------------
    # 5. Edge-aware denoising mask
    # --------------------------------------------------
    edge_protection = (1.0 - edge_mask).pow(edge_power)
    edge_aware_mask = dark_mask * edge_protection
    edge_aware_mask = torch.clamp(edge_aware_mask, 0.0, 1.0)

    # --------------------------------------------------
    # 6. Blend original Y and denoised Y
    # --------------------------------------------------
    vi_y_processed = (
        (1.0 - edge_aware_mask) * vi_y
        + edge_aware_mask * y_denoised
    )

    # --------------------------------------------------
    # 7. Detail restore
    # --------------------------------------------------
    if detail_restore_strength is not None and detail_restore_strength > 0:
        detail = vi_y - y_denoised
        vi_y_processed = (
            vi_y_processed
            + float(detail_restore_strength) * vi_edge * detail
        )

    vi_y_processed = torch.clamp(vi_y_processed, 0.0, 1.0)

    return vi_y_processed


def dark_region_chroma_process(
    cr,
    cb,
    vi_y,
    dark_threshold=DARK_THRESHOLD,
    mask_power=DARK_MASK_POWER,
    chroma_mode=CHROMA_MODE,
    chroma_strength=CHROMA_STRENGTH,
    chroma_blur_kernel=CHROMA_BLUR_KERNEL,
    chroma_edge_power=CHROMA_EDGE_POWER,
):
    if chroma_mode == "none" or chroma_strength <= 0:
        return cr, cb

    if cr.ndim != 4 or cb.ndim != 4 or cr.shape[1] != 1 or cb.shape[1] != 1:
        raise ValueError(
            f"Expected cr/cb shape [B, 1, H, W], but got cr={tuple(cr.shape)}, cb={tuple(cb.shape)}"
        )

    if vi_y.shape[-2:] != cr.shape[-2:] or vi_y.shape[-2:] != cb.shape[-2:]:
        raise ValueError(
            f"Spatial size mismatch: vi_y={tuple(vi_y.shape)}, cr={tuple(cr.shape)}, cb={tuple(cb.shape)}"
        )

    dark_mask = build_dark_mask(
        vi_y=vi_y,
        dark_threshold=dark_threshold,
        mask_power=mask_power,
    )

    vi_edge = sobel_edge_map(vi_y)
    edge_protection = (1.0 - vi_edge).pow(chroma_edge_power)

    chroma_mask = dark_mask * edge_protection * float(chroma_strength)
    chroma_mask = torch.clamp(chroma_mask, 0.0, 1.0)

    if chroma_mode == "dark_smooth":
        cr_smooth = box_filter_2d(cr, kernel_size=chroma_blur_kernel)
        cb_smooth = box_filter_2d(cb, kernel_size=chroma_blur_kernel)

        cr_processed = (1.0 - chroma_mask) * cr + chroma_mask * cr_smooth
        cb_processed = (1.0 - chroma_mask) * cb + chroma_mask * cb_smooth

    elif chroma_mode == "dark_neutral":
        neutral_cr = torch.full_like(cr, 0.5)
        neutral_cb = torch.full_like(cb, 0.5)

        cr_processed = (1.0 - chroma_mask) * cr + chroma_mask * neutral_cr
        cb_processed = (1.0 - chroma_mask) * cb + chroma_mask * neutral_cb

    else:
        raise ValueError(f"Unknown chroma_mode: {chroma_mode}")

    cr_processed = torch.clamp(cr_processed, 0.0, 1.0)
    cb_processed = torch.clamp(cb_processed, 0.0, 1.0)

    return cr_processed, cb_processed


# ==================================================
# Metric functions
# ==================================================
def entropy_y_tensor(y_tensor):
    y_np = (
        torch.clamp(y_tensor.detach(), 0.0, 1.0)
        .cpu()
        .numpy()
        .squeeze()
    )

    y_uint8 = np.clip(y_np * 255.0, 0, 255).round().astype(np.uint8)

    histogram, _ = np.histogram(y_uint8, bins=256, range=(0, 255))
    histogram = histogram.astype(np.float64)
    histogram = histogram / (np.sum(histogram) + 1e-12)

    entropy = -np.sum(histogram * np.log2(histogram + 1e-12))
    return float(entropy)


def mutual_information_2d_from_tensors(x, y, bins=MI_BINS):
    x_np = (
        torch.clamp(x.detach(), 0.0, 1.0)
        .cpu()
        .numpy()
        .squeeze()
        .astype(np.float32)
        .ravel()
    )

    y_np = (
        torch.clamp(y.detach(), 0.0, 1.0)
        .cpu()
        .numpy()
        .squeeze()
        .astype(np.float32)
        .ravel()
    )

    hist_2d, _, _ = np.histogram2d(
        x_np,
        y_np,
        bins=bins,
        range=[[0.0, 1.0], [0.0, 1.0]],
    )

    pxy = hist_2d.astype(np.float64)
    pxy = pxy / (np.sum(pxy) + 1e-12)

    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    px_py = px @ py

    nz = pxy > 0
    mi = np.sum(pxy[nz] * np.log2((pxy[nz] + 1e-12) / (px_py[nz] + 1e-12)))

    return float(mi)


def compute_info_gain_en(original_y, fused_y):
    original_en = entropy_y_tensor(original_y)
    fused_en = entropy_y_tensor(fused_y)
    return fused_en - original_en


def compute_vis_structure_ssim(original_y, fused_y):
    original_np = (
        torch.clamp(original_y.detach(), 0.0, 1.0)
        .cpu()
        .numpy()
        .squeeze()
        .astype(np.float32)
    )

    fused_np = (
        torch.clamp(fused_y.detach(), 0.0, 1.0)
        .cpu()
        .numpy()
        .squeeze()
        .astype(np.float32)
    )

    return float(ssim(original_np, fused_np, data_range=1.0))


def compute_thermal_transfer_mi(ir_y, fused_y):
    return mutual_information_2d_from_tensors(ir_y, fused_y, bins=MI_BINS)


def compute_dark_flat_noise(
    fused_y,
    original_y,
    dark_threshold=EVAL_DARK_THRESHOLD,
    edge_threshold=EVAL_EDGE_THRESHOLD,
    high_pass_kernel=EVAL_HIGH_PASS_KERNEL,
):
    original_y = torch.clamp(original_y, 0.0, 1.0)
    fused_y = torch.clamp(fused_y, 0.0, 1.0)

    edge_map = sobel_edge_map(original_y)
    dark_mask = (original_y < dark_threshold).float()
    flat_mask = (edge_map < edge_threshold).float()
    eval_mask = dark_mask * flat_mask

    smooth = box_filter_2d(fused_y, kernel_size=high_pass_kernel)
    residual = fused_y - smooth

    valid = eval_mask > 0.5
    num_valid = int(valid.sum().item())

    if num_valid < 16:
        return np.nan

    noise_score = float(residual[valid].std().item())
    return noise_score


def compute_core_metrics(original_y, ir_y, fused_y):
    return {
        "INFO_GAIN_EN": compute_info_gain_en(original_y, fused_y),
        "VIS_STRUCTURE_SSIM": compute_vis_structure_ssim(original_y, fused_y),
        "THERMAL_TRANSFER_MI": compute_thermal_transfer_mi(ir_y, fused_y),
        "DARK_FLAT_NOISE": compute_dark_flat_noise(fused_y, original_y),
    }


def save_final_metrics_txt(
    out_path,
    total_images,
    processed_images,
    skipped_images,
    failed_images,
    avg_baseline_metrics,
    avg_trial_metrics,
    avg_latency_sec,
    avg_latency_ms,
):
    metrics_path = os.path.join(out_path, "final_metrics.txt")

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("[Final fusion quality metrics]\n")
        f.write("Metric directions:\n")
        f.write("INFO_GAIN_EN: higher is better\n")
        f.write("VIS_STRUCTURE_SSIM: higher is better\n")
        f.write("THERMAL_TRANSFER_MI: higher is better\n")
        f.write("DARK_FLAT_NOISE: lower is better\n")
        f.write("\n")
        f.write(f"Average Baseline INFO_GAIN_EN: {format_float(avg_baseline_metrics['INFO_GAIN_EN'])}\n")
        f.write(f"Average Trial INFO_GAIN_EN: {format_float(avg_trial_metrics['INFO_GAIN_EN'])}\n")
        f.write(f"Average Baseline VIS_STRUCTURE_SSIM: {format_float(avg_baseline_metrics['VIS_STRUCTURE_SSIM'])}\n")
        f.write(f"Average Trial VIS_STRUCTURE_SSIM: {format_float(avg_trial_metrics['VIS_STRUCTURE_SSIM'])}\n")
        f.write(f"Average Baseline THERMAL_TRANSFER_MI: {format_float(avg_baseline_metrics['THERMAL_TRANSFER_MI'])}\n")
        f.write(f"Average Trial THERMAL_TRANSFER_MI: {format_float(avg_trial_metrics['THERMAL_TRANSFER_MI'])}\n")
        f.write(f"Average Baseline DARK_FLAT_NOISE: {format_float(avg_baseline_metrics['DARK_FLAT_NOISE'])}\n")
        f.write(f"Average Trial DARK_FLAT_NOISE: {format_float(avg_trial_metrics['DARK_FLAT_NOISE'])}\n")
        f.write("\n")
        f.write("[Latency]\n")
        f.write("Latency scope: RGB2YCrCb -> denoising/chroma processing -> LEFuse -> normalization -> RGB restore -> final RGB array\n")
        f.write("Excluded: disk image loading, disk image saving, baseline metric computation\n")
        f.write(f"Average image latency: {avg_latency_sec:.6f} sec/image\n")
        f.write(f"Average image latency: {avg_latency_ms:.3f} ms/image\n")
        f.write("\n")
        f.write("[Counts]\n")
        f.write(f"Total visible images: {total_images}\n")
        f.write(f"Processed images: {processed_images}\n")
        f.write(f"Skipped images: {skipped_images}\n")
        f.write(f"Failed images: {failed_images}\n")

    return metrics_path


def run_warmup_once(
    fuser,
    device,
    vi_img_path,
    ir_img_path,
):
    """
    CUDA / cuDNN / PyTorch kernel warm-up용.
    결과 저장 X, metric 계산 X, latency 기록 X.
    """
    print("----------------------------------------")
    print("Running one warm-up pass before latency measurement...")
    print(f"Warm-up visible: {vi_img_path}")
    print(f"Warm-up infrared: {ir_img_path}")
    print("----------------------------------------")

    with torch.no_grad():
        vi = image_read_cv2(vi_img_path, mode="RGB")[np.newaxis, ...] / 255.0
        vi = np.transpose(vi, (0, 3, 1, 2))
        vi = torch.FloatTensor(vi).to(device)

        ir = image_read_cv2(ir_img_path, mode="RGB")[np.newaxis, ...] / 255.0
        ir = np.transpose(ir, (0, 3, 1, 2))
        ir = torch.FloatTensor(ir).to(device)

        sync_cuda_if_needed(device)

        vi_y, cr, cb = RGB2YCrCb(vi)
        ir_y, _, _ = RGB2YCrCb(ir)

        vi_y_processed = dark_region_edge_aware_nlm_denoise_y(
            vi_y=vi_y,
            ir_y=ir_y,
            h=NLM_H,
            template_window_size=NLM_TEMPLATE_WINDOW,
            search_window_size=NLM_SEARCH_WINDOW,
            dark_threshold=DARK_THRESHOLD,
            mask_power=DARK_MASK_POWER,
            edge_power=EDGE_POWER,
            detail_restore_strength=DETAIL_RESTORE_STRENGTH,
            use_ir_edge_for_mask=USE_IR_EDGE_FOR_MASK,
        )

        cr_processed, cb_processed = dark_region_chroma_process(
            cr=cr,
            cb=cb,
            vi_y=vi_y,
            dark_threshold=DARK_THRESHOLD,
            mask_power=DARK_MASK_POWER,
            chroma_mode=CHROMA_MODE,
            chroma_strength=CHROMA_STRENGTH,
            chroma_blur_kernel=CHROMA_BLUR_KERNEL,
            chroma_edge_power=CHROMA_EDGE_POWER,
        )

        out_y = fuser(vi_y_processed, ir_y)
        out_y = normalize_fused_y(out_y, normalize_mode=NORMALIZE_MODE)

        out_rgb = YCbCr2RGB(out_y, cb_processed, cr_processed)
        out_rgb = torch.clamp(out_rgb, 0.0, 1.0)

        # CPU 변환까지 한 번 수행해서 실제 inference path를 warm-up
        _ = (
            (out_rgb * 255.0)
            .cpu()
            .numpy()
            .squeeze(0)
            .astype("uint8")
        )

        sync_cuda_if_needed(device)

    print("Warm-up finished.")
    print("----------------------------------------")


def find_first_valid_pair(vi_path, ir_path):
    img_names = sorted(os.listdir(vi_path))

    for img_name in img_names:
        base_name, ext = os.path.splitext(img_name)

        if ext.lower() not in VALID_EXTS:
            continue

        vi_img_path = os.path.join(vi_path, img_name)
        ir_img_path = find_image(ir_path, base_name)

        if ir_img_path is not None:
            return vi_img_path, ir_img_path

    return None, None


def infer_fixed(
    ckpt,
    vi_path,
    ir_path,
    out_path,
):
    os.makedirs(out_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # 1. Load LEFuse
    # --------------------------------------------------
    fuser = LEFuse().to(device)

    checkpoint = torch.load(ckpt, map_location=device)
    fuser.load_state_dict(checkpoint["model"])
    fuser.eval()

    print("----------------------------------------")
    print("Loaded LEFuse model")
    print(f"Checkpoint: {ckpt}")
    print(f"Device: {device}")
    print("----------------------------------------")

    # --------------------------------------------------
    # Warm-up before latency measurement
    # --------------------------------------------------
    warmup_vi_path, warmup_ir_path = find_first_valid_pair(vi_path, ir_path)

    if warmup_vi_path is not None and warmup_ir_path is not None:
        run_warmup_once(
            fuser=fuser,
            device=device,
            vi_img_path=warmup_vi_path,
            ir_img_path=warmup_ir_path,
        )
    else:
        print("Warning: No valid pair found for warm-up.")

    total_images = 0
    processed_images = 0
    skipped_images = 0
    failed_images = 0

    latency_list = []

    baseline_metric_lists = {
        "INFO_GAIN_EN": [],
        "VIS_STRUCTURE_SSIM": [],
        "THERMAL_TRANSFER_MI": [],
        "DARK_FLAT_NOISE": [],
    }

    trial_metric_lists = {
        "INFO_GAIN_EN": [],
        "VIS_STRUCTURE_SSIM": [],
        "THERMAL_TRANSFER_MI": [],
        "DARK_FLAT_NOISE": [],
    }

    img_names = sorted(os.listdir(vi_path))

    with torch.no_grad():
        for img_name in img_names:
            base_name, ext = os.path.splitext(img_name)

            if ext.lower() not in VALID_EXTS:
                continue

            total_images += 1

            vi_img_path = os.path.join(vi_path, img_name)
            ir_img_path = find_image(ir_path, base_name)

            if ir_img_path is None:
                skipped_images += 1
                print(f"[SKIP] Infrared image not found for {base_name}")
                continue

            save_path = os.path.join(out_path, f"{base_name}.png")

            try:
                # --------------------------------------------------
                # Read images
                # Disk loading is excluded from latency.
                # --------------------------------------------------
                vi = image_read_cv2(vi_img_path, mode="RGB")[np.newaxis, ...] / 255.0
                vi = np.transpose(vi, (0, 3, 1, 2))
                vi = torch.FloatTensor(vi).to(device)

                ir = image_read_cv2(ir_img_path, mode="RGB")[np.newaxis, ...] / 255.0
                ir = np.transpose(ir, (0, 3, 1, 2))
                ir = torch.FloatTensor(ir).to(device)

                # --------------------------------------------------
                # Trial inference latency starts here.
                # Includes:
                # RGB2YCrCb, denoising, chroma smoothing,
                # LEFuse forward, normalization, RGB restore,
                # final RGB array creation.
                # Excludes:
                # disk loading, disk saving, baseline metric computation.
                # --------------------------------------------------
                sync_cuda_if_needed(device)
                start_time = time.perf_counter()

                # 1. RGB -> YCrCb
                vi_y, cr, cb = RGB2YCrCb(vi)
                ir_y, _, _ = RGB2YCrCb(ir)

                # 2. Y-channel denoising
                vi_y_processed = dark_region_edge_aware_nlm_denoise_y(
                    vi_y=vi_y,
                    ir_y=ir_y,
                    h=NLM_H,
                    template_window_size=NLM_TEMPLATE_WINDOW,
                    search_window_size=NLM_SEARCH_WINDOW,
                    dark_threshold=DARK_THRESHOLD,
                    mask_power=DARK_MASK_POWER,
                    edge_power=EDGE_POWER,
                    detail_restore_strength=DETAIL_RESTORE_STRENGTH,
                    use_ir_edge_for_mask=USE_IR_EDGE_FOR_MASK,
                )

                # 3. Chroma smoothing
                cr_processed, cb_processed = dark_region_chroma_process(
                    cr=cr,
                    cb=cb,
                    vi_y=vi_y,
                    dark_threshold=DARK_THRESHOLD,
                    mask_power=DARK_MASK_POWER,
                    chroma_mode=CHROMA_MODE,
                    chroma_strength=CHROMA_STRENGTH,
                    chroma_blur_kernel=CHROMA_BLUR_KERNEL,
                    chroma_edge_power=CHROMA_EDGE_POWER,
                )

                # 4. LEFuse fusion
                trial_fused_y = fuser(vi_y_processed, ir_y)

                # 5. Robust normalization
                trial_fused_y = normalize_fused_y(
                    trial_fused_y,
                    normalize_mode=NORMALIZE_MODE,
                )

                # 6. Restore final RGB
                # 기존 LEFuse 코드의 호출 순서 유지:
                # YCbCr2RGB(y, cb, cr)
                out_rgb = YCbCr2RGB(trial_fused_y, cb_processed, cr_processed)

                out_rgb = torch.clamp(out_rgb, 0.0, 1.0)
                out_rgb_uint8 = (
                    (out_rgb * 255.0)
                    .cpu()
                    .numpy()
                    .squeeze(0)
                    .astype("uint8")
                )
                out_rgb_uint8 = np.transpose(out_rgb_uint8, (1, 2, 0))

                sync_cuda_if_needed(device)
                end_time = time.perf_counter()

                latency = end_time - start_time
                latency_list.append(latency)

                # --------------------------------------------------
                # Save fused image
                # Disk saving is excluded from latency.
                # --------------------------------------------------
                imsave(save_path, out_rgb_uint8)

                # --------------------------------------------------
                # Baseline no-denoise fusion for metric only
                # Not included in latency.
                # --------------------------------------------------
                baseline_fused_y = fuser(vi_y, ir_y)
                baseline_fused_y = normalize_fused_y(
                    baseline_fused_y,
                    normalize_mode=BASELINE_NORMALIZE_MODE,
                )

                baseline_metrics = compute_core_metrics(
                    original_y=vi_y,
                    ir_y=ir_y,
                    fused_y=baseline_fused_y,
                )

                trial_metrics = compute_core_metrics(
                    original_y=vi_y,
                    ir_y=ir_y,
                    fused_y=trial_fused_y,
                )

                for key in baseline_metric_lists.keys():
                    baseline_metric_lists[key].append(baseline_metrics[key])
                    trial_metric_lists[key].append(trial_metrics[key])

                processed_images += 1
                print(f"[OK] {img_name} -> {save_path} | latency: {latency * 1000:.3f} ms")

            except Exception as e:
                failed_images += 1
                print(f"[FAILED] {img_name}: {repr(e)}")

    avg_baseline_metrics = {
        key: safe_nanmean(values)
        for key, values in baseline_metric_lists.items()
    }

    avg_trial_metrics = {
        key: safe_nanmean(values)
        for key, values in trial_metric_lists.items()
    }

    avg_latency_sec = safe_nanmean(latency_list)
    if avg_latency_sec is None:
        avg_latency_sec = 0.0

    avg_latency_ms = avg_latency_sec * 1000.0

    metrics_path = save_final_metrics_txt(
        out_path=out_path,
        total_images=total_images,
        processed_images=processed_images,
        skipped_images=skipped_images,
        failed_images=failed_images,
        avg_baseline_metrics=avg_baseline_metrics,
        avg_trial_metrics=avg_trial_metrics,
        avg_latency_sec=avg_latency_sec,
        avg_latency_ms=avg_latency_ms,
    )

    print("----------------------------------------")
    print("Fixed inference finished")
    print(f"Total visible images: {total_images}")
    print(f"Processed images: {processed_images}")
    print(f"Skipped images: {skipped_images}")
    print(f"Failed images: {failed_images}")
    print(f"Output path: {out_path}")
    print(f"Final metrics txt: {metrics_path}")
    print(f"Average image latency: {avg_latency_sec:.6f} sec/image")
    print(f"Average image latency: {avg_latency_ms:.3f} ms/image")
    print("----------------------------------------")


if __name__ == "__main__":
    parser = TrainOptions_E()
    opts = parser.parse()

    infer_fixed(
        ckpt=opts.ckpt_path,
        vi_path=opts.vi_path,
        ir_path=opts.ir_path,
        out_path=opts.out_path,
    )